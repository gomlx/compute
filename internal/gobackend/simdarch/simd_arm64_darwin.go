// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && arm64 && goexperiment.simd

package simdarch

import (
	"fmt"
	"strings"
	"sync/atomic"
	"syscall"
)

// smeClusterLock is padded to 128 bytes (Apple Silicon cache line size) to prevent false sharing.
type smeClusterLock struct {
	active atomic.Int32
	_      [124]byte
}

var (
	globalSMEClusterLocks [MaxSMEUnits]smeClusterLock
	smeActiveTotal        atomic.Int32
)

func sysctlEnabled(name string) bool {
	val, err := syscall.SysctlUint32(name)
	return err == nil && val != 0
}

func sysctlInt(name string) int {
	val, err := syscall.SysctlUint32(name)
	if err != nil {
		return 0
	}
	return int(val)
}

func detectARM64Features() arm64Features {
	features := arm64Features{
		hasASIMD:   true, // AdvSIMD (NEON) is mandatory on all Apple Silicon / ARMv8-A Darwin CPUs.
		hasFP16:    sysctlEnabled("hw.optional.arm.FEAT_FP16") || sysctlEnabled("hw.optional.neon_fp16"),
		hasBF16:    sysctlEnabled("hw.optional.arm.FEAT_BF16"),
		hasDotProd: sysctlEnabled("hw.optional.arm.FEAT_DotProd"),
		hasI8MM:    sysctlEnabled("hw.optional.arm.FEAT_I8MM"),
		hasSVE:     sysctlEnabled("hw.optional.arm.FEAT_SVE"),
		hasSME:     sysctlEnabled("hw.optional.arm.FEAT_SME") || sysctlEnabled("hw.optional.arm.FEAT_SME2"),
	}
	if features.hasSME {
		detectDarwinSMETopology()
	}
	return features
}

// detectDarwinSMETopology queries macOS sysctl hw.perflevel* to determine how many
// distinct L2 caches (and therefore L2-attached SME coprocessor units) exist on the SoC.
//
// On Apple Silicon (M4/M5/M6, including Pro/Max/Ultra):
//   - Each L2 cache slice has 1 attached SME unit.
//   - Within a perflevel i, the number of L2 caches is max(1, physicalcpu / cpusperl2).
//   - When two perflevels share the same L2 cache (e.g., on Apple M6 where perflevel0 "Super"
//     has sharesl2=2 and perflevel1 "Performance" has sharesl2=1, sharing a single 20 MB L2 cache),
//     hw.perflevel<i>.sharesl2 has the bit (1 << j) set for the shared perflevel j < i, so it is
//     not double-counted.
func detectDarwinSMETopology() {
	nLevels := sysctlInt("hw.nperflevels")
	if nLevels <= 0 {
		NumHighPerfSMEUnits = 1
		NumEfficiencySMEUnits = 0
		NumSMEUnits = 1
		return
	}

	highPerfUnits := 0
	effUnits := 0
	for i := range nLevels {
		prefix := fmt.Sprintf("hw.perflevel%d.", i)
		cores := sysctlInt(prefix + "physicalcpu")
		perL2 := sysctlInt(prefix + "cpusperl2")
		sharesL2 := sysctlInt(prefix + "sharesl2")
		name, _ := syscall.Sysctl(prefix + "name")

		if cores <= 0 {
			continue
		}
		// Check if this perflevel shares its L2 cache with an already-counted earlier perflevel j < i.
		sharedWithEarlier := false
		if sharesL2 != 0 {
			for j := range i {
				if (sharesL2 & (1 << j)) != 0 {
					sharedWithEarlier = true
					break
				}
			}
		}
		if sharedWithEarlier {
			continue
		}

		units := 1
		if perL2 > 0 && cores/perL2 > 1 {
			units = cores / perL2
		}
		if strings.EqualFold(strings.TrimSpace(name), "Efficiency") || (i == nLevels-1 && nLevels > 1 && highPerfUnits > 0 && name == "") {
			effUnits += units
		} else {
			highPerfUnits += units
		}
	}

	if highPerfUnits == 0 && effUnits == 0 {
		highPerfUnits = 1
	}
	NumHighPerfSMEUnits = highPerfUnits
	NumEfficiencySMEUnits = effUnits
	NumSMEUnits = min(MaxSMEUnits, highPerfUnits+effUnits)
}

// ReadTPIDR_EL0 reads the ARM64 TPIDR_EL0 register in 1 CPU cycle.
// On Darwin/ARM64, bits [11:0] hold the physical CPU ID and bits [19:12] hold the L2 cluster ID.
//
//go:noescape
func ReadTPIDR_EL0() uint64

// CurrentSMECluster returns the current OS thread's hardware L2 cache cluster ID (which is also its SME unit ID),
// physical CPU ID, and whether the core belongs to an Efficiency cluster.
//
// On Darwin/ARM64 XNU, clusterID = (TPIDR_EL0 >> 12) & 0xff is 0 for the Efficiency L2 cluster(s)
// and >= NumEfficiencySMEUnits (e.g. 1 on M4/M5/M6) for the Performance / Super L2 cluster(s).
func CurrentSMECluster() (clusterID int, cpuID int, isEfficiencyCore bool) {
	tpidr := ReadTPIDR_EL0()
	clusterID = int((tpidr >> 12) & 0xff)
	cpuID = int(tpidr & 0xfff)
	isEfficiencyCore = NumEfficiencySMEUnits > 0 && clusterID < NumEfficiencySMEUnits
	return clusterID, cpuID, isEfficiencyCore
}

// AcquireSME reads the current core's L2 cluster ID via TPIDR_EL0 and attempts to acquire
// that L2 cluster's global SME unit lock using the default SMEWorkersPerCluster limit.
//
// If acquired, returns (clusterID, true, isEfficiencyCore) and the caller MUST call ReleaseSME(clusterID).
// If the cluster's SME unit is already occupied, returns (-1, false, isEfficiencyCore).
func AcquireSME() (clusterID int, useSME bool, isEfficiencyCore bool) {
	return AcquireSMEWithLimit(SMEWorkersPerCluster, SMEWorkersPerCluster)
}

// AcquireSMEWithLimit is like AcquireSME, but allows specifying separate concurrency limits
// for high-performance L2 clusters (Super/Performance) and efficiency L2 clusters.
func AcquireSMEWithLimit(maxFastPerCluster, maxSlowPerCluster int32) (clusterID int, useSME bool, isEfficiencyCore bool) {
	if !SMEUseHybridNEON {
		return -1, true, false
	}
	clusterID, _, isEfficiencyCore = CurrentSMECluster()
	if clusterID < 0 || clusterID >= MaxSMEUnits {
		clusterID = int(uint(clusterID) % MaxSMEUnits)
	}

	limit := maxFastPerCluster
	if isEfficiencyCore {
		limit = maxSlowPerCluster
	}
	if limit <= 0 {
		return -1, false, isEfficiencyCore
	}
	if globalSMEClusterLocks[clusterID].active.Load() >= limit {
		return -1, false, isEfficiencyCore
	}
	if globalSMEClusterLocks[clusterID].active.Add(1) <= limit {
		smeActiveTotal.Add(1)
		return clusterID, true, isEfficiencyCore
	}
	globalSMEClusterLocks[clusterID].active.Add(-1)
	return -1, false, isEfficiencyCore
}

// AcquireSMEWithSpin is like AcquireSMEWithLimit, but spins for up to maxSpins iterations
// if the current L2 cluster's SME unit is momentarily busy (e.g., with another short op).
func AcquireSMEWithSpin(maxFastPerCluster, maxSlowPerCluster int32, maxSpins int) (clusterID int, useSME bool, isEfficiencyCore bool) {
	clusterID, useSME, isEfficiencyCore = AcquireSMEWithLimit(maxFastPerCluster, maxSlowPerCluster)
	if useSME || maxSpins <= 0 {
		return clusterID, useSME, isEfficiencyCore
	}
	for range maxSpins {
		clusterID, useSME, isEfficiencyCore = AcquireSMEWithLimit(maxFastPerCluster, maxSlowPerCluster)
		if useSME {
			return clusterID, true, isEfficiencyCore
		}
	}
	return -1, false, isEfficiencyCore
}

// IsFastSMEIdle reports whether at least one high-performance L2 cluster SME unit
// (clusterID >= NumEfficiencySMEUnits) currently has zero active SME workers.
func IsFastSMEIdle() bool {
	if NumHighPerfSMEUnits <= 0 {
		return false
	}
	start := max(0, NumEfficiencySMEUnits)
	end := min(MaxSMEUnits, start+NumHighPerfSMEUnits)
	for c := start; c < end; c++ {
		if globalSMEClusterLocks[c].active.Load() == 0 {
			return true
		}
	}
	return false
}

// ReleaseSME releases a previously acquired global SME cluster lock.
func ReleaseSME(clusterID int) {
	if clusterID >= 0 && clusterID < MaxSMEUnits {
		smeActiveTotal.Add(-1)
		globalSMEClusterLocks[clusterID].active.Add(-1)
	}
}

// ActiveSMEWorkers returns the total number of workers currently holding an SME cluster lock.
func ActiveSMEWorkers() int32 {
	return smeActiveTotal.Load()
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simdarch

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSIMDArchFlags(t *testing.T) {
	t.Logf("GOARCH=%s GOOS=%s: ActiveArchitectures=%v IsAVX2Allowed=%v IsAVX512Allowed=%v IsNEONAllowed=%v IsARM64FP16Allowed=%v IsARM64BF16Allowed=%v IsARM64DotProdAllowed=%v IsARM64I8MMAllowed=%v IsSVEAllowed=%v IsSMEAllowed=%v",
		runtime.GOARCH, runtime.GOOS, ActiveArchitectures,
		IsAVX2Allowed, IsAVX512Allowed,
		IsNEONAllowed, IsARM64FP16Allowed, IsARM64BF16Allowed,
		IsARM64DotProdAllowed, IsARM64I8MMAllowed,
		IsSVEAllowed, IsSMEAllowed)
	t.Logf("UseAsm flags: AVX2UseAsm=%v AVX512UseAsm=%v NEONUseAsm=%v",
		AVX2UseAsm, AVX512UseAsm, NEONUseAsm)

	expectedOSArch := runtime.GOOS + "/" + runtime.GOARCH
	if len(ActiveArchitectures) == 0 || ActiveArchitectures[0] != expectedOSArch {
		t.Errorf("expected ActiveArchitectures[0] to be %q, got %v", expectedOSArch, ActiveArchitectures)
	}

	switch runtime.GOARCH {
	case "arm64":
		if IsAVX2Allowed || IsAVX512Allowed {
			t.Errorf("expected IsAVX2Allowed and IsAVX512Allowed to be false on arm64")
		}
	case "amd64":
		if IsNEONAllowed || IsARM64FP16Allowed || IsARM64BF16Allowed || IsARM64DotProdAllowed || IsARM64I8MMAllowed || IsSVEAllowed || IsSMEAllowed {
			t.Errorf("expected ARM64 flags to be false on amd64")
		}
	}
}

func TestDarwinSMETopologyAndLocking(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" || !IsSMEAllowed {
		t.Skip("SME on darwin/arm64 is not supported or allowed on this host")
	}

	t.Logf("Detected SME Topology: NumSMEUnits=%d (NumHighPerfSMEUnits=%d, NumEfficiencySMEUnits=%d)",
		NumSMEUnits, NumHighPerfSMEUnits, NumEfficiencySMEUnits)

	if NumSMEUnits < 1 || NumSMEUnits > MaxSMEUnits {
		t.Fatalf("expected NumSMEUnits in [1, %d], got %d", MaxSMEUnits, NumSMEUnits)
	}
	if NumHighPerfSMEUnits < 1 {
		t.Fatalf("expected NumHighPerfSMEUnits >= 1, got %d", NumHighPerfSMEUnits)
	}
	if NumSMEUnits != NumHighPerfSMEUnits+NumEfficiencySMEUnits {
		t.Fatalf("NumSMEUnits (%d) != NumHighPerfSMEUnits (%d) + NumEfficiencySMEUnits (%d)",
			NumSMEUnits, NumHighPerfSMEUnits, NumEfficiencySMEUnits)
	}

	// Spawn GOMAXPROCS goroutines simultaneously to verify TPIDR_EL0 L2 cluster IDs and SME locking.
	numProcs := runtime.GOMAXPROCS(0)
	prevProcs := runtime.GOMAXPROCS(numProcs + 2)
	defer runtime.GOMAXPROCS(prevProcs)

	var stopSpin atomic.Bool
	var wg sync.WaitGroup
	wg.Add(numProcs)

	var seenCluster [MaxSMEUnits]atomic.Int32
	var seenCPU [64]atomic.Int32
	var maxConcurrentAcquired atomic.Int32

	for range numProcs {
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			defer wg.Done()

			for !stopSpin.Load() {
				cl, cpu, _ := CurrentSMECluster()
				if cl >= 0 && cl < MaxSMEUnits {
					seenCluster[cl].Add(1)
				}
				if cpu >= 0 && cpu < 64 {
					seenCPU[cpu].Add(1)
				}

				smeID, ok, _ := AcquireSME()
				if ok {
					cur := ActiveSMEWorkers()
					for {
						prev := maxConcurrentAcquired.Load()
						if cur <= prev || maxConcurrentAcquired.CompareAndSwap(prev, cur) {
							break
						}
					}
					// Hold briefly so concurrent workers contend on the cluster locks
					for range 500 {
						_ = ReadTPIDR_EL0()
					}
					ReleaseSME(smeID)
				}
			}
		}()
	}

	time.Sleep(150 * time.Millisecond)
	stopSpin.Store(true)
	wg.Wait()

	if ActiveSMEWorkers() != 0 {
		t.Fatalf("expected ActiveSMEWorkers() == 0 after all workers finished, got %d", ActiveSMEWorkers())
	}

	distinctClusters := 0
	for cl := range MaxSMEUnits {
		if count := seenCluster[cl].Load(); count > 0 {
			distinctClusters++
			t.Logf("L2 Cluster %d observed %d times", cl, count)
		}
	}
	distinctCPUs := 0
	for cpu := range 64 {
		if seenCPU[cpu].Load() > 0 {
			distinctCPUs++
		}
	}
	t.Logf("Distinct L2 clusters observed=%d (expected <= %d), distinct CPUs observed=%d, maxConcurrentAcquired=%d",
		distinctClusters, NumSMEUnits, distinctCPUs, maxConcurrentAcquired.Load())

	if distinctClusters < 1 || distinctClusters > NumSMEUnits {
		t.Errorf("expected distinctClusters in [1, %d], got %d", NumSMEUnits, distinctClusters)
	}
	if int(maxConcurrentAcquired.Load()) > NumSMEUnits*int(SMEWorkersPerCluster) {
		t.Errorf("maxConcurrentAcquired (%d) exceeded NumSMEUnits*SMEWorkersPerCluster (%d)",
			maxConcurrentAcquired.Load(), NumSMEUnits*int(SMEWorkersPerCluster))
	}
}

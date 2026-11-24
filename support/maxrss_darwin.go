// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin

package support

import (
	"syscall"
	"unsafe"
)

const (
	// Darwin XNU syscall and proc_info constants (from <sys/syscall.h>, <sys/proc_info.h>, <sys/resource.h>).
	sysProcInfo          = 336 // SYS_proc_info
	procInfoCallPidUsage = 9   // PROC_INFO_CALL_PIDRUSAGE
	rusageInfoV4         = 4   // RUSAGE_INFO_V4
)

// rusageInfoV4Struct mirrors XNU's struct rusage_info_v4 (<sys/resource.h>).
type rusageInfoV4Struct struct {
	uuid                     [16]byte
	userTime                 uint64
	systemTime               uint64
	pkgIdleWkups             uint64
	interruptWkups           uint64
	pageins                  uint64
	wiredSize                uint64
	residentSize             uint64
	physFootprint            uint64
	procStartAbstime         uint64
	procExitAbstime          uint64
	childUserTime            uint64
	childSystemTime          uint64
	childPkgIdleWkups        uint64
	childInterruptWkups      uint64
	childPageins             uint64
	childElapsedAbstime      uint64
	diskioBytesread          uint64
	diskioByteswritten       uint64
	cpuTimeQosDefault        uint64
	cpuTimeQosMaintenance    uint64
	cpuTimeQosBackground     uint64
	cpuTimeQosUtility        uint64
	cpuTimeQosLegacy         uint64
	cpuTimeQosUserInitiated  uint64
	cpuTimeQosUserInteractive uint64
	billedSystemTime         uint64
	servicedSystemTime       uint64
	logicalWrites            uint64
	lifetimeMaxPhysFootprint uint64
	instructions             uint64
	cycles                   uint64
	billedEnergy             uint64
	servicedEnergy           uint64
	intervalMaxPhysFootprint uint64
	runnableTime             uint64
}

// maxRSSBytes returns the process's peak physical memory footprint (ri_lifetime_max_phys_footprint)
// on macOS via SYS_proc_info / PROC_INFO_CALL_PIDRUSAGE (RUSAGE_INFO_V4), falling back to
// getrusage(RUSAGE_SELF).Maxrss if the call fails.
//
// We prefer ri_lifetime_max_phys_footprint over getrusage.Maxrss on Darwin because Go's runtime
// returns unused heap pages via MADV_FREE_REUSABLE (golang/go#29844), which decreases
// phys_footprint immediately while keeping pages resident in ru_maxrss until system memory pressure.
func maxRSSBytes() int64 {
	var info rusageInfoV4Struct
	_, _, errno := syscall.Syscall6(
		sysProcInfo,
		procInfoCallPidUsage,
		uintptr(syscall.Getpid()),
		rusageInfoV4,
		0,
		uintptr(unsafe.Pointer(&info)),
		0,
	)
	if errno == 0 && info.lifetimeMaxPhysFootprint > 0 {
		return int64(info.lifetimeMaxPhysFootprint)
	}
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return int64(ru.Maxrss)
}

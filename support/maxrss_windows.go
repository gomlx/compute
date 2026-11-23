// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build windows

package support

import (
	"syscall"
	"unsafe"
)

type processMemoryCounters struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
}

var (
	psapiDLL                = syscall.NewLazyDLL("psapi.dll")
	procGetProcessMemInfo   = psapiDLL.NewProc("GetProcessMemoryInfo")
)

func maxRSSBytes() int64 {
	handle, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0
	}
	var pmc processMemoryCounters
	pmc.cb = uint32(unsafe.Sizeof(pmc))
	r1, _, _ := procGetProcessMemInfo.Call(
		uintptr(handle),
		uintptr(unsafe.Pointer(&pmc)),
		uintptr(pmc.cb),
	)
	if r1 == 0 {
		return 0
	}
	return int64(pmc.peakWorkingSetSize)
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin

package scheduler

import (
	"runtime"
	"syscall"
)

// detectPhysicalCores queries macOS sysctl hw.physicalcpu, falling back to runtime.NumCPU().
func detectPhysicalCores() int {
	if cores, err := syscall.SysctlUint32("hw.physicalcpu"); err == nil && cores > 0 {
		return int(cores)
	}
	return max(runtime.NumCPU(), 1)
}

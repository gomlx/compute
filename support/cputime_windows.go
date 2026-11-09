// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build windows

package support

import (
	"syscall"
	"time"
)

func cpuTimeNs() int64 {
	handle, err := syscall.GetCurrentProcess()
	if err != nil {
		return int64(time.Since(processStartTime))
	}
	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(handle, &creation, &exit, &kernel, &user); err != nil {
		return int64(time.Since(processStartTime))
	}
	return filetimeDurationNs(kernel) + filetimeDurationNs(user)
}

// filetimeDurationNs converts a Windows FILETIME representing a duration (in 100-ns ticks)
// into nanoseconds. Note: syscall.Filetime.Nanoseconds() cannot be used here because it
// treats the FILETIME as a wall-clock timestamp since 1601-01-01 and subtracts the Unix epoch.
func filetimeDurationNs(ft syscall.Filetime) int64 {
	ticks := (uint64(ft.HighDateTime) << 32) | uint64(ft.LowDateTime)
	return int64(ticks * 100)
}

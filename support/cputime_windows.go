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
	return kernel.Nanoseconds() + user.Nanoseconds()
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux && (amd64 || arm64 || riscv64 || ppc64le || s390x || loong64)

package support

import (
	"syscall"
	"time"
	"unsafe"
)

// clockProcessCPUTimeID is Linux's CLOCK_PROCESS_CPUTIME_ID (<linux/time.h>),
// which measures high-resolution user + system CPU time across all threads of the process.
const clockProcessCPUTimeID = 2

func cpuTimeNs() int64 {
	var ts syscall.Timespec
	_, _, errno := syscall.RawSyscall(
		syscall.SYS_CLOCK_GETTIME,
		clockProcessCPUTimeID,
		uintptr(unsafe.Pointer(&ts)),
		0,
	)
	if errno == 0 {
		return ts.Nano()
	}

	// Fallback to getrusage(RUSAGE_SELF) if clock_gettime fails.
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err == nil {
		return ru.Utime.Nano() + ru.Stime.Nano()
	}
	return int64(time.Since(processStartTime))
}

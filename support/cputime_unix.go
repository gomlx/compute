// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin || freebsd || openbsd || netbsd || dragonfly || (linux && !amd64 && !arm64 && !riscv64 && !ppc64le && !s390x && !loong64)

package support

import (
	"syscall"
	"time"
)

func cpuTimeNs() int64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return int64(time.Since(processStartTime))
	}
	return ru.Utime.Nano() + ru.Stime.Nano()
}

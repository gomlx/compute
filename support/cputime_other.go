// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !linux && !darwin && !freebsd && !openbsd && !netbsd && !dragonfly && !windows

package support

import "time"

func cpuTimeNs() int64 {
	return int64(time.Since(processStartTime))
}

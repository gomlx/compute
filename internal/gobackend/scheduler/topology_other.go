// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !linux && !darwin

package scheduler

import "runtime"

// detectPhysicalCores falls back to runtime.NumCPU() on platforms without sysfs/sysctl topology detection.
func detectPhysicalCores() int {
	return max(runtime.NumCPU(), 1)
}

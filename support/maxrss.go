// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package support

// MaxRSS returns the maximum resident set size (peak physical memory usage) of the current
// process in bytes:
//
//   - Linux, FreeBSD, OpenBSD, NetBSD, DragonFly: Queries getrusage(RUSAGE_SELF) and converts
//     ru_maxrss from KiB to bytes.
//   - macOS (Darwin): Queries getrusage(RUSAGE_SELF), where ru_maxrss is already reported in bytes.
//   - Windows: Queries GetProcessMemoryInfo and returns PeakWorkingSetSize in bytes.
//   - Other platforms (e.g., WebAssembly / WASI, Plan 9): Returns 0.
func MaxRSS() int64 {
	return maxRSSBytes()
}

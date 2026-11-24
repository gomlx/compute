// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package support

// MaxRSS returns the peak physical memory usage of the current process in bytes:
//
//   - Linux, FreeBSD, OpenBSD, NetBSD, DragonFly: Queries getrusage(RUSAGE_SELF) and converts
//     ru_maxrss from KiB to bytes. (On Linux since Go 1.16 — golang/go#42330 — the Go runtime
//     scavenges unused heap pages with MADV_DONTNEED, so ru_maxrss immediately reflects freed
//     memory.)
//   - macOS (Darwin): Queries proc_pid_rusage (RUSAGE_INFO_V4) for ri_lifetime_max_phys_footprint
//     (matching Activity Monitor and `/usr/bin/time -l`'s "peak memory footprint"), falling back
//     to getrusage(RUSAGE_SELF).Maxrss if unavailable.
//     Note: Go's Darwin runtime scavenges unused heap pages with MADV_FREE_REUSABLE (see
//     golang/go#29844), which subtracts freed pages from phys_footprint immediately while leaving
//     them resident in RAM until system memory pressure occurs. Consequently, getrusage's ru_maxrss
//     on macOS over-reports peak memory (especially when off-heap mmap arenas such as constant_arena
//     are used alongside scavenged Go-heap spans), whereas ri_lifetime_max_phys_footprint accurately
//     tracks peak physical memory footprint.
//   - Windows: Queries GetProcessMemoryInfo and returns PeakWorkingSetSize in bytes.
//   - Other platforms (e.g., WebAssembly / WASI, Plan 9): Returns 0.
func MaxRSS() int64 {
	return maxRSSBytes()
}

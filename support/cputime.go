// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package support

import "time"

// processStartTime is used as the reference point for wall-clock fallback implementations
// so that CPUTime returns a monotonic duration since process start.
var processStartTime = time.Now()

// CPUTime returns the cumulative CPU time (user + system) consumed by the current process
// across all threads, with nanosecond precision (as a time.Duration, where 1 unit == 1 nanosecond)
// and the highest resolution supported by the underlying OS/architecture:
//
//   - Linux (amd64, arm64, riscv64, ppc64le, s390x, loong64): Queries the kernel scheduler's
//     per-process high-resolution CPU clock via clock_gettime(CLOCK_PROCESS_CPUTIME_ID).
//     This accounts for both user and system (kernel) execution time across all threads of the
//     process (sum_exec_runtime) with nanosecond (1 ns) resolution. Falls back to getrusage(RUSAGE_SELF)
//     if the syscall fails.
//   - macOS (Darwin), FreeBSD, OpenBSD, NetBSD, DragonFly, and 32-bit Linux: Queries
//     getrusage(RUSAGE_SELF) and sums ru_utime + ru_stime across all threads of the process,
//     providing microsecond (1 µs = 1,000 ns) resolution.
//   - Windows: Queries GetProcessTimes(GetCurrentProcess(), ...) and sums KernelTime + UserTime
//     across all threads of the process, providing 100-nanosecond (100 ns) resolution.
//   - Other platforms (e.g., WebAssembly / WASI, Plan 9): Falls back to the monotonic wall-clock
//     elapsed time since the process started (time.Since(processStartTime)).
//
// Typical usage to measure CPU time of a section of code:
//
//	startCPU := support.CPUTime()
//	... do work ...
//	cpuElapsed := support.CPUTime() - startCPU
func CPUTime() time.Duration {
	return time.Duration(cpuTimeNs())
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
)

// smeLargeKernelFloat32Asm_16x16 implements a 16x16 GEMM microkernel using 1 ARM64 SME ZA tile (ZA0.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat32Asm_16x16(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat32Asm_16x64 implements a 16x64 GEMM microkernel using 4 ARM64 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat32Asm_16x64(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat32Asm_32x32 implements a 32x32 GEMM microkernel using 4 ARM64 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat32Asm_32x32(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat16Asm_16x16 implements a 16x16 GEMM microkernel for Float16 -> Float32 using 1 SME ZA tile (ZA0.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat16Asm_16x16(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat16Asm_32x32 implements a 32x32 GEMM microkernel for Float16 -> Float32 using 4 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat16Asm_32x32(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelBFloat16Asm_16x16 implements a 16x16 GEMM microkernel for BFloat16 -> Float32 using 1 SME ZA tile (ZA0.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelBFloat16Asm_16x16(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelBFloat16Asm_32x32 implements a 32x32 GEMM microkernel for BFloat16 -> Float32 using 4 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelBFloat16Asm_32x32(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat64Asm_8x8 implements an 8x8 GEMM microkernel for Float64 using 1 SME ZA tile (ZA0.D).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat64Asm_8x8(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat64Asm_16x16 implements a 16x16 GEMM microkernel for Float64 using 4 SME ZA tiles (ZA0.D-ZA3.D).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat64Asm_16x16(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// ===========================================================================
// Global Asymmetric SME Cluster Locks & Hybrid SME + NEON Co-Execution
// ===========================================================================
//
// Architectural Background (Apple M-Series / ARMv9 SME Topology):
// On ARMv9 processors with FEAT_SME (such as Apple M4/M5/M6), the SME matrix
// coprocessor is shared at the CPU cluster level rather than duplicated per core:
//   - Cluster 0 (Super cores):       shares 1 high-throughput SME unit (~2.0 TFlops/s)
//   - Cluster 1 (Performance cores): shares 1 high-throughput SME unit (~2.0 TFlops/s)
//   - Cluster 2 (Efficiency cores):  shares 1 smaller SME unit (~0.9 TFlops/s)
//
// When all worker goroutines enter streaming mode (`SMSTART`) simultaneously,
// multiple cores within the same cluster queue and time-slice on their cluster's
// single SME unit while their private per-core 128-bit NEON vector pipelines sit
// completely idle.
//
// Strategy:
//  1. Global Per-Cluster Locks (`globalSMEClusterLocks`):
//     Each of the 3 clusters has a cache-line-padded global atomic counter capped
//     at `SMEWorkersPerCluster` (default 1). Because the locks are package-global,
//     they coordinate SME unit ownership across intra-op workers, concurrent
//     inter-op matmuls, and concurrent inference calls (`Execute()`).
//  2. One-Time Start-of-MatMul Core Tier Probe (`smeProbeCoreClockTicks`):
//     At the start of a matmul worker, the goroutine pins itself to its OS thread
//     via `runtime.LockOSThread()` and runs a ~250-460ns in-register NEON `VFMLA`
//     probe timed with the hardware virtual counter `CNTVCT_EL0`.
//     Because `VFMLA` runs on the private per-core SIMD pipes (4 pipes at high clock
//     on Super/P-cores vs. 2 pipes at lower clock on E-cores), it never contends
//     across cores and cleanly identifies which cluster the OS thread is running on.
//  3. Noise & Thermal Throttling Resilience:
//     - If the first probe exceeds `smeSuperCutoffTicks` (e.g., due to an OS context
//       switch/interrupt mid-probe or cold core frequency ramp-up), the worker retries
//       the probe once (`min(t1, t2)`), since context switches can only inflate elapsed
//       wall-clock ticks, never reduce them.
//     - The measured ticks are mapped to the closest cluster center.
//     - If a fast core (Cluster 0 Super or Cluster 1 Performance) finds its primary
//       cluster lock already held (for instance, if thermal throttling shifted Super
//       core ticks into the Performance range), it automatically checks and claims the
//       adjacent fast cluster's lock if free, guaranteeing both fast-cluster SME units
//       are always utilized.
//  4. Zero-Repack Hybrid NEON Co-Execution (`neonLargeKernelFromSME*`):
//     Workers that do not acquire an SME lock execute `neonLargeKernelFromSME*` on
//     their private per-core NEON pipelines, reading directly from the exact same
//     `[K, 32]` (or `[K, 16]`) SME-packed LHS and RHS panels by advancing their
//     pointers by `kernelRows * sizeof(T)` and `kernelCols * sizeof(T)` bytes per K-step.

// smeProbeCoreClockTicks measures CNTVCT_EL0 ticks for 256x16 per-core NEON VFMLAs.
// Defined in sme_arm64.s.
//
//go:noescape
func smeProbeCoreClockTicks() uint64

//go:noescape
func neonSubTile4x16StridedFloat32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int,
	accumulate bool,
)

//go:noescape
func neonSubTile8x8StridedFloat16Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int,
	accumulate bool,
)

//go:noescape
func neonSubTile8x8StridedBFloat16Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int,
	accumulate bool,
)

//go:noescape
func neonSubTile8x4StridedFloat64Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int,
	accumulate bool,
)

const numSMEClusters = 3

// smeClusterLock is padded to 128 bytes (Apple Silicon cache line size) to prevent false sharing.
type smeClusterLock struct {
	active atomic.Int32
	_      [124]byte
}

var (
	// SMEUseHybridNEON enables global per-cluster SME locking with per-core NEON co-execution
	// on workers whose cluster SME unit is already saturated.
	SMEUseHybridNEON = true

	// SMEWorkersPerCluster is the maximum number of concurrent SME workers per physical CPU cluster.
	SMEWorkersPerCluster int32 = 1

	globalSMEClusterLocks [numSMEClusters]smeClusterLock
	smeActiveTotal        atomic.Int32

	smeCalibrateOnce    sync.Once
	smeSuperCutoffTicks uint64 = 275
	smePerfCutoffTicks  uint64 = 395
)

func init() {
	if gobackend.IsSMEAllowed {
		calibrateSMEClusters()
	}
}

// calibrateSMEClusters runs once to measure per-core NEON VFMLA ticks across all logical CPUs
// and calibrate the tick midpoints separating Super, Performance, and Efficiency clusters.
func calibrateSMEClusters() {
	smeCalibrateOnce.Do(func() {
		numCPU := runtime.NumCPU()
		if numCPU < 4 {
			return
		}
		var readyWg sync.WaitGroup
		readyWg.Add(numCPU)
		var goSignal atomic.Bool
		var doneWg sync.WaitGroup
		doneWg.Add(numCPU)

		samples := make([]uint64, numCPU)
		for i := range numCPU {
			go func(id int) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				for range 64 {
					_ = smeProbeCoreClockTicks()
				}
				readyWg.Done()
				for spins := 0; !goSignal.Load(); spins++ {
					if spins > 10000 {
						runtime.Gosched()
						spins = 0
					}
				}
				minTicks := uint64(999999)
				for range 5 {
					if t := smeProbeCoreClockTicks(); t > 0 && t < minTicks {
						minTicks = t
					}
				}
				samples[id] = minTicks
				doneWg.Done()
			}(i)
		}
		readyWg.Wait()
		goSignal.Store(true)
		doneWg.Wait()

		sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })

		// Identify distinct jumps (>12% gap) between sorted core tick measurements.
		var clusterCenters []uint64
		clusterSum := samples[0]
		clusterCount := uint64(1)
		prev := samples[0]
		for i := 1; i < len(samples); i++ {
			if samples[i] > prev+(prev/8) && len(clusterCenters) < 2 {
				clusterCenters = append(clusterCenters, clusterSum/clusterCount)
				clusterSum = samples[i]
				clusterCount = 1
			} else {
				clusterSum += samples[i]
				clusterCount++
			}
			prev = samples[i]
		}
		clusterCenters = append(clusterCenters, clusterSum/clusterCount)

		if len(clusterCenters) == 3 {
			superCut := (clusterCenters[0] + clusterCenters[1]) / 2
			perfCut := (clusterCenters[1] + clusterCenters[2]) / 2
			if superCut >= 240 && superCut <= 300 && perfCut >= 330 && perfCut <= 450 {
				smeSuperCutoffTicks = superCut
				smePerfCutoffTicks = perfCut
			}
		} else if len(clusterCenters) == 2 {
			// 2-cluster CPU (e.g. Performance + Efficiency without separate Super tier):
			// split fast cores across slots 0 and 1, and Efficiency cores on slot 2.
			perfCut := (clusterCenters[0] + clusterCenters[1]) / 2
			if perfCut >= 300 && perfCut <= 450 {
				smeSuperCutoffTicks = clusterCenters[0]
				smePerfCutoffTicks = perfCut
			}
		}
	})
}

// SMEAcquireCluster probes the current OS thread's CPU cluster and attempts to acquire
// that cluster's global SME unit lock.
// If acquired, returns (clusterIdx, true, false) and the caller MUST call SMEReleaseCluster(clusterIdx).
// If the cluster's SME unit is already busy, returns (-1, false, isSlowECore) where isSlowECore
// is true when the thread is on an Efficiency core (Cluster 2) whose SME unit is already occupied.
func SMEAcquireCluster() (clusterIdx int, useSME bool, isSlowECore bool) {
	return smeAcquireClusterWithLimit(SMEWorkersPerCluster, SMEWorkersPerCluster)
}

// SMEAcquireClusterWithLimit is like SMEAcquireCluster, but allows specifying a higher concurrency
// limit on fast clusters (Super and Performance) for two-stage pipelines (such as ConvGeneral)
// where each SME worker spends substantial time on its private CPU core packing LHS before
// entering SMSTART.
func SMEAcquireClusterWithLimit(maxFastPerCluster, maxSlowPerCluster int32) (clusterIdx int, useSME bool, isSlowECore bool) {
	return smeAcquireClusterWithLimit(maxFastPerCluster, maxSlowPerCluster)
}

// SMEReleaseCluster releases a previously acquired global SME cluster lock.
func SMEReleaseCluster(cluster int) {
	smeReleaseCluster(cluster)
}

// smeAcquireCluster probes the current OS thread's CPU cluster and attempts to acquire
// that cluster's global SME unit lock.
// If acquired, returns (clusterIdx, true, false) and the caller MUST call smeReleaseCluster(clusterIdx).
// If the cluster's SME unit is already busy, returns (-1, false, isSlowECore) where isSlowECore
// is true when the thread is on an Efficiency core (Cluster 2) whose SME unit is already occupied.
func smeAcquireCluster() (clusterIdx int, useSME bool, isSlowECore bool) {
	return smeAcquireClusterWithLimit(SMEWorkersPerCluster, SMEWorkersPerCluster)
}

func smeAcquireClusterWithLimit(maxFastPerCluster, maxSlowPerCluster int32) (clusterIdx int, useSME bool, isSlowECore bool) {
	if !SMEUseHybridNEON {
		return -1, true, false
	}
	calibrateSMEClusters()

	// 1. Run the ~300ns per-core NEON probe.
	// Retry once if ticks > smeSuperCutoffTicks to filter out context-switch spikes or cold-core ramp-up.
	ticks := smeProbeCoreClockTicks()
	if ticks > smeSuperCutoffTicks {
		if t2 := smeProbeCoreClockTicks(); t2 > 0 && t2 < ticks {
			ticks = t2
		}
	}

	// 2. Assign to the closest cluster (0 = Super, 1 = Performance, 2 = Efficiency).
	primaryCluster := 2
	if ticks <= smeSuperCutoffTicks {
		primaryCluster = 0
	} else if ticks <= smePerfCutoffTicks {
		primaryCluster = 1
	}

	tryClaim := func(c int) bool {
		limit := maxFastPerCluster
		if c == 2 {
			limit = maxSlowPerCluster
		}
		if globalSMEClusterLocks[c].active.Load() >= limit {
			return false
		}
		if globalSMEClusterLocks[c].active.Add(1) <= limit {
			smeActiveTotal.Add(1)
			return true
		}
		globalSMEClusterLocks[c].active.Add(-1)
		return false
	}

	// 3. Try primary cluster first; if on a fast core (Super or Performance) and primary is taken
	// (e.g. due to thermal throttling shifting ticks between Super and Performance), fall back to
	// the adjacent fast cluster if it is currently unoccupied.
	if tryClaim(primaryCluster) {
		return primaryCluster, true, false
	}
	if primaryCluster <= 1 {
		otherFast := 1 - primaryCluster
		if tryClaim(otherFast) {
			return otherFast, true, false
		}
	}
	return -1, false, primaryCluster == 2
}

// smeReleaseCluster releases a previously acquired global SME cluster lock.
func smeReleaseCluster(cluster int) {
	if cluster >= 0 && cluster < numSMEClusters {
		smeActiveTotal.Add(-1)
		globalSMEClusterLocks[cluster].active.Add(-1)
	}
}

// smeAppendHybridWorkItems partitions a matrix multiplication into tapered 2D (M, N) panel
// work items (coarse at the head for SME workers, fine micro-tiles at the tail for NEON workers)
// ordered diagonally so concurrent SME and NEON workers pack distinct LHS/RHS panels in parallel
// and load-balance dynamically without tail stalls.
func smeAppendHybridWorkItems(
	dst []matmul.WorkItem,
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *matmul.CacheParams,
	maxWorkers int,
) []matmul.WorkItem {
	mr := max(1, params.LHSL1KernelRows)
	nr := max(1, params.RHSL1KernelCols)
	mc := max(mr, params.LHSPanelCrossSize)

	maxMBlocks := max(1, (lhsCrossSize+mr-1)/mr)
	maxNBlocks := max(1, (rhsCrossSize+nr-1)/nr)

	maxCap := max(64, cap(dst))
	maxItemsPerBatch := max(1, maxCap/max(1, batchSize))
	if batchSize >= maxWorkers || maxMBlocks*maxNBlocks < 2*maxWorkers || maxItemsPerBatch < 2*maxWorkers {
		return matmul.AppendWorkItems(dst, batchSize, lhsCrossSize, rhsCrossSize, params, maxWorkers)
	}

	colUnit := max(nr, min(params.RHSPanelCrossSize, 4*nr))
	actualM := max(1, (lhsCrossSize+mc-1)/mc)
	actualN := max(1, (rhsCrossSize+colUnit-1)/colUnit)

	targetMinItems := min(maxItemsPerBatch, 8*maxWorkers)
	for actualM*actualN < targetMinItems {
		if actualM*2 <= maxMBlocks && (actualM <= actualN || actualN*2 > maxNBlocks) {
			actualM *= 2
		} else if actualN*2 <= maxNBlocks {
			actualN *= 2
		} else {
			break
		}
	}
	// Coarsen if total items exceed stack buffer capacity.
	for actualM*actualN > maxItemsPerBatch {
		if actualN >= actualM && actualN > 1 {
			actualN = (actualN + 1) / 2
		} else if actualM > 1 {
			actualM = (actualM + 1) / 2
		} else {
			break
		}
	}

	// Emit items in diagonal order (n = (m + step) % actualN) so initial workers pack distinct
	// LHS row panels and distinct RHS col panels simultaneously.
	for batchIdx := range batchSize {
		for step := range actualN {
			for m := range actualM {
				n := (m + step) % actualN
				lhsRowStart := ((m*maxMBlocks + actualM - 1) / actualM) * mr
				lhsRowEnd := min(lhsCrossSize, (((m+1)*maxMBlocks+actualM-1)/actualM)*mr)
				rhsColStart := ((n*maxNBlocks + actualN - 1) / actualN) * nr
				rhsColEnd := min(rhsCrossSize, (((n+1)*maxNBlocks+actualN-1)/actualN)*nr)
				if lhsRowStart >= lhsRowEnd || rhsColStart >= rhsColEnd {
					continue
				}
				dst = append(dst, matmul.WorkItem{
					BatchStart:  batchIdx,
					BatchEnd:    batchIdx + 1,
					LHSRowStart: lhsRowStart,
					LHSRowEnd:   lhsRowEnd,
					RHSColStart: rhsColStart,
					RHSColEnd:   rhsColEnd,
				})
			}
		}
	}
	return dst
}

// neonLargeKernelFromSMEFloat32 computes a panel [lhsActiveRows, rhsActiveCols] using per-core NEON
// 4x16 sub-tiles directly from SME-packed packedLHS ([K, kernelRows] strips) and packedRHS ([K, kernelCols] strips).
func neonLargeKernelFromSMEFloat32(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	kernelRows, kernelCols int,
	accumulate bool,
) {
	_ = lhsPanelRows
	if lhsActiveRows <= 0 || rhsActiveCols <= 0 {
		return
	}
	lhsStepBytes := kernelRows * 4
	rhsStepBytes := kernelCols * 4
	outRowStrideBytes := rhsPanelCols * 4
	var tailBuf [4 * 16]float32
	tailPtr := unsafe.Pointer(&tailBuf[0])

	for rowIdx := 0; rowIdx < lhsActiveRows; rowIdx += 4 {
		lhsStrip := rowIdx / kernelRows
		subRow := rowIdx % kernelRows
		lhsPtr := unsafe.Pointer(&packedLHS[lhsStrip*contractingLen*kernelRows+subRow])
		remRows := lhsActiveRows - rowIdx

		for colIdx := 0; colIdx < rhsActiveCols; colIdx += 16 {
			rhsStrip := colIdx / kernelCols
			subCol := colIdx % kernelCols
			rhsPtr := unsafe.Pointer(&packedRHS[rhsStrip*contractingLen*kernelCols+subCol])
			remCols := rhsActiveCols - colIdx

			if remRows >= 4 && remCols >= 16 {
				outPtr := unsafe.Pointer(&packedOutput[rowIdx*rhsPanelCols+colIdx])
				neonSubTile4x16StridedFloat32Asm(
					lhsPtr, rhsPtr, outPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes,
					accumulate,
				)
			} else {
				neonSubTile4x16StridedFloat32Asm(
					lhsPtr, rhsPtr, tailPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, 16*4,
					false,
				)
				rLim := min(4, remRows)
				cLim := min(16, remCols)
				if accumulate {
					for r := range rLim {
						outRow := packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:]
						tailRow := tailBuf[r*16:]
						for c := range cLim {
							outRow[c] += tailRow[c]
						}
					}
				} else {
					for r := range rLim {
						copy(
							packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:(rowIdx+r)*rhsPanelCols+colIdx+cLim],
							tailBuf[r*16:r*16+cLim],
						)
					}
				}
			}
		}
	}
}

// neonLargeKernelFromSMEFloat16 computes a panel [lhsActiveRows, rhsActiveCols] for Float16 -> Float32
// using per-core NEON 8x8 sub-tiles directly from SME-packed packedLHS and packedRHS.
func neonLargeKernelFromSMEFloat16(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	kernelRows, kernelCols int,
	accumulate bool,
) {
	_ = lhsPanelRows
	if lhsActiveRows <= 0 || rhsActiveCols <= 0 {
		return
	}
	lhsStepBytes := kernelRows * 2
	rhsStepBytes := kernelCols * 2
	outRowStrideBytes := rhsPanelCols * 4
	var tailBuf [8 * 8]float32
	tailPtr := unsafe.Pointer(&tailBuf[0])

	for rowIdx := 0; rowIdx < lhsActiveRows; rowIdx += 8 {
		lhsStrip := rowIdx / kernelRows
		subRow := rowIdx % kernelRows
		lhsPtr := unsafe.Pointer(&packedLHS[lhsStrip*contractingLen*kernelRows+subRow])
		remRows := lhsActiveRows - rowIdx

		for colIdx := 0; colIdx < rhsActiveCols; colIdx += 8 {
			rhsStrip := colIdx / kernelCols
			subCol := colIdx % kernelCols
			rhsPtr := unsafe.Pointer(&packedRHS[rhsStrip*contractingLen*kernelCols+subCol])
			remCols := rhsActiveCols - colIdx

			if remRows >= 8 && remCols >= 8 {
				outPtr := unsafe.Pointer(&packedOutput[rowIdx*rhsPanelCols+colIdx])
				neonSubTile8x8StridedFloat16Asm(
					lhsPtr, rhsPtr, outPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes,
					accumulate,
				)
			} else {
				neonSubTile8x8StridedFloat16Asm(
					lhsPtr, rhsPtr, tailPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, 8*4,
					false,
				)
				rLim := min(8, remRows)
				cLim := min(8, remCols)
				if accumulate {
					for r := range rLim {
						outRow := packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:]
						tailRow := tailBuf[r*8:]
						for c := range cLim {
							outRow[c] += tailRow[c]
						}
					}
				} else {
					for r := range rLim {
						copy(
							packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:(rowIdx+r)*rhsPanelCols+colIdx+cLim],
							tailBuf[r*8:r*8+cLim],
						)
					}
				}
			}
		}
	}
}

// neonLargeKernelFromSMEBFloat16 computes a panel [lhsActiveRows, rhsActiveCols] for BFloat16 -> Float32
// using per-core NEON 8x8 sub-tiles directly from SME-packed packedLHS and packedRHS.
func neonLargeKernelFromSMEBFloat16(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	kernelRows, kernelCols int,
	accumulate bool,
) {
	_ = lhsPanelRows
	if lhsActiveRows <= 0 || rhsActiveCols <= 0 {
		return
	}
	lhsStepBytes := kernelRows * 2
	rhsStepBytes := kernelCols * 2
	outRowStrideBytes := rhsPanelCols * 4
	var tailBuf [8 * 8]float32
	tailPtr := unsafe.Pointer(&tailBuf[0])

	for rowIdx := 0; rowIdx < lhsActiveRows; rowIdx += 8 {
		lhsStrip := rowIdx / kernelRows
		subRow := rowIdx % kernelRows
		lhsPtr := unsafe.Pointer(&packedLHS[lhsStrip*contractingLen*kernelRows+subRow])
		remRows := lhsActiveRows - rowIdx

		for colIdx := 0; colIdx < rhsActiveCols; colIdx += 8 {
			rhsStrip := colIdx / kernelCols
			subCol := colIdx % kernelCols
			rhsPtr := unsafe.Pointer(&packedRHS[rhsStrip*contractingLen*kernelCols+subCol])
			remCols := rhsActiveCols - colIdx

			if remRows >= 8 && remCols >= 8 {
				outPtr := unsafe.Pointer(&packedOutput[rowIdx*rhsPanelCols+colIdx])
				neonSubTile8x8StridedBFloat16Asm(
					lhsPtr, rhsPtr, outPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes,
					accumulate,
				)
			} else {
				neonSubTile8x8StridedBFloat16Asm(
					lhsPtr, rhsPtr, tailPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, 8*4,
					false,
				)
				rLim := min(8, remRows)
				cLim := min(8, remCols)
				if accumulate {
					for r := range rLim {
						outRow := packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:]
						tailRow := tailBuf[r*8:]
						for c := range cLim {
							outRow[c] += tailRow[c]
						}
					}
				} else {
					for r := range rLim {
						copy(
							packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:(rowIdx+r)*rhsPanelCols+colIdx+cLim],
							tailBuf[r*8:r*8+cLim],
						)
					}
				}
			}
		}
	}
}

// neonLargeKernelFromSMEFloat64 computes a panel [lhsActiveRows, rhsActiveCols] for Float64
// using per-core NEON 8x4 sub-tiles directly from SME-packed packedLHS and packedRHS.
func neonLargeKernelFromSMEFloat64(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	kernelRows, kernelCols int,
	accumulate bool,
) {
	_ = lhsPanelRows
	if lhsActiveRows <= 0 || rhsActiveCols <= 0 {
		return
	}
	lhsStepBytes := kernelRows * 8
	rhsStepBytes := kernelCols * 8
	outRowStrideBytes := rhsPanelCols * 8
	var tailBuf [8 * 4]float64
	tailPtr := unsafe.Pointer(&tailBuf[0])

	for rowIdx := 0; rowIdx < lhsActiveRows; rowIdx += 8 {
		lhsStrip := rowIdx / kernelRows
		subRow := rowIdx % kernelRows
		lhsPtr := unsafe.Pointer(&packedLHS[lhsStrip*contractingLen*kernelRows+subRow])
		remRows := lhsActiveRows - rowIdx

		for colIdx := 0; colIdx < rhsActiveCols; colIdx += 4 {
			rhsStrip := colIdx / kernelCols
			subCol := colIdx % kernelCols
			rhsPtr := unsafe.Pointer(&packedRHS[rhsStrip*contractingLen*kernelCols+subCol])
			remCols := rhsActiveCols - colIdx

			if remRows >= 8 && remCols >= 4 {
				outPtr := unsafe.Pointer(&packedOutput[rowIdx*rhsPanelCols+colIdx])
				neonSubTile8x4StridedFloat64Asm(
					lhsPtr, rhsPtr, outPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes,
					accumulate,
				)
			} else {
				neonSubTile8x4StridedFloat64Asm(
					lhsPtr, rhsPtr, tailPtr,
					contractingLen, lhsStepBytes, rhsStepBytes, 4*8,
					false,
				)
				rLim := min(8, remRows)
				cLim := min(4, remCols)
				if accumulate {
					for r := range rLim {
						outRow := packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:]
						tailRow := tailBuf[r*4:]
						for c := range cLim {
							outRow[c] += tailRow[c]
						}
					}
				} else {
					for r := range rLim {
						copy(
							packedOutput[(rowIdx+r)*rhsPanelCols+colIdx:(rowIdx+r)*rhsPanelCols+colIdx+cLim],
							tailBuf[r*4:r*4+cLim],
						)
					}
				}
			}
		}
	}
}


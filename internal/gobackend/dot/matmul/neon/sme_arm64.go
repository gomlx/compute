// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/internal/gobackend/simdarch"
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

// smePackLHS32Float32Asm packs [copyRows, contractingCols] of float32 into [ceil(copyRows/32), contractingCols, 32]
// using SME ZA0..ZA3 hardware horizontal-load and vertical-store. Caller must hold the cluster SME lock.
// Defined in sme_arm64.s.
//
//go:noescape
func smePackLHS32Float32Asm(
	lhsPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes uintptr,
	copyRows, contractingCols int,
)

// smeDirectNonTransposedFloat32Asm_32x32 performs a fused single-SMSTART 32x32 SME GEMM for LayoutNonTransposed
// (C = A @ B), transposing 32-row strips of LHS in-SME into packedLHSStrip [K, 32] and reading RHS directly
// from rhs without packing.
// Defined in sme_arm64.s.
//
//go:noescape
func smeDirectNonTransposedFloat32Asm_32x32(
	lhs, rhs, output, packedLHSStrip []float32,
	lhsActiveRows, rhsActiveCols, contractingLen int,
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
// coprocessor is attached to the L2 cache and shared across all CPU cores sharing
// that L2 cache (see package simdarch):
//   - L2 Cluster 1 (Super + Performance cores on M6): shares 1 high-throughput SME unit (~2.26 TFlops/s)
//   - L2 Cluster 0 (Efficiency cores on M6):          shares 1 smaller SME unit (~0.38 TFlops/s)
//
// Hardware L2 cluster identification (`(TPIDR_EL0 >> 12) & 0xff`) and global per-L2-cluster
// atomic locking are provided by `simdarch.AcquireSME` / `simdarch.ReleaseSME`.
// Workers that do not acquire an SME lock execute `neonLargeKernelFromSME*` on
// their private per-core NEON pipelines, reading directly from the exact same
// `[K, 32]` (or `[K, 16]`) SME-packed LHS and RHS panels by advancing their
// pointers by `kernelRows * sizeof(T)` and `kernelCols * sizeof(T)` bytes per K-step.

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

var (
	// SMEUseHybridNEON enables global per-cluster SME locking with per-core NEON co-execution
	// on workers whose L2 cluster SME unit is already saturated.
	SMEUseHybridNEON = simdarch.SMEUseHybridNEON

	// SMEWorkersPerCluster is the maximum number of concurrent SME workers per L2 cluster.
	SMEWorkersPerCluster int32 = simdarch.SMEWorkersPerCluster

	// SMEDropECoreNEON drops Tier E (Efficiency-core NEON fallback, ~50x slower than Tier A SME)
	// whenever at least one SME worker is active. With the 3-zone tapered work queue where NEON workers
	// only pull fine 32x32 tail tiles and yield when rem <= 8*smeActive, keeping Tier E enabled
	// adds ~5-8% throughput on large matrices.
	SMEDropECoreNEON = false

	// SMETwoSMEsOnly restricts execution to workers that acquire an SME cluster lock (Tier A + Tier B),
	// dropping all NEON fallback workers.
	SMETwoSMEsOnly = false
)

// SMEAcquireCluster reads the current OS thread's L2 cluster ID via TPIDR_EL0 and attempts to acquire
// that cluster's global SME unit lock.
func SMEAcquireCluster() (clusterIdx int, useSME bool, isEfficiencyCore bool) {
	return smeAcquireClusterWithLimit(SMEWorkersPerCluster, SMEWorkersPerCluster)
}

// SMEAcquireClusterWithLimit is like SMEAcquireCluster, but allows specifying a higher concurrency
// limit on fast L2 clusters (Super/Performance) for two-stage pipelines (such as ConvGeneral).
func SMEAcquireClusterWithLimit(maxFastPerCluster, maxSlowPerCluster int32) (clusterIdx int, useSME bool, isEfficiencyCore bool) {
	return smeAcquireClusterWithLimit(maxFastPerCluster, maxSlowPerCluster)
}

// SMEReleaseCluster releases a previously acquired global SME cluster lock.
func SMEReleaseCluster(cluster int) {
	smeReleaseCluster(cluster)
}

func smeCurrentCluster() (clusterIdx int, isEfficiencyCore bool) {
	cl, _, isEff := simdarch.CurrentSMECluster()
	return cl, isEff
}

func smeCallerOnFastCluster() bool {
	_, _, isEff := simdarch.CurrentSMECluster()
	return !isEff
}

// smeShouldUseSingleWorker decides whether an SME MatMul should run inline on the calling goroutine
// (Regime 2: small/medium ops without acquiring the whole-chip gate) rather than fanning out across
// multiple workers.
//   - On Cluster 1 (Super/Performance core), any matrix <= 16M MACs (<= 256^3, ~0.65-20 us on Tier A SME)
//     completes faster on a single SME worker than waking background workers.
//   - On Cluster 0 (Efficiency core), small matrices <= 2M MACs (<= 128^3) finish in <= 10 us on Tier B SME,
//     and medium matrices (2M < flops <= 16M MACs) also run inline on Cluster 0 SME whenever Cluster 1's
//     SME unit is already busy or a large op holds the whole-chip gate (so both physical SME units work
//     in parallel). If Cluster 1's SME unit is idle, medium matrices fan out so Tier A SME (2.26 TFlops/s)
//     executes the work.
func smeShouldUseSingleWorker(flops int, wholeChipHeld bool) bool {
	if flops > 16*1024*1024 {
		return false
	}
	if smeCallerOnFastCluster() {
		return true
	}
	// Caller is on an Efficiency core (Cluster 0):
	// If the whole-chip gate is held by a large op or Cluster 1's fast SME unit is already busy,
	// run this small/medium op inline on Cluster 0 SME rather than queuing behind the large op.
	// Otherwise, fan out so a Cluster 1 worker (2.26 TFlops/s) executes it.
	return wholeChipHeld || !simdarch.IsFastSMEIdle()
}

func smeAcquireCluster() (clusterIdx int, useSME bool, isEfficiencyCore bool) {
	return smeAcquireClusterWithLimit(SMEWorkersPerCluster, SMEWorkersPerCluster)
}

func smeAcquireClusterSingleWorker() (clusterIdx int, useSME bool, isEfficiencyCore bool) {
	if !SMEUseHybridNEON || !simdarch.SMEUseHybridNEON {
		return -1, true, false
	}
	return simdarch.AcquireSMEWithSpin(SMEWorkersPerCluster, SMEWorkersPerCluster, 256)
}

func smeAcquireClusterForWork(numWorkItems int) (clusterIdx int, useSME bool, isEfficiencyCore bool) {
	maxSlow := SMEWorkersPerCluster
	if simdarch.NumHighPerfSMEUnits > 0 && numWorkItems <= 16 {
		maxSlow = 0
	}
	return smeAcquireClusterWithLimit(2*SMEWorkersPerCluster, maxSlow)
}

func smeAcquireClusterWithLimit(maxFastPerCluster, maxSlowPerCluster int32) (clusterIdx int, useSME bool, isEfficiencyCore bool) {
	if !SMEUseHybridNEON || !simdarch.SMEUseHybridNEON {
		return -1, true, false
	}
	return simdarch.AcquireSMEWithLimit(maxFastPerCluster, maxSlowPerCluster)
}

func smeReleaseCluster(cluster int) {
	simdarch.ReleaseSME(cluster)
}

// smeAppendHybridWorkItems partitions a matrix multiplication into a 3-zone tapered 2D (M, N) queue
// ordered diagonally across L2 cache panels:
//   - Head zone (first ~65% of panels): coarse [mc x nc] (e.g. 64x128 = 8 SME 32x32 tiles) panels
//     pulled by Tier A (Cluster 1 SME) and early Tier B (Cluster 0 SME).
//   - Middle zone (next ~15% of panels): medium [mr x 2*nr] (e.g. 32x64 = 2 SME 32x32 tiles) sub-panels.
//   - Tail zone (last ~20% of panels): fine [mr x nr] (e.g. 32x32 = 1 SME tile) sub-panels pulled from
//     the tail by Tier C/D (Super/P-core NEON) and late Tier B/A workers so tail latency is bounded to ~10 µs.
//
// It returns the populated slice and coarseEnd (the index where the coarse head zone ends).
func smeAppendHybridWorkItems(
	dst []matmul.WorkItem,
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *matmul.CacheParams,
	maxWorkers int,
) ([]matmul.WorkItem, int) {
	mr := max(1, params.LHSL1KernelRows)
	nr := max(1, params.RHSL1KernelCols)
	mc := max(mr, params.LHSPanelCrossSize)
	nc := max(nr, params.RHSPanelCrossSize)

	maxMBlocks := max(1, (lhsCrossSize+mr-1)/mr)
	maxNBlocks := max(1, (rhsCrossSize+nr-1)/nr)

	maxCap := max(64, cap(dst))
	if batchSize >= maxWorkers || maxMBlocks*maxNBlocks < 2*maxWorkers {
		items := matmul.AppendWorkItems(dst, batchSize, lhsCrossSize, rhsCrossSize, params, maxWorkers)
		return items, len(items)
	}

	numMPanels := max(1, (lhsCrossSize+mc-1)/mc)
	numNPanels := max(1, (rhsCrossSize+nc-1)/nc)

	// If total panels exceed maxCap/2, group adjacent row panels in the coarse grid
	// so there is room in dst to split the middle and tail zones into fine 32x32 tiles.
	rowGroup := 1
	for batchSize*((numMPanels+rowGroup-1)/rowGroup)*numNPanels > maxCap/2 && (numMPanels+rowGroup-1)/rowGroup > 1 {
		rowGroup *= 2
	}
	gridM := max(1, (numMPanels+rowGroup-1)/rowGroup)
	gridN := numNPanels
	totalPanels := batchSize * gridM * gridN

	coarseSpanM := rowGroup * mc
	coarseSpanN := nc
	subRowStep := mr
	if numMPanels == 1 {
		// Keep single-row-panel skinny matrices (M <= mc) unsplit along M so each RHS column strip
		// is visited only once and does not require a sharedRHSPanels buffer.
		subRowStep = coarseSpanM
	}
	fineTilesPerPanel := max(1, max(1, coarseSpanM/subRowStep)*(coarseSpanN/nr))
	medTilesPerPanel := max(1, max(1, coarseSpanM/subRowStep)*max(1, coarseSpanN/(2*nr)))

	// Determine how many panels at the end to split into fine (mr x nr) and medium (mr x 2*nr) items
	// within the capacity maxCap: ~78% coarse at head (matching the ~80% throughput share of the 2 SME
	// units), ~10% medium in middle, and ~12% fine at tail (matching the ~20% throughput share of the
	// 5 Super/P-core NEON units).
	availExtra := max(0, maxCap-totalPanels)
	wantFinePanels := max(maxWorkers/2, totalPanels/8) // ~12.5% of panels (at least maxWorkers/2)
	wantMedPanels := max(maxWorkers/2, totalPanels/10) // ~10% of panels (at least maxWorkers/2)

	fineExtraPerPanel := max(1, fineTilesPerPanel-1)
	medExtraPerPanel := max(1, medTilesPerPanel-1)

	numFinePanels := min(wantFinePanels, availExtra/fineExtraPerPanel)
	availExtra -= numFinePanels * fineExtraPerPanel
	numMedPanels := min(wantMedPanels, availExtra/medExtraPerPanel)
	if numFinePanels+numMedPanels >= totalPanels {
		numFinePanels = max(0, totalPanels-1)
		numMedPanels = 0
	}
	coarsePanels := totalPanels - numFinePanels - numMedPanels

	panelIdx := 0
	coarseEnd := 0
	for batchIdx := range batchSize {
		for step := range gridN {
			for m := range gridM {
				n := (m + step) % gridN
				pRowStart := m * coarseSpanM
				pRowEnd := min(lhsCrossSize, (m+1)*coarseSpanM)
				pColStart := n * coarseSpanN
				pColEnd := min(rhsCrossSize, (n+1)*coarseSpanN)
				if pRowStart >= pRowEnd || pColStart >= pColEnd {
					panelIdx++
					continue
				}

				if panelIdx < coarsePanels {
					dst = append(dst, matmul.WorkItem{
						BatchStart:  batchIdx,
						BatchEnd:    batchIdx + 1,
						LHSRowStart: pRowStart,
						LHSRowEnd:   pRowEnd,
						RHSColStart: pColStart,
						RHSColEnd:   pColEnd,
					})
					coarseEnd = len(dst)
				} else if panelIdx < coarsePanels+numMedPanels {
					medColStep := max(nr, 2*nr)
					for r := pRowStart; r < pRowEnd; r += subRowStep {
						rEnd := min(pRowEnd, r+subRowStep)
						for c := pColStart; c < pColEnd; c += medColStep {
							cEnd := min(pColEnd, c+medColStep)
							dst = append(dst, matmul.WorkItem{
								BatchStart:  batchIdx,
								BatchEnd:    batchIdx + 1,
								LHSRowStart: r,
								LHSRowEnd:   rEnd,
								RHSColStart: c,
								RHSColEnd:   cEnd,
							})
						}
					}
				} else {
					for r := pRowStart; r < pRowEnd; r += subRowStep {
						rEnd := min(pRowEnd, r+subRowStep)
						for c := pColStart; c < pColEnd; c += nr {
							cEnd := min(pColEnd, c+nr)
							dst = append(dst, matmul.WorkItem{
								BatchStart:  batchIdx,
								BatchEnd:    batchIdx + 1,
								LHSRowStart: r,
								LHSRowEnd:   rEnd,
								RHSColStart: c,
								RHSColEnd:   cEnd,
							})
						}
					}
				}
				panelIdx++
			}
		}
	}
	if coarseEnd == 0 {
		coarseEnd = len(dst)
	}
	return dst, coarseEnd
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


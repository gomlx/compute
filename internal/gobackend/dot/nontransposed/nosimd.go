// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nontransposed

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
)

var (
	// NoSIMD32Params are generic assumptions for L1/L2/L3 cache sizes for 32 bits dtypes (float32, int32, uint32)
	//
	// These values are somewhat arbitrary, assuming "standard" modern cache sizes.
	// They are parameterized so they can be tuned or determined dynamically later.
	NoSIMD32Params = CacheParams{
		// Do not change these 2 values: they are hard-coded by the allocated registers in basicSymmetricMicroKernel8x8.
		LHSL1KernelRows: 2, // Mr: Rows of LHS in local registers.
		RHSL1KernelCols: 4, // Nr: Cols of RHS in local registers.

		PanelContractingSize: 512, // Kc: L1 Block contracting "depth".
		LHSPanelCrossSize:    2,   // Mc: Block Height fitting L2/L3 cache.
		RHSPanelCrossSize:    512, // Nc: Block Width fitting L2/L3 cache.
	}

	// Threshold in byte size for switching to the small matrix multiplication kernel.
	// If the total number of operations is below this threshold, the small
	// matrix multiplication kernel is used instead of the tiled implementation.
	// This is a heuristic and may need to be tuned for different architectures.
	// Expressed in number of bytes.
	noSIMDSmallMatMulSizeThreshold = 4 * 1024 * 1024

	// Minimum number of flops per worker: above this number, if possible we should
	// parallelize computation on separate goroutines.
	noSIMDMinMatMulFlopsPerWorker = 10 * 1024
)

func init() {
	registerNoSIMD(false)
}

func RegisterNoSIMDForTests() {
	registerNoSIMD(true)
}

func registerNoSIMD(forTests bool) {
	// DTypePairMap: callImplementationDTypePairMap (ints, same)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int16, noSIMDRouter[int16, int16], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int8, noSIMDRouter[int8, int8], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (ints, int32,int64)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int32, noSIMDRouter[int16, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int64, noSIMDRouter[int16, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int64, noSIMDRouter[int32, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int32, noSIMDRouter[int64, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int32, noSIMDRouter[int8, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int64, noSIMDRouter[int8, int64], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (uints, same)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint16, noSIMDRouter[uint16, uint16], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint8, noSIMDRouter[uint8, uint8], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (uints, uint32,uint64)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint32, noSIMDRouter[uint16, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint64, noSIMDRouter[uint16, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint64, noSIMDRouter[uint32, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint32, noSIMDRouter[uint64, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint32, noSIMDRouter[uint8, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint64, noSIMDRouter[uint8, uint64], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (floats, floats)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, noSIMDRouter[float32, float32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float64, noSIMDRouter[float32, float64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float32, noSIMDRouter[float64, float32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, noSIMDRouter[float64, float64], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (half, float32)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, noSIMDHalfPrecisionRouter[bfloat16.BFloat16, float32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, noSIMDHalfPrecisionRouter[float16.Float16, float32], gobackend.PriorityTyped, forTests)
}

// Auto-generate alternate specialized versions of noSIMD operations -- for half-precision input data types.
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_router.go -tags=half
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_small.go -tags=half

/*
func basicSymmetricGenericLargeGEMMParallel[T dtypes.Number](
	alpha, beta T,
	lhsFlat, rhsFlat []T, outputFlat []T,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	lhsBatchStride, rhsBatchStride, outputBatchStride int,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn,
	pool *workerspool.Pool) error {

	params := &NoSIMD32Params

	// Split work in reasonable number of "chunks".
	maxWorkers := 1
	if pool != nil {
		maxWorkers = pool.AdjustedMaxParallelism()
	}
	if maxWorkers <= 1 {
		// Do everything sequentially.
		packedLhsRef, packedLHS := bufAllocFn(params.LHSPanelCrossSize * params.PanelContractingSize)
		packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSPanelCrossSize)
		packedOutRef, packedOutput := bufAllocFn(params.LHSPanelCrossSize * params.RHSPanelCrossSize)
		defer func() {
			bufReleaseFn(packedLhsRef)
			bufReleaseFn(packedRhsRef)
			bufReleaseFn(packedOutRef)
		}()
		for batchIdx := range batchSize {
			batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
			batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
			batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
			basicSymmetricLargeGemmSlice(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				lhsCrossSize, rhsCrossSize, contractingSize,
				NoSIMD32Params,
				0, lhsCrossSize, 0, rhsCrossSize,
				packedLHS, packedRHS, packedOutput,
			)
		}
		return nil
	}

	// 1. Split work in workItems.
	workChan := make(chan workItem, max(2000, 2*maxWorkers))
	go feedWorkItems(
		batchSize, lhsCrossSize, rhsCrossSize,
		params, maxWorkers, workChan)

	// 2. Saturate (fan-out workers) on workItems.
	pool.Saturate(func() {
		packedLhsRef, packedLHS := bufAllocFn(params.LHSPanelCrossSize * params.PanelContractingSize)
		packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSPanelCrossSize)
		packedOutRef, packedOutput := bufAllocFn(params.LHSPanelCrossSize * params.RHSPanelCrossSize)
		defer func() {
			bufReleaseFn(packedLhsRef)
			bufReleaseFn(packedRhsRef)
			bufReleaseFn(packedOutRef)
		}()

		for item := range workChan {
			for batchIdx := item.batchStart; batchIdx < item.batchEnd; batchIdx++ {
				batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
				batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
				batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
				basicSymmetricLargeGemmSlice(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					NoSIMD32Params,
					item.lhsRowStart, item.lhsRowEnd, item.rhsColStart, item.rhsColEnd,
					packedLHS, packedRHS, packedOutput,
				)
			}
		}
	})
	return nil
}

// basicSymmetricLargeGemmSlice performs a slice of the matrix multiplication on one example: lhs, rhs an output
// must already have sliced one example of the batch dimension.
//
// packedLHS and packedRHS must be pre-allocated buffers of appropriate size.
func basicSymmetricLargeGemmSlice[T dtypes.Number](
	alpha, beta T,
	lhs, rhs, output []T,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	params CacheParams,
	rowStart, rowEnd, colStart, colEnd int,
	packedLHS, packedRHS, packedOutput []T,
) {
	// Loop 5 (jc): Tiling N (Output Columns)
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; rhsPanelColIdx += params.RHSPanelCrossSize {
		rhsPanelWidth := min(params.RHSPanelCrossSize, colEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling K (Depth)
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			packRHS(rhs, packedRHS, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling M (Output Rows)
			for lhsPanelRowIdx := rowStart; lhsPanelRowIdx < rowEnd; lhsPanelRowIdx += params.LHSPanelCrossSize {
				lhsPanelHeight := min(params.LHSPanelCrossSize, rowEnd-lhsPanelRowIdx)

				// PACK LHS
				packLHS(lhs, packedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				basicSymmetricPanel(
					packedLHS, packedRHS, packedOutput,
					params.LHSPanelCrossSize, params.RHSPanelCrossSize,
					contractingPanelWidth,
					lhsPanelHeight, rhsPanelWidth,
				)

				// Accumulate (or write) packedOutput to output.
				effectiveBeta := beta
				if contractingPanelIdx > 0 {
					effectiveBeta = 1
				}
				applyPackedOutput(
					packedOutput, output,
					alpha, effectiveBeta,
					params.RHSPanelCrossSize,
					lhsPanelRowIdx, rhsPanelColIdx,
					rhsCrossSize,
					lhsPanelHeight, rhsPanelWidth)
			}
		}
	}
}

// basicSymmetricPanel implements the gemm for a lhs and rhs packed panels
// into an output panel, using packedOutput as intermediate.
//
// It uses register blocking: it divides the 4x4 matrix in 4 4x4 sub-matrices.
// For each sub-matrix it iterates over k (contracting dim), accumulating the results
// in local variables (registers).
// finally it writes the results to output.
//
// It assumes lhsL1KernelRows=4 and rhsL1KernelCols=4.
//
// See basicSymmetricMicroKernel for documentation on arguments.
func basicSymmetricPanel[T dtypes.Number](
	packedLHS, packedRHS []T,
	packedOutput []T,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
) {
	const kernelRows = 2
	const kernelCols = 4

	// BCE hints
	_ = packedLHS[contractingLen]
	_ = packedRHS[contractingLen]
	_ = packedOutput[lhsPanelRows*rhsPanelCols-1]

	// Strides in the packed buffers for one block.
	lhsBlockStride := kernelRows * contractingLen
	rhsBlockStride := kernelCols * contractingLen
	lhsOffset := 0

	// Write active part of 4x4 block to output
	// Helper to write a row
	// Write active part of 4x4 block to output
	// Bounds check is not needed as packedOutput is allocated to panel size, and we will discard
	// whatever is written beyond the active part.

	for rowIdx := 0; rowIdx < lhsActiveRows; rowIdx += kernelRows {
		rhsOffset := 0
		for colIdx := 0; colIdx < rhsActiveCols; colIdx += kernelCols {
			// Process 2x4 block at (r, c)
			// Accumulators for 2x4 block
			var c00, c01, c02, c03 T
			var c10, c11, c12, c13 T

			idxLhs := lhsOffset
			idxRhs := rhsOffset

			// K-Loop unrolled by 4
			k := 0
			for ; k+3 < contractingLen; k += 4 {
				// We need 4 steps.
				// For each step (l is k offset):
				//   load lhs (2 vals), load rhs (4 vals), fma.

				// --- Step 0 ---
				// BCE hint
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 := packedLHS[idxLhs]
				l1 := packedLHS[idxLhs+1]

				r0 := packedRHS[idxRhs]
				r1 := packedRHS[idxRhs+1]
				r2 := packedRHS[idxRhs+2]
				r3 := packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols

				// --- Step 1 ---
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 = packedLHS[idxLhs]
				l1 = packedLHS[idxLhs+1]
				r0 = packedRHS[idxRhs]
				r1 = packedRHS[idxRhs+1]
				r2 = packedRHS[idxRhs+2]
				r3 = packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols

				// --- Step 2 ---
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 = packedLHS[idxLhs]
				l1 = packedLHS[idxLhs+1]
				r0 = packedRHS[idxRhs]
				r1 = packedRHS[idxRhs+1]
				r2 = packedRHS[idxRhs+2]
				r3 = packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols

				// --- Step 3 ---
				_ = packedLHS[idxLhs+1]
				_ = packedRHS[idxRhs+3]
				l0 = packedLHS[idxLhs]
				l1 = packedLHS[idxLhs+1]
				r0 = packedRHS[idxRhs]
				r1 = packedRHS[idxRhs+1]
				r2 = packedRHS[idxRhs+2]
				r3 = packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols
			}

			// K-Loop Tail
			for ; k < contractingLen; k++ {
				l0 := packedLHS[idxLhs]
				l1 := packedLHS[idxLhs+1]

				r0 := packedRHS[idxRhs]
				r1 := packedRHS[idxRhs+1]
				r2 := packedRHS[idxRhs+2]
				r3 := packedRHS[idxRhs+3]

				c00 += l0 * r0
				c01 += l0 * r1
				c02 += l0 * r2
				c03 += l0 * r3
				c10 += l1 * r0
				c11 += l1 * r1
				c12 += l1 * r2
				c13 += l1 * r3

				idxLhs += kernelRows
				idxRhs += kernelCols
			}

			// Optimization: write full 2x4 block directly to packedOutput.
			// The buffer is large enough even for fringe blocks.
			// Row 0
			rowOffset := rowIdx*rhsPanelCols + colIdx
			packedOutput[rowOffset] = c00
			packedOutput[rowOffset+1] = c01
			packedOutput[rowOffset+2] = c02
			packedOutput[rowOffset+3] = c03

			// Row 1
			rowOffset1 := rowOffset + rhsPanelCols
			packedOutput[rowOffset1] = c10
			packedOutput[rowOffset1+1] = c11
			packedOutput[rowOffset1+2] = c12
			packedOutput[rowOffset1+3] = c13

			rhsOffset += rhsBlockStride
		}
		lhsOffset += lhsBlockStride
	}
}

// applyPackedOutput applies the computed packedOutput to the final output.
func applyPackedOutput[T dtypes.Number](
	packedOutput, output []T,
	alpha, beta T,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int, // Global output offsets
	outputRowStride int,
	height, width int, // actual amount of data to copy
) {
	for r := range height {
		packedRowOffset := r * packedOutputRowStride
		outRowOffset := (lhsRowOffset+r)*outputRowStride + rhsColOffset
		for c := range width {
			val := packedOutput[packedRowOffset+c]
			basicWriteScalar(output, outRowOffset+c, alpha, beta, val)
		}
	}
}
*/

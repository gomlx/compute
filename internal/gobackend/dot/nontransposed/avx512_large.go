// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package nontransposed

import (
	"runtime"
	"simd/archsimd"
	"sync"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
)

// avx512LargeFloat32 implements a "packing" version of the non-SIMD matrix, and parallelizes if possible.
func avx512LargeFloat32( //alt:f32
	//alt:bf16 func avx512LargeBFloat16(
	backend *gobackend.Backend,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32, //alt:f32|bf16
) {

	params := AVX512Params16Registers
	maxWorkers := backend.Workers.AdjustedMaxParallelism()

	// Strides for each matrix in the batch.
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	if maxWorkers <= 1 {
		// No parallelism, do each matrix multiplication in the batch sequentially.
		packedLHSRef, packedLHS, ok := GetBuffer[float32](backend, params.LHSPanelCrossSize*params.PanelContractingSize) //alt:f32
		//alt:bf16 packedLHSRef, packedLHS, ok := GetBuffer[bfloat16.BFloat16](backend, params.LHSPanelCrossSize*params.PanelContractingSize)
		if !ok {
			return
		}
		defer ReleaseBuffer(packedLHSRef)
		packedRHSRef, packedRHS, ok := GetBuffer[float32](backend, params.PanelContractingSize*params.RHSPanelCrossSize) //alt:f32
		//alt:bf16 packedRHSRef, packedRHS, ok := GetBuffer[bfloat16.BFloat16](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
		if !ok {
			return
		}
		defer ReleaseBuffer(packedRHSRef)
		packedOutputRef, packedOutput, ok := GetBuffer[float32](backend, params.LHSPanelCrossSize*params.RHSPanelCrossSize) //alt:f32|bf16
		if !ok {
			return
		}
		defer ReleaseBuffer(packedOutputRef)
		lhsFlatIdx := 0
		rhsFlatIdx := 0
		outputFlatIdx := 0
		for range batchSize {
			batchLHS := lhs[lhsFlatIdx : lhsFlatIdx+lhsBatchStride]
			batchRHS := rhs[rhsFlatIdx : rhsFlatIdx+rhsBatchStride]
			batchOutput := output[outputFlatIdx : outputFlatIdx+outputBatchStride]
			avx512LargeMatrixSliceFloat32( //alt:f32
				//alt:bf16 avx512LargeMatrixSliceBFloat16(
				batchLHS, batchRHS, batchOutput,
				lhsCrossSize, rhsCrossSize, contractingSize,
				0, lhsCrossSize, 0, rhsCrossSize,
				params,
				packedLHS, packedRHS, packedOutput,
			)
			lhsFlatIdx += lhsBatchStride
			rhsFlatIdx += rhsBatchStride
			outputFlatIdx += outputBatchStride
		}
		return
	}

	// 1. Split work in workItems.
	workChan := make(chan workItem, max(2000, 2*maxWorkers))
	var wg sync.WaitGroup
	wg.Go(func() {
		feedWorkItems(
			batchSize, lhsCrossSize, rhsCrossSize,
			&params, maxWorkers, workChan)
	})

	// 2. Saturate (fan-out workers) on workItems.
	backend.Workers.Saturate(func() {
		// No parallelism, do everything sequentially.
		packedLHSRef, packedLHS, ok := GetBuffer[float32](backend, params.LHSPanelCrossSize*params.PanelContractingSize) //alt:f32
		//alt:bf16 packedLHSRef, packedLHS, ok := GetBuffer[bfloat16.BFloat16](backend, params.LHSPanelCrossSize*params.PanelContractingSize)
		if !ok {
			return
		}
		defer ReleaseBuffer(packedLHSRef)
		packedRHSRef, packedRHS, ok := GetBuffer[float32](backend, params.PanelContractingSize*params.RHSPanelCrossSize) //alt:f32
		//alt:bf16 packedRHSRef, packedRHS, ok := GetBuffer[bfloat16.BFloat16](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
		if !ok {
			return
		}
		defer ReleaseBuffer(packedRHSRef)
		packedOutputRef, packedOutput, ok := GetBuffer[float32](backend, params.LHSPanelCrossSize*params.RHSPanelCrossSize) //alt:f32|bf16
		if !ok {
			return
		}
		defer ReleaseBuffer(packedOutputRef)
		for item := range workChan {
			for batchIdx := item.batchStart; batchIdx < item.batchEnd; batchIdx++ {
				batchLhs := lhs[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
				batchRhs := rhs[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
				batchOutput := output[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
				avx512LargeMatrixSliceFloat32( //alt:generic
					//alt:bf16 avx512LargeMatrixSliceBFloat16(
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					item.lhsRowStart, item.lhsRowEnd, item.rhsColStart, item.rhsColEnd,
					params,
					packedLHS, packedRHS, packedOutput,
				)
			}
		}
	})
	wg.Wait()
}

// avx512LargeMatrixSliceFloat32 performs a slice of the matrix multiplication on one example: lhs, rhs an output
// must already have sliced one example of the batch dimension.
//
// Here there are no batch dimensions anymore, it only applies to a slice of one matrix (2D).
//
// packedLHS and packedRHS must be pre-allocated buffers of appropriate size.
func avx512LargeMatrixSliceFloat32( //alt:f32
	//alt:bf16 func avx512LargeMatrixSliceBFloat16(
	lhsMatrix, rhsMatrix []float32, outputMatrix []float32, //alt:f32
	//alt:bf16 lhsMatrix, rhsMatrix []bfloat16.BFloat16, outputMatrix []float32,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	rowStart, rowEnd, colStart, colEnd int,
	params CacheParams,
	packedLHS, packedRHS []float32, packedOutput []float32, //alt:f32
	//alt:bf16 packedLHS, packedRHS []bfloat16.BFloat16, packedOutput []float32,
) {
	_ = lhsCrossSize // Not used, rowStart and rowEnd < lhsCrossSize are enough.

	if params.LHSL1KernelRows != 4 || params.RHSL1KernelCols != 32 {
		panic(errors.Errorf("unsupported kernel L1 block sizes for avx512 kernel: lhsL1BlockRows=%d, rhsL1BlockCols=%d, wanted 4 and 32 respectively (params=%+v)",
			params.LHSL1KernelRows, params.RHSL1KernelCols, params))
	}

	// Loop 5 (jc): Tiling RHS cross axis (N), the output columns.
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; rhsPanelColIdx += params.RHSPanelCrossSize {
		rhsPanelWidth := min(params.RHSPanelCrossSize, colEnd-rhsPanelColIdx)

		// Loop 4 (p): Tiling the contracting axis (K)
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			avx512PackRHSFloat32(rhsMatrix, packedRHS, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols) //alt:f32
			//alt:bf16 avx512PackRHSBFloat16(rhsMatrix, packedRHS, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)

			// Loop 3 (ic): Tiling LHS cross axis (M), i.e. the output rows.
			for lhsPanelRowIdx := rowStart; lhsPanelRowIdx < rowEnd; lhsPanelRowIdx += params.LHSPanelCrossSize {
				lhsPanelHeight := min(params.LHSPanelCrossSize, rowEnd-lhsPanelRowIdx)
				avx512PackLHSFloat32(lhsMatrix, packedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows) //alt:f32
				//alt:bf16 avx512PackLHSBFloat16(lhsMatrix, packedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				avx512LargePanelFloat32( //alt:f32
					//alt:bf16 avx512LargePanelBFloat16(
					packedLHS, packedRHS, packedOutput,
					params.LHSPanelCrossSize, params.RHSPanelCrossSize,
					contractingPanelWidth,
					lhsPanelHeight, rhsPanelWidth,
				)

				// Accumulate (or write) packedOutput to output.
				isFirstContractingPanel := contractingPanelIdx == 0
				avx512ApplyPackedOutputFloat32( //alt:f32|bf16
					packedOutput, outputMatrix,
					isFirstContractingPanel,
					params.RHSPanelCrossSize,
					lhsPanelRowIdx, rhsPanelColIdx,
					rhsCrossSize,
					lhsPanelHeight, rhsPanelWidth)
			}
		}
	}
}

// avx512LargePanelFloat32 implements a kernel of the matrix multiplication for
// a lhs and rhs packed panels into an intermediate output panel.
func avx512LargePanelFloat32( //alt:f32
	//alt:bf16 func avx512LargePanelBFloat16(
	packedLHS, packedRHS []float32, //alt:f32
	//alt:bf16 packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32, //alt:f32
	//alt:bf16 packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
) {
	defer func() {
		runtime.KeepAlive(packedLHS)
		runtime.KeepAlive(packedRHS)
		runtime.KeepAlive(packedOutput)
	}()
	_ = lhsPanelRows // Not needed.

	// BCE hints
	_ = packedLHS[contractingLen*lhsActiveRows-1]
	_ = packedRHS[contractingLen*rhsActiveCols-1]
	_ = packedOutput[lhsActiveRows*rhsPanelCols-1]

	const (
		kernelRows           = 4  // Must match params.LHSL1BlockRows.
		kernelCols           = 32 // Must match params.RHSL1BlockCols: 2 ZMM registers * 16 elements each.
		numLanes             = 16
		bytesPerInputElement = 4 //alt:f32
		//alt:bf16 bytesPerInputElement = 2
		bytesPerOutputElement = 4 //alt:f32|bf16
	)

	outputBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedOutput)))
	rhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedRHS)))
	lhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedLHS)))

	// Loop 1 (ir): Micro-Kernel Rows (Mr == lhsL1BlockRows)
	for lhsRowIdx := 0; lhsRowIdx < lhsActiveRows; lhsRowIdx += kernelRows {
		idxRHS := 0
		// Loop 2 (jr): Micro-Kernel Columns (Nr == rhsL1BlockCols)
		for rhsColIdx := 0; rhsColIdx < rhsActiveCols; rhsColIdx += kernelCols {
			// Output index calculation (relative to panel)
			outputRowStart := lhsRowIdx
			outputColStart := rhsColIdx
			outputStride := rhsPanelCols

			// ---------------------------------------------------------
			// MICRO KERNEL BODY
			// ---------------------------------------------------------

			// ---------------------------------------------------------
			// 2. Initialize Accumulators (Registers) to 0.0
			// ---------------------------------------------------------
			// We use 4 rows (Mr) worth of registers at a time.
			accum_lhs0_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs0_rhs1 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs1_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs1_rhs1 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs2_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs2_rhs1 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs3_rhs0 := archsimd.BroadcastFloat32x16(0.0)
			accum_lhs3_rhs1 := archsimd.BroadcastFloat32x16(0.0)

			// ------------------------------------------------------------
			// Eliminate bound checks (BCE is not working well enough)
			// ------------------------------------------------------------

			// 1. Calculate the total range the loop will touch
			idxLHS := lhsRowIdx * contractingLen

			// Get the base pointers once
			rhsRowPtr := rhsBasePtr + uintptr(idxRHS*bytesPerInputElement)
			lhsRowPtr := lhsBasePtr + uintptr(idxLHS*bytesPerInputElement)
			idxRHS += contractingLen

			rOffset := uintptr(0)
			lOffset := uintptr(0)

			// Pre-calculate strides **in bytes**
			rhsRegisterStride := uintptr(numLanes * bytesPerInputElement)
			rhsStride := uintptr(kernelCols * bytesPerInputElement)
			lhsStride := uintptr(kernelRows * bytesPerInputElement)

			// ---------------------------------------------------------
			// 3. The K-Loop (Dot Product)
			// ---------------------------------------------------------
			for range contractingLen {
				// Load RHS (Broadcasting/Streaming)
				rhsPtr0 := unsafe.Pointer(rhsRowPtr + rOffset)
				rhsPtr1 := unsafe.Pointer(rhsRowPtr + rOffset + rhsRegisterStride)
				rhsVec0 := archsimd.LoadFloat32x16((*[16]float32)(rhsPtr0))
				rhsVec1 := archsimd.LoadFloat32x16((*[16]float32)(rhsPtr1))
				rOffset += rhsStride

				// Row 0
				lhsVal0 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 0)))
				lhsVec0 := archsimd.BroadcastFloat32x16(lhsVal0)
				accum_lhs0_rhs0 = rhsVec0.MulAdd(lhsVec0, accum_lhs0_rhs0)
				accum_lhs0_rhs1 = rhsVec1.MulAdd(lhsVec0, accum_lhs0_rhs1)

				// Row 1
				lhsVal1 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement)))
				lhsVec1 := archsimd.BroadcastFloat32x16(lhsVal1)
				accum_lhs1_rhs0 = rhsVec0.MulAdd(lhsVec1, accum_lhs1_rhs0)
				accum_lhs1_rhs1 = rhsVec1.MulAdd(lhsVec1, accum_lhs1_rhs1)

				// Row 2
				lhsVal2 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement)))
				lhsVec2 := archsimd.BroadcastFloat32x16(lhsVal2)
				accum_lhs2_rhs0 = rhsVec0.MulAdd(lhsVec2, accum_lhs2_rhs0)
				accum_lhs2_rhs1 = rhsVec1.MulAdd(lhsVec2, accum_lhs2_rhs1)

				// Row 3
				lhsVal3 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement)))
				lhsVec3 := archsimd.BroadcastFloat32x16(lhsVal3)
				accum_lhs3_rhs0 = rhsVec0.MulAdd(lhsVec3, accum_lhs3_rhs0)
				accum_lhs3_rhs1 = rhsVec1.MulAdd(lhsVec3, accum_lhs3_rhs1)

				lOffset += lhsStride
			}

			// ---------------------------------------------------------
			// 4. Write Back to Output
			// ---------------------------------------------------------
			outputIdx0 := uintptr((outputRowStart*outputStride + outputColStart) * bytesPerOutputElement)
			outputIdx1 := outputIdx0 + uintptr(rhsPanelCols*bytesPerOutputElement)
			outputIdx2 := outputIdx0 + uintptr(2*rhsPanelCols*bytesPerOutputElement)
			outputIdx3 := outputIdx0 + uintptr(3*rhsPanelCols*bytesPerOutputElement)
			registerStride := uintptr(numLanes * bytesPerOutputElement) // It should be 512 for AVX512.

			accum_lhs0_rhs0.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx0)))
			accum_lhs0_rhs1.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + registerStride)))
			accum_lhs1_rhs0.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx1)))
			accum_lhs1_rhs1.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + registerStride)))
			accum_lhs2_rhs0.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx2)))
			accum_lhs2_rhs1.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + registerStride)))
			accum_lhs3_rhs0.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx3)))
			accum_lhs3_rhs1.Store((*[16]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + registerStride)))

			// accum_lhs0_rhs0.Store(castToArray16(&packedOutput[outputIdx0]))
			// accum_lhs0_rhs1.Store(castToArray16(&packedOutput[outputIdx0+numLanes]))
			// accum_lhs1_rhs0.Store(castToArray16(&packedOutput[outputIdx1]))
			// accum_lhs1_rhs1.Store(castToArray16(&packedOutput[outputIdx1+numLanes]))
			// accum_lhs2_rhs0.Store(castToArray16(&packedOutput[outputIdx2]))
			// accum_lhs2_rhs1.Store(castToArray16(&packedOutput[outputIdx2+numLanes]))
			// accum_lhs3_rhs0.Store(castToArray16(&packedOutput[outputIdx3]))
			// accum_lhs3_rhs1.Store(castToArray16(&packedOutput[outputIdx3+numLanes]))
		}
	}
}

func castToArray16[T Number](ptr *T) *[16]T {
	return (*[16]T)(unsafe.Pointer(ptr))
}

// applyPackedOutputFloat32 applies the computed packedOutput to the final output.
func avx512ApplyPackedOutputFloat32( //alt:f32
	packedOutput, output []float32,
	isFirstContractingPanel bool,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int, // Global output offsets
	outputRowStride int,
	height, width int, // actual amount of data to copy
) {
	outputRowIdx := lhsRowOffset*outputRowStride + rhsColOffset
	packedRowIdx := 0
	if isFirstContractingPanel {
		// First contracting panel, so we overwrite to the output (as it may not have been zero-initialized).
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			// Vectorized loop
			for ; c+16 <= width; c += 16 {
				packedVal := archsimd.LoadFloat32x16(castToArray16(&packedOutput[packedColIdx]))
				packedVal.Store(castToArray16(&output[outputColIdx]))
				packedColIdx += 16
				outputColIdx += 16
			}

			// Scalar tail
			for i := range width - c {
				output[outputColIdx+i] = packedOutput[packedColIdx+i]
			}

			// Next row.
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}

	} else {
		// Not the first contracting panel, so we need to add to the existing values.
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			// Vectorized loop
			for ; c+16 <= width; c += 16 {
				packedVal := archsimd.LoadFloat32x16(castToArray16(&packedOutput[packedColIdx]))
				outputVal := archsimd.LoadFloat32x16(castToArray16(&output[outputColIdx]))
				outputVal = packedVal.Add(outputVal)
				outputVal.Store(castToArray16(&output[outputColIdx]))
				packedColIdx += 16
				outputColIdx += 16
			}

			// Scalar tail
			for i := range width - c {
				output[outputColIdx+i] += packedOutput[packedColIdx+i]
			}

			// Next row.
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	}
}

// avx512PackRHSFloat32 packs a slice of size [contractingRows, rhsCols] block from RHS into
// the panel reshaped+transposed to [ceil(rhsCols/RHSL1KernelCols), contractingRows, RHSL1KernelCols],
// padding the cols of the last strip with zeros if necessary.
func avx512PackRHSFloat32(src, dst []float32, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols int) {
	dstIdx := 0
	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < rhsCols; stripColIdx += RHSL1KernelCols {
		// How many columns valid in this strip?
		validCols := min(RHSL1KernelCols, rhsCols-stripColIdx)
		srcIdxBase := (srcRowStart * srcStrideCol) + srcColStart + stripColIdx

		if validCols == 32 && RHSL1KernelCols == 32 {
			// Fast path: no zero padding needed, and we can use AVX512
			for range contractingRows {
				// Copy 32 columns using two ZMM registers (16 elements each)
				v0 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdxBase]))
				v1 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdxBase+16]))
				v0.Store(castToArray16(&dst[dstIdx]))
				v1.Store(castToArray16(&dst[dstIdx+16]))
				dstIdx += 32
				srcIdxBase += srcStrideCol
			}
		} else {
			// Iterate over rows (k)
			for range contractingRows {
				// Copy valid columns
				copy(dst[dstIdx:], src[srcIdxBase:srcIdxBase+validCols])
				dstIdx += validCols
				srcIdxBase += srcStrideCol
				// Zero-pad if strip is incomplete (edge of matrix)
				for c := validCols; c < RHSL1KernelCols; c++ {
					dst[dstIdx] = 0
					dstIdx++
				}
			}
		}
	}
}

// packLHS packs a slice of size [lhsRows, contractingCols] block from LHS into
// a [ceil(lhsRows/lhsL1KernelRows), contractingCols, lhsL1KernelRows] "panel"
// (a block of size Mr x Kc) from LHS.
// It rearranges data into horizontal strips of height Mr (lhsL1BlockRows).
//
// How it is called:
//
//	packLHS(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize,
//		lhsPanelHeight, contractingPanelWidth,
//		params.LHSL1KernelRows)
func avx512PackLHSFloat32(src, dst []float32, srcRowStart, srcColStart, srcRowStride, lhsRows, contractingCols, lhsL1KernelRows int) {
	dstIdx := 0
	// Iterate over strips of height mr
	for stripRowIdx := 0; stripRowIdx < lhsRows; stripRowIdx += lhsL1KernelRows {
		validRows := min(lhsL1KernelRows, lhsRows-stripRowIdx)
		srcIdxBase := ((srcRowStart + stripRowIdx) * srcRowStride) + srcColStart

		if validRows == 4 { // Hard-coded to 4 (lhsL1KernelRows)
			c := 0
			// Vectorized loop over cols
			for ; c+16 <= contractingCols; c += 16 {
				v0 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdxBase+c]))
				v1 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdxBase+srcRowStride+c]))
				v2 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdxBase+2*srcRowStride+c]))
				v3 := archsimd.LoadFloat32x16(castToArray16(&src[srcIdxBase+3*srcRowStride+c]))

				var tmp0, tmp1, tmp2, tmp3 [16]float32
				v0.Store(&tmp0)
				v1.Store(&tmp1)
				v2.Store(&tmp2)
				v3.Store(&tmp3)

				for i := range 16 {
					dst[dstIdx] = tmp0[i]
					dst[dstIdx+1] = tmp1[i]
					dst[dstIdx+2] = tmp2[i]
					dst[dstIdx+3] = tmp3[i]
					dstIdx += 4
				}
			}

			// Scalar tail
			for ; c < contractingCols; c++ {
				dst[dstIdx] = src[srcIdxBase+c]
				dst[dstIdx+1] = src[srcIdxBase+srcRowStride+c]
				dst[dstIdx+2] = src[srcIdxBase+2*srcRowStride+c]
				dst[dstIdx+3] = src[srcIdxBase+3*srcRowStride+c]
				dstIdx += 4
			}
		} else {
			// Iterate over columns (contracting size k), we want LHS to be traversed K-first in the kernel
			for col := range contractingCols {
				srcIdx := srcIdxBase + col

				// Copy valid "rows" (they are the last axis in the returned panel)
				for range validRows {
					dst[dstIdx] = src[srcIdx]
					dstIdx++
					srcIdx += srcRowStride
				}

				// Zero-pad
				for r := validRows; r < lhsL1KernelRows; r++ {
					dst[dstIdx] = 0
					dstIdx++
				}
			}
		}
	}
}

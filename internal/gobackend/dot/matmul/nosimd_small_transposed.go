// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !no_unsafe

package matmul

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
)

// smallUnsafeNoSIMDGenericParallelNonTransposed implements a parallelized version of the non-SIMD matrix
// multiplication for the transposed layout. (Shapes lhs=[B,N,K] x rhs=[B,M,K] -> [B,N,M]).
//
// This is the "unsafe" version using pointers. It is faster because it bysteps unnecessary bound-checks.
// Use -tags=no_unsafe to force the safe version (in file nosimd_small_safe.go)
func smallNoSIMDGenericParallelTransposed[I, O gotype.NumericNotComplex]( //alt:generic
	//alt:half func smallNoSIMDHalfPrecisionParallelTransposed[I gotype.HalfPrecision[I], O gotype.NumericNotComplex](
	backend *gobackend.Backend,
	lhs, rhs []I,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []O, matricesPerTask int) {

	// Crate work that needs doing in a buffered channel.
	type chunkData struct {
		batchIdx, batchCount int
	}
	numChunks := (batchSize + matricesPerTask - 1) / matricesPerTask
	work := make(chan chunkData, numChunks)
	for batchIdx := 0; batchIdx < batchSize; batchIdx += matricesPerTask {
		batchCount := min(matricesPerTask, batchSize-batchIdx)
		work <- chunkData{batchIdx, batchCount}
	}
	close(work)

	// Execute the work in as many workers as available.
	backend.Workers.Saturate(func() {
		for chunk := range work {
			smallNoSIMDGenericTransposed( //alt:generic
				//alt:half smallNoSIMDHalfPrecisionTransposed(
				lhs, rhs,
				chunk.batchIdx, chunk.batchCount, lhsCrossSize, rhsCrossSize, contractingSize,
				output)
		}
	})
}

// smallNoSIMDGenericTransposed implements a non-SIMD matrix multiplication for the transposed layout.
//
// lhs:    shape [batchSize, lhsCrossSize, contractingSize].
// rhs:    shape [batchSize, rhsCrossSize, contractingSize]. (note: transposed)
// output: shape [batchSize, lhsCrossSize, rhsCrossSize].
//
// It is used for small inputs, where packing the data is not worth the cost.
func smallNoSIMDGenericTransposed[I, O gotype.NumericNotComplex]( //alt:generic
	//alt:half func smallNoSIMDHalfPrecisionTransposed[I gotype.HalfPrecision[I], O gotype.NumericNotComplex](
	lhs, rhs []I,
	batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []O) {
	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize

	// Bounds check hint for the compiler: the hope is that the compile won't need to
	// insert bounds checks inside the loops below.
	//
	// This should never happen.
	if len(lhs) < lhsStride*batchCount || len(rhs) < rhsStride*batchCount || len(output) < outputStride*batchCount {
		panic("out of bounds")
	}

	if batchCount == 0 || lhsCrossSize == 0 || rhsCrossSize == 0 || contractingSize == 0 {
		return
	}

	var iZero I
	iSize := unsafe.Sizeof(iZero)
	var oZero O
	oSize := unsafe.Sizeof(oZero)

	lhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(lhs)))
	rhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(rhs)))
	outputPtr := uintptr(unsafe.Pointer(unsafe.SliceData(output)))

	lhsByteStride := uintptr(lhsStride) * iSize
	rhsByteStride := uintptr(rhsStride) * iSize
	outputByteStride := uintptr(outputStride) * oSize

	lhsBase := lhsPtr + uintptr(batchStart)*lhsByteStride
	rhsBase := rhsPtr + uintptr(batchStart)*rhsByteStride
	outputBase := outputPtr + uintptr(batchStart)*outputByteStride

	for range batchCount {
		for row := range lhsCrossSize {
			lRowBase := lhsBase + uintptr(row*contractingSize)*iSize

			for col := range rhsCrossSize {
				rColBase := rhsBase + uintptr(col*contractingSize)*iSize
				var acc O

				lIdx := lRowBase
				rIdx := rColBase

				var contractingIdx int
				for ; contractingIdx+3 < contractingSize; contractingIdx += 4 {
					l0 := *(*I)(unsafe.Pointer(lIdx))           //alt:generic
					l1 := *(*I)(unsafe.Pointer(lIdx + iSize))   //alt:generic
					l2 := *(*I)(unsafe.Pointer(lIdx + 2*iSize)) //alt:generic
					l3 := *(*I)(unsafe.Pointer(lIdx + 3*iSize)) //alt:generic
					r0 := *(*I)(unsafe.Pointer(rIdx))           //alt:generic
					r1 := *(*I)(unsafe.Pointer(rIdx + iSize))   //alt:generic
					r2 := *(*I)(unsafe.Pointer(rIdx + 2*iSize)) //alt:generic
					r3 := *(*I)(unsafe.Pointer(rIdx + 3*iSize)) //alt:generic

					//alt:half l0 := (*(*I)(unsafe.Pointer(lIdx))).Float32()
					//alt:half l1 := (*(*I)(unsafe.Pointer(lIdx + iSize))).Float32()
					//alt:half l2 := (*(*I)(unsafe.Pointer(lIdx + 2*iSize))).Float32()
					//alt:half l3 := (*(*I)(unsafe.Pointer(lIdx + 3*iSize))).Float32()
					//alt:half r0 := (*(*I)(unsafe.Pointer(rIdx))).Float32()
					//alt:half r1 := (*(*I)(unsafe.Pointer(rIdx + iSize))).Float32()
					//alt:half r2 := (*(*I)(unsafe.Pointer(rIdx + 2*iSize))).Float32()
					//alt:half r3 := (*(*I)(unsafe.Pointer(rIdx + 3*iSize))).Float32()

					v0 := O(l0 * r0)
					v1 := O(l1 * r1)
					v2 := O(l2 * r2)
					v3 := O(l3 * r3)

					acc += v0 + v1 + v2 + v3
					lIdx += 4 * iSize
					rIdx += 4 * iSize
				}
				for ; contractingIdx < contractingSize; contractingIdx++ {
					l0 := *(*I)(unsafe.Pointer(lIdx)) //alt:generic
					r0 := *(*I)(unsafe.Pointer(rIdx)) //alt:generic
					//alt:half l0 := (*(*I)(unsafe.Pointer(lIdx))).Float32()
					//alt:half r0 := (*(*I)(unsafe.Pointer(rIdx))).Float32()
					acc += O(l0 * r0)
					lIdx += iSize
					rIdx += iSize
				}

				outputIdx := outputBase + uintptr(row*rhsCrossSize+col)*oSize
				*(*O)(unsafe.Pointer(outputIdx)) = acc
			}
		}

		lhsBase += lhsByteStride
		rhsBase += rhsByteStride
		outputBase += outputByteStride
	}
}

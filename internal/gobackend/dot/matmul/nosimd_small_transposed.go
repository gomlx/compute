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

	// Fast path for GEMV (vector output: rhsCrossSize == 1)
	if rhsCrossSize == 1 {
		for range batchCount {
			row := 0
			for ; row+3 < lhsCrossSize; row += 4 {
				lRow0Base := lhsBase + uintptr(row*contractingSize)*iSize
				lRow1Base := lRow0Base + uintptr(contractingSize)*iSize
				lRow2Base := lRow1Base + uintptr(contractingSize)*iSize
				lRow3Base := lRow2Base + uintptr(contractingSize)*iSize

				var c0, c1, c2, c3 O
				l0Idx := lRow0Base
				l1Idx := lRow1Base
				l2Idx := lRow2Base
				l3Idx := lRow3Base
				rIdx := rhsBase

				var k int
				for ; k+1 < contractingSize; k += 2 {
					la0 := *(*I)(unsafe.Pointer(l0Idx))         //alt:generic
					la1 := *(*I)(unsafe.Pointer(l0Idx + iSize)) //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx))         //alt:generic
					lb1 := *(*I)(unsafe.Pointer(l1Idx + iSize)) //alt:generic
					lc0 := *(*I)(unsafe.Pointer(l2Idx))         //alt:generic
					lc1 := *(*I)(unsafe.Pointer(l2Idx + iSize)) //alt:generic
					ld0 := *(*I)(unsafe.Pointer(l3Idx))         //alt:generic
					ld1 := *(*I)(unsafe.Pointer(l3Idx + iSize)) //alt:generic

					rk0 := *(*I)(unsafe.Pointer(rIdx))         //alt:generic
					rk1 := *(*I)(unsafe.Pointer(rIdx + iSize)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Pointer(l0Idx + iSize))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Pointer(l1Idx + iSize))).Float32()
					//alt:half lc0 := (*(*I)(unsafe.Pointer(l2Idx))).Float32()
					//alt:half lc1 := (*(*I)(unsafe.Pointer(l2Idx + iSize))).Float32()
					//alt:half ld0 := (*(*I)(unsafe.Pointer(l3Idx))).Float32()
					//alt:half ld1 := (*(*I)(unsafe.Pointer(l3Idx + iSize))).Float32()

					//alt:half rk0 := (*(*I)(unsafe.Pointer(rIdx))).Float32()
					//alt:half rk1 := (*(*I)(unsafe.Pointer(rIdx + iSize))).Float32()

					c0 += O(la0*rk0) + O(la1*rk1)
					c1 += O(lb0*rk0) + O(lb1*rk1)
					c2 += O(lc0*rk0) + O(lc1*rk1)
					c3 += O(ld0*rk0) + O(ld1*rk1)

					l0Idx += 2 * iSize
					l1Idx += 2 * iSize
					l2Idx += 2 * iSize
					l3Idx += 2 * iSize
					rIdx += 2 * iSize
				}
				for ; k < contractingSize; k++ {
					la0 := *(*I)(unsafe.Pointer(l0Idx)) //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx)) //alt:generic
					lc0 := *(*I)(unsafe.Pointer(l2Idx)) //alt:generic
					ld0 := *(*I)(unsafe.Pointer(l3Idx)) //alt:generic
					rk0 := *(*I)(unsafe.Pointer(rIdx))  //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half lc0 := (*(*I)(unsafe.Pointer(l2Idx))).Float32()
					//alt:half ld0 := (*(*I)(unsafe.Pointer(l3Idx))).Float32()
					//alt:half rk0 := (*(*I)(unsafe.Pointer(rIdx))).Float32()

					c0 += O(la0 * rk0)
					c1 += O(lb0 * rk0)
					c2 += O(lc0 * rk0)
					c3 += O(ld0 * rk0)

					l0Idx += iSize
					l1Idx += iSize
					l2Idx += iSize
					l3Idx += iSize
					rIdx += iSize
				}

				out0 := outputBase + uintptr(row)*oSize
				*(*O)(unsafe.Pointer(out0)) = c0
				*(*O)(unsafe.Pointer(out0 + oSize)) = c1
				*(*O)(unsafe.Pointer(out0 + 2*oSize)) = c2
				*(*O)(unsafe.Pointer(out0 + 3*oSize)) = c3
			}
			for ; row < lhsCrossSize; row++ {
				lIdx := lhsBase + uintptr(row*contractingSize)*iSize
				rIdx := rhsBase
				var c O
				for range contractingSize {
					la := *(*I)(unsafe.Pointer(lIdx)) //alt:generic
					rk := *(*I)(unsafe.Pointer(rIdx)) //alt:generic
					//alt:half la := (*(*I)(unsafe.Pointer(lIdx))).Float32()
					//alt:half rk := (*(*I)(unsafe.Pointer(rIdx))).Float32()
					c += O(la * rk)
					lIdx += iSize
					rIdx += iSize
				}
				*(*O)(unsafe.Pointer(outputBase + uintptr(row)*oSize)) = c
			}
			lhsBase += lhsByteStride
			rhsBase += rhsByteStride
			outputBase += outputByteStride
		}
		return
	}

	for range batchCount {
		row := 0
		for ; row+1 < lhsCrossSize; row += 2 {
			lRow0Base := lhsBase + uintptr(row*contractingSize)*iSize
			lRow1Base := lRow0Base + uintptr(contractingSize)*iSize

			col := 0
			for ; col+3 < rhsCrossSize; col += 4 {
				rCol0Base := rhsBase + uintptr(col*contractingSize)*iSize
				rCol1Base := rCol0Base + uintptr(contractingSize)*iSize
				rCol2Base := rCol1Base + uintptr(contractingSize)*iSize
				rCol3Base := rCol2Base + uintptr(contractingSize)*iSize

				var c00, c01, c02, c03 O
				var c10, c11, c12, c13 O

				l0Idx := lRow0Base
				l1Idx := lRow1Base
				r0Idx := rCol0Base
				r1Idx := rCol1Base
				r2Idx := rCol2Base
				r3Idx := rCol3Base

				var k int
				for ; k+1 < contractingSize; k += 2 {
					la0 := *(*I)(unsafe.Pointer(l0Idx))         //alt:generic
					la1 := *(*I)(unsafe.Pointer(l0Idx + iSize)) //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx))         //alt:generic
					lb1 := *(*I)(unsafe.Pointer(l1Idx + iSize)) //alt:generic

					r0_0 := *(*I)(unsafe.Pointer(r0Idx))         //alt:generic
					r0_1 := *(*I)(unsafe.Pointer(r0Idx + iSize)) //alt:generic
					r1_0 := *(*I)(unsafe.Pointer(r1Idx))         //alt:generic
					r1_1 := *(*I)(unsafe.Pointer(r1Idx + iSize)) //alt:generic
					r2_0 := *(*I)(unsafe.Pointer(r2Idx))         //alt:generic
					r2_1 := *(*I)(unsafe.Pointer(r2Idx + iSize)) //alt:generic
					r3_0 := *(*I)(unsafe.Pointer(r3Idx))         //alt:generic
					r3_1 := *(*I)(unsafe.Pointer(r3Idx + iSize)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Pointer(l0Idx + iSize))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Pointer(l1Idx + iSize))).Float32()

					//alt:half r0_0 := (*(*I)(unsafe.Pointer(r0Idx))).Float32()
					//alt:half r0_1 := (*(*I)(unsafe.Pointer(r0Idx + iSize))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Pointer(r1Idx))).Float32()
					//alt:half r1_1 := (*(*I)(unsafe.Pointer(r1Idx + iSize))).Float32()
					//alt:half r2_0 := (*(*I)(unsafe.Pointer(r2Idx))).Float32()
					//alt:half r2_1 := (*(*I)(unsafe.Pointer(r2Idx + iSize))).Float32()
					//alt:half r3_0 := (*(*I)(unsafe.Pointer(r3Idx))).Float32()
					//alt:half r3_1 := (*(*I)(unsafe.Pointer(r3Idx + iSize))).Float32()

					c00 += O(la0*r0_0) + O(la1*r0_1)
					c01 += O(la0*r1_0) + O(la1*r1_1)
					c02 += O(la0*r2_0) + O(la1*r2_1)
					c03 += O(la0*r3_0) + O(la1*r3_1)

					c10 += O(lb0*r0_0) + O(lb1*r0_1)
					c11 += O(lb0*r1_0) + O(lb1*r1_1)
					c12 += O(lb0*r2_0) + O(lb1*r2_1)
					c13 += O(lb0*r3_0) + O(lb1*r3_1)

					l0Idx += 2 * iSize
					l1Idx += 2 * iSize
					r0Idx += 2 * iSize
					r1Idx += 2 * iSize
					r2Idx += 2 * iSize
					r3Idx += 2 * iSize
				}
				for ; k < contractingSize; k++ {
					la0 := *(*I)(unsafe.Pointer(l0Idx))  //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx))  //alt:generic
					r0_0 := *(*I)(unsafe.Pointer(r0Idx)) //alt:generic
					r1_0 := *(*I)(unsafe.Pointer(r1Idx)) //alt:generic
					r2_0 := *(*I)(unsafe.Pointer(r2Idx)) //alt:generic
					r3_0 := *(*I)(unsafe.Pointer(r3Idx)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half r0_0 := (*(*I)(unsafe.Pointer(r0Idx))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Pointer(r1Idx))).Float32()
					//alt:half r2_0 := (*(*I)(unsafe.Pointer(r2Idx))).Float32()
					//alt:half r3_0 := (*(*I)(unsafe.Pointer(r3Idx))).Float32()

					c00 += O(la0 * r0_0)
					c01 += O(la0 * r1_0)
					c02 += O(la0 * r2_0)
					c03 += O(la0 * r3_0)
					c10 += O(lb0 * r0_0)
					c11 += O(lb0 * r1_0)
					c12 += O(lb0 * r2_0)
					c13 += O(lb0 * r3_0)

					l0Idx += iSize
					l1Idx += iSize
					r0Idx += iSize
					r1Idx += iSize
					r2Idx += iSize
					r3Idx += iSize
				}

				out0 := outputBase + uintptr(row*rhsCrossSize+col)*oSize
				out1 := outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize
				*(*O)(unsafe.Pointer(out0)) = c00
				*(*O)(unsafe.Pointer(out0 + oSize)) = c01
				*(*O)(unsafe.Pointer(out0 + 2*oSize)) = c02
				*(*O)(unsafe.Pointer(out0 + 3*oSize)) = c03
				*(*O)(unsafe.Pointer(out1)) = c10
				*(*O)(unsafe.Pointer(out1 + oSize)) = c11
				*(*O)(unsafe.Pointer(out1 + 2*oSize)) = c12
				*(*O)(unsafe.Pointer(out1 + 3*oSize)) = c13
			}

			// Col pair (2 rows x 2 cols)
			for ; col+1 < rhsCrossSize; col += 2 {
				rCol0Base := rhsBase + uintptr(col*contractingSize)*iSize
				rCol1Base := rCol0Base + uintptr(contractingSize)*iSize

				var c00, c01, c10, c11 O
				l0Idx := lRow0Base
				l1Idx := lRow1Base
				r0Idx := rCol0Base
				r1Idx := rCol1Base

				var k int
				for ; k+1 < contractingSize; k += 2 {
					la0 := *(*I)(unsafe.Pointer(l0Idx))         //alt:generic
					la1 := *(*I)(unsafe.Pointer(l0Idx + iSize)) //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx))         //alt:generic
					lb1 := *(*I)(unsafe.Pointer(l1Idx + iSize)) //alt:generic

					r0_0 := *(*I)(unsafe.Pointer(r0Idx))         //alt:generic
					r0_1 := *(*I)(unsafe.Pointer(r0Idx + iSize)) //alt:generic
					r1_0 := *(*I)(unsafe.Pointer(r1Idx))         //alt:generic
					r1_1 := *(*I)(unsafe.Pointer(r1Idx + iSize)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Pointer(l0Idx + iSize))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Pointer(l1Idx + iSize))).Float32()

					//alt:half r0_0 := (*(*I)(unsafe.Pointer(r0Idx))).Float32()
					//alt:half r0_1 := (*(*I)(unsafe.Pointer(r0Idx + iSize))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Pointer(r1Idx))).Float32()
					//alt:half r1_1 := (*(*I)(unsafe.Pointer(r1Idx + iSize))).Float32()

					c00 += O(la0*r0_0) + O(la1*r0_1)
					c01 += O(la0*r1_0) + O(la1*r1_1)
					c10 += O(lb0*r0_0) + O(lb1*r0_1)
					c11 += O(lb0*r1_0) + O(lb1*r1_1)

					l0Idx += 2 * iSize
					l1Idx += 2 * iSize
					r0Idx += 2 * iSize
					r1Idx += 2 * iSize
				}
				for ; k < contractingSize; k++ {
					la0 := *(*I)(unsafe.Pointer(l0Idx))  //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx))  //alt:generic
					r0_0 := *(*I)(unsafe.Pointer(r0Idx)) //alt:generic
					r1_0 := *(*I)(unsafe.Pointer(r1Idx)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half r0_0 := (*(*I)(unsafe.Pointer(r0Idx))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Pointer(r1Idx))).Float32()

					c00 += O(la0 * r0_0)
					c01 += O(la0 * r1_0)
					c10 += O(lb0 * r0_0)
					c11 += O(lb0 * r1_0)

					l0Idx += iSize
					l1Idx += iSize
					r0Idx += iSize
					r1Idx += iSize
				}

				out0 := outputBase + uintptr(row*rhsCrossSize+col)*oSize
				out1 := outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize
				*(*O)(unsafe.Pointer(out0)) = c00
				*(*O)(unsafe.Pointer(out0 + oSize)) = c01
				*(*O)(unsafe.Pointer(out1)) = c10
				*(*O)(unsafe.Pointer(out1 + oSize)) = c11
			}

			// Single col fringe (2 rows x 1 col)
			for ; col < rhsCrossSize; col++ {
				rColBase := rhsBase + uintptr(col*contractingSize)*iSize
				var c0, c1 O
				l0Idx := lRow0Base
				l1Idx := lRow1Base
				rIdx := rColBase
				var k int
				for ; k+1 < contractingSize; k += 2 {
					la0 := *(*I)(unsafe.Pointer(l0Idx))         //alt:generic
					la1 := *(*I)(unsafe.Pointer(l0Idx + iSize)) //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx))         //alt:generic
					lb1 := *(*I)(unsafe.Pointer(l1Idx + iSize)) //alt:generic
					rk0 := *(*I)(unsafe.Pointer(rIdx))         //alt:generic
					rk1 := *(*I)(unsafe.Pointer(rIdx + iSize)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Pointer(l0Idx + iSize))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Pointer(l1Idx + iSize))).Float32()
					//alt:half rk0 := (*(*I)(unsafe.Pointer(rIdx))).Float32()
					//alt:half rk1 := (*(*I)(unsafe.Pointer(rIdx + iSize))).Float32()

					c0 += O(la0*rk0) + O(la1*rk1)
					c1 += O(lb0*rk0) + O(lb1*rk1)

					l0Idx += 2 * iSize
					l1Idx += 2 * iSize
					rIdx += 2 * iSize
				}
				for ; k < contractingSize; k++ {
					la0 := *(*I)(unsafe.Pointer(l0Idx)) //alt:generic
					lb0 := *(*I)(unsafe.Pointer(l1Idx)) //alt:generic
					rk0 := *(*I)(unsafe.Pointer(rIdx))  //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(l0Idx))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Pointer(l1Idx))).Float32()
					//alt:half rk0 := (*(*I)(unsafe.Pointer(rIdx))).Float32()

					c0 += O(la0 * rk0)
					c1 += O(lb0 * rk0)

					l0Idx += iSize
					l1Idx += iSize
					rIdx += iSize
				}
				*(*O)(unsafe.Pointer(outputBase + uintptr(row*rhsCrossSize+col)*oSize)) = c0
				*(*O)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize)) = c1
			}
		}

		// Row fringe (1 row x cols)
		for ; row < lhsCrossSize; row++ {
			lRowBase := lhsBase + uintptr(row*contractingSize)*iSize
			for col := 0; col < rhsCrossSize; col++ {
				rColBase := rhsBase + uintptr(col*contractingSize)*iSize
				var c O
				lIdx := lRowBase
				rIdx := rColBase
				var k int
				for ; k+3 < contractingSize; k += 4 {
					la0 := *(*I)(unsafe.Pointer(lIdx))           //alt:generic
					la1 := *(*I)(unsafe.Pointer(lIdx + iSize))   //alt:generic
					la2 := *(*I)(unsafe.Pointer(lIdx + 2*iSize)) //alt:generic
					la3 := *(*I)(unsafe.Pointer(lIdx + 3*iSize)) //alt:generic
					rk0 := *(*I)(unsafe.Pointer(rIdx))           //alt:generic
					rk1 := *(*I)(unsafe.Pointer(rIdx + iSize))   //alt:generic
					rk2 := *(*I)(unsafe.Pointer(rIdx + 2*iSize)) //alt:generic
					rk3 := *(*I)(unsafe.Pointer(rIdx + 3*iSize)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Pointer(lIdx))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Pointer(lIdx + iSize))).Float32()
					//alt:half la2 := (*(*I)(unsafe.Pointer(lIdx + 2*iSize))).Float32()
					//alt:half la3 := (*(*I)(unsafe.Pointer(lIdx + 3*iSize))).Float32()
					//alt:half rk0 := (*(*I)(unsafe.Pointer(rIdx))).Float32()
					//alt:half rk1 := (*(*I)(unsafe.Pointer(rIdx + iSize))).Float32()
					//alt:half rk2 := (*(*I)(unsafe.Pointer(rIdx + 2*iSize))).Float32()
					//alt:half rk3 := (*(*I)(unsafe.Pointer(rIdx + 3*iSize))).Float32()

					c += O(la0*rk0) + O(la1*rk1) + O(la2*rk2) + O(la3*rk3)
					lIdx += 4 * iSize
					rIdx += 4 * iSize
				}
				for ; k < contractingSize; k++ {
					la := *(*I)(unsafe.Pointer(lIdx)) //alt:generic
					rk := *(*I)(unsafe.Pointer(rIdx)) //alt:generic
					//alt:half la := (*(*I)(unsafe.Pointer(lIdx))).Float32()
					//alt:half rk := (*(*I)(unsafe.Pointer(rIdx))).Float32()
					c += O(la * rk)
					lIdx += iSize
					rIdx += iSize
				}
				*(*O)(unsafe.Pointer(outputBase + uintptr(row*rhsCrossSize+col)*oSize)) = c
			}
		}

		lhsBase += lhsByteStride
		rhsBase += rhsByteStride
		outputBase += outputByteStride
	}
}

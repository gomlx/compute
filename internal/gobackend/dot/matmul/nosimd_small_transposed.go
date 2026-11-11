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

	backend.Scheduler.ParallelizeChunks(batchSize, matricesPerTask, func(start, end int) {
		smallNoSIMDGenericTransposed( //alt:generic
			//alt:half smallNoSIMDHalfPrecisionTransposed(
			lhs, rhs,
			start, end-start, lhsCrossSize, rhsCrossSize, contractingSize,
			output)
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

	lhsPtr := unsafe.Pointer(unsafe.SliceData(lhs))
	rhsPtr := unsafe.Pointer(unsafe.SliceData(rhs))
	outputPtr := unsafe.Pointer(unsafe.SliceData(output))

	lhsByteStride := uintptr(lhsStride) * iSize
	rhsByteStride := uintptr(rhsStride) * iSize
	outputByteStride := uintptr(outputStride) * oSize
	rowByteStride := uintptr(contractingSize) * iSize

	// Fast path for GEMV (vector output: rhsCrossSize == 1)
	if rhsCrossSize == 1 {
		for b := range batchCount {
			bIdx := uintptr(batchStart + b)
			lhsBase := unsafe.Add(lhsPtr, bIdx*lhsByteStride)
			rhsBase := unsafe.Add(rhsPtr, bIdx*rhsByteStride)
			outputBase := unsafe.Add(outputPtr, bIdx*outputByteStride)

			row := 0
			for ; row+3 < lhsCrossSize; row += 4 {
				lRow0Base := unsafe.Add(lhsBase, uintptr(row)*rowByteStride)
				lRow1Base := unsafe.Add(lRow0Base, rowByteStride)
				lRow2Base := unsafe.Add(lRow1Base, rowByteStride)
				lRow3Base := unsafe.Add(lRow2Base, rowByteStride)

				var c0, c1, c2, c3 O

				var k int
				for ; k+1 < contractingSize; k += 2 {
					kOff0 := uintptr(k) * iSize
					kOff1 := kOff0 + iSize

					la0 := *(*I)(unsafe.Add(lRow0Base, kOff0)) //alt:generic
					la1 := *(*I)(unsafe.Add(lRow0Base, kOff1)) //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff0)) //alt:generic
					lb1 := *(*I)(unsafe.Add(lRow1Base, kOff1)) //alt:generic
					lc0 := *(*I)(unsafe.Add(lRow2Base, kOff0)) //alt:generic
					lc1 := *(*I)(unsafe.Add(lRow2Base, kOff1)) //alt:generic
					ld0 := *(*I)(unsafe.Add(lRow3Base, kOff0)) //alt:generic
					ld1 := *(*I)(unsafe.Add(lRow3Base, kOff1)) //alt:generic

					rk0 := *(*I)(unsafe.Add(rhsBase, kOff0)) //alt:generic
					rk1 := *(*I)(unsafe.Add(rhsBase, kOff1)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff0))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Add(lRow0Base, kOff1))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff0))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Add(lRow1Base, kOff1))).Float32()
					//alt:half lc0 := (*(*I)(unsafe.Add(lRow2Base, kOff0))).Float32()
					//alt:half lc1 := (*(*I)(unsafe.Add(lRow2Base, kOff1))).Float32()
					//alt:half ld0 := (*(*I)(unsafe.Add(lRow3Base, kOff0))).Float32()
					//alt:half ld1 := (*(*I)(unsafe.Add(lRow3Base, kOff1))).Float32()

					//alt:half rk0 := (*(*I)(unsafe.Add(rhsBase, kOff0))).Float32()
					//alt:half rk1 := (*(*I)(unsafe.Add(rhsBase, kOff1))).Float32()

					c0 += O(la0*rk0) + O(la1*rk1)
					c1 += O(lb0*rk0) + O(lb1*rk1)
					c2 += O(lc0*rk0) + O(lc1*rk1)
					c3 += O(ld0*rk0) + O(ld1*rk1)
				}
				for ; k < contractingSize; k++ {
					kOff := uintptr(k) * iSize
					la0 := *(*I)(unsafe.Add(lRow0Base, kOff)) //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff)) //alt:generic
					lc0 := *(*I)(unsafe.Add(lRow2Base, kOff)) //alt:generic
					ld0 := *(*I)(unsafe.Add(lRow3Base, kOff)) //alt:generic
					rk0 := *(*I)(unsafe.Add(rhsBase, kOff))   //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff))).Float32()
					//alt:half lc0 := (*(*I)(unsafe.Add(lRow2Base, kOff))).Float32()
					//alt:half ld0 := (*(*I)(unsafe.Add(lRow3Base, kOff))).Float32()
					//alt:half rk0 := (*(*I)(unsafe.Add(rhsBase, kOff))).Float32()

					c0 += O(la0 * rk0)
					c1 += O(lb0 * rk0)
					c2 += O(lc0 * rk0)
					c3 += O(ld0 * rk0)
				}

				out0 := unsafe.Add(outputBase, uintptr(row)*oSize)
				*(*O)(out0) = c0
				*(*O)(unsafe.Add(out0, oSize)) = c1
				*(*O)(unsafe.Add(out0, 2*oSize)) = c2
				*(*O)(unsafe.Add(out0, 3*oSize)) = c3
			}
			for ; row < lhsCrossSize; row++ {
				lRowBase := unsafe.Add(lhsBase, uintptr(row)*rowByteStride)
				var c O
				for k := range contractingSize {
					kOff := uintptr(k) * iSize
					la := *(*I)(unsafe.Add(lRowBase, kOff)) //alt:generic
					rk := *(*I)(unsafe.Add(rhsBase, kOff))  //alt:generic
					//alt:half la := (*(*I)(unsafe.Add(lRowBase, kOff))).Float32()
					//alt:half rk := (*(*I)(unsafe.Add(rhsBase, kOff))).Float32()
					c += O(la * rk)
				}
				*(*O)(unsafe.Add(outputBase, uintptr(row)*oSize)) = c
			}
		}
		return
	}

	for b := range batchCount {
		bIdx := uintptr(batchStart + b)
		lhsBase := unsafe.Add(lhsPtr, bIdx*lhsByteStride)
		rhsBase := unsafe.Add(rhsPtr, bIdx*rhsByteStride)
		outputBase := unsafe.Add(outputPtr, bIdx*outputByteStride)

		row := 0
		for ; row+1 < lhsCrossSize; row += 2 {
			lRow0Base := unsafe.Add(lhsBase, uintptr(row)*rowByteStride)
			lRow1Base := unsafe.Add(lRow0Base, rowByteStride)

			col := 0
			for ; col+3 < rhsCrossSize; col += 4 {
				rCol0Base := unsafe.Add(rhsBase, uintptr(col)*rowByteStride)
				rCol1Base := unsafe.Add(rCol0Base, rowByteStride)
				rCol2Base := unsafe.Add(rCol1Base, rowByteStride)
				rCol3Base := unsafe.Add(rCol2Base, rowByteStride)

				var c00, c01, c02, c03 O
				var c10, c11, c12, c13 O

				var k int
				for ; k+1 < contractingSize; k += 2 {
					kOff0 := uintptr(k) * iSize
					kOff1 := kOff0 + iSize

					la0 := *(*I)(unsafe.Add(lRow0Base, kOff0)) //alt:generic
					la1 := *(*I)(unsafe.Add(lRow0Base, kOff1)) //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff0)) //alt:generic
					lb1 := *(*I)(unsafe.Add(lRow1Base, kOff1)) //alt:generic

					r0_0 := *(*I)(unsafe.Add(rCol0Base, kOff0)) //alt:generic
					r0_1 := *(*I)(unsafe.Add(rCol0Base, kOff1)) //alt:generic
					r1_0 := *(*I)(unsafe.Add(rCol1Base, kOff0)) //alt:generic
					r1_1 := *(*I)(unsafe.Add(rCol1Base, kOff1)) //alt:generic
					r2_0 := *(*I)(unsafe.Add(rCol2Base, kOff0)) //alt:generic
					r2_1 := *(*I)(unsafe.Add(rCol2Base, kOff1)) //alt:generic
					r3_0 := *(*I)(unsafe.Add(rCol3Base, kOff0)) //alt:generic
					r3_1 := *(*I)(unsafe.Add(rCol3Base, kOff1)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff0))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Add(lRow0Base, kOff1))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff0))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Add(lRow1Base, kOff1))).Float32()

					//alt:half r0_0 := (*(*I)(unsafe.Add(rCol0Base, kOff0))).Float32()
					//alt:half r0_1 := (*(*I)(unsafe.Add(rCol0Base, kOff1))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Add(rCol1Base, kOff0))).Float32()
					//alt:half r1_1 := (*(*I)(unsafe.Add(rCol1Base, kOff1))).Float32()
					//alt:half r2_0 := (*(*I)(unsafe.Add(rCol2Base, kOff0))).Float32()
					//alt:half r2_1 := (*(*I)(unsafe.Add(rCol2Base, kOff1))).Float32()
					//alt:half r3_0 := (*(*I)(unsafe.Add(rCol3Base, kOff0))).Float32()
					//alt:half r3_1 := (*(*I)(unsafe.Add(rCol3Base, kOff1))).Float32()

					c00 += O(la0*r0_0) + O(la1*r0_1)
					c01 += O(la0*r1_0) + O(la1*r1_1)
					c02 += O(la0*r2_0) + O(la1*r2_1)
					c03 += O(la0*r3_0) + O(la1*r3_1)

					c10 += O(lb0*r0_0) + O(lb1*r0_1)
					c11 += O(lb0*r1_0) + O(lb1*r1_1)
					c12 += O(lb0*r2_0) + O(lb1*r2_1)
					c13 += O(lb0*r3_0) + O(lb1*r3_1)
				}
				for ; k < contractingSize; k++ {
					kOff := uintptr(k) * iSize
					la0 := *(*I)(unsafe.Add(lRow0Base, kOff))  //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff))  //alt:generic
					r0_0 := *(*I)(unsafe.Add(rCol0Base, kOff)) //alt:generic
					r1_0 := *(*I)(unsafe.Add(rCol1Base, kOff)) //alt:generic
					r2_0 := *(*I)(unsafe.Add(rCol2Base, kOff)) //alt:generic
					r3_0 := *(*I)(unsafe.Add(rCol3Base, kOff)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff))).Float32()
					//alt:half r0_0 := (*(*I)(unsafe.Add(rCol0Base, kOff))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Add(rCol1Base, kOff))).Float32()
					//alt:half r2_0 := (*(*I)(unsafe.Add(rCol2Base, kOff))).Float32()
					//alt:half r3_0 := (*(*I)(unsafe.Add(rCol3Base, kOff))).Float32()

					c00 += O(la0 * r0_0)
					c01 += O(la0 * r1_0)
					c02 += O(la0 * r2_0)
					c03 += O(la0 * r3_0)
					c10 += O(lb0 * r0_0)
					c11 += O(lb0 * r1_0)
					c12 += O(lb0 * r2_0)
					c13 += O(lb0 * r3_0)
				}

				out0 := unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*oSize)
				out1 := unsafe.Add(outputBase, uintptr((row+1)*rhsCrossSize+col)*oSize)
				*(*O)(out0) = c00
				*(*O)(unsafe.Add(out0, oSize)) = c01
				*(*O)(unsafe.Add(out0, 2*oSize)) = c02
				*(*O)(unsafe.Add(out0, 3*oSize)) = c03
				*(*O)(out1) = c10
				*(*O)(unsafe.Add(out1, oSize)) = c11
				*(*O)(unsafe.Add(out1, 2*oSize)) = c12
				*(*O)(unsafe.Add(out1, 3*oSize)) = c13
			}

			// Col pair (2 rows x 2 cols)
			for ; col+1 < rhsCrossSize; col += 2 {
				rCol0Base := unsafe.Add(rhsBase, uintptr(col)*rowByteStride)
				rCol1Base := unsafe.Add(rCol0Base, rowByteStride)

				var c00, c01, c10, c11 O

				var k int
				for ; k+1 < contractingSize; k += 2 {
					kOff0 := uintptr(k) * iSize
					kOff1 := kOff0 + iSize

					la0 := *(*I)(unsafe.Add(lRow0Base, kOff0)) //alt:generic
					la1 := *(*I)(unsafe.Add(lRow0Base, kOff1)) //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff0)) //alt:generic
					lb1 := *(*I)(unsafe.Add(lRow1Base, kOff1)) //alt:generic

					r0_0 := *(*I)(unsafe.Add(rCol0Base, kOff0)) //alt:generic
					r0_1 := *(*I)(unsafe.Add(rCol0Base, kOff1)) //alt:generic
					r1_0 := *(*I)(unsafe.Add(rCol1Base, kOff0)) //alt:generic
					r1_1 := *(*I)(unsafe.Add(rCol1Base, kOff1)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff0))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Add(lRow0Base, kOff1))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff0))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Add(lRow1Base, kOff1))).Float32()

					//alt:half r0_0 := (*(*I)(unsafe.Add(rCol0Base, kOff0))).Float32()
					//alt:half r0_1 := (*(*I)(unsafe.Add(rCol0Base, kOff1))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Add(rCol1Base, kOff0))).Float32()
					//alt:half r1_1 := (*(*I)(unsafe.Add(rCol1Base, kOff1))).Float32()

					c00 += O(la0*r0_0) + O(la1*r0_1)
					c01 += O(la0*r1_0) + O(la1*r1_1)
					c10 += O(lb0*r0_0) + O(lb1*r0_1)
					c11 += O(lb0*r1_0) + O(lb1*r1_1)
				}
				for ; k < contractingSize; k++ {
					kOff := uintptr(k) * iSize
					la0 := *(*I)(unsafe.Add(lRow0Base, kOff))  //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff))  //alt:generic
					r0_0 := *(*I)(unsafe.Add(rCol0Base, kOff)) //alt:generic
					r1_0 := *(*I)(unsafe.Add(rCol1Base, kOff)) //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff))).Float32()
					//alt:half r0_0 := (*(*I)(unsafe.Add(rCol0Base, kOff))).Float32()
					//alt:half r1_0 := (*(*I)(unsafe.Add(rCol1Base, kOff))).Float32()

					c00 += O(la0 * r0_0)
					c01 += O(la0 * r1_0)
					c10 += O(lb0 * r0_0)
					c11 += O(lb0 * r1_0)
				}

				out0 := unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*oSize)
				out1 := unsafe.Add(outputBase, uintptr((row+1)*rhsCrossSize+col)*oSize)
				*(*O)(out0) = c00
				*(*O)(unsafe.Add(out0, oSize)) = c01
				*(*O)(out1) = c10
				*(*O)(unsafe.Add(out1, oSize)) = c11
			}

			// Single col fringe (2 rows x 1 col)
			for ; col < rhsCrossSize; col++ {
				rColBase := unsafe.Add(rhsBase, uintptr(col)*rowByteStride)
				var c0, c1 O
				var k int
				for ; k+1 < contractingSize; k += 2 {
					kOff0 := uintptr(k) * iSize
					kOff1 := kOff0 + iSize

					la0 := *(*I)(unsafe.Add(lRow0Base, kOff0)) //alt:generic
					la1 := *(*I)(unsafe.Add(lRow0Base, kOff1)) //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff0)) //alt:generic
					lb1 := *(*I)(unsafe.Add(lRow1Base, kOff1)) //alt:generic
					rk0 := *(*I)(unsafe.Add(rColBase, kOff0))  //alt:generic
					rk1 := *(*I)(unsafe.Add(rColBase, kOff1))  //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff0))).Float32()
					//alt:half la1 := (*(*I)(unsafe.Add(lRow0Base, kOff1))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff0))).Float32()
					//alt:half lb1 := (*(*I)(unsafe.Add(lRow1Base, kOff1))).Float32()
					//alt:half rk0 := (*(*I)(unsafe.Add(rColBase, kOff0))).Float32()
					//alt:half rk1 := (*(*I)(unsafe.Add(rColBase, kOff1))).Float32()

					c0 += O(la0*rk0) + O(la1*rk1)
					c1 += O(lb0*rk0) + O(lb1*rk1)
				}
				for ; k < contractingSize; k++ {
					kOff := uintptr(k) * iSize
					la0 := *(*I)(unsafe.Add(lRow0Base, kOff)) //alt:generic
					lb0 := *(*I)(unsafe.Add(lRow1Base, kOff)) //alt:generic
					rk0 := *(*I)(unsafe.Add(rColBase, kOff))  //alt:generic

					//alt:half la0 := (*(*I)(unsafe.Add(lRow0Base, kOff))).Float32()
					//alt:half lb0 := (*(*I)(unsafe.Add(lRow1Base, kOff))).Float32()
					//alt:half rk0 := (*(*I)(unsafe.Add(rColBase, kOff))).Float32()

					c0 += O(la0 * rk0)
					c1 += O(lb0 * rk0)
				}
				*(*O)(unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*oSize)) = c0
				*(*O)(unsafe.Add(outputBase, uintptr((row+1)*rhsCrossSize+col)*oSize)) = c1
			}
		}

		// Row fringe (1 row x cols)
		for ; row < lhsCrossSize; row++ {
			lRowBase := unsafe.Add(lhsBase, uintptr(row)*rowByteStride)
			for col := 0; col < rhsCrossSize; col++ {
				rColBase := unsafe.Add(rhsBase, uintptr(col)*rowByteStride)
				var c O
				var k int
				for ; k+3 < contractingSize; k += 4 {
					kOff := uintptr(k) * iSize
					lIdx := unsafe.Add(lRowBase, kOff)
					rIdx := unsafe.Add(rColBase, kOff)

					la0 := *(*I)(lIdx)                      //alt:generic
					la1 := *(*I)(unsafe.Add(lIdx, iSize))   //alt:generic
					la2 := *(*I)(unsafe.Add(lIdx, 2*iSize)) //alt:generic
					la3 := *(*I)(unsafe.Add(lIdx, 3*iSize)) //alt:generic
					rk0 := *(*I)(rIdx)                      //alt:generic
					rk1 := *(*I)(unsafe.Add(rIdx, iSize))   //alt:generic
					rk2 := *(*I)(unsafe.Add(rIdx, 2*iSize)) //alt:generic
					rk3 := *(*I)(unsafe.Add(rIdx, 3*iSize)) //alt:generic

					//alt:half la0 := (*(*I)(lIdx)).Float32()
					//alt:half la1 := (*(*I)(unsafe.Add(lIdx, iSize))).Float32()
					//alt:half la2 := (*(*I)(unsafe.Add(lIdx, 2*iSize))).Float32()
					//alt:half la3 := (*(*I)(unsafe.Add(lIdx, 3*iSize))).Float32()
					//alt:half rk0 := (*(*I)(rIdx)).Float32()
					//alt:half rk1 := (*(*I)(unsafe.Add(rIdx, iSize))).Float32()
					//alt:half rk2 := (*(*I)(unsafe.Add(rIdx, 2*iSize))).Float32()
					//alt:half rk3 := (*(*I)(unsafe.Add(rIdx, 3*iSize))).Float32()

					c += O(la0*rk0) + O(la1*rk1) + O(la2*rk2) + O(la3*rk3)
				}
				for ; k < contractingSize; k++ {
					kOff := uintptr(k) * iSize
					la := *(*I)(unsafe.Add(lRowBase, kOff)) //alt:generic
					rk := *(*I)(unsafe.Add(rColBase, kOff)) //alt:generic
					//alt:half la := (*(*I)(unsafe.Add(lRowBase, kOff))).Float32()
					//alt:half rk := (*(*I)(unsafe.Add(rColBase, kOff))).Float32()
					c += O(la * rk)
				}
				*(*O)(unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*oSize)) = c
			}
		}
	}
}

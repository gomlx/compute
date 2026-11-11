// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !no_unsafe

package matmul

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
)

// smallUnsafeNoSIMDGenericParallel implements a parallelized version of the non-SIMD matrix
// multiplication for the non-transposed layout.
//
// This is the "unsafe" version using pointers. It is faster because it bysteps unnecessary bound-checks.
// Use -tags=no_unsafe to force the safe version (in file nosimd_small_safe.go)
func smallNoSIMDGenericParallel[I, O gotype.NumericNotComplex]( //alt:generic
	//alt:half func smallNoSIMDHalfPrecisionParallel[I gotype.HalfPrecision[I], O gotype.NumericNotComplex](
	backend *gobackend.Backend,
	lhs, rhs []I,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []O, matricesPerTask int) {

	backend.Scheduler.ParallelizeChunks(batchSize, matricesPerTask, func(start, end int) {
		smallNoSIMDGeneric( //alt:generic
			//alt:half smallNoSIMDHalfPrecision(
			lhs, rhs,
			start, end-start, lhsCrossSize, rhsCrossSize, contractingSize,
			output)
	})
}

// smallNoSIMDGeneric implements a non-SIMD matrix multiplication for the non-transposed layout.
//
// lhs:    shape [batchSize, lhsCrossSize, contractingSize].
// rhs:    shape [batchSize, contractingSize, rhsCrossSize].
// output: shape [batchSize, lhsCrossSize, rhsCrossSize].
//
// It is used for small inputs, where packing the data is not worth the cost.
func smallNoSIMDGeneric[I, O gotype.NumericNotComplex]( //alt:generic
	//alt:half func smallNoSIMDHalfPrecision[I gotype.HalfPrecision[I], O gotype.NumericNotComplex](
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

	if rhsCrossSize == 1 {
		smallNoSIMDGenericTransposed( //alt:generic
			//alt:half smallNoSIMDHalfPrecisionTransposed(
			lhs, rhs,
			batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize,
			output)
		return
	}

	if rhsCrossSize <= 16 && contractingSize*rhsCrossSize <= 2048 {
		var buf [2048]I
		rhsTransposed := buf[:contractingSize*rhsCrossSize]
		for b := batchStart; b < batchStart+batchCount; b++ {
			rhsBatch := rhs[b*rhsStride : (b+1)*rhsStride]
			for col := range rhsCrossSize {
				dstRow := rhsTransposed[col*contractingSize : (col+1)*contractingSize]
				srcIdx := col
				for k := range contractingSize {
					dstRow[k] = rhsBatch[srcIdx]
					srcIdx += rhsCrossSize
				}
			}
			smallNoSIMDGenericTransposed( //alt:generic
				//alt:half smallNoSIMDHalfPrecisionTransposed(
				lhs[b*lhsStride:(b+1)*lhsStride], rhsTransposed,
				0, 1, lhsCrossSize, rhsCrossSize, contractingSize,
				output[b*outputStride:(b+1)*outputStride])
		}
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
	rhsRowStride := uintptr(rhsCrossSize) * iSize

	for b := range batchCount {
		bIdx := uintptr(batchStart + b)
		lhsBase := unsafe.Add(lhsPtr, bIdx*lhsByteStride)
		rhsBase := unsafe.Add(rhsPtr, bIdx*rhsByteStride)
		outputBase := unsafe.Add(outputPtr, bIdx*outputByteStride)

		row := 0
		// Main Loop: Process 3 rows at a time
		for ; row+2 < lhsCrossSize; row += 3 {
			// Pre-calculate base pointers for the 3 LHS rows
			lRow0Base := unsafe.Add(lhsBase, uintptr(row*contractingSize)*iSize)
			lRow1Base := unsafe.Add(lRow0Base, uintptr(contractingSize)*iSize)
			lRow2Base := unsafe.Add(lRow1Base, uintptr(contractingSize)*iSize)

			col := 0
			// Main Tile: Process 4 columns at a time
			for ; col+3 < rhsCrossSize; col += 4 {
				var c00, c01, c02, c03 O
				var c10, c11, c12, c13 O
				var c20, c21, c22, c23 O

				// rIdx tracks the current row in the RHS for these 4 columns
				rIdx := unsafe.Add(rhsBase, uintptr(col)*iSize)
				for k := 0; ; {
					kOff := uintptr(k) * iSize

					// Load RHS row segment
					r0 := *(*I)(rIdx)                      //alt:generic
					r1 := *(*I)(unsafe.Add(rIdx, iSize))   //alt:generic
					r2 := *(*I)(unsafe.Add(rIdx, 2*iSize)) //alt:generic
					r3 := *(*I)(unsafe.Add(rIdx, 3*iSize)) //alt:generic
					//alt:half r0 := (*(*I)(rIdx)).Float32()
					//alt:half r1 := (*(*I)(unsafe.Add(rIdx, iSize))).Float32()
					//alt:half r2 := (*(*I)(unsafe.Add(rIdx, 2*iSize))).Float32()
					//alt:half r3 := (*(*I)(unsafe.Add(rIdx, 3*iSize))).Float32()

					// Row 0
					l0 := *(*I)(unsafe.Add(lRow0Base, kOff)) //alt:generic
					//alt:half l0 := (*(*I)(unsafe.Add(lRow0Base, kOff))).Float32()
					c00 += O(l0 * r0)
					c01 += O(l0 * r1)
					c02 += O(l0 * r2)
					c03 += O(l0 * r3)
					// Row 1
					l1 := *(*I)(unsafe.Add(lRow1Base, kOff)) //alt:generic
					//alt:half l1 := (*(*I)(unsafe.Add(lRow1Base, kOff))).Float32()
					c10 += O(l1 * r0)
					c11 += O(l1 * r1)
					c12 += O(l1 * r2)
					c13 += O(l1 * r3)
					// Row 2
					l2 := *(*I)(unsafe.Add(lRow2Base, kOff)) //alt:generic
					//alt:half l2 := (*(*I)(unsafe.Add(lRow2Base, kOff))).Float32()
					c20 += O(l2 * r0)
					c21 += O(l2 * r1)
					c22 += O(l2 * r2)
					c23 += O(l2 * r3)

					if k++; k >= contractingSize {
						break
					}
					rIdx = unsafe.Add(rIdx, rhsRowStride)
				}

				// Write 3x4 tile results
				outputIdx0 := unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*oSize)
				outputIdx1 := unsafe.Add(outputIdx0, uintptr(rhsCrossSize)*oSize)
				outputIdx2 := unsafe.Add(outputIdx0, uintptr(2*rhsCrossSize)*oSize)

				*(*O)(outputIdx0) = c00
				*(*O)(unsafe.Add(outputIdx0, oSize)) = c01
				*(*O)(unsafe.Add(outputIdx0, 2*oSize)) = c02
				*(*O)(unsafe.Add(outputIdx0, 3*oSize)) = c03
				*(*O)(outputIdx1) = c10
				*(*O)(unsafe.Add(outputIdx1, oSize)) = c11
				*(*O)(unsafe.Add(outputIdx1, 2*oSize)) = c12
				*(*O)(unsafe.Add(outputIdx1, 3*oSize)) = c13
				*(*O)(outputIdx2) = c20
				*(*O)(unsafe.Add(outputIdx2, oSize)) = c21
				*(*O)(unsafe.Add(outputIdx2, 2*oSize)) = c22
				*(*O)(unsafe.Add(outputIdx2, 3*oSize)) = c23
			}

			// Columns-fringe: handle remaining columns for the current 3 rows
			for ; col < rhsCrossSize; col++ {
				var c0, c1, c2 O
				rIdx := unsafe.Add(rhsBase, uintptr(col)*iSize)
				for k := 0; ; {
					kOff := uintptr(k) * iSize
					rk := *(*I)(rIdx) //alt:generic
					//alt:half rk := (*(*I)(rIdx)).Float32()
					l0 := *(*I)(unsafe.Add(lRow0Base, kOff)) //alt:generic
					l1 := *(*I)(unsafe.Add(lRow1Base, kOff)) //alt:generic
					l2 := *(*I)(unsafe.Add(lRow2Base, kOff)) //alt:generic
					//alt:half l0 := (*(*I)(unsafe.Add(lRow0Base, kOff))).Float32()
					//alt:half l1 := (*(*I)(unsafe.Add(lRow1Base, kOff))).Float32()
					//alt:half l2 := (*(*I)(unsafe.Add(lRow2Base, kOff))).Float32()

					c0 += O(l0 * rk)
					c1 += O(l1 * rk)
					c2 += O(l2 * rk)

					if k++; k >= contractingSize {
						break
					}
					rIdx = unsafe.Add(rIdx, rhsRowStride)
				}
				outputIdx := unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*oSize)
				*(*O)(outputIdx) = c0
				*(*O)(unsafe.Add(outputIdx, uintptr(rhsCrossSize)*oSize)) = c1
				*(*O)(unsafe.Add(outputIdx, uintptr(2*rhsCrossSize)*oSize)) = c2
			}
		}

		// Row-Fringe: Handle remaining rows (fewer than 3)
		for ; row < lhsCrossSize; row++ {
			lhsRow := unsafe.Add(lhsBase, uintptr(row*contractingSize)*iSize)
			outRow := unsafe.Add(outputBase, uintptr(row*rhsCrossSize)*oSize)
			for col := range rhsCrossSize {
				var acc O
				rhsCol := unsafe.Add(rhsBase, uintptr(col)*iSize)

				var contractingIdx int
				for ; contractingIdx+3 < contractingSize; contractingIdx += 4 {
					lhsIdx := unsafe.Add(lhsRow, uintptr(contractingIdx)*iSize)
					rhsIdx0 := unsafe.Add(rhsCol, uintptr(contractingIdx)*rhsRowStride)
					rhsIdx1 := unsafe.Add(rhsIdx0, rhsRowStride)
					rhsIdx2 := unsafe.Add(rhsIdx0, 2*rhsRowStride)
					rhsIdx3 := unsafe.Add(rhsIdx0, 3*rhsRowStride)

					l0 := *(*I)(lhsIdx)                      //alt:generic
					l1 := *(*I)(unsafe.Add(lhsIdx, iSize))   //alt:generic
					l2 := *(*I)(unsafe.Add(lhsIdx, 2*iSize)) //alt:generic
					l3 := *(*I)(unsafe.Add(lhsIdx, 3*iSize)) //alt:generic
					r0 := *(*I)(rhsIdx0)                     //alt:generic
					r1 := *(*I)(rhsIdx1)                     //alt:generic
					r2 := *(*I)(rhsIdx2)                     //alt:generic
					r3 := *(*I)(rhsIdx3)                     //alt:generic

					//alt:half l0 := (*(*I)(lhsIdx)).Float32()
					//alt:half l1 := (*(*I)(unsafe.Add(lhsIdx, iSize))).Float32()
					//alt:half l2 := (*(*I)(unsafe.Add(lhsIdx, 2*iSize))).Float32()
					//alt:half l3 := (*(*I)(unsafe.Add(lhsIdx, 3*iSize))).Float32()
					//alt:half r0 := (*(*I)(rhsIdx0)).Float32()
					//alt:half r1 := (*(*I)(rhsIdx1)).Float32()
					//alt:half r2 := (*(*I)(rhsIdx2)).Float32()
					//alt:half r3 := (*(*I)(rhsIdx3)).Float32()

					v0 := O(l0 * r0)
					v1 := O(l1 * r1)
					v2 := O(l2 * r2)
					v3 := O(l3 * r3)

					acc += v0 + v1 + v2 + v3
				}
				for ; contractingIdx < contractingSize; contractingIdx++ {
					l0 := *(*I)(unsafe.Add(lhsRow, uintptr(contractingIdx)*iSize))        //alt:generic
					r0 := *(*I)(unsafe.Add(rhsCol, uintptr(contractingIdx)*rhsRowStride)) //alt:generic
					//alt:half l0 := (*(*I)(unsafe.Add(lhsRow, uintptr(contractingIdx)*iSize))).Float32()
					//alt:half r0 := (*(*I)(unsafe.Add(rhsCol, uintptr(contractingIdx)*rhsRowStride))).Float32()
					acc += O(l0 * r0)
				}
				*(*O)(unsafe.Add(outRow, uintptr(col)*oSize)) = acc
			}
		}
	}
}

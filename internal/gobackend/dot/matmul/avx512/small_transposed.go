// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"unsafe"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
	//alt:f16 "github.com/gomlx/compute/dtypes/float16"
)

// avx512SmallFloat32TransposedAsm uses the handwritten AVX-512 assembly 4x4 tile microkernel
// for full tiles, and scalar loops in registers for remainder rows/columns.
func avx512SmallFloat32TransposedAsm( //alt:f32
	//alt:bf16 func avx512SmallBFloat16TransposedAsm(
	//alt:f16 func avx512SmallFloat16TransposedAsm(
	//alt:f64 func avx512SmallFloat64TransposedAsm(
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32) { //alt:f32|bf16|f16
	//alt:f64 output []float64) {

	if batchCount == 0 || lhsCrossSize == 0 || rhsCrossSize == 0 || contractingSize == 0 {
		return
	}

	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize

	lhsPtr := unsafe.Pointer(unsafe.SliceData(lhs))
	rhsPtr := unsafe.Pointer(unsafe.SliceData(rhs))
	outputPtr := unsafe.Pointer(unsafe.SliceData(output))

	const elemSize = 4 //alt:f32
	//alt:bf16 const elemSize = 2
	//alt:f16 const elemSize = 2
	//alt:f64 const elemSize = 8

	const outputElemSize = 4 //alt:f32|bf16|f16
	//alt:f64 const outputElemSize = 8

	lhsByteStride := uintptr(lhsStride) * elemSize
	rhsByteStride := uintptr(rhsStride) * elemSize
	outputByteStride := uintptr(outputStride) * outputElemSize

	for b := range batchCount {
		bIdx := uintptr(batchStart + b)
		lhsBase := unsafe.Add(lhsPtr, bIdx*lhsByteStride)
		rhsBase := unsafe.Add(rhsPtr, bIdx*rhsByteStride)
		outputBase := unsafe.Add(outputPtr, bIdx*outputByteStride)

		for row := 0; row < lhsCrossSize; row += 4 {
			r0 := row
			r1 := min(row+1, lhsCrossSize-1)
			r2 := min(row+2, lhsCrossSize-1)
			r3 := min(row+3, lhsCrossSize-1)

			lRow0 := unsafe.Add(lhsBase, uintptr(r0*contractingSize)*elemSize)
			lRow1 := unsafe.Add(lhsBase, uintptr(r1*contractingSize)*elemSize)
			lRow2 := unsafe.Add(lhsBase, uintptr(r2*contractingSize)*elemSize)
			lRow3 := unsafe.Add(lhsBase, uintptr(r3*contractingSize)*elemSize)

			col := 0
			for ; col+3 < rhsCrossSize; col += 4 {
				rCol0 := unsafe.Add(rhsBase, uintptr((col+0)*contractingSize)*elemSize)
				rCol1 := unsafe.Add(rhsBase, uintptr((col+1)*contractingSize)*elemSize)
				rCol2 := unsafe.Add(rhsBase, uintptr((col+2)*contractingSize)*elemSize)
				rCol3 := unsafe.Add(rhsBase, uintptr((col+3)*contractingSize)*elemSize)

				outRow0 := unsafe.Add(outputBase, uintptr(r0*rhsCrossSize+col)*outputElemSize)
				outRow1 := unsafe.Add(outputBase, uintptr(r1*rhsCrossSize+col)*outputElemSize)
				outRow2 := unsafe.Add(outputBase, uintptr(r2*rhsCrossSize+col)*outputElemSize)
				outRow3 := unsafe.Add(outputBase, uintptr(r3*rhsCrossSize+col)*outputElemSize)

				avx512SmallTransposedTile4x4Float32Asm( //alt:f32
					//alt:bf16 avx512SmallTransposedTile4x4BFloat16Asm(
					//alt:f16 avx512SmallTransposedTile4x4Float16Asm(
					//alt:f64 avx512SmallTransposedTile4x4Float64Asm(
					lRow0, lRow1, lRow2, lRow3,
					rCol0, rCol1, rCol2, rCol3,
					contractingSize,
					outRow0, outRow1, outRow2, outRow3)
			}

			// Remainder columns for these rows
			for ; col < rhsCrossSize; col++ {
				rCol0 := unsafe.Add(rhsBase, uintptr(col*contractingSize)*elemSize)
				outRow0 := unsafe.Add(outputBase, uintptr(r0*rhsCrossSize+col)*outputElemSize)
				outRow1 := unsafe.Add(outputBase, uintptr(r1*rhsCrossSize+col)*outputElemSize)
				outRow2 := unsafe.Add(outputBase, uintptr(r2*rhsCrossSize+col)*outputElemSize)
				outRow3 := unsafe.Add(outputBase, uintptr(r3*rhsCrossSize+col)*outputElemSize)

				avx512SmallTransposedTile4x1Float32Asm( //alt:f32
					//alt:bf16 avx512SmallTransposedTile4x1BFloat16Asm(
					//alt:f16 avx512SmallTransposedTile4x1Float16Asm(
					//alt:f64 avx512SmallTransposedTile4x1Float64Asm(
					lRow0, lRow1, lRow2, lRow3,
					rCol0,
					contractingSize,
					outRow0, outRow1, outRow2, outRow3)
			}
		}
	}
}

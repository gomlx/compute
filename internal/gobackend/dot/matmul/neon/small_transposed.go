// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"unsafe"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
	//alt:f16 "github.com/gomlx/compute/dtypes/float16"
)

// neonSmallFloat32TransposedAsm uses handwritten ARM64 NEON assembly Tile4x4 and Tile4x1 microkernels.
func neonSmallFloat32TransposedAsm( //alt:f32
	//alt:bf16 func neonSmallBFloat16TransposedAsm(
	//alt:f16 func neonSmallFloat16TransposedAsm(
	//alt:f64 func neonSmallFloat64TransposedAsm(
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

	outStrideBytes := rhsCrossSize * outputElemSize
	//alt:bf16 useBFDOT := NEONUseBFDOT

	var tmpOut [16]float32 //alt:f32|bf16|f16
	//alt:f64 var tmpOut [16]float64
	tmpPtr := unsafe.Pointer(&tmpOut[0])

	for b := range batchCount {
		bIdx := uintptr(batchStart + b)
		lhsBase := unsafe.Add(lhsPtr, bIdx*lhsByteStride)
		rhsBase := unsafe.Add(rhsPtr, bIdx*rhsByteStride)
		outputBase := unsafe.Add(outputPtr, bIdx*outputByteStride)

		row := 0
		for ; row+3 < lhsCrossSize; row += 4 {
			lRow0 := unsafe.Add(lhsBase, uintptr((row+0)*contractingSize)*elemSize)
			lRow1 := unsafe.Add(lhsBase, uintptr((row+1)*contractingSize)*elemSize)
			lRow2 := unsafe.Add(lhsBase, uintptr((row+2)*contractingSize)*elemSize)
			lRow3 := unsafe.Add(lhsBase, uintptr((row+3)*contractingSize)*elemSize)

			col := 0
			for ; col+3 < rhsCrossSize; col += 4 {
				rCol0 := unsafe.Add(rhsBase, uintptr((col+0)*contractingSize)*elemSize)
				rCol1 := unsafe.Add(rhsBase, uintptr((col+1)*contractingSize)*elemSize)
				rCol2 := unsafe.Add(rhsBase, uintptr((col+2)*contractingSize)*elemSize)
				rCol3 := unsafe.Add(rhsBase, uintptr((col+3)*contractingSize)*elemSize)
				outPtr := unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*outputElemSize)

				neonSmallTransposedTile4x4Float32Asm( //alt:f32
					//alt:f16 neonSmallTransposedTile4x4Float16Asm(
					//alt:f64 neonSmallTransposedTile4x4Float64Asm(
					lRow0, lRow1, lRow2, lRow3, //alt:f32|f16|f64
					rCol0, rCol1, rCol2, rCol3, //alt:f32|f16|f64
					contractingSize, //alt:f32|f16|f64
					outPtr,          //alt:f32|f16|f64
					outStrideBytes)  //alt:f32|f16|f64
				//alt:bf16 neonSmallTransposedTile4x4BFloat16Asm(
				//alt:bf16 	lRow0, lRow1, lRow2, lRow3,
				//alt:bf16 	rCol0, rCol1, rCol2, rCol3,
				//alt:bf16 	contractingSize,
				//alt:bf16 	outPtr,
				//alt:bf16 	outStrideBytes,
				//alt:bf16 	useBFDOT)
			}

			for ; col < rhsCrossSize; col++ {
				rCol0 := unsafe.Add(rhsBase, uintptr(col*contractingSize)*elemSize)
				outPtr := unsafe.Add(outputBase, uintptr(row*rhsCrossSize+col)*outputElemSize)

				neonSmallTransposedTile4x1Float32Asm( //alt:f32
					//alt:f16 neonSmallTransposedTile4x1Float16Asm(
					//alt:f64 neonSmallTransposedTile4x1Float64Asm(
					lRow0, lRow1, lRow2, lRow3, //alt:f32|f16|f64
					rCol0,           //alt:f32|f16|f64
					contractingSize, //alt:f32|f16|f64
					outPtr,          //alt:f32|f16|f64
					outStrideBytes)  //alt:f32|f16|f64
				//alt:bf16 neonSmallTransposedTile4x1BFloat16Asm(
				//alt:bf16 	lRow0, lRow1, lRow2, lRow3,
				//alt:bf16 	rCol0,
				//alt:bf16 	contractingSize,
				//alt:bf16 	outPtr,
				//alt:bf16 	outStrideBytes,
				//alt:bf16 	useBFDOT)
			}
		}

		// Remainder rows (1..3 rows): use padded 4-row tile writing into stack tmpOut.
		if remRows := lhsCrossSize - row; remRows > 0 {
			r0Idx := row
			r1Idx := min(row+1, lhsCrossSize-1)
			r2Idx := min(row+2, lhsCrossSize-1)
			r3Idx := min(row+3, lhsCrossSize-1)

			lRow0 := unsafe.Add(lhsBase, uintptr(r0Idx*contractingSize)*elemSize)
			lRow1 := unsafe.Add(lhsBase, uintptr(r1Idx*contractingSize)*elemSize)
			lRow2 := unsafe.Add(lhsBase, uintptr(r2Idx*contractingSize)*elemSize)
			lRow3 := unsafe.Add(lhsBase, uintptr(r3Idx*contractingSize)*elemSize)

			col := 0
			for ; col+3 < rhsCrossSize; col += 4 {
				rCol0 := unsafe.Add(rhsBase, uintptr((col+0)*contractingSize)*elemSize)
				rCol1 := unsafe.Add(rhsBase, uintptr((col+1)*contractingSize)*elemSize)
				rCol2 := unsafe.Add(rhsBase, uintptr((col+2)*contractingSize)*elemSize)
				rCol3 := unsafe.Add(rhsBase, uintptr((col+3)*contractingSize)*elemSize)

				neonSmallTransposedTile4x4Float32Asm( //alt:f32
					//alt:f16 neonSmallTransposedTile4x4Float16Asm(
					//alt:f64 neonSmallTransposedTile4x4Float64Asm(
					lRow0, lRow1, lRow2, lRow3, //alt:f32|f16|f64
					rCol0, rCol1, rCol2, rCol3, //alt:f32|f16|f64
					contractingSize,  //alt:f32|f16|f64
					tmpPtr,           //alt:f32|f16|f64
					4*outputElemSize) //alt:f32|f16|f64
				//alt:bf16 neonSmallTransposedTile4x4BFloat16Asm(
				//alt:bf16 	lRow0, lRow1, lRow2, lRow3,
				//alt:bf16 	rCol0, rCol1, rCol2, rCol3,
				//alt:bf16 	contractingSize,
				//alt:bf16 	tmpPtr,
				//alt:bf16 	4*outputElemSize,
				//alt:bf16 	useBFDOT)

				for r := range remRows {
					outRowPtr := (*[4]float32)(unsafe.Add(outputBase, uintptr((row+r)*rhsCrossSize+col)*outputElemSize)) //alt:f32|bf16|f16
					//alt:f64 outRowPtr := (*[4]float64)(unsafe.Add(outputBase, uintptr((row+r)*rhsCrossSize+col)*outputElemSize))
					outRowPtr[0] = tmpOut[r*4+0]
					outRowPtr[1] = tmpOut[r*4+1]
					outRowPtr[2] = tmpOut[r*4+2]
					outRowPtr[3] = tmpOut[r*4+3]
				}
			}

			for ; col < rhsCrossSize; col++ {
				rCol0 := unsafe.Add(rhsBase, uintptr(col*contractingSize)*elemSize)

				neonSmallTransposedTile4x1Float32Asm( //alt:f32
					//alt:f16 neonSmallTransposedTile4x1Float16Asm(
					//alt:f64 neonSmallTransposedTile4x1Float64Asm(
					lRow0, lRow1, lRow2, lRow3, //alt:f32|f16|f64
					rCol0,           //alt:f32|f16|f64
					contractingSize, //alt:f32|f16|f64
					tmpPtr,          //alt:f32|f16|f64
					outputElemSize)  //alt:f32|f16|f64
				//alt:bf16 neonSmallTransposedTile4x1BFloat16Asm(
				//alt:bf16 	lRow0, lRow1, lRow2, lRow3,
				//alt:bf16 	rCol0,
				//alt:bf16 	contractingSize,
				//alt:bf16 	tmpPtr,
				//alt:bf16 	outputElemSize,
				//alt:bf16 	useBFDOT)

				for r := range remRows {
					*(*float32)(unsafe.Add(outputBase, uintptr((row+r)*rhsCrossSize+col)*outputElemSize)) = tmpOut[r] //alt:f32|bf16|f16
					//alt:f64 *(*float64)(unsafe.Add(outputBase, uintptr((row+r)*rhsCrossSize+col)*outputElemSize)) = tmpOut[r]
				}
			}
		}
	}
}

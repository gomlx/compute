// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"unsafe"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
	//alt:f16 "github.com/gomlx/compute/dtypes/float16"
)

// transposeRHSFloat32 transposes a matrix of shape [K, N] to [N, K].
func transposeRHSFloat32(src, dst []float32, K, N int) { //alt:f32
	//alt:bf16 func transposeRHSBFloat16(src, dst []bfloat16.BFloat16, K, N int) {
	//alt:f16 func transposeRHSFloat16(src, dst []float16.Float16, K, N int) {
	//alt:f64 func transposeRHSFloat64(src, dst []float64, K, N int) {
	switch N {
	case 1:
		copy(dst, src[:K])
	case 2:
		for k := range K {
			dst[0*K+k] = src[k*2+0]
			dst[1*K+k] = src[k*2+1]
		}
	case 4:
		for k := range K {
			dst[0*K+k] = src[k*4+0]
			dst[1*K+k] = src[k*4+1]
			dst[2*K+k] = src[k*4+2]
			dst[3*K+k] = src[k*4+3]
		}
	case 8:
		for k := range K {
			dst[0*K+k] = src[k*8+0]
			dst[1*K+k] = src[k*8+1]
			dst[2*K+k] = src[k*8+2]
			dst[3*K+k] = src[k*8+3]
			dst[4*K+k] = src[k*8+4]
			dst[5*K+k] = src[k*8+5]
			dst[6*K+k] = src[k*8+6]
			dst[7*K+k] = src[k*8+7]
		}
	default:
		for k := range K {
			for n := range N {
				dst[n*K+k] = src[k*N+n]
			}
		}
	}
}

// avx2SmallFloat32NonTransposedAsm implements NonTransposed small matmul (C = A * B).
// - For N == 1 (GEMV), zero-copy routes to Transposed.
// - For N < vecWidth, transposes RHS into a small stack buffer and routes to Transposed Asm kernel.
// - For N >= vecWidth, executes column-vectorized AVX2 microkernels accumulating across K in registers.
func avx2SmallFloat32NonTransposedAsm( //alt:f32
	//alt:bf16 func avx2SmallBFloat16NonTransposedAsm(
	//alt:f16 func avx2SmallFloat16NonTransposedAsm(
	//alt:f64 func avx2SmallFloat64NonTransposedAsm(
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

	// 1. If N == 1 (matrix-vector), RHS in memory is [K, 1] which is identical to [1, K].
	if rhsCrossSize == 1 {
		avx2SmallFloat32TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output) //alt:f32
		//alt:bf16 avx2SmallBFloat16TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output)
		//alt:f16 avx2SmallFloat16TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output)
		//alt:f64 avx2SmallFloat64TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output)
		return
	}

	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize

	const elemSize = 4 //alt:f32
	//alt:bf16 const elemSize = 2
	//alt:f16 const elemSize = 2
	//alt:f64 const elemSize = 8

	const outputElemSize = 4 //alt:f32|bf16|f16
	//alt:f64 const outputElemSize = 8

	const vecWidth = 8 //alt:f32|bf16|f16
	//alt:f64 const vecWidth = 4

	// 2. If N < vecWidth, transposing RHS [K, N] -> [N, K] allows full vectorization across K.
	if rhsCrossSize < vecWidth {
		var stackRhsT [2048]float32 //alt:f32
		//alt:bf16 var stackRhsT [2048]bfloat16.BFloat16
		//alt:f16 var stackRhsT [2048]float16.Float16
		//alt:f64 var stackRhsT [2048]float64
		var rhsT []float32 //alt:f32
		//alt:bf16 var rhsT []bfloat16.BFloat16
		//alt:f16 var rhsT []float16.Float16
		//alt:f64 var rhsT []float64
		kn := contractingSize * rhsCrossSize
		if kn <= len(stackRhsT) {
			rhsT = stackRhsT[:kn]
		} else {
			rhsT = make([]float32, kn) //alt:f32
			//alt:bf16 rhsT = make([]bfloat16.BFloat16, kn)
			//alt:f16 rhsT = make([]float16.Float16, kn)
			//alt:f64 rhsT = make([]float64, kn)
		}

		for b := range batchCount {
			bIdx := batchStart + b
			rhsSlice := rhs[bIdx*rhsStride : (bIdx+1)*rhsStride]
			transposeRHSFloat32(rhsSlice, rhsT, contractingSize, rhsCrossSize) //alt:f32
			//alt:bf16 transposeRHSBFloat16(rhsSlice, rhsT, contractingSize, rhsCrossSize)
			//alt:f16 transposeRHSFloat16(rhsSlice, rhsT, contractingSize, rhsCrossSize)
			//alt:f64 transposeRHSFloat64(rhsSlice, rhsT, contractingSize, rhsCrossSize)

			lhsSlice := lhs[bIdx*lhsStride : (bIdx+1)*lhsStride]
			outSlice := output[bIdx*outputStride : (bIdx+1)*outputStride]
			avx2SmallFloat32TransposedAsm(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice) //alt:f32
			//alt:bf16 avx2SmallBFloat16TransposedAsm(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice)
			//alt:f16 avx2SmallFloat16TransposedAsm(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice)
			//alt:f64 avx2SmallFloat64TransposedAsm(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice)
		}
		return
	}

	// 3. For N >= vecWidth: Column-vectorized execution accumulating in registers across K.
	lhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(lhs)))
	rhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(rhs)))
	outputPtr := uintptr(unsafe.Pointer(unsafe.SliceData(output)))

	lhsByteStride := uintptr(lhsStride) * elemSize
	rhsByteStride := uintptr(rhsStride) * elemSize
	outputByteStride := uintptr(outputStride) * outputElemSize

	lhsBase := lhsPtr + uintptr(batchStart)*lhsByteStride
	rhsBase := rhsPtr + uintptr(batchStart)*rhsByteStride
	outputBase := outputPtr + uintptr(batchStart)*outputByteStride

	rowStrideBytes := rhsCrossSize * elemSize

	for range batchCount {
		row := 0
		for ; row+3 < lhsCrossSize; row += 4 {
			lRow0 := unsafe.Pointer(lhsBase + uintptr((row+0)*contractingSize)*elemSize)
			lRow1 := unsafe.Pointer(lhsBase + uintptr((row+1)*contractingSize)*elemSize)
			lRow2 := unsafe.Pointer(lhsBase + uintptr((row+2)*contractingSize)*elemSize)
			lRow3 := unsafe.Pointer(lhsBase + uintptr((row+3)*contractingSize)*elemSize)

			col := 0
			// 4 rows x 16 cols tile //alt:f32|bf16|f16
			for ; col+15 < rhsCrossSize; col += 16 { //alt:f32|bf16|f16
				rhsCol := unsafe.Pointer(rhsBase + uintptr(col)*elemSize) //alt:f32|bf16|f16
				outRow0 := unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16
				outRow1 := unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16
				outRow2 := unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16
				outRow3 := unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16

				avx2SmallNonTransposedTile4x16Float32Asm( //alt:f32
					//alt:bf16 avx2SmallNonTransposedTile4x16BFloat16Asm(
					//alt:f16 avx2SmallNonTransposedTile4x16Float16Asm(
					lRow0, lRow1, lRow2, lRow3, //alt:f32|bf16|f16
					rhsCol, //alt:f32|bf16|f16
					contractingSize, //alt:f32|bf16|f16
					rowStrideBytes, //alt:f32|bf16|f16
					outRow0, outRow1, outRow2, outRow3) //alt:f32|bf16|f16
			} //alt:f32|bf16|f16

			// 4 rows x 8 cols tile //alt:f32|bf16|f16
			for ; col+7 < rhsCrossSize; col += 8 { //alt:f32|bf16|f16
				rhsCol := unsafe.Pointer(rhsBase + uintptr(col)*elemSize) //alt:f32|bf16|f16
				outRow0 := unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16
				outRow1 := unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16
				outRow2 := unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16
				outRow3 := unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*outputElemSize) //alt:f32|bf16|f16

				avx2SmallNonTransposedTile4x8Float32Asm( //alt:f32
					//alt:bf16 avx2SmallNonTransposedTile4x8BFloat16Asm(
					//alt:f16 avx2SmallNonTransposedTile4x8Float16Asm(
					lRow0, lRow1, lRow2, lRow3, //alt:f32|bf16|f16
					rhsCol, //alt:f32|bf16|f16
					contractingSize, //alt:f32|bf16|f16
					rowStrideBytes, //alt:f32|bf16|f16
					outRow0, outRow1, outRow2, outRow3) //alt:f32|bf16|f16
			} //alt:f32|bf16|f16

			// 4 rows x 8 cols tile //alt:f64
			//alt:f64 for ; col+7 < rhsCrossSize; col += 8 {
			//alt:f64 	rhsCol := unsafe.Pointer(rhsBase + uintptr(col)*elemSize)
			//alt:f64 	outRow0 := unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*outputElemSize)
			//alt:f64 	outRow1 := unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*outputElemSize)
			//alt:f64 	outRow2 := unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*outputElemSize)
			//alt:f64 	outRow3 := unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*outputElemSize)
			//alt:f64
			//alt:f64 	avx2SmallNonTransposedTile4x8Float64Asm(
			//alt:f64 		lRow0, lRow1, lRow2, lRow3,
			//alt:f64 		rhsCol,
			//alt:f64 		contractingSize,
			//alt:f64 		rowStrideBytes,
			//alt:f64 		outRow0, outRow1, outRow2, outRow3)
			//alt:f64 }

			// 4 rows x 4 cols tile //alt:f64
			//alt:f64 for ; col+3 < rhsCrossSize; col += 4 {
			//alt:f64 	rhsCol := unsafe.Pointer(rhsBase + uintptr(col)*elemSize)
			//alt:f64 	outRow0 := unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*outputElemSize)
			//alt:f64 	outRow1 := unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*outputElemSize)
			//alt:f64 	outRow2 := unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*outputElemSize)
			//alt:f64 	outRow3 := unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*outputElemSize)
			//alt:f64
			//alt:f64 	avx2SmallNonTransposedTile4x4Float64Asm(
			//alt:f64 		lRow0, lRow1, lRow2, lRow3,
			//alt:f64 		rhsCol,
			//alt:f64 		contractingSize,
			//alt:f64 		rowStrideBytes,
			//alt:f64 		outRow0, outRow1, outRow2, outRow3)
			//alt:f64 }

			// Remainder columns for these 4 rows
			for ; col < rhsCrossSize; col++ {
				var s0, s1, s2, s3 float32 //alt:f32|bf16|f16
				//alt:f64 var s0, s1, s2, s3 float64
				for k := range contractingSize {
					kOffset := uintptr(k) * elemSize
					l0 := *(*float32)(unsafe.Pointer(uintptr(lRow0) + kOffset)) //alt:f32
					//alt:bf16 l0 := (*(*bfloat16.BFloat16)(unsafe.Pointer(uintptr(lRow0) + kOffset))).Float32()
					//alt:f16 l0 := (*(*float16.Float16)(unsafe.Pointer(uintptr(lRow0) + kOffset))).Float32()
					//alt:f64 l0 := *(*float64)(unsafe.Pointer(uintptr(lRow0) + kOffset))

					l1 := *(*float32)(unsafe.Pointer(uintptr(lRow1) + kOffset)) //alt:f32
					//alt:bf16 l1 := (*(*bfloat16.BFloat16)(unsafe.Pointer(uintptr(lRow1) + kOffset))).Float32()
					//alt:f16 l1 := (*(*float16.Float16)(unsafe.Pointer(uintptr(lRow1) + kOffset))).Float32()
					//alt:f64 l1 := *(*float64)(unsafe.Pointer(uintptr(lRow1) + kOffset))

					l2 := *(*float32)(unsafe.Pointer(uintptr(lRow2) + kOffset)) //alt:f32
					//alt:bf16 l2 := (*(*bfloat16.BFloat16)(unsafe.Pointer(uintptr(lRow2) + kOffset))).Float32()
					//alt:f16 l2 := (*(*float16.Float16)(unsafe.Pointer(uintptr(lRow2) + kOffset))).Float32()
					//alt:f64 l2 := *(*float64)(unsafe.Pointer(uintptr(lRow2) + kOffset))

					l3 := *(*float32)(unsafe.Pointer(uintptr(lRow3) + kOffset)) //alt:f32
					//alt:bf16 l3 := (*(*bfloat16.BFloat16)(unsafe.Pointer(uintptr(lRow3) + kOffset))).Float32()
					//alt:f16 l3 := (*(*float16.Float16)(unsafe.Pointer(uintptr(lRow3) + kOffset))).Float32()
					//alt:f64 l3 := *(*float64)(unsafe.Pointer(uintptr(lRow3) + kOffset))

					rVal := *(*float32)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize)) //alt:f32
					//alt:bf16 rVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize))).Float32()
					//alt:f16 rVal := (*(*float16.Float16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize))).Float32()
					//alt:f64 rVal := *(*float64)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize))

					s0 += l0 * rVal
					s1 += l1 * rVal
					s2 += l2 * rVal
					s3 += l3 * rVal
				}
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*outputElemSize)) = s0 //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*outputElemSize)) = s0
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*outputElemSize)) = s1 //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*outputElemSize)) = s1
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*outputElemSize)) = s2 //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*outputElemSize)) = s2
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*outputElemSize)) = s3 //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*outputElemSize)) = s3
			}
		}

		// Remainder rows
		for ; row < lhsCrossSize; row++ {
			lRow := lhsBase + uintptr(row*contractingSize)*elemSize
			for col := range rhsCrossSize {
				var s float32 //alt:f32|bf16|f16
				//alt:f64 var s float64
				for k := range contractingSize {
					kOffset := uintptr(k) * elemSize
					lVal := *(*float32)(unsafe.Pointer(lRow + kOffset)) //alt:f32
					//alt:bf16 lVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow + kOffset))).Float32()
					//alt:f16 lVal := (*(*float16.Float16)(unsafe.Pointer(lRow + kOffset))).Float32()
					//alt:f64 lVal := *(*float64)(unsafe.Pointer(lRow + kOffset))

					rVal := *(*float32)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize)) //alt:f32
					//alt:bf16 rVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize))).Float32()
					//alt:f16 rVal := (*(*float16.Float16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize))).Float32()
					//alt:f64 rVal := *(*float64)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*elemSize))

					s += lVal * rVal
				}
				*(*float32)(unsafe.Pointer(outputBase + uintptr(row*rhsCrossSize+col)*outputElemSize)) = s //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr(row*rhsCrossSize+col)*outputElemSize)) = s
			}
		}

		lhsBase += lhsByteStride
		rhsBase += rhsByteStride
		outputBase += outputByteStride
	}
}

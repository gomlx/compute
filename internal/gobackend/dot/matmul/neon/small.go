// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
	//alt:f16 "github.com/gomlx/compute/dtypes/float16"
)

// neonSmallFloat32Parallel implements a parallelized version of the NEON small matrix
// multiplication.
func neonSmallFloat32Parallel( //alt:f32
	//alt:bf16 func neonSmallBFloat16Parallel(
	//alt:f16 func neonSmallFloat16Parallel(
	//alt:f64 func neonSmallFloat64Parallel(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32) { //alt:f32|bf16|f16
	//alt:f64 output []float64) {

	flopsPerMatrix := max(1, lhsCrossSize*rhsCrossSize*contractingSize)
	matricesPerWorker := max(1, (matmul.MinMatMulFlopsPerWorker+flopsPerMatrix-1)/flopsPerMatrix)

	if backend.Scheduler == nil || !backend.Scheduler.IsEnabled() {
		if layout == dot.LayoutNonTransposed {
			neonSmallFloat32( //alt:f32
				//alt:bf16 neonSmallBFloat16(
				//alt:f16 neonSmallFloat16(
				//alt:f64 neonSmallFloat64(
				lhs, rhs, 0, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		} else {
			neonSmallFloat32Transposed( //alt:f32
				//alt:bf16 neonSmallBFloat16Transposed(
				//alt:f16 neonSmallFloat16Transposed(
				//alt:f64 neonSmallFloat64Transposed(
				lhs, rhs, 0, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		}
		return
	}

	if batchSize > matricesPerWorker {
		backend.Scheduler.ParallelizeChunks(batchSize, matricesPerWorker, func(start, end int) {
			count := end - start
			if layout == dot.LayoutNonTransposed {
				neonSmallFloat32( //alt:f32
					//alt:bf16 neonSmallBFloat16(
					//alt:f16 neonSmallFloat16(
					//alt:f64 neonSmallFloat64(
					lhs, rhs, start, count, lhsCrossSize, rhsCrossSize, contractingSize, output)
			} else {
				neonSmallFloat32Transposed( //alt:f32
					//alt:bf16 neonSmallBFloat16Transposed(
					//alt:f16 neonSmallFloat16Transposed(
					//alt:f64 neonSmallFloat64Transposed(
					lhs, rhs, start, count, lhsCrossSize, rhsCrossSize, contractingSize, output)
			}
		})
		return
	}

	// Single-batch (batchSize == 1) with enough work to parallelize across LHS rows:
	if batchSize == 1 && flopsPerMatrix >= 2*matmul.MinMatMulFlopsPerWorker && lhsCrossSize >= 8 {
		flopsPerRow := max(1, rhsCrossSize*contractingSize)
		rowsPerWorker := max(4, ((matmul.MinMatMulFlopsPerWorker+flopsPerRow-1)/flopsPerRow+3)&^3)
		if rowsPerWorker < lhsCrossSize {
			backend.Scheduler.ParallelizeChunks(lhsCrossSize, rowsPerWorker, func(rowStart, rowEnd int) {
				rowCount := rowEnd - rowStart
				lhsSlice := lhs[rowStart*contractingSize : rowEnd*contractingSize]
				outSlice := output[rowStart*rhsCrossSize : rowEnd*rhsCrossSize]
				if layout == dot.LayoutNonTransposed {
					neonSmallFloat32( //alt:f32
						//alt:bf16 neonSmallBFloat16(
						//alt:f16 neonSmallFloat16(
						//alt:f64 neonSmallFloat64(
						lhsSlice, rhs, 0, 1, rowCount, rhsCrossSize, contractingSize, outSlice)
				} else {
					neonSmallFloat32Transposed( //alt:f32
						//alt:bf16 neonSmallBFloat16Transposed(
						//alt:f16 neonSmallFloat16Transposed(
						//alt:f64 neonSmallFloat64Transposed(
						lhsSlice, rhs, 0, 1, rowCount, rhsCrossSize, contractingSize, outSlice)
				}
			})
			return
		}
	}

	if layout == dot.LayoutNonTransposed {
		neonSmallFloat32( //alt:f32
			//alt:bf16 neonSmallBFloat16(
			//alt:f16 neonSmallFloat16(
			//alt:f64 neonSmallFloat64(
			lhs, rhs, 0, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
	} else {
		neonSmallFloat32Transposed( //alt:f32
			//alt:bf16 neonSmallBFloat16Transposed(
			//alt:f16 neonSmallFloat16Transposed(
			//alt:f64 neonSmallFloat64Transposed(
			lhs, rhs, 0, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
	}
}

// neonSmallFloat32 implements a NEON matrix multiplication for small inputs,
// without data packing. Layout is dot.LayoutNonTransposed.
func neonSmallFloat32( //alt:f32
	//alt:bf16 func neonSmallBFloat16(
	//alt:f16 func neonSmallFloat16(
	//alt:f64 func neonSmallFloat64(
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

	if NEONUseAsm {
		neonSmallFloat32NonTransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output) //alt:f32
		//alt:bf16 neonSmallBFloat16NonTransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
		//alt:f16 neonSmallFloat16NonTransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
		//alt:f64 neonSmallFloat64NonTransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
		return
	}

	if rhsCrossSize == 1 {
		neonSmallFloat32Transposed(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output) //alt:f32
		//alt:bf16 neonSmallBFloat16Transposed(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output)
		//alt:f16 neonSmallFloat16Transposed(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output)
		//alt:f64 neonSmallFloat64Transposed(lhs, rhs, batchStart, batchCount, lhsCrossSize, 1, contractingSize, output)
		return
	}

	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize

	const vecWidth = 4 //alt:f32
	//alt:bf16 const vecWidth = 8
	//alt:f16 const vecWidth = 8
	//alt:f64 const vecWidth = 2

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
			neonSmallFloat32Transposed(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice) //alt:f32
			//alt:bf16 neonSmallBFloat16Transposed(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice)
			//alt:f16 neonSmallFloat16Transposed(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice)
			//alt:f64 neonSmallFloat64Transposed(lhsSlice, rhsT, 0, 1, lhsCrossSize, rhsCrossSize, contractingSize, outSlice)
		}
		return
	}

	var iZero float32 //alt:f32
	//alt:bf16 var iZero bfloat16.BFloat16
	//alt:f16 var iZero float16.Float16
	//alt:f64 var iZero float64
	iSize := unsafe.Sizeof(iZero)
	var oZero float32 //alt:f32|bf16|f16
	//alt:f64 var oZero float64
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

	const outputVecWidth = 4 //alt:f32|bf16|f16
	//alt:f64 const outputVecWidth = 2

	for range batchCount {
		row := 0
		for ; row+3 < lhsCrossSize; row += 4 {
			lRow0Base := lhsBase + uintptr((row+0)*contractingSize)*iSize
			lRow1Base := lhsBase + uintptr((row+1)*contractingSize)*iSize
			lRow2Base := lhsBase + uintptr((row+2)*contractingSize)*iSize
			lRow3Base := lhsBase + uintptr((row+3)*contractingSize)*iSize

			col := 0
			// 4 rows x 2*outputVecWidth cols register-accumulated across K
			for ; col+2*outputVecWidth <= rhsCrossSize; col += 2 * outputVecWidth {
				zero := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
				//alt:f64 zero := archsimd.BroadcastFloat64x2(0.0)
				o0_0, o0_1 := zero, zero
				o1_0, o1_1 := zero, zero
				o2_0, o2_1 := zero, zero
				o3_0, o3_1 := zero, zero

				for k := range contractingSize {
					kOffset := uintptr(k) * iSize
					l0 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow0Base + kOffset))) //alt:f32
					l1 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow1Base + kOffset))) //alt:f32
					l2 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow2Base + kOffset))) //alt:f32
					l3 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow3Base + kOffset))) //alt:f32
					//alt:bf16 l0 := archsimd.BroadcastFloat32x4((*(*bfloat16.BFloat16)(unsafe.Pointer(lRow0Base + kOffset))).Float32())
					//alt:bf16 l1 := archsimd.BroadcastFloat32x4((*(*bfloat16.BFloat16)(unsafe.Pointer(lRow1Base + kOffset))).Float32())
					//alt:bf16 l2 := archsimd.BroadcastFloat32x4((*(*bfloat16.BFloat16)(unsafe.Pointer(lRow2Base + kOffset))).Float32())
					//alt:bf16 l3 := archsimd.BroadcastFloat32x4((*(*bfloat16.BFloat16)(unsafe.Pointer(lRow3Base + kOffset))).Float32())
					//alt:f16 l0 := archsimd.BroadcastFloat32x4((*(*float16.Float16)(unsafe.Pointer(lRow0Base + kOffset))).Float32())
					//alt:f16 l1 := archsimd.BroadcastFloat32x4((*(*float16.Float16)(unsafe.Pointer(lRow1Base + kOffset))).Float32())
					//alt:f16 l2 := archsimd.BroadcastFloat32x4((*(*float16.Float16)(unsafe.Pointer(lRow2Base + kOffset))).Float32())
					//alt:f16 l3 := archsimd.BroadcastFloat32x4((*(*float16.Float16)(unsafe.Pointer(lRow3Base + kOffset))).Float32())
					//alt:f64 l0 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow0Base + kOffset)))
					//alt:f64 l1 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow1Base + kOffset)))
					//alt:f64 l2 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow2Base + kOffset)))
					//alt:f64 l3 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow3Base + kOffset)))

					rColBase := rhsBase + uintptr(k*rhsCrossSize+col)*iSize
					var r0, r1 archsimd.Float32x4 //alt:f32|bf16|f16
					//alt:f64 var r0, r1 archsimd.Float64x2
					{ //alt:f32|bf16|f16|f64
						r0 = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rColBase)))                                 //alt:f32
						r1 = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rColBase + uintptr(outputVecWidth)*iSize))) //alt:f32
						//alt:bf16 r0_bf16 := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(rColBase)))
						//alt:bf16 r0, r1 = r0_bf16.ToFloat32()
						//alt:f16 r0_f16 := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(rColBase)))
						//alt:f16 r0, r1 = r0_f16.ToFloat32()
						//alt:f64 r0 = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rColBase)))
						//alt:f64 r1 = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rColBase + uintptr(outputVecWidth)*iSize)))
					} //alt:f32|bf16|f16|f64

					o0_0 = l0.MulAdd(r0, o0_0)
					o0_1 = l0.MulAdd(r1, o0_1)
					o1_0 = l1.MulAdd(r0, o1_0)
					o1_1 = l1.MulAdd(r1, o1_1)
					o2_0 = l2.MulAdd(r0, o2_0)
					o2_1 = l2.MulAdd(r1, o2_1)
					o3_0 = l3.MulAdd(r0, o3_0)
					o3_1 = l3.MulAdd(r1, o3_1)
				}

				outBase0 := outputBase + uintptr((row+0)*rhsCrossSize+col)*oSize
				outBase1 := outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize
				outBase2 := outputBase + uintptr((row+2)*rhsCrossSize+col)*oSize
				outBase3 := outputBase + uintptr((row+3)*rhsCrossSize+col)*oSize

				o0_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase0)))                                 //alt:f32|bf16|f16
				o0_1.StoreArray((*[4]float32)(unsafe.Pointer(outBase0 + uintptr(outputVecWidth)*oSize))) //alt:f32|bf16|f16
				o1_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase1)))                                 //alt:f32|bf16|f16
				o1_1.StoreArray((*[4]float32)(unsafe.Pointer(outBase1 + uintptr(outputVecWidth)*oSize))) //alt:f32|bf16|f16
				o2_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase2)))                                 //alt:f32|bf16|f16
				o2_1.StoreArray((*[4]float32)(unsafe.Pointer(outBase2 + uintptr(outputVecWidth)*oSize))) //alt:f32|bf16|f16
				o3_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase3)))                                 //alt:f32|bf16|f16
				o3_1.StoreArray((*[4]float32)(unsafe.Pointer(outBase3 + uintptr(outputVecWidth)*oSize))) //alt:f32|bf16|f16
				//alt:f64 o0_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase0)))
				//alt:f64 o0_1.StoreArray((*[2]float64)(unsafe.Pointer(outBase0 + uintptr(outputVecWidth)*oSize)))
				//alt:f64 o1_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase1)))
				//alt:f64 o1_1.StoreArray((*[2]float64)(unsafe.Pointer(outBase1 + uintptr(outputVecWidth)*oSize)))
				//alt:f64 o2_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase2)))
				//alt:f64 o2_1.StoreArray((*[2]float64)(unsafe.Pointer(outBase2 + uintptr(outputVecWidth)*oSize)))
				//alt:f64 o3_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase3)))
				//alt:f64 o3_1.StoreArray((*[2]float64)(unsafe.Pointer(outBase3 + uintptr(outputVecWidth)*oSize)))
			}

			for ; col+outputVecWidth <= rhsCrossSize; col += outputVecWidth { //alt:f32
				//alt:f64 for ; col+outputVecWidth <= rhsCrossSize; col += outputVecWidth {
				zero := archsimd.BroadcastFloat32x4(0.0) //alt:f32
				//alt:f64 zero := archsimd.BroadcastFloat64x2(0.0)
				o0_0, o1_0, o2_0, o3_0 := zero, zero, zero, zero //alt:f32|f64

				for k := range contractingSize { //alt:f32|f64
					kOffset := uintptr(k) * iSize                                                       //alt:f32|f64
					l0 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow0Base + kOffset))) //alt:f32
					l1 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow1Base + kOffset))) //alt:f32
					l2 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow2Base + kOffset))) //alt:f32
					l3 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRow3Base + kOffset))) //alt:f32
					//alt:f64 l0 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow0Base + kOffset)))
					//alt:f64 l1 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow1Base + kOffset)))
					//alt:f64 l2 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow2Base + kOffset)))
					//alt:f64 l3 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRow3Base + kOffset)))

					rColBase := rhsBase + uintptr(k*rhsCrossSize+col)*iSize         //alt:f32|f64
					r0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rColBase))) //alt:f32
					//alt:f64 r0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rColBase)))

					o0_0 = l0.MulAdd(r0, o0_0) //alt:f32|f64
					o1_0 = l1.MulAdd(r0, o1_0) //alt:f32|f64
					o2_0 = l2.MulAdd(r0, o2_0) //alt:f32|f64
					o3_0 = l3.MulAdd(r0, o3_0) //alt:f32|f64
				} //alt:f32|f64

				outBase0 := outputBase + uintptr((row+0)*rhsCrossSize+col)*oSize //alt:f32|f64
				outBase1 := outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize //alt:f32|f64
				outBase2 := outputBase + uintptr((row+2)*rhsCrossSize+col)*oSize //alt:f32|f64
				outBase3 := outputBase + uintptr((row+3)*rhsCrossSize+col)*oSize //alt:f32|f64

				o0_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase0))) //alt:f32
				o1_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase1))) //alt:f32
				o2_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase2))) //alt:f32
				o3_0.StoreArray((*[4]float32)(unsafe.Pointer(outBase3))) //alt:f32
				//alt:f64 o0_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase0)))
				//alt:f64 o1_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase1)))
				//alt:f64 o2_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase2)))
				//alt:f64 o3_0.StoreArray((*[2]float64)(unsafe.Pointer(outBase3)))
			} //alt:f32|f64

			for ; col < rhsCrossSize; col++ {
				var s0, s1, s2, s3 float32 //alt:f32|bf16|f16
				//alt:f64 var s0, s1, s2, s3 float64
				for k := range contractingSize {
					kOffset := uintptr(k) * iSize
					l0 := *(*float32)(unsafe.Pointer(lRow0Base + kOffset)) //alt:f32
					l1 := *(*float32)(unsafe.Pointer(lRow1Base + kOffset)) //alt:f32
					l2 := *(*float32)(unsafe.Pointer(lRow2Base + kOffset)) //alt:f32
					l3 := *(*float32)(unsafe.Pointer(lRow3Base + kOffset)) //alt:f32
					//alt:bf16 l0 := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow0Base + kOffset))).Float32()
					//alt:bf16 l1 := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow1Base + kOffset))).Float32()
					//alt:bf16 l2 := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow2Base + kOffset))).Float32()
					//alt:bf16 l3 := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow3Base + kOffset))).Float32()
					//alt:f16 l0 := (*(*float16.Float16)(unsafe.Pointer(lRow0Base + kOffset))).Float32()
					//alt:f16 l1 := (*(*float16.Float16)(unsafe.Pointer(lRow1Base + kOffset))).Float32()
					//alt:f16 l2 := (*(*float16.Float16)(unsafe.Pointer(lRow2Base + kOffset))).Float32()
					//alt:f16 l3 := (*(*float16.Float16)(unsafe.Pointer(lRow3Base + kOffset))).Float32()
					//alt:f64 l0 := *(*float64)(unsafe.Pointer(lRow0Base + kOffset))
					//alt:f64 l1 := *(*float64)(unsafe.Pointer(lRow1Base + kOffset))
					//alt:f64 l2 := *(*float64)(unsafe.Pointer(lRow2Base + kOffset))
					//alt:f64 l3 := *(*float64)(unsafe.Pointer(lRow3Base + kOffset))

					r0 := *(*float32)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize)) //alt:f32
					//alt:bf16 r0 := (*(*bfloat16.BFloat16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize))).Float32()
					//alt:f16 r0 := (*(*float16.Float16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize))).Float32()
					//alt:f64 r0 := *(*float64)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize))

					s0 += l0 * r0
					s1 += l1 * r0
					s2 += l2 * r0
					s3 += l3 * r0
				}
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*oSize)) = s0 //alt:f32|bf16|f16
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize)) = s1 //alt:f32|bf16|f16
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*oSize)) = s2 //alt:f32|bf16|f16
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*oSize)) = s3 //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*oSize)) = s0
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize)) = s1
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+2)*rhsCrossSize+col)*oSize)) = s2
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+3)*rhsCrossSize+col)*oSize)) = s3
			}
		}

		for ; row < lhsCrossSize; row++ {
			lRowBase := lhsBase + uintptr(row*contractingSize)*iSize
			const inputVecWidth = 4 //alt:f32
			//alt:bf16 const inputVecWidth = 8
			//alt:f16 const inputVecWidth = 8
			//alt:f64 const inputVecWidth = 2

			col := 0
			for ; col+inputVecWidth <= rhsCrossSize; col += inputVecWidth {
				o0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
				//alt:f64 o0 := archsimd.BroadcastFloat64x2(0.0)
				//alt:bf16 o1 := archsimd.BroadcastFloat32x4(0.0)
				//alt:f16 o1 := archsimd.BroadcastFloat32x4(0.0)

				for k := range contractingSize {
					kOffset := uintptr(k) * iSize
					l0 := archsimd.BroadcastFloat32x4(*(*float32)(unsafe.Pointer(lRowBase + kOffset))) //alt:f32
					//alt:bf16 l0 := archsimd.BroadcastFloat32x4((*(*bfloat16.BFloat16)(unsafe.Pointer(lRowBase + kOffset))).Float32())
					//alt:f16 l0 := archsimd.BroadcastFloat32x4((*(*float16.Float16)(unsafe.Pointer(lRowBase + kOffset))).Float32())
					//alt:f64 l0 := archsimd.BroadcastFloat64x2(*(*float64)(unsafe.Pointer(lRowBase + kOffset)))

					rColBase := rhsBase + uintptr(k*rhsCrossSize+col)*iSize
					var r0 archsimd.Float32x4 //alt:f32|bf16|f16
					//alt:f64 var r0 archsimd.Float64x2
					//alt:bf16 var r1 archsimd.Float32x4
					//alt:f16 var r1 archsimd.Float32x4
					{ //alt:f32|bf16|f16|f64
						r0 = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rColBase))) //alt:f32
						//alt:bf16 r0_bf16 := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(rColBase)))
						//alt:bf16 r0, r1 = r0_bf16.ToFloat32()
						//alt:f16 r0_f16 := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(rColBase)))
						//alt:f16 r0, r1 = r0_f16.ToFloat32()
						//alt:f64 r0 = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rColBase)))
					} //alt:f32|bf16|f16|f64

					o0 = l0.MulAdd(r0, o0)
					//alt:bf16 o1 = l0.MulAdd(r1, o1)
					//alt:f16 o1 = l0.MulAdd(r1, o1)
				}

				outBase := outputBase + uintptr(row*rhsCrossSize+col)*oSize
				o0.StoreArray((*[4]float32)(unsafe.Pointer(outBase))) //alt:f32|bf16|f16
				//alt:f64 o0.StoreArray((*[2]float64)(unsafe.Pointer(outBase)))
				//alt:bf16 o1.StoreArray((*[4]float32)(unsafe.Pointer(outBase + uintptr(outputVecWidth)*oSize)))
				//alt:f16 o1.StoreArray((*[4]float32)(unsafe.Pointer(outBase + uintptr(outputVecWidth)*oSize)))
			}

			for ; col < rhsCrossSize; col++ {
				var s float32 //alt:f32|bf16|f16
				//alt:f64 var s float64
				for k := range contractingSize {
					kOffset := uintptr(k) * iSize
					l0 := *(*float32)(unsafe.Pointer(lRowBase + kOffset)) //alt:f32
					//alt:bf16 l0 := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRowBase + kOffset))).Float32()
					//alt:f16 l0 := (*(*float16.Float16)(unsafe.Pointer(lRowBase + kOffset))).Float32()
					//alt:f64 l0 := *(*float64)(unsafe.Pointer(lRowBase + kOffset))

					r0 := *(*float32)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize)) //alt:f32
					//alt:bf16 r0 := (*(*bfloat16.BFloat16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize))).Float32()
					//alt:f16 r0 := (*(*float16.Float16)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize))).Float32()
					//alt:f64 r0 := *(*float64)(unsafe.Pointer(rhsBase + uintptr(k*rhsCrossSize+col)*iSize))

					s += l0 * r0
				}
				*(*float32)(unsafe.Pointer(outputBase + uintptr(row*rhsCrossSize+col)*oSize)) = s //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr(row*rhsCrossSize+col)*oSize)) = s
			}
		}

		lhsBase += lhsByteStride
		rhsBase += rhsByteStride
		outputBase += outputByteStride
	}
}

// neonSmallFloat32Transposed implements a NEON matrix multiplication for small inputs,
// without data packing. It vectorizes the contracting dimension.
// Layout is dot.LayoutTransposed.
func neonSmallFloat32Transposed( //alt:f32
	//alt:bf16 func neonSmallBFloat16Transposed(
	//alt:f16 func neonSmallFloat16Transposed(
	//alt:f64 func neonSmallFloat64Transposed(
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

	if NEONUseAsm {
		neonSmallFloat32TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output) //alt:f32
		//alt:bf16 neonSmallBFloat16TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
		//alt:f16 neonSmallFloat16TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
		//alt:f64 neonSmallFloat64TransposedAsm(lhs, rhs, batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
		return
	}

	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize

	var iZero float32 //alt:f32
	//alt:bf16 var iZero bfloat16.BFloat16
	//alt:f16 var iZero float16.Float16
	//alt:f64 var iZero float64
	iSize := unsafe.Sizeof(iZero)
	var oZero float32 //alt:f32|bf16|f16
	//alt:f64 var oZero float64
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

	const vecWidth = 4 //alt:f32
	//alt:bf16 const vecWidth = 8
	//alt:f16 const vecWidth = 8
	//alt:f64 const vecWidth = 2

	for range batchCount {
		row := 0
		for ; row+1 < lhsCrossSize; row += 2 {
			lRow0Base := lhsBase + uintptr((row+0)*contractingSize)*iSize
			lRow1Base := lhsBase + uintptr((row+1)*contractingSize)*iSize

			col := 0
			for ; col+1 < rhsCrossSize; col += 2 {
				rCol0Base := rhsBase + uintptr((col+0)*contractingSize)*iSize
				rCol1Base := rhsBase + uintptr((col+1)*contractingSize)*iSize

				zero := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
				//alt:f64 zero := archsimd.BroadcastFloat64x2(0.0)
				acc00, acc01, acc10, acc11 := zero, zero, zero, zero

				k := 0
				for ; k+vecWidth <= contractingSize; k += vecWidth {
					kOffset := uintptr(k) * iSize
					l0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(lRow0Base + kOffset))) //alt:f32
					l1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(lRow1Base + kOffset))) //alt:f32
					r0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rCol0Base + kOffset))) //alt:f32
					r1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rCol1Base + kOffset))) //alt:f32
					//alt:bf16 l0a, l0b := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(lRow0Base + kOffset))).ToFloat32()
					//alt:bf16 l1a, l1b := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(lRow1Base + kOffset))).ToFloat32()
					//alt:bf16 r0a, r0b := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(rCol0Base + kOffset))).ToFloat32()
					//alt:bf16 r1a, r1b := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(rCol1Base + kOffset))).ToFloat32()
					//alt:f16 l0a, l0b := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(lRow0Base + kOffset))).ToFloat32()
					//alt:f16 l1a, l1b := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(lRow1Base + kOffset))).ToFloat32()
					//alt:f16 r0a, r0b := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(rCol0Base + kOffset))).ToFloat32()
					//alt:f16 r1a, r1b := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(rCol1Base + kOffset))).ToFloat32()
					//alt:f64 l0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(lRow0Base + kOffset)))
					//alt:f64 l1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(lRow1Base + kOffset)))
					//alt:f64 r0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rCol0Base + kOffset)))
					//alt:f64 r1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rCol1Base + kOffset)))

					acc00 = l0.MulAdd(r0, acc00) //alt:f32|f64
					acc01 = l0.MulAdd(r1, acc01) //alt:f32|f64
					acc10 = l1.MulAdd(r0, acc10) //alt:f32|f64
					acc11 = l1.MulAdd(r1, acc11) //alt:f32|f64
					//alt:bf16 acc00 = l0b.MulAdd(r0b, l0a.MulAdd(r0a, acc00))
					//alt:bf16 acc01 = l0b.MulAdd(r1b, l0a.MulAdd(r1a, acc01))
					//alt:bf16 acc10 = l1b.MulAdd(r0b, l1a.MulAdd(r0a, acc10))
					//alt:bf16 acc11 = l1b.MulAdd(r1b, l1a.MulAdd(r1a, acc11))
					//alt:f16 acc00 = l0b.MulAdd(r0b, l0a.MulAdd(r0a, acc00))
					//alt:f16 acc01 = l0b.MulAdd(r1b, l0a.MulAdd(r1a, acc01))
					//alt:f16 acc10 = l1b.MulAdd(r0b, l1a.MulAdd(r0a, acc10))
					//alt:f16 acc11 = l1b.MulAdd(r1b, l1a.MulAdd(r1a, acc11))
				}

				res00 := neonReduceSumFloat32x4(acc00) //alt:f32|bf16|f16
				res01 := neonReduceSumFloat32x4(acc01) //alt:f32|bf16|f16
				res10 := neonReduceSumFloat32x4(acc10) //alt:f32|bf16|f16
				res11 := neonReduceSumFloat32x4(acc11) //alt:f32|bf16|f16
				//alt:f64 res00 := neonReduceSumFloat64x2(acc00)
				//alt:f64 res01 := neonReduceSumFloat64x2(acc01)
				//alt:f64 res10 := neonReduceSumFloat64x2(acc10)
				//alt:f64 res11 := neonReduceSumFloat64x2(acc11)

				for ; k < contractingSize; k++ {
					kOffset := uintptr(k) * iSize
					l0Val := *(*float32)(unsafe.Pointer(lRow0Base + kOffset)) //alt:f32
					l1Val := *(*float32)(unsafe.Pointer(lRow1Base + kOffset)) //alt:f32
					r0Val := *(*float32)(unsafe.Pointer(rCol0Base + kOffset)) //alt:f32
					r1Val := *(*float32)(unsafe.Pointer(rCol1Base + kOffset)) //alt:f32
					//alt:bf16 l0Val := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow0Base + kOffset))).Float32()
					//alt:bf16 l1Val := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow1Base + kOffset))).Float32()
					//alt:bf16 r0Val := (*(*bfloat16.BFloat16)(unsafe.Pointer(rCol0Base + kOffset))).Float32()
					//alt:bf16 r1Val := (*(*bfloat16.BFloat16)(unsafe.Pointer(rCol1Base + kOffset))).Float32()
					//alt:f16 l0Val := (*(*float16.Float16)(unsafe.Pointer(lRow0Base + kOffset))).Float32()
					//alt:f16 l1Val := (*(*float16.Float16)(unsafe.Pointer(lRow1Base + kOffset))).Float32()
					//alt:f16 r0Val := (*(*float16.Float16)(unsafe.Pointer(rCol0Base + kOffset))).Float32()
					//alt:f16 r1Val := (*(*float16.Float16)(unsafe.Pointer(rCol1Base + kOffset))).Float32()
					//alt:f64 l0Val := *(*float64)(unsafe.Pointer(lRow0Base + kOffset))
					//alt:f64 l1Val := *(*float64)(unsafe.Pointer(lRow1Base + kOffset))
					//alt:f64 r0Val := *(*float64)(unsafe.Pointer(rCol0Base + kOffset))
					//alt:f64 r1Val := *(*float64)(unsafe.Pointer(rCol1Base + kOffset))
					res00 += l0Val * r0Val
					res01 += l0Val * r1Val
					res10 += l1Val * r0Val
					res11 += l1Val * r1Val
				}

				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col+0)*oSize)) = res00 //alt:f32|bf16|f16
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col+1)*oSize)) = res01 //alt:f32|bf16|f16
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col+0)*oSize)) = res10 //alt:f32|bf16|f16
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col+1)*oSize)) = res11 //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col+0)*oSize)) = res00
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col+1)*oSize)) = res01
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col+0)*oSize)) = res10
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col+1)*oSize)) = res11
			}

			for ; col < rhsCrossSize; col++ {
				rCol0Base := rhsBase + uintptr(col*contractingSize)*iSize
				zero := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
				//alt:f64 zero := archsimd.BroadcastFloat64x2(0.0)
				acc00, acc10 := zero, zero
				k := 0
				for ; k+vecWidth <= contractingSize; k += vecWidth {
					kOffset := uintptr(k) * iSize
					l0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(lRow0Base + kOffset))) //alt:f32
					l1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(lRow1Base + kOffset))) //alt:f32
					r0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rCol0Base + kOffset))) //alt:f32
					//alt:bf16 l0a, l0b := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(lRow0Base + kOffset))).ToFloat32()
					//alt:bf16 l1a, l1b := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(lRow1Base + kOffset))).ToFloat32()
					//alt:bf16 r0a, r0b := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(rCol0Base + kOffset))).ToFloat32()
					//alt:f16 l0a, l0b := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(lRow0Base + kOffset))).ToFloat32()
					//alt:f16 l1a, l1b := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(lRow1Base + kOffset))).ToFloat32()
					//alt:f16 r0a, r0b := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(rCol0Base + kOffset))).ToFloat32()
					//alt:f64 l0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(lRow0Base + kOffset)))
					//alt:f64 l1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(lRow1Base + kOffset)))
					//alt:f64 r0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rCol0Base + kOffset)))

					acc00 = l0.MulAdd(r0, acc00) //alt:f32|f64
					acc10 = l1.MulAdd(r0, acc10) //alt:f32|f64
					//alt:bf16 acc00 = l0b.MulAdd(r0b, l0a.MulAdd(r0a, acc00))
					//alt:bf16 acc10 = l1b.MulAdd(r0b, l1a.MulAdd(r0a, acc10))
					//alt:f16 acc00 = l0b.MulAdd(r0b, l0a.MulAdd(r0a, acc00))
					//alt:f16 acc10 = l1b.MulAdd(r0b, l1a.MulAdd(r0a, acc10))
				}
				res00 := neonReduceSumFloat32x4(acc00) //alt:f32|bf16|f16
				res10 := neonReduceSumFloat32x4(acc10) //alt:f32|bf16|f16
				//alt:f64 res00 := neonReduceSumFloat64x2(acc00)
				//alt:f64 res10 := neonReduceSumFloat64x2(acc10)
				for ; k < contractingSize; k++ {
					kOffset := uintptr(k) * iSize
					l0Val := *(*float32)(unsafe.Pointer(lRow0Base + kOffset)) //alt:f32
					l1Val := *(*float32)(unsafe.Pointer(lRow1Base + kOffset)) //alt:f32
					r0Val := *(*float32)(unsafe.Pointer(rCol0Base + kOffset)) //alt:f32
					//alt:bf16 l0Val := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow0Base + kOffset))).Float32()
					//alt:bf16 l1Val := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRow1Base + kOffset))).Float32()
					//alt:bf16 r0Val := (*(*bfloat16.BFloat16)(unsafe.Pointer(rCol0Base + kOffset))).Float32()
					//alt:f16 l0Val := (*(*float16.Float16)(unsafe.Pointer(lRow0Base + kOffset))).Float32()
					//alt:f16 l1Val := (*(*float16.Float16)(unsafe.Pointer(lRow1Base + kOffset))).Float32()
					//alt:f16 r0Val := (*(*float16.Float16)(unsafe.Pointer(rCol0Base + kOffset))).Float32()
					//alt:f64 l0Val := *(*float64)(unsafe.Pointer(lRow0Base + kOffset))
					//alt:f64 l1Val := *(*float64)(unsafe.Pointer(lRow1Base + kOffset))
					//alt:f64 r0Val := *(*float64)(unsafe.Pointer(rCol0Base + kOffset))
					res00 += l0Val * r0Val
					res10 += l1Val * r0Val
				}
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*oSize)) = res00 //alt:f32|bf16|f16
				*(*float32)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize)) = res10 //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+0)*rhsCrossSize+col)*oSize)) = res00
				//alt:f64 *(*float64)(unsafe.Pointer(outputBase + uintptr((row+1)*rhsCrossSize+col)*oSize)) = res10
			}
		}

		for ; row < lhsCrossSize; row++ {
			lRowBase := lhsBase + uintptr(row*contractingSize)*iSize
			for col := range rhsCrossSize {
				rColBase := rhsBase + uintptr(col*contractingSize)*iSize
				acc := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
				//alt:f64 acc := archsimd.BroadcastFloat64x2(0.0)
				k := 0
				for ; k+vecWidth <= contractingSize; k += vecWidth {
					kOffset := uintptr(k) * iSize
					lVal := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(lRowBase + kOffset))) //alt:f32
					rVal := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(rColBase + kOffset))) //alt:f32
					//alt:bf16 la, lb := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + kOffset))).ToFloat32()
					//alt:bf16 ra, rb := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(rColBase + kOffset))).ToFloat32()
					//alt:f16 la, lb := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(lRowBase + kOffset))).ToFloat32()
					//alt:f16 ra, rb := float16.LoadFloat16x8((*[8]float16.Float16)(unsafe.Pointer(rColBase + kOffset))).ToFloat32()
					//alt:f64 lVal := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(lRowBase + kOffset)))
					//alt:f64 rVal := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(rColBase + kOffset)))
					acc = lVal.MulAdd(rVal, acc) //alt:f32|f64
					//alt:bf16 acc = lb.MulAdd(rb, la.MulAdd(ra, acc))
					//alt:f16 acc = lb.MulAdd(rb, la.MulAdd(ra, acc))
				}

				res := neonReduceSumFloat32x4(acc) //alt:f32|bf16|f16
				//alt:f64 res := neonReduceSumFloat64x2(acc)
				for ; k < contractingSize; k++ {
					kOffset := uintptr(k) * iSize
					lVal := *(*float32)(unsafe.Pointer(lRowBase + kOffset)) //alt:f32
					//alt:bf16 lVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRowBase + kOffset))).Float32()
					//alt:f16 lVal := (*(*float16.Float16)(unsafe.Pointer(lRowBase + kOffset))).Float32()
					//alt:f64 lVal := *(*float64)(unsafe.Pointer(lRowBase + kOffset))
					rVal := *(*float32)(unsafe.Pointer(rColBase + kOffset)) //alt:f32
					//alt:bf16 rVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + kOffset))).Float32()
					//alt:f16 rVal := (*(*float16.Float16)(unsafe.Pointer(rColBase + kOffset))).Float32()
					//alt:f64 rVal := *(*float64)(unsafe.Pointer(rColBase + kOffset))
					res += lVal * rVal
				}

				outputIdx := outputBase + uintptr(row*rhsCrossSize+col)*oSize
				*(*float32)(unsafe.Pointer(outputIdx)) = res //alt:f32|bf16|f16
				//alt:f64 *(*float64)(unsafe.Pointer(outputIdx)) = res
			}
		}

		lhsBase += lhsByteStride
		rhsBase += rhsByteStride
		outputBase += outputByteStride
	}
}

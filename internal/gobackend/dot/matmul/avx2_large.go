// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package matmul

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/pkg/errors"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
	//alt:f16 "github.com/gomlx/compute/dtypes/float16"
)

// avx2LargeFloat32 implements a "packing" version of the matrix multiplication for larger inputs.
func avx2LargeFloat32( //alt:f32
	//alt:bf16 func avx2LargeBFloat16(
	//alt:f16 func avx2LargeFloat16(
	//alt:f64 func avx2LargeFloat64(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32) { //alt:f32|bf16|f16
	//alt:f64 output []float64) {

	params := AVX2ParamsFloat32 //alt:f32
	//alt:bf16 params := AVX2ParamsBFloat16
	//alt:f16 params := AVX2ParamsFloat16
	//alt:f64 params := AVX2ParamsFloat64

	maxWorkers := 1
	if backend.Workers != nil {
		maxWorkers = backend.Workers.AdjustedMaxParallelism()
	}

	if maxWorkers <= 1 {
		for batchIdx := range batchSize {
			avx2LargeMatrixSliceFloat32( //alt:f32
				//alt:bf16 avx2LargeMatrixSliceBFloat16(
				//alt:f16 avx2LargeMatrixSliceFloat16(
				//alt:f64 avx2LargeMatrixSliceFloat64(
				lhs, rhs, output, batchIdx, lhsCrossSize, rhsCrossSize, contractingSize, layout, params)
		}
		return
	}

	work := make(chan int, batchSize)
	for batchIdx := range batchSize {
		work <- batchIdx
	}
	close(work)

	backend.Workers.Saturate(func() {
		for batchIdx := range work {
			avx2LargeMatrixSliceFloat32( //alt:f32
				//alt:bf16 avx2LargeMatrixSliceBFloat16(
				//alt:f16 avx2LargeMatrixSliceFloat16(
				//alt:f64 avx2LargeMatrixSliceFloat64(
				lhs, rhs, output, batchIdx, lhsCrossSize, rhsCrossSize, contractingSize, layout, params)
		}
	})
}

func avx2LargeMatrixSliceFloat32( //alt:f32
	//alt:bf16 func avx2LargeMatrixSliceBFloat16(
	//alt:f16 func avx2LargeMatrixSliceFloat16(
	//alt:f64 func avx2LargeMatrixSliceFloat64(
	lhs, rhs []float32, output []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16, output []float32,
	//alt:f16 lhs, rhs []float16.Float16, output []float32,
	//alt:f64 lhs, rhs, output []float64,
	batchIdx, lhsCrossSize, rhsCrossSize, contractingSize int,
	layout dot.Layout, params CacheParams) {

	if lhsCrossSize == 0 || rhsCrossSize == 0 || contractingSize == 0 {
		return
	}

	if params.LHSL1KernelRows != 4 || params.RHSL1KernelCols != 16 { //alt:f32|bf16|f16
		//alt:f64 if params.LHSL1KernelRows != 4 || params.RHSL1KernelCols != 8 {
		panic(errors.Errorf("unsupported kernel L1 block sizes for avx2 kernel: lhsL1BlockRows=%d, rhsL1BlockCols=%d, wanted 4 and 16 respectively (params=%+v)", //alt:f32|bf16|f16
			//alt:f64 panic(errors.Errorf("unsupported kernel L1 block sizes for avx2 kernel: lhsL1BlockRows=%d, rhsL1BlockCols=%d, wanted 4 and 8 respectively (params=%+v)",
			params.LHSL1KernelRows, params.RHSL1KernelCols, params))
	}

	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize
	lhsMatrix := lhs[batchIdx*lhsStride : (batchIdx+1)*lhsStride]
	rhsMatrix := rhs[batchIdx*rhsStride : (batchIdx+1)*rhsStride]
	outputMatrix := output[batchIdx*outputStride : (batchIdx+1)*outputStride]

	packedRHS := make([]float32, params.PanelContractingSize*params.RHSPanelCrossSize) //alt:f32
	//alt:bf16 packedRHS := make([]bfloat16.BFloat16, params.PanelContractingSize*params.RHSPanelCrossSize)
	//alt:f16 packedRHS := make([]float16.Float16, params.PanelContractingSize*params.RHSPanelCrossSize)
	//alt:f64 packedRHS := make([]float64, params.PanelContractingSize*params.RHSPanelCrossSize)
	packedLHS := make([]float32, params.LHSPanelCrossSize*params.PanelContractingSize) //alt:f32
	//alt:bf16 packedLHS := make([]bfloat16.BFloat16, params.LHSPanelCrossSize*params.PanelContractingSize)
	//alt:f16 packedLHS := make([]float16.Float16, params.LHSPanelCrossSize*params.PanelContractingSize)
	//alt:f64 packedLHS := make([]float64, params.LHSPanelCrossSize*params.PanelContractingSize)
	packedOutput := make([]float32, params.LHSPanelCrossSize*params.RHSL1KernelCols) //alt:f32|bf16|f16
	//alt:f64 packedOutput := make([]float64, params.LHSPanelCrossSize*params.RHSL1KernelCols)

	for rhsPanelColIdx := 0; rhsPanelColIdx < rhsCrossSize; rhsPanelColIdx += params.RHSPanelCrossSize {
		rhsPanelWidth := min(params.RHSPanelCrossSize, rhsCrossSize-rhsPanelColIdx)

		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			isFirstContractingPanel := contractingPanelIdx == 0

			if layout == dot.LayoutNonTransposed {
				avx2PackRHSNonTransposed(rhsMatrix, packedRHS, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)
			} else {
				// RHS is [N, K]. Contracting dimension is the last one (K).
				// We want to pack it into strips of width Nr (usually 16 for f32).
				// This is like packing LHS with Mr=Nr.
				packLHS(rhsMatrix, packedRHS, rhsPanelColIdx, contractingPanelIdx, contractingSize, rhsPanelWidth, contractingPanelWidth, params.RHSL1KernelCols)
			}

			for lhsPanelRowIdx := 0; lhsPanelRowIdx < lhsCrossSize; lhsPanelRowIdx += params.LHSPanelCrossSize {
				lhsPanelHeight := min(params.LHSPanelCrossSize, lhsCrossSize-lhsPanelRowIdx)

				avx2PackLHSKernelRows4(lhsMatrix, packedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)

				for rhsKernelColIdx := 0; rhsKernelColIdx < rhsPanelWidth; rhsKernelColIdx += params.RHSL1KernelCols {
					avx2LargeKernelFloat32( //alt:f32
						//alt:bf16 avx2LargeKernelBFloat16(
						//alt:f16 avx2LargeKernelFloat16(
						//alt:f64 avx2LargeKernelFloat64(
						packedLHS, packedRHS, packedOutput, lhsPanelHeight, contractingPanelWidth, rhsKernelColIdx, params)

					avx2ApplyPackedOutputFloat32( //alt:f32|bf16|f16
						//alt:f64 avx2ApplyPackedOutputFloat64(
						packedOutput, outputMatrix, isFirstContractingPanel, params.RHSL1KernelCols, lhsPanelRowIdx, rhsPanelColIdx+rhsKernelColIdx, rhsCrossSize, lhsPanelHeight, min(params.RHSL1KernelCols, rhsPanelWidth-rhsKernelColIdx))
				}
			}
		}
	}
}

func avx2LargeKernelFloat32( //alt:f32
	//alt:bf16 func avx2LargeKernelBFloat16(
	//alt:f16 func avx2LargeKernelFloat16(
	//alt:f64 func avx2LargeKernelFloat64(
	packedLHS, packedRHS []float32, packedOutput []float32, //alt:f32
	//alt:bf16 packedLHS, packedRHS []bfloat16.BFloat16, packedOutput []float32,
	//alt:f16 packedLHS, packedRHS []float16.Float16, packedOutput []float32,
	//alt:f64 packedLHS, packedRHS, packedOutput []float64,
	lhsPanelHeight, contractingPanelWidth, rhsKernelColIdx int, params CacheParams) {

	for i := range len(packedOutput) {
		packedOutput[i] = 0
	}

	lhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedLHS)))
	rhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedRHS))) + uintptr(rhsKernelColIdx*contractingPanelWidth)*unsafe.Sizeof(packedRHS[0])

	for row := 0; row < lhsPanelHeight; row += 4 {
		// Load accumulation registers
		var o0_0, o0_1, o1_0, o1_1, o2_0, o2_1, o3_0, o3_1 archsimd.Float32x8 //alt:f32|bf16|f16
		//alt:f64 var o0_0, o0_1, o1_0, o1_1, o2_0, o2_1, o3_0, o3_1 archsimd.Float64x4

		for k := 0; k < contractingPanelWidth; k++ {
			// Load LHS: 4 values for 4 rows
			l_base := lhsPtr + uintptr(row*contractingPanelWidth+k*4)*unsafe.Sizeof(packedLHS[0])
			l0 := archsimd.BroadcastFloat32x8(*(*float32)(unsafe.Pointer(l_base)))      //alt:f32
			l1 := archsimd.BroadcastFloat32x8(*(*float32)(unsafe.Pointer(l_base + 4)))  //alt:f32
			l2 := archsimd.BroadcastFloat32x8(*(*float32)(unsafe.Pointer(l_base + 8)))  //alt:f32
			l3 := archsimd.BroadcastFloat32x8(*(*float32)(unsafe.Pointer(l_base + 12))) //alt:f32
			//alt:bf16 l0 := archsimd.BroadcastFloat32x8((*(*bfloat16.BFloat16)(unsafe.Pointer(l_base))).Float32())
			//alt:bf16 l1 := archsimd.BroadcastFloat32x8((*(*bfloat16.BFloat16)(unsafe.Pointer(l_base + 2))).Float32())
			//alt:bf16 l2 := archsimd.BroadcastFloat32x8((*(*bfloat16.BFloat16)(unsafe.Pointer(l_base + 4))).Float32())
			//alt:bf16 l3 := archsimd.BroadcastFloat32x8((*(*bfloat16.BFloat16)(unsafe.Pointer(l_base + 6))).Float32())
			//alt:f16 l0 := archsimd.BroadcastFloat32x8((*(*float16.Float16)(unsafe.Pointer(l_base))).Float32())
			//alt:f16 l1 := archsimd.BroadcastFloat32x8((*(*float16.Float16)(unsafe.Pointer(l_base + 2))).Float32())
			//alt:f16 l2 := archsimd.BroadcastFloat32x8((*(*float16.Float16)(unsafe.Pointer(l_base + 4))).Float32())
			//alt:f16 l3 := archsimd.BroadcastFloat32x8((*(*float16.Float16)(unsafe.Pointer(l_base + 6))).Float32())
			//alt:f64 l0 := archsimd.BroadcastFloat64x4(*(*float64)(unsafe.Pointer(l_base)))
			//alt:f64 l1 := archsimd.BroadcastFloat64x4(*(*float64)(unsafe.Pointer(l_base + 8)))
			//alt:f64 l2 := archsimd.BroadcastFloat64x4(*(*float64)(unsafe.Pointer(l_base + 16)))
			//alt:f64 l3 := archsimd.BroadcastFloat64x4(*(*float64)(unsafe.Pointer(l_base + 24)))

			// Load RHS: 16 values (2 YMM registers)
			r_base := rhsPtr + uintptr(k*16)*unsafe.Sizeof(packedRHS[0]) //alt:f32|bf16|f16
			//alt:f64 r_base := rhsPtr + uintptr(k*8)*unsafe.Sizeof(packedRHS[0])
			r0 := archsimd.LoadFloat32x8((*[8]float32)(unsafe.Pointer(r_base)))      //alt:f32
			r1 := archsimd.LoadFloat32x8((*[8]float32)(unsafe.Pointer(r_base + 32))) //alt:f32
			//alt:bf16 r0, r1 := bfloat16.LoadBFloat16x16((*[16]bfloat16.BFloat16)(unsafe.Pointer(r_base))).ToFloat32()
			//alt:f16 r0, r1 := float16.LoadFloat16x16((*[16]float16.Float16)(unsafe.Pointer(r_base))).ToFloat32()
			//alt:f64 r0 := archsimd.LoadFloat64x4((*[4]float64)(unsafe.Pointer(r_base)))
			//alt:f64 r1 := archsimd.LoadFloat64x4((*[4]float64)(unsafe.Pointer(r_base + 32)))

			o0_0 = l0.MulAdd(r0, o0_0)
			o0_1 = l0.MulAdd(r1, o0_1)
			o1_0 = l1.MulAdd(r0, o1_0)
			o1_1 = l1.MulAdd(r1, o1_1)
			o2_0 = l2.MulAdd(r0, o2_0)
			o2_1 = l2.MulAdd(r1, o2_1)
			o3_0 = l3.MulAdd(r0, o3_0)
			o3_1 = l3.MulAdd(r1, o3_1)
		}

		// Store accumulated values
		o_base := uintptr(row*16) * unsafe.Sizeof(packedOutput[0]) //alt:f32|bf16|f16
		//alt:f64 o_base := uintptr(row*8) * unsafe.Sizeof(packedOutput[0])
		outputPtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedOutput))) + o_base
		o0_0.Store((*[8]float32)(unsafe.Pointer(outputPtr)))       //alt:f32|bf16|f16
		o0_1.Store((*[8]float32)(unsafe.Pointer(outputPtr + 32)))  //alt:f32|bf16|f16
		o1_0.Store((*[8]float32)(unsafe.Pointer(outputPtr + 64)))  //alt:f32|bf16|f16
		o1_1.Store((*[8]float32)(unsafe.Pointer(outputPtr + 96)))  //alt:f32|bf16|f16
		o2_0.Store((*[8]float32)(unsafe.Pointer(outputPtr + 128))) //alt:f32|bf16|f16
		o2_1.Store((*[8]float32)(unsafe.Pointer(outputPtr + 160))) //alt:f32|bf16|f16
		o3_0.Store((*[8]float32)(unsafe.Pointer(outputPtr + 192))) //alt:f32|bf16|f16
		o3_1.Store((*[8]float32)(unsafe.Pointer(outputPtr + 224))) //alt:f32|bf16|f16
		//alt:f64 o0_0.Store((*[4]float64)(unsafe.Pointer(outputPtr)))
		//alt:f64 o0_1.Store((*[4]float64)(unsafe.Pointer(outputPtr + 32)))
		//alt:f64 o1_0.Store((*[4]float64)(unsafe.Pointer(outputPtr + 64)))
		//alt:f64 o1_1.Store((*[4]float64)(unsafe.Pointer(outputPtr + 96)))
		//alt:f64 o2_0.Store((*[4]float64)(unsafe.Pointer(outputPtr + 128)))
		//alt:f64 o2_1.Store((*[4]float64)(unsafe.Pointer(outputPtr + 160)))
		//alt:f64 o3_0.Store((*[4]float64)(unsafe.Pointer(outputPtr + 192)))
		//alt:f64 o3_1.Store((*[4]float64)(unsafe.Pointer(outputPtr + 224)))
	}
}

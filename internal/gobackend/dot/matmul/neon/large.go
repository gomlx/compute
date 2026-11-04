// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

// This file is both an implementation for the "f32" tag, and the source for the following alternatives
// (you will find annotations in the form "alt:f32" or "alt:bf16" or "alt:f32|bf16" for both)
//
// - "bf16": Implements the MatMul for BFloat16 -> Float32 dtypes.
// - "f16": Implements the MatMul for Float16 -> Float32 dtypes.
// - "f64": Implements the MatMul for Float64 -> Float64 dtypes.

import (
	"runtime"
	"simd/archsimd"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/pkg/errors"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
	//alt:f16 "github.com/gomlx/compute/dtypes/float16"
)

// neonLargeFloat32 implements a cache-tiled, panel-packed matrix multiplication using ARM64 NEON.
func neonLargeFloat32( //alt:f32
	//alt:bf16 func neonLargeBFloat16(
	//alt:f16 func neonLargeFloat16(
	//alt:f64 func neonLargeFloat64(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32, //alt:f32|bf16|f16
	//alt:f64 output []float64,
	nodeData *dot.NodeData,
) {
	params := NEONParamsFloat32 //alt:f32
	//alt:bf16 params := NEONParamsBFloat16
	//alt:f16 params := NEONParamsFloat16
	//alt:f64 params := NEONParamsFloat64
	neonLargeWithParamsFloat32(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params) //alt:f32
	//alt:bf16 neonLargeWithParamsBFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params)
	//alt:f16 neonLargeWithParamsFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params)
	//alt:f64 neonLargeWithParamsFloat64(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params)
}

// smeLargeFloat32 implements a cache-tiled, panel-packed matrix multiplication using ARM64 SME FMOPA outer products.
func smeLargeFloat32( //alt:f32
	//alt:bf16 func smeLargeBFloat16(
	//alt:f16 func smeLargeFloat16(
	//alt:f64 func smeLargeFloat64(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32, //alt:f32|bf16|f16
	//alt:f64 output []float64,
	nodeData *dot.NodeData,
) {
	params := SMEParamsFloat32 //alt:f32
	//alt:bf16 params := SMEParamsBFloat16
	//alt:f16 params := SMEParamsFloat16
	//alt:f64 params := SMEParamsFloat64
	if nodeData == nil || (nodeData.LHSConstantData == nil && nodeData.RHSConstantData == nil) {
		if params.LHSL1KernelRows == 32 && (lhsCrossSize < 32 || rhsCrossSize < 32) { //alt:f32|bf16|f16
			//alt:f64 if params.LHSL1KernelRows == 16 && (lhsCrossSize < 16 || rhsCrossSize < 16) {
			params.LHSL1KernelRows = 16 //alt:f32|bf16|f16
			params.RHSL1KernelCols = 16 //alt:f32|bf16|f16
			params.LHSPanelCrossSize = 32 //alt:f32|bf16|f16
			//alt:f64 params.LHSL1KernelRows = 8
			//alt:f64 params.RHSL1KernelCols = 8
			//alt:f64 params.LHSPanelCrossSize = 32
		}
	}
	neonLargeWithParamsFloat32(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params) //alt:f32
	//alt:bf16 neonLargeWithParamsBFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params)
	//alt:f16 neonLargeWithParamsFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params)
	//alt:f64 neonLargeWithParamsFloat64(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData, params)
}

func neonLargeWithParamsFloat32( //alt:f32
	//alt:bf16 func neonLargeWithParamsBFloat16(
	//alt:f16 func neonLargeWithParamsFloat16(
	//alt:f64 func neonLargeWithParamsFloat64(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32, //alt:f32|bf16|f16
	//alt:f64 output []float64,
	nodeData *dot.NodeData,
	params matmul.CacheParams,
) {
	flops := batchSize * lhsCrossSize * rhsCrossSize * contractingSize
	maxWorkers := min(
		backend.Scheduler.AdjustedMaxParallelism(),
		max(1, (flops+matmul.MinLargeMatMulFlopsPerWorker-1)/matmul.MinLargeMatMulFlopsPerWorker),
	)

	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	numKPanels := (contractingSize + params.PanelContractingSize - 1) / params.PanelContractingSize
	numColPanels := (rhsCrossSize + params.RHSPanelCrossSize - 1) / params.RHSPanelCrossSize
	rhsPanelsPerBatch := numKPanels * numColPanels
	numRowPanels := (lhsCrossSize + params.LHSPanelCrossSize - 1) / params.LHSPanelCrossSize
	lhsPanelsPerBatch := numKPanels * numRowPanels

	var cachedRHSPanels, cachedLHSPanels [][]float32 //alt:f32
	//alt:bf16 var cachedRHSPanels, cachedLHSPanels [][]bfloat16.BFloat16
	//alt:f16 var cachedRHSPanels, cachedLHSPanels [][]float16.Float16
	//alt:f64 var cachedRHSPanels, cachedLHSPanels [][]float64
	if nodeData != nil {
		if nodeData.RHSConstantData != nil {
			cachedRHSPanels = gobackend.GetOrSetCachedForm(nodeData.RHSConstantData, func() [][]float32 { //alt:f32
				//alt:bf16 cachedRHSPanels = gobackend.GetOrSetCachedForm(nodeData.RHSConstantData, func() [][]bfloat16.BFloat16 {
				//alt:f16 cachedRHSPanels = gobackend.GetOrSetCachedForm(nodeData.RHSConstantData, func() [][]float16.Float16 {
				//alt:f64 cachedRHSPanels = gobackend.GetOrSetCachedForm(nodeData.RHSConstantData, func() [][]float64 {
				_, panels := neonPrepackRHSFloat32(backend, layout, rhs, batchSize, rhsCrossSize, contractingSize, params) //alt:f32
				//alt:bf16 _, panels := neonPrepackRHSBFloat16(backend, layout, rhs, batchSize, rhsCrossSize, contractingSize, params)
				//alt:f16 _, panels := neonPrepackRHSFloat16(backend, layout, rhs, batchSize, rhsCrossSize, contractingSize, params)
				//alt:f64 _, panels := neonPrepackRHSFloat64(backend, layout, rhs, batchSize, rhsCrossSize, contractingSize, params)
				return panels
			})
		}

		if nodeData.LHSConstantData != nil {
			cachedLHSPanels = gobackend.GetOrSetCachedForm(nodeData.LHSConstantData, func() [][]float32 { //alt:f32
				//alt:bf16 cachedLHSPanels = gobackend.GetOrSetCachedForm(nodeData.LHSConstantData, func() [][]bfloat16.BFloat16 {
				//alt:f16 cachedLHSPanels = gobackend.GetOrSetCachedForm(nodeData.LHSConstantData, func() [][]float16.Float16 {
				//alt:f64 cachedLHSPanels = gobackend.GetOrSetCachedForm(nodeData.LHSConstantData, func() [][]float64 {
				_, panels := neonPrepackLHSFloat32(backend, lhs, batchSize, lhsCrossSize, contractingSize, params) //alt:f32
				//alt:bf16 _, panels := neonPrepackLHSBFloat16(backend, lhs, batchSize, lhsCrossSize, contractingSize, params)
				//alt:f16 _, panels := neonPrepackLHSFloat16(backend, lhs, batchSize, lhsCrossSize, contractingSize, params)
				//alt:f64 _, panels := neonPrepackLHSFloat64(backend, lhs, batchSize, lhsCrossSize, contractingSize, params)
				return panels
			})
		}
	}

	var workBuf [256]matmul.WorkItem
	var workItems []matmul.WorkItem
	if maxWorkers > 1 {
		workItems = matmul.AppendWorkItems(
			workBuf[:0], batchSize, lhsCrossSize, rhsCrossSize, &params, maxWorkers)
		maxWorkers = min(maxWorkers, len(workItems))
	}

	lhsPanelBufSize := params.LHSPanelCrossSize * params.PanelContractingSize
	lhsAllocPanels := 1
	if numColPanels > 1 {
		lhsAllocPanels = max(1, min(16, numRowPanels*numKPanels))
	}
	isSMEKernel := NEONUseAsm && params.LHSL1KernelRows >= 16 //alt:f32|bf16|f16
	//alt:f64 isSMEKernel := NEONUseAsm && params.LHSL1KernelRows >= 8 && params.RHSL1KernelCols >= 8
	needTempOutput := !isSMEKernel && ((lhsCrossSize%params.LHSL1KernelRows != 0) || (rhsCrossSize%params.RHSL1KernelCols != 0))

	if maxWorkers <= 1 {
		var (
			packedLHSRef *gobackend.Buffer
			packedLHS    []float32 //alt:f32
			//alt:bf16 packedLHS []bfloat16.BFloat16
			//alt:f16 packedLHS []float16.Float16
			//alt:f64 packedLHS []float64
		)
		if len(cachedLHSPanels) == 0 {
			var ok bool
			packedLHSRef, packedLHS, ok = GetBuffer[float32](backend, lhsAllocPanels*lhsPanelBufSize) //alt:f32
			//alt:bf16 packedLHSRef, packedLHS, ok = GetBuffer[bfloat16.BFloat16](backend, lhsAllocPanels*lhsPanelBufSize)
			//alt:f16 packedLHSRef, packedLHS, ok = GetBuffer[float16.Float16](backend, lhsAllocPanels*lhsPanelBufSize)
			//alt:f64 packedLHSRef, packedLHS, ok = GetBuffer[float64](backend, lhsAllocPanels*lhsPanelBufSize)
			if !ok {
				return
			}
			defer ReleaseBuffer(packedLHSRef)
		}
		var (
			packedRHSRef *gobackend.Buffer
			packedRHS    []float32 //alt:f32
			//alt:bf16 packedRHS []bfloat16.BFloat16
			//alt:f16 packedRHS []float16.Float16
			//alt:f64 packedRHS []float64
		)
		if len(cachedRHSPanels) == 0 {
			var ok bool
			packedRHSRef, packedRHS, ok = GetBuffer[float32](backend, params.PanelContractingSize*params.RHSPanelCrossSize) //alt:f32
			//alt:bf16 packedRHSRef, packedRHS, ok = GetBuffer[bfloat16.BFloat16](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
			//alt:f16 packedRHSRef, packedRHS, ok = GetBuffer[float16.Float16](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
			//alt:f64 packedRHSRef, packedRHS, ok = GetBuffer[float64](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
			if !ok {
				return
			}
			defer ReleaseBuffer(packedRHSRef)
		}
		var (
			packedOutputRef, accumOutputRef *gobackend.Buffer
			packedOutput, accumOutput       []float32 //alt:f32|bf16|f16
			//alt:f64 packedOutput, accumOutput []float64
		)
		if needTempOutput {
			var ok bool
			packedOutputRef, packedOutput, ok = GetBuffer[float32](backend, params.LHSPanelCrossSize*params.RHSPanelCrossSize) //alt:f32|bf16|f16
			//alt:f64 packedOutputRef, packedOutput, ok = GetBuffer[float64](backend, params.LHSPanelCrossSize*params.RHSPanelCrossSize)
			if !ok {
				return
			}
			defer ReleaseBuffer(packedOutputRef)
			if numKPanels > 1 {
				const maxAccumPanels = 16
				accumOutputRef, accumOutput, ok = GetBuffer[float32](backend, maxAccumPanels*params.LHSPanelCrossSize*params.RHSPanelCrossSize) //alt:f32|bf16|f16
				//alt:f64 accumOutputRef, accumOutput, ok = GetBuffer[float64](backend, maxAccumPanels*params.LHSPanelCrossSize*params.RHSPanelCrossSize)
				if !ok {
					return
				}
				defer ReleaseBuffer(accumOutputRef)
			}
		}
		lhsFlatIdx := 0
		rhsFlatIdx := 0
		outputFlatIdx := 0
		for b := range batchSize {
			var (
				batchLHS, batchRHS               []float32 //alt:f32
				//alt:bf16 batchLHS, batchRHS []bfloat16.BFloat16
				//alt:f16 batchLHS, batchRHS []float16.Float16
				//alt:f64 batchLHS, batchRHS []float64
				batchCachedLHSPanels [][]float32 //alt:f32
				//alt:bf16 batchCachedLHSPanels [][]bfloat16.BFloat16
				//alt:f16 batchCachedLHSPanels [][]float16.Float16
				//alt:f64 batchCachedLHSPanels [][]float64
				batchCachedRHSPanels [][]float32 //alt:f32
				//alt:bf16 batchCachedRHSPanels [][]bfloat16.BFloat16
				//alt:f16 batchCachedRHSPanels [][]float16.Float16
				//alt:f64 batchCachedRHSPanels [][]float64
			)
			if len(cachedLHSPanels) > 0 {
				batchCachedLHSPanels = cachedLHSPanels[b*lhsPanelsPerBatch : (b+1)*lhsPanelsPerBatch]
			} else {
				batchLHS = lhs[lhsFlatIdx : lhsFlatIdx+lhsBatchStride]
			}
			if len(cachedRHSPanels) > 0 {
				batchCachedRHSPanels = cachedRHSPanels[b*rhsPanelsPerBatch : (b+1)*rhsPanelsPerBatch]
			} else {
				batchRHS = rhs[rhsFlatIdx : rhsFlatIdx+rhsBatchStride]
			}
			batchOutput := output[outputFlatIdx : outputFlatIdx+outputBatchStride]
			neonLargeMatrixSliceFloat32( //alt:f32
				//alt:bf16 neonLargeMatrixSliceBFloat16(
				//alt:f16 neonLargeMatrixSliceFloat16(
				//alt:f64 neonLargeMatrixSliceFloat64(
				layout,
				batchLHS, batchRHS, batchOutput,
				lhsCrossSize, rhsCrossSize, contractingSize,
				0, lhsCrossSize, 0, rhsCrossSize,
				params,
				packedLHS, packedRHS, packedOutput,
				accumOutput,
				batchCachedLHSPanels, batchCachedRHSPanels,
			)
			lhsFlatIdx += lhsBatchStride
			rhsFlatIdx += rhsBatchStride
			outputFlatIdx += outputBatchStride
		}
		return
	}

	var nextItem atomic.Int32
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		itemIdx := int(nextItem.Add(1) - 1)
		if itemIdx >= len(workItems) {
			return
		}
		var (
			packedLHSRef *gobackend.Buffer
			packedLHS    []float32 //alt:f32
			//alt:bf16 packedLHS []bfloat16.BFloat16
			//alt:f16 packedLHS []float16.Float16
			//alt:f64 packedLHS []float64
		)
		if len(cachedLHSPanels) == 0 {
			var ok bool
			packedLHSRef, packedLHS, ok = GetBuffer[float32](backend, lhsAllocPanels*lhsPanelBufSize) //alt:f32
			//alt:bf16 packedLHSRef, packedLHS, ok = GetBuffer[bfloat16.BFloat16](backend, lhsAllocPanels*lhsPanelBufSize)
			//alt:f16 packedLHSRef, packedLHS, ok = GetBuffer[float16.Float16](backend, lhsAllocPanels*lhsPanelBufSize)
			//alt:f64 packedLHSRef, packedLHS, ok = GetBuffer[float64](backend, lhsAllocPanels*lhsPanelBufSize)
			if !ok {
				return
			}
			defer ReleaseBuffer(packedLHSRef)
		}
		var (
			packedRHSRef *gobackend.Buffer
			packedRHS    []float32 //alt:f32
			//alt:bf16 packedRHS []bfloat16.BFloat16
			//alt:f16 packedRHS []float16.Float16
			//alt:f64 packedRHS []float64
		)
		if len(cachedRHSPanels) == 0 {
			var ok bool
			packedRHSRef, packedRHS, ok = GetBuffer[float32](backend, params.PanelContractingSize*params.RHSPanelCrossSize) //alt:f32
			//alt:bf16 packedRHSRef, packedRHS, ok = GetBuffer[bfloat16.BFloat16](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
			//alt:f16 packedRHSRef, packedRHS, ok = GetBuffer[float16.Float16](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
			//alt:f64 packedRHSRef, packedRHS, ok = GetBuffer[float64](backend, params.PanelContractingSize*params.RHSPanelCrossSize)
			if !ok {
				return
			}
			defer ReleaseBuffer(packedRHSRef)
		}
		var (
			packedOutputRef, accumOutputRef *gobackend.Buffer
			packedOutput, accumOutput       []float32 //alt:f32|bf16|f16
			//alt:f64 packedOutput, accumOutput []float64
		)
		if needTempOutput {
			var ok bool
			packedOutputRef, packedOutput, ok = GetBuffer[float32](backend, params.LHSPanelCrossSize*params.RHSPanelCrossSize) //alt:f32|bf16|f16
			//alt:f64 packedOutputRef, packedOutput, ok = GetBuffer[float64](backend, params.LHSPanelCrossSize*params.RHSPanelCrossSize)
			if !ok {
				return
			}
			defer ReleaseBuffer(packedOutputRef)
			if numKPanels > 1 {
				const maxAccumPanels = 64
				accumOutputRef, accumOutput, ok = GetBuffer[float32](backend, maxAccumPanels*params.LHSPanelCrossSize*params.RHSPanelCrossSize) //alt:f32|bf16|f16
				//alt:f64 accumOutputRef, accumOutput, ok = GetBuffer[float64](backend, maxAccumPanels*params.LHSPanelCrossSize*params.RHSPanelCrossSize)
				if !ok {
					return
				}
				defer ReleaseBuffer(accumOutputRef)
			}
		}
		for {
			item := workItems[itemIdx]
			for batchIdx := item.BatchStart; batchIdx < item.BatchEnd; batchIdx++ {
				var (
					batchLhs, batchRhs               []float32 //alt:f32
					//alt:bf16 batchLhs, batchRhs []bfloat16.BFloat16
					//alt:f16 batchLhs, batchRhs []float16.Float16
					//alt:f64 batchLhs, batchRhs []float64
					batchCachedLHSPanels [][]float32 //alt:f32
					//alt:bf16 batchCachedLHSPanels [][]bfloat16.BFloat16
					//alt:f16 batchCachedLHSPanels [][]float16.Float16
					//alt:f64 batchCachedLHSPanels [][]float64
					batchCachedRHSPanels [][]float32 //alt:f32
					//alt:bf16 batchCachedRHSPanels [][]bfloat16.BFloat16
					//alt:f16 batchCachedRHSPanels [][]float16.Float16
					//alt:f64 batchCachedRHSPanels [][]float64
				)
				if len(cachedLHSPanels) > 0 {
					batchCachedLHSPanels = cachedLHSPanels[batchIdx*lhsPanelsPerBatch : (batchIdx+1)*lhsPanelsPerBatch]
				} else {
					batchLhs = lhs[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
				}
				if len(cachedRHSPanels) > 0 {
					batchCachedRHSPanels = cachedRHSPanels[batchIdx*rhsPanelsPerBatch : (batchIdx+1)*rhsPanelsPerBatch]
				} else {
					batchRhs = rhs[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
				}
				batchOutput := output[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
				neonLargeMatrixSliceFloat32( //alt:f32
					//alt:bf16 neonLargeMatrixSliceBFloat16(
					//alt:f16 neonLargeMatrixSliceFloat16(
					//alt:f64 neonLargeMatrixSliceFloat64(
					layout,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,
					item.LHSRowStart, item.LHSRowEnd, item.RHSColStart, item.RHSColEnd,
					params,
					packedLHS, packedRHS, packedOutput,
					accumOutput,
					batchCachedLHSPanels, batchCachedRHSPanels,
				)
			}
			itemIdx = int(nextItem.Add(1) - 1)
			if itemIdx >= len(workItems) {
				break
			}
		}
	})
}

// neonDispatchKernelFloat32 dispatches to the configured microkernel geometry (8x8, 4x16, or 8x12)
// and implementation (Plan9 assembly or Go archsimd).
func neonDispatchKernelFloat32( //alt:f32
	//alt:bf16 func neonDispatchKernelBFloat16(
	//alt:f16 func neonDispatchKernelFloat16(
	//alt:f64 func neonDispatchKernelFloat64(
	packedLHS, packedRHS []float32, //alt:f32
	//alt:bf16 packedLHS, packedRHS []bfloat16.BFloat16,
	//alt:f16 packedLHS, packedRHS []float16.Float16,
	//alt:f64 packedLHS, packedRHS []float64,
	packedOutput []float32, //alt:f32|bf16|f16
	//alt:f64 packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	kernelRows, kernelCols int,
	accumulate bool,
) {
	if NEONUseAsm {
		if kernelRows == 8 && kernelCols == 8 { //alt:f32|bf16|f16
			//alt:f64 if kernelRows == 8 && kernelCols == 4 {
			//alt:bf16 if NEONUseBFDOT {
			//alt:bf16 	neonLargeKernelBFloat16BFDOTAsm(
			//alt:bf16 		packedLHS, packedRHS, packedOutput,
			//alt:bf16 		lhsPanelRows, rhsPanelCols,
			//alt:bf16 		contractingLen,
			//alt:bf16 		lhsActiveRows, rhsActiveCols,
			//alt:bf16 		accumulate,
			//alt:bf16 	)
			//alt:bf16 	return
			//alt:bf16 }
			neonLargeKernelFloat32Asm( //alt:f32
				//alt:bf16 neonLargeKernelBFloat16Asm(
				//alt:f16 neonLargeKernelFloat16Asm(
				//alt:f64 neonLargeKernelFloat64Asm(
				packedLHS, packedRHS, packedOutput,
				lhsPanelRows, rhsPanelCols,
				contractingLen,
				lhsActiveRows, rhsActiveCols,
				accumulate,
			)
			return
		}
		if kernelRows == 4 && kernelCols == 16 { //alt:f32|bf16|f16
			//alt:f64 if kernelRows == 4 && kernelCols == 8 {
			//alt:bf16 if NEONUseBFDOT {
			//alt:bf16 	neonLargeKernelBFloat16BFDOTAsm_4x16(
			//alt:bf16 		packedLHS, packedRHS, packedOutput,
			//alt:bf16 		lhsPanelRows, rhsPanelCols,
			//alt:bf16 		contractingLen,
			//alt:bf16 		lhsActiveRows, rhsActiveCols,
			//alt:bf16 		accumulate,
			//alt:bf16 	)
			//alt:bf16 	return
			//alt:bf16 }
			neonLargeKernelFloat32Asm_4x16( //alt:f32
				//alt:bf16 neonLargeKernelBFloat16Asm_4x16(
				//alt:f16 neonLargeKernelFloat16Asm_4x16(
				//alt:f64 neonLargeKernelFloat64Asm_4x8(
				packedLHS, packedRHS, packedOutput,
				lhsPanelRows, rhsPanelCols,
				contractingLen,
				lhsActiveRows, rhsActiveCols,
				accumulate,
			)
			return
		}
		if kernelRows == 8 && kernelCols == 12 { //alt:f32
			neonLargeKernelFloat32Asm_8x12( //alt:f32
				packedLHS, packedRHS, packedOutput, //alt:f32
				lhsPanelRows, rhsPanelCols, //alt:f32
				contractingLen, //alt:f32
				lhsActiveRows, rhsActiveCols, //alt:f32
				accumulate, //alt:f32
			) //alt:f32
			return //alt:f32
		} //alt:f32
		//alt:f64 if kernelRows == 8 && kernelCols == 8 {
		//alt:f64 	smeLargeKernelFloat64Asm_8x8(
		//alt:f64 		packedLHS, packedRHS, packedOutput,
		//alt:f64 		lhsPanelRows, rhsPanelCols,
		//alt:f64 		contractingLen,
		//alt:f64 		lhsActiveRows, rhsActiveCols,
		//alt:f64 		accumulate,
		//alt:f64 	)
		//alt:f64 	return
		//alt:f64 }
		if kernelRows == 16 && kernelCols == 16 {
			smeLargeKernelFloat32Asm_16x16( //alt:f32
				//alt:bf16 smeLargeKernelBFloat16Asm_16x16(
				//alt:f16 smeLargeKernelFloat16Asm_16x16(
				//alt:f64 smeLargeKernelFloat64Asm_16x16(
				packedLHS, packedRHS, packedOutput,
				lhsPanelRows, rhsPanelCols,
				contractingLen,
				lhsActiveRows, rhsActiveCols,
				accumulate,
			)
			return
		}
		if kernelRows == 16 && kernelCols == 64 { //alt:f32
			smeLargeKernelFloat32Asm_16x64( //alt:f32
				packedLHS, packedRHS, packedOutput, //alt:f32
				lhsPanelRows, rhsPanelCols, //alt:f32
				contractingLen, //alt:f32
				lhsActiveRows, rhsActiveCols, //alt:f32
				accumulate, //alt:f32
			) //alt:f32
			return //alt:f32
		} //alt:f32
		if kernelRows == 32 && kernelCols == 32 { //alt:f32|bf16|f16
			smeLargeKernelFloat32Asm_32x32( //alt:f32
				//alt:bf16 smeLargeKernelBFloat16Asm_32x32(
				//alt:f16 smeLargeKernelFloat16Asm_32x32(
				packedLHS, packedRHS, packedOutput, //alt:f32|bf16|f16
				lhsPanelRows, rhsPanelCols, //alt:f32|bf16|f16
				contractingLen, //alt:f32|bf16|f16
				lhsActiveRows, rhsActiveCols, //alt:f32|bf16|f16
				accumulate, //alt:f32|bf16|f16
			) //alt:f32|bf16|f16
			return //alt:f32|bf16|f16
		} //alt:f32|bf16|f16
	}

	if kernelRows == 4 && kernelCols == 16 { //alt:f32|bf16|f16
		//alt:f64 if kernelRows == 4 && kernelCols == 8 {
		neonLargeKernelFloat32( //alt:f32
			//alt:bf16 neonLargeKernelBFloat16(
			//alt:f16 neonLargeKernelFloat16(
			//alt:f64 neonLargeKernelFloat64(
			packedLHS, packedRHS, packedOutput,
			lhsPanelRows, rhsPanelCols,
			contractingLen,
			lhsActiveRows, rhsActiveCols,
			accumulate,
		)
		return
	}
	if kernelRows == 8 && kernelCols == 8 { //alt:f32|bf16|f16
		//alt:f64 if kernelRows == 8 && kernelCols == 4 {
		neonLargeKernelFloat32_8x8( //alt:f32
			//alt:bf16 neonLargeKernelBFloat16_8x8(
			//alt:f16 neonLargeKernelFloat16_8x8(
			//alt:f64 neonLargeKernelFloat64_8x4(
			packedLHS, packedRHS, packedOutput,
			lhsPanelRows, rhsPanelCols,
			contractingLen,
			lhsActiveRows, rhsActiveCols,
			accumulate,
		)
		return
	}
	if kernelRows == 8 && kernelCols == 12 { //alt:f32
		neonLargeKernelFloat32_8x12( //alt:f32
			packedLHS, packedRHS, packedOutput, //alt:f32
			lhsPanelRows, rhsPanelCols, //alt:f32
			contractingLen, //alt:f32
			lhsActiveRows, rhsActiveCols, //alt:f32
			accumulate, //alt:f32
		) //alt:f32
		return //alt:f32
	} //alt:f32

	panic(errors.Errorf("unsupported kernel L1 block sizes for neon kernel: lhsL1KernelRows=%d, rhsL1KernelCols=%d",
		kernelRows, kernelCols))
}

// neonLargeMatrixSliceFloat32 performs a slice of the matrix multiplication on one batch example.
func neonLargeMatrixSliceFloat32( //alt:f32
	//alt:bf16 func neonLargeMatrixSliceBFloat16(
	//alt:f16 func neonLargeMatrixSliceFloat16(
	//alt:f64 func neonLargeMatrixSliceFloat64(
	layout dot.Layout,
	lhsMatrix, rhsMatrix []float32, outputMatrix []float32, //alt:f32
	//alt:bf16 lhsMatrix, rhsMatrix []bfloat16.BFloat16, outputMatrix []float32,
	//alt:f16 lhsMatrix, rhsMatrix []float16.Float16, outputMatrix []float32,
	//alt:f64 lhsMatrix, rhsMatrix []float64, outputMatrix []float64,
	lhsCrossSize, rhsCrossSize, contractingSize int,
	rowStart, rowEnd, colStart, colEnd int,
	params matmul.CacheParams,
	packedLHS, packedRHS []float32, packedOutput []float32, //alt:f32
	//alt:bf16 packedLHS, packedRHS []bfloat16.BFloat16, packedOutput []float32,
	//alt:f16 packedLHS, packedRHS []float16.Float16, packedOutput []float32,
	//alt:f64 packedLHS, packedRHS []float64, packedOutput []float64,
	accumBuffer []float32, //alt:f32|bf16|f16
	//alt:f64 accumBuffer []float64,
	cachedLHSPanels, cachedRHSPanels [][]float32, //alt:f32
	//alt:bf16 cachedLHSPanels, cachedRHSPanels [][]bfloat16.BFloat16,
	//alt:f16 cachedLHSPanels, cachedRHSPanels [][]float16.Float16,
	//alt:f64 cachedLHSPanels, cachedRHSPanels [][]float64,
) {
	_ = lhsCrossSize

	numKPanels := (contractingSize + params.PanelContractingSize - 1) / params.PanelContractingSize
	lhsPanelBufSize := params.LHSPanelCrossSize * params.PanelContractingSize
	isSMEKernel := NEONUseAsm && params.LHSL1KernelRows >= 16 //alt:f32|bf16|f16
	//alt:f64 isSMEKernel := NEONUseAsm && params.LHSL1KernelRows >= 8 && params.RHSL1KernelCols >= 8

	// Loop 5 (jc): Tiling RHS cross axis (N), the output columns.
	for rhsPanelColIdx := colStart; rhsPanelColIdx < colEnd; {
		colPanelIdx := rhsPanelColIdx / params.RHSPanelCrossSize
		panelEnd := (colPanelIdx + 1) * params.RHSPanelCrossSize
		rhsPanelWidth := min(colEnd, panelEnd) - rhsPanelColIdx

		numMPanels := 0
		if rowEnd > rowStart {
			numMPanels = ((rowEnd - 1) / params.LHSPanelCrossSize) - (rowStart / params.LHSPanelCrossSize) + 1
		}
		canCacheAllLHS := len(packedLHS) >= numMPanels*numKPanels*lhsPanelBufSize
		accumPanelStride := params.RHSPanelCrossSize
		panelSize := params.LHSPanelCrossSize * accumPanelStride
		useAccum := len(accumBuffer) >= numMPanels*panelSize

		// Loop 4 (p): Tiling the contracting axis (K)
		for contractingPanelIdx := 0; contractingPanelIdx < contractingSize; contractingPanelIdx += params.PanelContractingSize {
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			kPanelIdx := contractingPanelIdx / params.PanelContractingSize
			if len(cachedRHSPanels) > 0 {
				numColPanels := (rhsCrossSize + params.RHSPanelCrossSize - 1) / params.RHSPanelCrossSize
				panel := cachedRHSPanels[kPanelIdx*numColPanels+colPanelIdx]
				stripOffset := (rhsPanelColIdx % params.RHSPanelCrossSize) / params.RHSL1KernelCols
				stripSize := contractingPanelWidth * params.RHSL1KernelCols
				packedRHS = panel[stripOffset*stripSize:]
			} else if layout == dot.LayoutNonTransposed {
				neonPackRHSNonTransposed(rhsMatrix, packedRHS, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)
			} else {
				neonPackLHS(rhsMatrix, packedRHS, rhsPanelColIdx, contractingPanelIdx, contractingSize,
					rhsPanelWidth, contractingPanelWidth, params.RHSL1KernelCols)
			}

			// Loop 3 (ic): Tiling LHS cross axis (M), i.e. the output rows.
			for mIdx, lhsPanelRowIdx := 0, rowStart; lhsPanelRowIdx < rowEnd; mIdx++ {
				rowPanelIdx := lhsPanelRowIdx / params.LHSPanelCrossSize
				lhsPanelEnd := (rowPanelIdx + 1) * params.LHSPanelCrossSize
				lhsPanelHeight := min(rowEnd, lhsPanelEnd) - lhsPanelRowIdx

				curPackedLHS := packedLHS
				if len(cachedLHSPanels) > 0 {
					numRowPanels := (lhsCrossSize + params.LHSPanelCrossSize - 1) / params.LHSPanelCrossSize
					panel := cachedLHSPanels[kPanelIdx*numRowPanels+rowPanelIdx]
					stripOffset := (lhsPanelRowIdx % params.LHSPanelCrossSize) / params.LHSL1KernelRows
					stripSize := contractingPanelWidth * params.LHSL1KernelRows
					curPackedLHS = panel[stripOffset*stripSize:]
				} else if canCacheAllLHS {
					panelIdx := mIdx*numKPanels + kPanelIdx
					curPackedLHS = packedLHS[panelIdx*lhsPanelBufSize : (panelIdx+1)*lhsPanelBufSize]
					if rhsPanelColIdx == colStart {
						neonPackLHS(lhsMatrix, curPackedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)
					}
				} else {
					neonPackLHS(lhsMatrix, curPackedLHS, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)
				}

				isFirstContractingPanel := contractingPanelIdx == 0
				accumulate := !isFirstContractingPanel
				canDirectOutput := isSMEKernel || ((lhsPanelHeight%params.LHSL1KernelRows == 0) && (rhsPanelWidth%params.RHSL1KernelCols == 0))

				if canDirectOutput {
					outOffset := lhsPanelRowIdx*rhsCrossSize + rhsPanelColIdx
					outSlice := outputMatrix[outOffset : outOffset+(lhsPanelHeight-1)*rhsCrossSize+rhsPanelWidth]
					neonDispatchKernelFloat32( //alt:f32
						//alt:bf16 neonDispatchKernelBFloat16(
						//alt:f16 neonDispatchKernelFloat16(
						//alt:f64 neonDispatchKernelFloat64(
						curPackedLHS, packedRHS, outSlice,
						params.LHSPanelCrossSize, rhsCrossSize,
						contractingPanelWidth,
						lhsPanelHeight, rhsPanelWidth,
						params.LHSL1KernelRows, params.RHSL1KernelCols,
						accumulate,
					)
				} else if useAccum {
					accumOffset := mIdx * panelSize
					accumSlice := accumBuffer[accumOffset : accumOffset+panelSize]
					neonDispatchKernelFloat32( //alt:f32
						//alt:bf16 neonDispatchKernelBFloat16(
						//alt:f16 neonDispatchKernelFloat16(
						//alt:f64 neonDispatchKernelFloat64(
						curPackedLHS, packedRHS, accumSlice,
						params.LHSPanelCrossSize, accumPanelStride,
						contractingPanelWidth,
						lhsPanelHeight, rhsPanelWidth,
						params.LHSL1KernelRows, params.RHSL1KernelCols,
						accumulate,
					)
				} else {
					neonDispatchKernelFloat32( //alt:f32
						//alt:bf16 neonDispatchKernelBFloat16(
						//alt:f16 neonDispatchKernelFloat16(
						//alt:f64 neonDispatchKernelFloat64(
						curPackedLHS, packedRHS, packedOutput,
						params.LHSPanelCrossSize, params.RHSPanelCrossSize,
						contractingPanelWidth,
						lhsPanelHeight, rhsPanelWidth,
						params.LHSL1KernelRows, params.RHSL1KernelCols,
						false,
					)

					neonApplyPackedOutputFloat32( //alt:f32|bf16|f16
						//alt:f64 neonApplyPackedOutputFloat64(
						packedOutput, outputMatrix,
						isFirstContractingPanel,
						params.RHSPanelCrossSize,
						lhsPanelRowIdx, rhsPanelColIdx,
						rhsCrossSize,
						lhsPanelHeight, rhsPanelWidth)
				}
				lhsPanelRowIdx += lhsPanelHeight
			}
		}

		if useAccum {
			for mIdx, lhsPanelRowIdx := 0, rowStart; lhsPanelRowIdx < rowEnd; mIdx++ {
				rowPanelIdx := lhsPanelRowIdx / params.LHSPanelCrossSize
				lhsPanelEnd := (rowPanelIdx + 1) * params.LHSPanelCrossSize
				lhsPanelHeight := min(rowEnd, lhsPanelEnd) - lhsPanelRowIdx

				canDirectOutput := isSMEKernel || ((lhsPanelHeight%params.LHSL1KernelRows == 0) && (rhsPanelWidth%params.RHSL1KernelCols == 0))
				if !canDirectOutput {
					accumOffset := mIdx * panelSize
					accumSlice := accumBuffer[accumOffset : accumOffset+lhsPanelHeight*accumPanelStride]
					neonApplyPackedOutputFloat32( //alt:f32|bf16|f16
						//alt:f64 neonApplyPackedOutputFloat64(
						accumSlice, outputMatrix,
						true,
						accumPanelStride,
						lhsPanelRowIdx, rhsPanelColIdx,
						rhsCrossSize,
						lhsPanelHeight, rhsPanelWidth)
				}
				lhsPanelRowIdx += lhsPanelHeight
			}
		}
		rhsPanelColIdx += rhsPanelWidth
	}
}

// neonPrepackRHSFloat32 pre-packs the RHS matrix into a single flat buffer and returns panel slices.
func neonPrepackRHSFloat32( //alt:f32
	//alt:bf16 func neonPrepackRHSBFloat16(
	//alt:f16 func neonPrepackRHSFloat16(
	//alt:f64 func neonPrepackRHSFloat64(
	backend *gobackend.Backend,
	layout dot.Layout,
	rhs []float32, //alt:f32
	//alt:bf16 rhs []bfloat16.BFloat16,
	//alt:f16 rhs []float16.Float16,
	//alt:f64 rhs []float64,
	batchSize, rhsCrossSize, contractingSize int,
	params matmul.CacheParams,
) (*gobackend.Buffer, [][]float32) { //alt:f32
	//alt:bf16 ) (*gobackend.Buffer, [][]bfloat16.BFloat16) {
	//alt:f16 ) (*gobackend.Buffer, [][]float16.Float16) {
	//alt:f64 ) (*gobackend.Buffer, [][]float64) {
	numKPanels := (contractingSize + params.PanelContractingSize - 1) / params.PanelContractingSize
	numColPanels := (rhsCrossSize + params.RHSPanelCrossSize - 1) / params.RHSPanelCrossSize
	rhsPanelsPerBatch := numKPanels * numColPanels
	rhsBatchStride := rhsCrossSize * contractingSize

	totalElements := 0
	for colPanelIdx := range numColPanels {
		rhsPanelColIdx := colPanelIdx * params.RHSPanelCrossSize
		rhsPanelWidth := min(params.RHSPanelCrossSize, rhsCrossSize-rhsPanelColIdx)
		numStrips := (rhsPanelWidth + params.RHSL1KernelCols - 1) / params.RHSL1KernelCols
		for kPanelIdx := range numKPanels {
			contractingPanelIdx := kPanelIdx * params.PanelContractingSize
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			totalElements += contractingPanelWidth * numStrips * params.RHSL1KernelCols
		}
	}
	totalElements *= batchSize

	ref, flatBuf, ok := GetExactBuffer[float32](backend, totalElements) //alt:f32
	//alt:bf16 ref, flatBuf, ok := GetExactBuffer[bfloat16.BFloat16](backend, totalElements)
	//alt:f16 ref, flatBuf, ok := GetExactBuffer[float16.Float16](backend, totalElements)
	//alt:f64 ref, flatBuf, ok := GetExactBuffer[float64](backend, totalElements)
	if !ok {
		return nil, nil
	}

	panels := make([][]float32, batchSize*rhsPanelsPerBatch) //alt:f32
	//alt:bf16 panels := make([][]bfloat16.BFloat16, batchSize*rhsPanelsPerBatch)
	//alt:f16 panels := make([][]float16.Float16, batchSize*rhsPanelsPerBatch)
	//alt:f64 panels := make([][]float64, batchSize*rhsPanelsPerBatch)

	offset := 0
	rhsFlatIdx := 0
	for b := range batchSize {
		batchRHS := rhs[rhsFlatIdx : rhsFlatIdx+rhsBatchStride]
		for colPanelIdx := range numColPanels {
			rhsPanelColIdx := colPanelIdx * params.RHSPanelCrossSize
			rhsPanelWidth := min(params.RHSPanelCrossSize, rhsCrossSize-rhsPanelColIdx)
			numStrips := (rhsPanelWidth + params.RHSL1KernelCols - 1) / params.RHSL1KernelCols
			for kPanelIdx := range numKPanels {
				contractingPanelIdx := kPanelIdx * params.PanelContractingSize
				contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
				panelLen := contractingPanelWidth * numStrips * params.RHSL1KernelCols
				panelBuf := flatBuf[offset : offset+panelLen]
				offset += panelLen

				if layout == dot.LayoutNonTransposed {
					neonPackRHSNonTransposed(batchRHS, panelBuf, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)
				} else {
					neonPackLHS(batchRHS, panelBuf, rhsPanelColIdx, contractingPanelIdx, contractingSize, rhsPanelWidth, contractingPanelWidth, params.RHSL1KernelCols)
				}
				panels[b*rhsPanelsPerBatch+kPanelIdx*numColPanels+colPanelIdx] = panelBuf
			}
		}
		rhsFlatIdx += rhsBatchStride
	}
	return ref, panels
}

// neonPrepackLHSFloat32 pre-packs the LHS matrix into a single flat buffer and returns panel slices.
func neonPrepackLHSFloat32( //alt:f32
	//alt:bf16 func neonPrepackLHSBFloat16(
	//alt:f16 func neonPrepackLHSFloat16(
	//alt:f64 func neonPrepackLHSFloat64(
	backend *gobackend.Backend,
	lhs []float32, //alt:f32
	//alt:bf16 lhs []bfloat16.BFloat16,
	//alt:f16 lhs []float16.Float16,
	//alt:f64 lhs []float64,
	batchSize, lhsCrossSize, contractingSize int,
	params matmul.CacheParams,
) (*gobackend.Buffer, [][]float32) { //alt:f32
	//alt:bf16 ) (*gobackend.Buffer, [][]bfloat16.BFloat16) {
	//alt:f16 ) (*gobackend.Buffer, [][]float16.Float16) {
	//alt:f64 ) (*gobackend.Buffer, [][]float64) {
	numKPanels := (contractingSize + params.PanelContractingSize - 1) / params.PanelContractingSize
	numRowPanels := (lhsCrossSize + params.LHSPanelCrossSize - 1) / params.LHSPanelCrossSize
	lhsPanelsPerBatch := numKPanels * numRowPanels
	lhsBatchStride := lhsCrossSize * contractingSize

	totalElements := 0
	for rowPanelIdx := range numRowPanels {
		lhsPanelRowIdx := rowPanelIdx * params.LHSPanelCrossSize
		lhsPanelHeight := min(params.LHSPanelCrossSize, lhsCrossSize-lhsPanelRowIdx)
		numStrips := (lhsPanelHeight + params.LHSL1KernelRows - 1) / params.LHSL1KernelRows
		for kPanelIdx := range numKPanels {
			contractingPanelIdx := kPanelIdx * params.PanelContractingSize
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			totalElements += contractingPanelWidth * numStrips * params.LHSL1KernelRows
		}
	}
	totalElements *= batchSize

	ref, flatBuf, ok := GetExactBuffer[float32](backend, totalElements) //alt:f32
	//alt:bf16 ref, flatBuf, ok := GetExactBuffer[bfloat16.BFloat16](backend, totalElements)
	//alt:f16 ref, flatBuf, ok := GetExactBuffer[float16.Float16](backend, totalElements)
	//alt:f64 ref, flatBuf, ok := GetExactBuffer[float64](backend, totalElements)
	if !ok {
		return nil, nil
	}

	panels := make([][]float32, batchSize*lhsPanelsPerBatch) //alt:f32
	//alt:bf16 panels := make([][]bfloat16.BFloat16, batchSize*lhsPanelsPerBatch)
	//alt:f16 panels := make([][]float16.Float16, batchSize*lhsPanelsPerBatch)
	//alt:f64 panels := make([][]float64, batchSize*lhsPanelsPerBatch)

	offset := 0
	lhsFlatIdx := 0
	for b := range batchSize {
		batchLHS := lhs[lhsFlatIdx : lhsFlatIdx+lhsBatchStride]
		for rowPanelIdx := range numRowPanels {
			lhsPanelRowIdx := rowPanelIdx * params.LHSPanelCrossSize
			lhsPanelHeight := min(params.LHSPanelCrossSize, lhsCrossSize-lhsPanelRowIdx)
			numStrips := (lhsPanelHeight + params.LHSL1KernelRows - 1) / params.LHSL1KernelRows
			for kPanelIdx := range numKPanels {
				contractingPanelIdx := kPanelIdx * params.PanelContractingSize
				contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
				panelLen := contractingPanelWidth * numStrips * params.LHSL1KernelRows
				panelBuf := flatBuf[offset : offset+panelLen]
				offset += panelLen

				neonPackLHS(batchLHS, panelBuf, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)
				panels[b*lhsPanelsPerBatch+kPanelIdx*numRowPanels+rowPanelIdx] = panelBuf
			}
		}
		lhsFlatIdx += lhsBatchStride
	}
	return ref, panels
}

// neonLargeKernelFloat32 implements a 4x16 (4x8 for Float64) GEMM microkernel using Go archsimd (16 accumulators).
func neonLargeKernelFloat32( //alt:f32
	//alt:bf16 func neonLargeKernelBFloat16(
	//alt:f16 func neonLargeKernelFloat16(
	//alt:f64 func neonLargeKernelFloat64(
	packedLHS, packedRHS []float32, //alt:f32
	//alt:bf16 packedLHS, packedRHS []bfloat16.BFloat16,
	//alt:f16 packedLHS, packedRHS []float16.Float16,
	//alt:f64 packedLHS, packedRHS []float64,
	packedOutput []float32, //alt:f32|bf16|f16
	//alt:f64 packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
) {
	defer func() {
		runtime.KeepAlive(packedLHS)
		runtime.KeepAlive(packedRHS)
		runtime.KeepAlive(packedOutput)
	}()
	_ = lhsPanelRows

	const (
		kernelRows = 4  //alt:f32|bf16|f16|f64
		kernelCols = 16 //alt:f32|bf16|f16
		//alt:f64 kernelCols = 8
		outputNumLanes = 4 //alt:f32|bf16|f16
		//alt:f64 outputNumLanes = 2
		bytesPerInputElement = 4 //alt:f32
		//alt:bf16 bytesPerInputElement = 2
		//alt:f16 bytesPerInputElement = 2
		//alt:f64 bytesPerInputElement = 8
		bytesPerOutputElement = 4 //alt:f32|bf16|f16
		//alt:f64 bytesPerOutputElement = 8
	)

	outputBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedOutput)))
	rhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedRHS)))
	lhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedLHS)))

	for lhsRowIdx := 0; lhsRowIdx < lhsActiveRows; lhsRowIdx += kernelRows {
		for rhsColIdx := 0; rhsColIdx < rhsActiveCols; rhsColIdx += kernelCols {
			outputRowStart := lhsRowIdx
			outputColStart := rhsColIdx
			outputStride := rhsPanelCols

			// 16 accumulators (4 rows x 4 RHS vectors)
			accum_lhs0_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs0_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs0_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs0_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs0_rhs2 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs0_rhs2 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs0_rhs3 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs0_rhs3 := archsimd.BroadcastFloat64x2(0.0)

			accum_lhs1_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs1_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs1_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs1_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs1_rhs2 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs1_rhs2 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs1_rhs3 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs1_rhs3 := archsimd.BroadcastFloat64x2(0.0)

			accum_lhs2_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs2_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs2_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs2_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs2_rhs2 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs2_rhs2 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs2_rhs3 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs2_rhs3 := archsimd.BroadcastFloat64x2(0.0)

			accum_lhs3_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs3_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs3_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs3_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs3_rhs2 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs3_rhs2 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs3_rhs3 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs3_rhs3 := archsimd.BroadcastFloat64x2(0.0)

			idxLHS := lhsRowIdx * contractingLen
			idxRHS := rhsColIdx * contractingLen

			rhsRowPtr := rhsBasePtr + uintptr(idxRHS*bytesPerInputElement)
			lhsRowPtr := lhsBasePtr + uintptr(idxLHS*bytesPerInputElement)

			rOffset := uintptr(0)
			lOffset := uintptr(0)

			rhsStride := uintptr(kernelCols * bytesPerInputElement)
			lhsStride := uintptr(kernelRows * bytesPerInputElement)

			for range contractingLen {
				rhsPtr0 := unsafe.Pointer(rhsRowPtr + rOffset)
				rhsPtr1 := unsafe.Pointer(rhsRowPtr + rOffset + 16)
				rhsPtr2 := unsafe.Pointer(rhsRowPtr + rOffset + 32) //alt:f32|f64
				rhsPtr3 := unsafe.Pointer(rhsRowPtr + rOffset + 48) //alt:f32|f64

				rhsVec0 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr0)) //alt:f32
				rhsVec1 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr1)) //alt:f32
				rhsVec2 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr2)) //alt:f32
				rhsVec3 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr3)) //alt:f32
				//alt:f64 rhsVec0 := archsimd.LoadFloat64x2Array((*[2]float64)(rhsPtr0))
				//alt:f64 rhsVec1 := archsimd.LoadFloat64x2Array((*[2]float64)(rhsPtr1))
				//alt:f64 rhsVec2 := archsimd.LoadFloat64x2Array((*[2]float64)(rhsPtr2))
				//alt:f64 rhsVec3 := archsimd.LoadFloat64x2Array((*[2]float64)(rhsPtr3))
				//alt:bf16 rhsVec0, rhsVec1 := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(rhsPtr0)).ToFloat32()
				//alt:bf16 rhsVec2, rhsVec3 := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(rhsPtr1)).ToFloat32()
				//alt:f16 rhsVec0, rhsVec1 := float16.LoadFloat16x8((*[8]float16.Float16)(rhsPtr0)).ToFloat32()
				//alt:f16 rhsVec2, rhsVec3 := float16.LoadFloat16x8((*[8]float16.Float16)(rhsPtr1)).ToFloat32()
				rOffset += rhsStride

				// Row 0
				lhsVal0 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 0))) //alt:f32
				//alt:bf16 lhsVal0 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 0))).Float32()
				//alt:f16 lhsVal0 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 0))).Float32()
				//alt:f64 lhsVal0 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 0)))
				lhsVec0 := archsimd.BroadcastFloat32x4(lhsVal0) //alt:f32|bf16|f16
				//alt:f64 lhsVec0 := archsimd.BroadcastFloat64x2(lhsVal0)
				accum_lhs0_rhs0 = rhsVec0.MulAdd(lhsVec0, accum_lhs0_rhs0)
				accum_lhs0_rhs1 = rhsVec1.MulAdd(lhsVec0, accum_lhs0_rhs1)
				accum_lhs0_rhs2 = rhsVec2.MulAdd(lhsVec0, accum_lhs0_rhs2)
				accum_lhs0_rhs3 = rhsVec3.MulAdd(lhsVec0, accum_lhs0_rhs3)

				// Row 1
				lhsVal1 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal1 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement))).Float32()
				//alt:f16 lhsVal1 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement))).Float32()
				//alt:f64 lhsVal1 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement)))
				lhsVec1 := archsimd.BroadcastFloat32x4(lhsVal1) //alt:f32|bf16|f16
				//alt:f64 lhsVec1 := archsimd.BroadcastFloat64x2(lhsVal1)
				accum_lhs1_rhs0 = rhsVec0.MulAdd(lhsVec1, accum_lhs1_rhs0)
				accum_lhs1_rhs1 = rhsVec1.MulAdd(lhsVec1, accum_lhs1_rhs1)
				accum_lhs1_rhs2 = rhsVec2.MulAdd(lhsVec1, accum_lhs1_rhs2)
				accum_lhs1_rhs3 = rhsVec3.MulAdd(lhsVec1, accum_lhs1_rhs3)

				// Row 2
				lhsVal2 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal2 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal2 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal2 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement)))
				lhsVec2 := archsimd.BroadcastFloat32x4(lhsVal2) //alt:f32|bf16|f16
				//alt:f64 lhsVec2 := archsimd.BroadcastFloat64x2(lhsVal2)
				accum_lhs2_rhs0 = rhsVec0.MulAdd(lhsVec2, accum_lhs2_rhs0)
				accum_lhs2_rhs1 = rhsVec1.MulAdd(lhsVec2, accum_lhs2_rhs1)
				accum_lhs2_rhs2 = rhsVec2.MulAdd(lhsVec2, accum_lhs2_rhs2)
				accum_lhs2_rhs3 = rhsVec3.MulAdd(lhsVec2, accum_lhs2_rhs3)

				// Row 3
				lhsVal3 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal3 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal3 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal3 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement)))
				lhsVec3 := archsimd.BroadcastFloat32x4(lhsVal3) //alt:f32|bf16|f16
				//alt:f64 lhsVec3 := archsimd.BroadcastFloat64x2(lhsVal3)
				accum_lhs3_rhs0 = rhsVec0.MulAdd(lhsVec3, accum_lhs3_rhs0)
				accum_lhs3_rhs1 = rhsVec1.MulAdd(lhsVec3, accum_lhs3_rhs1)
				accum_lhs3_rhs2 = rhsVec2.MulAdd(lhsVec3, accum_lhs3_rhs2)
				accum_lhs3_rhs3 = rhsVec3.MulAdd(lhsVec3, accum_lhs3_rhs3)

				lOffset += lhsStride
			}

			outputIdx0 := uintptr((outputRowStart*outputStride + outputColStart) * bytesPerOutputElement)
			outputIdx1 := outputIdx0 + uintptr(rhsPanelCols*bytesPerOutputElement)
			outputIdx2 := outputIdx0 + uintptr(2*rhsPanelCols*bytesPerOutputElement)
			outputIdx3 := outputIdx0 + uintptr(3*rhsPanelCols*bytesPerOutputElement)
			const regStride = uintptr(16) // 128 bits = 16 bytes

			if !accumulate {
				accum_lhs0_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0)))                  //alt:f32|bf16|f16
				accum_lhs0_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))      //alt:f32|bf16|f16
				accum_lhs0_rhs2.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + 2*regStride)))    //alt:f32|bf16|f16
				accum_lhs0_rhs3.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + 3*regStride)))    //alt:f32|bf16|f16
				accum_lhs1_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1)))                  //alt:f32|bf16|f16
				accum_lhs1_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))      //alt:f32|bf16|f16
				accum_lhs1_rhs2.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + 2*regStride)))    //alt:f32|bf16|f16
				accum_lhs1_rhs3.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + 3*regStride)))    //alt:f32|bf16|f16
				accum_lhs2_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2)))                  //alt:f32|bf16|f16
				accum_lhs2_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))      //alt:f32|bf16|f16
				accum_lhs2_rhs2.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + 2*regStride)))    //alt:f32|bf16|f16
				accum_lhs2_rhs3.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + 3*regStride)))    //alt:f32|bf16|f16
				accum_lhs3_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3)))                  //alt:f32|bf16|f16
				accum_lhs3_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))      //alt:f32|bf16|f16
				accum_lhs3_rhs2.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + 2*regStride)))    //alt:f32|bf16|f16
				accum_lhs3_rhs3.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + 3*regStride)))    //alt:f32|bf16|f16
				//alt:f64 accum_lhs0_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0)))
				//alt:f64 accum_lhs0_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))
				//alt:f64 accum_lhs0_rhs2.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + 2*regStride)))
				//alt:f64 accum_lhs0_rhs3.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + 3*regStride)))
				//alt:f64 accum_lhs1_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1)))
				//alt:f64 accum_lhs1_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))
				//alt:f64 accum_lhs1_rhs2.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + 2*regStride)))
				//alt:f64 accum_lhs1_rhs3.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + 3*regStride)))
				//alt:f64 accum_lhs2_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2)))
				//alt:f64 accum_lhs2_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))
				//alt:f64 accum_lhs2_rhs2.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + 2*regStride)))
				//alt:f64 accum_lhs2_rhs3.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + 3*regStride)))
				//alt:f64 accum_lhs3_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3)))
				//alt:f64 accum_lhs3_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))
				//alt:f64 accum_lhs3_rhs2.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + 2*regStride)))
				//alt:f64 accum_lhs3_rhs3.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + 3*regStride)))
			} else {
				accum_lhs0_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0)))                               //alt:f32|bf16|f16
				accum_lhs0_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))       //alt:f32|bf16|f16
				accum_lhs0_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + 2*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + 2*regStride)))   //alt:f32|bf16|f16
				accum_lhs0_rhs3.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + 3*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + 3*regStride)))   //alt:f32|bf16|f16
				accum_lhs1_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1)))                               //alt:f32|bf16|f16
				accum_lhs1_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))       //alt:f32|bf16|f16
				accum_lhs1_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + 2*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + 2*regStride)))   //alt:f32|bf16|f16
				accum_lhs1_rhs3.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + 3*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + 3*regStride)))   //alt:f32|bf16|f16
				accum_lhs2_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2)))                               //alt:f32|bf16|f16
				accum_lhs2_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))       //alt:f32|bf16|f16
				accum_lhs2_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + 2*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + 2*regStride)))   //alt:f32|bf16|f16
				accum_lhs2_rhs3.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + 3*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + 3*regStride)))   //alt:f32|bf16|f16
				accum_lhs3_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3)))                               //alt:f32|bf16|f16
				accum_lhs3_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))       //alt:f32|bf16|f16
				accum_lhs3_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + 2*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + 2*regStride)))   //alt:f32|bf16|f16
				accum_lhs3_rhs3.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + 3*regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + 3*regStride)))   //alt:f32|bf16|f16
				//alt:f64 accum_lhs0_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0)))
				//alt:f64 accum_lhs0_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))
				//alt:f64 accum_lhs0_rhs2.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + 2*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + 2*regStride)))
				//alt:f64 accum_lhs0_rhs3.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + 3*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + 3*regStride)))
				//alt:f64 accum_lhs1_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1)))
				//alt:f64 accum_lhs1_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))
				//alt:f64 accum_lhs1_rhs2.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + 2*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + 2*regStride)))
				//alt:f64 accum_lhs1_rhs3.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + 3*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + 3*regStride)))
				//alt:f64 accum_lhs2_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2)))
				//alt:f64 accum_lhs2_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))
				//alt:f64 accum_lhs2_rhs2.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + 2*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + 2*regStride)))
				//alt:f64 accum_lhs2_rhs3.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + 3*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + 3*regStride)))
				//alt:f64 accum_lhs3_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3)))
				//alt:f64 accum_lhs3_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))
				//alt:f64 accum_lhs3_rhs2.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + 2*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + 2*regStride)))
				//alt:f64 accum_lhs3_rhs3.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + 3*regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + 3*regStride)))
			}
		}
	}
}

// neonLargeKernelFloat32_8x8 implements an 8x8 (8x4 for Float64) GEMM microkernel using Go archsimd (16 accumulators).
func neonLargeKernelFloat32_8x8( //alt:f32
	//alt:bf16 func neonLargeKernelBFloat16_8x8(
	//alt:f16 func neonLargeKernelFloat16_8x8(
	//alt:f64 func neonLargeKernelFloat64_8x4(
	packedLHS, packedRHS []float32, //alt:f32
	//alt:bf16 packedLHS, packedRHS []bfloat16.BFloat16,
	//alt:f16 packedLHS, packedRHS []float16.Float16,
	//alt:f64 packedLHS, packedRHS []float64,
	packedOutput []float32, //alt:f32|bf16|f16
	//alt:f64 packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
) {
	defer func() {
		runtime.KeepAlive(packedLHS)
		runtime.KeepAlive(packedRHS)
		runtime.KeepAlive(packedOutput)
	}()
	_ = lhsPanelRows

	const (
		kernelRows = 8 //alt:f32|bf16|f16|f64
		kernelCols = 8 //alt:f32|bf16|f16
		//alt:f64 kernelCols = 4
		bytesPerInputElement = 4 //alt:f32
		//alt:bf16 bytesPerInputElement = 2
		//alt:f16 bytesPerInputElement = 2
		//alt:f64 bytesPerInputElement = 8
		bytesPerOutputElement = 4 //alt:f32|bf16|f16
		//alt:f64 bytesPerOutputElement = 8
	)

	outputBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedOutput)))
	rhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedRHS)))
	lhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(packedLHS)))

	for lhsRowIdx := 0; lhsRowIdx < lhsActiveRows; lhsRowIdx += kernelRows {
		for rhsColIdx := 0; rhsColIdx < rhsActiveCols; rhsColIdx += kernelCols {
			outputRowStart := lhsRowIdx
			outputColStart := rhsColIdx
			outputStride := rhsPanelCols

			// 16 accumulators (8 rows x 2 RHS vectors)
			accum_lhs0_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs0_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs0_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs0_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs1_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs1_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs1_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs1_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs2_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs2_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs2_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs2_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs3_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs3_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs3_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs3_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs4_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs4_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs4_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs4_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs5_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs5_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs5_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs5_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs6_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs6_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs6_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs6_rhs1 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs7_rhs0 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs7_rhs0 := archsimd.BroadcastFloat64x2(0.0)
			accum_lhs7_rhs1 := archsimd.BroadcastFloat32x4(0.0) //alt:f32|bf16|f16
			//alt:f64 accum_lhs7_rhs1 := archsimd.BroadcastFloat64x2(0.0)

			idxLHS := lhsRowIdx * contractingLen
			idxRHS := rhsColIdx * contractingLen

			rhsRowPtr := rhsBasePtr + uintptr(idxRHS*bytesPerInputElement)
			lhsRowPtr := lhsBasePtr + uintptr(idxLHS*bytesPerInputElement)

			rOffset := uintptr(0)
			lOffset := uintptr(0)

			rhsStride := uintptr(kernelCols * bytesPerInputElement)
			lhsStride := uintptr(kernelRows * bytesPerInputElement)

			for range contractingLen {
				rhsPtr0 := unsafe.Pointer(rhsRowPtr + rOffset)
				rhsPtr1 := unsafe.Pointer(rhsRowPtr + rOffset + 16) //alt:f32|f64
				rhsVec0 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr0)) //alt:f32
				rhsVec1 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr1)) //alt:f32
				//alt:f64 rhsVec0 := archsimd.LoadFloat64x2Array((*[2]float64)(rhsPtr0))
				//alt:f64 rhsVec1 := archsimd.LoadFloat64x2Array((*[2]float64)(rhsPtr1))
				//alt:bf16 rhsVec0, rhsVec1 := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(rhsPtr0)).ToFloat32()
				//alt:f16 rhsVec0, rhsVec1 := float16.LoadFloat16x8((*[8]float16.Float16)(rhsPtr0)).ToFloat32()
				rOffset += rhsStride

				lhsVal0 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 0))) //alt:f32
				//alt:bf16 lhsVal0 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 0))).Float32()
				//alt:f16 lhsVal0 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 0))).Float32()
				//alt:f64 lhsVal0 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 0)))
				lhsVec0 := archsimd.BroadcastFloat32x4(lhsVal0) //alt:f32|bf16|f16
				//alt:f64 lhsVec0 := archsimd.BroadcastFloat64x2(lhsVal0)
				accum_lhs0_rhs0 = rhsVec0.MulAdd(lhsVec0, accum_lhs0_rhs0)
				accum_lhs0_rhs1 = rhsVec1.MulAdd(lhsVec0, accum_lhs0_rhs1)

				lhsVal1 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal1 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement))).Float32()
				//alt:f16 lhsVal1 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement))).Float32()
				//alt:f64 lhsVal1 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + bytesPerInputElement)))
				lhsVec1 := archsimd.BroadcastFloat32x4(lhsVal1) //alt:f32|bf16|f16
				//alt:f64 lhsVec1 := archsimd.BroadcastFloat64x2(lhsVal1)
				accum_lhs1_rhs0 = rhsVec0.MulAdd(lhsVec1, accum_lhs1_rhs0)
				accum_lhs1_rhs1 = rhsVec1.MulAdd(lhsVec1, accum_lhs1_rhs1)

				lhsVal2 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal2 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal2 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal2 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 2*bytesPerInputElement)))
				lhsVec2 := archsimd.BroadcastFloat32x4(lhsVal2) //alt:f32|bf16|f16
				//alt:f64 lhsVec2 := archsimd.BroadcastFloat64x2(lhsVal2)
				accum_lhs2_rhs0 = rhsVec0.MulAdd(lhsVec2, accum_lhs2_rhs0)
				accum_lhs2_rhs1 = rhsVec1.MulAdd(lhsVec2, accum_lhs2_rhs1)

				lhsVal3 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal3 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal3 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal3 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 3*bytesPerInputElement)))
				lhsVec3 := archsimd.BroadcastFloat32x4(lhsVal3) //alt:f32|bf16|f16
				//alt:f64 lhsVec3 := archsimd.BroadcastFloat64x2(lhsVal3)
				accum_lhs3_rhs0 = rhsVec0.MulAdd(lhsVec3, accum_lhs3_rhs0)
				accum_lhs3_rhs1 = rhsVec1.MulAdd(lhsVec3, accum_lhs3_rhs1)

				lhsVal4 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 4*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal4 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 4*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal4 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 4*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal4 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 4*bytesPerInputElement)))
				lhsVec4 := archsimd.BroadcastFloat32x4(lhsVal4) //alt:f32|bf16|f16
				//alt:f64 lhsVec4 := archsimd.BroadcastFloat64x2(lhsVal4)
				accum_lhs4_rhs0 = rhsVec0.MulAdd(lhsVec4, accum_lhs4_rhs0)
				accum_lhs4_rhs1 = rhsVec1.MulAdd(lhsVec4, accum_lhs4_rhs1)

				lhsVal5 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 5*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal5 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 5*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal5 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 5*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal5 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 5*bytesPerInputElement)))
				lhsVec5 := archsimd.BroadcastFloat32x4(lhsVal5) //alt:f32|bf16|f16
				//alt:f64 lhsVec5 := archsimd.BroadcastFloat64x2(lhsVal5)
				accum_lhs5_rhs0 = rhsVec0.MulAdd(lhsVec5, accum_lhs5_rhs0)
				accum_lhs5_rhs1 = rhsVec1.MulAdd(lhsVec5, accum_lhs5_rhs1)

				lhsVal6 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 6*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal6 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 6*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal6 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 6*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal6 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 6*bytesPerInputElement)))
				lhsVec6 := archsimd.BroadcastFloat32x4(lhsVal6) //alt:f32|bf16|f16
				//alt:f64 lhsVec6 := archsimd.BroadcastFloat64x2(lhsVal6)
				accum_lhs6_rhs0 = rhsVec0.MulAdd(lhsVec6, accum_lhs6_rhs0)
				accum_lhs6_rhs1 = rhsVec1.MulAdd(lhsVec6, accum_lhs6_rhs1)

				lhsVal7 := *((*float32)(unsafe.Pointer(lhsRowPtr + lOffset + 7*bytesPerInputElement))) //alt:f32
				//alt:bf16 lhsVal7 := ((*bfloat16.BFloat16)(unsafe.Pointer(lhsRowPtr + lOffset + 7*bytesPerInputElement))).Float32()
				//alt:f16 lhsVal7 := ((*float16.Float16)(unsafe.Pointer(lhsRowPtr + lOffset + 7*bytesPerInputElement))).Float32()
				//alt:f64 lhsVal7 := *((*float64)(unsafe.Pointer(lhsRowPtr + lOffset + 7*bytesPerInputElement)))
				lhsVec7 := archsimd.BroadcastFloat32x4(lhsVal7) //alt:f32|bf16|f16
				//alt:f64 lhsVec7 := archsimd.BroadcastFloat64x2(lhsVal7)
				accum_lhs7_rhs0 = rhsVec0.MulAdd(lhsVec7, accum_lhs7_rhs0)
				accum_lhs7_rhs1 = rhsVec1.MulAdd(lhsVec7, accum_lhs7_rhs1)

				lOffset += lhsStride
			}

			rowByteStride := uintptr(rhsPanelCols * bytesPerOutputElement)
			outputIdx0 := uintptr((outputRowStart*outputStride + outputColStart) * bytesPerOutputElement)
			outputIdx1 := outputIdx0 + rowByteStride
			outputIdx2 := outputIdx1 + rowByteStride
			outputIdx3 := outputIdx2 + rowByteStride
			outputIdx4 := outputIdx3 + rowByteStride
			outputIdx5 := outputIdx4 + rowByteStride
			outputIdx6 := outputIdx5 + rowByteStride
			outputIdx7 := outputIdx6 + rowByteStride
			const regStride = uintptr(16)

			if !accumulate {
				accum_lhs0_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0)))               //alt:f32|bf16|f16
				accum_lhs0_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))   //alt:f32|bf16|f16
				accum_lhs1_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1)))               //alt:f32|bf16|f16
				accum_lhs1_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))   //alt:f32|bf16|f16
				accum_lhs2_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2)))               //alt:f32|bf16|f16
				accum_lhs2_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))   //alt:f32|bf16|f16
				accum_lhs3_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3)))               //alt:f32|bf16|f16
				accum_lhs3_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))   //alt:f32|bf16|f16
				accum_lhs4_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx4)))               //alt:f32|bf16|f16
				accum_lhs4_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx4 + regStride)))   //alt:f32|bf16|f16
				accum_lhs5_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx5)))               //alt:f32|bf16|f16
				accum_lhs5_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx5 + regStride)))   //alt:f32|bf16|f16
				accum_lhs6_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx6)))               //alt:f32|bf16|f16
				accum_lhs6_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx6 + regStride)))   //alt:f32|bf16|f16
				accum_lhs7_rhs0.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx7)))               //alt:f32|bf16|f16
				accum_lhs7_rhs1.StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx7 + regStride)))   //alt:f32|bf16|f16
				//alt:f64 accum_lhs0_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0)))
				//alt:f64 accum_lhs0_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))
				//alt:f64 accum_lhs1_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1)))
				//alt:f64 accum_lhs1_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))
				//alt:f64 accum_lhs2_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2)))
				//alt:f64 accum_lhs2_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))
				//alt:f64 accum_lhs3_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3)))
				//alt:f64 accum_lhs3_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))
				//alt:f64 accum_lhs4_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx4)))
				//alt:f64 accum_lhs4_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx4 + regStride)))
				//alt:f64 accum_lhs5_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx5)))
				//alt:f64 accum_lhs5_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx5 + regStride)))
				//alt:f64 accum_lhs6_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx6)))
				//alt:f64 accum_lhs6_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx6 + regStride)))
				//alt:f64 accum_lhs7_rhs0.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx7)))
				//alt:f64 accum_lhs7_rhs1.StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx7 + regStride)))
			} else {
				accum_lhs0_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0)))                             //alt:f32|bf16|f16
				accum_lhs0_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))     //alt:f32|bf16|f16
				accum_lhs1_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1)))                             //alt:f32|bf16|f16
				accum_lhs1_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))     //alt:f32|bf16|f16
				accum_lhs2_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2)))                             //alt:f32|bf16|f16
				accum_lhs2_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))     //alt:f32|bf16|f16
				accum_lhs3_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3)))                             //alt:f32|bf16|f16
				accum_lhs3_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))     //alt:f32|bf16|f16
				accum_lhs4_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx4)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx4)))                             //alt:f32|bf16|f16
				accum_lhs4_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx4 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx4 + regStride)))     //alt:f32|bf16|f16
				accum_lhs5_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx5)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx5)))                             //alt:f32|bf16|f16
				accum_lhs5_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx5 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx5 + regStride)))     //alt:f32|bf16|f16
				accum_lhs6_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx6)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx6)))                             //alt:f32|bf16|f16
				accum_lhs6_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx6 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx6 + regStride)))     //alt:f32|bf16|f16
				accum_lhs7_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx7)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx7)))                             //alt:f32|bf16|f16
				accum_lhs7_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx7 + regStride)))).StoreArray((*[4]float32)(unsafe.Pointer(outputBasePtr + outputIdx7 + regStride)))     //alt:f32|bf16|f16
				//alt:f64 accum_lhs0_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0)))
				//alt:f64 accum_lhs0_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx0 + regStride)))
				//alt:f64 accum_lhs1_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1)))
				//alt:f64 accum_lhs1_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx1 + regStride)))
				//alt:f64 accum_lhs2_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2)))
				//alt:f64 accum_lhs2_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx2 + regStride)))
				//alt:f64 accum_lhs3_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3)))
				//alt:f64 accum_lhs3_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx3 + regStride)))
				//alt:f64 accum_lhs4_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx4)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx4)))
				//alt:f64 accum_lhs4_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx4 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx4 + regStride)))
				//alt:f64 accum_lhs5_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx5)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx5)))
				//alt:f64 accum_lhs5_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx5 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx5 + regStride)))
				//alt:f64 accum_lhs6_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx6)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx6)))
				//alt:f64 accum_lhs6_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx6 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx6 + regStride)))
				//alt:f64 accum_lhs7_rhs0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx7)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx7)))
				//alt:f64 accum_lhs7_rhs1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx7 + regStride)))).StoreArray((*[2]float64)(unsafe.Pointer(outputBasePtr + outputIdx7 + regStride)))
			}
		}
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
	//alt:f16 "github.com/gomlx/compute/dtypes/float16"
)

// neonRouterFloat32 routes the matrix multiplication to the appropriate NEON implementation.
func neonRouterFloat32( //alt:f32
	//alt:bf16 func neonRouterBFloat16(
	//alt:f16 func neonRouterFloat16(
	//alt:f64 func neonRouterFloat64(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32, //alt:f32|bf16|f16
	//alt:f64 output []float64,
	nodeData *dot.NodeData) {

	flops := batchSize * lhsCrossSize * rhsCrossSize * contractingSize
	hasPackedCache := nodeData != nil && (nodeData.LHSConstantData.CachedForm() != nil || nodeData.RHSConstantData.CachedForm() != nil)
	mustUseLarge := (lhs == nil && nodeData != nil && nodeData.LHSConstantData != nil) ||
		(rhs == nil && nodeData != nil && nodeData.RHSConstantData != nil)
	forcePack := nodeData != nil && (nodeData.LHSConstantData.ForceCache() || nodeData.RHSConstantData.ForceCache())
	if !mustUseLarge && !matmul.ForceLargeVariant && (matmul.ForceSmallVariant || (!hasPackedCache && !forcePack && flops < matmul.SmallMatMulSizeThreshold)) {
		neonSmallFloat32Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output) //alt:f32
		//alt:bf16 neonSmallBFloat16Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		//alt:f16 neonSmallFloat16Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		//alt:f64 neonSmallFloat64Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		return
	}

	neonLargeFloat32(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData) //alt:f32
	//alt:bf16 neonLargeBFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
	//alt:f16 neonLargeFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
	//alt:f64 neonLargeFloat64(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
}

// smeRouterFloat32 routes the matrix multiplication to the appropriate ARM64 SME or NEON implementation.
func smeRouterFloat32( //alt:f32
	//alt:bf16 func smeRouterBFloat16(
	//alt:f16 func smeRouterFloat16(
	//alt:f64 func smeRouterFloat64(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	//alt:f16 lhs, rhs []float16.Float16,
	//alt:f64 lhs, rhs []float64,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32, //alt:f32|bf16|f16
	//alt:f64 output []float64,
	nodeData *dot.NodeData) {

	flops := batchSize * lhsCrossSize * rhsCrossSize * contractingSize
	hasPackedCache := nodeData != nil && (nodeData.LHSConstantData.CachedForm() != nil || nodeData.RHSConstantData.CachedForm() != nil)
	mustUseLarge := (lhs == nil && nodeData != nil && nodeData.LHSConstantData != nil) ||
		(rhs == nil && nodeData != nil && nodeData.RHSConstantData != nil)
	forcePack := nodeData != nil && (nodeData.LHSConstantData.ForceCache() || nodeData.RHSConstantData.ForceCache())
	if !mustUseLarge && !matmul.ForceLargeVariant && (matmul.ForceSmallVariant || (!hasPackedCache && !forcePack && (flops < SMESmallMatMulSizeThreshold || lhsCrossSize < 8 || rhsCrossSize < 8))) {
		if matmul.ForceSmallVariant || flops < matmul.SmallMatMulSizeThreshold {
			neonSmallFloat32Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output) //alt:f32
			//alt:bf16 neonSmallBFloat16Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
			//alt:f16 neonSmallFloat16Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
			//alt:f64 neonSmallFloat64Parallel(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
			return
		}
		neonLargeFloat32(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData) //alt:f32
		//alt:bf16 neonLargeBFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
		//alt:f16 neonLargeFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
		//alt:f64 neonLargeFloat64(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
		return
	}

	smeLargeFloat32(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData) //alt:f32
	//alt:bf16 smeLargeBFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
	//alt:f16 smeLargeFloat16(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
	//alt:f64 smeLargeFloat64(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)
}

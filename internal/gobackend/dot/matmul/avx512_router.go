// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package matmul

import (
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"k8s.io/klog/v2"

	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
)

// avx512RouterFloat32 implements a router that decides between the small no-SIMD version
// or the large AVX512 version for Float32.
func avx512RouterFloat32( //alt:f32
	//alt:bf16 func avx512RouterBFloat16(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32) {

	// Check if small matrix multiplication kernel can be used.
	flops := batchSize * lhsCrossSize * rhsCrossSize * contractingSize
	useSmallVariant := !ForceLargeVariant && (ForceSmallVariant || flops < noSIMDSmallMatMulSizeThreshold)
	if klog.V(1).Enabled() {
		variant := "large (avx512)"
		if useSmallVariant {
			variant = "small (nosimd)"
		}
		klog.Infof("Using %s variant for AVX512 dot-product kernel, layout=%s, "+
			"lhs=[%d, %d, %d], rhs=[%d, %d, %d], output=[%d, %d, %d]",
			variant, layout, batchSize, lhsCrossSize, contractingSize,
			batchSize, rhsCrossSize, contractingSize,
			batchSize, lhsCrossSize, rhsCrossSize)
	}

	if useSmallVariant {
		// Vector width: 16 for float32, 32 for bfloat16
		const vecWidth = 16 //alt:f32
		//alt:bf16 const vecWidth = 32
		if contractingSize >= vecWidth {
			avx512SmallFloat32Parallel( //alt:f32
				//alt:bf16 avx512SmallBFloat16Parallel(
				backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
			return
		}

		if layout == dot.LayoutNonTransposed {
			noSIMDRouter[float32, float32]( //alt:f32
				//alt:bf16 noSIMDHalfPrecisionRouter[bfloat16.BFloat16, float32](
				backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		} else {
			noSIMDRouter[float32, float32]( //alt:f32
				//alt:bf16 noSIMDHalfPrecisionRouter[bfloat16.BFloat16, float32](
				backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		}
		return
	}

	avx512LargeFloat32( //alt:f32
		//alt:bf16 avx512LargeBFloat16(
		backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
}

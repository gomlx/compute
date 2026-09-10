// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul/matmultest"
)

func TestAVX2Packing(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported on this architecture")
	}

	t.Run("Float32", func(t *testing.T) {
		matmultest.RunPackLHSTests(t, avx2PackLHSKernelRows4[float32], 4)
		matmultest.RunPackRHSTests(t, avx2PackRHSNonTransposed[float32], 16)
		matmultest.RunApplyPackedOutputTests(t, avx2ApplyPackedOutputFloat32)
	})
	t.Run("BFloat16", func(t *testing.T) {
		matmultest.RunPackLHSTestsHalfPrecision(t, avx2PackLHSKernelRows4[bfloat16.BFloat16], 4)
		matmultest.RunPackRHSTestsHalfPrecision(t, avx2PackRHSNonTransposed[bfloat16.BFloat16], 16)
	})
	t.Run("Float16", func(t *testing.T) {
		matmultest.RunPackLHSTestsHalfPrecision(t, avx2PackLHSKernelRows4[float16.Float16], 4)
		matmultest.RunPackRHSTestsHalfPrecision(t, avx2PackRHSNonTransposed[float16.Float16], 16)
	})
	t.Run("Float64", func(t *testing.T) {
		matmultest.RunPackLHSTests(t, avx2PackLHSKernelRows4[float64], 4)
		matmultest.RunPackRHSTests(t, avx2PackRHSNonTransposed[float64], 8)
		matmultest.RunApplyPackedOutputTests(t, avx2ApplyPackedOutputFloat64)
	})
}

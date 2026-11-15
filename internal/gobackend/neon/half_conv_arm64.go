// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

package neon

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsNEONAllowed {
		f32ToBF16 := bfloat16.FromFloat32SliceNEON
		if gobackend.IsARM64BF16Allowed {
			f32ToBF16 = neonBF16Float32ToBFloat16Slice
		}
		gobackend.SetHalfSliceConverters(gobackend.PriorityNEON, gobackend.HalfSliceConverters{
			Float16ToFloat32:  float16.ToFloat32SliceNEON,
			BFloat16ToFloat32: bfloat16.ToFloat32SliceNEON,
			Float32ToFloat16:  float16.FromFloat32SliceNEON,
			Float32ToBFloat16: f32ToBF16,
		})
	}
}

//go:noescape
func float32ToBFloat16ARM64BF16Asm(dst, src unsafe.Pointer, n int)

// neonBF16Float32ToBFloat16Slice converts a slice of float32 values to BFloat16 using ARMv8.6-A FEAT_BF16 BFCVTN/BFCVTN2 instructions.
func neonBF16Float32ToBFloat16Slice(dst []bfloat16.BFloat16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	vecN := n &^ 3
	if vecN > 0 {
		float32ToBFloat16ARM64BF16Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), vecN)
	}
	for i := vecN; i < n; i++ {
		dst[i] = bfloat16.FromFloat32(src[i])
	}
}

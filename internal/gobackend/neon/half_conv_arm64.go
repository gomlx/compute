// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsNEONAllowed {
		gobackend.SetHalfSliceConverters(gobackend.PriorityNEON, gobackend.HalfSliceConverters{
			Float16ToFloat32:  float16.ToFloat32SliceNEON,
			BFloat16ToFloat32: bfloat16.ToFloat32SliceNEON,
			Float32ToFloat16:  float16.FromFloat32SliceNEON,
			Float32ToBFloat16: bfloat16.FromFloat32SliceNEON,
		})
	}
}

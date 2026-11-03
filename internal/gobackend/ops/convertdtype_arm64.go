// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package ops

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if !gobackend.IsNEONAllowed {
		return
	}
	if gobackend.IsARM64FP16Allowed {
		ConvertDTypePairMap.Register(dtypes.Float16, dtypes.Float32, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
			float16.ToFloat32SliceNEON(output.Flat.([]float32), operand.Flat.([]float16.Float16))
		})
		ConvertDTypePairMap.Register(dtypes.Float32, dtypes.Float16, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
			float16.FromFloat32SliceNEON(output.Flat.([]float16.Float16), operand.Flat.([]float32))
		})
	}
	ConvertDTypePairMap.Register(dtypes.BFloat16, dtypes.Float32, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
		bfloat16.ToFloat32SliceNEON(output.Flat.([]float32), operand.Flat.([]bfloat16.BFloat16))
	})
	ConvertDTypePairMap.Register(dtypes.Float32, dtypes.BFloat16, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
		bfloat16.FromFloat32SliceNEON(output.Flat.([]bfloat16.BFloat16), operand.Flat.([]float32))
	})
}

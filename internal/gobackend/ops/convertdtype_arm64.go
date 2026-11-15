// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

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
	ConvertDTypePairMap.Register(dtypes.Float16, dtypes.Float32, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
		gobackend.ActiveHalfSliceConverters().Float16ToFloat32(output.Flat.([]float32), operand.Flat.([]float16.Float16))
	})
	ConvertDTypePairMap.Register(dtypes.Float32, dtypes.Float16, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
		gobackend.ActiveHalfSliceConverters().Float32ToFloat16(output.Flat.([]float16.Float16), operand.Flat.([]float32))
	})
	ConvertDTypePairMap.Register(dtypes.BFloat16, dtypes.Float32, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
		gobackend.ActiveHalfSliceConverters().BFloat16ToFloat32(output.Flat.([]float32), operand.Flat.([]bfloat16.BFloat16))
	})
	ConvertDTypePairMap.Register(dtypes.Float32, dtypes.BFloat16, gobackend.PriorityNEON, func(operand, output *gobackend.Buffer) {
		gobackend.ActiveHalfSliceConverters().Float32ToBFloat16(output.Flat.([]bfloat16.BFloat16), operand.Flat.([]float32))
	})
}

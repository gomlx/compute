// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package activations

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

const PriorityNoSIMD = gobackend.PriorityGeneric

func init() {
	registerNoSIMD()
}

func registerNoSIMD() {
	// Float32
	Register[float32]("nosimd:relu", compute.ActivationRelu, reluNoSIMD[float32], PriorityNoSIMD)
	Register[float32]("nosimd:hardswish", compute.ActivationHardSwish, hardSwishNoSIMD[float32], PriorityNoSIMD)
	Register[float32]("nosimd:silu", compute.ActivationSilu, siluNoSIMD[float32], PriorityNoSIMD)
	Register[float32]("nosimd:gelu", compute.ActivationGelu, geluNoSIMD[float32], PriorityNoSIMD)
	Register[float32]("nosimd:tanh", compute.ActivationTanh, tanhNoSIMD[float32], PriorityNoSIMD)

	// Float64
	Register[float64]("nosimd:relu", compute.ActivationRelu, reluNoSIMD[float64], PriorityNoSIMD)
	Register[float64]("nosimd:hardswish", compute.ActivationHardSwish, hardSwishNoSIMD[float64], PriorityNoSIMD)
	Register[float64]("nosimd:silu", compute.ActivationSilu, siluNoSIMD[float64], PriorityNoSIMD)
	Register[float64]("nosimd:gelu", compute.ActivationGelu, geluNoSIMD[float64], PriorityNoSIMD)
	Register[float64]("nosimd:tanh", compute.ActivationTanh, tanhNoSIMD[float64], PriorityNoSIMD)

	// BFloat16
	Register[bfloat16.BFloat16]("nosimd:relu", compute.ActivationRelu, reluBF16NoSIMD, PriorityNoSIMD)
	Register[bfloat16.BFloat16]("nosimd:hardswish", compute.ActivationHardSwish, hardSwishBF16NoSIMD, PriorityNoSIMD)
	Register[bfloat16.BFloat16]("nosimd:silu", compute.ActivationSilu, siluBF16NoSIMD, PriorityNoSIMD)
	Register[bfloat16.BFloat16]("nosimd:gelu", compute.ActivationGelu, geluBF16NoSIMD, PriorityNoSIMD)
	Register[bfloat16.BFloat16]("nosimd:tanh", compute.ActivationTanh, tanhBF16NoSIMD, PriorityNoSIMD)

	// Float16
	Register[float16.Float16]("nosimd:relu", compute.ActivationRelu, reluF16NoSIMD, PriorityNoSIMD)
	Register[float16.Float16]("nosimd:hardswish", compute.ActivationHardSwish, hardSwishF16NoSIMD, PriorityNoSIMD)
	Register[float16.Float16]("nosimd:silu", compute.ActivationSilu, siluF16NoSIMD, PriorityNoSIMD)
	Register[float16.Float16]("nosimd:gelu", compute.ActivationGelu, geluF16NoSIMD, PriorityNoSIMD)
	Register[float16.Float16]("nosimd:tanh", compute.ActivationTanh, tanhF16NoSIMD, PriorityNoSIMD)
}

func reluNoSIMD[T float32 | float64](data []T) {
	for i, x := range data {
		if x < 0 {
			data[i] = 0
		}
	}
}

func hardSwishNoSIMD[T float32 | float64](data []T) {
	const scale = 1.0 / 6.0
	const bias = 0.5
	for i, x := range data {
		shapeX := min(max(x*scale+bias, 0), 1)
		data[i] = x * shapeX
	}
}

func siluNoSIMD[T float32 | float64](data []T) {
	for i, x := range data {
		data[i] = x / (1.0 + T(math.Exp(float64(-x))))
	}
}

func geluNoSIMD[T float32 | float64](data []T) {
	sqrt2ByPi := T(math.Sqrt(2.0 / math.Pi))
	for i, x := range data {
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		data[i] = x * 0.5 * (1.0 + T(math.Tanh(float64(inner))))
	}
}

func tanhNoSIMD[T float32 | float64](data []T) {
	for i, x := range data {
		data[i] = T(math.Tanh(float64(x)))
	}
}

// BFloat16 implementations:
func reluBF16NoSIMD(data []bfloat16.BFloat16) {
	for i, x := range data {
		if x.Float32() < 0 {
			data[i] = bfloat16.BFloat16(0)
		}
	}
}

func hardSwishBF16NoSIMD(data []bfloat16.BFloat16) {
	const scale = float32(1.0 / 6.0)
	const bias = float32(0.5)
	for i, v := range data {
		x := v.Float32()
		shapeX := min(max(x*scale+bias, 0), 1)
		data[i] = bfloat16.FromFloat32(x * shapeX)
	}
}

func siluBF16NoSIMD(data []bfloat16.BFloat16) {
	for i, v := range data {
		x := v.Float32()
		val := x / (1.0 + float32(math.Exp(float64(-x))))
		data[i] = bfloat16.FromFloat32(val)
	}
}

func geluBF16NoSIMD(data []bfloat16.BFloat16) {
	sqrt2ByPi := float32(math.Sqrt(2.0 / math.Pi))
	for i, v := range data {
		x := v.Float32()
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		val := x * 0.5 * (1.0 + float32(math.Tanh(float64(inner))))
		data[i] = bfloat16.FromFloat32(val)
	}
}

func tanhBF16NoSIMD(data []bfloat16.BFloat16) {
	for i, v := range data {
		val := float32(math.Tanh(float64(v.Float32())))
		data[i] = bfloat16.FromFloat32(val)
	}
}

// Float16 implementations:
func reluF16NoSIMD(data []float16.Float16) {
	for i, x := range data {
		if x.Float32() < 0 {
			data[i] = float16.Float16(0)
		}
	}
}

func hardSwishF16NoSIMD(data []float16.Float16) {
	const scale = float32(1.0 / 6.0)
	const bias = float32(0.5)
	for i, v := range data {
		x := v.Float32()
		shapeX := min(max(x*scale+bias, 0), 1)
		data[i] = float16.FromFloat32(x * shapeX)
	}
}

func siluF16NoSIMD(data []float16.Float16) {
	for i, v := range data {
		x := v.Float32()
		val := x / (1.0 + float32(math.Exp(float64(-x))))
		data[i] = float16.FromFloat32(val)
	}
}

func geluF16NoSIMD(data []float16.Float16) {
	sqrt2ByPi := float32(math.Sqrt(2.0 / math.Pi))
	for i, v := range data {
		x := v.Float32()
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		val := x * 0.5 * (1.0 + float32(math.Tanh(float64(inner))))
		data[i] = float16.FromFloat32(val)
	}
}

func tanhF16NoSIMD(data []float16.Float16) {
	for i, v := range data {
		val := float32(math.Tanh(float64(v.Float32())))
		data[i] = float16.FromFloat32(val)
	}
}

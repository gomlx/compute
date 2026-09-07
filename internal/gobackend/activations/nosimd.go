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
	RegisterKernel[float32]("nosimd:relu", compute.ActivationRelu, ReluNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:sigmoid", compute.ActivationSigmoid, SigmoidNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:hardsigmoid", compute.ActivationHardSigmoid, HardSigmoidNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:leakyrelu", compute.ActivationLeakyRelu, LeakyReluNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:selu", compute.ActivationSelu, SeluNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:silu", compute.ActivationSilu, SiluNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:hardswish", compute.ActivationHardSwish, HardSwishNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:tanh", compute.ActivationTanh, TanhNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:gelu", compute.ActivationGelu, geluExactNoSIMD[float32], PriorityNoSIMD)
	RegisterKernel[float32]("nosimd:geluapprox", compute.ActivationGeluApproximate, GeluApproxNoSIMD[float32], PriorityNoSIMD)

	// Float64
	RegisterKernel[float64]("nosimd:relu", compute.ActivationRelu, ReluNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:sigmoid", compute.ActivationSigmoid, SigmoidNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:hardsigmoid", compute.ActivationHardSigmoid, HardSigmoidNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:leakyrelu", compute.ActivationLeakyRelu, LeakyReluNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:selu", compute.ActivationSelu, SeluNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:silu", compute.ActivationSilu, SiluNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:hardswish", compute.ActivationHardSwish, HardSwishNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:tanh", compute.ActivationTanh, TanhNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:gelu", compute.ActivationGelu, geluExactNoSIMD[float64], PriorityNoSIMD)
	RegisterKernel[float64]("nosimd:geluapprox", compute.ActivationGeluApproximate, GeluApproxNoSIMD[float64], PriorityNoSIMD)

	// BFloat16
	RegisterKernel[bfloat16.BFloat16]("nosimd:relu", compute.ActivationRelu, ReluBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:sigmoid", compute.ActivationSigmoid, sigmoidBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:hardsigmoid", compute.ActivationHardSigmoid, hardSigmoidBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:leakyrelu", compute.ActivationLeakyRelu, leakyReluBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:selu", compute.ActivationSelu, seluBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:silu", compute.ActivationSilu, SiluBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:hardswish", compute.ActivationHardSwish, hardSwishBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:tanh", compute.ActivationTanh, tanhBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:gelu", compute.ActivationGelu, geluExactBF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[bfloat16.BFloat16]("nosimd:geluapprox", compute.ActivationGeluApproximate, geluApproxBF16NoSIMD, PriorityNoSIMD)

	// Float16
	RegisterKernel[float16.Float16]("nosimd:relu", compute.ActivationRelu, ReluF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:sigmoid", compute.ActivationSigmoid, sigmoidF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:hardsigmoid", compute.ActivationHardSigmoid, hardSigmoidF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:leakyrelu", compute.ActivationLeakyRelu, leakyReluF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:selu", compute.ActivationSelu, seluF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:silu", compute.ActivationSilu, SiluF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:hardswish", compute.ActivationHardSwish, hardSwishF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:tanh", compute.ActivationTanh, tanhF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:gelu", compute.ActivationGelu, geluExactF16NoSIMD, PriorityNoSIMD)
	RegisterKernel[float16.Float16]("nosimd:geluapprox", compute.ActivationGeluApproximate, geluApproxF16NoSIMD, PriorityNoSIMD)

	// SwiGLU
	RegisterSwiGLU[float32]("nosimd:swiglu", swigluNoSIMD[float32], PriorityNoSIMD)
	RegisterSwiGLU[float64]("nosimd:swiglu", swigluNoSIMD[float64], PriorityNoSIMD)
	RegisterSwiGLU[bfloat16.BFloat16]("nosimd:swiglu", swigluBF16NoSIMD, PriorityNoSIMD)
	RegisterSwiGLU[float16.Float16]("nosimd:swiglu", swigluF16NoSIMD, PriorityNoSIMD)

	// VJP Float32
	RegisterVJPKernel[float32]("nosimd:vjp_relu", compute.ActivationRelu, vjpReluNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_sigmoid", compute.ActivationSigmoid, vjpSigmoidNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_hardsigmoid", compute.ActivationHardSigmoid, vjpHardSigmoidNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_leakyrelu", compute.ActivationLeakyRelu, vjpLeakyReluNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_selu", compute.ActivationSelu, vjpSeluNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_silu", compute.ActivationSilu, vjpSiluNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_hardswish", compute.ActivationHardSwish, vjpHardSwishNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_tanh", compute.ActivationTanh, vjpTanhNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_gelu", compute.ActivationGelu, vjpGeluExactNoSIMD[float32], PriorityNoSIMD)
	RegisterVJPKernel[float32]("nosimd:vjp_geluapprox", compute.ActivationGeluApproximate, vjpGeluApproxNoSIMD[float32], PriorityNoSIMD)

	// VJP Float64
	RegisterVJPKernel[float64]("nosimd:vjp_relu", compute.ActivationRelu, vjpReluNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_sigmoid", compute.ActivationSigmoid, vjpSigmoidNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_hardsigmoid", compute.ActivationHardSigmoid, vjpHardSigmoidNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_leakyrelu", compute.ActivationLeakyRelu, vjpLeakyReluNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_selu", compute.ActivationSelu, vjpSeluNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_silu", compute.ActivationSilu, vjpSiluNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_hardswish", compute.ActivationHardSwish, vjpHardSwishNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_tanh", compute.ActivationTanh, vjpTanhNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_gelu", compute.ActivationGelu, vjpGeluExactNoSIMD[float64], PriorityNoSIMD)
	RegisterVJPKernel[float64]("nosimd:vjp_geluapprox", compute.ActivationGeluApproximate, vjpGeluApproxNoSIMD[float64], PriorityNoSIMD)

	// VJP BFloat16
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_relu", compute.ActivationRelu, MakeBF16VJPKernelFromF32(vjpReluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_sigmoid", compute.ActivationSigmoid, MakeBF16VJPKernelFromF32(vjpSigmoidNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_hardsigmoid", compute.ActivationHardSigmoid, MakeBF16VJPKernelFromF32(vjpHardSigmoidNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_leakyrelu", compute.ActivationLeakyRelu, MakeBF16VJPKernelFromF32(vjpLeakyReluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_selu", compute.ActivationSelu, MakeBF16VJPKernelFromF32(vjpSeluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_silu", compute.ActivationSilu, MakeBF16VJPKernelFromF32(vjpSiluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_hardswish", compute.ActivationHardSwish, MakeBF16VJPKernelFromF32(vjpHardSwishNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_tanh", compute.ActivationTanh, MakeBF16VJPKernelFromF32(vjpTanhNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_gelu", compute.ActivationGelu, MakeBF16VJPKernelFromF32(vjpGeluExactNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("nosimd:vjp_geluapprox", compute.ActivationGeluApproximate, MakeBF16VJPKernelFromF32(vjpGeluApproxNoSIMD[float32]), PriorityNoSIMD)

	// VJP Float16
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_relu", compute.ActivationRelu, MakeF16VJPKernelFromF32(vjpReluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_sigmoid", compute.ActivationSigmoid, MakeF16VJPKernelFromF32(vjpSigmoidNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_hardsigmoid", compute.ActivationHardSigmoid, MakeF16VJPKernelFromF32(vjpHardSigmoidNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_leakyrelu", compute.ActivationLeakyRelu, MakeF16VJPKernelFromF32(vjpLeakyReluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_selu", compute.ActivationSelu, MakeF16VJPKernelFromF32(vjpSeluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_silu", compute.ActivationSilu, MakeF16VJPKernelFromF32(vjpSiluNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_hardswish", compute.ActivationHardSwish, MakeF16VJPKernelFromF32(vjpHardSwishNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_tanh", compute.ActivationTanh, MakeF16VJPKernelFromF32(vjpTanhNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_gelu", compute.ActivationGelu, MakeF16VJPKernelFromF32(vjpGeluExactNoSIMD[float32]), PriorityNoSIMD)
	RegisterVJPKernel[float16.Float16]("nosimd:vjp_geluapprox", compute.ActivationGeluApproximate, MakeF16VJPKernelFromF32(vjpGeluApproxNoSIMD[float32]), PriorityNoSIMD)

	// VJP SwiGLU
	RegisterSwiGLUVJP[float32]("nosimd:vjp_swiglu", vjpSwiGLUNoSIMD[float32], PriorityNoSIMD)
	RegisterSwiGLUVJP[float64]("nosimd:vjp_swiglu", vjpSwiGLUNoSIMD[float64], PriorityNoSIMD)
	RegisterSwiGLUVJP[bfloat16.BFloat16]("nosimd:vjp_swiglu", MakeBF16SwiGLUVJPFromF32(vjpSwiGLUNoSIMD[float32]), PriorityNoSIMD)
	RegisterSwiGLUVJP[float16.Float16]("nosimd:vjp_swiglu", MakeF16SwiGLUVJPFromF32(vjpSwiGLUNoSIMD[float32]), PriorityNoSIMD)
}

// Float32 / Float64 kernels:

func ReluNoSIMD[T float32 | float64](in, out []T) {
	for i, x := range in {
		if x < 0 {
			out[i] = 0
		} else {
			out[i] = x
		}
	}
}

func SigmoidNoSIMD[T float32 | float64](in, out []T) {
	for i, x := range in {
		out[i] = T(1.0 / (1.0 + math.Exp(float64(-x))))
	}
}

func HardSigmoidNoSIMD[T float32 | float64](in, out []T) {
	for i, x := range in {
		v := x*0.2 + 0.5
		if v < 0 {
			out[i] = 0
		} else if v > 1 {
			out[i] = 1
		} else {
			out[i] = v
		}
	}
}

func LeakyReluNoSIMD[T float32 | float64](in, out []T) {
	const alpha = 0.3
	for i, x := range in {
		if x >= 0 {
			out[i] = x
		} else {
			out[i] = T(alpha) * x
		}
	}
}

const (
	seluAlpha      = 1.6732632423543772848170429916717
	seluScale      = 1.0507009873554804934193349852946
	seluScaleAlpha = seluScale * seluAlpha
)

func SeluNoSIMD[T float32 | float64](in, out []T) {
	for i, x := range in {
		if x > 0 {
			out[i] = T(seluScale) * x
		} else {
			out[i] = T(seluScaleAlpha * (math.Exp(float64(x)) - 1.0))
		}
	}
}

func SiluNoSIMD[T float32 | float64](in, out []T) {
	for i, x := range in {
		out[i] = x / (1.0 + T(math.Exp(float64(-x))))
	}
}

func HardSwishNoSIMD[T float32 | float64](in, out []T) {
	const scale = 1.0 / 6.0
	const bias = 0.5
	for i, x := range in {
		shapeX := min(max(x*scale+bias, 0), 1)
		out[i] = x * shapeX
	}
}

func TanhNoSIMD[T float32 | float64](in, out []T) {
	for i, x := range in {
		out[i] = T(math.Tanh(float64(x)))
	}
}

func geluExactNoSIMD[T float32 | float64](in, out []T) {
	invSqrt2 := 1.0 / math.Sqrt2
	for i, x := range in {
		out[i] = x * 0.5 * T(1.0+math.Erf(float64(x)*invSqrt2))
	}
}

func GeluApproxNoSIMD[T float32 | float64](in, out []T) {
	sqrt2ByPi := T(math.Sqrt(2.0 / math.Pi))
	for i, x := range in {
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		out[i] = x * 0.5 * (1.0 + T(math.Tanh(float64(inner))))
	}
}

func swigluNoSIMD[T float32 | float64](in, out []T, numRows, hiddenDim int) {
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		for j := range hiddenDim {
			gate := in[inOffset+j]
			val := in[inOffset+hiddenDim+j]
			siluGate := gate / (1.0 + T(math.Exp(float64(-gate))))
			out[outOffset+j] = siluGate * val
		}
	}
}

// BFloat16 implementations:

func ReluBF16NoSIMD(in, out []bfloat16.BFloat16) {
	for i, v := range in {
		if v.Float32() < 0 {
			out[i] = bfloat16.BFloat16(0)
		} else {
			out[i] = v
		}
	}
}

func sigmoidBF16NoSIMD(in, out []bfloat16.BFloat16) {
	for i, v := range in {
		x := v.Float32()
		val := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		out[i] = bfloat16.FromFloat32(val)
	}
}

func hardSigmoidBF16NoSIMD(in, out []bfloat16.BFloat16) {
	for i, v := range in {
		x := v.Float32()
		val := min(max(x*0.2+0.5, 0), 1)
		out[i] = bfloat16.FromFloat32(val)
	}
}

func leakyReluBF16NoSIMD(in, out []bfloat16.BFloat16) {
	for i, v := range in {
		x := v.Float32()
		if x >= 0 {
			out[i] = v
		} else {
			out[i] = bfloat16.FromFloat32(0.3 * x)
		}
	}
}

func seluBF16NoSIMD(in, out []bfloat16.BFloat16) {
	for i, v := range in {
		x := v.Float32()
		var val float32
		if x > 0 {
			val = float32(seluScale) * x
		} else {
			val = float32(seluScaleAlpha * (math.Exp(float64(x)) - 1.0))
		}
		out[i] = bfloat16.FromFloat32(val)
	}
}

func SiluBF16NoSIMD(in, out []bfloat16.BFloat16) {
	for i, v := range in {
		x := v.Float32()
		val := x / (1.0 + float32(math.Exp(float64(-x))))
		out[i] = bfloat16.FromFloat32(val)
	}
}

func hardSwishBF16NoSIMD(in, out []bfloat16.BFloat16) {
	const scale = float32(1.0 / 6.0)
	const bias = float32(0.5)
	for i, v := range in {
		x := v.Float32()
		shapeX := min(max(x*scale+bias, 0), 1)
		out[i] = bfloat16.FromFloat32(x * shapeX)
	}
}

func tanhBF16NoSIMD(in, out []bfloat16.BFloat16) {
	for i, v := range in {
		val := float32(math.Tanh(float64(v.Float32())))
		out[i] = bfloat16.FromFloat32(val)
	}
}

func geluExactBF16NoSIMD(in, out []bfloat16.BFloat16) {
	invSqrt2 := 1.0 / math.Sqrt2
	for i, v := range in {
		x := v.Float32()
		val := x * 0.5 * float32(1.0+math.Erf(float64(x)*invSqrt2))
		out[i] = bfloat16.FromFloat32(val)
	}
}

func geluApproxBF16NoSIMD(in, out []bfloat16.BFloat16) {
	sqrt2ByPi := float32(math.Sqrt(2.0 / math.Pi))
	for i, v := range in {
		x := v.Float32()
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		val := x * 0.5 * (1.0 + float32(math.Tanh(float64(inner))))
		out[i] = bfloat16.FromFloat32(val)
	}
}

func swigluBF16NoSIMD(in, out []bfloat16.BFloat16, numRows, hiddenDim int) {
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		for j := range hiddenDim {
			gate := in[inOffset+j].Float32()
			val := in[inOffset+hiddenDim+j].Float32()
			siluGate := gate / (1.0 + float32(math.Exp(float64(-gate))))
			out[outOffset+j] = bfloat16.FromFloat32(siluGate * val)
		}
	}
}

// Float16 implementations:

func ReluF16NoSIMD(in, out []float16.Float16) {
	for i, v := range in {
		if v.Float32() < 0 {
			out[i] = float16.Float16(0)
		} else {
			out[i] = v
		}
	}
}

func sigmoidF16NoSIMD(in, out []float16.Float16) {
	for i, v := range in {
		x := v.Float32()
		val := float32(1.0 / (1.0 + math.Exp(float64(-x))))
		out[i] = float16.FromFloat32(val)
	}
}

func hardSigmoidF16NoSIMD(in, out []float16.Float16) {
	for i, v := range in {
		x := v.Float32()
		val := min(max(x*0.2+0.5, 0), 1)
		out[i] = float16.FromFloat32(val)
	}
}

func leakyReluF16NoSIMD(in, out []float16.Float16) {
	for i, v := range in {
		x := v.Float32()
		if x >= 0 {
			out[i] = v
		} else {
			out[i] = float16.FromFloat32(0.3 * x)
		}
	}
}

func seluF16NoSIMD(in, out []float16.Float16) {
	for i, v := range in {
		x := v.Float32()
		var val float32
		if x > 0 {
			val = float32(seluScale) * x
		} else {
			val = float32(seluScaleAlpha * (math.Exp(float64(x)) - 1.0))
		}
		out[i] = float16.FromFloat32(val)
	}
}

func SiluF16NoSIMD(in, out []float16.Float16) {
	for i, v := range in {
		x := v.Float32()
		val := x / (1.0 + float32(math.Exp(float64(-x))))
		out[i] = float16.FromFloat32(val)
	}
}

func hardSwishF16NoSIMD(in, out []float16.Float16) {
	const scale = float32(1.0 / 6.0)
	const bias = float32(0.5)
	for i, v := range in {
		x := v.Float32()
		shapeX := min(max(x*scale+bias, 0), 1)
		out[i] = float16.FromFloat32(x * shapeX)
	}
}

func tanhF16NoSIMD(in, out []float16.Float16) {
	for i, v := range in {
		val := float32(math.Tanh(float64(v.Float32())))
		out[i] = float16.FromFloat32(val)
	}
}

func geluExactF16NoSIMD(in, out []float16.Float16) {
	invSqrt2 := 1.0 / math.Sqrt2
	for i, v := range in {
		x := v.Float32()
		val := x * 0.5 * float32(1.0+math.Erf(float64(x)*invSqrt2))
		out[i] = float16.FromFloat32(val)
	}
}

func geluApproxF16NoSIMD(in, out []float16.Float16) {
	sqrt2ByPi := float32(math.Sqrt(2.0 / math.Pi))
	for i, v := range in {
		x := v.Float32()
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		val := x * 0.5 * (1.0 + float32(math.Tanh(float64(inner))))
		out[i] = float16.FromFloat32(val)
	}
}

func swigluF16NoSIMD(in, out []float16.Float16, numRows, hiddenDim int) {
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		for j := range hiddenDim {
			gate := in[inOffset+j].Float32()
			val := in[inOffset+hiddenDim+j].Float32()
			siluGate := gate / (1.0 + float32(math.Exp(float64(-gate))))
			out[outOffset+j] = float16.FromFloat32(siluGate * val)
		}
	}
}

// VJP scalar implementations:

func vjpReluNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	if len(y) > 0 {
		for i, dOut := range dOutput {
			if y[i] > 0 {
				dx[i] = dOut
			} else {
				dx[i] = 0
			}
		}
	} else {
		for i, dOut := range dOutput {
			if x[i] > 0 {
				dx[i] = dOut
			} else {
				dx[i] = 0
			}
		}
	}
}

func vjpSigmoidNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	if len(y) > 0 {
		for i, dOut := range dOutput {
			yi := y[i]
			dx[i] = dOut * yi * (1 - yi)
		}
	} else {
		for i, dOut := range dOutput {
			s := T(1.0 / (1.0 + math.Exp(-float64(x[i]))))
			dx[i] = dOut * s * (1 - s)
		}
	}
}

func vjpHardSigmoidNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	const slope = 0.2
	if len(y) > 0 {
		for i, dOut := range dOutput {
			if y[i] > 0 && y[i] < 1 {
				dx[i] = dOut * slope
			} else {
				dx[i] = 0
			}
		}
	} else {
		for i, dOut := range dOutput {
			if x[i] > -2.5 && x[i] < 2.5 {
				dx[i] = dOut * slope
			} else {
				dx[i] = 0
			}
		}
	}
}

func vjpLeakyReluNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	const alpha = 0.3
	if len(y) > 0 {
		for i, dOut := range dOutput {
			if y[i] >= 0 {
				dx[i] = dOut
			} else {
				dx[i] = dOut * alpha
			}
		}
	} else {
		for i, dOut := range dOutput {
			if x[i] >= 0 {
				dx[i] = dOut
			} else {
				dx[i] = dOut * alpha
			}
		}
	}
}

func vjpSeluNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	const scale = 1.0507009873554804934193349852946
	const alpha = 1.6732632423543772848170429916717
	const scaleAlpha = scale * alpha
	if len(y) > 0 {
		for i, dOut := range dOutput {
			if y[i] > 0 {
				dx[i] = dOut * scale
			} else {
				dx[i] = dOut * (y[i] + scaleAlpha)
			}
		}
	} else {
		for i, dOut := range dOutput {
			if x[i] > 0 {
				dx[i] = dOut * scale
			} else {
				dx[i] = dOut * scaleAlpha * T(math.Exp(float64(x[i])))
			}
		}
	}
}

func vjpTanhNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	if len(y) > 0 {
		for i, dOut := range dOutput {
			yi := y[i]
			dx[i] = dOut * (1 - yi*yi)
		}
	} else {
		for i, dOut := range dOutput {
			t := T(math.Tanh(float64(x[i])))
			dx[i] = dOut * (1 - t*t)
		}
	}
}

func vjpSiluNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	for i, dOut := range dOutput {
		xi := float64(x[i])
		s := 1.0 / (1.0 + math.Exp(-xi))
		fPrime := s * (1.0 + xi*(1.0-s))
		dx[i] = dOut * T(fPrime)
	}
}

func vjpHardSwishNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	for i, dOut := range dOutput {
		xi := x[i]
		if xi <= -3 {
			dx[i] = 0
		} else if xi >= 3 {
			dx[i] = dOut
		} else {
			dx[i] = dOut * (xi*(1.0/3.0) + 0.5)
		}
	}
}

func vjpGeluExactNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	const invSqrt2 = 1.0 / math.Sqrt2
	const invSqrt2Pi = 0.39894228040143267793994605993438 // 1/sqrt(2*pi)
	for i, dOut := range dOutput {
		xi := float64(x[i])
		cdf := 0.5 * (1.0 + math.Erf(xi*invSqrt2))
		pdf := invSqrt2Pi * math.Exp(-0.5*xi*xi)
		fPrime := cdf + xi*pdf
		dx[i] = dOut * T(fPrime)
	}
}

func vjpGeluApproxNoSIMD[T float32 | float64](y, x, dOutput, dx []T) {
	sqrt2ByPi := math.Sqrt(2.0 / math.Pi)
	for i, dOut := range dOutput {
		xi := float64(x[i])
		x2 := xi * xi
		u := sqrt2ByPi * (xi + 0.044715*xi*x2)
		uPrime := sqrt2ByPi * (1.0 + 0.134145*x2)
		t := math.Tanh(u)
		fPrime := 0.5*(1.0+t) + 0.5*xi*(1.0-t*t)*uPrime
		dx[i] = dOut * T(fPrime)
	}
}

func vjpSwiGLUNoSIMD[T float32 | float64](x, dOutput, dx []T, numRows, hiddenDim int) {
	for m := range numRows {
		xOffset := m * 2 * hiddenDim
		dOutOffset := m * hiddenDim
		for j := range hiddenDim {
			gate := float64(x[xOffset+j])
			val := float64(x[xOffset+hiddenDim+j])
			g := float64(dOutput[dOutOffset+j])

			s := 1.0 / (1.0 + math.Exp(-gate))
			swish := gate * s
			swishPrime := s * (1.0 + gate*(1.0-s))

			dx[xOffset+j] = T(g * val * swishPrime)
			dx[xOffset+hiddenDim+j] = T(g * swish)
		}
	}
}

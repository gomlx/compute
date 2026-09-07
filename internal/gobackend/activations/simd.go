// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package activations

import (
	"simd"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/support/simdmath"
)

const PrioritySIMD = gobackend.PriorityTyped + 5
const halfChunk = 64

func init() {
	registerSIMD()
}

func registerSIMD() {
	// Float32 Forward
	RegisterKernel[float32]("simd:relu", compute.ActivationRelu, ReluFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:sigmoid", compute.ActivationSigmoid, SigmoidFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:hardsigmoid", compute.ActivationHardSigmoid, HardSigmoidFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:leakyrelu", compute.ActivationLeakyRelu, LeakyReluFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:selu", compute.ActivationSelu, SeluFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:silu", compute.ActivationSilu, SiluFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:hardswish", compute.ActivationHardSwish, HardSwishFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:tanh", compute.ActivationTanh, TanhFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:gelu", compute.ActivationGelu, geluExactFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:geluapprox", compute.ActivationGeluApproximate, GeluApproxFloat32SIMD, PrioritySIMD)
	RegisterSwiGLU[float32]("simd:swiglu", swigluFloat32SIMD, PrioritySIMD)

	// Float64 Forward
	RegisterKernel[float64]("simd:relu", compute.ActivationRelu, ReluFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:sigmoid", compute.ActivationSigmoid, SigmoidFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:hardsigmoid", compute.ActivationHardSigmoid, HardSigmoidFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:leakyrelu", compute.ActivationLeakyRelu, LeakyReluFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:selu", compute.ActivationSelu, SeluFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:silu", compute.ActivationSilu, SiluFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:hardswish", compute.ActivationHardSwish, HardSwishFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:tanh", compute.ActivationTanh, TanhFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:gelu", compute.ActivationGelu, GeluExactFloat64SIMD, PrioritySIMD)
	RegisterKernel[float64]("simd:geluapprox", compute.ActivationGeluApproximate, GeluApproxFloat64SIMD, PrioritySIMD)
	RegisterSwiGLU[float64]("simd:swiglu", SwiGLUFloat64SIMD, PrioritySIMD)

	// BFloat16 Forward (upscale to float32 SIMD, then convert)
	RegisterKernel[bfloat16.BFloat16]("simd:relu", compute.ActivationRelu, MakeBF16KernelFromF32(ReluFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:sigmoid", compute.ActivationSigmoid, MakeBF16KernelFromF32(SigmoidFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:hardsigmoid", compute.ActivationHardSigmoid, MakeBF16KernelFromF32(HardSigmoidFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:leakyrelu", compute.ActivationLeakyRelu, MakeBF16KernelFromF32(LeakyReluFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:selu", compute.ActivationSelu, MakeBF16KernelFromF32(SeluFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:silu", compute.ActivationSilu, MakeBF16KernelFromF32(SiluFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:hardswish", compute.ActivationHardSwish, MakeBF16KernelFromF32(HardSwishFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:tanh", compute.ActivationTanh, MakeBF16KernelFromF32(TanhFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:gelu", compute.ActivationGelu, MakeBF16KernelFromF32(geluExactFloat32SIMD), PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:geluapprox", compute.ActivationGeluApproximate, MakeBF16KernelFromF32(GeluApproxFloat32SIMD), PrioritySIMD)
	RegisterSwiGLU[bfloat16.BFloat16]("simd:swiglu", swigluBF16SIMD, PrioritySIMD)

	// Float16 Forward (upscale to float32 SIMD, then convert)
	RegisterKernel[float16.Float16]("simd:relu", compute.ActivationRelu, MakeF16KernelFromF32(ReluFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:sigmoid", compute.ActivationSigmoid, MakeF16KernelFromF32(SigmoidFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:hardsigmoid", compute.ActivationHardSigmoid, MakeF16KernelFromF32(HardSigmoidFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:leakyrelu", compute.ActivationLeakyRelu, MakeF16KernelFromF32(LeakyReluFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:selu", compute.ActivationSelu, MakeF16KernelFromF32(SeluFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:silu", compute.ActivationSilu, MakeF16KernelFromF32(SiluFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:hardswish", compute.ActivationHardSwish, MakeF16KernelFromF32(HardSwishFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:tanh", compute.ActivationTanh, MakeF16KernelFromF32(TanhFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:gelu", compute.ActivationGelu, MakeF16KernelFromF32(geluExactFloat32SIMD), PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:geluapprox", compute.ActivationGeluApproximate, MakeF16KernelFromF32(GeluApproxFloat32SIMD), PrioritySIMD)
	RegisterSwiGLU[float16.Float16]("simd:swiglu", swigluF16SIMD, PrioritySIMD)

	// Float32 VJP
	RegisterVJPKernel[float32]("simd:vjp_relu", compute.ActivationRelu, vjpReluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_sigmoid", compute.ActivationSigmoid, vjpSigmoidFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_hardsigmoid", compute.ActivationHardSigmoid, vjpHardSigmoidFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_leakyrelu", compute.ActivationLeakyRelu, vjpLeakyReluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_selu", compute.ActivationSelu, vjpSeluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_silu", compute.ActivationSilu, vjpSiluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_hardswish", compute.ActivationHardSwish, vjpHardSwishFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_tanh", compute.ActivationTanh, vjpTanhFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_gelu", compute.ActivationGelu, vjpGeluExactFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_geluapprox", compute.ActivationGeluApproximate, vjpGeluApproxFloat32SIMD, PrioritySIMD)
	RegisterSwiGLUVJP[float32]("simd:vjp_swiglu", vjpSwiGLUFloat32SIMD, PrioritySIMD)

	// Float64 VJP
	RegisterVJPKernel[float64]("simd:vjp_relu", compute.ActivationRelu, vjpReluFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_sigmoid", compute.ActivationSigmoid, vjpSigmoidFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_hardsigmoid", compute.ActivationHardSigmoid, vjpHardSigmoidFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_leakyrelu", compute.ActivationLeakyRelu, vjpLeakyReluFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_selu", compute.ActivationSelu, vjpSeluFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_silu", compute.ActivationSilu, vjpSiluFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_hardswish", compute.ActivationHardSwish, vjpHardSwishFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_tanh", compute.ActivationTanh, vjpTanhFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_gelu", compute.ActivationGelu, vjpGeluExactFloat64SIMD, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_geluapprox", compute.ActivationGeluApproximate, vjpGeluApproxFloat64SIMD, PrioritySIMD)
	RegisterSwiGLUVJP[float64]("simd:vjp_swiglu", vjpSwiGLUFloat64SIMD, PrioritySIMD)

	// BFloat16 VJP
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_relu", compute.ActivationRelu, MakeBF16VJPKernelFromF32(vjpReluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_sigmoid", compute.ActivationSigmoid, MakeBF16VJPKernelFromF32(vjpSigmoidFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_hardsigmoid", compute.ActivationHardSigmoid, MakeBF16VJPKernelFromF32(vjpHardSigmoidFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_leakyrelu", compute.ActivationLeakyRelu, MakeBF16VJPKernelFromF32(vjpLeakyReluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_selu", compute.ActivationSelu, MakeBF16VJPKernelFromF32(vjpSeluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_silu", compute.ActivationSilu, MakeBF16VJPKernelFromF32(vjpSiluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_hardswish", compute.ActivationHardSwish, MakeBF16VJPKernelFromF32(vjpHardSwishFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_tanh", compute.ActivationTanh, MakeBF16VJPKernelFromF32(vjpTanhFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_gelu", compute.ActivationGelu, MakeBF16VJPKernelFromF32(vjpGeluExactFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_geluapprox", compute.ActivationGeluApproximate, MakeBF16VJPKernelFromF32(vjpGeluApproxFloat32SIMD), PrioritySIMD)
	RegisterSwiGLUVJP[bfloat16.BFloat16]("simd:vjp_swiglu", MakeBF16SwiGLUVJPFromF32(vjpSwiGLUFloat32SIMD), PrioritySIMD)

	// Float16 VJP
	RegisterVJPKernel[float16.Float16]("simd:vjp_relu", compute.ActivationRelu, MakeF16VJPKernelFromF32(vjpReluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_sigmoid", compute.ActivationSigmoid, MakeF16VJPKernelFromF32(vjpSigmoidFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_hardsigmoid", compute.ActivationHardSigmoid, MakeF16VJPKernelFromF32(vjpHardSigmoidFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_leakyrelu", compute.ActivationLeakyRelu, MakeF16VJPKernelFromF32(vjpLeakyReluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_selu", compute.ActivationSelu, MakeF16VJPKernelFromF32(vjpSeluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_silu", compute.ActivationSilu, MakeF16VJPKernelFromF32(vjpSiluFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_hardswish", compute.ActivationHardSwish, MakeF16VJPKernelFromF32(vjpHardSwishFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_tanh", compute.ActivationTanh, MakeF16VJPKernelFromF32(vjpTanhFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_gelu", compute.ActivationGelu, MakeF16VJPKernelFromF32(vjpGeluExactFloat32SIMD), PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_geluapprox", compute.ActivationGeluApproximate, MakeF16VJPKernelFromF32(vjpGeluApproxFloat32SIMD), PrioritySIMD)
	RegisterSwiGLUVJP[float16.Float16]("simd:vjp_swiglu", MakeF16SwiGLUVJPFromF32(vjpSwiGLUFloat32SIMD), PrioritySIMD)
}

// -------------------------------------------------------------------------------------------------
// Float32 SIMD implementations (100% SIMD, branchless/masked remainder)
// -------------------------------------------------------------------------------------------------

func ReluFloat32SIMD(in, out []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		v.Max(vZero).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		v.Max(vZero).StorePart(out[i:])
	}
}

func HardSigmoidFloat32SIMD(in, out []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vOne := simd.BroadcastFloat32s(1)
	vPointTwo := simd.BroadcastFloat32s(0.2)
	vHalf := simd.BroadcastFloat32s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		v.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		v.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne).StorePart(out[i:])
	}
}

func LeakyReluFloat32SIMD(in, out []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vAlpha := simd.BroadcastFloat32s(0.3)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		scaled := v.Mul(vAlpha)
		v.IfElse(v.GreaterEqual(vZero), scaled).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		scaled := v.Mul(vAlpha)
		v.IfElse(v.GreaterEqual(vZero), scaled).StorePart(out[i:])
	}
}

func HardSwishFloat32SIMD(in, out []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vOne := simd.BroadcastFloat32s(1)
	vOneSixth := simd.BroadcastFloat32s(1.0 / 6.0)
	vHalf := simd.BroadcastFloat32s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		scaled := v.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		v.Mul(scaled).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		scaled := v.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		v.Mul(scaled).StorePart(out[i:])
	}
}

func SigmoidFloat32SIMD(in, out []float32) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		simdmath.SigmoidFloat32(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		simdmath.SigmoidFloat32(v).StorePart(out[i:])
	}
}

func SiluFloat32SIMD(in, out []float32) {
	vOne := simd.BroadcastFloat32s(1.0)
	vLen := vOne.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		expNeg := simdmath.ExpFloat32(v.Neg())
		v.Div(vOne.Add(expNeg)).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		expNeg := simdmath.ExpFloat32(v.Neg())
		v.Div(vOne.Add(expNeg)).StorePart(out[i:])
	}
}

func SeluFloat32SIMD(in, out []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vScale := simd.BroadcastFloat32s(seluScale)
	vScaleAlpha := simd.BroadcastFloat32s(seluScaleAlpha)
	vOne := simd.BroadcastFloat32s(1.0)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		pos := v.Mul(vScale)
		neg := vScaleAlpha.Mul(simdmath.ExpFloat32(v).Sub(vOne))
		pos.IfElse(v.GreaterEqual(vZero), neg).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		pos := v.Mul(vScale)
		neg := vScaleAlpha.Mul(simdmath.ExpFloat32(v).Sub(vOne))
		pos.IfElse(v.GreaterEqual(vZero), neg).StorePart(out[i:])
	}
}

func TanhFloat32SIMD(in, out []float32) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		simdmath.TanhFloat32(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		simdmath.TanhFloat32(v).StorePart(out[i:])
	}
}

func geluExactFloat32SIMD(in, out []float32) {
	vHalf := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vRsqrt2 := simd.BroadcastFloat32s(0.7071067811865475)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		erfVal := simdmath.ErfFloat32(v.Mul(vRsqrt2))
		vHalf.Mul(v).Mul(vOne.Add(erfVal)).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		erfVal := simdmath.ErfFloat32(v.Mul(vRsqrt2))
		vHalf.Mul(v).Mul(vOne.Add(erfVal)).StorePart(out[i:])
	}
}

func GeluApproxFloat32SIMD(in, out []float32) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		simdmath.GeluApproxFloat32(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		simdmath.GeluApproxFloat32(v).StorePart(out[i:])
	}
}

func swigluFloat32SIMD(in, out []float32, numRows, hiddenDim int) {
	vOne := simd.BroadcastFloat32s(1.0)
	vLen := vOne.Len()
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		j := 0
		for ; j+vLen <= hiddenDim; j += vLen {
			vGate := simd.LoadFloat32s(in[inOffset+j:])
			vVal := simd.LoadFloat32s(in[inOffset+hiddenDim+j:])
			expNeg := simdmath.ExpFloat32(vGate.Neg())
			siluGate := vGate.Div(vOne.Add(expNeg))
			siluGate.Mul(vVal).Store(out[outOffset+j:])
		}
		if j < hiddenDim {
			vGate, _ := simd.LoadFloat32sPart(in[inOffset+j:])
			vVal, _ := simd.LoadFloat32sPart(in[inOffset+hiddenDim+j:])
			expNeg := simdmath.ExpFloat32(vGate.Neg())
			siluGate := vGate.Div(vOne.Add(expNeg))
			siluGate.Mul(vVal).StorePart(out[outOffset+j:])
		}
	}
}

// -------------------------------------------------------------------------------------------------
// Float64 SIMD implementations (100% SIMD, branchless/masked remainder)
// -------------------------------------------------------------------------------------------------

func ReluFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		v.Max(vZero).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		v.Max(vZero).StorePart(out[i:])
	}
}

func HardSigmoidFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vOne := simd.BroadcastFloat64s(1)
	vPointTwo := simd.BroadcastFloat64s(0.2)
	vHalf := simd.BroadcastFloat64s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		v.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		v.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne).StorePart(out[i:])
	}
}

func LeakyReluFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vAlpha := simd.BroadcastFloat64s(0.3)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		scaled := v.Mul(vAlpha)
		v.IfElse(v.GreaterEqual(vZero), scaled).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		scaled := v.Mul(vAlpha)
		v.IfElse(v.GreaterEqual(vZero), scaled).StorePart(out[i:])
	}
}

func HardSwishFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vOne := simd.BroadcastFloat64s(1)
	vOneSixth := simd.BroadcastFloat64s(1.0 / 6.0)
	vHalf := simd.BroadcastFloat64s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		scaled := v.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		v.Mul(scaled).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		scaled := v.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		v.Mul(scaled).StorePart(out[i:])
	}
}

func SigmoidFloat64SIMD(in, out []float64) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		simdmath.SigmoidFloat64(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		simdmath.SigmoidFloat64(v).StorePart(out[i:])
	}
}

func SiluFloat64SIMD(in, out []float64) {
	vOne := simd.BroadcastFloat64s(1.0)
	vLen := vOne.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		expNeg := simdmath.ExpFloat64(v.Neg())
		v.Div(vOne.Add(expNeg)).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		expNeg := simdmath.ExpFloat64(v.Neg())
		v.Div(vOne.Add(expNeg)).StorePart(out[i:])
	}
}

func SeluFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vScale := simd.BroadcastFloat64s(seluScale)
	vScaleAlpha := simd.BroadcastFloat64s(seluScaleAlpha)
	vOne := simd.BroadcastFloat64s(1.0)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		pos := v.Mul(vScale)
		neg := vScaleAlpha.Mul(simdmath.ExpFloat64(v).Sub(vOne))
		pos.IfElse(v.GreaterEqual(vZero), neg).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		pos := v.Mul(vScale)
		neg := vScaleAlpha.Mul(simdmath.ExpFloat64(v).Sub(vOne))
		pos.IfElse(v.GreaterEqual(vZero), neg).StorePart(out[i:])
	}
}

func TanhFloat64SIMD(in, out []float64) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		simdmath.TanhFloat64(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		simdmath.TanhFloat64(v).StorePart(out[i:])
	}
}

func GeluExactFloat64SIMD(in, out []float64) {
	vHalf := simd.BroadcastFloat64s(0.5)
	vOne := simd.BroadcastFloat64s(1.0)
	vRsqrt2 := simd.BroadcastFloat64s(0.7071067811865475244)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		erfVal := simdmath.ErfFloat64(v.Mul(vRsqrt2))
		vHalf.Mul(v).Mul(vOne.Add(erfVal)).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		erfVal := simdmath.ErfFloat64(v.Mul(vRsqrt2))
		vHalf.Mul(v).Mul(vOne.Add(erfVal)).StorePart(out[i:])
	}
}

func GeluApproxFloat64SIMD(in, out []float64) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		simdmath.GeluApproxFloat64(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		simdmath.GeluApproxFloat64(v).StorePart(out[i:])
	}
}

func SwiGLUFloat64SIMD(in, out []float64, numRows, hiddenDim int) {
	vOne := simd.BroadcastFloat64s(1.0)
	vLen := vOne.Len()
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		j := 0
		for ; j+vLen <= hiddenDim; j += vLen {
			vGate := simd.LoadFloat64s(in[inOffset+j:])
			vVal := simd.LoadFloat64s(in[inOffset+hiddenDim+j:])
			expNeg := simdmath.ExpFloat64(vGate.Neg())
			siluGate := vGate.Div(vOne.Add(expNeg))
			siluGate.Mul(vVal).Store(out[outOffset+j:])
		}
		if j < hiddenDim {
			vGate, _ := simd.LoadFloat64sPart(in[inOffset+j:])
			vVal, _ := simd.LoadFloat64sPart(in[inOffset+hiddenDim+j:])
			expNeg := simdmath.ExpFloat64(vGate.Neg())
			siluGate := vGate.Div(vOne.Add(expNeg))
			siluGate.Mul(vVal).StorePart(out[outOffset+j:])
		}
	}
}

// -------------------------------------------------------------------------------------------------
// Adapters for BFloat16 and Float16
// -------------------------------------------------------------------------------------------------

func MakeBF16KernelFromF32(f32Kernel func(in, out []float32)) func(in, out []bfloat16.BFloat16) {
	return func(in, out []bfloat16.BFloat16) {
		var inBuf [halfChunk]float32
		var outBuf [halfChunk]float32
		for i := 0; i < len(in); i += halfChunk {
			chunkLen := min(halfChunk, len(in)-i)
			for j := 0; j < chunkLen; j++ {
				inBuf[j] = in[i+j].Float32()
			}
			f32Kernel(inBuf[:chunkLen], outBuf[:chunkLen])
			for j := 0; j < chunkLen; j++ {
				out[i+j] = bfloat16.FromFloat32(outBuf[j])
			}
		}
	}
}

func MakeF16KernelFromF32(f32Kernel func(in, out []float32)) func(in, out []float16.Float16) {
	return func(in, out []float16.Float16) {
		var inBuf [halfChunk]float32
		var outBuf [halfChunk]float32
		for i := 0; i < len(in); i += halfChunk {
			chunkLen := min(halfChunk, len(in)-i)
			for j := 0; j < chunkLen; j++ {
				inBuf[j] = in[i+j].Float32()
			}
			f32Kernel(inBuf[:chunkLen], outBuf[:chunkLen])
			for j := 0; j < chunkLen; j++ {
				out[i+j] = float16.FromFloat32(outBuf[j])
			}
		}
	}
}

func swigluBF16SIMD(in, out []bfloat16.BFloat16, numRows, hiddenDim int) {
	const bufCap = 64
	var inBuf [bufCap * 2]float32
	var outBuf [bufCap]float32
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		for j := 0; j < hiddenDim; j += bufCap {
			chunkLen := min(bufCap, hiddenDim-j)
			for k := 0; k < chunkLen; k++ {
				inBuf[k] = in[inOffset+j+k].Float32()
				inBuf[chunkLen+k] = in[inOffset+hiddenDim+j+k].Float32()
			}
			swigluFloat32SIMD(inBuf[:chunkLen*2], outBuf[:chunkLen], 1, chunkLen)
			for k := 0; k < chunkLen; k++ {
				out[outOffset+j+k] = bfloat16.FromFloat32(outBuf[k])
			}
		}
	}
}

func swigluF16SIMD(in, out []float16.Float16, numRows, hiddenDim int) {
	const bufCap = 64
	var inBuf [bufCap * 2]float32
	var outBuf [bufCap]float32
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		for j := 0; j < hiddenDim; j += bufCap {
			chunkLen := min(bufCap, hiddenDim-j)
			for k := 0; k < chunkLen; k++ {
				inBuf[k] = in[inOffset+j+k].Float32()
				inBuf[chunkLen+k] = in[inOffset+hiddenDim+j+k].Float32()
			}
			swigluFloat32SIMD(inBuf[:chunkLen*2], outBuf[:chunkLen], 1, chunkLen)
			for k := 0; k < chunkLen; k++ {
				out[outOffset+j+k] = float16.FromFloat32(outBuf[k])
			}
		}
	}
}

// -------------------------------------------------------------------------------------------------
// Float32 SIMD VJP implementations (100% SIMD)
// -------------------------------------------------------------------------------------------------

func vjpReluFloat32SIMD(y, x, dOutput, dx []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vLen := vZero.Len()
	ref := y
	if len(ref) == 0 {
		ref = x
	}
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vRef := simd.LoadFloat32s(ref[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:])
		mask := vRef.Greater(vZero)
		vDOut.IfElse(mask, vZero).Store(dx[i:])
	}
	if i < len(dOutput) {
		vRef, _ := simd.LoadFloat32sPart(ref[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
		mask := vRef.Greater(vZero)
		vDOut.IfElse(mask, vZero).StorePart(dx[i:])
	}
}

func vjpSigmoidFloat32SIMD(y, x, dOutput, dx []float32) {
	vOne := simd.BroadcastFloat32s(1)
	vLen := vOne.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			vDOut.Mul(vY).Mul(vOne.Sub(vY)).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			vDOut.Mul(vY).Mul(vOne.Sub(vY)).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			s := simdmath.SigmoidFloat32(vX)
			vDOut.Mul(s).Mul(vOne.Sub(s)).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			s := simdmath.SigmoidFloat32(vX)
			vDOut.Mul(s).Mul(vOne.Sub(s)).StorePart(dx[i:])
		}
	}
}

func vjpHardSigmoidFloat32SIMD(y, x, dOutput, dx []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vSlope := simd.BroadcastFloat32s(0.2)
	vLen := vZero.Len()
	i := 0
	if len(y) > 0 {
		vOne := simd.BroadcastFloat32s(1)
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			mask := vY.Greater(vZero).And(vY.Less(vOne))
			vDOut.Mul(vSlope).IfElse(mask, vZero).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			mask := vY.Greater(vZero).And(vY.Less(vOne))
			vDOut.Mul(vSlope).IfElse(mask, vZero).StorePart(dx[i:])
		}
	} else {
		vLo := simd.BroadcastFloat32s(-2.5)
		vHi := simd.BroadcastFloat32s(2.5)
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			mask := vX.Greater(vLo).And(vX.Less(vHi))
			vDOut.Mul(vSlope).IfElse(mask, vZero).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			mask := vX.Greater(vLo).And(vX.Less(vHi))
			vDOut.Mul(vSlope).IfElse(mask, vZero).StorePart(dx[i:])
		}
	}
}

func vjpLeakyReluFloat32SIMD(y, x, dOutput, dx []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vAlpha := simd.BroadcastFloat32s(0.3)
	vLen := vZero.Len()
	ref := y
	if len(ref) == 0 {
		ref = x
	}
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vRef := simd.LoadFloat32s(ref[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:])
		mask := vRef.GreaterEqual(vZero)
		vDOut.IfElse(mask, vDOut.Mul(vAlpha)).Store(dx[i:])
	}
	if i < len(dOutput) {
		vRef, _ := simd.LoadFloat32sPart(ref[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
		mask := vRef.GreaterEqual(vZero)
		vDOut.IfElse(mask, vDOut.Mul(vAlpha)).StorePart(dx[i:])
	}
}

func vjpSeluFloat32SIMD(y, x, dOutput, dx []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vScale := simd.BroadcastFloat32s(seluScale)
	vScaleAlpha := simd.BroadcastFloat32s(seluScaleAlpha)
	vLen := vZero.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vY.Add(vScaleAlpha))
			pos.IfElse(vY.Greater(vZero), neg).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vY.Add(vScaleAlpha))
			pos.IfElse(vY.Greater(vZero), neg).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(vX))
			pos.IfElse(vX.Greater(vZero), neg).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(vX))
			pos.IfElse(vX.Greater(vZero), neg).StorePart(dx[i:])
		}
	}
}

func vjpTanhFloat32SIMD(y, x, dOutput, dx []float32) {
	vOne := simd.BroadcastFloat32s(1)
	vLen := vOne.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			vDOut.Mul(vOne.Sub(vY.Mul(vY))).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			vDOut.Mul(vOne.Sub(vY.Mul(vY))).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:])
			t := simdmath.TanhFloat32(vX)
			vDOut.Mul(vOne.Sub(t.Mul(t))).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
			t := simdmath.TanhFloat32(vX)
			vDOut.Mul(vOne.Sub(t.Mul(t))).StorePart(dx[i:])
		}
	}
}

func vjpSiluFloat32SIMD(y, x, dOutput, dx []float32) {
	vOne := simd.BroadcastFloat32s(1)
	vLen := vOne.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:])
		s := simdmath.SigmoidFloat32(vX)
		fPrime := s.Mul(vOne.Add(vX.Mul(vOne.Sub(s))))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
		s := simdmath.SigmoidFloat32(vX)
		fPrime := s.Mul(vOne.Add(vX.Mul(vOne.Sub(s))))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpHardSwishFloat32SIMD(y, x, dOutput, dx []float32) {
	vZero := simd.BroadcastFloat32s(0)
	vThree := simd.BroadcastFloat32s(3)
	vNegThree := simd.BroadcastFloat32s(-3)
	vOneThird := simd.BroadcastFloat32s(1.0 / 3.0)
	vHalf := simd.BroadcastFloat32s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:])
		mid := vDOut.Mul(vX.MulAdd(vOneThird, vHalf))
		res := mid.IfElse(vX.Greater(vNegThree), vZero)
		res = vDOut.IfElse(vX.GreaterEqual(vThree), res)
		res.Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
		mid := vDOut.Mul(vX.MulAdd(vOneThird, vHalf))
		res := mid.IfElse(vX.Greater(vNegThree), vZero)
		res = vDOut.IfElse(vX.GreaterEqual(vThree), res)
		res.StorePart(dx[i:])
	}
}

func vjpGeluExactFloat32SIMD(y, x, dOutput, dx []float32) {
	vHalf := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vInvSqrt2 := simd.BroadcastFloat32s(0.7071067811865475)
	vInvSqrt2Pi := simd.BroadcastFloat32s(0.39894228)
	vNegHalf := simd.BroadcastFloat32s(-0.5)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:])
		cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(vX.Mul(vInvSqrt2))))
		pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(vX.Mul(vX).Mul(vNegHalf)))
		fPrime := cdf.Add(vX.Mul(pdf))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
		cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(vX.Mul(vInvSqrt2))))
		pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(vX.Mul(vX).Mul(vNegHalf)))
		fPrime := cdf.Add(vX.Mul(pdf))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpGeluApproxFloat32SIMD(y, x, dOutput, dx []float32) {
	vHalf := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vSqrt2ByPi := simd.BroadcastFloat32s(0.7978845608)
	vConst044715 := simd.BroadcastFloat32s(0.044715)
	vConst0134145 := simd.BroadcastFloat32s(0.134145)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:])
		vX2 := vX.Mul(vX)
		vX3 := vX2.Mul(vX)
		u := vSqrt2ByPi.Mul(vX.MulAdd(vConst044715, vX3))
		uPrime := vSqrt2ByPi.Mul(vX2.MulAdd(vConst0134145, vOne))
		t := simdmath.TanhFloat32(u)
		fPrime := vHalf.Mul(vOne.Add(t)).Add(vHalf.Mul(vX).Mul(vOne.Sub(t.Mul(t))).Mul(uPrime))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:])
		vX2 := vX.Mul(vX)
		vX3 := vX2.Mul(vX)
		u := vSqrt2ByPi.Mul(vX.MulAdd(vConst044715, vX3))
		uPrime := vSqrt2ByPi.Mul(vX2.MulAdd(vConst0134145, vOne))
		t := simdmath.TanhFloat32(u)
		fPrime := vHalf.Mul(vOne.Add(t)).Add(vHalf.Mul(vX).Mul(vOne.Sub(t.Mul(t))).Mul(uPrime))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpSwiGLUFloat32SIMD(x, dOutput, dx []float32, numRows, hiddenDim int) {
	vOne := simd.BroadcastFloat32s(1)
	vLen := vOne.Len()
	for m := range numRows {
		xOffset := m * 2 * hiddenDim
		dOutOffset := m * hiddenDim
		j := 0
		for ; j+vLen <= hiddenDim; j += vLen {
			vGate := simd.LoadFloat32s(x[xOffset+j:])
			vVal := simd.LoadFloat32s(x[xOffset+hiddenDim+j:])
			vDOut := simd.LoadFloat32s(dOutput[dOutOffset+j:])

			s := simdmath.SigmoidFloat32(vGate)
			swish := vGate.Mul(s)
			swishPrime := s.Mul(vOne.Add(vGate.Mul(vOne.Sub(s))))

			vDOut.Mul(vVal).Mul(swishPrime).Store(dx[xOffset+j:])
			vDOut.Mul(swish).Store(dx[xOffset+hiddenDim+j:])
		}
		if j < hiddenDim {
			vGate, _ := simd.LoadFloat32sPart(x[xOffset+j:])
			vVal, _ := simd.LoadFloat32sPart(x[xOffset+hiddenDim+j:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[dOutOffset+j:])

			s := simdmath.SigmoidFloat32(vGate)
			swish := vGate.Mul(s)
			swishPrime := s.Mul(vOne.Add(vGate.Mul(vOne.Sub(s))))

			vDOut.Mul(vVal).Mul(swishPrime).StorePart(dx[xOffset+j:])
			vDOut.Mul(swish).StorePart(dx[xOffset+hiddenDim+j:])
		}
	}
}

// -------------------------------------------------------------------------------------------------
// Float64 SIMD VJP implementations (100% SIMD)
// -------------------------------------------------------------------------------------------------

func vjpReluFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vLen := vZero.Len()
	ref := y
	if len(ref) == 0 {
		ref = x
	}
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vRef := simd.LoadFloat64s(ref[i:])
		vDOut := simd.LoadFloat64s(dOutput[i:])
		mask := vRef.Greater(vZero)
		vDOut.IfElse(mask, vZero).Store(dx[i:])
	}
	if i < len(dOutput) {
		vRef, _ := simd.LoadFloat64sPart(ref[i:])
		vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		mask := vRef.Greater(vZero)
		vDOut.IfElse(mask, vZero).StorePart(dx[i:])
	}
}

func vjpSigmoidFloat64SIMD(y, x, dOutput, dx []float64) {
	vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			vDOut.Mul(vY).Mul(vOne.Sub(vY)).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			vDOut.Mul(vY).Mul(vOne.Sub(vY)).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			s := simdmath.SigmoidFloat64(vX)
			vDOut.Mul(s).Mul(vOne.Sub(s)).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			s := simdmath.SigmoidFloat64(vX)
			vDOut.Mul(s).Mul(vOne.Sub(s)).StorePart(dx[i:])
		}
	}
}

func vjpHardSigmoidFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vSlope := simd.BroadcastFloat64s(0.2)
	vLen := vZero.Len()
	i := 0
	if len(y) > 0 {
		vOne := simd.BroadcastFloat64s(1)
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			mask := vY.Greater(vZero).And(vY.Less(vOne))
			vDOut.Mul(vSlope).IfElse(mask, vZero).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			mask := vY.Greater(vZero).And(vY.Less(vOne))
			vDOut.Mul(vSlope).IfElse(mask, vZero).StorePart(dx[i:])
		}
	} else {
		vLo := simd.BroadcastFloat64s(-2.5)
		vHi := simd.BroadcastFloat64s(2.5)
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			mask := vX.Greater(vLo).And(vX.Less(vHi))
			vDOut.Mul(vSlope).IfElse(mask, vZero).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			mask := vX.Greater(vLo).And(vX.Less(vHi))
			vDOut.Mul(vSlope).IfElse(mask, vZero).StorePart(dx[i:])
		}
	}
}

func vjpLeakyReluFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vAlpha := simd.BroadcastFloat64s(0.3)
	vLen := vZero.Len()
	ref := y
	if len(ref) == 0 {
		ref = x
	}
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vRef := simd.LoadFloat64s(ref[i:])
		vDOut := simd.LoadFloat64s(dOutput[i:])
		mask := vRef.GreaterEqual(vZero)
		vDOut.IfElse(mask, vDOut.Mul(vAlpha)).Store(dx[i:])
	}
	if i < len(dOutput) {
		vRef, _ := simd.LoadFloat64sPart(ref[i:])
		vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		mask := vRef.GreaterEqual(vZero)
		vDOut.IfElse(mask, vDOut.Mul(vAlpha)).StorePart(dx[i:])
	}
}

func vjpSeluFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vScale := simd.BroadcastFloat64s(seluScale)
	vScaleAlpha := simd.BroadcastFloat64s(seluScaleAlpha)
	vLen := vZero.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vY.Add(vScaleAlpha))
			pos.IfElse(vY.Greater(vZero), neg).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vY.Add(vScaleAlpha))
			pos.IfElse(vY.Greater(vZero), neg).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat64(vX))
			pos.IfElse(vX.Greater(vZero), neg).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat64(vX))
			pos.IfElse(vX.Greater(vZero), neg).StorePart(dx[i:])
		}
	}
}

func vjpSiluFloat64SIMD(y, x, dOutput, dx []float64) {
	vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat64s(dOutput[i:])
		s := simdmath.SigmoidFloat64(vX)
		fPrime := s.Mul(vOne.Add(vX.Mul(vOne.Sub(s))))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		s := simdmath.SigmoidFloat64(vX)
		fPrime := s.Mul(vOne.Add(vX.Mul(vOne.Sub(s))))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpHardSwishFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat64s(0)
	vThree := simd.BroadcastFloat64s(3)
	vNegThree := simd.BroadcastFloat64s(-3)
	vOneThird := simd.BroadcastFloat64s(1.0 / 3.0)
	vHalf := simd.BroadcastFloat64s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat64s(dOutput[i:])
		mid := vDOut.Mul(vX.MulAdd(vOneThird, vHalf))
		res := mid.IfElse(vX.Greater(vNegThree), vZero)
		res = vDOut.IfElse(vX.GreaterEqual(vThree), res)
		res.Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		mid := vDOut.Mul(vX.MulAdd(vOneThird, vHalf))
		res := mid.IfElse(vX.Greater(vNegThree), vZero)
		res = vDOut.IfElse(vX.GreaterEqual(vThree), res)
		res.StorePart(dx[i:])
	}
}

func vjpTanhFloat64SIMD(y, x, dOutput, dx []float64) {
	vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			vDOut.Mul(vOne.Sub(vY.Mul(vY))).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			vDOut.Mul(vOne.Sub(vY.Mul(vY))).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat64s(dOutput[i:])
			t := simdmath.TanhFloat64(vX)
			vDOut.Mul(vOne.Sub(t.Mul(t))).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			t := simdmath.TanhFloat64(vX)
			vDOut.Mul(vOne.Sub(t.Mul(t))).StorePart(dx[i:])
		}
	}
}

func vjpGeluExactFloat64SIMD(y, x, dOutput, dx []float64) {
	vHalf := simd.BroadcastFloat64s(0.5)
	vOne := simd.BroadcastFloat64s(1.0)
	vInvSqrt2 := simd.BroadcastFloat64s(0.7071067811865475244)
	vInvSqrt2Pi := simd.BroadcastFloat64s(0.3989422804014326779)
	vNegHalf := simd.BroadcastFloat64s(-0.5)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat64s(dOutput[i:])
		cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat64(vX.Mul(vInvSqrt2))))
		pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat64(vX.Mul(vX).Mul(vNegHalf)))
		fPrime := cdf.Add(vX.Mul(pdf))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat64(vX.Mul(vInvSqrt2))))
		pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat64(vX.Mul(vX).Mul(vNegHalf)))
		fPrime := cdf.Add(vX.Mul(pdf))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpGeluApproxFloat64SIMD(y, x, dOutput, dx []float64) {
	vHalf := simd.BroadcastFloat64s(0.5)
	vOne := simd.BroadcastFloat64s(1.0)
	vSqrt2ByPi := simd.BroadcastFloat64s(0.7978845608028654)
	vConst044715 := simd.BroadcastFloat64s(0.044715)
	vConst0134145 := simd.BroadcastFloat64s(0.134145)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat64s(dOutput[i:])
		vX2 := vX.Mul(vX)
		vX3 := vX2.Mul(vX)
		u := vSqrt2ByPi.Mul(vX.MulAdd(vConst044715, vX3))
		uPrime := vSqrt2ByPi.Mul(vX2.MulAdd(vConst0134145, vOne))
		t := simdmath.TanhFloat64(u)
		fPrime := vHalf.Mul(vOne.Add(t)).Add(vHalf.Mul(vX).Mul(vOne.Sub(t.Mul(t))).Mul(uPrime))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		vX2 := vX.Mul(vX)
		vX3 := vX2.Mul(vX)
		u := vSqrt2ByPi.Mul(vX.MulAdd(vConst044715, vX3))
		uPrime := vSqrt2ByPi.Mul(vX2.MulAdd(vConst0134145, vOne))
		t := simdmath.TanhFloat64(u)
		fPrime := vHalf.Mul(vOne.Add(t)).Add(vHalf.Mul(vX).Mul(vOne.Sub(t.Mul(t))).Mul(uPrime))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpSwiGLUFloat64SIMD(x, dOutput, dx []float64, numRows, hiddenDim int) {
	vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	for m := range numRows {
		xOffset := m * 2 * hiddenDim
		dOutOffset := m * hiddenDim
		j := 0
		for ; j+vLen <= hiddenDim; j += vLen {
			vGate := simd.LoadFloat64s(x[xOffset+j:])
			vVal := simd.LoadFloat64s(x[xOffset+hiddenDim+j:])
			vDOut := simd.LoadFloat64s(dOutput[dOutOffset+j:])

			s := simdmath.SigmoidFloat64(vGate)
			swish := vGate.Mul(s)
			swishPrime := s.Mul(vOne.Add(vGate.Mul(vOne.Sub(s))))

			vDOut.Mul(vVal).Mul(swishPrime).Store(dx[xOffset+j:])
			vDOut.Mul(swish).Store(dx[xOffset+hiddenDim+j:])
		}
		if j < hiddenDim {
			vGate, _ := simd.LoadFloat64sPart(x[xOffset+j:])
			vVal, _ := simd.LoadFloat64sPart(x[xOffset+hiddenDim+j:])
			vDOut, _ := simd.LoadFloat64sPart(dOutput[dOutOffset+j:])

			s := simdmath.SigmoidFloat64(vGate)
			swish := vGate.Mul(s)
			swishPrime := s.Mul(vOne.Add(vGate.Mul(vOne.Sub(s))))

			vDOut.Mul(vVal).Mul(swishPrime).StorePart(dx[xOffset+j:])
			vDOut.Mul(swish).StorePart(dx[xOffset+hiddenDim+j:])
		}
	}
}

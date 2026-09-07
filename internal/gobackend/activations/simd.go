// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package activations

//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=simd_full_base.go -tags=float64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=simd_half_base.go -tags=float16

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"simd"
)

const PrioritySIMD = gobackend.PriorityTyped + 5

// selectFloat32s performs branchless bitwise vector selection without using mask registers,
// avoiding an AVX-512 SSA register allocation bug in Go 1.27 when shifts are present.
func selectFloat32s(mask simd.Mask32s, trueVal, falseVal simd.Float32s) simd.Float32s {
	m := mask.ToInt32s().ToBits()
	return trueVal.ToBits().And(m).Or(falseVal.ToBits().AndNot(m)).BitsToFloat32()
}

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
	RegisterKernel[float32]("simd:gelu", compute.ActivationGelu, GeluExactFloat32SIMD, PrioritySIMD)
	RegisterKernel[float32]("simd:geluapprox", compute.ActivationGeluApproximate, GeluApproxFloat32SIMD, PrioritySIMD)
	RegisterSwiGLU[float32]("simd:swiglu", SwiGLUFloat32SIMD, PrioritySIMD)

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

	// BFloat16 Forward
	RegisterKernel[bfloat16.BFloat16]("simd:relu", compute.ActivationRelu, ReluBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:sigmoid", compute.ActivationSigmoid, SigmoidBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:hardsigmoid", compute.ActivationHardSigmoid, HardSigmoidBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:leakyrelu", compute.ActivationLeakyRelu, LeakyReluBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:selu", compute.ActivationSelu, SeluBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:silu", compute.ActivationSilu, SiluBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:hardswish", compute.ActivationHardSwish, HardSwishBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:tanh", compute.ActivationTanh, TanhBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:gelu", compute.ActivationGelu, GeluExactBFloat16SIMD, PrioritySIMD)
	RegisterKernel[bfloat16.BFloat16]("simd:geluapprox", compute.ActivationGeluApproximate, GeluApproxBFloat16SIMD, PrioritySIMD)
	RegisterSwiGLU[bfloat16.BFloat16]("simd:swiglu", SwiGLUBFloat16SIMD, PrioritySIMD)

	// Float16 Forward
	RegisterKernel[float16.Float16]("simd:relu", compute.ActivationRelu, ReluFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:sigmoid", compute.ActivationSigmoid, SigmoidFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:hardsigmoid", compute.ActivationHardSigmoid, HardSigmoidFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:leakyrelu", compute.ActivationLeakyRelu, LeakyReluFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:selu", compute.ActivationSelu, SeluFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:silu", compute.ActivationSilu, SiluFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:hardswish", compute.ActivationHardSwish, HardSwishFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:tanh", compute.ActivationTanh, TanhFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:gelu", compute.ActivationGelu, GeluExactFloat16SIMD, PrioritySIMD)
	RegisterKernel[float16.Float16]("simd:geluapprox", compute.ActivationGeluApproximate, GeluApproxFloat16SIMD, PrioritySIMD)
	RegisterSwiGLU[float16.Float16]("simd:swiglu", SwiGLUFloat16SIMD, PrioritySIMD)

	// Float32 VJP
	RegisterVJPKernel[float32]("simd:vjp_relu", compute.ActivationRelu, vjpReluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_sigmoid", compute.ActivationSigmoid, vjpSigmoidFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_hardsigmoid", compute.ActivationHardSigmoid, vjpHardSigmoidFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_leakyrelu", compute.ActivationLeakyRelu, vjpLeakyReluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_selu", compute.ActivationSelu, vjpSeluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_silu", compute.ActivationSilu, vjpSiluFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_hardswish", compute.ActivationHardSwish, vjpHardSwishFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_tanh", compute.ActivationTanh, vjpTanhFloat32SIMD, PrioritySIMD)
	RegisterVJPKernel[float32]("simd:vjp_gelu", compute.ActivationGelu, GeluExactFloat32SIMDVJP, PrioritySIMD)
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
	RegisterVJPKernel[float64]("simd:vjp_gelu", compute.ActivationGelu, GeluExactFloat64SIMDVJP, PrioritySIMD)
	RegisterVJPKernel[float64]("simd:vjp_geluapprox", compute.ActivationGeluApproximate, vjpGeluApproxFloat64SIMD, PrioritySIMD)
	RegisterSwiGLUVJP[float64]("simd:vjp_swiglu", vjpSwiGLUFloat64SIMD, PrioritySIMD)

	// BFloat16 VJP
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_relu", compute.ActivationRelu, vjpReluBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_sigmoid", compute.ActivationSigmoid, vjpSigmoidBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_hardsigmoid", compute.ActivationHardSigmoid, vjpHardSigmoidBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_leakyrelu", compute.ActivationLeakyRelu, vjpLeakyReluBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_selu", compute.ActivationSelu, vjpSeluBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_silu", compute.ActivationSilu, vjpSiluBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_hardswish", compute.ActivationHardSwish, vjpHardSwishBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_tanh", compute.ActivationTanh, vjpTanhBFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_gelu", compute.ActivationGelu, GeluExactBFloat16SIMDVJP, PrioritySIMD)
	RegisterVJPKernel[bfloat16.BFloat16]("simd:vjp_geluapprox", compute.ActivationGeluApproximate, vjpGeluApproxBFloat16SIMD, PrioritySIMD)
	RegisterSwiGLUVJP[bfloat16.BFloat16]("simd:vjp_swiglu", vjpSwiGLUBFloat16SIMD, PrioritySIMD)

	// Float16 VJP
	RegisterVJPKernel[float16.Float16]("simd:vjp_relu", compute.ActivationRelu, vjpReluFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_sigmoid", compute.ActivationSigmoid, vjpSigmoidFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_hardsigmoid", compute.ActivationHardSigmoid, vjpHardSigmoidFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_leakyrelu", compute.ActivationLeakyRelu, vjpLeakyReluFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_selu", compute.ActivationSelu, vjpSeluFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_silu", compute.ActivationSilu, vjpSiluFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_hardswish", compute.ActivationHardSwish, vjpHardSwishFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_tanh", compute.ActivationTanh, vjpTanhFloat16SIMD, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_gelu", compute.ActivationGelu, GeluExactFloat16SIMDVJP, PrioritySIMD)
	RegisterVJPKernel[float16.Float16]("simd:vjp_geluapprox", compute.ActivationGeluApproximate, vjpGeluApproxFloat16SIMD, PrioritySIMD)
	RegisterSwiGLUVJP[float16.Float16]("simd:vjp_swiglu", vjpSwiGLUFloat16SIMD, PrioritySIMD)
}

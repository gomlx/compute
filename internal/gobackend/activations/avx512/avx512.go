// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
)

const PriorityAVX512 = gobackend.PriorityArch + 1

var (
	expMaxLog archsimd.Float32x16
	expMinLog archsimd.Float32x16
	expLog2E  archsimd.Float32x16
	expHalf   archsimd.Float32x16
	expLn2Hi  archsimd.Float32x16
	expLn2Lo  archsimd.Float32x16
	expP7     archsimd.Float32x16
	expP6     archsimd.Float32x16
	expP5     archsimd.Float32x16
	expP4     archsimd.Float32x16
	expP3     archsimd.Float32x16
	expP2     archsimd.Float32x16
	expOne    archsimd.Float32x16
	exp127    archsimd.Uint32x16

	erfP        archsimd.Float32x16
	erfA1       archsimd.Float32x16
	erfA2       archsimd.Float32x16
	erfA3       archsimd.Float32x16
	erfA4       archsimd.Float32x16
	erfA5       archsimd.Float32x16
	erfOne      archsimd.Float32x16
	erfHalf     archsimd.Float32x16
	erfRsqrt2   archsimd.Float32x16
	erfSignMask archsimd.Uint32x16

	geluHalf      archsimd.Float32x16
	geluOne       archsimd.Float32x16
	geluSqrt2ByPi archsimd.Float32x16
	geluC         archsimd.Float32x16
	tanhNine      archsimd.Float32x16
	tanhNegNine   archsimd.Float32x16
	tanhTwo       archsimd.Float32x16
	tanhOne       archsimd.Float32x16

	// Float64 constants
	expF64MaxLog archsimd.Float64x8
	expF64MinLog archsimd.Float64x8
	expF64Log2E  archsimd.Float64x8
	expF64Half   archsimd.Float64x8
	expF64Ln2Hi  archsimd.Float64x8
	expF64Ln2Lo  archsimd.Float64x8
	expF64One    archsimd.Float64x8
	expF641023   archsimd.Uint64x8

	expF64C11 archsimd.Float64x8
	expF64C10 archsimd.Float64x8
	expF64C9  archsimd.Float64x8
	expF64C8  archsimd.Float64x8
	expF64C7  archsimd.Float64x8
	expF64C6  archsimd.Float64x8
	expF64C5  archsimd.Float64x8
	expF64C4  archsimd.Float64x8
	expF64C3  archsimd.Float64x8
	expF64C2  archsimd.Float64x8

	erfF64P        archsimd.Float64x8
	erfF64A1       archsimd.Float64x8
	erfF64A2       archsimd.Float64x8
	erfF64A3       archsimd.Float64x8
	erfF64A4       archsimd.Float64x8
	erfF64A5       archsimd.Float64x8
	erfF64One      archsimd.Float64x8
	erfF64Half     archsimd.Float64x8
	erfF64Rsqrt2   archsimd.Float64x8
	erfF64SignMask archsimd.Uint64x8

	geluF64Half      archsimd.Float64x8
	geluF64One       archsimd.Float64x8
	geluF64Sqrt2ByPi archsimd.Float64x8
	geluF64C         archsimd.Float64x8
	tanhF6419        archsimd.Float64x8
	tanhF64Neg19     archsimd.Float64x8
	tanhF64Two       archsimd.Float64x8
	tanhF64One       archsimd.Float64x8

	// BFloat16 constants
	bf16_512Bias7FFF archsimd.Uint32x16
	bf16_512One      archsimd.Uint32x16
	bf16_512MaskHi   archsimd.Uint32x16
)

func init() {
	if gobackend.IsAVX512Allowed {
		initAVX512Constants()
		registerAVX512()
	}
}

func initAVX512Constants() {
	expMaxLog = archsimd.BroadcastFloat32x16(88.02969187150841)
	expMinLog = archsimd.BroadcastFloat32x16(-88.02969187150841)
	expLog2E = archsimd.BroadcastFloat32x16(1.44269504088896341)
	expHalf = archsimd.BroadcastFloat32x16(0.5)
	expLn2Hi = archsimd.BroadcastFloat32x16(0.693359375)
	expLn2Lo = archsimd.BroadcastFloat32x16(-2.12194440e-4)
	expP7 = archsimd.BroadcastFloat32x16(1.9875691500e-4)
	expP6 = archsimd.BroadcastFloat32x16(1.3981999507e-3)
	expP5 = archsimd.BroadcastFloat32x16(8.3334519073e-3)
	expP4 = archsimd.BroadcastFloat32x16(4.1665795894e-2)
	expP3 = archsimd.BroadcastFloat32x16(1.6666665459e-1)
	expP2 = archsimd.BroadcastFloat32x16(5.0000001201e-1)
	expOne = archsimd.BroadcastFloat32x16(1.0)
	exp127 = archsimd.BroadcastUint32x16(127)

	erfP = archsimd.BroadcastFloat32x16(0.3275911)
	erfA1 = archsimd.BroadcastFloat32x16(0.254829592)
	erfA2 = archsimd.BroadcastFloat32x16(-0.284496736)
	erfA3 = archsimd.BroadcastFloat32x16(1.421413741)
	erfA4 = archsimd.BroadcastFloat32x16(-1.453152027)
	erfA5 = archsimd.BroadcastFloat32x16(1.061405429)
	erfOne = archsimd.BroadcastFloat32x16(1.0)
	erfHalf = archsimd.BroadcastFloat32x16(0.5)
	erfRsqrt2 = archsimd.BroadcastFloat32x16(float32(1.0 / math.Sqrt2))
	erfSignMask = archsimd.BroadcastUint32x16(0x80000000)

	geluHalf = archsimd.BroadcastFloat32x16(0.5)
	geluOne = archsimd.BroadcastFloat32x16(1.0)
	geluSqrt2ByPi = archsimd.BroadcastFloat32x16(float32(math.Sqrt(2.0 / math.Pi)))
	geluC = archsimd.BroadcastFloat32x16(0.044715)

	tanhNine = archsimd.BroadcastFloat32x16(9.0)
	tanhNegNine = archsimd.BroadcastFloat32x16(-9.0)
	tanhTwo = archsimd.BroadcastFloat32x16(2.0)
	tanhOne = archsimd.BroadcastFloat32x16(1.0)

	// Float64
	expF64MaxLog = archsimd.BroadcastFloat64x8(709.7827128933840)
	expF64MinLog = archsimd.BroadcastFloat64x8(-708.3964185322641)
	expF64Log2E = archsimd.BroadcastFloat64x8(1.442695040888963407359924681001892137)
	expF64Half = archsimd.BroadcastFloat64x8(0.5)
	expF64Ln2Hi = archsimd.BroadcastFloat64x8(0.693147180559945309417232121458176568)
	expF64Ln2Lo = archsimd.BroadcastFloat64x8(2.31904681384629955841777123493867e-17)
	expF64One = archsimd.BroadcastFloat64x8(1.0)
	expF641023 = archsimd.BroadcastUint64x8(1023)

	expF64C11 = archsimd.BroadcastFloat64x8(2.505210838544172e-8)
	expF64C10 = archsimd.BroadcastFloat64x8(2.755731922398589e-7)
	expF64C9 = archsimd.BroadcastFloat64x8(2.755731922398589e-6)
	expF64C8 = archsimd.BroadcastFloat64x8(2.480158730158730e-5)
	expF64C7 = archsimd.BroadcastFloat64x8(1.984126984126984e-4)
	expF64C6 = archsimd.BroadcastFloat64x8(1.388888888888889e-3)
	expF64C5 = archsimd.BroadcastFloat64x8(8.333333333333333e-3)
	expF64C4 = archsimd.BroadcastFloat64x8(4.166666666666667e-2)
	expF64C3 = archsimd.BroadcastFloat64x8(1.666666666666667e-1)
	expF64C2 = archsimd.BroadcastFloat64x8(5.000000000000000e-1)

	erfF64P = archsimd.BroadcastFloat64x8(0.3275911)
	erfF64A1 = archsimd.BroadcastFloat64x8(0.254829592)
	erfF64A2 = archsimd.BroadcastFloat64x8(-0.284496736)
	erfF64A3 = archsimd.BroadcastFloat64x8(1.421413741)
	erfF64A4 = archsimd.BroadcastFloat64x8(-1.453152027)
	erfF64A5 = archsimd.BroadcastFloat64x8(1.061405429)
	erfF64One = archsimd.BroadcastFloat64x8(1.0)
	erfF64Half = archsimd.BroadcastFloat64x8(0.5)
	erfF64Rsqrt2 = archsimd.BroadcastFloat64x8(1.0 / math.Sqrt2)
	erfF64SignMask = archsimd.BroadcastUint64x8(0x8000000000000000)

	geluF64Half = archsimd.BroadcastFloat64x8(0.5)
	geluF64One = archsimd.BroadcastFloat64x8(1.0)
	geluF64Sqrt2ByPi = archsimd.BroadcastFloat64x8(math.Sqrt(2.0 / math.Pi))
	geluF64C = archsimd.BroadcastFloat64x8(0.044715)

	tanhF6419 = archsimd.BroadcastFloat64x8(19.0)
	tanhF64Neg19 = archsimd.BroadcastFloat64x8(-19.0)
	tanhF64Two = archsimd.BroadcastFloat64x8(2.0)
	tanhF64One = archsimd.BroadcastFloat64x8(1.0)

	// BFloat16
	bf16_512Bias7FFF = archsimd.BroadcastUint32x16(0x7FFF)
	bf16_512One = archsimd.BroadcastUint32x16(1)
	bf16_512MaskHi = archsimd.BroadcastUint32x16(0xFFFF0000)
}

func registerAVX512() {
	// Float32
	activations.Register[float32]("avx512:relu", compute.ActivationRelu, ReluAVX512, PriorityAVX512)
	activations.Register[float32]("avx512:sigmoid", compute.ActivationSigmoid, SigmoidAVX512, PriorityAVX512)
	activations.Register[float32]("avx512:hardsigmoid", compute.ActivationHardSigmoid, HardSigmoidAVX512, PriorityAVX512)
	activations.Register[float32]("avx512:leakyrelu", compute.ActivationLeakyRelu, LeakyReluAVX512, PriorityAVX512)
	activations.Register[float32]("avx512:selu", compute.ActivationSelu, SeluAVX512, PriorityAVX512)
	activations.Register[float32]("avx512:silu", compute.ActivationSilu, SiluAVX512, PriorityAVX512)
	activations.Register[float32]("avx512:hardswish", compute.ActivationHardSwish, HardSwishAVX512, PriorityAVX512)
	activations.Register[float32]("avx512:tanh", compute.ActivationTanh, TanhAVX512, PriorityAVX512)
	activations.RegisterKernel[float32]("avx512:gelu", compute.ActivationGelu, GeluExactAVX512, PriorityAVX512)
	activations.RegisterKernel[float32]("avx512:geluapprox", compute.ActivationGeluApproximate, GeluAVX512, PriorityAVX512)
	activations.RegisterSwiGLU[float32]("avx512:swiglu", SwiGLUAVX512, PriorityAVX512)

	// Float64
	activations.RegisterKernel[float64]("avx512:gelu", compute.ActivationGelu, GeluExactAVX512Float64, PriorityAVX512)
	activations.RegisterKernel[float64]("avx512:geluapprox", compute.ActivationGeluApproximate, GeluApproxAVX512Float64, PriorityAVX512)

	// BFloat16
	activations.Register[bfloat16.BFloat16]("avx512:relu", compute.ActivationRelu, reluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:sigmoid", compute.ActivationSigmoid, sigmoidBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:hardsigmoid", compute.ActivationHardSigmoid, hardSigmoidBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:leakyrelu", compute.ActivationLeakyRelu, leakyReluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:selu", compute.ActivationSelu, seluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:silu", compute.ActivationSilu, siluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:hardswish", compute.ActivationHardSwish, hardSwishBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:tanh", compute.ActivationTanh, tanhBF16AVX512, PriorityAVX512)
	activations.RegisterKernel[bfloat16.BFloat16]("avx512:gelu", compute.ActivationGelu, GeluExactAVX512BF16, PriorityAVX512)
	activations.RegisterKernel[bfloat16.BFloat16]("avx512:geluapprox", compute.ActivationGeluApproximate, GeluApproxAVX512BF16, PriorityAVX512)

	// Float16
	activations.Register[float16.Float16]("avx512:relu", compute.ActivationRelu, reluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:sigmoid", compute.ActivationSigmoid, sigmoidF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:hardsigmoid", compute.ActivationHardSigmoid, hardSigmoidF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:leakyrelu", compute.ActivationLeakyRelu, leakyReluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:selu", compute.ActivationSelu, seluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:silu", compute.ActivationSilu, siluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:hardswish", compute.ActivationHardSwish, hardSwishF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:tanh", compute.ActivationTanh, tanhF16AVX512, PriorityAVX512)
	activations.RegisterKernel[float16.Float16]("avx512:gelu", compute.ActivationGelu, GeluExactAVX512F16, PriorityAVX512)
	activations.RegisterKernel[float16.Float16]("avx512:geluapprox", compute.ActivationGeluApproximate, GeluApproxAVX512F16, PriorityAVX512)
}

func ReluAVX512(data []float32) {
	vZero := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		v.Max(vZero).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func HardSwishAVX512(data []float32) {
	vZero := archsimd.BroadcastFloat32x16(0)
	vOne := archsimd.BroadcastFloat32x16(1)
	vOneSixth := archsimd.BroadcastFloat32x16(1.0 / 6.0)
	vHalf := archsimd.BroadcastFloat32x16(0.5)

	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		scaled := v.MulAdd(vOneSixth, vHalf)
		clamped := scaled.Max(vZero).Min(vOne)
		v.Mul(clamped).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		x := data[i]
		shapeX := min(max(x*(1.0/6.0)+0.5, 0), 1)
		data[i] = x * shapeX
	}
}

// exp512 approximates e^x for 16 float32s using Cephes degree-7 Horner polynomial.
func exp512(x archsimd.Float32x16) archsimd.Float32x16 {
	xClamped := x.Max(expMinLog).Min(expMaxLog)
	z := xClamped.MulAdd(expLog2E, expHalf).FloorScaled(0)
	g := xClamped.Sub(z.Mul(expLn2Hi)).Sub(z.Mul(expLn2Lo))

	n := z.ConvertToInt32().AsUint32x16().Add(exp127).ShiftAllLeft(23).BitsToFloat32()

	poly := expP7.MulAdd(g, expP6)
	poly = poly.MulAdd(g, expP5)
	poly = poly.MulAdd(g, expP4)
	poly = poly.MulAdd(g, expP3)
	poly = poly.MulAdd(g, expP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(expOne)

	return n.Mul(poly)
}

func SiluAVX512(data []float32) {
	vOne := archsimd.BroadcastFloat32x16(1.0)
	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		negV := v.Neg()
		expNegV := exp512(negV)
		denom := vOne.Add(expNegV)
		v.Div(denom).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		x := data[i]
		data[i] = x / (1.0 + float32(math.Exp(float64(-x))))
	}
}

func tanh512(x archsimd.Float32x16) archsimd.Float32x16 {
	// For |x| >= 9.0, tanh(x) is +/-1.0 in float32. Clamping avoids exp overflow.
	xClamped := x.Max(tanhNegNine).Min(tanhNine)
	twoX := xClamped.Mul(tanhTwo)
	exp2x := exp512(twoX)
	num := exp2x.Sub(tanhOne)
	den := exp2x.Add(tanhOne)
	return num.Div(den)
}

func TanhAVX512(data []float32) {
	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		tanh512(v).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		data[i] = float32(math.Tanh(float64(data[i])))
	}
}

func GeluAVX512(in, out []float32) {
	// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	i := 0
	for ; i+32 <= len(in); i += 32 {
		x0 := archsimd.LoadFloat32x16(in[i : i+16])
		x1 := archsimd.LoadFloat32x16(in[i+16 : i+32])
		x3_0 := x0.Mul(x0).Mul(x0)
		x3_1 := x1.Mul(x1).Mul(x1)
		inner0 := geluSqrt2ByPi.Mul(x3_0.MulAdd(geluC, x0))
		inner1 := geluSqrt2ByPi.Mul(x3_1.MulAdd(geluC, x1))
		t0 := tanh512(inner0)
		t1 := tanh512(inner1)
		res0 := geluHalf.Mul(x0).Mul(geluOne.Add(t0))
		res1 := geluHalf.Mul(x1).Mul(geluOne.Add(t1))
		res0.Store(out[i : i+16])
		res1.Store(out[i+16 : i+32])
	}
	for ; i+16 <= len(in); i += 16 {
		x := archsimd.LoadFloat32x16(in[i : i+16])
		x3 := x.Mul(x).Mul(x)
		inner := geluSqrt2ByPi.Mul(x3.MulAdd(geluC, x))
		t := tanh512(inner)
		res := geluHalf.Mul(x).Mul(geluOne.Add(t))
		res.Store(out[i : i+16])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [16]float32
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat32x16(bufIn[:])
		x3 := x.Mul(x).Mul(x)
		inner := geluSqrt2ByPi.Mul(x3.MulAdd(geluC, x))
		t := tanh512(inner)
		res := geluHalf.Mul(x).Mul(geluOne.Add(t))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

func erf512(x archsimd.Float32x16) archsimd.Float32x16 {
	absX := x.Abs()
	t := erfOne.Div(erfOne.Add(erfP.Mul(absX)))

	poly := erfA5.MulAdd(t, erfA4)
	poly = poly.MulAdd(t, erfA3)
	poly = poly.MulAdd(t, erfA2)
	poly = poly.MulAdd(t, erfA1)
	poly = poly.Mul(t)

	expNegX2 := exp512(absX.Mul(absX).Neg())
	res := erfOne.Sub(poly.Mul(expNegX2))

	return res.ToBits().Xor(x.ToBits().And(erfSignMask)).BitsToFloat32()
}

func GeluExactAVX512(in, out []float32) {
	i := 0
	for ; i+32 <= len(in); i += 32 {
		x0 := archsimd.LoadFloat32x16(in[i : i+16])
		x1 := archsimd.LoadFloat32x16(in[i+16 : i+32])
		scaledX0 := x0.Mul(erfRsqrt2)
		scaledX1 := x1.Mul(erfRsqrt2)
		erfVal0 := erf512(scaledX0)
		erfVal1 := erf512(scaledX1)
		res0 := erfHalf.Mul(x0).Mul(erfOne.Add(erfVal0))
		res1 := erfHalf.Mul(x1).Mul(erfOne.Add(erfVal1))
		res0.Store(out[i : i+16])
		res1.Store(out[i+16 : i+32])
	}
	for ; i+16 <= len(in); i += 16 {
		x := archsimd.LoadFloat32x16(in[i : i+16])
		scaledX := x.Mul(erfRsqrt2)
		erfVal := erf512(scaledX)
		res := erfHalf.Mul(x).Mul(erfOne.Add(erfVal))
		res.Store(out[i : i+16])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [16]float32
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat32x16(bufIn[:])
		scaledX := x.Mul(erfRsqrt2)
		erfVal := erf512(scaledX)
		res := erfHalf.Mul(x).Mul(erfOne.Add(erfVal))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

// -----------------------------------------------------------------------------
// Float64 Vector Math & GELU Operations (AVX-512)
// -----------------------------------------------------------------------------

func exp512Float64(x archsimd.Float64x8) archsimd.Float64x8 {
	xClamped := x.Max(expF64MinLog).Min(expF64MaxLog)
	z := xClamped.MulAdd(expF64Log2E, expF64Half).FloorScaled(0)
	g := xClamped.Sub(z.Mul(expF64Ln2Hi)).Sub(z.Mul(expF64Ln2Lo))

	n := z.ConvertToInt64().AsUint64x8().Add(expF641023).ShiftAllLeft(52).BitsToFloat64()

	poly := expF64C11.MulAdd(g, expF64C10)
	poly = poly.MulAdd(g, expF64C9)
	poly = poly.MulAdd(g, expF64C8)
	poly = poly.MulAdd(g, expF64C7)
	poly = poly.MulAdd(g, expF64C6)
	poly = poly.MulAdd(g, expF64C5)
	poly = poly.MulAdd(g, expF64C4)
	poly = poly.MulAdd(g, expF64C3)
	poly = poly.MulAdd(g, expF64C2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(expF64One)

	return n.Mul(poly)
}

func tanh512Float64(x archsimd.Float64x8) archsimd.Float64x8 {
	xClamped := x.Max(tanhF64Neg19).Min(tanhF6419)
	twoX := xClamped.Mul(tanhF64Two)
	exp2x := exp512Float64(twoX)
	num := exp2x.Sub(tanhF64One)
	den := exp2x.Add(tanhF64One)
	return num.Div(den)
}

func erf512Float64(x archsimd.Float64x8) archsimd.Float64x8 {
	absX := x.Abs()
	t := erfF64One.Div(erfF64One.Add(erfF64P.Mul(absX)))

	poly := erfF64A5.MulAdd(t, erfF64A4)
	poly = poly.MulAdd(t, erfF64A3)
	poly = poly.MulAdd(t, erfF64A2)
	poly = poly.MulAdd(t, erfF64A1)
	poly = poly.Mul(t)

	expNegX2 := exp512Float64(absX.Mul(absX).Neg())
	res := erfF64One.Sub(poly.Mul(expNegX2))

	return res.ToBits().Xor(x.ToBits().And(erfF64SignMask)).BitsToFloat64()
}

func GeluExactAVX512Float64(in, out []float64) {
	i := 0
	for ; i+16 <= len(in); i += 16 {
		x0 := archsimd.LoadFloat64x8(in[i : i+8])
		x1 := archsimd.LoadFloat64x8(in[i+8 : i+16])
		scaledX0 := x0.Mul(erfF64Rsqrt2)
		scaledX1 := x1.Mul(erfF64Rsqrt2)
		erfVal0 := erf512Float64(scaledX0)
		erfVal1 := erf512Float64(scaledX1)
		res0 := erfF64Half.Mul(x0).Mul(erfF64One.Add(erfVal0))
		res1 := erfF64Half.Mul(x1).Mul(erfF64One.Add(erfVal1))
		res0.Store(out[i : i+8])
		res1.Store(out[i+8 : i+16])
	}
	for ; i+8 <= len(in); i += 8 {
		x := archsimd.LoadFloat64x8(in[i : i+8])
		scaledX := x.Mul(erfF64Rsqrt2)
		erfVal := erf512Float64(scaledX)
		res := erfF64Half.Mul(x).Mul(erfF64One.Add(erfVal))
		res.Store(out[i : i+8])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [8]float64
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat64x8(bufIn[:])
		scaledX := x.Mul(erfF64Rsqrt2)
		erfVal := erf512Float64(scaledX)
		res := erfF64Half.Mul(x).Mul(erfF64One.Add(erfVal))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

func GeluApproxAVX512Float64(in, out []float64) {
	i := 0
	for ; i+16 <= len(in); i += 16 {
		x0 := archsimd.LoadFloat64x8(in[i : i+8])
		x1 := archsimd.LoadFloat64x8(in[i+8 : i+16])
		x3_0 := x0.Mul(x0).Mul(x0)
		x3_1 := x1.Mul(x1).Mul(x1)
		inner0 := geluF64Sqrt2ByPi.Mul(x3_0.MulAdd(geluF64C, x0))
		inner1 := geluF64Sqrt2ByPi.Mul(x3_1.MulAdd(geluF64C, x1))
		t0 := tanh512Float64(inner0)
		t1 := tanh512Float64(inner1)
		res0 := geluF64Half.Mul(x0).Mul(geluF64One.Add(t0))
		res1 := geluF64Half.Mul(x1).Mul(geluF64One.Add(t1))
		res0.Store(out[i : i+8])
		res1.Store(out[i+8 : i+16])
	}
	for ; i+8 <= len(in); i += 8 {
		x := archsimd.LoadFloat64x8(in[i : i+8])
		x3 := x.Mul(x).Mul(x)
		inner := geluF64Sqrt2ByPi.Mul(x3.MulAdd(geluF64C, x))
		t := tanh512Float64(inner)
		res := geluF64Half.Mul(x).Mul(geluF64One.Add(t))
		res.Store(out[i : i+8])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [8]float64
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat64x8(bufIn[:])
		x3 := x.Mul(x).Mul(x)
		inner := geluF64Sqrt2ByPi.Mul(x3.MulAdd(geluF64C, x))
		t := tanh512Float64(inner)
		res := geluF64Half.Mul(x).Mul(geluF64One.Add(t))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

// -----------------------------------------------------------------------------
// BFloat16 Vector GELU Operations (AVX-512)
// -----------------------------------------------------------------------------

func GeluExactAVX512BF16(in, out []bfloat16.BFloat16) {
	if len(in) == 0 {
		return
	}
	i := 0
	for ; i+32 <= len(in); i += 32 {
		u32 := archsimd.LoadUint32x16(unsafe.Slice((*uint32)(unsafe.Pointer(&in[i])), 16))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16_512MaskHi).BitsToFloat32()

		scaledEven := even.Mul(erfRsqrt2)
		erfEven := erf512(scaledEven)
		evenRes := erfHalf.Mul(even).Mul(erfOne.Add(erfEven))

		scaledOdd := odd.Mul(erfRsqrt2)
		erfOdd := erf512(scaledOdd)
		oddRes := erfHalf.Mul(odd).Mul(erfOne.Add(erfOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16_512Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16_512One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16_512Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16_512One))
		oddRounded := oddBits.Add(oddBias).And(bf16_512MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&out[i])), 16))
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [32]bfloat16.BFloat16
		copy(bufIn[:rem], in[i:])
		u32 := archsimd.LoadUint32x16(unsafe.Slice((*uint32)(unsafe.Pointer(&bufIn[0])), 16))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16_512MaskHi).BitsToFloat32()

		scaledEven := even.Mul(erfRsqrt2)
		erfEven := erf512(scaledEven)
		evenRes := erfHalf.Mul(even).Mul(erfOne.Add(erfEven))

		scaledOdd := odd.Mul(erfRsqrt2)
		erfOdd := erf512(scaledOdd)
		oddRes := erfHalf.Mul(odd).Mul(erfOne.Add(erfOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16_512Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16_512One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16_512Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16_512One))
		oddRounded := oddBits.Add(oddBias).And(bf16_512MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&bufOut[0])), 16))
		copy(out[i:], bufOut[:rem])
	}
}

func GeluApproxAVX512BF16(in, out []bfloat16.BFloat16) {
	if len(in) == 0 {
		return
	}
	i := 0
	for ; i+32 <= len(in); i += 32 {
		u32 := archsimd.LoadUint32x16(unsafe.Slice((*uint32)(unsafe.Pointer(&in[i])), 16))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16_512MaskHi).BitsToFloat32()

		x3Even := even.Mul(even).Mul(even)
		innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, even))
		tEven := tanh512(innerEven)
		evenRes := geluHalf.Mul(even).Mul(geluOne.Add(tEven))

		x3Odd := odd.Mul(odd).Mul(odd)
		innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, odd))
		tOdd := tanh512(innerOdd)
		oddRes := geluHalf.Mul(odd).Mul(geluOne.Add(tOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16_512Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16_512One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16_512Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16_512One))
		oddRounded := oddBits.Add(oddBias).And(bf16_512MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&out[i])), 16))
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [32]bfloat16.BFloat16
		copy(bufIn[:rem], in[i:])
		u32 := archsimd.LoadUint32x16(unsafe.Slice((*uint32)(unsafe.Pointer(&bufIn[0])), 16))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16_512MaskHi).BitsToFloat32()

		x3Even := even.Mul(even).Mul(even)
		innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, even))
		tEven := tanh512(innerEven)
		evenRes := geluHalf.Mul(even).Mul(geluOne.Add(tEven))

		x3Odd := odd.Mul(odd).Mul(odd)
		innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, odd))
		tOdd := tanh512(innerOdd)
		oddRes := geluHalf.Mul(odd).Mul(geluOne.Add(tOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16_512Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16_512One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16_512Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16_512One))
		oddRounded := oddBits.Add(oddBias).And(bf16_512MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&bufOut[0])), 16))
		copy(out[i:], bufOut[:rem])
	}
}

// -----------------------------------------------------------------------------
// Float16 Vector GELU Operations (AVX-512)
// -----------------------------------------------------------------------------

func GeluExactAVX512F16(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	if vecLen == 32 {
		i := 0
		for ; i+32 <= n; i += 32 {
			v := float16.LoadFloat16s(in[i : i+32])
			even, odd := float16.ToFloat32SIMD(v)
			evenArch := *(*archsimd.Float32x16)(unsafe.Pointer(&even))
			oddArch := *(*archsimd.Float32x16)(unsafe.Pointer(&odd))

			scaledEven := evenArch.Mul(erfRsqrt2)
			erfEven := erf512(scaledEven)
			resEven := erfHalf.Mul(evenArch).Mul(erfOne.Add(erfEven))
			resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

			scaledOdd := oddArch.Mul(erfRsqrt2)
			erfOdd := erf512(scaledOdd)
			resOdd := erfHalf.Mul(oddArch).Mul(erfOne.Add(erfOdd))
			resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

			res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
			float16.StoreFloat16s(res, out[i : i+32])
		}
		if i < n {
			v, _ := float16.LoadFloat16sPart(in[i:])
			even, odd := float16.ToFloat32SIMD(v)
			evenArch := *(*archsimd.Float32x16)(unsafe.Pointer(&even))
			oddArch := *(*archsimd.Float32x16)(unsafe.Pointer(&odd))

			scaledEven := evenArch.Mul(erfRsqrt2)
			erfEven := erf512(scaledEven)
			resEven := erfHalf.Mul(evenArch).Mul(erfOne.Add(erfEven))
			resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

			scaledOdd := oddArch.Mul(erfRsqrt2)
			erfOdd := erf512(scaledOdd)
			resOdd := erfHalf.Mul(oddArch).Mul(erfOne.Add(erfOdd))
			resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

			res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
			float16.StoreFloat16sPart(res, out[i:])
		}
		return
	}
	// Fallback when portable simd vector length is not 32
	rsqrt2 := float32(1.0 / math.Sqrt2)
	for i := 0; i < n; i++ {
		x := in[i].Float32()
		erfVal := float32(math.Erf(float64(x * rsqrt2)))
		out[i] = float16.FromFloat32(0.5 * x * (1.0 + erfVal))
	}
}

func GeluApproxAVX512F16(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	if vecLen == 32 {
		i := 0
		for ; i+32 <= n; i += 32 {
			v := float16.LoadFloat16s(in[i : i+32])
			even, odd := float16.ToFloat32SIMD(v)
			evenArch := *(*archsimd.Float32x16)(unsafe.Pointer(&even))
			oddArch := *(*archsimd.Float32x16)(unsafe.Pointer(&odd))

			x3Even := evenArch.Mul(evenArch).Mul(evenArch)
			innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, evenArch))
			tEven := tanh512(innerEven)
			resEven := geluHalf.Mul(evenArch).Mul(geluOne.Add(tEven))
			resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

			x3Odd := oddArch.Mul(oddArch).Mul(oddArch)
			innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, oddArch))
			tOdd := tanh512(innerOdd)
			resOdd := geluHalf.Mul(oddArch).Mul(geluOne.Add(tOdd))
			resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

			res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
			float16.StoreFloat16s(res, out[i : i+32])
		}
		if i < n {
			v, _ := float16.LoadFloat16sPart(in[i:])
			even, odd := float16.ToFloat32SIMD(v)
			evenArch := *(*archsimd.Float32x16)(unsafe.Pointer(&even))
			oddArch := *(*archsimd.Float32x16)(unsafe.Pointer(&odd))

			x3Even := evenArch.Mul(evenArch).Mul(evenArch)
			innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, evenArch))
			tEven := tanh512(innerEven)
			resEven := geluHalf.Mul(evenArch).Mul(geluOne.Add(tEven))
			resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

			x3Odd := oddArch.Mul(oddArch).Mul(oddArch)
			innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, oddArch))
			tOdd := tanh512(innerOdd)
			resOdd := geluHalf.Mul(oddArch).Mul(geluOne.Add(tOdd))
			resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

			res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
			float16.StoreFloat16sPart(res, out[i:])
		}
		return
	}
	// Fallback when portable simd vector length is not 32
	sqrt2ByPi := float32(math.Sqrt(2.0 / math.Pi))
	for i := 0; i < n; i++ {
		x := in[i].Float32()
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		out[i] = float16.FromFloat32(0.5 * x * (1.0 + float32(math.Tanh(float64(inner)))))
	}
}

// Half-precision implementations:
const halfChunk = 64

func reluBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		ReluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func hardSwishBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		HardSwishAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func siluBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SiluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func tanhBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		TanhAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func reluF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		ReluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func hardSwishF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		HardSwishAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func siluF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SiluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func tanhF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		TanhAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func SigmoidAVX512(data []float32) {
	vOne := archsimd.BroadcastFloat32x16(1.0)
	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		expNegV := exp512(v.Neg())
		denom := vOne.Add(expNegV)
		vOne.Div(denom).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		data[i] = float32(1.0 / (1.0 + math.Exp(float64(-data[i]))))
	}
}

func HardSigmoidAVX512(data []float32) {
	vZero := archsimd.BroadcastFloat32x16(0)
	vOne := archsimd.BroadcastFloat32x16(1)
	vSlope := archsimd.BroadcastFloat32x16(0.2)
	vBias := archsimd.BroadcastFloat32x16(0.5)
	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		v.MulAdd(vSlope, vBias).Max(vZero).Min(vOne).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		data[i] = min(max(data[i]*0.2+0.5, 0), 1)
	}
}

func LeakyReluAVX512(data []float32) {
	vZero := archsimd.BroadcastFloat32x16(0)
	vAlpha := archsimd.BroadcastFloat32x16(0.3)
	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		scaled := v.Mul(vAlpha)
		mask := v.GreaterEqual(vZero)
		v.Merge(scaled, mask).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		if data[i] < 0 {
			data[i] = 0.3 * data[i]
		}
	}
}

const (
	seluScaleAVX512      = 1.0507009873554804934193349852946
	seluScaleAlphaAVX512 = 1.0507009873554804934193349852946 * 1.6732632423543772848170429916717
)

func SeluAVX512(data []float32) {
	vZero := archsimd.BroadcastFloat32x16(0)
	vOne := archsimd.BroadcastFloat32x16(1)
	vScale := archsimd.BroadcastFloat32x16(float32(seluScaleAVX512))
	vScaleAlpha := archsimd.BroadcastFloat32x16(float32(seluScaleAlphaAVX512))
	i := 0
	for ; i+16 <= len(data); i += 16 {
		v := archsimd.LoadFloat32x16(data[i : i+16])
		pos := v.Mul(vScale)
		neg := exp512(v).Sub(vOne).Mul(vScaleAlpha)
		mask := v.Greater(vZero)
		pos.Merge(neg, mask).Store(data[i : i+16])
	}
	for ; i < len(data); i++ {
		x := data[i]
		if x > 0 {
			data[i] = float32(seluScaleAVX512 * float64(x))
		} else {
			data[i] = float32(seluScaleAlphaAVX512 * (math.Exp(float64(x)) - 1.0))
		}
	}
}

func SwiGLUAVX512(in, out []float32, numRows, hiddenDim int) {
	vOne := archsimd.BroadcastFloat32x16(1.0)
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		j := 0
		for ; j+16 <= hiddenDim; j += 16 {
			gate := archsimd.LoadFloat32x16(in[inOffset+j : inOffset+j+16])
			val := archsimd.LoadFloat32x16(in[inOffset+hiddenDim+j : inOffset+hiddenDim+j+16])
			expNegGate := exp512(gate.Neg())
			denom := vOne.Add(expNegGate)
			siluGate := gate.Div(denom)
			siluGate.Mul(val).Store(out[outOffset+j : outOffset+j+16])
		}
		for ; j < hiddenDim; j++ {
			gate := in[inOffset+j]
			val := in[inOffset+hiddenDim+j]
			siluGate := gate / (1.0 + float32(math.Exp(float64(-gate))))
			out[outOffset+j] = siluGate * val
		}
	}
}

func sigmoidBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SigmoidAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func hardSigmoidBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		HardSigmoidAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func leakyReluBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		LeakyReluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func seluBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SeluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func sigmoidF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SigmoidAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func hardSigmoidF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		HardSigmoidAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func leakyReluF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		LeakyReluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func seluF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SeluAVX512(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

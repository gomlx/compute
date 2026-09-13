// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

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

const PriorityAVX2 = gobackend.PriorityArch

var (
	expMaxLog archsimd.Float32x8
	expMinLog archsimd.Float32x8
	expLog2E  archsimd.Float32x8
	expHalf   archsimd.Float32x8
	expLn2Hi  archsimd.Float32x8
	expLn2Lo  archsimd.Float32x8
	expP7     archsimd.Float32x8
	expP6     archsimd.Float32x8
	expP5     archsimd.Float32x8
	expP4     archsimd.Float32x8
	expP3     archsimd.Float32x8
	expP2     archsimd.Float32x8
	expOne    archsimd.Float32x8
	exp127    archsimd.Uint32x8

	erfP        archsimd.Float32x8
	erfA1       archsimd.Float32x8
	erfA2       archsimd.Float32x8
	erfA3       archsimd.Float32x8
	erfA4       archsimd.Float32x8
	erfA5       archsimd.Float32x8
	erfOne      archsimd.Float32x8
	erfHalf     archsimd.Float32x8
	erfRsqrt2   archsimd.Float32x8
	erfSignMask archsimd.Uint32x8

	geluHalf      archsimd.Float32x8
	geluOne       archsimd.Float32x8
	geluSqrt2ByPi archsimd.Float32x8
	geluC         archsimd.Float32x8
	tanhNine      archsimd.Float32x8
	tanhNegNine   archsimd.Float32x8
	tanhTwo       archsimd.Float32x8
	tanhOne       archsimd.Float32x8

	// Float64 constants
	expF64MaxLog archsimd.Float64x4
	expF64MinLog archsimd.Float64x4
	expF64Log2E  archsimd.Float64x4
	expF64Half   archsimd.Float64x4
	expF64Ln2Hi  archsimd.Float64x4
	expF64Ln2Lo  archsimd.Float64x4
	expF64One       archsimd.Float64x4
	expF641023      archsimd.Uint64x4
	expF64Magic     archsimd.Float64x4
	expF64MagicBits archsimd.Uint64x4

	expF64C11 archsimd.Float64x4
	expF64C10 archsimd.Float64x4
	expF64C9  archsimd.Float64x4
	expF64C8  archsimd.Float64x4
	expF64C7  archsimd.Float64x4
	expF64C6  archsimd.Float64x4
	expF64C5  archsimd.Float64x4
	expF64C4  archsimd.Float64x4
	expF64C3  archsimd.Float64x4
	expF64C2  archsimd.Float64x4

	erfF64P        archsimd.Float64x4
	erfF64A1       archsimd.Float64x4
	erfF64A2       archsimd.Float64x4
	erfF64A3       archsimd.Float64x4
	erfF64A4       archsimd.Float64x4
	erfF64A5       archsimd.Float64x4
	erfF64One      archsimd.Float64x4
	erfF64Half     archsimd.Float64x4
	erfF64Rsqrt2   archsimd.Float64x4
	erfF64SignMask archsimd.Uint64x4

	geluF64Half      archsimd.Float64x4
	geluF64One       archsimd.Float64x4
	geluF64Sqrt2ByPi archsimd.Float64x4
	geluF64C         archsimd.Float64x4
	tanhF6419        archsimd.Float64x4
	tanhF64Neg19     archsimd.Float64x4
	tanhF64Two       archsimd.Float64x4
	tanhF64One       archsimd.Float64x4

	// BFloat16 constants
	bf16Bias7FFF archsimd.Uint32x8
	bf16One      archsimd.Uint32x8
	bf16MaskHi   archsimd.Uint32x8
)

func init() {
	if gobackend.IsAVX2Allowed {
		initAVX2Constants()
		registerAVX2()
	}
}

func initAVX2Constants() {
	expMaxLog = archsimd.BroadcastFloat32x8(88.02969187150841)
	expMinLog = archsimd.BroadcastFloat32x8(-88.02969187150841)
	expLog2E = archsimd.BroadcastFloat32x8(1.44269504088896341)
	expHalf = archsimd.BroadcastFloat32x8(0.5)
	expLn2Hi = archsimd.BroadcastFloat32x8(0.693359375)
	expLn2Lo = archsimd.BroadcastFloat32x8(-2.12194440e-4)
	expP7 = archsimd.BroadcastFloat32x8(1.9875691500e-4)
	expP6 = archsimd.BroadcastFloat32x8(1.3981999507e-3)
	expP5 = archsimd.BroadcastFloat32x8(8.3334519073e-3)
	expP4 = archsimd.BroadcastFloat32x8(4.1665795894e-2)
	expP3 = archsimd.BroadcastFloat32x8(1.6666665459e-1)
	expP2 = archsimd.BroadcastFloat32x8(5.0000001201e-1)
	expOne = archsimd.BroadcastFloat32x8(1.0)
	exp127 = archsimd.BroadcastUint32x8(127)

	erfP = archsimd.BroadcastFloat32x8(0.3275911)
	erfA1 = archsimd.BroadcastFloat32x8(0.254829592)
	erfA2 = archsimd.BroadcastFloat32x8(-0.284496736)
	erfA3 = archsimd.BroadcastFloat32x8(1.421413741)
	erfA4 = archsimd.BroadcastFloat32x8(-1.453152027)
	erfA5 = archsimd.BroadcastFloat32x8(1.061405429)
	erfOne = archsimd.BroadcastFloat32x8(1.0)
	erfHalf = archsimd.BroadcastFloat32x8(0.5)
	erfRsqrt2 = archsimd.BroadcastFloat32x8(float32(1.0 / math.Sqrt2))
	erfSignMask = archsimd.BroadcastUint32x8(0x80000000)

	geluHalf = archsimd.BroadcastFloat32x8(0.5)
	geluOne = archsimd.BroadcastFloat32x8(1.0)
	geluSqrt2ByPi = archsimd.BroadcastFloat32x8(float32(math.Sqrt(2.0 / math.Pi)))
	geluC = archsimd.BroadcastFloat32x8(0.044715)

	tanhNine = archsimd.BroadcastFloat32x8(9.0)
	tanhNegNine = archsimd.BroadcastFloat32x8(-9.0)
	tanhTwo = archsimd.BroadcastFloat32x8(2.0)
	tanhOne = archsimd.BroadcastFloat32x8(1.0)

	// Float64
	expF64MaxLog = archsimd.BroadcastFloat64x4(709.7827128933840)
	expF64MinLog = archsimd.BroadcastFloat64x4(-708.3964185322641)
	expF64Log2E = archsimd.BroadcastFloat64x4(1.442695040888963407359924681001892137)
	expF64Half = archsimd.BroadcastFloat64x4(0.5)
	expF64Ln2Hi = archsimd.BroadcastFloat64x4(0.693147180559945309417232121458176568)
	expF64Ln2Lo = archsimd.BroadcastFloat64x4(2.31904681384629955841777123493867e-17)
	expF64One = archsimd.BroadcastFloat64x4(1.0)
	expF641023 = archsimd.BroadcastUint64x4(1023)
	expF64Magic = archsimd.BroadcastFloat64x4(6755399441055744.0)
	expF64MagicBits = archsimd.BroadcastUint64x4(0x4338000000000000)

	expF64C11 = archsimd.BroadcastFloat64x4(2.505210838544172e-8)
	expF64C10 = archsimd.BroadcastFloat64x4(2.755731922398589e-7)
	expF64C9 = archsimd.BroadcastFloat64x4(2.755731922398589e-6)
	expF64C8 = archsimd.BroadcastFloat64x4(2.480158730158730e-5)
	expF64C7 = archsimd.BroadcastFloat64x4(1.984126984126984e-4)
	expF64C6 = archsimd.BroadcastFloat64x4(1.388888888888889e-3)
	expF64C5 = archsimd.BroadcastFloat64x4(8.333333333333333e-3)
	expF64C4 = archsimd.BroadcastFloat64x4(4.166666666666667e-2)
	expF64C3 = archsimd.BroadcastFloat64x4(1.666666666666667e-1)
	expF64C2 = archsimd.BroadcastFloat64x4(5.000000000000000e-1)

	erfF64P = archsimd.BroadcastFloat64x4(0.3275911)
	erfF64A1 = archsimd.BroadcastFloat64x4(0.254829592)
	erfF64A2 = archsimd.BroadcastFloat64x4(-0.284496736)
	erfF64A3 = archsimd.BroadcastFloat64x4(1.421413741)
	erfF64A4 = archsimd.BroadcastFloat64x4(-1.453152027)
	erfF64A5 = archsimd.BroadcastFloat64x4(1.061405429)
	erfF64One = archsimd.BroadcastFloat64x4(1.0)
	erfF64Half = archsimd.BroadcastFloat64x4(0.5)
	erfF64Rsqrt2 = archsimd.BroadcastFloat64x4(1.0 / math.Sqrt2)
	erfF64SignMask = archsimd.BroadcastUint64x4(0x8000000000000000)

	geluF64Half = archsimd.BroadcastFloat64x4(0.5)
	geluF64One = archsimd.BroadcastFloat64x4(1.0)
	geluF64Sqrt2ByPi = archsimd.BroadcastFloat64x4(math.Sqrt(2.0 / math.Pi))
	geluF64C = archsimd.BroadcastFloat64x4(0.044715)

	tanhF6419 = archsimd.BroadcastFloat64x4(19.0)
	tanhF64Neg19 = archsimd.BroadcastFloat64x4(-19.0)
	tanhF64Two = archsimd.BroadcastFloat64x4(2.0)
	tanhF64One = archsimd.BroadcastFloat64x4(1.0)

	// BFloat16
	bf16Bias7FFF = archsimd.BroadcastUint32x8(0x7FFF)
	bf16One = archsimd.BroadcastUint32x8(1)
	bf16MaskHi = archsimd.BroadcastUint32x8(0xFFFF0000)
}

func registerAVX2() {
	// Float32
	activations.Register[float32]("avx2:relu", compute.ActivationRelu, ReluAVX2, PriorityAVX2)
	activations.Register[float32]("avx2:hardswish", compute.ActivationHardSwish, HardSwishAVX2, PriorityAVX2)
	activations.Register[float32]("avx2:silu", compute.ActivationSilu, SiluAVX2, PriorityAVX2)
	activations.RegisterKernel[float32]("avx2:gelu", compute.ActivationGelu, GeluExactAVX2, PriorityAVX2)
	activations.RegisterKernel[float32]("avx2:geluapprox", compute.ActivationGeluApproximate, GeluAVX2, PriorityAVX2)
	activations.Register[float32]("avx2:tanh", compute.ActivationTanh, TanhAVX2, PriorityAVX2)

	// Float64
	activations.RegisterKernel[float64]("avx2:gelu", compute.ActivationGelu, GeluExactAVX2Float64, PriorityAVX2)
	activations.RegisterKernel[float64]("avx2:geluapprox", compute.ActivationGeluApproximate, GeluApproxAVX2Float64, PriorityAVX2)

	// BFloat16
	activations.Register[bfloat16.BFloat16]("avx2:relu", compute.ActivationRelu, reluBF16AVX2, PriorityAVX2)
	activations.Register[bfloat16.BFloat16]("avx2:hardswish", compute.ActivationHardSwish, hardSwishBF16AVX2, PriorityAVX2)
	activations.Register[bfloat16.BFloat16]("avx2:silu", compute.ActivationSilu, siluBF16AVX2, PriorityAVX2)
	activations.RegisterKernel[bfloat16.BFloat16]("avx2:gelu", compute.ActivationGelu, GeluExactAVX2BF16, PriorityAVX2)
	activations.RegisterKernel[bfloat16.BFloat16]("avx2:geluapprox", compute.ActivationGeluApproximate, GeluApproxAVX2BF16, PriorityAVX2)
	activations.Register[bfloat16.BFloat16]("avx2:tanh", compute.ActivationTanh, tanhBF16AVX2, PriorityAVX2)

	// Float16
	activations.Register[float16.Float16]("avx2:relu", compute.ActivationRelu, reluF16AVX2, PriorityAVX2)
	activations.Register[float16.Float16]("avx2:hardswish", compute.ActivationHardSwish, hardSwishF16AVX2, PriorityAVX2)
	activations.Register[float16.Float16]("avx2:silu", compute.ActivationSilu, siluF16AVX2, PriorityAVX2)
	activations.RegisterKernel[float16.Float16]("avx2:gelu", compute.ActivationGelu, GeluExactAVX2F16, PriorityAVX2)
	activations.RegisterKernel[float16.Float16]("avx2:geluapprox", compute.ActivationGeluApproximate, GeluApproxAVX2F16, PriorityAVX2)
	activations.Register[float16.Float16]("avx2:tanh", compute.ActivationTanh, tanhF16AVX2, PriorityAVX2)
}

func ReluAVX2(data []float32) {
	vZero := archsimd.BroadcastFloat32x8(0)
	i := 0
	for ; i+8 <= len(data); i += 8 {
		v := archsimd.LoadFloat32x8(data[i : i+8])
		v.Max(vZero).Store(data[i : i+8])
	}
	for ; i < len(data); i++ {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func HardSwishAVX2(data []float32) {
	vZero := archsimd.BroadcastFloat32x8(0)
	vOne := archsimd.BroadcastFloat32x8(1)
	vOneSixth := archsimd.BroadcastFloat32x8(1.0 / 6.0)
	vHalf := archsimd.BroadcastFloat32x8(0.5)

	i := 0
	for ; i+8 <= len(data); i += 8 {
		v := archsimd.LoadFloat32x8(data[i : i+8])
		scaled := v.MulAdd(vOneSixth, vHalf)
		clamped := scaled.Max(vZero).Min(vOne)
		v.Mul(clamped).Store(data[i : i+8])
	}
	for ; i < len(data); i++ {
		x := data[i]
		shapeX := min(max(x*(1.0/6.0)+0.5, 0), 1)
		data[i] = x * shapeX
	}
}

// exp256 approximates e^x for 8 float32s using Cephes degree-7 Horner polynomial.
func exp256(x archsimd.Float32x8) archsimd.Float32x8 {
	xClamped := x.Max(expMinLog).Min(expMaxLog)
	z := xClamped.MulAdd(expLog2E, expHalf).Floor()
	g := xClamped.Sub(z.Mul(expLn2Hi)).Sub(z.Mul(expLn2Lo))

	n := z.ConvertToInt32().AsUint32x8().Add(exp127).ShiftAllLeft(23).BitsToFloat32()

	poly := expP7.MulAdd(g, expP6)
	poly = poly.MulAdd(g, expP5)
	poly = poly.MulAdd(g, expP4)
	poly = poly.MulAdd(g, expP3)
	poly = poly.MulAdd(g, expP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(expOne)

	return n.Mul(poly)
}

func SiluAVX2(data []float32) {
	vOne := archsimd.BroadcastFloat32x8(1.0)
	i := 0
	for ; i+8 <= len(data); i += 8 {
		v := archsimd.LoadFloat32x8(data[i : i+8])
		negV := v.Neg()
		expNegV := exp256(negV)
		denom := vOne.Add(expNegV)
		v.Div(denom).Store(data[i : i+8])
	}
	for ; i < len(data); i++ {
		x := data[i]
		data[i] = x / (1.0 + float32(math.Exp(float64(-x))))
	}
}

func tanh256(x archsimd.Float32x8) archsimd.Float32x8 {
	// For |x| >= 9.0, tanh(x) is +/-1.0 in float32. Clamping avoids exp overflow.
	xClamped := x.Max(tanhNegNine).Min(tanhNine)
	twoX := xClamped.Mul(tanhTwo)
	exp2x := exp256(twoX)
	num := exp2x.Sub(tanhOne)
	den := exp2x.Add(tanhOne)
	return num.Div(den)
}

func TanhAVX2(data []float32) {
	i := 0
	for ; i+8 <= len(data); i += 8 {
		v := archsimd.LoadFloat32x8(data[i : i+8])
		tanh256(v).Store(data[i : i+8])
	}
	for ; i < len(data); i++ {
		data[i] = float32(math.Tanh(float64(data[i])))
	}
}

func GeluAVX2(in, out []float32) {
	// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	i := 0
	for ; i+16 <= len(in); i += 16 {
		x0 := archsimd.LoadFloat32x8(in[i : i+8])
		x1 := archsimd.LoadFloat32x8(in[i+8 : i+16])
		x3_0 := x0.Mul(x0).Mul(x0)
		x3_1 := x1.Mul(x1).Mul(x1)
		inner0 := geluSqrt2ByPi.Mul(x3_0.MulAdd(geluC, x0))
		inner1 := geluSqrt2ByPi.Mul(x3_1.MulAdd(geluC, x1))
		t0 := tanh256(inner0)
		t1 := tanh256(inner1)
		res0 := geluHalf.Mul(x0).Mul(geluOne.Add(t0))
		res1 := geluHalf.Mul(x1).Mul(geluOne.Add(t1))
		res0.Store(out[i : i+8])
		res1.Store(out[i+8 : i+16])
	}
	for ; i+8 <= len(in); i += 8 {
		x := archsimd.LoadFloat32x8(in[i : i+8])
		x3 := x.Mul(x).Mul(x)
		inner := geluSqrt2ByPi.Mul(x3.MulAdd(geluC, x))
		t := tanh256(inner)
		res := geluHalf.Mul(x).Mul(geluOne.Add(t))
		res.Store(out[i : i+8])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [8]float32
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat32x8(bufIn[:])
		x3 := x.Mul(x).Mul(x)
		inner := geluSqrt2ByPi.Mul(x3.MulAdd(geluC, x))
		t := tanh256(inner)
		res := geluHalf.Mul(x).Mul(geluOne.Add(t))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

func erf256(x archsimd.Float32x8) archsimd.Float32x8 {
	absX := x.Abs()
	t := erfOne.Div(erfOne.Add(erfP.Mul(absX)))

	poly := erfA5.MulAdd(t, erfA4)
	poly = poly.MulAdd(t, erfA3)
	poly = poly.MulAdd(t, erfA2)
	poly = poly.MulAdd(t, erfA1)
	poly = poly.Mul(t)

	expNegX2 := exp256(absX.Mul(absX).Neg())
	res := erfOne.Sub(poly.Mul(expNegX2))

	return res.ToBits().Xor(x.ToBits().And(erfSignMask)).BitsToFloat32()
}

func GeluExactAVX2(in, out []float32) {
	i := 0
	for ; i+8 <= len(in); i += 8 {
		x := archsimd.LoadFloat32x8(in[i : i+8])
		scaledX := x.Mul(erfRsqrt2)
		erfVal := erf256(scaledX)
		res := erfHalf.Mul(x).Mul(erfOne.Add(erfVal))
		res.Store(out[i : i+8])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [8]float32
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat32x8(bufIn[:])
		scaledX := x.Mul(erfRsqrt2)
		erfVal := erf256(scaledX)
		res := erfHalf.Mul(x).Mul(erfOne.Add(erfVal))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

// Half-precision implementations:
const halfChunk = 64

func reluBF16AVX2(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		ReluAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func hardSwishBF16AVX2(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		HardSwishAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func siluBF16AVX2(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SiluAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func tanhBF16AVX2(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		TanhAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = bfloat16.FromFloat32(buf[j])
		}
	}
}

func reluF16AVX2(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		ReluAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func hardSwishF16AVX2(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		HardSwishAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func siluF16AVX2(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		SiluAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

func tanhF16AVX2(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		TanhAVX2(buf[:len(chunk)])
		for j := range chunk {
			chunk[j] = float16.FromFloat32(buf[j])
		}
	}
}

// -----------------------------------------------------------------------------
// Float64 Vector Math & GELU Operations (AVX2)
// -----------------------------------------------------------------------------

func exp256Float64(x archsimd.Float64x4) archsimd.Float64x4 {
	xClamped := x.Max(expF64MinLog).Min(expF64MaxLog)
	z := xClamped.Mul(expF64Log2E)

	// Round to nearest integer using IEEE-754 mantissa alignment.
	zRound := z.Add(expF64Magic)
	nFloat := zRound.Sub(expF64Magic)

	// Integer n in bits via two's complement subtraction.
	nInt := zRound.ToBits().Sub(expF64MagicBits)
	scale := nInt.Add(expF641023).ShiftAllLeft(52).BitsToFloat64()

	// Reduced argument g = x - n*ln(2).
	g := xClamped.Sub(nFloat.Mul(expF64Ln2Hi)).Sub(nFloat.Mul(expF64Ln2Lo))

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

	return scale.Mul(poly)
}

func tanh256Float64(x archsimd.Float64x4) archsimd.Float64x4 {
	xClamped := x.Max(tanhF64Neg19).Min(tanhF6419)
	twoX := xClamped.Mul(tanhF64Two)
	exp2x := exp256Float64(twoX)
	num := exp2x.Sub(tanhF64One)
	den := exp2x.Add(tanhF64One)
	return num.Div(den)
}

func erf256Float64(x archsimd.Float64x4) archsimd.Float64x4 {
	absX := x.Abs()
	t := erfF64One.Div(erfF64One.Add(erfF64P.Mul(absX)))

	poly := erfF64A5.MulAdd(t, erfF64A4)
	poly = poly.MulAdd(t, erfF64A3)
	poly = poly.MulAdd(t, erfF64A2)
	poly = poly.MulAdd(t, erfF64A1)
	poly = poly.Mul(t)

	expNegX2 := exp256Float64(absX.Mul(absX).Neg())
	res := erfF64One.Sub(poly.Mul(expNegX2))

	return res.ToBits().Xor(x.ToBits().And(erfF64SignMask)).BitsToFloat64()
}

func GeluExactAVX2Float64(in, out []float64) {
	i := 0
	for ; i+4 <= len(in); i += 4 {
		x := archsimd.LoadFloat64x4(in[i : i+4])
		scaledX := x.Mul(erfF64Rsqrt2)
		erfVal := erf256Float64(scaledX)
		res := erfF64Half.Mul(x).Mul(erfF64One.Add(erfVal))
		res.Store(out[i : i+4])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [4]float64
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat64x4(bufIn[:])
		scaledX := x.Mul(erfF64Rsqrt2)
		erfVal := erf256Float64(scaledX)
		res := erfF64Half.Mul(x).Mul(erfF64One.Add(erfVal))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

func GeluApproxAVX2Float64(in, out []float64) {
	i := 0
	for ; i+8 <= len(in); i += 8 {
		x0 := archsimd.LoadFloat64x4(in[i : i+4])
		x1 := archsimd.LoadFloat64x4(in[i+4 : i+8])
		x3_0 := x0.Mul(x0).Mul(x0)
		x3_1 := x1.Mul(x1).Mul(x1)
		inner0 := geluF64Sqrt2ByPi.Mul(x3_0.MulAdd(geluF64C, x0))
		inner1 := geluF64Sqrt2ByPi.Mul(x3_1.MulAdd(geluF64C, x1))
		t0 := tanh256Float64(inner0)
		t1 := tanh256Float64(inner1)
		res0 := geluF64Half.Mul(x0).Mul(geluF64One.Add(t0))
		res1 := geluF64Half.Mul(x1).Mul(geluF64One.Add(t1))
		res0.Store(out[i : i+4])
		res1.Store(out[i+4 : i+8])
	}
	for ; i+4 <= len(in); i += 4 {
		x := archsimd.LoadFloat64x4(in[i : i+4])
		x3 := x.Mul(x).Mul(x)
		inner := geluF64Sqrt2ByPi.Mul(x3.MulAdd(geluF64C, x))
		t := tanh256Float64(inner)
		res := geluF64Half.Mul(x).Mul(geluF64One.Add(t))
		res.Store(out[i : i+4])
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [4]float64
		copy(bufIn[:rem], in[i:])
		x := archsimd.LoadFloat64x4(bufIn[:])
		x3 := x.Mul(x).Mul(x)
		inner := geluF64Sqrt2ByPi.Mul(x3.MulAdd(geluF64C, x))
		t := tanh256Float64(inner)
		res := geluF64Half.Mul(x).Mul(geluF64One.Add(t))
		res.Store(bufOut[:])
		copy(out[i:], bufOut[:rem])
	}
}

// -----------------------------------------------------------------------------
// BFloat16 Vector GELU Operations (AVX2)
// -----------------------------------------------------------------------------

func GeluExactAVX2BF16(in, out []bfloat16.BFloat16) {
	if len(in) == 0 {
		return
	}
	i := 0
	for ; i+16 <= len(in); i += 16 {
		u32 := archsimd.LoadUint32x8(unsafe.Slice((*uint32)(unsafe.Pointer(&in[i])), 8))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16MaskHi).BitsToFloat32()

		scaledEven := even.Mul(erfRsqrt2)
		erfEven := erf256(scaledEven)
		evenRes := erfHalf.Mul(even).Mul(erfOne.Add(erfEven))

		scaledOdd := odd.Mul(erfRsqrt2)
		erfOdd := erf256(scaledOdd)
		oddRes := erfHalf.Mul(odd).Mul(erfOne.Add(erfOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16One))
		oddRounded := oddBits.Add(oddBias).And(bf16MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&out[i])), 8))
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [16]bfloat16.BFloat16
		copy(bufIn[:rem], in[i:])
		u32 := archsimd.LoadUint32x8(unsafe.Slice((*uint32)(unsafe.Pointer(&bufIn[0])), 8))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16MaskHi).BitsToFloat32()

		scaledEven := even.Mul(erfRsqrt2)
		erfEven := erf256(scaledEven)
		evenRes := erfHalf.Mul(even).Mul(erfOne.Add(erfEven))

		scaledOdd := odd.Mul(erfRsqrt2)
		erfOdd := erf256(scaledOdd)
		oddRes := erfHalf.Mul(odd).Mul(erfOne.Add(erfOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16One))
		oddRounded := oddBits.Add(oddBias).And(bf16MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&bufOut[0])), 8))
		copy(out[i:], bufOut[:rem])
	}
}

func GeluApproxAVX2BF16(in, out []bfloat16.BFloat16) {
	if len(in) == 0 {
		return
	}
	i := 0
	for ; i+16 <= len(in); i += 16 {
		u32 := archsimd.LoadUint32x8(unsafe.Slice((*uint32)(unsafe.Pointer(&in[i])), 8))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16MaskHi).BitsToFloat32()

		x3Even := even.Mul(even).Mul(even)
		innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, even))
		tEven := tanh256(innerEven)
		evenRes := geluHalf.Mul(even).Mul(geluOne.Add(tEven))

		x3Odd := odd.Mul(odd).Mul(odd)
		innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, odd))
		tOdd := tanh256(innerOdd)
		oddRes := geluHalf.Mul(odd).Mul(geluOne.Add(tOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16One))
		oddRounded := oddBits.Add(oddBias).And(bf16MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&out[i])), 8))
	}
	if i < len(in) {
		rem := len(in) - i
		var bufIn, bufOut [16]bfloat16.BFloat16
		copy(bufIn[:rem], in[i:])
		u32 := archsimd.LoadUint32x8(unsafe.Slice((*uint32)(unsafe.Pointer(&bufIn[0])), 8))
		even := u32.ShiftAllLeft(16).BitsToFloat32()
		odd := u32.And(bf16MaskHi).BitsToFloat32()

		x3Even := even.Mul(even).Mul(even)
		innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, even))
		tEven := tanh256(innerEven)
		evenRes := geluHalf.Mul(even).Mul(geluOne.Add(tEven))

		x3Odd := odd.Mul(odd).Mul(odd)
		innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, odd))
		tOdd := tanh256(innerOdd)
		oddRes := geluHalf.Mul(odd).Mul(geluOne.Add(tOdd))

		evenBits := evenRes.ToBits()
		evenBias := bf16Bias7FFF.Add(evenBits.ShiftAllRight(16).And(bf16One))
		evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

		oddBits := oddRes.ToBits()
		oddBias := bf16Bias7FFF.Add(oddBits.ShiftAllRight(16).And(bf16One))
		oddRounded := oddBits.Add(oddBias).And(bf16MaskHi)

		packed := evenRounded.Or(oddRounded)
		packed.Store(unsafe.Slice((*uint32)(unsafe.Pointer(&bufOut[0])), 8))
		copy(out[i:], bufOut[:rem])
	}
}

// -----------------------------------------------------------------------------
// Float16 Vector GELU Operations (AVX2)
// -----------------------------------------------------------------------------

func GeluExactAVX2F16(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := float16.ToFloat32SIMD(v)
		evenArch := *(*archsimd.Float32x8)(unsafe.Pointer(&even))
		oddArch := *(*archsimd.Float32x8)(unsafe.Pointer(&odd))

		scaledEven := evenArch.Mul(erfRsqrt2)
		erfEven := erf256(scaledEven)
		resEven := erfHalf.Mul(evenArch).Mul(erfOne.Add(erfEven))
		resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

		scaledOdd := oddArch.Mul(erfRsqrt2)
		erfOdd := erf256(scaledOdd)
		resOdd := erfHalf.Mul(oddArch).Mul(erfOne.Add(erfOdd))
		resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

		res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
		float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := float16.ToFloat32SIMD(v)
		evenArch := *(*archsimd.Float32x8)(unsafe.Pointer(&even))
		oddArch := *(*archsimd.Float32x8)(unsafe.Pointer(&odd))

		scaledEven := evenArch.Mul(erfRsqrt2)
		erfEven := erf256(scaledEven)
		resEven := erfHalf.Mul(evenArch).Mul(erfOne.Add(erfEven))
		resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

		scaledOdd := oddArch.Mul(erfRsqrt2)
		erfOdd := erf256(scaledOdd)
		resOdd := erfHalf.Mul(oddArch).Mul(erfOne.Add(erfOdd))
		resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

		res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
		float16.StoreFloat16sPart(res, out[i:])
	}
}

func GeluApproxAVX2F16(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := float16.ToFloat32SIMD(v)
		evenArch := *(*archsimd.Float32x8)(unsafe.Pointer(&even))
		oddArch := *(*archsimd.Float32x8)(unsafe.Pointer(&odd))

		x3Even := evenArch.Mul(evenArch).Mul(evenArch)
		innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, evenArch))
		tEven := tanh256(innerEven)
		resEven := geluHalf.Mul(evenArch).Mul(geluOne.Add(tEven))
		resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

		x3Odd := oddArch.Mul(oddArch).Mul(oddArch)
		innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, oddArch))
		tOdd := tanh256(innerOdd)
		resOdd := geluHalf.Mul(oddArch).Mul(geluOne.Add(tOdd))
		resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

		res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
		float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := float16.ToFloat32SIMD(v)
		evenArch := *(*archsimd.Float32x8)(unsafe.Pointer(&even))
		oddArch := *(*archsimd.Float32x8)(unsafe.Pointer(&odd))

		x3Even := evenArch.Mul(evenArch).Mul(evenArch)
		innerEven := geluSqrt2ByPi.Mul(x3Even.MulAdd(geluC, evenArch))
		tEven := tanh256(innerEven)
		resEven := geluHalf.Mul(evenArch).Mul(geluOne.Add(tEven))
		resEvenSimd := *(*simd.Float32s)(unsafe.Pointer(&resEven))

		x3Odd := oddArch.Mul(oddArch).Mul(oddArch)
		innerOdd := geluSqrt2ByPi.Mul(x3Odd.MulAdd(geluC, oddArch))
		tOdd := tanh256(innerOdd)
		resOdd := geluHalf.Mul(oddArch).Mul(geluOne.Add(tOdd))
		resOddSimd := *(*simd.Float32s)(unsafe.Pointer(&resOdd))

		res := float16.FromFloat32SIMD(resEvenSimd, resOddSimd)
		float16.StoreFloat16sPart(res, out[i:])
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"math"
	"simd/archsimd"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
)

const PriorityAVX512 = gobackend.PriorityArch + 1

func init() {
	if gobackend.IsAVX512Allowed {
		registerAVX512()
	}
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
	activations.Register[float32]("avx512:geluapprox", compute.ActivationGeluApproximate, GeluAVX512, PriorityAVX512)
	activations.RegisterSwiGLU[float32]("avx512:swiglu", SwiGLUAVX512, PriorityAVX512)

	// BFloat16
	activations.Register[bfloat16.BFloat16]("avx512:relu", compute.ActivationRelu, reluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:sigmoid", compute.ActivationSigmoid, sigmoidBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:hardsigmoid", compute.ActivationHardSigmoid, hardSigmoidBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:leakyrelu", compute.ActivationLeakyRelu, leakyReluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:selu", compute.ActivationSelu, seluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:silu", compute.ActivationSilu, siluBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:hardswish", compute.ActivationHardSwish, hardSwishBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:tanh", compute.ActivationTanh, tanhBF16AVX512, PriorityAVX512)
	activations.Register[bfloat16.BFloat16]("avx512:geluapprox", compute.ActivationGeluApproximate, geluBF16AVX512, PriorityAVX512)

	// Float16
	activations.Register[float16.Float16]("avx512:relu", compute.ActivationRelu, reluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:sigmoid", compute.ActivationSigmoid, sigmoidF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:hardsigmoid", compute.ActivationHardSigmoid, hardSigmoidF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:leakyrelu", compute.ActivationLeakyRelu, leakyReluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:selu", compute.ActivationSelu, seluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:silu", compute.ActivationSilu, siluF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:hardswish", compute.ActivationHardSwish, hardSwishF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:tanh", compute.ActivationTanh, tanhF16AVX512, PriorityAVX512)
	activations.Register[float16.Float16]("avx512:geluapprox", compute.ActivationGeluApproximate, geluF16AVX512, PriorityAVX512)
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
	const (
		maxLogF = 88.02969187150841
		minLogF = -88.02969187150841
		log2E   = 1.44269504088896341
		ln2Hi   = 0.693359375
		ln2Lo   = -2.12194440e-4

		p7 = 1.9875691500e-4
		p6 = 1.3981999507e-3
		p5 = 8.3334519073e-3
		p4 = 4.1665795894e-2
		p3 = 1.6666665459e-1
		p2 = 5.0000001201e-1
	)
	vMaxLog := archsimd.BroadcastFloat32x16(maxLogF)
	vMinLog := archsimd.BroadcastFloat32x16(minLogF)
	vLog2E := archsimd.BroadcastFloat32x16(log2E)
	vHalf := archsimd.BroadcastFloat32x16(0.5)
	vLn2Hi := archsimd.BroadcastFloat32x16(ln2Hi)
	vLn2Lo := archsimd.BroadcastFloat32x16(ln2Lo)

	vP7 := archsimd.BroadcastFloat32x16(p7)
	vP6 := archsimd.BroadcastFloat32x16(p6)
	vP5 := archsimd.BroadcastFloat32x16(p5)
	vP4 := archsimd.BroadcastFloat32x16(p4)
	vP3 := archsimd.BroadcastFloat32x16(p3)
	vP2 := archsimd.BroadcastFloat32x16(p2)
	vOne := archsimd.BroadcastFloat32x16(1.0)
	v127 := archsimd.BroadcastUint32x16(127)

	xClamped := x.Max(vMinLog).Min(vMaxLog)
	z := xClamped.MulAdd(vLog2E, vHalf).RoundScaled(0)
	g := xClamped.Sub(z.Mul(vLn2Hi)).Sub(z.Mul(vLn2Lo))

	n := z.ConvertToInt32().AsUint32x16().Add(v127).ShiftAllLeft(23).BitsToFloat32()

	poly := vP7.MulAdd(g, vP6)
	poly = poly.MulAdd(g, vP5)
	poly = poly.MulAdd(g, vP4)
	poly = poly.MulAdd(g, vP3)
	poly = poly.MulAdd(g, vP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(vOne)

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
	vNine := archsimd.BroadcastFloat32x16(9.0)
	vNegNine := archsimd.BroadcastFloat32x16(-9.0)
	xClamped := x.Max(vNegNine).Min(vNine)

	vTwo := archsimd.BroadcastFloat32x16(2.0)
	vOne := archsimd.BroadcastFloat32x16(1.0)
	twoX := xClamped.Mul(vTwo)
	exp2x := exp512(twoX)
	num := exp2x.Sub(vOne)
	den := exp2x.Add(vOne)
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

func GeluAVX512(data []float32) {
	vHalf := archsimd.BroadcastFloat32x16(0.5)
	vOne := archsimd.BroadcastFloat32x16(1.0)
	vSqrt2ByPi := archsimd.BroadcastFloat32x16(float32(math.Sqrt(2.0 / math.Pi)))
	vC := archsimd.BroadcastFloat32x16(0.044715)

	i := 0
	for ; i+16 <= len(data); i += 16 {
		x := archsimd.LoadFloat32x16(data[i : i+16])
		x3 := x.Mul(x).Mul(x)
		inner := vSqrt2ByPi.Mul(x3.MulAdd(vC, x))
		t := tanh512(inner)
		res := vHalf.Mul(x).Mul(vOne.Add(t))
		res.Store(data[i : i+16])
	}
	sqrt2ByPi := float32(math.Sqrt(2.0 / math.Pi))
	for ; i < len(data); i++ {
		x := data[i]
		inner := sqrt2ByPi * (x + 0.044715*x*x*x)
		data[i] = x * 0.5 * (1.0 + float32(math.Tanh(float64(inner))))
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

func geluBF16AVX512(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		GeluAVX512(buf[:len(chunk)])
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

func geluF16AVX512(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		GeluAVX512(buf[:len(chunk)])
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

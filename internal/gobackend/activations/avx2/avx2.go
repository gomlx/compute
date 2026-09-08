// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"simd/archsimd"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
)

const PriorityAVX2 = gobackend.PriorityArch

func init() {
	if gobackend.IsAVX2Allowed() {
		registerAVX2()
	}
}

func registerAVX2() {
// Float32
	activations.Register[float32]("avx2:relu", compute.ActivationRelu, ReluAVX2, PriorityAVX2)
	activations.Register[float32]("avx2:hardswish", compute.ActivationHardSwish, HardSwishAVX2, PriorityAVX2)
	activations.Register[float32]("avx2:silu", compute.ActivationSilu, SiluAVX2, PriorityAVX2)
	activations.Register[float32]("avx2:geluapprox", compute.ActivationGeluApproximate, GeluAVX2, PriorityAVX2)
	activations.Register[float32]("avx2:tanh", compute.ActivationTanh, TanhAVX2, PriorityAVX2)

	// BFloat16
	activations.Register[bfloat16.BFloat16]("avx2:relu", compute.ActivationRelu, reluBF16AVX2, PriorityAVX2)
	activations.Register[bfloat16.BFloat16]("avx2:hardswish", compute.ActivationHardSwish, hardSwishBF16AVX2, PriorityAVX2)
	activations.Register[bfloat16.BFloat16]("avx2:silu", compute.ActivationSilu, siluBF16AVX2, PriorityAVX2)
	activations.Register[bfloat16.BFloat16]("avx2:geluapprox", compute.ActivationGeluApproximate, geluBF16AVX2, PriorityAVX2)
	activations.Register[bfloat16.BFloat16]("avx2:tanh", compute.ActivationTanh, tanhBF16AVX2, PriorityAVX2)

	// Float16
	activations.Register[float16.Float16]("avx2:relu", compute.ActivationRelu, reluF16AVX2, PriorityAVX2)
	activations.Register[float16.Float16]("avx2:hardswish", compute.ActivationHardSwish, hardSwishF16AVX2, PriorityAVX2)
	activations.Register[float16.Float16]("avx2:silu", compute.ActivationSilu, siluF16AVX2, PriorityAVX2)
	activations.Register[float16.Float16]("avx2:geluapprox", compute.ActivationGeluApproximate, geluF16AVX2, PriorityAVX2)
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
	vMaxLog := archsimd.BroadcastFloat32x8(maxLogF)
	vMinLog := archsimd.BroadcastFloat32x8(minLogF)
	vLog2E := archsimd.BroadcastFloat32x8(log2E)
	vHalf := archsimd.BroadcastFloat32x8(0.5)
	vLn2Hi := archsimd.BroadcastFloat32x8(ln2Hi)
	vLn2Lo := archsimd.BroadcastFloat32x8(ln2Lo)

	vP7 := archsimd.BroadcastFloat32x8(p7)
	vP6 := archsimd.BroadcastFloat32x8(p6)
	vP5 := archsimd.BroadcastFloat32x8(p5)
	vP4 := archsimd.BroadcastFloat32x8(p4)
	vP3 := archsimd.BroadcastFloat32x8(p3)
	vP2 := archsimd.BroadcastFloat32x8(p2)
	vOne := archsimd.BroadcastFloat32x8(1.0)
	v127 := archsimd.BroadcastUint32x8(127)

	xClamped := x.Max(vMinLog).Min(vMaxLog)
	z := xClamped.MulAdd(vLog2E, vHalf).Floor()
	g := xClamped.Sub(z.Mul(vLn2Hi)).Sub(z.Mul(vLn2Lo))

	n := z.ConvertToInt32().AsUint32x8().Add(v127).ShiftAllLeft(23).BitsToFloat32()

	poly := vP7.MulAdd(g, vP6)
	poly = poly.MulAdd(g, vP5)
	poly = poly.MulAdd(g, vP4)
	poly = poly.MulAdd(g, vP3)
	poly = poly.MulAdd(g, vP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(vOne)

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
	vNine := archsimd.BroadcastFloat32x8(9.0)
	vNegNine := archsimd.BroadcastFloat32x8(-9.0)
	xClamped := x.Max(vNegNine).Min(vNine)

	vTwo := archsimd.BroadcastFloat32x8(2.0)
	vOne := archsimd.BroadcastFloat32x8(1.0)
	twoX := xClamped.Mul(vTwo)
	exp2x := exp256(twoX)
	num := exp2x.Sub(vOne)
	den := exp2x.Add(vOne)
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

func GeluAVX2(data []float32) {
	// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	vHalf := archsimd.BroadcastFloat32x8(0.5)
	vOne := archsimd.BroadcastFloat32x8(1.0)
	vSqrt2ByPi := archsimd.BroadcastFloat32x8(float32(math.Sqrt(2.0 / math.Pi)))
	vC := archsimd.BroadcastFloat32x8(0.044715)

	i := 0
	for ; i+8 <= len(data); i += 8 {
		x := archsimd.LoadFloat32x8(data[i : i+8])
		x3 := x.Mul(x).Mul(x)
		inner := vSqrt2ByPi.Mul(x3.MulAdd(vC, x))
		t := tanh256(inner)
		res := vHalf.Mul(x).Mul(vOne.Add(t))
		res.Store(data[i : i+8])
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

func geluBF16AVX2(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		GeluAVX2(buf[:len(chunk)])
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

func geluF16AVX2(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		end := min(i+halfChunk, len(data))
		chunk := data[i:end]
		for j, v := range chunk {
			buf[j] = v.Float32()
		}
		GeluAVX2(buf[:len(chunk)])
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

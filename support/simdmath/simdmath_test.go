// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package simdmath_test

import (
	"math"
	"simd"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/support/simdmath"
)

func TestFloat32Math(t *testing.T) {
	vLen := simd.BroadcastFloat32s(0).Len()
	in := make([]float32, vLen)
	for i := range in {
		in[i] = float32(i)*0.5 - 2.0
	}
	v := simd.LoadFloat32s(in)

	// Exp
	vExp := simdmath.ExpFloat32(v)
	outExp := make([]float32, vLen)
	vExp.Store(outExp)
	for i, x := range in {
		expected := float32(math.Exp(float64(x)))
		delta := float32(math.Abs(float64(outExp[i] - expected)))
		relDelta := delta / float32(math.Abs(float64(expected)))
		if relDelta > 1e-4 && delta > 1e-4 {
			t.Errorf("ExpFloat32(%v) = %v, expected %v (relDelta: %v)", x, outExp[i], expected, relDelta)
		}
	}

	// Tanh
	vTanh := simdmath.TanhFloat32(v)
	outTanh := make([]float32, vLen)
	vTanh.Store(outTanh)
	for i, x := range in {
		expected := float32(math.Tanh(float64(x)))
		delta := float32(math.Abs(float64(outTanh[i] - expected)))
		if delta > 1e-3 {
			t.Errorf("TanhFloat32(%v) = %v, expected %v (delta: %v)", x, outTanh[i], expected, delta)
		}
	}

	// Erf
	vErf := simdmath.ErfFloat32(v)
	outErf := make([]float32, vLen)
	vErf.Store(outErf)
	for i, x := range in {
		expected := float32(math.Erf(float64(x)))
		delta := float32(math.Abs(float64(outErf[i] - expected)))
		if delta > 1e-4 {
			t.Errorf("ErfFloat32(%v) = %v, expected %v (delta: %v)", x, outErf[i], expected, delta)
		}
	}

	// Sqrt & Inv
	posIn := make([]float32, vLen)
	for i := range posIn {
		posIn[i] = float32(i) + 1.0
	}
	vPos := simd.LoadFloat32s(posIn)
	vSqrt := simdmath.SqrtFloat32(vPos)
	outSqrt := make([]float32, vLen)
	vSqrt.Store(outSqrt)
	for i, x := range posIn {
		expected := float32(math.Sqrt(float64(x)))
		if math.Abs(float64(outSqrt[i]-expected)) > 1e-5 {
			t.Errorf("SqrtFloat32(%v) = %v, expected %v", x, outSqrt[i], expected)
		}
	}

	vInv := simdmath.InvFloat32(vPos)
	outInv := make([]float32, vLen)
	vInv.Store(outInv)
	for i, x := range posIn {
		expected := 1.0 / x
		if math.Abs(float64(outInv[i]-expected)) > 1e-5 {
			t.Errorf("InvFloat32(%v) = %v, expected %v", x, outInv[i], expected)
		}
	}

	// Edge cases for ExpFloat32 and TanhFloat32: near-overflow (88.03..88.72), subnormals, ±Inf, NaN.
	inf32 := float32(math.Inf(1))
	nan32 := float32(math.NaN())
	edgeIn32 := []float32{-inf32, -104.0, -95.0, 88.3, 88.72, 89.0, inf32, nan32}
	vEdge32, nLoaded32 := simd.LoadFloat32sPart(edgeIn32)
	outEdgeExp32 := make([]float32, nLoaded32)
	simdmath.ExpFloat32(vEdge32).StorePart(outEdgeExp32)
	for i, x := range edgeIn32[:nLoaded32] {
		want := float32(math.Exp(float64(x)))
		got := outEdgeExp32[i]
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("ExpFloat32(%v) = %v, expected NaN", x, got)
			}
			continue
		}
		if math.IsInf(float64(want), 0) {
			if got != want {
				t.Errorf("ExpFloat32(%v) = %v, expected %v", x, got, want)
			}
			continue
		}
		delta := math.Abs(float64(got - want))
		relDelta := 0.0
		if want != 0 {
			relDelta = delta / math.Abs(float64(want))
		}
		if relDelta > 1e-4 && delta > 1e-42 {
			t.Errorf("ExpFloat32(%v) = %v, expected %v (relDelta: %v)", x, got, want, relDelta)
		}
	}
	outEdgeTanh32 := make([]float32, nLoaded32)
	simdmath.TanhFloat32(vEdge32).StorePart(outEdgeTanh32)
	if !math.IsNaN(float64(outEdgeTanh32[nLoaded32-1])) {
		t.Errorf("TanhFloat32(NaN) = %v, expected NaN", outEdgeTanh32[nLoaded32-1])
	}
}

func TestFloat64Math(t *testing.T) {
	vLen := simd.BroadcastFloat64s(0).Len()
	in := make([]float64, vLen)
	for i := range in {
		in[i] = float64(i)*0.5 - 2.0
	}
	v := simd.LoadFloat64s(in)

	// Exp
	vExp := simdmath.ExpFloat64(v)
	outExp := make([]float64, vLen)
	vExp.Store(outExp)
	for i, x := range in {
		expected := math.Exp(x)
		delta := math.Abs(outExp[i] - expected)
		relDelta := delta / math.Abs(expected)
		if relDelta > 1e-5 && delta > 1e-5 {
			t.Errorf("ExpFloat64(%v) = %v, expected %v (relDelta: %v)", x, outExp[i], expected, relDelta)
		}
	}

	// Tanh
	vTanh := simdmath.TanhFloat64(v)
	outTanh := make([]float64, vLen)
	vTanh.Store(outTanh)
	for i, x := range in {
		expected := math.Tanh(x)
		delta := math.Abs(outTanh[i] - expected)
		if delta > 1e-4 {
			t.Errorf("TanhFloat64(%v) = %v, expected %v (delta: %v)", x, outTanh[i], expected, delta)
		}
	}

	// Erf
	vErf := simdmath.ErfFloat64(v)
	outErf := make([]float64, vLen)
	vErf.Store(outErf)
	for i, x := range in {
		expected := math.Erf(x)
		delta := math.Abs(outErf[i] - expected)
		if delta > 1e-4 {
			t.Errorf("ErfFloat64(%v) = %v, expected %v (delta: %v)", x, outErf[i], expected, delta)
		}
	}

	// Edge cases for ExpFloat64 and TanhFloat64: near-overflow (709.4..709.78), subnormals, ±Inf, NaN.
	edgeIn64 := []float64{math.Inf(-1), -746.0, -720.0, 709.4, 709.78, 710.0, math.Inf(1), math.NaN()}
	for _, x := range edgeIn64 {
		vX := simd.BroadcastFloat64s(x)
		out := make([]float64, vLen)
		simdmath.ExpFloat64(vX).Store(out)
		got := out[0]
		want := math.Exp(x)
		if x > 709.4 && x < 709.7827 {
			// Go's amd64 math.Exp assembly overflows early above 1023.5*ln(2) ≈ 709.436;
			// compute the reference without overflowing math.Exp.
			want = math.Exp(x-math.Ln2) * 2
		}
		if math.IsNaN(want) {
			if !math.IsNaN(got) {
				t.Errorf("ExpFloat64(%v) = %v, expected NaN", x, got)
			}
			continue
		}
		if math.IsInf(want, 0) {
			if got != want {
				t.Errorf("ExpFloat64(%v) = %v, expected %v", x, got, want)
			}
			continue
		}
		delta := math.Abs(got - want)
		relDelta := 0.0
		if want != 0 {
			relDelta = delta / math.Abs(want)
		}
		if relDelta > 1e-5 && delta > 1e-315 {
			t.Errorf("ExpFloat64(%v) = %v, expected %v (relDelta: %v)", x, got, want, relDelta)
		}
	}
	outTanhNaN64 := make([]float64, vLen)
	simdmath.TanhFloat64(simd.BroadcastFloat64s(math.NaN())).Store(outTanhNaN64)
	if !math.IsNaN(outTanhNaN64[0]) {
		t.Errorf("TanhFloat64(NaN) = %v, expected NaN", outTanhNaN64[0])
	}
}

func TestBF16F16Adapters(t *testing.T) {
	vLen := simd.BroadcastFloat32s(0).Len()
	bfIn := make([]bfloat16.BFloat16, vLen)
	f16In := make([]float16.Float16, vLen)
	for i := range bfIn {
		v := float32(i) * 0.25
		bfIn[i] = bfloat16.FromFloat32(v)
		f16In[i] = float16.FromFloat32(v)
	}

	vBF := simdmath.LoadBFloat16s(bfIn)
	bfOut := make([]bfloat16.BFloat16, vLen)
	simdmath.StoreBFloat16s(bfOut, vBF)
	for i := range bfIn {
		if bfOut[i] != bfIn[i] {
			t.Errorf("BFloat16 roundtrip at %d: got %v, expected %v", i, bfOut[i], bfIn[i])
		}
	}

	vF16 := simdmath.LoadFloat16s(f16In)
	f16Out := make([]float16.Float16, vLen)
	simdmath.StoreFloat16s(f16Out, vF16)
	for i := range f16In {
		if f16Out[i] != f16In[i] {
			t.Errorf("Float16 roundtrip at %d: got %v, expected %v", i, f16Out[i], f16In[i])
		}
	}
}

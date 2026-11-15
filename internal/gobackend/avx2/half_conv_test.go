// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

package avx2

import (
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func TestHalfSliceConvertersAVX2(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this machine")
	}

	vals := []float32{
		0, float32(math.Copysign(0, -1)), 1.0, -1.0, 0.5, -0.5, 3.1415927, -2.7182817,
		1e-7, -1e-7, 6e-5, -6e-5, 65504.0, -65504.0, 70000.0, -70000.0,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
		1.00048828125, 1.00146484375,
		math.Float32frombits(0x3F808000), math.Float32frombits(0x3F818000),
	}
	for len(vals) < 79 {
		vals = append(vals, float32(math.Sin(float64(len(vals)))*10.0))
	}

	f16 := make([]float16.Float16, len(vals))
	avx2Float32ToFloat16Slice(f16, vals)
	for i, v := range vals {
		want := float16.FromFloat32(v)
		if math.IsNaN(float64(v)) {
			if !math.IsNaN(float64(f16[i].Float32())) {
				t.Errorf("F32->F16[%d]: expected NaN, got 0x%04x", i, uint16(f16[i]))
			}
		} else if f16[i] != want {
			t.Errorf("F32->F16[%d] input=%v: got 0x%04x, want 0x%04x", i, v, uint16(f16[i]), uint16(want))
		}
	}

	f32FromF16 := make([]float32, len(f16))
	avx2Float16ToFloat32Slice(f32FromF16, f16)
	for i, h := range f16 {
		want := h.Float32()
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(f32FromF16[i])) {
				t.Errorf("F16->F32[%d]: expected NaN, got %v", i, f32FromF16[i])
			}
		} else if math.Float32bits(f32FromF16[i]) != math.Float32bits(want) {
			t.Errorf("F16->F32[%d] input=0x%04x: got 0x%08x, want 0x%08x", i, uint16(h), math.Float32bits(f32FromF16[i]), math.Float32bits(want))
		}
	}

	bf16 := make([]bfloat16.BFloat16, len(vals))
	avx2Float32ToBFloat16Slice(bf16, vals)
	for i, v := range vals {
		want := bfloat16.FromFloat32(v)
		if math.IsNaN(float64(v)) {
			if !math.IsNaN(float64(bf16[i].Float32())) {
				t.Errorf("F32->BF16[%d]: expected NaN, got 0x%04x", i, uint16(bf16[i]))
			}
		} else if bf16[i] != want {
			t.Errorf("F32->BF16[%d] input=%v: got 0x%04x, want 0x%04x", i, v, uint16(bf16[i]), uint16(want))
		}
	}

	f32FromBF16 := make([]float32, len(bf16))
	avx2BFloat16ToFloat32Slice(f32FromBF16, bf16)
	for i, h := range bf16 {
		want := h.Float32()
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(f32FromBF16[i])) {
				t.Errorf("BF16->F32[%d]: expected NaN, got %v", i, f32FromBF16[i])
			}
		} else if math.Float32bits(f32FromBF16[i]) != math.Float32bits(want) {
			t.Errorf("BF16->F32[%d] input=0x%04x: got 0x%08x, want 0x%08x", i, uint16(h), math.Float32bits(f32FromBF16[i]), math.Float32bits(want))
		}
	}
}

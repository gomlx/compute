// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

package neon

import (
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func TestHalfSliceConvertersNEON(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
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
	float16.FromFloat32SliceNEON(f16, vals)
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
	float16.ToFloat32SliceNEON(f32FromF16, f16)
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
	bfloat16.FromFloat32SliceNEON(bf16, vals)
	for i, v := range vals {
		want := bfloat16.FromFloat32(v)
		if math.IsNaN(float64(v)) {
			if !math.IsNaN(float64(bf16[i].Float32())) {
				t.Errorf("F32->BF16_NEON[%d]: expected NaN, got 0x%04x", i, uint16(bf16[i]))
			}
		} else if bf16[i] != want {
			t.Errorf("F32->BF16_NEON[%d] input=%v: got 0x%04x, want 0x%04x", i, v, uint16(bf16[i]), uint16(want))
		}
	}

	if gobackend.IsARM64BF16Allowed {
		bf16HW := make([]bfloat16.BFloat16, len(vals))
		neonBF16Float32ToBFloat16Slice(bf16HW, vals)
		for i, v := range vals {
			want := bfloat16.FromFloat32(v)
			if math.IsNaN(float64(v)) {
				if !math.IsNaN(float64(bf16HW[i].Float32())) {
					t.Errorf("F32->BF16_HW[%d]: expected NaN, got 0x%04x", i, uint16(bf16HW[i]))
				}
			} else if bf16HW[i] != want {
				t.Errorf("F32->BF16_HW[%d] input=%v: got 0x%04x, want 0x%04x", i, v, uint16(bf16HW[i]), uint16(want))
			}
		}
	}

	f32FromBF16 := make([]float32, len(bf16))
	bfloat16.ToFloat32SliceNEON(f32FromBF16, bf16)
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

func BenchmarkHalfSliceConvertersNEON(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}
	const n = 4096
	srcF32 := make([]float32, n)
	bufF16 := make([]float16.Float16, n)
	bufBF16 := make([]bfloat16.BFloat16, n)
	dstF32 := make([]float32, n)
	for i := range srcF32 {
		srcF32[i] = float32(i)*0.01 - 20.0
		bufF16[i] = float16.FromFloat32(srcF32[i])
		bufBF16[i] = bfloat16.FromFloat32(srcF32[i])
	}

	b.Run("F16ToF32_NEON", func(b *testing.B) {
		for b.Loop() {
			float16.ToFloat32SliceNEON(dstF32, bufF16)
		}
	})
	b.Run("F32ToF16_NEON", func(b *testing.B) {
		for b.Loop() {
			float16.FromFloat32SliceNEON(bufF16, srcF32)
		}
	})
	b.Run("BF16ToF32_NEON", func(b *testing.B) {
		for b.Loop() {
			bfloat16.ToFloat32SliceNEON(dstF32, bufBF16)
		}
	})
	b.Run("F32ToBF16_NEON_Soft", func(b *testing.B) {
		for b.Loop() {
			bfloat16.FromFloat32SliceNEON(bufBF16, srcF32)
		}
	})
	if gobackend.IsARM64BF16Allowed {
		b.Run("F32ToBF16_ARM64BF16_HW", func(b *testing.B) {
			for b.Loop() {
				neonBF16Float32ToBFloat16Slice(bufBF16, srcF32)
			}
		})
	}
}

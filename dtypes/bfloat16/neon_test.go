// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package bfloat16

import (
	"math"
	"testing"
)

func TestBFloat16NEON(t *testing.T) {
	var testFloats []float32
	for x := -10.0; x <= 10.0; x += 0.25 {
		testFloats = append(testFloats, float32(x))
	}
	testFloats = append(testFloats,
		0, float32(math.Copysign(0, -1)),
		float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()),
		1.2345, -7.891, 65504, -65504,
		SmallestNonzero.Float32(), -SmallestNonzero.Float32(),
	)
	for len(testFloats)%8 != 0 {
		testFloats = append(testFloats, 0)
	}

	inBF16 := make([]BFloat16, len(testFloats))
	for i, f := range testFloats {
		inBF16[i] = FromFloat32(f)
	}

	t.Run("BFloat16x8_Inline", func(t *testing.T) {
		outBF16 := make([]BFloat16, len(inBF16))
		outF32 := make([]float32, len(inBF16))
		for i := 0; i < len(inBF16); i += 8 {
			v := LoadBFloat16x8((*[8]BFloat16)(inBF16[i : i+8]))
			lo, hi := v.ToFloat32()
			lo.StoreArray((*[4]float32)(outF32[i : i+4]))
			hi.StoreArray((*[4]float32)(outF32[i+4 : i+8]))
			repacked := FromFloat32x4(lo, hi)
			repacked.Store((*[8]BFloat16)(outBF16[i : i+8]))
		}
		for i, want := range inBF16 {
			wantF32 := want.Float32()
			if math.IsNaN(float64(wantF32)) {
				if !math.IsNaN(float64(outF32[i])) || !math.IsNaN(float64(outBF16[i].Float32())) {
					t.Fatalf("index %d: expected NaN, got f32=%v bf16=0x%04x", i, outF32[i], outBF16[i])
				}
				continue
			}
			if outF32[i] != wantF32 {
				t.Fatalf("ToFloat32 mismatch at %d (input 0x%04x): got %v, want %v", i, want.Bits(), outF32[i], wantF32)
			}
			if outBF16[i] != want {
				t.Fatalf("FromFloat32x4 mismatch at %d: got 0x%04x (%v), want 0x%04x (%v)", i, outBF16[i].Bits(), outBF16[i], want.Bits(), want)
			}
		}
	})

	t.Run("SliceNEON", func(t *testing.T) {
		inWithTail := append([]BFloat16(nil), inBF16...)
		inWithTail = append(inWithTail, FromFloat32(3.5), FromFloat32(-0.125), SmallestNonzero)
		outF32 := make([]float32, len(inWithTail))
		outBF16 := make([]BFloat16, len(inWithTail))

		ToFloat32SliceNEON(outF32, inWithTail)
		FromFloat32SliceNEON(outBF16, outF32)

		for i, want := range inWithTail {
			wantF32 := want.Float32()
			if math.IsNaN(float64(wantF32)) {
				if !math.IsNaN(float64(outF32[i])) || !math.IsNaN(float64(outBF16[i].Float32())) {
					t.Fatalf("index %d: expected NaN, got f32=%v bf16=0x%04x", i, outF32[i], outBF16[i])
				}
				continue
			}
			if outF32[i] != wantF32 {
				t.Fatalf("ToFloat32SliceNEON mismatch at %d: got %v, want %v", i, outF32[i], wantF32)
			}
			if outBF16[i] != want {
				t.Fatalf("FromFloat32SliceNEON mismatch at %d: got 0x%04x, want 0x%04x", i, outBF16[i].Bits(), want.Bits())
			}
		}
	})
}

func BenchmarkBFloat16Conversions(b *testing.B) {
	const n = 4096
	srcBF16 := make([]BFloat16, n)
	dstF32 := make([]float32, n)
	dstBF16 := make([]BFloat16, n)
	for i := range srcBF16 {
		srcBF16[i] = FromFloat32(float32(i)*0.01 - 20.0)
	}

	b.Run("Scalar_ToF32", func(b *testing.B) {
		for b.Loop() {
			for i := range n {
				dstF32[i] = srcBF16[i].Float32()
			}
		}
	})
	b.Run("NEON_ToF32", func(b *testing.B) {
		for b.Loop() {
			ToFloat32SliceNEON(dstF32, srcBF16)
		}
	})
	b.Run("Scalar_FromF32", func(b *testing.B) {
		for b.Loop() {
			for i := range n {
				dstBF16[i] = FromFloat32(dstF32[i])
			}
		}
	})
	b.Run("NEON_FromF32", func(b *testing.B) {
		for b.Loop() {
			FromFloat32SliceNEON(dstBF16, dstF32)
		}
	})
}

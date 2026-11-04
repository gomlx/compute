// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package float16

import (
	"math"
	"testing"
)

func TestFloat16NEON(t *testing.T) {
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
		FromBits(0x0002).Float32(), FromBits(0x03FF).Float32(), FromBits(0x83FF).Float32(),
	)
	for len(testFloats)%8 != 0 {
		testFloats = append(testFloats, 0)
	}

	inF16 := make([]Float16, len(testFloats))
	for i, f := range testFloats {
		inF16[i] = FromFloat32(f)
	}

	t.Run("Float16x8_Inline", func(t *testing.T) {
		outF16 := make([]Float16, len(inF16))
		outF32 := make([]float32, len(inF16))
		for i := 0; i < len(inF16); i += 8 {
			v := LoadFloat16x8((*[8]Float16)(inF16[i : i+8]))
			lo, hi := v.ToFloat32()
			lo.Store((*[4]float32)(outF32[i : i+4])[:])
			hi.Store((*[4]float32)(outF32[i+4 : i+8])[:])
			repacked := FromFloat32x4(lo, hi)
			repacked.Store((*[8]Float16)(outF16[i : i+8]))
		}
		for i, want := range inF16 {
			wantF32 := want.Float32()
			if math.IsNaN(float64(wantF32)) {
				if !math.IsNaN(float64(outF32[i])) || !math.IsNaN(float64(outF16[i].Float32())) {
					t.Fatalf("index %d: expected NaN, got f32=%v f16=0x%04x", i, outF32[i], outF16[i])
				}
				continue
			}
			if outF32[i] != wantF32 {
				t.Fatalf("ToFloat32 mismatch at %d (input 0x%04x): got %v, want %v", i, want.Bits(), outF32[i], wantF32)
			}
			if outF16[i] != want {
				t.Fatalf("FromFloat32x4 mismatch at %d: got 0x%04x (%v), want 0x%04x (%v)", i, outF16[i].Bits(), outF16[i], want.Bits(), want)
			}
		}
	})

	t.Run("SliceNEON_Asm", func(t *testing.T) {
		// Test with non-multiple-of-8 length as well to exercise tail loop.
		inWithTail := append([]Float16(nil), inF16...)
		inWithTail = append(inWithTail, FromFloat32(3.5), FromFloat32(-0.125), SmallestNonzero)
		outF32 := make([]float32, len(inWithTail))
		outF16 := make([]Float16, len(inWithTail))

		ToFloat32SliceNEON(outF32, inWithTail)
		FromFloat32SliceNEON(outF16, outF32)

		for i, want := range inWithTail {
			wantF32 := want.Float32()
			if math.IsNaN(float64(wantF32)) {
				if !math.IsNaN(float64(outF32[i])) || !math.IsNaN(float64(outF16[i].Float32())) {
					t.Fatalf("index %d: expected NaN, got f32=%v f16=0x%04x", i, outF32[i], outF16[i])
				}
				continue
			}
			if outF32[i] != wantF32 {
				t.Fatalf("ToFloat32SliceNEON mismatch at %d (input 0x%04x): got %v, want %v", i, want.Bits(), outF32[i], wantF32)
			}
			if outF16[i] != want {
				t.Fatalf("FromFloat32SliceNEON mismatch at %d: got 0x%04x, want 0x%04x", i, outF16[i].Bits(), want.Bits())
			}
		}
	})
}

func BenchmarkFloat16Conversions(b *testing.B) {
	const n = 4096
	srcF16 := make([]Float16, n)
	dstF32 := make([]float32, n)
	dstF16 := make([]Float16, n)
	for i := range srcF16 {
		srcF16[i] = FromFloat32(float32(i)*0.01 - 20.0)
	}

	b.Run("Scalar_ToF32", func(b *testing.B) {
		for b.Loop() {
			for i := range n {
				dstF32[i] = srcF16[i].Float32()
			}
		}
	})
	b.Run("NEON_Inline_ToF32", func(b *testing.B) {
		for b.Loop() {
			for i := 0; i < n; i += 8 {
				v := LoadFloat16x8((*[8]Float16)(srcF16[i : i+8]))
				lo, hi := v.ToFloat32()
				lo.StoreArray((*[4]float32)(dstF32[i : i+4]))
				hi.StoreArray((*[4]float32)(dstF32[i+4 : i+8]))
			}
		}
	})
	b.Run("NEON_Asm_FCVTL_ToF32", func(b *testing.B) {
		for b.Loop() {
			ToFloat32SliceNEON(dstF32, srcF16)
		}
	})
	b.Run("Scalar_FromF32", func(b *testing.B) {
		for b.Loop() {
			for i := range n {
				dstF16[i] = FromFloat32(dstF32[i])
			}
		}
	})
	b.Run("NEON_Asm_FCVTN_FromF32", func(b *testing.B) {
		for b.Loop() {
			FromFloat32SliceNEON(dstF16, dstF32)
		}
	})
}

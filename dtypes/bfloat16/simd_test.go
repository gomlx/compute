// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package bfloat16

import (
	"math"
	"simd"
	"testing"
)

func TestBFloat16SIMD(t *testing.T) {
	dummy := simd.LoadUint16s(make([]uint16, 64))
	vecLen := dummy.Len()

	var testFloats []float32
	for x := -10.0; x <= 10.0; x += 0.25 {
		testFloats = append(testFloats, float32(x))
	}
	testFloats = append(testFloats, 0, -0.0, float32(math.Inf(1)), float32(math.Inf(-1)), 1.2345, -7.891)
	for len(testFloats)%vecLen != 0 {
		testFloats = append(testFloats, 0)
	}

	inBF16 := make([]BFloat16, len(testFloats))
	for i, f := range testFloats {
		inBF16[i] = FromFloat32(f)
	}

	outBF16 := make([]BFloat16, len(inBF16))

	for i := 0; i < len(inBF16); i += vecLen {
		v := LoadBFloat16s(inBF16[i : i+vecLen])
		even, odd := ToFloat32SIMD(v)
		res := FromFloat32SIMD(even, odd)
		StoreBFloat16s(res, outBF16[i : i+vecLen])
	}

	for i := range inBF16 {
		if inBF16[i] != outBF16[i] {
			t.Fatalf("mismatch at %d: got %v, expected %v", i, outBF16[i], inBF16[i])
		}
	}

	// Test partial / tail
	tailLen := vecLen / 2
	if tailLen == 0 {
		tailLen = 1
	}
	vPart, nLoaded := LoadBFloat16sPart(inBF16[:tailLen])
	if nLoaded != tailLen {
		t.Fatalf("expected loaded %d, got %d", tailLen, nLoaded)
	}
	even, odd := ToFloat32SIMD(vPart)
	res := FromFloat32SIMD(even, odd)
	tailOut := make([]BFloat16, tailLen)
	nStored := StoreBFloat16sPart(res, tailOut)
	if nStored != tailLen {
		t.Fatalf("expected stored %d, got %d", tailLen, nStored)
	}
	for i := 0; i < tailLen; i++ {
		if tailOut[i] != inBF16[i] {
			t.Fatalf("tail mismatch at %d: got %v, expected %v", i, tailOut[i], inBF16[i])
		}
	}
}

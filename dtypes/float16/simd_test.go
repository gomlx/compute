// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package float16

import (
	"math"
	"simd"
	"testing"
)

func TestFloat16SIMD(t *testing.T) {
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

	inF16 := make([]Float16, len(testFloats))
	for i, f := range testFloats {
		inF16[i] = FromFloat32(f)
	}

	outF16 := make([]Float16, len(inF16))

	for i := 0; i < len(inF16); i += vecLen {
		v := LoadFloat16s(inF16[i : i+vecLen])
		even, odd := ToFloat32SIMD(v)
		res := FromFloat32SIMD(even, odd)
		StoreFloat16s(res, outF16[i : i+vecLen])
	}

	for i := range inF16 {
		if inF16[i] != outF16[i] {
			t.Fatalf("mismatch at %d: got %v, expected %v", i, outF16[i], inF16[i])
		}
	}

	// Test partial / tail
	tailLen := vecLen / 2
	if tailLen == 0 {
		tailLen = 1
	}
	vPart, nLoaded := LoadFloat16sPart(inF16[:tailLen])
	if nLoaded != tailLen {
		t.Fatalf("expected loaded %d, got %d", tailLen, nLoaded)
	}
	even, odd := ToFloat32SIMD(vPart)
	res := FromFloat32SIMD(even, odd)
	tailOut := make([]Float16, tailLen)
	nStored := StoreFloat16sPart(res, tailOut)
	if nStored != tailLen {
		t.Fatalf("expected stored %d, got %d", tailLen, nStored)
	}
	for i := 0; i < tailLen; i++ {
		if tailOut[i] != inF16[i] {
			t.Fatalf("tail mismatch at %d: got %v, expected %v", i, tailOut[i], inF16[i])
		}
	}
}

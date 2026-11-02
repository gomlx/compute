// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package simdmath_test

import (
	"simd"
	"testing"

	"github.com/gomlx/compute/support/simdmath"
)

//go:noinline
func loadTailF32(s, dst []float32) int {
	vZero := simd.BroadcastFloat32s(0)
	v, n := simdmath.LoadFloat32sPart(s)
	v.Add(vZero).StorePart(dst)
	return n
}

// TestLoadPartPageBoundary verifies that LoadFloat32sPart does not over-read
// into unmapped pages when a slice ends at a heap arena/page boundary.
func TestLoadPartPageBoundary(t *testing.T) {
	const n = 768 // 3072 bytes (0xc00)
	var out [4]float32
	keep := make([][]float32, 0, 20000)
	for range 20000 {
		s := make([]float32, n)
		s[n-4] = 1
		s[n-3] = 2
		s[n-2] = 3
		s[n-1] = 4
		keep = append(keep, s)
		loaded := loadTailF32(s[n-4:], out[:])
		if loaded != 4 {
			t.Fatalf("expected 4 loaded elements, got %d", loaded)
		}
	}
	if out[0] != 1 || out[1] != 2 || out[2] != 3 || out[3] != 4 {
		t.Fatalf("unexpected tail values: %v (retained %d)", out, len(keep))
	}
}

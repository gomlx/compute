// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon_test

import (
	"math/rand/v2"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/activations/neon"
	"github.com/gomlx/compute/support/testutil"
)

func TestGeluNEONFloat32(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	sizes := []int{1, 3, 4, 5, 7, 8, 9, 12, 15, 16, 19, 31, 32, 1025}
	for _, sz := range sizes {
		in := make([]float32, sz)
		for i := range in {
			in[i] = float32(i-sz/2) * 0.25
		}
		if sz >= 8 {
			in[0] = -25.0
			in[1] = -9.5
			in[2] = -9.0
			in[3] = 0.0
			in[4] = 9.0
			in[5] = 9.5
			in[6] = 25.0
		}

		wantExact := make([]float32, sz)
		outExact := make([]float32, sz)
		activations.GeluExactNoSIMD(in, wantExact)
		neon.GeluExactFloat32NEON(in, outExact)
		for i, x := range in {
			if ok, diff := testutil.IsInDelta(wantExact[i], outExact[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluExactFloat32NEON[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}

		wantApprox := make([]float32, sz)
		outApprox := make([]float32, sz)
		activations.GeluApproxNoSIMD(in, wantApprox)
		neon.GeluApproxFloat32NEON(in, outApprox)
		for i, x := range in {
			if ok, diff := testutil.IsInDelta(wantApprox[i], outApprox[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluApproxFloat32NEON[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}
	}
}

func BenchmarkNEONComparison(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	sizes := []struct {
		name string
		n    int
	}{
		{"Small_512", 512},
		{"Large_65536", 65536},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			inF32 := make([]float32, sz.n)
			outF32 := make([]float32, sz.n)
			for i := range inF32 {
				inF32[i] = rand.Float32()*4 - 2
			}
			for _, tc := range []struct {
				name string
				fn   func(in, out []float32)
			}{
				{"GeluExact/NoSIMD", activations.GeluExactNoSIMD[float32]},
				{"GeluExact/PortableSIMD", activations.GeluExactFloat32SIMD},
				{"GeluExact/NEON", neon.GeluExactFloat32NEON},
				{"GeluApprox/NoSIMD", activations.GeluApproxNoSIMD[float32]},
				{"GeluApprox/PortableSIMD", activations.GeluApproxFloat32SIMD},
				{"GeluApprox/NEON", neon.GeluApproxFloat32NEON},
			} {
				b.Run(tc.name, func(b *testing.B) {
					for b.Loop() {
						tc.fn(inF32, outF32)
					}
				})
			}
		})
	}
}

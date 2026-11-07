// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package activations_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/support/testutil"
)

const PriorityNEON = 20

func init() {
	registerBaseline()
	if !gobackend.IsNEONAllowed {
		return
	}
	f32Registry.Register("GeluExact", "NEON_Asm", PriorityNEON, activations.GeluExactFloat32NEON)
	f32Registry.Register("GeluApprox", "NEON_Asm", PriorityNEON, activations.GeluApproxFloat32NEON)
}

func TestNEONFlavorsRegistered(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}
	for _, op := range []string{"GeluExact", "GeluApprox"} {
		found := false
		for _, f := range f32Registry.flavors[op] {
			if f.name == "NEON_Asm" {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("NEON_Asm flavor not registered for op %q", op)
		}
	}
}

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

		outExact := make([]float32, sz)
		activations.GeluExactFloat32NEON(in, outExact)
		for i, x := range in {
			xf := float64(x)
			want := float32(xf * 0.5 * (1.0 + math.Erf(xf/math.Sqrt2)))
			if ok, diff := testutil.IsInDelta(want, outExact[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluExactFloat32NEON[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}

		outApprox := make([]float32, sz)
		activations.GeluApproxFloat32NEON(in, outApprox)
		sqrt2ByPi := math.Sqrt(2.0 / math.Pi)
		for i, x := range in {
			xf := float64(x)
			inner := sqrt2ByPi * (xf + 0.044715*xf*xf*xf)
			want := float32(xf * 0.5 * (1.0 + math.Tanh(inner)))
			if ok, diff := testutil.IsInDelta(want, outApprox[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluApproxFloat32NEON[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}
	}
}

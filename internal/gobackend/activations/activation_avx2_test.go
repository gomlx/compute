// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package activations_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations/avx2"
	"github.com/gomlx/compute/support/testutil"
)

func init() {
	registerBaseline()
	if !gobackend.IsAVX2Allowed {
		return
	}
	f32Registry.RegisterInPlace("Relu", "AVX2_ArchSIMD", PriorityAVX2, avx2.ReluAVX2)
	f32Registry.RegisterInPlace("Silu", "AVX2_ArchSIMD", PriorityAVX2, avx2.SiluAVX2)
	f32Registry.RegisterInPlace("Tanh", "AVX2_ArchSIMD", PriorityAVX2, avx2.TanhAVX2)
	f32Registry.Register("GeluApprox", "AVX2_ArchSIMD", PriorityAVX2, avx2.GeluAVX2)
	f32Registry.Register("GeluExact", "AVX2_ArchSIMD", PriorityAVX2, avx2.GeluExactAVX2)
	f32Registry.RegisterInPlace("HardSwish", "AVX2_ArchSIMD", PriorityAVX2, avx2.HardSwishAVX2)
}

func TestAVX2FlavorsRegistered(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this host")
	}
	for _, op := range []string{"Relu", "Silu", "Tanh", "GeluApprox", "GeluExact", "HardSwish"} {
		found := false
		for _, f := range f32Registry.flavors[op] {
			if f.name == "AVX2_ArchSIMD" {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("AVX2_ArchSIMD flavor not registered for op %q", op)
		}
	}
}

func TestGeluAVX2Float32(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this host")
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
		avx2.GeluExactAVX2(in, outExact)
		for i, x := range in {
			xf := float64(x)
			want := float32(xf * 0.5 * (1.0 + math.Erf(xf/math.Sqrt2)))
			if ok, diff := testutil.IsInDelta(want, outExact[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluExactAVX2[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}

		outApprox := make([]float32, sz)
		avx2.GeluAVX2(in, outApprox)
		sqrt2ByPi := math.Sqrt(2.0 / math.Pi)
		for i, x := range in {
			xf := float64(x)
			inner := sqrt2ByPi * (xf + 0.044715*xf*xf*xf)
			want := float32(xf * 0.5 * (1.0 + math.Tanh(inner)))
			if ok, diff := testutil.IsInDelta(want, outApprox[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluAVX2[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}
	}
}

func TestGeluBiasTileAVX2Float32(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this host")
	}

	widths := []int{1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 515}
	const numRows = 5
	sqrt2ByPi := math.Sqrt(2.0 / math.Pi)

	for _, width := range widths {
		rowStride := width + 11 // non-contiguous stride to verify rowStride handling and guard zones
		totalLen := numRows * rowStride
		base := make([]float32, totalLen)
		for i := range base {
			base[i] = float32((i%41)-20) * 0.3
		}
		bias := make([]float32, width)
		for c := range width {
			bias[c] = float32(c-width/2) * 0.15
		}

		// Test Exact
		outExact := make([]float32, totalLen)
		copy(outExact, base)
		avx2.GeluExactBiasTileAVX2(outExact, rowStride, numRows, width, bias)
		for r := range numRows {
			for c := range rowStride {
				idx := r*rowStride + c
				if c < width {
					xf := float64(base[idx] + bias[c])
					want := float32(xf * 0.5 * (1.0 + math.Erf(xf/math.Sqrt2)))
					if ok, diff := testutil.IsInDelta(want, outExact[idx], 1e-4); !ok {
						t.Fatalf("width=%d row=%d col=%d GeluExactBiasTileAVX2 mismatch: %s", width, r, c, diff)
					}
				} else if outExact[idx] != base[idx] {
					t.Fatalf("width=%d row=%d col=%d guard zone overwritten: got %v, want %v", width, r, c, outExact[idx], base[idx])
				}
			}
		}

		// Test Approx
		outApprox := make([]float32, totalLen)
		copy(outApprox, base)
		avx2.GeluApproxBiasTileAVX2(outApprox, rowStride, numRows, width, bias)
		for r := range numRows {
			for c := range rowStride {
				idx := r*rowStride + c
				if c < width {
					xf := float64(base[idx] + bias[c])
					inner := sqrt2ByPi * (xf + 0.044715*xf*xf*xf)
					want := float32(xf * 0.5 * (1.0 + math.Tanh(inner)))
					if ok, diff := testutil.IsInDelta(want, outApprox[idx], 1e-4); !ok {
						t.Fatalf("width=%d row=%d col=%d GeluApproxBiasTileAVX2 mismatch: %s", width, r, c, diff)
					}
				} else if outApprox[idx] != base[idx] {
					t.Fatalf("width=%d row=%d col=%d guard zone overwritten: got %v, want %v", width, r, c, outApprox[idx], base[idx])
				}
			}
		}
	}
}

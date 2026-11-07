// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package activations_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations/avx512"
	"github.com/gomlx/compute/support/testutil"
)

func init() {
	registerBaseline()
	if !gobackend.IsAVX512Allowed {
		return
	}
	f32Registry.RegisterInPlace("Relu", "AVX512_ArchSIMD", PriorityAVX512, avx512.ReluAVX512)
	f32Registry.RegisterInPlace("Silu", "AVX512_ArchSIMD", PriorityAVX512, avx512.SiluAVX512)
	f32Registry.RegisterInPlace("Tanh", "AVX512_ArchSIMD", PriorityAVX512, avx512.TanhAVX512)
	f32Registry.Register("GeluApprox", "AVX512_ArchSIMD", PriorityAVX512, avx512.GeluAVX512)
	f32Registry.Register("GeluExact", "AVX512_ArchSIMD", PriorityAVX512, avx512.GeluExactAVX512)
	f32Registry.RegisterInPlace("Sigmoid", "AVX512_ArchSIMD", PriorityAVX512, avx512.SigmoidAVX512)
	f32Registry.RegisterInPlace("HardSigmoid", "AVX512_ArchSIMD", PriorityAVX512, avx512.HardSigmoidAVX512)
	f32Registry.RegisterInPlace("HardSwish", "AVX512_ArchSIMD", PriorityAVX512, avx512.HardSwishAVX512)
	f32Registry.RegisterInPlace("LeakyRelu", "AVX512_ArchSIMD", PriorityAVX512, avx512.LeakyReluAVX512)
	f32Registry.RegisterInPlace("Selu", "AVX512_ArchSIMD", PriorityAVX512, avx512.SeluAVX512)
}

func TestAVX512FlavorsRegistered(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}
	for _, op := range []string{"Relu", "Silu", "Tanh", "GeluApprox", "GeluExact", "Sigmoid", "HardSigmoid", "HardSwish", "LeakyRelu", "Selu"} {
		found := false
		for _, f := range f32Registry.flavors[op] {
			if f.name == "AVX512_ArchSIMD" {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("AVX512_ArchSIMD flavor not registered for op %q", op)
		}
	}
}

func TestGeluAVX512Float32(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}

	sizes := []int{1, 3, 4, 5, 7, 8, 9, 12, 15, 16, 19, 31, 32, 47, 64, 1025}
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
		avx512.GeluExactAVX512(in, outExact)
		for i, x := range in {
			xf := float64(x)
			want := float32(xf * 0.5 * (1.0 + math.Erf(xf/math.Sqrt2)))
			if ok, diff := testutil.IsInDelta(want, outExact[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluExactAVX512[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}

		outApprox := make([]float32, sz)
		avx512.GeluAVX512(in, outApprox)
		sqrt2ByPi := math.Sqrt(2.0 / math.Pi)
		for i, x := range in {
			xf := float64(x)
			inner := sqrt2ByPi * (xf + 0.044715*xf*xf*xf)
			want := float32(xf * 0.5 * (1.0 + math.Tanh(inner)))
			if ok, diff := testutil.IsInDelta(want, outApprox[i], 1e-4); !ok {
				t.Fatalf("size=%d GeluAVX512[%d] (x=%f) mismatch: %s", sz, i, x, diff)
			}
		}
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package activations_test

import (
	"simd/archsimd"
	"testing"

	"github.com/gomlx/compute/internal/gobackend/activations/avx512"
)

func init() {
	registerBaseline()
	if !archsimd.X86.AVX512() {
		return
	}
	f32Registry.RegisterInPlace("Relu", "AVX512_ArchSIMD", PriorityAVX512, avx512.ReluAVX512)
	f32Registry.RegisterInPlace("Silu", "AVX512_ArchSIMD", PriorityAVX512, avx512.SiluAVX512)
	f32Registry.RegisterInPlace("Tanh", "AVX512_ArchSIMD", PriorityAVX512, avx512.TanhAVX512)
	f32Registry.RegisterInPlace("GeluApprox", "AVX512_ArchSIMD", PriorityAVX512, avx512.GeluAVX512)
	f32Registry.RegisterInPlace("Sigmoid", "AVX512_ArchSIMD", PriorityAVX512, avx512.SigmoidAVX512)
	f32Registry.RegisterInPlace("HardSigmoid", "AVX512_ArchSIMD", PriorityAVX512, avx512.HardSigmoidAVX512)
	f32Registry.RegisterInPlace("HardSwish", "AVX512_ArchSIMD", PriorityAVX512, avx512.HardSwishAVX512)
	f32Registry.RegisterInPlace("LeakyRelu", "AVX512_ArchSIMD", PriorityAVX512, avx512.LeakyReluAVX512)
	f32Registry.RegisterInPlace("Selu", "AVX512_ArchSIMD", PriorityAVX512, avx512.SeluAVX512)
}

func TestAVX512FlavorsRegistered(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX512 not supported on this CPU")
	}
	for _, op := range []string{"Relu", "Silu", "Tanh", "GeluApprox", "Sigmoid", "HardSigmoid", "HardSwish", "LeakyRelu", "Selu"} {
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

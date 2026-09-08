// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package activations_test

import (
	"simd/archsimd"
	"testing"

	"github.com/gomlx/compute/internal/gobackend/activations/avx2"
)

func init() {
	registerBaseline()
	if !archsimd.X86.AVX2() {
		return
	}
	f32Registry.RegisterInPlace("Relu", "AVX2_ArchSIMD", PriorityAVX2, avx2.ReluAVX2)
	f32Registry.RegisterInPlace("Silu", "AVX2_ArchSIMD", PriorityAVX2, avx2.SiluAVX2)
	f32Registry.RegisterInPlace("Tanh", "AVX2_ArchSIMD", PriorityAVX2, avx2.TanhAVX2)
	f32Registry.RegisterInPlace("GeluApprox", "AVX2_ArchSIMD", PriorityAVX2, avx2.GeluAVX2)
	f32Registry.RegisterInPlace("HardSwish", "AVX2_ArchSIMD", PriorityAVX2, avx2.HardSwishAVX2)
}

func TestAVX2FlavorsRegistered(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not supported on this CPU")
	}
	for _, op := range []string{"Relu", "Silu", "Tanh", "GeluApprox", "HardSwish"} {
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

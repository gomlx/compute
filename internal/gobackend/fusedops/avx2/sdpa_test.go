// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
)

func TestDispatchSDPAAVX2_Correctness(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX-2 is not supported or allowed on this machine")
	}

	for _, headDim := range []int{16, 32, 64} {
		for _, causal := range []bool{false, true} {
			seqLen := 2
			kvLen := 2
			groupSize := 1

			q := make([]float32, seqLen*headDim)
			k := make([]float32, kvLen*headDim)
			v := make([]float32, kvLen*headDim)
			output := make([]float32, seqLen*headDim)
			scratch := make([]float32, seqLen*kvLen)

			for i := range q {
				q[i] = 1.0
			}
			for i := range k {
				k[i] = 1.0
			}
			// v[0] = 10, v[1] = 20
			for d := range headDim {
				v[0*headDim+d] = 10.0
				v[1*headDim+d] = 20.0
			}

			ok := DispatchSDPAAVX2(
				q, k, v, output,
				0, 0, headDim, headDim, headDim*seqLen,
				nil, nil, 0,
				nil, 0,
				scratch,
				groupSize, seqLen, kvLen, headDim,
				1.0, causal,
				seqLen, kvLen,
			)
			if !ok {
				t.Fatalf("DispatchSDPAAVX2 returned false for headDim=%d", headDim)
			}

			// Check query 0:
			// Under causal, query 0 only sees key 0 -> output is 10.0.
			// Under non-causal, query 0 sees both key 0 and key 1 with equal weight -> output is 15.0.
			wantQ0 := float32(15.0)
			if causal {
				wantQ0 = 10.0
			}
			// Query 1 sees both key 0 and key 1 with equal weight (causal or non-causal) -> output is 15.0.
			wantQ1 := float32(15.0)

			for d := range headDim {
				got0 := output[0*headDim+d]
				if math.Abs(float64(got0-wantQ0)) > 1e-4 {
					t.Errorf("headDim=%d causal=%v: query 0 dim %d mismatch: got %f, want %f", headDim, causal, d, got0, wantQ0)
					break
				}
				got1 := output[1*headDim+d]
				if math.Abs(float64(got1-wantQ1)) > 1e-4 {
					t.Errorf("headDim=%d causal=%v: query 1 dim %d mismatch: got %f, want %f", headDim, causal, d, got1, wantQ1)
					break
				}
			}
		}
	}
}

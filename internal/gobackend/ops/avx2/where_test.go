// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
)

func TestWhereFloat32AVX2(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this host")
	}

	sizes := []int{0, 1, 3, 7, 8, 9, 15, 16, 23, 32, 65, 1024}
	cases := []struct {
		name          string
		onTrueScalar  bool
		onFalseScalar bool
	}{
		{"VectorVector", false, false},
		{"VectorScalar", false, true},
		{"ScalarVector", true, false},
		{"ScalarScalar", true, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, size := range sizes {
				cond := make([]bool, size)
				for i := range size {
					cond[i] = (i%3 != 0)
				}
				var onTrue, onFalse []float32
				if c.onTrueScalar {
					onTrue = []float32{42.5}
				} else {
					onTrue = make([]float32, size)
					for i := range size {
						onTrue[i] = float32(i + 1)
					}
				}
				if c.onFalseScalar {
					onFalse = []float32{-99.25}
				} else {
					onFalse = make([]float32, size)
					for i := range size {
						onFalse[i] = float32(-(i + 1))
					}
				}
				out := make([]float32, size)
				if !DispatchWhereFloat32AVX2(cond, onTrue, onFalse, out, c.onTrueScalar, c.onFalseScalar) {
					t.Fatalf("DispatchWhereFloat32AVX2 returned false")
				}
				for i := range size {
					var want float32
					if cond[i] {
						if c.onTrueScalar {
							want = onTrue[0]
						} else {
							want = onTrue[i]
						}
					} else {
						if c.onFalseScalar {
							want = onFalse[0]
						} else {
							want = onFalse[i]
						}
					}
					if out[i] != want {
						t.Fatalf("%s size=%d idx=%d: got %v, want %v", c.name, size, i, out[i], want)
					}
				}
			}
		})
	}
}

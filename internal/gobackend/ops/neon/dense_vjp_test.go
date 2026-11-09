// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon_test

import (
	"math/rand/v2"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops/neon"
	"github.com/gomlx/compute/support/testutil"
)

func refAtB(a, b []float32, m, p, q int) []float32 {
	out := make([]float32, p*q)
	for i := range m {
		aRow := a[i*p : (i+1)*p]
		bRow := b[i*q : (i+1)*q]
		for r, av := range aRow {
			outRow := out[r*q : (r+1)*q]
			for c, bv := range bRow {
				outRow[c] += av * bv
			}
		}
	}
	return out
}

func TestTallSkinnyAtBFloat32ARM64(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	backend, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend failed: %+v", err)
	}

	shapesList := []struct {
		name    string
		m, p, q int
	}{
		{"Small_M64_P3_Q32", 64, 3, 32},
		{"Odd_M127_P19_Q37", 127, 19, 37},
		{"Medium_M1024_P32_Q64", 1024, 32, 64},
		{"Tall_M5000_P15_Q29", 5000, 15, 29},
		{"Tall_M8192_P64_Q96", 8192, 64, 96},
	}

	for _, sc := range shapesList {
		t.Run(sc.name, func(t *testing.T) {
			a := make([]float32, sc.m*sc.p)
			bMat := make([]float32, sc.m*sc.q)
			for i := range a {
				a[i] = (rand.Float32() - 0.5) * 0.5
			}
			for i := range bMat {
				bMat[i] = (rand.Float32() - 0.5) * 0.5
			}
			want := refAtB(a, bMat, sc.m, sc.p, sc.q)

			got := make([]float32, sc.p*sc.q)
			if !neon.ExecTallSkinnyAtBFloat32ARM64(backend, a, bMat, sc.m, sc.p, sc.q, got) {
				t.Fatalf("ExecTallSkinnyAtBFloat32ARM64 returned false for %+v", sc)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-2); !ok {
				t.Fatalf("ExecTallSkinnyAtBFloat32ARM64 mismatch: %s", diff)
			}

			if gobackend.IsSMEAllowed {
				origSME := gobackend.IsSMEAllowed
				gobackend.IsSMEAllowed = false
				defer func() { gobackend.IsSMEAllowed = origSME }()

				gotNEON := make([]float32, sc.p*sc.q)
				if !neon.ExecTallSkinnyAtBFloat32ARM64(backend, a, bMat, sc.m, sc.p, sc.q, gotNEON) {
					t.Fatalf("ExecTallSkinnyAtBFloat32ARM64 (NEON-only) returned false for %+v", sc)
				}
				if ok, diff := testutil.IsInDelta(want, gotNEON, 1e-2); !ok {
					t.Fatalf("ExecTallSkinnyAtBFloat32ARM64 (NEON-only) mismatch: %s", diff)
				}
			}
		})
	}
}

func TestReduceBiasFloat32ARM64(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	backend, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend failed: %+v", err)
	}

	for _, tc := range []struct {
		m, n int
	}{
		{10, 17},
		{256, 64},
		{2048, 97},
	} {
		dZ := make([]float32, tc.m*tc.n)
		for i := range dZ {
			dZ[i] = (rand.Float32() - 0.5) * 0.5
		}
		want := make([]float32, tc.n)
		for r := range tc.m {
			for c := range tc.n {
				want[c] += dZ[r*tc.n+c]
			}
		}
		got := make([]float32, tc.n)
		if !neon.ReduceBiasFloat32ARM64(backend, dZ, tc.m, tc.n, got) {
			t.Fatalf("ReduceBiasFloat32ARM64 returned false for m=%d, n=%d", tc.m, tc.n)
		}
		if ok, diff := testutil.IsInDelta(want, got, 1e-2); !ok {
			t.Fatalf("ReduceBiasFloat32ARM64(m=%d, n=%d) mismatch: %s", tc.m, tc.n, diff)
		}
	}
}

func BenchmarkTallSkinnyAtBFloat32ARM64(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}
	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("NewBackend failed: %+v", err)
	}
	m, p, q := 131072, 64, 64
	a := make([]float32, m*p)
	bMat := make([]float32, m*q)
	out := make([]float32, p*q)
	for i := range a {
		a[i] = 0.01
	}
	for i := range bMat {
		bMat[i] = 0.01
	}
	b.ResetTimer()
	for b.Loop() {
		neon.ExecTallSkinnyAtBFloat32ARM64(backend, a, bMat, m, p, q, out)
	}
}


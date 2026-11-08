// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/activations/avx2"
	"github.com/gomlx/compute/support/testutil"
)

func inPlaceToKernel[T any](fn func(data []T)) func(in, out []T) {
	return func(in, out []T) {
		copy(out, in)
		fn(out)
	}
}

func TestAVX2Float32Kernels(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this host")
	}

	kernels := []struct {
		name     string
		avx2Fn   func(in, out []float32)
		noSIMDFn func(in, out []float32)
	}{
		{"Relu", inPlaceToKernel(avx2.ReluAVX2), activations.ReluNoSIMD[float32]},
		{"Silu", avx2.SiluKernelAVX2, activations.SiluNoSIMD[float32]},
		{"Tanh", inPlaceToKernel(avx2.TanhAVX2), activations.TanhNoSIMD[float32]},
		{"GeluApprox", avx2.GeluAVX2, activations.GeluApproxNoSIMD[float32]},
		{"GeluExact", avx2.GeluExactAVX2, activations.GeluExactNoSIMD[float32]},
		{"HardSwish", inPlaceToKernel(avx2.HardSwishAVX2), activations.HardSwishNoSIMD[float32]},
	}

	sizes := []int{1, 3, 4, 5, 7, 8, 9, 12, 15, 16, 19, 31, 32, 65, 512, 1025}
	for _, k := range kernels {
		t.Run(k.name, func(t *testing.T) {
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

				want := make([]float32, sz)
				got := make([]float32, sz)
				k.noSIMDFn(in, want)
				k.avx2Fn(in, got)

				for i, x := range in {
					if ok, diff := testutil.IsInRelativeDelta(want[i], got[i], 1e-3); !ok {
						if math.Abs(float64(want[i]-got[i])) > 1e-4 {
							t.Fatalf("size=%d %s[%d] (x=%f) mismatch: got %f, want %f (%s)", sz, k.name, i, x, got[i], want[i], diff)
						}
					}
				}
			}
		})
	}
}

func TestBiasTileAVX2Float32(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this host")
	}

	tiles := []struct {
		name     string
		tileFn   activations.BiasTileFn[float32]
		noSIMDFn activations.KernelFn[float32]
	}{
		{"GeluExact", avx2.GeluExactBiasTileAVX2, activations.GeluExactNoSIMD[float32]},
		{"GeluApprox", avx2.GeluApproxBiasTileAVX2, activations.GeluApproxNoSIMD[float32]},
		{"Silu", avx2.SiluBiasTileAVX2, activations.SiluNoSIMD[float32]},
	}

	widths := []int{1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 515}
	const numRows = 5

	for _, tc := range tiles {
		t.Run(tc.name, func(t *testing.T) {
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

				want := make([]float32, totalLen)
				copy(want, base)
				activations.ApplyBiasTileNoSIMD(tc.noSIMDFn, want, rowStride, numRows, width, bias)

				got := make([]float32, totalLen)
				copy(got, base)
				tc.tileFn(got, rowStride, numRows, width, bias)

				for r := range numRows {
					for c := range rowStride {
						idx := r*rowStride + c
						if c < width {
							if ok, diff := testutil.IsInDelta(want[idx], got[idx], 1e-4); !ok {
								t.Fatalf("width=%d row=%d col=%d %s mismatch: %s", width, r, c, tc.name, diff)
							}
						} else if got[idx] != base[idx] {
							t.Fatalf("width=%d row=%d col=%d %s guard zone overwritten: got %v, want %v", width, r, c, tc.name, got[idx], base[idx])
						}
					}
				}
			}
		})
	}
}

func TestSiluAndVJPAVX2AllDTypes(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this host")
	}

	vjpBF16NoSIMD := activations.MakeBF16VJPKernelFromF32(activations.VJPSiluNoSIMD[float32])
	vjpF16NoSIMD := activations.MakeF16VJPKernelFromF32(activations.VJPSiluNoSIMD[float32])

	sizes := []int{1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 515}
	for _, sz := range sizes {
		inF32 := make([]float32, sz)
		dOutF32 := make([]float32, sz)
		inF64 := make([]float64, sz)
		dOutF64 := make([]float64, sz)
		inBF16 := make([]bfloat16.BFloat16, sz)
		dOutBF16 := make([]bfloat16.BFloat16, sz)
		inF16 := make([]float16.Float16, sz)
		dOutF16 := make([]float16.Float16, sz)

		for i := range sz {
			x := float32((i%41)-20) * 0.25
			d := float32(((i*7)%19)-9) * 0.2
			inF32[i] = x
			dOutF32[i] = d
			inF64[i] = float64(x)
			dOutF64[i] = float64(d)
			inBF16[i] = bfloat16.FromFloat32(x)
			dOutBF16[i] = bfloat16.FromFloat32(d)
			inF16[i] = float16.FromFloat32(x)
			dOutF16[i] = float16.FromFloat32(d)
		}

		wantOutF32 := make([]float32, sz)
		wantDxF32 := make([]float32, sz)
		activations.SiluNoSIMD(inF32, wantOutF32)
		activations.VJPSiluNoSIMD(nil, inF32, dOutF32, wantDxF32)
		outF32 := make([]float32, sz)
		dxF32 := make([]float32, sz)
		avx2.SiluKernelAVX2(inF32, outF32)
		avx2.SiluVJPAVX2(nil, inF32, dOutF32, dxF32)

		wantOutF64 := make([]float64, sz)
		wantDxF64 := make([]float64, sz)
		activations.SiluNoSIMD(inF64, wantOutF64)
		activations.VJPSiluNoSIMD(nil, inF64, dOutF64, wantDxF64)
		outF64 := make([]float64, sz)
		dxF64 := make([]float64, sz)
		avx2.SiluAVX2Float64(inF64, outF64)
		avx2.SiluVJPAVX2Float64(nil, inF64, dOutF64, dxF64)

		wantOutBF16 := make([]bfloat16.BFloat16, sz)
		wantDxBF16 := make([]bfloat16.BFloat16, sz)
		activations.SiluBF16NoSIMD(inBF16, wantOutBF16)
		vjpBF16NoSIMD(nil, inBF16, dOutBF16, wantDxBF16)
		outBF16 := make([]bfloat16.BFloat16, sz)
		dxBF16 := make([]bfloat16.BFloat16, sz)
		avx2.SiluAVX2BF16(inBF16, outBF16)
		avx2.SiluVJPAVX2BF16(nil, inBF16, dOutBF16, dxBF16)

		wantOutF16 := make([]float16.Float16, sz)
		wantDxF16 := make([]float16.Float16, sz)
		activations.SiluF16NoSIMD(inF16, wantOutF16)
		vjpF16NoSIMD(nil, inF16, dOutF16, wantDxF16)
		outF16 := make([]float16.Float16, sz)
		dxF16 := make([]float16.Float16, sz)
		avx2.SiluAVX2F16(inF16, outF16)
		avx2.SiluVJPAVX2F16(nil, inF16, dOutF16, dxF16)

		for i := range sz {
			if ok, diff := testutil.IsInDelta(wantOutF32[i], outF32[i], 1e-4); !ok {
				t.Fatalf("sz=%d SiluKernelAVX2[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxF32[i], dxF32[i], 1e-4); !ok {
				t.Fatalf("sz=%d SiluVJPAVX2[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantOutF64[i], outF64[i], 1e-6); !ok {
				t.Fatalf("sz=%d SiluAVX2Float64[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxF64[i], dxF64[i], 1e-6); !ok {
				t.Fatalf("sz=%d SiluVJPAVX2Float64[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantOutBF16[i].Float32(), outBF16[i].Float32(), 5e-2); !ok {
				t.Fatalf("sz=%d SiluAVX2BF16[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxBF16[i].Float32(), dxBF16[i].Float32(), 5e-2); !ok {
				t.Fatalf("sz=%d SiluVJPAVX2BF16[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantOutF16[i].Float32(), outF16[i].Float32(), 2e-2); !ok {
				t.Fatalf("sz=%d SiluAVX2F16[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxF16[i].Float32(), dxF16[i].Float32(), 2e-2); !ok {
				t.Fatalf("sz=%d SiluVJPAVX2F16[%d] mismatch: %s", sz, i, diff)
			}
		}
	}
}

func BenchmarkAVX2Comparison(b *testing.B) {
	if !gobackend.IsAVX2Allowed {
		b.Skip("AVX2 is not supported or allowed on this host")
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
				{"Silu/NoSIMD", activations.SiluNoSIMD[float32]},
				{"Silu/PortableSIMD", activations.SiluFloat32SIMD},
				{"Silu/AVX2", avx2.SiluKernelAVX2},
				{"GeluExact/NoSIMD", activations.GeluExactNoSIMD[float32]},
				{"GeluExact/PortableSIMD", activations.GeluExactFloat32SIMD},
				{"GeluExact/AVX2", avx2.GeluExactAVX2},
				{"GeluApprox/NoSIMD", activations.GeluApproxNoSIMD[float32]},
				{"GeluApprox/PortableSIMD", activations.GeluApproxFloat32SIMD},
				{"GeluApprox/AVX2", avx2.GeluAVX2},
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

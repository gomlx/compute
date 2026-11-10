// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/activations/avx512"
	"github.com/gomlx/compute/support/testutil"
)

func inPlaceToKernel[T any](fn func(data []T)) func(in, out []T) {
	return func(in, out []T) {
		copy(out, in)
		fn(out)
	}
}

func TestAVX512Float32Kernels(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}

	kernels := []struct {
		name     string
		avx512Fn func(in, out []float32)
		noSIMDFn func(in, out []float32)
	}{
		{"Relu", inPlaceToKernel(avx512.ReluAVX512), activations.ReluNoSIMD[float32]},
		{"Silu", avx512.SiluKernelAVX512, activations.SiluNoSIMD[float32]},
		{"Tanh", inPlaceToKernel(avx512.TanhAVX512), activations.TanhNoSIMD[float32]},
		{"GeluApprox", avx512.GeluAVX512, activations.GeluApproxNoSIMD[float32]},
		{"GeluExact", avx512.GeluExactAVX512, activations.GeluExactNoSIMD[float32]},
		{"Sigmoid", inPlaceToKernel(avx512.SigmoidAVX512), activations.SigmoidNoSIMD[float32]},
		{"HardSigmoid", inPlaceToKernel(avx512.HardSigmoidAVX512), activations.HardSigmoidNoSIMD[float32]},
		{"HardSwish", inPlaceToKernel(avx512.HardSwishAVX512), activations.HardSwishNoSIMD[float32]},
		{"LeakyRelu", inPlaceToKernel(avx512.LeakyReluAVX512), activations.LeakyReluNoSIMD[float32]},
		{"Selu", inPlaceToKernel(avx512.SeluAVX512), activations.SeluNoSIMD[float32]},
	}

	sizes := []int{1, 3, 4, 5, 7, 8, 9, 12, 15, 16, 19, 31, 32, 47, 64, 65, 512, 1025}
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
				k.avx512Fn(in, got)

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

func TestBiasTileAVX512Float32(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}

	tiles := []struct {
		name     string
		tileFn   activations.BiasTileFn[float32]
		noSIMDFn activations.KernelFn[float32]
	}{
		{"GeluExact", avx512.GeluExactBiasTileAVX512, activations.GeluExactNoSIMD[float32]},
		{"GeluApprox", avx512.GeluApproxBiasTileAVX512, activations.GeluApproxNoSIMD[float32]},
		{"Silu", avx512.SiluBiasTileAVX512, activations.SiluNoSIMD[float32]},
	}

	widths := []int{1, 3, 7, 15, 16, 17, 31, 32, 47, 64, 515}
	const numRows = 5

	for _, tc := range tiles {
		t.Run(tc.name, func(t *testing.T) {
			for _, width := range widths {
				rowStride := width + 13 // non-contiguous stride to verify rowStride handling and guard zones
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

func TestSiluAndVJPAVX512AllDTypes(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}

	vjpBF16NoSIMD := activations.MakeBF16VJPKernelFromF32(activations.VJPSiluNoSIMD[float32])
	vjpF16NoSIMD := activations.MakeF16VJPKernelFromF32(activations.VJPSiluNoSIMD[float32])

	sizes := []int{1, 3, 7, 15, 16, 17, 31, 32, 33, 64, 515}
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
		avx512.SiluKernelAVX512(inF32, outF32)
		avx512.SiluVJPAVX512(nil, inF32, dOutF32, dxF32)

		wantOutF64 := make([]float64, sz)
		wantDxF64 := make([]float64, sz)
		activations.SiluNoSIMD(inF64, wantOutF64)
		activations.VJPSiluNoSIMD(nil, inF64, dOutF64, wantDxF64)
		outF64 := make([]float64, sz)
		dxF64 := make([]float64, sz)
		avx512.SiluAVX512Float64(inF64, outF64)
		avx512.SiluVJPAVX512Float64(nil, inF64, dOutF64, dxF64)

		wantOutBF16 := make([]bfloat16.BFloat16, sz)
		wantDxBF16 := make([]bfloat16.BFloat16, sz)
		activations.SiluBF16NoSIMD(inBF16, wantOutBF16)
		vjpBF16NoSIMD(nil, inBF16, dOutBF16, wantDxBF16)
		outBF16 := make([]bfloat16.BFloat16, sz)
		dxBF16 := make([]bfloat16.BFloat16, sz)
		avx512.SiluAVX512BF16(inBF16, outBF16)
		avx512.SiluVJPAVX512BF16(nil, inBF16, dOutBF16, dxBF16)

		wantOutF16 := make([]float16.Float16, sz)
		wantDxF16 := make([]float16.Float16, sz)
		activations.SiluF16NoSIMD(inF16, wantOutF16)
		vjpF16NoSIMD(nil, inF16, dOutF16, wantDxF16)
		outF16 := make([]float16.Float16, sz)
		dxF16 := make([]float16.Float16, sz)
		avx512.SiluAVX512F16(inF16, outF16)
		avx512.SiluVJPAVX512F16(nil, inF16, dOutF16, dxF16)

		for i := range sz {
			if ok, diff := testutil.IsInDelta(wantOutF32[i], outF32[i], 1e-4); !ok {
				t.Fatalf("sz=%d SiluKernelAVX512[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxF32[i], dxF32[i], 1e-4); !ok {
				t.Fatalf("sz=%d SiluVJPAVX512[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantOutF64[i], outF64[i], 1e-6); !ok {
				t.Fatalf("sz=%d SiluAVX512Float64[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxF64[i], dxF64[i], 1e-6); !ok {
				t.Fatalf("sz=%d SiluVJPAVX512Float64[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantOutBF16[i].Float32(), outBF16[i].Float32(), 5e-2); !ok {
				t.Fatalf("sz=%d SiluAVX512BF16[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxBF16[i].Float32(), dxBF16[i].Float32(), 5e-2); !ok {
				t.Fatalf("sz=%d SiluVJPAVX512BF16[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantOutF16[i].Float32(), outF16[i].Float32(), 2e-2); !ok {
				t.Fatalf("sz=%d SiluAVX512F16[%d] mismatch: %s", sz, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDxF16[i].Float32(), dxF16[i].Float32(), 2e-2); !ok {
				t.Fatalf("sz=%d SiluVJPAVX512F16[%d] mismatch: %s", sz, i, diff)
			}
		}
	}
}

func TestSwiGLUAVX512AllDTypes(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}

	for _, tc := range []struct {
		numRows   int
		hiddenDim int
	}{
		{1, 1},
		{2, 3},
		{3, 4},
		{4, 7},
		{2, 8},
		{3, 15},
		{4, 16},
		{5, 31},
		{5, 32},
		{5, 33},
		{8, 128},
		{5, 1024},
	} {
		inLen := tc.numRows * 2 * tc.hiddenDim
		outLen := tc.numRows * tc.hiddenDim

		inF32 := make([]float32, inLen)
		inF64 := make([]float64, inLen)
		inBF16 := make([]bfloat16.BFloat16, inLen)
		inF16 := make([]float16.Float16, inLen)
		for i := range inLen {
			x := float32((i%41)-20) * 0.25
			inF32[i] = x
			inF64[i] = float64(x)
			inBF16[i] = bfloat16.FromFloat32(x)
			inF16[i] = float16.FromFloat32(x)
		}

		wantF32 := make([]float32, outLen)
		wantF64 := make([]float64, outLen)
		activations.SwiGLUFloat32SIMD(inF32, wantF32, tc.numRows, tc.hiddenDim)
		activations.SwiGLUFloat64SIMD(inF64, wantF64, tc.numRows, tc.hiddenDim)

		gotF32 := make([]float32, outLen)
		gotF64 := make([]float64, outLen)
		gotBF16 := make([]bfloat16.BFloat16, outLen)
		gotF16 := make([]float16.Float16, outLen)
		avx512.SwiGLUAVX512(inF32, gotF32, tc.numRows, tc.hiddenDim)
		avx512.SwiGLUAVX512Float64(inF64, gotF64, tc.numRows, tc.hiddenDim)
		avx512.SwiGLUAVX512BF16(inBF16, gotBF16, tc.numRows, tc.hiddenDim)
		avx512.SwiGLUAVX512F16(inF16, gotF16, tc.numRows, tc.hiddenDim)

		for i := range outLen {
			if ok, diff := testutil.IsInDelta(wantF32[i], gotF32[i], 1e-4); !ok {
				t.Fatalf("rows=%d hidden=%d SwiGLUAVX512[%d] mismatch: %s", tc.numRows, tc.hiddenDim, i, diff)
			}
			if ok, diff := testutil.IsInDelta(wantF64[i], gotF64[i], 1e-6); !ok {
				t.Fatalf("rows=%d hidden=%d SwiGLUAVX512Float64[%d] mismatch: %s", tc.numRows, tc.hiddenDim, i, diff)
			}
			if ok, diff := testutil.IsInDelta(bfloat16.FromFloat32(wantF32[i]).Float32(), gotBF16[i].Float32(), 0.15); !ok {
				t.Fatalf("rows=%d hidden=%d SwiGLUAVX512BF16[%d] mismatch: %s", tc.numRows, tc.hiddenDim, i, diff)
			}
			if ok, diff := testutil.IsInDelta(float16.FromFloat32(wantF32[i]).Float32(), gotF16[i].Float32(), 2e-2); !ok {
				t.Fatalf("rows=%d hidden=%d SwiGLUAVX512F16[%d] mismatch: %s", tc.numRows, tc.hiddenDim, i, diff)
			}
		}
	}
}

func BenchmarkAVX512Comparison(b *testing.B) {
	if !gobackend.IsAVX512Allowed {
		b.Skip("AVX-512 is not supported or allowed on this host")
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
				{"Silu/AVX512", avx512.SiluKernelAVX512},
				{"GeluExact/NoSIMD", activations.GeluExactNoSIMD[float32]},
				{"GeluExact/PortableSIMD", activations.GeluExactFloat32SIMD},
				{"GeluExact/AVX512", avx512.GeluExactAVX512},
				{"GeluApprox/NoSIMD", activations.GeluApproxNoSIMD[float32]},
				{"GeluApprox/PortableSIMD", activations.GeluApproxFloat32SIMD},
				{"GeluApprox/AVX512", avx512.GeluAVX512},
				{"Sigmoid/NoSIMD", activations.SigmoidNoSIMD[float32]},
				{"Sigmoid/AVX512", avx512.SigmoidKernelAVX512},
				{"Tanh/NoSIMD", activations.TanhNoSIMD[float32]},
				{"Tanh/AVX512", avx512.TanhKernelAVX512},
			} {
				b.Run(tc.name, func(b *testing.B) {
					for b.Loop() {
						tc.fn(inF32, outF32)
					}
				})
			}

			halfDim := sz.n / 2
			for _, tc := range []struct {
				name string
				fn   func(in, out []float32, numRows, hiddenDim int)
			}{
				{"SwiGLU/PortableSIMD", activations.SwiGLUFloat32SIMD},
				{"SwiGLU/AVX512", avx512.SwiGLUAVX512},
			} {
				b.Run(tc.name, func(b *testing.B) {
					for b.Loop() {
						tc.fn(inF32, outF32[:halfDim], 1, halfDim)
					}
				})
			}
		})
	}
}

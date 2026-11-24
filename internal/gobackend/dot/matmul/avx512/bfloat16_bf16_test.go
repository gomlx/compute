// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

package avx512

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/support/humanize"
)

func TestAVX512BFloat16Native(t *testing.T) {
	if !gobackend.IsAVX512BF16Allowed {
		t.Skip("AVX-512 BF16 (VDPBF16PS) is not supported or allowed on this host")
	}

	origUseAsm := AVX512UseAsm
	origUseBF16 := AVX512UseBF16
	defer func() {
		AVX512UseAsm = origUseAsm
		AVX512UseBF16 = origUseBF16
	}()
	AVX512UseAsm = true

	t.Run("PackK2", func(t *testing.T) {
		for _, tc := range []struct {
			rows, cols           int
			rowStart, colStart   int
			copyRows, copyCols   int
		}{
			{16, 64, 0, 0, 16, 64},
			{16, 65, 0, 0, 16, 65},
			{19, 37, 2, 3, 17, 33},
			{35, 70, 1, 4, 34, 65},
			{64, 1, 0, 0, 64, 1},
			{64, 2, 0, 0, 64, 2},
			{64, 3, 0, 0, 64, 3},
			{70, 130, 3, 2, 67, 127},
		} {
			src := make([]bfloat16.BFloat16, tc.rows*tc.cols)
			for i := range src {
				src[i] = bfloat16.FromFloat32(float32((i%29)-14) * 0.25)
			}

			// LHS K2 (kernelRows = 8 and kernelRows = 32)
			for _, kr := range []int{8, 32} {
				numStrips := (tc.copyRows + kr - 1) / kr
				paddedK := (tc.copyCols + 1) & ^1
				panel := make([]bfloat16.BFloat16, numStrips*paddedK*kr)
				for i := range panel {
					panel[i] = 0xBEEF
				}
				avx512PackLHSBFloat16K2(src, panel, tc.rowStart, tc.colStart, tc.cols, tc.copyRows, tc.copyCols, kr)
				kPairs := paddedK / 2
				for s := range numStrips {
					for p := range kPairs {
						for r := range kr {
							row := s*kr + r
							for j := range 2 {
								k := 2*p + j
								var want bfloat16.BFloat16
								if row < tc.copyRows && k < tc.copyCols {
									want = src[(tc.rowStart+row)*tc.cols+tc.colStart+k]
								}
								got := panel[(s*kPairs+p)*kr*2+r*2+j]
								if got != want {
									t.Fatalf("PackLHSK2(kr=%d, %+v) mismatch at (s=%d,p=%d,r=%d,j=%d): got 0x%x, want 0x%x",
										kr, tc, s, p, r, j, uint16(got), uint16(want))
								}
							}
						}
					}
				}
			}

			// RHS NonTransposed K2 (kernelCols = 32)
			numStrips := (tc.copyCols + 31) / 32
			paddedK := (tc.copyRows + 1) & ^1
			panel := make([]bfloat16.BFloat16, numStrips*paddedK*32)
			for i := range panel {
				panel[i] = 0xBEEF
			}
			avx512PackRHSNonTransposedBFloat16K2(src, panel, tc.rowStart, tc.colStart, tc.cols, tc.copyRows, tc.copyCols, 32)
			kPairs := paddedK / 2
			for s := range numStrips {
				for p := range kPairs {
					for c := range 32 {
						col := s*32 + c
						for j := range 2 {
							k := 2*p + j
							var want bfloat16.BFloat16
							if k < tc.copyRows && col < tc.copyCols {
								want = src[(tc.rowStart+k)*tc.cols+tc.colStart+col]
							}
							got := panel[(s*kPairs+p)*32*2+c*2+j]
							if got != want {
								t.Fatalf("PackRHSNonTK2(%+v) mismatch at (s=%d,p=%d,c=%d,j=%d): got 0x%x, want 0x%x",
									tc, s, p, c, j, uint16(got), uint16(want))
							}
						}
					}
				}
			}
		}
	})

	t.Run("LargeM1GEMV", func(t *testing.T) {
		AVX512UseBF16 = true
		backend, err := gobackend.NewBackend()
		if err != nil {
			t.Fatalf("NewBackend: %v", err)
		}
		defer backend.Finalize()

		for _, tc := range []struct {
			K, N int
		}{
			{1, 1}, {1, 32}, {1, 128}, {1, 129},
			{2, 32}, {3, 33}, {17, 160}, {65, 257},
			{256, 128}, {257, 128}, {512, 256}, {513, 513},
			{1536, 1536}, {1536, 2048}, {1537, 1539},
		} {
			K, N := tc.K, tc.N
			lhs := make([]bfloat16.BFloat16, K)
			for i := range lhs {
				lhs[i] = bfloat16.FromFloat32(float32((i%11)-5) * 0.25)
			}
			rhsNT := make([]bfloat16.BFloat16, K*N)
			rhsT := make([]bfloat16.BFloat16, N*K)
			for k := range K {
				for n := range N {
					v := bfloat16.FromFloat32(float32(((k*7+n*5)%13)-6) * 0.125)
					rhsNT[k*N+n] = v
					rhsT[n*K+k] = v
				}
			}
			ref := make([]float32, N)
			for n := range N {
				var sum float32
				for k := range K {
					sum += lhs[k].Float32() * rhsNT[k*N+n].Float32()
				}
				ref[n] = sum
			}

			for _, layout := range []dot.Layout{dot.LayoutNonTransposed, dot.LayoutTransposed} {
				rhs := rhsNT
				if layout == dot.LayoutTransposed {
					rhs = rhsT
				}
				// Test uncached and pre-packed cached RHS.
				for _, cached := range []bool{false, true} {
					var nd *dot.NodeData
					if cached {
						cd := &gobackend.ConstantData{}
						nd = &dot.NodeData{RHSConstantData: cd}
					}
					out := make([]float32, N)
					avx512LargeBFloat16(backend, layout, lhs, rhs, 1, 1, N, K, out, nd)
					for n, want := range ref {
						if math.Abs(float64(out[n]-want)) > 1e-3*math.Max(1, math.Abs(float64(want))) {
							t.Fatalf("LargeM1(K=%d,N=%d,layout=%v,cached=%v) col=%d: got %v, want %v",
								K, N, layout, cached, n, out[n], want)
						}
					}
				}
			}
		}
	})

	t.Run("SmallTransposedAndNonTransposed", func(t *testing.T) {
		for _, M := range []int{1, 2, 3, 4, 5, 7, 8, 13} {
			for _, N := range []int{1, 2, 3, 4, 5, 15, 16, 17, 32, 64, 69} {
				for _, K := range []int{1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 64, 69} {
					A := make([]bfloat16.BFloat16, M*K)
					for i := range A {
						A[i] = bfloat16.FromFloat32(float32((i%13)-6) * 0.25)
					}
					B := make([]bfloat16.BFloat16, K*N)
					BT := make([]bfloat16.BFloat16, N*K)
					for k := range K {
						for n := range N {
							v := bfloat16.FromFloat32(float32(((k*7+n*5)%17)-8) * 0.25)
							B[k*N+n] = v
							BT[n*K+k] = v
						}
					}
					ref := make([]float32, M*N)
					for r := range M {
						for c := range N {
							var sum float32
							for k := range K {
								sum += A[r*K+k].Float32() * B[k*N+c].Float32()
							}
							ref[r*N+c] = sum
						}
					}

					for _, useBF16 := range []bool{false, true} {
						AVX512UseBF16 = useBF16
						outNT := make([]float32, M*N)
						avx512SmallBFloat16NonTransposedAsm(A, B, 0, 1, M, N, K, outNT)
						outT := make([]float32, M*N)
						avx512SmallBFloat16TransposedAsm(A, BT, 0, 1, M, N, K, outT)
						for i, want := range ref {
							if math.Abs(float64(outNT[i]-want)) > 1e-4 {
								t.Fatalf("SmallNonT(useBF16=%v, M=%d,N=%d,K=%d) idx=%d: got %v, want %v",
									useBF16, M, N, K, i, outNT[i], want)
							}
							if math.Abs(float64(outT[i]-want)) > 1e-4 {
								t.Fatalf("SmallT(useBF16=%v, M=%d,N=%d,K=%d) idx=%d: got %v, want %v",
									useBF16, M, N, K, i, outT[i], want)
							}
						}
					}
				}
			}
		}
	})

	t.Run("LargeAllModes", func(t *testing.T) {
		backend, err := gobackend.NewBackend()
		if err != nil {
			t.Fatalf("failed to create backend: %+v", err)
		}
		defer backend.Finalize()

		cases := []struct {
			name       string
			B, M, K, N int
		}{
			{"OddK_1", 1, 16, 1, 32},
			{"OddK_3", 1, 16, 3, 64},
			{"OddAll_37x129x67", 1, 37, 129, 67},
			{"ExactMultiK_64x256x128", 1, 64, 256, 128},
			{"OddKMultiPanel_50x259x100", 1, 50, 259, 100},
			{"Batched_3x36x131x70", 3, 36, 131, 70},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				B, M, K, N := tc.B, tc.M, tc.K, tc.N
				lhs := make([]bfloat16.BFloat16, B*M*K)
				rhsNonT := make([]bfloat16.BFloat16, B*K*N)
				rhsTrans := make([]bfloat16.BFloat16, B*N*K)
				for i := range lhs {
					lhs[i] = bfloat16.FromFloat32(float32((i%15)-7) * 0.125)
				}
				for b := range B {
					for k := range K {
						for n := range N {
							v := bfloat16.FromFloat32(float32(((b*11+k*7+n*3)%17)-8) * 0.125)
							rhsNonT[b*K*N+k*N+n] = v
							rhsTrans[b*N*K+n*K+k] = v
						}
					}
				}

				// Scalar FP32 reference
				ref := make([]float32, B*M*N)
				for b := range B {
					for r := range M {
						for c := range N {
							var sum float32
							for k := range K {
								sum += lhs[b*M*K+r*K+k].Float32() * rhsNonT[b*K*N+k*N+c].Float32()
							}
							ref[b*M*N+r*N+c] = sum
						}
					}
				}

				for _, useBF16 := range []bool{false, true} {
					AVX512UseBF16 = useBF16
					for _, workers := range []int{1, 2, 4} {
						backend.Scheduler.SetMaxParallelism(workers)
						for _, layout := range []dot.Layout{dot.LayoutNonTransposed, dot.LayoutTransposed} {
							rhs := rhsNonT
							if layout == dot.LayoutTransposed {
								rhs = rhsTrans
							}
							for _, cacheMode := range []string{"Uncached", "CachedRHS", "CachedLHS"} {
								var nodeData *dot.NodeData
								switch cacheMode {
								case "CachedRHS":
									nodeData = &dot.NodeData{RHSConstantData: gobackend.NewAnonymousConstantData()}
								case "CachedLHS":
									nodeData = &dot.NodeData{LHSConstantData: gobackend.NewAnonymousConstantData()}
								}
								got := make([]float32, B*M*N)
								for i := range got {
									got[i] = -99999.0
								}
								avx512LargeBFloat16(backend, layout, lhs, rhs, B, M, N, K, got, nodeData)
								for i, want := range ref {
									if math.Abs(float64(got[i]-want)) > 1e-3 || math.IsNaN(float64(got[i])) {
										t.Fatalf("Large(useBF16=%v, w=%d, %v, %s) idx=%d: got %v, want %v",
											useBF16, workers, layout, cacheMode, i, got[i], want)
									}
								}
							}

							// MultiRHS path
							got0 := make([]float32, B*M*N)
							got1 := make([]float32, B*M*N)
							targets := []matmul.MultiRHSTarget[bfloat16.BFloat16, float32]{
								{RHS: rhs, RHSCrossSize: N, Output: got0},
								{RHS: rhs, RHSCrossSize: N, Output: got1, NodeData: &dot.NodeData{RHSConstantData: gobackend.NewAnonymousConstantData()}},
							}
							origForceLarge := matmul.ForceLargeVariant
							matmul.ForceLargeVariant = true
							ran := avx512LargeMultiBFloat16(backend, layout, lhs, B, M, K, targets)
							matmul.ForceLargeVariant = origForceLarge
							if ran {
								for i, want := range ref {
									if math.Abs(float64(got0[i]-want)) > 1e-3 || math.Abs(float64(got1[i]-want)) > 1e-3 {
										t.Fatalf("MultiRHS(useBF16=%v, w=%d, %v) idx=%d: got (%v,%v), want %v",
											useBF16, workers, layout, i, got0[i], got1[i], want)
									}
								}
							}
						}
					}
				}
			})
		}
	})
}

func BenchmarkBFloat16Microkernels(b *testing.B) {
	if !gobackend.IsAVX512Allowed {
		b.Skip("AVX512 is not supported on this architecture")
	}

	for _, tc := range []struct {
		name    string
		M, N, K int
	}{
		{"Panel_32x512x128", 32, 512, 128},
		{"Panel_32x512x256", 32, 512, 256},
		{"Panel_192x384x192", 192, 384, 192},
	} {
		M, N, K := tc.M, tc.N, tc.K
		flops := float64(2 * M * N * K)

		lhsF32 := make([]float32, M*K)
		rhsF32 := make([]float32, K*N)
		lhsBF16 := make([]bfloat16.BFloat16, M*K)
		rhsBF16 := make([]bfloat16.BFloat16, K*N)
		out := make([]float32, M*N)

		b.Run(tc.name+"/Float32_VFMADD231PS", func(b *testing.B) {
			for range 3 {
				avx512LargeKernelFloat32Asm(lhsF32, rhsF32, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				avx512LargeKernelFloat32Asm(lhsF32, rhsF32, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})

		b.Run(tc.name+"/BFloat16_UnpackF32", func(b *testing.B) {
			for range 3 {
				avx512LargeKernelBFloat16Asm(lhsBF16, rhsBF16, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				avx512LargeKernelBFloat16Asm(lhsBF16, rhsBF16, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})

		b.Run(tc.name+"/BFloat16_VDPBF16PS", func(b *testing.B) {
			if !gobackend.IsAVX512BF16Allowed {
				b.Skip("AVX-512 BF16 (VDPBF16PS) is not supported on this host")
			}
			for range 3 {
				avx512LargeKernelBFloat16BF16Asm(lhsBF16, rhsBF16, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				avx512LargeKernelBFloat16BF16Asm(lhsBF16, rhsBF16, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}
}

func BenchmarkBFloat16MatMul(b *testing.B) {
	if !gobackend.IsAVX512Allowed {
		b.Skip("AVX512 is not supported on this architecture")
	}

	origUseAsm := AVX512UseAsm
	origUseBF16 := AVX512UseBF16
	defer func() {
		AVX512UseAsm = origUseAsm
		AVX512UseBF16 = origUseBF16
	}()
	AVX512UseAsm = true

	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	cases := []struct {
		name       string
		layout     dot.Layout
		B, M, K, N int
	}{
		// Small / UCI-adult & test-cases
		{"Small_NonT_128x69x4", dot.LayoutNonTransposed, 1, 128, 69, 4},
		{"Small_T_128x69x4", dot.LayoutTransposed, 1, 128, 69, 4},
		{"Small_NonT_32x20x64", dot.LayoutNonTransposed, 1, 32, 20, 64},
		// Transformer / Dense sizes
		{"MiniLM_416x384x1152", dot.LayoutNonTransposed, 1, 416, 384, 1152},
		{"BGE_512x768x768", dot.LayoutNonTransposed, 1, 512, 768, 768},
		{"BGE_128x768x3072", dot.LayoutNonTransposed, 1, 128, 768, 3072},
		{"Square_1024x1024x1024", dot.LayoutNonTransposed, 1, 1024, 1024, 1024},
		// Usual Large matmul sizes (from backendtest/bench_dotgeneral.go)
		{"NoBatch-Large-1_1536x1920x1024", dot.LayoutNonTransposed, 1, 1536, 1920, 1024},
		{"NoBatch-Large-1_Transposed", dot.LayoutTransposed, 1, 1536, 1920, 1024},
		{"NoBatch-Large-2_1024x1920x1536", dot.LayoutNonTransposed, 1, 1024, 1920, 1536},
		{"NoBatch-Large-3_2048x2048x2048", dot.LayoutNonTransposed, 1, 2048, 2048, 2048},
		{"Batched-Large-1_16x1536x1920x1024", dot.LayoutNonTransposed, 16, 1536, 1920, 1024},
	}

	for _, tc := range cases {
		flops := float64(2 * tc.B * tc.M * tc.N * tc.K)
		lhsF32 := make([]float32, tc.B*tc.M*tc.K)
		rhsF32 := make([]float32, tc.B*tc.K*tc.N)
		lhsBF16 := make([]bfloat16.BFloat16, tc.B*tc.M*tc.K)
		rhsBF16 := make([]bfloat16.BFloat16, tc.B*tc.K*tc.N)
		for i := range lhsF32 {
			v := float32((i%13)-6) * 0.01
			lhsF32[i] = v
			lhsBF16[i] = bfloat16.FromFloat32(v)
		}
		for i := range rhsF32 {
			v := float32((i%17)-8) * 0.01
			rhsF32[i] = v
			rhsBF16[i] = bfloat16.FromFloat32(v)
		}
		out := make([]float32, tc.B*tc.M*tc.N)

		b.Run(fmt.Sprintf("%s/Float32", tc.name), func(b *testing.B) {
			for range 3 {
				avx512RouterFloat32(backend, tc.layout, lhsF32, rhsF32, tc.B, tc.M, tc.N, tc.K, out, nil)
			}
			for b.Loop() {
				avx512RouterFloat32(backend, tc.layout, lhsF32, rhsF32, tc.B, tc.M, tc.N, tc.K, out, nil)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})

		b.Run(fmt.Sprintf("%s/BF16_UnpackF32", tc.name), func(b *testing.B) {
			AVX512UseBF16 = false
			for range 3 {
				avx512RouterBFloat16(backend, tc.layout, lhsBF16, rhsBF16, tc.B, tc.M, tc.N, tc.K, out, nil)
			}
			for b.Loop() {
				avx512RouterBFloat16(backend, tc.layout, lhsBF16, rhsBF16, tc.B, tc.M, tc.N, tc.K, out, nil)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})

		b.Run(fmt.Sprintf("%s/BF16_VDPBF16PS", tc.name), func(b *testing.B) {
			if !gobackend.IsAVX512BF16Allowed {
				b.Skip("AVX-512 BF16 (VDPBF16PS) is not supported on this host")
			}
			AVX512UseBF16 = true
			for range 3 {
				avx512RouterBFloat16(backend, tc.layout, lhsBF16, rhsBF16, tc.B, tc.M, tc.N, tc.K, out, nil)
			}
			for b.Loop() {
				avx512RouterBFloat16(backend, tc.layout, lhsBF16, rhsBF16, tc.B, tc.M, tc.N, tc.K, out, nil)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}
}

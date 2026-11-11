// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"simd/archsimd"
	"testing"
	"time"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
)

func TestAVX2(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported on this architecture")
	}

	t.Run("Transpose/4x4x64bits", func(t *testing.T) {
		var input [4 * 4]uint64
		for i := range input {
			input[i] = uint64(i)
		}

		v0 := archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Pointer(&input[0*4])))
		v1 := archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Pointer(&input[1*4])))
		v2 := archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Pointer(&input[2*4])))
		v3 := archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Pointer(&input[3*4])))

		q0, q1, q2, q3 := avx2Transpose4x4x64bits(v0, v1, v2, v3)

		var output [4 * 4]uint64
		q0.StoreArray((*[4]uint64)(unsafe.Pointer(&output[0*4])))
		q1.StoreArray((*[4]uint64)(unsafe.Pointer(&output[1*4])))
		q2.StoreArray((*[4]uint64)(unsafe.Pointer(&output[2*4])))
		q3.StoreArray((*[4]uint64)(unsafe.Pointer(&output[3*4])))

		for c := range 4 { // logical column
			for r := range 4 { // logical row
				expected := uint64(r*4 + c)
				got := output[c*4+r]
				if got != expected {
					t.Errorf("At output col %d, row %d: got %d, expected %d", c, r, got, expected)
				}
			}
		}
	})

	t.Run("Transpose/4x8x32bits", func(t *testing.T) {
		var input [4 * 8]uint32
		for i := range input {
			input[i] = uint32(i)
		}

		v0 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&input[0*8])))
		v1 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&input[1*8])))
		v2 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&input[2*8])))
		v3 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&input[3*8])))

		q0, q1, q2, q3 := avx2Transpose4x8x32bits(v0, v1, v2, v3)

		var output [4 * 8]uint32
		q0.StoreArray((*[8]uint32)(unsafe.Pointer(&output[0*8])))
		q1.StoreArray((*[8]uint32)(unsafe.Pointer(&output[1*8])))
		q2.StoreArray((*[8]uint32)(unsafe.Pointer(&output[2*8])))
		q3.StoreArray((*[8]uint32)(unsafe.Pointer(&output[3*8])))

		for c := range 8 { // logical column
			for r := range 4 { // logical row
				expected := uint32(r*8 + c)
				got := output[c*4+r]
				if got != expected {
					t.Errorf("At output col %d, row %d: got %d, expected %d", c, r, got, expected)
				}
			}
		}
	})

	t.Run("Transpose/4x16x16bits", func(t *testing.T) {
		var input [4 * 16]uint16
		for i := range input {
			input[i] = uint16(i)
		}

		v0 := archsimd.LoadUint16x16Array((*[16]uint16)(unsafe.Pointer(&input[0*16])))
		v1 := archsimd.LoadUint16x16Array((*[16]uint16)(unsafe.Pointer(&input[1*16])))
		v2 := archsimd.LoadUint16x16Array((*[16]uint16)(unsafe.Pointer(&input[2*16])))
		v3 := archsimd.LoadUint16x16Array((*[16]uint16)(unsafe.Pointer(&input[3*16])))

		q0, q1, q2, q3 := avx2Transpose4x16x16bits(v0, v1, v2, v3)

		var output [4 * 16]uint16
		q0.StoreArray((*[16]uint16)(unsafe.Pointer(&output[0*16])))
		q1.StoreArray((*[16]uint16)(unsafe.Pointer(&output[1*16])))
		q2.StoreArray((*[16]uint16)(unsafe.Pointer(&output[2*16])))
		q3.StoreArray((*[16]uint16)(unsafe.Pointer(&output[3*16])))

		for c := range 16 { // logical column
			for r := range 4 { // logical row
				expected := uint16(r*16 + c)
				got := output[c*4+r]
				if got != expected {
					t.Errorf("At output col %d, row %d: got %d, expected %d", c, r, got, expected)
				}
			}
		}
	})

	t.Run("Small/Float32", func(t *testing.T) {
		for _, M := range []int{1, 3, 4, 7, 8, 15, 25, 49, 128} {
			for _, N := range []int{1, 2, 4, 7, 8, 15, 16, 20} {
				for _, K := range []int{1, 4, 7, 8, 15, 16, 17, 32, 69} {
					// 1. Transposed test: A is [M, K], BT is [N, K], refC = A * BT^T
					A := make([]float32, M*K)
					for i := range A {
						A[i] = float32(i%13 - 6)
					}
					BT := make([]float32, N*K)
					for i := range BT {
						BT[i] = float32(i%17 - 8)
					}
					refC := make([]float32, M*N)
					for r := 0; r < M; r++ {
						for c := 0; c < N; c++ {
							var sum float32
							for k := 0; k < K; k++ {
								sum += A[r*K+k] * BT[c*K+k]
							}
							refC[r*N+c] = sum
						}
					}

					outAsm := make([]float32, M*N)
					avx2SmallFloat32TransposedAsm(A, BT, 0, 1, M, N, K, outAsm)
					for r := 0; r < M; r++ {
						for c := 0; c < N; c++ {
							got := outAsm[r*N+c]
							expected := refC[r*N+c]
							if got != expected {
								t.Fatalf("Asm Transposed Mismatch at M=%d, N=%d, K=%d, (%d, %d): got %v, expected %v", M, N, K, r, c, got, expected)
							}
						}
					}

					// 2. NonTransposed test: A is [M, K], B is [K, N], refC = A * B
					B := make([]float32, K*N)
					for i := range B {
						B[i] = float32(i%17 - 8)
					}
					for r := 0; r < M; r++ {
						for c := 0; c < N; c++ {
							var sum float32
							for k := 0; k < K; k++ {
								sum += A[r*K+k] * B[k*N+c]
							}
							refC[r*N+c] = sum
						}
					}

					outNonT := make([]float32, M*N)
					avx2SmallFloat32NonTransposedAsm(A, B, 0, 1, M, N, K, outNonT)
					for r := 0; r < M; r++ {
						for c := 0; c < N; c++ {
							got := outNonT[r*N+c]
							expected := refC[r*N+c]
							if got != expected {
								t.Fatalf("Asm NonTransposed Mismatch at M=%d, N=%d, K=%d, (%d, %d): got %v, expected %v", M, N, K, r, c, got, expected)
							}
						}
					}
				}
			}
		}
	})

	t.Run("SmallAllDTypesCorrectness", func(t *testing.T) {
		t.Run("Float64", func(t *testing.T) {
			for _, M := range []int{1, 3, 4, 7, 8, 15, 25} {
				for _, N := range []int{1, 2, 4, 7, 8, 15, 16, 20} {
					for _, K := range []int{1, 4, 7, 8, 9, 16, 17, 32} {
						// Transposed:
						A := make([]float64, M*K)
						for i := range A {
							A[i] = float64(i%13 - 6)
						}
						BT := make([]float64, N*K)
						for i := range BT {
							BT[i] = float64(i%17 - 8)
						}
						refC := make([]float64, M*N)
						for r := 0; r < M; r++ {
							for c := 0; c < N; c++ {
								var sum float64
								for k := 0; k < K; k++ {
									sum += A[r*K+k] * BT[c*K+k]
								}
								refC[r*N+c] = sum
							}
						}
						outT := make([]float64, M*N)
						avx2SmallFloat64TransposedAsm(A, BT, 0, 1, M, N, K, outT)
						for i := range refC {
							if outT[i] != refC[i] {
								t.Fatalf("Float64 Transposed mismatch at M=%d, N=%d, K=%d, idx=%d: got %v, expected %v", M, N, K, i, outT[i], refC[i])
							}
						}

						// Non-Transposed:
						B := make([]float64, K*N)
						for i := range B {
							B[i] = float64(i%17 - 8)
						}
						refNonT := make([]float64, M*N)
						for r := 0; r < M; r++ {
							for c := 0; c < N; c++ {
								var sum float64
								for k := 0; k < K; k++ {
									sum += A[r*K+k] * B[k*N+c]
								}
								refNonT[r*N+c] = sum
							}
						}
						outNonT := make([]float64, M*N)
						avx2SmallFloat64NonTransposedAsm(A, B, 0, 1, M, N, K, outNonT)
						for i := range refNonT {
							if outNonT[i] != refNonT[i] {
								t.Fatalf("Float64 NonTransposed mismatch at M=%d, N=%d, K=%d, idx=%d: got %v, expected %v", M, N, K, i, outNonT[i], refNonT[i])
							}
						}
					}
				}
			}
		})

		t.Run("Float16", func(t *testing.T) {
			for _, M := range []int{1, 3, 4, 7, 8, 15, 25} {
				for _, N := range []int{1, 2, 4, 7, 8, 15, 16, 20} {
					for _, K := range []int{1, 4, 15, 16, 17, 32} {
						A := make([]float16.Float16, M*K)
						for i := range A {
							A[i] = float16.FromFloat32(float32(i%13 - 6))
						}
						BT := make([]float16.Float16, N*K)
						for i := range BT {
							BT[i] = float16.FromFloat32(float32(i%17 - 8))
						}
						refC := make([]float32, M*N)
						for r := 0; r < M; r++ {
							for c := 0; c < N; c++ {
								var sum float32
								for k := 0; k < K; k++ {
									sum += A[r*K+k].Float32() * BT[c*K+k].Float32()
								}
								refC[r*N+c] = sum
							}
						}
						outT := make([]float32, M*N)
						avx2SmallFloat16TransposedAsm(A, BT, 0, 1, M, N, K, outT)
						for i := range refC {
							if outT[i] != refC[i] {
								t.Fatalf("Float16 Transposed mismatch at M=%d, N=%d, K=%d, idx=%d: got %v, expected %v", M, N, K, i, outT[i], refC[i])
							}
						}

						B := make([]float16.Float16, K*N)
						for i := range B {
							B[i] = float16.FromFloat32(float32(i%17 - 8))
						}
						refNonT := make([]float32, M*N)
						for r := 0; r < M; r++ {
							for c := 0; c < N; c++ {
								var sum float32
								for k := 0; k < K; k++ {
									sum += A[r*K+k].Float32() * B[k*N+c].Float32()
								}
								refNonT[r*N+c] = sum
							}
						}
						outNonT := make([]float32, M*N)
						avx2SmallFloat16NonTransposedAsm(A, B, 0, 1, M, N, K, outNonT)
						for i := range refNonT {
							if outNonT[i] != refNonT[i] {
								t.Fatalf("Float16 NonTransposed mismatch at M=%d, N=%d, K=%d, idx=%d: got %v, expected %v", M, N, K, i, outNonT[i], refNonT[i])
							}
						}
					}
				}
			}
		})

		t.Run("BFloat16", func(t *testing.T) {
			for _, M := range []int{1, 3, 4, 7, 8, 15, 25} {
				for _, N := range []int{1, 2, 4, 7, 8, 15, 16, 20} {
					for _, K := range []int{1, 4, 15, 16, 17, 32} {
						A := make([]bfloat16.BFloat16, M*K)
						for i := range A {
							A[i] = bfloat16.FromFloat32(float32(i%13 - 6))
						}
						BT := make([]bfloat16.BFloat16, N*K)
						for i := range BT {
							BT[i] = bfloat16.FromFloat32(float32(i%17 - 8))
						}
						refC := make([]float32, M*N)
						for r := 0; r < M; r++ {
							for c := 0; c < N; c++ {
								var sum float32
								for k := 0; k < K; k++ {
									sum += A[r*K+k].Float32() * BT[c*K+k].Float32()
								}
								refC[r*N+c] = sum
							}
						}
						outT := make([]float32, M*N)
						avx2SmallBFloat16TransposedAsm(A, BT, 0, 1, M, N, K, outT)
						for i := range refC {
							if outT[i] != refC[i] {
								t.Fatalf("BFloat16 Transposed mismatch at M=%d, N=%d, K=%d, idx=%d: got %v, expected %v", M, N, K, i, outT[i], refC[i])
							}
						}

						B := make([]bfloat16.BFloat16, K*N)
						for i := range B {
							B[i] = bfloat16.FromFloat32(float32(i%17 - 8))
						}
						refNonT := make([]float32, M*N)
						for r := 0; r < M; r++ {
							for c := 0; c < N; c++ {
								var sum float32
								for k := 0; k < K; k++ {
									sum += A[r*K+k].Float32() * B[k*N+c].Float32()
								}
								refNonT[r*N+c] = sum
							}
						}
						outNonT := make([]float32, M*N)
						avx2SmallBFloat16NonTransposedAsm(A, B, 0, 1, M, N, K, outNonT)
						for i := range refNonT {
							if outNonT[i] != refNonT[i] {
								t.Fatalf("BFloat16 NonTransposed mismatch at M=%d, N=%d, K=%d, idx=%d: got %v, expected %v", M, N, K, i, outNonT[i], refNonT[i])
							}
						}
					}
				}
			}
		})
	})
}

// TestAVX2LargeParallelAndGoSIMDBounds tests multi-worker large GEMMs with unaligned row panels
// and multi-column-panel direct-output slices under both ASM and Go SIMD (AVX2UseAsm=false, where
// non-zero column tiles with rhsActiveCols < rhsCrossSize previously triggered a BCE bounds panic).
func TestAVX2LargeParallelAndGoSIMDBounds(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported on this architecture")
	}

	backend, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	shapes := []struct {
		name       string
		B, M, K, N int
	}{
		{"DirectOutputMultiColPanel_32x512x672", 1, 32, 512, 672},
		{"TimestralShape_36x512x672", 1, 36, 512, 672},
		{"UnalignedTail_50x400x540", 2, 50, 400, 540},
	}

	for _, useAsm := range []bool{true, false} {
		asmName := "Asm"
		if !useAsm {
			asmName = "GoSIMD"
		}
		t.Run(asmName, func(t *testing.T) {
			origUseAsm := AVX2UseAsm
			AVX2UseAsm = useAsm
			defer func() { AVX2UseAsm = origUseAsm }()

			for _, sc := range shapes {
				t.Run(sc.name, func(t *testing.T) {
					B, M, K, N := sc.B, sc.M, sc.K, sc.N
					lhsF32 := make([]float32, B*M*K)
					rhsF32 := make([]float32, B*K*N)
					for i := range lhsF32 {
						lhsF32[i] = float32((i%19)-9) * 0.125
					}
					for i := range rhsF32 {
						rhsF32[i] = float32((i%23)-11) * 0.125
					}

					backend.Scheduler.SetMaxParallelism(1)
					refOut := make([]float32, B*M*N)
					avx2LargeFloat32(backend, dot.LayoutNonTransposed, lhsF32, rhsF32, B, M, N, K, refOut, nil)

					for _, workers := range []int{2, 4} {
						backend.Scheduler.SetMaxParallelism(workers)
						got := make([]float32, B*M*N)
						for i := range got {
							got[i] = -77777.0
						}
						avx2LargeFloat32(backend, dot.LayoutNonTransposed, lhsF32, rhsF32, B, M, N, K, got, nil)
						for i, want := range refOut {
							if math.Abs(float64(got[i]-want)) > 1e-4 || math.IsNaN(float64(got[i])) {
								t.Fatalf("AVX2 Float32 mismatch (%s, workers=%d) at idx=%d: got %v, want %v",
									asmName, workers, i, got[i], want)
							}
						}
					}
				})
			}

			// Also verify BFloat16, Float16, and Float64 direct-output multi-col-panel bounds under GoSIMD/Asm.
			t.Run("AlternateDTypes_32x256x800", func(t *testing.T) {
				const M, K, N = 32, 256, 800 // N=800 > Nc=768 (BF16/F16) and Nc=256 (F64), with trailing cols divisible by Nr
				lhsBF16 := make([]bfloat16.BFloat16, M*K)
				rhsBF16 := make([]bfloat16.BFloat16, K*N)
				lhsF16 := make([]float16.Float16, M*K)
				rhsF16 := make([]float16.Float16, K*N)
				lhsF64 := make([]float64, M*K)
				rhsF64 := make([]float64, K*N)
				for i := range lhsF64 {
					v := float32((i%11)-5) * 0.25
					lhsBF16[i] = bfloat16.FromFloat32(v)
					lhsF16[i] = float16.FromFloat32(v)
					lhsF64[i] = float64(v)
				}
				for i := range rhsF64 {
					v := float32((i%13)-6) * 0.25
					rhsBF16[i] = bfloat16.FromFloat32(v)
					rhsF16[i] = float16.FromFloat32(v)
					rhsF64[i] = float64(v)
				}

				backend.Scheduler.SetMaxParallelism(2)
				outBF16 := make([]float32, M*N)
				outF16 := make([]float32, M*N)
				outF64 := make([]float64, M*N)
				avx2LargeBFloat16(backend, dot.LayoutNonTransposed, lhsBF16, rhsBF16, 1, M, N, K, outBF16, nil)
				avx2LargeFloat16(backend, dot.LayoutNonTransposed, lhsF16, rhsF16, 1, M, N, K, outF16, nil)
				avx2LargeFloat64(backend, dot.LayoutNonTransposed, lhsF64, rhsF64, 1, M, N, K, outF64, nil)
				for i := range M * N {
					if math.Abs(float64(outBF16[i])-outF64[i]) > 1e-3 || math.Abs(float64(outF16[i])-outF64[i]) > 1e-3 {
						t.Fatalf("AVX2 alternate dtype mismatch at idx=%d: bf16=%v, f16=%v, f64=%v",
							i, outBF16[i], outF16[i], outF64[i])
					}
				}
			})
		})
	}
}

func BenchmarkAVX2SmallMatMul(b *testing.B) {
	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	cases := []struct {
		name    string
		layout  dot.Layout
		M, K, N int
	}{
		{"NonTransposed/[128,4]x[4,1]", dot.LayoutNonTransposed, 128, 4, 1},
		{"NonTransposed/[128,69]x[69,4]", dot.LayoutNonTransposed, 128, 69, 4},
		{"NonTransposed/[25,4]x[4,1]", dot.LayoutNonTransposed, 25, 4, 1},
		{"NonTransposed/[25,69]x[69,4]", dot.LayoutNonTransposed, 25, 69, 4},
		{"NonTransposed/[49,4]x[4,1]", dot.LayoutNonTransposed, 49, 4, 1},
		{"NonTransposed/[49,69]x[69,4]", dot.LayoutNonTransposed, 49, 69, 4},

		{"Transposed/[128,4]x[1,4]", dot.LayoutTransposed, 128, 4, 1},
		{"Transposed/[128,69]x[4,69]", dot.LayoutTransposed, 128, 69, 4},
		{"Transposed/[25,4]x[1,4]", dot.LayoutTransposed, 25, 4, 1},
		{"Transposed/[25,69]x[4,69]", dot.LayoutTransposed, 25, 69, 4},
		{"Transposed/[49,4]x[1,4]", dot.LayoutTransposed, 49, 4, 1},
		{"Transposed/[49,69]x[4,69]", dot.LayoutTransposed, 49, 69, 4},
	}

	for _, tc := range cases {
		M, K, N := tc.M, tc.K, tc.N
		flops := float64(2 * M * N * K)
		lhs := make([]float32, M*K)
		var rhs []float32
		if tc.layout == dot.LayoutNonTransposed {
			rhs = make([]float32, K*N)
		} else {
			rhs = make([]float32, N*K)
		}
		out := make([]float32, M*N)

		b.Run(tc.name+"/Router", func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				avx2RouterFloat32(backend, tc.layout, lhs, rhs, 1, M, N, K, out, nil)
			}
			elapsed := b.Elapsed()
			if elapsed > 0 && b.N > 0 {
				gflops := (flops * float64(b.N) / elapsed.Seconds()) / 1e9
				durationPerOp := time.Duration(float64(elapsed) / float64(b.N))
				b.ReportMetric(gflops, "GFlops/s")
				b.ReportMetric(durationPerOp.Seconds()*1e6, "µs/op")
			}
		})

		if tc.layout == dot.LayoutNonTransposed {
			b.Run(tc.name+"/AsmNonTransposed", func(b *testing.B) {
				b.ResetTimer()
				for b.Loop() {
					avx2SmallFloat32NonTransposedAsm(lhs, rhs, 0, 1, M, N, K, out)
				}
				elapsed := b.Elapsed()
				if elapsed > 0 && b.N > 0 {
					gflops := (flops * float64(b.N) / elapsed.Seconds()) / 1e9
					durationPerOp := time.Duration(float64(elapsed) / float64(b.N))
					b.ReportMetric(gflops, "GFlops/s")
					b.ReportMetric(durationPerOp.Seconds()*1e6, "µs/op")
				}
			})
		} else {
			b.Run(tc.name+"/AsmTransposed", func(b *testing.B) {
				b.ResetTimer()
				for b.Loop() {
					avx2SmallFloat32TransposedAsm(lhs, rhs, 0, 1, M, N, K, out)
				}
				elapsed := b.Elapsed()
				if elapsed > 0 && b.N > 0 {
					gflops := (flops * float64(b.N) / elapsed.Seconds()) / 1e9
					durationPerOp := time.Duration(float64(elapsed) / float64(b.N))
					b.ReportMetric(gflops, "GFlops/s")
					b.ReportMetric(durationPerOp.Seconds()*1e6, "µs/op")
				}
			})
		}
	}
}

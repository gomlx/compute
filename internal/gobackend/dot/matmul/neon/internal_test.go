// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul/matmultest"
	"github.com/gomlx/compute/support/humanize"
)

func TestNEONInternal(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	t.Run("Pack", func(t *testing.T) {
		origUseAsm := NEONUseAsm
		defer func() { NEONUseAsm = origUseAsm }()

		for _, mode := range []struct {
			name   string
			useAsm bool
		}{
			{"GoSIMD", false},
			{"Asm", true},
		} {
			t.Run(mode.name, func(t *testing.T) {
				NEONUseAsm = mode.useAsm
				t.Run("Float32", func(t *testing.T) {
					matmultest.RunPackLHSTests(t, neonPackLHS[float32], 4)
					matmultest.RunPackLHSTests(t, neonPackLHS[float32], 8)
					matmultest.RunPackLHSTests(t, neonPackLHS[float32], 16)
					matmultest.RunPackLHSTests(t, neonPackLHS[float32], 32)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float32], 8)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float32], 12)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float32], 16)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float32], 32)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float32], 64)
					matmultest.RunApplyPackedOutputTests(t, neonApplyPackedOutputFloat32)
				})
				t.Run("BFloat16", func(t *testing.T) {
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[bfloat16.BFloat16], 4)
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[bfloat16.BFloat16], 8)
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[bfloat16.BFloat16], 16)
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[bfloat16.BFloat16], 32)
					matmultest.RunPackRHSTestsHalfPrecision(t, neonPackRHSNonTransposed[bfloat16.BFloat16], 8)
					matmultest.RunPackRHSTestsHalfPrecision(t, neonPackRHSNonTransposed[bfloat16.BFloat16], 16)
					matmultest.RunPackRHSTestsHalfPrecision(t, neonPackRHSNonTransposed[bfloat16.BFloat16], 32)
				})
				t.Run("Float16", func(t *testing.T) {
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[float16.Float16], 4)
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[float16.Float16], 8)
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[float16.Float16], 16)
					matmultest.RunPackLHSTestsHalfPrecision(t, neonPackLHS[float16.Float16], 32)
					matmultest.RunPackRHSTestsHalfPrecision(t, neonPackRHSNonTransposed[float16.Float16], 8)
					matmultest.RunPackRHSTestsHalfPrecision(t, neonPackRHSNonTransposed[float16.Float16], 16)
					matmultest.RunPackRHSTestsHalfPrecision(t, neonPackRHSNonTransposed[float16.Float16], 32)
				})
				t.Run("Float64", func(t *testing.T) {
					matmultest.RunPackLHSTests(t, neonPackLHS[float64], 4)
					matmultest.RunPackLHSTests(t, neonPackLHS[float64], 8)
					matmultest.RunPackLHSTests(t, neonPackLHS[float64], 16)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float64], 4)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float64], 8)
					matmultest.RunPackRHSTests(t, neonPackRHSNonTransposed[float64], 16)
					matmultest.RunApplyPackedOutputTests(t, neonApplyPackedOutputFloat64)
				})
			})
		}
	})

	t.Run("MicrokernelsFloat32", func(t *testing.T) {
		type kernelSpec struct {
			name string
			mr   int
			nr   int
			fn   func(packedLHS, packedRHS, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
		}
		specs := []kernelSpec{
			{"GoSIMD_4x16", 4, 16, neonLargeKernelFloat32},
			{"GoSIMD_8x8", 8, 8, neonLargeKernelFloat32_8x8},
			{"GoSIMD_8x12", 8, 12, neonLargeKernelFloat32_8x12},
			{"Asm_8x8", 8, 8, neonLargeKernelFloat32Asm},
			{"Asm_4x16", 4, 16, neonLargeKernelFloat32Asm_4x16},
			{"Asm_8x12", 8, 12, neonLargeKernelFloat32Asm_8x12},
		}

		for _, spec := range specs {
			t.Run(spec.name, func(t *testing.T) {
				for _, K := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 64} {
					M := spec.mr * 2 // test 2 row strips
					N := spec.nr * 2 // test 2 col strips
					A := make([]float32, M*K)
					for i := range A {
						A[i] = float32(i%13 - 6)
					}
					B := make([]float32, K*N)
					for i := range B {
						B[i] = float32(i%17 - 8)
					}
					refC := make([]float32, M*N)
					for r := range M {
						for c := range N {
							var sum float32
							for k := range K {
								sum += A[r*K+k] * B[k*N+c]
							}
							refC[r*N+c] = sum
						}
					}

					packedLHS := make([]float32, M*K)
					neonPackLHS(A, packedLHS, 0, 0, K, M, K, spec.mr)
					packedRHS := make([]float32, K*N)
					neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, spec.nr)

					out := make([]float32, M*N)
					spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
					for r := range M {
						for c := range N {
							if out[r*N+c] != refC[r*N+c] {
								t.Fatalf("%s K=%d overwrite mismatch at (%d,%d): got %v, want %v",
									spec.name, K, r, c, out[r*N+c], refC[r*N+c])
							}
						}
					}

					spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, true)
					for r := range M {
						for c := range N {
							if out[r*N+c] != 2*refC[r*N+c] {
								t.Fatalf("%s K=%d accumulate mismatch at (%d,%d): got %v, want %v",
									spec.name, K, r, c, out[r*N+c], 2*refC[r*N+c])
							}
						}
					}
				}
			})
		}
	})

	t.Run("MicrokernelsBFloat16", func(t *testing.T) {
		type kernelSpec struct {
			name      string
			mr        int
			nr        int
			needBFDOT bool
			fn        func(packedLHS, packedRHS []bfloat16.BFloat16, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
		}
		specs := []kernelSpec{
			{"GoSIMD_4x16", 4, 16, false, neonLargeKernelBFloat16},
			{"GoSIMD_8x8", 8, 8, false, neonLargeKernelBFloat16_8x8},
			{"Asm_8x8_SHLL", 8, 8, false, neonLargeKernelBFloat16Asm},
			{"Asm_4x16_SHLL", 4, 16, false, neonLargeKernelBFloat16Asm_4x16},
			{"Asm_8x8_BFDOT", 8, 8, true, neonLargeKernelBFloat16BFDOTAsm},
			{"Asm_4x16_BFDOT", 4, 16, true, neonLargeKernelBFloat16BFDOTAsm_4x16},
		}

		for _, spec := range specs {
			t.Run(spec.name, func(t *testing.T) {
				if spec.needBFDOT && !gobackend.IsARM64BF16Allowed {
					t.Skip("FEAT_BF16 is not supported or allowed on this host")
				}
				for _, K := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 64} {
					M := spec.mr * 2
					N := spec.nr * 2
					A := make([]bfloat16.BFloat16, M*K)
					for i := range A {
						A[i] = bfloat16.FromFloat32(float32(i%13-6) * 0.5)
					}
					B := make([]bfloat16.BFloat16, K*N)
					for i := range B {
						B[i] = bfloat16.FromFloat32(float32(i%17-8) * 0.5)
					}
					refC := make([]float32, M*N)
					for r := range M {
						for c := range N {
							var sum float32
							for k := range K {
								sum += A[r*K+k].Float32() * B[k*N+c].Float32()
							}
							refC[r*N+c] = sum
						}
					}

					packedLHS := make([]bfloat16.BFloat16, M*K)
					neonPackLHS(A, packedLHS, 0, 0, K, M, K, spec.mr)
					packedRHS := make([]bfloat16.BFloat16, K*N)
					neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, spec.nr)

					out := make([]float32, M*N)
					spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
					for r := range M {
						for c := range N {
							if math.Abs(float64(out[r*N+c]-refC[r*N+c])) > 1e-4 {
								t.Fatalf("%s K=%d overwrite mismatch at (%d,%d): got %v, want %v",
									spec.name, K, r, c, out[r*N+c], refC[r*N+c])
							}
						}
					}

					spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, true)
					for r := range M {
						for c := range N {
							if math.Abs(float64(out[r*N+c]-2*refC[r*N+c])) > 1e-4 {
								t.Fatalf("%s K=%d accumulate mismatch at (%d,%d): got %v, want %v",
									spec.name, K, r, c, out[r*N+c], 2*refC[r*N+c])
							}
						}
					}
				}
			})
		}
	})

	t.Run("MicrokernelsSME", func(t *testing.T) {
		if !gobackend.IsSMEAllowed {
			t.Skip("ARM64 SME is not supported or allowed on this host")
		}

		t.Run("Float32", func(t *testing.T) {
			specs := []struct {
				name   string
				mr, nr int
				fn     func(packedLHS, packedRHS, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
			}{
				{"SME_16x16", 16, 16, smeLargeKernelFloat32Asm_16x16},
				{"SME_16x64", 16, 64, smeLargeKernelFloat32Asm_16x64},
				{"SME_32x32", 32, 32, smeLargeKernelFloat32Asm_32x32},
			}
			for _, spec := range specs {
				t.Run(spec.name, func(t *testing.T) {
					for _, K := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 64} {
						for _, edge := range []struct {
							M, N int
						}{
							{spec.mr * 2, spec.nr * 2},
							{spec.mr + 5, spec.nr + 7}, // unaligned edge rows & cols (tests WHILELT predication)
							{3, 5},                     // sub-tile M, N
						} {
							M, N := edge.M, edge.N
							A := make([]float32, M*K)
							for i := range A {
								A[i] = float32(i%13-6) * 0.5
							}
							B := make([]float32, K*N)
							for i := range B {
								B[i] = float32(i%17-8) * 0.5
							}
							refC := make([]float32, M*N)
							for r := range M {
								for c := range N {
									var sum float32
									for k := range K {
										sum += A[r*K+k] * B[k*N+c]
									}
									refC[r*N+c] = sum
								}
							}
							numRowStrips := (M + spec.mr - 1) / spec.mr
							numColStrips := (N + spec.nr - 1) / spec.nr
							packedLHS := make([]float32, numRowStrips*spec.mr*K)
							neonPackLHS(A, packedLHS, 0, 0, K, M, K, spec.mr)
							packedRHS := make([]float32, numColStrips*spec.nr*K)
							neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, spec.nr)

							out := make([]float32, M*N)
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
							for r := range M {
								for c := range N {
									if math.Abs(float64(out[r*N+c]-refC[r*N+c])) > 1e-4 {
										t.Fatalf("%s M=%d N=%d K=%d overwrite mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], refC[r*N+c])
									}
								}
							}
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, true)
							for r := range M {
								for c := range N {
									if math.Abs(float64(out[r*N+c]-2*refC[r*N+c])) > 1e-4 {
										t.Fatalf("%s M=%d N=%d K=%d accumulate mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], 2*refC[r*N+c])
									}
								}
							}
						}
					}
				})
			}
		})

		t.Run("Float16", func(t *testing.T) {
			specs := []struct {
				name   string
				mr, nr int
				fn     func(packedLHS, packedRHS []float16.Float16, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
			}{
				{"SME_16x16", 16, 16, smeLargeKernelFloat16Asm_16x16},
				{"SME_32x32", 32, 32, smeLargeKernelFloat16Asm_32x32},
			}
			for _, spec := range specs {
				t.Run(spec.name, func(t *testing.T) {
					for _, K := range []int{1, 2, 3, 4, 7, 8, 17, 32} {
						for _, edge := range []struct {
							M, N int
						}{
							{spec.mr * 2, spec.nr * 2},
							{spec.mr + 5, spec.nr + 7},
						} {
							M, N := edge.M, edge.N
							A := make([]float16.Float16, M*K)
							for i := range A {
								A[i] = float16.FromFloat32(float32(i%13-6) * 0.5)
							}
							B := make([]float16.Float16, K*N)
							for i := range B {
								B[i] = float16.FromFloat32(float32(i%17-8) * 0.5)
							}
							refC := make([]float32, M*N)
							for r := range M {
								for c := range N {
									var sum float32
									for k := range K {
										sum += A[r*K+k].Float32() * B[k*N+c].Float32()
									}
									refC[r*N+c] = sum
								}
							}
							numRowStrips := (M + spec.mr - 1) / spec.mr
							numColStrips := (N + spec.nr - 1) / spec.nr
							packedLHS := make([]float16.Float16, numRowStrips*spec.mr*K)
							neonPackLHS(A, packedLHS, 0, 0, K, M, K, spec.mr)
							packedRHS := make([]float16.Float16, numColStrips*spec.nr*K)
							neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, spec.nr)

							out := make([]float32, M*N)
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
							for r := range M {
								for c := range N {
									if math.Abs(float64(out[r*N+c]-refC[r*N+c])) > 1e-4 {
										t.Fatalf("%s M=%d N=%d K=%d overwrite mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], refC[r*N+c])
									}
								}
							}
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, true)
							for r := range M {
								for c := range N {
									if math.Abs(float64(out[r*N+c]-2*refC[r*N+c])) > 1e-4 {
										t.Fatalf("%s M=%d N=%d K=%d accumulate mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], 2*refC[r*N+c])
									}
								}
							}
						}
					}
				})
			}
		})

		t.Run("BFloat16", func(t *testing.T) {
			specs := []struct {
				name   string
				mr, nr int
				fn     func(packedLHS, packedRHS []bfloat16.BFloat16, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
			}{
				{"SME_16x16", 16, 16, smeLargeKernelBFloat16Asm_16x16},
				{"SME_32x32", 32, 32, smeLargeKernelBFloat16Asm_32x32},
			}
			for _, spec := range specs {
				t.Run(spec.name, func(t *testing.T) {
					for _, K := range []int{1, 2, 3, 4, 7, 8, 17, 32} {
						for _, edge := range []struct {
							M, N int
						}{
							{spec.mr * 2, spec.nr * 2},
							{spec.mr + 5, spec.nr + 7},
						} {
							M, N := edge.M, edge.N
							A := make([]bfloat16.BFloat16, M*K)
							for i := range A {
								A[i] = bfloat16.FromFloat32(float32(i%13-6) * 0.5)
							}
							B := make([]bfloat16.BFloat16, K*N)
							for i := range B {
								B[i] = bfloat16.FromFloat32(float32(i%17-8) * 0.5)
							}
							refC := make([]float32, M*N)
							for r := range M {
								for c := range N {
									var sum float32
									for k := range K {
										sum += A[r*K+k].Float32() * B[k*N+c].Float32()
									}
									refC[r*N+c] = sum
								}
							}
							numRowStrips := (M + spec.mr - 1) / spec.mr
							numColStrips := (N + spec.nr - 1) / spec.nr
							packedLHS := make([]bfloat16.BFloat16, numRowStrips*spec.mr*K)
							neonPackLHS(A, packedLHS, 0, 0, K, M, K, spec.mr)
							packedRHS := make([]bfloat16.BFloat16, numColStrips*spec.nr*K)
							neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, spec.nr)

							out := make([]float32, M*N)
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
							for r := range M {
								for c := range N {
									if math.Abs(float64(out[r*N+c]-refC[r*N+c])) > 1e-4 {
										t.Fatalf("%s M=%d N=%d K=%d overwrite mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], refC[r*N+c])
									}
								}
							}
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, true)
							for r := range M {
								for c := range N {
									if math.Abs(float64(out[r*N+c]-2*refC[r*N+c])) > 1e-4 {
										t.Fatalf("%s M=%d N=%d K=%d accumulate mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], 2*refC[r*N+c])
									}
								}
							}
						}
					}
				})
			}
		})

		t.Run("Float64", func(t *testing.T) {
			specs := []struct {
				name   string
				mr, nr int
				fn     func(packedLHS, packedRHS, packedOutput []float64, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
			}{
				{"SME_8x8", 8, 8, smeLargeKernelFloat64Asm_8x8},
				{"SME_16x16", 16, 16, smeLargeKernelFloat64Asm_16x16},
			}
			for _, spec := range specs {
				t.Run(spec.name, func(t *testing.T) {
					for _, K := range []int{1, 2, 3, 4, 7, 8, 17, 32} {
						for _, edge := range []struct {
							M, N int
						}{
							{spec.mr * 2, spec.nr * 2},
							{spec.mr + 3, spec.nr + 5},
						} {
							M, N := edge.M, edge.N
							A := make([]float64, M*K)
							for i := range A {
								A[i] = float64(i%13-6) * 0.5
							}
							B := make([]float64, K*N)
							for i := range B {
								B[i] = float64(i%17-8) * 0.5
							}
							refC := make([]float64, M*N)
							for r := range M {
								for c := range N {
									var sum float64
									for k := range K {
										sum += A[r*K+k] * B[k*N+c]
									}
									refC[r*N+c] = sum
								}
							}
							numRowStrips := (M + spec.mr - 1) / spec.mr
							numColStrips := (N + spec.nr - 1) / spec.nr
							packedLHS := make([]float64, numRowStrips*spec.mr*K)
							neonPackLHS(A, packedLHS, 0, 0, K, M, K, spec.mr)
							packedRHS := make([]float64, numColStrips*spec.nr*K)
							neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, spec.nr)

							out := make([]float64, M*N)
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
							for r := range M {
								for c := range N {
									if math.Abs(out[r*N+c]-refC[r*N+c]) > 1e-9 {
										t.Fatalf("%s M=%d N=%d K=%d overwrite mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], refC[r*N+c])
									}
								}
							}
							spec.fn(packedLHS, packedRHS, out, M, N, K, M, N, true)
							for r := range M {
								for c := range N {
									if math.Abs(out[r*N+c]-2*refC[r*N+c]) > 1e-9 {
										t.Fatalf("%s M=%d N=%d K=%d accumulate mismatch at (%d,%d): got %v, want %v",
											spec.name, M, N, K, r, c, out[r*N+c], 2*refC[r*N+c])
									}
								}
							}
						}
					}
				})
			}
		})
	})

	t.Run("SmallMatMul", func(t *testing.T) {
		backend, err := gobackend.NewBackend()
		if err != nil {
			t.Fatalf("failed to create backend: %+v", err)
		}
		defer backend.Finalize()

		origUseAsm := NEONUseAsm
		origUseBFDOT := NEONUseBFDOT
		defer func() {
			NEONUseAsm = origUseAsm
			NEONUseBFDOT = origUseBFDOT
		}()

		testShapes := []struct {
			B, M, N, K int
		}{
			{1, 1, 1, 1},
			{1, 1, 1, 17},
			{1, 5, 1, 19},  // GEMV N=1
			{1, 7, 2, 15},  // Narrow N=2 < vecWidth
			{1, 6, 3, 23},  // Narrow N=3 < vecWidth
			{1, 3, 5, 11},  // M < 4, N=5
			{1, 4, 4, 8},   // Exact 4x4 tile
			{1, 5, 9, 13},  // Remainder M, N, K
			{1, 9, 19, 33}, // Multiple 4x16 + 4x8 + 4x4 + scalar cols, 8-unrolled K + remainder
			{2, 7, 17, 19}, // Batched
		}

		for _, mode := range []struct {
			name     string
			useAsm   bool
			useBFDOT bool
		}{
			{"GoSIMD", false, false},
			{"Asm_SHLL", true, false},
			{"Asm_BFDOT", true, true},
		} {
			t.Run(mode.name, func(t *testing.T) {
				if mode.useBFDOT && !gobackend.IsARM64BF16Allowed {
					t.Skip("FEAT_BF16 is not supported or allowed on this host")
				}
				NEONUseAsm = mode.useAsm
				NEONUseBFDOT = mode.useBFDOT

				for _, sh := range testShapes {
					shapeName := fmt.Sprintf("%dx%dx%dx%d", sh.B, sh.M, sh.N, sh.K)
					t.Run(shapeName, func(t *testing.T) {
						// Float32
						lhs32 := make([]float32, sh.B*sh.M*sh.K)
						for i := range lhs32 {
							lhs32[i] = float32(i%13-6) * 0.25
						}
						rhs32 := make([]float32, sh.B*sh.K*sh.N)
						for i := range rhs32 {
							rhs32[i] = float32(i%17-8) * 0.25
						}
						refNT := make([]float32, sh.B*sh.M*sh.N)
						refT := make([]float32, sh.B*sh.M*sh.N)
						for b := range sh.B {
							for r := range sh.M {
								for c := range sh.N {
									var sumNT, sumT float32
									for k := range sh.K {
										lVal := lhs32[(b*sh.M+r)*sh.K+k]
										sumNT += lVal * rhs32[(b*sh.K+k)*sh.N+c]
										sumT += lVal * rhs32[(b*sh.N+c)*sh.K+k]
									}
									refNT[(b*sh.M+r)*sh.N+c] = sumNT
									refT[(b*sh.M+r)*sh.N+c] = sumT
								}
							}
						}

						out32 := make([]float32, sh.B*sh.M*sh.N)
						neonSmallFloat32Parallel(backend, dot.LayoutNonTransposed, lhs32, rhs32, sh.B, sh.M, sh.N, sh.K, out32)
						for i := range out32 {
							if math.Abs(float64(out32[i]-refNT[i])) > 1e-4 {
								t.Fatalf("Float32 NonTransposed mismatch at %d: got %v, want %v", i, out32[i], refNT[i])
							}
						}
						clear(out32)
						neonSmallFloat32Parallel(backend, dot.LayoutTransposed, lhs32, rhs32, sh.B, sh.M, sh.N, sh.K, out32)
						for i := range out32 {
							if math.Abs(float64(out32[i]-refT[i])) > 1e-4 {
								t.Fatalf("Float32 Transposed mismatch at %d: got %v, want %v", i, out32[i], refT[i])
							}
						}

						// BFloat16
						lhsBF16 := make([]bfloat16.BFloat16, len(lhs32))
						for i, v := range lhs32 {
							lhsBF16[i] = bfloat16.FromFloat32(v)
						}
						rhsBF16 := make([]bfloat16.BFloat16, len(rhs32))
						for i, v := range rhs32 {
							rhsBF16[i] = bfloat16.FromFloat32(v)
						}
						clear(out32)
						neonSmallBFloat16Parallel(backend, dot.LayoutNonTransposed, lhsBF16, rhsBF16, sh.B, sh.M, sh.N, sh.K, out32)
						for i := range out32 {
							if math.Abs(float64(out32[i]-refNT[i])) > 1e-3 {
								t.Fatalf("BFloat16 NonTransposed mismatch at %d: got %v, want %v", i, out32[i], refNT[i])
							}
						}
						clear(out32)
						neonSmallBFloat16Parallel(backend, dot.LayoutTransposed, lhsBF16, rhsBF16, sh.B, sh.M, sh.N, sh.K, out32)
						for i := range out32 {
							if math.Abs(float64(out32[i]-refT[i])) > 1e-3 {
								t.Fatalf("BFloat16 Transposed mismatch at %d: got %v, want %v", i, out32[i], refT[i])
							}
						}

						// Float16
						lhsF16 := make([]float16.Float16, len(lhs32))
						for i, v := range lhs32 {
							lhsF16[i] = float16.FromFloat32(v)
						}
						rhsF16 := make([]float16.Float16, len(rhs32))
						for i, v := range rhs32 {
							rhsF16[i] = float16.FromFloat32(v)
						}
						clear(out32)
						neonSmallFloat16Parallel(backend, dot.LayoutNonTransposed, lhsF16, rhsF16, sh.B, sh.M, sh.N, sh.K, out32)
						for i := range out32 {
							if math.Abs(float64(out32[i]-refNT[i])) > 1e-3 {
								t.Fatalf("Float16 NonTransposed mismatch at %d: got %v, want %v", i, out32[i], refNT[i])
							}
						}
						clear(out32)
						neonSmallFloat16Parallel(backend, dot.LayoutTransposed, lhsF16, rhsF16, sh.B, sh.M, sh.N, sh.K, out32)
						for i := range out32 {
							if math.Abs(float64(out32[i]-refT[i])) > 1e-3 {
								t.Fatalf("Float16 Transposed mismatch at %d: got %v, want %v", i, out32[i], refT[i])
							}
						}

						// Float64
						lhs64 := make([]float64, len(lhs32))
						for i, v := range lhs32 {
							lhs64[i] = float64(v)
						}
						rhs64 := make([]float64, len(rhs32))
						for i, v := range rhs32 {
							rhs64[i] = float64(v)
						}
						out64 := make([]float64, sh.B*sh.M*sh.N)
						neonSmallFloat64Parallel(backend, dot.LayoutNonTransposed, lhs64, rhs64, sh.B, sh.M, sh.N, sh.K, out64)
						for i := range out64 {
							if math.Abs(out64[i]-float64(refNT[i])) > 1e-9 {
								t.Fatalf("Float64 NonTransposed mismatch at %d: got %v, want %v", i, out64[i], refNT[i])
							}
						}
						clear(out64)
						neonSmallFloat64Parallel(backend, dot.LayoutTransposed, lhs64, rhs64, sh.B, sh.M, sh.N, sh.K, out64)
						for i := range out64 {
							if math.Abs(out64[i]-float64(refT[i])) > 1e-9 {
								t.Fatalf("Float64 Transposed mismatch at %d: got %v, want %v", i, out64[i], refT[i])
							}
						}
					})
				}
			})
		}
	})
}

// BenchmarkMicrokernels benchmarks the raw L1 microkernel throughput across tile geometries
// (8x8, 4x16, 8x12) and implementations (Go archsimd vs Plan9 ARM64 assembly) on pre-packed panels.
func BenchmarkMicrokernels(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	// Choose M=32, N=48, K=256 so M is divisible by 4 and 8, and N is divisible by 8, 12, and 16.
	const M, N, K = 32, 48, 256
	flops := float64(2 * M * N * K)

	A := make([]float32, M*K)
	for i := range A {
		A[i] = float32(i%13-6) * 0.1
	}
	B := make([]float32, K*N)
	for i := range B {
		B[i] = float32(i%17-8) * 0.1
	}
	out := make([]float32, M*N)

	type kernelBench struct {
		name string
		mr   int
		nr   int
		fn   func(packedLHS, packedRHS, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}
	benches := []kernelBench{
		{"GoSIMD_4x16", 4, 16, neonLargeKernelFloat32},
		{"GoSIMD_8x8", 8, 8, neonLargeKernelFloat32_8x8},
		{"GoSIMD_8x12", 8, 12, neonLargeKernelFloat32_8x12},
		{"Asm_8x8", 8, 8, neonLargeKernelFloat32Asm},
		{"Asm_4x16", 4, 16, neonLargeKernelFloat32Asm_4x16},
		{"Asm_8x12", 8, 12, neonLargeKernelFloat32Asm_8x12},
	}

	for _, kb := range benches {
		b.Run("Float32/"+kb.name, func(b *testing.B) {
			packedLHS := make([]float32, M*K)
			neonPackLHS(A, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]float32, K*N)
			neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}

	// Float16 microkernels
	Af16 := make([]float16.Float16, M*K)
	for i := range Af16 {
		Af16[i] = float16.FromFloat32(A[i])
	}
	Bf16 := make([]float16.Float16, K*N)
	for i := range Bf16 {
		Bf16[i] = float16.FromFloat32(B[i])
	}
	f16Benches := []struct {
		name string
		mr   int
		nr   int
		fn   func(packedLHS, packedRHS []float16.Float16, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}{
		{"GoSIMD_4x16", 4, 16, neonLargeKernelFloat16},
		{"GoSIMD_8x8", 8, 8, neonLargeKernelFloat16_8x8},
		{"Asm_8x8", 8, 8, neonLargeKernelFloat16Asm},
		{"Asm_4x16", 4, 16, neonLargeKernelFloat16Asm_4x16},
	}
	for _, kb := range f16Benches {
		b.Run("Float16/"+kb.name, func(b *testing.B) {
			packedLHS := make([]float16.Float16, M*K)
			neonPackLHS(Af16, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]float16.Float16, K*N)
			neonPackRHSNonTransposed(Bf16, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}

	// BFloat16 microkernels
	Abf16 := make([]bfloat16.BFloat16, M*K)
	for i := range Abf16 {
		Abf16[i] = bfloat16.FromFloat32(A[i])
	}
	Bbf16 := make([]bfloat16.BFloat16, K*N)
	for i := range Bbf16 {
		Bbf16[i] = bfloat16.FromFloat32(B[i])
	}
	bf16Benches := []struct {
		name      string
		mr        int
		nr        int
		needBFDOT bool
		fn        func(packedLHS, packedRHS []bfloat16.BFloat16, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}{
		{"GoSIMD_4x16", 4, 16, false, neonLargeKernelBFloat16},
		{"GoSIMD_8x8", 8, 8, false, neonLargeKernelBFloat16_8x8},
		{"Asm_8x8_SHLL", 8, 8, false, neonLargeKernelBFloat16Asm},
		{"Asm_4x16_SHLL", 4, 16, false, neonLargeKernelBFloat16Asm_4x16},
		{"Asm_8x8_BFDOT", 8, 8, true, neonLargeKernelBFloat16BFDOTAsm},
		{"Asm_4x16_BFDOT", 4, 16, true, neonLargeKernelBFloat16BFDOTAsm_4x16},
	}
	for _, kb := range bf16Benches {
		b.Run("BFloat16/"+kb.name, func(b *testing.B) {
			if kb.needBFDOT && !gobackend.IsARM64BF16Allowed {
				b.Skip("FEAT_BF16 is not supported or allowed on this host")
			}
			packedLHS := make([]bfloat16.BFloat16, M*K)
			neonPackLHS(Abf16, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]bfloat16.BFloat16, K*N)
			neonPackRHSNonTransposed(Bbf16, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}

	// Float64 microkernels
	Af64 := make([]float64, M*K)
	for i := range Af64 {
		Af64[i] = float64(A[i])
	}
	Bf64 := make([]float64, K*N)
	for i := range Bf64 {
		Bf64[i] = float64(B[i])
	}
	out64 := make([]float64, M*N)
	f64Benches := []struct {
		name string
		mr   int
		nr   int
		fn   func(packedLHS, packedRHS, packedOutput []float64, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}{
		{"GoSIMD_4x8", 4, 8, neonLargeKernelFloat64},
		{"GoSIMD_8x4", 8, 4, neonLargeKernelFloat64_8x4},
		{"Asm_8x4", 8, 4, neonLargeKernelFloat64Asm},
		{"Asm_4x8", 4, 8, neonLargeKernelFloat64Asm_4x8},
	}
	for _, kb := range f64Benches {
		b.Run("Float64/"+kb.name, func(b *testing.B) {
			packedLHS := make([]float64, M*K)
			neonPackLHS(Af64, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]float64, K*N)
			neonPackRHSNonTransposed(Bf64, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out64, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out64, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}
}

// BenchmarkLargeGeometries benchmarks the full tiled GEMM across tile geometries and implementations.
func BenchmarkLargeGeometries(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	origUseAsm := NEONUseAsm
	origParams := NEONParamsFloat32
	defer func() {
		NEONUseAsm = origUseAsm
		NEONParamsFloat32 = origParams
	}()

	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	configs := []struct {
		name   string
		useAsm bool
		mr, nr int
		mc, nc int
	}{
		{"GoSIMD_4x16", false, 4, 16, 32, 512},
		{"GoSIMD_8x8", false, 8, 8, 32, 512},
		{"GoSIMD_8x12", false, 8, 12, 32, 504},
		{"Asm_8x8", true, 8, 8, 32, 512},
		{"Asm_4x16", true, 4, 16, 32, 512},
		{"Asm_8x12", true, 8, 12, 32, 504},
	}

	shapes := []struct {
		name       string
		B, M, N, K int
	}{
		{"Medium_1x128x128x128", 1, 128, 128, 128},
		{"Large-1_1x1536x1920x384", 1, 1536, 1920, 384},
		{"Large-3_1x2048x2048x2048", 1, 2048, 2048, 2048},
	}

	for _, sh := range shapes {
		lhs := make([]float32, sh.B*sh.M*sh.K)
		for i := range lhs {
			lhs[i] = float32(i%13-6) * 0.01
		}
		rhs := make([]float32, sh.B*sh.K*sh.N)
		for i := range rhs {
			rhs[i] = float32(i%17-8) * 0.01
		}
		output := make([]float32, sh.B*sh.M*sh.N)
		flops := float64(2 * sh.B * sh.M * sh.N * sh.K)

		for _, cfg := range configs {
			b.Run(sh.name+"/"+cfg.name, func(b *testing.B) {
				NEONUseAsm = cfg.useAsm
				NEONParamsFloat32 = matmul.CacheParams{
					LHSL1KernelRows:      cfg.mr,
					RHSL1KernelCols:      cfg.nr,
					PanelContractingSize: 256,
					LHSPanelCrossSize:    cfg.mc,
					RHSPanelCrossSize:    cfg.nc,
				}
				for range 3 {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}
}

// BenchmarkPack benchmarks neonPackLHS and neonPackRHSNonTransposed against matmul.UnsafePackLHS / UnsafePackRHS
// across both GoSIMD and Plan9 ARM64 assembly implementations.
func BenchmarkPack(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	origUseAsm := NEONUseAsm
	defer func() { NEONUseAsm = origUseAsm }()

	const totalRows, totalCols = 1536, 1920
	const panelRows, panelCols = 24, 128

	for _, kr := range []int{4, 8} {
		b.Run(fmt.Sprintf("PackLHS/Float32/kr=%d/Unsafe", kr), func(b *testing.B) {
			matmultest.RunBenchmarkPackLHS[float32](b, "float32", matmul.UnsafePackLHS[float32], totalRows, totalCols, panelRows, panelCols, kr)
		})
		b.Run(fmt.Sprintf("PackLHS/Float32/kr=%d/GoSIMD", kr), func(b *testing.B) {
			NEONUseAsm = false
			matmultest.RunBenchmarkPackLHS[float32](b, "float32", neonPackLHS[float32], totalRows, totalCols, panelRows, panelCols, kr)
		})
		b.Run(fmt.Sprintf("PackLHS/Float32/kr=%d/Asm", kr), func(b *testing.B) {
			NEONUseAsm = true
			matmultest.RunBenchmarkPackLHS[float32](b, "float32", neonPackLHS[float32], totalRows, totalCols, panelRows, panelCols, kr)
		})

		b.Run(fmt.Sprintf("PackLHS/BFloat16/kr=%d/Unsafe", kr), func(b *testing.B) {
			matmultest.RunBenchmarkPackLHS[bfloat16.BFloat16](b, "bfloat16", matmul.UnsafePackLHS[bfloat16.BFloat16], totalRows, totalCols, panelRows, panelCols, kr)
		})
		b.Run(fmt.Sprintf("PackLHS/BFloat16/kr=%d/GoSIMD", kr), func(b *testing.B) {
			NEONUseAsm = false
			matmultest.RunBenchmarkPackLHS[bfloat16.BFloat16](b, "bfloat16", neonPackLHS[bfloat16.BFloat16], totalRows, totalCols, panelRows, panelCols, kr)
		})
		b.Run(fmt.Sprintf("PackLHS/BFloat16/kr=%d/Asm", kr), func(b *testing.B) {
			NEONUseAsm = true
			matmultest.RunBenchmarkPackLHS[bfloat16.BFloat16](b, "bfloat16", neonPackLHS[bfloat16.BFloat16], totalRows, totalCols, panelRows, panelCols, kr)
		})

		b.Run(fmt.Sprintf("PackLHS/Float64/kr=%d/Unsafe", kr), func(b *testing.B) {
			matmultest.RunBenchmarkPackLHS[float64](b, "float64", matmul.UnsafePackLHS[float64], totalRows, totalCols, panelRows, panelCols, kr)
		})
		b.Run(fmt.Sprintf("PackLHS/Float64/kr=%d/GoSIMD", kr), func(b *testing.B) {
			NEONUseAsm = false
			matmultest.RunBenchmarkPackLHS[float64](b, "float64", neonPackLHS[float64], totalRows, totalCols, panelRows, panelCols, kr)
		})
		b.Run(fmt.Sprintf("PackLHS/Float64/kr=%d/Asm", kr), func(b *testing.B) {
			NEONUseAsm = true
			matmultest.RunBenchmarkPackLHS[float64](b, "float64", neonPackLHS[float64], totalRows, totalCols, panelRows, panelCols, kr)
		})
	}

	// PackRHS benchmarks on 1536 x 1920 (panel 256 x 512, matching NEONParamsFloat32):
	for _, kc := range []int{8, 16} {
		b.Run(fmt.Sprintf("PackRHS/Float32/kc=%d/Unsafe", kc), func(b *testing.B) {
			runBenchmarkPackRHS(b, matmul.UnsafePackRHS[float32], totalRows, totalCols, 256, 512, kc)
		})
		b.Run(fmt.Sprintf("PackRHS/Float32/kc=%d/GoSIMD", kc), func(b *testing.B) {
			NEONUseAsm = false
			runBenchmarkPackRHS(b, neonPackRHSNonTransposed[float32], totalRows, totalCols, 256, 512, kc)
		})
		b.Run(fmt.Sprintf("PackRHS/Float32/kc=%d/Asm", kc), func(b *testing.B) {
			NEONUseAsm = true
			runBenchmarkPackRHS(b, neonPackRHSNonTransposed[float32], totalRows, totalCols, 256, 512, kc)
		})
	}
}

func runBenchmarkPackRHS(b *testing.B, packFn matmultest.PackRHSFn[float32], contractingRows, rhsCols, panelContracting, panelCols, kernelCols int) {
	src := make([]float32, contractingRows*rhsCols)
	for i := range src {
		src[i] = float32(i)
	}
	numStrips := (panelCols + kernelCols - 1) / kernelCols
	dstSize := numStrips * panelContracting * kernelCols
	dst := make([]float32, dstSize)

	for range 3 {
		for colStart := 0; colStart < rhsCols; colStart += panelCols {
			copyCols := min(panelCols, rhsCols-colStart)
			for rowStart := 0; rowStart < contractingRows; rowStart += panelContracting {
				copyRows := min(panelContracting, contractingRows-rowStart)
				packFn(src, dst, rowStart, colStart, rhsCols, copyRows, copyCols, kernelCols)
			}
		}
	}
	for b.Loop() {
		for colStart := 0; colStart < rhsCols; colStart += panelCols {
			copyCols := min(panelCols, rhsCols-colStart)
			for rowStart := 0; rowStart < contractingRows; rowStart += panelContracting {
				copyRows := min(panelContracting, contractingRows-rowStart)
				packFn(src, dst, rowStart, colStart, rhsCols, copyRows, copyCols, kernelCols)
			}
		}
	}
	b.ReportMetric(humanize.DurationPerOp(b.Elapsed(), b.N))
}

// BenchmarkCacheGridSearch sweeps Kc, Mc, and Nc cache blocking parameters on ARM64 NEON.
func BenchmarkCacheGridSearch(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	origUseAsm := NEONUseAsm
	origParams := NEONParamsFloat32
	defer func() {
		NEONUseAsm = origUseAsm
		NEONParamsFloat32 = origParams
	}()
	NEONUseAsm = true

	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	shapes := []struct {
		name       string
		B, M, N, K int
	}{
		{"Transformer_416x1152x384", 1, 416, 1152, 384},
		{"Large-1_1536x1024x1920", 1, 1536, 1024, 1920},
		{"Large-3_2048x2048x2048", 1, 2048, 2048, 2048},
	}

	// 1. Sweep Kc (with Mc=32, Nc=512):
	for _, kc := range []int{128, 192, 256, 384, 512} {
		for _, sh := range shapes {
			lhs := make([]float32, sh.B*sh.M*sh.K)
			rhs := make([]float32, sh.B*sh.K*sh.N)
			output := make([]float32, sh.B*sh.M*sh.N)
			flops := float64(2 * sh.B * sh.M * sh.N * sh.K)

			b.Run(fmt.Sprintf("SweepKc/%s/Kc=%d", sh.name, kc), func(b *testing.B) {
				NEONParamsFloat32 = matmul.CacheParams{
					LHSL1KernelRows:      4,
					RHSL1KernelCols:      16,
					PanelContractingSize: kc,
					LHSPanelCrossSize:    32,
					RHSPanelCrossSize:    512,
				}
				for range 3 {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}

	// 2. Sweep Mc (with Kc=256, Nc=512):
	for _, mc := range []int{16, 32, 48, 64, 96, 128} {
		for _, sh := range shapes {
			lhs := make([]float32, sh.B*sh.M*sh.K)
			rhs := make([]float32, sh.B*sh.K*sh.N)
			output := make([]float32, sh.B*sh.M*sh.N)
			flops := float64(2 * sh.B * sh.M * sh.N * sh.K)

			b.Run(fmt.Sprintf("SweepMc/%s/Mc=%d", sh.name, mc), func(b *testing.B) {
				NEONParamsFloat32 = matmul.CacheParams{
					LHSL1KernelRows:      4,
					RHSL1KernelCols:      16,
					PanelContractingSize: 256,
					LHSPanelCrossSize:    mc,
					RHSPanelCrossSize:    512,
				}
				for range 3 {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}

	// 3. Sweep Nc (with Kc=256, Mc=32):
	for _, nc := range []int{128, 256, 384, 512, 768, 1024} {
		for _, sh := range shapes {
			lhs := make([]float32, sh.B*sh.M*sh.K)
			rhs := make([]float32, sh.B*sh.K*sh.N)
			output := make([]float32, sh.B*sh.M*sh.N)
			flops := float64(2 * sh.B * sh.M * sh.N * sh.K)

			b.Run(fmt.Sprintf("SweepNc/%s/Nc=%d", sh.name, nc), func(b *testing.B) {
				NEONParamsFloat32 = matmul.CacheParams{
					LHSL1KernelRows:      4,
					RHSL1KernelCols:      16,
					PanelContractingSize: 256,
					LHSPanelCrossSize:    32,
					RHSPanelCrossSize:    nc,
				}
				for range 3 {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}
}

// BenchmarkSmallMatMul compares SmallGoSIMD, SmallAsm, and LargeAsm across small and medium shapes
// to calibrate the Small vs Large crossover threshold and measure Stage 5 speedups.
func BenchmarkSmallMatMul(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	origUseAsm := NEONUseAsm
	defer func() { NEONUseAsm = origUseAsm }()

	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	shapes := []struct {
		name       string
		B, M, N, K int
	}{
		{"Tiny_16x16x16x16", 16, 16, 16, 16},
		{"GEMV_1x64x1x64", 1, 64, 1, 64},
		{"NarrowN_1x64x3x64", 1, 64, 3, 64},
		{"Small_1x16x16x64", 1, 16, 16, 64},
		{"Small_1x32x32x32", 1, 32, 32, 32},
		{"Small_1x48x48x48", 1, 48, 48, 48},
		{"Small_1x64x64x64", 1, 64, 64, 64},
		{"Medium_1x96x96x96", 1, 96, 96, 96},
		{"Medium_1x128x128x128", 1, 128, 128, 128},
		{"Medium_1x32x128x128", 1, 32, 128, 128},
		{"Medium_1x128x32x128", 1, 128, 32, 128},
	}

	for _, layout := range []struct {
		name   string
		layout dot.Layout
	}{
		{"NonTransposed", dot.LayoutNonTransposed},
		{"Transposed", dot.LayoutTransposed},
	} {
		for _, sh := range shapes {
			lhs := make([]float32, sh.B*sh.M*sh.K)
			for i := range lhs {
				lhs[i] = float32(i%13-6) * 0.01
			}
			rhs := make([]float32, sh.B*sh.K*sh.N)
			for i := range rhs {
				rhs[i] = float32(i%17-8) * 0.01
			}
			output := make([]float32, sh.B*sh.M*sh.N)
			flops := float64(2 * sh.B * sh.M * sh.N * sh.K)

			for _, impl := range []struct {
				name   string
				useAsm bool
				large  bool
			}{
				{"SmallGoSIMD", false, false},
				{"SmallAsm", true, false},
				{"LargeAsm", true, true},
			} {
				b.Run(fmt.Sprintf("%s/%s/%s", layout.name, sh.name, impl.name), func(b *testing.B) {
					NEONUseAsm = impl.useAsm
					runFn := func() {
						if impl.large {
							neonLargeFloat32(backend, layout.layout, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
						} else {
							neonSmallFloat32Parallel(backend, layout.layout, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output)
						}
					}
					for range 3 {
						runFn()
					}
					for b.Loop() {
						runFn()
					}
					elapsed := b.Elapsed()
					b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
					b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
				})
			}
		}
	}
}

// BenchmarkHalfPrecision benchmarks Float16 and BFloat16 (SHLL+FMLA vs BFDOT) on Small and Large shapes.
func BenchmarkHalfPrecision(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	origUseAsm := NEONUseAsm
	origUseBFDOT := NEONUseBFDOT
	defer func() {
		NEONUseAsm = origUseAsm
		NEONUseBFDOT = origUseBFDOT
	}()
	NEONUseAsm = true

	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	shapes := []struct {
		name       string
		B, M, N, K int
	}{
		{"Small_16x16x16x16", 16, 16, 16, 16},
		{"Small_1x32x32x64", 1, 32, 32, 64},
		{"Medium_1x128x128x128", 1, 128, 128, 128},
		{"Large_1x1536x1920x384", 1, 1536, 1920, 384},
	}

	for _, layout := range []struct {
		name   string
		layout dot.Layout
	}{
		{"NonTransposed", dot.LayoutNonTransposed},
		{"Transposed", dot.LayoutTransposed},
	} {
		for _, sh := range shapes {
			flops := float64(2 * sh.B * sh.M * sh.N * sh.K)
			lhsF16 := make([]float16.Float16, sh.B*sh.M*sh.K)
			rhsF16 := make([]float16.Float16, sh.B*sh.K*sh.N)
			lhsBF16 := make([]bfloat16.BFloat16, sh.B*sh.M*sh.K)
			rhsBF16 := make([]bfloat16.BFloat16, sh.B*sh.K*sh.N)
			for i := range lhsF16 {
				v := float32(i%13-6) * 0.01
				lhsF16[i] = float16.FromFloat32(v)
				lhsBF16[i] = bfloat16.FromFloat32(v)
			}
			for i := range rhsF16 {
				v := float32(i%17-8) * 0.01
				rhsF16[i] = float16.FromFloat32(v)
				rhsBF16[i] = bfloat16.FromFloat32(v)
			}
			output := make([]float32, sh.B*sh.M*sh.N)

			b.Run(fmt.Sprintf("%s/%s/Float16", layout.name, sh.name), func(b *testing.B) {
				for range 3 {
					neonRouterFloat16(backend, layout.layout, lhsF16, rhsF16, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonRouterFloat16(backend, layout.layout, lhsF16, rhsF16, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})

			b.Run(fmt.Sprintf("%s/%s/BFloat16_SHLL", layout.name, sh.name), func(b *testing.B) {
				NEONUseBFDOT = false
				for range 3 {
					neonRouterBFloat16(backend, layout.layout, lhsBF16, rhsBF16, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonRouterBFloat16(backend, layout.layout, lhsBF16, rhsBF16, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})

			b.Run(fmt.Sprintf("%s/%s/BFloat16_BFDOT", layout.name, sh.name), func(b *testing.B) {
				if !gobackend.IsARM64BF16Allowed {
					b.Skip("FEAT_BF16 is not supported or allowed on this host")
				}
				NEONUseBFDOT = true
				for range 3 {
					neonRouterBFloat16(backend, layout.layout, lhsBF16, rhsBF16, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonRouterBFloat16(backend, layout.layout, lhsBF16, rhsBF16, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}
}

// BenchmarkSMEMicrokernels benchmarks isolated SME outer-product (FMOPA) microkernels
// across register-sized 16x16 (1x1 ZA tile), 16x64 (1x4 ZA tiles), and 32x32 (2x2 ZA tiles)
// against NEON assembly microkernels on pre-packed L1 panels.
func BenchmarkSMEMicrokernels(b *testing.B) {
	if !gobackend.IsSMEAllowed {
		b.Skip("ARM64 SME is not supported or allowed on this host")
	}

	// M=64, N=64, K=256 is divisible by 4, 8, 16, 32, and 64.
	const M, N, K = 64, 64, 256
	flops := float64(2 * M * N * K)

	A := make([]float32, M*K)
	for i := range A {
		A[i] = float32(i%13-6) * 0.1
	}
	B := make([]float32, K*N)
	for i := range B {
		B[i] = float32(i%17-8) * 0.1
	}
	out := make([]float32, M*N)

	f32Benches := []struct {
		name   string
		mr, nr int
		fn     func(packedLHS, packedRHS, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}{
		{"NEON_Asm_4x16", 4, 16, neonLargeKernelFloat32Asm_4x16},
		{"NEON_Asm_8x8", 8, 8, neonLargeKernelFloat32Asm},
		{"SME_Asm_16x16", 16, 16, smeLargeKernelFloat32Asm_16x16},
		{"SME_Asm_16x64", 16, 64, smeLargeKernelFloat32Asm_16x64},
		{"SME_Asm_32x32", 32, 32, smeLargeKernelFloat32Asm_32x32},
	}
	for _, kb := range f32Benches {
		b.Run("Float32/"+kb.name, func(b *testing.B) {
			packedLHS := make([]float32, M*K)
			neonPackLHS(A, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]float32, K*N)
			neonPackRHSNonTransposed(B, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}

	Af16 := make([]float16.Float16, M*K)
	for i := range Af16 {
		Af16[i] = float16.FromFloat32(A[i])
	}
	Bf16 := make([]float16.Float16, K*N)
	for i := range Bf16 {
		Bf16[i] = float16.FromFloat32(B[i])
	}
	f16Benches := []struct {
		name   string
		mr, nr int
		fn     func(packedLHS, packedRHS []float16.Float16, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}{
		{"NEON_Asm_8x8", 8, 8, neonLargeKernelFloat16Asm},
		{"NEON_Asm_4x16", 4, 16, neonLargeKernelFloat16Asm_4x16},
		{"SME_Asm_16x16", 16, 16, smeLargeKernelFloat16Asm_16x16},
		{"SME_Asm_32x32", 32, 32, smeLargeKernelFloat16Asm_32x32},
	}
	for _, kb := range f16Benches {
		b.Run("Float16/"+kb.name, func(b *testing.B) {
			packedLHS := make([]float16.Float16, M*K)
			neonPackLHS(Af16, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]float16.Float16, K*N)
			neonPackRHSNonTransposed(Bf16, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}

	Abf16 := make([]bfloat16.BFloat16, M*K)
	for i := range Abf16 {
		Abf16[i] = bfloat16.FromFloat32(A[i])
	}
	Bbf16 := make([]bfloat16.BFloat16, K*N)
	for i := range Bbf16 {
		Bbf16[i] = bfloat16.FromFloat32(B[i])
	}
	bf16Benches := []struct {
		name   string
		mr, nr int
		fn     func(packedLHS, packedRHS []bfloat16.BFloat16, packedOutput []float32, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}{
		{"NEON_Asm_8x8_SHLL", 8, 8, neonLargeKernelBFloat16Asm},
		{"NEON_Asm_4x16_SHLL", 4, 16, neonLargeKernelBFloat16Asm_4x16},
		{"SME_Asm_16x16", 16, 16, smeLargeKernelBFloat16Asm_16x16},
		{"SME_Asm_32x32", 32, 32, smeLargeKernelBFloat16Asm_32x32},
	}
	for _, kb := range bf16Benches {
		b.Run("BFloat16/"+kb.name, func(b *testing.B) {
			packedLHS := make([]bfloat16.BFloat16, M*K)
			neonPackLHS(Abf16, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]bfloat16.BFloat16, K*N)
			neonPackRHSNonTransposed(Bbf16, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}

	Af64 := make([]float64, M*K)
	for i := range Af64 {
		Af64[i] = float64(A[i])
	}
	Bf64 := make([]float64, K*N)
	for i := range Bf64 {
		Bf64[i] = float64(B[i])
	}
	out64 := make([]float64, M*N)
	f64Benches := []struct {
		name   string
		mr, nr int
		fn     func(packedLHS, packedRHS, packedOutput []float64, lhsPanelRows, rhsPanelCols, contractingLen, lhsActiveRows, rhsActiveCols int, accumulate bool)
	}{
		{"NEON_Asm_8x4", 8, 4, neonLargeKernelFloat64Asm},
		{"NEON_Asm_4x8", 4, 8, neonLargeKernelFloat64Asm_4x8},
		{"SME_Asm_8x8", 8, 8, smeLargeKernelFloat64Asm_8x8},
		{"SME_Asm_16x16", 16, 16, smeLargeKernelFloat64Asm_16x16},
	}
	for _, kb := range f64Benches {
		b.Run("Float64/"+kb.name, func(b *testing.B) {
			packedLHS := make([]float64, M*K)
			neonPackLHS(Af64, packedLHS, 0, 0, K, M, K, kb.mr)
			packedRHS := make([]float64, K*N)
			neonPackRHSNonTransposed(Bf64, packedRHS, 0, 0, N, K, N, kb.nr)

			for range 3 {
				kb.fn(packedLHS, packedRHS, out64, M, N, K, M, N, false)
			}
			for b.Loop() {
				kb.fn(packedLHS, packedRHS, out64, M, N, K, M, N, false)
			}
			elapsed := b.Elapsed()
			b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
			b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
		})
	}
}

// BenchmarkSMEGeometries benchmarks full tiled GEMM across NEON and SME tile geometries (16x16, 16x64, 32x32).
func BenchmarkSMEGeometries(b *testing.B) {
	if !gobackend.IsSMEAllowed {
		b.Skip("ARM64 SME is not supported or allowed on this host")
	}

	origUseAsm := NEONUseAsm
	origParams32 := NEONParamsFloat32
	origParamsF16 := NEONParamsFloat16
	origParamsBF16 := NEONParamsBFloat16
	origParams64 := NEONParamsFloat64
	defer func() {
		NEONUseAsm = origUseAsm
		NEONParamsFloat32 = origParams32
		NEONParamsFloat16 = origParamsF16
		NEONParamsBFloat16 = origParamsBF16
		NEONParamsFloat64 = origParams64
	}()
	NEONUseAsm = true

	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	shapes := []struct {
		name       string
		B, M, N, K int
	}{
		{"Small_1x64x64x64", 1, 64, 64, 64},
		{"Medium_1x128x128x128", 1, 128, 128, 128},
		{"Medium_1x192x192x192", 1, 192, 192, 192},
		{"Medium_1x256x256x256", 1, 256, 256, 256},
		{"Transformer_1x416x1152x384", 1, 416, 1152, 384},
		{"Large-1_1x1536x1920x384", 1, 1536, 1920, 384},
		{"Large-3_1x2048x2048x2048", 1, 2048, 2048, 2048},
	}

	f32Configs := []struct {
		name       string
		mr, nr     int
		kc, mc, nc int
	}{
		{"NEON_4x16", 4, 16, 256, 48, 512},
		{"SME_16x16", 16, 16, 256, 64, 512},
		{"SME_16x64", 16, 64, 256, 64, 512},
		{"SME_32x32_Mc64", 32, 32, 256, 64, 512},
		{"SME_32x32_Mc128", 32, 32, 256, 128, 512},
	}

	for _, sh := range shapes {
		lhs := make([]float32, sh.B*sh.M*sh.K)
		for i := range lhs {
			lhs[i] = float32(i%13-6) * 0.01
		}
		rhs := make([]float32, sh.B*sh.K*sh.N)
		for i := range rhs {
			rhs[i] = float32(i%17-8) * 0.01
		}
		output := make([]float32, sh.B*sh.M*sh.N)
		flops := float64(2 * sh.B * sh.M * sh.N * sh.K)

		for _, cfg := range f32Configs {
			b.Run("Float32/"+sh.name+"/"+cfg.name, func(b *testing.B) {
				NEONParamsFloat32 = matmul.CacheParams{
					LHSL1KernelRows:      cfg.mr,
					RHSL1KernelCols:      cfg.nr,
					PanelContractingSize: cfg.kc,
					LHSPanelCrossSize:    cfg.mc,
					RHSPanelCrossSize:    cfg.nc,
				}
				for range 3 {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				for b.Loop() {
					neonLargeFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}

	// Half-precision and Float64 comparison on Large-1 (1x1536x1920x384):
	{
		const B, M, N, K = 1, 1536, 1920, 384
		flops := float64(2 * B * M * N * K)
		lhsF16 := make([]float16.Float16, B*M*K)
		rhsF16 := make([]float16.Float16, B*K*N)
		lhsBF16 := make([]bfloat16.BFloat16, B*M*K)
		rhsBF16 := make([]bfloat16.BFloat16, B*K*N)
		lhs64 := make([]float64, B*M*K)
		rhs64 := make([]float64, B*K*N)
		output := make([]float32, B*M*N)
		output64 := make([]float64, B*M*N)

		for _, cfg := range []struct {
			name       string
			mr, nr     int
			kc, mc, nc int
		}{
			{"NEON_8x8", 8, 8, 256, 48, 512},
			{"SME_16x16", 16, 16, 256, 64, 512},
			{"SME_32x32", 32, 32, 256, 64, 512},
		} {
			b.Run("Float16/Large-1_1x1536x1920x384/"+cfg.name, func(b *testing.B) {
				NEONParamsFloat16 = matmul.CacheParams{
					LHSL1KernelRows:      cfg.mr,
					RHSL1KernelCols:      cfg.nr,
					PanelContractingSize: cfg.kc,
					LHSPanelCrossSize:    cfg.mc,
					RHSPanelCrossSize:    cfg.nc,
				}
				for range 3 {
					neonLargeFloat16(backend, dot.LayoutNonTransposed, lhsF16, rhsF16, B, M, N, K, output, nil)
				}
				for b.Loop() {
					neonLargeFloat16(backend, dot.LayoutNonTransposed, lhsF16, rhsF16, B, M, N, K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
			b.Run("BFloat16/Large-1_1x1536x1920x384/"+cfg.name, func(b *testing.B) {
				NEONParamsBFloat16 = matmul.CacheParams{
					LHSL1KernelRows:      cfg.mr,
					RHSL1KernelCols:      cfg.nr,
					PanelContractingSize: cfg.kc,
					LHSPanelCrossSize:    cfg.mc,
					RHSPanelCrossSize:    cfg.nc,
				}
				for range 3 {
					neonLargeBFloat16(backend, dot.LayoutNonTransposed, lhsBF16, rhsBF16, B, M, N, K, output, nil)
				}
				for b.Loop() {
					neonLargeBFloat16(backend, dot.LayoutNonTransposed, lhsBF16, rhsBF16, B, M, N, K, output, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}

		for _, cfg := range []struct {
			name       string
			mr, nr     int
			kc, mc, nc int
		}{
			{"NEON_8x4", 8, 4, 128, 16, 256},
			{"SME_8x8", 8, 8, 128, 32, 256},
			{"SME_16x16", 16, 16, 128, 64, 256},
		} {
			b.Run("Float64/Large-1_1x1536x1920x384/"+cfg.name, func(b *testing.B) {
				NEONParamsFloat64 = matmul.CacheParams{
					LHSL1KernelRows:      cfg.mr,
					RHSL1KernelCols:      cfg.nr,
					PanelContractingSize: cfg.kc,
					LHSPanelCrossSize:    cfg.mc,
					RHSPanelCrossSize:    cfg.nc,
				}
				for range 3 {
					neonLargeFloat64(backend, dot.LayoutNonTransposed, lhs64, rhs64, B, M, N, K, output64, nil)
				}
				for b.Loop() {
					neonLargeFloat64(backend, dot.LayoutNonTransposed, lhs64, rhs64, B, M, N, K, output64, nil)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}
}

// TestSMEHybridCoExecution verifies that hybrid SME+NEON co-execution (SMEUseHybridNEON=true),
// neonLargeKernelFromSME* strided fallback microkernels (including non-multiple edge tails),
// and concurrent inter-call matmuls competing for globalSMEClusterLocks produce exact results.
func TestSMEHybridCoExecution(t *testing.T) {
	if !gobackend.IsSMEAllowed {
		t.Skip("ARM64 SME is not supported or allowed on this host")
	}

	origHybrid := SMEUseHybridNEON
	origPerCluster := SMEWorkersPerCluster
	defer func() {
		SMEUseHybridNEON = origHybrid
		SMEWorkersPerCluster = origPerCluster
	}()

	backend, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	// 1. Test non-multiple edge dimensions (M=301, N=531, K=263) across all 4 dtypes with:
	//    - Pure SME (SMEUseHybridNEON=false)
	//    - Hybrid SME+NEON (SMEUseHybridNEON=true, SMEWorkersPerCluster=1)
	//    - Pure Strided NEON on SME-packed panels (SMEUseHybridNEON=true, SMEWorkersPerCluster=0)
	const B, M, N, K = 1, 301, 531, 263
	lhs32 := make([]float32, B*M*K)
	rhs32 := make([]float32, B*K*N)
	lhsF16 := make([]float16.Float16, B*M*K)
	rhsF16 := make([]float16.Float16, B*K*N)
	lhsBF16 := make([]bfloat16.BFloat16, B*M*K)
	rhsBF16 := make([]bfloat16.BFloat16, B*K*N)
	lhs64 := make([]float64, B*M*K)
	rhs64 := make([]float64, B*K*N)
	for i := range lhs32 {
		v := float32(i%19-9) * 0.05
		lhs32[i] = v
		lhsF16[i] = float16.FromFloat32(v)
		lhsBF16[i] = bfloat16.FromFloat32(v)
		lhs64[i] = float64(v)
	}
	for i := range rhs32 {
		v := float32(i%23-11) * 0.05
		rhs32[i] = v
		rhsF16[i] = float16.FromFloat32(v)
		rhsBF16[i] = bfloat16.FromFloat32(v)
		rhs64[i] = float64(v)
	}

	SMEUseHybridNEON = false
	want32 := make([]float32, B*M*N)
	wantF16 := make([]float32, B*M*N)
	wantBF16 := make([]float32, B*M*N)
	want64 := make([]float64, B*M*N)
	neonLargeWithParamsFloat32(backend, dot.LayoutNonTransposed, lhs32, rhs32, B, M, N, K, want32, nil, SMEParamsFloat32)
	neonLargeWithParamsFloat16(backend, dot.LayoutNonTransposed, lhsF16, rhsF16, B, M, N, K, wantF16, nil, SMEParamsFloat16)
	neonLargeWithParamsBFloat16(backend, dot.LayoutNonTransposed, lhsBF16, rhsBF16, B, M, N, K, wantBF16, nil, SMEParamsBFloat16)
	neonLargeWithParamsFloat64(backend, dot.LayoutNonTransposed, lhs64, rhs64, B, M, N, K, want64, nil, SMEParamsFloat64)

	for _, workersPerCluster := range []int32{1, 0} {
		SMEUseHybridNEON = true
		SMEWorkersPerCluster = workersPerCluster

		got32 := make([]float32, B*M*N)
		gotF16 := make([]float32, B*M*N)
		gotBF16 := make([]float32, B*M*N)
		got64 := make([]float64, B*M*N)
		neonLargeWithParamsFloat32(backend, dot.LayoutNonTransposed, lhs32, rhs32, B, M, N, K, got32, nil, SMEParamsFloat32)
		neonLargeWithParamsFloat16(backend, dot.LayoutNonTransposed, lhsF16, rhsF16, B, M, N, K, gotF16, nil, SMEParamsFloat16)
		neonLargeWithParamsBFloat16(backend, dot.LayoutNonTransposed, lhsBF16, rhsBF16, B, M, N, K, gotBF16, nil, SMEParamsBFloat16)
		neonLargeWithParamsFloat64(backend, dot.LayoutNonTransposed, lhs64, rhs64, B, M, N, K, got64, nil, SMEParamsFloat64)

		for i := range want32 {
			if diff := math.Abs(float64(got32[i] - want32[i])); diff > 1e-4 {
				t.Fatalf("Float32 mismatch at %d (workersPerCluster=%d): got %v, want %v", i, workersPerCluster, got32[i], want32[i])
			}
			if diff := math.Abs(float64(gotF16[i] - wantF16[i])); diff > 1e-3 {
				t.Fatalf("Float16 mismatch at %d (workersPerCluster=%d): got %v, want %v", i, workersPerCluster, gotF16[i], wantF16[i])
			}
			if diff := math.Abs(float64(gotBF16[i] - wantBF16[i])); diff > 1e-3 {
				t.Fatalf("BFloat16 mismatch at %d (workersPerCluster=%d): got %v, want %v", i, workersPerCluster, gotBF16[i], wantBF16[i])
			}
			if diff := math.Abs(got64[i] - want64[i]); diff > 1e-9 {
				t.Fatalf("Float64 mismatch at %d (workersPerCluster=%d): got %v, want %v", i, workersPerCluster, got64[i], want64[i])
			}
		}
	}

	// 2. Test concurrent inter-op / inter-call contention across 4 parallel matmuls competing for globalSMEClusterLocks:
	SMEUseHybridNEON = true
	SMEWorkersPerCluster = 1
	const numConcurrent = 4
	errCh := make(chan error, numConcurrent)
	for range numConcurrent {
		go func() {
			got := make([]float32, B*M*N)
			for range 5 {
				neonLargeWithParamsFloat32(backend, dot.LayoutNonTransposed, lhs32, rhs32, B, M, N, K, got, nil, SMEParamsFloat32)
				for i := range want32 {
					if math.Abs(float64(got[i]-want32[i])) > 1e-4 {
						errCh <- fmt.Errorf("concurrent Float32 mismatch at %d: got %v, want %v", i, got[i], want32[i])
						return
					}
				}
			}
			errCh <- nil
		}()
	}
	for range numConcurrent {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}

// BenchmarkSMEHybrid compares Pure SME (SMEUseHybridNEON=false, all workers queuing on cluster SME units)
// against Hybrid SME+NEON (SMEUseHybridNEON=true, 1 SME worker per cluster + remaining workers on per-core NEON).
func BenchmarkSMEHybrid(b *testing.B) {
	if !gobackend.IsSMEAllowed {
		b.Skip("ARM64 SME is not supported or allowed on this host")
	}

	origHybrid := SMEUseHybridNEON
	defer func() { SMEUseHybridNEON = origHybrid }()

	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	shapes := []struct {
		name       string
		B, M, N, K int
		cachedRHS  bool
	}{
		{"Dynamic_1x1024x1024x1024", 1, 1024, 1024, 1024, false},
		{"CachedRHS_1x1024x1024x1024", 1, 1024, 1024, 1024, true},
		{"Dynamic_1x1536x1920x384", 1, 1536, 1920, 384, false},
		{"CachedRHS_1x1536x1920x384", 1, 1536, 1920, 384, true},
		{"Dynamic_1x2048x2048x2048", 1, 2048, 2048, 2048, false},
		{"CachedRHS_1x2048x2048x2048", 1, 2048, 2048, 2048, true},
	}

	for _, sh := range shapes {
		lhs := make([]float32, sh.B*sh.M*sh.K)
		for i := range lhs {
			lhs[i] = float32(i%13-6) * 0.01
		}
		rhs := make([]float32, sh.B*sh.K*sh.N)
		for i := range rhs {
			rhs[i] = float32(i%17-8) * 0.01
		}
		output := make([]float32, sh.B*sh.M*sh.N)
		flops := float64(2 * sh.B * sh.M * sh.N * sh.K)

		var nodeData *dot.NodeData
		if sh.cachedRHS {
			nodeData = &dot.NodeData{
				RHSConstantData: &gobackend.ConstantData{},
			}
		}

		for _, mode := range []struct {
			name   string
			hybrid bool
		}{
			{"PureSME", false},
			{"HybridSME_NEON", true},
		} {
			b.Run("Float32/"+sh.name+"/"+mode.name, func(b *testing.B) {
				SMEUseHybridNEON = mode.hybrid
				for range 3 {
					smeRouterFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nodeData)
				}
				for b.Loop() {
					smeRouterFloat32(backend, dot.LayoutNonTransposed, lhs, rhs, sh.B, sh.M, sh.N, sh.K, output, nodeData)
				}
				elapsed := b.Elapsed()
				b.ReportMetric(flops*float64(b.N)/elapsed.Seconds()/1e9, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			})
		}
	}
}

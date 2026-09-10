// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"testing"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func TestAVX2BinaryTrailingCorrectness(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 not allowed or not supported on this host")
	}

	testCases := []struct {
		A, B int
	}{
		{A: 1, B: 1},
		{A: 1, B: 2},
		{A: 2, B: 3},
		{A: 3, B: 4},
		{A: 4, B: 7},
		{A: 5, B: 8},
		{A: 7, B: 9},
		{A: 10, B: 15},
		{A: 11, B: 16},
		{A: 13, B: 17},
		{A: 20, B: 31},
		{A: 25, B: 32},
		{A: 30, B: 33},
		{A: 50, B: 64},
		{A: 50, B: 65},
		{A: 100, B: 128},
	}

	ops := []compute.OpType{
		compute.OpTypeAdd,
		compute.OpTypeSub,
		compute.OpTypeMul,
		compute.OpTypeDiv,
		compute.OpTypeMax,
		compute.OpTypeMin,
	}

	// 1. Float32
	t.Run("Float32", func(t *testing.T) {
		for _, op := range ops {
			for _, isLHS := range []bool{false, true} {
				for _, tc := range testCases {
					A, B := tc.A, tc.B
					lhs := make([]float32, A*B)
					rhs := make([]float32, A)
					if isLHS {
						lhs = make([]float32, A)
						rhs = make([]float32, A*B)
					}
					out := make([]float32, A*B)
					expected := make([]float32, A*B)

					for i := range lhs {
						lhs[i] = float32(i%100 + 1)
					}
					for i := range rhs {
						rhs[i] = float32(i%10 + 1)
					}

					// Compute expected
					for a := 0; a < A; a++ {
						for b := 0; b < B; b++ {
							idx := a*B + b
							var lVal, rVal float32
							if !isLHS {
								lVal = lhs[idx]
								rVal = rhs[a]
							} else {
								lVal = lhs[a]
								rVal = rhs[idx]
							}
							var res float32
							switch op {
							case compute.OpTypeAdd:
								res = lVal + rVal
							case compute.OpTypeSub:
								res = lVal - rVal
							case compute.OpTypeMul:
								res = lVal * rVal
							case compute.OpTypeDiv:
								res = lVal / rVal
							case compute.OpTypeMax:
								res = max(lVal, rVal)
							case compute.OpTypeMin:
								res = min(lVal, rVal)
							}
							expected[idx] = res
						}
					}

					ok := DispatchBinaryTrailingAVX2(op, isLHS, unsafe.Pointer(&lhs[0]), unsafe.Pointer(&rhs[0]), unsafe.Pointer(&out[0]), A, B, dtypes.Float32)
					if !ok {
						t.Fatalf("DispatchBinaryTrailingAVX2 returned false for %s f32", op)
					}

					for idx := 0; idx < A*B; idx++ {
						diff := math.Abs(float64(out[idx] - expected[idx]))
						if diff > 1e-4 {
							t.Fatalf("Mismatch for %s (isLHS=%v, A=%d, B=%d) at idx %d: got %v, expected %v",
								op, isLHS, A, B, idx, out[idx], expected[idx])
						}
					}
				}
			}
		}
	})

	// 2. Float64
	t.Run("Float64", func(t *testing.T) {
		for _, op := range ops {
			for _, isLHS := range []bool{false, true} {
				for _, tc := range testCases {
					A, B := tc.A, tc.B
					lhs := make([]float64, A*B)
					rhs := make([]float64, A)
					if isLHS {
						lhs = make([]float64, A)
						rhs = make([]float64, A*B)
					}
					out := make([]float64, A*B)
					expected := make([]float64, A*B)

					for i := range lhs {
						lhs[i] = float64(i%100 + 1)
					}
					for i := range rhs {
						rhs[i] = float64(i%10 + 1)
					}

					for a := 0; a < A; a++ {
						for b := 0; b < B; b++ {
							idx := a*B + b
							var lVal, rVal float64
							if !isLHS {
								lVal = lhs[idx]
								rVal = rhs[a]
							} else {
								lVal = lhs[a]
								rVal = rhs[idx]
							}
							var res float64
							switch op {
							case compute.OpTypeAdd:
								res = lVal + rVal
							case compute.OpTypeSub:
								res = lVal - rVal
							case compute.OpTypeMul:
								res = lVal * rVal
							case compute.OpTypeDiv:
								res = lVal / rVal
							case compute.OpTypeMax:
								res = max(lVal, rVal)
							case compute.OpTypeMin:
								res = min(lVal, rVal)
							}
							expected[idx] = res
						}
					}

					ok := DispatchBinaryTrailingAVX2(op, isLHS, unsafe.Pointer(&lhs[0]), unsafe.Pointer(&rhs[0]), unsafe.Pointer(&out[0]), A, B, dtypes.Float64)
					if !ok {
						t.Fatalf("DispatchBinaryTrailingAVX2 returned false for %s f64", op)
					}

					for idx := 0; idx < A*B; idx++ {
						diff := math.Abs(out[idx] - expected[idx])
						if diff > 1e-6 {
							t.Fatalf("Mismatch for %s (isLHS=%v, A=%d, B=%d) at idx %d: got %v, expected %v",
								op, isLHS, A, B, idx, out[idx], expected[idx])
						}
					}
				}
			}
		}
	})

	// 3. Int32
	t.Run("Int32", func(t *testing.T) {
		intOps := []compute.OpType{
			compute.OpTypeAdd,
			compute.OpTypeSub,
			compute.OpTypeMul,
			compute.OpTypeMax,
			compute.OpTypeMin,
		}
		for _, op := range intOps {
			for _, isLHS := range []bool{false, true} {
				for _, tc := range testCases {
					A, B := tc.A, tc.B
					lhs := make([]int32, A*B)
					rhs := make([]int32, A)
					if isLHS {
						lhs = make([]int32, A)
						rhs = make([]int32, A*B)
					}
					out := make([]int32, A*B)
					expected := make([]int32, A*B)

					for i := range lhs {
						lhs[i] = int32((i % 100) - 50)
					}
					for i := range rhs {
						rhs[i] = int32((i % 10) - 5)
					}

					for a := 0; a < A; a++ {
						for b := 0; b < B; b++ {
							idx := a*B + b
							var lVal, rVal int32
							if !isLHS {
								lVal = lhs[idx]
								rVal = rhs[a]
							} else {
								lVal = lhs[a]
								rVal = rhs[idx]
							}
							var res int32
							switch op {
							case compute.OpTypeAdd:
								res = lVal + rVal
							case compute.OpTypeSub:
								res = lVal - rVal
							case compute.OpTypeMul:
								res = lVal * rVal
							case compute.OpTypeMax:
								res = max(lVal, rVal)
							case compute.OpTypeMin:
								res = min(lVal, rVal)
							}
							expected[idx] = res
						}
					}

					ok := DispatchBinaryTrailingAVX2(op, isLHS, unsafe.Pointer(&lhs[0]), unsafe.Pointer(&rhs[0]), unsafe.Pointer(&out[0]), A, B, dtypes.Int32)
					if !ok {
						t.Fatalf("DispatchBinaryTrailingAVX2 returned false for %s int32", op)
					}

					for idx := 0; idx < A*B; idx++ {
						if out[idx] != expected[idx] {
							t.Fatalf("Mismatch for %s (isLHS=%v, A=%d, B=%d) at idx %d: got %v, expected %v",
								op, isLHS, A, B, idx, out[idx], expected[idx])
						}
					}
				}
			}
		}
	})

	// 3b. Uint32
	t.Run("Uint32", func(t *testing.T) {
		uintOps := []compute.OpType{
			compute.OpTypeAdd,
			compute.OpTypeSub,
			compute.OpTypeMul,
			compute.OpTypeMax,
			compute.OpTypeMin,
		}
		for _, op := range uintOps {
			for _, isLHS := range []bool{false, true} {
				for _, tc := range testCases {
					A, B := tc.A, tc.B
					lhs := make([]uint32, A*B)
					rhs := make([]uint32, A)
					if isLHS {
						lhs = make([]uint32, A)
						rhs = make([]uint32, A*B)
					}
					out := make([]uint32, A*B)
					expected := make([]uint32, A*B)

					for i := range lhs {
						lhs[i] = uint32(i*17 + 3)
					}
					for i := range rhs {
						rhs[i] = uint32(i*31 + 5)
					}

					for a := 0; a < A; a++ {
						for b := 0; b < B; b++ {
							idx := a*B + b
							var lVal, rVal uint32
							if !isLHS {
								lVal = lhs[idx]
								rVal = rhs[a]
							} else {
								lVal = lhs[a]
								rVal = rhs[idx]
							}
							var res uint32
							switch op {
							case compute.OpTypeAdd:
								res = lVal + rVal
							case compute.OpTypeSub:
								res = lVal - rVal
							case compute.OpTypeMul:
								res = lVal * rVal
							case compute.OpTypeMax:
								res = max(lVal, rVal)
							case compute.OpTypeMin:
								res = min(lVal, rVal)
							}
							expected[idx] = res
						}
					}

					ok := DispatchBinaryTrailingAVX2(op, isLHS, unsafe.Pointer(&lhs[0]), unsafe.Pointer(&rhs[0]), unsafe.Pointer(&out[0]), A, B, dtypes.Uint32)
					if !ok {
						t.Fatalf("DispatchBinaryTrailingAVX2 returned false for %s uint32", op)
					}

					for idx := 0; idx < A*B; idx++ {
						if out[idx] != expected[idx] {
							t.Fatalf("Mismatch for %s (isLHS=%v, A=%d, B=%d) at idx %d: got %v, expected %v",
								op, isLHS, A, B, idx, out[idx], expected[idx])
						}
					}
				}
			}
		}
	})

	// 4. Int64
	t.Run("Int64", func(t *testing.T) {
		int64Ops := []compute.OpType{
			compute.OpTypeAdd,
			compute.OpTypeSub,
		}
		for _, op := range int64Ops {
			for _, isLHS := range []bool{false, true} {
				for _, tc := range testCases {
					A, B := tc.A, tc.B
					lhs := make([]int64, A*B)
					rhs := make([]int64, A)
					if isLHS {
						lhs = make([]int64, A)
						rhs = make([]int64, A*B)
					}
					out := make([]int64, A*B)
					expected := make([]int64, A*B)

					for i := range lhs {
						lhs[i] = int64((i % 100) - 50)
					}
					for i := range rhs {
						rhs[i] = int64((i % 10) - 5)
					}

					for a := 0; a < A; a++ {
						for b := 0; b < B; b++ {
							idx := a*B + b
							var lVal, rVal int64
							if !isLHS {
								lVal = lhs[idx]
								rVal = rhs[a]
							} else {
								lVal = lhs[a]
								rVal = rhs[idx]
							}
							var res int64
							switch op {
							case compute.OpTypeAdd:
								res = lVal + rVal
							case compute.OpTypeSub:
								res = lVal - rVal
							}
							expected[idx] = res
						}
					}

					ok := DispatchBinaryTrailingAVX2(op, isLHS, unsafe.Pointer(&lhs[0]), unsafe.Pointer(&rhs[0]), unsafe.Pointer(&out[0]), A, B, dtypes.Int64)
					if !ok {
						t.Fatalf("DispatchBinaryTrailingAVX2 returned false for %s int64", op)
					}

					for idx := 0; idx < A*B; idx++ {
						if out[idx] != expected[idx] {
							t.Fatalf("Mismatch for %s (isLHS=%v, A=%d, B=%d) at idx %d: got %v, expected %v",
								op, isLHS, A, B, idx, out[idx], expected[idx])
						}
					}
				}
			}
		}
	})
}

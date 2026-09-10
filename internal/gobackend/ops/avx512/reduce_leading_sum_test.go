// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512_test

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops/avx512"
)

func requireTrue(t *testing.T, ok bool) {
	t.Helper()
	if !ok {
		t.Fatalf("expected true, got false")
	}
}

func assertInDelta[T ~float32 | ~float64](t *testing.T, expected, actual T, delta float64, msg string) {
	t.Helper()
	if diff := math.Abs(float64(expected - actual)); diff > delta {
		t.Errorf("%s: got %v, expected %v (diff %v > delta %v)", msg, actual, expected, diff, delta)
	}
}

func assertEqual[T comparable](t *testing.T, expected, actual T, msg string) {
	t.Helper()
	if expected != actual {
		t.Errorf("%s: got %v, expected %v", msg, actual, expected)
	}
}

func TestAVX512LeadingSumCorrectness(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 not allowed or not supported on this host")
	}
	bValues := []int{1, 2, 3, 4, 7, 8, 9, 15, 16, 23, 24, 31, 32, 33, 48, 63, 64, 65, 100, 127, 128, 129, 255, 256, 512, 1024}
	aValues := []int{1, 2, 3, 5, 17}

	t.Run("Float32", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]float32, a*b)
				for i := range in {
					in[i] = float32(i%17 - 8)
				}
				out := make([]float32, b)
				expected := make([]float32, b)
				for col := range b {
					var sum float32
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Float32)
				requireTrue(t, ok)
				for col := range b {
					assertInDelta(t, expected[col], out[col], 1e-4, fmt.Sprintf("Float32 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Float64", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]float64, a*b)
				for i := range in {
					in[i] = float64(i%17 - 8)
				}
				out := make([]float64, b)
				expected := make([]float64, b)
				for col := range b {
					var sum float64
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Float64)
				requireTrue(t, ok)
				for col := range b {
					assertInDelta(t, expected[col], out[col], 1e-6, fmt.Sprintf("Float64 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Float16", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]float16.Float16, a*b)
				for i := range in {
					in[i] = float16.FromFloat32(float32(i%11 - 5))
				}
				out := make([]float16.Float16, b)
				expected := make([]float16.Float16, b)
				for col := range b {
					var sum float32
					for row := range a {
						sum += in[row*b+col].Float32()
					}
					expected[col] = float16.FromFloat32(sum)
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Float16)
				requireTrue(t, ok)
				for col := range b {
					assertInDelta(t, expected[col].Float32(), out[col].Float32(), 1e-2, fmt.Sprintf("Float16 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("BFloat16", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]bfloat16.BFloat16, a*b)
				for i := range in {
					in[i] = bfloat16.FromFloat32(float32(i%11 - 5))
				}
				out := make([]bfloat16.BFloat16, b)
				expected := make([]bfloat16.BFloat16, b)
				for col := range b {
					var sum float32
					for row := range a {
						sum += in[row*b+col].Float32()
					}
					expected[col] = bfloat16.FromFloat32(sum)
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.BFloat16)
				requireTrue(t, ok)
				for col := range b {
					assertInDelta(t, expected[col].Float32(), out[col].Float32(), 1e-1, fmt.Sprintf("BFloat16 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Int32", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]int32, a*b)
				for i := range in {
					in[i] = int32(i%17 - 8)
				}
				out := make([]int32, b)
				expected := make([]int32, b)
				for col := range b {
					var sum int32
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int32)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Int32 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Uint32", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]uint32, a*b)
				for i := range in {
					in[i] = uint32(i%17 + 1)
				}
				out := make([]uint32, b)
				expected := make([]uint32, b)
				for col := range b {
					var sum uint32
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint32)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Uint32 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Int16", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]int16, a*b)
				for i := range in {
					in[i] = int16(i%17 - 8)
				}
				out := make([]int16, b)
				expected := make([]int16, b)
				for col := range b {
					var sum int16
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int16)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Int16 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Uint16", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]uint16, a*b)
				for i := range in {
					in[i] = uint16(i%17 + 1)
				}
				out := make([]uint16, b)
				expected := make([]uint16, b)
				for col := range b {
					var sum uint16
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint16)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Uint16 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Int8", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]int8, a*b)
				for i := range in {
					in[i] = int8(i%7 - 3)
				}
				out := make([]int8, b)
				expected := make([]int8, b)
				for col := range b {
					var sum int8
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int8)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Int8 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Uint8", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]uint8, a*b)
				for i := range in {
					in[i] = uint8(i%7 + 1)
				}
				out := make([]uint8, b)
				expected := make([]uint8, b)
				for col := range b {
					var sum uint8
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint8)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Uint8 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Int64", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]int64, a*b)
				for i := range in {
					in[i] = int64(i%17 - 8)
				}
				out := make([]int64, b)
				expected := make([]int64, b)
				for col := range b {
					var sum int64
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int64)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Int64 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})

	t.Run("Uint64", func(t *testing.T) {
		for _, a := range aValues {
			for _, b := range bValues {
				in := make([]uint64, a*b)
				for i := range in {
					in[i] = uint64(i%17 + 1)
				}
				out := make([]uint64, b)
				expected := make([]uint64, b)
				for col := range b {
					var sum uint64
					for row := range a {
						sum += in[row*b+col]
					}
					expected[col] = sum
				}
				ok := avx512.DispatchLeadingSumAVX512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint64)
				requireTrue(t, ok)
				for col := range b {
					assertEqual(t, expected[col], out[col], fmt.Sprintf("Uint64 failed for A=%d, B=%d, col=%d", a, b, col))
				}
			}
		}
	})
}

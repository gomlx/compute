// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops/avx2"
)

func TestAVX2TrailingSumCorrectness(t *testing.T) {
	if !gobackend.IsAVX2Allowed() {
		t.Skip("AVX2 not allowed or not supported on this host")
	}
	bValues := []int{1, 2, 3, 4, 7, 8, 9, 15, 16, 23, 24, 31, 32, 33, 48, 63, 64, 65, 100, 127, 128, 129, 255, 256, 512, 1024}
	a := 5

	t.Run("Float32", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]float32, a*b)
			for i := range in {
				in[i] = float32(i%17 - 8)
			}
			out := make([]float32, a)
			expected := make([]float32, a)
			for row := range a {
				var sum float32
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Float32)
			requireTrue(t, ok)
			for row := range a {
				assertInDelta(t, expected[row], out[row], 1e-4, fmt.Sprintf("Float32 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Float64", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]float64, a*b)
			for i := range in {
				in[i] = float64(i%17 - 8)
			}
			out := make([]float64, a)
			expected := make([]float64, a)
			for row := range a {
				var sum float64
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Float64)
			requireTrue(t, ok)
			for row := range a {
				assertInDelta(t, expected[row], out[row], 1e-6, fmt.Sprintf("Float64 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Float16", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]float16.Float16, a*b)
			for i := range in {
				in[i] = float16.FromFloat32(float32((i%7)-3) * 0.5)
			}
			out := make([]float16.Float16, a)
			expected := make([]float16.Float16, a)
			for row := range a {
				var sum float32
				for col := range b {
					sum += in[row*b+col].Float32()
				}
				expected[row] = float16.FromFloat32(sum)
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Float16)
			requireTrue(t, ok)
			for row := range a {
				assertInDelta(t, expected[row].Float32(), out[row].Float32(), 0.1, fmt.Sprintf("Float16 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("BFloat16", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]bfloat16.BFloat16, a*b)
			for i := range in {
				in[i] = bfloat16.FromFloat32(float32((i%7)-3) * 0.5)
			}
			out := make([]bfloat16.BFloat16, a)
			expected := make([]bfloat16.BFloat16, a)
			for row := range a {
				var sum float32
				for col := range b {
					sum += in[row*b+col].Float32()
				}
				expected[row] = bfloat16.FromFloat32(sum)
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.BFloat16)
			requireTrue(t, ok)
			for row := range a {
				assertInDelta(t, expected[row].Float32(), out[row].Float32(), 0.1, fmt.Sprintf("BFloat16 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Int32", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]int32, a*b)
			for i := range in {
				in[i] = int32((i % 11) - 5)
			}
			out := make([]int32, a)
			expected := make([]int32, a)
			for row := range a {
				var sum int32
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int32)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Int32 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Uint32", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]uint32, a*b)
			for i := range in {
				in[i] = uint32(i % 13)
			}
			out := make([]uint32, a)
			expected := make([]uint32, a)
			for row := range a {
				var sum uint32
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint32)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Uint32 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Int16", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]int16, a*b)
			for i := range in {
				in[i] = int16((i % 7) - 3)
			}
			out := make([]int16, a)
			expected := make([]int16, a)
			for row := range a {
				var sum int16
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int16)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Int16 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Uint16", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]uint16, a*b)
			for i := range in {
				in[i] = uint16(i % 7)
			}
			out := make([]uint16, a)
			expected := make([]uint16, a)
			for row := range a {
				var sum uint16
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint16)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Uint16 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Int8", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]int8, a*b)
			for i := range in {
				in[i] = int8((i % 5) - 2)
			}
			out := make([]int8, a)
			expected := make([]int8, a)
			for row := range a {
				var sum int8
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int8)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Int8 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Uint8", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]uint8, a*b)
			for i := range in {
				in[i] = uint8(i % 5)
			}
			out := make([]uint8, a)
			expected := make([]uint8, a)
			for row := range a {
				var sum uint8
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint8)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Uint8 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Int64", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]int64, a*b)
			for i := range in {
				in[i] = int64((i % 17) - 8)
			}
			out := make([]int64, a)
			expected := make([]int64, a)
			for row := range a {
				var sum int64
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Int64)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Int64 failed for B=%d, row=%d", b, row))
			}
		}
	})

	t.Run("Uint64", func(t *testing.T) {
		for _, b := range bValues {
			in := make([]uint64, a*b)
			for i := range in {
				in[i] = uint64(i % 17)
			}
			out := make([]uint64, a)
			expected := make([]uint64, a)
			for row := range a {
				var sum uint64
				for col := range b {
					sum += in[row*b+col]
				}
				expected[row] = sum
			}
			ok := avx2.DispatchTrailingSumAVX2(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), a, b, dtypes.Uint64)
			requireTrue(t, ok)
			for row := range a {
				assertEqual(t, expected[row], out[row], fmt.Sprintf("Uint64 failed for B=%d, row=%d", b, row))
			}
		}
	})
}

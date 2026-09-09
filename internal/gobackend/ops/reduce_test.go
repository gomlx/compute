// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func runReduceTest[T comparable](t *testing.T, opName string,
	reduceFn func(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error),
	inShape shapes.Shape, inData any,
	axes []int,
	expected []T) {
	t.Helper()
	builder := backend.Builder(opName).(*gobackend.Builder)
	main := builder.Main().(*gobackend.Function)

	inNode, err := main.Parameter("input", inShape, nil)
	if err != nil {
		t.Fatalf("Failed creating input parameter: %+v", err)
	}

	outNode, err := reduceFn(main, inNode, axes...)
	if err != nil {
		t.Fatalf("Failed adding op %s: %+v", opName, err)
	}

	err = main.Return([]compute.Value{outNode}, nil)
	if err != nil {
		t.Fatalf("Return failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %+v", err)
	}

	inBuf, err := backend.BufferFromFlatData(0, inData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}

	result := outputs[0].(*gobackend.Buffer).Flat.([]T)
	if ok, diff := testutil.IsEqual(expected, result); !ok {
		t.Errorf("Mismatch in %s:\n%s", opName, diff)
	}
}

func TestReduceFastPaths(t *testing.T) {
	// Shape [2, 3]:
	// row 0: 1, 2, 3
	// row 1: 4, 5, 6
	s2x3 := shapes.Make(dtypes.Float32, 2, 3)
	data2x3 := []float32{1, 2, 3, 4, 5, 6}

	t.Run("ReduceAll_Sum", func(t *testing.T) {
		// [2, 3] -> sum of all is 21
		runReduceTest(t, "ReduceAll_Sum", ops.ReduceSum, s2x3, data2x3, []int{0, 1}, []float32{21})
	})

	t.Run("ReduceTrailing_Sum", func(t *testing.T) {
		// [2, 3] reduce axis 1 -> [2]: row 0 = 6, row 1 = 15
		runReduceTest(t, "ReduceTrailing_Sum", ops.ReduceSum, s2x3, data2x3, []int{1}, []float32{6, 15})
	})

	t.Run("ReduceLeading_Sum", func(t *testing.T) {
		// [2, 3] reduce axis 0 -> [3]: col 0 = 5, col 1 = 7, col 2 = 9
		runReduceTest(t, "ReduceLeading_Sum", ops.ReduceSum, s2x3, data2x3, []int{0}, []float32{5, 7, 9})
	})

	// Shape [2, 3, 4]:
	// a in [0, 2), b in [0, 3), c in [0, 4)
	s2x3x4 := shapes.Make(dtypes.Float32, 2, 3, 4)
	data2x3x4 := make([]float32, 24)
	for i := range data2x3x4 {
		data2x3x4[i] = float32(i + 1)
	}

	t.Run("ReduceMiddle_Sum", func(t *testing.T) {
		// [2, 3, 4] reduce axis 1 -> [2, 4]
		// a=0:
		//   b=0: 1,  2,  3,  4
		//   b=1: 5,  6,  7,  8
		//   b=2: 9, 10, 11, 12
		//   sum: 15, 18, 21, 24
		// a=1:
		//   b=0: 13, 14, 15, 16
		//   b=1: 17, 18, 19, 20
		//   b=2: 21, 22, 23, 24
		//   sum: 51, 54, 57, 60
		expected := []float32{15, 18, 21, 24, 51, 54, 57, 60}
		runReduceTest(t, "ReduceMiddle_Sum", ops.ReduceSum, s2x3x4, data2x3x4, []int{1}, expected)
	})

	t.Run("ReduceTrailing_Max", func(t *testing.T) {
		// [2, 3] reduce axis 1 -> [2]: row 0 = 3, row 1 = 6
		runReduceTest(t, "ReduceTrailing_Max", ops.ReduceMax, s2x3, data2x3, []int{1}, []float32{3, 6})
	})

	t.Run("ReduceLeading_Min", func(t *testing.T) {
		// [2, 3] reduce axis 0 -> [3]: col 0 = 1, col 1 = 2, col 2 = 3
		runReduceTest(t, "ReduceLeading_Min", ops.ReduceMin, s2x3, data2x3, []int{0}, []float32{1, 2, 3})
	})

	t.Run("ReduceGeneral_Sum", func(t *testing.T) {
		// [2, 3, 4] reduce axes 0 and 2 -> [3]
		// For each b in [0, 3): sum over a in [0, 2) and c in [0, 4)
		// b=0: (1+2+3+4) + (13+14+15+16) = 10 + 58 = 68
		// b=1: (5+6+7+8) + (17+18+19+20) = 26 + 74 = 100
		// b=2: (9+10+11+12) + (21+22+23+24) = 42 + 90 = 132
		expected := []float32{68, 100, 132}
		runReduceTest(t, "ReduceGeneral_Sum", ops.ReduceSum, s2x3x4, data2x3x4, []int{0, 2}, expected)
	})

	t.Run("ReduceTrailing_BFloat16_Sum", func(t *testing.T) {
		s2x3_bf16 := shapes.Make(dtypes.BFloat16, 2, 3)
		data_bf16 := make([]bfloat16.BFloat16, 6)
		for i, v := range data2x3 {
			data_bf16[i] = bfloat16.FromFloat32(v)
		}
		expected := []bfloat16.BFloat16{bfloat16.FromFloat32(6), bfloat16.FromFloat32(15)}
		runReduceTest(t, "ReduceTrailing_BFloat16_Sum", ops.ReduceSum, s2x3_bf16, data_bf16, []int{1}, expected)
	})

	t.Run("ReduceTrailing_Float16_Sum", func(t *testing.T) {
		s2x3_f16 := shapes.Make(dtypes.Float16, 2, 3)
		data_f16 := make([]float16.Float16, 6)
		for i, v := range data2x3 {
			data_f16[i] = float16.FromFloat32(v)
		}
		expected := []float16.Float16{float16.FromFloat32(6), float16.FromFloat32(15)}
		runReduceTest(t, "ReduceTrailing_Float16_Sum", ops.ReduceSum, s2x3_f16, data_f16, []int{1}, expected)
	})

	t.Run("ReduceTrailing_BFloat16_Max", func(t *testing.T) {
		s2x3_bf16 := shapes.Make(dtypes.BFloat16, 2, 3)
		data_bf16 := make([]bfloat16.BFloat16, 6)
		for i, v := range data2x3 {
			data_bf16[i] = bfloat16.FromFloat32(v)
		}
		expected := []bfloat16.BFloat16{bfloat16.FromFloat32(3), bfloat16.FromFloat32(6)}
		runReduceTest(t, "ReduceTrailing_BFloat16_Max", ops.ReduceMax, s2x3_bf16, data_bf16, []int{1}, expected)
	})

	t.Run("ReduceTrailing_Float16_Max", func(t *testing.T) {
		s2x3_f16 := shapes.Make(dtypes.Float16, 2, 3)
		data_f16 := make([]float16.Float16, 6)
		for i, v := range data2x3 {
			data_f16[i] = float16.FromFloat32(v)
		}
		expected := []float16.Float16{float16.FromFloat32(3), float16.FromFloat32(6)}
		runReduceTest(t, "ReduceTrailing_Float16_Max", ops.ReduceMax, s2x3_f16, data_f16, []int{1}, expected)
	})

	t.Run("ReduceLeading_BFloat16_Min", func(t *testing.T) {
		s2x3_bf16 := shapes.Make(dtypes.BFloat16, 2, 3)
		data_bf16 := make([]bfloat16.BFloat16, 6)
		for i, v := range data2x3 {
			data_bf16[i] = bfloat16.FromFloat32(v)
		}
		expected := []bfloat16.BFloat16{bfloat16.FromFloat32(1), bfloat16.FromFloat32(2), bfloat16.FromFloat32(3)}
		runReduceTest(t, "ReduceLeading_BFloat16_Min", ops.ReduceMin, s2x3_bf16, data_bf16, []int{0}, expected)
	})

	t.Run("ReduceLeading_Float16_Min", func(t *testing.T) {
		s2x3_f16 := shapes.Make(dtypes.Float16, 2, 3)
		data_f16 := make([]float16.Float16, 6)
		for i, v := range data2x3 {
			data_f16[i] = float16.FromFloat32(v)
		}
		expected := []float16.Float16{float16.FromFloat32(1), float16.FromFloat32(2), float16.FromFloat32(3)}
		runReduceTest(t, "ReduceLeading_Float16_Min", ops.ReduceMin, s2x3_f16, data_f16, []int{0}, expected)
	})
}

func TestReduceThresholdFallback(t *testing.T) {
	// 1. Below threshold: shape [10, 4] for Float32 ReduceTrailing (threshold is 32)
	// Must fall back to scalar and succeed.
	s10x4 := shapes.Make(dtypes.Float32, 10, 4)
	data10x4 := make([]float32, 40)
	expected10 := make([]float32, 10)
	for r := range 10 {
		var sum float32
		for c := range 4 {
			val := float32(r*4 + c + 1)
			data10x4[r*4+c] = val
			sum += val
		}
		expected10[r] = sum
	}
	runReduceTest(t, "ReduceTrailing_BelowThreshold", ops.ReduceSum, s10x4, data10x4, []int{1}, expected10)

	// 2. Above threshold: shape [10, 64] for Float32 ReduceTrailing (threshold is 32)
	// Runs SIMD.
	s10x64 := shapes.Make(dtypes.Float32, 10, 64)
	data10x64 := make([]float32, 640)
	expected10_64 := make([]float32, 10)
	for r := range 10 {
		var sum float32
		for c := range 64 {
			val := float32(r*64 + c + 1)
			data10x64[r*64+c] = val
			sum += val
		}
		expected10_64[r] = sum
	}
	runReduceTest(t, "ReduceTrailing_AboveThreshold", ops.ReduceSum, s10x64, data10x64, []int{1}, expected10_64)
}

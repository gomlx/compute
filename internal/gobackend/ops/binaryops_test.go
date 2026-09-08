// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func runBinaryTest[T float32 | bool](t *testing.T, opName string, opFn func(f *gobackend.Function, lhs, rhs compute.Value) (compute.Value, error),
	lhsShape shapes.Shape, lhsData []float32,
	rhsShape shapes.Shape, rhsData []float32,
	expected []T) {
	t.Helper()
	builder := backend.Builder(opName).(*gobackend.Builder)
	main := builder.Main().(*gobackend.Function)

	lhsNode, err := main.Parameter("lhs", lhsShape, nil)
	if err != nil {
		t.Fatalf("Failed creating lhs parameter: %+v", err)
	}
	rhsNode, err := main.Parameter("rhs", rhsShape, nil)
	if err != nil {
		t.Fatalf("Failed creating rhs parameter: %+v", err)
	}

	outNode, err := opFn(main, lhsNode, rhsNode)
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

	lhsBuf, err := backend.BufferFromFlatData(0, lhsData, lhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData lhs failed: %+v", err)
	}
	rhsBuf, err := backend.BufferFromFlatData(0, rhsData, rhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData rhs failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{lhsBuf, rhsBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}

	result := outputs[0].(*gobackend.Buffer).Flat.([]T)
	if ok, diff := testutil.IsEqual(expected, result); !ok {
		t.Errorf("Mismatch in %s:\n%s", opName, diff)
	}
}

func TestBinaryBroadcastPatterns(t *testing.T) {
	s2x3 := shapes.Make(dtypes.Float32, 2, 3)
	s1x3 := shapes.Make(dtypes.Float32, 1, 3)
	s2x1 := shapes.Make(dtypes.Float32, 2, 1)

	lhs2x3 := []float32{10, 20, 30, 40, 50, 60}
	rhs1x3 := []float32{1, 2, 3}
	rhs2x1 := []float32{2, 10}

	t.Run("LeadingRHS_Add", func(t *testing.T) {
		// [2, 3] + [1, 3] -> [2, 3]
		expected := []float32{11, 22, 33, 41, 52, 63}
		runBinaryTest(t, "Add_LeadingRHS", ops.Add, s2x3, lhs2x3, s1x3, rhs1x3, expected)
	})

	t.Run("LeadingRHS_Sub", func(t *testing.T) {
		// [2, 3] - [1, 3] -> [2, 3]
		expected := []float32{9, 18, 27, 39, 48, 57}
		runBinaryTest(t, "Sub_LeadingRHS", ops.Sub, s2x3, lhs2x3, s1x3, rhs1x3, expected)
	})

	t.Run("LeadingRHS_Div", func(t *testing.T) {
		// [2, 3] / [1, 3] -> [2, 3]
		expected := []float32{10, 10, 10, 40, 25, 20}
		runBinaryTest(t, "Div_LeadingRHS", ops.Div, s2x3, lhs2x3, s1x3, rhs1x3, expected)
	})

	t.Run("LeadingLHS_Sub", func(t *testing.T) {
		// [1, 3] - [2, 3] -> [2, 3]
		expected := []float32{-9, -18, -27, -39, -48, -57}
		runBinaryTest(t, "Sub_LeadingLHS", ops.Sub, s1x3, rhs1x3, s2x3, lhs2x3, expected)
	})

	t.Run("LeadingLHS_Div", func(t *testing.T) {
		// [1, 3] / [2, 3] -> [2, 3]
		lhs1x3_div := []float32{100, 200, 300}
		rhs2x3_div := []float32{10, 20, 30, 2, 4, 5}
		expected := []float32{10, 10, 10, 50, 50, 60}
		runBinaryTest(t, "Div_LeadingLHS", ops.Div, s1x3, lhs1x3_div, s2x3, rhs2x3_div, expected)
	})

	t.Run("TrailingRHS_Sub", func(t *testing.T) {
		// [2, 3] - [2, 1] -> [2, 3]
		// row 0: {10, 20, 30} - 2 = {8, 18, 28}
		// row 1: {40, 50, 60} - 10 = {30, 40, 50}
		expected := []float32{8, 18, 28, 30, 40, 50}
		runBinaryTest(t, "Sub_TrailingRHS", ops.Sub, s2x3, lhs2x3, s2x1, rhs2x1, expected)
	})

	t.Run("TrailingRHS_Div", func(t *testing.T) {
		// [2, 3] / [2, 1] -> [2, 3]
		// row 0: {10, 20, 30} / 2 = {5, 10, 15}
		// row 1: {40, 50, 60} / 10 = {4, 5, 6}
		expected := []float32{5, 10, 15, 4, 5, 6}
		runBinaryTest(t, "Div_TrailingRHS", ops.Div, s2x3, lhs2x3, s2x1, rhs2x1, expected)
	})

	t.Run("TrailingLHS_Sub", func(t *testing.T) {
		// [2, 1] - [2, 3] -> [2, 3]
		// row 0: 2 - {10, 20, 30} = {-8, -18, -28}
		// row 1: 10 - {40, 50, 60} = {-30, -40, -50}
		expected := []float32{-8, -18, -28, -30, -40, -50}
		runBinaryTest(t, "Sub_TrailingLHS", ops.Sub, s2x1, rhs2x1, s2x3, lhs2x3, expected)
	})

	t.Run("TrailingLHS_Div", func(t *testing.T) {
		// [2, 1] / [2, 3] -> [2, 3]
		lhs2x1_div := []float32{60, 120}
		rhs2x3_div := []float32{1, 2, 3, 10, 20, 30}
		expected := []float32{60, 30, 20, 12, 6, 4}
		runBinaryTest(t, "Div_TrailingLHS", ops.Div, s2x1, lhs2x1_div, s2x3, rhs2x3_div, expected)
	})

	t.Run("RowCol_Sub", func(t *testing.T) {
		// [2, 1] - [1, 3] -> [2, 3]
		// row 0: 10 - {1, 2, 3} = {9, 8, 7}
		// row 1: 20 - {1, 2, 3} = {19, 18, 17}
		lhs2x1_rowcol := []float32{10, 20}
		expected := []float32{9, 8, 7, 19, 18, 17}
		runBinaryTest(t, "Sub_RowCol", ops.Sub, s2x1, lhs2x1_rowcol, s1x3, rhs1x3, expected)
	})

	t.Run("ColRow_Sub", func(t *testing.T) {
		// [1, 3] - [2, 1] -> [2, 3]
		// row 0: {1, 2, 3} - 10 = {-9, -8, -7}
		// row 1: {1, 2, 3} - 20 = {-19, -18, -17}
		rhs2x1_colrow := []float32{10, 20}
		expected := []float32{-9, -8, -7, -19, -18, -17}
		runBinaryTest(t, "Sub_ColRow", ops.Sub, s1x3, rhs1x3, s2x1, rhs2x1_colrow, expected)
	})

	t.Run("ColRow_Div", func(t *testing.T) {
		// [1, 3] / [2, 1] -> [2, 3]
		lhs1x3_div := []float32{60, 120, 180}
		rhs2x1_div := []float32{2, 10}
		expected := []float32{30, 60, 90, 6, 12, 18}
		runBinaryTest(t, "Div_ColRow", ops.Div, s1x3, lhs1x3_div, s2x1, rhs2x1_div, expected)
	})

	t.Run("Comparison_LessThan_LeadingRHS", func(t *testing.T) {
		// [2, 3] < [1, 3]
		lhs := []float32{0, 5, 2, 4, 1, 6}
		rhs := []float32{1, 2, 3}
		// row 0: 0 < 1 (true), 5 < 2 (false), 2 < 3 (true)
		// row 1: 4 < 1 (false), 1 < 2 (true), 6 < 3 (false)
		expected := []bool{true, false, true, false, true, false}
		runBinaryTest(t, "LessThan_LeadingRHS", ops.LessThan, s2x3, lhs, s1x3, rhs, expected)
	})
}


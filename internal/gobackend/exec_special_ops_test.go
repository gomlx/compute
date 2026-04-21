// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"math"
	"reflect"
	"slices"
	"sort"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/compute/support/xslices"
)

var (
	// Shortcuts:

	Bool = dtypes.Bool
	I8   = dtypes.Int8
	I32  = dtypes.Int32
	F32  = dtypes.Float32
	U64  = dtypes.Uint64
	MS   = shapes.Make

	// bf16 shortcut to create new BFloat16 numbers.
	bf16 = bfloat16.FromFloat32
)

func TestExecSpecialOps_Identity(t *testing.T) {
	y0, err := testutil.Exec1(backend, []any{bfloat16.FromFloat32(7)}, func(f compute.Function, params []compute.Value) (compute.Value, error) { return f.Identity(params[0]) })
	if err != nil {
		t.Errorf("Identity failed: %v", err)
	}
	if ok, diff := testutil.IsEqual(bfloat16.FromFloat32(7), y0); !ok {
		t.Errorf("Identity mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Where(t *testing.T) {
	buildWhere := func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.Where(params[0], params[1], params[2])
	}

	// All scalars.
	y0, err := testutil.Exec1(backend, []any{true, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11)}, buildWhere)
	if err != nil {
		t.Fatalf("Where (scalar) failed: %v", err)
	}
	if ok, diff := testutil.IsEqual(bfloat16.FromFloat32(7), y0); !ok {
		t.Errorf("Where (scalar) mismatch:\n%s", diff)
	}

	// Scalar cond, non-scalar values.
	y1, err := testutil.Exec1(backend, []any{false, []uint8{1, 2}, []uint8{11, 12}}, buildWhere)
	if err != nil {
		t.Fatalf("Where (scalar cond) failed: %v", err)
	}
	if ok, diff := testutil.IsEqual([]uint8{11, 12}, y1); !ok {
		t.Errorf("Where (scalar cond) mismatch:\n%s", diff)
	}

	// Non-scalar cond, scalar values.
	y2, err := testutil.Exec1(backend, []any{[]bool{true, false}, int32(1), int32(0)}, buildWhere)
	if err != nil {
		t.Fatalf("Where (non-scalar cond) failed: %v", err)
	}
	if ok, diff := testutil.IsEqual([]int32{1, 0}, y2); !ok {
		t.Errorf("Where (non-scalar cond) mismatch:\n%s", diff)
	}

	// Non-scalar cond and values.
	y3, err := testutil.Exec1(backend, []any{[]bool{false, true, true}, []float32{1, 2, 3}, []float32{101, 102, 103}}, buildWhere)
	if err != nil {
		t.Fatalf("Where (non-scalar cond and values) failed: %v", err)
	}
	if ok, diff := testutil.IsEqual([]float32{101, 2, 3}, y3); !ok {
		t.Errorf("Where (non-scalar cond and values) mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Reshape(t *testing.T) {
	// Reshape array to matrix.
	y0, err := testutil.Exec1(backend, []any{[]int32{42, 0, 1, 2}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.Reshape(params[0], 2, 2)
	})
	if err != nil {
		t.Fatalf("Reshape failed: %v", err)
	}
	if ok, diff := testutil.IsEqual([][]int32{{42, 0}, {1, 2}}, y0); !ok {
		t.Errorf("Reshape mismatch:\n%s", diff)
	}
}

// =================================================================================================================
// Reduce* ---------------------------------------------------------------------------------------------------------
// =================================================================================================================

func TestExecSpecialOps_Reduce(t *testing.T) {
	y0, _ := testutil.Exec1(backend, []any{[][]float32{{7, 0, 9}, {0, 3, 2}, {1001, 101, 11}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceMin(p[0], 1) })
	if ok, diff := testutil.IsEqual([]float32{0, 0, 11}, y0); !ok {
		t.Errorf("ReduceMin mismatch:\n%s", diff)
	}

	y1, _ := testutil.Exec1(backend, []any{[]float64{-1e8, -1e6, -1e16}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceMax(p[0], 0) })
	if ok, diff := testutil.IsEqual(-1.0e6, y1); !ok {
		t.Errorf("ReduceMax mismatch:\n%s", diff)
	}

	input2Data := make([][][][][]uint32, 2)
	idx := uint32(0)
	for i0 := range input2Data {
		input2Data[i0] = make([][][][]uint32, 2)
		for i1 := range input2Data[i0] {
			input2Data[i0][i1] = make([][][]uint32, 2)
			for i2 := range input2Data[i0][i1] {
				input2Data[i0][i1][i2] = make([][]uint32, 2)
				for i3 := range input2Data[i0][i1][i2] {
					input2Data[i0][i1][i2][i3] = make([]uint32, 2)
					for i4 := range input2Data[i0][i1][i2][i3] {
						input2Data[i0][i1][i2][i3][i4] = idx
						idx++
					}
				}
			}
		}
	}
	y2, _ := testutil.Exec1(backend, []any{input2Data}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.ReduceSum(p[0], 1, 3)
	})
	want2 := [][][]uint32{{{20, 24}, {36, 40}}, {{84, 88}, {100, 104}}}
	if ok, diff := testutil.IsEqual(want2, y2); !ok {
		t.Errorf("ReduceSum mismatch:\n%s", diff)
	}

	y3, _ := testutil.Exec1(backend, []any{[]float32{-1e-2, 1e5, -1e-3}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceProduct(p[0], 0) })
	if ok, diff := testutil.IsEqual(float32(1), y3); !ok {
		t.Errorf("ReduceMultiply mismatch:\n%s", diff)
	}

	y4, _ := testutil.Exec1(backend, []any{[]bfloat16.BFloat16{bf16(-11), bf16(-17), bf16(-8)}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceMin(p[0], 0) })
	if ok, diff := testutil.IsEqual(bf16(-17), y4); !ok {
		t.Errorf("ReduceMin (bf16) mismatch:\n%s", diff)
	}

	// Test full reduction to scalar if no axes are given.
	y5, _ := testutil.Exec1(backend, []any{[][]bfloat16.BFloat16{{bf16(-11), bf16(-17)}, {bf16(8), bf16(21)}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceSum(p[0], 0, 1) })
	if ok, diff := testutil.IsEqual(bf16(1), y5); !ok {
		t.Errorf("Full reduction mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_ReduceBitwise(t *testing.T) {
	y0, _ := testutil.Exec1(backend, []any{[]int32{7, 3, 2}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceBitwiseAnd(p[0], 0) })
	if ok, diff := testutil.IsEqual(int32(2), y0); !ok {
		t.Errorf("ReduceBitwiseAnd mismatch:\n%s", diff)
	}

	y1, _ := testutil.Exec1(backend, []any{[][]uint8{{3}, {12}, {17}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.ReduceBitwiseOr(p[0], 0, 1)
	})
	if ok, diff := testutil.IsEqual(uint8(31), y1); !ok {
		t.Errorf("ReduceBitwiseOr mismatch:\n%s", diff)
	}

	y2, _ := testutil.Exec1(backend, []any{[][]int64{{3}, {12}, {17}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceBitwiseXor(p[0], 0) })
	if ok, diff := testutil.IsEqual([]int64{30}, y2); !ok {
		t.Errorf("ReduceBitwiseXor mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_ReduceLogical(t *testing.T) {
	y0, _ := testutil.Exec1(backend, []any{[][]bool{{true, false}, {true, true}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceLogicalAnd(p[0], 1) })
	if ok, diff := testutil.IsEqual([]bool{false, true}, y0); !ok {
		t.Errorf("ReduceLogicalAnd mismatch:\n%s", diff)
	}

	y1, _ := testutil.Exec1(backend, []any{[][]bool{{true, false}, {false, false}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceLogicalOr(p[0], 0) })
	if ok, diff := testutil.IsEqual([]bool{true, false}, y1); !ok {
		t.Errorf("ReduceLogicalOr mismatch:\n%s", diff)
	}

	y2, _ := testutil.Exec1(backend, []any{[][]bool{{true, false}, {true, true}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) { return f.ReduceLogicalXor(p[0], 1) })
	if ok, diff := testutil.IsEqual([]bool{true, false}, y2); !ok {
		t.Errorf("ReduceLogicalXor mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_transposeIterator(t *testing.T) {
	operand := shapes.Make(dtypes.Int32, 2, 3, 4)
	permutations := []int{2, 0, 1}
	it := newTransposeIterator(operand, permutations)
	transposedFlatIndices := make([]int, 0, operand.Size())
	for range operand.Size() {
		transposedFlatIndices = append(transposedFlatIndices, it.next())
	}
	// fmt.Printf("\ttransposedFlatIndices=%#v\n", transposedFlatIndices)
	want := []int{
		// Operand axis 2 (the first being iterated) becomes output axis 0, in row-major order,
		// this is the largest one, with strides of 6:
		0, 6, 12, 18,
		1, 7, 13, 19,
		2, 8, 14, 20,

		3, 9, 15, 21,
		4, 10, 16, 22,
		5, 11, 17, 23}
	if ok, diff := testutil.IsEqual(want, transposedFlatIndices); !ok {
		t.Fatalf("transposeIterator mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Transpose(t *testing.T) {
	operandFlat := xslices.Iota(float32(0), 24)
	// We construct a nested slice from operandFlat.
	operandNested := make([][][]float32, 2)
	idx := 0
	for i := range operandNested {
		operandNested[i] = make([][]float32, 3)
		for j := range operandNested[i] {
			operandNested[i][j] = make([]float32, 4)
			for k := range operandNested[i][j] {
				operandNested[i][j][k] = operandFlat[idx]
				idx++
			}
		}
	}
	y0, _ := testutil.Exec1(backend, []any{operandNested}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.Transpose(p[0], 2, 0, 1)
	})
	want := [][][]float32{
		{{0, 4, 8}, {12, 16, 20}},
		{{1, 5, 9}, {13, 17, 21}},
		{{2, 6, 10}, {14, 18, 22}},
		{{3, 7, 11}, {15, 19, 23}}}
	if ok, diff := testutil.IsEqual(want, y0); !ok {
		t.Fatalf("Transpose result mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Iota(t *testing.T) {
	y0, _ := testutil.Exec1(backend, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
		return f.Iota(shapes.Make(dtypes.Int8, 2, 3), 1)
	})
	if ok, diff := testutil.IsEqual([][]int8{{0, 1, 2}, {0, 1, 2}}, y0); !ok {
		t.Fatalf("Iota y0 mismatch:\n%s", diff)
	}

	y1, _ := testutil.Exec1(backend, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
		return f.Iota(shapes.Make(dtypes.BFloat16, 2, 3), 0)
	})
	bf16 := bfloat16.FromFloat32
	if ok, diff := testutil.IsEqual([][]bfloat16.BFloat16{{bf16(0), bf16(0), bf16(0)}, {bf16(1), bf16(1), bf16(1)}}, y1); !ok {
		t.Fatalf("Iota y1 mismatch:\n%s", diff)
	}

}

func TestExecSpecialOps_Broadcast(t *testing.T) {
	y0, _ := testutil.Exec1(backend, []any{[]int8{1, 3}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.BroadcastInDim(p[0], shapes.Make(dtypes.Int8, 2, 3, 2), []int{2})
	})
	if ok, diff := testutil.IsEqual([][][]int8{{{1, 3}, {1, 3}, {1, 3}}, {{1, 3}, {1, 3}, {1, 3}}}, y0); !ok {
		t.Fatalf("Broadcast result mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_BroadcastInDim(t *testing.T) {
	y0, _ := testutil.Exec1(backend, []any{[][]int8{{1, 3}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.BroadcastInDim(p[0], shapes.Make(dtypes.Int8, 2, 3, 2), []int{0, 2})
	})
	if ok, diff := testutil.IsEqual([][][]int8{{{1, 3}, {1, 3}, {1, 3}}, {{1, 3}, {1, 3}, {1, 3}}}, y0); !ok {
		t.Errorf("BroadcastInDim y0 result mismatch:\n%s", diff)
	}

	y1, _ := testutil.Exec1(backend, []any{bf16(42)}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.BroadcastInDim(p[0], shapes.Make(dtypes.BFloat16, 2), []int{})
	})
	if ok, diff := testutil.IsEqual([]bfloat16.BFloat16{bf16(42), bf16(42)}, y1); !ok {
		t.Errorf("BroadcastInDim y1 result mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_gatherIterator(t *testing.T) {
	operandShape := shapes.Make(dtypes.F32, 4, 3, 2, 2)
	startIndicesShape := shapes.Make(dtypes.Int8, 3, 3, 2)
	startVectorAxis := 1
	offsetOutputAxes := []int{1, 3}
	collapsedSliceAxes := []int{0, 2}
	startIndexMap := []int{0, 2, 3}
	sliceSizes := []int{1, 3, 1, 1}
	outputShape, err := shapeinference.Gather(operandShape, startIndicesShape, startVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	if err != nil {
		t.Fatalf("shapeinference.Gather failed: %+v", err)
	}
	// fmt.Printf("\toutputShape=%s\n", outputShape)
	if err := outputShape.Check(dtypes.F32, 3, 3, 2, 1); err != nil {
		t.Fatalf("outputShape check failed: %+v", err)
	}
	it := newGatherIterator(startIndicesShape, startVectorAxis, outputShape, offsetOutputAxes)
	var gotStartIndices [][]int
	var gotOutputIndices []int
	indices := make([]int, 3)
	var outputBytesIdx int
	for it.Next(indices, &outputBytesIdx) {
		gotStartIndices = append(gotStartIndices, slices.Clone(indices))
		gotOutputIndices = append(gotOutputIndices, outputBytesIdx)
	}
	// fmt.Printf("\tgatherStartIndicesIterator got startIndices=%#v\n", gotStartIndices)
	// fmt.Printf("\tgatherStartIndicesIterator got outputBytesIndices=%#v\n", gotOutputIndices)
	wantStartIndirectIndices := [][]int{{0, 2, 4}, {1, 3, 5}, {6, 8, 10}, {7, 9, 11}, {12, 14, 16}, {13, 15, 17}}
	if ok, diff := testutil.IsEqual(wantStartIndirectIndices, gotStartIndices); !ok {
		t.Errorf("gotStartIndices mismatch:\n%s", diff)
	}
	dataSize := operandShape.DType.Size() // == 4 for Float32
	wantOutputFlatIndices := []int{0, 1, 6, 7, 12, 13}
	for ii := range wantOutputFlatIndices {
		wantOutputFlatIndices[ii] *= dataSize
	}
	if ok, diff := testutil.IsEqual(wantOutputFlatIndices, gotOutputIndices); !ok {
		t.Errorf("gotOutputIndices mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Gather(t *testing.T) {
	buildFn := func(f compute.Function, params []compute.Value) (compute.Value, error) {
		operand, err := f.Iota(shapes.Make(dtypes.F32, 4*3*2*2), 0)
		if err != nil {
			return nil, err
		}
		operand, err = f.Reshape(operand, 4, 3, 2, 2)
		if err != nil {
			return nil, err
		}
		startIndices := params[0]
		startVectorAxis := 1
		offsetOutputAxes := []int{1, 3}
		collapsedSliceAxes := []int{0, 2}
		startIndexMap := []int{0, 2, 3}
		sliceSizes := []int{1, 3, 1, 1}
		return f.Gather(operand, startIndices, startVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, false)
	}
	y0, _ := testutil.Exec1(backend, []any{[][][]int32{{{0, 1}, {0, 1}, {0, 1}}, {{0, 0}, {0, 0}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}}}, buildFn)
	want := [][][][]float32{
		{{{0}, {15}}, {{4}, {19}}, {{8}, {23}}},
		{{{1}, {1}}, {{5}, {5}}, {{9}, {9}}},
		{{{2}, {2}}, {{6}, {6}}, {{10}, {10}}}}
	if ok, diff := testutil.IsEqual(want, y0); !ok {
		t.Fatalf("Gather mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Concatenate(t *testing.T) {
	// Test Case 1: Concatenating vectors (rank 1) along axis 0
	y1, _ := testutil.Exec1(backend, []any{[]float32{1, 2, 3}, []float32{4, 5}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.Concatenate(0, p[0], p[1])
	})
	want1 := []float32{1, 2, 3, 4, 5}
	if ok, diff := testutil.IsEqual(want1, y1); !ok {
		t.Fatalf("Concatenate y1 mismatch:\n%s", diff)
	}

	// Test Case 2: Concatenating matrices (rank 2) along axis 0
	y2, _ := testutil.Exec1(backend, []any{[][]int8{{1, 2}, {3, 4}}, [][]int8{{5, 6}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.Concatenate(0, p[0], p[1])
	})
	want2 := [][]int8{{1, 2}, {3, 4}, {5, 6}}
	if ok, diff := testutil.IsEqual(want2, y2); !ok {
		t.Fatalf("Concatenate y2 mismatch:\n%s", diff)
	}

	// Test Case 3: Concatenating matrices (rank 2) along axis 1
	y3, _ := testutil.Exec1(backend, []any{[][]bfloat16.BFloat16{{bf16(1)}, {bf16(2)}}, [][]bfloat16.BFloat16{{bf16(3), bf16(4)}, {bf16(5), bf16(6)}}, [][]bfloat16.BFloat16{{bf16(7)}, {bf16(8)}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.Concatenate(1, p[0], p[1], p[2])
	})
	want3 := [][]bfloat16.BFloat16{{bf16(1), bf16(3), bf16(4), bf16(7)}, {bf16(2), bf16(5), bf16(6), bf16(8)}}
	if ok, diff := testutil.IsEqual(want3, y3); !ok {
		t.Fatalf("Concatenate y3 mismatch:\n%s", diff)
	}

	// Test Case 4: Concatenating rank 3 tensors along axis 1
	y4, _ := testutil.Exec1(backend, []any{[][][]int32{{{1, 2}}, {{3, 4}}}, [][][]int32{{{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.Concatenate(1, p[0], p[1])
	})
	want4 := [][][]int32{{{1, 2}, {5, 6}, {7, 8}}, {{3, 4}, {9, 10}, {11, 12}}}
	if ok, diff := testutil.IsEqual(want4, y4); !ok {
		t.Fatalf("Concatenate y4 mismatch:\n%s", diff)
	}

	// Test Case 5: Concatenating rank 3 tensors along axis 2
	y5, _ := testutil.Exec1(backend, []any{[][][]float64{{{1, 2}}, {{3, 4}}}, [][][]float64{{{5}}, {{6}}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.Concatenate(2, p[0], p[1])
	})
	want5 := [][][]float64{{{1, 2, 5}}, {{3, 4, 6}}}
	if ok, diff := testutil.IsEqual(want5, y5); !ok {
		t.Fatalf("Concatenate y5 mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Scatter(t *testing.T) {
	// Case 0: Typical scatter, except updates window is the first axis (usually it's the last)
	operandData0 := make([][][]float32, 2)
	for i := range operandData0 {
		operandData0[i] = make([][]float32, 2)
		for j := range operandData0[i] {
			operandData0[i][j] = make([]float32, 5)
		}
	}
	indicesData := [][]uint8{{0, 1}, {1, 0}}
	updatesData0 := [][]float32{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}}

	y0, _ := testutil.Exec1(backend, []any{operandData0, indicesData, updatesData0}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		indexVectorAxis := 1
		updateWindowAxes := []int{0}
		insertedWindowAxes := []int{0, 1}
		scatterAxesToOperandAxes := []int{0, 1}
		return f.ScatterMax(p[0], p[1], p[2], indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, true, true)
	})
	want0 := [][][]float32{{{0, 0, 0, 0, 0}, {1, 3, 5, 7, 9}}, {{2, 4, 6, 8, 10}, {0, 0, 0, 0, 0}}}
	if ok, diff := testutil.IsEqual(want0, y0); !ok {
		t.Errorf("ScatterMax mismatch:\n%s", diff)
	}

	// Case 1: operand axes shuffled; Operand initialized with ones instead.
	operandData1 := make([][][]float32, 2)
	for i := range operandData1 {
		operandData1[i] = make([][]float32, 5)
		for j := range operandData1[i] {
			operandData1[i][j] = []float32{1, 1}
		}
	}
	y1, _ := testutil.Exec1(backend, []any{operandData1, indicesData, updatesData0}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		indexVectorAxis := 1
		updateWindowAxes := []int{0}
		insertedWindowAxes := []int{0, 2}
		scatterAxesToOperandAxes := []int{0, 2}
		return f.ScatterSum(p[0], p[1], p[2], indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, true, true)
	})
	want1 := [][][]float32{{{1, 2}, {1, 4}, {1, 6}, {1, 8}, {1, 10}}, {{3, 1}, {5, 1}, {7, 1}, {9, 1}, {11, 1}}}
	if ok, diff := testutil.IsEqual(want1, y1); !ok {
		t.Errorf("ScatterSum mismatch:\n%s", diff)
	}

	// Case 2: multi-dimension updates.
	operandData2 := make([][][]bfloat16.BFloat16, 2)
	for i := range operandData2 {
		operandData2[i] = make([][]bfloat16.BFloat16, 3)
		for j := range operandData2[i] {
			operandData2[i][j] = []bfloat16.BFloat16{bf16(1), bf16(1)}
		}
	}
	updatesData2 := [][][]bfloat16.BFloat16{
		{{bf16(-4), bf16(-3)}, {bf16(-2), bf16(-1)}},
		{{bf16(0), bf16(1)}, {bf16(2), bf16(3)}},
	}
	y2, _ := testutil.Exec1(backend, []any{operandData2, indicesData, updatesData2}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		indexVectorAxis := 1
		updateWindowAxes := []int{1, 2}
		insertedWindowAxes := []int{0}
		scatterAxesToOperandAxes := []int{0, 1}
		return f.ScatterMin(p[0], p[1], p[2], indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, true, true)
	})
	want2 := [][][]bfloat16.BFloat16{{{bf16(1), bf16(1)}, {bf16(-4), bf16(-3)}, {bf16(-2), bf16(-1)}}, {{bf16(0), bf16(1)}, {bf16(1), bf16(1)}, {bf16(1), bf16(1)}}}
	if ok, diff := testutil.IsEqual(want2, y2); !ok {
		t.Errorf("ScatterMin mismatch:\n%s", diff)
	}
}

func TestExecSpecialOps_Slice(t *testing.T) {
	// Test Case 1: Simple 1D slice
	y1, _ := testutil.Exec1(backend, []any{[]int64{0, 1, 2, 3, 4}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		starts := []int{1}
		limits := []int{4}
		strides := []int{1}
		return f.Slice(p[0], starts, limits, strides)
	})
	want1 := []int64{1, 2, 3}
	if ok, diff := testutil.IsEqual(want1, y1); !ok {
		t.Fatalf("Slice y1 mismatch:\n%s", diff)
	}

	// Test Case 2: 2D slice with stride > 1
	y2, _ := testutil.Exec1(backend, []any{[][]int32{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		starts := []int{0, 0}
		limits := []int{3, 3}
		strides := []int{2, 2}
		return f.Slice(p[0], starts, limits, strides)
	})
	want2 := [][]int32{{0, 2}, {6, 8}}
	if ok, diff := testutil.IsEqual(want2, y2); !ok {
		t.Fatalf("Slice y2 mismatch:\n%s", diff)
	}

	// Test Case 3: Slice resulting in a rank-2 tensor with size 1x1
	y3, _ := testutil.Exec1(backend, []any{[][]int64{{0, 1}, {2, 3}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		starts := []int{1, 1}
		limits := []int{2, 2}
		strides := []int{1, 1}
		return f.Slice(p[0], starts, limits, strides)
	})
	want3 := [][]int64{{3}}
	if ok, diff := testutil.IsEqual(want3, y3); !ok {
		t.Fatalf("Slice y3 mismatch:\n%s", diff)
	}

	// Test Case 4: Slice with bfloat16 and stride > 1
	y4, _ := testutil.Exec1(backend, []any{[]bfloat16.BFloat16{bf16(0), bf16(1), bf16(2), bf16(3)}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		starts := []int{1}
		limits := []int{4}
		strides := []int{2}
		return f.Slice(p[0], starts, limits, strides)
	})
	want4 := []bfloat16.BFloat16{bf16(1), bf16(3)}
	if ok, diff := testutil.IsEqual(want4, y4); !ok {
		t.Fatalf("Slice y4 mismatch:\n%s", diff)
	}
}

func computeHistogram(values []float64, numBins int) []int {
	if len(values) == 0 {
		return nil
	}
	sort.Float64s(values)
	min, max := values[0], values[len(values)-1]
	binSize := (max - min) / float64(numBins)
	histogram := make([]int, numBins)
	for _, v := range values {
		bin := int((v - min) / binSize)
		if bin == numBins {
			bin--
		}
		histogram[bin]++
	}
	return histogram
}

func TestExecSpecialOps_RNGBitsGenerator(t *testing.T) {
	numSamples := 1000000
	numBins := 10
	tolerance := 0.05 // 5% deviation from expected counts

	testCases := []struct {
		dtype dtypes.DType
		name  string
	}{
		{dtypes.Uint32, "uint32"},
		{dtypes.Uint64, "uint64"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			shape := shapes.Make(tc.dtype, numSamples)
			y, err := testutil.Exec1(backend, []any{[]uint64{0, 0, 0}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				_, values, err := f.RNGBitGenerator(params[0], shape)
				return values, err
			})
			if err != nil {
				t.Fatalf("RNGBitGenerator failed: %v", err)
			}

			// Convert all values to float64 in [0, 1) for histogram computation
			values := make([]float64, numSamples)
			switch tc.dtype {
			case dtypes.Uint32:
				maxVal := float64(math.MaxUint32)
				for i, v := range y.([]uint32) {
					values[i] = float64(v) / maxVal
				}
			case dtypes.Uint64:
				maxVal := float64(math.MaxUint64)
				for i, v := range y.([]uint64) {
					values[i] = float64(v) / maxVal
				}
			}

			hist := computeHistogram(values, numBins)
			expectedPerBin := numSamples / numBins
			maxDeviation := float64(expectedPerBin) * tolerance

			// Check each bin is within tolerance of expected frequency
			for bin, count := range hist {
				deviation := math.Abs(float64(count) - float64(expectedPerBin))
				if deviation > maxDeviation {
					t.Errorf("Bin %d count %d deviates too much from expected %d (deviation: %.2f > %.2f)",
						bin, count, expectedPerBin, deviation, maxDeviation)
				}
			}
		})
	}
}

func TestExecSpecialOps_ArgMinMaxOp(t *testing.T) {
	// Test Case 1: Simple 1D argmin
	y0, _ := testutil.Exec1(backend, []any{[]float32{3, 1, 4, 1, 5}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.ArgMinMax(p[0], 0, dtypes.Int32, true)
	})
	if ok, diff := testutil.IsEqual(int32(1), y0); !ok {
		t.Errorf("ArgMin Case 1 mismatch:\n%s", diff)
	}

	// Test Case 2: 2D argmax along axis 1 (columns)
	y1, _ := testutil.Exec1(backend, []any{[][]int32{{1, 2, 3}, {4, 1, 2}, {7, 8, 5}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.ArgMinMax(p[0], 1, dtypes.Int32, false)
	})
	if ok, diff := testutil.IsEqual([]int32{2, 0, 1}, y1); !ok {
		t.Errorf("ArgMax Case 2 mismatch:\n%s", diff)
	}

	// Test Case 3: 2D argmin along axis 0 (rows) with BFloat16
	y2, _ := testutil.Exec1(backend, []any{[][]bfloat16.BFloat16{
		{bf16(1), bf16(2)},
		{bf16(-1), bf16(3)},
		{bf16(4), bf16(-2)}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.ArgMinMax(p[0], 0, dtypes.Int32, true)
	})
	if ok, diff := testutil.IsEqual([]int32{1, 2}, y2); !ok {
		t.Errorf("ArgMin Case 3 mismatch:\n%s", diff)
	}

	// Test Case 4: 3D argmax with repeated values
	y3, _ := testutil.Exec1(backend, []any{[][][]float32{
		{{1, 2}, {1, 0}, {1, -1}},
		{{4, 3}, {4, 5}, {4, 2}}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
		return f.ArgMinMax(p[0], 1, dtypes.Int32, false)
	})
	if ok, diff := testutil.IsEqual([][]int32{{0, 0}, {0, 1}}, y3); !ok {
		t.Errorf("ArgMax Case 4 mismatch:\n%s", diff)
	}
}

// =================================================================================================================
// ReduceWindow ----------------------------------------------------------------------------------------------------
// =================================================================================================================

func dtypeForSlice(slice any) dtypes.DType {
	t := reflect.TypeOf(slice)
	for t.Kind() == reflect.Slice {
		t = t.Elem()
	}
	return dtypes.FromGoType(t)
}

// Test case structure for ReduceWindow tests.
type reduceWindowGraphTestCase struct { // T is the Go type for data, e.g., float32, []float32
	name string
	// operandData will be the third argument to graph.MustExecOnce (inputs ...any)
	// graph.MustExecOnce infers shape and dtype from this.
	// If specific dtype/shape control is needed beyond inference, it's more complex.
	// For now, assume operandData's type and structure define the input tensor.
	operandData      any // e.g., []float32{1,2,3,4,5} or [][]int32{{1,2},{3,4}}
	reductionType    compute.ReduceOpType
	windowDimensions []int
	strides          []int    // Can be nil, f.ReduceWindow should handle defaults.
	paddings         [][2]int // Can be nil.
	baseDilations    []int    // Can be nil.
	windowDilations  []int    // Can be nil.
	expectedOutput   any      // e.g., []float32{3,5,7,9}
	expectedShape    []int    // For verifying output shape explicitly
}

func TestExecSpecialOps_ReduceWindow(t *testing.T) { // Renamed for common Go test naming, or use user's preference
	// Helper to create BFloat16 slices for test cases
	bf16Values := func(vals ...float32) []bfloat16.BFloat16 {
		res := make([]bfloat16.BFloat16, len(vals))
		for i, v := range vals {
			res[i] = bfloat16.FromFloat32(v)
		}
		return res
	}

	// --- Test Cases for Float32 ---
	for _, tc := range []reduceWindowGraphTestCase{
		{
			name:             "F32_1D_Sum_Win2_Stride1_DefaultPadDil",
			operandData:      []float32{1, 2, 3, 4, 5},
			reductionType:    compute.ReduceOpSum,
			windowDimensions: []int{2},
			strides:          []int{1},
			// Nil for paddings, baseDilations, windowDilations will use f.ReduceWindow defaults
			expectedOutput: []float32{3, 5, 7, 9},
			expectedShape:  []int{4},
		},
		{
			name:             "F32_1D_Product_Win2_Stride2_Pad1_1",
			operandData:      []float32{1, 2, 3, 4},
			reductionType:    compute.ReduceOpProduct,
			windowDimensions: []int{2},
			strides:          []int{2},
			paddings:         [][2]int{{1, 1}},
			// Calculation for expectedOutput:
			// Input: [1,2,3,4], Shape [4], DType F32
			// Window [2], Stride [2], Padding {{1,1}}
			// Shape inference: (InputDim + PadLow + PadHigh - WindowDim) / Stride + 1
			// (4 + 1 + 1 - 2) / 2 + 1 = (6 - 2) / 2 + 1 = 4 / 2 + 1 = 2 + 1 = 3. Output Shape [3]
			// Output[0]: input indices for window at output_idx 0: (0*stride - PadLow) to (0*stride - PadLow + WindowDim -1)
			// (0*2 - 1) = -1 to (0*2 - 1 + 2 -1) = 0. Indices: -1, 0. Valid: input[0]=1. Product=1 (init_val for padding/empty assumed 1 for product)
			// Output[1]: input indices for window at output_idx 1: (1*2 - 1) = 1 to (1*2 - 1 + 2 - 1) = 2. Indices: 1, 2. Valid: input[1]=2, input[2]=3. Prod=2*3=6.
			// Output[2]: input indices for window at output_idx 2: (2*2 - 1) = 3 to (2*2 - 1 + 2 - 1) = 4. Indices: 3, 4. Valid: input[3]=4. Prod=4.
			expectedOutput: []float32{1, 6, 4},
			expectedShape:  []int{3},
		},
		{
			name:             "F32_1D_Max_Win3_WindowDilation2",
			operandData:      []float32{1, 2, 3, 4, 5, 6, 7},
			reductionType:    compute.ReduceOpMax,
			windowDimensions: []int{3},
			strides:          []int{1},
			windowDilations:  []int{2}, // Effective window elements indices: k, k+2, k+4 related to input
			// Effective window span (DilatedWindowDim): (3-1)*2+1 = 5
			// Output shape: (7 - 5)/1 + 1 = 3.
			// Out[0]: input indices 0, 0+1*WinDil=2, 0+2*WinDil=4. Max(data[0], data[2], data[4]) = Max(1,3,5) = 5.
			// Out[1]: input indices 1, 1+1*WinDil=3, 1+2*WinDil=5. Max(data[1], data[3], data[5]) = Max(2,4,6) = 6.
			// Out[2]: input indices 2, 2+1*WinDil=4, 2+2*WinDil=6. Max(data[2], data[4], data[6]) = Max(3,5,7) = 7.
			expectedOutput: []float32{5, 6, 7},
			expectedShape:  []int{3},
		},
		{
			name:             "F32_2D_Sum_NoPadDilStride1",
			operandData:      [][]float32{{1, 2, 3}, {4, 5, 6}}, // Shape [2,3]
			reductionType:    compute.ReduceOpSum,
			windowDimensions: []int{2, 2},
			strides:          []int{1, 1},
			// Output shape: Dim0: (2-2)/1+1 = 1. Dim1: (3-2)/1+1 = 2. Shape [1,2]
			// Out[0,0]: sum of input[0:2, 0:2] = 1+2+4+5 = 12
			// Out[0,1]: sum of input[0:2, 1:3] = 2+3+5+6 = 16
			expectedOutput: [][]float32{{12, 16}},
			expectedShape:  []int{1, 2},
		},
		{
			name:             "I32_1D_Min_Win3_Stride2_BaseDil2",
			operandData:      []int32{10, 2, 5, 1, 8, 3, 9, 4}, // Shape [8]
			reductionType:    compute.ReduceOpMin,
			windowDimensions: []int{3},
			strides:          []int{2}, // Stride in the conceptually base-dilated input
			baseDilations:    []int{2}, // Conceptual input len (8-1)*2+1 = 15. Data: 10 H 2 H 5 H 1 H 8 H 3 H 9 H 4
			// Window takes 3 elements from conceptual input. EffWin=3.
			// Output shape on conceptual input (len 15): (15-3)/2+1 = 12/2+1=7.
			expectedOutput: []int32{2, 2, 1, 1, 3, 3, 4},
			expectedShape:  []int{7},
		},
		{
			name:             "I32_2D_Max",
			operandData:      [][]int32{{1, 5, 2}, {6, 3, 7}, {4, 9, 0}}, // Shape [3,3]
			reductionType:    compute.ReduceOpMax,
			windowDimensions: []int{2, 2},
			strides:          []int{1, 1},
			paddings:         [][2]int{{0, 1}, {1, 0}},
			expectedOutput:   [][]int32{{6, 6, 7}, {6, 9, 9}, {4, 9, 9}},
			expectedShape:    []int{3, 3},
		}, {
			name:             "I32_2D_Max_Win2x2_Stride1x1_NoPadDil",
			operandData:      [][]int32{{1, 2, 3}, {4, 5, 6}},
			reductionType:    compute.ReduceOpMax,
			windowDimensions: []int{2, 2},
			strides:          []int{1, 1},
			expectedOutput:   [][]int32{{5, 6}},
			expectedShape:    []int{1, 2},
		},
		{
			name:             "BF16_1D_Sum_Win2_NoParams",
			operandData:      bf16Values(1, 2, 3, 4), // Input as []bfloat16.Type
			reductionType:    compute.ReduceOpSum,
			windowDimensions: []int{2},
			strides:          []int{1},            // graph.ReduceWindow likely requires explicit strides
			expectedOutput:   bf16Values(3, 5, 7), // 1+2, 2+3, 3+4
			expectedShape:    []int{3},
		},
		{
			name:             "BF16_1D_Product_Win2_BaseDil2_Pad1",
			operandData:      bf16Values(2, 3, 4), // Shape [3]
			reductionType:    compute.ReduceOpProduct,
			windowDimensions: []int{2},
			strides:          []int{1},
			paddings:         [][2]int{{1, 0}}, // Pad low by 1
			baseDilations:    []int{2},         // Conceptual input: [2 H 3 H 4] (len 5). Padded: [PadVal 2 H 3 H 4]
			// Output shape on conceptual (len 5) with padding (1,0): (5+1+0 - 2)/1 + 1 = (6-2)/1+1 = 5
			// Assuming PadVal=1 for product identity if outside region
			// Out[0]: win over conceptual_padded indices [0,1] -> maps to input[0]=2 (via conceptual[1]). Product=2.
			// Out[1]: win over conceptual_padded indices [1,2] -> maps to input[0]=2 (via conceptual[1]), hole (via conceptual[2]). Product=2.
			// Out[2]: win over conceptual_padded indices [2,3] -> maps to input[1]=3 (via conceptual[3]), hole. Product=3.
			// Out[3]: win over conceptual_padded indices [3,4] -> maps to input[1]=3 (via conceptual[3]), hole. Product=3.
			// Out[4]: win over conceptual_padded indices [4,5] -> maps to input[2]=4 (via conceptual[5]), hole. Product=4.
			expectedOutput: bf16Values(2, 2, 3, 3, 4),
			expectedShape:  []int{5},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			y, err := testutil.Exec1(backend, []any{tc.operandData}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.ReduceWindow(p[0], tc.reductionType, tc.windowDimensions, tc.strides, tc.baseDilations, tc.windowDilations, tc.paddings)
			})
			if err != nil {
				t.Fatalf("ReduceWindow failed: %v", err)
			}
			if ok, diff := testutil.IsEqual(tc.expectedOutput, y); !ok {
				t.Errorf("ReduceWindow: test %q mismatch:\n%s", tc.name, diff)
			}
		})
	}
}

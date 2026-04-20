// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/testutil"
	"github.com/gomlx/compute/shapes"
)

func TestExecBinary_broadcastIterator(t *testing.T) {
	S := func(dims ...int) shapes.Shape {
		return shapes.Make(dtypes.Float32, dims...)
	}

	// Simple [2, 3] shape broadcast simultaneously by 2 different tensors.
	targetShape := S(2, 3)
	bi1 := newBroadcastIterator(S(2, 1), targetShape)
	bi2 := newBroadcastIterator(S(1, 3), targetShape)
	indices1 := make([]int, 0, targetShape.Size())
	indices2 := make([]int, 0, targetShape.Size())
	for range targetShape.Size() {
		indices1 = append(indices1, bi1.Next())
		indices2 = append(indices2, bi2.Next())
	}
	fmt.Printf("\tindices1=%v\n\tindices2=%v\n", indices1, indices2)
	if ok, diff := testutil.IsEqual([]int{0, 0, 0, 1, 1, 1}, indices1); !ok {
		t.Errorf("indices1 mismatch (-want +got):\n%s", diff)
	}
	if ok, diff := testutil.IsEqual([]int{0, 1, 2, 0, 1, 2}, indices2); !ok {
		t.Errorf("indices2 mismatch (-want +got):\n%s", diff)
	}

	// Alternating broadcast axes.
	targetShape = S(3, 2, 4, 2)
	b3 := newBroadcastIterator(S(3, 1, 4, 1), targetShape)
	indices3 := make([]int, 0, targetShape.Size())
	for range targetShape.Size() {
		indices3 = append(indices3, b3.Next())
	}
	fmt.Printf("\tindices3=%v\n", indices3)
	want3 := []int{
		0, 0, 1, 1, 2, 2, 3, 3,
		0, 0, 1, 1, 2, 2, 3, 3,
		4, 4, 5, 5, 6, 6, 7, 7,
		4, 4, 5, 5, 6, 6, 7, 7,
		8, 8, 9, 9, 10, 10, 11, 11,
		8, 8, 9, 9, 10, 10, 11, 11,
	}
	if ok, diff := testutil.IsEqual(want3, indices3); !ok {
		t.Errorf("indices3 mismatch (-want +got):\n%s", diff)
	}
}

func testExecBinaryOp(t *testing.T, buildFn func(f compute.Function, params []compute.Value) (compute.Value, error), lhs, rhs any) *Buffer {
	lhsFlat, lhsShape := testutil.ToFlatAndShape(lhs)
	rhsFlat, rhsShape := testutil.ToFlatAndShape(rhs)
	return testBackendMultiInput(t, []shapes.Shape{lhsShape, rhsShape}, []any{lhsFlat, rhsFlat}, buildFn)
}

func TestExecBinary(t *testing.T) {
	t.Run("Add", func(t *testing.T) {
		buildFnAdd := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Add(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnAdd, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(18)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		y1 := testExecBinaryOp(t, buildFnAdd, []int32{-1, 2}, []int32{1})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{0, 3}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		y2 := testExecBinaryOp(t, buildFnAdd, [][]int32{{-1}, {2}}, int32(-1))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-2}, {1}}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnAdd, [][]uint64{{1, 2}, {3, 4}}, [][]uint64{{4, 3}, {2, 1}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]uint64{{5, 5}, {5, 5}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnAdd, [][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{9, 99}, {12, 102}, {15, 105}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Mul", func(t *testing.T) {
		buildFnMul := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Mul(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnMul, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(77)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		y1 := testExecBinaryOp(t, buildFnMul, []int32{-1, 2}, []int32{2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{-2, 4}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		y2 := testExecBinaryOp(t, buildFnMul, [][]int32{{-1}, {2}}, int32(-1))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{1}, {-2}}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnMul, [][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 3}, {2, 1}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-6, 6}, {6, 4}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnMul, [][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-10, -100}, {20, 200}, {50, 500}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Sub", func(t *testing.T) {
		buildFnSub := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Sub(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnSub, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(-4)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnSub, []int32{-1, 2}, []int32{2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{-3, 0}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnSub, int32(5), []int32{1, 2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{4, 3}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnSub, [][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 3}, {2, 1}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-7, -1}, {1, 3}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnSub, [][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-11, -101}, {-8, -98}, {-5, -95}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Div", func(t *testing.T) {
		buildFnDiv := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Div(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnDiv, bfloat16.FromFloat32(10), bfloat16.FromFloat32(2))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(5)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnDiv, []int32{-4, 8}, []int32{2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{-2, 4}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnDiv, int32(6), []int32{2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{3, 2}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnDiv, [][]int32{{-6, 9}, {12, 15}}, [][]int32{{2, 3}, {4, 5}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-3, 3}, {3, 3}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnDiv, [][]int32{{-10}, {20}, {50}}, [][]int32{{2, 10}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-5, -1}, {10, 2}, {25, 5}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Rem", func(t *testing.T) {
		buildFnRem := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Rem(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnRem, bfloat16.FromFloat32(7), bfloat16.FromFloat32(4))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(3)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnRem, []int32{7, 9}, []int32{4})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{3, 1}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnRem, int32(7), []int32{4, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{3, 1}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnRem, [][]int32{{7, 8}, {9, 10}}, [][]int32{{4, 3}, {2, 3}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{3, 2}, {1, 1}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnRem, [][]int32{{7}, {8}, {9}}, [][]int32{{4, 3}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{3, 1}, {0, 2}, {1, 0}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Pow", func(t *testing.T) {
		buildFnPow := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Pow(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnPow, bfloat16.FromFloat32(16), bfloat16.FromFloat32(0.5))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(4)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnPow, []int32{2, 3}, []int32{2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{4, 9}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnPow, int32(2), []int32{2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{4, 8}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnPow, [][]int32{{2, 3}, {4, 5}}, [][]int32{{2, 2}, {2, 2}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{4, 9}, {16, 25}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnPow, [][]int32{{2}, {3}, {4}}, [][]int32{{2, 3}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{4, 8}, {9, 27}, {16, 64}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Max", func(t *testing.T) {
		buildFnMax := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Max(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnMax, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(11)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnMax, []int32{-1, 2}, []int32{0})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{0, 2}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnMax, int32(5), []int32{1, 8})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{5, 8}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnMax, [][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 1}, {2, 5}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{6, 2}, {3, 5}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnMax, [][]int32{{-1}, {2}, {5}}, [][]int32{{0, 3}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{0, 3}, {2, 3}, {5, 5}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Min", func(t *testing.T) {
		buildFnMin := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Min(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnMin, bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(7)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnMin, []int32{-1, 2}, []int32{0})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{-1, 0}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnMin, int32(5), []int32{1, 8})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{1, 5}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnMin, [][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 1}, {2, 5}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-1, 1}, {2, 4}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnMin, [][]int32{{-1}, {2}, {5}}, [][]int32{{0, 3}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{-1, -1}, {0, 2}, {0, 3}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("BitwiseAnd", func(t *testing.T) {
		buildFnBitwiseAnd := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.BitwiseAnd(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnBitwiseAnd, uint8(0b11110000), uint8(0b10101010))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(uint8(0b10100000)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnBitwiseAnd, []int32{0b1100, 0b0011}, []int32{0b1010})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{0b1000, 0b0010}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnBitwiseAnd, uint16(0b1111), []uint16{0b1010, 0b0101})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint16{0b1010, 0b0101}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnBitwiseAnd, [][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{0b1000, 0b0010}, {0b0101, 0b0000}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnBitwiseAnd, [][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]uint32{{0b1000, 0b0100}, {0b0010, 0b0001}, {0b1010, 0b0101}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("BitwiseOr", func(t *testing.T) {
		buildFnBitwiseOr := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.BitwiseOr(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnBitwiseOr, uint8(0b11110000), uint8(0b10101010))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(uint8(0b11111010)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnBitwiseOr, []int32{0b1100, 0b0011}, []int32{0b1010})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{0b1110, 0b1011}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnBitwiseOr, uint16(0b1111), []uint16{0b1010, 0b0101})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint16{0b1111, 0b1111}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnBitwiseOr, [][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{0b1110, 0b1011}, {0b1111, 0b0101}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnBitwiseOr, [][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]uint32{{0b1110, 0b1101}, {0b1011, 0b0111}, {0b1111, 0b1111}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("BitwiseXor", func(t *testing.T) {
		buildFnBitwiseXor := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.BitwiseXor(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnBitwiseXor, uint8(0b11110000), uint8(0b10101010))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(uint8(0b01011010)), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnBitwiseXor, []int32{0b1100, 0b0011}, []int32{0b1010})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{0b0110, 0b1001}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnBitwiseXor, uint16(0b1111), []uint16{0b1010, 0b0101})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint16{0b0101, 0b1010}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnBitwiseXor, [][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]int32{{0b0110, 0b1001}, {0b1010, 0b0101}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnBitwiseXor, [][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]uint32{{0b0110, 0b1001}, {0b1001, 0b0110}, {0b0101, 0b1010}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("LogicalAnd", func(t *testing.T) {
		buildFnLogicalAnd := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.LogicalAnd(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnLogicalAnd, true, false)
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(false), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnLogicalAnd, []bool{true, false}, []bool{true})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{true, false}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnLogicalAnd, true, []bool{true, false})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{true, false}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnLogicalAnd, [][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]bool{{true, false}, {false, true}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnLogicalAnd, [][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]bool{{true, false}, {false, false}, {true, false}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("LogicalOr", func(t *testing.T) {
		buildFnLogicalOr := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.LogicalOr(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnLogicalOr, true, false)
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnLogicalOr, []bool{true, false}, []bool{true})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{true, true}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnLogicalOr, true, []bool{true, false})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{true, true}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnLogicalOr, [][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]bool{{true, true}, {true, true}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnLogicalOr, [][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]bool{{true, true}, {true, false}, {true, true}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("LogicalXor", func(t *testing.T) {
		buildFnLogicalXor := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.LogicalXor(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0 := testExecBinaryOp(t, buildFnLogicalXor, true, false)
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		// Test scalar on right side
		y1 := testExecBinaryOp(t, buildFnLogicalXor, []bool{true, false}, []bool{true})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{false, true}), y1.flat); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		// Test scalar on left side
		y2 := testExecBinaryOp(t, buildFnLogicalXor, true, []bool{true, false})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{false, true}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3 := testExecBinaryOp(t, buildFnLogicalXor, [][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]bool{{false, true}, {true, false}}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4 := testExecBinaryOp(t, buildFnLogicalXor, [][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([][]bool{{false, true}, {true, false}, {false, true}}), y4.flat); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("Comparison", func(t *testing.T) {
		// Test Equal
		buildFnEqual := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Equal(params[0], params[1])
		}
		y0 := testExecBinaryOp(t, buildFnEqual, float32(1.5), float32(1.5))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y0.flat); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}
		y1 := testExecBinaryOp(t, buildFnEqual, bfloat16.FromFloat32(2.0), bfloat16.FromFloat32(2.0))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y1.flat); !ok {
			t.Errorf("y1 mismatch:\n%s", diff)
		}
		y2 := testExecBinaryOp(t, buildFnEqual, []uint16{1, 2, 3}, uint16(2))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{false, true, false}), y2.flat); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}
		y3 := testExecBinaryOp(t, buildFnEqual, []int32{5}, []int32{5, 6})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{true, false}), y3.flat); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test GreaterOrEqual
		buildFnGreaterOrEqual := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.GreaterOrEqual(params[0], params[1])
		}
		y4 := testExecBinaryOp(t, buildFnGreaterOrEqual, float32(2.5), float32(1.5))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y4.flat); !ok {
			t.Errorf("y4 mismatch:\n%s", diff)
		}
		y5 := testExecBinaryOp(t, buildFnGreaterOrEqual, bfloat16.FromFloat32(1.0), bfloat16.FromFloat32(2.0))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(false), y5.flat); !ok {
			t.Errorf("y5 mismatch:\n%s", diff)
		}
		y6 := testExecBinaryOp(t, buildFnGreaterOrEqual, []uint16{1, 2, 3}, uint16(2))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{false, true, true}), y6.flat); !ok {
			t.Errorf("y6 value mismatch (-want +got):\n%s", diff)
		}

		// Test GreaterThan
		buildFnGreaterThan := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.GreaterThan(params[0], params[1])
		}
		y7 := testExecBinaryOp(t, buildFnGreaterThan, float32(2.5), float32(1.5))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y7.flat); !ok {
			t.Errorf("y7 mismatch:\n%s", diff)
		}
		y8 := testExecBinaryOp(t, buildFnGreaterThan, []int32{1, 2, 3}, int32(2))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{false, false, true}), y8.flat); !ok {
			t.Errorf("y8 value mismatch (-want +got):\n%s", diff)
		}

		// Test LessOrEqual
		buildFnLessOrEqual := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.LessOrEqual(params[0], params[1])
		}
		y9 := testExecBinaryOp(t, buildFnLessOrEqual, bfloat16.FromFloat32(1.0), bfloat16.FromFloat32(2.0))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y9.flat); !ok {
			t.Errorf("y9 mismatch:\n%s", diff)
		}
		y10 := testExecBinaryOp(t, buildFnLessOrEqual, []uint16{1, 2, 3}, uint16(2))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{true, true, false}), y10.flat); !ok {
			t.Errorf("y10 value mismatch (-want +got):\n%s", diff)
		}

		// Test LessThan
		buildFnLessThan := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.LessThan(params[0], params[1])
		}
		y11 := testExecBinaryOp(t, buildFnLessThan, float32(1.5), float32(2.5))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y11.flat); !ok {
			t.Errorf("y11 mismatch:\n%s", diff)
		}
		y12 := testExecBinaryOp(t, buildFnLessThan, []int32{1, 2, 3}, int32(2))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{true, false, false}), y12.flat); !ok {
			t.Errorf("y12 value mismatch (-want +got):\n%s", diff)
		}
	})
}

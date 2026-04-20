// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/testutil"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/pkg/core/graph"
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

func TestExecBinary_Add(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Add)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	if val := y0.Value(); val != bfloat16.FromFloat32(18) {
		t.Errorf("Expected y0 to be 18, got %v", val)
	}

	y1 := exec.MustExec([]int32{-1, 2}, []int32{1})[0]
	if ok, diff := testutil.IsEqual([]int32{0, 3}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	y2 := exec.MustExec([][]int32{{-1}, {2}}, int32(-1))[0]
	if ok, diff := testutil.IsEqual([][]int32{{-2}, {1}}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]uint64{{1, 2}, {3, 4}}, [][]uint64{{4, 3}, {2, 1}})[0]
	if ok, diff := testutil.IsEqual([][]uint64{{5, 5}, {5, 5}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{9, 99}, {12, 102}, {15, 105}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Mul(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Mul)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	if val := y0.Value(); val != bfloat16.FromFloat32(77) {
		t.Errorf("Expected y0 to be 77, got %v", val)
	}

	y1 := exec.MustExec([]int32{-1, 2}, []int32{2})[0]
	if ok, diff := testutil.IsEqual([]int32{-2, 4}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	y2 := exec.MustExec([][]int32{{-1}, {2}}, int32(-1))[0]
	if ok, diff := testutil.IsEqual([][]int32{{1}, {-2}}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 3}, {2, 1}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-6, 6}, {6, 4}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-10, -100}, {20, 200}, {50, 500}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Sub(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Sub)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	if val := y0.Value(); val != bfloat16.FromFloat32(-4) {
		t.Errorf("Expected y0 to be -4, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{-1, 2}, []int32{2})[0]
	if ok, diff := testutil.IsEqual([]int32{-3, 0}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(int32(5), []int32{1, 2})[0]
	if ok, diff := testutil.IsEqual([]int32{4, 3}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 3}, {2, 1}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-7, -1}, {1, 3}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-11, -101}, {-8, -98}, {-5, -95}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Div(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Div)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(10), bfloat16.FromFloat32(2))[0]
	if val := y0.Value(); val != bfloat16.FromFloat32(5) {
		t.Errorf("Expected y0 to be 5, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{-4, 8}, []int32{2})[0]
	if ok, diff := testutil.IsEqual([]int32{-2, 4}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(int32(6), []int32{2, 3})[0]
	if ok, diff := testutil.IsEqual([]int32{3, 2}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{-6, 9}, {12, 15}}, [][]int32{{2, 3}, {4, 5}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-3, 3}, {3, 3}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{-10}, {20}, {50}}, [][]int32{{2, 10}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-5, -1}, {10, 2}, {25, 5}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Rem(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Rem)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(7), bfloat16.FromFloat32(4))[0]
	fmt.Printf("\ty0=%v\n", y0.GoStr())
	if val := y0.Value(); val != bfloat16.FromFloat32(3) {
		t.Errorf("Expected y0 to be 3, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{7, 9}, []int32{4})[0]
	if ok, diff := testutil.IsEqual([]int32{3, 1}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(int32(7), []int32{4, 3})[0]
	if ok, diff := testutil.IsEqual([]int32{3, 1}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{7, 8}, {9, 10}}, [][]int32{{4, 3}, {2, 3}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{3, 2}, {1, 1}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{7}, {8}, {9}}, [][]int32{{4, 3}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{3, 1}, {0, 2}, {1, 0}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Pow(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Pow)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(16), bfloat16.FromFloat32(0.5))[0]
	if val := y0.Value(); val != bfloat16.FromFloat32(4) {
		t.Errorf("Expected y0 to be 4, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{2, 3}, []int32{2})[0]
	if ok, diff := testutil.IsEqual([]int32{4, 9}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(int32(2), []int32{2, 3})[0]
	if ok, diff := testutil.IsEqual([]int32{4, 8}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{2, 3}, {4, 5}}, [][]int32{{2, 2}, {2, 2}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{4, 9}, {16, 25}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{2}, {3}, {4}}, [][]int32{{2, 3}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{4, 8}, {9, 27}, {16, 64}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Max(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Max)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	if val := y0.Value(); val != bfloat16.FromFloat32(11) {
		t.Errorf("Expected y0 to be 11, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{-1, 2}, []int32{0})[0]
	if ok, diff := testutil.IsEqual([]int32{0, 2}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(int32(5), []int32{1, 8})[0]
	if ok, diff := testutil.IsEqual([]int32{5, 8}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 1}, {2, 5}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{6, 2}, {3, 5}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{-1}, {2}, {5}}, [][]int32{{0, 3}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{0, 3}, {2, 3}, {5, 5}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Min(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Min)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(bfloat16.FromFloat32(7), bfloat16.FromFloat32(11))[0]
	if val := y0.Value(); val != bfloat16.FromFloat32(7) {
		t.Errorf("Expected y0 to be 7, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{-1, 2}, []int32{0})[0]
	if ok, diff := testutil.IsEqual([]int32{-1, 0}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(int32(5), []int32{1, 8})[0]
	if ok, diff := testutil.IsEqual([]int32{1, 5}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{-1, 2}, {3, 4}}, [][]int32{{6, 1}, {2, 5}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-1, 1}, {2, 4}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]int32{{-1}, {2}, {5}}, [][]int32{{0, 3}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{-1, -1}, {0, 2}, {0, 3}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_BitwiseAnd(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.BitwiseAnd)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(uint8(0b11110000), uint8(0b10101010))[0]
	if val := y0.Value(); val != uint8(0b10100000) {
		t.Errorf("Expected y0 to be 0b10100000, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{0b1100, 0b0011}, []int32{0b1010})[0]
	if ok, diff := testutil.IsEqual([]int32{0b1000, 0b0010}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(uint16(0b1111), []uint16{0b1010, 0b0101})[0]
	if ok, diff := testutil.IsEqual([]uint16{0b1010, 0b0101}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{0b1000, 0b0010}, {0b0101, 0b0000}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})[0]
	if ok, diff := testutil.IsEqual([][]uint32{{0b1000, 0b0100}, {0b0010, 0b0001}, {0b1010, 0b0101}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_BitwiseOr(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.BitwiseOr)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(uint8(0b11110000), uint8(0b10101010))[0]
	if val := y0.Value(); val != uint8(0b11111010) {
		t.Errorf("Expected y0 to be 0b11111010, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{0b1100, 0b0011}, []int32{0b1010})[0]
	if ok, diff := testutil.IsEqual([]int32{0b1110, 0b1011}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(uint16(0b1111), []uint16{0b1010, 0b0101})[0]
	if ok, diff := testutil.IsEqual([]uint16{0b1111, 0b1111}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{0b1110, 0b1011}, {0b1111, 0b0101}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})[0]
	if ok, diff := testutil.IsEqual([][]uint32{{0b1110, 0b1101}, {0b1011, 0b0111}, {0b1111, 0b1111}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_BitwiseXor(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.BitwiseXor)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(uint8(0b11110000), uint8(0b10101010))[0]
	if val := y0.Value(); val != uint8(0b01011010) {
		t.Errorf("Expected y0 to be 0b01011010, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]int32{0b1100, 0b0011}, []int32{0b1010})[0]
	if ok, diff := testutil.IsEqual([]int32{0b0110, 0b1001}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(uint16(0b1111), []uint16{0b1010, 0b0101})[0]
	if ok, diff := testutil.IsEqual([]uint16{0b0101, 0b1010}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]int32{{0b1100, 0b0011}, {0b1111, 0b0000}}, [][]int32{{0b1010, 0b1010}, {0b0101, 0b0101}})[0]
	if ok, diff := testutil.IsEqual([][]int32{{0b0110, 0b1001}, {0b1010, 0b0101}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]uint32{{0b1100}, {0b0011}, {0b1111}}, [][]uint32{{0b1010, 0b0101}})[0]
	if ok, diff := testutil.IsEqual([][]uint32{{0b0110, 0b1001}, {0b1001, 0b0110}, {0b0101, 0b1010}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_LogicalAnd(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.LogicalAnd)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(true, false)[0]
	if val := y0.Value(); val != false {
		t.Errorf("Expected y0 to be false, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]bool{true, false}, []bool{true})[0]
	if ok, diff := testutil.IsEqual([]bool{true, false}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(true, []bool{true, false})[0]
	if ok, diff := testutil.IsEqual([]bool{true, false}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})[0]
	if ok, diff := testutil.IsEqual([][]bool{{true, false}, {false, true}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})[0]
	if ok, diff := testutil.IsEqual([][]bool{{true, false}, {false, false}, {true, false}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_LogicalOr(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.LogicalOr)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(true, false)[0]
	if val := y0.Value(); val != true {
		t.Errorf("Expected y0 to be true, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]bool{true, false}, []bool{true})[0]
	if ok, diff := testutil.IsEqual([]bool{true, true}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(true, []bool{true, false})[0]
	if ok, diff := testutil.IsEqual([]bool{true, true}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})[0]
	if ok, diff := testutil.IsEqual([][]bool{{true, true}, {true, true}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})[0]
	if ok, diff := testutil.IsEqual([][]bool{{true, true}, {true, false}, {true, true}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_LogicalXor(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.LogicalXor)

	// Test with scalar (or of size 1) values.
	y0 := exec.MustExec(true, false)[0]
	if val := y0.Value(); val != true {
		t.Errorf("Expected y0 to be true, got %v", val)
	}

	// Test scalar on right side
	y1 := exec.MustExec([]bool{true, false}, []bool{true})[0]
	if ok, diff := testutil.IsEqual([]bool{false, true}, y1.Value()); !ok {
		t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
	}

	// Test scalar on left side
	y2 := exec.MustExec(true, []bool{true, false})[0]
	if ok, diff := testutil.IsEqual([]bool{false, true}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}

	// Test with same sized shapes:
	y3 := exec.MustExec([][]bool{{true, false}, {true, true}}, [][]bool{{true, true}, {false, true}})[0]
	if ok, diff := testutil.IsEqual([][]bool{{false, true}, {true, false}}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test with broadcasting from both sides.
	y4 := exec.MustExec([][]bool{{true}, {false}, {true}}, [][]bool{{true, false}})[0]
	if ok, diff := testutil.IsEqual([][]bool{{false, true}, {true, false}, {false, true}}, y4.Value()); !ok {
		t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
	}
}

func TestExecBinary_Comparison(t *testing.T) {
	// Test Equal
	execEq := graph.MustNewExec(backend, graph.Equal)
	y0 := execEq.MustExec(float32(1.5), float32(1.5))[0]
	if val := y0.Value(); val != true {
		t.Errorf("Expected y0 to be true, got %v", val)
	}
	y1 := execEq.MustExec(bfloat16.FromFloat32(2.0), bfloat16.FromFloat32(2.0))[0]
	if val := y1.Value(); val != true {
		t.Errorf("Expected y1 to be true, got %v", val)
	}
	y2 := execEq.MustExec([]uint16{1, 2, 3}, uint16(2))[0]
	if ok, diff := testutil.IsEqual([]bool{false, true, false}, y2.Value()); !ok {
		t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
	}
	y3 := execEq.MustExec([]int32{5}, []int32{5, 6})[0]
	if ok, diff := testutil.IsEqual([]bool{true, false}, y3.Value()); !ok {
		t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
	}

	// Test GreaterOrEqual
	execGe := graph.MustNewExec(backend, graph.GreaterOrEqual)
	y4 := execGe.MustExec(float32(2.5), float32(1.5))[0]
	if val := y4.Value(); val != true {
		t.Errorf("Expected y4 to be true, got %v", val)
	}
	y5 := execGe.MustExec(bfloat16.FromFloat32(1.0), bfloat16.FromFloat32(2.0))[0]
	if val := y5.Value(); val != false {
		t.Errorf("Expected y5 to be false, got %v", val)
	}
	y6 := execGe.MustExec([]uint16{1, 2, 3}, uint16(2))[0]
	if ok, diff := testutil.IsEqual([]bool{false, true, true}, y6.Value()); !ok {
		t.Errorf("y6 value mismatch (-want +got):\n%s", diff)
	}

	// Test GreaterThan
	execGt := graph.MustNewExec(backend, graph.GreaterThan)
	y7 := execGt.MustExec(float32(2.5), float32(1.5))[0]
	if val := y7.Value(); val != true {
		t.Errorf("Expected y7 to be true, got %v", val)
	}
	y8 := execGt.MustExec([]int32{1, 2, 3}, int32(2))[0]
	if ok, diff := testutil.IsEqual([]bool{false, false, true}, y8.Value()); !ok {
		t.Errorf("y8 value mismatch (-want +got):\n%s", diff)
	}

	// Test LessOrEqual
	execLe := graph.MustNewExec(backend, graph.LessOrEqual)
	y9 := execLe.MustExec(bfloat16.FromFloat32(1.0), bfloat16.FromFloat32(2.0))[0]
	if val := y9.Value(); val != true {
		t.Errorf("Expected y9 to be true, got %v", val)
	}
	y10 := execLe.MustExec([]uint16{1, 2, 3}, uint16(2))[0]
	if ok, diff := testutil.IsEqual([]bool{true, true, false}, y10.Value()); !ok {
		t.Errorf("y10 value mismatch (-want +got):\n%s", diff)
	}

	// Test LessThan
	execLt := graph.MustNewExec(backend, graph.LessThan)
	y11 := execLt.MustExec(float32(1.5), float32(2.5))[0]
	if val := y11.Value(); val != true {
		t.Errorf("Expected y11 to be true, got %v", val)
	}
	y12 := execLt.MustExec([]int32{1, 2, 3}, int32(2))[0]
	if ok, diff := testutil.IsEqual([]bool{true, false, false}, y12.Value()); !ok {
		t.Errorf("y12 value mismatch (-want +got):\n%s", diff)
	}
}

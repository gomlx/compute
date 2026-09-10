// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"slices"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestPadNegative(t *testing.T) {
	// Test Pad with negative start and end padding (cropping)
	builder := backend.Builder("test_pad_negative")
	mainFn := builder.Main()
	xParam, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 5), nil)
	if err != nil {
		t.Fatal(err)
	}
	fillVal, err := mainFn.Constant([]float32{0})
	if err != nil {
		t.Fatal(err)
	}
	padded, err := mainFn.Pad(xParam, fillVal, compute.PadAxis{Start: -1, End: -1})
	if err != nil {
		t.Fatal(err)
	}
	if err := mainFn.Return([]compute.Value{padded}, nil); err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}

	inputBuf := makeBuffer(t, shapes.Make(dtypes.Float32, 5), []float32{10, 20, 30, 40, 50})
	outBufs, err := exec.Execute([]compute.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 3)
	if err := outBufs[0].ToFlatData(got); err != nil {
		t.Fatal(err)
	}
	want := []float32{20, 30, 40}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestPadNegativeAndPositive2D(t *testing.T) {
	// Test Pad 2D with negative start and positive end on axis 0, positive start and negative end on axis 1.
	builder := backend.Builder("test_pad_2d")
	mainFn := builder.Main()
	xParam, err := mainFn.Parameter("x", shapes.Make(dtypes.Int32, 3, 3), nil)
	if err != nil {
		t.Fatal(err)
	}
	fillVal, err := mainFn.Constant([]int32{99})
	if err != nil {
		t.Fatal(err)
	}
	padded, err := mainFn.Pad(xParam, fillVal,
		compute.PadAxis{Start: -1, End: 1}, // dim 0: 3 -> -1 + 1 = 3 (rows 1, 2, fill)
		compute.PadAxis{Start: 1, End: -1}, // dim 1: 3 -> +1 - 1 = 3 (fill, col 0, col 1)
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := mainFn.Return([]compute.Value{padded}, nil); err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}

	inputBuf := makeBuffer(t, shapes.Make(dtypes.Int32, 3, 3), []int32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	outBufs, err := exec.Execute([]compute.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, 9)
	if err := outBufs[0].ToFlatData(got); err != nil {
		t.Fatal(err)
	}
	// Original rows:
	// row 0: 1 2 3 (skipped due to Start: -1)
	// row 1: 4 5 6 -> with axis 1 pad (Start: 1, End: -1): [99, 4, 5]
	// row 2: 7 8 9 -> with axis 1 pad (Start: 1, End: -1): [99, 7, 8]
	// row 3: fill -> [99, 99, 99]
	want := []int32{
		99, 4, 5,
		99, 7, 8,
		99, 99, 99,
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

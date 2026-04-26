// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestBuilder_Compile(t *testing.T) {
	// backend must be exclusive (not shared across tests) for this test to work.
	builder := backend.Builder("test")
	mainFn := builder.Main()
	x, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if x == nil {
		t.Fatalf("unexpected nil value")
	}
	x, err = mainFn.Neg(x)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if x == nil {
		t.Fatalf("unexpected nil value")
	}
	c, err := mainFn.Constant([]int64{1, 2, 3}, 3)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if c == nil {
		t.Fatalf("unexpected nil value")
	}

	err = mainFn.Return([]compute.Value{x, c}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if exec == nil {
		t.Fatalf("unexpected nil value")
	}

	// Check that it fails if fed the wrong number of parameters.
	i0, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	i1, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	_, err = exec.Execute([]compute.Buffer{i0, i1}, []bool{true, true}, 0)
	if err == nil {
		t.Errorf("Expected error when feeding wrong number of parameters")
	}

	// Check that it fails if fed incompatible parameters.
	i0, err = backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	_, err = exec.Execute([]compute.Buffer{i0}, []bool{true}, 0)
	if err == nil {
		t.Errorf("Expected error when feeding incompatible parameters (different size)")
	}

	i0, err = backend.BufferFromFlatData(0, []uint32{1, 2, 3}, shapes.Make(dtypes.Uint32, 3))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	_, err = exec.Execute([]compute.Buffer{i0}, []bool{true}, 0)
	if err == nil {
		t.Errorf("Expected error when feeding incompatible parameters (different dtype)")
	}

	// Checks correct execution with donated inputs, and that the output reused the input buffer.
	i0, err = backend.BufferFromFlatData(0, []float32{3, 5, 7}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	i0Data := i0.(*Buffer).flat.([]float32)
	outputs, err := exec.Execute([]compute.Buffer{i0}, []bool{true}, 0)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if len(outputs) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(outputs))
	}
	if &i0Data[0] != &(outputs[0].(*Buffer).flat.([]float32))[0] {
		t.Errorf("Expected output to reuse the donated input buffer")
	}
	outputShape, err := backend.BufferShape(outputs[1])
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if !outputShape.Equal(shapes.Make(dtypes.Int64, 3)) {
		t.Errorf("Expected output shape %s, got %s", shapes.Make(dtypes.Int64, 3), outputShape)
	}

	// Checks correct execution without donated inputs.
	// Notice the inputs were donated in the last iteration, so we have to set them again.
	i0, err = backend.BufferFromFlatData(0, []float32{3, 5, 7}, shapes.Make(dtypes.Float32, 3))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	outputs, err = exec.Execute([]compute.Buffer{i0}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if len(outputs) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(outputs))
	}
	if i0.(*Buffer) == outputs[0].(*Buffer) {
		t.Errorf("Expected output buffer to be different from input buffer when not donated")
	}
	outputShape, err = backend.BufferShape(outputs[1])
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if !outputShape.Equal(shapes.Make(dtypes.Int64, 3)) {
		t.Errorf("Expected output shape %s, got %s", shapes.Make(dtypes.Int64, 3), outputShape)
	}
}

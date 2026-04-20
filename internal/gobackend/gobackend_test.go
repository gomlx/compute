// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/must"
	"github.com/gomlx/compute/internal/testutil"
	"github.com/gomlx/compute/shapes"
	"k8s.io/klog/v2"
)

var backend compute.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", compute.List())
	// Perform your setup logic here
	if os.Getenv(compute.ConfigEnvVar) == "" {
		must.M(os.Setenv(compute.ConfigEnvVar, "go"))
	} else {
		fmt.Printf("\t$%s=%q\n", compute.ConfigEnvVar, os.Getenv(compute.ConfigEnvVar))
	}
	backend = compute.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

// buildGraph compiles a backend graph from the given input shapes and build function,
// and creates input buffers from the provided data. Used by both test and benchmark helpers.
func buildGraph(inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f compute.Function, params []compute.Value) (compute.Value, error),
) (compute.Executable, []compute.Buffer, error) {
	builder := backend.Builder("test")
	mainFn := builder.Main()

	params := make([]compute.Value, len(inputShapes))
	for i, s := range inputShapes {
		p, err := mainFn.Parameter(fmt.Sprintf("x%d", i), s, nil)
		if err != nil {
			return nil, nil, err
		}
		params[i] = p
	}

	out, err := buildFn(mainFn, params)
	if err != nil {
		return nil, nil, err
	}

	if err := mainFn.Return([]compute.Value{out}, nil); err != nil {
		return nil, nil, err
	}

	exec, err := builder.Compile()
	if err != nil {
		return nil, nil, err
	}

	inputs := make([]compute.Buffer, len(inputDatas))
	for i, data := range inputDatas {
		buf, err := backend.BufferFromFlatData(0, data, inputShapes[i])
		if err != nil {
			return nil, nil, err
		}
		inputs[i] = buf
	}

	return exec, inputs, nil
}

// testBackend builds, compiles, and executes a single-input, single-output backend graph.
func testBackend(t *testing.T, inputShape shapes.Shape, inputData any,
	buildFn func(f compute.Function, param compute.Value) (compute.Value, error),
) *Buffer {
	t.Helper()
	return testBackendMultiInput(t, []shapes.Shape{inputShape}, []any{inputData},
		func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return buildFn(f, params[0])
		},
	)
}

// testBackendMultiInput builds, compiles, and executes a multi-input, single-output backend graph.
func testBackendMultiInput(t *testing.T, inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f compute.Function, params []compute.Value) (compute.Value, error),
) *Buffer {
	t.Helper()
	exec, inputBufs, err := buildGraph(inputShapes, inputDatas, buildFn)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	outputs, err := exec.Execute(inputBufs, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}
	return outputs[0].(*Buffer)
}

func TestDuplicatedOutputNodes(t *testing.T) {
	// Create a builder and a node
	builder := backend.Builder("test_duplicated_outputs")
	mainFn := builder.Main()
	node, err := mainFn.Constant([]float32{1.0, 2.0, 3.0}, 3)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if node == nil {
		t.Fatalf("node is nil")
	}

	// Compile with the same node duplicated as outputs
	// This should create Identity nodes for the duplicate
	err = mainFn.Return([]compute.Value{node, node}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if exec == nil {
		t.Fatalf("exec is nil")
	}

	// Execute with no inputs (since we're using a constant)
	outputs, err := exec.Execute(nil, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if len(outputs) != 2 {
		t.Fatalf("Expected 2 outputs, got %d", len(outputs))
	}

	// Verify that the two output buffers are different (not the same pointer)
	output0 := outputs[0].(*Buffer)
	output1 := outputs[1].(*Buffer)
	if output0 == output1 {
		t.Errorf("duplicated output nodes should yield different buffers")
	}

	// Verify that the underlying flat data slices are also different
	// (they may have the same values but should be different slices)
	flat0 := output0.flat.([]float32)
	flat1 := output1.flat.([]float32)
	if &flat0[0] == &flat1[0] {
		t.Errorf("duplicated output nodes should have different underlying data slices")
	}

	// Verify that the values are correct (both should be [1.0, 2.0, 3.0])
	if ok, diff := testutil.IsEqual([]float32{1.0, 2.0, 3.0}, flat0); !ok {
		t.Errorf("output 0 mismatch:\n%s", diff)
	}
	if ok, diff := testutil.IsEqual([]float32{1.0, 2.0, 3.0}, flat1); !ok {
		t.Errorf("output 1 mismatch:\n%s", diff)
	}

	// Verify shapes are correct
	shape0, err := backend.BufferShape(outputs[0])
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if !shape0.Equal(shapes.Make(dtypes.Float32, 3)) {
		t.Errorf("Expected shape %s, got %s", shapes.Make(dtypes.Float32, 3), shape0)
	}

	shape1, err := backend.BufferShape(outputs[1])
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if !shape1.Equal(shapes.Make(dtypes.Float32, 3)) {
		t.Errorf("Expected shape %s, got %s", shapes.Make(dtypes.Float32, 3), shape1)
	}
}

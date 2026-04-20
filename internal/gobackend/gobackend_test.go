// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/must"
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

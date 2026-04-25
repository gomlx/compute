// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func testExecUnaryOp(t *testing.T, buildFn func(f compute.Function, param compute.Value) (compute.Value, error), input any) *Buffer {
	inputFlat, inputShape := testutil.ToFlatAndShape(input)
	return testBackendMultiInput(t, []shapes.Shape{inputShape}, []any{inputFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return buildFn(f, params[0])
	})
}

func testExecUnaryOpFails(t *testing.T, buildFn func(f compute.Function, param compute.Value) (compute.Value, error), input any) {
	inputFlat, inputShape := testutil.ToFlatAndShape(input)
	_, _, err := buildGraph([]shapes.Shape{inputShape}, []any{inputFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return buildFn(f, params[0])
	})
	if err == nil {
		t.Errorf("Expected error but got nil")
	}
}

func TestExecUnary(t *testing.T) {
	// Most unary tests are now in package backendtest, and called via TestGeneric.
	// Only backend-specific tests (like testing specific dispatching logic) should be here.
}

func TestBackendIsSimpleGo(t *testing.T) {
	if panicked, _ := testutil.Try(func() { _ = backend.(*Backend) }); panicked {
		t.Errorf("Expected no panic when casting backend to *Backend")
	}
}

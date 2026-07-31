// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

type dummyValue struct{}

func (d *dummyValue) Shape() shapes.Shape {
	return shapes.Make(dtypes.Int64)
}

func TestDynamicReshape(t *testing.T) {
	fakeVal := &dummyValue{}

	t.Run("StaticInput_FullyStaticSpecs", func(t *testing.T) {
		operand := shapes.Make(dtypes.Float32, 2, 3, 4) // size 24
		specs := []compute.DynamicDimensionSpec{
			{Static: 4},
			{Static: 6},
		}
		got, err := DynamicReshape(operand, specs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.IsDynamic() {
			t.Fatalf("expected static shape, got dynamic: %s", got)
		}
		if !got.Equal(shapes.Make(dtypes.Float32, 4, 6)) {
			t.Fatalf("expected [4, 6], got %s", got)
		}
	})

	t.Run("StaticInput_InferredDimResolvesToStatic", func(t *testing.T) {
		operand := shapes.Make(dtypes.Float32, 2, 3, 4) // size 24
		specs := []compute.DynamicDimensionSpec{
			{Static: 4},
			{Name: "inferred"}, // Static left untouched (0)
		}
		got, err := DynamicReshape(operand, specs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.IsDynamic() {
			t.Fatalf("expected static shape, got dynamic: %s", got)
		}
		expected := shapes.Make(dtypes.Float32, 4, 6).WithAxisNames("", "inferred")
		if !got.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, got)
		}
	})

	t.Run("StaticInput_WithRuntimeValue_ReturnsDynamic", func(t *testing.T) {
		operand := shapes.Make(dtypes.Float32, 2, 3, 4) // size 24
		specs := []compute.DynamicDimensionSpec{
			{Name: "dyn", Value: fakeVal}, // Static left untouched (0)
			{Name: "inferred"},
		}
		got, err := DynamicReshape(operand, specs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !got.IsDynamic() {
			t.Fatalf("expected dynamic shape, got static: %s", got)
		}
	})

	t.Run("NegativeStaticError", func(t *testing.T) {
		operand := shapes.Make(dtypes.Float32, 2, 3, 4)
		specs := []compute.DynamicDimensionSpec{
			{Static: -1},
		}
		_, err := DynamicReshape(operand, specs)
		if err == nil {
			t.Fatalf("expected error when Static < 0")
		}
	})

	t.Run("NameSetWithNonZeroStaticError", func(t *testing.T) {
		operand := shapes.Make(dtypes.Float32, 2, 3, 4)
		specs := []compute.DynamicDimensionSpec{
			{Name: "batch", Static: 16},
		}
		_, err := DynamicReshape(operand, specs)
		if err == nil {
			t.Fatalf("expected error when Name is set and Static != 0")
		}
	})

	t.Run("MultipleInferredDimsError", func(t *testing.T) {
		operand := shapes.Make(dtypes.Float32, 2, 3, 4)
		specs := []compute.DynamicDimensionSpec{
			{Name: "dim1"},
			{Name: "dim2"},
		}
		_, err := DynamicReshape(operand, specs)
		if err == nil {
			t.Fatalf("expected error for multiple inferred dims")
		}
	})

	t.Run("UnnamedDynamicValueAxisError", func(t *testing.T) {
		operand := shapes.Make(dtypes.Float32, 2, 3, 4)
		specs := []compute.DynamicDimensionSpec{
			{Name: "", Value: fakeVal},
		}
		_, err := DynamicReshape(operand, specs)
		if err == nil {
			t.Fatalf("expected error for unnamed dynamic axis with Value")
		}
	})
}

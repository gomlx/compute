// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// DynamicReshape calculates the output shape resulting from a DynamicReshape operation.
//
// DynamicDimensionSpec must be one of:
//   - Static: `Static >= 0`, `Name == ""`, `Value == nil`.
//   - Dynamic: `Name != ""`, `Value != nil`, `Static == 0`.
//   - Inferred: `Name != ""`, `Value == nil`, `Static == 0` (at most one inferred dimension).
//
// Static dimensions cannot be negative (e.g. -1 is invalid for Static).
func DynamicReshape(operand shapes.Shape, specs []compute.DynamicDimensionSpec) (output shapes.Shape, err error) {
	if len(specs) == 0 {
		return shapes.Invalid(), errors.Errorf("DynamicReshape() requires at least one dimension spec")
	}

	dims := make([]int, len(specs))
	axisNames := make([]string, len(specs))
	inferredIdx := -1

	for i, spec := range specs {
		if spec.Name != "" {
			if spec.Static != 0 {
				return shapes.Invalid(), errors.Errorf("DynamicReshape() spec at index %d has Name %q but Static is %d (must be 0 when Name is set)", i, spec.Name, spec.Static)
			}
			axisNames[i] = spec.Name
			dims[i] = shapes.DynamicDim

			if spec.Value == nil {
				// Dynamic inferred dimension with Name.
				if inferredIdx != -1 {
					return shapes.Invalid(), errors.Errorf("DynamicReshape() allows at most one inferred dimension, but found multiple at indices %d and %d", inferredIdx, i)
				}
				inferredIdx = i
			}
		} else {
			// Name == "" -> Static dimension
			if spec.Value != nil {
				return shapes.Invalid(), errors.Errorf("DynamicReshape() spec at index %d has a non-nil Value but Name is empty", i)
			}
			if spec.Static < 0 {
				return shapes.Invalid(), errors.Errorf("DynamicReshape() static dimension at index %d cannot be negative (%d)", i, spec.Static)
			}
			dims[i] = spec.Static
		}
	}

	// Case 1: Operand is static.
	if !operand.IsDynamic() {
		// Calculate product of all static dimensions in target.
		knownProduct := 1
		hasDynamicValue := false
		for _, spec := range specs {
			if spec.Name != "" {
				if spec.Value != nil {
					hasDynamicValue = true
				}
			} else {
				knownProduct *= spec.Static
			}
		}

		// If there are no dynamic Value nodes, and at most 1 inferred dim, we can resolve statically!
		if !hasDynamicValue {
			if inferredIdx != -1 {
				if knownProduct == 0 {
					return shapes.Invalid(), errors.Errorf("DynamicReshape() cannot infer dimension size when static dimensions multiply to 0")
				}
				if operand.Size()%knownProduct != 0 {
					return shapes.Invalid(), errors.Errorf("DynamicReshape() cannot reshape %s to specs %v: total size %d is not divisible by static product %d", operand, specs, operand.Size(), knownProduct)
				}
				dims[inferredIdx] = operand.Size() / knownProduct
				inferredIdx = -1 // resolved!
			}

			// If inferredIdx resolved (or was never present) and no dynamic values exist, return a static shape.
			if inferredIdx == -1 {
				output = shapes.Make(operand.DType, dims...)
				if operand.Size() != output.Size() {
					return shapes.Invalid(), errors.Errorf("DynamicReshape() shape mismatch: input %s (size %d) != output %s (size %d)", operand, operand.Size(), output, output.Size())
				}
				// If any axis names were provided, preserve them.
				if slices.ContainsFunc(axisNames, func(name string) bool { return name != "" }) {
					output = output.WithAxisNames(axisNames...)
				}
				return output, nil
			}
		}
	}

	// Case 2: Dynamic output shape.
	// Build dynamic shape with axis names.
	output = shapes.MakeDynamic(operand.DType, dims, axisNames)
	return output, nil
}

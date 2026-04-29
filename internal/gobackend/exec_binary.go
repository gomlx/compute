// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"github.com/gomlx/compute/shapes"
)

// This file implements binary operations.
// One optimization supported is specially handling the cases where one of the operands is a scalar (or of size 1),
// in which case it becomes almost a unary operation with a constant value.

// binaryOperandsAndOutput is a convenience function to get the inputs and output -- which may be the reuse of the input.
func binaryOperandsAndOutput(backend *Backend, inputs []*Buffer, inputsOwned []bool, outputShape shapes.Shape) (
	lhs, rhs, output *Buffer, lhsIsScalarOr1, rhsIsScalarOr1 bool) {
	lhs, rhs = inputs[0], inputs[1]
	lhsIsScalarOr1, rhsIsScalarOr1 = lhs.RawShape.Size() == 1, rhs.RawShape.Size() == 1
	switch {
	case inputsOwned[1] && rhs.RawShape.Equal(outputShape):
		output = rhs
		inputs[1] = nil
	case inputsOwned[0] && lhs.RawShape.Equal(outputShape):
		output = lhs
		inputs[0] = nil
	default:
		output, _ = backend.getBufferForShape(outputShape)
	}
	return
}

// execScalarPowIntGeneric is a O(num of bits) for Pow(base, exp) implementation for integers.
func execScalarPowIntGeneric[T PODIntegerConstraints](base, exp T) T {
	result := T(1)
	for exp > 0 {
		if exp%2 == 1 {
			result *= base
		}
		base *= base
		exp >>= 1 // exp /= 2
	}
	return result
}

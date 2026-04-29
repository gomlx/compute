// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"

	"github.com/gomlx/compute/shapes"
)

// TransposeIterator creates a dynamic iterator that yields output flat indices
// for the corresponding flat index on the input operand, assuming the operand flat index is moving
// incrementally.
type TransposeIterator struct {
	flatIdx                                int
	perAxisIdx, perAxisStrides, dimensions []int
}

// NewTransposeIterator creates a dynamic iterator that yields output flat indices
// for the corresponding flat index on the input operand, assuming the operand flat index is moving
// incrementally.
func NewTransposeIterator(operand shapes.Shape, permutations []int) *TransposeIterator {
	rank := operand.Rank()

	it := &TransposeIterator{
		perAxisIdx:     make([]int, rank),
		perAxisStrides: make([]int, rank),
		dimensions:     slices.Clone(operand.Dimensions),
	}

	// First, calculate strides on the output.
	stridesOnOutput := make([]int, rank)
	stride := 1
	reversePermutations := make([]int, rank)
	for outputAxis := rank - 1; outputAxis >= 0; outputAxis-- {
		stridesOnOutput[outputAxis] = stride
		operandAxis := permutations[outputAxis]
		stride *= operand.Dimensions[operandAxis]
		reversePermutations[operandAxis] = outputAxis
	}

	// Calculate per operand axis, what is the stride on the output.
	for operandAxis := range rank {
		outputAxis := reversePermutations[operandAxis]
		it.perAxisStrides[operandAxis] = stridesOnOutput[outputAxis]
	}
	return it
}

// Next returns the next flat index in the output.
func (it *TransposeIterator) Next() int {
	// Store current flatIdx first
	nextFlatIdx := it.flatIdx

	// Cache rank to avoid repeated len() calls
	rank := len(it.perAxisIdx)

	// Use local variables for array access to avoid repeated indirection
	perAxisIdx := it.perAxisIdx
	perAxisStrides := it.perAxisStrides
	dimensions := it.dimensions

	// Handle remaining axes only when needed
	for axis := rank - 1; axis >= 0; axis-- {
		perAxisIdx[axis]++
		it.flatIdx += perAxisStrides[axis]
		if perAxisIdx[axis] < dimensions[axis] {
			// We are done.
			return nextFlatIdx
		}
		perAxisIdx[axis] = 0
		it.flatIdx -= perAxisStrides[axis] * dimensions[axis]
	}

	return nextFlatIdx
}

// TransposeDTypeMap is used to dispatch Transpose to type-specific implementations.
var TransposeDTypeMap = NewDTypeMap("Transpose")

// Package support and its subpackages contains various supporting functionality for gomlx/compute
// that may be also useful for other users or GoMLX projects, as well as to other compute.Backend
// implementations.
package support

import (
	"github.com/gomlx/compute/shapes"
)

// DotGeneralFLOPs estimates the number of FLOPs (Floating Point Operations) for a DotGeneral operation.
// It assumes that the operation is implemented as a matrix multiplication.
func DotGeneralFLOPs(lhsShape shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhsShape shapes.Shape, rhsContractingAxes, rhsBatchAxes []int) int {
	batchSize, lhsCrossSize, contractingSize, _ := DotGeneralFindSizes(lhsShape, lhsContractingAxes, lhsBatchAxes)
	_, rhsCrossSize, _, _ := DotGeneralFindSizes(rhsShape, rhsContractingAxes, rhsBatchAxes)
	return batchSize * lhsCrossSize * rhsCrossSize * contractingSize * 2 // 1 mult + 1 add = 2 ops
}

// DotGeneralFindSizes finds the combined sizes of the 3 types of axes that matter:
// batch, cross, and contracting dimensions for a DotGeneral operation
func DotGeneralFindSizes(shape shapes.Shape, contractingAxes, batchAxes []int) (
	batchSize, crossSize, contractingSize int, crossDims []int) {
	rank := shape.Rank()
	axesTypes := make([]int, rank)

	// Mark axes types: 1 for contracting, 2 for batch
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	// Calculate sizes by multiplying dimensions according to the axis type.
	batchSize, crossSize, contractingSize = 1, 1, 1
	crossDims = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		switch axisType {
		case 0: // Cross axes (unmarked)
			crossSize *= dim
			crossDims = append(crossDims, dim)
		case 1: // Contracting axes
			contractingSize *= dim
		case 2: // Batch axes
			batchSize *= dim
		}
	}
	return
}

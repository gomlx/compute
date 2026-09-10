package dot

import (
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// Layout for inputs of DotGeneral.
type Layout int

//go:generate go tool enumer -type Layout -trimprefix=Layout -output=gen_layout_enumer.go layout.go

const (
	// LayoutNonTransposed, "MatMul", "Row-Major" or "Normal-Transposed" layout,
	// with lhs shaped [B, N, K]; rhs shaped [B, K, M] and output shaped [B, N, M].
	// B is the optional batch axis; N is the lhs "cross" axis; M is the rhs
	// "cross" axis; K is the contracting axis.
	LayoutNonTransposed Layout = iota

	// LayoutTransposed with lhs shaped [B, N, K]; rhs shaped [B, M, K] and
	// output shaped [B, N, M].
	// B is the optional batch axis; N is the lhs "cross" axis; M is the rhs
	// "cross" axis; K is the contracting axis.
	LayoutTransposed

	// LayoutIncompatible indicates that the layout is incompatible with the
	// underlying basic DotGeneral algorithms, and will require a Reshape or
	// Transpose operation before the DotGeneral operation.
	LayoutIncompatible
)

// LayoutForDotGeneral returns the layout for DotGeneral given the shapes of the
// inputs and the axes to contract and batch over.
//
// Memory Layout Expectations:
// The underlying GEMM implementations (CallRegisteredImplementation) operate on flat 1D
// memory buffers. In row-major memory order, adjacent dimensions of the same semantic
// type (batch, cross, or contracting) have linear strides identical to their scalar product.
// Therefore, tensors with any rank can be executed directly by the GEMM kernels as long as:
//   - All batch axes are leading and sequential: [0, 1, ..., numBatchAxes-1] on both LHS and RHS.
//   - Contracting axes match in count and are contiguous in memory.
//   - For LHS: contracting axes are trailing: [B..., N..., K...].
//   - For RHS LayoutTransposed: contracting axes are trailing: [B..., M..., K...].
//   - For RHS LayoutNonTransposed: contracting axes immediately follow batch axes: [B..., K..., M...].
//
// No reshape or axis merging is required when these conditions are met.
func LayoutForDotGeneral(lhsShape shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhsShape shapes.Shape, rhsContractingAxes, rhsBatchAxes []int) Layout {
	numContractingAxes := len(lhsContractingAxes)
	if numContractingAxes == 0 || len(rhsContractingAxes) != numContractingAxes {
		return LayoutIncompatible
	}

	// Batch axes must match in count, and must be leading and sequential (0, 1, 2, ...).
	numBatchAxes := len(lhsBatchAxes)
	if len(rhsBatchAxes) != numBatchAxes {
		return LayoutIncompatible
	}
	for i := range numBatchAxes {
		if lhsBatchAxes[i] != i || rhsBatchAxes[i] != i {
			return LayoutIncompatible
		}
	}

	lhsRank := lhsShape.Rank()
	rhsRank := rhsShape.Rank()

	// LHS contracting axes must be trailing and sequential: [B..., N..., K...]
	for i := range numContractingAxes {
		if lhsContractingAxes[i] != lhsRank-numContractingAxes+i {
			return LayoutIncompatible
		}
	}

	// Check if RHS contracting axes are leading after batch axes [B..., K..., M...] -> LayoutNonTransposed
	isNonTransposed := true
	for i := range numContractingAxes {
		if rhsContractingAxes[i] != numBatchAxes+i {
			isNonTransposed = false
			break
		}
	}
	if isNonTransposed {
		return LayoutNonTransposed
	}

	// Check if RHS contracting axes are trailing [B..., M..., K...] -> LayoutTransposed
	isTransposed := true
	for i := range numContractingAxes {
		if rhsContractingAxes[i] != rhsRank-numContractingAxes+i {
			isTransposed = false
			break
		}
	}
	if isTransposed {
		return LayoutTransposed
	}

	return LayoutIncompatible
}

// AxisType represents the type of an axis, based on its function in the DotGeneral.
type AxisType int

const (
	AxisTypeCross AxisType = iota
	AxisTypeBatch
	AxisTypeContracting
)

type axisInfo struct {
	typ AxisType
	idx int // index in batchAxes or contractingAxes
}

func getAxisInfo(axis int, batchAxes, contractingAxes []int) axisInfo {
	for i, a := range batchAxes {
		if a == axis {
			return axisInfo{AxisTypeBatch, i}
		}
	}
	for i, a := range contractingAxes {
		if a == axis {
			return axisInfo{AxisTypeContracting, i}
		}
	}
	return axisInfo{AxisTypeCross, -1}
}

// transposeSide is a helper to transpose one side to a specific ordering of (batch, cross, contracting).
func transposeSide(f *gobackend.Function, node *gobackend.Node, contractingAxes, batchAxes []int, layout Layout) (*gobackend.Node, []int, []int, error) {
	rank := node.Shape.Rank()
	isContracting := make([]bool, rank)
	for _, axis := range contractingAxes {
		isContracting[axis] = true
	}
	isBatch := make([]bool, rank)
	for _, axis := range batchAxes {
		isBatch[axis] = true
	}

	crossAxes := make([]int, 0, rank)
	for axis := 0; axis < rank; axis++ {
		if !isContracting[axis] && !isBatch[axis] {
			crossAxes = append(crossAxes, axis)
		}
	}

	var perm []int
	switch layout {
	case LayoutTransposed:
		// Batch, Cross, Contracting
		perm = append(perm, batchAxes...)
		perm = append(perm, crossAxes...)
		perm = append(perm, contractingAxes...)
	case LayoutNonTransposed:
		// Batch, Contracting, Cross
		perm = append(perm, batchAxes...)
		perm = append(perm, contractingAxes...)
		perm = append(perm, crossAxes...)
	default:
		return nil, nil, nil, errors.Errorf("unsupported layout %v in transposeSide", layout)
	}

	// Check if already in order
	alreadyInOrder := true
	for i, p := range perm {
		if p != i {
			alreadyInOrder = false
			break
		}
	}

	newNode := node
	if !alreadyInOrder {
		transposed, err := f.Transpose(node, perm...)
		if err != nil {
			return nil, nil, nil, err
		}
		newNode = transposed.(*gobackend.Node)
	}

	// Calculate new axes
	newBatchAxes := make([]int, len(batchAxes))
	for i := range batchAxes {
		newBatchAxes[i] = i
	}

	newContractingAxes := make([]int, len(contractingAxes))
	if layout == LayoutTransposed {
		start := len(batchAxes) + len(crossAxes)
		for i := range contractingAxes {
			newContractingAxes[i] = start + i
		}
	} else {
		start := len(batchAxes)
		for i := range contractingAxes {
			newContractingAxes[i] = start + i
		}
	}

	return newNode, newContractingAxes, newBatchAxes, nil
}

// TransposeToLayout transposes the axes of lhs and rhs so that their semantic groups
// (batch, cross, and contracting axes) form contiguous blocks matching the target layout:
//   - LHS: [Batch..., Cross..., Contracting...] (LayoutTransposed)
//   - RHS: [Batch..., Cross..., Contracting...] (LayoutTransposed) or [Batch..., Contracting..., Cross...] (LayoutNonTransposed)
//
// Memory Layout Invariant:
// In row-major format, any sequence of contiguous dimensions of the same semantic category
// (e.g. B0, B1 or N0, N1, N2 or K0, K1) has flat memory offsets that are identical to a single
// collapsed dimension with size equal to their product. The GEMM implementation operates directly
// on the flat data slices using the total combined sizes (batchSize, crossSize, contractingSize).
// Therefore, no physical or logical MergeAxes reshape is needed; keeping the original (unmerged)
// dimensions avoids reshape overhead and seamlessly supports dynamic dimensions.
func TransposeToLayout(f *gobackend.Function,
	lhs *gobackend.Node, lhsContractingAxes, lhsBatchAxes []int,
	rhs *gobackend.Node, rhsContractingAxes, rhsBatchAxes []int,
	layout Layout) (
	newLhs *gobackend.Node, newLhsContractingAxes, newLhsBatchAxes []int,
	newRhs *gobackend.Node, newRhsContractingAxes, newRhsBatchAxes []int,
	err error) {

	newLhs, newLhsContractingAxes, newLhsBatchAxes, err = transposeSide(f, lhs, lhsContractingAxes, lhsBatchAxes, LayoutTransposed)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	newRhs, newRhsContractingAxes, newRhsBatchAxes, err = transposeSide(f, rhs, rhsContractingAxes, rhsBatchAxes, layout)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	return newLhs, newLhsContractingAxes, newLhsBatchAxes,
		newRhs, newRhsContractingAxes, newRhsBatchAxes,
		nil
}

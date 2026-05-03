package dot

import (
	"github.com/gomlx/compute/internal/gobackend"
)

type revertMergeAxesFunc func(result *gobackend.Node) (*gobackend.Node, error)

type axisType int

const (
	axisTypeCross axisType = iota
	axisTypeBatch
	axisTypeContracting
)

type axisInfo struct {
	typ axisType
	idx int // index in batchAxes or contractingAxes
}

func getAxisInfo(axis int, batchAxes, contractingAxes []int) axisInfo {
	for i, a := range batchAxes {
		if a == axis {
			return axisInfo{axisTypeBatch, i}
		}
	}
	for i, a := range contractingAxes {
		if a == axis {
			return axisInfo{axisTypeContracting, i}
		}
	}
	return axisInfo{axisTypeCross, -1}
}

// MergeAxes checks if adjacent axes in lhs and rhs are used for the same purpose
// (batch, contracting or cross), and if so, merges them by reshaping.
// This simplifies the shape of the tensors, making it more likely that the
// operation can be matched to a fast execution path (like SmallMatMul or Packgemm).
//
// It returns the reshaped lhs and rhs, the new contracting and batch axes, and a function
// to revert the merging on the final result.
func MergeAxes(f *gobackend.Function,
	lhs *gobackend.Node, lhsContractingAxes, lhsBatchAxes []int,
	rhs *gobackend.Node, rhsContractingAxes, rhsBatchAxes []int) (
	newLhs *gobackend.Node, newLhsContractingAxes, newLhsBatchAxes []int,
	newRhs *gobackend.Node, newRhsContractingAxes, newRhsBatchAxes []int,
	revertFn revertMergeAxesFunc, err error) {

	lhsRank := lhs.Shape.Rank()
	rhsRank := rhs.Shape.Rank()

	// 1. Group adjacent mergable axes in LHS
	var lhsGroups [][]int
	if lhsRank > 0 {
		currentGroup := []int{0}
		for i := 1; i < lhsRank; i++ {
			prev := currentGroup[len(currentGroup)-1]
			infoPrev := getAxisInfo(prev, lhsBatchAxes, lhsContractingAxes)
			infoCurr := getAxisInfo(i, lhsBatchAxes, lhsContractingAxes)

			canMerge := false
			if infoPrev.typ == infoCurr.typ {
				if infoPrev.typ == axisTypeCross {
					canMerge = true
				} else if infoPrev.typ == axisTypeBatch {
					// Batch axes must be adjacent in the axes list and in correct order.
					if infoCurr.idx == infoPrev.idx+1 {
						rhsPrev := rhsBatchAxes[infoPrev.idx]
						rhsCurr := rhsBatchAxes[infoCurr.idx]
						// The corresponding RHS axes must be physically adjacent and in order.
						if rhsCurr == rhsPrev+1 {
							canMerge = true
						}
					}
				} else if infoPrev.typ == axisTypeContracting {
					// Contracting axes must map to adjacent physical axes in RHS.
					rhsPrev := rhsContractingAxes[infoPrev.idx]
					rhsCurr := rhsContractingAxes[infoCurr.idx]
					if rhsCurr == rhsPrev+1 {
						canMerge = true
					}
				}
			}

			if canMerge {
				currentGroup = append(currentGroup, i)
			} else {
				lhsGroups = append(lhsGroups, currentGroup)
				currentGroup = []int{i}
			}
		}
		if len(currentGroup) > 0 {
			lhsGroups = append(lhsGroups, currentGroup)
		}
	}

	// 2. Group adjacent mergable axes in RHS
	var rhsGroups [][]int
	if rhsRank > 0 {
		currentGroup := []int{0}
		for j := 1; j < rhsRank; j++ {
			prev := currentGroup[len(currentGroup)-1]
			infoPrev := getAxisInfo(prev, rhsBatchAxes, rhsContractingAxes)
			infoCurr := getAxisInfo(j, rhsBatchAxes, rhsContractingAxes)

			canMerge := false
			if infoPrev.typ == infoCurr.typ {
				if infoPrev.typ == axisTypeCross {
					canMerge = true
				} else if infoPrev.typ == axisTypeBatch {
					// Check if LHS merged them
					if infoCurr.idx == infoPrev.idx+1 {
						lhsPrev := lhsBatchAxes[infoPrev.idx]
						lhsCurr := lhsBatchAxes[infoCurr.idx]
						if lhsCurr == lhsPrev+1 {
							canMerge = true
						}
					}
				} else if infoPrev.typ == axisTypeContracting {
					// Check if LHS merged them
					lhsPrev := lhsContractingAxes[infoPrev.idx]
					lhsCurr := lhsContractingAxes[infoCurr.idx]
					if lhsCurr == lhsPrev+1 {
						canMerge = true
					}
				}
			}

			if canMerge {
				currentGroup = append(currentGroup, j)
			} else {
				rhsGroups = append(rhsGroups, currentGroup)
				currentGroup = []int{j}
			}
		}
		if len(currentGroup) > 0 {
			rhsGroups = append(rhsGroups, currentGroup)
		}
	}

	// Calculate new shapes
	newLhsDims := make([]int, len(lhsGroups))
	lhsOldToNew := make([]int, lhsRank)
	for newAxis, group := range lhsGroups {
		size := 1
		for _, oldAxis := range group {
			size *= lhs.Shape.Dimensions[oldAxis]
			lhsOldToNew[oldAxis] = newAxis
		}
		newLhsDims[newAxis] = size
	}

	newRhsDims := make([]int, len(rhsGroups))
	rhsOldToNew := make([]int, rhsRank)
	for newAxis, group := range rhsGroups {
		size := 1
		for _, oldAxis := range group {
			size *= rhs.Shape.Dimensions[oldAxis]
			rhsOldToNew[oldAxis] = newAxis
		}
		newRhsDims[newAxis] = size
	}

	// Reshape nodes if needed
	newLhs = lhs
	if len(newLhsDims) != lhsRank {
		reshaped, err := f.Reshape(lhs, newLhsDims...)
		if err != nil {
			return nil, nil, nil, nil, nil, nil, nil, err
		}
		newLhs = reshaped.(*gobackend.Node)
	}

	newRhs = rhs
	if len(newRhsDims) != rhsRank {
		reshaped, err := f.Reshape(rhs, newRhsDims...)
		if err != nil {
			return nil, nil, nil, nil, nil, nil, nil, err
		}
		newRhs = reshaped.(*gobackend.Node)
	}

	// Generate new batch and contracting axes mapping
	newLhsBatchAxes = make([]int, 0)
	for _, oldAxis := range lhsBatchAxes {
		newAxis := lhsOldToNew[oldAxis]
		if len(newLhsBatchAxes) == 0 || newLhsBatchAxes[len(newLhsBatchAxes)-1] != newAxis {
			newLhsBatchAxes = append(newLhsBatchAxes, newAxis)
		}
	}

	newRhsBatchAxes = make([]int, 0)
	for _, oldAxis := range rhsBatchAxes {
		newAxis := rhsOldToNew[oldAxis]
		if len(newRhsBatchAxes) == 0 || newRhsBatchAxes[len(newRhsBatchAxes)-1] != newAxis {
			newRhsBatchAxes = append(newRhsBatchAxes, newAxis)
		}
	}

	newLhsContractingAxes = make([]int, 0)
	newRhsContractingAxes = make([]int, 0)
	seenLhsContracting := make(map[int]bool)
	for i := range lhsContractingAxes {
		newLhsAxis := lhsOldToNew[lhsContractingAxes[i]]
		newRhsAxis := rhsOldToNew[rhsContractingAxes[i]]
		if !seenLhsContracting[newLhsAxis] {
			seenLhsContracting[newLhsAxis] = true
			newLhsContractingAxes = append(newLhsContractingAxes, newLhsAxis)
			newRhsContractingAxes = append(newRhsContractingAxes, newRhsAxis)
		}
	}

	// Provide revert function
	revertFn = func(result *gobackend.Node) (*gobackend.Node, error) {
		return result, nil // This will be wrapped by the caller to reshape to the original dimensions
	}

	return newLhs, newLhsContractingAxes, newLhsBatchAxes,
		newRhs, newRhsContractingAxes, newRhsBatchAxes,
		revertFn, nil
}

package gobackend

import (
	"iter"
	"slices"

	"github.com/gomlx/compute/internal/exceptions"
	"github.com/gomlx/compute/shapes"
)

// BroadcastIterator allows iteration over the flat indices of the target shape of a broadcast (where some axis
// dimensions grow)
//
// It is used by implicit broadcasting in binary ops and by the BroadcastInDim.
type BroadcastIterator struct {
	tgtSize     int
	srcStrides  []int
	isBroadcast []bool
	tgtDims     []int
}

// NewBroadcastIterator returns an iterator contructor over the flat indices of the target shape of a broadcast (where
// some axis dimensions grow).
//
// Pre-requisite: srcShape.Rank() == tgtShape.Rank().
//
// It is used by implicit broadcasting in binary ops and by the BroadcastInDim.
func NewBroadcastIterator(srcShape, tgtShape shapes.Shape) *BroadcastIterator {
	rank := srcShape.Rank()
	if rank != tgtShape.Rank() {
		exceptions.Panicf("broadcastIterator: rank mismatch srcShape=%s, tgtShape=%s", srcShape, tgtShape)
	}
	bi := &BroadcastIterator{
		tgtSize:     tgtShape.Size(),
		tgtDims:     slices.Clone(tgtShape.Dimensions),
		isBroadcast: make([]bool, rank),
		srcStrides:  srcShape.Strides(),
	}
	for axis := range rank {
		bi.isBroadcast[axis] = srcShape.Dimensions[axis] != tgtShape.Dimensions[axis]
	}
	return bi
}

// IterFlatIndices iterate over the source and target flat indices for broadcast.
// Notice since we are broadcasting, the target index will always be incremented by 1,
// while the source index may repeat (when it is being broadcast).
func (bi *BroadcastIterator) IterFlatIndices() iter.Seq2[int, int] {
	return func(yield func(srcIdx, tgtIdx int) bool) {
		rank := len(bi.tgtDims)
		perAxesIdx := make([]int, rank)
		srcFlatIdx := 0
		for dstFlatIdx := range bi.tgtSize {
			// Yield first.
			if !yield(srcFlatIdx, dstFlatIdx) {
				return
			}

			// Bump to next srcFlatIdx.
			srcFlatIdx++
			for axis := rank - 1; axis >= 0; axis-- {
				perAxesIdx[axis]++
				if perAxesIdx[axis] < bi.tgtDims[axis] {
					if bi.isBroadcast[axis] {
						// If we are broadcasting on this axis, we need to go back and repeat the same slice of the tensor.
						srcFlatIdx -= bi.srcStrides[axis]
					}
					break
				}
				perAxesIdx[axis] = 0
			}
		}
	}
}

// ZippedIndices of two BroadcastIterators.
type ZippedIndices struct {
	LHSFlatIdx, RHSFlatIdx int
	TgtFlatIdx             int
}

// ZipIterator allows iteration over the flat indices of the target shape of a broadcast (where some axis
// dimensions grow) for two input shapes (LHS and RHS).
//
// It implements the implicit broadcasting for binary ops like Add, Sub, Mul, Div, etc.
type ZipIterator struct {
	tgtSize        int
	tgtDims        []int
	lhsStrides     []int
	rhsStrides     []int
	lhsIsBroadcast []bool
	rhsIsBroadcast []bool
}

// NewZippedBroadcastIterator returns an iterator constructor over the flat indices of the target shape of a broadcast (where
// some axis dimensions grow) for two input shapes (LHS and RHS).
//
// Pre-requisite: lhsShape.Rank() == tgtShape.Rank() && rhsShape.Rank() == tgtShape.Rank().
//
// It implements the implicit broadcasting for binary ops like Add, Sub, Mul, Div, etc.
func NewZippedBroadcastIterator(lhsShape, rhsShape, tgtShape shapes.Shape) *ZipIterator {
	rank := tgtShape.Rank()
	if lhsShape.Rank() != rank || rhsShape.Rank() != rank {
		exceptions.Panicf("zippedBroadcastIterator: rank mismatch lhsShape=%s, rhsShape=%s, tgtShape=%s", lhsShape, rhsShape, tgtShape)
	}

	zi := &ZipIterator{
		tgtSize:        tgtShape.Size(),
		tgtDims:        slices.Clone(tgtShape.Dimensions),
		lhsStrides:     lhsShape.Strides(),
		rhsStrides:     rhsShape.Strides(),
		lhsIsBroadcast: make([]bool, rank),
		rhsIsBroadcast: make([]bool, rank),
	}
	for axis := range rank {
		zi.lhsIsBroadcast[axis] = lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis]
		zi.rhsIsBroadcast[axis] = rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis]
	}
	return zi
}

// EqualNodeData implements NodeDataComparable, so this can be used as NodeData -- the comparison
// is used for deduping repeated nodes during model building.
func (z *ZipIterator) EqualNodeData(other NodeDataComparable) bool {
	if z == nil && other == nil {
		return true
	}
	if z == nil || other == nil {
		return false
	}
	otherZ, ok := other.(*ZipIterator)
	if !ok {
		return false
	}
	return z.tgtSize == otherZ.tgtSize &&
		slices.Equal(z.tgtDims, otherZ.tgtDims) &&
		slices.Equal(z.lhsStrides, otherZ.lhsStrides) &&
		slices.Equal(z.rhsStrides, otherZ.rhsStrides) &&
		slices.Equal(z.lhsIsBroadcast, otherZ.lhsIsBroadcast) &&
		slices.Equal(z.rhsIsBroadcast, otherZ.rhsIsBroadcast)
}

// IterFlatIndices iterate over the lhs, rhs and target flat indices for broadcast.
// Notice since we are broadcasting, the target index will always be incremented by 1,
// while the source indices may repeat (when it is being broadcast).
func (zi *ZipIterator) IterFlatIndices() iter.Seq[ZippedIndices] {
	return func(yield func(ZippedIndices) bool) {
		rank := len(zi.tgtDims)
		perAxesIdx := make([]int, rank)
		lhsFlatIdx := 0
		rhsFlatIdx := 0

		for dstFlatIdx := range zi.tgtSize {
			// Yield first.
			if !yield(ZippedIndices{LHSFlatIdx: lhsFlatIdx, RHSFlatIdx: rhsFlatIdx, TgtFlatIdx: dstFlatIdx}) {
				return
			}

			// Bump to next
			lhsFlatIdx++
			rhsFlatIdx++
			for axis := rank - 1; axis >= 0; axis-- {
				perAxesIdx[axis]++
				if perAxesIdx[axis] < zi.tgtDims[axis] {
					if zi.lhsIsBroadcast[axis] {
						lhsFlatIdx -= zi.lhsStrides[axis]
					}
					if zi.rhsIsBroadcast[axis] {
						rhsFlatIdx -= zi.rhsStrides[axis]
					}
					break
				}
				perAxesIdx[axis] = 0
			}
		}
	}
}

// BroadcastPattern represents the classified pattern of broadcasting between two operands.
type BroadcastPattern int8

const (
	// BroadcastNone indicates both operands have identical shapes (no broadcasting).
	BroadcastNone BroadcastPattern = iota

	// BroadcastScalarRHS indicates RHS has size 1 (scalar broadcast).
	BroadcastScalarRHS

	// BroadcastScalarLHS indicates LHS has size 1 (scalar broadcast).
	BroadcastScalarLHS

	// BroadcastLeadingRHS indicates RHS has leading 1s collapsed into [1, B] and LHS is [A, B].
	BroadcastLeadingRHS

	// BroadcastLeadingLHS indicates LHS has leading 1s collapsed into [1, B] and RHS is [A, B].
	BroadcastLeadingLHS

	// BroadcastTrailingRHS indicates RHS has trailing 1s collapsed into [A, 1] and LHS is [A, B].
	BroadcastTrailingRHS

	// BroadcastTrailingLHS indicates LHS has trailing 1s collapsed into [A, 1] and RHS is [A, B].
	BroadcastTrailingLHS

	// BroadcastRowCol indicates LHS is collapsed into [A, 1] and RHS into [1, B], producing [A, B].
	BroadcastRowCol

	// BroadcastColRow indicates LHS is collapsed into [1, B] and RHS into [A, 1], producing [A, B].
	BroadcastColRow

	// BroadcastGeneral indicates arbitrary multi-axis broadcasting.
	BroadcastGeneral
)

// BroadcastConfig contains pre-computed broadcast pattern metadata.
type BroadcastConfig struct {
	Pattern BroadcastPattern
	A, B    int
	ZipIter *ZipIterator
}

// Compile-time checks.
var (
	_ NodeDataComparable   = (*BroadcastConfig)(nil)
	_ RecomputableNodeData = (*BroadcastConfig)(nil)
)

// EqualNodeData implements NodeDataComparable.
func (c *BroadcastConfig) EqualNodeData(other NodeDataComparable) bool {
	if c == nil && other == nil {
		return true
	}
	if c == nil || other == nil {
		return false
	}
	otherC, ok := other.(*BroadcastConfig)
	if !ok {
		return false
	}
	if c.Pattern != otherC.Pattern || c.A != otherC.A || c.B != otherC.B {
		return false
	}
	if c.ZipIter != nil || otherC.ZipIter != nil {
		return c.ZipIter.EqualNodeData(otherC.ZipIter)
	}
	return true
}

// Recompute implements RecomputableNodeData.
func (c *BroadcastConfig) Recompute(backend *Backend, resolvedNodes []*Node, originalNode *Node) (any, error) {
	lhs := resolvedNodes[originalNode.Inputs[0].Index]
	rhs := resolvedNodes[originalNode.Inputs[1].Index]
	cfg := DetermineBroadcastConfig(lhs.Shape, rhs.Shape, originalNode.Shape)
	return &cfg, nil
}

// DetermineBroadcastConfig classifies the broadcast pattern between lhsShape and rhsShape to produce tgtShape.
func DetermineBroadcastConfig(lhsShape, rhsShape, tgtShape shapes.Shape) BroadcastConfig {
	if lhsShape.IsDynamic() || rhsShape.IsDynamic() || tgtShape.IsDynamic() {
		return BroadcastConfig{Pattern: BroadcastGeneral}
	}

	if lhsShape.Equal(rhsShape) {
		return BroadcastConfig{Pattern: BroadcastNone}
	}

	if rhsShape.Size() == 1 {
		return BroadcastConfig{Pattern: BroadcastScalarRHS}
	}
	if lhsShape.Size() == 1 {
		return BroadcastConfig{Pattern: BroadcastScalarLHS}
	}

	rank := tgtShape.Rank()
	if lhsShape.Rank() != rank || rhsShape.Rank() != rank {
		// Should not happen for standard binary ops since shape inference aligns ranks,
		// but fallback gracefully.
		return BroadcastConfig{
			Pattern: BroadcastGeneral,
			ZipIter: NewZippedBroadcastIterator(lhsShape, rhsShape, tgtShape),
		}
	}

	// 1. Check BroadcastLeadingRHS: rhs has leading 1s, matching suffix.
	kLeadingRHS := -1
	for axis := range rank {
		if rhsShape.Dimensions[axis] > 1 {
			kLeadingRHS = axis
			break
		}
	}
	if kLeadingRHS > 0 {
		isMatch := true
		for axis := range kLeadingRHS {
			if rhsShape.Dimensions[axis] != 1 || lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
				isMatch = false
				break
			}
		}
		if isMatch {
			for axis := kLeadingRHS; axis < rank; axis++ {
				if rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] || lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
					isMatch = false
					break
				}
			}
		}
		if isMatch {
			a := 1
			for axis := range kLeadingRHS {
				a *= tgtShape.Dimensions[axis]
			}
			b := 1
			for axis := kLeadingRHS; axis < rank; axis++ {
				b *= tgtShape.Dimensions[axis]
			}
			if a > 1 && b > 0 {
				return BroadcastConfig{Pattern: BroadcastLeadingRHS, A: a, B: b}
			}
		}
	}

	// 2. Check BroadcastLeadingLHS: lhs has leading 1s, matching suffix.
	kLeadingLHS := -1
	for axis := range rank {
		if lhsShape.Dimensions[axis] > 1 {
			kLeadingLHS = axis
			break
		}
	}
	if kLeadingLHS > 0 {
		isMatch := true
		for axis := range kLeadingLHS {
			if lhsShape.Dimensions[axis] != 1 || rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
				isMatch = false
				break
			}
		}
		if isMatch {
			for axis := kLeadingLHS; axis < rank; axis++ {
				if lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] || rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
					isMatch = false
					break
				}
			}
		}
		if isMatch {
			a := 1
			for axis := range kLeadingLHS {
				a *= tgtShape.Dimensions[axis]
			}
			b := 1
			for axis := kLeadingLHS; axis < rank; axis++ {
				b *= tgtShape.Dimensions[axis]
			}
			if a > 1 && b > 0 {
				return BroadcastConfig{Pattern: BroadcastLeadingLHS, A: a, B: b}
			}
		}
	}

	// 3. Check BroadcastTrailingRHS: rhs has trailing 1s, matching prefix.
	kTrailingRHS := -1
	for axis := rank - 1; axis >= 0; axis-- {
		if rhsShape.Dimensions[axis] > 1 {
			kTrailingRHS = axis
			break
		}
	}
	if kTrailingRHS >= 0 && kTrailingRHS < rank-1 {
		split := kTrailingRHS + 1
		isMatch := true
		for axis := range split {
			if lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] || rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
				isMatch = false
				break
			}
		}
		if isMatch {
			for axis := split; axis < rank; axis++ {
				if rhsShape.Dimensions[axis] != 1 || lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
					isMatch = false
					break
				}
			}
		}
		if isMatch {
			a := 1
			for axis := range split {
				a *= tgtShape.Dimensions[axis]
			}
			b := 1
			for axis := split; axis < rank; axis++ {
				b *= tgtShape.Dimensions[axis]
			}
			if a > 0 && b > 1 {
				return BroadcastConfig{Pattern: BroadcastTrailingRHS, A: a, B: b}
			}
		}
	}

	// 4. Check BroadcastTrailingLHS: lhs has trailing 1s, matching prefix.
	kTrailingLHS := -1
	for axis := rank - 1; axis >= 0; axis-- {
		if lhsShape.Dimensions[axis] > 1 {
			kTrailingLHS = axis
			break
		}
	}
	if kTrailingLHS >= 0 && kTrailingLHS < rank-1 {
		split := kTrailingLHS + 1
		isMatch := true
		for axis := range split {
			if lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] || rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
				isMatch = false
				break
			}
		}
		if isMatch {
			for axis := split; axis < rank; axis++ {
				if lhsShape.Dimensions[axis] != 1 || rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
					isMatch = false
					break
				}
			}
		}
		if isMatch {
			a := 1
			for axis := range split {
				a *= tgtShape.Dimensions[axis]
			}
			b := 1
			for axis := split; axis < rank; axis++ {
				b *= tgtShape.Dimensions[axis]
			}
			if a > 0 && b > 1 {
				return BroadcastConfig{Pattern: BroadcastTrailingLHS, A: a, B: b}
			}
		}
	}

	// 5. Check BroadcastRowCol: LHS is [A, 1] and RHS is [1, B], producing [A, B].
	for split := 1; split < rank; split++ {
		isRowCol := true
		for axis := range split {
			if lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] || rhsShape.Dimensions[axis] != 1 {
				isRowCol = false
				break
			}
		}
		if isRowCol {
			for axis := split; axis < rank; axis++ {
				if lhsShape.Dimensions[axis] != 1 || rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
					isRowCol = false
					break
				}
			}
		}
		if isRowCol {
			a := 1
			for axis := range split {
				a *= tgtShape.Dimensions[axis]
			}
			b := 1
			for axis := split; axis < rank; axis++ {
				b *= tgtShape.Dimensions[axis]
			}
			if a > 1 && b > 1 {
				return BroadcastConfig{Pattern: BroadcastRowCol, A: a, B: b}
			}
		}
	}

	// 6. Check BroadcastColRow: LHS is [1, B] and RHS is [A, 1], producing [A, B].
	for split := 1; split < rank; split++ {
		isColRow := true
		for axis := range split {
			if lhsShape.Dimensions[axis] != 1 || rhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] {
				isColRow = false
				break
			}
		}
		if isColRow {
			for axis := split; axis < rank; axis++ {
				if lhsShape.Dimensions[axis] != tgtShape.Dimensions[axis] || rhsShape.Dimensions[axis] != 1 {
					isColRow = false
					break
				}
			}
		}
		if isColRow {
			a := 1
			for axis := range split {
				a *= tgtShape.Dimensions[axis]
			}
			b := 1
			for axis := split; axis < rank; axis++ {
				b *= tgtShape.Dimensions[axis]
			}
			if a > 1 && b > 1 {
				return BroadcastConfig{Pattern: BroadcastColRow, A: a, B: b}
			}
		}
	}

	// Fallback to general broadcast.
	return BroadcastConfig{
		Pattern: BroadcastGeneral,
		ZipIter: NewZippedBroadcastIterator(lhsShape, rhsShape, tgtShape),
	}
}

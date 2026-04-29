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

	srcFlatIdx int
	perAxesIdx []int
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
		perAxesIdx:  make([]int, rank),
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

// ZipppedBroadcastIterators iterates over two BroadcastIterators in parallel, yielding the zipped indices.
func ZippedBroadcastIterators(lhs, rhs *BroadcastIterator) iter.Seq[ZippedIndices] {
	return func(yield func(ZippedIndices) bool) {
		next1, stop1 := iter.Pull2(lhs.IterFlatIndices())
		defer stop1()

		next2, stop2 := iter.Pull2(rhs.IterFlatIndices())
		defer stop2()

		for {
			var zipped ZippedIndices
			var ok1, ok2 bool
			zipped.LHSFlatIdx, zipped.TgtFlatIdx, ok1 = next1()
			zipped.RHSFlatIdx, _, ok2 = next2()
			if !ok1 || !ok2 {
				break
			}
			if !yield(zipped) {
				break
			}
		}
	}
}

// Reset resets the iterator.
// Only needed if BroadcastIterator is used more than once.
func (bi *BroadcastIterator) Reset() {
	bi.srcFlatIdx = 0
	for i := range bi.perAxesIdx {
		bi.perAxesIdx[i] = 0
	}
}

// NextSrcIndex returns the next source index.
// It must be called sequentially.
func (bi *BroadcastIterator) Next() (srcFlatIdx int) {
	srcFlatIdx = bi.srcFlatIdx
	bi.srcFlatIdx++
	rank := len(bi.perAxesIdx)
	for axis := rank - 1; axis >= 0; axis-- {
		bi.perAxesIdx[axis]++
		if bi.perAxesIdx[axis] < bi.tgtDims[axis] {
			if bi.isBroadcast[axis] {
				// If we are broadcasting on this axis, we need to go back and repeat the same slice of the tensor.
				bi.srcFlatIdx -= bi.srcStrides[axis]
			}
			break
		}
		bi.perAxesIdx[axis] = 0
	}
	return
}

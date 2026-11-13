// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterSlice.Register(Slice, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeSlice, gobackend.PriorityGeneric, execSlice)
}

// sliceNode is attached to the gobackend.Node.data field for Slice.
type sliceNode struct {
	starts, limits, strides []int
}

// Recompute implements gobackend.RecomputableNodeData for sliceNode.
func (s *sliceNode) Recompute(backend *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
	operandShape := resolvedNodes[originalNode.Inputs[0].Index].Shape

	needsUpdate := false
	for i, lim := range s.limits {
		if lim == shapes.DynamicDim {
			if operandShape.Dimensions[i] == shapes.DynamicDim {
				return nil, errors.Errorf("recomputeSliceData: operand axis %d is still DynamicDim after resolution", i)
			}
			needsUpdate = true
			break
		}
	}
	if !needsUpdate {
		return s, nil
	}

	newData := &sliceNode{
		starts:  slices.Clone(s.starts),
		limits:  slices.Clone(s.limits),
		strides: slices.Clone(s.strides),
	}
	for i, lim := range newData.limits {
		if lim == shapes.DynamicDim {
			newData.limits[i] = operandShape.Dimensions[i]
		}
	}
	return newData, nil
}

// EqualNodeData implements nodeDataComparable for sliceNode.
func (s *sliceNode) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*sliceNode)
	return slices.Equal(s.starts, o.starts) &&
		slices.Equal(s.limits, o.limits) &&
		slices.Equal(s.strides, o.strides)
}

// Slice extracts a subarray from the input array.
//
// The subarray is of the same rank as the input and contains the values inside a bounding box within the input array
// where the dimensions and indices of the bounding box are given as arguments to the slice operation.
//
// The strides set the input stride of the slice in each axis and must be >= 1.
// It is optional, and if missing, it is assumed to be 1 for every dimension.
//
// The limits are defined on the x axes, and they are exclusive upper bounds, i.e. the slice includes
// elements from starts up to (but not including) limits.
//
// Examples:
//
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
func Slice(f *gobackend.Function, operandOp compute.Value, starts, limits, strides []int) (compute.Value, error) {
	opType := compute.OpTypeSlice
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	rank := operand.Shape.Rank()

	if len(starts) != rank || len(limits) != rank || (len(strides) != 0 && len(strides) != rank) {
		return nil, errors.Errorf("Slice op %s: starts, limits and strides (if provided) must have the same rank as the operand (%d), but got %d, %d and %d",
			opType, rank, len(starts), len(limits), len(strides))
	}

	data := &sliceNode{
		starts:  make([]int, rank),
		limits:  make([]int, rank),
		strides: make([]int, rank),
	}
	for axis := 0; axis < rank; axis++ {
		dimSize := operand.Shape.Dimensions[axis]

		// Start
		start := starts[axis]
		if dimSize != shapes.DynamicDim && start < 0 {
			start = dimSize + start
		}
		data.starts[axis] = start

		// Limit
		limit := limits[axis]
		if dimSize != shapes.DynamicDim && limit < 0 {
			limit = dimSize + limit
		}
		data.limits[axis] = limit

		// Stride
		stride := 1
		if len(strides) > axis {
			stride = max(1, strides[axis])
		}
		data.strides[axis] = stride
	}

	// Calculate output shape.
	outputShape, err := shapeinference.Slice(operand.Shape, data.starts, data.limits, data.strides)
	if err != nil {
		return nil, err
	}

	node, _ := f.GetOrCreateNode(opType, outputShape, []*gobackend.Node{operand}, data)
	return node, nil
}

func execSlice(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	sliceParams, ok := node.Data.(*sliceNode)
	if !ok {
		return nil, errors.Errorf("internal error: node.data for Slice op is not *sliceNode, but %T", node.Data)
	}

	outputShape := node.Shape
	runtimeParams := sliceParams
	hasRuntimeBounds := outputShape.IsDynamic()
	if !hasRuntimeBounds {
		for axis := range sliceParams.starts {
			if sliceParams.starts[axis] < 0 || sliceParams.limits[axis] < 0 {
				hasRuntimeBounds = true
				break
			}
		}
	}
	if hasRuntimeBounds {
		// Resolve symbolic or anonymous dimensions from the concrete runtime input.
		runtimeParams = &sliceNode{
			starts: slices.Clone(sliceParams.starts), limits: slices.Clone(sliceParams.limits), strides: slices.Clone(sliceParams.strides),
		}
		for axis, dim := range operand.RawShape.Dimensions {
			if runtimeParams.starts[axis] < 0 {
				runtimeParams.starts[axis] = dim + runtimeParams.starts[axis]
			}
			if runtimeParams.limits[axis] == shapes.DynamicDim {
				runtimeParams.limits[axis] = dim
			} else if runtimeParams.limits[axis] < 0 {
				runtimeParams.limits[axis] = dim + runtimeParams.limits[axis]
			}
		}
		var err error
		outputShape, err = shapeinference.Slice(operand.RawShape, runtimeParams.starts, runtimeParams.limits, runtimeParams.strides)
		if err != nil {
			return nil, errors.Wrap(err, "resolving runtime Slice output shape")
		}
	}
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || output.RawShape.Size() == 0 {
		return output, nil
	}

	if err := execSliceBytes(backend, operand, output, runtimeParams); err != nil {
		return nil, err
	}
	return output, nil
}

//gobackend:dtypemap execSliceGeneric ints,uints,floats,half,bool
var sliceDTypeMap = gobackend.NewDTypeMap("Slice")

func execSliceGeneric[T gobackend.SupportedTypesConstraints](backend *gobackend.Backend, operand, output *gobackend.Buffer, params *sliceNode) {
	_ = execSliceBytes(backend, operand, output, params)
}

// execSliceBytes implements slice data copying across all data types using byte-level strided block copies.
// It collapses contiguous trailing dimensions with stride == 1 into block copies and merges
// contiguous outer dimensions, parallelizing across chunks for large slices.
func execSliceBytes(backend *gobackend.Backend, operand, output *gobackend.Buffer, params *sliceNode) error {
	outputBytes, err := output.MutableBytes()
	if err != nil || len(outputBytes) == 0 {
		return err
	}
	operandBytes, err := operand.MutableBytes()
	if err != nil {
		return err
	}
	elemSize := output.RawShape.DType.Size()
	rank := operand.RawShape.Rank()
	if rank == 0 {
		copy(outputBytes[:elemSize], operandBytes[:elemSize])
		return nil
	}

	operandDimensions := operand.RawShape.Dimensions
	outputDimensions := output.RawShape.Dimensions

	var stridesBuf [8]int
	var operandFlatStrides []int
	if rank <= len(stridesBuf) {
		operandFlatStrides = stridesBuf[:rank]
	} else {
		operandFlatStrides = make([]int, rank)
	}
	currentStride := 1
	for axis := rank - 1; axis >= 0; axis-- {
		operandFlatStrides[axis] = currentStride
		currentStride *= operandDimensions[axis]
	}

	// Find operand start offset in elements.
	var baseOffset int
	for axis, start := range params.starts {
		baseOffset += start * operandFlatStrides[axis]
	}

	// Collapse trailing axes that are contiguous (stride == 1) into a single block copy size.
	blockSize := 1
	outerRank := rank
	for outerRank > 0 {
		axis := outerRank - 1
		if params.strides[axis] != 1 {
			break
		}
		blockSize *= outputDimensions[axis]
		outerRank--
		if params.starts[axis] != 0 || outputDimensions[axis] != operandDimensions[axis] {
			break
		}
	}

	// Build collapsed outer dimensions and operand steps, dropping size-1 axes and
	// merging adjacent axes where step[prev] == dim[curr] * step[curr].
	var dimsBuf, stepsBuf [8]int
	var outerDims, outerStepsBytes []int
	if outerRank <= len(dimsBuf) {
		outerDims = dimsBuf[:0]
		outerStepsBytes = stepsBuf[:0]
	} else {
		outerDims = make([]int, 0, outerRank)
		outerStepsBytes = make([]int, 0, outerRank)
	}
	for axis := range outerRank {
		dim := outputDimensions[axis]
		if dim <= 1 {
			continue
		}
		stepBytes := params.strides[axis] * operandFlatStrides[axis] * elemSize
		n := len(outerDims)
		if n > 0 && outerStepsBytes[n-1] == dim*stepBytes {
			outerDims[n-1] *= dim
			outerStepsBytes[n-1] = stepBytes
		} else {
			outerDims = append(outerDims, dim)
			outerStepsBytes = append(outerStepsBytes, stepBytes)
		}
	}

	baseOffsetBytes := baseOffset * elemSize
	blockBytes := blockSize * elemSize
	numOuter := len(outerDims)
	canParallelize := len(outputBytes) >= minParallelCopyBytes &&
		backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled()

	// Fast path 0: entire output is a single contiguous block in the operand.
	if numOuter == 0 {
		src := operandBytes[baseOffsetBytes : baseOffsetBytes+blockBytes]
		if canParallelize {
			backend.Scheduler.ParallelizeChunks(blockBytes, minParallelCopyChunkBytes, func(start, end int) {
				copy(outputBytes[start:end], src[start:end])
			})
		} else {
			copy(outputBytes[:blockBytes], src)
		}
		return nil
	}

	// Fast path 1: single outer dimension of contiguous blocks (covers most 2D/3D/4D slices).
	if numOuter == 1 {
		count := outerDims[0]
		stepBytes := outerStepsBytes[0]
		if canParallelize && count < 16 && blockBytes >= minParallelCopyBytes {
			// Few large blocks: split each block into sub-chunks so all workers can participate.
			subChunksPerBlock := (blockBytes + minParallelCopyChunkBytes - 1) / minParallelCopyChunkBytes
			totalSubChunks := count * subChunksPerBlock
			backend.Scheduler.ParallelizeChunks(totalSubChunks, 1, func(startSub, endSub int) {
				for sc := startSub; sc < endSub; sc++ {
					b := sc / subChunksPerBlock
					sub := sc % subChunksPerBlock
					subStart := sub * minParallelCopyChunkBytes
					subLen := min(minParallelCopyChunkBytes, blockBytes-subStart)
					srcIdx := baseOffsetBytes + b*stepBytes + subStart
					dstIdx := b*blockBytes + subStart
					copy(outputBytes[dstIdx:dstIdx+subLen], operandBytes[srcIdx:srcIdx+subLen])
				}
			})
			return nil
		}
		runRange := func(start, end int) {
			copyStridedBytes(
				outputBytes[start*blockBytes:],
				operandBytes[baseOffsetBytes+start*stepBytes:],
				end-start, blockBytes, blockBytes, stepBytes,
			)
		}
		if canParallelize {
			minBlocks := max(1, (minParallelCopyChunkBytes+blockBytes-1)/blockBytes)
			backend.Scheduler.ParallelizeChunks(count, minBlocks, runRange)
		} else {
			runRange(0, count)
		}
		return nil
	}

	// Fast path 2: 2D outer coordinates over contiguous inner blocks of size blockBytes.
	if numOuter == 2 {
		count0 := outerDims[0]
		step0Bytes := outerStepsBytes[0]
		count1 := outerDims[1]
		step1Bytes := outerStepsBytes[1]
		dstStride0Bytes := count1 * blockBytes
		if canParallelize && count0 >= 8 {
			minOuter0 := max(1, (minParallelCopyChunkBytes+dstStride0Bytes-1)/dstStride0Bytes)
			backend.Scheduler.ParallelizeChunks(count0, minOuter0, func(start0, end0 int) {
				for i0 := start0; i0 < end0; i0++ {
					copyStridedBytes(
						outputBytes[i0*dstStride0Bytes:],
						operandBytes[baseOffsetBytes+i0*step0Bytes:],
						count1, blockBytes, blockBytes, step1Bytes,
					)
				}
			})
			return nil
		} else if !canParallelize {
			for i0 := range count0 {
				copyStridedBytes(
					outputBytes[i0*dstStride0Bytes:],
					operandBytes[baseOffsetBytes+i0*step0Bytes:],
					count1, blockBytes, blockBytes, step1Bytes,
				)
			}
			return nil
		}
	}

	// General path: N-D outer coordinates over contiguous inner blocks, peeling the innermost outer axis
	// so copyStridedBytes handles outerDims[numOuter-1] blocks at a time.
	innerCount := outerDims[numOuter-1]
	innerStepBytes := outerStepsBytes[numOuter-1]
	rowOutBytes := innerCount * blockBytes
	coordRank := numOuter - 1
	totalRows := (len(outputBytes) / blockBytes) / innerCount

	runRows := func(startRow, endRow int) {
		var coordsBuf [8]int
		var coords []int
		if coordRank <= len(coordsBuf) {
			coords = coordsBuf[:coordRank]
		} else {
			coords = make([]int, coordRank)
		}
		rem := startRow
		srcIdx := baseOffsetBytes
		for axis := coordRank - 1; axis >= 0; axis-- {
			d := outerDims[axis]
			c := rem % d
			rem /= d
			coords[axis] = c
			srcIdx += c * outerStepsBytes[axis]
		}

		dstIdx := startRow * rowOutBytes
		for r := startRow; r < endRow; r++ {
			copyStridedBytes(
				outputBytes[dstIdx:],
				operandBytes[srcIdx:],
				innerCount, blockBytes, blockBytes, innerStepBytes,
			)
			dstIdx += rowOutBytes

			for axis := coordRank - 1; axis >= 0; axis-- {
				coords[axis]++
				if coords[axis] < outerDims[axis] {
					srcIdx += outerStepsBytes[axis]
					break
				}
				srcIdx -= (outerDims[axis] - 1) * outerStepsBytes[axis]
				coords[axis] = 0
			}
		}
	}
	if canParallelize {
		minRows := max(1, (minParallelCopyChunkBytes+rowOutBytes-1)/rowOutBytes)
		backend.Scheduler.ParallelizeChunks(totalRows, minRows, runRows)
	} else {
		runRows(0, totalRows)
	}
	return nil
}

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

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || output.RawShape.Size() == 0 {
		return output, nil
	}

	// Dispatch to the generic implementation based on DType.
	tmpAny, tmpErr := sliceDTypeMap.Get(node.Shape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	fn := tmpAny.(func(backend *gobackend.Backend, operand, output *gobackend.Buffer, params *sliceNode)) //nolint:errcheck
	fn(backend, operand, output, sliceParams)
	return output, nil
}

//gobackend:dtypemap execSliceGeneric ints,uints,floats,half,bool
var sliceDTypeMap = gobackend.NewDTypeMap("Slice")

const (
	minParallelSliceElements      = 64 * 1024
	minParallelSliceChunkElements = 16 * 1024
)

// execSliceGeneric implements the actual slice data copying. It is called via sliceDTypeMap.Dispatch.
// It collapses contiguous trailing dimensions with stride == 1 into block copies and merges
// contiguous outer dimensions, parallelizing across chunks for large slices.
func execSliceGeneric[T gobackend.SupportedTypesConstraints](backend *gobackend.Backend, operand, output *gobackend.Buffer, params *sliceNode) {
	outputFlat := output.Flat.([]T)
	if len(outputFlat) == 0 {
		return
	}
	operandFlat := operand.Flat.([]T)
	rank := operand.RawShape.Rank()
	if rank == 0 {
		outputFlat[0] = operandFlat[0]
		return
	}

	operandDimensions := operand.RawShape.Dimensions
	outputDimensions := output.RawShape.Dimensions
	operandFlatStrides := operand.RawShape.Strides()

	// Find operandFlatIdx start value.
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
	var outerDims, outerSteps []int
	if outerRank <= len(dimsBuf) {
		outerDims = dimsBuf[:0]
		outerSteps = stepsBuf[:0]
	} else {
		outerDims = make([]int, 0, outerRank)
		outerSteps = make([]int, 0, outerRank)
	}
	for axis := range outerRank {
		dim := outputDimensions[axis]
		if dim <= 1 {
			continue
		}
		step := params.strides[axis] * operandFlatStrides[axis]
		n := len(outerDims)
		if n > 0 && outerSteps[n-1] == dim*step {
			outerDims[n-1] *= dim
			outerSteps[n-1] = step
		} else {
			outerDims = append(outerDims, dim)
			outerSteps = append(outerSteps, step)
		}
	}

	numOuter := len(outerDims)
	canParallelize := len(outputFlat) >= minParallelSliceElements &&
		backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled()

	// Fast path 0: entire output is a single contiguous block in the operand.
	if numOuter == 0 {
		src := operandFlat[baseOffset : baseOffset+blockSize]
		if canParallelize {
			backend.Scheduler.ParallelizeChunks(blockSize, minParallelSliceChunkElements, func(start, end int) {
				copy(outputFlat[start:end], src[start:end])
			})
		} else {
			copy(outputFlat, src)
		}
		return
	}

	// Fast path 1: single outer dimension of contiguous blocks (covers most 2D/3D/4D slices).
	if numOuter == 1 {
		count := outerDims[0]
		step := outerSteps[0]
		runRange := func(start, end int) {
			srcIdx := baseOffset + start*step
			if blockSize == 1 {
				for b := start; b < end; b++ {
					outputFlat[b] = operandFlat[srcIdx]
					srcIdx += step
				}
			} else {
				dstIdx := start * blockSize
				for b := start; b < end; b++ {
					copy(outputFlat[dstIdx:dstIdx+blockSize], operandFlat[srcIdx:srcIdx+blockSize])
					dstIdx += blockSize
					srcIdx += step
				}
			}
		}
		if canParallelize {
			minBlocks := max(1, (minParallelSliceChunkElements+blockSize-1)/blockSize)
			backend.Scheduler.ParallelizeChunks(count, minBlocks, runRange)
		} else {
			runRange(0, count)
		}
		return
	}

	// General path: N-D outer coordinates over contiguous inner blocks of size blockSize.
	totalBlocks := len(outputFlat) / blockSize
	runBlocks := func(startBlock, endBlock int) {
		var coordsBuf [8]int
		var coords []int
		if numOuter <= len(coordsBuf) {
			coords = coordsBuf[:numOuter]
		} else {
			coords = make([]int, numOuter)
		}
		rem := startBlock
		srcIdx := baseOffset
		for axis := numOuter - 1; axis >= 0; axis-- {
			d := outerDims[axis]
			c := rem % d
			rem /= d
			coords[axis] = c
			srcIdx += c * outerSteps[axis]
		}

		dstIdx := startBlock * blockSize
		for b := startBlock; b < endBlock; b++ {
			if blockSize == 1 {
				outputFlat[dstIdx] = operandFlat[srcIdx]
			} else {
				copy(outputFlat[dstIdx:dstIdx+blockSize], operandFlat[srcIdx:srcIdx+blockSize])
			}
			dstIdx += blockSize

			for axis := numOuter - 1; axis >= 0; axis-- {
				coords[axis]++
				if coords[axis] < outerDims[axis] {
					srcIdx += outerSteps[axis]
					break
				}
				srcIdx -= (outerDims[axis] - 1) * outerSteps[axis]
				coords[axis] = 0
			}
		}
	}
	if canParallelize {
		minBlocks := max(1, (minParallelSliceChunkElements+blockSize-1)/blockSize)
		backend.Scheduler.ParallelizeChunks(totalBlocks, minBlocks, runBlocks)
	} else {
		runBlocks(0, totalBlocks)
	}
}

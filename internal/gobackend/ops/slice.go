package ops

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
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
	data := &sliceNode{
		starts,
		make([]int, len(limits)),
		strides,
	}
	// Trim limits to operand shape.
	for axis, limit := range limits {
		if limit < 0 {
			limit = operand.Shape.Dimensions[axis] + limit
		}
		limit = min(limit, operand.Shape.Dimensions[axis])
		limit = max(limit, 0)
		data.limits[axis] = limit
	}

	// Calculate output shape.
	outputShape, err := shapeinference.SliceOp(operand.Shape, data.starts, data.limits, data.strides)
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
		// Assuming node.data holds the necessary slice parameters.
		// If Builder.Slice stores data differently, this needs adjustment.
		return nil, errors.Errorf("internal error: node.data for Slice op is not *sliceNode, but %T", node.Data)
	}

	output, err := backend.GetBuffer(node.Shape.DType, node.Shape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = node.Shape
	if output.RawShape.Size() == 0 {
		// Simplest case, where the slice is of 0 elements, just return the empty buffer with the correct shape.
		return output, nil
	}

	// Dispatch to the generic implementation based on DType.
	// Note: limits are not used in the generic exec function but passed for potential future use or consistency.
	tmpAny, tmpErr := sliceDTypeMap.Get(node.Shape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	fn := tmpAny.(func(operand, output *gobackend.Buffer, params *sliceNode)) //nolint:errcheck
	fn(operand, output, sliceParams)
	return output, nil
}

//gobackend:dtypemap execSliceGeneric ints,uints,floats,half,bool
var sliceDTypeMap = gobackend.NewDTypeMap("Slice")

// execSliceGeneric implements the actual slice data copying. It is called via sliceDTypeMap.Dispatch.
// It iterates through the output buffer coordinates, calculates the corresponding coordinate
// in the operand buffer using starts and strides, and copies the value.
func execSliceGeneric[T gobackend.SupportedTypesConstraints](operand, output *gobackend.Buffer, params *sliceNode) {
	rank := operand.RawShape.Rank()
	outputFlat := output.Flat.([]T)
	operandFlat := operand.Flat.([]T)

	// Find operandFlatIdx start value.
	var operandFlatIdx int
	operandFlatStrides := operand.RawShape.Strides()
	for axis, idx := range params.starts {
		operandFlatIdx += operandFlatStrides[axis] * idx
	}

	operandPerAxisIdx := make([]int, rank)
	operandDimensions := operand.RawShape.Dimensions

	for outputFlatIdx := range outputFlat {
		// Copy value at current position.
		outputFlat[outputFlatIdx] = operandFlat[operandFlatIdx]

		// Iterate to the next operand position.
		for axis := rank - 1; axis >= 0; axis-- {
			if operandDimensions[axis] == 1 {
				// We don't iterate on this axis.
				continue
			}

			// Increment the current axis.
			operandPerAxisIdx[axis]++
			operandFlatIdx += operandFlatStrides[axis]
			if operandPerAxisIdx[axis] < params.limits[axis] {
				// Done for this iteration, don't need to go to the next axis yet.
				break
			}
			// Rewind the current axis to the start: we will increment the next axis for this iteration.
			operandFlatIdx -= operandDimensions[axis] * operandFlatStrides[axis]
			operandPerAxisIdx[axis] = params.starts[axis]
			operandFlatIdx += operandFlatStrides[axis] * params.strides[axis]
		}
	}
}

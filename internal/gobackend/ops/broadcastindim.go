package ops // BroadcastInDimsOp ====================================================================================================

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
)

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeBroadcastInDim, gobackend.PriorityGeneric, execBroadcastInDim)

	// DTypeMap: broadcastInDimDTypeMap
	broadcastInDimDTypeMap.Register(dtypes.Int8, gobackend.PriorityGeneric, execBroadcastInDimGeneric[int8])
	broadcastInDimDTypeMap.Register(dtypes.Int16, gobackend.PriorityGeneric, execBroadcastInDimGeneric[int16])
	broadcastInDimDTypeMap.Register(dtypes.Int32, gobackend.PriorityGeneric, execBroadcastInDimGeneric[int32])
	broadcastInDimDTypeMap.Register(dtypes.Int64, gobackend.PriorityGeneric, execBroadcastInDimGeneric[int64])
	broadcastInDimDTypeMap.Register(dtypes.Uint8, gobackend.PriorityGeneric, execBroadcastInDimGeneric[uint8])
	broadcastInDimDTypeMap.Register(dtypes.Uint16, gobackend.PriorityGeneric, execBroadcastInDimGeneric[uint16])
	broadcastInDimDTypeMap.Register(dtypes.Uint32, gobackend.PriorityGeneric, execBroadcastInDimGeneric[uint32])
	broadcastInDimDTypeMap.Register(dtypes.Uint64, gobackend.PriorityGeneric, execBroadcastInDimGeneric[uint64])
	broadcastInDimDTypeMap.Register(dtypes.Float32, gobackend.PriorityGeneric, execBroadcastInDimGeneric[float32])
	broadcastInDimDTypeMap.Register(dtypes.Float64, gobackend.PriorityGeneric, execBroadcastInDimGeneric[float64])
	broadcastInDimDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityGeneric, execBroadcastInDimGeneric[bfloat16.BFloat16])
	broadcastInDimDTypeMap.Register(dtypes.Float16, gobackend.PriorityGeneric, execBroadcastInDimGeneric[float16.Float16])
	broadcastInDimDTypeMap.Register(dtypes.Bool, gobackend.PriorityGeneric, execBroadcastInDimGeneric[bool])
}

func execBroadcastInDim(
	backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (
	*gobackend.Buffer, error) {
	_ = inputsOwned // We don't reuse the inputs.
	operand := inputs[0]
	output, err := backend.GetBuffer(node.Shape.DType, node.Shape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = node.Shape

	var iter *gobackend.BroadcastIterator

	if operand.RawShape.Size() == 1 {
		// Special case 1: just leave iter as nil.
	} else {
		// Reshape operand shape: same dimension as the operand on the corresponding axes, 1 elsewhere.
		// We are only changing the rank, but it stays the same size; hence the flat data doesn't change.
		// Notice: broadcastAxes is strictly increasing (no transpositions are happening).
		dims := xslices.SliceWithValue(output.RawShape.Rank(), 1)
		broadcastAxes := node.Data.([]int)
		for operandAxis, outputAxis := range broadcastAxes {
			dims[outputAxis] = operand.RawShape.Dimensions[operandAxis]
		}
		reshapedOperand := shapes.Make(operand.RawShape.DType, dims...)

		// Create broadcasting the iterator: it requires operand and output shapes to have the same rank.
		iter = gobackend.NewBroadcastIterator(reshapedOperand, output.RawShape)
	}

	// Call implementation for corresponding dtype.
	fnAny, err := broadcastInDimDTypeMap.Get(node.Shape.DType)
	if err != nil {
		return nil, err
	}
	fnAny.(func(*gobackend.Buffer, *gobackend.Buffer, *gobackend.BroadcastIterator))(operand, output, iter)
	return output, nil
}

var broadcastInDimDTypeMap = gobackend.NewDTypeMap("BroadcastInDim")

func execBroadcastInDimGeneric[T gobackend.SupportedTypesConstraints](
	operand, output *gobackend.Buffer, iter *gobackend.BroadcastIterator) {
	operandFlat, outputFlat := operand.Flat.([]T), output.Flat.([]T)
	if iter == nil {
		// Special cases:
		if len(operandFlat) == 1 {
			// 1. Where operand is a scalar (or size 1) that is broadcast everywhere.
			xslices.FillSlice(outputFlat, operandFlat[0])
		} else {
			// 2. Where we are simply broadcasting a prefix dimensions:
			repeats := len(outputFlat) / len(operandFlat)
			pos := 0
			for range repeats {
				copy(outputFlat[pos:], operandFlat)
				pos += len(operandFlat)
			}
		}
		return
	}

	// Arbitrary broadcasting using the flexible but slower broadcast iterator:
	for operandFlatIdx, outputFlatIdx := range iter.IterFlatIndices() {
		outputFlat[outputFlatIdx] = operandFlat[operandFlatIdx]
	}
}

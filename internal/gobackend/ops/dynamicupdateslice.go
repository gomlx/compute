// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterDynamicUpdateSlice.Register(DynamicUpdateSlice, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeDynamicUpdateSlice, gobackend.PriorityGeneric, execDynamicUpdateSlice)
}

// DynamicUpdateSlice implements compute.Function.
func DynamicUpdateSlice(f *gobackend.Function, operandOp, updateOp compute.Value, startIndicesOps []compute.Value) (compute.Value, error) {
	allOps := make([]compute.Value, 0, 2+len(startIndicesOps))
	allOps = append(allOps, operandOp, updateOp)
	allOps = append(allOps, startIndicesOps...)
	inputs, err := f.VerifyAndCastValues("DynamicUpdateSlice", allOps...)
	if err != nil {
		return nil, err
	}
	operand, update := inputs[0], inputs[1]
	startIndices := inputs[2:]
	rank := operand.Shape.Rank()
	if len(startIndices) == 1 && !startIndices[0].Shape.IsScalar() && startIndices[0].Shape.Rank() == 1 {
		if startIndices[0].Shape.Dimensions[0] != rank {
			return nil, errors.Errorf("DynamicUpdateSlice: 1D startIndices has length %d, but operand rank is %d",
				startIndices[0].Shape.Dimensions[0], rank)
		}
	} else if len(startIndices) != rank {
		return nil, errors.Errorf("DynamicUpdateSlice: len(startIndices) (%d) must match operand rank (%d)",
			len(startIndices), rank)
	}
	outputShape, err := shapeinference.DynamicUpdateSlice(operand.Shape, update.Shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeDynamicUpdateSlice, outputShape, inputs, nil)
	return node, nil
}

func scalarToIntAt(buf *gobackend.Buffer, idx int) int {
	switch buf.RawShape.DType {
	case dtypes.Int8:
		return int(buf.Flat.([]int8)[idx])
	case dtypes.Int16:
		return int(buf.Flat.([]int16)[idx])
	case dtypes.Int32:
		return int(buf.Flat.([]int32)[idx])
	case dtypes.Int64:
		return int(buf.Flat.([]int64)[idx])
	case dtypes.Uint8:
		return int(buf.Flat.([]uint8)[idx])
	case dtypes.Uint16:
		return int(buf.Flat.([]uint16)[idx])
	case dtypes.Uint32:
		return int(buf.Flat.([]uint32)[idx])
	case dtypes.Uint64:
		return int(buf.Flat.([]uint64)[idx])
	case dtypes.Float32:
		return int(buf.Flat.([]float32)[idx])
	case dtypes.Float64:
		return int(buf.Flat.([]float64)[idx])
	default:
		panic(errors.Errorf("unsupported dtype %s for index", buf.RawShape.DType))
	}
}

func copyUpdateSlice[T any](outputSlice, updateSlice []T, starts, updateDims, outStrides []int) {
	rank := len(updateDims)
	if rank == 0 {
		outputSlice[0] = updateSlice[0]
		return
	}
	lastDim := updateDims[rank-1]
	if lastDim == 0 {
		return
	}
	if rank == 1 {
		copy(outputSlice[starts[0]:starts[0]+lastDim], updateSlice[:lastDim])
		return
	}
	numSlices := len(updateSlice) / lastDim
	coord := make([]int, rank-1)
	updateIdx := 0
	for sliceIdx := 0; sliceIdx < numSlices; sliceIdx++ {
		outIdx := starts[rank-1]
		for a := 0; a < rank-1; a++ {
			outIdx += (starts[a] + coord[a]) * outStrides[a]
		}
		copy(outputSlice[outIdx:outIdx+lastDim], updateSlice[updateIdx:updateIdx+lastDim])
		updateIdx += lastDim

		for a := rank - 2; a >= 0; a-- {
			coord[a]++
			if coord[a] < updateDims[a] {
				break
			}
			coord[a] = 0
		}
	}
}

func execDynamicUpdateSlice(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	update := inputs[1]
	startIndices := inputs[2:]

	var output *gobackend.Buffer
	var err error
	if inputsOwned[0] {
		output = operand
		inputs[0] = nil
	} else {
		output, err = backend.CloneBuffer(operand)
		if err != nil {
			return nil, err
		}
	}

	if backend.NoOps || update.RawShape.Size() == 0 {
		return output, nil
	}

	rank := operand.RawShape.Rank()
	starts := make([]int, rank)
	if len(startIndices) == 1 && !startIndices[0].RawShape.IsScalar() && startIndices[0].RawShape.Rank() == 1 {
		for i := 0; i < rank; i++ {
			val := scalarToIntAt(startIndices[0], i)
			maxStart := operand.RawShape.Dimensions[i] - update.RawShape.Dimensions[i]
			if maxStart < 0 {
				maxStart = 0
			}
			starts[i] = min(max(val, 0), maxStart)
		}
	} else {
		for i := 0; i < rank; i++ {
			val := scalarToIntAt(startIndices[i], 0)
			maxStart := operand.RawShape.Dimensions[i] - update.RawShape.Dimensions[i]
			if maxStart < 0 {
				maxStart = 0
			}
			starts[i] = min(max(val, 0), maxStart)
		}
	}

	updateDims := update.RawShape.Dimensions
	outStrides := output.RawShape.Strides()

	switch output.RawShape.DType {
	case dtypes.Float32:
		copyUpdateSlice(output.Flat.([]float32), update.Flat.([]float32), starts, updateDims, outStrides)
	case dtypes.Float64:
		copyUpdateSlice(output.Flat.([]float64), update.Flat.([]float64), starts, updateDims, outStrides)
	case dtypes.BFloat16:
		copyUpdateSlice(output.Flat.([]bfloat16.BFloat16), update.Flat.([]bfloat16.BFloat16), starts, updateDims, outStrides)
	case dtypes.Float16:
		copyUpdateSlice(output.Flat.([]float16.Float16), update.Flat.([]float16.Float16), starts, updateDims, outStrides)
	case dtypes.Int32:
		copyUpdateSlice(output.Flat.([]int32), update.Flat.([]int32), starts, updateDims, outStrides)
	case dtypes.Int64:
		copyUpdateSlice(output.Flat.([]int64), update.Flat.([]int64), starts, updateDims, outStrides)
	case dtypes.Int16:
		copyUpdateSlice(output.Flat.([]int16), update.Flat.([]int16), starts, updateDims, outStrides)
	case dtypes.Int8:
		copyUpdateSlice(output.Flat.([]int8), update.Flat.([]int8), starts, updateDims, outStrides)
	case dtypes.Uint32:
		copyUpdateSlice(output.Flat.([]uint32), update.Flat.([]uint32), starts, updateDims, outStrides)
	case dtypes.Uint64:
		copyUpdateSlice(output.Flat.([]uint64), update.Flat.([]uint64), starts, updateDims, outStrides)
	case dtypes.Uint16:
		copyUpdateSlice(output.Flat.([]uint16), update.Flat.([]uint16), starts, updateDims, outStrides)
	case dtypes.Uint8:
		copyUpdateSlice(output.Flat.([]uint8), update.Flat.([]uint8), starts, updateDims, outStrides)
	case dtypes.Bool:
		copyUpdateSlice(output.Flat.([]bool), update.Flat.([]bool), starts, updateDims, outStrides)
	default:
		return nil, errors.Errorf("DynamicUpdateSlice: unsupported dtype %s", output.RawShape.DType)
	}

	return output, nil
}

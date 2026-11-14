// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/exceptions"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterIsNaN.Register(IsNaN, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeIsNaN, gobackend.PriorityGeneric, execIsNaN)
}

// IsNaN implements the compute.Builder interface.
func IsNaN(f *gobackend.Function, operandOp compute.Value) (compute.Value, error) {
	opType := compute.OpTypeIsNaN
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	if !operand.Shape.DType.IsFloat() {
		return nil, errors.Errorf("the operation IsNaN is only defined for float types (%s), cannot use it", operand.Shape.DType)
	}
	outShape := operand.Shape.Clone()
	outShape.DType = dtypes.Bool
	node, _ := f.GetOrCreateNode(opType, outShape, []*gobackend.Node{operand}, nil)
	return node, nil
}

func execIsNaN(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input := inputs[0]
	outShape := input.RawShape.Clone()
	outShape.DType = dtypes.Bool
	output, err := backend.GetBuffer(outShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || outShape.Size() == 0 {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		runUnarySlices(backend, input.Flat.([]float32), output.Flat.([]bool), execIsNaNGeneric[float32])
	case dtypes.Float64:
		runUnarySlices(backend, input.Flat.([]float64), output.Flat.([]bool), execIsNaNGeneric[float64])
	case dtypes.BFloat16:
		runUnarySlices(backend, input.Flat.([]bfloat16.BFloat16), output.Flat.([]bool), execIsNaNHalfPrecision[bfloat16.BFloat16])
	case dtypes.Float16:
		runUnarySlices(backend, input.Flat.([]float16.Float16), output.Flat.([]bool), execIsNaNHalfPrecision[float16.Float16])
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execIsNaNGeneric[T float32 | float64](inputs []T, outputs []bool) {
	for ii, input := range inputs {
		outputs[ii] = input != input
	}
}

func execIsNaNHalfPrecision[T gotype.HalfPrecision[T]](inputs []T, outputs []bool) {
	for ii, input := range inputs {
		f := input.Float32()
		outputs[ii] = f != f
	}
}

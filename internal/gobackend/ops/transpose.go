// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
)

func init() {
	gobackend.RegisterTranspose.Register(Transpose, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeTranspose, gobackend.PriorityGeneric, execTranspose)

	// DTypeMap: TransposeDTypeMap
	gobackend.TransposeDTypeMap.Register(dtypes.Int8, gobackend.PriorityGeneric, execTransposeGeneric[int8])
	gobackend.TransposeDTypeMap.Register(dtypes.Int16, gobackend.PriorityGeneric, execTransposeGeneric[int16])
	gobackend.TransposeDTypeMap.Register(dtypes.Int32, gobackend.PriorityGeneric, execTransposeGeneric[int32])
	gobackend.TransposeDTypeMap.Register(dtypes.Int64, gobackend.PriorityGeneric, execTransposeGeneric[int64])
	gobackend.TransposeDTypeMap.Register(dtypes.Uint8, gobackend.PriorityGeneric, execTransposeGeneric[uint8])
	gobackend.TransposeDTypeMap.Register(dtypes.Uint16, gobackend.PriorityGeneric, execTransposeGeneric[uint16])
	gobackend.TransposeDTypeMap.Register(dtypes.Uint32, gobackend.PriorityGeneric, execTransposeGeneric[uint32])
	gobackend.TransposeDTypeMap.Register(dtypes.Uint64, gobackend.PriorityGeneric, execTransposeGeneric[uint64])
	gobackend.TransposeDTypeMap.Register(dtypes.Float32, gobackend.PriorityGeneric, execTransposeGeneric[float32])
	gobackend.TransposeDTypeMap.Register(dtypes.Float64, gobackend.PriorityGeneric, execTransposeGeneric[float64])
	gobackend.TransposeDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityGeneric, execTransposeGeneric[bfloat16.BFloat16])
	gobackend.TransposeDTypeMap.Register(dtypes.Float16, gobackend.PriorityGeneric, execTransposeGeneric[float16.Float16])
	gobackend.TransposeDTypeMap.Register(dtypes.Bool, gobackend.PriorityGeneric, execTransposeGeneric[bool])
}

// Transpose axes of x.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func Transpose(f *gobackend.Function, operand *gobackend.Node, permutations ...int) (*gobackend.Node, error) {
	opType := compute.OpTypeTranspose
	outputShape, err := shapeinference.TransposeOp(operand.Shape, permutations)
	if err != nil {
		// This should have been validated during graph build, but we check again just in case.
		return nil, err
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*gobackend.Node{operand}, permutations)
	return node, nil
}

// execTranspose implements Transpose.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func execTranspose(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	permutations := node.Data.([]int)
	_ = inputsOwned // We don't reuse the inputs.

	// We can't write to the same buffer we read from because it's not done with swaps.
	output, err := backend.GetBuffer(operand.RawShape.DType, operand.RawShape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = node.Shape
	it := gobackend.NewTransposeIterator(operand.RawShape, permutations)
	dtype := node.Shape.DType
	tmpAny, tmpErr := gobackend.TransposeDTypeMap.Get(dtype)
	if tmpErr != nil {
		panic(tmpErr)
	}
	transposeFn := tmpAny.(func(operand, output *gobackend.Buffer, it *gobackend.TransposeIterator))
	transposeFn(operand, output, it)
	return output, nil
}

func execTransposeGeneric[T gobackend.SupportedTypesConstraints](operand, output *gobackend.Buffer, it *gobackend.TransposeIterator) {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	for _, value := range operandFlat {
		outputFlat[it.Next()] = value
	}
}

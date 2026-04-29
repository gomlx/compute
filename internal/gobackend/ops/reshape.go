// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
)

func init() {
	gobackend.RegisterReshape.Register(Reshape, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeReshape, gobackend.PriorityGeneric, execReshape)
}

// Reshape reshapes the operand to the given dimensions.
func Reshape(f *gobackend.Function, operand *gobackend.Node, dims ...int) (*gobackend.Node, error) {
	opType := compute.OpTypeReshape
	outputShape, err := shapeinference.ReshapeOp(operand.Shape, dims)
	if err != nil {
		return nil, err
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*gobackend.Node{operand}, nil)
	return node, nil
}

// execReshape implements Reshape.
func execReshape(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	var output *gobackend.Buffer
	var err error
	if inputsOwned[0] {
		output = operand
		inputs[0] = nil
	} else {
		output, err = backend.GetBuffer(operand.RawShape.DType, operand.RawShape.Size())
		if err != nil {
			return nil, err
		}
		gobackend.CopyFlat(output.Flat, operand.Flat)
	}
	output.RawShape = node.Shape
	return output, nil
}

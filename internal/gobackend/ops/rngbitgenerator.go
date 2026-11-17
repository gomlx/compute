// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterRNGBitGenerator.Register(RNGBitGenerator, gobackend.PriorityGeneric)
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeRNGBitGenerator] = execRNGBitGenerator
}

type rngNodeData struct {
	algorithm        compute.RNGAlgorithm
	axisInputIndices []int
}

// RNGBitGenerator generates a tensor of the given shape filled with uniformly distributed
// random bits using the specified Pseudo-Random Number Generator (PRNG) algorithm.
//
// It returns the updated state tensor and the generated random bit values.
func RNGBitGenerator(
	f *gobackend.Function,
	stateOp compute.Value,
	shape shapes.Shape,
	algorithm compute.RNGAlgorithm,
) (newState, values compute.Value, err error) {
	opType := compute.OpTypeRNGBitGenerator
	inputs, err := f.VerifyAndCastValues(opType.String(), stateOp)
	if err != nil {
		return nil, nil, err
	}
	state := inputs[0]

	caps := f.RawBuilder.Backend.Capabilities()
	expectedStateShape, err := caps.RNGStateShape(algorithm)
	if err != nil {
		return nil, nil, err
	}
	newStateShape, valuesShape, err := shapeinference.RNGBitGenerator(expectedStateShape, state.Shape, shape)
	if err != nil {
		return nil, nil, err
	}

	allInputs := []*gobackend.Node{state}
	var axisInputIndices []int
	if shape.IsDynamic() {
		axisInputIndices = make([]int, shape.Rank())
		var dynValues []compute.Value
		knownAxes := f.KnownDynamicAxisNames()
		for i, dim := range shape.Dimensions {
			axisInputIndices[i] = -1
			if dim != shapes.DynamicDim {
				continue
			}
			axisName := shape.AxisName(i)
			if axisName == "" {
				return nil, nil, errors.Errorf("RNGBitGenerator: dynamic axis %d of shape %s must have an axis name", i, shape)
			}
			dimVal, ok, err := f.DynamicAxisValue(axisName)
			if err != nil {
				return nil, nil, err
			}
			if ok {
				axisInputIndices[i] = 1 + len(dynValues)
				dynValues = append(dynValues, dimVal)
			} else if !knownAxes.Has(axisName) {
				return nil, nil, errors.Errorf("RNGBitGenerator: unknown dynamic axis %q at axis %d of shape %s", axisName, i, shape)
			}
		}
		if len(dynValues) > 0 {
			dynNodes, err := f.VerifyAndCastValues(opType.String(), dynValues...)
			if err != nil {
				return nil, nil, err
			}
			allInputs = append(allInputs, dynNodes...)
		}
	}

	resolved := caps.ResolveRNGAlgorithm(algorithm)
	outputShapes := []shapes.Shape{
		newStateShape,
		valuesShape,
	}
	node := f.NewMultiOutputsNode(opType, outputShapes, allInputs...)
	node.Data = &rngNodeData{
		algorithm:        resolved,
		axisInputIndices: axisInputIndices,
	}
	newState = node.MultiOutputsNodes[0]
	values = node.MultiOutputsNodes[1]
	return newState, values, nil
}

// execRNGBitGenerator is the executor function registered for compute.OpTypeRNGBitGenerator.
func execRNGBitGenerator(
	backend *gobackend.Backend,
	node *gobackend.Node,
	inputs []*gobackend.Buffer,
	inputsOwned []bool,
) ([]*gobackend.Buffer, error) {
	data, ok := node.Data.(*rngNodeData)
	if !ok || data == nil {
		return nil, errors.Errorf("execRNGBitGenerator: missing rngNodeData on node")
	}

	state := inputs[0]
	stateFlat := state.Flat.([]uint64)

	targetShape := node.MultiOutputsShapes[1]
	if targetShape.IsDynamic() {
		axisInputIndices := data.axisInputIndices
		concreteDims := make([]int, targetShape.Rank())
		for i, dim := range targetShape.Dimensions {
			if dim != shapes.DynamicDim {
				concreteDims[i] = dim
				continue
			}
			if i >= len(axisInputIndices) || axisInputIndices[i] < 0 || axisInputIndices[i] >= len(inputs) {
				return nil, errors.Errorf("execRNGBitGenerator: unresolved dynamic axis %d (%q) in shape %s", i, targetShape.AxisName(i), targetShape)
			}
			valSize, err := readScalarInt(inputs[axisInputIndices[i]])
			if err != nil {
				return nil, errors.Wrapf(err, "execRNGBitGenerator: reading dynamic dimension for axis %d (%q)", i, targetShape.AxisName(i))
			}
			if valSize < 0 {
				return nil, errors.Errorf("execRNGBitGenerator: dynamic dimension for axis %d (%q) must be nonnegative, got %d", i, targetShape.AxisName(i), valSize)
			}
			concreteDims[i] = valSize
		}
		targetShape = shapes.Make(targetShape.DType, concreteDims...)
	}

	rngData, err := backend.GetBuffer(targetShape)
	if err != nil {
		return nil, err
	}

	var outStateBuf *gobackend.Buffer
	if inputsOwned[0] {
		outStateBuf = state
		inputs[0] = nil
	} else {
		outStateBuf, err = backend.GetBuffer(node.MultiOutputsShapes[0])
		if err != nil {
			return nil, err
		}
		copy(outStateBuf.Flat.([]uint64), stateFlat)
	}
	outStateFlat := outStateBuf.Flat.([]uint64)

	if targetShape.Size() == 0 {
		return []*gobackend.Buffer{outStateBuf, rngData}, nil
	}

	if err := runRNGKernel(backend, data.algorithm, targetShape, stateFlat, outStateFlat, rngData); err != nil {
		return nil, err
	}
	return []*gobackend.Buffer{outStateBuf, rngData}, nil
}

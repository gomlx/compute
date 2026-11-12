// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// resolveRuntimeNode specializes a node from the buffers available when it is
// scheduled. Unlike parameter specialization, this handles dimensions produced
// by data-dependent operations. All copies are private to this invocation.
func resolveRuntimeNode(backend *Backend, node *Node, inputs []*Buffer, spec *ShapeSpecialization) (*Node, error) {
	needs := node.Shape.IsDynamic()
	for _, outputShape := range node.MultiOutputsShapes {
		needs = needs || outputShape.IsDynamic()
	}
	for _, input := range node.Inputs {
		template := input
		if spec != nil {
			template = spec.resolvedNodes[input.Index]
		}
		needs = needs || template.Shape.IsDynamic()
	}
	if !needs {
		return node, nil
	}

	bindings := make(shapes.AxisBindings)
	maxIndex := node.Index
	for i, input := range node.Inputs {
		if inputs[i] == nil {
			continue
		} // Cached constant input.
		if input.Shape.IsDynamic() {
			if err := bindings.Extract(input.Shape, inputs[i].RawShape); err != nil {
				return nil, err
			}
		}
		maxIndex = max(maxIndex, input.Index)
	}
	shape, err := node.Shape.ResolvePartial(bindings)
	if err != nil {
		return nil, err
	}
	var multiOutputsShapes []shapes.Shape
	if node.MultiOutputsShapes != nil {
		multiOutputsShapes = make([]shapes.Shape, len(node.MultiOutputsShapes))
		for i, outputShape := range node.MultiOutputsShapes {
			multiOutputsShapes[i], err = outputShape.ResolvePartial(bindings)
			if err != nil {
				return nil, errors.Wrapf(err, "resolving runtime shape for output %d of %s", i, node.OpType)
			}
		}
	}
	resolved := &Node{
		Index: node.Index, Inputs: node.Inputs, CapturedInputs: node.CapturedInputs,
		OpType: node.OpType, Shape: shape, Builder: node.Builder, Function: node.Function,
		Data: node.Data, IsNodeSelectOutput: node.IsNodeSelectOutput, SelectOutputIdx: node.SelectOutputIdx,
		MultiOutputsShapes: multiOutputsShapes, MultiOutputsNodes: node.MultiOutputsNodes,
	}
	resolved.cachedExecutorIdx.Store(node.cachedExecutorIdx.Load())
	if recomputable, ok := node.Data.(RecomputableNodeData); ok {
		nodes := make([]*Node, maxIndex+1)
		nodes[node.Index] = resolved
		for i, input := range node.Inputs {
			if inputs[i] != nil {
				nodes[input.Index] = &Node{Index: input.Index, Shape: inputs[i].RawShape}
			}
		}
		resolved.Data, err = recomputable.Recompute(backend, nodes, resolved)
		if err != nil {
			return nil, errors.Wrapf(err, "resolving runtime data for %s", node.OpType)
		}
	}
	return resolved, nil
}

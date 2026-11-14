// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// ShapeSpecialization holds a resolved-shape copy of the graph nodes
// for a specific set of parameter axis bindings.
//
// Dynamic shape specialization happens in two stages:
//  1. Function-level (specializeFunction): resolves named dynamic dimensions bound
//     by function parameters once at Execute entry, caching the result per AxisBindings.
//  2. Node-level (specializeNode): specializes individual nodes just before op dispatch
//     when dimensions could not be resolved from parameter bindings alone — such as
//     outputs of data-dependent operations (NonZero, DynamicReshape, DynamicIota,
//     DynamicBroadcastInDim) or unspecialized intermediate dynamic axes (e.g. strided
//     ConvGeneral or DynamicPad outputs with generated "#..." axis names).
//
// The resolvedNodes are shallow copies of the original Function.Nodes.
// Only the Shape field (and MultiOutputsShapes / RecomputableNodeData)
// is resolved; all other fields (Inputs, CapturedInputs, OpType, etc.)
// are shared with the originals. This allows executor functions to read
// node.Shape and get concrete values without any signature changes.
type ShapeSpecialization struct {
	// bindings that produced this specialization.
	bindings shapes.AxisBindings

	// resolvedNodes is a shallow copy of the original Function.Nodes slice.
	// Each node has its Shape field resolved as far as the bindings allow.
	// All other fields (Inputs, Data, OpType, etc.) point to the originals.
	resolvedNodes []*Node
}

// hasDynamicParameters returns true if any parameter node has dynamic dimensions.
func hasDynamicParameters(params []*Node) bool {
	for _, p := range params {
		if p.Shape.IsDynamic() {
			return true
		}
	}
	return false
}

// extractBindingsFromInputs extracts axis bindings by matching symbolic parameter
// shapes against concrete input buffer shapes. Returns merged bindings across all
// parameters, or an error if shapes are incompatible or bindings conflict.
func extractBindingsFromInputs(params []*Node, inputs []*Buffer) (shapes.AxisBindings, error) {
	bindings := make(shapes.AxisBindings)
	for i, param := range params {
		if !param.Shape.IsDynamic() {
			continue
		}
		err := bindings.Extract(param.Shape, inputs[i].RawShape)
		if err != nil {
			paramName := ""
			if pd, ok := param.Data.(*NodeParameter); ok {
				paramName = pd.Name
			}
			return nil, errors.WithMessagef(err, "parameter %d %q", i, paramName)
		}
	}
	return bindings, nil
}

// createSpecialization builds a ShapeSpecialization for the given bindings.
// It creates shallow copies of all nodes with shapes resolved, rewires
// multiOutputsNodes pointers to the resolved copies, and recomputes
// shape-dependent node.Data for operations that implement RecomputableNodeData.
func (e *Executable) createSpecialization(bindings shapes.AxisBindings) (spec *ShapeSpecialization, err error) {

	return specializeFunction(e.backend, e.builder.MainFn.Nodes, bindings)
}

func specializeFunction(backend *Backend, origNodes []*Node, bindings shapes.AxisBindings) (spec *ShapeSpecialization, err error) {
	resolved := make([]*Node, len(origNodes))

	// First pass: shallow copy each node and resolve its shape.
	for i, orig := range origNodes {
		if orig == nil {
			continue
		}
		resolvedShape := orig.Shape
		if orig.Shape.IsDynamic() {
			var err error
			resolvedShape, err = orig.Shape.ResolvePartial(bindings)
			if err != nil {
				return nil, errors.WithMessage(err, "createSpecialization: Shape.ResolvePartial")
			}
		}
		n := &Node{
			Index:              orig.Index,
			Inputs:             orig.Inputs,
			CapturedInputs:     orig.CapturedInputs,
			OpType:             orig.OpType,
			Shape:              resolvedShape,
			Builder:            orig.Builder,
			Function:           orig.Function,
			Data:               orig.Data,
			IsNodeSelectOutput: orig.IsNodeSelectOutput,
			SelectOutputIdx:    orig.SelectOutputIdx,
		}

		// Resolve multi-output shapes if present.
		if orig.MultiOutputsShapes != nil {
			n.MultiOutputsShapes = make([]shapes.Shape, len(orig.MultiOutputsShapes))
			for j, s := range orig.MultiOutputsShapes {
				n.MultiOutputsShapes[j], err = s.ResolvePartial(bindings)
				if err != nil {
					return nil, errors.WithMessage(err, "createSpecialization: Shape.Resolve multi-output")
				}
			}
			// MultiOutputsNodes will be rewired in the second pass.
			n.MultiOutputsNodes = orig.MultiOutputsNodes
		}

		resolved[i] = n
	}

	// Second pass: rewire MultiOutputsNodes to point to resolved copies.
	for _, n := range resolved {
		if n == nil {
			continue
		}
		if n.MultiOutputsNodes != nil {
			rewired := make([]*Node, len(n.MultiOutputsNodes))
			for i, sub := range n.MultiOutputsNodes {
				rewired[i] = resolved[sub.Index]
			}
			n.MultiOutputsNodes = rewired
		}
	}

	// Third pass: recompute shape-dependent node.Data for ops that need it.
	for i, orig := range origNodes {
		if orig == nil || !anyInputHasDynamicDims(orig, nil) {
			continue // Static inputs → original data is fine
		}
		// Data-dependent or anonymous dimensions may not be bound by parameters.
		// Leave their metadata unchanged here; specializeNode recomputes it just
		// before op dispatch once concrete input buffers are available.
		if anyInputHasDynamicDims(orig, resolved) {
			continue
		}
		if recomputable, ok := orig.Data.(RecomputableNodeData); ok {
			newData, recomputeErr := recomputable.Recompute(backend, resolved, resolved[i])
			if recomputeErr != nil {
				return nil, errors.WithMessagef(recomputeErr, "specialization: recomputing data for node %d (%s)", i, orig.OpType)
			}
			resolved[i].Data = newData
		}
	}

	return &ShapeSpecialization{
		bindings:      bindings,
		resolvedNodes: resolved,
	}, nil
}

// anyInputHasDynamicDims reports whether an input (including captured inputs)
// has dynamic dimensions. If resolvedNodes is non-nil, it checks those copies.
func anyInputHasDynamicDims(n *Node, resolvedNodes []*Node) bool {
	for _, input := range n.Inputs {
		if resolvedNodes != nil {
			input = resolvedNodes[input.Index]
		}
		if input.Shape.IsDynamic() {
			return true
		}
	}
	for _, closureCaptures := range n.CapturedInputs {
		for _, capturedInput := range closureCaptures {
			if resolvedNodes != nil {
				capturedInput = resolvedNodes[capturedInput.Index]
			}
			if capturedInput.Shape.IsDynamic() {
				return true
			}
		}
	}
	return false
}

// getOrCreateSpecialization returns a cached specialization for the given bindings,
// creating one if it doesn't exist. Thread-safe via sync.Map.
func (e *Executable) getOrCreateSpecialization(bindings shapes.AxisBindings) (*ShapeSpecialization, error) {
	key := bindings.Key()
	if spec, ok := e.specializations.Load(key); ok {
		return spec.(*ShapeSpecialization), nil
	}
	newSpec, err := e.createSpecialization(bindings)
	if err != nil {
		return nil, err
	}
	actual, _ := e.specializations.LoadOrStore(key, newSpec)
	return actual.(*ShapeSpecialization), nil
}

// specializeNode specializes a single node just before op dispatch using the
// concrete input buffers available when the node is scheduled.
//
// Unlike function-level specialization (specializeFunction), which resolves
// parameter-bound dynamic axes once at Execute entry, specializeNode handles
// dimensions that are only known during execution — such as outputs of
// data-dependent operations (NonZero, DynamicReshape, DynamicIota,
// DynamicBroadcastInDim) as well as unspecialized intermediate dynamic axes
// (e.g., strided ConvGeneral or DynamicPad outputs with generated "#..." axis names).
// All returned copies are private to the current node execution.
func specializeNode(backend *Backend, node *Node, inputs []*Buffer, spec *ShapeSpecialization) (*Node, error) {
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
		maxIndex = max(maxIndex, input.Index)
		if inputs[i] == nil {
			continue // Cached constant input.
		}
		if input.Shape.IsDynamic() {
			if err := bindings.Extract(input.Shape, inputs[i].RawShape); err != nil {
				return nil, err
			}
		}
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
				return nil, errors.Wrapf(err, "specializing shape for output %d of %s", i, node.OpType)
			}
		}
	}
	specialized := &Node{
		Index: node.Index, Inputs: node.Inputs, CapturedInputs: node.CapturedInputs,
		OpType: node.OpType, Shape: shape, Builder: node.Builder, Function: node.Function,
		Data: node.Data, IsNodeSelectOutput: node.IsNodeSelectOutput, SelectOutputIdx: node.SelectOutputIdx,
		MultiOutputsShapes: multiOutputsShapes, MultiOutputsNodes: node.MultiOutputsNodes,
	}
	specialized.cachedExecutorIdx.Store(node.cachedExecutorIdx.Load())
	if recomputable, ok := node.Data.(RecomputableNodeData); ok {
		nodes := make([]*Node, maxIndex+1)
		nodes[node.Index] = specialized
		for i, input := range node.Inputs {
			if inputs[i] != nil {
				nodes[input.Index] = &Node{Index: input.Index, Shape: inputs[i].RawShape}
			} else {
				nodes[input.Index] = input // Cached constant input: static Shape is already concrete.
			}
		}
		specialized.Data, err = recomputable.Recompute(backend, nodes, specialized)
		if err != nil {
			return nil, errors.Wrapf(err, "recomputing node data for %s", node.OpType)
		}
	}
	return specialized, nil
}

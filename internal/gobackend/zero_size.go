// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
)

// newEmptyConstantData creates a ConstantData holding a 0-byte Buffer of the given static zero-sized shape.
func newEmptyConstantData(backend *Backend, shape shapes.Shape) *ConstantData {
	cleanShape := shape.Clone()
	cleanShape.AxisNames = nil
	buf := NewBuffer(cleanShape)
	buf.RawBackend = backend
	cd := &ConstantData{
		backend: backend,
		shape:   cleanShape,
	}
	cd.rawBuffer.Store(buf)
	return cd
}

// foldZeroSizeNodes simplifies nodes in f[:numNodesToProcess] whose output shape has at least one
// static zero dimension (Shape.IsZeroSize() == true), prior to dead-code elimination (countNodeUsesAndDependents):
//
//  1. Fully static zero-size output (!node.Shape.IsDynamic(), e.g. Float32[0, 4]):
//     Converted in-place into a leaf OpTypeConstant with Inputs = nil, severing all upstream
//     data dependencies so unused producer subgraphs are dead-code-eliminated.
//  2. Dynamic zero-size output (node.Shape.IsDynamic(), e.g. Float32["batch", 0]):
//     Converted in-place into a zero-op shape-dependency node (IsZeroSizeShapeOnly = true) whose
//     Inputs are rewired directly to the earliest node(s) in f defining each dynamic axis variable
//     (e.g. the input Parameter), allowing intermediate compute subgraphs to be dead-code-eliminated
//     while still propagating concrete dynamic dimensions at execution time.
//
// Nodes whose inputs still require runtime shape/bounds validation (such as Dynamic* ops taking
// runtime scalar dimensions or binary/ternary ops unifying distinct "#..." dynamic axis names)
// are left untouched so runtime errors continue to be reported accurately.
func foldZeroSizeNodes(f *Function, numNodesToProcess int) {
	rootFn := f
	for rootFn.RawParent != nil {
		rootFn = rootFn.RawParent
	}

	for nodeIdx := range numNodesToProcess {
		node := f.Nodes[nodeIdx]
		if !canFoldZeroSizeNode(node) {
			continue
		}

		if !node.Shape.IsDynamic() {
			cleanShape := node.Shape.Clone()
			cleanShape.AxisNames = nil
			node.OpType = compute.OpTypeConstant
			node.Shape = cleanShape
			node.Data = newEmptyConstantData(f.RawBuilder.Backend, cleanShape)
			node.Inputs = nil
			node.CapturedInputs = nil
			node.IsZeroSizeShapeOnly = false
			node.isConstant.Store(2)
			continue
		}

		sourceInputs, ok := resolveZeroSizeDynamicSources(f, rootFn, node)
		if !ok {
			continue
		}
		node.IsZeroSizeShapeOnly = true
		node.Inputs = sourceInputs
		node.CapturedInputs = nil
		node.Data = nil
		node.isConstant.Store(1)
	}
}

func canFoldZeroSizeNode(node *Node) bool {
	if node == nil || !node.Shape.Ok() || !node.Shape.IsZeroSize() {
		return false
	}
	if node.IsMultiOutputs() || node.IsNodeSelectOutput {
		return false
	}
	switch node.OpType {
	case compute.OpTypeParameter,
		compute.OpTypeConstant,
		compute.OpTypeCapturedValue,
		compute.OpTypeCall,
		compute.OpTypeOptimizationBarrier,
		compute.OpTypeSchedulingBarrier,
		compute.OpTypeDynamicReshape,
		compute.OpTypeDynamicBroadcastInDim,
		compute.OpTypeDynamicPad,
		compute.OpTypeDynamicIota,
		compute.OpTypeDynamicSlice,
		compute.OpTypeDynamicUpdateSlice:
		return false
	}

	if !anyInputHasDynamicDims(node, nil) {
		return true
	}
	return isDynamicInputZeroSizeFoldSafe(node)
}

// isDynamicInputZeroSizeFoldSafe reports whether folding a zero-size node with dynamic input
// dimensions is guaranteed not to bypass any runtime dimension-mismatch or bounds check.
func isDynamicInputZeroSizeFoldSafe(node *Node) bool {
	op := node.OpType
	if shapeinference.StandardUnaryOperations.Has(op) {
		return true
	}
	switch op {
	case compute.OpTypeIdentity,
		compute.OpTypeConvertDType,
		compute.OpTypeIsFinite,
		compute.OpTypeIsNaN,
		compute.OpTypeReal,
		compute.OpTypeImag,
		compute.OpTypeConj,
		compute.OpTypeTranspose,
		compute.OpTypeReverse,
		compute.OpTypeBroadcastInDim,
		compute.OpTypeReduceSum,
		compute.OpTypeReduceProduct,
		compute.OpTypeReduceMax,
		compute.OpTypeReduceMin,
		compute.OpTypeReduceBitwiseAnd,
		compute.OpTypeReduceBitwiseOr,
		compute.OpTypeReduceBitwiseXor,
		compute.OpTypeReduceLogicalAnd,
		compute.OpTypeReduceLogicalOr,
		compute.OpTypeReduceLogicalXor:
		return true
	}

	if shapeinference.AllBinaryOperations.Has(op) || op == compute.OpTypeWhere {
		// Safe iff no axis permissively unified two different dynamic axis names (e.g. "#a_1" vs "#b_1"),
		// which must still be checked for concrete equality at runtime.
		rank := 0
		for _, inp := range node.Inputs {
			if !inp.Shape.IsScalar() {
				rank = inp.Shape.Rank()
				break
			}
		}
		for axis := range rank {
			sharedDynName := ""
			for _, inp := range node.Inputs {
				if inp.Shape.IsScalar() || inp.Shape.Dimensions[axis] != shapes.DynamicDim {
					continue
				}
				name := inp.Shape.AxisName(axis)
				if sharedDynName == "" {
					sharedDynName = name
				} else if name != sharedDynName {
					return false
				}
			}
		}
		return true
	}

	return false
}

// resolveZeroSizeDynamicSources finds the earliest defining node(s) in f (with Index < node.Index)
// for every dynamic axis variable referenced in node.Shape, so specializeNode can extract concrete
// AxisBindings directly from those root producers.
func resolveZeroSizeDynamicSources(f, rootFn *Function, node *Node) ([]*Node, bool) {
	var sources []*Node
	addSource := func(src *Node) {
		if !slices.Contains(sources, src) {
			sources = append(sources, src)
		}
	}

	for axis, dim := range node.Shape.Dimensions {
		if dim != shapes.DynamicDim {
			continue
		}
		axisName := node.Shape.AxisName(axis)
		if axisName == "" {
			return nil, false
		}
		varNames := extractDynamicAxisVarNames(axisName)
		if len(varNames) == 0 {
			return nil, false
		}
		for _, varName := range varNames {
			localSrc := findLocalAxisSourceNode(f, rootFn, node, varName)
			if localSrc == nil {
				return nil, false
			}
			addSource(localSrc)
		}
	}
	return sources, true
}

// extractDynamicAxisVarNames extracts the underlying axis variable names from an axis name,
// handling both direct axis names ("batch", "#conv_1") and symbolic sum expressions ("=4+batch").
func extractDynamicAxisVarNames(axisName string) []string {
	if !strings.HasPrefix(axisName, shapes.SymbolicAxisPrefix) {
		return []string{axisName}
	}
	expr := strings.TrimPrefix(axisName, shapes.SymbolicAxisPrefix)
	parts := strings.Split(expr, "+")
	var vars []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil
		}
		if _, err := strconv.Atoi(part); err == nil {
			continue
		}
		if !slices.Contains(vars, part) {
			vars = append(vars, part)
		}
	}
	return vars
}

func findLocalAxisSourceNode(f, rootFn *Function, node *Node, varName string) *Node {
	if rootFn.dynamicAxisSources != nil {
		if src, ok := rootFn.dynamicAxisSources[varName]; ok && src.node != nil {
			if src.node.Function == f && src.node.Index < node.Index && shapeDefinesAxisVar(src.node.Shape, varName) {
				return src.node
			}
			for capIdx, parentNode := range f.CapturedParentNodes {
				if parentNode == src.node && capIdx < len(f.CapturedLocalNodes) {
					localCap := f.CapturedLocalNodes[capIdx]
					if localCap.Index < node.Index && shapeDefinesAxisVar(localCap.Shape, varName) {
						return localCap
					}
				}
			}
		}
	}
	for _, inp := range node.Inputs {
		if inp.Function == f && inp.Index < node.Index && shapeDefinesAxisVar(inp.Shape, varName) {
			return inp
		}
	}
	return nil
}

func shapeDefinesAxisVar(s shapes.Shape, varName string) bool {
	for i, dim := range s.Dimensions {
		if dim == shapes.DynamicDim && s.AxisName(i) == varName {
			return true
		}
	}
	return false
}

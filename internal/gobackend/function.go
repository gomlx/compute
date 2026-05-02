// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// Function implements compute.Function for SimpleGo.
type Function struct {
	notimplemented.Function

	Builder *Builder
	name    string

	// RawParent is the parent function if this is a closure.
	// For top-level functions (including main), this is nil.
	RawParent *Function

	// IsReturned indicates Return() was called.
	IsReturned bool

	// Nodes are all Nodes created within this function, in DAG order.
	// Each node's idx field is its index in this slice.
	Nodes []*Node

	// Outputs stores the return values set by Return().
	Outputs []*Node

	// Parameters stores the parameter nodes for this function.
	Parameters []*Node

	// CapturedParentNodes stores nodes from parent scopes that are captured by this closure.
	// The order matches capturedLocalNodes - CapturedParentNodes[i] is the parent node for capturedLocalNodes[i].
	CapturedParentNodes []*Node

	// CapturedLocalNodes stores the proxy nodes in this closure for captured values.
	// These are OpTypeCapturedValue nodes that receive their values at execution time.
	CapturedLocalNodes []*Node

	// nodeDedup provides automatic de-duplication for nodes within this function.
	nodeDedup map[NodeDedupKey][]*Node

	// compiled holds pre-compiled execution info.
	// This is set during Return() to allow efficient execution.
	Compiled *FunctionExecutable
}

// capturedNodeData is the data stored in a captured value node.
// It just stores the capture index since the parent node is available
// via f.capturedParentNodes[captureIdx].
type capturedNodeData int

var _ compute.Function = (*Function)(nil)

// CheckValid returns an error if the builder or the function are not ok.
func (f *Function) CheckValid() error {
	if f == nil || f.Builder == nil {
		return errors.Errorf("function is nil or undefined for %q", BackendName)
	}
	if f.Builder.IsCompiled {
		return errors.Errorf("cannot add new op to Function %q, builder has already been compiled", f.name)
	}
	return nil
}

// Name returns the name of this function.
// For closures, this returns "".
func (f *Function) Name() string {
	return f.name
}

// Parent returns the parent function if this is a closure.
// Returns nil for top-level functions (including main).
func (f *Function) Parent() compute.Function {
	if f.RawParent == nil {
		return nil
	}
	return f.RawParent
}

// IsAncestorOf checks whether f is an ancestor of leafFunc.
// It returns true if f == leafFunc.
//
// Typically, leafFunc will be a closure.
func (f *Function) IsAncestorOf(leafFunc *Function) bool {
	for ; leafFunc != nil; leafFunc = leafFunc.RawParent {
		if leafFunc == f {
			return true
		}
	}
	return false
}

// Closure creates a new closure function within this function.
// Closures can access values from their parent function's scope.
func (f *Function) Closure() (compute.Function, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	closure := &Function{
		Function: notimplemented.Function{
			ErrFn: notImplementedError,
		},
		Builder:   f.Builder,
		name:      "", // Closures have empty names
		RawParent: f,
		nodeDedup: make(map[NodeDedupKey][]*Node),
	}
	return closure, nil
}

// NewNode adds a new node of the given opType and shape to the function's graph.
// It's used by the other ops when creating new nodes.
// Nodes are added to the function's nodes slice.
//
// Use getOrCreateNode instead for most operations.
func (f *Function) NewNode(opType compute.OpType, shape shapes.Shape, inputs ...*Node) *Node {
	n := &Node{
		Builder:  f.Builder,
		OpType:   opType,
		Index:    len(f.Nodes),
		Shape:    shape,
		Inputs:   slices.Clone(inputs),
		Function: f,
	}
	f.Nodes = append(f.Nodes, n)
	return n
}

// NewMultiOutputsNode creates the multi-outputs node, and its "select nodes", one per output.
// The node.multiOutputsNodes will be set with the individual outputs and can be used by the Builder to return
// to the user.
// Nodes are added to the function's nodes slice.
//
// Note: no de-duplication of multi-output nodes.
func (f *Function) NewMultiOutputsNode(
	opType compute.OpType,
	outputShapes []shapes.Shape,
	inputs ...*Node,
) (node *Node) {
	node = f.NewNode(opType, shapes.Invalid(), inputs...)
	node.MultiOutputsShapes = outputShapes
	node.MultiOutputsNodes = make([]*Node, len(outputShapes))
	for i, shape := range outputShapes {
		node.MultiOutputsNodes[i] = &Node{
			Builder:            f.Builder,
			OpType:             opType,
			Index:              len(f.Nodes),
			Shape:              shape,
			Inputs:             []*Node{node},
			IsNodeSelectOutput: true,
			SelectOutputIdx:    i,
			Function:           f,
		}
		f.Nodes = append(f.Nodes, node.MultiOutputsNodes[i])
	}
	return node
}

// VerifyAndCastValues sanity checks that the values (compute.Op) are valid and created with this builder.
// If a node belongs to a parent function, it creates a capture node to access the value.
// It returns the underlying *Node of the values (with capture nodes substituted for parent values).
func (f *Function) VerifyAndCastValues(name string, values ...compute.Value) ([]*Node, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	nodes, err := f.Builder.checkValues(name, values...)
	if err != nil {
		return nil, err
	}

	// Check each node and handle parent scope references
	for idx, node := range nodes {
		if node.Function == nil {
			return nil, errors.Errorf(
				"%s: input #%d has nil function (internal error)",
				name, idx)
		}
		if node.Function == f {
			continue // Same function, OK.
		}

		// Check if the node is from an ancestor function (closure capture)
		isFromAncestor := false
		for ancestor := f.RawParent; ancestor != nil; ancestor = ancestor.RawParent {
			if node.Function == ancestor {
				isFromAncestor = true
				break
			}
		}
		if isFromAncestor {
			// Create or reuse a capture node for this parent value
			nodes[idx] = f.GetOrCreateCaptureNode(node)
		} else {
			// Node from a completely different function (not an ancestor)
			return nil, errors.Errorf(
				"%s: input #%d uses a node from a different function scope",
				name, idx)
		}
	}

	return nodes, nil
}

// NodeParameter data.
type NodeParameter struct {
	Name     string
	InputIdx int
}

// EqualNodeData implements nodeDataComparable for nodeParameter.
func (n *NodeParameter) EqualNodeData(other NodeDataComparable) bool {
	o := other.(*NodeParameter)
	return n.Name == o.Name && n.InputIdx == o.InputIdx
}

// Parameter creates an input parameter for this function.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *compute.ShardingSpec) (compute.Value, error) {
	dtype := shape.DType
	if dtype == dtypes.InvalidDType {
		return nil, errors.Errorf("invalid shape %s for Parameter", shape)
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Parameter: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, f.Builder.Backend.Name())
	}
	if sharding != nil {
		return nil, errors.Wrapf(
			notimplemented.NotImplementedError,
			"sharding spec %+v not supported for %q builder", sharding, BackendName)
	}
	data := &NodeParameter{
		Name:     name,
		InputIdx: len(f.Parameters), // Index within this function's parameters
	}
	n, _ := f.GetOrCreateNode(compute.OpTypeParameter, shape, nil, data)
	f.Parameters = append(f.Parameters, n)
	return n, nil
}

// Constant creates a constant in the function with the given flat values and the shape defined by the dimensions.
func (f *Function) Constant(flat any, dims ...int) (compute.Value, error) {
	_, err := f.VerifyAndCastValues("Constant")
	if err != nil {
		return nil, err
	}
	dtype, flatLen, err := checkFlat(flat)
	if err != nil {
		return nil, errors.WithMessagef(err, "Constant op")
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Constant: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, f.Builder.Backend.Name())
	}
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != flatLen {
		return nil, errors.Errorf("flat ([%d]%s) and shape size (%d) mismatch for constant value",
			flatLen, dtype, shape.Size())
	}
	data := &Buffer{
		RawShape: shape,
		Flat:     flat,
		InUse:    true,
	}
	n, _ := f.GetOrCreateNode(compute.OpTypeConstant, shape, nil, data)
	return n, nil
}

// Return marks the outputs of this function.
func (f *Function) Return(outputs []compute.Value, shardings []*compute.ShardingSpec) error {
	if err := f.CheckValid(); err != nil {
		return err
	}
	if f.IsReturned {
		return errors.Errorf("Return() already called for function %q", f.name)
	}
	if len(outputs) == 0 {
		return errors.Errorf("Return() requires at least one output")
	}
	if len(shardings) != 0 {
		return errors.Errorf("sharding or distributed execution are not supported by Go backend")
	}

	outputNodes, err := f.VerifyAndCastValues("Return", outputs...)
	if err != nil {
		return err
	}

	for _, node := range outputNodes {
		if len(node.MultiOutputsShapes) != 0 {
			return errors.Errorf(
				"%s node %q is internal (with multiple-outputs) and cannot be used for output",
				f.Builder.Name(),
				node.OpType,
			)
		}
	}

	f.Outputs = outputNodes
	f.IsReturned = true

	// If this is a closure or a named function (not main), pre-compile it for efficient execution.
	// Main functions are compiled later in Builder.Compile() after
	// duplicate output handling.
	if f.RawParent != nil || f.name != compute.MainName {
		compiled, err := newFunctionExecutable(f)
		if err != nil {
			return errors.WithMessagef(err, "failed to compile function %q", f.name)
		}
		f.Compiled = compiled
	}

	return nil
}

// Identity implements the compute.Identity interface.
// This operation is not de-duplicated: if you issue it twice, it will not reuse the previous instance.
func (f *Function) Identity(operandOp compute.Value) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("Reshape", operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	node := f.NewNode(compute.OpTypeIdentity, operand.Shape, operand)
	return node, nil
}

// Where implements the compute.Builder interface.
func (f *Function) Where(conditionOp, onTrueOp, onFalseOp compute.Value) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("Where", conditionOp, onTrueOp, onFalseOp)
	if err != nil {
		return nil, err
	}
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]
	outputShape, err := shapeinference.WhereOp(condition.Shape, onTrue.Shape, onFalse.Shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeWhere, outputShape, []*Node{condition, onTrue, onFalse}, nil)
	return node, nil
}

// Concatenate joins a sequence of tensors along the given axis (it must exist already).
// All input tensors must have the same shape, except potentially in the concatenation dimension.
// They must also have the same data type (DType).
// It returns an error if inputs are invalid (e.g., no inputs, mismatched graphs, shapes, dtypes, or invalid dimension).
func (f *Function) Concatenate(axis int, operandOps ...compute.Value) (compute.Value, error) {
	if len(operandOps) == 0 {
		return nil, errors.Errorf("Concatenate requires at least one input tensor")
	}
	operands, err := f.VerifyAndCastValues("Concatenate", operandOps...)
	if err != nil {
		return nil, err
	}

	// Extract shapes for shape inference.
	inputShapes := make([]shapes.Shape, len(operands))
	for i, opNode := range operands {
		inputShapes[i] = opNode.Shape
	}
	outputShape, err := shapeinference.ConcatenateOp(inputShapes, axis)
	if err != nil {
		return nil, err
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeConcatenate, outputShape, operands, axis)
	return node, nil
}

// ConvertDType converts operandOp to the given dtype. It implements the compute.Builder interface.
func (f *Function) ConvertDType(operandOp compute.Value, dtype dtypes.DType) (compute.Value, error) {
	opType := compute.OpTypeConvertDType
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	if operand.Shape.DType == dtype {
		// No-op
		return operand, nil
	}
	outputShape := operand.Shape.Clone()
	outputShape.DType = dtype
	node, _ := f.GetOrCreateNode(opType, outputShape, []*Node{operand}, nil)
	return node, nil
}

// ScatterMax implements the compute.Builder interface.
func (f *Function) ScatterMax(
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpls(
		compute.OpTypeScatterMax,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

// ScatterMin implements the compute.Builder interface.
func (f *Function) ScatterMin(
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpls(
		compute.OpTypeScatterMin,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

// ScatterSum implements the compute.Builder interface.
func (f *Function) ScatterSum(
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return f.scatterImpls(
		compute.OpTypeScatterSum,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

// scatterNode is attached to the Node.data field for ScatterMax, ScatterMin, ScatterSum.
type scatterNode struct {
	indexVectorAxis                                                int
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int
	indicesAreSorted, uniqueIndices                                bool
}

// EqualNodeData implements nodeDataComparable for scatterNode.
func (s *scatterNode) EqualNodeData(other NodeDataComparable) bool {
	o := other.(*scatterNode)
	if s.indexVectorAxis != o.indexVectorAxis ||
		s.indicesAreSorted != o.indicesAreSorted ||
		s.uniqueIndices != o.uniqueIndices {
		return false
	}
	return slices.Equal(s.updateWindowAxes, o.updateWindowAxes) &&
		slices.Equal(s.insertedWindowAxes, o.insertedWindowAxes) &&
		slices.Equal(s.scatterAxesToOperandAxes, o.scatterAxesToOperandAxes)
}

func (f *Function) scatterImpls(
	scatterOpType compute.OpType,
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (
	compute.Value, error) {
	inputs, err := f.VerifyAndCastValues(scatterOpType.String(), operandOp, scatterIndicesOp, updatesOp)
	if err != nil {
		return nil, err
	}
	operand, indices, updates := inputs[0], inputs[1], inputs[2]
	// Check that parameters are valid.
	outputShape, err := shapeinference.ScatterOp(
		operand.Shape,
		indices.Shape,
		updates.Shape,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
	)
	if err != nil {
		return nil, err
	}

	// The output shape of the scatter is the operand shape.
	data := &scatterNode{
		updateWindowAxes:         updateWindowAxes,
		insertedWindowAxes:       insertedWindowAxes,
		scatterAxesToOperandAxes: scatterAxesToOperandAxes,
		indexVectorAxis:          indexVectorAxis,
		indicesAreSorted:         indicesAreSorted,
		uniqueIndices:            uniqueIndices,
	}
	node, _ := f.GetOrCreateNode(scatterOpType, outputShape, []*Node{operand, indices, updates}, data)
	return node, nil
}

// sliceNode is attached to the Node.data field for Slice.
type sliceNode struct {
	starts, limits, strides []int
}

// EqualNodeData implements nodeDataComparable for sliceNode.
func (s *sliceNode) EqualNodeData(other NodeDataComparable) bool {
	o := other.(*sliceNode)
	return slices.Equal(s.starts, o.starts) &&
		slices.Equal(s.limits, o.limits) &&
		slices.Equal(s.strides, o.strides)
}

// Slice extracts a subarray from the input array.
// The subarray is of the same rank as the input and contains the values inside a bounding box within the input array
// where the dimensions and indices of the bounding box are given as arguments to the slice operation.
// The strides set the input stride of the slice in each axis and must be >= 1.
// It is optional, and if missing, it is assumed to be 1 for every dimension.
// Examples:
//
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
func (f *Function) Slice(operandOp compute.Value, starts, limits, strides []int) (compute.Value, error) {
	opType := compute.OpTypeSlice
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.SliceOp(operand.Shape, starts, limits, strides)
	if err != nil {
		return nil, err
	}
	data := &sliceNode{
		starts,
		limits,
		strides,
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*Node{operand}, data)
	return node, nil
}

// RNGBitGenerator generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RNGState or RNGStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func (f *Function) RNGBitGenerator(stateOp compute.Value, shape shapes.Shape) (newState, values compute.Value, err error) {
	opType := compute.OpTypeRNGBitGenerator
	inputs, err := f.VerifyAndCastValues(opType.String(), stateOp)
	if err != nil {
		return nil, nil, err
	}
	state := inputs[0]
	if !state.Shape.Equal(compute.RNGStateShape) {
		err := errors.Errorf(
			"expected random state to be shaped %s, got state.shape=%s instead for RNGBitGenerator",
			compute.RNGStateShape,
			state.Shape,
		)
		return nil, nil, err
	}
	outputShapes := []shapes.Shape{
		state.Shape.Clone(),
		shape.Clone(),
	}
	node := f.NewMultiOutputsNode(opType, outputShapes, state)
	newState = node.MultiOutputsNodes[0]
	values = node.MultiOutputsNodes[1]
	return
}

// argMinMaxNode with the axis and isMin fields.
type argMinMaxNode struct {
	axis  int
	isMin bool
}

// EqualNodeData implements nodeDataComparable for argMinMaxNode.
func (a *argMinMaxNode) EqualNodeData(other NodeDataComparable) bool {
	o := other.(*argMinMaxNode)
	return a.axis == o.axis && a.isMin == o.isMin
}

// ArgMinMax calculates the "argmin" or "argmax" across an axis of the given input array x.
// outputDType defines the output of the argmin/argmax, it doesn't need to be the same as the input.
// It's a form of reduction on the given axis, and that axis goes away. So the rank of the result is one less than
// the rank of x.
// Examples:
//
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=1, isMin=true) -> {1, 0}  // (it chooses the 0 and the -3)
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=0, isMin=false) -> {0, 1, 0} // (it choose the 2, 4 and 7)
func (f *Function) ArgMinMax(
	operandOp compute.Value,
	axis int,
	outputDType dtypes.DType,
	isMin bool,
) (compute.Value, error) {
	opType := compute.OpTypeArgMinMax
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ArgMinMaxOp(operand.Shape, axis, outputDType)
	if err != nil {
		return nil, err
	}
	data := &argMinMaxNode{
		axis,
		isMin,
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*Node{operand}, data)
	return node, nil
}

type reduceWindowNode struct {
	reductionType                                              compute.ReduceOpType
	windowDimensions, strides, inputDilations, windowDilations []int
	paddings                                                   [][2]int
}

// EqualNodeData implements nodeDataComparable for reduceWindowNode.
func (r *reduceWindowNode) EqualNodeData(other NodeDataComparable) bool {
	o := other.(*reduceWindowNode)
	if r.reductionType != o.reductionType {
		return false
	}
	return slices.Equal(r.windowDimensions, o.windowDimensions) &&
		slices.Equal(r.strides, o.strides) &&
		slices.Equal(r.inputDilations, o.inputDilations) &&
		slices.Equal(r.windowDilations, o.windowDilations) &&
		slices.Equal(r.paddings, o.paddings)
}

// ReduceWindow runs a reduction function of the type given by reductionType,
// it can be either ReduceMaxNode, ReduceSumNode, or ReduceMultiplyNode.
//
//   - reductionType: the type of reduction to perform. E.g.: [ReduceOpMax], [ReduceOpSum],...
//   - windowDimensions: the dimensions of the window, must be defined for each axis.
//   - strides: stride over elements in each axis for each window reduction. If nil, it's assume to be the
//     same as the windowDimensions -- that is, the strides jump a window at a time.
//   - inputDilations: "virtually" expand the input by introducing "holes" between elements. I.e. if
//     inputDilations are 2, then the input is expanded by inserting `2-1` copies of `0` (or whatever
//     is the reduciton "zero" value) between the elements in each dimension.
//     If nil, it's assumed to be 1 (no dilation) for each axis. Values must be >= 1.
//   - windowDilations: "virtually" expand the window by inserting `2-1` copies of `0` between the
//     elements in each dimension.
//     If nil, it's assumed to be 1 (no dilation) for each axis. Values must be >= 1.
//   - paddings: virtual padding to be added to the input at the edges (start and end) of each axis.
//     If nil, it's assumed to be 0 for each axis.
func (f *Function) ReduceWindow(
	operandOp compute.Value,
	reductionType compute.ReduceOpType,
	windowDimensions, strides, inputDilations, windowDilations []int,
	paddings [][2]int,
) (compute.Value, error) {
	opType := compute.OpTypeReduceWindow
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ReduceWindowOp(
		operand.Shape,
		windowDimensions,
		strides,
		inputDilations,
		windowDilations,
		paddings,
	)
	if err != nil {
		return nil, err
	}
	data := &reduceWindowNode{
		reductionType:    reductionType,
		windowDimensions: windowDimensions,
		strides:          strides,
		inputDilations:   inputDilations,
		windowDilations:  windowDilations,
		paddings:         paddings,
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*Node{operand}, data)
	return node, nil
}

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (f *Function) Clamp(minV, x, maxV compute.Value) (compute.Value, error) {
	clamped, err := f.Max(minV, x)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	clamped, err = f.Min(clamped, maxV)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	return clamped, nil
}

// IsNaN implements compute.Builder interface.
func (f *Function) IsNaN(x compute.Value) (compute.Value, error) {
	result, err := f.NotEqual(x, x)
	if err != nil {
		return nil, errors.WithMessage(err, "while building op IsNaN")
	}
	return result, nil
}

// AllReduce implements the compute.CollectiveOps interface.
func (f *Function) AllReduce(_ []compute.Value, _ compute.ReduceOpType, _ [][]int) ([]compute.Value, error) {
	return nil, errors.Wrapf(
		notimplemented.NotImplementedError,
		"AllReduce not supported for %q builder", BackendName)
}

// Pad implements the compute.Builder interface.
func (f *Function) Pad(operandOp, fillValueOp compute.Value, axesConfig ...compute.PadAxis) (compute.Value, error) {
	opType := compute.OpTypePad
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp, fillValueOp)
	if err != nil {
		return nil, err
	}
	operand, fillValue := inputs[0], inputs[1]

	outputShape, err := shapeinference.PadOp(operand.Shape, axesConfig...)
	if err != nil {
		return nil, err
	}

	data := &padNode{axesConfig: slices.Clone(axesConfig)}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*Node{operand, fillValue}, data)
	return node, nil
}

type padNode struct {
	axesConfig []compute.PadAxis
}

// EqualNodeData implements nodeDataComparable for padNode.
func (p *padNode) EqualNodeData(other NodeDataComparable) bool {
	o := other.(*padNode)
	return slices.Equal(p.axesConfig, o.axesConfig)
}

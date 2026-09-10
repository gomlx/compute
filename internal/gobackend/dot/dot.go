// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package dot implements a general-purpose "dot product" ("Einsum")
// computation.
//
// It provides a registration system for the underlying "MatMul" (matrix
// multiplication) implementations -- to it allows pluggability of different
// implementations, so one can easily experiment with it. The sub-package `matmul`
// provides the base implementation (including some SIMD variants)
//
// The actual "MatMul" implementation used is in part selected at build-time
// (depending on the architecture and tags), at initialization time (based on
// presence of SIMD features) and during graph-build time, based on the input
// shapes, layout and dtypes.
//
// Environment variables that can be used to disable certain features:
//
//   - GOMLX_GO_DOT_MATMUL: set to false to disable the default matmul implementation.
//     if you haven't added other plugin implementations, it will effectively disable DotGeneral.
//   - GOMLX_GO_SIMD_AVX512: set to false to disable AVX512-specific implementations, even if the runtime architecture allows it.
//   - GOMLX_GO_SIMD_AVX2: set to false to disable AVX2-specific implementations, even if the runtime architecture allows it.
package dot

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Generate the DTypeMag and DTypePairMap registrations:
//go:generate go run ../../cmd/gobackend_dtypemap

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeDotGeneral, gobackend.PriorityGeneric, execDotGeneral)
	gobackend.RegisterDotGeneral.Register(DotGeneral, gobackend.PriorityGeneric)
}

// NodeData associated to a DotGeneral Node: gathered during graph building, it should include
// all the information needed to execute it.
type NodeData struct {
	InputDType, OutputDType                                dtypes.DType
	Config                                                 compute.DotGeneralConfig
	Layout                                                 Layout
	LHSContractingAxes, LHSBatchAxes                       []int
	RHSContractingAxes, RHSBatchAxes                       []int
	BatchSize, LHSCrossSize, RHSCrossSize, ContractingSize int

	// implementation for current layout.
	implementation *ImplementationRegistration
}

// SetSizes computes and sets the internal sizes (BatchSize, LHSCrossSize, RHSCrossSize, ContractingSize)
// from the concrete LHS and RHS shapes.
func (d *NodeData) SetSizes(lhsShape, rhsShape shapes.Shape) {
	numBatchAxes := len(d.LHSBatchAxes)
	d.BatchSize = 1
	for i := range numBatchAxes {
		dim := lhsShape.Dimensions[d.LHSBatchAxes[i]]
		if dim == shapes.DynamicDim {
			d.BatchSize = shapes.DynamicDim
			break
		}
		d.BatchSize *= dim
	}

	d.ContractingSize = 1
	for _, axis := range d.LHSContractingAxes {
		dim := lhsShape.Dimensions[axis]
		if dim == shapes.DynamicDim {
			d.ContractingSize = shapes.DynamicDim
			break
		}
		d.ContractingSize *= dim
	}

	lhsRank := lhsShape.Rank()
	d.LHSCrossSize = 1
	isBatchOrContractingLHS := make([]bool, lhsRank)
	for _, axis := range d.LHSBatchAxes {
		isBatchOrContractingLHS[axis] = true
	}
	for _, axis := range d.LHSContractingAxes {
		isBatchOrContractingLHS[axis] = true
	}
	for axis := range lhsRank {
		if !isBatchOrContractingLHS[axis] {
			dim := lhsShape.Dimensions[axis]
			if dim == shapes.DynamicDim {
				d.LHSCrossSize = shapes.DynamicDim
				break
			}
			d.LHSCrossSize *= dim
		}
	}

	rhsRank := rhsShape.Rank()
	d.RHSCrossSize = 1
	isBatchOrContractingRHS := make([]bool, rhsRank)
	for _, axis := range d.RHSBatchAxes {
		isBatchOrContractingRHS[axis] = true
	}
	for _, axis := range d.RHSContractingAxes {
		isBatchOrContractingRHS[axis] = true
	}
	for axis := range rhsRank {
		if !isBatchOrContractingRHS[axis] {
			dim := rhsShape.Dimensions[axis]
			if dim == shapes.DynamicDim {
				d.RHSCrossSize = shapes.DynamicDim
				break
			}
			d.RHSCrossSize *= dim
		}
	}
}

// EqualNodeData implements nodeDataComparable for dotGeneralNodeData.
func (d *NodeData) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*NodeData)
	if d.BatchSize != o.BatchSize ||
		d.LHSCrossSize != o.LHSCrossSize ||
		d.RHSCrossSize != o.RHSCrossSize ||
		d.ContractingSize != o.ContractingSize {
		return false
	}
	return slices.Equal(d.LHSContractingAxes, o.LHSContractingAxes) &&
		slices.Equal(d.LHSBatchAxes, o.LHSBatchAxes) &&
		slices.Equal(d.RHSContractingAxes, o.RHSContractingAxes) &&
		slices.Equal(d.RHSBatchAxes, o.RHSBatchAxes)
}

// Recompute implements gobackend.RecomputableNodeData for NodeData.
func (d *NodeData) Recompute(backend *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
	newData := &NodeData{
		InputDType:         d.InputDType,
		OutputDType:        d.OutputDType,
		Config:             d.Config,
		Layout:             d.Layout,
		LHSContractingAxes: slices.Clone(d.LHSContractingAxes),
		LHSBatchAxes:       slices.Clone(d.LHSBatchAxes),
		RHSContractingAxes: slices.Clone(d.RHSContractingAxes),
		RHSBatchAxes:       slices.Clone(d.RHSBatchAxes),
	}

	// Get resolved (concrete) input shapes.
	lhsShape := resolvedNodes[originalNode.Inputs[0].Index].Shape
	rhsShape := resolvedNodes[originalNode.Inputs[1].Index].Shape

	// Recompute sizes from concrete shapes.
	newData.SetSizes(lhsShape, rhsShape)

	// Re-find implementation (might change based on new sizes).
	newData.implementation = FindRegisteredImplementation(newData.Layout, newData.InputDType, newData.OutputDType)
	if newData.implementation == nil {
		return nil, errors.Errorf("specialization: no DotGeneral implementation found for layout=%s and dtypes=%s,%s",
			newData.Layout, newData.InputDType, newData.OutputDType)
	}

	return newData, nil
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// The resulting shape is [batchIndices..., <lhs cross indices...>, <rhs cross indices...>], the
// indices come in the order they were provided.
// The output dtype is by default the same as the input, except if configured otherwise in config.OutputDType.
//
// This is the graph building part of DotGeneral. It reshapes and transposes the inputs as needed
// to transform them into one of the two layouts the implementation functions (see execDotGeneral)
// know how to handle.
func DotGeneral(f *gobackend.Function,
	lhsValue compute.Value, lhsContractingAxes, lhsBatchAxes []int,
	rhsValue compute.Value, rhsContractingAxes, rhsBatchAxes []int,
	config compute.DotGeneralConfig) (compute.Value, error) {

	// Get and sanity check graph nodes from values.
	directInputs, err := f.VerifyAndCastValues("DotGeneral", lhsValue, rhsValue)
	if err != nil {
		return nil, err
	}
	lhs, rhs := directInputs[0], directInputs[1]
	if klog.V(1).Enabled() {
		klog.Infof("DotGeneral lhs=%s rhs=%s contracting=%v,%v batch=%v,%v - config=%+v\n",
			lhs.Shape, rhs.Shape,
			lhsContractingAxes, rhsContractingAxes,
			lhsBatchAxes, rhsBatchAxes, config)
	}

	// Verify inputs validity for DotGeneral and compute the output shape.
	outputShape, err := shapeinference.DotGeneral(lhs.Shape, lhsContractingAxes, lhsBatchAxes, rhs.Shape, rhsContractingAxes, rhsBatchAxes, config)
	if err != nil {
		return nil, err
	}

	// Create node params or the "normalized" dot-general, after all reshaping/transposition.
	// - LayoutNonTransposed: lhs=[batch, lhsCross, contracting], rhs=[batch, contracting, rhsCross]
	// - LayoutTransposed:    lhs=[batch, lhsCross, contracting], rhs=[batch, rhsCross, contracting]
	// In usual MatMul works, B=batch, M=lhsCross, K=contracting, N=rhsCross.
	params := &NodeData{
		InputDType:         lhs.Shape.DType,
		OutputDType:        lhs.Shape.DType, // output DType for now is assumed to be the same as the input.
		Config:             config,
		LHSContractingAxes: lhsContractingAxes,
		LHSBatchAxes:       lhsBatchAxes,
		RHSContractingAxes: rhsContractingAxes,
		RHSBatchAxes:       rhsBatchAxes,
	}
	lhs, rhs, params, err = reshapeToSupportedLayout(f, lhs, rhs, params)
	if err != nil {
		return nil, err
	}

	// Only for half-types inputs we always output (accumulator dtype) Float32 for the "normalized" dot-general.
	if params.InputDType.IsHalfPrecision() {
		params.OutputDType = dtypes.Float32
	}

	// Accumulator dtype conversion: except Float32 accumulator for half precision inputs,
	// we simply convert the inputs to the accumulator dtype.
	lhs, rhs, params, err = convertToAccumulatorDType(f, lhs, rhs, params)
	if err != nil {
		return nil, err
	}

	// Find sizes of the normalized operands (batchSize, crossSizes and contractSize).
	// The shape is already normalized (to a LayoutNonTranposed or LayoutTransposed), so
	// there is at most one axis of each type.
	params.SetSizes(lhs.Shape, rhs.Shape)

	// Find a registered implementation for the current layout.
	inputs := []*gobackend.Node{lhs, rhs}
	params.implementation = FindRegisteredImplementation(params.Layout, params.InputDType, params.OutputDType)
	if klog.V(1).Enabled() {
		if params.implementation != nil {
			klog.Infof("Using DotGeneral implementation %q for layout=%s and dtypes=%s,%s",
				params.implementation.name, params.Layout, params.InputDType, params.OutputDType)
		} else {
			klog.Infof("No registered DotGeneral implementation found for layout=%s and dtypes=%s,%s",
				params.Layout, params.InputDType, params.OutputDType)

		}
	}
	if params.implementation == nil {
		return nil, errors.Errorf("no DotGeneral implementation found for layout=%s and dtypes=%s,%s",
			params.Layout, params.InputDType, params.OutputDType)
	}

	// Create dot-general node: it directly generates outputShape (with OutputDType).
	//
	// Output Shape & Row-Major Layout Invariant:
	// Standard DotGeneral specifies output dimensions in the order:
	//   [BatchAxes..., LHSCrossAxes..., RHSCrossAxes...]
	// The underlying GEMM computes C[b, n, m] = sum_k A[b, n, k] * B[b, m, k] (or B[b, k, m])
	// and writes sequentially into output.Flat in row-major order:
	//   outer loop: batch (b)
	//   middle loop: lhs cross (n)
	//   inner loop: rhs cross (m)
	// In row-major layout of any multi-dimensional tensor with shape [B..., N..., M...],
	// the flat index b_flat * (N * M) + n_flat * M + m_flat exactly matches this GEMM output order.
	// Therefore, the node directly outputs outputShape (preserving all original dimensions, dynamic
	// axes, and names). No subsequent reshape (static or DynamicReshape) is needed.
	nodeOutputShape := outputShape
	if params.OutputDType != outputShape.DType {
		nodeOutputShape = outputShape.Clone()
		nodeOutputShape.DType = params.OutputDType
	}
	result, _ := f.GetOrCreateNode(compute.OpTypeDotGeneral, nodeOutputShape, inputs, params)

	if result.Shape.DType != outputShape.DType {
		// Requires final DType conversion:
		resultValue, err := f.ConvertDType(result, outputShape.DType)
		if err != nil {
			return nil, err
		}
		result = resultValue.(*gobackend.Node)
	}
	return result, nil
}

// reshapeToSupportedLayout checks if lhs and rhs already match a layout supported
// by GEMM kernels (LayoutTransposed or LayoutNonTransposed).
// If not, it transposes the inputs via TransposeToLayout to LayoutTransposed.
//
// Note on axis merging:
// No reshape is performed here for non-incompatible layouts. Contiguous axes belonging to the same semantic category
// (all batch axes, all cross axes, or all contracting axes) already have contiguous element
// offsets in row-major memory. The GEMM kernels operate on flat slices using total combined
// sizes (SetSizes), so preserving unmerged axes is both faster and fully compatible with dynamic dims.
func reshapeToSupportedLayout(
	f *gobackend.Function,
	lhs, rhs *gobackend.Node,
	params *NodeData,
) (lhsOut, rhsOut *gobackend.Node, paramsOut *NodeData, err error) {
	params.Layout = LayoutForDotGeneral(
		lhs.Shape, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs.Shape, params.RHSContractingAxes, params.RHSBatchAxes)
	if params.Layout != LayoutIncompatible {
		// Already a supported layout.
		return lhs, rhs, params, nil
	}

	// We need to transpose inputs to a supported layout. Since
	// LayoutTransposed is the fastest, we transpose to that.
	targetLayout := LayoutTransposed
	lhs, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs, params.RHSContractingAxes, params.RHSBatchAxes, err = TransposeToLayout(
		f,
		lhs, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs, params.RHSContractingAxes, params.RHSBatchAxes,
		targetLayout)
	if err != nil {
		return nil, nil, nil, err
	}
	params.Layout = targetLayout
	return lhs, rhs, params, nil
}

// convertToAccumulatorDType if an accumulator dtype is specified -- for the algorithms that don't support a different accumulator type.
func convertToAccumulatorDType(f *gobackend.Function, lhs, rhs *gobackend.Node, params *NodeData) (*gobackend.Node, *gobackend.Node, *NodeData, error) {
	accDType := params.Config.AccumulatorDType
	if accDType == dtypes.InvalidDType || accDType == params.InputDType {
		return lhs, rhs, params, nil
	}
	if accDType == params.OutputDType {
		// The output dtype will be used already, no need to convert.
		return lhs, rhs, params, nil
	}

	// Exception: Half-Precision types automatically uses Float32 for the computation.
	if params.InputDType.IsHalfPrecision() && accDType == dtypes.Float32 {
		return lhs, rhs, params, nil
	}

	// Convert inputs to accumulator dtype.
	if klog.V(2).Enabled() {
		klog.Infof("Converting inputs from %s to accumulator DType=%s\n", lhs.Shape.DType, accDType)
	}
	lhsOp, err := f.ConvertDType(lhs, accDType)
	if err != nil {
		return nil, nil, nil, err
	}
	lhs = lhsOp.(*gobackend.Node)
	rhsOp, err := f.ConvertDType(rhs, accDType)
	if err != nil {
		return nil, nil, nil, err
	}
	rhs = rhsOp.(*gobackend.Node)
	params.InputDType = accDType
	params.OutputDType = accDType
	return lhs, rhs, params, nil
}

// execDotGeneral executes the DotGeneral operation.
//
// The output buffer is allocated with node.Shape (which is outputShape).
// The GEMM kernels write into output.Flat in row-major order:
//
//	b_flat * (lhsCrossSize * rhsCrossSize) + lhsCross_flat * rhsCrossSize + rhsCross_flat
//
// This matches the row-major flat layout of [BatchAxes..., LHSCrossAxes..., RHSCrossAxes...] exactly,
// so the allocated buffer immediately possesses the target N-dimensional shape without any reshape.
func execDotGeneral(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	params := node.Data.(*NodeData)
	outputShape := node.Shape
	output, err := backend.GetBuffer(outputShape)

	if err != nil {
		return nil, err
	}

	if params.implementation == nil {
		return nil, errors.Errorf("no DotGeneral implementation found for layout=%s and dtypes=%s,%s",
			params.Layout, params.InputDType, params.OutputDType)
	}

	if backend.NoOps {
		return output, nil
	}

	// Use registered implementation.
	CallRegisteredImplementation(backend, params.implementation, lhs, rhs, output, params)
	return output, nil
}

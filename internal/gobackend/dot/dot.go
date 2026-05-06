// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package dot implements a general-purpose "dot product" ("Einsum") computation.
//
// It has various implmentations, each optimized for different circumstances:
//
//   - "normalized": the general implementation that works for any shape/dtype.
//   - "blocked": a cache-tiled algorithm that is faster for larger inputs.
//   - "smallmatmul": optimized for small matrices.
//   - "packgemm": uses the "packgemm" library for matrix multiplication.
//   - "highway": uses the "highway" library for matrix multiplication.
//   - "check": a debug path that checks all implementations against each other.
//
// The actual implementation used is selected at graph-build time based on the input shapes and dtypes.
package dot

import (
	"fmt"
	"math"
	"math/bits"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot/highway"
	"github.com/gomlx/compute/internal/gobackend/dot/packgemm"
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

	// Register DotGeneral config options for the backend.
	for _, option := range []string{"dotgeneral_normalized", "dotgeneral_blocked", "dotgeneral_check",
		"dotgeneral_smallmatmul", "dotgeneral_packgemm", "dotgeneral_highway"} {
		gobackend.KnownOptionsSetters[option] = SetBackendOption
	}
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
	LHSBlockedShape, RHSBlockedShape, OutputBlockedShape   shapes.Shape

	// execPath determines which execution strategy to use. Decided at graph-build time.
	execPath ExecutionPath
}

// EqualNodeData implements nodeDataComparable for dotGeneralNodeData.
func (d *NodeData) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*NodeData)
	if d.BatchSize != o.BatchSize ||
		d.LHSCrossSize != o.LHSCrossSize ||
		d.RHSCrossSize != o.RHSCrossSize ||
		d.ContractingSize != o.ContractingSize ||
		d.execPath != o.execPath {
		return false
	}
	return slices.Equal(d.LHSContractingAxes, o.LHSContractingAxes) &&
		slices.Equal(d.LHSBatchAxes, o.LHSBatchAxes) &&
		slices.Equal(d.RHSContractingAxes, o.RHSContractingAxes) &&
		slices.Equal(d.RHSBatchAxes, o.RHSBatchAxes) &&
		d.LHSBlockedShape.Equal(o.LHSBlockedShape) &&
		d.RHSBlockedShape.Equal(o.RHSBlockedShape) &&
		d.OutputBlockedShape.Equal(o.OutputBlockedShape)
}

// SetSizes sets the dot-general sizes according to the axes dimensions.
// Assumes shape has been normalized to one of the two supported layouts.
func (d *NodeData) SetSizes(lhsShape, rhsShape shapes.Shape) {
	d.BatchSize = 1
	if len(d.LHSBatchAxes) > 0 {
		d.BatchSize = lhsShape.Dimensions[d.LHSBatchAxes[0]]
	}
	d.LHSCrossSize = 1
	for i, dim := range lhsShape.Dimensions {
		if !slices.Contains(d.LHSContractingAxes, i) && !slices.Contains(d.LHSBatchAxes, i) {
			d.LHSCrossSize *= dim
		}
	}
	d.RHSCrossSize = 1
	for i, dim := range rhsShape.Dimensions {
		if !slices.Contains(d.RHSContractingAxes, i) && !slices.Contains(d.RHSBatchAxes, i) {
			d.RHSCrossSize *= dim
		}
	}
	d.ContractingSize = 1
	if len(d.LHSContractingAxes) > 0 {
		// We could have gotten from lhs or rhs, they must match.
		d.ContractingSize = lhsShape.Dimensions[d.LHSContractingAxes[0]]
	}
}

// adjustAxisToRank returns a positive axis, adjusting negative numbers to the correct rank.
func adjustAxisToRank(rank, axis int) (int, error) {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return -1, errors.Errorf("axis %d is out of range [0, %d)", axis, rank)
	}
	return axis, nil
}

// SetBackendOption process the configuration options for DotGeneral.
func SetBackendOption(b *gobackend.Backend, key string) error {
	switch key {
	case "dotgeneral_normalized":
		// Force DotGeneral to use the normalized path (transpose to [B,Cross,Contract] form).
		b.DotGeneralForceExecutionPath = int(SmallTransposedPath)
	case "dotgeneral_blocked":
		// Force DotGeneral to use the blocked/tiled path (cache-efficient for large matrices).
		b.DotGeneralForceExecutionPath = int(BlockedPath)
	case "dotgeneral_check":
		// Run both normalized and blocked paths and compare outputs (for debugging).
		b.DotGeneralForceExecutionPath = int(CheckPath)
	case "dotgeneral_smallmatmul":
		// Force DotGeneral to use the SmallMatMul fast path (for small float32 matrices).
		b.DotGeneralForceExecutionPath = int(SmallMatMulPath)
	case "dotgeneral_packgemm":
		// Force DotGeneral to use the packgemm for large matmuls.
		b.EnablePackgemm = true
		b.DotGeneralForceExecutionPath = int(PackgemmPath)
	case "dotgeneral_highway":
		// Force DotGeneral to use the highway for large matmuls.
		b.EnableHighway = true
		b.DotGeneralForceExecutionPath = int(HighwayPath)
	default:
		return errors.Errorf("unknown configuration option %q for Go backend!? It shouldn't have been registered, please report an issue in GoMLX.", key)
	}
	return nil
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

	// Only for half-types inputs we always output Float32 for the "normalized" dot-general.
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

	// Select execution path at build time based on problem size and matrix layout.
	// This enables proper deduplication of pre-blocked inputs via getOrCreateNode.
	params.execPath = selectExecPath(f.RawBuilder.Backend, params)
	if params.execPath == SmallTransposedPath && params.Layout == LayoutNonTransposed {
		// The "SmallTransposedPath" takes as input LayoutTransposed, we are forced to transpose the RHS.
		klog.V(1).Info("DotGeneral selecte SmallTransposedPath, transposing to required LayoutTransposed")
		rhs, params.RHSContractingAxes, params.RHSBatchAxes, err = transposeSide(f, rhs, params.RHSContractingAxes, params.RHSBatchAxes, LayoutTransposed)
		if err != nil {
			return nil, err
		}
		params.Layout = LayoutTransposed
	}
	klog.V(1).Infof("DotGeneral execPath for %s layout: %s\n", params.Layout, params.execPath)

	// For blockedPath, pre-block BOTH inputs at graph-build time.
	// This allows deduplication: if the same tensor is used in multiple DotGenerals,
	// the blocking is done once and shared.
	var lhsBlocked, rhsBlocked *gobackend.Node
	if params.execPath == BlockedPath || params.execPath == CheckPath {
		params.SetBlockedParams()
		lhsBlocked = blockForDotGeneral(f, lhs, params.LHSContractingAxes, params.LHSBatchAxes,
			params.BatchSize, params.LHSCrossSize, params.ContractingSize)
		rhsBlocked = blockForDotGeneral(f, rhs, params.RHSContractingAxes, params.RHSBatchAxes,
			params.BatchSize, params.RHSCrossSize, params.ContractingSize)
	}

	// Create dot-general node: it will generate a normalized output [batchSize, lhsCrossSize, rhsCrossSize].
	var inputs []*gobackend.Node
	switch params.execPath {
	case BlockedPath:
		inputs = []*gobackend.Node{lhsBlocked, rhsBlocked}
	case CheckPath:
		// Include inputs in both forms.
		inputs = []*gobackend.Node{lhsBlocked, rhsBlocked, lhs, rhs}
	default:
		inputs = []*gobackend.Node{lhs, rhs}
	}

	// Create dot-general node: it will generate a normalized output [batchSize, lhsCrossSize, rhsCrossSize].
	normalizedOutputShape := shapes.Make(params.OutputDType, params.BatchSize, params.LHSCrossSize, params.RHSCrossSize)
	result, _ := f.GetOrCreateNode(compute.OpTypeDotGeneral, normalizedOutputShape, inputs, params)
	if result.Shape.Equal(outputShape) {
		// If no de-normalization is needed, return the result immediately.
		return result, nil
	}

	// Reshape result to recover batch and cross dimensions.
	if result.Shape.DType != outputShape.DType {
		// Requires final DType conversion:
		resultValue, err := f.ConvertDType(result, outputShape.DType)
		if err != nil {
			return nil, err
		}
		result = resultValue.(*gobackend.Node)
	}
	if !result.Shape.Equal(outputShape) {
		// Reshape to axes that may have been merged during layout normalization.
		resultValue, err := f.Reshape(result, outputShape.Dimensions...)
		if err != nil {
			return nil, err
		}
		result = resultValue.(*gobackend.Node)
	}
	return result, nil
}

// reshapeToSupportedLayout reshapes/transposes lhs and rhs to a layout
// supported by the underlying execution backends.
//
// It returns the updated lhs, rhs and params (same as the input, with fields updated).
// The params.Layout field will be set to the supported layout.
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

	// First attempt to merge axes with same function:
	lhs, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs, params.RHSContractingAxes, params.RHSBatchAxes, err = MergeAxes(
		f, lhs, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs, params.RHSContractingAxes, params.RHSBatchAxes)
	if err != nil {
		return nil, nil, nil, err
	}
	params.Layout = LayoutForDotGeneral(lhs.Shape, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs.Shape, params.RHSContractingAxes, params.RHSBatchAxes)
	if err != nil {
		return nil, nil, nil, err
	}
	if params.Layout != LayoutIncompatible {
		// Merged axes make a supported layout.
		return lhs, rhs, params, nil
	}

	// We need to transpose inputs to a supported layout. Since the
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

// ExecutionPath indicates which execution strategy to use for DotGeneral.
// Path selection happens at graph-build time in DotGeneral(), not at execution time.
type ExecutionPath int

const (
	// AutoSelectPath means the execution path should be auto-selected based on matrix size.
	// This is used only for backend.dotGeneralForceExecutionPath; never stored in params.execPath.
	AutoSelectPath ExecutionPath = iota
	// SmallTransposedPath uses the normalized transpose path (small matrices)
	SmallTransposedPath
	// BlockedPath uses execDotGeneralBlocked (cache-tiled algorithm, large matrices)
	BlockedPath
	// SmallMatMulPath uses the SmallMatMul fast path (small float32 matrices in standard order)
	SmallMatMulPath
	// PackgemmPath uses the packgemm package with a fast matmul algorithm with continuous packing of the matrices.
	PackgemmPath
	// HighwayPath uses the highway package (uses go-highway) with a fast matmul algorithm with continuous packing of the matrices.
	HighwayPath
	// CheckPath runs both paths and compares outputs (for debugging)
	CheckPath
)

//go:generate go tool enumer -type ExecutionPath -output=gen_execution_path_enumer.go dot.go

// selectExecPath selects the execution path based on problem size and backend configuration.
// Called at graph-build time from DotGeneral().
func selectExecPath(backend *gobackend.Backend, params *NodeData) ExecutionPath {
	// If a specific path is forced via backend config, use that.
	execPath := ExecutionPath(backend.DotGeneralForceExecutionPath)
	if execPath != AutoSelectPath {
		// Checks whether the forced path is valid for the given problem.
		switch {
		case execPath == SmallMatMulPath && params.Layout == LayoutNonTransposed:
			return execPath
		case execPath == PackgemmPath && params.Layout == LayoutNonTransposed && packgemm.HasDTypeSupport(params.InputDType, params.OutputDType):
			return execPath
		case execPath == HighwayPath && params.Layout == LayoutNonTransposed && highway.HasDTypeSupport(params.InputDType, params.OutputDType):
			return execPath
		case execPath == SmallTransposedPath && params.Layout == LayoutNonTransposed:
			return execPath
		case execPath == BlockedPath && params.Layout == LayoutNonTransposed:
			return execPath
		}
		klog.V(1).Infof(
			"DotGeneral: forced path %s is invalid for problem with input dtype %s, output dtype %s and layout (%s)\n",
			execPath, params.InputDType, params.OutputDType, params.Layout)
	}

	// GEMM path:
	if backend.EnablePackgemm && packgemm.HasDTypeSupport(params.InputDType, params.OutputDType) &&
		params.Layout == LayoutNonTransposed {
		return PackgemmPath
	}

	// Highway path:
	if backend.EnableHighway && highway.HasDTypeSupport(params.InputDType, params.OutputDType) &&
		params.Layout == LayoutNonTransposed {
		return HighwayPath
	}

	// Check for SmallMatMul fast path first.
	// SmallMatMul is beneficial for small float32 matrices in standard [M,K]×[K,N] order.
	if UseSmallMatMul(params) {
		return SmallMatMulPath
	}

	// Default selection based on problem size.
	// For large matrices, the blocked path with cache-tiled algorithm is more efficient.
	crossesSize := params.RHSCrossSize * params.LHSCrossSize
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[params.InputDType]
	blockSize := blockDim * blockDim
	if crossesSize > DotGeneralBlockedPathThreshold*blockSize {
		return BlockedPath
	}
	return SmallTransposedPath
}

// execDotGeneral executes the DotGeneral operation.
// The execution path is pre-selected at graph-build time and stored in params.execPath.
// For blockedPath, inputs are already pre-blocked at build time.
func execDotGeneral(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	params := node.Data.(*NodeData)
	outputShape := node.Shape
	output, err := backend.GetBuffer(outputShape)

	if err != nil {
		return nil, err
	}
	switch params.execPath {
	case BlockedPath, CheckPath:
		// Inputs are pre-blocked at graph-build time. Extract block metadata from input nodes.
		lhsNode := node.Inputs[0]
		rhsNode := node.Inputs[1]
		_, ok := lhsNode.Data.(*preBlockNodeData)
		if !ok {
			backend.PutBuffer(output)
			return nil, errors.Errorf("blockedPath requires pre-blocked LHS input, got %T (node type: %s)",
				lhsNode.Data, lhsNode.OpType)
		}
		rhsBlockData, ok := rhsNode.Data.(*preBlockNodeData)
		if !ok {
			backend.PutBuffer(output)
			return nil, errors.Errorf("blockedPath requires pre-blocked RHS input, got %T (node type: %s)",
				rhsNode.Data, rhsNode.OpType)
		}
		hasBatch := len(rhsBlockData.batchAxes) > 0 && rhsBlockData.batchSize > 1 // batchSize is the same for lhs and rhs
		err = execDotGeneralBlocked(backend, lhs, rhs, hasBatch, params, output)
		inputDType := lhs.RawShape.DType

		// Now run checks against other algorithms.
		if err == nil && params.execPath == CheckPath {
			// The "checkPath" is the debug path: it uses the blocked path as a reference and runs all other possible paths
			// comparing the results.
			lhsRaw, rhsRaw := inputs[2], inputs[3]
			output2, err := backend.GetBuffer(outputShape)
			if err != nil {
				return nil, err
			}
			output2.Zeros()
			err = execSmallTransposed(backend, lhsRaw, rhsRaw, params, output2)
			if err != nil {
				backend.PutBuffer(output2)
				backend.PutBuffer(output)
				return nil, err
			}
			err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
			if err != nil {
				backend.PutBuffer(output2)
				backend.PutBuffer(output)
				return nil, err
			}

			// Also verify SmallMatMul path for matrices in matmul order
			rawDType := lhsRaw.RawShape.DType
			if rawDType < gobackend.MaxDTypes && dotGeneralSmallMatMulDTypeMap.Map[rawDType] != nil &&
				IsMatMulOrder(lhsRaw.RawShape, params.LHSContractingAxes, params.LHSBatchAxes,
					rhsRaw.RawShape, params.RHSContractingAxes, params.RHSBatchAxes) {
				output2.Zeros()
				execSmallMatMulFnAny, err := dotGeneralSmallMatMulDTypeMap.Get(rawDType)
				if err != nil {
					return nil, err
				}
				execSmallMatMulFn := execSmallMatMulFnAny.(func(*gobackend.Backend, *gobackend.Buffer, *gobackend.Buffer, *NodeData, *gobackend.Buffer))
				// BFloat16/Float16 implementations accumulate in float32 internally but write to native output
				execSmallMatMulFn(backend, lhsRaw, rhsRaw, params, output2)
				err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				if err != nil {
					backend.PutBuffer(output2)
					backend.PutBuffer(output)
					return nil, err
				}
			}

			// GEMM specialized executor.
			if backend.EnablePackgemm && IsMatMulOrder(lhsRaw.RawShape, params.LHSContractingAxes, params.LHSBatchAxes,
				rhsRaw.RawShape, params.RHSContractingAxes, params.RHSBatchAxes) &&
				packgemm.HasDTypeSupport(inputDType, inputDType) {
				err = packgemm.GEMM(float32(1), float32(0), lhsRaw.Flat.([]float32), rhsRaw.Flat.([]float32),
					params.BatchSize, params.LHSCrossSize, params.RHSCrossSize, params.ContractingSize,
					output2.Flat.([]float32),
					getBufAllocator[float32](backend), getBufReleaser(backend), backend.Workers)
				if err == nil {
					err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				}
				if err != nil {
					backend.PutBuffer(output2)
					backend.PutBuffer(output)
					return nil, err
				}
			}

			// Highway MatMul specialized executor.
			if backend.EnableHighway && IsMatMulOrder(lhsRaw.RawShape, params.LHSContractingAxes, params.LHSBatchAxes,
				rhsRaw.RawShape, params.RHSContractingAxes, params.RHSBatchAxes) &&
				highway.HasDTypeSupport(inputDType, inputDType) {
				err = highway.MatMulDynamic(inputDType, outputShape.DType, lhsRaw.Flat, rhsRaw.Flat,
					params.BatchSize, params.LHSCrossSize, params.RHSCrossSize, params.ContractingSize,
					output2.Flat,
					getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.Workers)
				if err == nil {
					err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				}
				if err != nil {
					backend.PutBuffer(output2)
					backend.PutBuffer(output)
					return nil, err
				}
			}

			backend.PutBuffer(output2) // Discard second output, no longer needed
			return output, nil
		}

	case SmallMatMulPath:
		// SmallMatMul fast path: small matrices in standard [M,K]×[K,N] order.
		// Path was selected at build time based on matrix layout and size.
		// Supports all numeric dtypes via DTypeMap registration.
		// BFloat16/Float16 implementations accumulate in float32 internally but write to native output.
		dtype := lhs.RawShape.DType
		execSmallMatMulFnAny, err := dotGeneralSmallMatMulDTypeMap.Get(dtype)
		if err != nil {
			return nil, err
		}
		execSmallMatMulFn := execSmallMatMulFnAny.(func(*gobackend.Backend, *gobackend.Buffer, *gobackend.Buffer, *NodeData, *gobackend.Buffer))
		execSmallMatMulFn(backend, lhs, rhs, params, output)
		return output, nil

	case SmallTransposedPath:
		// Transpose-based normalized path for small matrices
		output.Zeros()
		err = execSmallTransposed(backend, lhs, rhs, params, output)

	case PackgemmPath:
		// Custom GEMM path for large "malmul" order.
		inputDType := lhs.RawShape.DType
		outputDType := output.RawShape.DType
		if err = packgemm.GEMMDynamic(inputDType, outputDType, 1, 0, lhs.Flat.([]float32), rhs.Flat.([]float32),
			params.BatchSize, params.LHSCrossSize, params.RHSCrossSize, params.ContractingSize,
			output.Flat.([]float32),
			getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.Workers); err != nil {
			return nil, err
		}
		return output, nil

	case HighwayPath:
		// Highway MatMul path for large "malmul" order.
		inputDType := lhs.RawShape.DType
		outputDType := output.RawShape.DType
		err = highway.MatMulDynamic(inputDType, outputDType, lhs.Flat, rhs.Flat,
			params.BatchSize, params.LHSCrossSize, params.RHSCrossSize, params.ContractingSize,
			output.Flat,
			getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.Workers)
		return output, nil

	default:
		err = errors.Errorf("unknown execution path %d for DotGeneral", params.execPath)
	}

	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	return output, nil
}

// log2int return the log2(x) for integer values, rounded down.
// Only defined for positive values.
func log2int(x int) int {
	return bits.Len(uint(x)) - 1
}

var dotGeneralVersionsCheckDelta = 1e-3

func dotGeneralCheckVersions(_ *gobackend.Backend, lhs, rhs *gobackend.Buffer, params *NodeData, outputLarge, outputSmall *gobackend.Buffer) error {
	if klog.V(1).Enabled() {
		var value0 float64
		dtype := outputLarge.RawShape.DType
		switch dtype {
		case dtypes.Float32:
			value0 = float64(outputLarge.Flat.([]float32)[0])
		case dtypes.Float64:
			value0 = outputLarge.Flat.([]float64)[0]
		case dtypes.BFloat16:
			value0 = float64(outputLarge.Flat.([]bfloat16.BFloat16)[0].Float32())
		}

		fmt.Printf("> %s x %s -> %s (output[...0]=%.5f)\n", lhs.RawShape, rhs.RawShape, outputLarge.RawShape, value0)
	}
	messages, err := dotGeneralCheckVersionsCmp(outputLarge, outputSmall)
	if err == nil {
		return nil
	}
	fmt.Printf("ERROR: dotGeneral check versions failed:\n")
	fmt.Printf("\t- lhs=%s, lhsContractingAxes=%v, lhsBatchAxes=%v\n",
		lhs.RawShape, params.LHSContractingAxes, params.LHSBatchAxes)
	fmt.Printf("\t- rhs=%s, rhsContractingAxes=%v, rhsBatchAxes=%v\n",
		rhs.RawShape, params.RHSContractingAxes, params.RHSBatchAxes)
	fmt.Printf("\t- batchSize=%d, lhsCrossSize=%d, rhsCrossAxes=%d, contractingSize=%d\n",
		params.BatchSize, params.LHSCrossSize, params.RHSCrossSize, params.ContractingSize)
	fmt.Printf("\t- output=%s\n", outputLarge.RawShape)
	fmt.Printf("%s\n", strings.Join(messages, "\n"))
	return err
}

func dotGeneralCheckVersionsCmp(outputLarge, outputSmall *gobackend.Buffer) (messages []string, err error) {
	// Make sure shapes are the same.
	if !outputLarge.RawShape.Equal(outputSmall.RawShape) {
		return nil, errors.Errorf("outputs have different shapes")
	}
	flatIdx := 0
	dtype := outputLarge.RawShape.DType
	var mismatches int
	switch dtype {
	case dtypes.Float32:
		largeFlat := outputLarge.Flat.([]float32)
		smallFlat := outputSmall.Flat.([]float32)
		for indices := range outputLarge.RawShape.Iter() {
			largeValue := largeFlat[flatIdx]
			smallValue := smallFlat[flatIdx]
			if math.Abs(float64(largeValue)-float64(smallValue)) > dotGeneralVersionsCheckDelta {
				if mismatches < 3 {
					messages = append(
						messages,
						fmt.Sprintf("\tDotGeneral: index %v (flatIdx=%d) has a mismatch on versions: large=%f, small=%f", indices, flatIdx, largeValue, smallValue))
				} else if mismatches == 4 {
					fmt.Printf("\t...")
				}
				mismatches++
			}
			flatIdx++
		}

	default:
		// Not checking other dtypes.
	}
	if mismatches > 0 {
		return messages, errors.Errorf(
			"found %d mismatches (out of %d values) between DotGeneral large and small versions", mismatches, outputLarge.RawShape.Size())
	}
	return
}

// getBufAllocator returns a buffer allocator for the given numeric type.
// TODO: change signature to return the error
func getBufAllocator[T dtypes.NumberNotComplex](backend *gobackend.Backend) packgemm.BufAllocFn[T] {
	dtype := dtypes.FromGenericsType[T]()
	return func(size int) (ref any, data []T) {
		buf, err := backend.GetBuffer(shapes.Make(dtype, size))
		if err != nil {
			return nil, nil
		}
		return buf, buf.Flat.([]T)
	}
}

// getAnyBufAllocator returns a buffer allocator for the given dtype.
// TODO: change signature to return the error
func getAnyBufAllocator(backend *gobackend.Backend, dtype dtypes.DType) packgemm.BufAllocAnyFn {
	return func(size int) (ref any, data any) {
		buf, err := backend.GetBuffer(shapes.Make(dtype, size))
		if err != nil {
			return nil, nil
		}
		return buf, buf.Flat
	}
}

// getBufReleaser returns a buffer releaser for the given numeric type.
func getBufReleaser(backend *gobackend.Backend) packgemm.BufReleaseFn {
	return func(ref any) {
		backend.PutBuffer(ref.(*gobackend.Buffer))
	}
}

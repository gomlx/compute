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
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support"
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

type NodeData struct {
	LHSContractingAxes, LHSBatchAxes                       []int
	RHSContractingAxes, RHSBatchAxes                       []int
	BatchSize, LHSCrossSize, RHSCrossSize, ContractingSize int
	LHSBlockedShape, RHSBlockedShape, OutputBlockedShape   shapes.Shape
	LHSNormalization, RHSNormalization                     *NormalizationInfo

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

func (d *NodeData) VerifyAndAdjust(lhsShape, rhsShape shapes.Shape) error {
	lhsRank := lhsShape.Rank()
	rhsRank := rhsShape.Rank()

	// Validate and adjust axes.
	var err error
	for ii, axis := range d.LHSContractingAxes {
		d.LHSContractingAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return errors.WithMessagef(err,
				"while adjusting contractingAxes for DotGeneral(lhs=%s, lhsContractingAxes=%v)",
				lhsShape, d.LHSContractingAxes)
		}
	}
	for ii, axis := range d.LHSBatchAxes {
		d.LHSBatchAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return errors.WithMessagef(err,
				"while adjusting batchAxes for DotGeneral(lhs=%s, lhsBatchAxes=%v)", lhsShape, d.LHSBatchAxes)
		}
	}
	for ii, axis := range d.RHSContractingAxes {
		d.RHSContractingAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return errors.WithMessagef(err,
				"while adjusting contractingAxes for DotGeneral(rhs=%s, rhsContractingAxes=%v)",
				rhsShape, d.RHSContractingAxes)
		}
	}
	for ii, axis := range d.RHSBatchAxes {
		d.RHSBatchAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return errors.WithMessagef(err,
				"while adjusting batchAxes for DotGeneral(rhs=%s, rhsBatchAxes=%v)", rhsShape, d.RHSBatchAxes)
		}
	}

	// Check that batch and contracting dimensions from lhs and rhs match.
	for ii, lhsAxis := range d.LHSContractingAxes {
		rhsAxis := d.RHSContractingAxes[ii]
		if lhsShape.Dimensions[lhsAxis] != rhsShape.Dimensions[rhsAxis] {
			return errors.Errorf("DotGeneral contracting dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhsShape.Dimensions[lhsAxis], rhsAxis, rhsShape.Dimensions[rhsAxis])
		}
	}
	for ii, lhsAxis := range d.LHSBatchAxes {
		rhsAxis := d.RHSBatchAxes[ii]
		if lhsShape.Dimensions[lhsAxis] != rhsShape.Dimensions[rhsAxis] {
			return errors.Errorf("DotGeneral batch dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhsShape.Dimensions[lhsAxis], rhsAxis, rhsShape.Dimensions[rhsAxis])
		}
	}
	return nil
}

// SetBackendOption process the configuration options for DotGeneral.
func SetBackendOption(b *gobackend.Backend, key string) error {
	switch key {
	case "dotgeneral_normalized":
		// Force DotGeneral to use the normalized path (transpose to [B,Cross,Contract] form).
		b.DotGeneralForceExecutionPath = int(NormalizedPath)
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
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// This function implements compute.Builder interface.
//
// This is the graph building part of DotGeneral. It first transposes the operands to a normalized
// shape with rank=3 ([batchSize, crossSize, contractingSize]), and then it issues the DotGeneral
// node with normalized inputs. Finally, it reshapes back to the final result.
//
// See execDotGeneral for the implementation.
func DotGeneral(f *gobackend.Function,
	lhsValue compute.Value, lhsContractingAxes, lhsBatchAxes []int,
	rhsValue compute.Value, rhsContractingAxes, rhsBatchAxes []int,
	config compute.DotGeneralConfig) (compute.Value, error) {

	directInputs, err := f.VerifyAndCastValues("DotGeneral", lhsValue, rhsValue)
	if err != nil {
		return nil, err
	}
	lhs, rhs := directInputs[0], directInputs[1]
	if klog.V(1).Enabled() {
		klog.Infof("DotGeneral lhs=%s rhs=%s contracting=%v,%v batch=%v,%v\n",
			lhs.Shape, rhs.Shape,
			lhsContractingAxes, rhsContractingAxes,
			lhsBatchAxes, rhsBatchAxes)
	}

	// Parse the inputs.
	dtype := lhs.Shape.DType
	if dtype != rhs.Shape.DType {
		return nil, errors.Errorf("DotGeneral lhs (left-hand-side) and rhs operands don't match data types: %s and %s",
			dtype, rhs.Shape.DType)
	}

	// We don't yet support accumulator dtype, so we convert the inputs.
	// - Exception: Half-Precision types automatically use Float32 for the computation.
	isHalfPrecisionWithFloat32 := dtype.IsHalfPrecision() && config.AccumulatorDType == dtypes.Float32
	if !isHalfPrecisionWithFloat32 && config.AccumulatorDType != dtypes.InvalidDType && config.AccumulatorDType != dtype {
		lhsOp, err := f.ConvertDType(lhs, config.AccumulatorDType)
		if err != nil {
			return nil, err
		}
		lhs = lhsOp.(*gobackend.Node)
		rhsOp, err := f.ConvertDType(rhs, config.AccumulatorDType)
		if err != nil {
			return nil, err
		}
		rhs = rhsOp.(*gobackend.Node)
		dtype = config.AccumulatorDType
	}

	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		return nil, errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return nil, errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}

	// Create node params, and verify axes have valid dimensions.
	params := NodeData{
		LHSContractingAxes: lhsContractingAxes,
		LHSBatchAxes:       lhsBatchAxes,
		RHSContractingAxes: rhsContractingAxes,
		RHSBatchAxes:       rhsBatchAxes,
	}
	err = params.VerifyAndAdjust(lhs.Shape, rhs.Shape)
	if err != nil {
		return nil, err
	}

	// Find sizes of the normalized operands ([batchSize, crossSize, contractSize]).
	// We do this before merging axes so we capture the original unmerged dimensions
	// which are needed to correctly reshape the final output.
	batchDims := make([]int, len(lhsBatchAxes))
	for ii, lhsAxis := range params.LHSBatchAxes {
		batchDims[ii] = lhs.Shape.Dimensions[lhsAxis]
	}
	var lhsCrossDims, rhsCrossDims []int
	params.BatchSize, params.LHSCrossSize, params.ContractingSize, lhsCrossDims = support.DotGeneralFindSizes(
		lhs.Shape, lhsContractingAxes, lhsBatchAxes)
	_, params.RHSCrossSize, _, rhsCrossDims = support.DotGeneralFindSizes(rhs.Shape, rhsContractingAxes, rhsBatchAxes)

	// Merge adjacent axes used for the same purpose to simplify the operation.
	// This makes the physical memory layout simpler and allows matching to
	// fast paths (like SmallMatMul or Packgemm) more often.
	var revertFn revertMergeAxesFunc
	lhs, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs, params.RHSContractingAxes, params.RHSBatchAxes, revertFn, err = MergeAxes(
		f, lhs, params.LHSContractingAxes, params.LHSBatchAxes,
		rhs, params.RHSContractingAxes, params.RHSBatchAxes)
	if err != nil {
		return nil, err
	}

	// Check that all sizes are positive
	if params.BatchSize <= 0 || params.LHSCrossSize <= 0 || params.ContractingSize <= 0 || params.RHSCrossSize <= 0 {
		return nil, errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(cross=%d)",
			params.BatchSize, params.LHSCrossSize, params.ContractingSize,
			params.RHSCrossSize)
	}

	params.LHSNormalization = NormalizePrepare(lhs.Shape, params.LHSContractingAxes, params.LHSBatchAxes)
	params.RHSNormalization = NormalizePrepare(rhs.Shape, params.RHSContractingAxes, params.RHSBatchAxes)

	outputDType := dtype
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		// For 16 bits, store the intermediary results as float32 to minimize numerical errors during accumulation.
		// Notice the blockLog2Dim must be the same, because the block dimensions much match the inputs.
		outputDType = dtypes.Float32
	}

	// Select execution path at build time based on problem size and matrix layout.
	// This enables proper deduplication of pre-blocked inputs via getOrCreateNode.
	params.execPath = selectExecPath(f.Builder.Backend, lhs.Shape, rhs.Shape, &params)
	klog.V(1).Infof("DotGeneral execPath: %s\n", params.execPath)

	// For blockedPath, pre-block BOTH inputs at graph-build time.
	// This allows deduplication: if the same tensor is used in multiple DotGenerals,
	// the blocking is done once and shared.
	var lhsBlocked, rhsBlocked *gobackend.Node
	if params.execPath == BlockedPath || params.execPath == CheckPath {
		params.SetBlockedParams(dtype, outputDType)
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
	dotGeneral, _ := f.GetOrCreateNode(compute.OpTypeDotGeneral, shapes.Make(dtype, params.BatchSize, params.LHSCrossSize, params.RHSCrossSize), inputs, &params)

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := f.Reshape(dotGeneral, resultingDims...)
	if err != nil {
		return nil, err
	}

	// If config.OutputDType is different than the input, for now we simply convert it.
	if config.OutputDType != dtypes.InvalidDType && config.OutputDType != dtype {
		result, err = f.ConvertDType(result, config.OutputDType)
		if err != nil {
			return nil, err
		}
	}

	result, err = revertFn(result.(*gobackend.Node))
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ExecutionPath indicates which execution strategy to use for DotGeneral.
// Path selection happens at graph-build time in DotGeneral(), not at execution time.
type ExecutionPath int

const (
	// AutoSelectPath means the execution path should be auto-selected based on matrix size.
	// This is used only for backend.dotGeneralForceExecutionPath; never stored in params.execPath.
	AutoSelectPath ExecutionPath = iota
	// NormalizedPath uses the normalized transpose path (small matrices)
	NormalizedPath
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
func selectExecPath(backend *gobackend.Backend, lhsShape, rhsShape shapes.Shape, params *NodeData) ExecutionPath {
	dtype := lhsShape.DType
	outputDType := dtype
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		// For 16 bits, store the intermediary results as float32 to minimize numerical errors during accumulation.
		// Notice the blockLog2Dim must be the same, because the block dimensions much match the inputs.
		outputDType = dtypes.Float32
	}

	// If a specific path is forced via backend config, use that.
	execPath := ExecutionPath(backend.DotGeneralForceExecutionPath)
	if execPath != AutoSelectPath {
		// Checks whether the forced path is valid for the given problem.
		var valid bool
		switch execPath {
		case SmallMatMulPath:
			valid = IsMatMulOrder(lhsShape, params.LHSContractingAxes, params.LHSBatchAxes,
				rhsShape, params.RHSContractingAxes, params.RHSBatchAxes)
		case PackgemmPath:
			valid = backend.EnablePackgemm && packgemm.HasDTypeSupport(dtype, outputDType) &&
				IsMatMulOrder(lhsShape, params.LHSContractingAxes, params.LHSBatchAxes,
					rhsShape, params.RHSContractingAxes, params.RHSBatchAxes)
		case HighwayPath:
			valid = backend.EnableHighway && highway.HasDTypeSupport(dtype, outputDType) &&
				IsMatMulOrder(lhsShape, params.LHSContractingAxes, params.LHSBatchAxes,
					rhsShape, params.RHSContractingAxes, params.RHSBatchAxes)
		default:
			valid = true
		}
		if valid {
			return execPath
		}
		klog.V(1).Infof("DotGeneral: forced path %s is invalid for problem dtype or axes order %s×%s\n", execPath, lhsShape, rhsShape)
	}

	// GEMM path:
	if backend.EnablePackgemm && packgemm.HasDTypeSupport(dtype, outputDType) &&
		IsMatMulOrder(lhsShape, params.LHSContractingAxes, params.LHSBatchAxes,
			rhsShape, params.RHSContractingAxes, params.RHSBatchAxes) {
		return PackgemmPath
	}

	// Highway path:
	if backend.EnableHighway && highway.HasDTypeSupport(dtype, outputDType) &&
		IsMatMulOrder(lhsShape, params.LHSContractingAxes, params.LHSBatchAxes,
			rhsShape, params.RHSContractingAxes, params.RHSBatchAxes) {
		return HighwayPath
	}

	// Check for SmallMatMul fast path first.
	// SmallMatMul is beneficial for small float32 matrices in standard [M,K]×[K,N] order.
	if UseSmallMatMul(dtype, lhsShape, rhsShape, params) {
		return SmallMatMulPath
	}

	// Default selection based on problem size.
	// For large matrices, the blocked path with cache-tiled algorithm is more efficient.
	crossesSize := params.RHSCrossSize * params.LHSCrossSize
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtype]
	blockSize := blockDim * blockDim
	if crossesSize > DotGeneralBlockedPathThreshold*blockSize {
		return BlockedPath
	}
	return NormalizedPath
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
			err = execDotGeneralNormalized(backend, lhsRaw, rhsRaw, params, output2)
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

	case NormalizedPath:
		// Transpose-based normalized path for small matrices
		output.Zeros()
		err = execDotGeneralNormalized(backend, lhs, rhs, params, output)

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

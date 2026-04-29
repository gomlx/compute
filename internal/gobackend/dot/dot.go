// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

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
	"github.com/gomlx/compute/internal/gobackend/highway"
	"github.com/gomlx/compute/internal/gobackend/packgemm"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeDotGeneral, gobackend.PriorityGeneric, execDotGeneral)
}

type dotGeneralNodeData struct {
	lhsContractingAxes, lhsBatchAxes                       []int
	rhsContractingAxes, rhsBatchAxes                       []int
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int
	lhsBlockedShape, rhsBlockedShape, outputBlockedShape   shapes.Shape
	lhsNormalization, rhsNormalization                     *dgNormalizationInfo

	// execPath determines which execution strategy to use. Decided at graph-build time.
	execPath ExecutionPath
}

// EqualNodeData implements nodeDataComparable for dotGeneralNodeData.
func (d *dotGeneralNodeData) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*dotGeneralNodeData)
	if d.batchSize != o.batchSize ||
		d.lhsCrossSize != o.lhsCrossSize ||
		d.rhsCrossSize != o.rhsCrossSize ||
		d.contractingSize != o.contractingSize ||
		d.execPath != o.execPath {
		return false
	}
	return slices.Equal(d.lhsContractingAxes, o.lhsContractingAxes) &&
		slices.Equal(d.lhsBatchAxes, o.lhsBatchAxes) &&
		slices.Equal(d.rhsContractingAxes, o.rhsContractingAxes) &&
		slices.Equal(d.rhsBatchAxes, o.rhsBatchAxes) &&
		d.lhsBlockedShape.Equal(o.lhsBlockedShape) &&
		d.rhsBlockedShape.Equal(o.rhsBlockedShape) &&
		d.outputBlockedShape.Equal(o.outputBlockedShape)
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

func init() {
	// Register DotGeneral config options for the backend.
	for _, option := range []string{"dotgeneral_normalized", "dotgeneral_blocked", "dotgeneral_check", "dotgeneral_smallmatmul", "dotgeneral_packgemm", "dotgeneral_highway"} {
		gobackend.KnownOptionsSetters[option] = SetBackendOption
	}

	// Register DotGeneral handler.
	gobackend.RegisterDotGeneral.Register(DotGeneral, gobackend.PriorityGeneric)
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
	lhs *gobackend.Node, lhsContractingAxes, lhsBatchAxes []int,
	rhs *gobackend.Node, rhsContractingAxes, rhsBatchAxes []int,
	config compute.DotGeneralConfig) (*gobackend.Node, error) {
	// Parse the inputs.
	dtype := lhs.Shape.DType
	if dtype != rhs.Shape.DType {
		return nil, errors.Errorf("DotGeneral lhs (left-hand-side) and rhs operands don't match data types: %s and %s",
			dtype, rhs.Shape.DType)
	}

	// We don't yet support accumulator dtype, so we convert the inputs.
	// - Exception: Half-Precision types automatically use Float32 for the computation.
	isHalfPrecisionWithFloat32 := dtype.IsHalfPrecision() && config.AccumulatorDType == dtypes.Float32
	var err error
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

	lhsRank := lhs.Shape.Rank()
	rhsRank := rhs.Shape.Rank()
	params := dotGeneralNodeData{
		lhsContractingAxes: lhsContractingAxes,
		lhsBatchAxes:       lhsBatchAxes,
		rhsContractingAxes: rhsContractingAxes,
		rhsBatchAxes:       rhsBatchAxes,
	}

	// Validate and adjust axes.
	for ii, axis := range lhsContractingAxes {
		params.lhsContractingAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting contractingAxes for DotGeneral(lhs=%s, lhsContractingAxes=%v)",
				lhs.Shape, lhsContractingAxes)
		}
	}
	for ii, axis := range lhsBatchAxes {
		params.lhsBatchAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting batchAxes for DotGeneral(lhs=%s, lhsBatchAxes=%v)", lhs.Shape, lhsBatchAxes)
		}
	}
	for ii, axis := range rhsContractingAxes {
		params.rhsContractingAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting contractingAxes for DotGeneral(rhs=%s, rhsContractingAxes=%v)",
				rhs.Shape, rhsContractingAxes)
		}
	}
	for ii, axis := range rhsBatchAxes {
		params.rhsBatchAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting batchAxes for DotGeneral(rhs=%s, rhsBatchAxes=%v)", rhs.Shape, rhsBatchAxes)
		}
	}

	// Check that batch and contracting dimensions from lhs and rhs match.
	batchDims := make([]int, len(lhsBatchAxes))
	contractingDims := make([]int, len(lhsContractingAxes))
	for ii, lhsAxis := range params.lhsContractingAxes {
		rhsAxis := params.rhsContractingAxes[ii]
		if lhs.Shape.Dimensions[lhsAxis] != rhs.Shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral contracting dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.Shape.Dimensions[lhsAxis], rhsAxis, rhs.Shape.Dimensions[rhsAxis])
		}
		contractingDims[ii] = lhs.Shape.Dimensions[lhsAxis]
	}
	for ii, lhsAxis := range params.lhsBatchAxes {
		rhsAxis := params.rhsBatchAxes[ii]
		if lhs.Shape.Dimensions[lhsAxis] != rhs.Shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral batch dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.Shape.Dimensions[lhsAxis], rhsAxis, rhs.Shape.Dimensions[rhsAxis])
		}
		batchDims[ii] = lhs.Shape.Dimensions[lhsAxis]
	}

	// Find sizes of the normalized operands ([batchSize, crossSize, contractSize]).
	var lhsCrossDims, rhsCrossDims []int
	params.batchSize, params.lhsCrossSize, params.contractingSize, lhsCrossDims = support.DotGeneralFindSizes(
		lhs.Shape, lhsContractingAxes, lhsBatchAxes)
	_, params.rhsCrossSize, _, rhsCrossDims = support.DotGeneralFindSizes(rhs.Shape, rhsContractingAxes, rhsBatchAxes)

	// Check that all sizes are positive
	if params.batchSize <= 0 || params.lhsCrossSize <= 0 || params.contractingSize <= 0 || params.rhsCrossSize <= 0 {
		return nil, errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(cross=%d)",
			params.batchSize, params.lhsCrossSize, params.contractingSize,
			params.rhsCrossSize)
	}

	params.lhsNormalization = dgNormalizePrepare(lhs.Shape, params.lhsContractingAxes, params.lhsBatchAxes)
	params.rhsNormalization = dgNormalizePrepare(rhs.Shape, params.rhsContractingAxes, params.rhsBatchAxes)

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	params.lhsBlockedShape = CreateBlockedShape(
		dtype, params.batchSize, params.lhsCrossSize, params.contractingSize, blockLog2Dim)
	params.rhsBlockedShape = CreateBlockedShape(
		dtype, params.batchSize, params.rhsCrossSize, params.contractingSize, blockLog2Dim)
	outputDType := dtype
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		// For 16 bits, store the intermediary results as float32 to minimize numerical errors during accumulation.
		// Notice the blockLog2Dim must be the same, because the block dimensions much match the inputs.
		outputDType = dtypes.Float32
	}
	params.outputBlockedShape = CreateBlockedShape(
		outputDType, params.batchSize, params.lhsCrossSize, params.rhsCrossSize, blockLog2Dim)

	// Select execution path at build time based on problem size and matrix layout.
	// This enables proper deduplication of pre-blocked inputs via getOrCreateNode.
	params.execPath = selectExecPath(f.Builder.Backend, lhs.Shape, rhs.Shape, &params)
	klog.V(1).Infof("DotGeneral execPath: %s\n", params.execPath)

	// For blockedPath, pre-block BOTH inputs at graph-build time.
	// This allows deduplication: if the same tensor is used in multiple DotGenerals,
	// the blocking is done once and shared.
	var lhsBlocked, rhsBlocked *gobackend.Node
	if params.execPath == BlockedPath || params.execPath == CheckPath {
		lhsBlocked = blockForDotGeneral(f, lhs, params.lhsContractingAxes, params.lhsBatchAxes,
			params.batchSize, params.lhsCrossSize, params.contractingSize)
		rhsBlocked = blockForDotGeneral(f, rhs, params.rhsContractingAxes, params.rhsBatchAxes,
			params.batchSize, params.rhsCrossSize, params.contractingSize)
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
	dotGeneral, _ := f.GetOrCreateNode(compute.OpTypeDotGeneral, shapes.Make(dtype, params.batchSize, params.lhsCrossSize, params.rhsCrossSize), inputs, &params)

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
	return result.(*gobackend.Node), nil
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
func selectExecPath(backend *gobackend.Backend, lhsShape, rhsShape shapes.Shape, params *dotGeneralNodeData) ExecutionPath {
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
			valid = isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
				rhsShape, params.rhsContractingAxes, params.rhsBatchAxes)
		case PackgemmPath:
			valid = backend.EnablePackgemm && packgemm.HasDTypeSupport(dtype, outputDType) &&
				isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
					rhsShape, params.rhsContractingAxes, params.rhsBatchAxes)
		case HighwayPath:
			valid = backend.EnableHighway && highway.HasDTypeSupport(dtype, outputDType) &&
				isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
					rhsShape, params.rhsContractingAxes, params.rhsBatchAxes)
		default:
			valid = true
		}
		if valid {
			return execPath
		}
		klog.V(1).Infof("DotGeneral: forced path %s is invalid for problem size %s×%s\n", execPath, lhsShape, rhsShape)
	}

	// GEMM path:
	if backend.EnablePackgemm && packgemm.HasDTypeSupport(dtype, outputDType) &&
		isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
			rhsShape, params.rhsContractingAxes, params.rhsBatchAxes) {
		return PackgemmPath
	}

	// Highway path:
	if backend.EnableHighway && highway.HasDTypeSupport(dtype, outputDType) &&
		isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
			rhsShape, params.rhsContractingAxes, params.rhsBatchAxes) {
		return HighwayPath
	}

	// Check for SmallMatMul fast path first.
	// SmallMatMul is beneficial for small float32 matrices in standard [M,K]×[K,N] order.
	if useSmallMatMul(dtype, lhsShape, rhsShape, params) {
		return SmallMatMulPath
	}

	// Default selection based on problem size.
	// For large matrices, the blocked path with cache-tiled algorithm is more efficient.
	crossesSize := params.rhsCrossSize * params.lhsCrossSize
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
	params := node.Data.(*dotGeneralNodeData)
	outputShape := node.Shape
	output, err := backend.GetBufferForShape(outputShape)
	if err != nil {
		return nil, err
	}
	switch params.execPath {
	case BlockedPath, CheckPath:
		// Inputs are pre-blocked at graph-build time. Extract block metadata from input nodes.
		lhsNode := node.Inputs[0]
		rhsNode := node.Inputs[1]
		_, ok := lhsNode.Data.(*blockForDotGeneralData)
		if !ok {
			backend.putBuffer(output)
			return nil, errors.Errorf("blockedPath requires pre-blocked LHS input, got %T (node type: %s)",
				lhsNode.Data, lhsNode.OpType)
		}
		rhsBlockData, ok := rhsNode.Data.(*blockForDotGeneralData)
		if !ok {
			backend.putBuffer(output)
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
			output2, err := backend.getBufferForShape(outputShape)
			if err != nil {
				return nil, err
			}
			output2.Zeros()
			err = execDotGeneralNormalized(backend, lhsRaw, rhsRaw, params, output2)
			if err != nil {
				backend.putBuffer(output2)
				backend.putBuffer(output)
				return nil, err
			}
			err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
			if err != nil {
				backend.putBuffer(output2)
				backend.putBuffer(output)
				return nil, err
			}

			// Also verify SmallMatMul path for matrices in matmul order
			rawDType := lhsRaw.RawShape.DType
			if rawDType < MaxDTypes && dotGeneralSmallMatMulDTypeMap.Map[rawDType] != nil &&
				isMatMulOrder(lhsRaw.RawShape, params.lhsContractingAxes, params.lhsBatchAxes,
					rhsRaw.RawShape, params.rhsContractingAxes, params.rhsBatchAxes) {
				output2.Zeros()
				execSmallMatMulFnAny, err := dotGeneralSmallMatMulDTypeMap.Get(rawDType)
				if err != nil {
					return nil, err
				}
				execSmallMatMulFn := execSmallMatMulFnAny.(func(*gobackend.Backend, *gobackend.Buffer, *gobackend.Buffer, *dotGeneralNodeData, *gobackend.Buffer))
				// BFloat16/Float16 implementations accumulate in float32 internally but write to native output
				execSmallMatMulFn(backend, lhsRaw, rhsRaw, params, output2)
				err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				if err != nil {
					backend.putBuffer(output2)
					backend.putBuffer(output)
					return nil, err
				}
			}

			// GEMM specialized executor.
			if backend.enablePackgemm && isMatMulOrder(lhsRaw.RawShape, params.lhsContractingAxes, params.lhsBatchAxes,
				rhsRaw.RawShape, params.rhsContractingAxes, params.rhsBatchAxes) &&
				packgemm.HasDTypeSupport(inputDType, inputDType) {
				err = packgemm.GEMM(float32(1), float32(0), lhsRaw.Flat.([]float32), rhsRaw.Flat.([]float32),
					params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
					output2.Flat.([]float32),
					getBufAllocator[float32](backend), getBufReleaser(backend), backend.workers)
				if err == nil {
					err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				}
				if err != nil {
					backend.putBuffer(output2)
					backend.putBuffer(output)
					return nil, err
				}
			}

			// Highway MatMul specialized executor.
			if backend.enableHighway && isMatMulOrder(lhsRaw.RawShape, params.lhsContractingAxes, params.lhsBatchAxes,
				rhsRaw.RawShape, params.rhsContractingAxes, params.rhsBatchAxes) &&
				highway.HasDTypeSupport(inputDType, inputDType) {
				err = highway.MatMulDynamic(inputDType, outputShape.DType, lhsRaw.Flat, rhsRaw.Flat,
					params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
					output2.Flat,
					getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers)
				if err == nil {
					err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				}
				if err != nil {
					backend.putBuffer(output2)
					backend.putBuffer(output)
					return nil, err
				}
			}

			backend.putBuffer(output2) // Discard second output, no longer needed
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
		execSmallMatMulFn := execSmallMatMulFnAny.(func(*gobackend.Backend, *gobackend.Buffer, *gobackend.Buffer, *dotGeneralNodeData, *gobackend.Buffer))
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
			params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
			output.Flat.([]float32),
			getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers); err != nil {
			return nil, err
		}
		return output, nil

	case HighwayPath:
		// Highway MatMul path for large "malmul" order.
		inputDType := lhs.RawShape.DType
		outputDType := output.RawShape.DType
		err = highway.MatMulDynamic(inputDType, outputDType, lhs.Flat, rhs.Flat,
			params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
			output.Flat,
			getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers)
		return output, nil

	default:
		err = errors.Errorf("unknown execution path %d for DotGeneral", params.execPath)
	}

	if err != nil {
		backend.putBuffer(output)
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

func dotGeneralCheckVersions(_ *gobackend.Backend, lhs, rhs *gobackend.Buffer, params *dotGeneralNodeData, outputLarge, outputSmall *gobackend.Buffer) error {
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
		lhs.RawShape, params.lhsContractingAxes, params.lhsBatchAxes)
	fmt.Printf("\t- rhs=%s, rhsContractingAxes=%v, rhsBatchAxes=%v\n",
		rhs.RawShape, params.rhsContractingAxes, params.rhsBatchAxes)
	fmt.Printf("\t- batchSize=%d, lhsCrossSize=%d, rhsCrossAxes=%d, contractingSize=%d\n",
		params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize)
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
		buf, err := backend.GetBuffer(dtype, size)
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
		buf, err := backend.GetBuffer(dtype, size)
		if err != nil {
			return nil, nil
		}
		return buf, buf.Flat
	}
}

// getBufReleaser returns a buffer releaser for the given numeric type.
func getBufReleaser(backend *gobackend.Backend) packgemm.BufReleaseFn {
	return func(ref any) {
		backend.putBuffer(ref.(*gobackend.Buffer))
	}
}

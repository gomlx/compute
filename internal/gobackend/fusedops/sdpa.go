package fusedops

import (
	"math"
	"sync/atomic"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/fastmath"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// FusedScaledDotProductAttention computes multi-head scaled dot-product attention.
//
// output = softmax(query @ key^T * scale + mask) @ value, computed per-head with GQA support.
//
// Inputs:
//   - query, key, value: 4D tensors whose axis ordering is determined by axesLayout.
//     For AttentionAxesLayoutBHSD: query [batch, numHeads, seqLen, headDim],
//     key/value [batch, numKVHeads, kvLen, headDim].
//     For AttentionAxesLayoutBSHD: query [batch, seqLen, numHeads, headDim],
//     key/value [batch, kvLen, numKVHeads, headDim].
//   - mask: [seqLen, kvLen] (seqLen is the query sequence length): optional (can be nil) mask
//     that can be either boolean or additive (any dtype other than Bool). See also causal below.
//     Boolean mask: true = attend, false = ignore.
//     Float/additive mask: added to scores before softmax.
//     Must be broadcastable to the score tensor shape.
//
// Parameters:
//   - numHeads: number of query attention heads
//   - numKVHeads: number of key/value attention heads (for GQA; numHeads must be divisible by numKVHeads)
//   - axesLayout: determines the axis ordering of query/key/value tensors
//   - scale: scaling factor applied to query @ key^T (typically 1/sqrt(headDim))
//   - causal: if true, apply causal (lower-triangular) mask. Callers (e.g. attention.Core)
//     treat causal and mask as mutually exclusive, folding causal into the mask before calling
//     this method when both are needed. Backends may assume they won't both be set.
//   - options: optional optimization hints (nil uses defaults). See ScaledDotProductAttentionConfig.
//
// Output: same shape as query.
func FusedScaledDotProductAttention(
	f *gobackend.Function,
	query, key, value compute.Value,
	axesLayout compute.AttentionAxesLayout,
	options *compute.ScaledDotProductAttentionConfig) (output compute.Value, statesForVJP []compute.Value, err error) {
	// The Go backend has no fused flash backward, so it returns nil statesForVJP; its gradient
	// goes through the decomposed path (FusedScaledDotProductAttentionVJP is ErrNotImplemented).
	out, err := buildSDPANode(f, compute.OpTypeFusedScaledDotProductAttention, "FusedScaledDotProductAttention",
		query, key, value, axesLayout, options)
	return out, nil, err
}

func init() {
	gobackend.RegisterFusedScaledDotProductAttention.Register(FusedScaledDotProductAttention, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedScaledDotProductAttention, gobackend.PriorityTyped, execFusedScaledDotProductAttention)
}

type nodeScaledDotProductAttention struct {
	numHeads      int
	numKVHeads    int
	axesLayout    compute.AttentionAxesLayout
	scale         float64
	causal        bool
	hasMask       bool
	hasSeqLens    bool
	hasBias       bool
	options       *compute.ScaledDotProductAttentionConfig
	maskConstData *gobackend.ConstantData
	biasConstData *gobackend.ConstantData
}

func (d *nodeScaledDotProductAttention) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeScaledDotProductAttention)
	return d.numHeads == o.numHeads && d.numKVHeads == o.numKVHeads &&
		d.axesLayout == o.axesLayout && d.scale == o.scale && d.causal == o.causal &&
		d.hasMask == o.hasMask && d.hasSeqLens == o.hasSeqLens && d.hasBias == o.hasBias &&
		d.equalOptions(o)
}

func (d *nodeScaledDotProductAttention) equalOptions(o *nodeScaledDotProductAttention) bool {
	if d.options == nil && o.options == nil {
		return true
	}
	if d.options == nil || o.options == nil {
		return false
	}
	a, b := d.options, o.options
	// QuerySeqLen and KeyValueSeqLen are Value (any) and may hold non-comparable types,
	// so comparing them with == would panic. They are already passed as input operands
	// in buildSDPANode, so node dedup via input identity covers them; skip here.
	return a.QuantizedMatmuls == b.QuantizedMatmuls
}

// buildSDPANode builds the SDPA computation node.
func buildSDPANode(
	f *gobackend.Function, opType compute.OpType, opName string,
	query, key, value compute.Value,
	axesLayout compute.AttentionAxesLayout,
	options *compute.ScaledDotProductAttentionConfig) (compute.Value, error) {
	var mask compute.Value
	if options != nil {
		mask = options.Mask
	}
	values := []compute.Value{query, key, value}
	if mask != nil {
		values = append(values, mask)
	}

	if options != nil && (options.QuerySeqLen != nil) != (options.KeyValueSeqLen != nil) {
		return nil, errors.Errorf("%s: QuerySeqLen and KeyValueSeqLen must both be set or both nil", opName)
	}
	hasSeqLens := options != nil && options.QuerySeqLen != nil && options.KeyValueSeqLen != nil
	if hasSeqLens {
		values = append(values, options.QuerySeqLen, options.KeyValueSeqLen)
	}
	hasBias := options != nil && options.Bias != nil
	if hasBias {
		values = append(values, options.Bias)
	}

	inputs, err := f.VerifyAndCastValues(opName, values...)
	if err != nil {
		return nil, err
	}
	qNode := inputs[0]
	kNode := inputs[1]

	if qNode.Shape.Rank() != 4 {
		return nil, errors.Errorf("%s: query must have rank 4, got %d", opName, qNode.Shape.Rank())
	}
	if kNode.Shape.Rank() != 4 {
		return nil, errors.Errorf("%s: key must have rank 4, got %d", opName, kNode.Shape.Rank())
	}
	switch qNode.Shape.DType {
	case dtypes.Float32, dtypes.Float64:
		// Supported in go backend.
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"%s: dtype %s is not implemented in the go backend", opName, qNode.Shape.DType)
	}

	numHeads := qNode.Shape.Dimensions[axesLayout.HeadsAxis()]
	numKVHeads := kNode.Shape.Dimensions[axesLayout.HeadsAxis()]

	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		return nil, errors.Errorf("%s: numHeads (%d) must be positive and divisible by numKVHeads (%d)", opName, numHeads, numKVHeads)
	}

	var scale float64
	var causal bool
	if options != nil {
		scale = options.Scale
		causal = options.Causal
	}
	if scale == 0 {
		headDim := qNode.Shape.Dimensions[3] // headDim is always the last axis!
		scale = 1.0 / math.Sqrt(float64(headDim))
	}

	var batchDim, numHeadsDim, seqDim, kvDim int
	var batchName, numHeadsName, seqName, kvName string
	if axesLayout == compute.AttentionAxesLayoutBSHD {
		batchDim = qNode.Shape.Dimensions[0]
		batchName = qNode.Shape.AxisName(0)
		numHeadsDim = numHeads
		numHeadsName = qNode.Shape.AxisName(2)
		seqDim = qNode.Shape.Dimensions[1]
		seqName = qNode.Shape.AxisName(1)
		kvDim = kNode.Shape.Dimensions[1]
		kvName = kNode.Shape.AxisName(1)
	} else {
		// BHSD: [batch, heads, seq, headDim]
		batchDim = qNode.Shape.Dimensions[0]
		batchName = qNode.Shape.AxisName(0)
		numHeadsDim = numHeads
		numHeadsName = qNode.Shape.AxisName(1)
		seqDim = qNode.Shape.Dimensions[2]
		seqName = qNode.Shape.AxisName(2)
		kvDim = kNode.Shape.Dimensions[2]
		kvName = kNode.Shape.AxisName(2)
	}

	if mask != nil {
		maskNode := inputs[3]
		if maskNode.Shape.DType != dtypes.Bool && maskNode.Shape.DType != qNode.Shape.DType {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"%s: mask dtype %s is not supported in the go backend (expected Bool or %s)",
				opName, maskNode.Shape.DType, qNode.Shape.DType)
		}
		maskRank := maskNode.Shape.Rank()
		if maskRank < 2 || maskRank > 4 {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"%s: mask rank %d is not supported in the go backend (expected rank 2, 3, or 4)",
				opName, maskRank)
		}

		var mBatchDim, mHeadsDim, mSeqDim, mKVDim int
		var mBatchName, mHeadsName, mSeqName, mKVName string

		switch maskRank {
		case 4:
			if axesLayout == compute.AttentionAxesLayoutBSHD {
				mBatchDim = maskNode.Shape.Dimensions[0]
				mBatchName = maskNode.Shape.AxisName(0)
				mSeqDim = maskNode.Shape.Dimensions[1]
				mSeqName = maskNode.Shape.AxisName(1)
				mHeadsDim = maskNode.Shape.Dimensions[2]
				mHeadsName = maskNode.Shape.AxisName(2)
				mKVDim = maskNode.Shape.Dimensions[3]
				mKVName = maskNode.Shape.AxisName(3)
			} else {
				mBatchDim = maskNode.Shape.Dimensions[0]
				mBatchName = maskNode.Shape.AxisName(0)
				mHeadsDim = maskNode.Shape.Dimensions[1]
				mHeadsName = maskNode.Shape.AxisName(1)
				mSeqDim = maskNode.Shape.Dimensions[2]
				mSeqName = maskNode.Shape.AxisName(2)
				mKVDim = maskNode.Shape.Dimensions[3]
				mKVName = maskNode.Shape.AxisName(3)
			}
		case 3:
			mBatchDim = maskNode.Shape.Dimensions[0]
			mBatchName = maskNode.Shape.AxisName(0)
			mHeadsDim = 1
			mSeqDim = maskNode.Shape.Dimensions[1]
			mSeqName = maskNode.Shape.AxisName(1)
			mKVDim = maskNode.Shape.Dimensions[2]
			mKVName = maskNode.Shape.AxisName(2)
		case 2:
			mBatchDim = 1
			mHeadsDim = 1
			mSeqDim = maskNode.Shape.Dimensions[0]
			mSeqName = maskNode.Shape.AxisName(0)
			mKVDim = maskNode.Shape.Dimensions[1]
			mKVName = maskNode.Shape.AxisName(1)
			if mSeqName != "" && mSeqName == batchName {
				return nil, errors.Wrapf(compute.ErrNotImplemented,
					"%s: 2D mask has batch axis %q instead of query sequence axis", opName, mSeqName)
			}
		}

		if !isSDPABatchOrHeadCompatible(mBatchDim, mBatchName, batchDim, batchName) ||
			!isSDPABatchOrHeadCompatible(mHeadsDim, mHeadsName, numHeadsDim, numHeadsName) ||
			!isSDPABatchOrHeadCompatible(mSeqDim, mSeqName, seqDim, seqName) ||
			!isSDPASeqDimCompatible(mKVDim, mKVName, kvDim, kvName) {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"%s: mask shape %s is not supported by go backend fused SDPA (requires [batch, heads, %d, %d])",
				opName, maskNode.Shape, seqDim, kvDim)
		}
	}

	if hasBias {
		// Bias input is the last appended value; resolve its node from inputs.
		biasIdx := len(inputs) - 1
		biasNode := inputs[biasIdx]
		if biasNode.Shape.DType != qNode.Shape.DType {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"%s: bias must be a float dtype matching the compute dtype (%s), got %s",
				opName, qNode.Shape.DType, biasNode.Shape.DType)
		}
		biasRank := biasNode.Shape.Rank()
		if biasRank < 2 || biasRank > 4 {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"%s: bias must have rank 2, 3, or 4, got rank %d", opName, biasRank)
		}

		var bBatchDim, bHeadsDim, bSeqDim, bKVDim int
		var bBatchName, bHeadsName, bSeqName, bKVName string

		switch biasRank {
		case 4:
			if axesLayout == compute.AttentionAxesLayoutBSHD {
				bBatchDim = biasNode.Shape.Dimensions[0]
				bBatchName = biasNode.Shape.AxisName(0)
				bSeqDim = biasNode.Shape.Dimensions[1]
				bSeqName = biasNode.Shape.AxisName(1)
				bHeadsDim = biasNode.Shape.Dimensions[2]
				bHeadsName = biasNode.Shape.AxisName(2)
				bKVDim = biasNode.Shape.Dimensions[3]
				bKVName = biasNode.Shape.AxisName(3)
			} else {
				bBatchDim = biasNode.Shape.Dimensions[0]
				bBatchName = biasNode.Shape.AxisName(0)
				bHeadsDim = biasNode.Shape.Dimensions[1]
				bHeadsName = biasNode.Shape.AxisName(1)
				bSeqDim = biasNode.Shape.Dimensions[2]
				bSeqName = biasNode.Shape.AxisName(2)
				bKVDim = biasNode.Shape.Dimensions[3]
				bKVName = biasNode.Shape.AxisName(3)
			}
		case 3:
			bBatchDim = biasNode.Shape.Dimensions[0]
			bBatchName = biasNode.Shape.AxisName(0)
			bHeadsDim = 1
			bSeqDim = biasNode.Shape.Dimensions[1]
			bSeqName = biasNode.Shape.AxisName(1)
			bKVDim = biasNode.Shape.Dimensions[2]
			bKVName = biasNode.Shape.AxisName(2)
		case 2:
			bBatchDim = 1
			bHeadsDim = 1
			bSeqDim = biasNode.Shape.Dimensions[0]
			bSeqName = biasNode.Shape.AxisName(0)
			bKVDim = biasNode.Shape.Dimensions[1]
			bKVName = biasNode.Shape.AxisName(1)
			if bSeqName != "" && bSeqName == batchName {
				return nil, errors.Wrapf(compute.ErrNotImplemented,
					"%s: 2D bias has batch axis %q instead of query sequence axis", opName, bSeqName)
			}
		}

		if !isSDPABatchOrHeadCompatible(bBatchDim, bBatchName, batchDim, batchName) ||
			!isSDPABatchOrHeadCompatible(bHeadsDim, bHeadsName, numHeadsDim, numHeadsName) ||
			!isSDPABatchOrHeadCompatible(bSeqDim, bSeqName, seqDim, seqName) ||
			!isSDPASeqDimCompatible(bKVDim, bKVName, kvDim, kvName) {
			return nil, errors.Wrapf(compute.ErrNotImplemented,
				"%s: bias shape %s is not supported by go backend fused SDPA (requires [batch, heads, %d, %d])",
				opName, biasNode.Shape, seqDim, kvDim)
		}
	}

	data := &nodeScaledDotProductAttention{
		numHeads: numHeads, numKVHeads: numKVHeads, axesLayout: axesLayout,
		scale: scale, causal: causal,
		hasMask: mask != nil, hasSeqLens: hasSeqLens, hasBias: hasBias,
		options: options,
	}
	if mask != nil && inputs[3].IsConstant() {
		data.maskConstData = gobackend.NewAnonymousConstantData()
	}
	if hasBias && inputs[len(inputs)-1].IsConstant() {
		data.biasConstData = gobackend.NewAnonymousConstantData()
	}
	node, _ := f.GetOrCreateNode(opType, qNode.Shape.Clone(), inputs, data)
	return node, nil
}

// isSDPABatchOrHeadCompatible checks if a batch or head dimension can be broadcast (dim == 1)
// or matches the target dimension.
func isSDPABatchOrHeadCompatible(dim int, name string, targetDim int, targetName string) bool {
	if dim == 1 {
		return true
	}
	if dim > 0 && targetDim > 0 {
		return dim == targetDim
	}
	if dim < 0 && targetDim < 0 {
		if name != "" && targetName != "" {
			return name == targetName
		}
		return true
	}
	return false
}

// isSDPASeqDimCompatible checks if a sequence dimension in mask or bias matches the target sequence dimension.
// The go backend kernel cannot broadcast across sequence dimensions, so dim == 1 is only valid if targetDim == 1.
func isSDPASeqDimCompatible(dim int, name string, targetDim int, targetName string) bool {
	if targetDim == 1 {
		return dim == 1
	}
	if dim == 1 {
		// targetDim is either > 1 or dynamic (< 0). The go backend kernel cannot broadcast across
		// sequence length, so dim == 1 is not supported.
		return false
	}
	if dim > 0 && targetDim > 0 {
		return dim == targetDim
	}
	if dim < 0 && targetDim < 0 {
		if name != "" && targetName != "" {
			return name == targetName
		}
		return true
	}
	return false
}

// execFusedScaledDotProductAttention implements multi-head scaled dot-product attention.
// Both BHSD and BSHD layouts are handled directly via stride-based indexing in
// sdpaGeneric/sdpaMultiHeadGeneric, avoiding expensive transpose operations.
// mask: optional mask of rank 2–4 (broadcasting via strides). Can be boolean (true = attend,
// false = ignore) or additive (any float dtype, added to scores before softmax).
//
// Currently, quantized matmuls are not implemented (awaiting go-highway release),
// falling back to the non-quantized FusedScaledDotProductAttention using standard
// float32 arithmetic when QuantizedMatmuls is set in the options config.
func execFusedScaledDotProductAttention(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (
	*gobackend.Buffer, error) {
	data := node.Data.(*nodeScaledDotProductAttention)
	query := inputs[0]
	key := inputs[1]
	value := inputs[2]
	next := 3
	var mask *gobackend.Buffer
	if data.hasMask {
		mask = inputs[next]
		next++
	}
	var querySeqLen, keyValueSeqLen []int32
	if data.hasSeqLens {
		batchSize := inputs[0].RawShape.Dimensions[0]
		var ok bool
		querySeqLen, ok = inputs[next].Flat.([]int32)
		if !ok || len(querySeqLen) != batchSize {
			return nil, errors.Errorf("FusedScaledDotProductAttention: QuerySeqLen must be int32 with length batch (%d), got type %T len %d",
				batchSize, inputs[next].Flat, len(querySeqLen))
		}
		next++
		keyValueSeqLen, ok = inputs[next].Flat.([]int32)
		if !ok || len(keyValueSeqLen) != batchSize {
			return nil, errors.Errorf("FusedScaledDotProductAttention: KeyValueSeqLen must be int32 with length batch (%d), got type %T len %d",
				batchSize, inputs[next].Flat, len(keyValueSeqLen))
		}
		next++
	}
	var bias *gobackend.Buffer
	if data.hasBias {
		bias = inputs[next]
		next++
	}
	_ = next

	// Compute mask/bias strides for broadcasting (in BHSD convention: [batch, heads, seqLen, kvLen]).
	// For rank-4 BSHD masks/bias [batch, seq, heads, kvLen], only transpose when both seq > 1 and
	// heads > 1; if either is 1, swapping axes 1 and 2 does not alter the flat memory layout.
	var maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen int
	if mask != nil {
		mDims := mask.RawShape.Dimensions
		if data.axesLayout == compute.AttentionAxesLayoutBSHD && len(mDims) == 4 {
			if mDims[1] > 1 && mDims[2] > 1 {
				if data.maskConstData != nil {
					var transposeErr error
					mask = gobackend.GetOrSetCachedForm(data.maskConstData, func() *gobackend.Buffer {
						tb, err := transposeBuffer(backend, mask, []int{0, 2, 1, 3})
						if err != nil {
							transposeErr = err
							return nil
						}
						return tb
					})
					if transposeErr != nil {
						return nil, transposeErr
					}
				} else {
					var err error
					mask, err = transposeBuffer(backend, mask, []int{0, 2, 1, 3})
					if err != nil {
						return nil, err
					}
					defer backend.PutBuffer(mask)
				}
				mDims = mask.RawShape.Dimensions
			} else {
				mDims = []int{mDims[0], mDims[2], mDims[1], mDims[3]}
			}
		}
		maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen = sdpaComputeMaskStrides(mDims)
	}
	var biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen int
	if bias != nil {
		bDims := bias.RawShape.Dimensions
		if data.axesLayout == compute.AttentionAxesLayoutBSHD && len(bDims) == 4 {
			if bDims[1] > 1 && bDims[2] > 1 {
				if data.biasConstData != nil {
					var transposeErr error
					bias = gobackend.GetOrSetCachedForm(data.biasConstData, func() *gobackend.Buffer {
						tb, err := transposeBuffer(backend, bias, []int{0, 2, 1, 3})
						if err != nil {
							transposeErr = err
							return nil
						}
						return tb
					})
					if transposeErr != nil {
						return nil, transposeErr
					}
				} else {
					var err error
					bias, err = transposeBuffer(backend, bias, []int{0, 2, 1, 3})
					if err != nil {
						return nil, err
					}
					defer backend.PutBuffer(bias)
				}
				bDims = bias.RawShape.Dimensions
			} else {
				bDims = []int{bDims[0], bDims[2], bDims[1], bDims[3]}
			}
		}
		biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen = sdpaComputeMaskStrides(bDims)
	}

	output, err := backend.GetBuffer(query.RawShape.Clone())
	if err != nil {
		return nil, err
	}

	switch query.RawShape.DType {
	case dtypes.Float32:
		sdpaMultiHeadGeneric[float32](backend, query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen, querySeqLen, keyValueSeqLen)
	case dtypes.Float64:
		sdpaMultiHeadGeneric[float64](backend, query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen, querySeqLen, keyValueSeqLen)
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedScaledDotProductAttention: unsupported dtype %s", query.RawShape.DType)
	}

	return output, nil
}

// sdpaComputeMaskStrides returns (batchStride, headStride, seqStride, sliceLen) for indexing into
// a mask or bias tensor based on its rank. Dimensions of size 1 are broadcast (stride 0).
//
//	rank 2: [seqLen, kvLen]                     → (0, 0, seqStride, seqLen*kvLen)
//	rank 3: [batch, seqLen, kvLen]              → (batchStride, 0, seqStride, seqLen*kvLen)
//	rank 4: [batch, heads, seqLen, kvLen]       → (batchStride, headStride, seqStride, seqLen*kvLen)
func sdpaComputeMaskStrides(dims []int) (batchStride, headStride, seqStride, sliceLen int) {
	switch len(dims) {
	case 2:
		if dims[0] > 1 {
			seqStride = dims[1]
		}
		return 0, 0, seqStride, dims[0] * dims[1]
	case 3:
		if dims[0] > 1 {
			batchStride = dims[1] * dims[2]
		}
		if dims[1] > 1 {
			seqStride = dims[2]
		}
		return batchStride, 0, seqStride, dims[1] * dims[2]
	case 4:
		if dims[0] > 1 {
			batchStride = dims[1] * dims[2] * dims[3]
		}
		if dims[1] > 1 {
			headStride = dims[2] * dims[3]
		}
		if dims[2] > 1 {
			seqStride = dims[3]
		}
		return batchStride, headStride, seqStride, dims[2] * dims[3]
	default:
		panic(errors.Errorf("sdpaComputeMaskStrides: unsupported mask rank %d (dims=%v), expected rank 2, 3, or 4", len(dims), dims))
	}
}

// transposeBuffer transposes a buffer according to the given axis permutation,
// reusing the existing transposeIterator and transposeDTypeMap infrastructure.
func transposeBuffer(backend *gobackend.Backend, buf *gobackend.Buffer, permutations []int) (*gobackend.Buffer, error) {
	// Compute the output shape by permuting dimensions.
	dims := buf.RawShape.Dimensions
	outDims := make([]int, len(dims))
	for i, p := range permutations {
		outDims[i] = dims[p]
	}
	outShape := shapes.Make(buf.RawShape.DType, outDims...)

	output, err := backend.GetBuffer(outShape)
	if err != nil {
		return nil, err
	}
	it := ops.NewTransposeIterator(buf.RawShape, permutations)
	transposeFnAny, err := ops.TransposeDTypeMap.Get(buf.RawShape.DType)
	if err != nil {
		return nil, err
	}
	transposeFn := transposeFnAny.(func(operand, output *gobackend.Buffer, it *ops.TransposeIterator))
	transposeFn(buf, output, it)
	return output, nil
}

// sdpaGeneric computes scaled dot-product attention for a group of query heads
// that share the same key/value head (Grouped Query Attention / GQA).
// For standard multi-head attention, groupSize is 1.
//
// The q/k/v/output slices are the full flat arrays for the tensor; qOff and kvOff
// give the element-offset to the first element of this group at seq=0 (for Q, this
// is the first query head in the group). qSeqStride and kvSeqStride are the element
// stride between consecutive sequence positions for a single head (headDim for BHSD
// contiguous layout, numHeads*headDim for BSHD interleaved layout). qGroupStride is
// the element stride between consecutive query heads within the group.
// The output uses qOff/qSeqStride/qGroupStride (same layout as query).
//
// scores is a [kvLen] per-row scratch buffer.
// Masks are dense per-head [seqLen, kvLen] buffers, shared across the group when
// maskGroupStride is 0, or offset by maskGroupStride per group member for per-head masks.
// additiveBias (optional) is added to scores before additiveMask and softmax; indexed
// by biasGroupStride per group member (same stride convention as maskGroupStride).
func sdpaGeneric[T float32 | float64](
	q, k, v []T, qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []T,
	booleanMask []bool,
	maskGroupStride, maskSeqStride int,
	additiveBias []T,
	biasGroupStride, biasSeqStride int,
	scores []T,
	output []T,
	groupSize, qStart, qEnd, kvLen, headDim int, scale T, causal bool,
	qLimit, kvLimit int,
) {
	_, isF32 := any(scale).(float32)

	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride
		for qIdx := qStart; qIdx < qEnd; qIdx++ {
			outBase := gQOff + qIdx*qSeqStride
			outRow := output[outBase : outBase+headDim]
			if qIdx >= qLimit {
				// Padded query position: emit a zero output row, skip attention.
				clear(outRow)
				continue
			}

			kvLenUnmasked := kvLen
			if kvLimit < kvLenUnmasked {
				kvLenUnmasked = kvLimit
			}
			if causal {
				kvLenUnmasked = min(kvLenUnmasked, qIdx+1)
			}
			if kvLenUnmasked <= 0 {
				clear(outRow)
				continue
			}

			rowMax := T(math.Inf(-1))
			qBase := gQOff + qIdx*qSeqStride
			qRow := q[qBase : qBase+headDim]
			maskIdxBase := gMaskOff + qIdx*maskSeqStride
			biasIdxBase := gBiasOff + qIdx*biasSeqStride

			activeKVLen := kvLenUnmasked
			if len(booleanMask) > 0 {
				activeKVLen = 0
			}

			for kvIdx := range kvLenUnmasked {
				maskIdx := maskIdxBase + kvIdx
				if len(booleanMask) > 0 {
					if !booleanMask[maskIdx] {
						scores[kvIdx] = 0
						continue
					}
					activeKVLen = kvIdx + 1
				}
				kBase := kvOff + kvIdx*kvSeqStride
				kRow := k[kBase : kBase+headDim]

				var dot0, dot1, dot2, dot3 T
				d := 0
				for ; d+7 < headDim; d += 8 {
					dot0 += qRow[d]*kRow[d] + qRow[d+4]*kRow[d+4]
					dot1 += qRow[d+1]*kRow[d+1] + qRow[d+5]*kRow[d+5]
					dot2 += qRow[d+2]*kRow[d+2] + qRow[d+6]*kRow[d+6]
					dot3 += qRow[d+3]*kRow[d+3] + qRow[d+7]*kRow[d+7]
				}
				for ; d+3 < headDim; d += 4 {
					dot0 += qRow[d] * kRow[d]
					dot1 += qRow[d+1] * kRow[d+1]
					dot2 += qRow[d+2] * kRow[d+2]
					dot3 += qRow[d+3] * kRow[d+3]
				}
				dot := (dot0 + dot1) + (dot2 + dot3)
				for ; d < headDim; d++ {
					dot += qRow[d] * kRow[d]
				}

				s := dot * scale
				if len(additiveBias) > 0 {
					s += additiveBias[biasIdxBase+kvIdx]
				}
				if len(additiveMask) > 0 {
					s += additiveMask[maskIdx]
				}
				scores[kvIdx] = s
				if s > rowMax {
					rowMax = s
				}
			}

			kvLenUnmasked = activeKVLen
			if kvLenUnmasked <= 0 || rowMax == T(math.Inf(-1)) {
				// All positions were masked: output row is all zeros.
				clear(outRow)
				continue
			}

			// Softmax: exp(scores - max) and sum.
			var sum T
			if len(booleanMask) > 0 {
				if isF32 {
					for kvIdx := range kvLenUnmasked {
						if booleanMask[maskIdxBase+kvIdx] {
							e := T(fastmath.Exp32(float32(scores[kvIdx] - rowMax)))
							scores[kvIdx] = e
							sum += e
						} else {
							scores[kvIdx] = 0
						}
					}
				} else {
					for kvIdx := range kvLenUnmasked {
						if booleanMask[maskIdxBase+kvIdx] {
							e := T(math.Exp(float64(scores[kvIdx] - rowMax)))
							scores[kvIdx] = e
							sum += e
						} else {
							scores[kvIdx] = 0
						}
					}
				}
			} else {
				if isF32 {
					for kvIdx := range kvLenUnmasked {
						e := T(fastmath.Exp32(float32(scores[kvIdx] - rowMax)))
						scores[kvIdx] = e
						sum += e
					}
				} else {
					for kvIdx := range kvLenUnmasked {
						e := T(math.Exp(float64(scores[kvIdx] - rowMax)))
						scores[kvIdx] = e
						sum += e
					}
				}
			}

			var invSum T
			if sum != 0 {
				invSum = 1.0 / sum
			}
			for kvIdx := range kvLenUnmasked {
				scores[kvIdx] *= invSum
			}

			// output[qIdx][d] = sum_kvIdx(scores[kvIdx] * v[kvIdx][d])
			// Iterate kvIdx on the outside and d on the inside for contiguous L1 access.
			clear(outRow)
			for kvIdx := range kvLenUnmasked {
				w := scores[kvIdx]
				if w == 0 {
					continue
				}
				vBase := kvOff + kvIdx*kvSeqStride
				vRow := v[vBase : vBase+headDim]
				d := 0
				for ; d+7 < headDim; d += 8 {
					outRow[d] += w * vRow[d]
					outRow[d+1] += w * vRow[d+1]
					outRow[d+2] += w * vRow[d+2]
					outRow[d+3] += w * vRow[d+3]
					outRow[d+4] += w * vRow[d+4]
					outRow[d+5] += w * vRow[d+5]
					outRow[d+6] += w * vRow[d+6]
					outRow[d+7] += w * vRow[d+7]
				}
				for ; d < headDim; d++ {
					outRow[d] += w * vRow[d]
				}
			}
		}
	}
}

func sdpaMultiHeadGeneric[T float32 | float64](
	backend *gobackend.Backend,
	query, key, value, mask, bias, output *gobackend.Buffer,
	data *nodeScaledDotProductAttention,
	maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen int,
	querySeqLen, keyValueSeqLen []int32,
) {
	q := query.Flat.([]T)
	k := key.Flat.([]T)
	v := value.Flat.([]T)
	out := output.Flat.([]T)
	var additiveMask []T
	var booleanMask []bool
	if mask != nil {
		if mask.RawShape.DType == dtypes.Bool {
			booleanMask = mask.Flat.([]bool)
		} else {
			additiveMask = mask.Flat.([]T)
		}
	}
	var additiveBias []T
	if bias != nil {
		additiveBias = bias.Flat.([]T)
	}

	dims := query.RawShape.Dimensions
	batchSize := dims[0]
	numHeads := data.numHeads
	numKVHeads := data.numKVHeads
	scale := T(data.scale)
	causal := data.causal
	groupSize := numHeads / numKVHeads

	// Layout-dependent axis indices and strides.
	var seqLen, kvLen, headDim int
	var qSeqStride, kvSeqStride int     // element stride between consecutive seq positions for one head
	var qBatchStride, kvBatchStride int // element stride between consecutive batches
	var qHeadStride, kvHeadStride int   // element stride between consecutive heads at seq=0

	if data.axesLayout == compute.AttentionAxesLayoutBSHD {
		// [batch, seq, heads, dim]
		seqLen = dims[1]
		headDim = dims[3]
		kvDims := key.RawShape.Dimensions
		kvLen = kvDims[1]
		qSeqStride = numHeads * headDim
		kvSeqStride = numKVHeads * headDim
		qHeadStride = headDim
		kvHeadStride = headDim
		qBatchStride = seqLen * numHeads * headDim
		kvBatchStride = kvLen * numKVHeads * headDim
	} else {
		// BHSD: [batch, heads, seq, dim]
		seqLen = dims[2]
		headDim = dims[3]
		kvDims := key.RawShape.Dimensions
		kvLen = kvDims[2]
		qSeqStride = headDim
		kvSeqStride = headDim
		qHeadStride = seqLen * headDim
		kvHeadStride = kvLen * headDim
		qBatchStride = numHeads * seqLen * headDim
		kvBatchStride = numKVHeads * kvLen * headDim
	}

	archFn := gobackend.GetSDPAArchDispatcher()

	processHead := func(batchIdx, kvHeadIdx, qStart, qEnd int, scores []T) {
		qLimit := seqLen
		kvLimit := kvLen
		if len(querySeqLen) > 0 {
			qLimit = max(0, min(int(querySeqLen[batchIdx]), seqLen))
		}
		if len(keyValueSeqLen) > 0 {
			kvLimit = max(0, min(int(keyValueSeqLen[batchIdx]), kvLen))
		}
		qOff := batchIdx*qBatchStride + kvHeadIdx*groupSize*qHeadStride
		kvOff := batchIdx*kvBatchStride + kvHeadIdx*kvHeadStride

		// Compute mask slice and group stride for this KV head group.
		var additiveMaskSlice []T
		var booleanMaskSlice []bool
		maskGroupStride := 0
		if len(additiveMask) > 0 || len(booleanMask) > 0 {
			maskOffset := batchIdx*maskBatchStride + kvHeadIdx*groupSize*maskHeadStride
			maskEnd := maskOffset + maskSliceLen
			if maskHeadStride > 0 && groupSize > 1 {
				maskEnd = maskOffset + (groupSize-1)*maskHeadStride + maskSliceLen
				maskGroupStride = maskHeadStride
			}
			if len(additiveMask) > 0 {
				additiveMaskSlice = additiveMask[maskOffset:maskEnd]
			} else {
				booleanMaskSlice = booleanMask[maskOffset:maskEnd]
			}
			// Fast path for 1D key-padding masks (broadcast across query sequence and head group):
			// When valid prefix positions have 0 additive mask (or true boolean mask) and trailing
			// padded key positions have mask <= -1e4 (or false), trim kvLimit to the active prefix
			// and elide the zero/true mask slice.
			if maskSeqStride == 0 && maskGroupStride == 0 && kvLimit > 0 {
				if len(additiveMaskSlice) >= kvLimit {
					lastValid := kvLimit - 1
					for lastValid >= 0 && additiveMaskSlice[lastValid] <= -1e4 {
						lastValid--
					}
					if lastValid < 0 {
						if math.IsInf(float64(additiveMaskSlice[0]), -1) {
							kvLimit = 0
						}
					} else {
						allZero := true
						for i := 0; i <= lastValid; i++ {
							if additiveMaskSlice[i] != 0 {
								allZero = false
								break
							}
						}
						if allZero {
							kvLimit = lastValid + 1
							additiveMaskSlice = nil
						}
					}
				} else if len(booleanMaskSlice) >= kvLimit {
					lastValid := kvLimit - 1
					for lastValid >= 0 && !booleanMaskSlice[lastValid] {
						lastValid--
					}
					if lastValid < 0 {
						kvLimit = 0
					} else {
						kvLimit = lastValid + 1
						allTrue := true
						for i := 0; i < kvLimit; i++ {
							if !booleanMaskSlice[i] {
								allTrue = false
								break
							}
						}
						if allTrue {
							booleanMaskSlice = nil
						}
					}
				}
			}
		}
		// Compute bias slice and group stride for this KV head group.
		var additiveBiasSlice []T
		biasGroupStride := 0
		if len(additiveBias) > 0 {
			// *groupSize: bias is per-Q-head under GQA; first Q-head of this KV group starts at kvHeadIdx*groupSize.
			biasOffset := batchIdx*biasBatchStride + kvHeadIdx*groupSize*biasHeadStride
			biasEnd := biasOffset + biasSliceLen
			if biasHeadStride > 0 && groupSize > 1 {
				biasEnd = biasOffset + (groupSize-1)*biasHeadStride + biasSliceLen
				biasGroupStride = biasHeadStride
			}
			additiveBiasSlice = additiveBias[biasOffset:biasEnd]
		}

		if archFn != nil {
			if qF32, ok := any(q).([]float32); ok {
				kF32 := any(k).([]float32)
				vF32 := any(v).([]float32)
				outF32 := any(out).([]float32)
				var addMaskF32 []float32
				if len(additiveMaskSlice) > 0 {
					addMaskF32 = any(additiveMaskSlice).([]float32)
				}
				var addBiasF32 []float32
				if len(additiveBiasSlice) > 0 {
					addBiasF32 = any(additiveBiasSlice).([]float32)
				}
				scratchF32 := any(scores).([]float32)

				if archFn(
					qF32, kF32, vF32, outF32,
					qOff, kvOff, qSeqStride, kvSeqStride, qHeadStride,
					addMaskF32, booleanMaskSlice, maskGroupStride,
					addBiasF32, biasGroupStride,
					scratchF32,
					groupSize, qStart, qEnd, kvLen, headDim,
					float32(scale), causal,
					qLimit, kvLimit,
				) {
					return
				}
			}
		}

		sdpaGeneric(
			q, k, v, qOff, kvOff, qSeqStride, kvSeqStride, qHeadStride,
			additiveMaskSlice, booleanMaskSlice, maskGroupStride, maskSeqStride,
			additiveBiasSlice, biasGroupStride, biasSeqStride,
			scores,
			out,
			groupSize, qStart, qEnd, kvLen, headDim, scale, causal,
			qLimit, kvLimit,
		)
	}

	headTasks := batchSize * numKVHeads
	qBlockSize := max(seqLen, 1)
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() {
		numWorkers := backend.Scheduler.EffectiveParallelism()
		if numWorkers > 1 && seqLen > 1 && headTasks*seqLen*kvLen >= 1024 {
			targetTasks := numWorkers * 8
			if headTasks < targetTasks {
				neededQBlocks := (targetTasks + headTasks - 1) / headTasks
				qBlockSize = max(8, (seqLen+neededQBlocks-1)/neededQBlocks)
			}
		}
	}
	numQBlocks := 1
	if seqLen > 0 {
		numQBlocks = (seqLen + qBlockSize - 1) / qBlockSize
	}
	totalTasks := headTasks * numQBlocks
	scratchSize := max(32*kvLen+16, 64)
	scratchShape := shapes.Make(query.RawShape.DType, scratchSize)

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && totalTasks > 1 {
		numWorkers := backend.Scheduler.EffectiveParallelism()
		workersToStart := min(totalTasks, numWorkers)
		var taskCounter atomic.Int64

		backend.Scheduler.SaturateUpTo(workersToStart, func() {
			scoreBuf, err := backend.GetBuffer(scratchShape)
			var workerScores []T
			if err == nil && scoreBuf != nil {
				defer backend.PutBuffer(scoreBuf)
				workerScores = scoreBuf.Flat.([]T)
			} else {
				workerScores = make([]T, scratchSize)
			}
			for {
				taskIdx := int(taskCounter.Add(1) - 1)
				if taskIdx >= totalTasks {
					break
				}
				headTaskIdx := taskIdx / numQBlocks
				qBlockIdx := taskIdx % numQBlocks
				batchIdx := headTaskIdx / numKVHeads
				kvHeadIdx := headTaskIdx % numKVHeads
				qStart := qBlockIdx * qBlockSize
				qEnd := min(qStart+qBlockSize, seqLen)
				processHead(batchIdx, kvHeadIdx, qStart, qEnd, workerScores)
			}
		})
	} else {
		var scores []T
		if backend != nil {
			if scoreBuf, err := backend.GetBuffer(scratchShape); err == nil && scoreBuf != nil {
				defer backend.PutBuffer(scoreBuf)
				scores = scoreBuf.Flat.([]T)
			}
		}
		if scores == nil {
			scores = make([]T, scratchSize)
		}
		for batchIdx := range batchSize {
			for kvHeadIdx := range numKVHeads {
				processHead(batchIdx, kvHeadIdx, 0, seqLen, scores)
			}
		}
	}
}

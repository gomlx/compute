package fusedops

import (
	"math"
	"sync/atomic"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/fastmath"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/internal/gobackend/simdarch"
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
	case dtypes.Float32, dtypes.Float64, dtypes.Float16, dtypes.BFloat16:
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
			return shapes.AxisNamesCompatible(name, targetName)
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
			return shapes.AxisNamesCompatible(name, targetName)
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
	if backend.NoOps {
		return output, nil
	}

	switch query.RawShape.DType {
	case dtypes.Float32:
		sdpaMultiHeadGeneric[float32](backend, query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen, querySeqLen, keyValueSeqLen)
	case dtypes.Float64:
		sdpaMultiHeadGeneric[float64](backend, query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen, querySeqLen, keyValueSeqLen)
	case dtypes.Float16:
		conv := gobackend.ActiveHalfSliceConverters()
		err = sdpaMultiHeadHalf[float16.Float16](backend, query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen, querySeqLen, keyValueSeqLen, conv.Float16ToFloat32, conv.Float32ToFloat16)
	case dtypes.BFloat16:
		conv := gobackend.ActiveHalfSliceConverters()
		err = sdpaMultiHeadHalf[bfloat16.BFloat16](backend, query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen, querySeqLen, keyValueSeqLen, conv.BFloat16ToFloat32, conv.Float32ToBFloat16)
	default:
		backend.PutBuffer(output)
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedScaledDotProductAttention: unsupported dtype %s", query.RawShape.DType)
	}
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
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

type sdpaLayoutInfo struct {
	batchSize, numHeads, numKVHeads, groupSize int
	seqLen, kvLen, headDim                     int
	qSeqStride, kvSeqStride                    int
	qBatchStride, kvBatchStride                int
	qHeadStride, kvHeadStride                  int
	qBlockSize, numQBlocks, totalTasks         int
	scratchSize                                int
	totalMACs                                  int
	useSMEPath                                 bool
}

func computeSDPALayout(backend *gobackend.Backend, query, key *gobackend.Buffer, data *nodeScaledDotProductAttention, allowSME bool) sdpaLayoutInfo {
	dims := query.RawShape.Dimensions
	info := sdpaLayoutInfo{
		batchSize:  dims[0],
		numHeads:   data.numHeads,
		numKVHeads: data.numKVHeads,
		groupSize:  data.numHeads / data.numKVHeads,
	}

	if data.axesLayout == compute.AttentionAxesLayoutBSHD {
		info.seqLen = dims[1]
		info.headDim = dims[3]
		kvDims := key.RawShape.Dimensions
		info.kvLen = kvDims[1]
		info.qSeqStride = info.numHeads * info.headDim
		info.kvSeqStride = info.numKVHeads * info.headDim
		info.qHeadStride = info.headDim
		info.kvHeadStride = info.headDim
		info.qBatchStride = info.seqLen * info.numHeads * info.headDim
		info.kvBatchStride = info.kvLen * info.numKVHeads * info.headDim
	} else {
		info.seqLen = dims[2]
		info.headDim = dims[3]
		kvDims := key.RawShape.Dimensions
		info.kvLen = kvDims[2]
		info.qSeqStride = info.headDim
		info.kvSeqStride = info.headDim
		info.qHeadStride = info.seqLen * info.headDim
		info.kvHeadStride = info.kvLen * info.headDim
		info.qBatchStride = info.numHeads * info.seqLen * info.headDim
		info.kvBatchStride = info.numKVHeads * info.kvLen * info.headDim
	}

	headTasks := info.batchSize * info.numKVHeads
	info.totalMACs = info.batchSize * info.numHeads * info.seqLen * info.kvLen * info.headDim * 2
	info.useSMEPath = allowSME && gobackend.IsSMEAllowed && info.seqLen >= 4 && info.kvLen >= 16 && info.headDim >= 16 && info.headDim <= 256
	info.qBlockSize = max(info.seqLen, 1)
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() {
		numWorkers := backend.Scheduler.EffectiveParallelism()
		if info.useSMEPath {
			if numWorkers > 1 && info.seqLen > 64 && info.totalMACs >= 1024*1024 && headTasks < numWorkers {
				neededQBlocks := (numWorkers + headTasks - 1) / headTasks
				info.qBlockSize = max(32, ((info.seqLen+neededQBlocks-1)/neededQBlocks+31)&^31)
			}
		} else if numWorkers > 1 && info.seqLen >= 8 && info.totalMACs >= 512*1024 && headTasks < 4*numWorkers {
			targetTasks := 4 * numWorkers
			neededQBlocks := (targetTasks + headTasks - 1) / headTasks
			info.qBlockSize = max(8, ((info.seqLen+neededQBlocks-1)/neededQBlocks+3)&^3)
		}
	}
	info.numQBlocks = 1
	if info.seqLen > 0 {
		info.numQBlocks = (info.seqLen + info.qBlockSize - 1) / info.qBlockSize
	}
	info.totalTasks = headTasks * info.numQBlocks
	info.scratchSize = max(32*info.kvLen+64, 128)
	if info.useSMEPath {
		kvPadded := (info.kvLen + 31) &^ 31
		info.scratchSize = max(info.scratchSize, 32*info.kvLen+(32+info.headDim)*kvPadded+32*info.headDim+16)
	}
	return info
}

func runSDPATasks[W any](
	backend *gobackend.Backend,
	info sdpaLayoutInfo,
	initWorker func() W,
	cleanupWorker func(W),
	runTask func(worker W, batchIdx, kvHeadIdx, qStart, qEnd int),
) {
	minParallelMACs := 1024 * 1024
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.EffectiveParallelism() < backend.Scheduler.UnscaledParallelism() {
		minParallelMACs = 128 * 1024 * 1024
	}
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && info.totalTasks > 1 && info.totalMACs >= minParallelMACs && backend.Scheduler.EffectiveParallelism() > 1 {
		numWorkers := backend.Scheduler.EffectiveParallelism()
		workersToStart := min(info.totalTasks, numWorkers)
		if info.useSMEPath && simdarch.NumHighPerfSMEUnits > 0 {
			// Start up to numWorkers so Go's scheduler populates all Ps on Cluster 1 (Super/Perf cores),
			// while E-core workers yield tasks to Cluster 1 workers below.
			workersToStart = numWorkers
		}
		var taskCounter atomic.Int64
		var fastWorkers atomic.Int32
		var droppedWorkers atomic.Int32
		preferFastCluster := info.useSMEPath && simdarch.NumHighPerfSMEUnits > 0

		backend.Scheduler.SaturateUpTo(workersToStart, func() {
			if preferFastCluster {
				_, _, isEff := simdarch.CurrentSMECluster()
				if !isEff {
					fastWorkers.Add(1)
					defer fastWorkers.Add(-1)
				} else if int(droppedWorkers.Add(1)) < workersToStart {
					for spins := 0; spins < 2048 && fastWorkers.Load() == 0 && taskCounter.Load() < int64(info.totalTasks); spins++ {
					}
					if fastWorkers.Load() > 0 {
						for taskCounter.Load() < int64(info.totalTasks) && fastWorkers.Load() > 0 {
							rem := info.totalTasks - int(taskCounter.Load())
							if rem > 8*int(fastWorkers.Load()) {
								break
							}
						}
						if taskCounter.Load() >= int64(info.totalTasks) || info.totalTasks-int(taskCounter.Load()) <= 8*int(fastWorkers.Load()) {
							return
						}
					}
				}
			}

			w := initWorker()
			if cleanupWorker != nil {
				defer cleanupWorker(w)
			}
			for {
				taskIdx := int(taskCounter.Add(1) - 1)
				if taskIdx >= info.totalTasks {
					break
				}
				headTaskIdx := taskIdx / info.numQBlocks
				qBlockIdx := taskIdx % info.numQBlocks
				batchIdx := headTaskIdx / info.numKVHeads
				kvHeadIdx := headTaskIdx % info.numKVHeads
				qStart := qBlockIdx * info.qBlockSize
				qEnd := min(qStart+info.qBlockSize, info.seqLen)
				runTask(w, batchIdx, kvHeadIdx, qStart, qEnd)
			}
		})
		return
	}

	w := initWorker()
	if cleanupWorker != nil {
		defer cleanupWorker(w)
	}
	for batchIdx := range info.batchSize {
		for kvHeadIdx := range info.numKVHeads {
			runTask(w, batchIdx, kvHeadIdx, 0, info.seqLen)
		}
	}
}

func sdpaTrimBoolKeyMask(booleanMaskSlice []bool, kvLimit int) ([]bool, int) {
	if len(booleanMaskSlice) < kvLimit || kvLimit <= 0 {
		return booleanMaskSlice, kvLimit
	}
	lastValid := kvLimit - 1
	for lastValid >= 0 && !booleanMaskSlice[lastValid] {
		lastValid--
	}
	if lastValid < 0 {
		return booleanMaskSlice, 0
	}
	kvLimit = lastValid + 1
	for i := range kvLimit {
		if !booleanMaskSlice[i] {
			return booleanMaskSlice, kvLimit
		}
	}
	return nil, kvLimit
}

func sdpaTrimAdditiveKeyMaskFloat[T ~float32 | ~float64](additiveMaskSlice []T, kvLimit int) ([]T, int) {
	if len(additiveMaskSlice) < kvLimit || kvLimit <= 0 {
		return additiveMaskSlice, kvLimit
	}
	lastValid := kvLimit - 1
	for lastValid >= 0 && additiveMaskSlice[lastValid] <= -1e4 {
		lastValid--
	}
	if lastValid < 0 {
		if math.IsInf(float64(additiveMaskSlice[0]), -1) {
			return additiveMaskSlice, 0
		}
		return additiveMaskSlice, kvLimit
	}
	for i := 0; i <= lastValid; i++ {
		if additiveMaskSlice[i] != 0 {
			return additiveMaskSlice, kvLimit
		}
	}
	return nil, lastValid + 1
}

func sdpaMaskBiasGroupSlice[T any](data []T, batchIdx, kvHeadIdx, groupSize, batchStride, headStride, sliceLen int) (slice []T, groupStride int) {
	if len(data) == 0 {
		return nil, 0
	}
	offset := batchIdx*batchStride + kvHeadIdx*groupSize*headStride
	end := offset + sliceLen
	if headStride > 0 && groupSize > 1 {
		end = offset + (groupSize-1)*headStride + sliceLen
		groupStride = headStride
	}
	return data[offset:end], groupStride
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

	info := computeSDPALayout(backend, query, key, data, query.RawShape.DType == dtypes.Float32)
	scale := T(data.scale)
	causal := data.causal
	archFn := gobackend.GetSDPAArchDispatcher()
	scratchShape := shapes.Make(query.RawShape.DType, info.scratchSize)

	type genericWorker struct {
		buf    *gobackend.Buffer
		scores []T
	}

	runSDPATasks(
		backend,
		info,
		func() genericWorker {
			if backend != nil {
				if scoreBuf, err := backend.GetBuffer(scratchShape); err == nil && scoreBuf != nil {
					return genericWorker{buf: scoreBuf, scores: scoreBuf.Flat.([]T)}
				}
			}
			return genericWorker{scores: make([]T, info.scratchSize)}
		},
		func(w genericWorker) {
			if w.buf != nil && backend != nil {
				backend.PutBuffer(w.buf)
			}
		},
		func(w genericWorker, batchIdx, kvHeadIdx, qStart, qEnd int) {
			qLimit := info.seqLen
			kvLimit := info.kvLen
			if len(querySeqLen) > 0 {
				qLimit = max(0, min(int(querySeqLen[batchIdx]), info.seqLen))
			}
			if len(keyValueSeqLen) > 0 {
				kvLimit = max(0, min(int(keyValueSeqLen[batchIdx]), info.kvLen))
			}
			qOff := batchIdx*info.qBatchStride + kvHeadIdx*info.groupSize*info.qHeadStride
			kvOff := batchIdx*info.kvBatchStride + kvHeadIdx*info.kvHeadStride

			additiveMaskSlice, maskGroupStride := sdpaMaskBiasGroupSlice(additiveMask, batchIdx, kvHeadIdx, info.groupSize, maskBatchStride, maskHeadStride, maskSliceLen)
			booleanMaskSlice, boolGroupStride := sdpaMaskBiasGroupSlice(booleanMask, batchIdx, kvHeadIdx, info.groupSize, maskBatchStride, maskHeadStride, maskSliceLen)
			if len(booleanMaskSlice) > 0 {
				maskGroupStride = boolGroupStride
			}
			if maskSeqStride == 0 && maskGroupStride == 0 && kvLimit > 0 {
				if len(additiveMaskSlice) > 0 {
					additiveMaskSlice, kvLimit = sdpaTrimAdditiveKeyMaskFloat(additiveMaskSlice, kvLimit)
				} else if len(booleanMaskSlice) > 0 {
					booleanMaskSlice, kvLimit = sdpaTrimBoolKeyMask(booleanMaskSlice, kvLimit)
				}
			}

			additiveBiasSlice, biasGroupStride := sdpaMaskBiasGroupSlice(additiveBias, batchIdx, kvHeadIdx, info.groupSize, biasBatchStride, biasHeadStride, biasSliceLen)

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
					scratchF32 := any(w.scores).([]float32)

					if archFn(
						qF32, kF32, vF32, outF32,
						qOff, kvOff, info.qSeqStride, info.kvSeqStride, info.qHeadStride,
						addMaskF32, booleanMaskSlice, maskGroupStride,
						addBiasF32, biasGroupStride,
						scratchF32,
						info.groupSize, qStart, qEnd, info.kvLen, info.headDim,
						float32(scale), causal,
						qLimit, kvLimit,
					) {
						return
					}
				}
			}

			sdpaGeneric(
				q, k, v, qOff, kvOff, info.qSeqStride, info.kvSeqStride, info.qHeadStride,
				additiveMaskSlice, booleanMaskSlice, maskGroupStride, maskSeqStride,
				additiveBiasSlice, biasGroupStride, biasSeqStride,
				w.scores,
				out,
				info.groupSize, qStart, qEnd, info.kvLen, info.headDim, scale, causal,
				qLimit, kvLimit,
			)
		},
	)
}

const (
	sdpaHalfConvMinParallelElems = gobackend.ParallelHalfConvThreshold
	sdpaHalfConvBlockElems       = 1024
)

type sdpaHalfToF32Pair[T any] struct {
	dst    []float32
	src    []T
	blocks int
}

func sdpaAllocF32Buffer(backend *gobackend.Backend, like *gobackend.Buffer) (*gobackend.Buffer, error) {
	f32Shape := like.RawShape.Clone()
	f32Shape.DType = dtypes.Float32
	return backend.GetBuffer(f32Shape)
}

func sdpaConvertHalfInputsToF32[T any](
	backend *gobackend.Backend,
	toF32 func([]float32, []T),
	pairs ...sdpaHalfToF32Pair[T],
) {
	totalElems := 0
	totalBlocks := 0
	for i := range pairs {
		n := len(pairs[i].src)
		totalElems += n
		pairs[i].blocks = (n + sdpaHalfConvBlockElems - 1) / sdpaHalfConvBlockElems
		totalBlocks += pairs[i].blocks
	}
	if totalElems == 0 {
		return
	}
	if backend == nil || backend.Scheduler == nil || !backend.Scheduler.IsEnabled() || totalElems < sdpaHalfConvMinParallelElems {
		for _, p := range pairs {
			toF32(p.dst, p.src)
		}
		return
	}
	minBlocks := max(1, 32768/sdpaHalfConvBlockElems)
	backend.Scheduler.ParallelizeChunks(totalBlocks, minBlocks, func(bStart, bEnd int) {
		cursor := 0
		for _, p := range pairs {
			pStart := cursor
			pEnd := cursor + p.blocks
			cursor = pEnd
			if bEnd <= pStart || bStart >= pEnd {
				continue
			}
			localStart := max(0, bStart-pStart) * sdpaHalfConvBlockElems
			localEnd := min(len(p.src), (min(bEnd, pEnd)-pStart)*sdpaHalfConvBlockElems)
			toF32(p.dst[localStart:localEnd], p.src[localStart:localEnd])
		}
	})
}

func sdpaConvertHeadRowsToF32[T any](dst []float32, src []T, base, seqStride, headStride, groupSize, rows, headDim int, toF32 func([]float32, []T)) {
	if seqStride == headDim {
		chunk := rows * headDim
		for g := range groupSize {
			s := base + g*headStride
			toF32(dst[g*chunk:(g+1)*chunk], src[s:s+chunk])
		}
		return
	}
	for g := range groupSize {
		gDst := g * rows * headDim
		gSrc := base + g*headStride
		for r := range rows {
			toF32(dst[gDst+r*headDim:gDst+(r+1)*headDim], src[gSrc+r*seqStride:gSrc+r*seqStride+headDim])
		}
	}
}

func sdpaConvertHeadRowsFromF32[T any](dst []T, src []float32, base, seqStride, headStride, groupSize, rows, headDim int, fromF32 func([]T, []float32)) {
	if seqStride == headDim {
		chunk := rows * headDim
		for g := range groupSize {
			d := base + g*headStride
			fromF32(dst[d:d+chunk], src[g*chunk:(g+1)*chunk])
		}
		return
	}
	for g := range groupSize {
		gSrc := g * rows * headDim
		gDst := base + g*headStride
		for r := range rows {
			fromF32(dst[gDst+r*seqStride:gDst+r*seqStride+headDim], src[gSrc+r*headDim:gSrc+(r+1)*headDim])
		}
	}
}

// sdpaMultiHeadHalf executes Float16/BFloat16 SDPA using hardware-accelerated slice converters.
// When non-causal or when a single Q-block covers the head (numQBlocks == 1), each task converts its
// Q/output blocks inside a reusable worker L1/L2 Float32 scratch buffer (skipping global qF32/outF32
// allocations and the trailing conversion barrier; small KV heads or numQBlocks == 1 also keep K/V
// conversion per-worker).
func sdpaMultiHeadHalf[T any](
	backend *gobackend.Backend,
	query, key, value, mask, bias, output *gobackend.Buffer,
	data *nodeScaledDotProductAttention,
	maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen, biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen int,
	querySeqLen, keyValueSeqLen []int32,
	toF32 func([]float32, []T),
	fromF32 func([]T, []float32),
) error {
	info := computeSDPALayout(backend, query, key, data, true)
	localQOut := !data.causal || info.numQBlocks == 1
	// Keep K/V conversion inside each worker's L1/L2 scratch when a single Q-block
	// covers the head, or when each KV head fits comfortably in L1/L2 (<= 128 KB F32)
	// and fan-out duplication is modest, eliminating global kF32/vF32 allocations
	// and the upfront conversion barrier.
	kvHeadElems := info.kvLen * info.headDim
	localKV := localQOut && (info.numQBlocks == 1 || (info.numQBlocks <= 32 && kvHeadElems <= 32768))

	var qF32, kF32, vF32, outF32 *gobackend.Buffer
	var err error
	pairs := make([]sdpaHalfToF32Pair[T], 0, 5)

	if !localQOut {
		qF32, err = sdpaAllocF32Buffer(backend, query)
		if err != nil {
			return err
		}
		defer backend.PutBuffer(qF32)

		outF32, err = sdpaAllocF32Buffer(backend, output)
		if err != nil {
			return err
		}
		defer backend.PutBuffer(outF32)

		pairs = append(pairs, sdpaHalfToF32Pair[T]{dst: qF32.Flat.([]float32), src: query.Flat.([]T)})
	}

	if !localKV {
		kF32, err = sdpaAllocF32Buffer(backend, key)
		if err != nil {
			return err
		}
		defer backend.PutBuffer(kF32)

		vF32, err = sdpaAllocF32Buffer(backend, value)
		if err != nil {
			return err
		}
		defer backend.PutBuffer(vF32)

		pairs = append(pairs,
			sdpaHalfToF32Pair[T]{dst: kF32.Flat.([]float32), src: key.Flat.([]T)},
			sdpaHalfToF32Pair[T]{dst: vF32.Flat.([]float32), src: value.Flat.([]T)},
		)
	}

	maskF32 := mask
	if mask != nil && mask.RawShape.DType != dtypes.Bool && mask.RawShape.DType != dtypes.Float32 {
		maskF32, err = sdpaAllocF32Buffer(backend, mask)
		if err != nil {
			return err
		}
		defer backend.PutBuffer(maskF32)
		pairs = append(pairs, sdpaHalfToF32Pair[T]{dst: maskF32.Flat.([]float32), src: mask.Flat.([]T)})
	}

	biasF32 := bias
	if bias != nil && bias.RawShape.DType != dtypes.Float32 {
		biasF32, err = sdpaAllocF32Buffer(backend, bias)
		if err != nil {
			return err
		}
		defer backend.PutBuffer(biasF32)
		pairs = append(pairs, sdpaHalfToF32Pair[T]{dst: biasF32.Flat.([]float32), src: bias.Flat.([]T)})
	}

	if len(pairs) > 0 {
		sdpaConvertHalfInputsToF32(backend, toF32, pairs...)
	}

	if !localQOut {
		sdpaMultiHeadGeneric[float32](
			backend, qF32, kF32, vF32, maskF32, biasF32, outF32,
			data, maskBatchStride, maskHeadStride, maskSeqStride, maskSliceLen,
			biasBatchStride, biasHeadStride, biasSeqStride, biasSliceLen,
			querySeqLen, keyValueSeqLen,
		)
		gobackend.ParallelConvertSlice(backend, output.Flat.([]T), outF32.Flat.([]float32), fromF32)
		return nil
	}

	// Local Q/Out (and optional K/V) path: convert each task's slices in reusable worker scratch.
	qHalf := query.Flat.([]T)
	kHalf := key.Flat.([]T)
	vHalf := value.Flat.([]T)
	outHalf := output.Flat.([]T)

	var kAllF32, vAllF32 []float32
	if !localKV {
		kAllF32 = kF32.Flat.([]float32)
		vAllF32 = vF32.Flat.([]float32)
	}

	var additiveMask []float32
	var booleanMask []bool
	if maskF32 != nil {
		if maskF32.RawShape.DType == dtypes.Bool {
			booleanMask = maskF32.Flat.([]bool)
		} else {
			additiveMask = maskF32.Flat.([]float32)
		}
	}
	var additiveBias []float32
	if biasF32 != nil {
		additiveBias = biasF32.Flat.([]float32)
	}

	qBlockMaxElems := info.groupSize * info.qBlockSize * info.headDim
	kvBlockElems := 0
	if localKV {
		kvBlockElems = info.kvLen * info.headDim
	}
	workerTotalElems := info.scratchSize + 2*qBlockMaxElems + 2*kvBlockElems
	workerScratchShape := shapes.Make(dtypes.Float32, workerTotalElems)
	archFn := gobackend.GetSDPAArchDispatcher()
	scale := float32(data.scale)
	causal := data.causal

	type halfWorker struct {
		buf         *gobackend.Buffer
		scores      []float32
		qWork       []float32
		outW        []float32
		kWork       []float32
		vWork       []float32
		lastBatchKV *[2]int
	}

	runSDPATasks(
		backend,
		info,
		func() halfWorker {
			var buf *gobackend.Buffer
			var flat []float32
			if backend != nil {
				if b, bErr := backend.GetBuffer(workerScratchShape); bErr == nil && b != nil {
					buf = b
					flat = b.Flat.([]float32)
				}
			}
			if flat == nil {
				flat = make([]float32, workerTotalElems)
			}
			off := 0
			scores := flat[off : off+info.scratchSize]
			off += info.scratchSize
			qWork := flat[off : off+qBlockMaxElems]
			off += qBlockMaxElems
			outW := flat[off : off+qBlockMaxElems]
			off += qBlockMaxElems
			var kWork, vWork []float32
			if localKV {
				kWork = flat[off : off+kvBlockElems]
				off += kvBlockElems
				vWork = flat[off : off+kvBlockElems]
			}
			return halfWorker{
				buf:         buf,
				scores:      scores,
				qWork:       qWork,
				outW:        outW,
				kWork:       kWork,
				vWork:       vWork,
				lastBatchKV: &[2]int{-1, -1},
			}
		},
		func(w halfWorker) {
			if w.buf != nil && backend != nil {
				backend.PutBuffer(w.buf)
			}
		},
		func(w halfWorker, batchIdx, kvHeadIdx, qStart, qEnd int) {
			qLen := qEnd - qStart
			if qLen <= 0 {
				return
			}
			qLimit := info.seqLen
			kvLimit := info.kvLen
			if len(querySeqLen) > 0 {
				qLimit = max(0, min(int(querySeqLen[batchIdx]), info.seqLen))
			}
			if len(keyValueSeqLen) > 0 {
				kvLimit = max(0, min(int(keyValueSeqLen[batchIdx]), info.kvLen))
			}
			qLimitWork := max(0, qLimit-qStart)

			qOff := batchIdx*info.qBatchStride + kvHeadIdx*info.groupSize*info.qHeadStride
			kvOff := batchIdx*info.kvBatchStride + kvHeadIdx*info.kvHeadStride

			additiveMaskSlice, maskGroupStride := sdpaMaskBiasGroupSlice(additiveMask, batchIdx, kvHeadIdx, info.groupSize, maskBatchStride, maskHeadStride, maskSliceLen)
			booleanMaskSlice, boolGroupStride := sdpaMaskBiasGroupSlice(booleanMask, batchIdx, kvHeadIdx, info.groupSize, maskBatchStride, maskHeadStride, maskSliceLen)
			if len(booleanMaskSlice) > 0 {
				maskGroupStride = boolGroupStride
			}
			if maskSeqStride == 0 && maskGroupStride == 0 && kvLimit > 0 {
				if len(additiveMaskSlice) > 0 {
					additiveMaskSlice, kvLimit = sdpaTrimAdditiveKeyMaskFloat(additiveMaskSlice, kvLimit)
				} else if len(booleanMaskSlice) > 0 {
					booleanMaskSlice, kvLimit = sdpaTrimBoolKeyMask(booleanMaskSlice, kvLimit)
				}
			}
			if qStart > 0 && maskSeqStride > 0 {
				if len(additiveMaskSlice) > 0 {
					additiveMaskSlice = additiveMaskSlice[qStart*maskSeqStride:]
				}
				if len(booleanMaskSlice) > 0 {
					booleanMaskSlice = booleanMaskSlice[qStart*maskSeqStride:]
				}
			}

			additiveBiasSlice, biasGroupStride := sdpaMaskBiasGroupSlice(additiveBias, batchIdx, kvHeadIdx, info.groupSize, biasBatchStride, biasHeadStride, biasSliceLen)
			if qStart > 0 && biasSeqStride > 0 && len(additiveBiasSlice) > 0 {
				additiveBiasSlice = additiveBiasSlice[qStart*biasSeqStride:]
			}

			qGroupStrideWork := qLen * info.headDim
			qWork := w.qWork[:info.groupSize*qGroupStrideWork]
			outWork := w.outW[:info.groupSize*qGroupStrideWork]
			sdpaConvertHeadRowsToF32(qWork, qHalf, qOff+qStart*info.qSeqStride, info.qSeqStride, info.qHeadStride, info.groupSize, qLen, info.headDim, toF32)

			var kUse, vUse []float32
			var kvOffUse, kvSeqStrideUse int
			if localKV {
				if w.lastBatchKV[0] != batchIdx || w.lastBatchKV[1] != kvHeadIdx {
					sdpaConvertHeadRowsToF32(w.kWork, kHalf, kvOff, info.kvSeqStride, 0, 1, info.kvLen, info.headDim, toF32)
					sdpaConvertHeadRowsToF32(w.vWork, vHalf, kvOff, info.kvSeqStride, 0, 1, info.kvLen, info.headDim, toF32)
					w.lastBatchKV[0] = batchIdx
					w.lastBatchKV[1] = kvHeadIdx
				}
				kUse = w.kWork
				vUse = w.vWork
				kvOffUse = 0
				kvSeqStrideUse = info.headDim
			} else {
				kUse = kAllF32
				vUse = vAllF32
				kvOffUse = kvOff
				kvSeqStrideUse = info.kvSeqStride
			}

			handled := false
			if archFn != nil {
				handled = archFn(
					qWork, kUse, vUse, outWork,
					0, kvOffUse, info.headDim, kvSeqStrideUse, qGroupStrideWork,
					additiveMaskSlice, booleanMaskSlice, maskGroupStride,
					additiveBiasSlice, biasGroupStride,
					w.scores,
					info.groupSize, 0, qLen, info.kvLen, info.headDim,
					scale, causal,
					qLimitWork, kvLimit,
				)
			}
			if !handled {
				sdpaGeneric(
					qWork, kUse, vUse, 0, kvOffUse, info.headDim, kvSeqStrideUse, qGroupStrideWork,
					additiveMaskSlice, booleanMaskSlice, maskGroupStride, maskSeqStride,
					additiveBiasSlice, biasGroupStride, biasSeqStride,
					w.scores,
					outWork,
					info.groupSize, 0, qLen, info.kvLen, info.headDim, scale, causal,
					qLimitWork, kvLimit,
				)
			}
			sdpaConvertHeadRowsFromF32(outHalf, outWork, qOff+qStart*info.qSeqStride, info.qSeqStride, info.qHeadStride, info.groupSize, qLen, info.headDim, fromF32)
		},
	)
	return nil
}

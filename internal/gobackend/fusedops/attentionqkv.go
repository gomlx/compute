// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedAttentionQKVProjection.Register(FusedAttentionQKVProjection, gobackend.PriorityGeneric)
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeFusedAttentionQKVProjection] = execFusedAttentionQKVProjection
}

// qkvCachedConstant holds the pre-packed ConstantData containers (or split slices if small/generic)
// for a constant wQKV weight matrix in FusedAttentionQKVProjection.
type qkvCachedConstant struct {
	qConstData *gobackend.ConstantData
	kConstData *gobackend.ConstantData
	vConstData *gobackend.ConstantData
	wQ, wK, wV any
}

// nodeFusedAttentionQKVProjection stores parameters for the fused QKV projection.
// It does not implement gobackend.NodeDataComparable because multi-output nodes are not
// de-duplicated (see newMultiOutputsNode).
type nodeFusedAttentionQKVProjection struct {
	dtype         dtypes.DType
	inFeatures    int
	qDim          int
	kvDim         int
	hasBiasQ      bool
	hasBiasK      bool
	hasBiasV      bool
	wConstantData *gobackend.ConstantData
}

// EqualNodeData compares two node data structures for equality.
// The other value is guaranteed to be of type *nodeFusedAttentionQKVProjection.
func (d *nodeFusedAttentionQKVProjection) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedAttentionQKVProjection)
	return d.dtype == o.dtype &&
		d.inFeatures == o.inFeatures &&
		d.qDim == o.qDim &&
		d.kvDim == o.kvDim &&
		d.hasBiasQ == o.hasBiasQ &&
		d.hasBiasK == o.hasBiasK &&
		d.hasBiasV == o.hasBiasV
}

// Compile-time check that *nodeFusedAttentionQKVProjection implements gobackend.ConstantCacheNodeData.
var _ gobackend.ConstantCacheNodeData = (*nodeFusedAttentionQKVProjection)(nil)

func (d *nodeFusedAttentionQKVProjection) CacheSignature(inputIdx int) string {
	if d == nil || inputIdx != 1 {
		return ""
	}
	return fmt.Sprintf("QKVProj(dtype=%s,in=%d,q=%d,kv=%d)", d.dtype, d.inFeatures, d.qDim, d.kvDim)
}

func (d *nodeFusedAttentionQKVProjection) BindConstantData(inputIdx int, cd *gobackend.ConstantData) {
	if d == nil || inputIdx != 1 {
		return
	}
	d.wConstantData = cd
}

func (d *nodeFusedAttentionQKVProjection) IsInputCached(inputIdx int) bool {
	if d == nil || inputIdx != 1 {
		return false
	}
	return d.wConstantData != nil && d.wConstantData.CachedForm() != nil
}

// Recompute implements gobackend.RecomputableNodeData for nodeFusedAttentionQKVProjection.
func (d *nodeFusedAttentionQKVProjection) Recompute(_ *gobackend.Backend, _ []*gobackend.Node, _ *gobackend.Node) (any, error) {
	return d, nil
}

// FusedAttentionQKVProjection performs fused Query-Key-Value projection.
//
// It splits wQKV [inFeatures, qDim+2*kvDim] into per-projection weights and executes
// matmul.ExecuteWithEpilogue directly into the Q, K, and V output buffers with fused
// cache-resident bias addition, avoiding a large intermediate combined activation buffer
// and a separate post-matmul split+bias pass over DRAM.
func FusedAttentionQKVProjection(f *gobackend.Function, x, wQKV, biasQ, biasK, biasV compute.Value, queryDim, keyValueDim int) (queryOut, keyOut, valueOut compute.Value, err error) {
	if queryDim <= 0 || keyValueDim <= 0 {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: queryDim (%d) and keyValueDim (%d) must be > 0", queryDim, keyValueDim)
	}
	values := []compute.Value{x, wQKV}
	if biasQ != nil {
		values = append(values, biasQ)
	}
	if biasK != nil {
		values = append(values, biasK)
	}
	if biasV != nil {
		values = append(values, biasV)
	}
	inputs, err := f.VerifyAndCastValues("AttentionQKVProjection", values...)
	if err != nil {
		return nil, nil, nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	if xNode.Shape.Rank() < 1 {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: x must have rank >= 1, got %d", xNode.Shape.Rank())
	}
	if wNode.Shape.Rank() != 2 {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: wQKV must have rank 2, got %s", wNode.Shape)
	}
	if xNode.Shape.DType != wNode.Shape.DType {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: x (%s) and wQKV (%s) must have the same dtype", xNode.Shape.DType, wNode.Shape.DType)
	}
	inFeatures := xNode.Shape.Dimensions[xNode.Shape.Rank()-1]
	if inFeatures == shapes.DynamicDim {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: x's last dimension (inFeatures) cannot be dynamic")
	}
	totalOut := queryDim + 2*keyValueDim
	if wNode.Shape.Dimensions[0] != inFeatures || wNode.Shape.Dimensions[1] != totalOut {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: wQKV shape %s does not match [inFeatures=%d, queryDim+2*keyValueDim=%d]",
			wNode.Shape, inFeatures, totalOut)
	}

	batchDims := xNode.Shape.Dimensions[:xNode.Shape.Rank()-1]
	qDims := make([]int, len(batchDims)+1)
	copy(qDims, batchDims)
	qDims[len(batchDims)] = queryDim
	kvDims := make([]int, len(batchDims)+1)
	copy(kvDims, batchDims)
	kvDims[len(batchDims)] = keyValueDim

	var qShape, kShape, vShape shapes.Shape
	if xNode.Shape.IsDynamic() || len(xNode.Shape.AxisNames) > 0 {
		var qAxes, kvAxes []string
		if len(xNode.Shape.AxisNames) > 0 {
			batchAxes := xNode.Shape.AxisNames[:xNode.Shape.Rank()-1]
			qAxes = append(slices.Clone(batchAxes), "")
			kvAxes = append(slices.Clone(batchAxes), "")
		} else {
			qAxes = make([]string, len(qDims))
			kvAxes = make([]string, len(kvDims))
		}
		qShape = shapes.MakeDynamic(xNode.Shape.DType, qDims, qAxes)
		kShape = shapes.MakeDynamic(xNode.Shape.DType, kvDims, kvAxes)
		vShape = shapes.MakeDynamic(xNode.Shape.DType, kvDims, kvAxes)
	} else {
		qShape = shapes.Make(xNode.Shape.DType, qDims...)
		kShape = shapes.Make(xNode.Shape.DType, kvDims...)
		vShape = shapes.Make(xNode.Shape.DType, kvDims...)
	}

	data := &nodeFusedAttentionQKVProjection{
		dtype:      wNode.Shape.DType,
		inFeatures: inFeatures,
		qDim:       queryDim,
		kvDim:      keyValueDim,
		hasBiasQ:   biasQ != nil,
		hasBiasK:   biasK != nil,
		hasBiasV:   biasV != nil,
	}
	if wNode.IsConstant() {
		data.wConstantData = gobackend.NewAnonymousConstantData()
	}
	node := f.NewMultiOutputsNode(compute.OpTypeFusedAttentionQKVProjection, []shapes.Shape{qShape, kShape, vShape}, inputs...)
	node.Data = data
	queryOut = node.MultiOutputsNodes[0]
	keyOut = node.MultiOutputsNodes[1]
	valueOut = node.MultiOutputsNodes[2]
	return
}

// execFusedAttentionQKVProjection implements fused QKV projection.
// inputs[0]: x [batch..., inFeatures]
// inputs[1]: wQKV [inFeatures, qDim+2*kvDim]
// inputs[2..]: biasQ, biasK, biasV (optional, determined by node data flags)
// outputs: q [batch..., qDim], k [batch..., kvDim], v [batch..., kvDim]
func execFusedAttentionQKVProjection(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) ([]*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedAttentionQKVProjection)
	x := inputs[0]
	wQKV := inputs[1]

	// Determine bias buffers using flags from node data, not positional indexing.
	var biasQ, biasK, biasV *gobackend.Buffer
	biasIdx := 2
	if data.hasBiasQ {
		biasQ = inputs[biasIdx]
		biasIdx++
	}
	if data.hasBiasK {
		biasK = inputs[biasIdx]
		biasIdx++
	}
	if data.hasBiasV {
		biasV = inputs[biasIdx]
	}

	qShape, kShape, vShape := node.MultiOutputsShapes[0], node.MultiOutputsShapes[1], node.MultiOutputsShapes[2]
	qBuf, err := backend.GetBuffer(qShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", qShape)
	}
	kBuf, err := backend.GetBuffer(kShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", kShape)
	}
	vBuf, err := backend.GetBuffer(vShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", vShape)
	}
	if backend.NoOps {
		return []*gobackend.Buffer{qBuf, kBuf, vBuf}, nil
	}

	inFeatures := data.inFeatures
	qDim := data.qDim
	kvDim := data.kvDim

	switch x.RawShape.DType {
	case dtypes.Float32:
		if err := qkvExecMatmul[float32](backend, x, wQKV, biasQ, biasK, biasV, qBuf, kBuf, vBuf, inFeatures, qDim, kvDim, data.wConstantData); err != nil {
			return nil, err
		}
	case dtypes.Float64:
		if err := qkvExecMatmul[float64](backend, x, wQKV, biasQ, biasK, biasV, qBuf, kBuf, vBuf, inFeatures, qDim, kvDim, data.wConstantData); err != nil {
			return nil, err
		}
	case dtypes.Float16:
		if err := qkvExecMatmulHalf[float16.Float16](backend, x, wQKV, biasQ, biasK, biasV, qBuf, kBuf, vBuf, inFeatures, qDim, kvDim, data.wConstantData); err != nil {
			return nil, err
		}
	case dtypes.BFloat16:
		if err := qkvExecMatmulHalf[bfloat16.BFloat16](backend, x, wQKV, biasQ, biasK, biasV, qBuf, kBuf, vBuf, inFeatures, qDim, kvDim, data.wConstantData); err != nil {
			return nil, err
		}
	default:
		return nil, errors.Errorf("FusedAttentionQKVProjection: unsupported dtype %s", x.RawShape.DType)
	}

	return []*gobackend.Buffer{qBuf, kBuf, vBuf}, nil
}

func qkvSplitWeights[T any](wFlat, splitFlat []T, inFeatures, qDim, kvDim int) (wQ, wK, wV []T) {
	wQ = splitFlat[:inFeatures*qDim]
	wK = splitFlat[inFeatures*qDim : inFeatures*(qDim+kvDim)]
	wV = splitFlat[inFeatures*(qDim+kvDim) : inFeatures*(qDim+2*kvDim)]
	totalOut := qDim + 2*kvDim
	for r := range inFeatures {
		srcRow := wFlat[r*totalOut : (r+1)*totalOut]
		copy(wQ[r*qDim:(r+1)*qDim], srcRow[:qDim])
		copy(wK[r*kvDim:(r+1)*kvDim], srcRow[qDim:qDim+kvDim])
		copy(wV[r*kvDim:(r+1)*kvDim], srcRow[qDim+kvDim:])
	}
	return wQ, wK, wV
}

func qkvExecMatmul[T float32 | float64](
	backend *gobackend.Backend,
	xBuf, wQKVBuf, biasQBuf, biasKBuf, biasVBuf, qBuf, kBuf, vBuf *gobackend.Buffer,
	inFeatures, qDim, kvDim int,
	wConstantData *gobackend.ConstantData,
) error {
	xFlat := xBuf.Flat.([]T)
	qFlat := qBuf.Flat.([]T)
	kFlat := kBuf.Flat.([]T)
	vFlat := vBuf.Flat.([]T)
	lhsCrossSize := len(xFlat) / inFeatures
	if lhsCrossSize == 0 {
		return nil
	}

	var biasQ, biasK, biasV []T
	if biasQBuf != nil {
		biasQ = biasQBuf.Flat.([]T)
	}
	if biasKBuf != nil {
		biasK = biasKBuf.Flat.([]T)
	}
	if biasVBuf != nil {
		biasV = biasVBuf.Flat.([]T)
	}

	runThree := func(wQ, wK, wV []T, qDot, kDot, vDot *dot.NodeData) error {
		targets := [3]matmul.MultiRHSTarget[T, T]{
			{
				RHS:          wQ,
				RHSCrossSize: qDim,
				Output:       qFlat,
				Epilogue:     matmul.Epilogue[T]{Bias: biasQ},
				NodeData:     qDot,
			},
			{
				RHS:          wK,
				RHSCrossSize: kvDim,
				Output:       kFlat,
				Epilogue:     matmul.Epilogue[T]{Bias: biasK},
				NodeData:     kDot,
			},
			{
				RHS:          wV,
				RHSCrossSize: kvDim,
				Output:       vFlat,
				Epilogue:     matmul.Epilogue[T]{Bias: biasV},
				NodeData:     vDot,
			},
		}
		return matmul.ExecuteMultiRHSWithEpilogue(
			backend, dot.LayoutNonTransposed, xFlat,
			1, lhsCrossSize, inFeatures,
			targets[:],
		)
	}

	if wConstantData != nil {
		executed := false
		var execErr error
		cc := gobackend.GetOrSetCachedForm(wConstantData, func() *qkvCachedConstant {
			executed = true
			splitBuf, err := backend.GetBuffer(wQKVBuf.RawShape)
			if err != nil {
				execErr = err
				return nil
			}
			defer backend.PutBuffer(splitBuf)
			wQ, wK, wV := qkvSplitWeights(wQKVBuf.Flat.([]T), splitBuf.Flat.([]T), inFeatures, qDim, kvDim)
			qConst := gobackend.NewAnonymousConstantData()
			kConst := gobackend.NewAnonymousConstantData()
			vConst := gobackend.NewAnonymousConstantData()
			if err := runThree(
				wQ, wK, wV,
				&dot.NodeData{RHSConstantData: qConst},
				&dot.NodeData{RHSConstantData: kConst},
				&dot.NodeData{RHSConstantData: vConst},
			); err != nil {
				execErr = err
				return nil
			}
			res := &qkvCachedConstant{
				qConstData: qConst,
				kConstData: kConst,
				vConstData: vConst,
			}
			if qConst.CachedForm() == nil || kConst.CachedForm() == nil || vConst.CachedForm() == nil {
				res.wQ = slices.Clone(wQ)
				res.wK = slices.Clone(wK)
				res.wV = slices.Clone(wV)
			}
			return res
		})
		if execErr != nil {
			return execErr
		}
		if executed {
			return nil
		}
		var wQ, wK, wV []T
		if cc.wQ != nil {
			wQ = cc.wQ.([]T)
			wK = cc.wK.([]T)
			wV = cc.wV.([]T)
		}
		return runThree(
			wQ, wK, wV,
			&dot.NodeData{RHSConstantData: cc.qConstData},
			&dot.NodeData{RHSConstantData: cc.kConstData},
			&dot.NodeData{RHSConstantData: cc.vConstData},
		)
	}

	splitBuf, err := backend.GetBuffer(wQKVBuf.RawShape)
	if err != nil {
		return err
	}
	defer backend.PutBuffer(splitBuf)
	wQ, wK, wV := qkvSplitWeights(wQKVBuf.Flat.([]T), splitBuf.Flat.([]T), inFeatures, qDim, kvDim)
	return runThree(wQ, wK, wV, nil, nil, nil)
}

func qkvExecMatmulHalf[T gotype.HalfPrecision[T], P gotype.HalfPrecisionPtr[T]](
	backend *gobackend.Backend,
	xBuf, wQKVBuf, biasQBuf, biasKBuf, biasVBuf, qBuf, kBuf, vBuf *gobackend.Buffer,
	inFeatures, qDim, kvDim int,
	wConstantData *gobackend.ConstantData,
) error {
	xFlat := xBuf.Flat.([]T)
	qFlat := qBuf.Flat.([]T)
	kFlat := kBuf.Flat.([]T)
	vFlat := vBuf.Flat.([]T)
	lhsCrossSize := len(xFlat) / inFeatures
	if lhsCrossSize == 0 {
		return nil
	}

	toFloat32Bias := func(bBuf *gobackend.Buffer) []float32 {
		if bBuf == nil {
			return nil
		}
		bSlice := bBuf.Flat.([]T)
		out := make([]float32, len(bSlice))
		for i, v := range bSlice {
			out[i] = v.Float32()
		}
		return out
	}
	biasQ := toFloat32Bias(biasQBuf)
	biasK := toFloat32Bias(biasKBuf)
	biasV := toFloat32Bias(biasVBuf)

	totalOutDim := qDim + 2*kvDim
	tmpBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, lhsCrossSize*totalOutDim))
	if err != nil {
		return err
	}
	defer backend.PutBuffer(tmpBuf)
	tmpF32 := tmpBuf.Flat.([]float32)
	tmpQ := tmpF32[:lhsCrossSize*qDim]
	tmpK := tmpF32[lhsCrossSize*qDim : lhsCrossSize*(qDim+kvDim)]
	tmpV := tmpF32[lhsCrossSize*(qDim+kvDim) : lhsCrossSize*totalOutDim]

	runThree := func(wQ, wK, wV []T, qDot, kDot, vDot *dot.NodeData) error {
		targets := [3]matmul.MultiRHSTarget[T, float32]{
			{
				RHS:          wQ,
				RHSCrossSize: qDim,
				Output:       tmpQ,
				Epilogue:     matmul.Epilogue[float32]{Bias: biasQ},
				NodeData:     qDot,
			},
			{
				RHS:          wK,
				RHSCrossSize: kvDim,
				Output:       tmpK,
				Epilogue:     matmul.Epilogue[float32]{Bias: biasK},
				NodeData:     kDot,
			},
			{
				RHS:          wV,
				RHSCrossSize: kvDim,
				Output:       tmpV,
				Epilogue:     matmul.Epilogue[float32]{Bias: biasV},
				NodeData:     vDot,
			},
		}
		if err := matmul.ExecuteMultiRHSWithEpilogue(
			backend, dot.LayoutNonTransposed, xFlat,
			1, lhsCrossSize, inFeatures,
			targets[:],
		); err != nil {
			return err
		}
		for i, v := range tmpQ {
			P(&qFlat[i]).SetFloat32(v)
		}
		for i, v := range tmpK {
			P(&kFlat[i]).SetFloat32(v)
		}
		for i, v := range tmpV {
			P(&vFlat[i]).SetFloat32(v)
		}
		return nil
	}

	if wConstantData != nil {
		executed := false
		var execErr error
		cc := gobackend.GetOrSetCachedForm(wConstantData, func() *qkvCachedConstant {
			executed = true
			splitBuf, err := backend.GetBuffer(wQKVBuf.RawShape)
			if err != nil {
				execErr = err
				return nil
			}
			defer backend.PutBuffer(splitBuf)
			wQ, wK, wV := qkvSplitWeights(wQKVBuf.Flat.([]T), splitBuf.Flat.([]T), inFeatures, qDim, kvDim)
			qConst := gobackend.NewAnonymousConstantData()
			kConst := gobackend.NewAnonymousConstantData()
			vConst := gobackend.NewAnonymousConstantData()
			if err := runThree(
				wQ, wK, wV,
				&dot.NodeData{RHSConstantData: qConst},
				&dot.NodeData{RHSConstantData: kConst},
				&dot.NodeData{RHSConstantData: vConst},
			); err != nil {
				execErr = err
				return nil
			}
			res := &qkvCachedConstant{
				qConstData: qConst,
				kConstData: kConst,
				vConstData: vConst,
			}
			if qConst.CachedForm() == nil || kConst.CachedForm() == nil || vConst.CachedForm() == nil {
				res.wQ = slices.Clone(wQ)
				res.wK = slices.Clone(wK)
				res.wV = slices.Clone(wV)
			}
			return res
		})
		if execErr != nil {
			return execErr
		}
		if executed {
			return nil
		}
		var wQ, wK, wV []T
		if cc.wQ != nil {
			wQ = cc.wQ.([]T)
			wK = cc.wK.([]T)
			wV = cc.wV.([]T)
		}
		return runThree(
			wQ, wK, wV,
			&dot.NodeData{RHSConstantData: cc.qConstData},
			&dot.NodeData{RHSConstantData: cc.kConstData},
			&dot.NodeData{RHSConstantData: cc.vConstData},
		)
	}

	splitBuf, err := backend.GetBuffer(wQKVBuf.RawShape)
	if err != nil {
		return err
	}
	defer backend.PutBuffer(splitBuf)
	wQ, wK, wV := qkvSplitWeights(wQKVBuf.Flat.([]T), splitBuf.Flat.([]T), inFeatures, qDim, kvDim)
	return runThree(wQ, wK, wV, nil, nil, nil)
}

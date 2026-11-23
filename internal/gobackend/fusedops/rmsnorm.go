// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"math"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	gobackend.RegisterTransformation(90, &rmsNormFusion{})
}

type reduceAxesProvider interface {
	ReduceAxes() []int
}

// rmsNormFusion matches decomposed trailing-axis RMSNorm subgraphs (as emitted by
// gomlx/ml/layers/norm.RMSNorm) and fuses them into a single OpTypeFusedLayerNorm
// node with isRMSNorm=true.
//
// Matched pattern (after ImplicitBroadcastFusion at priority 100):
//
//	xNorm  = xOrig  (or ConvertDType(xOrig, Float32) for BFloat16/Float16)
//	x2     = Mul(xNorm, xNorm)
//	sum    = ReduceSum(x2, axis=rank-1)
//	mean   = Mul(sum, 1/N) | Div(sum, N | ConvertDType(DynamicDimensionSize(x2, rank-1)))
//	meanK  = Reshape/DynamicReshape(mean, [..., 1])   (optional)
//	varEps = Add(meanK, eps)                          (optional; eps=0 if omitted)
//	rms    = Sqrt(varEps)
//	normed = Div(xNorm, rms)
//	scaled = Mul(normed, Reshape/Broadcast([Add(ConvertDType(scale, F32), offset)])) (optional)
//	out    = ConvertDType(scaled, xOrig.DType)        (optional)
//
// Replacing ~11 separate ops per layer with a single L1-resident SIMD pass (and caching
// constant Float32 scales with offset baked in) removes ~1,900 op dispatches/token on
// 35-layer Gemma/Llama/Qwen transformers.
type rmsNormFusion struct{}

func (p *rmsNormFusion) Name() string { return "RMSNormFusion" }

func (p *rmsNormFusion) Apply(b *gobackend.Builder) (bool, error) {
	rewired := false
	for _, f := range b.Functions {
		if fuseRMSNormInFunction(f) {
			rewired = true
		}
	}
	return rewired, nil
}

func extractScalarFloat64(n *gobackend.Node) (float64, bool) {
	for n != nil && n.OpType == compute.OpTypeConvertDType && len(n.Inputs) == 1 {
		n = n.Inputs[0]
	}
	if n == nil || n.OpType != compute.OpTypeConstant || n.Shape.Size() != 1 {
		return 0, false
	}
	cd, ok := n.Data.(*gobackend.ConstantData)
	if !ok || cd == nil {
		return 0, false
	}
	buf := cd.Buffer()
	if buf == nil || buf.Flat == nil {
		return 0, false
	}
	switch v := buf.Flat.(type) {
	case []float32:
		return float64(v[0]), true
	case []float64:
		return v[0], true
	case []bfloat16.BFloat16:
		return float64(v[0].Float32()), true
	case []float16.Float16:
		return float64(v[0].Float32()), true
	case []int64:
		return float64(v[0]), true
	case []int32:
		return float64(v[0]), true
	default:
		return 0, false
	}
}

func unwrapReshapeOrBroadcast(n *gobackend.Node) *gobackend.Node {
	for n != nil && len(n.Inputs) >= 1 &&
		(n.OpType == compute.OpTypeReshape ||
			n.OpType == compute.OpTypeDynamicReshape ||
			n.OpType == compute.OpTypeBroadcastInDim ||
			n.OpType == compute.OpTypeDynamicBroadcastInDim) {
		n = n.Inputs[0]
	}
	return n
}

func isDynamicDimSizeOfAxis(n *gobackend.Node, target *gobackend.Node, axis int) bool {
	n = unwrapReshapeOrBroadcast(n)
	if n == nil {
		return false
	}
	if n.OpType == compute.OpTypeConvertDType && len(n.Inputs) == 1 {
		n = n.Inputs[0]
	}
	if n.OpType != compute.OpTypeDynamicDimensionSize || len(n.Inputs) != 1 {
		return false
	}
	nodeAxis, ok := n.Data.(int)
	if !ok || nodeAxis != axis {
		return false
	}
	// Accept any node whose trailing axis matches target's trailing axis (x2, xNorm, xOrig).
	in := n.Inputs[0]
	return in == target || (in.Shape.Rank() == target.Shape.Rank() &&
		in.Shape.Dimensions[axis] == target.Shape.Dimensions[axis])
}

func unwrapScaleExpr(scaleExpr *gobackend.Node, normSize int, normDType dtypes.DType) (scaleInput *gobackend.Node, scaleOffset float64, ok bool) {
	if scaleExpr == nil || scaleExpr.Shape.Rank() < 1 || scaleExpr.Shape.Dimensions[scaleExpr.Shape.Rank()-1] != normSize {
		return nil, 0, false
	}

	cur := scaleExpr
	for cur != nil && len(cur.Inputs) >= 1 &&
		(cur.OpType == compute.OpTypeReshape ||
			cur.OpType == compute.OpTypeDynamicReshape ||
			cur.OpType == compute.OpTypeBroadcastInDim ||
			cur.OpType == compute.OpTypeDynamicBroadcastInDim) {
		next := cur.Inputs[0]
		if !next.Shape.IsDynamic() && next.Shape.Size() != normSize {
			return nil, 0, false
		}
		cur = next
	}
	if cur == nil || cur.Shape.IsDynamic() || cur.Shape.Size() != normSize {
		return nil, 0, false
	}

	// Optional Add(scale, scalarOffset) — e.g. Gemma 1/2/3 WithScaleOffset(1.0).
	if cur.OpType == compute.OpTypeAdd && len(cur.Inputs) == 2 {
		if off, isConst := extractScalarFloat64(unwrapReshapeOrBroadcast(cur.Inputs[1])); isConst && cur.Inputs[0].Shape.Size() == normSize {
			scaleOffset = off
			cur = cur.Inputs[0]
		} else if off, isConst := extractScalarFloat64(unwrapReshapeOrBroadcast(cur.Inputs[0])); isConst && cur.Inputs[1].Shape.Size() == normSize {
			scaleOffset = off
			cur = cur.Inputs[1]
		}
	}

	// Optional ConvertDType(scaleOrig, normDType).
	if cur.OpType == compute.OpTypeConvertDType && len(cur.Inputs) == 1 && cur.Shape.DType == normDType {
		srcDT := cur.Inputs[0].Shape.DType
		if srcDT == dtypes.BFloat16 || srcDT == dtypes.Float16 || srcDT == dtypes.Float32 || srcDT == dtypes.Float64 {
			cur = cur.Inputs[0]
		}
	}

	if cur.Shape.IsDynamic() || cur.Shape.Size() != normSize {
		return nil, 0, false
	}
	dt := cur.Shape.DType
	if dt != dtypes.BFloat16 && dt != dtypes.Float16 && dt != dtypes.Float32 && dt != dtypes.Float64 {
		return nil, 0, false
	}
	return cur, scaleOffset, true
}

func fuseRMSNormInFunction(f *gobackend.Function) bool {
	if len(f.Nodes) == 0 {
		return false
	}

	// Build consumer counts and output membership so we only fold single-consumer tail ops.
	consumers := make([][]*gobackend.Node, len(f.Nodes))
	isOutput := make([]bool, len(f.Nodes))
	for _, out := range f.Outputs {
		if out != nil && out.Index >= 0 && out.Index < len(isOutput) {
			isOutput[out.Index] = true
		}
	}
	for _, node := range f.Nodes {
		if node == nil {
			continue
		}
		for _, in := range node.Inputs {
			if in != nil && in.Function == f && in.Index >= 0 && in.Index < len(consumers) {
				consumers[in.Index] = append(consumers[in.Index], node)
			}
		}
		for _, captures := range node.CapturedInputs {
			for _, in := range captures {
				if in != nil && in.Function == f && in.Index >= 0 && in.Index < len(consumers) {
					consumers[in.Index] = append(consumers[in.Index], node)
				}
			}
		}
	}

	rewired := false
	for _, divNode := range f.Nodes {
		if divNode == nil || divNode.OpType != compute.OpTypeDiv || len(divNode.Inputs) != 2 {
			continue
		}
		xNorm := divNode.Inputs[0]
		sqrtNode := unwrapReshapeOrBroadcast(divNode.Inputs[1])
		if xNorm == nil || sqrtNode == nil || sqrtNode.OpType != compute.OpTypeSqrt || len(sqrtNode.Inputs) != 1 {
			continue
		}
		normDType := xNorm.Shape.DType
		if normDType != dtypes.Float32 && normDType != dtypes.Float64 {
			continue
		}
		rank := xNorm.Shape.Rank()
		if rank < 1 {
			continue
		}
		normSize := xNorm.Shape.Dimensions[rank-1]
		if normSize <= 0 {
			// Trailing axis must be statically known (> 0). Leading axes may be dynamic.
			continue
		}

		// Match optional Add(mean, epsilon) before Sqrt.
		epsilon := 0.0
		meanShaped := sqrtNode.Inputs[0]
		if meanShaped.OpType == compute.OpTypeAdd && len(meanShaped.Inputs) == 2 {
			if eps, ok := extractScalarFloat64(unwrapReshapeOrBroadcast(meanShaped.Inputs[1])); ok {
				epsilon = eps
				meanShaped = meanShaped.Inputs[0]
			} else if eps, ok := extractScalarFloat64(unwrapReshapeOrBroadcast(meanShaped.Inputs[0])); ok {
				epsilon = eps
				meanShaped = meanShaped.Inputs[1]
			}
		}

		// Verify meanShaped has trailing dim 1 (from ReduceAndKeep) before unwrapping Reshape/Broadcast.
		if meanShaped.Shape.Rank() != rank || meanShaped.Shape.Dimensions[rank-1] != 1 {
			continue
		}
		meanNode := unwrapReshapeOrBroadcast(meanShaped)
		if meanNode == nil {
			continue
		}

		// Match mean = Mul(sum, 1/normSize) or Div(sum, normSize | DynamicDimensionSize(rank-1)).
		var sumNode *gobackend.Node
		switch meanNode.OpType {
		case compute.OpTypeMul:
			if len(meanNode.Inputs) != 2 {
				continue
			}
			invN := 1.0 / float64(normSize)
			if c, ok := extractScalarFloat64(unwrapReshapeOrBroadcast(meanNode.Inputs[1])); ok && math.Abs(c-invN) <= 1e-5*invN {
				sumNode = meanNode.Inputs[0]
			} else if c, ok := extractScalarFloat64(unwrapReshapeOrBroadcast(meanNode.Inputs[0])); ok && math.Abs(c-invN) <= 1e-5*invN {
				sumNode = meanNode.Inputs[1]
			}
		case compute.OpTypeDiv:
			if len(meanNode.Inputs) != 2 {
				continue
			}
			cand := meanNode.Inputs[0]
			denom := unwrapReshapeOrBroadcast(meanNode.Inputs[1])
			if c, ok := extractScalarFloat64(denom); ok && math.Abs(c-float64(normSize)) <= 1e-5*float64(normSize) {
				sumNode = cand
			} else if isDynamicDimSizeOfAxis(denom, xNorm, rank-1) {
				sumNode = cand
			}
		default:
			continue
		}
		if sumNode == nil || sumNode.OpType != compute.OpTypeReduceSum || len(sumNode.Inputs) != 1 {
			continue
		}
		// Verify ReduceSum reduces ONLY the trailing axis (rank-1).
		axesProv, ok := sumNode.Data.(reduceAxesProvider)
		if !ok {
			continue
		}
		redAxes := axesProv.ReduceAxes()
		if len(redAxes) != 1 || redAxes[0] != rank-1 {
			continue
		}

		// Verify sumNode.Inputs[0] is Mul(xNorm, xNorm).
		x2 := sumNode.Inputs[0]
		if x2 == nil || x2.OpType != compute.OpTypeMul || len(x2.Inputs) != 2 || x2.Inputs[0] != xNorm || x2.Inputs[1] != xNorm {
			continue
		}

		// Unwrap optional leading ConvertDType(xOrig, Float32) for half-precision inputs.
		xOrig := xNorm
		if xNorm.OpType == compute.OpTypeConvertDType && len(xNorm.Inputs) == 1 && normDType == dtypes.Float32 {
			srcDT := xNorm.Inputs[0].Shape.DType
			if srcDT == dtypes.BFloat16 || srcDT == dtypes.Float16 {
				xOrig = xNorm.Inputs[0]
			}
		}

		// Walk forward to fold optional scale Mul and optional trailing ConvertDType.
		tailNode := divNode
		var scaleInput *gobackend.Node
		scaleOffset := 0.0
		if !isOutput[tailNode.Index] && len(consumers[tailNode.Index]) == 1 {
			c := consumers[tailNode.Index][0]
			if c.OpType == compute.OpTypeMul && len(c.Inputs) == 2 {
				other := c.Inputs[1]
				if other == tailNode {
					other = c.Inputs[0]
				}
				if sIn, sOff, sOK := unwrapScaleExpr(other, normSize, normDType); sOK {
					scaleInput = sIn
					scaleOffset = sOff
					tailNode = c
				}
			}
		}

		if !isOutput[tailNode.Index] && len(consumers[tailNode.Index]) == 1 {
			c := consumers[tailNode.Index][0]
			if c.OpType == compute.OpTypeConvertDType && len(c.Inputs) == 1 && c.Inputs[0] == tailNode {
				dstDT := c.Shape.DType
				if dstDT == xOrig.Shape.DType && (dstDT == dtypes.BFloat16 || dstDT == dtypes.Float16) {
					tailNode = c
				}
			}
		}

		// Rewrite tailNode in-place as a fused RMSNorm (using OpTypeFusedLayerNorm with isRMSNorm=true).
		newInputs := []*gobackend.Node{xOrig}
		if scaleInput != nil {
			newInputs = append(newInputs, scaleInput)
		}
		tailNode.OpType = compute.OpTypeFusedLayerNorm
		tailNode.Inputs = newInputs
		tailNode.Data = &nodeFusedLayerNorm{
			axes:         []int{rank - 1},
			epsilon:      epsilon,
			hasGamma:     scaleInput != nil,
			isRMSNorm:    true,
			scaleOffset:  scaleOffset,
			scaleIsConst: scaleInput != nil && scaleInput.OpType == compute.OpTypeConstant,
		}
		rewired = true
	}
	return rewired
}

const (
	rmsNormStackCap       = 4096
	rmsNormMinParallelLen = 16384
)

// Hooks overridden by rmsnorm_simd.go when compiled with GOEXPERIMENT=simd.
var (
	rmsNormRowFloat32Fn = rmsNormRowFloat32Scalar
	rmsNormRowFloat64Fn = rmsNormRowFloat64Scalar
)

func rmsNormRowFloat32Scalar(in, out, scale []float32, epsilon float64) {
	n := len(in)
	var sumSq float32
	for i := range n {
		v := in[i]
		sumSq += v * v
	}
	invRMS := float32(1.0 / math.Sqrt(float64(sumSq)/float64(n)+epsilon))
	if scale != nil {
		for i := range n {
			out[i] = in[i] * invRMS * scale[i]
		}
	} else {
		for i := range n {
			out[i] = in[i] * invRMS
		}
	}
}

func rmsNormRowFloat64Scalar(in, out, scale []float64, epsilon float64) {
	n := len(in)
	var sumSq float64
	for i := range n {
		v := in[i]
		sumSq += v * v
	}
	invRMS := 1.0 / math.Sqrt(sumSq/float64(n)+epsilon)
	if scale != nil {
		for i := range n {
			out[i] = in[i] * invRMS * scale[i]
		}
	} else {
		for i := range n {
			out[i] = in[i] * invRMS
		}
	}
}

func bf16ToF32Slice(conv gobackend.HalfSliceConverters, dst []float32, src []bfloat16.BFloat16) {
	if conv.BFloat16ToFloat32 != nil {
		conv.BFloat16ToFloat32(dst, src)
		return
	}
	for i, v := range src {
		dst[i] = v.Float32()
	}
}

func f32ToBF16Slice(conv gobackend.HalfSliceConverters, dst []bfloat16.BFloat16, src []float32) {
	if conv.Float32ToBFloat16 != nil {
		conv.Float32ToBFloat16(dst, src)
		return
	}
	for i, v := range src {
		dst[i] = bfloat16.FromFloat32(v)
	}
}

func f16ToF32Slice(conv gobackend.HalfSliceConverters, dst []float32, src []float16.Float16) {
	if conv.Float16ToFloat32 != nil {
		conv.Float16ToFloat32(dst, src)
		return
	}
	for i, v := range src {
		dst[i] = v.Float32()
	}
}

func f32ToF16Slice(conv gobackend.HalfSliceConverters, dst []float16.Float16, src []float32) {
	if conv.Float32ToFloat16 != nil {
		conv.Float32ToFloat16(dst, src)
		return
	}
	for i, v := range src {
		dst[i] = float16.FromFloat32(v)
	}
}

func extractScaleFloat32WithOffset(buf *gobackend.Buffer, offset float64, scratch []float32, conv gobackend.HalfSliceConverters) []float32 {
	if buf == nil {
		return nil
	}
	n := buf.RawShape.Size()
	var dst []float32
	if n <= len(scratch) {
		dst = scratch[:n]
	} else {
		dst = make([]float32, n)
	}
	switch s := buf.Flat.(type) {
	case []float32:
		if offset == 0 {
			return s[:n]
		}
		off := float32(offset)
		for i := range n {
			dst[i] = s[i] + off
		}
	case []bfloat16.BFloat16:
		bf16ToF32Slice(conv, dst, s[:n])
		if offset != 0 {
			off := float32(offset)
			for i := range n {
				dst[i] += off
			}
		}
	case []float16.Float16:
		f16ToF32Slice(conv, dst, s[:n])
		if offset != 0 {
			off := float32(offset)
			for i := range n {
				dst[i] += off
			}
		}
	case []float64:
		for i := range n {
			dst[i] = float32(s[i] + offset)
		}
	}
	return dst
}

func extractScaleFloat64WithOffset(buf *gobackend.Buffer, offset float64, scratch []float64) []float64 {
	if buf == nil {
		return nil
	}
	n := buf.RawShape.Size()
	var dst []float64
	if n <= len(scratch) {
		dst = scratch[:n]
	} else {
		dst = make([]float64, n)
	}
	switch s := buf.Flat.(type) {
	case []float64:
		if offset == 0 {
			return s[:n]
		}
		for i := range n {
			dst[i] = s[i] + offset
		}
	case []float32:
		for i := range n {
			dst[i] = float64(s[i]) + offset
		}
	case []bfloat16.BFloat16:
		for i := range n {
			dst[i] = float64(s[i].Float32()) + offset
		}
	case []float16.Float16:
		for i := range n {
			dst[i] = float64(s[i].Float32()) + offset
		}
	}
	return dst
}

var rmsNormRowPool = sync.Pool{
	New: func() any {
		buf := make([]float32, 0, rmsNormStackCap)
		return &buf
	},
}

func getRMSNormRowF32(n int) (*[]float32, []float32) {
	p := rmsNormRowPool.Get().(*[]float32)
	if cap(*p) < n {
		*p = make([]float32, n)
	}
	return p, (*p)[:n]
}

func execFusedRMSNorm(backend *gobackend.Backend, node *gobackend.Node, data *nodeFusedLayerNorm, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input := inputs[0]
	outShape := node.Shape
	if outShape.IsDynamic() {
		outShape = input.RawShape.Clone()
		outShape.DType = node.Shape.DType
	}

	var output *gobackend.Buffer
	var err error
	if len(inputsOwned) > 0 && inputsOwned[0] && input.RawShape.Equal(outShape) {
		output = input
		inputs[0] = nil
	} else {
		output, err = backend.GetBuffer(outShape)
		if err != nil {
			return nil, err
		}
	}
	totalElems := input.RawShape.Size()
	if backend.NoOps || totalElems == 0 {
		return output, nil
	}

	rank := len(input.RawShape.Dimensions)
	normSize := input.RawShape.Dimensions[rank-1]
	outerSize := totalElems / normSize
	inDType := input.RawShape.DType
	outDType := output.RawShape.DType

	var scaleBuf *gobackend.Buffer
	if data.hasGamma && len(inputs) > 1 {
		scaleBuf = inputs[1]
	}

	canParallel := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		!backend.Scheduler.IsBusy() && backend.Scheduler.EffectiveParallelism() > 1 &&
		outerSize > 1 && totalElems >= rmsNormMinParallelLen

	// Float64 path
	if inDType == dtypes.Float64 {
		var scale []float64
		if scaleBuf != nil {
			if data.scaleIsConst {
				data.scaleCacheOnce.Do(func() {
					data.cachedScaleF64 = extractScaleFloat64WithOffset(scaleBuf, data.scaleOffset, make([]float64, scaleBuf.RawShape.Size()))
				})
				scale = data.cachedScaleF64
			} else {
				scale = extractScaleFloat64WithOffset(scaleBuf, data.scaleOffset, nil)
			}
		}
		inData := input.Flat.([]float64)
		outData := output.Flat.([]float64)
		runRows := func(rStart, rEnd int) {
			for r := rStart; r < rEnd; r++ {
				off := r * normSize
				rmsNormRowFloat64Fn(inData[off:off+normSize], outData[off:off+normSize], scale, data.epsilon)
			}
		}
		if canParallel {
			minRows := max(1, (rmsNormMinParallelLen/2)/max(1, normSize))
			backend.Scheduler.ParallelizeChunks(outerSize, minRows, runRows)
		} else {
			runRows(0, outerSize)
		}
		return output, nil
	}

	// Float32 / BFloat16 / Float16 path (all compute RMS in Float32).
	conv := gobackend.ActiveHalfSliceConverters()
	var scale []float32
	if scaleBuf != nil {
		if data.scaleIsConst {
			data.scaleCacheOnce.Do(func() {
				data.cachedScaleF32 = extractScaleFloat32WithOffset(scaleBuf, data.scaleOffset, make([]float32, scaleBuf.RawShape.Size()), conv)
			})
			scale = data.cachedScaleF32
		} else {
			pScale, scaleBufSlice := getRMSNormRowF32(scaleBuf.RawShape.Size())
			defer rmsNormRowPool.Put(pScale)
			scale = extractScaleFloat32WithOffset(scaleBuf, data.scaleOffset, scaleBufSlice, conv)
		}
	}

	if inDType == dtypes.Float32 && outDType == dtypes.Float32 {
		inData := input.Flat.([]float32)
		outData := output.Flat.([]float32)
		runRows := func(rStart, rEnd int) {
			for r := rStart; r < rEnd; r++ {
				off := r * normSize
				rmsNormRowFloat32Fn(inData[off:off+normSize], outData[off:off+normSize], scale, data.epsilon)
			}
		}
		if canParallel {
			minRows := max(1, (rmsNormMinParallelLen/2)/max(1, normSize))
			backend.Scheduler.ParallelizeChunks(outerSize, minRows, runRows)
		} else {
			runRows(0, outerSize)
		}
		return output, nil
	}

	// Half-precision input (BFloat16 or Float16), output either same half-precision or Float32.
	runHalfRows := func(rStart, rEnd int) {
		var pRow *[]float32
		var rowF32 []float32
		if outDType != dtypes.Float32 {
			pRow, rowF32 = getRMSNormRowF32(normSize)
			defer rmsNormRowPool.Put(pRow)
		}

		switch inDType {
		case dtypes.BFloat16:
			inBF := input.Flat.([]bfloat16.BFloat16)
			if outDType == dtypes.BFloat16 {
				outBF := output.Flat.([]bfloat16.BFloat16)
				for r := rStart; r < rEnd; r++ {
					off := r * normSize
					bf16ToF32Slice(conv, rowF32, inBF[off:off+normSize])
					rmsNormRowFloat32Fn(rowF32, rowF32, scale, data.epsilon)
					f32ToBF16Slice(conv, outBF[off:off+normSize], rowF32)
				}
			} else {
				outF32 := output.Flat.([]float32)
				for r := rStart; r < rEnd; r++ {
					off := r * normSize
					dst := outF32[off : off+normSize]
					bf16ToF32Slice(conv, dst, inBF[off:off+normSize])
					rmsNormRowFloat32Fn(dst, dst, scale, data.epsilon)
				}
			}
		case dtypes.Float16:
			inF16 := input.Flat.([]float16.Float16)
			if outDType == dtypes.Float16 {
				outF16 := output.Flat.([]float16.Float16)
				for r := rStart; r < rEnd; r++ {
					off := r * normSize
					f16ToF32Slice(conv, rowF32, inF16[off:off+normSize])
					rmsNormRowFloat32Fn(rowF32, rowF32, scale, data.epsilon)
					f32ToF16Slice(conv, outF16[off:off+normSize], rowF32)
				}
			} else {
				outF32 := output.Flat.([]float32)
				for r := rStart; r < rEnd; r++ {
					off := r * normSize
					dst := outF32[off : off+normSize]
					f16ToF32Slice(conv, dst, inF16[off:off+normSize])
					rmsNormRowFloat32Fn(dst, dst, scale, data.epsilon)
				}
			}
		}
	}

	if canParallel {
		minRows := max(1, (rmsNormMinParallelLen/2)/max(1, normSize))
		backend.Scheduler.ParallelizeChunks(outerSize, minRows, runHalfRows)
	} else {
		runHalfRows(0, outerSize)
	}
	return output, nil
}

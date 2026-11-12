// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedQuantizedDense.Register(FusedQuantizedDense, gobackend.PriorityGeneric)
	gobackend.RegisterFusedQuantizedDenseVJP.Register(FusedQuantizedDenseVJP, gobackend.PriorityGeneric)
	gobackend.RegisterQuantizedEmbeddingLookup.Register(QuantizedEmbeddingLookup, gobackend.PriorityGeneric)

	gobackend.SetNodeExecutor(compute.OpTypeFusedQuantizedDense, gobackend.PriorityTyped, execFusedQuantizedDense)
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeFusedQuantizedDenseVJP] = execFusedQuantizedDenseVJP
	gobackend.SetNodeExecutor(compute.OpTypeQuantizedEmbeddingLookup, gobackend.PriorityTyped, execQuantizedEmbeddingLookup)
}

// nodeFusedQuantizedDense stores compile-time validated metadata for FusedQuantizedDense.
//
// Node inputs order:
//
//	[x, weight, wScale?, wZeroPoint?, xScale?, xZeroPoint?, outScale?, outZeroPoint?, bias?]
type nodeFusedQuantizedDense struct {
	weightScheme     compute.QuantizationScheme
	weightGGMLType   compute.GGMLQuantType
	weightLayout     compute.DenseLayout
	hasXQuant        bool
	xScheme          compute.QuantizationScheme
	xGGMLType        compute.GGMLQuantType
	hasOutputQuant   bool
	outputQuantDType dtypes.DType
	quantizedMatmuls bool
	activation       compute.ActivationConfig

	hasWScale   bool
	hasWZP      bool
	hasXScale   bool
	hasXZP      bool
	hasOutScale bool
	hasOutZP    bool
	hasBias     bool

	info shapeinference.QuantizedDenseInfo
}

var _ gobackend.FusedDenseActivationTarget = (*nodeFusedQuantizedDense)(nil)

func (d *nodeFusedQuantizedDense) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedQuantizedDense)
	return d.weightScheme == o.weightScheme &&
		d.weightGGMLType == o.weightGGMLType &&
		d.weightLayout == o.weightLayout &&
		d.hasXQuant == o.hasXQuant &&
		d.xScheme == o.xScheme &&
		d.xGGMLType == o.xGGMLType &&
		d.hasOutputQuant == o.hasOutputQuant &&
		d.outputQuantDType == o.outputQuantDType &&
		d.quantizedMatmuls == o.quantizedMatmuls &&
		d.activation == o.activation &&
		d.hasWScale == o.hasWScale &&
		d.hasWZP == o.hasWZP &&
		d.hasXScale == o.hasXScale &&
		d.hasXZP == o.hasXZP &&
		d.hasOutScale == o.hasOutScale &&
		d.hasOutZP == o.hasOutZP &&
		d.hasBias == o.hasBias &&
		d.info.K == o.info.K &&
		d.info.N == o.info.N &&
		d.info.FloatComputeDType == o.info.FloatComputeDType &&
		d.info.WeightScaleLayout == o.info.WeightScaleLayout &&
		d.info.WeightZeroPointLayout == o.info.WeightZeroPointLayout &&
		d.info.XScaleLayout == o.info.XScaleLayout &&
		d.info.XZeroPointLayout == o.info.XZeroPointLayout &&
		d.info.OutputScaleLayout == o.info.OutputScaleLayout &&
		d.info.OutputZeroPointLayout == o.info.OutputZeroPointLayout
}

// TryFuseActivation implements gobackend.FusedDenseActivationTarget so single-consumer
// FusedQuantizedDense(ActivationNone) -> FusedActivation(cfg) pairs fuse automatically.
func (d *nodeFusedQuantizedDense) TryFuseActivation(cfg compute.ActivationConfig) bool {
	if d == nil || d.hasOutputQuant || d.activation.Type != compute.ActivationNone {
		return false
	}
	if cfg.Type == compute.ActivationNone || cfg.Type == compute.ActivationSwiGLU {
		return false
	}
	d.activation = cfg
	return true
}

// quantizationToShapes converts a *compute.Quantization into a *shapeinference.QuantizationShapes,
// verifying and casting any non-nil Scale / ZeroPoint compute.Value nodes along the way.
func extractQuantizationShapes(
	f *gobackend.Function,
	opName string,
	q *compute.Quantization,
) (qs *shapeinference.QuantizationShapes, scaleNode, zpNode *gobackend.Node, err error) {
	if q == nil {
		return nil, nil, nil, nil
	}
	qs = &shapeinference.QuantizationShapes{
		Scheme:         q.Scheme,
		BlockAxis:      q.BlockAxis,
		BlockSize:      q.BlockSize,
		GGMLType:       q.GGMLType,
		QuantizedDType: q.QuantizedDType,
	}
	if q.Scale != nil {
		nodes, err := f.VerifyAndCastValues(opName, q.Scale)
		if err != nil {
			return nil, nil, nil, err
		}
		scaleNode = nodes[0]
		qs.Scale = scaleNode.Shape
	}
	if q.ZeroPoint != nil {
		nodes, err := f.VerifyAndCastValues(opName, q.ZeroPoint)
		if err != nil {
			return nil, nil, nil, err
		}
		zpNode = nodes[0]
		qs.ZeroPoint = zpNode.Shape
	}
	return qs, scaleNode, zpNode, nil
}

// FusedQuantizedDense builds an OpTypeFusedQuantizedDense node in the Go backend graph.
func FusedQuantizedDense(
	f *gobackend.Function,
	x, weight, bias compute.Value,
	options compute.FusedQuantizedDenseConfig,
) (compute.Value, error) {
	if options.WeightQuantization == nil {
		return nil, errors.Errorf("FusedQuantizedDense: options.WeightQuantization must not be nil")
	}

	baseInputs, err := f.VerifyAndCastValues("FusedQuantizedDense", x, weight)
	if err != nil {
		return nil, err
	}
	xNode, wNode := baseInputs[0], baseInputs[1]

	wqs, wScaleNode, wZPNode, err := extractQuantizationShapes(f, "FusedQuantizedDense", options.WeightQuantization)
	if err != nil {
		return nil, err
	}
	xqs, xScaleNode, xZPNode, err := extractQuantizationShapes(f, "FusedQuantizedDense", options.XQuantization)
	if err != nil {
		return nil, err
	}
	oqs, oScaleNode, oZPNode, err := extractQuantizationShapes(f, "FusedQuantizedDense", options.OutputQuantization)
	if err != nil {
		return nil, err
	}

	var biasNode *gobackend.Node
	var biasShape shapes.Shape
	if bias != nil {
		bNodes, err := f.VerifyAndCastValues("FusedQuantizedDense", bias)
		if err != nil {
			return nil, err
		}
		biasNode = bNodes[0]
		biasShape = biasNode.Shape
	}

	cfgShapes := shapeinference.QuantizedDenseConfigShapes{
		WeightQuantization: wqs,
		XQuantization:      xqs,
		OutputQuantization: oqs,
		QuantizedMatmuls:   options.QuantizedMatmuls,
		Activation:         options.Activation,
		WeightLayout:       options.WeightLayout,
	}
	info, err := shapeinference.FusedQuantizedDenseWithInfo(xNode.Shape, wNode.Shape, biasShape, cfgShapes)
	if err != nil {
		return nil, err
	}

	// Extra GGML implementation check so unsupported enum values fail at build time.
	if wqs.Scheme == compute.QuantGGML {
		if _, err := ggmlDequantFunc(wqs.GGMLType); err != nil {
			return nil, err
		}
	}

	inputs := make([]*gobackend.Node, 0, 9)
	inputs = append(inputs, xNode, wNode)
	if wScaleNode != nil {
		inputs = append(inputs, wScaleNode)
	}
	if wZPNode != nil {
		inputs = append(inputs, wZPNode)
	}
	if xScaleNode != nil {
		inputs = append(inputs, xScaleNode)
	}
	if xZPNode != nil {
		inputs = append(inputs, xZPNode)
	}
	if oScaleNode != nil {
		inputs = append(inputs, oScaleNode)
	}
	if oZPNode != nil {
		inputs = append(inputs, oZPNode)
	}
	if biasNode != nil {
		inputs = append(inputs, biasNode)
	}

	data := &nodeFusedQuantizedDense{
		weightScheme:     wqs.Scheme,
		weightGGMLType:   wqs.GGMLType,
		weightLayout:     options.WeightLayout,
		hasXQuant:        xqs != nil,
		hasOutputQuant:   oqs != nil,
		outputQuantDType: info.OutputShape.DType,
		quantizedMatmuls: options.QuantizedMatmuls,
		activation:       options.Activation,
		hasWScale:        wScaleNode != nil,
		hasWZP:           wZPNode != nil,
		hasXScale:        xScaleNode != nil,
		hasXZP:           xZPNode != nil,
		hasOutScale:      oScaleNode != nil,
		hasOutZP:         oZPNode != nil,
		hasBias:          biasNode != nil,
		info:             info,
	}
	if xqs != nil {
		data.xScheme = xqs.Scheme
		data.xGGMLType = xqs.GGMLType
	}

	node, _ := f.GetOrCreateNode(compute.OpTypeFusedQuantizedDense, info.OutputShape, inputs, data)
	return node, nil
}

type unpackedQDInputs struct {
	x        *gobackend.Buffer
	weight   *gobackend.Buffer
	wScale   *gobackend.Buffer
	wZP      *gobackend.Buffer
	xScale   *gobackend.Buffer
	xZP      *gobackend.Buffer
	outScale *gobackend.Buffer
	outZP    *gobackend.Buffer
	bias     *gobackend.Buffer
}

func unpackQDInputs(data *nodeFusedQuantizedDense, inputs []*gobackend.Buffer) unpackedQDInputs {
	var res unpackedQDInputs
	res.x = inputs[0]
	res.weight = inputs[1]
	idx := 2
	if data.hasWScale {
		res.wScale = inputs[idx]
		idx++
	}
	if data.hasWZP {
		res.wZP = inputs[idx]
		idx++
	}
	if data.hasXScale {
		res.xScale = inputs[idx]
		idx++
	}
	if data.hasXZP {
		res.xZP = inputs[idx]
		idx++
	}
	if data.hasOutScale {
		res.outScale = inputs[idx]
		idx++
	}
	if data.hasOutZP {
		res.outZP = inputs[idx]
		idx++
	}
	if data.hasBias {
		res.bias = inputs[idx]
	}
	return res
}

// execFusedQuantizedDense dispatches FusedQuantizedDense across weight-only, pre-quantized,
// and on-the-fly quantized matmul kernels, followed by activation and optional output quantization.
func execFusedQuantizedDense(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedQuantizedDense)
	in := unpackQDInputs(data, inputs)

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || node.Shape.Size() == 0 {
		return output, nil
	}

	N := data.info.N
	K := data.info.K
	M := node.Shape.Size() / N

	// Allocate or alias Float32 accumulation buffer [M, N].
	var outF32 []float32
	var tmpOutBuf *gobackend.Buffer
	if !data.hasOutputQuant && output.RawShape.DType == dtypes.Float32 {
		outF32 = output.Flat.([]float32)
	} else {
		tmpOutBuf, err = backend.GetBuffer(shapes.Make(dtypes.Float32, M, N))
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(tmpOutBuf)
		outF32 = tmpOutBuf.Flat.([]float32)
	}

	switch data.weightScheme {
	case compute.QuantGGML:
		err = execQuantizedDenseGGMLCore(backend, data, in, M, K, N, outF32)
	case compute.QuantLinear, compute.QuantNF4:
		err = execQuantizedDenseLinearOrNF4Core(backend, data, in, M, K, N, outF32)
	default:
		err = errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDense: unsupported scheme %s", data.weightScheme)
	}
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}

	// Apply post-matmul+bias activation in Float32.
	applyActivationConfigFloat32(backend, data.activation, outF32)

	// Optional OutputQuantization (e.g. ONNX QLinearMatMul).
	if data.hasOutputQuant {
		if err := quantizeOutputFloat32(backend, outF32, M, N, in.outScale, in.outZP, data.info.OutputScaleLayout, data.info.OutputZeroPointLayout, output); err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		return output, nil
	}

	// Convert Float32 -> Float16 / BFloat16 if needed.
	switch output.RawShape.DType {
	case dtypes.Float32:
		// Already written directly into output.Flat.
	case dtypes.Float16:
		dst := output.Flat.([]float16.Float16)
		for i, v := range outF32 {
			dst[i] = float16.FromFloat32(v)
		}
	case dtypes.BFloat16:
		dst := output.Flat.([]bfloat16.BFloat16)
		for i, v := range outF32 {
			dst[i] = bfloat16.FromFloat32(v)
		}
	default:
		backend.PutBuffer(output)
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDense: unsupported output DType %s", output.RawShape.DType)
	}
	return output, nil
}

func applyActivationConfigFloat32(backend *gobackend.Backend, cfg compute.ActivationConfig, data []float32) {
	if cfg.Type == compute.ActivationNone || len(data) == 0 {
		return
	}
	activations.Apply(backend, cfg.Type, data)
}

func applyActivationVJPConfigFloat32(backend *gobackend.Backend, cfg compute.ActivationConfig, y, dOut, dz []float32) error {
	if len(dOut) == 0 {
		return nil
	}
	return activations.ExecuteVJPFromOutput(backend, cfg.Type, y, dOut, dz)
}

func quantizeOutputFloat32(
	backend *gobackend.Backend,
	yF32 []float32,
	M, N int,
	scaleBuf, zpBuf *gobackend.Buffer,
	scaleLayout, zpLayout shapeinference.ActivationParamLayout,
	outBuf *gobackend.Buffer,
) error {
	scales, releaseScale, err := bufferToFloat32Slice(backend, scaleBuf)
	if err != nil {
		return err
	}
	defer releaseScale()

	var zps []float32
	if zpBuf != nil {
		var releaseZP func()
		zps, releaseZP, err = bufferToZeroPointFloat32Slice(backend, zpBuf)
		if err != nil {
			return err
		}
		defer releaseZP()
	}

	switch outBuf.RawShape.DType {
	case dtypes.Int8:
		dst := outBuf.Flat.([]int8)
		for m := range M {
			rowOff := m * N
			for n := range N {
				sc := scales[scaleLayout.Index(m, n)]
				var zp float32
				if zps != nil {
					zp = zps[zpLayout.Index(m, n)]
				}
				q := float32(math.RoundToEven(float64(yF32[rowOff+n]/sc))) + zp
				q = max(float32(-128), min(float32(127), q))
				dst[rowOff+n] = int8(int32(math.RoundToEven(float64(q))))
			}
		}
	case dtypes.Uint8:
		dst := outBuf.Flat.([]uint8)
		for m := range M {
			rowOff := m * N
			for n := range N {
				sc := scales[scaleLayout.Index(m, n)]
				var zp float32
				if zps != nil {
					zp = zps[zpLayout.Index(m, n)]
				}
				q := float32(math.RoundToEven(float64(yF32[rowOff+n]/sc))) + zp
				q = max(float32(0), min(float32(255), q))
				dst[rowOff+n] = uint8(int32(math.RoundToEven(float64(q))))
			}
		}
	default:
		return errors.Errorf("FusedQuantizedDense: unsupported OutputQuantization DType %s", outBuf.RawShape.DType)
	}
	return nil
}

// bufferToFloat32Slice returns a []float32 view (zero-copy when already Float32) or a pooled converted buffer.
func bufferToFloat32Slice(backend *gobackend.Backend, buf *gobackend.Buffer) ([]float32, func(), error) {
	if buf == nil {
		return nil, func() {}, nil
	}
	switch buf.RawShape.DType {
	case dtypes.Float32:
		return buf.Flat.([]float32), func() {}, nil
	case dtypes.Float16:
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, buf.RawShape.Size()))
		if err != nil {
			return nil, nil, err
		}
		src := buf.Flat.([]float16.Float16)
		dst := tmp.Flat.([]float32)
		for i, v := range src {
			dst[i] = v.Float32()
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	case dtypes.BFloat16:
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, buf.RawShape.Size()))
		if err != nil {
			return nil, nil, err
		}
		src := buf.Flat.([]bfloat16.BFloat16)
		dst := tmp.Flat.([]float32)
		for i, v := range src {
			dst[i] = v.Float32()
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	default:
		return nil, nil, errors.Errorf("expected float buffer (Float32/Float16/BFloat16), got %s", buf.RawShape.DType)
	}
}

// isSubByteQuantDType reports whether dt is a packed sub-byte integer type (Int2/Uint2/Int4/Uint4).
func isSubByteQuantDType(dt dtypes.DType) bool {
	switch dt {
	case dtypes.Int2, dtypes.Uint2, dtypes.Int4, dtypes.Uint4:
		return true
	default:
		return false
	}
}

// bufferToZeroPointFloat32Slice converts an integer or float ZeroPoint buffer into a []float32 slice.
func bufferToZeroPointFloat32Slice(backend *gobackend.Backend, buf *gobackend.Buffer) ([]float32, func(), error) {
	if buf == nil {
		return nil, func() {}, nil
	}
	if buf.RawShape.DType == dtypes.Float32 {
		return buf.Flat.([]float32), func() {}, nil
	}
	if buf.RawShape.DType == dtypes.Float16 || buf.RawShape.DType == dtypes.BFloat16 {
		return bufferToFloat32Slice(backend, buf)
	}
	if isSubByteQuantDType(buf.RawShape.DType) {
		n := buf.RawShape.Size()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, n))
		if err != nil {
			return nil, nil, err
		}
		tmpI8 := make([]int8, n)
		unpackSubByteSpanInt8(buf.RawShape.DType, buf.Flat.([]byte), 0, n, tmpI8)
		dst := tmp.Flat.([]float32)
		for i, v := range tmpI8 {
			dst[i] = float32(v)
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	}

	tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, buf.RawShape.Dimensions...))
	if err != nil {
		return nil, nil, err
	}
	convertFnAny, err := ops.ConvertDTypePairMap.Get(buf.RawShape.DType, dtypes.Float32)
	if err != nil {
		backend.PutBuffer(tmp)
		return nil, nil, err
	}
	convertFnAny.(ops.ConvertFnType)(buf, tmp)
	return tmp.Flat.([]float32), func() { backend.PutBuffer(tmp) }, nil
}

// bufferToZeroPointInt32Slice converts a ZeroPoint buffer into a []int32 slice for integer matmul kernels.
func bufferToZeroPointInt32Slice(backend *gobackend.Backend, buf *gobackend.Buffer) ([]int32, func(), error) {
	if buf == nil {
		return nil, func() {}, nil
	}
	if buf.RawShape.DType == dtypes.Int32 {
		return buf.Flat.([]int32), func() {}, nil
	}
	if buf.RawShape.DType.IsFloat() {
		f32, releaseF32, err := bufferToFloat32Slice(backend, buf)
		if err != nil {
			return nil, nil, err
		}
		defer releaseF32()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Int32, len(f32)))
		if err != nil {
			return nil, nil, err
		}
		dst := tmp.Flat.([]int32)
		for i, v := range f32 {
			dst[i] = int32(math.RoundToEven(float64(v)))
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	}
	if isSubByteQuantDType(buf.RawShape.DType) {
		n := buf.RawShape.Size()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Int32, n))
		if err != nil {
			return nil, nil, err
		}
		tmpI8 := make([]int8, n)
		unpackSubByteSpanInt8(buf.RawShape.DType, buf.Flat.([]byte), 0, n, tmpI8)
		dst := tmp.Flat.([]int32)
		for i, v := range tmpI8 {
			dst[i] = int32(v)
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	}

	tmp, err := backend.GetBuffer(shapes.Make(dtypes.Int32, buf.RawShape.Dimensions...))
	if err != nil {
		return nil, nil, err
	}
	convertFnAny, err := ops.ConvertDTypePairMap.Get(buf.RawShape.DType, dtypes.Int32)
	if err != nil {
		backend.PutBuffer(tmp)
		return nil, nil, err
	}
	convertFnAny.(ops.ConvertFnType)(buf, tmp)
	return tmp.Flat.([]int32), func() { backend.PutBuffer(tmp) }, nil
}

// unpackSubByteSpanInt8 unpacks `count` sub-byte elements starting at logical element index `startElem`
// from `packed` into `dst` (as signed int8 for Int4/Int2, or 0..15 / 0..3 int8 for Uint4/Uint2, which all fit in int8!).
func unpackSubByteSpanInt8(dt dtypes.DType, packed []byte, startElem, count int, dst []int8) {
	switch dt {
	case dtypes.Int4:
		if startElem&1 == 0 && count&1 == 0 {
			byteOff := startElem >> 1
			half := count >> 1
			src := packed[byteOff : byteOff+half]
			for i, b := range src {
				lo := int8(b & 0x0F)
				if lo >= 8 {
					lo -= 16
				}
				hi := int8(b) >> 4
				dst[2*i] = lo
				dst[2*i+1] = hi
			}
			return
		}
		for i := range count {
			idx := startElem + i
			nib := int8((packed[idx>>1] >> uint((idx&1)*4)) & 0x0F)
			if nib >= 8 {
				nib -= 16
			}
			dst[i] = nib
		}

	case dtypes.Uint4:
		if startElem&1 == 0 && count&1 == 0 {
			byteOff := startElem >> 1
			half := count >> 1
			src := packed[byteOff : byteOff+half]
			for i, b := range src {
				dst[2*i] = int8(b & 0x0F)
				dst[2*i+1] = int8(b >> 4)
			}
			return
		}
		for i := range count {
			idx := startElem + i
			dst[i] = int8((packed[idx>>1] >> uint((idx&1)*4)) & 0x0F)
		}

	case dtypes.Int2:
		for i := range count {
			idx := startElem + i
			v := int8((packed[idx>>2] >> uint((idx&3)*2)) & 0x03)
			if v >= 2 {
				v -= 4
			}
			dst[i] = v
		}

	case dtypes.Uint2:
		for i := range count {
			idx := startElem + i
			dst[i] = int8((packed[idx>>2] >> uint((idx&3)*2)) & 0x03)
		}
	}
}

type nodeQuantizedEmbeddingLookup struct {
	ggmlType compute.GGMLQuantType
	ggmlK    int // Logical embedding dimension (valuesPerBlock * numBlocks).
}

func (d *nodeQuantizedEmbeddingLookup) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeQuantizedEmbeddingLookup)
	return d.ggmlType == o.ggmlType && d.ggmlK == o.ggmlK
}

// QuantizedEmbeddingLookup performs a quantized embedding lookup (row gather)
// with on-the-fly dequantization.
func QuantizedEmbeddingLookup(
	f *gobackend.Function,
	data, indices compute.Value,
	wq *compute.Quantization,
) (compute.Value, error) {
	if wq == nil {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: dataQuantization must not be nil")
	}
	inputs, err := f.VerifyAndCastValues("QuantizedEmbeddingLookup", data, indices)
	if err != nil {
		return nil, err
	}
	dNode, iNode := inputs[0], inputs[1]

	wqs, _, _, err := extractQuantizationShapes(f, "QuantizedEmbeddingLookup", wq)
	if err != nil {
		return nil, err
	}
	outShape, err := shapeinference.QuantizedEmbeddingLookup(dNode.Shape, iNode.Shape, wqs)
	if err != nil {
		return nil, err
	}
	if _, err := ggmlDequantFunc(wq.GGMLType); err != nil {
		return nil, err
	}

	nodeData := &nodeQuantizedEmbeddingLookup{
		ggmlType: wq.GGMLType,
		ggmlK:    outShape.Dimensions[outShape.Rank()-1],
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeQuantizedEmbeddingLookup, outShape, inputs, nodeData)
	return node, nil
}

// execQuantizedEmbeddingLookup performs quantized embedding lookup.
func execQuantizedEmbeddingLookup(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeQuantizedEmbeddingLookup)
	dataBuf := inputs[0]
	indicesBuf := inputs[1]

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || node.Shape.Size() == 0 {
		return output, nil
	}

	dataBytes := dataBuf.Flat.([]uint8)
	out := output.Flat.([]float32)
	K := data.ggmlK
	bytesPerRow := dataBuf.RawShape.Dimensions[1]

	dequantFn, err := ggmlDequantFunc(data.ggmlType)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}

	numIndices := indicesBuf.RawShape.Size()
	idxBuf, isIdxOwned, err := convertIndicesToInt64(backend, indicesBuf)
	if err != nil {
		backend.PutBuffer(output)
		return nil, errors.Wrapf(err, "QuantizedEmbeddingLookup")
	}
	if isIdxOwned {
		defer backend.PutBuffer(idxBuf)
	}
	indices := idxBuf.Flat.([]int64)

	vocabSize := int64(dataBuf.RawShape.Dimensions[0])
	for i, rowIdx := range indices[:numIndices] {
		if rowIdx < 0 || rowIdx >= vocabSize {
			backend.PutBuffer(output)
			return nil, errors.Errorf("QuantizedEmbeddingLookup: index %d out of range [0, %d)", rowIdx, vocabSize)
		}
		rowStart := rowIdx * int64(bytesPerRow)
		rowData := dataBytes[rowStart : rowStart+int64(bytesPerRow)]
		dequantFn(rowData, out[i*K:(i+1)*K])
	}
	return output, nil
}

func convertIndicesToInt64(backend *gobackend.Backend, indicesBuf *gobackend.Buffer) (*gobackend.Buffer, bool, error) {
	if indicesBuf.RawShape.DType == dtypes.Int64 {
		return indicesBuf, false, nil
	}
	outBuf, err := backend.GetBuffer(shapes.Make(dtypes.Int64, indicesBuf.RawShape.Dimensions...))
	if err != nil {
		return nil, false, err
	}
	convertFnAny, err := ops.ConvertDTypePairMap.Get(indicesBuf.RawShape.DType, dtypes.Int64)
	if err != nil {
		backend.PutBuffer(outBuf)
		return nil, false, err
	}
	convertFnAny.(ops.ConvertFnType)(indicesBuf, outBuf)
	return outBuf, true, nil
}

const quantizedDenseMinParallelizeChunk = 4096

func quantizedDenseParallelTileCount(backend *gobackend.Backend, M, K, N int) int {
	totalWork := M * K * N
	if backend == nil || !backend.Scheduler.IsEnabled() || totalWork <= quantizedDenseMinParallelizeChunk {
		return max(M, 1)
	}
	if M > 1 {
		return M
	}
	tileSize := max(quantizedDenseMinParallelizeChunk/max(K, 1), 1)
	return (N + tileSize - 1) / tileSize
}

func quantizedDenseParallel(backend *gobackend.Backend, M, K, N int, rowFn func(workerIdx, m, nStart, nEnd int)) {
	if M == 0 || N == 0 {
		return
	}
	totalWork := M * K * N
	if backend == nil || !backend.Scheduler.IsEnabled() || totalWork <= quantizedDenseMinParallelizeChunk {
		for m := range M {
			rowFn(m, m, 0, N)
		}
		return
	}

	if M > 1 {
		backend.Scheduler.ParallelizeChunks(M, 1, func(mStart, mEnd int) {
			for m := mStart; m < mEnd; m++ {
				rowFn(m, m, 0, N)
			}
		})
	} else {
		tileSize := max(quantizedDenseMinParallelizeChunk/max(K, 1), 1)
		numTiles := (N + tileSize - 1) / tileSize
		tasks := make([]func(), 0, numTiles)
		workerIdx := 0
		for nStart := 0; nStart < N; nStart += tileSize {
			nEnd := min(nStart+tileSize, N)
			idx := workerIdx
			tasks = append(tasks, func() {
				rowFn(idx, 0, nStart, nEnd)
			})
			workerIdx++
		}
		backend.Scheduler.RunIntraOpTasks(tasks)
	}
}

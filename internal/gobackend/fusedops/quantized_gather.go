// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterQuantizedGather.Register(QuantizedGather, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeQuantizedGather, gobackend.PriorityTyped, execQuantizedGather)
}

type nodeQuantizedGather struct {
	noQuant      bool
	scheme       compute.QuantizationScheme
	ggmlType     compute.GGMLQuantType
	packedDType  dtypes.DType
	info         shapeinference.QuantizedGatherInfo
	hasScale     bool
	hasZeroPoint bool
}

var _ gobackend.RecomputableNodeData = (*nodeQuantizedGather)(nil)

func (d *nodeQuantizedGather) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeQuantizedGather)
	return d.noQuant == o.noQuant &&
		d.scheme == o.scheme &&
		d.ggmlType == o.ggmlType &&
		d.packedDType == o.packedDType &&
		d.info.K == o.info.K &&
		d.info.OutputShape.DType == o.info.OutputShape.DType &&
		d.info.ScaleLayout == o.info.ScaleLayout &&
		d.info.ZeroPointLayout == o.info.ZeroPointLayout &&
		d.hasScale == o.hasScale &&
		d.hasZeroPoint == o.hasZeroPoint
}

func (d *nodeQuantizedGather) Recompute(_ *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
	indicesShape := resolvedNodes[originalNode.Inputs[1].Index].Shape
	if !indicesShape.IsDynamic() {
		outDims := slices.Clone(indicesShape.Dimensions)
		outDims[len(outDims)-1] = d.info.K
		resolvedNodes[originalNode.Index].Shape = shapes.Shape{
			DType:      d.info.OutputShape.DType,
			Dimensions: outDims,
		}
	}
	return d, nil
}

// QuantizedGather performs a quantized row gather (embedding lookup) with on-the-fly
// unpacking / dequantization across nil xQuantization (sub-byte/8-bit unpack to Int8/Uint8),
// QuantGGML, QuantLinear, and QuantNF4.
func QuantizedGather(
	f *gobackend.Function,
	x, indices compute.Value,
	xQuantization *compute.Quantization,
) (compute.Value, error) {
	baseInputs, err := f.VerifyAndCastValues("QuantizedGather", x, indices)
	if err != nil {
		return nil, err
	}
	xNode, iNode := baseInputs[0], baseInputs[1]

	xqs, scaleNode, zpNode, err := extractQuantizationShapes(f, "QuantizedGather", xQuantization)
	if err != nil {
		return nil, err
	}
	info, err := shapeinference.QuantizedGatherWithInfo(xNode.Shape, iNode.Shape, xqs)
	if err != nil {
		return nil, err
	}
	if xqs != nil && xqs.Scheme == compute.QuantGGML {
		if _, err := ggmlDequantFunc(xqs.GGMLType); err != nil {
			return nil, err
		}
	}

	inputs := make([]*gobackend.Node, 0, 4)
	inputs = append(inputs, xNode, iNode)
	if scaleNode != nil {
		inputs = append(inputs, scaleNode)
	}
	if zpNode != nil {
		inputs = append(inputs, zpNode)
	}

	nodeData := &nodeQuantizedGather{
		noQuant:      xQuantization == nil || xQuantization.Scheme == compute.QuantNone,
		info:         info,
		hasScale:     scaleNode != nil,
		hasZeroPoint: zpNode != nil,
	}
	if xQuantization != nil {
		nodeData.scheme = xQuantization.Scheme
		nodeData.ggmlType = xQuantization.GGMLType
		nodeData.packedDType = xQuantization.PackedDType
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeQuantizedGather, info.OutputShape, inputs, nodeData)
	return node, nil
}

func execQuantizedGather(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	nodeData := node.Data.(*nodeQuantizedGather)
	xBuf := inputs[0]
	indicesBuf := inputs[1]
	idx := 2
	var scaleBuf, zpBuf *gobackend.Buffer
	if nodeData.hasScale {
		scaleBuf = inputs[idx]
		idx++
	}
	if nodeData.hasZeroPoint {
		zpBuf = inputs[idx]
	}

	outShape := node.Shape
	if outShape.IsDynamic() {
		outDims := slices.Clone(indicesBuf.RawShape.Dimensions)
		outDims[len(outDims)-1] = nodeData.info.K
		outShape = shapes.Make(nodeData.info.OutputShape.DType, outDims...)
	}
	output, err := backend.GetBuffer(outShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || outShape.Size() == 0 {
		return output, nil
	}

	numIndices := indicesBuf.RawShape.Size()
	vocabSize := int64(xBuf.RawShape.Dimensions[0])
	rowAt, releaseIdx, err := resolveGatherIndices(backend, indicesBuf, numIndices, vocabSize)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	defer releaseIdx()

	K := nodeData.info.K
	canParallel := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		numIndices*K >= quantizedDenseMinParallelizeChunk && numIndices > 1
	minChunk := max(1, quantizedDenseMinParallelizeChunk/max(K, 1))

	if nodeData.noQuant {
		execQuantizedGatherUnpacked(backend, effectiveXInDType(nodeData, xBuf), xBuf, output, rowAt, numIndices, K, canParallel, minChunk)
		return output, nil
	}

	if nodeData.scheme == compute.QuantGGML {
		dataBytes := xBuf.Flat.([]uint8)
		out := output.Flat.([]float32)
		bytesPerRow := xBuf.RawShape.Dimensions[1]
		dequantFn, err := ggmlDequantFunc(nodeData.ggmlType)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		if canParallel {
			backend.Scheduler.ParallelizeChunks(numIndices, minChunk, func(start, end int) {
				for i := start; i < end; i++ {
					rowStart := rowAt(i) * bytesPerRow
					dequantFn(dataBytes[rowStart:rowStart+bytesPerRow], out[i*K:(i+1)*K])
				}
			})
		} else {
			for i := range numIndices {
				rowStart := rowAt(i) * bytesPerRow
				dequantFn(dataBytes[rowStart:rowStart+bytesPerRow], out[i*K:(i+1)*K])
			}
		}
		return output, nil
	}

	execQuantizedGatherLinearOrNF4(backend, nodeData, xBuf, scaleBuf, zpBuf, output, rowAt, numIndices, K, canParallel, minChunk)
	return output, nil
}

func resolveGatherIndices(
	backend *gobackend.Backend,
	indicesBuf *gobackend.Buffer,
	numIndices int,
	vocabSize int64,
) (func(int) int, func(), error) {
	if indicesBuf.RawShape.DType == dtypes.Int32 {
		indices := indicesBuf.Flat.([]int32)[:numIndices]
		for _, rawIdx := range indices {
			if int64(rawIdx) < 0 || int64(rawIdx) >= vocabSize {
				return nil, nil, errors.Errorf("QuantizedGather: index %d out of range [0, %d)", rawIdx, vocabSize)
			}
		}
		return func(i int) int { return int(indices[i]) }, func() {}, nil
	}

	idxBuf, isIdxOwned, err := convertIndicesToInt64(backend, indicesBuf)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "QuantizedGather")
	}
	release := func() {
		if isIdxOwned {
			backend.PutBuffer(idxBuf)
		}
	}
	indices := idxBuf.Flat.([]int64)[:numIndices]
	for _, rowIdx := range indices {
		if rowIdx < 0 || rowIdx >= vocabSize {
			release()
			return nil, nil, errors.Errorf("QuantizedGather: index %d out of range [0, %d)", rowIdx, vocabSize)
		}
	}
	return func(i int) int { return int(indices[i]) }, release, nil
}

func effectiveXInDType(nodeData *nodeQuantizedGather, xBuf *gobackend.Buffer) dtypes.DType {
	if nodeData.packedDType != dtypes.InvalidDType {
		return nodeData.packedDType
	}
	return xBuf.RawShape.DType
}

func execQuantizedGatherUnpacked(
	backend *gobackend.Backend,
	inDType dtypes.DType,
	xBuf, output *gobackend.Buffer,
	rowAt func(int) int,
	numIndices, K int,
	canParallel bool,
	minChunk int,
) {
	processRange := func(start, end int) {
		switch inDType {
		case dtypes.Int8:
			src := xBuf.Flat.([]int8)
			dst := output.Flat.([]int8)
			for i := start; i < end; i++ {
				r := rowAt(i)
				copy(dst[i*K:(i+1)*K], src[r*K:(r+1)*K])
			}
		case dtypes.Uint8:
			src := xBuf.Flat.([]uint8)
			dst := output.Flat.([]uint8)
			for i := start; i < end; i++ {
				r := rowAt(i)
				copy(dst[i*K:(i+1)*K], src[r*K:(r+1)*K])
			}
		case dtypes.Int2, dtypes.Int4:
			packed := xBuf.Flat.([]uint8)
			dst := output.Flat.([]int8)
			for i := start; i < end; i++ {
				r := rowAt(i)
				unpackSubByteSpanInt8(inDType, packed, r*K, K, dst[i*K:(i+1)*K])
			}
		case dtypes.Uint2, dtypes.Uint4:
			packed := xBuf.Flat.([]uint8)
			dst := output.Flat.([]uint8)
			scratch := make([]int8, K)
			for i := start; i < end; i++ {
				r := rowAt(i)
				unpackSubByteSpanInt8(inDType, packed, r*K, K, scratch)
				rowDst := dst[i*K : (i+1)*K]
				for c, v := range scratch {
					rowDst[c] = uint8(v)
				}
			}
		}
	}
	if canParallel {
		backend.Scheduler.ParallelizeChunks(numIndices, minChunk, processRange)
	} else {
		processRange(0, numIndices)
	}
}

func execQuantizedGatherLinearOrNF4(
	backend *gobackend.Backend,
	nodeData *nodeQuantizedGather,
	xBuf, scaleBuf, zpBuf, output *gobackend.Buffer,
	rowAt func(int) int,
	numIndices, K int,
	canParallel bool,
	minChunk int,
) {
	effectiveDType := effectiveXInDType(nodeData, xBuf)
	reader := makeWeightRowReaderWithDType(xBuf, effectiveDType, K)
	scaleReader := makeFloat32ElemReader(scaleBuf)
	var zpReader func(int) float32
	if zpBuf != nil {
		zpReader = makeFloat32ElemReader(zpBuf)
	}
	scaleLayout := nodeData.info.ScaleLayout
	zpLayout := nodeData.info.ZeroPointLayout
	isNF4 := nodeData.scheme == compute.QuantNF4
	isUint8 := effectiveDType == dtypes.Uint8

	outDType := output.RawShape.DType
	var outF32 []float32
	var outF16 []float16.Float16
	var outBF16 []bfloat16.BFloat16
	switch outDType {
	case dtypes.Float32:
		outF32 = output.Flat.([]float32)
	case dtypes.Float16:
		outF16 = output.Flat.([]float16.Float16)
	case dtypes.BFloat16:
		outBF16 = output.Flat.([]bfloat16.BFloat16)
	}

	processRange := func(start, end int) {
		var scratch []int8
		if reader.needsScratch() {
			scratch = make([]int8, K)
		}
		for i := start; i < end; i++ {
			rowIdx := rowAt(i)
			outBase := i * K

			var u8Row []uint8
			var i8Row []int8
			if isUint8 {
				u8Row = reader.rowUint8(rowIdx)
			} else {
				i8Row = reader.rowInt8(rowIdx, scratch)
			}

			for colStart := 0; colStart < K; colStart += scaleLayout.ColBlockSize {
				colEnd := min(colStart+scaleLayout.ColBlockSize, K)
				scale := scaleReader(scaleLayout.Index(rowIdx, colStart))
				var zp float32
				if zpReader != nil {
					zp = zpReader(zpLayout.Index(rowIdx, colStart))
				}

				switch {
				case isNF4 && isUint8:
					dequantNF4RowBlock(u8Row, colStart, colEnd, scale, outDType, outF32, outF16, outBF16, outBase)
				case isNF4:
					dequantNF4RowBlock(i8Row, colStart, colEnd, scale, outDType, outF32, outF16, outBF16, outBase)
				case isUint8:
					dequantLinearRowBlock(u8Row, colStart, colEnd, scale, zp, outDType, outF32, outF16, outBF16, outBase)
				default:
					dequantLinearRowBlock(i8Row, colStart, colEnd, scale, zp, outDType, outF32, outF16, outBF16, outBase)
				}
			}
		}
	}

	if canParallel {
		backend.Scheduler.ParallelizeChunks(numIndices, minChunk, processRange)
	} else {
		processRange(0, numIndices)
	}
}

func dequantLinearRowBlock[T ~uint8 | ~int8](
	row []T, colStart, colEnd int, scale, zp float32,
	outDType dtypes.DType, outF32 []float32, outF16 []float16.Float16, outBF16 []bfloat16.BFloat16, outBase int,
) {
	switch outDType {
	case dtypes.Float32:
		dst := outF32[outBase:]
		for c := colStart; c < colEnd; c++ {
			dst[c] = (float32(row[c]) - zp) * scale
		}
	case dtypes.Float16:
		dst := outF16[outBase:]
		for c := colStart; c < colEnd; c++ {
			dst[c] = float16.FromFloat32((float32(row[c]) - zp) * scale)
		}
	case dtypes.BFloat16:
		dst := outBF16[outBase:]
		for c := colStart; c < colEnd; c++ {
			dst[c] = bfloat16.FromFloat32((float32(row[c]) - zp) * scale)
		}
	}
}

func dequantNF4RowBlock[T ~uint8 | ~int8](
	row []T, colStart, colEnd int, scale float32,
	outDType dtypes.DType, outF32 []float32, outF16 []float16.Float16, outBF16 []bfloat16.BFloat16, outBase int,
) {
	switch outDType {
	case dtypes.Float32:
		dst := outF32[outBase:]
		for c := colStart; c < colEnd; c++ {
			dst[c] = compute.NF4LookupTable[uint8(row[c])&0x0F] * scale
		}
	case dtypes.Float16:
		dst := outF16[outBase:]
		for c := colStart; c < colEnd; c++ {
			dst[c] = float16.FromFloat32(compute.NF4LookupTable[uint8(row[c])&0x0F] * scale)
		}
	case dtypes.BFloat16:
		dst := outBF16[outBase:]
		for c := colStart; c < colEnd; c++ {
			dst[c] = bfloat16.FromFloat32(compute.NF4LookupTable[uint8(row[c])&0x0F] * scale)
		}
	}
}

// makeFloat32ElemReader returns a zero-allocation random-access float32 decoder for a Scale or ZeroPoint buffer.
func makeFloat32ElemReader(buf *gobackend.Buffer) func(int) float32 {
	switch buf.RawShape.DType {
	case dtypes.Float32:
		s := buf.Flat.([]float32)
		return func(i int) float32 { return s[i] }
	case dtypes.Float16:
		s := buf.Flat.([]float16.Float16)
		return func(i int) float32 { return s[i].Float32() }
	case dtypes.BFloat16:
		s := buf.Flat.([]bfloat16.BFloat16)
		return func(i int) float32 { return s[i].Float32() }
	case dtypes.Int8:
		s := buf.Flat.([]int8)
		return func(i int) float32 { return float32(s[i]) }
	case dtypes.Uint8:
		s := buf.Flat.([]uint8)
		return func(i int) float32 { return float32(s[i]) }
	case dtypes.Int32:
		s := buf.Flat.([]int32)
		return func(i int) float32 { return float32(s[i]) }
	case dtypes.Int4:
		b := buf.Flat.([]byte)
		return func(i int) float32 {
			v := int8((b[i>>1] >> uint((i&1)*4)) & 0x0F)
			if v >= 8 {
				v -= 16
			}
			return float32(v)
		}
	case dtypes.Uint4:
		b := buf.Flat.([]byte)
		return func(i int) float32 {
			return float32((b[i>>1] >> uint((i&1)*4)) & 0x0F)
		}
	case dtypes.Int2:
		b := buf.Flat.([]byte)
		return func(i int) float32 {
			v := int8((b[i>>2] >> uint((i&3)*2)) & 0x03)
			if v >= 2 {
				v -= 4
			}
			return float32(v)
		}
	case dtypes.Uint2:
		b := buf.Flat.([]byte)
		return func(i int) float32 {
			return float32((b[i>>2] >> uint((i&3)*2)) & 0x03)
		}
	default:
		return func(int) float32 { return 0 }
	}
}

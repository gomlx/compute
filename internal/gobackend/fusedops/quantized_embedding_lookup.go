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
	gobackend.RegisterQuantizedEmbeddingLookup.Register(QuantizedEmbeddingLookup, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeQuantizedEmbeddingLookup, gobackend.PriorityTyped, execQuantizedEmbeddingLookup)
}

type nodeQuantizedEmbeddingLookup struct {
	scheme       compute.QuantizationScheme
	ggmlType     compute.GGMLQuantType
	info         shapeinference.QuantizedEmbeddingLookupInfo
	hasScale     bool
	hasZeroPoint bool
}

var _ gobackend.RecomputableNodeData = (*nodeQuantizedEmbeddingLookup)(nil)

func (d *nodeQuantizedEmbeddingLookup) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeQuantizedEmbeddingLookup)
	return d.scheme == o.scheme &&
		d.ggmlType == o.ggmlType &&
		d.info.K == o.info.K &&
		d.info.OutputShape.DType == o.info.OutputShape.DType &&
		d.info.ScaleLayout == o.info.ScaleLayout &&
		d.info.ZeroPointLayout == o.info.ZeroPointLayout &&
		d.hasScale == o.hasScale &&
		d.hasZeroPoint == o.hasZeroPoint
}

func (d *nodeQuantizedEmbeddingLookup) Recompute(_ *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
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

// QuantizedEmbeddingLookup performs a quantized embedding lookup (row gather)
// with on-the-fly dequantization across QuantGGML, QuantLinear, and QuantNF4.
func QuantizedEmbeddingLookup(
	f *gobackend.Function,
	data, indices compute.Value,
	wq *compute.Quantization,
) (compute.Value, error) {
	if wq == nil {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: dataQuantization must not be nil")
	}
	baseInputs, err := f.VerifyAndCastValues("QuantizedEmbeddingLookup", data, indices)
	if err != nil {
		return nil, err
	}
	dNode, iNode := baseInputs[0], baseInputs[1]

	wqs, scaleNode, zpNode, err := extractQuantizationShapes(f, "QuantizedEmbeddingLookup", wq)
	if err != nil {
		return nil, err
	}
	info, err := shapeinference.QuantizedEmbeddingLookupWithInfo(dNode.Shape, iNode.Shape, wqs)
	if err != nil {
		return nil, err
	}
	if wqs.Scheme == compute.QuantGGML {
		if _, err := ggmlDequantFunc(wq.GGMLType); err != nil {
			return nil, err
		}
	}

	inputs := make([]*gobackend.Node, 0, 4)
	inputs = append(inputs, dNode, iNode)
	if scaleNode != nil {
		inputs = append(inputs, scaleNode)
	}
	if zpNode != nil {
		inputs = append(inputs, zpNode)
	}

	nodeData := &nodeQuantizedEmbeddingLookup{
		scheme:       wq.Scheme,
		ggmlType:     wq.GGMLType,
		info:         info,
		hasScale:     scaleNode != nil,
		hasZeroPoint: zpNode != nil,
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeQuantizedEmbeddingLookup, info.OutputShape, inputs, nodeData)
	return node, nil
}

func execQuantizedEmbeddingLookup(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	nodeData := node.Data.(*nodeQuantizedEmbeddingLookup)
	dataBuf := inputs[0]
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
	vocabSize := int64(dataBuf.RawShape.Dimensions[0])
	rowAt, releaseIdx, err := resolveEmbeddingIndices(backend, indicesBuf, numIndices, vocabSize)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	defer releaseIdx()

	K := nodeData.info.K
	canParallel := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		numIndices*K >= quantizedDenseMinParallelizeChunk && numIndices > 1
	minChunk := max(1, quantizedDenseMinParallelizeChunk/max(K, 1))

	if nodeData.scheme == compute.QuantGGML {
		dataBytes := dataBuf.Flat.([]uint8)
		out := output.Flat.([]float32)
		bytesPerRow := dataBuf.RawShape.Dimensions[1]
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

	execEmbeddingLookupLinearOrNF4(backend, nodeData, dataBuf, scaleBuf, zpBuf, output, rowAt, numIndices, K, canParallel, minChunk)
	return output, nil
}

func resolveEmbeddingIndices(
	backend *gobackend.Backend,
	indicesBuf *gobackend.Buffer,
	numIndices int,
	vocabSize int64,
) (func(int) int, func(), error) {
	if indicesBuf.RawShape.DType == dtypes.Int32 {
		indices := indicesBuf.Flat.([]int32)[:numIndices]
		for _, rawIdx := range indices {
			if int64(rawIdx) < 0 || int64(rawIdx) >= vocabSize {
				return nil, nil, errors.Errorf("QuantizedEmbeddingLookup: index %d out of range [0, %d)", rawIdx, vocabSize)
			}
		}
		return func(i int) int { return int(indices[i]) }, func() {}, nil
	}

	idxBuf, isIdxOwned, err := convertIndicesToInt64(backend, indicesBuf)
	if err != nil {
		return nil, nil, errors.Wrapf(err, "QuantizedEmbeddingLookup")
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
			return nil, nil, errors.Errorf("QuantizedEmbeddingLookup: index %d out of range [0, %d)", rowIdx, vocabSize)
		}
	}
	return func(i int) int { return int(indices[i]) }, release, nil
}

func execEmbeddingLookupLinearOrNF4(
	backend *gobackend.Backend,
	nodeData *nodeQuantizedEmbeddingLookup,
	dataBuf, scaleBuf, zpBuf, output *gobackend.Buffer,
	rowAt func(int) int,
	numIndices, K int,
	canParallel bool,
	minChunk int,
) {
	reader := makeWeightRowReader(dataBuf, K)
	scaleReader := makeFloat32ElemReader(scaleBuf)
	var zpReader func(int) float32
	if zpBuf != nil {
		zpReader = makeFloat32ElemReader(zpBuf)
	}
	scaleLayout := nodeData.info.ScaleLayout
	zpLayout := nodeData.info.ZeroPointLayout
	isNF4 := nodeData.scheme == compute.QuantNF4
	isUint8 := dataBuf.RawShape.DType == dtypes.Uint8

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

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// nodeFusedQuantizedDenseVJP stores compile-time metadata for FusedQuantizedDenseVJP.
//
// Node inputs order:
//
//	[weight, dOutput, x?, bias?, y?, wScale?, wZeroPoint?]
type nodeFusedQuantizedDenseVJP struct {
	weightScheme     compute.QuantizationScheme
	weightGGMLType   compute.GGMLQuantType
	weightLayout     compute.DenseLayout
	quantizedMatmuls bool
	activation       compute.ActivationConfig

	hasX      bool
	hasBias   bool
	hasY      bool
	hasWScale bool
	hasWZP    bool

	K                     int
	N                     int
	weightScaleLayout     shapeinference.BlockParamLayout
	weightZeroPointLayout shapeinference.BlockParamLayout
}

func (d *nodeFusedQuantizedDenseVJP) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedQuantizedDenseVJP)
	return d.weightScheme == o.weightScheme &&
		d.weightGGMLType == o.weightGGMLType &&
		d.weightLayout == o.weightLayout &&
		d.quantizedMatmuls == o.quantizedMatmuls &&
		d.activation == o.activation &&
		d.hasX == o.hasX &&
		d.hasBias == o.hasBias &&
		d.hasY == o.hasY &&
		d.hasWScale == o.hasWScale &&
		d.hasWZP == o.hasWZP &&
		d.K == o.K &&
		d.N == o.N &&
		d.weightScaleLayout == o.weightScaleLayout &&
		d.weightZeroPointLayout == o.weightZeroPointLayout
}

// FusedQuantizedDenseVJP builds an OpTypeFusedQuantizedDenseVJP multi-output node in the Go backend graph.
func FusedQuantizedDenseVJP(
	f *gobackend.Function,
	x, weight, bias, y, dOutput compute.Value,
	options compute.FusedQuantizedDenseConfig,
) (dx, dBias compute.Value, err error) {
	if options.WeightQuantization == nil {
		return nil, nil, errors.Errorf("FusedQuantizedDenseVJP: options.WeightQuantization must not be nil")
	}
	if weight == nil {
		return nil, nil, errors.Errorf("FusedQuantizedDenseVJP: weight must not be nil")
	}
	if dOutput == nil {
		return nil, nil, errors.Errorf("FusedQuantizedDenseVJP: dOutput must not be nil")
	}

	reqNodes, err := f.VerifyAndCastValues("FusedQuantizedDenseVJP", weight, dOutput)
	if err != nil {
		return nil, nil, err
	}
	wNode, dOutNode := reqNodes[0], reqNodes[1]

	var xNode, biasNode, yNode *gobackend.Node
	var xShape, biasShape, yShape shapes.Shape
	if x != nil {
		nodes, err := f.VerifyAndCastValues("FusedQuantizedDenseVJP", x)
		if err != nil {
			return nil, nil, err
		}
		xNode = nodes[0]
		xShape = xNode.Shape
	}
	if bias != nil {
		nodes, err := f.VerifyAndCastValues("FusedQuantizedDenseVJP", bias)
		if err != nil {
			return nil, nil, err
		}
		biasNode = nodes[0]
		biasShape = biasNode.Shape
	}
	if y != nil {
		nodes, err := f.VerifyAndCastValues("FusedQuantizedDenseVJP", y)
		if err != nil {
			return nil, nil, err
		}
		yNode = nodes[0]
		yShape = yNode.Shape
	}

	wqs, wScaleNode, wZPNode, err := extractQuantizationShapes(f, "FusedQuantizedDenseVJP", options.WeightQuantization)
	if err != nil {
		return nil, nil, err
	}
	xqs, _, _, err := extractQuantizationShapes(f, "FusedQuantizedDenseVJP", options.XQuantization)
	if err != nil {
		return nil, nil, err
	}
	oqs, _, _, err := extractQuantizationShapes(f, "FusedQuantizedDenseVJP", options.OutputQuantization)
	if err != nil {
		return nil, nil, err
	}

	cfgShapes := shapeinference.QuantizedDenseConfigShapes{
		WeightQuantization: wqs,
		XQuantization:      xqs,
		OutputQuantization: oqs,
		QuantizedMatmuls:   options.QuantizedMatmuls,
		Activation:         options.Activation,
		WeightLayout:       options.WeightLayout,
	}
	dxShape, dBiasShape, err := shapeinference.FusedQuantizedDenseVJP(xShape, wNode.Shape, biasShape, yShape, dOutNode.Shape, cfgShapes)
	if err != nil {
		return nil, nil, err
	}

	// Obtain resolved weight parameter layouts using dxShape as the effective x shape.
	info, err := shapeinference.FusedQuantizedDenseWithInfo(dxShape, wNode.Shape, biasShape, cfgShapes)
	if err != nil {
		return nil, nil, err
	}
	if wqs.Scheme == compute.QuantGGML {
		if _, err := ggmlDequantFunc(wqs.GGMLType); err != nil {
			return nil, nil, err
		}
	}

	inputs := make([]*gobackend.Node, 0, 7)
	inputs = append(inputs, wNode, dOutNode)
	if xNode != nil {
		inputs = append(inputs, xNode)
	}
	if biasNode != nil {
		inputs = append(inputs, biasNode)
	}
	if yNode != nil {
		inputs = append(inputs, yNode)
	}
	if wScaleNode != nil {
		inputs = append(inputs, wScaleNode)
	}
	if wZPNode != nil {
		inputs = append(inputs, wZPNode)
	}

	data := &nodeFusedQuantizedDenseVJP{
		weightScheme:          wqs.Scheme,
		weightGGMLType:        wqs.GGMLType,
		weightLayout:          options.WeightLayout,
		quantizedMatmuls:      options.QuantizedMatmuls,
		activation:            options.Activation,
		hasX:                  xNode != nil,
		hasBias:               biasNode != nil,
		hasY:                  yNode != nil,
		hasWScale:             wScaleNode != nil,
		hasWZP:                wZPNode != nil,
		K:                     info.K,
		N:                     info.N,
		weightScaleLayout:     info.WeightScaleLayout,
		weightZeroPointLayout: info.WeightZeroPointLayout,
	}

	var outputShapes []shapes.Shape
	if biasNode != nil {
		outputShapes = []shapes.Shape{dxShape, dBiasShape}
	} else {
		outputShapes = []shapes.Shape{dxShape}
	}

	node := f.NewMultiOutputsNode(compute.OpTypeFusedQuantizedDenseVJP, outputShapes, inputs...)
	node.Data = data

	dx = node.MultiOutputsNodes[0]
	if biasNode != nil {
		dBias = node.MultiOutputsNodes[1]
	}
	return dx, dBias, nil
}

// execFusedQuantizedDenseVJP computes dx = dZ @ dequant(weight)^T and optional dBias = sum_m(dZ)
// without materializing the full [K, N] or [N, K] floating-point weight matrix.
func execFusedQuantizedDenseVJP(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) ([]*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedQuantizedDenseVJP)
	wBuf := inputs[0]
	dOutBuf := inputs[1]
	idx := 2
	if data.hasX {
		idx++
	}
	if data.hasBias {
		idx++
	}
	var yBuf, wScaleBuf, wZPBuf *gobackend.Buffer
	if data.hasY {
		yBuf = inputs[idx]
		idx++
	}
	if data.hasWScale {
		wScaleBuf = inputs[idx]
		idx++
	}
	if data.hasWZP {
		wZPBuf = inputs[idx]
	}

	dxShape := node.MultiOutputsShapes[0]
	dxBuf, err := backend.GetBuffer(dxShape)
	if err != nil {
		return nil, err
	}
	var dbBuf *gobackend.Buffer
	if data.hasBias {
		dbBuf, err = backend.GetBuffer(node.MultiOutputsShapes[1])
		if err != nil {
			backend.PutBuffer(dxBuf)
			return nil, err
		}
	}

	putOutputs := func() {
		backend.PutBuffer(dxBuf)
		if dbBuf != nil {
			backend.PutBuffer(dbBuf)
		}
	}

	if backend.NoOps {
		if data.hasBias {
			return []*gobackend.Buffer{dxBuf, dbBuf}, nil
		}
		return []*gobackend.Buffer{dxBuf}, nil
	}

	K := data.K
	N := data.N
	M := dxShape.Size() / K
	if M == 0 {
		if dbBuf != nil {
			zeroBuffer(dbBuf)
			return []*gobackend.Buffer{dxBuf, dbBuf}, nil
		}
		return []*gobackend.Buffer{dxBuf}, nil
	}

	dOutF32, releaseDOut, err := bufferToFloat32Slice(backend, dOutBuf)
	if err != nil {
		putOutputs()
		return nil, err
	}
	defer releaseDOut()

	// Step 1: Compute pre-activation gradient dZ [M, N] in Float32.
	var dZF32 []float32
	if data.activation.Type == compute.ActivationNone {
		dZF32 = dOutF32
	} else {
		yF32, releaseY, err := bufferToFloat32Slice(backend, yBuf)
		if err != nil {
			putOutputs()
			return nil, err
		}
		defer releaseY()
		dzTmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, M*N))
		if err != nil {
			putOutputs()
			return nil, err
		}
		defer backend.PutBuffer(dzTmp)
		dZF32 = dzTmp.Flat.([]float32)
		if err := applyActivationVJPConfigFloat32(backend, data.activation, yF32, dOutF32, dZF32); err != nil {
			putOutputs()
			return nil, err
		}
	}

	// Step 2: Optional dBias = sum_{m=0..M-1} dZ[m, n].
	if dbBuf != nil {
		if err := computeDBias(backend, dZF32, M, N, dbBuf); err != nil {
			putOutputs()
			return nil, err
		}
	}

	// Step 3: Allocate or alias Float32 dx [M, K].
	var dxF32 []float32
	if dxBuf.RawShape.DType == dtypes.Float32 {
		dxF32 = dxBuf.Flat.([]float32)
	} else {
		dxTmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, M*K))
		if err != nil {
			putOutputs()
			return nil, err
		}
		defer backend.PutBuffer(dxTmp)
		dxF32 = dxTmp.Flat.([]float32)
	}

	// Step 4: Streaming quantized backward matmul dx = dZ @ dequant(W)^T.
	switch data.weightScheme {
	case compute.QuantGGML:
		err = execQuantizedDenseVJPGGML(backend, dZF32, wBuf, dxF32, M, K, N, data.weightGGMLType)
	case compute.QuantLinear, compute.QuantNF4:
		err = execQuantizedDenseVJPLinearOrNF4(backend, data, dZF32, wBuf, wScaleBuf, wZPBuf, dxF32, M, K, N)
	default:
		err = errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDenseVJP: unsupported scheme %s", data.weightScheme)
	}
	if err != nil {
		putOutputs()
		return nil, err
	}

	// Convert Float32 dx -> Float16 / BFloat16 if needed.
	switch dxBuf.RawShape.DType {
	case dtypes.Float32:
	case dtypes.Float16:
		dst := dxBuf.Flat.([]float16.Float16)
		for i, v := range dxF32 {
			dst[i] = float16.FromFloat32(v)
		}
	case dtypes.BFloat16:
		dst := dxBuf.Flat.([]bfloat16.BFloat16)
		for i, v := range dxF32 {
			dst[i] = bfloat16.FromFloat32(v)
		}
	default:
		putOutputs()
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDenseVJP: unsupported DType %s", dxBuf.RawShape.DType)
	}

	if data.hasBias {
		return []*gobackend.Buffer{dxBuf, dbBuf}, nil
	}
	return []*gobackend.Buffer{dxBuf}, nil
}

func zeroBuffer(buf *gobackend.Buffer) {
	switch flat := buf.Flat.(type) {
	case []float32:
		clear(flat)
	case []float16.Float16:
		clear(flat)
	case []bfloat16.BFloat16:
		clear(flat)
	}
}

func computeDBias(backend *gobackend.Backend, dZ []float32, M, N int, dbBuf *gobackend.Buffer) error {
	var acc []float32
	if dbBuf.RawShape.DType == dtypes.Float32 {
		acc = dbBuf.Flat.([]float32)
	} else {
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, N))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tmp)
		acc = tmp.Flat.([]float32)
	}
	clear(acc)
	for m := range M {
		row := dZ[m*N : (m+1)*N]
		for n, v := range row {
			acc[n] += v
		}
	}
	switch dbBuf.RawShape.DType {
	case dtypes.Float32:
	case dtypes.Float16:
		dst := dbBuf.Flat.([]float16.Float16)
		for n, v := range acc {
			dst[n] = float16.FromFloat32(v)
		}
	case dtypes.BFloat16:
		dst := dbBuf.Flat.([]bfloat16.BFloat16)
		for n, v := range acc {
			dst[n] = bfloat16.FromFloat32(v)
		}
	default:
		return errors.Errorf("FusedQuantizedDenseVJP: unsupported bias DType %s", dbBuf.RawShape.DType)
	}
	return nil
}

// execQuantizedDenseVJPGGML computes dx[m, k] = sum_n dZ[m, n] * W_deq[n, k] for GGML [N, bytesPerRow] weights.
// Dequantizes a single row W[n, 0..K) at a time into L1-resident worker scratch memory.
func execQuantizedDenseVJPGGML(
	backend *gobackend.Backend,
	dZ []float32,
	wBuf *gobackend.Buffer,
	dx []float32,
	M, K, N int,
	ggmlType compute.GGMLQuantType,
) error {
	dequantFn, err := ggmlDequantFunc(ggmlType)
	if err != nil {
		return err
	}
	weights := wBuf.Flat.([]uint8)
	bytesPerRow := wBuf.RawShape.Dimensions[1]

	totalWork := M * K * N
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && totalWork > quantizedDenseMinParallelizeChunk && M > 1 {
		// Chunk over M rows so each worker dequantizes row n once per chunk [mStart, mEnd).
		backend.Scheduler.ParallelizeChunks(M, 1, func(mStart, mEnd int) {
			wRowK := make([]float32, K)
			clear(dx[mStart*K : mEnd*K])
			for n := range N {
				dequantFn(weights[n*bytesPerRow:(n+1)*bytesPerRow], wRowK)
				for m := mStart; m < mEnd; m++ {
					dzVal := dZ[m*N+n]
					if dzVal == 0 {
						continue
					}
					dxRow := dx[m*K : (m+1)*K]
					for k := range K {
						dxRow[k] += dzVal * wRowK[k]
					}
				}
			}
		})
		return nil
	}

	// Single-token (M == 1) or small problem: stream row-by-row using a single [K] float32 buffer.
	wRowK := make([]float32, K)
	clear(dx[:M*K])
	for n := range N {
		// Skip dequantizing row n if dZ[:, n] is all zeros.
		anyNonZero := false
		for m := range M {
			if dZ[m*N+n] != 0 {
				anyNonZero = true
				break
			}
		}
		if !anyNonZero {
			continue
		}
		dequantFn(weights[n*bytesPerRow:(n+1)*bytesPerRow], wRowK)
		for m := range M {
			dzVal := dZ[m*N+n]
			if dzVal == 0 {
				continue
			}
			dxRow := dx[m*K : (m+1)*K]
			for k := range K {
				dxRow[k] += dzVal * wRowK[k]
			}
		}
	}
	return nil
}

// execQuantizedDenseVJPLinearOrNF4 computes dx = dZ @ dequant(W)^T for QuantLinear and QuantNF4
// without materializing the full [K, N] or [N, K] float weight matrix.
func execQuantizedDenseVJPLinearOrNF4(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDenseVJP,
	dZ []float32,
	wBuf, wScaleBuf, wZPBuf *gobackend.Buffer,
	dx []float32,
	M, K, N int,
) error {
	wScales, releaseScale, err := bufferToFloat32Slice(backend, wScaleBuf)
	if err != nil {
		return err
	}
	defer releaseScale()

	var wZPs []float32
	if wZPBuf != nil {
		var releaseZP func()
		wZPs, releaseZP, err = bufferToZeroPointFloat32Slice(backend, wZPBuf)
		if err != nil {
			return err
		}
		defer releaseZP()
	}

	sLayout := data.weightScaleLayout
	zpLayout := data.weightZeroPointLayout
	scheme := data.weightScheme

	if data.weightLayout == compute.DenseLayoutInputOutputs {
		// Weight is [K, N]: dx[m, k] = sum_{n=0..N-1} dZ[m, n] * W_deq[k, n].
		// Both dZ[m, 0..N) and W[k, 0..N) are stride-1 along n!
		reader := makeWeightRowReader(wBuf, N)
		numWorkers := quantizedDenseParallelTileCount(backend, M, N, K)
		var scratchRows [][]int8
		if reader.needsScratch() {
			scratchRows = make([][]int8, numWorkers)
			for i := range scratchRows {
				scratchRows[i] = make([]int8, N)
			}
		}

		nStep := sLayout.ColBlockSize
		if wZPs != nil && zpLayout.ColBlockSize < nStep {
			nStep = zpLayout.ColBlockSize
		}

		quantizedDenseParallel(backend, M, N, K, func(workerIdx, m, kStart, kEnd int) {
			var scratch []int8
			if scratchRows != nil {
				scratch = scratchRows[workerIdx]
			}
			dzRow := dZ[m*N : (m+1)*N]
			for k := kStart; k < kEnd; k++ {
				sRowOff := (k / sLayout.RowBlockSize) * sLayout.NumColBlocks
				var zpRowOff int
				if wZPs != nil {
					zpRowOff = (k / zpLayout.RowBlockSize) * zpLayout.NumColBlocks
				}
				var dot float32
				if scheme == compute.QuantNF4 {
					if reader.isUint8 {
						wRow := reader.rowUint8(k)
						for n0 := 0; n0 < N; n0 += nStep {
							n1 := min(n0+nStep, N)
							sc := wScales[sRowOff+n0/sLayout.ColBlockSize]
							var sub float32
							for n := n0; n < n1; n++ {
								sub += dzRow[n] * compute.NF4LookupTable[wRow[n]&0x0F]
							}
							dot += sub * sc
						}
					} else {
						wRow := reader.rowInt8(k, scratch)
						for n0 := 0; n0 < N; n0 += nStep {
							n1 := min(n0+nStep, N)
							sc := wScales[sRowOff+n0/sLayout.ColBlockSize]
							var sub float32
							for n := n0; n < n1; n++ {
								sub += dzRow[n] * compute.NF4LookupTable[uint8(wRow[n])&0x0F]
							}
							dot += sub * sc
						}
					}
				} else {
					if reader.isUint8 {
						wRow := reader.rowUint8(k)
						for n0 := 0; n0 < N; n0 += nStep {
							n1 := min(n0+nStep, N)
							sc := wScales[sRowOff+n0/sLayout.ColBlockSize]
							var zp float32
							if wZPs != nil {
								zp = wZPs[zpRowOff+n0/zpLayout.ColBlockSize]
							}
							var sub float32
							for n := n0; n < n1; n++ {
								sub += dzRow[n] * (float32(wRow[n]) - zp)
							}
							dot += sub * sc
						}
					} else {
						wRow := reader.rowInt8(k, scratch)
						for n0 := 0; n0 < N; n0 += nStep {
							n1 := min(n0+nStep, N)
							sc := wScales[sRowOff+n0/sLayout.ColBlockSize]
							var zp float32
							if wZPs != nil {
								zp = wZPs[zpRowOff+n0/zpLayout.ColBlockSize]
							}
							var sub float32
							for n := n0; n < n1; n++ {
								sub += dzRow[n] * (float32(wRow[n]) - zp)
							}
							dot += sub * sc
						}
					}
				}
				dx[m*K+k] = dot
			}
		})
		return nil
	}

	// DenseLayoutOutputsInput: Weight is [N, K], dx[m, k] = sum_{n=0..N-1} dZ[m, n] * W_deq[n, k].
	// Both dx[m, kStart:kEnd] and W[n, kStart:kEnd] are stride-1 along k!
	reader := makeWeightRowReader(wBuf, K)
	numWorkers := quantizedDenseParallelTileCount(backend, M, N, K)
	var scratchRows [][]int8
	if reader.needsScratch() {
		scratchRows = make([][]int8, numWorkers)
		for i := range scratchRows {
			scratchRows[i] = make([]int8, K)
		}
	}

	quantizedDenseParallel(backend, M, N, K, func(workerIdx, m, kStart, kEnd int) {
		var scratch []int8
		if scratchRows != nil {
			scratch = scratchRows[workerIdx]
		}
		dxSlice := dx[m*K+kStart : m*K+kEnd]
		clear(dxSlice)

		for n := range N {
			dzVal := dZ[m*N+n]
			if dzVal == 0 {
				continue
			}
			sRowOff := (n / sLayout.RowBlockSize) * sLayout.NumColBlocks
			var zpRowOff int
			if wZPs != nil {
				zpRowOff = (n / zpLayout.RowBlockSize) * zpLayout.NumColBlocks
			}

			if scheme == compute.QuantNF4 {
				if reader.isUint8 {
					wRow := reader.rowUint8(n)
					for k := kStart; k < kEnd; k++ {
						sc := wScales[sRowOff+k/sLayout.ColBlockSize]
						dxSlice[k-kStart] += dzVal * compute.NF4LookupTable[wRow[k]&0x0F] * sc
					}
				} else {
					wRow := reader.rowInt8(n, scratch)
					for k := kStart; k < kEnd; k++ {
						sc := wScales[sRowOff+k/sLayout.ColBlockSize]
						dxSlice[k-kStart] += dzVal * compute.NF4LookupTable[uint8(wRow[k])&0x0F] * sc
					}
				}
				continue
			}

			if reader.isUint8 {
				wRow := reader.rowUint8(n)
				if wZPs != nil {
					for k := kStart; k < kEnd; k++ {
						sc := wScales[sRowOff+k/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+k/zpLayout.ColBlockSize]
						dxSlice[k-kStart] += dzVal * (float32(wRow[k]) - zp) * sc
					}
				} else {
					for k := kStart; k < kEnd; k++ {
						sc := wScales[sRowOff+k/sLayout.ColBlockSize]
						dxSlice[k-kStart] += dzVal * float32(wRow[k]) * sc
					}
				}
			} else {
				wRow := reader.rowInt8(n, scratch)
				if wZPs != nil {
					for k := kStart; k < kEnd; k++ {
						sc := wScales[sRowOff+k/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+k/zpLayout.ColBlockSize]
						dxSlice[k-kStart] += dzVal * (float32(wRow[k]) - zp) * sc
					}
				} else {
					for k := kStart; k < kEnd; k++ {
						sc := wScales[sRowOff+k/sLayout.ColBlockSize]
						dxSlice[k-kStart] += dzVal * float32(wRow[k]) * sc
					}
				}
			}
		}
	})
	return nil
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
)

// defaultQuantizedFloatMaxScratchFloats caps temporary pooled float32 weight dequantization
// and chunked output buffers to 1M float32 elements (4 MiB) when go:quantized_cache is off.
// This bounds peak RSS overhead for multi-billion-parameter weight matrices while allowing
// standard layers (N*K <= 1,048,576, e.g. 1536x384 or 1024x1024) to execute in a single
// zero-copy matmul.ExecuteWithEpilogue call.
const defaultQuantizedFloatMaxScratchFloats = 1024 * 1024

// quantizedFloatChunkSize computes a column/row chunk size in [1, totalCols] such that
// chunkCols * rowLen <= maxScratchFloats, aligned down to a multiple of 32 when >= 32
// so AVX-512 / AVX2 GEMM kernels avoid edge-tile output scratch buffers.
func quantizedFloatChunkSize(totalCols, rowLen, maxScratchFloats int) int {
	if maxScratchFloats <= 0 {
		maxScratchFloats = defaultQuantizedFloatMaxScratchFloats
	}
	if totalCols <= 0 || totalCols*rowLen <= maxScratchFloats {
		return totalCols
	}
	maxCols := max(1, maxScratchFloats/max(1, rowLen))
	if maxCols >= 32 {
		maxCols &^= 31
	}
	return min(totalCols, maxCols)
}

func buildQuantizedFloatMatmulEpilogue(actCfg compute.ActivationConfig, bias []float32) (epilogue matmul.Epilogue[float32], appliedAct bool) {
	epilogue.Bias = bias
	if actCfg.Type != compute.ActivationNone && actCfg.Type != compute.ActivationSwiGLU {
		actFn, actMeta := activations.GetWithMetadata[float32](actCfg.Type)
		if actFn != nil {
			epilogue.Activation = actFn
			epilogue.ActivationMinParallelChunk = actMeta.EffectiveMinParallelChunk()
			epilogue.BiasTileActivation = activations.GetBiasTile[float32](actCfg.Type)
			appliedAct = true
		}
	}
	return epilogue, appliedAct
}

// execQuantizedFloatChunkedMatmul executes out = epilogue(x @ dequant(W)) by dequantizing
// column chunks [nStart, nEnd) into a bounded pooled float32 buffer and dispatching to
// matmul.ExecuteWithEpilogue (AVX-512 / AVX2 / NEON / NoSIMD).
//
// When dotNodeData != nil (go:quantized_cache enabled on constant weights), the full weight
// matrix is dequantized once on cold start so matmul caches pre-packed RHS GEMM panels in
// dotNodeData.RHSConstantData, and subsequent calls skip dequantization completely.
func execQuantizedFloatChunkedMatmul(
	backend *gobackend.Backend,
	layout dot.Layout,
	xF32, biasF32, outF32 []float32,
	M, K, N int,
	actCfg compute.ActivationConfig,
	dotNodeData *dot.NodeData,
	maxScratchFloats int,
	dequantChunkFn func(nStart, nEnd int, dst []float32),
) (bool, error) {
	if M <= 0 || K <= 0 || N <= 0 {
		return false, nil
	}

	// Fast cached path (go:quantized_cache enabled on constant weights):
	// Once matmul has cached the packed RHS panels, pass rhs = nil directly.
	if dotNodeData != nil && dotNodeData.RHSConstantData != nil {
		epilogue, appliedAct := buildQuantizedFloatMatmulEpilogue(actCfg, biasF32)
		if dotNodeData.RHSConstantData.CachedForm() != nil {
			if err := matmul.ExecuteWithEpilogue(
				backend, layout, xF32, nil,
				1, M, N, K,
				outF32, epilogue, dotNodeData,
			); err != nil {
				return false, err
			}
			return appliedAct, nil
		}

		wDeqBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, N*K))
		if err != nil {
			return false, err
		}
		defer backend.PutBuffer(wDeqBuf)
		wDeq := wDeqBuf.Flat.([]float32)[:N*K]
		dequantChunkFn(0, N, wDeq)

		if err := matmul.ExecuteWithEpilogue(
			backend, layout, xF32, wDeq,
			1, M, N, K,
			outF32, epilogue, dotNodeData,
		); err != nil {
			return false, err
		}
		return appliedAct, nil
	}

	chunkN := quantizedFloatChunkSize(N, K, maxScratchFloats)

	// Single-chunk fast path (N * K <= maxScratchFloats): zero output copy, inline tile epilogue.
	if chunkN >= N {
		wDeqBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, N*K))
		if err != nil {
			return false, err
		}
		defer backend.PutBuffer(wDeqBuf)
		wDeq := wDeqBuf.Flat.([]float32)[:N*K]
		dequantChunkFn(0, N, wDeq)

		epilogue, appliedAct := buildQuantizedFloatMatmulEpilogue(actCfg, biasF32)
		if err := matmul.ExecuteWithEpilogue(
			backend, layout, xF32, wDeq,
			1, M, N, K,
			outF32, epilogue,
		); err != nil {
			return false, err
		}
		return appliedAct, nil
	}

	// Multi-chunk path (N * K > maxScratchFloats): reuse bounded pooled wDeqBuf and chunkOutBuf.
	wDeqBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, chunkN*K))
	if err != nil {
		return false, err
	}
	defer backend.PutBuffer(wDeqBuf)
	wDeqFull := wDeqBuf.Flat.([]float32)

	if maxScratchFloats <= 0 {
		maxScratchFloats = defaultQuantizedFloatMaxScratchFloats
	}

	var chunkOutBuf *gobackend.Buffer
	var chunkOutFull []float32
	mChunk := 1
	if M > 1 {
		mChunk = quantizedFloatChunkSize(M, chunkN, maxScratchFloats)
		chunkOutBuf, err = backend.GetBuffer(shapes.Make(dtypes.Float32, mChunk*chunkN))
		if err != nil {
			return false, err
		}
		defer backend.PutBuffer(chunkOutBuf)
		chunkOutFull = chunkOutBuf.Flat.([]float32)
	}

	var appliedAct bool
	for nStart := 0; nStart < N; nStart += chunkN {
		nEnd := min(nStart+chunkN, N)
		nLen := nEnd - nStart
		wDeq := wDeqFull[:nLen*K]
		dequantChunkFn(nStart, nEnd, wDeq)

		var chunkBias []float32
		if biasF32 != nil {
			chunkBias = biasF32[nStart:nEnd]
		}
		chunkEpilogue, chunkApplied := buildQuantizedFloatMatmulEpilogue(actCfg, chunkBias)
		appliedAct = chunkApplied

		if M == 1 {
			if err := matmul.ExecuteWithEpilogue(
				backend, layout, xF32, wDeq,
				1, 1, nLen, K,
				outF32[nStart:nEnd], chunkEpilogue,
			); err != nil {
				return false, err
			}
			continue
		}

		for mStart := 0; mStart < M; mStart += mChunk {
			mEnd := min(mStart+mChunk, M)
			mLen := mEnd - mStart
			chunkOut := chunkOutFull[:mLen*nLen]
			if err := matmul.ExecuteWithEpilogue(
				backend, layout, xF32[mStart*K:mEnd*K], wDeq,
				1, mLen, nLen, K,
				chunkOut, chunkEpilogue,
			); err != nil {
				return false, err
			}
			scatterContiguousColsFloat32(backend, chunkOut, outF32, mStart, mLen, N, nStart, nLen)
		}
	}
	return appliedAct, nil
}

func scatterContiguousColsFloat32(
	backend *gobackend.Backend,
	src, dst []float32,
	mStart, mLen, dstStride, colStart, colLen int,
) {
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		mLen > 1 && mLen*colLen >= 16384 {
		minRows := max(1, 16384/max(1, colLen))
		backend.Scheduler.ParallelizeChunks(mLen, minRows, func(r0, r1 int) {
			for r := r0; r < r1; r++ {
				dstOff := (mStart+r)*dstStride + colStart
				copy(dst[dstOff:dstOff+colLen], src[r*colLen:(r+1)*colLen])
			}
		})
		return
	}
	for r := range mLen {
		dstOff := (mStart+r)*dstStride + colStart
		copy(dst[dstOff:dstOff+colLen], src[r*colLen:(r+1)*colLen])
	}
}

func gatherContiguousColsFloat32(
	backend *gobackend.Backend,
	src, dst []float32,
	mStart, mLen, srcStride, colStart, colLen int,
) {
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		mLen > 1 && mLen*colLen >= 16384 {
		minRows := max(1, 16384/max(1, colLen))
		backend.Scheduler.ParallelizeChunks(mLen, minRows, func(r0, r1 int) {
			for r := r0; r < r1; r++ {
				srcOff := (mStart+r)*srcStride + colStart
				copy(dst[r*colLen:(r+1)*colLen], src[srcOff:srcOff+colLen])
			}
		})
		return
	}
	for r := range mLen {
		srcOff := (mStart+r)*srcStride + colStart
		copy(dst[r*colLen:(r+1)*colLen], src[srcOff:srcOff+colLen])
	}
}

func addSliceFloat32(backend *gobackend.Backend, dst, src []float32) {
	n := len(dst)
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && n >= 32768 {
		backend.Scheduler.ParallelizeChunks(n, 16384, func(start, end int) {
			d := dst[start:end]
			s := src[start:end]
			for i, v := range s {
				d[i] += v
			}
		})
		return
	}
	for i, v := range src {
		dst[i] += v
	}
}

// --- QuantGGML Mode A (Float Matmul) ---

func execQuantizedDenseGGMLFloatMatmul(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDense,
	xF32 []float32,
	weights []uint8,
	biasF32, outF32 []float32,
	M, K, N, bytesPerRow int,
	ggmlType compute.GGMLQuantType,
	maxScratchFloats int,
) (bool, error) {
	dequantFn, err := ggmlDequantFunc(ggmlType)
	if err != nil {
		return false, err
	}
	return execQuantizedFloatChunkedMatmul(
		backend,
		dot.LayoutTransposed,
		xF32, biasF32, outF32,
		M, K, N,
		data.activation,
		data.floatDotNodeData,
		maxScratchFloats,
		func(nStart, nEnd int, dst []float32) {
			dequantGGMLRowsChunk(backend, weights, bytesPerRow, nStart, nEnd, K, dequantFn, dst)
		},
	)
}

func dequantGGMLRowsChunk(
	backend *gobackend.Backend,
	weights []uint8,
	bytesPerRow, rStart, rEnd, rowLen int,
	dequantFn func(data []uint8, output []float32),
	dst []float32,
) {
	numRows := rEnd - rStart
	if numRows <= 0 {
		return
	}
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		numRows > 1 && numRows*rowLen >= quantizedDenseMinParallelizeChunk {
		minChunkRows := max(1, quantizedDenseMinParallelizeChunk/max(1, rowLen))
		backend.Scheduler.ParallelizeChunks(numRows, minChunkRows, func(c0, c1 int) {
			for r := c0; r < c1; r++ {
				row := rStart + r
				dequantFn(weights[row*bytesPerRow:(row+1)*bytesPerRow], dst[r*rowLen:(r+1)*rowLen])
			}
		})
		return
	}
	for r := range numRows {
		row := rStart + r
		dequantFn(weights[row*bytesPerRow:(row+1)*bytesPerRow], dst[r*rowLen:(r+1)*rowLen])
	}
}

// --- QuantLinear & QuantNF4 Mode A (Float Matmul) ---

func execQuantizedDenseLinearOrNF4FloatMatmul(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDense,
	xF32 []float32,
	wBuf *gobackend.Buffer,
	wScales, wZPF32, biasF32, outF32 []float32,
	M, K, N int,
	maxScratchFloats int,
) (bool, error) {
	sLayout := data.info.WeightScaleLayout
	zpLayout := data.info.WeightZeroPointLayout
	scheme := data.weightScheme

	if data.weightLayout == compute.DenseLayoutInputOutputs {
		// Weight is [K, N]: dequantize column chunk [0..K, nStart..nEnd] into [K, nLen]
		// and execute with dot.LayoutNonTransposed.
		return execQuantizedFloatChunkedMatmul(
			backend,
			dot.LayoutNonTransposed,
			xF32, biasF32, outF32,
			M, K, N,
			data.activation,
			data.floatDotNodeData,
			maxScratchFloats,
			func(nStart, nEnd int, dst []float32) {
				dequantLinearOrNF4ColsChunk(backend, scheme, wBuf, wScales, wZPF32, K, N, nStart, nEnd, sLayout, zpLayout, dst)
			},
		)
	}

	// DenseLayoutOutputsInput: Weight is [N, K]; dequantize rows [nStart..nEnd, 0..K] into [nLen, K]
	// and execute with dot.LayoutTransposed.
	return execQuantizedFloatChunkedMatmul(
		backend,
		dot.LayoutTransposed,
		xF32, biasF32, outF32,
		M, K, N,
		data.activation,
		data.floatDotNodeData,
		maxScratchFloats,
		func(nStart, nEnd int, dst []float32) {
			dequantLinearOrNF4RowsChunk(backend, scheme, wBuf, wScales, wZPF32, nStart, nEnd, K, sLayout, zpLayout, dst)
		},
	)
}

// dequantLinearOrNF4RowsChunk dequantizes contiguous rows [rStart, rEnd) of a [TotalRows, RowLen]
// QuantLinear or QuantNF4 weight tensor into dst of shape [rEnd-rStart, RowLen].
func dequantLinearOrNF4RowsChunk(
	backend *gobackend.Backend,
	scheme compute.QuantizationScheme,
	wBuf *gobackend.Buffer,
	wScales, wZPs []float32,
	rStart, rEnd, rowLen int,
	sLayout, zpLayout shapeinference.BlockParamLayout,
	dst []float32,
) {
	numRows := rEnd - rStart
	if numRows <= 0 || rowLen <= 0 {
		return
	}
	reader := makeWeightRowReader(wBuf, rowLen)
	colStep := sLayout.ColBlockSize
	if wZPs != nil && zpLayout.ColBlockSize < colStep {
		colStep = zpLayout.ColBlockSize
	}
	if colStep <= 0 {
		colStep = rowLen
	}

	dequantRange := func(c0, c1 int) {
		var scratch []int8
		if reader.needsScratch() {
			scratch = make([]int8, rowLen)
		}
		for r := c0; r < c1; r++ {
			row := rStart + r
			dstRow := dst[r*rowLen : (r+1)*rowLen]
			sRowOff := (row / sLayout.RowBlockSize) * sLayout.NumColBlocks
			var zpRowOff int
			if wZPs != nil {
				zpRowOff = (row / zpLayout.RowBlockSize) * zpLayout.NumColBlocks
			}

			if scheme == compute.QuantNF4 {
				lut := &compute.NF4LookupTable
				if reader.isUint8 {
					wRow := reader.rowUint8(row)
					for k0 := 0; k0 < rowLen; k0 += colStep {
						k1 := min(k0+colStep, rowLen)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						for k := k0; k < k1; k++ {
							dstRow[k] = lut[wRow[k]&0x0F] * sc
						}
					}
				} else {
					wRow := reader.rowInt8(row, scratch)
					for k0 := 0; k0 < rowLen; k0 += colStep {
						k1 := min(k0+colStep, rowLen)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						for k := k0; k < k1; k++ {
							dstRow[k] = lut[uint8(wRow[k])&0x0F] * sc
						}
					}
				}
				continue
			}

			// QuantLinear: real = (float32(q) - zp) * scale
			if reader.isUint8 {
				wRow := reader.rowUint8(row)
				if wZPs != nil {
					for k0 := 0; k0 < rowLen; k0 += colStep {
						k1 := min(k0+colStep, rowLen)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+k0/zpLayout.ColBlockSize]
						for k := k0; k < k1; k++ {
							dstRow[k] = (float32(wRow[k]) - zp) * sc
						}
					}
				} else {
					for k0 := 0; k0 < rowLen; k0 += colStep {
						k1 := min(k0+colStep, rowLen)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						for k := k0; k < k1; k++ {
							dstRow[k] = float32(wRow[k]) * sc
						}
					}
				}
			} else {
				wRow := reader.rowInt8(row, scratch)
				if wZPs != nil {
					for k0 := 0; k0 < rowLen; k0 += colStep {
						k1 := min(k0+colStep, rowLen)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+k0/zpLayout.ColBlockSize]
						for k := k0; k < k1; k++ {
							dstRow[k] = (float32(wRow[k]) - zp) * sc
						}
					}
				} else {
					for k0 := 0; k0 < rowLen; k0 += colStep {
						k1 := min(k0+colStep, rowLen)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						for k := k0; k < k1; k++ {
							dstRow[k] = float32(wRow[k]) * sc
						}
					}
				}
			}
		}
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		numRows > 1 && numRows*rowLen >= quantizedDenseMinParallelizeChunk {
		minChunkRows := max(1, quantizedDenseMinParallelizeChunk/max(1, rowLen))
		backend.Scheduler.ParallelizeChunks(numRows, minChunkRows, dequantRange)
		return
	}
	dequantRange(0, numRows)
}

// dequantLinearOrNF4ColsChunk dequantizes column slice [0..K, nStart..nEnd) of a [K, N]
// QuantLinear or QuantNF4 weight tensor into dst of shape [K, nEnd-nStart].
func dequantLinearOrNF4ColsChunk(
	backend *gobackend.Backend,
	scheme compute.QuantizationScheme,
	wBuf *gobackend.Buffer,
	wScales, wZPs []float32,
	K, N, nStart, nEnd int,
	sLayout, zpLayout shapeinference.BlockParamLayout,
	dst []float32,
) {
	colLen := nEnd - nStart
	if K <= 0 || colLen <= 0 {
		return
	}
	reader := makeWeightRowReader(wBuf, N)
	colStep := sLayout.ColBlockSize
	if wZPs != nil && zpLayout.ColBlockSize < colStep {
		colStep = zpLayout.ColBlockSize
	}
	if colStep <= 0 {
		colStep = N
	}

	dequantRange := func(k0, k1 int) {
		var scratch []int8
		if reader.needsScratch() {
			scratch = make([]int8, colLen)
		}
		for k := k0; k < k1; k++ {
			dstRow := dst[k*colLen : (k+1)*colLen]
			sRowOff := (k / sLayout.RowBlockSize) * sLayout.NumColBlocks
			var zpRowOff int
			if wZPs != nil {
				zpRowOff = (k / zpLayout.RowBlockSize) * zpLayout.NumColBlocks
			}

			if scheme == compute.QuantNF4 {
				lut := &compute.NF4LookupTable
				if reader.isUint8 {
					wSpan := reader.spanUint8(k, nStart, colLen)
					for n := nStart; n < nEnd; {
						blkEnd := min(((n/colStep)+1)*colStep, nEnd)
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						for ; n < blkEnd; n++ {
							c := n - nStart
							dstRow[c] = lut[wSpan[c]&0x0F] * sc
						}
					}
				} else {
					wSpan := reader.spanInt8(k, nStart, colLen, scratch)
					for n := nStart; n < nEnd; {
						blkEnd := min(((n/colStep)+1)*colStep, nEnd)
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						for ; n < blkEnd; n++ {
							c := n - nStart
							dstRow[c] = lut[uint8(wSpan[c])&0x0F] * sc
						}
					}
				}
				continue
			}

			if reader.isUint8 {
				wSpan := reader.spanUint8(k, nStart, colLen)
				if wZPs != nil {
					for n := nStart; n < nEnd; {
						blkEnd := min(((n/colStep)+1)*colStep, nEnd)
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+n/zpLayout.ColBlockSize]
						for ; n < blkEnd; n++ {
							c := n - nStart
							dstRow[c] = (float32(wSpan[c]) - zp) * sc
						}
					}
				} else {
					for n := nStart; n < nEnd; {
						blkEnd := min(((n/colStep)+1)*colStep, nEnd)
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						for ; n < blkEnd; n++ {
							c := n - nStart
							dstRow[c] = float32(wSpan[c]) * sc
						}
					}
				}
			} else {
				wSpan := reader.spanInt8(k, nStart, colLen, scratch)
				if wZPs != nil {
					for n := nStart; n < nEnd; {
						blkEnd := min(((n/colStep)+1)*colStep, nEnd)
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+n/zpLayout.ColBlockSize]
						for ; n < blkEnd; n++ {
							c := n - nStart
							dstRow[c] = (float32(wSpan[c]) - zp) * sc
						}
					}
				} else {
					for n := nStart; n < nEnd; {
						blkEnd := min(((n/colStep)+1)*colStep, nEnd)
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						for ; n < blkEnd; n++ {
							c := n - nStart
							dstRow[c] = float32(wSpan[c]) * sc
						}
					}
				}
			}
		}
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		K > 1 && K*colLen >= quantizedDenseMinParallelizeChunk {
		minChunkRows := max(1, quantizedDenseMinParallelizeChunk/max(1, colLen))
		backend.Scheduler.ParallelizeChunks(K, minChunkRows, dequantRange)
		return
	}
	dequantRange(0, K)
}

// --- FusedQuantizedDenseVJP Chunked Float Matmul ---

// execQuantizedVJPOutputsInputMatmul computes dx [M, K] = dZ [M, N] @ dequant(W [N, K])
// using bounded pooled float32 weight chunks and matmul.ExecuteWithEpilogue.
func execQuantizedVJPOutputsInputMatmul(
	backend *gobackend.Backend,
	dZ, dx []float32,
	M, K, N int,
	maxScratchFloats int,
	dequantRowsFn func(nStart, nEnd int, dst []float32),
) error {
	if maxScratchFloats <= 0 {
		maxScratchFloats = defaultQuantizedFloatMaxScratchFloats
	}
	chunkN := quantizedFloatChunkSize(N, K, maxScratchFloats)

	// Single-chunk fast path (N * K <= maxScratchFloats): single matmul straight into dx.
	if chunkN >= N {
		wDeqBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, N*K))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(wDeqBuf)
		wDeq := wDeqBuf.Flat.([]float32)[:N*K]
		dequantRowsFn(0, N, wDeq)
		return matmul.ExecuteWithEpilogue(
			backend, dot.LayoutNonTransposed,
			dZ, wDeq,
			1, M, K, N,
			dx, matmul.Epilogue[float32]{},
		)
	}

	// Multi-chunk path along N: accumulate dx[mStart:mEnd, :] += dZ[mStart:mEnd, nStart:nEnd] @ W[nStart:nEnd, :].
	wDeqBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, chunkN*K))
	if err != nil {
		return err
	}
	defer backend.PutBuffer(wDeqBuf)
	wDeqFull := wDeqBuf.Flat.([]float32)

	mChunk := quantizedFloatChunkSize(M, max(K, chunkN), maxScratchFloats)
	dzSubBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, mChunk*chunkN))
	if err != nil {
		return err
	}
	defer backend.PutBuffer(dzSubBuf)
	dzSubFull := dzSubBuf.Flat.([]float32)

	tmpDXBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, mChunk*K))
	if err != nil {
		return err
	}
	defer backend.PutBuffer(tmpDXBuf)
	tmpDXFull := tmpDXBuf.Flat.([]float32)

	for nStart := 0; nStart < N; nStart += chunkN {
		nEnd := min(nStart+chunkN, N)
		nLen := nEnd - nStart
		wDeq := wDeqFull[:nLen*K]
		dequantRowsFn(nStart, nEnd, wDeq)

		for mStart := 0; mStart < M; mStart += mChunk {
			mEnd := min(mStart+mChunk, M)
			mLen := mEnd - mStart
			dzSub := dzSubFull[:mLen*nLen]
			gatherContiguousColsFloat32(backend, dZ, dzSub, mStart, mLen, N, nStart, nLen)

			dxSlice := dx[mStart*K : mEnd*K]
			if nStart == 0 {
				if err := matmul.ExecuteWithEpilogue(
					backend, dot.LayoutNonTransposed,
					dzSub, wDeq,
					1, mLen, K, nLen,
					dxSlice, matmul.Epilogue[float32]{},
				); err != nil {
					return err
				}
			} else {
				tmpDX := tmpDXFull[:mLen*K]
				if err := matmul.ExecuteWithEpilogue(
					backend, dot.LayoutNonTransposed,
					dzSub, wDeq,
					1, mLen, K, nLen,
					tmpDX, matmul.Epilogue[float32]{},
				); err != nil {
					return err
				}
				addSliceFloat32(backend, dxSlice, tmpDX)
			}
		}
	}
	return nil
}

// execQuantizedVJPInputOutputsMatmul computes dx [M, K] = dZ [M, N] @ dequant(W [K, N])^T
// by chunking along K (producing disjoint output column slices of dx without accumulation).
func execQuantizedVJPInputOutputsMatmul(
	backend *gobackend.Backend,
	dZ, dx []float32,
	M, K, N int,
	maxScratchFloats int,
	dequantRowsFn func(kStart, kEnd int, dst []float32),
) error {
	_, err := execQuantizedFloatChunkedMatmul(
		backend,
		dot.LayoutTransposed,
		dZ, nil, dx,
		M, N, K,
		compute.ActivationConfig{Type: compute.ActivationNone},
		nil,
		maxScratchFloats,
		dequantRowsFn,
	)
	return err
}

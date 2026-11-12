// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
)

func execQuantizedDenseLinearOrNF4Core(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDense,
	in unpackedQDInputs,
	M, K, N int,
	outF32 []float32,
) error {
	wScales, releaseWScale, err := bufferToFloat32Slice(backend, in.wScale)
	if err != nil {
		return err
	}
	defer releaseWScale()

	// Case 1: Pre-quantized XQuantization (QuantLinear Int32 matmul).
	if data.hasXQuant {
		return execQuantizedDenseLinearIntMatmul(backend, data, in, wScales, M, K, N, outF32)
	}

	// Float x input.
	xF32, releaseX, err := bufferToFloat32Slice(backend, in.x)
	if err != nil {
		return err
	}
	defer releaseX()

	var biasF32 []float32
	if in.bias != nil {
		var releaseBias func()
		biasF32, releaseBias, err = bufferToFloat32Slice(backend, in.bias)
		if err != nil {
			return err
		}
		defer releaseBias()
	}

	// Case 2: On-the-fly dynamic Int8 quantization hint (QuantizedMatmul == true) for QuantLinear.
	if data.quantizedMatmul && data.weightScheme == compute.QuantLinear {
		return execQuantizedDenseLinearDynamicInt8(backend, data, xF32, in.weight, wScales, in.wZP, biasF32, M, K, N, outF32)
	}

	// Case 3: Weight-only float matmul (QuantLinear or QuantNF4).
	var wZPF32 []float32
	if in.wZP != nil {
		var releaseWZP func()
		wZPF32, releaseWZP, err = bufferToZeroPointFloat32Slice(backend, in.wZP)
		if err != nil {
			return err
		}
		defer releaseWZP()
	}

	if data.weightLayout == compute.DenseLayoutInputOutputs {
		execWeightOnlyInputOutputs(backend, data.weightScheme, xF32, in.weight, wScales, wZPF32, biasF32, outF32, M, K, N,
			data.info.WeightScaleLayout, data.info.WeightZeroPointLayout)
	} else {
		execWeightOnlyOutputsInput(backend, data.weightScheme, xF32, in.weight, wScales, wZPF32, biasF32, outF32, M, K, N,
			data.info.WeightScaleLayout, data.info.WeightZeroPointLayout)
	}
	return nil
}

// weightRowReader provides zero-allocation row access for Int8, Uint8, and sub-byte [D0, D1] weights.
type weightRowReader struct {
	dtype    dtypes.DType
	isUint8  bool
	int8Data []int8
	u8Data   []uint8
	d1       int
}

func makeWeightRowReader(wBuf *gobackend.Buffer, d1 int) weightRowReader {
	dt := wBuf.RawShape.DType
	switch dt {
	case dtypes.Int8:
		return weightRowReader{dtype: dt, int8Data: wBuf.Flat.([]int8), d1: d1}
	case dtypes.Uint8:
		return weightRowReader{dtype: dt, isUint8: true, u8Data: wBuf.Flat.([]uint8), d1: d1}
	default:
		// Sub-byte packed []byte (Int4, Uint4, Int2, Uint2) unpacks into per-worker []int8 scratch.
		return weightRowReader{dtype: dt, u8Data: wBuf.Flat.([]uint8), d1: d1}
	}
}

func (r *weightRowReader) needsScratch() bool {
	return r.dtype != dtypes.Int8 && r.dtype != dtypes.Uint8
}

func (r *weightRowReader) rowInt8(row int, scratch []int8) []int8 {
	if r.dtype == dtypes.Int8 {
		return r.int8Data[row*r.d1 : (row+1)*r.d1]
	}
	unpackSubByteSpanInt8(r.dtype, r.u8Data, row*r.d1, r.d1, scratch)
	return scratch
}

func (r *weightRowReader) rowUint8(row int) []uint8 {
	return r.u8Data[row*r.d1 : (row+1)*r.d1]
}

// execWeightOnlyInputOutputs computes out = x @ dequant(W) + bias for W of shape [K, N].
// Uses cache-friendly (m, k, n) loop order so both W[k, n] and out[m, n] are accessed with stride-1.
func execWeightOnlyInputOutputs(
	backend *gobackend.Backend,
	scheme compute.QuantizationScheme,
	x []float32,
	wBuf *gobackend.Buffer,
	wScales, wZPs, bias, out []float32,
	M, K, N int,
	sLayout, zpLayout shapeinference.BlockParamLayout,
) {
	reader := makeWeightRowReader(wBuf, N)
	numWorkers := quantizedDenseParallelTileCount(backend, M, K, N)
	var scratchRows [][]int8
	if reader.needsScratch() {
		scratchRows = make([][]int8, numWorkers)
		for i := range scratchRows {
			scratchRows[i] = make([]int8, N)
		}
	}

	quantizedDenseParallel(backend, M, K, N, func(workerIdx, m, nStart, nEnd int) {
		outSlice := out[m*N+nStart : m*N+nEnd]
		if bias != nil {
			copy(outSlice, bias[nStart:nEnd])
		} else {
			clear(outSlice)
		}

		var scratch []int8
		if scratchRows != nil {
			scratch = scratchRows[workerIdx]
		}

		for k := range K {
			xVal := x[m*K+k]
			if xVal == 0 {
				continue
			}
			sRowOff := (k / sLayout.RowBlockSize) * sLayout.NumColBlocks
			var zpRowOff int
			if wZPs != nil {
				zpRowOff = (k / zpLayout.RowBlockSize) * zpLayout.NumColBlocks
			}

			if scheme == compute.QuantNF4 {
				if reader.isUint8 {
					wRow := reader.rowUint8(k)
					for n := nStart; n < nEnd; n++ {
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						outSlice[n-nStart] += xVal * compute.NF4LookupTable[wRow[n]&0x0F] * sc
					}
				} else {
					wRow := reader.rowInt8(k, scratch)
					for n := nStart; n < nEnd; n++ {
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						outSlice[n-nStart] += xVal * compute.NF4LookupTable[uint8(wRow[n])&0x0F] * sc
					}
				}
				continue
			}

			// QuantLinear: real = (float32(q) - zp) * scale
			if reader.isUint8 {
				wRow := reader.rowUint8(k)
				if wZPs != nil {
					for n := nStart; n < nEnd; n++ {
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+n/zpLayout.ColBlockSize]
						outSlice[n-nStart] += xVal * (float32(wRow[n]) - zp) * sc
					}
				} else {
					for n := nStart; n < nEnd; n++ {
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						outSlice[n-nStart] += xVal * float32(wRow[n]) * sc
					}
				}
			} else {
				wRow := reader.rowInt8(k, scratch)
				if wZPs != nil {
					for n := nStart; n < nEnd; n++ {
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						zp := wZPs[zpRowOff+n/zpLayout.ColBlockSize]
						outSlice[n-nStart] += xVal * (float32(wRow[n]) - zp) * sc
					}
				} else {
					for n := nStart; n < nEnd; n++ {
						sc := wScales[sRowOff+n/sLayout.ColBlockSize]
						outSlice[n-nStart] += xVal * float32(wRow[n]) * sc
					}
				}
			}
		}
	})
}

// execWeightOnlyOutputsInput computes out = x @ dequant(W)^T + bias for W of shape [N, K].
// Uses cache-friendly (m, n, k) loop order so both x[m, k] and W[n, k] are accessed with stride-1,
// hoisting block scale/zeroPoint outside the inner k-block dot product.
func execWeightOnlyOutputsInput(
	backend *gobackend.Backend,
	scheme compute.QuantizationScheme,
	x []float32,
	wBuf *gobackend.Buffer,
	wScales, wZPs, bias, out []float32,
	M, K, N int,
	sLayout, zpLayout shapeinference.BlockParamLayout,
) {
	reader := makeWeightRowReader(wBuf, K)
	numWorkers := quantizedDenseParallelTileCount(backend, M, K, N)
	var scratchRows [][]int8
	if reader.needsScratch() {
		scratchRows = make([][]int8, numWorkers)
		for i := range scratchRows {
			scratchRows[i] = make([]int8, K)
		}
	}

	kStep := sLayout.ColBlockSize
	if wZPs != nil && zpLayout.ColBlockSize < kStep {
		kStep = zpLayout.ColBlockSize
	}

	quantizedDenseParallel(backend, M, K, N, func(workerIdx, m, nStart, nEnd int) {
		var scratch []int8
		if scratchRows != nil {
			scratch = scratchRows[workerIdx]
		}
		xRow := x[m*K : (m+1)*K]

		for n := nStart; n < nEnd; n++ {
			sRowOff := (n / sLayout.RowBlockSize) * sLayout.NumColBlocks
			var zpRowOff int
			if wZPs != nil {
				zpRowOff = (n / zpLayout.RowBlockSize) * zpLayout.NumColBlocks
			}

			var dot float32
			if scheme == compute.QuantNF4 {
				if reader.isUint8 {
					wRow := reader.rowUint8(n)
					for k0 := 0; k0 < K; k0 += kStep {
						k1 := min(k0+kStep, K)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						var sub float32
						for k := k0; k < k1; k++ {
							sub += xRow[k] * compute.NF4LookupTable[wRow[k]&0x0F]
						}
						dot += sub * sc
					}
				} else {
					wRow := reader.rowInt8(n, scratch)
					for k0 := 0; k0 < K; k0 += kStep {
						k1 := min(k0+kStep, K)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						var sub float32
						for k := k0; k < k1; k++ {
							sub += xRow[k] * compute.NF4LookupTable[uint8(wRow[k])&0x0F]
						}
						dot += sub * sc
					}
				}
			} else {
				if reader.isUint8 {
					wRow := reader.rowUint8(n)
					for k0 := 0; k0 < K; k0 += kStep {
						k1 := min(k0+kStep, K)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						var zp float32
						if wZPs != nil {
							zp = wZPs[zpRowOff+k0/zpLayout.ColBlockSize]
						}
						var sub float32
						for k := k0; k < k1; k++ {
							sub += xRow[k] * (float32(wRow[k]) - zp)
						}
						dot += sub * sc
					}
				} else {
					wRow := reader.rowInt8(n, scratch)
					for k0 := 0; k0 < K; k0 += kStep {
						k1 := min(k0+kStep, K)
						sc := wScales[sRowOff+k0/sLayout.ColBlockSize]
						var zp float32
						if wZPs != nil {
							zp = wZPs[zpRowOff+k0/zpLayout.ColBlockSize]
						}
						var sub float32
						for k := k0; k < k1; k++ {
							sub += xRow[k] * (float32(wRow[k]) - zp)
						}
						dot += sub * sc
					}
				}
			}

			if bias != nil {
				dot += bias[n]
			}
			out[m*N+n] = dot
		}
	})
}

// execQuantizedDenseLinearIntMatmul executes exact Int32 matmul when XQuantization != nil (QuantLinear).
func execQuantizedDenseLinearIntMatmul(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDense,
	in unpackedQDInputs,
	wScales []float32,
	M, K, N int,
	outF32 []float32,
) error {
	xScales, releaseXScale, err := bufferToFloat32Slice(backend, in.xScale)
	if err != nil {
		return err
	}
	defer releaseXScale()

	var xZPs []int32
	if in.xZP != nil {
		var releaseXZP func()
		xZPs, releaseXZP, err = bufferToZeroPointInt32Slice(backend, in.xZP)
		if err != nil {
			return err
		}
		defer releaseXZP()
	}

	var wZPs []int32
	if in.wZP != nil {
		var releaseWZP func()
		wZPs, releaseWZP, err = bufferToZeroPointInt32Slice(backend, in.wZP)
		if err != nil {
			return err
		}
		defer releaseWZP()
	}

	var biasI32 []int32
	var biasF32 []float32
	if in.bias != nil {
		if in.bias.RawShape.DType == dtypes.Int32 {
			biasI32 = in.bias.Flat.([]int32)
		} else {
			var releaseBias func()
			biasF32, releaseBias, err = bufferToFloat32Slice(backend, in.bias)
			if err != nil {
				return err
			}
			defer releaseBias()
		}
	}

	xIsUint8 := in.x.RawShape.DType == dtypes.Uint8
	var xI8 []int8
	var xU8 []uint8
	if xIsUint8 {
		xU8 = in.x.Flat.([]uint8)
	} else {
		xI8 = in.x.Flat.([]int8)
	}

	xSLayout := data.info.XScaleLayout
	xZPLayout := data.info.XZeroPointLayout
	wSLayout := data.info.WeightScaleLayout
	wZPLayout := data.info.WeightZeroPointLayout

	if data.weightLayout == compute.DenseLayoutInputOutputs {
		reader := makeWeightRowReader(in.weight, N)
		numWorkers := quantizedDenseParallelTileCount(backend, M, K, N)
		accBufs := make([][]int32, numWorkers)
		for i := range accBufs {
			accBufs[i] = make([]int32, N)
		}
		var scratchRows [][]int8
		if reader.needsScratch() {
			scratchRows = make([][]int8, numWorkers)
			for i := range scratchRows {
				scratchRows[i] = make([]int8, N)
			}
		}

		quantizedDenseParallel(backend, M, K, N, func(workerIdx, m, nStart, nEnd int) {
			tileW := nEnd - nStart
			acc := accBufs[workerIdx][:tileW]
			outSlice := outF32[m*N+nStart : m*N+nEnd]
			clear(outSlice)

			var scratch []int8
			if scratchRows != nil {
				scratch = scratchRows[workerIdx]
			}

			for k0 := 0; k0 < K; {
				nextX := ((k0 / xSLayout.ColBlockSize) + 1) * xSLayout.ColBlockSize
				nextW := ((k0 / wSLayout.RowBlockSize) + 1) * wSLayout.RowBlockSize
				k1 := min(K, min(nextX, nextW))

				clear(acc)
				xSc := xScales[xSLayout.Index(m, k0)]
				var xZP int32
				if xZPs != nil {
					xZP = xZPs[xZPLayout.Index(m, k0)]
				}
				wSRowOff := (k0 / wSLayout.RowBlockSize) * wSLayout.NumColBlocks
				var wZPRowOff int
				if wZPs != nil {
					wZPRowOff = (k0 / wZPLayout.RowBlockSize) * wZPLayout.NumColBlocks
				}

				for k := k0; k < k1; k++ {
					var xDiff int32
					if xIsUint8 {
						xDiff = int32(xU8[m*K+k]) - xZP
					} else {
						xDiff = int32(xI8[m*K+k]) - xZP
					}
					if xDiff == 0 {
						continue
					}
					if reader.isUint8 {
						wRow := reader.rowUint8(k)
						if wZPs != nil {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xDiff * (int32(wRow[n]) - wZPs[wZPRowOff+n/wZPLayout.ColBlockSize])
							}
						} else {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xDiff * int32(wRow[n])
							}
						}
					} else {
						wRow := reader.rowInt8(k, scratch)
						if wZPs != nil {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xDiff * (int32(wRow[n]) - wZPs[wZPRowOff+n/wZPLayout.ColBlockSize])
							}
						} else {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xDiff * int32(wRow[n])
							}
						}
					}
				}

				for n := nStart; n < nEnd; n++ {
					sumI := acc[n-nStart]
					if k0 == 0 && biasI32 != nil {
						sumI += biasI32[n]
					}
					wSc := wScales[wSRowOff+n/wSLayout.ColBlockSize]
					outSlice[n-nStart] += float32(sumI) * (xSc * wSc)
				}
				k0 = k1
			}

			if biasF32 != nil {
				for n := nStart; n < nEnd; n++ {
					outSlice[n-nStart] += biasF32[n]
				}
			}
		})
		return nil
	}

	// DenseLayoutOutputsInput: weight is [N, K].
	reader := makeWeightRowReader(in.weight, K)
	numWorkers := quantizedDenseParallelTileCount(backend, M, K, N)
	var scratchRows [][]int8
	if reader.needsScratch() {
		scratchRows = make([][]int8, numWorkers)
		for i := range scratchRows {
			scratchRows[i] = make([]int8, K)
		}
	}

	quantizedDenseParallel(backend, M, K, N, func(workerIdx, m, nStart, nEnd int) {
		var scratch []int8
		if scratchRows != nil {
			scratch = scratchRows[workerIdx]
		}
		for n := nStart; n < nEnd; n++ {
			wSRowOff := (n / wSLayout.RowBlockSize) * wSLayout.NumColBlocks
			var wZPRowOff int
			if wZPs != nil {
				wZPRowOff = (n / wZPLayout.RowBlockSize) * wZPLayout.NumColBlocks
			}

			var wRowI8 []int8
			var wRowU8 []uint8
			if reader.isUint8 {
				wRowU8 = reader.rowUint8(n)
			} else {
				wRowI8 = reader.rowInt8(n, scratch)
			}

			var outVal float32
			for k0 := 0; k0 < K; {
				nextX := ((k0 / xSLayout.ColBlockSize) + 1) * xSLayout.ColBlockSize
				nextW := ((k0 / wSLayout.ColBlockSize) + 1) * wSLayout.ColBlockSize
				k1 := min(K, min(nextX, nextW))

				xSc := xScales[xSLayout.Index(m, k0)]
				var xZP int32
				if xZPs != nil {
					xZP = xZPs[xZPLayout.Index(m, k0)]
				}
				wSc := wScales[wSRowOff+k0/wSLayout.ColBlockSize]
				var wZP int32
				if wZPs != nil {
					wZP = wZPs[wZPRowOff+k0/wZPLayout.ColBlockSize]
				}

				var acc int32
				if xIsUint8 {
					if reader.isUint8 {
						for k := k0; k < k1; k++ {
							acc += (int32(xU8[m*K+k]) - xZP) * (int32(wRowU8[k]) - wZP)
						}
					} else {
						for k := k0; k < k1; k++ {
							acc += (int32(xU8[m*K+k]) - xZP) * (int32(wRowI8[k]) - wZP)
						}
					}
				} else {
					if reader.isUint8 {
						for k := k0; k < k1; k++ {
							acc += (int32(xI8[m*K+k]) - xZP) * (int32(wRowU8[k]) - wZP)
						}
					} else {
						for k := k0; k < k1; k++ {
							acc += (int32(xI8[m*K+k]) - xZP) * (int32(wRowI8[k]) - wZP)
						}
					}
				}
				if k0 == 0 && biasI32 != nil {
					acc += biasI32[n]
				}
				outVal += float32(acc) * (xSc * wSc)
				k0 = k1
			}
			if biasF32 != nil {
				outVal += biasF32[n]
			}
			outF32[m*N+n] = outVal
		}
	})
	return nil
}

// execQuantizedDenseLinearDynamicInt8 dynamically quantizes float32 x rows on-the-fly to symmetric Int8
// per k-block when QuantizedMatmul == true and XQuantization == nil.
func execQuantizedDenseLinearDynamicInt8(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDense,
	xF32 []float32,
	wBuf *gobackend.Buffer,
	wScales []float32,
	wZPBuf *gobackend.Buffer,
	biasF32 []float32,
	M, K, N int,
	outF32 []float32,
) error {
	var wZPs []int32
	var wZPF32 []float32
	if wZPBuf != nil {
		if wZPBuf.RawShape.DType.IsFloat() {
			// Non-integer float zero-point: fall back cleanly to exact float arithmetic.
			var releaseZP func()
			var err error
			wZPF32, releaseZP, err = bufferToZeroPointFloat32Slice(backend, wZPBuf)
			if err != nil {
				return err
			}
			defer releaseZP()
			if data.weightLayout == compute.DenseLayoutInputOutputs {
				execWeightOnlyInputOutputs(backend, compute.QuantLinear, xF32, wBuf, wScales, wZPF32, biasF32, outF32, M, K, N,
					data.info.WeightScaleLayout, data.info.WeightZeroPointLayout)
			} else {
				execWeightOnlyOutputsInput(backend, compute.QuantLinear, xF32, wBuf, wScales, wZPF32, biasF32, outF32, M, K, N,
					data.info.WeightScaleLayout, data.info.WeightZeroPointLayout)
			}
			return nil
		}
		var releaseZP func()
		var err error
		wZPs, releaseZP, err = bufferToZeroPointInt32Slice(backend, wZPBuf)
		if err != nil {
			return err
		}
		defer releaseZP()
	}

	wSLayout := data.info.WeightScaleLayout
	wZPLayout := data.info.WeightZeroPointLayout
	var kBlock int
	if data.weightLayout == compute.DenseLayoutInputOutputs {
		kBlock = wSLayout.RowBlockSize
	} else {
		kBlock = wSLayout.ColBlockSize
	}
	numKBlocks := (K + kBlock - 1) / kBlock

	// Dynamically quantize xF32 [M, K] -> xQ [M, K] int8 + xScales [M, numKBlocks] float32.
	xQ := make([]int8, M*K)
	xScales := make([]float32, M*numKBlocks)
	for m := range M {
		for kb := range numKBlocks {
			k0 := kb * kBlock
			k1 := min(k0+kBlock, K)
			xSpan := xF32[m*K+k0 : m*K+k1]
			var maxAbs float32
			for _, v := range xSpan {
				av := float32(math.Abs(float64(v)))
				if av > maxAbs {
					maxAbs = av
				}
			}
			if maxAbs == 0 {
				xScales[m*numKBlocks+kb] = 0
				continue
			}
			sc := maxAbs / 127.0
			invSc := 127.0 / maxAbs
			xScales[m*numKBlocks+kb] = sc
			dst := xQ[m*K+k0 : m*K+k1]
			for i, v := range xSpan {
				q := int32(math.RoundToEven(float64(v * invSc)))
				q = max(-127, min(127, q))
				dst[i] = int8(q)
			}
		}
	}

	// Execute integer accumulation using the dynamically quantized xQ & xScales.
	if data.weightLayout == compute.DenseLayoutInputOutputs {
		reader := makeWeightRowReader(wBuf, N)
		numWorkers := quantizedDenseParallelTileCount(backend, M, K, N)
		accBufs := make([][]int32, numWorkers)
		for i := range accBufs {
			accBufs[i] = make([]int32, N)
		}
		var scratchRows [][]int8
		if reader.needsScratch() {
			scratchRows = make([][]int8, numWorkers)
			for i := range scratchRows {
				scratchRows[i] = make([]int8, N)
			}
		}
		quantizedDenseParallel(backend, M, K, N, func(workerIdx, m, nStart, nEnd int) {
			acc := accBufs[workerIdx][:nEnd-nStart]
			outSlice := outF32[m*N+nStart : m*N+nEnd]
			if biasF32 != nil {
				copy(outSlice, biasF32[nStart:nEnd])
			} else {
				clear(outSlice)
			}
			var scratch []int8
			if scratchRows != nil {
				scratch = scratchRows[workerIdx]
			}
			for kb := range numKBlocks {
				xSc := xScales[m*numKBlocks+kb]
				if xSc == 0 {
					continue
				}
				k0 := kb * kBlock
				k1 := min(k0+kBlock, K)
				clear(acc)
				wSRowOff := kb * wSLayout.NumColBlocks
				var wZPRowOff int
				if wZPs != nil {
					wZPRowOff = (k0 / wZPLayout.RowBlockSize) * wZPLayout.NumColBlocks
				}
				for k := k0; k < k1; k++ {
					xq := int32(xQ[m*K+k])
					if xq == 0 {
						continue
					}
					if reader.isUint8 {
						wRow := reader.rowUint8(k)
						if wZPs != nil {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xq * (int32(wRow[n]) - wZPs[wZPRowOff+n/wZPLayout.ColBlockSize])
							}
						} else {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xq * int32(wRow[n])
							}
						}
					} else {
						wRow := reader.rowInt8(k, scratch)
						if wZPs != nil {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xq * (int32(wRow[n]) - wZPs[wZPRowOff+n/wZPLayout.ColBlockSize])
							}
						} else {
							for n := nStart; n < nEnd; n++ {
								acc[n-nStart] += xq * int32(wRow[n])
							}
						}
					}
				}
				for n := nStart; n < nEnd; n++ {
					outSlice[n-nStart] += float32(acc[n-nStart]) * (xSc * wScales[wSRowOff+n/wSLayout.ColBlockSize])
				}
			}
		})
		return nil
	}

	// DenseLayoutOutputsInput: weight is [N, K].
	reader := makeWeightRowReader(wBuf, K)
	numWorkers := quantizedDenseParallelTileCount(backend, M, K, N)
	var scratchRows [][]int8
	if reader.needsScratch() {
		scratchRows = make([][]int8, numWorkers)
		for i := range scratchRows {
			scratchRows[i] = make([]int8, K)
		}
	}
	quantizedDenseParallel(backend, M, K, N, func(workerIdx, m, nStart, nEnd int) {
		var scratch []int8
		if scratchRows != nil {
			scratch = scratchRows[workerIdx]
		}
		for n := nStart; n < nEnd; n++ {
			wSRowOff := (n / wSLayout.RowBlockSize) * wSLayout.NumColBlocks
			var wZPRowOff int
			if wZPs != nil {
				wZPRowOff = (n / wZPLayout.RowBlockSize) * wZPLayout.NumColBlocks
			}
			var wRowI8 []int8
			var wRowU8 []uint8
			if reader.isUint8 {
				wRowU8 = reader.rowUint8(n)
			} else {
				wRowI8 = reader.rowInt8(n, scratch)
			}
			var outVal float32
			if biasF32 != nil {
				outVal = biasF32[n]
			}
			for kb := range numKBlocks {
				xSc := xScales[m*numKBlocks+kb]
				if xSc == 0 {
					continue
				}
				k0 := kb * kBlock
				k1 := min(k0+kBlock, K)
				wSc := wScales[wSRowOff+kb]
				var wZP int32
				if wZPs != nil {
					wZP = wZPs[wZPRowOff+k0/wZPLayout.ColBlockSize]
				}
				var acc int32
				if reader.isUint8 {
					for k := k0; k < k1; k++ {
						acc += int32(xQ[m*K+k]) * (int32(wRowU8[k]) - wZP)
					}
				} else {
					for k := k0; k < k1; k++ {
						acc += int32(xQ[m*K+k]) * (int32(wRowI8[k]) - wZP)
					}
				}
				outVal += float32(acc) * (xSc * wSc)
			}
			outF32[m*N+n] = outVal
		}
	})
	return nil
}

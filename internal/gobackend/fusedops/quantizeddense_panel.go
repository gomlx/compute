// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"encoding/binary"
	"math"
	"sync"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/shapes"
)

const (
	// QuantizedPanelStripCols aliases gobackend.QuantizedPanelStripCols (16).
	QuantizedPanelStripCols = gobackend.QuantizedPanelStripCols

	// quantizedPanelMaxScratchBytes caps the temporary unpacked weight panel buffer (~2 MB)
	// when quantized_cache is disabled (the default) or weights are dynamic.
	quantizedPanelMaxScratchBytes = 2 * 1024 * 1024

	// quantizedPanelMinM is 1 so even M == 1 (single-token decode) leverages 1x32/1x16 SIMD/VNNI
	// strip-parallel microkernels and K4 panels.
	quantizedPanelMinM = 1
)

type linearConstantPanelCache struct {
	once  sync.Once
	panel []uint8
}

type sharedQuantActEntry struct {
	xBuf    *gobackend.Buffer
	M       int
	claims  int
	active  int
	ready   chan struct{}
	qxBuf   *gobackend.Buffer
	dxBuf   *gobackend.Buffer
	qx      []int8
	dxPairs []float32
	err     error
}

// sharedQuantActGroup coordinates sibling FusedQuantizedDense projections (such as Transformer
// Q/K/V or SwiGLU gate/up) that share the exact same input x Buffer and activation quantization
// layout, ensuring x is quantized once per forward step across both parallel and sequential modes.
type sharedQuantActGroup struct {
	numConsumers int
	mu           sync.Mutex
	entries      []*sharedQuantActEntry
}

func (g *sharedQuantActGroup) acquire(
	backend *gobackend.Backend,
	xBuf *gobackend.Buffer,
	M, qxLen, dxLen int,
	quantizeFn func(qx []int8, dxPairs []float32) error,
) ([]int8, []float32, func(), error) {
	if g == nil || xBuf == nil || g.numConsumers < 2 {
		qxBuf, err := backend.GetBuffer(shapes.Make(dtypes.Int8, qxLen))
		if err != nil {
			return nil, nil, nil, err
		}
		dxBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, dxLen))
		if err != nil {
			backend.PutBuffer(qxBuf)
			return nil, nil, nil, err
		}
		qx := qxBuf.Flat.([]int8)
		dxPairs := dxBuf.Flat.([]float32)
		if err := quantizeFn(qx, dxPairs); err != nil {
			backend.PutBuffer(dxBuf)
			backend.PutBuffer(qxBuf)
			return nil, nil, nil, err
		}
		return qx, dxPairs, func() {
			backend.PutBuffer(dxBuf)
			backend.PutBuffer(qxBuf)
		}, nil
	}

	g.mu.Lock()
	for _, e := range g.entries {
		if e.xBuf == xBuf && e.M == M && e.claims < g.numConsumers {
			e.claims++
			e.active++
			g.mu.Unlock()
			<-e.ready
			if e.err != nil {
				g.release(backend, e)
				return nil, nil, nil, e.err
			}
			return e.qx, e.dxPairs, func() { g.release(backend, e) }, nil
		}
	}
	entry := &sharedQuantActEntry{
		xBuf:   xBuf,
		M:      M,
		claims: 1,
		active: 1,
		ready:  make(chan struct{}),
	}
	g.entries = append(g.entries, entry)
	g.mu.Unlock()

	qxBuf, err := backend.GetBuffer(shapes.Make(dtypes.Int8, qxLen))
	if err == nil {
		var dxBuf *gobackend.Buffer
		dxBuf, err = backend.GetBuffer(shapes.Make(dtypes.Float32, dxLen))
		if err == nil {
			qx := qxBuf.Flat.([]int8)
			dxPairs := dxBuf.Flat.([]float32)
			if err = quantizeFn(qx, dxPairs); err == nil {
				entry.qxBuf = qxBuf
				entry.dxBuf = dxBuf
				entry.qx = qx
				entry.dxPairs = dxPairs
				close(entry.ready)
				return qx, dxPairs, func() { g.release(backend, entry) }, nil
			}
			backend.PutBuffer(dxBuf)
		}
		backend.PutBuffer(qxBuf)
	}

	g.mu.Lock()
	entry.err = err
	entry.claims = g.numConsumers
	entry.active--
	if entry.active == 0 {
		g.removeEntryLocked(entry)
	}
	close(entry.ready)
	g.mu.Unlock()
	return nil, nil, nil, err
}

func (g *sharedQuantActGroup) release(backend *gobackend.Backend, entry *sharedQuantActEntry) {
	var qxBuf, dxBuf *gobackend.Buffer
	g.mu.Lock()
	entry.active--
	if entry.active == 0 && entry.claims >= g.numConsumers {
		qxBuf = entry.qxBuf
		dxBuf = entry.dxBuf
		entry.qxBuf = nil
		entry.dxBuf = nil
		g.removeEntryLocked(entry)
	}
	g.mu.Unlock()
	if dxBuf != nil {
		backend.PutBuffer(dxBuf)
	}
	if qxBuf != nil {
		backend.PutBuffer(qxBuf)
	}
}

func (g *sharedQuantActGroup) removeEntryLocked(target *sharedQuantActEntry) {
	for i, e := range g.entries {
		if e == target {
			last := len(g.entries) - 1
			g.entries[i] = g.entries[last]
			g.entries[last] = nil
			g.entries = g.entries[:last]
			return
		}
	}
}

func linearPanelKBlock(data *nodeFusedQuantizedDense) (int, bool) {
	if data == nil || data.weightScheme != compute.QuantLinear || data.hasXQuant || !data.quantizedMatmul {
		return 0, false
	}
	if data.info.K <= 0 || data.info.N <= 0 {
		return 0, false
	}
	wSLayout := data.info.WeightScaleLayout
	wZPLayout := data.info.WeightZeroPointLayout
	isIO := data.weightLayout == compute.DenseLayoutInputOutputs
	var kBlock int
	if isIO {
		kBlock = wSLayout.RowBlockSize
	} else {
		kBlock = wSLayout.ColBlockSize
	}
	if kBlock <= 0 {
		return 0, false
	}
	if data.hasWZP {
		var zpKBlock int
		if isIO {
			zpKBlock = wZPLayout.RowBlockSize
		} else {
			zpKBlock = wZPLayout.ColBlockSize
		}
		if zpKBlock <= 0 || zpKBlock < kBlock || zpKBlock%kBlock != 0 {
			return 0, false
		}
	}
	return kBlock, true
}

func canCacheQuantLinearPanel(data *nodeFusedQuantizedDense) bool {
	_, ok := linearPanelKBlock(data)
	return ok && data.info.K > 0 && data.info.N > 0
}

// QuantizedPanelGEMMConfig aliases gobackend.QuantizedPanelGEMMConfig.
type QuantizedPanelGEMMConfig = gobackend.QuantizedPanelGEMMConfig

// QuantizedPanelStripSubBlockBytes returns the byte size of one sub-block in a 16-column strip.
func QuantizedPanelStripSubBlockBytes(quadsPerSubBlk int) int {
	return gobackend.QuantizedPanelStripSubBlockBytes(quadsPerSubBlk)
}

func selectQuantizedPanelColChunk(N, numSubBlocks, quadsPerSubBlk int) int {
	stripBytes := numSubBlocks * QuantizedPanelStripSubBlockBytes(quadsPerSubBlk)
	if stripBytes <= 0 {
		return N
	}
	totalStrips := (N + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
	if totalStrips*stripBytes <= quantizedPanelMaxScratchBytes {
		return N
	}
	// Align chunk strips to a multiple of 2 (32 columns) for AVX-512 2-strip kernels.
	maxStrips := max(quantizedPanelMaxScratchBytes/stripBytes, 2)
	maxStrips &^= 1
	if maxStrips < 2 {
		maxStrips = 2
	}
	return min(N, maxStrips*QuantizedPanelStripCols)
}

func dispatchQuantizedPanelGEMM(cfg *QuantizedPanelGEMMConfig) {
	if fn := gobackend.GetQuantizedPanelGEMMArchDispatcher(); fn != nil {
		fn(cfg)
		return
	}
	gobackend.ExecQuantizedPanelGEMMScalar(cfg)
}

// execQuantizedDenseLinearPanelDynamicInt8 executes QuantLinear with QuantizedMatmul == true
// using the unified K4-interleaved integer GEMM microkernel.
// Returns (true, nil) when handled, or (false, nil) if block layouts are non-uniform.
func execQuantizedDenseLinearPanelDynamicInt8(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDense,
	xBuf *gobackend.Buffer,
	wBuf *gobackend.Buffer,
	wScales []float32,
	wZPBuf *gobackend.Buffer,
	biasF32 []float32,
	M, K, N int,
	outF32 []float32,
	inTileAct activations.KernelFn[float32],
) (bool, error) {
	if M < quantizedPanelMinM || K == 0 || N == 0 {
		return false, nil
	}

	kBlock, ok := linearPanelKBlock(data)
	if !ok {
		return false, nil
	}
	wSLayout := data.info.WeightScaleLayout
	wZPLayout := data.info.WeightZeroPointLayout
	isIO := data.weightLayout == compute.DenseLayoutInputOutputs

	numSubBlocks := (K + kBlock - 1) / kBlock
	bPad := (kBlock + 3) &^ 3
	quadsPerSubBlk := bPad / 4
	kPadded := numSubBlocks * bPad
	subBlkBytes := QuantizedPanelStripSubBlockBytes(quadsPerSubBlk)
	stripBytes := numSubBlocks * subBlkBytes

	qx, dxPairs, releaseAct, err := data.actShareGroup.acquire(
		backend, xBuf, M, M*kPadded, M*numSubBlocks*2,
		func(qxDst []int8, dxDst []float32) error {
			xF32, releaseX, convErr := bufferToFloat32Slice(backend, xBuf)
			if convErr != nil {
				return convErr
			}
			defer releaseX()
			quantizeSymmetricInt8PanelRows(backend, xF32, qxDst, dxDst, M, K, kBlock, kPadded, numSubBlocks)
			return nil
		},
	)
	if err != nil {
		return false, err
	}
	defer releaseAct()

	wDT := wBuf.RawShape.DType
	wideUnsigned := wDT == dtypes.Int8 || wDT == dtypes.Uint8

	if data.linearPanelCache != nil {
		var buildErr error
		data.linearPanelCache.once.Do(func() {
			var wZPF32 []float32
			if wZPBuf != nil {
				var releaseZP func()
				wZPF32, releaseZP, buildErr = bufferToZeroPointFloat32Slice(backend, wZPBuf)
				if buildErr != nil {
					return
				}
				defer releaseZP()
			}
			totalStrips := (N + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
			cached := make([]uint8, totalStrips*stripBytes)
			unpackQuantLinearStrips(
				backend, wBuf, isIO, wScales, wZPF32, wSLayout, wZPLayout,
				K, N, 0, N, kBlock, quadsPerSubBlk, numSubBlocks, cached,
			)
			data.linearPanelCache.panel = cached
		})
		if buildErr != nil {
			return false, buildErr
		}
		var chunkBias []float32
		if biasF32 != nil {
			chunkBias = biasF32[:N]
		}
		cfg := QuantizedPanelGEMMConfig{
			Backend:        backend,
			Qx:             qx,
			DxPairs:        dxPairs,
			Panel:          data.linearPanelCache.panel,
			Bias:           chunkBias,
			Out:            outF32,
			M:              M,
			N:              N,
			NStart:         0,
			ChunkCols:      N,
			KPadded:        kPadded,
			NumSubBlocks:   numSubBlocks,
			QuadsPerSubBlk: quadsPerSubBlk,
			WideUnsigned:   wideUnsigned,
			ActFn:          inTileAct,
		}
		dispatchQuantizedPanelGEMM(&cfg)
		return true, nil
	}

	var wZPF32 []float32
	if wZPBuf != nil {
		var releaseZP func()
		wZPF32, releaseZP, err = bufferToZeroPointFloat32Slice(backend, wZPBuf)
		if err != nil {
			return false, err
		}
		defer releaseZP()
	}

	colChunk := selectQuantizedPanelColChunk(N, numSubBlocks, quadsPerSubBlk)
	maxStrips := (colChunk + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols

	panelBuf, err := backend.GetBuffer(shapes.Make(dtypes.Uint8, maxStrips*stripBytes))
	if err != nil {
		return false, err
	}
	defer backend.PutBuffer(panelBuf)
	panel := panelBuf.Flat.([]uint8)

	for nStart := 0; nStart < N; nStart += colChunk {
		chunkCols := min(colChunk, N-nStart)
		numStrips := (chunkCols + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
		unpackQuantLinearStrips(
			backend, wBuf, isIO, wScales, wZPF32, wSLayout, wZPLayout,
			K, N, nStart, chunkCols, kBlock, quadsPerSubBlk, numSubBlocks,
			panel[:numStrips*stripBytes],
		)

		var chunkBias []float32
		if biasF32 != nil {
			chunkBias = biasF32[nStart : nStart+chunkCols]
		}
		cfg := QuantizedPanelGEMMConfig{
			Backend:        backend,
			Qx:             qx,
			DxPairs:        dxPairs,
			Panel:          panel[:numStrips*stripBytes],
			Bias:           chunkBias,
			Out:            outF32,
			M:              M,
			N:              N,
			NStart:         nStart,
			ChunkCols:      chunkCols,
			KPadded:        kPadded,
			NumSubBlocks:   numSubBlocks,
			QuadsPerSubBlk: quadsPerSubBlk,
			WideUnsigned:   wideUnsigned,
			ActFn:          inTileAct,
		}
		dispatchQuantizedPanelGEMM(&cfg)
	}
	return true, nil
}

func quantizeSymmetricInt8PanelRows(
	backend *gobackend.Backend,
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	M, K, kBlock, kPadded, numSubBlocks int,
) {
	fn := quantizeSymmetricInt8PanelRowsScalar
	if archFn := gobackend.GetQuantizeSymmetricInt8ActivationsArchDispatcher(); archFn != nil {
		fn = archFn
	}
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		M*K >= quantizedDenseMinParallelizeChunk && M > 1 {
		backend.Scheduler.ParallelizeChunks(M, 1, func(mStart, mEnd int) {
			fn(xF32, qx, dxPairs, mStart, mEnd, K, kBlock, kPadded, numSubBlocks)
		})
		return
	}
	fn(xF32, qx, dxPairs, 0, M, K, kBlock, kPadded, numSubBlocks)
}

func quantizeSymmetricInt8PanelRowsScalar(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, kBlock, kPadded, numSubBlocks int,
) {
	bPad := (kBlock + 3) &^ 3
	for m := mStart; m < mEnd; m++ {
		xRow := xF32[m*K : (m+1)*K]
		qRow := qx[m*kPadded : (m+1)*kPadded]
		dRow := dxPairs[m*numSubBlocks*2 : (m+1)*numSubBlocks*2]
		for kb := range numSubBlocks {
			k0 := kb * kBlock
			k1 := min(k0+kBlock, K)
			span := xRow[k0:k1]
			dst := qRow[kb*bPad : (kb+1)*bPad]
			var maxAbs float32
			for _, v := range span {
				av := math.Abs(float64(v))
				if float32(av) > maxAbs {
					maxAbs = float32(av)
				}
			}
			if maxAbs == 0 {
				clear(dst)
				dRow[2*kb] = 0
				dRow[2*kb+1] = 0
				continue
			}
			sc := maxAbs / 127.0
			invSc := 127.0 / maxAbs
			var sumQ int32
			for i, v := range span {
				q := int32(math.RoundToEven(float64(v * invSc)))
				q = max(-127, min(127, q))
				dst[i] = int8(q)
				sumQ += q
			}
			if len(span) < bPad {
				clear(dst[len(span):])
			}
			dRow[2*kb] = sc
			dRow[2*kb+1] = sc * float32(sumQ)
		}
	}
}

func quantLinearShiftAndReader(wBuf *gobackend.Buffer) (shiftC int32, isUint8 bool) {
	switch wBuf.RawShape.DType {
	case dtypes.Uint8, dtypes.Uint4, dtypes.Uint2:
		return 0, wBuf.RawShape.DType == dtypes.Uint8
	case dtypes.Int8:
		return 128, false
	case dtypes.Int4:
		return 8, false
	case dtypes.Int2:
		return 2, false
	default:
		return 0, false
	}
}

func unpackQuantLinearStrips(
	backend *gobackend.Backend,
	wBuf *gobackend.Buffer,
	isIO bool,
	wScales, wZPF32 []float32,
	wSLayout, wZPLayout interface {
		Index(row, col int) int
	},
	K, N, nStart, chunkCols, kBlock, quadsPerSubBlk, numSubBlocks int,
	panel []uint8,
) {
	shiftC, _ := quantLinearShiftAndReader(wBuf)
	shiftF32 := float32(shiftC)
	bPad := quadsPerSubBlk * 4
	subBlkBytes := QuantizedPanelStripSubBlockBytes(quadsPerSubBlk)
	stripBytes := numSubBlocks * subBlkBytes
	numStrips := (chunkCols + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
	dwOff := quadsPerSubBlk * 64
	mwOff := dwOff + 64

	unpackRange := func(sStart, sEnd int) {
		if isIO {
			// W has shape [K, N]. Unpack row-by-row across the strip's 16 columns.
			reader := makeWeightRowReader(wBuf, N)
			var rowScratch []int8
			if reader.needsScratch() {
				rowScratch = make([]int8, N)
			}
			for s := sStart; s < sEnd; s++ {
				col0 := nStart + s*QuantizedPanelStripCols
				validCols := min(QuantizedPanelStripCols, nStart+chunkCols-col0)
				strip := panel[s*stripBytes : (s+1)*stripBytes]
				clear(strip)

				for b := range numSubBlocks {
					k0 := b * kBlock
					k1 := min(k0+kBlock, K)
					sb := strip[b*subBlkBytes : (b+1)*subBlkBytes]
					dw := unsafe.Slice((*float32)(unsafe.Pointer(&sb[dwOff])), QuantizedPanelStripCols)
					mw := unsafe.Slice((*float32)(unsafe.Pointer(&sb[mwOff])), QuantizedPanelStripCols)

					for c := range validCols {
						n := col0 + c
						sc := wScales[wSLayout.Index(k0, n)]
						var zp float32
						if wZPF32 != nil {
							zp = wZPF32[wZPLayout.Index(k0, n)]
						}
						dw[c] = sc
						mw[c] = -(shiftF32 + zp) * sc
					}

					for k := k0; k < k1; k++ {
						relK := k - k0
						qIdx := relK >> 2
						laneByte := relK & 3
						qBase := qIdx * 64
						if reader.isUint8 {
							wRow := reader.rowUint8(k)
							for c := range validCols {
								sb[qBase+4*c+laneByte] = wRow[col0+c]
							}
						} else {
							wRow := reader.rowInt8(k, rowScratch)
							for c := range validCols {
								sb[qBase+4*c+laneByte] = uint8(int32(wRow[col0+c]) + shiftC)
							}
						}
					}
				}
			}
			return
		}

		// OutputsInput: W has shape [N, K]. Each row n is contiguous along K!
		reader := makeWeightRowReader(wBuf, K)
		var rowScratch []int8
		if reader.needsScratch() {
			rowScratch = make([]int8, K)
		}
		uwTmp := make([]uint8, bPad)

		for s := sStart; s < sEnd; s++ {
			col0 := nStart + s*QuantizedPanelStripCols
			validCols := min(QuantizedPanelStripCols, nStart+chunkCols-col0)
			strip := panel[s*stripBytes : (s+1)*stripBytes]
			if validCols < QuantizedPanelStripCols {
				clear(strip)
			}

			for c := range validCols {
				n := col0 + c
				var wRowU8 []uint8
				var wRowI8 []int8
				if reader.isUint8 {
					wRowU8 = reader.rowUint8(n)
				} else {
					wRowI8 = reader.rowInt8(n, rowScratch)
				}

				for b := range numSubBlocks {
					k0 := b * kBlock
					k1 := min(k0+kBlock, K)
					kLen := k1 - k0
					sb := strip[b*subBlkBytes : (b+1)*subBlkBytes]
					dw := unsafe.Slice((*float32)(unsafe.Pointer(&sb[dwOff])), QuantizedPanelStripCols)
					mw := unsafe.Slice((*float32)(unsafe.Pointer(&sb[mwOff])), QuantizedPanelStripCols)

					sc := wScales[wSLayout.Index(n, k0)]
					var zp float32
					if wZPF32 != nil {
						zp = wZPF32[wZPLayout.Index(n, k0)]
					}
					dw[c] = sc
					mw[c] = -(shiftF32 + zp) * sc

					if reader.isUint8 {
						copy(uwTmp[:kLen], wRowU8[k0:k1])
					} else if shiftC == 128 {
						for i := range kLen {
							uwTmp[i] = uint8(wRowI8[k0+i]) ^ 0x80
						}
					} else {
						for i := range kLen {
							uwTmp[i] = uint8(int32(wRowI8[k0+i]) + shiftC)
						}
					}
					if kLen < bPad {
						clear(uwTmp[kLen:bPad])
					}
					for q := range quadsPerSubBlk {
						v := binary.LittleEndian.Uint32(uwTmp[4*q : 4*q+4])
						binary.LittleEndian.PutUint32(sb[q*64+4*c:q*64+4*c+4], v)
					}
				}
			}
		}
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		chunkCols*K >= quantizedDenseMinParallelizeChunk && numStrips > 1 {
		backend.Scheduler.ParallelizeChunks(numStrips, 1, unpackRange)
	} else {
		unpackRange(0, numStrips)
	}
}

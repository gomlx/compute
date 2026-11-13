// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"encoding/binary"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/shapes"
)

const (
	swarMask03 = uint64(0x0303030303030303)
	swarMask04 = uint64(0x0404040404040404)
	swarMask0F = uint64(0x0F0F0F0F0F0F0F0F)
	swarMask10 = uint64(0x1010101010101010)
	swarMask80 = uint64(0x8080808080808080)
)

var (
	// q5HighBitsLUT maps an 8-bit mask b to an 8-byte uint64 where byte k is 0x10 if bit k of b is set.
	q5HighBitsLUT [256]uint64
	// iq4nlUintTable shifts iq4nlIntTable[-127..113] by +128 into unsigned [1..241].
	iq4nlUintTable [16]uint8
)

func init() {
	for b := range 256 {
		var v uint64
		for k := range 8 {
			if (b>>uint(k))&1 != 0 {
				v |= uint64(0x10) << uint(8*k)
			}
		}
		q5HighBitsLUT[b] = v
	}
	for i, sv := range iq4nlIntTable {
		iq4nlUintTable[i] = uint8(int16(sv) + 128)
	}
}

func ggmlPanelSpec(wType compute.GGMLQuantType) (subBlockSize int, wideUnsigned bool, ok bool) {
	switch wType {
	case compute.GGMLQ4_0, compute.GGMLQ4_1, compute.GGMLQ5_0, compute.GGMLQ5_1,
		compute.GGMLQ4_K, compute.GGMLQ5_K:
		return 32, false, true
	case compute.GGMLQ8_0, compute.GGMLIQ4NL, compute.GGMLQ8_K:
		return 32, true, true
	case compute.GGMLQ2_K, compute.GGMLQ3_K, compute.GGMLQ6_K:
		return 16, false, true
	default:
		return 0, false, false
	}
}

func canCacheGGMLPanel(wType compute.GGMLQuantType, K, N int) bool {
	subBlockSize, _, ok := ggmlPanelSpec(wType)
	return ok && K > 0 && N > 0 && K%subBlockSize == 0
}

// execQuantizedDenseGGMLPanel executes GGML integer matmul (both pre-quantized Mode B1 and dynamic
// float-input Mode B2) via K4-interleaved 16-column panels and SIMD/VNNI microkernels.
func execQuantizedDenseGGMLPanel(
	backend *gobackend.Backend,
	xBuf *gobackend.Buffer,
	actShareGroup *sharedQuantActGroup,
	xBytes []uint8,
	bytesPerRowX int,
	wBuf *gobackend.Buffer,
	wConstantData *gobackend.ConstantData,
	biasF32, outF32 []float32,
	M, K, N int,
	wType compute.GGMLQuantType,
	inTileAct activations.KernelFn[float32],
) (bool, error) {
	if M < quantizedPanelMinM || K == 0 || N == 0 {
		return false, nil
	}
	subBlockSize, wideUnsigned, ok := ggmlPanelSpec(wType)
	if !ok || K%subBlockSize != 0 {
		return false, nil
	}

	numSubBlocks := K / subBlockSize
	quadsPerSubBlk := subBlockSize / 4
	subBlkBytes := QuantizedPanelStripSubBlockBytes(quadsPerSubBlk)
	stripBytes := numSubBlocks * subBlkBytes
	q8KSuperBlocks := wType.ValuesPerBlock() == 256

	var (
		qx         []int8
		dxPairs    []float32
		releaseAct func()
		err        error
	)
	if xBytes != nil {
		qxBuf, qErr := backend.GetBuffer(shapes.Make(dtypes.Int8, M*K))
		if qErr != nil {
			return false, qErr
		}
		dxBuf, dErr := backend.GetBuffer(shapes.Make(dtypes.Float32, M*numSubBlocks*2))
		if dErr != nil {
			backend.PutBuffer(qxBuf)
			return false, dErr
		}
		qx = qxBuf.Flat.([]int8)
		dxPairs = dxBuf.Flat.([]float32)
		unpackGGMLQuantizedActivations(backend, xBytes, bytesPerRowX, qx, dxPairs, M, K, subBlockSize, q8KSuperBlocks)
		releaseAct = func() {
			backend.PutBuffer(dxBuf)
			backend.PutBuffer(qxBuf)
		}
	} else {
		qx, dxPairs, releaseAct, err = actShareGroup.acquire(
			backend, xBuf, M, M*K, M*numSubBlocks*2,
			func(qxDst []int8, dxDst []float32) error {
				xF32, releaseX, convErr := bufferToFloat32Slice(backend, xBuf)
				if convErr != nil {
					return convErr
				}
				defer releaseX()
				quantizeGGMLPanelActivations(backend, xF32, qxDst, dxDst, M, K, subBlockSize, q8KSuperBlocks)
				return nil
			},
		)
		if err != nil {
			return false, err
		}
	}
	defer releaseAct()

	// Fast path: constant / donated GGML weight tensor pre-packed once across all calls and dynamic shapes.
	// Note: wBuf.Flat MUST only be read inside GetOrSetCachedForm's computeFn (under wConstantData.mu),
	// because GetOrSetCachedForm releases wConstantData's rawBuffer once cachedForm is populated.
	if wConstantData != nil {
		totalStrips := (N + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
		cachedPanel := gobackend.GetOrSetCachedForm(wConstantData, func() []uint8 {
			weights := wBuf.Flat.([]uint8)
			bytesPerRowW := wBuf.RawShape.Dimensions[1]
			p := make([]uint8, totalStrips*stripBytes)
			unpackGGMLWeightStrips(
				backend, weights, bytesPerRowW, wType,
				0, N, numSubBlocks, subBlkBytes, p,
			)
			return p
		})
		var chunkBias []float32
		if biasF32 != nil {
			chunkBias = biasF32[:N]
		}
		cfg := QuantizedPanelGEMMConfig{
			Backend:        backend,
			Qx:             qx,
			DxPairs:        dxPairs,
			Panel:          cachedPanel,
			Bias:           chunkBias,
			Out:            outF32,
			M:              M,
			N:              N,
			NStart:         0,
			ChunkCols:      N,
			KPadded:        K,
			NumSubBlocks:   numSubBlocks,
			QuadsPerSubBlk: quadsPerSubBlk,
			WideUnsigned:   wideUnsigned,
			ActFn:          inTileAct,
		}
		dispatchQuantizedPanelGEMM(&cfg)
		return true, nil
	}

	colChunk := selectQuantizedPanelColChunk(N, numSubBlocks, quadsPerSubBlk)
	maxStrips := (colChunk + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols

	panelBuf, err := backend.GetBuffer(shapes.Make(dtypes.Uint8, maxStrips*stripBytes))
	if err != nil {
		return false, err
	}
	defer backend.PutBuffer(panelBuf)
	panel := panelBuf.Flat.([]uint8)

	weights := wBuf.Flat.([]uint8)
	bytesPerRowW := wBuf.RawShape.Dimensions[1]
	for nStart := 0; nStart < N; nStart += colChunk {
		chunkCols := min(colChunk, N-nStart)
		numStrips := (chunkCols + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
		unpackGGMLWeightStrips(
			backend, weights, bytesPerRowW, wType,
			nStart, chunkCols, numSubBlocks, subBlkBytes,
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
			KPadded:        K,
			NumSubBlocks:   numSubBlocks,
			QuadsPerSubBlk: quadsPerSubBlk,
			WideUnsigned:   wideUnsigned,
			ActFn:          inTileAct,
		}
		dispatchQuantizedPanelGEMM(&cfg)
	}
	return true, nil
}

func quantizeGGMLPanelActivations(
	backend *gobackend.Backend,
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	M, K, subBlockSize int,
	q8KSuperBlocks bool,
) {
	fn := quantizeGGMLPanelActivationsScalar
	if archFn := gobackend.GetQuantizeGGMLActivationsArchDispatcher(); archFn != nil {
		fn = archFn
	}
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		M*K >= quantizedDenseMinParallelizeChunk && M > 1 {
		backend.Scheduler.ParallelizeChunks(M, 1, func(mStart, mEnd int) {
			fn(xF32, qx, dxPairs, mStart, mEnd, K, subBlockSize, q8KSuperBlocks)
		})
		return
	}
	fn(xF32, qx, dxPairs, 0, M, K, subBlockSize, q8KSuperBlocks)
}

// QuantizeGGMLPanelActivationsScalar is the portable reference quantizer producing canonical
// (qx, dxPairs) for GGML Q8_0 (32-val blocks, FP16 scale) and Q8_K (256-val blocks, FP32 scale).
func QuantizeGGMLPanelActivationsScalar(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, subBlockSize int,
	q8KSuperBlocks bool,
) {
	quantizeGGMLPanelActivationsScalar(xF32, qx, dxPairs, mStart, mEnd, K, subBlockSize, q8KSuperBlocks)
}

func quantizeGGMLPanelActivationsScalar(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, subBlockSize int,
	q8KSuperBlocks bool,
) {
	numSubBlocks := K / subBlockSize
	if !q8KSuperBlocks {
		// Q8_0 semantics: 32 values per block, scale rounded through FP16.
		for m := mStart; m < mEnd; m++ {
			xRow := xF32[m*K : (m+1)*K]
			qRow := qx[m*K : (m+1)*K]
			dRow := dxPairs[m*numSubBlocks*2 : (m+1)*numSubBlocks*2]
			for b := range numSubBlocks {
				xBlk := xRow[b*32 : (b+1)*32]
				qBlk := qRow[b*32 : (b+1)*32]
				var maxAbs float32
				for _, v := range xBlk {
					av := float32(math.Abs(float64(v)))
					if av > maxAbs {
						maxAbs = av
					}
				}
				if maxAbs == 0 {
					clear(qBlk)
					dRow[2*b] = 0
					dRow[2*b+1] = 0
					continue
				}
				dActual := float16.FromFloat32(maxAbs / 127.0).Float32()
				if dActual == 0 {
					clear(qBlk)
					dRow[2*b] = 0
					dRow[2*b+1] = 0
					continue
				}
				invD := 1.0 / dActual
				var sumQ int32
				for i, v := range xBlk {
					q := int32(math.RoundToEven(float64(v * invD)))
					q = max(-127, min(127, q))
					qBlk[i] = int8(q)
					sumQ += q
				}
				dRow[2*b] = dActual
				dRow[2*b+1] = dActual * float32(sumQ)
			}
		}
		return
	}

	// Q8_K semantics: 256 values per super-block with FP32 scale, partitioned into subBlockSize (16 or 32).
	nSuper := K / 256
	subsPerSuper := 256 / subBlockSize
	for m := mStart; m < mEnd; m++ {
		xRow := xF32[m*K : (m+1)*K]
		qRow := qx[m*K : (m+1)*K]
		dRow := dxPairs[m*numSubBlocks*2 : (m+1)*numSubBlocks*2]
		for sb := range nSuper {
			xSup := xRow[sb*256 : (sb+1)*256]
			qSup := qRow[sb*256 : (sb+1)*256]
			var maxAbs float32
			for _, v := range xSup {
				av := float32(math.Abs(float64(v)))
				if av > maxAbs {
					maxAbs = av
				}
			}
			baseSub := sb * subsPerSuper
			if maxAbs == 0 {
				clear(qSup)
				clear(dRow[2*baseSub : 2*(baseSub+subsPerSuper)])
				continue
			}
			d := maxAbs / 127.0
			invD := 127.0 / maxAbs
			for s := range subsPerSuper {
				off := s * subBlockSize
				var sumQ int32
				for i := range subBlockSize {
					q := int32(math.RoundToEven(float64(xSup[off+i] * invD)))
					q = max(-127, min(127, q))
					qSup[off+i] = int8(q)
					sumQ += q
				}
				dIdx := 2 * (baseSub + s)
				dRow[dIdx] = d
				dRow[dIdx+1] = d * float32(sumQ)
			}
		}
	}
}

func unpackGGMLQuantizedActivations(
	backend *gobackend.Backend,
	xBytes []uint8,
	bytesPerRowX int,
	qx []int8,
	dxPairs []float32,
	M, K, subBlockSize int,
	q8KSuperBlocks bool,
) {
	numSubBlocks := K / subBlockSize
	unpackRows := func(mStart, mEnd int) {
		if !q8KSuperBlocks {
			// Q8_0: 34 bytes per 32 values (fp16 d + int8 qs[32]).
			for m := mStart; m < mEnd; m++ {
				srcRow := xBytes[m*bytesPerRowX : (m+1)*bytesPerRowX]
				qRow := qx[m*K : (m+1)*K]
				dRow := dxPairs[m*numSubBlocks*2 : (m+1)*numSubBlocks*2]
				for b := range numSubBlocks {
					blk := srcRow[b*34 : (b+1)*34]
					d := ggmlFp16LE(blk[0], blk[1])
					qs := blk[2:34]
					var sumQ int32
					for i := range 32 {
						qi := int8(qs[i])
						qRow[b*32+i] = qi
						sumQ += int32(qi)
					}
					dRow[2*b] = d
					dRow[2*b+1] = d * float32(sumQ)
				}
			}
			return
		}

		// Q8_K: 292 bytes per 256 values (fp32 d + int8 qs[256] + int16 bsums[16]).
		nSuper := K / 256
		subsPerSuper := 256 / subBlockSize
		for m := mStart; m < mEnd; m++ {
			srcRow := xBytes[m*bytesPerRowX : (m+1)*bytesPerRowX]
			qRow := qx[m*K : (m+1)*K]
			dRow := dxPairs[m*numSubBlocks*2 : (m+1)*numSubBlocks*2]
			for sb := range nSuper {
				blk := srcRow[sb*292 : (sb+1)*292]
				d := ggmlFp32LE(blk[0:4])
				qs := blk[4:260]
				bsums := blk[260:292]
				qDst := qRow[sb*256 : (sb+1)*256]
				for i := range 256 {
					qDst[i] = int8(qs[i])
				}
				baseSub := sb * subsPerSuper
				if subBlockSize == 16 {
					for s := range 16 {
						bs := ggmlBsum16(bsums, s)
						dIdx := 2 * (baseSub + s)
						dRow[dIdx] = d
						dRow[dIdx+1] = d * float32(bs)
					}
				} else {
					for s := range 8 {
						bs := ggmlBsum16(bsums, 2*s) + ggmlBsum16(bsums, 2*s+1)
						dIdx := 2 * (baseSub + s)
						dRow[dIdx] = d
						dRow[dIdx+1] = d * float32(bs)
					}
				}
			}
		}
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		M*K >= quantizedDenseMinParallelizeChunk && M > 1 {
		backend.Scheduler.ParallelizeChunks(M, 1, unpackRows)
		return
	}
	unpackRows(0, M)
}

func writeStripSubBlock32(sb []uint8, col int, u0, u1, u2, u3 uint64, dw, mw float32) {
	cOff := 4 * col
	binary.LittleEndian.PutUint32(sb[0*64+cOff:0*64+cOff+4], uint32(u0))
	binary.LittleEndian.PutUint32(sb[1*64+cOff:1*64+cOff+4], uint32(u0>>32))
	binary.LittleEndian.PutUint32(sb[2*64+cOff:2*64+cOff+4], uint32(u1))
	binary.LittleEndian.PutUint32(sb[3*64+cOff:3*64+cOff+4], uint32(u1>>32))
	binary.LittleEndian.PutUint32(sb[4*64+cOff:4*64+cOff+4], uint32(u2))
	binary.LittleEndian.PutUint32(sb[5*64+cOff:5*64+cOff+4], uint32(u2>>32))
	binary.LittleEndian.PutUint32(sb[6*64+cOff:6*64+cOff+4], uint32(u3))
	binary.LittleEndian.PutUint32(sb[7*64+cOff:7*64+cOff+4], uint32(u3>>32))
	binary.LittleEndian.PutUint32(sb[512+cOff:512+cOff+4], math.Float32bits(dw))
	binary.LittleEndian.PutUint32(sb[576+cOff:576+cOff+4], math.Float32bits(mw))
}

func writeStripSubBlock16(sb []uint8, col int, u0, u1 uint64, dw, mw float32) {
	cOff := 4 * col
	binary.LittleEndian.PutUint32(sb[0*64+cOff:0*64+cOff+4], uint32(u0))
	binary.LittleEndian.PutUint32(sb[1*64+cOff:1*64+cOff+4], uint32(u0>>32))
	binary.LittleEndian.PutUint32(sb[2*64+cOff:2*64+cOff+4], uint32(u1))
	binary.LittleEndian.PutUint32(sb[3*64+cOff:3*64+cOff+4], uint32(u1>>32))
	binary.LittleEndian.PutUint32(sb[256+cOff:256+cOff+4], math.Float32bits(dw))
	binary.LittleEndian.PutUint32(sb[320+cOff:320+cOff+4], math.Float32bits(mw))
}

func unpackGGMLWeightStrips(
	backend *gobackend.Backend,
	weights []uint8,
	bytesPerRowW int,
	wType compute.GGMLQuantType,
	nStart, chunkCols, numSubBlocks, subBlkBytes int,
	panel []uint8,
) {
	stripBytes := numSubBlocks * subBlkBytes
	numStrips := (chunkCols + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols

	unpackRange := func(sStart, sEnd int) {
		for s := sStart; s < sEnd; s++ {
			col0 := nStart + s*QuantizedPanelStripCols
			validCols := min(QuantizedPanelStripCols, nStart+chunkCols-col0)
			strip := panel[s*stripBytes : (s+1)*stripBytes]
			if validCols < QuantizedPanelStripCols {
				clear(strip)
			}
			for c := range validCols {
				n := col0 + c
				wRow := weights[n*bytesPerRowW : (n+1)*bytesPerRowW]
				unpackGGMLRowToStripCol(wRow, wType, strip, c)
			}
		}
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		chunkCols*bytesPerRowW >= 2048 && numStrips > 1 {
		backend.Scheduler.ParallelizeChunks(numStrips, 1, unpackRange)
	} else {
		unpackRange(0, numStrips)
	}
}

func unpackGGMLRowToStripCol(wRow []uint8, wType compute.GGMLQuantType, strip []uint8, col int) {
	switch wType {
	case compute.GGMLQ8_0:
		const wBytes = 34
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			u0 := binary.LittleEndian.Uint64(wb[2:10]) ^ swarMask80
			u1 := binary.LittleEndian.Uint64(wb[10:18]) ^ swarMask80
			u2 := binary.LittleEndian.Uint64(wb[18:26]) ^ swarMask80
			u3 := binary.LittleEndian.Uint64(wb[26:34]) ^ swarMask80
			writeStripSubBlock32(strip[b*640:(b+1)*640], col, u0, u1, u2, u3, dW, -128.0*dW)
		}

	case compute.GGMLQ4_0:
		const wBytes = 18
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			lo0 := binary.LittleEndian.Uint64(wb[2:10])
			lo1 := binary.LittleEndian.Uint64(wb[10:18])
			writeStripSubBlock32(
				strip[b*640:(b+1)*640], col,
				lo0&swarMask0F, lo1&swarMask0F,
				(lo0>>4)&swarMask0F, (lo1>>4)&swarMask0F,
				dW, -8.0*dW,
			)
		}

	case compute.GGMLQ4_1:
		const wBytes = 20
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			mW := ggmlFp16LE(wb[2], wb[3])
			lo0 := binary.LittleEndian.Uint64(wb[4:12])
			lo1 := binary.LittleEndian.Uint64(wb[12:20])
			writeStripSubBlock32(
				strip[b*640:(b+1)*640], col,
				lo0&swarMask0F, lo1&swarMask0F,
				(lo0>>4)&swarMask0F, (lo1>>4)&swarMask0F,
				dW, mW,
			)
		}

	case compute.GGMLQ5_0:
		const wBytes = 22
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			lo0 := binary.LittleEndian.Uint64(wb[6:14])
			lo1 := binary.LittleEndian.Uint64(wb[14:22])
			u0 := (lo0 & swarMask0F) | q5HighBitsLUT[wb[2]]
			u1 := (lo1 & swarMask0F) | q5HighBitsLUT[wb[3]]
			u2 := ((lo0 >> 4) & swarMask0F) | q5HighBitsLUT[wb[4]]
			u3 := ((lo1 >> 4) & swarMask0F) | q5HighBitsLUT[wb[5]]
			writeStripSubBlock32(strip[b*640:(b+1)*640], col, u0, u1, u2, u3, dW, -16.0*dW)
		}

	case compute.GGMLQ5_1:
		const wBytes = 24
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			mW := ggmlFp16LE(wb[2], wb[3])
			lo0 := binary.LittleEndian.Uint64(wb[8:16])
			lo1 := binary.LittleEndian.Uint64(wb[16:24])
			u0 := (lo0 & swarMask0F) | q5HighBitsLUT[wb[4]]
			u1 := (lo1 & swarMask0F) | q5HighBitsLUT[wb[5]]
			u2 := ((lo0 >> 4) & swarMask0F) | q5HighBitsLUT[wb[6]]
			u3 := ((lo1 >> 4) & swarMask0F) | q5HighBitsLUT[wb[7]]
			writeStripSubBlock32(strip[b*640:(b+1)*640], col, u0, u1, u2, u3, dW, mW)
		}

	case compute.GGMLIQ4NL:
		const wBytes = 18
		nblocks := len(wRow) / wBytes
		var tmp [32]uint8
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			qw := wb[2:18]
			for i := range 16 {
				tmp[i] = iq4nlUintTable[qw[i]&0x0F]
				tmp[16+i] = iq4nlUintTable[qw[i]>>4]
			}
			u0 := binary.LittleEndian.Uint64(tmp[0:8])
			u1 := binary.LittleEndian.Uint64(tmp[8:16])
			u2 := binary.LittleEndian.Uint64(tmp[16:24])
			u3 := binary.LittleEndian.Uint64(tmp[24:32])
			writeStripSubBlock32(strip[b*640:(b+1)*640], col, u0, u1, u2, u3, dW, -128.0*dW)
		}

	case compute.GGMLQ2_K:
		const wBytes = 84
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			scales := wb[0:16]
			qs := wb[16:80]
			dW := ggmlFp16LE(wb[80], wb[81])
			dminW := ggmlFp16LE(wb[82], wb[83])
			subBase := b * 16
			is := 0
			for half := range 2 {
				qBase := half * 32
				q0 := binary.LittleEndian.Uint64(qs[qBase : qBase+8])
				q1 := binary.LittleEndian.Uint64(qs[qBase+8 : qBase+16])
				q2 := binary.LittleEndian.Uint64(qs[qBase+16 : qBase+24])
				q3 := binary.LittleEndian.Uint64(qs[qBase+24 : qBase+32])
				for s := range 4 {
					shift := uint(2 * s)
					sc0 := scales[is]
					sb0 := strip[(subBase+is)*384 : (subBase+is+1)*384]
					writeStripSubBlock16(
						sb0, col,
						(q0>>shift)&swarMask03, (q1>>shift)&swarMask03,
						dW*float32(sc0&0xF), -dminW*float32(sc0>>4),
					)
					is++

					sc1 := scales[is]
					sb1 := strip[(subBase+is)*384 : (subBase+is+1)*384]
					writeStripSubBlock16(
						sb1, col,
						(q2>>shift)&swarMask03, (q3>>shift)&swarMask03,
						dW*float32(sc1&0xF), -dminW*float32(sc1>>4),
					)
					is++
				}
			}
		}

	case compute.GGMLQ3_K:
		const wBytes = 110
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			hm0 := binary.LittleEndian.Uint64(wb[0:8])
			hm1 := binary.LittleEndian.Uint64(wb[8:16])
			hm2 := binary.LittleEndian.Uint64(wb[16:24])
			hm3 := binary.LittleEndian.Uint64(wb[24:32])
			qs := wb[32:96]
			scales := unpackQ3KScales(wb[96:108])
			dAll := ggmlFp16LE(wb[108], wb[109])

			subBase := b * 16
			is := 0
			bitIdx := 0
			for half := range 2 {
				qBase := half * 32
				q0 := binary.LittleEndian.Uint64(qs[qBase : qBase+8])
				q1 := binary.LittleEndian.Uint64(qs[qBase+8 : qBase+16])
				q2 := binary.LittleEndian.Uint64(qs[qBase+16 : qBase+24])
				q3 := binary.LittleEndian.Uint64(qs[qBase+24 : qBase+32])
				for s := range 4 {
					shift := uint(2 * s)
					var h0, h1, h2, h3 uint64
					if bitIdx <= 2 {
						sh := uint(2 - bitIdx)
						h0 = (hm0 << sh) & swarMask04
						h1 = (hm1 << sh) & swarMask04
						h2 = (hm2 << sh) & swarMask04
						h3 = (hm3 << sh) & swarMask04
					} else {
						sh := uint(bitIdx - 2)
						h0 = (hm0 >> sh) & swarMask04
						h1 = (hm1 >> sh) & swarMask04
						h2 = (hm2 >> sh) & swarMask04
						h3 = (hm3 >> sh) & swarMask04
					}

					dw0 := dAll * float32(int32(scales[is])-32)
					sb0 := strip[(subBase+is)*384 : (subBase+is+1)*384]
					writeStripSubBlock16(sb0, col, ((q0>>shift)&swarMask03)|h0, ((q1>>shift)&swarMask03)|h1, dw0, -4.0*dw0)
					is++

					dw1 := dAll * float32(int32(scales[is])-32)
					sb1 := strip[(subBase+is)*384 : (subBase+is+1)*384]
					writeStripSubBlock16(sb1, col, ((q2>>shift)&swarMask03)|h2, ((q3>>shift)&swarMask03)|h3, dw1, -4.0*dw1)
					is++

					bitIdx++
				}
			}
		}

	case compute.GGMLQ4_K:
		const wBytes = 144
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			dminW := ggmlFp16LE(wb[2], wb[3])
			scales := wb[4:16]
			qs := wb[16:144]
			subBase := b * 8
			for chunk := range 4 {
				is := chunk * 2
				sc1, m1 := getScaleMinK4(is, scales)
				sc2, m2 := getScaleMinK4(is+1, scales)
				qOff := chunk * 32
				q0 := binary.LittleEndian.Uint64(qs[qOff : qOff+8])
				q1 := binary.LittleEndian.Uint64(qs[qOff+8 : qOff+16])
				q2 := binary.LittleEndian.Uint64(qs[qOff+16 : qOff+24])
				q3 := binary.LittleEndian.Uint64(qs[qOff+24 : qOff+32])

				s0 := subBase + is
				writeStripSubBlock32(
					strip[s0*640:(s0+1)*640], col,
					q0&swarMask0F, q1&swarMask0F, q2&swarMask0F, q3&swarMask0F,
					dW*float32(sc1), -dminW*float32(m1),
				)
				s1 := s0 + 1
				writeStripSubBlock32(
					strip[s1*640:(s1+1)*640], col,
					(q0>>4)&swarMask0F, (q1>>4)&swarMask0F, (q2>>4)&swarMask0F, (q3>>4)&swarMask0F,
					dW*float32(sc2), -dminW*float32(m2),
				)
			}
		}

	case compute.GGMLQ5_K:
		const wBytes = 176
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp16LE(wb[0], wb[1])
			dminW := ggmlFp16LE(wb[2], wb[3])
			scales := wb[4:16]
			qh0 := binary.LittleEndian.Uint64(wb[16:24])
			qh1 := binary.LittleEndian.Uint64(wb[24:32])
			qh2 := binary.LittleEndian.Uint64(wb[32:40])
			qh3 := binary.LittleEndian.Uint64(wb[40:48])
			qs := wb[48:176]
			subBase := b * 8
			for chunk := range 4 {
				is := chunk * 2
				sc1, m1 := getScaleMinK4(is, scales)
				sc2, m2 := getScaleMinK4(is+1, scales)
				qOff := chunk * 32
				q0 := binary.LittleEndian.Uint64(qs[qOff : qOff+8])
				q1 := binary.LittleEndian.Uint64(qs[qOff+8 : qOff+16])
				q2 := binary.LittleEndian.Uint64(qs[qOff+16 : qOff+24])
				q3 := binary.LittleEndian.Uint64(qs[qOff+24 : qOff+32])

				bit0 := 2 * chunk
				bit1 := bit0 + 1
				var hb00, hb01, hb02, hb03 uint64
				if bit0 <= 4 {
					sh := uint(4 - bit0)
					hb00, hb01, hb02, hb03 = (qh0<<sh)&swarMask10, (qh1<<sh)&swarMask10, (qh2<<sh)&swarMask10, (qh3<<sh)&swarMask10
				} else {
					sh := uint(bit0 - 4)
					hb00, hb01, hb02, hb03 = (qh0>>sh)&swarMask10, (qh1>>sh)&swarMask10, (qh2>>sh)&swarMask10, (qh3>>sh)&swarMask10
				}
				var hb10, hb11, hb12, hb13 uint64
				if bit1 <= 4 {
					sh := uint(4 - bit1)
					hb10, hb11, hb12, hb13 = (qh0<<sh)&swarMask10, (qh1<<sh)&swarMask10, (qh2<<sh)&swarMask10, (qh3<<sh)&swarMask10
				} else {
					sh := uint(bit1 - 4)
					hb10, hb11, hb12, hb13 = (qh0>>sh)&swarMask10, (qh1>>sh)&swarMask10, (qh2>>sh)&swarMask10, (qh3>>sh)&swarMask10
				}

				s0 := subBase + is
				writeStripSubBlock32(
					strip[s0*640:(s0+1)*640], col,
					(q0&swarMask0F)|hb00, (q1&swarMask0F)|hb01, (q2&swarMask0F)|hb02, (q3&swarMask0F)|hb03,
					dW*float32(sc1), -dminW*float32(m1),
				)
				s1 := s0 + 1
				writeStripSubBlock32(
					strip[s1*640:(s1+1)*640], col,
					((q0>>4)&swarMask0F)|hb10, ((q1>>4)&swarMask0F)|hb11, ((q2>>4)&swarMask0F)|hb12, ((q3>>4)&swarMask0F)|hb13,
					dW*float32(sc2), -dminW*float32(m2),
				)
			}
		}

	case compute.GGMLQ6_K:
		const wBytes = 210
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			ql := wb[0:128]
			qh := wb[128:192]
			sc := wb[192:208]
			dW := ggmlFp16LE(wb[208], wb[209])
			subBase := b * 16
			for half := range 2 {
				qlOff := half * 64
				qhOff := half * 32
				scOff := half * 8
				sBase := subBase + half*8

				ql0 := binary.LittleEndian.Uint64(ql[qlOff : qlOff+8])
				ql1 := binary.LittleEndian.Uint64(ql[qlOff+8 : qlOff+16])
				ql2 := binary.LittleEndian.Uint64(ql[qlOff+16 : qlOff+24])
				ql3 := binary.LittleEndian.Uint64(ql[qlOff+24 : qlOff+32])
				ql4 := binary.LittleEndian.Uint64(ql[qlOff+32 : qlOff+40])
				ql5 := binary.LittleEndian.Uint64(ql[qlOff+40 : qlOff+48])
				ql6 := binary.LittleEndian.Uint64(ql[qlOff+48 : qlOff+56])
				ql7 := binary.LittleEndian.Uint64(ql[qlOff+56 : qlOff+64])

				qh0 := binary.LittleEndian.Uint64(qh[qhOff : qhOff+8])
				qh1 := binary.LittleEndian.Uint64(qh[qhOff+8 : qhOff+16])
				qh2 := binary.LittleEndian.Uint64(qh[qhOff+16 : qhOff+24])
				qh3 := binary.LittleEndian.Uint64(qh[qhOff+24 : qhOff+32])

				dw0 := dW * float32(int8(sc[scOff+0]))
				writeStripSubBlock16(strip[(sBase+0)*384:(sBase+1)*384], col,
					(ql0&swarMask0F)|((qh0&swarMask03)<<4),
					(ql1&swarMask0F)|((qh1&swarMask03)<<4),
					dw0, -32.0*dw0)

				dw1 := dW * float32(int8(sc[scOff+1]))
				writeStripSubBlock16(strip[(sBase+1)*384:(sBase+2)*384], col,
					(ql2&swarMask0F)|((qh2&swarMask03)<<4),
					(ql3&swarMask0F)|((qh3&swarMask03)<<4),
					dw1, -32.0*dw1)

				dw2 := dW * float32(int8(sc[scOff+2]))
				writeStripSubBlock16(strip[(sBase+2)*384:(sBase+3)*384], col,
					(ql4&swarMask0F)|(((qh0>>2)&swarMask03)<<4),
					(ql5&swarMask0F)|(((qh1>>2)&swarMask03)<<4),
					dw2, -32.0*dw2)

				dw3 := dW * float32(int8(sc[scOff+3]))
				writeStripSubBlock16(strip[(sBase+3)*384:(sBase+4)*384], col,
					(ql6&swarMask0F)|(((qh2>>2)&swarMask03)<<4),
					(ql7&swarMask0F)|(((qh3>>2)&swarMask03)<<4),
					dw3, -32.0*dw3)

				dw4 := dW * float32(int8(sc[scOff+4]))
				writeStripSubBlock16(strip[(sBase+4)*384:(sBase+5)*384], col,
					((ql0>>4)&swarMask0F)|(((qh0>>4)&swarMask03)<<4),
					((ql1>>4)&swarMask0F)|(((qh1>>4)&swarMask03)<<4),
					dw4, -32.0*dw4)

				dw5 := dW * float32(int8(sc[scOff+5]))
				writeStripSubBlock16(strip[(sBase+5)*384:(sBase+6)*384], col,
					((ql2>>4)&swarMask0F)|(((qh2>>4)&swarMask03)<<4),
					((ql3>>4)&swarMask0F)|(((qh3>>4)&swarMask03)<<4),
					dw5, -32.0*dw5)

				dw6 := dW * float32(int8(sc[scOff+6]))
				writeStripSubBlock16(strip[(sBase+6)*384:(sBase+7)*384], col,
					((ql4>>4)&swarMask0F)|(((qh0>>6)&swarMask03)<<4),
					((ql5>>4)&swarMask0F)|(((qh1>>6)&swarMask03)<<4),
					dw6, -32.0*dw6)

				dw7 := dW * float32(int8(sc[scOff+7]))
				writeStripSubBlock16(strip[(sBase+7)*384:(sBase+8)*384], col,
					((ql6>>4)&swarMask0F)|(((qh2>>6)&swarMask03)<<4),
					((ql7>>4)&swarMask0F)|(((qh3>>6)&swarMask03)<<4),
					dw7, -32.0*dw7)
			}
		}

	case compute.GGMLQ8_K:
		const wBytes = 292
		nblocks := len(wRow) / wBytes
		for b := range nblocks {
			wb := wRow[b*wBytes : (b+1)*wBytes]
			dW := ggmlFp32LE(wb[0:4])
			qw := wb[4:260]
			subBase := b * 8
			for s := range 8 {
				qOff := s * 32
				u0 := binary.LittleEndian.Uint64(qw[qOff:qOff+8]) ^ swarMask80
				u1 := binary.LittleEndian.Uint64(qw[qOff+8:qOff+16]) ^ swarMask80
				u2 := binary.LittleEndian.Uint64(qw[qOff+16:qOff+24]) ^ swarMask80
				u3 := binary.LittleEndian.Uint64(qw[qOff+24:qOff+32]) ^ swarMask80
				sb := strip[(subBase+s)*640 : (subBase+s+1)*640]
				writeStripSubBlock32(sb, col, u0, u1, u2, u3, dW, -128.0*dW)
			}
		}
	}
}

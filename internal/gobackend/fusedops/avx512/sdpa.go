// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX512Allowed {
		gobackend.SetSDPAArchDispatcher(gobackend.PriorityArch+1, DispatchSDPAAVX512)
	}
}

//go:noescape
func sdpaQKT4RowsAVX512Asm(
	qRow0, qRow1, qRow2, qRow3 unsafe.Pointer,
	kPtr unsafe.Pointer,
	kvByteStride, blockKVLen, headDim int,
	sRow0, sRow1, sRow2, sRow3 unsafe.Pointer,
)

//go:noescape
func sdpaQKT1RowAVX512Asm(
	qRow0, kPtr unsafe.Pointer,
	kvByteStride, blockKVLen, headDim int,
	sRow0 unsafe.Pointer,
)

//go:noescape
func sdpaScaleMaskSoftmaxRowAVX512Asm(
	scoresPtr, addMaskPtr, addBiasPtr, boolMaskPtr unsafe.Pointer,
	rowKVLen int,
	scale float32,
) int

//go:noescape
func sdpaValueAccum4RowsAVX512Asm(
	sRow0, sRow1, sRow2, sRow3 unsafe.Pointer,
	vPtr unsafe.Pointer,
	blockKVLen, kvByteStride, headDim int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func sdpaValueAccum1RowAVX512Asm(
	sRow0, vPtr unsafe.Pointer,
	blockKVLen, kvByteStride, headDim int,
	outRow0 unsafe.Pointer,
)

//go:noescape
func sdpaQueriesDim32NoMaskAVX512Asm(
	qPtr, kPtr, vPtr, outPtr, scoresPtr unsafe.Pointer,
	qByteStride, kvByteStride, numQueries, kvLen, qStart, causalInt int,
	scale float32,
)

func sdpaMaskBiasSeqStrides(additiveMask []float32, booleanMask []bool, maskGroupStride int, additiveBias []float32, biasGroupStride, kvLen int) (maskSeqStride, biasSeqStride int) {
	maskSeqStride = kvLen
	if maskLen := max(len(additiveMask), len(booleanMask)); maskLen == 0 || maskGroupStride == kvLen || (maskGroupStride == 0 && maskLen == kvLen) {
		maskSeqStride = 0
	}
	biasSeqStride = kvLen
	if len(additiveBias) == 0 || biasGroupStride == kvLen || (biasGroupStride == 0 && len(additiveBias) == kvLen) {
		biasSeqStride = 0
	}
	return maskSeqStride, biasSeqStride
}

// sdpaActiveKVLenForBlock returns the highest unmasked key index + 1 across activeRows query rows
// starting at qIdx, accounting for both causal masking and booleanMask.
func sdpaActiveKVLenForBlock(
	qIdx, activeRows, kvLenUnmaskedBase int,
	causal bool,
	booleanMask []bool,
	maskIdxBase, maskSeqStride int,
) int {
	limit := kvLenUnmaskedBase
	if causal && qIdx+activeRows < limit {
		limit = qIdx + activeRows
	}
	if limit <= 0 {
		return 0
	}
	if len(booleanMask) == 0 {
		return limit
	}
	if maskSeqStride == 0 {
		for ki := limit - 1; ki >= 0; ki-- {
			if booleanMask[maskIdxBase+ki] {
				return ki + 1
			}
		}
		return 0
	}
	maxActive := 0
	for r := activeRows - 1; r >= 0; r-- {
		rowLim := limit
		if causal && qIdx+r+1 < rowLim {
			rowLim = qIdx + r + 1
		}
		mStart := maskIdxBase + r*maskSeqStride
		for ki := rowLim - 1; ki >= maxActive; ki-- {
			if booleanMask[mStart+ki] {
				maxActive = ki + 1
				break
			}
		}
	}
	return maxActive
}

// isAllTrueBoolMask checks if all elements of b are true using 8-byte uint64 loads.
func isAllTrueBoolMask(b []bool) bool {
	n := len(b)
	i := 0
	if n >= 8 {
		ptr := unsafe.Pointer(&b[0])
		for ; i+8 <= n; i += 8 {
			if *(*uint64)(unsafe.Add(ptr, i)) != 0x0101010101010101 {
				return false
			}
		}
	}
	for ; i < n; i++ {
		if !b[i] {
			return false
		}
	}
	return true
}

// DispatchSDPAAVX512 executes scaled dot-product attention using handwritten AVX-512 assembly microkernels:
//   - 4-query-row batched Q @ K^T (sdpaQKT4RowsAVX512Asm), fused Scale + Mask/Bias + Softmax
//     (sdpaScaleMaskSoftmaxRowAVX512Asm), and 4-query-row batched P @ V (sdpaValueAccum4RowsAVX512Asm)
//   - 1-query-row assembly fast path (sdpaQKT1RowAVX512Asm + sdpaValueAccum1RowAVX512Asm) for single-row
//     decoding or remainder query rows.
func DispatchSDPAAVX512(
	q, k, v, output []float32,
	qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskGroupStride int,
	additiveBias []float32,
	biasGroupStride int,
	scoresScratch []float32,
	groupSize, qStart, qEnd, kvLen, headDim int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) bool {
	if !gobackend.IsAVX512Allowed || headDim <= 0 {
		return false
	}

	kvLenUnmaskedBase := min(kvLen, kvLimit)
	qValidEnd := min(qEnd, qLimit)
	qValidCount := max(0, qValidEnd-qStart)

	// Fast fused 1-row-at-a-time kernel for tiny query counts (1..2) when headDim == 32 and unmasked.
	if headDim == 32 && qValidCount <= 2 && len(booleanMask) == 0 && len(additiveMask) == 0 && len(additiveBias) == 0 {
		if len(scoresScratch) < kvLen+16 {
			scoresScratch = make([]float32, kvLen+16)
		}
		causalInt := 0
		if causal {
			causalInt = 1
		}
		for gIdx := range groupSize {
			gQOff := qOff + gIdx*qGroupStride
			activeQEnd := qValidEnd
			if activeQEnd > qStart && kvLenUnmaskedBase > 0 {
				numQ := activeQEnd - qStart
				qBase := gQOff + qStart*qSeqStride
				sdpaQueriesDim32NoMaskAVX512Asm(
					unsafe.Pointer(&q[qBase]),
					unsafe.Pointer(&k[kvOff]),
					unsafe.Pointer(&v[kvOff]),
					unsafe.Pointer(&output[qBase]),
					unsafe.Pointer(&scoresScratch[0]),
					qSeqStride*4,
					kvSeqStride*4,
					numQ,
					kvLenUnmaskedBase,
					qStart,
					causalInt,
					scale,
				)
			} else {
				activeQEnd = qStart
			}
			for qIdx := max(qStart, activeQEnd); qIdx < qEnd; qIdx++ {
				outBase := gQOff + qIdx*qSeqStride
				clear(output[outBase : outBase+32])
			}
		}
		return true
	}

	// Each row in scoresScratch needs at least kvLen + 16 floats so AVX-512 softmax can pad 16 -Inf lanes
	// at scores[rowKVLen : rowKVLen+16] without overwriting the next row in the 4-row block.
	scoresStride := (kvLen + 31) &^ 15
	needScratch := 4 * scoresStride
	if qValidCount <= 1 {
		needScratch = scoresStride
	}
	if len(scoresScratch) < needScratch {
		scoresScratch = make([]float32, needScratch)
	}

	maskSeqStride, biasSeqStride := sdpaMaskBiasSeqStrides(additiveMask, booleanMask, maskGroupStride, additiveBias, biasGroupStride, kvLen)

	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		qIdx := qStart
		if kvLenUnmaskedBase > 0 {
			for qIdx < qValidEnd {
				activeRows := min(4, qValidEnd-qIdx)
				sdpaAVX512BlockFloat32(
					q, k, v, output,
					qIdx, activeRows,
					gQOff+qIdx*qSeqStride, kvOff, qSeqStride, kvSeqStride,
					additiveMask, booleanMask, gMaskOff+qIdx*maskSeqStride, maskSeqStride,
					additiveBias, gBiasOff+qIdx*biasSeqStride, biasSeqStride,
					scoresScratch, scoresStride,
					kvLenUnmaskedBase, headDim, scale, causal,
				)
				qIdx += activeRows
			}
		}

		for ; qIdx < qEnd; qIdx++ {
			outBase := gQOff + qIdx*qSeqStride
			clear(output[outBase : outBase+headDim])
		}
	}
	return true
}

func sdpaAVX512BlockFloat32(
	q, k, v, output []float32,
	qIdx, activeRows int,
	qBase, kvOff, qSeqStride, kvSeqStride int,
	additiveMask []float32, booleanMask []bool, maskIdxBase, maskSeqStride int,
	additiveBias []float32, biasIdxBase, biasSeqStride int,
	scoresScratch []float32, scoresStride int,
	kvLenUnmaskedBase, headDim int,
	scale float32, causal bool,
) {
	blockKVLen := sdpaActiveKVLenForBlock(qIdx, activeRows, kvLenUnmaskedBase, causal, booleanMask, maskIdxBase, maskSeqStride)
	if blockKVLen <= 0 {
		for r := range activeRows {
			clear(output[qBase+r*qSeqStride : qBase+r*qSeqStride+headDim])
		}
		return
	}

	kvByteStride := kvSeqStride * 4
	kPtr := unsafe.Pointer(&k[kvOff])
	vPtr := unsafe.Pointer(&v[kvOff])

	// Fast 1-query-row path (avoids computing 3 dummy rows when activeRows == 1):
	if activeRows == 1 {
		qRow0 := unsafe.Pointer(&q[qBase])
		sRow0 := unsafe.Pointer(&scoresScratch[0])
		sdpaQKT1RowAVX512Asm(qRow0, kPtr, kvByteStride, blockKVLen, headDim, sRow0)

		var addMaskPtr, addBiasPtr, boolMaskPtr unsafe.Pointer
		if len(additiveMask) > 0 {
			addMaskPtr = unsafe.Pointer(&additiveMask[maskIdxBase])
		}
		if len(additiveBias) > 0 {
			addBiasPtr = unsafe.Pointer(&additiveBias[biasIdxBase])
		}
		if len(booleanMask) > 0 {
			boolMaskRow := booleanMask[maskIdxBase : maskIdxBase+blockKVLen]
			if !isAllTrueBoolMask(boolMaskRow) {
				boolMaskPtr = unsafe.Pointer(&boolMaskRow[0])
			}
		}

		if sdpaScaleMaskSoftmaxRowAVX512Asm(sRow0, addMaskPtr, addBiasPtr, boolMaskPtr, blockKVLen, scale) == 0 {
			clear(output[qBase : qBase+headDim])
			return
		}
		outRow0 := unsafe.Pointer(&output[qBase])
		sdpaValueAccum1RowAVX512Asm(sRow0, vPtr, blockKVLen, kvByteStride, headDim, outRow0)
		return
	}

	// 1. Stage 1: Q[activeRows, headDim] @ K[blockKVLen, headDim]^T -> scoresScratch[activeRows, scoresStride]
	qRow0 := unsafe.Pointer(&q[qBase+0*qSeqStride])
	qRow1 := qRow0
	qRow2 := qRow0
	qRow3 := qRow0
	if activeRows >= 2 {
		qRow1 = unsafe.Pointer(&q[qBase+1*qSeqStride])
	}
	if activeRows >= 3 {
		qRow2 = unsafe.Pointer(&q[qBase+2*qSeqStride])
	}
	if activeRows >= 4 {
		qRow3 = unsafe.Pointer(&q[qBase+3*qSeqStride])
	}

	sRow0 := unsafe.Pointer(&scoresScratch[0*scoresStride])
	sRow1 := unsafe.Pointer(&scoresScratch[1*scoresStride])
	sRow2 := unsafe.Pointer(&scoresScratch[2*scoresStride])
	sRow3 := unsafe.Pointer(&scoresScratch[3*scoresStride])

	sdpaQKT4RowsAVX512Asm(
		qRow0, qRow1, qRow2, qRow3,
		kPtr,
		kvByteStride, blockKVLen, headDim,
		sRow0, sRow1, sRow2, sRow3,
	)

	// 2. Stage 2: Fused Scale + Mask/Bias + Softmax on each of the activeRows score rows
	anyActiveRow := false
	for r := range activeRows {
		rowKVLen := blockKVLen
		if causal && qIdx+r+1 < rowKVLen {
			rowKVLen = qIdx + r + 1
		}
		var boolMaskPtr unsafe.Pointer
		if len(booleanMask) > 0 && rowKVLen > 0 {
			mStart := maskIdxBase + r*maskSeqStride
			boolMaskRow := booleanMask[mStart : mStart+rowKVLen]
			for rowKVLen > 0 && !boolMaskRow[rowKVLen-1] {
				rowKVLen--
			}
			if rowKVLen > 0 {
				boolMaskRow = boolMaskRow[:rowKVLen]
				if !isAllTrueBoolMask(boolMaskRow) {
					boolMaskPtr = unsafe.Pointer(&boolMaskRow[0])
				}
			}
		}
		rowOff := r * scoresStride
		if rowKVLen <= 0 {
			clear(scoresScratch[rowOff : rowOff+blockKVLen])
			continue
		}
		var addMaskPtr, addBiasPtr unsafe.Pointer
		if len(additiveMask) > 0 {
			addMaskPtr = unsafe.Pointer(&additiveMask[maskIdxBase+r*maskSeqStride])
		}
		if len(additiveBias) > 0 {
			addBiasPtr = unsafe.Pointer(&additiveBias[biasIdxBase+r*biasSeqStride])
		}
		if sdpaScaleMaskSoftmaxRowAVX512Asm(
			unsafe.Pointer(&scoresScratch[rowOff]),
			addMaskPtr, addBiasPtr, boolMaskPtr,
			rowKVLen, scale,
		) != 0 {
			anyActiveRow = true
		}
		if rowKVLen < blockKVLen {
			clear(scoresScratch[rowOff+rowKVLen : rowOff+blockKVLen])
		}
	}

	if !anyActiveRow {
		for r := range activeRows {
			clear(output[qBase+r*qSeqStride : qBase+r*qSeqStride+headDim])
		}
		return
	}

	// 3. Stage 3: Probs[activeRows, blockKVLen] @ V[blockKVLen, headDim] -> output[activeRows, headDim]
	outRow0 := unsafe.Pointer(&output[qBase+0*qSeqStride])
	outRow1 := unsafe.Pointer(&output[qBase+1*qSeqStride])
	if activeRows == 4 {
		outRow2 := unsafe.Pointer(&output[qBase+2*qSeqStride])
		outRow3 := unsafe.Pointer(&output[qBase+3*qSeqStride])
		sdpaValueAccum4RowsAVX512Asm(
			sRow0, sRow1, sRow2, sRow3,
			vPtr,
			blockKVLen, kvByteStride, headDim,
			outRow0, outRow1, outRow2, outRow3,
		)
		return
	}

	// Partial block (activeRows == 2 or 3): route unused output rows to a dummy buffer.
	var dummyStack [256]float32
	var dummySlice []float32
	if headDim <= len(dummyStack) {
		dummySlice = dummyStack[:headDim]
	} else {
		dummySlice = make([]float32, headDim)
	}
	dummyPtr := unsafe.Pointer(&dummySlice[0])
	outRow2 := dummyPtr
	outRow3 := dummyPtr
	if activeRows >= 3 {
		outRow2 = unsafe.Pointer(&output[qBase+2*qSeqStride])
	}
	sdpaValueAccum4RowsAVX512Asm(
		sRow0, sRow1, sRow2, sRow3,
		vPtr,
		blockKVLen, kvByteStride, headDim,
		outRow0, outRow1, outRow2, outRow3,
	)
}

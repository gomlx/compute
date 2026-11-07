// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"math"
	"runtime"
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsNEONAllowed {
		gobackend.SetSDPAArchDispatcher(gobackend.PriorityArch, DispatchSDPAARM64)
	}
}

// DispatchSDPAARM64 executes scaled dot-product attention on ARM64 using:
//  1. 32x32 SME outer-product (FMOPA) microkernels (smeLargeKernelFloat32Asm_32x32) when SME is
//     available, the worker acquires a cluster SME lock, and the unmasked block is >= 32x32.
//  2. 4-query-row handwritten ARM64 NEON assembly microkernels (neonSmallTransposedTile4x4Float32Asm
//     for Q @ K^T and neonSmallNonTransposedTile4x16Float32Asm for Probs @ V) with vectorized NEON
//     Softmax for 4 query rows at a time.
//  3. Register-pinned NEON SIMD (archsimd.Float32x4) for remainder rows (1..3 rows), causal masks,
//     or boolean masks.
func DispatchSDPAARM64(
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
	if !gobackend.IsNEONAllowed {
		return false
	}

	kvLenUnmaskedBase := min(kvLen, kvLimit)
	qValidEnd := min(qEnd, qLimit)
	qValidCount := max(0, qValidEnd-qStart)

	canBatchQueries := !causal && len(booleanMask) == 0 && kvLenUnmaskedBase > 0 && NEONUseAsm

	// Try acquiring a cluster SME unit when the problem size has at least one 32x32 tile.
	useSME := false
	clusterIdx := -1
	if canBatchQueries && gobackend.IsSMEAllowed &&
		qValidCount >= 32 && kvLenUnmaskedBase >= 32 &&
		(headDim == 32 || headDim == 64 || headDim == 128) &&
		len(scoresScratch) >= 32*kvLen {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		clusterIdx, useSME, _ = smeAcquireCluster()
		if useSME {
			defer smeReleaseCluster(clusterIdx)
		}
	}

	maskSeqStride := kvLen
	if maskLen := max(len(additiveMask), len(booleanMask)); maskLen == 0 || maskGroupStride == kvLen || (maskGroupStride == 0 && maskLen == kvLen) {
		maskSeqStride = 0
	}
	biasSeqStride := kvLen
	if len(additiveBias) == 0 || biasGroupStride == kvLen || (biasGroupStride == 0 && len(additiveBias) == kvLen) {
		biasSeqStride = 0
	}

	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		qIdx := qStart

		// 1. 32-Query-Row SME 32x32 Outer-Product Path:
		if useSME && kvLenUnmaskedBase >= 32 {
			for ; qIdx+32 <= qValidEnd; qIdx += 32 {
				sdpaSME32RowsFloat32(
					q, k, v, output,
					gQOff+qIdx*qSeqStride, kvOff, qSeqStride, kvSeqStride,
					additiveMask, gMaskOff+qIdx*maskSeqStride, maskSeqStride,
					additiveBias, gBiasOff+qIdx*biasSeqStride, biasSeqStride,
					scoresScratch,
					kvLen, kvLenUnmaskedBase, headDim, scale,
				)
			}
		}

		// 2. 4-Query-Row NEON Assembly Path (Tile4x4 Q@K^T + NEON Softmax + Tile4x16 P@V):
		if canBatchQueries && headDim >= 4 && len(scoresScratch) >= 4*kvLen {
			for ; qIdx+4 <= qValidEnd; qIdx += 4 {
				sdpaNEON4RowsFloat32(
					q, k, v, output,
					gQOff+qIdx*qSeqStride, kvOff, qSeqStride, kvSeqStride,
					additiveMask, gMaskOff+qIdx*maskSeqStride, maskSeqStride,
					additiveBias, gBiasOff+qIdx*biasSeqStride, biasSeqStride,
					scoresScratch,
					kvLen, kvLenUnmaskedBase, headDim, scale,
				)
			}
		}

		// 3. Remainder / Causal / BooleanMask Path (2-row and 1-row NEON archsimd):
		if canBatchQueries && headDim == 32 && len(scoresScratch) >= 2*kvLen {
			for ; qIdx+2 <= qValidEnd; qIdx += 2 {
				qBase0 := gQOff + qIdx*qSeqStride
				qBase1 := qBase0 + qSeqStride
				maskIdxBase0 := gMaskOff + qIdx*maskSeqStride
				maskIdxBase1 := maskIdxBase0 + maskSeqStride
				biasIdxBase0 := gBiasOff + qIdx*biasSeqStride
				biasIdxBase1 := biasIdxBase0 + biasSeqStride
				scores0 := scoresScratch[:kvLen]
				scores1 := scoresScratch[kvLen : 2*kvLen]

				rowMax0, rowMax1 := sdpaDotNEONDim32Pair(
					q, k,
					qBase0, qBase1, kvOff, kvSeqStride,
					additiveMask, additiveBias,
					maskIdxBase0, maskIdxBase1, biasIdxBase0, biasIdxBase1,
					scores0, scores1,
					kvLenUnmaskedBase, scale,
				)
				allMasked0 := rowMax0 == float32(math.Inf(-1))
				allMasked1 := rowMax1 == float32(math.Inf(-1))
				if !allMasked0 && !allMasked1 {
					sdpaSoftmaxRowNEON(scores0, kvLenUnmaskedBase, rowMax0)
					sdpaSoftmaxRowNEON(scores1, kvLenUnmaskedBase, rowMax1)
					sdpaValueAccumNEONDim32Pair(v, output, qBase0, qBase1, kvOff, kvSeqStride, scores0, scores1, kvLenUnmaskedBase)
					continue
				}
				if allMasked0 {
					clear(output[qBase0 : qBase0+32])
				} else {
					sdpaSoftmaxRowNEON(scores0, kvLenUnmaskedBase, rowMax0)
					sdpaValueAccumNEONDim32Single(v, output, qBase0, kvOff, kvSeqStride, scores0, kvLenUnmaskedBase)
				}
				if allMasked1 {
					clear(output[qBase1 : qBase1+32])
				} else {
					sdpaSoftmaxRowNEON(scores1, kvLenUnmaskedBase, rowMax1)
					sdpaValueAccumNEONDim32Single(v, output, qBase1, kvOff, kvSeqStride, scores1, kvLenUnmaskedBase)
				}
			}
		}

		for ; qIdx < qEnd; qIdx++ {
			outBase := gQOff + qIdx*qSeqStride
			if qIdx >= qLimit {
				clear(output[outBase : outBase+headDim])
				continue
			}

			kvLenUnmasked := kvLenUnmaskedBase
			if causal && qIdx+1 < kvLenUnmasked {
				kvLenUnmasked = qIdx + 1
			}
			if kvLenUnmasked <= 0 {
				clear(output[outBase : outBase+headDim])
				continue
			}

			qBase := gQOff + qIdx*qSeqStride
			maskIdxBase := gMaskOff + qIdx*maskSeqStride
			biasIdxBase := gBiasOff + qIdx*biasSeqStride

			if headDim == 32 {
				rowMax, activeKVLen := sdpaDotNEONDim32Single(
					q, k,
					qBase, kvOff, kvSeqStride,
					additiveMask, booleanMask, maskIdxBase,
					additiveBias, biasIdxBase,
					scoresScratch, kvLenUnmasked, scale,
				)
				if activeKVLen <= 0 || rowMax == float32(math.Inf(-1)) {
					clear(output[outBase : outBase+32])
					continue
				}
				sdpaSoftmaxRowNEON(scoresScratch, activeKVLen, rowMax)
				sdpaValueAccumNEONDim32Single(v, output, outBase, kvOff, kvSeqStride, scoresScratch, activeKVLen)
			} else {
				sdpaSingleRowNEONGeneral(
					q, k, v, output,
					qBase, outBase, kvOff, kvSeqStride,
					additiveMask, booleanMask, maskIdxBase,
					additiveBias, biasIdxBase,
					scoresScratch, kvLenUnmasked, headDim, scale,
				)
			}
		}
	}
	return true
}

// sdpaSME32RowsFloat32 computes 32 query rows of SDPA using SME 32x32 FMOPA microkernels for
// Q @ K^T and Probs @ V, with NEON Tile4x4/Tile4x1 handling any kvLenUnmasked % 32 remainder columns.
func sdpaSME32RowsFloat32(
	q, k, v, output []float32,
	qBase, kvOff, qSeqStride, kvSeqStride int,
	additiveMask []float32, maskIdxBase, maskSeqStride int,
	additiveBias []float32, biasIdxBase, biasSeqStride int,
	scoresScratch []float32,
	kvLen, kvLenUnmasked, headDim int,
	scale float32,
) {
	// Stack buffers for 32xheadDim (up to headDim=128 -> 4096 floats = 16 KB).
	var packedQBuf [32 * 128]float32
	var packedKBuf [32 * 128]float32
	packedQ := packedQBuf[:32*headDim]
	packedK := packedKBuf[:32*headDim]

	// 1. Pack Q[32, headDim] -> packedQ[headDim, 32]:
	neonPackLHS(q[qBase:], packedQ, 0, 0, qSeqStride, 32, headDim, 32)

	// 2. Compute S[32, kvLenUnmasked] = Q[32, headDim] @ K[kvLenUnmasked, headDim]^T:
	kvSMEEnd := (kvLenUnmasked / 32) * 32
	for ki := 0; ki < kvSMEEnd; ki += 32 {
		// Since K is transposed, packing 32 rows of K[ki:ki+32, headDim] with neonPackLHS produces
		// the exact [headDim, 32] strip layout expected as the RHS of smeLargeKernelFloat32Asm_32x32.
		neonPackLHS(k[kvOff+ki*kvSeqStride:], packedK, 0, 0, kvSeqStride, 32, headDim, 32)
		smeLargeKernelFloat32Asm_32x32(
			packedQ,
			packedK,
			scoresScratch[ki:],
			32,
			kvLen,
			headDim,
			32,
			32,
			false,
		)
	}

	// Remainder key columns [kvSMEEnd, kvLenUnmasked) using NEON Tile4x4 / Tile4x1 assembly:
	if kvSMEEnd < kvLenUnmasked {
		scoresStrideBytes := kvLen * 4
		for r := 0; r < 32; r += 4 {
			qRow0 := unsafe.Pointer(&q[qBase+(r+0)*qSeqStride])
			qRow1 := unsafe.Pointer(&q[qBase+(r+1)*qSeqStride])
			qRow2 := unsafe.Pointer(&q[qBase+(r+2)*qSeqStride])
			qRow3 := unsafe.Pointer(&q[qBase+(r+3)*qSeqStride])
			ki := kvSMEEnd
			for ; ki+3 < kvLenUnmasked; ki += 4 {
				kCol0 := unsafe.Pointer(&k[kvOff+(ki+0)*kvSeqStride])
				kCol1 := unsafe.Pointer(&k[kvOff+(ki+1)*kvSeqStride])
				kCol2 := unsafe.Pointer(&k[kvOff+(ki+2)*kvSeqStride])
				kCol3 := unsafe.Pointer(&k[kvOff+(ki+3)*kvSeqStride])
				outPtr := unsafe.Pointer(&scoresScratch[r*kvLen+ki])
				neonSmallTransposedTile4x4Float32Asm(
					qRow0, qRow1, qRow2, qRow3,
					kCol0, kCol1, kCol2, kCol3,
					headDim,
					outPtr,
					scoresStrideBytes,
				)
			}
			for ; ki < kvLenUnmasked; ki++ {
				kCol0 := unsafe.Pointer(&k[kvOff+ki*kvSeqStride])
				outPtr := unsafe.Pointer(&scoresScratch[r*kvLen+ki])
				neonSmallTransposedTile4x1Float32Asm(
					qRow0, qRow1, qRow2, qRow3,
					kCol0,
					headDim,
					outPtr,
					scoresStrideBytes,
				)
			}
		}
	}

	// 3. Fused Scale + Bias/Mask + Softmax on each of the 32 score rows:
	for r := range 32 {
		rowScores := scoresScratch[r*kvLen : r*kvLen+kvLenUnmasked]
		var maskRow, biasRow []float32
		if len(additiveMask) > 0 {
			mStart := maskIdxBase + r*maskSeqStride
			maskRow = additiveMask[mStart : mStart+kvLenUnmasked]
		}
		if len(additiveBias) > 0 {
			bStart := biasIdxBase + r*biasSeqStride
			biasRow = additiveBias[bStart : bStart+kvLenUnmasked]
		}
		sdpaScaleMaskSoftmaxRowNEON(rowScores, maskRow, biasRow, kvLenUnmasked, scale)
	}

	// 4. Compute Output[32, headDim] = Probs[32, kvLenUnmasked] @ V[kvLenUnmasked, headDim]:
	var stackPackedS [32 * 256]float32
	var stackPackedV [32 * 256]float32
	var packedS, packedV []float32
	if kvLenUnmasked <= 256 {
		packedS = stackPackedS[:32*kvLenUnmasked]
		packedV = stackPackedV[:32*kvLenUnmasked]
	} else {
		packedS = make([]float32, 32*kvLenUnmasked)
		packedV = make([]float32, 32*kvLenUnmasked)
	}

	neonPackLHS(scoresScratch, packedS, 0, 0, kvLen, 32, kvLenUnmasked, 32)
	for d := 0; d < headDim; d += 32 {
		neonPackRHSNonTransposed(v[kvOff+d:], packedV, 0, 0, kvSeqStride, kvLenUnmasked, 32, 32)
		smeLargeKernelFloat32Asm_32x32(
			packedS,
			packedV,
			output[qBase+d:],
			32,
			qSeqStride,
			kvLenUnmasked,
			32,
			32,
			false,
		)
	}
}

// sdpaNEON4RowsFloat32 computes 4 query rows of SDPA using handwritten ARM64 NEON assembly
// microkernels (neonSmallTransposedTile4x4Float32Asm for Q @ K^T and
// neonSmallNonTransposedTile4x16Float32Asm for Probs @ V) with zero packing.
func sdpaNEON4RowsFloat32(
	q, k, v, output []float32,
	qBase, kvOff, qSeqStride, kvSeqStride int,
	additiveMask []float32, maskIdxBase, maskSeqStride int,
	additiveBias []float32, biasIdxBase, biasSeqStride int,
	scoresScratch []float32,
	kvLen, kvLenUnmasked, headDim int,
	scale float32,
) {
	qRow0 := unsafe.Pointer(&q[qBase+0*qSeqStride])
	qRow1 := unsafe.Pointer(&q[qBase+1*qSeqStride])
	qRow2 := unsafe.Pointer(&q[qBase+2*qSeqStride])
	qRow3 := unsafe.Pointer(&q[qBase+3*qSeqStride])
	scoresStrideBytes := kvLen * 4

	// 1. Q[4, headDim] @ K[kvLenUnmasked, headDim]^T -> scoresScratch[4, kvLen]:
	ki := 0
	for ; ki+3 < kvLenUnmasked; ki += 4 {
		kCol0 := unsafe.Pointer(&k[kvOff+(ki+0)*kvSeqStride])
		kCol1 := unsafe.Pointer(&k[kvOff+(ki+1)*kvSeqStride])
		kCol2 := unsafe.Pointer(&k[kvOff+(ki+2)*kvSeqStride])
		kCol3 := unsafe.Pointer(&k[kvOff+(ki+3)*kvSeqStride])
		outPtr := unsafe.Pointer(&scoresScratch[ki])
		neonSmallTransposedTile4x4Float32Asm(
			qRow0, qRow1, qRow2, qRow3,
			kCol0, kCol1, kCol2, kCol3,
			headDim,
			outPtr,
			scoresStrideBytes,
		)
	}
	for ; ki < kvLenUnmasked; ki++ {
		kCol0 := unsafe.Pointer(&k[kvOff+ki*kvSeqStride])
		outPtr := unsafe.Pointer(&scoresScratch[ki])
		neonSmallTransposedTile4x1Float32Asm(
			qRow0, qRow1, qRow2, qRow3,
			kCol0,
			headDim,
			outPtr,
			scoresStrideBytes,
		)
	}

	// 2. Scale + Bias/Mask + Softmax on each of the 4 score rows:
	for r := range 4 {
		rowScores := scoresScratch[r*kvLen : r*kvLen+kvLenUnmasked]
		var maskRow, biasRow []float32
		if len(additiveMask) > 0 {
			mStart := maskIdxBase + r*maskSeqStride
			maskRow = additiveMask[mStart : mStart+kvLenUnmasked]
		}
		if len(additiveBias) > 0 {
			bStart := biasIdxBase + r*biasSeqStride
			biasRow = additiveBias[bStart : bStart+kvLenUnmasked]
		}
		sdpaScaleMaskSoftmaxRowNEON(rowScores, maskRow, biasRow, kvLenUnmasked, scale)
	}

	// 3. Probs[4, kvLenUnmasked] @ V[kvLenUnmasked, headDim] -> output[4, headDim]:
	sRow0 := unsafe.Pointer(&scoresScratch[0*kvLen])
	sRow1 := unsafe.Pointer(&scoresScratch[1*kvLen])
	sRow2 := unsafe.Pointer(&scoresScratch[2*kvLen])
	sRow3 := unsafe.Pointer(&scoresScratch[3*kvLen])
	vStrideBytes := kvSeqStride * 4

	d := 0
	for ; d+15 < headDim; d += 16 {
		rhsCol := unsafe.Pointer(&v[kvOff+d])
		outRow0 := unsafe.Pointer(&output[qBase+0*qSeqStride+d])
		outRow1 := unsafe.Pointer(&output[qBase+1*qSeqStride+d])
		outRow2 := unsafe.Pointer(&output[qBase+2*qSeqStride+d])
		outRow3 := unsafe.Pointer(&output[qBase+3*qSeqStride+d])
		neonSmallNonTransposedTile4x16Float32Asm(
			sRow0, sRow1, sRow2, sRow3,
			rhsCol,
			kvLenUnmasked,
			vStrideBytes,
			outRow0, outRow1, outRow2, outRow3,
		)
	}
	for ; d+3 < headDim; d += 4 {
		rhsCol := unsafe.Pointer(&v[kvOff+d])
		outRow0 := unsafe.Pointer(&output[qBase+0*qSeqStride+d])
		outRow1 := unsafe.Pointer(&output[qBase+1*qSeqStride+d])
		outRow2 := unsafe.Pointer(&output[qBase+2*qSeqStride+d])
		outRow3 := unsafe.Pointer(&output[qBase+3*qSeqStride+d])
		neonSmallNonTransposedTile4x4Float32Asm(
			sRow0, sRow1, sRow2, sRow3,
			rhsCol,
			kvLenUnmasked,
			vStrideBytes,
			outRow0, outRow1, outRow2, outRow3,
		)
	}
	for ; d < headDim; d++ {
		var acc0, acc1, acc2, acc3 float32
		for kIdx := range kvLenUnmasked {
			vVal := v[kvOff+kIdx*kvSeqStride+d]
			acc0 += scoresScratch[0*kvLen+kIdx] * vVal
			acc1 += scoresScratch[1*kvLen+kIdx] * vVal
			acc2 += scoresScratch[2*kvLen+kIdx] * vVal
			acc3 += scoresScratch[3*kvLen+kIdx] * vVal
		}
		output[qBase+0*qSeqStride+d] = acc0
		output[qBase+1*qSeqStride+d] = acc1
		output[qBase+2*qSeqStride+d] = acc2
		output[qBase+3*qSeqStride+d] = acc3
	}
}

// sdpaScaleMaskSoftmaxRowNEON applies s = s * scale + bias + mask and computes in-place Softmax.
func sdpaScaleMaskSoftmaxRowNEON(scores, maskRow, biasRow []float32, kvLenUnmasked int, scale float32) {
	vScale := archsimd.BroadcastFloat32x4(scale)
	hasBias := len(biasRow) > 0
	hasMask := len(maskRow) > 0
	rowMax := float32(math.Inf(-1))

	k := 0
	if kvLenUnmasked >= 4 {
		vMax := archsimd.BroadcastFloat32x4(float32(math.Inf(-1)))
		for ; k+4 <= kvLenUnmasked; k += 4 {
			vS := archsimd.LoadFloat32x4(scores[k : k+4]).Mul(vScale)
			if hasBias {
				vS = vS.Add(archsimd.LoadFloat32x4(biasRow[k : k+4]))
			}
			if hasMask {
				vS = vS.Add(archsimd.LoadFloat32x4(maskRow[k : k+4]))
			}
			vS.Store(scores[k : k+4])
			vMax = vMax.Max(vS)
		}
		rowMax = vMax.ReduceMax()
	}
	for ; k < kvLenUnmasked; k++ {
		s := scores[k] * scale
		if hasBias {
			s += biasRow[k]
		}
		if hasMask {
			s += maskRow[k]
		}
		scores[k] = s
		if s > rowMax {
			rowMax = s
		}
	}

	if rowMax == float32(math.Inf(-1)) {
		clear(scores[:kvLenUnmasked])
		return
	}
	sdpaSoftmaxRowNEON(scores, kvLenUnmasked, rowMax)
}

func sdpaSoftmaxRowNEON(scores []float32, kvLenUnmasked int, rowMax float32) {
	const (
		maxLogF = 88.02969187150841
		minLogF = -88.02969187150841
		log2E   = 1.44269504088896341
		magicF  = 12582912.0
		ln2Hi   = 0.693359375
		ln2Lo   = -2.12194440e-4

		p7 = 1.9875691500e-4
		p6 = 1.3981999507e-3
		p5 = 8.3334519073e-3
		p4 = 4.1665795894e-2
		p3 = 1.6666665459e-1
		p2 = 5.0000001201e-1
	)

	vMaxLog := archsimd.BroadcastFloat32x4(maxLogF)
	vMinLog := archsimd.BroadcastFloat32x4(minLogF)
	vLog2E := archsimd.BroadcastFloat32x4(log2E)
	vMagic := archsimd.BroadcastFloat32x4(magicF)
	vLn2Hi := archsimd.BroadcastFloat32x4(ln2Hi)
	vLn2Lo := archsimd.BroadcastFloat32x4(ln2Lo)

	vP7 := archsimd.BroadcastFloat32x4(p7)
	vP6 := archsimd.BroadcastFloat32x4(p6)
	vP5 := archsimd.BroadcastFloat32x4(p5)
	vP4 := archsimd.BroadcastFloat32x4(p4)
	vP3 := archsimd.BroadcastFloat32x4(p3)
	vP2 := archsimd.BroadcastFloat32x4(p2)
	vOne := archsimd.BroadcastFloat32x4(1.0)
	vShift23 := archsimd.BroadcastUint32x4(1 << 23)
	vBias127 := archsimd.BroadcastUint32x4(127 << 23)

	vRowMax := archsimd.BroadcastFloat32x4(rowMax)
	vSum := archsimd.BroadcastFloat32x4(0)

	k := 0
	for ; k+4 <= kvLenUnmasked; k += 4 {
		x := archsimd.LoadFloat32x4(scores[k : k+4]).Sub(vRowMax)
		x = x.Max(vMinLog).Min(vMaxLog)

		zRound := x.MulAdd(vLog2E, vMagic)
		z := zRound.Sub(vMagic)
		g := x.Sub(z.Mul(vLn2Hi)).Sub(z.Mul(vLn2Lo))

		g2 := g.Mul(g)
		g4 := g2.Mul(g2)
		q67 := g.MulAdd(vP7, vP6)
		q45 := g.MulAdd(vP5, vP4)
		q23 := g.MulAdd(vP3, vP2)
		q01 := g.Add(vOne)
		polyHi := g2.MulAdd(q67, q45)
		polyLo := g2.MulAdd(q23, q01)
		poly := g4.MulAdd(polyHi, polyLo)

		scale := zRound.ToBits().Mul(vShift23).Add(vBias127).BitsToFloat32()
		expV := poly.Mul(scale)
		expV.Store(scores[k : k+4])
		vSum = vSum.Add(expV)
	}

	var sum float32
	if k > 0 {
		sum = neonReduceSumFloat32x4(vSum)
	}
	for ; k < kvLenUnmasked; k++ {
		s := scores[k]
		if s == float32(math.Inf(-1)) {
			scores[k] = 0
			continue
		}
		x := s - rowMax
		if x < minLogF {
			scores[k] = 0
			continue
		}
		if x > maxLogF {
			x = maxLogF
		}
		zRound := x*log2E + magicF
		z := zRound - magicF
		g := x - z*ln2Hi - z*ln2Lo
		g2 := g * g
		g4 := g2 * g2
		poly := g4*(g2*(p7*g+p6)+(p5*g+p4)) + (g2*(p3*g+p2) + (g + 1.0))
		e := poly * math.Float32frombits((math.Float32bits(zRound)<<23)+(127<<23))
		scores[k] = e
		sum += e
	}

	var invSum float32
	if sum > 0 {
		invSum = 1.0 / sum
	}
	vInvSum := archsimd.BroadcastFloat32x4(invSum)
	k = 0
	for ; k+16 <= kvLenUnmasked; k += 16 {
		s0 := archsimd.LoadFloat32x4(scores[k : k+4]).Mul(vInvSum)
		s1 := archsimd.LoadFloat32x4(scores[k+4 : k+8]).Mul(vInvSum)
		s2 := archsimd.LoadFloat32x4(scores[k+8 : k+12]).Mul(vInvSum)
		s3 := archsimd.LoadFloat32x4(scores[k+12 : k+16]).Mul(vInvSum)
		s0.Store(scores[k : k+4])
		s1.Store(scores[k+4 : k+8])
		s2.Store(scores[k+8 : k+12])
		s3.Store(scores[k+12 : k+16])
	}
	for ; k+4 <= kvLenUnmasked; k += 4 {
		archsimd.LoadFloat32x4(scores[k : k+4]).Mul(vInvSum).Store(scores[k : k+4])
	}
	for ; k < kvLenUnmasked; k++ {
		scores[k] *= invSum
	}
}

func sdpaDotNEONDim32Pair(
	q, k []float32,
	qBase0, qBase1, kvOff, kvSeqStride int,
	additiveMask, additiveBias []float32,
	maskIdxBase0, maskIdxBase1, biasIdxBase0, biasIdxBase1 int,
	scores0, scores1 []float32,
	kvLenUnmasked int,
	scale float32,
) (rowMax0, rowMax1 float32) {
	qA0 := archsimd.LoadFloat32x4(q[qBase0 : qBase0+4])
	qA1 := archsimd.LoadFloat32x4(q[qBase0+4 : qBase0+8])
	qA2 := archsimd.LoadFloat32x4(q[qBase0+8 : qBase0+12])
	qA3 := archsimd.LoadFloat32x4(q[qBase0+12 : qBase0+16])
	qA4 := archsimd.LoadFloat32x4(q[qBase0+16 : qBase0+20])
	qA5 := archsimd.LoadFloat32x4(q[qBase0+20 : qBase0+24])
	qA6 := archsimd.LoadFloat32x4(q[qBase0+24 : qBase0+28])
	qA7 := archsimd.LoadFloat32x4(q[qBase0+28 : qBase0+32])

	qB0 := archsimd.LoadFloat32x4(q[qBase1 : qBase1+4])
	qB1 := archsimd.LoadFloat32x4(q[qBase1+4 : qBase1+8])
	qB2 := archsimd.LoadFloat32x4(q[qBase1+8 : qBase1+12])
	qB3 := archsimd.LoadFloat32x4(q[qBase1+12 : qBase1+16])
	qB4 := archsimd.LoadFloat32x4(q[qBase1+16 : qBase1+20])
	qB5 := archsimd.LoadFloat32x4(q[qBase1+20 : qBase1+24])
	qB6 := archsimd.LoadFloat32x4(q[qBase1+24 : qBase1+28])
	qB7 := archsimd.LoadFloat32x4(q[qBase1+28 : qBase1+32])

	rowMax0 = float32(math.Inf(-1))
	rowMax1 = float32(math.Inf(-1))

	for ki := range kvLenUnmasked {
		kBase := kvOff + ki*kvSeqStride
		k0 := archsimd.LoadFloat32x4(k[kBase : kBase+4])
		k1 := archsimd.LoadFloat32x4(k[kBase+4 : kBase+8])
		k2 := archsimd.LoadFloat32x4(k[kBase+8 : kBase+12])
		k3 := archsimd.LoadFloat32x4(k[kBase+12 : kBase+16])
		k4 := archsimd.LoadFloat32x4(k[kBase+16 : kBase+20])
		k5 := archsimd.LoadFloat32x4(k[kBase+20 : kBase+24])
		k6 := archsimd.LoadFloat32x4(k[kBase+24 : kBase+28])
		k7 := archsimd.LoadFloat32x4(k[kBase+28 : kBase+32])

		sA0 := qA4.MulAdd(k4, qA0.Mul(k0))
		sA1 := qA5.MulAdd(k5, qA1.Mul(k1))
		sA2 := qA6.MulAdd(k6, qA2.Mul(k2))
		sA3 := qA7.MulAdd(k7, qA3.Mul(k3))
		sumA := sA0.Add(sA1).Add(sA2.Add(sA3))

		sB0 := qB4.MulAdd(k4, qB0.Mul(k0))
		sB1 := qB5.MulAdd(k5, qB1.Mul(k1))
		sB2 := qB6.MulAdd(k6, qB2.Mul(k2))
		sB3 := qB7.MulAdd(k7, qB3.Mul(k3))
		sumB := sB0.Add(sB1).Add(sB2.Add(sB3))

		pairAB := sumA.ConcatAddPairs(sumB)
		pairAB = pairAB.ConcatAddPairs(pairAB)
		dot0 := pairAB.GetElem(0)
		dot1 := pairAB.GetElem(1)

		s0 := dot0 * scale
		s1 := dot1 * scale
		if len(additiveBias) > 0 {
			s0 += additiveBias[biasIdxBase0+ki]
			s1 += additiveBias[biasIdxBase1+ki]
		}
		if len(additiveMask) > 0 {
			s0 += additiveMask[maskIdxBase0+ki]
			s1 += additiveMask[maskIdxBase1+ki]
		}
		scores0[ki] = s0
		scores1[ki] = s1
		if s0 > rowMax0 {
			rowMax0 = s0
		}
		if s1 > rowMax1 {
			rowMax1 = s1
		}
	}
	return rowMax0, rowMax1
}

func sdpaValueAccumNEONDim32Pair(
	v, output []float32,
	outBase0, outBase1, kvOff, kvSeqStride int,
	scores0, scores1 []float32,
	kvLenUnmasked int,
) {
	oA0 := archsimd.BroadcastFloat32x4(0)
	oA1 := archsimd.BroadcastFloat32x4(0)
	oA2 := archsimd.BroadcastFloat32x4(0)
	oA3 := archsimd.BroadcastFloat32x4(0)
	oA4 := archsimd.BroadcastFloat32x4(0)
	oA5 := archsimd.BroadcastFloat32x4(0)
	oA6 := archsimd.BroadcastFloat32x4(0)
	oA7 := archsimd.BroadcastFloat32x4(0)

	oB0 := archsimd.BroadcastFloat32x4(0)
	oB1 := archsimd.BroadcastFloat32x4(0)
	oB2 := archsimd.BroadcastFloat32x4(0)
	oB3 := archsimd.BroadcastFloat32x4(0)
	oB4 := archsimd.BroadcastFloat32x4(0)
	oB5 := archsimd.BroadcastFloat32x4(0)
	oB6 := archsimd.BroadcastFloat32x4(0)
	oB7 := archsimd.BroadcastFloat32x4(0)

	for ki := range kvLenUnmasked {
		w0 := scores0[ki]
		w1 := scores1[ki]
		if w0 == 0 && w1 == 0 {
			continue
		}
		vW0 := archsimd.BroadcastFloat32x4(w0)
		vW1 := archsimd.BroadcastFloat32x4(w1)
		vBase := kvOff + ki*kvSeqStride
		v0 := archsimd.LoadFloat32x4(v[vBase : vBase+4])
		v1 := archsimd.LoadFloat32x4(v[vBase+4 : vBase+8])
		v2 := archsimd.LoadFloat32x4(v[vBase+8 : vBase+12])
		v3 := archsimd.LoadFloat32x4(v[vBase+12 : vBase+16])
		v4 := archsimd.LoadFloat32x4(v[vBase+16 : vBase+20])
		v5 := archsimd.LoadFloat32x4(v[vBase+20 : vBase+24])
		v6 := archsimd.LoadFloat32x4(v[vBase+24 : vBase+28])
		v7 := archsimd.LoadFloat32x4(v[vBase+28 : vBase+32])

		oA0 = vW0.MulAdd(v0, oA0)
		oA1 = vW0.MulAdd(v1, oA1)
		oA2 = vW0.MulAdd(v2, oA2)
		oA3 = vW0.MulAdd(v3, oA3)
		oA4 = vW0.MulAdd(v4, oA4)
		oA5 = vW0.MulAdd(v5, oA5)
		oA6 = vW0.MulAdd(v6, oA6)
		oA7 = vW0.MulAdd(v7, oA7)

		oB0 = vW1.MulAdd(v0, oB0)
		oB1 = vW1.MulAdd(v1, oB1)
		oB2 = vW1.MulAdd(v2, oB2)
		oB3 = vW1.MulAdd(v3, oB3)
		oB4 = vW1.MulAdd(v4, oB4)
		oB5 = vW1.MulAdd(v5, oB5)
		oB6 = vW1.MulAdd(v6, oB6)
		oB7 = vW1.MulAdd(v7, oB7)
	}

	oA0.Store(output[outBase0 : outBase0+4])
	oA1.Store(output[outBase0+4 : outBase0+8])
	oA2.Store(output[outBase0+8 : outBase0+12])
	oA3.Store(output[outBase0+12 : outBase0+16])
	oA4.Store(output[outBase0+16 : outBase0+20])
	oA5.Store(output[outBase0+20 : outBase0+24])
	oA6.Store(output[outBase0+24 : outBase0+28])
	oA7.Store(output[outBase0+28 : outBase0+32])

	oB0.Store(output[outBase1 : outBase1+4])
	oB1.Store(output[outBase1+4 : outBase1+8])
	oB2.Store(output[outBase1+8 : outBase1+12])
	oB3.Store(output[outBase1+12 : outBase1+16])
	oB4.Store(output[outBase1+16 : outBase1+20])
	oB5.Store(output[outBase1+20 : outBase1+24])
	oB6.Store(output[outBase1+24 : outBase1+28])
	oB7.Store(output[outBase1+28 : outBase1+32])
}

func sdpaDotNEONDim32Single(
	q, k []float32,
	qBase, kvOff, kvSeqStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskIdxBase int,
	additiveBias []float32,
	biasIdxBase int,
	scores []float32,
	kvLenUnmasked int,
	scale float32,
) (rowMax float32, activeKVLen int) {
	q0 := archsimd.LoadFloat32x4(q[qBase : qBase+4])
	q1 := archsimd.LoadFloat32x4(q[qBase+4 : qBase+8])
	q2 := archsimd.LoadFloat32x4(q[qBase+8 : qBase+12])
	q3 := archsimd.LoadFloat32x4(q[qBase+12 : qBase+16])
	q4 := archsimd.LoadFloat32x4(q[qBase+16 : qBase+20])
	q5 := archsimd.LoadFloat32x4(q[qBase+20 : qBase+24])
	q6 := archsimd.LoadFloat32x4(q[qBase+24 : qBase+28])
	q7 := archsimd.LoadFloat32x4(q[qBase+28 : qBase+32])

	rowMax = float32(math.Inf(-1))
	activeKVLen = kvLenUnmasked
	if len(booleanMask) > 0 {
		activeKVLen = 0
	}

	for ki := range kvLenUnmasked {
		maskIdx := maskIdxBase + ki
		if len(booleanMask) > 0 {
			if !booleanMask[maskIdx] {
				scores[ki] = float32(math.Inf(-1))
				continue
			}
			activeKVLen = ki + 1
		}
		kBase := kvOff + ki*kvSeqStride
		k0 := archsimd.LoadFloat32x4(k[kBase : kBase+4])
		k1 := archsimd.LoadFloat32x4(k[kBase+4 : kBase+8])
		k2 := archsimd.LoadFloat32x4(k[kBase+8 : kBase+12])
		k3 := archsimd.LoadFloat32x4(k[kBase+12 : kBase+16])
		k4 := archsimd.LoadFloat32x4(k[kBase+16 : kBase+20])
		k5 := archsimd.LoadFloat32x4(k[kBase+20 : kBase+24])
		k6 := archsimd.LoadFloat32x4(k[kBase+24 : kBase+28])
		k7 := archsimd.LoadFloat32x4(k[kBase+28 : kBase+32])

		s0 := q4.MulAdd(k4, q0.Mul(k0))
		s1 := q5.MulAdd(k5, q1.Mul(k1))
		s2 := q6.MulAdd(k6, q2.Mul(k2))
		s3 := q7.MulAdd(k7, q3.Mul(k3))
		dot := neonReduceSumFloat32x4(s0.Add(s1).Add(s2.Add(s3)))

		s := dot * scale
		if len(additiveBias) > 0 {
			s += additiveBias[biasIdxBase+ki]
		}
		if len(additiveMask) > 0 {
			s += additiveMask[maskIdx]
		}
		scores[ki] = s
		if s > rowMax {
			rowMax = s
		}
	}
	return rowMax, activeKVLen
}

func sdpaValueAccumNEONDim32Single(
	v, output []float32,
	outBase, kvOff, kvSeqStride int,
	scores []float32,
	kvLenUnmasked int,
) {
	o0 := archsimd.BroadcastFloat32x4(0)
	o1 := archsimd.BroadcastFloat32x4(0)
	o2 := archsimd.BroadcastFloat32x4(0)
	o3 := archsimd.BroadcastFloat32x4(0)
	o4 := archsimd.BroadcastFloat32x4(0)
	o5 := archsimd.BroadcastFloat32x4(0)
	o6 := archsimd.BroadcastFloat32x4(0)
	o7 := archsimd.BroadcastFloat32x4(0)

	for ki := range kvLenUnmasked {
		w := scores[ki]
		if w == 0 {
			continue
		}
		vW := archsimd.BroadcastFloat32x4(w)
		vBase := kvOff + ki*kvSeqStride
		v0 := archsimd.LoadFloat32x4(v[vBase : vBase+4])
		v1 := archsimd.LoadFloat32x4(v[vBase+4 : vBase+8])
		v2 := archsimd.LoadFloat32x4(v[vBase+8 : vBase+12])
		v3 := archsimd.LoadFloat32x4(v[vBase+12 : vBase+16])
		v4 := archsimd.LoadFloat32x4(v[vBase+16 : vBase+20])
		v5 := archsimd.LoadFloat32x4(v[vBase+20 : vBase+24])
		v6 := archsimd.LoadFloat32x4(v[vBase+24 : vBase+28])
		v7 := archsimd.LoadFloat32x4(v[vBase+28 : vBase+32])

		o0 = vW.MulAdd(v0, o0)
		o1 = vW.MulAdd(v1, o1)
		o2 = vW.MulAdd(v2, o2)
		o3 = vW.MulAdd(v3, o3)
		o4 = vW.MulAdd(v4, o4)
		o5 = vW.MulAdd(v5, o5)
		o6 = vW.MulAdd(v6, o6)
		o7 = vW.MulAdd(v7, o7)
	}

	o0.Store(output[outBase : outBase+4])
	o1.Store(output[outBase+4 : outBase+8])
	o2.Store(output[outBase+8 : outBase+12])
	o3.Store(output[outBase+12 : outBase+16])
	o4.Store(output[outBase+16 : outBase+20])
	o5.Store(output[outBase+20 : outBase+24])
	o6.Store(output[outBase+24 : outBase+28])
	o7.Store(output[outBase+28 : outBase+32])
}

func sdpaSingleRowNEONGeneral(
	q, k, v, output []float32,
	qBase, outBase, kvOff, kvSeqStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskIdxBase int,
	additiveBias []float32,
	biasIdxBase int,
	scoresScratch []float32,
	kvLenUnmasked, headDim int,
	scale float32,
) {
	vZero := archsimd.BroadcastFloat32x4(0)
	rowMax := float32(math.Inf(-1))
	activeKVLen := kvLenUnmasked
	if len(booleanMask) > 0 {
		activeKVLen = 0
	}

	for ki := range kvLenUnmasked {
		maskIdx := maskIdxBase + ki
		if len(booleanMask) > 0 {
			if !booleanMask[maskIdx] {
				scoresScratch[ki] = float32(math.Inf(-1))
				continue
			}
			activeKVLen = ki + 1
		}

		kBase := kvOff + ki*kvSeqStride
		d := 0
		s0 := vZero
		s1 := vZero
		s2 := vZero
		s3 := vZero
		for ; d+16 <= headDim; d += 16 {
			q0 := archsimd.LoadFloat32x4(q[qBase+d : qBase+d+4])
			q1 := archsimd.LoadFloat32x4(q[qBase+d+4 : qBase+d+8])
			q2 := archsimd.LoadFloat32x4(q[qBase+d+8 : qBase+d+12])
			q3 := archsimd.LoadFloat32x4(q[qBase+d+12 : qBase+d+16])
			k0 := archsimd.LoadFloat32x4(k[kBase+d : kBase+d+4])
			k1 := archsimd.LoadFloat32x4(k[kBase+d+4 : kBase+d+8])
			k2 := archsimd.LoadFloat32x4(k[kBase+d+8 : kBase+d+12])
			k3 := archsimd.LoadFloat32x4(k[kBase+d+12 : kBase+d+16])
			s0 = q0.MulAdd(k0, s0)
			s1 = q1.MulAdd(k1, s1)
			s2 = q2.MulAdd(k2, s2)
			s3 = q3.MulAdd(k3, s3)
		}
		sumVec := s0.Add(s1).Add(s2.Add(s3))
		for ; d+4 <= headDim; d += 4 {
			qV := archsimd.LoadFloat32x4(q[qBase+d : qBase+d+4])
			kV := archsimd.LoadFloat32x4(k[kBase+d : kBase+d+4])
			sumVec = qV.MulAdd(kV, sumVec)
		}
		var dot float32
		if d > 0 {
			dot = neonReduceSumFloat32x4(sumVec)
		}
		for ; d < headDim; d++ {
			dot += q[qBase+d] * k[kBase+d]
		}

		s := dot * scale
		if len(additiveBias) > 0 {
			s += additiveBias[biasIdxBase+ki]
		}
		if len(additiveMask) > 0 {
			s += additiveMask[maskIdx]
		}
		scoresScratch[ki] = s
		if s > rowMax {
			rowMax = s
		}
	}

	if activeKVLen <= 0 || rowMax == float32(math.Inf(-1)) {
		clear(output[outBase : outBase+headDim])
		return
	}

	sdpaSoftmaxRowNEON(scoresScratch, activeKVLen, rowMax)

	d := 0
	for ; d+32 <= headDim; d += 32 {
		o0 := vZero
		o1 := vZero
		o2 := vZero
		o3 := vZero
		o4 := vZero
		o5 := vZero
		o6 := vZero
		o7 := vZero
		for kIdx := range activeKVLen {
			w := scoresScratch[kIdx]
			if w == 0 {
				continue
			}
			vW := archsimd.BroadcastFloat32x4(w)
			vBase := kvOff + kIdx*kvSeqStride + d
			o0 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase:vBase+4]), o0)
			o1 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase+4:vBase+8]), o1)
			o2 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase+8:vBase+12]), o2)
			o3 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase+12:vBase+16]), o3)
			o4 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase+16:vBase+20]), o4)
			o5 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase+20:vBase+24]), o5)
			o6 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase+24:vBase+28]), o6)
			o7 = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase+28:vBase+32]), o7)
		}
		o0.Store(output[outBase+d : outBase+d+4])
		o1.Store(output[outBase+d+4 : outBase+d+8])
		o2.Store(output[outBase+d+8 : outBase+d+12])
		o3.Store(output[outBase+d+12 : outBase+d+16])
		o4.Store(output[outBase+d+16 : outBase+d+20])
		o5.Store(output[outBase+d+20 : outBase+d+24])
		o6.Store(output[outBase+d+24 : outBase+d+28])
		o7.Store(output[outBase+d+28 : outBase+d+32])
	}
	for ; d+4 <= headDim; d += 4 {
		outV := vZero
		for kIdx := range activeKVLen {
			w := scoresScratch[kIdx]
			if w == 0 {
				continue
			}
			vW := archsimd.BroadcastFloat32x4(w)
			vBase := kvOff + kIdx*kvSeqStride + d
			outV = vW.MulAdd(archsimd.LoadFloat32x4(v[vBase:vBase+4]), outV)
		}
		outV.Store(output[outBase+d : outBase+d+4])
	}
	for ; d < headDim; d++ {
		var acc float32
		for kIdx := range activeKVLen {
			w := scoresScratch[kIdx]
			if w != 0 {
				acc += w * v[kvOff+kIdx*kvSeqStride+d]
			}
		}
		output[outBase+d] = acc
	}
}

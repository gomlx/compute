// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
)

func sdpaScalarReference(
	q, k, v, output []float32,
	qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskGroupStride int,
	additiveBias []float32,
	biasGroupStride int,
	groupSize, qStart, qEnd, kvLen, headDim int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) {
	scores := make([]float32, max(kvLen, 1))
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
		for qIdx := qStart; qIdx < qEnd; qIdx++ {
			outBase := gQOff + qIdx*qSeqStride
			outRow := output[outBase : outBase+headDim]
			if qIdx >= qLimit {
				clear(outRow)
				continue
			}
			kvLenUnmasked := min(kvLen, kvLimit)
			if causal {
				kvLenUnmasked = min(kvLenUnmasked, qIdx+1)
			}
			if kvLenUnmasked <= 0 {
				clear(outRow)
				continue
			}
			rowMax := float32(math.Inf(-1))
			qBase := gQOff + qIdx*qSeqStride
			qRow := q[qBase : qBase+headDim]
			maskIdxBase := gMaskOff + qIdx*maskSeqStride
			biasIdxBase := gBiasOff + qIdx*biasSeqStride

			activeKVLen := kvLenUnmasked
			if len(booleanMask) > 0 {
				activeKVLen = 0
			}
			for kvIdx := range kvLenUnmasked {
				maskIdx := maskIdxBase + kvIdx
				if len(booleanMask) > 0 {
					if !booleanMask[maskIdx] {
						scores[kvIdx] = 0
						continue
					}
					activeKVLen = kvIdx + 1
				}
				kBase := kvOff + kvIdx*kvSeqStride
				kRow := k[kBase : kBase+headDim]
				var dot float32
				for d := range headDim {
					dot += qRow[d] * kRow[d]
				}
				s := dot * scale
				if len(additiveBias) > 0 {
					s += additiveBias[biasIdxBase+kvIdx]
				}
				if len(additiveMask) > 0 {
					s += additiveMask[maskIdx]
				}
				scores[kvIdx] = s
				if s > rowMax {
					rowMax = s
				}
			}
			kvLenUnmasked = activeKVLen
			if kvLenUnmasked <= 0 || rowMax == float32(math.Inf(-1)) {
				clear(outRow)
				continue
			}
			var sum float32
			for kvIdx := range kvLenUnmasked {
				if len(booleanMask) > 0 && !booleanMask[maskIdxBase+kvIdx] {
					scores[kvIdx] = 0
					continue
				}
				e := float32(math.Exp(float64(scores[kvIdx] - rowMax)))
				scores[kvIdx] = e
				sum += e
			}
			var invSum float32
			if sum > 0 {
				invSum = 1.0 / sum
			}
			for kvIdx := range kvLenUnmasked {
				scores[kvIdx] *= invSum
			}
			clear(outRow)
			for kvIdx := range kvLenUnmasked {
				w := scores[kvIdx]
				if w == 0 {
					continue
				}
				vBase := kvOff + kvIdx*kvSeqStride
				vRow := v[vBase : vBase+headDim]
				for d := range headDim {
					outRow[d] += w * vRow[d]
				}
			}
		}
	}
}

func TestDispatchSDPAARM64_AllPaths(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	// Test sizes that exercise:
	//  - 32-row SME path + 4-row NEON assembly path + 2-row NEON path + 1-row tail (qLimit = 39, seqLen = 42)
	//  - kvLimit = 45 (32-col SME block + 12-col NEON Tile4x4 + 1-col NEON Tile4x1)
	//  - Both BHSD (contiguous head stride) and BSHD (interleaved head stride)
	for _, headDim := range []int{16, 32, 64, 128} {
		for _, causal := range []bool{false, true} {
			for _, bshd := range []bool{false, true} {
				seqLen := 42
				kvLen := 48
				qLimit := 39
				kvLimit := 45
				groupSize := 2
				numKVHeads := 2
				numHeads := numKVHeads * groupSize

				var qSeqStride, kvSeqStride, qGroupStride int
				if bshd {
					qSeqStride = numHeads * headDim
					kvSeqStride = numKVHeads * headDim
					qGroupStride = headDim
				} else {
					qSeqStride = headDim
					kvSeqStride = headDim
					qGroupStride = seqLen * headDim
				}

				qSize := numHeads * seqLen * headDim
				kvSize := numKVHeads * kvLen * headDim
				q := make([]float32, qSize)
				k := make([]float32, kvSize)
				v := make([]float32, kvSize)
				got := make([]float32, qSize)
				want := make([]float32, qSize)
				scratch := make([]float32, 32*kvLen)

				for i := range q {
					q[i] = float32((i%17)-8) * 0.1
				}
				for i := range k {
					k[i] = float32((i%13)-6) * 0.1
				}
				for i := range v {
					v[i] = float32((i%19)-9) * 0.2
				}

				additiveMask := make([]float32, groupSize*seqLen*kvLen)
				additiveBias := make([]float32, groupSize*seqLen*kvLen)
				for i := range additiveMask {
					if i%11 == 0 {
						additiveMask[i] = -1.5
					}
					additiveBias[i] = float32((i%7)-3) * 0.05
				}
				maskGroupStride := seqLen * kvLen
				biasGroupStride := seqLen * kvLen
				scale := float32(1.0 / math.Sqrt(float64(headDim)))

				ok := DispatchSDPAARM64(
					q, k, v, got,
					0, 0, qSeqStride, kvSeqStride, qGroupStride,
					additiveMask, nil, maskGroupStride,
					additiveBias, biasGroupStride,
					scratch,
					groupSize, 0, seqLen, kvLen, headDim,
					scale, causal,
					qLimit, kvLimit,
				)
				if !ok {
					t.Fatalf("DispatchSDPAARM64 returned false for headDim=%d", headDim)
				}

				sdpaScalarReference(
					q, k, v, want,
					0, 0, qSeqStride, kvSeqStride, qGroupStride,
					additiveMask, nil, maskGroupStride,
					additiveBias, biasGroupStride,
					groupSize, 0, seqLen, kvLen, headDim,
					scale, causal,
					qLimit, kvLimit,
				)

				for i := range want {
					diff := float32(math.Abs(float64(got[i] - want[i])))
					if diff > 1e-4 {
						t.Fatalf("headDim=%d causal=%v bshd=%v mismatch at index %d: got %f, want %f (diff=%g)",
							headDim, causal, bshd, i, got[i], want[i], diff)
					}
				}

				// Also test query-seq broadcasted mask/bias (mSeqDim == 1, length == kvLen per head or shared).
				for _, perHeadMask := range []bool{false, true} {
					bcastMaskLen := kvLen
					bcastMaskGroupStride := 0
					if perHeadMask {
						bcastMaskLen = groupSize * kvLen
						bcastMaskGroupStride = kvLen
					}
					bcastMask := additiveMask[:bcastMaskLen]
					bcastBias := additiveBias[:bcastMaskLen]
					clear(got)
					clear(want)
					if !DispatchSDPAARM64(
						q, k, v, got,
						0, 0, qSeqStride, kvSeqStride, qGroupStride,
						bcastMask, nil, bcastMaskGroupStride,
						bcastBias, bcastMaskGroupStride,
						scratch,
						groupSize, 0, seqLen, kvLen, headDim,
						scale, causal,
						qLimit, kvLimit,
					) {
						t.Fatalf("DispatchSDPAARM64 returned false for broadcast mask")
					}
					sdpaScalarReference(
						q, k, v, want,
						0, 0, qSeqStride, kvSeqStride, qGroupStride,
						bcastMask, nil, bcastMaskGroupStride,
						bcastBias, bcastMaskGroupStride,
						groupSize, 0, seqLen, kvLen, headDim,
						scale, causal,
						qLimit, kvLimit,
					)
					for i := range want {
						diff := float32(math.Abs(float64(got[i] - want[i])))
						if diff > 1e-4 {
							t.Fatalf("bcastMask perHead=%v headDim=%d causal=%v bshd=%v mismatch at index %d: got %f, want %f (diff=%g)",
								perHeadMask, headDim, causal, bshd, i, got[i], want[i], diff)
						}
					}
				}
			}
		}
	}
}

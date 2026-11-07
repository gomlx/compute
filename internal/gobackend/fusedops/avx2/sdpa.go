// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX2Allowed {
		gobackend.SetSDPAArchDispatcher(gobackend.PriorityArch, DispatchSDPAAVX2)
	}
}

//go:noescape
func sdpaQueriesDim32NoMaskAVX2Asm(
	qPtr, kPtr, vPtr, outPtr, scoresPtr unsafe.Pointer,
	qByteStride, kvByteStride, numQueries, kvLen, qStart, causalInt int,
	scale float32,
)

//go:noescape
func sdpaSoftmaxAndValueAccumDim32AVX2Asm(
	vPtr, outPtr, scoresPtr unsafe.Pointer,
	kvByteStride, kvLenUnmasked int,
	rowMax float32,
)

// reduceSum8 reduces a Float32x8 vector to a single float32 sum.
func reduceSum8(v archsimd.Float32x8) float32 {
	v4 := v.GetLo().Add(v.GetHi())
	v2 := v4.ConcatAddPairs(v4)
	return v2.GetElem(0) + v2.GetElem(1)
}

// reduceMax8 reduces a Float32x8 vector to its maximum element.
func reduceMax8(v archsimd.Float32x8) float32 {
	v4 := v.GetLo().Max(v.GetHi())
	var arr [4]float32
	v4.StoreArray(&arr)
	m := max(arr[0], arr[1])
	return max(m, max(arr[2], arr[3]))
}

// exp256 approximates e^x for 8 float32s using Cephes degree-7 Estrin polynomial
// and magic-number range reduction. Lanes with x <= minLogF (-88.0296936) evaluate
// to exact 0.0f.
func exp256(x archsimd.Float32x8) archsimd.Float32x8 {
	const (
		maxLogF = 88.0296936
		minLogF = -88.0296936
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
	vMaxLog := archsimd.BroadcastFloat32x8(maxLogF)
	vMinLog := archsimd.BroadcastFloat32x8(minLogF)
	vLog2E := archsimd.BroadcastFloat32x8(log2E)
	vMagic := archsimd.BroadcastFloat32x8(magicF)
	vLn2Hi := archsimd.BroadcastFloat32x8(ln2Hi)
	vLn2Lo := archsimd.BroadcastFloat32x8(ln2Lo)

	vP7 := archsimd.BroadcastFloat32x8(p7)
	vP6 := archsimd.BroadcastFloat32x8(p6)
	vP5 := archsimd.BroadcastFloat32x8(p5)
	vP4 := archsimd.BroadcastFloat32x8(p4)
	vP3 := archsimd.BroadcastFloat32x8(p3)
	vP2 := archsimd.BroadcastFloat32x8(p2)
	vOne := archsimd.BroadcastFloat32x8(1.0)
	vBias127 := archsimd.BroadcastUint32x8(127 << 23)

	xClamped := x.Max(vMinLog).Min(vMaxLog)
	zRound := xClamped.MulAdd(vLog2E, vMagic)
	zFloor := zRound.Sub(vMagic)
	g := xClamped.Sub(zFloor.Mul(vLn2Hi)).Sub(zFloor.Mul(vLn2Lo))

	scale := zRound.AsUint32x8().ShiftAllLeft(23).Add(vBias127).AsFloat32x8()

	g2 := g.Mul(g)
	q67 := vP7.MulAdd(g, vP6)
	q45 := vP5.MulAdd(g, vP4)
	q23 := vP3.MulAdd(g, vP2)
	q01 := g.Add(vOne)
	g4 := g2.Mul(g2)
	q47 := q67.MulAdd(g2, q45)
	q03 := q23.MulAdd(g2, q01)
	poly := q47.MulAdd(g4, q03)

	return poly.Mul(scale)
}

// padScoresTail8 writes -Inf into scoresScratch[kvLenUnmasked : kvLenUnmasked+8]
// so 8-wide SIMD softmax loops can process the tail without a scalar remainder loop.
func padScoresTail8(scoresScratch []float32, kvLenUnmasked int) {
	vNegInf := archsimd.BroadcastFloat32x8(float32(math.Inf(-1)))
	vNegInf.Store(scoresScratch[kvLenUnmasked : kvLenUnmasked+8])
}

// DispatchSDPAAVX2 executes scaled dot-product attention using AVX2.
func DispatchSDPAAVX2(
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
	if len(scoresScratch) < kvLen+8 {
		scoresScratch = make([]float32, kvLen+8)
	}
	switch headDim {
	case 32:
		sdpaFloat32AVX2Dim32(
			q, k, v, output,
			qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride,
			additiveMask, booleanMask, maskGroupStride,
			additiveBias, biasGroupStride,
			scoresScratch,
			groupSize, qStart, qEnd, kvLen,
			scale, causal, qLimit, kvLimit,
		)
		return true
	case 64:
		sdpaFloat32AVX2Dim64(
			q, k, v, output,
			qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride,
			additiveMask, booleanMask, maskGroupStride,
			additiveBias, biasGroupStride,
			scoresScratch,
			groupSize, qStart, qEnd, kvLen,
			scale, causal, qLimit, kvLimit,
		)
		return true
	case 128:
		sdpaFloat32AVX2Dim128(
			q, k, v, output,
			qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride,
			additiveMask, booleanMask, maskGroupStride,
			additiveBias, biasGroupStride,
			scoresScratch,
			groupSize, qStart, qEnd, kvLen,
			scale, causal, qLimit, kvLimit,
		)
		return true
	default:
		sdpaFloat32AVX2General(
			q, k, v, output,
			qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride,
			additiveMask, booleanMask, maskGroupStride,
			additiveBias, biasGroupStride,
			scoresScratch,
			groupSize, qStart, qEnd, kvLen, headDim,
			scale, causal, qLimit, kvLimit,
		)
		return true
	}
}

func softmaxAndValueAccumDim32(
	v, output []float32,
	outBase, kvOff, kvSeqStride int,
	scoresScratch []float32,
	kvLenUnmasked int,
	rowMax float32,
) {
	sdpaSoftmaxAndValueAccumDim32AVX2Asm(
		unsafe.Pointer(&v[kvOff]),
		unsafe.Pointer(&output[outBase]),
		unsafe.Pointer(&scoresScratch[0]),
		kvSeqStride*4,
		kvLenUnmasked,
		rowMax,
	)
}

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

func sdpaFloat32AVX2Dim32(
	q, k, v, output []float32,
	qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskGroupStride int,
	additiveBias []float32,
	biasGroupStride int,
	scoresScratch []float32,
	groupSize, qStart, qEnd, kvLen int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) {
	if len(booleanMask) == 0 && len(additiveMask) == 0 && len(additiveBias) == 0 {
		kvLimitClamped := min(kvLen, kvLimit)
		causalInt := 0
		if causal {
			causalInt = 1
		}
		for gIdx := range groupSize {
			gQOff := qOff + gIdx*qGroupStride
			activeQEnd := min(qEnd, qLimit)
			if activeQEnd > qStart && kvLimitClamped > 0 {
				numQ := activeQEnd - qStart
				qBase := gQOff + qStart*qSeqStride
				sdpaQueriesDim32NoMaskAVX2Asm(
					unsafe.Pointer(&q[qBase]),
					unsafe.Pointer(&k[kvOff]),
					unsafe.Pointer(&v[kvOff]),
					unsafe.Pointer(&output[qBase]),
					unsafe.Pointer(&scoresScratch[0]),
					qSeqStride*4,
					kvSeqStride*4,
					numQ,
					kvLimitClamped,
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
		return
	}

	maskSeqStride, biasSeqStride := sdpaMaskBiasSeqStrides(additiveMask, booleanMask, maskGroupStride, additiveBias, biasGroupStride, kvLen)
	vScale := archsimd.BroadcastFloat32x8(scale)
	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		for qIdx := qStart; qIdx < qEnd; qIdx++ {
			outBase := gQOff + qIdx*qSeqStride
			if qIdx >= qLimit {
				clear(output[outBase : outBase+32])
				continue
			}

			kvLenUnmasked := kvLen
			if kvLimit < kvLenUnmasked {
				kvLenUnmasked = kvLimit
			}
			if causal && qIdx+1 < kvLenUnmasked {
				kvLenUnmasked = qIdx + 1
			}

			if kvLenUnmasked <= 0 {
				clear(output[outBase : outBase+32])
				continue
			}

			qBase := gQOff + qIdx*qSeqStride
			maskIdxBase := gMaskOff + qIdx*maskSeqStride
			biasIdxBase := gBiasOff + qIdx*biasSeqStride

			q0 := archsimd.LoadFloat32x8(q[qBase : qBase+8]).Mul(vScale)
			q1 := archsimd.LoadFloat32x8(q[qBase+8 : qBase+16]).Mul(vScale)
			q2 := archsimd.LoadFloat32x8(q[qBase+16 : qBase+24]).Mul(vScale)
			q3 := archsimd.LoadFloat32x8(q[qBase+24 : qBase+32]).Mul(vScale)

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
				k0 := archsimd.LoadFloat32x8(k[kBase : kBase+8])
				k1 := archsimd.LoadFloat32x8(k[kBase+8 : kBase+16])
				k2 := archsimd.LoadFloat32x8(k[kBase+16 : kBase+24])
				k3 := archsimd.LoadFloat32x8(k[kBase+24 : kBase+32])

				sumVec := q0.Mul(k0).Add(q1.Mul(k1)).Add(q2.Mul(k2)).Add(q3.Mul(k3))
				s := reduceSum8(sumVec)
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
				clear(output[outBase : outBase+32])
				continue
			}

			softmaxAndValueAccumDim32(v, output, outBase, kvOff, kvSeqStride, scoresScratch, activeKVLen, rowMax)
		}
	}
}

func softmaxAndValueAccumDim64(
	v, output []float32,
	outBase, kvOff, kvSeqStride int,
	scoresScratch []float32,
	kvLenUnmasked int,
	rowMax float32,
) {
	padScoresTail8(scoresScratch, kvLenUnmasked)

	vRowMax := archsimd.BroadcastFloat32x8(rowMax)
	vSum := archsimd.BroadcastFloat32x8(0)
	for k := 0; k < kvLenUnmasked; k += 8 {
		vScores := archsimd.LoadFloat32x8(scoresScratch[k : k+8])
		expV := exp256(vScores.Sub(vRowMax))
		expV.Store(scoresScratch[k : k+8])
		vSum = vSum.Add(expV)
	}
	sum := reduceSum8(vSum)

	var invSum float32
	if sum > 0 {
		invSum = 1.0 / sum
	}

	out0 := archsimd.BroadcastFloat32x8(0)
	out1 := archsimd.BroadcastFloat32x8(0)
	out2 := archsimd.BroadcastFloat32x8(0)
	out3 := archsimd.BroadcastFloat32x8(0)
	out4 := archsimd.BroadcastFloat32x8(0)
	out5 := archsimd.BroadcastFloat32x8(0)
	out6 := archsimd.BroadcastFloat32x8(0)
	out7 := archsimd.BroadcastFloat32x8(0)

	for ki := 0; ki < kvLenUnmasked; ki++ {
		vW := archsimd.BroadcastFloat32x8(scoresScratch[ki])
		vBase := kvOff + ki*kvSeqStride
		out0 = archsimd.LoadFloat32x8(v[vBase : vBase+8]).MulAdd(vW, out0)
		out1 = archsimd.LoadFloat32x8(v[vBase+8 : vBase+16]).MulAdd(vW, out1)
		out2 = archsimd.LoadFloat32x8(v[vBase+16 : vBase+24]).MulAdd(vW, out2)
		out3 = archsimd.LoadFloat32x8(v[vBase+24 : vBase+32]).MulAdd(vW, out3)
		out4 = archsimd.LoadFloat32x8(v[vBase+32 : vBase+40]).MulAdd(vW, out4)
		out5 = archsimd.LoadFloat32x8(v[vBase+40 : vBase+48]).MulAdd(vW, out5)
		out6 = archsimd.LoadFloat32x8(v[vBase+48 : vBase+56]).MulAdd(vW, out6)
		out7 = archsimd.LoadFloat32x8(v[vBase+56 : vBase+64]).MulAdd(vW, out7)
	}

	vInvSum := archsimd.BroadcastFloat32x8(invSum)
	out0.Mul(vInvSum).Store(output[outBase : outBase+8])
	out1.Mul(vInvSum).Store(output[outBase+8 : outBase+16])
	out2.Mul(vInvSum).Store(output[outBase+16 : outBase+24])
	out3.Mul(vInvSum).Store(output[outBase+24 : outBase+32])
	out4.Mul(vInvSum).Store(output[outBase+32 : outBase+40])
	out5.Mul(vInvSum).Store(output[outBase+40 : outBase+48])
	out6.Mul(vInvSum).Store(output[outBase+48 : outBase+56])
	out7.Mul(vInvSum).Store(output[outBase+56 : outBase+64])
}

func sdpaFloat32AVX2Dim64(
	q, k, v, output []float32,
	qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskGroupStride int,
	additiveBias []float32,
	biasGroupStride int,
	scoresScratch []float32,
	groupSize, qStart, qEnd, kvLen int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) {
	maskSeqStride, biasSeqStride := sdpaMaskBiasSeqStrides(additiveMask, booleanMask, maskGroupStride, additiveBias, biasGroupStride, kvLen)
	vScale := archsimd.BroadcastFloat32x8(scale)
	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		for qIdx := qStart; qIdx < qEnd; qIdx++ {
			outBase := gQOff + qIdx*qSeqStride
			if qIdx >= qLimit {
				clear(output[outBase : outBase+64])
				continue
			}

			kvLenUnmasked := kvLen
			if kvLimit < kvLenUnmasked {
				kvLenUnmasked = kvLimit
			}
			if causal && qIdx+1 < kvLenUnmasked {
				kvLenUnmasked = qIdx + 1
			}

			if kvLenUnmasked <= 0 {
				clear(output[outBase : outBase+64])
				continue
			}

			qBase := gQOff + qIdx*qSeqStride
			maskIdxBase := gMaskOff + qIdx*maskSeqStride
			biasIdxBase := gBiasOff + qIdx*biasSeqStride

			q0 := archsimd.LoadFloat32x8(q[qBase : qBase+8]).Mul(vScale)
			q1 := archsimd.LoadFloat32x8(q[qBase+8 : qBase+16]).Mul(vScale)
			q2 := archsimd.LoadFloat32x8(q[qBase+16 : qBase+24]).Mul(vScale)
			q3 := archsimd.LoadFloat32x8(q[qBase+24 : qBase+32]).Mul(vScale)
			q4 := archsimd.LoadFloat32x8(q[qBase+32 : qBase+40]).Mul(vScale)
			q5 := archsimd.LoadFloat32x8(q[qBase+40 : qBase+48]).Mul(vScale)
			q6 := archsimd.LoadFloat32x8(q[qBase+48 : qBase+56]).Mul(vScale)
			q7 := archsimd.LoadFloat32x8(q[qBase+56 : qBase+64]).Mul(vScale)

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
				k0 := archsimd.LoadFloat32x8(k[kBase : kBase+8])
				k1 := archsimd.LoadFloat32x8(k[kBase+8 : kBase+16])
				k2 := archsimd.LoadFloat32x8(k[kBase+16 : kBase+24])
				k3 := archsimd.LoadFloat32x8(k[kBase+24 : kBase+32])
				k4 := archsimd.LoadFloat32x8(k[kBase+32 : kBase+40])
				k5 := archsimd.LoadFloat32x8(k[kBase+40 : kBase+48])
				k6 := archsimd.LoadFloat32x8(k[kBase+48 : kBase+56])
				k7 := archsimd.LoadFloat32x8(k[kBase+56 : kBase+64])

				sumVec0 := q0.Mul(k0).Add(q1.Mul(k1)).Add(q2.Mul(k2)).Add(q3.Mul(k3))
				sumVec1 := q4.Mul(k4).Add(q5.Mul(k5)).Add(q6.Mul(k6)).Add(q7.Mul(k7))
				s := reduceSum8(sumVec0.Add(sumVec1))
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
				clear(output[outBase : outBase+64])
				continue
			}

			softmaxAndValueAccumDim64(v, output, outBase, kvOff, kvSeqStride, scoresScratch, activeKVLen, rowMax)
		}
	}
}

func sdpaFloat32AVX2Dim128(
	q, k, v, output []float32,
	qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskGroupStride int,
	additiveBias []float32,
	biasGroupStride int,
	scoresScratch []float32,
	groupSize, qStart, qEnd, kvLen int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) {
	// For Dim128 on AVX2, delegate to General (step 8).
	sdpaFloat32AVX2General(
		q, k, v, output,
		qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride,
		additiveMask, booleanMask, maskGroupStride,
		additiveBias, biasGroupStride,
		scoresScratch,
		groupSize, qStart, qEnd, kvLen, 128,
		scale, causal, qLimit, kvLimit,
	)
}

func sdpaFloat32AVX2General(
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
) {
	maskSeqStride, biasSeqStride := sdpaMaskBiasSeqStrides(additiveMask, booleanMask, maskGroupStride, additiveBias, biasGroupStride, kvLen)
	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		for qIdx := qStart; qIdx < qEnd; qIdx++ {
			outBase := gQOff + qIdx*qSeqStride
			if qIdx >= qLimit {
				clear(output[outBase : outBase+headDim])
				continue
			}

			kvLenUnmasked := kvLen
			if kvLimit < kvLenUnmasked {
				kvLenUnmasked = kvLimit
			}
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
				sumVec := archsimd.BroadcastFloat32x8(0)
				for ; d+8 <= headDim; d += 8 {
					qV := archsimd.LoadFloat32x8(q[qBase+d : qBase+d+8])
					kV := archsimd.LoadFloat32x8(k[kBase+d : kBase+d+8])
					sumVec = sumVec.Add(qV.Mul(kV))
				}
				var dot float32
				if d > 0 {
					dot = reduceSum8(sumVec)
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

			kvLenUnmasked = activeKVLen
			if kvLenUnmasked <= 0 || rowMax == float32(math.Inf(-1)) {
				clear(output[outBase : outBase+headDim])
				continue
			}

			// Softmax with padded tail
			padScoresTail8(scoresScratch, kvLenUnmasked)
			vRowMax := archsimd.BroadcastFloat32x8(rowMax)
			vSum := archsimd.BroadcastFloat32x8(0)
			for ki := 0; ki < kvLenUnmasked; ki += 8 {
				vScores := archsimd.LoadFloat32x8(scoresScratch[ki : ki+8])
				expV := exp256(vScores.Sub(vRowMax))
				expV.Store(scoresScratch[ki : ki+8])
				vSum = vSum.Add(expV)
			}
			sum := reduceSum8(vSum)

			var invSum float32
			if sum > 0 {
				invSum = 1.0 / sum
			}
			vInvSum := archsimd.BroadcastFloat32x8(invSum)

			// Value accumulation along d
			d := 0
			for ; d+8 <= headDim; d += 8 {
				outV0 := archsimd.BroadcastFloat32x8(0)
				outV1 := archsimd.BroadcastFloat32x8(0)
				kIdx := 0
				for ; kIdx+2 <= kvLenUnmasked; kIdx += 2 {
					vW0 := archsimd.BroadcastFloat32x8(scoresScratch[kIdx])
					vBase0 := kvOff + kIdx*kvSeqStride + d
					outV0 = archsimd.LoadFloat32x8(v[vBase0 : vBase0+8]).MulAdd(vW0, outV0)

					vW1 := archsimd.BroadcastFloat32x8(scoresScratch[kIdx+1])
					vBase1 := vBase0 + kvSeqStride
					outV1 = archsimd.LoadFloat32x8(v[vBase1 : vBase1+8]).MulAdd(vW1, outV1)
				}
				for ; kIdx < kvLenUnmasked; kIdx++ {
					vW := archsimd.BroadcastFloat32x8(scoresScratch[kIdx])
					vBase := kvOff + kIdx*kvSeqStride + d
					outV0 = archsimd.LoadFloat32x8(v[vBase : vBase+8]).MulAdd(vW, outV0)
				}
				outV0.Add(outV1).Mul(vInvSum).Store(output[outBase+d : outBase+d+8])
			}
			for ; d < headDim; d++ {
				var acc float32
				for kIdx := range kvLenUnmasked {
					acc += scoresScratch[kIdx] * v[kvOff+kIdx*kvSeqStride+d]
				}
				output[outBase+d] = acc * invSum
			}
		}
	}
}

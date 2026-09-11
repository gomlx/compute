// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"simd/archsimd"

	"github.com/gomlx/compute/internal/fastmath"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX2Allowed {
		gobackend.SetSDPAArchDispatcher(gobackend.PriorityArch, DispatchSDPAAVX2)
	}
}

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

// exp256 approximates e^x for 8 float32s using Cephes degree-7 Horner polynomial.
func exp256(x archsimd.Float32x8) archsimd.Float32x8 {
	const (
		maxLogF = 88.02969187150841
		minLogF = -88.02969187150841
		log2E   = 1.44269504088896341
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
	vHalf := archsimd.BroadcastFloat32x8(0.5)
	vLn2Hi := archsimd.BroadcastFloat32x8(ln2Hi)
	vLn2Lo := archsimd.BroadcastFloat32x8(ln2Lo)

	vP7 := archsimd.BroadcastFloat32x8(p7)
	vP6 := archsimd.BroadcastFloat32x8(p6)
	vP5 := archsimd.BroadcastFloat32x8(p5)
	vP4 := archsimd.BroadcastFloat32x8(p4)
	vP3 := archsimd.BroadcastFloat32x8(p3)
	vP2 := archsimd.BroadcastFloat32x8(p2)
	vOne := archsimd.BroadcastFloat32x8(1.0)
	v127 := archsimd.BroadcastUint32x8(127)

	xClamped := x.Max(vMinLog).Min(vMaxLog)
	z := xClamped.MulAdd(vLog2E, vHalf).Floor()
	g := xClamped.Sub(z.Mul(vLn2Hi)).Sub(z.Mul(vLn2Lo))

	n := z.ConvertToInt32().AsUint32x8().Add(v127).ShiftAllLeft(23).BitsToFloat32()

	poly := vP7.MulAdd(g, vP6)
	poly = poly.MulAdd(g, vP5)
	poly = poly.MulAdd(g, vP4)
	poly = poly.MulAdd(g, vP3)
	poly = poly.MulAdd(g, vP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(vOne)

	return n.Mul(poly)
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
	groupSize, seqLen, kvLen, headDim int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) bool {
	switch headDim {
	case 32:
		sdpaFloat32AVX2Dim32(
			q, k, v, output,
			qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride,
			additiveMask, booleanMask, maskGroupStride,
			additiveBias, biasGroupStride,
			scoresScratch,
			groupSize, seqLen, kvLen,
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
			groupSize, seqLen, kvLen,
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
			groupSize, seqLen, kvLen,
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
			groupSize, seqLen, kvLen, headDim,
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
	var sum float32
	k := 0
	vRowMax := archsimd.BroadcastFloat32x8(rowMax)
	vSum := archsimd.BroadcastFloat32x8(0)
	for ; k+8 <= kvLenUnmasked; k += 8 {
		vScores := archsimd.LoadFloat32x8(scoresScratch[k : k+8])
		expV := exp256(vScores.Sub(vRowMax))
		expV.Store(scoresScratch[k : k+8])
		vSum = vSum.Add(expV)
	}
	if k > 0 {
		sum += reduceSum8(vSum)
	}
	for ; k < kvLenUnmasked; k++ {
		s := scoresScratch[k]
		if s == float32(math.Inf(-1)) {
			scoresScratch[k] = 0
		} else {
			e := fastmath.Exp32(s - rowMax)
			scoresScratch[k] = e
			sum += e
		}
	}

	var invSum float32
	if sum > 0 {
		invSum = 1.0 / sum
	}

	k = 0
	vInvSum := archsimd.BroadcastFloat32x8(invSum)
	for ; k+8 <= kvLenUnmasked; k += 8 {
		vScores := archsimd.LoadFloat32x8(scoresScratch[k : k+8])
		vScores.Mul(vInvSum).Store(scoresScratch[k : k+8])
	}
	for ; k < kvLenUnmasked; k++ {
		scoresScratch[k] *= invSum
	}

	out0 := archsimd.BroadcastFloat32x8(0)
	out1 := archsimd.BroadcastFloat32x8(0)
	out2 := archsimd.BroadcastFloat32x8(0)
	out3 := archsimd.BroadcastFloat32x8(0)

	for ki := 0; ki < kvLenUnmasked; ki++ {
		w := scoresScratch[ki]
		if w == 0 {
			continue
		}
		vW := archsimd.BroadcastFloat32x8(w)
		vBase := kvOff + ki*kvSeqStride
		v0 := archsimd.LoadFloat32x8(v[vBase : vBase+8])
		v1 := archsimd.LoadFloat32x8(v[vBase+8 : vBase+16])
		v2 := archsimd.LoadFloat32x8(v[vBase+16 : vBase+24])
		v3 := archsimd.LoadFloat32x8(v[vBase+24 : vBase+32])
		out0 = vW.MulAdd(v0, out0)
		out1 = vW.MulAdd(v1, out1)
		out2 = vW.MulAdd(v2, out2)
		out3 = vW.MulAdd(v3, out3)
	}

	out0.Store(output[outBase : outBase+8])
	out1.Store(output[outBase+8 : outBase+16])
	out2.Store(output[outBase+16 : outBase+24])
	out3.Store(output[outBase+24 : outBase+32])
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
	groupSize, seqLen, kvLen int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) {
	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		for qIdx := range seqLen {
			outBase := gQOff + qIdx*qSeqStride
			if qIdx >= qLimit {
				for d := range 32 {
					output[outBase+d] = 0
				}
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
				for d := range 32 {
					output[outBase+d] = 0
				}
				continue
			}

			qBase := gQOff + qIdx*qSeqStride
			maskIdxBase := gMaskOff + qIdx*kvLen
			biasIdxBase := gBiasOff + qIdx*kvLen

			q0 := archsimd.LoadFloat32x8(q[qBase : qBase+8])
			q1 := archsimd.LoadFloat32x8(q[qBase+8 : qBase+16])
			q2 := archsimd.LoadFloat32x8(q[qBase+16 : qBase+24])
			q3 := archsimd.LoadFloat32x8(q[qBase+24 : qBase+32])

			rowMax := float32(math.Inf(-1))

			for ki := range kvLenUnmasked {
				maskIdx := maskIdxBase + ki
				if len(booleanMask) > 0 && !booleanMask[maskIdx] {
					scoresScratch[ki] = float32(math.Inf(-1))
					continue
				}

				kBase := kvOff + ki*kvSeqStride
				k0 := archsimd.LoadFloat32x8(k[kBase : kBase+8])
				k1 := archsimd.LoadFloat32x8(k[kBase+8 : kBase+16])
				k2 := archsimd.LoadFloat32x8(k[kBase+16 : kBase+24])
				k3 := archsimd.LoadFloat32x8(k[kBase+24 : kBase+32])

				sumVec := q0.Mul(k0).Add(q1.Mul(k1)).Add(q2.Mul(k2)).Add(q3.Mul(k3))
				dot := reduceSum8(sumVec)
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

			softmaxAndValueAccumDim32(v, output, outBase, kvOff, kvSeqStride, scoresScratch, kvLenUnmasked, rowMax)
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
	var sum float32
	k := 0
	vRowMax := archsimd.BroadcastFloat32x8(rowMax)
	vSum := archsimd.BroadcastFloat32x8(0)
	for ; k+8 <= kvLenUnmasked; k += 8 {
		vScores := archsimd.LoadFloat32x8(scoresScratch[k : k+8])
		expV := exp256(vScores.Sub(vRowMax))
		expV.Store(scoresScratch[k : k+8])
		vSum = vSum.Add(expV)
	}
	if k > 0 {
		sum += reduceSum8(vSum)
	}
	for ; k < kvLenUnmasked; k++ {
		s := scoresScratch[k]
		if s == float32(math.Inf(-1)) {
			scoresScratch[k] = 0
		} else {
			e := fastmath.Exp32(s - rowMax)
			scoresScratch[k] = e
			sum += e
		}
	}

	var invSum float32
	if sum > 0 {
		invSum = 1.0 / sum
	}

	k = 0
	vInvSum := archsimd.BroadcastFloat32x8(invSum)
	for ; k+8 <= kvLenUnmasked; k += 8 {
		vScores := archsimd.LoadFloat32x8(scoresScratch[k : k+8])
		vScores.Mul(vInvSum).Store(scoresScratch[k : k+8])
	}
	for ; k < kvLenUnmasked; k++ {
		scoresScratch[k] *= invSum
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
		w := scoresScratch[ki]
		if w == 0 {
			continue
		}
		vW := archsimd.BroadcastFloat32x8(w)
		vBase := kvOff + ki*kvSeqStride
		v0 := archsimd.LoadFloat32x8(v[vBase : vBase+8])
		v1 := archsimd.LoadFloat32x8(v[vBase+8 : vBase+16])
		v2 := archsimd.LoadFloat32x8(v[vBase+16 : vBase+24])
		v3 := archsimd.LoadFloat32x8(v[vBase+24 : vBase+32])
		v4 := archsimd.LoadFloat32x8(v[vBase+32 : vBase+40])
		v5 := archsimd.LoadFloat32x8(v[vBase+40 : vBase+48])
		v6 := archsimd.LoadFloat32x8(v[vBase+48 : vBase+56])
		v7 := archsimd.LoadFloat32x8(v[vBase+56 : vBase+64])
		out0 = vW.MulAdd(v0, out0)
		out1 = vW.MulAdd(v1, out1)
		out2 = vW.MulAdd(v2, out2)
		out3 = vW.MulAdd(v3, out3)
		out4 = vW.MulAdd(v4, out4)
		out5 = vW.MulAdd(v5, out5)
		out6 = vW.MulAdd(v6, out6)
		out7 = vW.MulAdd(v7, out7)
	}

	out0.Store(output[outBase : outBase+8])
	out1.Store(output[outBase+8 : outBase+16])
	out2.Store(output[outBase+16 : outBase+24])
	out3.Store(output[outBase+24 : outBase+32])
	out4.Store(output[outBase+32 : outBase+40])
	out5.Store(output[outBase+40 : outBase+48])
	out6.Store(output[outBase+48 : outBase+56])
	out7.Store(output[outBase+56 : outBase+64])
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
	groupSize, seqLen, kvLen int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) {
	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		for qIdx := range seqLen {
			outBase := gQOff + qIdx*qSeqStride
			if qIdx >= qLimit {
				for d := range 64 {
					output[outBase+d] = 0
				}
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
				for d := range 64 {
					output[outBase+d] = 0
				}
				continue
			}

			qBase := gQOff + qIdx*qSeqStride
			maskIdxBase := gMaskOff + qIdx*kvLen
			biasIdxBase := gBiasOff + qIdx*kvLen

			q0 := archsimd.LoadFloat32x8(q[qBase : qBase+8])
			q1 := archsimd.LoadFloat32x8(q[qBase+8 : qBase+16])
			q2 := archsimd.LoadFloat32x8(q[qBase+16 : qBase+24])
			q3 := archsimd.LoadFloat32x8(q[qBase+24 : qBase+32])
			q4 := archsimd.LoadFloat32x8(q[qBase+32 : qBase+40])
			q5 := archsimd.LoadFloat32x8(q[qBase+40 : qBase+48])
			q6 := archsimd.LoadFloat32x8(q[qBase+48 : qBase+56])
			q7 := archsimd.LoadFloat32x8(q[qBase+56 : qBase+64])

			rowMax := float32(math.Inf(-1))

			for ki := range kvLenUnmasked {
				maskIdx := maskIdxBase + ki
				if len(booleanMask) > 0 && !booleanMask[maskIdx] {
					scoresScratch[ki] = float32(math.Inf(-1))
					continue
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
				dot := reduceSum8(sumVec0.Add(sumVec1))
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

			softmaxAndValueAccumDim64(v, output, outBase, kvOff, kvSeqStride, scoresScratch, kvLenUnmasked, rowMax)
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
	groupSize, seqLen, kvLen int,
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
		groupSize, seqLen, kvLen, 128,
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
	groupSize, seqLen, kvLen, headDim int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) {
	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		gBiasOff := gIdx * biasGroupStride

		for qIdx := range seqLen {
			outBase := gQOff + qIdx*qSeqStride
			if qIdx >= qLimit {
				for d := range headDim {
					output[outBase+d] = 0
				}
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
				for d := range headDim {
					output[outBase+d] = 0
				}
				continue
			}

			qBase := gQOff + qIdx*qSeqStride
			maskIdxBase := gMaskOff + qIdx*kvLen
			biasIdxBase := gBiasOff + qIdx*kvLen

			rowMax := float32(math.Inf(-1))

			for ki := range kvLenUnmasked {
				maskIdx := maskIdxBase + ki
				if len(booleanMask) > 0 && !booleanMask[maskIdx] {
					scoresScratch[ki] = float32(math.Inf(-1))
					continue
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

			// Softmax
			var sum float32
			ki := 0
			vRowMax := archsimd.BroadcastFloat32x8(rowMax)
			vSum := archsimd.BroadcastFloat32x8(0)
			for ; ki+8 <= kvLenUnmasked; ki += 8 {
				vScores := archsimd.LoadFloat32x8(scoresScratch[ki : ki+8])
				expV := exp256(vScores.Sub(vRowMax))
				expV.Store(scoresScratch[ki : ki+8])
				vSum = vSum.Add(expV)
			}
			if ki > 0 {
				sum += reduceSum8(vSum)
			}
			for ; ki < kvLenUnmasked; ki++ {
				s := scoresScratch[ki]
				if s == float32(math.Inf(-1)) {
					scoresScratch[ki] = 0
				} else {
					e := fastmath.Exp32(s - rowMax)
					scoresScratch[ki] = e
					sum += e
				}
			}

			var invSum float32
			if sum > 0 {
				invSum = 1.0 / sum
			}

			ki = 0
			vInvSum := archsimd.BroadcastFloat32x8(invSum)
			for ; ki+8 <= kvLenUnmasked; ki += 8 {
				vScores := archsimd.LoadFloat32x8(scoresScratch[ki : ki+8])
				vScores.Mul(vInvSum).Store(scoresScratch[ki : ki+8])
			}
			for ; ki < kvLenUnmasked; ki++ {
				scoresScratch[ki] *= invSum
			}

			// Value accumulation along d
			d := 0
			for ; d+8 <= headDim; d += 8 {
				outV := archsimd.BroadcastFloat32x8(0)
				for kIdx := range kvLenUnmasked {
					w := scoresScratch[kIdx]
					if w == 0 {
						continue
					}
					vW := archsimd.BroadcastFloat32x8(w)
					vBase := kvOff + kIdx*kvSeqStride + d
					vVal := archsimd.LoadFloat32x8(v[vBase : vBase+8])
					outV = vW.MulAdd(vVal, outV)
				}
				outV.Store(output[outBase+d : outBase+d+8])
			}
			for ; d < headDim; d++ {
				var acc float32
				for kIdx := range kvLenUnmasked {
					w := scoresScratch[kIdx]
					if w != 0 {
						acc += w * v[kvOff+kIdx*kvSeqStride+d]
					}
				}
				output[outBase+d] = acc
			}
		}
	}
}

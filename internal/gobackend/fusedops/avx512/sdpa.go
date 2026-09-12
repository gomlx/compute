// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"math"
	"simd/archsimd"

	"github.com/gomlx/compute/internal/fastmath"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX512Allowed {
		gobackend.SetSDPAArchDispatcher(gobackend.PriorityArch+1, DispatchSDPAAVX512)
	}
}

// reduceSum8 reduces a Float32x8 vector to a single float32 sum.
func reduceSum8(v archsimd.Float32x8) float32 {
	v4 := v.GetLo().Add(v.GetHi())
	v2 := v4.ConcatAddPairs(v4)
	return v2.GetElem(0) + v2.GetElem(1)
}

// reduceSum16 reduces a Float32x16 vector to a single float32 sum.
func reduceSum16(v archsimd.Float32x16) float32 {
	return reduceSum8(v.GetLo().Add(v.GetHi()))
}

// reduceMax8 reduces a Float32x8 vector to its maximum element.
func reduceMax8(v archsimd.Float32x8) float32 {
	v4 := v.GetLo().Max(v.GetHi())
	var arr [4]float32
	v4.StoreArray(&arr)
	m := max(arr[0], arr[1])
	return max(m, max(arr[2], arr[3]))
}

// reduceMax16 reduces a Float32x16 vector to its maximum element.
func reduceMax16(v archsimd.Float32x16) float32 {
	return reduceMax8(v.GetLo().Max(v.GetHi()))
}

// exp512 approximates e^x for 16 float32s using Cephes degree-7 Horner polynomial.
func exp512(x archsimd.Float32x16) archsimd.Float32x16 {
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
	vMaxLog := archsimd.BroadcastFloat32x16(maxLogF)
	vMinLog := archsimd.BroadcastFloat32x16(minLogF)
	vLog2E := archsimd.BroadcastFloat32x16(log2E)
	vHalf := archsimd.BroadcastFloat32x16(0.5)
	vLn2Hi := archsimd.BroadcastFloat32x16(ln2Hi)
	vLn2Lo := archsimd.BroadcastFloat32x16(ln2Lo)

	vP7 := archsimd.BroadcastFloat32x16(p7)
	vP6 := archsimd.BroadcastFloat32x16(p6)
	vP5 := archsimd.BroadcastFloat32x16(p5)
	vP4 := archsimd.BroadcastFloat32x16(p4)
	vP3 := archsimd.BroadcastFloat32x16(p3)
	vP2 := archsimd.BroadcastFloat32x16(p2)
	vOne := archsimd.BroadcastFloat32x16(1.0)
	v127 := archsimd.BroadcastUint32x16(127)

	xClamped := x.Max(vMinLog).Min(vMaxLog)
	z := xClamped.MulAdd(vLog2E, vHalf).RoundScaled(0)
	g := xClamped.Sub(z.Mul(vLn2Hi)).Sub(z.Mul(vLn2Lo))

	n := z.ConvertToInt32().AsUint32x16().Add(v127).ShiftAllLeft(23).BitsToFloat32()

	poly := vP7.MulAdd(g, vP6)
	poly = poly.MulAdd(g, vP5)
	poly = poly.MulAdd(g, vP4)
	poly = poly.MulAdd(g, vP3)
	poly = poly.MulAdd(g, vP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(vOne)

	return n.Mul(poly)
}

// DispatchSDPAAVX512 executes scaled dot-product attention using AVX-512.
func DispatchSDPAAVX512(
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
		sdpaFloat32AVX512Dim32(
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
		sdpaFloat32AVX512Dim64(
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
		sdpaFloat32AVX512Dim128(
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
		sdpaFloat32AVX512General(
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
	// Softmax: exp(scores - rowMax) and sum
	var sum float32
	k := 0
	vRowMax := archsimd.BroadcastFloat32x16(rowMax)
	vSum := archsimd.BroadcastFloat32x16(0)
	for ; k+16 <= kvLenUnmasked; k += 16 {
		vScores := archsimd.LoadFloat32x16(scoresScratch[k : k+16])
		expV := exp512(vScores.Sub(vRowMax))
		expV.Store(scoresScratch[k : k+16])
		vSum = vSum.Add(expV)
	}
	if k > 0 {
		sum += reduceSum16(vSum)
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

	// Normalize scores
	k = 0
	vInvSum := archsimd.BroadcastFloat32x16(invSum)
	for ; k+16 <= kvLenUnmasked; k += 16 {
		vScores := archsimd.LoadFloat32x16(scoresScratch[k : k+16])
		vScores.Mul(vInvSum).Store(scoresScratch[k : k+16])
	}
	for ; k < kvLenUnmasked; k++ {
		scoresScratch[k] *= invSum
	}

	// Value accumulation into vector registers
	out0 := archsimd.BroadcastFloat32x16(0)
	out1 := archsimd.BroadcastFloat32x16(0)

	for ki := 0; ki < kvLenUnmasked; ki++ {
		w := scoresScratch[ki]
		if w == 0 {
			continue
		}
		vW := archsimd.BroadcastFloat32x16(w)
		vBase := kvOff + ki*kvSeqStride
		v0 := archsimd.LoadFloat32x16(v[vBase : vBase+16])
		v1 := archsimd.LoadFloat32x16(v[vBase+16 : vBase+32])
		out0 = vW.MulAdd(v0, out0)
		out1 = vW.MulAdd(v1, out1)
	}

	out0.Store(output[outBase : outBase+16])
	out1.Store(output[outBase+16 : outBase+32])
}

func sdpaFloat32AVX512Dim32(
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

			q0 := archsimd.LoadFloat32x16(q[qBase : qBase+16])
			q1 := archsimd.LoadFloat32x16(q[qBase+16 : qBase+32])

			rowMax := float32(math.Inf(-1))

			for ki := range kvLenUnmasked {
				maskIdx := maskIdxBase + ki
				if len(booleanMask) > 0 && !booleanMask[maskIdx] {
					scoresScratch[ki] = float32(math.Inf(-1))
					continue
				}

				kBase := kvOff + ki*kvSeqStride
				k0 := archsimd.LoadFloat32x16(k[kBase : kBase+16])
				k1 := archsimd.LoadFloat32x16(k[kBase+16 : kBase+32])

				sumVec := q0.Mul(k0).Add(q1.Mul(k1))
				dot := reduceSum16(sumVec)
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
	vRowMax := archsimd.BroadcastFloat32x16(rowMax)
	vSum := archsimd.BroadcastFloat32x16(0)
	for ; k+16 <= kvLenUnmasked; k += 16 {
		vScores := archsimd.LoadFloat32x16(scoresScratch[k : k+16])
		expV := exp512(vScores.Sub(vRowMax))
		expV.Store(scoresScratch[k : k+16])
		vSum = vSum.Add(expV)
	}
	if k > 0 {
		sum += reduceSum16(vSum)
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
	vInvSum := archsimd.BroadcastFloat32x16(invSum)
	for ; k+16 <= kvLenUnmasked; k += 16 {
		vScores := archsimd.LoadFloat32x16(scoresScratch[k : k+16])
		vScores.Mul(vInvSum).Store(scoresScratch[k : k+16])
	}
	for ; k < kvLenUnmasked; k++ {
		scoresScratch[k] *= invSum
	}

	out0 := archsimd.BroadcastFloat32x16(0)
	out1 := archsimd.BroadcastFloat32x16(0)
	out2 := archsimd.BroadcastFloat32x16(0)
	out3 := archsimd.BroadcastFloat32x16(0)

	for ki := 0; ki < kvLenUnmasked; ki++ {
		w := scoresScratch[ki]
		if w == 0 {
			continue
		}
		vW := archsimd.BroadcastFloat32x16(w)
		vBase := kvOff + ki*kvSeqStride
		v0 := archsimd.LoadFloat32x16(v[vBase : vBase+16])
		v1 := archsimd.LoadFloat32x16(v[vBase+16 : vBase+32])
		v2 := archsimd.LoadFloat32x16(v[vBase+32 : vBase+48])
		v3 := archsimd.LoadFloat32x16(v[vBase+48 : vBase+64])
		out0 = vW.MulAdd(v0, out0)
		out1 = vW.MulAdd(v1, out1)
		out2 = vW.MulAdd(v2, out2)
		out3 = vW.MulAdd(v3, out3)
	}

	out0.Store(output[outBase : outBase+16])
	out1.Store(output[outBase+16 : outBase+32])
	out2.Store(output[outBase+32 : outBase+48])
	out3.Store(output[outBase+48 : outBase+64])
}

func sdpaFloat32AVX512Dim64(
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

			q0 := archsimd.LoadFloat32x16(q[qBase : qBase+16])
			q1 := archsimd.LoadFloat32x16(q[qBase+16 : qBase+32])
			q2 := archsimd.LoadFloat32x16(q[qBase+32 : qBase+48])
			q3 := archsimd.LoadFloat32x16(q[qBase+48 : qBase+64])

			rowMax := float32(math.Inf(-1))

			for ki := range kvLenUnmasked {
				maskIdx := maskIdxBase + ki
				if len(booleanMask) > 0 && !booleanMask[maskIdx] {
					scoresScratch[ki] = float32(math.Inf(-1))
					continue
				}

				kBase := kvOff + ki*kvSeqStride
				k0 := archsimd.LoadFloat32x16(k[kBase : kBase+16])
				k1 := archsimd.LoadFloat32x16(k[kBase+16 : kBase+32])
				k2 := archsimd.LoadFloat32x16(k[kBase+32 : kBase+48])
				k3 := archsimd.LoadFloat32x16(k[kBase+48 : kBase+64])

				sumVec := q0.Mul(k0).Add(q1.Mul(k1)).Add(q2.Mul(k2)).Add(q3.Mul(k3))
				dot := reduceSum16(sumVec)
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

func softmaxAndValueAccumDim128(
	v, output []float32,
	outBase, kvOff, kvSeqStride int,
	scoresScratch []float32,
	kvLenUnmasked int,
	rowMax float32,
) {
	var sum float32
	k := 0
	vRowMax := archsimd.BroadcastFloat32x16(rowMax)
	vSum := archsimd.BroadcastFloat32x16(0)
	for ; k+16 <= kvLenUnmasked; k += 16 {
		vScores := archsimd.LoadFloat32x16(scoresScratch[k : k+16])
		expV := exp512(vScores.Sub(vRowMax))
		expV.Store(scoresScratch[k : k+16])
		vSum = vSum.Add(expV)
	}
	if k > 0 {
		sum += reduceSum16(vSum)
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
	vInvSum := archsimd.BroadcastFloat32x16(invSum)
	for ; k+16 <= kvLenUnmasked; k += 16 {
		vScores := archsimd.LoadFloat32x16(scoresScratch[k : k+16])
		vScores.Mul(vInvSum).Store(scoresScratch[k : k+16])
	}
	for ; k < kvLenUnmasked; k++ {
		scoresScratch[k] *= invSum
	}

	out0 := archsimd.BroadcastFloat32x16(0)
	out1 := archsimd.BroadcastFloat32x16(0)
	out2 := archsimd.BroadcastFloat32x16(0)
	out3 := archsimd.BroadcastFloat32x16(0)
	out4 := archsimd.BroadcastFloat32x16(0)
	out5 := archsimd.BroadcastFloat32x16(0)
	out6 := archsimd.BroadcastFloat32x16(0)
	out7 := archsimd.BroadcastFloat32x16(0)

	for ki := 0; ki < kvLenUnmasked; ki++ {
		w := scoresScratch[ki]
		if w == 0 {
			continue
		}
		vW := archsimd.BroadcastFloat32x16(w)
		vBase := kvOff + ki*kvSeqStride
		v0 := archsimd.LoadFloat32x16(v[vBase : vBase+16])
		v1 := archsimd.LoadFloat32x16(v[vBase+16 : vBase+32])
		v2 := archsimd.LoadFloat32x16(v[vBase+32 : vBase+48])
		v3 := archsimd.LoadFloat32x16(v[vBase+48 : vBase+64])
		v4 := archsimd.LoadFloat32x16(v[vBase+64 : vBase+80])
		v5 := archsimd.LoadFloat32x16(v[vBase+80 : vBase+96])
		v6 := archsimd.LoadFloat32x16(v[vBase+96 : vBase+112])
		v7 := archsimd.LoadFloat32x16(v[vBase+112 : vBase+128])
		out0 = vW.MulAdd(v0, out0)
		out1 = vW.MulAdd(v1, out1)
		out2 = vW.MulAdd(v2, out2)
		out3 = vW.MulAdd(v3, out3)
		out4 = vW.MulAdd(v4, out4)
		out5 = vW.MulAdd(v5, out5)
		out6 = vW.MulAdd(v6, out6)
		out7 = vW.MulAdd(v7, out7)
	}

	out0.Store(output[outBase : outBase+16])
	out1.Store(output[outBase+16 : outBase+32])
	out2.Store(output[outBase+32 : outBase+48])
	out3.Store(output[outBase+48 : outBase+64])
	out4.Store(output[outBase+64 : outBase+80])
	out5.Store(output[outBase+80 : outBase+96])
	out6.Store(output[outBase+96 : outBase+112])
	out7.Store(output[outBase+112 : outBase+128])
}

func sdpaFloat32AVX512Dim128(
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
				for d := range 128 {
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
				for d := range 128 {
					output[outBase+d] = 0
				}
				continue
			}

			qBase := gQOff + qIdx*qSeqStride
			maskIdxBase := gMaskOff + qIdx*kvLen
			biasIdxBase := gBiasOff + qIdx*kvLen

			q0 := archsimd.LoadFloat32x16(q[qBase : qBase+16])
			q1 := archsimd.LoadFloat32x16(q[qBase+16 : qBase+32])
			q2 := archsimd.LoadFloat32x16(q[qBase+32 : qBase+48])
			q3 := archsimd.LoadFloat32x16(q[qBase+48 : qBase+64])
			q4 := archsimd.LoadFloat32x16(q[qBase+64 : qBase+80])
			q5 := archsimd.LoadFloat32x16(q[qBase+80 : qBase+96])
			q6 := archsimd.LoadFloat32x16(q[qBase+96 : qBase+112])
			q7 := archsimd.LoadFloat32x16(q[qBase+112 : qBase+128])

			rowMax := float32(math.Inf(-1))

			for ki := range kvLenUnmasked {
				maskIdx := maskIdxBase + ki
				if len(booleanMask) > 0 && !booleanMask[maskIdx] {
					scoresScratch[ki] = float32(math.Inf(-1))
					continue
				}

				kBase := kvOff + ki*kvSeqStride
				k0 := archsimd.LoadFloat32x16(k[kBase : kBase+16])
				k1 := archsimd.LoadFloat32x16(k[kBase+16 : kBase+32])
				k2 := archsimd.LoadFloat32x16(k[kBase+32 : kBase+48])
				k3 := archsimd.LoadFloat32x16(k[kBase+48 : kBase+64])
				k4 := archsimd.LoadFloat32x16(k[kBase+64 : kBase+80])
				k5 := archsimd.LoadFloat32x16(k[kBase+80 : kBase+96])
				k6 := archsimd.LoadFloat32x16(k[kBase+96 : kBase+112])
				k7 := archsimd.LoadFloat32x16(k[kBase+112 : kBase+128])

				sumVec0 := q0.Mul(k0).Add(q1.Mul(k1)).Add(q2.Mul(k2)).Add(q3.Mul(k3))
				sumVec1 := q4.Mul(k4).Add(q5.Mul(k5)).Add(q6.Mul(k6)).Add(q7.Mul(k7))
				dot := reduceSum16(sumVec0.Add(sumVec1))
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

			softmaxAndValueAccumDim128(v, output, outBase, kvOff, kvSeqStride, scoresScratch, kvLenUnmasked, rowMax)
		}
	}
}

func sdpaFloat32AVX512General(
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
				sumVec := archsimd.BroadcastFloat32x16(0)
				for ; d+16 <= headDim; d += 16 {
					qV := archsimd.LoadFloat32x16(q[qBase+d : qBase+d+16])
					kV := archsimd.LoadFloat32x16(k[kBase+d : kBase+d+16])
					sumVec = sumVec.Add(qV.Mul(kV))
				}
				var dot float32
				if d > 0 {
					dot = reduceSum16(sumVec)
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
			vRowMax := archsimd.BroadcastFloat32x16(rowMax)
			vSum := archsimd.BroadcastFloat32x16(0)
			for ; ki+16 <= kvLenUnmasked; ki += 16 {
				vScores := archsimd.LoadFloat32x16(scoresScratch[ki : ki+16])
				expV := exp512(vScores.Sub(vRowMax))
				expV.Store(scoresScratch[ki : ki+16])
				vSum = vSum.Add(expV)
			}
			if ki > 0 {
				sum += reduceSum16(vSum)
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
			vInvSum := archsimd.BroadcastFloat32x16(invSum)
			for ; ki+16 <= kvLenUnmasked; ki += 16 {
				vScores := archsimd.LoadFloat32x16(scoresScratch[ki : ki+16])
				vScores.Mul(vInvSum).Store(scoresScratch[ki : ki+16])
			}
			for ; ki < kvLenUnmasked; ki++ {
				scoresScratch[ki] *= invSum
			}

			// Value accumulation along d
			d := 0
			for ; d+16 <= headDim; d += 16 {
				outV := archsimd.BroadcastFloat32x16(0)
				for kIdx := range kvLenUnmasked {
					w := scoresScratch[kIdx]
					if w == 0 {
						continue
					}
					vW := archsimd.BroadcastFloat32x16(w)
					vBase := kvOff + kIdx*kvSeqStride + d
					vVal := archsimd.LoadFloat32x16(v[vBase : vBase+16])
					outV = vW.MulAdd(vVal, outV)
				}
				outV.Store(output[outBase+d : outBase+d+16])
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

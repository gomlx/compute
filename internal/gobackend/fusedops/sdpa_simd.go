// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package fusedops

import (
	"math"
	"simd"

	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	gobackend.SetSDPAArchDispatcher(gobackend.PrioritySIMD, DispatchSDPASIMD)
}

func reduceSumFloat32s(v simd.Float32s) float32 {
	var buf [16]float32
	n := v.Len()
	v.Store(buf[:n])
	var sum float32
	for i := range n {
		sum += buf[i]
	}
	return sum
}

// DispatchSDPASIMD executes scaled dot-product attention using portable Go SIMD.
func DispatchSDPASIMD(
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
	sdpaFloat32SIMDGeneral(
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

func sdpaSoftmaxRowSIMD(scores []float32, kvLenUnmasked int, rowMax float32) {
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

	vZero := simd.BroadcastFloat32s(0)
	vLen := vZero.Len()
	vMaxLog := simd.BroadcastFloat32s(maxLogF)
	vMinLog := simd.BroadcastFloat32s(minLogF)
	vLog2E := simd.BroadcastFloat32s(log2E)
	vMagic := simd.BroadcastFloat32s(magicF)
	vLn2Hi := simd.BroadcastFloat32s(ln2Hi)
	vLn2Lo := simd.BroadcastFloat32s(ln2Lo)

	vP7 := simd.BroadcastFloat32s(p7)
	vP6 := simd.BroadcastFloat32s(p6)
	vP5 := simd.BroadcastFloat32s(p5)
	vP4 := simd.BroadcastFloat32s(p4)
	vP3 := simd.BroadcastFloat32s(p3)
	vP2 := simd.BroadcastFloat32s(p2)
	vOne := simd.BroadcastFloat32s(1.0)
	vShift23 := simd.BroadcastUint32s(1 << 23)
	vBias127 := simd.BroadcastUint32s(127 << 23)

	vRowMax := simd.BroadcastFloat32s(rowMax)
	vSum := vZero

	k := 0
	for ; k+vLen <= kvLenUnmasked; k += vLen {
		x := simd.LoadFloat32s(scores[k:]).Sub(vRowMax)
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
		expV.Store(scores[k:])
		vSum = vSum.Add(expV)
	}

	var sum float32
	if k > 0 {
		sum = reduceSumFloat32s(vSum)
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
	vInvSum := simd.BroadcastFloat32s(invSum)
	k = 0
	for ; k+4*vLen <= kvLenUnmasked; k += 4 * vLen {
		s0 := simd.LoadFloat32s(scores[k:]).Mul(vInvSum)
		s1 := simd.LoadFloat32s(scores[k+vLen:]).Mul(vInvSum)
		s2 := simd.LoadFloat32s(scores[k+2*vLen:]).Mul(vInvSum)
		s3 := simd.LoadFloat32s(scores[k+3*vLen:]).Mul(vInvSum)
		s0.Store(scores[k:])
		s1.Store(scores[k+vLen:])
		s2.Store(scores[k+2*vLen:])
		s3.Store(scores[k+3*vLen:])
	}
	for ; k+vLen <= kvLenUnmasked; k += vLen {
		simd.LoadFloat32s(scores[k:]).Mul(vInvSum).Store(scores[k:])
	}
	for ; k < kvLenUnmasked; k++ {
		scores[k] *= invSum
	}
}

func sdpaFloat32SIMDGeneral(
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
	vZero := simd.BroadcastFloat32s(0)
	vLen := vZero.Len()

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
			if qIdx >= qLimit {
				clear(output[outBase : outBase+headDim])
				continue
			}

			kvLenUnmasked := min(kvLen, kvLimit)
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
				s0 := vZero
				s1 := vZero
				s2 := vZero
				s3 := vZero
				for ; d+4*vLen <= headDim; d += 4 * vLen {
					q0 := simd.LoadFloat32s(q[qBase+d:])
					q1 := simd.LoadFloat32s(q[qBase+d+vLen:])
					q2 := simd.LoadFloat32s(q[qBase+d+2*vLen:])
					q3 := simd.LoadFloat32s(q[qBase+d+3*vLen:])
					k0 := simd.LoadFloat32s(k[kBase+d:])
					k1 := simd.LoadFloat32s(k[kBase+d+vLen:])
					k2 := simd.LoadFloat32s(k[kBase+d+2*vLen:])
					k3 := simd.LoadFloat32s(k[kBase+d+3*vLen:])
					s0 = q0.MulAdd(k0, s0)
					s1 = q1.MulAdd(k1, s1)
					s2 = q2.MulAdd(k2, s2)
					s3 = q3.MulAdd(k3, s3)
				}
				sumVec := s0.Add(s1).Add(s2.Add(s3))
				for ; d+vLen <= headDim; d += vLen {
					qV := simd.LoadFloat32s(q[qBase+d:])
					kV := simd.LoadFloat32s(k[kBase+d:])
					sumVec = qV.MulAdd(kV, sumVec)
				}
				var dot float32
				if d > 0 {
					dot = reduceSumFloat32s(sumVec)
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

			sdpaSoftmaxRowSIMD(scoresScratch, kvLenUnmasked, rowMax)

			// Value accumulation along d in blocks of 8*vLen registers.
			d := 0
			for ; d+8*vLen <= headDim; d += 8 * vLen {
				o0 := vZero
				o1 := vZero
				o2 := vZero
				o3 := vZero
				o4 := vZero
				o5 := vZero
				o6 := vZero
				o7 := vZero
				for kIdx := range kvLenUnmasked {
					w := scoresScratch[kIdx]
					if w == 0 {
						continue
					}
					vW := simd.BroadcastFloat32s(w)
					vBase := kvOff + kIdx*kvSeqStride + d
					o0 = vW.MulAdd(simd.LoadFloat32s(v[vBase:]), o0)
					o1 = vW.MulAdd(simd.LoadFloat32s(v[vBase+vLen:]), o1)
					o2 = vW.MulAdd(simd.LoadFloat32s(v[vBase+2*vLen:]), o2)
					o3 = vW.MulAdd(simd.LoadFloat32s(v[vBase+3*vLen:]), o3)
					o4 = vW.MulAdd(simd.LoadFloat32s(v[vBase+4*vLen:]), o4)
					o5 = vW.MulAdd(simd.LoadFloat32s(v[vBase+5*vLen:]), o5)
					o6 = vW.MulAdd(simd.LoadFloat32s(v[vBase+6*vLen:]), o6)
					o7 = vW.MulAdd(simd.LoadFloat32s(v[vBase+7*vLen:]), o7)
				}
				o0.Store(output[outBase+d:])
				o1.Store(output[outBase+d+vLen:])
				o2.Store(output[outBase+d+2*vLen:])
				o3.Store(output[outBase+d+3*vLen:])
				o4.Store(output[outBase+d+4*vLen:])
				o5.Store(output[outBase+d+5*vLen:])
				o6.Store(output[outBase+d+6*vLen:])
				o7.Store(output[outBase+d+7*vLen:])
			}
			for ; d+vLen <= headDim; d += vLen {
				outV := vZero
				for kIdx := range kvLenUnmasked {
					w := scoresScratch[kIdx]
					if w == 0 {
						continue
					}
					vW := simd.BroadcastFloat32s(w)
					vBase := kvOff + kIdx*kvSeqStride + d
					vVal := simd.LoadFloat32s(v[vBase:])
					outV = vW.MulAdd(vVal, outV)
				}
				outV.Store(output[outBase+d:])
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

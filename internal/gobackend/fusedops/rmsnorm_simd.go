// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd && go1.27

package fusedops

import (
	"math"
	"simd"
)

func init() {
	rmsNormRowFloat32Fn = rmsNormRowFloat32SIMD
	rmsNormRowFloat64Fn = rmsNormRowFloat64SIMD
}

func rmsNormRowFloat32SIMD(in, out, scale []float32, epsilon float64) {
	n := len(in)
	vLen := simd.BroadcastFloat32s(0).Len()
	if n < vLen {
		rmsNormRowFloat32Scalar(in, out, scale, epsilon)
		return
	}

	// Pass 1: 4x unrolled sum of squares in SIMD registers.
	acc0 := simd.BroadcastFloat32s(0)
	acc1 := simd.BroadcastFloat32s(0)
	acc2 := simd.BroadcastFloat32s(0)
	acc3 := simd.BroadcastFloat32s(0)
	i := 0
	step4 := 4 * vLen
	for ; i+step4 <= n; i += step4 {
		v0 := simd.LoadFloat32s(in[i:])
		v1 := simd.LoadFloat32s(in[i+vLen:])
		v2 := simd.LoadFloat32s(in[i+2*vLen:])
		v3 := simd.LoadFloat32s(in[i+3*vLen:])
		acc0 = v0.MulAdd(v0, acc0)
		acc1 = v1.MulAdd(v1, acc1)
		acc2 = v2.MulAdd(v2, acc2)
		acc3 = v3.MulAdd(v3, acc3)
	}
	acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
	for ; i+vLen <= n; i += vLen {
		v := simd.LoadFloat32s(in[i:])
		acc0 = v.MulAdd(v, acc0)
	}
	var tmp [64]float32
	acc0.Store(tmp[:vLen])
	var sumSq float32
	for lane := range vLen {
		sumSq += tmp[lane]
	}
	for ; i < n; i++ {
		v := in[i]
		sumSq += v * v
	}

	invRMS := float32(1.0 / math.Sqrt(float64(sumSq)/float64(n)+epsilon))
	vInvRMS := simd.BroadcastFloat32s(invRMS)

	// Pass 2: normalize and apply optional scale in SIMD.
	i = 0
	if scale != nil {
		for ; i+step4 <= n; i += step4 {
			v0 := simd.LoadFloat32s(in[i:]).Mul(vInvRMS).Mul(simd.LoadFloat32s(scale[i:]))
			v1 := simd.LoadFloat32s(in[i+vLen:]).Mul(vInvRMS).Mul(simd.LoadFloat32s(scale[i+vLen:]))
			v2 := simd.LoadFloat32s(in[i+2*vLen:]).Mul(vInvRMS).Mul(simd.LoadFloat32s(scale[i+2*vLen:]))
			v3 := simd.LoadFloat32s(in[i+3*vLen:]).Mul(vInvRMS).Mul(simd.LoadFloat32s(scale[i+3*vLen:]))
			v0.Store(out[i:])
			v1.Store(out[i+vLen:])
			v2.Store(out[i+2*vLen:])
			v3.Store(out[i+3*vLen:])
		}
		for ; i+vLen <= n; i += vLen {
			v := simd.LoadFloat32s(in[i:]).Mul(vInvRMS).Mul(simd.LoadFloat32s(scale[i:]))
			v.Store(out[i:])
		}
		for ; i < n; i++ {
			out[i] = in[i] * invRMS * scale[i]
		}
	} else {
		for ; i+step4 <= n; i += step4 {
			v0 := simd.LoadFloat32s(in[i:]).Mul(vInvRMS)
			v1 := simd.LoadFloat32s(in[i+vLen:]).Mul(vInvRMS)
			v2 := simd.LoadFloat32s(in[i+2*vLen:]).Mul(vInvRMS)
			v3 := simd.LoadFloat32s(in[i+3*vLen:]).Mul(vInvRMS)
			v0.Store(out[i:])
			v1.Store(out[i+vLen:])
			v2.Store(out[i+2*vLen:])
			v3.Store(out[i+3*vLen:])
		}
		for ; i+vLen <= n; i += vLen {
			v := simd.LoadFloat32s(in[i:]).Mul(vInvRMS)
			v.Store(out[i:])
		}
		for ; i < n; i++ {
			out[i] = in[i] * invRMS
		}
	}
}

func rmsNormRowFloat64SIMD(in, out, scale []float64, epsilon float64) {
	n := len(in)
	vLen := simd.BroadcastFloat64s(0).Len()
	if n < vLen {
		rmsNormRowFloat64Scalar(in, out, scale, epsilon)
		return
	}

	acc0 := simd.BroadcastFloat64s(0)
	acc1 := simd.BroadcastFloat64s(0)
	i := 0
	step2 := 2 * vLen
	for ; i+step2 <= n; i += step2 {
		v0 := simd.LoadFloat64s(in[i:])
		v1 := simd.LoadFloat64s(in[i+vLen:])
		acc0 = v0.MulAdd(v0, acc0)
		acc1 = v1.MulAdd(v1, acc1)
	}
	acc0 = acc0.Add(acc1)
	for ; i+vLen <= n; i += vLen {
		v := simd.LoadFloat64s(in[i:])
		acc0 = v.MulAdd(v, acc0)
	}
	var tmp [32]float64
	acc0.Store(tmp[:vLen])
	var sumSq float64
	for lane := range vLen {
		sumSq += tmp[lane]
	}
	for ; i < n; i++ {
		v := in[i]
		sumSq += v * v
	}

	invRMS := 1.0 / math.Sqrt(sumSq/float64(n)+epsilon)
	vInvRMS := simd.BroadcastFloat64s(invRMS)

	i = 0
	if scale != nil {
		for ; i+vLen <= n; i += vLen {
			v := simd.LoadFloat64s(in[i:]).Mul(vInvRMS).Mul(simd.LoadFloat64s(scale[i:]))
			v.Store(out[i:])
		}
		for ; i < n; i++ {
			out[i] = in[i] * invRMS * scale[i]
		}
	} else {
		for ; i+vLen <= n; i += vLen {
			v := simd.LoadFloat64s(in[i:]).Mul(vInvRMS)
			v.Store(out[i:])
		}
		for ; i < n; i++ {
			out[i] = in[i] * invRMS
		}
	}
}

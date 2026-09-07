// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package activations

import (
	"simd"

	"github.com/gomlx/compute/dtypes/bfloat16" //alt:bfloat16
	//alt:float16 "github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/support/simdmath"
)


// -------------------------------------------------------------------------------------------------
// Forward Activations
// -------------------------------------------------------------------------------------------------

func ReluBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func ReluFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = even.Max(vZero)
		odd = odd.Max(vZero)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = even.Max(vZero)
		odd = odd.Max(vZero)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func SigmoidBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func SigmoidFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = simdmath.SigmoidFloat32(even)
		odd = simdmath.SigmoidFloat32(odd)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = simdmath.SigmoidFloat32(even)
		odd = simdmath.SigmoidFloat32(odd)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func HardSigmoidBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func HardSigmoidFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vOne := simd.BroadcastFloat32s(1)
	vPointTwo := simd.BroadcastFloat32s(0.2)
	vHalf := simd.BroadcastFloat32s(0.5)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = even.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne)
		odd = odd.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = even.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne)
		odd = odd.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func LeakyReluBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func LeakyReluFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vAlpha := simd.BroadcastFloat32s(0.3)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = selectFloat32s(even.GreaterEqual(vZero), even, even.Mul(vAlpha))
		odd = selectFloat32s(odd.GreaterEqual(vZero), odd, odd.Mul(vAlpha))
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = selectFloat32s(even.GreaterEqual(vZero), even, even.Mul(vAlpha))
		odd = selectFloat32s(odd.GreaterEqual(vZero), odd, odd.Mul(vAlpha))
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func SeluBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func SeluFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vScale := simd.BroadcastFloat32s(seluScale)
	vScaleAlpha := simd.BroadcastFloat32s(seluScaleAlpha)
	vOne := simd.BroadcastFloat32s(1.0)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)

		evenPos := even.Mul(vScale)
		evenNeg := vScaleAlpha.Mul(simdmath.ExpFloat32(even).Sub(vOne))
		even = selectFloat32s(even.GreaterEqual(vZero), evenPos, evenNeg)

		oddPos := odd.Mul(vScale)
		oddNeg := vScaleAlpha.Mul(simdmath.ExpFloat32(odd).Sub(vOne))
		odd = selectFloat32s(odd.GreaterEqual(vZero), oddPos, oddNeg)

		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)

		evenPos := even.Mul(vScale)
		evenNeg := vScaleAlpha.Mul(simdmath.ExpFloat32(even).Sub(vOne))
		even = selectFloat32s(even.GreaterEqual(vZero), evenPos, evenNeg)

		oddPos := odd.Mul(vScale)
		oddNeg := vScaleAlpha.Mul(simdmath.ExpFloat32(odd).Sub(vOne))
		odd = selectFloat32s(odd.GreaterEqual(vZero), oddPos, oddNeg)

		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func SiluBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func SiluFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = even.Mul(simdmath.SigmoidFloat32(even))
		odd = odd.Mul(simdmath.SigmoidFloat32(odd))
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = even.Mul(simdmath.SigmoidFloat32(even))
		odd = odd.Mul(simdmath.SigmoidFloat32(odd))
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func HardSwishBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func HardSwishFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vOne := simd.BroadcastFloat32s(1)
	vOneSixth := simd.BroadcastFloat32s(1.0 / 6.0)
	vHalf := simd.BroadcastFloat32s(0.5)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)

		evenScaled := even.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		even = even.Mul(evenScaled)

		oddScaled := odd.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		odd = odd.Mul(oddScaled)

		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)

		evenScaled := even.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		even = even.Mul(evenScaled)

		oddScaled := odd.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		odd = odd.Mul(oddScaled)

		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func TanhBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func TanhFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = simdmath.TanhFloat32(even)
		odd = simdmath.TanhFloat32(odd)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = simdmath.TanhFloat32(even)
		odd = simdmath.TanhFloat32(odd)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func GeluExactBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func GeluExactFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vHalf := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vRsqrt2 := simd.BroadcastFloat32s(0.7071067811865475)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)

		evenErf := simdmath.ErfFloat32(even.Mul(vRsqrt2))
		even = vHalf.Mul(even).Mul(vOne.Add(evenErf))

		oddErf := simdmath.ErfFloat32(odd.Mul(vRsqrt2))
		odd = vHalf.Mul(odd).Mul(vOne.Add(oddErf))

		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)

		evenErf := simdmath.ErfFloat32(even.Mul(vRsqrt2))
		even = vHalf.Mul(even).Mul(vOne.Add(evenErf))

		oddErf := simdmath.ErfFloat32(odd.Mul(vRsqrt2))
		odd = vHalf.Mul(odd).Mul(vOne.Add(oddErf))

		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func GeluApproxBFloat16SIMD(in, out []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func GeluApproxFloat16SIMD(in, out []float16.Float16) {
	n := len(in)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		v := bfloat16.LoadBFloat16s(in[i : i+vecLen]) //alt:bfloat16
		//alt:float16 v := float16.LoadFloat16s(in[i : i+vecLen])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = simdmath.GeluApproxFloat32(even)
		odd = simdmath.GeluApproxFloat32(odd)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16s(res, out[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, out[i : i+vecLen])
	}
	if i < n {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:]) //alt:bfloat16
		//alt:float16 v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v) //alt:bfloat16
		//alt:float16 even, odd := float16.ToFloat32SIMD(v)
		even = simdmath.GeluApproxFloat32(even)
		odd = simdmath.GeluApproxFloat32(odd)
		res := bfloat16.FromFloat32SIMD(even, odd) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(even, odd)
		bfloat16.StoreBFloat16sPart(res, out[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, out[i:])
	}
}

func SwiGLUBFloat16SIMD(in, out []bfloat16.BFloat16, numRows, hiddenDim int) { //alt:bfloat16
//alt:float16 func SwiGLUFloat16SIMD(in, out []float16.Float16, numRows, hiddenDim int) {
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		gateSlice := in[inOffset : inOffset+hiddenDim]
		valSlice := in[inOffset+hiddenDim : inOffset+2*hiddenDim]
		outSlice := out[outOffset : outOffset+hiddenDim]

		j := 0
		for ; j+vecLen <= hiddenDim; j += vecLen {
			vGate := bfloat16.LoadBFloat16s(gateSlice[j : j+vecLen]) //alt:bfloat16
			//alt:float16 vGate := float16.LoadFloat16s(gateSlice[j : j+vecLen])
			evenGate, oddGate := bfloat16.ToFloat32SIMD(vGate) //alt:bfloat16
			//alt:float16 evenGate, oddGate := float16.ToFloat32SIMD(vGate)

			vVal := bfloat16.LoadBFloat16s(valSlice[j : j+vecLen]) //alt:bfloat16
			//alt:float16 vVal := float16.LoadFloat16s(valSlice[j : j+vecLen])
			evenVal, oddVal := bfloat16.ToFloat32SIMD(vVal) //alt:bfloat16
			//alt:float16 evenVal, oddVal := float16.ToFloat32SIMD(vVal)

			evenRes := evenVal.Mul(evenGate.Mul(simdmath.SigmoidFloat32(evenGate)))
			oddRes := oddVal.Mul(oddGate.Mul(simdmath.SigmoidFloat32(oddGate)))

			res := bfloat16.FromFloat32SIMD(evenRes, oddRes) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenRes, oddRes)
			bfloat16.StoreBFloat16s(res, outSlice[j : j+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, outSlice[j : j+vecLen])
		}
		if j < hiddenDim {
			vGate, _ := bfloat16.LoadBFloat16sPart(gateSlice[j:]) //alt:bfloat16
			//alt:float16 vGate, _ := float16.LoadFloat16sPart(gateSlice[j:])
			evenGate, oddGate := bfloat16.ToFloat32SIMD(vGate) //alt:bfloat16
			//alt:float16 evenGate, oddGate := float16.ToFloat32SIMD(vGate)

			vVal, _ := bfloat16.LoadBFloat16sPart(valSlice[j:]) //alt:bfloat16
			//alt:float16 vVal, _ := float16.LoadFloat16sPart(valSlice[j:])
			evenVal, oddVal := bfloat16.ToFloat32SIMD(vVal) //alt:bfloat16
			//alt:float16 evenVal, oddVal := float16.ToFloat32SIMD(vVal)

			evenRes := evenVal.Mul(evenGate.Mul(simdmath.SigmoidFloat32(evenGate)))
			oddRes := oddVal.Mul(oddGate.Mul(simdmath.SigmoidFloat32(oddGate)))

			res := bfloat16.FromFloat32SIMD(evenRes, oddRes) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenRes, oddRes)
			bfloat16.StoreBFloat16sPart(res, outSlice[j:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, outSlice[j:])
		}
	}
}

// -------------------------------------------------------------------------------------------------
// VJP (Vector-Jacobian Product) Kernels
// -------------------------------------------------------------------------------------------------

func vjpReluBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpReluFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	ref := y
	if len(ref) == 0 {
		ref = x
	}
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		vRef := bfloat16.LoadBFloat16s(ref[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vRef := float16.LoadFloat16s(ref[i : i+vecLen])
		evenRef, oddRef := bfloat16.ToFloat32SIMD(vRef) //alt:bfloat16
		//alt:float16 evenRef, oddRef := float16.ToFloat32SIMD(vRef)

		vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenDx := selectFloat32s(evenRef.Greater(vZero), evenDOut, vZero)
		oddDx := selectFloat32s(oddRef.Greater(vZero), oddDOut, vZero)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
	}
	if i < n {
		vRef, _ := bfloat16.LoadBFloat16sPart(ref[i:]) //alt:bfloat16
		//alt:float16 vRef, _ := float16.LoadFloat16sPart(ref[i:])
		evenRef, oddRef := bfloat16.ToFloat32SIMD(vRef) //alt:bfloat16
		//alt:float16 evenRef, oddRef := float16.ToFloat32SIMD(vRef)

		vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
		//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenDx := selectFloat32s(evenRef.Greater(vZero), evenDOut, vZero)
		oddDx := selectFloat32s(oddRef.Greater(vZero), oddDOut, vZero)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
	}
}

func vjpSigmoidBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpSigmoidFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vOne := simd.BroadcastFloat32s(1)
	i := 0
	if len(y) > 0 {
		for ; i+vecLen <= n; i += vecLen {
			vY := bfloat16.LoadBFloat16s(y[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vY := float16.LoadFloat16s(y[i : i+vecLen])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenDx := evenDOut.Mul(evenY).Mul(vOne.Sub(evenY))
			oddDx := oddDOut.Mul(oddY).Mul(vOne.Sub(oddY))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vY, _ := bfloat16.LoadBFloat16sPart(y[i:]) //alt:bfloat16
			//alt:float16 vY, _ := float16.LoadFloat16sPart(y[i:])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenDx := evenDOut.Mul(evenY).Mul(vOne.Sub(evenY))
			oddDx := oddDOut.Mul(oddY).Mul(vOne.Sub(oddY))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	} else {
		for ; i+vecLen <= n; i += vecLen {
			vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenS := simdmath.SigmoidFloat32(evenX)
			oddS := simdmath.SigmoidFloat32(oddX)

			evenDx := evenDOut.Mul(evenS).Mul(vOne.Sub(evenS))
			oddDx := oddDOut.Mul(oddS).Mul(vOne.Sub(oddS))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
			//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenS := simdmath.SigmoidFloat32(evenX)
			oddS := simdmath.SigmoidFloat32(oddX)

			evenDx := evenDOut.Mul(evenS).Mul(vOne.Sub(evenS))
			oddDx := oddDOut.Mul(oddS).Mul(vOne.Sub(oddS))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	}
}

func vjpHardSigmoidBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpHardSigmoidFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vSlope := simd.BroadcastFloat32s(0.2)
	i := 0
	if len(y) > 0 {
		vOne := simd.BroadcastFloat32s(1)
		for ; i+vecLen <= n; i += vecLen {
			vY := bfloat16.LoadBFloat16s(y[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vY := float16.LoadFloat16s(y[i : i+vecLen])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenMask := evenY.Greater(vZero).And(evenY.Less(vOne))
			oddMask := oddY.Greater(vZero).And(oddY.Less(vOne))

			evenDx := selectFloat32s(evenMask, evenDOut.Mul(vSlope), vZero)
			oddDx := selectFloat32s(oddMask, oddDOut.Mul(vSlope), vZero)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vY, _ := bfloat16.LoadBFloat16sPart(y[i:]) //alt:bfloat16
			//alt:float16 vY, _ := float16.LoadFloat16sPart(y[i:])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenMask := evenY.Greater(vZero).And(evenY.Less(vOne))
			oddMask := oddY.Greater(vZero).And(oddY.Less(vOne))

			evenDx := selectFloat32s(evenMask, evenDOut.Mul(vSlope), vZero)
			oddDx := selectFloat32s(oddMask, oddDOut.Mul(vSlope), vZero)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	} else {
		vMinusTwoPointFive := simd.BroadcastFloat32s(-2.5)
		vTwoPointFive := simd.BroadcastFloat32s(2.5)
		for ; i+vecLen <= n; i += vecLen {
			vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenMask := evenX.Greater(vMinusTwoPointFive).And(evenX.Less(vTwoPointFive))
			oddMask := oddX.Greater(vMinusTwoPointFive).And(oddX.Less(vTwoPointFive))

			evenDx := selectFloat32s(evenMask, evenDOut.Mul(vSlope), vZero)
			oddDx := selectFloat32s(oddMask, oddDOut.Mul(vSlope), vZero)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
			//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenMask := evenX.Greater(vMinusTwoPointFive).And(evenX.Less(vTwoPointFive))
			oddMask := oddX.Greater(vMinusTwoPointFive).And(oddX.Less(vTwoPointFive))

			evenDx := selectFloat32s(evenMask, evenDOut.Mul(vSlope), vZero)
			oddDx := selectFloat32s(oddMask, oddDOut.Mul(vSlope), vZero)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	}
}

func vjpLeakyReluBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpLeakyReluFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vSlope := simd.BroadcastFloat32s(0.3)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenDx := selectFloat32s(evenX.GreaterEqual(vZero), evenDOut, evenDOut.Mul(vSlope))
		oddDx := selectFloat32s(oddX.GreaterEqual(vZero), oddDOut, oddDOut.Mul(vSlope))

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
	}
	if i < n {
		vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
		//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
		//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenDx := selectFloat32s(evenX.GreaterEqual(vZero), evenDOut, evenDOut.Mul(vSlope))
		oddDx := selectFloat32s(oddX.GreaterEqual(vZero), oddDOut, oddDOut.Mul(vSlope))

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
	}
}

func vjpSeluBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpSeluFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vScale := simd.BroadcastFloat32s(seluScale)
	vScaleAlpha := simd.BroadcastFloat32s(seluScaleAlpha)
	i := 0
	if len(y) > 0 {
		for ; i+vecLen <= n; i += vecLen {
			vY := bfloat16.LoadBFloat16s(y[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vY := float16.LoadFloat16s(y[i : i+vecLen])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenPos := evenDOut.Mul(vScale)
			evenNeg := evenDOut.Mul(evenY.Add(vScaleAlpha))
			evenDx := selectFloat32s(evenY.Greater(vZero), evenPos, evenNeg)

			oddPos := oddDOut.Mul(vScale)
			oddNeg := oddDOut.Mul(oddY.Add(vScaleAlpha))
			oddDx := selectFloat32s(oddY.Greater(vZero), oddPos, oddNeg)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vY, _ := bfloat16.LoadBFloat16sPart(y[i:]) //alt:bfloat16
			//alt:float16 vY, _ := float16.LoadFloat16sPart(y[i:])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenPos := evenDOut.Mul(vScale)
			evenNeg := evenDOut.Mul(evenY.Add(vScaleAlpha))
			evenDx := selectFloat32s(evenY.Greater(vZero), evenPos, evenNeg)

			oddPos := oddDOut.Mul(vScale)
			oddNeg := oddDOut.Mul(oddY.Add(vScaleAlpha))
			oddDx := selectFloat32s(oddY.Greater(vZero), oddPos, oddNeg)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	} else {
		for ; i+vecLen <= n; i += vecLen {
			vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenPos := evenDOut.Mul(vScale)
			evenNeg := evenDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(evenX))
			evenDx := selectFloat32s(evenX.Greater(vZero), evenPos, evenNeg)

			oddPos := oddDOut.Mul(vScale)
			oddNeg := oddDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(oddX))
			oddDx := selectFloat32s(oddX.Greater(vZero), oddPos, oddNeg)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
			//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenPos := evenDOut.Mul(vScale)
			evenNeg := evenDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(evenX))
			evenDx := selectFloat32s(evenX.Greater(vZero), evenPos, evenNeg)

			oddPos := oddDOut.Mul(vScale)
			oddNeg := oddDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(oddX))
			oddDx := selectFloat32s(oddX.Greater(vZero), oddPos, oddNeg)

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	}
}

func vjpSiluBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpSiluFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vOne := simd.BroadcastFloat32s(1)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenS := simdmath.SigmoidFloat32(evenX)
		oddS := simdmath.SigmoidFloat32(oddX)

		evenFPrime := evenS.Mul(vOne.Add(evenX.Mul(vOne.Sub(evenS))))
		oddFPrime := oddS.Mul(vOne.Add(oddX.Mul(vOne.Sub(oddS))))

		evenDx := evenDOut.Mul(evenFPrime)
		oddDx := oddDOut.Mul(oddFPrime)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
	}
	if i < n {
		vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
		//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
		//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenS := simdmath.SigmoidFloat32(evenX)
		oddS := simdmath.SigmoidFloat32(oddX)

		evenFPrime := evenS.Mul(vOne.Add(evenX.Mul(vOne.Sub(evenS))))
		oddFPrime := oddS.Mul(vOne.Add(oddX.Mul(vOne.Sub(oddS))))

		evenDx := evenDOut.Mul(evenFPrime)
		oddDx := oddDOut.Mul(oddFPrime)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
	}
}

func vjpHardSwishBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpHardSwishFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vZero := simd.BroadcastFloat32s(0)
	vThree := simd.BroadcastFloat32s(3)
	vNegThree := simd.BroadcastFloat32s(-3)
	vOneThird := simd.BroadcastFloat32s(1.0 / 3.0)
	vHalf := simd.BroadcastFloat32s(0.5)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		midEven := evenDOut.Mul(evenX.MulAdd(vOneThird, vHalf))
		midOdd := oddDOut.Mul(oddX.MulAdd(vOneThird, vHalf))

		resEven := selectFloat32s(evenX.Greater(vNegThree), midEven, vZero)
		resOdd := selectFloat32s(oddX.Greater(vNegThree), midOdd, vZero)

		evenDx := selectFloat32s(evenX.GreaterEqual(vThree), evenDOut, resEven)
		oddDx := selectFloat32s(oddX.GreaterEqual(vThree), oddDOut, resOdd)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
	}
	if i < n {
		vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
		//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
		//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		midEven := evenDOut.Mul(evenX.MulAdd(vOneThird, vHalf))
		midOdd := oddDOut.Mul(oddX.MulAdd(vOneThird, vHalf))

		resEven := selectFloat32s(evenX.Greater(vNegThree), midEven, vZero)
		resOdd := selectFloat32s(oddX.Greater(vNegThree), midOdd, vZero)

		evenDx := selectFloat32s(evenX.GreaterEqual(vThree), evenDOut, resEven)
		oddDx := selectFloat32s(oddX.GreaterEqual(vThree), oddDOut, resOdd)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
	}
}

func vjpTanhBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpTanhFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vOne := simd.BroadcastFloat32s(1)
	i := 0
	if len(y) > 0 {
		for ; i+vecLen <= n; i += vecLen {
			vY := bfloat16.LoadBFloat16s(y[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vY := float16.LoadFloat16s(y[i : i+vecLen])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenDx := evenDOut.Mul(vOne.Sub(evenY.Mul(evenY)))
			oddDx := oddDOut.Mul(vOne.Sub(oddY.Mul(oddY)))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vY, _ := bfloat16.LoadBFloat16sPart(y[i:]) //alt:bfloat16
			//alt:float16 vY, _ := float16.LoadFloat16sPart(y[i:])
			evenY, oddY := bfloat16.ToFloat32SIMD(vY) //alt:bfloat16
			//alt:float16 evenY, oddY := float16.ToFloat32SIMD(vY)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenDx := evenDOut.Mul(vOne.Sub(evenY.Mul(evenY)))
			oddDx := oddDOut.Mul(vOne.Sub(oddY.Mul(oddY)))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	} else {
		for ; i+vecLen <= n; i += vecLen {
			vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenT := simdmath.TanhFloat32(evenX)
			oddT := simdmath.TanhFloat32(oddX)

			evenDx := evenDOut.Mul(vOne.Sub(evenT.Mul(evenT)))
			oddDx := oddDOut.Mul(vOne.Sub(oddT.Mul(oddT)))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
		}
		if i < n {
			vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
			//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
			evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
			//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenT := simdmath.TanhFloat32(evenX)
			oddT := simdmath.TanhFloat32(oddX)

			evenDx := evenDOut.Mul(vOne.Sub(evenT.Mul(evenT)))
			oddDx := oddDOut.Mul(vOne.Sub(oddT.Mul(oddT)))

			res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
			//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
			bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
		}
	}
}

func GeluExactBFloat16SIMDVJP(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func GeluExactFloat16SIMDVJP(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vHalf := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vInvSqrt2 := simd.BroadcastFloat32s(0.7071067811865475)
	vInvSqrt2Pi := simd.BroadcastFloat32s(0.3989422804014327)
	vNegHalf := simd.BroadcastFloat32s(-0.5)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenCdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(evenX.Mul(vInvSqrt2))))
		oddCdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(oddX.Mul(vInvSqrt2))))

		evenPdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(evenX.Mul(evenX).Mul(vNegHalf)))
		oddPdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(oddX.Mul(oddX).Mul(vNegHalf)))

		evenDx := evenDOut.Mul(evenCdf.Add(evenX.Mul(evenPdf)))
		oddDx := oddDOut.Mul(oddCdf.Add(oddX.Mul(oddPdf)))

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
	}
	if i < n {
		vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
		//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
		//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenCdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(evenX.Mul(vInvSqrt2))))
		oddCdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(oddX.Mul(vInvSqrt2))))

		evenPdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(evenX.Mul(evenX).Mul(vNegHalf)))
		oddPdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(oddX.Mul(oddX).Mul(vNegHalf)))

		evenDx := evenDOut.Mul(evenCdf.Add(evenX.Mul(evenPdf)))
		oddDx := oddDOut.Mul(oddCdf.Add(oddX.Mul(oddPdf)))

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
	}
}

func vjpGeluApproxBFloat16SIMD(y, x, dOutput, dx []bfloat16.BFloat16) { //alt:bfloat16
//alt:float16 func vjpGeluApproxFloat16SIMD(y, x, dOutput, dx []float16.Float16) {
	n := len(dOutput)
	if n == 0 {
		return
	}
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vHalf := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vSqrt2ByPi := simd.BroadcastFloat32s(0.7978845608)
	vConst044715 := simd.BroadcastFloat32s(0.044715)
	vConst0134145 := simd.BroadcastFloat32s(0.134145)
	i := 0
	for ; i+vecLen <= n; i += vecLen {
		vX := bfloat16.LoadBFloat16s(x[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vX := float16.LoadFloat16s(x[i : i+vecLen])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut := bfloat16.LoadBFloat16s(dOutput[i : i+vecLen]) //alt:bfloat16
		//alt:float16 vDOut := float16.LoadFloat16s(dOutput[i : i+vecLen])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenX2 := evenX.Mul(evenX)
		evenX3 := evenX2.Mul(evenX)
		evenU := vSqrt2ByPi.Mul(evenX.MulAdd(vConst044715, evenX3))
		evenUPrime := vSqrt2ByPi.Mul(evenX2.MulAdd(vConst0134145, vOne))
		evenT := simdmath.TanhFloat32(evenU)
		evenFPrime := vHalf.Mul(vOne.Add(evenT)).Add(vHalf.Mul(evenX).Mul(vOne.Sub(evenT.Mul(evenT))).Mul(evenUPrime))
		evenDx := evenDOut.Mul(evenFPrime)

		oddX2 := oddX.Mul(oddX)
		oddX3 := oddX2.Mul(oddX)
		oddU := vSqrt2ByPi.Mul(oddX.MulAdd(vConst044715, oddX3))
		oddUPrime := vSqrt2ByPi.Mul(oddX2.MulAdd(vConst0134145, vOne))
		oddT := simdmath.TanhFloat32(oddU)
		oddFPrime := vHalf.Mul(vOne.Add(oddT)).Add(vHalf.Mul(oddX).Mul(vOne.Sub(oddT.Mul(oddT))).Mul(oddUPrime))
		oddDx := oddDOut.Mul(oddFPrime)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16s(res, dx[i : i+vecLen]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16s(res, dx[i : i+vecLen])
	}
	if i < n {
		vX, _ := bfloat16.LoadBFloat16sPart(x[i:]) //alt:bfloat16
		//alt:float16 vX, _ := float16.LoadFloat16sPart(x[i:])
		evenX, oddX := bfloat16.ToFloat32SIMD(vX) //alt:bfloat16
		//alt:float16 evenX, oddX := float16.ToFloat32SIMD(vX)

		vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[i:]) //alt:bfloat16
		//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[i:])
		evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
		//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

		evenX2 := evenX.Mul(evenX)
		evenX3 := evenX2.Mul(evenX)
		evenU := vSqrt2ByPi.Mul(evenX.MulAdd(vConst044715, evenX3))
		evenUPrime := vSqrt2ByPi.Mul(evenX2.MulAdd(vConst0134145, vOne))
		evenT := simdmath.TanhFloat32(evenU)
		evenFPrime := vHalf.Mul(vOne.Add(evenT)).Add(vHalf.Mul(evenX).Mul(vOne.Sub(evenT.Mul(evenT))).Mul(evenUPrime))
		evenDx := evenDOut.Mul(evenFPrime)

		oddX2 := oddX.Mul(oddX)
		oddX3 := oddX2.Mul(oddX)
		oddU := vSqrt2ByPi.Mul(oddX.MulAdd(vConst044715, oddX3))
		oddUPrime := vSqrt2ByPi.Mul(oddX2.MulAdd(vConst0134145, vOne))
		oddT := simdmath.TanhFloat32(oddU)
		oddFPrime := vHalf.Mul(vOne.Add(oddT)).Add(vHalf.Mul(oddX).Mul(vOne.Sub(oddT.Mul(oddT))).Mul(oddUPrime))
		oddDx := oddDOut.Mul(oddFPrime)

		res := bfloat16.FromFloat32SIMD(evenDx, oddDx) //alt:bfloat16
		//alt:float16 res := float16.FromFloat32SIMD(evenDx, oddDx)
		bfloat16.StoreBFloat16sPart(res, dx[i:]) //alt:bfloat16
		//alt:float16 float16.StoreFloat16sPart(res, dx[i:])
	}
}

func vjpSwiGLUBFloat16SIMD(x, dOutput, dx []bfloat16.BFloat16, numRows, hiddenDim int) { //alt:bfloat16
//alt:float16 func vjpSwiGLUFloat16SIMD(x, dOutput, dx []float16.Float16, numRows, hiddenDim int) {
	dummy := simd.BroadcastUint16s(0)
	vecLen := dummy.Len()
	vOne := simd.BroadcastFloat32s(1)
	for m := range numRows {
		xOffset := m * 2 * hiddenDim
		dOutOffset := m * hiddenDim
		j := 0
		for ; j+vecLen <= hiddenDim; j += vecLen {
			vGate := bfloat16.LoadBFloat16s(x[xOffset+j : xOffset+j+vecLen]) //alt:bfloat16
			//alt:float16 vGate := float16.LoadFloat16s(x[xOffset+j : xOffset+j+vecLen])
			evenGate, oddGate := bfloat16.ToFloat32SIMD(vGate) //alt:bfloat16
			//alt:float16 evenGate, oddGate := float16.ToFloat32SIMD(vGate)

			vVal := bfloat16.LoadBFloat16s(x[xOffset+hiddenDim+j : xOffset+hiddenDim+j+vecLen]) //alt:bfloat16
			//alt:float16 vVal := float16.LoadFloat16s(x[xOffset+hiddenDim+j : xOffset+hiddenDim+j+vecLen])
			evenVal, oddVal := bfloat16.ToFloat32SIMD(vVal) //alt:bfloat16
			//alt:float16 evenVal, oddVal := float16.ToFloat32SIMD(vVal)

			vDOut := bfloat16.LoadBFloat16s(dOutput[dOutOffset+j : dOutOffset+j+vecLen]) //alt:bfloat16
			//alt:float16 vDOut := float16.LoadFloat16s(dOutput[dOutOffset+j : dOutOffset+j+vecLen])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenS := simdmath.SigmoidFloat32(evenGate)
			oddS := simdmath.SigmoidFloat32(oddGate)

			evenSwish := evenGate.Mul(evenS)
			oddSwish := oddGate.Mul(oddS)

			evenSwishPrime := evenS.Mul(vOne.Add(evenGate.Mul(vOne.Sub(evenS))))
			oddSwishPrime := oddS.Mul(vOne.Add(oddGate.Mul(vOne.Sub(oddS))))

			evenDGate := evenDOut.Mul(evenVal).Mul(evenSwishPrime)
			oddDGate := oddDOut.Mul(oddVal).Mul(oddSwishPrime)

			evenDVal := evenDOut.Mul(evenSwish)
			oddDVal := oddDOut.Mul(oddSwish)

			resGate := bfloat16.FromFloat32SIMD(evenDGate, oddDGate) //alt:bfloat16
			//alt:float16 resGate := float16.FromFloat32SIMD(evenDGate, oddDGate)
			bfloat16.StoreBFloat16s(resGate, dx[xOffset+j : xOffset+j+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(resGate, dx[xOffset+j : xOffset+j+vecLen])

			resVal := bfloat16.FromFloat32SIMD(evenDVal, oddDVal) //alt:bfloat16
			//alt:float16 resVal := float16.FromFloat32SIMD(evenDVal, oddDVal)
			bfloat16.StoreBFloat16s(resVal, dx[xOffset+hiddenDim+j : xOffset+hiddenDim+j+vecLen]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16s(resVal, dx[xOffset+hiddenDim+j : xOffset+hiddenDim+j+vecLen])
		}
		if j < hiddenDim {
			vGate, _ := bfloat16.LoadBFloat16sPart(x[xOffset+j:]) //alt:bfloat16
			//alt:float16 vGate, _ := float16.LoadFloat16sPart(x[xOffset+j:])
			evenGate, oddGate := bfloat16.ToFloat32SIMD(vGate) //alt:bfloat16
			//alt:float16 evenGate, oddGate := float16.ToFloat32SIMD(vGate)

			vVal, _ := bfloat16.LoadBFloat16sPart(x[xOffset+hiddenDim+j:]) //alt:bfloat16
			//alt:float16 vVal, _ := float16.LoadFloat16sPart(x[xOffset+hiddenDim+j:])
			evenVal, oddVal := bfloat16.ToFloat32SIMD(vVal) //alt:bfloat16
			//alt:float16 evenVal, oddVal := float16.ToFloat32SIMD(vVal)

			vDOut, _ := bfloat16.LoadBFloat16sPart(dOutput[dOutOffset+j:]) //alt:bfloat16
			//alt:float16 vDOut, _ := float16.LoadFloat16sPart(dOutput[dOutOffset+j:])
			evenDOut, oddDOut := bfloat16.ToFloat32SIMD(vDOut) //alt:bfloat16
			//alt:float16 evenDOut, oddDOut := float16.ToFloat32SIMD(vDOut)

			evenS := simdmath.SigmoidFloat32(evenGate)
			oddS := simdmath.SigmoidFloat32(oddGate)

			evenSwish := evenGate.Mul(evenS)
			oddSwish := oddGate.Mul(oddS)

			evenSwishPrime := evenS.Mul(vOne.Add(evenGate.Mul(vOne.Sub(evenS))))
			oddSwishPrime := oddS.Mul(vOne.Add(oddGate.Mul(vOne.Sub(oddS))))

			evenDGate := evenDOut.Mul(evenVal).Mul(evenSwishPrime)
			oddDGate := oddDOut.Mul(oddVal).Mul(oddSwishPrime)

			evenDVal := evenDOut.Mul(evenSwish)
			oddDVal := oddDOut.Mul(oddSwish)

			resGate := bfloat16.FromFloat32SIMD(evenDGate, oddDGate) //alt:bfloat16
			//alt:float16 resGate := float16.FromFloat32SIMD(evenDGate, oddDGate)
			bfloat16.StoreBFloat16sPart(resGate, dx[xOffset+j:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(resGate, dx[xOffset+j:])

			resVal := bfloat16.FromFloat32SIMD(evenDVal, oddDVal) //alt:bfloat16
			//alt:float16 resVal := float16.FromFloat32SIMD(evenDVal, oddDVal)
			bfloat16.StoreBFloat16sPart(resVal, dx[xOffset+hiddenDim+j:]) //alt:bfloat16
			//alt:float16 float16.StoreFloat16sPart(resVal, dx[xOffset+hiddenDim+j:])
		}
	}
}

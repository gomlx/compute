// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package activations

import (
	"simd"

	"github.com/gomlx/compute/support/simdmath"
)

// -------------------------------------------------------------------------------------------------
// Forward Activations
// -------------------------------------------------------------------------------------------------

func ReluFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func ReluFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		v.Max(vZero).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		v.Max(vZero).StorePart(out[i:])
	}
}

func SigmoidFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func SigmoidFloat64SIMD(in, out []float64) {
	vLen := simd.BroadcastFloat32s(0).Len() //alt:float32
	//alt:float64 vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		simdmath.SigmoidFloat32(v).Store(out[i:]) //alt:float32
		//alt:float64 simdmath.SigmoidFloat64(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		simdmath.SigmoidFloat32(v).StorePart(out[i:]) //alt:float32
		//alt:float64 simdmath.SigmoidFloat64(v).StorePart(out[i:])
	}
}

func HardSigmoidFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func HardSigmoidFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vOne := simd.BroadcastFloat32s(1) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1)
	vPointTwo := simd.BroadcastFloat32s(0.2) //alt:float32
	//alt:float64 vPointTwo := simd.BroadcastFloat64s(0.2)
	vHalf := simd.BroadcastFloat32s(0.5) //alt:float32
	//alt:float64 vHalf := simd.BroadcastFloat64s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		v.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		v.MulAdd(vPointTwo, vHalf).Max(vZero).Min(vOne).StorePart(out[i:])
	}
}

func LeakyReluFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func LeakyReluFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vAlpha := simd.BroadcastFloat32s(0.3) //alt:float32
	//alt:float64 vAlpha := simd.BroadcastFloat64s(0.3)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		scaled := v.Mul(vAlpha)
		v.IfElse(v.GreaterEqual(vZero), scaled).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		scaled := v.Mul(vAlpha)
		v.IfElse(v.GreaterEqual(vZero), scaled).StorePart(out[i:])
	}
}

func SeluFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func SeluFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vScale := simd.BroadcastFloat32s(seluScale) //alt:float32
	//alt:float64 vScale := simd.BroadcastFloat64s(seluScale)
	vScaleAlpha := simd.BroadcastFloat32s(seluScaleAlpha) //alt:float32
	//alt:float64 vScaleAlpha := simd.BroadcastFloat64s(seluScaleAlpha)
	vOne := simd.BroadcastFloat32s(1.0) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1.0)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		pos := v.Mul(vScale)
		neg := vScaleAlpha.Mul(simdmath.ExpFloat32(v).Sub(vOne)) //alt:float32
		//alt:float64 neg := vScaleAlpha.Mul(simdmath.ExpFloat64(v).Sub(vOne))
		pos.IfElse(v.GreaterEqual(vZero), neg).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		pos := v.Mul(vScale)
		neg := vScaleAlpha.Mul(simdmath.ExpFloat32(v).Sub(vOne)) //alt:float32
		//alt:float64 neg := vScaleAlpha.Mul(simdmath.ExpFloat64(v).Sub(vOne))
		pos.IfElse(v.GreaterEqual(vZero), neg).StorePart(out[i:])
	}
}

func SiluFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func SiluFloat64SIMD(in, out []float64) {
	vLen := simd.BroadcastFloat32s(0).Len() //alt:float32
	//alt:float64 vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		simdmath.SigmoidFloat32(v).Mul(v).Store(out[i:]) //alt:float32
		//alt:float64 simdmath.SigmoidFloat64(v).Mul(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		simdmath.SigmoidFloat32(v).Mul(v).StorePart(out[i:]) //alt:float32
		//alt:float64 simdmath.SigmoidFloat64(v).Mul(v).StorePart(out[i:])
	}
}

func HardSwishFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func HardSwishFloat64SIMD(in, out []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vOne := simd.BroadcastFloat32s(1) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1)
	vOneSixth := simd.BroadcastFloat32s(1.0 / 6.0) //alt:float32
	//alt:float64 vOneSixth := simd.BroadcastFloat64s(1.0 / 6.0)
	vHalf := simd.BroadcastFloat32s(0.5) //alt:float32
	//alt:float64 vHalf := simd.BroadcastFloat64s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		scaled := v.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		v.Mul(scaled).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		scaled := v.MulAdd(vOneSixth, vHalf).Max(vZero).Min(vOne)
		v.Mul(scaled).StorePart(out[i:])
	}
}

func TanhFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func TanhFloat64SIMD(in, out []float64) {
	vLen := simd.BroadcastFloat32s(0).Len() //alt:float32
	//alt:float64 vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		simdmath.TanhFloat32(v).Store(out[i:]) //alt:float32
		//alt:float64 simdmath.TanhFloat64(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		simdmath.TanhFloat32(v).StorePart(out[i:]) //alt:float32
		//alt:float64 simdmath.TanhFloat64(v).StorePart(out[i:])
	}
}

func GeluExactFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func GeluExactFloat64SIMD(in, out []float64) {
	vHalf := simd.BroadcastFloat32s(0.5) //alt:float32
	//alt:float64 vHalf := simd.BroadcastFloat64s(0.5)
	vOne := simd.BroadcastFloat32s(1.0) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1.0)
	vRsqrt2 := simd.BroadcastFloat32s(0.7071067811865475) //alt:float32
	//alt:float64 vRsqrt2 := simd.BroadcastFloat64s(0.7071067811865475)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		erfVal := simdmath.ErfFloat32(v.Mul(vRsqrt2)) //alt:float32
		//alt:float64 erfVal := simdmath.ErfFloat64(v.Mul(vRsqrt2))
		vHalf.Mul(v).Mul(vOne.Add(erfVal)).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		erfVal := simdmath.ErfFloat32(v.Mul(vRsqrt2)) //alt:float32
		//alt:float64 erfVal := simdmath.ErfFloat64(v.Mul(vRsqrt2))
		vHalf.Mul(v).Mul(vOne.Add(erfVal)).StorePart(out[i:])
	}
}

func GeluApproxFloat32SIMD(in, out []float32) { //alt:float32
//alt:float64 func GeluApproxFloat64SIMD(in, out []float64) {
	vLen := simd.BroadcastFloat32s(0).Len() //alt:float32
	//alt:float64 vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(in); i += vLen {
		v := simd.LoadFloat32s(in[i:]) //alt:float32
		//alt:float64 v := simd.LoadFloat64s(in[i:])
		simdmath.GeluApproxFloat32(v).Store(out[i:]) //alt:float32
		//alt:float64 simdmath.GeluApproxFloat64(v).Store(out[i:])
	}
	if i < len(in) {
		v, _ := simd.LoadFloat32sPart(in[i:]) //alt:float32
		//alt:float64 v, _ := simd.LoadFloat64sPart(in[i:])
		simdmath.GeluApproxFloat32(v).StorePart(out[i:]) //alt:float32
		//alt:float64 simdmath.GeluApproxFloat64(v).StorePart(out[i:])
	}
}

func SwiGLUFloat32SIMD(in, out []float32, numRows, hiddenDim int) { //alt:float32
//alt:float64 func SwiGLUFloat64SIMD(in, out []float64, numRows, hiddenDim int) {
	vLen := simd.BroadcastFloat32s(0).Len() //alt:float32
	//alt:float64 vLen := simd.BroadcastFloat64s(0).Len()
	for m := range numRows {
		inOffset := m * 2 * hiddenDim
		outOffset := m * hiddenDim
		j := 0
		for ; j+vLen <= hiddenDim; j += vLen {
			vGate := simd.LoadFloat32s(in[inOffset+j:]) //alt:float32
			//alt:float64 vGate := simd.LoadFloat64s(in[inOffset+j:])
			vVal := simd.LoadFloat32s(in[inOffset+hiddenDim+j:]) //alt:float32
			//alt:float64 vVal := simd.LoadFloat64s(in[inOffset+hiddenDim+j:])
			siluGate := vGate.Mul(simdmath.SigmoidFloat32(vGate)) //alt:float32
			//alt:float64 siluGate := vGate.Mul(simdmath.SigmoidFloat64(vGate))
			siluGate.Mul(vVal).Store(out[outOffset+j:])
		}
		if j < hiddenDim {
			vGate, _ := simd.LoadFloat32sPart(in[inOffset+j:]) //alt:float32
			//alt:float64 vGate, _ := simd.LoadFloat64sPart(in[inOffset+j:])
			vVal, _ := simd.LoadFloat32sPart(in[inOffset+hiddenDim+j:]) //alt:float32
			//alt:float64 vVal, _ := simd.LoadFloat64sPart(in[inOffset+hiddenDim+j:])
			siluGate := vGate.Mul(simdmath.SigmoidFloat32(vGate)) //alt:float32
			//alt:float64 siluGate := vGate.Mul(simdmath.SigmoidFloat64(vGate))
			siluGate.Mul(vVal).StorePart(out[outOffset+j:])
		}
	}
}

// -------------------------------------------------------------------------------------------------
// VJP (Vector-Jacobian Product) Kernels
// -------------------------------------------------------------------------------------------------

func vjpReluFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpReluFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vLen := vZero.Len()
	ref := y
	if len(ref) == 0 {
		ref = x
	}
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vRef := simd.LoadFloat32s(ref[i:]) //alt:float32
		//alt:float64 vRef := simd.LoadFloat64s(ref[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
		//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
		mask := vRef.Greater(vZero)
		vDOut.IfElse(mask, vZero).Store(dx[i:])
	}
	if i < len(dOutput) {
		vRef, _ := simd.LoadFloat32sPart(ref[i:]) //alt:float32
		//alt:float64 vRef, _ := simd.LoadFloat64sPart(ref[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
		//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		mask := vRef.Greater(vZero)
		vDOut.IfElse(mask, vZero).StorePart(dx[i:])
	}
}

func vjpSigmoidFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpSigmoidFloat64SIMD(y, x, dOutput, dx []float64) {
	vOne := simd.BroadcastFloat32s(1) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:]) //alt:float32
			//alt:float64 vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			vDOut.Mul(vY).Mul(vOne.Sub(vY)).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:]) //alt:float32
			//alt:float64 vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			vDOut.Mul(vY).Mul(vOne.Sub(vY)).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:]) //alt:float32
			//alt:float64 vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			s := simdmath.SigmoidFloat32(vX) //alt:float32
			//alt:float64 s := simdmath.SigmoidFloat64(vX)
			vDOut.Mul(s).Mul(vOne.Sub(s)).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
			//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			s := simdmath.SigmoidFloat32(vX) //alt:float32
			//alt:float64 s := simdmath.SigmoidFloat64(vX)
			vDOut.Mul(s).Mul(vOne.Sub(s)).StorePart(dx[i:])
		}
	}
}

func vjpHardSigmoidFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpHardSigmoidFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vSlope := simd.BroadcastFloat32s(0.2) //alt:float32
	//alt:float64 vSlope := simd.BroadcastFloat64s(0.2)
	vLen := vZero.Len()
	i := 0
	if len(y) > 0 {
		vOne := simd.BroadcastFloat32s(1) //alt:float32
		//alt:float64 vOne := simd.BroadcastFloat64s(1)
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:]) //alt:float32
			//alt:float64 vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			mask := vY.Greater(vZero).And(vY.Less(vOne))
			vDOut.Mul(vSlope).IfElse(mask, vZero).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:]) //alt:float32
			//alt:float64 vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			mask := vY.Greater(vZero).And(vY.Less(vOne))
			vDOut.Mul(vSlope).IfElse(mask, vZero).StorePart(dx[i:])
		}
	} else {
		vMinusTwoPointFive := simd.BroadcastFloat32s(-2.5) //alt:float32
		//alt:float64 vMinusTwoPointFive := simd.BroadcastFloat64s(-2.5)
		vTwoPointFive := simd.BroadcastFloat32s(2.5) //alt:float32
		//alt:float64 vTwoPointFive := simd.BroadcastFloat64s(2.5)
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:]) //alt:float32
			//alt:float64 vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			mask := vX.Greater(vMinusTwoPointFive).And(vX.Less(vTwoPointFive))
			vDOut.Mul(vSlope).IfElse(mask, vZero).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
			//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			mask := vX.Greater(vMinusTwoPointFive).And(vX.Less(vTwoPointFive))
			vDOut.Mul(vSlope).IfElse(mask, vZero).StorePart(dx[i:])
		}
	}
}

func vjpLeakyReluFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpLeakyReluFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vSlope := simd.BroadcastFloat32s(0.3) //alt:float32
	//alt:float64 vSlope := simd.BroadcastFloat64s(0.3)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:]) //alt:float32
		//alt:float64 vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
		//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
		vDOut.IfElse(vX.GreaterEqual(vZero), vDOut.Mul(vSlope)).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
		//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
		//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		vDOut.IfElse(vX.GreaterEqual(vZero), vDOut.Mul(vSlope)).StorePart(dx[i:])
	}
}

func vjpSeluFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpSeluFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vScale := simd.BroadcastFloat32s(seluScale) //alt:float32
	//alt:float64 vScale := simd.BroadcastFloat64s(seluScale)
	vScaleAlpha := simd.BroadcastFloat32s(seluScaleAlpha) //alt:float32
	//alt:float64 vScaleAlpha := simd.BroadcastFloat64s(seluScaleAlpha)
	vLen := vZero.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:]) //alt:float32
			//alt:float64 vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vY.Add(vScaleAlpha))
			pos.IfElse(vY.Greater(vZero), neg).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:]) //alt:float32
			//alt:float64 vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vY.Add(vScaleAlpha))
			pos.IfElse(vY.Greater(vZero), neg).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:]) //alt:float32
			//alt:float64 vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(vX)) //alt:float32
			//alt:float64 neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat64(vX))
			pos.IfElse(vX.Greater(vZero), neg).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
			//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			pos := vDOut.Mul(vScale)
			neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat32(vX)) //alt:float32
			//alt:float64 neg := vDOut.Mul(vScaleAlpha).Mul(simdmath.ExpFloat64(vX))
			pos.IfElse(vX.Greater(vZero), neg).StorePart(dx[i:])
		}
	}
}

func vjpSiluFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpSiluFloat64SIMD(y, x, dOutput, dx []float64) {
	vOne := simd.BroadcastFloat32s(1) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:]) //alt:float32
		//alt:float64 vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
		//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
		s := simdmath.SigmoidFloat32(vX) //alt:float32
		//alt:float64 s := simdmath.SigmoidFloat64(vX)
		fPrime := s.Mul(vOne.Add(vX.Mul(vOne.Sub(s))))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
		//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
		//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		s := simdmath.SigmoidFloat32(vX) //alt:float32
		//alt:float64 s := simdmath.SigmoidFloat64(vX)
		fPrime := s.Mul(vOne.Add(vX.Mul(vOne.Sub(s))))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpHardSwishFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpHardSwishFloat64SIMD(y, x, dOutput, dx []float64) {
	vZero := simd.BroadcastFloat32s(0) //alt:float32
	//alt:float64 vZero := simd.BroadcastFloat64s(0)
	vThree := simd.BroadcastFloat32s(3) //alt:float32
	//alt:float64 vThree := simd.BroadcastFloat64s(3)
	vNegThree := simd.BroadcastFloat32s(-3) //alt:float32
	//alt:float64 vNegThree := simd.BroadcastFloat64s(-3)
	vOneThird := simd.BroadcastFloat32s(1.0 / 3.0) //alt:float32
	//alt:float64 vOneThird := simd.BroadcastFloat64s(1.0 / 3.0)
	vHalf := simd.BroadcastFloat32s(0.5) //alt:float32
	//alt:float64 vHalf := simd.BroadcastFloat64s(0.5)
	vLen := vZero.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:]) //alt:float32
		//alt:float64 vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
		//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
		mid := vDOut.Mul(vX.MulAdd(vOneThird, vHalf))
		res := mid.IfElse(vX.Greater(vNegThree), vZero)
		res = vDOut.IfElse(vX.GreaterEqual(vThree), res)
		res.Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
		//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
		//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		mid := vDOut.Mul(vX.MulAdd(vOneThird, vHalf))
		res := mid.IfElse(vX.Greater(vNegThree), vZero)
		res = vDOut.IfElse(vX.GreaterEqual(vThree), res)
		res.StorePart(dx[i:])
	}
}

func vjpTanhFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpTanhFloat64SIMD(y, x, dOutput, dx []float64) {
	vOne := simd.BroadcastFloat32s(1) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	i := 0
	if len(y) > 0 {
		for ; i+vLen <= len(dOutput); i += vLen {
			vY := simd.LoadFloat32s(y[i:]) //alt:float32
			//alt:float64 vY := simd.LoadFloat64s(y[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			vDOut.Mul(vOne.Sub(vY.Mul(vY))).Store(dx[i:])
		}
		if i < len(dOutput) {
			vY, _ := simd.LoadFloat32sPart(y[i:]) //alt:float32
			//alt:float64 vY, _ := simd.LoadFloat64sPart(y[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			vDOut.Mul(vOne.Sub(vY.Mul(vY))).StorePart(dx[i:])
		}
	} else {
		for ; i+vLen <= len(dOutput); i += vLen {
			vX := simd.LoadFloat32s(x[i:]) //alt:float32
			//alt:float64 vX := simd.LoadFloat64s(x[i:])
			vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
			t := simdmath.TanhFloat32(vX) //alt:float32
			//alt:float64 t := simdmath.TanhFloat64(vX)
			vDOut.Mul(vOne.Sub(t.Mul(t))).Store(dx[i:])
		}
		if i < len(dOutput) {
			vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
			//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
			t := simdmath.TanhFloat32(vX) //alt:float32
			//alt:float64 t := simdmath.TanhFloat64(vX)
			vDOut.Mul(vOne.Sub(t.Mul(t))).StorePart(dx[i:])
		}
	}
}

func GeluExactFloat32SIMDVJP(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func GeluExactFloat64SIMDVJP(y, x, dOutput, dx []float64) {
	vHalf := simd.BroadcastFloat32s(0.5) //alt:float32
	//alt:float64 vHalf := simd.BroadcastFloat64s(0.5)
	vOne := simd.BroadcastFloat32s(1.0) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1.0)
	vInvSqrt2 := simd.BroadcastFloat32s(0.7071067811865475) //alt:float32
	//alt:float64 vInvSqrt2 := simd.BroadcastFloat64s(0.7071067811865475)
	vInvSqrt2Pi := simd.BroadcastFloat32s(0.3989422804014327) //alt:float32
	//alt:float64 vInvSqrt2Pi := simd.BroadcastFloat64s(0.3989422804014327)
	vNegHalf := simd.BroadcastFloat32s(-0.5) //alt:float32
	//alt:float64 vNegHalf := simd.BroadcastFloat64s(-0.5)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:]) //alt:float32
		//alt:float64 vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
		//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
		cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(vX.Mul(vInvSqrt2)))) //alt:float32
		//alt:float64 cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat64(vX.Mul(vInvSqrt2))))
		pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(vX.Mul(vX).Mul(vNegHalf))) //alt:float32
		//alt:float64 pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat64(vX.Mul(vX).Mul(vNegHalf)))
		fPrime := cdf.Add(vX.Mul(pdf))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
		//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
		//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat32(vX.Mul(vInvSqrt2)))) //alt:float32
		//alt:float64 cdf := vHalf.Mul(vOne.Add(simdmath.ErfFloat64(vX.Mul(vInvSqrt2))))
		pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat32(vX.Mul(vX).Mul(vNegHalf))) //alt:float32
		//alt:float64 pdf := vInvSqrt2Pi.Mul(simdmath.ExpFloat64(vX.Mul(vX).Mul(vNegHalf)))
		fPrime := cdf.Add(vX.Mul(pdf))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpGeluApproxFloat32SIMD(y, x, dOutput, dx []float32) { //alt:float32
//alt:float64 func vjpGeluApproxFloat64SIMD(y, x, dOutput, dx []float64) {
	vHalf := simd.BroadcastFloat32s(0.5) //alt:float32
	//alt:float64 vHalf := simd.BroadcastFloat64s(0.5)
	vOne := simd.BroadcastFloat32s(1.0) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1.0)
	vSqrt2ByPi := simd.BroadcastFloat32s(0.7978845608) //alt:float32
	//alt:float64 vSqrt2ByPi := simd.BroadcastFloat64s(0.7978845608)
	vConst044715 := simd.BroadcastFloat32s(0.044715) //alt:float32
	//alt:float64 vConst044715 := simd.BroadcastFloat64s(0.044715)
	vConst0134145 := simd.BroadcastFloat32s(0.134145) //alt:float32
	//alt:float64 vConst0134145 := simd.BroadcastFloat64s(0.134145)
	vLen := vHalf.Len()
	i := 0
	for ; i+vLen <= len(dOutput); i += vLen {
		vX := simd.LoadFloat32s(x[i:]) //alt:float32
		//alt:float64 vX := simd.LoadFloat64s(x[i:])
		vDOut := simd.LoadFloat32s(dOutput[i:]) //alt:float32
		//alt:float64 vDOut := simd.LoadFloat64s(dOutput[i:])
		x2 := vX.Mul(vX)
		x3 := x2.Mul(vX)
		u := vSqrt2ByPi.Mul(vX.MulAdd(vConst044715, x3))
		uPrime := vSqrt2ByPi.Mul(x2.MulAdd(vConst0134145, vOne))
		t := simdmath.TanhFloat32(u) //alt:float32
		//alt:float64 t := simdmath.TanhFloat64(u)
		fPrime := vHalf.Mul(vOne.Add(t)).Add(vHalf.Mul(vX).Mul(vOne.Sub(t.Mul(t))).Mul(uPrime))
		vDOut.Mul(fPrime).Store(dx[i:])
	}
	if i < len(dOutput) {
		vX, _ := simd.LoadFloat32sPart(x[i:]) //alt:float32
		//alt:float64 vX, _ := simd.LoadFloat64sPart(x[i:])
		vDOut, _ := simd.LoadFloat32sPart(dOutput[i:]) //alt:float32
		//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[i:])
		x2 := vX.Mul(vX)
		x3 := x2.Mul(vX)
		u := vSqrt2ByPi.Mul(vX.MulAdd(vConst044715, x3))
		uPrime := vSqrt2ByPi.Mul(x2.MulAdd(vConst0134145, vOne))
		t := simdmath.TanhFloat32(u) //alt:float32
		//alt:float64 t := simdmath.TanhFloat64(u)
		fPrime := vHalf.Mul(vOne.Add(t)).Add(vHalf.Mul(vX).Mul(vOne.Sub(t.Mul(t))).Mul(uPrime))
		vDOut.Mul(fPrime).StorePart(dx[i:])
	}
}

func vjpSwiGLUFloat32SIMD(x, dOutput, dx []float32, numRows, hiddenDim int) { //alt:float32
//alt:float64 func vjpSwiGLUFloat64SIMD(x, dOutput, dx []float64, numRows, hiddenDim int) {
	vOne := simd.BroadcastFloat32s(1) //alt:float32
	//alt:float64 vOne := simd.BroadcastFloat64s(1)
	vLen := vOne.Len()
	for m := range numRows {
		xOffset := m * 2 * hiddenDim
		dOutOffset := m * hiddenDim
		j := 0
		for ; j+vLen <= hiddenDim; j += vLen {
			vGate := simd.LoadFloat32s(x[xOffset+j:]) //alt:float32
			//alt:float64 vGate := simd.LoadFloat64s(x[xOffset+j:])
			vVal := simd.LoadFloat32s(x[xOffset+hiddenDim+j:]) //alt:float32
			//alt:float64 vVal := simd.LoadFloat64s(x[xOffset+hiddenDim+j:])
			vDOut := simd.LoadFloat32s(dOutput[dOutOffset+j:]) //alt:float32
			//alt:float64 vDOut := simd.LoadFloat64s(dOutput[dOutOffset+j:])

			s := simdmath.SigmoidFloat32(vGate) //alt:float32
			//alt:float64 s := simdmath.SigmoidFloat64(vGate)
			swish := vGate.Mul(s)
			swishPrime := s.Mul(vOne.Add(vGate.Mul(vOne.Sub(s))))

			vDOut.Mul(vVal).Mul(swishPrime).Store(dx[xOffset+j:])
			vDOut.Mul(swish).Store(dx[xOffset+hiddenDim+j:])
		}
		if j < hiddenDim {
			vGate, _ := simd.LoadFloat32sPart(x[xOffset+j:]) //alt:float32
			//alt:float64 vGate, _ := simd.LoadFloat64sPart(x[xOffset+j:])
			vVal, _ := simd.LoadFloat32sPart(x[xOffset+hiddenDim+j:]) //alt:float32
			//alt:float64 vVal, _ := simd.LoadFloat64sPart(x[xOffset+hiddenDim+j:])
			vDOut, _ := simd.LoadFloat32sPart(dOutput[dOutOffset+j:]) //alt:float32
			//alt:float64 vDOut, _ := simd.LoadFloat64sPart(dOutput[dOutOffset+j:])

			s := simdmath.SigmoidFloat32(vGate) //alt:float32
			//alt:float64 s := simdmath.SigmoidFloat64(vGate)
			swish := vGate.Mul(s)
			swishPrime := s.Mul(vOne.Add(vGate.Mul(vOne.Sub(s))))

			vDOut.Mul(vVal).Mul(swishPrime).StorePart(dx[xOffset+j:])
			vDOut.Mul(swish).StorePart(dx[xOffset+hiddenDim+j:])
		}
	}
}

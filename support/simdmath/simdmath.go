// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package simdmath

import (
	"simd"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

// -----------------------------------------------------------------------------
// Float32 SIMD Math Operations
// -----------------------------------------------------------------------------

// ExpFloat32 computes the element-wise exponential e^x using a degree-7 Cephes
// polynomial with range reduction.
func ExpFloat32(x simd.Float32s) simd.Float32s {
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
	vMaxLog := simd.BroadcastFloat32s(maxLogF)
	vMinLog := simd.BroadcastFloat32s(minLogF)
	vLog2E := simd.BroadcastFloat32s(log2E)
	vHalf := simd.BroadcastFloat32s(0.5)
	vLn2Hi := simd.BroadcastFloat32s(ln2Hi)
	vLn2Lo := simd.BroadcastFloat32s(ln2Lo)

	vP7 := simd.BroadcastFloat32s(p7)
	vP6 := simd.BroadcastFloat32s(p6)
	vP5 := simd.BroadcastFloat32s(p5)
	vP4 := simd.BroadcastFloat32s(p4)
	vP3 := simd.BroadcastFloat32s(p3)
	vP2 := simd.BroadcastFloat32s(p2)
	vOne := simd.BroadcastFloat32s(1.0)
	v127 := simd.BroadcastUint32s(127)

	xClamped := x.Max(vMinLog).Min(vMaxLog)
	z := xClamped.MulAdd(vLog2E, vHalf)
	intZ := z.ConvertToInt32()
	floatZ := intZ.ConvertToFloat32()
	adjMask := floatZ.Greater(z)
	zFloor := floatZ.Sub(vOne.Masked(adjMask))
	intFloor := intZ.Sub(simd.BroadcastInt32s(1).Masked(adjMask))

	g := xClamped.Sub(zFloor.Mul(vLn2Hi)).Sub(zFloor.Mul(vLn2Lo))
	n := intFloor.ToBits().Add(v127).ShiftAllLeft(23).BitsToFloat32()

	poly := vP7.MulAdd(g, vP6)
	poly = poly.MulAdd(g, vP5)
	poly = poly.MulAdd(g, vP4)
	poly = poly.MulAdd(g, vP3)
	poly = poly.MulAdd(g, vP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(vOne)

	return n.Mul(poly)
}

// InvFloat32 computes element-wise reciprocal 1 / x.
func InvFloat32(x simd.Float32s) simd.Float32s {
	vOne := simd.BroadcastFloat32s(1.0)
	return vOne.Div(x)
}

// SqrtFloat32 computes element-wise square root sqrt(x).
func SqrtFloat32(x simd.Float32s) simd.Float32s {
	return x.Sqrt()
}

// RsqrtFloat32 computes element-wise reciprocal square root 1 / sqrt(x).
func RsqrtFloat32(x simd.Float32s) simd.Float32s {
	vOne := simd.BroadcastFloat32s(1.0)
	return vOne.Div(x.Sqrt())
}

// SigmoidFloat32 computes element-wise 1 / (1 + e^-x).
func SigmoidFloat32(x simd.Float32s) simd.Float32s {
	vOne := simd.BroadcastFloat32s(1.0)
	expNeg := ExpFloat32(x.Neg())
	return vOne.Div(vOne.Add(expNeg))
}

// TanhFloat32 computes element-wise hyperbolic tangent tanh(x).
func TanhFloat32(x simd.Float32s) simd.Float32s {
	vNine := simd.BroadcastFloat32s(9.0)
	vNegNine := simd.BroadcastFloat32s(-9.0)
	vTwo := simd.BroadcastFloat32s(2.0)
	vOne := simd.BroadcastFloat32s(1.0)
	vNegOne := simd.BroadcastFloat32s(-1.0)

	clamped := x.Max(vNegNine).Min(vNine)
	exp2x := ExpFloat32(clamped.Mul(vTwo))
	res := vOne.Sub(vTwo.Div(exp2x.Add(vOne)))

	isGeNine := x.GreaterEqual(vNine)
	isLeNegNine := x.LessEqual(vNegNine)

	res = vOne.IfElse(isGeNine, res)
	res = vNegOne.IfElse(isLeNegNine, res)
	return res
}

// ErfFloat32 computes element-wise error function erf(x) using Abramowitz and
// Stegun approximation formula 7.1.26 (maximum error < 1.5e-7).
func ErfFloat32(x simd.Float32s) simd.Float32s {
	vZero := simd.BroadcastFloat32s(0.0)
	vOne := simd.BroadcastFloat32s(1.0)
	vP := simd.BroadcastFloat32s(0.3275911)
	vA1 := simd.BroadcastFloat32s(0.254829592)
	vA2 := simd.BroadcastFloat32s(-0.284496736)
	vA3 := simd.BroadcastFloat32s(1.421413741)
	vA4 := simd.BroadcastFloat32s(-1.453152027)
	vA5 := simd.BroadcastFloat32s(1.061405429)

	absX := x.Abs()
	t := vOne.Div(vOne.Add(vP.Mul(absX)))

	poly := vA5.MulAdd(t, vA4)
	poly = poly.MulAdd(t, vA3)
	poly = poly.MulAdd(t, vA2)
	poly = poly.MulAdd(t, vA1)
	poly = poly.Mul(t)

	expNegX2 := ExpFloat32(absX.Mul(absX).Neg())
	res := vOne.Sub(poly.Mul(expNegX2))

	isNeg := x.Less(vZero)
	return res.Neg().IfElse(isNeg, res)
}

// GeluApproxFloat32 computes element-wise approximate GELU:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func GeluApproxFloat32(x simd.Float32s) simd.Float32s {
	vHalf := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vSqrt2OverPi := simd.BroadcastFloat32s(0.7978845608)
	vCoeff := simd.BroadcastFloat32s(0.044715)

	x3 := x.Mul(x).Mul(x)
	inner := vSqrt2OverPi.Mul(x.Add(vCoeff.Mul(x3)))
	return vHalf.Mul(x).Mul(vOne.Add(TanhFloat32(inner)))
}

// -----------------------------------------------------------------------------
// Float64 SIMD Math Operations
// -----------------------------------------------------------------------------

// ExpFloat64 computes element-wise exponential e^x for float64 vectors using
// branchless SIMD range reduction and a degree-11 Horner polynomial.
func ExpFloat64(x simd.Float64s) simd.Float64s {
	const (
		maxLogF    = 709.7827128933840
		minLogF    = -708.3964185322641
		log2E      = 1.442695040888963407359924681001892137
		ln2Hi      = 0.693147180559945309417232121458176568
		ln2Lo      = 2.31904681384629955841777123493867e-17
		magicRound = 6755399441055744.0 // 1.5 * 2^52
		magicBits  = 0x4338000000000000 // IEEE 754 bits of 1.5 * 2^52

		c11 = 2.505210838544172e-8
		c10 = 2.755731922398589e-7
		c9  = 2.755731922398589e-6
		c8  = 2.480158730158730e-5
		c7  = 1.984126984126984e-4
		c6  = 1.388888888888889e-3
		c5  = 8.333333333333333e-3
		c4  = 4.166666666666667e-2
		c3  = 1.666666666666667e-1
		c2  = 5.000000000000000e-1
	)
	vMaxLog := simd.BroadcastFloat64s(maxLogF)
	vMinLog := simd.BroadcastFloat64s(minLogF)
	vLog2E := simd.BroadcastFloat64s(log2E)
	vLn2Hi := simd.BroadcastFloat64s(ln2Hi)
	vLn2Lo := simd.BroadcastFloat64s(ln2Lo)
	vMagic := simd.BroadcastFloat64s(magicRound)
	vMagicBits := simd.BroadcastUint64s(magicBits)
	vOne := simd.BroadcastFloat64s(1.0)
	v1023 := simd.BroadcastUint64s(1023)

	vC11 := simd.BroadcastFloat64s(c11)
	vC10 := simd.BroadcastFloat64s(c10)
	vC9 := simd.BroadcastFloat64s(c9)
	vC8 := simd.BroadcastFloat64s(c8)
	vC7 := simd.BroadcastFloat64s(c7)
	vC6 := simd.BroadcastFloat64s(c6)
	vC5 := simd.BroadcastFloat64s(c5)
	vC4 := simd.BroadcastFloat64s(c4)
	vC3 := simd.BroadcastFloat64s(c3)
	vC2 := simd.BroadcastFloat64s(c2)

	xClamped := x.Max(vMinLog).Min(vMaxLog)
	z := xClamped.Mul(vLog2E)

	// Round to nearest integer using IEEE-754 mantissa alignment.
	zRound := z.Add(vMagic)
	nFloat := zRound.Sub(vMagic)

	// Integer n in bits via two's complement subtraction.
	nInt := zRound.ToBits().Sub(vMagicBits)
	scale := nInt.Add(v1023).ShiftAllLeft(52).BitsToFloat64()

	// Reduced argument g = x - n*ln(2).
	g := xClamped.Sub(nFloat.Mul(vLn2Hi)).Sub(nFloat.Mul(vLn2Lo))

	// Horner polynomial for e^g.
	poly := vC11.MulAdd(g, vC10)
	poly = poly.MulAdd(g, vC9)
	poly = poly.MulAdd(g, vC8)
	poly = poly.MulAdd(g, vC7)
	poly = poly.MulAdd(g, vC6)
	poly = poly.MulAdd(g, vC5)
	poly = poly.MulAdd(g, vC4)
	poly = poly.MulAdd(g, vC3)
	poly = poly.MulAdd(g, vC2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(vOne)

	return scale.Mul(poly)
}

// InvFloat64 computes element-wise reciprocal 1 / x.
func InvFloat64(x simd.Float64s) simd.Float64s {
	vOne := simd.BroadcastFloat64s(1.0)
	return vOne.Div(x)
}

// SqrtFloat64 computes element-wise square root sqrt(x).
func SqrtFloat64(x simd.Float64s) simd.Float64s {
	return x.Sqrt()
}

// RsqrtFloat64 computes element-wise reciprocal square root 1 / sqrt(x).
func RsqrtFloat64(x simd.Float64s) simd.Float64s {
	vOne := simd.BroadcastFloat64s(1.0)
	return vOne.Div(x.Sqrt())
}

// SigmoidFloat64 computes element-wise 1 / (1 + e^-x).
func SigmoidFloat64(x simd.Float64s) simd.Float64s {
	vOne := simd.BroadcastFloat64s(1.0)
	expNeg := ExpFloat64(x.Neg())
	return vOne.Div(vOne.Add(expNeg))
}

// TanhFloat64 computes element-wise hyperbolic tangent tanh(x).
func TanhFloat64(x simd.Float64s) simd.Float64s {
	vNine := simd.BroadcastFloat64s(19.0)
	vNegNine := simd.BroadcastFloat64s(-19.0)
	vTwo := simd.BroadcastFloat64s(2.0)
	vOne := simd.BroadcastFloat64s(1.0)
	vNegOne := simd.BroadcastFloat64s(-1.0)

	clamped := x.Max(vNegNine).Min(vNine)
	exp2x := ExpFloat64(clamped.Mul(vTwo))
	res := vOne.Sub(vTwo.Div(exp2x.Add(vOne)))

	isGeNine := x.GreaterEqual(vNine)
	isLeNegNine := x.LessEqual(vNegNine)

	res = vOne.IfElse(isGeNine, res)
	res = vNegOne.IfElse(isLeNegNine, res)
	return res
}

// ErfFloat64 computes element-wise error function erf(x).
func ErfFloat64(x simd.Float64s) simd.Float64s {
	vZero := simd.BroadcastFloat64s(0.0)
	vOne := simd.BroadcastFloat64s(1.0)
	vP := simd.BroadcastFloat64s(0.3275911)
	vA1 := simd.BroadcastFloat64s(0.254829592)
	vA2 := simd.BroadcastFloat64s(-0.284496736)
	vA3 := simd.BroadcastFloat64s(1.421413741)
	vA4 := simd.BroadcastFloat64s(-1.453152027)
	vA5 := simd.BroadcastFloat64s(1.061405429)

	absX := x.Abs()
	t := vOne.Div(vOne.Add(vP.Mul(absX)))

	poly := vA5.MulAdd(t, vA4)
	poly = poly.MulAdd(t, vA3)
	poly = poly.MulAdd(t, vA2)
	poly = poly.MulAdd(t, vA1)
	poly = poly.Mul(t)

	expNegX2 := ExpFloat64(absX.Mul(absX).Neg())
	res := vOne.Sub(poly.Mul(expNegX2))

	isNeg := x.Less(vZero)
	return res.Neg().IfElse(isNeg, res)
}

// GeluApproxFloat64 computes element-wise approximate GELU:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func GeluApproxFloat64(x simd.Float64s) simd.Float64s {
	vHalf := simd.BroadcastFloat64s(0.5)
	vOne := simd.BroadcastFloat64s(1.0)
	vSqrt2OverPi := simd.BroadcastFloat64s(0.7978845608028654)
	vCoeff := simd.BroadcastFloat64s(0.044715)

	x3 := x.Mul(x).Mul(x)
	inner := vSqrt2OverPi.Mul(x.Add(vCoeff.Mul(x3)))
	return vHalf.Mul(x).Mul(vOne.Add(TanhFloat64(inner)))
}

// -----------------------------------------------------------------------------
// BFloat16 and Float16 Vector Adapters
// -----------------------------------------------------------------------------

// LoadBFloat16s loads a full vector of bfloat16 by upscaling to Float32s.
func LoadBFloat16s(s []bfloat16.BFloat16) simd.Float32s {
	var buf [16]float32
	n := simd.BroadcastFloat32s(0).Len()
	for i := 0; i < n; i++ {
		buf[i] = s[i].Float32()
	}
	return simd.LoadFloat32s(buf[:n])
}

// LoadBFloat16sPart loads a partial vector of bfloat16 upscaled to Float32s.
func LoadBFloat16sPart(s []bfloat16.BFloat16) (simd.Float32s, int) {
	var buf [16]float32
	vLen := simd.BroadcastFloat32s(0).Len()
	n := min(len(s), vLen)
	for i := 0; i < n; i++ {
		buf[i] = s[i].Float32()
	}
	v, loaded := simd.LoadFloat32sPart(buf[:n])
	return v, loaded
}

// StoreBFloat16s downscales a Float32s vector and stores into s.
func StoreBFloat16s(s []bfloat16.BFloat16, v simd.Float32s) {
	var buf [16]float32
	n := v.Len()
	v.Store(buf[:n])
	for i := 0; i < n; i++ {
		s[i] = bfloat16.FromFloat32(buf[i])
	}
}

// StoreBFloat16sPart downscales and stores the partial elements of v into s.
func StoreBFloat16sPart(s []bfloat16.BFloat16, v simd.Float32s) int {
	var buf [16]float32
	vLen := v.Len()
	n := min(len(s), vLen)
	v.Store(buf[:vLen])
	for i := 0; i < n; i++ {
		s[i] = bfloat16.FromFloat32(buf[i])
	}
	return n
}

// LoadFloat16s loads a full vector of float16 by upscaling to Float32s.
func LoadFloat16s(s []float16.Float16) simd.Float32s {
	var buf [16]float32
	n := simd.BroadcastFloat32s(0).Len()
	for i := 0; i < n; i++ {
		buf[i] = s[i].Float32()
	}
	return simd.LoadFloat32s(buf[:n])
}

// LoadFloat16sPart loads a partial vector of float16 upscaled to Float32s.
func LoadFloat16sPart(s []float16.Float16) (simd.Float32s, int) {
	var buf [16]float32
	vLen := simd.BroadcastFloat32s(0).Len()
	n := min(len(s), vLen)
	for i := 0; i < n; i++ {
		buf[i] = s[i].Float32()
	}
	v, loaded := simd.LoadFloat32sPart(buf[:n])
	return v, loaded
}

// StoreFloat16s downscales a Float32s vector and stores into s.
func StoreFloat16s(s []float16.Float16, v simd.Float32s) {
	var buf [16]float32
	n := v.Len()
	v.Store(buf[:n])
	for i := 0; i < n; i++ {
		s[i] = float16.FromFloat32(buf[i])
	}
}

// StoreFloat16sPart downscales and stores the partial elements of v into s.
func StoreFloat16sPart(s []float16.Float16, v simd.Float32s) int {
	var buf [16]float32
	vLen := v.Len()
	n := min(len(s), vLen)
	v.Store(buf[:vLen])
	for i := 0; i < n; i++ {
		s[i] = float16.FromFloat32(buf[i])
	}
	return n
}

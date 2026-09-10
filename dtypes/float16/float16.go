// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package float16 implements the IEEE 754 half-precision floating-point format (binary16).
package float16

import (
	"math"
	"strconv"
)

// Float16 (half-precision floating-point) format is a computer number format occupying 16 bits in
// computer memory; it represents a wide dynamic range of numeric values by using a floating radix point.
// This format is the IEEE 754 half-precision floating-point format (binary16).
type Float16 uint16

// Float32 converts the Float16 to a float32.
func (f Float16) Float32() float32 {
	const (
		signMask     = 0x8000
		expMask      = 0x7C00
		mantissaMask = 0x03FF
	)

	sign := uint32(f&signMask) << 16
	exp := int((f & expMask) >> 10)
	mantissa := uint32(f & mantissaMask)

	if exp == 0x1F {
		// NaN or Infinity
		if mantissa == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7F800000 | (mantissa << 13))
	}

	if exp == 0 {
		if mantissa == 0 {
			// Zero
			return math.Float32frombits(sign)
		}
		// Denormalized number
		// Normalize it by shifting the mantissa until the leading bit is at position 10.
		for (mantissa & 0x0400) == 0 {
			mantissa <<= 1
			exp--
		}
		exp++
		mantissa &= mantissaMask
	}

	// Normalized number
	exp32 := uint32(exp-15+127) << 23
	mantissa32 := mantissa << 13
	return math.Float32frombits(sign | exp32 | mantissa32)
}

// SetFloat32 sets the values from a float32.
func (f *Float16) SetFloat32(v float32) {
	*f = FromFloat32(v)
}

// Float64 converts the Float16 to a float64.
func (f Float16) Float64() float64 {
	return float64(f.Float32())
}

// SetFloat64 sets the values from a float64.
func (f *Float16) SetFloat64(v float64) {
	*f = FromFloat64(v)
}

type signedTypes interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type unsignedTypes interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

type floatTypes interface {
	~float32 | ~float64
}

type numberTypes interface {
	signedTypes | unsignedTypes | floatTypes
}

// From converts the value from a standard numeric type to Float16.
func From[T numberTypes](value T) Float16 {
	return FromFloat32(float32(value))
}

// FromNumbers converts a variadic list of numeric values to a []Float16.
func FromNumbers[T numberTypes](values ...T) []Float16 {
	out := make([]Float16, len(values))
	for i, v := range values {
		out[i] = From(v)
	}
	return out
}

// FromFloat32 converts a float32 to a Float16.
func FromFloat32(x float32) Float16 {
	bits := math.Float32bits(x)
	sign := uint16((bits >> 16) & 0x8000)
	uAbs := bits & 0x7FFFFFFF

	const f16MantissaMask = 0x03FF

	if uAbs >= 0x47800000 { // >= 65536.0f overflows to Inf or is NaN
		if uAbs > 0x7F800000 {
			// NaN: preserve payload bits and ensure non-zero mantissa
			return Float16(sign | 0x7C00 | uint16((uAbs>>13)&f16MantissaMask) | 1)
		}
		// Infinity
		return Float16(sign | 0x7C00)
	}

	if uAbs < 0x38800000 { // < 2^-14: subnormal or zero
		if uAbs < 0x33000000 { // < 2^-25: rounds to zero
			return Float16(sign)
		}
		fAbs := math.Float32frombits(uAbs)
		const denormMagicBits = 126 << 23
		fDenorm := fAbs + math.Float32frombits(denormMagicBits)
		denormMant := uint16((math.Float32bits(fDenorm) - denormMagicBits) & f16MantissaMask)
		return Float16(sign | denormMant)
	}

	// Normalized: round to nearest even
	bias := uint32(0x0FFF) + ((uAbs >> 13) & 1)
	rounded := uAbs + bias
	normExpAdjusted := rounded - (112 << 23)
	if normExpAdjusted >= (0x1F << 23) {
		// Overflow to infinity
		return Float16(sign | 0x7C00)
	}
	return Float16(sign | uint16(normExpAdjusted>>13))
}

// FromFloat64 converts a float64 to a Float16.
func FromFloat64(x float64) Float16 {
	return FromFloat32(float32(x))
}

// FromFloat32s converts a variadic list of float32s to a []Float16.
func FromFloat32s(values ...float32) []Float16 {
	out := make([]Float16, len(values))
	for i, v := range values {
		out[i] = FromFloat32(v)
	}
	return out
}

// FromBits convert an uint16 to a Float16.
func FromBits(uint16 uint16) Float16 {
	return Float16(uint16)
}

// Bits convert Float16 to an uint16.
func (f Float16) Bits() uint16 {
	return uint16(f)
}

// String implements fmt.Stringer, and prints a float representation of the Float16.
func (f Float16) String() string {
	return strconv.FormatFloat(float64(f.Float32()), 'f', -1, 32)
}

// Inf returns a Float16 with an infinity value with the specified sign.
// A sign >= returns positive infinity.
// A sign < 0 returns negative infinity.
func Inf(sign int) Float16 {
	if sign >= 0 {
		return Float16(0x7C00)
	}
	return Float16(0xFC00)
}

// NaN returns a Float16 with a NaN value.
func NaN() Float16 {
	return Float16(0x7C01)
}

// Neg returns the negative of the current value.
func (f Float16) Neg() Float16 {
	return f ^ 0x8000
}

// SmallestNonzero is the smallest nonzero denormal value for float16.
// For IEEE 754 binary16, the smallest non-zero denormalized value is 2^-24, which is approximately 5.96e-8.
const SmallestNonzero = Float16(0x0001)

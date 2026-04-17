// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package float16 is a trivial implementation for the IEEE 754 half-precision floating-point format (binary16),
// based on https://github.com/x448/float16
package float16

import (
	"math"
	"strconv"

	"github.com/x448/float16"
)

// Float16 (half-precision floating-point) format is a computer number format occupying 16 bits in
// computer memory; it represents a wide dynamic range of numeric values by using a floating radix point.
// This format is the IEEE 754 half-precision floating-point format (binary16).
type Float16 uint16

// Float32 converts the Float16 to a float32.
func (f Float16) Float32() float32 {
	return float16.Frombits(uint16(f)).Float32()
}

// Float64 converts the Float16 to a float64.
func (f Float16) Float64() float64 {
	return float64(f.Float32())
}

// FromFloat32 converts a float32 to a Float16.
func FromFloat32(x float32) Float16 {
	return Float16(float16.Fromfloat32(x).Bits())
}

// FromFloat64 converts a float64 to a Float16.
func FromFloat64(x float64) Float16 {
	return FromFloat32(float32(x))
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
	return FromFloat32(float32(math.Inf(sign)))
}

// NaN returns a Float16 with a NaN value.
func NaN() Float16 {
	return FromFloat64(math.NaN())
}

// SmallestNonzero is the smallest nonzero denormal value for float16.
// For IEEE 754 binary16, the smallest non-zero denormalized value is 2^-24, which is approximately 5.96e-8.
const SmallestNonzero = Float16(0x0001)

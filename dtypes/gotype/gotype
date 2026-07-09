// Package gotype defines several constraints based on Go types supported by the dtypes package.
package gotype

import (
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

// Supported constraints to the list of compute's supported Go types, including those not native (half-precision).
//
// Notice Go's `int` type is supported but not portable, since it may translate to dtypes Int32 or Int64 depending
// on the platform. You should prefer using Int32 or Int64 instead of int.
type Supported interface {
	bool | AnyHalfPrecision | Numeric
}

// Numeric constraints to the native Go numeric types.
// It includes complex numbers.
//
// It doesn't include half-precision types float16.Float16 or bfloat16.BFloat16 because they are not native number types.
type Numeric interface {
	float32 | float64 | int | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | complex64 | complex128
}

// Integer constraints to Go native integer types.
type Integer interface {
	int | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// Signed constraints to Go signed integer types.
type Signed interface {
	int | int8 | int16 | int32 | int64
}

// Unsigned constraints to Go unsigned integer types.
type Unsigned interface {
	uint8 | uint16 | uint32 | uint64
}

// NumericNotComplex constraints to native Go numeric types excluding complex numbers.
//
// See also Numeric.
type NumericNotComplex interface {
	float32 | float64 | int | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// Complex constraints to the Go complex types.
type Complex interface {
	complex64 | complex128
}

// AnyHalfPrecision constraints to the compute's representations for half-precision floating point numbers.
// They are not natively supported by Go, but rather aliases unit16 with extra methods (Float64, Float32, etc).
//
// Consider using HalfPrecision instead.
type AnyHalfPrecision interface {
	float16.Float16 | bfloat16.BFloat16
}

// Float constraints to continuous native Go numeric types.
// It doesn't include complex numbers or half-precision types (non-native).
type Float interface {
	float32 | float64
}

// HalfPrecision is an interface that represents half-precision floating point numbers,
// specifically float16 and bfloat16.
//
// It includes the methods to convert to float64 and float32, so it can be used in generic methods.
type HalfPrecision[T any] interface {
	AnyHalfPrecision
	Float64() float64
	Float32() float32
	Neg() T
}

// HalfPrecisionPtr is a pointer to a HalfPrecision wrapper type.
// It is used when one needs to set the value of a HalfPrecision type from a float32 or float64.
type HalfPrecisionPtr[T HalfPrecision[T]] interface {
	*T
	SetFloat32(float32)
	SetFloat64(float64)
}

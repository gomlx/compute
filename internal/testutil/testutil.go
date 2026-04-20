// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil

import (
	"fmt"
	"math"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/google/go-cmp/cmp"
)

// Must panics if there is an error.
func Must(err error) {
	if err != nil {
		panic(err)
	}
}

// Must1 panics if there is an error and returns the value.
func Must1[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func withinDeltaBase[T ~float32 | ~float64](a, b T, delta float64) bool {
	return math.Abs(float64(a-b)) < delta
}

type halfFloat interface {
	float16.Float16 | bfloat16.BFloat16
	Float64() float64
}

func withinDeltaHalfPrecision[T halfFloat](a, b T, delta float64) bool {
	return withinDeltaBase(a.Float64(), b.Float64(), delta)
}

// IsInDelta reports if want and got are equal within a given absolute delta.
// If they are not equal, it returns the diff using the format "-want +got").
func IsInDelta(want, got any, delta float64) (ok bool, diff string) {
	if tA, okA := want.(*tensors.Tensor); okA {
		// Special case tensors:
		tB, okB := got.(*tensors.Tensor)
		if !okB {
			return false, fmt.Sprintf("values have different types: %T and %T", want, got)
		}
		tA.ConstFlatData(func(tAFlat any) {
			tB.ConstFlatData(func(tBFlat any) {
				ok, diff = IsInDelta(tAFlat, tBFlat, delta)
			})
		})
		return ok, diff
	}

	opts := []cmp.Option{
		cmp.Comparer(func(a, b float32) bool { return withinDeltaBase(a, b, delta) }),
		cmp.Comparer(func(a, b float64) bool { return withinDeltaBase(a, b, delta) }),
		cmp.Comparer(func(a, b float16.Float16) bool { return withinDeltaHalfPrecision(a, b, delta) }),
		cmp.Comparer(func(a, b bfloat16.BFloat16) bool { return withinDeltaHalfPrecision(a, b, delta) }),
	}
	if cmp.Equal(want, got, opts...) {
		return true, ""
	}
	return false, cmp.Diff(want, got, opts...)
}

func withinRelativeDeltaBase[T ~float32 | ~float64](a, b T, relDelta float64) bool {
	delta := math.Abs(float64(a) - float64(b))
	mean := math.Abs(float64(b)) + math.Abs(float64(b))
	return delta/mean < relDelta
}

func withinRelativeDeltaHalfPrecision[T halfFloat](a, b T, relDelta float64) bool {
	return withinRelativeDeltaBase(a.Float64(), b.Float64(), relDelta)
}

// IsInRelativeDelta reports if want and got are equal within a given relative delta.
// If they are not equal, it returns the diff using the format "-want +got").
func IsInRelativeDelta(want, got any, relDelta float64) (ok bool, diff string) {
	if tA, okA := want.(*tensors.Tensor); okA {
		// Special case tensors:
		tB, okB := got.(*tensors.Tensor)
		if !okB {
			return false, fmt.Sprintf("values have different types: %T and %T", want, got)
		}
		tA.ConstFlatData(func(tAFlat any) {
			tB.ConstFlatData(func(tBFlat any) {
				ok, diff = IsInRelativeDelta(tAFlat, tBFlat, relDelta)
			})
		})
		return ok, diff
	}

	opts := []cmp.Option{
		cmp.Comparer(func(a, b float32) bool { return withinRelativeDeltaBase(a, b, relDelta) }),
		cmp.Comparer(func(a, b float64) bool { return withinRelativeDeltaBase(a, b, relDelta) }),
		cmp.Comparer(func(a, b float16.Float16) bool { return withinRelativeDeltaHalfPrecision(a, b, relDelta) }),
		cmp.Comparer(func(a, b bfloat16.BFloat16) bool { return withinRelativeDeltaHalfPrecision(a, b, relDelta) }),
	}
	if cmp.Equal(want, got, opts...) {
		return true, ""
	}
	return false, cmp.Diff(want, got, opts...)
}

// Try whether f panics, and also returns the panic value.
func Try(f func()) (panicked bool, reason any) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
			reason = r
		}
	}()
	f()
	return
}

// IsEqual returns whether want and got are equal using go-cmp.
// If they are not equal, it returns the diff using the format "-want +got").
func IsEqual(want, got any) (ok bool, diff string) {
	if tA, ok := want.(*tensors.Tensor); ok {
		tB, ok := got.(*tensors.Tensor)
		if !ok {
			return false, fmt.Sprintf("values have different types: %T and %T", want, got)
		}
		tA.ConstFlatData(func(tAFlat any) {
			tB.ConstFlatData(func(tBFlat any) {
				ok, diff = IsEqual(tAFlat, tBFlat)
			})
		})
		return ok, diff
	}

	if cmp.Equal(want, got) {
		return true, ""
	}
	return false, cmp.Diff(want, got)
}

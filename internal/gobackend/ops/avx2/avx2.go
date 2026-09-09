// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX2Allowed() {
		registerAVX2()
	}
}

func registerAVX2() {
	gobackend.SetReduceTrailingSumArchDispatcher(gobackend.PriorityArch, func(operand, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		inPtr := operand.UnsafePointer()
		outPtr := output.UnsafePointer()
		if inPtr == nil || outPtr == nil {
			return false
		}
		return DispatchTrailingSumAVX2(inPtr, outPtr, A, B, dtype)
	})
}

//go:noescape
func reduceTrailingSumFloat32AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumFloat64AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumFloat16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumBFloat16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt32AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint32AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt8AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint8AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt64AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint64AVX2(in, out unsafe.Pointer, A, B int)

// DispatchTrailingSumAVX2 dispatches trailing sum reduction to AVX2 assembly kernels.
// Returns true if handled, false if dtype is unsupported.
func DispatchTrailingSumAVX2(inPtr, outPtr unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	switch dtype {
	case dtypes.Float32:
		reduceTrailingSumFloat32AVX2(inPtr, outPtr, A, B)
	case dtypes.Float64:
		reduceTrailingSumFloat64AVX2(inPtr, outPtr, A, B)
	case dtypes.Float16:
		reduceTrailingSumFloat16AVX2(inPtr, outPtr, A, B)
	case dtypes.BFloat16:
		reduceTrailingSumBFloat16AVX2(inPtr, outPtr, A, B)
	case dtypes.Int32:
		reduceTrailingSumInt32AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint32:
		reduceTrailingSumUint32AVX2(inPtr, outPtr, A, B)
	case dtypes.Int64:
		reduceTrailingSumInt64AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint64:
		reduceTrailingSumUint64AVX2(inPtr, outPtr, A, B)
	case dtypes.Int16:
		reduceTrailingSumInt16AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint16:
		reduceTrailingSumUint16AVX2(inPtr, outPtr, A, B)
	case dtypes.Int8:
		reduceTrailingSumInt8AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint8:
		reduceTrailingSumUint8AVX2(inPtr, outPtr, A, B)
	default:
		return false
	}
	return true
}

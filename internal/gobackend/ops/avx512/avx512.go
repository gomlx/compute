// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX512Allowed() {
		registerAVX512()
	}
}

func registerAVX512() {
	gobackend.SetReduceTrailingSumArchDispatcher(gobackend.PriorityArch+1, func(operand, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		inPtr := operand.UnsafePointer()
		outPtr := output.UnsafePointer()
		if inPtr == nil || outPtr == nil {
			return false
		}
		return DispatchTrailingSumAVX512(inPtr, outPtr, A, B, dtype)
	})
}

//go:noescape
func reduceTrailingSumFloat32AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumFloat64AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumFloat16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumBFloat16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt32AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint32AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt8AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint8AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumInt64AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumUint64AVX512(in, out unsafe.Pointer, A, B int)

// DispatchTrailingSumAVX512 dispatches trailing sum reduction to AVX-512 assembly kernels.
// Returns true if handled, false if dtype is unsupported.
func DispatchTrailingSumAVX512(inPtr, outPtr unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	switch dtype {
	case dtypes.Float32:
		reduceTrailingSumFloat32AVX512(inPtr, outPtr, A, B)
	case dtypes.Float64:
		reduceTrailingSumFloat64AVX512(inPtr, outPtr, A, B)
	case dtypes.Float16:
		reduceTrailingSumFloat16AVX512(inPtr, outPtr, A, B)
	case dtypes.BFloat16:
		reduceTrailingSumBFloat16AVX512(inPtr, outPtr, A, B)
	case dtypes.Int32:
		reduceTrailingSumInt32AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint32:
		reduceTrailingSumUint32AVX512(inPtr, outPtr, A, B)
	case dtypes.Int64:
		reduceTrailingSumInt64AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint64:
		reduceTrailingSumUint64AVX512(inPtr, outPtr, A, B)
	case dtypes.Int16:
		reduceTrailingSumInt16AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint16:
		reduceTrailingSumUint16AVX512(inPtr, outPtr, A, B)
	case dtypes.Int8:
		reduceTrailingSumInt8AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint8:
		reduceTrailingSumUint8AVX512(inPtr, outPtr, A, B)
	default:
		return false
	}
	return true
}

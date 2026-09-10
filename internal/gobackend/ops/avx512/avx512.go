// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX512Allowed {
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
	gobackend.SetReduceLeadingSumArchDispatcher(gobackend.PriorityArch+1, func(operand, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		inPtr := operand.UnsafePointer()
		outPtr := output.UnsafePointer()
		if inPtr == nil || outPtr == nil {
			return false
		}
		return DispatchLeadingSumAVX512(inPtr, outPtr, A, B, dtype)
	})
	gobackend.SetBinaryTrailingArchDispatcher(gobackend.PriorityArch+1, func(op compute.OpType, isLHS bool, lhs, rhs, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		lhsPtr := lhs.UnsafePointer()
		rhsPtr := rhs.UnsafePointer()
		outPtr := output.UnsafePointer()
		if lhsPtr == nil || rhsPtr == nil || outPtr == nil {
			return false
		}
		return DispatchBinaryTrailingAVX512(op, isLHS, lhsPtr, rhsPtr, outPtr, A, B, dtype)
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

//go:noescape
func reduceLeadingSumFloat32AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumFloat64AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumFloat16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumBFloat16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt32AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint32AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint16AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt8AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint8AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt64AVX512(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint64AVX512(in, out unsafe.Pointer, A, B int)

// DispatchLeadingSumAVX512 dispatches leading sum reduction to AVX-512 assembly kernels.
// Returns true if handled, false if dtype is unsupported.
func DispatchLeadingSumAVX512(inPtr, outPtr unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	switch dtype {
	case dtypes.Float32:
		reduceLeadingSumFloat32AVX512(inPtr, outPtr, A, B)
	case dtypes.Float64:
		reduceLeadingSumFloat64AVX512(inPtr, outPtr, A, B)
	case dtypes.Float16:
		reduceLeadingSumFloat16AVX512(inPtr, outPtr, A, B)
	case dtypes.BFloat16:
		reduceLeadingSumBFloat16AVX512(inPtr, outPtr, A, B)
	case dtypes.Int32:
		reduceLeadingSumInt32AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint32:
		reduceLeadingSumUint32AVX512(inPtr, outPtr, A, B)
	case dtypes.Int64:
		reduceLeadingSumInt64AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint64:
		reduceLeadingSumUint64AVX512(inPtr, outPtr, A, B)
	case dtypes.Int16:
		reduceLeadingSumInt16AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint16:
		reduceLeadingSumUint16AVX512(inPtr, outPtr, A, B)
	case dtypes.Int8:
		reduceLeadingSumInt8AVX512(inPtr, outPtr, A, B)
	case dtypes.Uint8:
		reduceLeadingSumUint8AVX512(inPtr, outPtr, A, B)
	default:
		return false
	}
	return true
}

// ----------------------------------------------------------------------------
// Binary Trailing Kernels Declarations
// ----------------------------------------------------------------------------

//go:noescape
func binaryTrailingAddFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMulFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivRHSFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivLHSFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinFloat32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingAddFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMulFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivRHSFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivLHSFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinFloat64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingAddInt32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSInt32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSInt32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMulInt32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxInt32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinInt32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxUint32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinUint32AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingAddInt64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSInt64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSInt64AVX512(lhs, rhs, out unsafe.Pointer, A, B int)

// DispatchBinaryTrailingAVX512 dispatches binary trailing broadcast operations to AVX-512 assembly kernels.
// Returns true if handled, false if unsupported.
func DispatchBinaryTrailingAVX512(op compute.OpType, isLHS bool, lhs, rhs, out unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	vec := lhs
	scalar := rhs
	if isLHS {
		vec = rhs
		scalar = lhs
	}

	switch op {
	case compute.OpTypeAdd:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingAddFloat32AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingAddFloat64AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Int32, dtypes.Uint32:
			binaryTrailingAddInt32AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Int64, dtypes.Uint64:
			binaryTrailingAddInt64AVX512(vec, scalar, out, A, B)
			return true
		}
	case compute.OpTypeSub:
		if !isLHS {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingSubRHSFloat32AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingSubRHSFloat64AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Int32, dtypes.Uint32:
				binaryTrailingSubRHSInt32AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Int64, dtypes.Uint64:
				binaryTrailingSubRHSInt64AVX512(vec, scalar, out, A, B)
				return true
			}
		} else {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingSubLHSFloat32AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingSubLHSFloat64AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Int32, dtypes.Uint32:
				binaryTrailingSubLHSInt32AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Int64, dtypes.Uint64:
				binaryTrailingSubLHSInt64AVX512(vec, scalar, out, A, B)
				return true
			}
		}
	case compute.OpTypeMul:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMulFloat32AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMulFloat64AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Int32, dtypes.Uint32:
			binaryTrailingMulInt32AVX512(vec, scalar, out, A, B)
			return true
		}
	case compute.OpTypeDiv:
		if !isLHS {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingDivRHSFloat32AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingDivRHSFloat64AVX512(vec, scalar, out, A, B)
				return true
			}
		} else {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingDivLHSFloat32AVX512(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingDivLHSFloat64AVX512(vec, scalar, out, A, B)
				return true
			}
		}
	case compute.OpTypeMax:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMaxFloat32AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMaxFloat64AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Int32:
			binaryTrailingMaxInt32AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Uint32:
			binaryTrailingMaxUint32AVX512(vec, scalar, out, A, B)
			return true
		}
	case compute.OpTypeMin:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMinFloat32AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMinFloat64AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Int32:
			binaryTrailingMinInt32AVX512(vec, scalar, out, A, B)
			return true
		case dtypes.Uint32:
			binaryTrailingMinUint32AVX512(vec, scalar, out, A, B)
			return true
		}
	}
	return false
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX2Allowed {
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
	gobackend.SetReduceLeadingSumArchDispatcher(gobackend.PriorityArch, func(operand, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		inPtr := operand.UnsafePointer()
		outPtr := output.UnsafePointer()
		if inPtr == nil || outPtr == nil {
			return false
		}
		return DispatchLeadingSumAVX2(inPtr, outPtr, A, B, dtype)
	})
	gobackend.SetBinaryTrailingArchDispatcher(gobackend.PriorityArch, func(op compute.OpType, isLHS bool, lhs, rhs, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		lhsPtr := lhs.UnsafePointer()
		rhsPtr := rhs.UnsafePointer()
		outPtr := output.UnsafePointer()
		if lhsPtr == nil || rhsPtr == nil || outPtr == nil {
			return false
		}
		return DispatchBinaryTrailingAVX2(op, isLHS, lhsPtr, rhsPtr, outPtr, A, B, dtype)
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

//go:noescape
func reduceLeadingSumFloat32AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumFloat64AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumFloat16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumBFloat16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt32AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint32AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint16AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt8AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint8AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumInt64AVX2(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumUint64AVX2(in, out unsafe.Pointer, A, B int)

// DispatchLeadingSumAVX2 dispatches leading sum reduction to AVX2 assembly kernels.
// Returns true if handled, false if dtype is unsupported.
func DispatchLeadingSumAVX2(inPtr, outPtr unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	switch dtype {
	case dtypes.Float32:
		reduceLeadingSumFloat32AVX2(inPtr, outPtr, A, B)
	case dtypes.Float64:
		reduceLeadingSumFloat64AVX2(inPtr, outPtr, A, B)
	case dtypes.Float16:
		reduceLeadingSumFloat16AVX2(inPtr, outPtr, A, B)
	case dtypes.BFloat16:
		reduceLeadingSumBFloat16AVX2(inPtr, outPtr, A, B)
	case dtypes.Int32:
		reduceLeadingSumInt32AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint32:
		reduceLeadingSumUint32AVX2(inPtr, outPtr, A, B)
	case dtypes.Int64:
		reduceLeadingSumInt64AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint64:
		reduceLeadingSumUint64AVX2(inPtr, outPtr, A, B)
	case dtypes.Int16:
		reduceLeadingSumInt16AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint16:
		reduceLeadingSumUint16AVX2(inPtr, outPtr, A, B)
	case dtypes.Int8:
		reduceLeadingSumInt8AVX2(inPtr, outPtr, A, B)
	case dtypes.Uint8:
		reduceLeadingSumUint8AVX2(inPtr, outPtr, A, B)
	default:
		return false
	}
	return true
}

// ----------------------------------------------------------------------------
// Binary Trailing Kernels Declarations
// ----------------------------------------------------------------------------

//go:noescape
func binaryTrailingAddFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMulFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivRHSFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivLHSFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinFloat32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingAddFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMulFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivRHSFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingDivLHSFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinFloat64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingAddInt32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSInt32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSInt32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMulInt32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxInt32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinInt32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMaxUint32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingMinUint32AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingAddInt64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubRHSInt64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

//go:noescape
func binaryTrailingSubLHSInt64AVX2(lhs, rhs, out unsafe.Pointer, A, B int)

// DispatchBinaryTrailingAVX2 dispatches binary trailing broadcast operations to AVX2 assembly kernels.
// Returns true if handled, false if unsupported.
func DispatchBinaryTrailingAVX2(op compute.OpType, isLHS bool, lhs, rhs, out unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
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
			binaryTrailingAddFloat32AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingAddFloat64AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Int32, dtypes.Uint32:
			binaryTrailingAddInt32AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Int64, dtypes.Uint64:
			binaryTrailingAddInt64AVX2(vec, scalar, out, A, B)
			return true
		}
	case compute.OpTypeSub:
		if !isLHS {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingSubRHSFloat32AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingSubRHSFloat64AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Int32, dtypes.Uint32:
				binaryTrailingSubRHSInt32AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Int64, dtypes.Uint64:
				binaryTrailingSubRHSInt64AVX2(vec, scalar, out, A, B)
				return true
			}
		} else {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingSubLHSFloat32AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingSubLHSFloat64AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Int32, dtypes.Uint32:
				binaryTrailingSubLHSInt32AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Int64, dtypes.Uint64:
				binaryTrailingSubLHSInt64AVX2(vec, scalar, out, A, B)
				return true
			}
		}
	case compute.OpTypeMul:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMulFloat32AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMulFloat64AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Int32, dtypes.Uint32:
			binaryTrailingMulInt32AVX2(vec, scalar, out, A, B)
			return true
		}
	case compute.OpTypeDiv:
		if !isLHS {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingDivRHSFloat32AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingDivRHSFloat64AVX2(vec, scalar, out, A, B)
				return true
			}
		} else {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingDivLHSFloat32AVX2(vec, scalar, out, A, B)
				return true
			case dtypes.Float64:
				binaryTrailingDivLHSFloat64AVX2(vec, scalar, out, A, B)
				return true
			}
		}
	case compute.OpTypeMax:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMaxFloat32AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMaxFloat64AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Int32:
			binaryTrailingMaxInt32AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Uint32:
			binaryTrailingMaxUint32AVX2(vec, scalar, out, A, B)
			return true
		}
	case compute.OpTypeMin:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMinFloat32AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMinFloat64AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Int32:
			binaryTrailingMinInt32AVX2(vec, scalar, out, A, B)
			return true
		case dtypes.Uint32:
			binaryTrailingMinUint32AVX2(vec, scalar, out, A, B)
			return true
		}
	}
	return false
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// DispatchBinaryTrailingNEON dispatches binary trailing broadcast operations to ARM64 NEON kernels.
// Returns true if handled, false if unsupported.
func DispatchBinaryTrailingNEON(op compute.OpType, isLHS bool, lhs, rhs, out unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	if A <= 0 || B <= 0 {
		return true
	}
	vecPtr := lhs
	scalarPtr := rhs
	if isLHS {
		vecPtr = rhs
		scalarPtr = lhs
	}

	switch op {
	case compute.OpTypeAdd:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingAddFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
			return true
		case dtypes.Float64:
			binaryTrailingAddFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
			return true
		case dtypes.Int32, dtypes.Uint32:
			binaryTrailingAddInt32NEON(unsafe.Slice((*int32)(vecPtr), A*B), unsafe.Slice((*int32)(scalarPtr), A), unsafe.Slice((*int32)(out), A*B), A, B)
			return true
		case dtypes.Int64, dtypes.Uint64:
			binaryTrailingAddInt64NEON(unsafe.Slice((*int64)(vecPtr), A*B), unsafe.Slice((*int64)(scalarPtr), A), unsafe.Slice((*int64)(out), A*B), A, B)
			return true
		}
	case compute.OpTypeSub:
		if !isLHS {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingSubRHSFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
				return true
			case dtypes.Float64:
				binaryTrailingSubRHSFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
				return true
			case dtypes.Int32, dtypes.Uint32:
				binaryTrailingSubRHSInt32NEON(unsafe.Slice((*int32)(vecPtr), A*B), unsafe.Slice((*int32)(scalarPtr), A), unsafe.Slice((*int32)(out), A*B), A, B)
				return true
			case dtypes.Int64, dtypes.Uint64:
				binaryTrailingSubRHSInt64NEON(unsafe.Slice((*int64)(vecPtr), A*B), unsafe.Slice((*int64)(scalarPtr), A), unsafe.Slice((*int64)(out), A*B), A, B)
				return true
			}
		} else {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingSubLHSFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
				return true
			case dtypes.Float64:
				binaryTrailingSubLHSFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
				return true
			case dtypes.Int32, dtypes.Uint32:
				binaryTrailingSubLHSInt32NEON(unsafe.Slice((*int32)(vecPtr), A*B), unsafe.Slice((*int32)(scalarPtr), A), unsafe.Slice((*int32)(out), A*B), A, B)
				return true
			case dtypes.Int64, dtypes.Uint64:
				binaryTrailingSubLHSInt64NEON(unsafe.Slice((*int64)(vecPtr), A*B), unsafe.Slice((*int64)(scalarPtr), A), unsafe.Slice((*int64)(out), A*B), A, B)
				return true
			}
		}
	case compute.OpTypeMul:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMulFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMulFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
			return true
		case dtypes.Int32, dtypes.Uint32:
			binaryTrailingMulInt32NEON(unsafe.Slice((*int32)(vecPtr), A*B), unsafe.Slice((*int32)(scalarPtr), A), unsafe.Slice((*int32)(out), A*B), A, B)
			return true
		}
	case compute.OpTypeDiv:
		if !isLHS {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingDivRHSFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
				return true
			case dtypes.Float64:
				binaryTrailingDivRHSFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
				return true
			}
		} else {
			switch dtype {
			case dtypes.Float32:
				binaryTrailingDivLHSFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
				return true
			case dtypes.Float64:
				binaryTrailingDivLHSFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
				return true
			}
		}
	case compute.OpTypeMax:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMaxFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMaxFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
			return true
		case dtypes.Int32:
			binaryTrailingMaxInt32NEON(unsafe.Slice((*int32)(vecPtr), A*B), unsafe.Slice((*int32)(scalarPtr), A), unsafe.Slice((*int32)(out), A*B), A, B)
			return true
		case dtypes.Uint32:
			binaryTrailingMaxUint32NEON(unsafe.Slice((*uint32)(vecPtr), A*B), unsafe.Slice((*uint32)(scalarPtr), A), unsafe.Slice((*uint32)(out), A*B), A, B)
			return true
		}
	case compute.OpTypeMin:
		switch dtype {
		case dtypes.Float32:
			binaryTrailingMinFloat32NEON(unsafe.Slice((*float32)(vecPtr), A*B), unsafe.Slice((*float32)(scalarPtr), A), unsafe.Slice((*float32)(out), A*B), A, B)
			return true
		case dtypes.Float64:
			binaryTrailingMinFloat64NEON(unsafe.Slice((*float64)(vecPtr), A*B), unsafe.Slice((*float64)(scalarPtr), A), unsafe.Slice((*float64)(out), A*B), A, B)
			return true
		case dtypes.Int32:
			binaryTrailingMinInt32NEON(unsafe.Slice((*int32)(vecPtr), A*B), unsafe.Slice((*int32)(scalarPtr), A), unsafe.Slice((*int32)(out), A*B), A, B)
			return true
		case dtypes.Uint32:
			binaryTrailingMinUint32NEON(unsafe.Slice((*uint32)(vecPtr), A*B), unsafe.Slice((*uint32)(scalarPtr), A), unsafe.Slice((*uint32)(out), A*B), A, B)
			return true
		}
	}
	return false
}

// ----------------------------------------------------------------------------
// Float32 Binary Trailing Kernels
// ----------------------------------------------------------------------------

func binaryTrailingAddFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))).Add(sv)
			v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))).Add(sv)
			v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))).Add(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] + s
		}
	}
}

func binaryTrailingSubRHSFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))).Sub(sv)
			v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))).Sub(sv)
			v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))).Sub(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] - s
		}
	}
}

func binaryTrailingSubLHSFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := sv.Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))))
			v1 := sv.Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))))
			v2 := sv.Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))))
			v3 := sv.Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))))
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := sv.Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))))
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = s - vec[off+b]
		}
	}
}

func binaryTrailingMulFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Mul(sv)
			v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))).Mul(sv)
			v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))).Mul(sv)
			v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))).Mul(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Mul(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] * s
		}
	}
}

func binaryTrailingDivRHSFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Div(sv)
			v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))).Div(sv)
			v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))).Div(sv)
			v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))).Div(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Div(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] / s
		}
	}
}

func binaryTrailingDivLHSFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := sv.Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))))
			v1 := sv.Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))))
			v2 := sv.Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))))
			v3 := sv.Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))))
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := sv.Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))))
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = s / vec[off+b]
		}
	}
}

func binaryTrailingMaxFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))).Max(sv)
			v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))).Max(sv)
			v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))).Max(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = max(vec[off+b], s)
		}
	}
}

func binaryTrailingMinFloat32NEON(vec, scalar, out []float32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+4]))).Min(sv)
			v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+8]))).Min(sv)
			v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b+12]))).Min(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = min(vec[off+b], s)
		}
	}
}

// ----------------------------------------------------------------------------
// Float64 Binary Trailing Kernels
// ----------------------------------------------------------------------------

func binaryTrailingAddFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))).Add(sv)
			v2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))).Add(sv)
			v3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))).Add(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] + s
		}
	}
}

func binaryTrailingSubRHSFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))).Sub(sv)
			v2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))).Sub(sv)
			v3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))).Sub(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] - s
		}
	}
}

func binaryTrailingSubLHSFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := sv.Sub(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))))
			v1 := sv.Sub(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))))
			v2 := sv.Sub(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))))
			v3 := sv.Sub(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))))
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := sv.Sub(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))))
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = s - vec[off+b]
		}
	}
}

func binaryTrailingMulFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Mul(sv)
			v1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))).Mul(sv)
			v2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))).Mul(sv)
			v3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))).Mul(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Mul(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] * s
		}
	}
}

func binaryTrailingDivRHSFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Div(sv)
			v1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))).Div(sv)
			v2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))).Div(sv)
			v3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))).Div(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Div(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] / s
		}
	}
}

func binaryTrailingDivLHSFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := sv.Div(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))))
			v1 := sv.Div(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))))
			v2 := sv.Div(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))))
			v3 := sv.Div(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))))
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := sv.Div(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))))
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = s / vec[off+b]
		}
	}
}

func binaryTrailingMaxFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))).Max(sv)
			v2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))).Max(sv)
			v3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))).Max(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = max(vec[off+b], s)
		}
	}
}

func binaryTrailingMinFloat64NEON(vec, scalar, out []float64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastFloat64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+2]))).Min(sv)
			v2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+4]))).Min(sv)
			v3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b+6]))).Min(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v0.StoreArray((*[2]float64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = min(vec[off+b], s)
		}
	}
}

// ----------------------------------------------------------------------------
// Int32 / Uint32 / Int64 Binary Trailing Kernels
// ----------------------------------------------------------------------------

func binaryTrailingAddInt32NEON(vec, scalar, out []int32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+4]))).Add(sv)
			v2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+8]))).Add(sv)
			v3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+12]))).Add(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] + s
		}
	}
}

func binaryTrailingSubRHSInt32NEON(vec, scalar, out []int32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+4]))).Sub(sv)
			v2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+8]))).Sub(sv)
			v3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+12]))).Sub(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] - s
		}
	}
}

func binaryTrailingSubLHSInt32NEON(vec, scalar, out []int32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := sv.Sub(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))))
			v1 := sv.Sub(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+4]))))
			v2 := sv.Sub(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+8]))))
			v3 := sv.Sub(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+12]))))
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := sv.Sub(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))))
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = s - vec[off+b]
		}
	}
}

func binaryTrailingMulInt32NEON(vec, scalar, out []int32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Mul(sv)
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+4]))).Mul(sv)
			v2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+8]))).Mul(sv)
			v3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+12]))).Mul(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Mul(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] * s
		}
	}
}

func binaryTrailingMaxInt32NEON(vec, scalar, out []int32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+4]))).Max(sv)
			v2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+8]))).Max(sv)
			v3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+12]))).Max(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = max(vec[off+b], s)
		}
	}
}

func binaryTrailingMinInt32NEON(vec, scalar, out []int32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+4]))).Min(sv)
			v2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+8]))).Min(sv)
			v3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b+12]))).Min(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v0.StoreArray((*[4]int32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = min(vec[off+b], s)
		}
	}
}

func binaryTrailingMaxUint32NEON(vec, scalar, out []uint32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastUint32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v1 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b+4]))).Max(sv)
			v2 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b+8]))).Max(sv)
			v3 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b+12]))).Max(sv)
			v0.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b]))).Max(sv)
			v0.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = max(vec[off+b], s)
		}
	}
}

func binaryTrailingMinUint32NEON(vec, scalar, out []uint32, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastUint32x4(s)
		off := a * B
		b := 0
		for ; b+16 <= B; b += 16 {
			v0 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v1 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b+4]))).Min(sv)
			v2 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b+8]))).Min(sv)
			v3 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b+12]))).Min(sv)
			v0.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b+4])))
			v2.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b+8])))
			v3.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b+12])))
		}
		for ; b+4 <= B; b += 4 {
			v0 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&vec[off+b]))).Min(sv)
			v0.StoreArray((*[4]uint32)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = min(vec[off+b], s)
		}
	}
}

func binaryTrailingAddInt64NEON(vec, scalar, out []int64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v1 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+2]))).Add(sv)
			v2 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+4]))).Add(sv)
			v3 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+6]))).Add(sv)
			v0.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b]))).Add(sv)
			v0.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] + s
		}
	}
}

func binaryTrailingSubRHSInt64NEON(vec, scalar, out []int64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v1 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+2]))).Sub(sv)
			v2 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+4]))).Sub(sv)
			v3 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+6]))).Sub(sv)
			v0.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b]))).Sub(sv)
			v0.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = vec[off+b] - s
		}
	}
}

func binaryTrailingSubLHSInt64NEON(vec, scalar, out []int64, A, B int) {
	for a := range A {
		s := scalar[a]
		sv := archsimd.BroadcastInt64x2(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			v0 := sv.Sub(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b]))))
			v1 := sv.Sub(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+2]))))
			v2 := sv.Sub(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+4]))))
			v3 := sv.Sub(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b+6]))))
			v0.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b])))
			v1.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+2])))
			v2.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+4])))
			v3.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b+6])))
		}
		for ; b+2 <= B; b += 2 {
			v0 := sv.Sub(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&vec[off+b]))))
			v0.StoreArray((*[2]int64)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = s - vec[off+b]
		}
	}
}

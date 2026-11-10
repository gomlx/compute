// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/internal/gobackend/simdarch"
)

func init() {
	if simdarch.IsNEONAllowed {
		registerNEON()
	}
	if simdarch.IsSMEAllowed {
		gobackend.SetNodeExecutor(compute.OpTypeConvGeneral, gobackend.PrioritySME, execConvGeneralSME)
	}
}

func registerNEON() {
	gobackend.SetReduceTrailingSumArchDispatcher(gobackend.PriorityNEON, func(operand, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		inPtr := operand.UnsafePointer()
		outPtr := output.UnsafePointer()
		if inPtr == nil || outPtr == nil {
			return false
		}
		return DispatchTrailingSumNEON(inPtr, outPtr, A, B, dtype)
	})
	gobackend.SetReduceLeadingSumArchDispatcher(gobackend.PriorityNEON, func(operand, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		inPtr := operand.UnsafePointer()
		outPtr := output.UnsafePointer()
		if inPtr == nil || outPtr == nil {
			return false
		}
		return DispatchLeadingSumNEON(inPtr, outPtr, A, B, dtype)
	})
	gobackend.SetBinaryTrailingArchDispatcher(gobackend.PriorityNEON, func(op compute.OpType, isLHS bool, lhs, rhs, output *gobackend.Buffer, A, B int, dtype dtypes.DType) bool {
		lhsPtr := lhs.UnsafePointer()
		rhsPtr := rhs.UnsafePointer()
		outPtr := output.UnsafePointer()
		if lhsPtr == nil || rhsPtr == nil || outPtr == nil {
			return false
		}
		return DispatchBinaryTrailingNEON(op, isLHS, lhsPtr, rhsPtr, outPtr, A, B, dtype)
	})
	gobackend.SetStridedCopyArchDispatcher(
		gobackend.PriorityNEON,
		DispatchCopyStridedBytesNEON,
		DispatchDupStridedBytesNEON,
		DispatchConcat2StridedBytesNEON,
	)
	ops.SetWhereFloat32ArchDispatcher(gobackend.PriorityNEON, DispatchWhereFloat32NEON)
	gobackend.SetNodeExecutor(compute.OpTypeConvGeneral, gobackend.PriorityNEON, execConvGeneralNEON)
}

//go:noescape
func reduceTrailingSumFloat32NEONAsm(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumFloat64NEONAsm(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumFloat16NEONAsm(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceTrailingSumBFloat16NEONAsm(in, out unsafe.Pointer, A, B int)

//go:noescape
func reduceLeadingSumFloat16NEONAsm(in, out unsafe.Pointer, A, B int)

// DispatchTrailingSumNEON dispatches trailing sum reduction to ARM64 NEON kernels.
// Returns true if handled, false if dtype is unsupported.
func DispatchTrailingSumNEON(inPtr, outPtr unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	if A <= 0 || B <= 0 {
		return true
	}
	switch dtype {
	case dtypes.Float32:
		reduceTrailingSumFloat32NEONAsm(inPtr, outPtr, A, B)
	case dtypes.Float64:
		reduceTrailingSumFloat64NEONAsm(inPtr, outPtr, A, B)
	case dtypes.Float16:
		reduceTrailingSumFloat16NEONAsm(inPtr, outPtr, A, B)
	case dtypes.BFloat16:
		reduceTrailingSumBFloat16NEONAsm(inPtr, outPtr, A, B)
	case dtypes.Int32:
		reduceTrailingSumInt32NEON(unsafe.Slice((*int32)(inPtr), A*B), unsafe.Slice((*int32)(outPtr), A), A, B)
	case dtypes.Uint32:
		reduceTrailingSumUint32NEON(unsafe.Slice((*uint32)(inPtr), A*B), unsafe.Slice((*uint32)(outPtr), A), A, B)
	case dtypes.Int64:
		reduceTrailingSumInt64NEON(unsafe.Slice((*int64)(inPtr), A*B), unsafe.Slice((*int64)(outPtr), A), A, B)
	case dtypes.Uint64:
		reduceTrailingSumUint64NEON(unsafe.Slice((*uint64)(inPtr), A*B), unsafe.Slice((*uint64)(outPtr), A), A, B)
	case dtypes.Int16:
		reduceTrailingSumInt16NEON(unsafe.Slice((*int16)(inPtr), A*B), unsafe.Slice((*int16)(outPtr), A), A, B)
	case dtypes.Uint16:
		reduceTrailingSumUint16NEON(unsafe.Slice((*uint16)(inPtr), A*B), unsafe.Slice((*uint16)(outPtr), A), A, B)
	case dtypes.Int8:
		reduceTrailingSumInt8NEON(unsafe.Slice((*int8)(inPtr), A*B), unsafe.Slice((*int8)(outPtr), A), A, B)
	case dtypes.Uint8:
		reduceTrailingSumUint8NEON(unsafe.Slice((*uint8)(inPtr), A*B), unsafe.Slice((*uint8)(outPtr), A), A, B)
	default:
		return false
	}
	return true
}

func reduceTrailingSumInt32NEON(in, out []int32, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Int32x4
		b := 0
		for ; b+16 <= B; b += 16 {
			acc0 = acc0.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&row[b+4]))))
			acc2 = acc2.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&row[b+8]))))
			acc3 = acc3.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&row[b+12]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+4 <= B; b += 4 {
			acc0 = acc0.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ReduceSum()
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

func reduceTrailingSumUint32NEON(in, out []uint32, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Uint32x4
		b := 0
		for ; b+16 <= B; b += 16 {
			acc0 = acc0.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&row[b+4]))))
			acc2 = acc2.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&row[b+8]))))
			acc3 = acc3.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&row[b+12]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+4 <= B; b += 4 {
			acc0 = acc0.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ReduceSum()
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

func reduceTrailingSumInt64NEON(in, out []int64, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Int64x2
		b := 0
		for ; b+8 <= B; b += 8 {
			acc0 = acc0.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&row[b+2]))))
			acc2 = acc2.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&row[b+4]))))
			acc3 = acc3.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&row[b+6]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+2 <= B; b += 2 {
			acc0 = acc0.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ConcatAddPairs(acc0).GetElem(0)
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

func reduceTrailingSumUint64NEON(in, out []uint64, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Uint64x2
		b := 0
		for ; b+8 <= B; b += 8 {
			acc0 = acc0.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&row[b+2]))))
			acc2 = acc2.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&row[b+4]))))
			acc3 = acc3.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&row[b+6]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+2 <= B; b += 2 {
			acc0 = acc0.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ConcatAddPairs(acc0).GetElem(0)
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

func reduceTrailingSumInt16NEON(in, out []int16, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Int16x8
		b := 0
		for ; b+32 <= B; b += 32 {
			acc0 = acc0.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&row[b+8]))))
			acc2 = acc2.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&row[b+16]))))
			acc3 = acc3.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&row[b+24]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+8 <= B; b += 8 {
			acc0 = acc0.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ReduceSum()
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

func reduceTrailingSumUint16NEON(in, out []uint16, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Uint16x8
		b := 0
		for ; b+32 <= B; b += 32 {
			acc0 = acc0.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&row[b+8]))))
			acc2 = acc2.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&row[b+16]))))
			acc3 = acc3.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&row[b+24]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+8 <= B; b += 8 {
			acc0 = acc0.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ReduceSum()
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

func reduceTrailingSumInt8NEON(in, out []int8, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Int8x16
		b := 0
		for ; b+64 <= B; b += 64 {
			acc0 = acc0.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&row[b+16]))))
			acc2 = acc2.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&row[b+32]))))
			acc3 = acc3.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&row[b+48]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+16 <= B; b += 16 {
			acc0 = acc0.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ReduceSum()
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

func reduceTrailingSumUint8NEON(in, out []uint8, A, B int) {
	for a := range A {
		row := in[a*B : (a+1)*B]
		var acc0, acc1, acc2, acc3 archsimd.Uint8x16
		b := 0
		for ; b+64 <= B; b += 64 {
			acc0 = acc0.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&row[b]))))
			acc1 = acc1.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&row[b+16]))))
			acc2 = acc2.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&row[b+32]))))
			acc3 = acc3.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&row[b+48]))))
		}
		acc0 = acc0.Add(acc1).Add(acc2.Add(acc3))
		for ; b+16 <= B; b += 16 {
			acc0 = acc0.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&row[b]))))
		}
		sum := acc0.ReduceSum()
		for ; b < B; b++ {
			sum += row[b]
		}
		out[a] = sum
	}
}

// DispatchLeadingSumNEON dispatches leading sum reduction ([A, B] -> [B]) to ARM64 NEON kernels.
// Returns true if handled, false if dtype is unsupported.
func DispatchLeadingSumNEON(inPtr, outPtr unsafe.Pointer, A, B int, dtype dtypes.DType) bool {
	if A <= 0 || B <= 0 {
		return true
	}
	switch dtype {
	case dtypes.Float32:
		reduceLeadingSumFloat32NEON(unsafe.Slice((*float32)(inPtr), A*B), unsafe.Slice((*float32)(outPtr), B), A, B)
	case dtypes.Float64:
		reduceLeadingSumFloat64NEON(unsafe.Slice((*float64)(inPtr), A*B), unsafe.Slice((*float64)(outPtr), B), A, B)
	case dtypes.Float16:
		reduceLeadingSumFloat16NEONAsm(inPtr, outPtr, A, B)
	case dtypes.BFloat16:
		reduceLeadingSumBFloat16NEON(unsafe.Slice((*bfloat16.BFloat16)(inPtr), A*B), unsafe.Slice((*bfloat16.BFloat16)(outPtr), B), A, B)
	case dtypes.Int32:
		reduceLeadingSumInt32NEON(unsafe.Slice((*int32)(inPtr), A*B), unsafe.Slice((*int32)(outPtr), B), A, B)
	case dtypes.Uint32:
		reduceLeadingSumUint32NEON(unsafe.Slice((*uint32)(inPtr), A*B), unsafe.Slice((*uint32)(outPtr), B), A, B)
	case dtypes.Int64:
		reduceLeadingSumInt64NEON(unsafe.Slice((*int64)(inPtr), A*B), unsafe.Slice((*int64)(outPtr), B), A, B)
	case dtypes.Uint64:
		reduceLeadingSumUint64NEON(unsafe.Slice((*uint64)(inPtr), A*B), unsafe.Slice((*uint64)(outPtr), B), A, B)
	case dtypes.Int16:
		reduceLeadingSumInt16NEON(unsafe.Slice((*int16)(inPtr), A*B), unsafe.Slice((*int16)(outPtr), B), A, B)
	case dtypes.Uint16:
		reduceLeadingSumUint16NEON(unsafe.Slice((*uint16)(inPtr), A*B), unsafe.Slice((*uint16)(outPtr), B), A, B)
	case dtypes.Int8:
		reduceLeadingSumInt8NEON(unsafe.Slice((*int8)(inPtr), A*B), unsafe.Slice((*int8)(outPtr), B), A, B)
	case dtypes.Uint8:
		reduceLeadingSumUint8NEON(unsafe.Slice((*uint8)(inPtr), A*B), unsafe.Slice((*uint8)(outPtr), B), A, B)
	default:
		return false
	}
	return true
}

func reduceLeadingSumFloat32NEON(in, out []float32, A, B int) {
	b := 0
	for ; b+32 <= B; b += 32 {
		var acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 archsimd.Float32x4
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+4]))))
			acc2 = acc2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+8]))))
			acc3 = acc3.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+12]))))
			acc4 = acc4.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+16]))))
			acc5 = acc5.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+20]))))
			acc6 = acc6.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+24]))))
			acc7 = acc7.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+28]))))
		}
		acc0.StoreArray((*[4]float32)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+4])))
		acc2.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+8])))
		acc3.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+12])))
		acc4.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+16])))
		acc5.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+20])))
		acc6.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+24])))
		acc7.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+28])))
	}
	for ; b+16 <= B; b += 16 {
		var acc0, acc1, acc2, acc3 archsimd.Float32x4
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+4]))))
			acc2 = acc2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+8]))))
			acc3 = acc3.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[off+12]))))
		}
		acc0.StoreArray((*[4]float32)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+4])))
		acc2.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+8])))
		acc3.StoreArray((*[4]float32)(unsafe.Pointer(&out[b+12])))
	}
	for ; b+4 <= B; b += 4 {
		var acc archsimd.Float32x4
		for a := range A {
			acc = acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[4]float32)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum float32
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumFloat64NEON(in, out []float64, A, B int) {
	b := 0
	for ; b+8 <= B; b += 8 {
		var acc0, acc1, acc2, acc3 archsimd.Float64x2
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&in[off+2]))))
			acc2 = acc2.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&in[off+4]))))
			acc3 = acc3.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&in[off+6]))))
		}
		acc0.StoreArray((*[2]float64)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[2]float64)(unsafe.Pointer(&out[b+2])))
		acc2.StoreArray((*[2]float64)(unsafe.Pointer(&out[b+4])))
		acc3.StoreArray((*[2]float64)(unsafe.Pointer(&out[b+6])))
	}
	for ; b+2 <= B; b += 2 {
		var acc archsimd.Float64x2
		for a := range A {
			acc = acc.Add(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[2]float64)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum float64
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumBFloat16NEON(in, out []bfloat16.BFloat16, A, B int) {
	b := 0
	for ; b+8 <= B; b += 8 {
		var accLo, accHi archsimd.Float32x4
		for a := range A {
			v := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&in[a*B+b])))
			lo, hi := v.ToFloat32()
			accLo = accLo.Add(lo)
			accHi = accHi.Add(hi)
		}
		bfloat16.FromFloat32x4(accLo, accHi).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum float32
		for a := range A {
			sum += in[a*B+b].Float32()
		}
		out[b] = bfloat16.FromFloat32(sum)
	}
}

func reduceLeadingSumInt32NEON(in, out []int32, A, B int) {
	b := 0
	for ; b+16 <= B; b += 16 {
		var acc0, acc1, acc2, acc3 archsimd.Int32x4
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&in[off+4]))))
			acc2 = acc2.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&in[off+8]))))
			acc3 = acc3.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&in[off+12]))))
		}
		acc0.StoreArray((*[4]int32)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[4]int32)(unsafe.Pointer(&out[b+4])))
		acc2.StoreArray((*[4]int32)(unsafe.Pointer(&out[b+8])))
		acc3.StoreArray((*[4]int32)(unsafe.Pointer(&out[b+12])))
	}
	for ; b+4 <= B; b += 4 {
		var acc archsimd.Int32x4
		for a := range A {
			acc = acc.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[4]int32)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum int32
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumUint32NEON(in, out []uint32, A, B int) {
	b := 0
	for ; b+16 <= B; b += 16 {
		var acc0, acc1, acc2, acc3 archsimd.Uint32x4
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&in[off+4]))))
			acc2 = acc2.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&in[off+8]))))
			acc3 = acc3.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&in[off+12]))))
		}
		acc0.StoreArray((*[4]uint32)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[4]uint32)(unsafe.Pointer(&out[b+4])))
		acc2.StoreArray((*[4]uint32)(unsafe.Pointer(&out[b+8])))
		acc3.StoreArray((*[4]uint32)(unsafe.Pointer(&out[b+12])))
	}
	for ; b+4 <= B; b += 4 {
		var acc archsimd.Uint32x4
		for a := range A {
			acc = acc.Add(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[4]uint32)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum uint32
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumInt64NEON(in, out []int64, A, B int) {
	b := 0
	for ; b+8 <= B; b += 8 {
		var acc0, acc1, acc2, acc3 archsimd.Int64x2
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&in[off+2]))))
			acc2 = acc2.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&in[off+4]))))
			acc3 = acc3.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&in[off+6]))))
		}
		acc0.StoreArray((*[2]int64)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[2]int64)(unsafe.Pointer(&out[b+2])))
		acc2.StoreArray((*[2]int64)(unsafe.Pointer(&out[b+4])))
		acc3.StoreArray((*[2]int64)(unsafe.Pointer(&out[b+6])))
	}
	for ; b+2 <= B; b += 2 {
		var acc archsimd.Int64x2
		for a := range A {
			acc = acc.Add(archsimd.LoadInt64x2Array((*[2]int64)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[2]int64)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum int64
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumUint64NEON(in, out []uint64, A, B int) {
	b := 0
	for ; b+8 <= B; b += 8 {
		var acc0, acc1, acc2, acc3 archsimd.Uint64x2
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&in[off+2]))))
			acc2 = acc2.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&in[off+4]))))
			acc3 = acc3.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&in[off+6]))))
		}
		acc0.StoreArray((*[2]uint64)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[2]uint64)(unsafe.Pointer(&out[b+2])))
		acc2.StoreArray((*[2]uint64)(unsafe.Pointer(&out[b+4])))
		acc3.StoreArray((*[2]uint64)(unsafe.Pointer(&out[b+6])))
	}
	for ; b+2 <= B; b += 2 {
		var acc archsimd.Uint64x2
		for a := range A {
			acc = acc.Add(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[2]uint64)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum uint64
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumInt16NEON(in, out []int16, A, B int) {
	b := 0
	for ; b+32 <= B; b += 32 {
		var acc0, acc1, acc2, acc3 archsimd.Int16x8
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&in[off+8]))))
			acc2 = acc2.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&in[off+16]))))
			acc3 = acc3.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&in[off+24]))))
		}
		acc0.StoreArray((*[8]int16)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[8]int16)(unsafe.Pointer(&out[b+8])))
		acc2.StoreArray((*[8]int16)(unsafe.Pointer(&out[b+16])))
		acc3.StoreArray((*[8]int16)(unsafe.Pointer(&out[b+24])))
	}
	for ; b+8 <= B; b += 8 {
		var acc archsimd.Int16x8
		for a := range A {
			acc = acc.Add(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[8]int16)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum int16
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumUint16NEON(in, out []uint16, A, B int) {
	b := 0
	for ; b+32 <= B; b += 32 {
		var acc0, acc1, acc2, acc3 archsimd.Uint16x8
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&in[off+8]))))
			acc2 = acc2.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&in[off+16]))))
			acc3 = acc3.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&in[off+24]))))
		}
		acc0.StoreArray((*[8]uint16)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[8]uint16)(unsafe.Pointer(&out[b+8])))
		acc2.StoreArray((*[8]uint16)(unsafe.Pointer(&out[b+16])))
		acc3.StoreArray((*[8]uint16)(unsafe.Pointer(&out[b+24])))
	}
	for ; b+8 <= B; b += 8 {
		var acc archsimd.Uint16x8
		for a := range A {
			acc = acc.Add(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[8]uint16)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum uint16
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumInt8NEON(in, out []int8, A, B int) {
	b := 0
	for ; b+64 <= B; b += 64 {
		var acc0, acc1, acc2, acc3 archsimd.Int8x16
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&in[off+16]))))
			acc2 = acc2.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&in[off+32]))))
			acc3 = acc3.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&in[off+48]))))
		}
		acc0.StoreArray((*[16]int8)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[16]int8)(unsafe.Pointer(&out[b+16])))
		acc2.StoreArray((*[16]int8)(unsafe.Pointer(&out[b+32])))
		acc3.StoreArray((*[16]int8)(unsafe.Pointer(&out[b+48])))
	}
	for ; b+16 <= B; b += 16 {
		var acc archsimd.Int8x16
		for a := range A {
			acc = acc.Add(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[16]int8)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum int8
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

func reduceLeadingSumUint8NEON(in, out []uint8, A, B int) {
	b := 0
	for ; b+64 <= B; b += 64 {
		var acc0, acc1, acc2, acc3 archsimd.Uint8x16
		for a := range A {
			off := a*B + b
			acc0 = acc0.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&in[off]))))
			acc1 = acc1.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&in[off+16]))))
			acc2 = acc2.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&in[off+32]))))
			acc3 = acc3.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&in[off+48]))))
		}
		acc0.StoreArray((*[16]uint8)(unsafe.Pointer(&out[b])))
		acc1.StoreArray((*[16]uint8)(unsafe.Pointer(&out[b+16])))
		acc2.StoreArray((*[16]uint8)(unsafe.Pointer(&out[b+32])))
		acc3.StoreArray((*[16]uint8)(unsafe.Pointer(&out[b+48])))
	}
	for ; b+16 <= B; b += 16 {
		var acc archsimd.Uint8x16
		for a := range A {
			acc = acc.Add(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&in[a*B+b]))))
		}
		acc.StoreArray((*[16]uint8)(unsafe.Pointer(&out[b])))
	}
	for ; b < B; b++ {
		var sum uint8
		for a := range A {
			sum += in[a*B+b]
		}
		out[b] = sum
	}
}

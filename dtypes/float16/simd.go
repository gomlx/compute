// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package float16

import (
	"simd"
	"unsafe"
)

// LoadFloat16s loads a vector of Float16 values into a simd.Uint16s vector.
// Slice must have at least the vector length.
func LoadFloat16s(s []Float16) simd.Uint16s {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	return simd.LoadUint16s(u16Slice)
}

// LoadFloat16sPart loads a partial slice of Float16 values into a simd.Uint16s vector,
// returning the vector and the count of loaded elements.
func LoadFloat16sPart(s []Float16) (simd.Uint16s, int) {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	return simd.LoadUint16sPart(u16Slice)
}

// StoreFloat16s stores a simd.Uint16s vector into a Float16 slice.
func StoreFloat16s(v simd.Uint16s, s []Float16) {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	v.Store(u16Slice)
}

// StoreFloat16sPart stores a partial simd.Uint16s vector into a Float16 slice and returns
// the number of elements stored.
func StoreFloat16sPart(v simd.Uint16s, s []Float16) int {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	return v.StorePart(u16Slice)
}

func selectBits(mask simd.Mask32s, trueBits, falseBits simd.Uint32s) simd.Uint32s {
	m := mask.ToInt32s().ToBits()
	return (trueBits.And(m)).Or(falseBits.AndNot(m))
}

func selectFloat32s(mask simd.Mask32s, trueVal, falseVal simd.Float32s) simd.Float32s {
	bits := selectBits(mask, trueVal.ToBits(), falseVal.ToBits())
	return bits.BitsToFloat32()
}

func f16BitsToFloat32s(u simd.Uint32s) simd.Float32s {
	sign := u.And(simd.BroadcastUint32s(0x8000)).ShiftAllLeft(16)
	exp := u.And(simd.BroadcastUint32s(0x7C00)).ShiftAllRight(10)
	mantissa := u.And(simd.BroadcastUint32s(0x03FF))

	normExp := exp.Add(simd.BroadcastUint32s(112)).ShiftAllLeft(23)
	normMant := mantissa.ShiftAllLeft(13)
	normRes := sign.Or(normExp).Or(normMant).BitsToFloat32()

	isZero := exp.Equal(simd.BroadcastUint32s(0)).And(mantissa.Equal(simd.BroadcastUint32s(0)))
	isDenorm := exp.Equal(simd.BroadcastUint32s(0)).And(mantissa.Greater(simd.BroadcastUint32s(0)))
	isInfNaN := exp.Equal(simd.BroadcastUint32s(0x1F))

	zeroRes := sign.BitsToFloat32()
	infNaNRes := sign.Or(simd.BroadcastUint32s(0x7F800000)).Or(normMant).BitsToFloat32()

	// Hardware FPU subtraction aligns denormal mantissa: (magic + mantissa) - magic
	magicBits := simd.BroadcastUint32s(113 << 23).Or(normMant)
	magicF32 := magicBits.BitsToFloat32()
	biasF32 := simd.BroadcastUint32s(113 << 23).BitsToFloat32()
	denormF32 := magicF32.Sub(biasF32)
	isNeg := sign.Equal(simd.BroadcastUint32s(0x80000000))
	denormRes := selectFloat32s(isNeg, denormF32.Neg(), denormF32)

	res := normRes
	res = selectFloat32s(isZero, zeroRes, res)
	res = selectFloat32s(isInfNaN, infNaNRes, res)
	res = selectFloat32s(isDenorm, denormRes, res)
	return res
}

func float32sToF16Bits(f simd.Float32s) simd.Uint32s {
	u := f.ToBits()
	sign := u.ShiftAllRight(16).And(simd.BroadcastUint32s(0x8000))
	uAbs := u.And(simd.BroadcastUint32s(0x7FFFFFFF))

	isOverflow := uAbs.GreaterEqual(simd.BroadcastUint32s(0x477F8000))
	isNan := uAbs.Greater(simd.BroadcastUint32s(0x7F800000))
	nanRes := sign.Or(simd.BroadcastUint32s(0x7C00)).Or(uAbs.ShiftAllRight(13).And(simd.BroadcastUint32s(0x03FF))).Or(simd.BroadcastUint32s(1))
	infRes := sign.Or(simd.BroadcastUint32s(0x7C00))
	overRes := selectBits(isNan, nanRes, infRes)

	isUnder := uAbs.Less(simd.BroadcastUint32s(0x38800000))
	isZero := uAbs.Less(simd.BroadcastUint32s(0x33000000))

	fAbs := uAbs.BitsToFloat32()
	magicF32 := simd.BroadcastUint32s(113 << 23).BitsToFloat32()
	fDenorm := fAbs.Add(magicF32)
	denormMant := fDenorm.ToBits().Sub(simd.BroadcastUint32s(113 << 23)).And(simd.BroadcastUint32s(0x03FF))
	denormRes := selectBits(isZero, sign, sign.Or(denormMant))

	bias := simd.BroadcastUint32s(0x0FFF).Add(uAbs.ShiftAllRight(13).And(simd.BroadcastUint32s(1)))
	rounded := uAbs.Add(bias)
	normExpAdjusted := rounded.Sub(simd.BroadcastUint32s(112 << 23))
	isNormOverflow := normExpAdjusted.GreaterEqual(simd.BroadcastUint32s(0x1F << 23))
	normRes := selectBits(isNormOverflow, infRes, sign.Or(normExpAdjusted.ShiftAllRight(13)))

	res := normRes
	res = selectBits(isOverflow, overRes, res)
	res = selectBits(isUnder, denormRes, res)
	return res
}

// ToFloat32SIMD converts a simd.Uint16s vector representing Float16 values into two
// simd.Float32s vectors (even-indexed elements and odd-indexed elements).
func ToFloat32SIMD(v simd.Uint16s) (even, odd simd.Float32s) {
	u32 := v.ReshapeToUint32s()
	evenU32 := u32.And(simd.BroadcastUint32s(0xFFFF))
	oddU32 := u32.ShiftAllRight(16)
	return f16BitsToFloat32s(evenU32), f16BitsToFloat32s(oddU32)
}

// FromFloat32SIMD converts two simd.Float32s vectors (even and odd elements) into a single
// simd.Uint16s vector of Float16 values.
func FromFloat32SIMD(even, odd simd.Float32s) simd.Uint16s {
	evenF16 := float32sToF16Bits(even)
	oddF16 := float32sToF16Bits(odd)
	packed := evenF16.Or(oddF16.ShiftAllLeft(16))
	return packed.ReshapeToUint16s()
}

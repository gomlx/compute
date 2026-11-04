// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package float16

import (
	"simd/archsimd"
	"unsafe"
)

// Float16x8 is an alias for archsimd.Uint16x8 representing 8 Float16 values in a 128-bit NEON register.
type Float16x8 archsimd.Uint16x8

// LoadFloat16x8 loads 8 Float16 values from the given array pointer into a Float16x8 vector.
func LoadFloat16x8(ptr *[8]Float16) Float16x8 {
	vec := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(ptr)))
	return Float16x8(vec)
}

// LoadFloat16x8Slice loads 8 Float16 values from the beginning of the slice into a Float16x8 vector.
func LoadFloat16x8Slice(s []Float16) Float16x8 {
	vec := archsimd.LoadUint16x8((*[8]uint16)(unsafe.Pointer(&s[0]))[:])
	return Float16x8(vec)
}

// Store stores 8 Float16 values into the given array pointer.
func (v Float16x8) Store(ptr *[8]Float16) {
	archsimd.Uint16x8(v).StoreArray((*[8]uint16)(unsafe.Pointer(ptr)))
}

// StoreSlice stores 8 Float16 values into the beginning of the slice.
func (v Float16x8) StoreSlice(s []Float16) {
	archsimd.Uint16x8(v).Store((*[8]uint16)(unsafe.Pointer(&s[0]))[:])
}

// ToFloat32 converts a vector of 8 Float16 values into two archsimd.Float32x4 vectors (lower 4 and upper 4 lanes).
// It is fully inlineable inside Go SIMD loops.
func (v Float16x8) ToFloat32() (lo, hi archsimd.Float32x4) {
	vec := archsimd.Uint16x8(v)
	lo = neonConvertF16ToF32(vec.ExtendLo4ToUint32())
	hi = neonConvertF16ToF32(vec.HiToLo().ExtendLo4ToUint32())
	return lo, hi
}

// FromFloat32x4 converts two archsimd.Float32x4 vectors (lower 4 and upper 4 lanes) into a Float16x8 vector
// using IEEE-754 round-to-nearest-even. It is fully inlineable inside Go SIMD loops.
func FromFloat32x4(lo, hi archsimd.Float32x4) Float16x8 {
	loBits := neonConvertF32ToF16(lo)
	hiBits := neonConvertF32ToF16(hi)
	// Since each 32-bit lane in loBits and hiBits has its upper 16 bits zeroed,
	// ConcatAddPairs (VADDP on Uint16x8) packs [lo0, 0, lo1, 0, lo2, 0, lo3, 0] and
	// [hi0, 0, hi1, 0, hi2, 0, hi3, 0] into [lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3] in 1 instruction.
	packed := loBits.ReshapeToUint16s().ConcatAddPairs(hiBits.ReshapeToUint16s())
	return Float16x8(packed)
}

func neonConvertF16ToF32(u archsimd.Uint32x4) archsimd.Float32x4 {
	sign := u.And(archsimd.BroadcastUint32x4(0x8000)).ShiftAllLeft(16)
	exp := u.And(archsimd.BroadcastUint32x4(0x7C00)).ShiftAllRight(10)
	mantissa := u.And(archsimd.BroadcastUint32x4(0x03FF))

	normExp := exp.Add(archsimd.BroadcastUint32x4(112)).ShiftAllLeft(23)
	normMant := mantissa.ShiftAllLeft(13)
	normRes := sign.Or(normExp).Or(normMant).BitsToFloat32()

	zeroU32 := archsimd.BroadcastUint32x4(0)
	isZero := exp.Equal(zeroU32).And(mantissa.Equal(zeroU32))
	isDenorm := exp.Equal(zeroU32).And(mantissa.Greater(zeroU32))
	isInfNaN := exp.Equal(archsimd.BroadcastUint32x4(0x1F))

	zeroRes := sign.BitsToFloat32()
	infNaNRes := sign.Or(archsimd.BroadcastUint32x4(0x7F800000)).Or(normMant).BitsToFloat32()

	// Hardware FPU subtraction aligns denormal mantissa: (magic + mantissa) - magic
	magicBits := archsimd.BroadcastUint32x4(113 << 23).Or(normMant)
	magicF32 := magicBits.BitsToFloat32()
	biasF32 := archsimd.BroadcastUint32x4(113 << 23).BitsToFloat32()
	denormF32 := magicF32.Sub(biasF32)
	isNeg := sign.Equal(archsimd.BroadcastUint32x4(0x80000000))
	denormRes := denormF32.Neg().IfElse(isNeg, denormF32)

	res := normRes
	res = zeroRes.IfElse(isZero, res)
	res = infNaNRes.IfElse(isInfNaN, res)
	res = denormRes.IfElse(isDenorm, res)
	return res
}

func neonConvertF32ToF16(f archsimd.Float32x4) archsimd.Uint32x4 {
	u := f.ToBits()
	sign := u.ShiftAllRight(16).And(archsimd.BroadcastUint32x4(0x8000))
	uAbs := u.And(archsimd.BroadcastUint32x4(0x7FFFFFFF))

	f16MantissaMask := archsimd.BroadcastUint32x4(0x03FF)

	isOverflow := uAbs.GreaterEqual(archsimd.BroadcastUint32x4(0x47800000))
	isNan := uAbs.Greater(archsimd.BroadcastUint32x4(0x7F800000))
	nanRes := sign.Or(archsimd.BroadcastUint32x4(0x7C00)).Or(uAbs.ShiftAllRight(13).And(f16MantissaMask)).Or(archsimd.BroadcastUint32x4(1))
	infRes := sign.Or(archsimd.BroadcastUint32x4(0x7C00))
	overRes := nanRes.IfElse(isNan, infRes)

	isUnder := uAbs.Less(archsimd.BroadcastUint32x4(0x38800000))
	isZero := uAbs.Less(archsimd.BroadcastUint32x4(0x33000000))

	fAbs := uAbs.BitsToFloat32()
	denormMagicBits := archsimd.BroadcastUint32x4(126 << 23)
	denormMagic := denormMagicBits.BitsToFloat32()
	fDenorm := fAbs.Add(denormMagic)
	denormMant := fDenorm.ToBits().Sub(denormMagicBits).And(f16MantissaMask)
	denormRes := sign.IfElse(isZero, sign.Or(denormMant))

	bias := archsimd.BroadcastUint32x4(0x0FFF).Add(uAbs.ShiftAllRight(13).And(archsimd.BroadcastUint32x4(1)))
	rounded := uAbs.Add(bias)
	normExpAdjusted := rounded.Sub(archsimd.BroadcastUint32x4(112 << 23))
	isNormOverflow := normExpAdjusted.GreaterEqual(archsimd.BroadcastUint32x4(0x1F << 23))
	normRes := infRes.IfElse(isNormOverflow, sign.Or(normExpAdjusted.ShiftAllRight(13)))

	res := normRes
	res = overRes.IfElse(isOverflow, res)
	res = denormRes.IfElse(isUnder, res)
	return res
}

//go:noescape
func convertF16ToF32NEONAsm(dst, src unsafe.Pointer, n int)

//go:noescape
func convertF32ToF16NEONAsm(dst, src unsafe.Pointer, n int)

// ToFloat32SliceNEON converts a slice of Float16 values to float32 using ARM64 NEON FCVTL/FCVTL2 instructions.
func ToFloat32SliceNEON(dst []float32, src []Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	vecN := n &^ 7
	if vecN > 0 {
		convertF16ToF32NEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), vecN)
	}
	for i := vecN; i < n; i++ {
		dst[i] = src[i].Float32()
	}
}

// FromFloat32SliceNEON converts a slice of float32 values to Float16 using ARM64 NEON FCVTN/FCVTN2 instructions.
func FromFloat32SliceNEON(dst []Float16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	vecN := n &^ 7
	if vecN > 0 {
		convertF32ToF16NEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), vecN)
	}
	for i := vecN; i < n; i++ {
		dst[i] = FromFloat32(src[i])
	}
}

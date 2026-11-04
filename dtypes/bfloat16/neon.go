// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package bfloat16

import (
	"simd/archsimd"
	"unsafe"
)

// BFloat16x8 is an alias for archsimd.Uint16x8 representing 8 BFloat16 values in a 128-bit NEON register.
type BFloat16x8 archsimd.Uint16x8

// LoadBFloat16x8 loads 8 BFloat16 values from the given array pointer into a BFloat16x8 vector.
func LoadBFloat16x8(ptr *[8]BFloat16) BFloat16x8 {
	vec := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(ptr)))
	return BFloat16x8(vec)
}

// LoadBFloat16x8Slice loads 8 BFloat16 values from the beginning of the slice into a BFloat16x8 vector.
func LoadBFloat16x8Slice(s []BFloat16) BFloat16x8 {
	vec := archsimd.LoadUint16x8((*[8]uint16)(unsafe.Pointer(&s[0]))[:])
	return BFloat16x8(vec)
}

// Store stores 8 BFloat16 values into the given array pointer.
func (v BFloat16x8) Store(ptr *[8]BFloat16) {
	archsimd.Uint16x8(v).StoreArray((*[8]uint16)(unsafe.Pointer(ptr)))
}

// StoreSlice stores 8 BFloat16 values into the beginning of the slice.
func (v BFloat16x8) StoreSlice(s []BFloat16) {
	archsimd.Uint16x8(v).Store((*[8]uint16)(unsafe.Pointer(&s[0]))[:])
}

// ToFloat32 converts a vector of 8 BFloat16 values into two archsimd.Float32x4 vectors (lower 4 and upper 4 lanes).
func (v BFloat16x8) ToFloat32() (lo, hi archsimd.Float32x4) {
	vec := archsimd.Uint16x8(v)
	l := vec.ExtendLo4ToUint32().ShiftAllLeft(16)
	h := vec.HiToLo().ExtendLo4ToUint32().ShiftAllLeft(16)
	return l.BitsToFloat32(), h.BitsToFloat32()
}

// FromFloat32x4 converts two archsimd.Float32x4 vectors (lower 4 and upper 4 lanes) into a BFloat16x8 vector
// using IEEE-754 round-to-nearest-even.
func FromFloat32x4(lo, hi archsimd.Float32x4) BFloat16x8 {
	loBF16 := neonConvertF32ToBF16(lo)
	hiBF16 := neonConvertF32ToBF16(hi)
	packed := loBF16.ReshapeToUint16s().ConcatAddPairs(hiBF16.ReshapeToUint16s())
	return BFloat16x8(packed)
}

func neonConvertF32ToBF16(f archsimd.Float32x4) archsimd.Uint32x4 {
	bits := f.ToBits()
	notNaN := f.Equal(f).ToInt32x4().ToBits()
	lsb := bits.ShiftAllRight(16).And(archsimd.BroadcastUint32x4(1))
	bias := archsimd.BroadcastUint32x4(0x7FFF).Add(lsb).And(notNaN)
	return bits.Add(bias).ShiftAllRight(16)
}

//go:noescape
func convertBF16ToF32NEONAsm(dst, src unsafe.Pointer, n int)

//go:noescape
func convertF32ToBF16NEONAsm(dst, src unsafe.Pointer, n int)

// ToFloat32SliceNEON converts a slice of BFloat16 values to float32 using ARM64 NEON.
func ToFloat32SliceNEON(dst []float32, src []BFloat16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	vecN := n &^ 7
	if vecN > 0 {
		convertBF16ToF32NEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), vecN)
	}
	for i := vecN; i < n; i++ {
		dst[i] = src[i].Float32()
	}
}

// FromFloat32SliceNEON converts a slice of float32 values to BFloat16 using ARM64 NEON.
func FromFloat32SliceNEON(dst []BFloat16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	vecN := n &^ 7
	if vecN > 0 {
		convertF32ToBF16NEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), vecN)
	}
	for i := vecN; i < n; i++ {
		dst[i] = FromFloat32(src[i])
	}
}

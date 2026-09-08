// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package bfloat16

import (
	"simd"
	"unsafe"
)

// LoadBFloat16s loads a vector of BFloat16 values into a simd.Uint16s vector.
// Slice must have at least the vector length.
func LoadBFloat16s(s []BFloat16) simd.Uint16s {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	return simd.LoadUint16s(u16Slice)
}

// LoadBFloat16sPart loads a partial slice of BFloat16 values into a simd.Uint16s vector,
// returning the vector and the count of loaded elements.
func LoadBFloat16sPart(s []BFloat16) (simd.Uint16s, int) {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	return simd.LoadUint16sPart(u16Slice)
}

// StoreBFloat16s stores a simd.Uint16s vector into a BFloat16 slice.
func StoreBFloat16s(v simd.Uint16s, s []BFloat16) {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	v.Store(u16Slice)
}

// StoreBFloat16sPart stores a partial simd.Uint16s vector into a BFloat16 slice and returns
// the number of elements stored.
func StoreBFloat16sPart(v simd.Uint16s, s []BFloat16) int {
	u16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
	return v.StorePart(u16Slice)
}

// ToFloat32SIMD converts a simd.Uint16s vector representing BFloat16 values into two
// simd.Float32s vectors (even-indexed elements and odd-indexed elements).
func ToFloat32SIMD(v simd.Uint16s) (even, odd simd.Float32s) {
	u32 := v.ReshapeToUint32s()
	even = u32.ShiftAllLeft(16).BitsToFloat32()
	odd = u32.And(simd.BroadcastUint32s(0xFFFF0000)).BitsToFloat32()
	return even, odd
}

// FromFloat32SIMD converts two simd.Float32s vectors (even and odd elements) into a single
// simd.Uint16s vector of BFloat16 values using round-to-nearest-even.
func FromFloat32SIMD(even, odd simd.Float32s) simd.Uint16s {
	evenBits := even.ToBits()
	evenBias := simd.BroadcastUint32s(0x7FFF).Add(evenBits.ShiftAllRight(16).And(simd.BroadcastUint32s(1)))
	evenRounded := evenBits.Add(evenBias).ShiftAllRight(16)

	oddBits := odd.ToBits()
	oddBias := simd.BroadcastUint32s(0x7FFF).Add(oddBits.ShiftAllRight(16).And(simd.BroadcastUint32s(1)))
	oddRounded := oddBits.Add(oddBias).And(simd.BroadcastUint32s(0xFFFF0000))

	packed := evenRounded.Or(oddRounded)
	return packed.ReshapeToUint16s()
}

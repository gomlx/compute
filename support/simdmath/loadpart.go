// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package simdmath

import "simd"

// Temporary safe replacements for simd.Load*sPart.
//
// In Go 1.27.1–1.27.3, simd/archsimd.Load*Part on amd64 lowers to an unmasked
// full-vector load (VMOVDQU / VMOVDQU64) followed by a register-to-register mask
// instead of a hardware fault-suppressing masked load (VPMASKMOV* / VMOVDQU* K).
// When a tail slice is within 64 bytes of an unmapped page boundary, the over-read
// crosses into the unmapped page and triggers SIGSEGV.
//
// Staging the tail (< vLen elements) through a 64-byte stack buffer avoids reading
// past len(s).
//
// TODO: Once a Go release includes the upstream fix for https://github.com/golang/go/issues/81692,
// replace `simdmath.Load*Part` back with `simd.Load*Part` across the codebase and
// remove this file.

// LoadFloat32sPart safely loads up to vLen float32 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadFloat32sPart(s []float32) (simd.Float32s, int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if len(s) >= vLen {
		return simd.LoadFloat32s(s), vLen
	}
	var buf [16]float32
	n := copy(buf[:], s)
	return simd.LoadFloat32s(buf[:]), n
}

// LoadFloat64sPart safely loads up to vLen float64 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadFloat64sPart(s []float64) (simd.Float64s, int) {
	vLen := simd.BroadcastFloat64s(0).Len()
	if len(s) >= vLen {
		return simd.LoadFloat64s(s), vLen
	}
	var buf [8]float64
	n := copy(buf[:], s)
	return simd.LoadFloat64s(buf[:]), n
}

// LoadInt8sPart safely loads up to vLen int8 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadInt8sPart(s []int8) (simd.Int8s, int) {
	vLen := simd.BroadcastInt8s(0).Len()
	if len(s) >= vLen {
		return simd.LoadInt8s(s), vLen
	}
	var buf [64]int8
	n := copy(buf[:], s)
	return simd.LoadInt8s(buf[:]), n
}

// LoadInt16sPart safely loads up to vLen int16 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadInt16sPart(s []int16) (simd.Int16s, int) {
	vLen := simd.BroadcastInt16s(0).Len()
	if len(s) >= vLen {
		return simd.LoadInt16s(s), vLen
	}
	var buf [32]int16
	n := copy(buf[:], s)
	return simd.LoadInt16s(buf[:]), n
}

// LoadInt32sPart safely loads up to vLen int32 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadInt32sPart(s []int32) (simd.Int32s, int) {
	vLen := simd.BroadcastInt32s(0).Len()
	if len(s) >= vLen {
		return simd.LoadInt32s(s), vLen
	}
	var buf [16]int32
	n := copy(buf[:], s)
	return simd.LoadInt32s(buf[:]), n
}

// LoadInt64sPart safely loads up to vLen int64 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadInt64sPart(s []int64) (simd.Int64s, int) {
	vLen := simd.BroadcastInt64s(0).Len()
	if len(s) >= vLen {
		return simd.LoadInt64s(s), vLen
	}
	var buf [8]int64
	n := copy(buf[:], s)
	return simd.LoadInt64s(buf[:]), n
}

// LoadUint8sPart safely loads up to vLen uint8 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadUint8sPart(s []uint8) (simd.Uint8s, int) {
	vLen := simd.BroadcastUint8s(0).Len()
	if len(s) >= vLen {
		return simd.LoadUint8s(s), vLen
	}
	var buf [64]uint8
	n := copy(buf[:], s)
	return simd.LoadUint8s(buf[:]), n
}

// LoadUint16sPart safely loads up to vLen uint16 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadUint16sPart(s []uint16) (simd.Uint16s, int) {
	vLen := simd.BroadcastUint16s(0).Len()
	if len(s) >= vLen {
		return simd.LoadUint16s(s), vLen
	}
	var buf [32]uint16
	n := copy(buf[:], s)
	return simd.LoadUint16s(buf[:]), n
}

// LoadUint32sPart safely loads up to vLen uint32 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadUint32sPart(s []uint32) (simd.Uint32s, int) {
	vLen := simd.BroadcastUint32s(0).Len()
	if len(s) >= vLen {
		return simd.LoadUint32s(s), vLen
	}
	var buf [16]uint32
	n := copy(buf[:], s)
	return simd.LoadUint32s(buf[:]), n
}

// LoadUint64sPart safely loads up to vLen uint64 elements from s, zero-filling remaining lanes.
//
// Deprecated: temporary work around until release of fix in https://github.com/golang/go/issues/81692
func LoadUint64sPart(s []uint64) (simd.Uint64s, int) {
	vLen := simd.BroadcastUint64s(0).Len()
	if len(s) >= vLen {
		return simd.LoadUint64s(s), vLen
	}
	var buf [8]uint64
	n := copy(buf[:], s)
	return simd.LoadUint64s(buf[:]), n
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"simd/archsimd"
	"unsafe"
)

func expandCond16NEON(condPtr unsafe.Pointer) (m0, m1, m2, m3 archsimd.Mask32x4) {
	c16 := archsimd.LoadUint8x16Array((*[16]uint8)(condPtr))
	cLo8 := c16.ExtendLo8ToUint16()
	cHi8 := c16.HiToLo().ExtendLo8ToUint16()
	m0 = cLo8.ExtendLo4ToUint32().BitsToInt32().ToMask()
	m1 = cLo8.HiToLo().ExtendLo4ToUint32().BitsToInt32().ToMask()
	m2 = cHi8.ExtendLo4ToUint32().BitsToInt32().ToMask()
	m3 = cHi8.HiToLo().ExtendLo4ToUint32().BitsToInt32().ToMask()
	return
}

// DispatchWhereFloat32NEON executes Where on contiguous float32 slices using ARM64 NEON SIMD.
func DispatchWhereFloat32NEON(cond []bool, onTrue, onFalse, out []float32, onTrueIsScalar, onFalseIsScalar bool) bool {
	n := len(out)
	if n == 0 {
		return true
	}

	switch {
	case !onTrueIsScalar && !onFalseIsScalar:
		i := 0
		for ; i+16 <= n; i += 16 {
			m0, m1, m2, m3 := expandCond16NEON(unsafe.Pointer(&cond[i]))
			t0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i])))
			t1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i+4])))
			t2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i+8])))
			t3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i+12])))
			f0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i])))
			f1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i+4])))
			f2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i+8])))
			f3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i+12])))
			t0.IfElse(m0, f0).StoreArray((*[4]float32)(unsafe.Pointer(&out[i])))
			t1.IfElse(m1, f1).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+4])))
			t2.IfElse(m2, f2).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+8])))
			t3.IfElse(m3, f3).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+12])))
		}
		for ; i < n; i++ {
			if cond[i] {
				out[i] = onTrue[i]
			} else {
				out[i] = onFalse[i]
			}
		}

	case !onTrueIsScalar && onFalseIsScalar:
		fScalar := onFalse[0]
		fv := archsimd.BroadcastFloat32x4(fScalar)
		i := 0
		for ; i+16 <= n; i += 16 {
			m0, m1, m2, m3 := expandCond16NEON(unsafe.Pointer(&cond[i]))
			t0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i])))
			t1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i+4])))
			t2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i+8])))
			t3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onTrue[i+12])))
			t0.IfElse(m0, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i])))
			t1.IfElse(m1, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+4])))
			t2.IfElse(m2, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+8])))
			t3.IfElse(m3, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+12])))
		}
		for ; i < n; i++ {
			if cond[i] {
				out[i] = onTrue[i]
			} else {
				out[i] = fScalar
			}
		}

	case onTrueIsScalar && !onFalseIsScalar:
		tScalar := onTrue[0]
		tv := archsimd.BroadcastFloat32x4(tScalar)
		i := 0
		for ; i+16 <= n; i += 16 {
			m0, m1, m2, m3 := expandCond16NEON(unsafe.Pointer(&cond[i]))
			f0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i])))
			f1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i+4])))
			f2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i+8])))
			f3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&onFalse[i+12])))
			tv.IfElse(m0, f0).StoreArray((*[4]float32)(unsafe.Pointer(&out[i])))
			tv.IfElse(m1, f1).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+4])))
			tv.IfElse(m2, f2).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+8])))
			tv.IfElse(m3, f3).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+12])))
		}
		for ; i < n; i++ {
			if cond[i] {
				out[i] = tScalar
			} else {
				out[i] = onFalse[i]
			}
		}

	case onTrueIsScalar && onFalseIsScalar:
		tScalar := onTrue[0]
		fScalar := onFalse[0]
		tv := archsimd.BroadcastFloat32x4(tScalar)
		fv := archsimd.BroadcastFloat32x4(fScalar)
		i := 0
		for ; i+16 <= n; i += 16 {
			m0, m1, m2, m3 := expandCond16NEON(unsafe.Pointer(&cond[i]))
			tv.IfElse(m0, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i])))
			tv.IfElse(m1, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+4])))
			tv.IfElse(m2, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+8])))
			tv.IfElse(m3, fv).StoreArray((*[4]float32)(unsafe.Pointer(&out[i+12])))
		}
		for ; i < n; i++ {
			if cond[i] {
				out[i] = tScalar
			} else {
				out[i] = fScalar
			}
		}
	}

	return true
}

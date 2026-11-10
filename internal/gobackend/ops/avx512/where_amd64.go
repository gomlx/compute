// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import "unsafe"

//go:noescape
func whereVVFloat32AVX512Asm(cond, onTrue, onFalse, out unsafe.Pointer, count int)

//go:noescape
func whereVSFloat32AVX512Asm(cond, onTrue, onFalseScalar, out unsafe.Pointer, count int)

//go:noescape
func whereSVFloat32AVX512Asm(cond, onTrueScalar, onFalse, out unsafe.Pointer, count int)

//go:noescape
func whereSSFloat32AVX512Asm(cond, onTrueScalar, onFalseScalar, out unsafe.Pointer, count int)

// DispatchWhereFloat32AVX512 executes Where on contiguous float32 slices using AVX-512 assembly.
func DispatchWhereFloat32AVX512(cond []bool, onTrue, onFalse, out []float32, onTrueIsScalar, onFalseIsScalar bool) bool {
	if len(out) == 0 {
		return true
	}
	condPtr := unsafe.Pointer(&cond[0])
	outPtr := unsafe.Pointer(&out[0])
	var truePtr, falsePtr unsafe.Pointer
	if len(onTrue) > 0 {
		truePtr = unsafe.Pointer(&onTrue[0])
	}
	if len(onFalse) > 0 {
		falsePtr = unsafe.Pointer(&onFalse[0])
	}

	switch {
	case !onTrueIsScalar && !onFalseIsScalar:
		whereVVFloat32AVX512Asm(condPtr, truePtr, falsePtr, outPtr, len(out))
	case !onTrueIsScalar && onFalseIsScalar:
		whereVSFloat32AVX512Asm(condPtr, truePtr, falsePtr, outPtr, len(out))
	case onTrueIsScalar && !onFalseIsScalar:
		whereSVFloat32AVX512Asm(condPtr, truePtr, falsePtr, outPtr, len(out))
	case onTrueIsScalar && onFalseIsScalar:
		whereSSFloat32AVX512Asm(condPtr, truePtr, falsePtr, outPtr, len(out))
	}
	return true
}

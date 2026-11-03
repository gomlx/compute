// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package matmul

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

func addBiasFloat32Arch(row, bias []float32) bool {
	if !gobackend.IsNEONAllowed {
		return false
	}
	n := min(len(row), len(bias))
	i := 0
	for ; i+16 <= n; i += 16 {
		r0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&row[i])))
		r1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&row[i+4])))
		r2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&row[i+8])))
		r3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&row[i+12])))
		b0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i])))
		b1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i+4])))
		b2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i+8])))
		b3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i+12])))
		r0.Add(b0).StoreArray((*[4]float32)(unsafe.Pointer(&row[i])))
		r1.Add(b1).StoreArray((*[4]float32)(unsafe.Pointer(&row[i+4])))
		r2.Add(b2).StoreArray((*[4]float32)(unsafe.Pointer(&row[i+8])))
		r3.Add(b3).StoreArray((*[4]float32)(unsafe.Pointer(&row[i+12])))
	}
	for ; i+4 <= n; i += 4 {
		r0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&row[i])))
		b0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i])))
		r0.Add(b0).StoreArray((*[4]float32)(unsafe.Pointer(&row[i])))
	}
	for ; i < n; i++ {
		row[i] += bias[i]
	}
	return true
}

func copyAndAddBiasFloat32Arch(dst, src, bias []float32) bool {
	if !gobackend.IsNEONAllowed {
		return false
	}
	n := min(len(dst), len(src), len(bias))
	i := 0
	for ; i+16 <= n; i += 16 {
		s0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&src[i])))
		s1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&src[i+4])))
		s2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&src[i+8])))
		s3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&src[i+12])))
		b0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i])))
		b1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i+4])))
		b2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i+8])))
		b3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i+12])))
		s0.Add(b0).StoreArray((*[4]float32)(unsafe.Pointer(&dst[i])))
		s1.Add(b1).StoreArray((*[4]float32)(unsafe.Pointer(&dst[i+4])))
		s2.Add(b2).StoreArray((*[4]float32)(unsafe.Pointer(&dst[i+8])))
		s3.Add(b3).StoreArray((*[4]float32)(unsafe.Pointer(&dst[i+12])))
	}
	for ; i+4 <= n; i += 4 {
		s0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&src[i])))
		b0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bias[i])))
		s0.Add(b0).StoreArray((*[4]float32)(unsafe.Pointer(&dst[i])))
	}
	for ; i < n; i++ {
		dst[i] = src[i] + bias[i]
	}
	return true
}

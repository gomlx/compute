// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

package matmul

import (
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

//go:noescape
func addBiasFloat32AVX512Asm(row, bias unsafe.Pointer, n int)

//go:noescape
func addBiasFloat32AVX2Asm(row, bias unsafe.Pointer, n int)

//go:noescape
func copyAndAddBiasFloat32AVX512Asm(dst, src, bias unsafe.Pointer, n int)

//go:noescape
func copyAndAddBiasFloat32AVX2Asm(dst, src, bias unsafe.Pointer, n int)

var (
	hasAVX512 bool
	hasAVX2   bool
)

func init() {
	hasAVX512 = gobackend.IsAVX512Allowed
	hasAVX2 = gobackend.IsAVX2Allowed
}

func addBiasFloat32Arch(row, bias []float32) bool {
	n := min(len(row), len(bias))
	if n == 0 {
		return true
	}
	if hasAVX512 {
		addBiasFloat32AVX512Asm(unsafe.Pointer(&row[0]), unsafe.Pointer(&bias[0]), n)
		return true
	}
	if hasAVX2 {
		addBiasFloat32AVX2Asm(unsafe.Pointer(&row[0]), unsafe.Pointer(&bias[0]), n)
		return true
	}
	return false
}

func copyAndAddBiasFloat32Arch(dst, src, bias []float32) bool {
	n := min(len(dst), len(src), len(bias))
	if n == 0 {
		return true
	}
	if hasAVX512 {
		copyAndAddBiasFloat32AVX512Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), unsafe.Pointer(&bias[0]), n)
		return true
	}
	if hasAVX2 {
		copyAndAddBiasFloat32AVX2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), unsafe.Pointer(&bias[0]), n)
		return true
	}
	return false
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

package matmul

import "unsafe"

// avx2SmallTransposedTile4x2Float32Asm computes a 4 rows x 2 cols tile of transposed dot products for Float32.
//
//go:noescape
func avx2SmallTransposedTile4x2Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

// avx2SmallTransposedTile4x1Float32Asm computes a 4 rows x 1 col tile of transposed dot products for Float32.
//
//go:noescape
func avx2SmallTransposedTile4x1Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

// avx2SmallTransposedTile4x2Float64Asm computes a 4 rows x 2 cols tile of transposed dot products for Float64.
//
//go:noescape
func avx2SmallTransposedTile4x2Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

// avx2SmallTransposedTile4x1Float64Asm computes a 4 rows x 1 col tile of transposed dot products for Float64.
//
//go:noescape
func avx2SmallTransposedTile4x1Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

// avx2SmallTransposedTile4x2Float16Asm computes a 4 rows x 2 cols tile of transposed dot products for Float16 (accumulating into Float32).
//
//go:noescape
func avx2SmallTransposedTile4x2Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

// avx2SmallTransposedTile4x1Float16Asm computes a 4 rows x 1 col tile of transposed dot products for Float16 (accumulating into Float32).
//
//go:noescape
func avx2SmallTransposedTile4x1Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

// avx2SmallTransposedTile4x2BFloat16Asm computes a 4 rows x 2 cols tile of transposed dot products for BFloat16 (accumulating into Float32).
//
//go:noescape
func avx2SmallTransposedTile4x2BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

// avx2SmallTransposedTile4x1BFloat16Asm computes a 4 rows x 1 col tile of transposed dot products for BFloat16 (accumulating into Float32).
//
//go:noescape
func avx2SmallTransposedTile4x1BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingLen int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

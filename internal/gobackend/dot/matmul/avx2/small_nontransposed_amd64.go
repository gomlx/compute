// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

package avx2

import "unsafe"

// Float32 Non-Transposed Microkernels

// avx2SmallNonTransposedTile4x16Float32Asm computes a 4 rows x 16 cols tile for Float32.
//
//go:noescape
func avx2SmallNonTransposedTile4x16Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// avx2SmallNonTransposedTile4x8Float32Asm computes a 4 rows x 8 cols tile for Float32.
//
//go:noescape
func avx2SmallNonTransposedTile4x8Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// Float64 Non-Transposed Microkernels

// avx2SmallNonTransposedTile4x8Float64Asm computes a 4 rows x 8 cols tile for Float64.
//
//go:noescape
func avx2SmallNonTransposedTile4x8Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// avx2SmallNonTransposedTile4x4Float64Asm computes a 4 rows x 4 cols tile for Float64.
//
//go:noescape
func avx2SmallNonTransposedTile4x4Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// Float16 Non-Transposed Microkernels (accumulates into Float32)

// avx2SmallNonTransposedTile4x16Float16Asm computes a 4 rows x 16 cols tile for Float16.
//
//go:noescape
func avx2SmallNonTransposedTile4x16Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// avx2SmallNonTransposedTile4x8Float16Asm computes a 4 rows x 8 cols tile for Float16.
//
//go:noescape
func avx2SmallNonTransposedTile4x8Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// BFloat16 Non-Transposed Microkernels (accumulates into Float32)

// avx2SmallNonTransposedTile4x16BFloat16Asm computes a 4 rows x 16 cols tile for BFloat16.
//
//go:noescape
func avx2SmallNonTransposedTile4x16BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// avx2SmallNonTransposedTile4x8BFloat16Asm computes a 4 rows x 8 cols tile for BFloat16.
//
//go:noescape
func avx2SmallNonTransposedTile4x8BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingLen int,
	rowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

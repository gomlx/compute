// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package matmul

import "unsafe"

// avx512SmallTransposedTile4x4Float32Asm computes a single 4x4 tile:
// 4 rows of LHS x 4 rows of RHS over contractingSize floats.
// The 16 scalar results are written to outRow0..3 (4 floats each).
//
//go:noescape
func avx512SmallTransposedTile4x4Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

// avx512SmallTransposedTile4x4Float64Asm computes a single 4x4 tile for Float64.
//
//go:noescape
func avx512SmallTransposedTile4x4Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

// avx512SmallTransposedTile4x4Float16Asm computes a single 4x4 tile for Float16 (output Float32).
//
//go:noescape
func avx512SmallTransposedTile4x4Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

// avx512SmallTransposedTile4x4BFloat16Asm computes a single 4x4 tile for BFloat16 (output Float32).
//
//go:noescape
func avx512SmallTransposedTile4x4BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

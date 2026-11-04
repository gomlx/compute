// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import "unsafe"

// NonTransposed small matrix assembly microkernels (accumulating across K in V0-V15 registers).

//go:noescape
func neonSmallNonTransposedTile4x16Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func neonSmallNonTransposedTile4x4Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func neonSmallNonTransposedTile4x16Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func neonSmallNonTransposedTile4x8Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func neonSmallNonTransposedTile4x16BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
	useBFDOT bool,
)

//go:noescape
func neonSmallNonTransposedTile4x8BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
	useBFDOT bool,
)

//go:noescape
func neonSmallNonTransposedTile4x8Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func neonSmallNonTransposedTile4x2Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhsCol unsafe.Pointer,
	contractingSize int,
	rhsRowStrideBytes int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// Transposed small matrix assembly microkernels (vectorizing across K with FADDP horizontal reduction).

//go:noescape
func neonSmallTransposedTile4x4Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

//go:noescape
func neonSmallTransposedTile4x1Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

//go:noescape
func neonSmallTransposedTile4x4Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

//go:noescape
func neonSmallTransposedTile4x1Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

//go:noescape
func neonSmallTransposedTile4x4BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
	useBFDOT bool,
)

//go:noescape
func neonSmallTransposedTile4x1BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
	useBFDOT bool,
)

//go:noescape
func neonSmallTransposedTile4x4Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

//go:noescape
func neonSmallTransposedTile4x1Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingSize int,
	outPtr unsafe.Pointer,
	outStrideBytes int,
)

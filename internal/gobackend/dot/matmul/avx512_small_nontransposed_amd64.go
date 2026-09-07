// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package matmul

import "unsafe"

// Float32 microkernels:
//
//go:noescape
func avx512SmallNonTransposedTile4x64Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

//go:noescape
func avx512SmallNonTransposedTile4x16Float32Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

// Float64 microkernels:
//
//go:noescape
func avx512SmallNonTransposedTile4x32Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

//go:noescape
func avx512SmallNonTransposedTile4x8Float64Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

// Float16 microkernels:
//
//go:noescape
func avx512SmallNonTransposedTile4x64Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

//go:noescape
func avx512SmallNonTransposedTile4x16Float16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

// BFloat16 microkernels:
//
//go:noescape
func avx512SmallNonTransposedTile4x64BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

//go:noescape
func avx512SmallNonTransposedTile4x16BFloat16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer)

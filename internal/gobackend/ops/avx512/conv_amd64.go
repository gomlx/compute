// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import "unsafe"

// ConvAVX512Kernel8x32Float32Asm computes an 8x32 float32 output tile directly from
// 8 rows of LHS ([8, K] with row byte stride lhsRowStrideBytes) and 32 columns of
// packed RHS ([K, 32] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvAVX512Kernel8x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX512Kernel4x32Float32Asm computes a 4x32 float32 output tile directly from
// 4 rows of LHS ([4, K] with row byte stride lhsRowStrideBytes) and 32 columns of
// packed RHS ([K, 32] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvAVX512Kernel4x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvKernelGradAVX512_8x32Float32Asm accumulates an 8x32 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 8x32 result to outPtr.
//
//go:noescape
func ConvKernelGradAVX512_8x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradAVX512_4x32Float32Asm accumulates a 4x32 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 4x32 result to outPtr.
//
//go:noescape
func ConvKernelGradAVX512_4x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradAVX512_1x32Float32Asm accumulates a 1x32 float32 kernel-gradient row
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 32-float row to outPtr.
//
//go:noescape
func ConvKernelGradAVX512_1x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep int,
)

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import "unsafe"

// ConvAVX2Kernel4x16Float32Asm computes a 4x16 float32 output tile directly from
// 4 rows of LHS ([4, K] with row byte stride lhsRowStrideBytes) and 16 columns of
// packed RHS ([K, 16+] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvAVX2Kernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX2Direct2DKernel4x16Float32Asm computes a 4x16 float32 output tile for
// 4 horizontally adjacent interior 2D output pixels directly from the input tensor
// without im2col patch extraction, iterating across numKy valid filter rows of
// width kWC = kW * inC each and advancing LHS row pointers by lhsKyJumpBytes
// between filter rows.
//
//go:noescape
func ConvAVX2Direct2DKernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvKernelGradAVX2_4x16Float32Asm accumulates a 4x16 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 4x16 result to outPtr.
//
//go:noescape
func ConvKernelGradAVX2_4x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradAVX2_1x16Float32Asm accumulates a 1x16 float32 kernel-gradient row
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 16-float row to outPtr.
//
//go:noescape
func ConvKernelGradAVX2_1x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep int,
)

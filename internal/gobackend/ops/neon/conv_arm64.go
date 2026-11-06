// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import "unsafe"

// ConvNEONKernel4x16Float32Asm computes a 4x16 float32 output tile directly from
// 4 rows of LHS ([4, K] with row byte stride lhsRowStrideBytes) and 16 columns of
// packed RHS ([K, 16+] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvNEONKernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvNEONDirect2DKernel4x16Float32Asm computes a 4x16 float32 output tile directly from
// 4 horizontally adjacent 2D input positions (with pixel byte stride lhsRowStrideBytes)
// across numKy valid filter rows of width kWC = kW * inC (jumping by lhsKyJumpBytes after
// each filter row), without intermediate im2col patch extraction.
//
//go:noescape
func ConvNEONDirect2DKernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvKernelGradNEON4x16Float32Asm accumulates a 4x16 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 4x16 result to outPtr.
//
//go:noescape
func ConvKernelGradNEON4x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradNEON1x16Float32Asm accumulates a 1x16 float32 kernel-gradient row
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 16-float row to outPtr.
//
//go:noescape
func ConvKernelGradNEON1x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep int,
)

// ConvPackLHS8StridedFloat32Asm packs 8 rows of length K from srcPtr (with row byte stride
// srcRowStrideBytes) into an 8-column stripe at dstPtr (with panel row byte stride
// panelColStrideBytes, e.g., 128 bytes for a 32-row SME panel or 64 bytes for a 16-row panel).
//
//go:noescape
func ConvPackLHS8StridedFloat32Asm(
	srcPtr, dstPtr unsafe.Pointer,
	K, srcRowStrideBytes, panelColStrideBytes int,
)

// ConvSMEKernel32x32Float32Asm computes a single 32x32 float32 output tile (validRows <= 32,
// validCols <= 32) from packedLHS ([K, 32]) and packedRHS ([K, 32]) using 4 SME ZA tiles
// (ZA0..ZA3) and stores predicated rows/cols directly to outPtr.
//
//go:noescape
func ConvSMEKernel32x32Float32Asm(
	packedLHS, packedRHS, outPtr unsafe.Pointer,
	K, outRowStrideBytes, validRows, validCols int,
)

// ConvSMEKernel16x16Float32Asm computes a single 16x16 float32 output tile (validRows <= 16,
// validCols <= 16) from packedLHS ([K, 16]) and packedRHS ([K, 16]) using 1 SME ZA tile
// (ZA0) and stores predicated rows/cols directly to outPtr.
//
//go:noescape
func ConvSMEKernel16x16Float32Asm(
	packedLHS, packedRHS, outPtr unsafe.Pointer,
	K, outRowStrideBytes, validRows, validCols int,
)

// ConvKernelGradSME32x32Float32Asm accumulates a 32x32 float32 kernel-gradient tile
// (validRows <= 32, validCols <= 32) directly in SME ZA0..ZA3 across the 3D domain
// (batchSize, numY, numX) using predicated LD1W loads from xBase and vBase and stores
// predicated rows/cols to outPtr.
//
//go:noescape
func ConvKernelGradSME32x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
	validRows, validCols, accumulate int,
)

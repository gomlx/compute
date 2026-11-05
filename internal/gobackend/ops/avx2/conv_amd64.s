// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// ===========================================================================
// 1. ConvAVX2Kernel4x16Float32Asm: Direct 4x16 AVX2 Convolution Microkernel
// ===========================================================================
// Multiplies 4 rows of LHS ([4, K] with row stride lhsRowStrideBytes) by a
// 16-column panel of packed RHS ([K, 16+] with step stride rhsStepBytes) and
// writes the 4x16 tile to outPtr (with row stride outRowStrideBytes).
// Reads directly from the 4 LHS rows via VBROADCASTSS, requiring zero LHS
// transposition or packing.
//
// func ConvAVX2Kernel4x16Float32Asm(
//     lhsPtr, rhsPtr, outPtr unsafe.Pointer,
//     K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int)
TEXT ·ConvAVX2Kernel4x16Float32Asm(SB), NOSPLIT, $0-56
	MOVQ lhsPtr+0(FP), SI                // SI = LHS row 0
	MOVQ rhsPtr+8(FP), DI                // DI = RHS panel pointer
	MOVQ outPtr+16(FP), R8               // R8 = output pointer
	MOVQ K+24(FP), CX                    // CX = K
	MOVQ lhsRowStrideBytes+32(FP), DX    // DX = lhsRowStrideBytes
	MOVQ rhsStepBytes+40(FP), R9         // R9 = rhsStepBytes
	MOVQ outRowStrideBytes+48(FP), R10   // R10 = outRowStrideBytes

	// Zero 8 YMM accumulators (4 rows x 16 cols)
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	// Set up LHS row pointers for rows 1, 2, 3
	LEAQ (SI)(DX*1), R11                 // R11 = LHS row 1
	LEAQ (R11)(DX*1), R12                // R12 = LHS row 2
	LEAQ (R12)(DX*1), R13                // R13 = LHS row 3

	TESTQ CX, CX
	JLE conv_avx2_4x16_store

	MOVQ CX, AX
	SHRQ $1, AX                          // AX = K / 2 pairs
	JZ conv_avx2_4x16_odd

	// Prime Buffer A for step 0
	VMOVUPS (DI), Y8
	VMOVUPS 32(DI), Y9
	VBROADCASTSS (SI), Y10
	VBROADCASTSS (R11), Y11
	VBROADCASTSS (R12), Y12
	VBROADCASTSS (R13), Y13
	ADDQ R9, DI

	DECQ AX
	JZ conv_avx2_4x16_last_pair

	PCALIGN $32
conv_avx2_4x16_pair_loop:
	// Step A: FMAs using Buffer A (Y8, Y9) while loading Buffer B (Y14, Y15, Y10..Y13)
	VFMADD231PS Y8, Y10, Y0
	VMOVUPS (DI), Y14
	VFMADD231PS Y9, Y10, Y1
	VBROADCASTSS 4(SI), Y10

	VFMADD231PS Y8, Y11, Y2
	VMOVUPS 32(DI), Y15
	VFMADD231PS Y9, Y11, Y3
	VBROADCASTSS 4(R11), Y11

	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VBROADCASTSS 4(R12), Y12

	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS 4(R13), Y13

	ADDQ $8, SI
	ADDQ $8, R11
	ADDQ $8, R12
	ADDQ $8, R13
	ADDQ R9, DI

	// Step B: FMAs using Buffer B (Y14, Y15) while loading next Buffer A (Y8, Y9, Y10..Y13)
	VFMADD231PS Y14, Y10, Y0
	VMOVUPS (DI), Y8
	VFMADD231PS Y15, Y10, Y1
	VBROADCASTSS (SI), Y10

	VFMADD231PS Y14, Y11, Y2
	VMOVUPS 32(DI), Y9
	VFMADD231PS Y15, Y11, Y3
	VBROADCASTSS (R11), Y11

	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y15, Y12, Y5
	VBROADCASTSS (R12), Y12

	VFMADD231PS Y14, Y13, Y6
	VFMADD231PS Y15, Y13, Y7
	VBROADCASTSS (R13), Y13

	ADDQ R9, DI

	DECQ AX
	JNZ conv_avx2_4x16_pair_loop

conv_avx2_4x16_last_pair:
	// Step A: FMAs using Buffer A (Y8, Y9) while loading Buffer B (Y14, Y15, Y10..Y13)
	VFMADD231PS Y8, Y10, Y0
	VMOVUPS (DI), Y14
	VFMADD231PS Y9, Y10, Y1
	VBROADCASTSS 4(SI), Y10

	VFMADD231PS Y8, Y11, Y2
	VMOVUPS 32(DI), Y15
	VFMADD231PS Y9, Y11, Y3
	VBROADCASTSS 4(R11), Y11

	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VBROADCASTSS 4(R12), Y12

	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS 4(R13), Y13

	ADDQ $8, SI
	ADDQ $8, R11
	ADDQ $8, R12
	ADDQ $8, R13
	ADDQ R9, DI

	// Step B: Pure FMAs using Buffer B (Y14, Y15)
	VFMADD231PS Y14, Y10, Y0
	VFMADD231PS Y15, Y10, Y1
	VFMADD231PS Y14, Y11, Y2
	VFMADD231PS Y15, Y11, Y3
	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y15, Y12, Y5
	VFMADD231PS Y14, Y13, Y6
	VFMADD231PS Y15, Y13, Y7

conv_avx2_4x16_odd:
	TESTQ $1, CX
	JZ conv_avx2_4x16_store

	VMOVUPS (DI), Y8
	VMOVUPS 32(DI), Y9
	VBROADCASTSS (SI), Y10
	VBROADCASTSS (R11), Y11
	VBROADCASTSS (R12), Y12
	VBROADCASTSS (R13), Y13

	VFMADD231PS Y8, Y10, Y0
	VFMADD231PS Y9, Y10, Y1
	VFMADD231PS Y8, Y11, Y2
	VFMADD231PS Y9, Y11, Y3
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7

conv_avx2_4x16_store:
	VMOVUPS Y0, (R8)
	VMOVUPS Y1, 32(R8)
	ADDQ R10, R8
	VMOVUPS Y2, (R8)
	VMOVUPS Y3, 32(R8)
	ADDQ R10, R8
	VMOVUPS Y4, (R8)
	VMOVUPS Y5, 32(R8)
	ADDQ R10, R8
	VMOVUPS Y6, (R8)
	VMOVUPS Y7, 32(R8)
	VZEROUPPER
	RET


// ===========================================================================
// 2. ConvKernelGradAVX2_4x16Float32Asm: 4x16 Outer-Product Kernel Gradient
// ===========================================================================
// Accumulates a 4x16 tile of gradKernel[ky, kx, cIn:cIn+4, cOut:cOut+16] across
// the 3D spatial/batch domain (batchSize, numY, numX) directly in Y0..Y7.
//
// func ConvKernelGradAVX2_4x16Float32Asm(
//     xBase, vBase, outPtr unsafe.Pointer,
//     batchSize, numY, numX int,
//     xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int)
TEXT ·ConvKernelGradAVX2_4x16Float32Asm(SB), NOSPLIT, $0-104
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	MOVQ batchSize+24(FP), R10
	TESTQ R10, R10
	JLE kgrad_avx2_4x16_store

	MOVQ numY+32(FP), DX
	TESTQ DX, DX
	JLE kgrad_avx2_4x16_store

	MOVQ numX+40(FP), CX
	TESTQ CX, CX
	JLE kgrad_avx2_4x16_store

	MOVQ xBase+0(FP), R8
	MOVQ vBase+8(FP), R9
	MOVQ xStep+80(FP), R11
	MOVQ vStep+88(FP), R12

kgrad_avx2_4x16_b_loop:
	MOVQ numY+32(FP), R13                // R13 = y counter
	MOVQ R8, AX                          // AX = xYPtr
	MOVQ R9, BX                          // BX = vYPtr

kgrad_avx2_4x16_y_loop:
	MOVQ numX+40(FP), CX                 // CX = x counter
	MOVQ AX, SI                          // SI = xPtr
	MOVQ BX, DI                          // DI = vPtr

	CMPQ CX, $2
	JL kgrad_avx2_4x16_x_rem

	PCALIGN $32
kgrad_avx2_4x16_x_loop2:
	// Pixel 0 loads
	VMOVUPS (DI), Y8
	VMOVUPS 32(DI), Y9
	ADDQ R12, DI

	VBROADCASTSS 0(SI), Y10
	VBROADCASTSS 4(SI), Y11
	VBROADCASTSS 8(SI), Y12
	VBROADCASTSS 12(SI), Y13
	ADDQ R11, SI

	// Pixel 1 v loads interleaved with Pixel 0 FMAs
	VFMADD231PS Y8, Y10, Y0
	VMOVUPS (DI), Y14
	VFMADD231PS Y9, Y10, Y1
	VBROADCASTSS 0(SI), Y10

	VFMADD231PS Y8, Y11, Y2
	VMOVUPS 32(DI), Y15
	VFMADD231PS Y9, Y11, Y3
	VBROADCASTSS 4(SI), Y11

	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VBROADCASTSS 8(SI), Y12

	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS 12(SI), Y13

	ADDQ R12, DI
	ADDQ R11, SI

	// Pixel 1 FMAs
	VFMADD231PS Y14, Y10, Y0
	VFMADD231PS Y15, Y10, Y1
	VFMADD231PS Y14, Y11, Y2
	VFMADD231PS Y15, Y11, Y3
	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y15, Y12, Y5
	VFMADD231PS Y14, Y13, Y6
	VFMADD231PS Y15, Y13, Y7

	SUBQ $2, CX
	CMPQ CX, $2
	JGE kgrad_avx2_4x16_x_loop2

kgrad_avx2_4x16_x_rem:
	TESTQ CX, CX
	JZ kgrad_avx2_4x16_next_y

	VMOVUPS (DI), Y8
	VMOVUPS 32(DI), Y9
	VBROADCASTSS 0(SI), Y10
	VBROADCASTSS 4(SI), Y11
	VBROADCASTSS 8(SI), Y12
	VBROADCASTSS 12(SI), Y13

	VFMADD231PS Y8, Y10, Y0
	VFMADD231PS Y9, Y10, Y1
	VFMADD231PS Y8, Y11, Y2
	VFMADD231PS Y9, Y11, Y3
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7

kgrad_avx2_4x16_next_y:
	ADDQ xYStride+64(FP), AX
	ADDQ vYStride+72(FP), BX
	DECQ R13
	JNZ kgrad_avx2_4x16_y_loop

	ADDQ xBatchStride+48(FP), R8
	ADDQ vBatchStride+56(FP), R9
	DECQ R10
	JNZ kgrad_avx2_4x16_b_loop

kgrad_avx2_4x16_store:
	MOVQ outPtr+16(FP), DX
	MOVQ outRowStrideBytes+96(FP), R10

	VMOVUPS Y0, (DX)
	VMOVUPS Y1, 32(DX)
	ADDQ R10, DX
	VMOVUPS Y2, (DX)
	VMOVUPS Y3, 32(DX)
	ADDQ R10, DX
	VMOVUPS Y4, (DX)
	VMOVUPS Y5, 32(DX)
	ADDQ R10, DX
	VMOVUPS Y6, (DX)
	VMOVUPS Y7, 32(DX)
	VZEROUPPER
	RET


// ===========================================================================
// 3. ConvKernelGradAVX2_1x16Float32Asm: 1x16 Outer-Product Kernel Gradient
// ===========================================================================
// Accumulates a single row (1x16) of gradKernel[ky, kx, cIn, cOut:cOut+16]
// across (batchSize, numY, numX) using 4 independent accumulator pairs (Y0..Y7)
// to hide FMA latency when Cin == 1 or for remainder rows.
//
// func ConvKernelGradAVX2_1x16Float32Asm(
//     xBase, vBase, outPtr unsafe.Pointer,
//     batchSize, numY, numX int,
//     xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep int)
TEXT ·ConvKernelGradAVX2_1x16Float32Asm(SB), NOSPLIT, $0-96
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	MOVQ batchSize+24(FP), R10
	TESTQ R10, R10
	JLE kgrad_avx2_1x16_store

	MOVQ numY+32(FP), DX
	TESTQ DX, DX
	JLE kgrad_avx2_1x16_store

	MOVQ numX+40(FP), CX
	TESTQ CX, CX
	JLE kgrad_avx2_1x16_store

	MOVQ xBase+0(FP), R8
	MOVQ vBase+8(FP), R9
	MOVQ xStep+80(FP), R11
	MOVQ vStep+88(FP), R12

kgrad_avx2_1x16_b_loop:
	MOVQ numY+32(FP), R13
	MOVQ R8, AX
	MOVQ R9, BX

kgrad_avx2_1x16_y_loop:
	MOVQ numX+40(FP), CX
	MOVQ AX, SI
	MOVQ BX, DI

	CMPQ CX, $4
	JL kgrad_avx2_1x16_x_rem

	PCALIGN $32
kgrad_avx2_1x16_x_loop4:
	// Pixel 0
	VBROADCASTSS (SI), Y8
	ADDQ R11, SI
	VMOVUPS (DI), Y9
	VMOVUPS 32(DI), Y10
	ADDQ R12, DI
	VFMADD231PS Y9, Y8, Y0
	VFMADD231PS Y10, Y8, Y1

	// Pixel 1
	VBROADCASTSS (SI), Y11
	ADDQ R11, SI
	VMOVUPS (DI), Y12
	VMOVUPS 32(DI), Y13
	ADDQ R12, DI
	VFMADD231PS Y12, Y11, Y2
	VFMADD231PS Y13, Y11, Y3

	// Pixel 2
	VBROADCASTSS (SI), Y8
	ADDQ R11, SI
	VMOVUPS (DI), Y9
	VMOVUPS 32(DI), Y10
	ADDQ R12, DI
	VFMADD231PS Y9, Y8, Y4
	VFMADD231PS Y10, Y8, Y5

	// Pixel 3
	VBROADCASTSS (SI), Y11
	ADDQ R11, SI
	VMOVUPS (DI), Y12
	VMOVUPS 32(DI), Y13
	ADDQ R12, DI
	VFMADD231PS Y12, Y11, Y6
	VFMADD231PS Y13, Y11, Y7

	SUBQ $4, CX
	CMPQ CX, $4
	JGE kgrad_avx2_1x16_x_loop4

kgrad_avx2_1x16_x_rem:
	TESTQ CX, CX
	JZ kgrad_avx2_1x16_next_y

kgrad_avx2_1x16_x_loop1:
	VBROADCASTSS (SI), Y8
	ADDQ R11, SI
	VMOVUPS (DI), Y9
	VMOVUPS 32(DI), Y10
	ADDQ R12, DI
	VFMADD231PS Y9, Y8, Y0
	VFMADD231PS Y10, Y8, Y1
	DECQ CX
	JNZ kgrad_avx2_1x16_x_loop1

kgrad_avx2_1x16_next_y:
	ADDQ xYStride+64(FP), AX
	ADDQ vYStride+72(FP), BX
	DECQ R13
	JNZ kgrad_avx2_1x16_y_loop

	ADDQ xBatchStride+48(FP), R8
	ADDQ vBatchStride+56(FP), R9
	DECQ R10
	JNZ kgrad_avx2_1x16_b_loop

kgrad_avx2_1x16_store:
	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y6, Y4, Y4
	VADDPS Y7, Y5, Y5
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1

	MOVQ outPtr+16(FP), DX
	VMOVUPS Y0, (DX)
	VMOVUPS Y1, 32(DX)
	VZEROUPPER
	RET

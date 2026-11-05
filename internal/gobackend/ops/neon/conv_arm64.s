// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// ===========================================================================
// NEON & SME Instruction Encoding Macros
// ===========================================================================

// FMLA Vd.4S, Vn.4S, Vm.S[lane]
// Accumulates Vd.4S += Vn.4S * Vm.S[lane]
#define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// ZIP1 / ZIP2 32-bit and 64-bit vector interleave macros:
#define ZIP1_4S(Vd, Vn, Vm) WORD $(0x4E803800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define ZIP2_4S(Vd, Vn, Vm) WORD $(0x4E807800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define ZIP1_2D(Vd, Vn, Vm) WORD $(0x4EC03800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define ZIP2_2D(Vd, Vn, Vm) WORD $(0x4EC07800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

#define TRANSPOSE_4x4_32(r0, r1, r2, r3, c0, c1, c2, c3, t0, t1, t2, t3) \
	ZIP1_4S(t0, r0, r1); \
	ZIP2_4S(t1, r0, r1); \
	ZIP1_4S(t2, r2, r3); \
	ZIP2_4S(t3, r2, r3); \
	ZIP1_2D(c0, t0, t2); \
	ZIP2_2D(c1, t0, t2); \
	ZIP1_2D(c2, t1, t3); \
	ZIP2_2D(c3, t1, t3)

// SVE / SME encodings:
// WHILELT Pd.S, Xn, Xm
#define SVE_WHILELT_S(Pd, Xn, Xm) \
	WORD $(0x25A01400 | ((Xm)<<16) | ((Xn)<<5) | (Pd))

// LD1W {Zt.S}, Pg/Z, [Xn, #imm4, MUL VL]
#define SVE_LD1W_MUL_VL(Zt, Pg, Xn, imm4) \
	WORD $(0xA540A000 | (((imm4)&0xF)<<16) | ((Pg)<<10) | ((Xn)<<5) | (Zt))

// ST1W {Zt.S}, Pg, [Xn, #imm4, MUL VL]
#define SVE_ST1W_MUL_VL(Zt, Pg, Xn, imm4) \
	WORD $(0xE540E000 | (((imm4)&0xF)<<16) | ((Pg)<<10) | ((Xn)<<5) | (Zt))


// ===========================================================================
// 1. convNEONKernel4x16Float32Asm: Direct 4x16 NEON Convolution Microkernel
// ===========================================================================
// Multiplies 4 contiguous rows of `patchBuf` ([4, K] with row stride `patchRowStrideBytes`)
// by a 16-column panel of `rhs` ([K, 16] with row step `rhsStepBytes`) and writes the
// 4x16 tile to `outPtr` (with row stride `outRowStrideBytes`).
// Because each row of `patchBuf` is loaded as a 4-element vector along K (`V16..V19`)
// and multiplied using lane-indexed `FMLA_4S_LANE(..., lane=0..3)`, `patchBuf` does NOT
// need to be transposed or packed!
//
// func convNEONKernel4x16Float32Asm(
//     patchPtr, rhsPtr, outPtr unsafe.Pointer,
//     K, patchRowStrideBytes, rhsStepBytes, outRowStrideBytes int)
TEXT ·ConvNEONKernel4x16Float32Asm(SB), NOSPLIT, $0-56
	MOVD patchPtr+0(FP), R0
	MOVD rhsPtr+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD K+24(FP), R3
	MOVD patchRowStrideBytes+32(FP), R8
	MOVD rhsStepBytes+40(FP), R9
	MOVD outRowStrideBytes+48(FP), R10

	// Zero 16 accumulators V0..V15 (4 rows x 16 cols)
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	// Set up 4 row pointers R4, R5, R6, R7
	MOVD R0, R4
	ADD R8, R4, R5
	ADD R8, R5, R6
	ADD R8, R6, R7

	CMP $4, R3
	BLT conv_neon_4x16_rem

conv_neon_4x16_loop4:
	// Load 4 K-elements for each of the 4 rows
	VLD1.P 16(R4), [V16.S4]
	VLD1.P 16(R5), [V17.S4]
	VLD1.P 16(R6), [V18.S4]
	VLD1.P 16(R7), [V19.S4]

	// Step k+0 (lane 0)
	VLD1 (R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	ADD R9, R1, R1
	// Preload Step k+1 (lane 1) RHS into V24..V27
	VLD1 (R1), [V24.S4, V25.S4, V26.S4, V27.S4]
	ADD R9, R1, R1

	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	FMLA_4S_LANE(4, 20, 17, 0)
	FMLA_4S_LANE(5, 21, 17, 0)
	FMLA_4S_LANE(6, 22, 17, 0)
	FMLA_4S_LANE(7, 23, 17, 0)
	FMLA_4S_LANE(8, 20, 18, 0)
	FMLA_4S_LANE(9, 21, 18, 0)
	FMLA_4S_LANE(10, 22, 18, 0)
	FMLA_4S_LANE(11, 23, 18, 0)
	FMLA_4S_LANE(12, 20, 19, 0)
	FMLA_4S_LANE(13, 21, 19, 0)
	FMLA_4S_LANE(14, 22, 19, 0)
	FMLA_4S_LANE(15, 23, 19, 0)

	// Preload Step k+2 (lane 2) RHS into V20..V23
	VLD1 (R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	ADD R9, R1, R1

	// Step k+1 (lane 1)
	FMLA_4S_LANE(0, 24, 16, 1)
	FMLA_4S_LANE(1, 25, 16, 1)
	FMLA_4S_LANE(2, 26, 16, 1)
	FMLA_4S_LANE(3, 27, 16, 1)
	FMLA_4S_LANE(4, 24, 17, 1)
	FMLA_4S_LANE(5, 25, 17, 1)
	FMLA_4S_LANE(6, 26, 17, 1)
	FMLA_4S_LANE(7, 27, 17, 1)
	FMLA_4S_LANE(8, 24, 18, 1)
	FMLA_4S_LANE(9, 25, 18, 1)
	FMLA_4S_LANE(10, 26, 18, 1)
	FMLA_4S_LANE(11, 27, 18, 1)
	FMLA_4S_LANE(12, 24, 19, 1)
	FMLA_4S_LANE(13, 25, 19, 1)
	FMLA_4S_LANE(14, 26, 19, 1)
	FMLA_4S_LANE(15, 27, 19, 1)

	// Preload Step k+3 (lane 3) RHS into V24..V27
	VLD1 (R1), [V24.S4, V25.S4, V26.S4, V27.S4]
	ADD R9, R1, R1

	// Step k+2 (lane 2)
	FMLA_4S_LANE(0, 20, 16, 2)
	FMLA_4S_LANE(1, 21, 16, 2)
	FMLA_4S_LANE(2, 22, 16, 2)
	FMLA_4S_LANE(3, 23, 16, 2)
	FMLA_4S_LANE(4, 20, 17, 2)
	FMLA_4S_LANE(5, 21, 17, 2)
	FMLA_4S_LANE(6, 22, 17, 2)
	FMLA_4S_LANE(7, 23, 17, 2)
	FMLA_4S_LANE(8, 20, 18, 2)
	FMLA_4S_LANE(9, 21, 18, 2)
	FMLA_4S_LANE(10, 22, 18, 2)
	FMLA_4S_LANE(11, 23, 18, 2)
	FMLA_4S_LANE(12, 20, 19, 2)
	FMLA_4S_LANE(13, 21, 19, 2)
	FMLA_4S_LANE(14, 22, 19, 2)
	FMLA_4S_LANE(15, 23, 19, 2)

	// Step k+3 (lane 3)
	FMLA_4S_LANE(0, 24, 16, 3)
	FMLA_4S_LANE(1, 25, 16, 3)
	FMLA_4S_LANE(2, 26, 16, 3)
	FMLA_4S_LANE(3, 27, 16, 3)
	FMLA_4S_LANE(4, 24, 17, 3)
	FMLA_4S_LANE(5, 25, 17, 3)
	FMLA_4S_LANE(6, 26, 17, 3)
	FMLA_4S_LANE(7, 27, 17, 3)
	FMLA_4S_LANE(8, 24, 18, 3)
	FMLA_4S_LANE(9, 25, 18, 3)
	FMLA_4S_LANE(10, 26, 18, 3)
	FMLA_4S_LANE(11, 27, 18, 3)
	FMLA_4S_LANE(12, 24, 19, 3)
	FMLA_4S_LANE(13, 25, 19, 3)
	FMLA_4S_LANE(14, 26, 19, 3)
	FMLA_4S_LANE(15, 27, 19, 3)

	SUB $4, R3, R3
	CMP $4, R3
	BGE conv_neon_4x16_loop4

conv_neon_4x16_rem:
	CBZ R3, conv_neon_4x16_store

conv_neon_4x16_loop1:
	FMOVS.P 4(R4), F16
	FMOVS.P 4(R5), F17
	FMOVS.P 4(R6), F18
	FMOVS.P 4(R7), F19
	VLD1 (R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	ADD R9, R1, R1

	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	FMLA_4S_LANE(4, 20, 17, 0)
	FMLA_4S_LANE(5, 21, 17, 0)
	FMLA_4S_LANE(6, 22, 17, 0)
	FMLA_4S_LANE(7, 23, 17, 0)
	FMLA_4S_LANE(8, 20, 18, 0)
	FMLA_4S_LANE(9, 21, 18, 0)
	FMLA_4S_LANE(10, 22, 18, 0)
	FMLA_4S_LANE(11, 23, 18, 0)
	FMLA_4S_LANE(12, 20, 19, 0)
	FMLA_4S_LANE(13, 21, 19, 0)
	FMLA_4S_LANE(14, 22, 19, 0)
	FMLA_4S_LANE(15, 23, 19, 0)

	SUB $1, R3, R3
	CBNZ R3, conv_neon_4x16_loop1

conv_neon_4x16_store:
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R2)
	ADD R10, R2, R2
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R2)
	ADD R10, R2, R2
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R2)
	ADD R10, R2, R2
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R2)
	RET


// ===========================================================================
// 2. convKernelGradNEON4x16Float32Asm: 4x16 Outer-Product Kernel Gradient
// ===========================================================================
// Accumulates a 4x16 tile of gradKernel[ky, kx, cIn:cIn+4, cOut:cOut+16] across
// the 3D spatial/batch domain (batchSize, numY, numX) directly in V0..V15.
//
// func convKernelGradNEON4x16Float32Asm(
//     xBase, vBase, outPtr unsafe.Pointer,
//     batchSize, numY, numX int,
//     xBatchStrideBytes, vBatchStrideBytes int,
//     xYStrideBytes, vYStrideBytes int,
//     xStepBytes, vStepBytes int,
//     outRowStrideBytes int)
TEXT ·ConvKernelGradNEON4x16Float32Asm(SB), NOSPLIT, $0-104
	MOVD xBase+0(FP), R0
	MOVD vBase+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD batchSize+24(FP), R3
	MOVD numY+32(FP), R4
	MOVD numX+40(FP), R5
	MOVD xBatchStrideBytes+48(FP), R6
	MOVD vBatchStrideBytes+56(FP), R7
	MOVD xYStrideBytes+64(FP), R8
	MOVD vYStrideBytes+72(FP), R9
	MOVD xStepBytes+80(FP), R10
	MOVD vStepBytes+88(FP), R11
	MOVD outRowStrideBytes+96(FP), R12

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	CBZ R3, kgrad_neon_4x16_store
	CBZ R4, kgrad_neon_4x16_store
	CBZ R5, kgrad_neon_4x16_store

	MOVD R3, R13 // R13 = b counter

kgrad_neon_4x16_b_loop:
	MOVD R4, R14 // R14 = y counter
	MOVD R0, R15 // R15 = xYPtr
	MOVD R1, R16 // R16 = vYPtr

kgrad_neon_4x16_y_loop:
	MOVD R5, R17 // R17 = x counter
	MOVD R15, R19 // R19 = xPtr (skip R18!)
	MOVD R16, R20 // R20 = vPtr

	CMP $2, R17
	BLT kgrad_neon_4x16_x_rem

kgrad_neon_4x16_x_loop2:
	VLD1 (R19), [V16.S4]
	ADD R10, R19, R19
	VLD1 (R20), [V20.S4, V21.S4, V22.S4, V23.S4]
	ADD R11, R20, R20

	VLD1 (R19), [V17.S4]
	ADD R10, R19, R19
	VLD1 (R20), [V24.S4, V25.S4, V26.S4, V27.S4]
	ADD R11, R20, R20

	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	FMLA_4S_LANE(4, 20, 16, 1)
	FMLA_4S_LANE(5, 21, 16, 1)
	FMLA_4S_LANE(6, 22, 16, 1)
	FMLA_4S_LANE(7, 23, 16, 1)
	FMLA_4S_LANE(8, 20, 16, 2)
	FMLA_4S_LANE(9, 21, 16, 2)
	FMLA_4S_LANE(10, 22, 16, 2)
	FMLA_4S_LANE(11, 23, 16, 2)
	FMLA_4S_LANE(12, 20, 16, 3)
	FMLA_4S_LANE(13, 21, 16, 3)
	FMLA_4S_LANE(14, 22, 16, 3)
	FMLA_4S_LANE(15, 23, 16, 3)

	FMLA_4S_LANE(0, 24, 17, 0)
	FMLA_4S_LANE(1, 25, 17, 0)
	FMLA_4S_LANE(2, 26, 17, 0)
	FMLA_4S_LANE(3, 27, 17, 0)
	FMLA_4S_LANE(4, 24, 17, 1)
	FMLA_4S_LANE(5, 25, 17, 1)
	FMLA_4S_LANE(6, 26, 17, 1)
	FMLA_4S_LANE(7, 27, 17, 1)
	FMLA_4S_LANE(8, 24, 17, 2)
	FMLA_4S_LANE(9, 25, 17, 2)
	FMLA_4S_LANE(10, 26, 17, 2)
	FMLA_4S_LANE(11, 27, 17, 2)
	FMLA_4S_LANE(12, 24, 17, 3)
	FMLA_4S_LANE(13, 25, 17, 3)
	FMLA_4S_LANE(14, 26, 17, 3)
	FMLA_4S_LANE(15, 27, 17, 3)

	SUB $2, R17, R17
	CMP $2, R17
	BGE kgrad_neon_4x16_x_loop2

kgrad_neon_4x16_x_rem:
	CBZ R17, kgrad_neon_4x16_next_y

	VLD1 (R19), [V16.S4]
	VLD1 (R20), [V20.S4, V21.S4, V22.S4, V23.S4]

	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	FMLA_4S_LANE(4, 20, 16, 1)
	FMLA_4S_LANE(5, 21, 16, 1)
	FMLA_4S_LANE(6, 22, 16, 1)
	FMLA_4S_LANE(7, 23, 16, 1)
	FMLA_4S_LANE(8, 20, 16, 2)
	FMLA_4S_LANE(9, 21, 16, 2)
	FMLA_4S_LANE(10, 22, 16, 2)
	FMLA_4S_LANE(11, 23, 16, 2)
	FMLA_4S_LANE(12, 20, 16, 3)
	FMLA_4S_LANE(13, 21, 16, 3)
	FMLA_4S_LANE(14, 22, 16, 3)
	FMLA_4S_LANE(15, 23, 16, 3)

kgrad_neon_4x16_next_y:
	ADD R8, R15, R15
	ADD R9, R16, R16
	SUB $1, R14, R14
	CBNZ R14, kgrad_neon_4x16_y_loop

	ADD R6, R0, R0
	ADD R7, R1, R1
	SUB $1, R13, R13
	CBNZ R13, kgrad_neon_4x16_b_loop

kgrad_neon_4x16_store:
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R2)
	ADD R12, R2, R2
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R2)
	ADD R12, R2, R2
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R2)
	ADD R12, R2, R2
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R2)
	RET


// ===========================================================================
// 3. convKernelGradNEON1x16Float32Asm: 1x16 Outer-Product Kernel Gradient
// ===========================================================================
// Accumulates a single row (1x16) of gradKernel[ky, kx, cIn, cOut:cOut+16]
// across (batchSize, numY, numX). Used when Cin == 1 (e.g. MNIST Conv 1)
// or for remainder rows when Cin % 4 != 0.
//
// func convKernelGradNEON1x16Float32Asm(
//     xBase, vBase, outPtr unsafe.Pointer,
//     batchSize, numY, numX int,
//     xBatchStrideBytes, vBatchStrideBytes int,
//     xYStrideBytes, vYStrideBytes int,
//     xStepBytes, vStepBytes int)
TEXT ·ConvKernelGradNEON1x16Float32Asm(SB), NOSPLIT, $0-96
	MOVD xBase+0(FP), R0
	MOVD vBase+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD batchSize+24(FP), R3
	MOVD numY+32(FP), R4
	MOVD numX+40(FP), R5
	MOVD xBatchStrideBytes+48(FP), R6
	MOVD vBatchStrideBytes+56(FP), R7
	MOVD xYStrideBytes+64(FP), R8
	MOVD vYStrideBytes+72(FP), R9
	MOVD xStepBytes+80(FP), R10
	MOVD vStepBytes+88(FP), R11

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16

	CBZ R3, kgrad_neon_1x16_store
	CBZ R4, kgrad_neon_1x16_store
	CBZ R5, kgrad_neon_1x16_store

	MOVD R3, R13

kgrad_neon_1x16_b_loop:
	MOVD R4, R14
	MOVD R0, R15
	MOVD R1, R16

kgrad_neon_1x16_y_loop:
	MOVD R5, R17
	MOVD R15, R19
	MOVD R16, R20

	CMP $2, R17
	BLT kgrad_neon_1x16_x_rem

kgrad_neon_1x16_x_loop2:
	FMOVS (R19), F16
	ADD R10, R19, R19
	VLD1 (R20), [V20.S4, V21.S4, V22.S4, V23.S4]
	ADD R11, R20, R20

	FMOVS (R19), F17
	ADD R10, R19, R19
	VLD1 (R20), [V24.S4, V25.S4, V26.S4, V27.S4]
	ADD R11, R20, R20

	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)

	FMLA_4S_LANE(4, 24, 17, 0)
	FMLA_4S_LANE(5, 25, 17, 0)
	FMLA_4S_LANE(6, 26, 17, 0)
	FMLA_4S_LANE(7, 27, 17, 0)

	SUB $2, R17, R17
	CMP $2, R17
	BGE kgrad_neon_1x16_x_loop2

kgrad_neon_1x16_x_rem:
	CBZ R17, kgrad_neon_1x16_next_y

	FMOVS (R19), F16
	VLD1 (R20), [V20.S4, V21.S4, V22.S4, V23.S4]
	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)

kgrad_neon_1x16_next_y:
	ADD R8, R15, R15
	ADD R9, R16, R16
	SUB $1, R14, R14
	CBNZ R14, kgrad_neon_1x16_y_loop

	ADD R6, R0, R0
	ADD R7, R1, R1
	SUB $1, R13, R13
	CBNZ R13, kgrad_neon_1x16_b_loop

kgrad_neon_1x16_store:
	VFADD V4.S4, V0.S4, V0.S4
	VFADD V5.S4, V1.S4, V1.S4
	VFADD V6.S4, V2.S4, V2.S4
	VFADD V7.S4, V3.S4, V3.S4
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R2)
	RET


// ===========================================================================
// 4. convPackLHS8StridedFloat32Asm: Pack 8 rows of patchBuf into SME panel
// ===========================================================================
// func ConvPackLHS8StridedFloat32Asm(
//     srcPtr, dstPtr unsafe.Pointer,
//     K, srcRowStrideBytes, panelColStrideBytes int)
TEXT ·ConvPackLHS8StridedFloat32Asm(SB), NOSPLIT, $0-40
	MOVD srcPtr+0(FP), R0
	MOVD dstPtr+8(FP), R1
	MOVD K+16(FP), R10
	MOVD srcRowStrideBytes+24(FP), R2
	MOVD panelColStrideBytes+32(FP), R3

	CBZ R10, conv_lhs8s_done

	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	ADD R2, R7, R11
	ADD R2, R11, R12
	ADD R2, R12, R13
	ADD R2, R13, R14

conv_lhs8s_col4:
	CMP $4, R10
	BLT conv_lhs8s_col1

	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R5), [V1.S4]
	VLD1.P 16(R6), [V2.S4]
	VLD1.P 16(R7), [V3.S4]
	VLD1.P 16(R11), [V4.S4]
	VLD1.P 16(R12), [V5.S4]
	VLD1.P 16(R13), [V6.S4]
	VLD1.P 16(R14), [V7.S4]

	TRANSPOSE_4x4_32(0, 1, 2, 3, 16, 17, 18, 19, 24, 25, 26, 27)
	TRANSPOSE_4x4_32(4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26, 27)

	VST1 [V16.S4], (R1)
	ADD $16, R1, R15
	VST1 [V20.S4], (R15)
	ADD R3, R1, R1

	VST1 [V17.S4], (R1)
	ADD $16, R1, R15
	VST1 [V21.S4], (R15)
	ADD R3, R1, R1

	VST1 [V18.S4], (R1)
	ADD $16, R1, R15
	VST1 [V22.S4], (R15)
	ADD R3, R1, R1

	VST1 [V19.S4], (R1)
	ADD $16, R1, R15
	VST1 [V23.S4], (R15)
	ADD R3, R1, R1

	SUB $4, R10, R10
	B conv_lhs8s_col4

conv_lhs8s_col1:
	CBZ R10, conv_lhs8s_done
	FMOVS.P 4(R4), F0
	FMOVS.P 4(R5), F1
	FMOVS.P 4(R6), F2
	FMOVS.P 4(R7), F3
	FMOVS.P 4(R11), F4
	FMOVS.P 4(R12), F5
	FMOVS.P 4(R13), F6
	FMOVS.P 4(R14), F7
	FMOVS F0, 0(R1)
	FMOVS F1, 4(R1)
	FMOVS F2, 8(R1)
	FMOVS F3, 12(R1)
	FMOVS F4, 16(R1)
	FMOVS F5, 20(R1)
	FMOVS F6, 24(R1)
	FMOVS F7, 28(R1)
	ADD R3, R1, R1
	SUB $1, R10, R10
	B conv_lhs8s_col1

conv_lhs8s_done:
	RET


// ===========================================================================
// 5. convSMEKernel32x32Float32Asm: Single 32x32 Tile SME FMOPA Microkernel
// ===========================================================================
// Computes a single 32x32 output tile (validRows <= 32, validCols <= 32) from
// packedLHS ([K, 32]) and packedRHS ([K, 32]) and stores directly to outPtr.
//
// func ConvSMEKernel32x32Float32Asm(
//     packedLHS, packedRHS, outPtr unsafe.Pointer,
//     K, outRowStrideBytes, validRows, validCols int)
TEXT ·ConvSMEKernel32x32Float32Asm(SB), NOSPLIT, $0-56
	MOVD packedLHS+0(FP), R0
	MOVD packedRHS+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD K+24(FP), R4
	MOVD outRowStrideBytes+32(FP), R3
	MOVD validRows+40(FP), R5
	MOVD validCols+48(FP), R6

	WORD $0xD503477F // SMSTART

conv_sme_32x32_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1 (signal canary)
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	MOVD R0, R8
	MOVD R1, R9
	MOVD R4, R11

	CMP $4, R11
	BLT conv_sme_32x32_rem

conv_sme_32x32_loop4:
	// Step 0
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA541A101 // LD1W {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA540A122 // LD1W {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA541A123 // LD1W {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

	// Step 1
	WORD $0xA542A104 // LD1W {Z4.S}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA543A105 // LD1W {Z5.S}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA542A126 // LD1W {Z6.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA543A127 // LD1W {Z7.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x80860080 // FMOPA ZA0.S, P0/M, P0/M, Z4.S, Z6.S
	WORD $0x80870081 // FMOPA ZA1.S, P0/M, P0/M, Z4.S, Z7.S
	WORD $0x808600A2 // FMOPA ZA2.S, P0/M, P0/M, Z5.S, Z6.S
	WORD $0x808700A3 // FMOPA ZA3.S, P0/M, P0/M, Z5.S, Z7.S

	// Step 2
	WORD $0xA544A100 // LD1W {Z0.S}, P0/Z, [X8, #4, MUL VL]
	WORD $0xA545A101 // LD1W {Z1.S}, P0/Z, [X8, #5, MUL VL]
	WORD $0xA544A122 // LD1W {Z2.S}, P0/Z, [X9, #4, MUL VL]
	WORD $0xA545A123 // LD1W {Z3.S}, P0/Z, [X9, #5, MUL VL]
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

	// Step 3
	WORD $0xA546A104 // LD1W {Z4.S}, P0/Z, [X8, #6, MUL VL]
	WORD $0xA547A105 // LD1W {Z5.S}, P0/Z, [X8, #7, MUL VL]
	WORD $0xA546A126 // LD1W {Z6.S}, P0/Z, [X9, #6, MUL VL]
	WORD $0xA547A127 // LD1W {Z7.S}, P0/Z, [X9, #7, MUL VL]
	WORD $0x80860080 // FMOPA ZA0.S, P0/M, P0/M, Z4.S, Z6.S
	WORD $0x80870081 // FMOPA ZA1.S, P0/M, P0/M, Z4.S, Z7.S
	WORD $0x808600A2 // FMOPA ZA2.S, P0/M, P0/M, Z5.S, Z6.S
	WORD $0x808700A3 // FMOPA ZA3.S, P0/M, P0/M, Z5.S, Z7.S

	ADD $512, R8, R8
	ADD $512, R9, R9
	SUB $4, R11, R11
	CMP $4, R11
	BGE conv_sme_32x32_loop4

conv_sme_32x32_rem:
	CBZ R11, conv_sme_32x32_k_done

conv_sme_32x32_loop1:
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA541A101 // LD1W {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA540A122 // LD1W {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA541A123 // LD1W {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S
	ADD $128, R8, R8
	ADD $128, R9, R9
	SUB $1, R11, R11
	CBNZ R11, conv_sme_32x32_loop1

conv_sme_32x32_k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, conv_sme_32x32_retry

conv_sme_32x32_store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	MOVD $0, R14
	SVE_WHILELT_S(0, 14, 6) // WHILELT P0.S, X14, X6 (cols 0..15 < validCols)
	MOVD $16, R14
	SVE_WHILELT_S(1, 14, 6) // WHILELT P1.S, X14, X6 (cols 16..31 < validCols)

	MOVD R2, R16
	LSL $4, R3, R15
	ADD R2, R15, R17 // R17 = outPtr + 16 * outRowStrideBytes

	MOVD $0, R12 // W12 = slice index 0..15
	MOVD $16, R15 // R15 = bottom row index (12 + 16)

conv_sme_32x32_store_loop:
	CMP R5, R12
	BGE conv_sme_32x32_store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xC0820481 // MOVA Z1.S, P1/M, ZA1H.S[W12, 0]
	WORD $0xE540E200 // ST1W {Z0.S}, P0, [X16, #0, MUL VL]
	WORD $0xE541E601 // ST1W {Z1.S}, P1, [X16, #1, MUL VL]

	CMP R5, R15
	BGE conv_sme_32x32_skip_bot

	WORD $0xC0820102 // MOVA Z2.S, P0/M, ZA2H.S[W12, 0]
	WORD $0xC0820583 // MOVA Z3.S, P1/M, ZA3H.S[W12, 0]
	WORD $0xE540E222 // ST1W {Z2.S}, P0, [X17, #0, MUL VL]
	WORD $0xE541E623 // ST1W {Z3.S}, P1, [X17, #1, MUL VL]

conv_sme_32x32_skip_bot:
	ADD $1, R12, R12
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $16, R12
	BLT conv_sme_32x32_store_loop

conv_sme_32x32_store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, conv_sme_32x32_store_retry

	WORD $0xD503467F // SMSTOP
	RET


// ===========================================================================
// 6. convSMEKernel16x16Float32Asm: Single 16x16 Tile SME FMOPA Microkernel
// ===========================================================================
// func ConvSMEKernel16x16Float32Asm(
//     packedLHS, packedRHS, outPtr unsafe.Pointer,
//     K, outRowStrideBytes, validRows, validCols int)
TEXT ·ConvSMEKernel16x16Float32Asm(SB), NOSPLIT, $0-56
	MOVD packedLHS+0(FP), R0
	MOVD packedRHS+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD K+24(FP), R4
	MOVD outRowStrideBytes+32(FP), R3
	MOVD validRows+40(FP), R5
	MOVD validCols+48(FP), R6

	WORD $0xD503477F // SMSTART

conv_sme_16x16_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	MOVD R0, R8
	MOVD R1, R9
	MOVD R4, R11

	CMP $4, R11
	BLT conv_sme_16x16_rem

conv_sme_16x16_loop4:
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA540A121 // LD1W {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S

	WORD $0xA541A102 // LD1W {Z2.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA541A123 // LD1W {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80830040 // FMOPA ZA0.S, P0/M, P0/M, Z2.S, Z3.S

	WORD $0xA542A100 // LD1W {Z0.S}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA542A121 // LD1W {Z1.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S

	WORD $0xA543A102 // LD1W {Z2.S}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA543A123 // LD1W {Z3.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x80830040 // FMOPA ZA0.S, P0/M, P0/M, Z2.S, Z3.S

	ADD $256, R8, R8
	ADD $256, R9, R9
	SUB $4, R11, R11
	CMP $4, R11
	BGE conv_sme_16x16_loop4

conv_sme_16x16_rem:
	CBZ R11, conv_sme_16x16_k_done

conv_sme_16x16_loop1:
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA540A121 // LD1W {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
	ADD $64, R8, R8
	ADD $64, R9, R9
	SUB $1, R11, R11
	CBNZ R11, conv_sme_16x16_loop1

conv_sme_16x16_k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, conv_sme_16x16_retry

conv_sme_16x16_store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	MOVD $0, R14
	SVE_WHILELT_S(0, 14, 6) // WHILELT P0.S, X14, X6

	MOVD R2, R10
	MOVD $0, R12

conv_sme_16x16_store_loop:
	CMP R5, R12
	BGE conv_sme_16x16_store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xE540E140 // ST1W {Z0.S}, P0, [X10, #0, MUL VL]

	ADD $1, R12, R12
	ADD R3, R10, R10
	CMP $16, R12
	BLT conv_sme_16x16_store_loop

conv_sme_16x16_store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, conv_sme_16x16_store_retry

	WORD $0xD503467F // SMSTOP
	RET


// ===========================================================================
// 7. convKernelGradSME32x32Float32Asm: 32x32 SME Outer-Product Kernel Gradient
// ===========================================================================
// Accumulates a 32x32 tile (validRows <= 32, validCols <= 32) of
// gradKernel[ky, kx, cIn:cIn+validRows, cOut:cOut+validCols] directly in ZA0..ZA3
// across the 3D spatial/batch domain (batchSize, numY, numX) with ZERO packing
// and ZERO intermediate buffers!
//
// func ConvKernelGradSME32x32Float32Asm(
//     xBase, vBase, outPtr unsafe.Pointer,
//     batchSize, numY, numX int,
//     xBatchStrideBytes, vBatchStrideBytes int,
//     xYStrideBytes, vYStrideBytes int,
//     xStepBytes, vStepBytes int,
//     outRowStrideBytes, validRows, validCols, accumulate int)
TEXT ·ConvKernelGradSME32x32Float32Asm(SB), NOSPLIT, $0-128
	MOVD xBase+0(FP), R0
	MOVD vBase+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD batchSize+24(FP), R3
	MOVD numY+32(FP), R4
	MOVD numX+40(FP), R5
	MOVD xBatchStrideBytes+48(FP), R6
	MOVD vBatchStrideBytes+56(FP), R7
	MOVD xYStrideBytes+64(FP), R8
	MOVD vYStrideBytes+72(FP), R9
	MOVD xStepBytes+80(FP), R10
	MOVD outRowStrideBytes+96(FP), R25
	MOVD validRows+104(FP), R21
	MOVD validCols+112(FP), R22
	MOVD accumulate+120(FP), R26

	WORD $0xD503477F // SMSTART

kgrad_sme_32x32_retry:
	MOVD vStepBytes+88(FP), R11
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1 (signal canary)
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R26, kgrad_sme_32x32_init_za_done

	// accumulate != 0: load existing out tile into ZA0..ZA3
	MOVD $0, R14
	SVE_WHILELT_S(1, 14, 22) // WHILELT P1.S, X14, X22 (cols 0..15 < validCols)
	MOVD $16, R14
	SVE_WHILELT_S(2, 14, 22) // WHILELT P2.S, X14, X22 (cols 16..31 < validCols)

	MOVD R2, R16
	LSL $4, R25, R15
	ADD R2, R15, R17 // R17 = outPtr + 16 * outRowStrideBytes

	MOVD $0, R12  // W12 = slice index 0..15
	MOVD $16, R15 // R15 = bottom row index (12 + 16)

kgrad_sme_32x32_init_za_loop:
	CMP R21, R12
	BGE kgrad_sme_32x32_init_za_done

	WORD $0xA540A600 // LD1W {Z0.S}, P1/Z, [X16, #0, MUL VL]
	WORD $0xA541AA01 // LD1W {Z1.S}, P2/Z, [X16, #1, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	WORD $0xC0800824 // MOVA ZA1H.S[W12, 0], P2/M, Z1.S

	CMP R21, R15
	BGE kgrad_sme_32x32_init_za_skip_bot

	WORD $0xA540A622 // LD1W {Z2.S}, P1/Z, [X17, #0, MUL VL]
	WORD $0xA541AA23 // LD1W {Z3.S}, P2/Z, [X17, #1, MUL VL]
	WORD $0xC0800448 // MOVA ZA2H.S[W12, 0], P1/M, Z2.S
	WORD $0xC080086C // MOVA ZA3H.S[W12, 0], P2/M, Z3.S

kgrad_sme_32x32_init_za_skip_bot:
	ADD $1, R12, R12
	ADD $1, R15, R15
	ADD R25, R16, R16
	ADD R25, R17, R17
	CMP $16, R12
	BLT kgrad_sme_32x32_init_za_loop

kgrad_sme_32x32_init_za_done:
	// Set up predicates P1, P2 for validRows (Cin) and P3, P4 for validCols (Cout)
	MOVD $0, R14
	SVE_WHILELT_S(1, 14, 21) // WHILELT P1.S, X14, X21 (rows 0..15 < validRows)
	SVE_WHILELT_S(3, 14, 22) // WHILELT P3.S, X14, X22 (cols 0..15 < validCols)
	MOVD $16, R14
	SVE_WHILELT_S(2, 14, 21) // WHILELT P2.S, X14, X21 (rows 16..31 < validRows)
	SVE_WHILELT_S(4, 14, 22) // WHILELT P4.S, X14, X22 (cols 16..31 < validCols)

	CBZ R3, kgrad_sme_32x32_k_done
	CBZ R4, kgrad_sme_32x32_k_done
	CBZ R5, kgrad_sme_32x32_k_done

	MOVD R3, R13   // R13 = b counter
	MOVD R0, R23   // R23 = xBatchPtr
	MOVD R1, R24   // R24 = vBatchPtr

kgrad_sme_32x32_b_loop:
	MOVD R4, R14   // R14 = y counter
	MOVD R23, R15  // R15 = xYPtr
	MOVD R24, R16  // R16 = vYPtr

kgrad_sme_32x32_y_loop:
	MOVD R5, R17   // R17 = x counter
	MOVD R15, R19  // R19 = xPtr (skip R18!)
	MOVD R16, R20  // R20 = vPtr

	CMP $2, R17
	BLT kgrad_sme_32x32_x_rem

kgrad_sme_32x32_x_loop2:
	// Pixel 0
	SVE_LD1W_MUL_VL(0, 1, 19, 0) // LD1W {Z0.S}, P1/Z, [X19, #0, MUL VL]
	SVE_LD1W_MUL_VL(1, 2, 19, 1) // LD1W {Z1.S}, P2/Z, [X19, #1, MUL VL]
	ADD R10, R19, R19
	SVE_LD1W_MUL_VL(2, 3, 20, 0) // LD1W {Z2.S}, P3/Z, [X20, #0, MUL VL]
	SVE_LD1W_MUL_VL(3, 4, 20, 1) // LD1W {Z3.S}, P4/Z, [X20, #1, MUL VL]
	ADD R11, R20, R20

	// Pixel 1
	SVE_LD1W_MUL_VL(4, 1, 19, 0) // LD1W {Z4.S}, P1/Z, [X19, #0, MUL VL]
	SVE_LD1W_MUL_VL(5, 2, 19, 1) // LD1W {Z5.S}, P2/Z, [X19, #1, MUL VL]
	ADD R10, R19, R19
	SVE_LD1W_MUL_VL(6, 3, 20, 0) // LD1W {Z6.S}, P3/Z, [X20, #0, MUL VL]
	SVE_LD1W_MUL_VL(7, 4, 20, 1) // LD1W {Z7.S}, P4/Z, [X20, #1, MUL VL]
	ADD R11, R20, R20

	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

	WORD $0x80860080 // FMOPA ZA0.S, P0/M, P0/M, Z4.S, Z6.S
	WORD $0x80870081 // FMOPA ZA1.S, P0/M, P0/M, Z4.S, Z7.S
	WORD $0x808600A2 // FMOPA ZA2.S, P0/M, P0/M, Z5.S, Z6.S
	WORD $0x808700A3 // FMOPA ZA3.S, P0/M, P0/M, Z5.S, Z7.S

	SUB $2, R17, R17
	CMP $2, R17
	BGE kgrad_sme_32x32_x_loop2

kgrad_sme_32x32_x_rem:
	CBZ R17, kgrad_sme_32x32_next_y

	SVE_LD1W_MUL_VL(0, 1, 19, 0)
	SVE_LD1W_MUL_VL(1, 2, 19, 1)
	SVE_LD1W_MUL_VL(2, 3, 20, 0)
	SVE_LD1W_MUL_VL(3, 4, 20, 1)
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

kgrad_sme_32x32_next_y:
	ADD R8, R15, R15
	ADD R9, R16, R16
	SUB $1, R14, R14
	CBNZ R14, kgrad_sme_32x32_y_loop

	ADD R6, R23, R23
	ADD R7, R24, R24
	SUB $1, R13, R13
	CBNZ R13, kgrad_sme_32x32_b_loop

kgrad_sme_32x32_k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, kgrad_sme_32x32_retry

kgrad_sme_32x32_store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	MOVD $0, R14
	SVE_WHILELT_S(0, 14, 22) // WHILELT P0.S, X14, X22 (cols 0..15 < validCols)
	MOVD $16, R14
	SVE_WHILELT_S(1, 14, 22) // WHILELT P1.S, X14, X22 (cols 16..31 < validCols)

	MOVD R2, R16
	LSL $4, R25, R15
	ADD R2, R15, R17 // R17 = outPtr + 16 * outRowStrideBytes

	MOVD $0, R12  // W12 = slice index 0..15
	MOVD $16, R15 // R15 = bottom row index (12 + 16)

kgrad_sme_32x32_store_loop:
	CMP R21, R12
	BGE kgrad_sme_32x32_store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xC0820481 // MOVA Z1.S, P1/M, ZA1H.S[W12, 0]
	WORD $0xE540E200 // ST1W {Z0.S}, P0, [X16, #0, MUL VL]
	WORD $0xE541E601 // ST1W {Z1.S}, P1, [X16, #1, MUL VL]

	CMP R21, R15
	BGE kgrad_sme_32x32_skip_bot

	WORD $0xC0820102 // MOVA Z2.S, P0/M, ZA2H.S[W12, 0]
	WORD $0xC0820583 // MOVA Z3.S, P1/M, ZA3H.S[W12, 0]
	WORD $0xE540E222 // ST1W {Z2.S}, P0, [X17, #0, MUL VL]
	WORD $0xE541E623 // ST1W {Z3.S}, P1, [X17, #1, MUL VL]

kgrad_sme_32x32_skip_bot:
	ADD $1, R12, R12
	ADD $1, R15, R15
	ADD R25, R16, R16
	ADD R25, R17, R17
	CMP $16, R12
	BLT kgrad_sme_32x32_store_loop

kgrad_sme_32x32_store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, kgrad_sme_32x32_store_retry

	WORD $0xD503467F // SMSTOP
	RET

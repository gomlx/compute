// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// ============================================================================
// AVX2 Binary Mid / Leading / Contiguous Broadcast Kernels (Float32)
// ============================================================================
// Signature: func(vec, bcast, out unsafe.Pointer, A, B, C int)
//   vec:   [A, B, C] float32
//   bcast: [A, 1, C] float32
//   out:   [A, B, C] float32
//
// For RHS broadcast (isLHS == false): out[(a*B+b)*C + c] = vec[(a*B+b)*C + c] OP bcast[a*C + c]
// For LHS broadcast (isLHS == true):  out[(a*B+b)*C + c] = bcast[a*C + c] OP vec[(a*B+b)*C + c]
//
// Also covers:
//   - BroadcastNone ([N] op [N]):        A = 1, B = 1, C = N
//   - BroadcastLeadingRHS ([A,B] op [B]): A = 1, B = A_orig, C = B_orig
//   - BroadcastLeadingLHS ([B] op [A,B]): A = 1, B = A_orig, C = B_orig

#define BINARY_MID_RHS_BODY_F32(VOP, SOP) \
	MOVQ vec+0(FP), SI \
	MOVQ bcast+8(FP), DX \
	MOVQ out+16(FP), DI \
	MOVQ A+24(FP), R8 \
	MOVQ B+32(FP), R9 \
	MOVQ C+40(FP), R10 \
	TESTQ R8, R8 \
	JLE done \
	TESTQ R9, R9 \
	JLE done \
	TESTQ R10, R10 \
	JLE done \
	CMPQ R10, $64 \
	JNE a_loop \
a_loop64: \
	VMOVUPS (DX), Y8 \
	VMOVUPS 32(DX), Y9 \
	VMOVUPS 64(DX), Y10 \
	VMOVUPS 96(DX), Y11 \
	VMOVUPS 128(DX), Y12 \
	VMOVUPS 160(DX), Y13 \
	VMOVUPS 192(DX), Y14 \
	VMOVUPS 224(DX), Y15 \
	MOVQ R9, R11 \
b_loop64: \
	VMOVUPS (SI), Y0 \
	VMOVUPS 32(SI), Y1 \
	VMOVUPS 64(SI), Y2 \
	VMOVUPS 96(SI), Y3 \
	VOP Y8, Y0, Y0 \
	VOP Y9, Y1, Y1 \
	VOP Y10, Y2, Y2 \
	VOP Y11, Y3, Y3 \
	VMOVUPS Y0, (DI) \
	VMOVUPS Y1, 32(DI) \
	VMOVUPS Y2, 64(DI) \
	VMOVUPS Y3, 96(DI) \
	VMOVUPS 128(SI), Y4 \
	VMOVUPS 160(SI), Y5 \
	VMOVUPS 192(SI), Y6 \
	VMOVUPS 224(SI), Y7 \
	VOP Y12, Y4, Y4 \
	VOP Y13, Y5, Y5 \
	VOP Y14, Y6, Y6 \
	VOP Y15, Y7, Y7 \
	VMOVUPS Y4, 128(DI) \
	VMOVUPS Y5, 160(DI) \
	VMOVUPS Y6, 192(DI) \
	VMOVUPS Y7, 224(DI) \
	ADDQ $256, SI \
	ADDQ $256, DI \
	DECQ R11 \
	JNZ b_loop64 \
	ADDQ $256, DX \
	DECQ R8 \
	JNZ a_loop64 \
	VZEROUPPER \
	RET \
a_loop: \
	MOVQ R9, R11 \
b_loop: \
	MOVQ DX, BX \
	MOVQ R10, CX \
	SHRQ $5, CX \
	JZ tail8 \
c32_loop: \
	VMOVUPS (SI), Y0 \
	VMOVUPS 32(SI), Y1 \
	VMOVUPS 64(SI), Y2 \
	VMOVUPS 96(SI), Y3 \
	VMOVUPS (BX), Y4 \
	VMOVUPS 32(BX), Y5 \
	VMOVUPS 64(BX), Y6 \
	VMOVUPS 96(BX), Y7 \
	VOP Y4, Y0, Y0 \
	VOP Y5, Y1, Y1 \
	VOP Y6, Y2, Y2 \
	VOP Y7, Y3, Y3 \
	VMOVUPS Y0, (DI) \
	VMOVUPS Y1, 32(DI) \
	VMOVUPS Y2, 64(DI) \
	VMOVUPS Y3, 96(DI) \
	ADDQ $128, SI \
	ADDQ $128, BX \
	ADDQ $128, DI \
	DECQ CX \
	JNZ c32_loop \
tail8: \
	MOVQ R10, CX \
	SHRQ $3, CX \
	ANDQ $3, CX \
	JZ tail1 \
c8_loop: \
	VMOVUPS (SI), Y0 \
	VMOVUPS (BX), Y4 \
	VOP Y4, Y0, Y0 \
	VMOVUPS Y0, (DI) \
	ADDQ $32, SI \
	ADDQ $32, BX \
	ADDQ $32, DI \
	DECQ CX \
	JNZ c8_loop \
tail1: \
	MOVQ R10, CX \
	ANDQ $7, CX \
	JZ next_b \
c1_loop: \
	VMOVSS (SI), X0 \
	VMOVSS (BX), X4 \
	SOP X4, X0, X0 \
	VMOVSS X0, (DI) \
	ADDQ $4, SI \
	ADDQ $4, BX \
	ADDQ $4, DI \
	DECQ CX \
	JNZ c1_loop \
next_b: \
	DECQ R11 \
	JNZ b_loop \
	MOVQ BX, DX \
	DECQ R8 \
	JNZ a_loop \
done: \
	VZEROUPPER \
	RET

#define BINARY_MID_LHS_BODY_F32(VOP, SOP) \
	MOVQ vec+0(FP), SI \
	MOVQ bcast+8(FP), DX \
	MOVQ out+16(FP), DI \
	MOVQ A+24(FP), R8 \
	MOVQ B+32(FP), R9 \
	MOVQ C+40(FP), R10 \
	TESTQ R8, R8 \
	JLE done \
	TESTQ R9, R9 \
	JLE done \
	TESTQ R10, R10 \
	JLE done \
	CMPQ R10, $64 \
	JNE a_loop \
a_loop64: \
	VMOVUPS (DX), Y8 \
	VMOVUPS 32(DX), Y9 \
	VMOVUPS 64(DX), Y10 \
	VMOVUPS 96(DX), Y11 \
	VMOVUPS 128(DX), Y12 \
	VMOVUPS 160(DX), Y13 \
	VMOVUPS 192(DX), Y14 \
	VMOVUPS 224(DX), Y15 \
	MOVQ R9, R11 \
b_loop64: \
	VMOVUPS (SI), Y0 \
	VMOVUPS 32(SI), Y1 \
	VMOVUPS 64(SI), Y2 \
	VMOVUPS 96(SI), Y3 \
	VOP Y0, Y8, Y0 \
	VOP Y1, Y9, Y1 \
	VOP Y2, Y10, Y2 \
	VOP Y3, Y11, Y3 \
	VMOVUPS Y0, (DI) \
	VMOVUPS Y1, 32(DI) \
	VMOVUPS Y2, 64(DI) \
	VMOVUPS Y3, 96(DI) \
	VMOVUPS 128(SI), Y4 \
	VMOVUPS 160(SI), Y5 \
	VMOVUPS 192(SI), Y6 \
	VMOVUPS 224(SI), Y7 \
	VOP Y4, Y12, Y4 \
	VOP Y5, Y13, Y5 \
	VOP Y6, Y14, Y6 \
	VOP Y7, Y15, Y7 \
	VMOVUPS Y4, 128(DI) \
	VMOVUPS Y5, 160(DI) \
	VMOVUPS Y6, 192(DI) \
	VMOVUPS Y7, 224(DI) \
	ADDQ $256, SI \
	ADDQ $256, DI \
	DECQ R11 \
	JNZ b_loop64 \
	ADDQ $256, DX \
	DECQ R8 \
	JNZ a_loop64 \
	VZEROUPPER \
	RET \
a_loop: \
	MOVQ R9, R11 \
b_loop: \
	MOVQ DX, BX \
	MOVQ R10, CX \
	SHRQ $5, CX \
	JZ tail8 \
c32_loop: \
	VMOVUPS (SI), Y0 \
	VMOVUPS 32(SI), Y1 \
	VMOVUPS 64(SI), Y2 \
	VMOVUPS 96(SI), Y3 \
	VMOVUPS (BX), Y4 \
	VMOVUPS 32(BX), Y5 \
	VMOVUPS 64(BX), Y6 \
	VMOVUPS 96(BX), Y7 \
	VOP Y0, Y4, Y0 \
	VOP Y1, Y5, Y1 \
	VOP Y2, Y6, Y2 \
	VOP Y3, Y7, Y3 \
	VMOVUPS Y0, (DI) \
	VMOVUPS Y1, 32(DI) \
	VMOVUPS Y2, 64(DI) \
	VMOVUPS Y3, 96(DI) \
	ADDQ $128, SI \
	ADDQ $128, BX \
	ADDQ $128, DI \
	DECQ CX \
	JNZ c32_loop \
tail8: \
	MOVQ R10, CX \
	SHRQ $3, CX \
	ANDQ $3, CX \
	JZ tail1 \
c8_loop: \
	VMOVUPS (SI), Y0 \
	VMOVUPS (BX), Y4 \
	VOP Y0, Y4, Y0 \
	VMOVUPS Y0, (DI) \
	ADDQ $32, SI \
	ADDQ $32, BX \
	ADDQ $32, DI \
	DECQ CX \
	JNZ c8_loop \
tail1: \
	MOVQ R10, CX \
	ANDQ $7, CX \
	JZ next_b \
c1_loop: \
	VMOVSS (SI), X0 \
	VMOVSS (BX), X4 \
	SOP X0, X4, X0 \
	VMOVSS X0, (DI) \
	ADDQ $4, SI \
	ADDQ $4, BX \
	ADDQ $4, DI \
	DECQ CX \
	JNZ c1_loop \
next_b: \
	DECQ R11 \
	JNZ b_loop \
	MOVQ BX, DX \
	DECQ R8 \
	JNZ a_loop \
done: \
	VZEROUPPER \
	RET

TEXT ·binaryMidAddFloat32AVX2(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32(VADDPS, VADDSS)

TEXT ·binaryMidSubRHSFloat32AVX2(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32(VSUBPS, VSUBSS)

TEXT ·binaryMidSubLHSFloat32AVX2(SB), NOSPLIT, $0-48
	BINARY_MID_LHS_BODY_F32(VSUBPS, VSUBSS)

TEXT ·binaryMidMulFloat32AVX2(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32(VMULPS, VMULSS)

TEXT ·binaryMidDivRHSFloat32AVX2(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32(VDIVPS, VDIVSS)

TEXT ·binaryMidDivLHSFloat32AVX2(SB), NOSPLIT, $0-48
	BINARY_MID_LHS_BODY_F32(VDIVPS, VDIVSS)

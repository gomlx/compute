// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// ============================================================================
// AVX-512 Binary Mid / Leading / Contiguous Broadcast Kernels (Float32)
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
//   - BroadcastNone ([N] op [N]):         A = 1, B = 1, C = N
//   - BroadcastLeadingRHS ([A,B] op [B]): A = 1, B = A_orig, C = B_orig
//   - BroadcastLeadingLHS ([B] op [A,B]): A = 1, B = A_orig, C = B_orig

#define BINARY_MID_RHS_BODY_F32_AVX512(VOP, SOP) \
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
	JE a_loop64 \
	CMPQ R10, $128 \
	JE a_loop128 \
	JMP a_loop \
a_loop64: \
	VMOVUPS (DX), Z8 \
	VMOVUPS 64(DX), Z9 \
	VMOVUPS 128(DX), Z10 \
	VMOVUPS 192(DX), Z11 \
	MOVQ R9, R11 \
	SHRQ $1, R11 \
	JZ b_tail64 \
b_loop64_2x: \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VMOVUPS 256(SI), Z4 \
	VMOVUPS 320(SI), Z5 \
	VMOVUPS 384(SI), Z6 \
	VMOVUPS 448(SI), Z7 \
	VOP Z8, Z0, Z0 \
	VOP Z9, Z1, Z1 \
	VOP Z10, Z2, Z2 \
	VOP Z11, Z3, Z3 \
	VOP Z8, Z4, Z4 \
	VOP Z9, Z5, Z5 \
	VOP Z10, Z6, Z6 \
	VOP Z11, Z7, Z7 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	VMOVUPS Z4, 256(DI) \
	VMOVUPS Z5, 320(DI) \
	VMOVUPS Z6, 384(DI) \
	VMOVUPS Z7, 448(DI) \
	ADDQ $512, SI \
	ADDQ $512, DI \
	DECQ R11 \
	JNZ b_loop64_2x \
b_tail64: \
	TESTQ $1, R9 \
	JZ next_a64 \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VOP Z8, Z0, Z0 \
	VOP Z9, Z1, Z1 \
	VOP Z10, Z2, Z2 \
	VOP Z11, Z3, Z3 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	ADDQ $256, SI \
	ADDQ $256, DI \
next_a64: \
	ADDQ $256, DX \
	DECQ R8 \
	JNZ a_loop64 \
	VZEROUPPER \
	RET \
a_loop128: \
	VMOVUPS (DX), Z8 \
	VMOVUPS 64(DX), Z9 \
	VMOVUPS 128(DX), Z10 \
	VMOVUPS 192(DX), Z11 \
	VMOVUPS 256(DX), Z12 \
	VMOVUPS 320(DX), Z13 \
	VMOVUPS 384(DX), Z14 \
	VMOVUPS 448(DX), Z15 \
	MOVQ R9, R11 \
b_loop128: \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VMOVUPS 256(SI), Z4 \
	VMOVUPS 320(SI), Z5 \
	VMOVUPS 384(SI), Z6 \
	VMOVUPS 448(SI), Z7 \
	VOP Z8, Z0, Z0 \
	VOP Z9, Z1, Z1 \
	VOP Z10, Z2, Z2 \
	VOP Z11, Z3, Z3 \
	VOP Z12, Z4, Z4 \
	VOP Z13, Z5, Z5 \
	VOP Z14, Z6, Z6 \
	VOP Z15, Z7, Z7 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	VMOVUPS Z4, 256(DI) \
	VMOVUPS Z5, 320(DI) \
	VMOVUPS Z6, 384(DI) \
	VMOVUPS Z7, 448(DI) \
	ADDQ $512, SI \
	ADDQ $512, DI \
	DECQ R11 \
	JNZ b_loop128 \
	ADDQ $512, DX \
	DECQ R8 \
	JNZ a_loop128 \
	VZEROUPPER \
	RET \
a_loop: \
	MOVQ R9, R11 \
b_loop: \
	MOVQ DX, BX \
	MOVQ R10, CX \
	SHRQ $6, CX \
	JZ tail16 \
c64_loop: \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VMOVUPS (BX), Z4 \
	VMOVUPS 64(BX), Z5 \
	VMOVUPS 128(BX), Z6 \
	VMOVUPS 192(BX), Z7 \
	VOP Z4, Z0, Z0 \
	VOP Z5, Z1, Z1 \
	VOP Z6, Z2, Z2 \
	VOP Z7, Z3, Z3 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	ADDQ $256, SI \
	ADDQ $256, BX \
	ADDQ $256, DI \
	DECQ CX \
	JNZ c64_loop \
tail16: \
	MOVQ R10, CX \
	SHRQ $4, CX \
	ANDQ $3, CX \
	JZ tail8 \
c16_loop: \
	VMOVUPS (SI), Z0 \
	VMOVUPS (BX), Z4 \
	VOP Z4, Z0, Z0 \
	VMOVUPS Z0, (DI) \
	ADDQ $64, SI \
	ADDQ $64, BX \
	ADDQ $64, DI \
	DECQ CX \
	JNZ c16_loop \
tail8: \
	TESTQ $8, R10 \
	JZ tail1 \
	VMOVUPS (SI), Y0 \
	VMOVUPS (BX), Y4 \
	VOP Y4, Y0, Y0 \
	VMOVUPS Y0, (DI) \
	ADDQ $32, SI \
	ADDQ $32, BX \
	ADDQ $32, DI \
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

#define BINARY_MID_LHS_BODY_F32_AVX512(VOP, SOP) \
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
	JE a_loop64 \
	CMPQ R10, $128 \
	JE a_loop128 \
	JMP a_loop \
a_loop64: \
	VMOVUPS (DX), Z8 \
	VMOVUPS 64(DX), Z9 \
	VMOVUPS 128(DX), Z10 \
	VMOVUPS 192(DX), Z11 \
	MOVQ R9, R11 \
	SHRQ $1, R11 \
	JZ b_tail64 \
b_loop64_2x: \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VMOVUPS 256(SI), Z4 \
	VMOVUPS 320(SI), Z5 \
	VMOVUPS 384(SI), Z6 \
	VMOVUPS 448(SI), Z7 \
	VOP Z0, Z8, Z0 \
	VOP Z1, Z9, Z1 \
	VOP Z2, Z10, Z2 \
	VOP Z3, Z11, Z3 \
	VOP Z4, Z8, Z4 \
	VOP Z5, Z9, Z5 \
	VOP Z6, Z10, Z6 \
	VOP Z7, Z11, Z7 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	VMOVUPS Z4, 256(DI) \
	VMOVUPS Z5, 320(DI) \
	VMOVUPS Z6, 384(DI) \
	VMOVUPS Z7, 448(DI) \
	ADDQ $512, SI \
	ADDQ $512, DI \
	DECQ R11 \
	JNZ b_loop64_2x \
b_tail64: \
	TESTQ $1, R9 \
	JZ next_a64 \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VOP Z0, Z8, Z0 \
	VOP Z1, Z9, Z1 \
	VOP Z2, Z10, Z2 \
	VOP Z3, Z11, Z3 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	ADDQ $256, SI \
	ADDQ $256, DI \
next_a64: \
	ADDQ $256, DX \
	DECQ R8 \
	JNZ a_loop64 \
	VZEROUPPER \
	RET \
a_loop128: \
	VMOVUPS (DX), Z8 \
	VMOVUPS 64(DX), Z9 \
	VMOVUPS 128(DX), Z10 \
	VMOVUPS 192(DX), Z11 \
	VMOVUPS 256(DX), Z12 \
	VMOVUPS 320(DX), Z13 \
	VMOVUPS 384(DX), Z14 \
	VMOVUPS 448(DX), Z15 \
	MOVQ R9, R11 \
b_loop128: \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VMOVUPS 256(SI), Z4 \
	VMOVUPS 320(SI), Z5 \
	VMOVUPS 384(SI), Z6 \
	VMOVUPS 448(SI), Z7 \
	VOP Z0, Z8, Z0 \
	VOP Z1, Z9, Z1 \
	VOP Z2, Z10, Z2 \
	VOP Z3, Z11, Z3 \
	VOP Z4, Z12, Z4 \
	VOP Z5, Z13, Z5 \
	VOP Z6, Z14, Z6 \
	VOP Z7, Z15, Z7 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	VMOVUPS Z4, 256(DI) \
	VMOVUPS Z5, 320(DI) \
	VMOVUPS Z6, 384(DI) \
	VMOVUPS Z7, 448(DI) \
	ADDQ $512, SI \
	ADDQ $512, DI \
	DECQ R11 \
	JNZ b_loop128 \
	ADDQ $512, DX \
	DECQ R8 \
	JNZ a_loop128 \
	VZEROUPPER \
	RET \
a_loop: \
	MOVQ R9, R11 \
b_loop: \
	MOVQ DX, BX \
	MOVQ R10, CX \
	SHRQ $6, CX \
	JZ tail16 \
c64_loop: \
	VMOVUPS (SI), Z0 \
	VMOVUPS 64(SI), Z1 \
	VMOVUPS 128(SI), Z2 \
	VMOVUPS 192(SI), Z3 \
	VMOVUPS (BX), Z4 \
	VMOVUPS 64(BX), Z5 \
	VMOVUPS 128(BX), Z6 \
	VMOVUPS 192(BX), Z7 \
	VOP Z0, Z4, Z0 \
	VOP Z1, Z5, Z1 \
	VOP Z2, Z6, Z2 \
	VOP Z3, Z7, Z3 \
	VMOVUPS Z0, (DI) \
	VMOVUPS Z1, 64(DI) \
	VMOVUPS Z2, 128(DI) \
	VMOVUPS Z3, 192(DI) \
	ADDQ $256, SI \
	ADDQ $256, BX \
	ADDQ $256, DI \
	DECQ CX \
	JNZ c64_loop \
tail16: \
	MOVQ R10, CX \
	SHRQ $4, CX \
	ANDQ $3, CX \
	JZ tail8 \
c16_loop: \
	VMOVUPS (SI), Z0 \
	VMOVUPS (BX), Z4 \
	VOP Z0, Z4, Z0 \
	VMOVUPS Z0, (DI) \
	ADDQ $64, SI \
	ADDQ $64, BX \
	ADDQ $64, DI \
	DECQ CX \
	JNZ c16_loop \
tail8: \
	TESTQ $8, R10 \
	JZ tail1 \
	VMOVUPS (SI), Y0 \
	VMOVUPS (BX), Y4 \
	VOP Y0, Y4, Y0 \
	VMOVUPS Y0, (DI) \
	ADDQ $32, SI \
	ADDQ $32, BX \
	ADDQ $32, DI \
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

TEXT ·binaryMidAddFloat32AVX512(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32_AVX512(VADDPS, VADDSS)

TEXT ·binaryMidSubRHSFloat32AVX512(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32_AVX512(VSUBPS, VSUBSS)

TEXT ·binaryMidSubLHSFloat32AVX512(SB), NOSPLIT, $0-48
	BINARY_MID_LHS_BODY_F32_AVX512(VSUBPS, VSUBSS)

TEXT ·binaryMidMulFloat32AVX512(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32_AVX512(VMULPS, VMULSS)

TEXT ·binaryMidDivRHSFloat32AVX512(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_BODY_F32_AVX512(VDIVPS, VDIVSS)

TEXT ·binaryMidDivLHSFloat32AVX512(SB), NOSPLIT, $0-48
	BINARY_MID_LHS_BODY_F32_AVX512(VDIVPS, VDIVSS)

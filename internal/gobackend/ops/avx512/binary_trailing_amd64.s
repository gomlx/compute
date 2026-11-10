// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// ============================================================================
// AVX-512 Binary Trailing Broadcast Kernels
// ============================================================================
// Signature: func(lhs, rhs, out unsafe.Pointer, A, B int)

// ----------------------------------------------------------------------------
// FLOAT32 (VLEN = 16, 4x unrolled = 64 floats/iter)
// ----------------------------------------------------------------------------

#define BINARY_TRAILING_RHS_BODY_F32_AVX512(VOP, SOP) \
	MOVQ lhs+0(FP), SI \
	MOVQ rhs+8(FP), DX \
	MOVQ out+16(FP), DI \
	MOVQ A+24(FP), R8 \
	MOVQ B+32(FP), R9 \
	TESTQ R8, R8 \
	JLE done \
	TESTQ R9, R9 \
	JLE done \
	MOVQ R9, R11 \
	SHRQ $6, R11 \
	MOVQ R9, R12 \
	SHRQ $4, R12 \
	ANDQ $3, R12 \
	XORQ R10, R10 \
row_loop: \
	CMPQ R10, R8 \
	JGE done \
	VBROADCASTSS (DX), Z0 \
	MOVQ R11, CX \
	TESTQ CX, CX \
	JZ tail16 \
loop64: \
	VMOVUPS (SI), Z1 \
	VMOVUPS 64(SI), Z2 \
	VMOVUPS 128(SI), Z3 \
	VMOVUPS 192(SI), Z4 \
	VOP Z0, Z1, Z1 \
	VOP Z0, Z2, Z2 \
	VOP Z0, Z3, Z3 \
	VOP Z0, Z4, Z4 \
	VMOVUPS Z1, (DI) \
	VMOVUPS Z2, 64(DI) \
	VMOVUPS Z3, 128(DI) \
	VMOVUPS Z4, 192(DI) \
	ADDQ $256, SI \
	ADDQ $256, DI \
	DECQ CX \
	JNZ loop64 \
tail16: \
	MOVQ R12, CX \
	TESTQ CX, CX \
	JZ tail8 \
loop16: \
	VMOVUPS (SI), Z1 \
	VOP Z0, Z1, Z2 \
	VMOVUPS Z2, (DI) \
	ADDQ $64, SI \
	ADDQ $64, DI \
	DECQ CX \
	JNZ loop16 \
tail8: \
	TESTQ $8, R9 \
	JZ tail1 \
	VMOVUPS (SI), Y1 \
	VOP Y0, Y1, Y2 \
	VMOVUPS Y2, (DI) \
	ADDQ $32, SI \
	ADDQ $32, DI \
tail1: \
	MOVQ R9, CX \
	ANDQ $7, CX \
	JZ next_row \
scalar_loop: \
	VMOVSS (SI), X1 \
	SOP X0, X1, X1 \
	VMOVSS X1, (DI) \
	ADDQ $4, SI \
	ADDQ $4, DI \
	DECQ CX \
	JNZ scalar_loop \
next_row: \
	ADDQ $4, DX \
	INCQ R10 \
	JMP row_loop \
done: \
	VZEROUPPER \
	RET

#define BINARY_TRAILING_LHS_BODY_F32_AVX512(VOP, SOP) \
	MOVQ lhs+0(FP), SI \
	MOVQ rhs+8(FP), DX \
	MOVQ out+16(FP), DI \
	MOVQ A+24(FP), R8 \
	MOVQ B+32(FP), R9 \
	TESTQ R8, R8 \
	JLE done \
	TESTQ R9, R9 \
	JLE done \
	MOVQ R9, R11 \
	SHRQ $6, R11 \
	MOVQ R9, R12 \
	SHRQ $4, R12 \
	ANDQ $3, R12 \
	XORQ R10, R10 \
row_loop: \
	CMPQ R10, R8 \
	JGE done \
	VBROADCASTSS (DX), Z0 \
	MOVQ R11, CX \
	TESTQ CX, CX \
	JZ tail16 \
loop64: \
	VMOVUPS (SI), Z1 \
	VMOVUPS 64(SI), Z2 \
	VMOVUPS 128(SI), Z3 \
	VMOVUPS 192(SI), Z4 \
	VOP Z1, Z0, Z1 \
	VOP Z2, Z0, Z2 \
	VOP Z3, Z0, Z3 \
	VOP Z4, Z0, Z4 \
	VMOVUPS Z1, (DI) \
	VMOVUPS Z2, 64(DI) \
	VMOVUPS Z3, 128(DI) \
	VMOVUPS Z4, 192(DI) \
	ADDQ $256, SI \
	ADDQ $256, DI \
	DECQ CX \
	JNZ loop64 \
tail16: \
	MOVQ R12, CX \
	TESTQ CX, CX \
	JZ tail8 \
loop16: \
	VMOVUPS (SI), Z1 \
	VOP Z1, Z0, Z2 \
	VMOVUPS Z2, (DI) \
	ADDQ $64, SI \
	ADDQ $64, DI \
	DECQ CX \
	JNZ loop16 \
tail8: \
	TESTQ $8, R9 \
	JZ tail1 \
	VMOVUPS (SI), Y1 \
	VOP Y1, Y0, Y2 \
	VMOVUPS Y2, (DI) \
	ADDQ $32, SI \
	ADDQ $32, DI \
tail1: \
	MOVQ R9, CX \
	ANDQ $7, CX \
	JZ next_row \
scalar_loop: \
	SOP (SI), X0, X1 \
	VMOVSS X1, (DI) \
	ADDQ $4, SI \
	ADDQ $4, DI \
	DECQ CX \
	JNZ scalar_loop \
next_row: \
	ADDQ $4, DX \
	INCQ R10 \
	JMP row_loop \
done: \
	VZEROUPPER \
	RET

TEXT ·binaryTrailingAddFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_BODY_F32_AVX512(VADDPS, VADDSS)

TEXT ·binaryTrailingSubRHSFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_BODY_F32_AVX512(VSUBPS, VSUBSS)

TEXT ·binaryTrailingSubLHSFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_LHS_BODY_F32_AVX512(VSUBPS, VSUBSS)

TEXT ·binaryTrailingMulFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_BODY_F32_AVX512(VMULPS, VMULSS)

TEXT ·binaryTrailingDivRHSFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_BODY_F32_AVX512(VDIVPS, VDIVSS)

TEXT ·binaryTrailingDivLHSFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_LHS_BODY_F32_AVX512(VDIVPS, VDIVSS)

TEXT ·binaryTrailingMaxFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_BODY_F32_AVX512(VMAXPS, VMAXSS)

TEXT ·binaryTrailingMinFloat32AVX512(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_BODY_F32_AVX512(VMINPS, VMINSS)

// ----------------------------------------------------------------------------
// FLOAT64 (VLEN = 8)
// ----------------------------------------------------------------------------

TEXT ·binaryTrailingAddFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_add_f64
	TESTQ R9, R9
	JLE done_add_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_add_f64:
	CMPQ R10, R8
	JGE done_add_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_add_f64
loop_add_f64:
	VMOVUPD (SI), Z1
	VADDPD Z0, Z1, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_add_f64
tail_add_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_add_f64
scalar_add_f64:
	VMOVSD (SI), X1
	VADDSD X0, X1, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_add_f64
next_add_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_add_f64
done_add_f64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubRHSFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_subrhs_f64
	TESTQ R9, R9
	JLE done_subrhs_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_subrhs_f64:
	CMPQ R10, R8
	JGE done_subrhs_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_subrhs_f64
loop_subrhs_f64:
	VMOVUPD (SI), Z1
	VSUBPD Z0, Z1, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_subrhs_f64
tail_subrhs_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_subrhs_f64
scalar_subrhs_f64:
	VMOVSD (SI), X1
	VSUBSD X0, X1, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_subrhs_f64
next_subrhs_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_subrhs_f64
done_subrhs_f64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubLHSFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_sublhs_f64
	TESTQ R9, R9
	JLE done_sublhs_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_sublhs_f64:
	CMPQ R10, R8
	JGE done_sublhs_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_sublhs_f64
loop_sublhs_f64:
	VMOVUPD (SI), Z1
	VSUBPD Z1, Z0, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_sublhs_f64
tail_sublhs_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_sublhs_f64
scalar_sublhs_f64:
	VSUBSD (SI), X0, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_sublhs_f64
next_sublhs_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_sublhs_f64
done_sublhs_f64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMulFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_mul_f64
	TESTQ R9, R9
	JLE done_mul_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_mul_f64:
	CMPQ R10, R8
	JGE done_mul_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_mul_f64
loop_mul_f64:
	VMOVUPD (SI), Z1
	VMULPD Z0, Z1, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_mul_f64
tail_mul_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_mul_f64
scalar_mul_f64:
	VMOVSD (SI), X1
	VMULSD X0, X1, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_mul_f64
next_mul_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_mul_f64
done_mul_f64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingDivRHSFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_divrhs_f64
	TESTQ R9, R9
	JLE done_divrhs_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_divrhs_f64:
	CMPQ R10, R8
	JGE done_divrhs_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_divrhs_f64
loop_divrhs_f64:
	VMOVUPD (SI), Z1
	VDIVPD Z0, Z1, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_divrhs_f64
tail_divrhs_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_divrhs_f64
scalar_divrhs_f64:
	VMOVSD (SI), X1
	VDIVSD X0, X1, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_divrhs_f64
next_divrhs_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_divrhs_f64
done_divrhs_f64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingDivLHSFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_divlhs_f64
	TESTQ R9, R9
	JLE done_divlhs_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_divlhs_f64:
	CMPQ R10, R8
	JGE done_divlhs_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_divlhs_f64
loop_divlhs_f64:
	VMOVUPD (SI), Z1
	VDIVPD Z1, Z0, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_divlhs_f64
tail_divlhs_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_divlhs_f64
scalar_divlhs_f64:
	VDIVSD (SI), X0, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_divlhs_f64
next_divlhs_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_divlhs_f64
done_divlhs_f64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMaxFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_max_f64
	TESTQ R9, R9
	JLE done_max_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_max_f64:
	CMPQ R10, R8
	JGE done_max_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_max_f64
loop_max_f64:
	VMOVUPD (SI), Z1
	VMAXPD Z0, Z1, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_max_f64
tail_max_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_max_f64
scalar_max_f64:
	VMOVSD (SI), X1
	VMAXSD X0, X1, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_max_f64
next_max_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_max_f64
done_max_f64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMinFloat64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_min_f64
	TESTQ R9, R9
	JLE done_min_f64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_min_f64:
	CMPQ R10, R8
	JGE done_min_f64
	VBROADCASTSD (DX), Z0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_min_f64
loop_min_f64:
	VMOVUPD (SI), Z1
	VMINPD Z0, Z1, Z2
	VMOVUPD Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_min_f64
tail_min_f64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_min_f64
scalar_min_f64:
	VMOVSD (SI), X1
	VMINSD X0, X1, X1
	VMOVSD X1, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_min_f64
next_min_f64:
	ADDQ $8, DX
	INCQ R10
	JMP row_min_f64
done_min_f64:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// INT32 / UINT32 (VLEN = 16)
// ----------------------------------------------------------------------------

TEXT ·binaryTrailingAddInt32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_add_i32
	TESTQ R9, R9
	JLE done_add_i32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_add_i32:
	CMPQ R10, R8
	JGE done_add_i32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_add_i32
loop_add_i32:
	VMOVDQU32 (SI), Z1
	VPADDD Z0, Z1, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_add_i32
tail_add_i32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_add_i32
scalar_add_i32:
	MOVL (SI), AX
	ADDL BX, AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_add_i32
next_add_i32:
	ADDQ $4, DX
	INCQ R10
	JMP row_add_i32
done_add_i32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubRHSInt32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_subrhs_i32
	TESTQ R9, R9
	JLE done_subrhs_i32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_subrhs_i32:
	CMPQ R10, R8
	JGE done_subrhs_i32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_subrhs_i32
loop_subrhs_i32:
	VMOVDQU32 (SI), Z1
	VPSUBD Z0, Z1, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_subrhs_i32
tail_subrhs_i32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_subrhs_i32
scalar_subrhs_i32:
	MOVL (SI), AX
	SUBL BX, AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_subrhs_i32
next_subrhs_i32:
	ADDQ $4, DX
	INCQ R10
	JMP row_subrhs_i32
done_subrhs_i32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubLHSInt32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_sublhs_i32
	TESTQ R9, R9
	JLE done_sublhs_i32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_sublhs_i32:
	CMPQ R10, R8
	JGE done_sublhs_i32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_sublhs_i32
loop_sublhs_i32:
	VMOVDQU32 (SI), Z1
	VPSUBD Z1, Z0, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_sublhs_i32
tail_sublhs_i32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_sublhs_i32
scalar_sublhs_i32:
	MOVL BX, AX
	SUBL (SI), AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_sublhs_i32
next_sublhs_i32:
	ADDQ $4, DX
	INCQ R10
	JMP row_sublhs_i32
done_sublhs_i32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMulInt32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_mul_i32
	TESTQ R9, R9
	JLE done_mul_i32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_mul_i32:
	CMPQ R10, R8
	JGE done_mul_i32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_mul_i32
loop_mul_i32:
	VMOVDQU32 (SI), Z1
	VPMULLD Z0, Z1, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_mul_i32
tail_mul_i32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_mul_i32
scalar_mul_i32:
	MOVL (SI), AX
	IMULL BX, AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_mul_i32
next_mul_i32:
	ADDQ $4, DX
	INCQ R10
	JMP row_mul_i32
done_mul_i32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMaxInt32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_max_i32
	TESTQ R9, R9
	JLE done_max_i32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_max_i32:
	CMPQ R10, R8
	JGE done_max_i32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_max_i32
loop_max_i32:
	VMOVDQU32 (SI), Z1
	VPMAXSD Z0, Z1, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_max_i32
tail_max_i32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_max_i32
scalar_max_i32:
	MOVL (SI), AX
	CMPL AX, BX
	CMOVLLT BX, AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_max_i32
next_max_i32:
	ADDQ $4, DX
	INCQ R10
	JMP row_max_i32
done_max_i32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMinInt32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_min_i32
	TESTQ R9, R9
	JLE done_min_i32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_min_i32:
	CMPQ R10, R8
	JGE done_min_i32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_min_i32
loop_min_i32:
	VMOVDQU32 (SI), Z1
	VPMINSD Z0, Z1, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_min_i32
tail_min_i32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_min_i32
scalar_min_i32:
	MOVL (SI), AX
	CMPL AX, BX
	CMOVLGT BX, AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_min_i32
next_min_i32:
	ADDQ $4, DX
	INCQ R10
	JMP row_min_i32
done_min_i32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMaxUint32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_max_u32
	TESTQ R9, R9
	JLE done_max_u32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_max_u32:
	CMPQ R10, R8
	JGE done_max_u32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_max_u32
loop_max_u32:
	VMOVDQU32 (SI), Z1
	VPMAXUD Z0, Z1, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_max_u32
tail_max_u32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_max_u32
scalar_max_u32:
	MOVL (SI), AX
	CMPL AX, BX
	CMOVLCS BX, AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_max_u32
next_max_u32:
	ADDQ $4, DX
	INCQ R10
	JMP row_max_u32
done_max_u32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMinUint32AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_min_u32
	TESTQ R9, R9
	JLE done_min_u32
	MOVQ R9, R11
	SHRQ $4, R11
	MOVQ R9, R12
	ANDQ $15, R12
	XORQ R10, R10
row_min_u32:
	CMPQ R10, R8
	JGE done_min_u32
	VPBROADCASTD (DX), Z0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_min_u32
loop_min_u32:
	VMOVDQU32 (SI), Z1
	VPMINUD Z0, Z1, Z2
	VMOVDQU32 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_min_u32
tail_min_u32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_min_u32
scalar_min_u32:
	MOVL (SI), AX
	CMPL AX, BX
	CMOVLHI BX, AX
	MOVL AX, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_min_u32
next_min_u32:
	ADDQ $4, DX
	INCQ R10
	JMP row_min_u32
done_min_u32:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// INT64 / UINT64 (VLEN = 8)
// ----------------------------------------------------------------------------

TEXT ·binaryTrailingAddInt64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_add_i64
	TESTQ R9, R9
	JLE done_add_i64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_add_i64:
	CMPQ R10, R8
	JGE done_add_i64
	VPBROADCASTQ (DX), Z0
	MOVQ (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_add_i64
loop_add_i64:
	VMOVDQU64 (SI), Z1
	VPADDQ Z0, Z1, Z2
	VMOVDQU64 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_add_i64
tail_add_i64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_add_i64
scalar_add_i64:
	MOVQ (SI), AX
	ADDQ BX, AX
	MOVQ AX, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_add_i64
next_add_i64:
	ADDQ $8, DX
	INCQ R10
	JMP row_add_i64
done_add_i64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubRHSInt64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_subrhs_i64
	TESTQ R9, R9
	JLE done_subrhs_i64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_subrhs_i64:
	CMPQ R10, R8
	JGE done_subrhs_i64
	VPBROADCASTQ (DX), Z0
	MOVQ (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_subrhs_i64
loop_subrhs_i64:
	VMOVDQU64 (SI), Z1
	VPSUBQ Z0, Z1, Z2
	VMOVDQU64 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_subrhs_i64
tail_subrhs_i64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_subrhs_i64
scalar_subrhs_i64:
	MOVQ (SI), AX
	SUBQ BX, AX
	MOVQ AX, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_subrhs_i64
next_subrhs_i64:
	ADDQ $8, DX
	INCQ R10
	JMP row_subrhs_i64
done_subrhs_i64:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubLHSInt64AVX512(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_sublhs_i64
	TESTQ R9, R9
	JLE done_sublhs_i64
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_sublhs_i64:
	CMPQ R10, R8
	JGE done_sublhs_i64
	VPBROADCASTQ (DX), Z0
	MOVQ (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_sublhs_i64
loop_sublhs_i64:
	VMOVDQU64 (SI), Z1
	VPSUBQ Z1, Z0, Z2
	VMOVDQU64 Z2, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JNZ loop_sublhs_i64
tail_sublhs_i64:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_sublhs_i64
scalar_sublhs_i64:
	MOVQ BX, AX
	SUBQ (SI), AX
	MOVQ AX, (DI)
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ scalar_sublhs_i64
next_sublhs_i64:
	ADDQ $8, DX
	INCQ R10
	JMP row_sublhs_i64
done_sublhs_i64:
	VZEROUPPER
	RET

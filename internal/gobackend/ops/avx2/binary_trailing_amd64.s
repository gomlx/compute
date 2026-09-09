// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// ============================================================================
// AVX2 Binary Trailing Broadcast Kernels
// ============================================================================
// Signature: func(lhs, rhs, out unsafe.Pointer, A, B int)

// ----------------------------------------------------------------------------
// FLOAT32
// ----------------------------------------------------------------------------

TEXT ·binaryTrailingAddFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_add_f32
	TESTQ R9, R9
	JLE done_add_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_add_f32:
	CMPQ R10, R8
	JGE done_add_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_add_f32
loop_add_f32:
	VMOVUPS (SI), Y1
	VADDPS Y0, Y1, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_add_f32
tail_add_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_add_f32
scalar_add_f32:
	VMOVSS (SI), X1
	VADDSS X0, X1, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_add_f32
next_add_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_add_f32
done_add_f32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubRHSFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_subrhs_f32
	TESTQ R9, R9
	JLE done_subrhs_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_subrhs_f32:
	CMPQ R10, R8
	JGE done_subrhs_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_subrhs_f32
loop_subrhs_f32:
	VMOVUPS (SI), Y1
	VSUBPS Y0, Y1, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_subrhs_f32
tail_subrhs_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_subrhs_f32
scalar_subrhs_f32:
	VMOVSS (SI), X1
	VSUBSS X0, X1, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_subrhs_f32
next_subrhs_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_subrhs_f32
done_subrhs_f32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingSubLHSFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_sublhs_f32
	TESTQ R9, R9
	JLE done_sublhs_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_sublhs_f32:
	CMPQ R10, R8
	JGE done_sublhs_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_sublhs_f32
loop_sublhs_f32:
	VMOVUPS (SI), Y1
	VSUBPS Y1, Y0, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_sublhs_f32
tail_sublhs_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_sublhs_f32
scalar_sublhs_f32:
	VSUBSS (SI), X0, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_sublhs_f32
next_sublhs_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_sublhs_f32
done_sublhs_f32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMulFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_mul_f32
	TESTQ R9, R9
	JLE done_mul_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_mul_f32:
	CMPQ R10, R8
	JGE done_mul_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_mul_f32
loop_mul_f32:
	VMOVUPS (SI), Y1
	VMULPS Y0, Y1, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_mul_f32
tail_mul_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_mul_f32
scalar_mul_f32:
	VMOVSS (SI), X1
	VMULSS X0, X1, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_mul_f32
next_mul_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_mul_f32
done_mul_f32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingDivRHSFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_divrhs_f32
	TESTQ R9, R9
	JLE done_divrhs_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_divrhs_f32:
	CMPQ R10, R8
	JGE done_divrhs_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_divrhs_f32
loop_divrhs_f32:
	VMOVUPS (SI), Y1
	VDIVPS Y0, Y1, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_divrhs_f32
tail_divrhs_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_divrhs_f32
scalar_divrhs_f32:
	VMOVSS (SI), X1
	VDIVSS X0, X1, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_divrhs_f32
next_divrhs_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_divrhs_f32
done_divrhs_f32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingDivLHSFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_divlhs_f32
	TESTQ R9, R9
	JLE done_divlhs_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_divlhs_f32:
	CMPQ R10, R8
	JGE done_divlhs_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_divlhs_f32
loop_divlhs_f32:
	VMOVUPS (SI), Y1
	VDIVPS Y1, Y0, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_divlhs_f32
tail_divlhs_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_divlhs_f32
scalar_divlhs_f32:
	VDIVSS (SI), X0, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_divlhs_f32
next_divlhs_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_divlhs_f32
done_divlhs_f32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMaxFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_max_f32
	TESTQ R9, R9
	JLE done_max_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_max_f32:
	CMPQ R10, R8
	JGE done_max_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_max_f32
loop_max_f32:
	VMOVUPS (SI), Y1
	VMAXPS Y0, Y1, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_max_f32
tail_max_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_max_f32
scalar_max_f32:
	VMOVSS (SI), X1
	VMAXSS X0, X1, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_max_f32
next_max_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_max_f32
done_max_f32:
	VZEROUPPER
	RET

TEXT ·binaryTrailingMinFloat32AVX2(SB), NOSPLIT, $0-40
	MOVQ lhs+0(FP), SI
	MOVQ rhs+8(FP), DX
	MOVQ out+16(FP), DI
	MOVQ A+24(FP), R8
	MOVQ B+32(FP), R9
	TESTQ R8, R8
	JLE done_min_f32
	TESTQ R9, R9
	JLE done_min_f32
	MOVQ R9, R11
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_min_f32:
	CMPQ R10, R8
	JGE done_min_f32
	VBROADCASTSS (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_min_f32
loop_min_f32:
	VMOVUPS (SI), Y1
	VMINPS Y0, Y1, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop_min_f32
tail_min_f32:
	MOVQ R12, CX
	TESTQ CX, CX
	JZ next_min_f32
scalar_min_f32:
	VMOVSS (SI), X1
	VMINSS X0, X1, X1
	VMOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ scalar_min_f32
next_min_f32:
	ADDQ $4, DX
	INCQ R10
	JMP row_min_f32
done_min_f32:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// FLOAT64
// ----------------------------------------------------------------------------

TEXT ·binaryTrailingAddFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_add_f64:
	CMPQ R10, R8
	JGE done_add_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_add_f64
loop_add_f64:
	VMOVUPD (SI), Y1
	VADDPD Y0, Y1, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingSubRHSFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_subrhs_f64:
	CMPQ R10, R8
	JGE done_subrhs_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_subrhs_f64
loop_subrhs_f64:
	VMOVUPD (SI), Y1
	VSUBPD Y0, Y1, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingSubLHSFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_sublhs_f64:
	CMPQ R10, R8
	JGE done_sublhs_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_sublhs_f64
loop_sublhs_f64:
	VMOVUPD (SI), Y1
	VSUBPD Y1, Y0, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMulFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_mul_f64:
	CMPQ R10, R8
	JGE done_mul_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_mul_f64
loop_mul_f64:
	VMOVUPD (SI), Y1
	VMULPD Y0, Y1, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingDivRHSFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_divrhs_f64:
	CMPQ R10, R8
	JGE done_divrhs_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_divrhs_f64
loop_divrhs_f64:
	VMOVUPD (SI), Y1
	VDIVPD Y0, Y1, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingDivLHSFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_divlhs_f64:
	CMPQ R10, R8
	JGE done_divlhs_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_divlhs_f64
loop_divlhs_f64:
	VMOVUPD (SI), Y1
	VDIVPD Y1, Y0, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMaxFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_max_f64:
	CMPQ R10, R8
	JGE done_max_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_max_f64
loop_max_f64:
	VMOVUPD (SI), Y1
	VMAXPD Y0, Y1, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMinFloat64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_min_f64:
	CMPQ R10, R8
	JGE done_min_f64
	VBROADCASTSD (DX), Y0
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_min_f64
loop_min_f64:
	VMOVUPD (SI), Y1
	VMINPD Y0, Y1, Y2
	VMOVUPD Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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
// INT32 / UINT32
// ----------------------------------------------------------------------------

TEXT ·binaryTrailingAddInt32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_add_i32:
	CMPQ R10, R8
	JGE done_add_i32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_add_i32
loop_add_i32:
	VMOVDQU (SI), Y1
	VPADDD Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingSubRHSInt32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_subrhs_i32:
	CMPQ R10, R8
	JGE done_subrhs_i32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_subrhs_i32
loop_subrhs_i32:
	VMOVDQU (SI), Y1
	VPSUBD Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingSubLHSInt32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_sublhs_i32:
	CMPQ R10, R8
	JGE done_sublhs_i32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_sublhs_i32
loop_sublhs_i32:
	VMOVDQU (SI), Y1
	VPSUBD Y1, Y0, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMulInt32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_mul_i32:
	CMPQ R10, R8
	JGE done_mul_i32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_mul_i32
loop_mul_i32:
	VMOVDQU (SI), Y1
	VPMULLD Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMaxInt32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_max_i32:
	CMPQ R10, R8
	JGE done_max_i32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_max_i32
loop_max_i32:
	VMOVDQU (SI), Y1
	VPMAXSD Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMinInt32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_min_i32:
	CMPQ R10, R8
	JGE done_min_i32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_min_i32
loop_min_i32:
	VMOVDQU (SI), Y1
	VPMINSD Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMaxUint32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_max_u32:
	CMPQ R10, R8
	JGE done_max_u32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_max_u32
loop_max_u32:
	VMOVDQU (SI), Y1
	VPMAXUD Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingMinUint32AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $3, R11
	MOVQ R9, R12
	ANDQ $7, R12
	XORQ R10, R10
row_min_u32:
	CMPQ R10, R8
	JGE done_min_u32
	VPBROADCASTD (DX), Y0
	MOVL (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_min_u32
loop_min_u32:
	VMOVDQU (SI), Y1
	VPMINUD Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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
// INT64 / UINT64
// ----------------------------------------------------------------------------

TEXT ·binaryTrailingAddInt64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_add_i64:
	CMPQ R10, R8
	JGE done_add_i64
	VPBROADCASTQ (DX), Y0
	MOVQ (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_add_i64
loop_add_i64:
	VMOVDQU (SI), Y1
	VPADDQ Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingSubRHSInt64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_subrhs_i64:
	CMPQ R10, R8
	JGE done_subrhs_i64
	VPBROADCASTQ (DX), Y0
	MOVQ (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_subrhs_i64
loop_subrhs_i64:
	VMOVDQU (SI), Y1
	VPSUBQ Y0, Y1, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

TEXT ·binaryTrailingSubLHSInt64AVX2(SB), NOSPLIT, $0-40
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
	SHRQ $2, R11
	MOVQ R9, R12
	ANDQ $3, R12
	XORQ R10, R10
row_sublhs_i64:
	CMPQ R10, R8
	JGE done_sublhs_i64
	VPBROADCASTQ (DX), Y0
	MOVQ (DX), BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail_sublhs_i64
loop_sublhs_i64:
	VMOVDQU (SI), Y1
	VPSUBQ Y1, Y0, Y2
	VMOVDQU Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
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

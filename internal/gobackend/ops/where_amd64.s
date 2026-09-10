// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// ----------------------------------------------------------------------------
// AVX2 FLOAT32
// ----------------------------------------------------------------------------

// func whereVVFloat32AVX2Asm(cond, onTrue, onFalse, out unsafe.Pointer, count int)
TEXT ·whereVVFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrue+8(FP), BX
	MOVQ onFalse+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	CMPQ SI, $8
	JL tail_vv2

	VPXOR Y1, Y1, Y1

loop8_vv2:
	VMOVQ (AX), X0
	VPMOVSXBD X0, Y0
	VPCMPGTD Y1, Y0, Y0
	VMOVUPS (BX), Y2
	VMOVUPS (CX), Y3
	VANDPS Y0, Y2, Y4
	VANDNPS Y3, Y0, Y5
	VORPS Y4, Y5, Y6
	VMOVUPS Y6, (DX)

	ADDQ $8, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	SUBQ $8, SI
	CMPQ SI, $8
	JGE loop8_vv2

tail_vv2:
	TESTQ SI, SI
	JLE done_vv2

loop_scalar_vv2:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_vv2
	MOVL (BX), R9
	MOVL R9, (DX)
	JMP next_scalar_vv2
false_scalar_vv2:
	MOVL (CX), R9
	MOVL R9, (DX)
next_scalar_vv2:
	INCQ AX
	ADDQ $4, BX
	ADDQ $4, CX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_vv2

done_vv2:
	VZEROUPPER
	RET

// func whereVSFloat32AVX2Asm(cond, onTrue, onFalseScalar, out unsafe.Pointer, count int)
TEXT ·whereVSFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrue+8(FP), BX
	MOVQ onFalseScalar+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	VBROADCASTSS (CX), Y3
	VPXOR Y1, Y1, Y1

	CMPQ SI, $8
	JL tail_vs2

loop8_vs2:
	VMOVQ (AX), X0
	VPMOVSXBD X0, Y0
	VPCMPGTD Y1, Y0, Y0
	VMOVUPS (BX), Y2
	VANDPS Y0, Y2, Y4
	VANDNPS Y3, Y0, Y5
	VORPS Y4, Y5, Y6
	VMOVUPS Y6, (DX)

	ADDQ $8, AX
	ADDQ $32, BX
	ADDQ $32, DX
	SUBQ $8, SI
	CMPQ SI, $8
	JGE loop8_vs2

tail_vs2:
	TESTQ SI, SI
	JLE done_vs2
	MOVL (CX), R10

loop_scalar_vs2:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_vs2
	MOVL (BX), R9
	MOVL R9, (DX)
	JMP next_scalar_vs2
false_scalar_vs2:
	MOVL R10, (DX)
next_scalar_vs2:
	INCQ AX
	ADDQ $4, BX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_vs2

done_vs2:
	VZEROUPPER
	RET

// func whereSVFloat32AVX2Asm(cond, onTrueScalar, onFalse, out unsafe.Pointer, count int)
TEXT ·whereSVFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrueScalar+8(FP), BX
	MOVQ onFalse+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	VBROADCASTSS (BX), Y2
	VPXOR Y1, Y1, Y1

	CMPQ SI, $8
	JL tail_sv2

loop8_sv2:
	VMOVQ (AX), X0
	VPMOVSXBD X0, Y0
	VPCMPGTD Y1, Y0, Y0
	VMOVUPS (CX), Y3
	VANDPS Y0, Y2, Y4
	VANDNPS Y3, Y0, Y5
	VORPS Y4, Y5, Y6
	VMOVUPS Y6, (DX)

	ADDQ $8, AX
	ADDQ $32, CX
	ADDQ $32, DX
	SUBQ $8, SI
	CMPQ SI, $8
	JGE loop8_sv2

tail_sv2:
	TESTQ SI, SI
	JLE done_sv2
	MOVL (BX), R10

loop_scalar_sv2:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_sv2
	MOVL R10, (DX)
	JMP next_scalar_sv2
false_scalar_sv2:
	MOVL (CX), R9
	MOVL R9, (DX)
next_scalar_sv2:
	INCQ AX
	ADDQ $4, CX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_sv2

done_sv2:
	VZEROUPPER
	RET

// func whereSSFloat32AVX2Asm(cond, onTrueScalar, onFalseScalar, out unsafe.Pointer, count int)
TEXT ·whereSSFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrueScalar+8(FP), BX
	MOVQ onFalseScalar+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	VBROADCASTSS (BX), Y2
	VBROADCASTSS (CX), Y3
	VPXOR Y1, Y1, Y1

	CMPQ SI, $8
	JL tail_ss2

loop8_ss2:
	VMOVQ (AX), X0
	VPMOVSXBD X0, Y0
	VPCMPGTD Y1, Y0, Y0
	VANDPS Y0, Y2, Y4
	VANDNPS Y3, Y0, Y5
	VORPS Y4, Y5, Y6
	VMOVUPS Y6, (DX)

	ADDQ $8, AX
	ADDQ $32, DX
	SUBQ $8, SI
	CMPQ SI, $8
	JGE loop8_ss2

tail_ss2:
	TESTQ SI, SI
	JLE done_ss2
	MOVL (BX), R10
	MOVL (CX), R11

loop_scalar_ss2:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_ss2
	MOVL R10, (DX)
	JMP next_scalar_ss2
false_scalar_ss2:
	MOVL R11, (DX)
next_scalar_ss2:
	INCQ AX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_ss2

done_ss2:
	VZEROUPPER
	RET

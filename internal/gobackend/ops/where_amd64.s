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

// ----------------------------------------------------------------------------
// AVX-512 FLOAT32
// ----------------------------------------------------------------------------

// func whereVVFloat32AVX512Asm(cond, onTrue, onFalse, out unsafe.Pointer, count int)
TEXT ·whereVVFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrue+8(FP), BX
	MOVQ onFalse+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	CMPQ SI, $32
	JL check16_vv512

loop32_vv512:
	VPMOVZXBD (AX), Z0
	VPMOVZXBD 16(AX), Z1
	VPTESTMD Z0, Z0, K1
	VPTESTMD Z1, Z1, K2
	VMOVUPS (CX), Z4
	VMOVUPS 64(CX), Z5
	VMOVUPS (BX), K1, Z4
	VMOVUPS 64(BX), K2, Z5
	VMOVUPS Z4, (DX)
	VMOVUPS Z5, 64(DX)

	ADDQ $32, AX
	ADDQ $128, BX
	ADDQ $128, CX
	ADDQ $128, DX
	SUBQ $32, SI
	CMPQ SI, $32
	JGE loop32_vv512

check16_vv512:
	CMPQ SI, $16
	JL tail_vv512

	VPMOVZXBD (AX), Z0
	VPTESTMD Z0, Z0, K1
	VMOVUPS (CX), Z4
	VMOVUPS (BX), K1, Z4
	VMOVUPS Z4, (DX)

	ADDQ $16, AX
	ADDQ $64, BX
	ADDQ $64, CX
	ADDQ $64, DX
	SUBQ $16, SI

tail_vv512:
	TESTQ SI, SI
	JLE done_vv512

loop_scalar_vv512:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_vv512
	MOVL (BX), R9
	MOVL R9, (DX)
	JMP next_scalar_vv512
false_scalar_vv512:
	MOVL (CX), R9
	MOVL R9, (DX)
next_scalar_vv512:
	INCQ AX
	ADDQ $4, BX
	ADDQ $4, CX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_vv512

done_vv512:
	VZEROUPPER
	RET

// func whereVSFloat32AVX512Asm(cond, onTrue, onFalseScalar, out unsafe.Pointer, count int)
TEXT ·whereVSFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrue+8(FP), BX
	MOVQ onFalseScalar+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	VBROADCASTSS (CX), Z3

	CMPQ SI, $32
	JL check16_vs512

loop32_vs512:
	VPMOVZXBD (AX), Z0
	VPMOVZXBD 16(AX), Z1
	VPTESTMD Z0, Z0, K1
	VPTESTMD Z1, Z1, K2
	VMOVAPS Z3, Z4
	VMOVAPS Z3, Z5
	VMOVUPS (BX), K1, Z4
	VMOVUPS 64(BX), K2, Z5
	VMOVUPS Z4, (DX)
	VMOVUPS Z5, 64(DX)

	ADDQ $32, AX
	ADDQ $128, BX
	ADDQ $128, DX
	SUBQ $32, SI
	CMPQ SI, $32
	JGE loop32_vs512

check16_vs512:
	CMPQ SI, $16
	JL tail_vs512

	VPMOVZXBD (AX), Z0
	VPTESTMD Z0, Z0, K1
	VMOVAPS Z3, Z4
	VMOVUPS (BX), K1, Z4
	VMOVUPS Z4, (DX)

	ADDQ $16, AX
	ADDQ $64, BX
	ADDQ $64, DX
	SUBQ $16, SI

tail_vs512:
	TESTQ SI, SI
	JLE done_vs512
	MOVL (CX), R10

loop_scalar_vs512:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_vs512
	MOVL (BX), R9
	MOVL R9, (DX)
	JMP next_scalar_vs512
false_scalar_vs512:
	MOVL R10, (DX)
next_scalar_vs512:
	INCQ AX
	ADDQ $4, BX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_vs512

done_vs512:
	VZEROUPPER
	RET

// func whereSVFloat32AVX512Asm(cond, onTrueScalar, onFalse, out unsafe.Pointer, count int)
TEXT ·whereSVFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrueScalar+8(FP), BX
	MOVQ onFalse+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	VBROADCASTSS (BX), Z2

	CMPQ SI, $32
	JL check16_sv512

loop32_sv512:
	VPMOVZXBD (AX), Z0
	VPMOVZXBD 16(AX), Z1
	VPTESTMD Z0, Z0, K1
	VPTESTMD Z1, Z1, K2
	VMOVUPS (CX), Z4
	VMOVUPS 64(CX), Z5
	VMOVAPS Z2, K1, Z4
	VMOVAPS Z2, K2, Z5
	VMOVUPS Z4, (DX)
	VMOVUPS Z5, 64(DX)

	ADDQ $32, AX
	ADDQ $128, CX
	ADDQ $128, DX
	SUBQ $32, SI
	CMPQ SI, $32
	JGE loop32_sv512

check16_sv512:
	CMPQ SI, $16
	JL tail_sv512

	VPMOVZXBD (AX), Z0
	VPTESTMD Z0, Z0, K1
	VMOVUPS (CX), Z4
	VMOVAPS Z2, K1, Z4
	VMOVUPS Z4, (DX)

	ADDQ $16, AX
	ADDQ $64, CX
	ADDQ $64, DX
	SUBQ $16, SI

tail_sv512:
	TESTQ SI, SI
	JLE done_sv512
	MOVL (BX), R10

loop_scalar_sv512:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_sv512
	MOVL R10, (DX)
	JMP next_scalar_sv512
false_scalar_sv512:
	MOVL (CX), R9
	MOVL R9, (DX)
next_scalar_sv512:
	INCQ AX
	ADDQ $4, CX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_sv512

done_sv512:
	VZEROUPPER
	RET

// func whereSSFloat32AVX512Asm(cond, onTrueScalar, onFalseScalar, out unsafe.Pointer, count int)
TEXT ·whereSSFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ cond+0(FP), AX
	MOVQ onTrueScalar+8(FP), BX
	MOVQ onFalseScalar+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ count+32(FP), SI

	VBROADCASTSS (BX), Z2
	VBROADCASTSS (CX), Z3

	CMPQ SI, $32
	JL check16_ss512

loop32_ss512:
	VPMOVZXBD (AX), Z0
	VPMOVZXBD 16(AX), Z1
	VPTESTMD Z0, Z0, K1
	VPTESTMD Z1, Z1, K2
	VMOVAPS Z3, Z4
	VMOVAPS Z3, Z5
	VMOVAPS Z2, K1, Z4
	VMOVAPS Z2, K2, Z5
	VMOVUPS Z4, (DX)
	VMOVUPS Z5, 64(DX)

	ADDQ $32, AX
	ADDQ $128, DX
	SUBQ $32, SI
	CMPQ SI, $32
	JGE loop32_ss512

check16_ss512:
	CMPQ SI, $16
	JL tail_ss512

	VPMOVZXBD (AX), Z0
	VPTESTMD Z0, Z0, K1
	VMOVAPS Z3, Z4
	VMOVAPS Z2, K1, Z4
	VMOVUPS Z4, (DX)

	ADDQ $16, AX
	ADDQ $64, DX
	SUBQ $16, SI

tail_ss512:
	TESTQ SI, SI
	JLE done_ss512
	MOVL (BX), R10
	MOVL (CX), R11

loop_scalar_ss512:
	MOVB (AX), R8
	TESTB R8, R8
	JZ false_scalar_ss512
	MOVL R10, (DX)
	JMP next_scalar_ss512
false_scalar_ss512:
	MOVL R11, (DX)
next_scalar_ss512:
	INCQ AX
	ADDQ $4, DX
	DECQ SI
	JNZ loop_scalar_ss512

done_ss512:
	VZEROUPPER
	RET

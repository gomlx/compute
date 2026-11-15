// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// func float16ToFloat32AVX2Asm(dst, src unsafe.Pointer, n int)
// Converts n IEEE-754 float16 values from src to float32 in dst using F16C/AVX2.
TEXT ·float16ToFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX

	TESTQ CX, CX
	JLE   f16_to_f32_avx2_done

	PCALIGN $32
f16_to_f32_avx2_loop32:
	CMPQ CX, $32
	JL   f16_to_f32_avx2_loop8

	VCVTPH2PS (SI), Y0
	VCVTPH2PS 16(SI), Y1
	VCVTPH2PS 32(SI), Y2
	VCVTPH2PS 48(SI), Y3
	VMOVUPS   Y0, (DI)
	VMOVUPS   Y1, 32(DI)
	VMOVUPS   Y2, 64(DI)
	VMOVUPS   Y3, 96(DI)

	ADDQ $64, SI
	ADDQ $128, DI
	SUBQ $32, CX
	JMP  f16_to_f32_avx2_loop32

f16_to_f32_avx2_loop8:
	CMPQ CX, $8
	JL   f16_to_f32_avx2_check4

	VCVTPH2PS (SI), Y0
	VMOVUPS   Y0, (DI)
	ADDQ      $16, SI
	ADDQ      $32, DI
	SUBQ      $8, CX
	JMP       f16_to_f32_avx2_loop8

f16_to_f32_avx2_check4:
	CMPQ CX, $4
	JL   f16_to_f32_avx2_scalar

	VCVTPH2PS (SI), X0
	VMOVUPS   X0, (DI)
	ADDQ      $8, SI
	ADDQ      $16, DI
	SUBQ      $4, CX

f16_to_f32_avx2_scalar:
	TESTQ CX, CX
	JLE   f16_to_f32_avx2_done

f16_to_f32_avx2_scalar_loop:
	MOVWLZX   (SI), AX
	VMOVD     AX, X0
	VCVTPH2PS X0, X0
	VMOVSS    X0, (DI)
	ADDQ      $2, SI
	ADDQ      $4, DI
	DECQ      CX
	JNZ       f16_to_f32_avx2_scalar_loop

f16_to_f32_avx2_done:
	VZEROUPPER
	RET

// func float32ToFloat16AVX2Asm(dst, src unsafe.Pointer, n int)
// Converts n float32 values from src to IEEE-754 float16 (RNE, imm=0) in dst using F16C/AVX2.
TEXT ·float32ToFloat16AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX

	TESTQ CX, CX
	JLE   f32_to_f16_avx2_done

	PCALIGN $32
f32_to_f16_avx2_loop32:
	CMPQ CX, $32
	JL   f32_to_f16_avx2_loop8

	VMOVUPS   (SI), Y0
	VMOVUPS   32(SI), Y1
	VMOVUPS   64(SI), Y2
	VMOVUPS   96(SI), Y3
	VCVTPS2PH $0, Y0, X0
	VCVTPS2PH $0, Y1, X1
	VCVTPS2PH $0, Y2, X2
	VCVTPS2PH $0, Y3, X3
	VMOVDQU   X0, (DI)
	VMOVDQU   X1, 16(DI)
	VMOVDQU   X2, 32(DI)
	VMOVDQU   X3, 48(DI)

	ADDQ $128, SI
	ADDQ $64, DI
	SUBQ $32, CX
	JMP  f32_to_f16_avx2_loop32

f32_to_f16_avx2_loop8:
	CMPQ CX, $8
	JL   f32_to_f16_avx2_check4

	VMOVUPS   (SI), Y0
	VCVTPS2PH $0, Y0, X0
	VMOVDQU   X0, (DI)
	ADDQ      $32, SI
	ADDQ      $16, DI
	SUBQ      $8, CX
	JMP       f32_to_f16_avx2_loop8

f32_to_f16_avx2_check4:
	CMPQ CX, $4
	JL   f32_to_f16_avx2_scalar

	VMOVUPS   (SI), X0
	VCVTPS2PH $0, X0, X0
	VMOVQ     X0, (DI)
	ADDQ      $16, SI
	ADDQ      $8, DI
	SUBQ      $4, CX

f32_to_f16_avx2_scalar:
	TESTQ CX, CX
	JLE   f32_to_f16_avx2_done

f32_to_f16_avx2_scalar_loop:
	VMOVSS    (SI), X0
	VCVTPS2PH $0, X0, X0
	VMOVD     X0, AX
	MOVW      AX, (DI)
	ADDQ      $4, SI
	ADDQ      $2, DI
	DECQ      CX
	JNZ       f32_to_f16_avx2_scalar_loop

f32_to_f16_avx2_done:
	VZEROUPPER
	RET

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// func float16ToFloat32AVX512Asm(dst, src unsafe.Pointer, n int)
// Converts n IEEE-754 float16 values from src to float32 in dst using F16C/AVX-512.
TEXT ·float16ToFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX

	TESTQ CX, CX
	JLE   f16_to_f32_done

	PCALIGN $32
f16_to_f32_loop64:
	CMPQ CX, $64
	JL   f16_to_f32_loop16

	VCVTPH2PS (SI), Z0
	VCVTPH2PS 32(SI), Z1
	VCVTPH2PS 64(SI), Z2
	VCVTPH2PS 96(SI), Z3
	VMOVUPS   Z0, (DI)
	VMOVUPS   Z1, 64(DI)
	VMOVUPS   Z2, 128(DI)
	VMOVUPS   Z3, 192(DI)

	ADDQ $128, SI
	ADDQ $256, DI
	SUBQ $64, CX
	JMP  f16_to_f32_loop64

f16_to_f32_loop16:
	CMPQ CX, $16
	JL   f16_to_f32_check8

	VCVTPH2PS (SI), Z0
	VMOVUPS   Z0, (DI)
	ADDQ      $32, SI
	ADDQ      $64, DI
	SUBQ      $16, CX
	JMP       f16_to_f32_loop16

f16_to_f32_check8:
	CMPQ CX, $8
	JL   f16_to_f32_check4

	VCVTPH2PS (SI), Y0
	VMOVUPS   Y0, (DI)
	ADDQ      $16, SI
	ADDQ      $32, DI
	SUBQ      $8, CX

f16_to_f32_check4:
	CMPQ CX, $4
	JL   f16_to_f32_scalar

	VCVTPH2PS (SI), X0
	VMOVUPS   X0, (DI)
	ADDQ      $8, SI
	ADDQ      $16, DI
	SUBQ      $4, CX

f16_to_f32_scalar:
	TESTQ CX, CX
	JLE   f16_to_f32_done

f16_to_f32_scalar_loop:
	MOVWLZX   (SI), AX
	VMOVD     AX, X0
	VCVTPH2PS X0, X0
	VMOVSS    X0, (DI)
	ADDQ      $2, SI
	ADDQ      $4, DI
	DECQ      CX
	JNZ       f16_to_f32_scalar_loop

f16_to_f32_done:
	VZEROUPPER
	RET

// func float32ToFloat16AVX512Asm(dst, src unsafe.Pointer, n int)
// Converts n float32 values from src to IEEE-754 float16 (RNE, imm=0) in dst using F16C/AVX-512.
TEXT ·float32ToFloat16AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX

	TESTQ CX, CX
	JLE   f32_to_f16_done

	PCALIGN $32
f32_to_f16_loop64:
	CMPQ CX, $64
	JL   f32_to_f16_loop16

	VMOVUPS   (SI), Z0
	VMOVUPS   64(SI), Z1
	VMOVUPS   128(SI), Z2
	VMOVUPS   192(SI), Z3
	VCVTPS2PH $0, Z0, Y0
	VCVTPS2PH $0, Z1, Y1
	VCVTPS2PH $0, Z2, Y2
	VCVTPS2PH $0, Z3, Y3
	VMOVDQU   Y0, (DI)
	VMOVDQU   Y1, 32(DI)
	VMOVDQU   Y2, 64(DI)
	VMOVDQU   Y3, 96(DI)

	ADDQ $256, SI
	ADDQ $128, DI
	SUBQ $64, CX
	JMP  f32_to_f16_loop64

f32_to_f16_loop16:
	CMPQ CX, $16
	JL   f32_to_f16_check8

	VMOVUPS   (SI), Z0
	VCVTPS2PH $0, Z0, Y0
	VMOVDQU   Y0, (DI)
	ADDQ      $64, SI
	ADDQ      $32, DI
	SUBQ      $16, CX
	JMP       f32_to_f16_loop16

f32_to_f16_check8:
	CMPQ CX, $8
	JL   f32_to_f16_check4

	VMOVUPS   (SI), Y0
	VCVTPS2PH $0, Y0, X0
	VMOVDQU   X0, (DI)
	ADDQ      $32, SI
	ADDQ      $16, DI
	SUBQ      $8, CX

f32_to_f16_check4:
	CMPQ CX, $4
	JL   f32_to_f16_scalar

	VMOVUPS   (SI), X0
	VCVTPS2PH $0, X0, X0
	VMOVQ     X0, (DI)
	ADDQ      $16, SI
	ADDQ      $8, DI
	SUBQ      $4, CX

f32_to_f16_scalar:
	TESTQ CX, CX
	JLE   f32_to_f16_done

f32_to_f16_scalar_loop:
	VMOVSS    (SI), X0
	VCVTPS2PH $0, X0, X0
	VMOVD     X0, AX
	MOVW      AX, (DI)
	ADDQ      $4, SI
	ADDQ      $2, DI
	DECQ      CX
	JNZ       f32_to_f16_scalar_loop

f32_to_f16_done:
	VZEROUPPER
	RET

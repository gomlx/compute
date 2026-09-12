// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func addBiasFloat32AVX512Asm(row, bias unsafe.Pointer, n int)
TEXT ·addBiasFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ row+0(FP), DI
	MOVQ bias+8(FP), SI
	MOVQ n+16(FP), DX

loop64:
	CMPQ DX, $64
	JL loop16
	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1
	VMOVUPS 128(SI), Z2
	VMOVUPS 192(SI), Z3
	VADDPS 0(DI), Z0, Z0
	VADDPS 64(DI), Z1, Z1
	VADDPS 128(DI), Z2, Z2
	VADDPS 192(DI), Z3, Z3
	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)
	VMOVUPS Z2, 128(DI)
	VMOVUPS Z3, 192(DI)
	ADDQ $256, DI
	ADDQ $256, SI
	SUBQ $64, DX
	JMP loop64

loop16:
	CMPQ DX, $16
	JL loop1
	VMOVUPS 0(SI), Z0
	VADDPS 0(DI), Z0, Z0
	VMOVUPS Z0, 0(DI)
	ADDQ $64, DI
	ADDQ $64, SI
	SUBQ $16, DX
	JMP loop16

loop1:
	CMPQ DX, $0
	JLE done
	VMOVSS 0(SI), X0
	VADDSS 0(DI), X0, X0
	VMOVSS X0, 0(DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ DX
	JMP loop1

done:
	VZEROUPPER
	RET

// func addBiasFloat32AVX2Asm(row, bias unsafe.Pointer, n int)
TEXT ·addBiasFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ row+0(FP), DI
	MOVQ bias+8(FP), SI
	MOVQ n+16(FP), DX

loop32:
	CMPQ DX, $32
	JL loop8
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 64(SI), Y2
	VMOVUPS 96(SI), Y3
	VADDPS 0(DI), Y0, Y0
	VADDPS 32(DI), Y1, Y1
	VADDPS 64(DI), Y2, Y2
	VADDPS 96(DI), Y3, Y3
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)
	VMOVUPS Y2, 64(DI)
	VMOVUPS Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	SUBQ $32, DX
	JMP loop32

loop8:
	CMPQ DX, $8
	JL loop1_avx2
	VMOVUPS 0(SI), Y0
	VADDPS 0(DI), Y0, Y0
	VMOVUPS Y0, 0(DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, DX
	JMP loop8

loop1_avx2:
	CMPQ DX, $0
	JLE done_avx2
	VMOVSS 0(SI), X0
	VADDSS 0(DI), X0, X0
	VMOVSS X0, 0(DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ DX
	JMP loop1_avx2

done_avx2:
	VZEROUPPER
	RET

// func copyAndAddBiasFloat32AVX512Asm(dst, src, bias unsafe.Pointer, n int)
TEXT ·copyAndAddBiasFloat32AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ bias+16(FP), DX
	MOVQ n+24(FP), CX

copy_loop64_512:
	CMPQ CX, $64
	JL copy_loop16_512
	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1
	VMOVUPS 128(SI), Z2
	VMOVUPS 192(SI), Z3
	VADDPS 0(DX), Z0, Z0
	VADDPS 64(DX), Z1, Z1
	VADDPS 128(DX), Z2, Z2
	VADDPS 192(DX), Z3, Z3
	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)
	VMOVUPS Z2, 128(DI)
	VMOVUPS Z3, 192(DI)
	ADDQ $256, DI
	ADDQ $256, SI
	ADDQ $256, DX
	SUBQ $64, CX
	JMP copy_loop64_512

copy_loop16_512:
	CMPQ CX, $16
	JL copy_loop1_512
	VMOVUPS 0(SI), Z0
	VADDPS 0(DX), Z0, Z0
	VMOVUPS Z0, 0(DI)
	ADDQ $64, DI
	ADDQ $64, SI
	ADDQ $64, DX
	SUBQ $16, CX
	JMP copy_loop16_512

copy_loop1_512:
	CMPQ CX, $0
	JLE copy_done_512
	VMOVSS 0(SI), X0
	VADDSS 0(DX), X0, X0
	VMOVSS X0, 0(DI)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, DX
	DECQ CX
	JMP copy_loop1_512

copy_done_512:
	VZEROUPPER
	RET

// func copyAndAddBiasFloat32AVX2Asm(dst, src, bias unsafe.Pointer, n int)
TEXT ·copyAndAddBiasFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ bias+16(FP), DX
	MOVQ n+24(FP), CX

copy_loop32_avx2:
	CMPQ CX, $32
	JL copy_loop8_avx2
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 64(SI), Y2
	VMOVUPS 96(SI), Y3
	VADDPS 0(DX), Y0, Y0
	VADDPS 32(DX), Y1, Y1
	VADDPS 64(DX), Y2, Y2
	VADDPS 96(DX), Y3, Y3
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)
	VMOVUPS Y2, 64(DI)
	VMOVUPS Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	ADDQ $128, DX
	SUBQ $32, CX
	JMP copy_loop32_avx2

copy_loop8_avx2:
	CMPQ CX, $8
	JL copy_loop1_avx2
	VMOVUPS 0(SI), Y0
	VADDPS 0(DX), Y0, Y0
	VMOVUPS Y0, 0(DI)
	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, DX
	SUBQ $8, CX
	JMP copy_loop8_avx2

copy_loop1_avx2:
	CMPQ CX, $0
	JLE copy_done_avx2
	VMOVSS 0(SI), X0
	VADDSS 0(DX), X0, X0
	VMOVSS X0, 0(DI)
	ADDQ $4, DI
	ADDQ $4, SI
	ADDQ $4, DX
	DECQ CX
	JMP copy_loop1_avx2

copy_done_avx2:
	VZEROUPPER
	RET

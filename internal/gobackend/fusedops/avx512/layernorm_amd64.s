// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// func layerNormFloat32AVX512(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)
//
// Arguments:
//   in:      0(FP)
//   out:     8(FP)
//   gamma:   16(FP)
//   beta:    24(FP)
//   outer:   32(FP)
//   normSize:40(FP)
//   eps:     48(FP)
TEXT ·layerNormFloat32AVX512(SB), NOSPLIT, $0-52
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ gamma+16(FP), R8
	MOVQ beta+24(FP), R9
	MOVQ outerSize+32(FP), R10
	MOVQ normSize+40(FP), R11
	MOVSS epsilon+48(FP), X15 // X15 = epsilon

	// Precompute 1.0 / float32(normSize) in X14
	CVTSL2SS R11, X14
	MOVSS ·float32One(SB), X13
	DIVSS X14, X13
	MOVSS X13, X14 // X14 = 1.0 / float32(normSize)

	TESTQ R10, R10
	JLE done
	TESTQ R11, R11
	JLE done

outer_loop:
	// --- PASS 1: MEAN ---
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

	XORQ AX, AX
pass1_loop64:
	LEAQ 64(AX), DX
	CMPQ DX, R11
	JG pass1_loop16

	VMOVUPS (SI)(AX*4), Z4
	VMOVUPS 64(SI)(AX*4), Z5
	VMOVUPS 128(SI)(AX*4), Z6
	VMOVUPS 192(SI)(AX*4), Z7
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1
	VADDPS Z6, Z2, Z2
	VADDPS Z7, Z3, Z3
	ADDQ $64, AX
	JMP pass1_loop64

pass1_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass1_reduce

	VMOVUPS (SI)(AX*4), Z4
	VADDPS Z4, Z0, Z0
	ADDQ $16, AX
	JMP pass1_loop16

pass1_reduce:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0

	// Horizontal sum of Z0 to X0
	VEXTRACTF64X4 $1, Z0, Y1
	VADDPS Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

pass1_scalar:
	CMPQ AX, R11
	JGE pass1_done
	ADDSS (SI)(AX*4), X0
	INCQ AX
	JMP pass1_scalar

pass1_done:
	MULSS X14, X0
	VBROADCASTSS X0, Z12 // Z12 = mean

	// --- PASS 2: VARIANCE ---
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

	XORQ AX, AX
pass2_loop64:
	LEAQ 64(AX), DX
	CMPQ DX, R11
	JG pass2_loop16

	VMOVUPS (SI)(AX*4), Z4
	VMOVUPS 64(SI)(AX*4), Z5
	VMOVUPS 128(SI)(AX*4), Z6
	VMOVUPS 192(SI)(AX*4), Z7
	VSUBPS Z12, Z4, Z4
	VSUBPS Z12, Z5, Z5
	VSUBPS Z12, Z6, Z6
	VSUBPS Z12, Z7, Z7
	VFMADD231PS Z4, Z4, Z0
	VFMADD231PS Z5, Z5, Z1
	VFMADD231PS Z6, Z6, Z2
	VFMADD231PS Z7, Z7, Z3
	ADDQ $64, AX
	JMP pass2_loop64

pass2_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass2_reduce

	VMOVUPS (SI)(AX*4), Z4
	VSUBPS Z12, Z4, Z4
	VFMADD231PS Z4, Z4, Z0
	ADDQ $16, AX
	JMP pass2_loop16

pass2_reduce:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0

	VEXTRACTF64X4 $1, Z0, Y1
	VADDPS Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

pass2_scalar:
	CMPQ AX, R11
	JGE pass2_done
	MOVSS (SI)(AX*4), X1
	SUBSS X12, X1
	MULSS X1, X1
	ADDSS X1, X0
	INCQ AX
	JMP pass2_scalar

pass2_done:
	MULSS X14, X0
	ADDSS X15, X0
	SQRTSS X0, X0
	MOVSS ·float32One(SB), X1
	DIVSS X0, X1
	VBROADCASTSS X1, Z13 // Z13 = invStd

	// --- PASS 3: NORMALIZE + SCALE/BIAS ---
	TESTQ R8, R8
	JZ no_gamma
	TESTQ R9, R9
	JZ gamma_only

	// Both gamma & beta
	XORQ AX, AX
pass3_both_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass3_both_scalar

	VMOVUPS (SI)(AX*4), Z0
	VSUBPS Z12, Z0, Z0
	VMULPS Z13, Z0, Z0
	VMOVUPS (R8)(AX*4), Z4
	VMOVUPS (R9)(AX*4), Z5
	VFMADD213PS Z5, Z4, Z0
	VMOVUPS Z0, (DI)(AX*4)
	ADDQ $16, AX
	JMP pass3_both_loop16

pass3_both_scalar:
	CMPQ AX, R11
	JGE next_row
	MOVSS (SI)(AX*4), X0
	SUBSS X12, X0
	MULSS X13, X0
	MULSS (R8)(AX*4), X0
	ADDSS (R9)(AX*4), X0
	MOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP pass3_both_scalar

gamma_only:
	XORQ AX, AX
pass3_gamma_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass3_gamma_scalar

	VMOVUPS (SI)(AX*4), Z0
	VSUBPS Z12, Z0, Z0
	VMULPS Z13, Z0, Z0
	VMOVUPS (R8)(AX*4), Z4
	VMULPS Z4, Z0, Z0
	VMOVUPS Z0, (DI)(AX*4)
	ADDQ $16, AX
	JMP pass3_gamma_loop16

pass3_gamma_scalar:
	CMPQ AX, R11
	JGE next_row
	MOVSS (SI)(AX*4), X0
	SUBSS X12, X0
	MULSS X13, X0
	MULSS (R8)(AX*4), X0
	MOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP pass3_gamma_scalar

no_gamma:
	TESTQ R9, R9
	JZ no_affine

	// Beta only
	XORQ AX, AX
pass3_beta_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass3_beta_scalar

	VMOVUPS (SI)(AX*4), Z0
	VSUBPS Z12, Z0, Z0
	VMULPS Z13, Z0, Z0
	VMOVUPS (R9)(AX*4), Z5
	VADDPS Z5, Z0, Z0
	VMOVUPS Z0, (DI)(AX*4)
	ADDQ $16, AX
	JMP pass3_beta_loop16

pass3_beta_scalar:
	CMPQ AX, R11
	JGE next_row
	MOVSS (SI)(AX*4), X0
	SUBSS X12, X0
	MULSS X13, X0
	ADDSS (R9)(AX*4), X0
	MOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP pass3_beta_scalar

no_affine:
	XORQ AX, AX
pass3_none_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass3_none_scalar

	VMOVUPS (SI)(AX*4), Z0
	VSUBPS Z12, Z0, Z0
	VMULPS Z13, Z0, Z0
	VMOVUPS Z0, (DI)(AX*4)
	ADDQ $16, AX
	JMP pass3_none_loop16

pass3_none_scalar:
	CMPQ AX, R11
	JGE next_row
	MOVSS (SI)(AX*4), X0
	SUBSS X12, X0
	MULSS X13, X0
	MOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP pass3_none_scalar

next_row:
	MOVQ R11, AX
	SHLQ $2, AX
	ADDQ AX, SI
	ADDQ AX, DI
	DECQ R10
	JNZ outer_loop

done:
	VZEROUPPER
	RET

// func layerNormFloat64AVX512(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float64)
//
// Arguments:
//   in:      0(FP)
//   out:     8(FP)
//   gamma:   16(FP)
//   beta:    24(FP)
//   outer:   32(FP)
//   normSize:40(FP)
//   eps:     48(FP)
TEXT ·layerNormFloat64AVX512(SB), NOSPLIT, $0-56
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ gamma+16(FP), R8
	MOVQ beta+24(FP), R9
	MOVQ outerSize+32(FP), R10
	MOVQ normSize+40(FP), R11
	MOVSD epsilon+48(FP), X15 // X15 = epsilon

	// Precompute 1.0 / float64(normSize) in X14
	CVTSQ2SD R11, X14
	MOVSD ·float64One(SB), X13
	DIVSD X14, X13
	MOVSD X13, X14 // X14 = 1.0 / float64(normSize)

	TESTQ R10, R10
	JLE done64
	TESTQ R11, R11
	JLE done64

outer_loop64:
	// --- PASS 1: MEAN ---
	VXORPD Z0, Z0, Z0
	VXORPD Z1, Z1, Z1
	VXORPD Z2, Z2, Z2
	VXORPD Z3, Z3, Z3

	XORQ AX, AX
pass1_64_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG pass1_64_loop8

	VMOVUPD (SI)(AX*8), Z4
	VMOVUPD 64(SI)(AX*8), Z5
	VMOVUPD 128(SI)(AX*8), Z6
	VMOVUPD 192(SI)(AX*8), Z7
	VADDPD Z4, Z0, Z0
	VADDPD Z5, Z1, Z1
	VADDPD Z6, Z2, Z2
	VADDPD Z7, Z3, Z3
	ADDQ $32, AX
	JMP pass1_64_loop32

pass1_64_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass1_64_reduce

	VMOVUPD (SI)(AX*8), Z4
	VADDPD Z4, Z0, Z0
	ADDQ $8, AX
	JMP pass1_64_loop8

pass1_64_reduce:
	VADDPD Z1, Z0, Z0
	VADDPD Z3, Z2, Z2
	VADDPD Z2, Z0, Z0

	VEXTRACTF64X4 $1, Z0, Y1
	VADDPD Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

pass1_64_scalar:
	CMPQ AX, R11
	JGE pass1_64_done
	ADDSD (SI)(AX*8), X0
	INCQ AX
	JMP pass1_64_scalar

pass1_64_done:
	MULSD X14, X0
	VBROADCASTSD X0, Z12 // Z12 = mean

	// --- PASS 2: VARIANCE ---
	VXORPD Z0, Z0, Z0
	VXORPD Z1, Z1, Z1
	VXORPD Z2, Z2, Z2
	VXORPD Z3, Z3, Z3

	XORQ AX, AX
pass2_64_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG pass2_64_loop8

	VMOVUPD (SI)(AX*8), Z4
	VMOVUPD 64(SI)(AX*8), Z5
	VMOVUPD 128(SI)(AX*8), Z6
	VMOVUPD 192(SI)(AX*8), Z7
	VSUBPD Z12, Z4, Z4
	VSUBPD Z12, Z5, Z5
	VSUBPD Z12, Z6, Z6
	VSUBPD Z12, Z7, Z7
	VFMADD231PD Z4, Z4, Z0
	VFMADD231PD Z5, Z5, Z1
	VFMADD231PD Z6, Z6, Z2
	VFMADD231PD Z7, Z7, Z3
	ADDQ $32, AX
	JMP pass2_64_loop32

pass2_64_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass2_64_reduce

	VMOVUPD (SI)(AX*8), Z4
	VSUBPD Z12, Z4, Z4
	VFMADD231PD Z4, Z4, Z0
	ADDQ $8, AX
	JMP pass2_64_loop8

pass2_64_reduce:
	VADDPD Z1, Z0, Z0
	VADDPD Z3, Z2, Z2
	VADDPD Z2, Z0, Z0

	VEXTRACTF64X4 $1, Z0, Y1
	VADDPD Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

pass2_64_scalar:
	CMPQ AX, R11
	JGE pass2_64_done
	MOVSD (SI)(AX*8), X1
	SUBSD X12, X1
	MULSD X1, X1
	ADDSD X1, X0
	INCQ AX
	JMP pass2_64_scalar

pass2_64_done:
	MULSD X14, X0
	ADDSD X15, X0
	SQRTSD X0, X0
	MOVSD ·float64One(SB), X1
	DIVSD X0, X1
	VBROADCASTSD X1, Z13 // Z13 = invStd

	// --- PASS 3: NORMALIZE + SCALE/BIAS ---
	TESTQ R8, R8
	JZ no_gamma64
	TESTQ R9, R9
	JZ gamma_only64

	// Both gamma & beta
	XORQ AX, AX
pass3_64_both_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_64_both_scalar

	VMOVUPD (SI)(AX*8), Z0
	VSUBPD Z12, Z0, Z0
	VMULPD Z13, Z0, Z0
	VMOVUPD (R8)(AX*8), Z4
	VMOVUPD (R9)(AX*8), Z5
	VFMADD213PD Z5, Z4, Z0
	VMOVUPD Z0, (DI)(AX*8)
	ADDQ $8, AX
	JMP pass3_64_both_loop8

pass3_64_both_scalar:
	CMPQ AX, R11
	JGE next_row64
	MOVSD (SI)(AX*8), X0
	SUBSD X12, X0
	MULSD X13, X0
	MULSD (R8)(AX*8), X0
	ADDSD (R9)(AX*8), X0
	MOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP pass3_64_both_scalar

gamma_only64:
	XORQ AX, AX
pass3_64_gamma_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_64_gamma_scalar

	VMOVUPD (SI)(AX*8), Z0
	VSUBPD Z12, Z0, Z0
	VMULPD Z13, Z0, Z0
	VMOVUPD (R8)(AX*8), Z4
	VMULPD Z4, Z0, Z0
	VMOVUPD Z0, (DI)(AX*8)
	ADDQ $8, AX
	JMP pass3_64_gamma_loop8

pass3_64_gamma_scalar:
	CMPQ AX, R11
	JGE next_row64
	MOVSD (SI)(AX*8), X0
	SUBSD X12, X0
	MULSD X13, X0
	MULSD (R8)(AX*8), X0
	MOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP pass3_64_gamma_scalar

no_gamma64:
	TESTQ R9, R9
	JZ no_affine64

	// Beta only
	XORQ AX, AX
pass3_64_beta_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_64_beta_scalar

	VMOVUPD (SI)(AX*8), Z0
	VSUBPD Z12, Z0, Z0
	VMULPD Z13, Z0, Z0
	VMOVUPD (R9)(AX*8), Z5
	VADDPD Z5, Z0, Z0
	VMOVUPD Z0, (DI)(AX*8)
	ADDQ $8, AX
	JMP pass3_64_beta_loop8

pass3_64_beta_scalar:
	CMPQ AX, R11
	JGE next_row64
	MOVSD (SI)(AX*8), X0
	SUBSD X12, X0
	MULSD X13, X0
	ADDSD (R9)(AX*8), X0
	MOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP pass3_64_beta_scalar

no_affine64:
	XORQ AX, AX
pass3_64_none_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_64_none_scalar

	VMOVUPD (SI)(AX*8), Z0
	VSUBPD Z12, Z0, Z0
	VMULPD Z13, Z0, Z0
	VMOVUPD Z0, (DI)(AX*8)
	ADDQ $8, AX
	JMP pass3_64_none_loop8

pass3_64_none_scalar:
	CMPQ AX, R11
	JGE next_row64
	MOVSD (SI)(AX*8), X0
	SUBSD X12, X0
	MULSD X13, X0
	MOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP pass3_64_none_scalar

next_row64:
	MOVQ R11, AX
	SHLQ $3, AX
	ADDQ AX, SI
	ADDQ AX, DI
	DECQ R10
	JNZ outer_loop64

done64:
	VZEROUPPER
	RET

GLOBL ·float32One(SB), RODATA|NOPTR, $4
DATA ·float32One(SB)/4, $0x3f800000 // 1.0f

GLOBL ·float64One(SB), RODATA|NOPTR, $8
DATA ·float64One(SB)/8, $0x3ff0000000000000 // 1.0

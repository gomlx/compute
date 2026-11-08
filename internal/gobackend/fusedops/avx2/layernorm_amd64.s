// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// func layerNormFloat32AVX2(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)
//
// Arguments:
//   in:      0(FP)
//   out:     8(FP)
//   gamma:   16(FP)
//   beta:    24(FP)
//   outer:   32(FP)
//   normSize:40(FP)
//   eps:     48(FP)
TEXT ·layerNormFloat32AVX2(SB), NOSPLIT, $0-52
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ gamma+16(FP), R8
	MOVQ beta+24(FP), R9
	MOVQ outerSize+32(FP), R10
	MOVQ normSize+40(FP), R11
	VMOVSS epsilon+48(FP), X15 // X15 = epsilon

	// Precompute 1.0 / float32(normSize) in X14
	VCVTSI2SSL R11, X14, X14
	VMOVSS ·float32One(SB), X13
	VDIVSS X14, X13, X14 // X14 = 1.0 / float32(normSize)

	TESTQ R10, R10
	JLE done
	TESTQ R11, R11
	JLE done

outer_loop:
	// --- PASS 1: MEAN ---
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	XORQ AX, AX // index
pass1_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG pass1_loop8

	VMOVUPS (SI)(AX*4), Y4
	VMOVUPS 32(SI)(AX*4), Y5
	VMOVUPS 64(SI)(AX*4), Y6
	VMOVUPS 96(SI)(AX*4), Y7
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1
	VADDPS Y6, Y2, Y2
	VADDPS Y7, Y3, Y3
	ADDQ $32, AX
	JMP pass1_loop32

pass1_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass1_reduce

	VMOVUPS (SI)(AX*4), Y4
	VADDPS Y4, Y0, Y0
	ADDQ $8, AX
	JMP pass1_loop8

pass1_reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

	// Horizontal sum of Y0 to X0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0xEE, X0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0x01, X0, X1
	VADDSS X1, X0, X0 // X0[0] has vector sum

pass1_scalar:
	CMPQ AX, R11
	JGE pass1_done
	VADDSS (SI)(AX*4), X0, X0
	INCQ AX
	JMP pass1_scalar

pass1_done:
	VMULSS X14, X0, X0
	VBROADCASTSS X0, Y12 // Y12 = mean

	// --- PASS 2: VARIANCE ---
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	XORQ AX, AX
pass2_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG pass2_loop8

	VMOVUPS (SI)(AX*4), Y4
	VMOVUPS 32(SI)(AX*4), Y5
	VMOVUPS 64(SI)(AX*4), Y6
	VMOVUPS 96(SI)(AX*4), Y7
	VSUBPS Y12, Y4, Y4
	VSUBPS Y12, Y5, Y5
	VSUBPS Y12, Y6, Y6
	VSUBPS Y12, Y7, Y7
	VFMADD231PS Y4, Y4, Y0
	VFMADD231PS Y5, Y5, Y1
	VFMADD231PS Y6, Y6, Y2
	VFMADD231PS Y7, Y7, Y3
	ADDQ $32, AX
	JMP pass2_loop32

pass2_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass2_reduce

	VMOVUPS (SI)(AX*4), Y4
	VSUBPS Y12, Y4, Y4
	VFMADD231PS Y4, Y4, Y0
	ADDQ $8, AX
	JMP pass2_loop8

pass2_reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0xEE, X0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0x01, X0, X1
	VADDSS X1, X0, X0

pass2_scalar:
	CMPQ AX, R11
	JGE pass2_done
	VMOVSS (SI)(AX*4), X1
	VSUBSS X12, X1, X1
	VMULSS X1, X1, X1
	VADDSS X1, X0, X0
	INCQ AX
	JMP pass2_scalar

pass2_done:
	VMULSS X14, X0, X0
	VADDSS X15, X0, X0
	VSQRTSS X0, X0, X0
	VMOVSS ·float32One(SB), X1
	VDIVSS X0, X1, X1
	VBROADCASTSS X1, Y13 // Y13 = invStd

	// --- PASS 3: NORMALIZE + SCALE/BIAS ---
	TESTQ R8, R8
	JZ no_gamma
	TESTQ R9, R9
	JZ gamma_only

	// Both gamma (R8) and beta (R9)
	XORQ AX, AX
pass3_both_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG pass3_both_loop8

	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS 32(SI)(AX*4), Y1
	VMOVUPS 64(SI)(AX*4), Y2
	VMOVUPS 96(SI)(AX*4), Y3
	VSUBPS Y12, Y0, Y0
	VSUBPS Y12, Y1, Y1
	VSUBPS Y12, Y2, Y2
	VSUBPS Y12, Y3, Y3
	VMULPS Y13, Y0, Y0
	VMULPS Y13, Y1, Y1
	VMULPS Y13, Y2, Y2
	VMULPS Y13, Y3, Y3
	VMOVUPS (R8)(AX*4), Y4
	VMOVUPS 32(R8)(AX*4), Y5
	VMOVUPS 64(R8)(AX*4), Y6
	VMOVUPS 96(R8)(AX*4), Y7
	VMOVUPS (R9)(AX*4), Y8
	VMOVUPS 32(R9)(AX*4), Y9
	VMOVUPS 64(R9)(AX*4), Y10
	VMOVUPS 96(R9)(AX*4), Y11
	VFMADD213PS Y8, Y4, Y0
	VFMADD213PS Y9, Y5, Y1
	VFMADD213PS Y10, Y6, Y2
	VFMADD213PS Y11, Y7, Y3
	VMOVUPS Y0, (DI)(AX*4)
	VMOVUPS Y1, 32(DI)(AX*4)
	VMOVUPS Y2, 64(DI)(AX*4)
	VMOVUPS Y3, 96(DI)(AX*4)
	ADDQ $32, AX
	JMP pass3_both_loop32

pass3_both_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_both_scalar

	VMOVUPS (SI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS (R8)(AX*4), Y4
	VMOVUPS (R9)(AX*4), Y5
	VFMADD213PS Y5, Y4, Y0 // Y0 = Y0 * Y4 + Y5
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP pass3_both_loop8

pass3_both_scalar:
	CMPQ AX, R11
	JGE next_row
	VMOVSS (SI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VMULSS (R8)(AX*4), X0, X0
	VADDSS (R9)(AX*4), X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP pass3_both_scalar

gamma_only:
	XORQ AX, AX
pass3_gamma_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_gamma_scalar

	VMOVUPS (SI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS (R8)(AX*4), Y4
	VMULPS Y4, Y0, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP pass3_gamma_loop8

pass3_gamma_scalar:
	CMPQ AX, R11
	JGE next_row
	VMOVSS (SI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VMULSS (R8)(AX*4), X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP pass3_gamma_scalar

no_gamma:
	TESTQ R9, R9
	JZ no_affine

	// Beta only
	XORQ AX, AX
pass3_beta_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_beta_scalar

	VMOVUPS (SI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS (R9)(AX*4), Y5
	VADDPS Y5, Y0, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP pass3_beta_loop8

pass3_beta_scalar:
	CMPQ AX, R11
	JGE next_row
	VMOVSS (SI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VADDSS (R9)(AX*4), X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP pass3_beta_scalar

no_affine:
	XORQ AX, AX
pass3_none_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG pass3_none_scalar

	VMOVUPS (SI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP pass3_none_loop8

pass3_none_scalar:
	CMPQ AX, R11
	JGE next_row
	VMOVSS (SI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VMOVSS X0, (DI)(AX*4)
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

// func layerNormFloat64AVX2(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float64)
//
// Arguments:
//   in:      0(FP)
//   out:     8(FP)
//   gamma:   16(FP)
//   beta:    24(FP)
//   outer:   32(FP)
//   normSize:40(FP)
//   eps:     48(FP)
TEXT ·layerNormFloat64AVX2(SB), NOSPLIT, $0-56
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ gamma+16(FP), R8
	MOVQ beta+24(FP), R9
	MOVQ outerSize+32(FP), R10
	MOVQ normSize+40(FP), R11
	VMOVSD epsilon+48(FP), X15 // X15 = epsilon

	// Precompute 1.0 / float64(normSize) in X14
	VCVTSI2SDQ R11, X14, X14
	VMOVSD ·float64One(SB), X13
	VDIVSD X14, X13, X14 // X14 = 1.0 / float64(normSize)

	TESTQ R10, R10
	JLE done64
	TESTQ R11, R11
	JLE done64

outer_loop64:
	// --- PASS 1: MEAN ---
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3

	XORQ AX, AX
pass1_64_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass1_64_loop4

	VMOVUPD (SI)(AX*8), Y4
	VMOVUPD 32(SI)(AX*8), Y5
	VMOVUPD 64(SI)(AX*8), Y6
	VMOVUPD 96(SI)(AX*8), Y7
	VADDPD Y4, Y0, Y0
	VADDPD Y5, Y1, Y1
	VADDPD Y6, Y2, Y2
	VADDPD Y7, Y3, Y3
	ADDQ $16, AX
	JMP pass1_64_loop16

pass1_64_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG pass1_64_reduce

	VMOVUPD (SI)(AX*8), Y4
	VADDPD Y4, Y0, Y0
	ADDQ $4, AX
	JMP pass1_64_loop4

pass1_64_reduce:
	VADDPD Y1, Y0, Y0
	VADDPD Y3, Y2, Y2
	VADDPD Y2, Y0, Y0

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

pass1_64_scalar:
	CMPQ AX, R11
	JGE pass1_64_done
	VADDSD (SI)(AX*8), X0, X0
	INCQ AX
	JMP pass1_64_scalar

pass1_64_done:
	VMULSD X14, X0, X0
	VBROADCASTSD X0, Y12 // Y12 = mean

	// --- PASS 2: VARIANCE ---
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3

	XORQ AX, AX
pass2_64_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG pass2_64_loop4

	VMOVUPD (SI)(AX*8), Y4
	VMOVUPD 32(SI)(AX*8), Y5
	VMOVUPD 64(SI)(AX*8), Y6
	VMOVUPD 96(SI)(AX*8), Y7
	VSUBPD Y12, Y4, Y4
	VSUBPD Y12, Y5, Y5
	VSUBPD Y12, Y6, Y6
	VSUBPD Y12, Y7, Y7
	VFMADD231PD Y4, Y4, Y0
	VFMADD231PD Y5, Y5, Y1
	VFMADD231PD Y6, Y6, Y2
	VFMADD231PD Y7, Y7, Y3
	ADDQ $16, AX
	JMP pass2_64_loop16

pass2_64_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG pass2_64_reduce

	VMOVUPD (SI)(AX*8), Y4
	VSUBPD Y12, Y4, Y4
	VFMADD231PD Y4, Y4, Y0
	ADDQ $4, AX
	JMP pass2_64_loop4

pass2_64_reduce:
	VADDPD Y1, Y0, Y0
	VADDPD Y3, Y2, Y2
	VADDPD Y2, Y0, Y0

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

pass2_64_scalar:
	CMPQ AX, R11
	JGE pass2_64_done
	VMOVSD (SI)(AX*8), X1
	VSUBSD X12, X1, X1
	VMULSD X1, X1, X1
	VADDSD X1, X0, X0
	INCQ AX
	JMP pass2_64_scalar

pass2_64_done:
	VMULSD X14, X0, X0
	VADDSD X15, X0, X0
	VSQRTSD X0, X0, X0
	VMOVSD ·float64One(SB), X1
	VDIVSD X0, X1, X1
	VBROADCASTSD X1, Y13 // Y13 = invStd

	// --- PASS 3: NORMALIZE + SCALE/BIAS ---
	TESTQ R8, R8
	JZ no_gamma64
	TESTQ R9, R9
	JZ gamma_only64

	// Both gamma & beta
	XORQ AX, AX
pass3_64_both_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG pass3_64_both_scalar

	VMOVUPD (SI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD (R8)(AX*8), Y4
	VMOVUPD (R9)(AX*8), Y5
	VFMADD213PD Y5, Y4, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP pass3_64_both_loop4

pass3_64_both_scalar:
	CMPQ AX, R11
	JGE next_row64
	VMOVSD (SI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VMULSD (R8)(AX*8), X0, X0
	VADDSD (R9)(AX*8), X0, X0
	VMOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP pass3_64_both_scalar

gamma_only64:
	XORQ AX, AX
pass3_64_gamma_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG pass3_64_gamma_scalar

	VMOVUPD (SI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD (R8)(AX*8), Y4
	VMULPD Y4, Y0, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP pass3_64_gamma_loop4

pass3_64_gamma_scalar:
	CMPQ AX, R11
	JGE next_row64
	VMOVSD (SI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VMULSD (R8)(AX*8), X0, X0
	VMOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP pass3_64_gamma_scalar

no_gamma64:
	TESTQ R9, R9
	JZ no_affine64

	// Beta only
	XORQ AX, AX
pass3_64_beta_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG pass3_64_beta_scalar

	VMOVUPD (SI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD (R9)(AX*8), Y5
	VADDPD Y5, Y0, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP pass3_64_beta_loop4

pass3_64_beta_scalar:
	CMPQ AX, R11
	JGE next_row64
	VMOVSD (SI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VADDSD (R9)(AX*8), X0, X0
	VMOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP pass3_64_beta_scalar

no_affine64:
	XORQ AX, AX
pass3_64_none_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG pass3_64_none_scalar

	VMOVUPD (SI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP pass3_64_none_loop4

pass3_64_none_scalar:
	CMPQ AX, R11
	JGE next_row64
	VMOVSD (SI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VMOVSD X0, (DI)(AX*8)
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

// func layerNormResidualFloat32AVX2(in, residual, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)
TEXT ·layerNormResidualFloat32AVX2(SB), NOSPLIT, $0-60
	MOVQ in+0(FP), SI
	MOVQ residual+8(FP), CX
	MOVQ out+16(FP), DI
	MOVQ gamma+24(FP), R8
	MOVQ beta+32(FP), R9
	MOVQ outerSize+40(FP), R10
	MOVQ normSize+48(FP), R11
	VMOVSS epsilon+56(FP), X15 // X15 = epsilon

	VCVTSI2SSL R11, X14, X14
	VMOVSS ·float32One(SB), X13
	VDIVSS X14, X13, X14 // X14 = 1.0 / float32(normSize)

	TESTQ R10, R10
	JLE res_done
	TESTQ R11, R11
	JLE res_done

res_outer_loop:
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	XORQ AX, AX
res_pass1_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG res_pass1_loop8

	VMOVUPS (SI)(AX*4), Y4
	VMOVUPS 32(SI)(AX*4), Y5
	VMOVUPS 64(SI)(AX*4), Y6
	VMOVUPS 96(SI)(AX*4), Y7
	VADDPS (CX)(AX*4), Y4, Y4
	VADDPS 32(CX)(AX*4), Y5, Y5
	VADDPS 64(CX)(AX*4), Y6, Y6
	VADDPS 96(CX)(AX*4), Y7, Y7
	VMOVUPS Y4, (DI)(AX*4)
	VMOVUPS Y5, 32(DI)(AX*4)
	VMOVUPS Y6, 64(DI)(AX*4)
	VMOVUPS Y7, 96(DI)(AX*4)
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1
	VADDPS Y6, Y2, Y2
	VADDPS Y7, Y3, Y3
	ADDQ $32, AX
	JMP res_pass1_loop32

res_pass1_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG res_pass1_reduce

	VMOVUPS (SI)(AX*4), Y4
	VADDPS (CX)(AX*4), Y4, Y4
	VMOVUPS Y4, (DI)(AX*4)
	VADDPS Y4, Y0, Y0
	ADDQ $8, AX
	JMP res_pass1_loop8

res_pass1_reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0xEE, X0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0x01, X0, X1
	VADDSS X1, X0, X0

res_pass1_scalar:
	CMPQ AX, R11
	JGE res_pass1_done
	VMOVSS (SI)(AX*4), X1
	VADDSS (CX)(AX*4), X1, X1
	VMOVSS X1, (DI)(AX*4)
	VADDSS X1, X0, X0
	INCQ AX
	JMP res_pass1_scalar

res_pass1_done:
	VMULSS X14, X0, X0
	VBROADCASTSS X0, Y12 // Y12 = mean

	// --- PASS 2: VARIANCE (from DI) ---
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	XORQ AX, AX
res_pass2_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG res_pass2_loop8

	VMOVUPS (DI)(AX*4), Y4
	VMOVUPS 32(DI)(AX*4), Y5
	VMOVUPS 64(DI)(AX*4), Y6
	VMOVUPS 96(DI)(AX*4), Y7
	VSUBPS Y12, Y4, Y4
	VSUBPS Y12, Y5, Y5
	VSUBPS Y12, Y6, Y6
	VSUBPS Y12, Y7, Y7
	VFMADD231PS Y4, Y4, Y0
	VFMADD231PS Y5, Y5, Y1
	VFMADD231PS Y6, Y6, Y2
	VFMADD231PS Y7, Y7, Y3
	ADDQ $32, AX
	JMP res_pass2_loop32

res_pass2_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG res_pass2_reduce

	VMOVUPS (DI)(AX*4), Y4
	VSUBPS Y12, Y4, Y4
	VFMADD231PS Y4, Y4, Y0
	ADDQ $8, AX
	JMP res_pass2_loop8

res_pass2_reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0xEE, X0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0x01, X0, X1
	VADDSS X1, X0, X0

res_pass2_scalar:
	CMPQ AX, R11
	JGE res_pass2_done
	VMOVSS (DI)(AX*4), X1
	VSUBSS X12, X1, X1
	VMULSS X1, X1, X1
	VADDSS X1, X0, X0
	INCQ AX
	JMP res_pass2_scalar

res_pass2_done:
	VMULSS X14, X0, X0
	VADDSS X15, X0, X0
	VSQRTSS X0, X0, X0
	VMOVSS ·float32One(SB), X1
	VDIVSS X0, X1, X1
	VBROADCASTSS X1, Y13 // Y13 = invStd

	// --- PASS 3: NORMALIZE + SCALE/BIAS (in-place on DI) ---
	TESTQ R8, R8
	JZ res_no_gamma
	TESTQ R9, R9
	JZ res_gamma_only

	XORQ AX, AX
res_pass3_both_loop32:
	LEAQ 32(AX), DX
	CMPQ DX, R11
	JG res_pass3_both_loop8

	VMOVUPS (DI)(AX*4), Y0
	VMOVUPS 32(DI)(AX*4), Y1
	VMOVUPS 64(DI)(AX*4), Y2
	VMOVUPS 96(DI)(AX*4), Y3
	VSUBPS Y12, Y0, Y0
	VSUBPS Y12, Y1, Y1
	VSUBPS Y12, Y2, Y2
	VSUBPS Y12, Y3, Y3
	VMULPS Y13, Y0, Y0
	VMULPS Y13, Y1, Y1
	VMULPS Y13, Y2, Y2
	VMULPS Y13, Y3, Y3
	VMOVUPS (R8)(AX*4), Y4
	VMOVUPS 32(R8)(AX*4), Y5
	VMOVUPS 64(R8)(AX*4), Y6
	VMOVUPS 96(R8)(AX*4), Y7
	VMOVUPS (R9)(AX*4), Y8
	VMOVUPS 32(R9)(AX*4), Y9
	VMOVUPS 64(R9)(AX*4), Y10
	VMOVUPS 96(R9)(AX*4), Y11
	VFMADD213PS Y8, Y4, Y0
	VFMADD213PS Y9, Y5, Y1
	VFMADD213PS Y10, Y6, Y2
	VFMADD213PS Y11, Y7, Y3
	VMOVUPS Y0, (DI)(AX*4)
	VMOVUPS Y1, 32(DI)(AX*4)
	VMOVUPS Y2, 64(DI)(AX*4)
	VMOVUPS Y3, 96(DI)(AX*4)
	ADDQ $32, AX
	JMP res_pass3_both_loop32

res_pass3_both_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG res_pass3_both_scalar

	VMOVUPS (DI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS (R8)(AX*4), Y4
	VMOVUPS (R9)(AX*4), Y5
	VFMADD213PS Y5, Y4, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP res_pass3_both_loop8

res_pass3_both_scalar:
	CMPQ AX, R11
	JGE res_next_row
	VMOVSS (DI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VMULSS (R8)(AX*4), X0, X0
	VADDSS (R9)(AX*4), X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP res_pass3_both_scalar

res_gamma_only:
	XORQ AX, AX
res_pass3_gamma_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG res_pass3_gamma_scalar

	VMOVUPS (DI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS (R8)(AX*4), Y4
	VMULPS Y4, Y0, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP res_pass3_gamma_loop8

res_pass3_gamma_scalar:
	CMPQ AX, R11
	JGE res_next_row
	VMOVSS (DI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VMULSS (R8)(AX*4), X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP res_pass3_gamma_scalar

res_no_gamma:
	TESTQ R9, R9
	JZ res_no_affine

	XORQ AX, AX
res_pass3_beta_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG res_pass3_beta_scalar

	VMOVUPS (DI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS (R9)(AX*4), Y5
	VADDPS Y5, Y0, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP res_pass3_beta_loop8

res_pass3_beta_scalar:
	CMPQ AX, R11
	JGE res_next_row
	VMOVSS (DI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VADDSS (R9)(AX*4), X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP res_pass3_beta_scalar

res_no_affine:
	XORQ AX, AX
res_pass3_none_loop8:
	LEAQ 8(AX), DX
	CMPQ DX, R11
	JG res_pass3_none_scalar

	VMOVUPS (DI)(AX*4), Y0
	VSUBPS Y12, Y0, Y0
	VMULPS Y13, Y0, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ $8, AX
	JMP res_pass3_none_loop8

res_pass3_none_scalar:
	CMPQ AX, R11
	JGE res_next_row
	VMOVSS (DI)(AX*4), X0
	VSUBSS X12, X0, X0
	VMULSS X13, X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ AX
	JMP res_pass3_none_scalar

res_next_row:
	MOVQ R11, AX
	SHLQ $2, AX
	ADDQ AX, SI
	ADDQ AX, CX
	ADDQ AX, DI
	DECQ R10
	JNZ res_outer_loop

res_done:
	VZEROUPPER
	RET

// func layerNormResidualFloat64AVX2(in, residual, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float64)
TEXT ·layerNormResidualFloat64AVX2(SB), NOSPLIT, $0-64
	MOVQ in+0(FP), SI
	MOVQ residual+8(FP), CX
	MOVQ out+16(FP), DI
	MOVQ gamma+24(FP), R8
	MOVQ beta+32(FP), R9
	MOVQ outerSize+40(FP), R10
	MOVQ normSize+48(FP), R11
	VMOVSD epsilon+56(FP), X15 // X15 = epsilon

	VCVTSI2SDQ R11, X14, X14
	VMOVSD ·float64One(SB), X13
	VDIVSD X14, X13, X14 // X14 = 1.0 / float64(normSize)

	TESTQ R10, R10
	JLE res_done64
	TESTQ R11, R11
	JLE res_done64

res_outer_loop64:
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3

	XORQ AX, AX
res_pass1_64_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG res_pass1_64_loop4

	VMOVUPD (SI)(AX*8), Y4
	VMOVUPD 32(SI)(AX*8), Y5
	VMOVUPD 64(SI)(AX*8), Y6
	VMOVUPD 96(SI)(AX*8), Y7
	VADDPD (CX)(AX*8), Y4, Y4
	VADDPD 32(CX)(AX*8), Y5, Y5
	VADDPD 64(CX)(AX*8), Y6, Y6
	VADDPD 96(CX)(AX*8), Y7, Y7
	VMOVUPD Y4, (DI)(AX*8)
	VMOVUPD Y5, 32(DI)(AX*8)
	VMOVUPD Y6, 64(DI)(AX*8)
	VMOVUPD Y7, 96(DI)(AX*8)
	VADDPD Y4, Y0, Y0
	VADDPD Y5, Y1, Y1
	VADDPD Y6, Y2, Y2
	VADDPD Y7, Y3, Y3
	ADDQ $16, AX
	JMP res_pass1_64_loop16

res_pass1_64_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG res_pass1_64_reduce

	VMOVUPD (SI)(AX*8), Y4
	VADDPD (CX)(AX*8), Y4, Y4
	VMOVUPD Y4, (DI)(AX*8)
	VADDPD Y4, Y0, Y0
	ADDQ $4, AX
	JMP res_pass1_64_loop4

res_pass1_64_reduce:
	VADDPD Y1, Y0, Y0
	VADDPD Y3, Y2, Y2
	VADDPD Y2, Y0, Y0

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

res_pass1_64_scalar:
	CMPQ AX, R11
	JGE res_pass1_64_done
	VMOVSD (SI)(AX*8), X1
	VADDSD (CX)(AX*8), X1, X1
	VMOVSD X1, (DI)(AX*8)
	VADDSD X1, X0, X0
	INCQ AX
	JMP res_pass1_64_scalar

res_pass1_64_done:
	VMULSD X14, X0, X0
	VBROADCASTSD X0, Y12 // Y12 = mean

	// --- PASS 2: VARIANCE (from DI) ---
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3

	XORQ AX, AX
res_pass2_64_loop16:
	LEAQ 16(AX), DX
	CMPQ DX, R11
	JG res_pass2_64_loop4

	VMOVUPD (DI)(AX*8), Y4
	VMOVUPD 32(DI)(AX*8), Y5
	VMOVUPD 64(DI)(AX*8), Y6
	VMOVUPD 96(DI)(AX*8), Y7
	VSUBPD Y12, Y4, Y4
	VSUBPD Y12, Y5, Y5
	VSUBPD Y12, Y6, Y6
	VSUBPD Y12, Y7, Y7
	VFMADD231PD Y4, Y4, Y0
	VFMADD231PD Y5, Y5, Y1
	VFMADD231PD Y6, Y6, Y2
	VFMADD231PD Y7, Y7, Y3
	ADDQ $16, AX
	JMP res_pass2_64_loop16

res_pass2_64_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG res_pass2_64_reduce

	VMOVUPD (DI)(AX*8), Y4
	VSUBPD Y12, Y4, Y4
	VFMADD231PD Y4, Y4, Y0
	ADDQ $4, AX
	JMP res_pass2_64_loop4

res_pass2_64_reduce:
	VADDPD Y1, Y0, Y0
	VADDPD Y3, Y2, Y2
	VADDPD Y2, Y0, Y0

	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

res_pass2_64_scalar:
	CMPQ AX, R11
	JGE res_pass2_64_done
	VMOVSD (DI)(AX*8), X1
	VSUBSD X12, X1, X1
	VMULSD X1, X1, X1
	VADDSD X1, X0, X0
	INCQ AX
	JMP res_pass2_64_scalar

res_pass2_64_done:
	VMULSD X14, X0, X0
	VADDSD X15, X0, X0
	VSQRTSD X0, X0, X0
	VMOVSD ·float64One(SB), X1
	VDIVSD X0, X1, X1
	VBROADCASTSD X1, Y13 // Y13 = invStd

	// --- PASS 3: NORMALIZE + SCALE/BIAS (in-place on DI) ---
	TESTQ R8, R8
	JZ res_no_gamma64
	TESTQ R9, R9
	JZ res_gamma_only64

	XORQ AX, AX
res_pass3_64_both_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG res_pass3_64_both_scalar

	VMOVUPD (DI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD (R8)(AX*8), Y4
	VMOVUPD (R9)(AX*8), Y5
	VFMADD213PD Y5, Y4, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP res_pass3_64_both_loop4

res_pass3_64_both_scalar:
	CMPQ AX, R11
	JGE res_next_row64
	VMOVSD (DI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VMULSD (R8)(AX*8), X0, X0
	VADDSD (R9)(AX*8), X0, X0
	VMOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP res_pass3_64_both_scalar

res_gamma_only64:
	XORQ AX, AX
res_pass3_64_gamma_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG res_pass3_64_gamma_scalar

	VMOVUPD (DI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD (R8)(AX*8), Y4
	VMULPD Y4, Y0, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP res_pass3_64_gamma_loop4

res_pass3_64_gamma_scalar:
	CMPQ AX, R11
	JGE res_next_row64
	VMOVSD (DI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VMULSD (R8)(AX*8), X0, X0
	VMOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP res_pass3_64_gamma_scalar

res_no_gamma64:
	TESTQ R9, R9
	JZ res_no_affine64

	XORQ AX, AX
res_pass3_64_beta_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG res_pass3_64_beta_scalar

	VMOVUPD (DI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD (R9)(AX*8), Y5
	VADDPD Y5, Y0, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP res_pass3_64_beta_loop4

res_pass3_64_beta_scalar:
	CMPQ AX, R11
	JGE res_next_row64
	VMOVSD (DI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VADDSD (R9)(AX*8), X0, X0
	VMOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP res_pass3_64_beta_scalar

res_no_affine64:
	XORQ AX, AX
res_pass3_64_none_loop4:
	LEAQ 4(AX), DX
	CMPQ DX, R11
	JG res_pass3_64_none_scalar

	VMOVUPD (DI)(AX*8), Y0
	VSUBPD Y12, Y0, Y0
	VMULPD Y13, Y0, Y0
	VMOVUPD Y0, (DI)(AX*8)
	ADDQ $4, AX
	JMP res_pass3_64_none_loop4

res_pass3_64_none_scalar:
	CMPQ AX, R11
	JGE res_next_row64
	VMOVSD (DI)(AX*8), X0
	VSUBSD X12, X0, X0
	VMULSD X13, X0, X0
	VMOVSD X0, (DI)(AX*8)
	INCQ AX
	JMP res_pass3_64_none_scalar

res_next_row64:
	MOVQ R11, AX
	SHLQ $3, AX
	ADDQ AX, SI
	ADDQ AX, CX
	ADDQ AX, DI
	DECQ R10
	JNZ res_outer_loop64

res_done64:
	VZEROUPPER
	RET

GLOBL ·float32One(SB), RODATA|NOPTR, $4
DATA ·float32One(SB)/4, $0x3f800000 // 1.0f

GLOBL ·float64One(SB), RODATA|NOPTR, $8
DATA ·float64One(SB)/8, $0x3ff0000000000000 // 1.0

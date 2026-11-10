// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// ===========================================================================
// swigluFloat32AVX2Asm: SwiGLU forward for float32 using AVX2
//   out[m*hiddenDim + j] = (gate * val) / (1 + exp(-gate))
//   where gate = in[m*2*hiddenDim + j], val = in[m*2*hiddenDim + hiddenDim + j].
//   Uses 2-vector (16 float32) unrolling, magic-number 2^n scaling, and
//   Estrin's degree-7 polynomial tree with fused denominator FMA.
//   vecWidth must be a positive multiple of 8 (vecWidth <= hiddenDim).
//   Reuses ·siluAVX2Consts(SB) defined in silu_amd64.s.
// ===========================================================================
// func swigluFloat32AVX2Asm(in, out unsafe.Pointer, numRows, hiddenDim, vecWidth int)
TEXT ·swigluFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ numRows+16(FP), R11
	MOVQ hiddenDim+24(FP), R12
	MOVQ vecWidth+32(FP), R13

	TESTQ R11, R11
	JLE swiglu_avx2_done
	CMPQ R13, $8
	JL swiglu_avx2_done

	SHLQ $2, R12               // R12 = hiddenBytes = hiddenDim * 4 (out row stride & val offset)
	MOVQ R12, R14
	SHLQ $1, R14               // R14 = inRowBytes = 2 * hiddenDim * 4 (in row stride)

	LEAQ ·siluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14       // Y14: vP2 = 0.5 (0x3f000000)
	VMOVUPS 192(CX), Y15       // Y15: vOne / vBias127 = 1.0 (0x3f800000)

swiglu_avx2_row_loop:
	MOVQ SI, R8                // R8 = gatePtr
	LEAQ (SI)(R12*1), R9       // R9 = valPtr = gatePtr + hiddenBytes
	MOVQ DI, R10               // R10 = outPtr
	MOVQ R13, DX               // DX = remaining vecWidth elements

	CMPQ DX, $16
	JL swiglu_avx2_vec8_check

swiglu_avx2_loop16:
	VMOVUPS 0(R8), Y0
	VMOVUPS 32(R8), Y1

	// u = clamp(-gate, vMinLog, vMaxLog)
	VXORPS 224(CX), Y0, Y4
	VXORPS 224(CX), Y1, Y5
	VMAXPS 256(CX), Y4, Y4
	VMAXPS 256(CX), Y5, Y5
	VMINPS 288(CX), Y4, Y4
	VMINPS 288(CX), Y5, Y5

	// Range reduction: zRound = u * vLog2E + vMagic, zFloor = zRound - vMagic
	VMOVUPS 352(CX), Y12
	VMOVAPS Y12, Y6
	VMOVAPS Y12, Y7
	VFMADD231PS 320(CX), Y4, Y6
	VFMADD231PS 320(CX), Y5, Y7
	VSUBPS Y12, Y6, Y8
	VSUBPS Y12, Y7, Y9

	// g = u - zFloor * vLn2Hi - zFloor * vLn2Lo
	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 384(CX), Y9, Y5
	VFNMADD231PS 416(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y9, Y5

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Y6, Y6
	VPSLLD $23, Y7, Y7
	VPADDD Y15, Y6, Y6
	VPADDD Y15, Y7, Y7

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Y4, Y4, Y2
	VMULPS Y5, Y5, Y3
	VMOVUPS 32(CX), Y8
	VMOVAPS Y8, Y9
	VFMADD231PS 0(CX), Y4, Y8
	VFMADD231PS 0(CX), Y5, Y9
	VMOVUPS 96(CX), Y10
	VMOVAPS Y10, Y11
	VFMADD231PS 64(CX), Y4, Y10
	VFMADD231PS 64(CX), Y5, Y11
	VMOVAPS Y14, Y12
	VMOVAPS Y14, Y13
	VFMADD231PS 128(CX), Y4, Y12
	VFMADD231PS 128(CX), Y5, Y13
	VADDPS Y15, Y4, Y4
	VADDPS Y15, Y5, Y5
	VFMADD231PS Y2, Y8, Y10
	VFMADD231PS Y3, Y9, Y11
	VFMADD231PS Y2, Y12, Y4
	VFMADD231PS Y3, Y13, Y5
	VMULPS Y2, Y2, Y2
	VMULPS Y3, Y3, Y3
	VFMADD231PS Y2, Y10, Y4
	VFMADD231PS Y3, Y11, Y5

	// denom = 1.0 + scale * expPoly
	VMOVAPS Y15, Y8
	VMOVAPS Y15, Y9
	VFMADD231PS Y6, Y4, Y8
	VFMADD231PS Y7, Y5, Y9

	// out = (gate * val) / denom
	VMULPS 0(R9), Y0, Y0
	VMULPS 32(R9), Y1, Y1
	VDIVPS Y8, Y0, Y0
	VDIVPS Y9, Y1, Y1

	VMOVUPS Y0, 0(R10)
	VMOVUPS Y1, 32(R10)

	ADDQ $64, R8
	ADDQ $64, R9
	ADDQ $64, R10
	SUBQ $16, DX
	CMPQ DX, $16
	JGE swiglu_avx2_loop16

swiglu_avx2_vec8_check:
	CMPQ DX, $8
	JL swiglu_avx2_next_row

	VMOVUPS 0(R8), Y0
	VXORPS 224(CX), Y0, Y4
	VMAXPS 256(CX), Y4, Y4
	VMINPS 288(CX), Y4, Y4

	VMOVUPS 352(CX), Y12
	VMOVAPS Y12, Y6
	VFMADD231PS 320(CX), Y4, Y6
	VSUBPS Y12, Y6, Y8

	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y8, Y4

	VPSLLD $23, Y6, Y6
	VPADDD Y15, Y6, Y6

	VMULPS Y4, Y4, Y2
	VMOVUPS 32(CX), Y8
	VFMADD231PS 0(CX), Y4, Y8
	VMOVUPS 96(CX), Y10
	VFMADD231PS 64(CX), Y4, Y10
	VMOVAPS Y14, Y12
	VFMADD231PS 128(CX), Y4, Y12
	VADDPS Y15, Y4, Y4
	VFMADD231PS Y2, Y8, Y10
	VFMADD231PS Y2, Y12, Y4
	VMULPS Y2, Y2, Y2
	VFMADD231PS Y2, Y10, Y4

	VMOVAPS Y15, Y8
	VFMADD231PS Y6, Y4, Y8

	VMULPS 0(R9), Y0, Y0
	VDIVPS Y8, Y0, Y0
	VMOVUPS Y0, 0(R10)

swiglu_avx2_next_row:
	ADDQ R14, SI
	ADDQ R12, DI
	DECQ R11
	JNZ swiglu_avx2_row_loop

swiglu_avx2_done:
	VZEROUPPER
	RET

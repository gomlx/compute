// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// ===========================================================================
// swigluFloat32AVX512Asm: SwiGLU forward for float32 using AVX-512
//   out[m*hiddenDim + j] = (gate * val) / (1 + exp(-gate))
//   where gate = in[m*2*hiddenDim + j], val = in[m*2*hiddenDim + hiddenDim + j].
//   Uses 2-vector (32 float32) unrolling, magic-number 2^n scaling, and
//   Estrin's degree-7 polynomial tree with fused denominator FMA.
//   vecWidth must be a positive multiple of 16 (vecWidth <= hiddenDim).
//   Reuses ·siluAVX512Consts(SB) defined in silu_amd64.s.
// ===========================================================================
// func swigluFloat32AVX512Asm(in, out unsafe.Pointer, numRows, hiddenDim, vecWidth int)
TEXT ·swigluFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ numRows+16(FP), R11
	MOVQ hiddenDim+24(FP), R12
	MOVQ vecWidth+32(FP), R13

	TESTQ R11, R11
	JLE swiglu_avx512_done
	CMPQ R13, $16
	JL swiglu_avx512_done

	SHLQ $2, R12               // R12 = hiddenBytes = hiddenDim * 4 (out row stride & val offset)
	MOVQ R12, R14
	SHLQ $1, R14               // R14 = inRowBytes = 2 * hiddenDim * 4 (in row stride)

	// Pin all 14 constants in Z18..Z31:
	VBROADCASTSS ·siluAVX512Consts+0(SB), Z18  // Z18: vP7
	VBROADCASTSS ·siluAVX512Consts+4(SB), Z19  // Z19: vP6
	VBROADCASTSS ·siluAVX512Consts+8(SB), Z20  // Z20: vP5
	VBROADCASTSS ·siluAVX512Consts+12(SB), Z21 // Z21: vP4
	VBROADCASTSS ·siluAVX512Consts+16(SB), Z22 // Z22: vP3
	VBROADCASTSS ·siluAVX512Consts+20(SB), Z23 // Z23: vP2 = 0.5
	VBROADCASTSS ·siluAVX512Consts+24(SB), Z24 // Z24: vOne / vBias127 = 1.0
	VBROADCASTSS ·siluAVX512Consts+28(SB), Z25 // Z25: vSignMask
	VBROADCASTSS ·siluAVX512Consts+32(SB), Z26 // Z26: vMinLog
	VBROADCASTSS ·siluAVX512Consts+36(SB), Z27 // Z27: vMaxLog
	VBROADCASTSS ·siluAVX512Consts+40(SB), Z28 // Z28: vLog2E
	VBROADCASTSS ·siluAVX512Consts+44(SB), Z29 // Z29: vMagic
	VBROADCASTSS ·siluAVX512Consts+48(SB), Z30 // Z30: vLn2Hi
	VBROADCASTSS ·siluAVX512Consts+52(SB), Z31 // Z31: vLn2Lo

swiglu_avx512_row_loop:
	MOVQ SI, R8                // R8 = gatePtr
	LEAQ (SI)(R12*1), R9       // R9 = valPtr = gatePtr + hiddenBytes
	MOVQ DI, R10               // R10 = outPtr
	MOVQ R13, DX               // DX = remaining vecWidth elements

	CMPQ DX, $32
	JL swiglu_avx512_vec16_check

swiglu_avx512_loop32:
	VMOVUPS 0(R8), Z0
	VMOVUPS 64(R8), Z1

	// u = clamp(-gate, vMinLog, vMaxLog)
	VPXORD Z25, Z0, Z2
	VPXORD Z25, Z1, Z3
	VMAXPS Z26, Z2, Z2
	VMAXPS Z26, Z3, Z3
	VMINPS Z27, Z2, Z2
	VMINPS Z27, Z3, Z3

	// Range reduction: zRound = u * vLog2E + vMagic, zFloor = zRound - vMagic
	VMOVAPS Z29, Z4
	VMOVAPS Z29, Z5
	VFMADD231PS Z2, Z28, Z4
	VFMADD231PS Z3, Z28, Z5
	VSUBPS Z29, Z4, Z6
	VSUBPS Z29, Z5, Z7

	// g = u - zFloor * vLn2Hi - zFloor * vLn2Lo
	VFNMADD231PS Z6, Z30, Z2
	VFNMADD231PS Z7, Z30, Z3
	VFNMADD231PS Z6, Z31, Z2
	VFNMADD231PS Z7, Z31, Z3

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Z4, Z4
	VPSLLD $23, Z5, Z5
	VPADDD Z24, Z4, Z4
	VPADDD Z24, Z5, Z5

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Z2, Z2, Z6
	VMULPS Z3, Z3, Z7
	VMOVAPS Z19, Z8
	VMOVAPS Z19, Z9
	VFMADD231PS Z2, Z18, Z8
	VFMADD231PS Z3, Z18, Z9
	VMOVAPS Z21, Z10
	VMOVAPS Z21, Z11
	VFMADD231PS Z2, Z20, Z10
	VFMADD231PS Z3, Z20, Z11
	VMOVAPS Z23, Z12
	VMOVAPS Z23, Z13
	VFMADD231PS Z2, Z22, Z12
	VFMADD231PS Z3, Z22, Z13
	VADDPS Z24, Z2, Z14
	VADDPS Z24, Z3, Z15
	VFMADD231PS Z6, Z8, Z10
	VFMADD231PS Z7, Z9, Z11
	VFMADD231PS Z6, Z12, Z14
	VFMADD231PS Z7, Z13, Z15
	VMULPS Z6, Z6, Z6
	VMULPS Z7, Z7, Z7
	VFMADD231PS Z6, Z10, Z14
	VFMADD231PS Z7, Z11, Z15

	// denom = 1.0 + scale * expPoly
	VMOVAPS Z24, Z8
	VMOVAPS Z24, Z9
	VFMADD231PS Z4, Z14, Z8
	VFMADD231PS Z5, Z15, Z9

	// out = (gate * val) / denom
	VMULPS 0(R9), Z0, Z0
	VMULPS 64(R9), Z1, Z1
	VDIVPS Z8, Z0, Z0
	VDIVPS Z9, Z1, Z1

	VMOVUPS Z0, 0(R10)
	VMOVUPS Z1, 64(R10)

	ADDQ $128, R8
	ADDQ $128, R9
	ADDQ $128, R10
	SUBQ $32, DX
	CMPQ DX, $32
	JGE swiglu_avx512_loop32

swiglu_avx512_vec16_check:
	CMPQ DX, $16
	JL swiglu_avx512_next_row

	VMOVUPS 0(R8), Z0
	VPXORD Z25, Z0, Z2
	VMAXPS Z26, Z2, Z2
	VMINPS Z27, Z2, Z2

	VMOVAPS Z29, Z4
	VFMADD231PS Z2, Z28, Z4
	VSUBPS Z29, Z4, Z6
	VFNMADD231PS Z6, Z30, Z2
	VFNMADD231PS Z6, Z31, Z2

	VPSLLD $23, Z4, Z4
	VPADDD Z24, Z4, Z4

	VMULPS Z2, Z2, Z6
	VMOVAPS Z19, Z8
	VFMADD231PS Z2, Z18, Z8
	VMOVAPS Z21, Z10
	VFMADD231PS Z2, Z20, Z10
	VMOVAPS Z23, Z12
	VFMADD231PS Z2, Z22, Z12
	VADDPS Z24, Z2, Z14
	VFMADD231PS Z6, Z8, Z10
	VFMADD231PS Z6, Z12, Z14
	VMULPS Z6, Z6, Z6
	VFMADD231PS Z6, Z10, Z14

	VMOVAPS Z24, Z8
	VFMADD231PS Z4, Z14, Z8

	VMULPS 0(R9), Z0, Z0
	VDIVPS Z8, Z0, Z0
	VMOVUPS Z0, 0(R10)

swiglu_avx512_next_row:
	ADDQ R14, SI
	ADDQ R12, DI
	DECQ R11
	JNZ swiglu_avx512_row_loop

swiglu_avx512_done:
	VZEROUPPER
	RET

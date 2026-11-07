// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// Read-only constants table for AVX-512 SDPA kernels.
// Offsets (in bytes):
//   0:  -Inf         (0xFF800000)
//   4:  vMinLog      (-88.0296936, 0xC2B00F34)
//   8:  vMaxLog      ( 88.0296936, 0x42B00F34)
//   12: vLog2E       (1.4426950408889634, 0x3FB8AA3B)
//   16: vMagic       (12582912.0, 0x4B400000)
//   20: vLn2Hi       (0.693359375, 0x3F318000)
//   24: vLn2Lo       (-2.12194440e-4, 0xB95E8083)
//   28: vP7          (1.9875691500e-4, 0x39506967)
//   32: vP6          (1.3981999507e-3, 0x3AB743CE)
//   36: vP5          (8.3334519073e-3, 0x3C088895)
//   40: vP4          (4.1665795894e-2, 0x3D2AA9C1)
//   44: vP3          (1.6666665459e-1, 0x3E2AAA9F)
//   48: vP2          (5.0000001201e-1, 0x3F000000)
//   52: vOne/Bias127 (1.0f / 127<<23, 0x3F800000)
DATA ·sdpaAVX512Consts+0(SB)/4, $0xFF800000
DATA ·sdpaAVX512Consts+4(SB)/4, $0xC2B00F34
DATA ·sdpaAVX512Consts+8(SB)/4, $0x42B00F34
DATA ·sdpaAVX512Consts+12(SB)/4, $0x3FB8AA3B
DATA ·sdpaAVX512Consts+16(SB)/4, $0x4B400000
DATA ·sdpaAVX512Consts+20(SB)/4, $0x3F318000
DATA ·sdpaAVX512Consts+24(SB)/4, $0xB95E8083
DATA ·sdpaAVX512Consts+28(SB)/4, $0x39506967
DATA ·sdpaAVX512Consts+32(SB)/4, $0x3AB743CE
DATA ·sdpaAVX512Consts+36(SB)/4, $0x3C088895
DATA ·sdpaAVX512Consts+40(SB)/4, $0x3D2AA9C1
DATA ·sdpaAVX512Consts+44(SB)/4, $0x3E2AAA9F
DATA ·sdpaAVX512Consts+48(SB)/4, $0x3F000000
DATA ·sdpaAVX512Consts+52(SB)/4, $0x3F800000
GLOBL ·sdpaAVX512Consts(SB), RODATA|NOPTR, $56

// func sdpaQueriesDim32NoMaskAVX512Asm(
//     qPtr, kPtr, vPtr, outPtr, scoresPtr unsafe.Pointer,
//     qByteStride, kvByteStride, numQueries, kvLen, qStart, causalInt int,
//     scale float32,
// )
//
// Computes Scaled Dot-Product Attention for headDim=32 across numQueries consecutive
// queries when no boolean/additive mask or bias slice is active.
// scoresPtr must point to a buffer of at least (kvLen + 16) float32s.
//
// Pinned registers across the entire query loop:
//   Z17: -Inf (0xFF800000)
//   Z18: scale
//   Z19: vMinLog (-88.0296936)
//   Z20: vMaxLog ( 88.0296936)
//   Z21: vLog2E
//   Z22: vMagic (12582912.0)
//   Z23: vLn2Hi
//   Z24: vLn2Lo
//   Z25: vP7
//   Z26: vP6
//   Z27: vP5
//   Z28: vP4
//   Z29: vP3
//   Z30: vP2
//   Z31: vOne (1.0f) / vBias127 (127 << 23)
TEXT ·sdpaQueriesDim32NoMaskAVX512Asm(SB), NOSPLIT, $0-92
	MOVQ qPtr+0(FP), SI
	MOVQ kPtr+8(FP), R8
	MOVQ vPtr+16(FP), R9
	MOVQ outPtr+24(FP), DI
	MOVQ scoresPtr+32(FP), R10
	MOVQ kvByteStride+48(FP), R11
	MOVQ numQueries+56(FP), R13
	MOVQ qStart+72(FP), R14

	TESTQ R13, R13
	JLE sdpa_done

	// R12 = 3 * kvByteStride
	LEAQ (R11)(R11*2), R12

	// Load pinned constants into Z17..Z31 once
	LEAQ ·sdpaAVX512Consts(SB), AX
	VBROADCASTSS 0(AX), Z17
	VBROADCASTSS scale+88(FP), Z18
	VBROADCASTSS 4(AX), Z19
	VBROADCASTSS 8(AX), Z20
	VBROADCASTSS 12(AX), Z21
	VBROADCASTSS 16(AX), Z22
	VBROADCASTSS 20(AX), Z23
	VBROADCASTSS 24(AX), Z24
	VBROADCASTSS 28(AX), Z25
	VBROADCASTSS 32(AX), Z26
	VBROADCASTSS 36(AX), Z27
	VBROADCASTSS 40(AX), Z28
	VBROADCASTSS 44(AX), Z29
	VBROADCASTSS 48(AX), Z30
	VBROADCASTSS 52(AX), Z31

query_loop:
	// Determine kvLenUnmasked in R15 = kvLen (or min(kvLen, qIdx+1) if causal)
	MOVQ kvLen+64(FP), R15
	MOVQ causalInt+80(FP), AX
	TESTQ AX, AX
	JZ check_kv_len
	LEAQ 1(R14), AX
	CMPQ AX, R15
	CMOVQLT AX, R15

check_kv_len:
	TESTQ R15, R15
	JG valid_query

zero_query_out:
	// kvLenUnmasked <= 0 or all scores -Inf: zero output[0:32] and advance
	VXORPS Z0, Z0, Z0
	VMOVUPS Z0, 0(DI)
	VMOVUPS Z0, 64(DI)
	JMP next_query

valid_query:
	// Load query[0:32] and pre-scale by Z18 (scale)
	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1
	VMULPS Z18, Z0, Z0
	VMULPS Z18, Z1, Z1

	// --- STAGE 1: Q @ K^T (4 keys per iteration) ---
	VMOVAPS X17, X10 // X10 = [-Inf, -Inf, -Inf, -Inf] running max
	MOVQ R8, AX      // AX = kPtr cursor
	MOVQ R10, BX     // BX = scoresPtr cursor
	MOVQ R15, CX     // CX = remaining keys

stage1_loop4:
	CMPQ CX, $4
	JL stage1_reduce_max4

	// Load 4 keys (each 32 float32s = 2 ZMMs)
	VMOVUPS 0(AX), Z2
	VMOVUPS 64(AX), Z3
	VMOVUPS 0(AX)(R11*1), Z4
	VMOVUPS 64(AX)(R11*1), Z5
	VMOVUPS 0(AX)(R11*2), Z6
	VMOVUPS 64(AX)(R11*2), Z7
	VMOVUPS 0(AX)(R12*1), Z8
	VMOVUPS 64(AX)(R12*1), Z9

	VMULPS Z0, Z2, Z2
	VFMADD231PS Z1, Z3, Z2
	VMULPS Z0, Z4, Z4
	VFMADD231PS Z1, Z5, Z4
	VMULPS Z0, Z6, Z6
	VFMADD231PS Z1, Z7, Z6
	VMULPS Z0, Z8, Z8
	VFMADD231PS Z1, Z9, Z8

	// Fold 512-bit Z2, Z4, Z6, Z8 -> 256-bit Y2, Y4, Y6, Y8
	VEXTRACTF32X8 $1, Z2, Y3
	VEXTRACTF32X8 $1, Z4, Y5
	VEXTRACTF32X8 $1, Z6, Y7
	VEXTRACTF32X8 $1, Z8, Y9
	VADDPS Y3, Y2, Y2
	VADDPS Y5, Y4, Y4
	VADDPS Y7, Y6, Y6
	VADDPS Y9, Y8, Y8

	// Fold 256-bit Y2, Y4, Y6, Y8 -> 128-bit X2, X4, X6, X8
	VEXTRACTF128 $1, Y2, X3
	VEXTRACTF128 $1, Y4, X5
	VEXTRACTF128 $1, Y6, X7
	VEXTRACTF128 $1, Y8, X9
	VADDPS X3, X2, X2
	VADDPS X5, X4, X4
	VADDPS X7, X6, X6
	VADDPS X9, X8, X8

	// Transpose-and-add 4x4 X2, X4, X6, X8 -> X2 = [dot0, dot1, dot2, dot3]
	VUNPCKLPS X4, X2, X3
	VUNPCKHPS X4, X2, X2
	VUNPCKLPS X8, X6, X5
	VUNPCKHPS X8, X6, X6
	VADDPS X2, X3, X2
	VADDPS X6, X5, X5
	VSHUFPS $0x44, X5, X2, X3
	VSHUFPS $0xEE, X5, X2, X2
	VADDPS X2, X3, X2

	VMOVUPS X2, (BX)
	VMAXPS X2, X10, X10

	LEAQ (AX)(R11*4), AX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP stage1_loop4

stage1_reduce_max4:
	// Reduce 4 lanes of X10 to scalar X10[0]
	VPERMILPS $0xEE, X10, X11
	VMAXPS X11, X10, X10
	VPERMILPS $0x01, X10, X11
	VMAXSS X11, X10, X10

stage1_rem:
	TESTQ CX, CX
	JLE stage1_pad

	VMOVUPS 0(AX), Z2
	VMOVUPS 64(AX), Z3
	VMULPS Z0, Z2, Z2
	VFMADD231PS Z1, Z3, Z2

	VEXTRACTF32X8 $1, Z2, Y3
	VADDPS Y3, Y2, Y2
	VEXTRACTF128 $1, Y2, X3
	VADDPS X3, X2, X2
	VPERMILPS $0xEE, X2, X3
	VADDPS X3, X2, X2
	VPERMILPS $0x01, X2, X3
	VADDSS X3, X2, X2

	VMOVSS X2, (BX)
	VMAXSS X2, X10, X10

	ADDQ R11, AX
	ADDQ $4, BX
	DECQ CX
	JMP stage1_rem

stage1_pad:
	// Guard against all scores being -Inf or NaN
	VUCOMISS X17, X10
	JBE zero_query_out
	JP zero_query_out

	// Pad 16 float32s after scoresScratch[kvLenUnmasked] with -Inf so Stage 2
	// can process full 16-wide ZMM vectors without a scalar remainder loop.
	VMOVUPS Z17, (BX)

	// --- STAGE 2: SOFTMAX exp(scores - rowMax) ---
	VBROADCASTSS X10, Z10 // Z10 = vRowMax
	VXORPS Z11, Z11, Z11  // Z11 = vSum
	MOVQ R10, BX          // BX = scoresPtr cursor
	MOVQ R15, CX          // CX = kvLenUnmasked

stage2_loop16:
	VMOVUPS (BX), Z2
	VSUBPS Z10, Z2, Z2
	VMAXPS Z19, Z2, Z2
	VMINPS Z20, Z2, Z2

	// Range reduction: zRound = xClamped * log2e + magic
	VMOVAPS Z22, Z3
	VFMADD231PS Z2, Z21, Z3
	VSUBPS Z22, Z3, Z4
	VPSLLD $23, Z3, Z3
	VPADDD Z31, Z3, Z3 // scale = (zInt + 127) << 23 (exact 0.0f when x <= vMinLog)

	// Cody-Waite remainder g = xClamped - zFloor*ln2Hi - zFloor*ln2Lo
	VFNMADD231PS Z4, Z23, Z2
	VFNMADD231PS Z4, Z24, Z2

	// Estrin degree-7 polynomial for exp(g)
	VMULPS Z2, Z2, Z4
	VMOVAPS Z26, Z5
	VFMADD231PS Z2, Z25, Z5
	VMOVAPS Z28, Z6
	VFMADD231PS Z2, Z27, Z6
	VMOVAPS Z30, Z7
	VFMADD231PS Z2, Z29, Z7
	VADDPS Z31, Z2, Z8
	VMULPS Z4, Z4, Z9
	VFMADD213PS Z6, Z4, Z5
	VFMADD213PS Z8, Z4, Z7
	VFMADD213PS Z7, Z9, Z5
	VMULPS Z3, Z5, Z2

	VMOVUPS Z2, (BX)
	VADDPS Z2, Z11, Z11

	ADDQ $64, BX
	SUBQ $16, CX
	JG stage2_loop16

	// Horizontal sum of Z11 -> scalar X11[0]
	VEXTRACTF32X8 $1, Z11, Y12
	VADDPS Y12, Y11, Y11
	VEXTRACTF128 $1, Y11, X12
	VADDPS X12, X11, X11
	VPERMILPS $0xEE, X11, X12
	VADDPS X12, X11, X11
	VPERMILPS $0x01, X11, X12
	VADDSS X12, X11, X11

	// Issue VDIVSS early so its latency is hidden by Stage 3 value accumulation
	VDIVSS X11, X31, X11

	// --- STAGE 3: VALUE ACCUMULATION (P @ V, 4 keys per iteration) ---
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7

	MOVQ R9, AX  // AX = vPtr cursor
	MOVQ R10, BX // BX = scoresPtr cursor
	MOVQ R15, CX // CX = kvLenUnmasked

stage3_loop4:
	CMPQ CX, $4
	JL stage3_reduce

	VBROADCASTSS 0(BX), Z8
	VBROADCASTSS 4(BX), Z9
	VBROADCASTSS 8(BX), Z12
	VBROADCASTSS 12(BX), Z13

	VFMADD231PS 0(AX), Z8, Z0
	VFMADD231PS 64(AX), Z8, Z1
	VFMADD231PS 0(AX)(R11*1), Z9, Z2
	VFMADD231PS 64(AX)(R11*1), Z9, Z3
	VFMADD231PS 0(AX)(R11*2), Z12, Z4
	VFMADD231PS 64(AX)(R11*2), Z12, Z5
	VFMADD231PS 0(AX)(R12*1), Z13, Z6
	VFMADD231PS 64(AX)(R12*1), Z13, Z7

	LEAQ (AX)(R11*4), AX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP stage3_loop4

stage3_reduce:
	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z6, Z4, Z4
	VADDPS Z7, Z5, Z5
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1

stage3_rem:
	TESTQ CX, CX
	JLE stage3_store

	VBROADCASTSS 0(BX), Z8
	VFMADD231PS 0(AX), Z8, Z0
	VFMADD231PS 64(AX), Z8, Z1

	ADDQ R11, AX
	ADDQ $4, BX
	DECQ CX
	JMP stage3_rem

stage3_store:
	VBROADCASTSS X11, Z11
	VMULPS Z11, Z0, Z0
	VMULPS Z11, Z1, Z1
	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)

next_query:
	MOVQ qByteStride+40(FP), AX
	ADDQ AX, SI
	ADDQ AX, DI
	INCQ R14
	DECQ R13
	JG query_loop

sdpa_done:
	VZEROUPPER
	RET

// func sdpaSoftmaxAndValueAccumDim32AVX512Asm(
//     vPtr, outPtr, scoresPtr unsafe.Pointer,
//     kvByteStride, kvLenUnmasked int,
//     rowMax float32,
// )
//
// Executes Stage 2 (Softmax exp + sum) and Stage 3 (4x unrolled P @ V accumulation)
// for headDim=32 when Stage 1 was computed separately (e.g. with boolean/additive masks or biases).
// scoresPtr must have capacity for at least kvLenUnmasked + 16 float32s.
TEXT ·sdpaSoftmaxAndValueAccumDim32AVX512Asm(SB), NOSPLIT, $0-44
	MOVQ vPtr+0(FP), R9
	MOVQ outPtr+8(FP), DI
	MOVQ scoresPtr+16(FP), R10
	MOVQ kvByteStride+24(FP), R11
	MOVQ kvLenUnmasked+32(FP), R15
	VBROADCASTSS rowMax+40(FP), Z10

	TESTQ R15, R15
	JLE accum_done

	LEAQ (R11)(R11*2), R12

	// Load constants into Z17, Z19..Z31
	LEAQ ·sdpaAVX512Consts(SB), AX
	VBROADCASTSS 0(AX), Z17
	VBROADCASTSS 4(AX), Z19
	VBROADCASTSS 8(AX), Z20
	VBROADCASTSS 12(AX), Z21
	VBROADCASTSS 16(AX), Z22
	VBROADCASTSS 20(AX), Z23
	VBROADCASTSS 24(AX), Z24
	VBROADCASTSS 28(AX), Z25
	VBROADCASTSS 32(AX), Z26
	VBROADCASTSS 36(AX), Z27
	VBROADCASTSS 40(AX), Z28
	VBROADCASTSS 44(AX), Z29
	VBROADCASTSS 48(AX), Z30
	VBROADCASTSS 52(AX), Z31

	// Pad 16 float32s at scoresScratch[kvLenUnmasked] with -Inf
	LEAQ (R10)(R15*4), BX
	VMOVUPS Z17, (BX)

	VXORPS Z11, Z11, Z11
	MOVQ R10, BX
	MOVQ R15, CX

accum_stage2_loop16:
	VMOVUPS (BX), Z2
	VSUBPS Z10, Z2, Z2
	VMAXPS Z19, Z2, Z2
	VMINPS Z20, Z2, Z2

	VMOVAPS Z22, Z3
	VFMADD231PS Z2, Z21, Z3
	VSUBPS Z22, Z3, Z4
	VPSLLD $23, Z3, Z3
	VPADDD Z31, Z3, Z3

	VFNMADD231PS Z4, Z23, Z2
	VFNMADD231PS Z4, Z24, Z2

	VMULPS Z2, Z2, Z4
	VMOVAPS Z26, Z5
	VFMADD231PS Z2, Z25, Z5
	VMOVAPS Z28, Z6
	VFMADD231PS Z2, Z27, Z6
	VMOVAPS Z30, Z7
	VFMADD231PS Z2, Z29, Z7
	VADDPS Z31, Z2, Z8
	VMULPS Z4, Z4, Z9
	VFMADD213PS Z6, Z4, Z5
	VFMADD213PS Z8, Z4, Z7
	VFMADD213PS Z7, Z9, Z5
	VMULPS Z3, Z5, Z2

	VMOVUPS Z2, (BX)
	VADDPS Z2, Z11, Z11

	ADDQ $64, BX
	SUBQ $16, CX
	JG accum_stage2_loop16

	VEXTRACTF32X8 $1, Z11, Y12
	VADDPS Y12, Y11, Y11
	VEXTRACTF128 $1, Y11, X12
	VADDPS X12, X11, X11
	VPERMILPS $0xEE, X11, X12
	VADDPS X12, X11, X11
	VPERMILPS $0x01, X11, X12
	VADDSS X12, X11, X11

	VDIVSS X11, X31, X11

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7

	MOVQ R9, AX
	MOVQ R10, BX
	MOVQ R15, CX

accum_stage3_loop4:
	CMPQ CX, $4
	JL accum_stage3_reduce

	VBROADCASTSS 0(BX), Z8
	VBROADCASTSS 4(BX), Z9
	VBROADCASTSS 8(BX), Z12
	VBROADCASTSS 12(BX), Z13

	VFMADD231PS 0(AX), Z8, Z0
	VFMADD231PS 64(AX), Z8, Z1
	VFMADD231PS 0(AX)(R11*1), Z9, Z2
	VFMADD231PS 64(AX)(R11*1), Z9, Z3
	VFMADD231PS 0(AX)(R11*2), Z12, Z4
	VFMADD231PS 64(AX)(R11*2), Z12, Z5
	VFMADD231PS 0(AX)(R12*1), Z13, Z6
	VFMADD231PS 64(AX)(R12*1), Z13, Z7

	LEAQ (AX)(R11*4), AX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP accum_stage3_loop4

accum_stage3_reduce:
	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z6, Z4, Z4
	VADDPS Z7, Z5, Z5
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1

accum_stage3_rem:
	TESTQ CX, CX
	JLE accum_stage3_store

	VBROADCASTSS 0(BX), Z8
	VFMADD231PS 0(AX), Z8, Z0
	VFMADD231PS 64(AX), Z8, Z1

	ADDQ R11, AX
	ADDQ $4, BX
	DECQ CX
	JMP accum_stage3_rem

accum_stage3_store:
	VBROADCASTSS X11, Z11
	VMULPS Z11, Z0, Z0
	VMULPS Z11, Z1, Z1
	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)

accum_done:
	VZEROUPPER
	RET

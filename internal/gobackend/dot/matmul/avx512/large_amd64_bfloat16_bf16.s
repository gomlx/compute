// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

#include "textflag.h"

// VDPBF16PS Zsrc2, Zsrc1, Zdst  (Intel: vdpbf16ps zmm_dst, zmm_src1, zmm_src2)
// Computes 16 FP32 lanes:
//   dst[j] += bf16(src1[2j])*bf16(src2[2j]) + bf16(src1[2j+1])*bf16(src2[2j+1])
// Encoding: EVEX.512.F3.0F38.W0 52 /r
#define VDPBF16PS(src2, src1, dst) \
	BYTE $0x62; \
	BYTE $(((1^(((dst)>>3)&1))<<7) | ((1^(((src2)>>4)&1))<<6) | ((1^(((src2)>>3)&1))<<5) | ((1^(((dst)>>4)&1))<<4) | 0x02); \
	BYTE $(((15-((src1)&15))<<3) | 0x06); \
	BYTE $(0x40 | ((1^(((src1)>>4)&1))<<3)); \
	BYTE $0x52; \
	BYTE $(0xC0 | (((dst)&7)<<3) | ((src2)&7))

// func avx512LargeKernelBFloat16BF16Asm(
//     packedLHS, packedRHS []bfloat16.BFloat16,
//     packedOutput []float32,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int,
//     accumulate bool)
//
// Native AVX512_BF16 (VDPBF16PS) 8 rows x 32 cols GEMM microkernel.
// Expects K2-interleaved packed panels:
//   - packedLHS: [ceil(M/8), kPairs, 8, 2] bfloat16 (32 bytes per K-pair per 8-row strip)
//   - packedRHS: [ceil(N/32), kPairs, 32, 2] bfloat16 (128 bytes per K-pair per 32-col strip)
// where kPairs = (contractingLen + 1) / 2 and any odd trailing K is zero-padded in both panels.
// Iterates RHS strips outer and LHS strips inner to keep packedLHS + 1 RHS strip resident in L1D.
TEXT ·avx512LargeKernelBFloat16BF16Asm(SB), NOSPLIT, $0-120
	MOVQ packedLHS_base+0(FP), R8        // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9       // R9 = rhsBasePtr
	MOVQ packedOutput_base+48(FP), R10   // R10 = outBasePtr
	MOVQ rhsPanelCols+80(FP), R11        // R11 = outputStride (cols)
	MOVQ contractingLen+88(FP), R12      // R12 = contractingLen (K)
	MOVQ lhsActiveRows+96(FP), R13       // R13 = lhsActiveRows (M)
	MOVQ rhsActiveCols+104(FP), R14      // R14 = rhsActiveCols (N)
	MOVB accumulate+112(FP), R15         // R15 = accumulate (0 = overwrite, 1 = add)

	SHLQ $2, R11                         // R11 = outputStride in bytes (float32 = 4 bytes)

	// R12 = kPairs = (contractingLen + 1) >> 1 (if contractingLen <= 0, kPairs = 0).
	TESTQ R12, R12
	JLE kpairs_zero_bf16
	INCQ R12
	SHRQ $1, R12
	JMP start_rhs_bf16

kpairs_zero_bf16:
	XORQ R12, R12

start_rhs_bf16:
	XORQ BX, BX                          // BX = rhsColIdx = 0

loop_rhs_vdpbf16:
	CMPQ BX, R14
	JGE done_vdpbf16

	XORQ AX, AX                          // AX = lhsRowIdx = 0

loop_lhs_vdpbf16:
	CMPQ AX, R13
	JGE next_rhs_vdpbf16

	// 1. Zero all 16 accumulators (Z0..Z15)
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7
	VXORPS Z8, Z8, Z8
	VXORPS Z9, Z9, Z9
	VXORPS Z10, Z10, Z10
	VXORPS Z11, Z11, Z11
	VXORPS Z12, Z12, Z12
	VXORPS Z13, Z13, Z13
	VXORPS Z14, Z14, Z14
	VXORPS Z15, Z15, Z15

	// 2. Compute lhsPtr and rhsPtr for K2-interleaved strips:
	// LHS strip offset: (AX / 8) * kPairs * 32 = AX * kPairs * 4 bytes.
	MOVQ AX, SI
	IMULQ R12, SI
	SHLQ $2, SI
	ADDQ R8, SI

	// RHS strip offset: (BX / 32) * kPairs * 128 = BX * kPairs * 4 bytes.
	MOVQ BX, DI
	IMULQ R12, DI
	SHLQ $2, DI
	ADDQ R9, DI

	// 3. K-pair loop with 2-stage ping-pong pipeline (2 K-pairs = 4 K-steps per iter)
	MOVQ R12, CX                         // CX = kPairs
	TESTQ CX, CX
	JLE store_output_vdpbf16

	SHRQ $1, CX                          // CX = doublePairs = kPairs / 2
	JZ k_odd_pair_vdpbf16

	// Prime Buffer A for K-pair 0 outside the loop:
	VMOVDQU32 (DI), Z16                  // RHS cols 0..15 (16 K-pairs of BF16)
	VMOVDQU32 64(DI), Z17                // RHS cols 16..31 (16 K-pairs of BF16)
	VPBROADCASTD (SI), Z18               // LHS row 0 K-pair
	VPBROADCASTD 4(SI), Z19              // LHS row 1 K-pair
	VPBROADCASTD 8(SI), Z20              // LHS row 2 K-pair
	VPBROADCASTD 12(SI), Z21             // LHS row 3 K-pair
	VPBROADCASTD 16(SI), Z22             // LHS row 4 K-pair
	VPBROADCASTD 20(SI), Z23             // LHS row 5 K-pair
	VPBROADCASTD 24(SI), Z24             // LHS row 6 K-pair
	VPBROADCASTD 28(SI), Z25             // LHS row 7 K-pair

	ADDQ $32, SI
	ADDQ $128, DI

	DECQ CX
	JZ k_last_pair_vdpbf16

	PCALIGN $64
k_pair_loop_vdpbf16:
	// Step A: VDPBF16PS using Buffer A (Z16, Z17) while loading Buffer B (Z26, Z27, Z18..Z25)
	VDPBF16PS(16, 18, 0)
	VMOVDQU32 (DI), Z26
	VDPBF16PS(17, 18, 1)
	VPBROADCASTD (SI), Z18

	VDPBF16PS(16, 19, 2)
	VMOVDQU32 64(DI), Z27
	VDPBF16PS(17, 19, 3)
	VPBROADCASTD 4(SI), Z19

	VDPBF16PS(16, 20, 4)
	VDPBF16PS(17, 20, 5)
	VPBROADCASTD 8(SI), Z20

	VDPBF16PS(16, 21, 6)
	VDPBF16PS(17, 21, 7)
	VPBROADCASTD 12(SI), Z21

	VDPBF16PS(16, 22, 8)
	VDPBF16PS(17, 22, 9)
	VPBROADCASTD 16(SI), Z22

	VDPBF16PS(16, 23, 10)
	VDPBF16PS(17, 23, 11)
	VPBROADCASTD 20(SI), Z23

	VDPBF16PS(16, 24, 12)
	VDPBF16PS(17, 24, 13)
	VPBROADCASTD 24(SI), Z24

	VDPBF16PS(16, 25, 14)
	VDPBF16PS(17, 25, 15)
	VPBROADCASTD 28(SI), Z25

	ADDQ $32, SI
	ADDQ $128, DI

	// Step B: VDPBF16PS using Buffer B (Z26, Z27) while loading next Buffer A (Z16, Z17, Z18..Z25)
	VDPBF16PS(26, 18, 0)
	VMOVDQU32 (DI), Z16
	VDPBF16PS(27, 18, 1)
	VPBROADCASTD (SI), Z18

	VDPBF16PS(26, 19, 2)
	VMOVDQU32 64(DI), Z17
	VDPBF16PS(27, 19, 3)
	VPBROADCASTD 4(SI), Z19

	VDPBF16PS(26, 20, 4)
	VDPBF16PS(27, 20, 5)
	VPBROADCASTD 8(SI), Z20

	VDPBF16PS(26, 21, 6)
	VDPBF16PS(27, 21, 7)
	VPBROADCASTD 12(SI), Z21

	VDPBF16PS(26, 22, 8)
	VDPBF16PS(27, 22, 9)
	VPBROADCASTD 16(SI), Z22

	VDPBF16PS(26, 23, 10)
	VDPBF16PS(27, 23, 11)
	VPBROADCASTD 20(SI), Z23

	VDPBF16PS(26, 24, 12)
	VDPBF16PS(27, 24, 13)
	VPBROADCASTD 24(SI), Z24

	VDPBF16PS(26, 25, 14)
	VDPBF16PS(27, 25, 15)
	VPBROADCASTD 28(SI), Z25

	ADDQ $32, SI
	ADDQ $128, DI

	DECQ CX
	JNZ k_pair_loop_vdpbf16

k_last_pair_vdpbf16:
	// Step A: load final Buffer B
	VDPBF16PS(16, 18, 0)
	VMOVDQU32 (DI), Z26
	VDPBF16PS(17, 18, 1)
	VPBROADCASTD (SI), Z18

	VDPBF16PS(16, 19, 2)
	VMOVDQU32 64(DI), Z27
	VDPBF16PS(17, 19, 3)
	VPBROADCASTD 4(SI), Z19

	VDPBF16PS(16, 20, 4)
	VDPBF16PS(17, 20, 5)
	VPBROADCASTD 8(SI), Z20

	VDPBF16PS(16, 21, 6)
	VDPBF16PS(17, 21, 7)
	VPBROADCASTD 12(SI), Z21

	VDPBF16PS(16, 22, 8)
	VDPBF16PS(17, 22, 9)
	VPBROADCASTD 16(SI), Z22

	VDPBF16PS(16, 23, 10)
	VDPBF16PS(17, 23, 11)
	VPBROADCASTD 20(SI), Z23

	VDPBF16PS(16, 24, 12)
	VDPBF16PS(17, 24, 13)
	VPBROADCASTD 24(SI), Z24

	VDPBF16PS(16, 25, 14)
	VDPBF16PS(17, 25, 15)
	VPBROADCASTD 28(SI), Z25

	ADDQ $32, SI
	ADDQ $128, DI

	// Step B: final K-pair without prefetching past end
	VDPBF16PS(26, 18, 0)
	VDPBF16PS(27, 18, 1)
	VDPBF16PS(26, 19, 2)
	VDPBF16PS(27, 19, 3)
	VDPBF16PS(26, 20, 4)
	VDPBF16PS(27, 20, 5)
	VDPBF16PS(26, 21, 6)
	VDPBF16PS(27, 21, 7)
	VDPBF16PS(26, 22, 8)
	VDPBF16PS(27, 22, 9)
	VDPBF16PS(26, 23, 10)
	VDPBF16PS(27, 23, 11)
	VDPBF16PS(26, 24, 12)
	VDPBF16PS(27, 24, 13)
	VDPBF16PS(26, 25, 14)
	VDPBF16PS(27, 25, 15)

k_odd_pair_vdpbf16:
	TESTQ $1, R12
	JZ store_output_vdpbf16

	VMOVDQU32 (DI), Z16
	VMOVDQU32 64(DI), Z17
	VPBROADCASTD (SI), Z18
	VPBROADCASTD 4(SI), Z19
	VPBROADCASTD 8(SI), Z20
	VPBROADCASTD 12(SI), Z21
	VPBROADCASTD 16(SI), Z22
	VPBROADCASTD 20(SI), Z23
	VPBROADCASTD 24(SI), Z24
	VPBROADCASTD 28(SI), Z25

	VDPBF16PS(16, 18, 0)
	VDPBF16PS(17, 18, 1)
	VDPBF16PS(16, 19, 2)
	VDPBF16PS(17, 19, 3)
	VDPBF16PS(16, 20, 4)
	VDPBF16PS(17, 20, 5)
	VDPBF16PS(16, 21, 6)
	VDPBF16PS(17, 21, 7)
	VDPBF16PS(16, 22, 8)
	VDPBF16PS(17, 22, 9)
	VDPBF16PS(16, 23, 10)
	VDPBF16PS(17, 23, 11)
	VDPBF16PS(16, 24, 12)
	VDPBF16PS(17, 24, 13)
	VDPBF16PS(16, 25, 14)
	VDPBF16PS(17, 25, 15)

store_output_vdpbf16:
	// Compute outPtr = R10 + (lhsRowIdx * outputStrideBytes) + (rhsColIdx * 4)
	MOVQ AX, DX
	IMULQ R11, DX
	MOVQ BX, SI
	SHLQ $2, SI
	ADDQ SI, DX
	ADDQ R10, DX

	TESTB R15, R15
	JNZ store_output_vdpbf16_accum

	// Overwrite (FP-domain VMOVUPS avoids int<->FP domain crossing)
	VMOVUPS Z0, (DX)
	VMOVUPS Z1, 64(DX)

	ADDQ R11, DX
	VMOVUPS Z2, (DX)
	VMOVUPS Z3, 64(DX)

	ADDQ R11, DX
	VMOVUPS Z4, (DX)
	VMOVUPS Z5, 64(DX)

	ADDQ R11, DX
	VMOVUPS Z6, (DX)
	VMOVUPS Z7, 64(DX)

	ADDQ R11, DX
	VMOVUPS Z8, (DX)
	VMOVUPS Z9, 64(DX)

	ADDQ R11, DX
	VMOVUPS Z10, (DX)
	VMOVUPS Z11, 64(DX)

	ADDQ R11, DX
	VMOVUPS Z12, (DX)
	VMOVUPS Z13, 64(DX)

	ADDQ R11, DX
	VMOVUPS Z14, (DX)
	VMOVUPS Z15, 64(DX)

	ADDQ $8, AX
	JMP loop_lhs_vdpbf16

store_output_vdpbf16_accum:
	VADDPS (DX), Z0, Z0
	VMOVUPS Z0, (DX)
	VADDPS 64(DX), Z1, Z1
	VMOVUPS Z1, 64(DX)

	ADDQ R11, DX
	VADDPS (DX), Z2, Z2
	VMOVUPS Z2, (DX)
	VADDPS 64(DX), Z3, Z3
	VMOVUPS Z3, 64(DX)

	ADDQ R11, DX
	VADDPS (DX), Z4, Z4
	VMOVUPS Z4, (DX)
	VADDPS 64(DX), Z5, Z5
	VMOVUPS Z5, 64(DX)

	ADDQ R11, DX
	VADDPS (DX), Z6, Z6
	VMOVUPS Z6, (DX)
	VADDPS 64(DX), Z7, Z7
	VMOVUPS Z7, 64(DX)

	ADDQ R11, DX
	VADDPS (DX), Z8, Z8
	VMOVUPS Z8, (DX)
	VADDPS 64(DX), Z9, Z9
	VMOVUPS Z9, 64(DX)

	ADDQ R11, DX
	VADDPS (DX), Z10, Z10
	VMOVUPS Z10, (DX)
	VADDPS 64(DX), Z11, Z11
	VMOVUPS Z11, 64(DX)

	ADDQ R11, DX
	VADDPS (DX), Z12, Z12
	VMOVUPS Z12, (DX)
	VADDPS 64(DX), Z13, Z13
	VMOVUPS Z13, 64(DX)

	ADDQ R11, DX
	VADDPS (DX), Z14, Z14
	VMOVUPS Z14, (DX)
	VADDPS 64(DX), Z15, Z15
	VMOVUPS Z15, 64(DX)

	ADDQ $8, AX
	JMP loop_lhs_vdpbf16

next_rhs_vdpbf16:
	ADDQ $32, BX
	JMP loop_rhs_vdpbf16

done_vdpbf16:
	VZEROUPPER
	RET

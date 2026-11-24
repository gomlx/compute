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

// func avx512LargeKernelM1BFloat16BF16Asm(
//     lhs []bfloat16.BFloat16,
//     packedRHS []bfloat16.BFloat16,
//     out []float32,
//     contractingLen int,
//     rhsActiveCols int,
//     accumulate bool)
//
// Dedicated M=1 (GEMV) native AVX512_BF16 (VDPBF16PS) microkernel.
// Reads the raw contiguous 1-row BF16 lhs vector directly (zero LHS packing) and
// streams each 32-col K2-interleaved packedRHS strip linearly ([ceil(N/32), kPairs, 32, 2])
// to maximize hardware stream prefetcher efficiency and avoid multi-strip 16 KiB cache-set aliasing.
// Uses 4x K-pair unrolling (8 K-steps = 512 bytes/iter) across 4 independent accumulator pairs
// (Z0..Z7) to hide VDPBF16PS latency while executing 8x fewer dot products than the 8x32 GEMM kernel.
// out must have length >= ceil(rhsActiveCols/32)*32.
TEXT ·avx512LargeKernelM1BFloat16BF16Asm(SB), NOSPLIT, $0-96
	MOVQ lhs_base+0(FP), R8           // R8 = lhsBasePtr (raw [contractingLen] BF16)
	MOVQ packedRHS_base+24(FP), R9    // R9 = rhsBasePtr (advances linearly across strips)
	MOVQ out_base+48(FP), R10         // R10 = outPtr (advances by 128 bytes per 32 cols)
	MOVQ contractingLen+72(FP), CX    // CX = contractingLen (K)
	MOVQ rhsActiveCols+80(FP), R14    // R14 = rhsRemCols
	MOVB accumulate+88(FP), R15       // R15 = accumulate (0 = overwrite, 1 = add)

	TESTQ CX, CX
	JLE k_zero_m1_bf16
	MOVQ CX, R11
	SHRQ $1, R11                      // R11 = fullPairs = K >> 1
	MOVQ CX, R12
	ANDQ $1, R12                      // R12 = hasOddK = K & 1
	MOVQ CX, R13
	INCQ R13
	SHRQ $1, R13                      // R13 = kPairs = (K + 1) >> 1
	SHLQ $7, R13                      // R13 = stripBytes = kPairs * 128
	JMP loop_strip_m1_bf16

k_zero_m1_bf16:
	XORQ R11, R11
	XORQ R12, R12
	XORQ R13, R13

loop_strip_m1_bf16:
	TESTQ R14, R14
	JLE done_m1_bf16

	// 4 independent accumulator pairs for 32 output cols (Z0/Z1, Z2/Z3, Z4/Z5, Z6/Z7)
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7

	MOVQ R8, SI                       // SI = lhsPtr
	MOVQ R9, DI                       // DI = rhsStripPtr

	MOVQ R11, CX                      // CX = fullPairs
	SHRQ $2, CX                       // CX = quadPairs = fullPairs / 4 (512 bytes RHS / iter)
	JZ k_rem_pairs_m1_bf16

	PCALIGN $64
k_quadpair_m1_bf16:
	// Pair 0 (bytes 0..127)
	VPBROADCASTD (SI), Z16
	VMOVDQU32 (DI), Z20
	VMOVDQU32 64(DI), Z21
	VDPBF16PS(20, 16, 0)
	VDPBF16PS(21, 16, 1)

	// Pair 1 (bytes 128..255)
	VPBROADCASTD 4(SI), Z17
	VMOVDQU32 128(DI), Z22
	VMOVDQU32 192(DI), Z23
	VDPBF16PS(22, 17, 2)
	VDPBF16PS(23, 17, 3)

	// Pair 2 (bytes 256..383)
	VPBROADCASTD 8(SI), Z18
	VMOVDQU32 256(DI), Z24
	VMOVDQU32 320(DI), Z25
	VDPBF16PS(24, 18, 4)
	VDPBF16PS(25, 18, 5)

	// Pair 3 (bytes 384..511)
	VPBROADCASTD 12(SI), Z19
	VMOVDQU32 384(DI), Z26
	VMOVDQU32 448(DI), Z27
	VDPBF16PS(26, 19, 6)
	VDPBF16PS(27, 19, 7)

	ADDQ $16, SI
	ADDQ $512, DI
	DECQ CX
	JNZ k_quadpair_m1_bf16

	// Reduce 4 accumulator pairs into (Z0, Z1)
	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z6, Z4, Z4
	VADDPS Z7, Z5, Z5
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1

k_rem_pairs_m1_bf16:
	MOVQ R11, CX
	ANDQ $3, CX                       // 0..3 remaining full K-pairs
	JZ k_oddk_m1_bf16

k_singlepair_m1_bf16:
	VPBROADCASTD (SI), Z16
	VMOVDQU32 (DI), Z20
	VMOVDQU32 64(DI), Z21
	VDPBF16PS(20, 16, 0)
	VDPBF16PS(21, 16, 1)
	ADDQ $4, SI
	ADDQ $128, DI
	DECQ CX
	JNZ k_singlepair_m1_bf16

k_oddk_m1_bf16:
	TESTQ $1, R12
	JZ store_strip_m1_bf16

	MOVWLZX (SI), CX
	VMOVD CX, X16
	VPBROADCASTD X16, Z16
	VMOVDQU32 (DI), Z20
	VMOVDQU32 64(DI), Z21
	VDPBF16PS(20, 16, 0)
	VDPBF16PS(21, 16, 1)

store_strip_m1_bf16:
	TESTB R15, R15
	JNZ store_strip_accum_m1_bf16

	VMOVUPS Z0, (R10)
	VMOVUPS Z1, 64(R10)
	JMP next_strip_m1_bf16

store_strip_accum_m1_bf16:
	VADDPS (R10), Z0, Z0
	VMOVUPS Z0, (R10)
	VADDPS 64(R10), Z1, Z1
	VMOVUPS Z1, 64(R10)

next_strip_m1_bf16:
	ADDQ R13, R9                      // advance rhsBasePtr by 1 strip (contiguous!)
	ADDQ $128, R10                    // advance outPtr by 32 float32s
	SUBQ $32, R14
	JMP loop_strip_m1_bf16

done_m1_bf16:
	VZEROUPPER
	RET

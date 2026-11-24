// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2LargeKernelM1BFloat16Asm(
//     lhs []bfloat16.BFloat16,
//     packedRHS []bfloat16.BFloat16,
//     out []float32,
//     contractingLen int,
//     rhsActiveCols int,
//     accumulate bool)
//
// Dedicated M=1 (GEMV) AVX2 microkernel for BFloat16 -> Float32 (unpacking BF16 to F32).
// Reads raw contiguous 1-row BF16 lhs directly and streams each 16-col packedRHS strip linearly
// ([ceil(N/16), K, 16] bfloat16) using 4x K-unrolling across 4 accumulator pairs (Y0..Y7).
// out must have length >= ceil(rhsActiveCols/16)*16.
TEXT ·avx2LargeKernelM1BFloat16Asm(SB), NOSPLIT, $0-96
	MOVQ lhs_base+0(FP), R8           // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9    // R9 = rhsBasePtr
	MOVQ out_base+48(FP), R10         // R10 = outPtr
	MOVQ contractingLen+72(FP), R12   // R12 = K
	MOVQ rhsActiveCols+80(FP), R14    // R14 = rhsRemCols
	MOVB accumulate+88(FP), R15       // R15 = accumulate

	MOVQ R12, R13
	SHLQ $5, R13                      // R13 = stripBytes = K * 16 * 2 = K * 32

loop_strip_avx2_m1_bf16:
	TESTQ R14, R14
	JLE done_avx2_m1_bf16

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	MOVQ R8, SI
	MOVQ R9, DI

	MOVQ R12, CX
	TESTQ CX, CX
	JLE store_strip_avx2_m1_bf16

	SHRQ $2, CX                       // quads = K / 4 (128 bytes RHS / iter)
	JZ k_rem_avx2_m1_bf16

	PCALIGN $64
k_quad_avx2_m1_bf16:
	// Load 4 BF16 lhs values -> 4 FP32 in X8, broadcast each lane into Y8..Y11
	VPMOVZXWD (SI), X8
	VPSLLD $16, X8, X8
	VPERMILPS $0x55, X8, X9
	VPERMILPS $0xAA, X8, X10
	VPERMILPS $0xFF, X8, X11
	VBROADCASTSS X8, Y8
	VBROADCASTSS X9, Y9
	VBROADCASTSS X10, Y10
	VBROADCASTSS X11, Y11

	// K + 0 (bytes 0..31)
	VPMOVZXWD (DI), Y12
	VPSLLD $16, Y12, Y12
	VPMOVZXWD 16(DI), Y13
	VPSLLD $16, Y13, Y13
	VFMADD231PS Y12, Y8, Y0
	VFMADD231PS Y13, Y8, Y1

	// K + 1 (bytes 32..63)
	VPMOVZXWD 32(DI), Y14
	VPSLLD $16, Y14, Y14
	VPMOVZXWD 48(DI), Y15
	VPSLLD $16, Y15, Y15
	VFMADD231PS Y14, Y9, Y2
	VFMADD231PS Y15, Y9, Y3

	// K + 2 (bytes 64..95)
	VPMOVZXWD 64(DI), Y12
	VPSLLD $16, Y12, Y12
	VPMOVZXWD 80(DI), Y13
	VPSLLD $16, Y13, Y13
	VFMADD231PS Y12, Y10, Y4
	VFMADD231PS Y13, Y10, Y5

	// K + 3 (bytes 96..127)
	VPMOVZXWD 96(DI), Y14
	VPSLLD $16, Y14, Y14
	VPMOVZXWD 112(DI), Y15
	VPSLLD $16, Y15, Y15
	VFMADD231PS Y14, Y11, Y6
	VFMADD231PS Y15, Y11, Y7

	ADDQ $8, SI
	ADDQ $128, DI
	DECQ CX
	JNZ k_quad_avx2_m1_bf16

	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y6, Y4, Y4
	VADDPS Y7, Y5, Y5
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1

k_rem_avx2_m1_bf16:
	MOVQ R12, CX
	ANDQ $3, CX
	JZ store_strip_avx2_m1_bf16

k_single_avx2_m1_bf16:
	MOVWLZX (SI), AX
	SHLL $16, AX
	VMOVD AX, X8
	VBROADCASTSS X8, Y8
	VPMOVZXWD (DI), Y12
	VPSLLD $16, Y12, Y12
	VPMOVZXWD 16(DI), Y13
	VPSLLD $16, Y13, Y13
	VFMADD231PS Y12, Y8, Y0
	VFMADD231PS Y13, Y8, Y1
	ADDQ $2, SI
	ADDQ $32, DI
	DECQ CX
	JNZ k_single_avx2_m1_bf16

store_strip_avx2_m1_bf16:
	TESTB R15, R15
	JNZ store_strip_accum_avx2_m1_bf16

	VMOVUPS Y0, (R10)
	VMOVUPS Y1, 32(R10)
	JMP next_strip_avx2_m1_bf16

store_strip_accum_avx2_m1_bf16:
	VADDPS (R10), Y0, Y0
	VMOVUPS Y0, (R10)
	VADDPS 32(R10), Y1, Y1
	VMOVUPS Y1, 32(R10)

next_strip_avx2_m1_bf16:
	ADDQ R13, R9
	ADDQ $64, R10
	SUBQ $16, R14
	JMP loop_strip_avx2_m1_bf16

done_avx2_m1_bf16:
	VZEROUPPER
	RET

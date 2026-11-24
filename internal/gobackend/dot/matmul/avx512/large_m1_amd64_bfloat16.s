// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512LargeKernelM1BFloat16Asm(
//     lhs []bfloat16.BFloat16,
//     packedRHS []bfloat16.BFloat16,
//     out []float32,
//     contractingLen int,
//     rhsActiveCols int,
//     accumulate bool)
//
// Dedicated M=1 (GEMV) AVX-512 F32-widening microkernel for BFloat16 (used when AVX512_BF16
// VDPBF16PS is unavailable). Reads raw contiguous 1-row BF16 lhs directly and streams each
// 32-col standard packedRHS strip linearly ([ceil(N/32), K, 32] bfloat16) with 4x K-unrolling.
// out must have length >= ceil(rhsActiveCols/32)*32.
TEXT ·avx512LargeKernelM1BFloat16Asm(SB), NOSPLIT, $0-96
	MOVQ lhs_base+0(FP), R8           // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9    // R9 = rhsBasePtr
	MOVQ out_base+48(FP), R10         // R10 = outPtr
	MOVQ contractingLen+72(FP), R12   // R12 = K
	MOVQ rhsActiveCols+80(FP), R14    // R14 = rhsRemCols
	MOVB accumulate+88(FP), R15       // R15 = accumulate

	MOVQ R12, R13
	SHLQ $6, R13                      // R13 = stripBytes = K * 32 * 2 = K * 64

loop_strip_m1_bf16_widen:
	TESTQ R14, R14
	JLE done_m1_bf16_widen

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7

	MOVQ R8, SI
	MOVQ R9, DI

	MOVQ R12, CX
	TESTQ CX, CX
	JLE store_strip_m1_bf16_widen

	SHRQ $2, CX                       // quads = K / 4 (256 bytes RHS / iter)
	JZ k_rem_m1_bf16_widen

	PCALIGN $64
k_quad_m1_bf16_widen:
	// Load 4 BF16 lhs values -> 4 FP32 in X16, then broadcast each lane into Z16..Z19
	VPMOVZXWD (SI), X16
	VPSLLD $16, X16, X16
	VPERMILPS $0x55, X16, X17
	VPERMILPS $0xAA, X16, X18
	VPERMILPS $0xFF, X16, X19
	VBROADCASTSS X16, Z16
	VBROADCASTSS X17, Z17
	VBROADCASTSS X18, Z18
	VBROADCASTSS X19, Z19

	// K + 0 (bytes 0..63)
	VPMOVZXWD (DI), Z20
	VPSLLD $16, Z20, Z20
	VPMOVZXWD 32(DI), Z21
	VPSLLD $16, Z21, Z21
	VFMADD231PS Z20, Z16, Z0
	VFMADD231PS Z21, Z16, Z1

	// K + 1 (bytes 64..127)
	VPMOVZXWD 64(DI), Z22
	VPSLLD $16, Z22, Z22
	VPMOVZXWD 96(DI), Z23
	VPSLLD $16, Z23, Z23
	VFMADD231PS Z22, Z17, Z2
	VFMADD231PS Z23, Z17, Z3

	// K + 2 (bytes 128..191)
	VPMOVZXWD 128(DI), Z24
	VPSLLD $16, Z24, Z24
	VPMOVZXWD 160(DI), Z25
	VPSLLD $16, Z25, Z25
	VFMADD231PS Z24, Z18, Z4
	VFMADD231PS Z25, Z18, Z5

	// K + 3 (bytes 192..255)
	VPMOVZXWD 192(DI), Z26
	VPSLLD $16, Z26, Z26
	VPMOVZXWD 224(DI), Z27
	VPSLLD $16, Z27, Z27
	VFMADD231PS Z26, Z19, Z6
	VFMADD231PS Z27, Z19, Z7

	ADDQ $8, SI
	ADDQ $256, DI
	DECQ CX
	JNZ k_quad_m1_bf16_widen

	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z6, Z4, Z4
	VADDPS Z7, Z5, Z5
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1

k_rem_m1_bf16_widen:
	MOVQ R12, CX
	ANDQ $3, CX
	JZ store_strip_m1_bf16_widen

k_single_m1_bf16_widen:
	MOVWLZX (SI), AX
	SHLL $16, AX
	VMOVD AX, X16
	VBROADCASTSS X16, Z16
	VPMOVZXWD (DI), Z20
	VPSLLD $16, Z20, Z20
	VPMOVZXWD 32(DI), Z21
	VPSLLD $16, Z21, Z21
	VFMADD231PS Z20, Z16, Z0
	VFMADD231PS Z21, Z16, Z1
	ADDQ $2, SI
	ADDQ $64, DI
	DECQ CX
	JNZ k_single_m1_bf16_widen

store_strip_m1_bf16_widen:
	TESTB R15, R15
	JNZ store_strip_accum_m1_bf16_widen

	VMOVUPS Z0, (R10)
	VMOVUPS Z1, 64(R10)
	JMP next_strip_m1_bf16_widen

store_strip_accum_m1_bf16_widen:
	VADDPS (R10), Z0, Z0
	VMOVUPS Z0, (R10)
	VADDPS 64(R10), Z1, Z1
	VMOVUPS Z1, 64(R10)

next_strip_m1_bf16_widen:
	ADDQ R13, R9
	ADDQ $128, R10
	SUBQ $32, R14
	JMP loop_strip_m1_bf16_widen

done_m1_bf16_widen:
	VZEROUPPER
	RET

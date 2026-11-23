// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512LargeKernelM1Float32Asm(
//     lhs []float32,
//     packedRHS []float32,
//     out []float32,
//     contractingLen int,
//     rhsActiveCols int,
//     accumulate bool)
//
// Dedicated M=1 (GEMV) AVX-512 microkernel for Float32.
// Reads the raw contiguous 1-row lhs vector directly (zero LHS packing) and streams each
// 32-col packedRHS strip linearly ([ceil(N/32), K, 32] float32) using 4x K-unrolling across
// 4 independent accumulator pairs (Z0..Z7) to hide VFMADD231PS latency.
// out must have length >= ceil(rhsActiveCols/32)*32.
TEXT ·avx512LargeKernelM1Float32Asm(SB), NOSPLIT, $0-96
	MOVQ lhs_base+0(FP), R8           // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9    // R9 = rhsBasePtr
	MOVQ out_base+48(FP), R10         // R10 = outPtr
	MOVQ contractingLen+72(FP), R12   // R12 = K
	MOVQ rhsActiveCols+80(FP), R14    // R14 = rhsRemCols
	MOVB accumulate+88(FP), R15       // R15 = accumulate

	MOVQ R12, R13
	SHLQ $7, R13                      // R13 = stripBytes = K * 32 * 4 = K * 128

loop_strip_m1_f32:
	TESTQ R14, R14
	JLE done_m1_f32

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
	JLE store_strip_m1_f32

	SHRQ $2, CX                       // quads = K / 4 (512 bytes RHS / iter)
	JZ k_rem_m1_f32

	PCALIGN $64
k_quad_m1_f32:
	// K + 0
	VBROADCASTSS (SI), Z16
	VMOVUPS (DI), Z20
	VMOVUPS 64(DI), Z21
	VFMADD231PS Z20, Z16, Z0
	VFMADD231PS Z21, Z16, Z1

	// K + 1
	VBROADCASTSS 4(SI), Z17
	VMOVUPS 128(DI), Z22
	VMOVUPS 192(DI), Z23
	VFMADD231PS Z22, Z17, Z2
	VFMADD231PS Z23, Z17, Z3

	// K + 2
	VBROADCASTSS 8(SI), Z18
	VMOVUPS 256(DI), Z24
	VMOVUPS 320(DI), Z25
	VFMADD231PS Z24, Z18, Z4
	VFMADD231PS Z25, Z18, Z5

	// K + 3
	VBROADCASTSS 12(SI), Z19
	VMOVUPS 384(DI), Z26
	VMOVUPS 448(DI), Z27
	VFMADD231PS Z26, Z19, Z6
	VFMADD231PS Z27, Z19, Z7

	ADDQ $16, SI
	ADDQ $512, DI
	DECQ CX
	JNZ k_quad_m1_f32

	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z6, Z4, Z4
	VADDPS Z7, Z5, Z5
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1

k_rem_m1_f32:
	MOVQ R12, CX
	ANDQ $3, CX
	JZ store_strip_m1_f32

k_single_m1_f32:
	VBROADCASTSS (SI), Z16
	VMOVUPS (DI), Z20
	VMOVUPS 64(DI), Z21
	VFMADD231PS Z20, Z16, Z0
	VFMADD231PS Z21, Z16, Z1
	ADDQ $4, SI
	ADDQ $128, DI
	DECQ CX
	JNZ k_single_m1_f32

store_strip_m1_f32:
	TESTB R15, R15
	JNZ store_strip_accum_m1_f32

	VMOVUPS Z0, (R10)
	VMOVUPS Z1, 64(R10)
	JMP next_strip_m1_f32

store_strip_accum_m1_f32:
	VADDPS (R10), Z0, Z0
	VMOVUPS Z0, (R10)
	VADDPS 64(R10), Z1, Z1
	VMOVUPS Z1, 64(R10)

next_strip_m1_f32:
	ADDQ R13, R9
	ADDQ $128, R10
	SUBQ $32, R14
	JMP loop_strip_m1_f32

done_m1_f32:
	VZEROUPPER
	RET

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512LargeKernelM1Float64Asm(
//     lhs []float64,
//     packedRHS []float64,
//     out []float64,
//     contractingLen int,
//     rhsActiveCols int,
//     accumulate bool)
//
// Dedicated M=1 (GEMV) AVX-512 microkernel for Float64.
// Reads the raw contiguous 1-row lhs vector directly (zero LHS packing) and streams each
// 16-col packedRHS strip linearly ([ceil(N/16), K, 16] float64) using 4x K-unrolling across
// 4 independent accumulator pairs (Z0..Z7) to hide VFMADD231PD latency.
// out must have length >= ceil(rhsActiveCols/16)*16.
TEXT ·avx512LargeKernelM1Float64Asm(SB), NOSPLIT, $0-96
	MOVQ lhs_base+0(FP), R8           // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9    // R9 = rhsBasePtr
	MOVQ out_base+48(FP), R10         // R10 = outPtr
	MOVQ contractingLen+72(FP), R12   // R12 = K
	MOVQ rhsActiveCols+80(FP), R14    // R14 = rhsRemCols
	MOVB accumulate+88(FP), R15       // R15 = accumulate

	MOVQ R12, R13
	SHLQ $7, R13                      // R13 = stripBytes = K * 16 * 8 = K * 128

loop_strip_m1_f64:
	TESTQ R14, R14
	JLE done_m1_f64

	VXORPD Z0, Z0, Z0
	VXORPD Z1, Z1, Z1
	VXORPD Z2, Z2, Z2
	VXORPD Z3, Z3, Z3
	VXORPD Z4, Z4, Z4
	VXORPD Z5, Z5, Z5
	VXORPD Z6, Z6, Z6
	VXORPD Z7, Z7, Z7

	MOVQ R8, SI
	MOVQ R9, DI

	MOVQ R12, CX
	TESTQ CX, CX
	JLE store_strip_m1_f64

	SHRQ $2, CX                       // quads = K / 4 (512 bytes RHS / iter)
	JZ k_rem_m1_f64

	PCALIGN $64
k_quad_m1_f64:
	// K + 0
	VBROADCASTSD (SI), Z16
	VMOVUPD (DI), Z20
	VMOVUPD 64(DI), Z21
	VFMADD231PD Z20, Z16, Z0
	VFMADD231PD Z21, Z16, Z1

	// K + 1
	VBROADCASTSD 8(SI), Z17
	VMOVUPD 128(DI), Z22
	VMOVUPD 192(DI), Z23
	VFMADD231PD Z22, Z17, Z2
	VFMADD231PD Z23, Z17, Z3

	// K + 2
	VBROADCASTSD 16(SI), Z18
	VMOVUPD 256(DI), Z24
	VMOVUPD 320(DI), Z25
	VFMADD231PD Z24, Z18, Z4
	VFMADD231PD Z25, Z18, Z5

	// K + 3
	VBROADCASTSD 24(SI), Z19
	VMOVUPD 384(DI), Z26
	VMOVUPD 448(DI), Z27
	VFMADD231PD Z26, Z19, Z6
	VFMADD231PD Z27, Z19, Z7

	ADDQ $32, SI
	ADDQ $512, DI
	DECQ CX
	JNZ k_quad_m1_f64

	VADDPD Z2, Z0, Z0
	VADDPD Z3, Z1, Z1
	VADDPD Z6, Z4, Z4
	VADDPD Z7, Z5, Z5
	VADDPD Z4, Z0, Z0
	VADDPD Z5, Z1, Z1

k_rem_m1_f64:
	MOVQ R12, CX
	ANDQ $3, CX
	JZ store_strip_m1_f64

k_single_m1_f64:
	VBROADCASTSD (SI), Z16
	VMOVUPD (DI), Z20
	VMOVUPD 64(DI), Z21
	VFMADD231PD Z20, Z16, Z0
	VFMADD231PD Z21, Z16, Z1
	ADDQ $8, SI
	ADDQ $128, DI
	DECQ CX
	JNZ k_single_m1_f64

store_strip_m1_f64:
	TESTB R15, R15
	JNZ store_strip_accum_m1_f64

	VMOVUPD Z0, (R10)
	VMOVUPD Z1, 64(R10)
	JMP next_strip_m1_f64

store_strip_accum_m1_f64:
	VADDPD (R10), Z0, Z0
	VMOVUPD Z0, (R10)
	VADDPD 64(R10), Z1, Z1
	VMOVUPD Z1, 64(R10)

next_strip_m1_f64:
	ADDQ R13, R9
	ADDQ $128, R10
	SUBQ $16, R14
	JMP loop_strip_m1_f64

done_m1_f64:
	VZEROUPPER
	RET

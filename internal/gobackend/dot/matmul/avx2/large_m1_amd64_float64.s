// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2LargeKernelM1Float64Asm(
//     lhs []float64,
//     packedRHS []float64,
//     out []float64,
//     contractingLen int,
//     rhsActiveCols int,
//     accumulate bool)
//
// Dedicated M=1 (GEMV) AVX2 microkernel for Float64.
// Reads raw contiguous 1-row lhs directly and streams each 8-col packedRHS strip linearly
// ([ceil(N/8), K, 8] float64) using 4x K-unrolling across 4 accumulator pairs (Y0..Y7).
// out must have length >= ceil(rhsActiveCols/8)*8.
TEXT ·avx2LargeKernelM1Float64Asm(SB), NOSPLIT, $0-96
	MOVQ lhs_base+0(FP), R8           // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9    // R9 = rhsBasePtr
	MOVQ out_base+48(FP), R10         // R10 = outPtr
	MOVQ contractingLen+72(FP), R12   // R12 = K
	MOVQ rhsActiveCols+80(FP), R14    // R14 = rhsRemCols
	MOVB accumulate+88(FP), R15       // R15 = accumulate

	MOVQ R12, R13
	SHLQ $6, R13                      // R13 = stripBytes = K * 8 * 8 = K * 64

loop_strip_avx2_m1_f64:
	TESTQ R14, R14
	JLE done_avx2_m1_f64

	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3
	VXORPD Y4, Y4, Y4
	VXORPD Y5, Y5, Y5
	VXORPD Y6, Y6, Y6
	VXORPD Y7, Y7, Y7

	MOVQ R8, SI
	MOVQ R9, DI

	MOVQ R12, CX
	TESTQ CX, CX
	JLE store_strip_avx2_m1_f64

	SHRQ $2, CX                       // quads = K / 4 (256 bytes RHS / iter)
	JZ k_rem_avx2_m1_f64

	PCALIGN $64
k_quad_avx2_m1_f64:
	// K + 0
	VBROADCASTSD (SI), Y8
	VMOVUPD (DI), Y9
	VMOVUPD 32(DI), Y10
	VFMADD231PD Y9, Y8, Y0
	VFMADD231PD Y10, Y8, Y1

	// K + 1
	VBROADCASTSD 8(SI), Y11
	VMOVUPD 64(DI), Y12
	VMOVUPD 96(DI), Y13
	VFMADD231PD Y12, Y11, Y2
	VFMADD231PD Y13, Y11, Y3

	// K + 2
	VBROADCASTSD 16(SI), Y8
	VMOVUPD 128(DI), Y9
	VMOVUPD 160(DI), Y10
	VFMADD231PD Y9, Y8, Y4
	VFMADD231PD Y10, Y8, Y5

	// K + 3
	VBROADCASTSD 24(SI), Y11
	VMOVUPD 192(DI), Y12
	VMOVUPD 224(DI), Y13
	VFMADD231PD Y12, Y11, Y6
	VFMADD231PD Y13, Y11, Y7

	ADDQ $32, SI
	ADDQ $256, DI
	DECQ CX
	JNZ k_quad_avx2_m1_f64

	VADDPD Y2, Y0, Y0
	VADDPD Y3, Y1, Y1
	VADDPD Y6, Y4, Y4
	VADDPD Y7, Y5, Y5
	VADDPD Y4, Y0, Y0
	VADDPD Y5, Y1, Y1

k_rem_avx2_m1_f64:
	MOVQ R12, CX
	ANDQ $3, CX
	JZ store_strip_avx2_m1_f64

k_single_avx2_m1_f64:
	VBROADCASTSD (SI), Y8
	VMOVUPD (DI), Y9
	VMOVUPD 32(DI), Y10
	VFMADD231PD Y9, Y8, Y0
	VFMADD231PD Y10, Y8, Y1
	ADDQ $8, SI
	ADDQ $64, DI
	DECQ CX
	JNZ k_single_avx2_m1_f64

store_strip_avx2_m1_f64:
	TESTB R15, R15
	JNZ store_strip_accum_avx2_m1_f64

	VMOVUPD Y0, (R10)
	VMOVUPD Y1, 32(R10)
	JMP next_strip_avx2_m1_f64

store_strip_accum_avx2_m1_f64:
	VADDPD (R10), Y0, Y0
	VMOVUPD Y0, (R10)
	VADDPD 32(R10), Y1, Y1
	VMOVUPD Y1, 32(R10)

next_strip_avx2_m1_f64:
	ADDQ R13, R9
	ADDQ $64, R10
	SUBQ $8, R14
	JMP loop_strip_avx2_m1_f64

done_avx2_m1_f64:
	VZEROUPPER
	RET

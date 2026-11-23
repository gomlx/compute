// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2LargeKernelM1Float32Asm(
//     lhs []float32,
//     packedRHS []float32,
//     out []float32,
//     contractingLen int,
//     rhsActiveCols int,
//     accumulate bool)
//
// Dedicated M=1 (GEMV) AVX2 microkernel for Float32.
// Reads raw contiguous 1-row lhs directly and streams each 16-col packedRHS strip linearly
// ([ceil(N/16), K, 16] float32) using 4x K-unrolling across 4 accumulator pairs (Y0..Y7).
// out must have length >= ceil(rhsActiveCols/16)*16.
TEXT ·avx2LargeKernelM1Float32Asm(SB), NOSPLIT, $0-96
	MOVQ lhs_base+0(FP), R8           // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9    // R9 = rhsBasePtr
	MOVQ out_base+48(FP), R10         // R10 = outPtr
	MOVQ contractingLen+72(FP), R12   // R12 = K
	MOVQ rhsActiveCols+80(FP), R14    // R14 = rhsRemCols
	MOVB accumulate+88(FP), R15       // R15 = accumulate

	MOVQ R12, R13
	SHLQ $6, R13                      // R13 = stripBytes = K * 16 * 4 = K * 64

loop_strip_avx2_m1_f32:
	TESTQ R14, R14
	JLE done_avx2_m1_f32

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
	JLE store_strip_avx2_m1_f32

	SHRQ $2, CX                       // quads = K / 4 (256 bytes RHS / iter)
	JZ k_rem_avx2_m1_f32

	PCALIGN $64
k_quad_avx2_m1_f32:
	// K + 0
	VBROADCASTSS (SI), Y8
	VMOVUPS (DI), Y9
	VMOVUPS 32(DI), Y10
	VFMADD231PS Y9, Y8, Y0
	VFMADD231PS Y10, Y8, Y1

	// K + 1
	VBROADCASTSS 4(SI), Y11
	VMOVUPS 64(DI), Y12
	VMOVUPS 96(DI), Y13
	VFMADD231PS Y12, Y11, Y2
	VFMADD231PS Y13, Y11, Y3

	// K + 2
	VBROADCASTSS 8(SI), Y8
	VMOVUPS 128(DI), Y9
	VMOVUPS 160(DI), Y10
	VFMADD231PS Y9, Y8, Y4
	VFMADD231PS Y10, Y8, Y5

	// K + 3
	VBROADCASTSS 12(SI), Y11
	VMOVUPS 192(DI), Y12
	VMOVUPS 224(DI), Y13
	VFMADD231PS Y12, Y11, Y6
	VFMADD231PS Y13, Y11, Y7

	ADDQ $16, SI
	ADDQ $256, DI
	DECQ CX
	JNZ k_quad_avx2_m1_f32

	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y6, Y4, Y4
	VADDPS Y7, Y5, Y5
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1

k_rem_avx2_m1_f32:
	MOVQ R12, CX
	ANDQ $3, CX
	JZ store_strip_avx2_m1_f32

k_single_avx2_m1_f32:
	VBROADCASTSS (SI), Y8
	VMOVUPS (DI), Y9
	VMOVUPS 32(DI), Y10
	VFMADD231PS Y9, Y8, Y0
	VFMADD231PS Y10, Y8, Y1
	ADDQ $4, SI
	ADDQ $64, DI
	DECQ CX
	JNZ k_single_avx2_m1_f32

store_strip_avx2_m1_f32:
	TESTB R15, R15
	JNZ store_strip_accum_avx2_m1_f32

	VMOVUPS Y0, (R10)
	VMOVUPS Y1, 32(R10)
	JMP next_strip_avx2_m1_f32

store_strip_accum_avx2_m1_f32:
	VADDPS (R10), Y0, Y0
	VMOVUPS Y0, (R10)
	VADDPS 32(R10), Y1, Y1
	VMOVUPS Y1, 32(R10)

next_strip_avx2_m1_f32:
	ADDQ R13, R9
	ADDQ $64, R10
	SUBQ $16, R14
	JMP loop_strip_avx2_m1_f32

done_avx2_m1_f32:
	VZEROUPPER
	RET

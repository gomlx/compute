// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2LargeKernelFloat64Asm(
//     packedLHS, packedRHS, packedOutput []float64,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int,
//     accumulate bool)
TEXT ·avx2LargeKernelFloat64Asm(SB), NOSPLIT, $0-120
	MOVQ packedLHS_base+0(FP), R8        // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9       // R9 = rhsBasePtr
	MOVQ packedOutput_base+48(FP), R10   // R10 = outBasePtr
	MOVQ rhsPanelCols+80(FP), R11        // R11 = outputStride (cols)
	MOVQ contractingLen+88(FP), R12      // R12 = contractingLen (K)
	MOVQ lhsActiveRows+96(FP), R13       // R13 = lhsActiveRows (M)
	MOVQ rhsActiveCols+104(FP), R14      // R14 = rhsActiveCols (N)
	MOVB accumulate+112(FP), R15         // R15 = accumulate (0 = overwrite, 1 = add)

	SHLQ $3, R11                         // R11 = outputStride in bytes

	XORQ AX, AX                          // AX = lhsRowIdx = 0

loop_lhs:
	CMPQ AX, R13
	JGE done

	XORQ BX, BX                          // BX = rhsColIdx = 0

loop_rhs:
	CMPQ BX, R14
	JGE next_lhs

	// 1. Zero all 8 accumulators (Y0..Y7)
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3
	VXORPD Y4, Y4, Y4
	VXORPD Y5, Y5, Y5
	VXORPD Y6, Y6, Y6
	VXORPD Y7, Y7, Y7

	// 2. Compute lhsPtr and rhsPtr
	// For 4-row LHS strip: strip has 4 doubles = 32 bytes per step.
	// SI = R8 + (lhsRowIdx * contractingLen * 8)
	MOVQ AX, SI
	IMULQ R12, SI
	SHLQ $3, SI
	ADDQ R8, SI

	// For 8-col RHS strip: strip has 8 doubles = 64 bytes per step.
	// DI = R9 + (rhsColIdx * contractingLen * 8)
	MOVQ BX, DI
	IMULQ R12, DI
	SHLQ $3, DI
	ADDQ R9, DI

	// 3. K-loop with 2-stage ping-pong pipeline (unroll by 2)
	MOVQ R12, CX                         // CX = contractingLen
	TESTQ CX, CX
	JLE store_output

	SHRQ $1, CX                          // CX = pairs = contractingLen / 2
	JZ k_odd

	// Prime Buffer A for Step 0 outside the loop
	VMOVDQU (DI), Y8                     // RHS col 0..3
	VMOVDQU 32(DI), Y9                   // RHS col 4..7
	VBROADCASTSD (SI), Y10               // LHS row 0
	VBROADCASTSD 8(SI), Y11              // LHS row 1
	VBROADCASTSD 16(SI), Y12             // LHS row 2
	VBROADCASTSD 24(SI), Y13             // LHS row 3

	ADDQ $32, SI
	ADDQ $64, DI

	DECQ CX
	JZ k_last_pair

	PCALIGN $64
k_pair_loop:
	// Step A: FMAs using Buffer A (Y8, Y9) while loading Buffer B (Y14, Y15, Y10..Y13)
	VFMADD231PD Y8, Y10, Y0
	VMOVDQU (DI), Y14
	VFMADD231PD Y9, Y10, Y1
	VBROADCASTSD (SI), Y10

	VFMADD231PD Y8, Y11, Y2
	VMOVDQU 32(DI), Y15
	VFMADD231PD Y9, Y11, Y3
	VBROADCASTSD 8(SI), Y11

	VFMADD231PD Y8, Y12, Y4
	VFMADD231PD Y9, Y12, Y5
	VBROADCASTSD 16(SI), Y12

	VFMADD231PD Y8, Y13, Y6
	VFMADD231PD Y9, Y13, Y7
	VBROADCASTSD 24(SI), Y13

	ADDQ $32, SI
	ADDQ $64, DI

	// Step B: FMAs using Buffer B (Y14, Y15) while loading next Buffer A (Y8, Y9, Y10..Y13)
	VFMADD231PD Y14, Y10, Y0
	VMOVDQU (DI), Y8
	VFMADD231PD Y15, Y10, Y1
	VBROADCASTSD (SI), Y10

	VFMADD231PD Y14, Y11, Y2
	VMOVDQU 32(DI), Y9
	VFMADD231PD Y15, Y11, Y3
	VBROADCASTSD 8(SI), Y11

	VFMADD231PD Y14, Y12, Y4
	VFMADD231PD Y15, Y12, Y5
	VBROADCASTSD 16(SI), Y12

	VFMADD231PD Y14, Y13, Y6
	VFMADD231PD Y15, Y13, Y7
	VBROADCASTSD 24(SI), Y13

	ADDQ $32, SI
	ADDQ $64, DI

	DECQ CX
	JNZ k_pair_loop

k_last_pair:
	// Step A: FMAs using Buffer A (Y8, Y9) while loading Buffer B (Y14, Y15, Y10..Y13)
	VFMADD231PD Y8, Y10, Y0
	VMOVDQU (DI), Y14
	VFMADD231PD Y9, Y10, Y1
	VBROADCASTSD (SI), Y10

	VFMADD231PD Y8, Y11, Y2
	VMOVDQU 32(DI), Y15
	VFMADD231PD Y9, Y11, Y3
	VBROADCASTSD 8(SI), Y11

	VFMADD231PD Y8, Y12, Y4
	VFMADD231PD Y9, Y12, Y5
	VBROADCASTSD 16(SI), Y12

	VFMADD231PD Y8, Y13, Y6
	VFMADD231PD Y9, Y13, Y7
	VBROADCASTSD 24(SI), Y13

	ADDQ $32, SI
	ADDQ $64, DI

	// Step B: Pure FMAs using Buffer B (Y14, Y15) - NO further memory loads!
	VFMADD231PD Y14, Y10, Y0
	VFMADD231PD Y15, Y10, Y1
	VFMADD231PD Y14, Y11, Y2
	VFMADD231PD Y15, Y11, Y3
	VFMADD231PD Y14, Y12, Y4
	VFMADD231PD Y15, Y12, Y5
	VFMADD231PD Y14, Y13, Y6
	VFMADD231PD Y15, Y13, Y7

k_odd:
	TESTQ $1, R12
	JZ store_output

	VMOVDQU (DI), Y8
	VMOVDQU 32(DI), Y9
	VBROADCASTSD (SI), Y10
	VBROADCASTSD 8(SI), Y11
	VBROADCASTSD 16(SI), Y12
	VBROADCASTSD 24(SI), Y13

	VFMADD231PD Y8, Y10, Y0
	VFMADD231PD Y9, Y10, Y1
	VFMADD231PD Y8, Y11, Y2
	VFMADD231PD Y9, Y11, Y3
	VFMADD231PD Y8, Y12, Y4
	VFMADD231PD Y9, Y12, Y5
	VFMADD231PD Y8, Y13, Y6
	VFMADD231PD Y9, Y13, Y7

store_output:
	MOVQ AX, DX
	IMULQ R11, DX
	MOVQ BX, CX
	SHLQ $3, CX
	ADDQ CX, DX
	LEAQ (R10)(DX*1), DX

	TESTB R15, R15
	JNZ store_output_accum

	// Row 0
	VMOVDQU Y0, (DX)
	VMOVDQU Y1, 32(DX)
	// Row 1
	ADDQ R11, DX
	VMOVDQU Y2, (DX)
	VMOVDQU Y3, 32(DX)
	// Row 2
	ADDQ R11, DX
	VMOVDQU Y4, (DX)
	VMOVDQU Y5, 32(DX)
	// Row 3
	ADDQ R11, DX
	VMOVDQU Y6, (DX)
	VMOVDQU Y7, 32(DX)

	ADDQ $8, BX                          // rhsColIdx += 8
	JMP loop_rhs

store_output_accum:
	// Row 0
	VADDPD (DX), Y0, Y0
	VMOVDQU Y0, (DX)
	VADDPD 32(DX), Y1, Y1
	VMOVDQU Y1, 32(DX)
	// Row 1
	ADDQ R11, DX
	VADDPD (DX), Y2, Y2
	VMOVDQU Y2, (DX)
	VADDPD 32(DX), Y3, Y3
	VMOVDQU Y3, 32(DX)
	// Row 2
	ADDQ R11, DX
	VADDPD (DX), Y4, Y4
	VMOVDQU Y4, (DX)
	VADDPD 32(DX), Y5, Y5
	VMOVDQU Y5, 32(DX)
	// Row 3
	ADDQ R11, DX
	VADDPD (DX), Y6, Y6
	VMOVDQU Y6, (DX)
	VADDPD 32(DX), Y7, Y7
	VMOVDQU Y7, 32(DX)

	ADDQ $8, BX                          // rhsColIdx += 8
	JMP loop_rhs

next_lhs:
	ADDQ $4, AX                          // lhsRowIdx += 4
	JMP loop_lhs

done:
	VZEROUPPER
	RET

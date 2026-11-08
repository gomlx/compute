// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2LargeKernelFloat32Asm(
//     packedLHS, packedRHS, packedOutput []float32,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int,
//     accumulate bool)
TEXT ·avx2LargeKernelFloat32Asm(SB), NOSPLIT, $0-120
	MOVQ packedLHS_base+0(FP), R8        // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9       // R9 = rhsBasePtr
	MOVQ packedOutput_base+48(FP), R10   // R10 = outBasePtr
	MOVQ rhsPanelCols+80(FP), R11        // R11 = outputStride (cols)
	MOVQ contractingLen+88(FP), R12      // R12 = contractingLen (K)
	MOVQ lhsActiveRows+96(FP), R13       // R13 = lhsActiveRows (M)
	MOVQ rhsActiveCols+104(FP), R14      // R14 = rhsActiveCols (N)
	MOVB accumulate+112(FP), R15         // R15 = accumulate (0 = overwrite, 1 = add)

	SHLQ $2, R11                         // R11 = outputStride in bytes

	XORQ BX, BX                          // BX = rhsColIdx = 0

loop_rhs:
	CMPQ BX, R14
	JGE done

	XORQ AX, AX                          // AX = lhsRowIdx = 0

loop_lhs:
	CMPQ AX, R13
	JGE next_rhs

	// 1. Zero all 8 accumulators (Y0..Y7)
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	// 2. Compute lhsPtr and rhsPtr
	// For 4-row LHS strip: strip has 4 floats = 16 bytes per step.
	// SI = R8 + (lhsRowIdx * contractingLen * 4)
	MOVQ AX, SI
	IMULQ R12, SI
	SHLQ $2, SI
	ADDQ R8, SI

	// For 16-col RHS strip: strip has 16 floats = 64 bytes per step.
	// DI = R9 + (rhsColIdx * contractingLen * 4)
	MOVQ BX, DI
	IMULQ R12, DI
	SHLQ $2, DI
	ADDQ R9, DI

	// 3. K-loop with 4x unrolled 2-stage ping-pong pipeline
	MOVQ R12, CX                         // CX = contractingLen
	TESTQ CX, CX
	JLE store_output

	SHRQ $2, CX                          // CX = quads = contractingLen / 4
	JZ k_rem

	// Prime Buffer A (Step 0) outside the loop
	VMOVUPS 0(DI), Y8                    // RHS_A col 0..7
	VMOVUPS 32(DI), Y9                   // RHS_A col 8..15
	VBROADCASTSS 0(SI), Y12              // LHS_A row 0
	VBROADCASTSS 4(SI), Y13              // LHS_A row 1

	DECQ CX
	JZ k_last_quad

	PCALIGN $64
k_quad_loop:
	// --- Step 0: Compute A (rows 0..1), Load B (RHS + rows 0..1), Load A (rows 2..3) ---
	VFMADD231PS Y8, Y12, Y0
	VFMADD231PS Y9, Y12, Y1
	VBROADCASTSS 8(SI), Y12              // LHS_A row 2
	VFMADD231PS Y8, Y13, Y2
	VFMADD231PS Y9, Y13, Y3
	VBROADCASTSS 12(SI), Y13             // LHS_A row 3

	VMOVUPS 64(DI), Y10                  // RHS_B col 0..7 (Step 1)
	VMOVUPS 96(DI), Y11                  // RHS_B col 8..15 (Step 1)
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VBROADCASTSS 16(SI), Y14             // LHS_B row 0 (Step 1)
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS 20(SI), Y15             // LHS_B row 1 (Step 1)

	// --- Step 1: Compute B (rows 0..1), Load A (RHS + rows 0..1), Load B (rows 2..3) ---
	VFMADD231PS Y10, Y14, Y0
	VFMADD231PS Y11, Y14, Y1
	VBROADCASTSS 24(SI), Y14             // LHS_B row 2
	VFMADD231PS Y10, Y15, Y2
	VFMADD231PS Y11, Y15, Y3
	VBROADCASTSS 28(SI), Y15             // LHS_B row 3

	VMOVUPS 128(DI), Y8                  // RHS_A col 0..7 (Step 2)
	VMOVUPS 160(DI), Y9                  // RHS_A col 8..15 (Step 2)
	VFMADD231PS Y10, Y14, Y4
	VFMADD231PS Y11, Y14, Y5
	VBROADCASTSS 32(SI), Y12             // LHS_A row 0 (Step 2)
	VFMADD231PS Y10, Y15, Y6
	VFMADD231PS Y11, Y15, Y7
	VBROADCASTSS 36(SI), Y13             // LHS_A row 1 (Step 2)

	// --- Step 2: Compute A (rows 0..1), Load B (RHS + rows 0..1), Load A (rows 2..3) ---
	VFMADD231PS Y8, Y12, Y0
	VFMADD231PS Y9, Y12, Y1
	VBROADCASTSS 40(SI), Y12             // LHS_A row 2
	VFMADD231PS Y8, Y13, Y2
	VFMADD231PS Y9, Y13, Y3
	VBROADCASTSS 44(SI), Y13             // LHS_A row 3

	VMOVUPS 192(DI), Y10                 // RHS_B col 0..7 (Step 3)
	VMOVUPS 224(DI), Y11                 // RHS_B col 8..15 (Step 3)
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VBROADCASTSS 48(SI), Y14             // LHS_B row 0 (Step 3)
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS 52(SI), Y15             // LHS_B row 1 (Step 3)

	// --- Step 3: Compute B (rows 0..1), Load A for next quad (RHS + rows 0..1), Load B (rows 2..3) ---
	VFMADD231PS Y10, Y14, Y0
	VFMADD231PS Y11, Y14, Y1
	VBROADCASTSS 56(SI), Y14             // LHS_B row 2
	VFMADD231PS Y10, Y15, Y2
	VFMADD231PS Y11, Y15, Y3
	VBROADCASTSS 60(SI), Y15             // LHS_B row 3

	VMOVUPS 256(DI), Y8                  // RHS_A col 0..7 (Next Step 0)
	VMOVUPS 288(DI), Y9                  // RHS_A col 8..15 (Next Step 0)
	VFMADD231PS Y10, Y14, Y4
	VFMADD231PS Y11, Y14, Y5
	VBROADCASTSS 64(SI), Y12             // LHS_A row 0 (Next Step 0)
	VFMADD231PS Y10, Y15, Y6
	VFMADD231PS Y11, Y15, Y7
	VBROADCASTSS 68(SI), Y13             // LHS_A row 1 (Next Step 0)

	ADDQ $64, SI
	ADDQ $256, DI

	DECQ CX
	JNZ k_quad_loop

k_last_quad:
	// --- Step 0 of last quad ---
	VFMADD231PS Y8, Y12, Y0
	VFMADD231PS Y9, Y12, Y1
	VBROADCASTSS 8(SI), Y12
	VFMADD231PS Y8, Y13, Y2
	VFMADD231PS Y9, Y13, Y3
	VBROADCASTSS 12(SI), Y13

	VMOVUPS 64(DI), Y10
	VMOVUPS 96(DI), Y11
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VBROADCASTSS 16(SI), Y14
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS 20(SI), Y15

	// --- Step 1 of last quad ---
	VFMADD231PS Y10, Y14, Y0
	VFMADD231PS Y11, Y14, Y1
	VBROADCASTSS 24(SI), Y14
	VFMADD231PS Y10, Y15, Y2
	VFMADD231PS Y11, Y15, Y3
	VBROADCASTSS 28(SI), Y15

	VMOVUPS 128(DI), Y8
	VMOVUPS 160(DI), Y9
	VFMADD231PS Y10, Y14, Y4
	VFMADD231PS Y11, Y14, Y5
	VBROADCASTSS 32(SI), Y12
	VFMADD231PS Y10, Y15, Y6
	VFMADD231PS Y11, Y15, Y7
	VBROADCASTSS 36(SI), Y13

	// --- Step 2 of last quad ---
	VFMADD231PS Y8, Y12, Y0
	VFMADD231PS Y9, Y12, Y1
	VBROADCASTSS 40(SI), Y12
	VFMADD231PS Y8, Y13, Y2
	VFMADD231PS Y9, Y13, Y3
	VBROADCASTSS 44(SI), Y13

	VMOVUPS 192(DI), Y10
	VMOVUPS 224(DI), Y11
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VBROADCASTSS 48(SI), Y14
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS 52(SI), Y15

	// --- Step 3 of last quad (no next-quad preload) ---
	VFMADD231PS Y10, Y14, Y0
	VFMADD231PS Y11, Y14, Y1
	VBROADCASTSS 56(SI), Y14
	VFMADD231PS Y10, Y15, Y2
	VFMADD231PS Y11, Y15, Y3
	VBROADCASTSS 60(SI), Y15
	VFMADD231PS Y10, Y14, Y4
	VFMADD231PS Y11, Y14, Y5
	VFMADD231PS Y10, Y15, Y6
	VFMADD231PS Y11, Y15, Y7

	ADDQ $64, SI
	ADDQ $256, DI

k_rem:
	MOVQ R12, CX
	ANDQ $3, CX
	JZ store_output

k_rem_loop:
	VMOVUPS 0(DI), Y8
	VMOVUPS 32(DI), Y9
	VBROADCASTSS 0(SI), Y12
	VBROADCASTSS 4(SI), Y13
	VFMADD231PS Y8, Y12, Y0
	VFMADD231PS Y9, Y12, Y1
	VBROADCASTSS 8(SI), Y14
	VFMADD231PS Y8, Y13, Y2
	VFMADD231PS Y9, Y13, Y3
	VBROADCASTSS 12(SI), Y15
	VFMADD231PS Y8, Y14, Y4
	VFMADD231PS Y9, Y14, Y5
	VFMADD231PS Y8, Y15, Y6
	VFMADD231PS Y9, Y15, Y7

	ADDQ $16, SI
	ADDQ $64, DI
	DECQ CX
	JNZ k_rem_loop

store_output:
	// 4. Write or Accumulate to PackedOutput
	// outRowBase = lhsRowIdx * outputStride + rhsColIdx * 4
	MOVQ AX, DX
	IMULQ R11, DX
	MOVQ BX, CX
	SHLQ $2, CX
	ADDQ CX, DX
	LEAQ (R10)(DX*1), DX                 // DX = &packedOutput[outRowBase]

	TESTB R15, R15
	JNZ store_output_accum

	// Overwrite (accumulate == false)
	// Row 0
	VMOVUPS Y0, 0(DX)
	VMOVUPS Y1, 32(DX)
	// Row 1
	ADDQ R11, DX
	VMOVUPS Y2, 0(DX)
	VMOVUPS Y3, 32(DX)
	// Row 2
	ADDQ R11, DX
	VMOVUPS Y4, 0(DX)
	VMOVUPS Y5, 32(DX)
	// Row 3
	ADDQ R11, DX
	VMOVUPS Y6, 0(DX)
	VMOVUPS Y7, 32(DX)

	ADDQ $4, AX                          // lhsRowIdx += 4
	JMP loop_lhs

store_output_accum:
	// Accumulate (accumulate == true)
	// Row 0
	VADDPS 0(DX), Y0, Y0
	VMOVUPS Y0, 0(DX)
	VADDPS 32(DX), Y1, Y1
	VMOVUPS Y1, 32(DX)
	// Row 1
	ADDQ R11, DX
	VADDPS 0(DX), Y2, Y2
	VMOVUPS Y2, 0(DX)
	VADDPS 32(DX), Y3, Y3
	VMOVUPS Y3, 32(DX)
	// Row 2
	ADDQ R11, DX
	VADDPS 0(DX), Y4, Y4
	VMOVUPS Y4, 0(DX)
	VADDPS 32(DX), Y5, Y5
	VMOVUPS Y5, 32(DX)
	// Row 3
	ADDQ R11, DX
	VADDPS 0(DX), Y6, Y6
	VMOVUPS Y6, 0(DX)
	VADDPS 32(DX), Y7, Y7
	VMOVUPS Y7, 32(DX)

	ADDQ $4, AX                          // lhsRowIdx += 4
	JMP loop_lhs

next_rhs:
	ADDQ $16, BX                         // rhsColIdx += 16
	JMP loop_rhs

done:
	VZEROUPPER
	RET

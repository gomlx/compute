// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2PackLHSKernelRows4Float64Asm(
//     lhs, panel []float64,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx2PackLHSKernelRows4Float64Asm(SB), NOSPLIT, $0-88
	MOVQ lhs_base+0(FP), R8              // R8 = lhsBasePtr
	MOVQ panel_base+24(FP), R9           // R9 = panelBasePtr
	MOVQ lhsRowStart+48(FP), R10         // R10 = lhsRowStart
	MOVQ lhsColStart+56(FP), R11         // R11 = lhsColStart
	MOVQ lhsCols+64(FP), R12             // R12 = lhsCols
	MOVQ copyRows+72(FP), R13            // R13 = copyRows
	MOVQ contractingCols+80(FP), R14     // R14 = contractingCols

	CMPQ R13, $4
	JL done
	TESTQ R14, R14
	JLE done

	SHLQ $3, R12                         // R12 = lhsStrideBytes = lhsCols * 8
	SHLQ $3, R11                         // R11 = lhsColStartBytes = lhsColStart * 8

	// R15 = contractingCols4 = contractingCols & ~3
	MOVQ R14, R15
	ANDQ $~3, R15

	XORQ AX, AX                          // AX = stripRowIdx = 0
	ANDQ $~3, R13                        // R13 = fullStripLimit = copyRows & ~3

loop_strip:
	CMPQ AX, R13
	JGE done

	// Compute base pointer for row 0 of this strip:
	// SI = lhsBasePtr + (lhsRowStart + stripRowIdx) * lhsStrideBytes + lhsColStartBytes
	MOVQ R10, SI
	ADDQ AX, SI
	IMULQ R12, SI
	ADDQ R8, SI
	ADDQ R11, SI

	LEAQ (SI)(R12*1), DX                 // DX = row 1
	LEAQ (DX)(R12*1), DI                 // DI = row 2
	LEAQ (DI)(R12*1), CX                 // CX = row 3

	XORQ BX, BX                          // BX = colIdx = 0

	TESTQ R15, R15
	JZ check_tail_cols

	PCALIGN $32
loop_cols:
	CMPQ BX, R15
	JGE check_tail_cols

	// Load 4 float64s (32 bytes) from rows 0, 1, 2, 3
	VMOVDQU (SI), Y0
	VMOVDQU (DX), Y1
	VMOVDQU (DI), Y2
	VMOVDQU (CX), Y3

	// Stage 1: 64-bit unpack (intra-128-bit lane)
	VUNPCKLPD Y1, Y0, Y4                 // [a0 b0 | a2 b2]
	VUNPCKHPD Y1, Y0, Y5                 // [a1 b1 | a3 b3]
	VUNPCKLPD Y3, Y2, Y6                 // [c0 d0 | c2 d2]
	VUNPCKHPD Y3, Y2, Y7                 // [c1 d1 | c3 d3]

	// Stage 2: Cross-lane assemble into contiguous 4-element columns
	VPERM2F128 $0x20, Y6, Y4, Y8         // Col 0: [a0 b0 c0 d0]
	VPERM2F128 $0x20, Y7, Y5, Y9         // Col 1: [a1 b1 c1 d1]
	VPERM2F128 $0x31, Y6, Y4, Y10        // Col 2: [a2 b2 c2 d2]
	VPERM2F128 $0x31, Y7, Y5, Y11        // Col 3: [a3 b3 c3 d3]

	// Store 4 output columns (cols 0..3)
	VMOVDQU Y8, (R9)
	VMOVDQU Y9, 32(R9)
	VMOVDQU Y10, 64(R9)
	VMOVDQU Y11, 96(R9)

	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	ADDQ $32, CX
	ADDQ $128, R9                        // panelPtr += 128 bytes (16 float64s)
	ADDQ $4, BX
	JMP loop_cols

check_tail_cols:
	CMPQ BX, R14
	JGE next_strip

loop_tail_cols:
	VMOVSD (SI), X0
	VMOVSD X0, (R9)
	VMOVSD (DX), X0
	VMOVSD X0, 8(R9)
	VMOVSD (DI), X0
	VMOVSD X0, 16(R9)
	VMOVSD (CX), X0
	VMOVSD X0, 24(R9)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	ADDQ $8, CX
	ADDQ $32, R9
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_cols

next_strip:
	ADDQ $4, AX
	JMP loop_strip

done:
	VZEROUPPER
	RET

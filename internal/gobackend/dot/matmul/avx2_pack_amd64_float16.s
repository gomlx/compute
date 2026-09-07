// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2PackLHSKernelRows4BFloat16Asm(
//     lhs, panel []bfloat16.BFloat16,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx2PackLHSKernelRows4BFloat16Asm(SB), NOSPLIT, $0-88
	JMP ·avx2PackLHSKernelRows4Float16Asm(SB)

// func avx2PackLHSKernelRows4Float16Asm(
//     lhs, panel []float16.Float16,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx2PackLHSKernelRows4Float16Asm(SB), NOSPLIT, $0-88
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

	SHLQ $1, R12                         // R12 = lhsStrideBytes = lhsCols * 2
	SHLQ $1, R11                         // R11 = lhsColStartBytes = lhsColStart * 2

	// R15 = contractingCols16 = contractingCols & ~15
	MOVQ R14, R15
	ANDQ $~15, R15

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

	// Load 16 16-bit words (32 bytes) from rows 0, 1, 2, 3
	VMOVDQU (SI), Y0
	VMOVDQU (DX), Y1
	VMOVDQU (DI), Y2
	VMOVDQU (CX), Y3

	// Stage 1: 16-bit unpack (intra-128-bit lane)
	VPUNPCKLWD Y1, Y0, Y4
	VPUNPCKHWD Y1, Y0, Y5
	VPUNPCKLWD Y3, Y2, Y6
	VPUNPCKHWD Y3, Y2, Y7

	// Stage 2: 32-bit unpack (intra-128-bit lane)
	VPUNPCKLDQ Y6, Y4, Y8                // [Cols 0..1 | Cols 8..9]
	VPUNPCKHDQ Y6, Y4, Y9                // [Cols 2..3 | Cols 10..11]
	VPUNPCKLDQ Y7, Y5, Y10               // [Cols 4..5 | Cols 12..13]
	VPUNPCKHDQ Y7, Y5, Y11               // [Cols 6..7 | Cols 14..15]

	// Stage 3: Cross-lane assemble into contiguous 4-element columns
	VPERM2I128 $0x20, Y9, Y8, Y12        // [Cols 0..3]
	VPERM2I128 $0x20, Y11, Y10, Y13      // [Cols 4..7]
	VPERM2I128 $0x31, Y9, Y8, Y14        // [Cols 8..11]
	VPERM2I128 $0x31, Y11, Y10, Y15      // [Cols 12..15]

	// Store 4 groups of 4 columns (cols 0..15)
	VMOVDQU Y12, (R9)
	VMOVDQU Y13, 32(R9)
	VMOVDQU Y14, 64(R9)
	VMOVDQU Y15, 96(R9)

	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	ADDQ $32, CX
	ADDQ $128, R9                        // panelPtr += 128 bytes (64 float16s)
	ADDQ $16, BX
	JMP loop_cols

check_tail_cols:
	CMPQ BX, R14
	JGE next_strip

loop_tail_cols:
	MOVW (SI), R12
	MOVW R12, (R9)
	MOVW (DX), R12
	MOVW R12, 2(R9)
	MOVW (DI), R12
	MOVW R12, 4(R9)
	MOVW (CX), R12
	MOVW R12, 6(R9)

	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	ADDQ $2, CX
	ADDQ $8, R9
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_cols

next_strip:
	// Restore lhsCols * 2 into R12
	MOVQ lhsCols+64(FP), R12
	SHLQ $1, R12

	ADDQ $4, AX
	JMP loop_strip

done:
	VZEROUPPER
	RET

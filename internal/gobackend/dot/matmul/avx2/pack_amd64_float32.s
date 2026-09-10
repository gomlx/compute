// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2PackLHSKernelRows4Float32Asm(
//     lhs, panel []float32,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx2PackLHSKernelRows4Float32Asm(SB), NOSPLIT, $0-88
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

	SHLQ $2, R12                         // R12 = lhsStrideBytes = lhsCols * 4
	SHLQ $2, R11                         // R11 = lhsColStartBytes = lhsColStart * 4

	// R15 = contractingCols8 = contractingCols & ~7
	MOVQ R14, R15
	ANDQ $~7, R15

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

	// Load 8 float32s (32 bytes) from rows 0, 1, 2, 3
	VMOVDQU (SI), Y0
	VMOVDQU (DX), Y1
	VMOVDQU (DI), Y2
	VMOVDQU (CX), Y3

	// Stage 1: 32-bit unpack (intra-128-bit lane)
	VUNPCKLPS Y1, Y0, Y4
	VUNPCKHPS Y1, Y0, Y5
	VUNPCKLPS Y3, Y2, Y6
	VUNPCKHPS Y3, Y2, Y7

	// Stage 2: 64-bit unpack (intra-128-bit lane)
	VUNPCKLPD Y6, Y4, Y8                 // [Col 0 | Col 4]
	VUNPCKHPD Y6, Y4, Y9                 // [Col 1 | Col 5]
	VUNPCKLPD Y7, Y5, Y10                // [Col 2 | Col 6]
	VUNPCKHPD Y7, Y5, Y11                // [Col 3 | Col 7]

	// Stage 3: Cross-lane assemble into contiguous 4-element columns
	VPERM2F128 $0x20, Y9, Y8, Y12        // [Col 0 | Col 1]
	VPERM2F128 $0x20, Y11, Y10, Y13      // [Col 2 | Col 3]
	VPERM2F128 $0x31, Y9, Y8, Y14        // [Col 4 | Col 5]
	VPERM2F128 $0x31, Y11, Y10, Y15      // [Col 6 | Col 7]

	// Store 4 pairs of output columns (cols 0..7)
	VMOVDQU Y12, (R9)
	VMOVDQU Y13, 32(R9)
	VMOVDQU Y14, 64(R9)
	VMOVDQU Y15, 96(R9)

	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	ADDQ $32, CX
	ADDQ $128, R9                        // panelPtr += 128 bytes (32 floats)
	ADDQ $8, BX
	JMP loop_cols

check_tail_cols:
	CMPQ BX, R14
	JGE next_strip

loop_tail_cols:
	VMOVSS (SI), X0
	VMOVSS X0, (R9)
	VMOVSS (DX), X0
	VMOVSS X0, 4(R9)
	VMOVSS (DI), X0
	VMOVSS X0, 8(R9)
	VMOVSS (CX), X0
	VMOVSS X0, 12(R9)

	ADDQ $4, SI
	ADDQ $4, DX
	ADDQ $4, DI
	ADDQ $4, CX
	ADDQ $16, R9
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_cols

next_strip:
	ADDQ $4, AX
	JMP loop_strip

done:
	VZEROUPPER
	RET

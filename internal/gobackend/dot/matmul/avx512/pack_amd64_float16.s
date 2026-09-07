// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512PackLHSKernelRows4BFloat16Asm(
//     lhs, panel []bfloat16.BFloat16,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx512PackLHSKernelRows4BFloat16Asm(SB), NOSPLIT, $0-88
	JMP ·avx512PackLHSKernelRows4Float16Asm(SB)

// func avx512PackLHSKernelRows4Float16Asm(
//     lhs, panel []float16.Float16,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx512PackLHSKernelRows4Float16Asm(SB), NOSPLIT, $0-88
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

	// R15 = contractingCols32 = contractingCols & ~31
	MOVQ R14, R15
	ANDQ $~31, R15

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

	// Load 32 16-bit floats (64 bytes) from rows 0, 1, 2, 3
	VMOVDQU16 (SI), Z0
	VMOVDQU16 (DX), Z1
	VMOVDQU16 (DI), Z2
	VMOVDQU16 (CX), Z3

	// Stage 1: 16-bit unpack (intra-128-bit lane)
	VPUNPCKLWD Z1, Z0, Z4
	VPUNPCKHWD Z1, Z0, Z5
	VPUNPCKLWD Z3, Z2, Z6
	VPUNPCKHWD Z3, Z2, Z7

	// Stage 2: 32-bit unpack (intra-128-bit lane)
	VPUNPCKLDQ Z6, Z4, Z8
	VPUNPCKHDQ Z6, Z4, Z9
	VPUNPCKLDQ Z7, Z5, Z10
	VPUNPCKHDQ Z7, Z5, Z11

	// Stage 3: 128-bit cross-lane shuffle
	VSHUFI32X4 $0x44, Z9, Z8, Z12
	VSHUFI32X4 $0x44, Z11, Z10, Z13
	VSHUFI32X4 $0xEE, Z9, Z8, Z14
	VSHUFI32X4 $0xEE, Z11, Z10, Z15

	// Stage 4: Assemble Out0..Out3
	// Z16: cols 0..7
	// Z17: cols 8..15
	// Z18: cols 16..23
	// Z19: cols 24..31
	VSHUFI32X4 $0x88, Z13, Z12, Z16
	VSHUFI32X4 $0xDD, Z13, Z12, Z17
	VSHUFI32X4 $0x88, Z15, Z14, Z18
	VSHUFI32X4 $0xDD, Z15, Z14, Z19

	// Store 4 output strips (cols 0..31)
	VMOVDQU16 Z16, (R9)
	VMOVDQU16 Z17, 64(R9)
	VMOVDQU16 Z18, 128(R9)
	VMOVDQU16 Z19, 192(R9)

	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	ADDQ $64, CX
	ADDQ $256, R9                        // panelPtr += 256 bytes (32 cols * 4 rows * 2 bytes)
	ADDQ $32, BX
	JMP loop_cols

check_tail_cols:
	CMPQ BX, R14
	JGE next_strip

loop_tail_cols:
	MOVW (SI), R15
	MOVW R15, (R9)
	MOVW (DX), R15
	MOVW R15, 2(R9)
	MOVW (DI), R15
	MOVW R15, 4(R9)
	MOVW (CX), R15
	MOVW R15, 6(R9)

	ADDQ $2, SI
	ADDQ $2, DX
	ADDQ $2, DI
	ADDQ $2, CX
	ADDQ $8, R9                          // 4 rows * 2 bytes
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_cols

	// Restore R15 for next strip:
	MOVQ R14, R15
	ANDQ $~31, R15

next_strip:
	ADDQ $4, AX
	JMP loop_strip

done:
	VZEROUPPER
	RET

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512PackLHSKernelRows4Float32Asm(
//     lhs, panel []float32,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx512PackLHSKernelRows4Float32Asm(SB), NOSPLIT, $0-88
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

	// Load 16 float32s (64 bytes) from rows 0, 1, 2, 3
	VMOVDQU32 (SI), Z0
	VMOVDQU32 (DX), Z1
	VMOVDQU32 (DI), Z2
	VMOVDQU32 (CX), Z3

	// Stage 1: 32-bit unpack (intra-128-bit lane)
	VUNPCKLPS Z1, Z0, Z4
	VUNPCKHPS Z1, Z0, Z5
	VUNPCKLPS Z3, Z2, Z6
	VUNPCKHPS Z3, Z2, Z7

	// Stage 2: 64-bit unpack (intra-128-bit lane)
	VUNPCKLPD Z6, Z4, Z8
	VUNPCKHPD Z6, Z4, Z9
	VUNPCKLPD Z7, Z5, Z10
	VUNPCKHPD Z7, Z5, Z11

	// Stage 3: 128-bit cross-lane shuffle
	VSHUFI32X4 $0x44, Z9, Z8, Z12
	VSHUFI32X4 $0x44, Z11, Z10, Z13
	VSHUFI32X4 $0xEE, Z9, Z8, Z14
	VSHUFI32X4 $0xEE, Z11, Z10, Z15

	// Stage 4: Assemble Out0..Out3
	VSHUFI32X4 $0x88, Z13, Z12, Z16
	VSHUFI32X4 $0xDD, Z13, Z12, Z17
	VSHUFI32X4 $0x88, Z15, Z14, Z18
	VSHUFI32X4 $0xDD, Z15, Z14, Z19

	// Store 4 output strips (cols 0..15)
	VMOVDQU32 Z16, (R9)
	VMOVDQU32 Z17, 64(R9)
	VMOVDQU32 Z18, 128(R9)
	VMOVDQU32 Z19, 192(R9)

	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	ADDQ $64, CX
	ADDQ $256, R9                        // panelPtr += 256 bytes
	ADDQ $16, BX
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

// func avx512PackLHSKernelRows8Float32Asm(
//     lhs, panel []float32,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx512PackLHSKernelRows8Float32Asm(SB), NOSPLIT, $0-88
	MOVQ lhs_base+0(FP), R8              // R8 = lhsBasePtr
	MOVQ panel_base+24(FP), R9           // R9 = panelBasePtr
	MOVQ lhsRowStart+48(FP), R10         // R10 = lhsRowStart
	MOVQ lhsColStart+56(FP), R11         // R11 = lhsColStart
	MOVQ lhsCols+64(FP), R12             // R12 = lhsCols
	MOVQ copyRows+72(FP), R13            // R13 = copyRows
	MOVQ contractingCols+80(FP), R14     // R14 = contractingCols

	CMPQ R13, $8
	JL done8
	TESTQ R14, R14
	JLE done8

	SHLQ $2, R12                         // R12 = lhsStrideBytes = lhsCols * 4
	SHLQ $2, R11                         // R11 = lhsColStartBytes = lhsColStart * 4

	// R15 = contractingCols16 = contractingCols & ~15
	MOVQ R14, R15
	ANDQ $~15, R15

	XORQ AX, AX                          // AX = stripRowIdx = 0
	ANDQ $~7, R13                        // R13 = fullStripLimit = copyRows & ~7

loop_strip8:
	CMPQ AX, R13
	JGE done8

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

	// BP = 4 * lhsStrideBytes (used to access rows 4, 5, 6, 7)
	LEAQ (R12*4), BP

	XORQ BX, BX                          // BX = colIdx = 0

	TESTQ R15, R15
	JZ check_tail_cols8

	PCALIGN $32
loop_cols8:
	CMPQ BX, R15
	JGE check_tail_cols8

	// Load 16 float32s from rows 0, 1, 2, 3
	VMOVDQU32 (SI), Z0
	VMOVDQU32 (DX), Z1
	VMOVDQU32 (DI), Z2
	VMOVDQU32 (CX), Z3

	// Stage 1 for rows 0..3: 32-bit unpack (intra-128-bit lane)
	VUNPCKLPS Z1, Z0, Z8
	VUNPCKHPS Z1, Z0, Z9
	VUNPCKLPS Z3, Z2, Z10
	VUNPCKHPS Z3, Z2, Z11

	// Stage 2 for rows 0..3: 64-bit unpack (intra-128-bit lane)
	VUNPCKLPD Z10, Z8, Z12
	VUNPCKHPD Z10, Z8, Z13
	VUNPCKLPD Z11, Z9, Z14
	VUNPCKHPD Z11, Z9, Z15

	// Stage 3 for rows 0..3: 128-bit cross-lane shuffle
	VSHUFI32X4 $0x44, Z13, Z12, Z16
	VSHUFI32X4 $0x44, Z15, Z14, Z17
	VSHUFI32X4 $0xEE, Z13, Z12, Z18
	VSHUFI32X4 $0xEE, Z15, Z14, Z19

	// Stage 4 for rows 0..3: Assemble Out0..Out3
	VSHUFI32X4 $0x88, Z17, Z16, Z24      // Z24 = cols 0..3, rows 0..3
	VSHUFI32X4 $0xDD, Z17, Z16, Z25      // Z25 = cols 4..7, rows 0..3
	VSHUFI32X4 $0x88, Z19, Z18, Z26      // Z26 = cols 8..11, rows 0..3
	VSHUFI32X4 $0xDD, Z19, Z18, Z27      // Z27 = cols 12..15, rows 0..3

	// Load 16 float32s from rows 4, 5, 6, 7
	VMOVDQU32 (0)(SI)(BP*1), Z4
	VMOVDQU32 (0)(DX)(BP*1), Z5
	VMOVDQU32 (0)(DI)(BP*1), Z6
	VMOVDQU32 (0)(CX)(BP*1), Z7

	// Stage 1 for rows 4..7: 32-bit unpack (intra-128-bit lane)
	VUNPCKLPS Z5, Z4, Z8
	VUNPCKHPS Z5, Z4, Z9
	VUNPCKLPS Z7, Z6, Z10
	VUNPCKHPS Z7, Z6, Z11

	// Stage 2 for rows 4..7: 64-bit unpack (intra-128-bit lane)
	VUNPCKLPD Z10, Z8, Z12
	VUNPCKHPD Z10, Z8, Z13
	VUNPCKLPD Z11, Z9, Z14
	VUNPCKHPD Z11, Z9, Z15

	// Stage 3 for rows 4..7: 128-bit cross-lane shuffle
	VSHUFI32X4 $0x44, Z13, Z12, Z16
	VSHUFI32X4 $0x44, Z15, Z14, Z17
	VSHUFI32X4 $0xEE, Z13, Z12, Z18
	VSHUFI32X4 $0xEE, Z15, Z14, Z19

	// Stage 4 for rows 4..7: Assemble Out4..Out7
	VSHUFI32X4 $0x88, Z17, Z16, Z28      // Z28 = cols 0..3, rows 4..7
	VSHUFI32X4 $0xDD, Z17, Z16, Z29      // Z29 = cols 4..7, rows 4..7
	VSHUFI32X4 $0x88, Z19, Z18, Z30      // Z30 = cols 8..11, rows 4..7
	VSHUFI32X4 $0xDD, Z19, Z18, Z31      // Z31 = cols 12..15, rows 4..7

	// Interleave rows 0..3 and rows 4..7 for each column group:
	// Cols 0..3 -> output strips cols 0, 1 (Z0) and cols 2, 3 (Z1)
	VSHUFI32X4 $0x44, Z28, Z24, Z0
	VSHUFI32X4 $0xD8, Z0, Z0, Z0
	VSHUFI32X4 $0xEE, Z28, Z24, Z1
	VSHUFI32X4 $0xD8, Z1, Z1, Z1

	// Cols 4..7 -> output strips cols 4, 5 (Z2) and cols 6, 7 (Z3)
	VSHUFI32X4 $0x44, Z29, Z25, Z2
	VSHUFI32X4 $0xD8, Z2, Z2, Z2
	VSHUFI32X4 $0xEE, Z29, Z25, Z3
	VSHUFI32X4 $0xD8, Z3, Z3, Z3

	// Cols 8..11 -> output strips cols 8, 9 (Z4) and cols 10, 11 (Z5)
	VSHUFI32X4 $0x44, Z30, Z26, Z4
	VSHUFI32X4 $0xD8, Z4, Z4, Z4
	VSHUFI32X4 $0xEE, Z30, Z26, Z5
	VSHUFI32X4 $0xD8, Z5, Z5, Z5

	// Cols 12..15 -> output strips cols 12, 13 (Z6) and cols 14, 15 (Z7)
	VSHUFI32X4 $0x44, Z31, Z27, Z6
	VSHUFI32X4 $0xD8, Z6, Z6, Z6
	VSHUFI32X4 $0xEE, Z31, Z27, Z7
	VSHUFI32X4 $0xD8, Z7, Z7, Z7

	// Store 8 output strips (512 bytes = 16 columns * 8 rows * 4 bytes)
	VMOVDQU32 Z0, (R9)
	VMOVDQU32 Z1, 64(R9)
	VMOVDQU32 Z2, 128(R9)
	VMOVDQU32 Z3, 192(R9)
	VMOVDQU32 Z4, 256(R9)
	VMOVDQU32 Z5, 320(R9)
	VMOVDQU32 Z6, 384(R9)
	VMOVDQU32 Z7, 448(R9)

	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	ADDQ $64, CX
	ADDQ $512, R9                        // panelPtr += 512 bytes
	ADDQ $16, BX
	JMP loop_cols8

check_tail_cols8:
	CMPQ BX, R14
	JGE next_strip8

loop_tail_cols8:
	VMOVSS (SI), X0
	VMOVSS X0, (R9)
	VMOVSS (DX), X0
	VMOVSS X0, 4(R9)
	VMOVSS (DI), X0
	VMOVSS X0, 8(R9)
	VMOVSS (CX), X0
	VMOVSS X0, 12(R9)
	VMOVSS (0)(SI)(BP*1), X0
	VMOVSS X0, 16(R9)
	VMOVSS (0)(DX)(BP*1), X0
	VMOVSS X0, 20(R9)
	VMOVSS (0)(DI)(BP*1), X0
	VMOVSS X0, 24(R9)
	VMOVSS (0)(CX)(BP*1), X0
	VMOVSS X0, 28(R9)

	ADDQ $4, SI
	ADDQ $4, DX
	ADDQ $4, DI
	ADDQ $4, CX
	ADDQ $32, R9
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_cols8

next_strip8:
	MOVQ lhs_base+0(FP), R8              // reload R8 = lhsBasePtr for next strip
	ADDQ $8, AX
	JMP loop_strip8

done8:
	VZEROUPPER
	RET


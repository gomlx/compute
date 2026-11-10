// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512PackLHSKernelRows4Float64Asm(
//     lhs, panel []float64,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx512PackLHSKernelRows4Float64Asm(SB), NOSPLIT, $0-88
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

	// Load 8 float64s (64 bytes) from rows 0, 1, 2, 3
	VMOVDQU64 (SI), Z0
	VMOVDQU64 (DX), Z1
	VMOVDQU64 (DI), Z2
	VMOVDQU64 (CX), Z3

	// Stage 1: 64-bit unpack (intra-128-bit lane)
	// Z4 = [A0, B0 | A2, B2 | A4, B4 | A6, B6]
	// Z5 = [A1, B1 | A3, B3 | A5, B5 | A7, B7]
	// Z6 = [C0, D0 | C2, D2 | C4, D4 | C6, D6]
	// Z7 = [C1, D1 | C3, D3 | C5, D5 | C7, D7]
	VUNPCKLPD Z1, Z0, Z4
	VUNPCKHPD Z1, Z0, Z5
	VUNPCKLPD Z3, Z2, Z6
	VUNPCKHPD Z3, Z2, Z7

	// Stage 2: 128-bit cross-lane shuffle
	// Z8  = [Lane0(Z4), Lane1(Z4), Lane0(Z6), Lane1(Z6)]
	// Z9  = [Lane0(Z5), Lane1(Z5), Lane0(Z7), Lane1(Z7)]
	// Z10 = [Lane2(Z4), Lane3(Z4), Lane2(Z6), Lane3(Z6)]
	// Z11 = [Lane2(Z5), Lane3(Z5), Lane2(Z7), Lane3(Z7)]
	VSHUFI64X2 $0x44, Z6, Z4, Z8
	VSHUFI64X2 $0x44, Z7, Z5, Z9
	VSHUFI64X2 $0xEE, Z6, Z4, Z10
	VSHUFI64X2 $0xEE, Z7, Z5, Z11

	// Stage 3: Assemble Out0..Out3
	// Z12 (Out0): [A0, B0, C0, D0, A1, B1, C1, D1] (cols 0, 1)
	// Z13 (Out1): [A2, B2, C2, D2, A3, B3, C3, D3] (cols 2, 3)
	// Z14 (Out2): [A4, B4, C4, D4, A5, B5, C5, D5] (cols 4, 5)
	// Z15 (Out3): [A6, B6, C6, D6, A7, B7, C7, D7] (cols 6, 7)
	VSHUFI64X2 $0x88, Z9, Z8, Z12
	VSHUFI64X2 $0xDD, Z9, Z8, Z13
	VSHUFI64X2 $0x88, Z11, Z10, Z14
	VSHUFI64X2 $0xDD, Z11, Z10, Z15

	// Store 4 output strips (cols 0..7)
	VMOVDQU64 Z12, (R9)
	VMOVDQU64 Z13, 64(R9)
	VMOVDQU64 Z14, 128(R9)
	VMOVDQU64 Z15, 192(R9)

	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	ADDQ $64, CX
	ADDQ $256, R9                        // panelPtr += 256 bytes (8 cols * 4 rows * 8 bytes)
	ADDQ $8, BX
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
	ADDQ $32, R9                         // 4 rows * 8 bytes
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_cols

next_strip:
	ADDQ $4, AX
	JMP loop_strip

done:
	VZEROUPPER
	RET

// func avx512PackLHSKernelRows8Float64Asm(
//     lhs, panel []float64,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
TEXT ·avx512PackLHSKernelRows8Float64Asm(SB), NOSPLIT, $0-88
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

	SHLQ $3, R12                         // R12 = lhsStrideBytes = lhsCols * 8
	SHLQ $3, R11                         // R11 = lhsColStartBytes = lhsColStart * 8

	// R15 = contractingCols8 = contractingCols & ~7
	MOVQ R14, R15
	ANDQ $~7, R15

	XORQ AX, AX                          // AX = stripRowIdx = 0
	ANDQ $~7, R13                        // R13 = fullStripLimit = copyRows & ~7

loop_strip8:
	CMPQ AX, R13
	JGE done8

	// Compute base pointer for row 0 of this strip:
	MOVQ R10, SI
	ADDQ AX, SI
	IMULQ R12, SI
	ADDQ R8, SI
	ADDQ R11, SI

	LEAQ (SI)(R12*1), DX                 // DX = row 1
	LEAQ (DX)(R12*1), DI                 // DI = row 2
	LEAQ (DI)(R12*1), CX                 // CX = row 3
	LEAQ (R12*4), BP                     // BP = 4 * lhsStrideBytes (rows 4..7)

	XORQ BX, BX                          // BX = colIdx = 0

	TESTQ R15, R15
	JZ check_tail_cols8

	PCALIGN $32
loop_cols8:
	CMPQ BX, R15
	JGE check_tail_cols8

	// Load 8 float64s from rows 0..3
	VMOVDQU64 (SI), Z0
	VMOVDQU64 (DX), Z1
	VMOVDQU64 (DI), Z2
	VMOVDQU64 (CX), Z3

	VUNPCKLPD Z1, Z0, Z8
	VUNPCKHPD Z1, Z0, Z9
	VUNPCKLPD Z3, Z2, Z10
	VUNPCKHPD Z3, Z2, Z11

	VSHUFI64X2 $0x44, Z10, Z8, Z12
	VSHUFI64X2 $0x44, Z11, Z9, Z13
	VSHUFI64X2 $0xEE, Z10, Z8, Z14
	VSHUFI64X2 $0xEE, Z11, Z9, Z15

	VSHUFI64X2 $0x88, Z13, Z12, Z24      // cols 0, 1 rows 0..3
	VSHUFI64X2 $0xDD, Z13, Z12, Z25      // cols 2, 3 rows 0..3
	VSHUFI64X2 $0x88, Z15, Z14, Z26      // cols 4, 5 rows 0..3
	VSHUFI64X2 $0xDD, Z15, Z14, Z27      // cols 6, 7 rows 0..3

	// Load 8 float64s from rows 4..7
	VMOVDQU64 (0)(SI)(BP*1), Z4
	VMOVDQU64 (0)(DX)(BP*1), Z5
	VMOVDQU64 (0)(DI)(BP*1), Z6
	VMOVDQU64 (0)(CX)(BP*1), Z7

	VUNPCKLPD Z5, Z4, Z8
	VUNPCKHPD Z5, Z4, Z9
	VUNPCKLPD Z7, Z6, Z10
	VUNPCKHPD Z7, Z6, Z11

	VSHUFI64X2 $0x44, Z10, Z8, Z12
	VSHUFI64X2 $0x44, Z11, Z9, Z13
	VSHUFI64X2 $0xEE, Z10, Z8, Z14
	VSHUFI64X2 $0xEE, Z11, Z9, Z15

	VSHUFI64X2 $0x88, Z13, Z12, Z28      // cols 0, 1 rows 4..7
	VSHUFI64X2 $0xDD, Z13, Z12, Z29      // cols 2, 3 rows 4..7
	VSHUFI64X2 $0x88, Z15, Z14, Z30      // cols 4, 5 rows 4..7
	VSHUFI64X2 $0xDD, Z15, Z14, Z31      // cols 6, 7 rows 4..7

	// Assemble 8-row columns 0..7 (each ZMM holds 1 column of 8 rows = 64 bytes)
	VSHUFI64X2 $0x44, Z28, Z24, Z0
	VSHUFI64X2 $0xEE, Z28, Z24, Z1
	VSHUFI64X2 $0x44, Z29, Z25, Z2
	VSHUFI64X2 $0xEE, Z29, Z25, Z3
	VSHUFI64X2 $0x44, Z30, Z26, Z4
	VSHUFI64X2 $0xEE, Z30, Z26, Z5
	VSHUFI64X2 $0x44, Z31, Z27, Z6
	VSHUFI64X2 $0xEE, Z31, Z27, Z7

	VMOVDQU64 Z0, (R9)
	VMOVDQU64 Z1, 64(R9)
	VMOVDQU64 Z2, 128(R9)
	VMOVDQU64 Z3, 192(R9)
	VMOVDQU64 Z4, 256(R9)
	VMOVDQU64 Z5, 320(R9)
	VMOVDQU64 Z6, 384(R9)
	VMOVDQU64 Z7, 448(R9)

	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	ADDQ $64, CX
	ADDQ $512, R9                        // 8 cols * 8 rows * 8 bytes
	ADDQ $8, BX
	JMP loop_cols8

check_tail_cols8:
	CMPQ BX, R14
	JGE next_strip8

loop_tail_cols8:
	VMOVSD (SI), X0
	VMOVSD X0, (R9)
	VMOVSD (DX), X0
	VMOVSD X0, 8(R9)
	VMOVSD (DI), X0
	VMOVSD X0, 16(R9)
	VMOVSD (CX), X0
	VMOVSD X0, 24(R9)
	VMOVSD (0)(SI)(BP*1), X0
	VMOVSD X0, 32(R9)
	VMOVSD (0)(DX)(BP*1), X0
	VMOVSD X0, 40(R9)
	VMOVSD (0)(DI)(BP*1), X0
	VMOVSD X0, 48(R9)
	VMOVSD (0)(CX)(BP*1), X0
	VMOVSD X0, 56(R9)

	ADDQ $8, SI
	ADDQ $8, DX
	ADDQ $8, DI
	ADDQ $8, CX
	ADDQ $64, R9
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_cols8

next_strip8:
	ADDQ $8, AX
	JMP loop_strip8

done8:
	VZEROUPPER
	RET

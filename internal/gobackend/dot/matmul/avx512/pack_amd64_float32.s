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

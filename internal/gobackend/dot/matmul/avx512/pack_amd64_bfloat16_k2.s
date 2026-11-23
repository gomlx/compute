// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

#include "textflag.h"

// ============================================================================
// K2-interleaved BFloat16 packers for AVX512_BF16 (VDPBF16PS)
//
// In K2-interleaved layout, consecutive pairs along K (2*p, 2*p+1) are packed
// as 32-bit units (two BFloat16 values), with any odd trailing K zero-padded
// in the upper 16 bits.
// ============================================================================

// func avx512PackLHSKernelRows8BFloat16K2Asm(
//     lhs, panel []bfloat16.BFloat16,
//     lhsRowStart, lhsColStart, lhsCols,
//     copyRows, contractingCols int)
//
// Packs full 8-row strips (copyRows & ^7) of bfloat16 LHS into
// [numStrips, kPairs, 8, 2] where kPairs = (contractingCols + 1) / 2.
TEXT ·avx512PackLHSKernelRows8BFloat16K2Asm(SB), NOSPLIT, $0-88
	MOVQ lhs_base+0(FP), R8              // R8 = lhsBasePtr
	MOVQ panel_base+24(FP), R9           // R9 = panelPtr
	MOVQ lhsRowStart+48(FP), R10         // R10 = lhsRowStart
	MOVQ lhsColStart+56(FP), R11         // R11 = lhsColStart
	MOVQ lhsCols+64(FP), R12             // R12 = lhsCols
	MOVQ copyRows+72(FP), R13            // R13 = copyRows
	MOVQ contractingCols+80(FP), R14     // R14 = contractingCols

	CMPQ R13, $8
	JL done_lhs8_k2
	TESTQ R14, R14
	JLE done_lhs8_k2

	SHLQ $1, R12                         // R12 = lhsStrideBytes = lhsCols * 2
	SHLQ $1, R11                         // R11 = lhsColStartBytes = lhsColStart * 2

	// R14 = fullKPairs = contractingCols >> 1
	// R15 = fullKPairs16 = fullKPairs & ~15
	SHRQ $1, R14
	MOVQ R14, R15
	ANDQ $~15, R15

	XORQ AX, AX                          // AX = stripRowIdx = 0
	ANDQ $~7, R13                        // R13 = fullStripLimit = copyRows & ~7

loop_strip_lhs8_k2:
	CMPQ AX, R13
	JGE done_lhs8_k2

	// SI = lhsBasePtr + (lhsRowStart + stripRowIdx) * lhsStrideBytes + lhsColStartBytes
	MOVQ R10, SI
	ADDQ AX, SI
	IMULQ R12, SI
	ADDQ R8, SI
	ADDQ R11, SI

	LEAQ (SI)(R12*1), DX                 // row 1
	LEAQ (DX)(R12*1), DI                 // row 2
	LEAQ (DI)(R12*1), CX                 // row 3
	LEAQ (R12*4), BP                     // 4 * lhsStrideBytes (rows 4..7)

	XORQ BX, BX                          // BX = kPairIdx = 0

	TESTQ R15, R15
	JZ check_tail_lhs8_k2

	PCALIGN $32
loop_kpairs_lhs8_k2:
	CMPQ BX, R15
	JGE check_tail_lhs8_k2

	// Load 16 K-pairs (16 uint32s = 32 BF16s = 64 bytes) from rows 0..3
	VMOVDQU32 (SI), Z0
	VMOVDQU32 (DX), Z1
	VMOVDQU32 (DI), Z2
	VMOVDQU32 (CX), Z3

	// 8x16 uint32 transpose (all integer-domain instructions)
	VPUNPCKLDQ Z1, Z0, Z8
	VPUNPCKHDQ Z1, Z0, Z9
	VPUNPCKLDQ Z3, Z2, Z10
	VPUNPCKHDQ Z3, Z2, Z11

	VPUNPCKLQDQ Z10, Z8, Z12
	VPUNPCKHQDQ Z10, Z8, Z13
	VPUNPCKLQDQ Z11, Z9, Z14
	VPUNPCKHQDQ Z11, Z9, Z15

	VSHUFI32X4 $0x44, Z13, Z12, Z16
	VSHUFI32X4 $0x44, Z15, Z14, Z17
	VSHUFI32X4 $0xEE, Z13, Z12, Z18
	VSHUFI32X4 $0xEE, Z15, Z14, Z19

	VSHUFI32X4 $0x88, Z17, Z16, Z24      // pairs 0..3, rows 0..3
	VSHUFI32X4 $0xDD, Z17, Z16, Z25      // pairs 4..7, rows 0..3
	VSHUFI32X4 $0x88, Z19, Z18, Z26      // pairs 8..11, rows 0..3
	VSHUFI32X4 $0xDD, Z19, Z18, Z27      // pairs 12..15, rows 0..3

	// Load 16 K-pairs from rows 4..7
	VMOVDQU32 (0)(SI)(BP*1), Z4
	VMOVDQU32 (0)(DX)(BP*1), Z5
	VMOVDQU32 (0)(DI)(BP*1), Z6
	VMOVDQU32 (0)(CX)(BP*1), Z7

	VPUNPCKLDQ Z5, Z4, Z8
	VPUNPCKHDQ Z5, Z4, Z9
	VPUNPCKLDQ Z7, Z6, Z10
	VPUNPCKHDQ Z7, Z6, Z11

	VPUNPCKLQDQ Z10, Z8, Z12
	VPUNPCKHQDQ Z10, Z8, Z13
	VPUNPCKLQDQ Z11, Z9, Z14
	VPUNPCKHQDQ Z11, Z9, Z15

	VSHUFI32X4 $0x44, Z13, Z12, Z16
	VSHUFI32X4 $0x44, Z15, Z14, Z17
	VSHUFI32X4 $0xEE, Z13, Z12, Z18
	VSHUFI32X4 $0xEE, Z15, Z14, Z19

	VSHUFI32X4 $0x88, Z17, Z16, Z28      // pairs 0..3, rows 4..7
	VSHUFI32X4 $0xDD, Z17, Z16, Z29      // pairs 4..7, rows 4..7
	VSHUFI32X4 $0x88, Z19, Z18, Z30      // pairs 8..11, rows 4..7
	VSHUFI32X4 $0xDD, Z19, Z18, Z31      // pairs 12..15, rows 4..7

	// Interleave rows 0..3 and rows 4..7 into pairs 0..15 (2 pairs per ZMM)
	VSHUFI32X4 $0x44, Z28, Z24, Z0
	VSHUFI32X4 $0xD8, Z0, Z0, Z0
	VSHUFI32X4 $0xEE, Z28, Z24, Z1
	VSHUFI32X4 $0xD8, Z1, Z1, Z1

	VSHUFI32X4 $0x44, Z29, Z25, Z2
	VSHUFI32X4 $0xD8, Z2, Z2, Z2
	VSHUFI32X4 $0xEE, Z29, Z25, Z3
	VSHUFI32X4 $0xD8, Z3, Z3, Z3

	VSHUFI32X4 $0x44, Z30, Z26, Z4
	VSHUFI32X4 $0xD8, Z4, Z4, Z4
	VSHUFI32X4 $0xEE, Z30, Z26, Z5
	VSHUFI32X4 $0xD8, Z5, Z5, Z5

	VSHUFI32X4 $0x44, Z31, Z27, Z6
	VSHUFI32X4 $0xD8, Z6, Z6, Z6
	VSHUFI32X4 $0xEE, Z31, Z27, Z7
	VSHUFI32X4 $0xD8, Z7, Z7, Z7

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
	ADDQ $512, R9                        // 16 K-pairs * 8 rows * 4 bytes
	ADDQ $16, BX
	JMP loop_kpairs_lhs8_k2

check_tail_lhs8_k2:
	CMPQ BX, R14
	JGE check_odd_k_lhs8_k2

loop_tail_lhs8_k2:
	MOVL (SI), R15
	MOVL R15, (R9)
	MOVL (DX), R15
	MOVL R15, 4(R9)
	MOVL (DI), R15
	MOVL R15, 8(R9)
	MOVL (CX), R15
	MOVL R15, 12(R9)
	MOVL (0)(SI)(BP*1), R15
	MOVL R15, 16(R9)
	MOVL (0)(DX)(BP*1), R15
	MOVL R15, 20(R9)
	MOVL (0)(DI)(BP*1), R15
	MOVL R15, 24(R9)
	MOVL (0)(CX)(BP*1), R15
	MOVL R15, 28(R9)

	ADDQ $4, SI
	ADDQ $4, DX
	ADDQ $4, DI
	ADDQ $4, CX
	ADDQ $32, R9
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_lhs8_k2

check_odd_k_lhs8_k2:
	TESTQ $1, contractingCols+80(FP)
	JZ restore_r15_lhs8_k2

	// Odd trailing K: zero-extend 16-bit BF16 to 32-bit (bf16, 0x0000)
	MOVWLZX (SI), R15
	MOVL R15, (R9)
	MOVWLZX (DX), R15
	MOVL R15, 4(R9)
	MOVWLZX (DI), R15
	MOVL R15, 8(R9)
	MOVWLZX (CX), R15
	MOVL R15, 12(R9)
	MOVWLZX (0)(SI)(BP*1), R15
	MOVL R15, 16(R9)
	MOVWLZX (0)(DX)(BP*1), R15
	MOVL R15, 20(R9)
	MOVWLZX (0)(DI)(BP*1), R15
	MOVL R15, 24(R9)
	MOVWLZX (0)(CX)(BP*1), R15
	MOVL R15, 28(R9)
	ADDQ $32, R9

restore_r15_lhs8_k2:
	MOVQ R14, R15
	ANDQ $~15, R15

	ADDQ $8, AX
	JMP loop_strip_lhs8_k2

done_lhs8_k2:
	VZEROUPPER
	RET

// ============================================================================
// func avx512PackRHSTransposed32BFloat16K2Asm(
//     rhs, panel []bfloat16.BFloat16,
//     rhsRowStart, rhsColStart, rhsCols,
//     copyRows, contractingCols int)
//
// Packs full 32-row strips (copyRows & ^31) of transposed bfloat16 RHS [N, K]
// into [numStrips, kPairs, 32, 2] where kPairs = (contractingCols + 1) / 2.
// Iterates in 8-row sub-strips using the same 8x16 uint32 transpose.
// ============================================================================
TEXT ·avx512PackRHSTransposed32BFloat16K2Asm(SB), NOSPLIT, $0-88
	MOVQ rhs_base+0(FP), R8              // R8 = rhsBasePtr
	MOVQ rhsRowStart+48(FP), R10         // R10 = rhsRowStart
	MOVQ rhsColStart+56(FP), R11         // R11 = rhsColStart
	MOVQ rhsCols+64(FP), R12             // R12 = rhsCols
	MOVQ copyRows+72(FP), R13            // R13 = copyRows
	MOVQ contractingCols+80(FP), R14     // R14 = contractingCols

	CMPQ R13, $32
	JL done_rhst32_k2
	TESTQ R14, R14
	JLE done_rhst32_k2

	SHLQ $1, R12                         // R12 = rhsStrideBytes = rhsCols * 2
	SHLQ $1, R11                         // R11 = rhsColStartBytes = rhsColStart * 2

	SHRQ $1, R14                         // R14 = fullKPairs = contractingCols >> 1
	ANDQ $~31, R13                       // R13 = fullRows = copyRows & ~31

	XORQ AX, AX                          // AX = curRow = 0, 8, 16, ..., fullRows-8

loop_substrip_rhst32_k2:
	CMPQ AX, R13
	JGE done_rhst32_k2

	// Compute destination pointer for this 8-row sub-strip:
	// stripIdx = AX >> 5, subRow = AX & 31
	// paddedKPairs = (contractingCols + 1) >> 1
	// R9 = panelBase + stripIdx * paddedKPairs * 128 + subRow * 4
	MOVQ contractingCols+80(FP), R9
	INCQ R9
	SHRQ $1, R9                          // R9 = paddedKPairs
	MOVQ AX, R15
	SHRQ $5, R15                         // R15 = stripIdx
	IMULQ R15, R9
	SHLQ $7, R9                          // R9 = stripIdx * paddedKPairs * 128
	MOVQ AX, R15
	ANDQ $31, R15
	SHLQ $2, R15                         // R15 = subRow * 4
	ADDQ R15, R9
	ADDQ panel_base+24(FP), R9           // R9 = pDst for K-pair 0 of this 8-row sub-strip

	// Source pointers for rows curRow..curRow+7:
	MOVQ R10, SI
	ADDQ AX, SI
	IMULQ R12, SI
	ADDQ R8, SI
	ADDQ R11, SI

	LEAQ (SI)(R12*1), DX                 // row 1
	LEAQ (DX)(R12*1), DI                 // row 2
	LEAQ (DI)(R12*1), CX                 // row 3
	LEAQ (R12*4), BP                     // 4 * rhsStrideBytes (rows 4..7)

	MOVQ R14, R15
	ANDQ $~15, R15                       // R15 = fullKPairs16
	XORQ BX, BX                          // BX = kPairIdx = 0

	TESTQ R15, R15
	JZ check_tail_rhst32_k2

	PCALIGN $32
loop_kpairs_rhst32_k2:
	CMPQ BX, R15
	JGE check_tail_rhst32_k2

	VMOVDQU32 (SI), Z0
	VMOVDQU32 (DX), Z1
	VMOVDQU32 (DI), Z2
	VMOVDQU32 (CX), Z3

	VPUNPCKLDQ Z1, Z0, Z8
	VPUNPCKHDQ Z1, Z0, Z9
	VPUNPCKLDQ Z3, Z2, Z10
	VPUNPCKHDQ Z3, Z2, Z11

	VPUNPCKLQDQ Z10, Z8, Z12
	VPUNPCKHQDQ Z10, Z8, Z13
	VPUNPCKLQDQ Z11, Z9, Z14
	VPUNPCKHQDQ Z11, Z9, Z15

	VSHUFI32X4 $0x44, Z13, Z12, Z16
	VSHUFI32X4 $0x44, Z15, Z14, Z17
	VSHUFI32X4 $0xEE, Z13, Z12, Z18
	VSHUFI32X4 $0xEE, Z15, Z14, Z19

	VSHUFI32X4 $0x88, Z17, Z16, Z24
	VSHUFI32X4 $0xDD, Z17, Z16, Z25
	VSHUFI32X4 $0x88, Z19, Z18, Z26
	VSHUFI32X4 $0xDD, Z19, Z18, Z27

	VMOVDQU32 (0)(SI)(BP*1), Z4
	VMOVDQU32 (0)(DX)(BP*1), Z5
	VMOVDQU32 (0)(DI)(BP*1), Z6
	VMOVDQU32 (0)(CX)(BP*1), Z7

	VPUNPCKLDQ Z5, Z4, Z8
	VPUNPCKHDQ Z5, Z4, Z9
	VPUNPCKLDQ Z7, Z6, Z10
	VPUNPCKHDQ Z7, Z6, Z11

	VPUNPCKLQDQ Z10, Z8, Z12
	VPUNPCKHQDQ Z10, Z8, Z13
	VPUNPCKLQDQ Z11, Z9, Z14
	VPUNPCKHQDQ Z11, Z9, Z15

	VSHUFI32X4 $0x44, Z13, Z12, Z16
	VSHUFI32X4 $0x44, Z15, Z14, Z17
	VSHUFI32X4 $0xEE, Z13, Z12, Z18
	VSHUFI32X4 $0xEE, Z15, Z14, Z19

	VSHUFI32X4 $0x88, Z17, Z16, Z28
	VSHUFI32X4 $0xDD, Z17, Z16, Z29
	VSHUFI32X4 $0x88, Z19, Z18, Z30
	VSHUFI32X4 $0xDD, Z19, Z18, Z31

	// Assemble 16 K-pairs (2 K-pairs per ZMM: low YMM = even pair, high YMM = odd pair)
	// and store strided by 128 bytes per K-pair.
	VSHUFI32X4 $0x44, Z28, Z24, Z0
	VSHUFI32X4 $0xD8, Z0, Z0, Z0
	VMOVDQU32 Y0, 0(R9)
	VEXTRACTI64X4 $1, Z0, Y8
	VMOVDQU32 Y8, 128(R9)

	VSHUFI32X4 $0xEE, Z28, Z24, Z1
	VSHUFI32X4 $0xD8, Z1, Z1, Z1
	VMOVDQU32 Y1, 256(R9)
	VEXTRACTI64X4 $1, Z1, Y8
	VMOVDQU32 Y8, 384(R9)

	VSHUFI32X4 $0x44, Z29, Z25, Z2
	VSHUFI32X4 $0xD8, Z2, Z2, Z2
	VMOVDQU32 Y2, 512(R9)
	VEXTRACTI64X4 $1, Z2, Y8
	VMOVDQU32 Y8, 640(R9)

	VSHUFI32X4 $0xEE, Z29, Z25, Z3
	VSHUFI32X4 $0xD8, Z3, Z3, Z3
	VMOVDQU32 Y3, 768(R9)
	VEXTRACTI64X4 $1, Z3, Y8
	VMOVDQU32 Y8, 896(R9)

	VSHUFI32X4 $0x44, Z30, Z26, Z4
	VSHUFI32X4 $0xD8, Z4, Z4, Z4
	VMOVDQU32 Y4, 1024(R9)
	VEXTRACTI64X4 $1, Z4, Y8
	VMOVDQU32 Y8, 1152(R9)

	VSHUFI32X4 $0xEE, Z30, Z26, Z5
	VSHUFI32X4 $0xD8, Z5, Z5, Z5
	VMOVDQU32 Y5, 1280(R9)
	VEXTRACTI64X4 $1, Z5, Y8
	VMOVDQU32 Y8, 1408(R9)

	VSHUFI32X4 $0x44, Z31, Z27, Z6
	VSHUFI32X4 $0xD8, Z6, Z6, Z6
	VMOVDQU32 Y6, 1536(R9)
	VEXTRACTI64X4 $1, Z6, Y8
	VMOVDQU32 Y8, 1664(R9)

	VSHUFI32X4 $0xEE, Z31, Z27, Z7
	VSHUFI32X4 $0xD8, Z7, Z7, Z7
	VMOVDQU32 Y7, 1792(R9)
	VEXTRACTI64X4 $1, Z7, Y8
	VMOVDQU32 Y8, 1920(R9)

	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, DI
	ADDQ $64, CX
	ADDQ $2048, R9                       // 16 K-pairs * 128 bytes
	ADDQ $16, BX
	JMP loop_kpairs_rhst32_k2

check_tail_rhst32_k2:
	CMPQ BX, R14
	JGE check_odd_k_rhst32_k2

loop_tail_rhst32_k2:
	MOVL (SI), R15
	MOVL R15, 0(R9)
	MOVL (DX), R15
	MOVL R15, 4(R9)
	MOVL (DI), R15
	MOVL R15, 8(R9)
	MOVL (CX), R15
	MOVL R15, 12(R9)
	MOVL (0)(SI)(BP*1), R15
	MOVL R15, 16(R9)
	MOVL (0)(DX)(BP*1), R15
	MOVL R15, 20(R9)
	MOVL (0)(DI)(BP*1), R15
	MOVL R15, 24(R9)
	MOVL (0)(CX)(BP*1), R15
	MOVL R15, 28(R9)

	ADDQ $4, SI
	ADDQ $4, DX
	ADDQ $4, DI
	ADDQ $4, CX
	ADDQ $128, R9
	INCQ BX
	CMPQ BX, R14
	JL loop_tail_rhst32_k2

check_odd_k_rhst32_k2:
	TESTQ $1, contractingCols+80(FP)
	JZ next_substrip_rhst32_k2

	MOVWLZX (SI), R15
	MOVL R15, 0(R9)
	MOVWLZX (DX), R15
	MOVL R15, 4(R9)
	MOVWLZX (DI), R15
	MOVL R15, 8(R9)
	MOVWLZX (CX), R15
	MOVL R15, 12(R9)
	MOVWLZX (0)(SI)(BP*1), R15
	MOVL R15, 16(R9)
	MOVWLZX (0)(DX)(BP*1), R15
	MOVL R15, 20(R9)
	MOVWLZX (0)(DI)(BP*1), R15
	MOVL R15, 24(R9)
	MOVWLZX (0)(CX)(BP*1), R15
	MOVL R15, 28(R9)

next_substrip_rhst32_k2:
	ADDQ $8, AX
	JMP loop_substrip_rhst32_k2

done_rhst32_k2:
	VZEROUPPER
	RET

// ============================================================================
// func avx512PackRHSNonTransposed32BFloat16K2Asm(
//     rhsPtr, panelPtr unsafe.Pointer,
//     rhsStrideBytes uintptr,
//     contractingRows, numFullStrips int)
//
// Packs numFullStrips full 32-column strips of non-transposed bfloat16 RHS [K, N]
// into [numFullStrips, kPairs, 32, 2] where kPairs = (contractingRows + 1) / 2.
// Columns within each K-pair are stored in natural order:
//   bytes  0..63: cols  0..15 interleaved as (B[2p, c], B[2p+1, c])
//   bytes 64..127: cols 16..31 interleaved as (B[2p, c], B[2p+1, c])
// ============================================================================
TEXT ·avx512PackRHSNonTransposed32BFloat16K2Asm(SB), NOSPLIT, $0-40
	MOVQ rhsPtr+0(FP), R8                // R8 = rhsBasePtr
	MOVQ panelPtr+8(FP), R9              // R9 = panelPtr
	MOVQ rhsStrideBytes+16(FP), R10      // R10 = rhsStrideBytes
	MOVQ contractingRows+24(FP), R11     // R11 = contractingRows
	MOVQ numFullStrips+32(FP), R12       // R12 = numFullStrips

	TESTQ R12, R12
	JLE done_rhsnt32_k2
	TESTQ R11, R11
	JLE done_rhsnt32_k2

	XORQ AX, AX                          // AX = stripIdx = 0

loop_strip_rhsnt32_k2:
	CMPQ AX, R12
	JGE done_rhsnt32_k2

	// SI = R8 + stripIdx * 64 (32 bfloat16s = 64 bytes per strip)
	MOVQ AX, SI
	SHLQ $6, SI
	ADDQ R8, SI

	MOVQ R11, CX                         // CX = rowsLeft

	PCALIGN $32
loop_4rows_rhsnt32_k2:
	CMPQ CX, $4
	JL check_2rows_rhsnt32_k2

	LEAQ (SI)(R10*1), DX                 // row k+1
	LEAQ (DX)(R10*1), BX                 // row k+2
	LEAQ (BX)(R10*1), BP                 // row k+3

	VMOVDQU16 (SI), Z0
	VMOVDQU16 (DX), Z1
	VMOVDQU16 (BX), Z6
	VMOVDQU16 (BP), Z7

	// Interleave pair 0 (rows k, k+1) into natural column order [0..15], [16..31]
	VPUNPCKLWD Z1, Z0, Z2
	VPUNPCKHWD Z1, Z0, Z3
	VSHUFI32X4 $0x44, Z3, Z2, Z4
	VSHUFI32X4 $0xD8, Z4, Z4, Z4
	VSHUFI32X4 $0xEE, Z3, Z2, Z5
	VSHUFI32X4 $0xD8, Z5, Z5, Z5

	// Interleave pair 1 (rows k+2, k+3)
	VPUNPCKLWD Z7, Z6, Z8
	VPUNPCKHWD Z7, Z6, Z9
	VSHUFI32X4 $0x44, Z9, Z8, Z10
	VSHUFI32X4 $0xD8, Z10, Z10, Z10
	VSHUFI32X4 $0xEE, Z9, Z8, Z11
	VSHUFI32X4 $0xD8, Z11, Z11, Z11

	VMOVDQU32 Z4, 0(R9)
	VMOVDQU32 Z5, 64(R9)
	VMOVDQU32 Z10, 128(R9)
	VMOVDQU32 Z11, 192(R9)

	LEAQ (BP)(R10*1), SI
	ADDQ $256, R9
	SUBQ $4, CX
	JMP loop_4rows_rhsnt32_k2

check_2rows_rhsnt32_k2:
	CMPQ CX, $2
	JL check_1row_rhsnt32_k2

	LEAQ (SI)(R10*1), DX
	VMOVDQU16 (SI), Z0
	VMOVDQU16 (DX), Z1

	VPUNPCKLWD Z1, Z0, Z2
	VPUNPCKHWD Z1, Z0, Z3
	VSHUFI32X4 $0x44, Z3, Z2, Z4
	VSHUFI32X4 $0xD8, Z4, Z4, Z4
	VSHUFI32X4 $0xEE, Z3, Z2, Z5
	VSHUFI32X4 $0xD8, Z5, Z5, Z5

	VMOVDQU32 Z4, 0(R9)
	VMOVDQU32 Z5, 64(R9)

	LEAQ (DX)(R10*1), SI
	ADDQ $128, R9
	SUBQ $2, CX

check_1row_rhsnt32_k2:
	TESTQ CX, CX
	JZ next_strip_rhsnt32_k2

	// Odd trailing row: interleave with zero vector
	VMOVDQU16 (SI), Z0
	VPXORD Z1, Z1, Z1

	VPUNPCKLWD Z1, Z0, Z2
	VPUNPCKHWD Z1, Z0, Z3
	VSHUFI32X4 $0x44, Z3, Z2, Z4
	VSHUFI32X4 $0xD8, Z4, Z4, Z4
	VSHUFI32X4 $0xEE, Z3, Z2, Z5
	VSHUFI32X4 $0xD8, Z5, Z5, Z5

	VMOVDQU32 Z4, 0(R9)
	VMOVDQU32 Z5, 64(R9)
	ADDQ $128, R9

next_strip_rhsnt32_k2:
	INCQ AX
	JMP loop_strip_rhsnt32_k2

done_rhsnt32_k2:
	VZEROUPPER
	RET

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512PackRHSFullStripsAsm(
//     rhsPtr, panelPtr unsafe.Pointer,
//     rhsStrideBytes uintptr,
//     contractingRows, numStrips, kernelColsBytes int)
TEXT ·avx512PackRHSFullStripsAsm(SB), NOSPLIT, $0-48
	MOVQ rhsPtr+0(FP), R8             // R8 = rhsBasePtr
	MOVQ panelPtr+8(FP), R9           // R9 = panelPtr
	MOVQ rhsStrideBytes+16(FP), R10   // R10 = rhsStrideBytes
	MOVQ contractingRows+24(FP), R11  // R11 = contractingRows
	MOVQ numStrips+32(FP), R12        // R12 = numStrips
	MOVQ kernelColsBytes+40(FP), R13  // R13 = kernelColsBytes

	TESTQ R12, R12
	JLE done
	TESTQ R11, R11
	JLE done

	CMPQ R13, $256
	JE strip256
	CMPQ R13, $128
	JE strip128
	CMPQ R13, $64
	JE strip64
	JMP done

// -------------------------------------------------------------
// 256 bytes per row (e.g. 64 float32s)
// -------------------------------------------------------------
strip256:
	XORQ AX, AX                       // AX = stripIdx = 0

strip256_loop_strip:
	CMPQ AX, R12
	JGE done

	MOVQ AX, SI
	SHLQ $8, SI                       // SI = AX * 256
	ADDQ R8, SI                       // SI = rhsStripBase

	MOVQ R11, CX                      // CX = rowsLeft = contractingRows

	PCALIGN $32
strip256_loop_4rows:
	CMPQ CX, $4
	JL strip256_loop_1row

	LEAQ (SI)(R10*1), DX
	LEAQ (DX)(R10*1), BX
	LEAQ (BX)(R10*1), BP

	VMOVDQU64 0(SI), Z0
	VMOVDQU64 64(SI), Z1
	VMOVDQU64 128(SI), Z2
	VMOVDQU64 192(SI), Z3

	VMOVDQU64 0(DX), Z4
	VMOVDQU64 64(DX), Z5
	VMOVDQU64 128(DX), Z6
	VMOVDQU64 192(DX), Z7

	VMOVDQU64 0(BX), Z8
	VMOVDQU64 64(BX), Z9
	VMOVDQU64 128(BX), Z10
	VMOVDQU64 192(BX), Z11

	VMOVDQU64 0(BP), Z12
	VMOVDQU64 64(BP), Z13
	VMOVDQU64 128(BP), Z14
	VMOVDQU64 192(BP), Z15

	VMOVDQU64 Z0, 0(R9)
	VMOVDQU64 Z1, 64(R9)
	VMOVDQU64 Z2, 128(R9)
	VMOVDQU64 Z3, 192(R9)
	VMOVDQU64 Z4, 256(R9)
	VMOVDQU64 Z5, 320(R9)
	VMOVDQU64 Z6, 384(R9)
	VMOVDQU64 Z7, 448(R9)
	VMOVDQU64 Z8, 512(R9)
	VMOVDQU64 Z9, 576(R9)
	VMOVDQU64 Z10, 640(R9)
	VMOVDQU64 Z11, 704(R9)
	VMOVDQU64 Z12, 768(R9)
	VMOVDQU64 Z13, 832(R9)
	VMOVDQU64 Z14, 896(R9)
	VMOVDQU64 Z15, 960(R9)

	LEAQ (BP)(R10*1), SI
	ADDQ $1024, R9
	SUBQ $4, CX
	JMP strip256_loop_4rows

strip256_loop_1row:
	TESTQ CX, CX
	JZ strip256_next_strip

	VMOVDQU64 0(SI), Z0
	VMOVDQU64 64(SI), Z1
	VMOVDQU64 128(SI), Z2
	VMOVDQU64 192(SI), Z3

	VMOVDQU64 Z0, 0(R9)
	VMOVDQU64 Z1, 64(R9)
	VMOVDQU64 Z2, 128(R9)
	VMOVDQU64 Z3, 192(R9)

	ADDQ R10, SI
	ADDQ $256, R9
	DECQ CX
	JMP strip256_loop_1row

strip256_next_strip:
	INCQ AX
	JMP strip256_loop_strip

// -------------------------------------------------------------
// 128 bytes per row (e.g. 64 float16s/bfloat16s)
// -------------------------------------------------------------
strip128:
	XORQ AX, AX

strip128_loop_strip:
	CMPQ AX, R12
	JGE done

	MOVQ AX, SI
	SHLQ $7, SI                       // SI = AX * 128
	ADDQ R8, SI

	MOVQ R11, CX

	PCALIGN $32
strip128_loop_4rows:
	CMPQ CX, $4
	JL strip128_loop_1row

	LEAQ (SI)(R10*1), DX
	LEAQ (DX)(R10*1), BX
	LEAQ (BX)(R10*1), BP

	VMOVDQU64 0(SI), Z0
	VMOVDQU64 64(SI), Z1
	VMOVDQU64 0(DX), Z2
	VMOVDQU64 64(DX), Z3
	VMOVDQU64 0(BX), Z4
	VMOVDQU64 64(BX), Z5
	VMOVDQU64 0(BP), Z6
	VMOVDQU64 64(BP), Z7

	VMOVDQU64 Z0, 0(R9)
	VMOVDQU64 Z1, 64(R9)
	VMOVDQU64 Z2, 128(R9)
	VMOVDQU64 Z3, 192(R9)
	VMOVDQU64 Z4, 256(R9)
	VMOVDQU64 Z5, 320(R9)
	VMOVDQU64 Z6, 384(R9)
	VMOVDQU64 Z7, 448(R9)

	LEAQ (BP)(R10*1), SI
	ADDQ $512, R9
	SUBQ $4, CX
	JMP strip128_loop_4rows

strip128_loop_1row:
	TESTQ CX, CX
	JZ strip128_next_strip

	VMOVDQU64 0(SI), Z0
	VMOVDQU64 64(SI), Z1

	VMOVDQU64 Z0, 0(R9)
	VMOVDQU64 Z1, 64(R9)

	ADDQ R10, SI
	ADDQ $128, R9
	DECQ CX
	JMP strip128_loop_1row

strip128_next_strip:
	INCQ AX
	JMP strip128_loop_strip

// -------------------------------------------------------------
// 64 bytes per row (e.g. 8 float64s, or 16 float32s)
// -------------------------------------------------------------
strip64:
	XORQ AX, AX

strip64_loop_strip:
	CMPQ AX, R12
	JGE done

	MOVQ AX, SI
	SHLQ $6, SI                       // SI = AX * 64
	ADDQ R8, SI

	MOVQ R11, CX

	PCALIGN $32
strip64_loop_4rows:
	CMPQ CX, $4
	JL strip64_loop_1row

	LEAQ (SI)(R10*1), DX
	LEAQ (DX)(R10*1), BX
	LEAQ (BX)(R10*1), BP

	VMOVDQU64 0(SI), Z0
	VMOVDQU64 0(DX), Z1
	VMOVDQU64 0(BX), Z2
	VMOVDQU64 0(BP), Z3

	VMOVDQU64 Z0, 0(R9)
	VMOVDQU64 Z1, 64(R9)
	VMOVDQU64 Z2, 128(R9)
	VMOVDQU64 Z3, 192(R9)

	LEAQ (BP)(R10*1), SI
	ADDQ $256, R9
	SUBQ $4, CX
	JMP strip64_loop_4rows

strip64_loop_1row:
	TESTQ CX, CX
	JZ strip64_next_strip

	VMOVDQU64 0(SI), Z0
	VMOVDQU64 Z0, 0(R9)

	ADDQ R10, SI
	ADDQ $64, R9
	DECQ CX
	JMP strip64_loop_1row

strip64_next_strip:
	INCQ AX
	JMP strip64_loop_strip

done:
	VZEROUPPER
	RET

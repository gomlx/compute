// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2PackRHSFullStripsAsm(
//     rhsPtr, panelPtr unsafe.Pointer,
//     rhsStrideBytes uintptr,
//     contractingRows, numStrips, kernelColsBytes int)
TEXT ·avx2PackRHSFullStripsAsm(SB), NOSPLIT, $0-48
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

	CMPQ R13, $128
	JE strip128
	CMPQ R13, $64
	JE strip64
	CMPQ R13, $32
	JE strip32
	JMP done

// -------------------------------------------------------------
// 128 bytes per row
// -------------------------------------------------------------
strip128:
	XORQ AX, AX                       // AX = stripIdx = 0

strip128_loop_strip:
	CMPQ AX, R12
	JGE done

	MOVQ AX, SI
	SHLQ $7, SI                       // SI = AX * 128
	ADDQ R8, SI                       // SI = rhsStripBase

	MOVQ R11, CX                      // CX = rowsLeft = contractingRows

	PCALIGN $32
strip128_loop_4rows:
	CMPQ CX, $4
	JL strip128_loop_1row

	LEAQ (SI)(R10*1), DX
	LEAQ (DX)(R10*1), BX
	LEAQ (BX)(R10*1), BP

	VMOVDQU 0(SI), Y0
	VMOVDQU 32(SI), Y1
	VMOVDQU 64(SI), Y2
	VMOVDQU 96(SI), Y3

	VMOVDQU 0(DX), Y4
	VMOVDQU 32(DX), Y5
	VMOVDQU 64(DX), Y6
	VMOVDQU 96(DX), Y7

	VMOVDQU Y0, 0(R9)
	VMOVDQU Y1, 32(R9)
	VMOVDQU Y2, 64(R9)
	VMOVDQU Y3, 96(R9)
	VMOVDQU Y4, 128(R9)
	VMOVDQU Y5, 160(R9)
	VMOVDQU Y6, 192(R9)
	VMOVDQU Y7, 224(R9)

	VMOVDQU 0(BX), Y0
	VMOVDQU 32(BX), Y1
	VMOVDQU 64(BX), Y2
	VMOVDQU 96(BX), Y3

	VMOVDQU 0(BP), Y4
	VMOVDQU 32(BP), Y5
	VMOVDQU 64(BP), Y6
	VMOVDQU 96(BP), Y7

	VMOVDQU Y0, 256(R9)
	VMOVDQU Y1, 288(R9)
	VMOVDQU Y2, 320(R9)
	VMOVDQU Y3, 352(R9)
	VMOVDQU Y4, 384(R9)
	VMOVDQU Y5, 416(R9)
	VMOVDQU Y6, 448(R9)
	VMOVDQU Y7, 480(R9)

	ADDQ $512, R9
	LEAQ (BP)(R10*1), SI
	SUBQ $4, CX
	JMP strip128_loop_4rows

strip128_loop_1row:
	TESTQ CX, CX
	JZ strip128_next_strip

	VMOVDQU 0(SI), Y0
	VMOVDQU 32(SI), Y1
	VMOVDQU 64(SI), Y2
	VMOVDQU 96(SI), Y3

	VMOVDQU Y0, 0(R9)
	VMOVDQU Y1, 32(R9)
	VMOVDQU Y2, 64(R9)
	VMOVDQU Y3, 96(R9)

	ADDQ R10, SI
	ADDQ $128, R9
	DECQ CX
	JMP strip128_loop_1row

strip128_next_strip:
	INCQ AX
	JMP strip128_loop_strip

// -------------------------------------------------------------
// 64 bytes per row (e.g. 16 float32s or 8 float64s)
// -------------------------------------------------------------
strip64:
	XORQ AX, AX                       // AX = stripIdx = 0

strip64_loop_strip:
	CMPQ AX, R12
	JGE done

	MOVQ AX, SI
	SHLQ $6, SI                       // SI = AX * 64
	ADDQ R8, SI                       // SI = rhsStripBase

	MOVQ R11, CX                      // CX = rowsLeft = contractingRows

	PCALIGN $32
strip64_loop_4rows:
	CMPQ CX, $4
	JL strip64_loop_1row

	LEAQ (SI)(R10*1), DX
	LEAQ (DX)(R10*1), BX
	LEAQ (BX)(R10*1), BP

	VMOVDQU 0(SI), Y0
	VMOVDQU 32(SI), Y1
	VMOVDQU 0(DX), Y2
	VMOVDQU 32(DX), Y3
	VMOVDQU 0(BX), Y4
	VMOVDQU 32(BX), Y5
	VMOVDQU 0(BP), Y6
	VMOVDQU 32(BP), Y7

	VMOVDQU Y0, 0(R9)
	VMOVDQU Y1, 32(R9)
	VMOVDQU Y2, 64(R9)
	VMOVDQU Y3, 96(R9)
	VMOVDQU Y4, 128(R9)
	VMOVDQU Y5, 160(R9)
	VMOVDQU Y6, 192(R9)
	VMOVDQU Y7, 224(R9)

	ADDQ $256, R9
	LEAQ (BP)(R10*1), SI
	SUBQ $4, CX
	JMP strip64_loop_4rows

strip64_loop_1row:
	TESTQ CX, CX
	JZ strip64_next_strip

	VMOVDQU 0(SI), Y0
	VMOVDQU 32(SI), Y1

	VMOVDQU Y0, 0(R9)
	VMOVDQU Y1, 32(R9)

	ADDQ R10, SI
	ADDQ $64, R9
	DECQ CX
	JMP strip64_loop_1row

strip64_next_strip:
	INCQ AX
	JMP strip64_loop_strip

// -------------------------------------------------------------
// 32 bytes per row (e.g. 16 float16s / bfloat16s)
// -------------------------------------------------------------
strip32:
	XORQ AX, AX                       // AX = stripIdx = 0

strip32_loop_strip:
	CMPQ AX, R12
	JGE done

	MOVQ AX, SI
	SHLQ $5, SI                       // SI = AX * 32
	ADDQ R8, SI                       // SI = rhsStripBase

	MOVQ R11, CX                      // CX = rowsLeft = contractingRows

	PCALIGN $32
strip32_loop_4rows:
	CMPQ CX, $4
	JL strip32_loop_1row

	LEAQ (SI)(R10*1), DX
	LEAQ (DX)(R10*1), BX
	LEAQ (BX)(R10*1), BP

	VMOVDQU 0(SI), Y0
	VMOVDQU 0(DX), Y1
	VMOVDQU 0(BX), Y2
	VMOVDQU 0(BP), Y3

	VMOVDQU Y0, 0(R9)
	VMOVDQU Y1, 32(R9)
	VMOVDQU Y2, 64(R9)
	VMOVDQU Y3, 96(R9)

	ADDQ $128, R9
	LEAQ (BP)(R10*1), SI
	SUBQ $4, CX
	JMP strip32_loop_4rows

strip32_loop_1row:
	TESTQ CX, CX
	JZ strip32_next_strip

	VMOVDQU 0(SI), Y0
	VMOVDQU Y0, 0(R9)

	ADDQ R10, SI
	ADDQ $32, R9
	DECQ CX
	JMP strip32_loop_1row

strip32_next_strip:
	INCQ AX
	JMP strip32_loop_strip

done:
	VZEROUPPER
	RET

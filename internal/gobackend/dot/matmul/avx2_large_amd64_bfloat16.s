// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx2LargeKernelBFloat16Asm(
//     packedLHS, packedRHS []bfloat16.BFloat16,
//     packedOutput []float32,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int,
//     accumulate bool)
TEXT ·avx2LargeKernelBFloat16Asm(SB), NOSPLIT, $0-120
	MOVQ packedLHS_base+0(FP), R8        // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9       // R9 = rhsBasePtr
	MOVQ packedOutput_base+48(FP), R10   // R10 = outBasePtr
	MOVQ rhsPanelCols+80(FP), R11        // R11 = outputStride (cols)
	MOVQ contractingLen+88(FP), R12      // R12 = contractingLen (K)
	MOVQ lhsActiveRows+96(FP), R13       // R13 = lhsActiveRows (M)
	MOVQ rhsActiveCols+104(FP), R14      // R14 = rhsActiveCols (N)
	MOVB accumulate+112(FP), R15         // R15 = accumulate (0 = overwrite, 1 = add)

	SHLQ $2, R11                         // R11 = outputStride in bytes (float32 output)

	XORQ AX, AX                          // AX = lhsRowIdx = 0

loop_lhs:
	CMPQ AX, R13
	JGE done

	XORQ BX, BX                          // BX = rhsColIdx = 0

loop_rhs:
	CMPQ BX, R14
	JGE next_lhs

	// 1. Zero all 8 accumulators (Y0..Y7)
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	// 2. Compute lhsPtr and rhsPtr
	// For 4-row LHS strip: strip has 4 bfloat16s = 8 bytes per step.
	// SI = R8 + (lhsRowIdx * contractingLen * 2)
	MOVQ AX, SI
	IMULQ R12, SI
	SHLQ $1, SI
	ADDQ R8, SI

	// For 16-col RHS strip: strip has 16 bfloat16s = 32 bytes per step.
	// DI = R9 + (rhsColIdx * contractingLen * 2)
	MOVQ BX, DI
	IMULQ R12, DI
	SHLQ $1, DI
	ADDQ R9, DI

	// 3. K-loop with 2-stage ping-pong pipeline (unroll by 2)
	MOVQ R12, CX                         // CX = contractingLen
	TESTQ CX, CX
	JLE store_output

	SHRQ $1, CX                          // CX = pairs = contractingLen / 2
	JZ k_odd

	// Prime Buffer A for Step 0 outside the loop
	VPMOVZXWD (DI), Y8                   // RHS col 0..7
	VPSLLD $16, Y8, Y8
	VPMOVZXWD 16(DI), Y9                 // RHS col 8..15
	VPSLLD $16, Y9, Y9
	VPMOVZXWD (SI), X10                  // 4 bfloat16s -> 4 float32s in low XMM
	VPSLLD $16, X10, X10
	VPERMILPS $0x55, X10, X11
	VPERMILPS $0xAA, X10, X12
	VPERMILPS $0xFF, X10, X13
	VBROADCASTSS X10, Y10                // LHS row 0
	VBROADCASTSS X11, Y11                // LHS row 1
	VBROADCASTSS X12, Y12                // LHS row 2
	VBROADCASTSS X13, Y13                // LHS row 3

	ADDQ $8, SI
	ADDQ $32, DI

	DECQ CX
	JZ k_last_pair

	PCALIGN $64
k_pair_loop:
	// Step A: FMAs using Buffer A (Y8, Y9) while loading Buffer B (Y14, Y15, Y10..Y13)
	VFMADD231PS Y8, Y10, Y0
	VPMOVZXWD (DI), Y14
	VFMADD231PS Y9, Y10, Y1
	VPSLLD $16, Y14, Y14
	VPMOVZXWD 16(DI), Y15
	VPSLLD $16, Y15, Y15

	VFMADD231PS Y8, Y11, Y2
	VFMADD231PS Y9, Y11, Y3
	VPMOVZXWD (SI), X10                  // Step B LHS
	VPSLLD $16, X10, X10

	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VPERMILPS $0x55, X10, X11
	VPERMILPS $0xAA, X10, X12
	VPERMILPS $0xFF, X10, X13

	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS X10, Y10
	VBROADCASTSS X11, Y11
	VBROADCASTSS X12, Y12
	VBROADCASTSS X13, Y13

	ADDQ $8, SI
	ADDQ $32, DI

	// Step B: FMAs using Buffer B (Y14, Y15) while loading next Buffer A (Y8, Y9, Y10..Y13)
	VFMADD231PS Y14, Y10, Y0
	VPMOVZXWD (DI), Y8
	VFMADD231PS Y15, Y10, Y1
	VPSLLD $16, Y8, Y8
	VPMOVZXWD 16(DI), Y9
	VPSLLD $16, Y9, Y9

	VFMADD231PS Y14, Y11, Y2
	VFMADD231PS Y15, Y11, Y3
	VPMOVZXWD (SI), X10                  // next Step A LHS
	VPSLLD $16, X10, X10

	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y15, Y12, Y5
	VPERMILPS $0x55, X10, X11
	VPERMILPS $0xAA, X10, X12
	VPERMILPS $0xFF, X10, X13

	VFMADD231PS Y14, Y13, Y6
	VFMADD231PS Y15, Y13, Y7
	VBROADCASTSS X10, Y10
	VBROADCASTSS X11, Y11
	VBROADCASTSS X12, Y12
	VBROADCASTSS X13, Y13

	ADDQ $8, SI
	ADDQ $32, DI

	DECQ CX
	JNZ k_pair_loop

k_last_pair:
	// Step A: FMAs using Buffer A (Y8, Y9) while loading Buffer B (Y14, Y15, Y10..Y13)
	VFMADD231PS Y8, Y10, Y0
	VPMOVZXWD (DI), Y14
	VFMADD231PS Y9, Y10, Y1
	VPSLLD $16, Y14, Y14
	VPMOVZXWD 16(DI), Y15
	VPSLLD $16, Y15, Y15

	VFMADD231PS Y8, Y11, Y2
	VFMADD231PS Y9, Y11, Y3
	VPMOVZXWD (SI), X10
	VPSLLD $16, X10, X10

	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VPERMILPS $0x55, X10, X11
	VPERMILPS $0xAA, X10, X12
	VPERMILPS $0xFF, X10, X13

	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7
	VBROADCASTSS X10, Y10
	VBROADCASTSS X11, Y11
	VBROADCASTSS X12, Y12
	VBROADCASTSS X13, Y13

	ADDQ $8, SI
	ADDQ $32, DI

	// Step B: Pure FMAs using Buffer B (Y14, Y15) - NO further memory loads!
	VFMADD231PS Y14, Y10, Y0
	VFMADD231PS Y15, Y10, Y1
	VFMADD231PS Y14, Y11, Y2
	VFMADD231PS Y15, Y11, Y3
	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y15, Y12, Y5
	VFMADD231PS Y14, Y13, Y6
	VFMADD231PS Y15, Y13, Y7

k_odd:
	TESTQ $1, R12
	JZ store_output

	VPMOVZXWD (DI), Y8
	VPSLLD $16, Y8, Y8
	VPMOVZXWD 16(DI), Y9
	VPSLLD $16, Y9, Y9
	VPMOVZXWD (SI), X10
	VPSLLD $16, X10, X10
	VPERMILPS $0x55, X10, X11
	VPERMILPS $0xAA, X10, X12
	VPERMILPS $0xFF, X10, X13
	VBROADCASTSS X10, Y10
	VBROADCASTSS X11, Y11
	VBROADCASTSS X12, Y12
	VBROADCASTSS X13, Y13

	VFMADD231PS Y8, Y10, Y0
	VFMADD231PS Y9, Y10, Y1
	VFMADD231PS Y8, Y11, Y2
	VFMADD231PS Y9, Y11, Y3
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y12, Y5
	VFMADD231PS Y8, Y13, Y6
	VFMADD231PS Y9, Y13, Y7

store_output:
	MOVQ AX, DX
	IMULQ R11, DX
	MOVQ BX, CX
	SHLQ $2, CX
	ADDQ CX, DX
	LEAQ (R10)(DX*1), DX

	TESTB R15, R15
	JNZ store_output_accum

	// Row 0
	VMOVDQU Y0, (DX)
	VMOVDQU Y1, 32(DX)
	// Row 1
	ADDQ R11, DX
	VMOVDQU Y2, (DX)
	VMOVDQU Y3, 32(DX)
	// Row 2
	ADDQ R11, DX
	VMOVDQU Y4, (DX)
	VMOVDQU Y5, 32(DX)
	// Row 3
	ADDQ R11, DX
	VMOVDQU Y6, (DX)
	VMOVDQU Y7, 32(DX)

	ADDQ $16, BX                         // rhsColIdx += 16
	JMP loop_rhs

store_output_accum:
	// Row 0
	VADDPS (DX), Y0, Y0
	VMOVDQU Y0, (DX)
	VADDPS 32(DX), Y1, Y1
	VMOVDQU Y1, 32(DX)
	// Row 1
	ADDQ R11, DX
	VADDPS (DX), Y2, Y2
	VMOVDQU Y2, (DX)
	VADDPS 32(DX), Y3, Y3
	VMOVDQU Y3, 32(DX)
	// Row 2
	ADDQ R11, DX
	VADDPS (DX), Y4, Y4
	VMOVDQU Y4, (DX)
	VADDPS 32(DX), Y5, Y5
	VMOVDQU Y5, 32(DX)
	// Row 3
	ADDQ R11, DX
	VADDPS (DX), Y6, Y6
	VMOVDQU Y6, (DX)
	VADDPS 32(DX), Y7, Y7
	VMOVDQU Y7, 32(DX)

	ADDQ $16, BX                         // rhsColIdx += 16
	JMP loop_rhs

next_lhs:
	ADDQ $4, AX                          // lhsRowIdx += 4
	JMP loop_lhs

done:
	VZEROUPPER
	RET

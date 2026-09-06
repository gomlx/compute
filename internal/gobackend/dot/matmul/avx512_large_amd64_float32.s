// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512LargeKernelFloat32Asm(
//     packedLHS, packedRHS, packedOutput []float32,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int)
TEXT ·avx512LargeKernelFloat32Asm(SB), NOSPLIT, $0-112
	MOVQ packedLHS_base+0(FP), R8        // R8 = lhsBasePtr
	MOVQ packedRHS_base+24(FP), R9       // R9 = rhsBasePtr
	MOVQ packedOutput_base+48(FP), R10   // R10 = outBasePtr
	MOVQ rhsPanelCols+80(FP), R11        // R11 = outputStride (cols)
	MOVQ contractingLen+88(FP), R12      // R12 = contractingLen (K)
	MOVQ lhsActiveRows+96(FP), R13       // R13 = lhsActiveRows (M)
	MOVQ rhsActiveCols+104(FP), R14      // R14 = rhsActiveCols (N)

	SHLQ $2, R11                         // R11 = outputStride in bytes

	XORQ AX, AX                          // AX = lhsRowIdx = 0

loop_lhs:
	CMPQ AX, R13
	JGE done

	XORQ BX, BX                          // BX = rhsColIdx = 0

loop_rhs:
	CMPQ BX, R14
	JGE next_lhs

	// 1. Zero all 16 accumulators (Z0..Z15)
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7
	VXORPS Z8, Z8, Z8
	VXORPS Z9, Z9, Z9
	VXORPS Z10, Z10, Z10
	VXORPS Z11, Z11, Z11
	VXORPS Z12, Z12, Z12
	VXORPS Z13, Z13, Z13
	VXORPS Z14, Z14, Z14
	VXORPS Z15, Z15, Z15

	// 2. Compute lhsPtr and rhsPtr
	// idxLHS = lhsRowIdx * contractingLen
	MOVQ AX, SI
	IMULQ R12, SI
	SHLQ $2, SI
	LEAQ (R8)(SI*1), SI                  // SI = lhsPtr

	// idxRHS = rhsColIdx * contractingLen
	MOVQ BX, DI
	IMULQ R12, DI
	SHLQ $2, DI
	LEAQ (R9)(DI*1), DI                  // DI = rhsPtr

	// 3. K-loop with 2-stage ping-pong pipeline (unroll by 2)
	MOVQ R12, CX                         // CX = contractingLen
	TESTQ CX, CX
	JLE store_output

	SHRQ $1, CX                          // CX = pairs = contractingLen / 2
	JZ k_odd                             // If 0 pairs (contractingLen == 1), do odd iteration

	// Prime Buffer A for Step 0 outside the loop
	VMOVDQU32 (DI), Z16
	VMOVDQU32 64(DI), Z17
	VMOVDQU32 128(DI), Z18
	VMOVDQU32 192(DI), Z19
	VBROADCASTSS (SI), Z20
	VBROADCASTSS 4(SI), Z21
	VBROADCASTSS 8(SI), Z22
	VBROADCASTSS 12(SI), Z23
	ADDQ $16, SI                         // Advance past Step 0 (points to Step 1 = Buffer B)
	ADDQ $256, DI

	DECQ CX                              // CX = pairs - 1
	JZ k_last_pair                       // If only 1 pair total, skip k_pair_loop directly to k_last_pair

	PCALIGN $64
k_pair_loop:
	// Step A: Row 0 FMAs interleaved with Buffer B RHS loads (Z24..Z27)
	VFMADD231PS Z16, Z20, Z0
	VMOVDQU32 (DI), Z24
	VFMADD231PS Z17, Z20, Z1
	VMOVDQU32 64(DI), Z25
	VFMADD231PS Z18, Z20, Z2
	VMOVDQU32 128(DI), Z26
	VFMADD231PS Z19, Z20, Z3
	VMOVDQU32 192(DI), Z27

	// Step A: Row 1 FMAs interleaved with Buffer B LHS loads (Z28..Z31)
	VFMADD231PS Z16, Z21, Z4
	VBROADCASTSS (SI), Z28
	VFMADD231PS Z17, Z21, Z5
	VBROADCASTSS 4(SI), Z29
	VFMADD231PS Z18, Z21, Z6
	VBROADCASTSS 8(SI), Z30
	VFMADD231PS Z19, Z21, Z7
	VBROADCASTSS 12(SI), Z31

	ADDQ $16, SI                         // Advance past Step B
	ADDQ $256, DI

	// Step A: Rows 2-3 pure FMAs (provides 4-cycle runway for Buffer B loads to settle)
	VFMADD231PS Z16, Z22, Z8
	VFMADD231PS Z17, Z22, Z9
	VFMADD231PS Z18, Z22, Z10
	VFMADD231PS Z19, Z22, Z11

	VFMADD231PS Z16, Z23, Z12
	VFMADD231PS Z17, Z23, Z13
	VFMADD231PS Z18, Z23, Z14
	VFMADD231PS Z19, Z23, Z15

	// Step B: Row 0 FMAs interleaved with next-iter Buffer A RHS loads (Z16..Z19)
	VFMADD231PS Z24, Z28, Z0
	VMOVDQU32 (DI), Z16
	VFMADD231PS Z25, Z28, Z1
	VMOVDQU32 64(DI), Z17
	VFMADD231PS Z26, Z28, Z2
	VMOVDQU32 128(DI), Z18
	VFMADD231PS Z27, Z28, Z3
	VMOVDQU32 192(DI), Z19

	// Step B: Row 1 FMAs interleaved with next-iter Buffer A LHS loads (Z20..Z23)
	VFMADD231PS Z24, Z29, Z4
	VBROADCASTSS (SI), Z20
	VFMADD231PS Z25, Z29, Z5
	VBROADCASTSS 4(SI), Z21
	VFMADD231PS Z26, Z29, Z6
	VBROADCASTSS 8(SI), Z22
	VFMADD231PS Z27, Z29, Z7
	VBROADCASTSS 12(SI), Z23

	ADDQ $16, SI                         // Advance past Step A of next iter
	ADDQ $256, DI

	// Step B: Rows 2-3 pure FMAs (provides 4-cycle runway for Buffer A loads to settle)
	VFMADD231PS Z24, Z30, Z8
	VFMADD231PS Z25, Z30, Z9
	VFMADD231PS Z26, Z30, Z10
	VFMADD231PS Z27, Z30, Z11

	VFMADD231PS Z24, Z31, Z12
	VFMADD231PS Z25, Z31, Z13
	VFMADD231PS Z26, Z31, Z14
	VFMADD231PS Z27, Z31, Z15

	DECQ CX
	JNZ k_pair_loop

k_last_pair:
	// Step A: Row 0 FMAs interleaved with Buffer B RHS loads (Z24..Z27)
	VFMADD231PS Z16, Z20, Z0
	VMOVDQU32 (DI), Z24
	VFMADD231PS Z17, Z20, Z1
	VMOVDQU32 64(DI), Z25
	VFMADD231PS Z18, Z20, Z2
	VMOVDQU32 128(DI), Z26
	VFMADD231PS Z19, Z20, Z3
	VMOVDQU32 192(DI), Z27

	// Step A: Row 1 FMAs interleaved with Buffer B LHS loads (Z28..Z31)
	VFMADD231PS Z16, Z21, Z4
	VBROADCASTSS (SI), Z28
	VFMADD231PS Z17, Z21, Z5
	VBROADCASTSS 4(SI), Z29
	VFMADD231PS Z18, Z21, Z6
	VBROADCASTSS 8(SI), Z30
	VFMADD231PS Z19, Z21, Z7
	VBROADCASTSS 12(SI), Z31

	ADDQ $16, SI
	ADDQ $256, DI

	// Step A: Rows 2-3 pure FMAs
	VFMADD231PS Z16, Z22, Z8
	VFMADD231PS Z17, Z22, Z9
	VFMADD231PS Z18, Z22, Z10
	VFMADD231PS Z19, Z22, Z11

	VFMADD231PS Z16, Z23, Z12
	VFMADD231PS Z17, Z23, Z13
	VFMADD231PS Z18, Z23, Z14
	VFMADD231PS Z19, Z23, Z15

	// Step B: 16 FMAs using Buffer B (NO loads for next iter to prevent OOB)
	VFMADD231PS Z24, Z28, Z0
	VFMADD231PS Z25, Z28, Z1
	VFMADD231PS Z26, Z28, Z2
	VFMADD231PS Z27, Z28, Z3

	VFMADD231PS Z24, Z29, Z4
	VFMADD231PS Z25, Z29, Z5
	VFMADD231PS Z26, Z29, Z6
	VFMADD231PS Z27, Z29, Z7

	VFMADD231PS Z24, Z30, Z8
	VFMADD231PS Z25, Z30, Z9
	VFMADD231PS Z26, Z30, Z10
	VFMADD231PS Z27, Z30, Z11

	VFMADD231PS Z24, Z31, Z12
	VFMADD231PS Z25, Z31, Z13
	VFMADD231PS Z26, Z31, Z14
	VFMADD231PS Z27, Z31, Z15

k_odd:
	// Handle remaining odd iteration if contractingLen % 2 != 0
	TESTQ $1, R12
	JZ store_output

	VMOVDQU32 (DI), Z16
	VMOVDQU32 64(DI), Z17
	VMOVDQU32 128(DI), Z18
	VMOVDQU32 192(DI), Z19

	VBROADCASTSS (SI), Z20
	VFMADD231PS Z16, Z20, Z0
	VFMADD231PS Z17, Z20, Z1
	VFMADD231PS Z18, Z20, Z2
	VFMADD231PS Z19, Z20, Z3

	VBROADCASTSS 4(SI), Z21
	VFMADD231PS Z16, Z21, Z4
	VFMADD231PS Z17, Z21, Z5
	VFMADD231PS Z18, Z21, Z6
	VFMADD231PS Z19, Z21, Z7

	VBROADCASTSS 8(SI), Z22
	VFMADD231PS Z16, Z22, Z8
	VFMADD231PS Z17, Z22, Z9
	VFMADD231PS Z18, Z22, Z10
	VFMADD231PS Z19, Z22, Z11

	VBROADCASTSS 12(SI), Z23
	VFMADD231PS Z16, Z23, Z12
	VFMADD231PS Z17, Z23, Z13
	VFMADD231PS Z18, Z23, Z14
	VFMADD231PS Z19, Z23, Z15

store_output:
	// 4. Write back 16 accumulators to output:
	// outputBase = outBasePtr + (lhsRowIdx * outputStrideBytes) + (rhsColIdx * 4)
	MOVQ AX, DX                          // DX = lhsRowIdx
	IMULQ R11, DX                        // DX = lhsRowIdx * outputStrideBytes
	MOVQ BX, CX
	SHLQ $2, CX                          // CX = rhsColIdx * 4
	ADDQ CX, DX
	LEAQ (R10)(DX*1), DX                 // DX = outRow0Ptr

	// Row 0
	VMOVDQU32 Z0, (DX)
	VMOVDQU32 Z1, 64(DX)
	VMOVDQU32 Z2, 128(DX)
	VMOVDQU32 Z3, 192(DX)

	// Row 1
	ADDQ R11, DX                         // DX = outRow1Ptr
	VMOVDQU32 Z4, (DX)
	VMOVDQU32 Z5, 64(DX)
	VMOVDQU32 Z6, 128(DX)
	VMOVDQU32 Z7, 192(DX)

	// Row 2
	ADDQ R11, DX                         // DX = outRow2Ptr
	VMOVDQU32 Z8, (DX)
	VMOVDQU32 Z9, 64(DX)
	VMOVDQU32 Z10, 128(DX)
	VMOVDQU32 Z11, 192(DX)

	// Row 3
	ADDQ R11, DX                         // DX = outRow3Ptr
	VMOVDQU32 Z12, (DX)
	VMOVDQU32 Z13, 64(DX)
	VMOVDQU32 Z14, 128(DX)
	VMOVDQU32 Z15, 192(DX)

	ADDQ $64, BX                         // rhsColIdx += 64
	JMP loop_rhs

next_lhs:
	ADDQ $4, AX                          // lhsRowIdx += 4
	JMP loop_lhs

done:
	RET


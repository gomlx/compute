// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512LargeKernelFloat64Asm(
//     packedLHS, packedRHS, packedOutput []float64,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int)
TEXT ·avx512LargeKernelFloat64Asm(SB), NOSPLIT, $0-112
	MOVQ packedLHS_base+0(FP), R8        // R8 = lhsBasePtr (float64 = 8 bytes)
	MOVQ packedRHS_base+24(FP), R9       // R9 = rhsBasePtr (float64 = 8 bytes)
	MOVQ packedOutput_base+48(FP), R10   // R10 = outBasePtr (float64 = 8 bytes)
	MOVQ rhsPanelCols+80(FP), R11        // R11 = outputStride (cols)
	MOVQ contractingLen+88(FP), R12      // R12 = contractingLen (K)
	MOVQ lhsActiveRows+96(FP), R13       // R13 = lhsActiveRows (M)
	MOVQ rhsActiveCols+104(FP), R14      // R14 = rhsActiveCols (N)

	SHLQ $3, R11                         // R11 = outputStride in bytes (float64 = 8 bytes)

	XORQ AX, AX                          // AX = lhsRowIdx = 0

loop_lhs_f64:
	CMPQ AX, R13
	JGE done_f64

	XORQ BX, BX                          // BX = rhsColIdx = 0

loop_rhs_f64:
	CMPQ BX, R14
	JGE next_lhs_f64

	// 1. Zero all 16 accumulators (Z0..Z15)
	VXORPD Z0, Z0, Z0
	VXORPD Z1, Z1, Z1
	VXORPD Z2, Z2, Z2
	VXORPD Z3, Z3, Z3
	VXORPD Z4, Z4, Z4
	VXORPD Z5, Z5, Z5
	VXORPD Z6, Z6, Z6
	VXORPD Z7, Z7, Z7
	VXORPD Z8, Z8, Z8
	VXORPD Z9, Z9, Z9
	VXORPD Z10, Z10, Z10
	VXORPD Z11, Z11, Z11
	VXORPD Z12, Z12, Z12
	VXORPD Z13, Z13, Z13
	VXORPD Z14, Z14, Z14
	VXORPD Z15, Z15, Z15

	// 2. Compute lhsPtr and rhsPtr (elements are float64 = 8 bytes)
	// idxLHS = lhsRowIdx * contractingLen
	MOVQ AX, SI
	IMULQ R12, SI
	SHLQ $3, SI                          // SI = byte offset (float64 = 8 bytes)
	LEAQ (R8)(SI*1), SI                  // SI = lhsPtr

	// idxRHS = rhsColIdx * contractingLen
	MOVQ BX, DI
	IMULQ R12, DI
	SHLQ $3, DI                          // DI = byte offset (float64 = 8 bytes)
	LEAQ (R9)(DI*1), DI                  // DI = rhsPtr

	// 3. K-loop with 2-stage ping-pong pipeline (unroll by 2)
	MOVQ R12, CX                         // CX = contractingLen
	TESTQ CX, CX
	JLE store_output_f64

	SHRQ $1, CX                          // CX = pairs = contractingLen / 2
	JZ k_odd_f64                         // If 0 pairs (contractingLen == 1), do odd iteration

	// Prime Buffer A for Step 0 outside the loop
	// 4 RHS vectors (32 float64s = 256 bytes):
	VMOVDQU64 (DI), Z16
	VMOVDQU64 64(DI), Z17
	VMOVDQU64 128(DI), Z18
	VMOVDQU64 192(DI), Z19

	// 4 LHS scalars (4 float64s = 32 bytes):
	VBROADCASTSD (SI), Z20
	VBROADCASTSD 8(SI), Z21
	VBROADCASTSD 16(SI), Z22
	VBROADCASTSD 24(SI), Z23

	ADDQ $32, SI                         // Advance past Step 0 (4 float64s = 32 bytes)
	ADDQ $256, DI                        // Advance past Step 0 (32 float64s = 256 bytes)

	DECQ CX                              // CX = pairs - 1
	JZ k_last_pair_f64                   // If only 1 pair total, skip loop directly to k_last_pair_f64

	PCALIGN $64
k_pair_loop_f64:
	// Step A: Row 0 FMAs interleaved with Buffer B RHS loads (Z24..Z27)
	VFMADD231PD Z16, Z20, Z0
	VMOVDQU64 (DI), Z24
	VFMADD231PD Z17, Z20, Z1
	VMOVDQU64 64(DI), Z25
	VFMADD231PD Z18, Z20, Z2
	VMOVDQU64 128(DI), Z26
	VFMADD231PD Z19, Z20, Z3
	VMOVDQU64 192(DI), Z27

	// Step A: Row 1 FMAs interleaved with Buffer B LHS loads (Z28..Z31)
	VFMADD231PD Z16, Z21, Z4
	VBROADCASTSD (SI), Z28
	VFMADD231PD Z17, Z21, Z5
	VBROADCASTSD 8(SI), Z29
	VFMADD231PD Z18, Z21, Z6
	VBROADCASTSD 16(SI), Z30
	VFMADD231PD Z19, Z21, Z7
	VBROADCASTSD 24(SI), Z31

	ADDQ $32, SI                         // Advance past Step B
	ADDQ $256, DI

	// Step A: Rows 2-3 pure FMAs
	VFMADD231PD Z16, Z22, Z8
	VFMADD231PD Z17, Z22, Z9
	VFMADD231PD Z18, Z22, Z10
	VFMADD231PD Z19, Z22, Z11

	VFMADD231PD Z16, Z23, Z12
	VFMADD231PD Z17, Z23, Z13
	VFMADD231PD Z18, Z23, Z14
	VFMADD231PD Z19, Z23, Z15

	// Step B: Row 0 FMAs interleaved with next-iter Buffer A RHS loads (Z16..Z19)
	VFMADD231PD Z24, Z28, Z0
	VMOVDQU64 (DI), Z16
	VFMADD231PD Z25, Z28, Z1
	VMOVDQU64 64(DI), Z17
	VFMADD231PD Z26, Z28, Z2
	VMOVDQU64 128(DI), Z18
	VFMADD231PD Z27, Z28, Z3
	VMOVDQU64 192(DI), Z19

	// Step B: Row 1 FMAs interleaved with next-iter Buffer A LHS loads (Z20..Z23)
	VFMADD231PD Z24, Z29, Z4
	VBROADCASTSD (SI), Z20
	VFMADD231PD Z25, Z29, Z5
	VBROADCASTSD 8(SI), Z21
	VFMADD231PD Z26, Z29, Z6
	VBROADCASTSD 16(SI), Z22
	VFMADD231PD Z27, Z29, Z7
	VBROADCASTSD 24(SI), Z23

	ADDQ $32, SI                         // Advance past Step A of next iter
	ADDQ $256, DI

	// Step B: Rows 2-3 pure FMAs
	VFMADD231PD Z24, Z30, Z8
	VFMADD231PD Z25, Z30, Z9
	VFMADD231PD Z26, Z30, Z10
	VFMADD231PD Z27, Z30, Z11

	VFMADD231PD Z24, Z31, Z12
	VFMADD231PD Z25, Z31, Z13
	VFMADD231PD Z26, Z31, Z14
	VFMADD231PD Z27, Z31, Z15

	DECQ CX
	JNZ k_pair_loop_f64

k_last_pair_f64:
	// Step A: Row 0 FMAs interleaved with Buffer B RHS loads
	VFMADD231PD Z16, Z20, Z0
	VMOVDQU64 (DI), Z24
	VFMADD231PD Z17, Z20, Z1
	VMOVDQU64 64(DI), Z25
	VFMADD231PD Z18, Z20, Z2
	VMOVDQU64 128(DI), Z26
	VFMADD231PD Z19, Z20, Z3
	VMOVDQU64 192(DI), Z27

	// Step A: Row 1 FMAs interleaved with Buffer B LHS loads
	VFMADD231PD Z16, Z21, Z4
	VBROADCASTSD (SI), Z28
	VFMADD231PD Z17, Z21, Z5
	VBROADCASTSD 8(SI), Z29
	VFMADD231PD Z18, Z21, Z6
	VBROADCASTSD 16(SI), Z30
	VFMADD231PD Z19, Z21, Z7
	VBROADCASTSD 24(SI), Z31

	ADDQ $32, SI
	ADDQ $256, DI

	// Step A: Rows 2-3 pure FMAs
	VFMADD231PD Z16, Z22, Z8
	VFMADD231PD Z17, Z22, Z9
	VFMADD231PD Z18, Z22, Z10
	VFMADD231PD Z19, Z22, Z11

	VFMADD231PD Z16, Z23, Z12
	VFMADD231PD Z17, Z23, Z13
	VFMADD231PD Z18, Z23, Z14
	VFMADD231PD Z19, Z23, Z15

	// Step B: 16 FMAs using Buffer B (NO loads)
	VFMADD231PD Z24, Z28, Z0
	VFMADD231PD Z25, Z28, Z1
	VFMADD231PD Z26, Z28, Z2
	VFMADD231PD Z27, Z28, Z3

	VFMADD231PD Z24, Z29, Z4
	VFMADD231PD Z25, Z29, Z5
	VFMADD231PD Z26, Z29, Z6
	VFMADD231PD Z27, Z29, Z7

	VFMADD231PD Z24, Z30, Z8
	VFMADD231PD Z25, Z30, Z9
	VFMADD231PD Z26, Z30, Z10
	VFMADD231PD Z27, Z30, Z11

	VFMADD231PD Z24, Z31, Z12
	VFMADD231PD Z25, Z31, Z13
	VFMADD231PD Z26, Z31, Z14
	VFMADD231PD Z27, Z31, Z15

k_odd_f64:
	TESTQ $1, R12
	JZ store_output_f64

	VMOVDQU64 (DI), Z16
	VMOVDQU64 64(DI), Z17
	VMOVDQU64 128(DI), Z18
	VMOVDQU64 192(DI), Z19

	VBROADCASTSD (SI), Z20
	VBROADCASTSD 8(SI), Z21
	VBROADCASTSD 16(SI), Z22
	VBROADCASTSD 24(SI), Z23

	// Row 0
	VFMADD231PD Z16, Z20, Z0
	VFMADD231PD Z17, Z20, Z1
	VFMADD231PD Z18, Z20, Z2
	VFMADD231PD Z19, Z20, Z3

	// Row 1
	VFMADD231PD Z16, Z21, Z4
	VFMADD231PD Z17, Z21, Z5
	VFMADD231PD Z18, Z21, Z6
	VFMADD231PD Z19, Z21, Z7

	// Row 2
	VFMADD231PD Z16, Z22, Z8
	VFMADD231PD Z17, Z22, Z9
	VFMADD231PD Z18, Z22, Z10
	VFMADD231PD Z19, Z22, Z11

	// Row 3
	VFMADD231PD Z16, Z23, Z12
	VFMADD231PD Z17, Z23, Z13
	VFMADD231PD Z18, Z23, Z14
	VFMADD231PD Z19, Z23, Z15

store_output_f64:
	// Output is []float64: write back
	MOVQ AX, DX
	IMULQ R11, DX
	MOVQ BX, CX
	SHLQ $3, CX                          // float64 = 8 bytes
	ADDQ CX, DX
	LEAQ (R10)(DX*1), DX

	// Row 0
	VMOVDQU64 Z0, (DX)
	VMOVDQU64 Z1, 64(DX)
	VMOVDQU64 Z2, 128(DX)
	VMOVDQU64 Z3, 192(DX)

	// Row 1
	ADDQ R11, DX
	VMOVDQU64 Z4, (DX)
	VMOVDQU64 Z5, 64(DX)
	VMOVDQU64 Z6, 128(DX)
	VMOVDQU64 Z7, 192(DX)

	// Row 2
	ADDQ R11, DX
	VMOVDQU64 Z8, (DX)
	VMOVDQU64 Z9, 64(DX)
	VMOVDQU64 Z10, 128(DX)
	VMOVDQU64 Z11, 192(DX)

	// Row 3
	ADDQ R11, DX
	VMOVDQU64 Z12, (DX)
	VMOVDQU64 Z13, 64(DX)
	VMOVDQU64 Z14, 128(DX)
	VMOVDQU64 Z15, 192(DX)

	ADDQ $32, BX                         // rhsColIdx += 32
	JMP loop_rhs_f64

next_lhs_f64:
	ADDQ $4, AX                          // lhsRowIdx += 4
	JMP loop_lhs_f64

done_f64:
	RET

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func avx512LargeKernelBFloat16Asm(
//     packedLHS, packedRHS []bfloat16.BFloat16,
//     packedOutput []float32,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int)
TEXT ·avx512LargeKernelBFloat16Asm(SB), NOSPLIT, $0-112
	MOVQ packedLHS_base+0(FP), R8        // R8 = lhsBasePtr (bfloat16 = 2 bytes)
	MOVQ packedRHS_base+24(FP), R9       // R9 = rhsBasePtr (bfloat16 = 2 bytes)
	MOVQ packedOutput_base+48(FP), R10   // R10 = outBasePtr (float32 = 4 bytes)
	MOVQ rhsPanelCols+80(FP), R11        // R11 = outputStride (cols)
	MOVQ contractingLen+88(FP), R12      // R12 = contractingLen (K)
	MOVQ lhsActiveRows+96(FP), R13       // R13 = lhsActiveRows (M)
	MOVQ rhsActiveCols+104(FP), R14      // R14 = rhsActiveCols (N)

	SHLQ $2, R11                         // R11 = outputStride in bytes (float32 = 4 bytes)

	XORQ AX, AX                          // AX = lhsRowIdx = 0

loop_lhs_bf16:
	CMPQ AX, R13
	JGE done_bf16

	XORQ BX, BX                          // BX = rhsColIdx = 0

loop_rhs_bf16:
	CMPQ BX, R14
	JGE next_lhs_bf16

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

	// 2. Compute lhsPtr and rhsPtr (elements are bfloat16 = 2 bytes)
	// idxLHS = lhsRowIdx * contractingLen
	MOVQ AX, SI
	IMULQ R12, SI
	SHLQ $1, SI                          // SI = byte offset (bfloat16 = 2 bytes)
	LEAQ (R8)(SI*1), SI                  // SI = lhsPtr

	// idxRHS = rhsColIdx * contractingLen
	MOVQ BX, DI
	IMULQ R12, DI
	SHLQ $1, DI                          // DI = byte offset (bfloat16 = 2 bytes)
	LEAQ (R9)(DI*1), DI                  // DI = rhsPtr

	// 3. K-loop with 2-stage ping-pong pipeline (unroll by 2)
	MOVQ R12, CX                         // CX = contractingLen
	TESTQ CX, CX
	JLE store_output_bf16

	SHRQ $1, CX                          // CX = pairs = contractingLen / 2
	JZ k_odd_bf16                        // If 0 pairs (contractingLen == 1), do odd iteration

	// Prime Buffer A for Step 0 outside the loop
	// 4 RHS vectors (64 bfloat16s = 128 bytes) converted to float32:
	VPMOVZXWD (DI), Z16
	VPSLLD $16, Z16, Z16
	VPMOVZXWD 32(DI), Z17
	VPSLLD $16, Z17, Z17
	VPMOVZXWD 64(DI), Z18
	VPSLLD $16, Z18, Z18
	VPMOVZXWD 96(DI), Z19
	VPSLLD $16, Z19, Z19

	// 4 LHS scalars (4 bfloat16s = 8 bytes) converted to float32:
	VPMOVZXWD (SI), X20
	VPSLLD $16, X20, X20
	VPERMILPS $0x55, X20, X21
	VPERMILPS $0xAA, X20, X22
	VPERMILPS $0xFF, X20, X23
	VBROADCASTSS X20, Z20
	VBROADCASTSS X21, Z21
	VBROADCASTSS X22, Z22
	VBROADCASTSS X23, Z23

	ADDQ $8, SI                          // Advance past Step 0 (4 bfloat16s = 8 bytes)
	ADDQ $128, DI                        // Advance past Step 0 (64 bfloat16s = 128 bytes)

	DECQ CX                              // CX = pairs - 1
	JZ k_last_pair_bf16                  // If only 1 pair total, skip loop directly to k_last_pair_bf16

	PCALIGN $64
k_pair_loop_bf16:
	// Step A: Row 0 FMAs interleaved with Buffer B RHS loads (Z24..Z27)
	VFMADD231PS Z16, Z20, Z0
	VPMOVZXWD (DI), Z24
	VPSLLD $16, Z24, Z24
	VFMADD231PS Z17, Z20, Z1
	VPMOVZXWD 32(DI), Z25
	VPSLLD $16, Z25, Z25
	VFMADD231PS Z18, Z20, Z2
	VPMOVZXWD 64(DI), Z26
	VPSLLD $16, Z26, Z26
	VFMADD231PS Z19, Z20, Z3
	VPMOVZXWD 96(DI), Z27
	VPSLLD $16, Z27, Z27

	// Step A: Row 1 FMAs interleaved with Buffer B LHS loads (Z28..Z31)
	VFMADD231PS Z16, Z21, Z4
	VPMOVZXWD (SI), X28
	VPSLLD $16, X28, X28
	VPERMILPS $0x55, X28, X29
	VFMADD231PS Z17, Z21, Z5
	VPERMILPS $0xAA, X28, X30
	VPERMILPS $0xFF, X28, X31
	VFMADD231PS Z18, Z21, Z6
	VBROADCASTSS X28, Z28
	VBROADCASTSS X29, Z29
	VBROADCASTSS X30, Z30
	VBROADCASTSS X31, Z31
	VFMADD231PS Z19, Z21, Z7

	ADDQ $8, SI                          // Advance past Step B
	ADDQ $128, DI

	// Step A: Rows 2-3 pure FMAs
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
	VPMOVZXWD (DI), Z16
	VPSLLD $16, Z16, Z16
	VFMADD231PS Z25, Z28, Z1
	VPMOVZXWD 32(DI), Z17
	VPSLLD $16, Z17, Z17
	VFMADD231PS Z26, Z28, Z2
	VPMOVZXWD 64(DI), Z18
	VPSLLD $16, Z18, Z18
	VFMADD231PS Z27, Z28, Z3
	VPMOVZXWD 96(DI), Z19
	VPSLLD $16, Z19, Z19

	// Step B: Row 1 FMAs interleaved with next-iter Buffer A LHS loads (Z20..Z23)
	VFMADD231PS Z24, Z29, Z4
	VPMOVZXWD (SI), X20
	VPSLLD $16, X20, X20
	VPERMILPS $0x55, X20, X21
	VFMADD231PS Z25, Z29, Z5
	VPERMILPS $0xAA, X20, X22
	VPERMILPS $0xFF, X20, X23
	VFMADD231PS Z26, Z29, Z6
	VBROADCASTSS X20, Z20
	VBROADCASTSS X21, Z21
	VBROADCASTSS X22, Z22
	VBROADCASTSS X23, Z23
	VFMADD231PS Z27, Z29, Z7

	ADDQ $8, SI                          // Advance past Step A of next iter
	ADDQ $128, DI

	// Step B: Rows 2-3 pure FMAs
	VFMADD231PS Z24, Z30, Z8
	VFMADD231PS Z25, Z30, Z9
	VFMADD231PS Z26, Z30, Z10
	VFMADD231PS Z27, Z30, Z11

	VFMADD231PS Z24, Z31, Z12
	VFMADD231PS Z25, Z31, Z13
	VFMADD231PS Z26, Z31, Z14
	VFMADD231PS Z27, Z31, Z15

	DECQ CX
	JNZ k_pair_loop_bf16

k_last_pair_bf16:
	// Step A: Row 0 FMAs interleaved with Buffer B RHS loads
	VFMADD231PS Z16, Z20, Z0
	VPMOVZXWD (DI), Z24
	VPSLLD $16, Z24, Z24
	VFMADD231PS Z17, Z20, Z1
	VPMOVZXWD 32(DI), Z25
	VPSLLD $16, Z25, Z25
	VFMADD231PS Z18, Z20, Z2
	VPMOVZXWD 64(DI), Z26
	VPSLLD $16, Z26, Z26
	VFMADD231PS Z19, Z20, Z3
	VPMOVZXWD 96(DI), Z27
	VPSLLD $16, Z27, Z27

	// Step A: Row 1 FMAs interleaved with Buffer B LHS loads
	VFMADD231PS Z16, Z21, Z4
	VPMOVZXWD (SI), X28
	VPSLLD $16, X28, X28
	VPERMILPS $0x55, X28, X29
	VFMADD231PS Z17, Z21, Z5
	VPERMILPS $0xAA, X28, X30
	VPERMILPS $0xFF, X28, X31
	VFMADD231PS Z18, Z21, Z6
	VBROADCASTSS X28, Z28
	VBROADCASTSS X29, Z29
	VBROADCASTSS X30, Z30
	VBROADCASTSS X31, Z31
	VFMADD231PS Z19, Z21, Z7

	ADDQ $8, SI
	ADDQ $128, DI

	// Step A: Rows 2-3 pure FMAs
	VFMADD231PS Z16, Z22, Z8
	VFMADD231PS Z17, Z22, Z9
	VFMADD231PS Z18, Z22, Z10
	VFMADD231PS Z19, Z22, Z11

	VFMADD231PS Z16, Z23, Z12
	VFMADD231PS Z17, Z23, Z13
	VFMADD231PS Z18, Z23, Z14
	VFMADD231PS Z19, Z23, Z15

	// Step B: 16 FMAs using Buffer B (NO loads)
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

k_odd_bf16:
	TESTQ $1, R12
	JZ store_output_bf16

	VPMOVZXWD (DI), Z16
	VPSLLD $16, Z16, Z16
	VPMOVZXWD 32(DI), Z17
	VPSLLD $16, Z17, Z17
	VPMOVZXWD 64(DI), Z18
	VPSLLD $16, Z18, Z18
	VPMOVZXWD 96(DI), Z19
	VPSLLD $16, Z19, Z19

	VPMOVZXWD (SI), X20
	VPSLLD $16, X20, X20
	VPERMILPS $0x55, X20, X21
	VPERMILPS $0xAA, X20, X22
	VPERMILPS $0xFF, X20, X23
	VBROADCASTSS X20, Z20
	VBROADCASTSS X21, Z21
	VBROADCASTSS X22, Z22
	VBROADCASTSS X23, Z23

	// Row 0
	VFMADD231PS Z16, Z20, Z0
	VFMADD231PS Z17, Z20, Z1
	VFMADD231PS Z18, Z20, Z2
	VFMADD231PS Z19, Z20, Z3

	// Row 1
	VFMADD231PS Z16, Z21, Z4
	VFMADD231PS Z17, Z21, Z5
	VFMADD231PS Z18, Z21, Z6
	VFMADD231PS Z19, Z21, Z7

	// Row 2
	VFMADD231PS Z16, Z22, Z8
	VFMADD231PS Z17, Z22, Z9
	VFMADD231PS Z18, Z22, Z10
	VFMADD231PS Z19, Z22, Z11

	// Row 3
	VFMADD231PS Z16, Z23, Z12
	VFMADD231PS Z17, Z23, Z13
	VFMADD231PS Z18, Z23, Z14
	VFMADD231PS Z19, Z23, Z15

store_output_bf16:
	// Output is []float32: write back exactly as in Float32
	MOVQ AX, DX
	IMULQ R11, DX
	MOVQ BX, CX
	SHLQ $2, CX
	ADDQ CX, DX
	LEAQ (R10)(DX*1), DX

	// Row 0
	VMOVDQU32 Z0, (DX)
	VMOVDQU32 Z1, 64(DX)
	VMOVDQU32 Z2, 128(DX)
	VMOVDQU32 Z3, 192(DX)

	// Row 1
	ADDQ R11, DX
	VMOVDQU32 Z4, (DX)
	VMOVDQU32 Z5, 64(DX)
	VMOVDQU32 Z6, 128(DX)
	VMOVDQU32 Z7, 192(DX)

	// Row 2
	ADDQ R11, DX
	VMOVDQU32 Z8, (DX)
	VMOVDQU32 Z9, 64(DX)
	VMOVDQU32 Z10, 128(DX)
	VMOVDQU32 Z11, 192(DX)

	// Row 3
	ADDQ R11, DX
	VMOVDQU32 Z12, (DX)
	VMOVDQU32 Z13, 64(DX)
	VMOVDQU32 Z14, 128(DX)
	VMOVDQU32 Z15, 192(DX)

	ADDQ $64, BX
	JMP loop_rhs_bf16

next_lhs_bf16:
	ADDQ $4, AX
	JMP loop_lhs_bf16

done_bf16:
	RET

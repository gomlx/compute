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

	// 3. K-loop
	MOVQ R12, CX                         // CX = k counter = contractingLen
	TESTQ CX, CX
	JLE store_output

	PCALIGN $64
k_loop:
	// Load 4 RHS vectors (64 floats = 256 bytes)
	VMOVDQU32 (DI), Z16
	VMOVDQU32 64(DI), Z17
	VMOVDQU32 128(DI), Z18
	VMOVDQU32 192(DI), Z19

	// Row 0
	VBROADCASTSS (SI), Z20
	VFMADD231PS Z16, Z20, Z0
	VFMADD231PS Z17, Z20, Z1
	VFMADD231PS Z18, Z20, Z2
	VFMADD231PS Z19, Z20, Z3

	// Row 1
	VBROADCASTSS 4(SI), Z21
	VFMADD231PS Z16, Z21, Z4
	VFMADD231PS Z17, Z21, Z5
	VFMADD231PS Z18, Z21, Z6
	VFMADD231PS Z19, Z21, Z7

	// Row 2
	VBROADCASTSS 8(SI), Z22
	VFMADD231PS Z16, Z22, Z8
	VFMADD231PS Z17, Z22, Z9
	VFMADD231PS Z18, Z22, Z10
	VFMADD231PS Z19, Z22, Z11

	// Row 3
	VBROADCASTSS 12(SI), Z23
	VFMADD231PS Z16, Z23, Z12
	VFMADD231PS Z17, Z23, Z13
	VFMADD231PS Z18, Z23, Z14
	VFMADD231PS Z19, Z23, Z15

	ADDQ $16, SI                         // lhsPtr += kernelRows * 4 = 16
	ADDQ $256, DI                        // rhsPtr += kernelCols * 4 = 256
	DECQ CX
	JNZ k_loop

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

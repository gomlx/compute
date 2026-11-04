// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// FMLA Vd.4S, Vn.4S, Vm.S[lane]
// Accumulates Vd.4S += Vn.4S * Vm.S[lane]
#define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// ============================================================================
// 1. neonLargeKernelFloat32Asm: 8x8 Microkernel (16 accumulators V0-V15)
// ============================================================================
// func neonLargeKernelFloat32Asm(
//     packedLHS, packedRHS, packedOutput []float32,
//     lhsPanelRows, rhsPanelCols int,
//     contractingLen int,
//     lhsActiveRows, rhsActiveCols int,
//     accumulate bool)
TEXT ·neonLargeKernelFloat32Asm(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0        // R0 = lhsBasePtr
	MOVD packedRHS_base+24(FP), R1       // R1 = rhsBasePtr
	MOVD packedOutput_base+48(FP), R2    // R2 = outBasePtr
	MOVD rhsPanelCols+80(FP), R3         // R3 = outputStride (cols)
	MOVD contractingLen+88(FP), R4       // R4 = contractingLen (K)
	MOVD lhsActiveRows+96(FP), R5        // R5 = lhsActiveRows (M)
	MOVD rhsActiveCols+104(FP), R6       // R6 = rhsActiveCols (N)
	MOVBU accumulate+112(FP), R7         // R7 = accumulate (0 = overwrite, 1 = add)

	LSL $2, R3, R3                       // R3 = outputStride in bytes (cols * 4)

	MOVD $0, R8                          // R8 = lhsRowIdx = 0

loop_lhs_8x8:
	CMP R5, R8
	BGE done_8x8

	MOVD $0, R9                          // R9 = rhsColIdx = 0

loop_rhs_8x8:
	CMP R6, R9
	BGE next_lhs_8x8

	// 1. Zero all 16 accumulators (V0..V15)
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	// 2. Compute lhsPtr (R11) and rhsPtr (R12)
	// R11 = R0 + (lhsRowIdx * contractingLen * 4)
	MUL R4, R8, R11
	ADD R11<<2, R0, R11

	// R12 = R1 + (rhsColIdx * contractingLen * 4)
	MUL R4, R9, R12
	ADD R12<<2, R1, R12

	// 3. K-loop with 2-stage ping-pong pipeline (unroll by 2)
	CMP $0, R4
	BLE store_output_8x8

	LSR $1, R4, R13                      // R13 = pairs = contractingLen / 2
	CBZ R13, k_odd_8x8

	// Prime Buffer A for Step 0
	VLD1.P 32(R11), [V16.S4, V17.S4]     // LHS rows 0..3 (V16), 4..7 (V17)
	VLD1.P 32(R12), [V18.S4, V19.S4]     // RHS cols 0..3 (V18), 4..7 (V19)

	SUB $1, R13, R13
	CBZ R13, k_pair_tail_8x8

k_unroll2_loop_8x8:
	// --- Step 0 (uses Buffer A: V16-V19), interleaved with Buffer B loads (V20-V23) ---
	VLD1.P 32(R11), [V20.S4, V21.S4]     // Load next LHS (Buffer B)
	FMLA_4S_LANE(0, 18, 16, 0)
	FMLA_4S_LANE(1, 19, 16, 0)
	FMLA_4S_LANE(2, 18, 16, 1)
	FMLA_4S_LANE(3, 19, 16, 1)

	VLD1.P 32(R12), [V22.S4, V23.S4]     // Load next RHS (Buffer B)
	FMLA_4S_LANE(4, 18, 16, 2)
	FMLA_4S_LANE(5, 19, 16, 2)
	FMLA_4S_LANE(6, 18, 16, 3)
	FMLA_4S_LANE(7, 19, 16, 3)
	FMLA_4S_LANE(8, 18, 17, 0)
	FMLA_4S_LANE(9, 19, 17, 0)
	FMLA_4S_LANE(10, 18, 17, 1)
	FMLA_4S_LANE(11, 19, 17, 1)
	FMLA_4S_LANE(12, 18, 17, 2)
	FMLA_4S_LANE(13, 19, 17, 2)
	FMLA_4S_LANE(14, 18, 17, 3)
	FMLA_4S_LANE(15, 19, 17, 3)

	// --- Step 1 (uses Buffer B: V20-V23), interleaved with Buffer A loads (V16-V19) ---
	VLD1.P 32(R11), [V16.S4, V17.S4]     // Load next LHS (Buffer A)
	FMLA_4S_LANE(0, 22, 20, 0)
	FMLA_4S_LANE(1, 23, 20, 0)
	FMLA_4S_LANE(2, 22, 20, 1)
	FMLA_4S_LANE(3, 23, 20, 1)

	VLD1.P 32(R12), [V18.S4, V19.S4]     // Load next RHS (Buffer A)
	FMLA_4S_LANE(4, 22, 20, 2)
	FMLA_4S_LANE(5, 23, 20, 2)
	FMLA_4S_LANE(6, 22, 20, 3)
	FMLA_4S_LANE(7, 23, 20, 3)
	FMLA_4S_LANE(8, 22, 21, 0)
	FMLA_4S_LANE(9, 23, 21, 0)
	FMLA_4S_LANE(10, 22, 21, 1)
	FMLA_4S_LANE(11, 23, 21, 1)
	FMLA_4S_LANE(12, 22, 21, 2)
	FMLA_4S_LANE(13, 23, 21, 2)
	FMLA_4S_LANE(14, 22, 21, 3)
	FMLA_4S_LANE(15, 23, 21, 3)

	SUB $1, R13, R13
	CBNZ R13, k_unroll2_loop_8x8

k_pair_tail_8x8:
	// Final pair: Buffer A is already loaded
	VLD1.P 32(R11), [V20.S4, V21.S4]
	FMLA_4S_LANE(0, 18, 16, 0)
	FMLA_4S_LANE(1, 19, 16, 0)
	FMLA_4S_LANE(2, 18, 16, 1)
	FMLA_4S_LANE(3, 19, 16, 1)

	VLD1.P 32(R12), [V22.S4, V23.S4]
	FMLA_4S_LANE(4, 18, 16, 2)
	FMLA_4S_LANE(5, 19, 16, 2)
	FMLA_4S_LANE(6, 18, 16, 3)
	FMLA_4S_LANE(7, 19, 16, 3)
	FMLA_4S_LANE(8, 18, 17, 0)
	FMLA_4S_LANE(9, 19, 17, 0)
	FMLA_4S_LANE(10, 18, 17, 1)
	FMLA_4S_LANE(11, 19, 17, 1)
	FMLA_4S_LANE(12, 18, 17, 2)
	FMLA_4S_LANE(13, 19, 17, 2)
	FMLA_4S_LANE(14, 18, 17, 3)
	FMLA_4S_LANE(15, 19, 17, 3)

	// Step 1 of final pair (Buffer B)
	FMLA_4S_LANE(0, 22, 20, 0)
	FMLA_4S_LANE(1, 23, 20, 0)
	FMLA_4S_LANE(2, 22, 20, 1)
	FMLA_4S_LANE(3, 23, 20, 1)
	FMLA_4S_LANE(4, 22, 20, 2)
	FMLA_4S_LANE(5, 23, 20, 2)
	FMLA_4S_LANE(6, 22, 20, 3)
	FMLA_4S_LANE(7, 23, 20, 3)
	FMLA_4S_LANE(8, 22, 21, 0)
	FMLA_4S_LANE(9, 23, 21, 0)
	FMLA_4S_LANE(10, 22, 21, 1)
	FMLA_4S_LANE(11, 23, 21, 1)
	FMLA_4S_LANE(12, 22, 21, 2)
	FMLA_4S_LANE(13, 23, 21, 2)
	FMLA_4S_LANE(14, 22, 21, 3)
	FMLA_4S_LANE(15, 23, 21, 3)

	TBZ $0, R4, store_output_8x8

k_odd_8x8:
	VLD1.P 32(R11), [V16.S4, V17.S4]
	VLD1.P 32(R12), [V18.S4, V19.S4]
	FMLA_4S_LANE(0, 18, 16, 0)
	FMLA_4S_LANE(1, 19, 16, 0)
	FMLA_4S_LANE(2, 18, 16, 1)
	FMLA_4S_LANE(3, 19, 16, 1)
	FMLA_4S_LANE(4, 18, 16, 2)
	FMLA_4S_LANE(5, 19, 16, 2)
	FMLA_4S_LANE(6, 18, 16, 3)
	FMLA_4S_LANE(7, 19, 16, 3)
	FMLA_4S_LANE(8, 18, 17, 0)
	FMLA_4S_LANE(9, 19, 17, 0)
	FMLA_4S_LANE(10, 18, 17, 1)
	FMLA_4S_LANE(11, 19, 17, 1)
	FMLA_4S_LANE(12, 18, 17, 2)
	FMLA_4S_LANE(13, 19, 17, 2)
	FMLA_4S_LANE(14, 18, 17, 3)
	FMLA_4S_LANE(15, 19, 17, 3)

store_output_8x8:
	// Compute output pointer R14 = R2 + lhsRowIdx*outputStrideBytes + rhsColIdx*4
	MUL R3, R8, R14
	ADD R9<<2, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_8x8

	// Overwrite output (8 rows x 8 cols)
	VST1 [V0.S4, V1.S4], (R14)
	ADD R3, R14, R14
	VST1 [V2.S4, V3.S4], (R14)
	ADD R3, R14, R14
	VST1 [V4.S4, V5.S4], (R14)
	ADD R3, R14, R14
	VST1 [V6.S4, V7.S4], (R14)
	ADD R3, R14, R14
	VST1 [V8.S4, V9.S4], (R14)
	ADD R3, R14, R14
	VST1 [V10.S4, V11.S4], (R14)
	ADD R3, R14, R14
	VST1 [V12.S4, V13.S4], (R14)
	ADD R3, R14, R14
	VST1 [V14.S4, V15.S4], (R14)
	B next_rhs_8x8

accumulate_output_8x8:
	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VST1 [V0.S4, V1.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V2.S4, V2.S4
	VFADD V17.S4, V3.S4, V3.S4
	VST1 [V2.S4, V3.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V4.S4, V4.S4
	VFADD V17.S4, V5.S4, V5.S4
	VST1 [V4.S4, V5.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V6.S4, V6.S4
	VFADD V17.S4, V7.S4, V7.S4
	VST1 [V6.S4, V7.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V8.S4, V8.S4
	VFADD V17.S4, V9.S4, V9.S4
	VST1 [V8.S4, V9.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V10.S4, V10.S4
	VFADD V17.S4, V11.S4, V11.S4
	VST1 [V10.S4, V11.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V12.S4, V12.S4
	VFADD V17.S4, V13.S4, V13.S4
	VST1 [V12.S4, V13.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4]
	VFADD V16.S4, V14.S4, V14.S4
	VFADD V17.S4, V15.S4, V15.S4
	VST1 [V14.S4, V15.S4], (R14)

next_rhs_8x8:
	ADD $8, R9, R9
	B loop_rhs_8x8

next_lhs_8x8:
	ADD $8, R8, R8
	B loop_lhs_8x8

done_8x8:
	RET


// ============================================================================
// 2. neonLargeKernelFloat32Asm_4x16: 4x16 Microkernel (16 accumulators V0-V15)
// ============================================================================
TEXT ·neonLargeKernelFloat32Asm_4x16(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3

	MOVD $0, R8

loop_lhs_4x16:
	CMP R5, R8
	BGE done_4x16

	MOVD $0, R9

loop_rhs_4x16:
	CMP R6, R9
	BGE next_lhs_4x16

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	MUL R4, R8, R11
	ADD R11<<2, R0, R11

	MUL R4, R9, R12
	ADD R12<<2, R1, R12

	CMP $0, R4
	BLE store_output_4x16

	LSR $1, R4, R13
	CBZ R13, k_odd_4x16

	// Prime Buffer A (LHS: V16, RHS: V17..V20)
	VLD1.P 16(R11), [V16.S4]
	VLD1.P 64(R12), [V17.S4, V18.S4, V19.S4, V20.S4]

	SUB $1, R13, R13
	CBZ R13, k_pair_tail_4x16

k_unroll2_loop_4x16:
	// Step 0 (Buffer A: V16, V17..V20), load Buffer B (V21, V22..V25)
	VLD1.P 16(R11), [V21.S4]
	FMLA_4S_LANE(0, 17, 16, 0)
	FMLA_4S_LANE(1, 18, 16, 0)
	FMLA_4S_LANE(2, 19, 16, 0)
	FMLA_4S_LANE(3, 20, 16, 0)

	VLD1.P 64(R12), [V22.S4, V23.S4, V24.S4, V25.S4]
	FMLA_4S_LANE(4, 17, 16, 1)
	FMLA_4S_LANE(5, 18, 16, 1)
	FMLA_4S_LANE(6, 19, 16, 1)
	FMLA_4S_LANE(7, 20, 16, 1)
	FMLA_4S_LANE(8, 17, 16, 2)
	FMLA_4S_LANE(9, 18, 16, 2)
	FMLA_4S_LANE(10, 19, 16, 2)
	FMLA_4S_LANE(11, 20, 16, 2)
	FMLA_4S_LANE(12, 17, 16, 3)
	FMLA_4S_LANE(13, 18, 16, 3)
	FMLA_4S_LANE(14, 19, 16, 3)
	FMLA_4S_LANE(15, 20, 16, 3)

	// Step 1 (Buffer B: V21, V22..V25), load Buffer A (V16, V17..V20)
	VLD1.P 16(R11), [V16.S4]
	FMLA_4S_LANE(0, 22, 21, 0)
	FMLA_4S_LANE(1, 23, 21, 0)
	FMLA_4S_LANE(2, 24, 21, 0)
	FMLA_4S_LANE(3, 25, 21, 0)

	VLD1.P 64(R12), [V17.S4, V18.S4, V19.S4, V20.S4]
	FMLA_4S_LANE(4, 22, 21, 1)
	FMLA_4S_LANE(5, 23, 21, 1)
	FMLA_4S_LANE(6, 24, 21, 1)
	FMLA_4S_LANE(7, 25, 21, 1)
	FMLA_4S_LANE(8, 22, 21, 2)
	FMLA_4S_LANE(9, 23, 21, 2)
	FMLA_4S_LANE(10, 24, 21, 2)
	FMLA_4S_LANE(11, 25, 21, 2)
	FMLA_4S_LANE(12, 22, 21, 3)
	FMLA_4S_LANE(13, 23, 21, 3)
	FMLA_4S_LANE(14, 24, 21, 3)
	FMLA_4S_LANE(15, 25, 21, 3)

	SUB $1, R13, R13
	CBNZ R13, k_unroll2_loop_4x16

k_pair_tail_4x16:
	VLD1.P 16(R11), [V21.S4]
	FMLA_4S_LANE(0, 17, 16, 0)
	FMLA_4S_LANE(1, 18, 16, 0)
	FMLA_4S_LANE(2, 19, 16, 0)
	FMLA_4S_LANE(3, 20, 16, 0)

	VLD1.P 64(R12), [V22.S4, V23.S4, V24.S4, V25.S4]
	FMLA_4S_LANE(4, 17, 16, 1)
	FMLA_4S_LANE(5, 18, 16, 1)
	FMLA_4S_LANE(6, 19, 16, 1)
	FMLA_4S_LANE(7, 20, 16, 1)
	FMLA_4S_LANE(8, 17, 16, 2)
	FMLA_4S_LANE(9, 18, 16, 2)
	FMLA_4S_LANE(10, 19, 16, 2)
	FMLA_4S_LANE(11, 20, 16, 2)
	FMLA_4S_LANE(12, 17, 16, 3)
	FMLA_4S_LANE(13, 18, 16, 3)
	FMLA_4S_LANE(14, 19, 16, 3)
	FMLA_4S_LANE(15, 20, 16, 3)

	FMLA_4S_LANE(0, 22, 21, 0)
	FMLA_4S_LANE(1, 23, 21, 0)
	FMLA_4S_LANE(2, 24, 21, 0)
	FMLA_4S_LANE(3, 25, 21, 0)
	FMLA_4S_LANE(4, 22, 21, 1)
	FMLA_4S_LANE(5, 23, 21, 1)
	FMLA_4S_LANE(6, 24, 21, 1)
	FMLA_4S_LANE(7, 25, 21, 1)
	FMLA_4S_LANE(8, 22, 21, 2)
	FMLA_4S_LANE(9, 23, 21, 2)
	FMLA_4S_LANE(10, 24, 21, 2)
	FMLA_4S_LANE(11, 25, 21, 2)
	FMLA_4S_LANE(12, 22, 21, 3)
	FMLA_4S_LANE(13, 23, 21, 3)
	FMLA_4S_LANE(14, 24, 21, 3)
	FMLA_4S_LANE(15, 25, 21, 3)

	TBZ $0, R4, store_output_4x16

k_odd_4x16:
	VLD1.P 16(R11), [V16.S4]
	VLD1.P 64(R12), [V17.S4, V18.S4, V19.S4, V20.S4]
	FMLA_4S_LANE(0, 17, 16, 0)
	FMLA_4S_LANE(1, 18, 16, 0)
	FMLA_4S_LANE(2, 19, 16, 0)
	FMLA_4S_LANE(3, 20, 16, 0)
	FMLA_4S_LANE(4, 17, 16, 1)
	FMLA_4S_LANE(5, 18, 16, 1)
	FMLA_4S_LANE(6, 19, 16, 1)
	FMLA_4S_LANE(7, 20, 16, 1)
	FMLA_4S_LANE(8, 17, 16, 2)
	FMLA_4S_LANE(9, 18, 16, 2)
	FMLA_4S_LANE(10, 19, 16, 2)
	FMLA_4S_LANE(11, 20, 16, 2)
	FMLA_4S_LANE(12, 17, 16, 3)
	FMLA_4S_LANE(13, 18, 16, 3)
	FMLA_4S_LANE(14, 19, 16, 3)
	FMLA_4S_LANE(15, 20, 16, 3)

store_output_4x16:
	MUL R3, R8, R14
	ADD R9<<2, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_4x16

	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R14)
	ADD R3, R14, R14
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R14)
	ADD R3, R14, R14
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R14)
	ADD R3, R14, R14
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R14)
	B next_rhs_4x16

accumulate_output_4x16:
	VLD1 (R14), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VFADD V18.S4, V2.S4, V2.S4
	VFADD V19.S4, V3.S4, V3.S4
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V4.S4, V4.S4
	VFADD V17.S4, V5.S4, V5.S4
	VFADD V18.S4, V6.S4, V6.S4
	VFADD V19.S4, V7.S4, V7.S4
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V8.S4, V8.S4
	VFADD V17.S4, V9.S4, V9.S4
	VFADD V18.S4, V10.S4, V10.S4
	VFADD V19.S4, V11.S4, V11.S4
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V12.S4, V12.S4
	VFADD V17.S4, V13.S4, V13.S4
	VFADD V18.S4, V14.S4, V14.S4
	VFADD V19.S4, V15.S4, V15.S4
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R14)

next_rhs_4x16:
	ADD $16, R9, R9
	B loop_rhs_4x16

next_lhs_4x16:
	ADD $4, R8, R8
	B loop_lhs_4x16

done_4x16:
	RET


// ============================================================================
// 3. neonLargeKernelFloat32Asm_8x12: 8x12 Microkernel (24 accumulators V0-V23)
// ============================================================================
TEXT ·neonLargeKernelFloat32Asm_8x12(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3

	MOVD $0, R8

loop_lhs_8x12:
	CMP R5, R8
	BGE done_8x12

	MOVD $0, R9

loop_rhs_8x12:
	CMP R6, R9
	BGE next_lhs_8x12

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16

	MUL R4, R8, R11
	ADD R11<<2, R0, R11

	MUL R4, R9, R12
	ADD R12<<2, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_8x12

k_loop_8x12:
	VLD1.P 32(R11), [V24.S4, V25.S4]         // LHS rows 0..3 (V24), 4..7 (V25)
	VLD1.P 48(R12), [V26.S4, V27.S4, V28.S4] // RHS cols 0..3 (V26), 4..7 (V27), 8..11 (V28)

	// Row 0
	FMLA_4S_LANE(0, 26, 24, 0)
	FMLA_4S_LANE(1, 27, 24, 0)
	FMLA_4S_LANE(2, 28, 24, 0)
	// Row 1
	FMLA_4S_LANE(3, 26, 24, 1)
	FMLA_4S_LANE(4, 27, 24, 1)
	FMLA_4S_LANE(5, 28, 24, 1)
	// Row 2
	FMLA_4S_LANE(6, 26, 24, 2)
	FMLA_4S_LANE(7, 27, 24, 2)
	FMLA_4S_LANE(8, 28, 24, 2)
	// Row 3
	FMLA_4S_LANE(9, 26, 24, 3)
	FMLA_4S_LANE(10, 27, 24, 3)
	FMLA_4S_LANE(11, 28, 24, 3)
	// Row 4
	FMLA_4S_LANE(12, 26, 25, 0)
	FMLA_4S_LANE(13, 27, 25, 0)
	FMLA_4S_LANE(14, 28, 25, 0)
	// Row 5
	FMLA_4S_LANE(15, 26, 25, 1)
	FMLA_4S_LANE(16, 27, 25, 1)
	FMLA_4S_LANE(17, 28, 25, 1)
	// Row 6
	FMLA_4S_LANE(18, 26, 25, 2)
	FMLA_4S_LANE(19, 27, 25, 2)
	FMLA_4S_LANE(20, 28, 25, 2)
	// Row 7
	FMLA_4S_LANE(21, 26, 25, 3)
	FMLA_4S_LANE(22, 27, 25, 3)
	FMLA_4S_LANE(23, 28, 25, 3)

	SUB $1, R13, R13
	CBNZ R13, k_loop_8x12

store_output_8x12:
	MUL R3, R8, R14
	ADD R9<<2, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_8x12

	VST1 [V0.S4, V1.S4, V2.S4], (R14)
	ADD R3, R14, R14
	VST1 [V3.S4, V4.S4, V5.S4], (R14)
	ADD R3, R14, R14
	VST1 [V6.S4, V7.S4, V8.S4], (R14)
	ADD R3, R14, R14
	VST1 [V9.S4, V10.S4, V11.S4], (R14)
	ADD R3, R14, R14
	VST1 [V12.S4, V13.S4, V14.S4], (R14)
	ADD R3, R14, R14
	VST1 [V15.S4, V16.S4, V17.S4], (R14)
	ADD R3, R14, R14
	VST1 [V18.S4, V19.S4, V20.S4], (R14)
	ADD R3, R14, R14
	VST1 [V21.S4, V22.S4, V23.S4], (R14)
	B next_rhs_8x12

accumulate_output_8x12:
	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V0.S4, V0.S4
	VFADD V25.S4, V1.S4, V1.S4
	VFADD V26.S4, V2.S4, V2.S4
	VST1 [V0.S4, V1.S4, V2.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V3.S4, V3.S4
	VFADD V25.S4, V4.S4, V4.S4
	VFADD V26.S4, V5.S4, V5.S4
	VST1 [V3.S4, V4.S4, V5.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V6.S4, V6.S4
	VFADD V25.S4, V7.S4, V7.S4
	VFADD V26.S4, V8.S4, V8.S4
	VST1 [V6.S4, V7.S4, V8.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V9.S4, V9.S4
	VFADD V25.S4, V10.S4, V10.S4
	VFADD V26.S4, V11.S4, V11.S4
	VST1 [V9.S4, V10.S4, V11.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V12.S4, V12.S4
	VFADD V25.S4, V13.S4, V13.S4
	VFADD V26.S4, V14.S4, V14.S4
	VST1 [V12.S4, V13.S4, V14.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V15.S4, V15.S4
	VFADD V25.S4, V16.S4, V16.S4
	VFADD V26.S4, V17.S4, V17.S4
	VST1 [V15.S4, V16.S4, V17.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V18.S4, V18.S4
	VFADD V25.S4, V19.S4, V19.S4
	VFADD V26.S4, V20.S4, V20.S4
	VST1 [V18.S4, V19.S4, V20.S4], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V24.S4, V25.S4, V26.S4]
	VFADD V24.S4, V21.S4, V21.S4
	VFADD V25.S4, V22.S4, V22.S4
	VFADD V26.S4, V23.S4, V23.S4
	VST1 [V21.S4, V22.S4, V23.S4], (R14)

next_rhs_8x12:
	ADD $12, R9, R9
	B loop_rhs_8x12

next_lhs_8x12:
	ADD $8, R8, R8
	B loop_lhs_8x12

done_8x12:
	RET

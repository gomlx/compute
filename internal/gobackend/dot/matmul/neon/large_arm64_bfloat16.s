// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// FMLA Vd.4S, Vn.4S, Vm.S[lane]
#define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// SHLL Vd.4S, Vn.4H, #16 (widens lower 4 BF16 lanes to 4 FP32 lanes)
#define USHLL_4S_4H_16(Vd, Vn) \
	WORD $(0x2E613800 | ((Vn)<<5) | (Vd))

// SHLL2 Vd.4S, Vn.8H, #16 (widens upper 4 BF16 lanes to 4 FP32 lanes)
#define USHLL2_4S_8H_16(Vd, Vn) \
	WORD $(0x6E613800 | ((Vn)<<5) | (Vd))

// ZIP1 Vd.8H, Vn.8H, Vm.8H
#define ZIP1_8H(Vd, Vn, Vm) \
	WORD $(0x4E403800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// ZIP2 Vd.8H, Vn.8H, Vm.8H
#define ZIP2_8H(Vd, Vn, Vm) \
	WORD $(0x4E407800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// BFDOT Vd.4S, Vn.8H, Vm.2H[lane]
#define BFDOT_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F40F000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// ============================================================================
// 1. neonLargeKernelBFloat16Asm: 8x8 Microkernel (BFloat16 -> Float32)
// ============================================================================
TEXT ·neonLargeKernelBFloat16Asm(SB), NOSPLIT, $0-113
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

loop_lhs_8x8:
	CMP R5, R8
	BGE done_8x8

	MOVD $0, R9

loop_rhs_8x8:
	CMP R6, R9
	BGE next_lhs_8x8

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
	ADD R11<<1, R0, R11

	MUL R4, R9, R12
	ADD R12<<1, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_8x8

k_loop_8x8:
	VLD1.P 16(R11), [V20.H8]             // 8 BF16 LHS values
	VLD1.P 16(R12), [V21.H8]             // 8 BF16 RHS values

	USHLL_4S_4H_16(16, 20)               // V16 = LHS rows 0..3
	USHLL2_4S_8H_16(17, 20)              // V17 = LHS rows 4..7
	USHLL_4S_4H_16(18, 21)               // V18 = RHS cols 0..3
	USHLL2_4S_8H_16(19, 21)              // V19 = RHS cols 4..7

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

	SUB $1, R13, R13
	CBNZ R13, k_loop_8x8

store_output_8x8:
	MUL R3, R8, R14
	ADD R9<<2, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_8x8

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
// 2. neonLargeKernelBFloat16Asm_4x16: 4x16 Microkernel (BFloat16 -> Float32)
// ============================================================================
TEXT ·neonLargeKernelBFloat16Asm_4x16(SB), NOSPLIT, $0-113
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
	ADD R11<<1, R0, R11

	MUL R4, R9, R12
	ADD R12<<1, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_4x16

k_loop_4x16:
	VLD1.P 8(R11), [V21.H4]              // 4 BF16 LHS values
	VLD1.P 32(R12), [V22.H8, V23.H8]     // 16 BF16 RHS values

	USHLL_4S_4H_16(16, 21)               // V16 = LHS rows 0..3
	USHLL_4S_4H_16(17, 22)               // V17 = RHS cols 0..3
	USHLL2_4S_8H_16(18, 22)              // V18 = RHS cols 4..7
	USHLL_4S_4H_16(19, 23)               // V19 = RHS cols 8..11
	USHLL2_4S_8H_16(20, 23)              // V20 = RHS cols 12..15

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

	SUB $1, R13, R13
	CBNZ R13, k_loop_4x16

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
// 3. neonLargeKernelBFloat16BFDOTAsm: 8x8 Microkernel using FEAT_BF16 BFDOT
// ============================================================================
TEXT ·neonLargeKernelBFloat16BFDOTAsm(SB), NOSPLIT, $0-113
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

loop_lhs_bfdot_8x8:
	CMP R5, R8
	BGE done_bfdot_8x8

	MOVD $0, R9

loop_rhs_bfdot_8x8:
	CMP R6, R9
	BGE next_lhs_bfdot_8x8

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
	ADD R11<<1, R0, R11

	MUL R4, R9, R12
	ADD R12<<1, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_bfdot_8x8

k_loop2_bfdot_8x8:
	CMP $2, R13
	BLT k_loop1_bfdot_8x8

	VLD1.P 32(R11), [V20.H8, V21.H8]      // 2 K-steps of 8 BF16 LHS rows
	VLD1.P 32(R12), [V22.H8, V23.H8]      // 2 K-steps of 8 BF16 RHS cols

	ZIP1_8H(16, 20, 21)                   // V16.2H[0..3] = LHS rows 0..3 pairs (k, k+1)
	ZIP2_8H(17, 20, 21)                   // V17.2H[0..3] = LHS rows 4..7 pairs (k, k+1)
	ZIP1_8H(18, 22, 23)                   // V18.H8 = RHS cols 0..3 pairs (k, k+1)
	ZIP2_8H(19, 22, 23)                   // V19.H8 = RHS cols 4..7 pairs (k, k+1)

	BFDOT_LANE(0, 18, 16, 0)
	BFDOT_LANE(1, 19, 16, 0)
	BFDOT_LANE(2, 18, 16, 1)
	BFDOT_LANE(3, 19, 16, 1)
	BFDOT_LANE(4, 18, 16, 2)
	BFDOT_LANE(5, 19, 16, 2)
	BFDOT_LANE(6, 18, 16, 3)
	BFDOT_LANE(7, 19, 16, 3)
	BFDOT_LANE(8, 18, 17, 0)
	BFDOT_LANE(9, 19, 17, 0)
	BFDOT_LANE(10, 18, 17, 1)
	BFDOT_LANE(11, 19, 17, 1)
	BFDOT_LANE(12, 18, 17, 2)
	BFDOT_LANE(13, 19, 17, 2)
	BFDOT_LANE(14, 18, 17, 3)
	BFDOT_LANE(15, 19, 17, 3)

	SUB $2, R13, R13
	B k_loop2_bfdot_8x8

k_loop1_bfdot_8x8:
	CBZ R13, store_output_bfdot_8x8

	VLD1.P 16(R11), [V20.H8]
	VLD1.P 16(R12), [V21.H8]

	USHLL_4S_4H_16(16, 20)
	USHLL2_4S_8H_16(17, 20)
	USHLL_4S_4H_16(18, 21)
	USHLL2_4S_8H_16(19, 21)

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

store_output_bfdot_8x8:
	MUL R3, R8, R14
	ADD R9<<2, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_bfdot_8x8

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
	B next_rhs_bfdot_8x8

accumulate_output_bfdot_8x8:
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

next_rhs_bfdot_8x8:
	ADD $8, R9, R9
	B loop_rhs_bfdot_8x8

next_lhs_bfdot_8x8:
	ADD $8, R8, R8
	B loop_lhs_bfdot_8x8

done_bfdot_8x8:
	RET


// ============================================================================
// 4. neonLargeKernelBFloat16BFDOTAsm_4x16: 4x16 Microkernel using FEAT_BF16 BFDOT
// ============================================================================
TEXT ·neonLargeKernelBFloat16BFDOTAsm_4x16(SB), NOSPLIT, $0-113
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

loop_lhs_bfdot_4x16:
	CMP R5, R8
	BGE done_bfdot_4x16

	MOVD $0, R9

loop_rhs_bfdot_4x16:
	CMP R6, R9
	BGE next_lhs_bfdot_4x16

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
	ADD R11<<1, R0, R11

	MUL R4, R9, R12
	ADD R12<<1, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_bfdot_4x16

k_loop2_bfdot_4x16:
	CMP $2, R13
	BLT k_loop1_bfdot_4x16

	FMOVD.P 8(R11), F21                    // k+0: 4 BF16 LHS values
	FMOVD.P 8(R11), F22                    // k+1: 4 BF16 LHS values
	ZIP1_8H(16, 21, 22)                    // V16.2H[0..3] = LHS rows 0..3 pairs (k, k+1)

	VLD1.P 32(R12), [V24.H8, V25.H8]       // k+0: 16 BF16 RHS values
	VLD1.P 32(R12), [V26.H8, V27.H8]       // k+1: 16 BF16 RHS values
	ZIP1_8H(17, 24, 26)
	ZIP2_8H(18, 24, 26)
	ZIP1_8H(19, 25, 27)
	ZIP2_8H(20, 25, 27)

	BFDOT_LANE(0, 17, 16, 0)
	BFDOT_LANE(1, 18, 16, 0)
	BFDOT_LANE(2, 19, 16, 0)
	BFDOT_LANE(3, 20, 16, 0)
	BFDOT_LANE(4, 17, 16, 1)
	BFDOT_LANE(5, 18, 16, 1)
	BFDOT_LANE(6, 19, 16, 1)
	BFDOT_LANE(7, 20, 16, 1)
	BFDOT_LANE(8, 17, 16, 2)
	BFDOT_LANE(9, 18, 16, 2)
	BFDOT_LANE(10, 19, 16, 2)
	BFDOT_LANE(11, 20, 16, 2)
	BFDOT_LANE(12, 17, 16, 3)
	BFDOT_LANE(13, 18, 16, 3)
	BFDOT_LANE(14, 19, 16, 3)
	BFDOT_LANE(15, 20, 16, 3)

	SUB $2, R13, R13
	B k_loop2_bfdot_4x16

k_loop1_bfdot_4x16:
	CBZ R13, store_output_bfdot_4x16

	VLD1.P 8(R11), [V21.H4]
	VLD1.P 32(R12), [V22.H8, V23.H8]

	USHLL_4S_4H_16(16, 21)
	USHLL_4S_4H_16(17, 22)
	USHLL2_4S_8H_16(18, 22)
	USHLL_4S_4H_16(19, 23)
	USHLL2_4S_8H_16(20, 23)

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

store_output_bfdot_4x16:
	MUL R3, R8, R14
	ADD R9<<2, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_bfdot_4x16

	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R14)
	ADD R3, R14, R14
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R14)
	ADD R3, R14, R14
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R14)
	ADD R3, R14, R14
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R14)
	B next_rhs_bfdot_4x16

accumulate_output_bfdot_4x16:
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

next_rhs_bfdot_4x16:
	ADD $16, R9, R9
	B loop_rhs_bfdot_4x16

next_lhs_bfdot_4x16:
	ADD $4, R8, R8
	B loop_lhs_bfdot_4x16

done_bfdot_4x16:
	RET

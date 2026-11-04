// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// FMLA Vd.4S, Vn.4S, Vm.S[lane]
#define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// FCVTL Vd.4S, Vn.4H (widens lower 4 FP16 lanes to 4 FP32 lanes)
#define FCVTL_4S_4H(Vd, Vn) \
	WORD $(0x0E217800 | ((Vn)<<5) | (Vd))

// FCVTL2 Vd.4S, Vn.8H (widens upper 4 FP16 lanes to 4 FP32 lanes)
#define FCVTL2_4S_8H(Vd, Vn) \
	WORD $(0x4E217800 | ((Vn)<<5) | (Vd))

// ============================================================================
// 1. neonLargeKernelFloat16Asm: 8x8 Microkernel (Float16 -> Float32)
// ============================================================================
TEXT ·neonLargeKernelFloat16Asm(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3                       // R3 = outputStride in bytes (cols * 4)

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

	// Input elements are 2 bytes (Float16)
	MUL R4, R8, R11
	ADD R11<<1, R0, R11

	MUL R4, R9, R12
	ADD R12<<1, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_8x8

k_loop_8x8:
	VLD1.P 16(R11), [V20.H8]             // 8 FP16 LHS values
	VLD1.P 16(R12), [V21.H8]             // 8 FP16 RHS values

	FCVTL_4S_4H(16, 20)                  // V16 = LHS rows 0..3
	FCVTL2_4S_8H(17, 20)                 // V17 = LHS rows 4..7
	FCVTL_4S_4H(18, 21)                  // V18 = RHS cols 0..3
	FCVTL2_4S_8H(19, 21)                 // V19 = RHS cols 4..7

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
// 2. neonLargeKernelFloat16Asm_4x16: 4x16 Microkernel (Float16 -> Float32)
// ============================================================================
TEXT ·neonLargeKernelFloat16Asm_4x16(SB), NOSPLIT, $0-113
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
	VLD1.P 8(R11), [V21.H4]              // 4 FP16 LHS values
	VLD1.P 32(R12), [V22.H8, V23.H8]     // 16 FP16 RHS values

	FCVTL_4S_4H(16, 21)                  // V16 = LHS rows 0..3
	FCVTL_4S_4H(17, 22)                  // V17 = RHS cols 0..3
	FCVTL2_4S_8H(18, 22)                 // V18 = RHS cols 4..7
	FCVTL_4S_4H(19, 23)                  // V19 = RHS cols 8..11
	FCVTL2_4S_8H(20, 23)                 // V20 = RHS cols 12..15

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

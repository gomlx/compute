// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// FMLA Vd.2D, Vn.2D, Vm.D[lane]
#define FMLA_2D_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4FC01000 | (((lane)&1)<<11) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// ============================================================================
// 1. neonLargeKernelFloat64Asm: 8x4 Microkernel (16 accumulators V0-V15)
// ============================================================================
TEXT ·neonLargeKernelFloat64Asm(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $3, R3, R3                       // R3 = outputStride in bytes (cols * 8)

	MOVD $0, R8

loop_lhs_8x4:
	CMP R5, R8
	BGE done_8x4

	MOVD $0, R9

loop_rhs_8x4:
	CMP R6, R9
	BGE next_lhs_8x4

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
	ADD R11<<3, R0, R11

	MUL R4, R9, R12
	ADD R12<<3, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_8x4

k_loop_8x4:
	VLD1.P 64(R11), [V16.D2, V17.D2, V18.D2, V19.D2] // LHS rows 0..7
	VLD1.P 32(R12), [V20.D2, V21.D2]                 // RHS cols 0..3

	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	FMLA_2D_LANE(2, 20, 16, 1)
	FMLA_2D_LANE(3, 21, 16, 1)
	FMLA_2D_LANE(4, 20, 17, 0)
	FMLA_2D_LANE(5, 21, 17, 0)
	FMLA_2D_LANE(6, 20, 17, 1)
	FMLA_2D_LANE(7, 21, 17, 1)
	FMLA_2D_LANE(8, 20, 18, 0)
	FMLA_2D_LANE(9, 21, 18, 0)
	FMLA_2D_LANE(10, 20, 18, 1)
	FMLA_2D_LANE(11, 21, 18, 1)
	FMLA_2D_LANE(12, 20, 19, 0)
	FMLA_2D_LANE(13, 21, 19, 0)
	FMLA_2D_LANE(14, 20, 19, 1)
	FMLA_2D_LANE(15, 21, 19, 1)

	SUB $1, R13, R13
	CBNZ R13, k_loop_8x4

store_output_8x4:
	MUL R3, R8, R14
	ADD R9<<3, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_8x4

	VST1 [V0.D2, V1.D2], (R14)
	ADD R3, R14, R14
	VST1 [V2.D2, V3.D2], (R14)
	ADD R3, R14, R14
	VST1 [V4.D2, V5.D2], (R14)
	ADD R3, R14, R14
	VST1 [V6.D2, V7.D2], (R14)
	ADD R3, R14, R14
	VST1 [V8.D2, V9.D2], (R14)
	ADD R3, R14, R14
	VST1 [V10.D2, V11.D2], (R14)
	ADD R3, R14, R14
	VST1 [V12.D2, V13.D2], (R14)
	ADD R3, R14, R14
	VST1 [V14.D2, V15.D2], (R14)
	B next_rhs_8x4

accumulate_output_8x4:
	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V0.D2, V0.D2
	VFADD V17.D2, V1.D2, V1.D2
	VST1 [V0.D2, V1.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V2.D2, V2.D2
	VFADD V17.D2, V3.D2, V3.D2
	VST1 [V2.D2, V3.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V4.D2, V4.D2
	VFADD V17.D2, V5.D2, V5.D2
	VST1 [V4.D2, V5.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V6.D2, V6.D2
	VFADD V17.D2, V7.D2, V7.D2
	VST1 [V6.D2, V7.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V8.D2, V8.D2
	VFADD V17.D2, V9.D2, V9.D2
	VST1 [V8.D2, V9.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V10.D2, V10.D2
	VFADD V17.D2, V11.D2, V11.D2
	VST1 [V10.D2, V11.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V12.D2, V12.D2
	VFADD V17.D2, V13.D2, V13.D2
	VST1 [V12.D2, V13.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2]
	VFADD V16.D2, V14.D2, V14.D2
	VFADD V17.D2, V15.D2, V15.D2
	VST1 [V14.D2, V15.D2], (R14)

next_rhs_8x4:
	ADD $4, R9, R9
	B loop_rhs_8x4

next_lhs_8x4:
	ADD $8, R8, R8
	B loop_lhs_8x4

done_8x4:
	RET


// ============================================================================
// 2. neonLargeKernelFloat64Asm_4x8: 4x8 Microkernel (16 accumulators V0-V15)
// ============================================================================
TEXT ·neonLargeKernelFloat64Asm_4x8(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $3, R3, R3

	MOVD $0, R8

loop_lhs_4x8:
	CMP R5, R8
	BGE done_4x8

	MOVD $0, R9

loop_rhs_4x8:
	CMP R6, R9
	BGE next_lhs_4x8

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
	ADD R11<<3, R0, R11

	MUL R4, R9, R12
	ADD R12<<3, R1, R12

	MOVD R4, R13
	CMP $0, R13
	BLE store_output_4x8

k_loop_4x8:
	VLD1.P 32(R11), [V16.D2, V17.D2]                 // LHS rows 0..3
	VLD1.P 64(R12), [V18.D2, V19.D2, V20.D2, V21.D2] // RHS cols 0..7

	FMLA_2D_LANE(0, 18, 16, 0)
	FMLA_2D_LANE(1, 19, 16, 0)
	FMLA_2D_LANE(2, 20, 16, 0)
	FMLA_2D_LANE(3, 21, 16, 0)
	FMLA_2D_LANE(4, 18, 16, 1)
	FMLA_2D_LANE(5, 19, 16, 1)
	FMLA_2D_LANE(6, 20, 16, 1)
	FMLA_2D_LANE(7, 21, 16, 1)
	FMLA_2D_LANE(8, 18, 17, 0)
	FMLA_2D_LANE(9, 19, 17, 0)
	FMLA_2D_LANE(10, 20, 17, 0)
	FMLA_2D_LANE(11, 21, 17, 0)
	FMLA_2D_LANE(12, 18, 17, 1)
	FMLA_2D_LANE(13, 19, 17, 1)
	FMLA_2D_LANE(14, 20, 17, 1)
	FMLA_2D_LANE(15, 21, 17, 1)

	SUB $1, R13, R13
	CBNZ R13, k_loop_4x8

store_output_4x8:
	MUL R3, R8, R14
	ADD R9<<3, R14, R14
	ADD R2, R14, R14

	CBNZ R7, accumulate_output_4x8

	VST1 [V0.D2, V1.D2, V2.D2, V3.D2], (R14)
	ADD R3, R14, R14
	VST1 [V4.D2, V5.D2, V6.D2, V7.D2], (R14)
	ADD R3, R14, R14
	VST1 [V8.D2, V9.D2, V10.D2, V11.D2], (R14)
	ADD R3, R14, R14
	VST1 [V12.D2, V13.D2, V14.D2, V15.D2], (R14)
	B next_rhs_4x8

accumulate_output_4x8:
	VLD1 (R14), [V16.D2, V17.D2, V18.D2, V19.D2]
	VFADD V16.D2, V0.D2, V0.D2
	VFADD V17.D2, V1.D2, V1.D2
	VFADD V18.D2, V2.D2, V2.D2
	VFADD V19.D2, V3.D2, V3.D2
	VST1 [V0.D2, V1.D2, V2.D2, V3.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2, V18.D2, V19.D2]
	VFADD V16.D2, V4.D2, V4.D2
	VFADD V17.D2, V5.D2, V5.D2
	VFADD V18.D2, V6.D2, V6.D2
	VFADD V19.D2, V7.D2, V7.D2
	VST1 [V4.D2, V5.D2, V6.D2, V7.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2, V18.D2, V19.D2]
	VFADD V16.D2, V8.D2, V8.D2
	VFADD V17.D2, V9.D2, V9.D2
	VFADD V18.D2, V10.D2, V10.D2
	VFADD V19.D2, V11.D2, V11.D2
	VST1 [V8.D2, V9.D2, V10.D2, V11.D2], (R14)
	ADD R3, R14, R14

	VLD1 (R14), [V16.D2, V17.D2, V18.D2, V19.D2]
	VFADD V16.D2, V12.D2, V12.D2
	VFADD V17.D2, V13.D2, V13.D2
	VFADD V18.D2, V14.D2, V14.D2
	VFADD V19.D2, V15.D2, V15.D2
	VST1 [V12.D2, V13.D2, V14.D2, V15.D2], (R14)

next_rhs_4x8:
	ADD $8, R9, R9
	B loop_rhs_4x8

next_lhs_4x8:
	ADD $4, R8, R8
	B loop_lhs_4x8

done_4x8:
	RET

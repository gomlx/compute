// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

#include "textflag.h"

// FMLA Vd.4S, Vn.4S, Vm.S[lane]
#define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// FMLA Vd.2D, Vn.2D, Vm.D[lane]
#define FMLA_2D_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4FC01000 | (((lane)&1)<<11) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// SHLL Vd.4S, Vn.4H, #16 (widens lower 4 BF16 lanes to 4 FP32 lanes)
#define USHLL_4S_4H_16(Vd, Vn) \
	WORD $(0x2E613800 | ((Vn)<<5) | (Vd))

// SHLL2 Vd.4S, Vn.8H, #16 (widens upper 4 BF16 lanes to 4 FP32 lanes)
#define USHLL2_4S_8H_16(Vd, Vn) \
	WORD $(0x6E613800 | ((Vn)<<5) | (Vd))

// ============================================================================
// 1. neonLargeKernelM1Float32Asm: M=1 GEMV for Float32 (kernelCols = 32, 16, 8)
// ============================================================================
// func neonLargeKernelM1Float32Asm(
//     lhs, packedRHS, out []float32,
//     contractingLen, rhsActiveCols, kernelCols int,
//     accumulate bool)
//
// Reads raw contiguous 1-row lhs directly (zero LHS packing) and streams each
// kernelCols-wide packedRHS strip linearly ([ceil(N/Nr), K, Nr] float32), supporting
// both NEON-packed (Nr=16, 8) and SME-packed (Nr=32, 16) weight panels without SME lock.
TEXT ·neonLargeKernelM1Float32Asm(SB), NOSPLIT, $0-97
	MOVD lhs_base+0(FP), R0          // R0 = lhsBasePtr
	MOVD packedRHS_base+24(FP), R1   // R1 = rhsBasePtr
	MOVD out_base+48(FP), R2         // R2 = outPtr
	MOVD contractingLen+72(FP), R4   // R4 = K
	MOVD rhsActiveCols+80(FP), R6    // R6 = rhsRemCols
	MOVD kernelCols+88(FP), R5       // R5 = Nr (32, 16, or 8)
	MOVBU accumulate+96(FP), R7      // R7 = accumulate

	CMP $32, R5
	BEQ f32_m1_nr32
	CMP $16, R5
	BEQ f32_m1_nr16

// --- Nr = 8 (NEON 8x8 packed RHS: 32 bytes per K-step) ---
f32_m1_nr8:
	LSL $5, R4, R8                   // R8 = stripBytes = K * 32
loop_f32_nr8:
	CMP $0, R6
	BLE done_f32_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $2, R4, R13
	CBZ R13, k_rem_f32_nr8
k_quad_f32_nr8:
	VLD1.P 16(R11), [V16.S4]
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 1)
	FMLA_4S_LANE(3, 23, 16, 1)
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(4, 24, 16, 2)
	FMLA_4S_LANE(5, 25, 16, 2)
	FMLA_4S_LANE(6, 26, 16, 3)
	FMLA_4S_LANE(7, 27, 16, 3)
	SUB $1, R13, R13
	CBNZ R13, k_quad_f32_nr8
	VFADD V2.S4, V0.S4, V0.S4
	VFADD V3.S4, V1.S4, V1.S4
	VFADD V6.S4, V4.S4, V4.S4
	VFADD V7.S4, V5.S4, V5.S4
	VFADD V4.S4, V0.S4, V0.S4
	VFADD V5.S4, V1.S4, V1.S4
k_rem_f32_nr8:
	AND $3, R4, R13
	CBZ R13, store_f32_nr8
k_single_f32_nr8:
	MOVWU.P 4(R11), R14
	FMOVS R14, F16
	VLD1.P 32(R12), [V20.S4, V21.S4]
	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	SUB $1, R13, R13
	CBNZ R13, k_single_f32_nr8
store_f32_nr8:
	CBNZ R7, accum_f32_nr8
	VST1.P [V0.S4, V1.S4], 32(R2)
	B next_f32_nr8
accum_f32_nr8:
	VLD1 (R2), [V16.S4, V17.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VST1.P [V0.S4, V1.S4], 32(R2)
next_f32_nr8:
	ADD R8, R1, R1
	SUB $8, R6, R6
	B loop_f32_nr8

// --- Nr = 16 (NEON 4x16 or SME 16x16 packed RHS: 64 bytes per K-step) ---
f32_m1_nr16:
	LSL $6, R4, R8                   // R8 = stripBytes = K * 64
loop_f32_nr16:
	CMP $0, R6
	BLE done_f32_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $2, R4, R13
	CBZ R13, k_rem_f32_nr16
k_quad_f32_nr16:
	VLD1.P 16(R11), [V16.S4]
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(4, 24, 16, 1)
	FMLA_4S_LANE(5, 25, 16, 1)
	FMLA_4S_LANE(6, 26, 16, 1)
	FMLA_4S_LANE(7, 27, 16, 1)
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	FMLA_4S_LANE(0, 20, 16, 2)
	FMLA_4S_LANE(1, 21, 16, 2)
	FMLA_4S_LANE(2, 22, 16, 2)
	FMLA_4S_LANE(3, 23, 16, 2)
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(4, 24, 16, 3)
	FMLA_4S_LANE(5, 25, 16, 3)
	FMLA_4S_LANE(6, 26, 16, 3)
	FMLA_4S_LANE(7, 27, 16, 3)
	SUB $1, R13, R13
	CBNZ R13, k_quad_f32_nr16
	VFADD V4.S4, V0.S4, V0.S4
	VFADD V5.S4, V1.S4, V1.S4
	VFADD V6.S4, V2.S4, V2.S4
	VFADD V7.S4, V3.S4, V3.S4
k_rem_f32_nr16:
	AND $3, R4, R13
	CBZ R13, store_f32_nr16
k_single_f32_nr16:
	MOVWU.P 4(R11), R14
	FMOVS R14, F16
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	SUB $1, R13, R13
	CBNZ R13, k_single_f32_nr16
store_f32_nr16:
	CBNZ R7, accum_f32_nr16
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
	B next_f32_nr16
accum_f32_nr16:
	VLD1 (R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VFADD V18.S4, V2.S4, V2.S4
	VFADD V19.S4, V3.S4, V3.S4
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
next_f32_nr16:
	ADD R8, R1, R1
	SUB $16, R6, R6
	B loop_f32_nr16

// --- Nr = 32 (SME 32x32 packed RHS: 128 bytes per K-step, 8 accumulators V0..V7) ---
f32_m1_nr32:
	LSL $7, R4, R8                   // R8 = stripBytes = K * 128
loop_f32_nr32:
	CMP $0, R6
	BLE done_f32_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $2, R4, R13
	CBZ R13, k_rem_f32_nr32
k_quad_f32_nr32:
	VLD1.P 16(R11), [V16.S4]
	// K + 0
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	FMLA_4S_LANE(4, 24, 16, 0)
	FMLA_4S_LANE(5, 25, 16, 0)
	FMLA_4S_LANE(6, 26, 16, 0)
	FMLA_4S_LANE(7, 27, 16, 0)
	// K + 1
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(0, 20, 16, 1)
	FMLA_4S_LANE(1, 21, 16, 1)
	FMLA_4S_LANE(2, 22, 16, 1)
	FMLA_4S_LANE(3, 23, 16, 1)
	FMLA_4S_LANE(4, 24, 16, 1)
	FMLA_4S_LANE(5, 25, 16, 1)
	FMLA_4S_LANE(6, 26, 16, 1)
	FMLA_4S_LANE(7, 27, 16, 1)
	// K + 2
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(0, 20, 16, 2)
	FMLA_4S_LANE(1, 21, 16, 2)
	FMLA_4S_LANE(2, 22, 16, 2)
	FMLA_4S_LANE(3, 23, 16, 2)
	FMLA_4S_LANE(4, 24, 16, 2)
	FMLA_4S_LANE(5, 25, 16, 2)
	FMLA_4S_LANE(6, 26, 16, 2)
	FMLA_4S_LANE(7, 27, 16, 2)
	// K + 3
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(0, 20, 16, 3)
	FMLA_4S_LANE(1, 21, 16, 3)
	FMLA_4S_LANE(2, 22, 16, 3)
	FMLA_4S_LANE(3, 23, 16, 3)
	FMLA_4S_LANE(4, 24, 16, 3)
	FMLA_4S_LANE(5, 25, 16, 3)
	FMLA_4S_LANE(6, 26, 16, 3)
	FMLA_4S_LANE(7, 27, 16, 3)
	SUB $1, R13, R13
	CBNZ R13, k_quad_f32_nr32
k_rem_f32_nr32:
	AND $3, R4, R13
	CBZ R13, store_f32_nr32
k_single_f32_nr32:
	MOVWU.P 4(R11), R14
	FMOVS R14, F16
	VLD1.P 64(R12), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R12), [V24.S4, V25.S4, V26.S4, V27.S4]
	FMLA_4S_LANE(0, 20, 16, 0)
	FMLA_4S_LANE(1, 21, 16, 0)
	FMLA_4S_LANE(2, 22, 16, 0)
	FMLA_4S_LANE(3, 23, 16, 0)
	FMLA_4S_LANE(4, 24, 16, 0)
	FMLA_4S_LANE(5, 25, 16, 0)
	FMLA_4S_LANE(6, 26, 16, 0)
	FMLA_4S_LANE(7, 27, 16, 0)
	SUB $1, R13, R13
	CBNZ R13, k_single_f32_nr32
store_f32_nr32:
	CBNZ R7, accum_f32_nr32
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R2)
	B next_f32_nr32
accum_f32_nr32:
	VLD1 (R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VFADD V18.S4, V2.S4, V2.S4
	VFADD V19.S4, V3.S4, V3.S4
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
	VLD1 (R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V4.S4, V4.S4
	VFADD V17.S4, V5.S4, V5.S4
	VFADD V18.S4, V6.S4, V6.S4
	VFADD V19.S4, V7.S4, V7.S4
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R2)
next_f32_nr32:
	ADD R8, R1, R1
	SUB $32, R6, R6
	B loop_f32_nr32

done_f32_m1:
	RET

// ============================================================================
// 2. neonLargeKernelM1BFloat16Asm: M=1 GEMV for BFloat16 -> Float32 (Nr = 32, 16, 8)
// ============================================================================
// func neonLargeKernelM1BFloat16Asm(
//     lhs, packedRHS []bfloat16.BFloat16,
//     out []float32,
//     contractingLen, rhsActiveCols, kernelCols int,
//     accumulate bool)
TEXT ·neonLargeKernelM1BFloat16Asm(SB), NOSPLIT, $0-97
	MOVD lhs_base+0(FP), R0          // R0 = lhsBasePtr
	MOVD packedRHS_base+24(FP), R1   // R1 = rhsBasePtr
	MOVD out_base+48(FP), R2         // R2 = outPtr
	MOVD contractingLen+72(FP), R4   // R4 = K
	MOVD rhsActiveCols+80(FP), R6    // R6 = rhsRemCols
	MOVD kernelCols+88(FP), R5       // R5 = Nr (32, 16, or 8)
	MOVBU accumulate+96(FP), R7      // R7 = accumulate

	CMP $32, R5
	BEQ bf16_m1_nr32
	CMP $16, R5
	BEQ bf16_m1_nr16

// --- Nr = 8 (NEON 8x8 packed RHS: 16 bytes per K-step) ---
bf16_m1_nr8:
	LSL $4, R4, R8                   // R8 = stripBytes = K * 16
loop_bf16_nr8:
	CMP $0, R6
	BLE done_bf16_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $2, R4, R13
	CBZ R13, k_rem_bf16_nr8
k_quad_bf16_nr8:
	MOVD.P 8(R11), R14
	FMOVD R14, F20
	USHLL_4S_4H_16(16, 20)
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	FMLA_4S_LANE(0, 24, 16, 0)
	FMLA_4S_LANE(1, 25, 16, 0)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	FMLA_4S_LANE(2, 26, 16, 1)
	FMLA_4S_LANE(3, 27, 16, 1)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	FMLA_4S_LANE(4, 28, 16, 2)
	FMLA_4S_LANE(5, 29, 16, 2)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(6, 30, 16, 3)
	FMLA_4S_LANE(7, 31, 16, 3)
	SUB $1, R13, R13
	CBNZ R13, k_quad_bf16_nr8
	VFADD V2.S4, V0.S4, V0.S4
	VFADD V3.S4, V1.S4, V1.S4
	VFADD V6.S4, V4.S4, V4.S4
	VFADD V7.S4, V5.S4, V5.S4
	VFADD V4.S4, V0.S4, V0.S4
	VFADD V5.S4, V1.S4, V1.S4
k_rem_bf16_nr8:
	AND $3, R4, R13
	CBZ R13, store_bf16_nr8
k_single_bf16_nr8:
	MOVHU.P 2(R11), R14
	LSL $16, R14, R14
	FMOVS R14, F16
	VLD1.P 16(R12), [V20.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	FMLA_4S_LANE(0, 24, 16, 0)
	FMLA_4S_LANE(1, 25, 16, 0)
	SUB $1, R13, R13
	CBNZ R13, k_single_bf16_nr8
store_bf16_nr8:
	CBNZ R7, accum_bf16_nr8
	VST1.P [V0.S4, V1.S4], 32(R2)
	B next_bf16_nr8
accum_bf16_nr8:
	VLD1 (R2), [V16.S4, V17.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VST1.P [V0.S4, V1.S4], 32(R2)
next_bf16_nr8:
	ADD R8, R1, R1
	SUB $8, R6, R6
	B loop_bf16_nr8

// --- Nr = 16 (NEON 4x16 or SME 16x16 packed RHS: 32 bytes per K-step) ---
bf16_m1_nr16:
	LSL $5, R4, R8                   // R8 = stripBytes = K * 32
loop_bf16_nr16:
	CMP $0, R6
	BLE done_bf16_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $2, R4, R13
	CBZ R13, k_rem_bf16_nr16
k_quad_bf16_nr16:
	MOVD.P 8(R11), R14
	FMOVD R14, F20
	USHLL_4S_4H_16(16, 20)
	// K + 0 and K + 1
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	FMLA_4S_LANE(0, 24, 16, 0)
	FMLA_4S_LANE(1, 25, 16, 0)
	FMLA_4S_LANE(2, 26, 16, 0)
	FMLA_4S_LANE(3, 27, 16, 0)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(4, 28, 16, 1)
	FMLA_4S_LANE(5, 29, 16, 1)
	FMLA_4S_LANE(6, 30, 16, 1)
	FMLA_4S_LANE(7, 31, 16, 1)
	// K + 2 and K + 3
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	FMLA_4S_LANE(0, 24, 16, 2)
	FMLA_4S_LANE(1, 25, 16, 2)
	FMLA_4S_LANE(2, 26, 16, 2)
	FMLA_4S_LANE(3, 27, 16, 2)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(4, 28, 16, 3)
	FMLA_4S_LANE(5, 29, 16, 3)
	FMLA_4S_LANE(6, 30, 16, 3)
	FMLA_4S_LANE(7, 31, 16, 3)
	SUB $1, R13, R13
	CBNZ R13, k_quad_bf16_nr16
	VFADD V4.S4, V0.S4, V0.S4
	VFADD V5.S4, V1.S4, V1.S4
	VFADD V6.S4, V2.S4, V2.S4
	VFADD V7.S4, V3.S4, V3.S4
k_rem_bf16_nr16:
	AND $3, R4, R13
	CBZ R13, store_bf16_nr16
k_single_bf16_nr16:
	MOVHU.P 2(R11), R14
	LSL $16, R14, R14
	FMOVS R14, F16
	VLD1.P 32(R12), [V20.H8, V21.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	FMLA_4S_LANE(0, 24, 16, 0)
	FMLA_4S_LANE(1, 25, 16, 0)
	FMLA_4S_LANE(2, 26, 16, 0)
	FMLA_4S_LANE(3, 27, 16, 0)
	SUB $1, R13, R13
	CBNZ R13, k_single_bf16_nr16
store_bf16_nr16:
	CBNZ R7, accum_bf16_nr16
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
	B next_bf16_nr16
accum_bf16_nr16:
	VLD1 (R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VFADD V18.S4, V2.S4, V2.S4
	VFADD V19.S4, V3.S4, V3.S4
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
next_bf16_nr16:
	ADD R8, R1, R1
	SUB $16, R6, R6
	B loop_bf16_nr16

// --- Nr = 32 (SME 32x32 packed RHS: 64 bytes per K-step, 8 accumulators V0..V7) ---
bf16_m1_nr32:
	LSL $6, R4, R8                   // R8 = stripBytes = K * 64
loop_bf16_nr32:
	CMP $0, R6
	BLE done_bf16_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $2, R4, R13
	CBZ R13, k_rem_bf16_nr32
k_quad_bf16_nr32:
	MOVD.P 8(R11), R14
	FMOVD R14, F20
	USHLL_4S_4H_16(16, 20)
	// K + 0
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(0, 24, 16, 0)
	FMLA_4S_LANE(1, 25, 16, 0)
	FMLA_4S_LANE(2, 26, 16, 0)
	FMLA_4S_LANE(3, 27, 16, 0)
	FMLA_4S_LANE(4, 28, 16, 0)
	FMLA_4S_LANE(5, 29, 16, 0)
	FMLA_4S_LANE(6, 30, 16, 0)
	FMLA_4S_LANE(7, 31, 16, 0)
	// K + 1
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(0, 24, 16, 1)
	FMLA_4S_LANE(1, 25, 16, 1)
	FMLA_4S_LANE(2, 26, 16, 1)
	FMLA_4S_LANE(3, 27, 16, 1)
	FMLA_4S_LANE(4, 28, 16, 1)
	FMLA_4S_LANE(5, 29, 16, 1)
	FMLA_4S_LANE(6, 30, 16, 1)
	FMLA_4S_LANE(7, 31, 16, 1)
	// K + 2
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(0, 24, 16, 2)
	FMLA_4S_LANE(1, 25, 16, 2)
	FMLA_4S_LANE(2, 26, 16, 2)
	FMLA_4S_LANE(3, 27, 16, 2)
	FMLA_4S_LANE(4, 28, 16, 2)
	FMLA_4S_LANE(5, 29, 16, 2)
	FMLA_4S_LANE(6, 30, 16, 2)
	FMLA_4S_LANE(7, 31, 16, 2)
	// K + 3
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(0, 24, 16, 3)
	FMLA_4S_LANE(1, 25, 16, 3)
	FMLA_4S_LANE(2, 26, 16, 3)
	FMLA_4S_LANE(3, 27, 16, 3)
	FMLA_4S_LANE(4, 28, 16, 3)
	FMLA_4S_LANE(5, 29, 16, 3)
	FMLA_4S_LANE(6, 30, 16, 3)
	FMLA_4S_LANE(7, 31, 16, 3)
	SUB $1, R13, R13
	CBNZ R13, k_quad_bf16_nr32
k_rem_bf16_nr32:
	AND $3, R4, R13
	CBZ R13, store_bf16_nr32
k_single_bf16_nr32:
	MOVHU.P 2(R11), R14
	LSL $16, R14, R14
	FMOVS R14, F16
	VLD1.P 64(R12), [V20.H8, V21.H8, V22.H8, V23.H8]
	USHLL_4S_4H_16(24, 20)
	USHLL2_4S_8H_16(25, 20)
	USHLL_4S_4H_16(26, 21)
	USHLL2_4S_8H_16(27, 21)
	USHLL_4S_4H_16(28, 22)
	USHLL2_4S_8H_16(29, 22)
	USHLL_4S_4H_16(30, 23)
	USHLL2_4S_8H_16(31, 23)
	FMLA_4S_LANE(0, 24, 16, 0)
	FMLA_4S_LANE(1, 25, 16, 0)
	FMLA_4S_LANE(2, 26, 16, 0)
	FMLA_4S_LANE(3, 27, 16, 0)
	FMLA_4S_LANE(4, 28, 16, 0)
	FMLA_4S_LANE(5, 29, 16, 0)
	FMLA_4S_LANE(6, 30, 16, 0)
	FMLA_4S_LANE(7, 31, 16, 0)
	SUB $1, R13, R13
	CBNZ R13, k_single_bf16_nr32
store_bf16_nr32:
	CBNZ R7, accum_bf16_nr32
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R2)
	B next_bf16_nr32
accum_bf16_nr32:
	VLD1 (R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V0.S4, V0.S4
	VFADD V17.S4, V1.S4, V1.S4
	VFADD V18.S4, V2.S4, V2.S4
	VFADD V19.S4, V3.S4, V3.S4
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R2)
	VLD1 (R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFADD V16.S4, V4.S4, V4.S4
	VFADD V17.S4, V5.S4, V5.S4
	VFADD V18.S4, V6.S4, V6.S4
	VFADD V19.S4, V7.S4, V7.S4
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R2)
next_bf16_nr32:
	ADD R8, R1, R1
	SUB $32, R6, R6
	B loop_bf16_nr32

done_bf16_m1:
	RET

// ============================================================================
// 3. neonLargeKernelM1Float64Asm: M=1 GEMV for Float64 (kernelCols = 16, 8, 4)
// ============================================================================
// func neonLargeKernelM1Float64Asm(
//     lhs, packedRHS, out []float64,
//     contractingLen, rhsActiveCols, kernelCols int,
//     accumulate bool)
TEXT ·neonLargeKernelM1Float64Asm(SB), NOSPLIT, $0-97
	MOVD lhs_base+0(FP), R0          // R0 = lhsBasePtr
	MOVD packedRHS_base+24(FP), R1   // R1 = rhsBasePtr
	MOVD out_base+48(FP), R2         // R2 = outPtr
	MOVD contractingLen+72(FP), R4   // R4 = K
	MOVD rhsActiveCols+80(FP), R6    // R6 = rhsRemCols
	MOVD kernelCols+88(FP), R5       // R5 = Nr (16, 8, or 4)
	MOVBU accumulate+96(FP), R7      // R7 = accumulate

	CMP $16, R5
	BEQ f64_m1_nr16
	CMP $8, R5
	BEQ f64_m1_nr8

// --- Nr = 4 (NEON 8x4 packed RHS: 32 bytes per K-step) ---
f64_m1_nr4:
	LSL $5, R4, R8                   // R8 = stripBytes = K * 32
loop_f64_nr4:
	CMP $0, R6
	BLE done_f64_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $2, R4, R13
	CBZ R13, k_rem_f64_nr4
k_quad_f64_nr4:
	VLD1.P 32(R11), [V16.D2, V17.D2]
	VLD1.P 64(R12), [V20.D2, V21.D2, V22.D2, V23.D2]
	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	FMLA_2D_LANE(2, 22, 16, 1)
	FMLA_2D_LANE(3, 23, 16, 1)
	VLD1.P 64(R12), [V24.D2, V25.D2, V26.D2, V27.D2]
	FMLA_2D_LANE(4, 24, 17, 0)
	FMLA_2D_LANE(5, 25, 17, 0)
	FMLA_2D_LANE(6, 26, 17, 1)
	FMLA_2D_LANE(7, 27, 17, 1)
	SUB $1, R13, R13
	CBNZ R13, k_quad_f64_nr4
	VFADD V2.D2, V0.D2, V0.D2
	VFADD V3.D2, V1.D2, V1.D2
	VFADD V6.D2, V4.D2, V4.D2
	VFADD V7.D2, V5.D2, V5.D2
	VFADD V4.D2, V0.D2, V0.D2
	VFADD V5.D2, V1.D2, V1.D2
k_rem_f64_nr4:
	AND $3, R4, R13
	CBZ R13, store_f64_nr4
k_single_f64_nr4:
	MOVD.P 8(R11), R14
	FMOVD R14, F16
	VLD1.P 32(R12), [V20.D2, V21.D2]
	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	SUB $1, R13, R13
	CBNZ R13, k_single_f64_nr4
store_f64_nr4:
	CBNZ R7, accum_f64_nr4
	VST1.P [V0.D2, V1.D2], 32(R2)
	B next_f64_nr4
accum_f64_nr4:
	VLD1 (R2), [V16.D2, V17.D2]
	VFADD V16.D2, V0.D2, V0.D2
	VFADD V17.D2, V1.D2, V1.D2
	VST1.P [V0.D2, V1.D2], 32(R2)
next_f64_nr4:
	ADD R8, R1, R1
	SUB $4, R6, R6
	B loop_f64_nr4

// --- Nr = 8 (NEON 4x8 or SME 8x8 packed RHS: 64 bytes per K-step) ---
f64_m1_nr8:
	LSL $6, R4, R8                   // R8 = stripBytes = K * 64
loop_f64_nr8:
	CMP $0, R6
	BLE done_f64_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $1, R4, R13
	CBZ R13, k_rem_f64_nr8
k_pair_f64_nr8:
	VLD1.P 16(R11), [V16.D2]
	VLD1.P 64(R12), [V20.D2, V21.D2, V22.D2, V23.D2]
	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	FMLA_2D_LANE(2, 22, 16, 0)
	FMLA_2D_LANE(3, 23, 16, 0)
	VLD1.P 64(R12), [V24.D2, V25.D2, V26.D2, V27.D2]
	FMLA_2D_LANE(4, 24, 16, 1)
	FMLA_2D_LANE(5, 25, 16, 1)
	FMLA_2D_LANE(6, 26, 16, 1)
	FMLA_2D_LANE(7, 27, 16, 1)
	SUB $1, R13, R13
	CBNZ R13, k_pair_f64_nr8
	VFADD V4.D2, V0.D2, V0.D2
	VFADD V5.D2, V1.D2, V1.D2
	VFADD V6.D2, V2.D2, V2.D2
	VFADD V7.D2, V3.D2, V3.D2
k_rem_f64_nr8:
	TBZ $0, R4, store_f64_nr8
	MOVD.P 8(R11), R14
	FMOVD R14, F16
	VLD1.P 64(R12), [V20.D2, V21.D2, V22.D2, V23.D2]
	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	FMLA_2D_LANE(2, 22, 16, 0)
	FMLA_2D_LANE(3, 23, 16, 0)
store_f64_nr8:
	CBNZ R7, accum_f64_nr8
	VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R2)
	B next_f64_nr8
accum_f64_nr8:
	VLD1 (R2), [V16.D2, V17.D2, V18.D2, V19.D2]
	VFADD V16.D2, V0.D2, V0.D2
	VFADD V17.D2, V1.D2, V1.D2
	VFADD V18.D2, V2.D2, V2.D2
	VFADD V19.D2, V3.D2, V3.D2
	VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R2)
next_f64_nr8:
	ADD R8, R1, R1
	SUB $8, R6, R6
	B loop_f64_nr8

// --- Nr = 16 (SME 16x16 packed RHS: 128 bytes per K-step, 8 accumulators V0..V7) ---
f64_m1_nr16:
	LSL $7, R4, R8                   // R8 = stripBytes = K * 128
loop_f64_nr16:
	CMP $0, R6
	BLE done_f64_m1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R0, R11
	MOVD R1, R12
	LSR $1, R4, R13
	CBZ R13, k_rem_f64_nr16
k_pair_f64_nr16:
	VLD1.P 16(R11), [V16.D2]
	// K + 0
	VLD1.P 64(R12), [V20.D2, V21.D2, V22.D2, V23.D2]
	VLD1.P 64(R12), [V24.D2, V25.D2, V26.D2, V27.D2]
	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	FMLA_2D_LANE(2, 22, 16, 0)
	FMLA_2D_LANE(3, 23, 16, 0)
	FMLA_2D_LANE(4, 24, 16, 0)
	FMLA_2D_LANE(5, 25, 16, 0)
	FMLA_2D_LANE(6, 26, 16, 0)
	FMLA_2D_LANE(7, 27, 16, 0)
	// K + 1
	VLD1.P 64(R12), [V20.D2, V21.D2, V22.D2, V23.D2]
	VLD1.P 64(R12), [V24.D2, V25.D2, V26.D2, V27.D2]
	FMLA_2D_LANE(0, 20, 16, 1)
	FMLA_2D_LANE(1, 21, 16, 1)
	FMLA_2D_LANE(2, 22, 16, 1)
	FMLA_2D_LANE(3, 23, 16, 1)
	FMLA_2D_LANE(4, 24, 16, 1)
	FMLA_2D_LANE(5, 25, 16, 1)
	FMLA_2D_LANE(6, 26, 16, 1)
	FMLA_2D_LANE(7, 27, 16, 1)
	SUB $1, R13, R13
	CBNZ R13, k_pair_f64_nr16
k_rem_f64_nr16:
	TBZ $0, R4, store_f64_nr16
	MOVD.P 8(R11), R14
	FMOVD R14, F16
	VLD1.P 64(R12), [V20.D2, V21.D2, V22.D2, V23.D2]
	VLD1.P 64(R12), [V24.D2, V25.D2, V26.D2, V27.D2]
	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	FMLA_2D_LANE(2, 22, 16, 0)
	FMLA_2D_LANE(3, 23, 16, 0)
	FMLA_2D_LANE(4, 24, 16, 0)
	FMLA_2D_LANE(5, 25, 16, 0)
	FMLA_2D_LANE(6, 26, 16, 0)
	FMLA_2D_LANE(7, 27, 16, 0)
store_f64_nr16:
	CBNZ R7, accum_f64_nr16
	VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R2)
	VST1.P [V4.D2, V5.D2, V6.D2, V7.D2], 64(R2)
	B next_f64_nr16
accum_f64_nr16:
	VLD1 (R2), [V16.D2, V17.D2, V18.D2, V19.D2]
	VFADD V16.D2, V0.D2, V0.D2
	VFADD V17.D2, V1.D2, V1.D2
	VFADD V18.D2, V2.D2, V2.D2
	VFADD V19.D2, V3.D2, V3.D2
	VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R2)
	VLD1 (R2), [V16.D2, V17.D2, V18.D2, V19.D2]
	VFADD V16.D2, V4.D2, V4.D2
	VFADD V17.D2, V5.D2, V5.D2
	VFADD V18.D2, V6.D2, V6.D2
	VFADD V19.D2, V7.D2, V7.D2
	VST1.P [V4.D2, V5.D2, V6.D2, V7.D2], 64(R2)
next_f64_nr16:
	ADD R8, R1, R1
	SUB $16, R6, R6
	B loop_f64_nr16

done_f64_m1:
	RET

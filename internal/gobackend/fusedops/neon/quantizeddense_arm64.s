// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// USDOT Vd.4S, Vn.16B, Vm.4B[lane] (FEAT_I8MM: unsigned Vn.16B × signed Vm.4B[lane] -> signed Vd.4S, Vm in 0..15)
#define USDOT_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F80F000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// SDOT Vd.4S, Vn.16B, Vm.4B[lane] (FEAT_DotProd: signed Vn.16B × signed Vm.4B[lane] -> signed Vd.4S, Vm in 0..15)
#define SDOT_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F80E000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// FMUL Vd.4S, Vn.4S, Vm.S[lane] (Vm in 0..15)
#define FMUL_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F809000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// FMLA Vd.4S, Vn.4S, Vm.S[lane] (Vm in 0..15)
#define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// FMLA Vd.4S, Vn.4S, Vm.4S (Vd.4S += Vn.4S * Vm.4S)
#define FMLA_4S(Vd, Vn, Vm) \
	WORD $(0x4E20CC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// FMUL Vd.4S, Vn.4S, Vm.4S (Vd.4S = Vn.4S * Vm.4S)
#define FMUL_4S(Vd, Vn, Vm) \
	WORD $(0x6E20DC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// ADD Vd.4S, Vn.4S, Vm.4S (Vd.4S = Vn.4S + Vm.4S)
#define ADD_4S(Vd, Vn, Vm) \
	WORD $(0x4EA08400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// SCVTF Vd.4S, Vn.4S (int32 -> float32)
#define SCVTF_4S(Vd, Vn) \
	WORD $(0x4E21D800 | ((Vn)<<5) | (Vd))

// SCVTF Sd, Sn (scalar int32 -> float32)
#define SCVTF_S(Sd, Sn) \
	WORD $(0x5E21D800 | ((Sn)<<5) | (Sd))

// FCVTZS Vd.4S, Vn.4S (float32 -> int32)
#define FCVTZS_4S(Vd, Vn) \
	WORD $(0x4EA1B800 | ((Vn)<<5) | (Vd))

// FRINTN Vd.4S, Vn.4S (round to nearest even)
#define FRINTN_4S(Vd, Vn) \
	WORD $(0x4E218800 | ((Vn)<<5) | (Vd))

// FABS Vd.4S, Vn.4S
#define FABS_4S(Vd, Vn) \
	WORD $(0x4EA0F800 | ((Vn)<<5) | (Vd))

// FMAXNM Vd.4S, Vn.4S, Vm.4S
#define FMAXNM_4S(Vd, Vn, Vm) \
	WORD $(0x4E20C400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// FMAXNMV Sd, Vn.4S
#define FMAXNMV_4S(Sd, Vn) \
	WORD $(0x6E30C800 | ((Vn)<<5) | (Sd))

// ADDV Sd, Vn.4S
#define ADDV_4S(Sd, Vn) \
	WORD $(0x4EB1B800 | ((Vn)<<5) | (Sd))

// SMAX Vd.4S, Vn.4S, Vm.4S
#define SMAX_4S(Vd, Vn, Vm) \
	WORD $(0x4EA06400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// SMIN Vd.4S, Vn.4S, Vm.4S
#define SMIN_4S(Vd, Vn, Vm) \
	WORD $(0x4EA06C00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// SQXTN Vd.4H, Vn.4S
#define SQXTN_4H(Vd, Vn) \
	WORD $(0x0E614800 | ((Vn)<<5) | (Vd))

// SQXTN2 Vd.8H, Vn.4S
#define SQXTN2_8H(Vd, Vn) \
	WORD $(0x4E614800 | ((Vn)<<5) | (Vd))

// SQXTN Vd.8B, Vn.8H
#define SQXTN_8B(Vd, Vn) \
	WORD $(0x0E214800 | ((Vn)<<5) | (Vd))

// SQXTN2 Vd.16B, Vn.8H
#define SQXTN2_16B(Vd, Vn) \
	WORD $(0x4E214800 | ((Vn)<<5) | (Vd))

// FCVT Hd, Sn (FP32 -> FP16)
#define FCVT_H_S(Hd, Sn) \
	WORD $(0x1E23C000 | ((Sn)<<5) | (Hd))

// FCVT Sd, Hn (FP16 -> FP32)
#define FCVT_S_H(Sd, Hn) \
	WORD $(0x1EE24000 | ((Hn)<<5) | (Sd))

// DUP Vd.4S, Vn.S[0]
#define DUP_4S_S0(Vd, Vn) \
	WORD $(0x4E040400 | ((Vn)<<5) | (Vd))

// MOVI Vd.4S, #127
#define MOVI_4S_127(Vd) \
	WORD $(0x4F0307E0 | (Vd))

// MVNI Vd.4S, #126 (all lanes = -127)
#define MVNI_4S_126(Vd) \
	WORD $(0x6F0307C0 | (Vd))

// MOVI Vd.16B, #0x80
#define MOVI_16B_80(Vd) \
	WORD $(0x4F04E400 | (Vd))

// One quad step (16 cols × 4 K-bytes) for 3 rows using USDOT (FEAT_I8MM)
#define QUAD_STEP_3x16_USDOT(lane) \
	VLD1.P 64(R19), [V28.B16, V29.B16, V30.B16, V31.B16]; \
	USDOT_4S_LANE(16, 28, 0, lane); \
	USDOT_4S_LANE(17, 29, 0, lane); \
	USDOT_4S_LANE(18, 30, 0, lane); \
	USDOT_4S_LANE(19, 31, 0, lane); \
	USDOT_4S_LANE(20, 28, 1, lane); \
	USDOT_4S_LANE(21, 29, 1, lane); \
	USDOT_4S_LANE(22, 30, 1, lane); \
	USDOT_4S_LANE(23, 31, 1, lane); \
	USDOT_4S_LANE(24, 28, 2, lane); \
	USDOT_4S_LANE(25, 29, 2, lane); \
	USDOT_4S_LANE(26, 30, 2, lane); \
	USDOT_4S_LANE(27, 31, 2, lane)

// One quad step (16 cols × 4 K-bytes) for 3 rows using SDOT (FEAT_DotProd, XORing u_w with 0x80 in V3)
#define QUAD_STEP_3x16_SDOT(lane) \
	VLD1.P 64(R19), [V28.B16, V29.B16, V30.B16, V31.B16]; \
	VEOR V3.B16, V28.B16, V28.B16; \
	VEOR V3.B16, V29.B16, V29.B16; \
	VEOR V3.B16, V30.B16, V30.B16; \
	VEOR V3.B16, V31.B16, V31.B16; \
	SDOT_4S_LANE(16, 28, 0, lane); \
	SDOT_4S_LANE(17, 29, 0, lane); \
	SDOT_4S_LANE(18, 30, 0, lane); \
	SDOT_4S_LANE(19, 31, 0, lane); \
	SDOT_4S_LANE(20, 28, 1, lane); \
	SDOT_4S_LANE(21, 29, 1, lane); \
	SDOT_4S_LANE(22, 30, 1, lane); \
	SDOT_4S_LANE(23, 31, 1, lane); \
	SDOT_4S_LANE(24, 28, 2, lane); \
	SDOT_4S_LANE(25, 29, 2, lane); \
	SDOT_4S_LANE(26, 30, 2, lane); \
	SDOT_4S_LANE(27, 31, 2, lane)

// ============================================================================
// 1. quantizedPanelKernel3x16NEONAsm: 3 Rows × 16 Cols (1 Strip) Microkernel
// ============================================================================
// func quantizedPanelKernel3x16NEONAsm(
//     qxPtr unsafe.Pointer,       // +0(FP)
//     qxRowBytes int,             // +8(FP)
//     dxPairPtr unsafe.Pointer,   // +16(FP)
//     dxRowBytes int,             // +24(FP)
//     stripPtr unsafe.Pointer,    // +32(FP)
//     biasPtr unsafe.Pointer,     // +40(FP)
//     outPtr unsafe.Pointer,      // +48(FP)
//     outRowBytes int,            // +56(FP)
//     numSubBlocks int,           // +64(FP)
//     quadsPerSubBlk int,         // +72(FP)
//     numMBlocks int,             // +80(FP)
//     useUSDOT int,               // +88(FP)
// )
TEXT ·quantizedPanelKernel3x16NEONAsm(SB), NOSPLIT, $0-96
	MOVD qxPtr+0(FP), R0
	MOVD qxRowBytes+8(FP), R1
	MOVD dxPairPtr+16(FP), R2
	MOVD dxRowBytes+24(FP), R3
	MOVD stripPtr+32(FP), R4
	MOVD biasPtr+40(FP), R5
	MOVD outPtr+48(FP), R6
	MOVD outRowBytes+56(FP), R7
	MOVD numSubBlocks+64(FP), R8
	MOVD quadsPerSubBlk+72(FP), R9
	MOVD numMBlocks+80(FP), R10
	MOVD useUSDOT+88(FP), R11

	CMP $0, R10
	BLE done_3x16

	// Prepare V3 = 0x8080... if using SDOT fallback
	CBNZ R11, mblock_loop_3x16
	MOVI_16B_80(3)

mblock_loop_3x16:
	// Set row pointers for the 3 rows in this M-block:
	// qx rows: R12, R13, R14
	MOVD R0, R12
	ADD R1, R12, R13
	ADD R1, R13, R14
	// dx rows: R15, R16, R17
	MOVD R2, R15
	ADD R3, R15, R16
	ADD R3, R16, R17
	// Strip pointer cursor: R19
	MOVD R4, R19
	// Sub-block counter: R20
	MOVD R8, R20

	// Initialize 12 float32 accumulators (V4..V15) from bias or zero
	CBZ R5, zero_acc_3x16
	VLD1 (R5), [V4.S4, V5.S4, V6.S4, V7.S4]
	VMOV V4.B16, V8.B16
	VMOV V5.B16, V9.B16
	VMOV V6.B16, V10.B16
	VMOV V7.B16, V11.B16
	VMOV V4.B16, V12.B16
	VMOV V5.B16, V13.B16
	VMOV V6.B16, V14.B16
	VMOV V7.B16, V15.B16
	JMP check_subblocks_3x16

zero_acc_3x16:
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

check_subblocks_3x16:
	CMP $0, R20
	BLE store_3x16

subblock_loop_3x16:
	// Zero 12 int32 sub-block accumulators (V16..V27)
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	VEOR V24.B16, V24.B16, V24.B16
	VEOR V25.B16, V25.B16, V25.B16
	VEOR V26.B16, V26.B16, V26.B16
	VEOR V27.B16, V27.B16, V27.B16

	CBZ R11, sdot_dispatch_3x16

	// --- FEAT_I8MM USDOT Fast Paths ---
	CMP $8, R9
	BEQ usdot_quads8_3x16
	CMP $4, R9
	BEQ usdot_quads4_3x16

	MOVD R9, R21
usdot_quad_generic_3x16:
	MOVWU (R12), R22
	FMOVS R22, F0
	ADD $4, R12
	MOVWU (R13), R22
	FMOVS R22, F1
	ADD $4, R13
	MOVWU (R14), R22
	FMOVS R22, F2
	ADD $4, R14
	QUAD_STEP_3x16_USDOT(0)
	SUBS $1, R21, R21
	BNE usdot_quad_generic_3x16
	JMP subblock_epilogue_usdot_3x16

usdot_quads8_3x16:
	// Quads 0..3 (16 bytes of Qx per row)
	VLD1.P 16(R12), [V0.B16]
	VLD1.P 16(R13), [V1.B16]
	VLD1.P 16(R14), [V2.B16]
	QUAD_STEP_3x16_USDOT(0)
	QUAD_STEP_3x16_USDOT(1)
	QUAD_STEP_3x16_USDOT(2)
	QUAD_STEP_3x16_USDOT(3)

usdot_quads4_3x16:
	// Quads 4..7 (or 0..3 when quadsPerSubBlk == 4)
	VLD1.P 16(R12), [V0.B16]
	VLD1.P 16(R13), [V1.B16]
	VLD1.P 16(R14), [V2.B16]
	QUAD_STEP_3x16_USDOT(0)
	QUAD_STEP_3x16_USDOT(1)
	QUAD_STEP_3x16_USDOT(2)
	QUAD_STEP_3x16_USDOT(3)

subblock_epilogue_usdot_3x16:
	// Load {dx, dx*sum_qx} pairs (8 bytes = .S2) for rows 0, 1, 2 into V0, V1, V2
	VLD1.P 8(R15), [V0.S2]
	VLD1.P 8(R16), [V1.S2]
	VLD1.P 8(R17), [V2.S2]

	// Load d'_w (16 floats = 64 bytes) into V28..V31
	VLD1.P 64(R19), [V28.S4, V29.S4, V30.S4, V31.S4]

	// Convert int32 dots (V16..V27) to float32, scale by dx (V0..2.S[0]), and FMLA with d'_w (V28..V31)
	SCVTF_4S(16, 16)
	SCVTF_4S(17, 17)
	SCVTF_4S(18, 18)
	SCVTF_4S(19, 19)
	FMUL_4S_LANE(16, 16, 0, 0)
	FMUL_4S_LANE(17, 17, 0, 0)
	FMUL_4S_LANE(18, 18, 0, 0)
	FMUL_4S_LANE(19, 19, 0, 0)
	FMLA_4S(4, 16, 28)
	FMLA_4S(5, 17, 29)
	FMLA_4S(6, 18, 30)
	FMLA_4S(7, 19, 31)

	SCVTF_4S(20, 20)
	SCVTF_4S(21, 21)
	SCVTF_4S(22, 22)
	SCVTF_4S(23, 23)
	FMUL_4S_LANE(20, 20, 1, 0)
	FMUL_4S_LANE(21, 21, 1, 0)
	FMUL_4S_LANE(22, 22, 1, 0)
	FMUL_4S_LANE(23, 23, 1, 0)
	FMLA_4S(8, 20, 28)
	FMLA_4S(9, 21, 29)
	FMLA_4S(10, 22, 30)
	FMLA_4S(11, 23, 31)

	SCVTF_4S(24, 24)
	SCVTF_4S(25, 25)
	SCVTF_4S(26, 26)
	SCVTF_4S(27, 27)
	FMUL_4S_LANE(24, 24, 2, 0)
	FMUL_4S_LANE(25, 25, 2, 0)
	FMUL_4S_LANE(26, 26, 2, 0)
	FMUL_4S_LANE(27, 27, 2, 0)
	FMLA_4S(12, 24, 28)
	FMLA_4S(13, 25, 29)
	FMLA_4S(14, 26, 30)
	FMLA_4S(15, 27, 31)

	// Load m'_w (16 floats = 64 bytes) into V28..V31 and FMLA with dx*sum_qx (V0..2.S[1])
	VLD1.P 64(R19), [V28.S4, V29.S4, V30.S4, V31.S4]
	FMLA_4S_LANE(4, 28, 0, 1)
	FMLA_4S_LANE(5, 29, 0, 1)
	FMLA_4S_LANE(6, 30, 0, 1)
	FMLA_4S_LANE(7, 31, 0, 1)
	FMLA_4S_LANE(8, 28, 1, 1)
	FMLA_4S_LANE(9, 29, 1, 1)
	FMLA_4S_LANE(10, 30, 1, 1)
	FMLA_4S_LANE(11, 31, 1, 1)
	FMLA_4S_LANE(12, 28, 2, 1)
	FMLA_4S_LANE(13, 29, 2, 1)
	FMLA_4S_LANE(14, 30, 2, 1)
	FMLA_4S_LANE(15, 31, 2, 1)

	SUBS $1, R20, R20
	BNE subblock_loop_3x16
	JMP store_3x16

	// --- FEAT_DotProd SDOT Fallback Path (u_w XOR 0x80) ---
sdot_dispatch_3x16:
	CMP $8, R9
	BEQ sdot_quads8_3x16
	CMP $4, R9
	BEQ sdot_quads4_3x16

	MOVD R9, R21
sdot_quad_generic_3x16:
	MOVWU (R12), R22
	FMOVS R22, F0
	ADD $4, R12
	MOVWU (R13), R22
	FMOVS R22, F1
	ADD $4, R13
	MOVWU (R14), R22
	FMOVS R22, F2
	ADD $4, R14
	QUAD_STEP_3x16_SDOT(0)
	SUBS $1, R21, R21
	BNE sdot_quad_generic_3x16
	JMP subblock_epilogue_sdot_3x16

sdot_quads8_3x16:
	VLD1.P 16(R12), [V0.B16]
	VLD1.P 16(R13), [V1.B16]
	VLD1.P 16(R14), [V2.B16]
	QUAD_STEP_3x16_SDOT(0)
	QUAD_STEP_3x16_SDOT(1)
	QUAD_STEP_3x16_SDOT(2)
	QUAD_STEP_3x16_SDOT(3)

sdot_quads4_3x16:
	VLD1.P 16(R12), [V0.B16]
	VLD1.P 16(R13), [V1.B16]
	VLD1.P 16(R14), [V2.B16]
	QUAD_STEP_3x16_SDOT(0)
	QUAD_STEP_3x16_SDOT(1)
	QUAD_STEP_3x16_SDOT(2)
	QUAD_STEP_3x16_SDOT(3)

subblock_epilogue_sdot_3x16:
	VLD1.P 8(R15), [V0.S2]
	VLD1.P 8(R16), [V1.S2]
	VLD1.P 8(R17), [V2.S2]

	// Load d'_w (16 floats = 64 bytes) into V28..V31
	VLD1.P 64(R19), [V28.S4, V29.S4, V30.S4, V31.S4]

	SCVTF_4S(16, 16)
	SCVTF_4S(17, 17)
	SCVTF_4S(18, 18)
	SCVTF_4S(19, 19)
	FMUL_4S_LANE(16, 16, 0, 0)
	FMUL_4S_LANE(17, 17, 0, 0)
	FMUL_4S_LANE(18, 18, 0, 0)
	FMUL_4S_LANE(19, 19, 0, 0)
	FMLA_4S(4, 16, 28)
	FMLA_4S(5, 17, 29)
	FMLA_4S(6, 18, 30)
	FMLA_4S(7, 19, 31)

	SCVTF_4S(20, 20)
	SCVTF_4S(21, 21)
	SCVTF_4S(22, 22)
	SCVTF_4S(23, 23)
	FMUL_4S_LANE(20, 20, 1, 0)
	FMUL_4S_LANE(21, 21, 1, 0)
	FMUL_4S_LANE(22, 22, 1, 0)
	FMUL_4S_LANE(23, 23, 1, 0)
	FMLA_4S(8, 20, 28)
	FMLA_4S(9, 21, 29)
	FMLA_4S(10, 22, 30)
	FMLA_4S(11, 23, 31)

	SCVTF_4S(24, 24)
	SCVTF_4S(25, 25)
	SCVTF_4S(26, 26)
	SCVTF_4S(27, 27)
	FMUL_4S_LANE(24, 24, 2, 0)
	FMUL_4S_LANE(25, 25, 2, 0)
	FMUL_4S_LANE(26, 26, 2, 0)
	FMUL_4S_LANE(27, 27, 2, 0)
	FMLA_4S(12, 24, 28)
	FMLA_4S(13, 25, 29)
	FMLA_4S(14, 26, 30)
	FMLA_4S(15, 27, 31)

	// Load m'_w into V16..V19 and adjust m'_w += 128.0 * d'_w (since u_w = s_w + 128)
	VLD1.P 64(R19), [V16.S4, V17.S4, V18.S4, V19.S4]
	MOVD $0x43000000, R22
	FMOVS R22, F20
	DUP_4S_S0(20, 20)
	FMLA_4S(16, 28, 20)
	FMLA_4S(17, 29, 20)
	FMLA_4S(18, 30, 20)
	FMLA_4S(19, 31, 20)

	FMLA_4S_LANE(4, 16, 0, 1)
	FMLA_4S_LANE(5, 17, 0, 1)
	FMLA_4S_LANE(6, 18, 0, 1)
	FMLA_4S_LANE(7, 19, 0, 1)
	FMLA_4S_LANE(8, 16, 1, 1)
	FMLA_4S_LANE(9, 17, 1, 1)
	FMLA_4S_LANE(10, 18, 1, 1)
	FMLA_4S_LANE(11, 19, 1, 1)
	FMLA_4S_LANE(12, 16, 2, 1)
	FMLA_4S_LANE(13, 17, 2, 1)
	FMLA_4S_LANE(14, 18, 2, 1)
	FMLA_4S_LANE(15, 19, 2, 1)

	SUBS $1, R20, R20
	BNE subblock_loop_3x16

store_3x16:
	MOVD R6, R21
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R21)
	ADD R7, R21, R21
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R21)
	ADD R7, R21, R21
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R21)

	// Advance base pointers by 3 rows for next M-block
	ADD R1, R0, R0
	ADD R1<<1, R0, R0
	ADD R3, R2, R2
	ADD R3<<1, R2, R2
	ADD R7, R21, R6
	SUBS $1, R10, R10
	BNE mblock_loop_3x16

done_3x16:
	RET

// One quad step (16 cols × 4 K-bytes) for 1 row using USDOT (FEAT_I8MM)
#define QUAD_STEP_1x16_USDOT(lane) \
	VLD1.P 64(R2), [V28.B16, V29.B16, V30.B16, V31.B16]; \
	USDOT_4S_LANE(16, 28, 0, lane); \
	USDOT_4S_LANE(17, 29, 0, lane); \
	USDOT_4S_LANE(18, 30, 0, lane); \
	USDOT_4S_LANE(19, 31, 0, lane)

// One quad step (16 cols × 4 K-bytes) for 1 row using SDOT (FEAT_DotProd)
#define QUAD_STEP_1x16_SDOT(lane) \
	VLD1.P 64(R2), [V28.B16, V29.B16, V30.B16, V31.B16]; \
	VEOR V3.B16, V28.B16, V28.B16; \
	VEOR V3.B16, V29.B16, V29.B16; \
	VEOR V3.B16, V30.B16, V30.B16; \
	VEOR V3.B16, V31.B16, V31.B16; \
	SDOT_4S_LANE(16, 28, 0, lane); \
	SDOT_4S_LANE(17, 29, 0, lane); \
	SDOT_4S_LANE(18, 30, 0, lane); \
	SDOT_4S_LANE(19, 31, 0, lane)

// ============================================================================
// 2. quantizedPanelKernel1x16NEONAsm: 1 Row × 16 Cols Remainder Microkernel
// ============================================================================
// func quantizedPanelKernel1x16NEONAsm(
//     qxPtr unsafe.Pointer,       // +0(FP)
//     dxPairPtr unsafe.Pointer,   // +8(FP)
//     stripPtr unsafe.Pointer,    // +16(FP)
//     biasPtr unsafe.Pointer,     // +24(FP)
//     outPtr unsafe.Pointer,      // +32(FP)
//     numSubBlocks int,           // +40(FP)
//     quadsPerSubBlk int,         // +48(FP)
//     useUSDOT int,               // +56(FP)
// )
TEXT ·quantizedPanelKernel1x16NEONAsm(SB), NOSPLIT, $0-64
	MOVD qxPtr+0(FP), R0
	MOVD dxPairPtr+8(FP), R1
	MOVD stripPtr+16(FP), R2
	MOVD biasPtr+24(FP), R3
	MOVD outPtr+32(FP), R4
	MOVD numSubBlocks+40(FP), R5
	MOVD quadsPerSubBlk+48(FP), R6
	MOVD useUSDOT+56(FP), R7

	CBNZ R7, init_acc_1x16
	MOVI_16B_80(3)

init_acc_1x16:
	CBZ R3, zero_acc_1x16
	VLD1 (R3), [V4.S4, V5.S4, V6.S4, V7.S4]
	JMP check_subblocks_1x16

zero_acc_1x16:
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16

check_subblocks_1x16:
	CMP $0, R5
	BLE store_1x16

subblock_loop_1x16:
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

	CBZ R7, sdot_dispatch_1x16

	CMP $8, R6
	BEQ usdot_quads8_1x16
	CMP $4, R6
	BEQ usdot_quads4_1x16

	MOVD R6, R8
usdot_quad_generic_1x16:
	MOVWU (R0), R9
	FMOVS R9, F0
	ADD $4, R0
	QUAD_STEP_1x16_USDOT(0)
	SUBS $1, R8, R8
	BNE usdot_quad_generic_1x16
	JMP subblock_epilogue_usdot_1x16

usdot_quads8_1x16:
	VLD1.P 16(R0), [V0.B16]
	QUAD_STEP_1x16_USDOT(0)
	QUAD_STEP_1x16_USDOT(1)
	QUAD_STEP_1x16_USDOT(2)
	QUAD_STEP_1x16_USDOT(3)

usdot_quads4_1x16:
	VLD1.P 16(R0), [V0.B16]
	QUAD_STEP_1x16_USDOT(0)
	QUAD_STEP_1x16_USDOT(1)
	QUAD_STEP_1x16_USDOT(2)
	QUAD_STEP_1x16_USDOT(3)

subblock_epilogue_usdot_1x16:
	VLD1.P 8(R1), [V0.S2]
	VLD1.P 64(R2), [V28.S4, V29.S4, V30.S4, V31.S4]
	SCVTF_4S(16, 16)
	SCVTF_4S(17, 17)
	SCVTF_4S(18, 18)
	SCVTF_4S(19, 19)
	FMUL_4S_LANE(16, 16, 0, 0)
	FMUL_4S_LANE(17, 17, 0, 0)
	FMUL_4S_LANE(18, 18, 0, 0)
	FMUL_4S_LANE(19, 19, 0, 0)
	FMLA_4S(4, 16, 28)
	FMLA_4S(5, 17, 29)
	FMLA_4S(6, 18, 30)
	FMLA_4S(7, 19, 31)

	VLD1.P 64(R2), [V28.S4, V29.S4, V30.S4, V31.S4]
	FMLA_4S_LANE(4, 28, 0, 1)
	FMLA_4S_LANE(5, 29, 0, 1)
	FMLA_4S_LANE(6, 30, 0, 1)
	FMLA_4S_LANE(7, 31, 0, 1)

	SUBS $1, R5, R5
	BNE subblock_loop_1x16
	JMP store_1x16

sdot_dispatch_1x16:
	CMP $8, R6
	BEQ sdot_quads8_1x16
	CMP $4, R6
	BEQ sdot_quads4_1x16

	MOVD R6, R8
sdot_quad_generic_1x16:
	MOVWU (R0), R9
	FMOVS R9, F0
	ADD $4, R0
	QUAD_STEP_1x16_SDOT(0)
	SUBS $1, R8, R8
	BNE sdot_quad_generic_1x16
	JMP subblock_epilogue_sdot_1x16

sdot_quads8_1x16:
	VLD1.P 16(R0), [V0.B16]
	QUAD_STEP_1x16_SDOT(0)
	QUAD_STEP_1x16_SDOT(1)
	QUAD_STEP_1x16_SDOT(2)
	QUAD_STEP_1x16_SDOT(3)

sdot_quads4_1x16:
	VLD1.P 16(R0), [V0.B16]
	QUAD_STEP_1x16_SDOT(0)
	QUAD_STEP_1x16_SDOT(1)
	QUAD_STEP_1x16_SDOT(2)
	QUAD_STEP_1x16_SDOT(3)

subblock_epilogue_sdot_1x16:
	VLD1.P 8(R1), [V0.S2]
	VLD1.P 64(R2), [V28.S4, V29.S4, V30.S4, V31.S4]
	SCVTF_4S(16, 16)
	SCVTF_4S(17, 17)
	SCVTF_4S(18, 18)
	SCVTF_4S(19, 19)
	FMUL_4S_LANE(16, 16, 0, 0)
	FMUL_4S_LANE(17, 17, 0, 0)
	FMUL_4S_LANE(18, 18, 0, 0)
	FMUL_4S_LANE(19, 19, 0, 0)
	FMLA_4S(4, 16, 28)
	FMLA_4S(5, 17, 29)
	FMLA_4S(6, 18, 30)
	FMLA_4S(7, 19, 31)

	VLD1.P 64(R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	MOVD $0x43000000, R9
	FMOVS R9, F20
	DUP_4S_S0(20, 20)
	FMLA_4S(16, 28, 20)
	FMLA_4S(17, 29, 20)
	FMLA_4S(18, 30, 20)
	FMLA_4S(19, 31, 20)

	FMLA_4S_LANE(4, 16, 0, 1)
	FMLA_4S_LANE(5, 17, 0, 1)
	FMLA_4S_LANE(6, 18, 0, 1)
	FMLA_4S_LANE(7, 19, 0, 1)

	SUBS $1, R5, R5
	BNE subblock_loop_1x16

store_1x16:
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R4)
	RET

// ============================================================================
// 3. quantizeGGMLQ8_0RowsNEONAsm: Fast NEON + FP16 Q8_0 Activation Quantizer
// ============================================================================
// func quantizeGGMLQ8_0RowsNEONAsm(
//     xPtr unsafe.Pointer,   // +0(FP)
//     qxPtr unsafe.Pointer,  // +8(FP)
//     dxPtr unsafe.Pointer,  // +16(FP)
//     numBlocks int,         // +24(FP)
// )
TEXT ·quantizeGGMLQ8_0RowsNEONAsm(SB), NOSPLIT, $0-32
	MOVD xPtr+0(FP), R0
	MOVD qxPtr+8(FP), R1
	MOVD dxPtr+16(FP), R2
	MOVD numBlocks+24(FP), R3

	CMP $0, R3
	BLE q8_0_done

	// Hoist constants:
	// V26.4S = -127, V27.4S = 127
	MVNI_4S_126(26)
	MOVI_4S_127(27)
	// F28 = 1.0 (0x3F800000), F29 = 127.0 (0x42FE0000)
	MOVD $0x3F800000, R4
	FMOVS R4, F28
	MOVD $0x42FE0000, R4
	FMOVS R4, F29
	// V30, V31 = 0
	VEOR V30.B16, V30.B16, V30.B16
	VEOR V31.B16, V31.B16, V31.B16

q8_0_loop:
	// Load 32 floats (128 bytes) into V0..V7
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]

	// Compute max(|x|) across V0..V7
	FABS_4S(16, 0)
	FABS_4S(17, 1)
	FABS_4S(18, 2)
	FABS_4S(19, 3)
	FABS_4S(20, 4)
	FABS_4S(21, 5)
	FABS_4S(22, 6)
	FABS_4S(23, 7)
	FMAXNM_4S(16, 16, 17)
	FMAXNM_4S(18, 18, 19)
	FMAXNM_4S(20, 20, 21)
	FMAXNM_4S(22, 22, 23)
	FMAXNM_4S(16, 16, 18)
	FMAXNM_4S(20, 20, 22)
	FMAXNM_4S(16, 16, 20)
	FMAXNMV_4S(16, 16)

	FCMPS $(0.0), F16
	BEQ q8_0_zero

	// d = fp16_to_fp32(fp32_to_fp16(maxAbs / 127.0))
	FDIVS F29, F16, F17
	FCVT_H_S(17, 17)
	FCVT_S_H(17, 17)
	FCMPS $(0.0), F17
	BEQ q8_0_zero

	// invD = 1.0 / d
	FDIVS F17, F28, F18
	DUP_4S_S0(18, 18)

	// Quantize V0..V7: roundToEven(x * invD), convert to int32, clamp [-127, 127]
	FMUL_4S(0, 0, 18)
	FMUL_4S(1, 1, 18)
	FMUL_4S(2, 2, 18)
	FMUL_4S(3, 3, 18)
	FMUL_4S(4, 4, 18)
	FMUL_4S(5, 5, 18)
	FMUL_4S(6, 6, 18)
	FMUL_4S(7, 7, 18)

	FRINTN_4S(0, 0)
	FRINTN_4S(1, 1)
	FRINTN_4S(2, 2)
	FRINTN_4S(3, 3)
	FRINTN_4S(4, 4)
	FRINTN_4S(5, 5)
	FRINTN_4S(6, 6)
	FRINTN_4S(7, 7)

	FCVTZS_4S(0, 0)
	FCVTZS_4S(1, 1)
	FCVTZS_4S(2, 2)
	FCVTZS_4S(3, 3)
	FCVTZS_4S(4, 4)
	FCVTZS_4S(5, 5)
	FCVTZS_4S(6, 6)
	FCVTZS_4S(7, 7)

	SMAX_4S(0, 0, 26)
	SMAX_4S(1, 1, 26)
	SMAX_4S(2, 2, 26)
	SMAX_4S(3, 3, 26)
	SMAX_4S(4, 4, 26)
	SMAX_4S(5, 5, 26)
	SMAX_4S(6, 6, 26)
	SMAX_4S(7, 7, 26)

	SMIN_4S(0, 0, 27)
	SMIN_4S(1, 1, 27)
	SMIN_4S(2, 2, 27)
	SMIN_4S(3, 3, 27)
	SMIN_4S(4, 4, 27)
	SMIN_4S(5, 5, 27)
	SMIN_4S(6, 6, 27)
	SMIN_4S(7, 7, 27)

	// Sum all 32 int32s in V0..V7
	ADD_4S(19, 0, 1)
	ADD_4S(20, 2, 3)
	ADD_4S(21, 4, 5)
	ADD_4S(22, 6, 7)
	ADD_4S(19, 19, 20)
	ADD_4S(21, 21, 22)
	ADD_4S(19, 19, 21)
	ADDV_4S(19, 19)
	SCVTF_S(19, 19)
	FMULS F17, F19, F19

	// Store (d, d * sum_qx)
	FMOVS F17, (R2)
	FMOVS F19, 4(R2)
	ADD $8, R2

	// Narrow V0..V7 (int32) -> V0, V4 (int8) and store 32 bytes
	SQXTN_4H(0, 0)
	SQXTN2_8H(0, 1)
	SQXTN_4H(2, 2)
	SQXTN2_8H(2, 3)
	SQXTN_8B(0, 0)
	SQXTN2_16B(0, 2)

	SQXTN_4H(4, 4)
	SQXTN2_8H(4, 5)
	SQXTN_4H(6, 6)
	SQXTN2_8H(6, 7)
	SQXTN_8B(4, 4)
	SQXTN2_16B(4, 6)

	VST1.P [V0.B16], 16(R1)
	VST1.P [V4.B16], 16(R1)

	SUBS $1, R3, R3
	BNE q8_0_loop
	RET

q8_0_zero:
	VST1.P [V30.B16, V31.B16], 32(R1)
	MOVD ZR, (R2)
	ADD $8, R2
	SUBS $1, R3, R3
	BNE q8_0_loop

q8_0_done:
	RET

// ============================================================================
// 4. quantizeGGMLQ8_KRowsNEONAsm: Fast NEON Q8_K 256-Element Super-Block Quantizer
// ============================================================================
// func quantizeGGMLQ8_KRowsNEONAsm(
//     xPtr unsafe.Pointer,   // +0(FP)
//     qxPtr unsafe.Pointer,  // +8(FP)
//     dxPtr unsafe.Pointer,  // +16(FP)
//     numSuperBlocks int,    // +24(FP)
//     subBlockSize int,      // +32(FP)
// )
TEXT ·quantizeGGMLQ8_KRowsNEONAsm(SB), NOSPLIT, $0-40
	MOVD xPtr+0(FP), R0
	MOVD qxPtr+8(FP), R1
	MOVD dxPtr+16(FP), R2
	MOVD numSuperBlocks+24(FP), R3
	MOVD subBlockSize+32(FP), R4

	CMP $0, R3
	BLE q8_k_done

	// Hoist constants:
	MVNI_4S_126(26)
	MOVI_4S_127(27)
	MOVD $0x3C010204, R5
	FMOVS R5, F28
	MOVD $0x42FE0000, R5
	FMOVS R5, F29
	VEOR V30.B16, V30.B16, V30.B16
	VEOR V31.B16, V31.B16, V31.B16

q8_k_super_loop:
	// Pass 1: Max(|x|) across 256 floats (8 chunks of 32 floats)
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	MOVD R0, R5
	MOVD $8, R6

q8_k_max_loop:
	VLD1.P 64(R5), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R5), [V4.S4, V5.S4, V6.S4, V7.S4]
	FABS_4S(0, 0)
	FABS_4S(1, 1)
	FABS_4S(2, 2)
	FABS_4S(3, 3)
	FABS_4S(4, 4)
	FABS_4S(5, 5)
	FABS_4S(6, 6)
	FABS_4S(7, 7)
	FMAXNM_4S(0, 0, 1)
	FMAXNM_4S(2, 2, 3)
	FMAXNM_4S(4, 4, 5)
	FMAXNM_4S(6, 6, 7)
	FMAXNM_4S(0, 0, 2)
	FMAXNM_4S(4, 4, 6)
	FMAXNM_4S(16, 16, 0)
	FMAXNM_4S(17, 17, 4)
	SUBS $1, R6, R6
	BNE q8_k_max_loop

	FMAXNM_4S(16, 16, 17)
	FMAXNMV_4S(16, 16)

	FCMPS $(0.0), F16
	BEQ q8_k_zero

	// d = maxAbs * (1.0 / 127.0) (FP32), invD = 127.0 / maxAbs
	FMULS F28, F16, F17
	FDIVS F16, F29, F18
	DUP_4S_S0(18, 18)

	CMP $16, R4
	BEQ q8_k_sub16

	// subBlockSize == 32 (8 sub-blocks per super-block)
	MOVD $8, R6
q8_k_sub32_loop:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]

	FMUL_4S(0, 0, 18)
	FMUL_4S(1, 1, 18)
	FMUL_4S(2, 2, 18)
	FMUL_4S(3, 3, 18)
	FMUL_4S(4, 4, 18)
	FMUL_4S(5, 5, 18)
	FMUL_4S(6, 6, 18)
	FMUL_4S(7, 7, 18)

	FRINTN_4S(0, 0)
	FRINTN_4S(1, 1)
	FRINTN_4S(2, 2)
	FRINTN_4S(3, 3)
	FRINTN_4S(4, 4)
	FRINTN_4S(5, 5)
	FRINTN_4S(6, 6)
	FRINTN_4S(7, 7)

	FCVTZS_4S(0, 0)
	FCVTZS_4S(1, 1)
	FCVTZS_4S(2, 2)
	FCVTZS_4S(3, 3)
	FCVTZS_4S(4, 4)
	FCVTZS_4S(5, 5)
	FCVTZS_4S(6, 6)
	FCVTZS_4S(7, 7)

	SMAX_4S(0, 0, 26)
	SMAX_4S(1, 1, 26)
	SMAX_4S(2, 2, 26)
	SMAX_4S(3, 3, 26)
	SMAX_4S(4, 4, 26)
	SMAX_4S(5, 5, 26)
	SMAX_4S(6, 6, 26)
	SMAX_4S(7, 7, 26)

	SMIN_4S(0, 0, 27)
	SMIN_4S(1, 1, 27)
	SMIN_4S(2, 2, 27)
	SMIN_4S(3, 3, 27)
	SMIN_4S(4, 4, 27)
	SMIN_4S(5, 5, 27)
	SMIN_4S(6, 6, 27)
	SMIN_4S(7, 7, 27)

	ADD_4S(19, 0, 1)
	ADD_4S(20, 2, 3)
	ADD_4S(21, 4, 5)
	ADD_4S(22, 6, 7)
	ADD_4S(19, 19, 20)
	ADD_4S(21, 21, 22)
	ADD_4S(19, 19, 21)
	ADDV_4S(19, 19)
	SCVTF_S(19, 19)
	FMULS F17, F19, F19

	FMOVS F17, (R2)
	FMOVS F19, 4(R2)
	ADD $8, R2

	SQXTN_4H(0, 0)
	SQXTN2_8H(0, 1)
	SQXTN_4H(2, 2)
	SQXTN2_8H(2, 3)
	SQXTN_8B(0, 0)
	SQXTN2_16B(0, 2)

	SQXTN_4H(4, 4)
	SQXTN2_8H(4, 5)
	SQXTN_4H(6, 6)
	SQXTN2_8H(6, 7)
	SQXTN_8B(4, 4)
	SQXTN2_16B(4, 6)

	VST1.P [V0.B16], 16(R1)
	VST1.P [V4.B16], 16(R1)

	SUBS $1, R6, R6
	BNE q8_k_sub32_loop

	SUBS $1, R3, R3
	BNE q8_k_super_loop
	RET

q8_k_sub16:
	// subBlockSize == 16 (16 sub-blocks per super-block)
	MOVD $16, R6
q8_k_sub16_loop:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]

	FMUL_4S(0, 0, 18)
	FMUL_4S(1, 1, 18)
	FMUL_4S(2, 2, 18)
	FMUL_4S(3, 3, 18)

	FRINTN_4S(0, 0)
	FRINTN_4S(1, 1)
	FRINTN_4S(2, 2)
	FRINTN_4S(3, 3)

	FCVTZS_4S(0, 0)
	FCVTZS_4S(1, 1)
	FCVTZS_4S(2, 2)
	FCVTZS_4S(3, 3)

	SMAX_4S(0, 0, 26)
	SMAX_4S(1, 1, 26)
	SMAX_4S(2, 2, 26)
	SMAX_4S(3, 3, 26)

	SMIN_4S(0, 0, 27)
	SMIN_4S(1, 1, 27)
	SMIN_4S(2, 2, 27)
	SMIN_4S(3, 3, 27)

	ADD_4S(19, 0, 1)
	ADD_4S(20, 2, 3)
	ADD_4S(19, 19, 20)
	ADDV_4S(19, 19)
	SCVTF_S(19, 19)
	FMULS F17, F19, F19

	FMOVS F17, (R2)
	FMOVS F19, 4(R2)
	ADD $8, R2

	SQXTN_4H(0, 0)
	SQXTN2_8H(0, 1)
	SQXTN_4H(2, 2)
	SQXTN2_8H(2, 3)
	SQXTN_8B(0, 0)
	SQXTN2_16B(0, 2)

	VST1.P [V0.B16], 16(R1)

	SUBS $1, R6, R6
	BNE q8_k_sub16_loop

	SUBS $1, R3, R3
	BNE q8_k_super_loop
	RET

q8_k_zero:
	ADD $1024, R0
	MOVD $8, R6
q8_k_zero_qx:
	VST1.P [V30.B16, V31.B16], 32(R1)
	SUBS $1, R6, R6
	BNE q8_k_zero_qx

	MOVD $8, R6
	CMP $16, R4
	BNE q8_k_zero_dx
	MOVD $16, R6
q8_k_zero_dx:
	MOVD ZR, (R2)
	ADD $8, R2
	SUBS $1, R6, R6
	BNE q8_k_zero_dx

	SUBS $1, R3, R3
	BNE q8_k_super_loop

q8_k_done:
	RET

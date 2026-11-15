// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

#include "textflag.h"

// ARM64 AdvSIMD (.4S / .8H / .4H) base opcodes and instruction macros:
#define FADD_4S_BASE 0x4E20D400
#define FSUB_4S_BASE 0x4EA0D400
#define FMUL_4S_BASE 0x6E20DC00
#define FDIV_4S_BASE 0x6E20FC00
#define FMAX_4S_BASE 0x4E20F400
#define FMIN_4S_BASE 0x4EA0F400

// FOP_4S encodes Vd.4S = Vn.4S op Vm.4S
#define FOP_4S(BASE, Vd, Vn, Vm) WORD $((BASE) | ((Vm)<<16) | ((Vn)<<5) | (Vd))

#define FCVTL_4S_4H(Vd, Vn)  WORD $(0x0E217800 | ((Vn)<<5) | (Vd))
#define FCVTL2_4S_8H(Vd, Vn) WORD $(0x4E217800 | ((Vn)<<5) | (Vd))
#define FCVTN_4H_4S(Vd, Vn)  WORD $(0x0E216800 | ((Vn)<<5) | (Vd))
#define FCVTN2_8H_4S(Vd, Vn) WORD $(0x4E216800 | ((Vn)<<5) | (Vd))

#define DUP_4H_W(Vd, Rn)     WORD $(0x0E020C00 | ((Rn)<<5) | (Vd))
#define UMOV_W_H0(Rd, Vn)    WORD $(0x0E023C00 | ((Vn)<<5) | (Rd))

// ============================================================================
// Float16 Binary Mid / Leading / Contiguous Broadcast Kernels
// ============================================================================
// Signature: func(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)
//   vec:   [aDim, bDim, cDim] float16
//   bcast: [aDim, 1, cDim]    float16
//   out:   [aDim, bDim, cDim] float16

#define BINARY_MID_RHS_F16_BODY(BASE) \
	MOVD vec+0(FP), R0; \
	MOVD bcast+8(FP), R1; \
	MOVD out+16(FP), R2; \
	MOVD aDim+24(FP), R3; \
	MOVD bDim+32(FP), R4; \
	MOVD cDim+40(FP), R5; \
	CMP $0, R3; \
	BLE done; \
	CMP $0, R4; \
	BLE done; \
	CMP $0, R5; \
	BLE done; \
a_loop: \
	MOVD R4, R6; \
b_loop: \
	MOVD R1, R7; \
	MOVD R5, R8; \
	CMP $32, R8; \
	BLT check8; \
loop32: \
	VLD1.P 64(R0), [V0.H8, V1.H8, V2.H8, V3.H8]; \
	VLD1.P 64(R7), [V4.H8, V5.H8, V6.H8, V7.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FCVTL_4S_4H(18, 1); \
	FCVTL2_4S_8H(19, 1); \
	FCVTL_4S_4H(20, 2); \
	FCVTL2_4S_8H(21, 2); \
	FCVTL_4S_4H(22, 3); \
	FCVTL2_4S_8H(23, 3); \
	FCVTL_4S_4H(24, 4); \
	FCVTL2_4S_8H(25, 4); \
	FCVTL_4S_4H(26, 5); \
	FCVTL2_4S_8H(27, 5); \
	FCVTL_4S_4H(28, 6); \
	FCVTL2_4S_8H(29, 6); \
	FCVTL_4S_4H(30, 7); \
	FCVTL2_4S_8H(31, 7); \
	FOP_4S(BASE, 16, 16, 24); \
	FOP_4S(BASE, 17, 17, 25); \
	FOP_4S(BASE, 18, 18, 26); \
	FOP_4S(BASE, 19, 19, 27); \
	FOP_4S(BASE, 20, 20, 28); \
	FOP_4S(BASE, 21, 21, 29); \
	FOP_4S(BASE, 22, 22, 30); \
	FOP_4S(BASE, 23, 23, 31); \
	FCVTN_4H_4S(0, 16); \
	FCVTN_4H_4S(1, 18); \
	FCVTN_4H_4S(2, 20); \
	FCVTN_4H_4S(3, 22); \
	FCVTN2_8H_4S(0, 17); \
	FCVTN2_8H_4S(1, 19); \
	FCVTN2_8H_4S(2, 21); \
	FCVTN2_8H_4S(3, 23); \
	VST1.P [V0.H8, V1.H8, V2.H8, V3.H8], 64(R2); \
	SUB $32, R8, R8; \
	CMP $32, R8; \
	BGE loop32; \
check8: \
	CMP $8, R8; \
	BLT check1; \
loop8: \
	VLD1.P 16(R0), [V0.H8]; \
	VLD1.P 16(R7), [V4.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FCVTL_4S_4H(24, 4); \
	FCVTL2_4S_8H(25, 4); \
	FOP_4S(BASE, 16, 16, 24); \
	FOP_4S(BASE, 17, 17, 25); \
	FCVTN_4H_4S(0, 16); \
	FCVTN2_8H_4S(0, 17); \
	VST1.P [V0.H8], 16(R2); \
	SUB $8, R8, R8; \
	CMP $8, R8; \
	BGE loop8; \
check1: \
	CBZ R8, next_b; \
loop1: \
	MOVHU.P 2(R0), R9; \
	MOVHU.P 2(R7), R10; \
	DUP_4H_W(0, 9); \
	DUP_4H_W(4, 10); \
	FCVTL_4S_4H(16, 0); \
	FCVTL_4S_4H(24, 4); \
	FOP_4S(BASE, 16, 16, 24); \
	FCVTN_4H_4S(0, 16); \
	UMOV_W_H0(9, 0); \
	MOVH.P R9, 2(R2); \
	SUB $1, R8, R8; \
	CBNZ R8, loop1; \
next_b: \
	SUB $1, R6, R6; \
	CBNZ R6, b_loop; \
	MOVD R7, R1; \
	SUB $1, R3, R3; \
	CBNZ R3, a_loop; \
done: \
	RET

#define BINARY_MID_LHS_F16_BODY(BASE) \
	MOVD vec+0(FP), R0; \
	MOVD bcast+8(FP), R1; \
	MOVD out+16(FP), R2; \
	MOVD aDim+24(FP), R3; \
	MOVD bDim+32(FP), R4; \
	MOVD cDim+40(FP), R5; \
	CMP $0, R3; \
	BLE done; \
	CMP $0, R4; \
	BLE done; \
	CMP $0, R5; \
	BLE done; \
a_loop: \
	MOVD R4, R6; \
b_loop: \
	MOVD R1, R7; \
	MOVD R5, R8; \
	CMP $32, R8; \
	BLT check8; \
loop32: \
	VLD1.P 64(R0), [V0.H8, V1.H8, V2.H8, V3.H8]; \
	VLD1.P 64(R7), [V4.H8, V5.H8, V6.H8, V7.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FCVTL_4S_4H(18, 1); \
	FCVTL2_4S_8H(19, 1); \
	FCVTL_4S_4H(20, 2); \
	FCVTL2_4S_8H(21, 2); \
	FCVTL_4S_4H(22, 3); \
	FCVTL2_4S_8H(23, 3); \
	FCVTL_4S_4H(24, 4); \
	FCVTL2_4S_8H(25, 4); \
	FCVTL_4S_4H(26, 5); \
	FCVTL2_4S_8H(27, 5); \
	FCVTL_4S_4H(28, 6); \
	FCVTL2_4S_8H(29, 6); \
	FCVTL_4S_4H(30, 7); \
	FCVTL2_4S_8H(31, 7); \
	FOP_4S(BASE, 16, 24, 16); \
	FOP_4S(BASE, 17, 25, 17); \
	FOP_4S(BASE, 18, 26, 18); \
	FOP_4S(BASE, 19, 27, 19); \
	FOP_4S(BASE, 20, 28, 20); \
	FOP_4S(BASE, 21, 29, 21); \
	FOP_4S(BASE, 22, 30, 22); \
	FOP_4S(BASE, 23, 31, 23); \
	FCVTN_4H_4S(0, 16); \
	FCVTN_4H_4S(1, 18); \
	FCVTN_4H_4S(2, 20); \
	FCVTN_4H_4S(3, 22); \
	FCVTN2_8H_4S(0, 17); \
	FCVTN2_8H_4S(1, 19); \
	FCVTN2_8H_4S(2, 21); \
	FCVTN2_8H_4S(3, 23); \
	VST1.P [V0.H8, V1.H8, V2.H8, V3.H8], 64(R2); \
	SUB $32, R8, R8; \
	CMP $32, R8; \
	BGE loop32; \
check8: \
	CMP $8, R8; \
	BLT check1; \
loop8: \
	VLD1.P 16(R0), [V0.H8]; \
	VLD1.P 16(R7), [V4.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FCVTL_4S_4H(24, 4); \
	FCVTL2_4S_8H(25, 4); \
	FOP_4S(BASE, 16, 24, 16); \
	FOP_4S(BASE, 17, 25, 17); \
	FCVTN_4H_4S(0, 16); \
	FCVTN2_8H_4S(0, 17); \
	VST1.P [V0.H8], 16(R2); \
	SUB $8, R8, R8; \
	CMP $8, R8; \
	BGE loop8; \
check1: \
	CBZ R8, next_b; \
loop1: \
	MOVHU.P 2(R0), R9; \
	MOVHU.P 2(R7), R10; \
	DUP_4H_W(0, 9); \
	DUP_4H_W(4, 10); \
	FCVTL_4S_4H(16, 0); \
	FCVTL_4S_4H(24, 4); \
	FOP_4S(BASE, 16, 24, 16); \
	FCVTN_4H_4S(0, 16); \
	UMOV_W_H0(9, 0); \
	MOVH.P R9, 2(R2); \
	SUB $1, R8, R8; \
	CBNZ R8, loop1; \
next_b: \
	SUB $1, R6, R6; \
	CBNZ R6, b_loop; \
	MOVD R7, R1; \
	SUB $1, R3, R3; \
	CBNZ R3, a_loop; \
done: \
	RET

TEXT ·binaryMidAddFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_F16_BODY(FADD_4S_BASE)

TEXT ·binaryMidSubRHSFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_F16_BODY(FSUB_4S_BASE)

TEXT ·binaryMidSubLHSFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_LHS_F16_BODY(FSUB_4S_BASE)

TEXT ·binaryMidMulFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_F16_BODY(FMUL_4S_BASE)

TEXT ·binaryMidDivRHSFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_F16_BODY(FDIV_4S_BASE)

TEXT ·binaryMidDivLHSFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_LHS_F16_BODY(FDIV_4S_BASE)

TEXT ·binaryMidMaxFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_F16_BODY(FMAX_4S_BASE)

TEXT ·binaryMidMinFloat16NEONAsm(SB), NOSPLIT, $0-48
	BINARY_MID_RHS_F16_BODY(FMIN_4S_BASE)

// ============================================================================
// Float16 Binary Trailing Broadcast Kernels
// ============================================================================
// Signature: func(vec, scalar, out unsafe.Pointer, aDim, bDim int)
//   vec:    [aDim, bDim] float16
//   scalar: [aDim]       float16
//   out:    [aDim, bDim] float16

#define BINARY_TRAILING_RHS_F16_BODY(BASE) \
	MOVD vec+0(FP), R0; \
	MOVD scalar+8(FP), R1; \
	MOVD out+16(FP), R2; \
	MOVD aDim+24(FP), R3; \
	MOVD bDim+32(FP), R4; \
	CMP $0, R3; \
	BLE tr_done; \
	CMP $0, R4; \
	BLE tr_done; \
tr_a_loop: \
	MOVHU.P 2(R1), R9; \
	DUP_4H_W(4, 9); \
	FCVTL_4S_4H(24, 4); \
	MOVD R4, R8; \
	CMP $32, R8; \
	BLT tr_check8; \
tr_loop32: \
	VLD1.P 64(R0), [V0.H8, V1.H8, V2.H8, V3.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FCVTL_4S_4H(18, 1); \
	FCVTL2_4S_8H(19, 1); \
	FCVTL_4S_4H(20, 2); \
	FCVTL2_4S_8H(21, 2); \
	FCVTL_4S_4H(22, 3); \
	FCVTL2_4S_8H(23, 3); \
	FOP_4S(BASE, 16, 16, 24); \
	FOP_4S(BASE, 17, 17, 24); \
	FOP_4S(BASE, 18, 18, 24); \
	FOP_4S(BASE, 19, 19, 24); \
	FOP_4S(BASE, 20, 20, 24); \
	FOP_4S(BASE, 21, 21, 24); \
	FOP_4S(BASE, 22, 22, 24); \
	FOP_4S(BASE, 23, 23, 24); \
	FCVTN_4H_4S(0, 16); \
	FCVTN_4H_4S(1, 18); \
	FCVTN_4H_4S(2, 20); \
	FCVTN_4H_4S(3, 22); \
	FCVTN2_8H_4S(0, 17); \
	FCVTN2_8H_4S(1, 19); \
	FCVTN2_8H_4S(2, 21); \
	FCVTN2_8H_4S(3, 23); \
	VST1.P [V0.H8, V1.H8, V2.H8, V3.H8], 64(R2); \
	SUB $32, R8, R8; \
	CMP $32, R8; \
	BGE tr_loop32; \
tr_check8: \
	CMP $8, R8; \
	BLT tr_check1; \
tr_loop8: \
	VLD1.P 16(R0), [V0.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FOP_4S(BASE, 16, 16, 24); \
	FOP_4S(BASE, 17, 17, 24); \
	FCVTN_4H_4S(0, 16); \
	FCVTN2_8H_4S(0, 17); \
	VST1.P [V0.H8], 16(R2); \
	SUB $8, R8, R8; \
	CMP $8, R8; \
	BGE tr_loop8; \
tr_check1: \
	CBZ R8, tr_next_a; \
tr_loop1: \
	MOVHU.P 2(R0), R9; \
	DUP_4H_W(0, 9); \
	FCVTL_4S_4H(16, 0); \
	FOP_4S(BASE, 16, 16, 24); \
	FCVTN_4H_4S(0, 16); \
	UMOV_W_H0(9, 0); \
	MOVH.P R9, 2(R2); \
	SUB $1, R8, R8; \
	CBNZ R8, tr_loop1; \
tr_next_a: \
	SUB $1, R3, R3; \
	CBNZ R3, tr_a_loop; \
tr_done: \
	RET

#define BINARY_TRAILING_LHS_F16_BODY(BASE) \
	MOVD vec+0(FP), R0; \
	MOVD scalar+8(FP), R1; \
	MOVD out+16(FP), R2; \
	MOVD aDim+24(FP), R3; \
	MOVD bDim+32(FP), R4; \
	CMP $0, R3; \
	BLE tr_done; \
	CMP $0, R4; \
	BLE tr_done; \
tr_a_loop: \
	MOVHU.P 2(R1), R9; \
	DUP_4H_W(4, 9); \
	FCVTL_4S_4H(24, 4); \
	MOVD R4, R8; \
	CMP $32, R8; \
	BLT tr_check8; \
tr_loop32: \
	VLD1.P 64(R0), [V0.H8, V1.H8, V2.H8, V3.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FCVTL_4S_4H(18, 1); \
	FCVTL2_4S_8H(19, 1); \
	FCVTL_4S_4H(20, 2); \
	FCVTL2_4S_8H(21, 2); \
	FCVTL_4S_4H(22, 3); \
	FCVTL2_4S_8H(23, 3); \
	FOP_4S(BASE, 16, 24, 16); \
	FOP_4S(BASE, 17, 24, 17); \
	FOP_4S(BASE, 18, 24, 18); \
	FOP_4S(BASE, 19, 24, 19); \
	FOP_4S(BASE, 20, 24, 20); \
	FOP_4S(BASE, 21, 24, 21); \
	FOP_4S(BASE, 22, 24, 22); \
	FOP_4S(BASE, 23, 24, 23); \
	FCVTN_4H_4S(0, 16); \
	FCVTN_4H_4S(1, 18); \
	FCVTN_4H_4S(2, 20); \
	FCVTN_4H_4S(3, 22); \
	FCVTN2_8H_4S(0, 17); \
	FCVTN2_8H_4S(1, 19); \
	FCVTN2_8H_4S(2, 21); \
	FCVTN2_8H_4S(3, 23); \
	VST1.P [V0.H8, V1.H8, V2.H8, V3.H8], 64(R2); \
	SUB $32, R8, R8; \
	CMP $32, R8; \
	BGE tr_loop32; \
tr_check8: \
	CMP $8, R8; \
	BLT tr_check1; \
tr_loop8: \
	VLD1.P 16(R0), [V0.H8]; \
	FCVTL_4S_4H(16, 0); \
	FCVTL2_4S_8H(17, 0); \
	FOP_4S(BASE, 16, 24, 16); \
	FOP_4S(BASE, 17, 24, 17); \
	FCVTN_4H_4S(0, 16); \
	FCVTN2_8H_4S(0, 17); \
	VST1.P [V0.H8], 16(R2); \
	SUB $8, R8, R8; \
	CMP $8, R8; \
	BGE tr_loop8; \
tr_check1: \
	CBZ R8, tr_next_a; \
tr_loop1: \
	MOVHU.P 2(R0), R9; \
	DUP_4H_W(0, 9); \
	FCVTL_4S_4H(16, 0); \
	FOP_4S(BASE, 16, 24, 16); \
	FCVTN_4H_4S(0, 16); \
	UMOV_W_H0(9, 0); \
	MOVH.P R9, 2(R2); \
	SUB $1, R8, R8; \
	CBNZ R8, tr_loop1; \
tr_next_a: \
	SUB $1, R3, R3; \
	CBNZ R3, tr_a_loop; \
tr_done: \
	RET

TEXT ·binaryTrailingAddFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_F16_BODY(FADD_4S_BASE)

TEXT ·binaryTrailingSubRHSFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_F16_BODY(FSUB_4S_BASE)

TEXT ·binaryTrailingSubLHSFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_LHS_F16_BODY(FSUB_4S_BASE)

TEXT ·binaryTrailingMulFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_F16_BODY(FMUL_4S_BASE)

TEXT ·binaryTrailingDivRHSFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_F16_BODY(FDIV_4S_BASE)

TEXT ·binaryTrailingDivLHSFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_LHS_F16_BODY(FDIV_4S_BASE)

TEXT ·binaryTrailingMaxFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_F16_BODY(FMAX_4S_BASE)

TEXT ·binaryTrailingMinFloat16NEONAsm(SB), NOSPLIT, $0-40
	BINARY_TRAILING_RHS_F16_BODY(FMIN_4S_BASE)

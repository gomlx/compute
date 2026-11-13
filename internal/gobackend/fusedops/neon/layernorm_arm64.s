// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

#define FADD_4S(Vd, Vn, Vm)  WORD $(0x4E20D400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FSUB_4S(Vd, Vn, Vm)  WORD $(0x4EA0D400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMUL_4S(Vd, Vn, Vm)  WORD $(0x6E20DC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMLA_4S(Vd, Vn, Vm)  WORD $(0x4E20CC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FADDP_4S(Vd, Vn, Vm) WORD $(0x6E20D400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define DUP_4S_LANE0(Vd, Vn) WORD $(0x4E040400 | ((Vn)<<5) | (Vd))

// ===========================================================================
// 1. layerNormFloat32NEONAsm
// ===========================================================================
// func layerNormFloat32NEONAsm(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)
// Stack frame (52 bytes):
//   in:        0(FP)
//   out:       8(FP)
//   gamma:    16(FP)
//   beta:     24(FP)
//   outerSize:32(FP)
//   normSize: 40(FP)
//   epsilon:  48(FP)
TEXT ·layerNormFloat32NEONAsm(SB), NOSPLIT, $0-52
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD gamma+16(FP), R2
	MOVD beta+24(FP), R3
	MOVD outerSize+32(FP), R4
	MOVD normSize+40(FP), R5
	FMOVS epsilon+48(FP), F15

	CMP $0, R4
	BLE ln_done
	CMP $0, R5
	BLE ln_done

	SCVTFWS R5, F14
	FMOVS $(1.0), F13
	FDIVS F14, F13, F14 // F14 = 1.0 / float32(normSize)
	LSL $2, R5, R6      // rowBytes = normSize * 4

ln_outer_loop:
	// --- PASS 1: MEAN ---
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	MOVD R0, R8 // inPtr
	MOVD R5, R9 // rem = normSize

ln_p1_loop16:
	CMP $16, R9
	BLT ln_p1_loop4
	VLD1.P 64(R8), [V4.S4, V5.S4, V6.S4, V7.S4]
	FADD_4S(0, 0, 4)
	FADD_4S(1, 1, 5)
	FADD_4S(2, 2, 6)
	FADD_4S(3, 3, 7)
	SUB $16, R9, R9
	B ln_p1_loop16

ln_p1_loop4:
	CMP $4, R9
	BLT ln_p1_reduce
	VLD1.P 16(R8), [V4.S4]
	FADD_4S(0, 0, 4)
	SUB $4, R9, R9
	B ln_p1_loop4

ln_p1_reduce:
	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0) // F0 = horizontal sum

ln_p1_scalar:
	CBZ R9, ln_p1_done
	FMOVS.P 4(R8), F4
	FADDS F4, F0, F0
	SUB $1, R9, R9
	B ln_p1_scalar

ln_p1_done:
	FMULS F14, F0, F12     // F12 = mean
	DUP_4S_LANE0(12, 12)   // V12.4S = broadcast(mean)

	// --- PASS 2: VARIANCE ---
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	MOVD R0, R8
	MOVD R5, R9

ln_p2_loop16:
	CMP $16, R9
	BLT ln_p2_loop4
	VLD1.P 64(R8), [V4.S4, V5.S4, V6.S4, V7.S4]
	FSUB_4S(4, 4, 12)
	FSUB_4S(5, 5, 12)
	FSUB_4S(6, 6, 12)
	FSUB_4S(7, 7, 12)
	FMLA_4S(0, 4, 4)
	FMLA_4S(1, 5, 5)
	FMLA_4S(2, 6, 6)
	FMLA_4S(3, 7, 7)
	SUB $16, R9, R9
	B ln_p2_loop16

ln_p2_loop4:
	CMP $4, R9
	BLT ln_p2_reduce
	VLD1.P 16(R8), [V4.S4]
	FSUB_4S(4, 4, 12)
	FMLA_4S(0, 4, 4)
	SUB $4, R9, R9
	B ln_p2_loop4

ln_p2_reduce:
	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

ln_p2_scalar:
	CBZ R9, ln_p2_done
	FMOVS.P 4(R8), F4
	FSUBS F12, F4, F4
	FMULS F4, F4, F4
	FADDS F4, F0, F0
	SUB $1, R9, R9
	B ln_p2_scalar

ln_p2_done:
	FMULS F14, F0, F0
	FADDS F15, F0, F0
	FSQRTS F0, F0
	FDIVS F0, F13, F16     // F16 = invStd = 1.0 / sqrt(var + eps)
	DUP_4S_LANE0(16, 16)   // V16.4S = broadcast(invStd)

	// --- PASS 3: NORMALIZE + AFFINE ---
	MOVD R0, R8
	MOVD R1, R10
	MOVD R5, R9

	CBZ R2, ln_no_gamma
	CBZ R3, ln_gamma_only

	// Both gamma (R2) and beta (R3)
	MOVD R2, R11
	MOVD R3, R12
ln_p3_both_loop16:
	CMP $16, R9
	BLT ln_p3_both_loop4
	VLD1.P 64(R8), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R11), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R12), [V8.S4, V9.S4, V10.S4, V11.S4]
	FSUB_4S(0, 0, 12)
	FSUB_4S(1, 1, 12)
	FSUB_4S(2, 2, 12)
	FSUB_4S(3, 3, 12)
	FMUL_4S(4, 4, 16)
	FMUL_4S(5, 5, 16)
	FMUL_4S(6, 6, 16)
	FMUL_4S(7, 7, 16)
	FMLA_4S(8, 0, 4)
	FMLA_4S(9, 1, 5)
	FMLA_4S(10, 2, 6)
	FMLA_4S(11, 3, 7)
	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R10)
	SUB $16, R9, R9
	B ln_p3_both_loop16

ln_p3_both_loop4:
	CMP $4, R9
	BLT ln_p3_both_scalar
	VLD1.P 16(R8), [V0.S4]
	VLD1.P 16(R11), [V4.S4]
	VLD1.P 16(R12), [V8.S4]
	FSUB_4S(0, 0, 12)
	FMUL_4S(4, 4, 16)
	FMLA_4S(8, 0, 4)
	VST1.P [V8.S4], 16(R10)
	SUB $4, R9, R9
	B ln_p3_both_loop4

ln_p3_both_scalar:
	CBZ R9, ln_next_row
	FMOVS.P 4(R8), F0
	FMOVS.P 4(R11), F4
	FMOVS.P 4(R12), F8
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FMULS F4, F0, F0
	FADDS F8, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R9, R9
	B ln_p3_both_scalar

ln_gamma_only:
	MOVD R2, R11
ln_p3_gamma_loop4:
	CMP $4, R9
	BLT ln_p3_gamma_scalar
	VLD1.P 16(R8), [V0.S4]
	VLD1.P 16(R11), [V4.S4]
	FSUB_4S(0, 0, 12)
	FMUL_4S(0, 0, 16)
	FMUL_4S(0, 0, 4)
	VST1.P [V0.S4], 16(R10)
	SUB $4, R9, R9
	B ln_p3_gamma_loop4

ln_p3_gamma_scalar:
	CBZ R9, ln_next_row
	FMOVS.P 4(R8), F0
	FMOVS.P 4(R11), F4
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FMULS F4, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R9, R9
	B ln_p3_gamma_scalar

ln_no_gamma:
	CBZ R3, ln_no_affine
	MOVD R3, R12
ln_p3_beta_loop4:
	CMP $4, R9
	BLT ln_p3_beta_scalar
	VLD1.P 16(R8), [V0.S4]
	VLD1.P 16(R12), [V8.S4]
	FSUB_4S(0, 0, 12)
	FMLA_4S(8, 0, 16)
	VST1.P [V8.S4], 16(R10)
	SUB $4, R9, R9
	B ln_p3_beta_loop4

ln_p3_beta_scalar:
	CBZ R9, ln_next_row
	FMOVS.P 4(R8), F0
	FMOVS.P 4(R12), F8
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FADDS F8, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R9, R9
	B ln_p3_beta_scalar

ln_no_affine:
ln_p3_none_loop4:
	CMP $4, R9
	BLT ln_p3_none_scalar
	VLD1.P 16(R8), [V0.S4]
	FSUB_4S(0, 0, 12)
	FMUL_4S(0, 0, 16)
	VST1.P [V0.S4], 16(R10)
	SUB $4, R9, R9
	B ln_p3_none_loop4

ln_p3_none_scalar:
	CBZ R9, ln_next_row
	FMOVS.P 4(R8), F0
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R9, R9
	B ln_p3_none_scalar

ln_next_row:
	ADD R6, R0, R0
	ADD R6, R1, R1
	SUB $1, R4, R4
	CBNZ R4, ln_outer_loop

ln_done:
	RET

// ===========================================================================
// 2. layerNormResidualFloat32NEONAsm
// ===========================================================================
// func layerNormResidualFloat32NEONAsm(in, residual, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)
// Stack frame (60 bytes):
//   in:        0(FP)
//   residual:  8(FP)
//   out:      16(FP)
//   gamma:    24(FP)
//   beta:     32(FP)
//   outerSize:40(FP)
//   normSize: 48(FP)
//   epsilon:  56(FP)
TEXT ·layerNormResidualFloat32NEONAsm(SB), NOSPLIT, $0-60
	MOVD in+0(FP), R0
	MOVD residual+8(FP), R1
	MOVD out+16(FP), R2
	MOVD gamma+24(FP), R3
	MOVD beta+32(FP), R4
	MOVD outerSize+40(FP), R5
	MOVD normSize+48(FP), R6
	FMOVS epsilon+56(FP), F15

	CMP $0, R5
	BLE lnr_done
	CMP $0, R6
	BLE lnr_done

	SCVTFWS R6, F14
	FMOVS $(1.0), F13
	FDIVS F14, F13, F14 // F14 = 1.0 / float32(normSize)
	LSL $2, R6, R7      // rowBytes = normSize * 4

lnr_outer_loop:
	// --- PASS 1: FUSED (IN + RESIDUAL -> OUT) + MEAN ---
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	MOVD R0, R8  // inPtr
	MOVD R1, R9  // resPtr
	MOVD R2, R10 // outPtr
	MOVD R6, R11 // rem = normSize

lnr_p1_loop16:
	CMP $16, R11
	BLT lnr_p1_loop4
	VLD1.P 64(R8), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R9), [V16.S4, V17.S4, V18.S4, V19.S4]
	FADD_4S(4, 4, 16)
	FADD_4S(5, 5, 17)
	FADD_4S(6, 6, 18)
	FADD_4S(7, 7, 19)
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R10)
	FADD_4S(0, 0, 4)
	FADD_4S(1, 1, 5)
	FADD_4S(2, 2, 6)
	FADD_4S(3, 3, 7)
	SUB $16, R11, R11
	B lnr_p1_loop16

lnr_p1_loop4:
	CMP $4, R11
	BLT lnr_p1_reduce
	VLD1.P 16(R8), [V4.S4]
	VLD1.P 16(R9), [V16.S4]
	FADD_4S(4, 4, 16)
	VST1.P [V4.S4], 16(R10)
	FADD_4S(0, 0, 4)
	SUB $4, R11, R11
	B lnr_p1_loop4

lnr_p1_reduce:
	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

lnr_p1_scalar:
	CBZ R11, lnr_p1_done
	FMOVS.P 4(R8), F4
	FMOVS.P 4(R9), F16
	FADDS F16, F4, F4
	FMOVS.P F4, 4(R10)
	FADDS F4, F0, F0
	SUB $1, R11, R11
	B lnr_p1_scalar

lnr_p1_done:
	FMULS F14, F0, F12     // F12 = mean
	DUP_4S_LANE0(12, 12)   // V12.4S = broadcast(mean)

	// --- PASS 2: VARIANCE (from OUT) ---
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	MOVD R2, R8
	MOVD R6, R11

lnr_p2_loop16:
	CMP $16, R11
	BLT lnr_p2_loop4
	VLD1.P 64(R8), [V4.S4, V5.S4, V6.S4, V7.S4]
	FSUB_4S(4, 4, 12)
	FSUB_4S(5, 5, 12)
	FSUB_4S(6, 6, 12)
	FSUB_4S(7, 7, 12)
	FMLA_4S(0, 4, 4)
	FMLA_4S(1, 5, 5)
	FMLA_4S(2, 6, 6)
	FMLA_4S(3, 7, 7)
	SUB $16, R11, R11
	B lnr_p2_loop16

lnr_p2_loop4:
	CMP $4, R11
	BLT lnr_p2_reduce
	VLD1.P 16(R8), [V4.S4]
	FSUB_4S(4, 4, 12)
	FMLA_4S(0, 4, 4)
	SUB $4, R11, R11
	B lnr_p2_loop4

lnr_p2_reduce:
	FADD_4S(0, 0, 1)
	FADD_4S(2, 2, 3)
	FADD_4S(0, 0, 2)
	FADDP_4S(0, 0, 0)
	FADDP_4S(0, 0, 0)

lnr_p2_scalar:
	CBZ R11, lnr_p2_done
	FMOVS.P 4(R8), F4
	FSUBS F12, F4, F4
	FMULS F4, F4, F4
	FADDS F4, F0, F0
	SUB $1, R11, R11
	B lnr_p2_scalar

lnr_p2_done:
	FMULS F14, F0, F0
	FADDS F15, F0, F0
	FSQRTS F0, F0
	FDIVS F0, F13, F16     // F16 = invStd
	DUP_4S_LANE0(16, 16)   // V16.4S = broadcast(invStd)

	// --- PASS 3: NORMALIZE + AFFINE (in-place on OUT) ---
	MOVD R2, R8
	MOVD R2, R10
	MOVD R6, R11

	CBZ R3, lnr_no_gamma
	CBZ R4, lnr_gamma_only

	MOVD R3, R12
	MOVD R4, R13
lnr_p3_both_loop16:
	CMP $16, R11
	BLT lnr_p3_both_loop4
	VLD1.P 64(R8), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R12), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R13), [V8.S4, V9.S4, V10.S4, V11.S4]
	FSUB_4S(0, 0, 12)
	FSUB_4S(1, 1, 12)
	FSUB_4S(2, 2, 12)
	FSUB_4S(3, 3, 12)
	FMUL_4S(4, 4, 16)
	FMUL_4S(5, 5, 16)
	FMUL_4S(6, 6, 16)
	FMUL_4S(7, 7, 16)
	FMLA_4S(8, 0, 4)
	FMLA_4S(9, 1, 5)
	FMLA_4S(10, 2, 6)
	FMLA_4S(11, 3, 7)
	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R10)
	SUB $16, R11, R11
	B lnr_p3_both_loop16

lnr_p3_both_loop4:
	CMP $4, R11
	BLT lnr_p3_both_scalar
	VLD1.P 16(R8), [V0.S4]
	VLD1.P 16(R12), [V4.S4]
	VLD1.P 16(R13), [V8.S4]
	FSUB_4S(0, 0, 12)
	FMUL_4S(4, 4, 16)
	FMLA_4S(8, 0, 4)
	VST1.P [V8.S4], 16(R10)
	SUB $4, R11, R11
	B lnr_p3_both_loop4

lnr_p3_both_scalar:
	CBZ R11, lnr_next_row
	FMOVS.P 4(R8), F0
	FMOVS.P 4(R12), F4
	FMOVS.P 4(R13), F8
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FMULS F4, F0, F0
	FADDS F8, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R11, R11
	B lnr_p3_both_scalar

lnr_gamma_only:
	MOVD R3, R12
lnr_p3_gamma_loop4:
	CMP $4, R11
	BLT lnr_p3_gamma_scalar
	VLD1.P 16(R8), [V0.S4]
	VLD1.P 16(R12), [V4.S4]
	FSUB_4S(0, 0, 12)
	FMUL_4S(0, 0, 16)
	FMUL_4S(0, 0, 4)
	VST1.P [V0.S4], 16(R10)
	SUB $4, R11, R11
	B lnr_p3_gamma_loop4

lnr_p3_gamma_scalar:
	CBZ R11, lnr_next_row
	FMOVS.P 4(R8), F0
	FMOVS.P 4(R12), F4
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FMULS F4, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R11, R11
	B lnr_p3_gamma_scalar

lnr_no_gamma:
	CBZ R4, lnr_no_affine
	MOVD R4, R13
lnr_p3_beta_loop4:
	CMP $4, R11
	BLT lnr_p3_beta_scalar
	VLD1.P 16(R8), [V0.S4]
	VLD1.P 16(R13), [V8.S4]
	FSUB_4S(0, 0, 12)
	FMLA_4S(8, 0, 16)
	VST1.P [V8.S4], 16(R10)
	SUB $4, R11, R11
	B lnr_p3_beta_loop4

lnr_p3_beta_scalar:
	CBZ R11, lnr_next_row
	FMOVS.P 4(R8), F0
	FMOVS.P 4(R13), F8
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FADDS F8, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R11, R11
	B lnr_p3_beta_scalar

lnr_no_affine:
lnr_p3_none_loop4:
	CMP $4, R11
	BLT lnr_p3_none_scalar
	VLD1.P 16(R8), [V0.S4]
	FSUB_4S(0, 0, 12)
	FMUL_4S(0, 0, 16)
	VST1.P [V0.S4], 16(R10)
	SUB $4, R11, R11
	B lnr_p3_none_scalar

lnr_p3_none_scalar:
	CBZ R11, lnr_next_row
	FMOVS.P 4(R8), F0
	FSUBS F12, F0, F0
	FMULS F16, F0, F0
	FMOVS.P F0, 4(R10)
	SUB $1, R11, R11
	B lnr_p3_none_scalar

lnr_next_row:
	ADD R7, R0, R0
	ADD R7, R1, R1
	ADD R7, R2, R2
	SUB $1, R5, R5
	CBNZ R5, lnr_outer_loop

lnr_done:
	RET

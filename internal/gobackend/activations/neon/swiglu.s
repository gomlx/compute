// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// ARM64 AdvSIMD (.4S / .16B) instruction macros (destination Vd first: Vd = Vn op Vm):
#define FADD_4S(Vd, Vn, Vm)   WORD $(0x4E20D400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FSUB_4S(Vd, Vn, Vm)   WORD $(0x4EA0D400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMUL_4S(Vd, Vn, Vm)   WORD $(0x6E20DC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FDIV_4S(Vd, Vn, Vm)   WORD $(0x6E20FC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMLA_4S(Vd, Vn, Vm)   WORD $(0x4E20CC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMLS_4S(Vd, Vn, Vm)   WORD $(0x4EA0CC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMAX_4S(Vd, Vn, Vm)   WORD $(0x4E20F400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMIN_4S(Vd, Vn, Vm)   WORD $(0x4EA0F400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FNEG_4S(Vd, Vn)       WORD $(0x6EA0F800 | ((Vn)<<5) | (Vd))
#define FRECPE_4S(Vd, Vn)     WORD $(0x4EA1D800 | ((Vn)<<5) | (Vd))
#define FRECPS_4S(Vd, Vn, Vm) WORD $(0x4E20FC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define SHL23_4S(Vd, Vn)      WORD $(0x4F375400 | ((Vn)<<5) | (Vd))
#define ADD_4S(Vd, Vn, Vm)    WORD $(0x4EA08400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define MOV_16B(Vd, Vn)       WORD $(0x4EA01C00 | ((Vn)<<16) | ((Vn)<<5) | (Vd))
#define DUP_4S_W(Vd, Rn)      WORD $(0x4E040C00 | ((Rn)<<5) | (Vd))

// ===========================================================================
// swigluFloat32NEONAsm: SwiGLU forward for float32 using ARM64 NEON
//   For m in [0, numRows) and j in [0, vecWidth):
//     gate = in[m*2*hiddenDim + j]
//     val  = in[m*2*hiddenDim + hiddenDim + j]
//     out[m*hiddenDim + j] = (gate * val) / (1 + exp(-gate))
//   Uses 4-vector (16 float32) unrolling, magic-number 2^n scaling,
//   Estrin's degree-7 polynomial tree, and fast reciprocal (FRECPE + FRECPS).
//   vecWidth must be a positive multiple of 4.
// ===========================================================================
// func swigluFloat32NEONAsm(in, out unsafe.Pointer, numRows, hiddenDim, vecWidth int)
TEXT ·swigluFloat32NEONAsm(SB), NOSPLIT, $0-40
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD numRows+16(FP), R2
	MOVD hiddenDim+24(FP), R3
	MOVD vecWidth+32(FP), R4

	CMP $0, R2
	BLE swiglu_neon_done
	CMP $4, R4
	BLT swiglu_neon_done

	// R9 = hiddenDim * 4 (both halfByteOffset and outRowByteStride)
	// R10 = 2 * hiddenDim * 4 (inRowByteStride)
	LSL $2, R3, R9
	LSL $3, R3, R10

	// Broadcast all 13 constants into V19..V31 once outside the row loop:
	MOVD $0x39506967, R5; DUP_4S_W(19, 5) // V19: vP7 = 1.9875691500e-4
	MOVD $0x3ab743ce, R5; DUP_4S_W(20, 5) // V20: vP6 = 1.3981999507e-3
	MOVD $0x3c088895, R5; DUP_4S_W(21, 5) // V21: vP5 = 8.3334519073e-3
	MOVD $0x3d2aa9a5, R5; DUP_4S_W(22, 5) // V22: vP4 = 4.1665795894e-2
	MOVD $0x3e2aaaab, R5; DUP_4S_W(23, 5) // V23: vP3 = 1.6666665459e-1
	MOVD $0x3f000000, R5; DUP_4S_W(24, 5) // V24: vP2 / vHalf = 0.5
	MOVD $0x3f800000, R5; DUP_4S_W(25, 5) // V25: vOne / vBias127 = 1.0 (127 << 23)
	MOVD $0xc2aeac50, R5; DUP_4S_W(26, 5) // V26: vMinLog = -87.336544
	MOVD $0x42b00f34, R5; DUP_4S_W(27, 5) // V27: vMaxLog = 88.02969187
	MOVD $0x3fb8aa3b, R5; DUP_4S_W(28, 5) // V28: vLog2E = 1.44269504
	MOVD $0x4b400000, R5; DUP_4S_W(29, 5) // V29: vMagic = 12582912.0
	MOVD $0x3f318000, R5; DUP_4S_W(30, 5) // V30: vLn2Hi = 0.693359375
	MOVD $0xb95e8083, R5; DUP_4S_W(31, 5) // V31: vLn2Lo = -2.12194440e-4

swiglu_neon_row_loop:
	MOVD R0, R5
	ADD R9, R0, R6
	MOVD R1, R7
	MOVD R4, R8

	CMP $16, R8
	BLT swiglu_neon_vec8_check

swiglu_neon_vec16_loop:
	VLD1.P 64(R5), [V0.S4, V1.S4, V2.S4, V3.S4]   // gate[0..15]
	VLD1.P 64(R6), [V8.S4, V9.S4, V10.S4, V11.S4] // val[0..15]

	// u = clamp(-gate, vMinLog, vMaxLog); numer = gate * val
	FNEG_4S(4, 0)
	FNEG_4S(5, 1)
	FNEG_4S(6, 2)
	FNEG_4S(7, 3)
	FMUL_4S(0, 0, 8)
	FMUL_4S(1, 1, 9)
	FMUL_4S(2, 2, 10)
	FMUL_4S(3, 3, 11)
	FMAX_4S(4, 4, 26)
	FMAX_4S(5, 5, 26)
	FMAX_4S(6, 6, 26)
	FMAX_4S(7, 7, 26)
	FMIN_4S(4, 4, 27)
	FMIN_4S(5, 5, 27)
	FMIN_4S(6, 6, 27)
	FMIN_4S(7, 7, 27)

	// Range reduction across all 4 vectors: zRound = u * vLog2E + vMagic, zFloor = zRound - vMagic
	MOV_16B(8, 29)
	MOV_16B(9, 29)
	MOV_16B(10, 29)
	MOV_16B(11, 29)
	FMLA_4S(8, 4, 28)
	FMLA_4S(9, 5, 28)
	FMLA_4S(10, 6, 28)
	FMLA_4S(11, 7, 28)
	FSUB_4S(12, 8, 29)
	FSUB_4S(13, 9, 29)
	FSUB_4S(14, 10, 29)
	FSUB_4S(15, 11, 29)

	// g = u - zFloor * vLn2Hi - zFloor * vLn2Lo
	FMLS_4S(4, 12, 30)
	FMLS_4S(5, 13, 30)
	FMLS_4S(6, 14, 30)
	FMLS_4S(7, 15, 30)
	FMLS_4S(4, 12, 31)
	FMLS_4S(5, 13, 31)
	FMLS_4S(6, 14, 31)
	FMLS_4S(7, 15, 31)

	// scale = (zRound << 23) + vBias127 in V8..V11
	SHL23_4S(8, 8)
	SHL23_4S(9, 9)
	SHL23_4S(10, 10)
	SHL23_4S(11, 11)
	ADD_4S(8, 8, 25)
	ADD_4S(9, 9, 25)
	ADD_4S(10, 10, 25)
	ADD_4S(11, 11, 25)

	// Estrin's degree-7 polynomial for pair (V4, V5):
	// Scratch: V12, V13 = g2; V14, V15 = q67/q23; V16, V17 = q45; V4, V5 = q01 (in-place after linear terms)
	FMUL_4S(12, 4, 4)
	FMUL_4S(13, 5, 5)
	MOV_16B(14, 20)
	MOV_16B(15, 20)
	FMLA_4S(14, 4, 19)
	FMLA_4S(15, 5, 19)
	MOV_16B(16, 22)
	MOV_16B(17, 22)
	FMLA_4S(16, 4, 21)
	FMLA_4S(17, 5, 21)
	FMLA_4S(16, 12, 14)
	FMLA_4S(17, 13, 15)
	MOV_16B(14, 24)
	MOV_16B(15, 24)
	FMLA_4S(14, 4, 23)
	FMLA_4S(15, 5, 23)
	FADD_4S(4, 4, 25)
	FADD_4S(5, 5, 25)
	FMLA_4S(4, 12, 14)
	FMLA_4S(5, 13, 15)
	FMUL_4S(12, 12, 12)
	FMUL_4S(13, 13, 13)
	FMLA_4S(4, 12, 16)
	FMLA_4S(5, 13, 17)

	// Compute denom0, denom1 = 1.0 + scale * expPoly and issue FDIV on (V0, V1) early
	// so hardware division executes in the background while pair (V6, V7) evaluates exp(g):
	MOV_16B(12, 25)
	MOV_16B(13, 25)
	FMLA_4S(12, 4, 8)
	FMLA_4S(13, 5, 9)
	FDIV_4S(0, 0, 12)
	FDIV_4S(1, 1, 13)

	// Estrin's degree-7 polynomial for pair (V6, V7):
	FMUL_4S(12, 6, 6)
	FMUL_4S(13, 7, 7)
	MOV_16B(14, 20)
	MOV_16B(15, 20)
	FMLA_4S(14, 6, 19)
	FMLA_4S(15, 7, 19)
	MOV_16B(16, 22)
	MOV_16B(17, 22)
	FMLA_4S(16, 6, 21)
	FMLA_4S(17, 7, 21)
	FMLA_4S(16, 12, 14)
	FMLA_4S(17, 13, 15)
	MOV_16B(14, 24)
	MOV_16B(15, 24)
	FMLA_4S(14, 6, 23)
	FMLA_4S(15, 7, 23)
	FADD_4S(6, 6, 25)
	FADD_4S(7, 7, 25)
	FMLA_4S(6, 12, 14)
	FMLA_4S(7, 13, 15)
	FMUL_4S(12, 12, 12)
	FMUL_4S(13, 13, 13)
	FMLA_4S(6, 12, 16)
	FMLA_4S(7, 13, 17)

	// denom2, denom3 = 1.0 + scale * expPoly into V14, V15:
	MOV_16B(14, 25)
	MOV_16B(15, 25)
	FMLA_4S(14, 6, 10)
	FMLA_4S(15, 7, 11)

	// Fast reciprocal on V2 (FRECPE + 2x FRECPS on FMA pipes) in parallel with hardware FDIV on V3:
	FRECPE_4S(6, 14)
	FDIV_4S(3, 3, 15)
	FRECPS_4S(10, 14, 6)
	FMUL_4S(6, 6, 10)
	FRECPS_4S(10, 14, 6)
	FMUL_4S(6, 6, 10)
	FMUL_4S(2, 2, 6)

	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R7)

	SUB $16, R8, R8
	CMP $16, R8
	BGE swiglu_neon_vec16_loop

swiglu_neon_vec8_check:
	CMP $8, R8
	BLT swiglu_neon_vec4_check

swiglu_neon_vec8_loop:
	VLD1.P 32(R5), [V0.S4, V1.S4]   // gate
	VLD1.P 32(R6), [V16.S4, V17.S4] // val

	// u = clamp(-gate, vMinLog, vMaxLog); numer = gate * val
	FNEG_4S(4, 0)
	FNEG_4S(5, 1)
	FMUL_4S(0, 0, 16)
	FMUL_4S(1, 1, 17)
	FMAX_4S(4, 4, 26)
	FMAX_4S(5, 5, 26)
	FMIN_4S(4, 4, 27)
	FMIN_4S(5, 5, 27)

	// Range reduction: zRound = u * vLog2E + vMagic, zFloor = zRound - vMagic
	MOV_16B(6, 29)
	MOV_16B(7, 29)
	FMLA_4S(6, 4, 28)
	FMLA_4S(7, 5, 28)
	FSUB_4S(8, 6, 29)
	FSUB_4S(9, 7, 29)

	// g = u - zFloor * vLn2Hi - zFloor * vLn2Lo
	FMLS_4S(4, 8, 30)
	FMLS_4S(5, 9, 30)
	FMLS_4S(4, 8, 31)
	FMLS_4S(5, 9, 31)

	// scale = (zRound << 23) + vBias127
	SHL23_4S(6, 6)
	SHL23_4S(7, 7)
	ADD_4S(6, 6, 25)
	ADD_4S(7, 7, 25)

	// Estrin's scheme for exp(g) across A and B:
	FMUL_4S(2, 4, 4)
	FMUL_4S(3, 5, 5)
	MOV_16B(8, 20)
	MOV_16B(9, 20)
	FMLA_4S(8, 4, 19)
	FMLA_4S(9, 5, 19)
	MOV_16B(10, 22)
	MOV_16B(11, 22)
	FMLA_4S(10, 4, 21)
	FMLA_4S(11, 5, 21)
	MOV_16B(12, 24)
	MOV_16B(13, 24)
	FMLA_4S(12, 4, 23)
	FMLA_4S(13, 5, 23)
	FADD_4S(14, 4, 25)
	FADD_4S(15, 5, 25)
	FMLA_4S(10, 2, 8)
	FMLA_4S(11, 3, 9)
	FMLA_4S(14, 2, 12)
	FMLA_4S(15, 3, 13)
	FMUL_4S(2, 2, 2)
	FMUL_4S(3, 3, 3)
	FMLA_4S(14, 2, 10)
	FMLA_4S(15, 3, 11)

	// denom = 1.0 + scale * expPoly
	MOV_16B(8, 25)
	MOV_16B(9, 25)
	FMLA_4S(8, 14, 6)
	FMLA_4S(9, 15, 7)

	// Fast reciprocal on V0 (FRECPE + 2x FRECPS on FMA pipe) in parallel with hardware FDIV on V1:
	FRECPE_4S(4, 8)
	FDIV_4S(1, 1, 9)
	FRECPS_4S(6, 8, 4)
	FMUL_4S(4, 4, 6)
	FRECPS_4S(6, 8, 4)
	FMUL_4S(4, 4, 6)
	FMUL_4S(0, 0, 4)

	VST1.P [V0.S4, V1.S4], 32(R7)

	SUB $8, R8, R8
	CMP $8, R8
	BGE swiglu_neon_vec8_loop

swiglu_neon_vec4_check:
	CMP $4, R8
	BLT swiglu_neon_next_row

	VLD1.P 16(R5), [V0.S4]
	VLD1.P 16(R6), [V16.S4]
	FNEG_4S(4, 0)
	FMUL_4S(0, 0, 16)
	FMAX_4S(4, 4, 26)
	FMIN_4S(4, 4, 27)

	MOV_16B(6, 29)
	FMLA_4S(6, 4, 28)
	FSUB_4S(8, 6, 29)
	FMLS_4S(4, 8, 30)
	FMLS_4S(4, 8, 31)

	SHL23_4S(6, 6)
	ADD_4S(6, 6, 25)

	FMUL_4S(2, 4, 4)
	MOV_16B(8, 20)
	FMLA_4S(8, 4, 19)
	MOV_16B(10, 22)
	FMLA_4S(10, 4, 21)
	MOV_16B(12, 24)
	FMLA_4S(12, 4, 23)
	FADD_4S(14, 4, 25)
	FMLA_4S(10, 2, 8)
	FMLA_4S(14, 2, 12)
	FMUL_4S(2, 2, 2)
	FMLA_4S(14, 2, 10)

	MOV_16B(8, 25)
	FMLA_4S(8, 14, 6)
	FRECPE_4S(4, 8)
	FRECPS_4S(6, 8, 4)
	FMUL_4S(4, 4, 6)
	FRECPS_4S(6, 8, 4)
	FMUL_4S(4, 4, 6)
	FMUL_4S(0, 0, 4)
	VST1.P [V0.S4], 16(R7)

swiglu_neon_next_row:
	ADD R10, R0, R0
	ADD R9, R1, R1
	SUB $1, R2, R2
	CBNZ R2, swiglu_neon_row_loop

swiglu_neon_done:
	RET

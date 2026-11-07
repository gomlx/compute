// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// ARM64 AdvSIMD (.4S / .16B) instruction macros (destination Vd first: Vd = Vn op Vm):
#define FADD_4S(Vd, Vn, Vm) WORD $(0x4E20D400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FSUB_4S(Vd, Vn, Vm) WORD $(0x4EA0D400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMUL_4S(Vd, Vn, Vm) WORD $(0x6E20DC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FDIV_4S(Vd, Vn, Vm) WORD $(0x6E20FC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMLA_4S(Vd, Vn, Vm) WORD $(0x4E20CC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMLS_4S(Vd, Vn, Vm) WORD $(0x4EA0CC00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMAX_4S(Vd, Vn, Vm) WORD $(0x4E20F400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMIN_4S(Vd, Vn, Vm) WORD $(0x4EA0F400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FABS_4S(Vd, Vn)     WORD $(0x4EA0F800 | ((Vn)<<5) | (Vd))
#define FNEG_4S(Vd, Vn)     WORD $(0x6EA0F800 | ((Vn)<<5) | (Vd))
#define SHL23_4S(Vd, Vn)    WORD $(0x4F375400 | ((Vn)<<5) | (Vd))
#define ADD_4S(Vd, Vn, Vm)  WORD $(0x4EA08400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define AND_16B(Vd, Vn, Vm) WORD $(0x4E201C00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define EOR_16B(Vd, Vn, Vm) WORD $(0x6E201C00 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define MOV_16B(Vd, Vn)     WORD $(0x4EA01C00 | ((Vn)<<16) | ((Vn)<<5) | (Vd))
#define DUP_4S_W(Vd, Rn)    WORD $(0x4E040C00 | ((Rn)<<5) | (Vd))

// ===========================================================================
// geluExactFloat32NEONAsm: Exact GELU for float32 using ARM64 NEON
//   GELU(x) = 0.5 * x * (1 + erf(x * rsqrt(2)))
//   Uses Estrin's parallel polynomial trees and overlaps FDIV with exp(-x^2).
//   n must be a positive multiple of 4.
// ===========================================================================
// func geluExactFloat32NEONAsm(in, out unsafe.Pointer, n int)
TEXT ·geluExactFloat32NEONAsm(SB), NOSPLIT, $0-24
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD n+16(FP), R2

	CMP $4, R2
	BLT gelu_exact_done

	// Broadcast all 20 constants into V11..V30 once outside the loop:
	MOVD $0x39506967, R3; DUP_4S_W(11, 3) // V11: vP7 = 1.9875691500e-4
	MOVD $0x3ab743ce, R3; DUP_4S_W(12, 3) // V12: vP6 = 1.3981999507e-3
	MOVD $0x3c088895, R3; DUP_4S_W(13, 3) // V13: vP5 = 8.3334519073e-3
	MOVD $0x3d2aa9a5, R3; DUP_4S_W(14, 3) // V14: vP4 = 4.1665795894e-2
	MOVD $0x3e2aaaab, R3; DUP_4S_W(15, 3) // V15: vP3 = 1.6666665459e-1
	MOVD $0x3f000000, R3; DUP_4S_W(16, 3) // V16: vHalf / vP2 = 0.5
	MOVD $0x3f800000, R3; DUP_4S_W(17, 3) // V17: vOne / vBias127 = 1.0 (127 << 23)
	MOVD $0x3f3504f3, R3; DUP_4S_W(18, 3) // V18: vRsqrt2 = 0.70710678
	MOVD $0x80000000, R3; DUP_4S_W(19, 3) // V19: vSignMask = 0x80000000
	MOVD $0x3ea7ba05, R3; DUP_4S_W(20, 3) // V20: vP = 0.3275911
	MOVD $0x3e827906, R3; DUP_4S_W(21, 3) // V21: vA1 = 0.254829592
	MOVD $0xbe91a98e, R3; DUP_4S_W(22, 3) // V22: vA2 = -0.284496736
	MOVD $0x3fb5f0e3, R3; DUP_4S_W(23, 3) // V23: vA3 = 1.421413741
	MOVD $0xbfb9fffb, R3; DUP_4S_W(24, 3) // V24: vA4 = -1.453152027
	MOVD $0x3f87dc22, R3; DUP_4S_W(25, 3) // V25: vA5 = 1.061405429
	MOVD $0xc2aeac50, R3; DUP_4S_W(26, 3) // V26: vMinLog = -87.336544
	MOVD $0x3fb8aa3b, R3; DUP_4S_W(27, 3) // V27: vLog2E = 1.44269504
	MOVD $0x4b400000, R3; DUP_4S_W(28, 3) // V28: vMagic = 12582912.0
	MOVD $0x3f318000, R3; DUP_4S_W(29, 3) // V29: vLn2Hi = 0.693359375
	MOVD $0xb95e8083, R3; DUP_4S_W(30, 3) // V30: vLn2Lo = -2.12194440e-4

gelu_exact_vec4_loop:
	VLD1.P 16(R0), [V0.S4]

	// x = v * rsqrt2, absX = |x|
	FMUL_4S(1, 0, 18)
	FABS_4S(2, 1)

	// Issue t = 1.0 / (1.0 + p * absX) early so FDIV overlaps with exp(-x^2):
	MOV_16B(3, 17)
	FMLA_4S(3, 2, 20)
	FDIV_4S(3, 17, 3)

	// negX2 = max(-(absX * absX), vMinLog)
	FMUL_4S(4, 2, 2)
	FNEG_4S(4, 4)
	FMAX_4S(4, 4, 26)

	// zRound = xClamped * log2e + magic, zFloor = zRound - magic
	MOV_16B(5, 28)
	FMLA_4S(5, 4, 27)
	FSUB_4S(6, 5, 28)

	// g = xClamped - zFloor * ln2Hi - zFloor * ln2Lo
	FMLS_4S(4, 6, 29)
	FMLS_4S(4, 6, 30)

	// scale = (zRound << 23) + vBias127
	SHL23_4S(5, 5)
	ADD_4S(5, 5, 17)

	// Estrin's scheme for exp(g):
	FMUL_4S(6, 4, 4)     // V6: g2 = g * g
	MOV_16B(7, 12)
	FMLA_4S(7, 4, 11)    // V7: q67 = vP6 + g * vP7
	MOV_16B(8, 14)
	FMLA_4S(8, 4, 13)    // V8: q45 = vP4 + g * vP5
	MOV_16B(9, 16)
	FMLA_4S(9, 4, 15)    // V9: q23 = vP2 + g * vP3
	FADD_4S(10, 4, 17)   // V10: q01 = g + 1.0
	FMUL_4S(31, 6, 6)    // V31: g4 = g2 * g2
	FMLA_4S(8, 6, 7)     // V8: q47 = q45 + g2 * q67
	FMLA_4S(10, 6, 9)    // V10: q03 = q01 + g2 * q23
	FMLA_4S(10, 31, 8)   // V10: expPoly = q03 + g4 * q47
	FMUL_4S(5, 10, 5)    // V5: expNegX2 = expPoly * scale

	// Estrin's scheme for erfPoly(t) using V3 (t):
	FMUL_4S(6, 3, 3)     // V6: t2 = t * t
	MOV_16B(7, 24)
	FMLA_4S(7, 3, 25)    // V7: p45 = vA4 + t * vA5
	MOV_16B(8, 22)
	FMLA_4S(8, 3, 23)    // V8: p23 = vA2 + t * vA3
	FMLA_4S(8, 6, 7)     // V8: p25 = p23 + t2 * p45
	MOV_16B(9, 21)
	FMLA_4S(9, 3, 8)     // V9: p15 = vA1 + t * p25
	FMUL_4S(9, 9, 3)     // V9: erfPoly = p15 * t

	// res = 1.0 - erfPoly * expNegX2
	MOV_16B(4, 17)
	FMLS_4S(4, 9, 5)

	// Restore sign of x and compute 0.5 * v * (1.0 + erfVal)
	AND_16B(1, 1, 19)
	EOR_16B(4, 4, 1)
	FADD_4S(4, 4, 17)
	FMUL_4S(0, 0, 16)
	FMUL_4S(0, 0, 4)

	VST1.P [V0.S4], 16(R1)

	SUB $4, R2, R2
	CMP $4, R2
	BGE gelu_exact_vec4_loop

gelu_exact_done:
	RET

// ===========================================================================
// geluApproxFloat32NEONAsm: Approximate (tanh) GELU for float32 using ARM64 NEON
//   GELU(x) = 0.5 * x * (1 + tanh(u)) = (x * exp(2u)) / (exp(2u) + 1)
//   where u = sqrt(2/pi) * (x + 0.044715 * x^3).
//   Uses 2-vector (8 float32) unrolling and Estrin's parallel polynomial trees.
//   n must be a positive multiple of 4.
// ===========================================================================
// func geluApproxFloat32NEONAsm(in, out unsafe.Pointer, n int)
TEXT ·geluApproxFloat32NEONAsm(SB), NOSPLIT, $0-24
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD n+16(FP), R2

	CMP $4, R2
	BLT gelu_approx_done

	// Broadcast all 16 constants into V16..V31 once outside the loop:
	MOVD $0x3f000000, R3; DUP_4S_W(16, 3) // V16: vHalf / vP2 = 0.5
	MOVD $0x3f800000, R3; DUP_4S_W(17, 3) // V17: vOne / vBias127 = 1.0 (127 << 23)
	MOVD $0x40000000, R3; DUP_4S_W(18, 3) // V18: vTwo = 2.0
	MOVD $0x41100000, R3; DUP_4S_W(19, 3) // V19: vNine = 9.0
	MOVD $0xc1100000, R3; DUP_4S_W(20, 3) // V20: vNegNine = -9.0
	MOVD $0x3f4c422a, R3; DUP_4S_W(21, 3) // V21: vSqrt2OverPi = 0.7978845608
	MOVD $0x3d372713, R3; DUP_4S_W(22, 3) // V22: vCoeff = 0.044715
	MOVD $0x3fb8aa3b, R3; DUP_4S_W(23, 3) // V23: vLog2E = 1.44269504
	MOVD $0x4b400000, R3; DUP_4S_W(24, 3) // V24: vMagic = 12582912.0
	MOVD $0x3f318000, R3; DUP_4S_W(25, 3) // V25: vLn2Hi = 0.693359375
	MOVD $0xb95e8083, R3; DUP_4S_W(26, 3) // V26: vLn2Lo = -2.12194440e-4
	MOVD $0x39506967, R3; DUP_4S_W(27, 3) // V27: vP7 = 1.9875691500e-4
	MOVD $0x3ab743ce, R3; DUP_4S_W(28, 3) // V28: vP6 = 1.3981999507e-3
	MOVD $0x3c088895, R3; DUP_4S_W(29, 3) // V29: vP5 = 8.3334519073e-3
	MOVD $0x3d2aa9a5, R3; DUP_4S_W(30, 3) // V30: vP4 = 4.1665795894e-2
	MOVD $0x3e2aaaab, R3; DUP_4S_W(31, 3) // V31: vP3 = 1.6666665459e-1

	CMP $8, R2
	BLT gelu_approx_vec4_check

gelu_approx_vec8_loop:
	VLD1.P 32(R0), [V0.S4, V1.S4]

	// x3 = v * v * v
	FMUL_4S(2, 0, 0)
	FMUL_4S(3, 1, 1)
	FMUL_4S(2, 2, 0)
	FMUL_4S(3, 3, 1)

	// inner = vSqrt2OverPi * (v + vCoeff * x3)
	MOV_16B(4, 0)
	MOV_16B(5, 1)
	FMLA_4S(4, 2, 22)
	FMLA_4S(5, 3, 22)
	FMUL_4S(4, 4, 21)
	FMUL_4S(5, 5, 21)

	// Clamp inner to [-9.0, 9.0] and multiply by 2.0
	FMIN_4S(4, 4, 19)
	FMIN_4S(5, 5, 19)
	FMAX_4S(4, 4, 20)
	FMAX_4S(5, 5, 20)
	FMUL_4S(4, 4, 18)
	FMUL_4S(5, 5, 18)

	// Range reduction: zRound = twoX * log2e + magic, zFloor = zRound - magic
	MOV_16B(6, 24)
	MOV_16B(7, 24)
	FMLA_4S(6, 4, 23)
	FMLA_4S(7, 5, 23)
	FSUB_4S(8, 6, 24)
	FSUB_4S(9, 7, 24)

	// g = twoX - zFloor * ln2Hi - zFloor * ln2Lo
	FMLS_4S(4, 8, 25)
	FMLS_4S(5, 9, 25)
	FMLS_4S(4, 8, 26)
	FMLS_4S(5, 9, 26)

	// scale = (zRound << 23) + vBias127
	SHL23_4S(6, 6)
	SHL23_4S(7, 7)
	ADD_4S(6, 6, 17)
	ADD_4S(7, 7, 17)

	// Estrin's scheme for exp(g) across A and B:
	FMUL_4S(2, 4, 4)     // g2A
	FMUL_4S(3, 5, 5)     // g2B
	MOV_16B(8, 28)
	MOV_16B(9, 28)
	FMLA_4S(8, 4, 27)    // q67A
	FMLA_4S(9, 5, 27)    // q67B
	MOV_16B(10, 30)
	MOV_16B(11, 30)
	FMLA_4S(10, 4, 29)   // q45A
	FMLA_4S(11, 5, 29)   // q45B
	MOV_16B(12, 16)
	MOV_16B(13, 16)
	FMLA_4S(12, 4, 31)   // q23A
	FMLA_4S(13, 5, 31)   // q23B
	FADD_4S(14, 4, 17)   // q01A
	FADD_4S(15, 5, 17)   // q01B
	FMLA_4S(10, 2, 8)    // q47A = q45A + g2A * q67A
	FMLA_4S(11, 3, 9)    // q47B = q45B + g2B * q67B
	FMLA_4S(14, 2, 12)   // q03A = q01A + g2A * q23A
	FMLA_4S(15, 3, 13)   // q03B = q01B + g2B * q23B
	FMUL_4S(2, 2, 2)     // g4A
	FMUL_4S(3, 3, 3)     // g4B
	FMLA_4S(14, 2, 10)   // expPolyA = q03A + g4A * q47A
	FMLA_4S(15, 3, 11)   // expPolyB = q03B + g4B * q47B
	FMUL_4S(6, 14, 6)    // exp2xA = expPolyA * scaleA
	FMUL_4S(7, 15, 7)    // exp2xB = expPolyB * scaleB

	// out = (v * exp2x) / (exp2x + 1.0)
	FADD_4S(8, 6, 17)
	FADD_4S(9, 7, 17)
	FMUL_4S(0, 0, 6)
	FMUL_4S(1, 1, 7)
	FDIV_4S(0, 0, 8)
	FDIV_4S(1, 1, 9)

	VST1.P [V0.S4, V1.S4], 32(R1)

	SUB $8, R2, R2
	CMP $8, R2
	BGE gelu_approx_vec8_loop

gelu_approx_vec4_check:
	CMP $4, R2
	BLT gelu_approx_done

	VLD1.P 16(R0), [V0.S4]
	FMUL_4S(2, 0, 0)
	FMUL_4S(2, 2, 0)
	MOV_16B(4, 0)
	FMLA_4S(4, 2, 22)
	FMUL_4S(4, 4, 21)
	FMIN_4S(4, 4, 19)
	FMAX_4S(4, 4, 20)
	FMUL_4S(4, 4, 18)
	MOV_16B(6, 24)
	FMLA_4S(6, 4, 23)
	FSUB_4S(8, 6, 24)
	FMLS_4S(4, 8, 25)
	FMLS_4S(4, 8, 26)
	SHL23_4S(6, 6)
	ADD_4S(6, 6, 17)
	FMUL_4S(2, 4, 4)
	MOV_16B(8, 28)
	FMLA_4S(8, 4, 27)
	MOV_16B(10, 30)
	FMLA_4S(10, 4, 29)
	MOV_16B(12, 16)
	FMLA_4S(12, 4, 31)
	FADD_4S(14, 4, 17)
	FMLA_4S(10, 2, 8)
	FMLA_4S(14, 2, 12)
	FMUL_4S(2, 2, 2)
	FMLA_4S(14, 2, 10)
	FMUL_4S(6, 14, 6)
	FADD_4S(8, 6, 17)
	FMUL_4S(0, 0, 6)
	FDIV_4S(0, 0, 8)
	VST1.P [V0.S4], 16(R1)

gelu_approx_done:
	RET

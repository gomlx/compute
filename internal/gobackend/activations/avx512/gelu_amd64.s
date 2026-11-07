// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// RODATA constants for AVX-512 GELU (Exact and Approximate)
// Each entry is a 4-byte float32 or uint32 bit-pattern, broadcast via VBROADCASTSS.
GLOBL ·geluAVX512Consts(SB), RODATA|NOPTR, $104
DATA ·geluAVX512Consts+0(SB)/4, $0x39506967  //  0: vP7 = 1.9875691500e-4
DATA ·geluAVX512Consts+4(SB)/4, $0x3ab743ce  //  4: vP6 = 1.3981999507e-3
DATA ·geluAVX512Consts+8(SB)/4, $0x3c088895  //  8: vP5 = 8.3334519073e-3
DATA ·geluAVX512Consts+12(SB)/4, $0x3d2aa9a5 // 12: vP4 = 4.1665795894e-2
DATA ·geluAVX512Consts+16(SB)/4, $0x3e2aaaab // 16: vP3 = 1.6666665459e-1
DATA ·geluAVX512Consts+20(SB)/4, $0x3f000000 // 20: vHalf / vP2 = 0.5
DATA ·geluAVX512Consts+24(SB)/4, $0x3f800000 // 24: vOne / vBias127 = 1.0 (127 << 23)
DATA ·geluAVX512Consts+28(SB)/4, $0x3f3504f3 // 28: vRsqrt2 = 0.70710678
DATA ·geluAVX512Consts+32(SB)/4, $0x80000000 // 32: vSignMask = 0x80000000
DATA ·geluAVX512Consts+36(SB)/4, $0x3ea7ba05 // 36: vP = 0.3275911
DATA ·geluAVX512Consts+40(SB)/4, $0x3e827906 // 40: vA1 = 0.254829592
DATA ·geluAVX512Consts+44(SB)/4, $0xbe91a98e // 44: vA2 = -0.284496736
DATA ·geluAVX512Consts+48(SB)/4, $0x3fb5f0e3 // 48: vA3 = 1.421413741
DATA ·geluAVX512Consts+52(SB)/4, $0xbfb9fffb // 52: vA4 = -1.453152027
DATA ·geluAVX512Consts+56(SB)/4, $0x3f87dc22 // 56: vA5 = 1.061405429
DATA ·geluAVX512Consts+60(SB)/4, $0xc2aeac50 // 60: vMinLog = -87.336544
DATA ·geluAVX512Consts+64(SB)/4, $0x3fb8aa3b // 64: vLog2E = 1.44269504
DATA ·geluAVX512Consts+68(SB)/4, $0x4b400000 // 68: vMagic = 12582912.0
DATA ·geluAVX512Consts+72(SB)/4, $0x3f318000 // 72: vLn2Hi = 0.693359375
DATA ·geluAVX512Consts+76(SB)/4, $0xb95e8083 // 76: vLn2Lo = -2.12194440e-4
DATA ·geluAVX512Consts+80(SB)/4, $0x40000000 // 80: vTwo = 2.0
DATA ·geluAVX512Consts+84(SB)/4, $0x41100000 // 84: vNine = 9.0
DATA ·geluAVX512Consts+88(SB)/4, $0xc1100000 // 88: vNegNine = -9.0
DATA ·geluAVX512Consts+92(SB)/4, $0x3f4c422a // 92: vSqrt2OverPi = 0.7978845608
DATA ·geluAVX512Consts+96(SB)/4, $0x3d372713 // 96: vCoeff = 0.044715
DATA ·geluAVX512Consts+100(SB)/4, $0x7fffffff // 100: vAbsMask = 0x7fffffff

// ===========================================================================
// geluExactFloat32AVX512Asm: Exact GELU for float32 using AVX-512
//   GELU(x) = 0.5 * x * (1 + erf(x * rsqrt(2)))
//   Uses Estrin's parallel polynomial trees and overlaps VDIVPS with exp(-x^2).
//   n must be a positive multiple of 16.
// ===========================================================================
// func geluExactFloat32AVX512Asm(in, out unsafe.Pointer, n int)
TEXT ·geluExactFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ n+16(FP), DX

	CMPQ DX, $16
	JL gelu_exact_512_done

	// Pin all 21 constants in Z11..Z31 once outside the loop:
	VBROADCASTSS ·geluAVX512Consts+0(SB), Z11   // Z11: vP7
	VBROADCASTSS ·geluAVX512Consts+4(SB), Z12   // Z12: vP6
	VBROADCASTSS ·geluAVX512Consts+8(SB), Z13   // Z13: vP5
	VBROADCASTSS ·geluAVX512Consts+12(SB), Z14  // Z14: vP4
	VBROADCASTSS ·geluAVX512Consts+16(SB), Z15  // Z15: vP3
	VBROADCASTSS ·geluAVX512Consts+20(SB), Z16  // Z16: vHalf / vP2 = 0.5
	VBROADCASTSS ·geluAVX512Consts+24(SB), Z17  // Z17: vOne / vBias127 = 1.0
	VBROADCASTSS ·geluAVX512Consts+28(SB), Z18  // Z18: vRsqrt2 = 0.70710678
	VBROADCASTSS ·geluAVX512Consts+32(SB), Z19  // Z19: vSignMask = 0x80000000
	VBROADCASTSS ·geluAVX512Consts+36(SB), Z20  // Z20: vP = 0.3275911
	VBROADCASTSS ·geluAVX512Consts+40(SB), Z21  // Z21: vA1 = 0.254829592
	VBROADCASTSS ·geluAVX512Consts+44(SB), Z22  // Z22: vA2 = -0.284496736
	VBROADCASTSS ·geluAVX512Consts+48(SB), Z23  // Z23: vA3 = 1.421413741
	VBROADCASTSS ·geluAVX512Consts+52(SB), Z24  // Z24: vA4 = -1.453152027
	VBROADCASTSS ·geluAVX512Consts+56(SB), Z25  // Z25: vA5 = 1.061405429
	VBROADCASTSS ·geluAVX512Consts+60(SB), Z26  // Z26: vMinLog = -87.336544
	VBROADCASTSS ·geluAVX512Consts+64(SB), Z27  // Z27: vLog2E = 1.44269504
	VBROADCASTSS ·geluAVX512Consts+68(SB), Z28  // Z28: vMagic = 12582912.0
	VBROADCASTSS ·geluAVX512Consts+72(SB), Z29  // Z29: vLn2Hi = 0.693359375
	VBROADCASTSS ·geluAVX512Consts+76(SB), Z30  // Z30: vLn2Lo = -2.12194440e-4
	VBROADCASTSS ·geluAVX512Consts+100(SB), Z31 // Z31: vAbsMask = 0x7fffffff

	CMPQ DX, $32
	JL gelu_exact_512_vec16_check

gelu_exact_512_loop32:
	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1

	// x = v * rsqrt2, absX = |x|
	VMULPS Z18, Z0, Z2
	VMULPS Z18, Z1, Z3
	VPANDD Z31, Z2, Z4
	VPANDD Z31, Z3, Z5

	// Issue t = 1.0 / (1.0 + p * absX) early so VDIVPS overlaps with exp(-x^2):
	VMOVAPS Z17, Z6
	VMOVAPS Z17, Z7
	VFMADD231PS Z4, Z20, Z6
	VFMADD231PS Z5, Z20, Z7
	VDIVPS Z6, Z17, Z6
	VDIVPS Z7, Z17, Z7

	// negX2 = max(-(absX * absX), vMinLog)
	VMULPS Z4, Z4, Z4
	VMULPS Z5, Z5, Z5
	VPXORD Z19, Z4, Z4
	VPXORD Z19, Z5, Z5
	VMAXPS Z26, Z4, Z4
	VMAXPS Z26, Z5, Z5

	// zRound = xClamped * log2e + magic, zFloor = zRound - magic
	VMOVAPS Z28, Z8
	VMOVAPS Z28, Z9
	VFMADD231PS Z4, Z27, Z8
	VFMADD231PS Z5, Z27, Z9
	VSUBPS Z28, Z8, Z10
	VSUBPS Z28, Z9, Z0

	// g = xClamped - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS Z10, Z29, Z4
	VFNMADD231PS Z0, Z29, Z5
	VFNMADD231PS Z10, Z30, Z4
	VFNMADD231PS Z0, Z30, Z5

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Z8, Z8
	VPSLLD $23, Z9, Z9
	VPADDD Z17, Z8, Z8
	VPADDD Z17, Z9, Z9

	// Estrin's scheme for exp(g) across A and B:
	// First compute q67 and q45 for A and B:
	VMOVAPS Z12, Z10
	VMOVAPS Z12, Z0
	VFMADD231PS Z4, Z11, Z10  // Z10: q67A
	VFMADD231PS Z5, Z11, Z0   // Z0: q67B
	VMOVAPS Z14, Z1
	VFMADD231PS Z4, Z13, Z1   // Z1: q45A
	VMULPS Z4, Z4, Z2         // Z2: g2A
	VFMADD231PS Z2, Z10, Z1   // Z1: q47A = q45A + g2A * q67A
	VMOVAPS Z14, Z10
	VFMADD231PS Z5, Z13, Z10  // Z10: q45B
	VMULPS Z5, Z5, Z3         // Z3: g2B
	VFMADD231PS Z3, Z0, Z10   // Z10: q47B = q45B + g2B * q67B

	// Compute q23 and q01 for A and B:
	VMOVAPS Z16, Z0
	VFMADD231PS Z4, Z15, Z0   // Z0: q23A
	VADDPS Z17, Z4, Z4        // Z4: q01A
	VFMADD231PS Z2, Z0, Z4    // Z4: q03A = q01A + g2A * q23A
	VMOVAPS Z16, Z0
	VFMADD231PS Z5, Z15, Z0   // Z0: q23B
	VADDPS Z17, Z5, Z5        // Z5: q01B
	VFMADD231PS Z3, Z0, Z5    // Z5: q03B = q01B + g2B * q23B

	// Combine with g4 and multiply by scale:
	VMULPS Z2, Z2, Z2         // Z2: g4A
	VMULPS Z3, Z3, Z3         // Z3: g4B
	VFMADD231PS Z2, Z1, Z4    // Z4: expPolyA = q03A + g4A * q47A
	VFMADD231PS Z3, Z10, Z5   // Z5: expPolyB = q03B + g4B * q47B
	VMULPS Z8, Z4, Z8         // Z8: expNegX2A = expPolyA * scaleA
	VMULPS Z9, Z5, Z9         // Z9: expNegX2B = expPolyB * scaleB

	// Estrin's scheme for erfPoly(t) using Z6 (tA) and Z7 (tB):
	VMULPS Z6, Z6, Z2         // Z2: t2A
	VMULPS Z7, Z7, Z3         // Z3: t2B
	VMOVAPS Z24, Z4
	VMOVAPS Z24, Z5
	VFMADD231PS Z6, Z25, Z4   // Z4: p45A = vA4 + tA * vA5
	VFMADD231PS Z7, Z25, Z5   // Z5: p45B = vA4 + tB * vA5
	VMOVAPS Z22, Z1
	VMOVAPS Z22, Z10
	VFMADD231PS Z6, Z23, Z1   // Z1: p23A = vA2 + tA * vA3
	VFMADD231PS Z7, Z23, Z10  // Z10: p23B = vA2 + tB * vA3
	VFMADD231PS Z2, Z4, Z1    // Z1: p25A = p23A + t2A * p45A
	VFMADD231PS Z3, Z5, Z10   // Z10: p25B = p23B + t2B * p45B
	VMOVAPS Z21, Z4
	VMOVAPS Z21, Z5
	VFMADD231PS Z6, Z1, Z4    // Z4: p15A = vA1 + tA * p25A
	VFMADD231PS Z7, Z10, Z5   // Z5: p15B = vA1 + tB * p25B
	VMULPS Z6, Z4, Z4         // Z4: erfPolyA = p15A * tA
	VMULPS Z7, Z5, Z5         // Z5: erfPolyB = p15B * tB

	// res = 1.0 - erfPoly * expNegX2
	VMOVAPS Z17, Z6
	VMOVAPS Z17, Z7
	VFNMADD231PS Z4, Z8, Z6
	VFNMADD231PS Z5, Z9, Z7

	// Reload vA and vB from L1 cache, restore sign of v, and compute 0.5 * v * (1.0 + erfVal)
	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1
	VPANDD Z19, Z0, Z2
	VPANDD Z19, Z1, Z3
	VPXORD Z2, Z6, Z6
	VPXORD Z3, Z7, Z7
	VADDPS Z17, Z6, Z6
	VADDPS Z17, Z7, Z7
	VMULPS Z16, Z0, Z0
	VMULPS Z16, Z1, Z1
	VMULPS Z6, Z0, Z0
	VMULPS Z7, Z1, Z1

	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)

	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, DX
	CMPQ DX, $32
	JGE gelu_exact_512_loop32

gelu_exact_512_vec16_check:
	CMPQ DX, $16
	JL gelu_exact_512_done

gelu_exact_512_loop16:
	VMOVUPS (SI), Z0

	// x = v * rsqrt2, absX = |x|
	VMULPS Z18, Z0, Z1
	VPANDD Z31, Z1, Z2

	// Issue t = 1.0 / (1.0 + p * absX) early so VDIVPS overlaps with exp(-x^2):
	VMOVAPS Z17, Z3
	VFMADD231PS Z2, Z20, Z3
	VDIVPS Z3, Z17, Z3

	// negX2 = max(-(absX * absX), vMinLog)
	VMULPS Z2, Z2, Z4
	VPXORD Z19, Z4, Z4
	VMAXPS Z26, Z4, Z4

	// zRound = xClamped * log2e + magic, zFloor = zRound - magic
	VMOVAPS Z28, Z5
	VFMADD231PS Z4, Z27, Z5
	VSUBPS Z28, Z5, Z6

	// g = xClamped - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS Z6, Z29, Z4
	VFNMADD231PS Z6, Z30, Z4

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Z5, Z5
	VPADDD Z17, Z5, Z5

	// Estrin's scheme for exp(g):
	VMULPS Z4, Z4, Z6         // Z6: g2 = g * g
	VMOVAPS Z12, Z7
	VFMADD231PS Z4, Z11, Z7   // Z7: q67 = vP6 + g * vP7
	VMOVAPS Z14, Z8
	VFMADD231PS Z4, Z13, Z8   // Z8: q45 = vP4 + g * vP5
	VMOVAPS Z16, Z9
	VFMADD231PS Z4, Z15, Z9   // Z9: q23 = vP2 + g * vP3
	VADDPS Z17, Z4, Z10       // Z10: q01 = g + 1.0
	VFMADD231PS Z6, Z7, Z8    // Z8: q47 = q45 + g2 * q67
	VFMADD231PS Z6, Z9, Z10   // Z10: q03 = q01 + g2 * q23
	VMULPS Z6, Z6, Z6         // Z6: g4 = g2 * g2
	VFMADD231PS Z6, Z8, Z10   // Z10: expPoly = q03 + g4 * q47
	VMULPS Z5, Z10, Z5        // Z5: expNegX2 = expPoly * scale

	// Estrin's scheme for erfPoly(t) using Z3 (t):
	VMULPS Z3, Z3, Z6         // Z6: t2 = t * t
	VMOVAPS Z24, Z7
	VFMADD231PS Z3, Z25, Z7   // Z7: p45 = vA4 + t * vA5
	VMOVAPS Z22, Z8
	VFMADD231PS Z3, Z23, Z8   // Z8: p23 = vA2 + t * vA3
	VFMADD231PS Z6, Z7, Z8    // Z8: p25 = p23 + t2 * p45
	VMOVAPS Z21, Z9
	VFMADD231PS Z3, Z8, Z9    // Z9: p15 = vA1 + t * p25
	VMULPS Z3, Z9, Z9         // Z9: erfPoly = p15 * t

	// res = 1.0 - erfPoly * expNegX2
	VMOVAPS Z17, Z4
	VFNMADD231PS Z9, Z5, Z4

	// Restore sign of x and compute 0.5 * v * (1.0 + erfVal)
	VPANDD Z19, Z1, Z1
	VPXORD Z1, Z4, Z4
	VADDPS Z17, Z4, Z4
	VMULPS Z16, Z0, Z0
	VMULPS Z4, Z0, Z0

	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE gelu_exact_512_loop16

gelu_exact_512_done:
	VZEROUPPER
	RET

// ===========================================================================
// geluApproxFloat32AVX512Asm: Approximate (tanh) GELU for float32 using AVX-512
//   GELU(x) = 0.5 * x * (1 + tanh(u)) = (x * exp(2u)) / (exp(2u) + 1)
//   where u = sqrt(2/pi) * (x + 0.044715 * x^3).
//   Uses 2-vector (32 float32) unrolling and Estrin's parallel polynomial trees.
//   n must be a positive multiple of 16.
// ===========================================================================
// func geluApproxFloat32AVX512Asm(in, out unsafe.Pointer, n int)
TEXT ·geluApproxFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ n+16(FP), DX

	CMPQ DX, $16
	JL gelu_approx_512_done

	// Pin all 16 constants in Z16..Z31 once outside the loop:
	VBROADCASTSS ·geluAVX512Consts+20(SB), Z16 // Z16: vHalf / vP2 = 0.5
	VBROADCASTSS ·geluAVX512Consts+24(SB), Z17 // Z17: vOne / vBias127 = 1.0
	VBROADCASTSS ·geluAVX512Consts+80(SB), Z18 // Z18: vTwo = 2.0
	VBROADCASTSS ·geluAVX512Consts+84(SB), Z19 // Z19: vNine = 9.0
	VBROADCASTSS ·geluAVX512Consts+88(SB), Z20 // Z20: vNegNine = -9.0
	VBROADCASTSS ·geluAVX512Consts+92(SB), Z21 // Z21: vSqrt2OverPi = 0.7978845608
	VBROADCASTSS ·geluAVX512Consts+96(SB), Z22 // Z22: vCoeff = 0.044715
	VBROADCASTSS ·geluAVX512Consts+64(SB), Z23 // Z23: vLog2E = 1.44269504
	VBROADCASTSS ·geluAVX512Consts+68(SB), Z24 // Z24: vMagic = 12582912.0
	VBROADCASTSS ·geluAVX512Consts+72(SB), Z25 // Z25: vLn2Hi = 0.693359375
	VBROADCASTSS ·geluAVX512Consts+76(SB), Z26 // Z26: vLn2Lo = -2.12194440e-4
	VBROADCASTSS ·geluAVX512Consts+0(SB), Z27  // Z27: vP7 = 1.9875691500e-4
	VBROADCASTSS ·geluAVX512Consts+4(SB), Z28  // Z28: vP6 = 1.3981999507e-3
	VBROADCASTSS ·geluAVX512Consts+8(SB), Z29  // Z29: vP5 = 8.3334519073e-3
	VBROADCASTSS ·geluAVX512Consts+12(SB), Z30 // Z30: vP4 = 4.1665795894e-2
	VBROADCASTSS ·geluAVX512Consts+16(SB), Z31 // Z31: vP3 = 1.6666665459e-1

	CMPQ DX, $32
	JL gelu_approx_512_vec16_check

gelu_approx_512_loop32:
	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1

	// x3 = v * v * v
	VMULPS Z0, Z0, Z2
	VMULPS Z1, Z1, Z3
	VMULPS Z0, Z2, Z2
	VMULPS Z1, Z3, Z3

	// inner = vSqrt2OverPi * (v + vCoeff * x3)
	VMOVAPS Z0, Z4
	VMOVAPS Z1, Z5
	VFMADD231PS Z2, Z22, Z4
	VFMADD231PS Z3, Z22, Z5
	VMULPS Z21, Z4, Z4
	VMULPS Z21, Z5, Z5

	// Clamp inner to [-9.0, 9.0] and multiply by 2.0
	VMINPS Z19, Z4, Z4
	VMINPS Z19, Z5, Z5
	VMAXPS Z20, Z4, Z4
	VMAXPS Z20, Z5, Z5
	VMULPS Z18, Z4, Z4
	VMULPS Z18, Z5, Z5

	// Range reduction: zRound = twoX * log2e + magic, zFloor = zRound - magic
	VMOVAPS Z24, Z6
	VMOVAPS Z24, Z7
	VFMADD231PS Z4, Z23, Z6
	VFMADD231PS Z5, Z23, Z7
	VSUBPS Z24, Z6, Z8
	VSUBPS Z24, Z7, Z9

	// g = twoX - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS Z8, Z25, Z4
	VFNMADD231PS Z9, Z25, Z5
	VFNMADD231PS Z8, Z26, Z4
	VFNMADD231PS Z9, Z26, Z5

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Z6, Z6
	VPSLLD $23, Z7, Z7
	VPADDD Z17, Z6, Z6
	VPADDD Z17, Z7, Z7

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Z4, Z4, Z2         // g2A
	VMULPS Z5, Z5, Z3         // g2B
	VMOVAPS Z28, Z8
	VMOVAPS Z28, Z9
	VFMADD231PS Z4, Z27, Z8   // q67A
	VFMADD231PS Z5, Z27, Z9   // q67B
	VMOVAPS Z30, Z10
	VMOVAPS Z30, Z11
	VFMADD231PS Z4, Z29, Z10  // q45A
	VFMADD231PS Z5, Z29, Z11  // q45B
	VMOVAPS Z16, Z12
	VMOVAPS Z16, Z13
	VFMADD231PS Z4, Z31, Z12  // q23A
	VFMADD231PS Z5, Z31, Z13  // q23B
	VADDPS Z17, Z4, Z14       // q01A
	VADDPS Z17, Z5, Z15       // q01B
	VFMADD231PS Z2, Z8, Z10   // q47A = q45A + g2A * q67A
	VFMADD231PS Z3, Z9, Z11   // q47B = q45B + g2B * q67B
	VFMADD231PS Z2, Z12, Z14  // q03A = q01A + g2A * q23A
	VFMADD231PS Z3, Z13, Z15  // q03B = q01B + g2B * q23B
	VMULPS Z2, Z2, Z2         // g4A
	VMULPS Z3, Z3, Z3         // g4B
	VFMADD231PS Z2, Z10, Z14  // expPolyA = q03A + g4A * q47A
	VFMADD231PS Z3, Z11, Z15  // expPolyB = q03B + g4B * q47B
	VMULPS Z6, Z14, Z6        // exp2xA = expPolyA * scaleA
	VMULPS Z7, Z15, Z7        // exp2xB = expPolyB * scaleB

	// out = (v * exp2x) / (exp2x + 1.0)
	VADDPS Z17, Z6, Z8
	VADDPS Z17, Z7, Z9
	VMULPS Z6, Z0, Z0
	VMULPS Z7, Z1, Z1
	VDIVPS Z8, Z0, Z0
	VDIVPS Z9, Z1, Z1

	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)

	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, DX
	CMPQ DX, $32
	JGE gelu_approx_512_loop32

gelu_approx_512_vec16_check:
	CMPQ DX, $16
	JL gelu_approx_512_done

	VMOVUPS (SI), Z0
	VMULPS Z0, Z0, Z2
	VMULPS Z0, Z2, Z2
	VMOVAPS Z0, Z4
	VFMADD231PS Z2, Z22, Z4
	VMULPS Z21, Z4, Z4
	VMINPS Z19, Z4, Z4
	VMAXPS Z20, Z4, Z4
	VMULPS Z18, Z4, Z4
	VMOVAPS Z24, Z6
	VFMADD231PS Z4, Z23, Z6
	VSUBPS Z24, Z6, Z8
	VFNMADD231PS Z8, Z25, Z4
	VFNMADD231PS Z8, Z26, Z4
	VPSLLD $23, Z6, Z6
	VPADDD Z17, Z6, Z6
	VMULPS Z4, Z4, Z2
	VMOVAPS Z28, Z8
	VFMADD231PS Z4, Z27, Z8
	VMOVAPS Z30, Z10
	VFMADD231PS Z4, Z29, Z10
	VMOVAPS Z16, Z12
	VFMADD231PS Z4, Z31, Z12
	VADDPS Z17, Z4, Z14
	VFMADD231PS Z2, Z8, Z10
	VFMADD231PS Z2, Z12, Z14
	VMULPS Z2, Z2, Z2
	VFMADD231PS Z2, Z10, Z14
	VMULPS Z6, Z14, Z6
	VADDPS Z17, Z6, Z8
	VMULPS Z6, Z0, Z0
	VDIVPS Z8, Z0, Z0
	VMOVUPS Z0, (DI)

gelu_approx_512_done:
	VZEROUPPER
	RET

// ===========================================================================
// geluExactBiasTileFloat32AVX512Asm: Fused Bias + Exact GELU 2D tile for float32
//   Applies out[r*rowStrideBytes + c] = GeluExact(out[r*rowStrideBytes + c] + bias[c])
//   for r in [0, numRows) and c in [0, width).
//   width must be a positive multiple of 16.
// ===========================================================================
// func geluExactBiasTileFloat32AVX512Asm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)
TEXT ·geluExactBiasTileFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), R8
	MOVQ bias+8(FP), R9
	MOVQ rowStrideBytes+16(FP), R10
	MOVQ numRows+24(FP), R11
	MOVQ width+32(FP), R12

	CMPQ R11, $0
	JLE gelu_exact_bias_512_done
	CMPQ R12, $16
	JL gelu_exact_bias_512_done

	// Pin all 21 constants in Z11..Z31 once across the entire 2D tile:
	VBROADCASTSS ·geluAVX512Consts+0(SB), Z11   // Z11: vP7
	VBROADCASTSS ·geluAVX512Consts+4(SB), Z12   // Z12: vP6
	VBROADCASTSS ·geluAVX512Consts+8(SB), Z13   // Z13: vP5
	VBROADCASTSS ·geluAVX512Consts+12(SB), Z14  // Z14: vP4
	VBROADCASTSS ·geluAVX512Consts+16(SB), Z15  // Z15: vP3
	VBROADCASTSS ·geluAVX512Consts+20(SB), Z16  // Z16: vHalf / vP2 = 0.5
	VBROADCASTSS ·geluAVX512Consts+24(SB), Z17  // Z17: vOne / vBias127 = 1.0
	VBROADCASTSS ·geluAVX512Consts+28(SB), Z18  // Z18: vRsqrt2 = 0.70710678
	VBROADCASTSS ·geluAVX512Consts+32(SB), Z19  // Z19: vSignMask = 0x80000000
	VBROADCASTSS ·geluAVX512Consts+36(SB), Z20  // Z20: vP = 0.3275911
	VBROADCASTSS ·geluAVX512Consts+40(SB), Z21  // Z21: vA1 = 0.254829592
	VBROADCASTSS ·geluAVX512Consts+44(SB), Z22  // Z22: vA2 = -0.284496736
	VBROADCASTSS ·geluAVX512Consts+48(SB), Z23  // Z23: vA3 = 1.421413741
	VBROADCASTSS ·geluAVX512Consts+52(SB), Z24  // Z24: vA4 = -1.453152027
	VBROADCASTSS ·geluAVX512Consts+56(SB), Z25  // Z25: vA5 = 1.061405429
	VBROADCASTSS ·geluAVX512Consts+60(SB), Z26  // Z26: vMinLog = -87.336544
	VBROADCASTSS ·geluAVX512Consts+64(SB), Z27  // Z27: vLog2E = 1.44269504
	VBROADCASTSS ·geluAVX512Consts+68(SB), Z28  // Z28: vMagic = 12582912.0
	VBROADCASTSS ·geluAVX512Consts+72(SB), Z29  // Z29: vLn2Hi = 0.693359375
	VBROADCASTSS ·geluAVX512Consts+76(SB), Z30  // Z30: vLn2Lo = -2.12194440e-4
	VBROADCASTSS ·geluAVX512Consts+100(SB), Z31 // Z31: vAbsMask = 0x7fffffff

gelu_exact_bias_512_row_loop:
	MOVQ R8, DI
	MOVQ R9, SI
	MOVQ R12, DX

	CMPQ DX, $32
	JL gelu_exact_bias_512_vec16_check

gelu_exact_bias_512_loop32:
	VMOVUPS 0(DI), Z0
	VMOVUPS 64(DI), Z1
	VADDPS 0(SI), Z0, Z0
	VADDPS 64(SI), Z1, Z1

	// x = v * rsqrt2, absX = |x|
	VMULPS Z18, Z0, Z2
	VMULPS Z18, Z1, Z3
	VPANDD Z31, Z2, Z4
	VPANDD Z31, Z3, Z5

	// Issue t = 1.0 / (1.0 + p * absX) early so VDIVPS overlaps with exp(-x^2):
	VMOVAPS Z17, Z6
	VMOVAPS Z17, Z7
	VFMADD231PS Z4, Z20, Z6
	VFMADD231PS Z5, Z20, Z7
	VDIVPS Z6, Z17, Z6
	VDIVPS Z7, Z17, Z7

	// negX2 = max(-(absX * absX), vMinLog)
	VMULPS Z4, Z4, Z4
	VMULPS Z5, Z5, Z5
	VPXORD Z19, Z4, Z4
	VPXORD Z19, Z5, Z5
	VMAXPS Z26, Z4, Z4
	VMAXPS Z26, Z5, Z5

	// zRound = xClamped * log2e + magic, zFloor = zRound - magic
	VMOVAPS Z28, Z8
	VMOVAPS Z28, Z9
	VFMADD231PS Z4, Z27, Z8
	VFMADD231PS Z5, Z27, Z9
	VSUBPS Z28, Z8, Z10
	VSUBPS Z28, Z9, Z0

	// g = xClamped - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS Z10, Z29, Z4
	VFNMADD231PS Z0, Z29, Z5
	VFNMADD231PS Z10, Z30, Z4
	VFNMADD231PS Z0, Z30, Z5

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Z8, Z8
	VPSLLD $23, Z9, Z9
	VPADDD Z17, Z8, Z8
	VPADDD Z17, Z9, Z9

	// Estrin's scheme for exp(g) across A and B:
	VMOVAPS Z12, Z10
	VMOVAPS Z12, Z0
	VFMADD231PS Z4, Z11, Z10
	VFMADD231PS Z5, Z11, Z0
	VMOVAPS Z14, Z1
	VFMADD231PS Z4, Z13, Z1
	VMULPS Z4, Z4, Z2
	VFMADD231PS Z2, Z10, Z1
	VMOVAPS Z14, Z10
	VFMADD231PS Z5, Z13, Z10
	VMULPS Z5, Z5, Z3
	VFMADD231PS Z3, Z0, Z10

	VMOVAPS Z16, Z0
	VFMADD231PS Z4, Z15, Z0
	VADDPS Z17, Z4, Z4
	VFMADD231PS Z2, Z0, Z4
	VMOVAPS Z16, Z0
	VFMADD231PS Z5, Z15, Z0
	VADDPS Z17, Z5, Z5
	VFMADD231PS Z3, Z0, Z5

	VMULPS Z2, Z2, Z2
	VMULPS Z3, Z3, Z3
	VFMADD231PS Z2, Z1, Z4
	VFMADD231PS Z3, Z10, Z5
	VMULPS Z8, Z4, Z8
	VMULPS Z9, Z5, Z9

	// Estrin's scheme for erfPoly(t) using Z6 (tA) and Z7 (tB):
	VMULPS Z6, Z6, Z2
	VMULPS Z7, Z7, Z3
	VMOVAPS Z24, Z4
	VMOVAPS Z24, Z5
	VFMADD231PS Z6, Z25, Z4
	VFMADD231PS Z7, Z25, Z5
	VMOVAPS Z22, Z1
	VMOVAPS Z22, Z10
	VFMADD231PS Z6, Z23, Z1
	VFMADD231PS Z7, Z23, Z10
	VFMADD231PS Z2, Z4, Z1
	VFMADD231PS Z3, Z5, Z10
	VMOVAPS Z21, Z4
	VMOVAPS Z21, Z5
	VFMADD231PS Z6, Z1, Z4
	VFMADD231PS Z7, Z10, Z5
	VMULPS Z6, Z4, Z4
	VMULPS Z7, Z5, Z5

	// res = 1.0 - erfPoly * expNegX2
	VMOVAPS Z17, Z6
	VMOVAPS Z17, Z7
	VFNMADD231PS Z4, Z8, Z6
	VFNMADD231PS Z5, Z9, Z7

	// Recompute vA = out + bias from L1 cache, restore sign of v, and compute 0.5 * v * (1.0 + erfVal)
	VMOVUPS 0(DI), Z0
	VMOVUPS 64(DI), Z1
	VADDPS 0(SI), Z0, Z0
	VADDPS 64(SI), Z1, Z1
	VPANDD Z19, Z0, Z2
	VPANDD Z19, Z1, Z3
	VPXORD Z2, Z6, Z6
	VPXORD Z3, Z7, Z7
	VADDPS Z17, Z6, Z6
	VADDPS Z17, Z7, Z7
	VMULPS Z16, Z0, Z0
	VMULPS Z16, Z1, Z1
	VMULPS Z6, Z0, Z0
	VMULPS Z7, Z1, Z1

	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)

	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, DX
	CMPQ DX, $32
	JGE gelu_exact_bias_512_loop32

gelu_exact_bias_512_vec16_check:
	CMPQ DX, $16
	JL gelu_exact_bias_512_next_row

	VMOVUPS (DI), Z0
	VADDPS (SI), Z0, Z0

	VMULPS Z18, Z0, Z1
	VPANDD Z31, Z1, Z2

	VMOVAPS Z17, Z3
	VFMADD231PS Z2, Z20, Z3
	VDIVPS Z3, Z17, Z3

	VMULPS Z2, Z2, Z4
	VPXORD Z19, Z4, Z4
	VMAXPS Z26, Z4, Z4

	VMOVAPS Z28, Z5
	VFMADD231PS Z4, Z27, Z5
	VSUBPS Z28, Z5, Z6

	VFNMADD231PS Z6, Z29, Z4
	VFNMADD231PS Z6, Z30, Z4

	VPSLLD $23, Z5, Z5
	VPADDD Z17, Z5, Z5

	VMULPS Z4, Z4, Z6
	VMOVAPS Z12, Z7
	VFMADD231PS Z4, Z11, Z7
	VMOVAPS Z14, Z8
	VFMADD231PS Z4, Z13, Z8
	VMOVAPS Z16, Z9
	VFMADD231PS Z4, Z15, Z9
	VADDPS Z17, Z4, Z10
	VFMADD231PS Z6, Z7, Z8
	VFMADD231PS Z6, Z9, Z10
	VMULPS Z6, Z6, Z6
	VFMADD231PS Z6, Z8, Z10
	VMULPS Z5, Z10, Z5

	VMULPS Z3, Z3, Z6
	VMOVAPS Z24, Z7
	VFMADD231PS Z3, Z25, Z7
	VMOVAPS Z22, Z8
	VFMADD231PS Z3, Z23, Z8
	VFMADD231PS Z6, Z7, Z8
	VMOVAPS Z21, Z9
	VFMADD231PS Z3, Z8, Z9
	VMULPS Z3, Z9, Z9

	VMOVAPS Z17, Z4
	VFNMADD231PS Z9, Z5, Z4

	VPANDD Z19, Z1, Z1
	VPXORD Z1, Z4, Z4
	VADDPS Z17, Z4, Z4
	VMULPS Z16, Z0, Z0
	VMULPS Z4, Z0, Z0

	VMOVUPS Z0, (DI)

gelu_exact_bias_512_next_row:
	ADDQ R10, R8
	DECQ R11
	JNZ gelu_exact_bias_512_row_loop

gelu_exact_bias_512_done:
	VZEROUPPER
	RET

// ===========================================================================
// geluApproxBiasTileFloat32AVX512Asm: Fused Bias + Approx GELU 2D tile for float32
//   Applies out[r*rowStrideBytes + c] = GeluApprox(out[r*rowStrideBytes + c] + bias[c])
//   for r in [0, numRows) and c in [0, width).
//   width must be a positive multiple of 16.
// ===========================================================================
// func geluApproxBiasTileFloat32AVX512Asm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)
TEXT ·geluApproxBiasTileFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), R8
	MOVQ bias+8(FP), R9
	MOVQ rowStrideBytes+16(FP), R10
	MOVQ numRows+24(FP), R11
	MOVQ width+32(FP), R12

	CMPQ R11, $0
	JLE gelu_approx_bias_512_done
	CMPQ R12, $16
	JL gelu_approx_bias_512_done

	// Pin all 16 constants in Z16..Z31 once across the entire 2D tile:
	VBROADCASTSS ·geluAVX512Consts+20(SB), Z16 // Z16: vHalf / vP2 = 0.5
	VBROADCASTSS ·geluAVX512Consts+24(SB), Z17 // Z17: vOne / vBias127 = 1.0
	VBROADCASTSS ·geluAVX512Consts+80(SB), Z18 // Z18: vTwo = 2.0
	VBROADCASTSS ·geluAVX512Consts+84(SB), Z19 // Z19: vNine = 9.0
	VBROADCASTSS ·geluAVX512Consts+88(SB), Z20 // Z20: vNegNine = -9.0
	VBROADCASTSS ·geluAVX512Consts+92(SB), Z21 // Z21: vSqrt2OverPi = 0.7978845608
	VBROADCASTSS ·geluAVX512Consts+96(SB), Z22 // Z22: vCoeff = 0.044715
	VBROADCASTSS ·geluAVX512Consts+64(SB), Z23 // Z23: vLog2E = 1.44269504
	VBROADCASTSS ·geluAVX512Consts+68(SB), Z24 // Z24: vMagic = 12582912.0
	VBROADCASTSS ·geluAVX512Consts+72(SB), Z25 // Z25: vLn2Hi = 0.693359375
	VBROADCASTSS ·geluAVX512Consts+76(SB), Z26 // Z26: vLn2Lo = -2.12194440e-4
	VBROADCASTSS ·geluAVX512Consts+0(SB), Z27  // Z27: vP7 = 1.9875691500e-4
	VBROADCASTSS ·geluAVX512Consts+4(SB), Z28  // Z28: vP6 = 1.3981999507e-3
	VBROADCASTSS ·geluAVX512Consts+8(SB), Z29  // Z29: vP5 = 8.3334519073e-3
	VBROADCASTSS ·geluAVX512Consts+12(SB), Z30 // Z30: vP4 = 4.1665795894e-2
	VBROADCASTSS ·geluAVX512Consts+16(SB), Z31 // Z31: vP3 = 1.6666665459e-1

gelu_approx_bias_512_row_loop:
	MOVQ R8, DI
	MOVQ R9, SI
	MOVQ R12, DX

	CMPQ DX, $32
	JL gelu_approx_bias_512_vec16_check

gelu_approx_bias_512_loop32:
	VMOVUPS 0(DI), Z0
	VMOVUPS 64(DI), Z1
	VADDPS 0(SI), Z0, Z0
	VADDPS 64(SI), Z1, Z1

	// x3 = v * v * v
	VMULPS Z0, Z0, Z2
	VMULPS Z1, Z1, Z3
	VMULPS Z0, Z2, Z2
	VMULPS Z1, Z3, Z3

	// inner = vSqrt2OverPi * (v + vCoeff * x3)
	VMOVAPS Z0, Z4
	VMOVAPS Z1, Z5
	VFMADD231PS Z2, Z22, Z4
	VFMADD231PS Z3, Z22, Z5
	VMULPS Z21, Z4, Z4
	VMULPS Z21, Z5, Z5

	// Clamp inner to [-9.0, 9.0] and multiply by 2.0
	VMINPS Z19, Z4, Z4
	VMINPS Z19, Z5, Z5
	VMAXPS Z20, Z4, Z4
	VMAXPS Z20, Z5, Z5
	VMULPS Z18, Z4, Z4
	VMULPS Z18, Z5, Z5

	// Range reduction: zRound = twoX * log2e + magic, zFloor = zRound - magic
	VMOVAPS Z24, Z6
	VMOVAPS Z24, Z7
	VFMADD231PS Z4, Z23, Z6
	VFMADD231PS Z5, Z23, Z7
	VSUBPS Z24, Z6, Z8
	VSUBPS Z24, Z7, Z9

	// g = twoX - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS Z8, Z25, Z4
	VFNMADD231PS Z9, Z25, Z5
	VFNMADD231PS Z8, Z26, Z4
	VFNMADD231PS Z9, Z26, Z5

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Z6, Z6
	VPSLLD $23, Z7, Z7
	VPADDD Z17, Z6, Z6
	VPADDD Z17, Z7, Z7

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Z4, Z4, Z2
	VMULPS Z5, Z5, Z3
	VMOVAPS Z28, Z8
	VMOVAPS Z28, Z9
	VFMADD231PS Z4, Z27, Z8
	VFMADD231PS Z5, Z27, Z9
	VMOVAPS Z30, Z10
	VMOVAPS Z30, Z11
	VFMADD231PS Z4, Z29, Z10
	VFMADD231PS Z5, Z29, Z11
	VMOVAPS Z16, Z12
	VMOVAPS Z16, Z13
	VFMADD231PS Z4, Z31, Z12
	VFMADD231PS Z5, Z31, Z13
	VADDPS Z17, Z4, Z14
	VADDPS Z17, Z5, Z15
	VFMADD231PS Z2, Z8, Z10
	VFMADD231PS Z3, Z9, Z11
	VFMADD231PS Z2, Z12, Z14
	VFMADD231PS Z3, Z13, Z15
	VMULPS Z2, Z2, Z2
	VMULPS Z3, Z3, Z3
	VFMADD231PS Z2, Z10, Z14
	VFMADD231PS Z3, Z11, Z15
	VMULPS Z6, Z14, Z6
	VMULPS Z7, Z15, Z7

	// out = (v * exp2x) / (exp2x + 1.0)
	VADDPS Z17, Z6, Z8
	VADDPS Z17, Z7, Z9
	VMULPS Z6, Z0, Z0
	VMULPS Z7, Z1, Z1
	VDIVPS Z8, Z0, Z0
	VDIVPS Z9, Z1, Z1

	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)

	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, DX
	CMPQ DX, $32
	JGE gelu_approx_bias_512_loop32

gelu_approx_bias_512_vec16_check:
	CMPQ DX, $16
	JL gelu_approx_bias_512_next_row

	VMOVUPS (DI), Z0
	VADDPS (SI), Z0, Z0
	VMULPS Z0, Z0, Z2
	VMULPS Z0, Z2, Z2
	VMOVAPS Z0, Z4
	VFMADD231PS Z2, Z22, Z4
	VMULPS Z21, Z4, Z4
	VMINPS Z19, Z4, Z4
	VMAXPS Z20, Z4, Z4
	VMULPS Z18, Z4, Z4
	VMOVAPS Z24, Z6
	VFMADD231PS Z4, Z23, Z6
	VSUBPS Z24, Z6, Z8
	VFNMADD231PS Z8, Z25, Z4
	VFNMADD231PS Z8, Z26, Z4
	VPSLLD $23, Z6, Z6
	VPADDD Z17, Z6, Z6
	VMULPS Z4, Z4, Z2
	VMOVAPS Z28, Z8
	VFMADD231PS Z4, Z27, Z8
	VMOVAPS Z30, Z10
	VFMADD231PS Z4, Z29, Z10
	VMOVAPS Z16, Z12
	VFMADD231PS Z4, Z31, Z12
	VADDPS Z17, Z4, Z14
	VFMADD231PS Z2, Z8, Z10
	VFMADD231PS Z2, Z12, Z14
	VMULPS Z2, Z2, Z2
	VFMADD231PS Z2, Z10, Z14
	VMULPS Z6, Z14, Z6
	VADDPS Z17, Z6, Z8
	VMULPS Z6, Z0, Z0
	VDIVPS Z8, Z0, Z0
	VMOVUPS Z0, (DI)

gelu_approx_bias_512_next_row:
	ADDQ R10, R8
	DECQ R11
	JNZ gelu_approx_bias_512_row_loop

gelu_approx_bias_512_done:
	VZEROUPPER
	RET


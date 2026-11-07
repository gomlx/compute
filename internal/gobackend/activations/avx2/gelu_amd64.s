// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// RODATA constants for AVX2 GELU (Exact and Approximate)
// Each entry is 8 repeated float32/uint32 lanes (32 bytes = 256 bits) so it can be
// used directly as a 256-bit memory operand in VFMADD / VMULPS / VADDPS / VPAND / etc.
GLOBL ·geluAVX2Consts(SB), RODATA|NOPTR, $832

//   0: vP7 = 1.9875691500e-4 (0x39506967)
DATA ·geluAVX2Consts+0(SB)/4, $0x39506967
DATA ·geluAVX2Consts+4(SB)/4, $0x39506967
DATA ·geluAVX2Consts+8(SB)/4, $0x39506967
DATA ·geluAVX2Consts+12(SB)/4, $0x39506967
DATA ·geluAVX2Consts+16(SB)/4, $0x39506967
DATA ·geluAVX2Consts+20(SB)/4, $0x39506967
DATA ·geluAVX2Consts+24(SB)/4, $0x39506967
DATA ·geluAVX2Consts+28(SB)/4, $0x39506967

//  32: vP6 = 1.3981999507e-3 (0x3ab743ce)
DATA ·geluAVX2Consts+32(SB)/4, $0x3ab743ce
DATA ·geluAVX2Consts+36(SB)/4, $0x3ab743ce
DATA ·geluAVX2Consts+40(SB)/4, $0x3ab743ce
DATA ·geluAVX2Consts+44(SB)/4, $0x3ab743ce
DATA ·geluAVX2Consts+48(SB)/4, $0x3ab743ce
DATA ·geluAVX2Consts+52(SB)/4, $0x3ab743ce
DATA ·geluAVX2Consts+56(SB)/4, $0x3ab743ce
DATA ·geluAVX2Consts+60(SB)/4, $0x3ab743ce

//  64: vP5 = 8.3334519073e-3 (0x3c088895)
DATA ·geluAVX2Consts+64(SB)/4, $0x3c088895
DATA ·geluAVX2Consts+68(SB)/4, $0x3c088895
DATA ·geluAVX2Consts+72(SB)/4, $0x3c088895
DATA ·geluAVX2Consts+76(SB)/4, $0x3c088895
DATA ·geluAVX2Consts+80(SB)/4, $0x3c088895
DATA ·geluAVX2Consts+84(SB)/4, $0x3c088895
DATA ·geluAVX2Consts+88(SB)/4, $0x3c088895
DATA ·geluAVX2Consts+92(SB)/4, $0x3c088895

//  96: vP4 = 4.1665795894e-2 (0x3d2aa9a5)
DATA ·geluAVX2Consts+96(SB)/4, $0x3d2aa9a5
DATA ·geluAVX2Consts+100(SB)/4, $0x3d2aa9a5
DATA ·geluAVX2Consts+104(SB)/4, $0x3d2aa9a5
DATA ·geluAVX2Consts+108(SB)/4, $0x3d2aa9a5
DATA ·geluAVX2Consts+112(SB)/4, $0x3d2aa9a5
DATA ·geluAVX2Consts+116(SB)/4, $0x3d2aa9a5
DATA ·geluAVX2Consts+120(SB)/4, $0x3d2aa9a5
DATA ·geluAVX2Consts+124(SB)/4, $0x3d2aa9a5

// 128: vP3 = 1.6666665459e-1 (0x3e2aaaab)
DATA ·geluAVX2Consts+128(SB)/4, $0x3e2aaaab
DATA ·geluAVX2Consts+132(SB)/4, $0x3e2aaaab
DATA ·geluAVX2Consts+136(SB)/4, $0x3e2aaaab
DATA ·geluAVX2Consts+140(SB)/4, $0x3e2aaaab
DATA ·geluAVX2Consts+144(SB)/4, $0x3e2aaaab
DATA ·geluAVX2Consts+148(SB)/4, $0x3e2aaaab
DATA ·geluAVX2Consts+152(SB)/4, $0x3e2aaaab
DATA ·geluAVX2Consts+156(SB)/4, $0x3e2aaaab

// 160: vHalf / vP2 = 0.5 (0x3f000000)
DATA ·geluAVX2Consts+160(SB)/4, $0x3f000000
DATA ·geluAVX2Consts+164(SB)/4, $0x3f000000
DATA ·geluAVX2Consts+168(SB)/4, $0x3f000000
DATA ·geluAVX2Consts+172(SB)/4, $0x3f000000
DATA ·geluAVX2Consts+176(SB)/4, $0x3f000000
DATA ·geluAVX2Consts+180(SB)/4, $0x3f000000
DATA ·geluAVX2Consts+184(SB)/4, $0x3f000000
DATA ·geluAVX2Consts+188(SB)/4, $0x3f000000

// 192: vOne / vBias127 = 1.0 (0x3f800000)
DATA ·geluAVX2Consts+192(SB)/4, $0x3f800000
DATA ·geluAVX2Consts+196(SB)/4, $0x3f800000
DATA ·geluAVX2Consts+200(SB)/4, $0x3f800000
DATA ·geluAVX2Consts+204(SB)/4, $0x3f800000
DATA ·geluAVX2Consts+208(SB)/4, $0x3f800000
DATA ·geluAVX2Consts+212(SB)/4, $0x3f800000
DATA ·geluAVX2Consts+216(SB)/4, $0x3f800000
DATA ·geluAVX2Consts+220(SB)/4, $0x3f800000

// 224: vRsqrt2 = 0.70710678 (0x3f3504f3)
DATA ·geluAVX2Consts+224(SB)/4, $0x3f3504f3
DATA ·geluAVX2Consts+228(SB)/4, $0x3f3504f3
DATA ·geluAVX2Consts+232(SB)/4, $0x3f3504f3
DATA ·geluAVX2Consts+236(SB)/4, $0x3f3504f3
DATA ·geluAVX2Consts+240(SB)/4, $0x3f3504f3
DATA ·geluAVX2Consts+244(SB)/4, $0x3f3504f3
DATA ·geluAVX2Consts+248(SB)/4, $0x3f3504f3
DATA ·geluAVX2Consts+252(SB)/4, $0x3f3504f3

// 256: vSignMask = 0x80000000
DATA ·geluAVX2Consts+256(SB)/4, $0x80000000
DATA ·geluAVX2Consts+260(SB)/4, $0x80000000
DATA ·geluAVX2Consts+264(SB)/4, $0x80000000
DATA ·geluAVX2Consts+268(SB)/4, $0x80000000
DATA ·geluAVX2Consts+272(SB)/4, $0x80000000
DATA ·geluAVX2Consts+276(SB)/4, $0x80000000
DATA ·geluAVX2Consts+280(SB)/4, $0x80000000
DATA ·geluAVX2Consts+284(SB)/4, $0x80000000

// 288: vP = 0.3275911 (0x3ea7ba05)
DATA ·geluAVX2Consts+288(SB)/4, $0x3ea7ba05
DATA ·geluAVX2Consts+292(SB)/4, $0x3ea7ba05
DATA ·geluAVX2Consts+296(SB)/4, $0x3ea7ba05
DATA ·geluAVX2Consts+300(SB)/4, $0x3ea7ba05
DATA ·geluAVX2Consts+304(SB)/4, $0x3ea7ba05
DATA ·geluAVX2Consts+308(SB)/4, $0x3ea7ba05
DATA ·geluAVX2Consts+312(SB)/4, $0x3ea7ba05
DATA ·geluAVX2Consts+316(SB)/4, $0x3ea7ba05

// 320: vA1 = 0.254829592 (0x3e827906)
DATA ·geluAVX2Consts+320(SB)/4, $0x3e827906
DATA ·geluAVX2Consts+324(SB)/4, $0x3e827906
DATA ·geluAVX2Consts+328(SB)/4, $0x3e827906
DATA ·geluAVX2Consts+332(SB)/4, $0x3e827906
DATA ·geluAVX2Consts+336(SB)/4, $0x3e827906
DATA ·geluAVX2Consts+340(SB)/4, $0x3e827906
DATA ·geluAVX2Consts+344(SB)/4, $0x3e827906
DATA ·geluAVX2Consts+348(SB)/4, $0x3e827906

// 352: vA2 = -0.284496736 (0xbe91a98e)
DATA ·geluAVX2Consts+352(SB)/4, $0xbe91a98e
DATA ·geluAVX2Consts+356(SB)/4, $0xbe91a98e
DATA ·geluAVX2Consts+360(SB)/4, $0xbe91a98e
DATA ·geluAVX2Consts+364(SB)/4, $0xbe91a98e
DATA ·geluAVX2Consts+368(SB)/4, $0xbe91a98e
DATA ·geluAVX2Consts+372(SB)/4, $0xbe91a98e
DATA ·geluAVX2Consts+376(SB)/4, $0xbe91a98e
DATA ·geluAVX2Consts+380(SB)/4, $0xbe91a98e

// 384: vA3 = 1.421413741 (0x3fb5f0e3)
DATA ·geluAVX2Consts+384(SB)/4, $0x3fb5f0e3
DATA ·geluAVX2Consts+388(SB)/4, $0x3fb5f0e3
DATA ·geluAVX2Consts+392(SB)/4, $0x3fb5f0e3
DATA ·geluAVX2Consts+396(SB)/4, $0x3fb5f0e3
DATA ·geluAVX2Consts+400(SB)/4, $0x3fb5f0e3
DATA ·geluAVX2Consts+404(SB)/4, $0x3fb5f0e3
DATA ·geluAVX2Consts+408(SB)/4, $0x3fb5f0e3
DATA ·geluAVX2Consts+412(SB)/4, $0x3fb5f0e3

// 416: vA4 = -1.453152027 (0xbfb9fffb)
DATA ·geluAVX2Consts+416(SB)/4, $0xbfb9fffb
DATA ·geluAVX2Consts+420(SB)/4, $0xbfb9fffb
DATA ·geluAVX2Consts+424(SB)/4, $0xbfb9fffb
DATA ·geluAVX2Consts+428(SB)/4, $0xbfb9fffb
DATA ·geluAVX2Consts+432(SB)/4, $0xbfb9fffb
DATA ·geluAVX2Consts+436(SB)/4, $0xbfb9fffb
DATA ·geluAVX2Consts+440(SB)/4, $0xbfb9fffb
DATA ·geluAVX2Consts+444(SB)/4, $0xbfb9fffb

// 448: vA5 = 1.061405429 (0x3f87dc22)
DATA ·geluAVX2Consts+448(SB)/4, $0x3f87dc22
DATA ·geluAVX2Consts+452(SB)/4, $0x3f87dc22
DATA ·geluAVX2Consts+456(SB)/4, $0x3f87dc22
DATA ·geluAVX2Consts+460(SB)/4, $0x3f87dc22
DATA ·geluAVX2Consts+464(SB)/4, $0x3f87dc22
DATA ·geluAVX2Consts+468(SB)/4, $0x3f87dc22
DATA ·geluAVX2Consts+472(SB)/4, $0x3f87dc22
DATA ·geluAVX2Consts+476(SB)/4, $0x3f87dc22

// 480: vMinLog = -87.336544 (0xc2aeac50)
DATA ·geluAVX2Consts+480(SB)/4, $0xc2aeac50
DATA ·geluAVX2Consts+484(SB)/4, $0xc2aeac50
DATA ·geluAVX2Consts+488(SB)/4, $0xc2aeac50
DATA ·geluAVX2Consts+492(SB)/4, $0xc2aeac50
DATA ·geluAVX2Consts+496(SB)/4, $0xc2aeac50
DATA ·geluAVX2Consts+500(SB)/4, $0xc2aeac50
DATA ·geluAVX2Consts+504(SB)/4, $0xc2aeac50
DATA ·geluAVX2Consts+508(SB)/4, $0xc2aeac50

// 512: vLog2E = 1.44269504 (0x3fb8aa3b)
DATA ·geluAVX2Consts+512(SB)/4, $0x3fb8aa3b
DATA ·geluAVX2Consts+516(SB)/4, $0x3fb8aa3b
DATA ·geluAVX2Consts+520(SB)/4, $0x3fb8aa3b
DATA ·geluAVX2Consts+524(SB)/4, $0x3fb8aa3b
DATA ·geluAVX2Consts+528(SB)/4, $0x3fb8aa3b
DATA ·geluAVX2Consts+532(SB)/4, $0x3fb8aa3b
DATA ·geluAVX2Consts+536(SB)/4, $0x3fb8aa3b
DATA ·geluAVX2Consts+540(SB)/4, $0x3fb8aa3b

// 544: vMagic = 12582912.0 (0x4b400000)
DATA ·geluAVX2Consts+544(SB)/4, $0x4b400000
DATA ·geluAVX2Consts+548(SB)/4, $0x4b400000
DATA ·geluAVX2Consts+552(SB)/4, $0x4b400000
DATA ·geluAVX2Consts+556(SB)/4, $0x4b400000
DATA ·geluAVX2Consts+560(SB)/4, $0x4b400000
DATA ·geluAVX2Consts+564(SB)/4, $0x4b400000
DATA ·geluAVX2Consts+568(SB)/4, $0x4b400000
DATA ·geluAVX2Consts+572(SB)/4, $0x4b400000

// 576: vLn2Hi = 0.693359375 (0x3f318000)
DATA ·geluAVX2Consts+576(SB)/4, $0x3f318000
DATA ·geluAVX2Consts+580(SB)/4, $0x3f318000
DATA ·geluAVX2Consts+584(SB)/4, $0x3f318000
DATA ·geluAVX2Consts+588(SB)/4, $0x3f318000
DATA ·geluAVX2Consts+592(SB)/4, $0x3f318000
DATA ·geluAVX2Consts+596(SB)/4, $0x3f318000
DATA ·geluAVX2Consts+600(SB)/4, $0x3f318000
DATA ·geluAVX2Consts+604(SB)/4, $0x3f318000

// 608: vLn2Lo = -2.12194440e-4 (0xb95e8083)
DATA ·geluAVX2Consts+608(SB)/4, $0xb95e8083
DATA ·geluAVX2Consts+612(SB)/4, $0xb95e8083
DATA ·geluAVX2Consts+616(SB)/4, $0xb95e8083
DATA ·geluAVX2Consts+620(SB)/4, $0xb95e8083
DATA ·geluAVX2Consts+624(SB)/4, $0xb95e8083
DATA ·geluAVX2Consts+628(SB)/4, $0xb95e8083
DATA ·geluAVX2Consts+632(SB)/4, $0xb95e8083
DATA ·geluAVX2Consts+636(SB)/4, $0xb95e8083

// 640: vTwo = 2.0 (0x40000000)
DATA ·geluAVX2Consts+640(SB)/4, $0x40000000
DATA ·geluAVX2Consts+644(SB)/4, $0x40000000
DATA ·geluAVX2Consts+648(SB)/4, $0x40000000
DATA ·geluAVX2Consts+652(SB)/4, $0x40000000
DATA ·geluAVX2Consts+656(SB)/4, $0x40000000
DATA ·geluAVX2Consts+660(SB)/4, $0x40000000
DATA ·geluAVX2Consts+664(SB)/4, $0x40000000
DATA ·geluAVX2Consts+668(SB)/4, $0x40000000

// 672: vNine = 9.0 (0x41100000)
DATA ·geluAVX2Consts+672(SB)/4, $0x41100000
DATA ·geluAVX2Consts+676(SB)/4, $0x41100000
DATA ·geluAVX2Consts+680(SB)/4, $0x41100000
DATA ·geluAVX2Consts+684(SB)/4, $0x41100000
DATA ·geluAVX2Consts+688(SB)/4, $0x41100000
DATA ·geluAVX2Consts+692(SB)/4, $0x41100000
DATA ·geluAVX2Consts+696(SB)/4, $0x41100000
DATA ·geluAVX2Consts+700(SB)/4, $0x41100000

// 704: vNegNine = -9.0 (0xc1100000)
DATA ·geluAVX2Consts+704(SB)/4, $0xc1100000
DATA ·geluAVX2Consts+708(SB)/4, $0xc1100000
DATA ·geluAVX2Consts+712(SB)/4, $0xc1100000
DATA ·geluAVX2Consts+716(SB)/4, $0xc1100000
DATA ·geluAVX2Consts+720(SB)/4, $0xc1100000
DATA ·geluAVX2Consts+724(SB)/4, $0xc1100000
DATA ·geluAVX2Consts+728(SB)/4, $0xc1100000
DATA ·geluAVX2Consts+732(SB)/4, $0xc1100000

// 736: vSqrt2OverPi = 0.7978845608 (0x3f4c422a)
DATA ·geluAVX2Consts+736(SB)/4, $0x3f4c422a
DATA ·geluAVX2Consts+740(SB)/4, $0x3f4c422a
DATA ·geluAVX2Consts+744(SB)/4, $0x3f4c422a
DATA ·geluAVX2Consts+748(SB)/4, $0x3f4c422a
DATA ·geluAVX2Consts+752(SB)/4, $0x3f4c422a
DATA ·geluAVX2Consts+756(SB)/4, $0x3f4c422a
DATA ·geluAVX2Consts+760(SB)/4, $0x3f4c422a
DATA ·geluAVX2Consts+764(SB)/4, $0x3f4c422a

// 768: vCoeff = 0.044715 (0x3d372713)
DATA ·geluAVX2Consts+768(SB)/4, $0x3d372713
DATA ·geluAVX2Consts+772(SB)/4, $0x3d372713
DATA ·geluAVX2Consts+776(SB)/4, $0x3d372713
DATA ·geluAVX2Consts+780(SB)/4, $0x3d372713
DATA ·geluAVX2Consts+784(SB)/4, $0x3d372713
DATA ·geluAVX2Consts+788(SB)/4, $0x3d372713
DATA ·geluAVX2Consts+792(SB)/4, $0x3d372713
DATA ·geluAVX2Consts+796(SB)/4, $0x3d372713

// 800: vAbsMask = 0x7fffffff
DATA ·geluAVX2Consts+800(SB)/4, $0x7fffffff
DATA ·geluAVX2Consts+804(SB)/4, $0x7fffffff
DATA ·geluAVX2Consts+808(SB)/4, $0x7fffffff
DATA ·geluAVX2Consts+812(SB)/4, $0x7fffffff
DATA ·geluAVX2Consts+816(SB)/4, $0x7fffffff
DATA ·geluAVX2Consts+820(SB)/4, $0x7fffffff
DATA ·geluAVX2Consts+824(SB)/4, $0x7fffffff
DATA ·geluAVX2Consts+828(SB)/4, $0x7fffffff

// ===========================================================================
// geluExactFloat32AVX2Asm: Exact GELU for float32 using AVX2
//   GELU(x) = 0.5 * x * (1 + erf(x * rsqrt(2)))
//   Uses 2-vector (16 float32) unrolling, Estrin's parallel polynomial trees,
//   overlapped VDIVPS latency, and RODATA memory operands to avoid register spills.
//   n must be a positive multiple of 8.
// ===========================================================================
// func geluExactFloat32AVX2Asm(in, out unsafe.Pointer, n int)
TEXT ·geluExactFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ n+16(FP), DX

	CMPQ DX, $8
	JL gelu_exact_avx2_done

	LEAQ ·geluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14 // Y14: vHalf / vP2 = 0.5 (0x3f000000)
	VMOVUPS 192(CX), Y15 // Y15: vOne / vBias127 = 1.0 (0x3f800000)

	CMPQ DX, $16
	JL gelu_exact_avx2_vec8_check

gelu_exact_avx2_loop16:
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1

	// x = v * rsqrt2, absX = |x|
	VMULPS 224(CX), Y0, Y2
	VMULPS 224(CX), Y1, Y3
	VPAND 800(CX), Y2, Y2
	VPAND 800(CX), Y3, Y3

	// Issue t = 1.0 / (1.0 + p * absX) early so VDIVPS overlaps with exp(-x^2):
	VMOVAPS Y15, Y4
	VMOVAPS Y15, Y5
	VFMADD231PS 288(CX), Y2, Y4
	VFMADD231PS 288(CX), Y3, Y5
	VDIVPS Y4, Y15, Y4
	VDIVPS Y5, Y15, Y5

	// negX2 = max(-(absX * absX), vMinLog)
	VMULPS Y2, Y2, Y6
	VMULPS Y3, Y3, Y7
	VXORPS 256(CX), Y6, Y6
	VXORPS 256(CX), Y7, Y7
	VMAXPS 480(CX), Y6, Y6
	VMAXPS 480(CX), Y7, Y7

	// zRound = xClamped * log2e + magic, zFloor = zRound - magic
	VMOVUPS 544(CX), Y12
	VMOVAPS Y12, Y8
	VMOVAPS Y12, Y9
	VFMADD231PS 512(CX), Y6, Y8
	VFMADD231PS 512(CX), Y7, Y9
	VSUBPS Y12, Y8, Y10
	VSUBPS Y12, Y9, Y11

	// g = xClamped - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS 576(CX), Y10, Y6
	VFNMADD231PS 576(CX), Y11, Y7
	VFNMADD231PS 608(CX), Y10, Y6
	VFNMADD231PS 608(CX), Y11, Y7

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Y8, Y8
	VPSLLD $23, Y9, Y9
	VPADDD Y15, Y8, Y8
	VPADDD Y15, Y9, Y9

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Y6, Y6, Y2             // Y2: g2A
	VMULPS Y7, Y7, Y3             // Y3: g2B
	VMOVUPS 32(CX), Y10
	VMOVAPS Y10, Y11
	VFMADD231PS 0(CX), Y6, Y10    // Y10: q67A = vP6 + gA * vP7
	VFMADD231PS 0(CX), Y7, Y11    // Y11: q67B = vP6 + gB * vP7
	VMOVUPS 96(CX), Y12
	VMOVAPS Y12, Y13
	VFMADD231PS 64(CX), Y6, Y12   // Y12: q45A = vP4 + gA * vP5
	VFMADD231PS 64(CX), Y7, Y13   // Y13: q45B = vP4 + gB * vP5
	VFMADD231PS Y2, Y10, Y12      // Y12: q47A = q45A + g2A * q67A
	VFMADD231PS Y3, Y11, Y13      // Y13: q47B = q45B + g2B * q67B

	VMOVAPS Y14, Y10
	VMOVAPS Y14, Y11
	VFMADD231PS 128(CX), Y6, Y10  // Y10: q23A = vP2 + gA * vP3
	VFMADD231PS 128(CX), Y7, Y11  // Y11: q23B = vP2 + gB * vP3
	VADDPS Y15, Y6, Y6            // Y6: q01A = gA + 1.0
	VADDPS Y15, Y7, Y7            // Y7: q01B = gB + 1.0
	VFMADD231PS Y2, Y10, Y6       // Y6: q03A = q01A + g2A * q23A
	VFMADD231PS Y3, Y11, Y7       // Y7: q03B = q01B + g2B * q23B

	VMULPS Y2, Y2, Y2             // Y2: g4A
	VMULPS Y3, Y3, Y3             // Y3: g4B
	VFMADD231PS Y2, Y12, Y6       // Y6: expPolyA = q03A + g4A * q47A
	VFMADD231PS Y3, Y13, Y7       // Y7: expPolyB = q03B + g4B * q47B
	VMULPS Y8, Y6, Y8             // Y8: expNegX2A = expPolyA * scaleA
	VMULPS Y9, Y7, Y9             // Y9: expNegX2B = expPolyB * scaleB

	// Estrin's scheme for erfPoly(t) using Y4 (tA) and Y5 (tB):
	VMULPS Y4, Y4, Y2             // Y2: t2A
	VMULPS Y5, Y5, Y3             // Y3: t2B
	VMOVUPS 416(CX), Y10
	VMOVAPS Y10, Y11
	VFMADD231PS 448(CX), Y4, Y10  // Y10: p45A = vA4 + tA * vA5
	VFMADD231PS 448(CX), Y5, Y11  // Y11: p45B = vA4 + tB * vA5
	VMOVUPS 352(CX), Y12
	VMOVAPS Y12, Y13
	VFMADD231PS 384(CX), Y4, Y12  // Y12: p23A = vA2 + tA * vA3
	VFMADD231PS 384(CX), Y5, Y13  // Y13: p23B = vA2 + tB * vA3
	VFMADD231PS Y2, Y10, Y12      // Y12: p25A = p23A + t2A * p45A
	VFMADD231PS Y3, Y11, Y13      // Y13: p25B = p23B + t2B * p45B
	VFMADD213PS 320(CX), Y4, Y12  // Y12: p15A = p25A * tA + vA1
	VFMADD213PS 320(CX), Y5, Y13  // Y13: p15B = p25B * tB + vA1
	VMULPS Y4, Y12, Y12           // Y12: erfPolyA = p15A * tA
	VMULPS Y5, Y13, Y13           // Y13: erfPolyB = p15B * tB

	// res = 1.0 - erfPoly * expNegX2
	VMOVAPS Y15, Y6
	VMOVAPS Y15, Y7
	VFNMADD231PS Y12, Y8, Y6
	VFNMADD231PS Y13, Y9, Y7

	// Restore sign of v and compute 0.5 * v * (1.0 + erfVal)
	VPAND 256(CX), Y0, Y2
	VPAND 256(CX), Y1, Y3
	VXORPS Y2, Y6, Y6
	VXORPS Y3, Y7, Y7
	VADDPS Y15, Y6, Y6
	VADDPS Y15, Y7, Y7
	VMULPS Y14, Y0, Y0
	VMULPS Y14, Y1, Y1
	VMULPS Y6, Y0, Y0
	VMULPS Y7, Y1, Y1

	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE gelu_exact_avx2_loop16

gelu_exact_avx2_vec8_check:
	CMPQ DX, $8
	JL gelu_exact_avx2_done

	VMOVUPS 0(SI), Y0
	VMULPS 224(CX), Y0, Y2
	VPAND 800(CX), Y2, Y2
	VMOVAPS Y15, Y4
	VFMADD231PS 288(CX), Y2, Y4
	VDIVPS Y4, Y15, Y4
	VMULPS Y2, Y2, Y6
	VXORPS 256(CX), Y6, Y6
	VMAXPS 480(CX), Y6, Y6
	VMOVUPS 544(CX), Y8
	VFMADD231PS 512(CX), Y6, Y8
	VSUBPS 544(CX), Y8, Y10
	VFNMADD231PS 576(CX), Y10, Y6
	VFNMADD231PS 608(CX), Y10, Y6
	VPSLLD $23, Y8, Y8
	VPADDD Y15, Y8, Y8
	VMULPS Y6, Y6, Y2
	VMOVUPS 32(CX), Y10
	VFMADD231PS 0(CX), Y6, Y10
	VMOVUPS 96(CX), Y12
	VFMADD231PS 64(CX), Y6, Y12
	VFMADD231PS Y2, Y10, Y12
	VMOVAPS Y14, Y10
	VFMADD231PS 128(CX), Y6, Y10
	VADDPS Y15, Y6, Y6
	VFMADD231PS Y2, Y10, Y6
	VMULPS Y2, Y2, Y2
	VFMADD231PS Y2, Y12, Y6
	VMULPS Y8, Y6, Y8
	VMULPS Y4, Y4, Y2
	VMOVUPS 416(CX), Y10
	VFMADD231PS 448(CX), Y4, Y10
	VMOVUPS 352(CX), Y12
	VFMADD231PS 384(CX), Y4, Y12
	VFMADD231PS Y2, Y10, Y12
	VFMADD213PS 320(CX), Y4, Y12
	VMULPS Y4, Y12, Y12
	VMOVAPS Y15, Y6
	VFNMADD231PS Y12, Y8, Y6
	VPAND 256(CX), Y0, Y2
	VXORPS Y2, Y6, Y6
	VADDPS Y15, Y6, Y6
	VMULPS Y14, Y0, Y0
	VMULPS Y6, Y0, Y0
	VMOVUPS Y0, 0(DI)

gelu_exact_avx2_done:
	VZEROUPPER
	RET

// ===========================================================================
// geluApproxFloat32AVX2Asm: Approximate (tanh) GELU for float32 using AVX2
//   GELU(x) = 0.5 * x * (1 + tanh(u)) = (x * exp(2u)) / (exp(2u) + 1)
//   where u = sqrt(2/pi) * (x + 0.044715 * x^3).
//   Uses 2-vector (16 float32) unrolling and Estrin's parallel polynomial trees.
//   n must be a positive multiple of 8.
// ===========================================================================
// func geluApproxFloat32AVX2Asm(in, out unsafe.Pointer, n int)
TEXT ·geluApproxFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ n+16(FP), DX

	CMPQ DX, $8
	JL gelu_approx_avx2_done

	LEAQ ·geluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14 // Y14: vHalf / vP2 = 0.5 (0x3f000000)
	VMOVUPS 192(CX), Y15 // Y15: vOne / vBias127 = 1.0 (0x3f800000)

	CMPQ DX, $16
	JL gelu_approx_avx2_vec8_check

gelu_approx_avx2_loop16:
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1

	// x3 = v * v * v
	VMULPS Y0, Y0, Y2
	VMULPS Y1, Y1, Y3
	VMULPS Y0, Y2, Y2
	VMULPS Y1, Y3, Y3

	// inner = vSqrt2OverPi * (v + vCoeff * x3)
	VMOVAPS Y0, Y4
	VMOVAPS Y1, Y5
	VFMADD231PS 768(CX), Y2, Y4
	VFMADD231PS 768(CX), Y3, Y5
	VMULPS 736(CX), Y4, Y4
	VMULPS 736(CX), Y5, Y5

	// Clamp inner to [-9.0, 9.0] and multiply by 2.0
	VMINPS 672(CX), Y4, Y4
	VMINPS 672(CX), Y5, Y5
	VMAXPS 704(CX), Y4, Y4
	VMAXPS 704(CX), Y5, Y5
	VMULPS 640(CX), Y4, Y4
	VMULPS 640(CX), Y5, Y5

	// Range reduction: zRound = twoX * log2e + magic, zFloor = zRound - magic
	VMOVUPS 544(CX), Y12
	VMOVAPS Y12, Y6
	VMOVAPS Y12, Y7
	VFMADD231PS 512(CX), Y4, Y6
	VFMADD231PS 512(CX), Y5, Y7
	VSUBPS Y12, Y6, Y8
	VSUBPS Y12, Y7, Y9

	// g = twoX - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS 576(CX), Y8, Y4
	VFNMADD231PS 576(CX), Y9, Y5
	VFNMADD231PS 608(CX), Y8, Y4
	VFNMADD231PS 608(CX), Y9, Y5

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Y6, Y6
	VPSLLD $23, Y7, Y7
	VPADDD Y15, Y6, Y6
	VPADDD Y15, Y7, Y7

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Y4, Y4, Y2             // g2A
	VMULPS Y5, Y5, Y3             // g2B
	VMOVUPS 32(CX), Y8
	VMOVAPS Y8, Y9
	VFMADD231PS 0(CX), Y4, Y8     // q67A = vP6 + gA * vP7
	VFMADD231PS 0(CX), Y5, Y9     // q67B = vP6 + gB * vP7
	VMOVUPS 96(CX), Y10
	VMOVAPS Y10, Y11
	VFMADD231PS 64(CX), Y4, Y10   // q45A = vP4 + gA * vP5
	VFMADD231PS 64(CX), Y5, Y11   // q45B = vP4 + gB * vP5
	VMOVAPS Y14, Y12
	VMOVAPS Y14, Y13
	VFMADD231PS 128(CX), Y4, Y12  // q23A = vP2 + gA * vP3
	VFMADD231PS 128(CX), Y5, Y13  // q23B = vP2 + gB * vP3
	VADDPS Y15, Y4, Y4            // q01A
	VADDPS Y15, Y5, Y5            // q01B
	VFMADD231PS Y2, Y8, Y10       // q47A = q45A + g2A * q67A
	VFMADD231PS Y3, Y9, Y11       // q47B = q45B + g2B * q67B
	VFMADD231PS Y2, Y12, Y4       // q03A = q01A + g2A * q23A
	VFMADD231PS Y3, Y13, Y5       // q03B = q01B + g2B * q23B
	VMULPS Y2, Y2, Y2             // g4A
	VMULPS Y3, Y3, Y3             // g4B
	VFMADD231PS Y2, Y10, Y4       // expPolyA = q03A + g4A * q47A
	VFMADD231PS Y3, Y11, Y5       // expPolyB = q03B + g4B * q47B
	VMULPS Y6, Y4, Y6             // exp2xA = expPolyA * scaleA
	VMULPS Y7, Y5, Y7             // exp2xB = expPolyB * scaleB

	// out = (v * exp2x) / (exp2x + 1.0)
	VADDPS Y15, Y6, Y8
	VADDPS Y15, Y7, Y9
	VMULPS Y6, Y0, Y0
	VMULPS Y7, Y1, Y1
	VDIVPS Y8, Y0, Y0
	VDIVPS Y9, Y1, Y1

	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE gelu_approx_avx2_loop16

gelu_approx_avx2_vec8_check:
	CMPQ DX, $8
	JL gelu_approx_avx2_done

	VMOVUPS 0(SI), Y0
	VMULPS Y0, Y0, Y2
	VMULPS Y0, Y2, Y2
	VMOVAPS Y0, Y4
	VFMADD231PS 768(CX), Y2, Y4
	VMULPS 736(CX), Y4, Y4
	VMINPS 672(CX), Y4, Y4
	VMAXPS 704(CX), Y4, Y4
	VMULPS 640(CX), Y4, Y4
	VMOVUPS 544(CX), Y6
	VFMADD231PS 512(CX), Y4, Y6
	VSUBPS 544(CX), Y6, Y8
	VFNMADD231PS 576(CX), Y8, Y4
	VFNMADD231PS 608(CX), Y8, Y4
	VPSLLD $23, Y6, Y6
	VPADDD Y15, Y6, Y6
	VMULPS Y4, Y4, Y2
	VMOVUPS 32(CX), Y8
	VFMADD231PS 0(CX), Y4, Y8
	VMOVUPS 96(CX), Y10
	VFMADD231PS 64(CX), Y4, Y10
	VMOVAPS Y14, Y12
	VFMADD231PS 128(CX), Y4, Y12
	VADDPS Y15, Y4, Y4
	VFMADD231PS Y2, Y8, Y10
	VFMADD231PS Y2, Y12, Y4
	VMULPS Y2, Y2, Y2
	VFMADD231PS Y2, Y10, Y4
	VMULPS Y6, Y4, Y6
	VADDPS Y15, Y6, Y8
	VMULPS Y6, Y0, Y0
	VDIVPS Y8, Y0, Y0
	VMOVUPS Y0, 0(DI)

gelu_approx_avx2_done:
	VZEROUPPER
	RET

// ===========================================================================
// geluExactBiasTileFloat32AVX2Asm: Fused Bias + Exact GELU 2D tile for float32
//   Applies out[r*rowStrideBytes + c] = GeluExact(out[r*rowStrideBytes + c] + bias[c])
//   for r in [0, numRows) and c in [0, width).
//   width must be a positive multiple of 8.
// ===========================================================================
// func geluExactBiasTileFloat32AVX2Asm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)
TEXT ·geluExactBiasTileFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), R8
	MOVQ bias+8(FP), R9
	MOVQ rowStrideBytes+16(FP), R10
	MOVQ numRows+24(FP), R11
	MOVQ width+32(FP), R12

	CMPQ R11, $0
	JLE gelu_exact_bias_avx2_done
	CMPQ R12, $8
	JL gelu_exact_bias_avx2_done

	LEAQ ·geluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14 // Y14: vHalf / vP2 = 0.5
	VMOVUPS 192(CX), Y15 // Y15: vOne / vBias127 = 1.0

gelu_exact_bias_avx2_row_loop:
	MOVQ R8, DI
	MOVQ R9, SI
	MOVQ R12, DX

	CMPQ DX, $16
	JL gelu_exact_bias_avx2_vec8_check

gelu_exact_bias_avx2_loop16:
	VMOVUPS 0(DI), Y0
	VMOVUPS 32(DI), Y1
	VADDPS 0(SI), Y0, Y0
	VADDPS 32(SI), Y1, Y1

	// x = v * rsqrt2, absX = |x|
	VMULPS 224(CX), Y0, Y2
	VMULPS 224(CX), Y1, Y3
	VPAND 800(CX), Y2, Y2
	VPAND 800(CX), Y3, Y3

	// Issue t = 1.0 / (1.0 + p * absX) early so VDIVPS overlaps with exp(-x^2):
	VMOVAPS Y15, Y4
	VMOVAPS Y15, Y5
	VFMADD231PS 288(CX), Y2, Y4
	VFMADD231PS 288(CX), Y3, Y5
	VDIVPS Y4, Y15, Y4
	VDIVPS Y5, Y15, Y5

	// negX2 = max(-(absX * absX), vMinLog)
	VMULPS Y2, Y2, Y6
	VMULPS Y3, Y3, Y7
	VXORPS 256(CX), Y6, Y6
	VXORPS 256(CX), Y7, Y7
	VMAXPS 480(CX), Y6, Y6
	VMAXPS 480(CX), Y7, Y7

	// zRound = xClamped * log2e + magic, zFloor = zRound - magic
	VMOVUPS 544(CX), Y12
	VMOVAPS Y12, Y8
	VMOVAPS Y12, Y9
	VFMADD231PS 512(CX), Y6, Y8
	VFMADD231PS 512(CX), Y7, Y9
	VSUBPS Y12, Y8, Y10
	VSUBPS Y12, Y9, Y11

	// g = xClamped - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS 576(CX), Y10, Y6
	VFNMADD231PS 576(CX), Y11, Y7
	VFNMADD231PS 608(CX), Y10, Y6
	VFNMADD231PS 608(CX), Y11, Y7

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Y8, Y8
	VPSLLD $23, Y9, Y9
	VPADDD Y15, Y8, Y8
	VPADDD Y15, Y9, Y9

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Y6, Y6, Y2
	VMULPS Y7, Y7, Y3
	VMOVUPS 32(CX), Y10
	VMOVAPS Y10, Y11
	VFMADD231PS 0(CX), Y6, Y10
	VFMADD231PS 0(CX), Y7, Y11
	VMOVUPS 96(CX), Y12
	VMOVAPS Y12, Y13
	VFMADD231PS 64(CX), Y6, Y12
	VFMADD231PS 64(CX), Y7, Y13
	VFMADD231PS Y2, Y10, Y12
	VFMADD231PS Y3, Y11, Y13

	VMOVAPS Y14, Y10
	VMOVAPS Y14, Y11
	VFMADD231PS 128(CX), Y6, Y10
	VFMADD231PS 128(CX), Y7, Y11
	VADDPS Y15, Y6, Y6
	VADDPS Y15, Y7, Y7
	VFMADD231PS Y2, Y10, Y6
	VFMADD231PS Y3, Y11, Y7

	VMULPS Y2, Y2, Y2
	VMULPS Y3, Y3, Y3
	VFMADD231PS Y2, Y12, Y6
	VFMADD231PS Y3, Y13, Y7
	VMULPS Y8, Y6, Y8
	VMULPS Y9, Y7, Y9

	// Estrin's scheme for erfPoly(t) using Y4 (tA) and Y5 (tB):
	VMULPS Y4, Y4, Y2
	VMULPS Y5, Y5, Y3
	VMOVUPS 416(CX), Y10
	VMOVAPS Y10, Y11
	VFMADD231PS 448(CX), Y4, Y10
	VFMADD231PS 448(CX), Y5, Y11
	VMOVUPS 352(CX), Y12
	VMOVAPS Y12, Y13
	VFMADD231PS 384(CX), Y4, Y12
	VFMADD231PS 384(CX), Y5, Y13
	VFMADD231PS Y2, Y10, Y12
	VFMADD231PS Y3, Y11, Y13
	VFMADD213PS 320(CX), Y4, Y12
	VFMADD213PS 320(CX), Y5, Y13
	VMULPS Y4, Y12, Y12
	VMULPS Y5, Y13, Y13

	// res = 1.0 - erfPoly * expNegX2
	VMOVAPS Y15, Y6
	VMOVAPS Y15, Y7
	VFNMADD231PS Y12, Y8, Y6
	VFNMADD231PS Y13, Y9, Y7

	// Restore sign of v and compute 0.5 * v * (1.0 + erfVal)
	VPAND 256(CX), Y0, Y2
	VPAND 256(CX), Y1, Y3
	VXORPS Y2, Y6, Y6
	VXORPS Y3, Y7, Y7
	VADDPS Y15, Y6, Y6
	VADDPS Y15, Y7, Y7
	VMULPS Y14, Y0, Y0
	VMULPS Y14, Y1, Y1
	VMULPS Y6, Y0, Y0
	VMULPS Y7, Y1, Y1

	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE gelu_exact_bias_avx2_loop16

gelu_exact_bias_avx2_vec8_check:
	CMPQ DX, $8
	JL gelu_exact_bias_avx2_next_row

	VMOVUPS 0(DI), Y0
	VADDPS 0(SI), Y0, Y0
	VMULPS 224(CX), Y0, Y2
	VPAND 800(CX), Y2, Y2
	VMOVAPS Y15, Y4
	VFMADD231PS 288(CX), Y2, Y4
	VDIVPS Y4, Y15, Y4
	VMULPS Y2, Y2, Y6
	VXORPS 256(CX), Y6, Y6
	VMAXPS 480(CX), Y6, Y6
	VMOVUPS 544(CX), Y8
	VFMADD231PS 512(CX), Y6, Y8
	VSUBPS 544(CX), Y8, Y10
	VFNMADD231PS 576(CX), Y10, Y6
	VFNMADD231PS 608(CX), Y10, Y6
	VPSLLD $23, Y8, Y8
	VPADDD Y15, Y8, Y8
	VMULPS Y6, Y6, Y2
	VMOVUPS 32(CX), Y10
	VFMADD231PS 0(CX), Y6, Y10
	VMOVUPS 96(CX), Y12
	VFMADD231PS 64(CX), Y6, Y12
	VFMADD231PS Y2, Y10, Y12
	VMOVAPS Y14, Y10
	VFMADD231PS 128(CX), Y6, Y10
	VADDPS Y15, Y6, Y6
	VFMADD231PS Y2, Y10, Y6
	VMULPS Y2, Y2, Y2
	VFMADD231PS Y2, Y12, Y6
	VMULPS Y8, Y6, Y8
	VMULPS Y4, Y4, Y2
	VMOVUPS 416(CX), Y10
	VFMADD231PS 448(CX), Y4, Y10
	VMOVUPS 352(CX), Y12
	VFMADD231PS 384(CX), Y4, Y12
	VFMADD231PS Y2, Y10, Y12
	VFMADD213PS 320(CX), Y4, Y12
	VMULPS Y4, Y12, Y12
	VMOVAPS Y15, Y6
	VFNMADD231PS Y12, Y8, Y6
	VPAND 256(CX), Y0, Y2
	VXORPS Y2, Y6, Y6
	VADDPS Y15, Y6, Y6
	VMULPS Y14, Y0, Y0
	VMULPS Y6, Y0, Y0
	VMOVUPS Y0, 0(DI)

gelu_exact_bias_avx2_next_row:
	ADDQ R10, R8
	DECQ R11
	JNZ gelu_exact_bias_avx2_row_loop

gelu_exact_bias_avx2_done:
	VZEROUPPER
	RET

// ===========================================================================
// geluApproxBiasTileFloat32AVX2Asm: Fused Bias + Approx GELU 2D tile for float32
//   Applies out[r*rowStrideBytes + c] = GeluApprox(out[r*rowStrideBytes + c] + bias[c])
//   for r in [0, numRows) and c in [0, width).
//   width must be a positive multiple of 8.
// ===========================================================================
// func geluApproxBiasTileFloat32AVX2Asm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)
TEXT ·geluApproxBiasTileFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), R8
	MOVQ bias+8(FP), R9
	MOVQ rowStrideBytes+16(FP), R10
	MOVQ numRows+24(FP), R11
	MOVQ width+32(FP), R12

	CMPQ R11, $0
	JLE gelu_approx_bias_avx2_done
	CMPQ R12, $8
	JL gelu_approx_bias_avx2_done

	LEAQ ·geluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14 // Y14: vHalf / vP2 = 0.5
	VMOVUPS 192(CX), Y15 // Y15: vOne / vBias127 = 1.0

gelu_approx_bias_avx2_row_loop:
	MOVQ R8, DI
	MOVQ R9, SI
	MOVQ R12, DX

	CMPQ DX, $16
	JL gelu_approx_bias_avx2_vec8_check

gelu_approx_bias_avx2_loop16:
	VMOVUPS 0(DI), Y0
	VMOVUPS 32(DI), Y1
	VADDPS 0(SI), Y0, Y0
	VADDPS 32(SI), Y1, Y1

	// x3 = v * v * v
	VMULPS Y0, Y0, Y2
	VMULPS Y1, Y1, Y3
	VMULPS Y0, Y2, Y2
	VMULPS Y1, Y3, Y3

	// inner = vSqrt2OverPi * (v + vCoeff * x3)
	VMOVAPS Y0, Y4
	VMOVAPS Y1, Y5
	VFMADD231PS 768(CX), Y2, Y4
	VFMADD231PS 768(CX), Y3, Y5
	VMULPS 736(CX), Y4, Y4
	VMULPS 736(CX), Y5, Y5

	// Clamp inner to [-9.0, 9.0] and multiply by 2.0
	VMINPS 672(CX), Y4, Y4
	VMINPS 672(CX), Y5, Y5
	VMAXPS 704(CX), Y4, Y4
	VMAXPS 704(CX), Y5, Y5
	VMULPS 640(CX), Y4, Y4
	VMULPS 640(CX), Y5, Y5

	// Range reduction: zRound = twoX * log2e + magic, zFloor = zRound - magic
	VMOVUPS 544(CX), Y12
	VMOVAPS Y12, Y6
	VMOVAPS Y12, Y7
	VFMADD231PS 512(CX), Y4, Y6
	VFMADD231PS 512(CX), Y5, Y7
	VSUBPS Y12, Y6, Y8
	VSUBPS Y12, Y7, Y9

	// g = twoX - zFloor * ln2Hi - zFloor * ln2Lo
	VFNMADD231PS 576(CX), Y8, Y4
	VFNMADD231PS 576(CX), Y9, Y5
	VFNMADD231PS 608(CX), Y8, Y4
	VFNMADD231PS 608(CX), Y9, Y5

	// scale = (zRound << 23) + vBias127
	VPSLLD $23, Y6, Y6
	VPSLLD $23, Y7, Y7
	VPADDD Y15, Y6, Y6
	VPADDD Y15, Y7, Y7

	// Estrin's scheme for exp(g) across A and B:
	VMULPS Y4, Y4, Y2
	VMULPS Y5, Y5, Y3
	VMOVUPS 32(CX), Y8
	VMOVAPS Y8, Y9
	VFMADD231PS 0(CX), Y4, Y8
	VFMADD231PS 0(CX), Y5, Y9
	VMOVUPS 96(CX), Y10
	VMOVAPS Y10, Y11
	VFMADD231PS 64(CX), Y4, Y10
	VFMADD231PS 64(CX), Y5, Y11
	VMOVAPS Y14, Y12
	VMOVAPS Y14, Y13
	VFMADD231PS 128(CX), Y4, Y12
	VFMADD231PS 128(CX), Y5, Y13
	VADDPS Y15, Y4, Y4
	VADDPS Y15, Y5, Y5
	VFMADD231PS Y2, Y8, Y10
	VFMADD231PS Y3, Y9, Y11
	VFMADD231PS Y2, Y12, Y4
	VFMADD231PS Y3, Y13, Y5
	VMULPS Y2, Y2, Y2
	VMULPS Y3, Y3, Y3
	VFMADD231PS Y2, Y10, Y4
	VFMADD231PS Y3, Y11, Y5
	VMULPS Y6, Y4, Y6
	VMULPS Y7, Y5, Y7

	// out = (v * exp2x) / (exp2x + 1.0)
	VADDPS Y15, Y6, Y8
	VADDPS Y15, Y7, Y9
	VMULPS Y6, Y0, Y0
	VMULPS Y7, Y1, Y1
	VDIVPS Y8, Y0, Y0
	VDIVPS Y9, Y1, Y1

	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE gelu_approx_bias_avx2_loop16

gelu_approx_bias_avx2_vec8_check:
	CMPQ DX, $8
	JL gelu_approx_bias_avx2_next_row

	VMOVUPS 0(DI), Y0
	VADDPS 0(SI), Y0, Y0
	VMULPS Y0, Y0, Y2
	VMULPS Y0, Y2, Y2
	VMOVAPS Y0, Y4
	VFMADD231PS 768(CX), Y2, Y4
	VMULPS 736(CX), Y4, Y4
	VMINPS 672(CX), Y4, Y4
	VMAXPS 704(CX), Y4, Y4
	VMULPS 640(CX), Y4, Y4
	VMOVUPS 544(CX), Y6
	VFMADD231PS 512(CX), Y4, Y6
	VSUBPS 544(CX), Y6, Y8
	VFNMADD231PS 576(CX), Y8, Y4
	VFNMADD231PS 608(CX), Y8, Y4
	VPSLLD $23, Y6, Y6
	VPADDD Y15, Y6, Y6
	VMULPS Y4, Y4, Y2
	VMOVUPS 32(CX), Y8
	VFMADD231PS 0(CX), Y4, Y8
	VMOVUPS 96(CX), Y10
	VFMADD231PS 64(CX), Y4, Y10
	VMOVAPS Y14, Y12
	VFMADD231PS 128(CX), Y4, Y12
	VADDPS Y15, Y4, Y4
	VFMADD231PS Y2, Y8, Y10
	VFMADD231PS Y2, Y12, Y4
	VMULPS Y2, Y2, Y2
	VFMADD231PS Y2, Y10, Y4
	VMULPS Y6, Y4, Y6
	VADDPS Y15, Y6, Y8
	VMULPS Y6, Y0, Y0
	VDIVPS Y8, Y0, Y0
	VMOVUPS Y0, 0(DI)

gelu_approx_bias_avx2_next_row:
	ADDQ R10, R8
	DECQ R11
	JNZ gelu_approx_bias_avx2_row_loop

gelu_approx_bias_avx2_done:
	VZEROUPPER
	RET


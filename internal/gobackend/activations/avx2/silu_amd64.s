// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

#include "textflag.h"

// RODATA constants for AVX2 SiLU (Forward, BiasTile, and VJP)
// Each entry is 8 repeated float32/uint32 lanes (32 bytes = 256 bits) so it can be
// used directly as a 256-bit memory operand in VFMADD / VMULPS / VADDPS / VXORPS / etc.
GLOBL ·siluAVX2Consts(SB), RODATA|NOPTR, $448

//   0: vP7 = 1.9875691500e-4 (0x39506967)
DATA ·siluAVX2Consts+0(SB)/4, $0x39506967
DATA ·siluAVX2Consts+4(SB)/4, $0x39506967
DATA ·siluAVX2Consts+8(SB)/4, $0x39506967
DATA ·siluAVX2Consts+12(SB)/4, $0x39506967
DATA ·siluAVX2Consts+16(SB)/4, $0x39506967
DATA ·siluAVX2Consts+20(SB)/4, $0x39506967
DATA ·siluAVX2Consts+24(SB)/4, $0x39506967
DATA ·siluAVX2Consts+28(SB)/4, $0x39506967

//  32: vP6 = 1.3981999507e-3 (0x3ab743ce)
DATA ·siluAVX2Consts+32(SB)/4, $0x3ab743ce
DATA ·siluAVX2Consts+36(SB)/4, $0x3ab743ce
DATA ·siluAVX2Consts+40(SB)/4, $0x3ab743ce
DATA ·siluAVX2Consts+44(SB)/4, $0x3ab743ce
DATA ·siluAVX2Consts+48(SB)/4, $0x3ab743ce
DATA ·siluAVX2Consts+52(SB)/4, $0x3ab743ce
DATA ·siluAVX2Consts+56(SB)/4, $0x3ab743ce
DATA ·siluAVX2Consts+60(SB)/4, $0x3ab743ce

//  64: vP5 = 8.3334519073e-3 (0x3c088895)
DATA ·siluAVX2Consts+64(SB)/4, $0x3c088895
DATA ·siluAVX2Consts+68(SB)/4, $0x3c088895
DATA ·siluAVX2Consts+72(SB)/4, $0x3c088895
DATA ·siluAVX2Consts+76(SB)/4, $0x3c088895
DATA ·siluAVX2Consts+80(SB)/4, $0x3c088895
DATA ·siluAVX2Consts+84(SB)/4, $0x3c088895
DATA ·siluAVX2Consts+88(SB)/4, $0x3c088895
DATA ·siluAVX2Consts+92(SB)/4, $0x3c088895

//  96: vP4 = 4.1665795894e-2 (0x3d2aa9a5)
DATA ·siluAVX2Consts+96(SB)/4, $0x3d2aa9a5
DATA ·siluAVX2Consts+100(SB)/4, $0x3d2aa9a5
DATA ·siluAVX2Consts+104(SB)/4, $0x3d2aa9a5
DATA ·siluAVX2Consts+108(SB)/4, $0x3d2aa9a5
DATA ·siluAVX2Consts+112(SB)/4, $0x3d2aa9a5
DATA ·siluAVX2Consts+116(SB)/4, $0x3d2aa9a5
DATA ·siluAVX2Consts+120(SB)/4, $0x3d2aa9a5
DATA ·siluAVX2Consts+124(SB)/4, $0x3d2aa9a5

// 128: vP3 = 1.6666665459e-1 (0x3e2aaaab)
DATA ·siluAVX2Consts+128(SB)/4, $0x3e2aaaab
DATA ·siluAVX2Consts+132(SB)/4, $0x3e2aaaab
DATA ·siluAVX2Consts+136(SB)/4, $0x3e2aaaab
DATA ·siluAVX2Consts+140(SB)/4, $0x3e2aaaab
DATA ·siluAVX2Consts+144(SB)/4, $0x3e2aaaab
DATA ·siluAVX2Consts+148(SB)/4, $0x3e2aaaab
DATA ·siluAVX2Consts+152(SB)/4, $0x3e2aaaab
DATA ·siluAVX2Consts+156(SB)/4, $0x3e2aaaab

// 160: vP2 / vHalf = 0.5 (0x3f000000)
DATA ·siluAVX2Consts+160(SB)/4, $0x3f000000
DATA ·siluAVX2Consts+164(SB)/4, $0x3f000000
DATA ·siluAVX2Consts+168(SB)/4, $0x3f000000
DATA ·siluAVX2Consts+172(SB)/4, $0x3f000000
DATA ·siluAVX2Consts+176(SB)/4, $0x3f000000
DATA ·siluAVX2Consts+180(SB)/4, $0x3f000000
DATA ·siluAVX2Consts+184(SB)/4, $0x3f000000
DATA ·siluAVX2Consts+188(SB)/4, $0x3f000000

// 192: vOne / vBias127 = 1.0 (0x3f800000)
DATA ·siluAVX2Consts+192(SB)/4, $0x3f800000
DATA ·siluAVX2Consts+196(SB)/4, $0x3f800000
DATA ·siluAVX2Consts+200(SB)/4, $0x3f800000
DATA ·siluAVX2Consts+204(SB)/4, $0x3f800000
DATA ·siluAVX2Consts+208(SB)/4, $0x3f800000
DATA ·siluAVX2Consts+212(SB)/4, $0x3f800000
DATA ·siluAVX2Consts+216(SB)/4, $0x3f800000
DATA ·siluAVX2Consts+220(SB)/4, $0x3f800000

// 224: vSignMask = 0x80000000
DATA ·siluAVX2Consts+224(SB)/4, $0x80000000
DATA ·siluAVX2Consts+228(SB)/4, $0x80000000
DATA ·siluAVX2Consts+232(SB)/4, $0x80000000
DATA ·siluAVX2Consts+236(SB)/4, $0x80000000
DATA ·siluAVX2Consts+240(SB)/4, $0x80000000
DATA ·siluAVX2Consts+244(SB)/4, $0x80000000
DATA ·siluAVX2Consts+248(SB)/4, $0x80000000
DATA ·siluAVX2Consts+252(SB)/4, $0x80000000

// 256: vMinLog = -87.336544 (0xc2aeac50)
DATA ·siluAVX2Consts+256(SB)/4, $0xc2aeac50
DATA ·siluAVX2Consts+260(SB)/4, $0xc2aeac50
DATA ·siluAVX2Consts+264(SB)/4, $0xc2aeac50
DATA ·siluAVX2Consts+268(SB)/4, $0xc2aeac50
DATA ·siluAVX2Consts+272(SB)/4, $0xc2aeac50
DATA ·siluAVX2Consts+276(SB)/4, $0xc2aeac50
DATA ·siluAVX2Consts+280(SB)/4, $0xc2aeac50
DATA ·siluAVX2Consts+284(SB)/4, $0xc2aeac50

// 288: vMaxLog = 88.02969187 (0x42b00f34)
DATA ·siluAVX2Consts+288(SB)/4, $0x42b00f34
DATA ·siluAVX2Consts+292(SB)/4, $0x42b00f34
DATA ·siluAVX2Consts+296(SB)/4, $0x42b00f34
DATA ·siluAVX2Consts+300(SB)/4, $0x42b00f34
DATA ·siluAVX2Consts+304(SB)/4, $0x42b00f34
DATA ·siluAVX2Consts+308(SB)/4, $0x42b00f34
DATA ·siluAVX2Consts+312(SB)/4, $0x42b00f34
DATA ·siluAVX2Consts+316(SB)/4, $0x42b00f34

// 320: vLog2E = 1.44269504 (0x3fb8aa3b)
DATA ·siluAVX2Consts+320(SB)/4, $0x3fb8aa3b
DATA ·siluAVX2Consts+324(SB)/4, $0x3fb8aa3b
DATA ·siluAVX2Consts+328(SB)/4, $0x3fb8aa3b
DATA ·siluAVX2Consts+332(SB)/4, $0x3fb8aa3b
DATA ·siluAVX2Consts+336(SB)/4, $0x3fb8aa3b
DATA ·siluAVX2Consts+340(SB)/4, $0x3fb8aa3b
DATA ·siluAVX2Consts+344(SB)/4, $0x3fb8aa3b
DATA ·siluAVX2Consts+348(SB)/4, $0x3fb8aa3b

// 352: vMagic = 12582912.0 (0x4b400000)
DATA ·siluAVX2Consts+352(SB)/4, $0x4b400000
DATA ·siluAVX2Consts+356(SB)/4, $0x4b400000
DATA ·siluAVX2Consts+360(SB)/4, $0x4b400000
DATA ·siluAVX2Consts+364(SB)/4, $0x4b400000
DATA ·siluAVX2Consts+368(SB)/4, $0x4b400000
DATA ·siluAVX2Consts+372(SB)/4, $0x4b400000
DATA ·siluAVX2Consts+376(SB)/4, $0x4b400000
DATA ·siluAVX2Consts+380(SB)/4, $0x4b400000

// 384: vLn2Hi = 0.693359375 (0x3f318000)
DATA ·siluAVX2Consts+384(SB)/4, $0x3f318000
DATA ·siluAVX2Consts+388(SB)/4, $0x3f318000
DATA ·siluAVX2Consts+392(SB)/4, $0x3f318000
DATA ·siluAVX2Consts+396(SB)/4, $0x3f318000
DATA ·siluAVX2Consts+400(SB)/4, $0x3f318000
DATA ·siluAVX2Consts+404(SB)/4, $0x3f318000
DATA ·siluAVX2Consts+408(SB)/4, $0x3f318000
DATA ·siluAVX2Consts+412(SB)/4, $0x3f318000

// 416: vLn2Lo = -2.12194440e-4 (0xb95e8083)
DATA ·siluAVX2Consts+416(SB)/4, $0xb95e8083
DATA ·siluAVX2Consts+420(SB)/4, $0xb95e8083
DATA ·siluAVX2Consts+424(SB)/4, $0xb95e8083
DATA ·siluAVX2Consts+428(SB)/4, $0xb95e8083
DATA ·siluAVX2Consts+432(SB)/4, $0xb95e8083
DATA ·siluAVX2Consts+436(SB)/4, $0xb95e8083
DATA ·siluAVX2Consts+440(SB)/4, $0xb95e8083
DATA ·siluAVX2Consts+444(SB)/4, $0xb95e8083

// ===========================================================================
// siluFloat32AVX2Asm: SiLU (Swish) forward for float32 using AVX2
//   SiLU(x) = x / (1 + exp(-x))
//   Uses 2-vector (16 float32) unrolling, magic-number 2^n scaling, and
//   Estrin's degree-7 polynomial tree with fused denominator FMA.
//   n must be a positive multiple of 8.
// ===========================================================================
// func siluFloat32AVX2Asm(in, out unsafe.Pointer, n int)
TEXT ·siluFloat32AVX2Asm(SB), NOSPLIT, $0-24
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ n+16(FP), DX

	CMPQ DX, $8
	JL silu_avx2_done

	LEAQ ·siluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14 // Y14: vP2 = 0.5 (0x3f000000)
	VMOVUPS 192(CX), Y15 // Y15: vOne / vBias127 = 1.0 (0x3f800000)

	CMPQ DX, $16
	JL silu_avx2_vec8_check

silu_avx2_loop16:
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1

	// u = clamp(-x, vMinLog, vMaxLog)
	VXORPS 224(CX), Y0, Y4
	VXORPS 224(CX), Y1, Y5
	VMAXPS 256(CX), Y4, Y4
	VMAXPS 256(CX), Y5, Y5
	VMINPS 288(CX), Y4, Y4
	VMINPS 288(CX), Y5, Y5

	// Range reduction: zRound = u * vLog2E + vMagic, zFloor = zRound - vMagic
	VMOVUPS 352(CX), Y12
	VMOVAPS Y12, Y6
	VMOVAPS Y12, Y7
	VFMADD231PS 320(CX), Y4, Y6
	VFMADD231PS 320(CX), Y5, Y7
	VSUBPS Y12, Y6, Y8
	VSUBPS Y12, Y7, Y9

	// g = u - zFloor * vLn2Hi - zFloor * vLn2Lo
	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 384(CX), Y9, Y5
	VFNMADD231PS 416(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y9, Y5

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

	// denom = 1.0 + scale * expPoly
	VMOVAPS Y15, Y8
	VMOVAPS Y15, Y9
	VFMADD231PS Y6, Y4, Y8
	VFMADD231PS Y7, Y5, Y9

	// out = x / denom
	VDIVPS Y8, Y0, Y0
	VDIVPS Y9, Y1, Y1

	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE silu_avx2_loop16

silu_avx2_vec8_check:
	CMPQ DX, $8
	JL silu_avx2_done

	VMOVUPS 0(SI), Y0
	VXORPS 224(CX), Y0, Y4
	VMAXPS 256(CX), Y4, Y4
	VMINPS 288(CX), Y4, Y4

	VMOVUPS 352(CX), Y6
	VFMADD231PS 320(CX), Y4, Y6
	VSUBPS 352(CX), Y6, Y8
	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y8, Y4

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

	VMOVAPS Y15, Y8
	VFMADD231PS Y6, Y4, Y8
	VDIVPS Y8, Y0, Y0
	VMOVUPS Y0, 0(DI)

silu_avx2_done:
	VZEROUPPER
	RET

// ===========================================================================
// siluBiasTileFloat32AVX2Asm: Fused Bias + SiLU 2D tile for float32
//   Applies out[r*rowStrideBytes + c] = SiLU(out[r*rowStrideBytes + c] + bias[c])
//   for r in [0, numRows) and c in [0, width).
//   width must be a positive multiple of 8.
// ===========================================================================
// func siluBiasTileFloat32AVX2Asm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)
TEXT ·siluBiasTileFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ out+0(FP), R8
	MOVQ bias+8(FP), R9
	MOVQ rowStrideBytes+16(FP), R10
	MOVQ numRows+24(FP), R11
	MOVQ width+32(FP), R12

	CMPQ R11, $0
	JLE silu_bias_avx2_done
	CMPQ R12, $8
	JL silu_bias_avx2_done

	LEAQ ·siluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14 // Y14: vP2 = 0.5
	VMOVUPS 192(CX), Y15 // Y15: vOne / vBias127 = 1.0

silu_bias_avx2_row_loop:
	MOVQ R8, DI
	MOVQ R9, SI
	MOVQ R12, DX

	CMPQ DX, $16
	JL silu_bias_avx2_vec8_check

silu_bias_avx2_loop16:
	VMOVUPS 0(DI), Y0
	VMOVUPS 32(DI), Y1
	VADDPS 0(SI), Y0, Y0
	VADDPS 32(SI), Y1, Y1

	VXORPS 224(CX), Y0, Y4
	VXORPS 224(CX), Y1, Y5
	VMAXPS 256(CX), Y4, Y4
	VMAXPS 256(CX), Y5, Y5
	VMINPS 288(CX), Y4, Y4
	VMINPS 288(CX), Y5, Y5

	VMOVUPS 352(CX), Y12
	VMOVAPS Y12, Y6
	VMOVAPS Y12, Y7
	VFMADD231PS 320(CX), Y4, Y6
	VFMADD231PS 320(CX), Y5, Y7
	VSUBPS Y12, Y6, Y8
	VSUBPS Y12, Y7, Y9

	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 384(CX), Y9, Y5
	VFNMADD231PS 416(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y9, Y5

	VPSLLD $23, Y6, Y6
	VPSLLD $23, Y7, Y7
	VPADDD Y15, Y6, Y6
	VPADDD Y15, Y7, Y7

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

	VMOVAPS Y15, Y8
	VMOVAPS Y15, Y9
	VFMADD231PS Y6, Y4, Y8
	VFMADD231PS Y7, Y5, Y9

	VDIVPS Y8, Y0, Y0
	VDIVPS Y9, Y1, Y1

	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE silu_bias_avx2_loop16

silu_bias_avx2_vec8_check:
	CMPQ DX, $8
	JL silu_bias_avx2_next_row

	VMOVUPS 0(DI), Y0
	VADDPS 0(SI), Y0, Y0
	VXORPS 224(CX), Y0, Y4
	VMAXPS 256(CX), Y4, Y4
	VMINPS 288(CX), Y4, Y4

	VMOVUPS 352(CX), Y6
	VFMADD231PS 320(CX), Y4, Y6
	VSUBPS 352(CX), Y6, Y8
	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y8, Y4

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

	VMOVAPS Y15, Y8
	VFMADD231PS Y6, Y4, Y8
	VDIVPS Y8, Y0, Y0
	VMOVUPS Y0, 0(DI)

silu_bias_avx2_next_row:
	ADDQ R10, R8
	DECQ R11
	JNZ silu_bias_avx2_row_loop

silu_bias_avx2_done:
	VZEROUPPER
	RET

// ===========================================================================
// siluVJPFloat32AVX2Asm: SiLU (Swish) backward (VJP) for float32 using AVX2
//   sig = 1 / (1 + exp(-x))
//   dx = (dOut * sig) * ((1 + x) - x * sig)
//   n must be a positive multiple of 8.
// ===========================================================================
// func siluVJPFloat32AVX2Asm(x, dOut, dx unsafe.Pointer, n int)
TEXT ·siluVJPFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ x+0(FP), SI
	MOVQ dOut+8(FP), R8
	MOVQ dx+16(FP), DI
	MOVQ n+24(FP), DX

	CMPQ DX, $8
	JL silu_vjp_avx2_done

	LEAQ ·siluAVX2Consts(SB), CX
	VMOVUPS 160(CX), Y14 // Y14: vP2 = 0.5
	VMOVUPS 192(CX), Y15 // Y15: vOne / vBias127 = 1.0

	CMPQ DX, $16
	JL silu_vjp_avx2_vec8_check

silu_vjp_avx2_loop16:
	VMOVUPS 0(SI), Y0
	VMOVUPS 32(SI), Y1

	// u = clamp(-x, vMinLog, vMaxLog)
	VXORPS 224(CX), Y0, Y4
	VXORPS 224(CX), Y1, Y5
	VMAXPS 256(CX), Y4, Y4
	VMAXPS 256(CX), Y5, Y5
	VMINPS 288(CX), Y4, Y4
	VMINPS 288(CX), Y5, Y5

	// Range reduction: zRound = u * vLog2E + vMagic, zFloor = zRound - vMagic
	VMOVUPS 352(CX), Y12
	VMOVAPS Y12, Y6
	VMOVAPS Y12, Y7
	VFMADD231PS 320(CX), Y4, Y6
	VFMADD231PS 320(CX), Y5, Y7
	VSUBPS Y12, Y6, Y8
	VSUBPS Y12, Y7, Y9

	// g = u - zFloor * vLn2Hi - zFloor * vLn2Lo
	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 384(CX), Y9, Y5
	VFNMADD231PS 416(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y9, Y5

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

	// denom = 1.0 + scale * expPoly
	VMOVAPS Y15, Y8
	VMOVAPS Y15, Y9
	VFMADD231PS Y6, Y4, Y8
	VFMADD231PS Y7, Y5, Y9

	// sig = 1.0 / denom; overlap (1.0 + x) and dOut loads with VDIVPS latency
	VDIVPS Y8, Y15, Y8
	VDIVPS Y9, Y15, Y9
	VADDPS Y15, Y0, Y2
	VADDPS Y15, Y1, Y3
	VMOVUPS 0(R8), Y10
	VMOVUPS 32(R8), Y11

	// term = (1.0 + x) - x * sig; dOutSig = dOut * sig; dx = dOutSig * term
	VFNMADD231PS Y0, Y8, Y2
	VFNMADD231PS Y1, Y9, Y3
	VMULPS Y8, Y10, Y8
	VMULPS Y9, Y11, Y9
	VMULPS Y8, Y2, Y2
	VMULPS Y9, Y3, Y3

	VMOVUPS Y2, 0(DI)
	VMOVUPS Y3, 32(DI)

	ADDQ $64, SI
	ADDQ $64, R8
	ADDQ $64, DI
	SUBQ $16, DX
	CMPQ DX, $16
	JGE silu_vjp_avx2_loop16

silu_vjp_avx2_vec8_check:
	CMPQ DX, $8
	JL silu_vjp_avx2_done

	VMOVUPS 0(SI), Y0
	VXORPS 224(CX), Y0, Y4
	VMAXPS 256(CX), Y4, Y4
	VMINPS 288(CX), Y4, Y4

	VMOVUPS 352(CX), Y6
	VFMADD231PS 320(CX), Y4, Y6
	VSUBPS 352(CX), Y6, Y8
	VFNMADD231PS 384(CX), Y8, Y4
	VFNMADD231PS 416(CX), Y8, Y4

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

	VMOVAPS Y15, Y8
	VFMADD231PS Y6, Y4, Y8
	VDIVPS Y8, Y15, Y8
	VADDPS Y15, Y0, Y2
	VMOVUPS 0(R8), Y10
	VFNMADD231PS Y0, Y8, Y2
	VMULPS Y8, Y10, Y8
	VMULPS Y8, Y2, Y2
	VMOVUPS Y2, 0(DI)

silu_vjp_avx2_done:
	VZEROUPPER
	RET

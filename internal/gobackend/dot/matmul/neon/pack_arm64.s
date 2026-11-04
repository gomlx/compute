// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// ARM64 NEON vector interleave (ZIP1 / ZIP2) encodings:
// ZIP1 Vd.8H, Vn.8H, Vm.8H
#define ZIP1_8H(Vd, Vn, Vm) WORD $(0x4E403800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
// ZIP2 Vd.8H, Vn.8H, Vm.8H
#define ZIP2_8H(Vd, Vn, Vm) WORD $(0x4E407800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
// ZIP1 Vd.4S, Vn.4S, Vm.4S
#define ZIP1_4S(Vd, Vn, Vm) WORD $(0x4E803800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
// ZIP2 Vd.4S, Vn.4S, Vm.4S
#define ZIP2_4S(Vd, Vn, Vm) WORD $(0x4E807800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
// ZIP1 Vd.2D, Vn.2D, Vm.2D
#define ZIP1_2D(Vd, Vn, Vm) WORD $(0x4EC03800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))
// ZIP2 Vd.2D, Vn.2D, Vm.2D
#define ZIP2_2D(Vd, Vn, Vm) WORD $(0x4EC07800 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// TRANSPOSE_4x4_32 transposes 4x4 32-bit elements in (r0, r1, r2, r3) into (c0, c1, c2, c3)
// using scratch registers (t0, t1, t2, t3).
#define TRANSPOSE_4x4_32(r0, r1, r2, r3, c0, c1, c2, c3, t0, t1, t2, t3) \
	ZIP1_4S(t0, r0, r1); \
	ZIP2_4S(t1, r0, r1); \
	ZIP1_4S(t2, r2, r3); \
	ZIP2_4S(t3, r2, r3); \
	ZIP1_2D(c0, t0, t2); \
	ZIP2_2D(c1, t0, t2); \
	ZIP1_2D(c2, t1, t3); \
	ZIP2_2D(c3, t1, t3)

// func neonPackRHSFullStripsAsm(rhsPtr, panelPtr unsafe.Pointer, rhsStrideBytes uintptr, contractingRows, numFullStrips, kernelColsBytes int)
TEXT ·neonPackRHSFullStripsAsm(SB), NOSPLIT, $0-48
	MOVD rhsPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD rhsStrideBytes+16(FP), R2
	MOVD contractingRows+24(FP), R3
	MOVD numFullStrips+32(FP), R4
	MOVD kernelColsBytes+40(FP), R5

	CBZ R4, rhs_done
	CBZ R3, rhs_done

	CMP $256, R5
	BEQ rhs_strip_256
	CMP $128, R5
	BEQ rhs_strip_128
	CMP $64, R5
	BEQ rhs_strip_64
	CMP $32, R5
	BEQ rhs_strip_32
	CMP $16, R5
	BEQ rhs_strip_16
	RET

rhs_strip_256:
	MOVD R0, R6
	MOVD R3, R7
rhs_256_loop1:
	CBZ R7, rhs_256_next_strip
	VLD1 (R6), [V0.B16, V1.B16, V2.B16, V3.B16]
	ADD $64, R6, R11
	VLD1 (R11), [V4.B16, V5.B16, V6.B16, V7.B16]
	ADD $128, R6, R11
	VLD1 (R11), [V8.B16, V9.B16, V10.B16, V11.B16]
	ADD $192, R6, R11
	VLD1 (R11), [V12.B16, V13.B16, V14.B16, V15.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	VST1.P [V4.B16, V5.B16, V6.B16, V7.B16], 64(R1)
	VST1.P [V8.B16, V9.B16, V10.B16, V11.B16], 64(R1)
	VST1.P [V12.B16, V13.B16, V14.B16, V15.B16], 64(R1)
	SUB $1, R7, R7
	B rhs_256_loop1

rhs_256_next_strip:
	ADD $256, R0, R0
	SUB $1, R4, R4
	CBNZ R4, rhs_strip_256
	RET

rhs_strip_128:
	MOVD R0, R6
	MOVD R3, R7
rhs_128_loop2:
	CMP $2, R7
	BLT rhs_128_loop1
	VLD1 (R6), [V0.B16, V1.B16, V2.B16, V3.B16]
	ADD $64, R6, R11
	VLD1 (R11), [V4.B16, V5.B16, V6.B16, V7.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V8.B16, V9.B16, V10.B16, V11.B16]
	ADD $64, R6, R11
	VLD1 (R11), [V12.B16, V13.B16, V14.B16, V15.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	VST1.P [V4.B16, V5.B16, V6.B16, V7.B16], 64(R1)
	VST1.P [V8.B16, V9.B16, V10.B16, V11.B16], 64(R1)
	VST1.P [V12.B16, V13.B16, V14.B16, V15.B16], 64(R1)
	SUB $2, R7, R7
	B rhs_128_loop2

rhs_128_loop1:
	CBZ R7, rhs_128_next_strip
	VLD1 (R6), [V0.B16, V1.B16, V2.B16, V3.B16]
	ADD $64, R6, R11
	VLD1 (R11), [V4.B16, V5.B16, V6.B16, V7.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	VST1.P [V4.B16, V5.B16, V6.B16, V7.B16], 64(R1)
	SUB $1, R7, R7
	B rhs_128_loop1

rhs_128_next_strip:
	ADD $128, R0, R0
	SUB $1, R4, R4
	CBNZ R4, rhs_strip_128
	RET

rhs_strip_64:
	MOVD R0, R6
	MOVD R3, R7
rhs_64_loop4:
	CMP $4, R7
	BLT rhs_64_loop1
	VLD1 (R6), [V0.B16, V1.B16, V2.B16, V3.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V4.B16, V5.B16, V6.B16, V7.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V8.B16, V9.B16, V10.B16, V11.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V12.B16, V13.B16, V14.B16, V15.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	VST1.P [V4.B16, V5.B16, V6.B16, V7.B16], 64(R1)
	VST1.P [V8.B16, V9.B16, V10.B16, V11.B16], 64(R1)
	VST1.P [V12.B16, V13.B16, V14.B16, V15.B16], 64(R1)
	SUB $4, R7, R7
	B rhs_64_loop4

rhs_64_loop1:
	CBZ R7, rhs_64_next_strip
	VLD1 (R6), [V0.B16, V1.B16, V2.B16, V3.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	SUB $1, R7, R7
	B rhs_64_loop1

rhs_64_next_strip:
	ADD $64, R0, R0
	SUB $1, R4, R4
	CBNZ R4, rhs_strip_64
	RET

rhs_strip_32:
	MOVD R0, R6
	MOVD R3, R7
rhs_32_loop4:
	CMP $4, R7
	BLT rhs_32_loop1
	VLD1 (R6), [V0.B16, V1.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V2.B16, V3.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V4.B16, V5.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V6.B16, V7.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	VST1.P [V4.B16, V5.B16, V6.B16, V7.B16], 64(R1)
	SUB $4, R7, R7
	B rhs_32_loop4

rhs_32_loop1:
	CBZ R7, rhs_32_next_strip
	VLD1 (R6), [V0.B16, V1.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16], 32(R1)
	SUB $1, R7, R7
	B rhs_32_loop1

rhs_32_next_strip:
	ADD $32, R0, R0
	SUB $1, R4, R4
	CBNZ R4, rhs_strip_32
	RET

rhs_strip_16:
	MOVD R0, R6
	MOVD R3, R7
rhs_16_loop4:
	CMP $4, R7
	BLT rhs_16_loop1
	VLD1 (R6), [V0.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V1.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V2.B16]
	ADD R2, R6, R6
	VLD1 (R6), [V3.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R1)
	SUB $4, R7, R7
	B rhs_16_loop4

rhs_16_loop1:
	CBZ R7, rhs_16_next_strip
	VLD1 (R6), [V0.B16]
	ADD R2, R6, R6
	VST1.P [V0.B16], 16(R1)
	SUB $1, R7, R7
	B rhs_16_loop1

rhs_16_next_strip:
	ADD $16, R0, R0
	SUB $1, R4, R4
	CBNZ R4, rhs_strip_16

rhs_done:
	RET


// func neonPackLHSKernelRows4Float32Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes uintptr, fullRows, contractingCols int)
TEXT ·neonPackLHSKernelRows4Float32Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD fullRows+24(FP), R3
	MOVD contractingCols+32(FP), R8

	CMP $4, R3
	BLT lhs4_f32_done
	CBZ R8, lhs4_f32_done

	LSL $2, R2, R9 // R9 = 4 * lhsStrideBytes

lhs4_f32_strip_loop:
	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	MOVD R8, R10

lhs4_f32_col16:
	CMP $8, R10
	BLT lhs4_f32_col4

	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R5), [V1.S4]
	VLD1.P 16(R6), [V2.S4]
	VLD1.P 16(R7), [V3.S4]
	VLD1.P 16(R4), [V4.S4]
	VLD1.P 16(R5), [V5.S4]
	VLD1.P 16(R6), [V6.S4]
	VLD1.P 16(R7), [V7.S4]

	TRANSPOSE_4x4_32(0, 1, 2, 3, 16, 17, 18, 19, 24, 25, 26, 27)
	TRANSPOSE_4x4_32(4, 5, 6, 7, 20, 21, 22, 23, 28, 29, 30, 31)

	VST1.P [V16.S4], 16(R1)
	VST1.P [V17.S4], 16(R1)
	VST1.P [V18.S4], 16(R1)
	VST1.P [V19.S4], 16(R1)
	VST1.P [V20.S4], 16(R1)
	VST1.P [V21.S4], 16(R1)
	VST1.P [V22.S4], 16(R1)
	VST1.P [V23.S4], 16(R1)

	SUB $8, R10, R10
	B lhs4_f32_col16

lhs4_f32_col4:
	CMP $4, R10
	BLT lhs4_f32_col1

	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R5), [V1.S4]
	VLD1.P 16(R6), [V2.S4]
	VLD1.P 16(R7), [V3.S4]

	TRANSPOSE_4x4_32(0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23)
	VST1.P [V16.S4], 16(R1)
	VST1.P [V17.S4], 16(R1)
	VST1.P [V18.S4], 16(R1)
	VST1.P [V19.S4], 16(R1)

	SUB $4, R10, R10
	B lhs4_f32_col4

lhs4_f32_col1:
	CBZ R10, lhs4_f32_next_strip
	FMOVS.P 4(R4), F0
	FMOVS.P 4(R5), F1
	FMOVS.P 4(R6), F2
	FMOVS.P 4(R7), F3
	FMOVS F0, 0(R1)
	FMOVS F1, 4(R1)
	FMOVS F2, 8(R1)
	FMOVS F3, 12(R1)
	ADD $16, R1, R1
	SUB $1, R10, R10
	B lhs4_f32_col1

lhs4_f32_next_strip:
	ADD R9, R0, R0
	SUB $4, R3, R3
	CMP $4, R3
	BGE lhs4_f32_strip_loop

lhs4_f32_done:
	RET


// func neonPackLHSKernelRows8Float32Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes uintptr, fullRows, contractingCols int)
TEXT ·neonPackLHSKernelRows8Float32Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD fullRows+24(FP), R3
	MOVD contractingCols+32(FP), R8

	CMP $8, R3
	BLT lhs8_f32_done
	CBZ R8, lhs8_f32_done

	LSL $3, R2, R9 // R9 = 8 * lhsStrideBytes

lhs8_f32_strip_loop:
	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	ADD R2, R7, R11
	ADD R2, R11, R12
	ADD R2, R12, R13
	ADD R2, R13, R14
	MOVD R8, R10

lhs8_f32_col4:
	CMP $4, R10
	BLT lhs8_f32_col1

	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R5), [V1.S4]
	VLD1.P 16(R6), [V2.S4]
	VLD1.P 16(R7), [V3.S4]
	VLD1.P 16(R11), [V4.S4]
	VLD1.P 16(R12), [V5.S4]
	VLD1.P 16(R13), [V6.S4]
	VLD1.P 16(R14), [V7.S4]

	TRANSPOSE_4x4_32(0, 1, 2, 3, 16, 17, 18, 19, 24, 25, 26, 27)
	TRANSPOSE_4x4_32(4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26, 27)

	// Store [col0_top, col0_bot, col1_top, col1_bot, col2_top, col2_bot, col3_top, col3_bot]
	VST1.P [V16.S4], 16(R1)
	VST1.P [V20.S4], 16(R1)
	VST1.P [V17.S4], 16(R1)
	VST1.P [V21.S4], 16(R1)
	VST1.P [V18.S4], 16(R1)
	VST1.P [V22.S4], 16(R1)
	VST1.P [V19.S4], 16(R1)
	VST1.P [V23.S4], 16(R1)

	SUB $4, R10, R10
	B lhs8_f32_col4

lhs8_f32_col1:
	CBZ R10, lhs8_f32_next_strip
	FMOVS.P 4(R4), F0
	FMOVS.P 4(R5), F1
	FMOVS.P 4(R6), F2
	FMOVS.P 4(R7), F3
	FMOVS.P 4(R11), F4
	FMOVS.P 4(R12), F5
	FMOVS.P 4(R13), F6
	FMOVS.P 4(R14), F7
	FMOVS F0, 0(R1)
	FMOVS F1, 4(R1)
	FMOVS F2, 8(R1)
	FMOVS F3, 12(R1)
	FMOVS F4, 16(R1)
	FMOVS F5, 20(R1)
	FMOVS F6, 24(R1)
	FMOVS F7, 28(R1)
	ADD $32, R1, R1
	SUB $1, R10, R10
	B lhs8_f32_col1

lhs8_f32_next_strip:
	ADD R9, R0, R0
	SUB $8, R3, R3
	CMP $8, R3
	BGE lhs8_f32_strip_loop

lhs8_f32_done:
	RET


// func neonPackLHSKernelRows4Uint16Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes uintptr, fullRows, contractingCols int)
TEXT ·neonPackLHSKernelRows4Uint16Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD fullRows+24(FP), R3
	MOVD contractingCols+32(FP), R8

	CMP $4, R3
	BLT lhs4_u16_done
	CBZ R8, lhs4_u16_done

	LSL $2, R2, R9 // R9 = 4 * lhsStrideBytes

lhs4_u16_strip_loop:
	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	MOVD R8, R10

lhs4_u16_col8:
	CMP $8, R10
	BLT lhs4_u16_col1

	VLD1.P 16(R4), [V0.H8]
	VLD1.P 16(R5), [V1.H8]
	VLD1.P 16(R6), [V2.H8]
	VLD1.P 16(R7), [V3.H8]

	ZIP1_8H(16, 0, 1)
	ZIP2_8H(17, 0, 1)
	ZIP1_8H(18, 2, 3)
	ZIP2_8H(19, 2, 3)
	ZIP1_4S(20, 16, 18)
	ZIP2_4S(21, 16, 18)
	ZIP1_4S(22, 17, 19)
	ZIP2_4S(23, 17, 19)

	VST1.P [V20.B16, V21.B16, V22.B16, V23.B16], 64(R1)

	SUB $8, R10, R10
	B lhs4_u16_col8

lhs4_u16_col1:
	CBZ R10, lhs4_u16_next_strip
	MOVHU.P 2(R4), R11
	MOVHU.P 2(R5), R12
	MOVHU.P 2(R6), R13
	MOVHU.P 2(R7), R14
	MOVH R11, 0(R1)
	MOVH R12, 2(R1)
	MOVH R13, 4(R1)
	MOVH R14, 6(R1)
	ADD $8, R1, R1
	SUB $1, R10, R10
	B lhs4_u16_col1

lhs4_u16_next_strip:
	ADD R9, R0, R0
	SUB $4, R3, R3
	CMP $4, R3
	BGE lhs4_u16_strip_loop

lhs4_u16_done:
	RET


// func neonPackLHSKernelRows8Uint16Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes uintptr, fullRows, contractingCols int)
TEXT ·neonPackLHSKernelRows8Uint16Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD fullRows+24(FP), R3
	MOVD contractingCols+32(FP), R8

	CMP $8, R3
	BLT lhs8_u16_done
	CBZ R8, lhs8_u16_done

	LSL $3, R2, R9 // R9 = 8 * lhsStrideBytes

lhs8_u16_strip_loop:
	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	ADD R2, R7, R11
	ADD R2, R11, R12
	ADD R2, R12, R13
	ADD R2, R13, R14
	MOVD R8, R10

lhs8_u16_col8:
	CMP $8, R10
	BLT lhs8_u16_col1

	VLD1.P 16(R4), [V0.H8]
	VLD1.P 16(R5), [V1.H8]
	VLD1.P 16(R6), [V2.H8]
	VLD1.P 16(R7), [V3.H8]
	VLD1.P 16(R11), [V4.H8]
	VLD1.P 16(R12), [V5.H8]
	VLD1.P 16(R13), [V6.H8]
	VLD1.P 16(R14), [V7.H8]

	// Top 4x8 into V16..V19:
	ZIP1_8H(8, 0, 1)
	ZIP2_8H(9, 0, 1)
	ZIP1_8H(10, 2, 3)
	ZIP2_8H(11, 2, 3)
	ZIP1_4S(16, 8, 10)
	ZIP2_4S(17, 8, 10)
	ZIP1_4S(18, 9, 11)
	ZIP2_4S(19, 9, 11)

	// Bottom 4x8 into V20..V23:
	ZIP1_8H(12, 4, 5)
	ZIP2_8H(13, 4, 5)
	ZIP1_8H(14, 6, 7)
	ZIP2_8H(15, 6, 7)
	ZIP1_4S(20, 12, 14)
	ZIP2_4S(21, 12, 14)
	ZIP1_4S(22, 13, 15)
	ZIP2_4S(23, 13, 15)

	// Combine top and bottom 64-bit halves into 8 columns V24..V31:
	ZIP1_2D(24, 16, 20)
	ZIP2_2D(25, 16, 20)
	ZIP1_2D(26, 17, 21)
	ZIP2_2D(27, 17, 21)
	ZIP1_2D(28, 18, 22)
	ZIP2_2D(29, 18, 22)
	ZIP1_2D(30, 19, 23)
	ZIP2_2D(31, 19, 23)

	VST1.P [V24.B16, V25.B16, V26.B16, V27.B16], 64(R1)
	VST1.P [V28.B16, V29.B16, V30.B16, V31.B16], 64(R1)

	SUB $8, R10, R10
	B lhs8_u16_col8

lhs8_u16_col1:
	CBZ R10, lhs8_u16_next_strip
	MOVHU.P 2(R4), R15
	MOVH R15, 0(R1)
	MOVHU.P 2(R5), R15
	MOVH R15, 2(R1)
	MOVHU.P 2(R6), R15
	MOVH R15, 4(R1)
	MOVHU.P 2(R7), R15
	MOVH R15, 6(R1)
	MOVHU.P 2(R11), R15
	MOVH R15, 8(R1)
	MOVHU.P 2(R12), R15
	MOVH R15, 10(R1)
	MOVHU.P 2(R13), R15
	MOVH R15, 12(R1)
	MOVHU.P 2(R14), R15
	MOVH R15, 14(R1)
	ADD $16, R1, R1
	SUB $1, R10, R10
	B lhs8_u16_col1

lhs8_u16_next_strip:
	ADD R9, R0, R0
	SUB $8, R3, R3
	CMP $8, R3
	BGE lhs8_u16_strip_loop

lhs8_u16_done:
	RET


// func neonPackLHSKernelRows4Float64Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes uintptr, fullRows, contractingCols int)
TEXT ·neonPackLHSKernelRows4Float64Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD fullRows+24(FP), R3
	MOVD contractingCols+32(FP), R8

	CMP $4, R3
	BLT lhs4_f64_done
	CBZ R8, lhs4_f64_done

	LSL $2, R2, R9 // R9 = 4 * lhsStrideBytes

lhs4_f64_strip_loop:
	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	MOVD R8, R10

lhs4_f64_col2:
	CMP $2, R10
	BLT lhs4_f64_col1

	VLD1.P 16(R4), [V0.D2]
	VLD1.P 16(R5), [V1.D2]
	VLD1.P 16(R6), [V2.D2]
	VLD1.P 16(R7), [V3.D2]

	ZIP1_2D(16, 0, 1)
	ZIP1_2D(17, 2, 3)
	ZIP2_2D(18, 0, 1)
	ZIP2_2D(19, 2, 3)

	VST1.P [V16.D2, V17.D2, V18.D2, V19.D2], 64(R1)

	SUB $2, R10, R10
	B lhs4_f64_col2

lhs4_f64_col1:
	CBZ R10, lhs4_f64_next_strip
	FMOVD.P 8(R4), F0
	FMOVD.P 8(R5), F1
	FMOVD.P 8(R6), F2
	FMOVD.P 8(R7), F3
	FMOVD F0, 0(R1)
	FMOVD F1, 8(R1)
	FMOVD F2, 16(R1)
	FMOVD F3, 24(R1)
	ADD $32, R1, R1
	SUB $1, R10, R10
	B lhs4_f64_col1

lhs4_f64_next_strip:
	ADD R9, R0, R0
	SUB $4, R3, R3
	CMP $4, R3
	BGE lhs4_f64_strip_loop

lhs4_f64_done:
	RET


// func neonPackLHSKernelRows8Float64Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes uintptr, fullRows, contractingCols int)
TEXT ·neonPackLHSKernelRows8Float64Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD fullRows+24(FP), R3
	MOVD contractingCols+32(FP), R8

	CMP $8, R3
	BLT lhs8_f64_done
	CBZ R8, lhs8_f64_done

	LSL $3, R2, R9 // R9 = 8 * lhsStrideBytes

lhs8_f64_strip_loop:
	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	ADD R2, R7, R11
	ADD R2, R11, R12
	ADD R2, R12, R13
	ADD R2, R13, R14
	MOVD R8, R10

lhs8_f64_col2:
	CMP $2, R10
	BLT lhs8_f64_col1

	VLD1.P 16(R4), [V0.D2]
	VLD1.P 16(R5), [V1.D2]
	VLD1.P 16(R6), [V2.D2]
	VLD1.P 16(R7), [V3.D2]
	VLD1.P 16(R11), [V4.D2]
	VLD1.P 16(R12), [V5.D2]
	VLD1.P 16(R13), [V6.D2]
	VLD1.P 16(R14), [V7.D2]

	// Column 0:
	ZIP1_2D(16, 0, 1)
	ZIP1_2D(17, 2, 3)
	ZIP1_2D(18, 4, 5)
	ZIP1_2D(19, 6, 7)
	// Column 1:
	ZIP2_2D(20, 0, 1)
	ZIP2_2D(21, 2, 3)
	ZIP2_2D(22, 4, 5)
	ZIP2_2D(23, 6, 7)

	VST1.P [V16.D2, V17.D2, V18.D2, V19.D2], 64(R1)
	VST1.P [V20.D2, V21.D2, V22.D2, V23.D2], 64(R1)

	SUB $2, R10, R10
	B lhs8_f64_col2

lhs8_f64_col1:
	CBZ R10, lhs8_f64_next_strip
	FMOVD.P 8(R4), F0
	FMOVD.P 8(R5), F1
	FMOVD.P 8(R6), F2
	FMOVD.P 8(R7), F3
	FMOVD.P 8(R11), F4
	FMOVD.P 8(R12), F5
	FMOVD.P 8(R13), F6
	FMOVD.P 8(R14), F7
	FMOVD F0, 0(R1)
	FMOVD F1, 8(R1)
	FMOVD F2, 16(R1)
	FMOVD F3, 24(R1)
	FMOVD F4, 32(R1)
	FMOVD F5, 40(R1)
	FMOVD F6, 48(R1)
	FMOVD F7, 56(R1)
	ADD $64, R1, R1
	SUB $1, R10, R10
	B lhs8_f64_col1

lhs8_f64_next_strip:
	ADD R9, R0, R0
	SUB $8, R3, R3
	CMP $8, R3
	BGE lhs8_f64_strip_loop

lhs8_f64_done:
	RET


// func neonPackLHSKernelRows8StridedFloat32Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes, panelColStrideBytes uintptr, contractingCols int)
TEXT ·neonPackLHSKernelRows8StridedFloat32Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD panelColStrideBytes+24(FP), R3
	MOVD contractingCols+32(FP), R10

	CBZ R10, lhs8s_f32_done

	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	ADD R2, R7, R11
	ADD R2, R11, R12
	ADD R2, R12, R13
	ADD R2, R13, R14

lhs8s_f32_col4:
	CMP $4, R10
	BLT lhs8s_f32_col1

	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R5), [V1.S4]
	VLD1.P 16(R6), [V2.S4]
	VLD1.P 16(R7), [V3.S4]
	VLD1.P 16(R11), [V4.S4]
	VLD1.P 16(R12), [V5.S4]
	VLD1.P 16(R13), [V6.S4]
	VLD1.P 16(R14), [V7.S4]

	TRANSPOSE_4x4_32(0, 1, 2, 3, 16, 17, 18, 19, 24, 25, 26, 27)
	TRANSPOSE_4x4_32(4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26, 27)

	VST1 [V16.S4], (R1)
	ADD $16, R1, R15
	VST1 [V20.S4], (R15)
	ADD R3, R1, R1

	VST1 [V17.S4], (R1)
	ADD $16, R1, R15
	VST1 [V21.S4], (R15)
	ADD R3, R1, R1

	VST1 [V18.S4], (R1)
	ADD $16, R1, R15
	VST1 [V22.S4], (R15)
	ADD R3, R1, R1

	VST1 [V19.S4], (R1)
	ADD $16, R1, R15
	VST1 [V23.S4], (R15)
	ADD R3, R1, R1

	SUB $4, R10, R10
	B lhs8s_f32_col4

lhs8s_f32_col1:
	CBZ R10, lhs8s_f32_done
	FMOVS.P 4(R4), F0
	FMOVS.P 4(R5), F1
	FMOVS.P 4(R6), F2
	FMOVS.P 4(R7), F3
	FMOVS.P 4(R11), F4
	FMOVS.P 4(R12), F5
	FMOVS.P 4(R13), F6
	FMOVS.P 4(R14), F7
	FMOVS F0, 0(R1)
	FMOVS F1, 4(R1)
	FMOVS F2, 8(R1)
	FMOVS F3, 12(R1)
	FMOVS F4, 16(R1)
	FMOVS F5, 20(R1)
	FMOVS F6, 24(R1)
	FMOVS F7, 28(R1)
	ADD R3, R1, R1
	SUB $1, R10, R10
	B lhs8s_f32_col1

lhs8s_f32_done:
	RET


// func neonPackLHSKernelRows8StridedUint16Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes, panelColStrideBytes uintptr, contractingCols int)
TEXT ·neonPackLHSKernelRows8StridedUint16Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD panelColStrideBytes+24(FP), R3
	MOVD contractingCols+32(FP), R10

	CBZ R10, lhs8s_u16_done

	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	ADD R2, R7, R11
	ADD R2, R11, R12
	ADD R2, R12, R13
	ADD R2, R13, R14

lhs8s_u16_col8:
	CMP $8, R10
	BLT lhs8s_u16_col1

	VLD1.P 16(R4), [V0.H8]
	VLD1.P 16(R5), [V1.H8]
	VLD1.P 16(R6), [V2.H8]
	VLD1.P 16(R7), [V3.H8]
	VLD1.P 16(R11), [V4.H8]
	VLD1.P 16(R12), [V5.H8]
	VLD1.P 16(R13), [V6.H8]
	VLD1.P 16(R14), [V7.H8]

	ZIP1_8H(8, 0, 1)
	ZIP2_8H(9, 0, 1)
	ZIP1_8H(10, 2, 3)
	ZIP2_8H(11, 2, 3)
	ZIP1_4S(16, 8, 10)
	ZIP2_4S(17, 8, 10)
	ZIP1_4S(18, 9, 11)
	ZIP2_4S(19, 9, 11)

	ZIP1_8H(12, 4, 5)
	ZIP2_8H(13, 4, 5)
	ZIP1_8H(14, 6, 7)
	ZIP2_8H(15, 6, 7)
	ZIP1_4S(20, 12, 14)
	ZIP2_4S(21, 12, 14)
	ZIP1_4S(22, 13, 15)
	ZIP2_4S(23, 13, 15)

	ZIP1_2D(24, 16, 20)
	ZIP2_2D(25, 16, 20)
	ZIP1_2D(26, 17, 21)
	ZIP2_2D(27, 17, 21)
	ZIP1_2D(28, 18, 22)
	ZIP2_2D(29, 18, 22)
	ZIP1_2D(30, 19, 23)
	ZIP2_2D(31, 19, 23)

	VST1 [V24.B16], (R1)
	ADD R3, R1, R1
	VST1 [V25.B16], (R1)
	ADD R3, R1, R1
	VST1 [V26.B16], (R1)
	ADD R3, R1, R1
	VST1 [V27.B16], (R1)
	ADD R3, R1, R1
	VST1 [V28.B16], (R1)
	ADD R3, R1, R1
	VST1 [V29.B16], (R1)
	ADD R3, R1, R1
	VST1 [V30.B16], (R1)
	ADD R3, R1, R1
	VST1 [V31.B16], (R1)
	ADD R3, R1, R1

	SUB $8, R10, R10
	B lhs8s_u16_col8

lhs8s_u16_col1:
	CBZ R10, lhs8s_u16_done
	MOVHU.P 2(R4), R15
	MOVH R15, 0(R1)
	MOVHU.P 2(R5), R15
	MOVH R15, 2(R1)
	MOVHU.P 2(R6), R15
	MOVH R15, 4(R1)
	MOVHU.P 2(R7), R15
	MOVH R15, 6(R1)
	MOVHU.P 2(R11), R15
	MOVH R15, 8(R1)
	MOVHU.P 2(R12), R15
	MOVH R15, 10(R1)
	MOVHU.P 2(R13), R15
	MOVH R15, 12(R1)
	MOVHU.P 2(R14), R15
	MOVH R15, 14(R1)
	ADD R3, R1, R1
	SUB $1, R10, R10
	B lhs8s_u16_col1

lhs8s_u16_done:
	RET


// func neonPackLHSKernelRows8StridedFloat64Asm(lhsStartPtr, panelPtr unsafe.Pointer, lhsStrideBytes, panelColStrideBytes uintptr, contractingCols int)
TEXT ·neonPackLHSKernelRows8StridedFloat64Asm(SB), NOSPLIT, $0-40
	MOVD lhsStartPtr+0(FP), R0
	MOVD panelPtr+8(FP), R1
	MOVD lhsStrideBytes+16(FP), R2
	MOVD panelColStrideBytes+24(FP), R3
	MOVD contractingCols+32(FP), R10

	CBZ R10, lhs8s_f64_done

	MOVD R0, R4
	ADD R2, R4, R5
	ADD R2, R5, R6
	ADD R2, R6, R7
	ADD R2, R7, R11
	ADD R2, R11, R12
	ADD R2, R12, R13
	ADD R2, R13, R14

lhs8s_f64_col2:
	CMP $2, R10
	BLT lhs8s_f64_col1

	VLD1.P 16(R4), [V0.D2]
	VLD1.P 16(R5), [V1.D2]
	VLD1.P 16(R6), [V2.D2]
	VLD1.P 16(R7), [V3.D2]
	VLD1.P 16(R11), [V4.D2]
	VLD1.P 16(R12), [V5.D2]
	VLD1.P 16(R13), [V6.D2]
	VLD1.P 16(R14), [V7.D2]

	ZIP1_2D(16, 0, 1)
	ZIP1_2D(17, 2, 3)
	ZIP1_2D(18, 4, 5)
	ZIP1_2D(19, 6, 7)
	ZIP2_2D(20, 0, 1)
	ZIP2_2D(21, 2, 3)
	ZIP2_2D(22, 4, 5)
	ZIP2_2D(23, 6, 7)

	VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R1)
	ADD R3, R1, R1
	VST1 [V20.D2, V21.D2, V22.D2, V23.D2], (R1)
	ADD R3, R1, R1

	SUB $2, R10, R10
	B lhs8s_f64_col2

lhs8s_f64_col1:
	CBZ R10, lhs8s_f64_done
	FMOVD.P 8(R4), F0
	FMOVD.P 8(R5), F1
	FMOVD.P 8(R6), F2
	FMOVD.P 8(R7), F3
	FMOVD.P 8(R11), F4
	FMOVD.P 8(R12), F5
	FMOVD.P 8(R13), F6
	FMOVD.P 8(R14), F7
	FMOVD F0, 0(R1)
	FMOVD F1, 8(R1)
	FMOVD F2, 16(R1)
	FMOVD F3, 24(R1)
	FMOVD F4, 32(R1)
	FMOVD F5, 40(R1)
	FMOVD F6, 48(R1)
	FMOVD F7, 56(R1)
	ADD R3, R1, R1
	SUB $1, R10, R10
	B lhs8s_f64_col1

lhs8s_f64_done:
	RET

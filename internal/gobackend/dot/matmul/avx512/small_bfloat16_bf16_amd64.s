// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

#include "textflag.h"

// VDPBF16PS Zsrc2, Zsrc1, Zdst  (Intel: vdpbf16ps zmm_dst, zmm_src1, zmm_src2)
// Encoding: EVEX.512.F3.0F38.W0 52 /r
#define VDPBF16PS(src2, src1, dst) \
	BYTE $0x62; \
	BYTE $(((1^(((dst)>>3)&1))<<7) | ((1^(((src2)>>4)&1))<<6) | ((1^(((src2)>>3)&1))<<5) | ((1^(((dst)>>4)&1))<<4) | 0x02); \
	BYTE $(((15-((src1)&15))<<3) | 0x06); \
	BYTE $(0x40 | ((1^(((src1)>>4)&1))<<3)); \
	BYTE $0x52; \
	BYTE $(0xC0 | (((dst)&7)<<3) | ((src2)&7))

#define REDUCE_Z(z, y, x) \
	VEXTRACTF32X8 $1, z, Y24 \
	VADDPS Y24, y, y \
	VEXTRACTF32X4 $1, y, X24 \
	VADDPS X24, x, x \
	VMOVSHDUP x, X24 \
	VADDPS X24, x, x \
	VPERMILPS $0x02, x, X24 \
	VADDSS X24, x, x

// De-permute a pair of accumulators (E = cols [0..3|8..11|16..19|24..27],
// O = cols [4..7|12..15|20..23|28..31]) and store 32 FP32 values in natural order.
#define DEPERMUTE_STORE_32(E, O, DST, OFF) \
	VSHUFF32X4 $0x44, O, E, Z24; \
	VSHUFF32X4 $0xD8, Z24, Z24, Z24; \
	VSHUFF32X4 $0xEE, O, E, Z25; \
	VSHUFF32X4 $0xD8, Z25, Z25, Z25; \
	VMOVUPS Z24, OFF(DST); \
	VMOVUPS Z25, (OFF+64)(DST)

// ============================================================================
// BFloat16 Small Transposed 4x4 using AVX512_BF16 (VDPBF16PS)
// Processes 32 BFloat16s per iteration + zero-masked K1 tail.
// ============================================================================
TEXT ·avx512SmallTransposedTile4x4BFloat16BF16Asm(SB), NOSPLIT, $0-104
	MOVQ lRow0+0(FP), R8
	MOVQ lRow1+8(FP), R9
	MOVQ lRow2+16(FP), R10
	MOVQ lRow3+24(FP), R11
	MOVQ rCol0+32(FP), R12
	MOVQ rCol1+40(FP), R13
	MOVQ rCol2+48(FP), R14
	MOVQ rCol3+56(FP), R15
	MOVQ contractingSize+64(FP), BX

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7
	VXORPS Z8, Z8, Z8
	VXORPS Z9, Z9, Z9
	VXORPS Z10, Z10, Z10
	VXORPS Z11, Z11, Z11
	VXORPS Z12, Z12, Z12
	VXORPS Z13, Z13, Z13
	VXORPS Z14, Z14, Z14
	VXORPS Z15, Z15, Z15

	XORQ SI, SI

loop_k32_4x4_vdpbf16:
	CMPQ BX, $32
	JL tail_k_4x4_vdpbf16

	VMOVDQU16 (0)(R8)(SI*1), Z16
	VMOVDQU16 (0)(R9)(SI*1), Z17
	VMOVDQU16 (0)(R10)(SI*1), Z18
	VMOVDQU16 (0)(R11)(SI*1), Z19

	VMOVDQU16 (0)(R12)(SI*1), Z20
	VMOVDQU16 (0)(R13)(SI*1), Z21
	VMOVDQU16 (0)(R14)(SI*1), Z22
	VMOVDQU16 (0)(R15)(SI*1), Z23

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(21, 16, 1)
	VDPBF16PS(22, 16, 2)
	VDPBF16PS(23, 16, 3)

	VDPBF16PS(20, 17, 4)
	VDPBF16PS(21, 17, 5)
	VDPBF16PS(22, 17, 6)
	VDPBF16PS(23, 17, 7)

	VDPBF16PS(20, 18, 8)
	VDPBF16PS(21, 18, 9)
	VDPBF16PS(22, 18, 10)
	VDPBF16PS(23, 18, 11)

	VDPBF16PS(20, 19, 12)
	VDPBF16PS(21, 19, 13)
	VDPBF16PS(22, 19, 14)
	VDPBF16PS(23, 19, 15)

	ADDQ $64, SI
	SUBQ $32, BX
	JMP loop_k32_4x4_vdpbf16

tail_k_4x4_vdpbf16:
	TESTQ BX, BX
	JLE reduce_4x4_vdpbf16

	// Build 16-bit element mask K1 for 1 <= BX <= 31 elements: (1 << BX) - 1
	MOVL $1, AX
	MOVQ BX, CX
	SHLL CL, AX
	DECL AX
	KMOVD AX, K1

	VMOVDQU16.Z (0)(R8)(SI*1), K1, Z16
	VMOVDQU16.Z (0)(R9)(SI*1), K1, Z17
	VMOVDQU16.Z (0)(R10)(SI*1), K1, Z18
	VMOVDQU16.Z (0)(R11)(SI*1), K1, Z19

	VMOVDQU16.Z (0)(R12)(SI*1), K1, Z20
	VMOVDQU16.Z (0)(R13)(SI*1), K1, Z21
	VMOVDQU16.Z (0)(R14)(SI*1), K1, Z22
	VMOVDQU16.Z (0)(R15)(SI*1), K1, Z23

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(21, 16, 1)
	VDPBF16PS(22, 16, 2)
	VDPBF16PS(23, 16, 3)

	VDPBF16PS(20, 17, 4)
	VDPBF16PS(21, 17, 5)
	VDPBF16PS(22, 17, 6)
	VDPBF16PS(23, 17, 7)

	VDPBF16PS(20, 18, 8)
	VDPBF16PS(21, 18, 9)
	VDPBF16PS(22, 18, 10)
	VDPBF16PS(23, 18, 11)

	VDPBF16PS(20, 19, 12)
	VDPBF16PS(21, 19, 13)
	VDPBF16PS(22, 19, 14)
	VDPBF16PS(23, 19, 15)

reduce_4x4_vdpbf16:
	REDUCE_Z(Z0, Y0, X0)
	REDUCE_Z(Z1, Y1, X1)
	REDUCE_Z(Z2, Y2, X2)
	REDUCE_Z(Z3, Y3, X3)
	REDUCE_Z(Z4, Y4, X4)
	REDUCE_Z(Z5, Y5, X5)
	REDUCE_Z(Z6, Y6, X6)
	REDUCE_Z(Z7, Y7, X7)
	REDUCE_Z(Z8, Y8, X8)
	REDUCE_Z(Z9, Y9, X9)
	REDUCE_Z(Z10, Y10, X10)
	REDUCE_Z(Z11, Y11, X11)
	REDUCE_Z(Z12, Y12, X12)
	REDUCE_Z(Z13, Y13, X13)
	REDUCE_Z(Z14, Y14, X14)
	REDUCE_Z(Z15, Y15, X15)

	MOVQ outRow0+72(FP), R8
	MOVQ outRow1+80(FP), R9
	MOVQ outRow2+88(FP), R10
	MOVQ outRow3+96(FP), R11

	VMOVSS X0, 0(R8)
	VMOVSS X1, 4(R8)
	VMOVSS X2, 8(R8)
	VMOVSS X3, 12(R8)

	VMOVSS X4, 0(R9)
	VMOVSS X5, 4(R9)
	VMOVSS X6, 8(R9)
	VMOVSS X7, 12(R9)

	VMOVSS X8, 0(R10)
	VMOVSS X9, 4(R10)
	VMOVSS X10, 8(R10)
	VMOVSS X11, 12(R10)

	VMOVSS X12, 0(R11)
	VMOVSS X13, 4(R11)
	VMOVSS X14, 8(R11)
	VMOVSS X15, 12(R11)

	VZEROUPPER
	RET

// ============================================================================
// BFloat16 Small Transposed 4x1 using AVX512_BF16 (VDPBF16PS)
// ============================================================================
TEXT ·avx512SmallTransposedTile4x1BFloat16BF16Asm(SB), NOSPLIT, $0-80
	MOVQ lRow0+0(FP), R8
	MOVQ lRow1+8(FP), R9
	MOVQ lRow2+16(FP), R10
	MOVQ lRow3+24(FP), R11
	MOVQ rCol0+32(FP), R12
	MOVQ contractingSize+40(FP), BX

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

	XORQ SI, SI

loop_k32_4x1_vdpbf16:
	CMPQ BX, $32
	JL tail_k_4x1_vdpbf16

	VMOVDQU16 (0)(R8)(SI*1), Z16
	VMOVDQU16 (0)(R9)(SI*1), Z17
	VMOVDQU16 (0)(R10)(SI*1), Z18
	VMOVDQU16 (0)(R11)(SI*1), Z19
	VMOVDQU16 (0)(R12)(SI*1), Z20

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(20, 17, 1)
	VDPBF16PS(20, 18, 2)
	VDPBF16PS(20, 19, 3)

	ADDQ $64, SI
	SUBQ $32, BX
	JMP loop_k32_4x1_vdpbf16

tail_k_4x1_vdpbf16:
	TESTQ BX, BX
	JLE reduce_4x1_vdpbf16

	MOVL $1, AX
	MOVQ BX, CX
	SHLL CL, AX
	DECL AX
	KMOVD AX, K1

	VMOVDQU16.Z (0)(R8)(SI*1), K1, Z16
	VMOVDQU16.Z (0)(R9)(SI*1), K1, Z17
	VMOVDQU16.Z (0)(R10)(SI*1), K1, Z18
	VMOVDQU16.Z (0)(R11)(SI*1), K1, Z19
	VMOVDQU16.Z (0)(R12)(SI*1), K1, Z20

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(20, 17, 1)
	VDPBF16PS(20, 18, 2)
	VDPBF16PS(20, 19, 3)

reduce_4x1_vdpbf16:
	REDUCE_Z(Z0, Y0, X0)
	REDUCE_Z(Z1, Y1, X1)
	REDUCE_Z(Z2, Y2, X2)
	REDUCE_Z(Z3, Y3, X3)

	MOVQ outRow0+48(FP), R8
	MOVQ outRow1+56(FP), R9
	MOVQ outRow2+64(FP), R10
	MOVQ outRow3+72(FP), R11

	VMOVSS X0, 0(R8)
	VMOVSS X1, 0(R9)
	VMOVSS X2, 0(R10)
	VMOVSS X3, 0(R11)

	VZEROUPPER
	RET

// ============================================================================
// BFloat16 Small Non-Transposed 4x64 using AVX512_BF16 (VDPBF16PS)
// Processes 2 K-steps per iteration; de-permutes accumulator columns once on store.
// ============================================================================
TEXT ·avx512SmallNonTransposedTile4x64BFloat16BF16Asm(SB), NOSPLIT, $0-88
	MOVQ lRow0+0(FP), R8
	MOVQ lRow1+8(FP), R9
	MOVQ lRow2+16(FP), R10
	MOVQ lRow3+24(FP), R11
	MOVQ rhs+32(FP), R12
	MOVQ contractingSize+40(FP), BX
	MOVQ rhsByteStride+48(FP), DX

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7
	VXORPS Z8, Z8, Z8
	VXORPS Z9, Z9, Z9
	VXORPS Z10, Z10, Z10
	VXORPS Z11, Z11, Z11
	VXORPS Z12, Z12, Z12
	VXORPS Z13, Z13, Z13
	VXORPS Z14, Z14, Z14
	VXORPS Z15, Z15, Z15

	TESTQ BX, BX
	JLE write_output_64_vdpbf16

	XORQ SI, SI
	XORQ R13, R13                        // RHS byte offset

loop_k2_64_vdpbf16:
	CMPQ BX, $2
	JL tail_k1_64_vdpbf16

	// Broadcast 2 contiguous BF16s (k, k+1) from each of the 4 LHS rows
	VPBROADCASTD (0)(R8)(SI*1), Z16
	VPBROADCASTD (0)(R9)(SI*1), Z17
	VPBROADCASTD (0)(R10)(SI*1), Z18
	VPBROADCASTD (0)(R11)(SI*1), Z19

	LEAQ (R12)(R13*1), CX                // RHS row k
	LEAQ (CX)(DX*1), BP                  // RHS row k+1

	VMOVDQU16 0(CX), Z24
	VMOVDQU16 0(BP), Z25
	VPUNPCKLWD Z25, Z24, Z20             // cols [0..3|8..11|16..19|24..27]
	VPUNPCKHWD Z25, Z24, Z21             // cols [4..7|12..15|20..23|28..31]

	VMOVDQU16 64(CX), Z26
	VMOVDQU16 64(BP), Z27
	VPUNPCKLWD Z27, Z26, Z22             // cols [32..35|40..43|48..51|56..59]
	VPUNPCKHWD Z27, Z26, Z23             // cols [36..39|44..47|52..55|60..63]

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(21, 16, 1)
	VDPBF16PS(22, 16, 2)
	VDPBF16PS(23, 16, 3)

	VDPBF16PS(20, 17, 4)
	VDPBF16PS(21, 17, 5)
	VDPBF16PS(22, 17, 6)
	VDPBF16PS(23, 17, 7)

	VDPBF16PS(20, 18, 8)
	VDPBF16PS(21, 18, 9)
	VDPBF16PS(22, 18, 10)
	VDPBF16PS(23, 18, 11)

	VDPBF16PS(20, 19, 12)
	VDPBF16PS(21, 19, 13)
	VDPBF16PS(22, 19, 14)
	VDPBF16PS(23, 19, 15)

	ADDQ $4, SI
	LEAQ (R13)(DX*2), R13
	SUBQ $2, BX
	JMP loop_k2_64_vdpbf16

tail_k1_64_vdpbf16:
	TESTQ BX, BX
	JZ write_output_64_vdpbf16

	// Odd trailing K: zero-extend single BF16 to (bf16, 0x0000) and interleave RHS with 0
	MOVWLZX (0)(R8)(SI*1), AX;  VPBROADCASTD AX, Z16
	MOVWLZX (0)(R9)(SI*1), AX;  VPBROADCASTD AX, Z17
	MOVWLZX (0)(R10)(SI*1), AX; VPBROADCASTD AX, Z18
	MOVWLZX (0)(R11)(SI*1), AX; VPBROADCASTD AX, Z19

	LEAQ (R12)(R13*1), CX
	VPXORD Z25, Z25, Z25

	VMOVDQU16 0(CX), Z24
	VPUNPCKLWD Z25, Z24, Z20
	VPUNPCKHWD Z25, Z24, Z21

	VMOVDQU16 64(CX), Z26
	VPUNPCKLWD Z25, Z26, Z22
	VPUNPCKHWD Z25, Z26, Z23

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(21, 16, 1)
	VDPBF16PS(22, 16, 2)
	VDPBF16PS(23, 16, 3)

	VDPBF16PS(20, 17, 4)
	VDPBF16PS(21, 17, 5)
	VDPBF16PS(22, 17, 6)
	VDPBF16PS(23, 17, 7)

	VDPBF16PS(20, 18, 8)
	VDPBF16PS(21, 18, 9)
	VDPBF16PS(22, 18, 10)
	VDPBF16PS(23, 18, 11)

	VDPBF16PS(20, 19, 12)
	VDPBF16PS(21, 19, 13)
	VDPBF16PS(22, 19, 14)
	VDPBF16PS(23, 19, 15)

write_output_64_vdpbf16:
	MOVQ outRow0+56(FP), R13
	MOVQ outRow1+64(FP), R14
	MOVQ outRow2+72(FP), R15
	MOVQ outRow3+80(FP), DI

	DEPERMUTE_STORE_32(Z0, Z1, R13, 0)
	DEPERMUTE_STORE_32(Z2, Z3, R13, 128)

	DEPERMUTE_STORE_32(Z4, Z5, R14, 0)
	DEPERMUTE_STORE_32(Z6, Z7, R14, 128)

	DEPERMUTE_STORE_32(Z8, Z9, R15, 0)
	DEPERMUTE_STORE_32(Z10, Z11, R15, 128)

	DEPERMUTE_STORE_32(Z12, Z13, DI, 0)
	DEPERMUTE_STORE_32(Z14, Z15, DI, 128)

	VZEROUPPER
	RET

// ============================================================================
// BFloat16 Small Non-Transposed 4x16 using AVX512_BF16 (VDPBF16PS)
// ============================================================================
TEXT ·avx512SmallNonTransposedTile4x16BFloat16BF16Asm(SB), NOSPLIT, $0-88
	MOVQ lRow0+0(FP), R8
	MOVQ lRow1+8(FP), R9
	MOVQ lRow2+16(FP), R10
	MOVQ lRow3+24(FP), R11
	MOVQ rhs+32(FP), R12
	MOVQ contractingSize+40(FP), BX
	MOVQ rhsByteStride+48(FP), DX

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

	TESTQ BX, BX
	JLE write_output_16_vdpbf16

	XORQ SI, SI
	XORQ R13, R13

loop_k2_16_vdpbf16:
	CMPQ BX, $2
	JL tail_k1_16_vdpbf16

	VPBROADCASTD (0)(R8)(SI*1), Z16
	VPBROADCASTD (0)(R9)(SI*1), Z17
	VPBROADCASTD (0)(R10)(SI*1), Z18
	VPBROADCASTD (0)(R11)(SI*1), Z19

	LEAQ (R12)(R13*1), CX                // RHS row k
	LEAQ (CX)(DX*1), BP                  // RHS row k+1

	VMOVDQU16 0(CX), Y24                 // 16 BF16s from row k
	VMOVDQU16 0(BP), Y25                 // 16 BF16s from row k+1
	VPUNPCKLWD Y25, Y24, Y20             // cols [0..3 | 8..11]
	VPUNPCKHWD Y25, Y24, Y21             // cols [4..7 | 12..15]
	VINSERTI64X4 $1, Y21, Z20, Z20       // Z20 = [0..3 | 8..11 | 4..7 | 12..15]

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(20, 17, 1)
	VDPBF16PS(20, 18, 2)
	VDPBF16PS(20, 19, 3)

	ADDQ $4, SI
	LEAQ (R13)(DX*2), R13
	SUBQ $2, BX
	JMP loop_k2_16_vdpbf16

tail_k1_16_vdpbf16:
	TESTQ BX, BX
	JZ write_output_16_vdpbf16

	MOVWLZX (0)(R8)(SI*1), AX;  VPBROADCASTD AX, Z16
	MOVWLZX (0)(R9)(SI*1), AX;  VPBROADCASTD AX, Z17
	MOVWLZX (0)(R10)(SI*1), AX; VPBROADCASTD AX, Z18
	MOVWLZX (0)(R11)(SI*1), AX; VPBROADCASTD AX, Z19

	LEAQ (R12)(R13*1), CX
	VMOVDQU16 0(CX), Y24
	VPXORD Y25, Y25, Y25
	VPUNPCKLWD Y25, Y24, Y20
	VPUNPCKHWD Y25, Y24, Y21
	VINSERTI64X4 $1, Y21, Z20, Z20

	VDPBF16PS(20, 16, 0)
	VDPBF16PS(20, 17, 1)
	VDPBF16PS(20, 18, 2)
	VDPBF16PS(20, 19, 3)

write_output_16_vdpbf16:
	MOVQ outRow0+56(FP), R13
	MOVQ outRow1+64(FP), R14
	MOVQ outRow2+72(FP), R15
	MOVQ outRow3+80(FP), DI

	// De-permute [0..3 | 8..11 | 4..7 | 12..15] -> [0..3 | 4..7 | 8..11 | 12..15]
	VSHUFF32X4 $0xD8, Z0, Z0, Z0
	VSHUFF32X4 $0xD8, Z1, Z1, Z1
	VSHUFF32X4 $0xD8, Z2, Z2, Z2
	VSHUFF32X4 $0xD8, Z3, Z3, Z3

	VMOVUPS Z0, 0(R13)
	VMOVUPS Z1, 0(R14)
	VMOVUPS Z2, 0(R15)
	VMOVUPS Z3, 0(DI)

	VZEROUPPER
	RET

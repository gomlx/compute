// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// func reduceTrailingSumFloat32NEONAsm(in, out unsafe.Pointer, A, B int)
// Reduces [A, B] -> [A] for float32 using 4-way unrolled NEON accumulators and VFADDP horizontal reduction.
TEXT ·reduceTrailingSumFloat32NEONAsm(SB), NOSPLIT, $0-32
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD A+16(FP), R2
	MOVD B+24(FP), R3

	CMP $0, R2
	BLE f32_trail_done

	// Byte stride per row = B * 4
	LSL $2, R3, R8

f32_trail_row_loop:
	MOVD R0, R4         // R4 = current row pointer
	MOVD R3, R5         // R5 = remaining elements in row
	VEOR V0.B16, V0.B16, V0.B16

	CMP $16, R5
	BLT f32_trail_vec4_check

	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

f32_trail_vec16_loop:
	VLD1.P 64(R4), [V4.S4, V5.S4, V6.S4, V7.S4]
	VFADD V4.S4, V0.S4, V0.S4
	VFADD V5.S4, V1.S4, V1.S4
	VFADD V6.S4, V2.S4, V2.S4
	VFADD V7.S4, V3.S4, V3.S4
	SUB $16, R5, R5
	CMP $16, R5
	BGE f32_trail_vec16_loop

	VFADD V1.S4, V0.S4, V0.S4
	VFADD V3.S4, V2.S4, V2.S4
	VFADD V2.S4, V0.S4, V0.S4

f32_trail_vec4_check:
	CMP $4, R5
	BLT f32_trail_hsum

f32_trail_vec4_loop:
	VLD1.P 16(R4), [V4.S4]
	VFADD V4.S4, V0.S4, V0.S4
	SUB $4, R5, R5
	CMP $4, R5
	BGE f32_trail_vec4_loop

f32_trail_hsum:
	// Horizontal sum of V0.S4 into F0 (S0)
	VFADDP V0.S4, V0.S4, V0.S4
	VFADDP V0.S4, V0.S4, V0.S4

	CBZ R5, f32_trail_store

f32_trail_scalar_loop:
	FMOVS.P 4(R4), F4
	FADDS F4, F0, F0
	SUBS $1, R5, R5
	BGT f32_trail_scalar_loop

f32_trail_store:
	FMOVS.P F0, 4(R1)
	ADD R8, R0, R0
	SUBS $1, R2, R2
	BGT f32_trail_row_loop

f32_trail_done:
	RET

// func reduceTrailingSumFloat64NEONAsm(in, out unsafe.Pointer, A, B int)
// Reduces [A, B] -> [A] for float64 using 4-way unrolled NEON accumulators and VFADDP horizontal reduction.
TEXT ·reduceTrailingSumFloat64NEONAsm(SB), NOSPLIT, $0-32
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD A+16(FP), R2
	MOVD B+24(FP), R3

	CMP $0, R2
	BLE f64_trail_done

	// Byte stride per row = B * 8
	LSL $3, R3, R8

f64_trail_row_loop:
	MOVD R0, R4
	MOVD R3, R5
	VEOR V0.B16, V0.B16, V0.B16

	CMP $8, R5
	BLT f64_trail_vec2_check

	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

f64_trail_vec8_loop:
	VLD1.P 64(R4), [V4.D2, V5.D2, V6.D2, V7.D2]
	VFADD V4.D2, V0.D2, V0.D2
	VFADD V5.D2, V1.D2, V1.D2
	VFADD V6.D2, V2.D2, V2.D2
	VFADD V7.D2, V3.D2, V3.D2
	SUB $8, R5, R5
	CMP $8, R5
	BGE f64_trail_vec8_loop

	VFADD V1.D2, V0.D2, V0.D2
	VFADD V3.D2, V2.D2, V2.D2
	VFADD V2.D2, V0.D2, V0.D2

f64_trail_vec2_check:
	CMP $2, R5
	BLT f64_trail_hsum

f64_trail_vec2_loop:
	VLD1.P 16(R4), [V4.D2]
	VFADD V4.D2, V0.D2, V0.D2
	SUB $2, R5, R5
	CMP $2, R5
	BGE f64_trail_vec2_loop

f64_trail_hsum:
	// Horizontal sum of V0.D2 into F0 (D0)
	VFADDP V0.D2, V0.D2, V0.D2

	CBZ R5, f64_trail_store

	FMOVD (R4), F4
	FADDD F4, F0, F0

f64_trail_store:
	FMOVD.P F0, 8(R1)
	ADD R8, R0, R0
	SUBS $1, R2, R2
	BGT f64_trail_row_loop

f64_trail_done:
	RET

// func reduceTrailingSumFloat16NEONAsm(in, out unsafe.Pointer, A, B int)
// Reduces [A, B] -> [A] for Float16 using hardware FCVTL/FCVTL2 to Float32x4 accumulators and FCVTN.
TEXT ·reduceTrailingSumFloat16NEONAsm(SB), NOSPLIT, $0-32
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD A+16(FP), R2
	MOVD B+24(FP), R3

	CMP $0, R2
	BLE f16_trail_done

	// Byte stride per row = B * 2
	LSL $1, R3, R8

f16_trail_row_loop:
	MOVD R0, R4
	MOVD R3, R5
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16

	CMP $8, R5
	BLT f16_trail_hsum

f16_trail_vec8_loop:
	VLD1.P 16(R4), [V2.H8]
	// FCVTL V3.4S, V2.4H  (0x0E217800 | (2<<5) | 3 = 0x0E217843)
	WORD $0x0E217843
	// FCVTL2 V4.4S, V2.8H (0x4E217800 | (2<<5) | 4 = 0x4E217844)
	WORD $0x4E217844
	VFADD V3.S4, V0.S4, V0.S4
	VFADD V4.S4, V1.S4, V1.S4
	SUB $8, R5, R5
	CMP $8, R5
	BGE f16_trail_vec8_loop

	VFADD V1.S4, V0.S4, V0.S4

f16_trail_hsum:
	VFADDP V0.S4, V0.S4, V0.S4
	VFADDP V0.S4, V0.S4, V0.S4

	CBZ R5, f16_trail_store

f16_trail_scalar_loop:
	MOVHU.P 2(R4), R6
	VMOV R6, V2.H[0]
	// FCVTL V3.4S, V2.4H (0x0E217843)
	WORD $0x0E217843
	FADDS F3, F0, F0
	SUBS $1, R5, R5
	BGT f16_trail_scalar_loop

f16_trail_store:
	// FCVTN V0.4H, V0.4S (0x0E216800)
	WORD $0x0E216800
	VMOV V0.H[0], R6
	MOVH.P R6, 2(R1)
	ADD R8, R0, R0
	SUBS $1, R2, R2
	BGT f16_trail_row_loop

f16_trail_done:
	RET

// func reduceTrailingSumBFloat16NEONAsm(in, out unsafe.Pointer, A, B int)
// Reduces [A, B] -> [A] for BFloat16 using SHLL/SHLL2 (#16) to Float32x4 accumulators.
TEXT ·reduceTrailingSumBFloat16NEONAsm(SB), NOSPLIT, $0-32
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD A+16(FP), R2
	MOVD B+24(FP), R3

	CMP $0, R2
	BLE bf16_trail_done

	// Byte stride per row = B * 2
	LSL $1, R3, R8

bf16_trail_row_loop:
	MOVD R0, R4
	MOVD R3, R5
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16

	CMP $8, R5
	BLT bf16_trail_hsum

bf16_trail_vec8_loop:
	VLD1.P 16(R4), [V2.H8]
	// SHLL V3.4S, V2.4H, #16  (0x2E613800 | (2<<5) | 3 = 0x2E613843)
	WORD $0x2E613843
	// SHLL2 V4.4S, V2.8H, #16 (0x6E613800 | (2<<5) | 4 = 0x6E613844)
	WORD $0x6E613844
	VFADD V3.S4, V0.S4, V0.S4
	VFADD V4.S4, V1.S4, V1.S4
	SUB $8, R5, R5
	CMP $8, R5
	BGE bf16_trail_vec8_loop

	VFADD V1.S4, V0.S4, V0.S4

bf16_trail_hsum:
	VFADDP V0.S4, V0.S4, V0.S4
	VFADDP V0.S4, V0.S4, V0.S4

	CBZ R5, bf16_trail_store

bf16_trail_scalar_loop:
	MOVHU.P 2(R4), R6
	LSL $16, R6, R6
	FMOVS R6, F3
	FADDS F3, F0, F0
	SUBS $1, R5, R5
	BGT bf16_trail_scalar_loop

bf16_trail_store:
	// Convert F0 (float32) to BFloat16 using round-to-nearest-even
	FMOVS F0, R6
	AND $0x7F800000, R6, R7
	CMP $0x7F800000, R7
	BEQ bf16_trail_no_round
	LSR $16, R6, R7
	AND $1, R7, R7
	ADD $0x7FFF, R7, R7
	ADD R7, R6, R6
bf16_trail_no_round:
	LSR $16, R6, R6
	MOVH.P R6, 2(R1)
	ADD R8, R0, R0
	SUBS $1, R2, R2
	BGT bf16_trail_row_loop

bf16_trail_done:
	RET

// func reduceLeadingSumFloat16NEONAsm(in, out unsafe.Pointer, A, B int)
// Reduces [A, B] -> [B] for Float16 by strip-mining 8 columns at a time in Float32x4 accumulators.
TEXT ·reduceLeadingSumFloat16NEONAsm(SB), NOSPLIT, $0-32
	MOVD in+0(FP), R0
	MOVD out+8(FP), R1
	MOVD A+16(FP), R2
	MOVD B+24(FP), R3

	CMP $0, R2
	BLE f16_lead_done
	CMP $0, R3
	BLE f16_lead_done

	// R8 = row stride in bytes = B * 2
	LSL $1, R3, R8
	// R9 = remaining columns
	MOVD R3, R9

	CMP $8, R9
	BLT f16_lead_tail_check

f16_lead_col8_loop:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	MOVD R0, R4         // R4 = &in[0, b]
	MOVD R2, R5         // R5 = A rows

f16_lead_col8_row_loop:
	VLD1 (R4), [V2.H8]
	// FCVTL V3.4S, V2.4H  (0x0E217843)
	WORD $0x0E217843
	// FCVTL2 V4.4S, V2.8H (0x4E217844)
	WORD $0x4E217844
	VFADD V3.S4, V0.S4, V0.S4
	VFADD V4.S4, V1.S4, V1.S4
	ADD R8, R4, R4
	SUBS $1, R5, R5
	BGT f16_lead_col8_row_loop

	// Narrow V0.S4, V1.S4 to V2.H8 via FCVTN / FCVTN2
	// FCVTN V2.4H, V0.4S  (0x0E216800 | (0<<5) | 2 = 0x0E216802)
	WORD $0x0E216802
	// FCVTN2 V2.8H, V1.4S (0x4E216800 | (1<<5) | 2 = 0x4E216822)
	WORD $0x4E216822
	VST1.P [V2.H8], 16(R1)

	ADD $16, R0, R0
	SUB $8, R9, R9
	CMP $8, R9
	BGE f16_lead_col8_loop

f16_lead_tail_check:
	CBZ R9, f16_lead_done

f16_lead_tail_col_loop:
	VEOR V0.B16, V0.B16, V0.B16
	MOVD R0, R4
	MOVD R2, R5

f16_lead_tail_row_loop:
	MOVHU (R4), R6
	VMOV R6, V2.H[0]
	// FCVTL V3.4S, V2.4H (0x0E217843)
	WORD $0x0E217843
	FADDS F3, F0, F0
	ADD R8, R4, R4
	SUBS $1, R5, R5
	BGT f16_lead_tail_row_loop

	// FCVTN V0.4H, V0.4S (0x0E216800)
	WORD $0x0E216800
	VMOV V0.H[0], R6
	MOVH.P R6, 2(R1)

	ADD $2, R0, R0
	SUBS $1, R9, R9
	BGT f16_lead_tail_col_loop

f16_lead_done:
	RET

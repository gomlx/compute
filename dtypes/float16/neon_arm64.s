// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

#include "textflag.h"

// func convertF16ToF32NEONAsm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 4) Float16 values at src to float32 at dst using NEON FCVTL / FCVTL2.
TEXT ·convertF16ToF32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $32, R2
	BLT f16_to_f32_check8

f16_to_f32_loop32:
	// Load 32 float16s (64 bytes) into V0..V3, post-increment R1 by 64
	VLD1.P 64(R1), [V0.H8, V1.H8, V2.H8, V3.H8]
	// FCVTL / FCVTL2 V0..V3 -> V16..V23
	WORD $0x0E217810 // FCVTL  V16.4S, V0.4H
	WORD $0x4E217811 // FCVTL2 V17.4S, V0.8H
	WORD $0x0E217832 // FCVTL  V18.4S, V1.4H
	WORD $0x4E217833 // FCVTL2 V19.4S, V1.8H
	WORD $0x0E217854 // FCVTL  V20.4S, V2.4H
	WORD $0x4E217855 // FCVTL2 V21.4S, V2.8H
	WORD $0x0E217876 // FCVTL  V22.4S, V3.4H
	WORD $0x4E217877 // FCVTL2 V23.4S, V3.8H
	// Store 32 float32s (128 bytes) from V16..V23, post-increment R0 by 64 twice
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R0)
	VST1.P [V20.S4, V21.S4, V22.S4, V23.S4], 64(R0)

	SUB $32, R2, R2
	CMP $32, R2
	BGE f16_to_f32_loop32

f16_to_f32_check8:
	CMP $8, R2
	BLT f16_to_f32_check4

f16_to_f32_loop8:
	// Load 8 float16s (16 bytes) into V0, post-increment R1 by 16
	VLD1.P 16(R1), [V0.H8]
	// FCVTL V1.4S, V0.4H  (0x0E217800 | (0 << 5) | 1 = 0x0E217801)
	WORD $0x0E217801
	// FCVTL2 V2.4S, V0.8H (0x4E217800 | (0 << 5) | 2 = 0x4E217802)
	WORD $0x4E217802
	// Store 8 float32s (32 bytes) from V1, V2, post-increment R0 by 32
	VST1.P [V1.S4, V2.S4], 32(R0)

	SUB $8, R2, R2
	CMP $8, R2
	BGE f16_to_f32_loop8

f16_to_f32_check4:
	CMP $4, R2
	BLT f16_to_f32_done

	VLD1.P 8(R1), [V0.H4]
	WORD $0x0E217801 // FCVTL V1.4S, V0.4H
	VST1.P [V1.S4], 16(R0)

f16_to_f32_done:
	RET

// func convertF32ToF16NEONAsm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 4) float32 values at src to Float16 at dst using NEON FCVTN / FCVTN2.
TEXT ·convertF32ToF16NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $32, R2
	BLT f32_to_f16_check8

f32_to_f16_loop32:
	// Load 32 float32s (128 bytes) into V16..V23
	VLD1.P 64(R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	// FCVTN (lower halves) first, then FCVTN2 (upper halves) to break back-to-back destination dependencies
	WORD $0x0E216A00 // FCVTN  V0.4H, V16.4S
	WORD $0x0E216A41 // FCVTN  V1.4H, V18.4S
	WORD $0x0E216A82 // FCVTN  V2.4H, V20.4S
	WORD $0x0E216AC3 // FCVTN  V3.4H, V22.4S
	WORD $0x4E216A20 // FCVTN2 V0.8H, V17.4S
	WORD $0x4E216A61 // FCVTN2 V1.8H, V19.4S
	WORD $0x4E216AA2 // FCVTN2 V2.8H, V21.4S
	WORD $0x4E216AE3 // FCVTN2 V3.8H, V23.4S
	// Store 32 float16s (64 bytes) from V0..V3
	VST1.P [V0.H8, V1.H8, V2.H8, V3.H8], 64(R0)

	SUB $32, R2, R2
	CMP $32, R2
	BGE f32_to_f16_loop32

f32_to_f16_check8:
	CMP $8, R2
	BLT f32_to_f16_check4

f32_to_f16_loop8:
	// Load 8 float32s (32 bytes) into V1, V2, post-increment R1 by 32
	VLD1.P 32(R1), [V1.S4, V2.S4]
	// FCVTN V0.4H, V1.4S  (0x0E216800 | (1 << 5) | 0 = 0x0E216820)
	WORD $0x0E216820
	// FCVTN2 V0.8H, V2.4S (0x4E216800 | (2 << 5) | 0 = 0x4E216840)
	WORD $0x4E216840
	// Store 8 float16s (16 bytes) from V0, post-increment R0 by 16
	VST1.P [V0.H8], 16(R0)

	SUB $8, R2, R2
	CMP $8, R2
	BGE f32_to_f16_loop8

f32_to_f16_check4:
	CMP $4, R2
	BLT f32_to_f16_done

	VLD1.P 16(R1), [V1.S4]
	WORD $0x0E216820 // FCVTN V0.4H, V1.4S
	VST1.P [V0.H4], 8(R0)

f32_to_f16_done:
	RET

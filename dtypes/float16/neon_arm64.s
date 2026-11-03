// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// func convertF16ToF32NEONAsm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 8) Float16 values at src to float32 at dst using NEON FCVTL / FCVTL2.
TEXT ·convertF16ToF32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $0, R2
	BLE f16_to_f32_done

f16_to_f32_loop:
	// Load 8 float16s (16 bytes) into V0, post-increment R1 by 16
	VLD1.P 16(R1), [V0.H8]
	// FCVTL V1.4S, V0.4H  (0x0E217800 | (0 << 5) | 1 = 0x0E217801)
	WORD $0x0E217801
	// FCVTL2 V2.4S, V0.8H (0x4E217800 | (0 << 5) | 2 = 0x4E217802)
	WORD $0x4E217802
	// Store 8 float32s (32 bytes) from V1, V2, post-increment R0 by 32
	VST1.P [V1.S4, V2.S4], 32(R0)

	SUBS $8, R2, R2
	BGT f16_to_f32_loop

f16_to_f32_done:
	RET

// func convertF32ToF16NEONAsm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 8) float32 values at src to Float16 at dst using NEON FCVTN / FCVTN2.
TEXT ·convertF32ToF16NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $0, R2
	BLE f32_to_f16_done

f32_to_f16_loop:
	// Load 8 float32s (32 bytes) into V1, V2, post-increment R1 by 32
	VLD1.P 32(R1), [V1.S4, V2.S4]
	// FCVTN V0.4H, V1.4S  (0x0E216800 | (1 << 5) | 0 = 0x0E216820)
	WORD $0x0E216820
	// FCVTN2 V0.8H, V2.4S (0x4E216800 | (2 << 5) | 0 = 0x4E216840)
	WORD $0x4E216840
	// Store 8 float16s (16 bytes) from V0, post-increment R0 by 16
	VST1.P [V0.H8], 16(R0)

	SUBS $8, R2, R2
	BGT f32_to_f16_loop

f32_to_f16_done:
	RET

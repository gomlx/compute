// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// func convertBF16ToF32NEONAsm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 8) BFloat16 values at src to float32 at dst using NEON SHLL / SHLL2 (#16).
TEXT ·convertBF16ToF32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $0, R2
	BLE bf16_to_f32_done

bf16_to_f32_loop:
	// Load 8 bfloat16s (16 bytes) into V0, post-increment R1 by 16
	VLD1.P 16(R1), [V0.H8]
	// SHLL V1.4S, V0.4H, #16  (0x2E613800 | (0 << 5) | 1 = 0x2E613801)
	WORD $0x2E613801
	// SHLL2 V2.4S, V0.8H, #16 (0x6E613800 | (0 << 5) | 2 = 0x6E613802)
	WORD $0x6E613802
	// Store 8 float32s (32 bytes) from V1, V2, post-increment R0 by 32
	VST1.P [V1.S4, V2.S4], 32(R0)

	SUBS $8, R2, R2
	BGT bf16_to_f32_loop

bf16_to_f32_done:
	RET

// func convertF32ToBF16NEONAsm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 8) float32 values at src to BFloat16 at dst using round-to-nearest-even in NEON.
TEXT ·convertF32ToBF16NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $0, R2
	BLE f32_to_bf16_done

	// V6 = BroadcastUint32x4(0x7FFF)
	MOVD $0x7FFF, R3
	VDUP R3, V6.S4
	// V7 = BroadcastUint32x4(1)
	MOVD $1, R4
	VDUP R4, V7.S4

f32_to_bf16_loop:
	// Load 8 float32s (32 bytes) into V1, V2, post-increment R1 by 32
	VLD1.P 32(R1), [V1.S4, V2.S4]

	// FCMEQ V3.4S, V1.4S, V1.4S (0x0E20E400 | (1<<16) | (1<<5) | 3 = 0x0E21E423)
	WORD $0x0E21E423
	// FCMEQ V4.4S, V2.4S, V2.4S (0x0E20E400 | (2<<16) | (2<<5) | 4 = 0x0E22E444)
	WORD $0x0E22E444

	// USHR V16.4S, V1.4S, #16 (0x2F300400 | (1<<5) | 16 = 0x2F300430)
	WORD $0x2F300430
	// USHR V17.4S, V2.4S, #16 (0x2F300400 | (2<<5) | 17 = 0x2F300451)
	WORD $0x2F300451

	// V16 = (V1 >> 16) & 1; V17 = (V2 >> 16) & 1
	VAND V7.B16, V16.B16, V16.B16
	VAND V7.B16, V17.B16, V17.B16

	// V16 = (0x7FFF + lsb) & notNaN
	VADD V6.S4, V16.S4, V16.S4
	VADD V6.S4, V17.S4, V17.S4
	VAND V3.B16, V16.B16, V16.B16
	VAND V4.B16, V17.B16, V17.B16

	// V1 = V1 + bias; V2 = V2 + bias
	VADD V16.S4, V1.S4, V1.S4
	VADD V17.S4, V2.S4, V2.S4

	// SHRN V0.4H, V1.4S, #16  (0x0F108400 | (1<<5) | 0 = 0x0F108420)
	WORD $0x0F108420
	// SHRN2 V0.8H, V2.4S, #16 (0x4F108400 | (2<<5) | 0 = 0x4F108440)
	WORD $0x4F108440

	// Store 8 bfloat16s (16 bytes) from V0, post-increment R0 by 16
	VST1.P [V0.H8], 16(R0)

	SUBS $8, R2, R2
	BGT f32_to_bf16_loop

f32_to_bf16_done:
	RET

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

#include "textflag.h"

// func convertBF16ToF32NEONAsm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 4) BFloat16 values at src to float32 at dst using NEON SHLL / SHLL2 (#16).
TEXT ·convertBF16ToF32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $32, R2
	BLT bf16_to_f32_check8

bf16_to_f32_loop32:
	// Load 32 bfloat16s (64 bytes) into V0..V3, post-increment R1 by 64
	VLD1.P 64(R1), [V0.H8, V1.H8, V2.H8, V3.H8]
	// SHLL / SHLL2 V0..V3 -> V16..V23 (#16)
	WORD $0x2E613810 // SHLL  V16.4S, V0.4H, #16
	WORD $0x6E613811 // SHLL2 V17.4S, V0.8H, #16
	WORD $0x2E613832 // SHLL  V18.4S, V1.4H, #16
	WORD $0x6E613833 // SHLL2 V19.4S, V1.8H, #16
	WORD $0x2E613854 // SHLL  V20.4S, V2.4H, #16
	WORD $0x6E613855 // SHLL2 V21.4S, V2.8H, #16
	WORD $0x2E613876 // SHLL  V22.4S, V3.4H, #16
	WORD $0x6E613877 // SHLL2 V23.4S, V3.8H, #16
	// Store 32 float32s (128 bytes) from V16..V23, post-increment R0 by 64 twice
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R0)
	VST1.P [V20.S4, V21.S4, V22.S4, V23.S4], 64(R0)

	SUB $32, R2, R2
	CMP $32, R2
	BGE bf16_to_f32_loop32

bf16_to_f32_check8:
	CMP $8, R2
	BLT bf16_to_f32_check4

bf16_to_f32_loop8:
	// Load 8 bfloat16s (16 bytes) into V0, post-increment R1 by 16
	VLD1.P 16(R1), [V0.H8]
	// SHLL V1.4S, V0.4H, #16  (0x2E613800 | (0 << 5) | 1 = 0x2E613801)
	WORD $0x2E613801
	// SHLL2 V2.4S, V0.8H, #16 (0x6E613800 | (0 << 5) | 2 = 0x6E613802)
	WORD $0x6E613802
	// Store 8 float32s (32 bytes) from V1, V2, post-increment R0 by 32
	VST1.P [V1.S4, V2.S4], 32(R0)

	SUB $8, R2, R2
	CMP $8, R2
	BGE bf16_to_f32_loop8

bf16_to_f32_check4:
	CMP $4, R2
	BLT bf16_to_f32_done

	VLD1.P 8(R1), [V0.H4]
	WORD $0x2E613801 // SHLL V1.4S, V0.4H, #16
	VST1.P [V1.S4], 16(R0)

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

	CMP $16, R2
	BLT f32_to_bf16_check8

f32_to_bf16_loop16:
	// Load 16 float32s (64 bytes) into V2, V3, V4, V5
	VLD1.P 64(R1), [V2.S4, V3.S4, V4.S4, V5.S4]

	// FCMEQ.4S notNaN masks (Q=1: 0x4E20E400 | (Rm<<16) | (Rn<<5) | Rd)
	WORD $0x4E22E450 // FCMEQ V16.4S, V2.4S, V2.4S
	WORD $0x4E23E471 // FCMEQ V17.4S, V3.4S, V3.4S
	WORD $0x4E24E492 // FCMEQ V18.4S, V4.4S, V4.4S
	WORD $0x4E25E4B3 // FCMEQ V19.4S, V5.4S, V5.4S

	// USHR.4S #16 (Q=1: 0x6F300400 | (Rn<<5) | Rd)
	WORD $0x6F300454 // USHR V20.4S, V2.4S, #16
	WORD $0x6F300475 // USHR V21.4S, V3.4S, #16
	WORD $0x6F300496 // USHR V22.4S, V4.4S, #16
	WORD $0x6F3004B7 // USHR V23.4S, V5.4S, #16

	// lsb = (x >> 16) & 1
	VAND V7.B16, V20.B16, V20.B16
	VAND V7.B16, V21.B16, V21.B16
	VAND V7.B16, V22.B16, V22.B16
	VAND V7.B16, V23.B16, V23.B16

	// bias = (0x7FFF + lsb) & notNaN
	VADD V6.S4, V20.S4, V20.S4
	VADD V6.S4, V21.S4, V21.S4
	VADD V6.S4, V22.S4, V22.S4
	VADD V6.S4, V23.S4, V23.S4
	VAND V16.B16, V20.B16, V20.B16
	VAND V17.B16, V21.B16, V21.B16
	VAND V18.B16, V22.B16, V22.B16
	VAND V19.B16, V23.B16, V23.B16

	// rounded = x + bias
	VADD V20.S4, V2.S4, V2.S4
	VADD V21.S4, V3.S4, V3.S4
	VADD V22.S4, V4.S4, V4.S4
	VADD V23.S4, V5.S4, V5.S4

	// SHRN (lower halves) first, then SHRN2 (upper halves)
	WORD $0x0F108440 // SHRN  V0.4H, V2.4S, #16
	WORD $0x0F108481 // SHRN  V1.4H, V4.4S, #16
	WORD $0x4F108460 // SHRN2 V0.8H, V3.4S, #16
	WORD $0x4F1084A1 // SHRN2 V1.8H, V5.4S, #16

	// Store 16 bfloat16s (32 bytes) from V0, V1
	VST1.P [V0.H8, V1.H8], 32(R0)

	SUB $16, R2, R2
	CMP $16, R2
	BGE f32_to_bf16_loop16

f32_to_bf16_check8:
	CMP $8, R2
	BLT f32_to_bf16_done

	// Load 8 float32s (32 bytes) into V1, V2, post-increment R1 by 32
	VLD1.P 32(R1), [V1.S4, V2.S4]

	WORD $0x4E21E423 // FCMEQ V3.4S, V1.4S, V1.4S
	WORD $0x4E22E444 // FCMEQ V4.4S, V2.4S, V2.4S
	WORD $0x6F300430 // USHR  V16.4S, V1.4S, #16
	WORD $0x6F300451 // USHR  V17.4S, V2.4S, #16

	VAND V7.B16, V16.B16, V16.B16
	VAND V7.B16, V17.B16, V17.B16
	VADD V6.S4, V16.S4, V16.S4
	VADD V6.S4, V17.S4, V17.S4
	VAND V3.B16, V16.B16, V16.B16
	VAND V4.B16, V17.B16, V17.B16
	VADD V16.S4, V1.S4, V1.S4
	VADD V17.S4, V2.S4, V2.S4

	WORD $0x0F108420 // SHRN  V0.4H, V1.4S, #16
	WORD $0x4F108440 // SHRN2 V0.8H, V2.4S, #16

	VST1.P [V0.H8], 16(R0)

f32_to_bf16_done:
	RET

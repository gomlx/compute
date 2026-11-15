// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

#include "textflag.h"

// func float32ToBFloat16ARM64BF16Asm(dst, src unsafe.Pointer, n int)
// Converts n (multiple of 4) float32 values at src to BFloat16 at dst using ARMv8.6-A FEAT_BF16 BFCVTN / BFCVTN2.
TEXT ·float32ToBFloat16ARM64BF16Asm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

	CMP $32, R2
	BLT f32_to_bf16_hw_check8

f32_to_bf16_hw_loop32:
	// Load 32 float32s (128 bytes) into V16..V23
	VLD1.P 64(R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	// BFCVTN (lower halves) first, then BFCVTN2 (upper halves) to break back-to-back destination dependencies
	WORD $0x0EA16A00 // BFCVTN  V0.4H, V16.4S
	WORD $0x0EA16A41 // BFCVTN  V1.4H, V18.4S
	WORD $0x0EA16A82 // BFCVTN  V2.4H, V20.4S
	WORD $0x0EA16AC3 // BFCVTN  V3.4H, V22.4S
	WORD $0x4EA16A20 // BFCVTN2 V0.8H, V17.4S
	WORD $0x4EA16A61 // BFCVTN2 V1.8H, V19.4S
	WORD $0x4EA16AA2 // BFCVTN2 V2.8H, V21.4S
	WORD $0x4EA16AE3 // BFCVTN2 V3.8H, V23.4S
	// Store 32 bfloat16s (64 bytes) from V0..V3
	VST1.P [V0.H8, V1.H8, V2.H8, V3.H8], 64(R0)

	SUB $32, R2, R2
	CMP $32, R2
	BGE f32_to_bf16_hw_loop32

f32_to_bf16_hw_check8:
	CMP $8, R2
	BLT f32_to_bf16_hw_check4

f32_to_bf16_hw_loop8:
	VLD1.P 32(R1), [V1.S4, V2.S4]
	WORD $0x0EA16820 // BFCVTN  V0.4H, V1.4S
	WORD $0x4EA16840 // BFCVTN2 V0.8H, V2.4S
	VST1.P [V0.H8], 16(R0)

	SUB $8, R2, R2
	CMP $8, R2
	BGE f32_to_bf16_hw_loop8

f32_to_bf16_hw_check4:
	CMP $4, R2
	BLT f32_to_bf16_hw_done

	VLD1.P 16(R1), [V1.S4]
	WORD $0x0EA16820 // BFCVTN V0.4H, V1.4S
	VST1.P [V0.H4], 8(R0)

f32_to_bf16_hw_done:
	RET

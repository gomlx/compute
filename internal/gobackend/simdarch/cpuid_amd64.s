// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

#include "textflag.h"

// func hasAVX512BF16() bool
// Queries CPUID.(EAX=7, ECX=1):EAX[bit 5] (AVX512_BF16).
// Caller must already have verified AVX-512 support (which guarantees leaf 7 and OS ZMM state support).
TEXT ·hasAVX512BF16(SB), NOSPLIT, $0-1
	// Check max subleaf for CPUID leaf 7 (EAX=7, ECX=0 -> EAX = max subleaf).
	MOVL $7, AX
	XORL CX, CX
	CPUID
	CMPL AX, $1
	JB no_bf16

	// Query CPUID.(EAX=7, ECX=1):EAX[bit 5] = AVX512_BF16.
	MOVL $7, AX
	MOVL $1, CX
	CPUID
	SHRL $5, AX
	ANDL $1, AX
	MOVB AL, ret+0(FP)
	RET

no_bf16:
	MOVB $0, ret+0(FP)
	RET

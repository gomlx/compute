// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && arm64 && goexperiment.simd

#include "textflag.h"

// func ReadTPIDR_EL0() uint64
//
// On Darwin/ARM64 (XNU), TPIDR_EL0 encodes the current core's topology:
//   - bits [11:0]  (TPIDR_EL0 & 0xfff):        physical CPU ID (0..N-1)
//   - bits [19:12] ((TPIDR_EL0 >> 12) & 0xff): hardware L2 cache cluster ID (0..NumSMEUnits-1)
TEXT ·ReadTPIDR_EL0(SB), NOSPLIT, $0-8
	MRS	TPIDR_EL0, R0
	MOVD	R0, ret+0(FP)
	RET

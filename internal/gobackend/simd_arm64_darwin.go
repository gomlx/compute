// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && arm64 && goexperiment.simd

package gobackend

import "syscall"

func sysctlEnabled(name string) bool {
	val, err := syscall.SysctlUint32(name)
	return err == nil && val != 0
}

func detectARM64Features() arm64Features {
	return arm64Features{
		hasASIMD:   true, // AdvSIMD (NEON) is mandatory on all Apple Silicon / ARMv8-A Darwin CPUs.
		hasFP16:    sysctlEnabled("hw.optional.arm.FEAT_FP16") || sysctlEnabled("hw.optional.neon_fp16"),
		hasBF16:    sysctlEnabled("hw.optional.arm.FEAT_BF16"),
		hasDotProd: sysctlEnabled("hw.optional.arm.FEAT_DotProd"),
		hasI8MM:    sysctlEnabled("hw.optional.arm.FEAT_I8MM"),
		hasSVE:     sysctlEnabled("hw.optional.arm.FEAT_SVE"),
		hasSME:     sysctlEnabled("hw.optional.arm.FEAT_SME") || sysctlEnabled("hw.optional.arm.FEAT_SME2"),
	}
}

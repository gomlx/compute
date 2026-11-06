// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package gobackend

import (
	"github.com/gomlx/compute/support/envutil"
)

type arm64Features struct {
	hasASIMD   bool
	hasFP16    bool
	hasBF16    bool
	hasDotProd bool
	hasI8MM    bool
	hasSVE     bool
	hasSME     bool
}

var (
	// X86 SIMD flags are always false on arm64.
	IsAVX2Allowed   = false
	IsAVX512Allowed = false

	arm64CPU = detectARM64Features()

	// IsNEONAllowed returns true if ARM64 AdvSIMD (NEON) is supported and not disabled by GOMLX_GO_SIMD_NEON.
	IsNEONAllowed = envutil.MustReadBool(envutil.GoBackendSIMD_NEON, true) && arm64CPU.hasASIMD

	// IsARM64FP16Allowed returns true if FEAT_FP16 is supported and NEON is allowed.
	IsARM64FP16Allowed = IsNEONAllowed && arm64CPU.hasFP16

	// IsARM64BF16Allowed returns true if FEAT_BF16 is supported and NEON is allowed.
	IsARM64BF16Allowed = IsNEONAllowed && arm64CPU.hasBF16

	// IsARM64DotProdAllowed returns true if FEAT_DotProd (SDOT/UDOT) is supported and NEON is allowed.
	IsARM64DotProdAllowed = IsNEONAllowed && arm64CPU.hasDotProd

	// IsARM64I8MMAllowed returns true if FEAT_I8MM (SMMLA/UMMLA) is supported and NEON is allowed.
	IsARM64I8MMAllowed = IsNEONAllowed && arm64CPU.hasI8MM

	// IsSVEAllowed returns true if ARM64 SVE is supported by the CPU and not disabled by GOMLX_GO_SIMD_SVE.
	IsSVEAllowed = envutil.MustReadBool(envutil.GoBackendSIMD_SVE, true) && arm64CPU.hasSVE

	// IsSMEAllowed returns true if ARM64 SME is supported by the CPU and not disabled by GOMLX_GO_SIMD_SME.
	IsSMEAllowed = envutil.MustReadBool(envutil.GoBackendSIMD_SME, true) && arm64CPU.hasSME
)

func init() {
	ActiveArchitectures = append(ActiveArchitectures, "SIMD")
	if IsNEONAllowed {
		ActiveArchitectures = append(ActiveArchitectures, "NEON")
	}
	if IsSVEAllowed {
		ActiveArchitectures = append(ActiveArchitectures, "SVE")
	}
	if IsSMEAllowed {
		ActiveArchitectures = append(ActiveArchitectures, "SME")
	}
}

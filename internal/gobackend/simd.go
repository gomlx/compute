// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import "github.com/gomlx/compute/internal/gobackend/simdarch"

var (
	// ActiveArchitectures lists the active OS/architecture and enabled SIMD architectures.
	ActiveArchitectures = simdarch.ActiveArchitectures

	// IsAVX2Allowed returns true if AVX2 is supported by the CPU and not disabled by GOMLX_GO_SIMD_AVX2.
	IsAVX2Allowed = simdarch.IsAVX2Allowed

	// IsAVXVNNIAllowed returns true if AVX-VNNI is supported and AVX2 is allowed.
	IsAVXVNNIAllowed = simdarch.IsAVXVNNIAllowed

	// IsAVX512Allowed returns true if AVX-512 is supported by the CPU and not disabled by GOMLX_GO_SIMD_AVX512.
	IsAVX512Allowed = simdarch.IsAVX512Allowed

	// IsAVX512VNNIAllowed returns true if AVX-512 VNNI is supported and AVX-512 is allowed.
	IsAVX512VNNIAllowed = simdarch.IsAVX512VNNIAllowed

	// IsNEONAllowed returns true if ARM64 AdvSIMD (NEON) is supported and not disabled by GOMLX_GO_SIMD_NEON.
	IsNEONAllowed = simdarch.IsNEONAllowed

	// IsARM64FP16Allowed returns true if FEAT_FP16 is supported and NEON is allowed.
	IsARM64FP16Allowed = simdarch.IsARM64FP16Allowed

	// IsARM64BF16Allowed returns true if FEAT_BF16 is supported and NEON is allowed.
	IsARM64BF16Allowed = simdarch.IsARM64BF16Allowed

	// IsARM64DotProdAllowed returns true if FEAT_DotProd (SDOT/UDOT) is supported and NEON is allowed.
	IsARM64DotProdAllowed = simdarch.IsARM64DotProdAllowed

	// IsARM64I8MMAllowed returns true if FEAT_I8MM (SMMLA/UMMLA) is supported and NEON is allowed.
	IsARM64I8MMAllowed = simdarch.IsARM64I8MMAllowed

	// IsSVEAllowed returns true if ARM64 SVE is supported by the CPU and not disabled by GOMLX_GO_SIMD_SVE.
	IsSVEAllowed = simdarch.IsSVEAllowed

	// IsSMEAllowed returns true if ARM64 SME is supported by the CPU and not disabled by GOMLX_GO_SIMD_SME.
	IsSMEAllowed = simdarch.IsSMEAllowed
)

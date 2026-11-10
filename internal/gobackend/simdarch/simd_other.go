// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build (!amd64 && !arm64) || !goexperiment.simd

package simdarch

var (
	// IsAVX512Allowed returns false on non-amd64 architectures or when SIMD is disabled.
	IsAVX512Allowed = false

	// IsAVX2Allowed returns false on non-amd64 architectures or when SIMD is disabled.
	IsAVX2Allowed = false

	// ARM64 SIMD flags return false on non-arm64 architectures or when SIMD is disabled.
	IsNEONAllowed         = false
	IsARM64FP16Allowed    = false
	IsARM64BF16Allowed    = false
	IsARM64DotProdAllowed = false
	IsARM64I8MMAllowed    = false
	IsSVEAllowed          = false
	IsSMEAllowed          = false
)

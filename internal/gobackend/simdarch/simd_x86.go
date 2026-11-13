// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package simdarch

import (
	"simd/archsimd"

	"github.com/gomlx/compute/support/envutil"
)

var (
	// IsAVX2Allowed returns true if AVX2 is supported by the CPU and not disabled by GOMLX_GO_SIMD_AVX2.
	// Note that this only controls AVX2-specific implementations (e.g. specialized matmul and activations);
	// generic portable SIMD operations will still use hardware vector features if available.
	IsAVX2Allowed = envutil.MustReadBool(envutil.GoBackendSIMD_AVX2, true) && archsimd.X86.AVX2()

	// IsAVXVNNIAllowed returns true if VEX-encoded AVX-VNNI is supported and AVX2 is allowed.
	IsAVXVNNIAllowed = IsAVX2Allowed && archsimd.X86.AVXVNNI()

	// IsAVX512Allowed returns true if AVX-512 is supported by the CPU and not disabled by GOMLX_GO_SIMD_AVX512.
	// Note that this only controls AVX512-specific implementations (e.g. specialized matmul and activations);
	// generic portable SIMD operations will still use hardware vector features if available.
	IsAVX512Allowed = envutil.MustReadBool(envutil.GoBackendSIMD_AVX512, true) && archsimd.X86.AVX512()

	// IsAVX512VNNIAllowed returns true if AVX-512 VNNI (VPDPBUSD) is supported and AVX-512 is allowed.
	IsAVX512VNNIAllowed = IsAVX512Allowed && archsimd.X86.AVX512VNNI()

	// ARM64 feature flags are always false on amd64.
	IsNEONAllowed         = false
	IsARM64FP16Allowed    = false
	IsARM64BF16Allowed    = false
	IsARM64DotProdAllowed = false
	IsARM64I8MMAllowed    = false
	IsSVEAllowed          = false
	IsSMEAllowed          = false
)

func init() {
	ActiveArchitectures = append(ActiveArchitectures, "SIMD")
	if IsAVX2Allowed {
		ActiveArchitectures = append(ActiveArchitectures, "AVX2")
	}
	if IsAVX512Allowed {
		ActiveArchitectures = append(ActiveArchitectures, "AVX512")
	}
}

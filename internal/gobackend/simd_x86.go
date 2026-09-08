// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package gobackend

import (
	"simd/archsimd"

	"github.com/gomlx/compute/support/envutil"
)

// IsAVX512Allowed returns true if AVX-512 is supported by the CPU and not disabled by GOMLX_GO_SIMD_AVX512.
func IsAVX512Allowed() bool {
	return envutil.MustReadBool(envutil.GoBackendSIMD_AVX512, true) && archsimd.X86.AVX512()
}

// IsAVX2Allowed returns true if AVX2 is supported by the CPU and not disabled by GOMLX_GO_SIMD_AVX2.
func IsAVX2Allowed() bool {
	return envutil.MustReadBool(envutil.GoBackendSIMD_AVX2, true) && archsimd.X86.AVX2()
}

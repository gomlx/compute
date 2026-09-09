// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build (!amd64) || (!goexperiment.simd)

package gobackend

// IsAVX512Allowed returns false on non-amd64 architectures or when SIMD is disabled.
func IsAVX512Allowed() bool {
	return false
}

// IsAVX2Allowed returns false on non-amd64 architectures or when SIMD is disabled.
func IsAVX2Allowed() bool {
	return false
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !darwin && !linux && arm64 && goexperiment.simd

package simdarch

func detectARM64Features() arm64Features {
	return arm64Features{
		hasASIMD: true, // AdvSIMD (NEON) is mandatory in ARMv8-A.
	}
}

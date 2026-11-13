// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func registerLayerNormNEON() {
	gobackend.SetLayerNormTrailingArchDispatcher(gobackend.PriorityNEON, DispatchLayerNormNEON)
	gobackend.SetLayerNormResidualTrailingArchDispatcher(gobackend.PriorityNEON, DispatchLayerNormResidualNEON)
}

//go:noescape
func layerNormFloat32NEONAsm(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)

//go:noescape
func layerNormResidualFloat32NEONAsm(in, residual, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)

// DispatchLayerNormNEON executes LayerNorm on trailing contiguous axes using ARM64 NEON assembly.
func DispatchLayerNormNEON(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float64, dtype dtypes.DType) bool {
	if dtype == dtypes.Float32 {
		layerNormFloat32NEONAsm(in, out, gamma, beta, outerSize, normSize, float32(epsilon))
		return true
	}
	return false
}

// DispatchLayerNormResidualNEON executes fused (in + residual) + LayerNorm on trailing contiguous axes using ARM64 NEON assembly.
func DispatchLayerNormResidualNEON(in, residual, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float64, dtype dtypes.DType) bool {
	if dtype == dtypes.Float32 {
		layerNormResidualFloat32NEONAsm(in, residual, out, gamma, beta, outerSize, normSize, float32(epsilon))
		return true
	}
	return false
}

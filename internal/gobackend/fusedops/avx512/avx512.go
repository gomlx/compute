// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX512Allowed {
		registerAVX512()
	}
}

func registerAVX512() {
	gobackend.SetLayerNormTrailingArchDispatcher(gobackend.PriorityArch+1, DispatchLayerNormAVX512)
}

//go:noescape
func layerNormFloat32AVX512(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)

//go:noescape
func layerNormFloat64AVX512(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float64)

// DispatchLayerNormAVX512 executes LayerNorm on trailing contiguous axes using AVX-512.
func DispatchLayerNormAVX512(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float64, dtype dtypes.DType) bool {
	switch dtype {
	case dtypes.Float32:
		layerNormFloat32AVX512(in, out, gamma, beta, outerSize, normSize, float32(epsilon))
		return true
	case dtypes.Float64:
		layerNormFloat64AVX512(in, out, gamma, beta, outerSize, normSize, epsilon)
		return true
	}
	return false
}

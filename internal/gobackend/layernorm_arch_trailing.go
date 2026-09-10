// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes"
)

// LayerNormTrailingArchFn is a function hook to execute LayerNorm on trailing contiguous axes
// using architecture-specific kernels (e.g., AVX2, AVX-512).
//
// Parameters:
//   - inPtr, outPtr: unsafe.Pointer to flat input/output data
//   - gammaPtr, betaPtr: optional unsafe.Pointer to 1D scale/bias parameter buffers (may be nil)
//   - outerSize: number of outer rows to normalize
//   - normSize: number of elements along normalization axis per row
//   - epsilon: numerical stability epsilon
//   - dtype: element data type (e.g. Float32, Float64)
type LayerNormTrailingArchFn func(
	inPtr, outPtr, gammaPtr, betaPtr unsafe.Pointer,
	outerSize, normSize int,
	epsilon float64,
	dtype dtypes.DType,
) bool

var (
	layerNormTrailingArchFn       LayerNormTrailingArchFn
	layerNormTrailingArchPriority RegisterPriority
)

// SetLayerNormTrailingArchDispatcher registers the architecture-specific LayerNorm dispatcher.
// Higher priority replaces lower priority dispatcher.
func SetLayerNormTrailingArchDispatcher(priority RegisterPriority, fn LayerNormTrailingArchFn) {
	if priority >= layerNormTrailingArchPriority {
		layerNormTrailingArchPriority = priority
		layerNormTrailingArchFn = fn
	}
}

// GetLayerNormTrailingArchDispatcher returns the registered architecture-specific LayerNorm dispatcher, if any.
func GetLayerNormTrailingArchDispatcher() LayerNormTrailingArchFn {
	return layerNormTrailingArchFn
}

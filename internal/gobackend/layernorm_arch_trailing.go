// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
)

// LayerNormTrailingArchFn is a function hook to execute LayerNorm on trailing contiguous axes
// using architecture-specific kernels (e.g., AVX2, AVX-512, NEON).
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

type layerNormTrailingArchEntry struct {
	priority RegisterPriority
	fn       LayerNormTrailingArchFn
}

var (
	layerNormTrailingArchEntries []layerNormTrailingArchEntry
	layerNormTrailingArchFn      LayerNormTrailingArchFn
)

// SetLayerNormTrailingArchDispatcher registers an architecture-specific LayerNorm dispatcher.
// Dispatchers are tried in descending order of priority; if a higher-priority dispatcher returns false, the next one is tried.
func SetLayerNormTrailingArchDispatcher(priority RegisterPriority, fn LayerNormTrailingArchFn) {
	if priority < 0 || fn == nil {
		return
	}
	idx := 0
	for idx < len(layerNormTrailingArchEntries) && layerNormTrailingArchEntries[idx].priority > priority {
		idx++
	}
	if idx < len(layerNormTrailingArchEntries) && layerNormTrailingArchEntries[idx].priority == priority {
		layerNormTrailingArchEntries[idx].fn = fn
	} else {
		layerNormTrailingArchEntries = slices.Insert(layerNormTrailingArchEntries, idx, layerNormTrailingArchEntry{priority: priority, fn: fn})
	}
	rebuildLayerNormTrailingArchFn()
}

func rebuildLayerNormTrailingArchFn() {
	switch len(layerNormTrailingArchEntries) {
	case 0:
		layerNormTrailingArchFn = nil
	case 1:
		layerNormTrailingArchFn = layerNormTrailingArchEntries[0].fn
	default:
		entries := slices.Clone(layerNormTrailingArchEntries)
		layerNormTrailingArchFn = func(
			inPtr, outPtr, gammaPtr, betaPtr unsafe.Pointer,
			outerSize, normSize int,
			epsilon float64,
			dtype dtypes.DType,
		) bool {
			for _, e := range entries {
				if e.fn(inPtr, outPtr, gammaPtr, betaPtr, outerSize, normSize, epsilon, dtype) {
					return true
				}
			}
			return false
		}
	}
}

// GetLayerNormTrailingArchDispatcher returns the registered architecture-specific LayerNorm dispatcher, if any.
func GetLayerNormTrailingArchDispatcher() LayerNormTrailingArchFn {
	return layerNormTrailingArchFn
}

// LayerNormResidualTrailingArchFn is a function hook to execute fused (in + residual) + LayerNorm
// on trailing contiguous axes using architecture-specific kernels (e.g., AVX2, AVX-512, NEON).
//
// Parameters:
//   - inPtr, residualPtr, outPtr: unsafe.Pointer to flat input, residual, and output data
//   - gammaPtr, betaPtr: optional unsafe.Pointer to 1D scale/bias parameter buffers (may be nil)
//   - outerSize: number of outer rows to normalize
//   - normSize: number of elements along normalization axis per row
//   - epsilon: numerical stability epsilon
//   - dtype: element data type (e.g. Float32, Float64)
type LayerNormResidualTrailingArchFn func(
	inPtr, residualPtr, outPtr, gammaPtr, betaPtr unsafe.Pointer,
	outerSize, normSize int,
	epsilon float64,
	dtype dtypes.DType,
) bool

type layerNormResidualTrailingArchEntry struct {
	priority RegisterPriority
	fn       LayerNormResidualTrailingArchFn
}

var (
	layerNormResidualTrailingArchEntries []layerNormResidualTrailingArchEntry
	layerNormResidualTrailingArchFn      LayerNormResidualTrailingArchFn
)

// SetLayerNormResidualTrailingArchDispatcher registers an architecture-specific fused residual+LayerNorm dispatcher.
func SetLayerNormResidualTrailingArchDispatcher(priority RegisterPriority, fn LayerNormResidualTrailingArchFn) {
	if priority < 0 || fn == nil {
		return
	}
	idx := 0
	for idx < len(layerNormResidualTrailingArchEntries) && layerNormResidualTrailingArchEntries[idx].priority > priority {
		idx++
	}
	if idx < len(layerNormResidualTrailingArchEntries) && layerNormResidualTrailingArchEntries[idx].priority == priority {
		layerNormResidualTrailingArchEntries[idx].fn = fn
	} else {
		layerNormResidualTrailingArchEntries = slices.Insert(layerNormResidualTrailingArchEntries, idx, layerNormResidualTrailingArchEntry{priority: priority, fn: fn})
	}
	rebuildLayerNormResidualTrailingArchFn()
}

func rebuildLayerNormResidualTrailingArchFn() {
	switch len(layerNormResidualTrailingArchEntries) {
	case 0:
		layerNormResidualTrailingArchFn = nil
	case 1:
		layerNormResidualTrailingArchFn = layerNormResidualTrailingArchEntries[0].fn
	default:
		entries := slices.Clone(layerNormResidualTrailingArchEntries)
		layerNormResidualTrailingArchFn = func(
			inPtr, residualPtr, outPtr, gammaPtr, betaPtr unsafe.Pointer,
			outerSize, normSize int,
			epsilon float64,
			dtype dtypes.DType,
		) bool {
			for _, e := range entries {
				if e.fn(inPtr, residualPtr, outPtr, gammaPtr, betaPtr, outerSize, normSize, epsilon, dtype) {
					return true
				}
			}
			return false
		}
	}
}

// GetLayerNormResidualTrailingArchDispatcher returns the registered architecture-specific fused residual+LayerNorm dispatcher, if any.
func GetLayerNormResidualTrailingArchDispatcher() LayerNormResidualTrailingArchFn {
	return layerNormResidualTrailingArchFn
}

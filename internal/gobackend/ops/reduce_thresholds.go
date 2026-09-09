// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package ops

import (
	"github.com/gomlx/compute/dtypes"
)

// reduceThresholdsConfig specifies the minimum dimension thresholds below which
// SIMD reduction falls back to generic (scalar) execution because the scalar loop
// is faster than the vector setup and horizontal reduction overhead.
type reduceThresholdsConfig struct {
	// TrailingMinB is the minimum inner dimension B for ReduceTrailing [A, B] -> [A].
	// If B < TrailingMinB[dtype], SIMD falls back to scalar.
	TrailingMinB map[dtypes.DType]int

	// LeadingMinB is the minimum inner dimension B for ReduceLeading [A, B] -> [B].
	// Vectorization happens along B; if B < LeadingMinB[dtype], SIMD falls back to scalar.
	LeadingMinB map[dtypes.DType]int

	// AllMinN is the minimum tensor size N for ReduceAll [N] -> [1].
	// If N < AllMinN[dtype], SIMD falls back to scalar.
	AllMinN map[dtypes.DType]int
}

// reduceThresholds stores the active thresholds. Initialized to defaults (e.g. ARM64 NEON),
// and overwritten on amd64 in reduce_thresholds_amd64.go depending on runtime CPU features.
var reduceThresholds = reduceThresholdsConfig{
	TrailingMinB: map[dtypes.DType]int{
		dtypes.Float32:  16,
		dtypes.Float64:  16,
		dtypes.Int32:    8,
		dtypes.Uint32:   8,
		dtypes.Int16:    16,
		dtypes.Uint16:   16,
		dtypes.Int8:     16,
		dtypes.Uint8:    16,
		dtypes.BFloat16: 16,
		dtypes.Float16:  8,
	},
	LeadingMinB: map[dtypes.DType]int{
		dtypes.Float32:  8,
		dtypes.Float64:  8,
		dtypes.Int32:    8,
		dtypes.Uint32:   8,
		dtypes.Int16:    8,
		dtypes.Uint16:   8,
		dtypes.Int8:     8,
		dtypes.Uint8:    8,
		dtypes.BFloat16: 4,
		dtypes.Float16:  4,
	},
	AllMinN: map[dtypes.DType]int{
		dtypes.Float32:  8,
		dtypes.Float64:  8,
		dtypes.Int32:    8,
		dtypes.Uint32:   8,
		dtypes.Int16:    8,
		dtypes.Uint16:   8,
		dtypes.Int8:     8,
		dtypes.Uint8:    8,
		dtypes.BFloat16: 8,
		dtypes.Float16:  8,
	},
}

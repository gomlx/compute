// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package ops

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

// avx2ReduceThresholds defines the crossover points measured on AVX2 hardware.
// Determined via TestFindReduceThresholds in reduce_bench_test.go.
var avx2ReduceThresholds = reduceThresholdsConfig{
	TrailingMinB: map[dtypes.DType]int{
		dtypes.Float32:  32,
		dtypes.Float64:  32,
		dtypes.Int32:    12,
		dtypes.Uint32:   12,
		dtypes.Int16:    16,
		dtypes.Uint16:   16,
		dtypes.Int8:     32,
		dtypes.Uint8:    32,
		dtypes.BFloat16: 32,
		dtypes.Float16:  8,
	},
	LeadingMinB: map[dtypes.DType]int{
		dtypes.Float32:  8,
		dtypes.Float64:  8,
		dtypes.Int32:    8,
		dtypes.Uint32:   8,
		dtypes.Int16:    8,
		dtypes.Uint16:   8,
		dtypes.Int8:     16,
		dtypes.Uint8:    16,
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
		dtypes.Int8:     16,
		dtypes.Uint8:    16,
		dtypes.BFloat16: 8,
		dtypes.Float16:  8,
	},
}

// avx512ReduceThresholds defines the crossover points for AVX-512 hardware.
// Initialized from AVX2 baseline; can be tuned with AVX-512 benchmark results.
var avx512ReduceThresholds = reduceThresholdsConfig{
	TrailingMinB: map[dtypes.DType]int{
		dtypes.Float32:  32,
		dtypes.Float64:  32,
		dtypes.Int32:    12,
		dtypes.Uint32:   12,
		dtypes.Int16:    16,
		dtypes.Uint16:   16,
		dtypes.Int8:     32,
		dtypes.Uint8:    32,
		dtypes.BFloat16: 32,
		dtypes.Float16:  8,
	},
	LeadingMinB: map[dtypes.DType]int{
		dtypes.Float32:  8,
		dtypes.Float64:  8,
		dtypes.Int32:    8,
		dtypes.Uint32:   8,
		dtypes.Int16:    8,
		dtypes.Uint16:   8,
		dtypes.Int8:     16,
		dtypes.Uint8:    16,
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
		dtypes.Int8:     16,
		dtypes.Uint8:    16,
		dtypes.BFloat16: 8,
		dtypes.Float16:  8,
	},
}

func init() {
	if gobackend.IsAVX512Allowed() {
		reduceThresholds = avx512ReduceThresholds
	} else if gobackend.IsAVX2Allowed() {
		reduceThresholds = avx2ReduceThresholds
	}
}

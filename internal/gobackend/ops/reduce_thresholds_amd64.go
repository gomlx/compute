// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package ops

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

// avx2ReduceThresholds defines the crossover points measured on AVX2 hardware.
// Determined via TestFindReduceThresholds in reduce_bench_test.go.
//
// Estimated on a 12th Gen Intel(R) Core(TM) i9-12900K
var avx2ReduceThresholds = ReduceThresholdsConfig{
	TrailingMinB: map[dtypes.DType]int{
		dtypes.Float32:  ThresholdNone,
		dtypes.Float64:  ThresholdNone,
		dtypes.Int32:    ThresholdNone,
		dtypes.Uint32:   ThresholdNone,
		dtypes.Int64:    ThresholdNone,
		dtypes.Uint64:   ThresholdNone,
		dtypes.Int16:    ThresholdNone,
		dtypes.Uint16:   ThresholdNone,
		dtypes.Int8:     24,
		dtypes.Uint8:    24,
		dtypes.Float16:  ThresholdNone,
		dtypes.BFloat16: ThresholdNone,
	},
	LeadingMinB: map[dtypes.DType]int{
		dtypes.Float32:  256,
		dtypes.Float64:  ThresholdAlwaysFallBack,
		dtypes.Int32:    8,
		dtypes.Uint32:   64,
		dtypes.Int64:    8,
		dtypes.Uint64:   8,
		dtypes.Int16:    ThresholdAlwaysFallBack,
		dtypes.Uint16:   ThresholdAlwaysFallBack,
		dtypes.Int8:     ThresholdAlwaysFallBack,
		dtypes.Uint8:    ThresholdAlwaysFallBack,
		dtypes.Float16:  ThresholdNone,
		dtypes.BFloat16: 32,
	},
	AllMinN: map[dtypes.DType]int{
		dtypes.Float32:  ThresholdNone,
		dtypes.Float64:  12,
		dtypes.Int32:    256,
		dtypes.Uint32:   128,
		dtypes.Int64:    ThresholdNone,
		dtypes.Uint64:   ThresholdNone,
		dtypes.Int16:    ThresholdNone,
		dtypes.Uint16:   ThresholdNone,
		dtypes.Int8:     ThresholdNone,
		dtypes.Uint8:    ThresholdNone,
		dtypes.Float16:  ThresholdNone,
		dtypes.BFloat16: ThresholdNone,
	},
}

// avx512ReduceThresholds defines the crossover points for AVX-512 hardware.
// Initialized from AVX2 baseline; can be tuned with AVX-512 benchmark results.
var avx512ReduceThresholds = ReduceThresholdsConfig{
	TrailingMinB: map[dtypes.DType]int{
		dtypes.Float32:  12,
		dtypes.Float64:  ThresholdNone,
		dtypes.Int32:    ThresholdNone,
		dtypes.Uint32:   ThresholdNone,
		dtypes.Int64:    ThresholdNone,
		dtypes.Uint64:   ThresholdNone,
		dtypes.Int16:    24,
		dtypes.Uint16:   24,
		dtypes.Int8:     48,
		dtypes.Uint8:    48,
		dtypes.Float16:  ThresholdNone,
		dtypes.BFloat16: ThresholdNone,
	},
	LeadingMinB: map[dtypes.DType]int{
		dtypes.Float32:  8,
		dtypes.Float64:  8,
		dtypes.Int32:    8,
		dtypes.Uint32:   8,
		dtypes.Int64:    8,
		dtypes.Uint64:   8,
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
		dtypes.Int64:    8,
		dtypes.Uint64:   8,
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

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package ops

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

// neonReduceThresholds defines the crossover points measured on ARM64 NEON hardware.
// Determined via TestFindReduceThresholds in reduce_bench_test.go.
var neonReduceThresholds = ReduceThresholdsConfig{
	TrailingMinB: map[dtypes.DType]int{
		dtypes.Float32:  16,
		dtypes.Float64:  12,
		dtypes.Int32:    ThresholdNone,
		dtypes.Uint32:   ThresholdNone,
		dtypes.Int64:    ThresholdNone,
		dtypes.Uint64:   ThresholdNone,
		dtypes.Int16:    4,
		dtypes.Uint16:   4,
		dtypes.Int8:     12,
		dtypes.Uint8:    12,
		dtypes.Float16:  ThresholdNone,
		dtypes.BFloat16: 12,
	},
	LeadingMinB: map[dtypes.DType]int{
		dtypes.Float32:  ThresholdNone,
		dtypes.Float64:  ThresholdNone,
		dtypes.Int32:    ThresholdNone,
		dtypes.Uint32:   ThresholdNone,
		dtypes.Int64:    ThresholdNone,
		dtypes.Uint64:   ThresholdNone,
		dtypes.Int16:    ThresholdNone,
		dtypes.Uint16:   ThresholdNone,
		dtypes.Int8:     ThresholdNone,
		dtypes.Uint8:    ThresholdNone,
		dtypes.Float16:  ThresholdNone,
		dtypes.BFloat16: 4,
	},
	AllMinN: map[dtypes.DType]int{
		dtypes.Float32:  ThresholdNone,
		dtypes.Float64:  ThresholdNone,
		dtypes.Int32:    ThresholdNone,
		dtypes.Uint32:   ThresholdNone,
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

func init() {
	if gobackend.IsNEONAllowed {
		reduceThresholds = neonReduceThresholds
	}
}

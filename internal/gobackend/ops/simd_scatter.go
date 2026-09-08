// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package ops

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

func init() {
	// Float32
	combineSliceSumDTypeMap.Register(dtypes.Float32, PrioritySIMD, func(out, upd []float32) {
		simdAddVVFloat32(out, upd, out)
	})
	combineSliceMaxDTypeMap.Register(dtypes.Float32, PrioritySIMD, func(out, upd []float32) {
		simdMaxVVFloat32(out, upd, out)
	})
	combineSliceMinDTypeMap.Register(dtypes.Float32, PrioritySIMD, func(out, upd []float32) {
		simdMinVVFloat32(out, upd, out)
	})

	// Float64
	combineSliceSumDTypeMap.Register(dtypes.Float64, PrioritySIMD, func(out, upd []float64) {
		simdAddVVFloat64(out, upd, out)
	})
	combineSliceMaxDTypeMap.Register(dtypes.Float64, PrioritySIMD, func(out, upd []float64) {
		simdMaxVVFloat64(out, upd, out)
	})
	combineSliceMinDTypeMap.Register(dtypes.Float64, PrioritySIMD, func(out, upd []float64) {
		simdMinVVFloat64(out, upd, out)
	})

	// Int32
	combineSliceSumDTypeMap.Register(dtypes.Int32, PrioritySIMD, func(out, upd []int32) {
		simdAddVVInt32(out, upd, out)
	})
	combineSliceMaxDTypeMap.Register(dtypes.Int32, PrioritySIMD, func(out, upd []int32) {
		simdMaxVVInt32(out, upd, out)
	})
	combineSliceMinDTypeMap.Register(dtypes.Int32, PrioritySIMD, func(out, upd []int32) {
		simdMinVVInt32(out, upd, out)
	})

	// Uint32
	combineSliceSumDTypeMap.Register(dtypes.Uint32, PrioritySIMD, func(out, upd []uint32) {
		simdAddVVUint32(out, upd, out)
	})
	combineSliceMaxDTypeMap.Register(dtypes.Uint32, PrioritySIMD, func(out, upd []uint32) {
		simdMaxVVUint32(out, upd, out)
	})
	combineSliceMinDTypeMap.Register(dtypes.Uint32, PrioritySIMD, func(out, upd []uint32) {
		simdMinVVUint32(out, upd, out)
	})

	// Int64
	combineSliceSumDTypeMap.Register(dtypes.Int64, PrioritySIMD, func(out, upd []int64) {
		simdAddVVInt64(out, upd, out)
	})

	// Uint64
	combineSliceSumDTypeMap.Register(dtypes.Uint64, PrioritySIMD, func(out, upd []uint64) {
		simdAddVVUint64(out, upd, out)
	})

	// BFloat16
	combineSliceSumDTypeMap.Register(dtypes.BFloat16, PrioritySIMD, func(out, upd []bfloat16.BFloat16) {
		simdAddVVBFloat16(out, upd, out)
	})
	combineSliceMaxDTypeMap.Register(dtypes.BFloat16, PrioritySIMD, func(out, upd []bfloat16.BFloat16) {
		simdMaxVVBFloat16(out, upd, out)
	})
	combineSliceMinDTypeMap.Register(dtypes.BFloat16, PrioritySIMD, func(out, upd []bfloat16.BFloat16) {
		simdMinVVBFloat16(out, upd, out)
	})

	// Float16
	combineSliceSumDTypeMap.Register(dtypes.Float16, PrioritySIMD, func(out, upd []float16.Float16) {
		simdAddVVFloat16(out, upd, out)
	})
	combineSliceMaxDTypeMap.Register(dtypes.Float16, PrioritySIMD, func(out, upd []float16.Float16) {
		simdMaxVVFloat16(out, upd, out)
	})
	combineSliceMinDTypeMap.Register(dtypes.Float16, PrioritySIMD, func(out, upd []float16.Float16) {
		simdMinVVFloat16(out, upd, out)
	})
}

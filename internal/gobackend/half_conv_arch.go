// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

// HalfSliceConverters holds optional architecture-optimized slice conversion kernels
// between Float16/BFloat16 and Float32.
type HalfSliceConverters struct {
	Float16ToFloat32  func(dst []float32, src []float16.Float16)
	BFloat16ToFloat32 func(dst []float32, src []bfloat16.BFloat16)
	Float32ToFloat16  func(dst []float16.Float16, src []float32)
	Float32ToBFloat16 func(dst []bfloat16.BFloat16, src []float32)
}

type halfSliceConvertersEntry struct {
	priority RegisterPriority
	conv     HalfSliceConverters
}

var (
	halfSliceConvertersEntries []halfSliceConvertersEntry
	halfSliceConverters        HalfSliceConverters
)

// SetHalfSliceConverters registers architecture-optimized Float16/BFloat16 <-> Float32 slice converters
// at the given priority. Higher priorities take precedence regardless of registration order.
func SetHalfSliceConverters(priority RegisterPriority, conv HalfSliceConverters) {
	if priority < 0 {
		return
	}
	idx := 0
	for idx < len(halfSliceConvertersEntries) && halfSliceConvertersEntries[idx].priority > priority {
		idx++
	}
	if idx < len(halfSliceConvertersEntries) && halfSliceConvertersEntries[idx].priority == priority {
		halfSliceConvertersEntries[idx].conv = conv
	} else {
		halfSliceConvertersEntries = slices.Insert(halfSliceConvertersEntries, idx, halfSliceConvertersEntry{
			priority: priority,
			conv:     conv,
		})
	}
	rebuildHalfSliceConverters()
}

func rebuildHalfSliceConverters() {
	var merged HalfSliceConverters
	for _, entry := range halfSliceConvertersEntries {
		if merged.Float16ToFloat32 == nil {
			merged.Float16ToFloat32 = entry.conv.Float16ToFloat32
		}
		if merged.BFloat16ToFloat32 == nil {
			merged.BFloat16ToFloat32 = entry.conv.BFloat16ToFloat32
		}
		if merged.Float32ToFloat16 == nil {
			merged.Float32ToFloat16 = entry.conv.Float32ToFloat16
		}
		if merged.Float32ToBFloat16 == nil {
			merged.Float32ToBFloat16 = entry.conv.Float32ToBFloat16
		}
	}
	halfSliceConverters = merged
}

// GetHalfSliceConverters returns the highest-priority registered slice converters
// (any field may be nil if no architecture implementation was registered).
func GetHalfSliceConverters() HalfSliceConverters {
	return halfSliceConverters
}

// ScalarFloat16ToFloat32Slice converts Float16 elements to float32 using scalar rounding.
func ScalarFloat16ToFloat32Slice(dst []float32, src []float16.Float16) {
	n := min(len(dst), len(src))
	for i := range n {
		dst[i] = src[i].Float32()
	}
}

// ScalarBFloat16ToFloat32Slice converts BFloat16 elements to float32 using scalar bit shifts.
func ScalarBFloat16ToFloat32Slice(dst []float32, src []bfloat16.BFloat16) {
	n := min(len(dst), len(src))
	for i := range n {
		dst[i] = src[i].Float32()
	}
}

// ScalarFloat32ToFloat16Slice converts float32 elements to Float16 using scalar RNE rounding.
func ScalarFloat32ToFloat16Slice(dst []float16.Float16, src []float32) {
	n := min(len(dst), len(src))
	for i := range n {
		dst[i].SetFloat32(src[i])
	}
}

// ScalarFloat32ToBFloat16Slice converts float32 elements to BFloat16 using scalar RNE rounding.
func ScalarFloat32ToBFloat16Slice(dst []bfloat16.BFloat16, src []float32) {
	n := min(len(dst), len(src))
	for i := range n {
		dst[i].SetFloat32(src[i])
	}
}

// ActiveHalfSliceConverters returns the highest-priority registered slice converters,
// filling any nil field with its scalar implementation so every returned function is non-nil.
func ActiveHalfSliceConverters() HalfSliceConverters {
	conv := halfSliceConverters
	if conv.Float16ToFloat32 == nil {
		conv.Float16ToFloat32 = ScalarFloat16ToFloat32Slice
	}
	if conv.BFloat16ToFloat32 == nil {
		conv.BFloat16ToFloat32 = ScalarBFloat16ToFloat32Slice
	}
	if conv.Float32ToFloat16 == nil {
		conv.Float32ToFloat16 = ScalarFloat32ToFloat16Slice
	}
	if conv.Float32ToBFloat16 == nil {
		conv.Float32ToBFloat16 = ScalarFloat32ToBFloat16Slice
	}
	return conv
}

// ParallelHalfConvThreshold is the minimum element count before half-precision slice conversions
// dispatch across worker goroutines.
const ParallelHalfConvThreshold = 65536

// ParallelConvertSlice converts src into dst using fn, parallelizing in 64-element-aligned chunks
// across backend.Scheduler when the slice length meets ParallelHalfConvThreshold.
func ParallelConvertSlice[Dst, Src any](backend *Backend, dst []Dst, src []Src, fn func(dst []Dst, src []Src)) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	if backend == nil || backend.Scheduler == nil || !backend.Scheduler.IsEnabled() || n < ParallelHalfConvThreshold {
		fn(dst[:n], src[:n])
		return
	}
	backend.Scheduler.ParallelizeChunks(n, 32768, func(start, end int) {
		fn(dst[start:end], src[start:end])
	})
}

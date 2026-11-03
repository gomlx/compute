// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import "slices"

// SDPAFloat32ArchFn is a function hook to execute Scaled Dot-Product Attention for float32
// using architecture-specific kernels (e.g. AVX2, AVX-512, NEON).
// It processes one KV-head group (with groupSize query heads) over query range [qStart, qEnd)
// with zero transpositions.
type SDPAFloat32ArchFn func(
	q, k, v, output []float32,
	qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskGroupStride int,
	additiveBias []float32,
	biasGroupStride int,
	scoresScratch []float32,
	groupSize, qStart, qEnd, kvLen, headDim int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) bool

type sdpaFloat32ArchEntry struct {
	priority RegisterPriority
	fn       SDPAFloat32ArchFn
}

var (
	sdpaFloat32ArchEntries []sdpaFloat32ArchEntry
	sdpaFloat32ArchFn      SDPAFloat32ArchFn
)

// SetSDPAArchDispatcher registers an architecture-specific SDPA dispatcher with a priority.
// Dispatchers are tried in descending order of priority; if a higher-priority dispatcher returns false, the next one is tried.
func SetSDPAArchDispatcher(priority RegisterPriority, fn SDPAFloat32ArchFn) {
	if priority < 0 || fn == nil {
		return
	}
	idx := 0
	for idx < len(sdpaFloat32ArchEntries) && sdpaFloat32ArchEntries[idx].priority > priority {
		idx++
	}
	if idx < len(sdpaFloat32ArchEntries) && sdpaFloat32ArchEntries[idx].priority == priority {
		sdpaFloat32ArchEntries[idx].fn = fn
	} else {
		sdpaFloat32ArchEntries = slices.Insert(sdpaFloat32ArchEntries, idx, sdpaFloat32ArchEntry{priority: priority, fn: fn})
	}
	rebuildSDPAArchFn()
}

func rebuildSDPAArchFn() {
	switch len(sdpaFloat32ArchEntries) {
	case 0:
		sdpaFloat32ArchFn = nil
	case 1:
		sdpaFloat32ArchFn = sdpaFloat32ArchEntries[0].fn
	default:
		entries := slices.Clone(sdpaFloat32ArchEntries)
		sdpaFloat32ArchFn = func(
			q, k, v, output []float32,
			qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
			additiveMask []float32,
			booleanMask []bool,
			maskGroupStride int,
			additiveBias []float32,
			biasGroupStride int,
			scoresScratch []float32,
			groupSize, qStart, qEnd, kvLen, headDim int,
			scale float32, causal bool,
			qLimit, kvLimit int,
		) bool {
			for _, e := range entries {
				if e.fn(
					q, k, v, output,
					qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride,
					additiveMask, booleanMask, maskGroupStride,
					additiveBias, biasGroupStride, scoresScratch,
					groupSize, qStart, qEnd, kvLen, headDim,
					scale, causal, qLimit, kvLimit,
				) {
					return true
				}
			}
			return false
		}
	}
}

// GetSDPAArchDispatcher returns the registered architecture-specific SDPA dispatcher, if any.
func GetSDPAArchDispatcher() SDPAFloat32ArchFn {
	return sdpaFloat32ArchFn
}

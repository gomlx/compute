// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

// SDPAFloat32ArchFn is a function hook to execute Scaled Dot-Product Attention for float32
// using architecture-specific kernels (e.g. AVX2, AVX-512).
// It processes one KV-head group (with groupSize query heads) with zero transpositions.
type SDPAFloat32ArchFn func(
	q, k, v, output []float32,
	qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []float32,
	booleanMask []bool,
	maskGroupStride int,
	additiveBias []float32,
	biasGroupStride int,
	scoresScratch []float32,
	groupSize, seqLen, kvLen, headDim int,
	scale float32, causal bool,
	qLimit, kvLimit int,
) bool

var (
	sdpaFloat32ArchFn       SDPAFloat32ArchFn
	sdpaFloat32ArchPriority RegisterPriority
)

// SetSDPAArchDispatcher registers the architecture-specific SDPA dispatcher with a priority.
// Higher priority replaces lower priority dispatcher.
func SetSDPAArchDispatcher(priority RegisterPriority, fn SDPAFloat32ArchFn) {
	if priority >= sdpaFloat32ArchPriority {
		sdpaFloat32ArchPriority = priority
		sdpaFloat32ArchFn = fn
	}
}

// GetSDPAArchDispatcher returns the registered architecture-specific SDPA dispatcher, if any.
func GetSDPAArchDispatcher() SDPAFloat32ArchFn {
	return sdpaFloat32ArchFn
}

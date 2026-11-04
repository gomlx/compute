// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// BinaryTrailingArchFn is a function hook to execute binary trailing broadcast (RHS or LHS) using architecture-specific kernels.
//
// When isLHS is false (BroadcastTrailingRHS):
//
//	lhs is [A, B]
//	rhs is [A, 1] (flat slice length A)
//	output is [A, B]
//	output[a*B + b] = lhs[a*B + b] op rhs[a]
//
// When isLHS is true (BroadcastTrailingLHS):
//
//	lhs is [A, 1] (flat slice length A)
//	rhs is [A, B]
//	output is [A, B]
//	output[a*B + b] = lhs[a] op rhs[a*B + b]
type BinaryTrailingArchFn func(op compute.OpType, isLHS bool, lhs, rhs, output *Buffer, A, B int, dtype dtypes.DType) bool

type binaryTrailingArchEntry struct {
	priority RegisterPriority
	fn       BinaryTrailingArchFn
}

var (
	binaryTrailingArchEntries []binaryTrailingArchEntry
	binaryTrailingArchFn      BinaryTrailingArchFn
)

// SetBinaryTrailingArchDispatcher registers an architecture-specific binary trailing broadcast dispatcher with a priority.
// Dispatchers are tried in descending order of priority; if a higher-priority dispatcher returns false, the next one is tried.
func SetBinaryTrailingArchDispatcher(priority RegisterPriority, fn BinaryTrailingArchFn) {
	if priority < 0 || fn == nil {
		return
	}
	idx := 0
	for idx < len(binaryTrailingArchEntries) && binaryTrailingArchEntries[idx].priority > priority {
		idx++
	}
	if idx < len(binaryTrailingArchEntries) && binaryTrailingArchEntries[idx].priority == priority {
		binaryTrailingArchEntries[idx].fn = fn
	} else {
		binaryTrailingArchEntries = slices.Insert(binaryTrailingArchEntries, idx, binaryTrailingArchEntry{priority: priority, fn: fn})
	}
	rebuildBinaryTrailingArchFn()
}

func rebuildBinaryTrailingArchFn() {
	switch len(binaryTrailingArchEntries) {
	case 0:
		binaryTrailingArchFn = nil
	case 1:
		binaryTrailingArchFn = binaryTrailingArchEntries[0].fn
	default:
		entries := slices.Clone(binaryTrailingArchEntries)
		binaryTrailingArchFn = func(op compute.OpType, isLHS bool, lhs, rhs, output *Buffer, A, B int, dtype dtypes.DType) bool {
			for _, e := range entries {
				if e.fn(op, isLHS, lhs, rhs, output, A, B, dtype) {
					return true
				}
			}
			return false
		}
	}
}

// GetBinaryTrailingArchDispatcher returns the registered architecture-specific binary trailing broadcast dispatcher, if any.
func GetBinaryTrailingArchDispatcher() BinaryTrailingArchFn {
	return binaryTrailingArchFn
}

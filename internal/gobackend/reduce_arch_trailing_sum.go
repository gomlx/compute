// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"

	"github.com/gomlx/compute/dtypes"
)

// ReduceTrailingSumArchFn is a function hook to execute trailing sum reduction using architecture-specific kernels.
type ReduceTrailingSumArchFn func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool

type reduceTrailingSumArchEntry struct {
	priority RegisterPriority
	fn       ReduceTrailingSumArchFn
}

var (
	reduceTrailingSumArchEntries []reduceTrailingSumArchEntry
	reduceTrailingSumArchFn      ReduceTrailingSumArchFn
)

// SetReduceTrailingSumArchDispatcher registers an architecture-specific trailing sum dispatcher with a priority.
// Dispatchers are tried in descending order of priority; if a higher-priority dispatcher returns false, the next one is tried.
func SetReduceTrailingSumArchDispatcher(priority RegisterPriority, fn ReduceTrailingSumArchFn) {
	if priority < 0 || fn == nil {
		return
	}
	idx := 0
	for idx < len(reduceTrailingSumArchEntries) && reduceTrailingSumArchEntries[idx].priority > priority {
		idx++
	}
	if idx < len(reduceTrailingSumArchEntries) && reduceTrailingSumArchEntries[idx].priority == priority {
		reduceTrailingSumArchEntries[idx].fn = fn
	} else {
		reduceTrailingSumArchEntries = slices.Insert(reduceTrailingSumArchEntries, idx, reduceTrailingSumArchEntry{priority: priority, fn: fn})
	}
	rebuildReduceTrailingSumArchFn()
}

func rebuildReduceTrailingSumArchFn() {
	switch len(reduceTrailingSumArchEntries) {
	case 0:
		reduceTrailingSumArchFn = nil
	case 1:
		reduceTrailingSumArchFn = reduceTrailingSumArchEntries[0].fn
	default:
		entries := slices.Clone(reduceTrailingSumArchEntries)
		reduceTrailingSumArchFn = func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool {
			for _, e := range entries {
				if e.fn(operand, output, A, B, dtype) {
					return true
				}
			}
			return false
		}
	}
}

// GetReduceTrailingSumArchDispatcher returns the registered architecture-specific trailing sum dispatcher, if any.
func GetReduceTrailingSumArchDispatcher() ReduceTrailingSumArchFn {
	return reduceTrailingSumArchFn
}

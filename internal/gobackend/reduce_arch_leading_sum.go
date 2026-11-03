// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"slices"

	"github.com/gomlx/compute/dtypes"
)

// ReduceLeadingSumArchFn is a function hook to execute leading sum reduction using architecture-specific kernels.
type ReduceLeadingSumArchFn func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool

type reduceLeadingSumArchEntry struct {
	priority RegisterPriority
	fn       ReduceLeadingSumArchFn
}

var (
	reduceLeadingSumArchEntries []reduceLeadingSumArchEntry
	reduceLeadingSumArchFn      ReduceLeadingSumArchFn
)

// SetReduceLeadingSumArchDispatcher registers an architecture-specific leading sum dispatcher with a priority.
// Dispatchers are tried in descending order of priority; if a higher-priority dispatcher returns false, the next one is tried.
func SetReduceLeadingSumArchDispatcher(priority RegisterPriority, fn ReduceLeadingSumArchFn) {
	if priority < 0 || fn == nil {
		return
	}
	idx := 0
	for idx < len(reduceLeadingSumArchEntries) && reduceLeadingSumArchEntries[idx].priority > priority {
		idx++
	}
	if idx < len(reduceLeadingSumArchEntries) && reduceLeadingSumArchEntries[idx].priority == priority {
		reduceLeadingSumArchEntries[idx].fn = fn
	} else {
		reduceLeadingSumArchEntries = slices.Insert(reduceLeadingSumArchEntries, idx, reduceLeadingSumArchEntry{priority: priority, fn: fn})
	}
	rebuildReduceLeadingSumArchFn()
}

func rebuildReduceLeadingSumArchFn() {
	switch len(reduceLeadingSumArchEntries) {
	case 0:
		reduceLeadingSumArchFn = nil
	case 1:
		reduceLeadingSumArchFn = reduceLeadingSumArchEntries[0].fn
	default:
		entries := slices.Clone(reduceLeadingSumArchEntries)
		reduceLeadingSumArchFn = func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool {
			for _, e := range entries {
				if e.fn(operand, output, A, B, dtype) {
					return true
				}
			}
			return false
		}
	}
}

// GetReduceLeadingSumArchDispatcher returns the registered architecture-specific leading sum dispatcher, if any.
func GetReduceLeadingSumArchDispatcher() ReduceLeadingSumArchFn {
	return reduceLeadingSumArchFn
}

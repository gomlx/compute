// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"github.com/gomlx/compute/dtypes"
)

// ReduceLeadingSumArchFn is a function hook to execute leading sum reduction using architecture-specific kernels.
type ReduceLeadingSumArchFn func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool

var (
	reduceLeadingSumArchFn       ReduceLeadingSumArchFn
	reduceLeadingSumArchPriority RegisterPriority
)

// SetReduceLeadingSumArchDispatcher registers the architecture-specific leading sum dispatcher with a priority.
// Higher priority replaces lower priority dispatcher.
func SetReduceLeadingSumArchDispatcher(priority RegisterPriority, fn ReduceLeadingSumArchFn) {
	if priority >= reduceLeadingSumArchPriority {
		reduceLeadingSumArchPriority = priority
		reduceLeadingSumArchFn = fn
	}
}

// GetReduceLeadingSumArchDispatcher returns the registered architecture-specific leading sum dispatcher, if any.
func GetReduceLeadingSumArchDispatcher() ReduceLeadingSumArchFn {
	return reduceLeadingSumArchFn
}

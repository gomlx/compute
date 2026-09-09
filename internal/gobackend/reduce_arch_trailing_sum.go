// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"github.com/gomlx/compute/dtypes"
)

// ReduceTrailingSumArchFn is a function hook to execute trailing sum reduction using architecture-specific kernels.
type ReduceTrailingSumArchFn func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool

var (
	reduceTrailingSumArchFn       ReduceTrailingSumArchFn
	reduceTrailingSumArchPriority RegisterPriority
)

// SetReduceTrailingSumArchDispatcher registers the architecture-specific trailing sum dispatcher with a priority.
// Higher priority replaces lower priority dispatcher.
func SetReduceTrailingSumArchDispatcher(priority RegisterPriority, fn ReduceTrailingSumArchFn) {
	if priority >= reduceTrailingSumArchPriority {
		reduceTrailingSumArchPriority = priority
		reduceTrailingSumArchFn = fn
	}
}

// GetReduceTrailingSumArchDispatcher returns the registered architecture-specific trailing sum dispatcher, if any.
func GetReduceTrailingSumArchDispatcher() ReduceTrailingSumArchFn {
	return reduceTrailingSumArchFn
}

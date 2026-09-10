// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
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

var (
	binaryTrailingArchFn       BinaryTrailingArchFn
	binaryTrailingArchPriority RegisterPriority
)

// SetBinaryTrailingArchDispatcher registers the architecture-specific binary trailing broadcast dispatcher with a priority.
// Higher priority replaces lower priority dispatcher.
func SetBinaryTrailingArchDispatcher(priority RegisterPriority, fn BinaryTrailingArchFn) {
	if priority >= binaryTrailingArchPriority {
		binaryTrailingArchPriority = priority
		binaryTrailingArchFn = fn
	}
}

// GetBinaryTrailingArchDispatcher returns the registered architecture-specific binary trailing broadcast dispatcher, if any.
func GetBinaryTrailingArchDispatcher() BinaryTrailingArchFn {
	return binaryTrailingArchFn
}

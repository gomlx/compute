// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !amd64

package ops

func whereFloat32Arch(cond []bool, onTrue, onFalse, out []float32, onTrueIsScalar, onFalseIsScalar bool) bool {
	return false
}

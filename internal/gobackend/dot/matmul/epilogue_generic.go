// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !amd64

package matmul

func addBiasFloat32Arch(row, bias []float32) bool {
	return false
}

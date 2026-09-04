// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

package matmul

// avx512LargeKernelFloat32Asm is the assembly implementation of the 4 rows x 64 cols GEMM microkernel.
// Defined in avx512_large_amd64.s.
//
//go:noescape
func avx512LargeKernelFloat32Asm(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
)

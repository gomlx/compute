// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

package matmul

import (
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

// avx512LargeKernelFloat32Asm is the assembly implementation of the 4 rows x 64 cols GEMM microkernel for Float32.
// Defined in avx512_large_amd64_float32.s.
//
//go:noescape
func avx512LargeKernelFloat32Asm(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
)

// avx512LargeKernelFloat16Asm is the assembly implementation of the 4 rows x 64 cols GEMM microkernel for Float16.
// Defined in avx512_large_amd64_float16.s.
//
//go:noescape
func avx512LargeKernelFloat16Asm(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
)

// avx512LargeKernelBFloat16Asm is the assembly implementation of the 4 rows x 64 cols GEMM microkernel for BFloat16.
// Defined in avx512_large_amd64_bfloat16.s.
//
//go:noescape
func avx512LargeKernelBFloat16Asm(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
)

// avx512LargeKernelFloat64Asm is the assembly implementation of the 4 rows x 32 cols GEMM microkernel for Float64.
// Defined in avx512_large_amd64_float64.s.
//
//go:noescape
func avx512LargeKernelFloat64Asm(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
)



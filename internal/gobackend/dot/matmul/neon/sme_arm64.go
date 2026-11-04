// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

// smeLargeKernelFloat32Asm_16x16 implements a 16x16 GEMM microkernel using 1 ARM64 SME ZA tile (ZA0.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat32Asm_16x16(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat32Asm_16x64 implements a 16x64 GEMM microkernel using 4 ARM64 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat32Asm_16x64(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat32Asm_32x32 implements a 32x32 GEMM microkernel using 4 ARM64 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat32Asm_32x32(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat16Asm_16x16 implements a 16x16 GEMM microkernel for Float16 -> Float32 using 1 SME ZA tile (ZA0.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat16Asm_16x16(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat16Asm_32x32 implements a 32x32 GEMM microkernel for Float16 -> Float32 using 4 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat16Asm_32x32(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelBFloat16Asm_16x16 implements a 16x16 GEMM microkernel for BFloat16 -> Float32 using 1 SME ZA tile (ZA0.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelBFloat16Asm_16x16(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelBFloat16Asm_32x32 implements a 32x32 GEMM microkernel for BFloat16 -> Float32 using 4 SME ZA tiles (ZA0.S-ZA3.S).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelBFloat16Asm_32x32(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat64Asm_8x8 implements an 8x8 GEMM microkernel for Float64 using 1 SME ZA tile (ZA0.D).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat64Asm_8x8(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// smeLargeKernelFloat64Asm_16x16 implements a 16x16 GEMM microkernel for Float64 using 4 SME ZA tiles (ZA0.D-ZA3.D).
// Defined in sme_arm64.s.
//
//go:noescape
func smeLargeKernelFloat64Asm_16x16(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

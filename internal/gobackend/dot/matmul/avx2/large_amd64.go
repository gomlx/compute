// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

package avx2

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/support/envutil"
)

var (
	// AVX2UseAsm enables the assembly microkernel for large matrices (4 rows x 16 cols for F32/F16/BF16, 4 rows x 8 cols for F64).
	// Set GOMLX_GO_AVX2_ASM=false to disable and use the Go SIMD kernel.
	AVX2UseAsm = envutil.MustReadBool(envutil.GoBackendAVX2_ASM, true)
)

// avx2LargeKernelFloat32Asm is the assembly implementation of the 4 rows x 16 cols GEMM microkernel for Float32.
// Defined in avx2_large_amd64_float32.s.
//
//go:noescape
func avx2LargeKernelFloat32Asm(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// avx2LargeKernelFloat64Asm is the assembly implementation of the 4 rows x 8 cols GEMM microkernel for Float64.
// Defined in avx2_large_amd64_float64.s.
//
//go:noescape
func avx2LargeKernelFloat64Asm(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// avx2LargeKernelFloat16Asm is the assembly implementation of the 4 rows x 16 cols GEMM microkernel for Float16.
// Defined in avx2_large_amd64_float16.s.
//
//go:noescape
func avx2LargeKernelFloat16Asm(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// avx2LargeKernelBFloat16Asm is the assembly implementation of the 4 rows x 16 cols GEMM microkernel for BFloat16.
// Defined in avx2_large_amd64_bfloat16.s.
//
//go:noescape
func avx2LargeKernelBFloat16Asm(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// avx2PackLHSKernelRows4Float32Asm packs 4 rows of float32 LHS matrix in strips of 4 into panel.
// Only full 4-row strips are packed; any remaining partial strip is handled by the caller.
// Defined in avx2_pack_amd64_float32.s.
//
//go:noescape
func avx2PackLHSKernelRows4Float32Asm(
	lhs, panel []float32,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols int,
)

// avx2PackLHSKernelRows4Float64Asm packs 4 rows of float64 LHS matrix in strips of 4 into panel.
// Defined in avx2_pack_amd64_float64.s.
//
//go:noescape
func avx2PackLHSKernelRows4Float64Asm(
	lhs, panel []float64,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols int,
)

// avx2PackLHSKernelRows4Float16Asm packs 4 rows of float16 LHS matrix in strips of 4 into panel.
// Defined in avx2_pack_amd64_float16.s.
//
//go:noescape
func avx2PackLHSKernelRows4Float16Asm(
	lhs, panel []float16.Float16,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols int,
)

// avx2PackLHSKernelRows4BFloat16Asm packs 4 rows of bfloat16 LHS matrix in strips of 4 into panel.
// Defined in avx2_pack_amd64_bfloat16.s.
//
//go:noescape
func avx2PackLHSKernelRows4BFloat16Asm(
	lhs, panel []bfloat16.BFloat16,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols int,
)

// avx2PackRHSFullStripsAsm packs full strips of RHS matrix into panel using AVX2.
// kernelColsBytes must be 64 or 32.
// Defined in avx2_pack_rhs_amd64.s.
//
//go:noescape
func avx2PackRHSFullStripsAsm(
	rhsPtr, panelPtr unsafe.Pointer,
	rhsStrideBytes uintptr,
	contractingRows, numStrips, kernelColsBytes int,
)

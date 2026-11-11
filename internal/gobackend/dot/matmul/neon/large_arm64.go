// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"runtime"
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

// neonLargeKernelFloat32Asm is the Plan9 ARM64 assembly implementation of the 8 rows x 8 cols GEMM microkernel for Float32.
// Defined in large_arm64_float32.s.
//
//go:noescape
func neonLargeKernelFloat32Asm(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelFloat32Asm_4x16 is the Plan9 ARM64 assembly implementation of the 4 rows x 16 cols GEMM microkernel for Float32.
// Defined in large_arm64_float32.s.
//
//go:noescape
func neonLargeKernelFloat32Asm_4x16(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelFloat32Asm_8x12 is the Plan9 ARM64 assembly implementation of the 8 rows x 12 cols GEMM microkernel for Float32.
// Defined in large_arm64_float32.s.
//
//go:noescape
func neonLargeKernelFloat32Asm_8x12(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelFloat16Asm is the Plan9 ARM64 assembly implementation of the 8 rows x 8 cols GEMM microkernel for Float16 -> Float32.
// Defined in large_arm64_float16.s.
//
//go:noescape
func neonLargeKernelFloat16Asm(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelFloat16Asm_4x16 is the Plan9 ARM64 assembly implementation of the 4 rows x 16 cols GEMM microkernel for Float16 -> Float32.
// Defined in large_arm64_float16.s.
//
//go:noescape
func neonLargeKernelFloat16Asm_4x16(
	packedLHS, packedRHS []float16.Float16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelBFloat16Asm is the Plan9 ARM64 assembly implementation of the 8 rows x 8 cols GEMM microkernel for BFloat16 -> Float32.
// Defined in large_arm64_bfloat16.s.
//
//go:noescape
func neonLargeKernelBFloat16Asm(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelBFloat16Asm_4x16 is the Plan9 ARM64 assembly implementation of the 4 rows x 16 cols GEMM microkernel for BFloat16 -> Float32.
// Defined in large_arm64_bfloat16.s.
//
//go:noescape
func neonLargeKernelBFloat16Asm_4x16(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelBFloat16BFDOTAsm is the Plan9 ARM64 assembly implementation of the 8 rows x 8 cols GEMM microkernel for BFloat16 -> Float32 using FEAT_BF16 BFDOT.
// Defined in large_arm64_bfloat16.s.
//
//go:noescape
func neonLargeKernelBFloat16BFDOTAsm(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelBFloat16BFDOTAsm_4x16 is the Plan9 ARM64 assembly implementation of the 4 rows x 16 cols GEMM microkernel for BFloat16 -> Float32 using FEAT_BF16 BFDOT.
// Defined in large_arm64_bfloat16.s.
//
//go:noescape
func neonLargeKernelBFloat16BFDOTAsm_4x16(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelFloat64Asm is the Plan9 ARM64 assembly implementation of the 8 rows x 4 cols GEMM microkernel for Float64.
// Defined in large_arm64_float64.s.
//
//go:noescape
func neonLargeKernelFloat64Asm(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonLargeKernelFloat64Asm_4x8 is the Plan9 ARM64 assembly implementation of the 4 rows x 8 cols GEMM microkernel for Float64.
// Defined in large_arm64_float64.s.
//
//go:noescape
func neonLargeKernelFloat64Asm_4x8(
	packedLHS, packedRHS, packedOutput []float64,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

// neonPackRHSFullStripsAsm packs full strips of RHS (width 16, 32, or 64 bytes) unrolled by 4 rows.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackRHSFullStripsAsm(
	rhsPtr, panelPtr unsafe.Pointer,
	rhsStrideBytes uintptr,
	contractingRows, numFullStrips, kernelColsBytes int,
)

// neonPackLHSKernelRows4Float32Asm packs full 4-row strips of 32-bit LHS matrix into panel.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows4Float32Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes uintptr,
	fullRows, contractingCols int,
)

// neonPackLHSKernelRows8Float32Asm packs full 8-row strips of 32-bit LHS matrix into panel.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows8Float32Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes uintptr,
	fullRows, contractingCols int,
)

// neonPackLHSKernelRows4Uint16Asm packs full 4-row strips of 16-bit (Float16/BFloat16) LHS matrix into panel.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows4Uint16Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes uintptr,
	fullRows, contractingCols int,
)

// neonPackLHSKernelRows8Uint16Asm packs full 8-row strips of 16-bit (Float16/BFloat16) LHS matrix into panel.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows8Uint16Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes uintptr,
	fullRows, contractingCols int,
)

// neonPackLHSKernelRows4Float64Asm packs full 4-row strips of 64-bit LHS matrix into panel.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows4Float64Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes uintptr,
	fullRows, contractingCols int,
)

// neonPackLHSKernelRows8Float64Asm packs full 8-row strips of 64-bit LHS matrix into panel.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows8Float64Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes uintptr,
	fullRows, contractingCols int,
)

// neonPackLHSKernelRows8StridedFloat32Asm packs a single 8-row sub-strip of 32-bit LHS matrix into panel with column stride panelColStrideBytes.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows8StridedFloat32Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes, panelColStrideBytes uintptr,
	contractingCols int,
)

// neonPackLHSKernelRows8StridedUint16Asm packs a single 8-row sub-strip of 16-bit LHS matrix into panel with column stride panelColStrideBytes.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows8StridedUint16Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes, panelColStrideBytes uintptr,
	contractingCols int,
)

// neonPackLHSKernelRows8StridedFloat64Asm packs a single 8-row sub-strip of 64-bit LHS matrix into panel with column stride panelColStrideBytes.
// Defined in pack_arm64.s.
//
//go:noescape
func neonPackLHSKernelRows8StridedFloat64Asm(
	lhsStartPtr, panelPtr unsafe.Pointer,
	lhsStrideBytes, panelColStrideBytes uintptr,
	contractingCols int,
)

// neonLargeKernelFloat32_8x12 implements an 8x12 GEMM microkernel using Go archsimd (24 accumulators).
func neonLargeKernelFloat32_8x12(
	packedLHS, packedRHS, packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
) {
	defer func() {
		runtime.KeepAlive(packedLHS)
		runtime.KeepAlive(packedRHS)
		runtime.KeepAlive(packedOutput)
	}()
	_ = lhsPanelRows

	const (
		kernelRows            = 8
		kernelCols            = 12
		bytesPerInputElement  = 4
		bytesPerOutputElement = 4
	)

	outputBasePtr := unsafe.Pointer(unsafe.SliceData(packedOutput))
	rhsBasePtr := unsafe.Pointer(unsafe.SliceData(packedRHS))
	lhsBasePtr := unsafe.Pointer(unsafe.SliceData(packedLHS))

	for lhsRowIdx := 0; lhsRowIdx < lhsActiveRows; lhsRowIdx += kernelRows {
		for rhsColIdx := 0; rhsColIdx < rhsActiveCols; rhsColIdx += kernelCols {
			outputRowStart := lhsRowIdx
			outputColStart := rhsColIdx
			outputStride := rhsPanelCols

			// 24 accumulators (8 rows x 3 RHS vectors)
			accum_lhs0_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs0_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs0_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			accum_lhs1_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs1_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs1_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			accum_lhs2_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs2_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs2_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			accum_lhs3_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs3_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs3_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			accum_lhs4_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs4_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs4_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			accum_lhs5_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs5_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs5_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			accum_lhs6_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs6_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs6_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			accum_lhs7_rhs0 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs7_rhs1 := archsimd.BroadcastFloat32x4(0.0)
			accum_lhs7_rhs2 := archsimd.BroadcastFloat32x4(0.0)

			idxLHS := lhsRowIdx * contractingLen
			idxRHS := rhsColIdx * contractingLen

			rhsRowPtr := unsafe.Add(rhsBasePtr, uintptr(idxRHS*bytesPerInputElement))
			lhsRowPtr := unsafe.Add(lhsBasePtr, uintptr(idxLHS*bytesPerInputElement))

			rOffset := uintptr(0)
			lOffset := uintptr(0)

			rhsStride := uintptr(kernelCols * bytesPerInputElement)
			lhsStride := uintptr(kernelRows * bytesPerInputElement)

			for range contractingLen {
				rhsPtr0 := unsafe.Add(rhsRowPtr, rOffset)
				rhsPtr1 := unsafe.Add(rhsRowPtr, rOffset+16)
				rhsPtr2 := unsafe.Add(rhsRowPtr, rOffset+32)
				rhsVec0 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr0))
				rhsVec1 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr1))
				rhsVec2 := archsimd.LoadFloat32x4Array((*[4]float32)(rhsPtr2))
				rOffset += rhsStride

				lhsPtr := unsafe.Add(lhsRowPtr, lOffset)

				lhsVal0 := *((*float32)(lhsPtr))
				lhsVec0 := archsimd.BroadcastFloat32x4(lhsVal0)
				accum_lhs0_rhs0 = rhsVec0.MulAdd(lhsVec0, accum_lhs0_rhs0)
				accum_lhs0_rhs1 = rhsVec1.MulAdd(lhsVec0, accum_lhs0_rhs1)
				accum_lhs0_rhs2 = rhsVec2.MulAdd(lhsVec0, accum_lhs0_rhs2)

				lhsVal1 := *((*float32)(unsafe.Add(lhsPtr, 4)))
				lhsVec1 := archsimd.BroadcastFloat32x4(lhsVal1)
				accum_lhs1_rhs0 = rhsVec0.MulAdd(lhsVec1, accum_lhs1_rhs0)
				accum_lhs1_rhs1 = rhsVec1.MulAdd(lhsVec1, accum_lhs1_rhs1)
				accum_lhs1_rhs2 = rhsVec2.MulAdd(lhsVec1, accum_lhs1_rhs2)

				lhsVal2 := *((*float32)(unsafe.Add(lhsPtr, 8)))
				lhsVec2 := archsimd.BroadcastFloat32x4(lhsVal2)
				accum_lhs2_rhs0 = rhsVec0.MulAdd(lhsVec2, accum_lhs2_rhs0)
				accum_lhs2_rhs1 = rhsVec1.MulAdd(lhsVec2, accum_lhs2_rhs1)
				accum_lhs2_rhs2 = rhsVec2.MulAdd(lhsVec2, accum_lhs2_rhs2)

				lhsVal3 := *((*float32)(unsafe.Add(lhsPtr, 12)))
				lhsVec3 := archsimd.BroadcastFloat32x4(lhsVal3)
				accum_lhs3_rhs0 = rhsVec0.MulAdd(lhsVec3, accum_lhs3_rhs0)
				accum_lhs3_rhs1 = rhsVec1.MulAdd(lhsVec3, accum_lhs3_rhs1)
				accum_lhs3_rhs2 = rhsVec2.MulAdd(lhsVec3, accum_lhs3_rhs2)

				lhsVal4 := *((*float32)(unsafe.Add(lhsPtr, 16)))
				lhsVec4 := archsimd.BroadcastFloat32x4(lhsVal4)
				accum_lhs4_rhs0 = rhsVec0.MulAdd(lhsVec4, accum_lhs4_rhs0)
				accum_lhs4_rhs1 = rhsVec1.MulAdd(lhsVec4, accum_lhs4_rhs1)
				accum_lhs4_rhs2 = rhsVec2.MulAdd(lhsVec4, accum_lhs4_rhs2)

				lhsVal5 := *((*float32)(unsafe.Add(lhsPtr, 20)))
				lhsVec5 := archsimd.BroadcastFloat32x4(lhsVal5)
				accum_lhs5_rhs0 = rhsVec0.MulAdd(lhsVec5, accum_lhs5_rhs0)
				accum_lhs5_rhs1 = rhsVec1.MulAdd(lhsVec5, accum_lhs5_rhs1)
				accum_lhs5_rhs2 = rhsVec2.MulAdd(lhsVec5, accum_lhs5_rhs2)

				lhsVal6 := *((*float32)(unsafe.Add(lhsPtr, 24)))
				lhsVec6 := archsimd.BroadcastFloat32x4(lhsVal6)
				accum_lhs6_rhs0 = rhsVec0.MulAdd(lhsVec6, accum_lhs6_rhs0)
				accum_lhs6_rhs1 = rhsVec1.MulAdd(lhsVec6, accum_lhs6_rhs1)
				accum_lhs6_rhs2 = rhsVec2.MulAdd(lhsVec6, accum_lhs6_rhs2)

				lhsVal7 := *((*float32)(unsafe.Add(lhsPtr, 28)))
				lhsVec7 := archsimd.BroadcastFloat32x4(lhsVal7)
				accum_lhs7_rhs0 = rhsVec0.MulAdd(lhsVec7, accum_lhs7_rhs0)
				accum_lhs7_rhs1 = rhsVec1.MulAdd(lhsVec7, accum_lhs7_rhs1)
				accum_lhs7_rhs2 = rhsVec2.MulAdd(lhsVec7, accum_lhs7_rhs2)

				lOffset += lhsStride
			}

			rowByteStride := uintptr(rhsPanelCols * bytesPerOutputElement)
			outputIdx0 := uintptr((outputRowStart*outputStride + outputColStart) * bytesPerOutputElement)
			outPtr0 := unsafe.Add(outputBasePtr, outputIdx0)
			outPtr1 := unsafe.Add(outPtr0, rowByteStride)
			outPtr2 := unsafe.Add(outPtr1, rowByteStride)
			outPtr3 := unsafe.Add(outPtr2, rowByteStride)
			outPtr4 := unsafe.Add(outPtr3, rowByteStride)
			outPtr5 := unsafe.Add(outPtr4, rowByteStride)
			outPtr6 := unsafe.Add(outPtr5, rowByteStride)
			outPtr7 := unsafe.Add(outPtr6, rowByteStride)
			const regStride = uintptr(16)

			if !accumulate {
				accum_lhs0_rhs0.StoreArray((*[4]float32)(outPtr0))
				accum_lhs0_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr0, regStride)))
				accum_lhs0_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr0, 2*regStride)))
				accum_lhs1_rhs0.StoreArray((*[4]float32)(outPtr1))
				accum_lhs1_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr1, regStride)))
				accum_lhs1_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr1, 2*regStride)))
				accum_lhs2_rhs0.StoreArray((*[4]float32)(outPtr2))
				accum_lhs2_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr2, regStride)))
				accum_lhs2_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr2, 2*regStride)))
				accum_lhs3_rhs0.StoreArray((*[4]float32)(outPtr3))
				accum_lhs3_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr3, regStride)))
				accum_lhs3_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr3, 2*regStride)))
				accum_lhs4_rhs0.StoreArray((*[4]float32)(outPtr4))
				accum_lhs4_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr4, regStride)))
				accum_lhs4_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr4, 2*regStride)))
				accum_lhs5_rhs0.StoreArray((*[4]float32)(outPtr5))
				accum_lhs5_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr5, regStride)))
				accum_lhs5_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr5, 2*regStride)))
				accum_lhs6_rhs0.StoreArray((*[4]float32)(outPtr6))
				accum_lhs6_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr6, regStride)))
				accum_lhs6_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr6, 2*regStride)))
				accum_lhs7_rhs0.StoreArray((*[4]float32)(outPtr7))
				accum_lhs7_rhs1.StoreArray((*[4]float32)(unsafe.Add(outPtr7, regStride)))
				accum_lhs7_rhs2.StoreArray((*[4]float32)(unsafe.Add(outPtr7, 2*regStride)))
			} else {
				accum_lhs0_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr0))).StoreArray((*[4]float32)(outPtr0))
				accum_lhs0_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr0, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr0, regStride)))
				accum_lhs0_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr0, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr0, 2*regStride)))
				accum_lhs1_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr1))).StoreArray((*[4]float32)(outPtr1))
				accum_lhs1_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr1, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr1, regStride)))
				accum_lhs1_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr1, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr1, 2*regStride)))
				accum_lhs2_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr2))).StoreArray((*[4]float32)(outPtr2))
				accum_lhs2_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr2, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr2, regStride)))
				accum_lhs2_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr2, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr2, 2*regStride)))
				accum_lhs3_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr3))).StoreArray((*[4]float32)(outPtr3))
				accum_lhs3_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr3, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr3, regStride)))
				accum_lhs3_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr3, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr3, 2*regStride)))
				accum_lhs4_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr4))).StoreArray((*[4]float32)(outPtr4))
				accum_lhs4_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr4, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr4, regStride)))
				accum_lhs4_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr4, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr4, 2*regStride)))
				accum_lhs5_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr5))).StoreArray((*[4]float32)(outPtr5))
				accum_lhs5_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr5, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr5, regStride)))
				accum_lhs5_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr5, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr5, 2*regStride)))
				accum_lhs6_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr6))).StoreArray((*[4]float32)(outPtr6))
				accum_lhs6_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr6, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr6, regStride)))
				accum_lhs6_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr6, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr6, 2*regStride)))
				accum_lhs7_rhs0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(outPtr7))).StoreArray((*[4]float32)(outPtr7))
				accum_lhs7_rhs1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr7, regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr7, regStride)))
				accum_lhs7_rhs2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(outPtr7, 2*regStride)))).StoreArray((*[4]float32)(unsafe.Add(outPtr7, 2*regStride)))
			}
		}
	}
}

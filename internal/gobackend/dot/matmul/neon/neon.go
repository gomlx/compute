// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"simd/archsimd"
	"sync"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/internal/gobackend/simdarch"
	"github.com/gomlx/compute/support/envutil"
)

// Auto-generate alternate specialized versions of NEON operations for half-precision and float64 input data types.
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=router.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small_nontransposed.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small_transposed.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=large.go -tags=bf16,f16,f64

var (
	// NEONUseAsm enables the handwritten Plan9 ARM64 assembly microkernels.
	// Set GOMLX_GO_NEON_ASM=false to disable and use the pure-Go archsimd kernels.
	NEONUseAsm = simdarch.NEONUseAsm

	// NEONUseBFDOT enables the FEAT_BF16 hardware BFDOT instructions for BFloat16 -> Float32
	// matrix multiplication when supported by the host CPU.
	// Disabled by default because on Apple M-series NEON, SHLL+FMLA (119.6 GFlops/s single-core,
	// 710.5 GFlops/s multi-core) is 1.6x faster than BFDOT (71.5 GFlops/s single-core, 436.4 GFlops/s multi-core)
	// due to SHLL executing on the decoupled shift/permute pipes while FMLA runs across all 4 FP32 FMA units.
	NEONUseBFDOT = false

	// NEONParamsFloat32 are the cache and microkernel tiling parameters for Float32.
	// ARM64 NEON has 32 128-bit registers (V0-V31), allowing 16 accumulators (e.g. 4x16 or 8x8)
	// or 24 accumulators (8x12). Default is 4x16 (loads a full 64B cache line of RHS per K-step).
	NEONParamsFloat32 = matmul.CacheParams{
		LHSL1KernelRows:      4,   // Mr: 4 rows
		RHSL1KernelCols:      16,  // Nr: 16 cols (4x Float32x4 vectors = 64B cache line)
		PanelContractingSize: 256, // Kc: L1 contracting depth
		LHSPanelCrossSize:    48,  // Mc: L2 row panel height (multiple of both 4 and 8)
		RHSPanelCrossSize:    512, // Nc: L2/L3 col panel width (multiple of Nr)
	}

	// NEONParamsBFloat16 are the parameters for BFloat16 -> Float32.
	NEONParamsBFloat16 = matmul.CacheParams{
		LHSL1KernelRows:      8,   // Mr: 8 rows
		RHSL1KernelCols:      8,   // Nr: 8 cols (widened from 1x BFloat16x8 into 2x Float32x4)
		PanelContractingSize: 256, // Kc: L1 contracting depth
		LHSPanelCrossSize:    48,  // Mc: L2 row panel height (multiple of both 4 and 8)
		RHSPanelCrossSize:    512, // Nc: L2/L3 col panel width
	}

	// NEONParamsFloat16 are the parameters for Float16 -> Float32.
	NEONParamsFloat16 = NEONParamsBFloat16

	// NEONParamsFloat64 are the parameters for Float64.
	NEONParamsFloat64 = matmul.CacheParams{
		LHSL1KernelRows:      8,   // Mr: 8 rows (or 4 rows for 4x8)
		RHSL1KernelCols:      4,   // Nr: 4 cols (2x Float64x2 vectors)
		PanelContractingSize: 128, // Kc: L1 contracting depth
		LHSPanelCrossSize:    16,  // Mc: L2 row panel height
		RHSPanelCrossSize:    256, // Nc: L2/L3 col panel width
	}

	// SMESmallMatMulSizeThreshold is theB*M*N*K threshold below which SME routes to the
	// non-packing NEON Small assembly path (avoiding SMSTART/SMSTOP and panel packing overhead on tiny matrices).
	SMESmallMatMulSizeThreshold = 128 * 1024

	// SMEParamsFloat32 are the cache and microkernel tiling parameters for ARM64 SME Float32.
	// Uses a 32x32 outer-product (FMOPA) microkernel across all 4 ZA.S tiles (ZA0.S-ZA3.S = 1024 FP32 accumulators).
	SMEParamsFloat32 = matmul.CacheParams{
		LHSL1KernelRows:      32,  // Mr: 32 rows (2x 16-lane streaming vectors)
		RHSL1KernelCols:      32,  // Nr: 32 cols (2x 16-lane streaming vectors)
		PanelContractingSize: 256, // Kc: L1 contracting depth
		LHSPanelCrossSize:    64,  // Mc: L2 row panel height (multiple of 32)
		RHSPanelCrossSize:    512, // Nc: L2/L3 col panel width (multiple of 32)
	}

	// SMEParamsBFloat16 are the parameters for ARM64 SME BFloat16 -> Float32 (32x32 outer product).
	SMEParamsBFloat16 = SMEParamsFloat32

	// SMEParamsFloat16 are the parameters for ARM64 SME Float16 -> Float32 (32x32 outer product).
	SMEParamsFloat16 = SMEParamsFloat32

	// SMEParamsFloat64 are the parameters for ARM64 SME Float64 (16x16 outer product across 4 ZA.D tiles).
	SMEParamsFloat64 = matmul.CacheParams{
		LHSL1KernelRows:      16,  // Mr: 16 rows (2x 8-lane streaming vectors)
		RHSL1KernelCols:      16,  // Nr: 16 cols (2x 8-lane streaming vectors)
		PanelContractingSize: 128, // Kc: L1 contracting depth
		LHSPanelCrossSize:    64,  // Mc: L2 row panel height (multiple of 16)
		RHSPanelCrossSize:    256, // Nc: L2/L3 col panel width (multiple of 16)
	}
)

func init() {
	if !NEONUseAsm {
		// Default Go SIMD kernel uses 4x16 for Float32/F16/BF16 and 4x8 for Float64.
		NEONParamsFloat32.LHSL1KernelRows = 4
		NEONParamsFloat32.RHSL1KernelCols = 16
		NEONParamsFloat16.LHSL1KernelRows = 4
		NEONParamsFloat16.RHSL1KernelCols = 16
		NEONParamsBFloat16.LHSL1KernelRows = 4
		NEONParamsBFloat16.RHSL1KernelCols = 16
		NEONParamsFloat64.LHSL1KernelRows = 4
		NEONParamsFloat64.RHSL1KernelCols = 8
	}
	if kc := envutil.MustReadInt(envutil.GoBackendNEON_KC, 0); kc > 0 {
		NEONParamsFloat32.PanelContractingSize = kc
		NEONParamsFloat16.PanelContractingSize = kc
		NEONParamsBFloat16.PanelContractingSize = kc
		NEONParamsFloat64.PanelContractingSize = kc
	}
	if mc := envutil.MustReadInt(envutil.GoBackendNEON_MC, 0); mc > 0 {
		NEONParamsFloat32.LHSPanelCrossSize = mc
		NEONParamsFloat16.LHSPanelCrossSize = mc
		NEONParamsBFloat16.LHSPanelCrossSize = mc
		NEONParamsFloat64.LHSPanelCrossSize = mc
	}
	if nc := envutil.MustReadInt(envutil.GoBackendNEON_NC, 0); nc > 0 {
		NEONParamsFloat32.RHSPanelCrossSize = nc
		NEONParamsFloat16.RHSPanelCrossSize = nc
		NEONParamsBFloat16.RHSPanelCrossSize = nc
		NEONParamsFloat64.RHSPanelCrossSize = nc
	}
	if !envutil.MustReadBool(matmul.EnabledEnv, true) {
		return
	}

	if simdarch.IsNEONAllowed {
		registerNEON(false)
	}
	if simdarch.IsSMEAllowed && NEONUseAsm {
		registerSME(false)
	}
}

// RegisterNEONForTests registers the NEON DotGeneral implementations for unit tests.
func RegisterNEONForTests() {
	registerNEON(true)
}

// RegisterSMEForTests registers the SME DotGeneral implementations for unit tests.
func RegisterSMEForTests() {
	registerSME(true)
}

func registerNEON(forTests bool) {
	dot.RegisterImplementation("simd:neon", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, neonRouterFloat32, matmul.PriorityNEON, forTests)
	dot.RegisterImplementation("simd:neon", dot.LayoutTransposed, dtypes.Float32, dtypes.Float32, neonRouterFloat32, matmul.PriorityNEON, forTests)

	dot.RegisterImplementation("simd:neon", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, neonRouterBFloat16, matmul.PriorityNEON, forTests)
	dot.RegisterImplementation("simd:neon", dot.LayoutTransposed, dtypes.BFloat16, dtypes.Float32, neonRouterBFloat16, matmul.PriorityNEON, forTests)

	dot.RegisterImplementation("simd:neon", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, neonRouterFloat16, matmul.PriorityNEON, forTests)
	dot.RegisterImplementation("simd:neon", dot.LayoutTransposed, dtypes.Float16, dtypes.Float32, neonRouterFloat16, matmul.PriorityNEON, forTests)

	dot.RegisterImplementation("simd:neon", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, neonRouterFloat64, matmul.PriorityNEON, forTests)
	dot.RegisterImplementation("simd:neon", dot.LayoutTransposed, dtypes.Float64, dtypes.Float64, neonRouterFloat64, matmul.PriorityNEON, forTests)
}

func registerSME(forTests bool) {
	dot.RegisterImplementation("simd:sme", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, smeRouterFloat32, matmul.PrioritySME, forTests)
	dot.RegisterImplementation("simd:sme", dot.LayoutTransposed, dtypes.Float32, dtypes.Float32, smeRouterFloat32, matmul.PrioritySME, forTests)

	dot.RegisterImplementation("simd:sme", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, smeRouterBFloat16, matmul.PrioritySME, forTests)
	dot.RegisterImplementation("simd:sme", dot.LayoutTransposed, dtypes.BFloat16, dtypes.Float32, smeRouterBFloat16, matmul.PrioritySME, forTests)

	dot.RegisterImplementation("simd:sme", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, smeRouterFloat16, matmul.PrioritySME, forTests)
	dot.RegisterImplementation("simd:sme", dot.LayoutTransposed, dtypes.Float16, dtypes.Float32, smeRouterFloat16, matmul.PrioritySME, forTests)

	dot.RegisterImplementation("simd:sme", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, smeRouterFloat64, matmul.PrioritySME, forTests)
	dot.RegisterImplementation("simd:sme", dot.LayoutTransposed, dtypes.Float64, dtypes.Float64, smeRouterFloat64, matmul.PrioritySME, forTests)
}

// neonReduceSumFloat32x4 performs a horizontal reduction of a Float32x4 vector.
func neonReduceSumFloat32x4(x4 archsimd.Float32x4) float32 {
	x4sum := x4.ConcatAddPairs(x4)
	return x4sum.GetElem(0) + x4sum.GetElem(1)
}

// neonReduceSumFloat64x2 performs a horizontal reduction of a Float64x2 vector.
func neonReduceSumFloat64x2(x2 archsimd.Float64x2) float64 {
	return x2.GetElem(0) + x2.GetElem(1)
}

// neonPackRHSNonTransposed packs a slice of the RHS matrix into a panel composed of "strips" of width kernelCols.
func neonPackRHSNonTransposed[T gotype.ScalarNotComplex](
	rhs, panel []T,
	rhsRowStart, rhsColStart, rhsCols,
	contractingRows, copyCols, kernelCols int) {
	if contractingRows == 0 || copyCols == 0 {
		return
	}

	var zero T
	bytesPerElement := uintptr(unsafe.Sizeof(zero))
	rhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(rhs)))
	panelBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(panel)))
	copyColsBytesAll := uintptr(copyCols) * bytesPerElement
	rhsStrideBytes := uintptr(rhsCols) * bytesPerElement
	rhsColStartBytes := uintptr(rhsColStart) * bytesPerElement
	kernelColsBytes := uintptr(kernelCols) * bytesPerElement
	if kernelColsBytes%16 != 0 {
		matmul.UnsafePackRHS(rhs, panel, rhsRowStart, rhsColStart, rhsCols, contractingRows, copyCols, kernelCols)
		return
	}

	panelPtr := panelBasePtr
	stripColIdx := uintptr(0)
	if NEONUseAsm && (kernelColsBytes == 256 || kernelColsBytes == 128 || kernelColsBytes == 64 || kernelColsBytes == 32 || kernelColsBytes == 16) {
		numFullStrips := int(copyColsBytesAll / kernelColsBytes)
		if numFullStrips > 0 {
			rhsStartPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes
			neonPackRHSFullStripsAsm(
				unsafe.Pointer(rhsStartPtr),
				unsafe.Pointer(panelBasePtr),
				rhsStrideBytes,
				contractingRows,
				numFullStrips,
				int(kernelColsBytes),
			)
			stripColIdx = uintptr(numFullStrips) * kernelColsBytes
			panelPtr = panelBasePtr + uintptr(numFullStrips)*uintptr(contractingRows)*kernelColsBytes
		}
	}
	switch kernelColsBytes {
	case 64:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr)))
				v1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr + 16)))
				v2 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr + 32)))
				v3 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr + 48)))
				v0.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr)))
				v1.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr + 16)))
				v2.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr + 32)))
				v3.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr + 48)))
				panelPtr += 64
				rhsPtr += rhsStrideBytes
			}
		}
	case 48:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr)))
				v1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr + 16)))
				v2 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr + 32)))
				v0.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr)))
				v1.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr + 16)))
				v2.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr + 32)))
				panelPtr += 48
				rhsPtr += rhsStrideBytes
			}
		}
	case 32:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr)))
				v1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr + 16)))
				v0.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr)))
				v1.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr + 16)))
				panelPtr += 32
				rhsPtr += rhsStrideBytes
			}
		}
	case 16:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr)))
				v0.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr)))
				panelPtr += 16
				rhsPtr += rhsStrideBytes
			}
		}
	default:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				rowRhsPtr := rhsPtr
				for kernelColIdx := uintptr(0); kernelColIdx < kernelColsBytes; kernelColIdx += 16 {
					v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rowRhsPtr)))
					v0.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr)))
					panelPtr += 16
					rowRhsPtr += 16
				}
				rhsPtr += rhsStrideBytes
			}
		}
	}

	copyColsBytes := copyColsBytesAll - stripColIdx
	if copyColsBytes == 0 {
		return
	}

	rhsStripStartPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
	for range contractingRows {
		rhsPtr := rhsStripStartPtr
		colByteIdx := uintptr(0)
		for ; colByteIdx+16 <= copyColsBytes; colByteIdx += 16 {
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(rhsPtr)))
			v0.StoreArray((*[16]uint8)(unsafe.Pointer(panelPtr)))
			rhsPtr += 16
			panelPtr += 16
		}
		remainingBytes := copyColsBytes - colByteIdx
		if remainingBytes > 0 {
			rhsSlice := unsafe.Slice((*uint8)(unsafe.Pointer(rhsPtr)), remainingBytes)
			panelSlice := unsafe.Slice((*uint8)(unsafe.Pointer(panelPtr)), remainingBytes)
			copy(panelSlice, rhsSlice)
			panelPtr += remainingBytes
		}
		padBytes := kernelColsBytes - copyColsBytes
		if padBytes > 0 {
			clear(unsafe.Slice((*uint8)(unsafe.Pointer(panelPtr)), padBytes))
			panelPtr += padBytes
		}
		rhsStripStartPtr += rhsStrideBytes
	}
}

// neonTranspose4x4x32bits transposes a 4x4 matrix of 32-bit values in registers using InterleaveLo/InterleaveHi (ZIP1/ZIP2).
func neonTranspose4x4x32bits(r0, r1, r2, r3 archsimd.Uint32x4) (c0, c1, c2, c3 archsimd.Uint32x4) {
	t0 := r0.InterleaveLo(r1).ReshapeToUint64s()
	t1 := r0.InterleaveHi(r1).ReshapeToUint64s()
	t2 := r2.InterleaveLo(r3).ReshapeToUint64s()
	t3 := r2.InterleaveHi(r3).ReshapeToUint64s()
	c0 = t0.InterleaveLo(t2).ReshapeToUint32s()
	c1 = t0.InterleaveHi(t2).ReshapeToUint32s()
	c2 = t1.InterleaveLo(t3).ReshapeToUint32s()
	c3 = t1.InterleaveHi(t3).ReshapeToUint32s()
	return
}

// neonTranspose4x8x16bits transposes 4 rows x 8 columns of 16-bit values into 4 vectors of 8 uint16s
// (each output vector holds 2 consecutive columns of 4 rows).
func neonTranspose4x8x16bits(r0, r1, r2, r3 archsimd.Uint16x8) (c01, c23, c45, c67 archsimd.Uint16x8) {
	t0 := r0.InterleaveLo(r1).ReshapeToUint32s()
	t1 := r0.InterleaveHi(r1).ReshapeToUint32s()
	t2 := r2.InterleaveLo(r3).ReshapeToUint32s()
	t3 := r2.InterleaveHi(r3).ReshapeToUint32s()
	c01 = t0.InterleaveLo(t2).ReshapeToUint16s()
	c23 = t0.InterleaveHi(t2).ReshapeToUint16s()
	c45 = t1.InterleaveLo(t3).ReshapeToUint16s()
	c67 = t1.InterleaveHi(t3).ReshapeToUint16s()
	return
}

// neonPackLHSSME packs a block of size [copyRows, contractingCols] from the LHS matrix into a panel
// structured as [ceil(copyRows/kernelRows), contractingCols, kernelRows]. When useSME is true and
// kernelRows == 32 for 32-bit elements, it uses SME ZA0..ZA3 hardware horizontal-to-vertical transpose.
func neonPackLHSSME[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols, kernelRows int,
	useSME bool,
) {
	if useSME && NEONUseAsm && kernelRows == 32 {
		if copyRows == 0 || contractingCols == 0 {
			return
		}
		var zero T
		if unsafe.Sizeof(zero) == 4 {
			lhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(lhs)))
			panelBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(panel)))
			lhsStrideBytes := uintptr(lhsCols) * 4
			lhsStartPtr := unsafe.Pointer(lhsBasePtr + uintptr(lhsRowStart)*lhsStrideBytes + uintptr(lhsColStart)*4)
			smePackLHS32Float32Asm(lhsStartPtr, unsafe.Pointer(panelBasePtr), lhsStrideBytes, copyRows, contractingCols)
			return
		}
	}
	neonPackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
}

// neonPackLHS packs a block of size [copyRows, contractingCols] from the LHS matrix into a panel
// structured as [ceil(copyRows/kernelRows), contractingCols, kernelRows].
func neonPackLHS[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols, kernelRows int) {
	if copyRows == 0 || contractingCols == 0 {
		return
	}
	if kernelRows != 4 && kernelRows != 8 && kernelRows != 16 && kernelRows != 32 {
		matmul.UnsafePackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
		return
	}

	var zero T
	bytesPerElement := uintptr(unsafe.Sizeof(zero))
	if bytesPerElement != 2 && bytesPerElement != 4 && bytesPerElement != 8 {
		matmul.UnsafePackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
		return
	}

	lhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(lhs)))
	panelBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(panel)))
	lhsStrideBytes := uintptr(lhsCols) * bytesPerElement
	lhsColStartBytes := uintptr(lhsColStart) * bytesPerElement

	fullRows := copyRows & ^(kernelRows - 1)
	if (NEONUseAsm || kernelRows > 8) && fullRows > 0 {
		lhsStartPtr := unsafe.Pointer(lhsBasePtr + uintptr(lhsRowStart)*lhsStrideBytes + lhsColStartBytes)
		panelStartPtr := unsafe.Pointer(panelBasePtr)
		switch kernelRows {
		case 4:
			switch bytesPerElement {
			case 4:
				neonPackLHSKernelRows4Float32Asm(lhsStartPtr, panelStartPtr, lhsStrideBytes, fullRows, contractingCols)
			case 2:
				neonPackLHSKernelRows4Uint16Asm(lhsStartPtr, panelStartPtr, lhsStrideBytes, fullRows, contractingCols)
			case 8:
				neonPackLHSKernelRows4Float64Asm(lhsStartPtr, panelStartPtr, lhsStrideBytes, fullRows, contractingCols)
			}
		case 8:
			switch bytesPerElement {
			case 4:
				neonPackLHSKernelRows8Float32Asm(lhsStartPtr, panelStartPtr, lhsStrideBytes, fullRows, contractingCols)
			case 2:
				neonPackLHSKernelRows8Uint16Asm(lhsStartPtr, panelStartPtr, lhsStrideBytes, fullRows, contractingCols)
			case 8:
				neonPackLHSKernelRows8Float64Asm(lhsStartPtr, panelStartPtr, lhsStrideBytes, fullRows, contractingCols)
			}
		default:
			// kernelRows == 16 or 32: pack 8 rows at a time into strided columns of width kernelRows.
			panelColStrideBytes := uintptr(kernelRows) * bytesPerElement
			stripPanelBytes := uintptr(contractingCols) * panelColStrideBytes
			for stripRowIdx := 0; stripRowIdx < fullRows; stripRowIdx += kernelRows {
				stripPanelBase := panelBasePtr + uintptr(stripRowIdx/kernelRows)*stripPanelBytes
				for subRow := 0; subRow < kernelRows; subRow += 8 {
					subLhsPtr := unsafe.Pointer(lhsBasePtr + uintptr(lhsRowStart+stripRowIdx+subRow)*lhsStrideBytes + lhsColStartBytes)
					subPanelPtr := unsafe.Pointer(stripPanelBase + uintptr(subRow)*bytesPerElement)
					switch bytesPerElement {
					case 4:
						neonPackLHSKernelRows8StridedFloat32Asm(subLhsPtr, subPanelPtr, lhsStrideBytes, panelColStrideBytes, contractingCols)
					case 2:
						neonPackLHSKernelRows8StridedUint16Asm(subLhsPtr, subPanelPtr, lhsStrideBytes, panelColStrideBytes, contractingCols)
					case 8:
						neonPackLHSKernelRows8StridedFloat64Asm(subLhsPtr, subPanelPtr, lhsStrideBytes, panelColStrideBytes, contractingCols)
					}
				}
			}
		}
		if fullRows == copyRows {
			return
		}
		panelOffset := fullRows * contractingCols
		matmul.UnsafePackLHS(lhs, panel[panelOffset:], lhsRowStart+fullRows, lhsColStart, lhsCols, copyRows-fullRows, contractingCols, kernelRows)
		return
	}

	// Pure Go archsimd implementation for full strips:
	panelPtr := panelBasePtr
	if kernelRows == 4 {
		for stripRowIdx := 0; stripRowIdx < fullRows; stripRowIdx += 4 {
			lhsRow0Ptr := lhsBasePtr + uintptr(lhsRowStart+stripRowIdx)*lhsStrideBytes + lhsColStartBytes
			lhsRow1Ptr := lhsRow0Ptr + lhsStrideBytes
			lhsRow2Ptr := lhsRow1Ptr + lhsStrideBytes
			lhsRow3Ptr := lhsRow2Ptr + lhsStrideBytes
			col := 0
			switch bytesPerElement {
			case 4:
				for ; col+4 <= contractingCols; col += 4 {
					r0 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow0Ptr)))
					r1 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow1Ptr)))
					r2 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow2Ptr)))
					r3 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow3Ptr)))
					c0, c1, c2, c3 := neonTranspose4x4x32bits(r0, r1, r2, r3)
					c0.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr)))
					c1.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 16)))
					c2.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 32)))
					c3.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 48)))
					lhsRow0Ptr += 16
					lhsRow1Ptr += 16
					lhsRow2Ptr += 16
					lhsRow3Ptr += 16
					panelPtr += 64
				}
				for ; col < contractingCols; col++ {
					*(*uint32)(unsafe.Pointer(panelPtr)) = *(*uint32)(unsafe.Pointer(lhsRow0Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 4)) = *(*uint32)(unsafe.Pointer(lhsRow1Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 8)) = *(*uint32)(unsafe.Pointer(lhsRow2Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 12)) = *(*uint32)(unsafe.Pointer(lhsRow3Ptr))
					lhsRow0Ptr += 4
					lhsRow1Ptr += 4
					lhsRow2Ptr += 4
					lhsRow3Ptr += 4
					panelPtr += 16
				}
			case 2:
				for ; col+8 <= contractingCols; col += 8 {
					r0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow0Ptr)))
					r1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow1Ptr)))
					r2 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow2Ptr)))
					r3 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow3Ptr)))
					c01, c23, c45, c67 := neonTranspose4x8x16bits(r0, r1, r2, r3)
					c01.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr)))
					c23.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 16)))
					c45.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 32)))
					c67.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 48)))
					lhsRow0Ptr += 16
					lhsRow1Ptr += 16
					lhsRow2Ptr += 16
					lhsRow3Ptr += 16
					panelPtr += 64
				}
				for ; col < contractingCols; col++ {
					*(*uint16)(unsafe.Pointer(panelPtr)) = *(*uint16)(unsafe.Pointer(lhsRow0Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 2)) = *(*uint16)(unsafe.Pointer(lhsRow1Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 4)) = *(*uint16)(unsafe.Pointer(lhsRow2Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 6)) = *(*uint16)(unsafe.Pointer(lhsRow3Ptr))
					lhsRow0Ptr += 2
					lhsRow1Ptr += 2
					lhsRow2Ptr += 2
					lhsRow3Ptr += 2
					panelPtr += 8
				}
			case 8:
				for ; col+2 <= contractingCols; col += 2 {
					r0 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow0Ptr)))
					r1 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow1Ptr)))
					r2 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow2Ptr)))
					r3 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow3Ptr)))
					c0Top := r0.InterleaveLo(r1)
					c0Bot := r2.InterleaveLo(r3)
					c1Top := r0.InterleaveHi(r1)
					c1Bot := r2.InterleaveHi(r3)
					c0Top.StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr)))
					c0Bot.StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 16)))
					c1Top.StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 32)))
					c1Bot.StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 48)))
					lhsRow0Ptr += 16
					lhsRow1Ptr += 16
					lhsRow2Ptr += 16
					lhsRow3Ptr += 16
					panelPtr += 64
				}
				for ; col < contractingCols; col++ {
					*(*uint64)(unsafe.Pointer(panelPtr)) = *(*uint64)(unsafe.Pointer(lhsRow0Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 8)) = *(*uint64)(unsafe.Pointer(lhsRow1Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 16)) = *(*uint64)(unsafe.Pointer(lhsRow2Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 24)) = *(*uint64)(unsafe.Pointer(lhsRow3Ptr))
					lhsRow0Ptr += 8
					lhsRow1Ptr += 8
					lhsRow2Ptr += 8
					lhsRow3Ptr += 8
					panelPtr += 32
				}
			}
		}
	} else {
		// kernelRows == 8
		for stripRowIdx := 0; stripRowIdx < fullRows; stripRowIdx += 8 {
			lhsRow0Ptr := lhsBasePtr + uintptr(lhsRowStart+stripRowIdx)*lhsStrideBytes + lhsColStartBytes
			lhsRow1Ptr := lhsRow0Ptr + lhsStrideBytes
			lhsRow2Ptr := lhsRow1Ptr + lhsStrideBytes
			lhsRow3Ptr := lhsRow2Ptr + lhsStrideBytes
			lhsRow4Ptr := lhsRow3Ptr + lhsStrideBytes
			lhsRow5Ptr := lhsRow4Ptr + lhsStrideBytes
			lhsRow6Ptr := lhsRow5Ptr + lhsStrideBytes
			lhsRow7Ptr := lhsRow6Ptr + lhsStrideBytes
			col := 0
			switch bytesPerElement {
			case 4:
				for ; col+4 <= contractingCols; col += 4 {
					r0 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow0Ptr)))
					r1 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow1Ptr)))
					r2 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow2Ptr)))
					r3 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow3Ptr)))
					r4 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow4Ptr)))
					r5 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow5Ptr)))
					r6 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow6Ptr)))
					r7 := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(lhsRow7Ptr)))
					t0, t1, t2, t3 := neonTranspose4x4x32bits(r0, r1, r2, r3)
					b0, b1, b2, b3 := neonTranspose4x4x32bits(r4, r5, r6, r7)
					t0.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr)))
					b0.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 16)))
					t1.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 32)))
					b1.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 48)))
					t2.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 64)))
					b2.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 80)))
					t3.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 96)))
					b3.StoreArray((*[4]uint32)(unsafe.Pointer(panelPtr + 112)))
					lhsRow0Ptr += 16
					lhsRow1Ptr += 16
					lhsRow2Ptr += 16
					lhsRow3Ptr += 16
					lhsRow4Ptr += 16
					lhsRow5Ptr += 16
					lhsRow6Ptr += 16
					lhsRow7Ptr += 16
					panelPtr += 128
				}
				for ; col < contractingCols; col++ {
					*(*uint32)(unsafe.Pointer(panelPtr)) = *(*uint32)(unsafe.Pointer(lhsRow0Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 4)) = *(*uint32)(unsafe.Pointer(lhsRow1Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 8)) = *(*uint32)(unsafe.Pointer(lhsRow2Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 12)) = *(*uint32)(unsafe.Pointer(lhsRow3Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 16)) = *(*uint32)(unsafe.Pointer(lhsRow4Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 20)) = *(*uint32)(unsafe.Pointer(lhsRow5Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 24)) = *(*uint32)(unsafe.Pointer(lhsRow6Ptr))
					*(*uint32)(unsafe.Pointer(panelPtr + 28)) = *(*uint32)(unsafe.Pointer(lhsRow7Ptr))
					lhsRow0Ptr += 4
					lhsRow1Ptr += 4
					lhsRow2Ptr += 4
					lhsRow3Ptr += 4
					lhsRow4Ptr += 4
					lhsRow5Ptr += 4
					lhsRow6Ptr += 4
					lhsRow7Ptr += 4
					panelPtr += 32
				}
			case 2:
				for ; col+8 <= contractingCols; col += 8 {
					r0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow0Ptr)))
					r1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow1Ptr)))
					r2 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow2Ptr)))
					r3 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow3Ptr)))
					r4 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow4Ptr)))
					r5 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow5Ptr)))
					r6 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow6Ptr)))
					r7 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(lhsRow7Ptr)))
					t01, t23, t45, t67 := neonTranspose4x8x16bits(r0, r1, r2, r3)
					b01, b23, b45, b67 := neonTranspose4x8x16bits(r4, r5, r6, r7)
					c0 := t01.ReshapeToUint64s().InterleaveLo(b01.ReshapeToUint64s()).ReshapeToUint16s()
					c1 := t01.ReshapeToUint64s().InterleaveHi(b01.ReshapeToUint64s()).ReshapeToUint16s()
					c2 := t23.ReshapeToUint64s().InterleaveLo(b23.ReshapeToUint64s()).ReshapeToUint16s()
					c3 := t23.ReshapeToUint64s().InterleaveHi(b23.ReshapeToUint64s()).ReshapeToUint16s()
					c4 := t45.ReshapeToUint64s().InterleaveLo(b45.ReshapeToUint64s()).ReshapeToUint16s()
					c5 := t45.ReshapeToUint64s().InterleaveHi(b45.ReshapeToUint64s()).ReshapeToUint16s()
					c6 := t67.ReshapeToUint64s().InterleaveLo(b67.ReshapeToUint64s()).ReshapeToUint16s()
					c7 := t67.ReshapeToUint64s().InterleaveHi(b67.ReshapeToUint64s()).ReshapeToUint16s()
					c0.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr)))
					c1.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 16)))
					c2.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 32)))
					c3.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 48)))
					c4.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 64)))
					c5.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 80)))
					c6.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 96)))
					c7.StoreArray((*[8]uint16)(unsafe.Pointer(panelPtr + 112)))
					lhsRow0Ptr += 16
					lhsRow1Ptr += 16
					lhsRow2Ptr += 16
					lhsRow3Ptr += 16
					lhsRow4Ptr += 16
					lhsRow5Ptr += 16
					lhsRow6Ptr += 16
					lhsRow7Ptr += 16
					panelPtr += 128
				}
				for ; col < contractingCols; col++ {
					*(*uint16)(unsafe.Pointer(panelPtr)) = *(*uint16)(unsafe.Pointer(lhsRow0Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 2)) = *(*uint16)(unsafe.Pointer(lhsRow1Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 4)) = *(*uint16)(unsafe.Pointer(lhsRow2Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 6)) = *(*uint16)(unsafe.Pointer(lhsRow3Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 8)) = *(*uint16)(unsafe.Pointer(lhsRow4Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 10)) = *(*uint16)(unsafe.Pointer(lhsRow5Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 12)) = *(*uint16)(unsafe.Pointer(lhsRow6Ptr))
					*(*uint16)(unsafe.Pointer(panelPtr + 14)) = *(*uint16)(unsafe.Pointer(lhsRow7Ptr))
					lhsRow0Ptr += 2
					lhsRow1Ptr += 2
					lhsRow2Ptr += 2
					lhsRow3Ptr += 2
					lhsRow4Ptr += 2
					lhsRow5Ptr += 2
					lhsRow6Ptr += 2
					lhsRow7Ptr += 2
					panelPtr += 16
				}
			case 8:
				for ; col+2 <= contractingCols; col += 2 {
					r0 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow0Ptr)))
					r1 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow1Ptr)))
					r2 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow2Ptr)))
					r3 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow3Ptr)))
					r4 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow4Ptr)))
					r5 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow5Ptr)))
					r6 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow6Ptr)))
					r7 := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Pointer(lhsRow7Ptr)))
					r0.InterleaveLo(r1).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr)))
					r2.InterleaveLo(r3).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 16)))
					r4.InterleaveLo(r5).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 32)))
					r6.InterleaveLo(r7).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 48)))
					r0.InterleaveHi(r1).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 64)))
					r2.InterleaveHi(r3).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 80)))
					r4.InterleaveHi(r5).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 96)))
					r6.InterleaveHi(r7).StoreArray((*[2]uint64)(unsafe.Pointer(panelPtr + 112)))
					lhsRow0Ptr += 16
					lhsRow1Ptr += 16
					lhsRow2Ptr += 16
					lhsRow3Ptr += 16
					lhsRow4Ptr += 16
					lhsRow5Ptr += 16
					lhsRow6Ptr += 16
					lhsRow7Ptr += 16
					panelPtr += 128
				}
				for ; col < contractingCols; col++ {
					*(*uint64)(unsafe.Pointer(panelPtr)) = *(*uint64)(unsafe.Pointer(lhsRow0Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 8)) = *(*uint64)(unsafe.Pointer(lhsRow1Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 16)) = *(*uint64)(unsafe.Pointer(lhsRow2Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 24)) = *(*uint64)(unsafe.Pointer(lhsRow3Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 32)) = *(*uint64)(unsafe.Pointer(lhsRow4Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 40)) = *(*uint64)(unsafe.Pointer(lhsRow5Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 48)) = *(*uint64)(unsafe.Pointer(lhsRow6Ptr))
					*(*uint64)(unsafe.Pointer(panelPtr + 56)) = *(*uint64)(unsafe.Pointer(lhsRow7Ptr))
					lhsRow0Ptr += 8
					lhsRow1Ptr += 8
					lhsRow2Ptr += 8
					lhsRow3Ptr += 8
					lhsRow4Ptr += 8
					lhsRow5Ptr += 8
					lhsRow6Ptr += 8
					lhsRow7Ptr += 8
					panelPtr += 64
				}
			}
		}
	}

	if fullRows < copyRows {
		panelOffset := fullRows * contractingCols
		matmul.UnsafePackLHS(lhs, panel[panelOffset:], lhsRowStart+fullRows, lhsColStart, lhsCols, copyRows-fullRows, contractingCols, kernelRows)
	}
}

// neonApplyPackedOutputFloat32 applies the computed packedOutput to the final output.
func neonApplyPackedOutputFloat32(
	packedOutput, output []float32,
	isFirstContractingPanel bool,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int,
	outputRowStride int,
	height, width int,
) {
	outputRowIdx := lhsRowOffset*outputRowStride + rhsColOffset
	packedRowIdx := 0
	if isFirstContractingPanel {
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			for ; c+16 <= width; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx])))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx+4])))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx+8])))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx+12])))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx+12])))
				packedColIdx += 16
				outputColIdx += 16
			}
			for ; c+4 <= width; c += 4 {
				packedVal := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx])))
				packedVal.StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx])))
				packedColIdx += 4
				outputColIdx += 4
			}
			for i := range width - c {
				output[outputColIdx+i] = packedOutput[packedColIdx+i]
			}
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	} else {
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			for ; c+16 <= width; c += 16 {
				p0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx])))
				p1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx+4])))
				p2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx+8])))
				p3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx+12])))
				o0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&output[outputColIdx])))
				o1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&output[outputColIdx+4])))
				o2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&output[outputColIdx+8])))
				o3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&output[outputColIdx+12])))
				p0.Add(o0).StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx])))
				p1.Add(o1).StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx+4])))
				p2.Add(o2).StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx+8])))
				p3.Add(o3).StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx+12])))
				packedColIdx += 16
				outputColIdx += 16
			}
			for ; c+4 <= width; c += 4 {
				packedVal := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&packedOutput[packedColIdx])))
				outputVal := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&output[outputColIdx])))
				packedVal.Add(outputVal).StoreArray((*[4]float32)(unsafe.Pointer(&output[outputColIdx])))
				packedColIdx += 4
				outputColIdx += 4
			}
			for i := range width - c {
				output[outputColIdx+i] += packedOutput[packedColIdx+i]
			}
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	}
}

// neonApplyPackedOutputFloat64 applies the computed packedOutput to the final output.
func neonApplyPackedOutputFloat64(
	packedOutput, output []float64,
	isFirstContractingPanel bool,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int,
	outputRowStride int,
	height, width int,
) {
	outputRowIdx := lhsRowOffset*outputRowStride + rhsColOffset
	packedRowIdx := 0
	if isFirstContractingPanel {
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			for ; c+8 <= width; c += 8 {
				v0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx])))
				v1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx+2])))
				v2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx+4])))
				v3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx+6])))
				v0.StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx])))
				v1.StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx+2])))
				v2.StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx+4])))
				v3.StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx+6])))
				packedColIdx += 8
				outputColIdx += 8
			}
			for ; c+2 <= width; c += 2 {
				packedVal := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx])))
				packedVal.StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx])))
				packedColIdx += 2
				outputColIdx += 2
			}
			for i := range width - c {
				output[outputColIdx+i] = packedOutput[packedColIdx+i]
			}
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	} else {
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			for ; c+8 <= width; c += 8 {
				p0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx])))
				p1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx+2])))
				p2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx+4])))
				p3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx+6])))
				o0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&output[outputColIdx])))
				o1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&output[outputColIdx+2])))
				o2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&output[outputColIdx+4])))
				o3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&output[outputColIdx+6])))
				p0.Add(o0).StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx])))
				p1.Add(o1).StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx+2])))
				p2.Add(o2).StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx+4])))
				p3.Add(o3).StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx+6])))
				packedColIdx += 8
				outputColIdx += 8
			}
			for ; c+2 <= width; c += 2 {
				packedVal := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&packedOutput[packedColIdx])))
				outputVal := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&output[outputColIdx])))
				packedVal.Add(outputVal).StoreArray((*[2]float64)(unsafe.Pointer(&output[outputColIdx])))
				packedColIdx += 2
				outputColIdx += 2
			}
			for i := range width - c {
				output[outputColIdx+i] += packedOutput[packedColIdx+i]
			}
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	}
}

// GetBuffer forwards buffer allocation to the shared matmul package.
func GetBuffer[T gotype.Supported](backend *gobackend.Backend, length int) (*gobackend.Buffer, []T, bool) {
	return matmul.GetBuffer[T](backend, length)
}

// GetExactBuffer forwards exact-sized buffer allocation to the shared matmul package.
func GetExactBuffer[T gotype.Supported](backend *gobackend.Backend, length int) (*gobackend.Buffer, []T, bool) {
	return matmul.GetExactBuffer[T](backend, length)
}

// ReleaseBuffer forwards buffer release to the shared matmul package.
func ReleaseBuffer(ref *gobackend.Buffer) {
	matmul.ReleaseBuffer(ref)
}

const neonScratchBytes = 2 * 1024 * 1024 // 2 MB aligned scratch buffer for single-worker packedLHS + packedRHS

type neonScratchBuf struct {
	data [neonScratchBytes / 8]uint64
}

var neonScratchPool = sync.Pool{
	New: func() any {
		return new(neonScratchBuf)
	},
}

// GetScratchPair acquires a combined 8-byte-aligned scratch buffer from neonScratchPool
// with zero heap allocations and zero sync.Map lookups when total bytes <= neonScratchBytes.
func GetScratchPair[T gotype.Supported](lhsLen, rhsLen int) (scratch *neonScratchBuf, lhsSlice, rhsSlice []T, ok bool) {
	var zero T
	elemBytes := int(unsafe.Sizeof(zero))
	lhsBytes := ((lhsLen*elemBytes + 63) &^ 63)
	rhsBytes := ((rhsLen*elemBytes + 63) &^ 63)
	if lhsBytes+rhsBytes > neonScratchBytes {
		return nil, nil, nil, false
	}
	scratch = neonScratchPool.Get().(*neonScratchBuf)
	basePtr := uintptr(unsafe.Pointer(&scratch.data[0]))
	if lhsLen > 0 {
		lhsSlice = unsafe.Slice((*T)(unsafe.Pointer(basePtr)), lhsLen)
	}
	if rhsLen > 0 {
		rhsSlice = unsafe.Slice((*T)(unsafe.Pointer(basePtr+uintptr(lhsBytes))), rhsLen)
	}
	return scratch, lhsSlice, rhsSlice, true
}

// PutScratchPair returns a scratch buffer acquired via GetScratchPair back to neonScratchPool.
func PutScratchPair(scratch *neonScratchBuf) {
	if scratch != nil {
		neonScratchPool.Put(scratch)
	}
}

// unsafePackLHS forwards to matmul.UnsafePackLHS.
func unsafePackLHS[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows int) {
	matmul.UnsafePackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
}

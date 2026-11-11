// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/support/envutil"
)

// Auto-generate alternate specialized versions of AVX2 operations -- for half-precision input data types.
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=router.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small_transposed.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small_nontransposed.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=large.go -tags=bf16,f16,f64

var (
	// AVX2ParamsFloat32 are the parameters to use for Float32, tuned for the 16 registers implementations.
	AVX2ParamsFloat32 = matmul.CacheParams{
		LHSL1KernelRows:      4,   // Mr: Uses 4 YMM registers for accumulation rows, this number must be a multiple of 4
		RHSL1KernelCols:      16,  // Nr: Uses 2 YMM registers for accumulation cols, each holds 8 values
		PanelContractingSize: 128, // Kc: A strip fits comfortably in L1D cache
		LHSPanelCrossSize:    32,  // Mc: Fits in L2 cache (multiple of LHSL1KernelRows)
		RHSPanelCrossSize:    256, // Nc: Fits in L2/L3 cache (multiple of RHSL1KernelCols)
	}

	// AVX2ParamsBFloat16 are the parameters to use for BFloat16, tuned for the 16 registers implementations.
	AVX2ParamsBFloat16 = matmul.CacheParams{
		LHSL1KernelRows:      4,   // Mr: Uses 4 YMM registers for accumulation rows
		RHSL1KernelCols:      16,  // Nr: Uses 2 YMM registers for accumulation cols, each holds 8 values (since we convert to F32)
		PanelContractingSize: 128, // Kc: A strip fits comfortably in L1D cache
		LHSPanelCrossSize:    32,  // Mc: Fits in L2 cache
		RHSPanelCrossSize:    768, // Nc: Fits in L3 cache
	}

	// AVX2ParamsFloat16 are the parameters to use for Float16.
	AVX2ParamsFloat16 = AVX2ParamsBFloat16

	// AVX2ParamsFloat64 are the parameters to use for Float64.
	AVX2ParamsFloat64 = matmul.CacheParams{
		LHSL1KernelRows:      4,   // Mr: Uses 4 YMM registers for accumulation rows
		RHSL1KernelCols:      8,   // Nr: Uses 2 YMM registers for accumulation cols, each holds 4 values
		PanelContractingSize: 64,  // Kc: A strip fits in L1 cache
		LHSPanelCrossSize:    16,  // Mc: Fits in L2 cache
		RHSPanelCrossSize:    256, // Nc: Fits in L3 cache
	}
)

func init() {
	if !AVX2UseAsm {
		AVX2ParamsFloat32.RHSL1KernelCols = 16
		AVX2ParamsFloat32.RHSPanelCrossSize = 512
	}
	if kc := envutil.MustReadInt(envutil.GoBackendAVX2_KC, 0); kc > 0 {
		AVX2ParamsFloat32.PanelContractingSize = kc
		AVX2ParamsFloat16.PanelContractingSize = kc
		AVX2ParamsBFloat16.PanelContractingSize = kc
		AVX2ParamsFloat64.PanelContractingSize = kc
	}
	if mc := envutil.MustReadInt(envutil.GoBackendAVX2_MC, 0); mc > 0 {
		AVX2ParamsFloat32.LHSPanelCrossSize = mc
		AVX2ParamsFloat16.LHSPanelCrossSize = mc
		AVX2ParamsBFloat16.LHSPanelCrossSize = mc
		AVX2ParamsFloat64.LHSPanelCrossSize = mc
	}
	if nc := envutil.MustReadInt(envutil.GoBackendAVX2_NC, 0); nc > 0 {
		AVX2ParamsFloat32.RHSPanelCrossSize = nc
		AVX2ParamsFloat16.RHSPanelCrossSize = nc
		AVX2ParamsBFloat16.RHSPanelCrossSize = nc
		AVX2ParamsFloat64.RHSPanelCrossSize = nc
	}
	if !envutil.MustReadBool(matmul.EnabledEnv, true) {
		return
	}

	if gobackend.IsAVX2Allowed {
		registerAVX2(false)
	}
}

func RegisterAVX2ForTests() {
	registerAVX2(true)
}

func registerAVX2(forTests bool) {
	dot.RegisterImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, avx2RouterFloat32, matmul.PriorityAVX2, forTests)
	dot.RegisterImplementation("simd:avx2", dot.LayoutTransposed, dtypes.Float32, dtypes.Float32, avx2RouterFloat32, matmul.PriorityAVX2, forTests)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, avx2LargeMultiFloat32)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutTransposed, dtypes.Float32, dtypes.Float32, avx2LargeMultiFloat32)

	dot.RegisterImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, avx2RouterBFloat16, matmul.PriorityAVX2, forTests)
	dot.RegisterImplementation("simd:avx2", dot.LayoutTransposed, dtypes.BFloat16, dtypes.Float32, avx2RouterBFloat16, matmul.PriorityAVX2, forTests)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, avx2LargeMultiBFloat16)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutTransposed, dtypes.BFloat16, dtypes.Float32, avx2LargeMultiBFloat16)

	dot.RegisterImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, avx2RouterFloat16, matmul.PriorityAVX2, forTests)
	dot.RegisterImplementation("simd:avx2", dot.LayoutTransposed, dtypes.Float16, dtypes.Float32, avx2RouterFloat16, matmul.PriorityAVX2, forTests)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, avx2LargeMultiFloat16)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutTransposed, dtypes.Float16, dtypes.Float32, avx2LargeMultiFloat16)

	dot.RegisterImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, avx2RouterFloat64, matmul.PriorityAVX2, forTests)
	dot.RegisterImplementation("simd:avx2", dot.LayoutTransposed, dtypes.Float64, dtypes.Float64, avx2RouterFloat64, matmul.PriorityAVX2, forTests)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, avx2LargeMultiFloat64)
	matmul.RegisterMultiRHSImplementation("simd:avx2", dot.LayoutTransposed, dtypes.Float64, dtypes.Float64, avx2LargeMultiFloat64)
}

// avx2ReduceSumFloat32x8 performs a horizontal reduction of a Float32x8 vector.
func avx2ReduceSumFloat32x8(x8 archsimd.Float32x8) float32 {
	x4 := x8.GetHi().Add(x8.GetLo())
	x4sum := x4.ConcatAddPairs(x4)
	return x4sum.GetElem(0) + x4sum.GetElem(1)
}

// avx2ReduceSumFloat64x4 performs a horizontal reduction of a Float64x4 vector.
func avx2ReduceSumFloat64x4(x4 archsimd.Float64x4) float64 {
	x2 := x4.GetHi().Add(x4.GetLo())
	return x2.GetElem(0) + x2.GetElem(1)
}

// avx2PackRHSNonTransposed packs a slice of the RHS matrix into a panel composed of "strips".
func avx2PackRHSNonTransposed[T gotype.ScalarNotComplex](
	rhs, panel []T,
	rhsRowStart, rhsColStart, rhsCols,
	contractingRows, copyCols, kernelCols int) {
	if contractingRows == 0 || copyCols == 0 {
		return
	}

	var zero T
	bytesPerElement := uintptr(unsafe.Sizeof(zero))
	rhsBasePtr := unsafe.Pointer(unsafe.SliceData(rhs))
	panelBasePtr := unsafe.Pointer(unsafe.SliceData(panel))
	copyColsBytesAll := uintptr(copyCols) * bytesPerElement
	rhsStrideBytes := uintptr(rhsCols) * bytesPerElement
	rhsColStartBytes := uintptr(rhsColStart) * bytesPerElement
	kernelColsBytes := uintptr(kernelCols) * bytesPerElement
	if kernelColsBytes%32 != 0 {
		panic("avx2PackRHS: kernelColsBytes must be a multiple of 32")
	}

	panelOffset := uintptr(0)
	stripColIdx := uintptr(0)
	if AVX2UseAsm && (kernelColsBytes == 128 || kernelColsBytes == 96 || kernelColsBytes == 64 || kernelColsBytes == 32) {
		numFullStrips := int(copyColsBytesAll / kernelColsBytes)
		if numFullStrips > 0 {
			rhsStartPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+rhsColStartBytes)
			avx2PackRHSFullStripsAsm(rhsStartPtr, panelBasePtr, rhsStrideBytes, contractingRows, numFullStrips, int(kernelColsBytes))
			stripColIdx = uintptr(numFullStrips) * kernelColsBytes
			panelOffset = uintptr(numFullStrips*contractingRows) * kernelColsBytes
		}
	} else {
		switch kernelColsBytes {
		case 128:
			for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
				rhsPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+rhsColStartBytes+stripColIdx)
				pDst := unsafe.Add(panelBasePtr, panelOffset)
				for k := 0; ; {
					v0 := archsimd.LoadUint8x32Array((*[32]uint8)(rhsPtr))
					v1 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsPtr, 32)))
					v2 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsPtr, 64)))
					v3 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsPtr, 96)))
					v0.StoreArray((*[32]uint8)(pDst))
					v1.StoreArray((*[32]uint8)(unsafe.Add(pDst, 32)))
					v2.StoreArray((*[32]uint8)(unsafe.Add(pDst, 64)))
					v3.StoreArray((*[32]uint8)(unsafe.Add(pDst, 96)))
					if k++; k >= contractingRows {
						break
					}
					pDst = unsafe.Add(pDst, kernelColsBytes)
					rhsPtr = unsafe.Add(rhsPtr, rhsStrideBytes)
				}
				panelOffset += uintptr(contractingRows) * kernelColsBytes
			}
		case 96:
			for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
				rhsPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+rhsColStartBytes+stripColIdx)
				pDst := unsafe.Add(panelBasePtr, panelOffset)
				for k := 0; ; {
					v0 := archsimd.LoadUint8x32Array((*[32]uint8)(rhsPtr))
					v1 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsPtr, 32)))
					v2 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsPtr, 64)))
					v0.StoreArray((*[32]uint8)(pDst))
					v1.StoreArray((*[32]uint8)(unsafe.Add(pDst, 32)))
					v2.StoreArray((*[32]uint8)(unsafe.Add(pDst, 64)))
					if k++; k >= contractingRows {
						break
					}
					pDst = unsafe.Add(pDst, kernelColsBytes)
					rhsPtr = unsafe.Add(rhsPtr, rhsStrideBytes)
				}
				panelOffset += uintptr(contractingRows) * kernelColsBytes
			}
		case 64:
			for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
				rhsPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+rhsColStartBytes+stripColIdx)
				pDst := unsafe.Add(panelBasePtr, panelOffset)
				for k := 0; ; {
					v0 := archsimd.LoadUint8x32Array((*[32]uint8)(rhsPtr))
					v1 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsPtr, 32)))
					v0.StoreArray((*[32]uint8)(pDst))
					v1.StoreArray((*[32]uint8)(unsafe.Add(pDst, 32)))
					if k++; k >= contractingRows {
						break
					}
					pDst = unsafe.Add(pDst, kernelColsBytes)
					rhsPtr = unsafe.Add(rhsPtr, rhsStrideBytes)
				}
				panelOffset += uintptr(contractingRows) * kernelColsBytes
			}
		case 32:
			for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
				rhsPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+rhsColStartBytes+stripColIdx)
				pDst := unsafe.Add(panelBasePtr, panelOffset)
				for k := 0; ; {
					v0 := archsimd.LoadUint8x32Array((*[32]uint8)(rhsPtr))
					v0.StoreArray((*[32]uint8)(pDst))
					if k++; k >= contractingRows {
						break
					}
					pDst = unsafe.Add(pDst, kernelColsBytes)
					rhsPtr = unsafe.Add(rhsPtr, rhsStrideBytes)
				}
				panelOffset += uintptr(contractingRows) * kernelColsBytes
			}
		default:
			for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
				rhsPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+rhsColStartBytes+stripColIdx)
				pDstRow := unsafe.Add(panelBasePtr, panelOffset)
				for k := 0; ; {
					for kernelColIdx := uintptr(0); kernelColIdx < kernelColsBytes; kernelColIdx += 32 {
						v0 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsPtr, kernelColIdx)))
						v0.StoreArray((*[32]uint8)(unsafe.Add(pDstRow, kernelColIdx)))
					}
					if k++; k >= contractingRows {
						break
					}
					pDstRow = unsafe.Add(pDstRow, kernelColsBytes)
					rhsPtr = unsafe.Add(rhsPtr, rhsStrideBytes)
				}
				panelOffset += uintptr(contractingRows) * kernelColsBytes
			}
		}
	}

	copyColsBytes := copyColsBytesAll - stripColIdx
	if copyColsBytes == 0 {
		return
	}

	rhsRowPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+rhsColStartBytes+stripColIdx)
	panelRowPtr := unsafe.Add(panelBasePtr, panelOffset)
	padBytes := kernelColsBytes - copyColsBytes
	for k := 0; ; {
		colByteIdx := uintptr(0)
		for ; colByteIdx+32 <= copyColsBytes; colByteIdx += 32 {
			v0 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(rhsRowPtr, colByteIdx)))
			v0.StoreArray((*[32]uint8)(unsafe.Add(panelRowPtr, colByteIdx)))
		}
		remainingBytes := copyColsBytes - colByteIdx
		if remainingBytes > 0 {
			rhsSlice := unsafe.Slice((*uint8)(unsafe.Add(rhsRowPtr, colByteIdx)), remainingBytes)
			panelSlice := unsafe.Slice((*uint8)(unsafe.Add(panelRowPtr, colByteIdx)), remainingBytes)
			copy(panelSlice, rhsSlice)
		}
		if padBytes > 0 {
			clear(unsafe.Slice((*uint8)(unsafe.Add(panelRowPtr, copyColsBytes)), padBytes))
		}
		if k++; k >= contractingRows {
			break
		}
		rhsRowPtr = unsafe.Add(rhsRowPtr, rhsStrideBytes)
		panelRowPtr = unsafe.Add(panelRowPtr, kernelColsBytes)
	}
}

// avx2PackLHSKernelRows4 packs a block of size [copyRows, contractingCols] from the lhs matrix into a panel.
func avx2PackLHSKernelRows4[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols, kernelRows int) {
	if copyRows == 0 || contractingCols == 0 {
		return
	}

	var zero T
	bytesPerElement := uintptr(unsafe.Sizeof(zero))
	lhsBasePtr := unsafe.Pointer(unsafe.SliceData(lhs))
	panelBasePtr := unsafe.Pointer(unsafe.SliceData(panel))
	lhsStrideBytes := uintptr(lhsCols) * bytesPerElement
	lhsColStartBytes := uintptr(lhsColStart) * bytesPerElement
	contractingColsBytes := uintptr(contractingCols) * bytesPerElement

	kernelRowsBytes := uintptr(kernelRows) * bytesPerElement
	panelOffset := uintptr(0)

	stripRowIdx := 0

	if kernelRows != 4 {
		if kernelRows > 0 && kernelRows%8 == 0 && bytesPerElement == 4 && copyRows > 0 && contractingCols > 0 {
			fullRows := (copyRows / kernelRows) * kernelRows
			if fullRows > 0 {
				avx2PackLHSStrided8x8Uint32(lhsBasePtr, panelBasePtr, lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols, kernelRows)
			}
			if fullRows == copyRows {
				return
			}
			panelElemOffset := fullRows * contractingCols
			unsafePackLHS(lhs, panel[panelElemOffset:], lhsRowStart+fullRows, lhsColStart, lhsCols, copyRows-fullRows, contractingCols, kernelRows)
			return
		}
		unsafePackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
		return
	}

	if AVX2UseAsm {
		handled := false
		fullRows := copyRows & ^3
		switch lhsTyped := any(lhs).(type) {
		case []float32:
			if fullRows > 0 {
				avx2PackLHSKernelRows4Float32Asm(lhsTyped, any(panel).([]float32), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		case []float64:
			if fullRows > 0 {
				avx2PackLHSKernelRows4Float64Asm(lhsTyped, any(panel).([]float64), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		case []float16.Float16:
			if fullRows > 0 {
				avx2PackLHSKernelRows4Float16Asm(lhsTyped, any(panel).([]float16.Float16), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		case []bfloat16.BFloat16:
			if fullRows > 0 {
				avx2PackLHSKernelRows4BFloat16Asm(lhsTyped, any(panel).([]bfloat16.BFloat16), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		}

		if handled {
			if fullRows == copyRows {
				return
			}
			stripRowIdx = fullRows
			panelOffset = uintptr(stripRowIdx*contractingCols) * bytesPerElement
			goto edge_strip
		}
	}

	for ; stripRowIdx < copyRows-kernelRows+1; stripRowIdx += kernelRows {
		colByteIdx := uintptr(0)
		lhsRow0Base := unsafe.Add(lhsBasePtr, uintptr(lhsRowStart+stripRowIdx)*lhsStrideBytes+lhsColStartBytes)
		lhsRow1Base := unsafe.Add(lhsRow0Base, lhsStrideBytes)
		lhsRow2Base := unsafe.Add(lhsRow1Base, lhsStrideBytes)
		lhsRow3Base := unsafe.Add(lhsRow2Base, lhsStrideBytes)

		for ; colByteIdx+32 <= contractingColsBytes; colByteIdx += 32 {
			v0 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(lhsRow0Base, colByteIdx)))
			v1 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(lhsRow1Base, colByteIdx)))
			v2 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(lhsRow2Base, colByteIdx)))
			v3 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(lhsRow3Base, colByteIdx)))
			pDst := unsafe.Add(panelBasePtr, panelOffset)

			switch bytesPerElement {
			case 4:
				t0, t1, t2, t3 := avx2Transpose4x8x32bits(v0.AsUint32x8(), v1.AsUint32x8(), v2.AsUint32x8(), v3.AsUint32x8())
				t0.StoreArray((*[8]uint32)(pDst))
				t1.StoreArray((*[8]uint32)(unsafe.Add(pDst, 32)))
				t2.StoreArray((*[8]uint32)(unsafe.Add(pDst, 64)))
				t3.StoreArray((*[8]uint32)(unsafe.Add(pDst, 96)))
				panelOffset += 128
			case 8:
				t0, t1, t2, t3 := avx2Transpose4x4x64bits(v0.AsUint64x4(), v1.AsUint64x4(), v2.AsUint64x4(), v3.AsUint64x4())
				t0.StoreArray((*[4]uint64)(pDst))
				t1.StoreArray((*[4]uint64)(unsafe.Add(pDst, 32)))
				t2.StoreArray((*[4]uint64)(unsafe.Add(pDst, 64)))
				t3.StoreArray((*[4]uint64)(unsafe.Add(pDst, 96)))
				panelOffset += 128
			case 2:
				t0, t1, t2, t3 := avx2Transpose4x16x16bits(v0.AsUint16x16(), v1.AsUint16x16(), v2.AsUint16x16(), v3.AsUint16x16())
				t0.StoreArray((*[16]uint16)(pDst))
				t1.StoreArray((*[16]uint16)(unsafe.Add(pDst, 32)))
				t2.StoreArray((*[16]uint16)(unsafe.Add(pDst, 64)))
				t3.StoreArray((*[16]uint16)(unsafe.Add(pDst, 96)))
				panelOffset += 128
			case 1:
				var tmp0, tmp1, tmp2, tmp3 [32]uint8
				v0.StoreArray(&tmp0)
				v1.StoreArray(&tmp1)
				v2.StoreArray(&tmp2)
				v3.StoreArray(&tmp3)
				for i := uintptr(0); i < 32; i++ {
					pElem := unsafe.Add(pDst, i*kernelRowsBytes)
					*(*uint8)(pElem) = tmp0[i]
					*(*uint8)(unsafe.Add(pElem, 1)) = tmp1[i]
					*(*uint8)(unsafe.Add(pElem, 2)) = tmp2[i]
					*(*uint8)(unsafe.Add(pElem, 3)) = tmp3[i]
				}
				panelOffset += 32 * kernelRowsBytes
			}
		}

		for col := int(colByteIdx / bytesPerElement); col < contractingCols; col++ {
			srcOff := uintptr(col) * bytesPerElement
			pDst := unsafe.Add(panelBasePtr, panelOffset)
			switch bytesPerElement {
			case 8:
				*(*uint64)(pDst) = *(*uint64)(unsafe.Add(lhsRow0Base, srcOff))
				*(*uint64)(unsafe.Add(pDst, 8)) = *(*uint64)(unsafe.Add(lhsRow1Base, srcOff))
				*(*uint64)(unsafe.Add(pDst, 16)) = *(*uint64)(unsafe.Add(lhsRow2Base, srcOff))
				*(*uint64)(unsafe.Add(pDst, 24)) = *(*uint64)(unsafe.Add(lhsRow3Base, srcOff))
			case 4:
				*(*uint32)(pDst) = *(*uint32)(unsafe.Add(lhsRow0Base, srcOff))
				*(*uint32)(unsafe.Add(pDst, 4)) = *(*uint32)(unsafe.Add(lhsRow1Base, srcOff))
				*(*uint32)(unsafe.Add(pDst, 8)) = *(*uint32)(unsafe.Add(lhsRow2Base, srcOff))
				*(*uint32)(unsafe.Add(pDst, 12)) = *(*uint32)(unsafe.Add(lhsRow3Base, srcOff))
			case 2:
				*(*uint16)(pDst) = *(*uint16)(unsafe.Add(lhsRow0Base, srcOff))
				*(*uint16)(unsafe.Add(pDst, 2)) = *(*uint16)(unsafe.Add(lhsRow1Base, srcOff))
				*(*uint16)(unsafe.Add(pDst, 4)) = *(*uint16)(unsafe.Add(lhsRow2Base, srcOff))
				*(*uint16)(unsafe.Add(pDst, 6)) = *(*uint16)(unsafe.Add(lhsRow3Base, srcOff))
			case 1:
				*(*uint8)(pDst) = *(*uint8)(unsafe.Add(lhsRow0Base, srcOff))
				*(*uint8)(unsafe.Add(pDst, 1)) = *(*uint8)(unsafe.Add(lhsRow1Base, srcOff))
				*(*uint8)(unsafe.Add(pDst, 2)) = *(*uint8)(unsafe.Add(lhsRow2Base, srcOff))
				*(*uint8)(unsafe.Add(pDst, 3)) = *(*uint8)(unsafe.Add(lhsRow3Base, srcOff))
			}
			panelOffset += kernelRowsBytes
		}
	}

edge_strip:
	if stripRowIdx < copyRows {
		remainingRows := copyRows - stripRowIdx
		lhsRow0Base := unsafe.Add(lhsBasePtr, uintptr(lhsRowStart+stripRowIdx)*lhsStrideBytes+lhsColStartBytes)
		for col := range contractingCols {
			pColSrc := unsafe.Add(lhsRow0Base, uintptr(col)*bytesPerElement)
			pDst := unsafe.Add(panelBasePtr, panelOffset)
			for r := range remainingRows {
				lhsPtr := unsafe.Add(pColSrc, uintptr(r)*lhsStrideBytes)
				switch bytesPerElement {
				case 8:
					*(*uint64)(unsafe.Add(pDst, uintptr(r)*8)) = *(*uint64)(lhsPtr)
				case 4:
					*(*uint32)(unsafe.Add(pDst, uintptr(r)*4)) = *(*uint32)(lhsPtr)
				case 2:
					*(*uint16)(unsafe.Add(pDst, uintptr(r)*2)) = *(*uint16)(lhsPtr)
				case 1:
					*(*uint8)(unsafe.Add(pDst, uintptr(r))) = *(*uint8)(lhsPtr)
				}
			}
			if remainingRows < kernelRows {
				padBytes := uintptr(kernelRows-remainingRows) * bytesPerElement
				clear(unsafe.Slice((*uint8)(unsafe.Add(pDst, uintptr(remainingRows)*bytesPerElement)), padBytes))
			}
			panelOffset += kernelRowsBytes
		}
	}
}

// avx2ApplyPackedOutputFloat32 applies the computed packedOutput to the final output.
func avx2ApplyPackedOutputFloat32(
	packedOutput, output []float32,
	isFirstContractingPanel bool,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int, // Global output offsets
	outputRowStride int,
	height, width int, // actual amount of data to copy
) {
	outputRowIdx := lhsRowOffset*outputRowStride + rhsColOffset
	packedRowIdx := 0
	if isFirstContractingPanel {
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			for ; c+8 <= width; c += 8 {
				packedVal := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Pointer(&packedOutput[packedColIdx])))
				packedVal.StoreArray((*[8]float32)(unsafe.Pointer(&output[outputColIdx])))
				packedColIdx += 8
				outputColIdx += 8
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
				packedVal := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Pointer(&packedOutput[packedColIdx])))
				outputVal := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Pointer(&output[outputColIdx])))
				outputVal = packedVal.Add(outputVal)
				outputVal.StoreArray((*[8]float32)(unsafe.Pointer(&output[outputColIdx])))
				packedColIdx += 8
				outputColIdx += 8
			}
			for i := range width - c {
				output[outputColIdx+i] += packedOutput[packedColIdx+i]
			}
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	}
}

// avx2ApplyPackedOutputFloat64 applies the computed packedOutput to the final output.
func avx2ApplyPackedOutputFloat64(
	packedOutput, output []float64,
	isFirstContractingPanel bool,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int, // Global output offsets
	outputRowStride int,
	height, width int, // actual amount of data to copy
) {
	outputRowIdx := lhsRowOffset*outputRowStride + rhsColOffset
	packedRowIdx := 0
	if isFirstContractingPanel {
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			for ; c+4 <= width; c += 4 {
				packedVal := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Pointer(&packedOutput[packedColIdx])))
				packedVal.StoreArray((*[4]float64)(unsafe.Pointer(&output[outputColIdx])))
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
			for ; c+4 <= width; c += 4 {
				packedVal := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Pointer(&packedOutput[packedColIdx])))
				outputVal := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Pointer(&output[outputColIdx])))
				outputVal = packedVal.Add(outputVal)
				outputVal.StoreArray((*[4]float64)(unsafe.Pointer(&output[outputColIdx])))
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

// unsafePackLHS forwards to matmul.UnsafePackLHS.
func unsafePackLHS[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows int) {
	matmul.UnsafePackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
}

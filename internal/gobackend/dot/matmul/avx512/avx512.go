// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

// The AVX512 implementation must have one generate version per dtype pair, since generics are not supported in archsimd.
// (Maybe a later version with go-highway or midway will change that)
//
// For now we only have implemented (InputDType, OutputDType):
//
// - (Float32, Float32)
// - (Bfloat16, Float32)
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

// Auto-generate alternate specialized versions of AVX512 operations -- for half-precision input data types.
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=router.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small_transposed.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=small_nontransposed.go -tags=bf16,f16,f64
//go:generate go run github.com/gomlx/compute/internal/cmd/alternates_generator -base=large.go -tags=bf16,f16,f64

var (
	// AVX512UseAsm enables the assembly microkernel for large matrices (8 rows x 32 cols for F32/F16/BF16, 8 rows x 16 cols for F64).
	// Set GOMLX_GO_AVX512_ASM=false to disable and use the Go SIMD kernel (4 rows x 32 cols).
	AVX512UseAsm = envutil.MustReadBool(envutil.GoBackendAVX512_ASM, true)

	// AVX512ParamsFloat32 are the parameters to use for Float32, tuned for the 16 registers implementations.
	AVX512ParamsFloat32 = matmul.CacheParams{
		LHSL1KernelRows:      8,   // Mr: 8 rows
		RHSL1KernelCols:      32,  // Nr: 32 cols
		PanelContractingSize: 192, // Kc: A strip fits in L1 cache
		LHSPanelCrossSize:    32,  // Mc: Fits in L2 cache (multiple of LHSL1KernelRows)
		RHSPanelCrossSize:    512, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols)
	}

	// AVX512ParamsBFloat16 are the parameters to use for BFloat16, tuned for the 16 registers implementations.
	AVX512ParamsBFloat16 = matmul.CacheParams{
		LHSL1KernelRows:      8,   // Mr: 8 rows
		RHSL1KernelCols:      32,  // Nr: 32 cols
		PanelContractingSize: 128, // Kc: A strip fits in L1 cache
		LHSPanelCrossSize:    32,  // Mc: Fits in L2 cache (multiple of LHSL1KernelRows)
		RHSPanelCrossSize:    768, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols)
	}

	// AVX512ParamsFloat16 are the parameters to use for Float16, tuned for the 16 registers implementations.
	AVX512ParamsFloat16 = AVX512ParamsBFloat16

	// AVX512ParamsFloat64 are the parameters to use for Float64, tuned for the 16 registers implementations.
	AVX512ParamsFloat64 = matmul.CacheParams{
		LHSL1KernelRows:      8,   // Mr: 8 rows
		RHSL1KernelCols:      16,  // Nr: 16 cols
		PanelContractingSize: 64,  // Kc: A strip fits in L1 cache
		LHSPanelCrossSize:    16,  // Mc: Fits in L2 cache (multiple of LHSL1KernelRows)
		RHSPanelCrossSize:    256, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols)
	}
)

func init() {
	if !AVX512UseAsm {
		AVX512ParamsFloat32.LHSL1KernelRows = 4
		AVX512ParamsFloat16.LHSL1KernelRows = 4
		AVX512ParamsBFloat16.LHSL1KernelRows = 4
		AVX512ParamsFloat64.LHSL1KernelRows = 4
	}
	if kc := envutil.MustReadInt(envutil.GoBackendAVX512_KC, 0); kc > 0 {
		AVX512ParamsFloat32.PanelContractingSize = kc
		AVX512ParamsFloat16.PanelContractingSize = kc
		AVX512ParamsBFloat16.PanelContractingSize = kc
		AVX512ParamsFloat64.PanelContractingSize = kc
	}
	if mc := envutil.MustReadInt(envutil.GoBackendAVX512_MC, 0); mc > 0 {
		AVX512ParamsFloat32.LHSPanelCrossSize = mc
		AVX512ParamsFloat16.LHSPanelCrossSize = mc
		AVX512ParamsBFloat16.LHSPanelCrossSize = mc
		AVX512ParamsFloat64.LHSPanelCrossSize = mc
	}
	if nc := envutil.MustReadInt(envutil.GoBackendAVX512_NC, 0); nc > 0 {
		AVX512ParamsFloat32.RHSPanelCrossSize = nc
		AVX512ParamsFloat16.RHSPanelCrossSize = nc
		AVX512ParamsBFloat16.RHSPanelCrossSize = nc
		AVX512ParamsFloat64.RHSPanelCrossSize = nc
	}
	if !envutil.MustReadBool(matmul.EnabledEnv, true) {
		return
	}

	allowed := envutil.MustReadBool(envutil.GoBackendSIMD_AVX512, true)
	if allowed && archsimd.X86.AVX512() {
		registerAVX512(false)
	}
}

func RegisterAVX512ForTests() {
	registerAVX512(true)
}

func registerAVX512(forTests bool) {
	dot.RegisterImplementation("simd:avx512", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, avx512RouterFloat32, matmul.PriorityAVX512, forTests)
	dot.RegisterImplementation("simd:avx512", dot.LayoutTransposed, dtypes.Float32, dtypes.Float32, avx512RouterFloat32, matmul.PriorityAVX512, forTests)

	dot.RegisterImplementation("simd:avx512", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, avx512RouterBFloat16, matmul.PriorityAVX512, forTests)
	dot.RegisterImplementation("simd:avx512", dot.LayoutTransposed, dtypes.BFloat16, dtypes.Float32, avx512RouterBFloat16, matmul.PriorityAVX512, forTests)

	dot.RegisterImplementation("simd:avx512", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, avx512RouterFloat16, matmul.PriorityAVX512, forTests)
	dot.RegisterImplementation("simd:avx512", dot.LayoutTransposed, dtypes.Float16, dtypes.Float32, avx512RouterFloat16, matmul.PriorityAVX512, forTests)

	dot.RegisterImplementation("simd:avx512", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, avx512RouterFloat64, matmul.PriorityAVX512, forTests)
	dot.RegisterImplementation("simd:avx512", dot.LayoutTransposed, dtypes.Float64, dtypes.Float64, avx512RouterFloat64, matmul.PriorityAVX512, forTests)
}

// avx512ReduceSumFloat32x16 performs a horizontal reduction of a Float32x16 vector.
func avx512ReduceSumFloat32x16(x16 archsimd.Float32x16) float32 {
	x8 := x16.GetHi().Add(x16.GetLo())
	x4 := x8.GetHi().Add(x8.GetLo())
	x4sum := x4.ConcatAddPairs(x4)
	return x4sum.GetElem(0) + x4sum.GetElem(1)
}

// avx512ReduceSumFloat64x8 performs a horizontal reduction of a Float64x8 vector.
func avx512ReduceSumFloat64x8(x8 archsimd.Float64x8) float64 {
	x4 := x8.GetHi().Add(x8.GetLo())
	x2 := x4.GetHi().Add(x4.GetLo())
	return x2.GetElem(0) + x2.GetElem(1)
}

// castToArray16 is just a shortcut to help cast a pointer to a pointer to an array used by SIMD loaders.
func castToArray16[T gotype.ScalarNotComplex](ptr *T) *[16]T {
	return (*[16]T)(unsafe.Pointer(ptr))
}

// castToArray8 is just a shortcut to help cast a pointer to a pointer to an array used by SIMD loaders.
func castToArray8[T gotype.ScalarNotComplex](ptr *T) *[8]T {
	return (*[8]T)(unsafe.Pointer(ptr))
}

// avx512ApplyPackedOutputFloat32 applies the computed packedOutput to the final output.
// This code is hard-coded to Float32 for now.
func avx512ApplyPackedOutputFloat32(
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
		// First contracting panel, so we overwrite to the output (as it may not have been zero-initialized).
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			// 64-float unrolled loop (4 ZMM registers)
			for ; c+64 <= width; c += 64 {
				v0 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx]))
				v1 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx+16]))
				v2 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx+32]))
				v3 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx+48]))
				v0.StoreArray(castToArray16(&output[outputColIdx]))
				v1.StoreArray(castToArray16(&output[outputColIdx+16]))
				v2.StoreArray(castToArray16(&output[outputColIdx+32]))
				v3.StoreArray(castToArray16(&output[outputColIdx+48]))
				packedColIdx += 64
				outputColIdx += 64
			}
			// Vectorized loop (1 ZMM register)
			for ; c+16 <= width; c += 16 {
				packedVal := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx]))
				packedVal.StoreArray(castToArray16(&output[outputColIdx]))
				packedColIdx += 16
				outputColIdx += 16
			}

			// Scalar tail
			for i := range width - c {
				output[outputColIdx+i] = packedOutput[packedColIdx+i]
			}

			// Next row.
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}

	} else {
		// Not the first contracting panel, so we need to add to the existing values.
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			// 64-float unrolled loop (4 ZMM registers)
			for ; c+64 <= width; c += 64 {
				p0 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx]))
				p1 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx+16]))
				p2 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx+32]))
				p3 := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx+48]))
				o0 := archsimd.LoadFloat32x16Array(castToArray16(&output[outputColIdx]))
				o1 := archsimd.LoadFloat32x16Array(castToArray16(&output[outputColIdx+16]))
				o2 := archsimd.LoadFloat32x16Array(castToArray16(&output[outputColIdx+32]))
				o3 := archsimd.LoadFloat32x16Array(castToArray16(&output[outputColIdx+48]))
				p0.Add(o0).StoreArray(castToArray16(&output[outputColIdx]))
				p1.Add(o1).StoreArray(castToArray16(&output[outputColIdx+16]))
				p2.Add(o2).StoreArray(castToArray16(&output[outputColIdx+32]))
				p3.Add(o3).StoreArray(castToArray16(&output[outputColIdx+48]))
				packedColIdx += 64
				outputColIdx += 64
			}
			// Vectorized loop
			for ; c+16 <= width; c += 16 {
				packedVal := archsimd.LoadFloat32x16Array(castToArray16(&packedOutput[packedColIdx]))
				outputVal := archsimd.LoadFloat32x16Array(castToArray16(&output[outputColIdx]))
				outputVal = packedVal.Add(outputVal)
				outputVal.StoreArray(castToArray16(&output[outputColIdx]))
				packedColIdx += 16
				outputColIdx += 16
			}

			// Scalar tail
			for i := range width - c {
				output[outputColIdx+i] += packedOutput[packedColIdx+i]
			}

			// Next row.
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	}
}

// avx512ApplyPackedOutputFloat64 applies the computed packedOutput to the final output.
func avx512ApplyPackedOutputFloat64(
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
		// First contracting panel, so we overwrite to the output (as it may not have been zero-initialized).
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			// 32-float unrolled loop (4 ZMM registers)
			for ; c+32 <= width; c += 32 {
				v0 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx]))
				v1 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx+8]))
				v2 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx+16]))
				v3 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx+24]))
				v0.StoreArray(castToArray8(&output[outputColIdx]))
				v1.StoreArray(castToArray8(&output[outputColIdx+8]))
				v2.StoreArray(castToArray8(&output[outputColIdx+16]))
				v3.StoreArray(castToArray8(&output[outputColIdx+24]))
				packedColIdx += 32
				outputColIdx += 32
			}
			// Vectorized loop
			for ; c+8 <= width; c += 8 {
				packedVal := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx]))
				packedVal.StoreArray(castToArray8(&output[outputColIdx]))
				packedColIdx += 8
				outputColIdx += 8
			}

			// Scalar tail
			for i := range width - c {
				output[outputColIdx+i] = packedOutput[packedColIdx+i]
			}

			// Next row.
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}

	} else {
		// Not the first contracting panel, so we need to add to the existing values.
		for range height {
			c := 0
			outputColIdx := outputRowIdx
			packedColIdx := packedRowIdx
			// 32-float unrolled loop (4 ZMM registers)
			for ; c+32 <= width; c += 32 {
				p0 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx]))
				p1 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx+8]))
				p2 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx+16]))
				p3 := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx+24]))
				o0 := archsimd.LoadFloat64x8Array(castToArray8(&output[outputColIdx]))
				o1 := archsimd.LoadFloat64x8Array(castToArray8(&output[outputColIdx+8]))
				o2 := archsimd.LoadFloat64x8Array(castToArray8(&output[outputColIdx+16]))
				o3 := archsimd.LoadFloat64x8Array(castToArray8(&output[outputColIdx+24]))
				p0.Add(o0).StoreArray(castToArray8(&output[outputColIdx]))
				p1.Add(o1).StoreArray(castToArray8(&output[outputColIdx+8]))
				p2.Add(o2).StoreArray(castToArray8(&output[outputColIdx+16]))
				p3.Add(o3).StoreArray(castToArray8(&output[outputColIdx+24]))
				packedColIdx += 32
				outputColIdx += 32
			}
			// Vectorized loop
			for ; c+8 <= width; c += 8 {
				packedVal := archsimd.LoadFloat64x8Array(castToArray8(&packedOutput[packedColIdx]))
				outputVal := archsimd.LoadFloat64x8Array(castToArray8(&output[outputColIdx]))
				outputVal = packedVal.Add(outputVal)
				outputVal.StoreArray(castToArray8(&output[outputColIdx]))
				packedColIdx += 8
				outputColIdx += 8
			}

			// Scalar tail
			for i := range width - c {
				output[outputColIdx+i] += packedOutput[packedColIdx+i]
			}

			// Next row.
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	}
}

// avx512PackRHSNonTransposed packs a slice of the RHS (right-hand-side) matrix into a panel compose of "strips" for optimized
// matrix multiplication. The that stip is padded with 0 (important).
//
//   - rhs: matrix [rhsRows, rhsCols], where rhsRows >= rhsRowStart + contractingRows.
//   - panel: packed panel with enough space to store the [numStrips, contractingRows, kernelCols].
//     Where numStrips = ceil(copyCols / kernelCols), the last strip padded with 0s.
//   - rhsRowStart, rhsColStart: start of the slice that will be packed into the panel.
//   - rhsCols: number of columns in the rhs matrix.
//   - contractingRows: how many rows of rhs (the maximum allowed is given by CacheParams.PanelContractingSize)
//     that are going to be copied to the panel.
//   - copyCols: number of columns to copy to the panel, arragend in strips of kernelCols size. The last kernelCols
//     is padded with 0.
//   - kernelCols: we are packing in strips shaped [contractingRows, kernelCols] size (optimal for our algorithm).
//     For the AVX512 implementation kernelCols * sizeOf(T) (bytes) must be multiple of 64 bytes,
//     it will panic otherwise.
func avx512PackRHSNonTransposed[T gotype.ScalarNotComplex](
	rhs, panel []T,
	rhsRowStart, rhsColStart, rhsCols,
	contractingRows, copyCols, kernelCols int) {

	// We do our very best to convert the input into byte slices.
	var zero T
	bytesPerElement := uintptr(unsafe.Sizeof(zero))
	rhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(rhs)))
	panelBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(panel)))
	copyColsBytesAll := uintptr(copyCols) * bytesPerElement
	rhsStrideBytes := uintptr(rhsCols) * bytesPerElement // The row stride of the original matrix
	rhsColStartBytes := uintptr(rhsColStart) * bytesPerElement
	kernelColsBytes := uintptr(kernelCols) * bytesPerElement // Kernel columns in bytes (should be a multiple of 64 for AVX512)
	if kernelColsBytes%64 != 0 {
		panic("avx512PackRHS: kernelColsBytes must be a multiple of 64")
	}

	// From here on, we work only on byte indices.
	panelPtr := panelBasePtr
	// Iterate over full-strips (using all the kernelCols, so no padding needed).
	stripColIdx := uintptr(0)
	if AVX512UseAsm && (kernelColsBytes == 256 || kernelColsBytes == 128 || kernelColsBytes == 64) {
		numFullStrips := int(copyColsBytesAll / kernelColsBytes)
		if numFullStrips > 0 {
			rhsStartPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes
			avx512PackRHSFullStripsAsm(
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
	switch kernelColsBytes { // Multiple of 64.
	case 256:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				v0 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr)))
				v1 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr + 64)))
				v2 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr + 128)))
				v3 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr + 192)))
				v0.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr)))
				v1.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr + 64)))
				v2.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr + 128)))
				v3.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr + 192)))
				panelPtr += kernelColsBytes
				rhsPtr += rhsStrideBytes
			}
		}
	case 128:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				v0 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr)))
				v1 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr + 64)))
				v0.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr)))
				v1.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr + 64)))
				panelPtr += kernelColsBytes
				rhsPtr += rhsStrideBytes
			}
		}
	case 64:
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				v0 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr)))
				v0.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr)))
				panelPtr += kernelColsBytes
				rhsPtr += rhsStrideBytes
			}
		}
	default:
		// Copy 64 bytes at a time.
		for ; stripColIdx+kernelColsBytes <= copyColsBytesAll; stripColIdx += kernelColsBytes {
			rhsPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx
			for range contractingRows {
				rowRhsPtr := rhsPtr
				// kernelColIdx is a multiple of 64, we checked at the start.
				for kernelColIdx := uintptr(0); kernelColIdx < kernelColsBytes; kernelColIdx += 64 {
					v0 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rowRhsPtr)))
					v0.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr)))
					panelPtr += 64
					rowRhsPtr += 64
				}
				rhsPtr += rhsStrideBytes
			}
		}
	}

	// The last strip will have fewer than kernelColsBytes, and will need copying/padding.
	copyColsBytes := copyColsBytesAll - stripColIdx
	if copyColsBytes == 0 {
		return
	}

	// The panelPtr is already at the correct position for the last strip.
	rhsStripStartPtr := rhsBasePtr + uintptr(rhsRowStart)*rhsStrideBytes + rhsColStartBytes + stripColIdx

	// Iterate over the rows of rhs
	for range contractingRows {
		rhsPtr := rhsStripStartPtr
		colByteIdx := uintptr(0)
		// Assuming copy() doesn't use AVX512, we copy 64 bytes at a time ourselves.
		for ; colByteIdx+64 <= copyColsBytes; colByteIdx += 64 {
			// Copy 64 bytes at a time.
			v0 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(rhsPtr)))
			v0.StoreArray((*[64]uint8)(unsafe.Pointer(panelPtr)))
			rhsPtr += 64
			panelPtr += 64
		}
		// Copy the remaining bytes.
		remainingBytes := copyColsBytes - colByteIdx
		if remainingBytes > 0 {
			rhsSlice := unsafe.Slice((*uint8)(unsafe.Pointer(rhsPtr)), remainingBytes)
			panelSlice := unsafe.Slice((*uint8)(unsafe.Pointer(panelPtr)), remainingBytes)
			copy(panelSlice, rhsSlice)
			panelPtr += remainingBytes
		}

		// Zero-pad if strip is incomplete (edge of matrix)
		padBytes := kernelColsBytes - copyColsBytes
		for range padBytes {
			*(*uint8)(unsafe.Pointer(panelPtr)) = 0
			panelPtr++
		}

		rhsStripStartPtr += rhsStrideBytes
	}
}

// avx512PackLHSKernelRows4 packs a block of size [copyRows, contractingCols] from the lhs matrix into
// a panel. The panel is structured as [ceil(copyRows/kernelRows), contractingCols, kernelRows].
// It rearranges data into horizontal strips of height kernelRows.
//
//   - lhs: matrix [lhsRows, lhsCols], where lhsRows >= lhsRowStart + copyRows.
//   - panel: packed panel with enough space to store the [numStrips, contractingCols, kernelRows].
//     Where numStrips = ceil(copyRows / kernelRows), the last strip padded with 0s.
//   - lhsRowStart, lhsColStart: start of the slice that will be packed into the panel.
//   - lhsCols: number of columns in the lhs matrix (row stride).
//   - copyRows: how many rows of lhs to copy to the panel.
//   - contractingCols: number of columns to copy to the panel.
//   - kernelRows: we are packing in strips of kernelRows size.
//     For this AVX512 implementation kernelRows must be 4, it will panic otherwise.
func avx512PackLHSKernelRows4[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols, kernelRows int) {

	var zero T
	bytesPerElement := uintptr(unsafe.Sizeof(zero))
	lhsBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(lhs)))
	panelBasePtr := uintptr(unsafe.Pointer(unsafe.SliceData(panel)))

	lhsStrideBytes := uintptr(lhsCols) * bytesPerElement
	lhsColStartBytes := uintptr(lhsColStart) * bytesPerElement
	contractingColsBytes := uintptr(contractingCols) * bytesPerElement

	if kernelRows == 8 {
		unsafePackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
		return
	}
	if kernelRows != 4 {
		panic("avx512PackLHSKernelRows4: kernelRows must be set to 4 or 8")
	}

	kernelRowsBytes := uintptr(kernelRows) * bytesPerElement
	panelPtr := panelBasePtr
	stripRowIdx := 0

	if AVX512UseAsm {
		handled := false
		fullRows := copyRows & ^3
		switch lhsTyped := any(lhs).(type) {
		case []float32:
			if fullRows > 0 {
				avx512PackLHSKernelRows4Float32Asm(lhsTyped, any(panel).([]float32), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		case []float64:
			if fullRows > 0 {
				avx512PackLHSKernelRows4Float64Asm(lhsTyped, any(panel).([]float64), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		case []float16.Float16:
			if fullRows > 0 {
				avx512PackLHSKernelRows4Float16Asm(lhsTyped, any(panel).([]float16.Float16), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		case []bfloat16.BFloat16:
			if fullRows > 0 {
				avx512PackLHSKernelRows4BFloat16Asm(lhsTyped, any(panel).([]bfloat16.BFloat16), lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			}
			handled = true
		}

		if handled {
			if fullRows == copyRows {
				return
			}
			stripRowIdx = fullRows
			panelPtr = panelBasePtr + uintptr(stripRowIdx*contractingCols)*bytesPerElement
			goto edge_strip
		}
	}

	// Iterate over full strips first:
	for ; stripRowIdx < copyRows-kernelRows+1; stripRowIdx += kernelRows {
		colByteIdx := uintptr(0)
		lhsRow0Ptr := lhsBasePtr + uintptr(lhsRowStart+stripRowIdx)*lhsStrideBytes + lhsColStartBytes
		lhsRow1Ptr := lhsRow0Ptr + lhsStrideBytes
		lhsRow2Ptr := lhsRow1Ptr + lhsStrideBytes
		lhsRow3Ptr := lhsRow2Ptr + lhsStrideBytes

		// Vectorized loop over cols (64 bytes at a time)
		for ; colByteIdx+64 <= contractingColsBytes; colByteIdx += 64 {
			v0 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(lhsRow0Ptr)))
			v1 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(lhsRow1Ptr)))
			v2 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(lhsRow2Ptr)))
			v3 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Pointer(lhsRow3Ptr)))

			switch bytesPerElement {
			case 4:
				t0, t1, t2, t3 := avx512Transpose4x16x32bits(v0.AsUint32x16(), v1.AsUint32x16(), v2.AsUint32x16(), v3.AsUint32x16())
				t0.StoreArray((*[16]uint32)(unsafe.Pointer(panelPtr)))
				t1.StoreArray((*[16]uint32)(unsafe.Pointer(panelPtr + 64)))
				t2.StoreArray((*[16]uint32)(unsafe.Pointer(panelPtr + 128)))
				t3.StoreArray((*[16]uint32)(unsafe.Pointer(panelPtr + 192)))

				panelPtr += 256 // 4 x 16 x uint32

			case 8:
				t0, t1, t2, t3 := avx512Transpose4x8x64bits(v0.AsUint64x8(), v1.AsUint64x8(), v2.AsUint64x8(), v3.AsUint64x8())
				t0.StoreArray((*[8]uint64)(unsafe.Pointer(panelPtr)))
				t1.StoreArray((*[8]uint64)(unsafe.Pointer(panelPtr + 64)))
				t2.StoreArray((*[8]uint64)(unsafe.Pointer(panelPtr + 128)))
				t3.StoreArray((*[8]uint64)(unsafe.Pointer(panelPtr + 192)))
				panelPtr += 256
			case 2:
				t0, t1, t2, t3 := avx512Transpose4x32x16bits(v0.AsUint16x32(), v1.AsUint16x32(), v2.AsUint16x32(), v3.AsUint16x32())
				t0.StoreArray((*[32]uint16)(unsafe.Pointer(panelPtr)))
				t1.StoreArray((*[32]uint16)(unsafe.Pointer(panelPtr + 64)))
				t2.StoreArray((*[32]uint16)(unsafe.Pointer(panelPtr + 128)))
				t3.StoreArray((*[32]uint16)(unsafe.Pointer(panelPtr + 192)))
				panelPtr += 256 // 4 x 16 x uint32

			case 1:
				var tmp0, tmp1, tmp2, tmp3 [64]uint8
				v0.StoreArray(&tmp0)
				v1.StoreArray(&tmp1)
				v2.StoreArray(&tmp2)
				v3.StoreArray(&tmp3)
				for i := uintptr(0); i < 64; i++ {
					*(*uint8)(unsafe.Pointer(panelPtr)) = tmp0[i]
					*(*uint8)(unsafe.Pointer(panelPtr + 1)) = tmp1[i]
					*(*uint8)(unsafe.Pointer(panelPtr + 2)) = tmp2[i]
					*(*uint8)(unsafe.Pointer(panelPtr + 3)) = tmp3[i]
					panelPtr += kernelRowsBytes
				}
			}

			lhsRow0Ptr += 64
			lhsRow1Ptr += 64
			lhsRow2Ptr += 64
			lhsRow3Ptr += 64
		}

		// Scalar tail
		for col := int(colByteIdx / bytesPerElement); col < contractingCols; col++ {
			switch bytesPerElement {
			case 8:
				*(*uint64)(unsafe.Pointer(panelPtr)) = *(*uint64)(unsafe.Pointer(lhsRow0Ptr))
				*(*uint64)(unsafe.Pointer(panelPtr + 8)) = *(*uint64)(unsafe.Pointer(lhsRow1Ptr))
				*(*uint64)(unsafe.Pointer(panelPtr + 16)) = *(*uint64)(unsafe.Pointer(lhsRow2Ptr))
				*(*uint64)(unsafe.Pointer(panelPtr + 24)) = *(*uint64)(unsafe.Pointer(lhsRow3Ptr))
			case 4:
				*(*uint32)(unsafe.Pointer(panelPtr)) = *(*uint32)(unsafe.Pointer(lhsRow0Ptr))
				*(*uint32)(unsafe.Pointer(panelPtr + 4)) = *(*uint32)(unsafe.Pointer(lhsRow1Ptr))
				*(*uint32)(unsafe.Pointer(panelPtr + 8)) = *(*uint32)(unsafe.Pointer(lhsRow2Ptr))
				*(*uint32)(unsafe.Pointer(panelPtr + 12)) = *(*uint32)(unsafe.Pointer(lhsRow3Ptr))
			case 2:
				*(*uint16)(unsafe.Pointer(panelPtr)) = *(*uint16)(unsafe.Pointer(lhsRow0Ptr))
				*(*uint16)(unsafe.Pointer(panelPtr + 2)) = *(*uint16)(unsafe.Pointer(lhsRow1Ptr))
				*(*uint16)(unsafe.Pointer(panelPtr + 4)) = *(*uint16)(unsafe.Pointer(lhsRow2Ptr))
				*(*uint16)(unsafe.Pointer(panelPtr + 6)) = *(*uint16)(unsafe.Pointer(lhsRow3Ptr))
			case 1:
				*(*uint8)(unsafe.Pointer(panelPtr)) = *(*uint8)(unsafe.Pointer(lhsRow0Ptr))
				*(*uint8)(unsafe.Pointer(panelPtr + 1)) = *(*uint8)(unsafe.Pointer(lhsRow1Ptr))
				*(*uint8)(unsafe.Pointer(panelPtr + 2)) = *(*uint8)(unsafe.Pointer(lhsRow2Ptr))
				*(*uint8)(unsafe.Pointer(panelPtr + 3)) = *(*uint8)(unsafe.Pointer(lhsRow3Ptr))
			}
			lhsRow0Ptr += bytesPerElement
			lhsRow1Ptr += bytesPerElement
			lhsRow2Ptr += bytesPerElement
			lhsRow3Ptr += bytesPerElement
			panelPtr += kernelRowsBytes
		}
	}

edge_strip:
	// Last strip, with less than kernelRows (4) valid rows, the rest needs to be zero-padded:
	if stripRowIdx < copyRows {
		remainingRows := copyRows - stripRowIdx
		lhsRow0Ptr := lhsBasePtr + uintptr(lhsRowStart+stripRowIdx)*lhsStrideBytes + lhsColStartBytes

		// Less than kernelRows (4) valid rows
		for range contractingCols {
			lhsPtr := lhsRow0Ptr

			for r := range remainingRows {
				switch bytesPerElement {
				case 8:
					*(*uint64)(unsafe.Pointer(panelPtr + uintptr(r)*8)) = *(*uint64)(unsafe.Pointer(lhsPtr))
				case 4:
					*(*uint32)(unsafe.Pointer(panelPtr + uintptr(r)*4)) = *(*uint32)(unsafe.Pointer(lhsPtr))
				case 2:
					*(*uint16)(unsafe.Pointer(panelPtr + uintptr(r)*2)) = *(*uint16)(unsafe.Pointer(lhsPtr))
				case 1:
					*(*uint8)(unsafe.Pointer(panelPtr + uintptr(r))) = *(*uint8)(unsafe.Pointer(lhsPtr))
				}
				lhsPtr += lhsStrideBytes
			}

			// Zero-pad if strip is incomplete (edge of matrix)
			if remainingRows < kernelRows {
				padBytes := uintptr(kernelRows-remainingRows) * bytesPerElement
				padStart := panelPtr + uintptr(remainingRows)*bytesPerElement
				for i := uintptr(0); i < padBytes; i++ {
					*(*uint8)(unsafe.Pointer(padStart + i)) = 0
				}
			}

			lhsRow0Ptr += bytesPerElement
			panelPtr += kernelRowsBytes
		}
	}
}

// GetBuffer forwards buffer allocation to the shared matmul package.
func GetBuffer[T gotype.Supported](backend *gobackend.Backend, length int) (*gobackend.Buffer, []T, bool) {
	return matmul.GetBuffer[T](backend, length)
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


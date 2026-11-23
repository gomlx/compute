// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

package avx512

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/gobackend"
)

// AVX512UseBF16 enables native AVX512_BF16 (VDPBF16PS) kernels and K2-interleaved packing
// for BFloat16 matrix multiplications. Defaults to gobackend.IsAVX512BF16Allowed (CPUID check
// gated by GOMLX_GO_SIMD_AVX512_BF16); when false, falls back to VPMOVZXWD+VPSLLD+VFMADD231PS.
var AVX512UseBF16 = gobackend.IsAVX512BF16Allowed

//go:noescape
func avx512LargeKernelBFloat16BF16Asm(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
)

//go:noescape
func avx512LargeKernelM1BFloat16BF16Asm(
	lhs []bfloat16.BFloat16,
	packedRHS []bfloat16.BFloat16,
	out []float32,
	contractingLen int,
	rhsActiveCols int,
	accumulate bool,
)

//go:noescape
func avx512PackLHSKernelRows8BFloat16K2Asm(
	lhs, panel []bfloat16.BFloat16,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols int,
)

//go:noescape
func avx512PackRHSTransposed32BFloat16K2Asm(
	rhs, panel []bfloat16.BFloat16,
	rhsRowStart, rhsColStart, rhsCols,
	copyRows, contractingCols int,
)

//go:noescape
func avx512PackRHSNonTransposed32BFloat16K2Asm(
	rhsPtr, panelPtr unsafe.Pointer,
	rhsStrideBytes uintptr,
	contractingRows, numFullStrips int,
)

//go:noescape
func avx512SmallTransposedTile4x4BFloat16BF16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func avx512SmallTransposedTile4x1BFloat16BF16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func avx512SmallNonTransposedTile4x64BFloat16BF16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

//go:noescape
func avx512SmallNonTransposedTile4x16BFloat16BF16Asm(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
)

// avx512BFloat16PaddedK returns the strip K dimension for BFloat16 panels:
// rounded up to even when using K2-interleaved VDPBF16PS layout, or unchanged otherwise.
func avx512BFloat16PaddedK(k int) int {
	if AVX512UseAsm && AVX512UseBF16 {
		return (k + 1) & ^1
	}
	return k
}

// avx512LargeKernelBFloat16AsmDispatch routes to the native VDPBF16PS microkernel
// when AVX512_BF16 is supported and enabled, or the F32-widening VFMADD231PS fallback otherwise.
func avx512LargeKernelBFloat16AsmDispatch(
	packedLHS, packedRHS []bfloat16.BFloat16,
	packedOutput []float32,
	lhsPanelRows, rhsPanelCols int,
	contractingLen int,
	lhsActiveRows, rhsActiveCols int,
	accumulate bool,
) {
	if AVX512UseBF16 {
		avx512LargeKernelBFloat16BF16Asm(
			packedLHS, packedRHS, packedOutput,
			lhsPanelRows, rhsPanelCols,
			contractingLen,
			lhsActiveRows, rhsActiveCols,
			accumulate,
		)
		return
	}
	avx512LargeKernelBFloat16Asm(
		packedLHS, packedRHS, packedOutput,
		lhsPanelRows, rhsPanelCols,
		contractingLen,
		lhsActiveRows, rhsActiveCols,
		accumulate,
	)
}

// avx512LargeKernelM1BFloat16AsmDispatch routes M=1 BFloat16 GEMV to the native VDPBF16PS
// microkernel when AVX512_BF16 is enabled, or the F32-widening VFMADD231PS M=1 fallback otherwise.
func avx512LargeKernelM1BFloat16AsmDispatch(
	lhs, packedRHS []bfloat16.BFloat16,
	out []float32,
	contractingLen int,
	rhsActiveCols int,
	accumulate bool,
) {
	if AVX512UseBF16 {
		avx512LargeKernelM1BFloat16BF16Asm(lhs, packedRHS, out, contractingLen, rhsActiveCols, accumulate)
		return
	}
	avx512LargeKernelM1BFloat16Asm(lhs, packedRHS, out, contractingLen, rhsActiveCols, accumulate)
}

func avx512SmallTransposedTile4x4BFloat16Dispatch(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0, rCol1, rCol2, rCol3 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
) {
	if AVX512UseBF16 {
		avx512SmallTransposedTile4x4BFloat16BF16Asm(
			lRow0, lRow1, lRow2, lRow3,
			rCol0, rCol1, rCol2, rCol3,
			contractingSize,
			outRow0, outRow1, outRow2, outRow3,
		)
		return
	}
	avx512SmallTransposedTile4x4BFloat16Asm(
		lRow0, lRow1, lRow2, lRow3,
		rCol0, rCol1, rCol2, rCol3,
		contractingSize,
		outRow0, outRow1, outRow2, outRow3,
	)
}

func avx512SmallTransposedTile4x1BFloat16Dispatch(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rCol0 unsafe.Pointer,
	contractingSize int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
) {
	if AVX512UseBF16 {
		avx512SmallTransposedTile4x1BFloat16BF16Asm(
			lRow0, lRow1, lRow2, lRow3,
			rCol0,
			contractingSize,
			outRow0, outRow1, outRow2, outRow3,
		)
		return
	}
	avx512SmallTransposedTile4x1BFloat16Asm(
		lRow0, lRow1, lRow2, lRow3,
		rCol0,
		contractingSize,
		outRow0, outRow1, outRow2, outRow3,
	)
}

func avx512SmallNonTransposedTile4x64BFloat16Dispatch(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
) {
	if AVX512UseBF16 {
		avx512SmallNonTransposedTile4x64BFloat16BF16Asm(
			lRow0, lRow1, lRow2, lRow3,
			rhs,
			contractingSize,
			rhsByteStride,
			outRow0, outRow1, outRow2, outRow3,
		)
		return
	}
	avx512SmallNonTransposedTile4x64BFloat16Asm(
		lRow0, lRow1, lRow2, lRow3,
		rhs,
		contractingSize,
		rhsByteStride,
		outRow0, outRow1, outRow2, outRow3,
	)
}

func avx512SmallNonTransposedTile4x16BFloat16Dispatch(
	lRow0, lRow1, lRow2, lRow3 unsafe.Pointer,
	rhs unsafe.Pointer,
	contractingSize int,
	rhsByteStride int,
	outRow0, outRow1, outRow2, outRow3 unsafe.Pointer,
) {
	if AVX512UseBF16 {
		avx512SmallNonTransposedTile4x16BFloat16BF16Asm(
			lRow0, lRow1, lRow2, lRow3,
			rhs,
			contractingSize,
			rhsByteStride,
			outRow0, outRow1, outRow2, outRow3,
		)
		return
	}
	avx512SmallNonTransposedTile4x16BFloat16Asm(
		lRow0, lRow1, lRow2, lRow3,
		rhs,
		contractingSize,
		rhsByteStride,
		outRow0, outRow1, outRow2, outRow3,
	)
}

// avx512PackLHSBFloat16 packs LHS (or transposed RHS) for BFloat16, selecting K2-interleaved
// packing when AVX512UseAsm && AVX512UseBF16, and standard packing otherwise.
func avx512PackLHSBFloat16(
	lhs, panel []bfloat16.BFloat16,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols, kernelRows int,
) {
	if AVX512UseAsm && AVX512UseBF16 {
		avx512PackLHSBFloat16K2(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
		return
	}
	avx512PackLHSKernelRows4(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
}

// avx512PackRHSNonTransposedBFloat16 packs non-transposed RHS for BFloat16, selecting
// K2-interleaved packing when AVX512UseAsm && AVX512UseBF16, and standard packing otherwise.
func avx512PackRHSNonTransposedBFloat16(
	rhs, panel []bfloat16.BFloat16,
	rhsRowStart, rhsColStart, rhsCols,
	contractingRows, copyCols, kernelCols int,
) {
	if AVX512UseAsm && AVX512UseBF16 {
		avx512PackRHSNonTransposedBFloat16K2(rhs, panel, rhsRowStart, rhsColStart, rhsCols, contractingRows, copyCols, kernelCols)
		return
	}
	avx512PackRHSNonTransposed(rhs, panel, rhsRowStart, rhsColStart, rhsCols, contractingRows, copyCols, kernelCols)
}

// avx512PackLHSBFloat16K2 packs [copyRows, contractingCols] from lhs into K2-interleaved strips:
// [ceil(copyRows/kernelRows), kPairs, kernelRows, 2] where kPairs = (contractingCols+1)/2.
// Any incomplete trailing strip rows or odd trailing K are zero-padded.
func avx512PackLHSBFloat16K2(
	lhs, panel []bfloat16.BFloat16,
	lhsRowStart, lhsColStart, lhsCols,
	copyRows, contractingCols, kernelRows int,
) {
	if copyRows == 0 || contractingCols == 0 {
		return
	}
	paddedK := (contractingCols + 1) & ^1
	kPairs := paddedK >> 1
	fullKPairs := contractingCols >> 1
	hasOddK := (contractingCols & 1) != 0

	stripRowIdx := 0
	if kernelRows == 8 {
		fullRows := copyRows & ^7
		if fullRows > 0 {
			avx512PackLHSKernelRows8BFloat16K2Asm(lhs, panel, lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			stripRowIdx = fullRows
		}
	} else if kernelRows == 32 {
		fullRows := copyRows & ^31
		if fullRows > 0 {
			avx512PackRHSTransposed32BFloat16K2Asm(lhs, panel, lhsRowStart, lhsColStart, lhsCols, fullRows, contractingCols)
			stripRowIdx = fullRows
		}
	}

	// Handle any remaining full strips (if kernelRows != 8 && != 32) and the partial edge strip.
	for ; stripRowIdx < copyRows; stripRowIdx += kernelRows {
		remRows := min(kernelRows, copyRows-stripRowIdx)
		stripBase := (stripRowIdx / kernelRows) * paddedK * kernelRows
		if remRows < kernelRows {
			clear(panel[stripBase : stripBase+paddedK*kernelRows])
		}
		for r := range remRows {
			srcRowBase := (lhsRowStart+stripRowIdx+r)*lhsCols + lhsColStart
			dstBase := stripBase + r*2
			for p := range fullKPairs {
				panel[dstBase] = lhs[srcRowBase+2*p]
				panel[dstBase+1] = lhs[srcRowBase+2*p+1]
				dstBase += kernelRows * 2
			}
			if hasOddK {
				panel[dstBase] = lhs[srcRowBase+2*fullKPairs]
				panel[dstBase+1] = 0
			}
		}
		_ = kPairs
	}
}

// avx512PackRHSNonTransposedBFloat16K2 packs [contractingRows, copyCols] from non-transposed rhs
// into K2-interleaved strips [ceil(copyCols/kernelCols), kPairs, kernelCols, 2].
// Any incomplete trailing strip columns or odd trailing K are zero-padded.
func avx512PackRHSNonTransposedBFloat16K2(
	rhs, panel []bfloat16.BFloat16,
	rhsRowStart, rhsColStart, rhsCols,
	contractingRows, copyCols, kernelCols int,
) {
	if contractingRows == 0 || copyCols == 0 {
		return
	}
	paddedK := (contractingRows + 1) & ^1
	fullKPairs := contractingRows >> 1
	hasOddK := (contractingRows & 1) != 0

	stripColIdx := 0
	if kernelCols == 32 {
		numFullStrips := copyCols / 32
		if numFullStrips > 0 {
			rhsBasePtr := unsafe.Pointer(unsafe.SliceData(rhs))
			panelBasePtr := unsafe.Pointer(unsafe.SliceData(panel))
			rhsStrideBytes := uintptr(rhsCols) * 2
			rhsStartPtr := unsafe.Add(rhsBasePtr, uintptr(rhsRowStart)*rhsStrideBytes+uintptr(rhsColStart)*2)
			avx512PackRHSNonTransposed32BFloat16K2Asm(
				rhsStartPtr,
				panelBasePtr,
				rhsStrideBytes,
				contractingRows,
				numFullStrips,
			)
			stripColIdx = numFullStrips * 32
		}
	}

	for ; stripColIdx < copyCols; stripColIdx += kernelCols {
		remCols := min(kernelCols, copyCols-stripColIdx)
		stripBase := (stripColIdx / kernelCols) * paddedK * kernelCols
		if remCols < kernelCols {
			clear(panel[stripBase : stripBase+paddedK*kernelCols])
		}
		for p := range fullKPairs {
			srcRow0 := (rhsRowStart+2*p)*rhsCols + rhsColStart + stripColIdx
			srcRow1 := srcRow0 + rhsCols
			dstPairBase := stripBase + p*kernelCols*2
			for c := range remCols {
				panel[dstPairBase+2*c] = rhs[srcRow0+c]
				panel[dstPairBase+2*c+1] = rhs[srcRow1+c]
			}
		}
		if hasOddK {
			srcRow0 := (rhsRowStart+2*fullKPairs)*rhsCols + rhsColStart + stripColIdx
			dstPairBase := stripBase + fullKPairs*kernelCols*2
			for c := range remCols {
				panel[dstPairBase+2*c] = rhs[srcRow0+c]
				panel[dstPairBase+2*c+1] = 0
			}
		}
	}
}

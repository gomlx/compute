// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes/gotype"
)

// packRHS packs a slice of size [contractingRows, rhsCols] block from RHS into
// the panel reshaped+transposed to [ceil(rhsCols/RHSL1KernelCols), contractingRows, RHSL1KernelCols],
// padding the cols of the last strip with zeros if necessary.
//
//   - src: [contractingSize, rhsCrossSize]
//   - dst: a slice with enough size to hold the panel
//   - srcRowStart: start row in src
//   - srcColStart: start col in src
//   - srcStrideCol: stride of src
//   - contractingRows: number of rows to be copied in the panel (must fit total panel allocated size)
//   - rhsCols: number of columns to be copied in the panel (excluding padding), will be padded to a RHSL1KernelCols
//     multiple with zeros.
//   - RHSL1KernelCols: number of columns in each "L1 kernel"
func packRHS[T gotype.ScalarNotComplex](src, dst []T, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols int) {
	dstIdx := 0
	// Iterate over strips of width nr
	for stripColIdx := 0; stripColIdx < rhsCols; stripColIdx += RHSL1KernelCols {
		// How many columns valid in this strip?
		validCols := min(RHSL1KernelCols, rhsCols-stripColIdx)
		srcIdxBase := (srcRowStart * srcStrideCol) + srcColStart + stripColIdx

		if validCols == RHSL1KernelCols {
			// Fast path: no zero padding needed
			for range contractingRows {
				// Copy valid columns
				copy(dst[dstIdx:], src[srcIdxBase:srcIdxBase+validCols])
				dstIdx += validCols
				srcIdxBase += srcStrideCol
			}
		} else {
			// Iterate over rows (k)
			for range contractingRows {
				// Copy valid columns
				copy(dst[dstIdx:], src[srcIdxBase:srcIdxBase+validCols])
				dstIdx += validCols
				srcIdxBase += srcStrideCol
				// Zero-pad if strip is incomplete (edge of matrix)
				for c := validCols; c < RHSL1KernelCols; c++ {
					dst[dstIdx] = T(0)
					dstIdx++
				}
			}
		}
	}
}

// unsafePackRHS is an optimized version of packRHS that eliminates bounds checks and slice overhead
// using unsafe pointers. For the common case where RHSL1KernelCols == 4, it performs direct 4-element
// loads and stores (which the Go compiler lowers to 128-bit vector moves or paired loads/stores).
func unsafePackRHS[T gotype.ScalarNotComplex](
	src, dst []T,
	srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols int) {
	if contractingRows == 0 || rhsCols == 0 {
		return
	}

	srcPtr := uintptr(unsafe.Pointer(&src[0]))
	dstPtr := uintptr(unsafe.Pointer(&dst[0]))
	elemSize := unsafe.Sizeof(T(0))
	srcStrideBytes := uintptr(srcStrideCol) * elemSize
	kernelColsBytes := uintptr(RHSL1KernelCols) * elemSize

	for stripColIdx := 0; stripColIdx < rhsCols; stripColIdx += RHSL1KernelCols {
		validCols := min(RHSL1KernelCols, rhsCols-stripColIdx)
		srcIdxBase := (srcRowStart * srcStrideCol) + srcColStart + stripColIdx
		pSrcRow := srcPtr + uintptr(srcIdxBase)*elemSize

		if validCols == RHSL1KernelCols {
			switch RHSL1KernelCols {
			case 4:
				for range contractingRows {
					*(*[4]T)(unsafe.Pointer(dstPtr)) = *(*[4]T)(unsafe.Pointer(pSrcRow))
					dstPtr += 4 * elemSize
					pSrcRow += srcStrideBytes
				}
			case 8:
				for range contractingRows {
					*(*[8]T)(unsafe.Pointer(dstPtr)) = *(*[8]T)(unsafe.Pointer(pSrcRow))
					dstPtr += 8 * elemSize
					pSrcRow += srcStrideBytes
				}
			case 16:
				for range contractingRows {
					*(*[16]T)(unsafe.Pointer(dstPtr)) = *(*[16]T)(unsafe.Pointer(pSrcRow))
					dstPtr += 16 * elemSize
					pSrcRow += srcStrideBytes
				}
			default:
				for range contractingRows {
					pSrc := pSrcRow
					pDst := dstPtr
					for range RHSL1KernelCols {
						*(*T)(unsafe.Pointer(pDst)) = *(*T)(unsafe.Pointer(pSrc))
						pSrc += elemSize
						pDst += elemSize
					}
					dstPtr += kernelColsBytes
					pSrcRow += srcStrideBytes
				}
			}
		} else {
			// Fringe strip with partial valid columns and zero-padding
			for range contractingRows {
				pSrc := pSrcRow
				pDst := dstPtr
				for range validCols {
					*(*T)(unsafe.Pointer(pDst)) = *(*T)(unsafe.Pointer(pSrc))
					pSrc += elemSize
					pDst += elemSize
				}
				for range RHSL1KernelCols - validCols {
					*(*T)(unsafe.Pointer(pDst)) = T(0)
					pDst += elemSize
				}
				dstPtr += kernelColsBytes
				pSrcRow += srcStrideBytes
			}
		}
	}
}

// packLHS packs a block of size [copyRows, contractingCols] from the lhs matrix into a panel.
// The panel is structured as [ceil(copyRows/kernelRows), contractingCols, kernelRows].
// It rearranges data into horizontal strips of height kernelRows.
//
// Notice, it can also be used to pack a RHS, if the RHS has a transposed layout
// (shaped [rhsCrossSize, contractingSize])
//
//   - lhs: matrix [lhsRows, lhsCols], where lhsRows >= lhsRowStart + copyRows.
//   - panel: packed panel with enough space to store the [numStrips, contractingCols, kernelRows].
//     Where numStrips = ceil(copyRows / kernelRows), the last strip padded with 0s.
//   - lhsRowStart, lhsColStart: start of the slice that will be packed into the panel.
//   - lhsCols: number of columns in the lhs matrix (row stride).
//   - copyRows: how many rows of lhs to copy to the panel.
//   - contractingCols: number of columns to copy to the panel.
//   - kernelRows: we are packing in strips of kernelRows size.
//     For this AVX512 implementation kernelRows must be a multiple of 4, it will panic otherwise.
func packLHS[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows int) {
	panelIdx := 0
	fullStripsRows := (copyRows / kernelRows) * kernelRows

	// Iterate over full strips of height kernelRows
	for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
		srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart

		for r := range kernelRows {
			srcIdx := srcIdxBase + r*lhsCols
			pIdx := panelIdx + r
			for col := range contractingCols {
				panel[pIdx] = lhs[srcIdx+col]
				pIdx += kernelRows
			}
		}
		panelIdx += contractingCols * kernelRows
	}

	// Last strip
	if fullStripsRows < copyRows {
		stripRowIdx := fullStripsRows
		validRows := copyRows - stripRowIdx
		srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart

		for r := range validRows {
			srcIdx := srcIdxBase + r*lhsCols
			pIdx := panelIdx + r
			for col := range contractingCols {
				panel[pIdx] = lhs[srcIdx+col]
				pIdx += kernelRows
			}
		}

		for r := validRows; r < kernelRows; r++ {
			pIdx := panelIdx + r
			for range contractingCols {
				panel[pIdx] = T(0)
				pIdx += kernelRows
			}
		}
		panelIdx += contractingCols * kernelRows
	}
}

// unsafePackLHS is identical to packLHS but eliminates boundary checks by using unsafe pointers.
// This has a 10% improvement gain over packLHS.
func unsafePackLHS[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows int) {
	if copyRows == 0 || contractingCols == 0 {
		return
	}

	panelPtr := uintptr(unsafe.Pointer(&panel[0]))
	lhsPtr := uintptr(unsafe.Pointer(&lhs[0]))
	elemSize := unsafe.Sizeof(T(0))
	kernelRowsBytes := uintptr(kernelRows) * elemSize
	lhsColsBytes := uintptr(lhsCols) * elemSize
	lhsColsBytes4 := 4 * lhsColsBytes
	elemSize4 := 4 * elemSize
	stripSizeBytes := uintptr(contractingCols) * kernelRowsBytes

	fullStripsRows := (copyRows / kernelRows) * kernelRows

	// Iterate over full strips of height kernelRows
	switch {
	case kernelRows == 2:
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pSrcBase := lhsPtr + uintptr(srcIdxBase)*elemSize

			pSrc0 := pSrcBase
			pSrc1 := pSrcBase + lhsColsBytes
			pDst := panelPtr

			col := 0
			for ; col+1 < contractingCols; col += 2 {
				v0 := *(*T)(unsafe.Pointer(pSrc0))
				v1 := *(*T)(unsafe.Pointer(pSrc1))
				*(*[2]T)(unsafe.Pointer(pDst)) = [2]T{v0, v1}

				v0_1 := *(*T)(unsafe.Pointer(pSrc0 + elemSize))
				v1_1 := *(*T)(unsafe.Pointer(pSrc1 + elemSize))
				*(*[2]T)(unsafe.Pointer(pDst + 2*elemSize)) = [2]T{v0_1, v1_1}

				pSrc0 += 2 * elemSize
				pSrc1 += 2 * elemSize
				pDst += 4 * elemSize
			}
			for ; col < contractingCols; col++ {
				v0 := *(*T)(unsafe.Pointer(pSrc0))
				v1 := *(*T)(unsafe.Pointer(pSrc1))
				*(*[2]T)(unsafe.Pointer(pDst)) = [2]T{v0, v1}

				pSrc0 += elemSize
				pSrc1 += elemSize
				pDst += 2 * elemSize
			}
			panelPtr += stripSizeBytes
		}
	case kernelRows == 4:
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pSrcBase := lhsPtr + uintptr(srcIdxBase)*elemSize

			pSrc0 := pSrcBase
			pSrc1 := pSrc0 + lhsColsBytes
			pSrc2 := pSrc1 + lhsColsBytes
			pSrc3 := pSrc2 + lhsColsBytes

			pDst := panelPtr

			col := 0
			for ; col+1 < contractingCols; col += 2 {
				v0 := *(*T)(unsafe.Pointer(pSrc0))
				v1 := *(*T)(unsafe.Pointer(pSrc1))
				v2 := *(*T)(unsafe.Pointer(pSrc2))
				v3 := *(*T)(unsafe.Pointer(pSrc3))
				*(*[4]T)(unsafe.Pointer(pDst)) = [4]T{v0, v1, v2, v3}

				v0_1 := *(*T)(unsafe.Pointer(pSrc0 + elemSize))
				v1_1 := *(*T)(unsafe.Pointer(pSrc1 + elemSize))
				v2_1 := *(*T)(unsafe.Pointer(pSrc2 + elemSize))
				v3_1 := *(*T)(unsafe.Pointer(pSrc3 + elemSize))
				*(*[4]T)(unsafe.Pointer(pDst + 4*elemSize)) = [4]T{v0_1, v1_1, v2_1, v3_1}

				pSrc0 += 2 * elemSize
				pSrc1 += 2 * elemSize
				pSrc2 += 2 * elemSize
				pSrc3 += 2 * elemSize
				pDst += 8 * elemSize
			}
			for ; col < contractingCols; col++ {
				v0 := *(*T)(unsafe.Pointer(pSrc0))
				v1 := *(*T)(unsafe.Pointer(pSrc1))
				v2 := *(*T)(unsafe.Pointer(pSrc2))
				v3 := *(*T)(unsafe.Pointer(pSrc3))
				*(*[4]T)(unsafe.Pointer(pDst)) = [4]T{v0, v1, v2, v3}

				pSrc0 += elemSize
				pSrc1 += elemSize
				pSrc2 += elemSize
				pSrc3 += elemSize
				pDst += 4 * elemSize
			}
			panelPtr += stripSizeBytes
		}

	default:
		// Larger values of kernelRows must be multiple of 4.
		if kernelRows%4 != 0 {
			panic("kernelRows must be a multiple of 4")
		}
		// The kernelRow is multiple of 4, but > 4.
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pSrcBase := lhsPtr + uintptr(srcIdxBase)*elemSize
			pDstBase := panelPtr

			for r := 0; r < kernelRows; r += 4 {
				pSrc0 := pSrcBase
				pSrc1 := pSrc0 + lhsColsBytes
				pSrc2 := pSrc1 + lhsColsBytes
				pSrc3 := pSrc2 + lhsColsBytes

				pDst0 := pDstBase
				pDst1 := pDst0 + elemSize
				pDst2 := pDst1 + elemSize
				pDst3 := pDst2 + elemSize

				for range contractingCols {
					*(*T)(unsafe.Pointer(pDst0)) = *(*T)(unsafe.Pointer(pSrc0))
					*(*T)(unsafe.Pointer(pDst1)) = *(*T)(unsafe.Pointer(pSrc1))
					*(*T)(unsafe.Pointer(pDst2)) = *(*T)(unsafe.Pointer(pSrc2))
					*(*T)(unsafe.Pointer(pDst3)) = *(*T)(unsafe.Pointer(pSrc3))

					pSrc0 += elemSize
					pSrc1 += elemSize
					pSrc2 += elemSize
					pSrc3 += elemSize

					pDst0 += kernelRowsBytes
					pDst1 += kernelRowsBytes
					pDst2 += kernelRowsBytes
					pDst3 += kernelRowsBytes
				}
				pSrcBase += lhsColsBytes4
				pDstBase += elemSize4
			}
			panelPtr += stripSizeBytes
		}
	}

	// Last strip
	if fullStripsRows < copyRows {
		stripRowIdx := fullStripsRows
		validRows := copyRows - stripRowIdx
		srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
		pSrcBase := lhsPtr + uintptr(srcIdxBase)*elemSize
		pDstBase := panelPtr

		for range validRows {
			pSrc := pSrcBase
			pDst := pDstBase

			for range contractingCols {
				*(*T)(unsafe.Pointer(pDst)) = *(*T)(unsafe.Pointer(pSrc))
				pSrc += elemSize
				pDst += kernelRowsBytes
			}
			pSrcBase += lhsColsBytes
			pDstBase += elemSize
		}

		for r := validRows; r < kernelRows; r++ {
			pDst := pDstBase
			for range contractingCols {
				*(*T)(unsafe.Pointer(pDst)) = T(0)
				pDst += kernelRowsBytes
			}
			pDstBase += elemSize
		}
		panelPtr += uintptr(contractingCols) * kernelRowsBytes
	}
}

// PackLHS packs a block of size [copyRows, contractingCols] from the lhs matrix into a panel.
func PackLHS[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows int) {
	packLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
}

// PackRHS packs a slice of size [contractingRows, rhsCols] block from RHS into the panel.
func PackRHS[T gotype.ScalarNotComplex](
	src, dst []T, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols int) {
	packRHS(src, dst, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols)
}

// UnsafePackLHS is identical to PackLHS but eliminates boundary checks by using unsafe pointers.
func UnsafePackLHS[T gotype.ScalarNotComplex](
	lhs, panel []T,
	lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows int) {
	unsafePackLHS(lhs, panel, lhsRowStart, lhsColStart, lhsCols, copyRows, contractingCols, kernelRows)
}

// UnsafePackRHS is an optimized version of PackRHS that eliminates bounds checks and slice overhead.
func UnsafePackRHS[T gotype.ScalarNotComplex](
	src, dst []T, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols int) {
	unsafePackRHS(src, dst, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols)
}


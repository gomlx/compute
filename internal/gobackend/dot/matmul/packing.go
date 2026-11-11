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

	srcPtr := unsafe.Pointer(unsafe.SliceData(src))
	dstBasePtr := unsafe.Pointer(unsafe.SliceData(dst))
	elemSize := unsafe.Sizeof(T(0))
	srcStrideBytes := uintptr(srcStrideCol) * elemSize
	kernelColsBytes := uintptr(RHSL1KernelCols) * elemSize
	dstOffset := uintptr(0)

	for stripColIdx := 0; stripColIdx < rhsCols; stripColIdx += RHSL1KernelCols {
		validCols := min(RHSL1KernelCols, rhsCols-stripColIdx)
		srcIdxBase := (srcRowStart * srcStrideCol) + srcColStart + stripColIdx
		pSrcRow := unsafe.Add(srcPtr, uintptr(srcIdxBase)*elemSize)
		pDstRow := unsafe.Add(dstBasePtr, dstOffset)

		if validCols == RHSL1KernelCols {
			switch RHSL1KernelCols {
			case 4:
				for k := 0; ; {
					*(*[4]T)(pDstRow) = *(*[4]T)(pSrcRow)
					if k++; k >= contractingRows {
						break
					}
					pDstRow = unsafe.Add(pDstRow, 4*elemSize)
					pSrcRow = unsafe.Add(pSrcRow, srcStrideBytes)
				}
			case 8:
				for k := 0; ; {
					*(*[8]T)(pDstRow) = *(*[8]T)(pSrcRow)
					if k++; k >= contractingRows {
						break
					}
					pDstRow = unsafe.Add(pDstRow, 8*elemSize)
					pSrcRow = unsafe.Add(pSrcRow, srcStrideBytes)
				}
			case 16:
				for k := 0; ; {
					*(*[16]T)(pDstRow) = *(*[16]T)(pSrcRow)
					if k++; k >= contractingRows {
						break
					}
					pDstRow = unsafe.Add(pDstRow, 16*elemSize)
					pSrcRow = unsafe.Add(pSrcRow, srcStrideBytes)
				}
			default:
				for k := 0; ; {
					for c := range RHSL1KernelCols {
						off := uintptr(c) * elemSize
						*(*T)(unsafe.Add(pDstRow, off)) = *(*T)(unsafe.Add(pSrcRow, off))
					}
					if k++; k >= contractingRows {
						break
					}
					pDstRow = unsafe.Add(pDstRow, kernelColsBytes)
					pSrcRow = unsafe.Add(pSrcRow, srcStrideBytes)
				}
			}
		} else {
			// Fringe strip with partial valid columns and zero-padding
			for k := 0; ; {
				for c := range validCols {
					off := uintptr(c) * elemSize
					*(*T)(unsafe.Add(pDstRow, off)) = *(*T)(unsafe.Add(pSrcRow, off))
				}
				for c := validCols; c < RHSL1KernelCols; c++ {
					*(*T)(unsafe.Add(pDstRow, uintptr(c)*elemSize)) = T(0)
				}
				if k++; k >= contractingRows {
					break
				}
				pDstRow = unsafe.Add(pDstRow, kernelColsBytes)
				pSrcRow = unsafe.Add(pSrcRow, srcStrideBytes)
			}
		}
		dstOffset += uintptr(contractingRows) * kernelColsBytes
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

	panelBasePtr := unsafe.Pointer(unsafe.SliceData(panel))
	lhsPtr := unsafe.Pointer(unsafe.SliceData(lhs))
	elemSize := unsafe.Sizeof(T(0))
	kernelRowsBytes := uintptr(kernelRows) * elemSize
	lhsColsBytes := uintptr(lhsCols) * elemSize
	stripSizeBytes := uintptr(contractingCols) * kernelRowsBytes
	panelOffsetBytes := uintptr(0)

	fullStripsRows := (copyRows / kernelRows) * kernelRows

	// Iterate over full strips of height kernelRows
	switch {
	case kernelRows == 2:
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pSrc0 := unsafe.Add(lhsPtr, uintptr(srcIdxBase)*elemSize)
			pSrc1 := unsafe.Add(pSrc0, lhsColsBytes)
			pDstBase := unsafe.Add(panelBasePtr, panelOffsetBytes)

			col := 0
			for ; col+1 < contractingCols; col += 2 {
				srcOff := uintptr(col) * elemSize
				dstOff := uintptr(col) * (2 * elemSize)
				v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
				v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
				*(*[2]T)(unsafe.Add(pDstBase, dstOff)) = [2]T{v0, v1}

				srcOff1 := srcOff + elemSize
				v0_1 := *(*T)(unsafe.Add(pSrc0, srcOff1))
				v1_1 := *(*T)(unsafe.Add(pSrc1, srcOff1))
				*(*[2]T)(unsafe.Add(pDstBase, dstOff+2*elemSize)) = [2]T{v0_1, v1_1}
			}
			for ; col < contractingCols; col++ {
				srcOff := uintptr(col) * elemSize
				dstOff := uintptr(col) * (2 * elemSize)
				v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
				v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
				*(*[2]T)(unsafe.Add(pDstBase, dstOff)) = [2]T{v0, v1}
			}
			panelOffsetBytes += stripSizeBytes
		}
	case kernelRows == 4:
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pSrc0 := unsafe.Add(lhsPtr, uintptr(srcIdxBase)*elemSize)
			pSrc1 := unsafe.Add(pSrc0, lhsColsBytes)
			pSrc2 := unsafe.Add(pSrc1, lhsColsBytes)
			pSrc3 := unsafe.Add(pSrc2, lhsColsBytes)
			pDstBase := unsafe.Add(panelBasePtr, panelOffsetBytes)

			col := 0
			for ; col+1 < contractingCols; col += 2 {
				srcOff := uintptr(col) * elemSize
				dstOff := uintptr(col) * (4 * elemSize)
				v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
				v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
				v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
				v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
				*(*[4]T)(unsafe.Add(pDstBase, dstOff)) = [4]T{v0, v1, v2, v3}

				srcOff1 := srcOff + elemSize
				v0_1 := *(*T)(unsafe.Add(pSrc0, srcOff1))
				v1_1 := *(*T)(unsafe.Add(pSrc1, srcOff1))
				v2_1 := *(*T)(unsafe.Add(pSrc2, srcOff1))
				v3_1 := *(*T)(unsafe.Add(pSrc3, srcOff1))
				*(*[4]T)(unsafe.Add(pDstBase, dstOff+4*elemSize)) = [4]T{v0_1, v1_1, v2_1, v3_1}
			}
			for ; col < contractingCols; col++ {
				srcOff := uintptr(col) * elemSize
				dstOff := uintptr(col) * (4 * elemSize)
				v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
				v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
				v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
				v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
				*(*[4]T)(unsafe.Add(pDstBase, dstOff)) = [4]T{v0, v1, v2, v3}
			}
			panelOffsetBytes += stripSizeBytes
		}
	case kernelRows == 8:
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pSrc0 := unsafe.Add(lhsPtr, uintptr(srcIdxBase)*elemSize)
			pSrc1 := unsafe.Add(pSrc0, lhsColsBytes)
			pSrc2 := unsafe.Add(pSrc1, lhsColsBytes)
			pSrc3 := unsafe.Add(pSrc2, lhsColsBytes)
			pSrc4 := unsafe.Add(pSrc3, lhsColsBytes)
			pSrc5 := unsafe.Add(pSrc4, lhsColsBytes)
			pSrc6 := unsafe.Add(pSrc5, lhsColsBytes)
			pSrc7 := unsafe.Add(pSrc6, lhsColsBytes)
			pDstBase := unsafe.Add(panelBasePtr, panelOffsetBytes)

			col := 0
			for ; col+1 < contractingCols; col += 2 {
				srcOff := uintptr(col) * elemSize
				dstOff := uintptr(col) * (8 * elemSize)
				v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
				v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
				v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
				v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
				v4 := *(*T)(unsafe.Add(pSrc4, srcOff))
				v5 := *(*T)(unsafe.Add(pSrc5, srcOff))
				v6 := *(*T)(unsafe.Add(pSrc6, srcOff))
				v7 := *(*T)(unsafe.Add(pSrc7, srcOff))
				*(*[8]T)(unsafe.Add(pDstBase, dstOff)) = [8]T{v0, v1, v2, v3, v4, v5, v6, v7}

				srcOff1 := srcOff + elemSize
				v0_1 := *(*T)(unsafe.Add(pSrc0, srcOff1))
				v1_1 := *(*T)(unsafe.Add(pSrc1, srcOff1))
				v2_1 := *(*T)(unsafe.Add(pSrc2, srcOff1))
				v3_1 := *(*T)(unsafe.Add(pSrc3, srcOff1))
				v4_1 := *(*T)(unsafe.Add(pSrc4, srcOff1))
				v5_1 := *(*T)(unsafe.Add(pSrc5, srcOff1))
				v6_1 := *(*T)(unsafe.Add(pSrc6, srcOff1))
				v7_1 := *(*T)(unsafe.Add(pSrc7, srcOff1))
				*(*[8]T)(unsafe.Add(pDstBase, dstOff+8*elemSize)) = [8]T{v0_1, v1_1, v2_1, v3_1, v4_1, v5_1, v6_1, v7_1}
			}
			for ; col < contractingCols; col++ {
				srcOff := uintptr(col) * elemSize
				dstOff := uintptr(col) * (8 * elemSize)
				v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
				v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
				v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
				v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
				v4 := *(*T)(unsafe.Add(pSrc4, srcOff))
				v5 := *(*T)(unsafe.Add(pSrc5, srcOff))
				v6 := *(*T)(unsafe.Add(pSrc6, srcOff))
				v7 := *(*T)(unsafe.Add(pSrc7, srcOff))
				*(*[8]T)(unsafe.Add(pDstBase, dstOff)) = [8]T{v0, v1, v2, v3, v4, v5, v6, v7}
			}
			panelOffsetBytes += stripSizeBytes
		}

	case kernelRows%8 == 0:
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pStripSrc := unsafe.Add(lhsPtr, uintptr(srcIdxBase)*elemSize)
			pStripDst := unsafe.Add(panelBasePtr, panelOffsetBytes)

			for r := 0; r < kernelRows; r += 8 {
				pSrc0 := unsafe.Add(pStripSrc, uintptr(r)*lhsColsBytes)
				pSrc1 := unsafe.Add(pSrc0, lhsColsBytes)
				pSrc2 := unsafe.Add(pSrc1, lhsColsBytes)
				pSrc3 := unsafe.Add(pSrc2, lhsColsBytes)
				pSrc4 := unsafe.Add(pSrc3, lhsColsBytes)
				pSrc5 := unsafe.Add(pSrc4, lhsColsBytes)
				pSrc6 := unsafe.Add(pSrc5, lhsColsBytes)
				pSrc7 := unsafe.Add(pSrc6, lhsColsBytes)
				pDstBase := unsafe.Add(pStripDst, uintptr(r)*elemSize)

				col := 0
				for ; col+1 < contractingCols; col += 2 {
					srcOff := uintptr(col) * elemSize
					dstOff := uintptr(col) * kernelRowsBytes
					v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
					v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
					v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
					v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
					v4 := *(*T)(unsafe.Add(pSrc4, srcOff))
					v5 := *(*T)(unsafe.Add(pSrc5, srcOff))
					v6 := *(*T)(unsafe.Add(pSrc6, srcOff))
					v7 := *(*T)(unsafe.Add(pSrc7, srcOff))
					*(*[8]T)(unsafe.Add(pDstBase, dstOff)) = [8]T{v0, v1, v2, v3, v4, v5, v6, v7}

					srcOff1 := srcOff + elemSize
					v0_1 := *(*T)(unsafe.Add(pSrc0, srcOff1))
					v1_1 := *(*T)(unsafe.Add(pSrc1, srcOff1))
					v2_1 := *(*T)(unsafe.Add(pSrc2, srcOff1))
					v3_1 := *(*T)(unsafe.Add(pSrc3, srcOff1))
					v4_1 := *(*T)(unsafe.Add(pSrc4, srcOff1))
					v5_1 := *(*T)(unsafe.Add(pSrc5, srcOff1))
					v6_1 := *(*T)(unsafe.Add(pSrc6, srcOff1))
					v7_1 := *(*T)(unsafe.Add(pSrc7, srcOff1))
					*(*[8]T)(unsafe.Add(pDstBase, dstOff+kernelRowsBytes)) = [8]T{v0_1, v1_1, v2_1, v3_1, v4_1, v5_1, v6_1, v7_1}
				}
				for ; col < contractingCols; col++ {
					srcOff := uintptr(col) * elemSize
					dstOff := uintptr(col) * kernelRowsBytes
					v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
					v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
					v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
					v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
					v4 := *(*T)(unsafe.Add(pSrc4, srcOff))
					v5 := *(*T)(unsafe.Add(pSrc5, srcOff))
					v6 := *(*T)(unsafe.Add(pSrc6, srcOff))
					v7 := *(*T)(unsafe.Add(pSrc7, srcOff))
					*(*[8]T)(unsafe.Add(pDstBase, dstOff)) = [8]T{v0, v1, v2, v3, v4, v5, v6, v7}
				}
			}
			panelOffsetBytes += stripSizeBytes
		}

	default:
		// Larger values of kernelRows must be multiple of 4.
		if kernelRows%4 != 0 {
			panic("kernelRows must be a multiple of 4")
		}
		// The kernelRow is multiple of 4, but > 4.
		for stripRowIdx := 0; stripRowIdx < fullStripsRows; stripRowIdx += kernelRows {
			srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
			pStripSrc := unsafe.Add(lhsPtr, uintptr(srcIdxBase)*elemSize)
			pStripDst := unsafe.Add(panelBasePtr, panelOffsetBytes)

			for r := 0; r < kernelRows; r += 4 {
				pSrc0 := unsafe.Add(pStripSrc, uintptr(r)*lhsColsBytes)
				pSrc1 := unsafe.Add(pSrc0, lhsColsBytes)
				pSrc2 := unsafe.Add(pSrc1, lhsColsBytes)
				pSrc3 := unsafe.Add(pSrc2, lhsColsBytes)
				pDstBase := unsafe.Add(pStripDst, uintptr(r)*elemSize)

				col := 0
				for ; col+1 < contractingCols; col += 2 {
					srcOff := uintptr(col) * elemSize
					dstOff := uintptr(col) * kernelRowsBytes
					v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
					v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
					v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
					v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
					*(*[4]T)(unsafe.Add(pDstBase, dstOff)) = [4]T{v0, v1, v2, v3}

					srcOff1 := srcOff + elemSize
					v0_1 := *(*T)(unsafe.Add(pSrc0, srcOff1))
					v1_1 := *(*T)(unsafe.Add(pSrc1, srcOff1))
					v2_1 := *(*T)(unsafe.Add(pSrc2, srcOff1))
					v3_1 := *(*T)(unsafe.Add(pSrc3, srcOff1))
					*(*[4]T)(unsafe.Add(pDstBase, dstOff+kernelRowsBytes)) = [4]T{v0_1, v1_1, v2_1, v3_1}
				}
				for ; col < contractingCols; col++ {
					srcOff := uintptr(col) * elemSize
					dstOff := uintptr(col) * kernelRowsBytes
					v0 := *(*T)(unsafe.Add(pSrc0, srcOff))
					v1 := *(*T)(unsafe.Add(pSrc1, srcOff))
					v2 := *(*T)(unsafe.Add(pSrc2, srcOff))
					v3 := *(*T)(unsafe.Add(pSrc3, srcOff))
					*(*[4]T)(unsafe.Add(pDstBase, dstOff)) = [4]T{v0, v1, v2, v3}
				}
			}
			panelOffsetBytes += stripSizeBytes
		}
	}

	// Last strip
	if fullStripsRows < copyRows {
		stripRowIdx := fullStripsRows
		validRows := copyRows - stripRowIdx
		srcIdxBase := ((lhsRowStart + stripRowIdx) * lhsCols) + lhsColStart
		pStripSrc := unsafe.Add(lhsPtr, uintptr(srcIdxBase)*elemSize)
		pStripDst := unsafe.Add(panelBasePtr, panelOffsetBytes)

		for r := range validRows {
			pSrcRow := unsafe.Add(pStripSrc, uintptr(r)*lhsColsBytes)
			pDstRow := unsafe.Add(pStripDst, uintptr(r)*elemSize)

			for col := range contractingCols {
				*(*T)(unsafe.Add(pDstRow, uintptr(col)*kernelRowsBytes)) = *(*T)(unsafe.Add(pSrcRow, uintptr(col)*elemSize))
			}
		}

		for r := validRows; r < kernelRows; r++ {
			pDstRow := unsafe.Add(pStripDst, uintptr(r)*elemSize)
			for col := range contractingCols {
				*(*T)(unsafe.Add(pDstRow, uintptr(col)*kernelRowsBytes)) = T(0)
			}
		}
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


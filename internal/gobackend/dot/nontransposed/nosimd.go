// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nontransposed

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/envutil"
	"k8s.io/klog/v2"
)

var (
	// NoSIMD32Params are generic assumptions for L1/L2/L3 cache sizes for 32 bits dtypes (float32, int32, uint32)
	//
	// These values are somewhat arbitrary, assuming "standard" modern cache sizes.
	// They are parameterized so they can be tuned or determined dynamically later.
	NoSIMD32Params = CacheParams{
		// Do not change these 2 values: they are hard-coded by the allocated registers in basicSymmetricMicroKernel8x8.
		LHSL1KernelRows: 2, // Mr: Rows of LHS in local registers.
		RHSL1KernelCols: 4, // Nr: Cols of RHS in local registers.

		PanelContractingSize: 512, // Kc: L1 Block contracting "depth".
		LHSPanelCrossSize:    2,   // Mc: Block Height fitting L2/L3 cache.
		RHSPanelCrossSize:    512, // Nc: Block Width fitting L2/L3 cache.
	}

	// Threshold in byte size for switching to the small matrix multiplication kernel.
	// If the total number of operations is below this threshold, the small
	// matrix multiplication kernel is used instead of the tiled implementation.
	// This is a heuristic and may need to be tuned for different architectures.
	// Expressed in number of bytes.
	noSIMDSmallMatMulSizeThreshold = 4 * 1024 * 1024

	// Minimum number of flops per worker: above this number, if possible we should
	// parallelize computation on separate goroutines.
	noSIMDMinMatMulFlopsPerWorker = 32 * 1024
)

func init() {
	enabled := envutil.MustReadBool("GOMLX_DOT_NOSIMD", true)
	if enabled {
		registerNoSIMD(false)
	} else {
		klog.Info("NoSIMD disabled")
	}
}

func RegisterNoSIMDForTests() {
	registerNoSIMD(true)
}

func registerNoSIMD(forTests bool) {
	// DTypePairMap: callImplementationDTypePairMap (ints, same)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int16, noSIMDRouter[int16, int16], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int8, noSIMDRouter[int8, int8], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (ints, int32,int64)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int32, noSIMDRouter[int16, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int64, noSIMDRouter[int16, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int64, noSIMDRouter[int32, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int32, noSIMDRouter[int64, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int32, noSIMDRouter[int8, int32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int64, noSIMDRouter[int8, int64], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (uints, same)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint16, noSIMDRouter[uint16, uint16], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint8, noSIMDRouter[uint8, uint8], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (uints, uint32,uint64)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint32, noSIMDRouter[uint16, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint64, noSIMDRouter[uint16, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint64, noSIMDRouter[uint32, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint32, noSIMDRouter[uint64, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint32, noSIMDRouter[uint8, uint32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint64, noSIMDRouter[uint8, uint64], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (floats, floats)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, noSIMDRouter[float32, float32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float64, noSIMDRouter[float32, float64], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float32, noSIMDRouter[float64, float32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, noSIMDRouter[float64, float64], gobackend.PriorityTyped, forTests)

	// DTypePairMap: callImplementationDTypePairMap (half, float32)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, noSIMDHalfPrecisionRouter[bfloat16.BFloat16, float32], gobackend.PriorityTyped, forTests)
	dot.RegisterImplementation("no-simd", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, noSIMDHalfPrecisionRouter[float16.Float16, float32], gobackend.PriorityTyped, forTests)
}

// Auto-generate alternate specialized versions of noSIMD operations -- for half-precision input data types.
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_router.go -tags=half
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_small.go -tags=half
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_large.go -tags=half

// GetBuffer simplifies the process of getting a buffer from a backend and getting the flat slice.
func GetBuffer[T dtypes.Supported](backend *gobackend.Backend, length int) (ref *gobackend.Buffer, flat []T, success bool) {
	var err error
	ref, err = backend.GetBuffer(shapes.Make(dtypes.FromGenericsType[T](), length))
	if err != nil {
		klog.Errorf("Failed to allocate buffer for DotGeneral, undefined values are returned: %+v", err)
		return nil, nil, false
	}
	flat = ref.Flat.([]T)
	success = true
	return
}

// ReleaseBuffer releases a buffer obtained through GetBuffer.
func ReleaseBuffer(ref *gobackend.Buffer) {
	ref.Backend().(*gobackend.Backend).PutBuffer(ref)
}

// workItem is used when parallelizing the DotGeneral: it allows spliting the work into batch/lhs/rhs slices.
type workItem struct {
	batchStart, batchEnd,
	lhsRowStart, lhsRowEnd,
	rhsColStart, rhsColEnd int
}

// feedWorkItems split the matrix-multiplication tasks is "workItems" optimized (as large as possible, prioritizing whole batch items)
// for maxWokers (>=1).
// It closes workChan on exit.
//
// feedWorkItems is typically called on a separate goroutine, and it uses almost no CPU.
func feedWorkItems(
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *CacheParams,
	maxWorkers int,
	workChan chan<- workItem) {
	defer func() {
		// Invariant: it closes the channel on exit.
		close(workChan)
	}()
	if batchSize >= 2*maxWorkers {
		// Split the work on the batch dimension only.
		batchStep := batchSize / maxWorkers
		for batchIdx := 0; batchIdx < batchSize; batchIdx += batchStep {
			workChan <- workItem{
				batchIdx, batchIdx + min(batchStep, batchSize-batchIdx),
				0, lhsCrossSize,
				0, rhsCrossSize}
		}
		return
	}

	// First maxWorkers batch examples are handled as one at a time:
	batchIdx := 0
	if batchSize >= maxWorkers {
		for ; batchIdx < maxWorkers; batchIdx++ {
			workChan <- workItem{
				batchIdx, 1,
				0, lhsCrossSize,
				0, rhsCrossSize}
		}
	}

	// The remaining work is split into RHS or LHS slices.
	batchCountRemaining := batchSize - batchIdx
	if batchCountRemaining == 0 {
		return // We are finished.
	}
	splitFactor := (maxWorkers + batchCountRemaining - 1) / batchCountRemaining
	if lhsCrossSize > rhsCrossSize {
		// Split on the LHS dimension, in multiples of LHSPanelCrossSize.
		lhsSplitSize := (lhsCrossSize + splitFactor - 1) / splitFactor
		lhsSplitSize = max(1, lhsSplitSize/params.LHSPanelCrossSize) * params.LHSPanelCrossSize
		batchStart := batchIdx
		for lhsRowIdx := 0; lhsRowIdx < lhsCrossSize; lhsRowIdx += lhsSplitSize {
			for batchIdx = batchStart; batchIdx < batchSize; batchIdx++ {
				workChan <- workItem{
					batchIdx, batchIdx + 1,
					lhsRowIdx, lhsRowIdx + min(lhsSplitSize, lhsCrossSize-lhsRowIdx),
					0, rhsCrossSize}
			}
		}
	} else {
		// Split on the RHS dimension, in multiples of RHSPanelCrossSize.
		rhsSplitSize := (rhsCrossSize + splitFactor - 1) / splitFactor
		rhsSplitSize = max(1, rhsSplitSize/params.RHSPanelCrossSize) * params.RHSPanelCrossSize
		batchStart := batchIdx
		for rhsColIdx := 0; rhsColIdx < rhsCrossSize; rhsColIdx += rhsSplitSize {
			for batchIdx = batchStart; batchIdx < batchSize; batchIdx++ {
				workChan <- workItem{
					batchIdx, batchIdx + 1,
					0, lhsCrossSize,
					rhsColIdx, rhsColIdx + min(rhsSplitSize, rhsCrossSize-rhsColIdx)}
			}
		}
	}
}

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
func packRHS[T interface {
	dtypes.NumberNotComplex | dtypes.NumberHalfPrecision
}](src, dst []T, srcRowStart, srcColStart, srcStrideCol, contractingRows, rhsCols, RHSL1KernelCols int) {
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

// packLHS packs a slice of size [lhsRows, contractingCols] block from LHS into
// a [ceil(lhsRows/lhsL1KernelRows), contractingCols, lhsL1KernelRows] "panel"
// (a block of size Mr x Kc) from LHS.
// It rearranges data into horizontal strips of height Mr (lhsL1BlockRows).
//
// How it is called:
//
//	packLHS(lhs, packedLhs, lhsPanelRowIdx, contractingPanelIdx, contractingSize,
//		lhsPanelHeight, contractingPanelWidth,
//		params.LHSL1KernelRows)
func packLHS[T interface {
	dtypes.NumberNotComplex | dtypes.NumberHalfPrecision
}](src, dst []T, srcRowStart, srcColStart, srcRowStride, lhsRows, contractingCols, lhsL1KernelRows int) {
	dstIdx := 0
	// Iterate over strips of height mr
	for stripRowIdx := 0; stripRowIdx < lhsRows; stripRowIdx += lhsL1KernelRows {
		validRows := min(lhsL1KernelRows, lhsRows-stripRowIdx)
		srcIdxBase := ((srcRowStart + stripRowIdx) * srcRowStride) + srcColStart

		if validRows == lhsL1KernelRows {
			// Iterate over columns (contracting size k), we want LHS to be traversed K-first in the kernel
			for col := range contractingCols {
				srcIdx := srcIdxBase + col

				// Copy valid "rows" (they are the last axis in the returned panel)
				for range validRows {
					dst[dstIdx] = src[srcIdx]
					dstIdx++
					srcIdx += srcRowStride
				}
			}
		} else {
			// Iterate over columns (contracting size k), we want LHS to be traversed K-first in the kernel
			for col := range contractingCols {
				srcIdx := srcIdxBase + col

				// Copy valid "rows" (they are the last axis in the returned panel)
				for range validRows {
					dst[dstIdx] = src[srcIdx]
					dstIdx++
					srcIdx += srcRowStride
				}

				// Zero-pad
				for r := validRows; r < lhsL1KernelRows; r++ {
					dst[dstIdx] = T(0)
					dstIdx++
				}
			}
		}
	}
}

// applyPackedOutput applies the computed packedOutput to the final output.
func noSIMDApplyPackedOutput[T dtypes.NumberNotComplex](
	packedOutput, output []T,
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
			copy(output[outputRowIdx:outputRowIdx+width], packedOutput[packedRowIdx:packedRowIdx+width])
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	} else {
		// Not the first contracting panel, so we need to add to the existing values.
		for range height {
			packedSlice := packedOutput[packedRowIdx : packedRowIdx+width]
			outputSlice := output[outputRowIdx : outputRowIdx+width]
			for i, val := range packedSlice {
				outputSlice[i] += val
			}
			packedRowIdx += packedOutputRowStride
			outputRowIdx += outputRowStride
		}
	}
}

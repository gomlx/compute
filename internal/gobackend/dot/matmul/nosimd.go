// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/envutil"
	"k8s.io/klog/v2"
)

// Auto-generate alternate specialized versions of noSIMD operations -- for half-precision input data types.
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_router.go -tags=half
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_small.go -tags=half
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_small_transposed.go -tags=half
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_small_safe.go -tags=half
//go:generate go run ../../../cmd/alternates_generator -base=nosimd_large.go -tags=half

var (
	// NoSIMDParams are generic assumptions for L1/L2/L3 cache sizes -- it is optimized for 32 bits dtypes
	// but for now we use it for every type.
	//
	// These values are somewhat arbitrary, assuming "standard" modern cache sizes.
	// They are parameterized so they can be tuned or determined dynamically later.
	NoSIMDParams = CacheParams{
		LHSL1KernelRows: 2, // Mr: Rows of LHS in local registers (2).
		RHSL1KernelCols: 4, // Nr: Cols of RHS in local registers (4).

		PanelContractingSize: 512, // Kc: L1 Block contracting "depth".
		LHSPanelCrossSize:    2,   // Mc: Block Height fitting L2/L3 cache.
		RHSPanelCrossSize:    512, // Nc: Block Width fitting L2/L3 cache.
	}

	// SmallMatMulSizeThreshold is the threshold in byte size for switching to the small matrix multiplication kernel.
	// If the total number of operations is below this threshold, the small
	// matrix multiplication kernel is used instead of the tiled implementation.
	// This is a heuristic and may need to be tuned for different architectures.
	// Expressed in number of bytes.
	SmallMatMulSizeThreshold = 4 * 1024 * 1024
	noSIMDSmallMatMulSizeThreshold = SmallMatMulSizeThreshold

	// MinMatMulFlopsPerWorker is the minimum number of flops per worker: above this number, if possible we should
	// parallelize computation on separate goroutines.
	MinMatMulFlopsPerWorker = 32 * 1024
	noSIMDMinMatMulFlopsPerWorker = MinMatMulFlopsPerWorker
)

func init() {
	if !envutil.MustReadBool(EnabledEnv, true) {
		klog.Info("dot/nontransposed MatMul implementations disabled")
		return
	}
	if v := envutil.MustReadInt(envutil.GoBackendNoSIMD_KC, 0); v > 0 {
		NoSIMDParams.PanelContractingSize = v
	}
	if v := envutil.MustReadInt(envutil.GoBackendNoSIMD_MC, 0); v > 0 {
		NoSIMDParams.LHSPanelCrossSize = v
	}
	if v := envutil.MustReadInt(envutil.GoBackendNoSIMD_NC, 0); v > 0 {
		NoSIMDParams.RHSPanelCrossSize = v
	}
	registerNoSIMD(false)
}

func RegisterNoSIMDForTests() {
	registerNoSIMD(true)
}

func registerNoSIMD(forTests bool) {
	// DTypePairMap: callImplementationDTypePairMap (ints, same)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int16, noSIMDRouter[int16, int16], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int8, noSIMDRouter[int8, int8], PriorityNoSIMD, forTests)

	// DTypePairMap: callImplementationDTypePairMap (ints, int32,int64)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int32, noSIMDRouter[int16, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int16, dtypes.Int64, noSIMDRouter[int16, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int32, dtypes.Int64, noSIMDRouter[int32, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int32, noSIMDRouter[int64, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int32, noSIMDRouter[int8, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Int8, dtypes.Int64, noSIMDRouter[int8, int64], PriorityNoSIMD, forTests)

	// DTypePairMap: callImplementationDTypePairMap (uints, same)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint16, noSIMDRouter[uint16, uint16], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint8, noSIMDRouter[uint8, uint8], PriorityNoSIMD, forTests)

	// DTypePairMap: callImplementationDTypePairMap (uints, uint32,uint64)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint32, noSIMDRouter[uint16, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint16, dtypes.Uint64, noSIMDRouter[uint16, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint32, dtypes.Uint64, noSIMDRouter[uint32, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint32, noSIMDRouter[uint64, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint32, noSIMDRouter[uint8, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Uint8, dtypes.Uint64, noSIMDRouter[uint8, uint64], PriorityNoSIMD, forTests)

	// DTypePairMap: callImplementationDTypePairMap (floats, floats)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, noSIMDRouter[float32, float32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float64, noSIMDRouter[float32, float64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float32, noSIMDRouter[float64, float32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Float64, dtypes.Float64, noSIMDRouter[float64, float64], PriorityNoSIMD, forTests)

	// DTypePairMap: callImplementationDTypePairMap (half, float32)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.Float32, noSIMDHalfPrecisionRouter[bfloat16.BFloat16, float32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutNonTransposed, dtypes.Float16, dtypes.Float32, noSIMDHalfPrecisionRouter[float16.Float16, float32], PriorityNoSIMD, forTests)

	// Transposed versions:
	// Pairs from (ints, same)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int16, dtypes.Int16, noSIMDRouter[int16, int16], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int8, dtypes.Int8, noSIMDRouter[int8, int8], PriorityNoSIMD, forTests)

	// Pairs from (ints, int32,int64)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int16, dtypes.Int32, noSIMDRouter[int16, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int16, dtypes.Int64, noSIMDRouter[int16, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int32, dtypes.Int32, noSIMDRouter[int32, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int32, dtypes.Int64, noSIMDRouter[int32, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int64, dtypes.Int32, noSIMDRouter[int64, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int64, dtypes.Int64, noSIMDRouter[int64, int64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int8, dtypes.Int32, noSIMDRouter[int8, int32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Int8, dtypes.Int64, noSIMDRouter[int8, int64], PriorityNoSIMD, forTests)

	// Pairs from (uints, same)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint16, dtypes.Uint16, noSIMDRouter[uint16, uint16], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint8, dtypes.Uint8, noSIMDRouter[uint8, uint8], PriorityNoSIMD, forTests)

	// Pairs from (uints, uint32,uint64)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint16, dtypes.Uint32, noSIMDRouter[uint16, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint16, dtypes.Uint64, noSIMDRouter[uint16, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint32, dtypes.Uint32, noSIMDRouter[uint32, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint32, dtypes.Uint64, noSIMDRouter[uint32, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint64, dtypes.Uint32, noSIMDRouter[uint64, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint64, dtypes.Uint64, noSIMDRouter[uint64, uint64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint8, dtypes.Uint32, noSIMDRouter[uint8, uint32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Uint8, dtypes.Uint64, noSIMDRouter[uint8, uint64], PriorityNoSIMD, forTests)

	// Pairs from (floats, floats)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Float32, dtypes.Float32, noSIMDRouter[float32, float32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Float32, dtypes.Float64, noSIMDRouter[float32, float64], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Float64, dtypes.Float32, noSIMDRouter[float64, float32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Float64, dtypes.Float64, noSIMDRouter[float64, float64], PriorityNoSIMD, forTests)

	// Pairs from (half, float32)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.BFloat16, dtypes.Float32, noSIMDHalfPrecisionRouter[bfloat16.BFloat16, float32], PriorityNoSIMD, forTests)
	dot.RegisterImplementation("simd:nosimd", dot.LayoutTransposed, dtypes.Float16, dtypes.Float32, noSIMDHalfPrecisionRouter[float16.Float16, float32], PriorityNoSIMD, forTests)

}

// GetBuffer simplifies the process of getting a buffer from a backend and getting the flat slice.
func GetBuffer[T gotype.Supported](backend *gobackend.Backend, length int) (ref *gobackend.Buffer, flat []T, success bool) {
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

// WorkItem is used when parallelizing the DotGeneral: it allows splitting the work into batch/lhs/rhs slices.
type WorkItem struct {
	BatchStart, BatchEnd,
	LHSRowStart, LHSRowEnd,
	RHSColStart, RHSColEnd int
}

type workItem = WorkItem

// Choose2DSplit determines the 2D grid (numM, numN) of workers to divide an M x N matrix multiplication.
func Choose2DSplit(M, N, targetWorkers int, params *CacheParams) (numM, numN int) {
	if targetWorkers <= 1 {
		return 1, 1
	}

	minCol := max(1, params.RHSL1KernelCols)
	minRow := max(1, params.LHSL1KernelRows)
	targetRow := minRow
	targetCol := max(minCol, params.RHSPanelCrossSize)

	bestNumM := targetWorkers
	bestNumN := 1
	bestScore := -1.0

	// Consider candidate numN values: powers of 2 from 1 up to targetWorkers
	for candN := 1; candN <= targetWorkers; candN *= 2 {
		candM := (targetWorkers + candN - 1) / candN

		colChunk := (N + candN - 1) / candN
		rowChunk := (M + candM - 1) / candM

		if colChunk < minCol || rowChunk < minRow {
			continue
		}

		// Score components:
		// 1. Worker utilization: penalize if candM * candN != targetWorkers
		workerRatio := float64(candM*candN) / float64(targetWorkers)
		if workerRatio < 1.0 {
			workerRatio = 1.0 / workerRatio
		}
		utilPenalty := (workerRatio - 1.0) * 100.0

		// 2. Row panel reuse: rowChunk should be at least targetRow (LHSPanelCrossSize)
		// so that packed RHS in L2/L3 cache is reused across multiple LHS panels.
		rowPenalty := 0.0
		if rowChunk < targetRow {
			rowPenalty = float64(targetRow-rowChunk) / float64(targetRow) * 500.0
		}

		// 3. Col panel size: colChunk <= targetCol avoids re-packing LHS multiple times per worker.
		// Also colChunk >= 2 * minCol ensures efficient SIMD loop unrolling.
		colPenalty := 0.0
		if colChunk > targetCol {
			colPenalty = float64(colChunk-targetCol) / float64(targetCol) * 100.0
		} else if colChunk < 2*minCol && N >= 4*minCol {
			colPenalty = 50.0
		}

		// 4. Memory traffic estimation:
		// RHS packing traffic is proportional to candM * N.
		// LHS packing traffic is proportional to candN * M * ceil(colChunk / targetCol).
		nPanels := (colChunk + targetCol - 1) / targetCol
		traffic := float64(candM*N + candN*M*nPanels)
		trafficScore := traffic / float64(M+N)

		totalScore := trafficScore + utilPenalty + rowPenalty + colPenalty
		// Bonus for candN == 1 if rowChunk is large enough:
		// When rowChunk is already >= targetRow, splitting along M alone avoids
		// multiple workers having to write to different columns of the same row.
		if candN == 1 && rowChunk >= targetRow {
			totalScore -= 200.0
		}
		if bestScore < 0 || totalScore < bestScore {
			bestScore = totalScore
			bestNumM = candM
			bestNumN = candN
		}
	}

	return bestNumM, bestNumN
}

var choose2DSplit = Choose2DSplit

// FeedWorkItems splits the matrix-multiplication tasks into "WorkItem"s optimized
// for maxWorkers (>=1).
// It closes workChan on exit.
//
// FeedWorkItems is typically called on a separate goroutine, and it uses almost no CPU.
func FeedWorkItems(
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *CacheParams,
	maxWorkers int,
	workChan chan<- WorkItem) {
	defer func() {
		// Invariant: it closes the channel on exit.
		close(workChan)
	}()

	if maxWorkers <= 1 {
		workChan <- WorkItem{
			0, batchSize,
			0, lhsCrossSize,
			0, rhsCrossSize,
		}
		return
	}

	if batchSize >= 2*maxWorkers {
		// Split the work on the batch dimension only.
		batchStep := (batchSize + maxWorkers - 1) / maxWorkers
		for batchIdx := 0; batchIdx < batchSize; batchIdx += batchStep {
			workChan <- WorkItem{
				batchIdx, min(batchSize, batchIdx+batchStep),
				0, lhsCrossSize,
				0, rhsCrossSize,
			}
		}
		return
	}

	if batchSize >= maxWorkers {
		// Plenty of batch items: each worker gets whole batch items dynamically.
		for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
			workChan <- WorkItem{
				batchIdx, batchIdx + 1,
				0, lhsCrossSize,
				0, rhsCrossSize,
			}
		}
		return
	}

	// batchSize < maxWorkers: distribute workers across the batch items,
	// using 2D grid partitioning on each matrix slice.
	// Target 2 * maxWorkers to saturate the worker pool (2 workers per GOMAXPROCS).
	targetWorkersPerBatch := (2*maxWorkers + batchSize - 1) / batchSize
	numM, numN := choose2DSplit(lhsCrossSize, rhsCrossSize, targetWorkersPerBatch, params)

	rhsSplitSize := (rhsCrossSize + numN - 1) / numN
	if params.RHSL1KernelCols > 0 {
		rhsSplitSize = max(params.RHSL1KernelCols, (rhsSplitSize/params.RHSL1KernelCols)*params.RHSL1KernelCols)
	}
	lhsSplitSize := (lhsCrossSize + numM - 1) / numM
	if params.LHSL1KernelRows > 0 {
		lhsSplitSize = max(params.LHSL1KernelRows, (lhsSplitSize/params.LHSL1KernelRows)*params.LHSL1KernelRows)
	}

	for lhsRowIdx := 0; lhsRowIdx < lhsCrossSize; lhsRowIdx += lhsSplitSize {
		rowEnd := min(lhsCrossSize, lhsRowIdx+lhsSplitSize)
		for rhsColIdx := 0; rhsColIdx < rhsCrossSize; rhsColIdx += rhsSplitSize {
			colEnd := min(rhsCrossSize, rhsColIdx+rhsSplitSize)
			for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
				workChan <- WorkItem{
					batchIdx, batchIdx + 1,
					lhsRowIdx, rowEnd,
					rhsColIdx, colEnd,
				}
			}
		}
	}
}

var feedWorkItems = FeedWorkItems

// applyPackedOutput applies the computed packedOutput to the final output.
func noSIMDApplyPackedOutput[T gotype.ScalarNotComplex](
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

// ApplyPackedOutput applies the computed packedOutput to the final output.
func ApplyPackedOutput[T gotype.ScalarNotComplex](
	packedOutput, output []T,
	isFirstContractingPanel bool,
	packedOutputRowStride int,
	lhsRowOffset, rhsColOffset int, // Global output offsets
	outputRowStride int,
	height, width int, // actual amount of data to copy
) {
	noSIMDApplyPackedOutput(packedOutput, output, isFirstContractingPanel, packedOutputRowStride, lhsRowOffset, rhsColOffset, outputRowStride, height, width)
}

// TestNoSIMDRouterFloat32 exposes noSIMDRouter for testing and benchmarking.
func TestNoSIMDRouterFloat32(backend *gobackend.Backend, layout dot.Layout, lhs, rhs []float32, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, output []float32) {
	noSIMDRouter[float32, float32](backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nil)
}


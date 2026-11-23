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
		LHSPanelCrossSize:    2,   // Mc: Block Height fitting L1/L2 cache.
		RHSPanelCrossSize:    512, // Nc: Block Width fitting L2/L3 cache.
	}

	// SmallMatMulSizeThreshold is the threshold in byte size for switching to the small matrix multiplication kernel.
	// If the total number of operations is below this threshold, the small
	// matrix multiplication kernel is used instead of the tiled implementation.
	// This is a heuristic and may need to be tuned for different architectures.
	// Expressed in number of bytes.
	SmallMatMulSizeThreshold       = 4 * 1024 * 1024
	noSIMDSmallMatMulSizeThreshold = SmallMatMulSizeThreshold

	// MinMatMulFlopsPerWorker is the minimum number of flops per worker: above this number, if possible we should
	// parallelize computation on separate goroutines.
	MinMatMulFlopsPerWorker       = 96 * 1024
	noSIMDMinMatMulFlopsPerWorker = 8 * 1024

	// MinLargeMatMulFlopsPerWorker is the minimum number of multiply-adds per worker for the tiled/packed
	// large matrix multiplication kernels so each worker receives enough compute to amortize panel packing
	// and thread scheduling.
	MinLargeMatMulFlopsPerWorker = 256 * 1024
	// MinCachedLargeMatMulFlopsPerWorker is the minimum multiply-adds per worker when RHS (or LHS) panels
	// are already pre-packed/cached, where there is no packing overhead to amortize (e.g. M=1 LLM decode).
	MinCachedLargeMatMulFlopsPerWorker = 128 * 1024
	noSIMDMinLargeMatMulFlopsPerWorker = 32 * 1024
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

// GetExactBuffer allocates an exact-sized, non-pooled buffer for long-lived caches (e.g., PrepackRHS/PrepackLHS).
func GetExactBuffer[T gotype.Supported](_ *gobackend.Backend, length int) (ref *gobackend.Buffer, flat []T, success bool) {
	ref = gobackend.NewBuffer(shapes.Make(dtypes.FromGenericsType[T](), length))
	gobackend.AdviseHugePages(ref.RawBytes)
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

// MultiRHSM1WorkItem represents a column slice [ColStart, ColEnd) of target TargetIdx
// for M=1 Multi-RHS GEMV execution in a single SaturateUpTo wave.
type MultiRHSM1WorkItem struct {
	TargetIdx int
	ColStart  int
	ColEnd    int
}

// MultiRHSWorkItem represents a 2D work tile for Multi-RHS (or single-RHS epilogue) execution.
// When TargetIdx < 0, the worker processes all targets across their full column width [0, RHSCrossSize)
// for [LHSRowStart, LHSRowEnd), reusing packed LHS across all targets.
// When TargetIdx >= 0, the worker processes only [RHSColStart, RHSColEnd) of target TargetIdx.
type MultiRHSWorkItem struct {
	BatchStart, BatchEnd   int
	LHSRowStart, LHSRowEnd int
	TargetIdx              int
	RHSColStart, RHSColEnd int
}

// appendBalancedTargetColChunks distributes up to numChunks column chunks proportionally across
// targetCrossSizes using D'Hondt allocation (maximizing blocks/assigned per step) so every
// worker gets an evenly balanced, strip-aligned column slice with zero wasted cores.
func appendBalancedTargetColChunks(
	dst []MultiRHSM1WorkItem,
	targetCrossSizes []int,
	minCol int,
	numChunks int,
) []MultiRHSM1WorkItem {
	minCol = max(1, minCol)
	var blocksBuf [8]int
	var assignedBuf [8]int
	var blocks, assigned []int
	if len(targetCrossSizes) <= len(blocksBuf) {
		blocks = blocksBuf[:len(targetCrossSizes)]
		assigned = assignedBuf[:len(targetCrossSizes)]
	} else {
		blocks = make([]int, len(targetCrossSizes))
		assigned = make([]int, len(targetCrossSizes))
	}

	totalBlocks := 0
	activeTargets := 0
	for i, n := range targetCrossSizes {
		if n > 0 {
			b := (n + minCol - 1) / minCol
			blocks[i] = b
			totalBlocks += b
			activeTargets++
		}
	}
	if totalBlocks == 0 {
		return dst
	}
	if numChunks <= 1 || totalBlocks <= 1 {
		for tIdx, n := range targetCrossSizes {
			if n > 0 {
				dst = append(dst, MultiRHSM1WorkItem{TargetIdx: tIdx, ColStart: 0, ColEnd: n})
			}
		}
		return dst
	}

	targetChunks := min(numChunks, totalBlocks)
	assignedTotal := 0
	for i, b := range blocks {
		if b > 0 {
			assigned[i] = 1
			assignedTotal++
		}
	}
	// If there are more active targets than targetChunks, each active target gets 1 chunk.
	for assignedTotal < targetChunks {
		bestIdx := -1
		bestLoad := -1.0
		for i, b := range blocks {
			if b == 0 || assigned[i] >= b {
				continue
			}
			load := float64(b) / float64(assigned[i])
			if load > bestLoad {
				bestLoad = load
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		assigned[bestIdx]++
		assignedTotal++
	}

	for tIdx, n := range targetCrossSizes {
		chunks := assigned[tIdx]
		if chunks <= 0 || n <= 0 {
			continue
		}
		b := blocks[tIdx]
		baseBlocks := b / chunks
		remBlocks := b % chunks
		col := 0
		for c := range chunks {
			cBlocks := baseBlocks
			if c < remBlocks {
				cBlocks++
			}
			nextCol := min(n, col+cBlocks*minCol)
			if nextCol > col {
				dst = append(dst, MultiRHSM1WorkItem{TargetIdx: tIdx, ColStart: col, ColEnd: nextCol})
			}
			col = nextCol
		}
	}
	return dst
}

// AppendMultiRHSM1WorkItems splits M=1 work across one or more RHS targets into balanced,
// panel-friendly column chunks for a single parallel wave saturating maxWorkers.
// When total work allows (totalBlocks >= 16), chunks are allocated in 4-strip units (128 cols on
// AVX-512, 1/4 of a 512-col panel) to avoid strided DRAM streams and panel-boundary straddling.
func AppendMultiRHSM1WorkItems(
	dst []MultiRHSM1WorkItem,
	targetCrossSizes []int,
	params *CacheParams,
	maxWorkers int,
) []MultiRHSM1WorkItem {
	minCol := max(1, params.RHSL1KernelCols)
	totalBlocks := 0
	for _, n := range targetCrossSizes {
		if n > 0 {
			totalBlocks += (n + minCol - 1) / minCol
		}
	}
	unitStrips := 1
	if totalBlocks >= 16 {
		unitStrips = 4
	} else if totalBlocks >= 4 {
		unitStrips = 2
	}
	return appendBalancedTargetColChunks(dst, targetCrossSizes, unitStrips*minCol, maxWorkers)
}

// AppendMultiRHS2DWorkItems partitions Multi-RHS (M > 1) work across batch, M (rows), and
// targets/N (columns) while keeping each tile's row height close to LHSPanelCrossSize (Mc)
// so RHS weight panels get high L2/L3 reuse instead of collapsing to 1-Mr-strip DRAM streaming,
// and oversubscribing by ~2x so heterogeneous targets (e.g. gate+GELU vs up) stay load-balanced.
func AppendMultiRHS2DWorkItems(
	dst []MultiRHSWorkItem,
	batchSize, lhsCrossSize int,
	targetCrossSizes []int,
	params *CacheParams,
	maxWorkers int,
) []MultiRHSWorkItem {
	totalRHSCrossSize := 0
	activeTargets := 0
	for _, n := range targetCrossSizes {
		if n > 0 {
			totalRHSCrossSize += n
			activeTargets++
		}
	}
	if totalRHSCrossSize == 0 || lhsCrossSize == 0 || batchSize == 0 {
		return dst
	}
	if maxWorkers <= 1 {
		return append(dst, MultiRHSWorkItem{
			BatchStart:  0,
			BatchEnd:    batchSize,
			LHSRowStart: 0,
			LHSRowEnd:   lhsCrossSize,
			TargetIdx:   -1,
		})
	}

	mr := max(1, params.LHSL1KernelRows)
	mc := max(mr, params.LHSPanelCrossSize)
	nr := max(1, params.RHSL1KernelCols)

	// When batchSize >= maxWorkers or M has enough Mr strips to saturate maxWorkers on its own
	// (e.g. M >= 128 on AVX-512), row-only tapered splitting (TargetIdx = -1) is strictly optimal:
	// each worker packs a compact L1/L2-resident LHS strip once and reuses it across all RHS targets
	// while all cores sweep the shared RHS weight panels in L3 lockstep with a tapered tail.
	totalMBlocks := max(1, (lhsCrossSize+mr-1)/mr)
	workersPerBatch := (maxWorkers + batchSize - 1) / batchSize
	if batchSize >= maxWorkers || totalMBlocks >= workersPerBatch {
		var rowBuf [256]WorkItem
		rowItems := AppendRowWorkItems(rowBuf[:0], batchSize, lhsCrossSize, totalRHSCrossSize, params, maxWorkers)
		for _, wi := range rowItems {
			dst = append(dst, MultiRHSWorkItem{
				BatchStart:  wi.BatchStart,
				BatchEnd:    wi.BatchEnd,
				LHSRowStart: wi.LHSRowStart,
				LHSRowEnd:   wi.LHSRowEnd,
				TargetIdx:   -1,
			})
		}
		return dst
	}

	totalNBlocks := 0
	for _, n := range targetCrossSizes {
		if n > 0 {
			totalNBlocks += (n + nr - 1) / nr
		}
	}

	// Small-to-medium M (1 < M < maxWorkers*Mr, e.g. M=15..120): M alone cannot saturate all workers,
	// so split across (M, targets+N) using an exact divisor of workersPerBatch to avoid tail waves.
	idealM := min(totalMBlocks, max(1, (lhsCrossSize+mc-1)/mc))
	maxUsefulN := totalNBlocks
	if totalMBlocks > 1 && totalNBlocks >= 4*activeTargets {
		maxUsefulN = max(activeTargets, totalNBlocks/4)
	}
	// Find the divisor of workersPerBatch closest to idealM (never exceeding totalMBlocks)
	// whose complementary numN fits within maxUsefulN.
	numM := 1
	bestDiff := totalMBlocks + workersPerBatch
	for candM := 1; candM <= min(workersPerBatch, totalMBlocks); candM++ {
		if workersPerBatch%candM != 0 {
			continue
		}
		candN := workersPerBatch / candM
		if candN > maxUsefulN {
			continue
		}
		diff := candM - idealM
		if diff < 0 {
			diff = -diff
		}
		if diff < bestDiff {
			bestDiff = diff
			numM = candM
		}
	}
	numN := min(maxUsefulN, max(1, workersPerBatch/numM))
	if numM*numN < workersPerBatch && totalMBlocks > numM {
		numM = min(totalMBlocks, (workersPerBatch+numN-1)/numN)
	}
	// When all tiles have generous row height (near Mc) and N is wide enough, double numN to 2x
	// (an exact 2-wave schedule, e.g. 4x8 = 32 tiles on 16 workers) so heterogeneous targets
	// (e.g. gate+GELU vs up) and OS jitter stay load-balanced while reusing packedLHS across waves.
	if numM*numN == workersPerBatch && 2*numN <= maxUsefulN && lhsCrossSize/numM >= mc/2 && totalNBlocks >= 4*2*numN {
		numN *= 2
	}

	// If we didn't split across N at all (only 1 column chunk total), use TargetIdx = -1 so all
	// targets share packed LHS directly.
	if numN <= 1 && activeTargets > 1 {
		for batchIdx := range batchSize {
			for m := range numM {
				rowStart := ((m*totalMBlocks + numM - 1) / numM) * mr
				rowEnd := min(lhsCrossSize, (((m+1)*totalMBlocks+numM-1)/numM)*mr)
				if rowStart < rowEnd {
					dst = append(dst, MultiRHSWorkItem{
						BatchStart:  batchIdx,
						BatchEnd:    batchIdx + 1,
						LHSRowStart: rowStart,
						LHSRowEnd:   rowEnd,
						TargetIdx:   -1,
					})
				}
			}
		}
		return dst
	}

	var colBuf [128]MultiRHSM1WorkItem
	colChunks := appendBalancedTargetColChunks(colBuf[:0], targetCrossSizes, nr, numN)

	// Emit items ordered by (batch, m, colChunk) so workers grabbing adjacent items often share
	// the same LHS row range and can reuse their packedLHS buffer in L2 cache.
	for batchIdx := range batchSize {
		for m := range numM {
			rowStart := ((m*totalMBlocks + numM - 1) / numM) * mr
			rowEnd := min(lhsCrossSize, (((m+1)*totalMBlocks+numM-1)/numM)*mr)
			if rowStart >= rowEnd {
				continue
			}
			for _, cc := range colChunks {
				dst = append(dst, MultiRHSWorkItem{
					BatchStart:  batchIdx,
					BatchEnd:    batchIdx + 1,
					LHSRowStart: rowStart,
					LHSRowEnd:   rowEnd,
					TargetIdx:   cc.TargetIdx,
					RHSColStart: cc.ColStart,
					RHSColEnd:   cc.ColEnd,
				})
			}
		}
	}
	return dst
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
	maxPossibleM := max(1, M/minRow)

	bestNumM := min(targetWorkers, maxPossibleM)
	bestNumN := 1
	bestScore := -1.0

	// When M alone can saturate targetWorkers, keep a wide minimum column chunk (8*minCol)
	// so row-splitting is preferred for RHS L2/L3 reuse. When M is small (maxPossibleM < targetWorkers,
	// e.g. M=1 autoregressive decode), relax the column floor down to minCol (1 microkernel strip,
	// 128 bytes of float32/float64 output = 2 cache lines, zero false sharing) so narrow-N projections
	// can still utilize all available cores.
	minColChunk := 8 * minCol
	if maxPossibleM < targetWorkers {
		neededN := (targetWorkers + maxPossibleM - 1) / maxPossibleM
		targetChunk := (N / max(1, neededN) / minCol) * minCol
		minColChunk = max(minCol, min(8*minCol, targetChunk))
	}

	// Consider candidate numN values: powers of 2 from 1 up to targetWorkers
	for candN := 1; candN <= targetWorkers; candN *= 2 {
		candM := (targetWorkers + candN - 1) / candN
		if candM > maxPossibleM {
			candM = maxPossibleM
		}

		colChunk := (N + candN - 1) / candN
		if minCol > 1 {
			colChunk = ((colChunk + minCol - 1) / minCol) * minCol
		}
		rowChunk := (M + candM - 1) / candM
		if minRow > 1 {
			rowChunk = ((rowChunk + minRow - 1) / minRow) * minRow
		}

		if colChunk < minCol || rowChunk < minRow {
			continue
		}
		if candN > 1 {
			if colChunk < minColChunk {
				continue
			}
		}

		actualM := max(1, (M+rowChunk-1)/rowChunk)
		actualN := max(1, (N+colChunk-1)/colChunk)

		// Score components:
		// 1. Worker utilization: penalize if actualM * actualN != targetWorkers
		workerRatio := float64(actualM*actualN) / float64(targetWorkers)
		if workerRatio < 1.0 {
			workerRatio = 1.0 / workerRatio
		}
		utilPenalty := (workerRatio - 1.0) * 100.0

		// 2. Row panel reuse: rowChunk should be at least targetRow
		// so that packed RHS in L2/L3 cache is reused across multiple LHS panels.
		rowPenalty := 0.0
		if rowChunk < targetRow {
			rowPenalty = float64(targetRow-rowChunk) / float64(targetRow) * 500.0
		}

		// 3. Col panel size and LHS re-packing:
		// Each worker caches multiple packedLHS panels across column panels (up to 16 panels in SIMD
		// and 64 panels in NoSIMD). Only when rowChunk exceeds the cached LHS capacity does
		// colChunk > targetCol require re-packing LHS across column panels.
		maxCachedLHSRows := max(16*params.LHSPanelCrossSize, 128)
		nPanels := 1
		if rowChunk > maxCachedLHSRows {
			nPanels = (colChunk + targetCol - 1) / targetCol
		}
		colPenalty := 0.0
		if nPanels > 1 && colChunk > targetCol {
			colPenalty = float64(colChunk-targetCol) / float64(targetCol) * 100.0
		} else if colChunk < 2*minCol && N >= 4*minCol {
			colPenalty = 50.0
		}

		// 4. Memory traffic estimation:
		// RHS packing traffic is proportional to candM * N.
		// LHS packing traffic is proportional to candN * M * nPanels.
		traffic := float64(candM*N + candN*M*nPanels)
		trafficScore := traffic / float64(M+N)

		totalScore := trafficScore + utilPenalty + rowPenalty + colPenalty
		// Bonus for candN == 1 if rowChunk is large enough:
		// When rowChunk is already >= targetRow, splitting along M alone avoids
		// multiple workers having to pack the same LHS rows and write to the same output rows.
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

// AppendWorkItems appends the matrix-multiplication WorkItems optimized for maxWorkers (>=1) to dst
// and returns the resulting slice. Passing a stack-allocated backing slice avoids heap allocations.
//
// It uses a 3-phase tapered chunking strategy for asymmetric multi-core systems:
//  1. Head (multi-batch chunks when batchSize is large): groups ~remaining/(2*maxWorkers) whole batches.
//  2. Middle (single-batch chunks): 1 whole batch per WorkItem.
//  3. Tail (sub-batch < 1 batch 2D tiles): splits the final batches across (M, N) into 2D sub-tiles
//     so slower cores (e.g. E-cores or NEON fallback workers) never hold an entire batch hostage at the barrier.
func AppendWorkItems(
	dst []WorkItem,
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *CacheParams,
	maxWorkers int,
) []WorkItem {
	if maxWorkers <= 1 {
		return append(dst, WorkItem{
			0, batchSize,
			0, lhsCrossSize,
			0, rhsCrossSize,
		})
	}

	mr := max(1, params.LHSL1KernelRows)
	nr := max(1, params.RHSL1KernelCols)
	totalMBlocks := max(1, (lhsCrossSize+mr-1)/mr)
	totalNBlocks := max(1, (rhsCrossSize+nr-1)/nr)
	maxCap := max(64, cap(dst))

	if batchSize >= maxWorkers {
		// Determine whether the final few batches should be sub-split into < 1 batch 2D tiles.
		tailSubSplitBatches := 0
		actualSubM, actualSubN := 1, 1
		if totalMBlocks*totalNBlocks > 1 && lhsCrossSize*rhsCrossSize >= 4*mr*nr && batchSize > 1 {
			subM, subN := choose2DSplit(lhsCrossSize, rhsCrossSize, min(8, maxWorkers), params)
			actualSubM = min(max(1, subM), totalMBlocks)
			actualSubN = min(max(1, subN), totalNBlocks)
			tilesPerTailBatch := actualSubM * actualSubN
			if tilesPerTailBatch > 1 {
				if batchSize < 2*maxWorkers {
					tailSubSplitBatches = min(batchSize-1, max(1, maxWorkers/3))
				} else {
					tailSubSplitBatches = min(4, max(1, maxWorkers/3))
				}
				for tailSubSplitBatches > 0 && tailSubSplitBatches*tilesPerTailBatch > maxCap/3 {
					tailSubSplitBatches--
				}
			}
		}

		fullBatches := batchSize - tailSubSplitBatches
		headCap := max(1, maxCap-tailSubSplitBatches*actualSubM*actualSubN)
		divisor := 2 * maxWorkers

		// Phase 1 & 2: Tapered multi-batch -> single-batch chunks.
		for b := 0; b < fullBatches; {
			rem := fullBatches - b
			slotsLeft := max(1, headCap-len(dst))
			chunk := max(1, (rem+divisor-1)/divisor)
			minChunkForCap := (rem + slotsLeft - 1) / slotsLeft
			chunk = min(rem, max(chunk, minChunkForCap))
			dst = append(dst, WorkItem{
				b, b + chunk,
				0, lhsCrossSize,
				0, rhsCrossSize,
			})
			b += chunk
		}

		// Phase 3: Sub-batch (< 1 batch) 2D tail chunks.
		for b := fullBatches; b < batchSize; b++ {
			for m := range actualSubM {
				lhsRowStart := ((m*totalMBlocks + actualSubM - 1) / actualSubM) * mr
				lhsRowEnd := min(lhsCrossSize, (((m+1)*totalMBlocks+actualSubM-1)/actualSubM)*mr)
				if lhsRowStart >= lhsRowEnd {
					continue
				}
				for n := range actualSubN {
					rhsColStart := ((n*totalNBlocks + actualSubN - 1) / actualSubN) * nr
					rhsColEnd := min(rhsCrossSize, (((n+1)*totalNBlocks+actualSubN-1)/actualSubN)*nr)
					if rhsColStart >= rhsColEnd {
						continue
					}
					dst = append(dst, WorkItem{
						b, b + 1,
						lhsRowStart, lhsRowEnd,
						rhsColStart, rhsColEnd,
					})
				}
			}
		}
		return dst
	}

	// batchSize < maxWorkers: distribute workers across the batch items using 2D grid partitioning.
	// When the matrix has enough M blocks, split the first ~75% of M blocks into standard coarse tiles
	// and taper the final ~25% of M blocks into finer tiles so straggler cores finish quickly.
	if totalMBlocks >= 8 && totalMBlocks*totalNBlocks*batchSize >= 4*maxWorkers {
		headMBlocks := (totalMBlocks * 3) / 4
		tailMBlocks := totalMBlocks - headMBlocks
		headRows := headMBlocks * mr

		targetHead := (2*maxWorkers + batchSize - 1) / batchSize
		headNumM, headNumN := choose2DSplit(headRows, rhsCrossSize, targetHead, params)
		actualHeadM := min(max(1, headNumM), headMBlocks)
		actualHeadN := min(max(1, headNumN), totalNBlocks)

		targetTail := (2*maxWorkers + batchSize - 1) / batchSize
		tailNumM, tailNumN := choose2DSplit(lhsCrossSize-headRows, rhsCrossSize, targetTail, params)
		actualTailM := min(max(1, tailNumM), tailMBlocks)
		actualTailN := min(max(1, tailNumN), totalNBlocks)

		if (actualHeadM*actualHeadN+actualTailM*actualTailN)*batchSize <= maxCap {
			for m := range actualHeadM {
				lhsRowStart := ((m*headMBlocks + actualHeadM - 1) / actualHeadM) * mr
				lhsRowEnd := min(headRows, (((m+1)*headMBlocks+actualHeadM-1)/actualHeadM)*mr)
				if lhsRowStart >= lhsRowEnd {
					continue
				}
				for n := range actualHeadN {
					rhsColStart := ((n*totalNBlocks + actualHeadN - 1) / actualHeadN) * nr
					rhsColEnd := min(rhsCrossSize, (((n+1)*totalNBlocks+actualHeadN-1)/actualHeadN)*nr)
					if rhsColStart >= rhsColEnd {
						continue
					}
					for batchIdx := range batchSize {
						dst = append(dst, WorkItem{
							batchIdx, batchIdx + 1,
							lhsRowStart, lhsRowEnd,
							rhsColStart, rhsColEnd,
						})
					}
				}
			}
			for m := range actualTailM {
				lhsRowStart := headRows + ((m*tailMBlocks+actualTailM-1)/actualTailM)*mr
				lhsRowEnd := min(lhsCrossSize, headRows+(((m+1)*tailMBlocks+actualTailM-1)/actualTailM)*mr)
				if lhsRowStart >= lhsRowEnd {
					continue
				}
				for n := range actualTailN {
					rhsColStart := ((n*totalNBlocks + actualTailN - 1) / actualTailN) * nr
					rhsColEnd := min(rhsCrossSize, (((n+1)*totalNBlocks+actualTailN-1)/actualTailN)*nr)
					if rhsColStart >= rhsColEnd {
						continue
					}
					for batchIdx := range batchSize {
						dst = append(dst, WorkItem{
							batchIdx, batchIdx + 1,
							lhsRowStart, lhsRowEnd,
							rhsColStart, rhsColEnd,
						})
					}
				}
			}
			return dst
		}
	}

	if totalMBlocks == 1 {
		// 1D split along N (e.g. M=1 autoregressive decode): create one balanced N-chunk per worker
		// without 2x oversubscription or power-of-2 rounding, avoiding uneven waves (e.g. 24 items on
		// 16 workers) and excessive K-panel striding on narrow-N projections.
		// Keep each worker chunk at least 4 microkernel strips (128 cols on AVX-512, 1/4 of a 512-col
		// panel) when N is wide enough so workers avoid strided DRAM thrashing and load imbalance
		// (e.g. N=1536 splits into 12 exact 128-col chunks aligned with 512-col panels).
		minStrips := 1
		if totalNBlocks >= 16 {
			minStrips = 4
		} else if totalNBlocks >= 4 {
			minStrips = 2
		}
		workersPerBatch := (maxWorkers + batchSize - 1) / batchSize
		actualN := min(workersPerBatch, max(1, totalNBlocks/minStrips))
		for n := range actualN {
			rhsColStart := ((n*totalNBlocks + actualN - 1) / actualN) * nr
			rhsColEnd := min(rhsCrossSize, (((n+1)*totalNBlocks+actualN-1)/actualN)*nr)
			if rhsColStart >= rhsColEnd {
				continue
			}
			for batchIdx := range batchSize {
				dst = append(dst, WorkItem{
					batchIdx, batchIdx + 1,
					0, lhsCrossSize,
					rhsColStart, rhsColEnd,
				})
			}
		}
		return dst
	}

	targetWorkersPerBatch := (2*maxWorkers + batchSize - 1) / batchSize
	numM, numN := choose2DSplit(lhsCrossSize, rhsCrossSize, targetWorkersPerBatch, params)

	actualM := min(max(1, numM), totalMBlocks)
	actualN := min(max(1, numN), totalNBlocks)

	for m := range actualM {
		lhsRowStart := ((m*totalMBlocks + actualM - 1) / actualM) * mr
		lhsRowEnd := min(lhsCrossSize, (((m+1)*totalMBlocks+actualM-1)/actualM)*mr)
		for n := range actualN {
			rhsColStart := ((n*totalNBlocks + actualN - 1) / actualN) * nr
			rhsColEnd := min(rhsCrossSize, (((n+1)*totalNBlocks+actualN-1)/actualN)*nr)
			for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
				dst = append(dst, WorkItem{
					batchIdx, batchIdx + 1,
					lhsRowStart, lhsRowEnd,
					rhsColStart, rhsColEnd,
				})
			}
		}
	}
	return dst
}

// AppendRowWorkItems partitions work across batch and LHS rows (M) only (RHSColStart=0, RHSColEnd=rhsCrossSize),
// aligning row chunks to LHSPanelCrossSize when M is large so each worker can pack its LHS rows once
// and reuse them across multiple RHS targets and inline row epilogues.
func AppendRowWorkItems(
	dst []WorkItem,
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *CacheParams,
	maxWorkers int,
) []WorkItem {
	if maxWorkers <= 1 {
		return append(dst, WorkItem{
			0, batchSize,
			0, lhsCrossSize,
			0, rhsCrossSize,
		})
	}

	mr := max(1, params.LHSL1KernelRows)
	mc := max(mr, params.LHSPanelCrossSize)
	totalMBlocks := max(1, (lhsCrossSize+mr-1)/mr)
	maxCap := max(64, cap(dst))

	if batchSize >= maxWorkers {
		divisor := 2 * maxWorkers
		for b := 0; b < batchSize; {
			rem := batchSize - b
			slotsLeft := max(1, maxCap-len(dst))
			chunk := max(1, (rem+divisor-1)/divisor)
			minChunkForCap := (rem + slotsLeft - 1) / slotsLeft
			chunk = min(rem, max(chunk, minChunkForCap))
			dst = append(dst, WorkItem{
				b, b + chunk,
				0, lhsCrossSize,
				0, rhsCrossSize,
			})
			b += chunk
		}
		return dst
	}

	rowQuantum := mr
	if lhsCrossSize >= 2*maxWorkers*mc {
		rowQuantum = mc
	}
	totalBlocks := max(1, (lhsCrossSize+rowQuantum-1)/rowQuantum)
	workersPerBatch := (maxWorkers + batchSize - 1) / batchSize
	targetPerBatch := (2*maxWorkers + batchSize - 1) / batchSize

	if totalBlocks >= 8 {
		headBlocks := (totalBlocks * 3) / 4
		headRows := min(lhsCrossSize, headBlocks*rowQuantum)
		tailMBlocks := max(1, (lhsCrossSize-headRows+mr-1)/mr)

		// Size head chunks at ~workersPerBatch (so each head chunk gets >= 2*Mr rows when M allows,
		// giving 2x+ better RHS panel reuse than 1-Mr-strip chunks) and keep the final 25% tapered
		// into finer tail chunks for load balancing.
		actualHeadM := min(max(1, workersPerBatch), headBlocks)
		actualTailM := min(max(1, targetPerBatch), tailMBlocks)
		if (actualHeadM+actualTailM)*batchSize <= maxCap {
			for m := range actualHeadM {
				lhsRowStart := ((m*headBlocks + actualHeadM - 1) / actualHeadM) * rowQuantum
				lhsRowEnd := min(headRows, (((m+1)*headBlocks+actualHeadM-1)/actualHeadM)*rowQuantum)
				if lhsRowStart >= lhsRowEnd {
					continue
				}
				for batchIdx := range batchSize {
					dst = append(dst, WorkItem{
						batchIdx, batchIdx + 1,
						lhsRowStart, lhsRowEnd,
						0, rhsCrossSize,
					})
				}
			}
			for m := range actualTailM {
				lhsRowStart := headRows + ((m*tailMBlocks+actualTailM-1)/actualTailM)*mr
				lhsRowEnd := min(lhsCrossSize, headRows+(((m+1)*tailMBlocks+actualTailM-1)/actualTailM)*mr)
				if lhsRowStart >= lhsRowEnd {
					continue
				}
				for batchIdx := range batchSize {
					dst = append(dst, WorkItem{
						batchIdx, batchIdx + 1,
						lhsRowStart, lhsRowEnd,
						0, rhsCrossSize,
					})
				}
			}
			return dst
		}
	}

	actualM := min(max(1, targetPerBatch), totalMBlocks)
	for m := range actualM {
		lhsRowStart := ((m*totalMBlocks + actualM - 1) / actualM) * mr
		lhsRowEnd := min(lhsCrossSize, (((m+1)*totalMBlocks+actualM-1)/actualM)*mr)
		if lhsRowStart >= lhsRowEnd {
			continue
		}
		for batchIdx := range batchSize {
			dst = append(dst, WorkItem{
				batchIdx, batchIdx + 1,
				lhsRowStart, lhsRowEnd,
				0, rhsCrossSize,
			})
		}
	}
	return dst
}

// FeedWorkItems splits the matrix-multiplication tasks into "WorkItem"s optimized
// for maxWorkers (>=1) and sends them to workChan, closing workChan on exit.
func FeedWorkItems(
	batchSize, lhsCrossSize, rhsCrossSize int,
	params *CacheParams,
	maxWorkers int,
	workChan chan<- WorkItem) {
	defer close(workChan)
	var stackBuf [256]WorkItem
	items := AppendWorkItems(stackBuf[:0], batchSize, lhsCrossSize, rhsCrossSize, params, maxWorkers)
	for _, item := range items {
		workChan <- item
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


// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dot

import (
	"sync"

	"github.com/gomlx/compute/internal/gobackend"
)

// execSmallTransposed executes the dot general operation for normalized shapes:
// both rhs and lhs are shaped [batchSize, crossSize, contractingSize].
func execSmallTransposed(
	backend *gobackend.Backend,
	lhs, rhs *gobackend.Buffer,
	params *NodeData,
	output *gobackend.Buffer) error {
	batchSize := params.BatchSize

	normalizeDotGeneralFnAny, err := SmallTransposedDTypeMap.Get(params.InputDType)
	if err != nil {
		return err
	}
	normalizeDotGeneralFn := normalizeDotGeneralFnAny.(func(lhs, rhs, output *gobackend.Buffer,
		params *NodeData, batchStartIdx, batchEndIdx int))

	// Decide on using parallelism across the batch -- each example is started on a separate worker.
	useBatchParallelism := backend.Workers.IsEnabled()
	maxParallelism := backend.Workers.MaxParallelism()
	batchSplitSize := 1
	if useBatchParallelism && !backend.Workers.IsUnlimited() {
		batchSplitSize = (params.BatchSize + maxParallelism - 1) / maxParallelism
	}

	if !useBatchParallelism {
		// Process the whole batch in one call inline in the current worker.
		normalizeDotGeneralFn(lhs, rhs, output, params, 0, batchSize)
	} else {
		// Split in batchSplitSize
		wg := sync.WaitGroup{}
		for batchStartIdx := 0; batchStartIdx < batchSize; batchStartIdx += batchSplitSize {
			batchEndIdx := min(batchStartIdx+batchSplitSize, batchSize)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				normalizeDotGeneralFn(lhs, rhs, output, params, batchStartIdx, batchEndIdx)
				wg.Done()
			})
		}
		wg.Wait()
	}
	return nil
}

//gobackend:dtypemap execSmallTransposedGeneric ints,uints,floats
//gobackend:dtypemap execSmallTransposedHalfPrecision half
var SmallTransposedDTypeMap = gobackend.NewDTypeMap("SmallTransposedNormalized")

// Auto-generate alternate specialized versions of execNormalizedDotGeneral
// (that can't easily be refactored into smaller functions due to latency penalities)
//go:generate go run ../../cmd/alternates_generator -base=small_transposed_alt_base.go -tags=half

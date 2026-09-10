// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"sync"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

// Epilogue specifies post-processing operations (optional bias addition and optional activation)
// executed while output data is fresh in CPU cache.
type Epilogue[T any] struct {
	// Bias is an optional 1D slice of size equal to the output features (rhsCrossSize).
	// If non-nil and non-empty, Bias is added to each output row.
	Bias []T

	// Activation is an optional in-place activation function applied to a row or slice.
	// Matmul remains completely agnostic to the specific activation implementation.
	Activation func(data []T)
}

// HasWork returns true if the epilogue contains either a bias or an activation.
func (e Epilogue[T]) HasWork() bool {
	return len(e.Bias) > 0 || e.Activation != nil
}

// ApplyEpilogue applies the epilogue across output.
// - output has shape [batchSize, lhsCrossSize, rhsCrossSize].
// - bias has shape [rhsCrossSize].
func ApplyEpilogue[T any](
	backend *gobackend.Backend,
	output []T,
	batchSize, lhsCrossSize, rhsCrossSize int,
	epilogue Epilogue[T],
) {
	if !epilogue.HasWork() || len(output) == 0 {
		return
	}

	numRows := batchSize * lhsCrossSize
	rowSize := rhsCrossSize
	bias := epilogue.Bias
	hasBias := len(bias) > 0
	act := epilogue.Activation

	// Fast path: No bias, only activation. We can apply activation on the whole buffer or in chunks.
	if !hasBias && act != nil {
		if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && len(output) > 32768 {
			numWorkers := backend.Workers.AdjustedMaxParallelism()
			targetChunks := max(1, numWorkers*2)
			chunkSize := max(16384, (len(output)+targetChunks-1)/targetChunks)
			var wg sync.WaitGroup
			for i := 0; i < len(output); i += chunkSize {
				end := min(i+chunkSize, len(output))
				chunk := output[i:end]
				wg.Add(1)
				backend.Workers.WaitToStart(func() {
					act(chunk)
					wg.Done()
				})
			}
			wg.Wait()
		} else {
			act(output)
		}
		return
	}

	// Path with bias (and optional activation): Process row-by-row so each row
	// is modified and activated while resident in L1 cache.
	applyRows := func(rStart, rEnd int) {
		for r := rStart; r < rEnd; r++ {
			rowOffset := r * rowSize
			row := output[rowOffset : rowOffset+rowSize]
			addBias(row, bias)
			if act != nil {
				act(row)
			}
		}
	}

	totalElements := numRows * rowSize
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && numRows > 1 && totalElements > 32768 {
		numWorkers := backend.Workers.AdjustedMaxParallelism()
		targetChunks := max(1, numWorkers*2)
		chunkSize := max(16384, (totalElements+targetChunks-1)/targetChunks)
		rowsPerChunk := max(1, chunkSize/rowSize)
		var wg sync.WaitGroup
		for r := 0; r < numRows; r += rowsPerChunk {
			rStart := r
			rEnd := min(r+rowsPerChunk, numRows)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				applyRows(rStart, rEnd)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		applyRows(0, numRows)
	}
}

func addBias[T any](row, bias []T) {
	switch r := any(row).(type) {
	case []float32:
		b := any(bias).([]float32)
		if addBiasFloat32Arch(r, b) {
			return
		}
		for i, val := range b {
			r[i] += val
		}
	case []float64:
		b := any(bias).([]float64)
		for i, val := range b {
			r[i] += val
		}
	case []bfloat16.BFloat16:
		b := any(bias).([]bfloat16.BFloat16)
		for i, val := range b {
			r[i] = bfloat16.FromFloat32(r[i].Float32() + val.Float32())
		}
	case []float16.Float16:
		b := any(bias).([]float16.Float16)
		for i, val := range b {
			r[i] = float16.FromFloat32(r[i].Float32() + val.Float32())
		}
	}
}

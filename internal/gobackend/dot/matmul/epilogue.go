// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
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

	// BiasTileActivation is an optional fused bias + activation 2D tile kernel:
	//   out[r*rowStride + c] = act(out[r*rowStride + c] + bias[c])
	// When non-nil and Bias is non-empty, ApplyEpilogue* uses this to fuse bias addition
	// and activation in a single register pass over a 2D tile (amortizing constant loads).
	BiasTileActivation func(out []T, rowStride, numRows, width int, bias []T)

	// ActivationMinParallelChunk is the minimum element chunk size from the activation's KernelMetadata.
	// When <= 0, defaults to 16384.
	ActivationMinParallelChunk int

	// TilePostStore is an optional hook invoked immediately after Bias and Activation have been
	// applied to a finished output tile [rowStart:rowEnd, colStart:colEnd] while it is still hot
	// in L1/L2 cache (e.g. to convert Float32 tiles directly into a BFloat16/Float16 output buffer
	// without a trailing full-tensor ParallelConvertSlice pass).
	TilePostStore func(outputMatrix []T, rhsCrossSize, rowStart, rowEnd, colStart, colEnd int)
}

// HasWork returns true if the epilogue contains a bias, activation, or post-store tile hook.
func (e Epilogue[T]) HasWork() bool {
	return len(e.Bias) > 0 || e.Activation != nil || e.BiasTileActivation != nil || e.TilePostStore != nil
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
	biasTileAct := epilogue.BiasTileActivation
	postStore := epilogue.TilePostStore

	minChunkElements := 16384
	if (act != nil || biasTileAct != nil) && epilogue.ActivationMinParallelChunk > 0 {
		minChunkElements = epilogue.ActivationMinParallelChunk
	}
	minElementsThreshold := minChunkElements * 2

	// Fast path: No bias and no postStore, only activation.
	if !hasBias && postStore == nil && act != nil {
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && len(output) > minElementsThreshold {
			backend.Scheduler.ParallelizeChunks(len(output), minChunkElements, func(start, end int) {
				act(output[start:end])
			})
		} else {
			act(output)
		}
		return
	}

	// Single-row fast path with postStore (e.g. M=1 fallback small/nosimd): parallelize across columns if large.
	if numRows == 1 && postStore != nil && rowSize > minElementsThreshold &&
		backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() {
		backend.Scheduler.ParallelizeChunks(rowSize, minChunkElements, func(cStart, cEnd int) {
			ApplyEpilogueTile(output, rowSize, 0, 1, cStart, cEnd, epilogue)
		})
		return
	}

	// Path with bias, activation, and/or postStore: Process row chunks so each chunk
	// is modified, activated, and converted while resident in L1/L2 cache.
	applyRows := func(rStart, rEnd int) {
		if rEnd <= rStart {
			return
		}
		ApplyEpilogueTile(output, rowSize, rStart, rEnd, 0, rowSize, epilogue)
	}

	totalElements := numRows * rowSize
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && numRows > 1 && totalElements > minElementsThreshold {
		minRows := max(1, minChunkElements/max(1, rowSize))
		backend.Scheduler.ParallelizeChunks(numRows, minRows, applyRows)
	} else {
		applyRows(0, numRows)
	}
}

// ApplyEpilogueRows applies the epilogue inline on a contiguous slice of output rows
// (numRows * rowSize elements) that are already resident in the current worker's L1/L2 cache.
func ApplyEpilogueRows[T any](
	outRows []T,
	numRows, rowSize int,
	epilogue Epilogue[T],
) {
	ApplyEpilogueTile(outRows, rowSize, 0, numRows, 0, rowSize, epilogue)
}

// ApplyEpilogueTile applies the epilogue inline on a 2D sub-tile [rowStart:rowEnd, colStart:colEnd]
// of outputMatrix (with row stride rhsCrossSize) while the tile is hot in L1/L2 cache.
func ApplyEpilogueTile[T any](
	outputMatrix []T,
	rhsCrossSize int,
	rowStart, rowEnd, colStart, colEnd int,
	epilogue Epilogue[T],
) {
	numRows := rowEnd - rowStart
	width := colEnd - colStart
	if !epilogue.HasWork() || len(outputMatrix) == 0 || numRows <= 0 || width <= 0 {
		return
	}
	bias := epilogue.Bias
	hasBias := len(bias) > 0
	act := epilogue.Activation
	biasTileAct := epilogue.BiasTileActivation
	postStore := epilogue.TilePostStore

	if !hasBias {
		if act != nil {
			if colStart == 0 && width == rhsCrossSize {
				act(outputMatrix[rowStart*rhsCrossSize : rowEnd*rhsCrossSize])
			} else {
				for r := rowStart; r < rowEnd; r++ {
					rowOff := r*rhsCrossSize + colStart
					act(outputMatrix[rowOff : rowOff+width])
				}
			}
		}
		if postStore != nil {
			postStore(outputMatrix, rhsCrossSize, rowStart, rowEnd, colStart, colEnd)
		}
		return
	}

	biasSlice := bias[colStart:colEnd]
	if biasTileAct != nil {
		outOff := rowStart*rhsCrossSize + colStart
		biasTileAct(outputMatrix[outOff:outOff+(numRows-1)*rhsCrossSize+width], rhsCrossSize, numRows, width, biasSlice)
		if postStore != nil {
			postStore(outputMatrix, rhsCrossSize, rowStart, rowEnd, colStart, colEnd)
		}
		return
	}

	for r := rowStart; r < rowEnd; r++ {
		rowOff := r*rhsCrossSize + colStart
		row := outputMatrix[rowOff : rowOff+width]
		addBias(row, biasSlice)
		if act != nil {
			act(row)
		}
	}
	if postStore != nil {
		postStore(outputMatrix, rhsCrossSize, rowStart, rowEnd, colStart, colEnd)
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

// AddBias adds bias to row in-place using SIMD/assembly where available.
func AddBias[T any](row, bias []T) {
	addBias(row, bias)
}

// CopyAndAddBiasFloat32 copies src to dst while adding bias in a single pass.
// If bias is nil, it performs a standard copy. Uses AVX-512 or AVX2 assembly where available.
func CopyAndAddBiasFloat32(dst, src, bias []float32) {
	if bias == nil {
		copy(dst, src)
		return
	}
	if copyAndAddBiasFloat32Arch(dst, src, bias) {
		return
	}
	for i, val := range bias {
		dst[i] = src[i] + val
	}
}

// CopyAndAddBiasFloat64 copies src to dst while adding bias in a single pass.
// If bias is nil, it performs a standard copy.
func CopyAndAddBiasFloat64(dst, src, bias []float64) {
	if bias == nil {
		copy(dst, src)
		return
	}
	for i, val := range bias {
		dst[i] = src[i] + val
	}
}

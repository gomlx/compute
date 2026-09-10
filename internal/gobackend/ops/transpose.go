// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"slices"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
)

func init() {
	gobackend.RegisterTranspose.Register(Transpose, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeTranspose, gobackend.PriorityGeneric, execTranspose)
}

// Transpose axes of x.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func Transpose(f *gobackend.Function, operandValue compute.Value, permutations ...int) (compute.Value, error) {
	opType := compute.OpTypeTranspose
	inputs, err := f.VerifyAndCastValues(opType.String(), operandValue)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	outputShape, err := shapeinference.Transpose(operand.Shape, permutations)
	if err != nil {
		// This should have been validated during graph build, but we check again just in case.
		return nil, err
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, inputs, permutations)
	return node, nil
}

// TransposeDTypeMap is used to dispatch Transpose to type-specific implementations.
// gobackend:dtypemap execTransposeGeneric ints,uints,floats,half,bool
var TransposeDTypeMap = gobackend.NewDTypeMap("Transpose")

func init() {
	TransposeDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, execTransposeFloat16)
	TransposeDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, execTransposeBFloat16)
}

// ExecuteTranspose transposes operand into output given permutations, using fast paths if available.
func ExecuteTranspose(backend *gobackend.Backend, operand, output *gobackend.Buffer, permutations []int) {
	if dispatchFastTranspose(backend, operand, output, permutations) {
		return
	}
	it := NewTransposeIterator(operand.RawShape, permutations)
	dtype := output.RawShape.DType
	tmpAny, tmpErr := TransposeDTypeMap.Get(dtype)
	if tmpErr != nil {
		panic(tmpErr)
	}
	transposeFn := tmpAny.(func(operand, output *gobackend.Buffer, it *TransposeIterator))
	transposeFn(operand, output, it)
}

// execTranspose implements Transpose.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func execTranspose(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	permutations := node.Data.([]int)
	_ = inputsOwned // We don't reuse the inputs.

	// We can't write to the same buffer we read from because it's not done with swaps.
	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	ExecuteTranspose(backend, operand, output, permutations)
	return output, nil
}

func isIdentityPermutation(permutations []int) bool {
	for i, p := range permutations {
		if p != i {
			return false
		}
	}
	return true
}

func dispatchFastTranspose(backend *gobackend.Backend, operand, output *gobackend.Buffer, permutations []int) bool {
	dims := operand.RawShape.Dimensions
	rank := len(dims)
	if rank == 0 || operand.RawShape.Size() == 0 {
		return true
	}

	// Case 0: Identity permutation
	if isIdentityPermutation(permutations) {
		return dispatchIdentity(operand, output)
	}

	// Case 1: 4D permutation [0, 2, 1, 3] or 3D [1, 0, 2] (multi-head attention transpose)
	if (rank == 4 && permutations[0] == 0 && permutations[1] == 2 && permutations[2] == 1 && permutations[3] == 3) ||
		(rank == 3 && permutations[0] == 1 && permutations[1] == 0 && permutations[2] == 2) {
		var bDim, sDim, hDim, dDim int
		if rank == 4 {
			bDim, sDim, hDim, dDim = dims[0], dims[1], dims[2], dims[3]
		} else {
			bDim, sDim, hDim, dDim = 1, dims[0], dims[1], dims[2]
		}
		return dispatchTranspose4D_0213(backend, operand, output, bDim, sDim, hDim, dDim)
	}

	// Case 2: 2D transpose [1, 0]
	if rank == 2 && permutations[0] == 1 && permutations[1] == 0 {
		return dispatchTranspose2D(backend, operand, output, dims[0], dims[1])
	}

	// Case 3: Any rank where trailing dimension is preserved (permutations[rank-1] == rank-1)
	if rank > 2 && permutations[rank-1] == rank-1 {
		return dispatchTransposeTrailing(backend, operand, output, dims, permutations)
	}

	return false
}

func dispatchIdentity(operand, output *gobackend.Buffer) bool {
	switch in := operand.Flat.(type) {
	case []float32:
		copy(output.Flat.([]float32), in)
	case []float64:
		copy(output.Flat.([]float64), in)
	case []bfloat16.BFloat16:
		copy(output.Flat.([]bfloat16.BFloat16), in)
	case []float16.Float16:
		copy(output.Flat.([]float16.Float16), in)
	case []int32:
		copy(output.Flat.([]int32), in)
	case []int64:
		copy(output.Flat.([]int64), in)
	case []int16:
		copy(output.Flat.([]int16), in)
	case []int8:
		copy(output.Flat.([]int8), in)
	case []uint32:
		copy(output.Flat.([]uint32), in)
	case []uint64:
		copy(output.Flat.([]uint64), in)
	case []uint16:
		copy(output.Flat.([]uint16), in)
	case []uint8:
		copy(output.Flat.([]uint8), in)
	case []bool:
		copy(output.Flat.([]bool), in)
	default:
		return false
	}
	return true
}

func dispatchTranspose4D_0213(backend *gobackend.Backend, operand, output *gobackend.Buffer, bDim, sDim, hDim, dDim int) bool {
	switch in := operand.Flat.(type) {
	case []float32:
		transpose4D_0213(backend, in, output.Flat.([]float32), bDim, sDim, hDim, dDim)
	case []float64:
		transpose4D_0213(backend, in, output.Flat.([]float64), bDim, sDim, hDim, dDim)
	case []bfloat16.BFloat16:
		transpose4D_0213(backend, in, output.Flat.([]bfloat16.BFloat16), bDim, sDim, hDim, dDim)
	case []float16.Float16:
		transpose4D_0213(backend, in, output.Flat.([]float16.Float16), bDim, sDim, hDim, dDim)
	case []int32:
		transpose4D_0213(backend, in, output.Flat.([]int32), bDim, sDim, hDim, dDim)
	case []int64:
		transpose4D_0213(backend, in, output.Flat.([]int64), bDim, sDim, hDim, dDim)
	case []int16:
		transpose4D_0213(backend, in, output.Flat.([]int16), bDim, sDim, hDim, dDim)
	case []int8:
		transpose4D_0213(backend, in, output.Flat.([]int8), bDim, sDim, hDim, dDim)
	case []uint32:
		transpose4D_0213(backend, in, output.Flat.([]uint32), bDim, sDim, hDim, dDim)
	case []uint64:
		transpose4D_0213(backend, in, output.Flat.([]uint64), bDim, sDim, hDim, dDim)
	case []uint16:
		transpose4D_0213(backend, in, output.Flat.([]uint16), bDim, sDim, hDim, dDim)
	case []uint8:
		transpose4D_0213(backend, in, output.Flat.([]uint8), bDim, sDim, hDim, dDim)
	case []bool:
		transpose4D_0213(backend, in, output.Flat.([]bool), bDim, sDim, hDim, dDim)
	default:
		return false
	}
	return true
}

func transpose4D_0213[T any](backend *gobackend.Backend, in, out []T, bDim, sDim, hDim, dDim int) {
	totalElements := bDim * sDim * hDim * dDim
	totalOuter := bDim * hDim
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && totalOuter > 1 && totalElements > 16384 {
		numWorkers := backend.Workers.AdjustedMaxParallelism()
		targetChunks := min(totalOuter, max(1, numWorkers*2))
		outerPerChunk := max(1, (totalOuter+targetChunks-1)/targetChunks)
		var wg sync.WaitGroup
		for start := 0; start < totalOuter; start += outerPerChunk {
			end := min(start+outerPerChunk, totalOuter)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				for outer := start; outer < end; outer++ {
					b := outer / hDim
					h := outer % hDim
					outBase := outer * (sDim * dDim)
					inBase := b*(sDim*hDim*dDim) + h*dDim
					strideIn := hDim * dDim
					for s := 0; s < sDim; s++ {
						inOff := inBase + s*strideIn
						outOff := outBase + s*dDim
						copy(out[outOff:outOff+dDim], in[inOff:inOff+dDim])
					}
				}
				wg.Done()
			})
		}
		wg.Wait()
	} else if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && totalOuter == 1 && sDim > 512 && totalElements > 16384 {
		numWorkers := backend.Workers.AdjustedMaxParallelism()
		targetChunks := min(sDim, max(1, numWorkers*2))
		sPerChunk := max(1, (sDim+targetChunks-1)/targetChunks)
		var wg sync.WaitGroup
		for start := 0; start < sDim; start += sPerChunk {
			end := min(start+sPerChunk, sDim)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				strideIn := hDim * dDim
				for s := start; s < end; s++ {
					inOff := s * strideIn
					outOff := s * dDim
					copy(out[outOff:outOff+dDim], in[inOff:inOff+dDim])
				}
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		for outer := 0; outer < totalOuter; outer++ {
			b := outer / hDim
			h := outer % hDim
			outBase := outer * (sDim * dDim)
			inBase := b*(sDim*hDim*dDim) + h*dDim
			strideIn := hDim * dDim
			for s := 0; s < sDim; s++ {
				inOff := inBase + s*strideIn
				outOff := outBase + s*dDim
				copy(out[outOff:outOff+dDim], in[inOff:inOff+dDim])
			}
		}
	}
}

func dispatchTranspose2D(backend *gobackend.Backend, operand, output *gobackend.Buffer, H, W int) bool {
	switch in := operand.Flat.(type) {
	case []float32:
		transpose2D(backend, in, output.Flat.([]float32), H, W)
	case []float64:
		transpose2D(backend, in, output.Flat.([]float64), H, W)
	case []bfloat16.BFloat16:
		transpose2D(backend, in, output.Flat.([]bfloat16.BFloat16), H, W)
	case []float16.Float16:
		transpose2D(backend, in, output.Flat.([]float16.Float16), H, W)
	case []int32:
		transpose2D(backend, in, output.Flat.([]int32), H, W)
	case []int64:
		transpose2D(backend, in, output.Flat.([]int64), H, W)
	case []int16:
		transpose2D(backend, in, output.Flat.([]int16), H, W)
	case []int8:
		transpose2D(backend, in, output.Flat.([]int8), H, W)
	case []uint32:
		transpose2D(backend, in, output.Flat.([]uint32), H, W)
	case []uint64:
		transpose2D(backend, in, output.Flat.([]uint64), H, W)
	case []uint16:
		transpose2D(backend, in, output.Flat.([]uint16), H, W)
	case []uint8:
		transpose2D(backend, in, output.Flat.([]uint8), H, W)
	case []bool:
		transpose2D(backend, in, output.Flat.([]bool), H, W)
	default:
		return false
	}
	return true
}

func transpose2D[T any](backend *gobackend.Backend, in, out []T, H, W int) {
	tileSize := 32
	totalElements := H * W
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && H > 32 && totalElements > 16384 {
		numWorkers := backend.Workers.AdjustedMaxParallelism()
		rTiles := (H + tileSize - 1) / tileSize
		targetChunks := min(rTiles, max(1, numWorkers*2))
		tilesPerChunk := max(1, (rTiles+targetChunks-1)/targetChunks)
		var wg sync.WaitGroup
		for t := 0; t < rTiles; t += tilesPerChunk {
			tStart := t * tileSize
			tEnd := min((t+tilesPerChunk)*tileSize, H)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				for r0 := tStart; r0 < tEnd; r0 += tileSize {
					rLimit := min(r0+tileSize, tEnd)
					for c0 := 0; c0 < W; c0 += tileSize {
						cLimit := min(c0+tileSize, W)
						for r := r0; r < rLimit; r++ {
							for c := c0; c < cLimit; c++ {
								out[c*H+r] = in[r*W+c]
							}
						}
					}
				}
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		for r0 := 0; r0 < H; r0 += tileSize {
			rLimit := min(r0+tileSize, H)
			for c0 := 0; c0 < W; c0 += tileSize {
				cLimit := min(c0+tileSize, W)
				for r := r0; r < rLimit; r++ {
					for c := c0; c < cLimit; c++ {
						out[c*H+r] = in[r*W+c]
					}
				}
			}
		}
	}
}

func dispatchTransposeTrailing(backend *gobackend.Backend, operand, output *gobackend.Buffer, dims, permutations []int) bool {
	dtype := operand.RawShape.DType
	switch in := operand.Flat.(type) {
	case []float32:
		transposeTrailing(backend, in, output.Flat.([]float32), dims, permutations, dtype)
	case []float64:
		transposeTrailing(backend, in, output.Flat.([]float64), dims, permutations, dtype)
	case []bfloat16.BFloat16:
		transposeTrailing(backend, in, output.Flat.([]bfloat16.BFloat16), dims, permutations, dtype)
	case []float16.Float16:
		transposeTrailing(backend, in, output.Flat.([]float16.Float16), dims, permutations, dtype)
	case []int32:
		transposeTrailing(backend, in, output.Flat.([]int32), dims, permutations, dtype)
	case []int64:
		transposeTrailing(backend, in, output.Flat.([]int64), dims, permutations, dtype)
	case []int16:
		transposeTrailing(backend, in, output.Flat.([]int16), dims, permutations, dtype)
	case []int8:
		transposeTrailing(backend, in, output.Flat.([]int8), dims, permutations, dtype)
	case []uint32:
		transposeTrailing(backend, in, output.Flat.([]uint32), dims, permutations, dtype)
	case []uint64:
		transposeTrailing(backend, in, output.Flat.([]uint64), dims, permutations, dtype)
	case []uint16:
		transposeTrailing(backend, in, output.Flat.([]uint16), dims, permutations, dtype)
	case []uint8:
		transposeTrailing(backend, in, output.Flat.([]uint8), dims, permutations, dtype)
	case []bool:
		transposeTrailing(backend, in, output.Flat.([]bool), dims, permutations, dtype)
	default:
		return false
	}
	return true
}

func transposeTrailing[T any](backend *gobackend.Backend, in, out []T, dims, permutations []int, dtype dtypes.DType) {
	rank := len(dims)
	dDim := dims[rank-1]
	outerShape := shapes.Make(dtype, dims[:rank-1]...)
	outerPerm := permutations[:rank-1]
	it := NewTransposeIterator(outerShape, outerPerm)
	outerCount := outerShape.Size()
	for m := 0; m < outerCount; m++ {
		outIdx := it.Next()
		inOff := m * dDim
		outOff := outIdx * dDim
		copy(out[outOff:outOff+dDim], in[inOff:inOff+dDim])
	}
}


// TransposeIterator creates a dynamic iterator that yields output flat indices
// for the corresponding flat index on the input operand, assuming the operand flat index is moving
// incrementally.
type TransposeIterator struct {
	flatIdx                                int
	perAxisIdx, perAxisStrides, dimensions []int
}

// NewTransposeIterator creates a dynamic iterator that yields output flat indices
// for the corresponding flat index on the input operand, assuming the operand flat index is moving
// incrementally.
func NewTransposeIterator(operand shapes.Shape, permutations []int) *TransposeIterator {
	rank := operand.Rank()

	it := &TransposeIterator{
		perAxisIdx:     make([]int, rank),
		perAxisStrides: make([]int, rank),
		dimensions:     slices.Clone(operand.Dimensions),
	}

	// First, calculate strides on the output.
	stridesOnOutput := make([]int, rank)
	stride := 1
	reversePermutations := make([]int, rank)
	for outputAxis := rank - 1; outputAxis >= 0; outputAxis-- {
		stridesOnOutput[outputAxis] = stride
		operandAxis := permutations[outputAxis]
		stride *= operand.Dimensions[operandAxis]
		reversePermutations[operandAxis] = outputAxis
	}

	// Calculate per operand axis, what is the stride on the output.
	for operandAxis := range rank {
		outputAxis := reversePermutations[operandAxis]
		it.perAxisStrides[operandAxis] = stridesOnOutput[outputAxis]
	}
	return it
}

// Next returns the next flat index in the output.
func (it *TransposeIterator) Next() int {
	// Store current flatIdx first
	nextFlatIdx := it.flatIdx

	// Cache rank to avoid repeated len() calls
	rank := len(it.perAxisIdx)

	// Use local variables for array access to avoid repeated indirection
	perAxisIdx := it.perAxisIdx
	perAxisStrides := it.perAxisStrides
	dimensions := it.dimensions

	// Handle remaining axes only when needed
	for axis := rank - 1; axis >= 0; axis-- {
		perAxisIdx[axis]++
		it.flatIdx += perAxisStrides[axis]
		if perAxisIdx[axis] < dimensions[axis] {
			// We are done.
			return nextFlatIdx
		}
		perAxisIdx[axis] = 0
		it.flatIdx -= perAxisStrides[axis] * dimensions[axis]
	}

	return nextFlatIdx
}

func execTransposeGeneric[T gobackend.SupportedTypesConstraints](operand, output *gobackend.Buffer, it *TransposeIterator) {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	for _, value := range operandFlat {
		outputFlat[it.Next()] = value
	}
}

func execTransposeFloat16(operand, output *gobackend.Buffer, it *TransposeIterator) {
	operandFlat := operand.Flat.([]float16.Float16)
	outputFlat := output.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputFlat[it.Next()] = value
	}
}

func execTransposeBFloat16(operand, output *gobackend.Buffer, it *TransposeIterator) {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputFlat[it.Next()] = value
	}
}

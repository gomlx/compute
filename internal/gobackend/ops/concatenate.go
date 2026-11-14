// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterConcatenate.Register(Concatenate, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeConcatenate, gobackend.PriorityGeneric, execConcatenate)
}

// Concatenate joins a sequence of tensors along the given axis (it must exist already).
// All input tensors must have the same shape, except potentially in the concatenation dimension.
// They must also have the same data type (DType).
// It returns an error if inputs are invalid (e.g., no inputs, mismatched graphs, shapes, dtypes, or invalid dimension).
func Concatenate(f *gobackend.Function, axis int, operandOps ...compute.Value) (compute.Value, error) {
	if len(operandOps) == 0 {
		return nil, errors.Errorf("Concatenate requires at least one input tensor")
	}
	operands, err := f.VerifyAndCastValues("Concatenate", operandOps...)
	if err != nil {
		return nil, err
	}

	// Extract shapes for shape inference.
	inputShapes := make([]shapes.Shape, len(operands))
	for i, opNode := range operands {
		inputShapes[i] = opNode.Shape
	}
	outputShape, err := shapeinference.Concatenate(inputShapes, axis)
	if err != nil {
		return nil, err
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeConcatenate, outputShape, operands, axis)
	return node, nil
}

// execConcatenate implements the Concatenate op using direct byte copying with offsets and strides.
func execConcatenate(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	axis := node.Data.(int) // Renamed from dimension
	outputShape := node.Shape
	if outputShape.IsDynamic() {
		concreteShape := outputShape.Clone()
		for d := range concreteShape.Rank() {
			if concreteShape.Dimensions[d] == shapes.DynamicDim {
				if d == axis {
					concatSize := 0
					for _, inputBuf := range inputs {
						concatSize += inputBuf.RawShape.Dimensions[d]
					}
					concreteShape.Dimensions[d] = concatSize
				} else {
					for _, inputBuf := range inputs {
						if inputBuf.RawShape.Dimensions[d] != shapes.DynamicDim {
							concreteShape.Dimensions[d] = inputBuf.RawShape.Dimensions[d]
							break
						}
					}
				}
			}
		}
		outputShape = concreteShape
	}
	rank := outputShape.Rank()
	for i, inputBuf := range inputs {
		for d := range rank {
			if d != axis && inputBuf.RawShape.Dimensions[d] != outputShape.Dimensions[d] {
				return nil, errors.Errorf(
					"Concatenate runtime dimension mismatch at axis %d: expected %d, input #%d has %d (shapes %s vs %s)",
					d, outputShape.Dimensions[d], i, inputBuf.RawShape.Dimensions[d], outputShape, inputBuf.RawShape)
			}
		}
	}
	dtype := outputShape.DType
	elemSize := dtype.Size()
	_ = inputsOwned // We don't reuse the inputs.

	// Allocate output buffer.
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || outputShape.Size() == 0 {
		return output, nil
	}
	outputBytes, err := output.MutableBytes()
	if err != nil || len(outputBytes) == 0 {
		return output, err
	}

	// Calculate the size of the blocks before and after the concatenation axis.
	outerBlockSize := 1 // Number of independent blocks to copy
	for i := range axis {
		outerBlockSize *= outputShape.Dimensions[i]
	}
	innerBlockSize := 1 // Size of the innermost contiguous block (in elements)
	for i := axis + 1; i < rank; i++ {
		innerBlockSize *= outputShape.Dimensions[i]
	}
	innerBlockBytes := innerBlockSize * elemSize

	// Total size in bytes of one full "row" along the concatenation axis in the output.
	// This is the stride needed to jump from one outer block to the next in the output.
	outputConcatAxisStrideBytes := outputShape.Dimensions[axis] * innerBlockBytes

	type inputInfo struct {
		bytes      []byte
		blockBytes int
		outOffset  int
	}
	var stackInfos [8]inputInfo
	var infos []inputInfo
	if len(inputs) <= len(stackInfos) {
		infos = stackInfos[:0]
	} else {
		infos = make([]inputInfo, 0, len(inputs))
	}

	outputAxisOffsetBytes := 0
	for _, inputBuf := range inputs {
		inputBlockBytes := inputBuf.RawShape.Dimensions[axis] * innerBlockBytes
		if inputBlockBytes == 0 {
			continue
		}
		inputBytes, err := inputBuf.MutableBytes()
		if err != nil {
			return nil, err
		}
		infos = append(infos, inputInfo{
			bytes:      inputBytes,
			blockBytes: inputBlockBytes,
			outOffset:  outputAxisOffsetBytes,
		})
		outputAxisOffsetBytes += inputBlockBytes
	}

	if len(infos) == 0 {
		return output, nil
	}

	canParallelize := len(outputBytes) >= minParallelCopyBytes &&
		backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled()

	// Fast path 0: single non-empty input is a flat contiguous copy.
	if len(infos) == 1 {
		src := infos[0].bytes[:len(outputBytes)]
		if canParallelize {
			backend.Scheduler.ParallelizeChunks(len(outputBytes), minParallelCopyChunkBytes, func(start, end int) {
				copy(outputBytes[start:end], src[start:end])
			})
		} else {
			copy(outputBytes, src)
		}
		return output, nil
	}

	// Fast path 1: outerBlockSize == 1 (concatenating along leading axis).
	if outerBlockSize == 1 {
		if canParallelize {
			backend.Scheduler.ParallelizeChunks(len(outputBytes), minParallelCopyChunkBytes, func(startByte, endByte int) {
				for _, info := range infos {
					s := max(startByte, info.outOffset)
					e := min(endByte, info.outOffset+info.blockBytes)
					if s < e {
						copy(outputBytes[s:e], info.bytes[s-info.outOffset:e-info.outOffset])
					}
				}
			})
		} else {
			for _, info := range infos {
				copy(outputBytes[info.outOffset:info.outOffset+info.blockBytes], info.bytes[:info.blockBytes])
			}
		}
		return output, nil
	}

	// Fast path 2: few outer blocks with large inner blocks — sub-chunk along the concatenation stride
	// so all worker threads participate.
	if canParallelize && outerBlockSize < 16 && outputConcatAxisStrideBytes >= minParallelCopyBytes {
		subChunksPerOuter := (outputConcatAxisStrideBytes + minParallelCopyChunkBytes - 1) / minParallelCopyChunkBytes
		totalSubChunks := outerBlockSize * subChunksPerOuter
		backend.Scheduler.ParallelizeChunks(totalSubChunks, 1, func(startSub, endSub int) {
			for sc := startSub; sc < endSub; sc++ {
				outerIdx := sc / subChunksPerOuter
				sub := sc % subChunksPerOuter
				subStart := sub * minParallelCopyChunkBytes
				subEnd := min(outputConcatAxisStrideBytes, subStart+minParallelCopyChunkBytes)
				outBase := outerIdx * outputConcatAxisStrideBytes
				for _, info := range infos {
					s := max(subStart, info.outOffset)
					e := min(subEnd, info.outOffset+info.blockBytes)
					if s < e {
						inOff := outerIdx*info.blockBytes + (s - info.outOffset)
						copy(outputBytes[outBase+s:outBase+e], info.bytes[inOff:inOff+(e-s)])
					}
				}
			}
		})
		return output, nil
	}

	copyOuterRange := func(startOuter, endOuter int) {
		count := endOuter - startOuter
		if count <= 0 {
			return
		}
		if len(infos) == 2 {
			b0 := infos[0].blockBytes
			b1 := infos[1].blockBytes
			concat2StridedBytes(
				outputBytes[startOuter*outputConcatAxisStrideBytes:],
				infos[0].bytes[startOuter*b0:],
				infos[1].bytes[startOuter*b1:],
				count, b0, b1,
			)
			return
		}

		tileRows := max(16, min(256, 16384/max(1, outputConcatAxisStrideBytes)))
		for r := startOuter; r < endOuter; r += tileRows {
			tRows := min(tileRows, endOuter-r)
			outBase := r * outputConcatAxisStrideBytes
			for _, info := range infos {
				copyStridedBytes(
					outputBytes[outBase+info.outOffset:],
					info.bytes[r*info.blockBytes:],
					tRows, info.blockBytes, outputConcatAxisStrideBytes, info.blockBytes,
				)
			}
		}
	}

	if canParallelize {
		minOuterChunk := max(1, (minParallelCopyChunkBytes+outputConcatAxisStrideBytes-1)/outputConcatAxisStrideBytes)
		backend.Scheduler.ParallelizeChunks(outerBlockSize, minOuterChunk, copyOuterRange)
	} else {
		copyOuterRange(0, outerBlockSize)
	}

	return output, nil
}

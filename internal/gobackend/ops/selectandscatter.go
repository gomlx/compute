// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"slices"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
)

func init() {
	gobackend.RegisterSelectAndScatterMax.Register(SelectAndScatterMax, gobackend.PriorityGeneric)
	gobackend.RegisterSelectAndScatterMin.Register(SelectAndScatterMin, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeSelectAndScatterMax, gobackend.PriorityGeneric, execSelectAndScatter)
	gobackend.SetNodeExecutor(compute.OpTypeSelectAndScatterMin, gobackend.PriorityGeneric, execSelectAndScatter)
}

type selectAndScatterNode struct {
	isMin                     bool
	windowDimensions, strides []int
	paddings                  [][2]int
}

// EqualNodeData implements nodeDataComparable for selectAndScatterNode.
func (s *selectAndScatterNode) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*selectAndScatterNode)
	if s.isMin != o.isMin {
		return false
	}
	return slices.Equal(s.windowDimensions, o.windowDimensions) &&
		slices.Equal(s.strides, o.strides) &&
		slices.Equal(s.paddings, o.paddings)
}

// SelectAndScatterMax runs windows over operand, selects maximum elements in each window, and scatters source values to the output.
func SelectAndScatterMax(f *gobackend.Function,
	operandOp, sourceOp compute.Value,
	windowDimensions, strides []int,
	paddings [][2]int,
) (compute.Value, error) {
	return selectAndScatterImpl(f, compute.OpTypeSelectAndScatterMax, operandOp, sourceOp, windowDimensions, strides, paddings, false)
}

// SelectAndScatterMin runs windows over operand, selects minimum elements in each window, and scatters source values to the output.
func SelectAndScatterMin(f *gobackend.Function,
	operandOp, sourceOp compute.Value,
	windowDimensions, strides []int,
	paddings [][2]int,
) (compute.Value, error) {
	return selectAndScatterImpl(f, compute.OpTypeSelectAndScatterMin, operandOp, sourceOp, windowDimensions, strides, paddings, true)
}

func selectAndScatterImpl(f *gobackend.Function,
	opType compute.OpType,
	operandOp, sourceOp compute.Value,
	windowDimensions, strides []int,
	paddings [][2]int,
	isMin bool,
) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp, sourceOp)
	if err != nil {
		return nil, err
	}
	operand, source := inputs[0], inputs[1]
	outputShape, err := shapeinference.SelectAndScatter(
		operand.Shape,
		source.Shape,
		windowDimensions,
		strides,
		paddings,
	)
	if err != nil {
		return nil, err
	}
	data := &selectAndScatterNode{
		isMin:            isMin,
		windowDimensions: windowDimensions,
		strides:          strides,
		paddings:         paddings,
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*gobackend.Node{operand, source}, data)
	return node, nil
}

func execSelectAndScatter(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	source := inputs[1]
	operandShape := operand.RawShape
	rank := operandShape.Rank()
	dtype := operandShape.DType
	outputShape := node.Shape
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	output.Zeros()

	opData := node.Data.(*selectAndScatterNode)

	effWindowDimensions := opData.windowDimensions
	if effWindowDimensions == nil {
		effWindowDimensions = xslices.SliceWithValue(rank, 1)
	}
	effStrides := opData.strides
	if effStrides == nil {
		effStrides = effWindowDimensions
	}
	effPaddings := opData.paddings
	if effPaddings == nil {
		effPaddings = xslices.SliceWithValue(rank, [2]int{0, 0})
	}

	execFnAny, err := selectAndScatterDTypeMap.Get(dtype)
	if err != nil {
		return nil, err
	}
	execFn := execFnAny.(func(backend *gobackend.Backend, operand, source, output *gobackend.Buffer, effWindowDimensions, effStrides []int, effPaddings [][2]int, isMin bool))
	execFn(backend, operand, source, output, effWindowDimensions, effStrides, effPaddings, opData.isMin)
	return output, nil
}

var (
	//gobackend:dtypemap execSelectAndScatterGeneric ints,uints,floats
	//gobackend:dtypemap execSelectAndScatterGenericHalf half
	selectAndScatterDTypeMap = gobackend.NewDTypeMap("selectAndScatterDTypeMap")
)

func execSelectAndScatterGeneric[T gobackend.PODNumericConstraints](
	backend *gobackend.Backend,
	operand, source, output *gobackend.Buffer,
	effWindowDimensions, effStrides []int,
	effPaddings [][2]int,
	isMin bool,
) {
	operandFlat := operand.Flat.([]T)
	sourceFlat := source.Flat.([]T)
	outputFlat := output.Flat.([]T)

	operandShape := operand.RawShape
	sourceShape := source.RawShape
	rank := operandShape.Rank()
	dtype := operandShape.DType

	windowShape := shapes.Make(dtype, effWindowDimensions...)
	windowSize := windowShape.Size()

	// Fast path for 1D/2D ChannelsLast unpooling where batch and channel windows/strides are 1.
	if (rank == 3 || rank == 4) &&
		effWindowDimensions[0] == 1 && effWindowDimensions[rank-1] == 1 &&
		effStrides[0] == 1 && effStrides[rank-1] == 1 &&
		effPaddings[0][0] == 0 && effPaddings[0][1] == 0 &&
		effPaddings[rank-1][0] == 0 && effPaddings[rank-1][1] == 0 {
		execSelectAndScatterChannelsLastGeneric(
			backend, operandFlat, sourceFlat, outputFlat,
			operandShape.Dimensions, sourceShape.Dimensions,
			effWindowDimensions, effStrides, effPaddings, isMin,
		)
		return
	}

	operandStrides := make([]int, rank)
	stride := 1
	for axis := rank - 1; axis >= 0; axis-- {
		operandStrides[axis] = stride
		stride *= operandShape.Dimensions[axis]
	}

	sourceStrides := sourceShape.Strides()
	sourceDimensions := sourceShape.Dimensions

	runChunk := func(startFlatIdx, endFlatIdx int, targetOutput []T) {
		sourceIndices := make([]int, rank)
		windowIndices := make([]int, rank)

		remainder := startFlatIdx
		for i := 0; i < rank; i++ {
			sourceIndices[i] = remainder / sourceStrides[i]
			remainder %= sourceStrides[i]
		}

		for sourceFlatIdx := startFlatIdx; sourceFlatIdx < endFlatIdx; sourceFlatIdx++ {
			bestOperandFlatIdx := -1
			var bestOperandVal T
			clear(windowIndices)
			for wStep := 0; wStep < windowSize; wStep++ {
				valid := true
				operandFlatIdx := 0
				for axis := range rank {
					operandIdx := sourceIndices[axis]*effStrides[axis] - effPaddings[axis][0] + windowIndices[axis]
					if operandIdx < 0 || operandIdx >= operandShape.Dimensions[axis] {
						valid = false
						break
					}
					operandFlatIdx += operandIdx * operandStrides[axis]
				}
				if valid {
					val := operandFlat[operandFlatIdx]
					if bestOperandFlatIdx == -1 {
						bestOperandFlatIdx = operandFlatIdx
						bestOperandVal = val
					} else if isMin {
						valIsNaN := val != val
						if val < bestOperandVal || valIsNaN {
							bestOperandVal = val
							bestOperandFlatIdx = operandFlatIdx
						}
					} else {
						valIsNaN := val != val
						if val > bestOperandVal || valIsNaN {
							bestOperandVal = val
							bestOperandFlatIdx = operandFlatIdx
						}
					}
				}
				for axis := rank - 1; axis >= 0; axis-- {
					windowIndices[axis]++
					if windowIndices[axis] < effWindowDimensions[axis] {
						break
					}
					windowIndices[axis] = 0
				}
			}
			if bestOperandFlatIdx != -1 {
				targetOutput[bestOperandFlatIdx] += sourceFlat[sourceFlatIdx]
			}

			for i := rank - 1; i >= 0; i-- {
				sourceIndices[i]++
				if sourceIndices[i] < sourceDimensions[i] {
					break
				}
				sourceIndices[i] = 0
			}
		}
	}

	lenSource := len(sourceFlat)
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && lenSource >= 512 {
		maxWorkers := backend.Scheduler.AdjustedMaxParallelism()
		if maxWorkers > 1 {
			if rank >= 2 && effWindowDimensions[0] == 1 && effStrides[0] >= 1 && effPaddings[0][0] == 0 && effPaddings[0][1] == 0 && sourceDimensions[0] > 1 {
				batchCount := sourceDimensions[0]
				elementsPerBatch := lenSource / batchCount
				backend.Scheduler.ParallelizeChunks(batchCount, 1, func(startBatch, endBatch int) {
					runChunk(startBatch*elementsPerBatch, endBatch*elementsPerBatch, outputFlat)
				})
				return
			} else {
				numChunks := min(maxWorkers, (lenSource+255)/256)
				chunkSize := (lenSource + numChunks - 1) / numChunks
				var mu sync.Mutex
				tasks := make([]func(), 0, numChunks)
				for c := 0; c < numChunks; c++ {
					startIdx := c * chunkSize
					if startIdx >= lenSource {
						break
					}
					endIdx := min(startIdx+chunkSize, lenSource)
					isMain := (c == 0)
					tasks = append(tasks, func() {
						if isMain {
							runChunk(startIdx, endIdx, outputFlat)
						} else {
							localOutput := make([]T, len(outputFlat))
							runChunk(startIdx, endIdx, localOutput)
							mu.Lock()
							for i, v := range localOutput {
								if v != 0 {
									outputFlat[i] += v
								}
							}
							mu.Unlock()
						}
					})
				}
				backend.Scheduler.RunIntraOpTasks(tasks)
				return
			}
		}
	}

	runChunk(0, lenSource, outputFlat)
}

func execSelectAndScatterGenericHalf[T gotype.HalfPrecision[T], P gotype.HalfPrecisionPtr[T]](
	backend *gobackend.Backend,
	operand, source, output *gobackend.Buffer,
	effWindowDimensions, effStrides []int,
	effPaddings [][2]int,
	isMin bool,
) {
	operandFlat := operand.Flat.([]T)
	sourceFlat := source.Flat.([]T)
	outputFlat := output.Flat.([]T)

	operandShape := operand.RawShape
	sourceShape := source.RawShape
	rank := operandShape.Rank()
	dtype := operandShape.DType

	windowShape := shapes.Make(dtype, effWindowDimensions...)
	windowSize := windowShape.Size()

	operandStrides := make([]int, rank)
	stride := 1
	for axis := rank - 1; axis >= 0; axis-- {
		operandStrides[axis] = stride
		stride *= operandShape.Dimensions[axis]
	}

	sourceStrides := sourceShape.Strides()
	sourceDimensions := sourceShape.Dimensions

	runChunk := func(startFlatIdx, endFlatIdx int, targetOutput []T) {
		sourceIndices := make([]int, rank)
		windowIndices := make([]int, rank)

		remainder := startFlatIdx
		for i := 0; i < rank; i++ {
			sourceIndices[i] = remainder / sourceStrides[i]
			remainder %= sourceStrides[i]
		}

		for sourceFlatIdx := startFlatIdx; sourceFlatIdx < endFlatIdx; sourceFlatIdx++ {
			bestOperandFlatIdx := -1
			var bestOperandVal float32
			clear(windowIndices)
			for wStep := 0; wStep < windowSize; wStep++ {
				valid := true
				operandFlatIdx := 0
				for axis := range rank {
					operandIdx := sourceIndices[axis]*effStrides[axis] - effPaddings[axis][0] + windowIndices[axis]
					if operandIdx < 0 || operandIdx >= operandShape.Dimensions[axis] {
						valid = false
						break
					}
					operandFlatIdx += operandIdx * operandStrides[axis]
				}
				if valid {
					val := operandFlat[operandFlatIdx].Float32()
					if bestOperandFlatIdx == -1 {
						bestOperandFlatIdx = operandFlatIdx
						bestOperandVal = val
					} else if isMin {
						valIsNaN := val != val
						if val < bestOperandVal || valIsNaN {
							bestOperandVal = val
							bestOperandFlatIdx = operandFlatIdx
						}
					} else {
						valIsNaN := val != val
						if val > bestOperandVal || valIsNaN {
							bestOperandVal = val
							bestOperandFlatIdx = operandFlatIdx
						}
					}
				}
				for axis := rank - 1; axis >= 0; axis-- {
					windowIndices[axis]++
					if windowIndices[axis] < effWindowDimensions[axis] {
						break
					}
					windowIndices[axis] = 0
				}
			}
			if bestOperandFlatIdx != -1 {
				sum := targetOutput[bestOperandFlatIdx].Float32() + sourceFlat[sourceFlatIdx].Float32()
				P(&targetOutput[bestOperandFlatIdx]).SetFloat32(sum)
			}

			for i := rank - 1; i >= 0; i-- {
				sourceIndices[i]++
				if sourceIndices[i] < sourceDimensions[i] {
					break
				}
				sourceIndices[i] = 0
			}
		}
	}

	lenSource := len(sourceFlat)
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && lenSource >= 512 {
		maxWorkers := backend.Scheduler.AdjustedMaxParallelism()
		if maxWorkers > 1 {
			if rank >= 2 && effWindowDimensions[0] == 1 && effStrides[0] >= 1 && effPaddings[0][0] == 0 && effPaddings[0][1] == 0 && sourceDimensions[0] > 1 {
				batchCount := sourceDimensions[0]
				elementsPerBatch := lenSource / batchCount
				backend.Scheduler.ParallelizeChunks(batchCount, 1, func(startBatch, endBatch int) {
					runChunk(startBatch*elementsPerBatch, endBatch*elementsPerBatch, outputFlat)
				})
				return
			} else {
				numChunks := min(maxWorkers, (lenSource+255)/256)
				chunkSize := (lenSource + numChunks - 1) / numChunks
				var mu sync.Mutex
				tasks := make([]func(), 0, numChunks)
				for c := 0; c < numChunks; c++ {
					startIdx := c * chunkSize
					if startIdx >= lenSource {
						break
					}
					endIdx := min(startIdx+chunkSize, lenSource)
					isMain := (c == 0)
					tasks = append(tasks, func() {
						if isMain {
							runChunk(startIdx, endIdx, outputFlat)
						} else {
							localOutput := make([]T, len(outputFlat))
							runChunk(startIdx, endIdx, localOutput)
							mu.Lock()
							for i, v := range localOutput {
								if v.Float32() != 0 {
									sum := outputFlat[i].Float32() + v.Float32()
									P(&outputFlat[i]).SetFloat32(sum)
								}
							}
							mu.Unlock()
						}
					})
				}
				backend.Scheduler.RunIntraOpTasks(tasks)
				return
			}
		}
	}

	runChunk(0, lenSource, outputFlat)
}

func execSelectAndScatterChannelsLastGeneric[T gobackend.PODNumericConstraints](
	backend *gobackend.Backend,
	operandFlat, sourceFlat, outputFlat []T,
	inDims, srcDims []int,
	effWindowDimensions, effStrides []int,
	effPaddings [][2]int,
	isMin bool,
) {
	rank := len(inDims)
	batchSize := inDims[0]
	channels := inDims[rank-1]
	if batchSize == 0 || channels == 0 || len(sourceFlat) == 0 {
		return
	}

	var inH, inW, srcH, srcW, winH, winW, strideH, strideW, padTop, padLeft int
	if rank == 3 {
		inH, inW = 1, inDims[1]
		srcH, srcW = 1, srcDims[1]
		winH, winW = 1, effWindowDimensions[1]
		strideH, strideW = 1, effStrides[1]
		padTop, padLeft = 0, effPaddings[1][0]
	} else {
		inH, inW = inDims[1], inDims[2]
		srcH, srcW = srcDims[1], srcDims[2]
		winH, winW = effWindowDimensions[1], effWindowDimensions[2]
		strideH, strideW = effStrides[1], effStrides[2]
		padTop, padLeft = effPaddings[1][0], effPaddings[2][0]
	}

	inRowStride := inW * channels
	inBatchStride := inH * inRowStride
	srcRowStride := srcW * channels
	srcBatchStride := srcH * srcRowStride

	runBatches := func(startBatch, endBatch int) {
		for b := startBatch; b < endBatch; b++ {
			inBatchBase := b * inBatchStride
			srcBatchBase := b * srcBatchStride

			for sy := range srcH {
				inY0 := sy*strideH - padTop
				wyStart := max(0, -inY0)
				wyEnd := min(winH, inH-inY0)
				if wyStart >= wyEnd {
					continue
				}
				srcRowBase := srcBatchBase + sy*srcRowStride

				for sx := range srcW {
					inX0 := sx*strideW - padLeft
					wxStart := max(0, -inX0)
					wxEnd := min(winW, inW-inX0)
					if wxStart >= wxEnd {
						continue
					}
					srcPixelBase := srcRowBase + sx*channels
					firstRowBase := inBatchBase + (inY0+wyStart)*inRowStride + (inX0+wxStart)*channels

					if isMin {
						for c := range channels {
							bestIdx := firstRowBase + c
							bestVal := operandFlat[bestIdx]
							for wy := wyStart; wy < wyEnd; wy++ {
								rowBase := inBatchBase + (inY0+wy)*inRowStride
								for wx := wxStart; wx < wxEnd; wx++ {
									idx := rowBase + (inX0+wx)*channels + c
									val := operandFlat[idx]
									if val < bestVal || val != val {
										bestVal = val
										bestIdx = idx
									}
								}
							}
							outputFlat[bestIdx] += sourceFlat[srcPixelBase+c]
						}
					} else {
						for c := range channels {
							bestIdx := firstRowBase + c
							bestVal := operandFlat[bestIdx]
							for wy := wyStart; wy < wyEnd; wy++ {
								rowBase := inBatchBase + (inY0+wy)*inRowStride
								for wx := wxStart; wx < wxEnd; wx++ {
									idx := rowBase + (inX0+wx)*channels + c
									val := operandFlat[idx]
									if val > bestVal || val != val {
										bestVal = val
										bestIdx = idx
									}
								}
							}
							outputFlat[bestIdx] += sourceFlat[srcPixelBase+c]
						}
					}
				}
			}
		}
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && batchSize > 1 && len(sourceFlat) >= 512 {
		backend.Scheduler.ParallelizeChunks(batchSize, 1, runBatches)
		return
	}
	runBatches(0, batchSize)
}

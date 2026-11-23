package ops

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterArgMinMax.Register(ArgMinMax, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeArgMinMax, gobackend.PriorityGeneric, execArgMinMax)
}

// argMinMaxNode with the axis and isMin fields.
type argMinMaxNode struct {
	axis  int
	isMin bool
}

// EqualNodeData implements nodeDataComparable for argMinMaxNode.
func (a *argMinMaxNode) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*argMinMaxNode)
	return a.axis == o.axis && a.isMin == o.isMin
}

// ArgMinMax calculates the "argmin" or "argmax" across an axis of the given input array x.
// outputDType defines the output of the argmin/argmax, it doesn't need to be the same as the input.
// It's a form of reduction on the given axis, and that axis goes away. So the rank of the result is one less than
// the rank of x.
// Examples:
//
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=1, isMin=true) -> {1, 0}  // (it chooses the 0 and the -3)
//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=0, isMin=false) -> {0, 1, 0} // (it choose the 2, 4 and 7)
func ArgMinMax(f *gobackend.Function,
	operandOp compute.Value,
	axis int,
	outputDType dtypes.DType,
	isMin bool,
) (compute.Value, error) {
	opType := compute.OpTypeArgMinMax
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ArgMinMax(operand.Shape, axis, outputDType)
	if err != nil {
		return nil, err
	}
	data := &argMinMaxNode{
		axis,
		isMin,
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*gobackend.Node{operand}, data)
	return node, nil
}

const MaxArgMinMaxReductionSize int64 = 0x8000_0000

// execArgMinMax is the executor function registered for compute.OpTypeArgMinMax.
func execArgMinMax(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	reduceAxis := node.Data.(*argMinMaxNode).axis
	isMin := node.Data.(*argMinMaxNode).isMin
	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}

	// There are 3 sizes to iterate over: before and after the reduceAxis, and the size (dimension) of the reduced axis itself.
	operandDims := operand.RawShape.Dimensions
	operandRank := len(operandDims)
	suffixSize := 1
	for axis := reduceAxis + 1; axis < operandRank; axis++ {
		suffixSize *= operandDims[axis]
	}
	prefixSize := 1
	for axis := range reduceAxis {
		prefixSize *= operand.RawShape.Dimensions[axis]
	}
	reduceSize := operandDims[reduceAxis]
	if int64(reduceSize) >= MaxArgMinMaxReductionSize {
		// If we need larger, change buildArgMinMax to use int64 instead of int32.
		return nil, errors.Errorf("ArgMaxMin implementation only supports reduction on dimensions < %d, got operand shaped %s and reduce axis is %d",
			MaxArgMinMaxReductionSize, operand.RawShape, reduceAxis)
	}

	// Instantiate the function to copy over results from ints:
	tmpAny, tmpErr := argMinMaxCopyIntsDTypeMap.Get(output.RawShape.DType)
	if tmpErr != nil {
		return nil, tmpErr
	}
	buildCopyIntsFn := tmpAny.(func(output *gobackend.Buffer) func(flatIdx int, values []int32))
	copyIntsFn := buildCopyIntsFn(output)

	// Dispatch to the generic implementation based on DType.
	tmpAny2, tmpErr2 := argMinMaxDTypeMap.Get(operand.RawShape.DType)
	if tmpErr2 != nil {
		return nil, tmpErr2
	}
	argMinMaxFn := tmpAny2.(func(backend *gobackend.Backend, operand *gobackend.Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize, suffixSize int, isMin bool))
	argMinMaxFn(backend, operand, copyIntsFn, prefixSize, reduceSize, suffixSize, isMin)
	return output, nil
}

var (
	//gobackend:dtypemap execArgMinMaxGeneric ints,uints,floats
	//gobackend:dtypemap execArgMinMaxGenericHalf half
	argMinMaxDTypeMap = gobackend.NewDTypeMap("ArgMinMaxRun")
	//gobackend:dtypemap buildArgMinMaxCopyIntsFn ints,uints
	argMinMaxCopyIntsDTypeMap = gobackend.NewDTypeMap("ArgMinMaxCopyInts")
)

// buildArgMinMaxCopyIntsFn creates a "copyInts" function to copy the given values starting at the flatIdx to
// the output buffer.
func buildArgMinMaxCopyIntsFn[T gobackend.PODIntegerConstraints](output *gobackend.Buffer) func(flatIdx int, values []int32) {
	outputFlat := output.Flat.([]T)
	return func(flatIdx int, values []int32) {
		for _, value := range values {
			outputFlat[flatIdx] = T(value)
			flatIdx++
		}
	}
}

const (
	argMinMaxParallelChunkSize = 16384
	argMinMaxMinParallelElems  = 32768
)

// execArgMinMaxGenericSuffix1 is the fast path when reducing the innermost contiguous axis (suffixSize == 1).
func execArgMinMaxGenericSuffix1[T gobackend.PODNumericConstraints](
	backend *gobackend.Backend, operandFlat []T, copyIntsFn func(flatIdx int, values []int32),
	prefixSize, reduceSize int, isMin bool) {
	if reduceSize == 0 {
		return
	}
	canParallelReduce := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		!backend.Scheduler.IsBusy() && backend.Scheduler.EffectiveParallelism() > 1 &&
		reduceSize >= argMinMaxMinParallelElems

	var single [1]int32
	for p := range prefixSize {
		row := operandFlat[p*reduceSize : (p+1)*reduceSize]
		if canParallelReduce {
			numChunks := (reduceSize + argMinMaxParallelChunkSize - 1) / argMinMaxParallelChunkSize
			chunkVals := make([]T, numChunks)
			chunkIdxs := make([]int32, numChunks)
			backend.Scheduler.ParallelizeChunks(numChunks, 1, func(cStart, cEnd int) {
				for c := cStart; c < cEnd; c++ {
					lo := c * argMinMaxParallelChunkSize
					hi := min(lo+argMinMaxParallelChunkSize, reduceSize)
					bestVal := row[lo]
					bestIdx := int32(lo)
					if isMin {
						for idx := lo + 1; idx < hi; idx++ {
							v := row[idx]
							if v < bestVal || v != v {
								bestVal = v
								bestIdx = int32(idx)
							}
						}
					} else {
						for idx := lo + 1; idx < hi; idx++ {
							v := row[idx]
							if v > bestVal || v != v {
								bestVal = v
								bestIdx = int32(idx)
							}
						}
					}
					chunkVals[c] = bestVal
					chunkIdxs[c] = bestIdx
				}
			})
			bestVal := chunkVals[0]
			bestIdx := chunkIdxs[0]
			if isMin {
				for c := 1; c < numChunks; c++ {
					v := chunkVals[c]
					if v < bestVal || v != v {
						bestVal = v
						bestIdx = chunkIdxs[c]
					}
				}
			} else {
				for c := 1; c < numChunks; c++ {
					v := chunkVals[c]
					if v > bestVal || v != v {
						bestVal = v
						bestIdx = chunkIdxs[c]
					}
				}
			}
			single[0] = bestIdx
			copyIntsFn(p, single[:])
			continue
		}

		bestVal := row[0]
		bestIdx := int32(0)
		if isMin {
			for idx := 1; idx < reduceSize; idx++ {
				v := row[idx]
				if v < bestVal || v != v {
					bestVal = v
					bestIdx = int32(idx)
				}
			}
		} else {
			for idx := 1; idx < reduceSize; idx++ {
				v := row[idx]
				if v > bestVal || v != v {
					bestVal = v
					bestIdx = int32(idx)
				}
			}
		}
		single[0] = bestIdx
		copyIntsFn(p, single[:])
	}
}

// TODO: handle the error condition.
func execArgMinMaxGeneric[T gobackend.PODNumericConstraints](
	backend *gobackend.Backend, operand *gobackend.Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize,
	suffixSize int, isMin bool) {
	operandFlat := operand.Flat.([]T)
	if suffixSize == 1 {
		execArgMinMaxGenericSuffix1(backend, operandFlat, copyIntsFn, prefixSize, reduceSize, isMin)
		return
	}

	// Temporary data to store argMax results, so we can traverse the operand sequentially.
	currentBestBuffer, _ := backend.GetBuffer(shapes.Make(operand.RawShape.DType, suffixSize))

	currentBest := currentBestBuffer.Flat.([]T)
	currentArgBestBuffer, _ := backend.GetBuffer(shapes.Make(dtypes.Int32, suffixSize))
	currentArgBest := currentArgBestBuffer.Flat.([]int32)

	outputFlatIdx := 0
	operandFlatIdx := 0
	for range prefixSize {
		// Initialize the current best with the first element of the reduced axis:
		for suffixIdx := range suffixSize {
			currentBest[suffixIdx] = operandFlat[operandFlatIdx]
			currentArgBest[suffixIdx] = 0
			operandFlatIdx++
		}

		// Iterate over the rest of the elements of reduce axis:
		if isMin {
			// ArgMin
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx]
					operandFlatIdx++
					operandValueIsNaN := operandValue != operandValue
					if operandValue < currentBest[suffixIdx] || operandValueIsNaN {
						currentBest[suffixIdx] = operandValue
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
				}
			}
		} else {
			// ArgMax
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx]
					operandFlatIdx++
					operandValueIsNaN := operandValue != operandValue
					if operandValue > currentBest[suffixIdx] || operandValueIsNaN {
						currentBest[suffixIdx] = operandValue
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
				}
			}
		}

		// Copy over the result of the whole suffix.
		copyIntsFn(outputFlatIdx, currentArgBest)
		outputFlatIdx += suffixSize
	}
	backend.PutBuffer(currentBestBuffer)
	backend.PutBuffer(currentArgBestBuffer)
}

func execArgMinMaxBFloat16Suffix1(
	backend *gobackend.Backend, operandFlat []bfloat16.BFloat16, copyIntsFn func(flatIdx int, values []int32),
	prefixSize, reduceSize int, isMin bool) {
	if reduceSize == 0 {
		return
	}
	canParallelReduce := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		!backend.Scheduler.IsBusy() && backend.Scheduler.EffectiveParallelism() > 1 &&
		reduceSize >= argMinMaxMinParallelElems

	var single [1]int32
	for p := range prefixSize {
		row := operandFlat[p*reduceSize : (p+1)*reduceSize]
		if canParallelReduce {
			numChunks := (reduceSize + argMinMaxParallelChunkSize - 1) / argMinMaxParallelChunkSize
			chunkVals := make([]float32, numChunks)
			chunkIdxs := make([]int32, numChunks)
			backend.Scheduler.ParallelizeChunks(numChunks, 1, func(cStart, cEnd int) {
				for c := cStart; c < cEnd; c++ {
					lo := c * argMinMaxParallelChunkSize
					hi := min(lo+argMinMaxParallelChunkSize, reduceSize)
					bestVal := math.Float32frombits(uint32(row[lo]) << 16)
					bestIdx := int32(lo)
					if isMin {
						for idx := lo + 1; idx < hi; idx++ {
							v := math.Float32frombits(uint32(row[idx]) << 16)
							if v < bestVal {
								bestVal = v
								bestIdx = int32(idx)
							}
						}
					} else {
						for idx := lo + 1; idx < hi; idx++ {
							v := math.Float32frombits(uint32(row[idx]) << 16)
							if v > bestVal {
								bestVal = v
								bestIdx = int32(idx)
							}
						}
					}
					chunkVals[c] = bestVal
					chunkIdxs[c] = bestIdx
				}
			})
			bestVal := chunkVals[0]
			bestIdx := chunkIdxs[0]
			if isMin {
				for c := 1; c < numChunks; c++ {
					if chunkVals[c] < bestVal {
						bestVal = chunkVals[c]
						bestIdx = chunkIdxs[c]
					}
				}
			} else {
				for c := 1; c < numChunks; c++ {
					if chunkVals[c] > bestVal {
						bestVal = chunkVals[c]
						bestIdx = chunkIdxs[c]
					}
				}
			}
			single[0] = bestIdx
			copyIntsFn(p, single[:])
			continue
		}

		bestVal := math.Float32frombits(uint32(row[0]) << 16)
		bestIdx := int32(0)
		if isMin {
			for idx := 1; idx < reduceSize; idx++ {
				v := math.Float32frombits(uint32(row[idx]) << 16)
				if v < bestVal {
					bestVal = v
					bestIdx = int32(idx)
				}
			}
		} else {
			for idx := 1; idx < reduceSize; idx++ {
				v := math.Float32frombits(uint32(row[idx]) << 16)
				if v > bestVal {
					bestVal = v
					bestIdx = int32(idx)
				}
			}
		}
		single[0] = bestIdx
		copyIntsFn(p, single[:])
	}
}

func execArgMinMaxGenericHalfSuffix1[T gotype.HalfPrecision[T]](
	backend *gobackend.Backend, operandFlat []T, copyIntsFn func(flatIdx int, values []int32),
	prefixSize, reduceSize int, isMin bool) {
	if reduceSize == 0 {
		return
	}
	canParallelReduce := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		!backend.Scheduler.IsBusy() && backend.Scheduler.EffectiveParallelism() > 1 &&
		reduceSize >= argMinMaxMinParallelElems

	var single [1]int32
	for p := range prefixSize {
		row := operandFlat[p*reduceSize : (p+1)*reduceSize]
		if canParallelReduce {
			numChunks := (reduceSize + argMinMaxParallelChunkSize - 1) / argMinMaxParallelChunkSize
			chunkVals := make([]float32, numChunks)
			chunkIdxs := make([]int32, numChunks)
			backend.Scheduler.ParallelizeChunks(numChunks, 1, func(cStart, cEnd int) {
				for c := cStart; c < cEnd; c++ {
					lo := c * argMinMaxParallelChunkSize
					hi := min(lo+argMinMaxParallelChunkSize, reduceSize)
					bestVal := row[lo].Float32()
					bestIdx := int32(lo)
					if isMin {
						for idx := lo + 1; idx < hi; idx++ {
							v := row[idx].Float32()
							if v < bestVal {
								bestVal = v
								bestIdx = int32(idx)
							}
						}
					} else {
						for idx := lo + 1; idx < hi; idx++ {
							v := row[idx].Float32()
							if v > bestVal {
								bestVal = v
								bestIdx = int32(idx)
							}
						}
					}
					chunkVals[c] = bestVal
					chunkIdxs[c] = bestIdx
				}
			})
			bestVal := chunkVals[0]
			bestIdx := chunkIdxs[0]
			if isMin {
				for c := 1; c < numChunks; c++ {
					if chunkVals[c] < bestVal {
						bestVal = chunkVals[c]
						bestIdx = chunkIdxs[c]
					}
				}
			} else {
				for c := 1; c < numChunks; c++ {
					if chunkVals[c] > bestVal {
						bestVal = chunkVals[c]
						bestIdx = chunkIdxs[c]
					}
				}
			}
			single[0] = bestIdx
			copyIntsFn(p, single[:])
			continue
		}

		bestVal := row[0].Float32()
		bestIdx := int32(0)
		if isMin {
			for idx := 1; idx < reduceSize; idx++ {
				v := row[idx].Float32()
				if v < bestVal {
					bestVal = v
					bestIdx = int32(idx)
				}
			}
		} else {
			for idx := 1; idx < reduceSize; idx++ {
				v := row[idx].Float32()
				if v > bestVal {
					bestVal = v
					bestIdx = int32(idx)
				}
			}
		}
		single[0] = bestIdx
		copyIntsFn(p, single[:])
	}
}

// TODO: handle the error condition
func execArgMinMaxGenericHalf[T gotype.HalfPrecision[T], P gotype.HalfPrecisionPtr[T]](
	backend *gobackend.Backend, operand *gobackend.Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize,
	suffixSize int, isMin bool) {
	operandFlat := operand.Flat.([]T)
	if suffixSize == 1 {
		if bf16Flat, ok := operand.Flat.([]bfloat16.BFloat16); ok {
			execArgMinMaxBFloat16Suffix1(backend, bf16Flat, copyIntsFn, prefixSize, reduceSize, isMin)
			return
		}
		execArgMinMaxGenericHalfSuffix1(backend, operandFlat, copyIntsFn, prefixSize, reduceSize, isMin)
		return
	}

	// Temporary data to store argMax results, so we can traverse the operand sequentially.
	currentBestBuffer, _ := backend.GetBuffer(shapes.Make(operand.RawShape.DType, suffixSize))
	currentBest := currentBestBuffer.Flat.([]T)
	currentArgBestBuffer, _ := backend.GetBuffer(shapes.Make(dtypes.Int32, suffixSize))
	currentArgBest := currentArgBestBuffer.Flat.([]int32)

	outputFlatIdx := 0
	operandFlatIdx := 0
	for range prefixSize {
		// Initialize the current best with the first element of reduced axis:
		for suffixIdx := range suffixSize {
			currentBest[suffixIdx] = operandFlat[operandFlatIdx]
			currentArgBest[suffixIdx] = 0
			operandFlatIdx++
		}

		// Iterate over the rest of the elements of reduce axis:
		if isMin {
			// ArgMin
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx].Float32()
					if operandValue < currentBest[suffixIdx].Float32() {
						currentBest[suffixIdx] = operandFlat[operandFlatIdx]
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
					operandFlatIdx++
				}
			}
		} else {
			// ArgMax
			for reduceIdx := 1; reduceIdx < reduceSize; reduceIdx++ {
				for suffixIdx := range suffixSize {
					operandValue := operandFlat[operandFlatIdx].Float32()
					if operandValue > currentBest[suffixIdx].Float32() {
						currentBest[suffixIdx] = operandFlat[operandFlatIdx]
						currentArgBest[suffixIdx] = int32(reduceIdx)
					}
					operandFlatIdx++
				}
			}
		}

		// Copy over the result of the whole suffix.
		copyIntsFn(outputFlatIdx, currentArgBest)
		outputFlatIdx += suffixSize
	}
	backend.PutBuffer(currentBestBuffer)
	backend.PutBuffer(currentArgBestBuffer)
}

package ops

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/pkg/errors"
)

const broadcastMinParallelElems = 16384

type dynamicBroadcastInDimData struct {
	broadcastAxes []int
	specs         []compute.DynamicDimensionSpec
}

type broadcastRun struct {
	size        int
	isBroadcast bool
}

type broadcastPlan struct {
	runs []broadcastRun
	iter *gobackend.BroadcastIterator
}

func init() {
	gobackend.RegisterBroadcastInDim.Register(BroadcastInDim, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeBroadcastInDim, gobackend.PriorityGeneric, execBroadcastInDim)

	gobackend.RegisterDynamicBroadcastInDim.Register(DynamicBroadcastInDim, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeDynamicBroadcastInDim, gobackend.PriorityGeneric, execDynamicBroadcastInDim)
}

// BroadcastInDim broadcasts x to an output with the given shape.
//
//   - outputShape will be the new shape after x is broadcast.
//   - broadcastAxes maps x-axes to the corresponding outputShape axes (len(broadcastAxes) == x.Shape.Rank()),
//     the i-th axis of x is mapped to the broadcastAxes[i]-th dimension of the output.
//     broadcastAxes must be strictly increasing: this operation cannot be used to transpose axes,
//     it will only broadcast and introduce new axes in-between.
func BroadcastInDim(
	f *gobackend.Function,
	operandValue compute.Value,
	outputShape shapes.Shape,
	broadcastAxes []int,
) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("BroadcastInDim", operandValue)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	opType := compute.OpTypeBroadcastInDim
	err = shapeinference.BroadcastInDim(operand.Shape, outputShape, broadcastAxes, f.KnownDynamicAxisNames())
	if err != nil {
		return nil, err
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, inputs, broadcastAxes)
	return node, nil
}

// DynamicBroadcastInDim broadcasts operand to target dimensions specified by specs.
func DynamicBroadcastInDim(
	f *gobackend.Function,
	operandValue compute.Value,
	broadcastAxes []int,
	specs ...compute.DynamicDimensionSpec,
) (compute.Value, error) {
	allValues := []compute.Value{operandValue}
	for _, spec := range specs {
		if spec.Value != nil {
			allValues = append(allValues, spec.Value)
		}
	}
	inputs, err := f.VerifyAndCastValues("DynamicBroadcastInDim", allValues...)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	opType := compute.OpTypeDynamicBroadcastInDim
	outputShape, err := shapeinference.DynamicBroadcastInDim(operand.Shape, broadcastAxes, specs, f.KnownDynamicAxisNames())
	if err != nil {
		return nil, err
	}
	data := dynamicBroadcastInDimData{
		broadcastAxes: slices.Clone(broadcastAxes),
		specs:         slices.Clone(specs),
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, inputs, data)
	return node, nil
}

func buildBroadcastPlan(operandShape, outputShape shapes.Shape, broadcastAxes []int) broadcastPlan {
	var runsBuf [8]broadcastRun
	runs := runsBuf[:0]

	var reshapedBuf [8]int
	var reshapedDims []int
	outRank := outputShape.Rank()
	if outRank <= len(reshapedBuf) {
		reshapedDims = reshapedBuf[:outRank]
		for i := range reshapedDims {
			reshapedDims[i] = 1
		}
	} else {
		reshapedDims = xslices.SliceWithValue(outRank, 1)
	}
	for operandAxis, outputAxis := range broadcastAxes {
		reshapedDims[outputAxis] = operandShape.Dimensions[operandAxis]
	}

	for a, outDim := range outputShape.Dimensions {
		if outDim <= 1 {
			continue
		}
		isBcast := reshapedDims[a] == 1
		if n := len(runs); n > 0 && runs[n-1].isBroadcast == isBcast {
			runs[n-1].size *= outDim
		} else {
			runs = append(runs, broadcastRun{size: outDim, isBroadcast: isBcast})
		}
	}

	if len(runs) >= 4 {
		reshapedOperand := shapes.Make(operandShape.DType, reshapedDims...)
		return broadcastPlan{
			runs: slices.Clone(runs),
			iter: gobackend.NewBroadcastIterator(reshapedOperand, outputShape),
		}
	}
	return broadcastPlan{runs: slices.Clone(runs)}
}

func broadcastToShape(backend *gobackend.Backend, targetShape shapes.Shape, broadcastAxes []int, operand *gobackend.Buffer) (*gobackend.Buffer, error) {
	output, err := backend.GetBuffer(targetShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || targetShape.Size() == 0 {
		return output, nil
	}

	var plan broadcastPlan
	if operand.RawShape.Size() > 1 {
		plan = buildBroadcastPlan(operand.RawShape, output.RawShape, broadcastAxes)
	}

	fnAny, err := broadcastInDimDTypeMap.Get(targetShape.DType)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	fnAny.(func(*gobackend.Backend, *gobackend.Buffer, *gobackend.Buffer, broadcastPlan))(backend, operand, output, plan)
	return output, nil
}

// execBroadcastInDim executes the BroadcastInDim operation.
func execBroadcastInDim(
	backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	_ = inputsOwned
	operand := inputs[0]
	broadcastAxes := node.Data.([]int)
	return broadcastToShape(backend, node.Shape, broadcastAxes, operand)
}

// execDynamicBroadcastInDim executes the DynamicBroadcastInDim operation.
func execDynamicBroadcastInDim(
	backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	_ = inputsOwned
	operand := inputs[0]
	data := node.Data.(dynamicBroadcastInDimData)
	specs := data.specs
	broadcastAxes := data.broadcastAxes

	concreteDims := make([]int, len(specs))
	valIdx := 1

	for i, spec := range specs {
		if spec.Name == "" {
			concreteDims[i] = spec.Static
		} else {
			if spec.Value != nil {
				if valIdx >= len(inputs) {
					return nil, errors.Errorf("execDynamicBroadcastInDim: missing input buffer for dynamic dimension value at spec index %d", i)
				}
				valSize, err := readScalarInt(inputs[valIdx])
				if err != nil {
					return nil, errors.Wrapf(err, "execDynamicBroadcastInDim reading dynamic dimension spec %d (%q)", i, spec.Name)
				}
				valIdx++
				if valSize < 0 {
					return nil, errors.Errorf("execDynamicBroadcastInDim: dynamic dimension size for axis %q must be nonnegative, got %d", spec.Name, valSize)
				}
				concreteDims[i] = valSize
			} else {
				resolved := false
				for operandAxis, outputAxis := range broadcastAxes {
					if outputAxis == i {
						concreteDims[i] = operand.RawShape.Dimensions[operandAxis]
						resolved = true
						break
					}
				}
				if !resolved {
					return nil, errors.Errorf("execDynamicBroadcastInDim: dynamic dimension size for axis %d (%q) could not be resolved from input or dynamic value", i, spec.Name)
				}
			}
		}
	}

	targetShape := shapes.Make(operand.RawShape.DType, concreteDims...)
	return broadcastToShape(backend, targetShape, broadcastAxes, operand)
}

// repeatSliceDoubling tiles src across dst (whose length is a multiple of len(src))
// using exponential doubling so large contiguous memmove calls saturate cache/DRAM bandwidth.
func repeatSliceDoubling[T any](dst, src []T) {
	k := len(src)
	total := len(dst)
	if k == 0 || total == 0 {
		return
	}
	copy(dst[:k], src)
	for copied := k; copied < total; {
		n := min(copied, total-copied)
		copy(dst[copied:copied+n], dst[:n])
		copied += n
	}
}

//gobackend:dtypemap execBroadcastInDimGeneric ints,uints,floats,half,bool
var broadcastInDimDTypeMap = gobackend.NewDTypeMap("BroadcastInDim")

func execBroadcastInDimGeneric[T gobackend.SupportedTypesConstraints](
	backend *gobackend.Backend, operand, output *gobackend.Buffer, plan broadcastPlan,
) {
	operandFlat, outputFlat := operand.Flat.([]T), output.Flat.([]T)
	totalOut := len(outputFlat)
	if totalOut == 0 {
		return
	}
	canParallel := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && totalOut >= broadcastMinParallelElems

	// Case 0: Scalar (or size-1) operand broadcast everywhere.
	if len(operandFlat) == 1 {
		val := operandFlat[0]
		if canParallel {
			backend.Scheduler.ParallelizeChunks(totalOut, 4096, func(start, end int) {
				xslices.FillSlice(outputFlat[start:end], val)
			})
			return
		}
		xslices.FillSlice(outputFlat, val)
		return
	}

	switch len(plan.runs) {
	case 0, 1:
		// Pure identity copy [N] -> [N] (since size-1 broadcast was handled above).
		if canParallel {
			backend.Scheduler.ParallelizeChunks(totalOut, 4096, func(start, end int) {
				copy(outputFlat[start:end], operandFlat[start:end])
			})
			return
		}
		copy(outputFlat, operandFlat)
		return

	case 2:
		if plan.runs[0].isBroadcast {
			// Prefix broadcast: [1, K] -> [Outer, K].
			outer, k := plan.runs[0].size, plan.runs[1].size
			src := operandFlat[:k]
			if canParallel && outer > 1 {
				minOuter := max(1, 4096/k)
				backend.Scheduler.ParallelizeChunks(outer, minOuter, func(oStart, oEnd int) {
					repeatSliceDoubling(outputFlat[oStart*k:oEnd*k], src)
				})
				return
			}
			repeatSliceDoubling(outputFlat, src)
			return
		}
		// Suffix broadcast: [Outer, 1] -> [Outer, Inner].
		outer, inner := plan.runs[0].size, plan.runs[1].size
		if canParallel && outer > 1 {
			minOuter := max(1, 4096/inner)
			backend.Scheduler.ParallelizeChunks(outer, minOuter, func(oStart, oEnd int) {
				for o := oStart; o < oEnd; o++ {
					xslices.FillSlice(outputFlat[o*inner:(o+1)*inner], operandFlat[o])
				}
			})
			return
		}
		for o := range outer {
			xslices.FillSlice(outputFlat[o*inner:(o+1)*inner], operandFlat[o])
		}
		return

	case 3:
		if !plan.runs[0].isBroadcast && plan.runs[1].isBroadcast && !plan.runs[2].isBroadcast {
			// Middle broadcast: [Outer, 1, Inner] -> [Outer, Mid, Inner].
			outer, mid, inner := plan.runs[0].size, plan.runs[1].size, plan.runs[2].size
			outStride := mid * inner
			if canParallel && outer*mid > 1 {
				if outer >= backend.Scheduler.EffectiveParallelism() {
					minOuter := max(1, 4096/outStride)
					backend.Scheduler.ParallelizeChunks(outer, minOuter, func(oStart, oEnd int) {
						for o := oStart; o < oEnd; o++ {
							repeatSliceDoubling(outputFlat[o*outStride:(o+1)*outStride], operandFlat[o*inner:(o+1)*inner])
						}
					})
				} else {
					totalPairs := outer * mid
					minPairs := max(1, 4096/inner)
					backend.Scheduler.ParallelizeChunks(totalPairs, minPairs, func(pStart, pEnd int) {
						for p := pStart; p < pEnd; {
							o := p / mid
							mLo := p % mid
							pGroupEnd := min(pEnd, (o+1)*mid)
							mHi := mLo + (pGroupEnd - p)
							repeatSliceDoubling(outputFlat[o*outStride+mLo*inner:o*outStride+mHi*inner], operandFlat[o*inner:(o+1)*inner])
							p = pGroupEnd
						}
					})
				}
				return
			}
			for o := range outer {
				repeatSliceDoubling(outputFlat[o*outStride:(o+1)*outStride], operandFlat[o*inner:(o+1)*inner])
			}
			return
		}
		if plan.runs[0].isBroadcast && !plan.runs[1].isBroadcast && plan.runs[2].isBroadcast {
			// Outer + Inner broadcast: [1, Mid, 1] -> [Outer, Mid, Inner].
			outer, mid, inner := plan.runs[0].size, plan.runs[1].size, plan.runs[2].size
			sliceLen := mid * inner
			firstSlice := outputFlat[:sliceLen]
			for m := range mid {
				xslices.FillSlice(firstSlice[m*inner:(m+1)*inner], operandFlat[m])
			}
			if outer > 1 {
				rest := outputFlat[sliceLen:]
				if canParallel && outer-1 > 1 {
					minOuter := max(1, 4096/sliceLen)
					backend.Scheduler.ParallelizeChunks(outer-1, minOuter, func(oStart, oEnd int) {
						repeatSliceDoubling(rest[oStart*sliceLen:oEnd*sliceLen], firstSlice)
					})
				} else {
					repeatSliceDoubling(rest, firstSlice)
				}
			}
			return
		}
	}

	// Fallback for 4+ alternating broadcast runs:
	for operandFlatIdx, outputFlatIdx := range plan.iter.IterFlatIndices() {
		outputFlat[outputFlatIdx] = operandFlat[operandFlatIdx]
	}
}

package ops

type reduceWindowNode struct {
	reductionType                                              compute.ReduceOpType
	windowDimensions, strides, inputDilations, windowDilations []int
	paddings                                                   [][2]int
}

// EqualNodeData implements nodeDataComparable for reduceWindowNode.
func (r *reduceWindowNode) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*reduceWindowNode)
	if r.reductionType != o.reductionType {
		return false
	}
	return slices.Equal(r.windowDimensions, o.windowDimensions) &&
		slices.Equal(r.strides, o.strides) &&
		slices.Equal(r.inputDilations, o.inputDilations) &&
		slices.Equal(r.windowDilations, o.windowDilations) &&
		slices.Equal(r.paddings, o.paddings)
}

// ReduceWindow runs a reduction function of the type given by reductionType,
// it can be either ReduceMaxNode, ReduceSumNode, or ReduceMultiplyNode.
//
//   - reductionType: the type of reduction to perform. E.g.: [ReduceOpMax], [ReduceOpSum],...
//   - windowDimensions: the dimensions of the window, must be defined for each axis.
//   - strides: stride over elements in each axis for each window reduction. If nil, it's assume to be the
//     same as the windowDimensions -- that is, the strides jump a window at a time.
//   - inputDilations: "virtually" expand the input by introducing "holes" between elements. I.e. if
//     inputDilations are 2, then the input is expanded by inserting `2-1` copies of `0` (or whatever
//     is the reduciton "zero" value) between the elements in each dimension.
//     If nil, it's assumed to be 1 (no dilation) for each axis. Values must be >= 1.
//   - windowDilations: "virtually" expand the window by inserting `2-1` copies of `0` between the
//     elements in each dimension.
//     If nil, it's assumed to be 1 (no dilation) for each axis. Values must be >= 1.
//   - paddings: virtual padding to be added to the input at the edges (start and end) of each axis.
//     If nil, it's assumed to be 0 for each axis.

func init() {
	gobackend.RegisterReduceWindow.Register(ReduceWindow, gobackend.PriorityGeneric)
}

func ReduceWindow(f *gobackend.Function, 
	operandOp compute.Value,
	reductionType compute.ReduceOpType,
	windowDimensions, strides, inputDilations, windowDilations []int,
	paddings [][2]int,
) (compute.Value, error) {
	opType := compute.OpTypeReduceWindow
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	outputShape, err := shapeinference.ReduceWindowOp(
		operand.Shape,
		windowDimensions,
		strides,
		inputDilations,
		windowDilations,
		paddings,
	)
	if err != nil {
		return nil, err
	}
	data := &reduceWindowNode{
		reductionType:    reductionType,
		windowDimensions: windowDimensions,
		strides:          strides,
		inputDilations:   inputDilations,
		windowDilations:  windowDilations,
		paddings:         paddings,
	}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*gobackend.Node{operand}, data)
	return node, nil
}

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeReduceWindow, gobackend.PriorityGeneric, execReduceWindow)
}

func execReduceWindow(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	operandShape := operand.RawShape
	rank := operandShape.Rank()
	dtype := operandShape.DType
	outputShape := node.Shape
	output, err := backend.GetBufferForShape(outputShape)
	if err != nil {
		return nil, err
	}
	opData := node.Data.(*reduceWindowNode)

	// resolve the effective parameters, assuming shapeinference.ReduceWindowOp handled nils by defaulting them:
	// - windowDimensions is guaranteed non-nil by the builder.
	// - strides, paddings, inputDilations, windowDilations default if their opData fields are nil.
	effWindowDimensions := opData.windowDimensions
	if effWindowDimensions == nil {
		effWindowDimensions = xslices.SliceWithValue(rank, 1)
	}
	windowShape := shapes.Make(dtype, effWindowDimensions...) // the dtype here is not used.
	effStrides := opData.strides
	if effStrides == nil {
		effStrides = effWindowDimensions
	}
	effPaddings := opData.paddings
	if effPaddings == nil {
		effPaddings = xslices.SliceWithValue(rank, [2]int{0, 0})
	}
	effBaseDilations := opData.inputDilations
	if opData.inputDilations == nil {
		effBaseDilations = xslices.SliceWithValue(rank, 1)
	}
	effWindowDilations := opData.windowDilations
	if effWindowDilations == nil {
		effWindowDilations = xslices.SliceWithValue(rank, 1)
	}

	// Initialize output and updateFn according to the reduction type
	var buildUpdateFnMap *DTypeMap
	switch opData.reductionType { //nolint:exhaustive
	case compute.ReduceOpMax:
		err := output.Fill(dtype.LowestValue())
		if err != nil {
			return nil, err
		}
		buildUpdateFnMap = reduceWindowMaxDTypeMap
	case compute.ReduceOpMin:
		err := output.Fill(dtype.HighestValue())
		if err != nil {
			return nil, err
		}
		buildUpdateFnMap = reduceWindowMinDTypeMap
	case compute.ReduceOpProduct:
		output.Ones()
		buildUpdateFnMap = reduceWindowProductDTypeMap
	case compute.ReduceOpSum:
		output.Zeros()
		buildUpdateFnMap = reduceWindowSumDTypeMap
	default:
		return nil, errors.Errorf("ReduceWindow: invalid reduction type: %s", opData.reductionType)
	}
	// updateFn will aggregate the operand value into the corresponding output value.
	updateFnAny, tmpErr := buildUpdateFnMap.Get(dtype)
	if tmpErr != nil {
		return nil, tmpErr
	}
	updateFn := updateFnAny.(func(operand, output *gobackend.Buffer) reduceWindowUpdateFn)(operand, output)

	// Find the window effective sizes, accounting for the diffusion.
	windowSizes := make([]int, rank)
	for axis := range rank {
		windowSizes[axis] = (effWindowDimensions[axis]-1)*effWindowDilations[axis] + 1
	}
	// fmt.Printf("windowSizes=%v\n", windowSizes)

	// Find the shift from an output position to the corresponding window start in the operand.
	operandShifts := make([]int, rank)
	for axis := range rank {
		operandShifts[axis] = -effPaddings[axis][0]
	}
	// fmt.Printf("operandShifts=%v\n", operandShifts)

	// Find operand strides to convert operand indices to a flat index.
	operandStrides := make([]int, rank)
	stride := 1
	for axis := rank - 1; axis >= 0; axis-- {
		operandStrides[axis] = stride
		stride *= operandShape.Dimensions[axis]
	}

	// Main loop: loop over outputs, then over window, then calculate the corresponding operand position
	// that needs to be aggregated, and update the output correspondingly.
	//
	// TODO(optimizations):
	// - If the window will break the cache (outer dimensions of the window), probably that part of the window
	//   can be moved to the outer loop, so instead of having O(N*W) cache misses (random accesses),
	//   we will have O(W) cache misses and the O(N) part will be sequential or in local cache.
	//   More specifically we would split windowShape into "nonCachedWindowShape" and "cachedWindowShape", and
	//   iterate over the nonCachedWindowShape first.
	// - Can we refactor the check of baseDilation to outside of the loop ?
	windowIndices := make([]int, rank)
	operandIndices := make([]int, rank)
	for outputFlatIdx, outputIndices := range outputShape.Iter() {
		// fmt.Printf("Output %v:\n", outputIndices)
	iterWindowIndices:
		for _, windowIndices = range windowShape.IterOn(windowIndices) {
			// fmt.Printf("\t- window %v\n", windowIndices)
			for axis := range rank {
				operandIdx := outputIndices[axis]*effStrides[axis] + operandShifts[axis]
				operandIdx += windowIndices[axis] * effWindowDilations[axis]
				if operandIdx < 0 {
					// This index is out of the operand values (padding), nothing to update.
					continue iterWindowIndices
				}
				if effBaseDilations[axis] > 1 {
					if operandIdx%effBaseDilations[axis] != 0 {
						// This index is not aligned with the baseDilation, nothing to update.
						continue iterWindowIndices
					}
					operandIdx /= effBaseDilations[axis]
				}
				if operandIdx >= operandShape.Dimensions[axis] {
					// This index is out of the operand values (padding), nothing to update.
					continue iterWindowIndices
				}
				operandIndices[axis] = operandIdx
			}
			operandFlatIdx := 0
			for axis, operandIdx := range operandIndices {
				operandFlatIdx += operandIdx * operandStrides[axis]
			}
			updateFn(operandFlatIdx, outputFlatIdx)
		}
	}
	return output, nil
}

type reduceWindowUpdateFn func(operandFlatIdx, outputFlatIdx int)

var (
	//gobackend:dtypemap reduceWindowMaxBuildUpdateFn ints,uints,floats
	reduceWindowMaxDTypeMap = gobackend.NewDTypeMap("reduceWindowMaxDTypeMap")
	//gobackend:dtypemap reduceWindowMinBuildUpdateFn ints,uints,floats
	reduceWindowMinDTypeMap = gobackend.NewDTypeMap("reduceWindowMinDTypeMap")
	//gobackend:dtypemap reduceWindowSumBuildUpdateFn ints,uints,floats
	reduceWindowSumDTypeMap = gobackend.NewDTypeMap("reduceWindowSumDTypeMap")
	//gobackend:dtypemap reduceWindowProductBuildUpdateFn ints,uints,floats
	reduceWindowProductDTypeMap = gobackend.NewDTypeMap("reduceWindowProductDTypeMap")
)

func init() {
	reduceWindowMaxDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, reduceWindowMaxBuildUpdateFnBFloat16)
	reduceWindowMinDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, reduceWindowMinBuildUpdateFnBFloat16)
	reduceWindowSumDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, reduceWindowSumBuildUpdateFnBFloat16)
	reduceWindowProductDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, reduceWindowProductBuildUpdateFnBFloat16)
}

// Generic functions that build a function that will update the output at outputFlatIdx from the operand at operandFlatIdx.

func reduceWindowMaxBuildUpdateFn[T PODNumericConstraints](operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = max(outputFlat[outputFlatIdx], operandFlat[operandFlatIdx])
	}
}

func reduceWindowMaxBuildUpdateFnBFloat16(operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			max(outputFlat[outputFlatIdx].Float32(), operandFlat[operandFlatIdx].Float32()))
	}
}

func reduceWindowMinBuildUpdateFn[T PODNumericConstraints](operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = min(outputFlat[outputFlatIdx], operandFlat[operandFlatIdx])
	}
}

func reduceWindowMinBuildUpdateFnBFloat16(operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			min(outputFlat[outputFlatIdx].Float32(), operandFlat[operandFlatIdx].Float32()))
	}
}

func reduceWindowSumBuildUpdateFn[T PODNumericConstraints](operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] += operandFlat[operandFlatIdx]
	}
}

func reduceWindowSumBuildUpdateFnBFloat16(operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			outputFlat[outputFlatIdx].Float32() + operandFlat[operandFlatIdx].Float32())
	}
}

func reduceWindowProductBuildUpdateFn[T PODNumericConstraints](operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] *= operandFlat[operandFlatIdx]
	}
}

func reduceWindowProductBuildUpdateFnBFloat16(operand, output *gobackend.Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			outputFlat[outputFlatIdx].Float32() * operandFlat[operandFlatIdx].Float32())
	}
}

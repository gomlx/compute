package ops

// Pad implements the compute.Builder interface.

func init() {
	gobackend.RegisterPad.Register(Pad, gobackend.PriorityGeneric)
}

func Pad(f *gobackend.Function, operandOp, fillValueOp compute.Value, axesConfig ...compute.PadAxis) (compute.Value, error) {
	opType := compute.OpTypePad
	inputs, err := f.VerifyAndCastValues(opType.String(), operandOp, fillValueOp)
	if err != nil {
		return nil, err
	}
	operand, fillValue := inputs[0], inputs[1]

	outputShape, err := shapeinference.PadOp(operand.Shape, axesConfig...)
	if err != nil {
		return nil, err
	}

	data := &padNode{axesConfig: slices.Clone(axesConfig)}
	node, _ := f.GetOrCreateNode(opType, outputShape, []*gobackend.Node{operand, fillValue}, data)
	return node, nil
}

type padNode struct {
	axesConfig []compute.PadAxis
}

// EqualNodeData implements nodeDataComparable for padNode.
func (p *padNode) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*padNode)
	return slices.Equal(p.axesConfig, o.axesConfig)
}

func init() {
	gobackend.SetNodeExecutor(compute.OpTypePad, gobackend.PriorityGeneric, execPad)
}

func execPad(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	operand := inputs[0]
	fillValue := inputs[1]

	if node.Shape.DType.Size() < 1 {
		return nil, errors.Errorf("Pad operation does not support sub-byte types like %s", node.Shape.DType)
	}
	elementSize := node.Shape.DType.Size()

	output, err := backend.GetBufferForShape(node.Shape)
	if err != nil {
		return nil, err
	}

	params := node.Data.(*padNode)
	axesConfig := params.axesConfig

	operandBytes, err := operand.MutableBytes()
	if err != nil {
		return nil, err
	}
	outputBytes, err := output.MutableBytes()
	if err != nil {
		return nil, err
	}
	fillValueBytes, err := fillValue.MutableBytes()
	if err != nil {
		return nil, err
	}

	// Fill output buffer
	// Check if fillValue is all zeroes
	isZero := true
	for _, b := range fillValueBytes {
		if b != 0 {
			isZero = false
			break
		}
	}

	if isZero {
		// Fast path: just zero the output buffer
		output.Zeros()
	} else if len(outputBytes) > 0 {
		// Fill output buffer with the repeated fill value
		copy(outputBytes, fillValueBytes)
		for i := len(fillValueBytes); i < len(outputBytes); i *= 2 {
			copy(outputBytes[i:], outputBytes[:i])
		}
	}

	if len(operandBytes) == 0 {
		return output, nil // Nothing to copy
	}

	// Merge consecutive untouched axes
	type mergedAxis struct {
		operandDim int
		outputDim  int
		config     compute.PadAxis
	}
	var mergedAxes []mergedAxis

	isUntouched := func(config compute.PadAxis) bool {
		return config.Start == 0 && config.End == 0 && config.Interior == 0
	}

	rank := operand.RawShape.Rank()
	for i := 0; i < rank; {
		if i >= len(axesConfig) || isUntouched(axesConfig[i]) {
			// Find how many consecutive untouched axes there are
			operandDim := operand.RawShape.Dimensions[i]
			j := i + 1
			for j < rank && (j >= len(axesConfig) || isUntouched(axesConfig[j])) {
				operandDim *= operand.RawShape.Dimensions[j]
				j++
			}
			mergedAxes = append(mergedAxes, mergedAxis{
				operandDim: operandDim,
				outputDim:  operandDim,
				config:     compute.PadAxis{Start: 0, End: 0, Interior: 0},
			})
			i = j
		} else {
			outDim := operand.RawShape.Dimensions[i] + axesConfig[i].Start + axesConfig[i].End
			if operand.RawShape.Dimensions[i] > 0 {
				outDim += (operand.RawShape.Dimensions[i] - 1) * axesConfig[i].Interior
			}
			mergedAxes = append(mergedAxes, mergedAxis{
				operandDim: operand.RawShape.Dimensions[i],
				outputDim:  outDim,
				config:     axesConfig[i],
			})
			i++
		}
	}

	// Calculate element stride in bytes: if the last merged axis is untouched, we can copy it altogether.
	virtualElementSize := elementSize
	numMerged := len(mergedAxes)
	if numMerged > 0 && isUntouched(mergedAxes[numMerged-1].config) {
		virtualElementSize *= mergedAxes[numMerged-1].operandDim
		mergedAxes = mergedAxes[:numMerged-1]
		numMerged--
	}

	// Compute strides for operand and output
	operandStrides := make([]int, numMerged)
	outputStrides := make([]int, numMerged)
	opStride := virtualElementSize
	outStride := virtualElementSize
	for i := numMerged - 1; i >= 0; i-- {
		operandStrides[i] = opStride
		outputStrides[i] = outStride
		opStride *= mergedAxes[i].operandDim
		outStride *= mergedAxes[i].outputDim
	}

	// Recursive copy
	var copyND func(axis int, operandOffset, outputOffset int)
	copyND = func(axis int, operandOffset, outputOffset int) {
		if axis == numMerged {
			// Copy virtual element
			copy(outputBytes[outputOffset:outputOffset+virtualElementSize], operandBytes[operandOffset:operandOffset+virtualElementSize])
			return
		}

		mAxis := mergedAxes[axis]
		outStride := outputStrides[axis]

		outOffset := outputOffset + mAxis.config.Start*outStride
		opOffset := operandOffset

		for i := 0; i < mAxis.operandDim; i++ {
			copyND(axis+1, opOffset, outOffset)
			opOffset += operandStrides[axis]
			outOffset += outStride * (1 + mAxis.config.Interior)
		}
	}

	copyND(0, 0, 0)

	return output, nil
}

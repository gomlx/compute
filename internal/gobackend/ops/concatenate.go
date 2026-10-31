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
	dtype := outputShape.DType
	elemSize := dtype.Size()
	rank := outputShape.Rank()
	_ = inputsOwned // We don't reuse the inputs.

	// Allocate output buffer.
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	outputBytes, err := output.MutableBytes()
	if err != nil {
		return nil, err
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
		infos = stackInfos[:len(inputs)]
	} else {
		infos = make([]inputInfo, len(inputs))
	}

	outputAxisOffsetBytes := 0
	for i, inputBuf := range inputs {
		inputBytes, err := inputBuf.MutableBytes()
		if err != nil {
			return nil, err
		}
		inputBlockBytes := inputBuf.RawShape.Dimensions[axis] * innerBlockBytes
		infos[i] = inputInfo{
			bytes:      inputBytes,
			blockBytes: inputBlockBytes,
			outOffset:  outputAxisOffsetBytes,
		}
		outputAxisOffsetBytes += inputBlockBytes
	}

	copyOuterRange := func(startOuter, endOuter int) {
		for outerIdx := startOuter; outerIdx < endOuter; outerIdx++ {
			outBase := outerIdx * outputConcatAxisStrideBytes
			for _, info := range infos {
				if info.blockBytes == 0 {
					continue
				}
				inStart := outerIdx * info.blockBytes
				outStart := outBase + info.outOffset
				copy(outputBytes[outStart:outStart+info.blockBytes], info.bytes[inStart:inStart+info.blockBytes])
			}
		}
	}

	totalElements := outputShape.Size()
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && totalElements >= 65536 && outerBlockSize > 1 {
		elementsPerOuter := max(1, totalElements/outerBlockSize)
		minOuterChunk := max(1, 32768/elementsPerOuter)
		backend.Scheduler.ParallelizeChunks(outerBlockSize, minOuterChunk, copyOuterRange)
	} else {
		copyOuterRange(0, outerBlockSize)
	}

	return output, nil
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"encoding/binary"
	"math/rand/v2"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/sets"
	"github.com/gomlx/compute/support/xslices"
	"github.com/pkg/errors"
)

func init() {
	SetNodeExecutor(compute.OpTypeIdentity, PriorityGeneric, execIdentity)
	SetNodeExecutor(compute.OpTypeWhere, PriorityGeneric, execWhere)
	SetNodeExecutor(compute.OpTypeConcatenate, PriorityGeneric, execConcatenate)
	SetNodeExecutor(compute.OpTypeConvertDType, PriorityGeneric, execConvertDType)
	SetNodeExecutor(compute.OpTypeScatterMax, PriorityGeneric, execScatter)
	SetNodeExecutor(compute.OpTypeScatterMin, PriorityGeneric, execScatter)
	SetNodeExecutor(compute.OpTypeScatterSum, PriorityGeneric, execScatter)
	SetNodeExecutor(compute.OpTypeSlice, PriorityGeneric, execSlice)
	SetNodeExecutor(compute.OpTypeArgMinMax, PriorityGeneric, execArgMinMax)
	SetNodeExecutor(compute.OpTypeReduceWindow, PriorityGeneric, execReduceWindow)
	SetNodeExecutor(compute.OpTypePad, PriorityGeneric, execPad)

	// For nodes with multiple outputs:
	MultiOutputsNodeExecutors[compute.OpTypeRNGBitGenerator] = execRNGBitGenerator
}

// calculateStrides of a tensor assuming row-major order of the flat data.
func calculateStrides(dims []int) []int {
	rank := len(dims)
	stride := 1
	strides := make([]int, rank)
	for axis := rank - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= dims[axis]
	}
	return strides
}

// IdentityOp ====================================================================================================

// execIdentity implements the Identity op.
func execIdentity(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	_ = node
	operand := inputs[0]
	if inputsOwned[0] {
		// Mark the input (operand) as consumed and return it as the output.
		inputs[0] = nil
		return operand, nil
	}

	// If the input is still in use, we make a copy.
	output, err := backend.GetBuffer(operand.RawShape.DType, operand.RawShape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = operand.RawShape
	CopyFlat(output.Flat, operand.Flat)
	return output, nil
}

// WhereOp ====================================================================================================

// execWhere implements the Where op.
// onTrue and onFalse must have the same dtype (validated at graph build time in shapeinference.WhereOp).
func execWhere(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]

	// Figure out what the outputBuffer is going to be.
	outputShape := node.Shape

	var output *Buffer
	var err error
	switch {
	case onTrue.RawShape.Equal(outputShape) && inputsOwned[1]:
		output = onTrue
		inputs[1] = nil
	case onFalse.RawShape.Equal(outputShape) && inputsOwned[2]:
		output = onFalse
		inputs[2] = nil
	default:
		output, err = backend.GetBuffer(outputShape.DType, outputShape.Size())
		if err != nil {
			return nil, err
		}
		output.RawShape = outputShape
	}
	tmpAny, tmpErr := whereDTypeMap.Get(outputShape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	fn := tmpAny.(func(conditionBuf, onTrueBuf, onFalseBuf, outputBuf *Buffer))
	fn(condition, onTrue, onFalse, output)
	return output, nil
}

//gobackend:dtypemap execWhereGeneric ints,uints,floats,half,bool
var whereDTypeMap = NewDTypeMap("Where")

func execWhereGeneric[T SupportedTypesConstraints](conditionBuf, onTrueBuf, onFalseBuf, outputBuf *Buffer) {
	if conditionBuf.RawShape.IsScalar() {
		// Case 1: condition is a scalar, either we take onTrue or onFalse as a whole (with potential broadcast).
		if conditionBuf.Flat.([]bool)[0] {
			execWhereSetOutputWithValue[T](outputBuf, onTrueBuf)
		} else {
			execWhereSetOutputWithValue[T](outputBuf, onFalseBuf)
		}
		return
	}

	conditionFlat := conditionBuf.Flat.([]bool)
	onTrueFlat := onTrueBuf.Flat.([]T)
	onFalseFlat := onFalseBuf.Flat.([]T)
	outputFlat := outputBuf.Flat.([]T)
	onTrueIsScalar := onTrueBuf.RawShape.IsScalar()
	onFalseIsScalar := onFalseBuf.RawShape.IsScalar()
	onTrue := onTrueFlat[0]
	onFalse := onFalseFlat[0]
	for outputIdx, condition := range conditionFlat {
		if condition {
			if !onTrueIsScalar {
				onTrue = onTrueFlat[outputIdx]
			}
			outputFlat[outputIdx] = onTrue
		} else {
			if !onFalseIsScalar {
				onFalse = onFalseFlat[outputIdx]
			}
			outputFlat[outputIdx] = onFalse
		}
	}
}

func execWhereSetOutputWithValue[T SupportedTypesConstraints](outputBuf, valueBuf *Buffer) {
	if valueBuf == outputBuf {
		// The output is reusing the value buffer, nothing to do.
		return
	}
	if valueBuf.RawShape.Equal(outputBuf.RawShape) {
		// Copy over values.
		copy(outputBuf.Flat.([]T), valueBuf.Flat.([]T))
		return
	}
	// Value must then be a scalar:
	c := valueBuf.Flat.([]T)[0]
	outputSlice := outputBuf.Flat.([]T)
	for outputIdx := range outputSlice {
		outputSlice[outputIdx] = c
	}
}

// ConcatenateOp ====================================================================================================

// execConcatenate implements the Concatenate op using direct byte copying with offsets and strides.
func execConcatenate(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	axis := node.Data.(int) // Renamed from dimension
	outputShape := node.Shape
	dtype := outputShape.DType
	elemSize := dtype.Size()
	rank := outputShape.Rank()
	_ = inputsOwned // We don't reuse the inputs.

	// Allocate output buffer.
	output, err := backend.GetBuffer(dtype, outputShape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = outputShape
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

	// Current offset in bytes along the concatenation axis *within* an outer block in the output buffer.
	// This accumulates as we process each input tensor.
	outputAxisOffsetBytes := 0

	for _, inputBuf := range inputs {
		inputShape := inputBuf.RawShape
		inputDims := inputShape.Dimensions
		inputBytes, err := inputBuf.MutableBytes() // Use mutableBytes() for input
		if err != nil {
			return nil, err
		}

		// Size of the concatenation axis for this specific input.
		inputConcatAxisSize := inputDims[axis]

		// Total size in bytes to copy from this input *per outer block*.
		inputBlockBytes := inputConcatAxisSize * innerBlockBytes

		// Iterate through all outer dimension blocks.
		for outerIdx := range outerBlockSize {
			// Calculate the starting byte position for the current outer block in the input.
			// This is simply the outer block index times the size of the block to copy for this input.
			inputStartOffset := outerIdx * inputBlockBytes

			// Calculate the starting byte position for the current outer block in the output.
			// This is the outer block index times the total output stride along the concat axis,
			// plus the accumulated offset from previous inputs along the concat axis.
			outputStartOffset := outerIdx*outputConcatAxisStrideBytes + outputAxisOffsetBytes

			// Copy the relevant block of bytes for the current outer block.
			copy(outputBytes[outputStartOffset:outputStartOffset+inputBlockBytes], inputBytes[inputStartOffset:inputStartOffset+inputBlockBytes])
		}

		// Update the offset for the next input along the concatenation axis.
		outputAxisOffsetBytes += inputBlockBytes
	}

	return output, nil
}

// Scatter{Max,Min,Sum}Op ==========================================================================================

// execScatter implements the Scatter operation (Max, Min, Sum variants).
func execScatter(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	operand, indices, updates := inputs[0], inputs[1], inputs[2]
	scatterParams, ok := node.Data.(*scatterNode)
	if !ok {
		return nil, errors.Errorf("internal error: node.data for Scatter op is not *scatterData, but %T", node.Data)
	}

	// Output starts as a copy of the operand.
	// We might be able to reuse the operand buffer if it's owned.
	var output *Buffer
	var err error
	if inputsOwned[0] {
		output = operand
		inputs[0] = nil // Mark operand as consumed.
	} else {
		output, err = backend.CloneBuffer(operand) // Creates a new buffer with copied data.
		if err != nil {
			return nil, err
		}
	}
	output.RawShape = node.Shape // Output shape is the same as operand shape.

	// Dispatch to a type-specific scatter loop based on the operation type.
	dtype := output.RawShape.DType
	type scatterFnT = func(opType compute.OpType, output, indices, updates *Buffer, scatterParams *scatterNode) error
	tmpAny, tmpErr := scatterDTypeMap.Get(dtype)
	if tmpErr != nil {
		panic(tmpErr)
	}
	scatterFn := tmpAny.(scatterFnT)
	err = scatterFn(node.OpType, output, indices, updates, scatterParams)
	if err != nil {
		return nil, err
	}
	return output, nil
}

//gobackend:dtypemap execScatterGeneric ints,uints,floats,half
var scatterDTypeMap = NewDTypeMap("ScatterMax")

// execScatterGeneric assumes the operand is already copied to the output.
func execScatterGeneric[T SupportedTypesConstraints](opType compute.OpType, output, indices, updates *Buffer,
	scatterParams *scatterNode) error {
	// Get combineFn for operand's dtype.
	dtype := output.RawShape.DType
	type combineFnT = func(a, b T) T
	var combineFn combineFnT
	switch opType { //nolint:exhaustive
	case compute.OpTypeScatterMax:
		tmpAny, tmpErr := combineMaxDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		combineFn = tmpAny.(combineFnT) //nolint:errcheck
	case compute.OpTypeScatterMin:
		tmpAny, tmpErr := combineMinDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		combineFn = tmpAny.(combineFnT) //nolint:errcheck
	case compute.OpTypeScatterSum:
		tmpAny, tmpErr := combineSumDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		combineFn = tmpAny.(combineFnT) //nolint:errcheck
	default:
		return errors.Errorf("unsupported scatter op type %q", opType)
	}
	_ = combineFn

	outputShape := output.RawShape
	outputFlat := output.Flat.([]T) //nolint:errcheck  // it will panic
	indicesFlat := indices.Flat
	updatesShape := updates.RawShape
	updatesFlat := updates.Flat.([]T) //nolint:errcheck  // it will panic

	// Initialize gather of the scatter indices.
	indicesShape := indices.RawShape
	tmpAny, tmpErr := dereferenceIntsDTypeMap.Get(indicesShape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	deferenceIndicesFn := tmpAny.(func(flat any, in, out []int))
	_, _ = indicesFlat, deferenceIndicesFn
	indicesIt := newSubIndicesIterator(indices.RawShape, scatterParams.indexVectorAxis)
	indexVectorStride := 1
	indexVectorSize := 1
	if scatterParams.indexVectorAxis != indicesShape.Rank() {
		indexVectorSize = indices.RawShape.Dimensions[scatterParams.indexVectorAxis]
		indexVectorStride = indicesIt.PerAxisStride[scatterParams.indexVectorAxis]
	}
	indirectScatterIndices := make([]int, indexVectorSize)
	elemIndices := make([]int, indexVectorSize)
	// fmt.Printf("\tindexVectorSize=%d, indexVectorStride=%d\n", numBatchAxes, indexVectorStride)

	// Initialize iterator over the updates:
	updatesIt := newSubIndicesIterator(updatesShape, scatterParams.updateWindowAxes...)
	numBatchAxes := indicesShape.Rank() - 1
	if scatterParams.indexVectorAxis == indicesShape.Rank() {
		numBatchAxes++
	}
	updatesBatchAxes := make([]int, 0, numBatchAxes)
	updatesWindowAxesSet := sets.MakeWith(scatterParams.updateWindowAxes...)
	for axis := range updatesShape.Rank() {
		if !updatesWindowAxesSet.Has(axis) {
			updatesBatchAxes = append(updatesBatchAxes, axis)
		}
	}
	innerUpdatesIt := newSubIndicesIterator(updatesShape, updatesBatchAxes...)

	// Initialize an inner iterator over the output:
	innerOutputIt := newSubIndicesIterator(outputShape, scatterParams.insertedWindowAxes...)

	// Outer-loop: range over the pointed indices
	for {
		// Find scatter indices -> where the values are going to be combined in the output:
		flatIndirectIndex := indicesIt.FlatIdx
		for ii := range indexVectorSize {
			indirectScatterIndices[ii] = flatIndirectIndex
			flatIndirectIndex += indexVectorStride
		}
		deferenceIndicesFn(indicesFlat, indirectScatterIndices, elemIndices)
		// fmt.Printf("\tindices%v = indices.flat[%d] = %v\n", indicesIt.PerAxisIdx, indicesIt.FlatIdx, elemIndices)

		// Prepare innerOutputIt to start from the position set indices.
		for axis := range innerOutputIt.PerAxisIdx {
			innerOutputIt.PerAxisIdx[axis] = 0
		}
		innerOutputIt.FlatIdx = 0
		for scatterAxis, idx := range elemIndices {
			outputAxis := scatterParams.scatterAxesToOperandAxes[scatterAxis]
			innerOutputIt.PerAxisIdx[outputAxis] = idx
			innerOutputIt.FlatIdx += idx * innerOutputIt.PerAxisStride[outputAxis]
		}

		// Prepare innerUpdatesIt to start from the indices in the updatesIt.
		innerUpdatesIt.FlatIdx = updatesIt.FlatIdx
		copy(innerUpdatesIt.PerAxisIdx, updatesIt.PerAxisIdx)

		// Inner-loop: combine slice (window) of update into output.
		for {
			outputIdx := innerOutputIt.FlatIdx
			updatesIdx := innerUpdatesIt.FlatIdx
			// fmt.Println("\t\tCombine:")
			// fmt.Printf("\t\t- updates%v (updatesFlat[%d])=%v\n", innerUpdatesIt.PerAxisIdx, updatesIdx, updatesFlat[updatesIdx])
			// fmt.Printf("\t\t-  output%v (outputFlat[%d])=%v\n", innerOutputIt.PerAxisIdx, outputIdx, outputFlat[outputIdx])
			outputFlat[outputIdx] = combineFn(outputFlat[outputIdx], updatesFlat[updatesIdx])
			// fmt.Printf("\t\t- result=%v\n", outputFlat[outputIdx])
			if !innerUpdatesIt.Increment() {
				break
			}
			innerOutputIt.Increment()
		}

		// Next in indices:
		if !indicesIt.Increment() {
			break
		}
		updatesIt.Increment()
	}
	return nil
}

type subIndicesIterator struct {
	// FlatIdx is the current flat index to the shape.
	FlatIdx int

	// PerAxisIdx is the current indices in the shape.
	PerAxisIdx []int

	PerAxisSize   []int
	PerAxisStride []int
}

func newSubIndicesIterator(shape shapes.Shape, skipAxes ...int) *subIndicesIterator {
	rank := shape.Rank()
	it := &subIndicesIterator{
		PerAxisIdx:  make([]int, rank),
		PerAxisSize: slices.Clone(shape.Dimensions),
	}
	it.PerAxisStride = calculateStrides(shape.Dimensions)
	for _, axis := range skipAxes {
		if axis < rank {
			// Set size for axis we don't want to iterate over to 1.
			it.PerAxisSize[axis] = 1
		}
	}
	return it
}

// Increment indices. It returns true if the new index is still valid, or false if it reached the end.
func (it *subIndicesIterator) Increment() bool {
	if it.FlatIdx < 0 {
		return false
	}
	rank := len(it.PerAxisSize)
	for axis := rank - 1; axis >= 0; axis-- {
		if it.PerAxisSize[axis] == 1 {
			continue
		}
		it.PerAxisIdx[axis]++
		it.FlatIdx += it.PerAxisStride[axis]
		if it.PerAxisIdx[axis] < it.PerAxisSize[axis] {
			return true
		}

		// We are going to move to the next axis.
		if axis == 0 {
			break
		}
		it.PerAxisIdx[axis] = 0
		it.FlatIdx -= it.PerAxisStride[axis-1] // Rewind FlatIdx to start of the current axis.
	}

	// Reached end.
	it.FlatIdx = -1
	return false
}

//gobackend:dtypemap dereferenceIntsGeneric ints,uints
var dereferenceIntsDTypeMap = NewDTypeMap("Scatter Indices")

func dereferenceIntsGeneric[T PODIntegerConstraints](flatAny any, indicesIn, indicesOut []int) {
	flat := flatAny.([]T)
	for ii, index := range indicesIn {
		indicesOut[ii] = int(flat[index])
	}
}

var (
	//gobackend:dtypemap combineForScatterMaxGeneric ints,uints,floats
	combineMaxDTypeMap = NewDTypeMap("Max(a, b) for ScatterMax")
	//gobackend:dtypemap combineForScatterMinGeneric ints,uints,floats
	combineMinDTypeMap = NewDTypeMap("Min(a, b) for ScatterMin")
	//gobackend:dtypemap combineForScatterSumGeneric ints,uints,floats
	combineSumDTypeMap = NewDTypeMap("Sum(a, b) for ScatterSum")
)

func init() {
	combineMaxDTypeMap.Register(dtypes.BFloat16, PriorityTyped, combineForScatterMaxBFloat16)
	combineMinDTypeMap.Register(dtypes.BFloat16, PriorityTyped, combineForScatterMinBFloat16)
	combineSumDTypeMap.Register(dtypes.BFloat16, PriorityTyped, combineForScatterSumBFloat16)
}

func combineForScatterMaxGeneric[T PODNumericConstraints](a, b T) T {
	return max(a, b)
}

func combineForScatterMaxBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(max(a.Float32(), b.Float32()))
}

func combineForScatterMinGeneric[T PODNumericConstraints](a, b T) T {
	return min(a, b)
}

func combineForScatterMinBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(min(a.Float32(), b.Float32()))
}

func combineForScatterSumGeneric[T PODNumericConstraints](a, b T) T {
	return a + b
}

func combineForScatterSumBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(a.Float32() + b.Float32())
}

// SliceOp ========================================================================================================

// execSlice is the executor function registered for compute.OpTypeSlice.
func execSlice(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	operand := inputs[0]
	sliceParams, ok := node.Data.(*sliceNode)
	if !ok {
		// Assuming node.data holds the necessary slice parameters.
		// If Builder.Slice stores data differently, this needs adjustment.
		return nil, errors.Errorf("internal error: node.data for Slice op is not *sliceNode, but %T", node.Data)
	}

	output, err := backend.GetBuffer(node.Shape.DType, node.Shape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = node.Shape

	// Dispatch to the generic implementation based on DType.
	// Note: limits are not used in the generic exec function but passed for potential future use or consistency.
	tmpAny, tmpErr := sliceDTypeMap.Get(node.Shape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	fn := tmpAny.(func(operand, output *Buffer, params *sliceNode)) //nolint:errcheck
	fn(operand, output, sliceParams)
	return output, nil
}

//gobackend:dtypemap execSliceGeneric ints,uints,floats,half,bool
var sliceDTypeMap = NewDTypeMap("Slice")

// execSliceGeneric implements the actual slice data copying. It is called via sliceDTypeMap.Dispatch.
// It iterates through the output buffer coordinates, calculates the corresponding coordinate
// in the operand buffer using starts and strides, and copies the value.
func execSliceGeneric[T SupportedTypesConstraints](operand, output *Buffer, params *sliceNode) {
	rank := operand.RawShape.Rank()
	outputFlat := output.Flat.([]T)
	operandFlat := operand.Flat.([]T)

	// Find operandFlatIdx start value.
	var operandFlatIdx int
	operandFlatStrides := calculateStrides(operand.RawShape.Dimensions)
	for axis, idx := range params.starts {
		operandFlatIdx += operandFlatStrides[axis] * idx

		// Scale the flat index strides by the requested strides for this axis.
		operandFlatStrides[axis] *= params.strides[axis]
	}

	operandPerAxisIdx := make([]int, rank)
	operandPerAxisSize := output.RawShape.Dimensions

	for outputFlatIdx := range outputFlat {
		// Copy value at current position.
		outputFlat[outputFlatIdx] = operandFlat[operandFlatIdx]

		// Iterate to the next operand position.
		for axis := rank - 1; axis >= 0; axis-- {
			if operandPerAxisSize[axis] == 1 {
				// We don't iterate on this axis.
				continue
			}

			// Increment the current axis.
			operandPerAxisIdx[axis]++
			operandFlatIdx += operandFlatStrides[axis]
			if operandPerAxisIdx[axis] < operandPerAxisSize[axis] {
				// Done for this iteration.
				break
			}

			// Rewind the current axis: we will bump the next axis for this iteration.
			operandPerAxisIdx[axis] = 0
			operandFlatIdx -= operandPerAxisSize[axis] * operandFlatStrides[axis]
		}
	}
}

// RNGBitGenerator ====================================================================================================

// execRNGBitGenerator is the executor function registered for compute.OpTypeRngBitGenerator.
func execRNGBitGenerator(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	state := inputs[0]
	stateFlat := state.Flat.([]uint64)

	// Reserved outputs:
	rngData, err := backend.GetBuffer(node.MultiOutputsShapes[1].DType, node.MultiOutputsShapes[1].Size())
	if err != nil {
		return nil, err
	}
	rngData.RawShape = node.MultiOutputsShapes[1].Clone()
	rngDataBytes, err := rngData.MutableBytes()
	if err != nil {
		return nil, err
	}

	// Generate random using rand/v2:
	rng := rand.NewPCG(stateFlat[0], stateFlat[1]) // Use state and increment as seed
	var randomBits uint64
	for idx := range rngDataBytes {
		if idx%8 == 0 {
			randomBits = rng.Uint64()
		}
		// Take one byte from the randomBits.
		rngDataBytes[idx] = byte(randomBits & 0xFF)
		randomBits >>= 8
	}

	// Update state output - PCG internal state after generating random bytes
	if inputsOwned[0] {
		// We re-use the current state.
		inputs[0] = nil
	} else {
		state.RawShape = node.MultiOutputsShapes[0]
		state, err = backend.GetBuffer(state.RawShape.DType, state.RawShape.Size())
		if err != nil {
			return nil, err
		}
	}
	stateFlat = state.Flat.([]uint64)

	// See details on Go source code src/math/rand/v2/pcg.go:
	rngState, err := rng.MarshalBinary()
	if err != nil {
		panic(errors.Wrapf(err, "cannot update RNGBitGenerator state"))
	}
	if len(rngState) != 20 && string(rngState[:4]) != "pcg:" {
		return nil, errors.Errorf("format of PCG random number generator changed (we got %d bytes starting with %q, "+
			"we wanted 20 and starting with the string 'pcg:'), pls open an issue in GoMLX",
			len(rngState), rngState[:4])
	}
	stateFlat[0] = binary.LittleEndian.Uint64(rngState[4 : 4+8])
	stateFlat[1] = binary.LittleEndian.Uint64(rngState[4+8 : 4+16])
	return []*Buffer{state, rngData}, nil
}

// execArgMinMax ====================================================================================================

const MaxArgMinMaxReductionSize = 0x8000_0000

// execArgMinMax is the executor function registered for compute.OpTypeArgMinMax.
func execArgMinMax(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	operand := inputs[0]
	reduceAxis := node.Data.(*argMinMaxNode).axis
	isMin := node.Data.(*argMinMaxNode).isMin
	output, err := backend.GetBuffer(node.Shape.DType, node.Shape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = node.Shape

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
	if reduceSize >= MaxArgMinMaxReductionSize {
		// If we need larger, change buildArgMinMax to use int64 instead of int32.
		return nil, errors.Errorf("ArgMaxMin implementation only supports reduction on dimensions < %d, got operand shaped %s and reduce axis is %d",
			MaxArgMinMaxReductionSize, operand.RawShape, reduceAxis)
	}

	// Instantiate the function to copy over results from ints:
	tmpAny, tmpErr := argMinMaxCopyIntsDTypeMap.Get(output.RawShape.DType)
	if tmpErr != nil {
		return nil, tmpErr
	}
	buildCopyIntsFn := tmpAny.(func(output *Buffer) func(flatIdx int, values []int32))
	copyIntsFn := buildCopyIntsFn(output)

	// Dispatch to the generic implementation based on DType.
	tmpAny2, tmpErr2 := argMinMaxDTypeMap.Get(operand.RawShape.DType)
	if tmpErr2 != nil {
		return nil, tmpErr2
	}
	argMinMaxFn := tmpAny2.(func(backend *Backend, operand *Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize, suffixSize int, isMin bool))
	argMinMaxFn(backend, operand, copyIntsFn, prefixSize, reduceSize, suffixSize, isMin)
	return output, nil
}

var (
	//gobackend:dtypemap execArgMinMaxGeneric ints,uints,floats
	argMinMaxDTypeMap = NewDTypeMap("ArgMinMaxRun")
	//gobackend:dtypemap buildArgMinMaxCopyIntsFn ints,uints
	argMinMaxCopyIntsDTypeMap = NewDTypeMap("ArgMinMaxCopyInts")
)

// buildArgMinMaxCopyIntsFn creates a "copyInts" function to copy the given values starting at the flatIdx to
// the output buffer.
func buildArgMinMaxCopyIntsFn[T PODIntegerConstraints](output *Buffer) func(flatIdx int, values []int32) {
	outputFlat := output.Flat.([]T)
	return func(flatIdx int, values []int32) {
		for _, value := range values {
			outputFlat[flatIdx] = T(value)
			flatIdx++
		}
	}
}

// TODO: handle the error condition.
func execArgMinMaxGeneric[T PODNumericConstraints](
	backend *Backend, operand *Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize,
	suffixSize int, isMin bool) {
	operandFlat := operand.Flat.([]T)

	// Temporary data to store argMax results, so we can traverse the operand sequentially.
	currentBestBuffer, _ := backend.GetBuffer(operand.RawShape.DType, suffixSize)

	currentBest := currentBestBuffer.Flat.([]T)
	currentArgBestBuffer, _ := backend.GetBuffer(dtypes.Int32, suffixSize)
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

func init() {
	argMinMaxDTypeMap.Register(dtypes.BFloat16, PriorityTyped, execArgMinMaxGenericBFloat16)
}

// TODO: handle the error condition
func execArgMinMaxGenericBFloat16(
	backend *Backend, operand *Buffer, copyIntsFn func(flatIdx int, values []int32), prefixSize, reduceSize,
	suffixSize int, isMin bool) {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)

	// Temporary data to store argMax results, so we can traverse the operand sequentially.
	currentBestBuffer, _ := backend.GetBuffer(operand.RawShape.DType, suffixSize)
	currentBest := currentBestBuffer.Flat.([]bfloat16.BFloat16)
	currentArgBestBuffer, _ := backend.GetBuffer(dtypes.Int32, suffixSize)
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

// =================================================================================================================
// ReduceWindow ----------------------------------------------------------------------------------------------------
// =================================================================================================================
func execReduceWindow(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
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
	updateFn := updateFnAny.(func(operand, output *Buffer) reduceWindowUpdateFn)(operand, output)

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
	reduceWindowMaxDTypeMap = NewDTypeMap("reduceWindowMaxDTypeMap")
	//gobackend:dtypemap reduceWindowMinBuildUpdateFn ints,uints,floats
	reduceWindowMinDTypeMap = NewDTypeMap("reduceWindowMinDTypeMap")
	//gobackend:dtypemap reduceWindowSumBuildUpdateFn ints,uints,floats
	reduceWindowSumDTypeMap = NewDTypeMap("reduceWindowSumDTypeMap")
	//gobackend:dtypemap reduceWindowProductBuildUpdateFn ints,uints,floats
	reduceWindowProductDTypeMap = NewDTypeMap("reduceWindowProductDTypeMap")
)

func init() {
	reduceWindowMaxDTypeMap.Register(dtypes.BFloat16, PriorityTyped, reduceWindowMaxBuildUpdateFnBFloat16)
	reduceWindowMinDTypeMap.Register(dtypes.BFloat16, PriorityTyped, reduceWindowMinBuildUpdateFnBFloat16)
	reduceWindowSumDTypeMap.Register(dtypes.BFloat16, PriorityTyped, reduceWindowSumBuildUpdateFnBFloat16)
	reduceWindowProductDTypeMap.Register(dtypes.BFloat16, PriorityTyped, reduceWindowProductBuildUpdateFnBFloat16)
}

// Generic functions that build a function that will update the output at outputFlatIdx from the operand at operandFlatIdx.

func reduceWindowMaxBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = max(outputFlat[outputFlatIdx], operandFlat[operandFlatIdx])
	}
}

func reduceWindowMaxBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			max(outputFlat[outputFlatIdx].Float32(), operandFlat[operandFlatIdx].Float32()))
	}
}

func reduceWindowMinBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = min(outputFlat[outputFlatIdx], operandFlat[operandFlatIdx])
	}
}

func reduceWindowMinBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			min(outputFlat[outputFlatIdx].Float32(), operandFlat[operandFlatIdx].Float32()))
	}
}

func reduceWindowSumBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] += operandFlat[operandFlatIdx]
	}
}

func reduceWindowSumBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			outputFlat[outputFlatIdx].Float32() + operandFlat[operandFlatIdx].Float32())
	}
}

func reduceWindowProductBuildUpdateFn[T PODNumericConstraints](operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] *= operandFlat[operandFlatIdx]
	}
}

func reduceWindowProductBuildUpdateFnBFloat16(operand, output *Buffer) reduceWindowUpdateFn {
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	return func(operandFlatIdx, outputFlatIdx int) {
		outputFlat[outputFlatIdx] = bfloat16.FromFloat32(
			outputFlat[outputFlatIdx].Float32() * operandFlat[operandFlatIdx].Float32())
	}
}

func execPad(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
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

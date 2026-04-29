// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

// Float16 implementations of special operations.
// These are separated from exec_special_ops.go to keep files organized by dtype.

import (
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
)

// Float16 reduce operations

func init() {
	reduceMaxDTypeMap.Register(dtypes.Float16, PriorityTyped, execReduceMaxFloat16)
	reduceMinDTypeMap.Register(dtypes.Float16, PriorityTyped, execReduceMinFloat16)
	reduceSumDTypeMap.Register(dtypes.Float16, PriorityTyped, execReduceSumFloat16)
	reduceProductDTypeMap.Register(dtypes.Float16, PriorityTyped, execReduceProductFloat16)
}

func execReduceMaxFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with the lowest value.
	initialValue := dtype.LowestValue().(float16.Float16)
	outputFlat := output.Flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	// Reduce from operand.
	operandFlat := operand.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.FromFloat32(max(a, b))
	}
}

func execReduceMinFloat16(operand, output *Buffer, it *reduceOutputIterator, dtype dtypes.DType) {
	// Initialize with the highest value.
	initialValue := dtype.HighestValue().(float16.Float16)
	outputFlat := output.Flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.FromFloat32(min(a, b))
	}
}

func execReduceSumFloat16(operand, output *Buffer, it *reduceOutputIterator, _ dtypes.DType) {
	// Initialize with 0.
	initialValue := float16.FromFloat32(0)
	outputFlat := output.Flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.FromFloat32(a + b)
	}
}

func execReduceProductFloat16(operand, output *Buffer, it *reduceOutputIterator, _ dtypes.DType) {
	// Initialize with 1.
	initialValue := float16.FromFloat32(1)
	outputFlat := output.Flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}

	operandFlat := operand.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.FromFloat32(a * b)
	}
}

// Float16 conversion functions

// Float16 buffer operations

func mutableBytesFloat16(b *Buffer) ([]byte, error) {
	flat := b.Flat.([]float16.Float16)
	bytePointer := (*byte)(unsafe.Pointer(&flat[0]))
	return unsafe.Slice(bytePointer, len(flat)*2), nil // Float16 is 2 bytes
}

func fillBufferFloat16(b *Buffer, valueAny any) {
	var value float16.Float16
	if valueAny != nil {
		value = valueAny.(float16.Float16)
	}
	flat := b.Flat.([]float16.Float16)
	for i := range flat {
		flat[i] = value
	}
}

func execWhereFloat16(conditionBuf, onTrueBuf, onFalseBuf, outputBuf *Buffer) {
	if conditionBuf.RawShape.IsScalar() {
		if conditionBuf.Flat.([]bool)[0] {
			execWhereSetOutputFloat16(outputBuf, onTrueBuf)
		} else {
			execWhereSetOutputFloat16(outputBuf, onFalseBuf)
		}
		return
	}
	conditionFlat := conditionBuf.Flat.([]bool)
	onTrueFlat := onTrueBuf.Flat.([]float16.Float16)
	onFalseFlat := onFalseBuf.Flat.([]float16.Float16)
	outputFlat := outputBuf.Flat.([]float16.Float16)
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

func execWhereSetOutputFloat16(outputBuf, valueBuf *Buffer) {
	if valueBuf == outputBuf {
		return
	}
	if valueBuf.RawShape.Equal(outputBuf.RawShape) {
		copy(outputBuf.Flat.([]float16.Float16), valueBuf.Flat.([]float16.Float16))
		return
	}
	// Broadcast scalar
	value := valueBuf.Flat.([]float16.Float16)[0]
	output := outputBuf.Flat.([]float16.Float16)
	for i := range output {
		output[i] = value
	}
}

func execTransposeFloat16(operand, output *Buffer, it *transposeIterator) {
	operandFlat := operand.Flat.([]float16.Float16)
	outputFlat := output.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputFlat[it.next()] = value
	}
}

func execSliceFloat16(operand, output *Buffer, params *sliceNode) {
	rank := operand.RawShape.Rank()
	outputFlat := output.Flat.([]float16.Float16)
	operandFlat := operand.Flat.([]float16.Float16)

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

func init() {
	// Register Float16 buffer and misc operations
	mutableBytesDTypeMap.Register(dtypes.Float16, PriorityTyped, mutableBytesFloat16)
	fillBufferDTypeMap.Register(dtypes.Float16, PriorityTyped, fillBufferFloat16)
	whereDTypeMap.Register(dtypes.Float16, PriorityTyped, execWhereFloat16)
	transposeDTypeMap.Register(dtypes.Float16, PriorityTyped, execTransposeFloat16)
	sliceDTypeMap.Register(dtypes.Float16, PriorityTyped, execSliceFloat16)
}

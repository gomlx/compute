// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"math"
	"math/bits"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/exceptions"
)

func init() {
	SetNodeExecutor(compute.OpTypeNeg, PriorityGeneric, execNeg)
	SetNodeExecutor(compute.OpTypeAbs, PriorityGeneric, execAbs)
	SetNodeExecutor(compute.OpTypeSign, PriorityGeneric, execSign)
	SetNodeExecutor(compute.OpTypeLogicalNot, PriorityGeneric, execLogicalNot)
	SetNodeExecutor(compute.OpTypeBitwiseNot, PriorityGeneric, execBitwiseNot)
	SetNodeExecutor(compute.OpTypeBitCount, PriorityGeneric, execBitCount)
	SetNodeExecutor(compute.OpTypeClz, PriorityGeneric, execClz)
	SetNodeExecutor(compute.OpTypeExp, PriorityGeneric, execExp)
	SetNodeExecutor(compute.OpTypeExpm1, PriorityGeneric, execExpm1)
	SetNodeExecutor(compute.OpTypeLog, PriorityGeneric, execLog)
	SetNodeExecutor(compute.OpTypeLog1p, PriorityGeneric, execLog1p)
	SetNodeExecutor(compute.OpTypeCeil, PriorityGeneric, execCeil)
	SetNodeExecutor(compute.OpTypeFloor, PriorityGeneric, execFloor)
	SetNodeExecutor(compute.OpTypeRound, PriorityGeneric, execRound)
	SetNodeExecutor(compute.OpTypeRsqrt, PriorityGeneric, execRsqrt)
	SetNodeExecutor(compute.OpTypeSqrt, PriorityGeneric, execSqrt)
	SetNodeExecutor(compute.OpTypeCos, PriorityGeneric, execCos)
	SetNodeExecutor(compute.OpTypeSin, PriorityGeneric, execSin)
	SetNodeExecutor(compute.OpTypeTanh, PriorityGeneric, execTanh)
	SetNodeExecutor(compute.OpTypeIsFinite, PriorityGeneric, execIsFinite)
	SetNodeExecutor(compute.OpTypeLogistic, PriorityGeneric, execLogistic)
	SetNodeExecutor(compute.OpTypeErf, PriorityGeneric, execErf)
}

// unaryOperandAndOutput is a convenience function to get the input and output -- which may be the reuse of the input
func unaryOperandAndOutput(backend *Backend, inputs []*Buffer, inputsOwned []bool) (input, output *Buffer, err error) {
	input = inputs[0]
	if inputsOwned[0] {
		output = input
		inputs[0] = nil // This tells the executor that we took over the buffer.
		return
	}
	output, err = backend.GetBuffer(input.RawShape.DType, input.RawShape.Size())
	if err != nil {
		return input, nil, err // as output is nil
	}
	output.RawShape = input.RawShape.Clone()
	return input, output, nil
}

// execNeg executes the unary op Neg.
func execNeg(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Int8:
		execNegGeneric[int8](input.Flat.([]int8), output.Flat.([]int8))
	case dtypes.Int16:
		execNegGeneric[int16](input.Flat.([]int16), output.Flat.([]int16))
	case dtypes.Int32:
		execNegGeneric[int32](input.Flat.([]int32), output.Flat.([]int32))
	case dtypes.Int64:
		execNegGeneric[int64](input.Flat.([]int64), output.Flat.([]int64))
	case dtypes.Float32:
		execNegGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execNegGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execNegBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execNegGeneric[T PODSignedNumericConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = -input
	}
}

func execNegBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(-input.Float32())
	}
}

// execAbs executes the unary op Abs.
func execAbs(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Int8:
		execAbsGeneric[int8](input.Flat.([]int8), output.Flat.([]int8))
	case dtypes.Int16:
		execAbsGeneric[int16](input.Flat.([]int16), output.Flat.([]int16))
	case dtypes.Int32:
		execAbsGeneric[int32](input.Flat.([]int32), output.Flat.([]int32))
	case dtypes.Int64:
		execAbsGeneric[int64](input.Flat.([]int64), output.Flat.([]int64))
	case dtypes.Uint8:
		execAbsUnsignedGeneric[uint8](input, output)
	case dtypes.Uint16:
		execAbsUnsignedGeneric[uint16](input, output)
	case dtypes.Uint32:
		execAbsUnsignedGeneric[uint32](input, output)
	case dtypes.Uint64:
		execAbsUnsignedGeneric[uint64](input, output)
	case dtypes.Float32:
		execAbsGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execAbsGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execAbsBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execAbsGeneric[T PODSignedNumericConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		if input < 0 {
			outputs[ii] = -input
		} else {
			outputs[ii] = input
		}
	}
}

func execAbsUnsignedGeneric[T PODUnsignedConstraints](input, output *Buffer) {
	if input == output {
		return
	}
	copy(output.Flat.([]T), input.Flat.([]T))
}

func execAbsBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		f := input.Float32()
		if f < 0 {
			outputs[ii] = bfloat16.FromFloat32(-f)
		} else {
			outputs[ii] = input
		}
	}
}

// execSign executes the unary op Sign.
func execSign(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Int8:
		execSignGeneric[int8](input.Flat.([]int8), output.Flat.([]int8))
	case dtypes.Int16:
		execSignGeneric[int16](input.Flat.([]int16), output.Flat.([]int16))
	case dtypes.Int32:
		execSignGeneric[int32](input.Flat.([]int32), output.Flat.([]int32))
	case dtypes.Int64:
		execSignGeneric[int64](input.Flat.([]int64), output.Flat.([]int64))
	case dtypes.Uint8:
		execSignForUnsignedGeneric[uint8](input.Flat.([]uint8), output.Flat.([]uint8))
	case dtypes.Uint16:
		execSignForUnsignedGeneric[uint16](input.Flat.([]uint16), output.Flat.([]uint16))
	case dtypes.Uint32:
		execSignForUnsignedGeneric[uint32](input.Flat.([]uint32), output.Flat.([]uint32))
	case dtypes.Uint64:
		execSignForUnsignedGeneric[uint64](input.Flat.([]uint64), output.Flat.([]uint64))
	case dtypes.Float32:
		execSignGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execSignGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execSignBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execSignGeneric[T PODSignedNumericConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		switch {
		case input < 0:
			outputs[ii] = -1
		case input > 0:
			outputs[ii] = 1
		default:
			outputs[ii] = 0
		}
	}
}

func execSignForUnsignedGeneric[T PODUnsignedConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		if input > 0 {
			outputs[ii] = 1
		} else {
			outputs[ii] = 0
		}
	}
}

func execSignBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		f := input.Float32()
		switch {
		case f < 0:
			outputs[ii] = bfloat16.FromFloat32(-1.0)
		case f > 0:
			outputs[ii] = bfloat16.FromFloat32(1.0)
		default:
			outputs[ii] = bfloat16.FromFloat32(0.0)
		}
	}
}

// execLogicalNot executes the unary op LogicalNot.
func execLogicalNot(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if input.RawShape.DType != dtypes.Bool {
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	for ii, val := range input.Flat.([]bool) {
		output.Flat.([]bool)[ii] = !val
	}
	return output, nil
}

// execBitwiseNot executes the unary op BitwiseNot.
func execBitwiseNot(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Int8:
		execBitwiseNotGeneric[int8](input.Flat.([]int8), output.Flat.([]int8))
	case dtypes.Int16:
		execBitwiseNotGeneric[int16](input.Flat.([]int16), output.Flat.([]int16))
	case dtypes.Int32:
		execBitwiseNotGeneric[int32](input.Flat.([]int32), output.Flat.([]int32))
	case dtypes.Int64:
		execBitwiseNotGeneric[int64](input.Flat.([]int64), output.Flat.([]int64))
	case dtypes.Uint8:
		execBitwiseNotGeneric[uint8](input.Flat.([]uint8), output.Flat.([]uint8))
	case dtypes.Uint16:
		execBitwiseNotGeneric[uint16](input.Flat.([]uint16), output.Flat.([]uint16))
	case dtypes.Uint32:
		execBitwiseNotGeneric[uint32](input.Flat.([]uint32), output.Flat.([]uint32))
	case dtypes.Uint64:
		execBitwiseNotGeneric[uint64](input.Flat.([]uint64), output.Flat.([]uint64))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execBitwiseNotGeneric[T PODIntegerConstraints](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = ^input
	}
}

// execBitCount executes the unary op BitCount.
func execBitCount(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Int8:
		execBitCountGeneric8[int8](input.Flat.([]int8), output.Flat.([]int8))
	case dtypes.Int16:
		execBitCountGeneric16[int16](input.Flat.([]int16), output.Flat.([]int16))
	case dtypes.Int32:
		execBitCountGeneric32[int32](input.Flat.([]int32), output.Flat.([]int32))
	case dtypes.Int64:
		execBitCountGeneric64[int64](input.Flat.([]int64), output.Flat.([]int64))
	case dtypes.Uint8:
		execBitCountGeneric8[uint8](input.Flat.([]uint8), output.Flat.([]uint8))
	case dtypes.Uint16:
		execBitCountGeneric16[uint16](input.Flat.([]uint16), output.Flat.([]uint16))
	case dtypes.Uint32:
		execBitCountGeneric32[uint32](input.Flat.([]uint32), output.Flat.([]uint32))
	case dtypes.Uint64:
		execBitCountGeneric64[uint64](input.Flat.([]uint64), output.Flat.([]uint64))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execBitCountGeneric8[T int8 | uint8](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.OnesCount8(uint8(input)))
	}
}

func execBitCountGeneric16[T int16 | uint16](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.OnesCount16(uint16(input)))
	}
}

func execBitCountGeneric32[T int32 | uint32](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.OnesCount32(uint32(input)))
	}
}

func execBitCountGeneric64[T int64 | uint64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.OnesCount64(uint64(input)))
	}
}

// execClz executes the unary op Clz.
func execClz(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Int8:
		execClzGeneric8[int8](input.Flat.([]int8), output.Flat.([]int8))
	case dtypes.Int16:
		execClzGeneric16[int16](input.Flat.([]int16), output.Flat.([]int16))
	case dtypes.Int32:
		execClzGeneric32[int32](input.Flat.([]int32), output.Flat.([]int32))
	case dtypes.Int64:
		execClzGeneric64[int64](input.Flat.([]int64), output.Flat.([]int64))
	case dtypes.Uint8:
		execClzGeneric8[uint8](input.Flat.([]uint8), output.Flat.([]uint8))
	case dtypes.Uint16:
		execClzGeneric16[uint16](input.Flat.([]uint16), output.Flat.([]uint16))
	case dtypes.Uint32:
		execClzGeneric32[uint32](input.Flat.([]uint32), output.Flat.([]uint32))
	case dtypes.Uint64:
		execClzGeneric64[uint64](input.Flat.([]uint64), output.Flat.([]uint64))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execClzGeneric8[T int8 | uint8](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.LeadingZeros8(uint8(input)))
	}
}

func execClzGeneric16[T int16 | uint16](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.LeadingZeros16(uint16(input)))
	}
}

func execClzGeneric32[T int32 | uint32](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.LeadingZeros32(uint32(input)))
	}
}

func execClzGeneric64[T int64 | uint64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(bits.LeadingZeros64(uint64(input)))
	}
}

// execExp executes the unary op Exp.
func execExp(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execExpGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execExpGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execExpBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execExpGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Exp(float64(input)))
	}
}

func execExpBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Exp(float64(input.Float32()))))
	}
}

// execExpm1 executes the unary op Expm1.
func execExpm1(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execExpm1Generic[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execExpm1Generic[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execExpm1BF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execExpm1Generic[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Expm1(float64(input)))
	}
}

func execExpm1BF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Expm1(float64(input.Float32()))))
	}
}

// execLog executes the unary op Log.
func execLog(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execLogGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execLogGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execLogBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execLogGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Log(float64(input)))
	}
}

func execLogBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Log(float64(input.Float32()))))
	}
}

// execLog1p executes the unary op Log1p.
func execLog1p(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execLog1pGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execLog1pGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execLog1pBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execLog1pGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Log1p(float64(input)))
	}
}

func execLog1pBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Log1p(float64(input.Float32()))))
	}
}

// execCeil executes the unary op Ceil.
func execCeil(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execCeilGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execCeilGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execCeilBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execCeilGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Ceil(float64(input)))
	}
}

func execCeilBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Ceil(float64(input.Float32()))))
	}
}

// execFloor executes the unary op Floor.
func execFloor(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execFloorGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execFloorGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execFloorBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execFloorGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Floor(float64(input)))
	}
}

func execFloorBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Floor(float64(input.Float32()))))
	}
}

// execRound executes the unary op Round.
func execRound(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execRoundGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execRoundGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execRoundBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execRoundGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.RoundToEven(float64(input)))
	}
}

func execRoundBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.RoundToEven(float64(input.Float32()))))
	}
}

// execRsqrt executes the unary op Rsqrt.
func execRsqrt(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execRsqrtGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execRsqrtGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execRsqrtBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execRsqrtGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(1.0 / math.Sqrt(float64(input)))
	}
}

func execRsqrtBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(1.0 / math.Sqrt(float64(input.Float32()))))
	}
}

// execSqrt executes the unary op Sqrt.
func execSqrt(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execSqrtGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execSqrtGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execSqrtBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	case dtypes.Float16:
		execSqrtF16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execSqrtGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Sqrt(float64(input)))
	}
}

func execSqrtBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Sqrt(float64(input.Float32()))))
	}
}

func execSqrtF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Sqrt(float64(input.Float32()))))
	}
}

// execCos executes the unary op Cos.
func execCos(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execCosGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execCosGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execCosBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execCosGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Cos(float64(input)))
	}
}

func execCosBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Cos(float64(input.Float32()))))
	}
}

// execSin executes the unary op Sin.
func execSin(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execSinGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execSinGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execSinBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execSinGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Sin(float64(input)))
	}
}

func execSinBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Sin(float64(input.Float32()))))
	}
}

// execLogistic executes the unary op Logistic.
func execLogistic(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execLogisticGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execLogisticGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execLogisticBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execLogisticGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		if input >= 0 {
			outputs[ii] = T(1.0 / (1.0 + math.Exp(-float64(input))))
		} else {
			e_x := math.Exp(float64(input))
			outputs[ii] = T(e_x / (1.0 + e_x))
		}
	}
}

func execLogisticBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		input64 := float64(input.Float32())
		var output64 float64
		if input64 >= 0 {
			output64 = 1.0 / (1.0 + math.Exp(-input64))
		} else {
			e_x := math.Exp(input64)
			output64 = e_x / (1.0 + e_x)
		}
		outputs[ii] = bfloat16.FromFloat32(float32(output64))
	}
}

// execTanh executes the unary op Tanh.
func execTanh(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execTanhGeneric[float32](input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execTanhGeneric[float64](input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execTanhBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execTanhGeneric[T float32 | float64](inputs, outputs []T) {
	for ii, input := range inputs {
		outputs[ii] = T(math.Tanh(float64(input)))
	}
}

func execTanhBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Tanh(float64(input.Float32()))))
	}
}

// execIsFinite executes the unary op IsFinite.
func execIsFinite(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	input := inputs[0]
	// Output has the same shape as the input, but different dtypes: it is a bool.
	output, err := backend.GetBuffer(dtypes.Bool, input.RawShape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = node.Shape
	switch input.RawShape.DType {
	case dtypes.Float32:
		execIsFiniteGeneric[float32](input.Flat.([]float32), output.Flat.([]bool))
	case dtypes.Float64:
		execIsFiniteGeneric[float64](input.Flat.([]float64), output.Flat.([]bool))
	case dtypes.BFloat16:
		execIsFiniteBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bool))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

func execIsFiniteGeneric[T float32 | float64](inputs []T, outputs []bool) {
	for ii, input := range inputs {
		outputs[ii] = !math.IsInf(float64(input), 0) && !math.IsNaN(float64(input))
	}
}

func execIsFiniteBF16(inputs []bfloat16.BFloat16, outputs []bool) {
	for ii, input := range inputs {
		f := input.Float32()
		outputs[ii] = !math.IsInf(float64(f), 0) && !math.IsNaN(float64(f))
	}
}

// execErf executes the unary op Erf.
func execErf(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		execErfGeneric[float32](backend, input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		execErfGeneric[float64](backend, input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		execErfBF16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	default:
		exceptions.Panicf("unsupported data type %s for %s", input.RawShape.DType, node.OpType)
	}
	return output, nil
}

const unaryMinParallelizeChunk = 4096

func execErfGeneric[T float32 | float64](backend *Backend, inputs, outputs []T) {
	lenInputs := len(inputs)
	if backend.Workers.IsEnabled() && lenInputs > unaryMinParallelizeChunk {
		// Parallelize operation into chunks.
		var wg sync.WaitGroup
		for ii := 0; ii < lenInputs; ii += unaryMinParallelizeChunk {
			iiEnd := min(ii+unaryMinParallelizeChunk, lenInputs)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				for jj := ii; jj < iiEnd; jj++ {
					outputs[jj] = T(math.Erf(float64(inputs[jj])))
				}
				wg.Done()
			})
		}
		wg.Wait()

	} else {
		// Sequentially processing it.
		for ii, input := range inputs {
			outputs[ii] = T(math.Erf(float64(input)))
		}
	}
}

func execErfBF16(inputs, outputs []bfloat16.BFloat16) {
	for ii, input := range inputs {
		outputs[ii] = bfloat16.FromFloat32(float32(math.Erf(float64(input.Float32()))))
	}
}

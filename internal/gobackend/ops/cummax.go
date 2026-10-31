// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
)

var (
	//gobackend:dtypemap execCumMaxGeneric ints,uints,floats
	//gobackend:dtypemap execCumMaxHalfPrecision half
	cumMaxDTypeMap = gobackend.NewDTypeMap("CumMax")
)

func init() {
	gobackend.RegisterCumMax.Register(CumMax, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeCumMax, gobackend.PriorityGeneric, execCumMax)
}

// CumMax returns the cumulative maximum of the elements along the given axis.
func CumMax(f *gobackend.Function, operandValue compute.Value, axis int, options compute.CumSumOptions) (compute.Value, error) {
	opType := compute.OpTypeCumMax
	inputs, err := f.VerifyAndCastValues(opType.String(), operandValue)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	if !operand.Shape.DType.IsFloat() && !operand.Shape.DType.IsInt() {
		return nil, errors.Errorf("CumMax: operand must have a float or int dtype, got %s", operand.Shape.DType)
	}
	if operand.Shape.IsScalar() {
		return nil, errors.Errorf("CumMax: cannot perform CumMax on scalar shape %s", operand.Shape)
	}
	if axis < 0 || axis >= operand.Shape.Rank() {
		return nil, errors.Errorf("CumMax: axis %d out of range for rank %d", axis, operand.Shape.Rank())
	}
	node, _ := f.GetOrCreateNode(opType, operand.Shape, inputs, cumSumData{axis: axis, options: options})
	return node, nil
}

func execCumMax(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	_ = inputsOwned
	operand := inputs[0]
	data := node.Data.(cumSumData)
	output, err := backend.GetBuffer(operand.RawShape)
	if err != nil {
		return nil, err
	}
	cumMaxFn, err := cumMaxDTypeMap.Get(operand.RawShape.DType)
	if err != nil {
		return nil, err
	}
	typedCumMaxFn := cumMaxFn.(func(operand, output *gobackend.Buffer, data cumSumData) error)
	err = typedCumMaxFn(operand, output, data)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func execCumMaxGeneric[T gobackend.PODNumericConstraints](operand, output *gobackend.Buffer, data cumSumData) error {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	shape := operand.RawShape
	axis := data.axis
	exclusive := data.options.Exclusive
	reverse := data.options.Reverse
	lowest := shape.DType.LowestValue().(T)

	outerSize := 1
	for i := range axis {
		outerSize *= shape.Dimensions[i]
	}
	axisSize := shape.Dimensions[axis]
	innerSize := 1
	for i := axis + 1; i < shape.Rank(); i++ {
		innerSize *= shape.Dimensions[i]
	}
	stride := innerSize
	sliceStep := axisSize * innerSize

	for o := range outerSize {
		outerOffset := o * sliceStep
		for i := range innerSize {
			base := outerOffset + i
			if !reverse {
				if !exclusive {
					acc := lowest
					for a := range axisSize {
						idx := base + a*stride
						acc = max(acc, operandFlat[idx])
						outputFlat[idx] = acc
					}
				} else {
					acc := lowest
					for a := range axisSize {
						idx := base + a*stride
						val := operandFlat[idx]
						outputFlat[idx] = acc
						acc = max(acc, val)
					}
				}
			} else {
				if !exclusive {
					acc := lowest
					for a := axisSize - 1; a >= 0; a-- {
						idx := base + a*stride
						acc = max(acc, operandFlat[idx])
						outputFlat[idx] = acc
					}
				} else {
					acc := lowest
					for a := axisSize - 1; a >= 0; a-- {
						idx := base + a*stride
						val := operandFlat[idx]
						outputFlat[idx] = acc
						acc = max(acc, val)
					}
				}
			}
		}
	}
	return nil
}

func execCumMaxHalfPrecision[T gotype.HalfPrecision[T], P gotype.HalfPrecisionPtr[T]](operand, output *gobackend.Buffer, data cumSumData) error {
	operandFlat := operand.Flat.([]T)
	outputFlat := output.Flat.([]T)
	shape := operand.RawShape
	axis := data.axis
	exclusive := data.options.Exclusive
	reverse := data.options.Reverse
	lowest := float32(math.Inf(-1))

	outerSize := 1
	for i := range axis {
		outerSize *= shape.Dimensions[i]
	}
	axisSize := shape.Dimensions[axis]
	innerSize := 1
	for i := axis + 1; i < shape.Rank(); i++ {
		innerSize *= shape.Dimensions[i]
	}
	stride := innerSize
	sliceStep := axisSize * innerSize

	for o := range outerSize {
		outerOffset := o * sliceStep
		for i := range innerSize {
			base := outerOffset + i
			if !reverse {
				if !exclusive {
					acc := lowest
					for a := range axisSize {
						idx := base + a*stride
						acc = max(acc, operandFlat[idx].Float32())
						P(&outputFlat[idx]).SetFloat32(acc)
					}
				} else {
					acc := lowest
					for a := range axisSize {
						idx := base + a*stride
						val := operandFlat[idx].Float32()
						P(&outputFlat[idx]).SetFloat32(acc)
						acc = max(acc, val)
					}
				}
			} else {
				if !exclusive {
					acc := lowest
					for a := axisSize - 1; a >= 0; a-- {
						idx := base + a*stride
						acc = max(acc, operandFlat[idx].Float32())
						P(&outputFlat[idx]).SetFloat32(acc)
					}
				} else {
					acc := lowest
					for a := axisSize - 1; a >= 0; a-- {
						idx := base + a*stride
						val := operandFlat[idx].Float32()
						P(&outputFlat[idx]).SetFloat32(acc)
						acc = max(acc, val)
					}
				}
			}
		}
	}
	return nil
}

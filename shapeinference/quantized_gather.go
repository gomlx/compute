// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// QuantizedGatherInfo holds validated dimensions and resolved parameter block layouts
// for QuantizedGather.
type QuantizedGatherInfo struct {
	OutputShape     shapes.Shape
	K               int // Logical embedding dimension
	ScaleLayout     BlockParamLayout
	ZeroPointLayout BlockParamLayout
}

// QuantizedGather validates inputs and returns the output shape of QuantizedGather.
func QuantizedGather(x, indices shapes.Shape, xQuantization *QuantizationShapes) (shapes.Shape, error) {
	info, err := QuantizedGatherWithInfo(x, indices, xQuantization)
	if err != nil {
		return shapes.Invalid(), err
	}
	return info.OutputShape, nil
}

// QuantizedGatherWithInfo validates inputs and returns both the output shape and resolved
// scale/zero-point layouts for QuantizedGather.
func QuantizedGatherWithInfo(x, indices shapes.Shape, xQuantization *QuantizationShapes) (QuantizedGatherInfo, error) {
	var info QuantizedGatherInfo
	if !x.Ok() {
		return info, errors.Errorf("QuantizedGather: x shape is invalid")
	}
	if !indices.Ok() {
		return info, errors.Errorf("QuantizedGather: indices shape is invalid")
	}
	if x.Rank() != 2 || x.IsDynamic() || x.Dimensions[0] <= 0 || x.Dimensions[1] <= 0 {
		return info, errors.Errorf(
			"QuantizedGather: x must have static positive shape [vocabSize, ...], got %s", x)
	}
	if !indices.DType.IsInt() {
		return info, errors.Errorf(
			"QuantizedGather: indices must have integer DType, got %s", indices.DType)
	}
	if indices.Rank() < 1 || indices.Dimensions[indices.Rank()-1] != 1 {
		return info, errors.Errorf(
			"QuantizedGather: indices last dimension must be 1, got %s", indices)
	}

	var K int
	var outDType dtypes.DType
	switch {
	case xQuantization == nil:
		switch x.DType {
		case dtypes.Int2, dtypes.Int4, dtypes.Int8:
			outDType = dtypes.Int8
		case dtypes.Uint2, dtypes.Uint4, dtypes.Uint8:
			outDType = dtypes.Uint8
		default:
			return info, errors.Errorf(
				"QuantizedGather: when xQuantization is nil, x must have Int2/Uint2/Int4/Uint4/Int8/Uint8 DType, got %s", x.DType)
		}
		K = x.Dimensions[1]

	case xQuantization.Scheme == compute.QuantGGML:
		if x.DType != dtypes.Uint8 {
			return info, errors.Errorf(
				"QuantizedGather: QuantGGML x must have DType Uint8, got %s", x.DType)
		}
		if xQuantization.Scale.Ok() || xQuantization.ZeroPoint.Ok() {
			return info, errors.Errorf(
				"QuantizedGather: QuantGGML xQuantization must not set Scale or ZeroPoint")
		}
		var err error
		K, err = DeriveGGMLK("QuantizedGather", x.Dimensions[1], xQuantization.GGMLType)
		if err != nil {
			return info, err
		}
		outDType = dtypes.Float32

	case xQuantization.Scheme == compute.QuantLinear || xQuantization.Scheme == compute.QuantNF4:
		var err error
		K, _, info.ScaleLayout, info.ZeroPointLayout, err = validateQuantizedWeight(
			"QuantizedGather", x, xQuantization, compute.DenseLayoutOutputsInput)
		if err != nil {
			return info, err
		}
		outDType = xQuantization.Scale.DType

	default:
		return info, errors.Wrapf(compute.ErrNotImplemented,
			"QuantizedGather: unsupported quantization scheme %s", xQuantization.Scheme)
	}

	outDims := make([]int, indices.Rank())
	copy(outDims, indices.Dimensions)
	outDims[len(outDims)-1] = K
	var outAxisNames []string
	if indices.AxisNames != nil {
		outAxisNames = make([]string, indices.Rank())
		copy(outAxisNames[:len(outAxisNames)-1], indices.AxisNames[:len(indices.AxisNames)-1])
	}
	info.K = K
	info.OutputShape = shapes.Shape{
		DType:      outDType,
		Dimensions: outDims,
		AxisNames:  outAxisNames,
	}
	return info, nil
}

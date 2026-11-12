// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// QuantizationShapes is the shape-only counterpart of compute.Quantization for shape inference.
type QuantizationShapes struct {
	Scheme         compute.QuantizationScheme
	Scale          shapes.Shape // shapes.Invalid() if Scale == nil
	ZeroPoint      shapes.Shape // shapes.Invalid() if ZeroPoint == nil
	BlockAxis      int
	BlockSize      int
	GGMLType       compute.GGMLQuantType
	QuantizedDType dtypes.DType
}

// QuantizedDenseConfigShapes is the shape-only counterpart of compute.FusedQuantizedDenseConfig.
type QuantizedDenseConfigShapes struct {
	WeightQuantization *QuantizationShapes
	XQuantization      *QuantizationShapes // nil when x is floating-point
	OutputQuantization *QuantizationShapes // nil when output is floating-point
	QuantizedMatmul    bool
	Activation         compute.ActivationConfig
	WeightLayout       compute.DenseLayout
}

// BlockParamLayout describes how a 2D weight tensor [D0, D1] maps (row, col) coordinates
// into a flat Scale or ZeroPoint slice:
//
//	idx(row, col) = (row / RowBlockSize) * NumColBlocks + (col / ColBlockSize)
type BlockParamLayout struct {
	Present      bool
	RowBlockSize int
	ColBlockSize int
	NumColBlocks int
}

// Index returns the flat parameter index for (row, col) in [D0, D1].
func (l BlockParamLayout) Index(row, col int) int {
	return (row/l.RowBlockSize)*l.NumColBlocks + (col / l.ColBlockSize)
}

// ActivationParamKind classifies how a Scale or ZeroPoint broadcasts over a flattened [M, F] activation/output tensor.
type ActivationParamKind uint8

const (
	ActParamNone ActivationParamKind = iota
	ActParamPerTensor
	ActParamPerColBlock
	ActParamPerRow
	ActParamPerRowColBlock
)

// ActivationParamLayout describes how a Scale or ZeroPoint maps (row m, feature col) on [M, F]
// (including when M comes from dynamic leading dimensions).
type ActivationParamLayout struct {
	Kind         ActivationParamKind
	ColBlockSize int
	NumColBlocks int
}

// Present reports whether the parameter tensor was provided.
func (l ActivationParamLayout) Present() bool {
	return l.Kind != ActParamNone
}

// Index returns the flat parameter index for (m, col) in [M, F].
func (l ActivationParamLayout) Index(m, col int) int {
	switch l.Kind {
	case ActParamPerTensor:
		return 0
	case ActParamPerColBlock:
		return col / l.ColBlockSize
	case ActParamPerRow:
		return m
	case ActParamPerRowColBlock:
		return m*l.NumColBlocks + col/l.ColBlockSize
	default:
		return 0
	}
}

// QuantizedDenseInfo holds validated dimensions and resolved parameter block layouts for FusedQuantizedDense
// and FusedQuantizedDenseVJP.
type QuantizedDenseInfo struct {
	OutputShape           shapes.Shape
	K                     int // Logical input-features (contracting) dimension
	N                     int // Logical output-features dimension
	FloatComputeDType     dtypes.DType
	WeightScaleLayout     BlockParamLayout
	WeightZeroPointLayout BlockParamLayout
	XScaleLayout          ActivationParamLayout
	XZeroPointLayout      ActivationParamLayout
	OutputScaleLayout     ActivationParamLayout
	OutputZeroPointLayout ActivationParamLayout
}

// FusedQuantizedDense validates inputs and returns the output shape of FusedQuantizedDense.
func FusedQuantizedDense(x, weight, bias shapes.Shape, options QuantizedDenseConfigShapes) (shapes.Shape, error) {
	info, err := FusedQuantizedDenseWithInfo(x, weight, bias, options)
	if err != nil {
		return shapes.Invalid(), err
	}
	return info.OutputShape, nil
}

// FusedQuantizedDenseWithInfo validates inputs and returns both the output shape and resolved layout metadata.
func FusedQuantizedDenseWithInfo(x, weight, bias shapes.Shape, options QuantizedDenseConfigShapes) (QuantizedDenseInfo, error) {
	var info QuantizedDenseInfo
	if !x.Ok() {
		return info, errors.Errorf("FusedQuantizedDense: x shape is invalid")
	}
	if !weight.Ok() {
		return info, errors.Errorf("FusedQuantizedDense: weight shape is invalid")
	}
	if x.Rank() < 1 {
		return info, errors.Errorf("FusedQuantizedDense: x must have rank >= 1, got %d", x.Rank())
	}
	if options.WeightQuantization == nil {
		return info, errors.Errorf("FusedQuantizedDense: options.WeightQuantization must not be nil")
	}

	// Validate activation.
	if options.Activation.Type == compute.ActivationSwiGLU {
		return info, errors.Wrapf(compute.ErrNotImplemented,
			"FusedQuantizedDense does not support SwiGLU activation due to output shape change (use FusedActivation separately)")
	}
	if options.Activation.Type < compute.ActivationNone || options.Activation.Type > compute.ActivationSwiGLU {
		return info, errors.Wrapf(compute.ErrNotImplemented,
			"FusedQuantizedDense: unsupported activation %v", options.Activation.Type)
	}

	// Resolve weight dimensions K, N, and weight Scale/ZeroPoint layouts.
	K, N, wScaleLayout, wZPLayout, err := validateQuantizedWeight("FusedQuantizedDense", weight, options.WeightQuantization, options.WeightLayout)
	if err != nil {
		return info, err
	}
	info.K = K
	info.N = N
	info.WeightScaleLayout = wScaleLayout
	info.WeightZeroPointLayout = wZPLayout

	wq := options.WeightQuantization
	xq := options.XQuantization
	lastDimX := x.Dimensions[x.Rank()-1]
	if lastDimX == shapes.DynamicDim {
		return info, errors.Errorf("FusedQuantizedDense: x's last dimension (contracting axis) cannot be dynamic")
	}

	// Validate x and XQuantization.
	if xq == nil {
		if !isSupportedFloatDType(x.DType) {
			return info, errors.Errorf("FusedQuantizedDense: floating-point x must be Float32, Float16, or BFloat16 when XQuantization is nil, got %s", x.DType)
		}
		if lastDimX != K {
			return info, errors.Errorf("FusedQuantizedDense: x's contracting dimension (%d) must match weight's K (%d)", lastDimX, K)
		}
		info.FloatComputeDType = x.DType
	} else {
		info.FloatComputeDType = dtypes.Float32
		switch xq.Scheme {
		case compute.QuantLinear:
			if wq.Scheme != compute.QuantLinear {
				return info, errors.Wrapf(compute.ErrNotImplemented,
					"FusedQuantizedDense: QuantLinear XQuantization requires QuantLinear WeightQuantization, got %s", wq.Scheme)
			}
			if x.DType != dtypes.Int8 && x.DType != dtypes.Uint8 {
				return info, errors.Wrapf(compute.ErrNotImplemented,
					"FusedQuantizedDense: QuantLinear XQuantization requires x DType Int8 or Uint8, got %s", x.DType)
			}
			if lastDimX != K {
				return info, errors.Errorf("FusedQuantizedDense: x's contracting dimension (%d) must match weight's K (%d)", lastDimX, K)
			}
			xScaleLayout, xZPLayout, err := validateActivationQuantization("FusedQuantizedDense (XQuantization)", x, K, xq, false)
			if err != nil {
				return info, err
			}
			info.XScaleLayout = xScaleLayout
			info.XZeroPointLayout = xZPLayout

		case compute.QuantGGML:
			if wq.Scheme != compute.QuantGGML {
				return info, errors.Wrapf(compute.ErrNotImplemented,
					"FusedQuantizedDense: QuantGGML XQuantization requires QuantGGML WeightQuantization, got %s", wq.Scheme)
			}
			if x.DType != dtypes.Uint8 {
				return info, errors.Errorf("FusedQuantizedDense: QuantGGML x must have DType Uint8, got %s", x.DType)
			}
			if xq.Scale.Ok() || xq.ZeroPoint.Ok() {
				return info, errors.Errorf("FusedQuantizedDense: QuantGGML XQuantization must not set Scale or ZeroPoint")
			}
			if err := validateGGMLActivationPairing(xq.GGMLType, wq.GGMLType); err != nil {
				return info, err
			}
			expectedBytesPerRowX := (K / xq.GGMLType.ValuesPerBlock()) * xq.GGMLType.BytesPerBlock()
			if lastDimX != expectedBytesPerRowX {
				return info, errors.Errorf(
					"FusedQuantizedDense: QuantGGML(%s) x last dim must be %d bytes for K=%d, got %d",
					xq.GGMLType, expectedBytesPerRowX, K, lastDimX)
			}

		default:
			return info, errors.Wrapf(compute.ErrNotImplemented,
				"FusedQuantizedDense: unsupported XQuantization scheme %s", xq.Scheme)
		}
	}

	// Build output dimensions & axis names: [batch..., N].
	batchRank := x.Rank() - 1
	outDims := make([]int, batchRank+1)
	copy(outDims, x.Dimensions[:batchRank])
	outDims[batchRank] = N

	var outAxisNames []string
	var weightOutAxisName string
	if weight.AxisNames != nil {
		if options.WeightLayout == compute.DenseLayoutInputOutputs {
			weightOutAxisName = weight.AxisNames[1]
		} else {
			weightOutAxisName = weight.AxisNames[0]
		}
	}
	if x.AxisNames != nil || weightOutAxisName != "" {
		outAxisNames = make([]string, batchRank+1)
		if x.AxisNames != nil {
			copy(outAxisNames, x.AxisNames[:batchRank])
		}
		outAxisNames[batchRank] = weightOutAxisName
	}

	outDType := info.FloatComputeDType
	floatOutShape := shapes.Shape{
		DType:      info.FloatComputeDType,
		Dimensions: outDims,
		AxisNames:  outAxisNames,
	}

	// Validate bias if provided.
	if bias.Ok() {
		if bias.IsDynamic() {
			return info, errors.Errorf("FusedQuantizedDense: bias must have static shape, got %s", bias)
		}
		if xq == nil {
			if bias.DType != x.DType {
				return info, errors.Errorf("FusedQuantizedDense: bias must have same DType as x (%s), got %s", x.DType, bias.DType)
			}
		} else {
			validBiasDType := isSupportedFloatDType(bias.DType) || (xq.Scheme == compute.QuantLinear && bias.DType == dtypes.Int32)
			if !validBiasDType {
				return info, errors.Errorf(
					"FusedQuantizedDense: with XQuantization(%s), bias must be float or Int32 (for QuantLinear), got %s",
					xq.Scheme, bias.DType)
			}
			if bias.DType == dtypes.Int32 {
				wKBlockSize := info.WeightScaleLayout.RowBlockSize
				if options.WeightLayout == compute.DenseLayoutOutputsInput {
					wKBlockSize = info.WeightScaleLayout.ColBlockSize
				}
				if info.XScaleLayout.ColBlockSize != K || wKBlockSize != K {
					return info, errors.Errorf(
						"FusedQuantizedDense: Int32 bias requires unblocked K scales (per-tensor or per-row XQuantization.Scale and per-tensor or per-channel WeightQuantization.Scale); use a floating-point bias when K is partitioned into multiple scale blocks")
				}
			}
		}
		if bias.Rank() != 1 || bias.Dimensions[0] != N {
			return info, errors.Errorf("FusedQuantizedDense: bias shape %s must be [%d]", bias, N)
		}
	}

	// Validate OutputQuantization if provided.
	if oq := options.OutputQuantization; oq != nil {
		if oq.Scheme != compute.QuantLinear {
			return info, errors.Wrapf(compute.ErrNotImplemented,
				"FusedQuantizedDense: OutputQuantization only supports QuantLinear, got %s", oq.Scheme)
		}
		oScaleLayout, oZPLayout, err := validateActivationQuantization("FusedQuantizedDense (OutputQuantization)", floatOutShape, N, oq, true)
		if err != nil {
			return info, err
		}
		info.OutputScaleLayout = oScaleLayout
		info.OutputZeroPointLayout = oZPLayout

		resolvedDType, err := resolveOutputQuantizedDType(oq)
		if err != nil {
			return info, err
		}
		outDType = resolvedDType
	}

	info.OutputShape = shapes.Shape{
		DType:      outDType,
		Dimensions: outDims,
		AxisNames:  outAxisNames,
	}
	return info, nil
}

// FusedQuantizedDenseVJP validates inputs and returns the gradient shapes (dx, dBias) for FusedQuantizedDenseVJP.
//
// Memory-saving nil inputs:
//   - x may be shapes.Invalid() when !options.Activation.Type.VJPRequiresInput(); dx's batch dimensions and
//     float DType are then inferred from dOutput.
//   - y may be shapes.Invalid() when options.Activation.Type == compute.ActivationNone.
func FusedQuantizedDenseVJP(x, weight, bias, y, dOutput shapes.Shape, options QuantizedDenseConfigShapes) (dx, dBias shapes.Shape, err error) {
	if options.XQuantization != nil || options.OutputQuantization != nil {
		return shapes.Invalid(), shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedQuantizedDenseVJP does not support XQuantization or OutputQuantization (discrete quantized activations/outputs are non-differentiable)")
	}
	if options.Activation.Type.VJPRequiresInput() {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedQuantizedDenseVJP does not support activation %s because it requires the pre-activation input (VJPRequiresInput() == true); "+
				"use FusedQuantizedDense with ActivationNone followed by FusedActivation instead",
			options.Activation.Type)
	}
	if options.Activation.Type < compute.ActivationNone || options.Activation.Type > compute.ActivationSwiGLU {
		return shapes.Invalid(), shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedQuantizedDenseVJP: unsupported activation %v", options.Activation.Type)
	}
	if options.Activation.Type == compute.ActivationSwiGLU {
		return shapes.Invalid(), shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedQuantizedDenseVJP does not support SwiGLU activation")
	}
	if !dOutput.Ok() {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf("FusedQuantizedDenseVJP: dOutput shape is invalid")
	}
	if dOutput.Rank() < 1 {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf("FusedQuantizedDenseVJP: dOutput must have rank >= 1, got %d", dOutput.Rank())
	}
	if !isSupportedFloatDType(dOutput.DType) {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedQuantizedDenseVJP: dOutput must be Float32, Float16, or BFloat16, got %s", dOutput.DType)
	}
	if options.WeightQuantization == nil {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf("FusedQuantizedDenseVJP: options.WeightQuantization must not be nil")
	}

	K, N, _, _, err := validateQuantizedWeight("FusedQuantizedDenseVJP", weight, options.WeightQuantization, options.WeightLayout)
	if err != nil {
		return shapes.Invalid(), shapes.Invalid(), err
	}

	// Build effective x shape (either from x directly, or derived from dOutput + K when x was omitted).
	batchRank := dOutput.Rank() - 1
	if dOutput.Dimensions[batchRank] != N {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedQuantizedDenseVJP: dOutput's last dimension (%d) must match weight's output features N (%d)",
			dOutput.Dimensions[batchRank], N)
	}

	effectiveX := x
	if !effectiveX.Ok() {
		xDims := make([]int, batchRank+1)
		copy(xDims, dOutput.Dimensions[:batchRank])
		xDims[batchRank] = K
		var xAxisNames []string
		if dOutput.AxisNames != nil {
			xAxisNames = make([]string, batchRank+1)
			copy(xAxisNames, dOutput.AxisNames[:batchRank])
		}
		effectiveX = shapes.Shape{
			DType:      dOutput.DType,
			Dimensions: xDims,
			AxisNames:  xAxisNames,
		}
	}

	expectedY, err := FusedQuantizedDense(effectiveX, weight, bias, options)
	if err != nil {
		return shapes.Invalid(), shapes.Invalid(), errors.Wrapf(err, "FusedQuantizedDenseVJP forward shape validation failed")
	}

	if dOutput.DType != expectedY.DType || !shapesMatch(dOutput, expectedY) {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedQuantizedDenseVJP: dOutput shape %s does not match expected output shape %s", dOutput, expectedY)
	}

	// Validate y: required when Activation != ActivationNone, optional when Activation == ActivationNone.
	if y.Ok() {
		if y.DType != expectedY.DType || !shapesMatch(y, expectedY) {
			return shapes.Invalid(), shapes.Invalid(), errors.Errorf(
				"FusedQuantizedDenseVJP: y shape %s does not match expected output shape %s", y, expectedY)
		}
	} else if options.Activation.Type != compute.ActivationNone {
		return shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedQuantizedDenseVJP: y shape is required when activation is %s (only optional for ActivationNone)",
			options.Activation.Type)
	}

	dx = effectiveX.Clone()
	if bias.Ok() {
		dBias = bias.Clone()
	} else {
		dBias = shapes.Invalid()
	}
	return dx, dBias, nil
}

// QuantizedEmbeddingLookup validates inputs and returns the output shape of QuantizedEmbeddingLookup.
func QuantizedEmbeddingLookup(data, indices shapes.Shape, dataQuantization *QuantizationShapes) (shapes.Shape, error) {
	if !data.Ok() {
		return shapes.Invalid(), errors.Errorf("QuantizedEmbeddingLookup: data shape is invalid")
	}
	if !indices.Ok() {
		return shapes.Invalid(), errors.Errorf("QuantizedEmbeddingLookup: indices shape is invalid")
	}
	if dataQuantization == nil {
		return shapes.Invalid(), errors.Errorf("QuantizedEmbeddingLookup: dataQuantization must not be nil")
	}
	if dataQuantization.Scheme != compute.QuantGGML {
		return shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"QuantizedEmbeddingLookup: only QuantGGML scheme is supported, got %s", dataQuantization.Scheme)
	}
	if data.Rank() != 2 || data.DType != dtypes.Uint8 || data.IsDynamic() {
		return shapes.Invalid(), errors.Errorf(
			"QuantizedEmbeddingLookup: data must have static shape [vocabSize, bytesPerRow] Uint8, got %s", data)
	}
	if !indices.DType.IsInt() {
		return shapes.Invalid(), errors.Errorf(
			"QuantizedEmbeddingLookup: indices must have integer DType, got %s", indices.DType)
	}
	if indices.Rank() < 1 || indices.Dimensions[indices.Rank()-1] != 1 {
		return shapes.Invalid(), errors.Errorf(
			"QuantizedEmbeddingLookup: indices last dimension must be 1, got %s", indices)
	}

	K, err := DeriveGGMLK("QuantizedEmbeddingLookup", data.Dimensions[1], dataQuantization.GGMLType)
	if err != nil {
		return shapes.Invalid(), err
	}

	outDims := make([]int, indices.Rank())
	copy(outDims, indices.Dimensions)
	outDims[len(outDims)-1] = K
	var outAxisNames []string
	if indices.AxisNames != nil {
		outAxisNames = make([]string, indices.Rank())
		copy(outAxisNames[:len(outAxisNames)-1], indices.AxisNames[:len(indices.AxisNames)-1])
	}
	return shapes.Shape{
		DType:      dtypes.Float32,
		Dimensions: outDims,
		AxisNames:  outAxisNames,
	}, nil
}

// DeriveGGMLK validates a GGMLQuantType and derives the logical element count K from bytesPerRow.
func DeriveGGMLK(opName string, bytesPerRow int, ggmlType compute.GGMLQuantType) (int, error) {
	bpb := ggmlType.BytesPerBlock()
	vpb := ggmlType.ValuesPerBlock()
	if bpb <= 0 || vpb <= 0 {
		return 0, errors.Wrapf(compute.ErrNotImplemented, "%s: unsupported GGML quantization type %v", opName, ggmlType)
	}
	if bytesPerRow <= 0 || bytesPerRow%bpb != 0 {
		return 0, errors.Errorf("%s: bytesPerRow (%d) must be positive and divisible by %s bytesPerBlock (%d)",
			opName, bytesPerRow, ggmlType, bpb)
	}
	return (bytesPerRow / bpb) * vpb, nil
}

func isSupportedFloatDType(dt dtypes.DType) bool {
	switch dt {
	case dtypes.Float32, dtypes.Float16, dtypes.BFloat16:
		return true
	default:
		return false
	}
}

func isSupportedLinearWeightDType(dt dtypes.DType) bool {
	switch dt {
	case dtypes.Int2, dtypes.Uint2, dtypes.Int4, dtypes.Uint4, dtypes.Int8, dtypes.Uint8:
		return true
	default:
		return false
	}
}

func isSupportedZeroPointDType(dt dtypes.DType) bool {
	return isSupportedLinearWeightDType(dt) || dt == dtypes.Int32 || isSupportedFloatDType(dt)
}

func validateGGMLActivationPairing(xType, wType compute.GGMLQuantType) error {
	switch wType.ValuesPerBlock() {
	case 32:
		if xType != compute.GGMLQ8_0 {
			return errors.Wrapf(compute.ErrNotImplemented,
				"FusedQuantizedDense: 32-value GGML weight type %s requires XQuantization GGMLQ8_0, got %s", wType, xType)
		}
	case 256:
		if xType != compute.GGMLQ8_K {
			return errors.Wrapf(compute.ErrNotImplemented,
				"FusedQuantizedDense: 256-value K-quant GGML weight type %s requires XQuantization GGMLQ8_K, got %s", wType, xType)
		}
	default:
		return errors.Wrapf(compute.ErrNotImplemented,
			"FusedQuantizedDense: unsupported GGML weight type %v", wType)
	}
	return nil
}

func validateQuantizedWeight(opName string, weight shapes.Shape, wq *QuantizationShapes, layout compute.DenseLayout) (
	K, N int, scaleLayout, zpLayout BlockParamLayout, err error,
) {
	if weight.Rank() != 2 {
		return 0, 0, scaleLayout, zpLayout, errors.Errorf("%s: weight must have rank 2, got %s", opName, weight)
	}
	if weight.IsDynamic() {
		return 0, 0, scaleLayout, zpLayout, errors.Errorf("%s: weight must have static shape, got %s", opName, weight)
	}
	if weight.Dimensions[0] <= 0 || weight.Dimensions[1] <= 0 {
		return 0, 0, scaleLayout, zpLayout, errors.Errorf("%s: weight dimensions must be positive, got %s", opName, weight)
	}
	if layout != compute.DenseLayoutInputOutputs && layout != compute.DenseLayoutOutputsInput {
		return 0, 0, scaleLayout, zpLayout, errors.Errorf("%s: unknown WeightLayout %v", opName, layout)
	}

	switch wq.Scheme {
	case compute.QuantGGML:
		if layout != compute.DenseLayoutOutputsInput {
			return 0, 0, scaleLayout, zpLayout, errors.Errorf(
				"%s: QuantGGML requires WeightLayout == DenseLayoutOutputsInput ([N, bytesPerRow]), got %s", opName, layout)
		}
		if weight.DType != dtypes.Uint8 {
			return 0, 0, scaleLayout, zpLayout, errors.Errorf(
				"%s: QuantGGML weight must have DType Uint8, got %s", opName, weight.DType)
		}
		if wq.Scale.Ok() || wq.ZeroPoint.Ok() {
			return 0, 0, scaleLayout, zpLayout, errors.Errorf(
				"%s: QuantGGML WeightQuantization must not set Scale or ZeroPoint", opName)
		}
		N = weight.Dimensions[0]
		K, err = DeriveGGMLK(opName, weight.Dimensions[1], wq.GGMLType)
		if err != nil {
			return 0, 0, scaleLayout, zpLayout, err
		}
		return K, N, scaleLayout, zpLayout, nil

	case compute.QuantLinear, compute.QuantNF4:
		if wq.Scheme == compute.QuantLinear {
			if !isSupportedLinearWeightDType(weight.DType) {
				return 0, 0, scaleLayout, zpLayout, errors.Errorf(
					"%s: QuantLinear weight must have DType Int2/Uint2/Int4/Uint4/Int8/Uint8, got %s", opName, weight.DType)
			}
		} else {
			switch weight.DType {
			case dtypes.Int4, dtypes.Uint4, dtypes.Int8, dtypes.Uint8:
			default:
				return 0, 0, scaleLayout, zpLayout, errors.Errorf(
					"%s: QuantNF4 weight must have DType Int4/Uint4/Int8/Uint8, got %s", opName, weight.DType)
			}
			if wq.ZeroPoint.Ok() {
				return 0, 0, scaleLayout, zpLayout, errors.Errorf(
					"%s: ZeroPoint must be nil/invalid for QuantNF4 scheme", opName)
			}
		}

		if layout == compute.DenseLayoutInputOutputs {
			K = weight.Dimensions[0]
			N = weight.Dimensions[1]
		} else {
			N = weight.Dimensions[0]
			K = weight.Dimensions[1]
		}

		if !wq.Scale.Ok() {
			return 0, 0, scaleLayout, zpLayout, errors.Errorf("%s: WeightQuantization.Scale is required for %s", opName, wq.Scheme)
		}
		if wq.Scale.IsDynamic() {
			return 0, 0, scaleLayout, zpLayout, errors.Errorf("%s: WeightQuantization.Scale must have static shape, got %s", opName, wq.Scale)
		}
		if !isSupportedFloatDType(wq.Scale.DType) {
			return 0, 0, scaleLayout, zpLayout, errors.Errorf(
				"%s: WeightQuantization.Scale must be Float32, Float16, or BFloat16, got %s", opName, wq.Scale.DType)
		}

		d0, d1 := weight.Dimensions[0], weight.Dimensions[1]
		scaleLayout, err = resolveWeightParamLayout(opName, "Scale", wq.Scale, d0, d1, N, layout, wq.BlockAxis, wq.BlockSize)
		if err != nil {
			return 0, 0, scaleLayout, zpLayout, err
		}

		if wq.ZeroPoint.Ok() {
			if wq.ZeroPoint.IsDynamic() {
				return 0, 0, scaleLayout, zpLayout, errors.Errorf("%s: WeightQuantization.ZeroPoint must have static shape, got %s", opName, wq.ZeroPoint)
			}
			if !isSupportedZeroPointDType(wq.ZeroPoint.DType) {
				return 0, 0, scaleLayout, zpLayout, errors.Errorf(
					"%s: unsupported WeightQuantization.ZeroPoint DType %s", opName, wq.ZeroPoint.DType)
			}
			zpLayout, err = resolveWeightParamLayout(opName, "ZeroPoint", wq.ZeroPoint, d0, d1, N, layout, wq.BlockAxis, wq.BlockSize)
			if err != nil {
				return 0, 0, scaleLayout, zpLayout, err
			}
			if !isWeightZPBroadcastCompatible(zpLayout, scaleLayout, d0, d1) {
				return 0, 0, scaleLayout, zpLayout, errors.Errorf(
					"%s: WeightQuantization.ZeroPoint shape %s is not broadcast-compatible with Scale shape %s",
					opName, wq.ZeroPoint, wq.Scale)
			}
		}
		return K, N, scaleLayout, zpLayout, nil

	default:
		return 0, 0, scaleLayout, zpLayout, errors.Wrapf(compute.ErrNotImplemented,
			"%s: unsupported quantization scheme %v", opName, wq.Scheme)
	}
}

func resolveWeightParamLayout(
	opName, paramName string,
	param shapes.Shape,
	d0, d1, N int,
	layout compute.DenseLayout,
	blockAxis, blockSize int,
) (BlockParamLayout, error) {
	if blockSize < 0 {
		return BlockParamLayout{}, errors.Errorf("%s: BlockSize must be >= 0, got %d", opName, blockSize)
	}

	// Scalar [] or [1] or [1, 1] -> per-tensor.
	if param.Size() == 1 && param.Rank() <= 2 {
		return BlockParamLayout{
			Present:      true,
			RowBlockSize: d0,
			ColBlockSize: d1,
			NumColBlocks: 1,
		}, nil
	}

	switch param.Rank() {
	case 1:
		if param.Dimensions[0] != N {
			return BlockParamLayout{}, errors.Errorf(
				"%s: 1D WeightQuantization.%s shape %s must have length N=%d (or 1)", opName, paramName, param, N)
		}
		if layout == compute.DenseLayoutInputOutputs {
			// Weight is [K, N]: broadcast along K.
			return BlockParamLayout{
				Present:      true,
				RowBlockSize: d0,
				ColBlockSize: 1,
				NumColBlocks: N,
			}, nil
		}
		// Weight is [N, K]: broadcast along K.
		return BlockParamLayout{
			Present:      true,
			RowBlockSize: 1,
			ColBlockSize: d1,
			NumColBlocks: 1,
		}, nil

	case 2:
		s0, s1 := param.Dimensions[0], param.Dimensions[1]
		switch blockAxis {
		case 0:
			// Also accept [D0, 1] when BlockSize == 0 (unambiguous full-row broadcast along axis 1).
			if s1 != d1 && blockSize == 0 && s0 == d0 && s1 == 1 {
				return BlockParamLayout{
					Present:      true,
					RowBlockSize: 1,
					ColBlockSize: d1,
					NumColBlocks: 1,
				}, nil
			}
			if s1 != d1 {
				return BlockParamLayout{}, errors.Errorf(
					"%s: 2D WeightQuantization.%s shape %s with BlockAxis=0 must have dim 1 == %d", opName, paramName, param, d1)
			}
			b := blockSize
			if b == 0 {
				if s0 <= 0 || d0%s0 != 0 {
					return BlockParamLayout{}, errors.Errorf(
						"%s: cannot infer BlockSize for WeightQuantization.%s shape %s on [%d, %d] (dim 0 %d not divisible by %d; set BlockSize explicitly)",
						opName, paramName, param, d0, d1, d0, s0)
				}
				b = d0 / s0
			} else {
				expectedS0 := (d0 + b - 1) / b
				if s0 != expectedS0 {
					return BlockParamLayout{}, errors.Errorf(
						"%s: WeightQuantization.%s shape %s with BlockAxis=0, BlockSize=%d expected [%d, %d]",
						opName, paramName, param, b, expectedS0, d1)
				}
			}
			return BlockParamLayout{
				Present:      true,
				RowBlockSize: b,
				ColBlockSize: 1,
				NumColBlocks: d1,
			}, nil

		case 1:
			if s0 != d0 && blockSize == 0 && s0 == 1 && s1 == d1 {
				return BlockParamLayout{
					Present:      true,
					RowBlockSize: d0,
					ColBlockSize: 1,
					NumColBlocks: d1,
				}, nil
			}
			if s0 != d0 {
				return BlockParamLayout{}, errors.Errorf(
					"%s: 2D WeightQuantization.%s shape %s with BlockAxis=1 must have dim 0 == %d", opName, paramName, param, d0)
			}
			b := blockSize
			if b == 0 {
				if s1 <= 0 || d1%s1 != 0 {
					return BlockParamLayout{}, errors.Errorf(
						"%s: cannot infer BlockSize for WeightQuantization.%s shape %s on [%d, %d] (dim 1 %d not divisible by %d; set BlockSize explicitly)",
						opName, paramName, param, d0, d1, d1, s1)
				}
				b = d1 / s1
			} else {
				expectedS1 := (d1 + b - 1) / b
				if s1 != expectedS1 {
					return BlockParamLayout{}, errors.Errorf(
						"%s: WeightQuantization.%s shape %s with BlockAxis=1, BlockSize=%d expected [%d, %d]",
						opName, paramName, param, b, d0, expectedS1)
				}
			}
			return BlockParamLayout{
				Present:      true,
				RowBlockSize: 1,
				ColBlockSize: b,
				NumColBlocks: s1,
			}, nil

		default:
			return BlockParamLayout{}, errors.Errorf(
				"%s: WeightQuantization.BlockAxis must be 0 or 1, got %d", opName, blockAxis)
		}

	default:
		return BlockParamLayout{}, errors.Errorf(
			"%s: WeightQuantization.%s must have rank 0, 1, or 2, got %s", opName, paramName, param)
	}
}

func isWeightZPBroadcastCompatible(zp, scale BlockParamLayout, d0, d1 int) bool {
	rowOK := zp.RowBlockSize == d0 || (zp.RowBlockSize >= scale.RowBlockSize && zp.RowBlockSize%scale.RowBlockSize == 0)
	colOK := zp.ColBlockSize == d1 || (zp.ColBlockSize >= scale.ColBlockSize && zp.ColBlockSize%scale.ColBlockSize == 0)
	return rowOK && colOK
}

func validateActivationQuantization(
	opName string,
	tensor shapes.Shape,
	features int,
	q *QuantizationShapes,
	isOutput bool,
) (scaleLayout, zpLayout ActivationParamLayout, err error) {
	if !q.Scale.Ok() {
		return scaleLayout, zpLayout, errors.Errorf("%s: Scale must be provided", opName)
	}
	if !isSupportedFloatDType(q.Scale.DType) {
		return scaleLayout, zpLayout, errors.Errorf("%s: Scale must be Float32, Float16, or BFloat16, got %s", opName, q.Scale.DType)
	}
	scaleLayout, err = resolveActivationParamLayout(opName, "Scale", tensor, features, q.Scale, q.BlockSize)
	if err != nil {
		return scaleLayout, zpLayout, err
	}

	if q.ZeroPoint.Ok() {
		if isOutput {
			switch q.ZeroPoint.DType {
			case dtypes.Int8, dtypes.Uint8, dtypes.Int32, dtypes.Float32, dtypes.Float16, dtypes.BFloat16:
			default:
				return scaleLayout, zpLayout, errors.Errorf("%s: unsupported ZeroPoint DType %s", opName, q.ZeroPoint.DType)
			}
		} else {
			if !isSupportedZeroPointDType(q.ZeroPoint.DType) {
				return scaleLayout, zpLayout, errors.Errorf("%s: unsupported ZeroPoint DType %s", opName, q.ZeroPoint.DType)
			}
		}
		zpLayout, err = resolveActivationParamLayout(opName, "ZeroPoint", tensor, features, q.ZeroPoint, q.BlockSize)
		if err != nil {
			return scaleLayout, zpLayout, err
		}
		if !isActivationZPBroadcastCompatible(zpLayout, scaleLayout, features) {
			return scaleLayout, zpLayout, errors.Errorf(
				"%s: ZeroPoint shape %s is not broadcast-compatible with Scale shape %s",
				opName, q.ZeroPoint, q.Scale)
		}
	}
	return scaleLayout, zpLayout, nil
}

func resolveActivationParamLayout(
	opName, paramName string,
	tensor shapes.Shape,
	features int,
	param shapes.Shape,
	blockSize int,
) (ActivationParamLayout, error) {
	if blockSize < 0 {
		return ActivationParamLayout{}, errors.Errorf("%s: BlockSize must be >= 0, got %d", opName, blockSize)
	}

	// Scalar [] or [1] -> per-tensor.
	if !param.IsDynamic() && param.Size() == 1 && param.Rank() <= tensor.Rank() {
		allOnes := true
		for _, d := range param.Dimensions {
			if d != 1 {
				allOnes = false
				break
			}
		}
		if allOnes {
			return ActivationParamLayout{
				Kind:         ActParamPerTensor,
				ColBlockSize: features,
				NumColBlocks: 1,
			}, nil
		}
	}

	// 1D [features] -> per-feature across all rows.
	if param.Rank() == 1 && !param.IsDynamic() && param.Dimensions[0] == features {
		return ActivationParamLayout{
			Kind:         ActParamPerColBlock,
			ColBlockSize: 1,
			NumColBlocks: features,
		}, nil
	}

	batchRank := tensor.Rank() - 1

	// Rank == batchRank (e.g. [batch...], unambiguous when batchRank > 1 or dim != features).
	if batchRank >= 1 && param.Rank() == batchRank {
		batchAxesShape := shapes.Shape{
			DType:      tensor.DType,
			Dimensions: tensor.Dimensions[:batchRank],
			AxisNames:  sliceAxisNames(tensor.AxisNames, batchRank),
		}
		if shapesMatch(param, batchAxesShape) {
			return ActivationParamLayout{
				Kind:         ActParamPerRow,
				ColBlockSize: features,
				NumColBlocks: 1,
			}, nil
		}
	}

	// Same rank as tensor: [b0, ..., b_{R-2}, sLast].
	if param.Rank() == tensor.Rank() {
		sLast := param.Dimensions[batchRank]
		if sLast == shapes.DynamicDim || sLast <= 0 {
			return ActivationParamLayout{}, errors.Errorf(
				"%s: %s last dimension cannot be dynamic or <= 0, got %s", opName, paramName, param)
		}

		allBatchOne := true
		for i := range batchRank {
			if param.Dimensions[i] != 1 {
				allBatchOne = false
				break
			}
		}

		batchMatches := true
		for i := range batchRank {
			pd := param.Dimensions[i]
			td := tensor.Dimensions[i]
			if pd == shapes.DynamicDim && td == shapes.DynamicDim {
				if !shapes.AxisNameEqual(param.AxisName(i), tensor.AxisName(i)) {
					batchMatches = false
					break
				}
			} else if pd != shapes.DynamicDim && td != shapes.DynamicDim && pd != td {
				batchMatches = false
				break
			}
		}

		if !allBatchOne && !batchMatches {
			return ActivationParamLayout{}, errors.Errorf(
				"%s: %s leading dimensions in %s must either all be 1 or match tensor batch dimensions %v",
				opName, paramName, param, tensor.Dimensions[:batchRank])
		}

		colBlock, err := resolveFeatureBlockSize(opName, paramName, param, features, sLast, blockSize)
		if err != nil {
			return ActivationParamLayout{}, err
		}

		if allBatchOne {
			if sLast == 1 {
				return ActivationParamLayout{
					Kind:         ActParamPerTensor,
					ColBlockSize: features,
					NumColBlocks: 1,
				}, nil
			}
			return ActivationParamLayout{
				Kind:         ActParamPerColBlock,
				ColBlockSize: colBlock,
				NumColBlocks: sLast,
			}, nil
		}

		if sLast == 1 {
			return ActivationParamLayout{
				Kind:         ActParamPerRow,
				ColBlockSize: features,
				NumColBlocks: 1,
			}, nil
		}
		return ActivationParamLayout{
			Kind:         ActParamPerRowColBlock,
			ColBlockSize: colBlock,
			NumColBlocks: sLast,
		}, nil
	}

	return ActivationParamLayout{}, errors.Errorf(
		"%s: %s shape %s is not valid for tensor shape %s", opName, paramName, param, tensor)
}

func resolveFeatureBlockSize(opName, paramName string, param shapes.Shape, features, numBlocks, blockSize int) (int, error) {
	if numBlocks == 1 {
		return features, nil
	}
	if blockSize > 0 {
		expected := (features + blockSize - 1) / blockSize
		if numBlocks != expected {
			return 0, errors.Errorf(
				"%s: %s shape %s with BlockSize=%d expected last dim %d for features=%d",
				opName, paramName, param, blockSize, expected, features)
		}
		return blockSize, nil
	}
	if features%numBlocks != 0 {
		return 0, errors.Errorf(
			"%s: cannot infer BlockSize for %s shape %s (features %d not divisible by %d; set BlockSize explicitly)",
			opName, paramName, param, features, numBlocks)
	}
	return features / numBlocks, nil
}

func isActivationZPBroadcastCompatible(zp, scale ActivationParamLayout, features int) bool {
	if zp.Kind == ActParamPerTensor {
		return true
	}
	colOK := zp.ColBlockSize == features || (zp.ColBlockSize >= scale.ColBlockSize && zp.ColBlockSize%scale.ColBlockSize == 0)
	if !colOK {
		return false
	}
	switch zp.Kind {
	case ActParamPerColBlock:
		return scale.Kind == ActParamPerColBlock || scale.Kind == ActParamPerRowColBlock
	case ActParamPerRow:
		return scale.Kind == ActParamPerRow || scale.Kind == ActParamPerRowColBlock
	case ActParamPerRowColBlock:
		return scale.Kind == ActParamPerRowColBlock
	default:
		return false
	}
}

func resolveOutputQuantizedDType(oq *QuantizationShapes) (dtypes.DType, error) {
	if oq.QuantizedDType != dtypes.InvalidDType {
		if oq.QuantizedDType != dtypes.Int8 && oq.QuantizedDType != dtypes.Uint8 {
			return dtypes.InvalidDType, errors.Errorf(
				"FusedQuantizedDense: OutputQuantization.QuantizedDType must be Int8 or Uint8, got %s", oq.QuantizedDType)
		}
		return oq.QuantizedDType, nil
	}
	if oq.ZeroPoint.Ok() && (oq.ZeroPoint.DType == dtypes.Int8 || oq.ZeroPoint.DType == dtypes.Uint8) {
		return oq.ZeroPoint.DType, nil
	}
	return dtypes.Int8, nil
}

func sliceAxisNames(names []string, n int) []string {
	if names == nil {
		return nil
	}
	out := make([]string, n)
	copy(out, names[:n])
	return out
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// FusedDense returns the output shape of a FusedDense operation.
//
// Inputs:
//   - x: [batch..., in_features]
//   - weight: [in_features, out_features...] (if WeightLayout is DenseLayoutInputOutputs)
//     or [out_features..., in_features] (if WeightLayout is DenseLayoutOutputsInput)
//   - bias: [out_features...] (optional, shapes.Invalid() if unused).
//   - options: configuration including activation and weight layout.
//
// Output shape: [batch..., out_features...]
func FusedDense(x, weight, bias shapes.Shape, options compute.DenseConfig) (shapes.Shape, error) {
	if !x.Ok() {
		return shapes.Invalid(), errors.Errorf("FusedDense: x shape is invalid")
	}
	if !weight.Ok() {
		return shapes.Invalid(), errors.Errorf("FusedDense: weight shape is invalid")
	}
	if x.DType != weight.DType {
		return shapes.Invalid(), errors.Errorf("FusedDense: x and weight must have the same DType, got x=%s and weight=%s", x.DType, weight.DType)
	}
	if x.Rank() < 1 {
		return shapes.Invalid(), errors.Errorf("FusedDense: x must have rank >= 1, got %d", x.Rank())
	}
	if weight.Rank() < 2 {
		return shapes.Invalid(), errors.Errorf("FusedDense: weight must have rank >= 2, got %d", weight.Rank())
	}

	// Validate activation.
	if options.Activation.Type == compute.ActivationSwiGLU {
		return shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedDense does not support SwiGLU activation due to output shape change (use FusedActivation separately)")
	}
	if options.Activation.Type < compute.ActivationNone || options.Activation.Type > compute.ActivationSwiGLU {
		return shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedDense: unsupported activation %v", options.Activation.Type)
	}

	if weight.IsDynamic() {
		return shapes.Invalid(), errors.Errorf("FusedDense: weight must have static shape, got %s", weight)
	}
	if bias.Ok() && bias.IsDynamic() {
		return shapes.Invalid(), errors.Errorf("FusedDense: bias must have static shape, got %s", bias)
	}

	inFeaturesAxisX := x.Rank() - 1
	inFeaturesDimX := x.Dimensions[inFeaturesAxisX]
	if inFeaturesDimX == shapes.DynamicDim {
		return shapes.Invalid(), errors.Errorf("FusedDense: x's last dimension (in_features) cannot be dynamic")
	}

	var inFeaturesDimW int
	var inFeaturesAxisW int
	var outFeatureDims []int
	var outFeatureNames []string

	switch options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		inFeaturesAxisW = 0
		inFeaturesDimW = weight.Dimensions[0]
		outFeatureDims = weight.Dimensions[1:]
		if weight.AxisNames != nil {
			outFeatureNames = weight.AxisNames[1:]
		}
	case compute.DenseLayoutOutputsInput:
		inFeaturesAxisW = weight.Rank() - 1
		inFeaturesDimW = weight.Dimensions[inFeaturesAxisW]
		outFeatureDims = weight.Dimensions[:inFeaturesAxisW]
		if weight.AxisNames != nil {
			outFeatureNames = weight.AxisNames[:inFeaturesAxisW]
		}
	default:
		return shapes.Invalid(), errors.Errorf("FusedDense: unknown WeightLayout %v", options.WeightLayout)
	}

	// Check contracting dimension compatibility.
	if inFeaturesDimX != inFeaturesDimW {
		return shapes.Invalid(), errors.Errorf(
			"FusedDense: x's contracting dimension (%d) must match weight's contracting dimension (%d)",
			inFeaturesDimX, inFeaturesDimW)
	}

	// Output shape: [batch..., out_features...]
	batchRank := x.Rank() - 1
	outRank := batchRank + len(outFeatureDims)
	outDims := make([]int, 0, outRank)
	outDims = append(outDims, x.Dimensions[:batchRank]...)
	outDims = append(outDims, outFeatureDims...)

	var axisNames []string
	if x.AxisNames != nil || len(outFeatureNames) > 0 {
		axisNames = make([]string, 0, outRank)
		if x.AxisNames != nil {
			axisNames = append(axisNames, x.AxisNames[:batchRank]...)
		} else {
			for range batchRank {
				axisNames = append(axisNames, "")
			}
		}
		if len(outFeatureNames) > 0 {
			axisNames = append(axisNames, outFeatureNames...)
		} else {
			for range len(outFeatureDims) {
				axisNames = append(axisNames, "")
			}
		}
	}

	outShape := shapes.Shape{
		DType:      x.DType,
		Dimensions: outDims,
		AxisNames:  axisNames,
	}

	// Validate bias if provided.
	if bias.Ok() {
		if bias.DType != x.DType {
			return shapes.Invalid(), errors.Errorf("FusedDense: bias must have same DType as x (%s), got %s", x.DType, bias.DType)
		}
		if err := validateDenseBias(bias, outFeatureDims, outFeatureNames); err != nil {
			return shapes.Invalid(), err
		}
	}

	return outShape, nil
}

// validateDenseBias verifies that bias matches out_features.
// Bias can have rank matching len(outFeatureDims), or 1D if total out_features size matches.
func validateDenseBias(bias shapes.Shape, outFeatureDims []int, outFeatureNames []string) error {
	// Check exact match of dimensions first.
	if bias.Rank() == len(outFeatureDims) {
		match := true
		for i := range outFeatureDims {
			bDim := bias.Dimensions[i]
			oDim := outFeatureDims[i]
			if bDim == shapes.DynamicDim && oDim == shapes.DynamicDim {
				bName := bias.AxisName(i)
				var oName string
				if len(outFeatureNames) > i {
					oName = outFeatureNames[i]
				}
				if !shapes.AxisNameEqual(bName, oName) {
					return errors.Errorf("FusedDense: bias axis %d dynamic axis name (%q) does not match output feature axis name (%q)",
						i, bName, oName)
				}
			} else if bDim != shapes.DynamicDim && oDim != shapes.DynamicDim && bDim != oDim {
				match = false
				break
			}
		}
		if match {
			return nil
		}
	}

	// Also allow 1D flattened bias if total static size matches.
	if bias.Rank() == 1 && !bias.IsDynamic() {
		outStaticSize := 1
		hasDynamic := false
		for _, d := range outFeatureDims {
			if d == shapes.DynamicDim {
				hasDynamic = true
				break
			}
			outStaticSize *= d
		}
		if !hasDynamic && bias.Dimensions[0] == outStaticSize {
			return nil
		}
	}

	return errors.Errorf("FusedDense: bias shape %s does not match output features dimensions %v", bias, outFeatureDims)
}

// FusedDenseVJP returns the output shapes of the gradients (dx, dWeight, dBias) for a FusedDense operation.
//
// Inputs:
//   - x: original input tensor shape [batch..., in_features].
//   - weight: weight tensor shape.
//   - bias: bias tensor shape (can be invalid if no bias was used).
//   - y: output tensor shape of the forward FusedDense.
//   - dOutput: incoming adjoint gradient shape with respect to y.
//   - options: configuration passed to the forward FusedDense call.
//
// Returns:
//   - dx: gradient with respect to x (same shape as x).
//   - dWeight: gradient with respect to weight (same shape as weight).
//   - dBias: gradient with respect to bias (same shape as bias, or shapes.Invalid() if bias was unused).
//
// Note: FusedDenseVJP only works for activations where ActivationType.VJPRequiresInput() is false.
// If the activation requires input (e.g. GELU, SiLU), FusedDense without activation followed by
// FusedActivation should be used instead.
func FusedDenseVJP(x, weight, bias, y, dOutput shapes.Shape, options compute.DenseConfig) (dx, dWeight, dBias shapes.Shape, err error) {
	// Check VJPRequiresInput for activation.
	if options.Activation.Type.VJPRequiresInput() {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedDenseVJP does not support activation %s because it requires the pre-activation input (VJPRequiresInput() == true); "+
				"use FusedDense with ActivationNone followed by FusedActivation instead",
			options.Activation.Type)
	}

	// Check if activation is valid.
	if options.Activation.Type < compute.ActivationNone || options.Activation.Type > compute.ActivationSwiGLU {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedDenseVJP: unsupported activation %v", options.Activation.Type)
	}

	// Infer and validate forward output shape y.
	expectedY, err := FusedDense(x, weight, bias, options)
	if err != nil {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Wrapf(err, "FusedDenseVJP forward shape validation failed")
	}

	// Validate y matches expected forward output shape.
	if !y.Ok() {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Errorf("FusedDenseVJP: y shape is invalid")
	}
	if y.DType != expectedY.DType {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedDenseVJP: y DType (%s) does not match expected output DType (%s)", y.DType, expectedY.DType)
	}
	if !shapesMatch(y, expectedY) {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedDenseVJP: y shape %s does not match expected output shape %s", y, expectedY)
	}

	// Validate dOutput matches y.
	if !dOutput.Ok() {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Errorf("FusedDenseVJP: dOutput shape is invalid")
	}
	if dOutput.DType != y.DType {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedDenseVJP: dOutput DType (%s) does not match y DType (%s)", dOutput.DType, y.DType)
	}
	if !shapesMatch(dOutput, y) {
		return shapes.Invalid(), shapes.Invalid(), shapes.Invalid(), errors.Errorf(
			"FusedDenseVJP: dOutput shape %s does not match y shape %s", dOutput, y)
	}

	// dx has same shape as x.
	dx = x.Clone()

	// dWeight has same shape as weight.
	dWeight = weight.Clone()

	// dBias has same shape as bias, if bias was provided.
	if bias.Ok() {
		dBias = bias.Clone()
	} else {
		dBias = shapes.Invalid()
	}

	return dx, dWeight, dBias, nil
}

// shapesMatch returns true if two shapes have compatible ranks, dimensions, and dynamic axis names.
func shapesMatch(s1, s2 shapes.Shape) bool {
	if s1.Rank() != s2.Rank() {
		return false
	}
	for i := range s1.Dimensions {
		d1 := s1.Dimensions[i]
		d2 := s2.Dimensions[i]
		if d1 == shapes.DynamicDim && d2 == shapes.DynamicDim {
			if !shapes.AxisNameEqual(s1.AxisName(i), s2.AxisName(i)) {
				return false
			}
		} else if d1 != shapes.DynamicDim && d2 != shapes.DynamicDim && d1 != d2 {
			return false
		}
	}
	return true
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// FusedActivation returns the output shape resulting from a FusedActivation operation.
//
// For all standard activations, the output shape is identical to the input shape x.
// For ActivationSwiGLU, the rank must be >= 1, the last dimension must be even (if static),
// and the output shape has its last dimension halved: [..., lastDim/2].
func FusedActivation(x shapes.Shape, cfg compute.ActivationConfig) (shapes.Shape, error) {
	if !x.Ok() {
		return shapes.Invalid(), errors.Errorf("FusedActivation: invalid input shape %s", x)
	}
	if cfg.Type < compute.ActivationNone || cfg.Type > compute.ActivationSwiGLU {
		return shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedActivation: unsupported activation %v", cfg.Type)
	}

	if cfg.Type == compute.ActivationSwiGLU {
		rank := x.Rank()
		if rank < 1 {
			return shapes.Invalid(), errors.Errorf("FusedActivation: SwiGLU requires rank >= 1, got %d", rank)
		}
		lastDim := x.Dimensions[rank-1]
		if lastDim == shapes.DynamicDim {
			outShape := x.Clone()
			outShape.Dimensions[rank-1] = shapes.DynamicDim
			// SwiGLU halving creates a modified dynamic axis dimension.
			outShape.AxisNames[rank-1] = shapes.AnonymousAxis
			return outShape, nil
		}
		if lastDim%2 != 0 {
			return shapes.Invalid(), errors.Errorf("FusedActivation: SwiGLU requires even last dimension, got %d", lastDim)
		}
		outShape := x.Clone()
		outShape.Dimensions[rank-1] = lastDim / 2
		return outShape, nil
	}

	return x.Clone(), nil
}

// FusedActivationVJP returns the output shape (dx) of a FusedActivationVJP operation.
//
// Parameters:
//   - y: output of the activation (can be invalid/empty if not provided).
//   - x: input to the activation (can be invalid/empty if not provided and not required).
//   - dOutput: incoming adjoint gradient (must be valid).
//   - cfg: activation configuration.
//
// For SwiGLU, x is required, and the output dx has the same shape as x (double the last dimension of dOutput).
// For other activations, dx has the same shape as x (if provided) or dOutput / y.
func FusedActivationVJP(y, x, dOutput shapes.Shape, cfg compute.ActivationConfig) (shapes.Shape, error) {
	if !dOutput.Ok() {
		return shapes.Invalid(), errors.Errorf("FusedActivationVJP: dOutput cannot be invalid")
	}
	if cfg.Type < compute.ActivationNone || cfg.Type > compute.ActivationSwiGLU {
		return shapes.Invalid(), errors.Wrapf(compute.ErrNotImplemented,
			"FusedActivationVJP: unsupported activation %v", cfg.Type)
	}
	if cfg.Type.VJPRequiresInput() && !x.Ok() {
		return shapes.Invalid(), errors.Errorf("FusedActivationVJP: activation %s requires input x, got invalid shape", cfg.Type)
	}
	if !y.Ok() && !x.Ok() {
		return shapes.Invalid(), errors.Errorf("FusedActivationVJP: at least one of y or x must be valid")
	}

	if cfg.Type == compute.ActivationSwiGLU {
		if !x.Ok() {
			return shapes.Invalid(), errors.Errorf("FusedActivationVJP: SwiGLU requires x")
		}
		if x.DType != dOutput.DType {
			return shapes.Invalid(), errors.Errorf("FusedActivationVJP: x DType (%s) must match dOutput DType (%s)", x.DType, dOutput.DType)
		}
		rank := x.Rank()
		if rank < 1 {
			return shapes.Invalid(), errors.Errorf("FusedActivationVJP: SwiGLU requires rank >= 1, got %d", rank)
		}
		lastDim := x.Dimensions[rank-1]
		if lastDim != shapes.DynamicDim && lastDim%2 != 0 {
			return shapes.Invalid(), errors.Errorf("FusedActivationVJP: SwiGLU requires even last dimension, got %d", lastDim)
		}
		if dOutput.Rank() != rank {
			return shapes.Invalid(), errors.Errorf("FusedActivationVJP: SwiGLU dOutput rank %d must match x rank %d", dOutput.Rank(), rank)
		}
		for i := 0; i < rank-1; i++ {
			if !dimMatches(dOutput.Dimensions[i], x.Dimensions[i]) {
				return shapes.Invalid(), errors.Errorf("FusedActivationVJP: SwiGLU dimension mismatch at axis %d: dOutput=%d, x=%d",
					i, dOutput.Dimensions[i], x.Dimensions[i])
			}
		}
		if lastDim != shapes.DynamicDim && dOutput.Dimensions[rank-1] != shapes.DynamicDim {
			if dOutput.Dimensions[rank-1] != lastDim/2 {
				return shapes.Invalid(), errors.Errorf("FusedActivationVJP: SwiGLU dOutput last dim %d must be half of x last dim %d",
					dOutput.Dimensions[rank-1], lastDim)
			}
		}
		return x.Clone(), nil
	}

	// For non-SwiGLU activations, dx has the same shape as x (if present), or dOutput/y.
	var refShape shapes.Shape
	if x.Ok() {
		refShape = x
	} else if y.Ok() {
		refShape = y
	} else {
		refShape = dOutput
	}

	if dOutput.DType != refShape.DType {
		return shapes.Invalid(), errors.Errorf("FusedActivationVJP: dOutput DType (%s) must match operand DType (%s)",
			dOutput.DType, refShape.DType)
	}
	if !shapesMatch(dOutput, refShape) {
		return shapes.Invalid(), errors.Errorf("FusedActivationVJP: dOutput shape %s does not match operand shape %s",
			dOutput, refShape)
	}
	if y.Ok() && x.Ok() {
		if !shapesMatch(y, x) {
			return shapes.Invalid(), errors.Errorf("FusedActivationVJP: y shape %s does not match x shape %s", y, x)
		}
	}

	return refShape.Clone(), nil
}

// dimMatches checks if two dimensions match, taking DynamicDim into account.
func dimMatches(d1, d2 int) bool {
	return d1 == shapes.DynamicDim || d2 == shapes.DynamicDim || d1 == d2
}

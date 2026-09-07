// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
)

func TestFusedActivation(t *testing.T) {
	t.Run("StandardActivations", func(t *testing.T) {
		x := S(F32, 2, 3, 4)
		acts := []compute.ActivationType{
			compute.ActivationNone,
			compute.ActivationRelu,
			compute.ActivationSigmoid,
			compute.ActivationHardSigmoid,
			compute.ActivationLeakyRelu,
			compute.ActivationSelu,
			compute.ActivationSilu,
			compute.ActivationHardSwish,
			compute.ActivationTanh,
			compute.ActivationGelu,
			compute.ActivationGeluApproximate,
		}
		for _, act := range acts {
			cfg := compute.ActivationConfig{Type: act}
			out, err := FusedActivation(x, cfg)
			if err != nil {
				t.Fatalf("unexpected error for %s: %+v", act, err)
			}
			if !out.Equal(x) {
				t.Fatalf("expected %s, got %s", x, out)
			}
		}
	})

	t.Run("SwiGLU", func(t *testing.T) {
		x := S(F32, 2, 3, 8)
		cfg := compute.ActivationConfig{Type: compute.ActivationSwiGLU}
		out, err := FusedActivation(x, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		expected := S(F32, 2, 3, 4)
		if !out.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, out)
		}
	})

	t.Run("SwiGLUOddLastDim", func(t *testing.T) {
		x := S(F32, 2, 3, 7)
		cfg := compute.ActivationConfig{Type: compute.ActivationSwiGLU}
		_, err := FusedActivation(x, cfg)
		if err == nil {
			t.Fatalf("expected error for odd last dim, got nil")
		}
	})

	t.Run("SwiGLUScalar", func(t *testing.T) {
		x := S(F32)
		cfg := compute.ActivationConfig{Type: compute.ActivationSwiGLU}
		_, err := FusedActivation(x, cfg)
		if err == nil {
			t.Fatalf("expected error for scalar SwiGLU, got nil")
		}
	})
}

func TestFusedActivationVJP(t *testing.T) {
	t.Run("StandardWithoutX", func(t *testing.T) {
		y := S(F32, 2, 4)
		dOutput := S(F32, 2, 4)
		cfg := compute.ActivationConfig{Type: compute.ActivationRelu}

		out, err := FusedActivationVJP(y, shapes.Invalid(), dOutput, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !out.Equal(dOutput) {
			t.Fatalf("expected %s, got %s", dOutput, out)
		}
	})

	t.Run("RequiresInputRequiresX", func(t *testing.T) {
		y := S(F32, 2, 4)
		dOutput := S(F32, 2, 4)
		cfg := compute.ActivationConfig{Type: compute.ActivationGelu}

		_, err := FusedActivationVJP(y, shapes.Invalid(), dOutput, cfg)
		if err == nil {
			t.Fatalf("expected error when x is missing for Gelu, got nil")
		}

		x := S(F32, 2, 4)
		out, err := FusedActivationVJP(y, x, dOutput, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !out.Equal(x) {
			t.Fatalf("expected %s, got %s", x, out)
		}
	})

	t.Run("SwiGLUVJP", func(t *testing.T) {
		x := S(F32, 2, 8)
		y := S(F32, 2, 4)
		dOutput := S(F32, 2, 4)
		cfg := compute.ActivationConfig{Type: compute.ActivationSwiGLU}

		out, err := FusedActivationVJP(y, x, dOutput, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !out.Equal(x) {
			t.Fatalf("expected %s, got %s", x, out)
		}
	})

	t.Run("SwiGLUVJPMismatch", func(t *testing.T) {
		x := S(F32, 2, 8)
		dOutput := S(F32, 2, 5) // incompatible (should be 4)
		cfg := compute.ActivationConfig{Type: compute.ActivationSwiGLU}

		_, err := FusedActivationVJP(shapes.Invalid(), x, dOutput, cfg)
		if err == nil {
			t.Fatalf("expected error for SwiGLU VJP dim mismatch, got nil")
		}
	})
}

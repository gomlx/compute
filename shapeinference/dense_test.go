// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
)

func TestFusedDense(t *testing.T) {
	t.Run("BasicInputOutputs", func(t *testing.T) {
		x := S(F32, 2, 3, 10)
		w := S(F32, 10, 20)
		bias := S(F32, 20)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		out, err := FusedDense(x, w, bias, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		expected := S(F32, 2, 3, 20)
		if !out.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, out)
		}
	})

	t.Run("BasicOutputsInput", func(t *testing.T) {
		x := S(F32, 2, 3, 10)
		w := S(F32, 20, 10)
		bias := S(F32, 20)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationSigmoid},
			WeightLayout: compute.DenseLayoutOutputsInput,
		}
		out, err := FusedDense(x, w, bias, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		expected := S(F32, 2, 3, 20)
		if !out.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, out)
		}
	})

	t.Run("NoBias", func(t *testing.T) {
		x := S(F32, 5, 10)
		w := S(F32, 10, 15)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		out, err := FusedDense(x, w, shapes.Invalid(), cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		expected := S(F32, 5, 15)
		if !out.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, out)
		}
	})

	t.Run("MultiDimOutputs", func(t *testing.T) {
		x := S(F32, 4, 10)
		w := S(F32, 10, 6, 8)
		bias := S(F32, 6, 8)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationTanh},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		out, err := FusedDense(x, w, bias, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		expected := S(F32, 4, 6, 8)
		if !out.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, out)
		}

		// Also flattened 1D bias should work.
		biasFlat := S(F32, 48)
		outFlat, err := FusedDense(x, w, biasFlat, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !outFlat.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, outFlat)
		}
	})

	t.Run("DynamicBatchAndNames", func(t *testing.T) {
		x := SD(F32, []int{-1, 10}, []string{"batch", ""})
		w := S(F32, 10, 20).WithAxisNames("", "features")
		bias := S(F32, 20).WithAxisNames("features")
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		out, err := FusedDense(x, w, bias, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		expected := SD(F32, []int{-1, 20}, []string{"batch", "features"})
		if !out.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, out)
		}
	})

	t.Run("ContractingMismatch", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := S(F32, 12, 20)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		_, err := FusedDense(x, w, shapes.Invalid(), cfg)
		if err == nil {
			t.Fatalf("expected error for contracting mismatch, got nil")
		}
	})

	t.Run("SwiGLURejected", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := S(F32, 10, 20)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationSwiGLU},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		_, err := FusedDense(x, w, shapes.Invalid(), cfg)
		if err == nil {
			t.Fatalf("expected error for SwiGLU, got nil")
		}
		if !strings.Contains(err.Error(), "SwiGLU") {
			t.Fatalf("expected error mentioning SwiGLU, got: %v", err)
		}
	})

	t.Run("DynamicInFeaturesRejected", func(t *testing.T) {
		x := SD(F32, []int{2, -1}, []string{"", "features"})
		w := S(F32, 10, 20)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		_, err := FusedDense(x, w, shapes.Invalid(), cfg)
		if err == nil {
			t.Fatalf("expected error when x last dim is dynamic, got nil")
		}
	})

	t.Run("DynamicWeightRejected", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := SD(F32, []int{10, -1}, []string{"", "out"})
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		_, err := FusedDense(x, w, shapes.Invalid(), cfg)
		if err == nil {
			t.Fatalf("expected error when weight is dynamic, got nil")
		}
	})

	t.Run("DynamicBiasRejected", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := S(F32, 10, 20)
		bias := SD(F32, []int{-1}, []string{"out"})
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		_, err := FusedDense(x, w, bias, cfg)
		if err == nil {
			t.Fatalf("expected error when bias is dynamic, got nil")
		}
	})
}

func TestFusedDenseVJP(t *testing.T) {
	t.Run("ValidVJP", func(t *testing.T) {
		x := S(F32, 2, 3, 10)
		w := S(F32, 10, 20)
		bias := S(F32, 20)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		y := S(F32, 2, 3, 20)
		dOutput := S(F32, 2, 3, 20)

		dx, dWeight, dBias, err := FusedDenseVJP(x, w, bias, y, dOutput, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !dx.Equal(x) {
			t.Fatalf("expected dx=%s, got %s", x, dx)
		}
		if !dWeight.Equal(w) {
			t.Fatalf("expected dWeight=%s, got %s", w, dWeight)
		}
		if !dBias.Equal(bias) {
			t.Fatalf("expected dBias=%s, got %s", bias, dBias)
		}
	})

	t.Run("ValidVJPWithoutBias", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := S(F32, 20, 10)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationTanh},
			WeightLayout: compute.DenseLayoutOutputsInput,
		}
		y := S(F32, 2, 20)
		dOutput := S(F32, 2, 20)

		dx, dWeight, dBias, err := FusedDenseVJP(x, w, shapes.Invalid(), y, dOutput, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !dx.Equal(x) {
			t.Fatalf("expected dx=%s, got %s", x, dx)
		}
		if !dWeight.Equal(w) {
			t.Fatalf("expected dWeight=%s, got %s", w, dWeight)
		}
		if dBias.Ok() {
			t.Fatalf("expected invalid dBias when no bias provided, got %s", dBias)
		}
	})

	t.Run("RequiresInputActivationRejected", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := S(F32, 10, 20)
		bias := S(F32, 20)
		y := S(F32, 2, 20)
		dOutput := S(F32, 2, 20)

		// Test each activation where VJPRequiresInput() == true.
		rejectedActivations := []compute.ActivationType{
			compute.ActivationSilu,
			compute.ActivationHardSwish,
			compute.ActivationGelu,
			compute.ActivationGeluApproximate,
			compute.ActivationSwiGLU,
		}

		for _, act := range rejectedActivations {
			cfg := compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: act},
				WeightLayout: compute.DenseLayoutInputOutputs,
			}
			_, _, _, err := FusedDenseVJP(x, w, bias, y, dOutput, cfg)
			if err == nil {
				t.Fatalf("expected error for activation %s which requires input, got nil", act)
			}
			if !strings.Contains(err.Error(), "VJPRequiresInput") {
				t.Fatalf("expected error mentioning VJPRequiresInput, got: %v", err)
			}
		}
	})

	t.Run("AllowedActivationsAccepted", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := S(F32, 10, 20)
		bias := S(F32, 20)
		y := S(F32, 2, 20)
		dOutput := S(F32, 2, 20)

		allowedActivations := []compute.ActivationType{
			compute.ActivationNone,
			compute.ActivationRelu,
			compute.ActivationSigmoid,
			compute.ActivationHardSigmoid,
			compute.ActivationLeakyRelu,
			compute.ActivationSelu,
			compute.ActivationTanh,
		}

		for _, act := range allowedActivations {
			cfg := compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: act},
				WeightLayout: compute.DenseLayoutInputOutputs,
			}
			_, _, _, err := FusedDenseVJP(x, w, bias, y, dOutput, cfg)
			if err != nil {
				t.Fatalf("unexpected error for activation %s: %+v", act, err)
			}
		}
	})

	t.Run("MismatchedYOrDOutput", func(t *testing.T) {
		x := S(F32, 2, 10)
		w := S(F32, 10, 20)
		cfg := compute.DenseConfig{
			Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		wrongY := S(F32, 2, 15)
		dOutput := S(F32, 2, 20)

		_, _, _, err := FusedDenseVJP(x, w, shapes.Invalid(), wrongY, dOutput, cfg)
		if err == nil {
			t.Fatalf("expected error for mismatched y, got nil")
		}

		correctY := S(F32, 2, 20)
		wrongDOutput := S(F32, 2, 25)
		_, _, _, err = FusedDenseVJP(x, w, shapes.Invalid(), correctY, wrongDOutput, cfg)
		if err == nil {
			t.Fatalf("expected error for mismatched dOutput, got nil")
		}
	})
}

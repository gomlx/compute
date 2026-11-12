// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"errors"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestFusedQuantizedDense(t *testing.T) {
	t.Run("Linear_BlockAxis0_InputOutputs", func(t *testing.T) {
		x := S(F32, 2, 3, 64)
		w := S(dtypes.Int4, 64, 32)
		scale := S(F32, 2, 32) // BlockAxis=0, BlockSize=32
		zp := S(dtypes.Int4, 2, 32)
		bias := S(F32, 32)
		cfg := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:    compute.QuantLinear,
				Scale:     scale,
				ZeroPoint: zp,
				BlockAxis: 0,
				BlockSize: 32,
			},
			Activation:   compute.ActivationConfig{Type: compute.ActivationSilu},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		info, err := FusedQuantizedDenseWithInfo(x, w, bias, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		expected := S(F32, 2, 3, 32)
		if !info.OutputShape.Equal(expected) {
			t.Fatalf("expected %s, got %s", expected, info.OutputShape)
		}
		if info.WeightScaleLayout.RowBlockSize != 32 || info.WeightScaleLayout.ColBlockSize != 1 || info.WeightScaleLayout.NumColBlocks != 32 {
			t.Fatalf("unexpected scale layout: %+v", info.WeightScaleLayout)
		}
		if idx := info.WeightScaleLayout.Index(35, 7); idx != 1*32+7 {
			t.Fatalf("expected index 39, got %d", idx)
		}
	})

	t.Run("Linear_BlockAxis1_InputOutputs_And_OutputsInput", func(t *testing.T) {
		// [K, N/G] on [K, N]
		x := S(dtypes.Float16, 4, 16)
		w := S(dtypes.Int8, 16, 32)
		scale := S(dtypes.Float16, 16, 4)
		cfg := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:    compute.QuantLinear,
				Scale:     scale,
				BlockAxis: 1,
				BlockSize: 8,
			},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		out, err := FusedQuantizedDense(x, w, shapes.Invalid(), cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !out.Equal(S(dtypes.Float16, 4, 32)) {
			t.Fatalf("expected Float16[4, 32], got %s", out)
		}

		// [N, K/G] on [N, K] (OutputsInput + BlockAxis=1)
		wT := S(dtypes.Uint4, 32, 16)
		scaleT := S(dtypes.BFloat16, 32, 2)
		xBF16 := S(dtypes.BFloat16, 4, 16)
		cfgT := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:    compute.QuantNF4,
				Scale:     scaleT,
				BlockAxis: 1,
				BlockSize: 8,
			},
			WeightLayout: compute.DenseLayoutOutputsInput,
		}
		outT, err := FusedQuantizedDense(xBF16, wT, shapes.Invalid(), cfgT)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !outT.Equal(S(dtypes.BFloat16, 4, 32)) {
			t.Fatalf("expected BFloat16[4, 32], got %s", outT)
		}
	})

	t.Run("FlexibleScales_Scalar_1D_1xN", func(t *testing.T) {
		x := S(F32, 5, 16)
		w := S(dtypes.Int8, 16, 24)
		for _, sc := range []shapes.Shape{
			shapes.Make(F32),
			S(F32, 1),
			S(F32, 24),
			S(F32, 1, 24),
		} {
			cfg := QuantizedDenseConfigShapes{
				WeightQuantization: &QuantizationShapes{
					Scheme: compute.QuantLinear,
					Scale:  sc,
				},
				WeightLayout: compute.DenseLayoutInputOutputs,
			}
			out, err := FusedQuantizedDense(x, w, shapes.Invalid(), cfg)
			if err != nil {
				t.Fatalf("scale %s failed: %+v", sc, err)
			}
			if !out.Equal(S(F32, 5, 24)) {
				t.Fatalf("scale %s: expected [5, 24], got %s", sc, out)
			}
		}
	})

	t.Run("GGML_FloatX_And_PreQuantizedX", func(t *testing.T) {
		// Q4_0: 18 bytes / 32 values -> K=64 => bytesPerRow=36
		wQ4 := S(dtypes.Uint8, 20, 36)
		xFloat := S(F32, 3, 64)
		cfgFloat := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ4_0,
			},
			WeightLayout: compute.DenseLayoutOutputsInput,
		}
		outFloat, err := FusedQuantizedDense(xFloat, wQ4, S(F32, 20), cfgFloat)
		if err != nil {
			t.Fatalf("GGML float x failed: %+v", err)
		}
		if !outFloat.Equal(S(F32, 3, 20)) {
			t.Fatalf("expected F32[3, 20], got %s", outFloat)
		}

		// Q8_0 x: 34 bytes / 32 values -> K=64 => bytesPerRowX=68
		xQ8 := S(dtypes.Uint8, 3, 68)
		cfgQ8 := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ4_0,
			},
			XQuantization: &QuantizationShapes{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ8_0,
			},
			WeightLayout: compute.DenseLayoutOutputsInput,
		}
		outQ8, err := FusedQuantizedDense(xQ8, wQ4, S(F32, 20), cfgQ8)
		if err != nil {
			t.Fatalf("GGML Q8_0 x failed: %+v", err)
		}
		if !outQ8.Equal(S(F32, 3, 20)) {
			t.Fatalf("expected F32[3, 20], got %s", outQ8)
		}

		// Q4_1 (20B), Q5_0 (22B), Q5_1 (24B): 32 values/block -> K=64 => bytesPerRow=40, 44, 48 paired with Q8_0
		for _, tc := range []struct {
			t   compute.GGMLQuantType
			bpr int
		}{
			{compute.GGMLQ4_1, 40},
			{compute.GGMLQ5_0, 44},
			{compute.GGMLQ5_1, 48},
		} {
			out, err := FusedQuantizedDense(xQ8, S(dtypes.Uint8, 20, tc.bpr), S(F32, 20), QuantizedDenseConfigShapes{
				WeightQuantization: &QuantizationShapes{Scheme: compute.QuantGGML, GGMLType: tc.t},
				XQuantization:      &QuantizationShapes{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0},
				WeightLayout:       compute.DenseLayoutOutputsInput,
			})
			if err != nil || !out.Equal(S(F32, 3, 20)) {
				t.Fatalf("GGML(%s) Q8_0 x failed: out=%s err=%v", tc.t, out, err)
			}
		}

		// Q4_K weight (144 bytes / 256 values) + Q8_K x (292 bytes / 256 values) for K=256
		wQ4K := S(dtypes.Uint8, 12, 144)
		xQ8K := S(dtypes.Uint8, 2, 292)
		cfgQ8K := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ4_K,
			},
			XQuantization: &QuantizationShapes{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ8_K,
			},
			WeightLayout: compute.DenseLayoutOutputsInput,
		}
		outQ8K, err := FusedQuantizedDense(xQ8K, wQ4K, shapes.Invalid(), cfgQ8K)
		if err != nil {
			t.Fatalf("GGML Q8_K x failed: %+v", err)
		}
		if !outQ8K.Equal(S(F32, 2, 12)) {
			t.Fatalf("expected F32[2, 12], got %s", outQ8K)
		}
	})

	t.Run("QLinearMatMul_XQuant_And_OutputQuant", func(t *testing.T) {
		x := S(dtypes.Uint8, 4, 16)
		w := S(dtypes.Int8, 16, 8)
		cfg := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:    compute.QuantLinear,
				Scale:     S(F32, 8),
				ZeroPoint: S(dtypes.Int8, 8),
			},
			XQuantization: &QuantizationShapes{
				Scheme:    compute.QuantLinear,
				Scale:     S(F32, 4, 1),
				ZeroPoint: S(dtypes.Uint8, 4, 1),
			},
			OutputQuantization: &QuantizationShapes{
				Scheme:    compute.QuantLinear,
				Scale:     shapes.Make(F32),
				ZeroPoint: shapes.Make(dtypes.Uint8),
			},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		out, err := FusedQuantizedDense(x, w, S(dtypes.Int32, 8), cfg)
		if err != nil {
			t.Fatalf("QLinearMatMul shape inference failed: %+v", err)
		}
		if !out.Equal(S(dtypes.Uint8, 4, 8)) {
			t.Fatalf("expected Uint8[4, 8], got %s", out)
		}
	})

	t.Run("DynamicBatchDims", func(t *testing.T) {
		x := shapes.Shape{
			DType:      F32,
			Dimensions: []int{shapes.DynamicDim, 32},
			AxisNames:  []string{"batch", ""},
		}
		w := S(dtypes.Int4, 32, 16)
		cfg := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:    compute.QuantLinear,
				Scale:     S(F32, 1, 16),
				BlockAxis: 0,
			},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		out, err := FusedQuantizedDense(x, w, shapes.Invalid(), cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if out.Dimensions[0] != shapes.DynamicDim || out.AxisName(0) != "batch" || out.Dimensions[1] != 16 {
			t.Fatalf("unexpected dynamic output shape: %s", out)
		}
	})

	t.Run("Errors", func(t *testing.T) {
		// SwiGLU rejected with ErrNotImplemented
		_, err := FusedQuantizedDense(S(F32, 2, 16), S(dtypes.Int8, 16, 8), shapes.Invalid(), QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{Scheme: compute.QuantLinear, Scale: S(F32, 8)},
			Activation:         compute.ActivationConfig{Type: compute.ActivationSwiGLU},
		})
		if !errors.Is(err, compute.ErrNotImplemented) {
			t.Fatalf("expected ErrNotImplemented for SwiGLU, got %v", err)
		}

		// Int32 bias rejected when K is partitioned into multiple scale blocks
		_, err = FusedQuantizedDense(S(dtypes.Uint8, 2, 16), S(dtypes.Int8, 16, 8), S(dtypes.Int32, 8), QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:    compute.QuantLinear,
				Scale:     S(F32, 2, 8),
				BlockAxis: 0,
				BlockSize: 8,
			},
			XQuantization: &QuantizationShapes{
				Scheme: compute.QuantLinear,
				Scale:  shapes.Make(F32),
			},
		})
		if err == nil {
			t.Fatalf("expected error for Int32 bias with K-blocked weight scales")
		}

		// QuantGGML with InputOutputs layout rejected
		_, err = FusedQuantizedDense(S(F32, 2, 32), S(dtypes.Uint8, 8, 18), shapes.Invalid(), QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_0},
			WeightLayout:       compute.DenseLayoutInputOutputs,
		})
		if err == nil || !strings.Contains(err.Error(), "DenseLayoutOutputsInput") {
			t.Fatalf("expected DenseLayoutOutputsInput error, got %v", err)
		}

		// Mismatched GGML pairing (Q4_K weight with Q8_0 x)
		_, err = FusedQuantizedDense(S(dtypes.Uint8, 2, 272), S(dtypes.Uint8, 8, 144), shapes.Invalid(), QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_K},
			XQuantization:      &QuantizationShapes{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0},
			WeightLayout:       compute.DenseLayoutOutputsInput,
		})
		if !errors.Is(err, compute.ErrNotImplemented) {
			t.Fatalf("expected ErrNotImplemented for mismatched GGML pairing, got %v", err)
		}
	})
}

func TestFusedQuantizedDenseVJP(t *testing.T) {
	t.Run("WithXAndY", func(t *testing.T) {
		x := S(F32, 2, 4, 32)
		w := S(dtypes.Uint4, 32, 16)
		bias := S(F32, 16)
		y := S(F32, 2, 4, 16)
		dOut := S(F32, 2, 4, 16)
		cfg := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:    compute.QuantNF4,
				Scale:     S(F32, 2, 16),
				BlockAxis: 0,
				BlockSize: 16,
			},
			Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
			WeightLayout: compute.DenseLayoutInputOutputs,
		}
		dx, dBias, err := FusedQuantizedDenseVJP(x, w, bias, y, dOut, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !dx.Equal(x) {
			t.Fatalf("expected dx=%s, got %s", x, dx)
		}
		if !dBias.Equal(bias) {
			t.Fatalf("expected dBias=%s, got %s", bias, dBias)
		}
	})

	t.Run("MemorySavingNilXAndNilY", func(t *testing.T) {
		w := S(dtypes.Uint8, 12, 18) // Q4_0: K=32, N=12
		dOut := S(dtypes.BFloat16, 3, 12)
		cfg := QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{
				Scheme:   compute.QuantGGML,
				GGMLType: compute.GGMLQ4_0,
			},
			Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
			WeightLayout: compute.DenseLayoutOutputsInput,
		}
		dx, dBias, err := FusedQuantizedDenseVJP(shapes.Invalid(), w, shapes.Invalid(), shapes.Invalid(), dOut, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !dx.Equal(S(dtypes.BFloat16, 3, 32)) {
			t.Fatalf("expected BFloat16[3, 32], got %s", dx)
		}
		if dBias.Ok() {
			t.Fatalf("expected invalid dBias when bias is invalid, got %s", dBias)
		}
	})

	t.Run("Errors", func(t *testing.T) {
		w := S(dtypes.Int8, 16, 8)
		dOut := S(F32, 2, 8)
		// Nil y when Activation != ActivationNone
		_, _, err := FusedQuantizedDenseVJP(shapes.Invalid(), w, shapes.Invalid(), shapes.Invalid(), dOut, QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{Scheme: compute.QuantLinear, Scale: S(F32, 8)},
			Activation:         compute.ActivationConfig{Type: compute.ActivationRelu},
		})
		if err == nil {
			t.Fatalf("expected error when y is invalid with ActivationRelu")
		}

		// VJPRequiresInput (Gelu)
		_, _, err = FusedQuantizedDenseVJP(S(F32, 2, 16), w, shapes.Invalid(), dOut, dOut, QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{Scheme: compute.QuantLinear, Scale: S(F32, 8)},
			Activation:         compute.ActivationConfig{Type: compute.ActivationGelu},
		})
		if err == nil {
			t.Fatalf("expected error for Gelu VJPRequiresInput")
		}

		// XQuantization != nil returns ErrNotImplemented
		_, _, err = FusedQuantizedDenseVJP(S(dtypes.Int8, 2, 16), w, shapes.Invalid(), dOut, dOut, QuantizedDenseConfigShapes{
			WeightQuantization: &QuantizationShapes{Scheme: compute.QuantLinear, Scale: S(F32, 8)},
			XQuantization:      &QuantizationShapes{Scheme: compute.QuantLinear, Scale: shapes.Make(F32)},
		})
		if !errors.Is(err, compute.ErrNotImplemented) {
			t.Fatalf("expected ErrNotImplemented for XQuantization in VJP, got %v", err)
		}
	})
}

func TestQuantizedEmbeddingLookup(t *testing.T) {
	data := S(dtypes.Uint8, 1000, 36) // Q4_0: K=64
	indices := S(dtypes.Int32, 2, 5, 1)
	out, err := QuantizedEmbeddingLookup(data, indices, &QuantizationShapes{
		Scheme:   compute.QuantGGML,
		GGMLType: compute.GGMLQ4_0,
	})
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if !out.Equal(S(F32, 2, 5, 64)) {
		t.Fatalf("expected F32[2, 5, 64], got %s", out)
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/compute/support/xslices"
)

// testFusedActivation tests the forward execution of FusedActivation across all ActivationTypes,
// shapes, and dtypes based on tests from gomlx/ml/layers/activation/activation_test.go.
func testFusedActivation(t *testing.T, b compute.Backend) {
	testutil.SkipIfMissing(t, b, compute.OpTypeFusedActivation)

	// 1D test cases based on gomlx/ml/layers/activation/activation_test.go
	input1D := []float32{0, -1, 2, -3, 4, -5, 6}

	tests := []struct {
		name      string
		actType   compute.ActivationType
		want      []float32
		tolerance float64
	}{
		{
			name:      "None",
			actType:   compute.ActivationNone,
			want:      []float32{0, -1, 2, -3, 4, -5, 6},
			tolerance: xslices.Epsilon,
		},
		{
			name:      "Relu",
			actType:   compute.ActivationRelu,
			want:      []float32{0, 0, 2, 0, 4, 0, 6},
			tolerance: xslices.Epsilon,
		},
		{
			name:      "Sigmoid",
			actType:   compute.ActivationSigmoid,
			want:      []float32{0.5, 0.26894143, 0.8807971, 0.047425873, 0.98201376, 0.006692851, 0.9975274},
			tolerance: 1e-5,
		},
		{
			name:      "HardSigmoid",
			actType:   compute.ActivationHardSigmoid,
			want:      []float32{0.5, 0.3, 0.9, 0.0, 1.0, 0.0, 1.0},
			tolerance: xslices.Epsilon,
		},
		{
			name:      "LeakyRelu",
			actType:   compute.ActivationLeakyRelu,
			want:      []float32{0, -0.3, 2, -0.9, 4, -1.5, 6},
			tolerance: xslices.Epsilon,
		},
		{
			name:      "Selu",
			actType:   compute.ActivationSelu,
			want:      []float32{0.0, -1.1113307, 2.101402, -1.6705687, 4.202804, -1.7462534, 6.304206},
			tolerance: 1e-5,
		},
		{
			name:      "Silu",
			actType:   compute.ActivationSilu,
			want:      []float32{0, -0.26894143, 1.7615942, -0.14227763, 3.928055, -0.03346425, 5.9851646},
			tolerance: 1e-5,
		},
		{
			name:      "HardSwish",
			actType:   compute.ActivationHardSwish,
			want:      []float32{0, -0.33333334, 1.6666666, 0, 4, 0, 6},
			tolerance: xslices.Epsilon,
		},
		{
			name:      "Tanh",
			actType:   compute.ActivationTanh,
			want:      []float32{0, -0.76159416, 0.9640276, -0.99505475, 0.9993293, -0.9999092, 0.9999877},
			tolerance: 1e-5,
		},
		{
			name:      "GeluExact",
			actType:   compute.ActivationGelu,
			want:      []float32{0, -0.15865526, 1.9544997, -4.0496886e-03, 3.9998736, -1.3411045e-06, 6},
			tolerance: 1e-5,
		},
		{
			name:      "GeluApproximate",
			actType:   compute.ActivationGeluApproximate,
			want:      []float32{0, -0.15880796, 1.9545977, -3.6375225e-03, 3.9999294, 0, 6},
			tolerance: 0.01,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{input1D}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedActivation(params[0], compute.ActivationConfig{Type: tc.actType})
			})
			if err != nil {
				t.Fatalf("FusedActivation(%s) failed: %+v", tc.name, err)
			}
			if ok, diff := testutil.IsInDelta(tc.want, got, tc.tolerance); !ok {
				t.Errorf("FusedActivation(%s) result mismatch:\n%s", tc.name, diff)
			}
		})
	}

	t.Run("GeluDiffersFromApproximate", func(t *testing.T) {
		gotExact, err := testutil.Exec1(b, []any{input1D}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationGelu})
		})
		if err != nil {
			t.Fatalf("FusedActivation(Gelu) failed: %+v", err)
		}
		gotApprox, err := testutil.Exec1(b, []any{input1D}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationGeluApproximate})
		})
		if err != nil {
			t.Fatalf("FusedActivation(GeluApproximate) failed: %+v", err)
		}
		exactSlice := gotExact.([]float32)
		approxSlice := gotApprox.([]float32)
		differ := false
		for i := range approxSlice {
			if math.Abs(float64(approxSlice[i]-exactSlice[i])) > 1e-6 {
				differ = true
				break
			}
		}
		if !differ {
			t.Errorf("approximate and exact GELU should differ for non-zero inputs")
		}
	})

	t.Run("SwiGLU", func(t *testing.T) {
		input := [][]float32{
			{0, 1, 2, 3},
			{-1, -2, 4, 5},
		}
		got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationSwiGLU})
		})
		if err != nil {
			t.Fatalf("FusedActivation(SwiGLU) failed: %+v", err)
		}
		want := [][]float32{
			{0, 2.1931758},
			{-1.0757657, -1.1920292},
		}
		if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
			t.Errorf("SwiGLU result mismatch:\n%s", diff)
		}

		// SwiGLU must reject odd last dimension.
		builder := b.Builder("swiglu_odd_dim")
		mainFn := builder.Main()
		param, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 1, 3), nil)
		_, err = mainFn.FusedActivation(param, compute.ActivationConfig{Type: compute.ActivationSwiGLU})
		if err == nil {
			t.Errorf("FusedActivation with SwiGLU should reject odd last dimension")
		}
	})

	t.Run("Float16", func(t *testing.T) {
		f16 := float16.FromFloat32
		inputF16 := []float16.Float16{f16(0), f16(-1), f16(2), f16(-3), f16(4)}
		got, err := testutil.Exec1(b, []any{inputF16}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationRelu})
		})
		if err != nil {
			t.Fatalf("FusedActivation Float16 failed: %+v", err)
		}
		wantF16 := []float16.Float16{f16(0), f16(0), f16(2), f16(0), f16(4)}
		if ok, diff := testutil.IsInDelta(wantF16, got, 1e-3); !ok {
			t.Errorf("Float16 Relu mismatch:\n%s", diff)
		}
	})

	t.Run("BFloat16", func(t *testing.T) {
		bf16 := bfloat16.FromFloat32
		inputBF16 := []bfloat16.BFloat16{bf16(0), bf16(-1), bf16(2), bf16(-3), bf16(4)}
		got, err := testutil.Exec1(b, []any{inputBF16}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationRelu})
		})
		if err != nil {
			t.Fatalf("FusedActivation BFloat16 failed: %+v", err)
		}
		wantBF16 := []bfloat16.BFloat16{bf16(0), bf16(0), bf16(2), bf16(0), bf16(4)}
		if ok, diff := testutil.IsInDelta(wantBF16, got, 1e-3); !ok {
			t.Errorf("BFloat16 Relu mismatch:\n%s", diff)
		}
	})
}

// testFusedActivationVJP tests vector-jacobian product execution of FusedActivationVJP.
func testFusedActivationVJP(t *testing.T, b compute.Backend) {
	testutil.SkipIfMissing(t, b, compute.OpTypeFusedActivationVJP)

	t.Run("Relu", func(t *testing.T) {
		x := []float32{0, -1, 2, -3, 4, -5, 6}
		y := []float32{0, 0, 2, 0, 4, 0, 6}
		dOutput := []float32{1, 1, 1, 1, 1, 1, 1}
		got, err := testutil.Exec1(b, []any{y, x, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], params[1], params[2], compute.ActivationConfig{Type: compute.ActivationRelu})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Relu) failed: %+v", err)
		}
		want := []float32{0, 0, 1, 0, 1, 0, 1}
		if ok, diff := testutil.IsInDelta(want, got, xslices.Epsilon); !ok {
			t.Errorf("Relu VJP mismatch:\n%s", diff)
		}

		// Relu does not require x when y is provided: passing nil x should work.
		gotNilX, err := testutil.Exec1(b, []any{y, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], nil, params[1], compute.ActivationConfig{Type: compute.ActivationRelu})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Relu, x=nil) failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want, gotNilX, xslices.Epsilon); !ok {
			t.Errorf("Relu VJP (x=nil) mismatch:\n%s", diff)
		}
	})

	t.Run("Tanh", func(t *testing.T) {
		x := []float32{0, -1, 2}
		y := []float32{0, float32(math.Tanh(-1)), float32(math.Tanh(2))}
		dOutput := []float32{1, 2, 3}
		got, err := testutil.Exec1(b, []any{y, x, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], params[1], params[2], compute.ActivationConfig{Type: compute.ActivationTanh})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Tanh) failed: %+v", err)
		}
		// dx = dOutput * (1 - y^2)
		want := []float32{
			1.0 * (1.0 - y[0]*y[0]),
			2.0 * (1.0 - y[1]*y[1]),
			3.0 * (1.0 - y[2]*y[2]),
		}
		if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
			t.Errorf("Tanh VJP mismatch:\n%s", diff)
		}

		// Tanh does not require x when y is provided: passing nil x should work.
		gotNilX, err := testutil.Exec1(b, []any{y, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], nil, params[1], compute.ActivationConfig{Type: compute.ActivationTanh})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Tanh, x=nil) failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want, gotNilX, 1e-5); !ok {
			t.Errorf("Tanh VJP (x=nil) mismatch:\n%s", diff)
		}
	})

	t.Run("Sigmoid", func(t *testing.T) {
		x := []float32{0, -1, 2}
		s := func(v float64) float32 { return float32(1.0 / (1.0 + math.Exp(-v))) }
		y := []float32{s(0), s(-1), s(2)}
		dOutput := []float32{1, 2, 3}
		got, err := testutil.Exec1(b, []any{y, x, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], params[1], params[2], compute.ActivationConfig{Type: compute.ActivationSigmoid})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Sigmoid) failed: %+v", err)
		}
		// dx = dOutput * y * (1 - y)
		want := []float32{
			1.0 * y[0] * (1.0 - y[0]),
			2.0 * y[1] * (1.0 - y[1]),
			3.0 * y[2] * (1.0 - y[2]),
		}
		if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
			t.Errorf("Sigmoid VJP mismatch:\n%s", diff)
		}

		// Sigmoid does not require x when y is provided.
		gotNilX, err := testutil.Exec1(b, []any{y, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], nil, params[1], compute.ActivationConfig{Type: compute.ActivationSigmoid})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Sigmoid, x=nil) failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want, gotNilX, 1e-5); !ok {
			t.Errorf("Sigmoid VJP (x=nil) mismatch:\n%s", diff)
		}
	})

	t.Run("HardSigmoid", func(t *testing.T) {
		x := []float32{0, -1, 2, -3, 4}
		y := []float32{0.5, 0.3, 0.9, 0.0, 1.0}
		dOutput := []float32{1, 1, 1, 1, 1}
		got, err := testutil.Exec1(b, []any{y, x, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], params[1], params[2], compute.ActivationConfig{Type: compute.ActivationHardSigmoid})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(HardSigmoid) failed: %+v", err)
		}
		// dx = dOutput * 0.2 if 0 < y < 1 else 0
		want := []float32{0.2, 0.2, 0.2, 0.0, 0.0}
		if ok, diff := testutil.IsInDelta(want, got, xslices.Epsilon); !ok {
			t.Errorf("HardSigmoid VJP mismatch:\n%s", diff)
		}
	})

	t.Run("LeakyRelu", func(t *testing.T) {
		x := []float32{0, -1, 2, -3, 4}
		y := []float32{0, -0.3, 2, -0.9, 4}
		dOutput := []float32{1, 1, 1, 1, 1}
		got, err := testutil.Exec1(b, []any{y, x, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(params[0], params[1], params[2], compute.ActivationConfig{Type: compute.ActivationLeakyRelu})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(LeakyRelu) failed: %+v", err)
		}
		// dx = dOutput * 1 if y >= 0 else 0.3
		want := []float32{1.0, 0.3, 1.0, 0.3, 1.0}
		if ok, diff := testutil.IsInDelta(want, got, xslices.Epsilon); !ok {
			t.Errorf("LeakyRelu VJP mismatch:\n%s", diff)
		}
	})

	t.Run("Gelu", func(t *testing.T) {
		x := []float32{0, 1, 2}
		dOutput := []float32{1, 1, 1}
		got, err := testutil.Exec1(b, []any{x, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(nil, params[0], params[1], compute.ActivationConfig{Type: compute.ActivationGelu})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Gelu) failed: %+v", err)
		}
		// dx = Phi(x) + x * phi(x)
		want := []float32{0.5, 1.0833154, 1.0852319}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Gelu VJP mismatch:\n%s", diff)
		}

		// GELU requires x: passing nil x must return an error.
		builder := b.Builder("gelu_vjp_nil_x")
		mainFn := builder.Main()
		paramY, _ := mainFn.Parameter("y", shapes.Make(dtypes.Float32, 3), nil)
		paramDOut, _ := mainFn.Parameter("dOutput", shapes.Make(dtypes.Float32, 3), nil)
		_, err = mainFn.FusedActivationVJP(paramY, nil, paramDOut, compute.ActivationConfig{Type: compute.ActivationGelu})
		if err == nil {
			t.Errorf("FusedActivationVJP with Gelu should reject nil x")
		}
	})

	t.Run("Silu", func(t *testing.T) {
		x := []float32{0, 1, -1}
		dOutput := []float32{1, 1, 1}
		got, err := testutil.Exec1(b, []any{x, dOutput}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedActivationVJP(nil, params[0], params[1], compute.ActivationConfig{Type: compute.ActivationSilu})
		})
		if err != nil {
			t.Fatalf("FusedActivationVJP(Silu) failed: %+v", err)
		}
		// dx = dOutput * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
		want := []float32{0.5, 0.9276703, 0.0723297}
		if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
			t.Errorf("Silu VJP mismatch:\n%s", diff)
		}

		// SiLU requires x: passing nil x must return an error.
		builder := b.Builder("silu_vjp_nil_x")
		mainFn := builder.Main()
		paramY, _ := mainFn.Parameter("y", shapes.Make(dtypes.Float32, 3), nil)
		paramDOut, _ := mainFn.Parameter("dOutput", shapes.Make(dtypes.Float32, 3), nil)
		_, err = mainFn.FusedActivationVJP(paramY, nil, paramDOut, compute.ActivationConfig{Type: compute.ActivationSilu})
		if err == nil {
			t.Errorf("FusedActivationVJP with Silu should reject nil x")
		}
	})
}

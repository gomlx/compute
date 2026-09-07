// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dense_test

import (
	"errors"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/activations"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx512"
	_ "github.com/gomlx/compute/internal/gobackend/dot"
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul"
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul/avx512"
	_ "github.com/gomlx/compute/internal/gobackend/fusedops/dense"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/compute/support/xslices"
)

func newTestBackend(t *testing.T) compute.Backend {
	b, err := gobackend.New("")
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	return b
}

func TestDenseLayouts(t *testing.T) {
	b := newTestBackend(t)

	// x: [2, 3]
	x := [][]float32{{1, 2, 3}, {4, 5, 6}}

	t.Run("InputOutputs", func(t *testing.T) {
		// w: [3, 2]
		w := [][]float32{{1, 2}, {3, 4}, {5, 6}}
		bias := []float32{10, 20}
		// x @ w = [[1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6], [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6]]
		//       = [[22, 28], [49, 64]]
		// + bias = [[32, 48], [59, 84]]
		want := [][]float32{{32, 48}, {59, 84}}

		got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
				WeightLayout: compute.DenseLayoutInputOutputs,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("InputOutputs mismatch:\n%s", diff)
		}
	})

	t.Run("OutputsInput", func(t *testing.T) {
		// w: [2, 3] (transposed)
		w := [][]float32{{1, 3, 5}, {2, 4, 6}}
		bias := []float32{10, 20}
		want := [][]float32{{32, 48}, {59, 84}}

		got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
				WeightLayout: compute.DenseLayoutOutputsInput,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("OutputsInput mismatch:\n%s", diff)
		}
	})

	t.Run("NoBias", func(t *testing.T) {
		w := [][]float32{{1, 2}, {3, 4}, {5, 6}}
		want := [][]float32{{22, 28}, {49, 64}}

		got, err := testutil.Exec1(b, []any{x, w}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], nil, compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
				WeightLayout: compute.DenseLayoutInputOutputs,
			})
		})
		if err != nil {
			t.Fatalf("FusedDense failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("NoBias mismatch:\n%s", diff)
		}
	})
}

func TestDenseActivations(t *testing.T) {
	b := newTestBackend(t)

	// Test all supported activations using 1D inputs mapped through an identity weight matrix.
	input1D := []float32{0, -1, 2, -3, 4, -5, 6}
	n := len(input1D)
	x := [][]float32{input1D} // [1, 7]
	w := make([][]float32, n)
	for i := range w {
		w[i] = make([]float32, n)
		w[i][i] = 1.0
	}
	bias := make([]float32, n)

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
			got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
					Activation: compute.ActivationConfig{Type: tc.actType},
				})
			})
			if err != nil {
				t.Fatalf("FusedDense(%s) failed: %+v", tc.name, err)
			}
			want2D := [][]float32{tc.want}
			if ok, diff := testutil.IsInDelta(want2D, got, tc.tolerance); !ok {
				t.Errorf("FusedDense(%s) result mismatch:\n%s", tc.name, diff)
			}
		})
	}

	t.Run("SwiGLU_Rejected", func(t *testing.T) {
		builder := b.Builder("test_swiglu_reject")
		mainFn := builder.Main()
		xVal, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 1, 4), nil)
		wVal, _ := mainFn.Parameter("w", shapes.Make(dtypes.Float32, 4, 4), nil)
		_, err := mainFn.FusedDense(xVal, wVal, nil, compute.DenseConfig{Activation: compute.ActivationConfig{Type: compute.ActivationSwiGLU}})
		if !errors.Is(err, compute.ErrNotImplemented) {
			t.Fatalf("expected ErrNotImplemented for SwiGLU, got %+v", err)
		}
	})

	t.Run("InvalidActivation_Rejected", func(t *testing.T) {
		builder := b.Builder("test_invalid_act")
		mainFn := builder.Main()
		xVal, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 1, 4), nil)
		wVal, _ := mainFn.Parameter("w", shapes.Make(dtypes.Float32, 4, 4), nil)
		_, err := mainFn.FusedDense(xVal, wVal, nil, compute.DenseConfig{Activation: compute.ActivationConfig{Type: compute.ActivationType(999)}})
		if !errors.Is(err, compute.ErrNotImplemented) {
			t.Fatalf("expected ErrNotImplemented for invalid activation, got %+v", err)
		}
	})
}

func TestDenseFloat64(t *testing.T) {
	b := newTestBackend(t)
	x := [][]float64{{0, -1, 2, -3, 4}}
	w := make([][]float64, 5)
	for i := range w {
		w[i] = make([]float64, 5)
		w[i][i] = 1.0
	}
	bias := make([]float64, 5)
	for _, act := range []compute.ActivationType{compute.ActivationRelu, compute.ActivationSigmoid, compute.ActivationTanh, compute.ActivationGelu} {
		t.Run(act.String(), func(t *testing.T) {
			_, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
					Activation: compute.ActivationConfig{Type: act},
				})
			})
			if err != nil {
				t.Fatalf("FusedDense Float64 (%s) failed: %+v", act, err)
			}
		})
	}
}

func TestDenseBFloat16(t *testing.T) {
	b := newTestBackend(t)

	bf16 := bfloat16.FromFloat32
	x := [][]bfloat16.BFloat16{{bf16(1), bf16(-2)}}
	w := [][]bfloat16.BFloat16{{bf16(2), bf16(1)}, {bf16(-1), bf16(2)}}
	bias := []bfloat16.BFloat16{bf16(-2), bf16(1)}
	// x @ w + bias = [2, -2]

	for _, act := range []compute.ActivationType{compute.ActivationNone, compute.ActivationRelu, compute.ActivationSigmoid, compute.ActivationTanh, compute.ActivationGelu} {
		t.Run(act.String(), func(t *testing.T) {
			_, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
					Activation: compute.ActivationConfig{Type: act},
				})
			})
			if err != nil {
				t.Fatalf("FusedDense BF16 (%s) failed: %+v", act, err)
			}
		})
	}
}

func TestDenseFloat16(t *testing.T) {
	b := newTestBackend(t)

	f16 := float16.FromFloat32
	x := [][]float16.Float16{{f16(1), f16(-2)}}
	w := [][]float16.Float16{{f16(2), f16(1)}, {f16(-1), f16(2)}}
	bias := []float16.Float16{f16(-2), f16(1)}

	for _, act := range []compute.ActivationType{compute.ActivationNone, compute.ActivationRelu, compute.ActivationSigmoid, compute.ActivationTanh, compute.ActivationGelu} {
		t.Run(act.String(), func(t *testing.T) {
			_, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{
					Activation: compute.ActivationConfig{Type: act},
				})
			})
			if err != nil {
				t.Fatalf("FusedDense F16 (%s) failed: %+v", act, err)
			}
		})
	}
}

func TestDenseDynamic(t *testing.T) {
	backend := newTestBackend(t)
	builder := backend.Builder("test_dense_dynamic")
	mainFn := builder.Main()

	// x has dynamic batch dimension: [batchSize=-1, 3]
	paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
	x, err := mainFn.Parameter("x", paramShape, nil)
	if err != nil {
		t.Fatalf("Parameter failed: %+v", err)
	}

	w, err := mainFn.Constant([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	if err != nil {
		t.Fatalf("Constant w failed: %+v", err)
	}

	bias, err := mainFn.Constant([]float32{10, 20}, 2)
	if err != nil {
		t.Fatalf("Constant bias failed: %+v", err)
	}

	y, err := mainFn.FusedDense(x, w, bias, compute.DenseConfig{
		Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
		WeightLayout: compute.DenseLayoutInputOutputs,
	})
	if err != nil {
		t.Fatalf("FusedDense failed: %+v", err)
	}

	err = mainFn.Return([]compute.Value{y}, nil)
	if err != nil {
		t.Fatalf("Return failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %+v", err)
	}
	defer exec.Finalize()

	// Execute with batch=2: [[1, 2, 3], [4, 5, 6]]
	// x @ w = [[22, 28], [49, 64]] + bias = [[32, 48], [59, 84]]
	inputVal2 := []float32{1, 2, 3, 4, 5, 6}
	inputBuf2, err := backend.BufferFromFlatData(0, inputVal2, shapes.Make(dtypes.Float32, 2, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %+v", err)
	}

	outputs2, err := exec.Execute([]compute.Buffer{inputBuf2}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}
	got2 := make([]float32, 4)
	err = outputs2[0].ToFlatData(got2)
	if err != nil {
		t.Fatalf("ToFlatData failed: %+v", err)
	}
	want2 := []float32{32, 48, 59, 84}
	if ok, diff := testutil.IsEqual(want2, got2); !ok {
		t.Errorf("batch=2 mismatch:\n%s", diff)
	}

	// Execute with batch=1: [[1, 0, 1]]
	// x @ w = [6, 8] + bias = [16, 28]
	inputVal1 := []float32{1, 0, 1}
	inputBuf1, err := backend.BufferFromFlatData(0, inputVal1, shapes.Make(dtypes.Float32, 1, 3))
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %+v", err)
	}

	outputs1, err := exec.Execute([]compute.Buffer{inputBuf1}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}
	got1 := make([]float32, 2)
	err = outputs1[0].ToFlatData(got1)
	if err != nil {
		t.Fatalf("ToFlatData failed: %+v", err)
	}
	want1 := []float32{16, 28}
	if ok, diff := testutil.IsEqual(want1, got1); !ok {
		t.Errorf("batch=1 mismatch:\n%s", diff)
	}
}

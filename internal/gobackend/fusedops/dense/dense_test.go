// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dense_test

import (
	"testing"

	"github.com/gomlx/compute"
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
	"github.com/gomlx/compute/support/testutil"
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
				Activation:   compute.ActivationNone,
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
				Activation:   compute.ActivationNone,
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
				Activation:   compute.ActivationNone,
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

	// x: [1, 2], w: [2, 2], bias: [2]
	x := [][]float32{{1, -2}}
	w := [][]float32{{2, 1}, {-1, 2}}
	// x @ w = [1*2 + -2*-1, 1*1 + -2*2] = [4, -3]
	bias := []float32{-2, 1}
	// x @ w + bias = [2, -2]

	t.Run("Relu", func(t *testing.T) {
		want := [][]float32{{2, 0}}
		got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{Activation: compute.ActivationRelu})
		})
		if err != nil {
			t.Fatalf("FusedDense Relu failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("Relu mismatch:\n%s", diff)
		}
	})

	t.Run("HardSwish", func(t *testing.T) {
		// x=2: 2 * min(max(2/6 + 0.5, 0), 1) = 2 * (5/6) = 5/3 = 1.666667
		// x=-2: -2 * min(max(-2/6 + 0.5, 0), 1) = -2 * (1/6) = -1/3 = -0.333333
		want := [][]float32{{5.0 / 3.0, -1.0 / 3.0}}
		got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{Activation: compute.ActivationHardSwish})
		})
		if err != nil {
			t.Fatalf("FusedDense HardSwish failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
			t.Errorf("HardSwish mismatch:\n%s", diff)
		}
	})
}

func TestDenseBFloat16(t *testing.T) {
	b := newTestBackend(t)

	bf16 := bfloat16.FromFloat32
	x := [][]bfloat16.BFloat16{{bf16(1), bf16(2)}}
	w := [][]bfloat16.BFloat16{{bf16(3), bf16(4)}, {bf16(5), bf16(6)}}
	bias := []bfloat16.BFloat16{bf16(10), bf16(20)}
	// x @ w = [1*3 + 2*5, 1*4 + 2*6] = [13, 16]
	// + bias = [23, 36]
	want := [][]bfloat16.BFloat16{{bf16(23), bf16(36)}}

	got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{Activation: compute.ActivationNone})
	})
	if err != nil {
		t.Fatalf("FusedDense BF16 failed: %+v", err)
	}
	if ok, diff := testutil.IsEqual(want, got); !ok {
		t.Errorf("BF16 mismatch:\n%s", diff)
	}
}

func TestDenseFloat16(t *testing.T) {
	b := newTestBackend(t)

	f16 := float16.FromFloat32
	x := [][]float16.Float16{{f16(1), f16(2)}}
	w := [][]float16.Float16{{f16(3), f16(4)}, {f16(5), f16(6)}}
	bias := []float16.Float16{f16(10), f16(20)}
	want := [][]float16.Float16{{f16(23), f16(36)}}

	got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{Activation: compute.ActivationNone})
	})
	if err != nil {
		t.Fatalf("FusedDense F16 failed: %+v", err)
	}
	if ok, diff := testutil.IsEqual(want, got); !ok {
		t.Errorf("F16 mismatch:\n%s", diff)
	}
}

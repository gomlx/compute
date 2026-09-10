// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/activations"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx512"
	_ "github.com/gomlx/compute/internal/gobackend/fusedops"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestActivationDynamic(t *testing.T) {
	backend, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	t.Run("StandardReluDynamicBatch", func(t *testing.T) {
		builder := backend.Builder("test_relu_dynamic")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %+v", err)
		}

		y, err := mainFn.FusedActivation(x, compute.ActivationConfig{Type: compute.ActivationRelu})
		if err != nil {
			t.Fatalf("FusedActivation failed: %+v", err)
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

		// Execute with batch=2
		inVal2 := []float32{-1, 2, -3, 4, -5, 6}
		inBuf2, err := backend.BufferFromFlatData(0, inVal2, shapes.Make(dtypes.Float32, 2, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %+v", err)
		}

		out2, err := exec.Execute([]compute.Buffer{inBuf2}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		got2 := make([]float32, 6)
		err = out2[0].ToFlatData(got2)
		if err != nil {
			t.Fatalf("ToFlatData failed: %+v", err)
		}
		want2 := []float32{0, 2, 0, 4, 0, 6}
		if ok, diff := testutil.IsEqual(want2, got2); !ok {
			t.Errorf("batch=2 mismatch:\n%s", diff)
		}

		// Execute with batch=1
		inVal1 := []float32{10, -20, 30}
		inBuf1, err := backend.BufferFromFlatData(0, inVal1, shapes.Make(dtypes.Float32, 1, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %+v", err)
		}

		out1, err := exec.Execute([]compute.Buffer{inBuf1}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		got1 := make([]float32, 3)
		err = out1[0].ToFlatData(got1)
		if err != nil {
			t.Fatalf("ToFlatData failed: %+v", err)
		}
		want1 := []float32{10, 0, 30}
		if ok, diff := testutil.IsEqual(want1, got1); !ok {
			t.Errorf("batch=1 mismatch:\n%s", diff)
		}
	})

	t.Run("SwiGLUDynamicBatch", func(t *testing.T) {
		builder := backend.Builder("test_swiglu_dynamic")
		mainFn := builder.Main()

		// Last dim must be static even number (4), batch dim is dynamic
		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %+v", err)
		}

		y, err := mainFn.FusedActivation(x, compute.ActivationConfig{Type: compute.ActivationSwiGLU})
		if err != nil {
			t.Fatalf("FusedActivation failed: %+v", err)
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

		// Execute with batch=2: input [2, 4] -> output [2, 2]
		inVal2 := []float32{
			1, 2, 3, 4,
			0, 1, 2, 3,
		}
		inBuf2, err := backend.BufferFromFlatData(0, inVal2, shapes.Make(dtypes.Float32, 2, 4))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %+v", err)
		}

		out2, err := exec.Execute([]compute.Buffer{inBuf2}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		got2 := make([]float32, 4)
		err = out2[0].ToFlatData(got2)
		if err != nil {
			t.Fatalf("ToFlatData failed: %+v", err)
		}
		// swish(x1) * x2 where x1 = input[:2], x2 = input[2:]
		// row 0: swish(1)*3, swish(2)*4
		// swish(1) = 1/(1+exp(-1)) = 0.7310586 -> * 3 = 2.1931758
		// swish(2) = 2/(1+exp(-2)) = 1.7615942 -> * 4 = 7.0463768
		if len(got2) != 4 {
			t.Fatalf("expected 4 outputs, got %d", len(got2))
		}
		if got2[0] < 2.19 || got2[0] > 2.20 {
			t.Errorf("got2[0] = %f, expected ~2.193", got2[0])
		}
	})

	t.Run("SwiGLUDynamicLastDimError", func(t *testing.T) {
		builder := backend.Builder("test_swiglu_bad_dynamic")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{2, shapes.DynamicDim}, []string{"", "features"})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %+v", err)
		}

		_, err = mainFn.FusedActivation(x, compute.ActivationConfig{Type: compute.ActivationSwiGLU})
		if err == nil {
			t.Fatalf("expected error for SwiGLU with dynamic last dim, got nil")
		}
	})

	t.Run("ActivationVJPDynamic", func(t *testing.T) {
		builder := backend.Builder("test_vjp_dynamic")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %+v", err)
		}
		dOutput, err := mainFn.Parameter("dOutput", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %+v", err)
		}

		dx, err := mainFn.FusedActivationVJP(nil, x, dOutput, compute.ActivationConfig{Type: compute.ActivationRelu})
		if err != nil {
			t.Fatalf("FusedActivationVJP failed: %+v", err)
		}

		err = mainFn.Return([]compute.Value{dx}, nil)
		if err != nil {
			t.Fatalf("Return failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %+v", err)
		}
		defer exec.Finalize()

		// Execute with batch=2
		xVal2 := []float32{-1, 2, -3, 4, -5, 6}
		dOutVal2 := []float32{1, 1, 1, 1, 1, 1}
		xBuf2, err := backend.BufferFromFlatData(0, xVal2, shapes.Make(dtypes.Float32, 2, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %+v", err)
		}
		dOutBuf2, err := backend.BufferFromFlatData(0, dOutVal2, shapes.Make(dtypes.Float32, 2, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %+v", err)
		}

		out2, err := exec.Execute([]compute.Buffer{xBuf2, dOutBuf2}, []bool{false, false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		got2 := make([]float32, 6)
		err = out2[0].ToFlatData(got2)
		if err != nil {
			t.Fatalf("ToFlatData failed: %+v", err)
		}
		want2 := []float32{0, 1, 0, 1, 0, 1}
		if ok, diff := testutil.IsEqual(want2, got2); !ok {
			t.Errorf("VJP batch=2 mismatch:\n%s", diff)
		}
	})
}

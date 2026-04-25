// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestExec(t *testing.T, b compute.Backend) {
	t.Run("CompileAndExecute", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeNeg)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		builder := b.Builder("test")
		mainFn := builder.Main()
		x, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		negX, err := mainFn.Neg(x)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		c, err := mainFn.Constant([]int64{1, 2, 3}, 3)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}

		err = mainFn.Return([]compute.Value{negX, c}, nil)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}

		i0, err := b.BufferFromFlatData(0, []float32{3, 5, 7}, shapes.Make(dtypes.Float32, 3))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		outputs, err := exec.Execute([]compute.Buffer{i0}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if len(outputs) != 2 {
			t.Fatalf("expected 2 outputs, got %d", len(outputs))
		}

		gotNegX, err := testutil.FromBuffer(b, outputs[0])
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if ok, diff := testutil.IsEqual([]float32{-3, -5, -7}, gotNegX); !ok {
			t.Errorf("Value mismatch for negX:\n%s", diff)
		}

		gotC, err := testutil.FromBuffer(b, outputs[1])
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if ok, diff := testutil.IsEqual([]int64{1, 2, 3}, gotC); !ok {
			t.Errorf("Value mismatch for constant C:\n%s", diff)
		}
	})

	t.Run("WrongNumberOfParameters", func(t *testing.T) {
		builder := b.Builder("test_err")
		mainFn := builder.Main()
		x, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
		_ = mainFn.Return([]compute.Value{x}, nil)
		exec, _ := builder.Compile()

		i0, _ := b.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
		i1, _ := b.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
		_, err := exec.Execute([]compute.Buffer{i0, i1}, []bool{true, true}, 0)
		if err == nil {
			t.Errorf("Expected error when feeding wrong number of parameters")
		}
	})

	t.Run("IncompatibleParameters", func(t *testing.T) {
		builder := b.Builder("test_err_incompatible")
		mainFn := builder.Main()
		x, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
		_ = mainFn.Return([]compute.Value{x}, nil)
		exec, _ := builder.Compile()

		// Different size
		i0, _ := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
		_, err := exec.Execute([]compute.Buffer{i0}, []bool{true}, 0)
		if err == nil {
			t.Errorf("Expected error when feeding incompatible parameters (different size)")
		}

		// Different dtype
		i1, _ := b.BufferFromFlatData(0, []uint32{1, 2, 3}, shapes.Make(dtypes.Uint32, 3))
		_, err = exec.Execute([]compute.Buffer{i1}, []bool{true}, 0)
		if err == nil {
			t.Errorf("Expected error when feeding incompatible parameters (different dtype)")
		}
	})
}

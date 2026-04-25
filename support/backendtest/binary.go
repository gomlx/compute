// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/support/testutil"
)

func TestBinaryOps(t *testing.T, b compute.Backend) {
	t.Run("Add", func(t *testing.T) {
		buildFnAdd := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Add(params[0], params[1])
		}

		// Test with scalar (or of size 1) values.
		y0, err := testutil.Exec1(b, []any{bfloat16.FromFloat32(7), bfloat16.FromFloat32(11)}, buildFnAdd)
		if err != nil {
			t.Fatalf("Failed to execute Add: %+v", err)
		}
		if ok, diff := testutil.IsEqual(bfloat16.FromFloat32(18), y0); !ok {
			t.Errorf("y0 mismatch:\n%s", diff)
		}

		y1, err := testutil.Exec1(b, []any{[]int32{-1, 2}, []int32{1}}, buildFnAdd)
		if err != nil {
			t.Fatalf("Failed to execute Add: %+v", err)
		}
		if ok, diff := testutil.IsEqual([]int32{0, 3}, y1); !ok {
			t.Errorf("y1 value mismatch (-want +got):\n%s", diff)
		}

		y2, err := testutil.Exec1(b, []any{[][]int32{{-1}, {2}}, int32(-1)}, buildFnAdd)
		if err != nil {
			t.Fatalf("Failed to execute Add: %+v", err)
		}
		if ok, diff := testutil.IsEqual([][]int32{{-2}, {1}}, y2); !ok {
			t.Errorf("y2 value mismatch (-want +got):\n%s", diff)
		}

		// Test with same sized shapes:
		y3, err := testutil.Exec1(b, []any{[][]uint64{{1, 2}, {3, 4}}, [][]uint64{{4, 3}, {2, 1}}}, buildFnAdd)
		if err != nil {
			t.Fatalf("Failed to execute Add: %+v", err)
		}
		if ok, diff := testutil.IsEqual([][]uint64{{5, 5}, {5, 5}}, y3); !ok {
			t.Errorf("y3 value mismatch (-want +got):\n%s", diff)
		}

		// Test with broadcasting from both sides.
		y4, err := testutil.Exec1(b, []any{[][]int32{{-1}, {2}, {5}}, [][]int32{{10, 100}}}, buildFnAdd)
		if err != nil {
			t.Fatalf("Failed to execute Add: %+v", err)
		}
		if ok, diff := testutil.IsEqual([][]int32{{9, 99}, {12, 102}, {15, 105}}, y4); !ok {
			t.Errorf("y4 value mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("Mul", func(t *testing.T) {
		buildFnMul := func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Mul(params[0], params[1])
		}

		y0, err := testutil.Exec1(b, []any{[]float32{1, 2, 3}, float32(2)}, buildFnMul)
		if err != nil {
			t.Fatalf("Failed to execute Mul: %+v", err)
		}
		if ok, diff := testutil.IsEqual([]float32{2, 4, 6}, y0); !ok {
			t.Errorf("y0 value mismatch (-want +got):\n%s", diff)
		}
	})
}

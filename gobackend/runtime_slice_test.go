// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"reflect"
	"strconv"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestSliceAfterRuntimeReshape(t *testing.T) {
	builder := backend.Builder("slice_after_runtime_reshape")
	f := builder.Main()
	x, err := f.Parameter("x", shapes.MakeDynamic(dtypes.Int64, []int{shapes.DynamicDim}, []string{"items"}), nil)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := f.Parameter("rows", shapes.Make(dtypes.Int64), nil)
	if err != nil {
		t.Fatal(err)
	}
	reshaped, err := f.DynamicReshape(x,
		compute.DynamicDimensionSpec{Name: "runtime_rows", Value: rows},
		compute.DynamicDimensionSpec{Name: "runtime_columns"})
	if err != nil {
		t.Fatal(err)
	}
	column, err := f.Slice(reshaped, []int{0, 1}, []int{shapes.DynamicDim, 2}, []int{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{column}, nil); err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()

	xBuffer, err := backend.BufferFromFlatData(0, []int64{1, 2, 3, 4, 5, 6}, shapes.Make(dtypes.Int64, 6))
	if err != nil {
		t.Fatal(err)
	}
	defer xBuffer.Finalize()
	// The input shapes stay identical, so these calls reuse the same cached
	// specialization while the reshape's actual dimensions change.
	for _, tc := range []struct {
		rows int64
		want []int64
	}{
		{2, []int64{2, 5}},
		{3, []int64{2, 4, 6}},
		{2, []int64{2, 5}},
	} {
		t.Run(strconv.FormatInt(tc.rows, 10), func(t *testing.T) {
			rowsBuffer, err := backend.BufferFromFlatData(0, []int64{tc.rows}, shapes.Make(dtypes.Int64))
			if err != nil {
				t.Fatal(err)
			}
			defer rowsBuffer.Finalize()
			outputs, err := exec.Execute([]compute.Buffer{xBuffer, rowsBuffer}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer outputs[0].Finalize()
			outputShape, err := outputs[0].Shape()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(outputShape.Dimensions, []int{int(tc.rows), 1}) {
				t.Fatalf("output shape = %s, want [%d, 1]", outputShape, tc.rows)
			}
			if got := readBufferInt64(t, outputs[0]); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("output = %v, want %v", got, tc.want)
			}
		})
	}
}

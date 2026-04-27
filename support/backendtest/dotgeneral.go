// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/compute/support/xslices"
)

func TestDotGeneral(t *testing.T, b compute.Backend) {
	testutil.SkipIfMissing(t, b, compute.OpTypeDotGeneral)

	t.Run("Shape", func(t *testing.T) {
		S := shapes.Make
		F32 := dtypes.Float32
		builder := b.Builder("DotGeneral Test")
		mainFn := builder.Main()
		lhs, err := mainFn.Parameter("lhs", S(F32, 2, 3, 4, 5), nil)
		if err != nil {
			t.Fatalf("Unexpected error: %+v", err)
		}
		rhs, err := mainFn.Parameter("rhs", S(F32, 5, 1, 2, 3), nil)
		if err != nil {
			t.Fatalf("Unexpected error: %+v", err)
		}
		gotOp, err := mainFn.DotGeneral(
			lhs, []int{1}, []int{3, 0},
			rhs, []int{3}, []int{0, 2},
			compute.DotGeneralConfig{},
		)
		if err != nil {
			t.Fatalf("Unexpected error: %+v", err)
		}
		err = mainFn.Return([]compute.Value{gotOp}, nil)
		if err != nil {
			t.Fatalf("Unexpected error: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Unexpected error: %+v", err)
		}

		// Use dummy inputs to execute and get output shape.
		i0, _ := b.BufferFromFlatData(0, make([]float32, S(F32, 2, 3, 4, 5).Size()), S(F32, 2, 3, 4, 5))
		i1, _ := b.BufferFromFlatData(0, make([]float32, S(F32, 5, 1, 2, 3).Size()), S(F32, 5, 1, 2, 3))
		outputs, err := exec.Execute([]compute.Buffer{i0, i1}, nil, 0)
		if err != nil {
			t.Fatalf("Unexpected error: %+v", err)
		}
		outputShape, err := outputs[0].Shape()
		if err != nil {
			t.Fatalf("Unexpected error: %+v", err)
		}

		// Batch dims: 5 , 2
		// Contracting dims: 3
		// Cross dims: 4 (lhs) and 1 (rhs)
		if err := outputShape.Check(F32, 5, 2, 4, 1); err != nil {
			t.Errorf("Unexpected error: %+v", err)
		}
	})

	t.Run("Float32", func(t *testing.T) {
		// Larger example, with multiple axes.
		y0, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
			// We construct the input constants directly inside the compute.Function since we don't have
			// nested slice generators handy.
			lhs, _ := f.Constant(xslices.Iota(float32(1), 2*3*1*5), 2, 3, 1, 5)
			rhs, _ := f.Constant(xslices.Iota(float32(1), 5*3*2*4), 5, 3, 2, 4)
			return f.DotGeneral(lhs, []int{1}, []int{3, 0}, rhs, []int{1}, []int{0, 2}, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}
		want := [][][][]float32{
			{
				{{242, 260, 278, 296}},
				{{899, 962, 1025, 1088}},
			}, {
				{{773, 794, 815, 836}},
				{{2522, 2588, 2654, 2720}},
			}, {
				{{1448, 1472, 1496, 1520}},
				{{4289, 4358, 4427, 4496}},
			}, {
				{{2267, 2294, 2321, 2348}},
				{{6200, 6272, 6344, 6416}},
			}, {
				{{3230, 3260, 3290, 3320}},
				{{8255, 8330, 8405, 8480}},
			}}
		if ok, diff := testutil.IsEqual(want, y0); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	})

	t.Run("AxisTransposition", func(t *testing.T) {
		lhs := [][][]float32{{{1, 2, 3}}, {{4, 5, 6}}}
		rhs := [][][]float32{{{1, 1}, {1, 1}, {1, 1}}}
		y1, err := testutil.Exec1(b, []any{lhs, rhs}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, []int{2, 0}, params[1], []int{0}, []int{1, 2}, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}
		want1 := [][]float32{{1, 4}, {2, 5}, {3, 6}}
		if ok, diff := testutil.IsEqual(want1, y1); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	})

	t.Run("VeryLarge", func(t *testing.T) {
		y3, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
			lhsShape := shapes.Make(dtypes.Float64, 16, 13, 384)
			lhsFlat := make([]float64, lhsShape.Size())
			for i := range lhsFlat {
				lhsFlat[i] = (float64(i) + 1.0) * 1e-5
			}
			lhs, _ := f.Constant(lhsFlat, 16, 13, 384)
			rhsShape := shapes.Make(dtypes.Float64, 384, 1536)
			rhsFlat := make([]float64, rhsShape.Size())
			for i := range rhsFlat {
				rhsFlat[i] = 1.0
			}
			rhs, _ := f.Constant(rhsFlat, 384, 1536)
			out, _ := f.DotGeneral(
				lhs, []int{2}, nil,
				rhs, []int{0}, nil, compute.DotGeneralConfig{})
			return f.Slice(out, []int{0, 0, 0}, []int{1, 1, 1}, []int{1, 1, 1})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}
		if ok, diff := testutil.IsInDelta(y3.([][][]float64)[0][0][0], 0.7392, 1e-4); !ok {
			t.Fatalf("Result not within delta 1e-4:\n%s", diff)
		}
	})

	t.Run("BFloat16-with-f32-acc", func(t *testing.T) {
		// The default accumulator dtype for half-precision (BFloat16 and Float16) is Float32.
		bf16 := bfloat16.FromFloat32
		y2, err := testutil.Exec1(b, []any{
			[][]bfloat16.BFloat16{{bf16(1), bf16(2), bf16(3)}},
			[][]bfloat16.BFloat16{{bf16(10)}, {bf16(11)}, {bf16(12)}},
		}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{AccumulatorDType: dtypes.Float32})
		})
		if err != nil {
			t.Fatalf("failed with an error: %+v", err)
		}
		if ok, diff := testutil.IsEqual(float32(10+22+36), y2.([][]bfloat16.BFloat16)[0][0].Float32()); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	})

	t.Run("Float16", func(t *testing.T) {
		f16 := float16.FromFloat32
		y2, err := testutil.Exec1(b, []any{
			[][]float16.Float16{{f16(1), f16(2), f16(3)}},
			[][]float16.Float16{{f16(10)}, {f16(11)}, {f16(12)}},
		}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}
		if ok, diff := testutil.IsEqual(float32(10+22+36), y2.([][]float16.Float16)[0][0].Float32()); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	})

	t.Run("ConfigDTypes", func(t *testing.T) {
		// Define common input shapes and values
		lhsData := float16.FromFloat32s(1, 2, 3, 4, 5, 6)
		rhsData := float16.FromFloat32s(7, 8, 9, 10, 11, 12)
		// Flat shapes for testutil.Exec1: it doesn't currently support tensors.Tensor.
		lhsFlat := [][]float16.Float16{{lhsData[0], lhsData[1], lhsData[2]}, {lhsData[3], lhsData[4], lhsData[5]}}
		rhsFlat := [][]float16.Float16{{rhsData[0], rhsData[1]}, {rhsData[2], rhsData[3]}, {rhsData[4], rhsData[5]}}

		t.Run("AccumulatorDType", func(t *testing.T) {
			result, err := testutil.Exec1(b, []any{lhsFlat, rhsFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{AccumulatorDType: dtypes.Float32})
			})
			if err != nil {
				t.Fatalf("unexpected error: %+v", err)
			}
			want := float16.FromFloat32s(1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12)
			gotFlat := testutil.FlattenSlice(result)
			if ok, diff := testutil.IsInDelta(want, gotFlat, 1e-2); !ok {
				t.Fatalf("Result not within delta 1e-2:\n%s", diff)
			}
		})

		t.Run("OutputDType", func(t *testing.T) {
			result, err := testutil.Exec1(b, []any{lhsFlat, rhsFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{OutputDType: dtypes.Float32})
			})
			if err != nil {
				t.Fatalf("unexpected error: %+v", err)
			}
			want := []float32{1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12, 4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12}
			gotFlat := testutil.FlattenSlice(result)
			if ok, diff := testutil.IsInDelta(want, gotFlat, 1e-2); !ok {
				t.Fatalf("Result not within delta 1e-2: (-want +got):\n%s", diff)
			}
		})
	})

	t.Run("Dot", func(t *testing.T) {
		y0, err := testutil.Exec1(b, []any{[]float32{1, 2, 3}, []float32{10, 11, 12}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{0}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if ok, diff := testutil.IsEqual(float32(1*10+2*11+3*12), y0); !ok {
			t.Errorf("Unexpected result (-want +got):\n%s", diff)
		}

		y1, err := testutil.Exec1(b, []any{[][]float32{{1, 2, 3}, {2, 4, 6}}, []float32{10, 11, 12}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if ok, diff := testutil.IsEqual([]float32{1*10 + 2*11 + 3*12, 2*10 + 4*11 + 6*12}, y1); !ok {
			t.Errorf("Unexpected result (-want +got):\n%s", diff)
		}
	})
}

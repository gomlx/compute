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
)

func TestDataDependentOps(t *testing.T, b compute.Backend) {
	t.Run("NonZero", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeNonZero)
		testutil.SkipIfMissingDynamicShapes(t, b)
		cases := []struct {
			name  string
			input any
			want  any
		}{
			{"multi_dimensional", [][]float32{{0, 4, 0}, {2, 0, 3}}, [][]int64{{0, 1}, {1, 0}, {1, 2}}},
			{"all_zero", [][]float32{{0, 0}, {0, 0}}, [][]int64{}},
			{"scalar", float32(7), [][]int64{{}}},
		}
		t.Run("empty", func(t *testing.T) {
			got, err := execWithInputShape(b, []float32{}, shapes.Make(dtypes.Float32, 0), func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.NonZero(p[0], compute.NonZeroConfig{})
			})
			if err != nil {
				t.Fatalf("NonZero empty tensor failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([][]int64{}, got); !ok {
				t.Errorf("NonZero empty tensor mismatch:\n%s", diff)
			}
		})
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.NonZero(p[0], compute.NonZeroConfig{})
				})
				if err != nil {
					t.Fatalf("NonZero failed: %+v", err)
				}
				if ok, diff := testutil.IsEqual(tc.want, got); !ok {
					t.Errorf("NonZero mismatch:\n%s", diff)
				}
			})
		}

		t.Run("Float16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float16)
			input := []float16.Float16{float16.FromFloat32(0), float16.FromFloat32(2)}
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.NonZero(p[0], compute.NonZeroConfig{})
			})
			if err != nil {
				t.Fatalf("NonZero Float16 failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([][]int64{{1}}, got); !ok {
				t.Errorf("NonZero Float16 mismatch:\n%s", diff)
			}
		})
		t.Run("BFloat16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.BFloat16)
			input := []bfloat16.BFloat16{bfloat16.FromFloat32(0), bfloat16.FromFloat32(2)}
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.NonZero(p[0], compute.NonZeroConfig{})
			})
			if err != nil {
				t.Fatalf("NonZero BFloat16 failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([][]int64{{1}}, got); !ok {
				t.Errorf("NonZero BFloat16 mismatch:\n%s", diff)
			}
		})
		t.Run("Bool", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Bool)
			got, err := testutil.Exec1(b, []any{[]bool{false, true}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.NonZero(p[0], compute.NonZeroConfig{Transpose: true})
			})
			if err != nil {
				t.Fatalf("NonZero Bool failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([][]int64{{1}}, got); !ok {
				t.Errorf("NonZero Bool mismatch:\n%s", diff)
			}
		})
		t.Run("DownstreamSlice", func(t *testing.T) {
			testutil.SkipIfMissing(t, b, compute.OpTypeSlice)
			got, err := testutil.Exec1(b, []any{[][]float32{{0, 4}, {2, 0}}}, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				coords, err := f.NonZero(p[0], compute.NonZeroConfig{})
				if err != nil {
					return nil, err
				}
				return f.Slice(coords, []int{0, 0}, []int{shapes.DynamicDim, 2}, []int{1, 1})
			})
			if err != nil {
				t.Fatalf("NonZero followed by Slice failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([][]int64{{0, 1}, {1, 0}}, got); !ok {
				t.Errorf("NonZero followed by Slice mismatch:\n%s", diff)
			}
		})
	})

	t.Run("FusedTopK", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedTopK)
		for _, tc := range []struct {
			name        string
			input       any
			cfg         compute.FusedTopKConfig
			wantValues  any
			wantIndices any
		}{
			{"largest_ties_negative_axis", [][]float32{{5, 5, 1, 5}, {2, 9, 9, 1}}, compute.FusedTopKConfig{Axis: -1}, [][]float32{{5, 5}, {9, 9}}, [][]int32{{0, 1}, {1, 2}}},
			{"smallest_non_last_axis", [][]float32{{3, 1}, {2, 0}, {2, 5}}, compute.FusedTopKConfig{Axis: 0, Smallest: true}, [][]float32{{2, 0}, {2, 1}}, [][]int32{{1, 1}, {2, 0}}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				values, indices, err := testutil.Exec2(b, []any{tc.input}, func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
					return f.FusedTopK(p[0], 2, tc.cfg)
				})
				skipIfFusedNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("FusedTopK failed: %+v", err)
				}
				if ok, diff := testutil.IsEqual(tc.wantValues, values); !ok {
					t.Errorf("FusedTopK values mismatch:\n%s", diff)
				}
				if ok, diff := testutil.IsEqual(tc.wantIndices, indices); !ok {
					t.Errorf("FusedTopK indices mismatch:\n%s", diff)
				}
			})
		}

		// All sortable DTypes with duplicate extrema to verify tie-breaking toward lower original index:
		// values ordered as [mid, max_tie0, min_tie0, max_tie1, min_tie1] ->
		//   largest  k=3: indices [1, 3, 0]
		//   smallest k=3: indices [2, 4, 0]
		t.Run("DTypes", func(t *testing.T) {
			for _, tc := range []struct {
				dtype   dtypes.DType
				input   any
				wantMax any
				wantMin any
			}{
				{dtypes.Float32, []float32{3, 9, -2, 9, -2}, []float32{9, 9, 3}, []float32{-2, -2, 3}},
				{dtypes.Float64, []float64{3, 9, -2, 9, -2}, []float64{9, 9, 3}, []float64{-2, -2, 3}},
				{
					dtypes.Float16,
					[]float16.Float16{float16.FromFloat32(3), float16.FromFloat32(9), float16.FromFloat32(-2), float16.FromFloat32(9), float16.FromFloat32(-2)},
					[]float16.Float16{float16.FromFloat32(9), float16.FromFloat32(9), float16.FromFloat32(3)},
					[]float16.Float16{float16.FromFloat32(-2), float16.FromFloat32(-2), float16.FromFloat32(3)},
				},
				{
					dtypes.BFloat16,
					[]bfloat16.BFloat16{bfloat16.FromFloat32(3), bfloat16.FromFloat32(9), bfloat16.FromFloat32(-2), bfloat16.FromFloat32(9), bfloat16.FromFloat32(-2)},
					[]bfloat16.BFloat16{bfloat16.FromFloat32(9), bfloat16.FromFloat32(9), bfloat16.FromFloat32(3)},
					[]bfloat16.BFloat16{bfloat16.FromFloat32(-2), bfloat16.FromFloat32(-2), bfloat16.FromFloat32(3)},
				},
				{dtypes.Int8, []int8{3, 9, -2, 9, -2}, []int8{9, 9, 3}, []int8{-2, -2, 3}},
				{dtypes.Int16, []int16{3, 9, -2, 9, -2}, []int16{9, 9, 3}, []int16{-2, -2, 3}},
				{dtypes.Int32, []int32{3, 9, -2, 9, -2}, []int32{9, 9, 3}, []int32{-2, -2, 3}},
				{dtypes.Int64, []int64{3, 9, -2, 9, -2}, []int64{9, 9, 3}, []int64{-2, -2, 3}},
				{dtypes.Uint8, []uint8{5, 200, 1, 200, 1}, []uint8{200, 200, 5}, []uint8{1, 1, 5}},
				{dtypes.Uint16, []uint16{5, 50000, 1, 50000, 1}, []uint16{50000, 50000, 5}, []uint16{1, 1, 5}},
				{dtypes.Uint32, []uint32{5, 3000000000, 1, 3000000000, 1}, []uint32{3000000000, 3000000000, 5}, []uint32{1, 1, 5}},
				{dtypes.Uint64, []uint64{uint64(1)<<63 - 1, ^uint64(0), 0, ^uint64(0), 0}, []uint64{^uint64(0), ^uint64(0), uint64(1)<<63 - 1}, []uint64{0, 0, uint64(1)<<63 - 1}},
			} {
				t.Run(tc.dtype.String(), func(t *testing.T) {
					testutil.SkipIfMissingDType(t, b, tc.dtype)
					for _, smallest := range []bool{false, true} {
						values, indices, err := testutil.Exec2(b, []any{tc.input}, func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
							return f.FusedTopK(p[0], 3, compute.FusedTopKConfig{Axis: -1, Smallest: smallest})
						})
						skipIfFusedNotImplemented(t, b, err)
						if err != nil {
							t.Fatalf("FusedTopK(%s, smallest=%v) failed: %+v", tc.dtype, smallest, err)
						}
						wantVals := tc.wantMax
						wantIdxs := []int32{1, 3, 0}
						if smallest {
							wantVals = tc.wantMin
							wantIdxs = []int32{2, 4, 0}
						}
						if ok, diff := testutil.IsEqual(wantVals, values); !ok {
							t.Errorf("FusedTopK(%s, smallest=%v) values mismatch:\n%s", tc.dtype, smallest, diff)
						}
						if ok, diff := testutil.IsEqual(wantIdxs, indices); !ok {
							t.Errorf("FusedTopK(%s, smallest=%v) indices mismatch:\n%s", tc.dtype, smallest, diff)
						}
					}
				})
			}
		})

		t.Run("DynamicAndIndicesOnly", func(t *testing.T) {
			testutil.SkipIfMissingDynamicShapes(t, b)
			template := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim}, []string{"batch", "vocab"})
			runtimeShape := shapes.Make(dtypes.Float32, 3, 4)
			flatInput := []float32{
				1, 5, 5, 2,
				9, 0, 4, 8,
				-3, -1, -2, -1,
			}
			got, err := execWithTemplateInputShape(b, flatInput, runtimeShape, template, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				vals, idxs, err := f.FusedTopK(p[0], 2, compute.FusedTopKConfig{Axis: 1})
				if err != nil {
					return nil, err
				}
				valsShape, err := f.Shape(vals)
				if err != nil {
					return nil, err
				}
				idxsShape, err := f.Shape(idxs)
				if err != nil {
					return nil, err
				}
				if valsShape.AxisName(0) != "batch" || valsShape.Dimensions[1] != 2 {
					t.Errorf("unexpected values shape: %s", valsShape)
				}
				if idxsShape.DType != dtypes.Int32 || idxsShape.AxisName(0) != "batch" || idxsShape.Dimensions[1] != 2 {
					t.Errorf("unexpected indices shape: %s", idxsShape)
				}
				return idxs, nil
			})
			skipIfFusedNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("FusedTopK dynamic indices-only failed: %+v", err)
			}
			want := [][]int32{{1, 2}, {0, 3}, {1, 3}}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Errorf("FusedTopK dynamic indices-only mismatch:\n%s", diff)
			}
		})

		t.Run("Validation", func(t *testing.T) {
			builder := b.Builder("backendtest-fusedtopk-validation")
			fn := builder.Main()
			p, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 4), nil)
			if err != nil {
				t.Fatalf("Parameter failed: %+v", err)
			}
			if _, _, err := fn.FusedTopK(p, 0, compute.FusedTopKConfig{}); err == nil {
				t.Error("expected error for k=0")
			}
			if _, _, err := fn.FusedTopK(p, 5, compute.FusedTopKConfig{}); err == nil {
				t.Error("expected error for k > dim")
			}
			if _, _, err := fn.FusedTopK(p, 2, compute.FusedTopKConfig{Axis: 1}); err == nil {
				t.Error("expected error for out-of-range axis")
			}
			if b.Capabilities().DTypes[dtypes.Bool] {
				pBool, err := fn.Parameter("b", shapes.Make(dtypes.Bool, 4), nil)
				if err != nil {
					t.Fatalf("Bool Parameter failed: %+v", err)
				}
				if _, _, err := fn.FusedTopK(pBool, 2, compute.FusedTopKConfig{}); err == nil {
					t.Error("expected error for Bool operand")
				}
			}
		})
	})

	t.Run("ConvAnonymousDynamicAxisDownstream", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeConvGeneral)
		testutil.SkipIfMissing(t, b, compute.OpTypeNeg)
		testutil.SkipIfMissingDynamicShapes(t, b)
		template := shapes.MakeDynamic(dtypes.Float32, []int{1, shapes.DynamicDim, 1}, []string{"", shapes.UniqueAxisName("spatial"), ""})
		runtimeShape := shapes.Make(dtypes.Float32, 1, 5, 1)
		got, err := execWithTemplateInputShape(b, []float32{1, 2, 3, 4, 5}, runtimeShape, template, func(f compute.Function, p []compute.Value) (compute.Value, error) {
			kernel, err := f.Constant([]float32{1, 1}, 2, 1, 1)
			if err != nil {
				return nil, err
			}
			conv, err := f.ConvGeneral(p[0], kernel, compute.ConvolveAxesConfig{
				InputBatch: 0, InputChannels: 2, InputSpatial: []int{1},
				KernelInputChannels: 1, KernelOutputChannels: 2, KernelSpatial: []int{0},
				OutputBatch: 0, OutputChannels: 2, OutputSpatial: []int{1},
			}, []int{2}, [][2]int{{0, 0}}, []int{1}, []int{1}, 1, 1)
			if err != nil {
				return nil, err
			}
			return f.Neg(conv)
		})
		if err != nil {
			t.Fatalf("ConvGeneral followed by Neg failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual([][][]float32{{{-3}, {-7}}}, got); !ok {
			t.Errorf("ConvGeneral followed by Neg mismatch:\n%s", diff)
		}
	})
}

func execWithInputShape(b compute.Backend, flat any, inputShape shapes.Shape,
	buildFn func(compute.Function, []compute.Value) (compute.Value, error)) (any, error) {
	return execWithTemplateInputShape(b, flat, inputShape, inputShape, buildFn)
}

func execWithTemplateInputShape(b compute.Backend, flat any, inputShape, templateShape shapes.Shape,
	buildFn func(compute.Function, []compute.Value) (compute.Value, error)) (any, error) {
	input, err := b.BufferFromFlatData(0, flat, inputShape)
	if err != nil {
		return nil, err
	}
	defer input.Finalize()
	builder := b.Builder("backendtest-data-dependent")
	fn := builder.Main()
	param, err := fn.Parameter("input", templateShape, nil)
	if err != nil {
		return nil, err
	}
	output, err := buildFn(fn, []compute.Value{param})
	if err != nil {
		return nil, err
	}
	if err = fn.Return([]compute.Value{output}, nil); err != nil {
		return nil, err
	}
	executable, err := builder.Compile()
	if err != nil {
		return nil, err
	}
	defer executable.Finalize()
	outputs, err := executable.Execute([]compute.Buffer{input}, nil, 0)
	if err != nil {
		return nil, err
	}
	defer outputs[0].Finalize()
	return testutil.FromBuffer(b, outputs[0])
}

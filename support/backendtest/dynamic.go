// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// TestDynamicOps tests dynamic shape operations, anonymous/minted axis names,
// zero-sized dimensions, and related operator corner cases across backends.
func TestDynamicOps(t *testing.T, b compute.Backend) {
	t.Run("DynamicShapeAndSize", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicShape)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicDimensionSize)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		builder := b.Builder("test_dyn_shape")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}

		shapeVal, err := mainFn.DynamicShape(x)
		if err != nil {
			t.Fatalf("DynamicShape failed: %v", err)
		}
		dim0, err := mainFn.DynamicDimensionSize(x, 0)
		if err != nil {
			t.Fatalf("DynamicDimensionSize 0 failed: %v", err)
		}
		dim1, err := mainFn.DynamicDimensionSize(x, 1)
		if err != nil {
			t.Fatalf("DynamicDimensionSize 1 failed: %v", err)
		}

		if err := mainFn.Return([]compute.Value{shapeVal, dim0, dim1}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8}, shapes.Make(dtypes.Float32, 2, 4))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		if len(outputs) != 3 {
			t.Fatalf("expected 3 outputs, got %d", len(outputs))
		}
		for _, out := range outputs {
			defer out.Finalize()
		}

		gotShape, err := testutil.FromBuffer(b, outputs[0])
		if err != nil {
			t.Fatalf("FromBuffer(shapeVal) failed: %v", err)
		}
		gotDim0, err := testutil.FromBuffer(b, outputs[1])
		if err != nil {
			t.Fatalf("FromBuffer(dim0) failed: %v", err)
		}
		gotDim1, err := testutil.FromBuffer(b, outputs[2])
		if err != nil {
			t.Fatalf("FromBuffer(dim1) failed: %v", err)
		}

		dimDType := b.Capabilities().DynamicDimDType
		if dimDType == dtypes.Int32 {
			if ok, diff := testutil.IsEqual([]int32{2, 4}, gotShape); !ok {
				t.Errorf("DynamicShape mismatch:\n%s", diff)
			}
			if gotDim0 != int32(2) || gotDim1 != int32(4) {
				t.Errorf("DynamicDimensionSize got (%v, %v), want (2, 4)", gotDim0, gotDim1)
			}
		} else {
			if ok, diff := testutil.IsEqual([]int64{2, 4}, gotShape); !ok {
				t.Errorf("DynamicShape mismatch:\n%s", diff)
			}
			if gotDim0 != int64(2) || gotDim1 != int64(4) {
				t.Errorf("DynamicDimensionSize got (%v, %v), want (2, 4)", gotDim0, gotDim1)
			}
		}
	})

	t.Run("DynamicReshape", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicReshape)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		t.Run("DynamicToDynamic", func(t *testing.T) {
			testutil.SkipIfMissing(t, b, compute.OpTypeDynamicDimensionSize)
			testutil.SkipIfMissing(t, b, compute.OpTypeMul)

			builder := b.Builder("test_dyn_reshape")
			mainFn := builder.Main()

			paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim}, []string{"batch", "seq"})
			x, err := mainFn.Parameter("x", paramShape, nil)
			if err != nil {
				t.Fatalf("Parameter failed: %v", err)
			}
			batchSize, err := mainFn.DynamicDimensionSize(x, 0)
			if err != nil {
				t.Fatalf("DynamicDimensionSize 0 failed: %v", err)
			}
			seqLen, err := mainFn.DynamicDimensionSize(x, 1)
			if err != nil {
				t.Fatalf("DynamicDimensionSize 1 failed: %v", err)
			}
			numTokens, err := mainFn.Mul(batchSize, seqLen)
			if err != nil {
				t.Fatalf("Mul failed: %v", err)
			}
			reshaped, err := mainFn.DynamicReshape(x, compute.DynamicDimensionSpec{
				Name:  "num_tokens",
				Value: numTokens,
			})
			if err != nil {
				t.Fatalf("DynamicReshape failed: %v", err)
			}
			if err := mainFn.Return([]compute.Value{reshaped}, nil); err != nil {
				t.Fatalf("Return failed: %v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile failed: %v", err)
			}
			defer exec.Finalize()

			inBuf, err := b.BufferFromFlatData(0, []float32{10, 20, 30, 40, 50, 60}, shapes.Make(dtypes.Float32, 2, 3))
			if err != nil {
				t.Fatalf("BufferFromFlatData failed: %v", err)
			}
			defer inBuf.Finalize()

			outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
			if err != nil {
				t.Fatalf("Execute failed: %v", err)
			}
			defer outputs[0].Finalize()

			outShape, err := outputs[0].Shape()
			if err != nil {
				t.Fatalf("Shape failed: %v", err)
			}
			if !slices.Equal(outShape.Dimensions, []int{6}) {
				t.Fatalf("expected outShape [6], got %v", outShape.Dimensions)
			}
			got, err := testutil.FromBuffer(b, outputs[0])
			if err != nil {
				t.Fatalf("FromBuffer failed: %v", err)
			}
			if ok, diff := testutil.IsEqual([]float32{10, 20, 30, 40, 50, 60}, got); !ok {
				t.Errorf("DynamicReshape mismatch:\n%s", diff)
			}
		})

		t.Run("DynamicToStatic", func(t *testing.T) {
			builder := b.Builder("test_dyn_reshape_to_static")
			mainFn := builder.Main()

			dynIn, err := mainFn.Parameter("dyn_in", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 2}, []string{"batch", ""}), nil)
			if err != nil {
				t.Fatalf("Parameter failed: %v", err)
			}
			staticReshaped, err := mainFn.DynamicReshape(
				dynIn,
				compute.DynamicDimensionSpec{Static: 2},
				compute.DynamicDimensionSpec{Static: 2},
			)
			if err != nil {
				t.Fatalf("DynamicReshape dynamic->static failed: %v", err)
			}
			if err := mainFn.Return([]compute.Value{staticReshaped}, nil); err != nil {
				t.Fatalf("Return failed: %v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile failed: %v", err)
			}
			defer exec.Finalize()

			inBuf, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 2, 2))
			if err != nil {
				t.Fatalf("BufferFromFlatData failed: %v", err)
			}
			defer inBuf.Finalize()

			outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute failed: %v", err)
			}
			defer outs[0].Finalize()

			got, err := testutil.FromBuffer(b, outs[0])
			if err != nil {
				t.Fatalf("FromBuffer failed: %v", err)
			}
			if ok, diff := testutil.IsEqual([][]float32{{1, 2}, {3, 4}}, got); !ok {
				t.Errorf("DynamicReshape dynamic->static mismatch:\n%s", diff)
			}
		})
	})

	t.Run("DynamicBroadcastInDim", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicDimensionSize)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicBroadcastInDim)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		builder := b.Builder("test_dyn_broadcast")
		mainFn := builder.Main()

		paramShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 1}, []string{"batch", ""})
		x, err := mainFn.Parameter("x", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}
		batchSize, err := mainFn.DynamicDimensionSize(x, 0)
		if err != nil {
			t.Fatalf("DynamicDimensionSize failed: %v", err)
		}
		c, err := mainFn.Constant([]float32{10, 20, 30}, 1, 3)
		if err != nil {
			t.Fatalf("Constant failed: %v", err)
		}
		broadcasted, err := mainFn.DynamicBroadcastInDim(c, []int{0, 1},
			compute.DynamicDimensionSpec{Name: "batch", Value: batchSize},
			compute.DynamicDimensionSpec{Static: 3},
		)
		if err != nil {
			t.Fatalf("DynamicBroadcastInDim failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{broadcasted}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := b.BufferFromFlatData(0, []float32{1, 2}, shapes.Make(dtypes.Float32, 2, 1))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outputs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outputs[0].Finalize()

		got, err := testutil.FromBuffer(b, outputs[0])
		if err != nil {
			t.Fatalf("FromBuffer failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([][]float32{{10, 20, 30}, {10, 20, 30}}, got); !ok {
			t.Errorf("DynamicBroadcastInDim mismatch:\n%s", diff)
		}
	})

	t.Run("DynamicIota", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicIota)
		testutil.SkipIfMissingDType(t, b, dtypes.Int32)

		builder := b.Builder("test_dynamic_iota")
		mainFn := builder.Main()

		batchParam, err := mainFn.Parameter("batch", shapes.Make(dtypes.Int32), nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}
		iotaVal, err := mainFn.DynamicIota(dtypes.Int32, 1,
			compute.DynamicDimensionSpec{Name: "batch", Value: batchParam},
			compute.DynamicDimensionSpec{Static: 3},
		)
		if err != nil {
			t.Fatalf("DynamicIota failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{iotaVal}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := b.BufferFromFlatData(0, []int32{2}, shapes.Make(dtypes.Int32))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		got, err := testutil.FromBuffer(b, outBufs[0])
		if err != nil {
			t.Fatalf("FromBuffer failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([][]int32{{0, 1, 2}, {0, 1, 2}}, got); !ok {
			t.Errorf("DynamicIota mismatch:\n%s", diff)
		}
	})

	t.Run("DynamicPad", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicPad)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)
		testutil.SkipIfMissingDType(t, b, dtypes.Int32)

		builder := b.Builder("test_dynamic_pad")
		mainFn := builder.Main()

		xParam, err := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 2, 2), nil)
		if err != nil {
			t.Fatalf("Parameter x failed: %v", err)
		}
		padStartParam, err := mainFn.Parameter("pad_start", shapes.Make(dtypes.Int32), nil)
		if err != nil {
			t.Fatalf("Parameter pad_start failed: %v", err)
		}
		fillVal, err := mainFn.Constant([]float32{0})
		if err != nil {
			t.Fatalf("Constant failed: %v", err)
		}

		padded, err := mainFn.DynamicPad(xParam, fillVal,
			compute.DynamicPadAxis{StartValue: padStartParam, End: 1},
			compute.DynamicPadAxis{Start: 1, End: 0},
		)
		if err != nil {
			t.Fatalf("DynamicPad failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{padded}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inX, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 2, 2))
		if err != nil {
			t.Fatalf("BufferFromFlatData x failed: %v", err)
		}
		defer inX.Finalize()

		inPad, err := b.BufferFromFlatData(0, []int32{1}, shapes.Make(dtypes.Int32))
		if err != nil {
			t.Fatalf("BufferFromFlatData pad failed: %v", err)
		}
		defer inPad.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inX, inPad}, []bool{false, false}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		got, err := testutil.FromBuffer(b, outBufs[0])
		if err != nil {
			t.Fatalf("FromBuffer failed: %v", err)
		}
		want := [][]float32{
			{0, 0, 0},
			{0, 1, 2},
			{0, 3, 4},
			{0, 0, 0},
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("DynamicPad mismatch:\n%s", diff)
		}
	})

	t.Run("AnonymousDynamicConcatenate", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConcatenate)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		builder := b.Builder("test_anon_dyn_concat")
		mainFn := builder.Main()

		// Two independent generated "#" axes at index 0 must not be forced to the same runtime length.
		p0, err := mainFn.Parameter("a", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("a")}), nil)
		if err != nil {
			t.Fatalf("Parameter a failed: %v", err)
		}
		p1, err := mainFn.Parameter("b", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("b")}), nil)
		if err != nil {
			t.Fatalf("Parameter b failed: %v", err)
		}
		concat, err := mainFn.Concatenate(0, p0, p1)
		if err != nil {
			t.Fatalf("Concatenate failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{concat}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inA, err := b.BufferFromFlatData(0, []float32{1, 2}, shapes.Make(dtypes.Float32, 2))
		if err != nil {
			t.Fatalf("BufferFromFlatData a failed: %v", err)
		}
		defer inA.Finalize()
		inB, err := b.BufferFromFlatData(0, []float32{3, 4, 5}, shapes.Make(dtypes.Float32, 3))
		if err != nil {
			t.Fatalf("BufferFromFlatData b failed: %v", err)
		}
		defer inB.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inA, inB}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		got, err := testutil.FromBuffer(b, outBufs[0])
		if err != nil {
			t.Fatalf("FromBuffer failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([]float32{1, 2, 3, 4, 5}, got); !ok {
			t.Errorf("AnonymousDynamicConcatenate mismatch:\n%s", diff)
		}
	})

	t.Run("PoolAndArgMaxDynamic", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeReduceWindow)
		testutil.SkipIfMissing(t, b, compute.OpTypeArgMinMax)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)
		testutil.SkipIfMissingDType(t, b, dtypes.Int32)

		builder := b.Builder("test_pool_argmax_dyn")
		mainFn := builder.Main()

		inShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4, 1}, []string{"batch", "", ""})
		x, err := mainFn.Parameter("x", inShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}
		pooled, err := mainFn.ReduceWindow(x, compute.ReduceOpMax, []int{1, 2, 1}, []int{1, 2, 1}, nil, nil, nil)
		if err != nil {
			t.Fatalf("ReduceWindow failed: %v", err)
		}
		argMax, err := mainFn.ArgMinMax(pooled, 1, dtypes.Int32, false)
		if err != nil {
			t.Fatalf("ArgMinMax failed: %v", err)
		}
		if err := mainFn.Return([]compute.Value{argMax}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		inBuf, err := b.BufferFromFlatData(0, []float32{1, 5, 3, 2, 9, 4, 7, 8}, shapes.Make(dtypes.Float32, 2, 4, 1))
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %v", err)
		}
		defer inBuf.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %v", err)
		}
		defer outBufs[0].Finalize()

		got, err := testutil.FromBuffer(b, outBufs[0])
		if err != nil {
			t.Fatalf("FromBuffer failed: %v", err)
		}
		if ok, diff := testutil.IsEqual([][]int32{{0}, {0}}, got); !ok {
			t.Errorf("PoolAndArgMaxDynamic mismatch:\n%s", diff)
		}
	})

	t.Run("NonZeroAndUniqueAxisNames", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeNonZero)
		testutil.SkipIfMissing(t, b, compute.OpTypeDynamicIota)
		testutil.SkipIfMissing(t, b, compute.OpTypeBroadcastInDim)
		testutil.SkipIfMissing(t, b, compute.OpTypeAdd)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)
		testutil.SkipIfMissingDType(t, b, dtypes.Int64)

		builder := b.Builder("test_nonzero_unique_axis")
		mainFn := builder.Main()

		xShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"rows", ""})
		x, err := mainFn.Parameter("x", xShape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %v", err)
		}

		coords, err := mainFn.NonZero(x, compute.NonZeroConfig{})
		if err != nil {
			t.Fatalf("NonZero failed: %v", err)
		}
		coordsShape, err := mainFn.Shape(coords)
		if err != nil {
			t.Fatalf("Shape(coords) failed: %v", err)
		}
		countAxis := coordsShape.AxisName(0)
		if !strings.HasPrefix(countAxis, "#nonzeros_") {
			t.Fatalf("expected minted #nonzeros_* axis name, got %q", countAxis)
		}

		// Duplicate explicit CountAxisName must fail.
		if _, err := mainFn.NonZero(x, compute.NonZeroConfig{CountAxisName: countAxis}); err == nil {
			t.Fatalf("expected error on duplicate CountAxisName %q", countAxis)
		}
		if _, err := mainFn.NonZero(x, compute.NonZeroConfig{CountAxisName: "rows"}); err == nil {
			t.Fatalf("expected error on CountAxisName conflicting with parameter axis")
		}

		// Downstream ops referencing the minted countAxis name without explicit Value node:
		rowIds, err := mainFn.DynamicIota(dtypes.Int64, 0,
			compute.DynamicDimensionSpec{Name: countAxis},
			compute.DynamicDimensionSpec{Static: 2},
		)
		if err != nil {
			t.Fatalf("DynamicIota with minted countAxis failed: %v", err)
		}
		oneScalar, err := mainFn.Constant([]int64{1})
		if err != nil {
			t.Fatalf("Constant failed: %v", err)
		}
		onesBroadcast, err := mainFn.BroadcastInDim(oneScalar, coordsShape, nil)
		if err != nil {
			t.Fatalf("BroadcastInDim to minted countAxis failed: %v", err)
		}
		sum1, err := mainFn.Add(coords, rowIds)
		if err != nil {
			t.Fatalf("Add(coords, rowIds) failed: %v", err)
		}
		sum2, err := mainFn.Add(sum1, onesBroadcast)
		if err != nil {
			t.Fatalf("Add(sum1, onesBroadcast) failed: %v", err)
		}

		if err := mainFn.Return([]compute.Value{coords, sum2}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		// Execute with 3 nonzeros, then 0 nonzeros, then 2 nonzeros on the same compiled executable.
		for _, tc := range []struct {
			rows       int
			flat       []float32
			wantCoords []int64
			wantSum    []int64
		}{
			{
				rows:       2,
				flat:       []float32{0, 4, 0, 2, 0, 3},
				wantCoords: []int64{0, 1, 1, 0, 1, 2},
				wantSum:    []int64{1, 2, 3, 2, 4, 5}, // coords + [[0,0],[1,1],[2,2]] + 1
			},
			{
				rows:       1,
				flat:       []float32{0, 0, 0},
				wantCoords: []int64{},
				wantSum:    []int64{},
			},
			{
				rows:       1,
				flat:       []float32{5, 0, 9},
				wantCoords: []int64{0, 0, 0, 2},
				wantSum:    []int64{1, 1, 2, 4},
			},
		} {
			inBuf, err := b.BufferFromFlatData(0, tc.flat, shapes.Make(dtypes.Float32, tc.rows, 3))
			if err != nil {
				t.Fatalf("BufferFromFlatData failed: %v", err)
			}
			outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
			inBuf.Finalize()
			if err != nil {
				t.Fatalf("Execute failed: %v", err)
			}
			gotCoords := make([]int64, len(tc.wantCoords))
			if err := outs[0].ToFlatData(gotCoords); err != nil {
				t.Fatalf("ToFlatData coords failed: %v", err)
			}
			gotSum := make([]int64, len(tc.wantSum))
			if err := outs[1].ToFlatData(gotSum); err != nil {
				t.Fatalf("ToFlatData sum failed: %v", err)
			}
			outs[0].Finalize()
			outs[1].Finalize()
			if !slices.Equal(gotCoords, tc.wantCoords) {
				t.Fatalf("coords = %v, want %v", gotCoords, tc.wantCoords)
			}
			if !slices.Equal(gotSum, tc.wantSum) {
				t.Fatalf("sum = %v, want %v", gotSum, tc.wantSum)
			}
		}
	})

	t.Run("NonZeroAllDTypesAndScalarZero", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeNonZero)
		testutil.SkipIfMissingDType(t, b, dtypes.Int64)

		runNZ1D := func(t *testing.T, dtype dtypes.DType, flat any) []int64 {
			t.Helper()
			testutil.SkipIfMissingDType(t, b, dtype)
			builder := b.Builder("nz_" + dtype.String())
			fn := builder.Main()
			p, err := fn.Parameter("x", shapes.Make(dtype, 3), nil)
			if err != nil {
				t.Fatalf("Parameter(%s): %v", dtype, err)
			}
			nz, err := fn.NonZero(p, compute.NonZeroConfig{})
			if err != nil {
				t.Fatalf("NonZero(%s): %v", dtype, err)
			}
			if err := fn.Return([]compute.Value{nz}, nil); err != nil {
				t.Fatalf("Return(%s): %v", dtype, err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile(%s): %v", dtype, err)
			}
			defer ex.Finalize()
			inBuf, err := b.BufferFromFlatData(0, flat, shapes.Make(dtype, 3))
			if err != nil {
				t.Fatalf("BufferFromFlatData(%s): %v", dtype, err)
			}
			defer inBuf.Finalize()
			outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute(%s): %v", dtype, err)
			}
			defer outs[0].Finalize()
			sh, err := outs[0].Shape()
			if err != nil {
				t.Fatalf("Shape(%s): %v", dtype, err)
			}
			got := make([]int64, sh.Size())
			if err := outs[0].ToFlatData(got); err != nil {
				t.Fatalf("ToFlatData(%s): %v", dtype, err)
			}
			return got
		}

		for _, tc := range []struct {
			dtype dtypes.DType
			flat  any
		}{
			{dtypes.Bool, []bool{false, true, true}},
			{dtypes.Int8, []int8{0, -2, 3}},
			{dtypes.Int16, []int16{0, -2, 3}},
			{dtypes.Int32, []int32{0, -2, 3}},
			{dtypes.Int64, []int64{0, -2, 3}},
			{dtypes.Uint8, []uint8{0, 2, 3}},
			{dtypes.Uint16, []uint16{0, 2, 3}},
			{dtypes.Uint32, []uint32{0, 2, 3}},
			{dtypes.Uint64, []uint64{0, 2, uint64(1) << 63}},
			{dtypes.Float16, []float16.Float16{float16.FromFloat32(0), float16.FromFloat32(-1.5), float16.FromFloat32(2)}},
			{dtypes.BFloat16, []bfloat16.BFloat16{bfloat16.FromFloat32(0), bfloat16.FromFloat32(-1.5), bfloat16.FromFloat32(2)}},
			{dtypes.Float32, []float32{0, -1.5, 2}},
			{dtypes.Float64, []float64{0, 1e-300, -2.5}},
		} {
			t.Run(tc.dtype.String(), func(t *testing.T) {
				got := runNZ1D(t, tc.dtype, tc.flat)
				if !slices.Equal(got, []int64{1, 2}) {
					t.Fatalf("NonZero(%s) = %v, want [1 2]", tc.dtype, got)
				}
			})
		}

		t.Run("Scalar", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float32)
			for _, transpose := range []bool{false, true} {
				builder := b.Builder("nz_scalar")
				fn := builder.Main()
				p, err := fn.Parameter("x", shapes.Make(dtypes.Float32), nil)
				if err != nil {
					t.Fatal(err)
				}
				nz, err := fn.NonZero(p, compute.NonZeroConfig{Transpose: transpose})
				if err != nil {
					t.Fatal(err)
				}
				if err := fn.Return([]compute.Value{nz}, nil); err != nil {
					t.Fatal(err)
				}
				ex, err := builder.Compile()
				if err != nil {
					t.Fatal(err)
				}
				defer ex.Finalize()

				for _, val := range []float32{0, 7} {
					inBuf, err := b.BufferFromFlatData(0, []float32{val}, shapes.Make(dtypes.Float32))
					if err != nil {
						t.Fatal(err)
					}
					outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
					inBuf.Finalize()
					if err != nil {
						t.Fatal(err)
					}
					sh, err := outs[0].Shape()
					outs[0].Finalize()
					if err != nil {
						t.Fatal(err)
					}
					wantDims := []int{0, 0}
					if val != 0 {
						if transpose {
							wantDims = []int{0, 1}
						} else {
							wantDims = []int{1, 0}
						}
					}
					if !slices.Equal(sh.Dimensions, wantDims) {
						t.Fatalf("scalar(%v, transpose=%v) shape = %v, want %v", val, transpose, sh.Dimensions, wantDims)
					}
				}
			}
		})
	})

	t.Run("NameAnonymousDynamicAxesInConvGeneral", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeConvGeneral)
		testutil.SkipIfMissing(t, b, compute.OpTypeNeg)
		testutil.SkipIfMissing(t, b, compute.OpTypeAdd)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		builder := b.Builder("test_conv_named_anon")
		fn := builder.Main()

		inShape := shapes.MakeDynamic(dtypes.Float32, []int{1, shapes.DynamicDim, 1}, []string{"", shapes.UniqueAxisName("spatial"), ""})
		x, err := fn.Parameter("x", inShape, nil)
		if err != nil {
			t.Fatal(err)
		}
		kernel, err := fn.Constant([]float32{1, 1}, 2, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		conv, err := fn.ConvGeneral(x, kernel, compute.ConvolveAxesConfig{
			InputBatch: 0, InputChannels: 2, InputSpatial: []int{1},
			KernelInputChannels: 1, KernelOutputChannels: 2, KernelSpatial: []int{0},
			OutputBatch: 0, OutputChannels: 2, OutputSpatial: []int{1},
		}, []int{2}, [][2]int{{0, 0}}, []int{1}, []int{1}, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		convShape, err := fn.Shape(conv)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(convShape.AxisName(1), "#conv_") {
			t.Fatalf("expected #conv_* axis name on strided ConvGeneral output, got %q", convShape.AxisName(1))
		}
		// Binary op between conv and Neg(conv) requires AxisNameEqual to hold on the derived dynamic axis.
		negConv, err := fn.Neg(conv)
		if err != nil {
			t.Fatal(err)
		}
		zeroCheck, err := fn.Add(conv, negConv)
		if err != nil {
			t.Fatalf("Add(conv, Neg(conv)) failed: %v", err)
		}
		if err := fn.Return([]compute.Value{zeroCheck}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		inBuf, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5}, shapes.Make(dtypes.Float32, 1, 5, 1))
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()
		outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outs[0].Finalize()
		got, err := testutil.FromBuffer(b, outs[0])
		if err != nil {
			t.Fatal(err)
		}
		if ok, diff := testutil.IsEqual([][][]float32{{{0}, {0}}}, got); !ok {
			t.Errorf("NameAnonymousDynamicAxesInConvGeneral mismatch:\n%s", diff)
		}
	})

	t.Run("ConstantSliceMutation", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeAdd)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		builder := b.Builder("const_clone")
		fn := builder.Main()
		if _, err := fn.Constant(nil); err == nil {
			t.Fatalf("expected error for nil Constant")
		}
		scratch := []float32{1, 2, 3}
		c1, err := fn.Constant(scratch, 3)
		if err != nil {
			t.Fatal(err)
		}
		// Mutate caller slice after Constant creation; c1 must retain [1, 2, 3].
		scratch[0] = 10
		scratch[1] = 20
		scratch[2] = 30
		c2, err := fn.Constant(scratch, 3)
		if err != nil {
			t.Fatal(err)
		}
		scratch[0] = -999

		total, err := fn.Add(c1, c2)
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{total}, nil); err != nil {
			t.Fatal(err)
		}
		ex, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer ex.Finalize()
		outs, err := ex.Execute(nil, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outs[0].Finalize()
		got, err := testutil.FromBuffer(b, outs[0])
		if err != nil {
			t.Fatal(err)
		}
		if ok, diff := testutil.IsEqual([]float32{11, 22, 33}, got); !ok {
			t.Errorf("ConstantSliceMutation mismatch:\n%s", diff)
		}
	})

	t.Run("IsNaNAllFloatDTypes", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeIsNaN)
		testutil.SkipIfMissingDType(t, b, dtypes.Bool)

		nan32 := float32(math.NaN())
		posInf32 := float32(math.Inf(1))
		negInf32 := float32(math.Inf(-1))
		want := []bool{false, true, false, false, false}

		t.Run("Float32", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float32)
			got, err := testutil.Exec1(b, []any{[]float32{0, nan32, posInf32, negInf32, -2.5}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.IsNaN(params[0])
			})
			if err != nil {
				t.Fatalf("IsNaN Float32: %v", err)
			}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Fatalf("IsNaN Float32 mismatch:\n%s", diff)
			}
		})

		t.Run("Float64", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float64)
			got, err := testutil.Exec1(b, []any{[]float64{math.MaxFloat64, math.NaN(), math.Inf(1), math.Inf(-1), -0.0}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.IsNaN(params[0])
			})
			if err != nil {
				t.Fatalf("IsNaN Float64: %v", err)
			}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Fatalf("IsNaN Float64 mismatch:\n%s", diff)
			}
		})

		t.Run("Float16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float16)
			got, err := testutil.Exec1(b, []any{[]float16.Float16{
				float16.FromFloat32(0),
				float16.FromFloat32(nan32),
				float16.FromFloat32(posInf32),
				float16.FromFloat32(negInf32),
				float16.FromFloat32(1.5),
			}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.IsNaN(params[0])
			})
			if err != nil {
				t.Fatalf("IsNaN Float16: %v", err)
			}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Fatalf("IsNaN Float16 mismatch:\n%s", diff)
			}
		})

		t.Run("BFloat16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.BFloat16)
			got, err := testutil.Exec1(b, []any{[]bfloat16.BFloat16{
				bfloat16.FromFloat32(0),
				bfloat16.FromFloat32(nan32),
				bfloat16.FromFloat32(posInf32),
				bfloat16.FromFloat32(negInf32),
				bfloat16.FromFloat32(1.5),
			}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.IsNaN(params[0])
			})
			if err != nil {
				t.Fatalf("IsNaN BFloat16: %v", err)
			}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Fatalf("IsNaN BFloat16 mismatch:\n%s", diff)
			}
		})

		t.Run("AnonymousDynamic", func(t *testing.T) {
			testutil.SkipIfMissingDynamicShapes(t, b)
			testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
			testutil.SkipIfMissingDType(t, b, dtypes.Float32)

			builder := b.Builder("isnan_dyn_anon")
			fn := builder.Main()
			p, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("x")}), nil)
			if err != nil {
				t.Fatal(err)
			}
			out, err := fn.IsNaN(p)
			if err != nil {
				t.Fatal(err)
			}
			if err := fn.Return([]compute.Value{out}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Finalize()
			inBuf, err := b.BufferFromFlatData(0, []float32{nan32, 3.0}, shapes.Make(dtypes.Float32, 2))
			if err != nil {
				t.Fatal(err)
			}
			defer inBuf.Finalize()
			outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer outs[0].Finalize()
			gotDyn, err := testutil.FromBuffer(b, outs[0])
			if err != nil {
				t.Fatal(err)
			}
			if ok, diff := testutil.IsEqual([]bool{true, false}, gotDyn); !ok {
				t.Fatalf("IsNaN AnonymousDynamic mismatch:\n%s", diff)
			}
		})
	})

	t.Run("WhereBoolAndAnonymousDynamic", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeWhere)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissingDType(t, b, dtypes.Bool)

		t.Run("MultidirectionalBroadcast", func(t *testing.T) {
			got, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
				cond, err := f.Constant([]bool{true, false}, 2, 1)
				if err != nil {
					return nil, err
				}
				onT, err := f.Constant([]bool{true, false, true}, 1, 3)
				if err != nil {
					return nil, err
				}
				onF, err := f.Constant([]bool{false})
				if err != nil {
					return nil, err
				}
				return f.Where(cond, onT, onF)
			})
			if err != nil {
				t.Fatalf("Where Bool broadcast: %v", err)
			}
			want := [][]bool{{true, false, true}, {false, false, false}}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Fatalf("Where Bool broadcast mismatch:\n%s", diff)
			}
		})

		t.Run("AnonymousDynamic", func(t *testing.T) {
			testutil.SkipIfMissingDynamicShapes(t, b)
			testutil.SkipIfMissing(t, b, compute.OpTypeParameter)

			// Anonymous dynamic axis "?" on condition with Bool branches (triggers failure if decomposed via BinaryOp on two "?" tensors).
			builder := b.Builder("where_bool_anon")
			fn := builder.Main()
			condP, err := fn.Parameter("cond", shapes.MakeDynamic(dtypes.Bool, []int{shapes.DynamicDim, 2}, []string{shapes.UniqueAxisName("cond"), ""}), nil)
			if err != nil {
				t.Fatal(err)
			}
			tConst, err := fn.Constant([]bool{false, true}, 1, 2)
			if err != nil {
				t.Fatal(err)
			}
			fConst, err := fn.Constant([]bool{true, false}, 1, 2)
			if err != nil {
				t.Fatal(err)
			}
			wOut, err := fn.Where(condP, tConst, fConst)
			if err != nil {
				t.Fatalf("Where Bool with anonymous axis failed: %v", err)
			}
			if err := fn.Return([]compute.Value{wOut}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Finalize()
			inBuf, err := b.BufferFromFlatData(0, []bool{true, false, false, true}, shapes.Make(dtypes.Bool, 2, 2))
			if err != nil {
				t.Fatal(err)
			}
			defer inBuf.Finalize()
			outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer outs[0].Finalize()
			gotAnon, err := testutil.FromBuffer(b, outs[0])
			if err != nil {
				t.Fatal(err)
			}
			if ok, diff := testutil.IsEqual([][]bool{{false, false}, {true, true}}, gotAnon); !ok {
				t.Fatalf("Where Bool AnonymousDynamic mismatch:\n%s", diff)
			}
		})
	})

	t.Run("ConvGeneralInputDilations", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeConvGeneral)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		axes1D := compute.ConvolveAxesConfig{
			InputBatch:           0,
			InputChannels:        1,
			InputSpatial:         []int{2},
			KernelInputChannels:  1,
			KernelOutputChannels: 0,
			KernelSpatial:        []int{2},
			OutputBatch:          0,
			OutputChannels:       1,
			OutputSpatial:        []int{2},
		}

		t.Run("StaticInputDilations", func(t *testing.T) {
			// 1D dilated input [1, 1, 3] = [1, 2, 3], inputDilation = [2] -> dilated [1, 0, 2, 0, 3]
			// With padding [1, 1] -> [0, 1, 0, 2, 0, 3, 0], kernel [1, 2, 3] -> [2, 7, 4, 11, 6].
			gotStatic, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
				inp, err := f.Constant([]float32{1, 2, 3}, 1, 1, 3)
				if err != nil {
					return nil, err
				}
				ker, err := f.Constant([]float32{1, 2, 3}, 1, 1, 3)
				if err != nil {
					return nil, err
				}
				return f.ConvGeneral(inp, ker, axes1D, []int{1}, [][2]int{{1, 1}}, []int{2}, nil, 1, 1)
			})
			if compute.IsNotImplemented(err) {
				t.Skipf("Backend %q does not support ConvGeneral inputDilations: %v", b.Name(), err)
			}
			if err != nil {
				t.Fatalf("ConvGeneral static inputDilations: %v", err)
			}
			if ok, diff := testutil.IsInDelta([][][]float32{{{2, 7, 4, 11, 6}}}, gotStatic, 1e-5); !ok {
				t.Fatalf("ConvGeneral static inputDilations mismatch:\n%s", diff)
			}
		})

		t.Run("NegativePaddingAndInputDilations", func(t *testing.T) {
			// Negative padding (cropping) with inputDilations = [2]: dilated [1, 0, 2, 0, 3] cropped [-1, -1] -> [0, 2, 0], kernel [1, 2, 3] -> [4].
			gotNegPad, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
				inp, err := f.Constant([]float32{1, 2, 3}, 1, 1, 3)
				if err != nil {
					return nil, err
				}
				ker, err := f.Constant([]float32{1, 2, 3}, 1, 1, 3)
				if err != nil {
					return nil, err
				}
				return f.ConvGeneral(inp, ker, axes1D, []int{1}, [][2]int{{-1, -1}}, []int{2}, nil, 1, 1)
			})
			if compute.IsNotImplemented(err) {
				t.Skipf("Backend %q does not support ConvGeneral negative paddings / inputDilations: %v", b.Name(), err)
			}
			if err != nil {
				t.Fatalf("ConvGeneral negative padding + inputDilations: %v", err)
			}
			if ok, diff := testutil.IsInDelta([][][]float32{{{4}}}, gotNegPad, 1e-5); !ok {
				t.Fatalf("ConvGeneral negative padding mismatch:\n%s", diff)
			}
		})

		t.Run("DynamicInputDilations", func(t *testing.T) {
			testutil.SkipIfMissingDynamicShapes(t, b)
			testutil.SkipIfMissing(t, b, compute.OpTypeParameter)

			builder := b.Builder("conv_dyn_input_dilation")
			fn := builder.Main()
			inpP, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 1, shapes.DynamicDim}, []string{"batch", "", "len"}), nil)
			if err != nil {
				t.Fatal(err)
			}
			kerC, err := fn.Constant([]float32{1, 2, 3}, 1, 1, 3)
			if err != nil {
				t.Fatal(err)
			}
			convOut, err := fn.ConvGeneral(inpP, kerC, axes1D, []int{1}, [][2]int{{1, 1}}, []int{2}, nil, 1, 1)
			if compute.IsNotImplemented(err) {
				t.Skipf("Backend %q does not support dynamic ConvGeneral inputDilations: %v", b.Name(), err)
			}
			if err != nil {
				t.Fatalf("ConvGeneral dynamic inputDilations build failed: %v", err)
			}
			convShape, err := fn.Shape(convOut)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(convShape.AxisName(2), "#conv_") {
				t.Fatalf("expected #conv_* axis name on dilated spatial output, got %q (shape %s)", convShape.AxisName(2), convShape)
			}
			if err := fn.Return([]compute.Value{convOut}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Finalize()
			inBuf, err := b.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 1, 1, 3))
			if err != nil {
				t.Fatal(err)
			}
			defer inBuf.Finalize()
			outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer outs[0].Finalize()
			gotDyn, err := testutil.FromBuffer(b, outs[0])
			if err != nil {
				t.Fatal(err)
			}
			if ok, diff := testutil.IsInDelta([][][]float32{{{2, 7, 4, 11, 6}}}, gotDyn, 1e-5); !ok {
				t.Fatalf("ConvGeneral dynamic inputDilations mismatch:\n%s", diff)
			}
		})
	})

	t.Run("GatherWindowsAndDynamic", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeGather)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)

		t.Run("Axis1WindowAndClamping", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float32)
			testutil.SkipIfMissingDType(t, b, dtypes.Int32)

			// Out-of-order mapped axis (axis 1 of [3, 4]) with sliceSizes = [3, 2], including clamped indices (-2 -> 0, 9 -> 2).
			got1, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
				op, err := f.Constant([]float32{
					0, 1, 2, 3,
					10, 11, 12, 13,
					20, 21, 22, 23,
				}, 3, 4)
				if err != nil {
					return nil, err
				}
				idx, err := f.Constant([]int32{-2, 1, 9}, 3, 1)
				if err != nil {
					return nil, err
				}
				return f.Gather(op, idx, 1, []int{1, 2}, nil, []int{1}, []int{3, 2}, false)
			})
			if err != nil {
				t.Fatalf("Gather window on axis 1 failed: %v", err)
			}
			want1 := [][][]float32{
				{{0, 1}, {10, 11}, {20, 21}},
				{{1, 2}, {11, 12}, {21, 22}},
				{{2, 3}, {12, 13}, {22, 23}},
			}
			if ok, diff := testutil.IsEqual(want1, got1); !ok {
				t.Fatalf("Gather window on axis 1 mismatch:\n%s", diff)
			}
		})

		t.Run("TwoMappedWindowAxes", func(t *testing.T) {
			testutil.SkipIfMissing(t, b, compute.OpTypeIota)
			testutil.SkipIfMissing(t, b, compute.OpTypeBroadcastInDim)
			testutil.SkipIfMissing(t, b, compute.OpTypeMul)
			testutil.SkipIfMissing(t, b, compute.OpTypeAdd)
			testutil.SkipIfMissingDType(t, b, dtypes.Int32)
			testutil.SkipIfMissingDType(t, b, dtypes.Int64)

			got2, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
				op, err := f.Iota(shapes.Make(dtypes.Int32, 4, 4), 1)
				if err != nil {
					return nil, err
				}
				row, err := f.Iota(shapes.Make(dtypes.Int32, 4, 4), 0)
				if err != nil {
					return nil, err
				}
				ten, err := f.Constant([]int32{10})
				if err != nil {
					return nil, err
				}
				tenB, err := f.BroadcastInDim(ten, shapes.Make(dtypes.Int32, 4, 4), nil)
				if err != nil {
					return nil, err
				}
				rowScaled, err := f.Mul(row, tenB)
				if err != nil {
					return nil, err
				}
				grid, err := f.Add(rowScaled, op)
				if err != nil {
					return nil, err
				}
				idx, err := f.Constant([]int64{0, 1, 2, 0}, 2, 2)
				if err != nil {
					return nil, err
				}
				return f.Gather(grid, idx, 1, []int{1, 2}, nil, []int{0, 1}, []int{2, 3}, false)
			})
			if err != nil {
				t.Fatalf("Gather 2D window failed: %v", err)
			}
			want2 := [][][]int32{
				{{1, 2, 3}, {11, 12, 13}},
				{{20, 21, 22}, {30, 31, 32}},
			}
			if ok, diff := testutil.IsEqual(want2, got2); !ok {
				t.Fatalf("Gather 2D window mismatch:\n%s", diff)
			}
		})

		t.Run("AnonymousDynamicWindowAndEmpty", func(t *testing.T) {
			testutil.SkipIfMissingDynamicShapes(t, b)
			testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
			testutil.SkipIfMissingDType(t, b, dtypes.Float32)
			testutil.SkipIfMissingDType(t, b, dtypes.Int32)

			builder := b.Builder("gather_anon_window")
			fn := builder.Main()
			opC, err := fn.Constant([]float32{10, 20, 30, 40, 50}, 5)
			if err != nil {
				t.Fatal(err)
			}
			idxP, err := fn.Parameter("idx", shapes.MakeDynamic(dtypes.Int32, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("idx")}), nil)
			if err != nil {
				t.Fatal(err)
			}
			gOut, err := fn.Gather(opC, idxP, 1, []int{1}, nil, []int{0}, []int{2}, false)
			if err != nil {
				t.Fatalf("Gather anonymous dynamic window build failed: %v", err)
			}
			gEmpty, err := fn.Gather(opC, idxP, 1, []int{1}, nil, []int{0}, []int{0}, false)
			if err != nil {
				t.Fatalf("Gather anonymous dynamic empty window build failed: %v", err)
			}
			if err := fn.Return([]compute.Value{gOut, gEmpty}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Finalize()
			inBuf, err := b.BufferFromFlatData(0, []int32{1, 10}, shapes.Make(dtypes.Int32, 2))
			if err != nil {
				t.Fatal(err)
			}
			defer inBuf.Finalize()
			outs, err := ex.Execute([]compute.Buffer{inBuf}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer outs[0].Finalize()
			defer outs[1].Finalize()

			gotWin, err := testutil.FromBuffer(b, outs[0])
			if err != nil {
				t.Fatal(err)
			}
			if ok, diff := testutil.IsEqual([][]float32{{20, 30}, {40, 50}}, gotWin); !ok {
				t.Fatalf("Gather anonymous dynamic window mismatch:\n%s", diff)
			}
			emptySh, err := outs[1].Shape()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(emptySh.Dimensions, []int{2, 0}) {
				t.Fatalf("expected empty gather shape [2, 0], got %v", emptySh.Dimensions)
			}
		})
	})

	t.Run("BroadcastZeroDimensions", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeBroadcastInDim)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		t.Run("StaticZero", func(t *testing.T) {
			b1 := b.Builder("bcast_static_zero")
			fn1 := b1.Main()
			p1, err := fn1.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
			if err != nil {
				t.Fatal(err)
			}
			c1, err := fn1.Constant([]float32{1, 2, 3}, 3)
			if err != nil {
				t.Fatal(err)
			}
			bcParam, err := fn1.BroadcastInDim(p1, shapes.Make(dtypes.Float32, 0, 3), []int{1})
			if err != nil {
				t.Fatal(err)
			}
			bcConst, err := fn1.BroadcastInDim(c1, shapes.Make(dtypes.Float32, 0, 3), []int{1})
			if err != nil {
				t.Fatal(err)
			}
			if err := fn1.Return([]compute.Value{bcParam, bcConst}, nil); err != nil {
				t.Fatal(err)
			}
			ex1, err := b1.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex1.Finalize()
			in1, err := b.BufferFromFlatData(0, []float32{4, 5, 6}, shapes.Make(dtypes.Float32, 3))
			if err != nil {
				t.Fatal(err)
			}
			defer in1.Finalize()
			outs1, err := ex1.Execute([]compute.Buffer{in1}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer outs1[0].Finalize()
			defer outs1[1].Finalize()
			for i := range 2 {
				sh, err := outs1[i].Shape()
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(sh.Dimensions, []int{0, 3}) {
					t.Fatalf("output %d: expected [0, 3], got %v", i, sh.Dimensions)
				}
			}
		})

		t.Run("RuntimeZero", func(t *testing.T) {
			testutil.SkipIfMissingDynamicShapes(t, b)
			testutil.SkipIfMissing(t, b, compute.OpTypeNonZero)
			testutil.SkipIfMissingDType(t, b, dtypes.Int32)

			b2 := b.Builder("bcast_runtime_zero")
			fn2 := b2.Main()
			maskP, err := fn2.Parameter("mask", shapes.Make(dtypes.Int32, 4), nil)
			if err != nil {
				t.Fatal(err)
			}
			nz, err := fn2.NonZero(maskP, compute.NonZeroConfig{})
			if err != nil {
				t.Fatal(err)
			}
			nzShape, err := fn2.Shape(nz)
			if err != nil {
				t.Fatal(err)
			}
			countName := nzShape.AxisName(0)
			rowConst, err := fn2.Constant([]float32{7, 8, 9}, 3)
			if err != nil {
				t.Fatal(err)
			}
			bcRuntimeZero, err := fn2.BroadcastInDim(
				rowConst,
				shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{countName, ""}),
				[]int{1},
			)
			if err != nil {
				t.Fatal(err)
			}
			bcStaticZero, err := fn2.BroadcastInDim(
				rowConst,
				shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 0, 3}, []string{countName, "", ""}),
				[]int{2},
			)
			if err != nil {
				t.Fatal(err)
			}
			if err := fn2.Return([]compute.Value{bcRuntimeZero, bcStaticZero}, nil); err != nil {
				t.Fatal(err)
			}
			ex2, err := b2.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex2.Finalize()
			maskBuf, err := b.BufferFromFlatData(0, []int32{0, 0, 0, 0}, shapes.Make(dtypes.Int32, 4))
			if err != nil {
				t.Fatal(err)
			}
			defer maskBuf.Finalize()
			outs2, err := ex2.Execute([]compute.Buffer{maskBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute with runtime 0 broadcast failed: %v", err)
			}
			defer outs2[0].Finalize()
			defer outs2[1].Finalize()
			sh0, err := outs2[0].Shape()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(sh0.Dimensions, []int{0, 3}) {
				t.Fatalf("bcRuntimeZero: expected [0, 3], got %v", sh0.Dimensions)
			}
			sh1, err := outs2[1].Shape()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(sh1.Dimensions, []int{0, 0, 3}) {
				t.Fatalf("bcStaticZero: expected [0, 0, 3], got %v", sh1.Dimensions)
			}
		})
	})

	t.Run("GeneratedAxisSelfAndRuntimeCheck", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeAdd)
		testutil.SkipIfMissing(t, b, compute.OpTypeMul)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		t.Run("SelfDerivedBinaryOp", func(t *testing.T) {
			// x * (x + 1) on a generated "#..." axis must build and execute across varying sizes.
			builder := b.Builder("gen_axis_self_derived")
			fn := builder.Main()
			x, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("x")}), nil)
			if err != nil {
				t.Fatal(err)
			}
			one, err := fn.Constant([]float32{1})
			if err != nil {
				t.Fatal(err)
			}
			xPlusOne, err := fn.Add(x, one)
			if err != nil {
				t.Fatalf("Add(x, 1) failed: %v", err)
			}
			prod, err := fn.Mul(x, xPlusOne)
			if err != nil {
				t.Fatalf("Mul(x, x+1) failed: %v", err)
			}
			if err := fn.Return([]compute.Value{prod}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Finalize()

			for _, tc := range []struct {
				in   []float32
				want []float32
			}{
				{[]float32{2, 3, 4}, []float32{6, 12, 20}},
				{[]float32{5, 6}, []float32{30, 42}},
			} {
				buf, err := b.BufferFromFlatData(0, tc.in, shapes.Make(dtypes.Float32, len(tc.in)))
				if err != nil {
					t.Fatal(err)
				}
				outs, err := ex.Execute([]compute.Buffer{buf}, nil, 0)
				buf.Finalize()
				if err != nil {
					t.Fatalf("Execute(%v) failed: %v", tc.in, err)
				}
				got, err := testutil.FromBuffer(b, outs[0])
				outs[0].Finalize()
				if err != nil {
					t.Fatal(err)
				}
				if ok, diff := testutil.IsEqual(tc.want, got); !ok {
					t.Fatalf("Mul(x, x+1) mismatch:\n%s", diff)
				}
			}
		})

		t.Run("DistinctGeneratedAxesRuntimeCheck", func(t *testing.T) {
			// Two distinct generated "#" axes unify at build time and check compatibility at runtime.
			builder := b.Builder("gen_axis_runtime_check")
			fn := builder.Main()
			a, err := fn.Parameter("a", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("a")}), nil)
			if err != nil {
				t.Fatal(err)
			}
			bParam, err := fn.Parameter("b", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{shapes.UniqueAxisName("b")}), nil)
			if err != nil {
				t.Fatal(err)
			}
			sum, err := fn.Add(a, bParam)
			if err != nil {
				t.Fatalf("Add(#a, #b) should succeed at build time, got: %v", err)
			}
			if err := fn.Return([]compute.Value{sum}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Finalize()

			// Matching runtime sizes succeed.
			bufA, err := b.BufferFromFlatData(0, []float32{1, 2, 3}, shapes.Make(dtypes.Float32, 3))
			if err != nil {
				t.Fatal(err)
			}
			defer bufA.Finalize()
			bufBMatch, err := b.BufferFromFlatData(0, []float32{10, 20, 30}, shapes.Make(dtypes.Float32, 3))
			if err != nil {
				t.Fatal(err)
			}
			defer bufBMatch.Finalize()
			outs, err := ex.Execute([]compute.Buffer{bufA, bufBMatch}, nil, 0)
			if err != nil {
				t.Fatalf("Execute with matching runtime dims failed: %v", err)
			}
			got, err := testutil.FromBuffer(b, outs[0])
			outs[0].Finalize()
			if err != nil {
				t.Fatal(err)
			}
			if ok, diff := testutil.IsEqual([]float32{11, 22, 33}, got); !ok {
				t.Fatalf("Add(#a, #b) mismatch:\n%s", diff)
			}

			// Mismatched runtime sizes must return an error.
			bufBMismatch, err := b.BufferFromFlatData(0, []float32{10, 20}, shapes.Make(dtypes.Float32, 2))
			if err != nil {
				t.Fatal(err)
			}
			defer bufBMismatch.Finalize()
			if _, err := ex.Execute([]compute.Buffer{bufA, bufBMismatch}, nil, 0); err == nil {
				t.Fatal("expected runtime error when distinct generated axes have incompatible sizes 3 vs 2")
			}
		})
	})

	t.Run("DynamicRNGBitGenerator", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		testutil.SkipIfMissing(t, b, compute.OpTypeParameter)
		testutil.SkipIfMissing(t, b, compute.OpTypeConstant)
		testutil.SkipIfMissing(t, b, compute.OpTypeRNGBitGenerator)
		testutil.SkipIfMissingDType(t, b, dtypes.Uint64)
		testutil.SkipIfMissingDType(t, b, dtypes.Uint32)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		builder := b.Builder("test_dyn_rng")
		fn := builder.Main()

		xShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
		_, err := fn.Parameter("x", xShape, nil)
		if err != nil {
			t.Fatalf("Parameter x failed: %v", err)
		}
		stateParam, err := fn.Parameter("state", compute.RNGStateShape, nil)
		if err != nil {
			t.Fatalf("Parameter state failed: %v", err)
		}
		rngOutShape := shapes.MakeDynamic(dtypes.Uint32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
		newState, bits, err := fn.RNGBitGenerator(stateParam, rngOutShape)
		if err != nil {
			t.Fatalf("RNGBitGenerator with dynamic shape failed: %v", err)
		}
		if err := fn.Return([]compute.Value{newState, bits}, nil); err != nil {
			t.Fatalf("Return failed: %v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %v", err)
		}
		defer exec.Finalize()

		for _, batch := range []int{2, 5, 0} {
			inX, err := b.BufferFromFlatData(0, make([]float32, batch*3), shapes.Make(dtypes.Float32, batch, 3))
			if err != nil {
				t.Fatalf("BufferFromFlatData x(batch=%d) failed: %v", batch, err)
			}
			inState, err := b.BufferFromFlatData(0, []uint64{42, 1, 0}, compute.RNGStateShape)
			if err != nil {
				inX.Finalize()
				t.Fatalf("BufferFromFlatData state failed: %v", err)
			}
			outs, err := exec.Execute([]compute.Buffer{inX, inState}, nil, 0)
			inX.Finalize()
			inState.Finalize()
			if err != nil {
				t.Fatalf("Execute(batch=%d) failed: %v", batch, err)
			}
			gotShape, err := outs[1].Shape()
			if err != nil {
				outs[0].Finalize()
				outs[1].Finalize()
				t.Fatalf("Shape(batch=%d) failed: %v", batch, err)
			}
			wantShape := shapes.Make(dtypes.Uint32, batch, 3)
			if !gotShape.Equal(wantShape) {
				outs[0].Finalize()
				outs[1].Finalize()
				t.Fatalf("batch=%d: got shape %s, want %s", batch, gotShape, wantShape)
			}
			outs[0].Finalize()
			outs[1].Finalize()
		}
	})
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
)

func TestWhere(t *testing.T) {
	sizes := []int{1, 3, 7, 8, 9, 15, 16, 23, 32, 65, 1000, 50000}
	cases := []struct {
		name          string
		onTrueScalar  bool
		onFalseScalar bool
	}{
		{"VectorVector", false, false},
		{"VectorScalar", false, true},
		{"ScalarVector", true, false},
		{"ScalarScalar", true, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, size := range sizes {
				condShape := shapes.Make(dtypes.Bool, size)
				var trueShape, falseShape shapes.Shape
				if c.onTrueScalar {
					trueShape = shapes.Make(dtypes.Float32)
				} else {
					trueShape = shapes.Make(dtypes.Float32, size)
				}
				if c.onFalseScalar {
					falseShape = shapes.Make(dtypes.Float32)
				} else {
					falseShape = shapes.Make(dtypes.Float32, size)
				}

				condData := make([]bool, size)
				for i := 0; i < size; i++ {
					condData[i] = (i%3 != 0)
				}

				var onTrueData, onFalseData []float32
				if c.onTrueScalar {
					onTrueData = []float32{42.0}
				} else {
					onTrueData = make([]float32, size)
					for i := 0; i < size; i++ {
						onTrueData[i] = float32(i + 1)
					}
				}

				if c.onFalseScalar {
					onFalseData = []float32{-999.0}
				} else {
					onFalseData = make([]float32, size)
					for i := 0; i < size; i++ {
						onFalseData[i] = float32(-(i + 1))
					}
				}

				builder := backend.Builder("test_where").(*gobackend.Builder)
				main := builder.Main().(*gobackend.Function)

				condNode, err := main.Parameter("cond", condShape, nil)
				if err != nil {
					t.Fatalf("Parameter cond failed: %+v", err)
				}
				onTrueNode, err := main.Parameter("onTrue", trueShape, nil)
				if err != nil {
					t.Fatalf("Parameter onTrue failed: %+v", err)
				}
				onFalseNode, err := main.Parameter("onFalse", falseShape, nil)
				if err != nil {
					t.Fatalf("Parameter onFalse failed: %+v", err)
				}

				outNode, err := ops.Where(main, condNode, onTrueNode, onFalseNode)
				if err != nil {
					t.Fatalf("Where failed: %+v", err)
				}
				err = main.Return([]compute.Value{outNode}, nil)
				if err != nil {
					t.Fatalf("Return failed: %+v", err)
				}

				exec, err := builder.Compile()
				if err != nil {
					t.Fatalf("Compile failed: %+v", err)
				}

				condBuf := makeBuffer(t, condShape, condData)
				onTrueBuf := makeBuffer(t, trueShape, onTrueData)
				onFalseBuf := makeBuffer(t, falseShape, onFalseData)

				outputs, err := exec.Execute([]compute.Buffer{condBuf, onTrueBuf, onFalseBuf}, nil, 0)
				if err != nil {
					t.Fatalf("Execute failed: %+v", err)
				}

				outFlat := outputs[0].(*gobackend.Buffer).Flat.([]float32)
				for i := 0; i < size; i++ {
					var expected float32
					if condData[i] {
						if c.onTrueScalar {
							expected = onTrueData[0]
						} else {
							expected = onTrueData[i]
						}
					} else {
						if c.onFalseScalar {
							expected = onFalseData[0]
						} else {
							expected = onFalseData[i]
						}
					}
					if outFlat[i] != expected {
						t.Fatalf("%s size=%d, idx=%d: got %f, want %f", c.name, size, i, outFlat[i], expected)
					}
				}
				exec.Finalize()
			}
		})
	}

	t.Run("TrailingBroadcast5D", func(t *testing.T) {
		tgtShape := shapes.Make(dtypes.Float32, 2, 4, 5, 7, 64)
		condShape := shapes.Make(dtypes.Bool, 2, 4, 5, 7, 64)
		trueShape := shapes.Make(dtypes.Float32, 2, 4, 5, 7, 1)
		falseShape := tgtShape

		outerSize := 2 * 4 * 5 * 7
		innerSize := 64
		n := outerSize * innerSize

		condData := make([]bool, n)
		for i := range n {
			condData[i] = (i%5 == 0)
		}
		trueData := make([]float32, outerSize)
		for r := range outerSize {
			trueData[r] = float32(r + 100)
		}
		falseData := make([]float32, n)
		for i := range n {
			falseData[i] = float32(-i - 1)
		}

		builder := backend.Builder("test_where_5d").(*gobackend.Builder)
		main := builder.Main().(*gobackend.Function)
		condNode, _ := main.Parameter("cond", condShape, nil)
		onTrueNode, _ := main.Parameter("onTrue", trueShape, nil)
		onFalseNode, _ := main.Parameter("onFalse", falseShape, nil)
		outNode, err := ops.Where(main, condNode, onTrueNode, onFalseNode)
		if err != nil {
			t.Fatalf("Where failed: %+v", err)
		}
		_ = main.Return([]compute.Value{outNode}, nil)
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %+v", err)
		}
		defer exec.Finalize()

		condBuf := makeBuffer(t, condShape, condData)
		onTrueBuf := makeBuffer(t, trueShape, trueData)
		onFalseBuf := makeBuffer(t, falseShape, falseData)
		// Donate onFalseBuf (index 2) to exercise in-place outAliasesFalse path.
		outputs, err := exec.Execute([]compute.Buffer{condBuf, onTrueBuf, onFalseBuf}, []bool{false, false, true}, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		outFlat := outputs[0].(*gobackend.Buffer).Flat.([]float32)
		for r := range outerSize {
			for c := range innerSize {
				idx := r*innerSize + c
				want := float32(-idx - 1)
				if condData[idx] {
					want = float32(r + 100)
				}
				if outFlat[idx] != want {
					t.Fatalf("idx=%d (r=%d, c=%d): got %v, want %v", idx, r, c, outFlat[idx], want)
				}
			}
		}
	})
}

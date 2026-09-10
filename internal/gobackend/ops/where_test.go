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
}

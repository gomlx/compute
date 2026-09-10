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
	for _, size := range []int{10, 1000, 40000} {
		shape := shapes.Make(dtypes.Float32, size)
		condShape := shapes.Make(dtypes.Bool, size)
		scalarShape := shapes.Make(dtypes.Float32)

		condData := make([]bool, size)
		onTrueData := make([]float32, size)
		onFalseData := []float32{-10000.0} // scalar onFalse

		for i := 0; i < size; i++ {
			condData[i] = (i % 2 == 0)
			onTrueData[i] = float32(i + 1)
		}

		builder := backend.Builder("test_where").(*gobackend.Builder)
		main := builder.Main().(*gobackend.Function)

		condNode, err := main.Parameter("cond", condShape, nil)
		if err != nil {
			t.Fatalf("Parameter cond failed: %+v", err)
		}
		onTrueNode, err := main.Parameter("onTrue", shape, nil)
		if err != nil {
			t.Fatalf("Parameter onTrue failed: %+v", err)
		}
		onFalseNode, err := main.Parameter("onFalse", scalarShape, nil)
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
		defer exec.Finalize()

		condBuf := makeBuffer(t, condShape, condData)
		onTrueBuf := makeBuffer(t, shape, onTrueData)
		onFalseBuf := makeBuffer(t, scalarShape, onFalseData)

		outputs, err := exec.Execute([]compute.Buffer{condBuf, onTrueBuf, onFalseBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}

		outFlat := outputs[0].(*gobackend.Buffer).Flat.([]float32)
		for i := 0; i < size; i++ {
			var expected float32
			if condData[i] {
				expected = onTrueData[i]
			} else {
				expected = onFalseData[0]
			}
			if outFlat[i] != expected {
				t.Fatalf("size=%d, idx=%d: got %f, want %f", size, i, outFlat[i], expected)
			}
		}
	}
}

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

func TestCachedNodeExecutor(t *testing.T) {
	builder := backend.Builder("test_cached_executor").(*gobackend.Builder)
	main := builder.Main().(*gobackend.Function)

	inShape := shapes.Make(dtypes.Float32, 4, 8)
	inNode, err := main.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Failed creating parameter: %+v", err)
	}

	outNode, err := ops.ReduceSum(main, inNode, 1)
	if err != nil {
		t.Fatalf("ReduceSum failed: %+v", err)
	}

	err = main.Return([]compute.Value{outNode}, nil)
	if err != nil {
		t.Fatalf("Return failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %+v", err)
	}

	inData := make([]float32, 32)
	for i := range inData {
		inData[i] = float32(i + 1)
	}
	inBuf, err := backend.BufferFromFlatData(0, inData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %+v", err)
	}

	// First execution: populates node.cachedExecutor
	outputs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed on first run: %+v", err)
	}
	_ = outputs

	// Second execution: hits node.cachedExecutor
	outputs2, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed on second run: %+v", err)
	}
	_ = outputs2
}

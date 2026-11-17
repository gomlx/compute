// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"sync"
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

// TestConcurrentFallbackExecutorCaching verifies that concurrent first-time
// executions of nodes whose SIMD executor falls back to the generic executor
// (e.g., Int64 binary and reduce ops with owned intermediate inputs) do not
// race on node.cachedExecutorIdx and skip eligibility checks after consuming
// inputs.
func TestConcurrentFallbackExecutorCaching(t *testing.T) {
	inShape := shapes.Make(dtypes.Int64, 4, 8)
	inData := make([]int64, inShape.Size())
	for i := range inData {
		inData[i] = int64(i + 2)
	}

	const numWorkers = 8
	const numIterations = 25

	for range numIterations {
		builder := backend.Builder("test_concurrent_fallback").(*gobackend.Builder)
		main := builder.Main().(*gobackend.Function)

		x, err := main.Parameter("x", inShape, nil)
		if err != nil {
			t.Fatalf("Parameter: %+v", err)
		}
		// Produce owned intermediate buffers so binaryOperandsAndOutputForExecution
		// would consume inputs[i] if a SIMD executor mistakenly entered its body.
		a, err := ops.Neg(main, x)
		if err != nil {
			t.Fatalf("Neg(x): %+v", err)
		}
		b, err := ops.Neg(main, x)
		if err != nil {
			t.Fatalf("Neg(x): %+v", err)
		}
		divNode, err := ops.Div(main, a, b)
		if err != nil {
			t.Fatalf("Div: %+v", err)
		}
		mulNode, err := ops.Mul(main, a, b)
		if err != nil {
			t.Fatalf("Mul: %+v", err)
		}
		addNode, err := ops.Add(main, divNode, mulNode)
		if err != nil {
			t.Fatalf("Add: %+v", err)
		}
		redSum, err := ops.ReduceSum(main, addNode, 1)
		if err != nil {
			t.Fatalf("ReduceSum: %+v", err)
		}
		redMax, err := ops.ReduceMax(main, addNode, 1)
		if err != nil {
			t.Fatalf("ReduceMax: %+v", err)
		}

		if err := main.Return([]compute.Value{redSum, redMax}, nil); err != nil {
			t.Fatalf("Return: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile: %+v", err)
		}

		var startWG, doneWG sync.WaitGroup
		startWG.Add(1)
		errs := make(chan error, numWorkers)

		for range numWorkers {
			doneWG.Add(1)
			go func() {
				defer doneWG.Done()
				inBuf, bufErr := backend.BufferFromFlatData(0, inData, inShape)
				if bufErr != nil {
					errs <- bufErr
					return
				}
				defer func() { _ = inBuf.Finalize() }()

				startWG.Wait()
				outs, execErr := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
				if execErr != nil {
					errs <- execErr
					return
				}
				for _, out := range outs {
					_ = out.Finalize()
				}
			}()
		}

		startWG.Done()
		doneWG.Wait()
		close(errs)
		for execErr := range errs {
			t.Fatalf("concurrent Execute failed: %+v", execErr)
		}
		exec.Finalize()
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend_test

import (
	"runtime"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
)

// outputPoolSide is the side of the square float32 outputs of these tests: 256 KiB each.
const (
	outputPoolSide  = 256
	outputPoolElems = outputPoolSide * outputPoolSide
)

// outputPoolShape is the shape of the input and the output of the test functions.
var outputPoolShape = shapes.Make(dtypes.Float32, outputPoolSide, outputPoolSide)

// compileOutputPool compiles f(x) = Transpose(Transpose(x + x)), or f(x) = x when
// identity is set. The first Transpose frees x + x to the buffer pool, and the second
// Transpose takes that buffer from the pool for its output: the output is a pooled
// buffer.
func compileOutputPool(t *testing.T, backend compute.Backend, identity bool) compute.Executable {
	t.Helper()
	builder := backend.Builder("output_pool")
	fn := builder.Main()
	x, err := fn.Parameter("x", outputPoolShape, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := x
	if !identity {
		sum, err := fn.Add(x, x)
		if err != nil {
			t.Fatal(err)
		}
		tr, err := fn.Transpose(sum, 1, 0)
		if err != nil {
			t.Fatal(err)
		}
		if out, err = fn.Transpose(tr, 1, 0); err != nil {
			t.Fatal(err)
		}
	}
	if err := fn.Return([]compute.Value{out}, nil); err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(exec.Finalize)
	return exec
}

func heapInUse() uint64 {
	var ms runtime.MemStats
	runtime.GC()
	runtime.GC()
	runtime.ReadMemStats(&ms)
	return ms.HeapInuse
}

// TestExecuteOutputMemoryReleased: an output that the caller finalizes does not stay
// alive in the buffer pool. 500 calls with 256 KiB outputs would keep 125 MiB.
func TestExecuteOutputMemoryReleased(t *testing.T) {
	backend, err := gobackend.New("")
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Finalize()
	exec := compileOutputPool(t, backend, false)
	in, err := backend.BufferFromFlatData(0, make([]float32, outputPoolElems), outputPoolShape)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Finalize()
	run := func(n int) {
		for range n {
			outs, err := exec.Execute([]compute.Buffer{in}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			if err := outs[0].Finalize(); err != nil {
				t.Fatal(err)
			}
		}
	}
	run(20)
	before := heapInUse()
	run(500)
	after := heapInUse()
	if grown := int64(after) - int64(before); grown > 16<<20 {
		t.Fatalf("heap in use grew by %d MiB over 500 calls", grown>>20)
	}
}

// TestExecuteDonatedOutputNotAliased: when a function returns a donated input that came
// from the pool, the output is a handle of its own. Finalizing the donated input handle
// leaves the output intact.
func TestExecuteDonatedOutputNotAliased(t *testing.T) {
	backend, err := gobackend.New("")
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Finalize()
	double := compileOutputPool(t, backend, false)
	identity := compileOutputPool(t, backend, true)
	flat := make([]float32, outputPoolElems)
	for i := range flat {
		flat[i] = float32(i)
	}
	in, err := backend.BufferFromFlatData(0, flat, outputPoolShape)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Finalize()
	// y comes from the pool.
	ys, err := double.Execute([]compute.Buffer{in}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	y := ys[0]
	outs, err := identity.Execute([]compute.Buffer{y}, []bool{true}, 0)
	if err != nil {
		t.Fatal(err)
	}
	out := outs[0]
	defer out.Finalize()
	if out == y {
		t.Fatal("the output is the handle of the donated input")
	}
	if err := y.Finalize(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, outputPoolElems)
	if err := out.ToFlatData(got); err != nil {
		t.Fatalf("the output after the donated input was finalized: %v", err)
	}
	for i, v := range got {
		if v != 2*float32(i) {
			t.Fatalf("output[%d] = %g, want %g", i, v, 2*float32(i))
		}
	}
}

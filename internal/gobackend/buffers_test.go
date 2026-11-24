// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend_test

import (
	"runtime"
	"sync"
	"testing"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestBuffers_Bytes(t *testing.T) {
	buf, err := backend.(*gobackend.Backend).GetBuffer(shapes.Make(dtypes.Int32, 3))
	if err != nil {
		t.Fatalf("Failed to get buffer: %+v", err)
	}
	buf.Zeros()
	if len(buf.Flat.([]int32)) != 3 {
		t.Fatalf("Expected length 3, got %d", len(buf.Flat.([]int32)))
	}
	flatBytes, err := buf.MutableBytes()
	if err != nil {
		t.Fatalf("Failed to get mutable bytes: %+v", err)
	}
	if len(flatBytes) != 3*int(dtypes.Int32.Size()) {
		t.Fatalf("Expected length %d, got %d", 3*int(dtypes.Int32.Size()), len(flatBytes))
	}
	flatBytes[0] = 1
	flatBytes[4] = 7
	flatBytes[8] = 3
	if ok, diff := testutil.IsEqual([]int32{1, 7, 3}, buf.Flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
	runtime.KeepAlive(buf)
}

func TestBuffers_Fill(t *testing.T) {
	buf, err := backend.(*gobackend.Backend).GetBuffer(shapes.Make(dtypes.Int32, 3))
	if err != nil {
		t.Fatalf("Failed to get buffer: %+v", err)
	}
	if len(buf.Flat.([]int32)) != 3 {
		t.Fatalf("Expected length 3, got %d", len(buf.Flat.([]int32)))
	}
	if err := buf.Fill(int32(3)); err != nil {
		t.Fatalf("Failed to fill buffer: %+v", err)
	}
	if ok, diff := testutil.IsEqual([]int32{3, 3, 3}, buf.Flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}

	buf.Zeros()
	if ok, diff := testutil.IsEqual([]int32{0, 0, 0}, buf.Flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
}

func TestBufferPool_LIFOAndConcurrent(t *testing.T) {
	be, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	shape := shapes.Make(dtypes.Float32, 64)
	b1, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer b1 failed: %+v", err)
	}
	b2, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer b2 failed: %+v", err)
	}
	if b1 == b2 {
		t.Fatalf("Expected distinct buffers when both are in use")
	}

	// Return b1 then b2; LIFO stack must return b2 first, then b1.
	be.PutBuffer(b1)
	be.PutBuffer(b2)

	got1, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer got1 failed: %+v", err)
	}
	if got1 != b2 {
		t.Fatalf("Expected LIFO order: got %p, want b2 (%p)", got1, b2)
	}
	got2, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer got2 failed: %+v", err)
	}
	if got2 != b1 {
		t.Fatalf("Expected LIFO order: got %p, want b1 (%p)", got2, b1)
	}
	be.PutBuffer(got1)
	be.PutBuffer(got2)

	// Concurrent stress test across 32 goroutines hammering the same bucket size.
	const numWorkers = 32
	const iters = 2000
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for w := range numWorkers {
		go func(workerID int) {
			defer wg.Done()
			val := float32(workerID + 1)
			for range iters {
				buf1, err := be.GetBuffer(shape)
				if err != nil {
					t.Errorf("GetBuffer failed: %+v", err)
					return
				}
				flat1 := buf1.Flat.([]float32)
				flat1[0] = val
				flat1[len(flat1)-1] = val

				buf2, err := be.GetBuffer(shape)
				if err != nil {
					t.Errorf("GetBuffer 2 failed: %+v", err)
					return
				}
				flat2 := buf2.Flat.([]float32)
				flat2[0] = -val

				if flat1[0] != val || flat1[len(flat1)-1] != val {
					t.Errorf("Concurrent buffer corruption on worker %d: got (%v, %v), want %v",
						workerID, flat1[0], flat1[len(flat1)-1], val)
					return
				}
				be.PutBuffer(buf2)
				be.PutBuffer(buf1)
			}
		}(w)
	}
	wg.Wait()
}

func TestBuffers_Alignment64(t *testing.T) {
	be, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	// Small buffers (< 64 bytes) skip alignment padding; verify both < 64B (no crash/length mismatch)
	// and >= 64B (including exact 8 KiB / 32 KiB / 64 KiB page multiples from TwoBitBucketLen).
	for _, size := range []int{1, 3, 7, 15, 16, 17, 64, 1023, 8192, 16384} {
		shape := shapes.Make(dtypes.Float32, size)
		wantAligned := size*4 >= 64

		pooledBuf, err := be.GetBuffer(shape)
		if err != nil {
			t.Fatalf("GetBuffer(%s) failed: %+v", shape, err)
		}
		if len(pooledBuf.RawBytes) < size*4 {
			t.Errorf("GetBuffer(%s): RawBytes len %d < %d", shape, len(pooledBuf.RawBytes), size*4)
		}
		if ptr := uintptr(unsafe.Pointer(unsafe.SliceData(pooledBuf.RawBytes))); wantAligned && ptr%64 != 0 {
			t.Errorf("GetBuffer(%s): RawBytes ptr 0x%x is not 64-byte aligned (mod=%d)", shape, ptr, ptr%64)
		}
		be.PutBuffer(pooledBuf)

		exactBuf := gobackend.NewBuffer(shape)
		if len(exactBuf.RawBytes) != size*4 {
			t.Errorf("NewBuffer(%s): RawBytes len %d != %d", shape, len(exactBuf.RawBytes), size*4)
		}
		if ptr := uintptr(unsafe.Pointer(unsafe.SliceData(exactBuf.RawBytes))); wantAligned && ptr%64 != 0 {
			t.Errorf("NewBuffer(%s): RawBytes ptr 0x%x is not 64-byte aligned (mod=%d)", shape, ptr, ptr%64)
		}
		_ = exactBuf.Finalize()
	}
}

func BenchmarkBufferPoolSequential(b *testing.B) {
	be, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	shape := shapes.Make(dtypes.Float32, 1024)
	buf, err := be.GetBuffer(shape)
	if err != nil {
		b.Fatalf("GetBuffer failed: %+v", err)
	}
	be.PutBuffer(buf)

	for b.Loop() {
		buf, err := be.GetBuffer(shape)
		if err != nil {
			b.Fatalf("GetBuffer failed: %+v", err)
		}
		be.PutBuffer(buf)
	}
}

func BenchmarkBufferPoolParallel(b *testing.B) {
	be, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	shape := shapes.Make(dtypes.Float32, 1024)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf, err := be.GetBuffer(shape)
			if err != nil {
				b.Fatalf("GetBuffer failed: %+v", err)
			}
			be.PutBuffer(buf)
		}
	})
}

// outputPoolSide is the side of the square float32 outputs of TestExecuteOutput: 256 KiB each.
const (
	outputPoolSide  = 256
	outputPoolElems = outputPoolSide * outputPoolSide
)

// outputPoolShape is the shape of the input and the output of the TestExecuteOutput functions.
var outputPoolShape = shapes.Make(dtypes.Float32, outputPoolSide, outputPoolSide)

// compileOutputPool compiles f(x) = Transpose(Transpose(x + x)), or f(x) = x when
// identity is set. The first Transpose frees x + x to the buffer pool, and the second
// Transpose takes that buffer from the pool for its output: the output is a pooled
// buffer.
func compileOutputPool(t *testing.T, be compute.Backend, identity bool) compute.Executable {
	t.Helper()
	builder := be.Builder("output_pool")
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

func TestExecuteOutput(t *testing.T) {
	// MemoryReleased: an output from a pooled buffer does not stay alive in the buffer pool,
	// both when the caller calls Finalize() and when the caller drops the output for GC.
	// 500 calls with 256 KiB outputs would keep ~125 MiB if retained by the pool.
	t.Run("MemoryReleased", func(t *testing.T) {
		for _, finalizeOutput := range []bool{true, false} {
			name := "WithFinalize"
			if !finalizeOutput {
				name = "WithoutFinalizeGC"
			}
			t.Run(name, func(t *testing.T) {
				be, err := gobackend.NewBackend()
				if err != nil {
					t.Fatal(err)
				}
				defer be.Finalize()
				exec := compileOutputPool(t, be, false)
				in, err := be.BufferFromFlatData(0, make([]float32, outputPoolElems), outputPoolShape)
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
						if finalizeOutput {
							if err := outs[0].Finalize(); err != nil {
								t.Fatal(err)
							}
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
			})
		}
	})

	// DonatedOutputNotAliased: when a function returns a donated input that came from the
	// pool, the output is a handle of its own, the donated input handle is immediately
	// invalidated, and finalizing the donated input handle is a no-op leaving the output intact.
	t.Run("DonatedOutputNotAliased", func(t *testing.T) {
		be, err := gobackend.NewBackend()
		if err != nil {
			t.Fatal(err)
		}
		defer be.Finalize()
		double := compileOutputPool(t, be, false)
		identity := compileOutputPool(t, be, true)
		flat := make([]float32, outputPoolElems)
		for i := range flat {
			flat[i] = float32(i)
		}
		in, err := be.BufferFromFlatData(0, flat, outputPoolShape)
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
		// Donated input y should already be invalidated right after Execute.
		probe := make([]float32, outputPoolElems)
		if err := y.ToFlatData(probe); err == nil {
			t.Fatal("expected donated input y to be invalidated after Execute")
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
	})
}

// TestBufferPool_SlotReusePastCap verifies that repeated GetBuffer+Finalize cycles (which
// remove buffers from the pool without returning them via PutBuffer) recycle slot indices
// cleanly and do not exhaust the 16,384-slot per-bucket cap.
func TestBufferPool_SlotReusePastCap(t *testing.T) {
	be, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	shape := shapes.Make(dtypes.Float32, 16)
	const iterations = 20_000 // > poolMaxSlots (16,384)
	for range iterations {
		// Put a buffer in, pop it out, then finalize it (simulating a pooled output handed to a user).
		b, err := be.GetBuffer(shape)
		if err != nil {
			t.Fatalf("GetBuffer failed: %+v", err)
		}
		be.PutBuffer(b)
		out, err := be.GetBuffer(shape)
		if err != nil {
			t.Fatalf("GetBuffer (reused) failed: %+v", err)
		}
		if err := out.Finalize(); err != nil {
			t.Fatalf("Finalize failed: %+v", err)
		}
	}

	// After >16,384 cycles, the pool must still accept and reuse a buffer in this bucket.
	b, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer post-loop failed: %+v", err)
	}
	be.PutBuffer(b)
	reused, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer reuse check failed: %+v", err)
	}
	if reused != b {
		t.Fatalf("pool stopped reusing buffers after %d iterations: got %p, want %p", iterations, reused, b)
	}
	be.PutBuffer(reused)
}

// TestExecute_DonatedInputFinalizedOnError verifies that donated input buffers are
// finalized/invalidated even when Execute returns a validation error.
func TestExecute_DonatedInputFinalizedOnError(t *testing.T) {
	be, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	builder := be.Builder("donate_err")
	fn := builder.Main()
	x, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 4), nil)
	if err != nil {
		t.Fatal(err)
	}
	y, err := fn.Parameter("y", shapes.Make(dtypes.Float32, 4), nil)
	if err != nil {
		t.Fatal(err)
	}
	sum, err := fn.Add(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if err := fn.Return([]compute.Value{sum}, nil); err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()

	t.Run("ShapeMismatch", func(t *testing.T) {
		good, err := be.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
		if err != nil {
			t.Fatal(err)
		}
		bad, err := be.BufferFromFlatData(0, []float32{1, 2}, shapes.Make(dtypes.Float32, 2))
		if err != nil {
			t.Fatal(err)
		}
		defer bad.Finalize()

		if _, err := exec.Execute([]compute.Buffer{good, bad}, []bool{true, false}, 0); err == nil {
			t.Fatal("expected shape mismatch error")
		}
		// Donated buffer 'good' must be invalidated even though Execute failed.
		if _, err := good.Shape(); err == nil {
			t.Fatal("expected donated buffer to be finalized after validation error")
		}
		if err := good.Finalize(); err != nil {
			t.Fatalf("idempotent Finalize on donated buffer failed: %+v", err)
		}
		// Non-donated buffer 'bad' must remain valid.
		if _, err := bad.Shape(); err != nil {
			t.Fatalf("expected non-donated buffer to remain valid, got: %+v", err)
		}
	})

	t.Run("DuplicateDonated", func(t *testing.T) {
		buf, err := be.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := exec.Execute([]compute.Buffer{buf, buf}, []bool{true, true}, 0); err == nil {
			t.Fatal("expected error when donating the same buffer twice")
		}
		if _, err := buf.Shape(); err == nil {
			t.Fatal("expected doubly-donated buffer to be finalized on error")
		}
		if err := buf.Finalize(); err != nil {
			t.Fatalf("idempotent Finalize failed: %+v", err)
		}
	})

	t.Run("DonatedAndNonDonatedAlias", func(t *testing.T) {
		buf, err := be.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 4))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := exec.Execute([]compute.Buffer{buf, buf}, []bool{false, true}, 0); err == nil {
			t.Fatal("expected error when same buffer is both donated and non-donated")
		}
		if _, err := buf.Shape(); err == nil {
			t.Fatal("expected donated buffer to be finalized on error")
		}
		if err := buf.Finalize(); err != nil {
			t.Fatalf("idempotent Finalize failed: %+v", err)
		}
	})
}

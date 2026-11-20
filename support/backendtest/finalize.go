// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// TestFinalize verifies backend finalization semantics:
//   - Idempotent Backend.Finalize() and IsFinalized() reporting.
//   - Derived objects (Builder, Executable, Buffer) return errors wrapping compute.ErrBackendFinalized
//     instead of panicking or crashing after Backend.Finalize().
//   - Finalize() on derived Buffer and Executable objects after Backend.Finalize() is a safe no-op.
//   - Concurrent Execute / buffer creation / buffer reads race cleanly with Backend.Finalize(),
//     which waits for in-flight operations before tearing down resources.
//
// Because these tests finalize the backend, callers provide a factory function newBackend
// that constructs a fresh Backend instance per subtest.
func TestFinalize(t *testing.T, newBackend func() (compute.Backend, error)) {
	t.Run("PostFinalizeErrors", func(t *testing.T) {
		b, err := newBackend()
		if err != nil {
			t.Fatalf("newBackend failed: %+v", err)
		}
		if b.IsFinalized() {
			t.Fatalf("fresh backend reported IsFinalized() == true")
		}
		testutil.SkipIfMissing(t, b, compute.OpTypeAdd)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		shape := shapes.Make(dtypes.Float32, 4)

		// Compile an executable before finalizing.
		builder := b.Builder("finalize_add")
		fn := builder.Main()
		x, err := fn.Parameter("x", shape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %+v", err)
		}
		y, err := fn.Add(x, x)
		if err != nil {
			t.Fatalf("Add failed: %+v", err)
		}
		if err := fn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatalf("Return failed: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %+v", err)
		}

		// Create an uncompiled builder before finalizing.
		uncompiled := b.Builder("finalize_uncompiled")
		uFn := uncompiled.Main()
		ux, err := uFn.Parameter("x", shape, nil)
		if err != nil {
			t.Fatalf("uncompiled Parameter failed: %+v", err)
		}
		uy, err := uFn.Add(ux, ux)
		if err != nil {
			t.Fatalf("uncompiled Add failed: %+v", err)
		}
		if err := uFn.Return([]compute.Value{uy}, nil); err != nil {
			t.Fatalf("uncompiled Return failed: %+v", err)
		}

		// Create a buffer and run once to produce an output buffer.
		inBuf, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shape)
		if err != nil {
			t.Fatalf("BufferFromFlatData failed: %+v", err)
		}
		outputs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatalf("initial Execute failed: %+v", err)
		}
		outBuf := outputs[0]

		var sharedBuf compute.Buffer
		hasShared := b.HasSharedBuffers()
		if hasShared {
			sharedBuf, _, err = b.NewSharedBuffer(0, shape)
			if err != nil {
				t.Fatalf("NewSharedBuffer failed: %+v", err)
			}
		}

		// Finalize backend twice to check idempotency.
		b.Finalize()
		b.Finalize()
		if !b.IsFinalized() {
			t.Fatalf("expected IsFinalized() == true after Finalize()")
		}

		// Backend operations must fail with ErrBackendFinalized.
		if _, err := b.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shape); !compute.IsBackendFinalized(err) {
			t.Errorf("BufferFromFlatData after Finalize: expected ErrBackendFinalized, got: %v", err)
		}
		if hasShared {
			if _, _, err := b.NewSharedBuffer(0, shape); !compute.IsBackendFinalized(err) {
				t.Errorf("NewSharedBuffer after Finalize: expected ErrBackendFinalized, got: %v", err)
			}
		}

		// Uncompiled Builder.Compile must fail with ErrBackendFinalized.
		if _, err := uncompiled.Compile(); !compute.IsBackendFinalized(err) {
			t.Errorf("Builder.Compile after Finalize: expected ErrBackendFinalized, got: %v", err)
		}

		// Executable.Execute must fail with ErrBackendFinalized.
		if _, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0); !compute.IsBackendFinalized(err) {
			t.Errorf("Executable.Execute after Finalize: expected ErrBackendFinalized, got: %v", err)
		}

		// Buffer operations must fail with ErrBackendFinalized.
		dst := make([]float32, 4)
		if err := inBuf.ToFlatData(dst); !compute.IsBackendFinalized(err) {
			t.Errorf("inBuf.ToFlatData after Finalize: expected ErrBackendFinalized, got: %v", err)
		}
		if err := outBuf.ToFlatData(dst); !compute.IsBackendFinalized(err) {
			t.Errorf("outBuf.ToFlatData after Finalize: expected ErrBackendFinalized, got: %v", err)
		}
		if sharedBuf != nil {
			if _, err := sharedBuf.Data(); !compute.IsBackendFinalized(err) {
				t.Errorf("sharedBuf.Data after Finalize: expected ErrBackendFinalized, got: %v", err)
			}
		}

		// Finalize on derived objects after Backend.Finalize must be safe no-ops.
		if err := inBuf.Finalize(); err != nil {
			t.Errorf("inBuf.Finalize after Backend.Finalize returned error: %+v", err)
		}
		if err := outBuf.Finalize(); err != nil {
			t.Errorf("outBuf.Finalize after Backend.Finalize returned error: %+v", err)
		}
		if sharedBuf != nil {
			if err := sharedBuf.Finalize(); err != nil {
				t.Errorf("sharedBuf.Finalize after Backend.Finalize returned error: %+v", err)
			}
		}
		exec.Finalize()
		exec.Finalize()
	})

	t.Run("ConcurrentExecuteAndFinalize", func(t *testing.T) {
		b, err := newBackend()
		if err != nil {
			t.Fatalf("newBackend failed: %+v", err)
		}
		testutil.SkipIfMissing(t, b, compute.OpTypeAdd)
		testutil.SkipIfMissingDType(t, b, dtypes.Float32)

		const n = 256
		shape := shapes.Make(dtypes.Float32, n)
		builder := b.Builder("concurrent_finalize")
		fn := builder.Main()
		x, err := fn.Parameter("x", shape, nil)
		if err != nil {
			t.Fatalf("Parameter failed: %+v", err)
		}
		y, err := fn.Add(x, x)
		if err != nil {
			t.Fatalf("Add failed: %+v", err)
		}
		if err := fn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatalf("Return failed: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile failed: %+v", err)
		}

		const numWorkers = 4
		const warmupIters = 2
		var completed atomic.Int64
		var wg sync.WaitGroup
		var warmupWG sync.WaitGroup
		wg.Add(numWorkers)
		warmupWG.Add(numWorkers)

		for w := range numWorkers {
			go func(workerID int) {
				defer wg.Done()
				warmed := false
				defer func() {
					if !warmed {
						warmupWG.Done()
					}
				}()
				inData := make([]float32, n)
				outData := make([]float32, n)
				for i := range inData {
					inData[i] = float32(workerID + 1)
				}
				want := float32(2 * (workerID + 1))

				for iter := 0; ; iter++ {
					inBuf, err := b.BufferFromFlatData(0, inData, shape)
					if err != nil {
						if !compute.IsBackendFinalized(err) {
							t.Errorf("worker %d: unexpected BufferFromFlatData error: %+v", workerID, err)
						}
						return
					}
					outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
					if err != nil {
						_ = inBuf.Finalize()
						if !compute.IsBackendFinalized(err) {
							t.Errorf("worker %d: unexpected Execute error: %+v", workerID, err)
						}
						return
					}
					err = outs[0].ToFlatData(outData)
					_ = outs[0].Finalize()
					_ = inBuf.Finalize()
					if err != nil {
						if !compute.IsBackendFinalized(err) {
							t.Errorf("worker %d: unexpected ToFlatData error: %+v", workerID, err)
						}
						return
					}
					if outData[0] != want || outData[n-1] != want {
						t.Errorf("worker %d: unexpected output (%v, %v), want %v", workerID, outData[0], outData[n-1], want)
						return
					}
					completed.Add(1)
					if !warmed && iter+1 >= warmupIters {
						warmed = true
						warmupWG.Done()
					}
				}
			}(w)
		}

		// Wait until every worker has completed warmup iterations before finalizing concurrently.
		warmupWG.Wait()

		// Call Finalize concurrently from two goroutines while workers are active.
		var finWG sync.WaitGroup
		finWG.Add(2)
		for range 2 {
			go func() {
				defer finWG.Done()
				b.Finalize()
			}()
		}
		finWG.Wait()
		wg.Wait()

		if !b.IsFinalized() {
			t.Fatalf("expected backend to be finalized")
		}
		if completed.Load() == 0 {
			t.Errorf("expected at least one successful execution before Finalize")
		}
	})
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dense_test

import (
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	igobackend "github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
)

func buildFusedDenseBF16Exec(t *testing.T, b compute.Backend, wBuf compute.Buffer, donate bool, kDim, nDim int) compute.Executable {
	t.Helper()
	bldr := b.Builder("dense_arena")
	fn := bldr.Main()
	x, err := fn.Parameter("x", shapes.Make(dtypes.BFloat16, 4, kDim), nil)
	if err != nil {
		t.Fatalf("Parameter: %+v", err)
	}
	w, err := fn.ConstantFromBuffer(wBuf, donate)
	if err != nil {
		t.Fatalf("ConstantFromBuffer: %+v", err)
	}
	y, err := fn.FusedDense(x, w, nil, compute.DenseConfig{})
	if err != nil {
		t.Fatalf("FusedDense: %+v", err)
	}
	if err := fn.Return([]compute.Value{y}, nil); err != nil {
		t.Fatalf("Return: %+v", err)
	}
	exec, err := bldr.Compile()
	if err != nil {
		t.Fatalf("Compile: %+v", err)
	}
	return exec
}

func makeTestBF16Buffer(t *testing.T, b compute.Backend, dims ...int) compute.Buffer {
	t.Helper()
	shape := shapes.Make(dtypes.BFloat16, dims...)
	data := make([]bfloat16.BFloat16, shape.Size())
	for i := range data {
		v := float32((i%17)-8) * 0.125
		data[i] = bfloat16.FromFloat32(v)
	}
	buf, err := b.BufferFromFlatData(0, data, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData: %+v", err)
	}
	return buf
}

func execBF16AndRead(t *testing.T, exec compute.Executable, xBuf compute.Buffer) []float32 {
	t.Helper()
	outs, err := exec.Execute([]compute.Buffer{xBuf}, []bool{false}, 0)
	if err != nil {
		t.Fatalf("Execute: %+v", err)
	}
	if len(outs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outs))
	}
	defer func() { _ = outs[0].Finalize() }()
	shape, err := outs[0].Shape()
	if err != nil {
		t.Fatalf("Shape: %+v", err)
	}
	raw := make([]bfloat16.BFloat16, shape.Size())
	if err := outs[0].ToFlatData(raw); err != nil {
		t.Fatalf("ToFlatData: %+v", err)
	}
	res := make([]float32, len(raw))
	for i, v := range raw {
		res[i] = v.Float32()
	}
	return res
}

func assertF32Close(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("mismatch at %d: got %g want %g (tol %g)", i, got[i], want[i], tol)
		}
	}
}

func TestConstantArenaFusedDenseRefcounting(t *testing.T) {
	const (
		kDim = 128
		nDim = 128
	)

	t.Run("non_donated_constant_refcount_and_repack", func(t *testing.T) {
		b, err := igobackend.New("constant_arena")
		if err != nil {
			t.Fatalf("New: %+v", err)
		}
		defer b.Finalize()
		ib := b.(*igobackend.Backend)

		wBuf := makeTestBF16Buffer(t, b, kDim, nDim)
		xBuf := makeTestBF16Buffer(t, b, 4, kDim)
		defer func() { _ = xBuf.Finalize() }()

		exec1 := buildFusedDenseBF16Exec(t, b, wBuf, false, kDim, nDim)
		exec2 := buildFusedDenseBF16Exec(t, b, wBuf, false, kDim, nDim)

		// Before first execution, no pre-packed panels have been allocated yet.
		chunks, _, _ := ib.ConstantArenaStats()
		if chunks != 0 {
			t.Fatalf("expected 0 chunks before first execute, got %d", chunks)
		}

		out1 := execBF16AndRead(t, exec1, xBuf)
		chunks, mapped, used := ib.ConstantArenaStats()
		if chunks != 1 || mapped <= 0 || used <= 0 {
			t.Fatalf("expected 1 mapped chunk after first execute, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}

		out2 := execBF16AndRead(t, exec2, xBuf)
		assertF32Close(t, out2, out1, 1e-4)

		// Finalizing exec1 leaves exec2 alive -> chunk must remain mapped and exec2 must work.
		exec1.Finalize()
		chunks, _, _ = ib.ConstantArenaStats()
		if chunks != 1 {
			t.Fatalf("expected 1 chunk while exec2 is still live, got %d", chunks)
		}
		out2After := execBF16AndRead(t, exec2, xBuf)
		assertF32Close(t, out2After, out1, 1e-4)

		// Finalizing exec2 drops refcount to 0 -> chunk must be immediately unmapped!
		exec2.Finalize()
		chunks, mapped, used = ib.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected 0 chunks after exec2 finalize, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}

		// Because wBuf was NOT donated, compiling and running exec3 from the same wBuf
		// seamlessly re-packs it into a fresh arena chunk.
		exec3 := buildFusedDenseBF16Exec(t, b, wBuf, false, kDim, nDim)
		out3 := execBF16AndRead(t, exec3, xBuf)
		assertF32Close(t, out3, out1, 1e-4)
		chunks, _, _ = ib.ConstantArenaStats()
		if chunks != 1 {
			t.Fatalf("expected 1 chunk after repack, got %d", chunks)
		}
		exec3.Finalize()
		chunks, _, _ = ib.ConstantArenaStats()
		if chunks != 0 {
			t.Fatalf("expected 0 chunks after exec3 finalize, got %d", chunks)
		}
		_ = wBuf.Finalize()
	})

	t.Run("donated_constant_shared_across_prefill_and_decode", func(t *testing.T) {
		b, err := igobackend.New("constant_arena")
		if err != nil {
			t.Fatalf("New: %+v", err)
		}
		defer b.Finalize()
		ib := b.(*igobackend.Backend)

		wBuf := makeTestBF16Buffer(t, b, kDim, nDim)
		xBuf := makeTestBF16Buffer(t, b, 4, kDim)
		defer func() { _ = xBuf.Finalize() }()

		// Shared ConstantData (simulating GoMLX's SharedConstants across prefill & decode graphs).
		constData, err := igobackend.NewConstantDataFromBuffer(ib, wBuf.(*igobackend.Buffer), true)
		if err != nil {
			t.Fatalf("NewConstantDataFromBuffer: %+v", err)
		}

		buildFromConstData := func(name string) (compute.Executable, error) {
			bldr := b.Builder(name).(*igobackend.Builder)
			fn := bldr.MainFn
			x, err := fn.Parameter("x", shapes.Make(dtypes.BFloat16, 4, kDim), nil)
			if err != nil {
				return nil, err
			}
			wNode := fn.NewNode(compute.OpTypeConstant, constData.Shape())
			wNode.Data = constData
			y, err := fn.FusedDense(x, wNode, nil, compute.DenseConfig{})
			if err != nil {
				return nil, err
			}
			if err := fn.Return([]compute.Value{y}, nil); err != nil {
				return nil, err
			}
			return bldr.Compile()
		}

		execPrefill, err := buildFromConstData("prefill")
		if err != nil {
			t.Fatalf("compile prefill: %+v", err)
		}
		execDecode, err := buildFromConstData("decode")
		if err != nil {
			t.Fatalf("compile decode: %+v", err)
		}

		out1 := execBF16AndRead(t, execPrefill, xBuf)
		if constData.Buffer() != nil {
			t.Fatalf("donated raw buffer should be freed after first pre-pack")
		}
		chunks, _, _ := ib.ConstantArenaStats()
		if chunks != 1 {
			t.Fatalf("expected 1 chunk, got %d", chunks)
		}

		// Finalize prefill while decode is still alive -> chunk stays mapped and decode succeeds.
		execPrefill.Finalize()
		chunks, _, _ = ib.ConstantArenaStats()
		if chunks != 1 {
			t.Fatalf("expected 1 chunk while decode is live, got %d", chunks)
		}
		out2 := execBF16AndRead(t, execDecode, xBuf)
		assertF32Close(t, out2, out1, 1e-4)

		// Finalize decode -> last user gone -> arena chunk immediately unmapped!
		execDecode.Finalize()
		chunks, mapped, used := ib.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected 0 chunks after decode finalize, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}

		// Attempting to compile a new graph with the already-released donated ConstantData must
		// return a descriptive error rather than panicking or dereferencing unmapped memory.
		_, err = buildFromConstData("after_release")
		if err == nil || !strings.Contains(err.Error(), "constant_arena") {
			t.Fatalf("expected constant_arena release error, got %v", err)
		}
		chunks, _, _ = ib.ConstantArenaStats()
		if chunks != 0 {
			t.Fatalf("expected 0 chunks after failed compile, got %d", chunks)
		}
	})

	t.Run("multi_rhs_fused_dense_and_concurrent_race_stress", func(t *testing.T) {
		b, err := igobackend.New("constant_arena")
		if err != nil {
			t.Fatalf("New: %+v", err)
		}
		defer b.Finalize()
		ib := b.(*igobackend.Backend)

		w1Buf := makeTestBF16Buffer(t, b, kDim, nDim)
		w2Buf := makeTestBF16Buffer(t, b, kDim, nDim)
		xBuf := makeTestBF16Buffer(t, b, 1, kDim)
		defer func() {
			_ = w1Buf.Finalize()
			_ = w2Buf.Finalize()
			_ = xBuf.Finalize()
		}()

		constData1, err := igobackend.NewConstantDataFromBuffer(ib, w1Buf.(*igobackend.Buffer), false)
		if err != nil {
			t.Fatalf("NewConstantDataFromBuffer(w1): %+v", err)
		}
		constData2, err := igobackend.NewConstantDataFromBuffer(ib, w2Buf.(*igobackend.Buffer), false)
		if err != nil {
			t.Fatalf("NewConstantDataFromBuffer(w2): %+v", err)
		}

		buildMulti := func(name string) (compute.Executable, error) {
			bldr := b.Builder(name).(*igobackend.Builder)
			fn := bldr.MainFn
			x, err := fn.Parameter("x", shapes.Make(dtypes.BFloat16, 1, kDim), nil)
			if err != nil {
				return nil, err
			}
			w1 := fn.NewNode(compute.OpTypeConstant, constData1.Shape())
			w1.Data = constData1
			w2 := fn.NewNode(compute.OpTypeConstant, constData2.Shape())
			w2.Data = constData2
			y1, err := fn.FusedDense(x, w1, nil, compute.DenseConfig{})
			if err != nil {
				return nil, err
			}
			y2, err := fn.FusedDense(x, w2, nil, compute.DenseConfig{})
			if err != nil {
				return nil, err
			}
			if err := fn.Return([]compute.Value{y1, y2}, nil); err != nil {
				return nil, err
			}
			return bldr.Compile()
		}

		// Concurrent compile + execute + finalize stress test to verify lock ordering and race freedom.
		var wg sync.WaitGroup
		const workers = 6
		const iters = 8
		errCh := make(chan error, workers*iters)
		for range workers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range iters {
					exec, err := buildMulti("stress")
					if err != nil {
						errCh <- err
						return
					}
					outs, err := exec.Execute([]compute.Buffer{xBuf}, []bool{false}, 0)
					if err != nil {
						exec.Finalize()
						errCh <- err
						return
					}
					for _, out := range outs {
						_ = out.Finalize()
					}
					exec.Finalize()
				}
			}()
		}
		wg.Wait()
		close(errCh)
		for err := range errCh {
			t.Fatalf("concurrent worker error: %+v", err)
		}

		chunks, mapped, used := ib.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected 0 chunks after stress test, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}
	})

	t.Run("concurrent_cold_same_exec_and_shared_model_execs_with_gc_reclaim", func(t *testing.T) {
		b, err := igobackend.New("constant_arena")
		if err != nil {
			t.Fatalf("New: %+v", err)
		}
		defer b.Finalize()
		ib := b.(*igobackend.Backend)
		// Force donated-buffer GC reclamation on small test tensors so the concurrent
		// debug.FreeOSMemory() and trailing post-concurrent reclaim paths execute deterministically.
		ib.ConstantArenaGCThresholdBytes = 1024

		// Build a 2-layer model with shared donated weights used by BOTH:
		//   - execPrefill (M=4)
		//   - execDecode  (M=1)
		w1Buf := makeTestBF16Buffer(t, b, kDim, nDim)
		w2Buf := makeTestBF16Buffer(t, b, nDim, nDim)
		xPrefill := makeTestBF16Buffer(t, b, 4, kDim)
		xDecode := makeTestBF16Buffer(t, b, 1, kDim)
		defer func() {
			_ = xPrefill.Finalize()
			_ = xDecode.Finalize()
		}()

		constData1, err := igobackend.NewConstantDataFromBuffer(ib, w1Buf.(*igobackend.Buffer), true)
		if err != nil {
			t.Fatalf("NewConstantDataFromBuffer(w1): %+v", err)
		}
		constData2, err := igobackend.NewConstantDataFromBuffer(ib, w2Buf.(*igobackend.Buffer), true)
		if err != nil {
			t.Fatalf("NewConstantDataFromBuffer(w2): %+v", err)
		}

		buildTwoLayer := func(name string, mDim int) compute.Executable {
			bldr := b.Builder(name).(*igobackend.Builder)
			fn := bldr.MainFn
			x, err := fn.Parameter("x", shapes.Make(dtypes.BFloat16, mDim, kDim), nil)
			if err != nil {
				t.Fatalf("Parameter: %+v", err)
			}
			w1 := fn.NewNode(compute.OpTypeConstant, constData1.Shape())
			w1.Data = constData1
			w2 := fn.NewNode(compute.OpTypeConstant, constData2.Shape())
			w2.Data = constData2
			h, err := fn.FusedDense(x, w1, nil, compute.DenseConfig{})
			if err != nil {
				t.Fatalf("FusedDense 1: %+v", err)
			}
			y, err := fn.FusedDense(h, w2, nil, compute.DenseConfig{})
			if err != nil {
				t.Fatalf("FusedDense 2: %+v", err)
			}
			if err := fn.Return([]compute.Value{y}, nil); err != nil {
				t.Fatalf("Return: %+v", err)
			}
			exec, err := bldr.Compile()
			if err != nil {
				t.Fatalf("Compile(%s): %+v", name, err)
			}
			return exec
		}

		execPrefill := buildTwoLayer("prefill_shared", 4)
		execDecode := buildTwoLayer("decode_shared", 1)

		// Launch concurrent cold calls to BOTH the same Executable (multiple workers on execPrefill
		// and multiple workers on execDecode) AND across the two different Executables sharing the
		// same donated constants simultaneously.
		const workersPerExec = 4
		const iters = 4
		startGate := make(chan struct{})
		prefillResults := make([][]float32, workersPerExec)
		decodeResults := make([][]float32, workersPerExec)
		errCh := make(chan error, 2*workersPerExec*iters)

		var wg sync.WaitGroup
		for w := range workersPerExec {
			wg.Add(2)
			go func(idx int) {
				defer wg.Done()
				<-startGate
				for iter := range iters {
					outs, err := execPrefill.Execute([]compute.Buffer{xPrefill}, []bool{false}, 0)
					if err != nil {
						errCh <- err
						return
					}
					if iter == 0 {
						shape, err := outs[0].Shape()
						if err != nil {
							_ = outs[0].Finalize()
							errCh <- err
							return
						}
						raw := make([]bfloat16.BFloat16, shape.Size())
						if err := outs[0].ToFlatData(raw); err != nil {
							_ = outs[0].Finalize()
							errCh <- err
							return
						}
						res := make([]float32, len(raw))
						for i, v := range raw {
							res[i] = v.Float32()
						}
						prefillResults[idx] = res
					}
					_ = outs[0].Finalize()
				}
			}(w)
			go func(idx int) {
				defer wg.Done()
				<-startGate
				for iter := range iters {
					outs, err := execDecode.Execute([]compute.Buffer{xDecode}, []bool{false}, 0)
					if err != nil {
						errCh <- err
						return
					}
					if iter == 0 {
						shape, err := outs[0].Shape()
						if err != nil {
							_ = outs[0].Finalize()
							errCh <- err
							return
						}
						raw := make([]bfloat16.BFloat16, shape.Size())
						if err := outs[0].ToFlatData(raw); err != nil {
							_ = outs[0].Finalize()
							errCh <- err
							return
						}
						res := make([]float32, len(raw))
						for i, v := range raw {
							res[i] = v.Float32()
						}
						decodeResults[idx] = res
					}
					_ = outs[0].Finalize()
				}
			}(w)
		}
		close(startGate)
		wg.Wait()
		close(errCh)
		for err := range errCh {
			t.Fatalf("concurrent cold execution failed: %+v", err)
		}

		// Verify all concurrent workers got identical results and donated raw buffers were released.
		for w := 1; w < workersPerExec; w++ {
			assertF32Close(t, prefillResults[w], prefillResults[0], 1e-4)
			assertF32Close(t, decodeResults[w], decodeResults[0], 1e-4)
		}
		if constData1.Buffer() != nil || constData2.Buffer() != nil {
			t.Fatalf("expected donated raw buffers to be released after concurrent cold execution")
		}
		chunks, _, _ := ib.ConstantArenaStats()
		if chunks < 1 {
			t.Fatalf("expected >= 1 mapped constant_arena chunk, got %d", chunks)
		}

		// Finalize execPrefill while concurrently executing execDecode — shared arena must remain valid.
		var decodeWG sync.WaitGroup
		decodeErrCh := make(chan error, 8)
		for range 4 {
			decodeWG.Add(1)
			go func() {
				defer decodeWG.Done()
				for range 4 {
					outs, err := execDecode.Execute([]compute.Buffer{xDecode}, []bool{false}, 0)
					if err != nil {
						decodeErrCh <- err
						return
					}
					_ = outs[0].Finalize()
				}
			}()
		}
		execPrefill.Finalize()
		decodeWG.Wait()
		close(decodeErrCh)
		for err := range decodeErrCh {
			t.Fatalf("execDecode failed while execPrefill finalized: %+v", err)
		}

		// Finalize execDecode -> all users gone -> arena chunks unmapped.
		execDecode.Finalize()
		chunks, mapped, used := ib.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected 0 chunks after both executables finalized, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}
	})
}

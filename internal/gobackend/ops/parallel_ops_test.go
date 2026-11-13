// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
)

func execSingleGraph(t *testing.T, buildFn func(f compute.Function) (compute.Value, error), inputs ...compute.Buffer) *gobackend.Buffer {
	t.Helper()
	builder := backend.Builder("test_parallel_ops")
	mainFn := builder.Main()
	outVal, err := buildFn(mainFn)
	if err != nil {
		t.Fatalf("buildFn failed: %+v", err)
	}
	if err := mainFn.Return([]compute.Value{outVal}, nil); err != nil {
		t.Fatalf("Return failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %+v", err)
	}
	defer exec.Finalize()
	donate := make([]bool, len(inputs))
	outs, err := exec.Execute(inputs, donate, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}
	return outs[0].(*gobackend.Buffer)
}

func TestParallelBroadcastInDim(t *testing.T) {
	t.Run("Scalar", func(t *testing.T) {
		inBuf := makeBuffer(t, shapes.Make(dtypes.Float32), []float32{42.5})
		defer inBuf.Finalize()
		outShape := shapes.Make(dtypes.Float32, 64, 512) // 32768 > 16384
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.BroadcastInDim(p, outShape, nil)
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]float32)
		for i, v := range got {
			if v != 42.5 {
				t.Fatalf("at %d: got %v, want 42.5", i, v)
			}
		}
	})

	t.Run("Prefix", func(t *testing.T) {
		const outer, k = 128, 384 // 49152 > 16384
		src := make([]float32, k)
		for i := range src {
			src[i] = float32(i*3 - 100)
		}
		inBuf := makeBuffer(t, shapes.Make(dtypes.Float32, 1, k), src)
		defer inBuf.Finalize()
		outShape := shapes.Make(dtypes.Float32, outer, k)
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.BroadcastInDim(p, outShape, []int{0, 1})
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]float32)
		for o := range outer {
			for i := range k {
				if got[o*k+i] != src[i] {
					t.Fatalf("at (%d,%d): got %v, want %v", o, i, got[o*k+i], src[i])
				}
			}
		}
	})

	t.Run("Suffix", func(t *testing.T) {
		const outer, inner = 256, 256 // 65536 > 16384
		src := make([]bool, outer)
		for i := range src {
			src[i] = (i%3 == 0)
		}
		inBuf := makeBuffer(t, shapes.Make(dtypes.Bool, outer), src)
		defer inBuf.Finalize()
		outShape := shapes.Make(dtypes.Bool, outer, inner)
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.BroadcastInDim(p, outShape, []int{0})
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]bool)
		for o := range outer {
			for i := range inner {
				if got[o*inner+i] != src[o] {
					t.Fatalf("at (%d,%d): got %v, want %v", o, i, got[o*inner+i], src[o])
				}
			}
		}
	})

	t.Run("MiddleSmallOuter", func(t *testing.T) {
		const outer, mid, inner = 4, 64, 128 // 32768 > 16384, outer < workers
		src := make([]float32, outer*inner)
		for i := range src {
			src[i] = float32(i + 1)
		}
		inBuf := makeBuffer(t, shapes.Make(dtypes.Float32, outer, 1, inner), src)
		defer inBuf.Finalize()
		outShape := shapes.Make(dtypes.Float32, outer, mid, inner)
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.BroadcastInDim(p, outShape, []int{0, 1, 2})
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]float32)
		for o := range outer {
			for m := range mid {
				for i := range inner {
					want := src[o*inner+i]
					if got[(o*mid+m)*inner+i] != want {
						t.Fatalf("at (%d,%d,%d): got %v, want %v", o, m, i, got[(o*mid+m)*inner+i], want)
					}
				}
			}
		}
	})

	t.Run("OuterAndInner", func(t *testing.T) {
		const outer, mid, inner = 16, 64, 32 // 32768 > 16384
		src := make([]int32, mid)
		for i := range src {
			src[i] = int32(i*7 - 50)
		}
		inBuf := makeBuffer(t, shapes.Make(dtypes.Int32, 1, mid, 1), src)
		defer inBuf.Finalize()
		outShape := shapes.Make(dtypes.Int32, outer, mid, inner)
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.BroadcastInDim(p, outShape, []int{0, 1, 2})
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]int32)
		for o := range outer {
			for m := range mid {
				for i := range inner {
					if got[(o*mid+m)*inner+i] != src[m] {
						t.Fatalf("at (%d,%d,%d): got %v, want %v", o, m, i, got[(o*mid+m)*inner+i], src[m])
					}
				}
			}
		}
	})

	t.Run("FourRunsFallback", func(t *testing.T) {
		// [2, 1, 3, 1] -> [2, 4, 3, 5] (4 alternating runs -> iterator fallback)
		src := []float32{1, 2, 3, 4, 5, 6}
		inBuf := makeBuffer(t, shapes.Make(dtypes.Float32, 2, 1, 3, 1), src)
		defer inBuf.Finalize()
		outShape := shapes.Make(dtypes.Float32, 2, 4, 3, 5)
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.BroadcastInDim(p, outShape, []int{0, 1, 2, 3})
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]float32)
		for a := range 2 {
			for b := range 4 {
				for c := range 3 {
					for d := range 5 {
						want := src[a*3+c]
						idx := ((a*4+b)*3+c)*5 + d
						if got[idx] != want {
							t.Fatalf("at (%d,%d,%d,%d): got %v, want %v", a, b, c, d, got[idx], want)
						}
					}
				}
			}
		}
	})
}

func TestParallelConvertDTypeAndGather(t *testing.T) {
	t.Run("ConvertBoolToFloat32Large", func(t *testing.T) {
		const n = 33001 // unaligned > 16384
		src := make([]bool, n)
		for i := range src {
			switch {
			case i < 1000:
				src[i] = true
			case i < 2000:
				src[i] = false
			default:
				src[i] = (i%5 == 0) || (i%7 == 0)
			}
		}
		inBuf := makeBuffer(t, shapes.Make(dtypes.Bool, n), src)
		defer inBuf.Finalize()
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.ConvertDType(p, dtypes.Float32)
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]float32)
		for i, b := range src {
			want := float32(0)
			if b {
				want = 1
			}
			if got[i] != want {
				t.Fatalf("at %d: got %v, want %v", i, got[i], want)
			}
		}
	})

	t.Run("ConvertPackedInt4ToFloat32Large", func(t *testing.T) {
		const n = 32770 // even, > 16384, unaligned to 64
		packed := make([]byte, n/2)
		for i := range packed {
			packed[i] = byte((i * 37) & 0xFF)
		}
		inBuf := makeBuffer(t, shapes.Make(dtypes.Int4, n), packed)
		defer inBuf.Finalize()
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			p, err := f.Parameter("x", inBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.ConvertDType(p, dtypes.Float32)
		}, inBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]float32)
		for i := range n {
			b := packed[i/2]
			var nib int8
			if i%2 == 0 {
				nib = int8(b<<4) >> 4
			} else {
				nib = int8(b) >> 4
			}
			if got[i] != float32(nib) {
				t.Fatalf("at %d: got %v, want %v", i, got[i], float32(nib))
			}
		}
	})

	t.Run("GatherContiguousBatchLarge", func(t *testing.T) {
		const vocab, dim, batch = 128, 384, 64 // output bytes = 64 * 384 * 4 = 98304 > 16384
		table := make([]float32, vocab*dim)
		for i := range table {
			table[i] = float32(i) * 0.25
		}
		indices := make([]int32, batch)
		for b := range batch {
			switch b {
			case 0:
				indices[b] = -5 // clamps to 0
			case 1:
				indices[b] = vocab + 10 // clamps to vocab-1
			default:
				indices[b] = int32((b * 13) % vocab)
			}
		}
		opBuf := makeBuffer(t, shapes.Make(dtypes.Float32, vocab, dim), table)
		defer opBuf.Finalize()
		idxBuf := makeBuffer(t, shapes.Make(dtypes.Int32, batch, 1), indices)
		defer idxBuf.Finalize()
		outBuf := execSingleGraph(t, func(f compute.Function) (compute.Value, error) {
			pOp, err := f.Parameter("op", opBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			pIdx, err := f.Parameter("idx", idxBuf.RawShape, nil)
			if err != nil {
				return nil, err
			}
			return f.Gather(pOp, pIdx, 1, []int{1}, []int{0}, []int{0}, []int{1, dim}, false)
		}, opBuf, idxBuf)
		defer outBuf.Finalize()
		got := outBuf.Flat.([]float32)
		for b := range batch {
			clamped := max(0, min(vocab-1, int(indices[b])))
			for d := range dim {
				want := table[clamped*dim+d]
				if got[b*dim+d] != want {
					t.Fatalf("at (%d,%d): got %v, want %v", b, d, got[b*dim+d], want)
				}
			}
		}
	})
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	_ "github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
)

func TestReshapeZeroCopyViewSafety(t *testing.T) {
	b, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Finalize()

	t.Run("NonOwnedInputViewAndInPlaceConsumer", func(t *testing.T) {
		// Graph: y = Add(Reshape(x, [6]), Reshape(x, [6])) + also returns x reshaped directly as output 1.
		// Verifies:
		//  1. Reshape of non-owned input creates zero-copy view without mutating x.
		//  2. Downstream Add does not mutate the view or x in place.
		//  3. Escaping view output is cloned independently so Finalize does not corrupt x.
		builder := b.Builder("reshape_view_safety")
		f := builder.Main()
		p, err := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
		if err != nil {
			t.Fatal(err)
		}
		r1, err := f.Reshape(p, 6)
		if err != nil {
			t.Fatal(err)
		}
		r2, err := f.Reshape(r1, 3, 2)
		if err != nil {
			t.Fatal(err)
		}
		r3, err := f.Reshape(r2, 6)
		if err != nil {
			t.Fatal(err)
		}
		sum, err := f.Add(r1, r3)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Return([]compute.Value{sum, r2}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		src := []float32{1, 2, 3, 4, 5, 6}
		inBuf, err := b.BufferFromFlatData(0, src, shapes.Make(dtypes.Float32, 2, 3))
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()

		for range 4 {
			outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
			if err != nil {
				t.Fatal(err)
			}
			gotSum := make([]float32, 6)
			if err := outs[0].ToFlatData(gotSum); err != nil {
				t.Fatal(err)
			}
			gotView := make([]float32, 6)
			if err := outs[1].ToFlatData(gotView); err != nil {
				t.Fatal(err)
			}
			outs[0].Finalize()
			outs[1].Finalize()

			for i := range 6 {
				if gotSum[i] != 2*src[i] {
					t.Fatalf("sum[%d] = %v, want %v", i, gotSum[i], 2*src[i])
				}
				if gotView[i] != src[i] {
					t.Fatalf("view[%d] = %v, want %v", i, gotView[i], src[i])
				}
			}
			// Verify original input buffer was never mutated or prematurely freed.
			gotIn := make([]float32, 6)
			if err := inBuf.ToFlatData(gotIn); err != nil {
				t.Fatal(err)
			}
			for i := range 6 {
				if gotIn[i] != src[i] {
					t.Fatalf("input mutated at %d: got %v, want %v", i, gotIn[i], src[i])
				}
			}
		}
	})

	t.Run("MultiConsumerIntermediateViewLifetime", func(t *testing.T) {
		// Intermediate z = Mul(x, x) has 2 consumers: Reshape(z) and Add(z, one).
		// Even if Add(z, one) finishes first (or runs in parallel), z must stay pinned
		// until Reshape(z)'s consumer finishes.
		builder := b.Builder("reshape_intermediate_pin")
		f := builder.Main()
		p, err := f.Parameter("x", shapes.Make(dtypes.Float32, 1, 1, 256), nil)
		if err != nil {
			t.Fatal(err)
		}
		one, err := f.Constant([]float32{1})
		if err != nil {
			t.Fatal(err)
		}
		z, err := f.Mul(p, p)
		if err != nil {
			t.Fatal(err)
		}
		zr, err := f.Reshape(z, 256)
		if err != nil {
			t.Fatal(err)
		}
		zPlus1, err := f.Add(z, one)
		if err != nil {
			t.Fatal(err)
		}
		zPlus1Flat, err := f.Reshape(zPlus1, 256)
		if err != nil {
			t.Fatal(err)
		}
		out, err := f.Add(zr, zPlus1Flat)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Return([]compute.Value{out}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		src := make([]float32, 256)
		for i := range src {
			src[i] = float32(i%13) - 6
		}
		inBuf, err := b.BufferFromFlatData(0, src, shapes.Make(dtypes.Float32, 1, 1, 256))
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()

		for range 8 {
			outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, 256)
			if err := outs[0].ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			outs[0].Finalize()
			for i, v := range src {
				want := 2*v*v + 1
				if got[i] != want {
					t.Fatalf("out[%d] = %v, want %v", i, got[i], want)
				}
			}
		}
	})
}

func TestArgMinMaxFastSuffix1(t *testing.T) {
	b, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Finalize()

	const n = 262144 // Gemma 4 vocab size (triggers parallel chunking)

	t.Run("BFloat16Large", func(t *testing.T) {
		data := make([]bfloat16.BFloat16, n)
		for i := range data {
			data[i] = bfloat16.FromFloat32(float32((i%997)-500) * 0.125)
		}
		// Tie-breaking check: two identical maxima -> first index must win.
		data[131071] = bfloat16.FromFloat32(999.0)
		data[200000] = bfloat16.FromFloat32(999.0)
		// Two identical minima -> first index must win.
		data[50000] = bfloat16.FromFloat32(-999.0)
		data[250000] = bfloat16.FromFloat32(-999.0)

		for _, tc := range []struct {
			isMin bool
			want  int32
		}{
			{false, 131071},
			{true, 50000},
		} {
			builder := b.Builder("argmax_bf16")
			f := builder.Main()
			p, _ := f.Parameter("x", shapes.Make(dtypes.BFloat16, 1, n), nil)
			res, err := f.ArgMinMax(p, 1, dtypes.Int32, tc.isMin)
			if err != nil {
				t.Fatal(err)
			}
			_ = f.Return([]compute.Value{res}, nil)
			exec, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			inBuf, err := b.BufferFromFlatData(0, data, shapes.Make(dtypes.BFloat16, 1, n))
			if err != nil {
				t.Fatal(err)
			}
			outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]int32, 1)
			_ = outs[0].ToFlatData(got)
			outs[0].Finalize()
			inBuf.Finalize()
			exec.Finalize()
			if got[0] != tc.want {
				t.Fatalf("BF16 isMin=%v: got %d, want %d", tc.isMin, got[0], tc.want)
			}
		}
	})

	t.Run("Float16Large", func(t *testing.T) {
		data := make([]float16.Float16, n)
		for i := range data {
			data[i] = float16.FromFloat32(float32((i%251)-125) * 0.25)
		}
		data[190000] = float16.FromFloat32(1024.0)
		builder := b.Builder("argmax_f16")
		f := builder.Main()
		p, _ := f.Parameter("x", shapes.Make(dtypes.Float16, 1, n), nil)
		res, _ := f.ArgMinMax(p, 1, dtypes.Int32, false)
		_ = f.Return([]compute.Value{res}, nil)
		exec, _ := builder.Compile()
		defer exec.Finalize()
		inBuf, _ := b.BufferFromFlatData(0, data, shapes.Make(dtypes.Float16, 1, n))
		defer inBuf.Finalize()
		outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outs[0].Finalize()
		got := make([]int32, 1)
		_ = outs[0].ToFlatData(got)
		if got[0] != 190000 {
			t.Fatalf("F16 argmax: got %d, want 190000", got[0])
		}
	})

	t.Run("Float32LargeWithNaN", func(t *testing.T) {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i % 100)
		}
		data[77777] = float32(math.NaN())
		data[177777] = float32(math.NaN())
		builder := b.Builder("argmax_f32_nan")
		f := builder.Main()
		p, _ := f.Parameter("x", shapes.Make(dtypes.Float32, 1, n), nil)
		res, _ := f.ArgMinMax(p, 1, dtypes.Int32, false)
		_ = f.Return([]compute.Value{res}, nil)
		exec, _ := builder.Compile()
		defer exec.Finalize()
		inBuf, _ := b.BufferFromFlatData(0, data, shapes.Make(dtypes.Float32, 1, n))
		defer inBuf.Finalize()
		outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outs[0].Finalize()
		got := make([]int32, 1)
		_ = outs[0].ToFlatData(got)
		// Last NaN wins (matching sequential loop semantics).
		if got[0] != 177777 {
			t.Fatalf("F32 NaN argmax: got %d, want 177777", got[0])
		}
	})
}

func BenchmarkArgMaxBFloat16Vocab262144(b *testing.B) {
	backend, err := gobackend.NewBackend()
	if err != nil {
		b.Fatal(err)
	}
	defer backend.Finalize()
	const n = 262144
	data := make([]bfloat16.BFloat16, n)
	for i := range data {
		data[i] = bfloat16.FromFloat32(float32(i % 1000))
	}
	builder := backend.Builder("bench_argmax_bf16")
	f := builder.Main()
	p, _ := f.Parameter("x", shapes.Make(dtypes.BFloat16, 1, n), nil)
	res, _ := f.ArgMinMax(p, 1, dtypes.Int32, false)
	_ = f.Return([]compute.Value{res}, nil)
	exec, err := builder.Compile()
	if err != nil {
		b.Fatal(err)
	}
	defer exec.Finalize()
	inBuf, err := backend.BufferFromFlatData(0, data, shapes.Make(dtypes.BFloat16, 1, n))
	if err != nil {
		b.Fatal(err)
	}
	defer inBuf.Finalize()
	b.ResetTimer()
	for range b.N {
		outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
		if err != nil {
			b.Fatal(err)
		}
		outs[0].Finalize()
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend_test

import (
	"slices"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	igobackend "github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
)

func TestZeroSizeDCEAndShapeDependency(t *testing.T) {
	backend, err := gobackend.New("")
	if err != nil {
		t.Fatalf("gobackend.New failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("StaticZeroOutputFoldsToConstantAndEliminatesInputs", func(t *testing.T) {
		builder := backend.Builder("static_zero_dce")
		fn := builder.Main()

		p, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 1, 4), nil)
		if err != nil {
			t.Fatal(err)
		}
		// Build a chain of ops whose result is consumed only by a [1, 4] + [0, 4] -> [0, 4] Add.
		h1, err := fn.Exp(p)
		if err != nil {
			t.Fatal(err)
		}
		h2, err := fn.Tanh(h1)
		if err != nil {
			t.Fatal(err)
		}
		empty, err := fn.Constant([]float32{}, 0, 4)
		if err != nil {
			t.Fatal(err)
		}
		out, err := fn.Add(h2, empty)
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{out}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		rawFn := fn.(*igobackend.Function)
		fe := rawFn.Compiled
		h1Node := h1.(*igobackend.Node)
		h2Node := h2.(*igobackend.Node)
		outNode := out.(*igobackend.Node)

		if outNode.OpType != compute.OpTypeConstant {
			t.Fatalf("expected static zero-sized output node to fold to OpTypeConstant, got %s", outNode.OpType)
		}
		if fe.NumUses[h1Node.Index] != 0 || fe.NumUses[h2Node.Index] != 0 {
			t.Fatalf("expected upstream nodes h1/h2 to be dead-code-eliminated (NumUses=0), got h1=%d, h2=%d",
				fe.NumUses[h1Node.Index], fe.NumUses[h2Node.Index])
		}

		inBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapes.Make(dtypes.Float32, 1, 4))
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()
		outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		defer outs[0].Finalize()
		sh, _ := outs[0].Shape()
		if !sh.Equal(shapes.Make(dtypes.Float32, 0, 4)) {
			t.Fatalf("got output shape %s, want [0, 4]", sh)
		}
	})

	t.Run("DynamicZeroOutputRewiresToDefiningParameterAndEliminatesChain", func(t *testing.T) {
		builder := backend.Builder("dynamic_zero_shape_dep_dce")
		fn := builder.Main()

		x, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4}, []string{"batch", ""}), nil)
		if err != nil {
			t.Fatal(err)
		}
		// Intermediate compute chain on dynamic tensor x:
		h1, err := fn.Exp(x)
		if err != nil {
			t.Fatal(err)
		}
		h2, err := fn.Tanh(h1)
		if err != nil {
			t.Fatal(err)
		}
		// Slice to ["batch", 1], then broadcast-add with [1, 0] -> ["batch", 0], followed by Transpose -> [0, "batch"].
		h3, err := fn.ReduceSum(h2, 1)
		if err != nil {
			t.Fatal(err)
		}
		h4, err := fn.BroadcastInDim(h3, shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 1}, []string{"batch", ""}), []int{0})
		if err != nil {
			t.Fatal(err)
		}
		empty, err := fn.Constant([]float32{}, 1, 0)
		if err != nil {
			t.Fatal(err)
		}
		z1, err := fn.Add(h4, empty)
		if err != nil {
			t.Fatal(err)
		}
		z2, err := fn.Transpose(z1, 1, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{z2}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		rawFn := fn.(*igobackend.Function)
		fe := rawFn.Compiled
		xNode := x.(*igobackend.Node)
		z2Node := z2.(*igobackend.Node)

		if !z2Node.IsZeroSizeShapeOnly {
			t.Fatalf("expected z2 to be marked IsZeroSizeShapeOnly")
		}
		if len(z2Node.Inputs) != 1 || z2Node.Inputs[0] != xNode {
			t.Fatalf("expected z2.Inputs to point directly to defining parameter x, got %v", z2Node.Inputs)
		}
		for name, v := range map[string]compute.Value{"h1": h1, "h2": h2, "h3": h3, "h4": h4, "z1": z1} {
			n := v.(*igobackend.Node)
			if fe.NumUses[n.Index] != 0 {
				t.Fatalf("expected intermediate node %s (idx %d) to be DCE'd (NumUses=0), got %d", name, n.Index, fe.NumUses[n.Index])
			}
		}

		for _, batch := range []int{6, 0, 3} {
			inBuf, err := backend.BufferFromFlatData(0, make([]float32, batch*4), shapes.Make(dtypes.Float32, batch, 4))
			if err != nil {
				t.Fatal(err)
			}
			outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
			inBuf.Finalize()
			if err != nil {
				t.Fatalf("Execute(batch=%d) failed: %+v", batch, err)
			}
			sh, _ := outs[0].Shape()
			outs[0].Finalize()
			if !slices.Equal(sh.Dimensions, []int{0, batch}) {
				t.Fatalf("batch=%d: got shape %v, want [0, %d]", batch, sh.Dimensions, batch)
			}
		}
	})

	t.Run("SymbolicSumAxisZeroOutputRewiresToParameter", func(t *testing.T) {
		builder := backend.Builder("symbolic_sum_zero_dce")
		fn := builder.Main()

		x, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4}, []string{"batch", ""}), nil)
		if err != nil {
			t.Fatal(err)
		}
		twoRows, err := fn.Constant(make([]float32, 2*4), 2, 4)
		if err != nil {
			t.Fatal(err)
		}
		// Concatenate on axis 0 -> shape ["=batch+2", 4], then slice axis 1 to 0..0 -> ["=batch+2", 0], then Exp.
		concat, err := fn.Concatenate(0, x, twoRows)
		if err != nil {
			t.Fatal(err)
		}
		slicedEmpty, err := fn.Slice(concat, []int{0, 0}, []int{shapes.DynamicDim, 0}, []int{1, 1})
		if err != nil {
			t.Fatal(err)
		}
		out, err := fn.Exp(slicedEmpty)
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{out}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		rawFn := fn.(*igobackend.Function)
		fe := rawFn.Compiled
		concatNode := concat.(*igobackend.Node)
		slicedNode := slicedEmpty.(*igobackend.Node)
		if fe.NumUses[concatNode.Index] != 0 || fe.NumUses[slicedNode.Index] != 0 {
			t.Fatalf("expected Concatenate and Slice nodes to be DCE'd when symbolic axis '=batch+2' rewires to parameter x, got concat=%d, slice=%d",
				fe.NumUses[concatNode.Index], fe.NumUses[slicedNode.Index])
		}

		inBuf, err := backend.BufferFromFlatData(0, make([]float32, 3*4), shapes.Make(dtypes.Float32, 3, 4))
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()
		outs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		defer outs[0].Finalize()
		sh, _ := outs[0].Shape()
		if !slices.Equal(sh.Dimensions, []int{5, 0}) {
			t.Fatalf("got shape %v, want [5, 0]", sh.Dimensions)
		}
	})

	t.Run("ZeroSizeBufferPoolZeroAllocs", func(t *testing.T) {
		rawBackend := backend.(*igobackend.Backend)
		zeroShape := shapes.Make(dtypes.Float32, 0, 4)
		// Warm up pool slot.
		buf, err := rawBackend.GetBuffer(zeroShape)
		if err != nil {
			t.Fatal(err)
		}
		rawBackend.PutBuffer(buf)

		allocs := testing.AllocsPerRun(100, func() {
			b, err := rawBackend.GetBuffer(zeroShape)
			if err != nil {
				t.Fatal(err)
			}
			rawBackend.PutBuffer(b)
		})
		if allocs != 0 {
			t.Fatalf("expected 0 allocs/op for pooled zero-size GetBuffer/PutBuffer, got %v", allocs)
		}
	})

	t.Run("DistinctGeneratedAxesDoNotImplicitlyBroadcastOrFoldAwayValidation", func(t *testing.T) {
		// Distinct generated "#" axes in BinaryOp and zero-sized BinaryOp must still validate equality at runtime
		// (a runtime dimension of 1 must NOT implicitly broadcast to 3 or 0).
		builder := backend.Builder("distinct_gen_axes_no_runtime_bcast")
		fn := builder.Main()
		a, err := fn.Parameter("a", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 0}, []string{shapes.UniqueAxisName("a"), ""}), nil)
		if err != nil {
			t.Fatal(err)
		}
		bParam, err := fn.Parameter("b", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 0}, []string{shapes.UniqueAxisName("b"), ""}), nil)
		if err != nil {
			t.Fatal(err)
		}
		sum, err := fn.Add(a, bParam)
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{sum}, nil); err != nil {
			t.Fatal(err)
		}
		ex, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer ex.Finalize()

		b3, _ := backend.BufferFromFlatData(0, []float32{}, shapes.Make(dtypes.Float32, 3, 0))
		defer b3.Finalize()
		b1, _ := backend.BufferFromFlatData(0, []float32{}, shapes.Make(dtypes.Float32, 1, 0))
		defer b1.Finalize()
		b0, _ := backend.BufferFromFlatData(0, []float32{}, shapes.Make(dtypes.Float32, 0, 0))
		defer b0.Finalize()

		if _, err := ex.Execute([]compute.Buffer{b3, b1}, nil, 0); err == nil {
			t.Fatal("expected runtime error for Add([3, 0], [1, 0]) on distinct generated axes")
		}
		if _, err := ex.Execute([]compute.Buffer{b1, b0}, nil, 0); err == nil {
			t.Fatal("expected runtime error for Add([1, 0], [0, 0]) on distinct generated axes")
		}

		// Matching concrete dimensions succeed.
		outs, err := ex.Execute([]compute.Buffer{b3, b3}, nil, 0)
		if err != nil {
			t.Fatalf("expected Add([3, 0], [3, 0]) to succeed, got: %+v", err)
		}
		sh, _ := outs[0].Shape()
		outs[0].Finalize()
		if !sh.Equal(shapes.Make(dtypes.Float32, 3, 0)) {
			t.Fatalf("got shape %s, want [3, 0]", sh)
		}
	})
}

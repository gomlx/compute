// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestFunctions(t *testing.T, b compute.Backend) {
	testutil.SkipIfMissingFunctions(t, b)
	t.Run("Capabilities", func(t *testing.T) {
		caps := b.Capabilities()
		if !caps.Functions {
			t.Errorf("Backend should support Functions capability")
		}
	})

	t.Run("ClosureCreation", func(t *testing.T) {
		builder := b.Builder("test_closure_creation")
		mainFn := builder.Main()
		if mainFn == nil {
			t.Fatalf("mainFn is nil")
		}

		// Create a closure from the main function
		closure, err := mainFn.Closure()
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if closure == nil {
			t.Fatalf("closure is nil")
		}

		// Verify closure properties
		if closure.Parent() != mainFn {
			t.Errorf("Closure parent mismatch")
		}
	})

	t.Run("NestedClosures", func(t *testing.T) {
		builder := b.Builder("test_nested_closures")
		mainFn := builder.Main()

		// Create first level closure
		closure1, err := mainFn.Closure()
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if closure1 == nil {
			t.Fatalf("closure1 is nil")
		}
		if closure1.Parent() != mainFn {
			t.Errorf("closure1 parent mismatch")
		}

		// Create second level closure
		closure2, err := closure1.Closure()
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if closure2 == nil {
			t.Fatalf("closure2 is nil")
		}
		if closure2.Parent() != closure1 {
			t.Errorf("closure2 parent mismatch")
		}
	})

	t.Run("NamedFunctionCreation", func(t *testing.T) {
		builder := b.Builder("test_named_function")

		// Create a named function
		fn, err := builder.NewFunction("my_function")
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if fn == nil {
			t.Fatalf("fn is nil")
		}

		if fn.Name() != "my_function" {
			t.Errorf("Expected function name %q, got %q", "my_function", fn.Name())
		}
	})

	t.Run("Call", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeCall)

		t.Run("SingleCall", func(t *testing.T) {
			builder := b.Builder("test_call_single")
			subFn, err := builder.NewFunction("scale_and_shift")
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("NewFunction not implemented: %v", err)
				}
				t.Fatalf("NewFunction failed: %+v", err)
			}
			sx, err := subFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
			if err != nil {
				t.Fatalf("subFn.Parameter failed: %+v", err)
			}
			scale, _ := subFn.Constant([]float32{2, 2, 2}, 3)
			shift, _ := subFn.Constant([]float32{1, 2, 3}, 3)
			prod, _ := subFn.Mul(sx, scale)
			out, _ := subFn.Add(prod, shift)
			if err := subFn.Return([]compute.Value{out}, nil); err != nil {
				t.Fatalf("subFn.Return failed: %+v", err)
			}

			mainFn := builder.Main()
			p0, _ := mainFn.Parameter("p0", shapes.Make(dtypes.Float32, 3), nil)
			res0, err := mainFn.Call(subFn, p0)
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Call not implemented: %v", err)
				}
				t.Fatalf("mainFn.Call failed: %+v", err)
			}
			_ = mainFn.Return(res0, nil)

			exec, err := builder.Compile()
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Compile with Call not implemented: %v", err)
				}
				t.Fatalf("Compile failed: %+v", err)
			}
			defer exec.Finalize()

			b0, _ := testutil.ToBuffer(b, []float32{1, 2, 3}) // 2*[1,2,3]+[1,2,3] = [3,6,9]
			defer b0.Finalize()

			outputs, err := exec.Execute([]compute.Buffer{b0}, nil, 0)
			if err != nil {
				t.Fatalf("Execute failed: %+v", err)
			}
			defer outputs[0].Finalize()
			got, _ := testutil.FromBuffer(b, outputs[0])
			if ok, diff := testutil.IsEqual([]float32{3, 6, 9}, got); !ok {
				t.Errorf("SingleCall mismatch:\n%s", diff)
			}
		})

		t.Run("MultipleCalls", func(t *testing.T) {
			builder := b.Builder("test_call_multi")
			subFn, err := builder.NewFunction("scale_and_shift")
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("NewFunction not implemented: %v", err)
				}
				t.Fatalf("NewFunction failed: %+v", err)
			}
			sx, err := subFn.Parameter("x", shapes.Make(dtypes.Float32, 3), nil)
			if err != nil {
				t.Fatalf("subFn.Parameter failed: %+v", err)
			}
			scale, _ := subFn.Constant([]float32{2, 2, 2}, 3)
			shift, _ := subFn.Constant([]float32{1, 2, 3}, 3)
			prod, _ := subFn.Mul(sx, scale)
			out, _ := subFn.Add(prod, shift)
			if err := subFn.Return([]compute.Value{out}, nil); err != nil {
				t.Fatalf("subFn.Return failed: %+v", err)
			}

			mainFn := builder.Main()
			p0, _ := mainFn.Parameter("p0", shapes.Make(dtypes.Float32, 3), nil)
			p1, _ := mainFn.Parameter("p1", shapes.Make(dtypes.Float32, 3), nil)
			res0, err := mainFn.Call(subFn, p0)
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Call not implemented: %v", err)
				}
				t.Fatalf("mainFn.Call #1 failed: %+v", err)
			}
			res1, err := mainFn.Call(subFn, p1)
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Multiple Call not implemented: %v", err)
				}
				t.Fatalf("mainFn.Call #2 failed: %+v", err)
			}
			sum, err := mainFn.Add(res0[0], res1[0])
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Post-Call op not implemented: %v", err)
				}
				t.Fatalf("mainFn.Add after Call failed: %+v", err)
			}
			_ = mainFn.Return([]compute.Value{sum}, nil)

			exec, err := builder.Compile()
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Compile with Call not implemented: %v", err)
				}
				t.Fatalf("Compile failed: %+v", err)
			}
			defer exec.Finalize()

			b0, _ := testutil.ToBuffer(b, []float32{1, 2, 3})    // 2*[1,2,3]+[1,2,3] = [3,6,9]
			b1, _ := testutil.ToBuffer(b, []float32{10, 20, 30}) // 2*[10,20,30]+[1,2,3] = [21,42,63]
			defer b0.Finalize()
			defer b1.Finalize()

			outputs, err := exec.Execute([]compute.Buffer{b0, b1}, nil, 0)
			if err != nil {
				t.Fatalf("Execute failed: %+v", err)
			}
			defer outputs[0].Finalize()
			got, _ := testutil.FromBuffer(b, outputs[0])
			if ok, diff := testutil.IsEqual([]float32{24, 48, 72}, got); !ok {
				t.Errorf("MultipleCalls mismatch:\n%s", diff)
			}
		})

		t.Run("MultiOutput", func(t *testing.T) {
			builder := b.Builder("test_call_multi_out")
			subFn, err := builder.NewFunction("add_and_sub")
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("NewFunction not implemented: %v", err)
				}
				t.Fatalf("NewFunction failed: %+v", err)
			}
			sa, _ := subFn.Parameter("a", shapes.Make(dtypes.Float32, 2), nil)
			sb, _ := subFn.Parameter("b", shapes.Make(dtypes.Float32, 2), nil)
			sAdd, _ := subFn.Add(sa, sb)
			sSub, _ := subFn.Sub(sa, sb)
			_ = subFn.Return([]compute.Value{sAdd, sSub}, nil)

			mainFn := builder.Main()
			pa, _ := mainFn.Parameter("a", shapes.Make(dtypes.Float32, 2), nil)
			pb, _ := mainFn.Parameter("b", shapes.Make(dtypes.Float32, 2), nil)
			outs, err := mainFn.Call(subFn, pa, pb)
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Call not implemented: %v", err)
				}
				t.Fatalf("Call failed: %+v", err)
			}
			_ = mainFn.Return(outs, nil)

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile failed: %+v", err)
			}
			defer exec.Finalize()

			ba, _ := testutil.ToBuffer(b, []float32{10, 20})
			bb, _ := testutil.ToBuffer(b, []float32{3, 5})
			defer ba.Finalize()
			defer bb.Finalize()

			res, err := exec.Execute([]compute.Buffer{ba, bb}, nil, 0)
			if err != nil {
				t.Fatalf("Execute failed: %+v", err)
			}
			defer res[0].Finalize()
			defer res[1].Finalize()

			got0, _ := testutil.FromBuffer(b, res[0])
			got1, _ := testutil.FromBuffer(b, res[1])
			if ok, diff := testutil.IsEqual([]float32{13, 25}, got0); !ok {
				t.Errorf("Call output 0 mismatch:\n%s", diff)
			}
			if ok, diff := testutil.IsEqual([]float32{7, 15}, got1); !ok {
				t.Errorf("Call output 1 mismatch:\n%s", diff)
			}
		})
	})

	t.Run("Sort", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeSort)

		t.Run("1D_Ascending", func(t *testing.T) {
			y, err := testutil.Exec1(b, []any{[]float32{5, 2, 8, 1, 3}},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					comp, err := f.Closure()
					if err != nil {
						return nil, err
					}
					lhs, _ := comp.Parameter("lhs", shapes.Make(dtypes.Float32), nil)
					rhs, _ := comp.Parameter("rhs", shapes.Make(dtypes.Float32), nil)
					lt, _ := comp.LessThan(lhs, rhs)
					_ = comp.Return([]compute.Value{lt}, nil)
					outs, err := f.Sort(comp, 0, false, p[0])
					if err != nil {
						return nil, err
					}
					return outs[0], nil
				})
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Sort not implemented: %v", err)
				}
				t.Fatalf("Sort(1D_Ascending) failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([]float32{1, 2, 3, 5, 8}, y); !ok {
				t.Errorf("Sort(1D_Ascending) mismatch:\n%s", diff)
			}
		})

		t.Run("1D_Descending", func(t *testing.T) {
			y, err := testutil.Exec1(b, []any{[]float32{5, 2, 8, 1, 3}},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					comp, err := f.Closure()
					if err != nil {
						return nil, err
					}
					lhs, _ := comp.Parameter("lhs", shapes.Make(dtypes.Float32), nil)
					rhs, _ := comp.Parameter("rhs", shapes.Make(dtypes.Float32), nil)
					gt, _ := comp.GreaterThan(lhs, rhs)
					_ = comp.Return([]compute.Value{gt}, nil)
					outs, err := f.Sort(comp, 0, false, p[0])
					if err != nil {
						return nil, err
					}
					return outs[0], nil
				})
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Sort(1D_Descending) not implemented: %v", err)
				}
				t.Fatalf("Sort(1D_Descending) failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([]float32{8, 5, 3, 2, 1}, y); !ok {
				t.Errorf("Sort(1D_Descending) mismatch:\n%s", diff)
			}
		})

		t.Run("2D_Axis1", func(t *testing.T) {
			y, err := testutil.Exec1(b, []any{[][]int32{{3, 1, 2}, {9, 7, 8}}},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					comp, err := f.Closure()
					if err != nil {
						return nil, err
					}
					lhs, _ := comp.Parameter("lhs", shapes.Make(dtypes.Int32), nil)
					rhs, _ := comp.Parameter("rhs", shapes.Make(dtypes.Int32), nil)
					lt, _ := comp.LessThan(lhs, rhs)
					_ = comp.Return([]compute.Value{lt}, nil)
					outs, err := f.Sort(comp, 1, false, p[0])
					if err != nil {
						return nil, err
					}
					return outs[0], nil
				})
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Sort(2D_Axis1) not implemented: %v", err)
				}
				t.Fatalf("Sort(2D_Axis1) failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual([][]int32{{1, 2, 3}, {7, 8, 9}}, y); !ok {
				t.Errorf("Sort(2D_Axis1) mismatch:\n%s", diff)
			}
		})

		t.Run("MultiTensor_KeyValue_Stable", func(t *testing.T) {
			builder := b.Builder("test_sort_kv")
			mainFn := builder.Main()
			keys, _ := mainFn.Parameter("keys", shapes.Make(dtypes.Float32, 5), nil)
			vals, _ := mainFn.Parameter("vals", shapes.Make(dtypes.Int32, 5), nil)

			comp, _ := mainFn.Closure()
			kLhs, _ := comp.Parameter("k_lhs", shapes.Make(dtypes.Float32), nil)
			kRhs, _ := comp.Parameter("k_rhs", shapes.Make(dtypes.Float32), nil)
			_, _ = comp.Parameter("v_lhs", shapes.Make(dtypes.Int32), nil)
			_, _ = comp.Parameter("v_rhs", shapes.Make(dtypes.Int32), nil)
			lt, _ := comp.LessThan(kLhs, kRhs)
			_ = comp.Return([]compute.Value{lt}, nil)

			outs, err := mainFn.Sort(comp, 0, true, keys, vals)
			if err != nil {
				if compute.IsNotImplemented(err) {
					t.Skipf("Sort(MultiTensor) not implemented: %v", err)
				}
				t.Fatalf("Sort(MultiTensor) failed: %+v", err)
			}
			_ = mainFn.Return(outs, nil)

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile failed: %+v", err)
			}
			defer exec.Finalize()

			bKeys, _ := testutil.ToBuffer(b, []float32{3, 1, 2, 1, 2})
			bVals, _ := testutil.ToBuffer(b, []int32{30, 10, 20, 11, 21})
			defer bKeys.Finalize()
			defer bVals.Finalize()

			res, err := exec.Execute([]compute.Buffer{bKeys, bVals}, nil, 0)
			if err != nil {
				t.Fatalf("Execute failed: %+v", err)
			}
			defer res[0].Finalize()
			defer res[1].Finalize()

			gotKeys, _ := testutil.FromBuffer(b, res[0])
			gotVals, _ := testutil.FromBuffer(b, res[1])
			if ok, diff := testutil.IsEqual([]float32{1, 1, 2, 2, 3}, gotKeys); !ok {
				t.Errorf("Sorted keys mismatch:\n%s", diff)
			}
			if ok, diff := testutil.IsEqual([]int32{10, 11, 20, 21, 30}, gotVals); !ok {
				t.Errorf("Sorted values mismatch:\n%s", diff)
			}
		})
	})
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// tolerance for floating point comparison.
const fusedTestTolerance = 1e-6

func TestFusedOps(t *testing.T, b compute.Backend) {
	t.Run("FusedSoftmax", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedSoftmax)
		t.Run("1D", func(t *testing.T) {
			input := []float32{1.0, 2.0, 3.0, 4.0}
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedSoftmax(params[0], 0)
			})
			if err != nil {
				t.Fatalf("FusedSoftmax failed: %+v", err)
			}
			// Known-correct softmax([1,2,3,4]).
			want := []float32{0.0320586, 0.0871443, 0.2368828, 0.6439143}
			if ok, diff := testutil.IsInDelta(want, got, fusedTestTolerance); !ok {
				t.Errorf("Result not within delta %f:\n%s", fusedTestTolerance, diff)
			}
		})

		t.Run("2D", func(t *testing.T) {
			input := [][]float32{{1, 2, 3}, {4, 5, 6}}
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedSoftmax(params[0], 1)
			})
			if err != nil {
				t.Fatalf("FusedSoftmax failed: %+v", err)
			}
			gotSlice := got.([][]float32)
			// Each row should sum to 1.
			for row := range 2 {
				sum := gotSlice[row][0] + gotSlice[row][1] + gotSlice[row][2]
				if ok, diff := testutil.IsInDelta(float32(1.0), sum, fusedTestTolerance); !ok {
					t.Errorf("row %d: result not within delta %f:\n%s", row, fusedTestTolerance, diff)
				}
			}
		})

		t.Run("NegativeAxis", func(t *testing.T) {
			builder := b.Builder("fused_test")
			mainFn := builder.Main()
			param, _ := mainFn.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
			_, err := mainFn.FusedSoftmax(param, -1)
			if err == nil {
				t.Errorf("FusedSoftmax should reject negative axis")
			}
		})

		t.Run("Stability", func(t *testing.T) {
			input := []float32{1000, 1001, 1002}
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedSoftmax(params[0], 0)
			})
			if err != nil {
				t.Fatalf("FusedSoftmax failed: %+v", err)
			}
			gotSlice := got.([]float32)
			var sum float32
			for _, v := range gotSlice {
				sum += v
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("softmax produced NaN/Inf")
				}
			}
			if ok, diff := testutil.IsInDelta(float32(1.0), sum, fusedTestTolerance); !ok {
				t.Errorf("Result not within delta %f:\n%s", fusedTestTolerance, diff)
			}
		})
	})

	t.Run("FusedGelu", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedGelu)
		input := []float32{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}
		got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.FusedGelu(params[0], true)
		})
		if err != nil {
			t.Fatalf("FusedGelu failed: %+v", err)
		}
		want := []float32{-0.04550028, -0.15865526, -0.15426877, 0.0, 0.34573123, 0.84134474, 1.9544997}
		if ok, diff := testutil.IsInDelta(want, got, fusedTestTolerance); !ok {
			t.Errorf("Gelu result mismatch:\n%s", diff)
		}

		t.Run("Approximate", func(t *testing.T) {
			gotApprox, err := testutil.Exec1(b, []any{input}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedGelu(params[0], false)
			})
			if err != nil {
				t.Fatalf("FusedGelu failed: %+v", err)
			}
			// Differ count check
			exactGot := got.([]float32)
			approxGot := gotApprox.([]float32)
			differ := false
			for i := range approxGot {
				if math.Abs(float64(approxGot[i]-exactGot[i])) > 1e-7 {
					differ = true
					break
				}
			}
			if !differ {
				t.Errorf("approximate and exact GELU should differ for non-zero inputs")
			}
		})
	})

	t.Run("FusedLayerNorm", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedLayerNorm)
		t.Run("Simple", func(t *testing.T) {
			input := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
			epsilon := 1e-5
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedLayerNorm(params[0], []int{1}, epsilon, nil, nil)
			})
			if err != nil {
				t.Fatalf("FusedLayerNorm failed: %+v", err)
			}
			gotSlice := got.([][]float32)
			for row := range 2 {
				var sum, varSum float32
				for i := range 4 {
					sum += gotSlice[row][i]
				}
				mean := sum / 4.0
				if math.Abs(float64(mean)) > 1e-5 {
					t.Errorf("row %d mean %f too large", row, mean)
				}
				for i := range 4 {
					diff := gotSlice[row][i] - mean
					varSum += diff * diff
				}
				variance := varSum / 4.0
				if ok, diff := testutil.IsInDelta(float32(1.0), variance, 1e-3); !ok {
					t.Errorf("row %d variance mismatch:\n%s", row, diff)
				}
			}
		})

		t.Run("WithGammaBeta", func(t *testing.T) {
			input := []float32{1, 2, 3}
			gamma := []float32{2, 2, 2}
			beta := []float32{1, 1, 1}
			epsilon := 1e-5
			got, err := testutil.Exec1(b, []any{input, gamma, beta}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedLayerNorm(params[0], []int{0}, epsilon, params[1], params[2])
			})
			if err != nil {
				t.Fatalf("FusedLayerNorm failed: %+v", err)
			}
			// mean=2, var=2/3
			invStd := float32(1.0 / math.Sqrt(2.0/3.0+epsilon))
			want := []float32{(1.0 - 2.0)*invStd*2.0 + 1.0, (2.0 - 2.0)*invStd*2.0 + 1.0, (3.0 - 2.0)*invStd*2.0 + 1.0}
			if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
				t.Errorf("LayerNorm with gamma/beta mismatch:\n%s", diff)
			}
		})
	})

	t.Run("FusedDense", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedDense)
		x := [][]float32{{1, 2, 3}, {4, 5, 6}}
		w := [][]float32{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}}
		bias := []float32{10, 20, 30, 40}

		t.Run("None", func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x, w, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedDense(params[0], params[1], params[2], compute.ActivationNone)
			})
			if err != nil {
				t.Fatalf("FusedDense failed: %+v", err)
			}
			want := [][]float32{{11, 22, 33, 46}, {14, 25, 36, 55}}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Errorf("FusedDense mismatch:\n%s", diff)
			}
		})

		t.Run("Relu", func(t *testing.T) {
			x2 := [][]float32{{1, -1}}
			w2 := [][]float32{{1, 1}, {0, -1}}
			b2 := []float32{-1, -1}
			got, err := testutil.Exec1(b, []any{x2, w2, b2}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedDense(params[0], params[1], params[2], compute.ActivationRelu)
			})
			if err != nil {
				t.Fatalf("FusedDense failed: %+v", err)
			}
			want := [][]float32{{0, 1}}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Errorf("FusedDense Relu mismatch:\n%s", diff)
			}
		})
	})

	t.Run("FusedScaledDotProductAttention", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedScaledDotProductAttention)
		t.Run("BHSD_Causal", func(t *testing.T) {
			q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
			k := [][][][]float32{{{{1}, {1}}}}
			v := [][][][]float32{{{{10}, {20}}}}
			got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, nil)
			})
			if err != nil {
				t.Fatalf("SDPA failed: %+v", err)
			}
			want := [][][][]float32{{{{10}, {15}}}}
			if ok, diff := testutil.IsInDelta(want, got, fusedTestTolerance); !ok {
				t.Errorf("SDPA causal mismatch:\n%s", diff)
			}
		})

		t.Run("BSHD_Causal", func(t *testing.T) {
			q := [][][][]float32{{{{1}}, {{1}}}} // [1,2,1,1]
			k := [][][][]float32{{{{1}}, {{1}}}}
			v := [][][][]float32{{{{10}}, {{20}}}}
			got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBSHD, 1.0, true, nil)
			})
			if err != nil {
				t.Fatalf("SDPA failed: %+v", err)
			}
			want := [][][][]float32{{{{10}}, {{15}}}}
			if ok, diff := testutil.IsInDelta(want, got, fusedTestTolerance); !ok {
				t.Errorf("SDPA causal mismatch:\n%s", diff)
			}
		})

		t.Run("WithBooleanMask", func(t *testing.T) {
			q := [][][][]float32{{{{1}}}} // [1,1,1,1]
			k := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
			v := [][][][]float32{{{{10}, {20}}}} // [1,1,2,1]
			mask := [][]bool{{true, false}} // [1, 2]
			got, err := testutil.Exec1(b, []any{q, k, v, mask}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedScaledDotProductAttention(params[0], params[1], params[2], params[3], 1, 1, compute.AxesLayoutBHSD, 1.0, false, nil)
			})
			if err != nil {
				t.Fatalf("SDPA failed: %+v", err)
			}
			want := [][][][]float32{{{{10}}}}
			if ok, diff := testutil.IsInDelta(want, got, fusedTestTolerance); !ok {
				t.Errorf("SDPA with mask mismatch:\n%s", diff)
			}
		})

		t.Run("QuantizedMatmuls", func(t *testing.T) {
			q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
			k := [][][][]float32{{{{1}, {1}}}}
			v := [][][][]float32{{{{10}, {20}}}}
			got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, &compute.ScaledDotProductAttentionConfig{QuantizedMatmuls: true})
			})
			if err != nil {
				t.Fatalf("SDPA failed: %+v", err)
			}
			want := [][][][]float32{{{{10}, {15}}}}
			if ok, diff := testutil.IsInDelta(want, got, fusedTestTolerance); !ok {
				t.Errorf("SDPA quantized matmuls mismatch:\n%s", diff)
			}
		})
	})

	t.Run("FusedAttentionQKVProjection", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedAttentionQKVProjection)
		x := [][]float32{{1, 2, 3}}
		wQKV := [][]float32{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}}
		bq := []float32{10, 20}
		bk := []float32{100}
		bv := []float32{1000}

		gotQ, gotK, gotV, err := testutil.Exec3(b, []any{x, wQKV, bq, bk, bv}, func(f compute.Function, params []compute.Value) (compute.Value, compute.Value, compute.Value, error) {
			return f.FusedAttentionQKVProjection(params[0], params[1], params[2], params[3], params[4], 2, 1)
		})
		if err != nil {
			t.Fatalf("QKV Projection failed: %+v", err)
		}

		if ok, diff := testutil.IsEqual([][]float32{{11, 22}}, gotQ); !ok {
			t.Errorf("Q mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsEqual([][]float32{{103}}, gotK); !ok {
			t.Errorf("K mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsEqual([][]float32{{1006}}, gotV); !ok {
			t.Errorf("V mismatch:\n%s", diff)
		}
	})
}

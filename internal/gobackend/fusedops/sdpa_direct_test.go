package fusedops

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// init registers the SDPA executor on the go backend for this test binary only.
// The production init() in sdpa.go is gated `if false` (the CPU fused path is ~3x
// slower than SIMD matmul + decomposed, so the capability stays off). These tests
// exercise the reference directly, bypassing the capability gate, without enabling
// the op for real callers.
func init() {
	gobackend.RegisterFusedScaledDotProductAttention.Register(FusedScaledDotProductAttention, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedScaledDotProductAttention, gobackend.PriorityTyped, execFusedScaledDotProductAttention)
}

// newGoBackend builds the CPU go backend for direct reference tests.
func newGoBackend(t *testing.T) compute.Backend {
	t.Helper()
	b, err := gobackend.New("")
	if err != nil {
		t.Fatalf("gobackend.New: %+v", err)
	}
	return b
}

func TestSDPADirect_ConfigFieldsCompile(t *testing.T) {
	b := newGoBackend(t)
	// Setting every new config field with nil/zero values must be accepted
	// (no panic in option equality, output equals the no-config result).
	q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}
	v := [][][][]float32{{{{10}, {20}}}}
	cfg := &compute.ScaledDotProductAttentionConfig{
		QuantizedMatmuls: false,
		QuerySeqLen:      nil,
		KeyValueSeqLen:   nil,
	}
	got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with zero-value config failed: %+v", err)
	}
	// Causal: q0 sees k0 only -> 10; q1 sees k0,k1 (raw scores equal) -> 15.
	want := [][][][]float32{{{{10}, {15}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("SDPA zero-value config mismatch:\n%s", diff)
	}
	_ = math.Exp // keep math imported for later tasks in this file
}

func TestSDPADirect_WithSeqLens(t *testing.T) {
	b := newGoBackend(t)
	// batch=1, 1 head, seqLen=2 queries, kvLen=2 keys. KeyValueSeqLen=1 means
	// only key 0 is valid; the padding mask must match a materialized mask
	// that allows only key 0. QuerySeqLen=2 (both queries valid).
	q := [][][][]float32{{{{1}, {1}}}}   // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}   // [1,1,2,1]
	v := [][][][]float32{{{{10}, {20}}}} // [1,1,2,1]
	qLen := []int32{2}
	kvLen := []int32{1}
	got, err := testutil.Exec1(b, []any{q, k, v, qLen, kvLen}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with seqlens failed: %+v", err)
	}
	// Decomposed reference: only key0 valid -> every query attends to key0 only -> output 10.
	want := [][][][]float32{{{{10}, {10}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("SDPA seqlens padding mask mismatch:\n%s", diff)
	}
}

func TestSDPADirect_WithSeqLensCausal(t *testing.T) {
	b := newGoBackend(t)
	// causal + KeyValueSeqLen=2 (no key padding) reduces to plain causal.
	// QuerySeqLen=2. query0 sees key0 only (causal) -> 10; query1 sees key0,key1 -> 15.
	q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}
	v := [][][][]float32{{{{10}, {20}}}}
	qLen := []int32{2}
	kvLen := []int32{2}
	got, err := testutil.Exec1(b, []any{q, k, v, qLen, kvLen}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with seqlens+causal failed: %+v", err)
	}
	want := [][][][]float32{{{{10}, {15}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("SDPA seqlens+causal mismatch:\n%s", diff)
	}
}

// TestSDPADirect_OptionEqualityNoValuePanic verifies that equalOptions does not attempt
// == comparison on Value-typed seqlen fields, which would panic for non-comparable types.
func TestSDPADirect_OptionEqualityNoValuePanic(t *testing.T) {
	// []int is not comparable; using it as a Value exercises the panic-free path.
	nonComparable := []int{1, 2, 3}
	a := &nodeScaledDotProductAttention{
		options: &compute.ScaledDotProductAttentionConfig{
			QuantizedMatmuls: false,
			QuerySeqLen:      nonComparable,
			KeyValueSeqLen:   nonComparable,
		},
	}
	b := &nodeScaledDotProductAttention{
		options: &compute.ScaledDotProductAttentionConfig{
			QuantizedMatmuls: false,
			QuerySeqLen:      nonComparable,
			KeyValueSeqLen:   nonComparable,
		},
	}
	// Must not panic.
	if !a.EqualNodeData(b) {
		t.Error("expected equal node data for matching configs")
	}
	b.options.QuantizedMatmuls = true
	if a.EqualNodeData(b) {
		t.Error("expected unequal node data when QuantizedMatmuls differs")
	}
}

// TestSDPADirect_SeqLenWrongDtype verifies that passing a non-int32 seqlen tensor
// returns an error rather than panicking.
func TestSDPADirect_SeqLenWrongDtype(t *testing.T) {
	b := newGoBackend(t)
	q := [][][][]float32{{{{1}, {1}}}}
	k := [][][][]float32{{{{1}, {1}}}}
	v := [][][][]float32{{{{10}, {20}}}}
	// int64 seqlen: valid value but wrong element type.
	qLen := []int64{2}
	kvLen := []int64{2}
	_, err := testutil.Exec1(b, []any{q, k, v, qLen, kvLen}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
		return out, err
	})
	if err == nil {
		t.Fatal("expected error for wrong-dtype seqlen, got nil")
	}
}

// TestSDPADirect_SeqLenClamped verifies that out-of-range seqlen values are clamped
// to [0, dim] rather than producing negative loop counts or accessing out-of-bounds data.
func TestSDPADirect_SeqLenClamped(t *testing.T) {
	b := newGoBackend(t)
	q := [][][][]float32{{{{1}, {1}}}}
	k := [][][][]float32{{{{1}, {1}}}}
	v := [][][][]float32{{{{10}, {20}}}}

	// Negative seqlen: should behave as 0 valid positions (all-zero output).
	qLenNeg := []int32{-5}
	kvLenPos := []int32{2}
	got, err := testutil.Exec1(b, []any{q, k, v, qLenNeg, kvLenPos}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with negative qLen failed: %+v", err)
	}
	// qLimit clamped to 0: all positions padded -> all zeros.
	want := [][][][]float32{{{{0}, {0}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("negative qLen clamp: %s", diff)
	}

	// Too-large seqlen: should be clamped to actual dim (seqLen=2), no out-of-bounds.
	qLenLarge := []int32{999}
	kvLenLarge := []int32{999}
	got2, err := testutil.Exec1(b, []any{q, k, v, qLenLarge, kvLenLarge}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with too-large seqLen failed: %+v", err)
	}
	// With no causal and both seqlens clamped to 2, scores equal -> avg of values.
	want2 := [][][][]float32{{{{15}, {15}}}}
	if ok, diff := testutil.IsInDelta(want2, got2, 1e-5); !ok {
		t.Errorf("too-large seqLen clamp: %s", diff)
	}
}

// TestSDPADirect_WithBias verifies that an additive attention-score bias shifts
// softmax weights and produces an output that matches a decomposed reference.
// Bias strongly favors key 1 so the test is sensitive to whether bias is applied.
func TestSDPADirect_WithBias(t *testing.T) {
	b := newGoBackend(t)
	// batch=1, 1 head, seqLen=2 queries, kvLen=2 keys, headDim=1.
	// Raw scores (scale=1): q@k^T = [[1,1],[1,1]].
	// Bias [1,1,2,2]: [[[[-10, 10], [-10, 10]]]] strongly favors key 1.
	// After bias: scores = [[-9, 11], [-9, 11]].
	// softmax([-9,11]) ≈ [~0, ~1] -> output ≈ v[1] = 20 for both queries.
	q := [][][][]float32{{{{1}, {1}}}}        // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}        // [1,1,2,1]
	v := [][][][]float32{{{{10}, {20}}}}      // [1,1,2,1]
	bias := [][][][]float32{{{{-10, 10}, {-10, 10}}}} // [1,1,2,2] batch,head,seqLen,kvLen
	got, err := testutil.Exec1(b, []any{q, k, v, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{Bias: params[3]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with bias failed: %+v", err)
	}
	// softmax([-9,11]) ≈ [sigma_neg, sigma_pos]; sigma_neg ≈ 1/(1+e^20) ≈ 2e-9.
	// output ≈ 20 - tiny correction; tolerance 0.01 covers the residual.
	softmaxPos := float32(1.0 / (1.0 + math.Exp(-20)))
	softmaxNeg := 1 - softmaxPos
	wantVal := float32(softmaxNeg*10 + softmaxPos*20)
	want := [][][][]float32{{{{wantVal}, {wantVal}}}}
	if ok, diff := testutil.IsInDelta(want, got, 0.01); !ok {
		t.Errorf("SDPA with bias mismatch:\n%s", diff)
	}
}

// TestSDPADirect_BiasWithCausal verifies that additive bias and causal mask combine
// correctly: causal zeroes future positions, bias shifts scores in valid positions.
func TestSDPADirect_BiasWithCausal(t *testing.T) {
	b := newGoBackend(t)
	// batch=1, 1 head, seqLen=2, kvLen=2, headDim=1, scale=1.
	// Bias [1,1,2,2]: [[[[ 0, -100], [0, 10]]]] — large negative on (q0,k1) and large positive on (q1,k1).
	// Causal: q0 sees only k0 (k1 is future); q1 sees k0 and k1.
	// q0: causal masks k1 -> attends only k0 -> output = v[0] = 10 (bias on k1 irrelevant).
	// q1: scores = [1+0, 1+10] = [1, 11]; softmax([1,11]) -> large weight on k1 -> output ≈ 20.
	q := [][][][]float32{{{{1}, {1}}}}    // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}    // [1,1,2,1]
	v := [][][][]float32{{{{10}, {20}}}}  // [1,1,2,1]
	bias := [][][][]float32{{{{0, -100}, {0, 10}}}} // [1,1,2,2]
	got, err := testutil.Exec1(b, []any{q, k, v, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{Bias: params[3]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA bias+causal failed: %+v", err)
	}
	// q0: causal -> only k0 -> output=10.
	// q1: softmax([1+0, 1+10]) = softmax([1,11]) -> weight on k1 ≈ 1/(1+e^-10) ≈ ~1.
	softmaxQ1K1 := float32(1.0 / (1.0 + math.Exp(-10)))
	softmaxQ1K0 := 1 - softmaxQ1K1
	wantQ1 := float32(softmaxQ1K0*10 + softmaxQ1K1*20)
	want := [][][][]float32{{{{10}, {wantQ1}}}}
	if ok, diff := testutil.IsInDelta(want, got, 0.01); !ok {
		t.Errorf("SDPA bias+causal mismatch:\n%s", diff)
	}
}

func TestSDPADirect_FP8NotImplemented(t *testing.T) {
	b := newGoBackend(t)
	// The go backend rejects F8 parameters at creation, so feed F8 by converting
	// a float32 param to F8E4M3FN inside the graph (verified path), then assert
	// SDPA reports NotImplemented for the F8 dtype.
	builder := b.Builder("fused_fp8_test")
	mainFn := builder.Main()
	p, err := mainFn.Parameter("q", shapes.Make(dtypes.Float32, 1, 1, 2, 1), nil)
	if err != nil {
		t.Fatalf("Parameter failed: %+v", err)
	}
	q8, err := mainFn.ConvertDType(p, dtypes.F8E4M3FN)
	if err != nil {
		t.Fatalf("ConvertDType to F8E4M3FN failed: %+v", err)
	}
	_, _, err = mainFn.FusedScaledDotProductAttention(q8, q8, q8, nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, nil)
	if err == nil {
		t.Fatalf("SDPA with F8 input must return an error, got nil")
	}
	if !compute.IsNotImplemented(err) {
		t.Errorf("SDPA with F8 input must return ErrNotImplemented, got: %+v", err)
	}
}

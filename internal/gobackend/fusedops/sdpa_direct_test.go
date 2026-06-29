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

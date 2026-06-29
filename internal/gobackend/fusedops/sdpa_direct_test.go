package fusedops

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
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

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
	b, err := NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
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
		Scale:            1.0,
		Causal:           true,
	}
	got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, cfg)
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
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4], Scale: 1.0, Causal: false}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, cfg)
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
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4], Scale: 1.0, Causal: true}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, cfg)
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
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4], Scale: 1.0, Causal: false}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, cfg)
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
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4], Scale: 1.0, Causal: false}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, cfg)
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
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4], Scale: 1.0, Causal: false}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, cfg)
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

// TestSDPADirect_BiasValidation verifies that a malformed bias operand returns a clear
// validation error from buildSDPANode rather than silently panicking in the kernel.
func TestSDPADirect_BiasValidation(t *testing.T) {
	b := newGoBackend(t)
	q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1] BHSD: batch=1 head=1 seq=2 dim=1
	k := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1] kv=2
	v := [][][][]float32{{{{10}, {20}}}}

	cases := []struct {
		name string
		bias any
	}{
		// rank-1 bias: invalid (must be rank 2-4).
		{"rank1", []float32{1, 2}},
		// rank-2 bias [2,3]: second dim (3) != kvLen (2) and != 1, not broadcastable.
		{"rank2_bad_kvdim", [][]float32{{1, 2, 3}, {1, 2, 3}}},
		// rank-4 bias [1,1,3,2]: seqLen dim 3 != score seqLen 2 and != 1.
		{"rank4_bad_seqdim", [][][][]float32{{{{1, 2}, {1, 2}, {1, 2}}}}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := testutil.Exec1(b, []any{q, k, v, tc.bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				cfg := &compute.ScaledDotProductAttentionConfig{Bias: params[3], Scale: 1.0, Causal: false}
				out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], compute.AttentionAxesLayoutBHSD, cfg)
				return out, err
			})
			if err == nil {
				t.Fatalf("case %s: expected bias validation error, got nil", tc.name)
			}
		})
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
	_, _, err = mainFn.FusedScaledDotProductAttention(q8, q8, q8, compute.AttentionAxesLayoutBHSD, &compute.ScaledDotProductAttentionConfig{Scale: 1.0, Causal: true})
	if err == nil {
		t.Fatalf("SDPA with F8 input must return an error, got nil")
	}
	if !compute.IsNotImplemented(err) {
		t.Errorf("SDPA with F8 input must return ErrNotImplemented, got: %+v", err)
	}
}

func TestSDPADirect_MaskQuerySeqBroadcast(t *testing.T) {
	b := newGoBackend(t)

	// Verify that masks with mSeqDim == 1 (e.g. [batch, 1, 1, kv_len], [batch, 1, kv_len], [1, kv_len])
	// produce the exact same result as a full [batch, 1, seq_len, kv_len] mask in both BHSD and BSHD,
	// and also compile with dynamic sequence dimensions.
	const (
		batchSize = 2
		numHeads  = 2
		seqLen    = 39 // exercises 32-row SME, 4-row NEON, 2-row NEON, and 1-row tail
		kvLen     = 45
		headDim   = 32
	)

	qBHSD := make([]float32, batchSize*numHeads*seqLen*headDim)
	kBHSD := make([]float32, batchSize*numHeads*kvLen*headDim)
	vBHSD := make([]float32, batchSize*numHeads*kvLen*headDim)
	for i := range qBHSD {
		qBHSD[i] = float32((i%17)-8) * 0.1
	}
	for i := range kBHSD {
		kBHSD[i] = float32((i%13)-6) * 0.1
	}
	for i := range vBHSD {
		vBHSD[i] = float32((i%19)-9) * 0.2
	}

	// Build [batch, 1, 1, kvLen] mask and expanded [batch, 1, seqLen, kvLen] mask.
	mask1D := make([]float32, batchSize*kvLen)
	maskFull := make([]float32, batchSize*seqLen*kvLen)
	for bIdx := range batchSize {
		for kvIdx := range kvLen {
			var val float32
			if kvIdx >= kvLen-5-bIdx*3 {
				val = -1e9
			}
			mask1D[bIdx*kvLen+kvIdx] = val
			for sIdx := range seqLen {
				maskFull[(bIdx*seqLen+sIdx)*kvLen+kvIdx] = val
			}
		}
	}

	qShape := shapes.Make(dtypes.Float32, batchSize, numHeads, seqLen, headDim)
	kvShape := shapes.Make(dtypes.Float32, batchSize, numHeads, kvLen, headDim)
	mask1DShape := shapes.Make(dtypes.Float32, batchSize, 1, 1, kvLen)
	maskFullShape := shapes.Make(dtypes.Float32, batchSize, 1, seqLen, kvLen)

	runBHSD := func(maskData []float32, mShape shapes.Shape) []float32 {
		builder := b.Builder("sdpa_mask_bcast")
		fn := builder.Main()
		qP, _ := fn.Parameter("q", qShape, nil)
		kP, _ := fn.Parameter("k", kvShape, nil)
		vP, _ := fn.Parameter("v", kvShape, nil)
		mP, _ := fn.Parameter("m", mShape, nil)
		out, _, err := fn.FusedScaledDotProductAttention(qP, kP, vP, compute.AttentionAxesLayoutBHSD, &compute.ScaledDotProductAttentionConfig{
			Mask:  mP,
			Scale: 0.176776695,
		})
		if err != nil {
			t.Fatalf("FusedScaledDotProductAttention build failed for mask %s: %+v", mShape, err)
		}
		if err := fn.Return([]compute.Value{out}, nil); err != nil {
			t.Fatalf("Return: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile: %+v", err)
		}
		qBuf, _ := b.BufferFromFlatData(0, qBHSD, qShape)
		kBuf, _ := b.BufferFromFlatData(0, kBHSD, kvShape)
		vBuf, _ := b.BufferFromFlatData(0, vBHSD, kvShape)
		mBuf, _ := b.BufferFromFlatData(0, maskData, mShape)
		outs, err := exec.Execute([]compute.Buffer{qBuf, kBuf, vBuf, mBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute: %+v", err)
		}
		return outs[0].(*gobackend.Buffer).Flat.([]float32)
	}

	want := runBHSD(maskFull, maskFullShape)
	got := runBHSD(mask1D, mask1DShape)
	if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
		t.Fatalf("BHSD [batch, 1, 1, kvLen] mask broadcast mismatch:\n%s", diff)
	}

	// Also verify dynamic shape compilation with [batch=?, seq=?, heads, dim] in BSHD and [batch=?, 1, 1, seq=?] mask.
	{
		builder := b.Builder("sdpa_dynamic_bshd_mask")
		fn := builder.Main()
		qDynShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim, numHeads, headDim}, []string{"B", "S", "", ""})
		mDynShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 1, 1, shapes.DynamicDim}, []string{"B", "", "", "S"})
		qP, _ := fn.Parameter("q", qDynShape, nil)
		kP, _ := fn.Parameter("k", qDynShape, nil)
		vP, _ := fn.Parameter("v", qDynShape, nil)
		mP, _ := fn.Parameter("m", mDynShape, nil)
		out, _, err := fn.FusedScaledDotProductAttention(qP, kP, vP, compute.AttentionAxesLayoutBSHD, &compute.ScaledDotProductAttentionConfig{
			Mask:  mP,
			Scale: 0.176776695,
		})
		if err != nil {
			t.Fatalf("Dynamic BSHD FusedScaledDotProductAttention with [B, 1, 1, S] mask failed: %+v", err)
		}
		if err := fn.Return([]compute.Value{out}, nil); err != nil {
			t.Fatalf("Return: %+v", err)
		}
		if _, err := builder.Compile(); err != nil {
			t.Fatalf("Compile dynamic BSHD SDPA: %+v", err)
		}
	}
}

package fusedops

import (
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// NewBackend returns a "go" backend (*compute/internal/gobackend.Backend) for test.
// It returns an error if the backend is not a "go" backend.
func NewBackend() (*gobackend.Backend, error) {
	return gobackend.NewBackend()
}

func TestQKVProjectionConstantCaching(t *testing.T) {
	b, err := NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
	}
	defer b.Finalize()

	const inFeatures = 16
	const qDim = 16
	const kvDim = 8
	const totalOut = qDim + 2*kvDim

	wData := make([]float32, inFeatures*totalOut)
	for i := range wData {
		wData[i] = float32((i%7)-3) * 0.25
	}
	biasQData := make([]float32, qDim)
	biasKData := make([]float32, kvDim)
	biasVData := make([]float32, kvDim)
	for i := range biasQData {
		biasQData[i] = float32(i) * 0.1
	}
	for i := range biasKData {
		biasKData[i] = float32(i) * -0.1
		biasVData[i] = float32(i) * 0.05
	}

	buildAndRun := func(sharedW compute.Value, batch int) (compute.Value, []float32, []float32, []float32) {
		builder := b.Builder("qkv_const")
		fn := builder.Main()
		xShape := shapes.Make(dtypes.Float32, batch, inFeatures)
		xParam, err := fn.Parameter("x", xShape, nil)
		if err != nil {
			t.Fatalf("Parameter: %+v", err)
		}
		var wConst compute.Value
		if sharedW == nil {
			wConst, err = fn.Constant(wData, inFeatures, totalOut)
			if err != nil {
				t.Fatalf("Constant w: %+v", err)
			}
		} else {
			wConst = sharedW
		}
		bQ, _ := fn.Constant(biasQData, qDim)
		bK, _ := fn.Constant(biasKData, kvDim)
		bV, _ := fn.Constant(biasVData, kvDim)
		qOut, kOut, vOut, err := fn.FusedAttentionQKVProjection(xParam, wConst, bQ, bK, bV, qDim, kvDim)
		if err != nil {
			t.Fatalf("FusedAttentionQKVProjection: %+v", err)
		}
		if err := fn.Return([]compute.Value{qOut, kOut, vOut}, nil); err != nil {
			t.Fatalf("Return: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile: %+v", err)
		}
		t.Cleanup(exec.Finalize)

		xData := make([]float32, batch*inFeatures)
		for i := range xData {
			xData[i] = float32((i%5)+1) * 0.5
		}
		xBuf, err := b.BufferFromFlatData(0, xData, xShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData: %+v", err)
		}

		// Execute twice: first populates cache and evicts raw wQKV buffer, second uses cached form.
		for run := range 2 {
			outs, err := exec.Execute([]compute.Buffer{xBuf}, []bool{false}, 0)
			if err != nil {
				t.Fatalf("Execute run %d: %+v", run, err)
			}
			if run == 0 {
				continue
			}
			gotQ := make([]float32, batch*qDim)
			gotK := make([]float32, batch*kvDim)
			gotV := make([]float32, batch*kvDim)
			_ = outs[0].ToFlatData(gotQ)
			_ = outs[1].ToFlatData(gotK)
			_ = outs[2].ToFlatData(gotV)
			return wConst, gotQ, gotK, gotV
		}
		return wConst, nil, nil, nil
	}

	wConst, gotQ1, gotK1, gotV1 := buildAndRun(nil, 4)
	wNode := wConst.(*gobackend.Node)
	cd := wNode.Data.(*gobackend.ConstantData)
	if cd.CachedForm() == nil {
		t.Fatalf("expected wQKV ConstantData.CachedForm() to be populated after execution")
	}
	if cd.Buffer() != nil {
		t.Fatalf("expected wQKV ConstantData raw buffer to be released after caching")
	}

	// Second compilation with different batch size reusing the already-packed SharedConstant.
	_, gotQ2, gotK2, gotV2 := buildAndRun(wConst, 8)
	if ok, diff := testutil.IsInDelta(gotQ1, gotQ2[:4*qDim], 1e-5); !ok {
		t.Errorf("Q mismatch on shared constant recompilation:\n%s", diff)
	}
	if ok, diff := testutil.IsInDelta(gotK1, gotK2[:4*kvDim], 1e-5); !ok {
		t.Errorf("K mismatch on shared constant recompilation:\n%s", diff)
	}
	if ok, diff := testutil.IsInDelta(gotV1, gotV2[:4*kvDim], 1e-5); !ok {
		t.Errorf("V mismatch on shared constant recompilation:\n%s", diff)
	}
}

// TestConcurrentLayerNormFallbackAndConstantCaching verifies that concurrent
// first-time executions of a non-trailing FusedLayerNorm (falling back from SIMD
// to generic) and a Constant -> Transpose -> DotGeneral subgraph (releasing the
// raw constant buffer on first cache population) run race-free and consistently.
func TestConcurrentLayerNormFallbackAndConstantCaching(t *testing.T) {
	b, err := NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
	}
	defer b.Finalize()

	const (
		batch = 2
		seq   = 4
		feat  = 8
		numWorkers = 8
	)

	xShape := shapes.Make(dtypes.Float32, batch, seq, feat)
	xData := make([]float32, xShape.Size())
	for i := range xData {
		xData[i] = float32((i%11)-5) * 0.25
	}
	wData := make([]float32, feat*feat)
	for i := range wData {
		wData[i] = float32((i%7)-3) * 0.125
	}

	for iter := range 10 {
		builder := b.Builder("concurrent_ln_const")
		fn := builder.Main()

		x, err := fn.Parameter("x", xShape, nil)
		if err != nil {
			t.Fatalf("Parameter: %+v", err)
		}
		// Axis 0 (non-trailing, non-middle-contiguous when batch*seq*feat has d=seq*feat>1)
		// triggers FusedLayerNorm SIMD fallback to generic.
		ln, err := fn.FusedLayerNorm(x, []int{0}, 1e-5, nil, nil)
		if err != nil {
			t.Fatalf("FusedLayerNorm: %+v", err)
		}
		wConst, err := fn.Constant(wData, feat, feat)
		if err != nil {
			t.Fatalf("Constant: %+v", err)
		}
		wT, err := fn.Transpose(wConst, 1, 0)
		if err != nil {
			t.Fatalf("Transpose: %+v", err)
		}
		out, err := fn.DotGeneral(ln, []int{2}, nil, wT, []int{0}, nil, compute.DotGeneralConfig{})
		if err != nil {
			t.Fatalf("DotGeneral: %+v", err)
		}
		if err := fn.Return([]compute.Value{out}, nil); err != nil {
			t.Fatalf("Return: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile: %+v", err)
		}

		// Compute reference result on a fresh single run of an equivalent graph.
		var startWG, doneWG sync.WaitGroup
		startWG.Add(1)
		results := make([][]float32, numWorkers)
		errs := make(chan error, numWorkers)

		for w := range numWorkers {
			doneWG.Add(1)
			go func(workerIdx int) {
				defer doneWG.Done()
				xBuf, bufErr := b.BufferFromFlatData(0, xData, xShape)
				if bufErr != nil {
					errs <- bufErr
					return
				}
				defer func() { _ = xBuf.Finalize() }()

				startWG.Wait()
				outs, execErr := exec.Execute([]compute.Buffer{xBuf}, []bool{false}, 0)
				if execErr != nil {
					errs <- execErr
					return
				}
				got := make([]float32, xShape.Size())
				if copyErr := outs[0].ToFlatData(got); copyErr != nil {
					errs <- copyErr
					return
				}
				_ = outs[0].Finalize()
				results[workerIdx] = got
			}(w)
		}

		startWG.Done()
		doneWG.Wait()
		close(errs)
		for execErr := range errs {
			exec.Finalize()
			t.Fatalf("iter %d concurrent Execute failed: %+v", iter, execErr)
		}
		exec.Finalize()

		for w := 1; w < numWorkers; w++ {
			if ok, diff := testutil.IsInDelta(results[0], results[w], 1e-5); !ok {
				t.Fatalf("iter %d worker %d output mismatch:\n%s", iter, w, diff)
			}
		}
	}
}

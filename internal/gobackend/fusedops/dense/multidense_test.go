// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dense_test

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/avx512"
	_ "github.com/gomlx/compute/internal/gobackend/neon"
	_ "github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
)

func TestSiblingFusedMultiDense(t *testing.T) {
	testCases := []struct {
		name         string
		M, K         int
		Ns           []int
		weightLayout compute.DenseLayout
		withBias     []bool
		activations  []compute.ActivationType
		donateWeight bool
		dynamicM     bool
	}{
		{
			name:         "DecodeM1_GateUp_Transposed_Donated",
			M:            1,
			K:            256,
			Ns:           []int{384, 384},
			weightLayout: compute.DenseLayoutOutputsInput,
			withBias:     []bool{false, false},
			activations:  []compute.ActivationType{compute.ActivationGeluApproximate, compute.ActivationNone},
			donateWeight: true,
		},
		{
			name:         "DecodeM1_QKV_HeterogeneousN_Bias",
			M:            1,
			K:            256,
			Ns:           []int{512, 128, 128},
			weightLayout: compute.DenseLayoutOutputsInput,
			withBias:     []bool{true, false, true},
			activations:  []compute.ActivationType{compute.ActivationNone, compute.ActivationRelu, compute.ActivationSilu},
			donateWeight: true,
		},
		{
			name:         "DecodeM1_UnalignedTailN",
			M:            1,
			K:            137,
			Ns:           []int{73, 199},
			weightLayout: compute.DenseLayoutInputOutputs,
			withBias:     []bool{true, true},
			activations:  []compute.ActivationType{compute.ActivationGelu, compute.ActivationNone},
			donateWeight: false,
		},
		{
			name:         "Prefill_DynamicSeqLen_GateUp",
			M:            37,
			K:            192,
			Ns:           []int{256, 256},
			weightLayout: compute.DenseLayoutOutputsInput,
			withBias:     []bool{false, true},
			activations:  []compute.ActivationType{compute.ActivationSilu, compute.ActivationNone},
			donateWeight: true,
			dynamicM:     true,
		},
		{
			name:         "Prefill_DynamicSeqLen_QKV_HeterogeneousN",
			M:            64,
			K:            192,
			Ns:           []int{384, 96, 96},
			weightLayout: compute.DenseLayoutInputOutputs,
			withBias:     []bool{true, true, false},
			activations:  []compute.ActivationType{compute.ActivationNone, compute.ActivationNone, compute.ActivationNone},
			donateWeight: true,
			dynamicM:     true,
		},
	}

	dtypesToTest := []dtypes.DType{
		dtypes.Float32,
		dtypes.Float64,
		dtypes.BFloat16,
		dtypes.Float16,
	}

	for _, dt := range dtypesToTest {
		t.Run(dt.String(), func(t *testing.T) {
			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					runMultiDenseCase(t, dt, tc.M, tc.K, tc.Ns, tc.weightLayout, tc.withBias, tc.activations, tc.donateWeight, tc.dynamicM)
				})
			}
		})
	}
}

func runMultiDenseCase(
	t *testing.T,
	dt dtypes.DType,
	M, K int,
	Ns []int,
	layout compute.DenseLayout,
	withBias []bool,
	acts []compute.ActivationType,
	donateWeight bool,
	dynamicM bool,
) {
	t.Helper()
	b, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
	}
	defer b.Finalize()

	// Deterministic pseudo-random inputs in [-0.5, 0.5].
	xF32 := make([]float32, M*K)
	for i := range xF32 {
		xF32[i] = float32(((i*31+7)%23)-11) * 0.04
	}
	numSibs := len(Ns)
	wsF32 := make([][]float32, numSibs)
	bsF32 := make([][]float32, numSibs)
	for s, N := range Ns {
		wsF32[s] = make([]float32, K*N)
		for i := range wsF32[s] {
			wsF32[s][i] = float32(((i*(17+s*13)+(s+1)*11)%29)-14) * 0.03
		}
		if withBias[s] {
			bsF32[s] = make([]float32, N)
			for i := range bsF32[s] {
				bsF32[s][i] = float32(((i*(7+s*5)+3)%19)-9) * 0.05
			}
		}
	}

	makeBuf := func(shape shapes.Shape, f32 []float32) compute.Buffer {
		var flat any
		switch dt {
		case dtypes.Float32:
			flat = append([]float32(nil), f32...)
		case dtypes.Float64:
			out := make([]float64, len(f32))
			for i, v := range f32 {
				out[i] = float64(v)
			}
			flat = out
		case dtypes.BFloat16:
			out := make([]bfloat16.BFloat16, len(f32))
			for i, v := range f32 {
				out[i] = bfloat16.FromFloat32(v)
			}
			flat = out
		case dtypes.Float16:
			out := make([]float16.Float16, len(f32))
			for i, v := range f32 {
				out[i] = float16.FromFloat32(v)
			}
			flat = out
		}
		buf, err := b.BufferFromFlatData(0, flat, shape)
		if err != nil {
			t.Fatalf("BufferFromFlatData: %+v", err)
		}
		return buf
	}

	readF32 := func(buf compute.Buffer) []float32 {
		shape, err := buf.Shape()
		if err != nil {
			t.Fatalf("buf.Shape: %+v", err)
		}
		n := shape.Size()
		switch dt {
		case dtypes.Float32:
			out := make([]float32, n)
			if err := buf.ToFlatData(out); err != nil {
				t.Fatalf("ToFlatData: %+v", err)
			}
			return out
		case dtypes.Float64:
			raw := make([]float64, n)
			if err := buf.ToFlatData(raw); err != nil {
				t.Fatalf("ToFlatData: %+v", err)
			}
			out := make([]float32, n)
			for i, v := range raw {
				out[i] = float32(v)
			}
			return out
		case dtypes.BFloat16:
			raw := make([]bfloat16.BFloat16, n)
			if err := buf.ToFlatData(raw); err != nil {
				t.Fatalf("ToFlatData: %+v", err)
			}
			out := make([]float32, n)
			for i, v := range raw {
				out[i] = v.Float32()
			}
			return out
		case dtypes.Float16:
			raw := make([]float16.Float16, n)
			if err := buf.ToFlatData(raw); err != nil {
				t.Fatalf("ToFlatData: %+v", err)
			}
			out := make([]float32, n)
			for i, v := range raw {
				out[i] = v.Float32()
			}
			return out
		}
		return nil
	}

	buildAndRun := func(enableMultiFusion bool) [][]float32 {
		savedFn := gobackend.BuildFusedMultiDenseDataFn
		if !enableMultiFusion {
			gobackend.BuildFusedMultiDenseDataFn = nil
		}
		defer func() { gobackend.BuildFusedMultiDenseDataFn = savedFn }()

		builder := b.Builder("test_multi_dense")
		f := builder.Main()

		var xParamShape shapes.Shape
		if dynamicM {
			xParamShape = shapes.MakeDynamic(dt, []int{1, shapes.DynamicDim, K}, []string{"", "seq_len", ""})
		} else {
			xParamShape = shapes.Make(dt, 1, M, K)
		}
		xVal, err := f.Parameter("x", xParamShape, nil)
		if err != nil {
			t.Fatalf("Parameter x: %+v", err)
		}

		execInputs := []compute.Buffer{makeBuf(shapes.Make(dt, 1, M, K), xF32)}
		outs := make([]compute.Value, numSibs)
		for s, N := range Ns {
			var wShape shapes.Shape
			if layout == compute.DenseLayoutOutputsInput {
				wShape = shapes.Make(dt, N, K)
			} else {
				wShape = shapes.Make(dt, K, N)
			}
			wBuf := makeBuf(wShape, wsF32[s])
			var wVal compute.Value
			if donateWeight {
				wVal, err = f.ConstantFromBuffer(wBuf, true)
				if err != nil {
					t.Fatalf("ConstantFromBuffer w[%d]: %+v", s, err)
				}
			} else {
				wVal, err = f.Parameter("w", wShape, nil)
				if err != nil {
					t.Fatalf("Parameter w[%d]: %+v", s, err)
				}
				execInputs = append(execInputs, wBuf)
			}

			var bVal compute.Value
			if withBias[s] {
				bShape := shapes.Make(dt, N)
				bBuf := makeBuf(bShape, bsF32[s])
				bVal, err = f.Parameter("b", bShape, nil)
				if err != nil {
					t.Fatalf("Parameter b[%d]: %+v", s, err)
				}
				execInputs = append(execInputs, bBuf)
			}

			outs[s], err = f.FusedDense(xVal, wVal, bVal, compute.DenseConfig{
				Activation:   compute.ActivationConfig{Type: acts[s]},
				WeightLayout: layout,
			})
			if err != nil {
				t.Fatalf("FusedDense[%d]: %+v", s, err)
			}
		}

		if err := f.Return(outs, nil); err != nil {
			t.Fatalf("Return: %+v", err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile: %+v", err)
		}
		defer exec.Finalize()

		// Execute twice to verify both first-run pre-packing and cached-run execution.
		var results [][]float32
		for run := range 2 {
			gotBufs, err := exec.Execute(execInputs, nil, 0)
			if err != nil {
				t.Fatalf("Execute (run %d, fused=%v): %+v", run, enableMultiFusion, err)
			}
			if run == 1 {
				results = make([][]float32, numSibs)
				for s, gb := range gotBufs {
					results[s] = readF32(gb)
				}
			}
			for _, gb := range gotBufs {
				_ = gb.Finalize()
			}
		}
		for _, ib := range execInputs {
			_ = ib.Finalize()
		}
		return results
	}

	want := buildAndRun(false)
	got := buildAndRun(true)

	tol := 1e-4
	switch dt {
	case dtypes.BFloat16:
		tol = 2e-2
	case dtypes.Float16:
		tol = 5e-3
	}
	for s := range numSibs {
		if len(got[s]) != len(want[s]) {
			t.Fatalf("sibling %d length mismatch: got %d, want %d", s, len(got[s]), len(want[s]))
		}
		for i := range want[s] {
			diff := math.Abs(float64(got[s][i] - want[s][i]))
			if diff > tol {
				t.Fatalf("sibling %d idx %d: got=%v, want=%v (diff=%v > tol=%v)",
					s, i, got[s][i], want[s][i], diff, tol)
			}
		}
	}
}

// TestSiblingFusedMultiDenseCycleSafety verifies that if one sibling's weight depends
// on another sibling's output, fusion does not introduce a cycle in the DAG.
func TestSiblingFusedMultiDenseCycleSafety(t *testing.T) {
	b, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
	}
	defer b.Finalize()

	builder := b.Builder("cycle_safety")
	f := builder.Main()
	x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 4, 4), nil)
	if err != nil {
		t.Fatalf("Parameter x: %+v", err)
	}
	w1, err := f.Parameter("w1", shapes.Make(dtypes.Float32, 4, 4), nil)
	if err != nil {
		t.Fatalf("Parameter w1: %+v", err)
	}
	// y1 = x @ w1
	y1, err := f.FusedDense(x, w1, nil, compute.DenseConfig{WeightLayout: compute.DenseLayoutInputOutputs})
	if err != nil {
		t.Fatalf("FusedDense y1: %+v", err)
	}
	// y2 = x @ y1  (y2 shares x with y1, but its weight depends on y1 -> cannot fuse!)
	y2, err := f.FusedDense(x, y1, nil, compute.DenseConfig{WeightLayout: compute.DenseLayoutInputOutputs})
	if err != nil {
		t.Fatalf("FusedDense y2: %+v", err)
	}
	if err := f.Return([]compute.Value{y1, y2}, nil); err != nil {
		t.Fatalf("Return: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed (possible cycle?): %+v", err)
	}
	defer exec.Finalize()

	xData := make([]float32, 16)
	w1Data := make([]float32, 16)
	for i := range 4 {
		xData[i*4+i] = 2.0
		w1Data[i*4+i] = 3.0
	}
	xBuf, _ := b.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, 4, 4))
	wBuf, _ := b.BufferFromFlatData(0, w1Data, shapes.Make(dtypes.Float32, 4, 4))
	defer xBuf.Finalize()
	defer wBuf.Finalize()

	outs, err := exec.Execute([]compute.Buffer{xBuf, wBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute: %+v", err)
	}
	defer outs[0].Finalize()
	defer outs[1].Finalize()

	got1 := make([]float32, 16)
	got2 := make([]float32, 16)
	_ = outs[0].ToFlatData(got1)
	_ = outs[1].ToFlatData(got2)
	for i := range 4 {
		if got1[i*4+i] != 6.0 {
			t.Fatalf("y1[%d,%d] = %v, want 6.0", i, i, got1[i*4+i])
		}
		if got2[i*4+i] != 12.0 {
			t.Fatalf("y2[%d,%d] = %v, want 12.0", i, i, got2[i*4+i])
		}
	}
}

func BenchmarkSiblingFusedMultiDense(b *testing.B) {
	cases := []struct {
		name string
		M, K int
		Ns   []int
		acts []compute.ActivationType
	}{
		{"M1_GateUp_12288", 1, 1536, []int{12288, 12288}, []compute.ActivationType{compute.ActivationGeluApproximate, compute.ActivationNone}},
		{"M1_GateUp_6144", 1, 1536, []int{6144, 6144}, []compute.ActivationType{compute.ActivationGeluApproximate, compute.ActivationNone}},
		{"M1_QKV_2048_256_256", 1, 1536, []int{2048, 256, 256}, []compute.ActivationType{compute.ActivationNone, compute.ActivationNone, compute.ActivationNone}},
		{"M234_GateUp_12288", 234, 1536, []int{12288, 12288}, []compute.ActivationType{compute.ActivationGeluApproximate, compute.ActivationNone}},
		{"M356_GateUp_12288", 356, 1536, []int{12288, 12288}, []compute.ActivationType{compute.ActivationGeluApproximate, compute.ActivationNone}},
		{"M356_QKV_2048_256_256", 356, 1536, []int{2048, 256, 256}, []compute.ActivationType{compute.ActivationNone, compute.ActivationNone, compute.ActivationNone}},
	}

	par := min(16, runtime.NumCPU())
	for _, tc := range cases {
		for _, mode := range []struct {
			label    string
			fused    bool
			execConf string
		}{
			{"Unfused_Seq", false, fmt.Sprintf("parallelism=%d,ops_sequential", par)},
			{"Unfused_Par", false, fmt.Sprintf("parallelism=%d,ops_parallel", par)},
			{"Fused_Seq", true, fmt.Sprintf("parallelism=%d,ops_sequential", par)},
		} {
			b.Run(fmt.Sprintf("%s/%s", tc.name, mode.label), func(b *testing.B) {
				be, err := gobackend.New(mode.execConf)
				if err != nil {
					b.Fatalf("NewBackend: %+v", err)
				}
				defer be.Finalize()

				savedFn := gobackend.BuildFusedMultiDenseDataFn
				if !mode.fused {
					gobackend.BuildFusedMultiDenseDataFn = nil
				}
				defer func() { gobackend.BuildFusedMultiDenseDataFn = savedFn }()

				xData := make([]bfloat16.BFloat16, tc.M*tc.K)
				for i := range xData {
					xData[i] = bfloat16.FromFloat32(float32((i%17)-8) * 0.05)
				}
				xBuf, err := be.BufferFromFlatData(0, xData, shapes.Make(dtypes.BFloat16, 1, tc.M, tc.K))
				if err != nil {
					b.Fatalf("xBuf: %+v", err)
				}
				defer xBuf.Finalize()

				builder := be.Builder("bench_multi_dense")
				f := builder.Main()
				xVal, err := f.Parameter("x", shapes.Make(dtypes.BFloat16, 1, tc.M, tc.K), nil)
				if err != nil {
					b.Fatalf("Parameter x: %+v", err)
				}
				outs := make([]compute.Value, len(tc.Ns))
				for s, N := range tc.Ns {
					wData := make([]bfloat16.BFloat16, N*tc.K)
					for i := range wData {
						wData[i] = bfloat16.FromFloat32(float32(((i+s*7)%19)-9) * 0.02)
					}
					wBuf, err := be.BufferFromFlatData(0, wData, shapes.Make(dtypes.BFloat16, N, tc.K))
					if err != nil {
						b.Fatalf("wBuf: %+v", err)
					}
					wVal, err := f.ConstantFromBuffer(wBuf, true)
					if err != nil {
						b.Fatalf("ConstantFromBuffer: %+v", err)
					}
					outs[s], err = f.FusedDense(xVal, wVal, nil, compute.DenseConfig{
						Activation:   compute.ActivationConfig{Type: tc.acts[s]},
						WeightLayout: compute.DenseLayoutOutputsInput,
					})
					if err != nil {
						b.Fatalf("FusedDense: %+v", err)
					}
				}
				if err := f.Return(outs, nil); err != nil {
					b.Fatalf("Return: %+v", err)
				}
				exec, err := builder.Compile()
				if err != nil {
					b.Fatalf("Compile: %+v", err)
				}
				defer exec.Finalize()

				// Warm-up (pre-packs weights).
				warmOuts, err := exec.Execute([]compute.Buffer{xBuf}, nil, 0)
				if err != nil {
					b.Fatalf("warmup Execute: %+v", err)
				}
				for _, o := range warmOuts {
					_ = o.Finalize()
				}

				b.ResetTimer()
				for b.Loop() {
					gotOuts, err := exec.Execute([]compute.Buffer{xBuf}, nil, 0)
					if err != nil {
						b.Fatalf("Execute: %+v", err)
					}
					for _, o := range gotOuts {
						_ = o.Finalize()
					}
				}
			})
		}
	}
}

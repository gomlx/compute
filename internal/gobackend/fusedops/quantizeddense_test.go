// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"encoding/binary"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// TestGGMLAllFormatsVecDotVsDequant verifies that every GGML block-integer dot product
// (Mode B: Q8_0 / Q8_K) matches float dequantization + dot product (Mode A) on identical quantized inputs.
func TestGGMLAllFormatsVecDotVsDequant(t *testing.T) {
	allTypes := []compute.GGMLQuantType{
		compute.GGMLQ4_0,
		compute.GGMLQ4_1,
		compute.GGMLQ5_0,
		compute.GGMLQ5_1,
		compute.GGMLQ8_0,
		compute.GGMLIQ4NL,
		compute.GGMLQ2_K,
		compute.GGMLQ3_K,
		compute.GGMLQ4_K,
		compute.GGMLQ5_K,
		compute.GGMLQ6_K,
		compute.GGMLQ8_K,
	}

	for _, wType := range allTypes {
		t.Run(wType.String(), func(t *testing.T) {
			vpb := wType.ValuesPerBlock()
			bpb := wType.BytesPerBlock()
			const numBlocks = 2
			K := numBlocks * vpb

			// Build deterministic valid block bytes (avoiding NaN/Inf fp16 headers).
			wRow := make([]uint8, numBlocks*bpb)
			for i := range wRow {
				wRow[i] = uint8((i*37 + 13) & 0xFF)
			}
			dF16 := float16.FromFloat32(0.05)
			dminF16 := float16.FromFloat32(0.02)
			for blk := range numBlocks {
				off := blk * bpb
				switch wType {
				case compute.GGMLQ4_0, compute.GGMLQ5_0, compute.GGMLQ8_0, compute.GGMLIQ4NL:
					binary.LittleEndian.PutUint16(wRow[off:off+2], dF16.Bits())
				case compute.GGMLQ4_1, compute.GGMLQ5_1:
					binary.LittleEndian.PutUint16(wRow[off:off+2], dF16.Bits())
					binary.LittleEndian.PutUint16(wRow[off+2:off+4], dminF16.Bits())
				case compute.GGMLQ2_K:
					binary.LittleEndian.PutUint16(wRow[off+80:off+82], dF16.Bits())
					binary.LittleEndian.PutUint16(wRow[off+82:off+84], dminF16.Bits())
				case compute.GGMLQ3_K:
					binary.LittleEndian.PutUint16(wRow[off+108:off+110], dF16.Bits())
				case compute.GGMLQ4_K, compute.GGMLQ5_K:
					binary.LittleEndian.PutUint16(wRow[off:off+2], dF16.Bits())
					binary.LittleEndian.PutUint16(wRow[off+2:off+4], dminF16.Bits())
				case compute.GGMLQ6_K:
					binary.LittleEndian.PutUint16(wRow[off+208:off+210], dF16.Bits())
				case compute.GGMLQ8_K:
					var tmp [256]float32
					for j := range 256 {
						tmp[j] = float32(((blk*31+j*7)%29)-14) * 0.1
					}
					quantizeRowQ8_K(tmp[:], wRow[off:off+bpb])
				}
			}

			// Prepare float x -> quantize to Q8_0 or Q8_K -> dequantize xQ8 back to float32.
			xSrc := make([]float32, K)
			for i := range xSrc {
				xSrc[i] = float32(((i*17+5)%23)-11) * 0.15
			}
			var xType compute.GGMLQuantType
			if vpb == 32 {
				xType = compute.GGMLQ8_0
			} else {
				xType = compute.GGMLQ8_K
			}
			xQ8 := make([]uint8, numBlocks*xType.BytesPerBlock())
			xDeq := make([]float32, K)
			if xType == compute.GGMLQ8_0 {
				quantizeRowQ8_0(xSrc, xQ8)
				dequantQ8_0Row(xQ8, xDeq)
			} else {
				quantizeRowQ8_K(xSrc, xQ8)
				dequantQ8_KRow(xQ8, xDeq)
			}

			wDeqFn, err := ggmlDequantFunc(wType)
			if err != nil {
				t.Fatalf("ggmlDequantFunc(%s): %+v", wType, err)
			}
			wDeq := make([]float32, K)
			wDeqFn(wRow, wDeq)

			var want float32
			for k := range K {
				want += wDeq[k] * xDeq[k]
			}

			dotFn, err := ggmlVecDotFunc(wType)
			if err != nil {
				t.Fatalf("ggmlVecDotFunc(%s): %+v", wType, err)
			}
			got := dotFn(wRow, xQ8)

			if ok, diff := testutil.IsInDelta(want, got, 1e-2); !ok {
				t.Errorf("%s vecDot mismatch: want=%f got=%f\n%s", wType, want, got, diff)
			}
		})
	}
}

func TestQuantizedDenseDynamicBatchAndParallelTiles(t *testing.T) {
	b, err := NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
	}
	defer b.Finalize()

	// Use K=128, N=64 so M=1 triggers N/K column tiling (1*128*64 = 8192 > 4096)
	// and M=6 triggers M-row parallelization, both via a single dynamic-batch Executable!
	const K, N, G = 128, 64, 32
	wData := make([]int8, K*N)
	for i := range wData {
		wData[i] = int8((i % 15) - 7)
	}
	scalesData := make([]float32, (K/G)*N)
	for i := range scalesData {
		scalesData[i] = 0.02 + float32(i%7)*0.01
	}
	biasData := make([]float32, N)
	for n := range N {
		biasData[n] = float32(n-32) * 0.05
	}

	builder := b.Builder("dyn_qd_vjp")
	fn := builder.Main()
	dynXShape := shapes.Shape{
		DType:      dtypes.Float32,
		Dimensions: []int{shapes.DynamicDim, K},
		AxisNames:  []string{"batch", ""},
	}
	dynDOutShape := shapes.Shape{
		DType:      dtypes.Float32,
		Dimensions: []int{shapes.DynamicDim, N},
		AxisNames:  []string{"batch", ""},
	}
	xParam, err := fn.Parameter("x", dynXShape, nil)
	if err != nil {
		t.Fatalf("Parameter x: %+v", err)
	}
	dOutParam, err := fn.Parameter("dOut", dynDOutShape, nil)
	if err != nil {
		t.Fatalf("Parameter dOut: %+v", err)
	}
	wConst, _ := fn.Constant(wData, K, N)
	sConst, _ := fn.Constant(scalesData, K/G, N)
	bConst, _ := fn.Constant(biasData, N)

	cfg := compute.FusedQuantizedDenseConfig{
		WeightQuantization: &compute.Quantization{
			Scheme:    compute.QuantLinear,
			Scale:     sConst,
			BlockAxis: 0,
			BlockSize: G,
		},
		Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
		WeightLayout: compute.DenseLayoutInputOutputs,
	}
	y, err := fn.FusedQuantizedDense(xParam, wConst, bConst, cfg)
	if err != nil {
		t.Fatalf("FusedQuantizedDense: %+v", err)
	}
	dx, dBias, err := fn.FusedQuantizedDenseVJP(nil, wConst, bConst, y, dOutParam, cfg)
	if err != nil {
		t.Fatalf("FusedQuantizedDenseVJP: %+v", err)
	}
	if err := fn.Return([]compute.Value{y, dx, dBias}, nil); err != nil {
		t.Fatalf("Return: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile: %+v", err)
	}
	defer exec.Finalize()

	for _, M := range []int{1, 6} {
		xData := make([]float32, M*K)
		for i := range xData {
			xData[i] = float32((i%13)-6) * 0.1
		}
		dOutData := make([]float32, M*N)
		for i := range dOutData {
			dOutData[i] = float32((i%9)-4) * 0.2
		}
		xBuf, _ := b.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, M, K))
		dOutBuf, _ := b.BufferFromFlatData(0, dOutData, shapes.Make(dtypes.Float32, M, N))

		outs, err := exec.Execute([]compute.Buffer{xBuf, dOutBuf}, []bool{true, true}, 0)
		if err != nil {
			t.Fatalf("Execute(M=%d): %+v", M, err)
		}

		gotY := make([]float32, M*N)
		gotDx := make([]float32, M*K)
		gotDBias := make([]float32, N)
		_ = outs[0].ToFlatData(gotY)
		_ = outs[1].ToFlatData(gotDx)
		_ = outs[2].ToFlatData(gotDBias)

		wantY := make([]float32, M*N)
		wantDx := make([]float32, M*K)
		wantDBias := make([]float32, N)
		for m := range M {
			dZ := make([]float32, N)
			for n := range N {
				sum := biasData[n]
				for k := range K {
					sum += xData[m*K+k] * float32(wData[k*N+n]) * scalesData[(k/G)*N+n]
				}
				if sum > 0 {
					wantY[m*N+n] = sum
					dZ[n] = dOutData[m*N+n]
				}
				wantDBias[n] += dZ[n]
			}
			for k := range K {
				var g float32
				for n := range N {
					g += dZ[n] * float32(wData[k*N+n]) * scalesData[(k/G)*N+n]
				}
				wantDx[m*K+k] = g
			}
		}

		if ok, diff := testutil.IsInDelta(wantY, gotY, 1e-4); !ok {
			t.Errorf("M=%d y mismatch:\n%s", M, diff)
		}
		if ok, diff := testutil.IsInDelta(wantDx, gotDx, 1e-4); !ok {
			t.Errorf("M=%d dx mismatch:\n%s", M, diff)
		}
		if ok, diff := testutil.IsInDelta(wantDBias, gotDBias, 1e-4); !ok {
			t.Errorf("M=%d dBias mismatch:\n%s", M, diff)
		}
	}
}

func TestQuantizedDenseVJPFiniteDifference(t *testing.T) {
	b, err := NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
	}
	defer b.Finalize()

	const M, K, N = 2, 6, 4
	x := [][]float32{
		{0.3, -0.4, 0.7, -0.2, 0.5, -0.6},
		{-0.5, 0.8, -0.1, 0.4, -0.3, 0.2},
	}
	w := [][]int8{
		{3, -2, 5, -4},
		{-1, 4, -3, 2},
		{2, -5, 1, 3},
		{-4, 1, 4, -2},
		{5, 2, -1, 4},
		{-2, -3, 2, -5},
	}
	scale := []float32{0.1, 0.15, 0.2, 0.12}
	zp := []int8{1, -1, 0, 2}
	bias := []float32{0.2, -0.1, 0.05, 0.3}
	dOut := [][]float32{
		{0.5, -0.3, 0.8, -0.2},
		{-0.4, 0.6, -0.1, 0.7},
	}

	evalLoss := func(xIn [][]float32, biasIn []float32) float64 {
		gotY, err := testutil.Exec1(b, []any{xIn, w, scale, zp, biasIn},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], p[4], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[2],
						ZeroPoint: p[3],
					},
					Activation:   compute.ActivationConfig{Type: compute.ActivationTanh},
					WeightLayout: compute.DenseLayoutInputOutputs,
				})
			})
		if err != nil {
			t.Fatalf("evalLoss failed: %+v", err)
		}
		yMat := gotY.([][]float32)
		var loss float64
		for m := range M {
			for n := range N {
				loss += float64(yMat[m][n]) * float64(dOut[m][n])
			}
		}
		return loss
	}

	gotDxAny, gotDBiasAny, err := testutil.Exec2(b, []any{x, w, scale, zp, bias, dOut},
		func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
			cfg := compute.FusedQuantizedDenseConfig{
				WeightQuantization: &compute.Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     p[2],
					ZeroPoint: p[3],
				},
				Activation:   compute.ActivationConfig{Type: compute.ActivationTanh},
				WeightLayout: compute.DenseLayoutInputOutputs,
			}
			y, err := f.FusedQuantizedDense(p[0], p[1], p[4], cfg)
			if err != nil {
				return nil, nil, err
			}
			return f.FusedQuantizedDenseVJP(p[0], p[1], p[4], y, p[5], cfg)
		})
	if err != nil {
		t.Fatalf("FusedQuantizedDenseVJP failed: %+v", err)
	}
	gotDx := gotDxAny.([][]float32)
	gotDBias := gotDBiasAny.([]float32)

	const eps = 1e-3
	for m := range M {
		for k := range K {
			orig := x[m][k]
			x[m][k] = orig + eps
			lp := evalLoss(x, bias)
			x[m][k] = orig - eps
			lm := evalLoss(x, bias)
			x[m][k] = orig
			numGrad := float32((lp - lm) / (2 * eps))
			if ok, diff := testutil.IsInDelta(numGrad, gotDx[m][k], 2e-3); !ok {
				t.Errorf("finite-diff dx[%d,%d]: num=%f analytic=%f\n%s", m, k, numGrad, gotDx[m][k], diff)
			}
		}
	}
	for n := range N {
		orig := bias[n]
		bias[n] = orig + eps
		lp := evalLoss(x, bias)
		bias[n] = orig - eps
		lm := evalLoss(x, bias)
		bias[n] = orig
		numGrad := float32((lp - lm) / (2 * eps))
		if ok, diff := testutil.IsInDelta(numGrad, gotDBias[n], 2e-3); !ok {
			t.Errorf("finite-diff dBias[%d]: num=%f analytic=%f\n%s", n, numGrad, gotDBias[n], diff)
		}
	}
}

func makeTestGGMLWeightMatrix(N, K int, wType compute.GGMLQuantType, seed int) []uint8 {
	vpb := wType.ValuesPerBlock()
	bpb := wType.BytesPerBlock()
	numBlocks := K / vpb
	bytesPerRow := numBlocks * bpb
	out := make([]uint8, N*bytesPerRow)
	dF16 := float16.FromFloat32(0.04)
	dminF16 := float16.FromFloat32(0.015)
	for n := range N {
		wRow := out[n*bytesPerRow : (n+1)*bytesPerRow]
		for i := range wRow {
			wRow[i] = uint8((i*37 + n*19 + seed) & 0xFF)
		}
		for blk := range numBlocks {
			off := blk * bpb
			switch wType {
			case compute.GGMLQ4_0, compute.GGMLQ5_0, compute.GGMLQ8_0, compute.GGMLIQ4NL:
				binary.LittleEndian.PutUint16(wRow[off:off+2], dF16.Bits())
			case compute.GGMLQ4_1, compute.GGMLQ5_1, compute.GGMLQ4_K, compute.GGMLQ5_K:
				binary.LittleEndian.PutUint16(wRow[off:off+2], dF16.Bits())
				binary.LittleEndian.PutUint16(wRow[off+2:off+4], dminF16.Bits())
			case compute.GGMLQ6_K:
				binary.LittleEndian.PutUint16(wRow[off+208:off+210], dF16.Bits())
			}
		}
	}
	return out
}

func TestSiblingSharedQuantActGGMLAndLinear(t *testing.T) {
	b, err := NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %+v", err)
	}
	defer b.Finalize()

	t.Run("GGML_QKV_DynamicAndConcurrent", func(t *testing.T) {
		const K, N = 64, 48
		wQ := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ4_0, 1)
		wK := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ5_0, 2)
		wV := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ8_0, 3)

		dynXShape := shapes.Shape{DType: dtypes.Float32, Dimensions: []int{shapes.DynamicDim, K}, AxisNames: []string{"M", ""}}
		compileSingle := func(wBytes []uint8, wType compute.GGMLQuantType) compute.Executable {
			builder := b.Builder("single_" + wType.String())
			fn := builder.Main()
			xParam, err := fn.Parameter("x", dynXShape, nil)
			if err != nil {
				t.Fatal(err)
			}
			bpr := (K / wType.ValuesPerBlock()) * wType.BytesPerBlock()
			wConst, err := fn.Constant(wBytes, N, bpr)
			if err != nil {
				t.Fatal(err)
			}
			y, err := fn.FusedQuantizedDense(xParam, wConst, nil, compute.FusedQuantizedDenseConfig{
				WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: wType},
				QuantizedMatmul:    true,
				WeightLayout:       compute.DenseLayoutOutputsInput,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := fn.Return([]compute.Value{y}, nil); err != nil {
				t.Fatal(err)
			}
			ex, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			return ex
		}

		refQ := compileSingle(wQ, compute.GGMLQ4_0)
		defer refQ.Finalize()
		refK := compileSingle(wK, compute.GGMLQ5_0)
		defer refK.Finalize()
		refV := compileSingle(wV, compute.GGMLQ8_0)
		defer refV.Finalize()

		builder := b.Builder("shared_qkv_ggml")
		fn := builder.Main()
		xParam, err := fn.Parameter("x", dynXShape, nil)
		if err != nil {
			t.Fatal(err)
		}
		makeProj := func(wBytes []uint8, wType compute.GGMLQuantType) compute.Value {
			bpr := (K / wType.ValuesPerBlock()) * wType.BytesPerBlock()
			wConst, cErr := fn.Constant(wBytes, N, bpr)
			if cErr != nil {
				t.Fatal(cErr)
			}
			y, dErr := fn.FusedQuantizedDense(xParam, wConst, nil, compute.FusedQuantizedDenseConfig{
				WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: wType},
				QuantizedMatmul:    true,
				WeightLayout:       compute.DenseLayoutOutputsInput,
			})
			if dErr != nil {
				t.Fatal(dErr)
			}
			return y
		}
		yQ := makeProj(wQ, compute.GGMLQ4_0)
		yK := makeProj(wK, compute.GGMLQ5_0)
		yV := makeProj(wV, compute.GGMLQ8_0)
		if err := fn.Return([]compute.Value{yQ, yK, yV}, nil); err != nil {
			t.Fatal(err)
		}
		sharedExec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer sharedExec.Finalize()

		for _, M := range []int{1, 7, 64} {
			xData := make([]float32, M*K)
			for i := range xData {
				xData[i] = float32(((i*13+M)%29)-14) * 0.12
			}
			runRef := func(ex compute.Executable) []float32 {
				xBuf, _ := b.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, M, K))
				outs, exErr := ex.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
				if exErr != nil {
					t.Fatal(exErr)
				}
				defer outs[0].Finalize()
				res := make([]float32, M*N)
				_ = outs[0].ToFlatData(res)
				return res
			}
			wantQ := runRef(refQ)
			wantK := runRef(refK)
			wantV := runRef(refV)

			var wg sync.WaitGroup
			for g := range 8 {
				wg.Add(1)
				go func(gIdx int) {
					defer wg.Done()
					xBuf, _ := b.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, M, K))
					outs, exErr := sharedExec.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
					if exErr != nil {
						t.Errorf("goroutine %d Execute(M=%d): %+v", gIdx, M, exErr)
						return
					}
					defer func() {
						for _, o := range outs {
							o.Finalize()
						}
					}()
					gotQ := make([]float32, M*N)
					gotK := make([]float32, M*N)
					gotV := make([]float32, M*N)
					_ = outs[0].ToFlatData(gotQ)
					_ = outs[1].ToFlatData(gotK)
					_ = outs[2].ToFlatData(gotV)
					// Allow small FP32 reassociation delta (~1-2 ULPs at magnitude ~25): on ARM64 SME2
					// hosts, single-threaded reference runs acquire SME (separate FMOPA offset
					// accumulator) while concurrent goroutines may fall back to NEON (interleaved
					// per-sub-block FMLA), differing by up to ~2e-6 on Q8_0 outputs.
					const tolerance = 1e-4
					if ok, diff := testutil.IsInDelta(wantQ, gotQ, tolerance); !ok {
						t.Errorf("M=%d Q mismatch:\n%s", M, diff)
					}
					if ok, diff := testutil.IsInDelta(wantK, gotK, tolerance); !ok {
						t.Errorf("M=%d K mismatch:\n%s", M, diff)
					}
					if ok, diff := testutil.IsInDelta(wantV, gotV, tolerance); !ok {
						t.Errorf("M=%d V mismatch:\n%s", M, diff)
					}
				}(g)
			}
			wg.Wait()
		}
	})

	t.Run("GGML_MixedSubBlockSizes_Q4K_Q5K_Q6K", func(t *testing.T) {
		const M, K, N = 16, 256, 32
		w4K := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ4_K, 11)
		w5K := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ5_K, 22)
		w6K := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ6_K, 33)

		builder := b.Builder("mixed_kquants")
		fn := builder.Main()
		xParam, err := fn.Parameter("x", shapes.Make(dtypes.Float32, M, K), nil)
		if err != nil {
			t.Fatal(err)
		}
		addProj := func(wBytes []uint8, wType compute.GGMLQuantType) compute.Value {
			bpr := (K / wType.ValuesPerBlock()) * wType.BytesPerBlock()
			wConst, cErr := fn.Constant(wBytes, N, bpr)
			if cErr != nil {
				t.Fatal(cErr)
			}
			y, dErr := fn.FusedQuantizedDense(xParam, wConst, nil, compute.FusedQuantizedDenseConfig{
				WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: wType},
				QuantizedMatmul:    true,
				WeightLayout:       compute.DenseLayoutOutputsInput,
			})
			if dErr != nil {
				t.Fatal(dErr)
			}
			return y
		}
		y4 := addProj(w4K, compute.GGMLQ4_K)
		y5 := addProj(w5K, compute.GGMLQ5_K)
		y6 := addProj(w6K, compute.GGMLQ6_K)
		if err := fn.Return([]compute.Value{y4, y5, y6}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		xData := make([]float32, M*K)
		for i := range xData {
			xData[i] = float32(((i*7+3)%19)-9) * 0.1
		}
		xBuf, _ := b.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, M, K))
		outs, err := exec.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, o := range outs {
			o.Finalize()
		}
	})

	t.Run("QuantLinear_ConcurrentSiblings", func(t *testing.T) {
		const M, K, N, G = 32, 64, 32, 32
		xData := make([]float32, M*K)
		for i := range xData {
			xData[i] = float32(((i*11+5)%23)-11) * 0.1
		}
		makeLinearWeights := func(seed int) ([]int8, []float32) {
			w := make([]int8, K*N)
			for i := range w {
				w[i] = int8(((i*17 + seed) % 15) - 7)
			}
			sc := make([]float32, (K/G)*N)
			for i := range sc {
				sc[i] = 0.03 + float32((i+seed)%5)*0.01
			}
			return w, sc
		}
		w1, sc1 := makeLinearWeights(1)
		w2, sc2 := makeLinearWeights(2)

		builder := b.Builder("shared_linear")
		fn := builder.Main()
		xParam, err := fn.Parameter("x", shapes.Make(dtypes.Float32, M, K), nil)
		if err != nil {
			t.Fatal(err)
		}
		addLin := func(w []int8, sc []float32) compute.Value {
			wC, _ := fn.Constant(w, K, N)
			scC, _ := fn.Constant(sc, K/G, N)
			y, dErr := fn.FusedQuantizedDense(xParam, wC, nil, compute.FusedQuantizedDenseConfig{
				WeightQuantization: &compute.Quantization{
					Scheme:    compute.QuantLinear,
					Scale:     scC,
					BlockAxis: 0,
					BlockSize: G,
				},
				QuantizedMatmul: true,
				WeightLayout:    compute.DenseLayoutInputOutputs,
			})
			if dErr != nil {
				t.Fatal(dErr)
			}
			return y
		}
		y1 := addLin(w1, sc1)
		y2 := addLin(w2, sc2)
		if err := fn.Return([]compute.Value{y1, y2}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		var wg sync.WaitGroup
		for range 8 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				xBuf, _ := b.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, M, K))
				outs, exErr := exec.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
				if exErr != nil {
					t.Errorf("Execute: %+v", exErr)
					return
				}
				for _, o := range outs {
					o.Finalize()
				}
			}()
		}
		wg.Wait()
	})
}

func TestQuantizedCacheOption(t *testing.T) {
	const M, K, N = 8, 64, 32
	wBytes := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ4_0, 7)
	bpr := (K / compute.GGMLQ4_0.ValuesPerBlock()) * compute.GGMLQ4_0.BytesPerBlock()
	xData := make([]float32, M*K)
	for i := range xData {
		xData[i] = float32(((i*13+3)%19)-9) * 0.1
	}

	runWithConfig := func(t *testing.T, cfgStr string, wantPolicy gobackend.QuantizedCachePolicy, wantCached bool) []float32 {
		t.Helper()
		bRaw, err := gobackend.New(cfgStr)
		if err != nil {
			t.Fatalf("gobackend.New(%q): %+v", cfgStr, err)
		}
		defer bRaw.Finalize()
		be := bRaw.(*gobackend.Backend)
		if be.QuantizedCache != wantPolicy {
			t.Fatalf("config %q: got QuantizedCache=%v, want %v", cfgStr, be.QuantizedCache, wantPolicy)
		}

		builder := bRaw.Builder("qcache_" + cfgStr)
		fn := builder.Main()
		xParam, err := fn.Parameter("x", shapes.Make(dtypes.Float32, M, K), nil)
		if err != nil {
			t.Fatal(err)
		}
		wConst, err := fn.Constant(wBytes, N, bpr)
		if err != nil {
			t.Fatal(err)
		}
		y, err := fn.FusedQuantizedDense(xParam, wConst, nil, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_0},
			QuantizedMatmul:    true,
			WeightLayout:       compute.DenseLayoutOutputsInput,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		qdNode := y.(*gobackend.Node)
		qdData := qdNode.Data.(*nodeFusedQuantizedDense)
		if (qdData.wConstantData != nil) != wantCached {
			t.Fatalf("config %q after Compile: got wConstantData!=nil == %v, want %v", cfgStr, qdData.wConstantData != nil, wantCached)
		}

		// Run 8 concurrent executions to verify cold-start caching race safety when wantCached==true.
		var wg sync.WaitGroup
		results := make([][]float32, 8)
		for g := range 8 {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				xBuf, _ := bRaw.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, M, K))
				outs, exErr := exec.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
				if exErr != nil {
					t.Errorf("Execute: %+v", exErr)
					return
				}
				defer outs[0].Finalize()
				res := make([]float32, M*N)
				_ = outs[0].ToFlatData(res)
				results[idx] = res
			}(g)
		}
		wg.Wait()

		if gotCached := qdData.IsInputCached(1); gotCached != wantCached {
			t.Fatalf("config %q after Execute: IsInputCached(1)=%v, want %v", cfgStr, gotCached, wantCached)
		}
		for g := 1; g < 8; g++ {
			if ok, diff := testutil.IsInDelta(results[0], results[g], 1e-6); !ok {
				t.Fatalf("config %q goroutine %d mismatch:\n%s", cfgStr, g, diff)
			}
		}
		return results[0]
	}

	gotNever := runWithConfig(t, "quantized_cache=never", gobackend.QuantizedCacheNever, false)
	for _, tc := range []struct {
		cfg        string
		wantPolicy gobackend.QuantizedCachePolicy
		wantCached bool
	}{
		{"", gobackend.QuantizedCacheBalanced, true},
		{"quantized_cache=balanced", gobackend.QuantizedCacheBalanced, true},
		{"quantized_cache", gobackend.QuantizedCacheAlways, true},
		{"quantized_cache=always", gobackend.QuantizedCacheAlways, true},
		{"quantized_cache=true", gobackend.QuantizedCacheAlways, true},
		{"quantized_cache=1", gobackend.QuantizedCacheAlways, true},
		{"quantized_cache=none", gobackend.QuantizedCacheNever, false},
		{"quantized_cache=false", gobackend.QuantizedCacheNever, false},
		{"quantized_cache=0", gobackend.QuantizedCacheNever, false},
	} {
		got := runWithConfig(t, tc.cfg, tc.wantPolicy, tc.wantCached)
		if ok, diff := testutil.IsInDelta(gotNever, got, 1e-6); !ok {
			t.Fatalf("never vs %q mismatch:\n%s", tc.cfg, diff)
		}
	}
}

func TestQuantizedFloatChunkedMatmul(t *testing.T) {
	bRaw, err := gobackend.New("")
	if err != nil {
		t.Fatal(err)
	}
	defer bRaw.Finalize()
	be := bRaw.(*gobackend.Backend)

	for _, M := range []int{1, 37} {
		const K, N = 64, 100 // 64*100 = 6400 > maxScratchFloats(256), forcing multi-chunk N and M!
		xData := make([]float32, M*K)
		for i := range xData {
			xData[i] = float32(((i*13+5)%23)-11) * 0.05
		}
		biasData := make([]float32, N)
		for i := range biasData {
			biasData[i] = float32((i%9)-4) * 0.1
		}

		// 1. QuantGGML (Q4_0) single-chunk vs multi-chunk (maxScratchFloats=256)
		wGGML := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ4_0, 11)
		bpr := (K / compute.GGMLQ4_0.ValuesPerBlock()) * compute.GGMLQ4_0.BytesPerBlock()
		dataGGML := &nodeFusedQuantizedDense{
			weightScheme:   compute.QuantGGML,
			weightGGMLType: compute.GGMLQ4_0,
			weightLayout:   compute.DenseLayoutOutputsInput,
			activation:     compute.ActivationConfig{Type: compute.ActivationGelu},
		}
		outSingle := make([]float32, M*N)
		outChunked := make([]float32, M*N)
		actSingle, err := execQuantizedDenseGGMLFloatMatmul(be, dataGGML, xData, wGGML, biasData, outSingle, M, K, N, bpr, compute.GGMLQ4_0, 0)
		if err != nil || !actSingle {
			t.Fatalf("M=%d single GGML float matmul: act=%v, err=%+v", M, actSingle, err)
		}
		actChunked, err := execQuantizedDenseGGMLFloatMatmul(be, dataGGML, xData, wGGML, biasData, outChunked, M, K, N, bpr, compute.GGMLQ4_0, 256)
		if err != nil || !actChunked {
			t.Fatalf("M=%d chunked GGML float matmul: act=%v, err=%+v", M, actChunked, err)
		}
		if ok, diff := testutil.IsInDelta(outSingle, outChunked, 1e-3); !ok {
			t.Fatalf("M=%d GGML single vs chunked mismatch:\n%s", M, diff)
		}

		// 2. QuantLinear (Int4, InputOutputs & OutputsInput) + QuantNF4 single vs multi-chunk
		for _, layout := range []compute.DenseLayout{compute.DenseLayoutInputOutputs, compute.DenseLayoutOutputsInput} {
			for _, scheme := range []compute.QuantizationScheme{compute.QuantLinear, compute.QuantNF4} {
				var d0, d1, blockAxis int
				if layout == compute.DenseLayoutInputOutputs {
					d0, d1, blockAxis = K, N, 0
				} else {
					d0, d1, blockAxis = N, K, 1
				}
				const G = 16
				dt := dtypes.Int4
				if scheme == compute.QuantNF4 {
					dt = dtypes.Uint4
				}
				packedBytes := make([]byte, (d0*d1+1)/2)
				for i := range packedBytes {
					packedBytes[i] = byte((i*37 + 19) & 0xFF)
				}
				wBuf := gobackend.NewBuffer(shapes.Make(dt, d0, d1))
				copy(wBuf.Flat.([]byte), packedBytes)

				var sShape shapes.Shape
				if layout == compute.DenseLayoutInputOutputs {
					sShape = shapes.Make(dtypes.Float32, K/G, N)
				} else {
					sShape = shapes.Make(dtypes.Float32, N, K/G)
				}
				scales := make([]float32, sShape.Size())
				for i := range scales {
					scales[i] = 0.02 + float32(i%7)*0.01
				}
				var zps []float32
				var zpShape shapes.Shape
				if scheme == compute.QuantLinear {
					zpShape = sShape
					zps = make([]float32, zpShape.Size())
					for i := range zps {
						zps[i] = float32((i % 5) - 2)
					}
				}
				info, err := shapeinference.FusedQuantizedDenseWithInfo(
					shapes.Make(dtypes.Float32, M, K),
					wBuf.RawShape,
					shapes.Make(dtypes.Float32, N),
					shapeinference.QuantizedDenseConfigShapes{
						WeightQuantization: &shapeinference.QuantizationShapes{
							Scheme:         scheme,
							Scale:          sShape,
							ZeroPoint:      zpShape,
							BlockAxis:      blockAxis,
							BlockSize:      G,
							QuantizedDType: dt,
						},
						Activation:   compute.ActivationConfig{Type: compute.ActivationGelu},
						WeightLayout: layout,
					},
				)
				if err != nil {
					t.Fatal(err)
				}
				nd := &nodeFusedQuantizedDense{
					weightScheme: scheme,
					weightLayout: layout,
					activation:   compute.ActivationConfig{Type: compute.ActivationGelu},
					info:         info,
				}
				linSingle := make([]float32, M*N)
				linChunked := make([]float32, M*N)
				if _, err := execQuantizedDenseLinearOrNF4FloatMatmul(be, nd, xData, wBuf, scales, zps, biasData, linSingle, M, K, N, 0); err != nil {
					t.Fatal(err)
				}
				if _, err := execQuantizedDenseLinearOrNF4FloatMatmul(be, nd, xData, wBuf, scales, zps, biasData, linChunked, M, K, N, 256); err != nil {
					t.Fatal(err)
				}
				if ok, diff := testutil.IsInDelta(linSingle, linChunked, 1e-3); !ok {
					t.Fatalf("M=%d layout=%v scheme=%v single vs chunked mismatch:\n%s", M, layout, scheme, diff)
				}

				// 3. VJP chunked vs single-chunk
				if M >= 2 {
					dZ := make([]float32, M*N)
					for i := range dZ {
						dZ[i] = float32(((i*7+3)%19)-9) * 0.04
					}
					dxSingle := make([]float32, M*K)
					dxChunked := make([]float32, M*K)
					sL := info.WeightScaleLayout
					zpL := info.WeightZeroPointLayout
					if layout == compute.DenseLayoutInputOutputs {
						if err := execQuantizedVJPInputOutputsMatmul(be, dZ, dxSingle, M, K, N, 0, func(k0, k1 int, dst []float32) {
							dequantLinearOrNF4RowsChunk(be, scheme, wBuf, scales, zps, k0, k1, N, sL, zpL, dst)
						}); err != nil {
							t.Fatal(err)
						}
						if err := execQuantizedVJPInputOutputsMatmul(be, dZ, dxChunked, M, K, N, 256, func(k0, k1 int, dst []float32) {
							dequantLinearOrNF4RowsChunk(be, scheme, wBuf, scales, zps, k0, k1, N, sL, zpL, dst)
						}); err != nil {
							t.Fatal(err)
						}
					} else {
						if err := execQuantizedVJPOutputsInputMatmul(be, dZ, dxSingle, M, K, N, 0, func(n0, n1 int, dst []float32) {
							dequantLinearOrNF4RowsChunk(be, scheme, wBuf, scales, zps, n0, n1, K, sL, zpL, dst)
						}); err != nil {
							t.Fatal(err)
						}
						if err := execQuantizedVJPOutputsInputMatmul(be, dZ, dxChunked, M, K, N, 256, func(n0, n1 int, dst []float32) {
							dequantLinearOrNF4RowsChunk(be, scheme, wBuf, scales, zps, n0, n1, K, sL, zpL, dst)
						}); err != nil {
							t.Fatal(err)
						}
					}
					if ok, diff := testutil.IsInDelta(dxSingle, dxChunked, 1e-3); !ok {
						t.Fatalf("VJP M=%d layout=%v scheme=%v single vs chunked mismatch:\n%s", M, layout, scheme, diff)
					}
				}
			}
		}
	}
}

func TestQuantizedFloatModeACacheAndDynamic(t *testing.T) {
	const kQ4K, N = 256, 48
	wQ4K := makeTestGGMLWeightMatrix(N, kQ4K, compute.GGMLQ4_K, 23)
	bpr := (kQ4K / compute.GGMLQ4_K.ValuesPerBlock()) * compute.GGMLQ4_K.BytesPerBlock()
	biasData := make([]float32, N)
	for i := range biasData {
		biasData[i] = float32((i%7)-3) * 0.05
	}

	runModeA := func(t *testing.T, cfgStr string, batchSizes []int) [][]float32 {
		t.Helper()
		bRaw, err := gobackend.New(cfgStr)
		if err != nil {
			t.Fatal(err)
		}
		defer bRaw.Finalize()

		builder := bRaw.Builder("modea_" + cfgStr)
		fn := builder.Main()
		xParam, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, kQ4K}, []string{"batch", ""}), nil)
		if err != nil {
			t.Fatal(err)
		}
		wConst, err := fn.Constant(wQ4K, N, bpr)
		if err != nil {
			t.Fatal(err)
		}
		bConst, err := fn.Constant(biasData, N)
		if err != nil {
			t.Fatal(err)
		}
		y, err := fn.FusedQuantizedDense(xParam, wConst, bConst, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_K},
			QuantizedMatmul:    false,
			Activation:         compute.ActivationConfig{Type: compute.ActivationGelu},
			WeightLayout:       compute.DenseLayoutOutputsInput,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		outByBatch := make([][]float32, len(batchSizes))
		for bIdx, M := range batchSizes {
			xData := make([]float32, M*kQ4K)
			for i := range xData {
				xData[i] = float32(((i*17+3)%29)-14) * 0.03
			}

			// Execute concurrently on cold start + warm path across changing dynamic M.
			var wg sync.WaitGroup
			results := make([][]float32, 6)
			for g := range 6 {
				wg.Add(1)
				go func(idx int) {
					defer wg.Done()
					xBuf, _ := bRaw.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, M, kQ4K))
					outs, exErr := exec.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
					if exErr != nil {
						t.Errorf("Execute: %+v", exErr)
						return
					}
					defer outs[0].Finalize()
					res := make([]float32, M*N)
					_ = outs[0].ToFlatData(res)
					results[idx] = res
				}(g)
			}
			wg.Wait()
			// Concurrent executions may dynamically switch between LargeMatMul and SmallMatMul
			// (or SME and NEON) as NumLiveExecutions changes EffectiveParallelism(), shifting
			// FP32 FMA accumulation order over K=256 by a few ULPs (~3e-5 at magnitude ~50).
			for g := 1; g < 6; g++ {
				if ok, diff := testutil.IsInDelta(results[0], results[g], 1e-3); !ok {
					t.Fatalf("cfg=%q M=%d goroutine %d mismatch:\n%s", cfgStr, M, g, diff)
				}
			}
			outByBatch[bIdx] = results[0]
		}
		return outByBatch
	}

	batchSizes := []int{1, 19, 64}
	off := runModeA(t, "quantized_cache=never", batchSizes)
	bal := runModeA(t, "quantized_cache=balanced", batchSizes)
	on := runModeA(t, "quantized_cache=always", batchSizes)
	for i, M := range batchSizes {
		if ok, diff := testutil.IsInDelta(off[i], bal[i], 1e-3); !ok {
			t.Fatalf("Mode A M=%d never vs balanced mismatch:\n%s", M, diff)
		}
		if ok, diff := testutil.IsInDelta(off[i], on[i], 1e-3); !ok {
			t.Fatalf("Mode A M=%d never vs always mismatch:\n%s", M, diff)
		}
	}
}

func TestQuantizedCacheBalancedPolicy(t *testing.T) {
	checkGGML := func(t *testing.T, cfg string, wType compute.GGMLQuantType, K, N int, quantMatmul, donate bool, wantCached, wantRawReleased bool) {
		t.Helper()
		bRaw, err := gobackend.New(cfg)
		if err != nil {
			t.Fatalf("gobackend.New(%q): %+v", cfg, err)
		}
		defer bRaw.Finalize()

		bpr := (K / wType.ValuesPerBlock()) * wType.BytesPerBlock()
		wBytes := makeTestGGMLWeightMatrix(N, K, wType, 5)
		wBuf, err := bRaw.BufferFromFlatData(0, wBytes, shapes.Make(dtypes.Uint8, N, bpr))
		if err != nil {
			t.Fatal(err)
		}
		if !donate {
			defer wBuf.Finalize()
		}

		builder := bRaw.Builder("balanced_check")
		fn := builder.Main()
		xParam, err := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, K), nil)
		if err != nil {
			t.Fatal(err)
		}
		wConst, err := fn.ConstantFromBuffer(wBuf, donate)
		if err != nil {
			t.Fatal(err)
		}
		wCD := wConst.(*gobackend.Node).Data.(*gobackend.ConstantData)
		y, err := fn.FusedQuantizedDense(xParam, wConst, nil, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: wType},
			QuantizedMatmul:    quantMatmul,
			WeightLayout:       compute.DenseLayoutOutputsInput,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := fn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		qdData := y.(*gobackend.Node).Data.(*nodeFusedQuantizedDense)
		if got := qdData.weightIsCached(); got != wantCached {
			t.Fatalf("cfg=%q type=%s K=%d N=%d qmm=%v donate=%v: weightIsCached=%v, want %v",
				cfg, wType, K, N, quantMatmul, donate, got, wantCached)
		}

		xData := make([]float32, 2*K)
		for i := range xData {
			xData[i] = float32((i%11)-5) * 0.05
		}
		xBuf, err := bRaw.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, 2, K))
		if err != nil {
			t.Fatal(err)
		}
		outs, err := exec.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
		if err != nil {
			t.Fatalf("Execute: %+v", err)
		}
		outs[0].Finalize()

		if got := qdData.IsInputCached(1); got != wantCached {
			t.Fatalf("cfg=%q type=%s: IsInputCached(1)=%v, want %v", cfg, wType, got, wantCached)
		}
		gotRawReleased := wCD.Buffer() == nil
		if gotRawReleased != wantRawReleased {
			t.Fatalf("cfg=%q type=%s donate=%v: rawReleased=%v, want %v", cfg, wType, donate, gotRawReleased, wantRawReleased)
		}
	}

	t.Run("RatioCap_Q2K_ModeB", func(t *testing.T) {
		// Q2_K (+3.57x extra) exceeds 2.0x ratio: skipped in balanced, cached in always.
		checkGGML(t, "quantized_cache=balanced", compute.GGMLQ2_K, 256, 32, true, true, false, false)
		checkGGML(t, "quantized_cache=always", compute.GGMLQ2_K, 256, 32, true, true, true, true)
		checkGGML(t, "quantized_cache=never", compute.GGMLQ2_K, 256, 32, true, true, false, false)
	})

	t.Run("RatioCap_ModeA_Float32", func(t *testing.T) {
		// Mode A Q4_K (+6.11x extra) exceeds 2.0x ratio: skipped in balanced, cached & raw freed in always.
		checkGGML(t, "quantized_cache=balanced", compute.GGMLQ4_K, 256, 32, false, true, false, false)
		checkGGML(t, "quantized_cache=always", compute.GGMLQ4_K, 256, 32, false, true, true, true)
		checkGGML(t, "quantized_cache=never", compute.GGMLQ4_K, 256, 32, false, true, false, false)
	})

	t.Run("PerWeightSizeCap_4MiB", func(t *testing.T) {
		// Q8_0 with K=1024, N=3584 -> 224 strips * 32 subblocks * 640 B = 4,587,520 B (> 4 MiB),
		// even though net extra ratio is only +0.18x!
		checkGGML(t, "quantized_cache=balanced", compute.GGMLQ8_0, 1024, 3584, true, true, false, false)
		checkGGML(t, "quantized_cache=always", compute.GGMLQ8_0, 1024, 3584, true, true, true, true)
	})

	t.Run("DonatedVsNonDonated", func(t *testing.T) {
		// Q4_0 donated: +1.22x extra <= 2.0x -> cached and raw released in balanced.
		checkGGML(t, "quantized_cache=balanced", compute.GGMLQ4_0, 64, 32, true, true, true, true)
		// Q4_0 non-donated: raw not freed -> +2.22x extra > 2.0x -> skipped in balanced, cached (raw kept) in always.
		checkGGML(t, "quantized_cache=balanced", compute.GGMLQ4_0, 64, 32, true, false, false, false)
		checkGGML(t, "quantized_cache=always", compute.GGMLQ4_0, 64, 32, true, false, true, false)
		// Q8_0 non-donated: +1.18x extra <= 2.0x -> cached (raw kept!) in balanced.
		checkGGML(t, "quantized_cache=balanced", compute.GGMLQ8_0, 64, 32, true, false, true, false)
	})
}

func TestQuantizedCacheQuantLinearAndSupportConstants(t *testing.T) {
	const K, N, G = 64, 32, 32
	packedBytes := make([]byte, (N*K)/2)
	for i := range packedBytes {
		packedBytes[i] = byte((i*37 + 11) & 0xFF)
	}
	scaleF16 := make([]float16.Float16, N*(K/G))
	for i := range scaleF16 {
		scaleF16[i] = float16.FromFloat32(0.02 + float32(i%5)*0.01)
	}
	zpI8 := make([]int8, N*(K/G))
	for i := range zpI8 {
		zpI8[i] = int8((i % 5) - 2)
	}
	biasF16 := make([]float16.Float16, N)
	for i := range biasF16 {
		biasF16[i] = float16.FromFloat32(float32((i%7)-3) * 0.05)
	}

	runTwoGraphs := func(t *testing.T, cfg string, quantMatmul bool) (
		outM1, outM2 []float32,
		wRawFreed, scaleRawFreed, zpRawFreed, biasRawFreed bool,
		wCached, scaleCached, zpCached, biasCached bool,
	) {
		t.Helper()
		bRaw, err := gobackend.New(cfg)
		if err != nil {
			t.Fatalf("gobackend.New(%q): %+v", cfg, err)
		}
		defer bRaw.Finalize()

		wBuf, _ := bRaw.BufferFromFlatData(0, packedBytes, shapes.Make(dtypes.Int4, N, K))
		sBuf, _ := bRaw.BufferFromFlatData(0, scaleF16, shapes.Make(dtypes.Float16, N, K/G))
		zpBuf, _ := bRaw.BufferFromFlatData(0, zpI8, shapes.Make(dtypes.Int8, N, K/G))
		biasBuf, _ := bRaw.BufferFromFlatData(0, biasF16, shapes.Make(dtypes.Float16, N))

		// Graph 1 (static M=4, Float16 x + Float16 bias): creates donated shared constants.
		b1 := bRaw.Builder("g1")
		fn1 := b1.Main()
		x1, _ := fn1.Parameter("x", shapes.Make(dtypes.Float16, 4, K), nil)
		wC1, err := fn1.ConstantFromBuffer(wBuf, true)
		if err != nil {
			t.Fatal(err)
		}
		sC1, err := fn1.ConstantFromBuffer(sBuf, true)
		if err != nil {
			t.Fatal(err)
		}
		zpC1, err := fn1.ConstantFromBuffer(zpBuf, true)
		if err != nil {
			t.Fatal(err)
		}
		biasC1, err := fn1.ConstantFromBuffer(biasBuf, true)
		if err != nil {
			t.Fatal(err)
		}
		qCfg1 := compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{
				Scheme:         compute.QuantLinear,
				Scale:          sC1,
				ZeroPoint:      zpC1,
				BlockAxis:      1,
				BlockSize:      G,
				QuantizedDType: dtypes.Int4,
			},
			QuantizedMatmul: quantMatmul,
			WeightLayout:    compute.DenseLayoutOutputsInput,
		}
		y1, err := fn1.FusedQuantizedDense(x1, wC1, biasC1, qCfg1)
		if err != nil {
			t.Fatal(err)
		}
		_ = fn1.Return([]compute.Value{y1}, nil)
		exec1, err := b1.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec1.Finalize()

		makeXF16 := func(M int) []float16.Float16 {
			x := make([]float16.Float16, M*K)
			for i := range x {
				x[i] = float16.FromFloat32(float32(((i*13 + 7) % 19) - 9) * 0.04)
			}
			return x
		}
		readOutF32 := func(buf compute.Buffer, n int) []float32 {
			f16 := make([]float16.Float16, n)
			_ = buf.ToFlatData(f16)
			out := make([]float32, n)
			for i, v := range f16 {
				out[i] = v.Float32()
			}
			return out
		}

		// Execute Graph 1 concurrently on cold start.
		x1Data := makeXF16(4)
		var wg sync.WaitGroup
		res1 := make([][]float32, 6)
		for g := range 6 {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				xb, _ := bRaw.BufferFromFlatData(0, x1Data, shapes.Make(dtypes.Float16, 4, K))
				outs, exErr := exec1.Execute([]compute.Buffer{xb}, []bool{true}, 0)
				if exErr != nil {
					t.Errorf("exec1: %+v", exErr)
					return
				}
				defer outs[0].Finalize()
				res1[idx] = readOutF32(outs[0], 4*N)
			}(g)
		}
		wg.Wait()
		for g := 1; g < 6; g++ {
			if ok, diff := testutil.IsInDelta(res1[0], res1[g], 1e-3); !ok {
				t.Fatalf("cfg=%q g1 goroutine %d mismatch:\n%s", cfg, g, diff)
			}
		}

		qd1 := y1.(*gobackend.Node).Data.(*nodeFusedQuantizedDense)
		wCD := wC1.(*gobackend.Node).Data.(*gobackend.ConstantData)
		sCD := sC1.(*gobackend.Node).Data.(*gobackend.ConstantData)
		zpCD := zpC1.(*gobackend.Node).Data.(*gobackend.ConstantData)
		biasCD := biasC1.(*gobackend.Node).Data.(*gobackend.ConstantData)

		wCached = qd1.IsInputCached(1)
		scaleCached = qd1.IsInputCached(qd1.wScaleIdx)
		zpCached = qd1.IsInputCached(qd1.wZPIdx)
		biasCached = qd1.IsInputCached(qd1.biasIdx)
		wRawFreed = wCD.Buffer() == nil
		scaleRawFreed = sCD.Buffer() == nil
		zpRawFreed = zpCD.Buffer() == nil
		biasRawFreed = biasCD.Buffer() == nil

		// Graph 2 (different static M=9): shares the EXACT SAME donated constants across graphs!
		b2 := bRaw.Builder("g2")
		fn2 := b2.Main()
		x2, _ := fn2.Parameter("x", shapes.Make(dtypes.Float16, 9, K), nil)
		qCfg2 := compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{
				Scheme:         compute.QuantLinear,
				Scale:          sC1,
				ZeroPoint:      zpC1,
				BlockAxis:      1,
				BlockSize:      G,
				QuantizedDType: dtypes.Int4,
			},
			QuantizedMatmul: quantMatmul,
			WeightLayout:    compute.DenseLayoutOutputsInput,
		}
		y2, err := fn2.FusedQuantizedDense(x2, wC1, biasC1, qCfg2)
		if err != nil {
			t.Fatal(err)
		}
		_ = fn2.Return([]compute.Value{y2}, nil)
		exec2, err := b2.Compile()
		if err != nil {
			t.Fatalf("cfg=%q g2 Compile failed (cross-graph shared constant reuse): %+v", cfg, err)
		}
		defer exec2.Finalize()

		x2Buf, _ := bRaw.BufferFromFlatData(0, makeXF16(9), shapes.Make(dtypes.Float16, 9, K))
		outs2, err := exec2.Execute([]compute.Buffer{x2Buf}, []bool{true}, 0)
		if err != nil {
			t.Fatalf("cfg=%q g2 Execute failed: %+v", cfg, err)
		}
		defer outs2[0].Finalize()
		out2 := readOutF32(outs2[0], 9*N)

		return res1[0], out2, wRawFreed, scaleRawFreed, zpRawFreed, biasRawFreed, wCached, scaleCached, zpCached, biasCached
	}

	t.Run("QuantLinear_ModeB2_BalancedAndAlways_CrossGraph", func(t *testing.T) {
		neverM1, neverM2, wF, sF, zpF, bF, wC, sC, zpC, bC := runTwoGraphs(t, "quantized_cache=never", true)
		if wC || sC || zpC || bC || wF || sF || zpF || bF {
			t.Fatalf("never: expected nothing cached/freed, got cached=(%v,%v,%v,%v) freed=(%v,%v,%v,%v)",
				wC, sC, zpC, bC, wF, sF, zpF, bF)
		}

		for _, cfg := range []string{"quantized_cache=balanced", "quantized_cache=always"} {
			m1, m2, wF, sF, zpF, bF, wC, sC, zpC, bC := runTwoGraphs(t, cfg, true)
			if !wC || !sC || !zpC || !bC || !wF || !sF || !zpF || !bF {
				t.Fatalf("%s: expected weight+scale+zp+bias all cached & raw freed, got cached=(%v,%v,%v,%v) freed=(%v,%v,%v,%v)",
					cfg, wC, sC, zpC, bC, wF, sF, zpF, bF)
			}
			if ok, diff := testutil.IsInDelta(neverM1, m1, 1e-4); !ok {
				t.Fatalf("%s M1 mismatch:\n%s", cfg, diff)
			}
			if ok, diff := testutil.IsInDelta(neverM2, m2, 1e-4); !ok {
				t.Fatalf("%s M2 cross-graph mismatch:\n%s", cfg, diff)
			}
		}
	})

	t.Run("QuantLinear_ModeA_BalancedSkipsWeightCachesF16Support_AlwaysCachesAll", func(t *testing.T) {
		neverM1, neverM2, _, _, _, _, _, _, _, _ := runTwoGraphs(t, "quantized_cache=never", false)

		// Under balanced: Mode A weight (Float32 = +6x extra) and Int8->F32 zp (+3x extra) stay uncached,
		// while Float16 scale and Float16 bias (+1x extra) ARE cached & their donated raw buffers freed!
		balM1, balM2, wF, sF, zpF, bF, wC, sC, zpC, bC := runTwoGraphs(t, "quantized_cache=balanced", false)
		if wC || wF || zpC || zpF || !sC || !sF || !bC || !bF {
			t.Fatalf("Mode A balanced: expected w/zp uncached and f16 scale/bias cached+freed, got cached=(%v,%v,%v,%v) freed=(%v,%v,%v,%v)",
				wC, sC, zpC, bC, wF, sF, zpF, bF)
		}
		if ok, diff := testutil.IsInDelta(neverM1, balM1, 1e-3); !ok {
			t.Fatalf("Mode A balanced M1 mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsInDelta(neverM2, balM2, 1e-3); !ok {
			t.Fatalf("Mode A balanced M2 mismatch:\n%s", diff)
		}

		// Under always: Mode A weight, baked scale & zp, and f16 bias ALL cache, release raw, and share across graphs!
		allM1, allM2, wF, sF, zpF, bF, wC, sC, zpC, bC := runTwoGraphs(t, "quantized_cache=always", false)
		if !wC || !sC || !zpC || !bC || !wF || !sF || !zpF || !bF {
			t.Fatalf("Mode A always: expected all cached & raw freed, got cached=(%v,%v,%v,%v) freed=(%v,%v,%v,%v)",
				wC, sC, zpC, bC, wF, sF, zpF, bF)
		}
		if ok, diff := testutil.IsInDelta(neverM1, allM1, 1e-3); !ok {
			t.Fatalf("Mode A always M1 mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsInDelta(neverM2, allM2, 1e-3); !ok {
			t.Fatalf("Mode A always M2 cross-graph mismatch:\n%s", diff)
		}
	})

	t.Run("QuantLinear_ModeB1_NoWeightCacheAndNoF32ScaleCache", func(t *testing.T) {
		bRaw, err := gobackend.New("quantized_cache=always")
		if err != nil {
			t.Fatal(err)
		}
		defer bRaw.Finalize()

		wI8 := make([]int8, K*N)
		for i := range wI8 {
			wI8[i] = int8((i % 15) - 7)
		}
		wScaleF32 := make([]float32, N)
		for i := range wScaleF32 {
			wScaleF32[i] = 0.05
		}
		xScaleF16 := []float16.Float16{float16.FromFloat32(0.1)}

		wBuf, _ := bRaw.BufferFromFlatData(0, wI8, shapes.Make(dtypes.Int8, K, N))
		wsBuf, _ := bRaw.BufferFromFlatData(0, wScaleF32, shapes.Make(dtypes.Float32, 1, N))
		xsBuf, _ := bRaw.BufferFromFlatData(0, xScaleF16, shapes.Make(dtypes.Float16, 1))
		bBuf, _ := bRaw.BufferFromFlatData(0, biasF16, shapes.Make(dtypes.Float16, N))

		builder := bRaw.Builder("b1")
		fn := builder.Main()
		xParam, _ := fn.Parameter("x", shapes.Make(dtypes.Int8, 2, K), nil)
		wC, _ := fn.ConstantFromBuffer(wBuf, true)
		wsC, _ := fn.ConstantFromBuffer(wsBuf, true)
		xsC, _ := fn.ConstantFromBuffer(xsBuf, true)
		bC, _ := fn.ConstantFromBuffer(bBuf, true)

		y, err := fn.FusedQuantizedDense(xParam, wC, bC, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{
				Scheme:         compute.QuantLinear,
				Scale:          wsC,
				QuantizedDType: dtypes.Int8,
			},
			XQuantization: &compute.Quantization{
				Scheme:         compute.QuantLinear,
				Scale:          xsC,
				QuantizedDType: dtypes.Int8,
			},
			WeightLayout: compute.DenseLayoutInputOutputs,
		})
		if err != nil {
			t.Fatal(err)
		}
		_ = fn.Return([]compute.Value{y}, nil)
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		xData := make([]int8, 2*K)
		for i := range xData {
			xData[i] = int8((i % 9) - 4)
		}
		xBuf, _ := bRaw.BufferFromFlatData(0, xData, shapes.Make(dtypes.Int8, 2, K))
		outs, err := exec.Execute([]compute.Buffer{xBuf}, []bool{true}, 0)
		if err != nil {
			t.Fatal(err)
		}
		outs[0].Finalize()

		qd := y.(*gobackend.Node).Data.(*nodeFusedQuantizedDense)
		if qd.IsInputCached(1) {
			t.Errorf("Mode B1: weight should never trigger caching (no repacking)")
		}
		if qd.IsInputCached(qd.wScaleIdx) {
			t.Errorf("Mode B1: Float32 wScale should never trigger caching (already Float32)")
		}
		if !qd.IsInputCached(qd.xScaleIdx) || !qd.IsInputCached(qd.biasIdx) {
			t.Errorf("Mode B1: Float16 xScale and bias should be cached, got xScale=%v bias=%v",
				qd.IsInputCached(qd.xScaleIdx), qd.IsInputCached(qd.biasIdx))
		}
		if wC.(*gobackend.Node).Data.(*gobackend.ConstantData).Buffer() == nil {
			t.Errorf("Mode B1: uncached weight raw buffer should still be alive")
		}
		if bC.(*gobackend.Node).Data.(*gobackend.ConstantData).Buffer() != nil {
			t.Errorf("Mode B1: donated Float16 bias raw buffer should be released after caching")
		}
	})
}

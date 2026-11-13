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
					if ok, diff := testutil.IsInDelta(wantQ, gotQ, 1e-6); !ok {
						t.Errorf("M=%d Q mismatch:\n%s", M, diff)
					}
					if ok, diff := testutil.IsInDelta(wantK, gotK, 1e-6); !ok {
						t.Errorf("M=%d K mismatch:\n%s", M, diff)
					}
					if ok, diff := testutil.IsInDelta(wantV, gotV, 1e-6); !ok {
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

	runWithConfig := func(t *testing.T, cfgStr string, wantCached bool) []float32 {
		t.Helper()
		bRaw, err := gobackend.New(cfgStr)
		if err != nil {
			t.Fatalf("gobackend.New(%q): %+v", cfgStr, err)
		}
		defer bRaw.Finalize()
		be := bRaw.(*gobackend.Backend)
		if be.QuantizedCache != wantCached {
			t.Fatalf("config %q: got QuantizedCache=%v, want %v", cfgStr, be.QuantizedCache, wantCached)
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
		qdNode := y.(*gobackend.Node)
		qdData := qdNode.Data.(*nodeFusedQuantizedDense)
		if (qdData.wConstantData != nil) != wantCached {
			t.Fatalf("config %q: got wConstantData!=nil == %v, want %v", cfgStr, qdData.wConstantData != nil, wantCached)
		}

		if err := fn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

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

	gotDefaultOff := runWithConfig(t, "", false)
	gotExplicitOff := runWithConfig(t, "quantized_cache=false", false)
	gotCacheOn := runWithConfig(t, "quantized_cache", true)
	gotCacheExplicitOn := runWithConfig(t, "quantized_cache=true", true)

	if ok, diff := testutil.IsInDelta(gotDefaultOff, gotExplicitOff, 1e-6); !ok {
		t.Fatalf("default vs quantized_cache=false mismatch:\n%s", diff)
	}
	if ok, diff := testutil.IsInDelta(gotDefaultOff, gotCacheOn, 1e-6); !ok {
		t.Fatalf("default vs quantized_cache mismatch:\n%s", diff)
	}
	if ok, diff := testutil.IsInDelta(gotDefaultOff, gotCacheExplicitOn, 1e-6); !ok {
		t.Fatalf("default vs quantized_cache=true mismatch:\n%s", diff)
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
		if ok, diff := testutil.IsInDelta(outSingle, outChunked, 1e-5); !ok {
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
				if ok, diff := testutil.IsInDelta(linSingle, linChunked, 1e-5); !ok {
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
					if ok, diff := testutil.IsInDelta(dxSingle, dxChunked, 1e-4); !ok {
						t.Fatalf("VJP M=%d layout=%v scheme=%v single vs chunked mismatch:\n%s", M, layout, scheme, diff)
					}
				}
			}
		}
	}
}

func TestQuantizedFloatModeACacheAndDynamic(t *testing.T) {
	const K, N = 64, 48
	wBytes := makeTestGGMLWeightMatrix(N, K, compute.GGMLQ4_K, 23)
	// Wait: Q4_K requires K % 256 == 0! Let's use K = 256 so Q4_K is tested!
	_ = wBytes
	const kQ4K = 256
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
			for g := 1; g < 6; g++ {
				if ok, diff := testutil.IsInDelta(results[0], results[g], 1e-6); !ok {
					t.Fatalf("cfg=%q M=%d goroutine %d mismatch:\n%s", cfgStr, M, g, diff)
				}
			}
			outByBatch[bIdx] = results[0]
		}
		return outByBatch
	}

	batchSizes := []int{1, 19, 64}
	off := runModeA(t, "", batchSizes)
	on := runModeA(t, "quantized_cache", batchSizes)
	for i, M := range batchSizes {
		if ok, diff := testutil.IsInDelta(off[i], on[i], 1e-4); !ok {
			t.Fatalf("Mode A M=%d cache off vs on mismatch:\n%s", M, diff)
		}
	}
}

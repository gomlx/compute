// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"encoding/binary"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// TestGGMLAllFormatsVecDotVsDequant verifies that every GGML block-integer dot product
// (Mode B: Q8_0 / Q8_K) matches float dequantization + dot product (Mode A) on identical quantized inputs.
func TestGGMLAllFormatsVecDotVsDequant(t *testing.T) {
	allTypes := []compute.GGMLQuantType{
		compute.GGMLQ4_0,
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
				case compute.GGMLQ4_0, compute.GGMLQ8_0, compute.GGMLIQ4NL:
					binary.LittleEndian.PutUint16(wRow[off:off+2], dF16.Bits())
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

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func skipIfQuantizedDenseVJPNotImplemented(t *testing.T, b compute.Backend, err error) {
	t.Helper()
	if err != nil && compute.IsNotImplemented(err) {
		t.Skipf("Skipping unsupported FusedQuantizedDenseVJP configuration on %q: %v", b.Name(), err)
	}
}

func testFusedQuantizedDenseVJP(t *testing.T, b compute.Backend) {
	t.Run("FusedQuantizedDenseVJP", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedQuantizedDenseVJP)

		t.Run("QuantNF4_LayoutsAndBlockAxes", func(t *testing.T) {
			testQDVJPNF4LayoutsAndBlockAxes(t, b)
		})
		t.Run("QuantLinear_LayoutsDTypesZeroPointAndQuantizedMatmul", func(t *testing.T) {
			testQDVJPLinearLayoutsDTypesAndQuantizedMatmul(t, b)
		})
		t.Run("QuantGGML_AllFormatsAndNilInputs", func(t *testing.T) {
			testQDVJPGGMLAllFormatsAndNilInputs(t, b)
		})
		t.Run("Activations_FiniteDifference", func(t *testing.T) {
			testQDVJPActivationsFiniteDifference(t, b)
		})
		t.Run("Ranks_DynamicShapes_And_HalfPrecision", func(t *testing.T) {
			testQDVJPRanksDynamicAndHalfPrecision(t, b)
		})
		t.Run("ValidationErrors_Rejected", func(t *testing.T) {
			testQDVJPValidationErrors(t, b)
		})
	})
}

func testQDVJPNF4LayoutsAndBlockAxes(t *testing.T, b compute.Backend) {
	testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
	testutil.SkipIfMissing(t, b, compute.OpTypeReshape)
	testutil.SkipIfMissingDType(t, b, dtypes.Uint4)

	const M, K, N, G = 3, 16, 8, 8
	x := makeFloat32Matrix(M, K, 0.3)
	nibIO := make([][]uint8, K)
	wDeqIO := make([][]float32, K)
	scalesIO := make([][]float32, K/G)
	for kb := range K / G {
		scalesIO[kb] = makeFloat32PositiveVec(N, 0.2, 0.02)
	}
	for k := range K {
		nibIO[k] = make([]uint8, N)
		wDeqIO[k] = make([]float32, N)
		for n := range N {
			code := uint8((k*3 + n*7 + 1) & 0x0F)
			nibIO[k][n] = code
			wDeqIO[k][n] = compute.NF4LookupTable[code] * scalesIO[k/G][n]
		}
	}
	nibOI := transposeUint8Matrix(nibIO)
	packedIO := pack4BitMatrix(nibIO)
	packedOI := pack4BitMatrix(nibOI)
	scalesOI := transposeFloat32Matrix(scalesIO)
	bias := makeFloat32Vec(N, 0.15)
	dOut := makeFloat32Matrix(M, N, 0.25)

	wantDx, wantDBias := refQuantizedDenseVJP2D(x, wDeqIO, bias, dOut, compute.ActivationRelu)

	for _, layout := range []compute.DenseLayout{compute.DenseLayoutInputOutputs, compute.DenseLayoutOutputsInput} {
		t.Run(layout.String(), func(t *testing.T) {
			packed := packedIO
			sc := scalesIO
			blockAxis := 0
			d0, d1 := K, N
			if layout == compute.DenseLayoutOutputsInput {
				packed = packedOI
				sc = scalesOI
				blockAxis = 1
				d0, d1 = N, K
			}
			gotDx, gotDBias, err := testutil.Exec2(b, []any{x, packed, sc, bias, dOut},
				func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
					wU4, err := f.Bitcast(p[1], dtypes.Uint4)
					if err != nil {
						return nil, nil, err
					}
					wU4, err = f.Reshape(wU4, d0, d1)
					if err != nil {
						return nil, nil, err
					}
					cfg := compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantNF4,
							Scale:     p[2],
							BlockAxis: blockAxis,
							BlockSize: G,
						},
						Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
						WeightLayout: layout,
					}
					y, err := f.FusedQuantizedDense(p[0], wU4, p[3], cfg)
					if err != nil {
						return nil, nil, err
					}
					// Pass x=nil to also verify memory-saving nil x with non-trivial Relu activation!
					return f.FusedQuantizedDenseVJP(nil, wU4, p[3], y, p[4], cfg)
				})
			skipIfQuantizedDenseVJPNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("NF4 VJP %s failed: %+v", layout, err)
			}
			if ok, diff := testutil.IsInDelta(wantDx, gotDx, quantizedDenseFloat32Tol); !ok {
				t.Errorf("NF4 VJP %s dx mismatch:\n%s", layout, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDBias, gotDBias, quantizedDenseFloat32Tol); !ok {
				t.Errorf("NF4 VJP %s dBias mismatch:\n%s", layout, diff)
			}
		})
	}
}

func testQDVJPLinearLayoutsDTypesAndQuantizedMatmul(t *testing.T, b compute.Backend) {
	// 1. Uint8 OutputsInput with ZeroPoint and nil x, nil y, nil bias
	t.Run("Uint8_OutputsInput_WithZeroPoint_NilXNilYNilBias", func(t *testing.T) {
		const M, K, N, G = 2, 12, 8, 4
		wU8IO := makeUint8Matrix(K, N, 100, 40)
		wU8OI := transposeUint8Matrix(wU8IO)
		scalesOI := make([][]float32, N)
		zpsOI := make([][]uint8, N)
		for n := range N {
			scalesOI[n] = makeFloat32PositiveVec(K/G, 0.05*float32(n+1), 0.02)
			zpsOI[n] = []uint8{120, 124, 128}
		}
		wDeqIO := make([][]float32, K)
		for k := range K {
			wDeqIO[k] = make([]float32, N)
			for n := range N {
				wDeqIO[k][n] = (float32(wU8OI[n][k]) - float32(zpsOI[n][k/G])) * scalesOI[n][k/G]
			}
		}
		dOut := makeFloat32Matrix(M, N, 0.3)
		wantDx, _ := refQuantizedDenseVJP2D(makeFloat32Matrix(M, K, 0), wDeqIO, nil, dOut, compute.ActivationNone)

		gotDx, err := testutil.Exec1(b, []any{wU8OI, scalesOI, zpsOI, dOut},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				dx, dBias, err := f.FusedQuantizedDenseVJP(nil, p[0], nil, nil, p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[1],
						ZeroPoint: p[2],
						BlockAxis: 1,
						BlockSize: G,
					},
					WeightLayout: compute.DenseLayoutOutputsInput,
				})
				if err != nil {
					return nil, err
				}
				if dBias != nil {
					return nil, compute.ErrNotImplemented
				}
				return dx, nil
			})
		skipIfQuantizedDenseVJPNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Uint8 OutputsInput VJP failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(wantDx, gotDx, quantizedDenseFloat32Tol); !ok {
			t.Errorf("Uint8 OutputsInput VJP mismatch:\n%s", diff)
		}
	})

	// 2. Sub-byte Int2 InputOutputs VJP + QuantizedMatmul=true VJP
	t.Run("SubByteInt2_And_QuantizedMatmulHint", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
		testutil.SkipIfMissingDType(t, b, dtypes.Int2)
		const M, K, N = 2, 8, 8
		crumbs := make([][]uint8, K)
		wDeqIO := make([][]float32, K)
		scale := makeFloat32PositiveVec(N, 0.25, 0.05)
		for k := range K {
			crumbs[k] = make([]uint8, N)
			wDeqIO[k] = make([]float32, N)
			for n := range N {
				c := uint8((k*3 + n*5) & 0x03)
				crumbs[k][n] = c
				v := int(c)
				if v >= 2 {
					v -= 4
				}
				wDeqIO[k][n] = float32(v) * scale[n]
			}
		}
		packed := pack2BitMatrix(crumbs)
		dOut := makeFloat32Matrix(M, N, 0.4)
		wantDx, _ := refQuantizedDenseVJP2D(makeFloat32Matrix(M, K, 0), wDeqIO, nil, dOut, compute.ActivationNone)

		gotDx, err := testutil.Exec1(b, []any{packed, scale, dOut},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				wI2, err := f.Bitcast(p[0], dtypes.Int2)
				if err != nil {
					return nil, err
				}
				wI2, err = f.Reshape(wI2, K, N)
				if err != nil {
					return nil, err
				}
				dx, _, err := f.FusedQuantizedDenseVJP(nil, wI2, nil, nil, p[2], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[1]},
					QuantizedMatmul:    true,
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
				return dx, err
			})
		skipIfQuantizedDenseVJPNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Int2 VJP failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(wantDx, gotDx, 0.05); !ok {
			t.Errorf("Int2 VJP mismatch:\n%s", diff)
		}
	})
}

func testQDVJPGGMLAllFormatsAndNilInputs(t *testing.T, b compute.Backend) {
	for _, wType := range allComplianceGGMLTypes {
		t.Run(wType.String(), func(t *testing.T) {
			const M, N = 2, 4
			K := wType.ValuesPerBlock()
			wBytes := make([][]uint8, N)
			wDeqOI := make([][]float32, N)
			for n := range N {
				wBytes[n], wDeqOI[n] = makeTestGGMLWeightRow(wType, 1, n+2)
			}
			wDeqIO := transposeFloat32Matrix(wDeqOI)
			dOut := makeFloat32Matrix(M, N, 0.25)
			wantDx, _ := refQuantizedDenseVJP2D(makeFloat32Matrix(M, K, 0), wDeqIO, nil, dOut, compute.ActivationNone)

			gotDx, err := testutil.Exec1(b, []any{wBytes, dOut},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					dx, dBias, err := f.FusedQuantizedDenseVJP(nil, p[0], nil, nil, p[1], compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:   compute.QuantGGML,
							GGMLType: wType,
						},
						Activation:   compute.ActivationConfig{Type: compute.ActivationNone},
						WeightLayout: compute.DenseLayoutOutputsInput,
					})
					if err != nil {
						return nil, err
					}
					if dBias != nil {
						return nil, compute.ErrNotImplemented
					}
					return dx, nil
				})
			skipIfQuantizedDenseVJPNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("GGML(%s) VJP failed: %+v", wType, err)
			}
			if ok, diff := testutil.IsInDelta(wantDx, gotDx, quantizedDenseFloat32Tol); !ok {
				t.Errorf("GGML(%s) VJP dx mismatch:\n%s", wType, diff)
			}
		})
	}
}

// testQDVJPActivationsFiniteDifference checks analytical VJP gradients against central finite differences
// across all supported output-only-VJP ActivationTypes!
func testQDVJPActivationsFiniteDifference(t *testing.T, b compute.Backend) {
	const M, K, N = 2, 6, 4
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
	dOut := [][]float32{
		{0.5, -0.3, 0.8, -0.2},
		{-0.4, 0.6, -0.1, 0.7},
	}

	// Choose inputs/biases carefully away from non-differentiable kinks at z=0 or z=+/-2.5 (Relu/LeakyRelu/Selu/HardSigmoid).
	xBase := [][]float32{
		{0.35, -0.42, 0.71, -0.23, 0.54, -0.61},
		{-0.52, 0.83, -0.14, 0.41, -0.33, 0.27},
	}
	biasBase := []float32{0.35, -0.45, 0.25, 0.55}

	acts := []compute.ActivationType{
		compute.ActivationNone,
		compute.ActivationRelu,
		compute.ActivationSigmoid,
		compute.ActivationTanh,
		compute.ActivationLeakyRelu,
		compute.ActivationSelu,
		compute.ActivationHardSigmoid,
	}

	for _, actType := range acts {
		t.Run(actType.String(), func(t *testing.T) {
			x := make([][]float32, M)
			for m := range M {
				x[m] = append([]float32(nil), xBase[m]...)
			}
			bias := append([]float32(nil), biasBase...)

			gotDxAny, gotDBiasAny, err := testutil.Exec2(b, []any{x, w, scale, zp, bias, dOut},
				func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
					cfg := compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							ZeroPoint: p[3],
						},
						Activation:   compute.ActivationConfig{Type: actType},
						WeightLayout: compute.DenseLayoutInputOutputs,
					}
					y, err := f.FusedQuantizedDense(p[0], p[1], p[4], cfg)
					if err != nil {
						return nil, nil, err
					}
					return f.FusedQuantizedDenseVJP(p[0], p[1], p[4], y, p[5], cfg)
				})
			skipIfQuantizedDenseVJPNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("FusedQuantizedDenseVJP(%s) failed: %+v", actType, err)
			}
			gotDx := gotDxAny.([][]float32)
			gotDBias := gotDBiasAny.([]float32)

			evalLoss := func(xIn [][]float32, biasIn []float32) float64 {
				gotY, err := testutil.Exec1(b, []any{xIn, w, scale, zp, biasIn},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						return f.FusedQuantizedDense(p[0], p[1], p[4], compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{
								Scheme:    compute.QuantLinear,
								Scale:     p[2],
								ZeroPoint: p[3],
							},
							Activation:   compute.ActivationConfig{Type: actType},
							WeightLayout: compute.DenseLayoutInputOutputs,
						})
					})
				if err != nil {
					t.Fatalf("evalLoss(%s) failed: %+v", actType, err)
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
					if ok, diff := testutil.IsInDelta(numGrad, gotDx[m][k], 3e-3); !ok {
						t.Errorf("%s finite-diff dx[%d,%d]: num=%f analytic=%f\n%s", actType, m, k, numGrad, gotDx[m][k], diff)
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
				if ok, diff := testutil.IsInDelta(numGrad, gotDBias[n], 3e-3); !ok {
					t.Errorf("%s finite-diff dBias[%d]: num=%f analytic=%f\n%s", actType, n, numGrad, gotDBias[n], diff)
				}
			}
		})
	}
}

func testQDVJPRanksDynamicAndHalfPrecision(t *testing.T, b compute.Backend) {
	const K, N = 8, 5
	w := makeInt8Matrix(K, N)
	scale := makeFloat32PositiveVec(N, 0.15, 0.02)
	wDeqIO := make([][]float32, K)
	for k := range K {
		wDeqIO[k] = make([]float32, N)
		for n := range N {
			wDeqIO[k][n] = float32(w[k][n]) * scale[n]
		}
	}

	// 1. 1D [K] and 3D [B, S, K]
	t.Run("Rank1_And_Rank3", func(t *testing.T) {
		dOut1D := makeFloat32Vec(N, 0.3)
		want2D, _ := refQuantizedDenseVJP2D([][]float32{make([]float32, K)}, wDeqIO, nil, [][]float32{dOut1D}, compute.ActivationNone)
		got1D, err := testutil.Exec1(b, []any{w, scale, dOut1D},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				dx, _, err := f.FusedQuantizedDenseVJP(nil, p[0], nil, nil, p[2], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[1]},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
				return dx, err
			})
		skipIfQuantizedDenseVJPNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("1D VJP failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want2D[0], got1D, quantizedDenseFloat32Tol); !ok {
			t.Errorf("1D VJP mismatch:\n%s", diff)
		}

		const B, S = 2, 3
		dOut3D := make([][][]float32, B)
		want3D := make([][][]float32, B)
		var wantDBias []float32 = make([]float32, N)
		bias := makeFloat32Vec(N, 0.1)
		for bi := range B {
			dOut3D[bi] = makeFloat32Matrix(S, N, float32(bi+1)*0.2)
			var db []float32
			want3D[bi], db = refQuantizedDenseVJP2D(makeFloat32Matrix(S, K, 0), wDeqIO, bias, dOut3D[bi], compute.ActivationNone)
			for n := range N {
				wantDBias[n] += db[n]
			}
		}
		got3D, gotDB3D, err := testutil.Exec2(b, []any{w, scale, bias, dOut3D},
			func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
				return f.FusedQuantizedDenseVJP(nil, p[0], p[2], nil, p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[1]},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseVJPNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("3D VJP failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want3D, got3D, quantizedDenseFloat32Tol); !ok {
			t.Errorf("3D VJP dx mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsInDelta(wantDBias, gotDB3D, quantizedDenseFloat32Tol); !ok {
			t.Errorf("3D VJP dBias mismatch:\n%s", diff)
		}
	})

	// 2. Dynamic batch VJP [DynamicDim, K] across M=1 and M=5
	t.Run("DynamicBatch", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		builder := b.Builder("qd_vjp_dyn")
		fn := builder.Main()
		pW, _ := fn.Parameter("w", shapes.Make(dtypes.Int8, K, N), nil)
		pS, _ := fn.Parameter("s", shapes.Make(dtypes.Float32, N), nil)
		pDOut, _ := fn.Parameter("dOut", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, N}, []string{"batch", ""}), nil)
		dx, _, err := fn.FusedQuantizedDenseVJP(nil, pW, nil, nil, pDOut, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: pS},
			WeightLayout:       compute.DenseLayoutInputOutputs,
		})
		skipIfQuantizedDenseVJPNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Dynamic VJP build failed: %+v", err)
		}
		_ = fn.Return([]compute.Value{dx}, nil)
		exec, err := builder.Compile()
		skipIfQuantizedDenseVJPNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Dynamic VJP compile failed: %+v", err)
		}
		defer exec.Finalize()

		for _, M := range []int{1, 5} {
			dOut := makeFloat32Matrix(M, N, 0.25)
			wantDx, _ := refQuantizedDenseVJP2D(makeFloat32Matrix(M, K, 0), wDeqIO, nil, dOut, compute.ActivationNone)
			bufW, _ := testutil.ToBuffer(b, w)
			bufS, _ := testutil.ToBuffer(b, scale)
			bufDOut, _ := testutil.ToBuffer(b, dOut)
			outs, err := exec.Execute([]compute.Buffer{bufW, bufS, bufDOut}, nil, 0)
			if err != nil {
				t.Fatalf("Dynamic VJP Execute(M=%d) failed: %+v", M, err)
			}
			gotDx, _ := testutil.FromBuffer(b, outs[0])
			if ok, diff := testutil.IsInDelta(wantDx, gotDx, quantizedDenseFloat32Tol); !ok {
				t.Errorf("Dynamic VJP M=%d mismatch:\n%s", M, diff)
			}
		}
	})

	// 3. Half-precision Float16 and BFloat16 VJP
	t.Run("Float16_And_BFloat16", func(t *testing.T) {
		const M = 2
		dOutRef := makeFloat32Matrix(M, N, 0.25)

		t.Run("Float16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float16)
			dOutF16 := make([][]float16.Float16, M)
			dOutExact := make([][]float32, M)
			for m := range M {
				dOutF16[m] = make([]float16.Float16, N)
				dOutExact[m] = make([]float32, N)
				for n := range N {
					v := float16.FromFloat32(dOutRef[m][n])
					dOutF16[m][n] = v
					dOutExact[m][n] = v.Float32()
				}
			}
			biasF16 := make([]float16.Float16, N)
			for n := range N {
				biasF16[n] = float16.FromFloat32(0.1)
			}
			wantDxF32, wantDBF32 := refQuantizedDenseVJP2D(makeFloat32Matrix(M, K, 0), wDeqIO, makeFloat32PositiveVec(N, 0.1, 0), dOutExact, compute.ActivationNone)
			wantDx := make([][]float16.Float16, M)
			for m := range M {
				wantDx[m] = make([]float16.Float16, K)
				for k := range K {
					wantDx[m][k] = float16.FromFloat32(wantDxF32[m][k])
				}
			}
			wantDB := make([]float16.Float16, N)
			for n := range N {
				wantDB[n] = float16.FromFloat32(wantDBF32[n])
			}

			gotDx, gotDB, err := testutil.Exec2(b, []any{w, scale, biasF16, dOutF16},
				func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
					return f.FusedQuantizedDenseVJP(nil, p[0], p[2], nil, p[3], compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[1]},
						WeightLayout:       compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseVJPNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("Float16 VJP failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantDx, gotDx, 1e-2); !ok {
				t.Errorf("Float16 VJP dx mismatch:\n%s", diff)
			}
			if ok, diff := testutil.IsInDelta(wantDB, gotDB, 1e-2); !ok {
				t.Errorf("Float16 VJP dBias mismatch:\n%s", diff)
			}
		})

		t.Run("BFloat16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.BFloat16)
			dOutBF16 := make([][]bfloat16.BFloat16, M)
			dOutExact := make([][]float32, M)
			for m := range M {
				dOutBF16[m] = make([]bfloat16.BFloat16, N)
				dOutExact[m] = make([]float32, N)
				for n := range N {
					v := bfloat16.FromFloat32(dOutRef[m][n])
					dOutBF16[m][n] = v
					dOutExact[m][n] = v.Float32()
				}
			}
			biasBF16 := make([]bfloat16.BFloat16, N)
			for n := range N {
				biasBF16[n] = bfloat16.FromFloat32(0.125)
			}
			wantDxF32, wantDBF32 := refQuantizedDenseVJP2D(makeFloat32Matrix(M, K, 0), wDeqIO, makeFloat32PositiveVec(N, 0.125, 0), dOutExact, compute.ActivationNone)
			wantDx := make([][]bfloat16.BFloat16, M)
			for m := range M {
				wantDx[m] = make([]bfloat16.BFloat16, K)
				for k := range K {
					wantDx[m][k] = bfloat16.FromFloat32(wantDxF32[m][k])
				}
			}
			wantDB := make([]bfloat16.BFloat16, N)
			for n := range N {
				wantDB[n] = bfloat16.FromFloat32(wantDBF32[n])
			}

			gotDx, gotDB, err := testutil.Exec2(b, []any{w, scale, biasBF16, dOutBF16},
				func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
					return f.FusedQuantizedDenseVJP(nil, p[0], p[2], nil, p[3], compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[1]},
						WeightLayout:       compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseVJPNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("BFloat16 VJP failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantDx, gotDx, 2e-2); !ok {
				t.Errorf("BFloat16 VJP dx mismatch:\n%s", diff)
			}
			if ok, diff := testutil.IsInDelta(wantDB, gotDB, 2e-2); !ok {
				t.Errorf("BFloat16 VJP dBias mismatch:\n%s", diff)
			}
		})
	})
}

func testQDVJPValidationErrors(t *testing.T, b compute.Backend) {
	builder := b.Builder("qd_vjp_reject_tests")
	fn := builder.Main()
	xVal, _ := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 4), nil)
	wVal, _ := fn.Parameter("w", shapes.Make(dtypes.Int8, 4, 4), nil)
	sVal, _ := fn.Parameter("s", shapes.Make(dtypes.Float32, 4), nil)
	yVal, _ := fn.Parameter("y", shapes.Make(dtypes.Float32, 2, 4), nil)
	dOutVal, _ := fn.Parameter("dOut", shapes.Make(dtypes.Float32, 2, 4), nil)

	// 1. VJPRequiresInput() == true (e.g. Gelu, Silu) must fail at build time.
	for _, act := range []compute.ActivationType{compute.ActivationGelu, compute.ActivationSilu, compute.ActivationSwiGLU} {
		if _, _, err := fn.FusedQuantizedDenseVJP(xVal, wVal, nil, yVal, dOutVal, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: sVal},
			Activation:         compute.ActivationConfig{Type: act},
		}); err == nil {
			t.Errorf("Expected FusedQuantizedDenseVJP to reject activation %s", act)
		}
	}

	// 2. Missing y when Activation != ActivationNone (e.g. Relu) must fail.
	if _, _, err := fn.FusedQuantizedDenseVJP(xVal, wVal, nil, nil, dOutVal, compute.FusedQuantizedDenseConfig{
		WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: sVal},
		Activation:         compute.ActivationConfig{Type: compute.ActivationRelu},
	}); err == nil {
		t.Errorf("Expected FusedQuantizedDenseVJP to require y when Activation is Relu")
	}

	// 3. XQuantization != nil or OutputQuantization != nil must fail (discrete step functions).
	xInt, _ := fn.Parameter("x_int", shapes.Make(dtypes.Int8, 2, 4), nil)
	sScalar, _ := fn.Parameter("s_scalar", shapes.Make(dtypes.Float32), nil)
	if _, _, err := fn.FusedQuantizedDenseVJP(xInt, wVal, nil, yVal, dOutVal, compute.FusedQuantizedDenseConfig{
		WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: sVal},
		XQuantization:      &compute.Quantization{Scheme: compute.QuantLinear, Scale: sScalar},
	}); err == nil {
		t.Errorf("Expected FusedQuantizedDenseVJP to reject XQuantization != nil")
	}
	if _, _, err := fn.FusedQuantizedDenseVJP(xVal, wVal, nil, nil, dOutVal, compute.FusedQuantizedDenseConfig{
		WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: sVal},
		OutputQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: sScalar},
	}); err == nil {
		t.Errorf("Expected FusedQuantizedDenseVJP to reject OutputQuantization != nil")
	}
}

func refQuantizedDenseVJP2D(
	x [][]float32,
	wDeqIO [][]float32,
	bias []float32,
	dOut [][]float32,
	act compute.ActivationType,
) (dx [][]float32, dBias []float32) {
	M, K := len(dOut), len(wDeqIO)
	N := len(wDeqIO[0])
	dx = make([][]float32, M)
	if bias != nil {
		dBias = make([]float32, N)
	}
	for m := range M {
		dx[m] = make([]float32, K)
		dZ := make([]float32, N)
		for n := range N {
			if act == compute.ActivationNone {
				dZ[n] = dOut[m][n]
			} else {
				var z float32
				if bias != nil {
					z = bias[n]
				}
				for k := range K {
					z += x[m][k] * wDeqIO[k][n]
				}
				y := refApplyActivationScalar(act, z)
				switch act {
				case compute.ActivationRelu:
					if y > 0 {
						dZ[n] = dOut[m][n]
					}
				case compute.ActivationSigmoid:
					dZ[n] = dOut[m][n] * y * (1 - y)
				case compute.ActivationTanh:
					dZ[n] = dOut[m][n] * (1 - y*y)
				case compute.ActivationLeakyRelu:
					if y >= 0 {
						dZ[n] = dOut[m][n]
					} else {
						dZ[n] = 0.3 * dOut[m][n]
					}
				}
			}
			if dBias != nil {
				dBias[n] += dZ[n]
			}
		}
		for k := range K {
			var sum float32
			for n := range N {
				sum += dZ[n] * wDeqIO[k][n]
			}
			dx[m][k] = sum
		}
	}
	return dx, dBias
}

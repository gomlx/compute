// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/compute/support/xslices"
)

func skipIfQuantizedDenseNotImplemented(t *testing.T, b compute.Backend, err error) {
	t.Helper()
	if err != nil && compute.IsNotImplemented(err) {
		t.Skipf("Skipping unsupported FusedQuantizedDense configuration on %q: %v", b.Name(), err)
	}
}

func testFusedQuantizedDense(t *testing.T, b compute.Backend) {
	t.Run("FusedQuantizedDense", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedQuantizedDense)

		t.Run("QuantLinear_GranularitiesAndZeroPoints", func(t *testing.T) {
			testQDLinearGranularitiesAndZeroPoints(t, b)
		})
		t.Run("QuantLinear_SubByteDTypesAndLayouts", func(t *testing.T) {
			testQDLinearSubByteDTypesAndLayouts(t, b)
		})
		t.Run("QuantNF4_LayoutsAxesAndDTypes", func(t *testing.T) {
			testQDNF4LayoutsAxesAndDTypes(t, b)
		})
		t.Run("QuantLinear_QuantizedMatmuls", func(t *testing.T) {
			testQDLinearQuantizedMatmuls(t, b)
		})
		t.Run("PreQuantizedX_QuantLinear", func(t *testing.T) {
			testQDPreQuantizedXLinear(t, b)
		})
		t.Run("OutputQuantization", func(t *testing.T) {
			testQDOutputQuantization(t, b)
		})
		t.Run("Ranks_Tiling_And_DynamicShapes", func(t *testing.T) {
			testQDRanksTilingAndDynamicShapes(t, b)
		})
		t.Run("HalfPrecision_And_MixedScaleDTypes", func(t *testing.T) {
			testQDHalfPrecisionAndMixedScaleDTypes(t, b)
		})
		t.Run("Activations_Fusion_Constants_And_Errors", func(t *testing.T) {
			testQDActivationsFusionConstantsAndErrors(t, b)
		})
		t.Run("QuantGGML_AllModesAndFormats", func(t *testing.T) {
			testQDGGMLAllModesAndFormats(t, b)
		})
	})
}

// testQDLinearGranularitiesAndZeroPoints covers QuantLinear across:
//   - Scale shapes: scalar [], [1], 1D [N], 2D BlockAxis=0 (inferred & explicit BlockSize, including non-divisible ceil(K/G)),
//     and 2D BlockAxis=1 (inferred & non-divisible ceil(N/G)).
//   - WeightLayouts: DenseLayoutInputOutputs [K, N] and DenseLayoutOutputsInput [N, K].
//   - Weight DTypes: Int8 and Uint8.
//   - ZeroPoints: nil (symmetric), matching 2D integer ZP, coarser broadcast ZP (scalar [] and 1D [N]),
//     Int32 ZP, and Float32 ZP.
func testQDLinearGranularitiesAndZeroPoints(t *testing.T, b compute.Backend) {
	t.Run("PerTensorScalar_And_1DPerChannel", func(t *testing.T) {
		const M, K, N = 3, 10, 7
		x := makeFloat32Matrix(M, K, 0.25)
		wIO := makeInt8Matrix(K, N)
		wOI := transposeInt8Matrix(wIO)
		bias := makeFloat32Vec(N, 0.1)

		scalarScale := float32(0.0625)
		scalarZP := int8(-2)
		wantScalar := refQuantLinear2D(x, int8ToF32Matrix(wIO), bias, compute.ActivationNone, func(_, _ int) (float32, float32) {
			return scalarScale, float32(scalarZP)
		})

		// Scalar [] Scale + Scalar [] ZeroPoint (InputOutputs)
		t.Run("Scalar_InputOutputs", func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x, wIO, scalarScale, scalarZP, bias},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], p[4], compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							ZeroPoint: p[3],
						},
						WeightLayout: compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("Scalar InputOutputs failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantScalar, got, 1e-5); !ok {
				t.Errorf("Scalar InputOutputs mismatch:\n%s", diff)
			}
		})

		// [1] Scale + nil ZeroPoint (OutputsInput)
		t.Run("Len1Scale_Symmetric_OutputsInput", func(t *testing.T) {
			wantSym := refQuantLinear2D(x, int8ToF32Matrix(wIO), nil, compute.ActivationNone, func(_, _ int) (float32, float32) {
				return scalarScale, 0
			})
			got, err := testutil.Exec1(b, []any{x, wOI, []float32{scalarScale}},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme: compute.QuantLinear,
							Scale:  p[2],
						},
						WeightLayout: compute.DenseLayoutOutputsInput,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("Len1Scale OutputsInput failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantSym, got, 1e-5); !ok {
				t.Errorf("Len1Scale OutputsInput mismatch:\n%s", diff)
			}
		})

		// 1D [N] per-channel Scale + Int32 ZeroPoint [N] on both layouts.
		chanScale := make([]float32, N)
		chanZPI32 := make([]int32, N)
		for n := range N {
			chanScale[n] = 0.03 + float32(n)*0.01
			chanZPI32[n] = int32((n % 5) - 2)
		}
		wantChan := refQuantLinear2D(x, int8ToF32Matrix(wIO), bias, compute.ActivationRelu, func(_, n int) (float32, float32) {
			return chanScale[n], float32(chanZPI32[n])
		})
		for _, layout := range []compute.DenseLayout{compute.DenseLayoutInputOutputs, compute.DenseLayoutOutputsInput} {
			t.Run("PerChannel1D_"+layout.String(), func(t *testing.T) {
				var wArg any = wIO
				if layout == compute.DenseLayoutOutputsInput {
					wArg = wOI
				}
				got, err := testutil.Exec1(b, []any{x, wArg, chanScale, chanZPI32, bias},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						return f.FusedQuantizedDense(p[0], p[1], p[4], compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{
								Scheme:    compute.QuantLinear,
								Scale:     p[2],
								ZeroPoint: p[3],
							},
							Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
							WeightLayout: layout,
						})
					})
				skipIfQuantizedDenseNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("PerChannel1D %s failed: %+v", layout, err)
				}
				if ok, diff := testutil.IsInDelta(wantChan, got, 1e-5); !ok {
					t.Errorf("PerChannel1D %s mismatch:\n%s", layout, diff)
				}
			})
		}
	})

	t.Run("2DBlockAxes_DivisibleAndNonDivisible_BroadcastZP_Uint8AndFloatZP", func(t *testing.T) {
		// Use K=10, N=9 with BlockSize=4 so ceil(10/4)=3 and ceil(9/4)=3 exercise non-divisible tail blocks!
		const M, K, N, G = 2, 10, 9, 4
		x := makeFloat32Matrix(M, K, 0.2)
		wU8IO := makeUint8Matrix(K, N, 110, 35)
		wU8OI := transposeUint8Matrix(wU8IO)

		// 1. BlockAxis=0 on [K, N] with non-divisible BlockSize=4 -> scales [3, 9], broadcast 1D ZP [9] (Uint8)
		numKB := (K + G - 1) / G // 3
		scalesKBlock := make([][]float32, numKB)
		for kb := range numKB {
			scalesKBlock[kb] = make([]float32, N)
			for n := range N {
				scalesKBlock[kb][n] = 0.04 + float32(kb*N+n)*0.005
			}
		}
		zpChanU8 := make([]uint8, N)
		for n := range N {
			zpChanU8[n] = uint8(120 + n)
		}
		wantNonDivK := refQuantLinear2D(x, uint8ToF32Matrix(wU8IO), nil, compute.ActivationNone, func(k, n int) (float32, float32) {
			return scalesKBlock[k/G][n], float32(zpChanU8[n])
		})

		t.Run("BlockAxis0_NonDivisible_Broadcast1DZP_InputOutputs", func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x, wU8IO, scalesKBlock, zpChanU8},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							ZeroPoint: p[3],
							BlockAxis: 0,
							BlockSize: G,
						},
						WeightLayout: compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("BlockAxis0 NonDivisible InputOutputs failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantNonDivK, got, 1e-5); !ok {
				t.Errorf("BlockAxis0 NonDivisible InputOutputs mismatch:\n%s", diff)
			}
		})

		// Equivalent transposed layout: [N, K] with BlockAxis=1, BlockSize=4 -> scales [9, 3], ZP [9]
		scalesKBlockOI := make([][]float32, N)
		for n := range N {
			scalesKBlockOI[n] = make([]float32, numKB)
			for kb := range numKB {
				scalesKBlockOI[n][kb] = scalesKBlock[kb][n]
			}
		}
		t.Run("BlockAxis1_NonDivisible_Broadcast1DZP_OutputsInput", func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x, wU8OI, scalesKBlockOI, zpChanU8},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							ZeroPoint: p[3],
							BlockAxis: 1,
							BlockSize: G,
						},
						WeightLayout: compute.DenseLayoutOutputsInput,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("BlockAxis1 NonDivisible OutputsInput failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantNonDivK, got, 1e-5); !ok {
				t.Errorf("BlockAxis1 NonDivisible OutputsInput mismatch:\n%s", diff)
			}
		})

		// 2. BlockAxis=1 on [K, N] (blocking along N: scales [K, ceil(N/G)] = [10, 3]) with Float32 ZeroPoint [10, 3]
		numNB := (N + G - 1) / G // 3
		scalesNBlock := make([][]float32, K)
		zpFloatNBlock := make([][]float32, K)
		for k := range K {
			scalesNBlock[k] = make([]float32, numNB)
			zpFloatNBlock[k] = make([]float32, numNB)
			for nb := range numNB {
				scalesNBlock[k][nb] = 0.03 + float32(k*numNB+nb)*0.004
				zpFloatNBlock[k][nb] = 125.5 + float32(nb)*0.5
			}
		}
		wantNBlock := refQuantLinear2D(x, uint8ToF32Matrix(wU8IO), nil, compute.ActivationNone, func(k, n int) (float32, float32) {
			return scalesNBlock[k][n/G], zpFloatNBlock[k][n/G]
		})

		t.Run("BlockAxis1_AlongN_FloatZP_InputOutputs", func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x, wU8IO, scalesNBlock, zpFloatNBlock},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							ZeroPoint: p[3],
							BlockAxis: 1,
							BlockSize: G,
						},
						WeightLayout: compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("BlockAxis1 AlongN FloatZP InputOutputs failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantNBlock, got, 1e-4); !ok {
				t.Errorf("BlockAxis1 AlongN FloatZP InputOutputs mismatch:\n%s", diff)
			}
		})

		// Equivalent transposed layout: [N, K] with BlockAxis=0, BlockSize=4 -> scales [3, 10], ZP [3, 10]
		scalesNBlockOI := make([][]float32, numNB)
		zpFloatNBlockOI := make([][]float32, numNB)
		for nb := range numNB {
			scalesNBlockOI[nb] = make([]float32, K)
			zpFloatNBlockOI[nb] = make([]float32, K)
			for k := range K {
				scalesNBlockOI[nb][k] = scalesNBlock[k][nb]
				zpFloatNBlockOI[nb][k] = zpFloatNBlock[k][nb]
			}
		}
		t.Run("BlockAxis0_AlongN_FloatZP_OutputsInput", func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{x, wU8OI, scalesNBlockOI, zpFloatNBlockOI},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							ZeroPoint: p[3],
							BlockAxis: 0,
							BlockSize: G,
						},
						WeightLayout: compute.DenseLayoutOutputsInput,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("BlockAxis0 AlongN FloatZP OutputsInput failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(wantNBlock, got, 1e-4); !ok {
				t.Errorf("BlockAxis0 AlongN FloatZP OutputsInput mismatch:\n%s", diff)
			}
		})

		// 3. Inferred BlockSize == 0 when divisible (K=8, N=8, scales [2, 8] -> inferred BlockSize=4)
		t.Run("InferredBlockSizeZero", func(t *testing.T) {
			const K2, N2 = 8, 8
			x2 := makeFloat32Matrix(2, K2, 0.3)
			w2 := makeInt8Matrix(K2, N2)
			sc2 := [][]float32{
				{0.1, 0.2, 0.15, 0.25, 0.05, 0.3, 0.12, 0.18},
				{0.2, 0.1, 0.25, 0.05, 0.15, 0.22, 0.08, 0.14},
			}
			want2 := refQuantLinear2D(x2, int8ToF32Matrix(w2), nil, compute.ActivationNone, func(k, n int) (float32, float32) {
				return sc2[k/4][n], 0
			})
			got, err := testutil.Exec1(b, []any{x2, w2, sc2},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							BlockAxis: 0,
							BlockSize: 0, // Inferred as 8 / 2 = 4.
						},
						WeightLayout: compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("InferredBlockSizeZero failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want2, got, 1e-5); !ok {
				t.Errorf("InferredBlockSizeZero mismatch:\n%s", diff)
			}
		})
	})
}

// testQDLinearSubByteDTypesAndLayouts tests Int4, Uint4, Int2, and Uint2 weights in BOTH
// DenseLayoutInputOutputs and DenseLayoutOutputsInput, including sub-byte ZeroPoints (even and odd element counts).
func testQDLinearSubByteDTypesAndLayouts(t *testing.T, b compute.Backend) {
	testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
	testutil.SkipIfMissing(t, b, compute.OpTypeReshape)

	const M, K, N, G = 2, 8, 8, 4
	x := makeFloat32Matrix(M, K, 0.35)
	scalesIO := [][]float32{
		{0.1, 0.2, 0.15, 0.25, 0.05, 0.3, 0.12, 0.18},
		{0.2, 0.1, 0.25, 0.05, 0.15, 0.22, 0.08, 0.14},
	}
	scalesOI := make([][]float32, N)
	for n := range N {
		scalesOI[n] = []float32{scalesIO[0][n], scalesIO[1][n]}
	}

	// 4-bit weights (Int4 and Uint4) in both InputOutputs and OutputsInput, with Uint4 ZeroPoint (including odd scalar [1] via Slice).
	for _, isSigned := range []bool{true, false} {
		dt := dtypes.Uint4
		dtName := "Uint4"
		if isSigned {
			dt = dtypes.Int4
			dtName = "Int4"
		}
		t.Run(dtName, func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dt)
			wValsIO := make([][]float32, K)
			wNibIO := make([][]uint8, K)
			for k := range K {
				wValsIO[k] = make([]float32, N)
				wNibIO[k] = make([]uint8, N)
				for n := range N {
					nib := uint8((k*5 + n*3 + 1) & 0x0F)
					wNibIO[k][n] = nib
					if isSigned && nib >= 8 {
						wValsIO[k][n] = float32(int(nib) - 16)
					} else {
						wValsIO[k][n] = float32(nib)
					}
				}
			}
			wNibOI := transposeUint8Matrix(wNibIO)
			packedIO := pack4BitMatrix(wNibIO)
			packedOI := pack4BitMatrix(wNibOI)

			// Sub-byte ZeroPoint: scalar [1] sliced from 2-element sub-byte vector (tests odd sub-byte ZP count!).
			zpNib := uint8(7)
			zpVal := float32(7)
			if isSigned {
				zpNib = 0x0E // -2 in signed Int4
				zpVal = -2
			}
			zpPackedByte := []uint8{zpNib | (3 << 4)}
			want := refQuantLinear2D(x, wValsIO, nil, compute.ActivationNone, func(k, n int) (float32, float32) {
				return scalesIO[k/G][n], zpVal
			})

			t.Run("InputOutputs_WithSubByteZP", func(t *testing.T) {
				testutil.SkipIfMissing(t, b, compute.OpTypeSlice)
				got, err := testutil.Exec1(b, []any{x, packedIO, scalesIO, zpPackedByte},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						wSub, err := f.Bitcast(p[1], dt)
						if err != nil {
							return nil, err
						}
						wSub, err = f.Reshape(wSub, K, N)
						if err != nil {
							return nil, err
						}
						zpSub, err := f.Bitcast(p[3], dt)
						if err != nil {
							return nil, err
						}
						zpSub, err = f.Reshape(zpSub, 2)
						if err != nil {
							return nil, err
						}
						zpScalar1, err := f.Slice(zpSub, []int{0}, []int{1}, []int{1})
						if err != nil {
							return nil, err
						}
						return f.FusedQuantizedDense(p[0], wSub, nil, compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{
								Scheme:    compute.QuantLinear,
								Scale:     p[2],
								ZeroPoint: zpScalar1,
								BlockAxis: 0,
								BlockSize: G,
							},
							WeightLayout: compute.DenseLayoutInputOutputs,
						})
					})
				skipIfQuantizedDenseNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("%s InputOutputs failed: %+v", dtName, err)
				}
				if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
					t.Errorf("%s InputOutputs mismatch:\n%s", dtName, diff)
				}
			})

			t.Run("OutputsInput_Symmetric", func(t *testing.T) {
				wantSym := refQuantLinear2D(x, wValsIO, nil, compute.ActivationNone, func(k, n int) (float32, float32) {
					return scalesIO[k/G][n], 0
				})
				got, err := testutil.Exec1(b, []any{x, packedOI, scalesOI},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						wSub, err := f.Bitcast(p[1], dt)
						if err != nil {
							return nil, err
						}
						wSub, err = f.Reshape(wSub, N, K)
						if err != nil {
							return nil, err
						}
						return f.FusedQuantizedDense(p[0], wSub, nil, compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{
								Scheme:    compute.QuantLinear,
								Scale:     p[2],
								BlockAxis: 1,
								BlockSize: G,
							},
							WeightLayout: compute.DenseLayoutOutputsInput,
						})
					})
				skipIfQuantizedDenseNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("%s OutputsInput failed: %+v", dtName, err)
				}
				if ok, diff := testutil.IsInDelta(wantSym, got, 1e-5); !ok {
					t.Errorf("%s OutputsInput mismatch:\n%s", dtName, diff)
				}
			})
		})
	}

	// 2-bit weights (Int2 and Uint2) in both InputOutputs and OutputsInput!
	for _, isSigned := range []bool{true, false} {
		dt := dtypes.Uint2
		dtName := "Uint2"
		if isSigned {
			dt = dtypes.Int2
			dtName = "Int2"
		}
		t.Run(dtName, func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dt)
			wValsIO := make([][]float32, K)
			wCrumbIO := make([][]uint8, K)
			for k := range K {
				wValsIO[k] = make([]float32, N)
				wCrumbIO[k] = make([]uint8, N)
				for n := range N {
					c := uint8((k*3 + n*2 + 1) & 0x03)
					wCrumbIO[k][n] = c
					if isSigned && c >= 2 {
						wValsIO[k][n] = float32(int(c) - 4)
					} else {
						wValsIO[k][n] = float32(c)
					}
				}
			}
			wCrumbOI := transposeUint8Matrix(wCrumbIO)
			packedIO := pack2BitMatrix(wCrumbIO)
			packedOI := pack2BitMatrix(wCrumbOI)

			zpVal := float32(0)
			if !isSigned {
				zpVal = 1
			}
			want := refQuantLinear2D(x, wValsIO, nil, compute.ActivationNone, func(k, n int) (float32, float32) {
				return scalesIO[k/G][n], zpVal
			})

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
					inputs := []any{x, packed, sc}
					if !isSigned {
						inputs = append(inputs, uint8(1))
					}
					got, err := testutil.Exec1(b, inputs,
						func(f compute.Function, p []compute.Value) (compute.Value, error) {
							wSub, err := f.Bitcast(p[1], dt)
							if err != nil {
								return nil, err
							}
							wSub, err = f.Reshape(wSub, d0, d1)
							if err != nil {
								return nil, err
							}
							wq := &compute.Quantization{
								Scheme:    compute.QuantLinear,
								Scale:     p[2],
								BlockAxis: blockAxis,
								BlockSize: G,
							}
							if !isSigned {
								wq.ZeroPoint = p[3]
							}
							return f.FusedQuantizedDense(p[0], wSub, nil, compute.FusedQuantizedDenseConfig{
								WeightQuantization: wq,
								WeightLayout:       layout,
							})
						})
					skipIfQuantizedDenseNotImplemented(t, b, err)
					if err != nil {
						t.Fatalf("%s %s failed: %+v", dtName, layout, err)
					}
					if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
						t.Errorf("%s %s mismatch:\n%s", dtName, layout, diff)
					}
				})
			}
		})
	}
}

// testQDNF4LayoutsAxesAndDTypes tests QuantNF4 across Uint4, Int4, Uint8, Int8 weights,
// both WeightLayouts (InputOutputs and OutputsInput), and both BlockAxes (0 and 1).
func testQDNF4LayoutsAxesAndDTypes(t *testing.T, b compute.Backend) {
	const M, K, N, G = 2, 8, 8, 4
	x := makeFloat32Matrix(M, K, 0.3)
	bias := makeFloat32Vec(N, 0.08)

	nibIO := make([][]uint8, K)
	nf4ValsIO := make([][]float32, K)
	int8NibIO := make([][]int8, K)
	for k := range K {
		nibIO[k] = make([]uint8, N)
		nf4ValsIO[k] = make([]float32, N)
		int8NibIO[k] = make([]int8, N)
		for n := range N {
			code := uint8((k*3 + n*5 + 2) & 0x0F)
			nibIO[k][n] = code
			nf4ValsIO[k][n] = compute.NF4LookupTable[code]
			if code >= 8 {
				int8NibIO[k][n] = int8(int(code) - 16)
			} else {
				int8NibIO[k][n] = int8(code)
			}
		}
	}
	nibOI := transposeUint8Matrix(nibIO)
	packedIO := pack4BitMatrix(nibIO)
	packedOI := pack4BitMatrix(nibOI)

	// BlockAxis=1 on [K, N]: scales [K, N/G] = [8, 2] (classic QLoRA BlockAxis=1 layout)
	scalesAxis1 := make([][]float32, K)
	for k := range K {
		scalesAxis1[k] = []float32{0.15 + float32(k)*0.02, 0.25 + float32(k)*0.01}
	}
	wantAxis1 := refQuantLinear2D(x, nf4ValsIO, bias, compute.ActivationNone, func(k, n int) (float32, float32) {
		return scalesAxis1[k][n/G], 0
	})

	t.Run("Uint4_InputOutputs_BlockAxis1", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
		testutil.SkipIfMissingDType(t, b, dtypes.Uint4)
		got, err := testutil.Exec1(b, []any{x, packedIO, scalesAxis1, bias},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				w, err := f.Bitcast(p[1], dtypes.Uint4)
				if err != nil {
					return nil, err
				}
				w, err = f.Reshape(w, K, N)
				if err != nil {
					return nil, err
				}
				return f.FusedQuantizedDense(p[0], w, p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{
						Scheme:    compute.QuantNF4,
						Scale:     p[2],
						BlockAxis: 1,
						BlockSize: G,
					},
					WeightLayout: compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("NF4 Uint4 InputOutputs BlockAxis1 failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(wantAxis1, got, 1e-5); !ok {
			t.Errorf("NF4 Uint4 InputOutputs BlockAxis1 mismatch:\n%s", diff)
		}
	})

	// Transposed equivalent: Int4 [N, K] OutputsInput with BlockAxis=0, scales [N/G, K] = [2, 8]
	scalesAxis0OI := make([][]float32, N/G)
	for nb := range N / G {
		scalesAxis0OI[nb] = make([]float32, K)
		for k := range K {
			scalesAxis0OI[nb][k] = scalesAxis1[k][nb]
		}
	}
	t.Run("Int4_OutputsInput_BlockAxis0", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
		testutil.SkipIfMissingDType(t, b, dtypes.Int4)
		got, err := testutil.Exec1(b, []any{x, packedOI, scalesAxis0OI, bias},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				w, err := f.Bitcast(p[1], dtypes.Int4)
				if err != nil {
					return nil, err
				}
				w, err = f.Reshape(w, N, K)
				if err != nil {
					return nil, err
				}
				return f.FusedQuantizedDense(p[0], w, p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{
						Scheme:    compute.QuantNF4,
						Scale:     p[2],
						BlockAxis: 0,
						BlockSize: G,
					},
					WeightLayout: compute.DenseLayoutOutputsInput,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("NF4 Int4 OutputsInput BlockAxis0 failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(wantAxis1, got, 1e-5); !ok {
			t.Errorf("NF4 Int4 OutputsInput BlockAxis0 mismatch:\n%s", diff)
		}
	})

	// Unpacked Uint8 and Int8 NF4 weights with per-channel [N] scales.
	chanScales := makeFloat32PositiveVec(N, 0.1, 0.03)
	wantChan := refQuantLinear2D(x, nf4ValsIO, bias, compute.ActivationNone, func(_, n int) (float32, float32) {
		return chanScales[n], 0
	})
	t.Run("Uint8_And_Int8_UnpackedCodes", func(t *testing.T) {
		for _, useInt8 := range []bool{false, true} {
			var wArg any = nibIO
			name := "Uint8"
			if useInt8 {
				wArg = int8NibIO
				name = "Int8"
			}
			got, err := testutil.Exec1(b, []any{x, wArg, chanScales, bias},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme: compute.QuantNF4,
							Scale:  p[2],
						},
						WeightLayout: compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("NF4 %s failed: %+v", name, err)
			}
			if ok, diff := testutil.IsInDelta(wantChan, got, 1e-5); !ok {
				t.Errorf("NF4 %s mismatch:\n%s", name, diff)
			}
		}
	})
}

// testQDLinearQuantizedMatmuls verifies QuantizedMatmuls=true (on-the-fly dynamic Int8 activation quantization)
// on QuantLinear weights across layouts, dtypes (Int8, Uint8, Int4), and integer/float zero-points.
func testQDLinearQuantizedMatmuls(t *testing.T, b compute.Backend) {
	const M, K, N, G = 3, 16, 10, 8
	x := makeFloat32Matrix(M, K, 0.25)
	// Include a zero row to test maxAbs == 0 handling in dynamic row quantization!
	xZero := make([][]float32, M)
	for m := range M {
		xZero[m] = make([]float32, K)
		if m != 1 {
			copy(xZero[m], x[m])
		}
	}
	wI8IO := makeInt8Matrix(K, N)
	wI8OI := transposeInt8Matrix(wI8IO)
	bias := makeFloat32Vec(N, 0.1)

	scalesIO := make([][]float32, K/G)
	zpI8IO := make([][]int8, K/G)
	zpF32IO := make([][]float32, K/G)
	for kb := range K / G {
		scalesIO[kb] = makeFloat32PositiveVec(N, 0.05, 0.01)
		zpI8IO[kb] = make([]int8, N)
		zpF32IO[kb] = make([]float32, N)
		for n := range N {
			zpI8IO[kb][n] = int8((kb + n) % 3)
			zpF32IO[kb][n] = float32(zpI8IO[kb][n]) + 0.25
		}
	}
	scalesOI := transposeFloat32Matrix(scalesIO)
	zpI8OI := transposeInt8Matrix(zpI8IO)

	wantIntZP := refQuantLinear2D(xZero, int8ToF32Matrix(wI8IO), bias, compute.ActivationNone, func(k, n int) (float32, float32) {
		return scalesIO[k/G][n], float32(zpI8IO[k/G][n])
	})

	for _, layout := range []compute.DenseLayout{compute.DenseLayoutInputOutputs, compute.DenseLayoutOutputsInput} {
		t.Run("Int8_"+layout.String(), func(t *testing.T) {
			var wArg, scArg, zpArg any = wI8IO, scalesIO, zpI8IO
			blockAxis := 0
			if layout == compute.DenseLayoutOutputsInput {
				wArg, scArg, zpArg = wI8OI, scalesOI, zpI8OI
				blockAxis = 1
			}
			got, err := testutil.Exec1(b, []any{xZero, wArg, scArg, zpArg, bias},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], p[4], compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							ZeroPoint: p[3],
							BlockAxis: blockAxis,
							BlockSize: G,
						},
						QuantizedMatmuls: true,
						WeightLayout:     layout,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("QuantizedMatmuls %s failed: %+v", layout, err)
			}
			// Dynamic Int8 quantization introduces minor rounding (~1/127), so tolerance 0.05 checks close agreement.
			if ok, diff := testutil.IsInDelta(wantIntZP, got, 0.05); !ok {
				t.Errorf("QuantizedMatmuls %s drifted too far from float:\n%s", layout, diff)
			}
		})
	}

	// Float ZeroPoint with QuantizedMatmuls=true must remain accurate.
	t.Run("FloatZeroPoint_FallbackAccuracy", func(t *testing.T) {
		wantFloatZP := refQuantLinear2D(x, int8ToF32Matrix(wI8IO), bias, compute.ActivationNone, func(k, n int) (float32, float32) {
			return scalesIO[k/G][n], zpF32IO[k/G][n]
		})
		got, err := testutil.Exec1(b, []any{x, wI8IO, scalesIO, zpF32IO, bias},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], p[4], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[2],
						ZeroPoint: p[3],
						BlockAxis: 0,
						BlockSize: G,
					},
					QuantizedMatmuls: true,
					WeightLayout:     compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("QuantizedMatmuls FloatZP failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(wantFloatZP, got, 0.05); !ok {
			t.Errorf("QuantizedMatmuls FloatZP mismatch:\n%s", diff)
		}
	})
}

// testQDPreQuantizedXLinear tests XQuantization (QuantLinear) across:
//   - All 4 signed/unsigned 8-bit combinations (Uint8 x Int8, Uint8 x Uint8, Int8 x Int8, Int8 x Uint8) + Int4 weights
//   - Both WeightLayouts (InputOutputs and OutputsInput)
//   - Per-tensor, per-row ([M, 1] and [M]), and per-block ([M, K/G] x [K/G, N]) scales/zero-points
//   - Nil bias, Int32 bias, and Float32 bias.
func testQDPreQuantizedXLinear(t *testing.T, b compute.Backend) {
	const M, K, N = 3, 12, 8
	xU8 := makeUint8Matrix(M, K, 110, 40)
	xI8 := makeInt8Matrix(M, K)
	wU8IO := makeUint8Matrix(K, N, 120, 30)
	wI8IO := makeInt8Matrix(K, N)
	wI8OI := transposeInt8Matrix(wI8IO)

	xScaleRow := [][]float32{{0.02}, {0.03}, {0.04}}
	xZPU8Row := [][]uint8{{120}, {125}, {130}}
	xZPI8Row := [][]int8{{-3}, {0}, {4}}

	wScaleChan := makeFloat32PositiveVec(N, 0.02, 0.005)
	wZPU8Chan := make([]uint8, N)
	wZPI8Chan := make([]int8, N)
	biasI32 := make([]int32, N)
	biasF32 := makeFloat32Vec(N, 0.15)
	for n := range N {
		wZPU8Chan[n] = uint8(124 + n)
		wZPI8Chan[n] = int8((n % 5) - 2)
		biasI32[n] = int32((n - 3) * 12)
	}

	combos := []struct {
		name            string
		xUnsigned       bool
		wUnsigned       bool
		layout          compute.DenseLayout
		useFloat32Bias  bool
		symmetricZeroPt bool
	}{
		{"Uint8_x_Int8_InputOutputs_Int32Bias", true, false, compute.DenseLayoutInputOutputs, false, false},
		{"Uint8_x_Uint8_InputOutputs_Float32Bias", true, true, compute.DenseLayoutInputOutputs, true, false},
		{"Int8_x_Int8_OutputsInput_Int32Bias", false, false, compute.DenseLayoutOutputsInput, false, false},
		{"Int8_x_Uint8_InputOutputs_Symmetric", false, true, compute.DenseLayoutInputOutputs, true, true},
	}

	for _, tc := range combos {
		t.Run(tc.name, func(t *testing.T) {
			want := make([][]float32, M)
			for m := range M {
				want[m] = make([]float32, N)
				for n := range N {
					var acc int32
					if !tc.useFloat32Bias {
						acc = biasI32[n]
					}
					for k := range K {
						var xv, xzp, wv, wzp int32
						if tc.xUnsigned {
							xv = int32(xU8[m][k])
							if !tc.symmetricZeroPt {
								xzp = int32(xZPU8Row[m][0])
							}
						} else {
							xv = int32(xI8[m][k])
							if !tc.symmetricZeroPt {
								xzp = int32(xZPI8Row[m][0])
							}
						}
						if tc.wUnsigned {
							wv = int32(wU8IO[k][n])
							if !tc.symmetricZeroPt {
								wzp = int32(wZPU8Chan[n])
							}
						} else {
							wv = int32(wI8IO[k][n])
							if !tc.symmetricZeroPt {
								wzp = int32(wZPI8Chan[n])
							}
						}
						acc += (xv - xzp) * (wv - wzp)
					}
					val := float32(acc) * (xScaleRow[m][0] * wScaleChan[n])
					if tc.useFloat32Bias {
						val += biasF32[n]
					}
					want[m][n] = val
				}
			}

			var xArg any = xI8
			var xZPArg any = xZPI8Row
			if tc.xUnsigned {
				xArg = xU8
				xZPArg = xZPU8Row
			}
			var wArg any = wI8IO
			var wZPArg any = wZPI8Chan
			if tc.wUnsigned {
				wArg = wU8IO
				wZPArg = wZPU8Chan
			} else if tc.layout == compute.DenseLayoutOutputsInput {
				wArg = wI8OI
			}
			var biasArg any = biasI32
			if tc.useFloat32Bias {
				biasArg = biasF32
			}

			params := []any{xArg, wArg, wScaleChan, xScaleRow, biasArg}
			if !tc.symmetricZeroPt {
				params = append(params, wZPArg, xZPArg)
			}

			got, err := testutil.Exec1(b, params, func(f compute.Function, p []compute.Value) (compute.Value, error) {
				wq := &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]}
				xq := &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[3]}
				if !tc.symmetricZeroPt {
					wq.ZeroPoint = p[5]
					xq.ZeroPoint = p[6]
				}
				return f.FusedQuantizedDense(p[0], p[1], p[4], compute.FusedQuantizedDenseConfig{
					WeightQuantization: wq,
					XQuantization:      xq,
					WeightLayout:       tc.layout,
				})
			})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("%s failed: %+v", tc.name, err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
				t.Errorf("%s mismatch:\n%s", tc.name, diff)
			}
		})
	}

	// Per-block XQuantization [M, K/G] paired with per-block WeightQuantization [K/G, N]!
	t.Run("PerKBlock_XScale_And_WScale", func(t *testing.T) {
		const G = 4
		numKB := K / G // 3
		xBlockScale := make([][]float32, M)
		for m := range M {
			xBlockScale[m] = makeFloat32PositiveVec(numKB, 0.02, 0.01)
		}
		wBlockScale := make([][]float32, numKB)
		for kb := range numKB {
			wBlockScale[kb] = makeFloat32PositiveVec(N, 0.03, 0.005)
		}

		want := make([][]float32, M)
		for m := range M {
			want[m] = make([]float32, N)
			for n := range N {
				var sum float32
				for kb := range numKB {
					var acc int32
					for k := kb * G; k < (kb+1)*G; k++ {
						acc += int32(xI8[m][k]) * int32(wI8IO[k][n])
					}
					sum += float32(acc) * (xBlockScale[m][kb] * wBlockScale[kb][n])
				}
				want[m][n] = sum
			}
		}

		got, err := testutil.Exec1(b, []any{xI8, wI8IO, wBlockScale, xBlockScale},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[2],
						BlockAxis: 0,
						BlockSize: G,
					},
					XQuantization: &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[3],
						BlockSize: G,
					},
					WeightLayout: compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("PerKBlock XQuantization failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
			t.Errorf("PerKBlock XQuantization mismatch:\n%s", diff)
		}
	})
}

// testQDOutputQuantization tests OutputQuantization across:
//   - Pre-quantized XQuantization (QLinearMatMul -> Uint8 and Int8)
//   - Weight-only float x (QuantLinear and QuantNF4 -> quantized output)
//   - DType resolution: inferred Uint8 from Uint8 ZP, inferred Int8 from Int8 ZP, default Int8 when ZP==nil,
//     and explicit QuantizedDType=Uint8 with Float32 ZP
//   - OutputQuantization granularities: scalar [], per-channel [N], and per-row [M, 1], with ActivationRelu.
func testQDOutputQuantization(t *testing.T, b compute.Backend) {
	const M, K, N = 3, 8, 6
	xF32 := makeFloat32Matrix(M, K, 0.4)
	wI8 := makeInt8Matrix(K, N)
	wScale := makeFloat32PositiveVec(N, 0.1, 0.02)
	bias := makeFloat32Vec(N, 0.2)

	linearRelu := refQuantLinear2D(xF32, int8ToF32Matrix(wI8), bias, compute.ActivationRelu, func(_, n int) (float32, float32) {
		return wScale[n], 0
	})

	// 1. Default symmetric OutputQuantization (ZeroPoint == nil, QuantizedDType == InvalidDType -> Int8) with per-channel [N] yScale
	t.Run("SymmetricDefaultInt8_PerChannel", func(t *testing.T) {
		yScale := makeFloat32PositiveVec(N, 0.05, 0.01)
		want := make([][]int8, M)
		for m := range M {
			want[m] = make([]int8, N)
			for n := range N {
				q := math.RoundToEven(float64(linearRelu[m][n] / yScale[n]))
				q = max(-128, min(127, q))
				want[m][n] = int8(int32(q))
			}
		}
		got, err := testutil.Exec1(b, []any{xF32, wI8, wScale, bias, yScale},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					OutputQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[4]},
					Activation:         compute.ActivationConfig{Type: compute.ActivationRelu},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("SymmetricDefaultInt8 OutputQuantization failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("SymmetricDefaultInt8 OutputQuantization mismatch:\n%s", diff)
		}
	})

	// 2. Explicit QuantizedDType=Uint8 with Float32 ZeroPoint and per-row [M, 1] yScale
	t.Run("ExplicitUint8_FloatZP_PerRowScale", func(t *testing.T) {
		yScaleRow := [][]float32{{0.04}, {0.06}, {0.08}}
		yZPF32 := float32(128.0)
		want := make([][]uint8, M)
		for m := range M {
			want[m] = make([]uint8, N)
			for n := range N {
				q := float32(math.RoundToEven(float64(linearRelu[m][n]/yScaleRow[m][0]))) + yZPF32
				q = max(0, min(255, q))
				want[m][n] = uint8(int32(math.RoundToEven(float64(q))))
			}
		}
		got, err := testutil.Exec1(b, []any{xF32, wI8, wScale, bias, yScaleRow, yZPF32},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					OutputQuantization: &compute.Quantization{
						Scheme:         compute.QuantLinear,
						Scale:          p[4],
						ZeroPoint:      p[5],
						QuantizedDType: dtypes.Uint8,
					},
					Activation:   compute.ActivationConfig{Type: compute.ActivationRelu},
					WeightLayout: compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("ExplicitUint8 OutputQuantization failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("ExplicitUint8 OutputQuantization mismatch:\n%s", diff)
		}
	})

	// 3. QLinearMatMul: Pre-quantized Uint8 x * Int8 weight -> inferred Int8 output (from Int8 yZeroPoint)
	t.Run("QLinearMatMul_InferredInt8Output", func(t *testing.T) {
		xU8 := makeUint8Matrix(M, K, 120, 30)
		xScale := float32(0.05)
		xZP := uint8(128)
		yScale := float32(0.1)
		yZP := int8(-10)

		want := make([][]int8, M)
		for m := range M {
			want[m] = make([]int8, N)
			for n := range N {
				var acc int32
				for k := range K {
					acc += (int32(xU8[m][k]) - int32(xZP)) * int32(wI8[k][n])
				}
				yf := float32(acc) * (xScale * wScale[n])
				q := float32(math.RoundToEven(float64(yf/yScale))) + float32(yZP)
				q = max(-128, min(127, q))
				want[m][n] = int8(int32(math.RoundToEven(float64(q))))
			}
		}

		got, err := testutil.Exec1(b, []any{xU8, wI8, wScale, xScale, xZP, yScale, yZP},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					XQuantization:      &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[3], ZeroPoint: p[4]},
					OutputQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[5], ZeroPoint: p[6]},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("QLinearMatMul Int8 output failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual(want, got); !ok {
			t.Errorf("QLinearMatMul Int8 output mismatch:\n%s", diff)
		}
	})
}

// testQDRanksTilingAndDynamicShapes tests:
//   - Rank 1 ([K] -> [N])
//   - Rank 2 M=1 large ([1, 128] x [128, 96] -> [1, 96], exercising single-token column tiling)
//   - Rank 3 ([B, S, K] -> [B, S, N])
//   - Rank 4 ([B, H, S, K] -> [B, H, S, N])
//   - Dynamic batch dimensions ([DynamicDim, DynamicDim, K]) executed across multiple concrete shapes.
func testQDRanksTilingAndDynamicShapes(t *testing.T, b compute.Backend) {
	// 1. 1D [K] -> [N]
	t.Run("Rank1_1D", func(t *testing.T) {
		const K, N = 8, 5
		x1D := makeFloat32Vec(K, 0.3)
		w := makeInt8Matrix(K, N)
		scale := makeFloat32PositiveVec(N, 0.1, 0.02)
		bias := makeFloat32Vec(N, 0.05)

		want2D := refQuantLinear2D([][]float32{x1D}, int8ToF32Matrix(w), bias, compute.ActivationNone, func(_, n int) (float32, float32) {
			return scale[n], 0
		})
		got, err := testutil.Exec1(b, []any{x1D, w, scale, bias},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("1D FusedQuantizedDense failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want2D[0], got, 1e-5); !ok {
			t.Errorf("1D FusedQuantizedDense mismatch:\n%s", diff)
		}
	})

	// 2. Large M=1 (column-tiled decode) and M=8 (row-parallel prefill)
	t.Run("LargeSingleTokenAndMultiRowParallel", func(t *testing.T) {
		const K, N, G = 128, 96, 32
		w := makeInt8Matrix(K, N)
		scales := make([][]float32, K/G)
		for kb := range K / G {
			scales[kb] = makeFloat32PositiveVec(N, 0.02, 0.002)
		}
		bias := makeFloat32Vec(N, 0.05)
		wF32 := int8ToF32Matrix(w)

		for _, M := range []int{1, 8} {
			x := makeFloat32Matrix(M, K, 0.1)
			want := refQuantLinear2D(x, wF32, bias, compute.ActivationNone, func(k, n int) (float32, float32) {
				return scales[k/G][n], 0
			})
			got, err := testutil.Exec1(b, []any{x, w, scales, bias},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{
							Scheme:    compute.QuantLinear,
							Scale:     p[2],
							BlockAxis: 0,
							BlockSize: G,
						},
						WeightLayout: compute.DenseLayoutInputOutputs,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("Large M=%d failed: %+v", M, err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-4); !ok {
				t.Errorf("Large M=%d mismatch:\n%s", M, diff)
			}
		}
	})

	// 3. 3D [B, S, K] and 4D [B, H, S, K]
	t.Run("Rank3_And_Rank4", func(t *testing.T) {
		const B, H, S, K, N = 2, 2, 3, 8, 4
		w := makeInt8Matrix(K, N)
		scale := makeFloat32PositiveVec(N, 0.1, 0.03)
		wF32 := int8ToF32Matrix(w)

		x3D := make([][][]float32, B)
		want3D := make([][][]float32, B)
		for bi := range B {
			x3D[bi] = makeFloat32Matrix(S, K, float32(bi+1)*0.2)
			want3D[bi] = refQuantLinear2D(x3D[bi], wF32, nil, compute.ActivationNone, func(_, n int) (float32, float32) {
				return scale[n], 0
			})
		}
		got3D, err := testutil.Exec1(b, []any{x3D, w, scale},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("3D FusedQuantizedDense failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want3D, got3D, 1e-5); !ok {
			t.Errorf("3D FusedQuantizedDense mismatch:\n%s", diff)
		}

		x4D := make([][][][]float32, B)
		want4D := make([][][][]float32, B)
		for bi := range B {
			x4D[bi] = make([][][]float32, H)
			want4D[bi] = make([][][]float32, H)
			for hi := range H {
				x4D[bi][hi] = makeFloat32Matrix(S, K, float32(bi*H+hi+1)*0.15)
				want4D[bi][hi] = refQuantLinear2D(x4D[bi][hi], wF32, nil, compute.ActivationNone, func(_, n int) (float32, float32) {
					return scale[n], 0
				})
			}
		}
		got4D, err := testutil.Exec1(b, []any{x4D, w, scale},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
			})
		if err != nil {
			t.Fatalf("4D FusedQuantizedDense failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want4D, got4D, 1e-5); !ok {
			t.Errorf("4D FusedQuantizedDense mismatch:\n%s", diff)
		}
	})

	// 4. Dynamic 3D batch shapes [DynamicDim, DynamicDim, K] (both float x and pre-quantized x with dynamic xScale!)
	t.Run("DynamicBatch3D_FloatAndPreQuantized", func(t *testing.T) {
		testutil.SkipIfMissingDynamicShapes(t, b)
		const K, N = 8, 6
		w := makeInt8Matrix(K, N)
		wScale := makeFloat32PositiveVec(N, 0.1, 0.02)
		wF32 := int8ToF32Matrix(w)

		builder := b.Builder("qd_dyn_3d")
		fn := builder.Main()
		pX, err := fn.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, shapes.DynamicDim, K}, []string{"batch", "seq", ""}), nil)
		if err != nil {
			t.Fatalf("Parameter x: %+v", err)
		}
		pW, _ := fn.Parameter("w", shapes.Make(dtypes.Int8, K, N), nil)
		pS, _ := fn.Parameter("s", shapes.Make(dtypes.Float32, N), nil)
		y, err := fn.FusedQuantizedDense(pX, pW, nil, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: pS},
			WeightLayout:       compute.DenseLayoutInputOutputs,
		})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Dynamic FusedQuantizedDense build failed: %+v", err)
		}
		if err := fn.Return([]compute.Value{y}, nil); err != nil {
			t.Fatalf("Return: %+v", err)
		}
		exec, err := builder.Compile()
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Compile: %+v", err)
		}
		defer exec.Finalize()

		for _, dims := range [][2]int{{1, 1}, {2, 4}} {
			B, S := dims[0], dims[1]
			xVal := make([][][]float32, B)
			wantVal := make([][][]float32, B)
			for bi := range B {
				xVal[bi] = makeFloat32Matrix(S, K, float32(bi+1)*0.25)
				wantVal[bi] = refQuantLinear2D(xVal[bi], wF32, nil, compute.ActivationNone, func(_, n int) (float32, float32) {
					return wScale[n], 0
				})
			}
			bufX, _ := testutil.ToBuffer(b, xVal)
			bufW, _ := testutil.ToBuffer(b, w)
			bufS, _ := testutil.ToBuffer(b, wScale)
			outs, err := exec.Execute([]compute.Buffer{bufX, bufW, bufS}, nil, 0)
			if err != nil {
				t.Fatalf("Dynamic Execute(%dx%d) failed: %+v", B, S, err)
			}
			gotVal, _ := testutil.FromBuffer(b, outs[0])
			if ok, diff := testutil.IsInDelta(wantVal, gotVal, 1e-5); !ok {
				t.Errorf("Dynamic (%dx%d) mismatch:\n%s", B, S, diff)
			}
		}
	})
}

// testQDHalfPrecisionAndMixedScaleDTypes tests Float16 and BFloat16 x/bias as well as
// Float16/BFloat16 Scale/ZeroPoint paired with Float32 x.
func testQDHalfPrecisionAndMixedScaleDTypes(t *testing.T, b compute.Backend) {
	const M, K, N = 2, 8, 6
	w := makeInt8Matrix(K, N)
	wF32 := int8ToF32Matrix(w)

	t.Run("Float16_X_Scale_Bias", func(t *testing.T) {
		testutil.SkipIfMissingDType(t, b, dtypes.Float16)
		xF16 := make([][]float16.Float16, M)
		xRef := make([][]float32, M)
		for m := range M {
			xF16[m] = make([]float16.Float16, K)
			xRef[m] = make([]float32, K)
			for k := range K {
				v := float16.FromFloat32(float32(m*3+k-4) * 0.25)
				xF16[m][k] = v
				xRef[m][k] = v.Float32()
			}
		}
		scF16 := make([]float16.Float16, N)
		biasF16 := make([]float16.Float16, N)
		biasRef := make([]float32, N)
		for n := range N {
			scF16[n] = float16.FromFloat32(0.125 * float32(n+1))
			biasF16[n] = float16.FromFloat32(float32(n-2) * 0.25)
			biasRef[n] = biasF16[n].Float32()
		}
		wantF32 := refQuantLinear2D(xRef, wF32, biasRef, compute.ActivationRelu, func(_, n int) (float32, float32) {
			return scF16[n].Float32(), 0
		})
		wantF16 := make([][]float16.Float16, M)
		for m := range M {
			wantF16[m] = make([]float16.Float16, N)
			for n := range N {
				wantF16[m][n] = float16.FromFloat32(wantF32[m][n])
			}
		}

		got, err := testutil.Exec1(b, []any{xF16, w, scF16, biasF16},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					Activation:         compute.ActivationConfig{Type: compute.ActivationRelu},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Float16 FusedQuantizedDense failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(wantF16, got, 1e-2); !ok {
			t.Errorf("Float16 FusedQuantizedDense mismatch:\n%s", diff)
		}
	})

	t.Run("BFloat16_X_Float32Scale_And_ActivationFusion", func(t *testing.T) {
		testutil.SkipIfMissingDType(t, b, dtypes.BFloat16)
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedActivation)
		xBF16 := make([][]bfloat16.BFloat16, M)
		xRef := make([][]float32, M)
		for m := range M {
			xBF16[m] = make([]bfloat16.BFloat16, K)
			xRef[m] = make([]float32, K)
			for k := range K {
				v := bfloat16.FromFloat32(float32(m+k-3) * 0.5)
				xBF16[m][k] = v
				xRef[m][k] = v.Float32()
			}
		}
		scF32 := makeFloat32PositiveVec(N, 0.25, 0.0)
		biasBF16 := make([]bfloat16.BFloat16, N)
		biasRef := make([]float32, N)
		for n := range N {
			biasBF16[n] = bfloat16.FromFloat32(float32(n) * 0.125)
			biasRef[n] = biasBF16[n].Float32()
		}
		wantF32 := refQuantLinear2D(xRef, wF32, biasRef, compute.ActivationRelu, func(_, n int) (float32, float32) {
			return scF32[n], 0
		})
		wantBF16 := make([][]bfloat16.BFloat16, M)
		for m := range M {
			wantBF16[m] = make([]bfloat16.BFloat16, N)
			for n := range N {
				wantBF16[m][n] = bfloat16.FromFloat32(wantF32[m][n])
			}
		}

		got, err := testutil.Exec1(b, []any{xBF16, w, scF32, biasBF16},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				y, err := f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
				if err != nil {
					return nil, err
				}
				return f.FusedActivation(y, compute.ActivationConfig{Type: compute.ActivationRelu})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("BFloat16 + Float32Scale failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(wantBF16, got, 2e-2); !ok {
			t.Errorf("BFloat16 + Float32Scale mismatch:\n%s", diff)
		}
	})

	t.Run("Float32X_With_Float16ScaleAndZP", func(t *testing.T) {
		testutil.SkipIfMissingDType(t, b, dtypes.Float16)
		xF32 := makeFloat32Matrix(M, K, 0.25)
		scF16 := make([]float16.Float16, N)
		zpF16 := make([]float16.Float16, N)
		for n := range N {
			scF16[n] = float16.FromFloat32(0.125)
			zpF16[n] = float16.FromFloat32(float32(n - 2))
		}
		want := refQuantLinear2D(xF32, wF32, nil, compute.ActivationNone, func(_, n int) (float32, float32) {
			return scF16[n].Float32(), zpF16[n].Float32()
		})
		got, err := testutil.Exec1(b, []any{xF32, w, scF16, zpF16},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[2],
						ZeroPoint: p[3],
					},
					WeightLayout: compute.DenseLayoutInputOutputs,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("Float32X with Float16 Scale/ZP failed: %+v", err)
		}
		if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
			t.Errorf("Float32X with Float16 Scale/ZP mismatch:\n%s", diff)
		}
	})
}

// testQDActivationsFusionConstantsAndErrors tests:
//   - All 11 supported ActivationTypes
//   - Single-consumer vs multi-consumer FusedActivation fusion
//   - Constant vs Parameter weights/scales/biases
//   - Build-time rejection of invalid configurations (SwiGLU, nil WeightQuantization, etc.).
func testQDActivationsFusionConstantsAndErrors(t *testing.T, b compute.Backend) {
	t.Run("AllActivations", func(t *testing.T) {
		input1D := []float32{0, -1, 2, -3, 4, -5, 6}
		n := len(input1D)
		x2D := [][]float32{input1D}
		// Quantized identity matrix: wIdentity[i][i] = 1 (Int8), scale = 1.0 (Float32).
		wIdentity := make([][]int8, n)
		for i := range n {
			wIdentity[i] = make([]int8, n)
			wIdentity[i][i] = 1
		}
		scaleOne := float32(1.0)
		biasZero := make([]float32, n)

		tests := []struct {
			name      string
			actType   compute.ActivationType
			want      []float32
			tolerance float64
		}{
			{"None", compute.ActivationNone, []float32{0, -1, 2, -3, 4, -5, 6}, xslices.Epsilon},
			{"Relu", compute.ActivationRelu, []float32{0, 0, 2, 0, 4, 0, 6}, xslices.Epsilon},
			{"Sigmoid", compute.ActivationSigmoid, []float32{0.5, 0.26894143, 0.8807971, 0.047425873, 0.98201376, 0.006692851, 0.9975274}, 1e-5},
			{"HardSigmoid", compute.ActivationHardSigmoid, []float32{0.5, 0.3, 0.9, 0.0, 1.0, 0.0, 1.0}, xslices.Epsilon},
			{"LeakyRelu", compute.ActivationLeakyRelu, []float32{0, -0.3, 2, -0.9, 4, -1.5, 6}, xslices.Epsilon},
			{"Selu", compute.ActivationSelu, []float32{0.0, -1.1113307, 2.101402, -1.6705687, 4.202804, -1.7462534, 6.304206}, 1e-5},
			{"Silu", compute.ActivationSilu, []float32{0, -0.26894143, 1.7615942, -0.14227763, 3.928055, -0.03346425, 5.9851646}, 1e-5},
			{"HardSwish", compute.ActivationHardSwish, []float32{0, -0.33333334, 1.6666666, 0, 4, 0, 6}, xslices.Epsilon},
			{"Tanh", compute.ActivationTanh, []float32{0, -0.76159416, 0.9640276, -0.99505475, 0.9993293, -0.9999092, 0.9999877}, 1e-5},
			{"GeluExact", compute.ActivationGelu, []float32{0, -0.15865526, 1.9544997, -4.0496886e-03, 3.9998736, -1.3411045e-06, 6}, 1e-5},
			{"GeluApproximate", compute.ActivationGeluApproximate, []float32{0, -0.15880796, 1.9545977, -3.6375225e-03, 3.9999294, 0, 6}, 0.01},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{x2D, wIdentity, scaleOne, biasZero},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						return f.FusedQuantizedDense(p[0], p[1], p[3], compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2]},
							Activation:         compute.ActivationConfig{Type: tc.actType},
							WeightLayout:       compute.DenseLayoutInputOutputs,
						})
					})
				skipIfQuantizedDenseNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("FusedQuantizedDense(%s) failed: %+v", tc.name, err)
				}
				if ok, diff := testutil.IsInDelta([][]float32{tc.want}, got, tc.tolerance); !ok {
					t.Errorf("FusedQuantizedDense(%s) mismatch:\n%s", tc.name, diff)
				}
			})
		}
	})

	t.Run("MultiConsumer_NoBrokenFusion_And_ConstantWeights", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeFusedActivation)
		x := [][]float32{{1.0, -2.0, 0.5, 1.5}}
		wFlat := []int8{2, -3, 1, 4, -2, 2, 3, -1} // [4, 2]
		sFlat := []float32{0.5, 0.25}
		bFlat := []float32{-1.0, 0.5}

		// Return BOTH pre-activation z and post-activation relu(z) so z has 2 consumers (must NOT overwrite z!).
		gotZ, gotRelu, err := testutil.Exec2(b, []any{x},
			func(f compute.Function, p []compute.Value) (compute.Value, compute.Value, error) {
				wC, _ := f.Constant(wFlat, 4, 2)
				sC, _ := f.Constant(sFlat, 2)
				bC, _ := f.Constant(bFlat, 2)
				z, err := f.FusedQuantizedDense(p[0], wC, bC, compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: sC},
					WeightLayout:       compute.DenseLayoutInputOutputs,
				})
				if err != nil {
					return nil, nil, err
				}
				y, err := f.FusedActivation(z, compute.ActivationConfig{Type: compute.ActivationRelu})
				return z, y, err
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("MultiConsumer ConstantWeights failed: %+v", err)
		}
		// Manual dot:
		// col 0: (1*2 + (-2)*1 + 0.5*(-2) + 1.5*3)*0.5 - 1.0 = (2 - 2 - 1 + 4.5)*0.5 - 1 = 0.75
		// col 1: (1*(-3) + (-2)*4 + 0.5*2 + 1.5*(-1))*0.25 + 0.5 = (-3 - 8 + 1 - 1.5)*0.25 + 0.5 = -2.875 + 0.5 = -2.375
		wantZ := [][]float32{{0.75, -2.375}}
		wantRelu := [][]float32{{0.75, 0.0}}
		if ok, diff := testutil.IsInDelta(wantZ, gotZ, 1e-5); !ok {
			t.Errorf("MultiConsumer pre-activation z mismatch:\n%s", diff)
		}
		if ok, diff := testutil.IsInDelta(wantRelu, gotRelu, 1e-5); !ok {
			t.Errorf("MultiConsumer post-activation relu(z) mismatch:\n%s", diff)
		}
	})

	t.Run("ValidationErrors_Rejected", func(t *testing.T) {
		builder := b.Builder("qd_reject_tests")
		fn := builder.Main()
		xVal, _ := fn.Parameter("x", shapes.Make(dtypes.Float32, 2, 4), nil)
		wVal, _ := fn.Parameter("w", shapes.Make(dtypes.Int8, 4, 4), nil)
		sVal, _ := fn.Parameter("s", shapes.Make(dtypes.Float32, 4), nil)

		// 1. Nil WeightQuantization
		if _, err := fn.FusedQuantizedDense(xVal, wVal, nil, compute.FusedQuantizedDenseConfig{}); err == nil {
			t.Errorf("Expected error when WeightQuantization is nil")
		}
		// 2. SwiGLU activation
		if _, err := fn.FusedQuantizedDense(xVal, wVal, nil, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: sVal},
			Activation:         compute.ActivationConfig{Type: compute.ActivationSwiGLU},
		}); err == nil {
			t.Errorf("Expected error when Activation is SwiGLU")
		}
		// 3. QuantGGML with InputOutputs layout
		wGGML, _ := fn.Parameter("w_ggml", shapes.Make(dtypes.Uint8, 4, 18), nil)
		xGGML, _ := fn.Parameter("x_ggml", shapes.Make(dtypes.Float32, 2, 32), nil)
		if _, err := fn.FusedQuantizedDense(xGGML, wGGML, nil, compute.FusedQuantizedDenseConfig{
			WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_0},
			WeightLayout:       compute.DenseLayoutInputOutputs,
		}); err == nil {
			t.Errorf("Expected error when QuantGGML uses DenseLayoutInputOutputs")
		}
	})
}

// --- Shared Compliance Test Helpers ---

func makeFloat32Vec(n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = float32((i*5)%11-5) * scale
	}
	return out
}

func makeFloat32PositiveVec(n int, base, step float32) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = base + float32(i)*step
	}
	return out
}

func makeFloat32Matrix(rows, cols int, scale float32) [][]float32 {
	out := make([][]float32, rows)
	for r := range rows {
		out[r] = make([]float32, cols)
		for c := range cols {
			out[r][c] = float32(((r*7+c*5)%13)-6) * scale
		}
	}
	return out
}

func makeInt8Matrix(rows, cols int) [][]int8 {
	out := make([][]int8, rows)
	for r := range rows {
		out[r] = make([]int8, cols)
		for c := range cols {
			out[r][c] = int8(((r*11 + c*7) % 19) - 9)
		}
	}
	return out
}

func makeUint8Matrix(rows, cols, base, span int) [][]uint8 {
	out := make([][]uint8, rows)
	for r := range rows {
		out[r] = make([]uint8, cols)
		for c := range cols {
			out[r][c] = uint8(base + (r*13+c*7)%span)
		}
	}
	return out
}

func transposeInt8Matrix(m [][]int8) [][]int8 {
	rows, cols := len(m), len(m[0])
	out := make([][]int8, cols)
	for c := range cols {
		out[c] = make([]int8, rows)
		for r := range rows {
			out[c][r] = m[r][c]
		}
	}
	return out
}

func transposeUint8Matrix(m [][]uint8) [][]uint8 {
	rows, cols := len(m), len(m[0])
	out := make([][]uint8, cols)
	for c := range cols {
		out[c] = make([]uint8, rows)
		for r := range rows {
			out[c][r] = m[r][c]
		}
	}
	return out
}

func transposeFloat32Matrix(m [][]float32) [][]float32 {
	rows, cols := len(m), len(m[0])
	out := make([][]float32, cols)
	for c := range cols {
		out[c] = make([]float32, rows)
		for r := range rows {
			out[c][r] = m[r][c]
		}
	}
	return out
}

func int8ToF32Matrix(m [][]int8) [][]float32 {
	out := make([][]float32, len(m))
	for r := range m {
		out[r] = make([]float32, len(m[r]))
		for c, v := range m[r] {
			out[r][c] = float32(v)
		}
	}
	return out
}

func uint8ToF32Matrix(m [][]uint8) [][]float32 {
	out := make([][]float32, len(m))
	for r := range m {
		out[r] = make([]float32, len(m[r]))
		for c, v := range m[r] {
			out[r][c] = float32(v)
		}
	}
	return out
}

// pack4BitMatrix packs a 2D [rows, cols] matrix of 4-bit nibbles (cols must be even) into [rows, cols/2] uint8.
func pack4BitMatrix(nibbles [][]uint8) [][]uint8 {
	rows, cols := len(nibbles), len(nibbles[0])
	packed := make([][]uint8, rows)
	for r := range rows {
		packed[r] = make([]uint8, cols/2)
		for j := range cols / 2 {
			packed[r][j] = (nibbles[r][2*j] & 0x0F) | ((nibbles[r][2*j+1] & 0x0F) << 4)
		}
	}
	return packed
}

// pack2BitMatrix packs a 2D [rows, cols] matrix of 2-bit values (cols divisible by 4) into [rows, cols/4] uint8.
func pack2BitMatrix(crumbs [][]uint8) [][]uint8 {
	rows, cols := len(crumbs), len(crumbs[0])
	packed := make([][]uint8, rows)
	for r := range rows {
		packed[r] = make([]uint8, cols/4)
		for j := range cols / 4 {
			packed[r][j] = (crumbs[r][4*j] & 0x03) |
				((crumbs[r][4*j+1] & 0x03) << 2) |
				((crumbs[r][4*j+2] & 0x03) << 4) |
				((crumbs[r][4*j+3] & 0x03) << 6)
		}
	}
	return packed
}

func refQuantLinear2D(
	x [][]float32,
	wIO [][]float32,
	bias []float32,
	act compute.ActivationType,
	scaleZPAt func(k, n int) (scale, zp float32),
) [][]float32 {
	M, K := len(x), len(x[0])
	N := len(wIO[0])
	out := make([][]float32, M)
	for m := range M {
		out[m] = make([]float32, N)
		for n := range N {
			var sum float32
			if bias != nil {
				sum = bias[n]
			}
			for k := range K {
				sc, zp := scaleZPAt(k, n)
				sum += x[m][k] * (wIO[k][n] - zp) * sc
			}
			out[m][n] = refApplyActivationScalar(act, sum)
		}
	}
	return out
}

func refApplyActivationScalar(act compute.ActivationType, v float32) float32 {
	switch act {
	case compute.ActivationNone:
		return v
	case compute.ActivationRelu:
		if v < 0 {
			return 0
		}
		return v
	case compute.ActivationSigmoid:
		return float32(1.0 / (1.0 + math.Exp(float64(-v))))
	case compute.ActivationTanh:
		return float32(math.Tanh(float64(v)))
	case compute.ActivationLeakyRelu:
		if v < 0 {
			return 0.3 * v
		}
		return v
	default:
		return v
	}
}

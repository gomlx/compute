// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/fusedops"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func runLayerNormTest[T float32 | float64](
	t *testing.T,
	name string,
	dtype dtypes.DType,
	inShape shapes.Shape,
	inData []T,
	axes []int,
	epsilon float64,
	gammaData, betaData []T,
) {
	t.Helper()
	backend, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder(name)
	mainFn := builder.Main()

	xNode, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Failed to create x parameter: %+v", err)
	}

	normSize := 1
	for _, a := range axes {
		normSize *= inShape.Dimensions[a]
	}

	var gammaNode, betaNode compute.Value
	var inBuffers []compute.Buffer

	inBuf, err := backend.BufferFromFlatData(0, inData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData x failed: %+v", err)
	}
	inBuffers = append(inBuffers, inBuf)

	paramShape := shapes.Make(dtype, normSize)
	if gammaData != nil {
		gammaNode, err = mainFn.Parameter("gamma", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter gamma failed: %+v", err)
		}
		gBuf, err := backend.BufferFromFlatData(0, gammaData, paramShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData gamma failed: %+v", err)
		}
		inBuffers = append(inBuffers, gBuf)
	}
	if betaData != nil {
		betaNode, err = mainFn.Parameter("beta", paramShape, nil)
		if err != nil {
			t.Fatalf("Parameter beta failed: %+v", err)
		}
		bBuf, err := backend.BufferFromFlatData(0, betaData, paramShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData beta failed: %+v", err)
		}
		inBuffers = append(inBuffers, bBuf)
	}

	outNode, err := mainFn.FusedLayerNorm(xNode, axes, epsilon, gammaNode, betaNode)
	if err != nil {
		t.Fatalf("FusedLayerNorm failed: %+v", err)
	}

	err = mainFn.Return([]compute.Value{outNode}, nil)
	if err != nil {
		t.Fatalf("Return failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %+v", err)
	}
	defer exec.Finalize()

	outputs, err := exec.Execute(inBuffers, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}

	got := outputs[0].(*gobackend.Buffer).Flat.([]T)

	// Compute ground truth reference
	expected := make([]T, len(inData))
	outerSize := len(inData) / normSize
	normSizeF := T(normSize)

	for outer := range outerSize {
		base := outer * normSize
		var sum T
		for i := range normSize {
			sum += inData[base+i]
		}
		mean := sum / normSizeF

		var varSum T
		for i := range normSize {
			diff := inData[base+i] - mean
			varSum += diff * diff
		}
		variance := varSum / normSizeF
		invStd := T(1.0 / math.Sqrt(float64(variance)+epsilon))

		for i := range normSize {
			val := (inData[base+i] - mean) * invStd
			if gammaData != nil {
				val *= gammaData[i]
			}
			if betaData != nil {
				val += betaData[i]
			}
			expected[base+i] = val
		}
	}

	delta := 1e-4
	if dtype == dtypes.Float64 {
		delta = 1e-6
	}
	if ok, diff := testutil.IsInDelta(expected, got, delta); !ok {
		t.Errorf("Mismatch in %s:\n%s", name, diff)
	}
}

func TestSIMDLayerNorm(t *testing.T) {
	t.Run("Float32_Trailing16", func(t *testing.T) {
		// Matching Adult hidden size = 16
		shape := shapes.Make(dtypes.Float32, 4, 16)
		in := make([]float32, 64)
		for i := range 64 {
			in[i] = float32(i + 1)
		}
		gamma := make([]float32, 16)
		beta := make([]float32, 16)
		for i := range 16 {
			gamma[i] = float32(i)*0.1 + 0.5
			beta[i] = float32(i) * 0.2
		}
		runLayerNormTest(t, "Float32_Trailing16_WithAffine", dtypes.Float32, shape, in, []int{1}, 1e-5, gamma, beta)
		runLayerNormTest(t, "Float32_Trailing16_NoAffine", dtypes.Float32, shape, in, []int{1}, 1e-5, nil, nil)
	})

	t.Run("Float32_TrailingOddSize", func(t *testing.T) {
		// Odd size 19 with partial vector tail
		shape := shapes.Make(dtypes.Float32, 3, 19)
		in := make([]float32, 3*19)
		for i := range in {
			in[i] = float32(i*2 - 10)
		}
		gamma := make([]float32, 19)
		beta := make([]float32, 19)
		for i := range 19 {
			gamma[i] = 1.0
			beta[i] = 0.5
		}
		runLayerNormTest(t, "Float32_Trailing19", dtypes.Float32, shape, in, []int{1}, 1e-5, gamma, beta)
	})

	t.Run("Float64_Trailing16", func(t *testing.T) {
		shape := shapes.Make(dtypes.Float64, 4, 16)
		in := make([]float64, 64)
		for i := range 64 {
			in[i] = float64(i + 1)
		}
		gamma := make([]float64, 16)
		beta := make([]float64, 16)
		for i := range 16 {
			gamma[i] = float64(i)*0.1 + 0.5
			beta[i] = float64(i) * 0.2
		}
		runLayerNormTest(t, "Float64_Trailing16", dtypes.Float64, shape, in, []int{1}, 1e-5, gamma, beta)
	})

	t.Run("Float32_MiddleAxes_Spatial", func(t *testing.T) {
		// [B, H, W, C] normalized over axes=[1, 2]
		for _, cDim := range []int{16, 32, 64, 67} {
			B, H, W, C := 4, 8, 8, cDim
			bDim := H * W
			shape := shapes.Make(dtypes.Float32, B, H, W, C)
			in := make([]float32, B*bDim*C)
			for i := range in {
				in[i] = float32((i%89)-44) * 0.1
			}
			gamma := make([]float32, bDim)
			beta := make([]float32, bDim)
			for i := range bDim {
				gamma[i] = 0.5 + float32(i%11)*0.1
				beta[i] = -0.2 + float32(i%7)*0.05
			}

			backend, err := gobackend.NewBackend()
			if err != nil {
				t.Fatalf("Failed to create backend: %+v", err)
			}
			defer backend.Finalize()

			for _, withAffine := range []bool{false, true} {
				builder := backend.Builder("MiddleAxes")
				mainFn := builder.Main()
				xNode, _ := mainFn.Parameter("x", shape, nil)
				inBuf, _ := backend.BufferFromFlatData(0, in, shape)
				inBuffers := []compute.Buffer{inBuf}

				var gNode, bNode compute.Value
				if withAffine {
					pShape := shapes.Make(dtypes.Float32, H, W)
					gNode, _ = mainFn.Parameter("gamma", pShape, nil)
					bNode, _ = mainFn.Parameter("beta", pShape, nil)
					gBuf, _ := backend.BufferFromFlatData(0, gamma, pShape)
					bBuf, _ := backend.BufferFromFlatData(0, beta, pShape)
					inBuffers = append(inBuffers, gBuf, bBuf)
				}
				outNode, err := mainFn.FusedLayerNorm(xNode, []int{1, 2}, 1e-5, gNode, bNode)
				if err != nil {
					t.Fatalf("FusedLayerNorm failed: %+v", err)
				}
				_ = mainFn.Return([]compute.Value{outNode}, nil)
				exec, err := builder.Compile()
				if err != nil {
					t.Fatalf("Compile failed: %+v", err)
				}
				outputs, err := exec.Execute(inBuffers, nil, 0)
				if err != nil {
					t.Fatalf("Execute failed: %+v", err)
				}
				got := outputs[0].(*gobackend.Buffer).Flat.([]float32)

				expected := make([]float32, len(in))
				for a := range B {
					baseA := a * bDim * C
					for c := range C {
						var sum float32
						for b := range bDim {
							sum += in[baseA+b*C+c]
						}
						mean := sum / float32(bDim)
						var varSum float32
						for b := range bDim {
							d := in[baseA+b*C+c] - mean
							varSum += d * d
						}
						invStd := float32(1.0 / math.Sqrt(float64(varSum/float32(bDim))+1e-5))
						for b := range bDim {
							val := (in[baseA+b*C+c] - mean) * invStd
							if withAffine {
								val = val*gamma[b] + beta[b]
							}
							expected[baseA+b*C+c] = val
						}
					}
				}
				if ok, diff := testutil.IsInDelta(expected, got, 1e-4); !ok {
					t.Errorf("Mismatch for C=%d withAffine=%v:\n%s", C, withAffine, diff)
				}
				exec.Finalize()
			}
		}
	})

	t.Run("HalfPrecision_DirectAndFusedResidual", func(t *testing.T) {
		backend, err := gobackend.NewBackend()
		if err != nil {
			t.Fatalf("Failed to create backend: %+v", err)
		}
		defer backend.Finalize()

		outerSize, normSize := 8, 384
		total := outerSize * normSize
		xF32 := make([]float32, total)
		rF32 := make([]float32, total)
		xF16 := make([]float16.Float16, total)
		rF16 := make([]float16.Float16, total)
		xBF16 := make([]bfloat16.BFloat16, total)
		rBF16 := make([]bfloat16.BFloat16, total)
		for i := range total {
			xv := float32((i%41)-20) * 0.125
			rv := float32(((i*7)%37)-18) * 0.125
			xF16[i] = float16.FromFloat32(xv)
			rF16[i] = float16.FromFloat32(rv)
			xBF16[i] = bfloat16.FromFloat32(xv)
			rBF16[i] = bfloat16.FromFloat32(rv)
			xF32[i] = xF16[i].Float32()
			rF32[i] = rF16[i].Float32()
		}
		gammaF32 := make([]float32, normSize)
		betaF32 := make([]float32, normSize)
		for i := range normSize {
			gammaF32[i] = 0.5 + float32(i%9)*0.125
			betaF32[i] = -0.25 + float32(i%5)*0.125
		}
		eps := 1e-5
		wantF32 := make([]float32, total)
		for r := range outerSize {
			base := r * normSize
			var sum float32
			for i := range normSize {
				sum += xF32[base+i] + rF32[base+i]
			}
			mean := sum / float32(normSize)
			var varSum float32
			for i := range normSize {
				d := (xF32[base+i] + rF32[base+i]) - mean
				varSum += d * d
			}
			invStd := float32(1.0 / math.Sqrt(float64(varSum/float32(normSize))+eps))
			for i := range normSize {
				val := ((xF32[base+i] + rF32[base+i]) - mean) * invStd
				wantF32[base+i] = val*gammaF32[i] + betaF32[i]
			}
		}

		// Test Float16 Add -> ConvertDType(Float32) -> FusedLayerNorm -> ConvertDType(Float16) fusion
		shapeF16 := shapes.Make(dtypes.Float16, outerSize, normSize)
		pShapeF32 := shapes.Make(dtypes.Float32, normSize)
		builder := backend.Builder("F16FusedResidualLN")
		mainFn := builder.Main()
		xNode, _ := mainFn.Parameter("x", shapeF16, nil)
		rNode, _ := mainFn.Parameter("r", shapeF16, nil)
		gNode, _ := mainFn.Parameter("g", pShapeF32, nil)
		bNode, _ := mainFn.Parameter("b", pShapeF32, nil)
		sumNode, _ := mainFn.Add(xNode, rNode)
		sumF32, _ := mainFn.ConvertDType(sumNode, dtypes.Float32)
		lnF32, _ := mainFn.FusedLayerNorm(sumF32, []int{1}, eps, gNode, bNode)
		outF16, _ := mainFn.ConvertDType(lnF32, dtypes.Float16)
		_ = mainFn.Return([]compute.Value{outF16}, nil)
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile F16FusedResidualLN failed: %+v", err)
		}
		defer exec.Finalize()

		xBuf, _ := backend.BufferFromFlatData(0, xF16, shapeF16)
		rBuf, _ := backend.BufferFromFlatData(0, rF16, shapeF16)
		gBuf, _ := backend.BufferFromFlatData(0, gammaF32, pShapeF32)
		bBuf, _ := backend.BufferFromFlatData(0, betaF32, pShapeF32)
		outs, err := exec.Execute([]compute.Buffer{xBuf, rBuf, gBuf, bBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute F16FusedResidualLN failed: %+v", err)
		}
		gotF16 := outs[0].(*gobackend.Buffer).Flat.([]float16.Float16)
		gotF32 := make([]float32, total)
		for i, v := range gotF16 {
			gotF32[i] = v.Float32()
		}
		if ok, diff := testutil.IsInDelta(wantF32, gotF32, 5e-3); !ok {
			t.Errorf("F16FusedResidualLN mismatch:\n%s", diff)
		}

		// Test BFloat16 Add -> ConvertDType(Float32) -> FusedLayerNorm -> ConvertDType(BFloat16) fusion
		shapeBF16 := shapes.Make(dtypes.BFloat16, outerSize, normSize)
		builderBF := backend.Builder("BF16FusedResidualLN")
		mainBF := builderBF.Main()
		xNodeBF, _ := mainBF.Parameter("x", shapeBF16, nil)
		rNodeBF, _ := mainBF.Parameter("r", shapeBF16, nil)
		gNodeBF, _ := mainBF.Parameter("g", pShapeF32, nil)
		bNodeBF, _ := mainBF.Parameter("b", pShapeF32, nil)
		sumNodeBF, _ := mainBF.Add(xNodeBF, rNodeBF)
		sumBF32, _ := mainBF.ConvertDType(sumNodeBF, dtypes.Float32)
		lnBF32, _ := mainBF.FusedLayerNorm(sumBF32, []int{1}, eps, gNodeBF, bNodeBF)
		outBF16, _ := mainBF.ConvertDType(lnBF32, dtypes.BFloat16)
		_ = mainBF.Return([]compute.Value{outBF16}, nil)
		execBF, err := builderBF.Compile()
		if err != nil {
			t.Fatalf("Compile BF16FusedResidualLN failed: %+v", err)
		}
		defer execBF.Finalize()

		xBufBF, _ := backend.BufferFromFlatData(0, xBF16, shapeBF16)
		rBufBF, _ := backend.BufferFromFlatData(0, rBF16, shapeBF16)
		gBufBF, _ := backend.BufferFromFlatData(0, gammaF32, pShapeF32)
		bBufBF, _ := backend.BufferFromFlatData(0, betaF32, pShapeF32)
		outsBF, err := execBF.Execute([]compute.Buffer{xBufBF, rBufBF, gBufBF, bBufBF}, nil, 0)
		if err != nil {
			t.Fatalf("Execute BF16FusedResidualLN failed: %+v", err)
		}
		gotBF16 := outsBF[0].(*gobackend.Buffer).Flat.([]bfloat16.BFloat16)
		for i, v := range gotBF16 {
			gotF32[i] = v.Float32()
		}
		if ok, diff := testutil.IsInDelta(wantF32, gotF32, 2e-2); !ok {
			t.Errorf("BF16FusedResidualLN mismatch:\n%s", diff)
		}
	})
}

func TestFusedLayerNormVJP(t *testing.T) {
	backend, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("Failed to create backend: %+v", err)
	}
	defer backend.Finalize()

	t.Run("TrailingAxes", func(t *testing.T) {
		for _, normSize := range []int{16, 19, 64} {
			outerSize := 128 // > 4096 elements for normSize=64 to exercise parallel path
			total := outerSize * normSize
			shape := shapes.Make(dtypes.Float32, outerSize, normSize)
			in := make([]float32, total)
			dy := make([]float32, total)
			for i := range total {
				in[i] = float32((i%41)-20) * 0.1
				dy[i] = float32((i*7%37)-18) * 0.05
			}
			gamma := make([]float32, normSize)
			beta := make([]float32, normSize)
			for i := range normSize {
				gamma[i] = 0.5 + float32(i%9)*0.1
				beta[i] = -0.3 + float32(i%5)*0.1
			}
			eps := 1e-5

			// Ground truth
			wantDX := make([]float32, total)
			wantDG := make([]float32, normSize)
			wantDB := make([]float32, normSize)
			invN := 1.0 / float32(normSize)
			for r := range outerSize {
				base := r * normSize
				var sum float32
				for i := range normSize {
					sum += in[base+i]
				}
				mean := sum * invN
				var varSum, sumG, sumGD float32
				for i := range normSize {
					d := in[base+i] - mean
					varSum += d * d
					g := dy[base+i] * gamma[i]
					sumG += g
					sumGD += g * d
				}
				invStd := float32(1.0 / math.Sqrt(float64(varSum*invN)+eps))
				meanG := sumG * invN
				meanGHat := sumGD * invN * invStd
				for i := range normSize {
					xHat := (in[base+i] - mean) * invStd
					g := dy[base+i] * gamma[i]
					wantDX[base+i] = invStd * (g - meanG - xHat*meanGHat)
					wantDG[i] += dy[base+i] * xHat
					wantDB[i] += dy[base+i]
				}
			}

			builder := backend.Builder("TrailingVJP")
			mainFn := builder.Main()
			xNode, _ := mainFn.Parameter("x", shape, nil)
			dyNode, _ := mainFn.Parameter("dy", shape, nil)
			pShape := shapes.Make(dtypes.Float32, normSize)
			gNode, _ := mainFn.Parameter("gamma", pShape, nil)
			bNode, _ := mainFn.Parameter("beta", pShape, nil)
			dxNode, dgNode, dbNode, err := mainFn.FusedLayerNormVJP(xNode, dyNode, []int{1}, eps, gNode, bNode)
			if err != nil {
				t.Fatalf("FusedLayerNormVJP failed: %+v", err)
			}
			_ = mainFn.Return([]compute.Value{dxNode, dgNode, dbNode}, nil)
			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile failed: %+v", err)
			}
			xBuf, _ := backend.BufferFromFlatData(0, in, shape)
			dyBuf, _ := backend.BufferFromFlatData(0, dy, shape)
			gBuf, _ := backend.BufferFromFlatData(0, gamma, pShape)
			bBuf, _ := backend.BufferFromFlatData(0, beta, pShape)
			outs, err := exec.Execute([]compute.Buffer{xBuf, dyBuf, gBuf, bBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute failed: %+v", err)
			}
			gotDX := outs[0].(*gobackend.Buffer).Flat.([]float32)
			gotDG := outs[1].(*gobackend.Buffer).Flat.([]float32)
			gotDB := outs[2].(*gobackend.Buffer).Flat.([]float32)
			if ok, diff := testutil.IsInDelta(wantDX, gotDX, 1e-4); !ok {
				t.Errorf("TrailingVJP normSize=%d dx mismatch:\n%s", normSize, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDG, gotDG, 1e-3); !ok {
				t.Errorf("TrailingVJP normSize=%d dGamma mismatch:\n%s", normSize, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDB, gotDB, 1e-3); !ok {
				t.Errorf("TrailingVJP normSize=%d dBeta mismatch:\n%s", normSize, diff)
			}
			exec.Finalize()
		}
	})

	t.Run("MiddleAxes_Spatial", func(t *testing.T) {
		for _, cDim := range []int{16, 64, 67} {
			B, H, W, C := 4, 8, 8, cDim
			bDim := H * W
			total := B * bDim * C
			shape := shapes.Make(dtypes.Float32, B, H, W, C)
			pShape := shapes.Make(dtypes.Float32, 1, H, W, 1)
			in := make([]float32, total)
			dy := make([]float32, total)
			for i := range total {
				in[i] = float32((i%89)-44) * 0.1
				dy[i] = float32((i*13%53)-26) * 0.05
			}
			gamma := make([]float32, bDim)
			beta := make([]float32, bDim)
			for i := range bDim {
				gamma[i] = 0.5 + float32(i%11)*0.1
				beta[i] = -0.2 + float32(i%7)*0.05
			}
			eps := 1e-5
			invN := 1.0 / float32(bDim)

			wantDX := make([]float32, total)
			wantDG := make([]float32, bDim)
			wantDB := make([]float32, bDim)
			for a := range B {
				baseA := a * bDim * C
				for c := range C {
					var sum float32
					for b := range bDim {
						sum += in[baseA+b*C+c]
					}
					mean := sum * invN
					var varSum, sumG, sumGD float32
					for b := range bDim {
						d := in[baseA+b*C+c] - mean
						varSum += d * d
						g := dy[baseA+b*C+c] * gamma[b]
						sumG += g
						sumGD += g * d
					}
					invStd := float32(1.0 / math.Sqrt(float64(varSum*invN)+eps))
					meanG := sumG * invN
					meanGHat := sumGD * invN * invStd
					for b := range bDim {
						off := baseA + b*C + c
						xHat := (in[off] - mean) * invStd
						g := dy[off] * gamma[b]
						wantDX[off] = invStd * (g - meanG - xHat*meanGHat)
						wantDG[b] += dy[off] * xHat
						wantDB[b] += dy[off]
					}
				}
			}

			builder := backend.Builder("MiddleAxesVJP")
			mainFn := builder.Main()
			xNode, _ := mainFn.Parameter("x", shape, nil)
			dyNode, _ := mainFn.Parameter("dy", shape, nil)
			gNode, _ := mainFn.Parameter("gamma", pShape, nil)
			bNode, _ := mainFn.Parameter("beta", pShape, nil)
			dxNode, dgNode, dbNode, err := mainFn.FusedLayerNormVJP(xNode, dyNode, []int{1, 2}, eps, gNode, bNode)
			if err != nil {
				t.Fatalf("FusedLayerNormVJP failed: %+v", err)
			}
			_ = mainFn.Return([]compute.Value{dxNode, dgNode, dbNode}, nil)
			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile failed: %+v", err)
			}
			xBuf, _ := backend.BufferFromFlatData(0, in, shape)
			dyBuf, _ := backend.BufferFromFlatData(0, dy, shape)
			gBuf, _ := backend.BufferFromFlatData(0, gamma, pShape)
			bBuf, _ := backend.BufferFromFlatData(0, beta, pShape)
			outs, err := exec.Execute([]compute.Buffer{xBuf, dyBuf, gBuf, bBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute failed: %+v", err)
			}
			gotDX := outs[0].(*gobackend.Buffer).Flat.([]float32)
			gotDG := outs[1].(*gobackend.Buffer).Flat.([]float32)
			gotDB := outs[2].(*gobackend.Buffer).Flat.([]float32)
			if ok, diff := testutil.IsInDelta(wantDX, gotDX, 1e-4); !ok {
				t.Errorf("MiddleAxesVJP C=%d dx mismatch:\n%s", C, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDG, gotDG, 1e-3); !ok {
				t.Errorf("MiddleAxesVJP C=%d dGamma mismatch:\n%s", C, diff)
			}
			if ok, diff := testutil.IsInDelta(wantDB, gotDB, 1e-3); !ok {
				t.Errorf("MiddleAxesVJP C=%d dBeta mismatch:\n%s", C, diff)
			}
			exec.Finalize()
		}
	})
}

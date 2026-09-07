// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/exceptions"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"k8s.io/klog/v2"
)

func RunAllBenchmarks(b *testing.B, backend compute.Backend) {
	b.Run("Softmax", func(b *testing.B) {
		BenchmarkSoftmax(b, backend)
	})
	b.Run("Gelu", func(b *testing.B) {
		BenchmarkGelu(b, backend)
	})
	b.Run("LayerNorm", func(b *testing.B) {
		BenchmarkLayerNorm(b, backend)
	})
	b.Run("Dense", func(b *testing.B) {
		BenchmarkDense(b, backend)
	})
	b.Run("QuantizedDense", func(b *testing.B) {
		BenchmarkQuantizedDense(b, backend)
	})
	b.Run("DotGeneral", func(b *testing.B) {
		BenchmarkDotGeneral(b, backend)
	})
}

// benchMust panics on error, used in benchmark setup.
func benchMust[T any](v T, err error) T {
	if err != nil {
		klog.Errorf("Error: %+v", err)
		panic(err)
	}
	return v
}

// benchExec holds a compiled executable and its input buffers for benchmarking.
type benchExec struct {
	backend compute.Backend
	exec    compute.Executable
	inputs  []compute.Buffer
}

func (be *benchExec) run(b *testing.B) {
	b.Helper()
	// Warm up before the benchmark loop. b.Loop() automatically resets
	// the timer on its first call, so warm-up iterations are not measured.
	for range 3 {
		outputs, err := be.exec.Execute(be.inputs, nil, 0)
		if err != nil {
			if errors.Is(err, compute.ErrNotImplemented) {
				b.Skipf("Execute not implemented: %+v", err)
			}
			b.Fatalf("Execute failed: %+v", err)
		}
		for _, buf := range outputs {
			err = buf.Finalize()
			if err != nil {
				b.Fatalf("Failed to finalize buffer: %+v", err)
			}
		}
	}
	for b.Loop() {
		outputs, err := be.exec.Execute(be.inputs, nil, 0)
		if err != nil {
			if errors.Is(err, compute.ErrNotImplemented) {
				b.Skipf("Execute not implemented: %+v", err)
			}
			b.Fatalf("Execute failed: %+v", err)
		}
		for _, buf := range outputs {
			err = buf.Finalize()
			if err != nil {
				b.Fatalf("Failed to finalize buffer: %+v", err)
			}
		}
	}
	elapsed := b.Elapsed()
	if elapsed > 0 && b.N > 0 {
		durationPerOp := time.Duration(float64(elapsed) / float64(b.N))
		durStr := humanize.Duration(durationPerOp)
		splitIdx := strings.IndexFunc(durStr, func(r rune) bool {
			return !unicode.IsDigit(r) && r != '.' && r != '-'
		})
		if splitIdx > 0 {
			valStr := durStr[:splitIdx]
			unitStr := durStr[splitIdx:]
			if strings.ContainsAny(unitStr, "0123456789") {
				b.ReportMetric(durationPerOp.Seconds(), "s/op")
			} else if val, err := strconv.ParseFloat(valStr, 64); err == nil {
				b.ReportMetric(val, unitStr+"/op")
			}
		}
	}
}

// newBenchExec builds, compiles, and prepares inputs for a benchmark.
func newBenchExec(backend compute.Backend, inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f compute.Function, params []compute.Value) (compute.Value, error),
) (*benchExec, error) {
	be := &benchExec{backend: backend}
	var buildErr error
	panicErr := exceptions.TryCatch[error](func() {
		be.exec, be.inputs, buildErr = buildGraph(backend, inputShapes, inputDatas, buildFn)
	})
	if panicErr != nil {
		return nil, panicErr
	}
	if buildErr != nil {
		return nil, buildErr
	}
	return be, nil
}

// buildGraph compiles a backend graph from the given input shapes and build function,
// and creates input buffers from the provided data. Used by both test and benchmark helpers.
func buildGraph(backend compute.Backend, inputShapes []shapes.Shape, inputDatas []any,
	buildFn func(f compute.Function, params []compute.Value) (compute.Value, error),
) (compute.Executable, []compute.Buffer, error) {
	builder := backend.Builder("test")
	mainFn := builder.Main()

	params := make([]compute.Value, len(inputShapes))
	for i, s := range inputShapes {
		p, err := mainFn.Parameter(fmt.Sprintf("x%d", i), s, nil)
		if err != nil {
			return nil, nil, err
		}
		params[i] = p
	}

	out, err := buildFn(mainFn, params)
	if err != nil {
		return nil, nil, err
	}

	if err := mainFn.Return([]compute.Value{out}, nil); err != nil {
		return nil, nil, err
	}

	exec, err := builder.Compile()
	if err != nil {
		return nil, nil, err
	}

	inputs := make([]compute.Buffer, len(inputDatas))
	for i, data := range inputDatas {
		buf, err := backend.BufferFromFlatData(0, data, inputShapes[i])
		if err != nil {
			return nil, nil, err
		}
		inputs[i] = buf
	}

	return exec, inputs, nil
}

// reduceAndKeep performs ReduceMax or ReduceSum, reshapes back to preserve the rank, and finally broadcasts it back to the original shape.
func reduceAndKeep(f compute.Function, x compute.Value, reduceFn func(compute.Value, ...int) (compute.Value, error), shape shapes.Shape, axis int) compute.Value {
	reduced := benchMust(reduceFn(x, axis))
	// Reshape to keep dimension: insert a size-1 at the axis position.
	keepDims := make([]int, shape.Rank())
	copy(keepDims, shape.Dimensions)
	keepDims[axis] = 1
	reshaped := benchMust(f.Reshape(reduced, keepDims...))

	// Broadcast back to original shape
	// TODO: improve "implicit broadcasting" of operations: this shouldn't be necessary, but without it, it's much slower.
	broadcastAxes := make([]int, shape.Rank())
	for i := range broadcastAxes {
		broadcastAxes[i] = i
	}
	return benchMust(f.BroadcastInDim(reshaped, shape, broadcastAxes))
}

func randomFloat32(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return data
}

func randomBFloat16(n int) []bfloat16.BFloat16 {
	data := make([]bfloat16.BFloat16, n)
	for i := range data {
		data[i] = bfloat16.FromFloat32(rand.Float32()*2 - 1)
	}
	return data
}

// --- Softmax Benchmarks ---

func BenchmarkSoftmax(b *testing.B, backend compute.Backend) {
	sizes := []struct {
		name string
		dims []int
		axis int
	}{
		{"8x64_axis1", []int{8, 64}, 1},
		{"32x128_axis1", []int{32, 128}, 1},
		{"64x512_axis1", []int{64, 512}, 1},
		{"8x16x64_axis2", []int{8, 16, 64}, 2},
		{"4x8x32x128_axis3", []int{4, 8, 32, 128}, 3},
	}

	for _, sz := range sizes {
		shape := shapes.Make(dtypes.Float32, sz.dims...)
		data := randomFloat32(shape.Size())
		axis := sz.axis

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) {
			fused, err := newBenchExec(backend, []shapes.Shape{shape}, []any{data},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.FusedSoftmax(params[0], axis)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create fused benchmark: %+v", err)
			}
			fused.run(b)
		})

		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			decomposed, err := newBenchExec(backend, []shapes.Shape{shape}, []any{data},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					x := params[0]
					maxVal := reduceAndKeep(f, x, f.ReduceMax, shape, axis)
					shifted := benchMust(f.Sub(x, maxVal))
					exps := benchMust(f.Exp(shifted))
					sumExps := reduceAndKeep(f, exps, f.ReduceSum, shape, axis)
					return f.Div(exps, sumExps)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create decomposed benchmark: %+v", err)
			}
			decomposed.run(b)
		})
	}
}

// --- GELU Benchmarks ---

func BenchmarkGelu(b *testing.B, backend compute.Backend) {
	sizes := []struct {
		name string
		dims []int
	}{
		{"512", []int{512}},
		{"4096", []int{4096}},
		{"32x1024", []int{32, 1024}},
		{"64x4096", []int{64, 4096}},
	}

	for _, sz := range sizes {
		shape := shapes.Make(dtypes.Float32, sz.dims...)
		data := randomFloat32(shape.Size())

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) {
			fused, err := newBenchExec(backend, []shapes.Shape{shape}, []any{data},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.FusedActivation(params[0], compute.ActivationConfig{Type: compute.ActivationGelu})
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create fused benchmark: %+v", err)
			}
			fused.run(b)
		})

		// Decomposed GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			decomposed, err := newBenchExec(backend, []shapes.Shape{shape}, []any{data},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					x := params[0]
					sqrt2Inv := benchMust(f.Constant([]float32{float32(1.0 / 1.4142135623730951)}, 1))
					sqrt2InvBroadcast := benchMust(f.BroadcastInDim(sqrt2Inv, shape, []int{0}))
					half := benchMust(f.Constant([]float32{0.5}, 1))
					halfBroadcast := benchMust(f.BroadcastInDim(half, shape, []int{0}))
					one := benchMust(f.Constant([]float32{1.0}, 1))
					oneBroadcast := benchMust(f.BroadcastInDim(one, shape, []int{0}))

					scaled := benchMust(f.Mul(x, sqrt2InvBroadcast))
					erfVal := benchMust(f.Erf(scaled))
					onePlusErf := benchMust(f.Add(oneBroadcast, erfVal))
					xHalf := benchMust(f.Mul(x, halfBroadcast))
					return f.Mul(xHalf, onePlusErf)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create decomposed benchmark: %+v", err)
			}
			decomposed.run(b)
		})
	}
}

// --- LayerNorm Benchmarks ---

func BenchmarkLayerNorm(b *testing.B, backend compute.Backend) {
	sizes := []struct {
		name string
		dims []int
		axis int
	}{
		{"8x64_axis1", []int{8, 64}, 1},
		{"32x256_axis1", []int{32, 256}, 1},
		{"64x768_axis1", []int{64, 768}, 1},
		{"8x16x64_axis2", []int{8, 16, 64}, 2},
	}

	for _, sz := range sizes {
		shape := shapes.Make(dtypes.Float32, sz.dims...)
		data := randomFloat32(shape.Size())
		normDim := sz.dims[sz.axis]
		gammaData := randomFloat32(normDim)
		betaData := randomFloat32(normDim)
		gammaShape := shapes.Make(dtypes.Float32, normDim)
		betaShape := shapes.Make(dtypes.Float32, normDim)
		axis := sz.axis

		allShapes := []shapes.Shape{shape, gammaShape, betaShape}
		allDatas := []any{data, gammaData, betaData}

		b.Run(fmt.Sprintf("Fused/%s", sz.name), func(b *testing.B) {
			fused, err := newBenchExec(backend, allShapes, allDatas,
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.FusedLayerNorm(params[0], []int{axis}, 1e-5, params[1], params[2])
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create fused benchmark: %+v", err)
			}
			fused.run(b)
		})

		// Decomposed: mean, variance, normalize, scale, offset.
		b.Run(fmt.Sprintf("Decomposed/%s", sz.name), func(b *testing.B) {
			decomposed, err := newBenchExec(backend, allShapes, allDatas,
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					x := params[0]
					gamma := params[1]
					beta := params[2]

					// Compute normSize as float constant.
					normSizeF := float32(sz.dims[axis])
					normSizeConst := benchMust(f.Constant([]float32{normSizeF}, 1))
					normSizeBroadcast := benchMust(f.BroadcastInDim(normSizeConst, shape, []int{0}))

					// Mean.
					sum := reduceAndKeep(f, x, f.ReduceSum, shape, axis)
					mean := benchMust(f.Div(sum, normSizeBroadcast))

					// Variance.
					diff := benchMust(f.Sub(x, mean))
					diffSq := benchMust(f.Mul(diff, diff))
					varSum := reduceAndKeep(f, diffSq, f.ReduceSum, shape, axis)
					variance := benchMust(f.Div(varSum, normSizeBroadcast))

					// Normalize.
					epsConst := benchMust(f.Constant([]float32{1e-5}, 1))
					epsBroadcast := benchMust(f.BroadcastInDim(epsConst, shape, []int{0}))
					varPlusEps := benchMust(f.Add(variance, epsBroadcast))
					invStd := benchMust(f.Rsqrt(varPlusEps))
					normalized := benchMust(f.Mul(diff, invStd))

					// Scale and offset: gamma and beta have shape [normDim], need to broadcast.
					broadcastShape := shape.Clone()
					for i := range broadcastShape.Dimensions {
						broadcastShape.Dimensions[i] = 1
					}
					broadcastShape.Dimensions[axis] = normDim
					gammaReshaped := benchMust(f.Reshape(gamma, broadcastShape.Dimensions...))
					broadcastAxes := make([]int, shape.Rank())
					for i := range broadcastAxes {
						broadcastAxes[i] = i
					}
					gammaBroadcast := benchMust(f.BroadcastInDim(gammaReshaped, shape, broadcastAxes))
					scaled := benchMust(f.Mul(normalized, gammaBroadcast))

					betaReshaped := benchMust(f.Reshape(beta, broadcastShape.Dimensions...))
					betaBroadcast := benchMust(f.BroadcastInDim(betaReshaped, shape, broadcastAxes))
					return f.Add(scaled, betaBroadcast)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create decomposed benchmark: %+v", err)
			}
			decomposed.run(b)
		})
	}
}

// --- Dense Benchmarks ---

func BenchmarkDense(b *testing.B, backend compute.Backend) {
	sizes := []struct {
		name        string
		batch       int
		inFeatures  int
		outFeatures int
	}{
		// UCI Adult demo test cases (from github.com/gomlx/gomlx/examples/adult)
		{"adult-demo/128x69x4", 128, 69, 4},
		{"adult-demo/128x4x1", 128, 4, 1},
		{"adult-demo/25x69x4", 25, 69, 4},
		{"adult-demo/49x69x4", 49, 69, 4},

		// Standard dense benchmark cases
		{"standard/1x64x64", 1, 64, 64},
		{"standard/8x128x256", 8, 128, 256},
		{"standard/32x512x1024", 32, 512, 1024},

		// Transformer / Large matrix cases
		{"all-MiniLM/416x384x1152", 416, 384, 1152},
		{"large/1024x1024x1024", 1024, 1024, 1024},
	}

	activationsList := []struct {
		name string
		act  compute.ActivationType
	}{
		{"None", compute.ActivationNone},
		{"Relu", compute.ActivationRelu},
		{"Gelu", compute.ActivationGelu},
		{"Silu", compute.ActivationSilu},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			xShapeF32 := shapes.Make(dtypes.Float32, sz.batch, sz.inFeatures)
			wShapeF32 := shapes.Make(dtypes.Float32, sz.inFeatures, sz.outFeatures)
			bShapeF32 := shapes.Make(dtypes.Float32, sz.outFeatures)
			outShapeF32 := shapes.Make(dtypes.Float32, sz.batch, sz.outFeatures)

			xDataF32 := randomFloat32(xShapeF32.Size())
			wDataF32 := randomFloat32(wShapeF32.Size())
			biasDataF32 := randomFloat32(bShapeF32.Size())

			allShapesF32 := []shapes.Shape{xShapeF32, wShapeF32, bShapeF32}
			allDatasF32 := []any{xDataF32, wDataF32, biasDataF32}

			// Fused Float32 across activations
			for _, act := range activationsList {
				b.Run(fmt.Sprintf("Float32/Fused/%s", act.name), func(b *testing.B) {
					fused, err := newBenchExec(backend, allShapesF32, allDatasF32,
						func(f compute.Function, params []compute.Value) (compute.Value, error) {
							return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{Activation: act.act})
						})
					if err != nil {
						if errors.Is(err, compute.ErrNotImplemented) {
							b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
						}
						b.Fatalf("Failed to create fused benchmark: %+v", err)
					}
					fused.run(b)
				})
			}

			// Decomposed Float32: DotGeneral + bias add
			b.Run("Float32/Decomposed/None", func(b *testing.B) {
				decomposed, err := newBenchExec(backend, allShapesF32, allDatasF32,
					func(f compute.Function, params []compute.Value) (compute.Value, error) {
						x := params[0]
						weight := params[1]
						bias := params[2]

						y := benchMust(f.DotGeneral(x, []int{1}, nil, weight, []int{0}, nil, compute.DotGeneralConfig{}))
						biasBroadcast := benchMust(f.BroadcastInDim(bias, outShapeF32, []int{1}))
						return f.Add(y, biasBroadcast)
					})
				if err != nil {
					if errors.Is(err, compute.ErrNotImplemented) {
						b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
					}
					b.Fatalf("Failed to create decomposed benchmark: %+v", err)
				}
				decomposed.run(b)
			})

			// Fused BFloat16 across activations
			xShapeBF16 := shapes.Make(dtypes.BFloat16, sz.batch, sz.inFeatures)
			wShapeBF16 := shapes.Make(dtypes.BFloat16, sz.inFeatures, sz.outFeatures)
			bShapeBF16 := shapes.Make(dtypes.BFloat16, sz.outFeatures)

			xDataBF16 := randomBFloat16(xShapeBF16.Size())
			wDataBF16 := randomBFloat16(wShapeBF16.Size())
			biasDataBF16 := randomBFloat16(bShapeBF16.Size())

			allShapesBF16 := []shapes.Shape{xShapeBF16, wShapeBF16, bShapeBF16}
			allDatasBF16 := []any{xDataBF16, wDataBF16, biasDataBF16}

			for _, act := range []string{"None", "Relu"} {
				actType := compute.ActivationNone
				if act == "Relu" {
					actType = compute.ActivationRelu
				}
				b.Run(fmt.Sprintf("BFloat16/Fused/%s", act), func(b *testing.B) {
					fused, err := newBenchExec(backend, allShapesBF16, allDatasBF16,
						func(f compute.Function, params []compute.Value) (compute.Value, error) {
							return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{Activation: actType})
						})
					if err != nil {
						if errors.Is(err, compute.ErrNotImplemented) {
							b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
						}
						b.Fatalf("Failed to create fused BF16 benchmark: %+v", err)
					}
					fused.run(b)
				})
			}
		})
	}
}

// --- QuantizedDense Benchmarks ---

func randomInt8(n int) []int8 {
	data := make([]int8, n)
	for i := range data {
		data[i] = int8(rand.IntN(256) - 128)
	}
	return data
}

func randomUint8(n int) []uint8 {
	data := make([]uint8, n)
	for i := range data {
		data[i] = uint8(rand.IntN(256))
	}
	return data
}

// BenchmarkQuantizedDense compares fused quantized-dense (which uses highway SIMD
// when available) against a decomposed path and float32 Dense as a reference.
//
// The decomposed path is only benchmarked for Int8 because the go backend
// does not support bitwise shift ops needed for NF4/Int4 nibble extraction.
func BenchmarkQuantizedDense(b *testing.B, backend compute.Backend) {
	sizes := []struct {
		name        string
		batch       int
		inFeatures  int
		outFeatures int
		groupSize   int
	}{
		{"1x64x64_g64", 1, 64, 64, 64},
		{"8x128x256_g128", 8, 128, 256, 128},
		{"32x512x1024_g128", 32, 512, 1024, 128},
	}

	for _, sz := range sizes {
		M, K, N := sz.batch, sz.inFeatures, sz.outFeatures
		groupSize := sz.groupSize
		numGroups := (N + groupSize - 1) / groupSize
		xData := randomFloat32(M * K)
		biasData := randomFloat32(N)
		scalesData := randomFloat32(K * numGroups)

		xShape := shapes.Make(dtypes.Float32, M, K)
		biasShape := shapes.Make(dtypes.Float32, N)
		scalesShape := shapes.Make(dtypes.Float32, K, numGroups)
		outShape := shapes.Make(dtypes.Float32, M, N)

		// --- NF4 ---
		nf4Data := randomUint8(K * N)
		nf4Shape := shapes.Make(dtypes.Uint8, K, N)

		b.Run(fmt.Sprintf("NF4/Fused/%s", sz.name), func(b *testing.B) {
			nf4Fused, err := newBenchExec(backend, []shapes.Shape{xShape, nf4Shape, scalesShape, biasShape},
				[]any{xData, nf4Data, scalesData, biasData},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(params[0], params[1], params[3],
						&compute.Quantization{Scheme: compute.QuantNF4, Scale: params[2], BlockAxis: 1, BlockSize: groupSize},
						compute.ActivationNone)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create NF4/Fused benchmark: %+v", err)
			}
			nf4Fused.run(b)
		})

		// --- Linear Int8 (second set) ---
		int4WeightsData := randomInt8(K * N)
		int4WeightsShape := shapes.Make(dtypes.Int8, K, N)

		b.Run(fmt.Sprintf("LinearInt8_2/Fused/%s", sz.name), func(b *testing.B) {
			int4Fused, err := newBenchExec(
				backend, []shapes.Shape{xShape, int4WeightsShape, scalesShape, biasShape},
				[]any{xData, int4WeightsData, scalesData, biasData},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(params[0], params[1], params[3],
						&compute.Quantization{Scheme: compute.QuantLinear, Scale: params[2], BlockAxis: 1, BlockSize: groupSize},
						compute.ActivationNone)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create LinearInt8_2/Fused benchmark: %+v", err)
			}
			int4Fused.run(b)
		})

		// --- Int8 ---
		int8WeightsData := randomInt8(K * N)
		int8WeightsShape := shapes.Make(dtypes.Int8, K, N)

		b.Run(fmt.Sprintf("Int8/Fused/%s", sz.name), func(b *testing.B) {
			int8Fused, err := newBenchExec(backend, []shapes.Shape{xShape, int8WeightsShape, scalesShape, biasShape},
				[]any{xData, int8WeightsData, scalesData, biasData},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(params[0], params[1], params[3],
						&compute.Quantization{Scheme: compute.QuantLinear, Scale: params[2], BlockAxis: 1, BlockSize: groupSize},
						compute.ActivationNone)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create Int8/Fused benchmark: %+v", err)
			}
			int8Fused.run(b)
		})

		// Int8 Decomposed: ConvertDType + Mul(scales) + DotGeneral + bias.
		// Scales are pre-expanded from [K, numGroups] to [K, N] in Go because
		// the Gather-based expansion is a small fraction of total cost and
		// keeps the benchmark focused on the materialization overhead.
		expandedScalesData := make([]float32, K*N)
		for k := range K {
			for n := range N {
				expandedScalesData[k*N+n] = scalesData[k*numGroups+n/groupSize]
			}
		}
		expandedScalesShape := shapes.Make(dtypes.Float32, K, N)

		b.Run(fmt.Sprintf("Int8/Decomposed/%s", sz.name), func(b *testing.B) {
			int8Decomposed, err := newBenchExec(backend,
				[]shapes.Shape{xShape, int8WeightsShape, expandedScalesShape, biasShape},
				[]any{xData, int8WeightsData, expandedScalesData, biasData},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					x := params[0]
					weights := params[1]
					expandedScales := params[2]
					bias := params[3]

					// Dequantize: float32(int8) * scales → [K, N] float32.
					wFloat := benchMust(f.ConvertDType(weights, dtypes.Float32))
					wDequant := benchMust(f.Mul(wFloat, expandedScales))

					// Matmul: x [M, K] @ wDequant [K, N] → [M, N].
					y := benchMust(f.DotGeneral(x, []int{1}, nil, wDequant, []int{0}, nil, compute.DotGeneralConfig{}))

					// Add bias.
					biasBroadcast := benchMust(f.BroadcastInDim(bias, outShape, []int{1}))
					return f.Add(y, biasBroadcast)
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create Int8/Decomposed benchmark: %+v", err)
			}
			int8Decomposed.run(b)
		})

		// Float32 Dense reference (same M×K×N, full-precision weights).
		f32WeightsData := randomFloat32(K * N)
		f32WeightsShape := shapes.Make(dtypes.Float32, K, N)

		b.Run(fmt.Sprintf("Float32Dense/%s", sz.name), func(b *testing.B) {
			f32Dense, err := newBenchExec(backend,
				[]shapes.Shape{xShape, f32WeightsShape, biasShape},
				[]any{xData, f32WeightsData, biasData},
				func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.FusedDense(params[0], params[1], params[2], compute.DenseConfig{Activation: compute.ActivationNone})
				})
			if err != nil {
				if errors.Is(err, compute.ErrNotImplemented) {
					b.Skipf("Skipping benchmark %s: %+v", sz.name, err)
				}
				b.Fatalf("Failed to create Float32Dense benchmark: %+v", err)
			}
			f32Dense.run(b)
		})
	}
}

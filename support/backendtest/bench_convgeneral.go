// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"errors"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
)

// BenchmarkConvGeneral runs standard 1D and 2D convolution benchmarks (forward, VJP wrt input, and VJP wrt kernel).
func BenchmarkConvGeneral(b *testing.B, backend compute.Backend) {
	type convGeneralCase struct {
		name            string
		model           string // model if set groups subsequent cases under a sub-benchmark.
		inputShape      shapes.Shape
		kernelShape     shapes.Shape
		axes            compute.ConvolveAxesConfig
		strides         []int
		paddings        [][2]int
		inputDilations  []int
		kernelDilations []int
	}

	axes1DFwd := compute.ConvolveAxesConfig{
		InputBatch:           0,
		InputSpatial:         []int{1},
		InputChannels:        2,
		KernelSpatial:        []int{0},
		KernelInputChannels:  1,
		KernelOutputChannels: 2,
		OutputBatch:          0,
		OutputSpatial:        []int{1},
		OutputChannels:       2,
	}
	axes2DFwd := compute.ConvolveAxesConfig{
		InputBatch:           0,
		InputSpatial:         []int{1, 2},
		InputChannels:        3,
		KernelSpatial:        []int{0, 1},
		KernelInputChannels:  2,
		KernelOutputChannels: 3,
		OutputBatch:          0,
		OutputSpatial:        []int{1, 2},
		OutputChannels:       3,
	}
	axes2DVJPInput := compute.ConvolveAxesConfig{
		InputBatch:           0,
		InputSpatial:         []int{1, 2},
		InputChannels:        3,
		KernelSpatial:        []int{0, 1},
		KernelInputChannels:  3,
		KernelOutputChannels: 2,
		OutputBatch:          0,
		OutputSpatial:        []int{1, 2},
		OutputChannels:       3,
	}
	axes2DVJPKernel := compute.ConvolveAxesConfig{
		InputBatch:           3,
		InputSpatial:         []int{1, 2},
		InputChannels:        0,
		KernelSpatial:        []int{1, 2},
		KernelInputChannels:  0,
		KernelOutputChannels: 3,
		OutputBatch:          2,
		OutputSpatial:        []int{0, 1},
		OutputChannels:       3,
	}

	benchCases := []convGeneralCase{
		{
			model: "Small1D",
		},
		{
			name:        "Fwd:4x64x16_k3x16x32",
			inputShape:  shapes.Make(dtypes.Float32, 4, 64, 16),
			kernelShape: shapes.Make(dtypes.Float32, 3, 16, 32),
			axes:        axes1DFwd,
			strides:     []int{1},
			paddings:    [][2]int{{1, 1}},
		},
		{
			name:        "FwdStrided:8x128x32_k5x32x64_s2",
			inputShape:  shapes.Make(dtypes.Float32, 8, 128, 32),
			kernelShape: shapes.Make(dtypes.Float32, 5, 32, 64),
			axes:        axes1DFwd,
			strides:     []int{2},
			paddings:    [][2]int{{2, 2}},
		},

		{
			model: "Small2D",
		},
		{
			name:        "Fwd1x1:4x14x14x32_k1x1x32x64",
			inputShape:  shapes.Make(dtypes.Float32, 4, 14, 14, 32),
			kernelShape: shapes.Make(dtypes.Float32, 1, 1, 32, 64),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{0, 0}, {0, 0}},
		},
		{
			name:        "Fwd3x3:4x28x28x1_k3x3x1x32",
			inputShape:  shapes.Make(dtypes.Float32, 4, 28, 28, 1),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 1, 32),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Fwd3x3:4x14x14x32_k3x3x32x64",
			inputShape:  shapes.Make(dtypes.Float32, 4, 14, 14, 32),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 32, 64),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Fwd3x3Strided:4x28x28x16_k3x3x16x32_s2",
			inputShape:  shapes.Make(dtypes.Float32, 4, 28, 28, 16),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 16, 32),
			axes:        axes2DFwd,
			strides:     []int{2, 2},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPInput3x3:4x14x14x64_k3x3x32x64",
			inputShape:  shapes.Make(dtypes.Float32, 4, 14, 14, 64),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 32, 64),
			axes:        axes2DVJPInput,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPKernel3x3:4x14x14x32_v4x14x14x64",
			inputShape:  shapes.Make(dtypes.Float32, 4, 14, 14, 32),
			kernelShape: shapes.Make(dtypes.Float32, 4, 14, 14, 64),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},

		{
			model: "MNIST-CNN",
		},
		{
			name:        "Conv1Fwd:32x28x28x1_k3x3x1x32",
			inputShape:  shapes.Make(dtypes.Float32, 32, 28, 28, 1),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 1, 32),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Conv2Fwd:32x14x14x32_k3x3x32x64",
			inputShape:  shapes.Make(dtypes.Float32, 32, 14, 14, 32),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 32, 64),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Conv2VJPInput:32x14x14x64_k3x3x32x64",
			inputShape:  shapes.Make(dtypes.Float32, 32, 14, 14, 64),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 32, 64),
			axes:        axes2DVJPInput,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Conv1VJPKernel:32x28x28x1_v32x28x28x32",
			inputShape:  shapes.Make(dtypes.Float32, 32, 28, 28, 1),
			kernelShape: shapes.Make(dtypes.Float32, 32, 28, 28, 32),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Conv2VJPKernel:32x14x14x32_v32x14x14x64",
			inputShape:  shapes.Make(dtypes.Float32, 32, 14, 14, 32),
			kernelShape: shapes.Make(dtypes.Float32, 32, 14, 14, 64),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},

		{
			model: "Diffusion-LargeBatch",
		},
		{
			name:        "Fwd3x3:32x64x64x32_k3x3x32x32",
			inputShape:  shapes.Make(dtypes.Float32, 32, 64, 64, 32),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 32, 32),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Fwd3x3:32x32x32x64_k3x3x64x64",
			inputShape:  shapes.Make(dtypes.Float32, 32, 32, 32, 64),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 64, 64),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPInput3x3:32x64x64x32_k3x3x32x32",
			inputShape:  shapes.Make(dtypes.Float32, 32, 64, 64, 32),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 32, 32),
			axes:        axes2DVJPInput,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPKernel3x3:32x64x64x32_v32x64x64x32",
			inputShape:  shapes.Make(dtypes.Float32, 32, 64, 64, 32),
			kernelShape: shapes.Make(dtypes.Float32, 32, 64, 64, 32),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPKernel3x3:32x32x32x64_v32x32x32x64",
			inputShape:  shapes.Make(dtypes.Float32, 32, 32, 32, 64),
			kernelShape: shapes.Make(dtypes.Float32, 32, 32, 32, 64),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPKernel3x3:32x16x16x96_v32x16x16x96",
			inputShape:  shapes.Make(dtypes.Float32, 32, 16, 16, 96),
			kernelShape: shapes.Make(dtypes.Float32, 32, 16, 16, 96),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},

		{
			model: "SmallBatch-LargeSpatial",
		},
		{
			name:        "Fwd3x3:1x256x256x32_k3x3x32x32",
			inputShape:  shapes.Make(dtypes.Float32, 1, 256, 256, 32),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 32, 32),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "Fwd3x3:2x128x128x64_k3x3x64x64",
			inputShape:  shapes.Make(dtypes.Float32, 2, 128, 128, 64),
			kernelShape: shapes.Make(dtypes.Float32, 3, 3, 64, 64),
			axes:        axes2DFwd,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPKernel3x3:1x256x256x32_v1x256x256x32",
			inputShape:  shapes.Make(dtypes.Float32, 1, 256, 256, 32),
			kernelShape: shapes.Make(dtypes.Float32, 1, 256, 256, 32),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
		{
			name:        "VJPKernel3x3:2x128x128x64_v2x128x128x64",
			inputShape:  shapes.Make(dtypes.Float32, 2, 128, 128, 64),
			kernelShape: shapes.Make(dtypes.Float32, 2, 128, 128, 64),
			axes:        axes2DVJPKernel,
			strides:     []int{1, 1},
			paddings:    [][2]int{{1, 1}, {1, 1}},
		},
	}

	var benchIdx int
	for benchIdx < len(benchCases) {
		benchCase := benchCases[benchIdx]
		for benchCase.model == "" {
			benchIdx++
			if benchIdx >= len(benchCases) {
				return
			}
			benchCase = benchCases[benchIdx]
		}
		benchIdx++

		b.Run(benchCase.model, func(b *testing.B) {
			for innerBenchIdx := benchIdx; innerBenchIdx < len(benchCases); innerBenchIdx++ {
				tc := benchCases[innerBenchIdx]
				if tc.model != "" {
					break
				}

				inData := randomFloat32(tc.inputShape.Size())
				kerData := randomFloat32(tc.kernelShape.Size())

				b.Run(tc.name, func(b *testing.B) {
					be, err := newBenchExec(
						backend,
						[]shapes.Shape{tc.inputShape, tc.kernelShape},
						[]any{inData, kerData},
						func(f compute.Function, params []compute.Value) (compute.Value, error) {
							return f.ConvGeneral(
								params[0], params[1], tc.axes,
								tc.strides, tc.paddings,
								tc.inputDilations, tc.kernelDilations,
								1, 1,
							)
						},
					)
					if err != nil {
						if errors.Is(err, compute.ErrNotImplemented) {
							b.Skipf("Skipping benchmark %s: %+v", tc.name, err)
						}
						b.Fatalf("Failed to create benchmark %s: %+v", tc.name, err)
					}

					outShape, err := shapeinference.ConvGeneral(
						tc.inputShape, tc.kernelShape, tc.axes,
						tc.strides, tc.paddings,
						tc.inputDilations, tc.kernelDilations,
						1, 1,
					)
					var flops float64
					if err == nil {
						kernelSpatialSize := 1
						for _, axis := range tc.axes.KernelSpatial {
							kernelSpatialSize *= tc.kernelShape.Dimensions[axis]
						}
						contractingChannels := tc.kernelShape.Dimensions[tc.axes.KernelInputChannels]
						flops = float64(2 * outShape.Size() * kernelSpatialSize * contractingChannels)
					}

					be.run(b)

					elapsed := b.Elapsed()
					if flops > 0 && elapsed > 0 && b.N > 0 {
						gflops := (flops * float64(b.N)) / (elapsed.Seconds() * 1e9)
						b.ReportMetric(gflops, "GFlops/s")
					}
				})
			}
		})
	}
}

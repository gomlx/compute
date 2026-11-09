// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
)

func compareConvBuffers(t *testing.T, label string, gotBuf, wantBuf *gobackend.Buffer, tol float64) {
	t.Helper()
	got := gotBuf.Flat.([]float32)
	want := wantBuf.Flat.([]float32)
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		relTol := tol * math.Max(1.0, math.Abs(float64(want[i])))
		if diff > relTol || math.IsNaN(float64(got[i])) {
			t.Fatalf("%s: mismatch at flat idx %d: got %v, want %v (diff %v > %v)",
				label, i, got[i], want[i], diff, relTol)
		}
	}
}

func makeDeterministicBuffer(t *testing.T, backend *gobackend.Backend, shape shapes.Shape, seed float32) *gobackend.Buffer {
	buf, err := backend.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer failed: %+v", err)
	}
	flat := buf.Flat.([]float32)
	for i := range flat {
		v := float32(((i*73+19)%101)-50) * 0.01
		flat[i] = v + seed*0.001
	}
	return buf
}

func TestConvGeneralAVX512(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}

	bIntf, err := gobackend.New("")
	if err != nil {
		t.Fatalf("gobackend.New failed: %+v", err)
	}
	backend := bIntf.(*gobackend.Backend)
	defer backend.Finalize()

	type convCase struct {
		name             string
		inputDims        []int
		kernelDims       []int
		axes             compute.ConvolveAxesConfig
		strides          []int
		paddings         [][2]int
		inputDilations   []int
		kernelDilations  []int
		expectedKind     ops.Conv2DKind
		expectIs1x1Unit  bool
		expectTransposed bool
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
	axes1DVJPInput := compute.ConvolveAxesConfig{
		InputBatch:           0,
		InputSpatial:         []int{1},
		InputChannels:        2,
		KernelSpatial:        []int{0},
		KernelInputChannels:  2,
		KernelOutputChannels: 1,
		OutputBatch:          0,
		OutputSpatial:        []int{1},
		OutputChannels:       2,
	}
	axes1DVJPKernel := compute.ConvolveAxesConfig{
		InputBatch:           2,
		InputSpatial:         []int{1},
		InputChannels:        0,
		KernelSpatial:        []int{1},
		KernelInputChannels:  0,
		KernelOutputChannels: 2,
		OutputBatch:          1,
		OutputSpatial:        []int{0},
		OutputChannels:       2,
	}

	cases := []convCase{
		{
			name:         "2D_Fwd_MNIST_Conv1",
			inputDims:    []int{4, 28, 28, 1},
			kernelDims:   []int{3, 3, 1, 32},
			axes:         axes2DFwd,
			strides:      []int{1, 1},
			paddings:     [][2]int{{1, 1}, {1, 1}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:         "2D_Fwd_MNIST_Conv2",
			inputDims:    []int{4, 14, 14, 32},
			kernelDims:   []int{3, 3, 32, 64},
			axes:         axes2DFwd,
			strides:      []int{1, 1},
			paddings:     [][2]int{{1, 1}, {1, 1}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:            "2D_Fwd_1x1_Unit",
			inputDims:       []int{2, 9, 9, 19},
			kernelDims:      []int{1, 1, 19, 37},
			axes:            axes2DFwd,
			expectedKind:    ops.Conv2DStandard,
			expectIs1x1Unit: true,
		},
		{
			name:            "2D_Fwd_1x1_Unit_SmallM",
			inputDims:       []int{1, 2, 3, 11},
			kernelDims:      []int{1, 1, 11, 32},
			axes:            axes2DFwd,
			expectedKind:    ops.Conv2DStandard,
			expectIs1x1Unit: true,
		},
		{
			name:         "2D_Fwd_Small16x16",
			inputDims:    []int{1, 3, 4, 7},
			kernelDims:   []int{3, 3, 7, 13},
			axes:         axes2DFwd,
			paddings:     [][2]int{{1, 1}, {1, 1}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:            "2D_Fwd_StridedDilatedOddChannels",
			inputDims:       []int{2, 11, 13, 5},
			kernelDims:      []int{3, 3, 5, 19},
			axes:            axes2DFwd,
			strides:         []int{2, 2},
			paddings:        [][2]int{{1, 2}, {2, 1}},
			kernelDilations: []int{2, 2},
			expectedKind:    ops.Conv2DStandard,
		},
		{
			name:             "2D_VJPInput_MNIST_Conv2",
			inputDims:        []int{4, 14, 14, 64},
			kernelDims:       []int{3, 3, 32, 64},
			axes:             axes2DVJPInput,
			paddings:         [][2]int{{1, 1}, {1, 1}},
			expectedKind:     ops.Conv2DStandard,
			expectTransposed: true,
		},
		{
			name:             "2D_VJPInput_StridedInputDilation",
			inputDims:        []int{2, 6, 7, 19},
			kernelDims:       []int{3, 3, 7, 19},
			axes:             axes2DVJPInput,
			paddings:         [][2]int{{2, 1}, {1, 2}},
			inputDilations:   []int{2, 2},
			expectedKind:     ops.Conv2DStandard,
			expectTransposed: true,
		},
		{
			name:            "2D_VJPKernel_MNIST_Conv1_Cin1",
			inputDims:       []int{4, 28, 28, 1},
			kernelDims:      []int{4, 28, 28, 32},
			axes:            axes2DVJPKernel,
			strides:         []int{1, 1},
			paddings:        [][2]int{{1, 1}, {1, 1}},
			kernelDilations: []int{1, 1},
			expectedKind:    ops.Conv2DKernelGrad,
		},
		{
			name:            "2D_VJPKernel_MNIST_Conv2_Cin32_Cout64",
			inputDims:       []int{4, 14, 14, 32},
			kernelDims:      []int{4, 14, 14, 64},
			axes:            axes2DVJPKernel,
			strides:         []int{1, 1},
			paddings:        [][2]int{{1, 1}, {1, 1}},
			kernelDilations: []int{1, 1},
			expectedKind:    ops.Conv2DKernelGrad,
		},
		{
			name:            "2D_VJPKernel_OddChannels_Strided",
			inputDims:       []int{3, 15, 17, 11},
			kernelDims:      []int{3, 8, 9, 27},
			axes:            axes2DVJPKernel,
			strides:         []int{1, 1},
			paddings:        [][2]int{{1, 1}, {1, 1}},
			kernelDilations: []int{2, 2},
			expectedKind:    ops.Conv2DKernelGrad,
		},
		{
			name:            "2D_VJPKernel_LargeChannels35x50",
			inputDims:       []int{2, 8, 8, 35},
			kernelDims:      []int{2, 8, 8, 50},
			axes:            axes2DVJPKernel,
			strides:         []int{1, 1},
			paddings:        [][2]int{{1, 1}, {1, 1}},
			kernelDilations: []int{1, 1},
			expectedKind:    ops.Conv2DKernelGrad,
		},
		{
			name:         "2D_Fwd_Direct2D_LargeSpatial",
			inputDims:    []int{2, 32, 32, 16},
			kernelDims:   []int{3, 3, 16, 35},
			axes:         axes2DFwd,
			strides:      []int{1, 1},
			paddings:     [][2]int{{1, 1}, {1, 1}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:         "2D_Fwd_Direct2D_64x64",
			inputDims:    []int{2, 64, 64, 32},
			kernelDims:   []int{3, 3, 32, 32},
			axes:         axes2DFwd,
			strides:      []int{1, 1},
			paddings:     [][2]int{{1, 1}, {1, 1}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:         "2D_Fwd_Direct2D_8x8_Bottleneck",
			inputDims:    []int{4, 8, 8, 96},
			kernelDims:   []int{3, 3, 96, 128},
			axes:         axes2DFwd,
			strides:      []int{1, 1},
			paddings:     [][2]int{{1, 1}, {1, 1}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:         "2D_Fwd_Direct2D_OddWidths",
			inputDims:    []int{2, 11, 15, 16},
			kernelDims:   []int{3, 3, 16, 35},
			axes:         axes2DFwd,
			strides:      []int{1, 1},
			paddings:     [][2]int{{1, 1}, {1, 1}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:            "2D_VJPKernel_SmallBatch_LargeSpatial",
			inputDims:       []int{1, 64, 64, 16},
			kernelDims:      []int{1, 64, 64, 32},
			axes:            axes2DVJPKernel,
			strides:         []int{1, 1},
			paddings:        [][2]int{{1, 1}, {1, 1}},
			kernelDilations: []int{1, 1},
			expectedKind:    ops.Conv2DKernelGrad,
		},
		{
			name:         "1D_Fwd_Standard",
			inputDims:    []int{4, 33, 12},
			kernelDims:   []int{5, 12, 28},
			axes:         axes1DFwd,
			strides:      []int{2},
			paddings:     [][2]int{{2, 2}},
			expectedKind: ops.Conv2DStandard,
		},
		{
			name:             "1D_VJPInput",
			inputDims:        []int{3, 17, 25},
			kernelDims:       []int{3, 9, 25},
			axes:             axes1DVJPInput,
			paddings:         [][2]int{{1, 1}},
			inputDilations:   []int{2},
			expectedKind:     ops.Conv2DStandard,
			expectTransposed: true,
		},
		{
			name:            "1D_VJPKernel",
			inputDims:       []int{3, 25, 14},
			kernelDims:      []int{3, 25, 22},
			axes:            axes1DVJPKernel,
			strides:         []int{1},
			paddings:        [][2]int{{2, 2}},
			kernelDilations: []int{1},
			expectedKind:    ops.Conv2DKernelGrad,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder(tc.name)
			fn := builder.Main()
			inShape := shapes.Make(dtypes.Float32, tc.inputDims...)
			kerShape := shapes.Make(dtypes.Float32, tc.kernelDims...)
			inOp, err := fn.Parameter("x", inShape, nil)
			if err != nil {
				t.Fatalf("Parameter x failed: %+v", err)
			}
			kerOp, err := fn.Parameter("k", kerShape, nil)
			if err != nil {
				t.Fatalf("Parameter k failed: %+v", err)
			}
			outOp, err := fn.ConvGeneral(
				inOp, kerOp, tc.axes,
				tc.strides, tc.paddings,
				tc.inputDilations, tc.kernelDilations,
				1, 1,
			)
			if err != nil {
				t.Fatalf("ConvGeneral build failed: %+v", err)
			}
			convNode := outOp.(*gobackend.Node)
			p := ops.GetConv2DParams(convNode)
			if p.Kind != tc.expectedKind {
				t.Fatalf("expected Conv2DKind %v, got %v", tc.expectedKind, p.Kind)
			}
			if p.Is1x1Unit != tc.expectIs1x1Unit {
				t.Fatalf("expected Is1x1Unit=%v, got %v", tc.expectIs1x1Unit, p.Is1x1Unit)
			}
			if p.IsKernelTransposed != tc.expectTransposed {
				t.Fatalf("expected IsKernelTransposed=%v, got %v", tc.expectTransposed, p.IsKernelTransposed)
			}

			inBuf := makeDeterministicBuffer(t, backend, inShape, 1.0)
			defer backend.PutBuffer(inBuf)
			kerBuf := makeDeterministicBuffer(t, backend, kerShape, 2.0)
			defer backend.PutBuffer(kerBuf)

			inputs := []*gobackend.Buffer{inBuf, kerBuf}
			owned := []bool{false, false}

			wantBuf, err := ops.ExecConvGeneral(backend, convNode, inputs, owned)
			if err != nil {
				t.Fatalf("ExecConvGeneral failed: %+v", err)
			}
			defer backend.PutBuffer(wantBuf)

			avx512Buf, err := execConvGeneralAVX512(backend, convNode, inputs, owned)
			if err != nil {
				t.Fatalf("execConvGeneralAVX512 failed: %+v", err)
			}
			defer backend.PutBuffer(avx512Buf)
			compareConvBuffers(t, "AVX512", avx512Buf, wantBuf, 1e-4)
		})
	}
}

func TestConvMicrokernelsAVX512Direct(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this host")
	}

	for _, K := range []int{1, 2, 3, 4, 5, 8, 9, 17, 32} {
		t.Run(fmt.Sprintf("K%d", K), func(t *testing.T) {
			const rows = 8
			const cols = 32
			lhs := make([]float32, rows*K)
			rhs := make([]float32, K*cols)
			for i := range lhs {
				lhs[i] = float32((i%31)-15) * 0.1
			}
			for i := range rhs {
				rhs[i] = float32((i%29)-14) * 0.1
			}

			want := make([]float32, rows*cols)
			for r := range rows {
				for c := range cols {
					var sum float32
					for k := range K {
						sum += lhs[r*K+k] * rhs[k*cols+c]
					}
					want[r*cols+c] = sum
				}
			}

			got8x32 := make([]float32, 8*32)
			ConvAVX512Kernel8x32Float32Asm(
				unsafe.Pointer(&lhs[0]),
				unsafe.Pointer(&rhs[0]),
				unsafe.Pointer(&got8x32[0]),
				K, K*4, cols*4, 32*4,
			)
			for i := range got8x32 {
				diff := math.Abs(float64(got8x32[i] - want[i]))
				if diff > 1e-4 {
					t.Fatalf("AVX512 8x32 mismatch at %d: got %v, want %v", i, got8x32[i], want[i])
				}
			}

			got4x32 := make([]float32, 4*32)
			ConvAVX512Kernel4x32Float32Asm(
				unsafe.Pointer(&lhs[0]),
				unsafe.Pointer(&rhs[0]),
				unsafe.Pointer(&got4x32[0]),
				K, K*4, cols*4, 32*4,
			)
			for i := range got4x32 {
				diff := math.Abs(float64(got4x32[i] - want[i]))
				if diff > 1e-4 {
					t.Fatalf("AVX512 4x32 mismatch at %d: got %v, want %v", i, got4x32[i], want[i])
				}
			}
		})
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
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
}

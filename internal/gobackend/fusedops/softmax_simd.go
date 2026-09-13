// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package fusedops

import (
	"math"
	"simd"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/support/simdmath"
)

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeFusedSoftmax, PrioritySIMD, execFusedSoftmaxSIMD)
}

func execFusedSoftmaxSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedSoftmax)
	input := inputs[0]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}

	outerSize, axisSize, innerSize := fusedSoftmaxComputeAxisStrides(node.Shape, data.axis)
	if innerSize != 1 || axisSize < 2 {
		return nil, gobackend.ErrFallback
	}

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}

	inFlat := input.Flat.([]float32)
	outFlat := output.Flat.([]float32)

	totalElements := outerSize * axisSize
	processOuter := func(start, end int) {
		for outer := start; outer < end; outer++ {
			rowStart := outer * axisSize
			inRow := inFlat[rowStart : rowStart+axisSize]
			outRow := outFlat[rowStart : rowStart+axisSize]
			fusedSoftmaxRowFloat32(inRow, outRow)
		}
	}

	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && outerSize > 1 && totalElements > 16384 {
		numWorkers := backend.Workers.AdjustedMaxParallelism()
		targetChunks := min(outerSize, max(1, numWorkers*2))
		outerPerChunk := max(1, (outerSize+targetChunks-1)/targetChunks)
		var wg sync.WaitGroup
		for start := 0; start < outerSize; start += outerPerChunk {
			end := min(start+outerPerChunk, outerSize)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				processOuter(start, end)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		processOuter(0, outerSize)
	}

	return output, nil
}

func fusedSoftmaxRowFloat32(inRow, outRow []float32) {
	n := len(inRow)
	if n == 0 {
		return
	}
	if n == 1 {
		outRow[0] = 1.0
		return
	}

	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	negInf := float32(math.Inf(-1))

	var tmpBuf [64]float32

	// Fast path for rows that fit entirely in a single vector register.
	if n <= vLen {
		copy(tmpBuf[:n], inRow)
		for j := n; j < vLen; j++ {
			tmpBuf[j] = negInf
		}
		vIn := simd.LoadFloat32s(tmpBuf[:vLen])

		// Max reduction
		vIn.Store(tmpBuf[:vLen])
		maxVal := tmpBuf[0]
		for j := 1; j < n; j++ {
			if tmpBuf[j] > maxVal {
				maxVal = tmpBuf[j]
			}
		}

		// Vector exp
		vExp := simdmath.ExpFloat32(vIn.Sub(simd.BroadcastFloat32s(maxVal)))
		vExp.Store(tmpBuf[:vLen])

		// Sum reduction of active elements
		var sumVal float32
		for j := 0; j < n; j++ {
			sumVal += tmpBuf[j]
		}

		invSum := float32(1.0) / sumVal
		vNorm := vExp.Mul(simd.BroadcastFloat32s(invSum))
		vNorm.Store(tmpBuf[:vLen])
		copy(outRow, tmpBuf[:n])
		return
	}

	// Multi-vector path (n > vLen).
	// Pass 1: Find max using SIMD
	vMax := simd.LoadFloat32s(inRow)
	i := vLen
	for ; i+vLen <= n; i += vLen {
		vMax = vMax.Max(simd.LoadFloat32s(inRow[i:]))
	}
	vMax.Store(tmpBuf[:vLen])
	maxVal := tmpBuf[0]
	for j := 1; j < vLen; j++ {
		if tmpBuf[j] > maxVal {
			maxVal = tmpBuf[j]
		}
	}
	for ; i < n; i++ {
		if inRow[i] > maxVal {
			maxVal = inRow[i]
		}
	}

	// Pass 2: Vector exp and sum
	vMaxVal := simd.BroadcastFloat32s(maxVal)
	vSum := simd.BroadcastFloat32s(0)
	i = 0
	for ; i+vLen <= n; i += vLen {
		vX := simd.LoadFloat32s(inRow[i:]).Sub(vMaxVal)
		vExp := simdmath.ExpFloat32(vX)
		vExp.Store(outRow[i:])
		vSum = vSum.Add(vExp)
	}
	vSum.Store(tmpBuf[:vLen])
	var sumVal float32
	for j := 0; j < vLen; j++ {
		sumVal += tmpBuf[j]
	}
	if rem := n - i; rem > 0 {
		copy(tmpBuf[:rem], inRow[i:n])
		for j := rem; j < vLen; j++ {
			tmpBuf[j] = negInf
		}
		vRem := simd.LoadFloat32s(tmpBuf[:vLen])
		vExpRem := simdmath.ExpFloat32(vRem.Sub(vMaxVal))
		vExpRem.Store(tmpBuf[:vLen])
		for j := 0; j < rem; j++ {
			outRow[i+j] = tmpBuf[j]
			sumVal += tmpBuf[j]
		}
	}

	// Pass 3: Normalize
	invSum := float32(1.0) / sumVal
	vInvSum := simd.BroadcastFloat32s(invSum)
	i = 0
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(outRow[i:]).Mul(vInvSum).Store(outRow[i:])
	}
	for ; i < n; i++ {
		outRow[i] *= invSum
	}
}

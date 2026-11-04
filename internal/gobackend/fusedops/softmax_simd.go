// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package fusedops

import (
	"math"
	"simd"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/fastmath"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeFusedSoftmax, PrioritySIMD, execFusedSoftmaxSIMD)
}

func execFusedSoftmaxSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedSoftmax)
	input := inputs[0]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}

	outerSize, axisSize, innerSize := fusedSoftmaxComputeAxisStrides(node.Shape, data.axis)
	if innerSize != 1 || axisSize < 2 {
		return nil, gobackend.ErrFallback
	}

	var output *gobackend.Buffer
	var err error
	if len(inputsOwned) > 0 && inputsOwned[0] && input.RawShape.Equal(node.Shape) {
		output = input
		inputs[0] = nil
	} else {
		output, err = backend.GetBuffer(node.Shape)
		if err != nil {
			return nil, err
		}
	}
	if backend.NoOps {
		return output, nil
	}

	inFlat := input.Flat.([]float32)
	outFlat := output.Flat.([]float32)

	totalElements := outerSize * axisSize
	processOuter := func(start, end int) {
		fusedSoftmaxRowsFloat32(inFlat, outFlat, start, end, axisSize)
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && outerSize > 1 && totalElements >= 8192 {
		minOuter := max(1, 8192/max(1, axisSize))
		backend.Scheduler.ParallelizeChunks(outerSize, minOuter, processOuter)
	} else {
		processOuter(0, outerSize)
	}

	return output, nil
}

func fusedSoftmaxRowsFloat32(inFlat, outFlat []float32, startOuter, endOuter, axisSize int) {
	if axisSize == 0 || startOuter >= endOuter {
		return
	}
	if axisSize == 1 {
		for outer := startOuter; outer < endOuter; outer++ {
			outFlat[outer] = 1.0
		}
		return
	}
	if axisSize < 8 {
		for outer := startOuter; outer < endOuter; outer++ {
			rowStart := outer * axisSize
			inRow := inFlat[rowStart : rowStart+axisSize]
			outRow := outFlat[rowStart : rowStart+axisSize]
			maxVal := inRow[0]
			for j := 1; j < axisSize; j++ {
				if inRow[j] > maxVal {
					maxVal = inRow[j]
				}
			}
			var sumVal float32
			for j := range axisSize {
				e := fastmath.Exp32(inRow[j] - maxVal)
				outRow[j] = e
				sumVal += e
			}
			invSum := float32(1.0) / sumVal
			for j := range axisSize {
				outRow[j] *= invSum
			}
		}
		return
	}

	vMaxLog := simd.BroadcastFloat32s(88.0296919)
	vMinLog := simd.BroadcastFloat32s(-88.0296919)
	vLog2E := simd.BroadcastFloat32s(1.44269504088896341)
	vMagic := simd.BroadcastFloat32s(12582912.0)
	vLn2Hi := simd.BroadcastFloat32s(0.693359375)
	vLn2Lo := simd.BroadcastFloat32s(-2.12194440e-4)
	vP7 := simd.BroadcastFloat32s(1.9875691500e-4)
	vP6 := simd.BroadcastFloat32s(1.3981999507e-3)
	vP5 := simd.BroadcastFloat32s(8.3334519073e-3)
	vP4 := simd.BroadcastFloat32s(4.1665795894e-2)
	vP3 := simd.BroadcastFloat32s(1.6666665459e-1)
	vP2 := simd.BroadcastFloat32s(5.0000001201e-1)
	vOne := simd.BroadcastFloat32s(1.0)
	vZero := simd.BroadcastFloat32s(0.0)
	vShift23 := simd.BroadcastUint32s(1 << 23)
	vBias127 := simd.BroadcastUint32s(127 << 23)
	vLen := vZero.Len()
	negInf := float32(math.Inf(-1))

	var tmpBuf [64]float32

	for outer := startOuter; outer < endOuter; outer++ {
		rowStart := outer * axisSize
		inRow := inFlat[rowStart : rowStart+axisSize]
		outRow := outFlat[rowStart : rowStart+axisSize]
		n := axisSize

		// Fast path for rows that fit entirely in a single vector register (e.g. AVX-512 with 8 <= n <= 16).
		if n <= vLen {
			copy(tmpBuf[:n], inRow)
			for j := n; j < vLen; j++ {
				tmpBuf[j] = negInf
			}
			vIn := simd.LoadFloat32s(tmpBuf[:vLen])
			maxVal := inRow[0]
			for j := 1; j < n; j++ {
				if inRow[j] > maxVal {
					maxVal = inRow[j]
				}
			}

			vX := vIn.Sub(simd.BroadcastFloat32s(maxVal))
			xClamped := vMinLog.IfElse(vX.Less(vMinLog), vMaxLog.IfElse(vX.Greater(vMaxLog), vX))
			zRound := xClamped.MulAdd(vLog2E, vMagic)
			zFloor := zRound.Sub(vMagic)
			g := xClamped.Sub(zFloor.Mul(vLn2Hi)).Sub(zFloor.Mul(vLn2Lo))
			scale := zRound.ToBits().Mul(vShift23).Add(vBias127).BitsToFloat32()
			g2 := g.Mul(g)
			q67 := vP7.MulAdd(g, vP6)
			q45 := vP5.MulAdd(g, vP4)
			q23 := vP3.MulAdd(g, vP2)
			q01 := g.Add(vOne)
			g4 := g2.Mul(g2)
			q47 := q67.MulAdd(g2, q45)
			q03 := q23.MulAdd(g2, q01)
			vExp := q47.MulAdd(g4, q03).Mul(scale)
			vExp.Store(tmpBuf[:vLen])

			var sumVal float32
			for j := 0; j < n; j++ {
				sumVal += tmpBuf[j]
			}
			invSum := float32(1.0) / sumVal
			vNorm := vExp.Mul(simd.BroadcastFloat32s(invSum))
			vNorm.Store(tmpBuf[:vLen])
			copy(outRow, tmpBuf[:n])
			continue
		}

		// Multi-vector path (n > vLen).
		// Pass 1: Find max using 4-way unrolled SIMD reduction.
		var vMax simd.Float32s
		i := 0
		if n >= 4*vLen {
			vMax0 := simd.LoadFloat32s(inRow)
			vMax1 := simd.LoadFloat32s(inRow[vLen:])
			vMax2 := simd.LoadFloat32s(inRow[2*vLen:])
			vMax3 := simd.LoadFloat32s(inRow[3*vLen:])
			i = 4 * vLen
			for ; i+4*vLen <= n; i += 4 * vLen {
				vMax0 = vMax0.Max(simd.LoadFloat32s(inRow[i:]))
				vMax1 = vMax1.Max(simd.LoadFloat32s(inRow[i+vLen:]))
				vMax2 = vMax2.Max(simd.LoadFloat32s(inRow[i+2*vLen:]))
				vMax3 = vMax3.Max(simd.LoadFloat32s(inRow[i+3*vLen:]))
			}
			vMax = vMax0.Max(vMax1).Max(vMax2.Max(vMax3))
		} else {
			vMax = simd.LoadFloat32s(inRow)
			i = vLen
		}
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

		// Pass 2: Inlined vector exp and sum.
		vMaxVal := simd.BroadcastFloat32s(maxVal)
		vSum := vZero
		i = 0
		for ; i+vLen <= n; i += vLen {
			vX := simd.LoadFloat32s(inRow[i:]).Sub(vMaxVal)
			xClamped := vMinLog.IfElse(vX.Less(vMinLog), vMaxLog.IfElse(vX.Greater(vMaxLog), vX))
			zRound := xClamped.MulAdd(vLog2E, vMagic)
			zFloor := zRound.Sub(vMagic)
			g := xClamped.Sub(zFloor.Mul(vLn2Hi)).Sub(zFloor.Mul(vLn2Lo))
			scale := zRound.ToBits().Mul(vShift23).Add(vBias127).BitsToFloat32()
			g2 := g.Mul(g)
			q67 := vP7.MulAdd(g, vP6)
			q45 := vP5.MulAdd(g, vP4)
			q23 := vP3.MulAdd(g, vP2)
			q01 := g.Add(vOne)
			g4 := g2.Mul(g2)
			q47 := q67.MulAdd(g2, q45)
			q03 := q23.MulAdd(g2, q01)
			vExp := q47.MulAdd(g4, q03).Mul(scale)
			vExp.Store(outRow[i:])
			vSum = vSum.Add(vExp)
		}
		vSum.Store(tmpBuf[:vLen])
		var sumVal float32
		for j := 0; j < vLen; j++ {
			sumVal += tmpBuf[j]
		}
		for ; i < n; i++ {
			e := fastmath.Exp32(inRow[i] - maxVal)
			outRow[i] = e
			sumVal += e
		}

		// Pass 3: Normalize (4-way unrolled).
		invSum := float32(1.0) / sumVal
		vInvSum := simd.BroadcastFloat32s(invSum)
		i = 0
		for ; i+4*vLen <= n; i += 4 * vLen {
			v0 := simd.LoadFloat32s(outRow[i:]).Mul(vInvSum)
			v1 := simd.LoadFloat32s(outRow[i+vLen:]).Mul(vInvSum)
			v2 := simd.LoadFloat32s(outRow[i+2*vLen:]).Mul(vInvSum)
			v3 := simd.LoadFloat32s(outRow[i+3*vLen:]).Mul(vInvSum)
			v0.Store(outRow[i:])
			v1.Store(outRow[i+vLen:])
			v2.Store(outRow[i+2*vLen:])
			v3.Store(outRow[i+3*vLen:])
		}
		for ; i+vLen <= n; i += vLen {
			simd.LoadFloat32s(outRow[i:]).Mul(vInvSum).Store(outRow[i:])
		}
		for ; i < n; i++ {
			outRow[i] *= invSum
		}
	}
}

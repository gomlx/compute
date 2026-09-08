// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package fusedops

import (
	"math"
	"simd"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

const PrioritySIMD = gobackend.PriorityTyped + 5

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeFusedLayerNorm, PrioritySIMD, execFusedLayerNormSIMD)
}

func execFusedLayerNormSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedLayerNorm)
	input := inputs[0]
	dtype := input.RawShape.DType
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return nil, gobackend.ErrNotImplemented
	}

	dims := input.RawShape.Dimensions
	rank := len(dims)
	axes := data.axes

	// Check if normalization axes are contiguous trailing axes.
	normSize := 1
	for _, a := range axes {
		normSize *= dims[a]
	}
	isTrailingAxes := true
	for i, a := range axes {
		if a != rank-len(axes)+i {
			isTrailingAxes = false
			break
		}
	}
	if !isTrailingAxes || normSize <= 0 {
		return nil, gobackend.ErrNotImplemented
	}

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}

	var gamma, beta *gobackend.Buffer
	if len(inputs) > 1 {
		gamma = inputs[1]
	}
	if len(inputs) > 2 {
		beta = inputs[2]
	}

	switch dtype {
	case dtypes.Float32:
		var gammaData, betaData []float32
		if gamma != nil {
			gammaData = gamma.Flat.([]float32)
		}
		if beta != nil {
			betaData = beta.Flat.([]float32)
		}
		simdLayerNormTrailingAxesFloat32(input.Flat.([]float32), output.Flat.([]float32), gammaData, betaData, normSize, data.epsilon)
	case dtypes.Float64:
		var gammaData, betaData []float64
		if gamma != nil {
			gammaData = gamma.Flat.([]float64)
		}
		if beta != nil {
			betaData = beta.Flat.([]float64)
		}
		simdLayerNormTrailingAxesFloat64(input.Flat.([]float64), output.Flat.([]float64), gammaData, betaData, normSize, data.epsilon)
	default:
		return nil, gobackend.ErrNotImplemented
	}

	return output, nil
}

// -----------------------------------------------------------------------------
// Float32 SIMD LayerNorm
// -----------------------------------------------------------------------------

func simdLayerNormTrailingAxesFloat32(
	inData, outData, gammaData, betaData []float32,
	normSize int,
	epsilon float64,
) {
	normSizeF := float32(normSize)
	invNormSizeF := 1.0 / normSizeF
	outerSize := len(inData) / normSize

	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()

	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}

	// Fast path: normSize == vLen (e.g. 16 on AVX-512, 8 on AVX2)
	if normSize == vLen {
		hasGamma := len(gammaData) >= vLen
		hasBeta := len(betaData) >= vLen
		var vGamma, vBeta simd.Float32s
		if hasGamma {
			vGamma = simd.LoadFloat32s(gammaData)
		}
		if hasBeta {
			vBeta = simd.LoadFloat32s(betaData)
		}

		for outer := range outerSize {
			base := outer * normSize
			row := inData[base : base+normSize]
			outRow := outData[base : base+normSize]

			vIn := simd.LoadFloat32s(row)
			vIn.Store(tmp)
			var sum float32
			for _, v := range tmp {
				sum += v
			}
			mean := sum * invNormSizeF
			vMean := simd.BroadcastFloat32s(mean)

			vDiff := vIn.Sub(vMean)
			vDiffSq := vDiff.Mul(vDiff)
			vDiffSq.Store(tmp)
			var varSum float32
			for _, v := range tmp {
				varSum += v
			}
			variance := varSum * invNormSizeF
			invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))
			vInvStd := simd.BroadcastFloat32s(invStd)

			vNorm := vDiff.Mul(vInvStd)
			if hasGamma && hasBeta {
				vNorm = vNorm.MulAdd(vGamma, vBeta)
			} else if hasGamma {
				vNorm = vNorm.Mul(vGamma)
			} else if hasBeta {
				vNorm = vNorm.Add(vBeta)
			}
			vNorm.Store(outRow)
		}
		return
	}

	// Fast path: normSize == 2*vLen (e.g. 16 on AVX2)
	if normSize == 2*vLen {
		hasGamma := len(gammaData) >= 2*vLen
		hasBeta := len(betaData) >= 2*vLen
		var vGamma0, vGamma1, vBeta0, vBeta1 simd.Float32s
		if hasGamma {
			vGamma0 = simd.LoadFloat32s(gammaData)
			vGamma1 = simd.LoadFloat32s(gammaData[vLen:])
		}
		if hasBeta {
			vBeta0 = simd.LoadFloat32s(betaData)
			vBeta1 = simd.LoadFloat32s(betaData[vLen:])
		}

		for outer := range outerSize {
			base := outer * normSize
			row := inData[base : base+normSize]
			outRow := outData[base : base+normSize]

			vIn0 := simd.LoadFloat32s(row)
			vIn1 := simd.LoadFloat32s(row[vLen:])
			vIn0.Add(vIn1).Store(tmp)
			var sum float32
			for _, v := range tmp {
				sum += v
			}
			mean := sum * invNormSizeF
			vMean := simd.BroadcastFloat32s(mean)

			vDiff0 := vIn0.Sub(vMean)
			vDiff1 := vIn1.Sub(vMean)
			vDiffSq0 := vDiff0.Mul(vDiff0)
			vDiffSq1 := vDiff1.Mul(vDiff1)
			vDiffSq0.Add(vDiffSq1).Store(tmp)
			var varSum float32
			for _, v := range tmp {
				varSum += v
			}
			variance := varSum * invNormSizeF
			invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))
			vInvStd := simd.BroadcastFloat32s(invStd)

			vNorm0 := vDiff0.Mul(vInvStd)
			vNorm1 := vDiff1.Mul(vInvStd)
			if hasGamma && hasBeta {
				vNorm0 = vNorm0.MulAdd(vGamma0, vBeta0)
				vNorm1 = vNorm1.MulAdd(vGamma1, vBeta1)
			} else if hasGamma {
				vNorm0 = vNorm0.Mul(vGamma0)
				vNorm1 = vNorm1.Mul(vGamma1)
			} else if hasBeta {
				vNorm0 = vNorm0.Add(vBeta0)
				vNorm1 = vNorm1.Add(vBeta1)
			}
			vNorm0.Store(outRow)
			vNorm1.Store(outRow[vLen:])
		}
		return
	}

	// General trailing axes loop
	for outer := range outerSize {
		base := outer * normSize
		row := inData[base : base+normSize]
		outRow := outData[base : base+normSize]

		// 1. Mean
		var mean float32
		if normSize < vLen {
			var sum float32
			for _, v := range row {
				sum += v
			}
			mean = sum * invNormSizeF
		} else {
			vAcc := simd.LoadFloat32s(row)
			i := vLen
			for ; i+vLen <= normSize; i += vLen {
				vAcc = vAcc.Add(simd.LoadFloat32s(row[i:]))
			}
			vAcc.Store(tmp)
			var sum float32
			for _, v := range tmp {
				sum += v
			}
			for ; i < normSize; i++ {
				sum += row[i]
			}
			mean = sum * invNormSizeF
		}
		vMean := simd.BroadcastFloat32s(mean)

		// 2. Variance
		var variance float32
		if normSize < vLen {
			var varSum float32
			for _, v := range row {
				d := v - mean
				varSum += d * d
			}
			variance = varSum * invNormSizeF
		} else {
			vDiff0 := simd.LoadFloat32s(row).Sub(vMean)
			vVarAcc := vDiff0.Mul(vDiff0)
			i := vLen
			for ; i+vLen <= normSize; i += vLen {
				vD := simd.LoadFloat32s(row[i:]).Sub(vMean)
				vVarAcc = vVarAcc.Add(vD.Mul(vD))
			}
			vVarAcc.Store(tmp)
			var varSum float32
			for _, v := range tmp {
				varSum += v
			}
			for ; i < normSize; i++ {
				d := row[i] - mean
				varSum += d * d
			}
			variance = varSum * invNormSizeF
		}
		invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))
		vInvStd := simd.BroadcastFloat32s(invStd)

		// 3. Normalize + Affine (gamma, beta)
		i := 0
		for ; i+vLen <= normSize; i += vLen {
			vIn := simd.LoadFloat32s(row[i:])
			vNorm := vIn.Sub(vMean).Mul(vInvStd)
			if gammaData != nil && betaData != nil {
				vNorm = vNorm.MulAdd(simd.LoadFloat32s(gammaData[i:]), simd.LoadFloat32s(betaData[i:]))
			} else if gammaData != nil {
				vNorm = vNorm.Mul(simd.LoadFloat32s(gammaData[i:]))
			} else if betaData != nil {
				vNorm = vNorm.Add(simd.LoadFloat32s(betaData[i:]))
			}
			vNorm.Store(outRow[i:])
		}
		if i < normSize {
			vIn, _ := simd.LoadFloat32sPart(row[i:])
			vNorm := vIn.Sub(vMean).Mul(vInvStd)
			if gammaData != nil && betaData != nil {
				vG, _ := simd.LoadFloat32sPart(gammaData[i:])
				vB, _ := simd.LoadFloat32sPart(betaData[i:])
				vNorm = vNorm.MulAdd(vG, vB)
			} else if gammaData != nil {
				vG, _ := simd.LoadFloat32sPart(gammaData[i:])
				vNorm = vNorm.Mul(vG)
			} else if betaData != nil {
				vB, _ := simd.LoadFloat32sPart(betaData[i:])
				vNorm = vNorm.Add(vB)
			}
			vNorm.StorePart(outRow[i:])
		}
	}
}

// -----------------------------------------------------------------------------
// Float64 SIMD LayerNorm
// -----------------------------------------------------------------------------

func simdLayerNormTrailingAxesFloat64(
	inData, outData, gammaData, betaData []float64,
	normSize int,
	epsilon float64,
) {
	normSizeF := float64(normSize)
	invNormSizeF := 1.0 / normSizeF
	outerSize := len(inData) / normSize

	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()

	var tmpBuf [32]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}

	if normSize == vLen {
		hasGamma := len(gammaData) >= vLen
		hasBeta := len(betaData) >= vLen
		var vGamma, vBeta simd.Float64s
		if hasGamma {
			vGamma = simd.LoadFloat64s(gammaData)
		}
		if hasBeta {
			vBeta = simd.LoadFloat64s(betaData)
		}

		for outer := range outerSize {
			base := outer * normSize
			row := inData[base : base+normSize]
			outRow := outData[base : base+normSize]

			vIn := simd.LoadFloat64s(row)
			vIn.Store(tmp)
			var sum float64
			for _, v := range tmp {
				sum += v
			}
			mean := sum * invNormSizeF
			vMean := simd.BroadcastFloat64s(mean)

			vDiff := vIn.Sub(vMean)
			vDiffSq := vDiff.Mul(vDiff)
			vDiffSq.Store(tmp)
			var varSum float64
			for _, v := range tmp {
				varSum += v
			}
			variance := varSum * invNormSizeF
			invStd := 1.0 / math.Sqrt(variance+epsilon)
			vInvStd := simd.BroadcastFloat64s(invStd)

			vNorm := vDiff.Mul(vInvStd)
			if hasGamma && hasBeta {
				vNorm = vNorm.MulAdd(vGamma, vBeta)
			} else if hasGamma {
				vNorm = vNorm.Mul(vGamma)
			} else if hasBeta {
				vNorm = vNorm.Add(vBeta)
			}
			vNorm.Store(outRow)
		}
		return
	}

	for outer := range outerSize {
		base := outer * normSize
		row := inData[base : base+normSize]
		outRow := outData[base : base+normSize]

		var mean float64
		if normSize < vLen {
			var sum float64
			for _, v := range row {
				sum += v
			}
			mean = sum * invNormSizeF
		} else {
			vAcc := simd.LoadFloat64s(row)
			i := vLen
			for ; i+vLen <= normSize; i += vLen {
				vAcc = vAcc.Add(simd.LoadFloat64s(row[i:]))
			}
			vAcc.Store(tmp)
			var sum float64
			for _, v := range tmp {
				sum += v
			}
			for ; i < normSize; i++ {
				sum += row[i]
			}
			mean = sum * invNormSizeF
		}
		vMean := simd.BroadcastFloat64s(mean)

		var variance float64
		if normSize < vLen {
			var varSum float64
			for _, v := range row {
				d := v - mean
				varSum += d * d
			}
			variance = varSum * invNormSizeF
		} else {
			vDiff0 := simd.LoadFloat64s(row).Sub(vMean)
			vVarAcc := vDiff0.Mul(vDiff0)
			i := vLen
			for ; i+vLen <= normSize; i += vLen {
				vD := simd.LoadFloat64s(row[i:]).Sub(vMean)
				vVarAcc = vVarAcc.Add(vD.Mul(vD))
			}
			vVarAcc.Store(tmp)
			var varSum float64
			for _, v := range tmp {
				varSum += v
			}
			for ; i < normSize; i++ {
				d := row[i] - mean
				varSum += d * d
			}
			variance = varSum * invNormSizeF
		}
		invStd := 1.0 / math.Sqrt(variance+epsilon)
		vInvStd := simd.BroadcastFloat64s(invStd)

		i := 0
		for ; i+vLen <= normSize; i += vLen {
			vIn := simd.LoadFloat64s(row[i:])
			vNorm := vIn.Sub(vMean).Mul(vInvStd)
			if gammaData != nil && betaData != nil {
				vNorm = vNorm.MulAdd(simd.LoadFloat64s(gammaData[i:]), simd.LoadFloat64s(betaData[i:]))
			} else if gammaData != nil {
				vNorm = vNorm.Mul(simd.LoadFloat64s(gammaData[i:]))
			} else if betaData != nil {
				vNorm = vNorm.Add(simd.LoadFloat64s(betaData[i:]))
			}
			vNorm.Store(outRow[i:])
		}
		if i < normSize {
			vIn, _ := simd.LoadFloat64sPart(row[i:])
			vNorm := vIn.Sub(vMean).Mul(vInvStd)
			if gammaData != nil && betaData != nil {
				vG, _ := simd.LoadFloat64sPart(gammaData[i:])
				vB, _ := simd.LoadFloat64sPart(betaData[i:])
				vNorm = vNorm.MulAdd(vG, vB)
			} else if gammaData != nil {
				vG, _ := simd.LoadFloat64sPart(gammaData[i:])
				vNorm = vNorm.Mul(vG)
			} else if betaData != nil {
				vB, _ := simd.LoadFloat64sPart(betaData[i:])
				vNorm = vNorm.Add(vB)
			}
			vNorm.StorePart(outRow[i:])
		}
	}
}

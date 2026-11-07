// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package fusedops

import (
	"math"
	"simd"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/support/simdmath"
)

const PrioritySIMD = gobackend.PriorityTyped + 5

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeFusedLayerNorm, PrioritySIMD, execFusedLayerNormSIMD)
}

func execFusedLayerNormSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedLayerNorm)
	input := inputs[0]
	dtype := input.RawShape.DType
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return nil, gobackend.ErrFallback
	}

	dims := input.RawShape.Dimensions
	rank := len(dims)
	axes := data.axes

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

	var aDim, bDim, cDim int
	isMiddleContiguous := false
	if !isTrailingAxes {
		if a, b, c, d, ok := classifyStridedLayerNormAxes(dims, axes); ok && d == 1 && c > 1 && dtype == dtypes.Float32 {
			aDim, bDim, cDim = a, b, c
			isMiddleContiguous = true
		}
	}

	if !node.IsExecutorCached() {
		if (!isTrailingAxes && !isMiddleContiguous) || normSize <= 0 {
			return nil, gobackend.ErrFallback
		}
	}

	paramInputs := inputs
	var residual *gobackend.Buffer
	resInputIdx := -1
	if data.hasResidual && len(inputs) > 1 {
		resInputIdx = len(inputs) - 1
		residual = inputs[resInputIdx]
		paramInputs = inputs[:resInputIdx]
	}

	var output *gobackend.Buffer
	var err error
	if len(inputsOwned) > 0 && inputsOwned[0] && input.RawShape.Equal(node.Shape) {
		output = input
		inputs[0] = nil
	} else if resInputIdx >= 0 && len(inputsOwned) > resInputIdx && inputsOwned[resInputIdx] && residual.RawShape.Equal(node.Shape) {
		output = residual
		inputs[resInputIdx] = nil
	} else {
		output, err = backend.GetBuffer(node.Shape)
		if err != nil {
			return nil, err
		}
	}
	if backend.NoOps {
		return output, nil
	}

	var gamma, beta *gobackend.Buffer
	if len(paramInputs) > 1 {
		gamma = paramInputs[1]
	}
	if len(paramInputs) > 2 {
		beta = paramInputs[2]
	}

	if isMiddleContiguous {
		inData := input.Flat.([]float32)
		outData := output.Flat.([]float32)
		var resData []float32
		if residual != nil {
			resData = residual.Flat.([]float32)
		}
		var gData, bData []float32
		if gamma != nil {
			gData = gamma.Flat.([]float32)
		}
		if beta != nil {
			bData = beta.Flat.([]float32)
		}
		totalElements := len(inData)
		sliceSize := bDim * cDim
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && aDim > 1 && totalElements > minLayerNormParallelElements {
			minA := max(1, minLayerNormParallelElements/max(1, sliceSize))
			backend.Scheduler.ParallelizeChunks(aDim, minA, func(aStart, aEnd int) {
				inSlice := inData[aStart*sliceSize : aEnd*sliceSize]
				outSlice := outData[aStart*sliceSize : aEnd*sliceSize]
				if resData != nil {
					resSlice := resData[aStart*sliceSize : aEnd*sliceSize]
					for i := range outSlice {
						outSlice[i] = inSlice[i] + resSlice[i]
					}
					inSlice = outSlice
				}
				simdLayerNormMiddleAxesFloat32(inSlice, outSlice, gData, bData, aEnd-aStart, bDim, cDim, data.epsilon)
			})
		} else {
			srcSlice := inData
			if resData != nil {
				for i := range outData {
					outData[i] = inData[i] + resData[i]
				}
				srcSlice = outData
			}
			simdLayerNormMiddleAxesFloat32(srcSlice, outData, gData, bData, aDim, bDim, cDim, data.epsilon)
		}
		return output, nil
	}

	outerSize := input.RawShape.Size() / normSize
	archFn := gobackend.GetLayerNormTrailingArchDispatcher()
	resArchFn := gobackend.GetLayerNormResidualTrailingArchDispatcher()
	var gPtr, bPtr unsafe.Pointer
	if gamma != nil {
		gPtr = gamma.UnsafePointer()
	}
	if beta != nil {
		bPtr = beta.UnsafePointer()
	}

	totalElements := input.RawShape.Size()
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && outerSize > 1 && totalElements > 16384 {
		inPtr := input.UnsafePointer()
		outPtr := output.UnsafePointer()
		var resPtr unsafe.Pointer
		if residual != nil {
			resPtr = residual.UnsafePointer()
		}
		elemSize := dtype.Size()

		minOuter := max(1, 16384/max(1, normSize))
		backend.Scheduler.ParallelizeChunks(outerSize, minOuter, func(rStart, rEnd int) {
			chunkOuter := rEnd - rStart
			rowOffset := rStart * normSize
			chunkIn := unsafe.Pointer(uintptr(inPtr) + uintptr(rowOffset*elemSize))
			chunkOut := unsafe.Pointer(uintptr(outPtr) + uintptr(rowOffset*elemSize))
			if resPtr != nil {
				chunkRes := unsafe.Pointer(uintptr(resPtr) + uintptr(rowOffset*elemSize))
				if resArchFn != nil && resArchFn(chunkIn, chunkRes, chunkOut, gPtr, bPtr, chunkOuter, normSize, data.epsilon, dtype) {
					return
				}
			} else if archFn != nil {
				if archFn(chunkIn, chunkOut, gPtr, bPtr, chunkOuter, normSize, data.epsilon, dtype) {
					return
				}
			}
			// Fallback path
			switch dtype {
			case dtypes.Float32:
				var gData, bData []float32
				if gamma != nil {
					gData = gamma.Flat.([]float32)
				}
				if beta != nil {
					bData = beta.Flat.([]float32)
				}
				inSlice := input.Flat.([]float32)[rowOffset : rowOffset+chunkOuter*normSize]
				outSlice := output.Flat.([]float32)[rowOffset : rowOffset+chunkOuter*normSize]
				if residual != nil {
					resSlice := residual.Flat.([]float32)[rowOffset : rowOffset+chunkOuter*normSize]
					for i := range outSlice {
						outSlice[i] = inSlice[i] + resSlice[i]
					}
					inSlice = outSlice
				}
				simdLayerNormTrailingAxesFloat32(inSlice, outSlice, gData, bData, normSize, data.epsilon)
			case dtypes.Float64:
				var gData, bData []float64
				if gamma != nil {
					gData = gamma.Flat.([]float64)
				}
				if beta != nil {
					bData = beta.Flat.([]float64)
				}
				inSlice := input.Flat.([]float64)[rowOffset : rowOffset+chunkOuter*normSize]
				outSlice := output.Flat.([]float64)[rowOffset : rowOffset+chunkOuter*normSize]
				if residual != nil {
					resSlice := residual.Flat.([]float64)[rowOffset : rowOffset+chunkOuter*normSize]
					for i := range outSlice {
						outSlice[i] = inSlice[i] + resSlice[i]
					}
					inSlice = outSlice
				}
				simdLayerNormTrailingAxesFloat64(inSlice, outSlice, gData, bData, normSize, data.epsilon)
			}
		})
		return output, nil
	}

	if residual != nil {
		if resArchFn != nil && resArchFn(input.UnsafePointer(), residual.UnsafePointer(), output.UnsafePointer(), gPtr, bPtr, outerSize, normSize, data.epsilon, dtype) {
			return output, nil
		}
	} else if archFn != nil {
		if archFn(input.UnsafePointer(), output.UnsafePointer(), gPtr, bPtr, outerSize, normSize, data.epsilon, dtype) {
			return output, nil
		}
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
		inSlice := input.Flat.([]float32)
		outSlice := output.Flat.([]float32)
		if residual != nil {
			resSlice := residual.Flat.([]float32)
			for i := range outSlice {
				outSlice[i] = inSlice[i] + resSlice[i]
			}
			inSlice = outSlice
		}
		simdLayerNormTrailingAxesFloat32(inSlice, outSlice, gammaData, betaData, normSize, data.epsilon)
	case dtypes.Float64:
		var gammaData, betaData []float64
		if gamma != nil {
			gammaData = gamma.Flat.([]float64)
		}
		if beta != nil {
			betaData = beta.Flat.([]float64)
		}
		inSlice := input.Flat.([]float64)
		outSlice := output.Flat.([]float64)
		if residual != nil {
			resSlice := residual.Flat.([]float64)
			for i := range outSlice {
				outSlice[i] = inSlice[i] + resSlice[i]
			}
			inSlice = outSlice
		}
		simdLayerNormTrailingAxesFloat64(inSlice, outSlice, gammaData, betaData, normSize, data.epsilon)
	default:
		return nil, gobackend.ErrFallback
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
	if len(inData) == 0 || normSize == 0 {
		return
	}
	outerSize := len(inData) / normSize
	if archFn := gobackend.GetLayerNormTrailingArchDispatcher(); archFn != nil {
		var gPtr, bPtr unsafe.Pointer
		if len(gammaData) > 0 {
			gPtr = unsafe.Pointer(&gammaData[0])
		}
		if len(betaData) > 0 {
			bPtr = unsafe.Pointer(&betaData[0])
		}
		if archFn(
			unsafe.Pointer(&inData[0]),
			unsafe.Pointer(&outData[0]),
			gPtr,
			bPtr,
			outerSize, normSize, epsilon, dtypes.Float32,
		) {
			return
		}
	}

	normSizeF := float32(normSize)
	invNormSizeF := 1.0 / normSizeF

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
	hasGamma := len(gammaData) >= normSize
	hasBeta := len(betaData) >= normSize
	for outer := range outerSize {
		base := outer * normSize
		row := inData[base : base+normSize]
		outRow := outData[base : base+normSize]

		// 1. Mean (4-way unrolled)
		var mean float32
		if normSize < vLen {
			var sum float32
			for _, v := range row {
				sum += v
			}
			mean = sum * invNormSizeF
		} else {
			var vAcc simd.Float32s
			i := 0
			if normSize >= 4*vLen {
				vAcc0 := simd.LoadFloat32s(row)
				vAcc1 := simd.LoadFloat32s(row[vLen:])
				vAcc2 := simd.LoadFloat32s(row[2*vLen:])
				vAcc3 := simd.LoadFloat32s(row[3*vLen:])
				i = 4 * vLen
				for ; i+4*vLen <= normSize; i += 4 * vLen {
					vAcc0 = vAcc0.Add(simd.LoadFloat32s(row[i:]))
					vAcc1 = vAcc1.Add(simd.LoadFloat32s(row[i+vLen:]))
					vAcc2 = vAcc2.Add(simd.LoadFloat32s(row[i+2*vLen:]))
					vAcc3 = vAcc3.Add(simd.LoadFloat32s(row[i+3*vLen:]))
				}
				vAcc = vAcc0.Add(vAcc1).Add(vAcc2.Add(vAcc3))
			} else {
				vAcc = simd.LoadFloat32s(row)
				i = vLen
			}
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

		// 2. Variance (4-way unrolled with FMA)
		var variance float32
		if normSize < vLen {
			var varSum float32
			for _, v := range row {
				d := v - mean
				varSum += d * d
			}
			variance = varSum * invNormSizeF
		} else {
			var vVarAcc simd.Float32s
			i := 0
			if normSize >= 4*vLen {
				vD0 := simd.LoadFloat32s(row).Sub(vMean)
				vD1 := simd.LoadFloat32s(row[vLen:]).Sub(vMean)
				vD2 := simd.LoadFloat32s(row[2*vLen:]).Sub(vMean)
				vD3 := simd.LoadFloat32s(row[3*vLen:]).Sub(vMean)
				vVar0 := vD0.Mul(vD0)
				vVar1 := vD1.Mul(vD1)
				vVar2 := vD2.Mul(vD2)
				vVar3 := vD3.Mul(vD3)
				i = 4 * vLen
				for ; i+4*vLen <= normSize; i += 4 * vLen {
					d0 := simd.LoadFloat32s(row[i:]).Sub(vMean)
					d1 := simd.LoadFloat32s(row[i+vLen:]).Sub(vMean)
					d2 := simd.LoadFloat32s(row[i+2*vLen:]).Sub(vMean)
					d3 := simd.LoadFloat32s(row[i+3*vLen:]).Sub(vMean)
					vVar0 = d0.MulAdd(d0, vVar0)
					vVar1 = d1.MulAdd(d1, vVar1)
					vVar2 = d2.MulAdd(d2, vVar2)
					vVar3 = d3.MulAdd(d3, vVar3)
				}
				vVarAcc = vVar0.Add(vVar1).Add(vVar2.Add(vVar3))
			} else {
				vDiff0 := simd.LoadFloat32s(row).Sub(vMean)
				vVarAcc = vDiff0.Mul(vDiff0)
				i = vLen
			}
			for ; i+vLen <= normSize; i += vLen {
				vD := simd.LoadFloat32s(row[i:]).Sub(vMean)
				vVarAcc = vD.MulAdd(vD, vVarAcc)
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
		if hasGamma && hasBeta {
			for ; i+4*vLen <= normSize; i += 4 * vLen {
				v0 := simd.LoadFloat32s(row[i:]).Sub(vMean).Mul(vInvStd).MulAdd(simd.LoadFloat32s(gammaData[i:]), simd.LoadFloat32s(betaData[i:]))
				v1 := simd.LoadFloat32s(row[i+vLen:]).Sub(vMean).Mul(vInvStd).MulAdd(simd.LoadFloat32s(gammaData[i+vLen:]), simd.LoadFloat32s(betaData[i+vLen:]))
				v2 := simd.LoadFloat32s(row[i+2*vLen:]).Sub(vMean).Mul(vInvStd).MulAdd(simd.LoadFloat32s(gammaData[i+2*vLen:]), simd.LoadFloat32s(betaData[i+2*vLen:]))
				v3 := simd.LoadFloat32s(row[i+3*vLen:]).Sub(vMean).Mul(vInvStd).MulAdd(simd.LoadFloat32s(gammaData[i+3*vLen:]), simd.LoadFloat32s(betaData[i+3*vLen:]))
				v0.Store(outRow[i:])
				v1.Store(outRow[i+vLen:])
				v2.Store(outRow[i+2*vLen:])
				v3.Store(outRow[i+3*vLen:])
			}
			for ; i+vLen <= normSize; i += vLen {
				vNorm := simd.LoadFloat32s(row[i:]).Sub(vMean).Mul(vInvStd).MulAdd(simd.LoadFloat32s(gammaData[i:]), simd.LoadFloat32s(betaData[i:]))
				vNorm.Store(outRow[i:])
			}
		} else if hasGamma {
			for ; i+vLen <= normSize; i += vLen {
				vNorm := simd.LoadFloat32s(row[i:]).Sub(vMean).Mul(vInvStd).Mul(simd.LoadFloat32s(gammaData[i:]))
				vNorm.Store(outRow[i:])
			}
		} else if hasBeta {
			for ; i+vLen <= normSize; i += vLen {
				vNorm := simd.LoadFloat32s(row[i:]).Sub(vMean).Mul(vInvStd).Add(simd.LoadFloat32s(betaData[i:]))
				vNorm.Store(outRow[i:])
			}
		} else {
			for ; i+4*vLen <= normSize; i += 4 * vLen {
				v0 := simd.LoadFloat32s(row[i:]).Sub(vMean).Mul(vInvStd)
				v1 := simd.LoadFloat32s(row[i+vLen:]).Sub(vMean).Mul(vInvStd)
				v2 := simd.LoadFloat32s(row[i+2*vLen:]).Sub(vMean).Mul(vInvStd)
				v3 := simd.LoadFloat32s(row[i+3*vLen:]).Sub(vMean).Mul(vInvStd)
				v0.Store(outRow[i:])
				v1.Store(outRow[i+vLen:])
				v2.Store(outRow[i+2*vLen:])
				v3.Store(outRow[i+3*vLen:])
			}
			for ; i+vLen <= normSize; i += vLen {
				vNorm := simd.LoadFloat32s(row[i:]).Sub(vMean).Mul(vInvStd)
				vNorm.Store(outRow[i:])
			}
		}
		if i < normSize {
			vIn, _ := simdmath.LoadFloat32sPart(row[i:])
			vNorm := vIn.Sub(vMean).Mul(vInvStd)
			if hasGamma && hasBeta {
				vG, _ := simdmath.LoadFloat32sPart(gammaData[i:])
				vB, _ := simdmath.LoadFloat32sPart(betaData[i:])
				vNorm = vNorm.MulAdd(vG, vB)
			} else if hasGamma {
				vG, _ := simdmath.LoadFloat32sPart(gammaData[i:])
				vNorm = vNorm.Mul(vG)
			} else if hasBeta {
				vB, _ := simdmath.LoadFloat32sPart(betaData[i:])
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
	if len(inData) == 0 || normSize == 0 {
		return
	}
	outerSize := len(inData) / normSize
	if archFn := gobackend.GetLayerNormTrailingArchDispatcher(); archFn != nil {
		var gPtr, bPtr unsafe.Pointer
		if len(gammaData) > 0 {
			gPtr = unsafe.Pointer(&gammaData[0])
		}
		if len(betaData) > 0 {
			bPtr = unsafe.Pointer(&betaData[0])
		}
		if archFn(
			unsafe.Pointer(&inData[0]),
			unsafe.Pointer(&outData[0]),
			gPtr,
			bPtr,
			outerSize, normSize, epsilon, dtypes.Float64,
		) {
			return
		}
	}

	normSizeF := float64(normSize)
	invNormSizeF := 1.0 / normSizeF

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
			vIn, _ := simdmath.LoadFloat64sPart(row[i:])
			vNorm := vIn.Sub(vMean).Mul(vInvStd)
			if gammaData != nil && betaData != nil {
				vG, _ := simdmath.LoadFloat64sPart(gammaData[i:])
				vB, _ := simdmath.LoadFloat64sPart(betaData[i:])
				vNorm = vNorm.MulAdd(vG, vB)
			} else if gammaData != nil {
				vG, _ := simdmath.LoadFloat64sPart(gammaData[i:])
				vNorm = vNorm.Mul(vG)
			} else if betaData != nil {
				vB, _ := simdmath.LoadFloat64sPart(betaData[i:])
				vNorm = vNorm.Add(vB)
			}
			vNorm.StorePart(outRow[i:])
		}
	}
}

// simdLayerNormMiddleAxesFloat32 executes LayerNorm on a collapsed [aDim, bDim, cDim] tensor
// where bDim is the normalized dimension (normSize = bDim) and aDim, cDim are kept dimensions.
// It vectorizes along the contiguous cDim dimension.
func simdLayerNormMiddleAxesFloat32(
	inData, outData, gammaData, betaData []float32,
	aDim, bDim, cDim int,
	epsilon float64,
) {
	if aDim == 0 || bDim == 0 || cDim == 0 {
		return
	}
	invNormSizeF := 1.0 / float32(bDim)
	strideBC := bDim * cDim
	vZero := simd.BroadcastFloat32s(0)
	vLen := vZero.Len()

	var stackMeans [256]float32
	var stackVars [256]float32
	var means, vars []float32
	if cDim <= len(stackMeans) {
		means = stackMeans[:cDim]
		vars = stackVars[:cDim]
	} else {
		means = make([]float32, cDim)
		vars = make([]float32, cDim)
	}

	hasGamma := len(gammaData) >= bDim
	hasBeta := len(betaData) >= bDim

	for a := range aDim {
		baseA := a * strideBC

		// Pass 1 & 2: Compute mean[c] and invStd[c] across b in 0..bDim-1.
		// Process c in register-pinned blocks of 4*vLen (64 floats), 2*vLen (32 floats), vLen (16 floats), and remainder.
		c := 0
		for ; c+4*vLen <= cDim; c += 4 * vLen {
			vSum0, vSum1, vSum2, vSum3 := vZero, vZero, vZero, vZero
			for b := range bDim {
				off := baseA + b*cDim + c
				vSum0 = vSum0.Add(simd.LoadFloat32s(inData[off:]))
				vSum1 = vSum1.Add(simd.LoadFloat32s(inData[off+vLen:]))
				vSum2 = vSum2.Add(simd.LoadFloat32s(inData[off+2*vLen:]))
				vSum3 = vSum3.Add(simd.LoadFloat32s(inData[off+3*vLen:]))
			}
			vInvN := simd.BroadcastFloat32s(invNormSizeF)
			vMean0 := vSum0.Mul(vInvN)
			vMean1 := vSum1.Mul(vInvN)
			vMean2 := vSum2.Mul(vInvN)
			vMean3 := vSum3.Mul(vInvN)
			vMean0.Store(means[c:])
			vMean1.Store(means[c+vLen:])
			vMean2.Store(means[c+2*vLen:])
			vMean3.Store(means[c+3*vLen:])

			vVar0, vVar1, vVar2, vVar3 := vZero, vZero, vZero, vZero
			for b := range bDim {
				off := baseA + b*cDim + c
				d0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0)
				d1 := simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1)
				d2 := simd.LoadFloat32s(inData[off+2*vLen:]).Sub(vMean2)
				d3 := simd.LoadFloat32s(inData[off+3*vLen:]).Sub(vMean3)
				vVar0 = d0.MulAdd(d0, vVar0)
				vVar1 = d1.MulAdd(d1, vVar1)
				vVar2 = d2.MulAdd(d2, vVar2)
				vVar3 = d3.MulAdd(d3, vVar3)
			}
			vVar0.Store(vars[c:])
			vVar1.Store(vars[c+vLen:])
			vVar2.Store(vars[c+2*vLen:])
			vVar3.Store(vars[c+3*vLen:])
		}
		for ; c+2*vLen <= cDim; c += 2 * vLen {
			vSum0, vSum1 := vZero, vZero
			for b := range bDim {
				off := baseA + b*cDim + c
				vSum0 = vSum0.Add(simd.LoadFloat32s(inData[off:]))
				vSum1 = vSum1.Add(simd.LoadFloat32s(inData[off+vLen:]))
			}
			vInvN := simd.BroadcastFloat32s(invNormSizeF)
			vMean0 := vSum0.Mul(vInvN)
			vMean1 := vSum1.Mul(vInvN)
			vMean0.Store(means[c:])
			vMean1.Store(means[c+vLen:])

			vVar0, vVar1 := vZero, vZero
			for b := range bDim {
				off := baseA + b*cDim + c
				d0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0)
				d1 := simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1)
				vVar0 = d0.MulAdd(d0, vVar0)
				vVar1 = d1.MulAdd(d1, vVar1)
			}
			vVar0.Store(vars[c:])
			vVar1.Store(vars[c+vLen:])
		}
		for ; c+vLen <= cDim; c += vLen {
			vSum0 := vZero
			for b := range bDim {
				off := baseA + b*cDim + c
				vSum0 = vSum0.Add(simd.LoadFloat32s(inData[off:]))
			}
			vInvN := simd.BroadcastFloat32s(invNormSizeF)
			vMean0 := vSum0.Mul(vInvN)
			vMean0.Store(means[c:])

			vVar0 := vZero
			for b := range bDim {
				off := baseA + b*cDim + c
				d0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0)
				vVar0 = d0.MulAdd(d0, vVar0)
			}
			vVar0.Store(vars[c:])
		}
		for ; c < cDim; c++ {
			var s float32
			for b := range bDim {
				s += inData[baseA+b*cDim+c]
			}
			m := s * invNormSizeF
			means[c] = m
			var vSum float32
			for b := range bDim {
				d := inData[baseA+b*cDim+c] - m
				vSum += d * d
			}
			vars[c] = vSum
		}

		for i := range cDim {
			vars[i] = float32(1.0 / math.Sqrt(float64(vars[i]*invNormSizeF)+epsilon))
		}

		// Pass 3: Normalize out[b, c] = (in[b, c] - mean[c]) * invStd[c] * gamma[b] + beta[b]
		c = 0
		for ; c+4*vLen <= cDim; c += 4 * vLen {
			vMean0 := simd.LoadFloat32s(means[c:])
			vMean1 := simd.LoadFloat32s(means[c+vLen:])
			vMean2 := simd.LoadFloat32s(means[c+2*vLen:])
			vMean3 := simd.LoadFloat32s(means[c+3*vLen:])
			vInvStd0 := simd.LoadFloat32s(vars[c:])
			vInvStd1 := simd.LoadFloat32s(vars[c+vLen:])
			vInvStd2 := simd.LoadFloat32s(vars[c+2*vLen:])
			vInvStd3 := simd.LoadFloat32s(vars[c+3*vLen:])

			switch {
			case hasGamma && hasBeta:
				for b := range bDim {
					off := baseA + b*cDim + c
					vG := simd.BroadcastFloat32s(gammaData[b])
					vB := simd.BroadcastFloat32s(betaData[b])
					vScale0 := vInvStd0.Mul(vG)
					vScale1 := vInvStd1.Mul(vG)
					vScale2 := vInvStd2.Mul(vG)
					vScale3 := vInvStd3.Mul(vG)
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).MulAdd(vScale0, vB).Store(outData[off:])
					simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1).MulAdd(vScale1, vB).Store(outData[off+vLen:])
					simd.LoadFloat32s(inData[off+2*vLen:]).Sub(vMean2).MulAdd(vScale2, vB).Store(outData[off+2*vLen:])
					simd.LoadFloat32s(inData[off+3*vLen:]).Sub(vMean3).MulAdd(vScale3, vB).Store(outData[off+3*vLen:])
				}
			case hasGamma:
				for b := range bDim {
					off := baseA + b*cDim + c
					vG := simd.BroadcastFloat32s(gammaData[b])
					vScale0 := vInvStd0.Mul(vG)
					vScale1 := vInvStd1.Mul(vG)
					vScale2 := vInvStd2.Mul(vG)
					vScale3 := vInvStd3.Mul(vG)
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).Mul(vScale0).Store(outData[off:])
					simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1).Mul(vScale1).Store(outData[off+vLen:])
					simd.LoadFloat32s(inData[off+2*vLen:]).Sub(vMean2).Mul(vScale2).Store(outData[off+2*vLen:])
					simd.LoadFloat32s(inData[off+3*vLen:]).Sub(vMean3).Mul(vScale3).Store(outData[off+3*vLen:])
				}
			case hasBeta:
				for b := range bDim {
					off := baseA + b*cDim + c
					vB := simd.BroadcastFloat32s(betaData[b])
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).MulAdd(vInvStd0, vB).Store(outData[off:])
					simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1).MulAdd(vInvStd1, vB).Store(outData[off+vLen:])
					simd.LoadFloat32s(inData[off+2*vLen:]).Sub(vMean2).MulAdd(vInvStd2, vB).Store(outData[off+2*vLen:])
					simd.LoadFloat32s(inData[off+3*vLen:]).Sub(vMean3).MulAdd(vInvStd3, vB).Store(outData[off+3*vLen:])
				}
			default:
				for b := range bDim {
					off := baseA + b*cDim + c
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).Mul(vInvStd0).Store(outData[off:])
					simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1).Mul(vInvStd1).Store(outData[off+vLen:])
					simd.LoadFloat32s(inData[off+2*vLen:]).Sub(vMean2).Mul(vInvStd2).Store(outData[off+2*vLen:])
					simd.LoadFloat32s(inData[off+3*vLen:]).Sub(vMean3).Mul(vInvStd3).Store(outData[off+3*vLen:])
				}
			}
		}
		for ; c+vLen <= cDim; c += vLen {
			vMean0 := simd.LoadFloat32s(means[c:])
			vInvStd0 := simd.LoadFloat32s(vars[c:])
			switch {
			case hasGamma && hasBeta:
				for b := range bDim {
					off := baseA + b*cDim + c
					vScale0 := vInvStd0.Mul(simd.BroadcastFloat32s(gammaData[b]))
					vB := simd.BroadcastFloat32s(betaData[b])
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).MulAdd(vScale0, vB).Store(outData[off:])
				}
			case hasGamma:
				for b := range bDim {
					off := baseA + b*cDim + c
					vScale0 := vInvStd0.Mul(simd.BroadcastFloat32s(gammaData[b]))
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).Mul(vScale0).Store(outData[off:])
				}
			case hasBeta:
				for b := range bDim {
					off := baseA + b*cDim + c
					vB := simd.BroadcastFloat32s(betaData[b])
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).MulAdd(vInvStd0, vB).Store(outData[off:])
				}
			default:
				for b := range bDim {
					off := baseA + b*cDim + c
					simd.LoadFloat32s(inData[off:]).Sub(vMean0).Mul(vInvStd0).Store(outData[off:])
				}
			}
		}
		for ; c < cDim; c++ {
			m := means[c]
			invS := vars[c]
			switch {
			case hasGamma && hasBeta:
				for b := range bDim {
					off := baseA + b*cDim + c
					outData[off] = (inData[off]-m)*invS*gammaData[b] + betaData[b]
				}
			case hasGamma:
				for b := range bDim {
					off := baseA + b*cDim + c
					outData[off] = (inData[off] - m) * invS * gammaData[b]
				}
			case hasBeta:
				for b := range bDim {
					off := baseA + b*cDim + c
					outData[off] = (inData[off]-m)*invS + betaData[b]
				}
			default:
				for b := range bDim {
					off := baseA + b*cDim + c
					outData[off] = (inData[off] - m) * invS
				}
			}
		}
	}
}

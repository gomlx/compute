// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package fusedops

import (
	"math"
	"simd"
	"sync"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/support/simdmath"
)

const PrioritySIMD = gobackend.PriorityTyped + 5

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeFusedLayerNorm, PrioritySIMD, execFusedLayerNormSIMD)
	layerNormVJPSIMDHook = execFusedLayerNormVJPSIMD
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
	inIdx := 1
	if data.hasGamma && len(paramInputs) > inIdx {
		gamma = paramInputs[inIdx]
		inIdx++
	}
	if data.hasBeta && len(paramInputs) > inIdx {
		beta = paramInputs[inIdx]
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

// -----------------------------------------------------------------------------
// SIMD FusedLayerNormVJP
// -----------------------------------------------------------------------------

func execFusedLayerNormVJPSIMD(
	backend *gobackend.Backend,
	data *nodeFusedLayerNormVJP,
	xBuf, dOutBuf, gammaBuf, dxBuf, dGammaBuf, dBetaBuf *gobackend.Buffer,
) bool {
	dtype := xBuf.RawShape.DType
	if dtype != dtypes.Float32 && dtype != dtypes.Float64 {
		return false
	}

	dims := xBuf.RawShape.Dimensions
	rank := len(dims)
	axes := data.axes

	normSize := 1
	for _, a := range axes {
		normSize *= dims[a]
	}
	totalElements := xBuf.RawShape.Size()
	if normSize <= 0 || totalElements == 0 {
		return false
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

	if !isTrailingAxes && !isMiddleContiguous {
		return false
	}

	if isMiddleContiguous {
		inData := xBuf.Flat.([]float32)
		dOutData := dOutBuf.Flat.([]float32)
		dxData := dxBuf.Flat.([]float32)
		var gData, dgData, dbData []float32
		if gammaBuf != nil {
			gData = gammaBuf.Flat.([]float32)
		}
		if dGammaBuf != nil {
			dgData = dGammaBuf.Flat.([]float32)
		}
		if dBetaBuf != nil {
			dbData = dBetaBuf.Flat.([]float32)
		}
		sliceSize := bDim * cDim
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && aDim > 1 && totalElements > minLayerNormParallelElements {
			minA := max(1, minLayerNormParallelElements/max(1, sliceSize))
			var mu sync.Mutex
			backend.Scheduler.ParallelizeChunks(aDim, minA, func(aStart, aEnd int) {
				off := aStart * sliceSize
				end := aEnd * sliceSize
				var stackDG, stackDB [4096]float32
				var localDG, localDB []float32
				if dgData != nil {
					if bDim <= len(stackDG) {
						localDG = stackDG[:bDim]
					} else {
						localDG = make([]float32, bDim)
					}
				}
				if dbData != nil {
					if bDim <= len(stackDB) {
						localDB = stackDB[:bDim]
					} else {
						localDB = make([]float32, bDim)
					}
				}
				simdLayerNormMiddleAxesVJPFloat32(
					inData[off:end],
					dOutData[off:end],
					gData,
					dxData[off:end],
					localDG,
					localDB,
					aEnd-aStart, bDim, cDim,
					data.epsilon,
				)
				if localDG != nil || localDB != nil {
					mu.Lock()
					addFloat32SlicesSIMD(dgData, localDG)
					addFloat32SlicesSIMD(dbData, localDB)
					mu.Unlock()
				}
			})
		} else {
			simdLayerNormMiddleAxesVJPFloat32(inData, dOutData, gData, dxData, dgData, dbData, aDim, bDim, cDim, data.epsilon)
		}
		return true
	}

	outerSize := totalElements / normSize
	switch dtype {
	case dtypes.Float32:
		inData := xBuf.Flat.([]float32)
		dOutData := dOutBuf.Flat.([]float32)
		dxData := dxBuf.Flat.([]float32)
		var gData, dgData, dbData []float32
		if gammaBuf != nil {
			gData = gammaBuf.Flat.([]float32)
		}
		if dGammaBuf != nil {
			dgData = dGammaBuf.Flat.([]float32)
		}
		if dBetaBuf != nil {
			dbData = dBetaBuf.Flat.([]float32)
		}
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && outerSize > 1 && totalElements > minLayerNormParallelElements {
			minOuter := max(1, minLayerNormParallelElements/max(1, normSize))
			var mu sync.Mutex
			backend.Scheduler.ParallelizeChunks(outerSize, minOuter, func(rStart, rEnd int) {
				rowOffset := rStart * normSize
				endOffset := rEnd * normSize
				var stackDG, stackDB [4096]float32
				var localDG, localDB []float32
				if dgData != nil {
					if normSize <= len(stackDG) {
						localDG = stackDG[:normSize]
					} else {
						localDG = make([]float32, normSize)
					}
				}
				if dbData != nil {
					if normSize <= len(stackDB) {
						localDB = stackDB[:normSize]
					} else {
						localDB = make([]float32, normSize)
					}
				}
				simdLayerNormTrailingAxesVJPFloat32(
					inData[rowOffset:endOffset],
					dOutData[rowOffset:endOffset],
					gData,
					dxData[rowOffset:endOffset],
					localDG,
					localDB,
					normSize,
					data.epsilon,
				)
				if localDG != nil || localDB != nil {
					mu.Lock()
					addFloat32SlicesSIMD(dgData, localDG)
					addFloat32SlicesSIMD(dbData, localDB)
					mu.Unlock()
				}
			})
		} else {
			simdLayerNormTrailingAxesVJPFloat32(inData, dOutData, gData, dxData, dgData, dbData, normSize, data.epsilon)
		}
		return true

	case dtypes.Float64:
		inData := xBuf.Flat.([]float64)
		dOutData := dOutBuf.Flat.([]float64)
		dxData := dxBuf.Flat.([]float64)
		var gData, dgData, dbData []float64
		if gammaBuf != nil {
			gData = gammaBuf.Flat.([]float64)
		}
		if dGammaBuf != nil {
			dgData = dGammaBuf.Flat.([]float64)
		}
		if dBetaBuf != nil {
			dbData = dBetaBuf.Flat.([]float64)
		}
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && outerSize > 1 && totalElements > minLayerNormParallelElements {
			minOuter := max(1, minLayerNormParallelElements/max(1, normSize))
			var mu sync.Mutex
			backend.Scheduler.ParallelizeChunks(outerSize, minOuter, func(rStart, rEnd int) {
				rowOffset := rStart * normSize
				endOffset := rEnd * normSize
				var stackDG, stackDB [2048]float64
				var localDG, localDB []float64
				if dgData != nil {
					if normSize <= len(stackDG) {
						localDG = stackDG[:normSize]
					} else {
						localDG = make([]float64, normSize)
					}
				}
				if dbData != nil {
					if normSize <= len(stackDB) {
						localDB = stackDB[:normSize]
					} else {
						localDB = make([]float64, normSize)
					}
				}
				simdLayerNormTrailingAxesVJPFloat64(
					inData[rowOffset:endOffset],
					dOutData[rowOffset:endOffset],
					gData,
					dxData[rowOffset:endOffset],
					localDG,
					localDB,
					normSize,
					data.epsilon,
				)
				if localDG != nil || localDB != nil {
					mu.Lock()
					for i, v := range localDG {
						dgData[i] += v
					}
					for i, v := range localDB {
						dbData[i] += v
					}
					mu.Unlock()
				}
			})
		} else {
			simdLayerNormTrailingAxesVJPFloat64(inData, dOutData, gData, dxData, dgData, dbData, normSize, data.epsilon)
		}
		return true
	}
	return false
}

func addFloat32SlicesSIMD(dst, src []float32) {
	if len(dst) == 0 || len(src) == 0 {
		return
	}
	n := min(len(dst), len(src))
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(dst[i:]).Add(simd.LoadFloat32s(src[i:])).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += src[i]
	}
}

func sumFloat32s(v simd.Float32s, tmp []float32) float32 {
	v.Store(tmp)
	switch len(tmp) {
	case 16:
		return ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7])) +
			((tmp[8] + tmp[9]) + (tmp[10] + tmp[11])) + ((tmp[12] + tmp[13]) + (tmp[14] + tmp[15]))
	case 8:
		return ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]))
	case 4:
		return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3])
	default:
		var s float32
		for _, x := range tmp {
			s += x
		}
		return s
	}
}

func sumFloat64s(v simd.Float64s, tmp []float64) float64 {
	v.Store(tmp)
	switch len(tmp) {
	case 8:
		return ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]))
	case 4:
		return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3])
	case 2:
		return tmp[0] + tmp[1]
	default:
		var s float64
		for _, x := range tmp {
			s += x
		}
		return s
	}
}

func simdLayerNormTrailingAxesVJPFloat32(
	inData, dOutData, gammaData, dxData, dGammaData, dBetaData []float32,
	normSize int,
	epsilon float64,
) {
	if len(inData) == 0 || normSize == 0 {
		return
	}
	outerSize := len(inData) / normSize
	invNormSizeF := 1.0 / float32(normSize)
	vZero := simd.BroadcastFloat32s(0)
	vLen := vZero.Len()

	var tmpBuf [64]float32
	tmp := tmpBuf[:vLen]

	hasGamma := len(gammaData) >= normSize
	hasDGamma := len(dGammaData) >= normSize
	hasDBeta := len(dBetaData) >= normSize

	for outer := range outerSize {
		base := outer * normSize
		inRow := inData[base : base+normSize]
		dyRow := dOutData[base : base+normSize]
		dxRow := dxData[base : base+normSize]

		// Pass 1: Mean
		var mean float32
		if normSize < vLen {
			var sum float32
			for _, v := range inRow {
				sum += v
			}
			mean = sum * invNormSizeF
		} else {
			vAcc := vZero
			i := 0
			for ; i+4*vLen <= normSize; i += 4 * vLen {
				v0 := simd.LoadFloat32s(inRow[i:]).Add(simd.LoadFloat32s(inRow[i+vLen:]))
				v1 := simd.LoadFloat32s(inRow[i+2*vLen:]).Add(simd.LoadFloat32s(inRow[i+3*vLen:]))
				vAcc = vAcc.Add(v0.Add(v1))
			}
			for ; i+vLen <= normSize; i += vLen {
				vAcc = vAcc.Add(simd.LoadFloat32s(inRow[i:]))
			}
			sum := sumFloat32s(vAcc, tmp)
			for ; i < normSize; i++ {
				sum += inRow[i]
			}
			mean = sum * invNormSizeF
		}
		vMean := simd.BroadcastFloat32s(mean)

		// Pass 2: Variance, sumG, sumGD
		vVar, vSumG, vSumGD := vZero, vZero, vZero
		i := 0
		if hasGamma {
			for ; i+vLen <= normSize; i += vLen {
				d := simd.LoadFloat32s(inRow[i:]).Sub(vMean)
				g := simd.LoadFloat32s(dyRow[i:]).Mul(simd.LoadFloat32s(gammaData[i:]))
				vVar = d.MulAdd(d, vVar)
				vSumG = vSumG.Add(g)
				vSumGD = g.MulAdd(d, vSumGD)
			}
		} else {
			for ; i+vLen <= normSize; i += vLen {
				d := simd.LoadFloat32s(inRow[i:]).Sub(vMean)
				g := simd.LoadFloat32s(dyRow[i:])
				vVar = d.MulAdd(d, vVar)
				vSumG = vSumG.Add(g)
				vSumGD = g.MulAdd(d, vSumGD)
			}
		}
		varSum := sumFloat32s(vVar, tmp)
		sumG := sumFloat32s(vSumG, tmp)
		sumGD := sumFloat32s(vSumGD, tmp)
		for ; i < normSize; i++ {
			d := inRow[i] - mean
			varSum += d * d
			g := dyRow[i]
			if hasGamma {
				g *= gammaData[i]
			}
			sumG += g
			sumGD += g * d
		}
		invStd := float32(1.0 / math.Sqrt(float64(varSum*invNormSizeF)+epsilon))
		meanG := sumG * invNormSizeF
		meanGHat := sumGD * invNormSizeF * invStd

		// Pass 3: dx, dGamma, dBeta
		vInvStd := simd.BroadcastFloat32s(invStd)
		vMeanG := simd.BroadcastFloat32s(meanG)
		vNegMeanGHat := simd.BroadcastFloat32s(-meanGHat)
		i = 0
		for ; i+vLen <= normSize; i += vLen {
			vIn := simd.LoadFloat32s(inRow[i:])
			vDy := simd.LoadFloat32s(dyRow[i:])
			vXHat := vIn.Sub(vMean).Mul(vInvStd)
			vG := vDy
			if hasGamma {
				vG = vDy.Mul(simd.LoadFloat32s(gammaData[i:]))
			}
			vDx := vXHat.MulAdd(vNegMeanGHat, vG.Sub(vMeanG)).Mul(vInvStd)
			vDx.Store(dxRow[i:])
			if hasDGamma {
				vDy.MulAdd(vXHat, simd.LoadFloat32s(dGammaData[i:])).Store(dGammaData[i:])
			}
			if hasDBeta {
				vDy.Add(simd.LoadFloat32s(dBetaData[i:])).Store(dBetaData[i:])
			}
		}
		for ; i < normSize; i++ {
			xHat := (inRow[i] - mean) * invStd
			dy := dyRow[i]
			g := dy
			if hasGamma {
				g *= gammaData[i]
			}
			dxRow[i] = invStd * (g - meanG - xHat*meanGHat)
			if hasDGamma {
				dGammaData[i] += dy * xHat
			}
			if hasDBeta {
				dBetaData[i] += dy
			}
		}
	}
}

func simdLayerNormTrailingAxesVJPFloat64(
	inData, dOutData, gammaData, dxData, dGammaData, dBetaData []float64,
	normSize int,
	epsilon float64,
) {
	if len(inData) == 0 || normSize == 0 {
		return
	}
	outerSize := len(inData) / normSize
	invNormSizeF := 1.0 / float64(normSize)
	vZero := simd.BroadcastFloat64s(0)
	vLen := vZero.Len()

	var tmpBuf [32]float64
	tmp := tmpBuf[:vLen]

	hasGamma := len(gammaData) >= normSize
	hasDGamma := len(dGammaData) >= normSize
	hasDBeta := len(dBetaData) >= normSize

	for outer := range outerSize {
		base := outer * normSize
		inRow := inData[base : base+normSize]
		dyRow := dOutData[base : base+normSize]
		dxRow := dxData[base : base+normSize]

		vAcc := vZero
		i := 0
		for ; i+vLen <= normSize; i += vLen {
			vAcc = vAcc.Add(simd.LoadFloat64s(inRow[i:]))
		}
		sum := sumFloat64s(vAcc, tmp)
		for ; i < normSize; i++ {
			sum += inRow[i]
		}
		mean := sum * invNormSizeF
		vMean := simd.BroadcastFloat64s(mean)

		vVar, vSumG, vSumGD := vZero, vZero, vZero
		i = 0
		if hasGamma {
			for ; i+vLen <= normSize; i += vLen {
				d := simd.LoadFloat64s(inRow[i:]).Sub(vMean)
				g := simd.LoadFloat64s(dyRow[i:]).Mul(simd.LoadFloat64s(gammaData[i:]))
				vVar = d.MulAdd(d, vVar)
				vSumG = vSumG.Add(g)
				vSumGD = g.MulAdd(d, vSumGD)
			}
		} else {
			for ; i+vLen <= normSize; i += vLen {
				d := simd.LoadFloat64s(inRow[i:]).Sub(vMean)
				g := simd.LoadFloat64s(dyRow[i:])
				vVar = d.MulAdd(d, vVar)
				vSumG = vSumG.Add(g)
				vSumGD = g.MulAdd(d, vSumGD)
			}
		}
		varSum := sumFloat64s(vVar, tmp)
		sumG := sumFloat64s(vSumG, tmp)
		sumGD := sumFloat64s(vSumGD, tmp)
		for ; i < normSize; i++ {
			d := inRow[i] - mean
			varSum += d * d
			g := dyRow[i]
			if hasGamma {
				g *= gammaData[i]
			}
			sumG += g
			sumGD += g * d
		}
		invStd := 1.0 / math.Sqrt(varSum*invNormSizeF+epsilon)
		meanG := sumG * invNormSizeF
		meanGHat := sumGD * invNormSizeF * invStd

		vInvStd := simd.BroadcastFloat64s(invStd)
		vMeanG := simd.BroadcastFloat64s(meanG)
		vNegMeanGHat := simd.BroadcastFloat64s(-meanGHat)
		i = 0
		for ; i+vLen <= normSize; i += vLen {
			vIn := simd.LoadFloat64s(inRow[i:])
			vDy := simd.LoadFloat64s(dyRow[i:])
			vXHat := vIn.Sub(vMean).Mul(vInvStd)
			vG := vDy
			if hasGamma {
				vG = vDy.Mul(simd.LoadFloat64s(gammaData[i:]))
			}
			vDx := vXHat.MulAdd(vNegMeanGHat, vG.Sub(vMeanG)).Mul(vInvStd)
			vDx.Store(dxRow[i:])
			if hasDGamma {
				vDy.MulAdd(vXHat, simd.LoadFloat64s(dGammaData[i:])).Store(dGammaData[i:])
			}
			if hasDBeta {
				vDy.Add(simd.LoadFloat64s(dBetaData[i:])).Store(dBetaData[i:])
			}
		}
		for ; i < normSize; i++ {
			xHat := (inRow[i] - mean) * invStd
			dy := dyRow[i]
			g := dy
			if hasGamma {
				g *= gammaData[i]
			}
			dxRow[i] = invStd * (g - meanG - xHat*meanGHat)
			if hasDGamma {
				dGammaData[i] += dy * xHat
			}
			if hasDBeta {
				dBetaData[i] += dy
			}
		}
	}
}

// simdLayerNormMiddleAxesVJPFloat32 executes LayerNorm backward pass on a collapsed [aDim, bDim, cDim]
// tensor where bDim is the normalized dimension (normSize = bDim) and aDim, cDim are kept dimensions.
// It processes cDim in register-pinned strips of 4*vLen so each [bDim, 4*vLen] slice remains cache-resident
// across all 3 passes.
func simdLayerNormMiddleAxesVJPFloat32(
	inData, dOutData, gammaData, dxData, dGammaData, dBetaData []float32,
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
	vInvN := simd.BroadcastFloat32s(invNormSizeF)

	var tmpBuf [64]float32
	tmp := tmpBuf[:vLen]
	var stripBuf [256]float32

	hasGamma := len(gammaData) >= bDim
	hasDGamma := len(dGammaData) >= bDim
	hasDBeta := len(dBetaData) >= bDim

	for a := range aDim {
		baseA := a * strideBC
		c := 0

		// 4*vLen strip loop (64 floats on AVX-512, 32 floats on AVX2)
		for ; c+4*vLen <= cDim; c += 4 * vLen {
			// Pass 1: Mean over b
			vSum0, vSum1, vSum2, vSum3 := vZero, vZero, vZero, vZero
			for b := range bDim {
				off := baseA + b*cDim + c
				vSum0 = vSum0.Add(simd.LoadFloat32s(inData[off:]))
				vSum1 = vSum1.Add(simd.LoadFloat32s(inData[off+vLen:]))
				vSum2 = vSum2.Add(simd.LoadFloat32s(inData[off+2*vLen:]))
				vSum3 = vSum3.Add(simd.LoadFloat32s(inData[off+3*vLen:]))
			}
			vMean0 := vSum0.Mul(vInvN)
			vMean1 := vSum1.Mul(vInvN)
			vMean2 := vSum2.Mul(vInvN)
			vMean3 := vSum3.Mul(vInvN)

			// Pass 2: Variance, sumG, sumGD over b
			vVar0, vVar1, vVar2, vVar3 := vZero, vZero, vZero, vZero
			vSumG0, vSumG1, vSumG2, vSumG3 := vZero, vZero, vZero, vZero
			vSumGD0, vSumGD1, vSumGD2, vSumGD3 := vZero, vZero, vZero, vZero
			if hasGamma {
				for b := range bDim {
					off := baseA + b*cDim + c
					vG := simd.BroadcastFloat32s(gammaData[b])
					d0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0)
					d1 := simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1)
					d2 := simd.LoadFloat32s(inData[off+2*vLen:]).Sub(vMean2)
					d3 := simd.LoadFloat32s(inData[off+3*vLen:]).Sub(vMean3)
					vVar0 = d0.MulAdd(d0, vVar0)
					vVar1 = d1.MulAdd(d1, vVar1)
					vVar2 = d2.MulAdd(d2, vVar2)
					vVar3 = d3.MulAdd(d3, vVar3)
					g0 := simd.LoadFloat32s(dOutData[off:]).Mul(vG)
					g1 := simd.LoadFloat32s(dOutData[off+vLen:]).Mul(vG)
					g2 := simd.LoadFloat32s(dOutData[off+2*vLen:]).Mul(vG)
					g3 := simd.LoadFloat32s(dOutData[off+3*vLen:]).Mul(vG)
					vSumG0 = vSumG0.Add(g0)
					vSumG1 = vSumG1.Add(g1)
					vSumG2 = vSumG2.Add(g2)
					vSumG3 = vSumG3.Add(g3)
					vSumGD0 = g0.MulAdd(d0, vSumGD0)
					vSumGD1 = g1.MulAdd(d1, vSumGD1)
					vSumGD2 = g2.MulAdd(d2, vSumGD2)
					vSumGD3 = g3.MulAdd(d3, vSumGD3)
				}
			} else {
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
					g0 := simd.LoadFloat32s(dOutData[off:])
					g1 := simd.LoadFloat32s(dOutData[off+vLen:])
					g2 := simd.LoadFloat32s(dOutData[off+2*vLen:])
					g3 := simd.LoadFloat32s(dOutData[off+3*vLen:])
					vSumG0 = vSumG0.Add(g0)
					vSumG1 = vSumG1.Add(g1)
					vSumG2 = vSumG2.Add(g2)
					vSumG3 = vSumG3.Add(g3)
					vSumGD0 = g0.MulAdd(d0, vSumGD0)
					vSumGD1 = g1.MulAdd(d1, vSumGD1)
					vSumGD2 = g2.MulAdd(d2, vSumGD2)
					vSumGD3 = g3.MulAdd(d3, vSumGD3)
				}
			}

			vVar0.Store(stripBuf[:vLen])
			vVar1.Store(stripBuf[vLen : 2*vLen])
			vVar2.Store(stripBuf[2*vLen : 3*vLen])
			vVar3.Store(stripBuf[3*vLen : 4*vLen])
			for i := range 4 * vLen {
				stripBuf[i] = float32(1.0 / math.Sqrt(float64(stripBuf[i]*invNormSizeF)+epsilon))
			}
			vInvStd0 := simd.LoadFloat32s(stripBuf[:vLen])
			vInvStd1 := simd.LoadFloat32s(stripBuf[vLen : 2*vLen])
			vInvStd2 := simd.LoadFloat32s(stripBuf[2*vLen : 3*vLen])
			vInvStd3 := simd.LoadFloat32s(stripBuf[3*vLen : 4*vLen])

			vMeanG0 := vSumG0.Mul(vInvN)
			vMeanG1 := vSumG1.Mul(vInvN)
			vMeanG2 := vSumG2.Mul(vInvN)
			vMeanG3 := vSumG3.Mul(vInvN)
			vNegMeanGHat0 := vZero.Sub(vSumGD0.Mul(vInvN).Mul(vInvStd0))
			vNegMeanGHat1 := vZero.Sub(vSumGD1.Mul(vInvN).Mul(vInvStd1))
			vNegMeanGHat2 := vZero.Sub(vSumGD2.Mul(vInvN).Mul(vInvStd2))
			vNegMeanGHat3 := vZero.Sub(vSumGD3.Mul(vInvN).Mul(vInvStd3))

			// Pass 3: dx and dGamma/dBeta over b
			for b := range bDim {
				off := baseA + b*cDim + c
				dy0 := simd.LoadFloat32s(dOutData[off:])
				dy1 := simd.LoadFloat32s(dOutData[off+vLen:])
				dy2 := simd.LoadFloat32s(dOutData[off+2*vLen:])
				dy3 := simd.LoadFloat32s(dOutData[off+3*vLen:])

				xHat0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0).Mul(vInvStd0)
				xHat1 := simd.LoadFloat32s(inData[off+vLen:]).Sub(vMean1).Mul(vInvStd1)
				xHat2 := simd.LoadFloat32s(inData[off+2*vLen:]).Sub(vMean2).Mul(vInvStd2)
				xHat3 := simd.LoadFloat32s(inData[off+3*vLen:]).Sub(vMean3).Mul(vInvStd3)

				if hasDGamma {
					vDG := dy0.Mul(xHat0)
					vDG = dy1.MulAdd(xHat1, vDG)
					vDG = dy2.MulAdd(xHat2, vDG)
					vDG = dy3.MulAdd(xHat3, vDG)
					dGammaData[b] += sumFloat32s(vDG, tmp)
				}
				if hasDBeta {
					vDB := dy0.Add(dy1).Add(dy2.Add(dy3))
					dBetaData[b] += sumFloat32s(vDB, tmp)
				}

				g0, g1, g2, g3 := dy0, dy1, dy2, dy3
				if hasGamma {
					vG := simd.BroadcastFloat32s(gammaData[b])
					g0 = dy0.Mul(vG)
					g1 = dy1.Mul(vG)
					g2 = dy2.Mul(vG)
					g3 = dy3.Mul(vG)
				}
				xHat0.MulAdd(vNegMeanGHat0, g0.Sub(vMeanG0)).Mul(vInvStd0).Store(dxData[off:])
				xHat1.MulAdd(vNegMeanGHat1, g1.Sub(vMeanG1)).Mul(vInvStd1).Store(dxData[off+vLen:])
				xHat2.MulAdd(vNegMeanGHat2, g2.Sub(vMeanG2)).Mul(vInvStd2).Store(dxData[off+2*vLen:])
				xHat3.MulAdd(vNegMeanGHat3, g3.Sub(vMeanG3)).Mul(vInvStd3).Store(dxData[off+3*vLen:])
			}
		}

		// 1*vLen strip loop
		for ; c+vLen <= cDim; c += vLen {
			vSum0 := vZero
			for b := range bDim {
				vSum0 = vSum0.Add(simd.LoadFloat32s(inData[baseA+b*cDim+c:]))
			}
			vMean0 := vSum0.Mul(vInvN)

			vVar0, vSumG0, vSumGD0 := vZero, vZero, vZero
			if hasGamma {
				for b := range bDim {
					off := baseA + b*cDim + c
					d0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0)
					g0 := simd.LoadFloat32s(dOutData[off:]).Mul(simd.BroadcastFloat32s(gammaData[b]))
					vVar0 = d0.MulAdd(d0, vVar0)
					vSumG0 = vSumG0.Add(g0)
					vSumGD0 = g0.MulAdd(d0, vSumGD0)
				}
			} else {
				for b := range bDim {
					off := baseA + b*cDim + c
					d0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0)
					g0 := simd.LoadFloat32s(dOutData[off:])
					vVar0 = d0.MulAdd(d0, vVar0)
					vSumG0 = vSumG0.Add(g0)
					vSumGD0 = g0.MulAdd(d0, vSumGD0)
				}
			}
			vVar0.Store(stripBuf[:vLen])
			for i := range vLen {
				stripBuf[i] = float32(1.0 / math.Sqrt(float64(stripBuf[i]*invNormSizeF)+epsilon))
			}
			vInvStd0 := simd.LoadFloat32s(stripBuf[:vLen])
			vMeanG0 := vSumG0.Mul(vInvN)
			vNegMeanGHat0 := vZero.Sub(vSumGD0.Mul(vInvN).Mul(vInvStd0))

			for b := range bDim {
				off := baseA + b*cDim + c
				dy0 := simd.LoadFloat32s(dOutData[off:])
				xHat0 := simd.LoadFloat32s(inData[off:]).Sub(vMean0).Mul(vInvStd0)
				if hasDGamma {
					dGammaData[b] += sumFloat32s(dy0.Mul(xHat0), tmp)
				}
				if hasDBeta {
					dBetaData[b] += sumFloat32s(dy0, tmp)
				}
				g0 := dy0
				if hasGamma {
					g0 = dy0.Mul(simd.BroadcastFloat32s(gammaData[b]))
				}
				xHat0.MulAdd(vNegMeanGHat0, g0.Sub(vMeanG0)).Mul(vInvStd0).Store(dxData[off:])
			}
		}

		// Scalar remainder loop
		for ; c < cDim; c++ {
			var s float32
			for b := range bDim {
				s += inData[baseA+b*cDim+c]
			}
			mean := s * invNormSizeF

			var vSum, sG, sGD float32
			for b := range bDim {
				off := baseA + b*cDim + c
				d := inData[off] - mean
				vSum += d * d
				g := dOutData[off]
				if hasGamma {
					g *= gammaData[b]
				}
				sG += g
				sGD += g * d
			}
			invStd := float32(1.0 / math.Sqrt(float64(vSum*invNormSizeF)+epsilon))
			meanG := sG * invNormSizeF
			meanGHat := sGD * invNormSizeF * invStd

			for b := range bDim {
				off := baseA + b*cDim + c
				xHat := (inData[off] - mean) * invStd
				dy := dOutData[off]
				g := dy
				if hasGamma {
					g *= gammaData[b]
				}
				dxData[off] = invStd * (g - meanG - xHat*meanGHat)
				if hasDGamma {
					dGammaData[b] += dy * xHat
				}
				if hasDBeta {
					dBetaData[b] += dy
				}
			}
		}
	}
}

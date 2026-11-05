// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package ops

import (
	"simd/archsimd"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops/avx2"
	"github.com/gomlx/compute/internal/gobackend/ops/avx512"
	"github.com/gomlx/compute/shapes"
)

func init() {
	if gobackend.IsAVX2Allowed {
		gobackend.SetNodeExecutor(compute.OpTypeConvGeneral, gobackend.PriorityAVX2, execConvGeneralAVX2)
	}
	if gobackend.IsAVX512Allowed {
		gobackend.SetNodeExecutor(compute.OpTypeConvGeneral, gobackend.PriorityAVX512, execConvGeneralAVX512)
	}
}

func getScratchFloat32(backend *gobackend.Backend, size int) (*gobackend.Buffer, []float32, error) {
	buf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, size))
	if err != nil {
		return nil, nil, err
	}
	return buf, buf.Flat.([]float32), nil
}

func execConvGeneralAVX2(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}
	p := GetConv2DParams(node)
	switch p.Kind {
	case Conv2DStandard:
		return execConvStandardAVX2(backend, &p, input, kernel, node.Shape)
	case Conv2DKernelGrad:
		return execConvKernelGradAVX2(backend, &p, input, kernel, node.Shape)
	default:
		return nil, gobackend.ErrFallback
	}
}

func execConvGeneralAVX512(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}
	p := GetConv2DParams(node)
	switch p.Kind {
	case Conv2DStandard:
		return execConvStandardAVX512(backend, &p, input, kernel, node.Shape)
	case Conv2DKernelGrad:
		return execConvKernelGradAVX512(backend, &p, input, kernel, node.Shape)
	default:
		return nil, gobackend.ErrFallback
	}
}

// execConvStandardAVX2 executes 1D and 2D ChannelsLast Forward and VJP-wrt-Input convolutions
// using the 4x16 AVX2 microkernel (ConvAVX2Kernel4x16Float32Asm) without LHS transposition.
func execConvStandardAVX2(
	backend *gobackend.Backend,
	p *Conv2DParams,
	inputBuf, kernelBuf *gobackend.Buffer,
	outputShape shapes.Shape,
) (*gobackend.Buffer, error) {
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	M := p.BatchSize * p.OutH * p.OutW
	K := p.KernelH * p.KernelW * p.InChannels
	N := p.OutChannels
	if M == 0 || N == 0 {
		return output, nil
	}
	if K == 0 {
		output.Zeros()
		return output, nil
	}

	input := inputBuf.Flat.([]float32)
	kernel := kernelBuf.Flat.([]float32)
	out := output.Flat.([]float32)

	const panelCols = 16
	numPanels := (N + panelCols - 1) / panelCols
	panelSize := K * panelCols

	packedRHSBuf, packedRHS, err := getScratchFloat32(backend, numPanels*panelSize)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	defer backend.PutBuffer(packedRHSBuf)
	PackConvKernelFloat32(p, kernel, packedRHS, panelCols)

	mTile := max(32, min(256, ((4096/max(1, K))+31)&^31))
	numMTiles := (M + mTile - 1) / mTile

	processMTile := func(mStart, mCount int, patchBuf []float32) {
		var lhsSlice []float32
		if p.Is1x1Unit {
			if mCount%4 == 0 {
				lhsSlice = input[mStart*K : (mStart+mCount)*K]
			} else {
				copy(patchBuf[:mCount*K], input[mStart*K:(mStart+mCount)*K])
				lhsSlice = patchBuf
			}
		} else {
			ExtractPatches2DFloat32(p, input, patchBuf, mStart, mCount)
			lhsSlice = patchBuf
		}

		var edgeOut [4 * 16]float32
		lhsRowStrideBytes := K * 4
		outRowStrideBytes := N * 4

		for r := 0; r < mCount; r += 4 {
			validRows := min(4, mCount-r)
			lhsRowPtr := unsafe.Pointer(&lhsSlice[r*K])

			for panelIdx := range numPanels {
				colStart := panelIdx * 16
				validCols := min(16, N-colStart)
				rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize])

				if validRows == 4 && validCols == 16 {
					outPtr := unsafe.Pointer(&out[(mStart+r)*N+colStart])
					avx2.ConvAVX2Kernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 64, outRowStrideBytes)
				} else {
					avx2.ConvAVX2Kernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 64, 64)
					for ir := range validRows {
						dstOff := (mStart+r+ir)*N + colStart
						copy(out[dstOff:dstOff+validCols], edgeOut[ir*16:ir*16+validCols])
					}
				}
			}
		}
	}

	maxWorkers := min(numMTiles, backend.Scheduler.AdjustedMaxParallelism())
	maxWorkers = min(maxWorkers, max(1, (M*K*N)/262144))
	needPatchBuf := !p.Is1x1Unit || M%4 != 0
	if maxWorkers <= 1 || M*K*N < 65536 {
		var patchBuf []float32
		if needPatchBuf {
			patchBufObj, bufSlice, err := getScratchFloat32(backend, mTile*K)
			if err != nil {
				backend.PutBuffer(output)
				return nil, err
			}
			defer backend.PutBuffer(patchBufObj)
			patchBuf = bufSlice
		}
		for tileIdx := range numMTiles {
			mStart := tileIdx * mTile
			mCount := min(mTile, M-mStart)
			processMTile(mStart, mCount, patchBuf)
		}
		return output, nil
	}

	var allPatchBuf []float32
	if needPatchBuf {
		patchBufObj, bufSlice, err := getScratchFloat32(backend, maxWorkers*mTile*K)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(patchBufObj)
		allPatchBuf = bufSlice
	}

	var nextWorker atomic.Int64
	var nextTile atomic.Int64
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		var patchBuf []float32
		if needPatchBuf {
			wIdx := int(nextWorker.Add(1) - 1)
			patchBuf = allPatchBuf[wIdx*mTile*K : (wIdx+1)*mTile*K]
		}
		for {
			tileIdx := int(nextTile.Add(1) - 1)
			if tileIdx >= numMTiles {
				break
			}
			mStart := tileIdx * mTile
			mCount := min(mTile, M-mStart)
			processMTile(mStart, mCount, patchBuf)
		}
	})
	return output, nil
}

// execConvStandardAVX512 executes 1D and 2D ChannelsLast Forward and VJP-wrt-Input convolutions
// using the 8x32 and 4x32 AVX-512 microkernels (ConvAVX512Kernel8x32Float32Asm and
// ConvAVX512Kernel4x32Float32Asm) without LHS transposition.
func execConvStandardAVX512(
	backend *gobackend.Backend,
	p *Conv2DParams,
	inputBuf, kernelBuf *gobackend.Buffer,
	outputShape shapes.Shape,
) (*gobackend.Buffer, error) {
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	M := p.BatchSize * p.OutH * p.OutW
	K := p.KernelH * p.KernelW * p.InChannels
	N := p.OutChannels
	if M == 0 || N == 0 {
		return output, nil
	}
	if K == 0 {
		output.Zeros()
		return output, nil
	}

	input := inputBuf.Flat.([]float32)
	kernel := kernelBuf.Flat.([]float32)
	out := output.Flat.([]float32)

	const panelCols = 32
	numPanels := (N + panelCols - 1) / panelCols
	panelSize := K * panelCols

	packedRHSBuf, packedRHS, err := getScratchFloat32(backend, numPanels*panelSize)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	defer backend.PutBuffer(packedRHSBuf)
	PackConvKernelFloat32(p, kernel, packedRHS, panelCols)

	mTile := max(32, min(256, ((4096/max(1, K))+31)&^31))
	numMTiles := (M + mTile - 1) / mTile

	processMTile := func(mStart, mCount int, patchBuf []float32) {
		var lhsSlice []float32
		if p.Is1x1Unit {
			if mCount%8 == 0 {
				lhsSlice = input[mStart*K : (mStart+mCount)*K]
			} else {
				copy(patchBuf[:mCount*K], input[mStart*K:(mStart+mCount)*K])
				lhsSlice = patchBuf
			}
		} else {
			ExtractPatches2DFloat32(p, input, patchBuf, mStart, mCount)
			lhsSlice = patchBuf
		}

		var edgeOut [8 * 32]float32
		lhsRowStrideBytes := K * 4
		outRowStrideBytes := N * 4

		for r := 0; r < mCount; r += 8 {
			validRows := min(8, mCount-r)
			lhsRowPtr := unsafe.Pointer(&lhsSlice[r*K])

			for panelIdx := range numPanels {
				colStart := panelIdx * 32
				validCols := min(32, N-colStart)
				rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize])

				if validRows == 8 && validCols == 32 {
					outPtr := unsafe.Pointer(&out[(mStart+r)*N+colStart])
					avx512.ConvAVX512Kernel8x32Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 128, outRowStrideBytes)
				} else if validRows == 4 && validCols == 32 {
					outPtr := unsafe.Pointer(&out[(mStart+r)*N+colStart])
					avx512.ConvAVX512Kernel4x32Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 128, outRowStrideBytes)
				} else if validRows <= 4 {
					avx512.ConvAVX512Kernel4x32Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 128)
					for ir := range validRows {
						dstOff := (mStart+r+ir)*N + colStart
						copy(out[dstOff:dstOff+validCols], edgeOut[ir*32:ir*32+validCols])
					}
				} else {
					avx512.ConvAVX512Kernel8x32Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 128)
					for ir := range validRows {
						dstOff := (mStart+r+ir)*N + colStart
						copy(out[dstOff:dstOff+validCols], edgeOut[ir*32:ir*32+validCols])
					}
				}
			}
		}
	}

	maxWorkers := min(numMTiles, backend.Scheduler.AdjustedMaxParallelism())
	maxWorkers = min(maxWorkers, max(1, (M*K*N)/262144))
	needPatchBuf := !p.Is1x1Unit || M%8 != 0
	if maxWorkers <= 1 || M*K*N < 65536 {
		var patchBuf []float32
		if needPatchBuf {
			patchBufObj, bufSlice, err := getScratchFloat32(backend, mTile*K)
			if err != nil {
				backend.PutBuffer(output)
				return nil, err
			}
			defer backend.PutBuffer(patchBufObj)
			patchBuf = bufSlice
		}
		for tileIdx := range numMTiles {
			mStart := tileIdx * mTile
			mCount := min(mTile, M-mStart)
			processMTile(mStart, mCount, patchBuf)
		}
		return output, nil
	}

	var allPatchBuf []float32
	if needPatchBuf {
		patchBufObj, bufSlice, err := getScratchFloat32(backend, maxWorkers*mTile*K)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(patchBufObj)
		allPatchBuf = bufSlice
	}

	var nextWorker atomic.Int64
	var nextTile atomic.Int64
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		var patchBuf []float32
		if needPatchBuf {
			wIdx := int(nextWorker.Add(1) - 1)
			patchBuf = allPatchBuf[wIdx*mTile*K : (wIdx+1)*mTile*K]
		}
		for {
			tileIdx := int(nextTile.Add(1) - 1)
			if tileIdx >= numMTiles {
				break
			}
			mStart := tileIdx * mTile
			mCount := min(mTile, M-mStart)
			processMTile(mStart, mCount, patchBuf)
		}
	})
	return output, nil
}

// execConvKernelGradAVX2 computes VJP wrt Kernel (Conv2DKernelGrad) using AVX2 outer-product
// microkernels (ConvKernelGradAVX2_4x16Float32Asm and ConvKernelGradAVX2_1x16Float32Asm) with
// zero packing and zero intermediate buffers.
func execConvKernelGradAVX2(
	backend *gobackend.Backend,
	p *Conv2DParams,
	inputBuf, kernelBuf *gobackend.Buffer,
	outputShape shapes.Shape,
) (*gobackend.Buffer, error) {
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	output.Zeros()
	if output.RawShape.Size() == 0 || inputBuf.RawShape.Size() == 0 || kernelBuf.RawShape.Size() == 0 {
		return output, nil
	}

	x := inputBuf.Flat.([]float32)
	v := kernelBuf.Flat.([]float32)
	out := output.Flat.([]float32)

	kH, kW := p.KernelH, p.KernelW
	inC, outC := p.InChannels, p.OutChannels
	const rowTile = 8
	numRowPanels := (inC + rowTile - 1) / rowTile
	numColPanels := (outC + 15) / 16
	panelsPerFilter := numRowPanels * numColPanels
	totalTasks := kH * kW * panelsPerFilter

	processTask := func(taskIdx int) {
		spatialIdx := taskIdx / panelsPerFilter
		rem := taskIdx % panelsPerFilter
		rowPanel := rem / numColPanels
		colPanel := rem % numColPanels
		ky := spatialIdx / kW
		kx := spatialIdx % kW
		cInStart := rowPanel * rowTile
		validRows := min(rowTile, inC-cInStart)
		cOutStart := colPanel * 16
		validCols := min(16, outC-cOutStart)
		convKernelGradAVX2TileFloat32(p, x, v, out, ky, kx, cInStart, validRows, cOutStart, validCols)
	}

	maxWorkers := min(totalTasks, backend.Scheduler.AdjustedMaxParallelism())
	if maxWorkers <= 1 {
		for taskIdx := range totalTasks {
			processTask(taskIdx)
		}
		return output, nil
	}

	var nextTask atomic.Int64
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		for {
			taskIdx := int(nextTask.Add(1) - 1)
			if taskIdx >= totalTasks {
				break
			}
			processTask(taskIdx)
		}
	})
	return output, nil
}

// execConvKernelGradAVX512 computes VJP wrt Kernel (Conv2DKernelGrad) using AVX-512 outer-product
// microkernels (ConvKernelGradAVX512_8x32Float32Asm, 4x32, and 1x32) with zero packing and zero
// intermediate buffers.
func execConvKernelGradAVX512(
	backend *gobackend.Backend,
	p *Conv2DParams,
	inputBuf, kernelBuf *gobackend.Buffer,
	outputShape shapes.Shape,
) (*gobackend.Buffer, error) {
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	output.Zeros()
	if output.RawShape.Size() == 0 || inputBuf.RawShape.Size() == 0 || kernelBuf.RawShape.Size() == 0 {
		return output, nil
	}

	x := inputBuf.Flat.([]float32)
	v := kernelBuf.Flat.([]float32)
	out := output.Flat.([]float32)

	kH, kW := p.KernelH, p.KernelW
	inC, outC := p.InChannels, p.OutChannels
	const rowTile = 8
	numRowPanels := (inC + rowTile - 1) / rowTile
	numColPanels := (outC + 31) / 32
	panelsPerFilter := numRowPanels * numColPanels
	totalTasks := kH * kW * panelsPerFilter

	processTask := func(taskIdx int) {
		spatialIdx := taskIdx / panelsPerFilter
		rem := taskIdx % panelsPerFilter
		rowPanel := rem / numColPanels
		colPanel := rem % numColPanels
		ky := spatialIdx / kW
		kx := spatialIdx % kW
		cInStart := rowPanel * rowTile
		validRows := min(rowTile, inC-cInStart)
		cOutStart := colPanel * 32
		validCols := min(32, outC-cOutStart)
		convKernelGradAVX512TileFloat32(p, x, v, out, ky, kx, cInStart, validRows, cOutStart, validCols)
	}

	maxWorkers := min(totalTasks, backend.Scheduler.AdjustedMaxParallelism())
	if maxWorkers <= 1 {
		for taskIdx := range totalTasks {
			processTask(taskIdx)
		}
		return output, nil
	}

	var nextTask atomic.Int64
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		for {
			taskIdx := int(nextTask.Add(1) - 1)
			if taskIdx >= totalTasks {
				break
			}
			processTask(taskIdx)
		}
	})
	return output, nil
}

func convKernelGradAVX512TileFloat32(
	p *Conv2DParams,
	x, v, out []float32,
	ky, kx, cInStart, validRows, cOutStart, validCols int,
) {
	outYStart, outYEnd, inYStart, outXStart, outXEnd, inXStart := p.KernelGradValidRange(ky, kx)
	numY := outYEnd - outYStart
	numX := outXEnd - outXStart
	if numY <= 0 || numX <= 0 {
		return
	}

	batchSize := p.BatchSize
	inC := p.InChannels
	outC := p.OutChannels
	xStepFloats := p.KerDilW * inC
	vStepFloats := outC
	xYStrideFloats := p.KerDilH * p.InW * inC
	vYStrideFloats := p.OutW * outC
	xBatchStrideFloats := p.InH * p.InW * inC
	vBatchStrideFloats := p.OutH * p.OutW * outC

	if p.KerDilH == 1 && p.KerDilW == 1 && outXStart == 0 && outXEnd == p.OutW && inXStart == 0 && p.InW == p.OutW {
		numX = numY * p.OutW
		numY = 1
		if outYStart == 0 && outYEnd == p.OutH && inYStart == 0 && p.InH == p.OutH {
			numX = batchSize * numX
			batchSize = 1
		}
	}

	colOff := 0
	for ; colOff+32 <= validCols; colOff += 32 {
		cOut := cOutStart + colOff
		vBaseIdx := (outYStart*p.OutW+outXStart)*outC + cOut

		rOff := 0
		for ; rOff+8 <= validRows; rOff += 8 {
			cIn := cInStart + rOff
			xBaseIdx := (inYStart*p.InW+inXStart)*inC + cIn
			outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
			avx512.ConvKernelGradAVX512_8x32Float32Asm(
				unsafe.Pointer(&x[xBaseIdx]),
				unsafe.Pointer(&v[vBaseIdx]),
				unsafe.Pointer(&out[outIdx]),
				batchSize, numY, numX,
				xBatchStrideFloats*4, vBatchStrideFloats*4,
				xYStrideFloats*4, vYStrideFloats*4,
				xStepFloats*4, vStepFloats*4,
				outC*4,
			)
		}
		for ; rOff+4 <= validRows; rOff += 4 {
			cIn := cInStart + rOff
			xBaseIdx := (inYStart*p.InW+inXStart)*inC + cIn
			outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
			avx512.ConvKernelGradAVX512_4x32Float32Asm(
				unsafe.Pointer(&x[xBaseIdx]),
				unsafe.Pointer(&v[vBaseIdx]),
				unsafe.Pointer(&out[outIdx]),
				batchSize, numY, numX,
				xBatchStrideFloats*4, vBatchStrideFloats*4,
				xYStrideFloats*4, vYStrideFloats*4,
				xStepFloats*4, vStepFloats*4,
				outC*4,
			)
		}
		for ; rOff < validRows; rOff++ {
			cIn := cInStart + rOff
			xBaseIdx := (inYStart*p.InW+inXStart)*inC + cIn
			outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
			avx512.ConvKernelGradAVX512_1x32Float32Asm(
				unsafe.Pointer(&x[xBaseIdx]),
				unsafe.Pointer(&v[vBaseIdx]),
				unsafe.Pointer(&out[outIdx]),
				batchSize, numY, numX,
				xBatchStrideFloats*4, vBatchStrideFloats*4,
				xYStrideFloats*4, vYStrideFloats*4,
				xStepFloats*4, vStepFloats*4,
			)
		}
	}

	if colOff < validCols {
		convKernelGradAVX2TileFloat32(p, x, v, out, ky, kx, cInStart, validRows, cOutStart+colOff, validCols-colOff)
	}
}

func convKernelGradAVX2TileFloat32(
	p *Conv2DParams,
	x, v, out []float32,
	ky, kx, cInStart, validRows, cOutStart, validCols int,
) {
	outYStart, outYEnd, inYStart, outXStart, outXEnd, inXStart := p.KernelGradValidRange(ky, kx)
	numY := outYEnd - outYStart
	numX := outXEnd - outXStart
	if numY <= 0 || numX <= 0 {
		return
	}

	batchSize := p.BatchSize
	inC := p.InChannels
	outC := p.OutChannels
	xStepFloats := p.KerDilW * inC
	vStepFloats := outC
	xYStrideFloats := p.KerDilH * p.InW * inC
	vYStrideFloats := p.OutW * outC
	xBatchStrideFloats := p.InH * p.InW * inC
	vBatchStrideFloats := p.OutH * p.OutW * outC

	if p.KerDilH == 1 && p.KerDilW == 1 && outXStart == 0 && outXEnd == p.OutW && inXStart == 0 && p.InW == p.OutW {
		numX = numY * p.OutW
		numY = 1
		if outYStart == 0 && outYEnd == p.OutH && inYStart == 0 && p.InH == p.OutH {
			numX = batchSize * numX
			batchSize = 1
		}
	}

	for colOff := 0; colOff < validCols; colOff += 16 {
		cOut := cOutStart + colOff
		subCols := min(16, validCols-colOff)
		vBaseIdx := (outYStart*p.OutW+outXStart)*outC + cOut

		if subCols == 16 {
			rOff := 0
			for ; rOff+4 <= validRows; rOff += 4 {
				cIn := cInStart + rOff
				xBaseIdx := (inYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				avx2.ConvKernelGradAVX2_4x16Float32Asm(
					unsafe.Pointer(&x[xBaseIdx]),
					unsafe.Pointer(&v[vBaseIdx]),
					unsafe.Pointer(&out[outIdx]),
					batchSize, numY, numX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
					outC*4,
				)
			}
			for ; rOff < validRows; rOff++ {
				cIn := cInStart + rOff
				xBaseIdx := (inYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				avx2.ConvKernelGradAVX2_1x16Float32Asm(
					unsafe.Pointer(&x[xBaseIdx]),
					unsafe.Pointer(&v[vBaseIdx]),
					unsafe.Pointer(&out[outIdx]),
					batchSize, numY, numX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
				)
			}
		} else {
			xBaseIdx := (inYStart*p.InW+inXStart)*inC + cInStart
			outBaseIdx := ((ky*p.KernelW+kx)*inC+cInStart)*outC + cOut
			convKernelGradTailFloat32(
				x, v, out,
				batchSize, numY, numX,
				xBaseIdx, vBaseIdx, outBaseIdx,
				xBatchStrideFloats, vBatchStrideFloats,
				xYStrideFloats, vYStrideFloats,
				xStepFloats, vStepFloats,
				outC, validRows, subCols,
			)
		}
	}
}

// convKernelGradTailFloat32 handles remainder columns (validCols < 16) in Conv2DKernelGrad
// without reading past the end of x or v.
func convKernelGradTailFloat32(
	x, v, out []float32,
	batchSize, numY, numX int,
	xBaseIdx, vBaseIdx, outBaseIdx int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStride int,
	validRows, validCols int,
) {
	col := 0
	for ; col+8 <= validCols; col += 8 {
		r := 0
		for ; r+4 <= validRows; r += 4 {
			acc0 := archsimd.BroadcastFloat32x8(0)
			acc1 := archsimd.BroadcastFloat32x8(0)
			acc2 := archsimd.BroadcastFloat32x8(0)
			acc3 := archsimd.BroadcastFloat32x8(0)

			xB := xBaseIdx + r
			vB := vBaseIdx + col
			for range batchSize {
				xY := xB
				vY := vB
				for range numY {
					xP := xY
					vP := vY
					for range numX {
						vv := archsimd.LoadFloat32x8(v[vP : vP+8])
						acc0 = vv.MulAdd(archsimd.BroadcastFloat32x8(x[xP+0]), acc0)
						acc1 = vv.MulAdd(archsimd.BroadcastFloat32x8(x[xP+1]), acc1)
						acc2 = vv.MulAdd(archsimd.BroadcastFloat32x8(x[xP+2]), acc2)
						acc3 = vv.MulAdd(archsimd.BroadcastFloat32x8(x[xP+3]), acc3)
						xP += xStep
						vP += vStep
					}
					xY += xYStride
					vY += vYStride
				}
				xB += xBatchStride
				vB += vBatchStride
			}
			outOff := outBaseIdx + r*outRowStride + col
			acc0.Store(out[outOff : outOff+8])
			acc1.Store(out[outOff+outRowStride : outOff+outRowStride+8])
			acc2.Store(out[outOff+2*outRowStride : outOff+2*outRowStride+8])
			acc3.Store(out[outOff+3*outRowStride : outOff+3*outRowStride+8])
		}
		for ; r < validRows; r++ {
			acc0 := archsimd.BroadcastFloat32x8(0)
			xB := xBaseIdx + r
			vB := vBaseIdx + col
			for range batchSize {
				xY := xB
				vY := vB
				for range numY {
					xP := xY
					vP := vY
					for range numX {
						vv := archsimd.LoadFloat32x8(v[vP : vP+8])
						acc0 = vv.MulAdd(archsimd.BroadcastFloat32x8(x[xP]), acc0)
						xP += xStep
						vP += vStep
					}
					xY += xYStride
					vY += vYStride
				}
				xB += xBatchStride
				vB += vBatchStride
			}
			outOff := outBaseIdx + r*outRowStride + col
			acc0.Store(out[outOff : outOff+8])
		}
	}

	for ; col+4 <= validCols; col += 4 {
		r := 0
		for ; r+4 <= validRows; r += 4 {
			acc0 := archsimd.BroadcastFloat32x4(0)
			acc1 := archsimd.BroadcastFloat32x4(0)
			acc2 := archsimd.BroadcastFloat32x4(0)
			acc3 := archsimd.BroadcastFloat32x4(0)

			xB := xBaseIdx + r
			vB := vBaseIdx + col
			for range batchSize {
				xY := xB
				vY := vB
				for range numY {
					xP := xY
					vP := vY
					for range numX {
						vv := archsimd.LoadFloat32x4(v[vP : vP+4])
						acc0 = vv.MulAdd(archsimd.BroadcastFloat32x4(x[xP+0]), acc0)
						acc1 = vv.MulAdd(archsimd.BroadcastFloat32x4(x[xP+1]), acc1)
						acc2 = vv.MulAdd(archsimd.BroadcastFloat32x4(x[xP+2]), acc2)
						acc3 = vv.MulAdd(archsimd.BroadcastFloat32x4(x[xP+3]), acc3)
						xP += xStep
						vP += vStep
					}
					xY += xYStride
					vY += vYStride
				}
				xB += xBatchStride
				vB += vBatchStride
			}
			outOff := outBaseIdx + r*outRowStride + col
			acc0.Store(out[outOff : outOff+4])
			acc1.Store(out[outOff+outRowStride : outOff+outRowStride+4])
			acc2.Store(out[outOff+2*outRowStride : outOff+2*outRowStride+4])
			acc3.Store(out[outOff+3*outRowStride : outOff+3*outRowStride+4])
		}
		for ; r < validRows; r++ {
			acc0 := archsimd.BroadcastFloat32x4(0)
			xB := xBaseIdx + r
			vB := vBaseIdx + col
			for range batchSize {
				xY := xB
				vY := vB
				for range numY {
					xP := xY
					vP := vY
					for range numX {
						vv := archsimd.LoadFloat32x4(v[vP : vP+4])
						acc0 = vv.MulAdd(archsimd.BroadcastFloat32x4(x[xP]), acc0)
						xP += xStep
						vP += vStep
					}
					xY += xYStride
					vY += vYStride
				}
				xB += xBatchStride
				vB += vBatchStride
			}
			outOff := outBaseIdx + r*outRowStride + col
			acc0.Store(out[outOff : outOff+4])
		}
	}

	for ; col < validCols; col++ {
		for r := range validRows {
			var sum float32
			xB := xBaseIdx + r
			vB := vBaseIdx + col
			for range batchSize {
				xY := xB
				vY := vB
				for range numY {
					xP := xY
					vP := vY
					for range numX {
						sum += x[xP] * v[vP]
						xP += xStep
						vP += vStep
					}
					xY += xYStride
					vY += vYStride
				}
				xB += xBatchStride
				vB += vBatchStride
			}
			out[outBaseIdx+r*outRowStride+col] = sum
		}
	}
}

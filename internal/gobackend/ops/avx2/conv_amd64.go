// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"simd/archsimd"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
)

// ConvAVX2Kernel4x16Float32Asm computes a 4x16 float32 output tile directly from
// 4 rows of LHS ([4, K] with row byte stride lhsRowStrideBytes) and 16 columns of
// packed RHS ([K, 16+] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvAVX2Kernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX2Direct2DKernel4x16Float32Asm computes a 4x16 float32 output tile for
// 4 horizontally adjacent interior 2D output pixels directly from the input tensor
// without im2col patch extraction, iterating across numKy valid filter rows of
// width kWC = kW * inC each and advancing LHS row pointers by lhsKyJumpBytes
// between filter rows.
//
//go:noescape
func ConvAVX2Direct2DKernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX2Direct2DKernel2x16Float32Asm computes a 2x16 float32 output tile for
// 2 horizontally adjacent interior 2D output pixels directly from the input tensor
// without im2col patch extraction.
//
//go:noescape
func ConvAVX2Direct2DKernel2x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX2Direct2DKernel1x16Float32Asm computes a 1x16 float32 output row for a
// single 2D output pixel (interior or clipped border) directly from the input tensor
// without im2col patch extraction, advancing lhsPtr by lhsKyJumpBytes and rhsPtr by
// rhsKyJumpBytes between filter rows.
//
//go:noescape
func ConvAVX2Direct2DKernel1x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsKyJumpBytes, rhsKyJumpBytes, rhsStepBytes int,
)

// ConvKernelGradAVX2_4x16Float32Asm accumulates a 4x16 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 4x16 result to outPtr.
//
//go:noescape
func ConvKernelGradAVX2_4x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradAVX2_1x16Float32Asm accumulates a 1x16 float32 kernel-gradient row
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 16-float row to outPtr.
//
//go:noescape
func ConvKernelGradAVX2_1x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep int,
)

func execConvGeneralAVX2(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}
	p := ops.GetConv2DParams(node)
	switch p.Kind {
	case ops.Conv2DStandard:
		return execConvStandardAVX2(backend, &p, input, kernel, node.Shape)
	case ops.Conv2DKernelGrad:
		return execConvKernelGradAVX2(backend, &p, input, kernel, node.Shape)
	default:
		return nil, gobackend.ErrFallback
	}
}

// execConvStandardAVX2 executes 1D and 2D ChannelsLast Forward and VJP-wrt-Input convolutions
// using the 4x16 AVX2 microkernel (ConvAVX2Kernel4x16Float32Asm) without LHS transposition.
func execConvStandardAVX2(
	backend *gobackend.Backend,
	p *ops.Conv2DParams,
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

	packedRHSBuf, packedRHS, err := ops.GetScratchFloat32(backend, numPanels*panelSize)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	defer backend.PutBuffer(packedRHSBuf)
	ops.PackConvKernelFloat32(p, kernel, packedRHS, panelCols)

	// Direct zero-im2col 2D row fast path when InDilH == 1 && InDilW == 1 && KerDilW == 1.
	if !p.Is1x1Unit && p.InDilH == 1 && p.InDilW == 1 && p.KerDilW == 1 {
		outW := p.OutW
		outXIntStart := 0
		if p.PadLeft > 0 {
			outXIntStart = min(outW, (p.PadLeft+p.StrideW-1)/p.StrideW)
		}
		outXIntEnd := outXIntStart
		if maxInX0 := p.InW + p.PadLeft - p.KernelW; maxInX0 >= 0 {
			outXIntEnd = max(outXIntStart, min(outW, maxInX0/p.StrideW+1))
		}
		totalRows := p.BatchSize * p.OutH
		kWC := p.KernelW * p.InChannels
		if totalRows >= 4 || M*K*N < 65536 {
			directLhsRowStrideBytes := p.StrideW * p.InChannels * 4
			directLhsKyJumpBytes := (p.KerDilH*p.InW - p.KernelW) * p.InChannels * 4
			outRowStrideBytes := N * 4

			processRow := func(byIdx int) {
				b := byIdx / p.OutH
				outY := byIdx % p.OutH
				rowMStart := byIdx * outW

				kyStart := 0
				if minNeed := p.PadTop - outY*p.StrideH; minNeed > 0 {
					kyStart = min(p.KernelH, (minNeed+p.KerDilH-1)/p.KerDilH)
				}
				kyEnd := 0
				if maxNeed := p.InH + p.PadTop - outY*p.StrideH; maxNeed > 0 {
					kyEnd = min(p.KernelH, (maxNeed+p.KerDilH-1)/p.KerDilH)
				}
				numKy := kyEnd - kyStart
				if numKy <= 0 {
					clear(out[rowMStart*N : (rowMStart+outW)*N])
					return
				}

				inYStart := outY*p.StrideH + kyStart*p.KerDilH - p.PadTop
				inYBase := (b*p.InH + inYStart) * p.InW * p.InChannels
				rhsKyOff := kyStart * kWC * panelCols
				var edgeOut [4 * 16]float32

				processBorderPixel := func(outX int) {
					kxStart := max(0, p.PadLeft-outX*p.StrideW)
					kxEnd := min(p.KernelW, p.InW+p.PadLeft-outX*p.StrideW)
					numKx := kxEnd - kxStart
					if numKx <= 0 {
						clear(out[(rowMStart+outX)*N : (rowMStart+outX+1)*N])
						return
					}
					inXStart := outX*p.StrideW + kxStart - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inXStart*p.InChannels])
					kWCValid := numKx * p.InChannels
					lhsKyJump := (p.KerDilH*p.InW - numKx) * p.InChannels * 4
					rhsKyJump := (p.KernelW - numKx) * p.InChannels * panelCols * 4
					rhsKxStartOff := (kyStart*p.KernelW + kxStart) * p.InChannels * panelCols
					for panelIdx := range numPanels {
						colStart := panelIdx * 16
						validCols := min(16, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKxStartOff])
						if validCols == 16 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX2Direct2DKernel1x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWCValid, numKy, lhsKyJump, rhsKyJump, 64,
							)
						} else {
							ConvAVX2Direct2DKernel1x16Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWCValid, numKy, lhsKyJump, rhsKyJump, 64,
							)
							dstOff := (rowMStart+outX)*N + colStart
							copy(out[dstOff:dstOff+validCols], edgeOut[:validCols])
						}
					}
				}

				for outX := range outXIntStart {
					processBorderPixel(outX)
				}

				outX := outXIntStart
				for ; outX+4 <= outXIntEnd; outX += 4 {
					inX0 := outX*p.StrideW - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
					for panelIdx := range numPanels {
						colStart := panelIdx * 16
						validCols := min(16, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOff])
						if validCols == 16 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX2Direct2DKernel4x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 64, outRowStrideBytes,
							)
						} else {
							ConvAVX2Direct2DKernel4x16Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 64, 64,
							)
							for ir := range 4 {
								dstOff := (rowMStart+outX+ir)*N + colStart
								copy(out[dstOff:dstOff+validCols], edgeOut[ir*16:ir*16+validCols])
							}
						}
					}
				}
				for ; outX+2 <= outXIntEnd; outX += 2 {
					inX0 := outX*p.StrideW - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
					for panelIdx := range numPanels {
						colStart := panelIdx * 16
						validCols := min(16, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOff])
						if validCols == 16 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX2Direct2DKernel2x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 64, outRowStrideBytes,
							)
						} else {
							ConvAVX2Direct2DKernel2x16Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 64, 64,
							)
							for ir := range 2 {
								dstOff := (rowMStart+outX+ir)*N + colStart
								copy(out[dstOff:dstOff+validCols], edgeOut[ir*16:ir*16+validCols])
							}
						}
					}
				}
				for ; outX < outXIntEnd; outX++ {
					inX0 := outX*p.StrideW - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
					for panelIdx := range numPanels {
						colStart := panelIdx * 16
						validCols := min(16, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOff])
						if validCols == 16 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX2Direct2DKernel1x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directLhsKyJumpBytes, 0, 64,
							)
						} else {
							ConvAVX2Direct2DKernel1x16Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directLhsKyJumpBytes, 0, 64,
							)
							dstOff := (rowMStart+outX)*N + colStart
							copy(out[dstOff:dstOff+validCols], edgeOut[:validCols])
						}
					}
				}

				for outX := outXIntEnd; outX < outW; outX++ {
					processBorderPixel(outX)
				}
			}

			maxWorkers := min(totalRows, backend.Scheduler.AdjustedMaxParallelism())
			maxWorkers = min(maxWorkers, max(1, (M*K*N)/262144))
			if maxWorkers <= 1 || M*K*N < 65536 {
				for byIdx := range totalRows {
					processRow(byIdx)
				}
				return output, nil
			}

			var nextRow atomic.Int64
			backend.Scheduler.SaturateUpTo(maxWorkers, func() {
				for {
					byIdx := int(nextRow.Add(1) - 1)
					if byIdx >= totalRows {
						break
					}
					processRow(byIdx)
				}
			})
			return output, nil
		}
	}

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
			ops.ExtractPatches2DFloat32(p, input, patchBuf, mStart, mCount)
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
					ConvAVX2Kernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 64, outRowStrideBytes)
				} else {
					ConvAVX2Kernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 64, 64)
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
			patchBufObj, bufSlice, err := ops.GetScratchFloat32(backend, mTile*K)
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
		patchBufObj, bufSlice, err := ops.GetScratchFloat32(backend, maxWorkers*mTile*K)
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
	p *ops.Conv2DParams,
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

func convKernelGradAVX2TileFloat32(
	p *ops.Conv2DParams,
	x, v, out []float32,
	ky, kx, cInStart, validRows, cOutStart, validCols int,
) {
	outYStart, outYEnd, _, outXStart, outXEnd, _ := p.KernelGradValidRange(ky, kx)
	if outYEnd <= outYStart || outXEnd <= outXStart {
		return
	}
	convKernelGradAVX2SubRangeFloat32(p, x, v, out, ky, kx, cInStart, validRows, cOutStart, validCols, 0, p.BatchSize, outYStart, outYEnd)
}

func convKernelGradAVX2SubRangeFloat32(
	p *ops.Conv2DParams,
	x, v, out []float32,
	ky, kx, cInStart, validRows, cOutStart, validCols int,
	bStart, bCount, yStart, yEnd int,
) {
	outYStart, _, inYStart, outXStart, outXEnd, inXStart := p.KernelGradValidRange(ky, kx)
	numY := yEnd - yStart
	numX := outXEnd - outXStart
	if numY <= 0 || numX <= 0 || bCount <= 0 {
		return
	}

	batchSize := bCount
	inC := p.InChannels
	outC := p.OutChannels
	xStepFloats := p.KerDilW * inC
	vStepFloats := outC
	xYStrideFloats := p.KerDilH * p.InW * inC
	vYStrideFloats := p.OutW * outC
	xBatchStrideFloats := p.InH * p.InW * inC
	vBatchStrideFloats := p.OutH * p.OutW * outC
	curInYStart := inYStart + (yStart-outYStart)*p.KerDilH

	if p.KerDilH == 1 && p.KerDilW == 1 && outXStart == 0 && outXEnd == p.OutW && inXStart == 0 && p.InW == p.OutW {
		numX = numY * p.OutW
		numY = 1
		if yStart == 0 && yEnd == p.OutH && curInYStart == 0 && p.InH == p.OutH {
			numX = batchSize * numX
			batchSize = 1
		}
	}

	for colOff := 0; colOff < validCols; colOff += 16 {
		cOut := cOutStart + colOff
		subCols := min(16, validCols-colOff)
		vBaseIdx := bStart*vBatchStrideFloats + (yStart*p.OutW+outXStart)*outC + cOut

		if subCols == 16 {
			rOff := 0
			for ; rOff+4 <= validRows; rOff += 4 {
				cIn := cInStart + rOff
				xBaseIdx := bStart*xBatchStrideFloats + (curInYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				ConvKernelGradAVX2_4x16Float32Asm(
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
				xBaseIdx := bStart*xBatchStrideFloats + (curInYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				ConvKernelGradAVX2_1x16Float32Asm(
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
			xBaseIdx := bStart*xBatchStrideFloats + (curInYStart*p.InW+inXStart)*inC + cInStart
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
			acc0.Add(archsimd.LoadFloat32x8(out[outOff : outOff+8])).Store(out[outOff : outOff+8])
			acc1.Add(archsimd.LoadFloat32x8(out[outOff+outRowStride : outOff+outRowStride+8])).Store(out[outOff+outRowStride : outOff+outRowStride+8])
			acc2.Add(archsimd.LoadFloat32x8(out[outOff+2*outRowStride : outOff+2*outRowStride+8])).Store(out[outOff+2*outRowStride : outOff+2*outRowStride+8])
			acc3.Add(archsimd.LoadFloat32x8(out[outOff+3*outRowStride : outOff+3*outRowStride+8])).Store(out[outOff+3*outRowStride : outOff+3*outRowStride+8])
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
			acc0.Add(archsimd.LoadFloat32x8(out[outOff : outOff+8])).Store(out[outOff : outOff+8])
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
			acc0.Add(archsimd.LoadFloat32x4(out[outOff : outOff+4])).Store(out[outOff : outOff+4])
			acc1.Add(archsimd.LoadFloat32x4(out[outOff+outRowStride : outOff+outRowStride+4])).Store(out[outOff+outRowStride : outOff+outRowStride+4])
			acc2.Add(archsimd.LoadFloat32x4(out[outOff+2*outRowStride : outOff+2*outRowStride+4])).Store(out[outOff+2*outRowStride : outOff+2*outRowStride+4])
			acc3.Add(archsimd.LoadFloat32x4(out[outOff+3*outRowStride : outOff+3*outRowStride+4])).Store(out[outOff+3*outRowStride : outOff+3*outRowStride+4])
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
			acc0.Add(archsimd.LoadFloat32x4(out[outOff : outOff+4])).Store(out[outOff : outOff+4])
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
			out[outBaseIdx+r*outRowStride+col] += sum
		}
	}
}

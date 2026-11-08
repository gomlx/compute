// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"simd/archsimd"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
)

// ConvAVX512Kernel8x32Float32Asm computes an 8x32 float32 output tile directly from
// 8 rows of LHS ([8, K] with row byte stride lhsRowStrideBytes) and 32 columns of
// packed RHS ([K, 32] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvAVX512Kernel8x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX512Kernel4x32Float32Asm computes a 4x32 float32 output tile directly from
// 4 rows of LHS ([4, K] with row byte stride lhsRowStrideBytes) and 32 columns of
// packed RHS ([K, 32] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvAVX512Kernel4x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX512Direct2DKernel8x32Float32Asm computes an 8x32 float32 output tile for
// 8 horizontally adjacent interior 2D output pixels directly from the input tensor
// without im2col patch extraction, iterating across numKy valid filter rows of
// width kWC = kW * inC each and advancing LHS row pointers by lhsKyJumpBytes
// between filter rows.
//
//go:noescape
func ConvAVX512Direct2DKernel8x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX512Direct2DKernel4x32Float32Asm computes a 4x32 float32 output tile for
// 4 horizontally adjacent interior 2D output pixels directly from the input tensor
// without im2col patch extraction, iterating across numKy valid filter rows of
// width kWC = kW * inC each and advancing LHS row pointers by lhsKyJumpBytes
// between filter rows.
//
//go:noescape
func ConvAVX512Direct2DKernel4x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX512Direct2DKernel2x32Float32Asm computes a 2x32 float32 output tile for
// 2 horizontally adjacent interior 2D output pixels directly from the input tensor
// without im2col patch extraction.
//
//go:noescape
func ConvAVX512Direct2DKernel2x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvAVX512Direct2DKernel1x32Float32Asm computes a 1x32 float32 output row for a
// single 2D output pixel (interior or clipped border) directly from the input tensor
// without im2col patch extraction, advancing lhsPtr by lhsKyJumpBytes and rhsPtr by
// rhsKyJumpBytes between filter rows.
//
//go:noescape
func ConvAVX512Direct2DKernel1x32Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsKyJumpBytes, rhsKyJumpBytes, rhsStepBytes int,
)

// ConvKernelGradAVX512_8x32Float32Asm accumulates an 8x32 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 8x32 result to outPtr.
//
//go:noescape
func ConvKernelGradAVX512_8x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradAVX512_4x32Float32Asm accumulates a 4x32 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 4x32 result to outPtr.
//
//go:noescape
func ConvKernelGradAVX512_4x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradAVX512_1x32Float32Asm accumulates a 1x32 float32 kernel-gradient row
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 32-float row to outPtr.
//
//go:noescape
func ConvKernelGradAVX512_1x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep int,
)

func execConvGeneralAVX512(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}
	p := ops.GetConv2DParams(node)
	switch p.Kind {
	case ops.Conv2DStandard:
		return execConvStandardAVX512(backend, &p, input, kernel, node.Shape)
	case ops.Conv2DKernelGrad:
		return execConvKernelGradAVX512(backend, &p, input, kernel, node.Shape)
	default:
		return nil, gobackend.ErrFallback
	}
}

// execConvStandardAVX512 executes 1D and 2D ChannelsLast Forward and VJP-wrt-Input convolutions
// using the 8x32 and 4x32 AVX-512 microkernels (ConvAVX512Kernel8x32Float32Asm and
// ConvAVX512Kernel4x32Float32Asm) without LHS transposition.
func execConvStandardAVX512(
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

	const panelCols = 32
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
				var edgeOut [8 * 32]float32

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
						colStart := panelIdx * 32
						validCols := min(32, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKxStartOff])
						if validCols == 32 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX512Direct2DKernel1x32Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWCValid, numKy, lhsKyJump, rhsKyJump, 128,
							)
						} else {
							ConvAVX512Direct2DKernel1x32Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWCValid, numKy, lhsKyJump, rhsKyJump, 128,
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
				for ; outX+8 <= outXIntEnd; outX += 8 {
					inX0 := outX*p.StrideW - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
					for panelIdx := range numPanels {
						colStart := panelIdx * 32
						validCols := min(32, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOff])
						if validCols == 32 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX512Direct2DKernel8x32Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 128, outRowStrideBytes,
							)
						} else {
							ConvAVX512Direct2DKernel8x32Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 128, 128,
							)
							for ir := range 8 {
								dstOff := (rowMStart+outX+ir)*N + colStart
								copy(out[dstOff:dstOff+validCols], edgeOut[ir*32:ir*32+validCols])
							}
						}
					}
				}
				for ; outX+4 <= outXIntEnd; outX += 4 {
					inX0 := outX*p.StrideW - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
					for panelIdx := range numPanels {
						colStart := panelIdx * 32
						validCols := min(32, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOff])
						if validCols == 32 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX512Direct2DKernel4x32Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 128, outRowStrideBytes,
							)
						} else {
							ConvAVX512Direct2DKernel4x32Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 128, 128,
							)
							for ir := range 4 {
								dstOff := (rowMStart+outX+ir)*N + colStart
								copy(out[dstOff:dstOff+validCols], edgeOut[ir*32:ir*32+validCols])
							}
						}
					}
				}
				for ; outX+2 <= outXIntEnd; outX += 2 {
					inX0 := outX*p.StrideW - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
					for panelIdx := range numPanels {
						colStart := panelIdx * 32
						validCols := min(32, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOff])
						if validCols == 32 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX512Direct2DKernel2x32Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 128, outRowStrideBytes,
							)
						} else {
							ConvAVX512Direct2DKernel2x32Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directLhsRowStrideBytes, directLhsKyJumpBytes, 128, 128,
							)
							for ir := range 2 {
								dstOff := (rowMStart+outX+ir)*N + colStart
								copy(out[dstOff:dstOff+validCols], edgeOut[ir*32:ir*32+validCols])
							}
						}
					}
				}
				for ; outX < outXIntEnd; outX++ {
					inX0 := outX*p.StrideW - p.PadLeft
					lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
					for panelIdx := range numPanels {
						colStart := panelIdx * 32
						validCols := min(32, N-colStart)
						rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOff])
						if validCols == 32 {
							outPtr := unsafe.Pointer(&out[(rowMStart+outX)*N+colStart])
							ConvAVX512Direct2DKernel1x32Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directLhsKyJumpBytes, 0, 128,
							)
						} else {
							ConvAVX512Direct2DKernel1x32Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directLhsKyJumpBytes, 0, 128,
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
			if mCount%8 == 0 {
				lhsSlice = input[mStart*K : (mStart+mCount)*K]
			} else {
				copy(patchBuf[:mCount*K], input[mStart*K:(mStart+mCount)*K])
				lhsSlice = patchBuf
			}
		} else {
			ops.ExtractPatches2DFloat32(p, input, patchBuf, mStart, mCount)
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
					ConvAVX512Kernel8x32Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 128, outRowStrideBytes)
				} else if validRows == 4 && validCols == 32 {
					outPtr := unsafe.Pointer(&out[(mStart+r)*N+colStart])
					ConvAVX512Kernel4x32Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 128, outRowStrideBytes)
				} else if validRows <= 4 {
					ConvAVX512Kernel4x32Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 128)
					for ir := range validRows {
						dstOff := (mStart+r+ir)*N + colStart
						copy(out[dstOff:dstOff+validCols], edgeOut[ir*32:ir*32+validCols])
					}
				} else {
					ConvAVX512Kernel8x32Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 128)
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

// execConvKernelGradAVX512 computes VJP wrt Kernel (Conv2DKernelGrad) using AVX-512 outer-product
// microkernels (ConvKernelGradAVX512_8x32Float32Asm, 4x32, and 1x32) with zero packing and zero
// intermediate buffers. When the spatial/batch reduction domain (BatchSize * OutH) is large, it
// tiles across (b, outY) rows so each L2-resident slice of x and v is reused across all filter
// positions and channel tiles.
func execConvKernelGradAVX512(
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
	numColPanels := (outC + 31) / 32
	panelsPerFilter := numRowPanels * numColPanels
	baseTasks := kH * kW * panelsPerFilter
	totalBY := p.BatchSize * p.OutH

	maxParallelism := backend.Scheduler.AdjustedMaxParallelism()
	totalPixels := totalBY * p.OutW
	outSize := output.RawShape.Size()

	// Choose numSlices (power of 2: 1, 2, 4, 8) to split the (BatchSize, OutH) reduction
	// domain when baseTasks is small or unevenly distributed across maxParallelism,
	// while keeping the 8x32 microkernel (rowTile = 8) for high arithmetic intensity.
	numSlices := 1
	if maxParallelism > 1 && totalPixels >= 4096 && outSize <= 262144 {
		for numSlices < 8 && baseTasks*numSlices < maxParallelism*8 && totalPixels/(numSlices*2) >= 1536 {
			if numSlices*outSize > 524288 {
				break
			}
			nextSlices := numSlices * 2
			if p.BatchSize >= nextSlices || (nextSlices%p.BatchSize == 0 && p.OutH >= (nextSlices/p.BatchSize)*4) {
				numSlices = nextSlices
			} else {
				break
			}
		}
	}

	var scratchBuf *gobackend.Buffer
	var scratchOut []float32
	if numSlices > 1 {
		var err error
		scratchBuf, scratchOut, err = ops.GetScratchFloat32(backend, (numSlices-1)*outSize)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(scratchBuf)
		clear(scratchOut[:(numSlices-1)*outSize])
	}

	totalTasks := numSlices * baseTasks
	processTask := func(globalIdx int) {
		sliceIdx := globalIdx / baseTasks
		taskIdx := globalIdx % baseTasks
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

		var dstOut []float32
		if sliceIdx == 0 {
			dstOut = out
		} else {
			dstOut = scratchOut[(sliceIdx-1)*outSize : sliceIdx*outSize]
		}

		byStart := (sliceIdx * totalBY) / numSlices
		byEnd := ((sliceIdx + 1) * totalBY) / numSlices
		convKernelGradAVX512TileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, byStart, byEnd)
	}

	maxWorkers := min(totalTasks, maxParallelism)
	if maxWorkers <= 1 {
		for taskIdx := range totalTasks {
			processTask(taskIdx)
		}
	} else {
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
	}

	if numSlices > 1 {
		for s := range numSlices - 1 {
			src := scratchOut[s*outSize : (s+1)*outSize]
			i := 0
			for ; i+16 <= outSize; i += 16 {
				vo := archsimd.LoadFloat32x16(out[i : i+16])
				vs := archsimd.LoadFloat32x16(src[i : i+16])
				vo.Add(vs).Store(out[i : i+16])
			}
			for ; i < outSize; i++ {
				out[i] += src[i]
			}
		}
	}
	return output, nil
}

func convKernelGradAVX512TileFloat32(
	p *ops.Conv2DParams,
	x, v, out []float32,
	ky, kx, cInStart, validRows, cOutStart, validCols int,
	byStart, byEnd int,
) {
	outYStart, outYEnd, inYStart, outXStart, outXEnd, inXStart := p.KernelGradValidRange(ky, kx)
	numYValid := outYEnd - outYStart
	numXValid := outXEnd - outXStart
	if numYValid <= 0 || numXValid <= 0 || byStart >= byEnd {
		return
	}

	inC := p.InChannels
	outC := p.OutChannels
	xStepFloats := p.KerDilW * inC
	vStepFloats := outC
	xYStrideFloats := p.KerDilH * p.InW * inC
	vYStrideFloats := p.OutW * outC
	xBatchStrideFloats := p.InH * p.InW * inC
	vBatchStrideFloats := p.OutH * p.OutW * outC
	canCollapseX := p.KerDilH == 1 && p.KerDilW == 1 && outXStart == 0 && outXEnd == p.OutW && inXStart == 0 && p.InW == p.OutW

	runSubRange := func(bStart, bCount, yStart, yEnd int) {
		numY := yEnd - yStart
		if numY <= 0 || bCount <= 0 {
			return
		}
		batchSize := bCount
		numX := numXValid
		curInYStart := inYStart + (yStart-outYStart)*p.KerDilH
		if canCollapseX {
			numX = numY * p.OutW
			numY = 1
			if yStart == 0 && yEnd == p.OutH && curInYStart == 0 && p.InH == p.OutH {
				numX = batchSize * numX
				batchSize = 1
			}
		}

		colOff := 0
		for ; colOff+32 <= validCols; colOff += 32 {
			cOut := cOutStart + colOff
			vBaseIdx := bStart*vBatchStrideFloats + (yStart*p.OutW+outXStart)*outC + cOut

			rOff := 0
			for ; rOff+8 <= validRows; rOff += 8 {
				cIn := cInStart + rOff
				xBaseIdx := bStart*xBatchStrideFloats + (curInYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				ConvKernelGradAVX512_8x32Float32Asm(
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
				xBaseIdx := bStart*xBatchStrideFloats + (curInYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				ConvKernelGradAVX512_4x32Float32Asm(
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
				ConvKernelGradAVX512_1x32Float32Asm(
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
			cOut := cOutStart + colOff
			xBaseIdx := bStart*xBatchStrideFloats + (curInYStart*p.InW+inXStart)*inC + cInStart
			vBaseIdx := bStart*vBatchStrideFloats + (yStart*p.OutW+outXStart)*outC + cOut
			outBaseIdx := ((ky*p.KernelW+kx)*inC+cInStart)*outC + cOut
			convKernelGradTailAVX512Float32(
				x, v, out,
				batchSize, numY, numX,
				xBaseIdx, vBaseIdx, outBaseIdx,
				xBatchStrideFloats, vBatchStrideFloats,
				xYStrideFloats, vYStrideFloats,
				xStepFloats, vStepFloats,
				outC, validRows, validCols-colOff,
			)
		}
	}

	b0 := byStart / p.OutH
	y0 := byStart % p.OutH
	b1 := (byEnd - 1) / p.OutH
	y1 := (byEnd-1)%p.OutH + 1

	if b0 == b1 {
		runSubRange(b0, 1, max(outYStart, y0), min(outYEnd, y1))
		return
	}
	if y0 > 0 {
		runSubRange(b0, 1, max(outYStart, y0), outYEnd)
		b0++
	}
	fullBatches := b1 - b0
	if y1 == p.OutH {
		fullBatches++
	}
	if fullBatches > 0 {
		runSubRange(b0, fullBatches, outYStart, outYEnd)
	}
	if y1 < p.OutH {
		runSubRange(b1, 1, outYStart, min(outYEnd, y1))
	}
}

// convKernelGradTailAVX512Float32 handles remainder columns (validCols < 32) in Conv2DKernelGrad
// without reading past the end of x or v.
func convKernelGradTailAVX512Float32(
	x, v, out []float32,
	batchSize, numY, numX int,
	xBaseIdx, vBaseIdx, outBaseIdx int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStride int,
	validRows, validCols int,
) {
	col := 0
	for ; col+16 <= validCols; col += 16 {
		r := 0
		for ; r+4 <= validRows; r += 4 {
			acc0 := archsimd.BroadcastFloat32x16(0)
			acc1 := archsimd.BroadcastFloat32x16(0)
			acc2 := archsimd.BroadcastFloat32x16(0)
			acc3 := archsimd.BroadcastFloat32x16(0)

			xB := xBaseIdx + r
			vB := vBaseIdx + col
			for range batchSize {
				xY := xB
				vY := vB
				for range numY {
					xP := xY
					vP := vY
					for range numX {
						vv := archsimd.LoadFloat32x16(v[vP : vP+16])
						acc0 = vv.MulAdd(archsimd.BroadcastFloat32x16(x[xP+0]), acc0)
						acc1 = vv.MulAdd(archsimd.BroadcastFloat32x16(x[xP+1]), acc1)
						acc2 = vv.MulAdd(archsimd.BroadcastFloat32x16(x[xP+2]), acc2)
						acc3 = vv.MulAdd(archsimd.BroadcastFloat32x16(x[xP+3]), acc3)
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
			acc0.Add(archsimd.LoadFloat32x16(out[outOff : outOff+16])).Store(out[outOff : outOff+16])
			acc1.Add(archsimd.LoadFloat32x16(out[outOff+outRowStride : outOff+outRowStride+16])).Store(out[outOff+outRowStride : outOff+outRowStride+16])
			acc2.Add(archsimd.LoadFloat32x16(out[outOff+2*outRowStride : outOff+2*outRowStride+16])).Store(out[outOff+2*outRowStride : outOff+2*outRowStride+16])
			acc3.Add(archsimd.LoadFloat32x16(out[outOff+3*outRowStride : outOff+3*outRowStride+16])).Store(out[outOff+3*outRowStride : outOff+3*outRowStride+16])
		}
		for ; r < validRows; r++ {
			acc0 := archsimd.BroadcastFloat32x16(0)
			xB := xBaseIdx + r
			vB := vBaseIdx + col
			for range batchSize {
				xY := xB
				vY := vB
				for range numY {
					xP := xY
					vP := vY
					for range numX {
						vv := archsimd.LoadFloat32x16(v[vP : vP+16])
						acc0 = vv.MulAdd(archsimd.BroadcastFloat32x16(x[xP]), acc0)
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
			acc0.Add(archsimd.LoadFloat32x16(out[outOff : outOff+16])).Store(out[outOff : outOff+16])
		}
	}

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

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"runtime"
	"simd/archsimd"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/internal/gobackend/simdarch"
	"github.com/gomlx/compute/shapes"
)

// ConvNEONKernel4x16Float32Asm computes a 4x16 float32 output tile directly from
// 4 rows of LHS ([4, K] with row byte stride lhsRowStrideBytes) and 16 columns of
// packed RHS ([K, 16+] with step byte stride rhsStepBytes), storing to outPtr
// (row byte stride outRowStrideBytes).
//
//go:noescape
func ConvNEONKernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	K, lhsRowStrideBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvNEONDirect2DKernel4x16Float32Asm computes a 4x16 float32 output tile directly from
// 4 horizontally adjacent 2D input positions (with pixel byte stride lhsRowStrideBytes)
// across numKy valid filter rows of width kWC = kW * inC (jumping by lhsKyJumpBytes after
// each filter row), without intermediate im2col patch extraction.
//
//go:noescape
func ConvNEONDirect2DKernel4x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvNEONDirect2DKernel2x16Float32Asm computes a 2x16 float32 output tile directly from
// 2 horizontally adjacent 2D input positions without intermediate im2col patch extraction.
//
//go:noescape
func ConvNEONDirect2DKernel2x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsRowStrideBytes, lhsKyJumpBytes, rhsStepBytes, outRowStrideBytes int,
)

// ConvNEONDirect2DKernel1x16Float32Asm computes a 1x16 float32 output row for a single
// 2D output pixel (interior or clipped border) directly from the input tensor without
// intermediate im2col patch extraction, advancing lhsPtr by lhsKyJumpBytes and rhsPtr
// by rhsKyJumpBytes between filter rows.
//
//go:noescape
func ConvNEONDirect2DKernel1x16Float32Asm(
	lhsPtr, rhsPtr, outPtr unsafe.Pointer,
	kWC, numKy, lhsKyJumpBytes, rhsKyJumpBytes, rhsStepBytes int,
)

// ConvKernelGradNEON4x16Float32Asm accumulates a 4x16 float32 kernel-gradient tile
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 4x16 result to outPtr.
//
//go:noescape
func ConvKernelGradNEON4x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
)

// ConvKernelGradNEON1x16Float32Asm accumulates a 1x16 float32 kernel-gradient row
// across the 3D domain (batchSize, numY, numX) directly from xBase and vBase and
// stores the 16-float row to outPtr.
//
//go:noescape
func ConvKernelGradNEON1x16Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep int,
)

// ConvPackLHS8StridedFloat32Asm packs 8 rows of length K from srcPtr (with row byte stride
// srcRowStrideBytes) into an 8-column stripe at dstPtr (with panel row byte stride
// panelColStrideBytes, e.g., 128 bytes for a 32-row SME panel or 64 bytes for a 16-row panel).
//
//go:noescape
func ConvPackLHS8StridedFloat32Asm(
	srcPtr, dstPtr unsafe.Pointer,
	K, srcRowStrideBytes, panelColStrideBytes int,
)

// ConvSMEKernel32x32Float32Asm computes a single 32x32 float32 output tile (validRows <= 32,
// validCols <= 32) from packedLHS ([K, 32]) and packedRHS ([K, 32]) using 4 SME ZA tiles
// (ZA0..ZA3) and stores predicated rows/cols directly to outPtr.
//
//go:noescape
func ConvSMEKernel32x32Float32Asm(
	packedLHS, packedRHS, outPtr unsafe.Pointer,
	K, outRowStrideBytes, validRows, validCols int,
)

// ConvSMEKernel16x16Float32Asm computes a single 16x16 float32 output tile (validRows <= 16,
// validCols <= 16) from packedLHS ([K, 16]) and packedRHS ([K, 16]) using 1 SME ZA tile
// (ZA0) and stores predicated rows/cols directly to outPtr.
//
//go:noescape
func ConvSMEKernel16x16Float32Asm(
	packedLHS, packedRHS, outPtr unsafe.Pointer,
	K, outRowStrideBytes, validRows, validCols int,
)

// ConvKernelGradSME32x32Float32Asm accumulates a 32x32 float32 kernel-gradient tile
// (validRows <= 32, validCols <= 32) directly in SME ZA0..ZA3 across the 3D domain
// (batchSize, numY, numX) using predicated LD1W loads from xBase and vBase and stores
// predicated rows/cols to outPtr.
//
//go:noescape
func ConvKernelGradSME32x32Float32Asm(
	xBase, vBase, outPtr unsafe.Pointer,
	batchSize, numY, numX int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStrideBytes int,
	validRows, validCols, accumulate int,
)

func execConvGeneralNEON(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}
	p := ops.GetConv2DParams(node)
	switch p.Kind {
	case ops.Conv2DStandard:
		return execConvStandardNEON(backend, &p, input, kernel, node.Shape)
	case ops.Conv2DKernelGrad:
		return execConvKernelGradNEON(backend, &p, input, kernel, node.Shape)
	default:
		return nil, gobackend.ErrFallback
	}
}

func execConvGeneralSME(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}
	p := ops.GetConv2DParams(node)
	switch p.Kind {
	case ops.Conv2DStandard:
		return execConvStandardSME(backend, &p, input, kernel, node.Shape)
	case ops.Conv2DKernelGrad:
		return execConvKernelGradSME(backend, &p, input, kernel, node.Shape)
	default:
		return nil, gobackend.ErrFallback
	}
}

func addSliceFloat32NEON(dst, src []float32) {
	n := len(dst)
	i := 0
	for ; i+4 <= n; i += 4 {
		vd := archsimd.LoadFloat32x4(dst[i : i+4])
		vs := archsimd.LoadFloat32x4(src[i : i+4])
		vd.Add(vs).Store(dst[i : i+4])
	}
	for ; i < n; i++ {
		dst[i] += src[i]
	}
}

// execConvStandardNEON executes 1D and 2D ChannelsLast Forward and VJP-wrt-Input convolutions
// using the 4x16 NEON lane-indexed microkernel (ConvNEONKernel4x16Float32Asm) without LHS transposition.
func execConvStandardNEON(
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

	canDirect2D := !p.Is1x1Unit && p.InDilH == 1 && p.InDilW == 1 && p.KerDilW == 1
	outHW := p.OutH * p.OutW
	inHW := p.InH * p.InW
	kWC := p.KernelW * p.InChannels
	directStrideBytes := p.StrideW * p.InChannels * 4
	directKyJumpBytes := (p.KerDilH*p.InW - p.KernelW) * p.InChannels * 4

	// Direct zero-im2col 2D row fast path when InDilH == 1 && InDilW == 1 && KerDilW == 1.
	if canDirect2D {
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
		if totalRows >= 4 || M*K*N < 65536 {
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
							ConvNEONDirect2DKernel1x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWCValid, numKy, lhsKyJump, rhsKyJump, 64,
							)
						} else {
							ConvNEONDirect2DKernel1x16Float32Asm(
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
							ConvNEONDirect2DKernel4x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directStrideBytes, directKyJumpBytes, 64, outRowStrideBytes,
							)
						} else {
							ConvNEONDirect2DKernel4x16Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directStrideBytes, directKyJumpBytes, 64, 64,
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
							ConvNEONDirect2DKernel2x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directStrideBytes, directKyJumpBytes, 64, outRowStrideBytes,
							)
						} else {
							ConvNEONDirect2DKernel2x16Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directStrideBytes, directKyJumpBytes, 64, 64,
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
							ConvNEONDirect2DKernel1x16Float32Asm(
								lhsPtr, rhsPanelPtr, outPtr,
								kWC, numKy, directKyJumpBytes, 0, 64,
							)
						} else {
							ConvNEONDirect2DKernel1x16Float32Asm(
								lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]),
								kWC, numKy, directKyJumpBytes, 0, 64,
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

	const mTile = 32
	numMTiles := (M + mTile - 1) / mTile
	patchBufSize := 4 * K

	processMTile := func(mStart, mCount int, patchBuf []float32) {
		var edgeOut [4 * 16]float32
		outRowStrideBytes := N * 4

		for r := 0; r < mCount; r += 4 {
			m0 := mStart + r
			validRows := min(4, mCount-r)

			if canDirect2D && validRows == 4 && outHW > 0 {
				b := m0 / outHW
				rem := m0 % outHW
				outY := rem / p.OutW
				outX := rem % p.OutW
				if outX+4 <= p.OutW {
					inX0 := outX*p.StrideW - p.PadLeft
					if inX0 >= 0 && inX0+3*p.StrideW+p.KernelW <= p.InW {
						baseInY := outY*p.StrideH - p.PadTop
						kyStart := 0
						if baseInY < 0 {
							kyStart = (-baseInY + p.KerDilH - 1) / p.KerDilH
						}
						kyEnd := p.KernelH
						if maxInY := baseInY + (p.KernelH-1)*p.KerDilH; maxInY >= p.InH {
							kyEnd = (p.InH - 1 - baseInY) / p.KerDilH + 1
							if kyEnd < 0 {
								kyEnd = 0
							}
						}
						numKy := kyEnd - kyStart
						if numKy <= 0 {
							clear(out[m0*N : (m0+4)*N])
							continue
						}
						inYStart := baseInY + kyStart*p.KerDilH
						lhsPtr := unsafe.Pointer(&input[(b*inHW+inYStart*p.InW+inX0)*p.InChannels])
						rhsKyOffset := kyStart * kWC * 16

						for panelIdx := range numPanels {
							colStart := panelIdx * 16
							validCols := min(16, N-colStart)
							rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOffset])

							if validCols == 16 {
								outPtr := unsafe.Pointer(&out[m0*N+colStart])
								ConvNEONDirect2DKernel4x16Float32Asm(lhsPtr, rhsPanelPtr, outPtr, kWC, numKy, directStrideBytes, directKyJumpBytes, 64, outRowStrideBytes)
							} else {
								ConvNEONDirect2DKernel4x16Float32Asm(lhsPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), kWC, numKy, directStrideBytes, directKyJumpBytes, 64, 64)
								for ir := range 4 {
									dstOff := (m0+ir)*N + colStart
									copy(out[dstOff:dstOff+validCols], edgeOut[ir*16:ir*16+validCols])
								}
							}
						}
						continue
					}
				}
			}

			var lhsRowPtr unsafe.Pointer
			if p.Is1x1Unit {
				if validRows == 4 {
					lhsRowPtr = unsafe.Pointer(&input[m0*K])
				} else {
					copy(patchBuf[:validRows*K], input[m0*K:(m0+validRows)*K])
					lhsRowPtr = unsafe.Pointer(&patchBuf[0])
				}
			} else {
				ops.ExtractPatches2DFloat32(p, input, patchBuf, m0, validRows)
				lhsRowPtr = unsafe.Pointer(&patchBuf[0])
			}

			lhsRowStrideBytes := K * 4
			for panelIdx := range numPanels {
				colStart := panelIdx * 16
				validCols := min(16, N-colStart)
				rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize])

				if validRows == 4 && validCols == 16 {
					outPtr := unsafe.Pointer(&out[m0*N+colStart])
					ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 64, outRowStrideBytes)
				} else {
					ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 64, 64)
					for ir := range validRows {
						dstOff := (m0+ir)*N + colStart
						copy(out[dstOff:dstOff+validCols], edgeOut[ir*16:ir*16+validCols])
					}
				}
			}
		}
	}

	maxWorkers := min(numMTiles, backend.Scheduler.AdjustedMaxParallelism())
	if maxWorkers <= 1 || M*K*N < 16384 {
		var patchBuf []float32
		if !p.Is1x1Unit || M%4 != 0 {
			patchBufObj, bufSlice, err := ops.GetScratchFloat32(backend, patchBufSize)
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

	var nextTile atomic.Int64
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		var patchBuf []float32
		if !p.Is1x1Unit || M%4 != 0 {
			patchBufObj, bufSlice, err := ops.GetScratchFloat32(backend, patchBufSize)
			if err != nil {
				return
			}
			defer backend.PutBuffer(patchBufObj)
			patchBuf = bufSlice
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

// execConvStandardSME executes 1D and 2D ChannelsLast Forward and VJP-wrt-Input convolutions
// using ARM64 SME FMOPA outer-product tiles (32x32 or 16x16) co-scheduled with per-core
// NEON 4x16 microkernels across CPU clusters.
func execConvStandardSME(
	backend *gobackend.Backend,
	p *ops.Conv2DParams,
	inputBuf, kernelBuf *gobackend.Buffer,
	outputShape shapes.Shape,
) (*gobackend.Buffer, error) {
	M := p.BatchSize * p.OutH * p.OutW
	K := p.KernelH * p.KernelW * p.InChannels
	N := p.OutChannels
	if M == 0 || N == 0 {
		return backend.GetBuffer(outputShape)
	}
	if K == 0 {
		output, err := backend.GetBuffer(outputShape)
		if err != nil {
			return nil, err
		}
		output.Zeros()
		return output, nil
	}

	// For tiny pixel counts or very shallow contracting dimensions with a single output panel
	// (e.g. MNIST Conv1 where K=9, N=32), SME LHS transposition + streaming transition overhead
	// exceeds direct NEON execution.
	if (M < 8 && N < 8) || (K < 32 && N <= 32) {
		return execConvStandardNEON(backend, p, inputBuf, kernelBuf, outputShape)
	}

	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	input := inputBuf.Flat.([]float32)
	kernel := kernelBuf.Flat.([]float32)
	out := output.Flat.([]float32)

	// Fast 16x16 single-tile SME path when both M <= 16 and N <= 16.
	if M <= 16 && N <= 16 {
		const panelCols = 16
		packedRHSBuf, packedRHS, err := ops.GetScratchFloat32(backend, K*panelCols)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(packedRHSBuf)
		ops.PackConvKernelFloat32(p, kernel, packedRHS, panelCols)

		patchBufObj, patchBuf, err := ops.GetScratchFloat32(backend, 16*K)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(patchBufObj)
		if p.Is1x1Unit {
			copy(patchBuf[:M*K], input[:M*K])
		} else {
			ops.ExtractPatches2DFloat32(p, input, patchBuf, 0, M)
		}

		packedLHSBuf, packedLHS, err := ops.GetScratchFloat32(backend, K*16)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(packedLHSBuf)

		ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&patchBuf[0]), unsafe.Pointer(&packedLHS[0]), K, K*4, 64)
		if M > 8 {
			ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&patchBuf[8*K]), unsafe.Pointer(&packedLHS[8]), K, K*4, 64)
		}
		ConvSMEKernel16x16Float32Asm(
			unsafe.Pointer(&packedLHS[0]),
			unsafe.Pointer(&packedRHS[0]),
			unsafe.Pointer(&out[0]),
			K, N*4, M, N,
		)
		return output, nil
	}

	// 32x32 SME + Hybrid NEON co-execution path.
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

	const mTile = 32
	numMTiles := (M + mTile - 1) / mTile

	canDirect2D := !p.Is1x1Unit && p.InDilH == 1 && p.InDilW == 1 && p.KerDilW == 1
	outHW := p.OutH * p.OutW
	inHW := p.InH * p.InW
	kWC := p.KernelW * p.InChannels
	directStrideBytes := p.StrideW * p.InChannels * 4
	directKyJumpBytes := (p.KerDilH*p.InW - p.KernelW) * p.InChannels * 4

	processMTileSME := func(mStart, mCount int, patchBuf, packedLHS []float32) {
		if canDirect2D && outHW > 0 {
			b0 := mStart / outHW
			rem0 := mStart % outHW
			outY0 := rem0 / p.OutW
			outX0 := rem0 % p.OutW
			if outX0+mCount <= p.OutW {
				baseInY := outY0*p.StrideH - p.PadTop
				kyStart := 0
				if baseInY < 0 {
					kyStart = (-baseInY + p.KerDilH - 1) / p.KerDilH
				}
				kyEnd := p.KernelH
				if maxInY := baseInY + (p.KernelH-1)*p.KerDilH; maxInY >= p.InH {
					kyEnd = (p.InH - 1 - baseInY) / p.KerDilH + 1
					if kyEnd < 0 {
						kyEnd = 0
					}
				}
				numKy := kyEnd - kyStart
				if numKy <= 0 {
					clear(out[mStart*N : (mStart+mCount)*N])
					return
				}
				kValid := numKy * kWC
				for s := 0; s < mCount; s += 8 {
					validStripe := min(8, mCount-s)
					stripeOutX := outX0 + s
					inX0 := stripeOutX*p.StrideW - p.PadLeft
					if validStripe == 8 {
						if inX0 >= 0 && inX0+7*p.StrideW+p.KernelW <= p.InW {
							for ky := kyStart; ky < kyEnd; ky++ {
								inY := baseInY + ky*p.KerDilH
								inIdx := (b0*inHW + inY*p.InW + inX0) * p.InChannels
								dstIdx := (ky-kyStart)*kWC*32 + s
								ConvPackLHS8StridedFloat32Asm(
									unsafe.Pointer(&input[inIdx]),
									unsafe.Pointer(&packedLHS[dstIdx]),
									kWC, directStrideBytes, 128,
								)
							}
						} else {
							kxValidStart := max(0, -inX0)
							kxValidEnd := min(p.KernelW, p.InW-(inX0+7*p.StrideW))
							inCStrideBytes := p.InChannels * 4
							for ky := kyStart; ky < kyEnd; ky++ {
								inY := baseInY + ky*p.KerDilH
								rowBase := (b0*inHW + inY*p.InW) * p.InChannels
								kyDstBase := (ky-kyStart)*kWC*32 + s
								if kxValidStart < kxValidEnd {
									spanFloats := (kxValidEnd - kxValidStart) * p.InChannels
									inIdx := rowBase + (inX0+kxValidStart)*p.InChannels
									dstIdx := kyDstBase + kxValidStart*p.InChannels*32
									ConvPackLHS8StridedFloat32Asm(
										unsafe.Pointer(&input[inIdx]),
										unsafe.Pointer(&packedLHS[dstIdx]),
										spanFloats, directStrideBytes, 128,
									)
								}
								for kx := range p.KernelW {
									if kxValidStart < kxValidEnd && kx >= kxValidStart && kx < kxValidEnd {
										continue
									}
									for pIdx := range 8 {
										dstOff := pIdx * p.InChannels
										inX := inX0 + pIdx*p.StrideW + kx
										if uint(inX) < uint(p.InW) {
											srcOff := rowBase + inX*p.InChannels
											copy(patchBuf[dstOff:dstOff+p.InChannels], input[srcOff:srcOff+p.InChannels])
										} else {
											clear(patchBuf[dstOff : dstOff+p.InChannels])
										}
									}
									dstIdx := kyDstBase + kx*p.InChannels*32
									ConvPackLHS8StridedFloat32Asm(
										unsafe.Pointer(&patchBuf[0]),
										unsafe.Pointer(&packedLHS[dstIdx]),
										p.InChannels, inCStrideBytes, 128,
									)
								}
							}
						}
					} else {
						ops.ExtractPatches2DFloat32(p, input, patchBuf, mStart+s, validStripe)
						ConvPackLHS8StridedFloat32Asm(
							unsafe.Pointer(&patchBuf[kyStart*kWC]),
							unsafe.Pointer(&packedLHS[s]),
							kValid, K*4, 128,
						)
					}
				}

				packedLHSPtr := unsafe.Pointer(&packedLHS[0])
				outRowStrideBytes := N * 4
				rhsKyOffset := kyStart * kWC * 32
				for panelIdx := range numPanels {
					colStart := panelIdx * 32
					validCols := min(32, N-colStart)
					rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize+rhsKyOffset])
					outPtr := unsafe.Pointer(&out[mStart*N+colStart])
					ConvSMEKernel32x32Float32Asm(packedLHSPtr, rhsPanelPtr, outPtr, kValid, outRowStrideBytes, mCount, validCols)
				}
				return
			}
		}

		var lhsSlice []float32
		if p.Is1x1Unit && mCount == 32 {
			lhsSlice = input[mStart*K : (mStart+32)*K]
		} else if p.Is1x1Unit {
			copy(patchBuf[:mCount*K], input[mStart*K:(mStart+mCount)*K])
			lhsSlice = patchBuf
		} else {
			ops.ExtractPatches2DFloat32(p, input, patchBuf, mStart, mCount)
			lhsSlice = patchBuf
		}

		lhsRowStrideBytes := K * 4
		ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[0]), unsafe.Pointer(&packedLHS[0]), K, lhsRowStrideBytes, 128)
		if mCount > 8 {
			ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[8*K]), unsafe.Pointer(&packedLHS[8]), K, lhsRowStrideBytes, 128)
		}
		if mCount > 16 {
			ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[16*K]), unsafe.Pointer(&packedLHS[16]), K, lhsRowStrideBytes, 128)
		}
		if mCount > 24 {
			ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[24*K]), unsafe.Pointer(&packedLHS[24]), K, lhsRowStrideBytes, 128)
		}

		packedLHSPtr := unsafe.Pointer(&packedLHS[0])
		outRowStrideBytes := N * 4
		for panelIdx := range numPanels {
			colStart := panelIdx * 32
			validCols := min(32, N-colStart)
			rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize])
			outPtr := unsafe.Pointer(&out[mStart*N+colStart])
			ConvSMEKernel32x32Float32Asm(packedLHSPtr, rhsPanelPtr, outPtr, K, outRowStrideBytes, mCount, validCols)
		}
	}

	processMTileNEONFromSME := func(mStart, mCount int, patchBuf []float32) {
		var edgeOut [4 * 16]float32
		outRowStrideBytes := N * 4

		for r := 0; r < mCount; r += 4 {
			m0 := mStart + r
			validRows := min(4, mCount-r)

			if canDirect2D && outHW > 0 {
				b := m0 / outHW
				rem := m0 % outHW
				outY := rem / p.OutW
				outX := rem % p.OutW
				if outX+validRows <= p.OutW {
					baseInY := outY*p.StrideH - p.PadTop
					kyStart := 0
					if baseInY < 0 {
						kyStart = (-baseInY + p.KerDilH - 1) / p.KerDilH
					}
					kyEnd := p.KernelH
					if maxInY := baseInY + (p.KernelH-1)*p.KerDilH; maxInY >= p.InH {
						kyEnd = (p.InH - 1 - baseInY) / p.KerDilH + 1
						if kyEnd < 0 {
							kyEnd = 0
						}
					}
					numKy := kyEnd - kyStart
					if numKy <= 0 {
						clear(out[m0*N : (m0+validRows)*N])
						continue
					}
					inYStart := baseInY + kyStart*p.KerDilH
					inYBase := (b*inHW + inYStart*p.InW) * p.InChannels
					rhsKyOffset := kyStart * kWC * 32
					inX0 := outX*p.StrideW - p.PadLeft

					if validRows == 4 && inX0 >= 0 && inX0+3*p.StrideW+p.KernelW <= p.InW {
						lhsPtr := unsafe.Pointer(&input[inYBase+inX0*p.InChannels])
						for panelIdx := range numPanels {
							colStart := panelIdx * 32
							validCols := min(32, N-colStart)
							panelBase := panelIdx*panelSize + rhsKyOffset

							subCols0 := min(16, validCols)
							rhsPtr0 := unsafe.Pointer(&packedRHS[panelBase])
							if subCols0 == 16 {
								outPtr0 := unsafe.Pointer(&out[m0*N+colStart])
								ConvNEONDirect2DKernel4x16Float32Asm(lhsPtr, rhsPtr0, outPtr0, kWC, numKy, directStrideBytes, directKyJumpBytes, 128, outRowStrideBytes)
							} else {
								ConvNEONDirect2DKernel4x16Float32Asm(lhsPtr, rhsPtr0, unsafe.Pointer(&edgeOut[0]), kWC, numKy, directStrideBytes, directKyJumpBytes, 128, 64)
								for ir := range 4 {
									dstOff := (m0+ir)*N + colStart
									copy(out[dstOff:dstOff+subCols0], edgeOut[ir*16:ir*16+subCols0])
								}
							}

							if validCols > 16 {
								subCols1 := validCols - 16
								rhsPtr1 := unsafe.Pointer(&packedRHS[panelBase+16])
								if subCols1 == 16 {
									outPtr1 := unsafe.Pointer(&out[m0*N+colStart+16])
									ConvNEONDirect2DKernel4x16Float32Asm(lhsPtr, rhsPtr1, outPtr1, kWC, numKy, directStrideBytes, directKyJumpBytes, 128, outRowStrideBytes)
								} else {
									ConvNEONDirect2DKernel4x16Float32Asm(lhsPtr, rhsPtr1, unsafe.Pointer(&edgeOut[0]), kWC, numKy, directStrideBytes, directKyJumpBytes, 128, 64)
									for ir := range 4 {
										dstOff := (m0+ir)*N + colStart + 16
										copy(out[dstOff:dstOff+subCols1], edgeOut[ir*16:ir*16+subCols1])
									}
								}
							}
						}
						continue
					}

					// Border or partial (<4) group on the same outY row: dispatch 2x16 / 1x16 directly without im2col.
					for ir := 0; ir < validRows; {
						curOutX := outX + ir
						curM := m0 + ir
						curInX0 := curOutX*p.StrideW - p.PadLeft
						if ir+2 <= validRows && curInX0 >= 0 && curInX0+p.StrideW+p.KernelW <= p.InW {
							lhsPtr := unsafe.Pointer(&input[inYBase+curInX0*p.InChannels])
							for panelIdx := range numPanels {
								colStart := panelIdx * 32
								validCols := min(32, N-colStart)
								panelBase := panelIdx*panelSize + rhsKyOffset

								subCols0 := min(16, validCols)
								rhsPtr0 := unsafe.Pointer(&packedRHS[panelBase])
								if subCols0 == 16 {
									outPtr0 := unsafe.Pointer(&out[curM*N+colStart])
									ConvNEONDirect2DKernel2x16Float32Asm(lhsPtr, rhsPtr0, outPtr0, kWC, numKy, directStrideBytes, directKyJumpBytes, 128, outRowStrideBytes)
								} else {
									ConvNEONDirect2DKernel2x16Float32Asm(lhsPtr, rhsPtr0, unsafe.Pointer(&edgeOut[0]), kWC, numKy, directStrideBytes, directKyJumpBytes, 128, 64)
									for rr := range 2 {
										dstOff := (curM+rr)*N + colStart
										copy(out[dstOff:dstOff+subCols0], edgeOut[rr*16:rr*16+subCols0])
									}
								}

								if validCols > 16 {
									subCols1 := validCols - 16
									rhsPtr1 := unsafe.Pointer(&packedRHS[panelBase+16])
									if subCols1 == 16 {
										outPtr1 := unsafe.Pointer(&out[curM*N+colStart+16])
										ConvNEONDirect2DKernel2x16Float32Asm(lhsPtr, rhsPtr1, outPtr1, kWC, numKy, directStrideBytes, directKyJumpBytes, 128, outRowStrideBytes)
									} else {
										ConvNEONDirect2DKernel2x16Float32Asm(lhsPtr, rhsPtr1, unsafe.Pointer(&edgeOut[0]), kWC, numKy, directStrideBytes, directKyJumpBytes, 128, 64)
										for rr := range 2 {
											dstOff := (curM+rr)*N + colStart + 16
											copy(out[dstOff:dstOff+subCols1], edgeOut[rr*16:rr*16+subCols1])
										}
									}
								}
							}
							ir += 2
						} else {
							kxStart := max(0, p.PadLeft-curOutX*p.StrideW)
							kxEnd := min(p.KernelW, p.InW+p.PadLeft-curOutX*p.StrideW)
							numKx := kxEnd - kxStart
							if numKx <= 0 {
								clear(out[curM*N : (curM+1)*N])
							} else {
								inXStart := curOutX*p.StrideW + kxStart - p.PadLeft
								lhsPtr := unsafe.Pointer(&input[inYBase+inXStart*p.InChannels])
								kWCValid := numKx * p.InChannels
								lhsKyJump := (p.KerDilH*p.InW - numKx) * p.InChannels * 4
								rhsKyJump := (p.KernelW - numKx) * p.InChannels * 32 * 4
								rhsKxStartOff := (kyStart*p.KernelW + kxStart) * p.InChannels * 32
								for panelIdx := range numPanels {
									colStart := panelIdx * 32
									validCols := min(32, N-colStart)
									panelBase := panelIdx*panelSize + rhsKxStartOff

									subCols0 := min(16, validCols)
									rhsPtr0 := unsafe.Pointer(&packedRHS[panelBase])
									if subCols0 == 16 {
										outPtr0 := unsafe.Pointer(&out[curM*N+colStart])
										ConvNEONDirect2DKernel1x16Float32Asm(lhsPtr, rhsPtr0, outPtr0, kWCValid, numKy, lhsKyJump, rhsKyJump, 128)
									} else {
										ConvNEONDirect2DKernel1x16Float32Asm(lhsPtr, rhsPtr0, unsafe.Pointer(&edgeOut[0]), kWCValid, numKy, lhsKyJump, rhsKyJump, 128)
										dstOff := curM*N + colStart
										copy(out[dstOff:dstOff+subCols0], edgeOut[:subCols0])
									}

									if validCols > 16 {
										subCols1 := validCols - 16
										rhsPtr1 := unsafe.Pointer(&packedRHS[panelBase+16])
										if subCols1 == 16 {
											outPtr1 := unsafe.Pointer(&out[curM*N+colStart+16])
											ConvNEONDirect2DKernel1x16Float32Asm(lhsPtr, rhsPtr1, outPtr1, kWCValid, numKy, lhsKyJump, rhsKyJump, 128)
										} else {
											ConvNEONDirect2DKernel1x16Float32Asm(lhsPtr, rhsPtr1, unsafe.Pointer(&edgeOut[0]), kWCValid, numKy, lhsKyJump, rhsKyJump, 128)
											dstOff := curM*N + colStart + 16
											copy(out[dstOff:dstOff+subCols1], edgeOut[:subCols1])
										}
									}
								}
							}
							ir++
						}
					}
					continue
				}
			}

			var lhsRowPtr unsafe.Pointer
			if p.Is1x1Unit {
				if validRows == 4 {
					lhsRowPtr = unsafe.Pointer(&input[m0*K])
				} else {
					copy(patchBuf[:validRows*K], input[m0*K:(m0+validRows)*K])
					lhsRowPtr = unsafe.Pointer(&patchBuf[0])
				}
			} else {
				ops.ExtractPatches2DFloat32(p, input, patchBuf, m0, validRows)
				lhsRowPtr = unsafe.Pointer(&patchBuf[0])
			}

			lhsRowStrideBytes := K * 4
			for panelIdx := range numPanels {
				colStart := panelIdx * 32
				validCols := min(32, N-colStart)
				panelBase := panelIdx * panelSize

				// Left 16 columns of the 32-wide RHS panel (stride = 128 bytes per K step).
				subCols0 := min(16, validCols)
				rhsPtr0 := unsafe.Pointer(&packedRHS[panelBase])
				if validRows == 4 && subCols0 == 16 {
					outPtr0 := unsafe.Pointer(&out[m0*N+colStart])
					ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr0, outPtr0, K, lhsRowStrideBytes, 128, outRowStrideBytes)
				} else {
					ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr0, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 64)
					for ir := range validRows {
						dstOff := (m0+ir)*N + colStart
						copy(out[dstOff:dstOff+subCols0], edgeOut[ir*16:ir*16+subCols0])
					}
				}

				// Right 16 columns of the 32-wide RHS panel.
				if validCols > 16 {
					subCols1 := validCols - 16
					rhsPtr1 := unsafe.Pointer(&packedRHS[panelBase+16])
					if validRows == 4 && subCols1 == 16 {
						outPtr1 := unsafe.Pointer(&out[m0*N+colStart+16])
						ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr1, outPtr1, K, lhsRowStrideBytes, 128, outRowStrideBytes)
					} else {
						ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr1, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 64)
						for ir := range validRows {
							dstOff := (m0+ir)*N + colStart + 16
							copy(out[dstOff:dstOff+subCols1], edgeOut[ir*16:ir*16+subCols1])
						}
					}
				}
			}
		}
	}

	maxWorkers := min(numMTiles, backend.Scheduler.AdjustedMaxParallelism())
	if maxWorkers <= 1 || M*K*N < 16384 {
		patchBufObj, patchBuf, err := ops.GetScratchFloat32(backend, mTile*K)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(patchBufObj)

		packedLHSBuf, packedLHS, err := ops.GetScratchFloat32(backend, K*32)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(packedLHSBuf)

		for tileIdx := range numMTiles {
			mStart := tileIdx * mTile
			mCount := min(mTile, M-mStart)
			processMTileSME(mStart, mCount, patchBuf, packedLHS)
		}
		return output, nil
	}

	var nextTile atomic.Int64
	var activeSME atomic.Int32
	var activeFastSME atomic.Int32
	neonNeedsPatchBuf := !(canDirect2D && outHW > 0 && p.OutW%4 == 0 && M%4 == 0) && !(p.Is1x1Unit && M%4 == 0)
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		clusterID, useSME, isEfficiencyCore := simdarch.AcquireSMEWithLimit(2, 1)
		if useSME {
			activeSME.Add(1)
			if !isEfficiencyCore {
				activeFastSME.Add(1)
			}
			defer func() {
				if !isEfficiencyCore {
					activeFastSME.Add(-1)
				}
				activeSME.Add(-1)
				simdarch.ReleaseSME(clusterID)
			}()
		} else if isEfficiencyCore {
			// Efficiency cluster SME is already saturated; avoid slow E-core NEON tail stalls.
			return
		}

		var patchBuf []float32
		if useSME || neonNeedsPatchBuf {
			patchBufObj, bufSlice, err := ops.GetScratchFloat32(backend, mTile*K)
			if err != nil {
				return
			}
			defer backend.PutBuffer(patchBufObj)
			patchBuf = bufSlice
		}

		var packedLHS []float32
		if useSME {
			packedLHSBuf, bufSlice, err := ops.GetScratchFloat32(backend, K*32)
			if err != nil {
				return
			}
			defer backend.PutBuffer(packedLHSBuf)
			packedLHS = bufSlice
		}

		for {
			if useSME {
				if isEfficiencyCore {
					rem := numMTiles - int(nextTile.Load())
					fastSME := int(activeFastSME.Load())
					if fastSME > 0 && rem <= 2*fastSME {
						break
					}
				}
			} else {
				rem := numMTiles - int(nextTile.Load())
				smeCount := int(activeFastSME.Load())
				if smeCount > 0 && rem <= 2*smeCount {
					break
				}
			}
			tileIdx := int(nextTile.Add(1) - 1)
			if tileIdx >= numMTiles {
				break
			}
			mStart := tileIdx * mTile
			mCount := min(mTile, M-mStart)
			if useSME {
				processMTileSME(mStart, mCount, patchBuf, packedLHS)
			} else {
				processMTileNEONFromSME(mStart, mCount, patchBuf)
			}
		}
	})
	return output, nil
}

// execConvKernelGradNEON computes VJP wrt Kernel (Conv2DKernelGrad) using NEON outer-product
// microkernels (ConvKernelGradNEON4x16Float32Asm and ConvKernelGradNEON1x16Float32Asm) with
// zero packing, 2D (Cin x Cout) panel tiling, and unified (BatchSize, OutH) reduction slicing.
func execConvKernelGradNEON(
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
	baseTasks := kH * kW * panelsPerFilter

	maxParallelism := backend.Scheduler.AdjustedMaxParallelism()
	totalOutRows := p.BatchSize * p.OutH
	totalPixels := totalOutRows * p.OutW
	numSlices := 1
	if maxParallelism > 1 && totalOutRows >= 4 && totalPixels >= 16384 {
		switch {
		case totalPixels >= 131072 && totalOutRows >= 16:
			numSlices = 8
		case totalPixels >= 65536 && totalOutRows >= 8:
			numSlices = 4
		default:
			numSlices = 2
		}
		for numSlices > 1 && totalOutRows/numSlices < 2 {
			numSlices /= 2
		}
	}

	outSize := len(out)
	partialSlices := make([][]float32, numSlices)
	partialSlices[0] = out
	if numSlices > 1 {
		partialBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, (numSlices-1)*outSize))
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		partialBuf.Zeros()
		defer backend.PutBuffer(partialBuf)
		partialFlat := partialBuf.Flat.([]float32)
		for s := 1; s < numSlices; s++ {
			partialSlices[s] = partialFlat[(s-1)*outSize : s*outSize]
		}
	}

	rowsPerSlice := (totalOutRows + numSlices - 1) / numSlices
	totalTasks := numSlices * baseTasks

	runRange := func(dstOut []float32, ky, kx, cInStart, validRows, cOutStart, validCols, rStart, rEnd int) {
		bFirst := rStart / p.OutH
		yFirst := rStart % p.OutH
		bLast := (rEnd - 1) / p.OutH
		yLast := (rEnd-1)%p.OutH + 1

		if bFirst == bLast {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bFirst, 1, yFirst, yLast)
			return
		}
		bCur := bFirst
		if yFirst > 0 {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bFirst, 1, yFirst, p.OutH)
			bCur++
		}
		fullBatches := bLast - bCur
		if yLast == p.OutH {
			fullBatches++
		}
		if fullBatches > 0 {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bCur, fullBatches, 0, p.OutH)
		}
		if yLast < p.OutH {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bLast, 1, 0, yLast)
		}
	}

	processTask := func(taskIdx int) {
		sliceIdx := taskIdx / baseTasks
		rStart := sliceIdx * rowsPerSlice
		if rStart >= totalOutRows {
			return
		}
		rEnd := min(totalOutRows, rStart+rowsPerSlice)

		baseIdx := taskIdx % baseTasks
		spatialIdx := baseIdx / panelsPerFilter
		rem := baseIdx % panelsPerFilter
		rowPanel := rem / numColPanels
		colPanel := rem % numColPanels
		ky := spatialIdx / kW
		kx := spatialIdx % kW
		cInStart := rowPanel * rowTile
		validRows := min(rowTile, inC-cInStart)
		cOutStart := colPanel * 16
		validCols := min(16, outC-cOutStart)
		runRange(partialSlices[sliceIdx], ky, kx, cInStart, validRows, cOutStart, validCols, rStart, rEnd)
	}

	maxWorkers := min(totalTasks, maxParallelism)
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

	for s := 1; s < numSlices; s++ {
		addSliceFloat32NEON(out, partialSlices[s])
	}
	return output, nil
}

// execConvKernelGradSME computes VJP wrt Kernel (Conv2DKernelGrad) using 32x32 SME FMOPA
// outer-product microkernels (ConvKernelGradSME32x32Float32Asm) co-scheduled with per-core
// NEON 4x16/1x16 microkernels across CPU clusters.
func execConvKernelGradSME(
	backend *gobackend.Backend,
	p *ops.Conv2DParams,
	inputBuf, kernelBuf *gobackend.Buffer,
	outputShape shapes.Shape,
) (*gobackend.Buffer, error) {
	// When InChannels < 4 or OutChannels < 8, a 32x32 SME tile would leave most ZA rows/cols unused;
	// running NEON 4x16 / 1x16 across all cores is faster and avoids SME cluster lock contention.
	if p.InChannels < 4 || p.OutChannels < 8 {
		return execConvKernelGradNEON(backend, p, inputBuf, kernelBuf, outputShape)
	}

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
	numRowPanels := (inC + 31) / 32
	numColPanels := (outC + 31) / 32
	panelsPerFilter := numRowPanels * numColPanels
	baseTasks := kH * kW * panelsPerFilter

	maxParallelism := backend.Scheduler.AdjustedMaxParallelism()
	totalOutRows := p.BatchSize * p.OutH
	totalPixels := totalOutRows * p.OutW
	numSlices := 1
	if maxParallelism > 1 && totalOutRows >= 16 && totalPixels >= 32768 {
		switch {
		case baseTasks < 18 && totalPixels >= 65536 && totalOutRows >= 32:
			numSlices = 4
		case baseTasks < 36 || totalPixels >= 131072:
			numSlices = 2
		}
	}

	outSize := len(out)
	partialSlices := make([][]float32, numSlices)
	partialSlices[0] = out
	if numSlices > 1 {
		partialBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, (numSlices-1)*outSize))
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		partialBuf.Zeros()
		defer backend.PutBuffer(partialBuf)
		partialFlat := partialBuf.Flat.([]float32)
		for s := 1; s < numSlices; s++ {
			partialSlices[s] = partialFlat[(s-1)*outSize : s*outSize]
		}
	}

	rowsPerSlice := (totalOutRows + numSlices - 1) / numSlices
	totalTasks := numSlices * baseTasks

	runSubRangeSME := func(dstOut []float32, ky, kx, cInStart, validRows, cOutStart, validCols, bStart, bCount, ySliceStart, ySliceEnd int, acc *int) {
		outYStart, outYEnd, inYStart, outXStart, outXEnd, inXStart := p.KernelGradValidRange(ky, kx)
		if ySliceStart > outYStart {
			diff := ySliceStart - outYStart
			outYStart = ySliceStart
			inYStart += diff * p.KerDilH
		}
		if ySliceEnd < outYEnd {
			outYEnd = ySliceEnd
		}
		numY := outYEnd - outYStart
		numX := outXEnd - outXStart
		if numY <= 0 || numX <= 0 || bCount <= 0 {
			return
		}

		batchSize := bCount
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

		xBaseIdx := bStart*xBatchStrideFloats + (inYStart*p.InW+inXStart)*inC + cInStart
		vBaseIdx := bStart*vBatchStrideFloats + (outYStart*p.OutW+outXStart)*outC + cOutStart
		outBaseIdx := ((ky*kW+kx)*inC+cInStart)*outC + cOutStart

		// Chunk large contracting domains into <= 4096 pixels per SME call so async SIGURG
		// preemption signals never cause long retry loops while minimizing SMSTART/SMSTOP transitions.
		const maxPixelsPerChunk = 4096
		chunkPixels := batchSize * numY * numX
		if chunkPixels <= maxPixelsPerChunk {
			ConvKernelGradSME32x32Float32Asm(
				unsafe.Pointer(&x[xBaseIdx]),
				unsafe.Pointer(&v[vBaseIdx]),
				unsafe.Pointer(&dstOut[outBaseIdx]),
				batchSize, numY, numX,
				xBatchStrideFloats*4, vBatchStrideFloats*4,
				xYStrideFloats*4, vYStrideFloats*4,
				xStepFloats*4, vStepFloats*4,
				outC*4,
				validRows, validCols, *acc,
			)
			*acc = 1
			return
		}

		if numY == 1 && batchSize == 1 {
			for xOff := 0; xOff < numX; xOff += maxPixelsPerChunk {
				chunkX := min(maxPixelsPerChunk, numX-xOff)
				ConvKernelGradSME32x32Float32Asm(
					unsafe.Pointer(&x[xBaseIdx+xOff*xStepFloats]),
					unsafe.Pointer(&v[vBaseIdx+xOff*vStepFloats]),
					unsafe.Pointer(&dstOut[outBaseIdx]),
					1, 1, chunkX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
					outC*4,
					validRows, validCols, *acc,
				)
				*acc = 1
			}
			return
		}

		pixelsPerBatch := numY * numX
		if pixelsPerBatch <= maxPixelsPerChunk {
			batchChunk := max(1, maxPixelsPerChunk/pixelsPerBatch)
			for bOff := 0; bOff < batchSize; bOff += batchChunk {
				bc := min(batchChunk, batchSize-bOff)
				ConvKernelGradSME32x32Float32Asm(
					unsafe.Pointer(&x[xBaseIdx+bOff*xBatchStrideFloats]),
					unsafe.Pointer(&v[vBaseIdx+bOff*vBatchStrideFloats]),
					unsafe.Pointer(&dstOut[outBaseIdx]),
					bc, numY, numX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
					outC*4,
					validRows, validCols, *acc,
				)
				*acc = 1
			}
			return
		}

		if numY == 1 {
			for b := range batchSize {
				xB := xBaseIdx + b*xBatchStrideFloats
				vB := vBaseIdx + b*vBatchStrideFloats
				for xOff := 0; xOff < numX; xOff += maxPixelsPerChunk {
					chunkX := min(maxPixelsPerChunk, numX-xOff)
					ConvKernelGradSME32x32Float32Asm(
						unsafe.Pointer(&x[xB+xOff*xStepFloats]),
						unsafe.Pointer(&v[vB+xOff*vStepFloats]),
						unsafe.Pointer(&dstOut[outBaseIdx]),
						1, 1, chunkX,
						xBatchStrideFloats*4, vBatchStrideFloats*4,
						xYStrideFloats*4, vYStrideFloats*4,
						xStepFloats*4, vStepFloats*4,
						outC*4,
						validRows, validCols, *acc,
					)
					*acc = 1
				}
			}
			return
		}

		yChunk := max(1, maxPixelsPerChunk/max(1, numX))
		for b := range batchSize {
			xB := xBaseIdx + b*xBatchStrideFloats
			vB := vBaseIdx + b*vBatchStrideFloats
			for yStart := 0; yStart < numY; yStart += yChunk {
				yCount := min(yChunk, numY-yStart)
				ConvKernelGradSME32x32Float32Asm(
					unsafe.Pointer(&x[xB+yStart*xYStrideFloats]),
					unsafe.Pointer(&v[vB+yStart*vYStrideFloats]),
					unsafe.Pointer(&dstOut[outBaseIdx]),
					1, yCount, numX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
					outC*4,
					validRows, validCols, *acc,
				)
				*acc = 1
			}
		}
	}

	processRangeSME := func(dstOut []float32, ky, kx, cInStart, validRows, cOutStart, validCols, rStart, rEnd int) {
		bFirst := rStart / p.OutH
		yFirst := rStart % p.OutH
		bLast := (rEnd - 1) / p.OutH
		yLast := (rEnd-1)%p.OutH + 1

		acc := 0
		if bFirst == bLast {
			runSubRangeSME(dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bFirst, 1, yFirst, yLast, &acc)
			return
		}
		bCur := bFirst
		if yFirst > 0 {
			runSubRangeSME(dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bFirst, 1, yFirst, p.OutH, &acc)
			bCur++
		}
		fullBatches := bLast - bCur
		if yLast == p.OutH {
			fullBatches++
		}
		if fullBatches > 0 {
			runSubRangeSME(dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bCur, fullBatches, 0, p.OutH, &acc)
		}
		if yLast < p.OutH {
			runSubRangeSME(dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bLast, 1, 0, yLast, &acc)
		}
	}

	processRangeNEON := func(dstOut []float32, ky, kx, cInStart, validRows, cOutStart, validCols, rStart, rEnd int) {
		bFirst := rStart / p.OutH
		yFirst := rStart % p.OutH
		bLast := (rEnd - 1) / p.OutH
		yLast := (rEnd-1)%p.OutH + 1

		if bFirst == bLast {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bFirst, 1, yFirst, yLast)
			return
		}
		bCur := bFirst
		if yFirst > 0 {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bFirst, 1, yFirst, p.OutH)
			bCur++
		}
		fullBatches := bLast - bCur
		if yLast == p.OutH {
			fullBatches++
		}
		if fullBatches > 0 {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bCur, fullBatches, 0, p.OutH)
		}
		if yLast < p.OutH {
			convKernelGradNEONTileFloat32(p, x, v, dstOut, ky, kx, cInStart, validRows, cOutStart, validCols, bLast, 1, 0, yLast)
		}
	}

	maxWorkers := min(totalTasks, maxParallelism)
	if maxWorkers <= 1 {
		for taskIdx := range totalTasks {
			sliceIdx := taskIdx / baseTasks
			rStart := sliceIdx * rowsPerSlice
			if rStart >= totalOutRows {
				continue
			}
			rEnd := min(totalOutRows, rStart+rowsPerSlice)

			baseIdx := taskIdx % baseTasks
			spatialIdx := baseIdx / panelsPerFilter
			rem := baseIdx % panelsPerFilter
			rowPanel := rem / numColPanels
			colPanel := rem % numColPanels
			ky := spatialIdx / kW
			kx := spatialIdx % kW
			cInStart := rowPanel * 32
			validRows := min(32, inC-cInStart)
			cOutStart := colPanel * 32
			validCols := min(32, outC-cOutStart)
			processRangeSME(partialSlices[sliceIdx], ky, kx, cInStart, validRows, cOutStart, validCols, rStart, rEnd)
		}
		for s := 1; s < numSlices; s++ {
			addSliceFloat32NEON(out, partialSlices[s])
		}
		return output, nil
	}

	var nextTask atomic.Int64
	var activeSME atomic.Int32
	var remainingWorkers atomic.Int32
	remainingWorkers.Store(int32(maxWorkers))
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		clusterID, useSME, isEfficiencyCore := simdarch.AcquireSME()
		if useSME {
			activeSME.Add(1)
			defer func() {
				activeSME.Add(-1)
				simdarch.ReleaseSME(clusterID)
				remainingWorkers.Add(-1)
			}()
		} else {
			if remainingWorkers.Add(-1) > 0 || activeSME.Load() > 0 {
				return
			}
		}

		for {
			// Avoid letting the slower Efficiency cluster SME unit grab one
			// of the final 2 tasks when faster Super/Performance SME units are active.
			if useSME && isEfficiencyCore && activeSME.Load() > 1 && totalTasks-int(nextTask.Load()) <= 2 {
				break
			}
			taskIdx := int(nextTask.Add(1) - 1)
			if taskIdx >= totalTasks {
				break
			}
			sliceIdx := taskIdx / baseTasks
			rStart := sliceIdx * rowsPerSlice
			if rStart >= totalOutRows {
				continue
			}
			rEnd := min(totalOutRows, rStart+rowsPerSlice)

			baseIdx := taskIdx % baseTasks
			spatialIdx := baseIdx / panelsPerFilter
			rem := baseIdx % panelsPerFilter
			rowPanel := rem / numColPanels
			colPanel := rem % numColPanels
			ky := spatialIdx / kW
			kx := spatialIdx % kW
			cInStart := rowPanel * 32
			validRows := min(32, inC-cInStart)
			cOutStart := colPanel * 32
			validCols := min(32, outC-cOutStart)

			if useSME {
				processRangeSME(partialSlices[sliceIdx], ky, kx, cInStart, validRows, cOutStart, validCols, rStart, rEnd)
			} else {
				processRangeNEON(partialSlices[sliceIdx], ky, kx, cInStart, validRows, cOutStart, validCols, rStart, rEnd)
			}
		}
	})

	for s := 1; s < numSlices; s++ {
		addSliceFloat32NEON(out, partialSlices[s])
	}
	return output, nil
}

func convKernelGradNEONTileFloat32(
	p *ops.Conv2DParams,
	x, v, out []float32,
	ky, kx, cInStart, validRows, cOutStart, validCols, bStart, bCount, ySliceStart, ySliceEnd int,
) {
	outYStart, outYEnd, inYStart, outXStart, outXEnd, inXStart := p.KernelGradValidRange(ky, kx)
	if ySliceStart > outYStart {
		diff := ySliceStart - outYStart
		outYStart = ySliceStart
		inYStart += diff * p.KerDilH
	}
	if ySliceEnd < outYEnd {
		outYEnd = ySliceEnd
	}
	numY := outYEnd - outYStart
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

	if p.KerDilH == 1 && p.KerDilW == 1 && outXStart == 0 && outXEnd == p.OutW && inXStart == 0 && p.InW == p.OutW {
		numX = numY * p.OutW
		numY = 1
		if outYStart == 0 && outYEnd == p.OutH && inYStart == 0 && p.InH == p.OutH {
			numX = batchSize * numX
			batchSize = 1
		}
	}

	xBatchOffset := bStart * xBatchStrideFloats
	vBatchOffset := bStart * vBatchStrideFloats

	for colOff := 0; colOff < validCols; colOff += 16 {
		cOut := cOutStart + colOff
		subCols := min(16, validCols-colOff)
		vBaseIdx := vBatchOffset + (outYStart*p.OutW+outXStart)*outC + cOut

		if subCols == 16 {
			rOff := 0
			for ; rOff+4 <= validRows; rOff += 4 {
				cIn := cInStart + rOff
				xBaseIdx := xBatchOffset + (inYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				ConvKernelGradNEON4x16Float32Asm(
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
				xBaseIdx := xBatchOffset + (inYStart*p.InW+inXStart)*inC + cIn
				outIdx := ((ky*p.KernelW+kx)*inC+cIn)*outC + cOut
				ConvKernelGradNEON1x16Float32Asm(
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
			xBaseIdx := xBatchOffset + (inYStart*p.InW+inXStart)*inC + cInStart
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

// convKernelGradTailFloat32 handles remainder columns (validCols < 16) in NEON Conv2DKernelGrad
// without reading past the end of x or v.
func convKernelGradTailFloat32(
	x, v, out []float32,
	batchSize, numY, numX int,
	xBaseIdx, vBaseIdx, outBaseIdx int,
	xBatchStride, vBatchStride, xYStride, vYStride, xStep, vStep, outRowStride int,
	validRows, validCols int,
) {
	col := 0
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

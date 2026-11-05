// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package ops

import (
	"runtime"
	"simd/archsimd"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	matmulneon "github.com/gomlx/compute/internal/gobackend/dot/matmul/neon"
	"github.com/gomlx/compute/internal/gobackend/ops/neon"
	"github.com/gomlx/compute/shapes"
)

func init() {
	if gobackend.IsNEONAllowed {
		gobackend.SetNodeExecutor(compute.OpTypeConvGeneral, gobackend.PriorityNEON, execConvGeneralNEON)
	}
	if gobackend.IsSMEAllowed {
		gobackend.SetNodeExecutor(compute.OpTypeConvGeneral, gobackend.PrioritySME, execConvGeneralSME)
	}
}

func getScratchFloat32(backend *gobackend.Backend, size int) (*gobackend.Buffer, []float32, error) {
	buf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, size))
	if err != nil {
		return nil, nil, err
	}
	return buf, buf.Flat.([]float32), nil
}

func execConvGeneralNEON(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	input, kernel := inputs[0], inputs[1]
	if input.RawShape.DType != dtypes.Float32 {
		return nil, gobackend.ErrFallback
	}
	p := GetConv2DParams(node)
	switch p.Kind {
	case Conv2DStandard:
		return execConvStandardNEON(backend, &p, input, kernel, node.Shape)
	case Conv2DKernelGrad:
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
	p := GetConv2DParams(node)
	switch p.Kind {
	case Conv2DStandard:
		return execConvStandardSME(backend, &p, input, kernel, node.Shape)
	case Conv2DKernelGrad:
		return execConvKernelGradSME(backend, &p, input, kernel, node.Shape)
	default:
		return nil, gobackend.ErrFallback
	}
}

// execConvStandardNEON executes 1D and 2D ChannelsLast Forward and VJP-wrt-Input convolutions
// using the 4x16 NEON lane-indexed microkernel (ConvNEONKernel4x16Float32Asm) without LHS transposition.
func execConvStandardNEON(
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

	const mTile = 32
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
					neon.ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, outPtr, K, lhsRowStrideBytes, 64, outRowStrideBytes)
				} else {
					neon.ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPanelPtr, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 64, 64)
					for ir := range validRows {
						dstOff := (mStart+r+ir)*N + colStart
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

	var nextTile atomic.Int64
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		var patchBuf []float32
		if !p.Is1x1Unit || M%4 != 0 {
			patchBufObj, bufSlice, err := getScratchFloat32(backend, mTile*K)
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
	p *Conv2DParams,
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

	// For tiny pixel counts where SME streaming transition overhead dominates, use NEON.
	if M < 8 && N < 8 {
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
		packedRHSBuf, packedRHS, err := getScratchFloat32(backend, K*panelCols)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(packedRHSBuf)
		PackConvKernelFloat32(p, kernel, packedRHS, panelCols)

		patchBufObj, patchBuf, err := getScratchFloat32(backend, 16*K)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(patchBufObj)
		if p.Is1x1Unit {
			copy(patchBuf[:M*K], input[:M*K])
		} else {
			ExtractPatches2DFloat32(p, input, patchBuf, 0, M)
		}

		packedLHSBuf, packedLHS, err := getScratchFloat32(backend, K*16)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(packedLHSBuf)

		neon.ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&patchBuf[0]), unsafe.Pointer(&packedLHS[0]), K, K*4, 64)
		if M > 8 {
			neon.ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&patchBuf[8*K]), unsafe.Pointer(&packedLHS[8]), K, K*4, 64)
		}
		neon.ConvSMEKernel16x16Float32Asm(
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

	packedRHSBuf, packedRHS, err := getScratchFloat32(backend, numPanels*panelSize)
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}
	defer backend.PutBuffer(packedRHSBuf)
	PackConvKernelFloat32(p, kernel, packedRHS, panelCols)

	const mTile = 32
	numMTiles := (M + mTile - 1) / mTile

	processMTileSME := func(mStart, mCount int, patchBuf, packedLHS []float32) {
		var lhsSlice []float32
		if p.Is1x1Unit && mCount == 32 {
			lhsSlice = input[mStart*K : (mStart+32)*K]
		} else if p.Is1x1Unit {
			copy(patchBuf[:mCount*K], input[mStart*K:(mStart+mCount)*K])
			lhsSlice = patchBuf
		} else {
			ExtractPatches2DFloat32(p, input, patchBuf, mStart, mCount)
			lhsSlice = patchBuf
		}

		lhsRowStrideBytes := K * 4
		neon.ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[0]), unsafe.Pointer(&packedLHS[0]), K, lhsRowStrideBytes, 128)
		if mCount > 8 {
			neon.ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[8*K]), unsafe.Pointer(&packedLHS[8]), K, lhsRowStrideBytes, 128)
		}
		if mCount > 16 {
			neon.ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[16*K]), unsafe.Pointer(&packedLHS[16]), K, lhsRowStrideBytes, 128)
		}
		if mCount > 24 {
			neon.ConvPackLHS8StridedFloat32Asm(unsafe.Pointer(&lhsSlice[24*K]), unsafe.Pointer(&packedLHS[24]), K, lhsRowStrideBytes, 128)
		}

		packedLHSPtr := unsafe.Pointer(&packedLHS[0])
		outRowStrideBytes := N * 4
		for panelIdx := range numPanels {
			colStart := panelIdx * 32
			validCols := min(32, N-colStart)
			rhsPanelPtr := unsafe.Pointer(&packedRHS[panelIdx*panelSize])
			outPtr := unsafe.Pointer(&out[mStart*N+colStart])
			neon.ConvSMEKernel32x32Float32Asm(packedLHSPtr, rhsPanelPtr, outPtr, K, outRowStrideBytes, mCount, validCols)
		}
	}

	processMTileNEONFromSME := func(mStart, mCount int, patchBuf []float32) {
		var lhsSlice []float32
		if p.Is1x1Unit && mCount%4 == 0 {
			lhsSlice = input[mStart*K : (mStart+mCount)*K]
		} else if p.Is1x1Unit {
			copy(patchBuf[:mCount*K], input[mStart*K:(mStart+mCount)*K])
			lhsSlice = patchBuf
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
				colStart := panelIdx * 32
				validCols := min(32, N-colStart)
				panelBase := panelIdx * panelSize

				// Left 16 columns of the 32-wide RHS panel (stride = 128 bytes per K step).
				subCols0 := min(16, validCols)
				rhsPtr0 := unsafe.Pointer(&packedRHS[panelBase])
				if validRows == 4 && subCols0 == 16 {
					outPtr0 := unsafe.Pointer(&out[(mStart+r)*N+colStart])
					neon.ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr0, outPtr0, K, lhsRowStrideBytes, 128, outRowStrideBytes)
				} else {
					neon.ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr0, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 64)
					for ir := range validRows {
						dstOff := (mStart+r+ir)*N + colStart
						copy(out[dstOff:dstOff+subCols0], edgeOut[ir*16:ir*16+subCols0])
					}
				}

				// Right 16 columns of the 32-wide RHS panel.
				if validCols > 16 {
					subCols1 := validCols - 16
					rhsPtr1 := unsafe.Pointer(&packedRHS[panelBase+16])
					if validRows == 4 && subCols1 == 16 {
						outPtr1 := unsafe.Pointer(&out[(mStart+r)*N+colStart+16])
						neon.ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr1, outPtr1, K, lhsRowStrideBytes, 128, outRowStrideBytes)
					} else {
						neon.ConvNEONKernel4x16Float32Asm(lhsRowPtr, rhsPtr1, unsafe.Pointer(&edgeOut[0]), K, lhsRowStrideBytes, 128, 64)
						for ir := range validRows {
							dstOff := (mStart+r+ir)*N + colStart + 16
							copy(out[dstOff:dstOff+subCols1], edgeOut[ir*16:ir*16+subCols1])
						}
					}
				}
			}
		}
	}

	maxWorkers := min(numMTiles, backend.Scheduler.AdjustedMaxParallelism())
	if maxWorkers <= 1 || M*K*N < 16384 {
		patchBufObj, patchBuf, err := getScratchFloat32(backend, mTile*K)
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(patchBufObj)

		packedLHSBuf, packedLHS, err := getScratchFloat32(backend, K*32)
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
	backend.Scheduler.SaturateUpTo(maxWorkers, func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		clusterID, useSME, isSlowECore := matmulneon.SMEAcquireCluster()
		if useSME {
			activeSME.Add(1)
			defer func() {
				activeSME.Add(-1)
				matmulneon.SMEReleaseCluster(clusterID)
			}()
		} else if isSlowECore {
			// Efficiency cluster SME is already saturated; avoid slow E-core NEON tail stalls.
			return
		}

		patchBufObj, patchBuf, err := getScratchFloat32(backend, mTile*K)
		if err != nil {
			return
		}
		defer backend.PutBuffer(patchBufObj)

		var packedLHS []float32
		if useSME {
			packedLHSBuf, bufSlice, err := getScratchFloat32(backend, K*32)
			if err != nil {
				return
			}
			defer backend.PutBuffer(packedLHSBuf)
			packedLHS = bufSlice
		}

		for {
			if !useSME {
				rem := numMTiles - int(nextTile.Load())
				smeCount := int(activeSME.Load())
				if smeCount > 0 && rem <= 4*smeCount {
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
// zero packing and zero intermediate buffers.
func execConvKernelGradNEON(
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
	numColPanels := (outC + 15) / 16
	totalTasks := kH * kW * numColPanels

	processTask := func(taskIdx int) {
		panelIdx := taskIdx % numColPanels
		spatialIdx := taskIdx / numColPanels
		ky := spatialIdx / kW
		kx := spatialIdx % kW
		cOutStart := panelIdx * 16
		validCols := min(16, outC-cOutStart)
		convKernelGradNEONTileFloat32(p, x, v, out, ky, kx, 0, inC, cOutStart, validCols)
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

// execConvKernelGradSME computes VJP wrt Kernel (Conv2DKernelGrad) using 32x32 SME FMOPA
// outer-product microkernels (ConvKernelGradSME32x32Float32Asm) co-scheduled with per-core
// NEON 4x16/1x16 microkernels across CPU clusters.
func execConvKernelGradSME(
	backend *gobackend.Backend,
	p *Conv2DParams,
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
	totalTasks := kH * kW * panelsPerFilter

	processTaskSME := func(ky, kx, cInStart, validRows, cOutStart, validCols int) {
		outYStart, outYEnd, inYStart, outXStart, outXEnd, inXStart := p.KernelGradValidRange(ky, kx)
		numY := outYEnd - outYStart
		numX := outXEnd - outXStart
		if numY <= 0 || numX <= 0 {
			return
		}

		batchSize := p.BatchSize
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

		xBaseIdx := (inYStart*p.InW+inXStart)*inC + cInStart
		vBaseIdx := (outYStart*p.OutW+outXStart)*outC + cOutStart
		outBaseIdx := ((ky*kW+kx)*inC+cInStart)*outC + cOutStart

		// Chunk large contracting domains into <= 2048 pixels per SME call so async SIGURG
		// preemption signals never cause long retry loops.
		const maxPixelsPerChunk = 2048
		totalPixels := batchSize * numY * numX
		if totalPixels <= maxPixelsPerChunk {
			neon.ConvKernelGradSME32x32Float32Asm(
				unsafe.Pointer(&x[xBaseIdx]),
				unsafe.Pointer(&v[vBaseIdx]),
				unsafe.Pointer(&out[outBaseIdx]),
				batchSize, numY, numX,
				xBatchStrideFloats*4, vBatchStrideFloats*4,
				xYStrideFloats*4, vYStrideFloats*4,
				xStepFloats*4, vStepFloats*4,
				outC*4,
				validRows, validCols, 0,
			)
			return
		}

		// When numY == 1 (collapsed contiguous spatial domain) and numX > maxPixelsPerChunk:
		if numY == 1 && batchSize == 1 {
			acc := 0
			for xOff := 0; xOff < numX; xOff += maxPixelsPerChunk {
				chunkX := min(maxPixelsPerChunk, numX-xOff)
				neon.ConvKernelGradSME32x32Float32Asm(
					unsafe.Pointer(&x[xBaseIdx+xOff*xStepFloats]),
					unsafe.Pointer(&v[vBaseIdx+xOff*vStepFloats]),
					unsafe.Pointer(&out[outBaseIdx]),
					1, 1, chunkX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
					outC*4,
					validRows, validCols, acc,
				)
				acc = 1
			}
			return
		}

		// Chunk across batchSize and/or numY so each call processes <= maxPixelsPerChunk pixels.
		pixelsPerBatch := numY * numX
		if pixelsPerBatch <= maxPixelsPerChunk {
			batchChunk := max(1, maxPixelsPerChunk/pixelsPerBatch)
			acc := 0
			for bStart := 0; bStart < batchSize; bStart += batchChunk {
				bCount := min(batchChunk, batchSize-bStart)
				neon.ConvKernelGradSME32x32Float32Asm(
					unsafe.Pointer(&x[xBaseIdx+bStart*xBatchStrideFloats]),
					unsafe.Pointer(&v[vBaseIdx+bStart*vBatchStrideFloats]),
					unsafe.Pointer(&out[outBaseIdx]),
					bCount, numY, numX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
					outC*4,
					validRows, validCols, acc,
				)
				acc = 1
			}
			return
		}

		// A single batch has > maxPixelsPerChunk pixels: chunk across b and y (and x if numY == 1).
		acc := 0
		if numY == 1 {
			for b := range batchSize {
				xB := xBaseIdx + b*xBatchStrideFloats
				vB := vBaseIdx + b*vBatchStrideFloats
				for xOff := 0; xOff < numX; xOff += maxPixelsPerChunk {
					chunkX := min(maxPixelsPerChunk, numX-xOff)
					neon.ConvKernelGradSME32x32Float32Asm(
						unsafe.Pointer(&x[xB+xOff*xStepFloats]),
						unsafe.Pointer(&v[vB+xOff*vStepFloats]),
						unsafe.Pointer(&out[outBaseIdx]),
						1, 1, chunkX,
						xBatchStrideFloats*4, vBatchStrideFloats*4,
						xYStrideFloats*4, vYStrideFloats*4,
						xStepFloats*4, vStepFloats*4,
						outC*4,
						validRows, validCols, acc,
					)
					acc = 1
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
				neon.ConvKernelGradSME32x32Float32Asm(
					unsafe.Pointer(&x[xB+yStart*xYStrideFloats]),
					unsafe.Pointer(&v[vB+yStart*vYStrideFloats]),
					unsafe.Pointer(&out[outBaseIdx]),
					1, yCount, numX,
					xBatchStrideFloats*4, vBatchStrideFloats*4,
					xYStrideFloats*4, vYStrideFloats*4,
					xStepFloats*4, vStepFloats*4,
					outC*4,
					validRows, validCols, acc,
				)
				acc = 1
			}
		}
	}

	maxWorkers := min(totalTasks, backend.Scheduler.AdjustedMaxParallelism())
	if maxWorkers <= 1 {
		for taskIdx := range totalTasks {
			spatialIdx := taskIdx / panelsPerFilter
			rem := taskIdx % panelsPerFilter
			rowPanel := rem / numColPanels
			colPanel := rem % numColPanels
			ky := spatialIdx / kW
			kx := spatialIdx % kW
			cInStart := rowPanel * 32
			validRows := min(32, inC-cInStart)
			cOutStart := colPanel * 32
			validCols := min(32, outC-cOutStart)
			processTaskSME(ky, kx, cInStart, validRows, cOutStart, validCols)
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

		clusterID, useSME, _ := matmulneon.SMEAcquireCluster()
		if useSME {
			activeSME.Add(1)
			defer func() {
				activeSME.Add(-1)
				matmulneon.SMEReleaseCluster(clusterID)
				remainingWorkers.Add(-1)
			}()
		} else {
			if remainingWorkers.Add(-1) > 0 || activeSME.Load() > 0 {
				return
			}
		}

		for {
			taskIdx := int(nextTask.Add(1) - 1)
			if taskIdx >= totalTasks {
				break
			}
			spatialIdx := taskIdx / panelsPerFilter
			rem := taskIdx % panelsPerFilter
			rowPanel := rem / numColPanels
			colPanel := rem % numColPanels
			ky := spatialIdx / kW
			kx := spatialIdx % kW
			cInStart := rowPanel * 32
			validRows := min(32, inC-cInStart)
			cOutStart := colPanel * 32
			validCols := min(32, outC-cOutStart)

			if useSME {
				processTaskSME(ky, kx, cInStart, validRows, cOutStart, validCols)
			} else {
				convKernelGradNEONTileFloat32(p, x, v, out, ky, kx, cInStart, validRows, cOutStart, validCols)
			}
		}
	})
	return output, nil
}

func convKernelGradNEONTileFloat32(
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
				neon.ConvKernelGradNEON4x16Float32Asm(
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
				neon.ConvKernelGradNEON1x16Float32Asm(
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

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/simdarch"
)

func init() {
	if gobackend.IsNEONAllowed {
		registerLayerNormNEON()
		gobackend.SetQuantizeGGMLActivationsArchDispatcher(gobackend.PriorityNEON, QuantizeGGMLActivationsNEON)
		gobackend.SetQuantizeSymmetricInt8ActivationsArchDispatcher(gobackend.PriorityNEON, QuantizeSymmetricInt8ActivationsNEON)
		if gobackend.IsARM64I8MMAllowed || gobackend.IsARM64DotProdAllowed {
			gobackend.SetQuantizedPanelGEMMArchDispatcher(gobackend.PriorityNEON, DispatchQuantizedPanelGEMMNEON)
		}
		if gobackend.IsSMEAllowed && gobackend.IsARM64I8MMAllowed && simdarch.NEONUseAsm {
			gobackend.SetQuantizedPanelGEMMArchDispatcher(gobackend.PrioritySME, DispatchQuantizedPanelGEMMSME)
		}
	}
}

//go:noescape
func quantizedPanelKernel3x16NEONAsm(
	qxPtr unsafe.Pointer,
	qxRowBytes int,
	dxPairPtr unsafe.Pointer,
	dxRowBytes int,
	stripPtr unsafe.Pointer,
	biasPtr unsafe.Pointer,
	outPtr unsafe.Pointer,
	outRowBytes int,
	numSubBlocks int,
	quadsPerSubBlk int,
	numMBlocks int,
	useUSDOT int,
)

//go:noescape
func quantizedPanelKernel1x16NEONAsm(
	qxPtr unsafe.Pointer,
	dxPairPtr unsafe.Pointer,
	stripPtr unsafe.Pointer,
	biasPtr unsafe.Pointer,
	outPtr unsafe.Pointer,
	numSubBlocks int,
	quadsPerSubBlk int,
	useUSDOT int,
)

//go:noescape
func quantizeGGMLQ8_0RowsNEONAsm(
	xPtr unsafe.Pointer,
	qxPtr unsafe.Pointer,
	dxPtr unsafe.Pointer,
	numBlocks int,
)

//go:noescape
func quantizeGGMLQ8_KRowsNEONAsm(
	xPtr unsafe.Pointer,
	qxPtr unsafe.Pointer,
	dxPtr unsafe.Pointer,
	numSuperBlocks int,
	subBlockSize int,
)

// DispatchQuantizedPanelGEMMNEON executes K4-interleaved quantized integer GEMM using
// ARM64 NEON I8MM (USDOT) or DotProd (SDOT) 3x16 and 1x16 assembly microkernels with
// L1d/L2 cache-optimal strip-outer tiling and 2D parallel scheduling across M-blocks and 16-column strips.
func DispatchQuantizedPanelGEMMNEON(cfg *gobackend.QuantizedPanelGEMMConfig) {
	useUSDOT := 0
	if gobackend.IsARM64I8MMAllowed {
		useUSDOT = 1
	}
	dispatchQuantizedPanelGEMMNEONWithMode(cfg, useUSDOT)
}

func runQuantizedRowRangeNEON(cfg *gobackend.QuantizedPanelGEMMConfig, mRangeStart, mRangeEnd, sStart, sEnd, useUSDOT int) {
	if mRangeStart >= mRangeEnd || sStart >= sEnd {
		return
	}
	const mr = 3
	const nr = 16
	const mcRows = 96

	N := cfg.N
	nStart := cfg.NStart
	chunkCols := cfg.ChunkCols
	subBlkBytes := gobackend.QuantizedPanelStripSubBlockBytes(cfg.QuadsPerSubBlk)
	stripBytes := cfg.NumSubBlocks * subBlkBytes
	qxRowBytes := cfg.KPadded
	dxStride := cfg.NumSubBlocks * 2
	dxRowBytes := dxStride * 4
	outRowBytes := N * 4

	var biasTail [nr]float32
	var outTail [mr * nr]float32

	colRangeStart := sStart * nr
	colRangeEnd := min(sEnd*nr, chunkCols)

	for mStart := mRangeStart; mStart < mRangeEnd; mStart += mcRows {
		mLimit := min(mStart+mcRows, mRangeEnd)
		fullMB := (mLimit - mStart) / mr
		remRows := (mLimit - mStart) % mr

		for s := sStart; s < sEnd; s++ {
			colOff := s * nr
			validCols := min(nr, chunkCols-colOff)
			sPtr := unsafe.Pointer(&cfg.Panel[s*stripBytes])

			var bPtr unsafe.Pointer
			if validCols == nr {
				if cfg.Bias != nil {
					bPtr = unsafe.Pointer(&cfg.Bias[colOff])
				}
			} else if cfg.Bias != nil {
				clear(biasTail[:])
				copy(biasTail[:validCols], cfg.Bias[colOff:colOff+validCols])
				bPtr = unsafe.Pointer(&biasTail[0])
			}

			if fullMB > 0 {
				if validCols == nr {
					qxBase := unsafe.Pointer(&cfg.Qx[mStart*qxRowBytes])
					dxBase := unsafe.Pointer(&cfg.DxPairs[mStart*dxStride])
					oPtr := unsafe.Pointer(&cfg.Out[mStart*N+nStart+colOff])
					quantizedPanelKernel3x16NEONAsm(
						qxBase, qxRowBytes,
						dxBase, dxRowBytes,
						sPtr, bPtr,
						oPtr, outRowBytes,
						cfg.NumSubBlocks, cfg.QuadsPerSubBlk, fullMB, useUSDOT,
					)
				} else {
					for mb := range fullMB {
						m0 := mStart + mb*mr
						qxBase := unsafe.Pointer(&cfg.Qx[m0*qxRowBytes])
						dxBase := unsafe.Pointer(&cfg.DxPairs[m0*dxStride])
						quantizedPanelKernel3x16NEONAsm(
							qxBase, qxRowBytes,
							dxBase, dxRowBytes,
							sPtr, bPtr,
							unsafe.Pointer(&outTail[0]), nr*4,
							cfg.NumSubBlocks, cfg.QuadsPerSubBlk, 1, useUSDOT,
						)
						for r := range mr {
							dstOff := (m0+r)*N + nStart + colOff
							copy(cfg.Out[dstOff:dstOff+validCols], outTail[r*nr:r*nr+validCols])
						}
					}
				}
			}

			if remRows > 0 {
				mRemStart := mStart + fullMB*mr
				for r := range remRows {
					m := mRemStart + r
					qxRowPtr := unsafe.Pointer(&cfg.Qx[m*qxRowBytes])
					dxRowPtr := unsafe.Pointer(&cfg.DxPairs[m*dxStride])
					if validCols == nr {
						oPtr := unsafe.Pointer(&cfg.Out[m*N+nStart+colOff])
						quantizedPanelKernel1x16NEONAsm(
							qxRowPtr, dxRowPtr,
							sPtr, bPtr, oPtr,
							cfg.NumSubBlocks, cfg.QuadsPerSubBlk, useUSDOT,
						)
					} else {
						quantizedPanelKernel1x16NEONAsm(
							qxRowPtr, dxRowPtr,
							sPtr, bPtr, unsafe.Pointer(&outTail[0]),
							cfg.NumSubBlocks, cfg.QuadsPerSubBlk, useUSDOT,
						)
						dstOff := m*N + nStart + colOff
						copy(cfg.Out[dstOff:dstOff+validCols], outTail[:validCols])
					}
				}
			}
		}

		if cfg.ActFn != nil && colRangeStart < colRangeEnd {
			if nStart == 0 && colRangeStart == 0 && colRangeEnd == N {
				span := cfg.Out[mStart*N : mLimit*N]
				cfg.ActFn(span, span)
			} else {
				for m := mStart; m < mLimit; m++ {
					rowSlice := cfg.Out[m*N+nStart+colRangeStart : m*N+nStart+colRangeEnd]
					cfg.ActFn(rowSlice, rowSlice)
				}
			}
		}
	}
}

func dispatchQuantizedPanelGEMMNEONWithMode(cfg *gobackend.QuantizedPanelGEMMConfig, useUSDOT int) {
	const mr = 3

	M := cfg.M
	chunkCols := cfg.ChunkCols
	if M == 0 || chunkCols == 0 {
		return
	}

	numStrips := (chunkCols + gobackend.QuantizedPanelStripCols - 1) / gobackend.QuantizedPanelStripCols
	runTile := func(mbStart, mbEnd, sStart, sEnd int) {
		runQuantizedRowRangeNEON(cfg, mbStart*mr, min(mbEnd*mr, M), sStart, sEnd, useUSDOT)
	}

	numMBlocks := (M + mr - 1) / mr
	totalWork := M * cfg.KPadded * chunkCols
	if cfg.Backend != nil && cfg.Backend.Scheduler != nil && cfg.Backend.Scheduler.IsEnabled() && totalWork > 4096 {
		numWorkers := cfg.Backend.Scheduler.EffectiveParallelism()
		if numWorkers > 1 {
			if numMBlocks >= numWorkers || numStrips <= 1 {
				cfg.Backend.Scheduler.ParallelizeChunks(numMBlocks, 1, func(mbStart, mbEnd int) {
					runTile(mbStart, mbEnd, 0, numStrips)
				})
				return
			}
			if numMBlocks == 1 {
				cfg.Backend.Scheduler.ParallelizeChunks(numStrips, 1, func(sStart, sEnd int) {
					runTile(0, 1, sStart, sEnd)
				})
				return
			}
			totalTiles := numStrips * numMBlocks
			cfg.Backend.Scheduler.ParallelizeChunks(totalTiles, 1, func(tStart, tEnd int) {
				for t := tStart; t < tEnd; {
					s := t / numMBlocks
					mbLo := t % numMBlocks
					sNextTile := (s + 1) * numMBlocks
					tGroupEnd := min(tEnd, sNextTile)
					mbHi := mbLo + (tGroupEnd - t)
					runTile(mbLo, mbHi, s, s+1)
					t = tGroupEnd
				}
			})
			return
		}
	}
	runTile(0, numMBlocks, 0, numStrips)
}

// QuantizeGGMLActivationsNEON quantizes float32 activation rows directly into canonical
// (qx, dxPairs) using ARM64 NEON assembly with in-register reductions and hardware FP16 scale rounding.
func QuantizeGGMLActivationsNEON(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, subBlockSize int,
	q8KSuperBlocks bool,
) {
	rows := mEnd - mStart
	if rows <= 0 || K <= 0 {
		return
	}
	numSubBlocks := K / subBlockSize
	xPtr := unsafe.Pointer(&xF32[mStart*K])
	qxPtr := unsafe.Pointer(&qx[mStart*K])
	dxPtr := unsafe.Pointer(&dxPairs[mStart*numSubBlocks*2])

	if !q8KSuperBlocks {
		if gobackend.IsARM64FP16Allowed {
			quantizeGGMLQ8_0RowsNEONAsm(xPtr, qxPtr, dxPtr, rows*numSubBlocks)
			return
		}
		quantizeGGMLQ8_0FallbackNEON(xF32, qx, dxPairs, mStart, mEnd, K, numSubBlocks)
		return
	}
	quantizeGGMLQ8_KRowsNEONAsm(xPtr, qxPtr, dxPtr, rows*(K/256), subBlockSize)
}

func reduceMaxFloat32x4(v archsimd.Float32x4) float32 {
	var buf [4]float32
	v.StoreArray(&buf)
	return max(max(buf[0], buf[1]), max(buf[2], buf[3]))
}

func storeInt32x4AsInt8AndSum(iv archsimd.Int32x4, dst []int8) int32 {
	var tmp [4]int32
	iv.StoreArray(&tmp)
	dst[0] = int8(tmp[0])
	dst[1] = int8(tmp[1])
	dst[2] = int8(tmp[2])
	dst[3] = int8(tmp[3])
	return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3])
}

func quantizeGGMLQ8_0FallbackNEON(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, numSubBlocks int,
) {
	clampMin := archsimd.BroadcastInt32x4(-127)
	clampMax := archsimd.BroadcastInt32x4(127)
	for m := mStart; m < mEnd; m++ {
		xRow := xF32[m*K : (m+1)*K]
		qRow := qx[m*K : (m+1)*K]
		dRow := dxPairs[m*numSubBlocks*2 : (m+1)*numSubBlocks*2]
		for sb := range numSubBlocks {
			span := xRow[sb*32 : (sb+1)*32]
			dst := qRow[sb*32 : (sb+1)*32]
			vMax0 := archsimd.LoadFloat32x4(span[0:4]).Abs().Max(archsimd.LoadFloat32x4(span[4:8]).Abs())
			vMax1 := archsimd.LoadFloat32x4(span[8:12]).Abs().Max(archsimd.LoadFloat32x4(span[12:16]).Abs())
			vMax2 := archsimd.LoadFloat32x4(span[16:20]).Abs().Max(archsimd.LoadFloat32x4(span[20:24]).Abs())
			vMax3 := archsimd.LoadFloat32x4(span[24:28]).Abs().Max(archsimd.LoadFloat32x4(span[28:32]).Abs())
			maxAbs := reduceMaxFloat32x4(vMax0.Max(vMax1).Max(vMax2.Max(vMax3)))
			if maxAbs == 0 {
				clear(dst)
				dRow[2*sb] = 0
				dRow[2*sb+1] = 0
				continue
			}
			d := float16.FromFloat32(maxAbs / 127.0).Float32()
			if d == 0 {
				clear(dst)
				dRow[2*sb] = 0
				dRow[2*sb+1] = 0
				continue
			}
			vInv := archsimd.BroadcastFloat32x4(1.0 / d)
			var sumQ int32
			for i := 0; i < 32; i += 4 {
				iv := archsimd.LoadFloat32x4(span[i : i+4]).Mul(vInv).Round().ConvertToInt32().Max(clampMin).Min(clampMax)
				sumQ += storeInt32x4AsInt8AndSum(iv, dst[i:i+4])
			}
			dRow[2*sb] = d
			dRow[2*sb+1] = d * float32(sumQ)
		}
	}
}

// QuantizeSymmetricInt8ActivationsNEON quantizes float32 activation rows per kBlock using NEON.
func QuantizeSymmetricInt8ActivationsNEON(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, kBlock, kPadded, numSubBlocks int,
) {
	bPad := (kBlock + 3) &^ 3
	clampMin := archsimd.BroadcastInt32x4(-127)
	clampMax := archsimd.BroadcastInt32x4(127)

	for m := mStart; m < mEnd; m++ {
		xRow := xF32[m*K : (m+1)*K]
		qRow := qx[m*kPadded : (m+1)*kPadded]
		dRow := dxPairs[m*numSubBlocks*2 : (m+1)*numSubBlocks*2]

		for kb := range numSubBlocks {
			k0 := kb * kBlock
			k1 := min(k0+kBlock, K)
			span := xRow[k0:k1]
			dst := qRow[kb*bPad : (kb+1)*bPad]
			n := len(span)

			var maxAbs float32
			i := 0
			if n >= 4 {
				vMax := archsimd.LoadFloat32x4(span[0:4]).Abs()
				for i = 4; i+4 <= n; i += 4 {
					vMax = vMax.Max(archsimd.LoadFloat32x4(span[i : i+4]).Abs())
				}
				maxAbs = reduceMaxFloat32x4(vMax)
			}
			for ; i < n; i++ {
				av := span[i]
				if av < 0 {
					av = -av
				}
				if av > maxAbs {
					maxAbs = av
				}
			}
			if maxAbs == 0 {
				clear(dst)
				dRow[2*kb] = 0
				dRow[2*kb+1] = 0
				continue
			}

			sc := maxAbs * (1.0 / 127.0)
			invSc := 127.0 / maxAbs
			vInv := archsimd.BroadcastFloat32x4(invSc)
			var sumQ int32
			i = 0
			for ; i+4 <= n; i += 4 {
				iv := archsimd.LoadFloat32x4(span[i : i+4]).Mul(vInv).Round().ConvertToInt32().Max(clampMin).Min(clampMax)
				sumQ += storeInt32x4AsInt8AndSum(iv, dst[i:i+4])
			}
			for ; i < n; i++ {
				var single [4]float32
				single[0] = span[i] * invSc
				q := archsimd.LoadFloat32x4Array(&single).Round().ConvertToInt32().Max(clampMin).Min(clampMax)
				var qArr [4]int32
				q.StoreArray(&qArr)
				dst[i] = int8(qArr[0])
				sumQ += qArr[0]
			}
			if n < bPad {
				clear(dst[n:])
			}
			dRow[2*kb] = sc
			dRow[2*kb+1] = sc * float32(sumQ)
		}
	}
}

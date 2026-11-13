// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX2Allowed {
		gobackend.SetQuantizeGGMLActivationsArchDispatcher(gobackend.PriorityArch, QuantizeGGMLActivationsAVX2)
		gobackend.SetQuantizeSymmetricInt8ActivationsArchDispatcher(gobackend.PriorityArch, QuantizeSymmetricInt8ActivationsAVX2)
		gobackend.SetQuantizedPanelGEMMArchDispatcher(gobackend.PriorityArch, DispatchQuantizedPanelGEMMAVX2)
	}
}

//go:noescape
func quantizedPanelKernel3x16AVX2Asm(
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
	wideUnsigned int,
)

//go:noescape
func quantizedPanelKernel1x16AVX2Asm(
	qxPtr unsafe.Pointer,
	dxPairPtr unsafe.Pointer,
	stripPtr unsafe.Pointer,
	biasPtr unsafe.Pointer,
	outPtr unsafe.Pointer,
	numSubBlocks int,
	quadsPerSubBlk int,
	wideUnsigned int,
)

//go:noescape
func quantizeGGMLQ8_0RowsAVX2Asm(
	xPtr unsafe.Pointer,
	qxPtr unsafe.Pointer,
	dxPtr unsafe.Pointer,
	numBlocks int,
)

//go:noescape
func quantizeGGMLQ8_KRowsAVX2Asm(
	xPtr unsafe.Pointer,
	qxPtr unsafe.Pointer,
	dxPtr unsafe.Pointer,
	numSuperBlocks int,
	subBlockSize int,
)

// DispatchQuantizedPanelGEMMAVX2 executes K4-interleaved quantized integer GEMM using
// pure-VEX AVX2 (VPMADDUBSW + VPMADDWD) 3x16 and 1x16 assembly microkernels with L1d/L2
// cache-optimal strip-outer tiling and 2D parallel scheduling across M-blocks and 16-column strips.
func DispatchQuantizedPanelGEMMAVX2(cfg *gobackend.QuantizedPanelGEMMConfig) {
	const mr = 3
	const nr = 16
	// mcBlocks = 32 (96 rows) keeps Qx + DxPairs inside L2 cache while reusing each 16-column
	// weight strip across up to 96 activation rows in L1d/L2.
	const mcBlocks = 32

	M := cfg.M
	N := cfg.N
	nStart := cfg.NStart
	chunkCols := cfg.ChunkCols
	if M == 0 || chunkCols == 0 {
		return
	}

	numStrips := (chunkCols + gobackend.QuantizedPanelStripCols - 1) / gobackend.QuantizedPanelStripCols
	subBlkBytes := gobackend.QuantizedPanelStripSubBlockBytes(cfg.QuadsPerSubBlk)
	stripBytes := cfg.NumSubBlocks * subBlkBytes
	qxRowBytes := cfg.KPadded
	dxStride := cfg.NumSubBlocks * 2
	dxRowBytes := dxStride * 4
	outRowBytes := N * 4
	wideFlag := 0
	if cfg.WideUnsigned {
		wideFlag = 1
	}

	runTile := func(mbStart, mbEnd, sStart, sEnd int) {
		if mbStart >= mbEnd || sStart >= sEnd {
			return
		}
		var biasTail [nr]float32
		var outTail [mr * nr]float32

		colRangeStart := sStart * nr
		colRangeEnd := min(sEnd*nr, chunkCols)

		for mb0 := mbStart; mb0 < mbEnd; mb0 += mcBlocks {
			mb1 := min(mb0+mcBlocks, mbEnd)

			// Strip outer loop: keeps the active 16-column weight strip (7.5-30.7 KB)
			// pinned in L1d across all M-blocks in [mb0, mb1).
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

				for mb := mb0; mb < mb1; mb++ {
					m0 := mb * mr
					rows := min(mr, M-m0)
					if rows == mr {
						qxBase := unsafe.Pointer(&cfg.Qx[m0*qxRowBytes])
						dxBase := unsafe.Pointer(&cfg.DxPairs[m0*dxStride])
						if validCols == nr {
							oPtr := unsafe.Pointer(&cfg.Out[m0*N+nStart+colOff])
							quantizedPanelKernel3x16AVX2Asm(
								qxBase, qxRowBytes,
								dxBase, dxRowBytes,
								sPtr, bPtr,
								oPtr, outRowBytes,
								cfg.NumSubBlocks, cfg.QuadsPerSubBlk, wideFlag,
							)
						} else {
							quantizedPanelKernel3x16AVX2Asm(
								qxBase, qxRowBytes,
								dxBase, dxRowBytes,
								sPtr, bPtr,
								unsafe.Pointer(&outTail[0]), nr*4,
								cfg.NumSubBlocks, cfg.QuadsPerSubBlk, wideFlag,
							)
							for r := range mr {
								dstOff := (m0+r)*N + nStart + colOff
								copy(cfg.Out[dstOff:dstOff+validCols], outTail[r*nr:r*nr+validCols])
							}
						}
					} else {
						for r := range rows {
							m := m0 + r
							qxRowPtr := unsafe.Pointer(&cfg.Qx[m*qxRowBytes])
							dxRowPtr := unsafe.Pointer(&cfg.DxPairs[m*dxStride])
							if validCols == nr {
								oPtr := unsafe.Pointer(&cfg.Out[m*N+nStart+colOff])
								quantizedPanelKernel1x16AVX2Asm(
									qxRowPtr, dxRowPtr,
									sPtr, bPtr, oPtr,
									cfg.NumSubBlocks, cfg.QuadsPerSubBlk, wideFlag,
								)
							} else {
								quantizedPanelKernel1x16AVX2Asm(
									qxRowPtr, dxRowPtr,
									sPtr, bPtr, unsafe.Pointer(&outTail[0]),
									cfg.NumSubBlocks, cfg.QuadsPerSubBlk, wideFlag,
								)
								dstOff := m*N + nStart + colOff
								copy(cfg.Out[dstOff:dstOff+validCols], outTail[:validCols])
							}
						}
					}
				}
			}

			if cfg.ActFn != nil && colRangeStart < colRangeEnd {
				mStart := mb0 * mr
				mLimit := min(mb1*mr, M)
				for m := mStart; m < mLimit; m++ {
					rowSlice := cfg.Out[m*N+nStart+colRangeStart : m*N+nStart+colRangeEnd]
					cfg.ActFn(rowSlice, rowSlice)
				}
			}
		}
	}

	numMBlocks := (M + mr - 1) / mr
	totalWork := M * cfg.KPadded * chunkCols
	if cfg.Backend != nil && cfg.Backend.Scheduler != nil && cfg.Backend.Scheduler.IsEnabled() && totalWork > 4096 {
		numWorkers := cfg.Backend.Scheduler.EffectiveParallelism()
		if numWorkers > 1 {
			if numMBlocks >= numWorkers || numStrips <= 1 {
				if numMBlocks > 1 {
					cfg.Backend.Scheduler.ParallelizeChunks(numMBlocks, 1, func(mbStart, mbEnd int) {
						runTile(mbStart, mbEnd, 0, numStrips)
					})
					return
				}
			} else if numMBlocks == 1 {
				cfg.Backend.Scheduler.ParallelizeChunks(numStrips, 1, func(sStart, sEnd int) {
					runTile(0, 1, sStart, sEnd)
				})
				return
			} else {
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
	}
	runTile(0, numMBlocks, 0, numStrips)
}

// QuantizeGGMLActivationsAVX2 quantizes float32 activation rows directly into canonical
// (qx, dxPairs) using pure-VEX AVX2 + F16C assembly with in-register reductions.
func QuantizeGGMLActivationsAVX2(
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
		quantizeGGMLQ8_0RowsAVX2Asm(xPtr, qxPtr, dxPtr, rows*numSubBlocks)
		return
	}
	quantizeGGMLQ8_KRowsAVX2Asm(xPtr, qxPtr, dxPtr, rows*(K/256), subBlockSize)
}

func reduceMaxFloat32x8(v archsimd.Float32x8) float32 {
	m4 := v.GetLo().Max(v.GetHi())
	var buf [4]float32
	m4.StoreArray(&buf)
	return max(max(buf[0], buf[1]), max(buf[2], buf[3]))
}

func storeInt32x8AsInt8AndSum(iv archsimd.Int32x8, dst []int8) int32 {
	var tmp [8]int32
	iv.StoreArray(&tmp)
	dst[0] = int8(tmp[0])
	dst[1] = int8(tmp[1])
	dst[2] = int8(tmp[2])
	dst[3] = int8(tmp[3])
	dst[4] = int8(tmp[4])
	dst[5] = int8(tmp[5])
	dst[6] = int8(tmp[6])
	dst[7] = int8(tmp[7])
	return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) + (tmp[4] + tmp[5]) + (tmp[6] + tmp[7])
}

// QuantizeSymmetricInt8ActivationsAVX2 quantizes float32 activation rows per kBlock using AVX2.
func QuantizeSymmetricInt8ActivationsAVX2(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, kBlock, kPadded, numSubBlocks int,
) {
	bPad := (kBlock + 3) &^ 3
	clampMin := archsimd.BroadcastInt32x8(-127)
	clampMax := archsimd.BroadcastInt32x8(127)

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
			if n >= 8 {
				vMax := archsimd.LoadFloat32x8(span[0:8]).Abs()
				for i = 8; i+8 <= n; i += 8 {
					vMax = vMax.Max(archsimd.LoadFloat32x8(span[i : i+8]).Abs())
				}
				maxAbs = reduceMaxFloat32x8(vMax)
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
			vInv := archsimd.BroadcastFloat32x8(invSc)
			var sumQ int32
			i = 0
			for ; i+8 <= n; i += 8 {
				iv := archsimd.LoadFloat32x8(span[i : i+8]).Mul(vInv).Round().ConvertToInt32().Max(clampMin).Min(clampMax)
				sumQ += storeInt32x8AsInt8AndSum(iv, dst[i:i+8])
			}
			for ; i < n; i++ {
				var single [8]float32
				single[0] = span[i] * invSc
				q := archsimd.LoadFloat32x8Array(&single).Round().ConvertToInt32().Max(clampMin).Min(clampMax)
				var qArr [4]int32
				q.GetLo().StoreArray(&qArr)
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

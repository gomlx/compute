// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX512Allowed {
		gobackend.SetQuantizeGGMLActivationsArchDispatcher(gobackend.PriorityArch+1, QuantizeGGMLActivationsAVX512)
		gobackend.SetQuantizeSymmetricInt8ActivationsArchDispatcher(gobackend.PriorityArch+1, QuantizeSymmetricInt8ActivationsAVX512)
	}
	if gobackend.IsAVX512VNNIAllowed {
		gobackend.SetQuantizedPanelGEMMArchDispatcher(gobackend.PriorityArch+1, DispatchQuantizedPanelGEMMAVX512VNNI)
	}
}

//go:noescape
func quantizedPanelKernel6x32AVX512VNNIAsm(
	qxPtr unsafe.Pointer,
	qxRowBytes int,
	dxPairPtr unsafe.Pointer,
	dxRowBytes int,
	strip0Ptr unsafe.Pointer,
	strip1Ptr unsafe.Pointer,
	biasPtr unsafe.Pointer,
	outPtr unsafe.Pointer,
	outRowBytes int,
	numSubBlocks int,
	quadsPerSubBlk int,
	numMBlocks int,
)

//go:noescape
func quantizedPanelKernel1x32AVX512VNNIAsm(
	qxPtr unsafe.Pointer,
	dxPairPtr unsafe.Pointer,
	strip0Ptr unsafe.Pointer,
	strip1Ptr unsafe.Pointer,
	biasPtr unsafe.Pointer,
	outPtr unsafe.Pointer,
	numSubBlocks int,
	quadsPerSubBlk int,
)

//go:noescape
func quantizeGGMLQ8_0RowsAVX512Asm(
	xPtr unsafe.Pointer,
	qxPtr unsafe.Pointer,
	dxPtr unsafe.Pointer,
	numBlocks int,
)

//go:noescape
func quantizeGGMLQ8_KRowsAVX512Asm(
	xPtr unsafe.Pointer,
	qxPtr unsafe.Pointer,
	dxPtr unsafe.Pointer,
	numSuperBlocks int,
	subBlockSize int,
)

// DispatchQuantizedPanelGEMMAVX512VNNI executes K4-interleaved quantized integer GEMM using
// AVX-512 VNNI (VPDPBUSD) 6x32 and 1x32 assembly microkernels with L1d/L2 cache-optimal
// strip-outer tiling and 2D parallel scheduling across both M-blocks and 32-column strip pairs.
func DispatchQuantizedPanelGEMMAVX512VNNI(cfg *gobackend.QuantizedPanelGEMMConfig) {
	const mr = 6
	const nr = 32
	// mcBlocks = 16 (96 rows) keeps Qx + DxPairs well inside per-core L2 cache while reusing each
	// 32-column weight strip pair across up to 96 activation rows in L1d/L2.
	const mcBlocks = 16

	M := cfg.M
	N := cfg.N
	nStart := cfg.NStart
	chunkCols := cfg.ChunkCols
	if M == 0 || chunkCols == 0 {
		return
	}

	numStrips := (chunkCols + gobackend.QuantizedPanelStripCols - 1) / gobackend.QuantizedPanelStripCols
	numStripGroups := (numStrips + 1) / 2
	subBlkBytes := gobackend.QuantizedPanelStripSubBlockBytes(cfg.QuadsPerSubBlk)
	stripBytes := cfg.NumSubBlocks * subBlkBytes
	qxRowBytes := cfg.KPadded
	dxStride := cfg.NumSubBlocks * 2
	dxRowBytes := dxStride * 4
	outRowBytes := N * 4

	runTile := func(mbStart, mbEnd, sgStart, sgEnd int) {
		if mbStart >= mbEnd || sgStart >= sgEnd {
			return
		}
		var biasTail [nr]float32
		var outTail [mr * nr]float32

		colRangeStart := sgStart * nr
		colRangeEnd := min(sgEnd*nr, chunkCols)

		for mb0 := mbStart; mb0 < mbEnd; mb0 += mcBlocks {
			mb1 := min(mb0+mcBlocks, mbEnd)

			// Strip-pair outer loop: keeps the active 32-column weight strip pair (15-61 KB)
			// pinned in L1d/L2 across all M-blocks in [mb0, mb1).
			for sg := sgStart; sg < sgEnd; sg++ {
				s := sg * 2
				colOff := s * gobackend.QuantizedPanelStripCols
				validCols := min(nr, chunkCols-colOff)
				s0Ptr := unsafe.Pointer(&cfg.Panel[s*stripBytes])
				s1Ptr := s0Ptr
				if s+1 < numStrips {
					s1Ptr = unsafe.Pointer(&cfg.Panel[(s+1)*stripBytes])
				}

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

				// Execute all full 6-row M-blocks in [mb0, fullEnd) in a single assembly call
				// when validCols == nr (avoiding per-tile Go->Asm ABI transitions and VZEROUPPER).
				fullEnd := mb1
				if fullEnd > mb0 && min(mr, M-(fullEnd-1)*mr) < mr {
					fullEnd--
				}
				if validCols == nr && fullEnd > mb0 {
					m0 := mb0 * mr
					quantizedPanelKernel6x32AVX512VNNIAsm(
						unsafe.Pointer(&cfg.Qx[m0*qxRowBytes]), qxRowBytes,
						unsafe.Pointer(&cfg.DxPairs[m0*dxStride]), dxRowBytes,
						s0Ptr, s1Ptr,
						bPtr,
						unsafe.Pointer(&cfg.Out[m0*N+nStart+colOff]), outRowBytes,
						cfg.NumSubBlocks, cfg.QuadsPerSubBlk,
						fullEnd-mb0,
					)
				} else {
					fullEnd = mb0
				}

				for mb := fullEnd; mb < mb1; mb++ {
					m0 := mb * mr
					rows := min(mr, M-m0)
					if rows == mr {
						qxBase := unsafe.Pointer(&cfg.Qx[m0*qxRowBytes])
						dxBase := unsafe.Pointer(&cfg.DxPairs[m0*dxStride])
						quantizedPanelKernel6x32AVX512VNNIAsm(
							qxBase, qxRowBytes,
							dxBase, dxRowBytes,
							s0Ptr, s1Ptr,
							bPtr,
							unsafe.Pointer(&outTail[0]), nr*4,
							cfg.NumSubBlocks, cfg.QuadsPerSubBlk,
							1,
						)
						for r := range mr {
							dstOff := (m0+r)*N + nStart + colOff
							copy(cfg.Out[dstOff:dstOff+validCols], outTail[r*nr:r*nr+validCols])
						}
					} else {
						for r := range rows {
							m := m0 + r
							qxRowPtr := unsafe.Pointer(&cfg.Qx[m*qxRowBytes])
							dxRowPtr := unsafe.Pointer(&cfg.DxPairs[m*dxStride])
							if validCols == nr {
								oPtr := unsafe.Pointer(&cfg.Out[m*N+nStart+colOff])
								quantizedPanelKernel1x32AVX512VNNIAsm(
									qxRowPtr, dxRowPtr,
									s0Ptr, s1Ptr,
									bPtr, oPtr,
									cfg.NumSubBlocks, cfg.QuadsPerSubBlk,
								)
							} else {
								quantizedPanelKernel1x32AVX512VNNIAsm(
									qxRowPtr, dxRowPtr,
									s0Ptr, s1Ptr,
									bPtr, unsafe.Pointer(&outTail[0]),
									cfg.NumSubBlocks, cfg.QuadsPerSubBlk,
								)
								dstOff := m*N + nStart + colOff
								copy(cfg.Out[dstOff:dstOff+validCols], outTail[:validCols])
							}
						}
					}
				}
			}

			// Apply elementwise fused activation while Out[mb0*mr..mb1*mr, colRangeStart..colRangeEnd] is hot in L2.
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
			if numMBlocks >= numWorkers || numStripGroups <= 1 {
				if numMBlocks > 1 {
					cfg.Backend.Scheduler.ParallelizeChunks(numMBlocks, 1, func(mbStart, mbEnd int) {
						runTile(mbStart, mbEnd, 0, numStripGroups)
					})
					return
				}
			} else if numMBlocks == 1 {
				// Single M-block (e.g. M == 1..6): parallelize directly across 32-column strip groups.
				cfg.Backend.Scheduler.ParallelizeChunks(numStripGroups, 1, func(sgStart, sgEnd int) {
					runTile(0, 1, sgStart, sgEnd)
				})
				return
			} else {
				// Small M (1 < numMBlocks < numWorkers): 2D tile across (sg, mb) with sg outer so
				// contiguous worker chunks reuse the same weight strip group across adjacent M-blocks.
				totalTiles := numStripGroups * numMBlocks
				cfg.Backend.Scheduler.ParallelizeChunks(totalTiles, 1, func(tStart, tEnd int) {
					for t := tStart; t < tEnd; {
						sg := t / numMBlocks
						mbLo := t % numMBlocks
						sgNextTile := (sg + 1) * numMBlocks
						tGroupEnd := min(tEnd, sgNextTile)
						mbHi := mbLo + (tGroupEnd - t)
						runTile(mbLo, mbHi, sg, sg+1)
						t = tGroupEnd
					}
				})
				return
			}
		}
	}
	runTile(0, numMBlocks, 0, numStripGroups)
}

// QuantizeGGMLActivationsAVX512 quantizes float32 activation rows directly into canonical
// (qx, dxPairs) using handwritten AVX-512 + F16C assembly with in-register reductions.
func QuantizeGGMLActivationsAVX512(
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
		quantizeGGMLQ8_0RowsAVX512Asm(xPtr, qxPtr, dxPtr, rows*numSubBlocks)
		return
	}
	quantizeGGMLQ8_KRowsAVX512Asm(xPtr, qxPtr, dxPtr, rows*(K/256), subBlockSize)
}

func reduceMaxFloat32x16(v archsimd.Float32x16) float32 {
	m8 := v.GetLo().Max(v.GetHi())
	m4 := m8.GetLo().Max(m8.GetHi())
	var buf [4]float32
	m4.StoreArray(&buf)
	return max(max(buf[0], buf[1]), max(buf[2], buf[3]))
}

func reduceSumInt32x16(v archsimd.Int32x16) int32 {
	s8 := v.GetLo().Add(v.GetHi())
	s4 := s8.GetLo().Add(s8.GetHi())
	var buf [4]int32
	s4.StoreArray(&buf)
	return buf[0] + buf[1] + buf[2] + buf[3]
}

// QuantizeSymmetricInt8ActivationsAVX512 quantizes float32 activation rows per kBlock into
// canonical (qx, dxPairs) using AVX-512 SIMD with scalar tail handling for arbitrary kBlock.
func QuantizeSymmetricInt8ActivationsAVX512(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, kBlock, kPadded, numSubBlocks int,
) {
	bPad := (kBlock + 3) &^ 3
	clampMin := archsimd.BroadcastInt32x16(-127)
	clampMax := archsimd.BroadcastInt32x16(127)

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
			if n >= 16 {
				vMax := archsimd.LoadFloat32x16(span[0:16]).Abs()
				for i = 16; i+16 <= n; i += 16 {
					vMax = vMax.Max(archsimd.LoadFloat32x16(span[i : i+16]).Abs())
				}
				maxAbs = reduceMaxFloat32x16(vMax)
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
			vInv := archsimd.BroadcastFloat32x16(invSc)
			var vSum archsimd.Int32x16
			i = 0
			for ; i+16 <= n; i += 16 {
				iv := archsimd.LoadFloat32x16(span[i : i+16]).Mul(vInv).RoundScaled(0).ConvertToInt32().Max(clampMin).Min(clampMax)
				iv.SaturateToInt8().Store(dst[i : i+16])
				vSum = vSum.Add(iv)
			}
			sumQ := reduceSumInt32x16(vSum)
			for ; i < n; i++ {
				var single [16]float32
				single[0] = span[i] * invSc
				q := archsimd.LoadFloat32x16Array(&single).RoundScaled(0).ConvertToInt32().Max(clampMin).Min(clampMax)
				var qArr [4]int32
				q.GetLo().GetLo().StoreArray(&qArr)
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

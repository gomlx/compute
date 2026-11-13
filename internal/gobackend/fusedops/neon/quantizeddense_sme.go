// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/simdarch"
)

// smeQuantizedTask matches the exact memory layout expected by smeQuantizedPanelRowsAsm
// in quantizeddense_sme_arm64.s.
type smeQuantizedTask struct {
	qxPtr            unsafe.Pointer // 0:   &cfg.Qx[mStart*cfg.KPadded]
	dxPairsPtr       unsafe.Pointer // 8:   &cfg.DxPairs[mStart*cfg.NumSubBlocks*2]
	packedQxPtr      unsafe.Pointer // 16:  scratch for [totalQuads, 16]uint32
	packedDxPairsPtr unsafe.Pointer // 24:  scratch for [numSubBlocks*2, 16]float32
	wPanelPtr        unsafe.Pointer // 32:  &cfg.Panel[sStart*stripBytes]
	outPtr           unsafe.Pointer // 40:  &cfg.Out[mStart*cfg.N + nStart + sStart*16]
	biasPtr          unsafe.Pointer // 48:  &cfg.Bias[sStart*16] (or nil)
	totalRows        int            // 56:  mLimit - mStart
	totalQuads       int            // 64:  cfg.KPadded / 4
	numSubBlocks     int            // 72:  cfg.NumSubBlocks
	quadsPerSB       int            // 80:  cfg.QuadsPerSubBlk
	stripBytes       int            // 88:  cfg.NumSubBlocks * subBlkBytes
	numStrips        int            // 96:  sEnd - sStart
	startCol         int            // 104: sStart * 16 (relative to chunkCols)
	chunkCols        int            // 112: cfg.ChunkCols
	outRowBytes      int            // 120: cfg.N * 4
}

//go:noescape
func smeQuantizedPanelRowsAsm(task *smeQuantizedTask)

const smeQuantScratchBytes = 64 * 1024

type smeQuantScratchBuf struct {
	data [smeQuantScratchBytes / 8]uint64
}

var smeQuantScratchPool = sync.Pool{
	New: func() any {
		return new(smeQuantScratchBuf)
	},
}

func acquireSMEQuantScratch(totalQuads, numSubBlocks int) (buf *smeQuantScratchBuf, packedQxPtr, packedDxPairsPtr unsafe.Pointer) {
	qxBytes := ((totalQuads + 15) &^ 15) * 64
	dxBytes := ((numSubBlocks*2 + 15) &^ 15) * 64
	totalBytes := qxBytes + dxBytes
	if totalBytes <= smeQuantScratchBytes {
		buf = smeQuantScratchPool.Get().(*smeQuantScratchBuf)
		base := unsafe.Pointer(&buf.data[0])
		return buf, base, unsafe.Add(base, uintptr(qxBytes))
	}
	words := make([]uint64, (totalBytes+7)/8)
	base := unsafe.Pointer(&words[0])
	return nil, base, unsafe.Add(base, uintptr(qxBytes))
}

func releaseSMEQuantScratch(buf *smeQuantScratchBuf) {
	if buf != nil {
		smeQuantScratchPool.Put(buf)
	}
}

func runQuantizedRowRangeSME(
	cfg *gobackend.QuantizedPanelGEMMConfig,
	mStart, mLimit, sStart, sEnd int,
	packedQxPtr, packedDxPairsPtr unsafe.Pointer,
) {
	if mStart >= mLimit || sStart >= sEnd {
		return
	}
	const nr = gobackend.QuantizedPanelStripCols
	N := cfg.N
	nStart := cfg.NStart
	chunkCols := cfg.ChunkCols
	subBlkBytes := gobackend.QuantizedPanelStripSubBlockBytes(cfg.QuadsPerSubBlk)
	stripBytes := cfg.NumSubBlocks * subBlkBytes
	totalQuads := cfg.KPadded / 4
	dxStride := cfg.NumSubBlocks * 2
	colOff := sStart * nr

	var bPtr unsafe.Pointer
	if cfg.Bias != nil {
		bPtr = unsafe.Pointer(&cfg.Bias[colOff])
	}

	task := smeQuantizedTask{
		qxPtr:            unsafe.Pointer(&cfg.Qx[mStart*cfg.KPadded]),
		dxPairsPtr:       unsafe.Pointer(&cfg.DxPairs[mStart*dxStride]),
		packedQxPtr:      packedQxPtr,
		packedDxPairsPtr: packedDxPairsPtr,
		wPanelPtr:        unsafe.Pointer(&cfg.Panel[sStart*stripBytes]),
		outPtr:           unsafe.Pointer(&cfg.Out[mStart*N+nStart+colOff]),
		biasPtr:          bPtr,
		totalRows:        mLimit - mStart,
		totalQuads:       totalQuads,
		numSubBlocks:     cfg.NumSubBlocks,
		quadsPerSB:       cfg.QuadsPerSubBlk,
		stripBytes:       stripBytes,
		numStrips:        sEnd - sStart,
		startCol:         colOff,
		chunkCols:        chunkCols,
		outRowBytes:      N * 4,
	}
	smeQuantizedPanelRowsAsm(&task)

	colRangeStart := sStart * nr
	colRangeEnd := min(sEnd*nr, chunkCols)
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

// DispatchQuantizedPanelGEMMSME executes K4-interleaved quantized integer GEMM using
// hybrid ARM64 SME2 (SUMOPA + FMOPA 16x16 outer-product microkernel on L2 cluster SME units)
// and per-core ARM64 NEON (USDOT 3x16 microkernel) co-execution over a dual-ended row queue.
func DispatchQuantizedPanelGEMMSME(cfg *gobackend.QuantizedPanelGEMMConfig) {
	M := cfg.M
	chunkCols := cfg.ChunkCols
	if M == 0 || chunkCols == 0 {
		return
	}
	if !gobackend.IsSMEAllowed || !gobackend.IsARM64I8MMAllowed || !simdarch.NEONUseAsm || M < 24 {
		DispatchQuantizedPanelGEMMNEON(cfg)
		return
	}

	numStrips := (chunkCols + gobackend.QuantizedPanelStripCols - 1) / gobackend.QuantizedPanelStripCols
	totalQuads := cfg.KPadded / 4
	totalWork := M * cfg.KPadded * chunkCols

	numWorkers := 1
	if cfg.Backend != nil && cfg.Backend.Scheduler != nil && cfg.Backend.Scheduler.IsEnabled() && totalWork > 8*1024*1024 {
		maxByWork := (totalWork + 4*1024*1024 - 1) / (4 * 1024 * 1024)
		numWorkers = min(cfg.Backend.Scheduler.EffectiveParallelism(), maxByWork)
	}

	if numWorkers <= 1 || M < 64 {
		cl, useSME, _ := simdarch.AcquireSMEWithLimit(simdarch.SMEWorkersPerCluster, simdarch.SMEWorkersPerCluster)
		if useSME {
			scratch, pQx, pDx := acquireSMEQuantScratch(totalQuads, cfg.NumSubBlocks)
			runQuantizedRowRangeSME(cfg, 0, M, 0, numStrips, pQx, pDx)
			releaseSMEQuantScratch(scratch)
			simdarch.ReleaseSME(cl)
			return
		}
		DispatchQuantizedPanelGEMMNEON(cfg)
		return
	}

	wholeChip := cfg.Backend.Scheduler.AcquireWholeChip()
	defer cfg.Backend.Scheduler.ReleaseWholeChip(wholeChip)
	numWorkers = cfg.Backend.Scheduler.UnscaledParallelism()

	// Dual-ended atomic row queue: upper 32 bits = headRow (0..M), lower 32 bits = tailRow (M..0).
	//   - SME workers (holding their L2 cluster's SME lock) pull 16-row blocks from headRow.
	//   - NEON workers (whose L2 cluster SME unit is busy) pull 3-row blocks from tailRow.
	var bounds atomic.Uint64
	bounds.Store(uint64(uint32(M)))
	var smeActive atomic.Int32
	var fastSMEActive atomic.Int32
	var allClaimed atomic.Uint32

	cfg.Backend.Scheduler.SaturateUpTo(numWorkers, func() {
		smeCluster := -1
		useSME := false
		isEff := false
		var scratch *smeQuantScratchBuf
		var pQx, pDx unsafe.Pointer

		releaseMySME := func() {
			if smeCluster >= 0 {
				if !isEff {
					fastSMEActive.Add(-1)
				}
				smeActive.Add(-1)
				simdarch.ReleaseSME(smeCluster)
				smeCluster = -1
				useSME = false
			}
		}
		refreshSME := func() {
			curCl, _, curEff := simdarch.CurrentSMECluster()
			if smeCluster >= 0 && smeCluster == curCl {
				return
			}
			if smeCluster >= 0 {
				releaseMySME()
			}
			isEff = curEff
			cl, ok, eff := simdarch.AcquireSMEWithLimit(simdarch.SMEWorkersPerCluster, simdarch.SMEWorkersPerCluster)
			if ok {
				smeCluster = cl
				useSME = true
				isEff = eff
				smeActive.Add(1)
				if !isEff {
					fastSMEActive.Add(1)
				}
				if pQx == nil {
					scratch, pQx, pDx = acquireSMEQuantScratch(totalQuads, cfg.NumSubBlocks)
				}
			}
		}

		defer func() {
			releaseMySME()
			releaseSMEQuantScratch(scratch)
		}()

		for {
			refreshSME()
			b := bounds.Load()
			head := int(b >> 32)
			tail := int(uint32(b))
			rem := tail - head
			if rem <= 0 {
				allClaimed.Store(1)
				break
			}

			if useSME {
				if isEff && fastSMEActive.Load() > 0 && rem <= 32 {
					releaseMySME()
					break
				}
				var chunk int
				if isEff || rem < 256 {
					chunk = min(16, rem)
				} else {
					chunk = min(32, rem)
				}
				nextHead := head + chunk
				if bounds.CompareAndSwap(b, (uint64(uint32(nextHead))<<32)|uint64(uint32(tail))) {
					if nextHead >= tail {
						allClaimed.Store(1)
					}
					runQuantizedRowRangeSME(cfg, head, nextHead, 0, numStrips, pQx, pDx)
				}
				continue
			}

			// NEON worker path: pull 3-row multiples from tailRow.
			smeCount := int(smeActive.Load())
			if smeCount > 0 {
				if isEff && rem <= 48*smeCount {
					break
				}
				if !isEff && rem <= 16*smeCount {
					break
				}
			}
			var chunk int
			if isEff {
				chunk = min(3, rem)
			} else if rem >= 96 {
				chunk = 12
			} else if rem >= 24 {
				chunk = 6
			} else {
				chunk = min(3, rem)
			}
			nextTail := tail - chunk
			if bounds.CompareAndSwap(b, (uint64(uint32(head))<<32)|uint64(uint32(nextTail))) {
				if head >= nextTail {
					allClaimed.Store(1)
				}
				runQuantizedRowRangeNEON(cfg, nextTail, tail, 0, numStrips, 1)
			}
		}

		releaseMySME()
		for allClaimed.Load() == 0 && smeActive.Load() > 0 {
		}
	})
}

// dispatchQuantizedPanelGEMMSMEOnly is used by unit tests to verify the SME2 SUMOPA+FMOPA
// kernel directly across all row/column/sub-block shapes without hybrid NEON fallback.
func dispatchQuantizedPanelGEMMSMEOnly(cfg *gobackend.QuantizedPanelGEMMConfig) {
	M := cfg.M
	chunkCols := cfg.ChunkCols
	if M == 0 || chunkCols == 0 {
		return
	}
	numStrips := (chunkCols + gobackend.QuantizedPanelStripCols - 1) / gobackend.QuantizedPanelStripCols
	totalQuads := cfg.KPadded / 4
	scratch, pQx, pDx := acquireSMEQuantScratch(totalQuads, cfg.NumSubBlocks)
	defer releaseSMEQuantScratch(scratch)
	runQuantizedRowRangeSME(cfg, 0, M, 0, numStrips, pQx, pDx)
}

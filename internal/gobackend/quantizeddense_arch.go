// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import "unsafe"

// QuantizedPanelStripCols is the column width (16) of a single K4-interleaved weight strip:
// 16 columns * 4 uint8 k-elements = 64 bytes (1 ZMM register on AVX-512, 2 YMM registers on AVX2,
// or 4 128-bit NEON registers on ARM64).
const QuantizedPanelStripCols = 16

// QuantizedPanelStripSubBlockBytes returns the byte size of one sub-block in a 16-column strip:
// quadsPerSubBlk * 64 bytes (u_w) + 64 bytes (d'_w) + 64 bytes (m'_w).
func QuantizedPanelStripSubBlockBytes(quadsPerSubBlk int) int {
	return quadsPerSubBlk*64 + 128
}

// QuantizedPanelGEMMConfig holds all parameters for executing a tiled quantized integer GEMM
// over a prepared activation matrix Qx [M, KPadded] and an unpacked weight panel chunk [ChunkCols].
type QuantizedPanelGEMMConfig struct {
	Backend        *Backend
	Qx             []int8                // [M * KPadded] signed int8 activations
	DxPairs        []float32             // [M * NumSubBlocks * 2] interleaved {dx, dx * sum_qx}
	Panel          []uint8               // [numStrips * StripBytes] K4-interleaved weight strips
	Bias           []float32             // [ChunkCols] float32 bias slice for [NStart : NStart+ChunkCols], or nil
	Out            []float32             // [M * N] output matrix
	M              int                   // Total rows
	N              int                   // Full output row stride
	NStart         int                   // Starting output column of this chunk
	ChunkCols      int                   // Number of valid columns in this chunk (1..N-NStart)
	KPadded        int                   // Padded K stride per row of Qx (NumSubBlocks * QuadsPerSubBlk * 4)
	NumSubBlocks   int                   // Number of K sub-blocks per row
	QuadsPerSubBlk int                   // Number of 4-byte quads per sub-block (B_pad / 4)
	WideUnsigned   bool                  // True if unsigned weight values can exceed 63 (e.g. 8-bit formats)
	ActFn          func(in, out []float32) // Optional in-tile elementwise activation applied in L1 cache
}

// QuantizedPanelGEMMArchFn executes a tiled quantized panel GEMM chunk.
type QuantizedPanelGEMMArchFn func(cfg *QuantizedPanelGEMMConfig)

// QuantizeGGMLActivationsArchFn quantizes float32 rows [mStart:mEnd, K] directly into canonical
// (qx [M, K] int8, dxPairs [M, numSubBlocks*2] float32) using GGML Q8_0 or Q8_K scale semantics.
type QuantizeGGMLActivationsArchFn func(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, subBlockSize int,
	q8KSuperBlocks bool,
)

// QuantizeSymmetricInt8ActivationsArchFn quantizes float32 rows [mStart:mEnd, K] per kBlock into
// canonical (qx [M, kPadded] int8, dxPairs [M, numSubBlocks*2] float32).
type QuantizeSymmetricInt8ActivationsArchFn func(
	xF32 []float32,
	qx []int8,
	dxPairs []float32,
	mStart, mEnd, K, kBlock, kPadded, numSubBlocks int,
)

var (
	quantizedPanelGEMMArchFn   QuantizedPanelGEMMArchFn
	quantizedPanelGEMMPriority RegisterPriority = -1

	quantizeGGMLActArchFn   QuantizeGGMLActivationsArchFn
	quantizeGGMLActPriority RegisterPriority = -1

	quantizeSymInt8ActArchFn   QuantizeSymmetricInt8ActivationsArchFn
	quantizeSymInt8ActPriority RegisterPriority = -1
)

// SetQuantizedPanelGEMMArchDispatcher registers an architecture-specific K4-interleaved quantized panel GEMM kernel.
func SetQuantizedPanelGEMMArchDispatcher(priority RegisterPriority, fn QuantizedPanelGEMMArchFn) {
	if fn != nil && priority > quantizedPanelGEMMPriority {
		quantizedPanelGEMMPriority = priority
		quantizedPanelGEMMArchFn = fn
	}
}

// GetQuantizedPanelGEMMArchDispatcher returns the highest-priority registered panel GEMM kernel, or nil.
func GetQuantizedPanelGEMMArchDispatcher() QuantizedPanelGEMMArchFn {
	return quantizedPanelGEMMArchFn
}

// SetQuantizeGGMLActivationsArchDispatcher registers an architecture-specific SIMD activation quantizer for GGML.
func SetQuantizeGGMLActivationsArchDispatcher(priority RegisterPriority, fn QuantizeGGMLActivationsArchFn) {
	if fn != nil && priority > quantizeGGMLActPriority {
		quantizeGGMLActPriority = priority
		quantizeGGMLActArchFn = fn
	}
}

// GetQuantizeGGMLActivationsArchDispatcher returns the highest-priority registered GGML activation quantizer, or nil.
func GetQuantizeGGMLActivationsArchDispatcher() QuantizeGGMLActivationsArchFn {
	return quantizeGGMLActArchFn
}

// SetQuantizeSymmetricInt8ActivationsArchDispatcher registers an architecture-specific SIMD activation quantizer
// for symmetric per-block Int8.
func SetQuantizeSymmetricInt8ActivationsArchDispatcher(priority RegisterPriority, fn QuantizeSymmetricInt8ActivationsArchFn) {
	if fn != nil && priority > quantizeSymInt8ActPriority {
		quantizeSymInt8ActPriority = priority
		quantizeSymInt8ActArchFn = fn
	}
}

// GetQuantizeSymmetricInt8ActivationsArchDispatcher returns the highest-priority symmetric Int8 activation quantizer, or nil.
func GetQuantizeSymmetricInt8ActivationsArchDispatcher() QuantizeSymmetricInt8ActivationsArchFn {
	return quantizeSymInt8ActArchFn
}

// ExecQuantizedPanelGEMMScalar is the portable reference implementation of the K4-interleaved
// quantized panel GEMM.
func ExecQuantizedPanelGEMMScalar(cfg *QuantizedPanelGEMMConfig) {
	M := cfg.M
	N := cfg.N
	nStart := cfg.NStart
	chunkCols := cfg.ChunkCols
	numStrips := (chunkCols + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
	subBlkBytes := QuantizedPanelStripSubBlockBytes(cfg.QuadsPerSubBlk)
	stripBytes := cfg.NumSubBlocks * subBlkBytes
	quads := cfg.QuadsPerSubBlk
	dwOff := quads * 64
	mwOff := dwOff + 64

	runRows := func(mStart, mEnd int) {
		for m := mStart; m < mEnd; m++ {
			qxRow := cfg.Qx[m*cfg.KPadded : (m+1)*cfg.KPadded]
			dxRow := cfg.DxPairs[m*cfg.NumSubBlocks*2 : (m+1)*cfg.NumSubBlocks*2]
			outRow := cfg.Out[m*N+nStart : m*N+nStart+chunkCols]

			for s := range numStrips {
				colOff := s * QuantizedPanelStripCols
				validCols := min(QuantizedPanelStripCols, chunkCols-colOff)
				strip := cfg.Panel[s*stripBytes : (s+1)*stripBytes]

				var accF32 [QuantizedPanelStripCols]float32
				for b := range cfg.NumSubBlocks {
					sb := strip[b*subBlkBytes : (b+1)*subBlkBytes]
					dw := unsafe.Slice((*float32)(unsafe.Pointer(&sb[dwOff])), QuantizedPanelStripCols)
					mw := unsafe.Slice((*float32)(unsafe.Pointer(&sb[mwOff])), QuantizedPanelStripCols)
					dx := dxRow[2*b]
					dxSum := dxRow[2*b+1]
					qxSub := qxRow[b*quads*4 : (b+1)*quads*4]

					var dot [QuantizedPanelStripCols]int32
					for q := range quads {
						qOff := q * 64
						x0 := int32(qxSub[4*q+0])
						x1 := int32(qxSub[4*q+1])
						x2 := int32(qxSub[4*q+2])
						x3 := int32(qxSub[4*q+3])
						for c := range validCols {
							wOff := qOff + 4*c
							dot[c] += int32(sb[wOff+0])*x0 +
								int32(sb[wOff+1])*x1 +
								int32(sb[wOff+2])*x2 +
								int32(sb[wOff+3])*x3
						}
					}
					for c := range validCols {
						accF32[c] += (dx*dw[c])*float32(dot[c]) + dxSum*mw[c]
					}
				}
				if cfg.Bias != nil {
					for c := range validCols {
						accF32[c] += cfg.Bias[colOff+c]
					}
				}
				copy(outRow[colOff:colOff+validCols], accF32[:validCols])
			}
			if cfg.ActFn != nil {
				cfg.ActFn(outRow, outRow)
			}
		}
	}

	totalWork := M * cfg.KPadded * chunkCols
	if cfg.Backend != nil && cfg.Backend.Scheduler != nil && cfg.Backend.Scheduler.IsEnabled() &&
		totalWork > 4096 && M > 1 {
		cfg.Backend.Scheduler.ParallelizeChunks(M, 1, runRows)
	} else {
		runRows(0, M)
	}
}

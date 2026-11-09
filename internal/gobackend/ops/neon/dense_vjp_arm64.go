// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"runtime"
	"simd/archsimd"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	matmulneon "github.com/gomlx/compute/internal/gobackend/dot/matmul/neon"
	"github.com/gomlx/compute/shapes"
)

// ExecTallSkinnyAtBFloat32ARM64 computes out = a^T @ b ([p, q]) directly from row-major
// matrices a ([m, p]) and b ([m, q]) when the contracting dimension m is tall and the
// output matrix [p, q] is small enough to stay L2-resident across K-split workers.
// Uses hybrid ARM64 SME 32x32 outer-product tiles (ConvKernelGradSME32x32Float32Asm)
// and per-core NEON 4x16 / 1x16 outer-product microkernels (ConvKernelGradNEON4x16Float32Asm,
// ConvKernelGradNEON1x16Float32Asm) without transposing or packing a or b.
func ExecTallSkinnyAtBFloat32ARM64(
	backend *gobackend.Backend,
	a, b []float32,
	m, p, q int,
	out []float32,
) bool {
	if !gobackend.IsNEONAllowed ||
		m < 64 || p <= 0 || q <= 0 || p*q > 131072 || m < max(p, q) {
		return false
	}

	outSize := p * q
	clear(out[:outSize])

	useSME := gobackend.IsSMEAllowed && p >= 4 && q >= 8
	rowGroupSize := 16
	colPanelSize := 16
	if useSME {
		rowGroupSize = 32
		colPanelSize = 32
	}

	numRowGroups := (p + rowGroupSize - 1) / rowGroupSize
	numColPanels := (q + colPanelSize - 1) / colPanelSize
	numSpatialTiles := numRowGroups * numColPanels

	maxParallelism := 1
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && m*p*q >= 16384 {
		maxParallelism = backend.Scheduler.AdjustedMaxParallelism()
	}

	numSlices := 1
	if maxParallelism > 1 && m >= 512 {
		targetSlices := max(1, (maxParallelism+numSpatialTiles-1)/numSpatialTiles)
		maxSlicesByM := m / 256
		maxSlicesByMem := 1 + 524288/max(1, outSize)
		numSlices = min(targetSlices, min(maxParallelism, min(maxSlicesByM, maxSlicesByMem)))
		numSlices = max(1, numSlices)
	}

	var scratchBuf *gobackend.Buffer
	var scratchOut []float32
	if numSlices > 1 {
		var err error
		scratchBuf, err = backend.GetBuffer(shapes.Make(dtypes.Float32, (numSlices-1)*outSize))
		if err != nil {
			numSlices = 1
		} else {
			defer backend.PutBuffer(scratchBuf)
			scratchOut = scratchBuf.Flat.([]float32)[:(numSlices-1)*outSize]
			clear(scratchOut)
		}
	}

	totalTasks := numSlices * numSpatialTiles
	const mBlockNEON = 512
	const mBlockSME = 4096

	processTaskNEON := func(taskIdx int) {
		sliceIdx := taskIdx / numSpatialTiles
		rem := taskIdx % numSpatialTiles
		rowGroup := rem / numColPanels
		colPanel := rem % numColPanels

		mStart := (sliceIdx * m) / numSlices
		mEnd := ((sliceIdx + 1) * m) / numSlices
		if mStart >= mEnd {
			return
		}

		pStart := rowGroup * rowGroupSize
		validRows := min(rowGroupSize, p-pStart)
		qStart := colPanel * colPanelSize
		validCols := min(colPanelSize, q-qStart)

		var dstOut []float32
		if sliceIdx == 0 {
			dstOut = out
		} else {
			dstOut = scratchOut[(sliceIdx-1)*outSize : sliceIdx*outSize]
		}

		for m0 := mStart; m0 < mEnd; m0 += mBlockNEON {
			mCount := min(mBlockNEON, mEnd-m0)
			tallSkinnyAtBTileNEONFloat32(a, b, dstOut, p, q, m0, mCount, pStart, validRows, qStart, validCols)
		}
	}

	processTaskSME := func(taskIdx int) {
		sliceIdx := taskIdx / numSpatialTiles
		rem := taskIdx % numSpatialTiles
		rowGroup := rem / numColPanels
		colPanel := rem % numColPanels

		mStart := (sliceIdx * m) / numSlices
		mEnd := ((sliceIdx + 1) * m) / numSlices
		if mStart >= mEnd {
			return
		}

		pStart := rowGroup * rowGroupSize
		validRows := min(rowGroupSize, p-pStart)
		qStart := colPanel * colPanelSize
		validCols := min(colPanelSize, q-qStart)

		var dstOut []float32
		if sliceIdx == 0 {
			dstOut = out
		} else {
			dstOut = scratchOut[(sliceIdx-1)*outSize : sliceIdx*outSize]
		}

		outIdx := pStart*q + qStart
		acc := 0
		for m0 := mStart; m0 < mEnd; m0 += mBlockSME {
			mCount := min(mBlockSME, mEnd-m0)
			aBaseIdx := m0*p + pStart
			bBaseIdx := m0*q + qStart
			ConvKernelGradSME32x32Float32Asm(
				unsafe.Pointer(&a[aBaseIdx]),
				unsafe.Pointer(&b[bBaseIdx]),
				unsafe.Pointer(&dstOut[outIdx]),
				1, 1, mCount,
				0, 0, 0, 0,
				p*4, q*4, q*4,
				validRows, validCols, acc,
			)
			acc = 1
		}
	}

	maxWorkers := min(totalTasks, maxParallelism)
	if maxWorkers <= 1 {
		for taskIdx := range totalTasks {
			if useSME {
				processTaskSME(taskIdx)
			} else {
				processTaskNEON(taskIdx)
			}
		}
	} else if useSME {
		var nextTask atomic.Int64
		var activeFastSME atomic.Int32
		backend.Scheduler.SaturateUpTo(maxWorkers, func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			clusterID, hasSME, isSlowECore := matmulneon.SMEAcquireCluster()
			if hasSME {
				if clusterID == 0 || clusterID == 1 {
					activeFastSME.Add(1)
				}
				defer func() {
					if clusterID == 0 || clusterID == 1 {
						activeFastSME.Add(-1)
					}
					matmulneon.SMEReleaseCluster(clusterID)
				}()
			} else if isSlowECore {
				return
			}

			for {
				if !hasSME || clusterID == 2 {
					rem := totalTasks - int(nextTask.Load())
					fastSME := int(activeFastSME.Load())
					if fastSME > 0 && rem <= 2*fastSME {
						break
					}
				}
				taskIdx := int(nextTask.Add(1) - 1)
				if taskIdx >= totalTasks {
					break
				}
				if hasSME {
					processTaskSME(taskIdx)
				} else {
					processTaskNEON(taskIdx)
				}
			}
		})
	} else {
		var nextTask atomic.Int64
		backend.Scheduler.SaturateUpTo(maxWorkers, func() {
			for {
				taskIdx := int(nextTask.Add(1) - 1)
				if taskIdx >= totalTasks {
					break
				}
				processTaskNEON(taskIdx)
			}
		})
	}

	if numSlices > 1 {
		for s := range numSlices - 1 {
			src := scratchOut[s*outSize : (s+1)*outSize]
			addFloat32SliceNEON(out[:outSize], src)
		}
	}
	return true
}

func tallSkinnyAtBTileNEONFloat32(
	a, b, out []float32,
	p, q, m0, mCount, pStart, validRows, qStart, validCols int,
) {
	colOff := 0
	for ; colOff+16 <= validCols; colOff += 16 {
		cOut := qStart + colOff
		bBaseIdx := m0*q + cOut
		rOff := 0
		for ; rOff+4 <= validRows; rOff += 4 {
			cIn := pStart + rOff
			aBaseIdx := m0*p + cIn
			outIdx := cIn*q + cOut
			ConvKernelGradNEON4x16Float32Asm(
				unsafe.Pointer(&a[aBaseIdx]),
				unsafe.Pointer(&b[bBaseIdx]),
				unsafe.Pointer(&out[outIdx]),
				1, 1, mCount,
				0, 0, 0, 0,
				p*4, q*4, q*4,
			)
		}
		for ; rOff < validRows; rOff++ {
			cIn := pStart + rOff
			aBaseIdx := m0*p + cIn
			outIdx := cIn*q + cOut
			ConvKernelGradNEON1x16Float32Asm(
				unsafe.Pointer(&a[aBaseIdx]),
				unsafe.Pointer(&b[bBaseIdx]),
				unsafe.Pointer(&out[outIdx]),
				1, 1, mCount,
				0, 0, 0, 0,
				p*4, q*4,
			)
		}
	}

	for ; colOff+4 <= validCols; colOff += 4 {
		cOut := qStart + colOff
		rOff := 0
		for ; rOff+4 <= validRows; rOff += 4 {
			cIn := pStart + rOff
			acc0 := archsimd.BroadcastFloat32x4(0)
			acc1 := archsimd.BroadcastFloat32x4(0)
			acc2 := archsimd.BroadcastFloat32x4(0)
			acc3 := archsimd.BroadcastFloat32x4(0)
			aP := m0*p + cIn
			bP := m0*q + cOut
			for range mCount {
				vb := archsimd.LoadFloat32x4(b[bP : bP+4])
				acc0 = vb.MulAdd(archsimd.BroadcastFloat32x4(a[aP+0]), acc0)
				acc1 = vb.MulAdd(archsimd.BroadcastFloat32x4(a[aP+1]), acc1)
				acc2 = vb.MulAdd(archsimd.BroadcastFloat32x4(a[aP+2]), acc2)
				acc3 = vb.MulAdd(archsimd.BroadcastFloat32x4(a[aP+3]), acc3)
				aP += p
				bP += q
			}
			outOff := cIn*q + cOut
			acc0.Add(archsimd.LoadFloat32x4(out[outOff : outOff+4])).Store(out[outOff : outOff+4])
			acc1.Add(archsimd.LoadFloat32x4(out[outOff+q : outOff+q+4])).Store(out[outOff+q : outOff+q+4])
			acc2.Add(archsimd.LoadFloat32x4(out[outOff+2*q : outOff+2*q+4])).Store(out[outOff+2*q : outOff+2*q+4])
			acc3.Add(archsimd.LoadFloat32x4(out[outOff+3*q : outOff+3*q+4])).Store(out[outOff+3*q : outOff+3*q+4])
		}
		for ; rOff < validRows; rOff++ {
			cIn := pStart + rOff
			acc0 := archsimd.BroadcastFloat32x4(0)
			aP := m0*p + cIn
			bP := m0*q + cOut
			for range mCount {
				vb := archsimd.LoadFloat32x4(b[bP : bP+4])
				acc0 = vb.MulAdd(archsimd.BroadcastFloat32x4(a[aP]), acc0)
				aP += p
				bP += q
			}
			outOff := cIn*q + cOut
			acc0.Add(archsimd.LoadFloat32x4(out[outOff : outOff+4])).Store(out[outOff : outOff+4])
		}
	}

	for ; colOff < validCols; colOff++ {
		cOut := qStart + colOff
		for rOff := range validRows {
			cIn := pStart + rOff
			var sum float32
			aP := m0*p + cIn
			bP := m0*q + cOut
			for range mCount {
				sum += a[aP] * b[bP]
				aP += p
				bP += q
			}
			out[cIn*q+cOut] += sum
		}
	}
}

// ReduceBiasFloat32ARM64 reduces dZ of shape [m, n] along the m dimension into dBias of shape [n]
// using parallel row-chunking and 4x-unrolled ARM64 NEON vector accumulation.
func ReduceBiasFloat32ARM64(backend *gobackend.Backend, dZ []float32, m, n int, dBias []float32) bool {
	if !gobackend.IsNEONAllowed || n > 4096 {
		return false
	}
	const minParallelWork = 16384
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && m*n >= minParallelWork && m >= 64 {
		clear(dBias[:n])
		rowsPerChunk := max(64, minParallelWork/max(n, 1))
		var mu sync.Mutex
		backend.Scheduler.ParallelizeChunks(m, rowsPerChunk, func(r0, rEnd int) {
			var stackAcc [4096]float32
			acc := stackAcc[:n]
			copy(acc, dZ[r0*n:(r0+1)*n])
			for r := r0 + 1; r < rEnd; r++ {
				addFloat32SliceNEON(acc, dZ[r*n:(r+1)*n])
			}
			mu.Lock()
			addFloat32SliceNEON(dBias[:n], acc)
			mu.Unlock()
		})
		return true
	}

	copy(dBias[:n], dZ[:n])
	for r := 1; r < m; r++ {
		addFloat32SliceNEON(dBias[:n], dZ[r*n:(r+1)*n])
	}
	return true
}

func addFloat32SliceNEON(dst, src []float32) {
	n := len(dst)
	i := 0
	for ; i+16 <= n; i += 16 {
		d0 := archsimd.LoadFloat32x4(dst[i : i+4])
		d1 := archsimd.LoadFloat32x4(dst[i+4 : i+8])
		d2 := archsimd.LoadFloat32x4(dst[i+8 : i+12])
		d3 := archsimd.LoadFloat32x4(dst[i+12 : i+16])
		s0 := archsimd.LoadFloat32x4(src[i : i+4])
		s1 := archsimd.LoadFloat32x4(src[i+4 : i+8])
		s2 := archsimd.LoadFloat32x4(src[i+8 : i+12])
		s3 := archsimd.LoadFloat32x4(src[i+12 : i+16])
		d0.Add(s0).Store(dst[i : i+4])
		d1.Add(s1).Store(dst[i+4 : i+8])
		d2.Add(s2).Store(dst[i+8 : i+12])
		d3.Add(s3).Store(dst[i+12 : i+16])
	}
	for ; i+4 <= n; i += 4 {
		vd := archsimd.LoadFloat32x4(dst[i : i+4])
		vs := archsimd.LoadFloat32x4(src[i : i+4])
		vd.Add(vs).Store(dst[i : i+4])
	}
	for ; i < n; i++ {
		dst[i] += src[i]
	}
}

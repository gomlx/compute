// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"simd/archsimd"
	"unsafe"
)

var (
	transpose32BitIndices0 = [16]uint32{
		0, 16, 0, 0, // Strip 0
		1, 17, 0, 0, // Strip 1
		2, 18, 0, 0, // Strip 2
		3, 19, 0, 0, // Strip 3
	}
	transpose32BitIndices1 = [16]uint32{
		0 + 4, 16 + 4, 0, 0, // Strip 0
		1 + 4, 17 + 4, 0, 0, // Strip 1
		2 + 4, 18 + 4, 0, 0, // Strip 2
		3 + 4, 19 + 4, 0, 0, // Strip 3
	}
	transpose32BitIndices2 = [16]uint32{
		0 + 8, 16 + 8, 0, 0, // Strip 0
		1 + 8, 17 + 8, 0, 0, // Strip 1
		2 + 8, 18 + 8, 0, 0, // Strip 2
		3 + 8, 19 + 8, 0, 0, // Strip 3
	}
	transpose32BitIndices3 = [16]uint32{
		0 + 12, 16 + 12, 0, 0, // Strip 0
		1 + 12, 17 + 12, 0, 0, // Strip 1
		2 + 12, 18 + 12, 0, 0, // Strip 2
		3 + 12, 19 + 12, 0, 0, // Strip 3
	}

	transpose16BitIndices0 = [32]uint16{
		0, 32, 1, 33, 0, 0, 0, 0, // Lane 0: cols 0,1
		2, 34, 3, 35, 0, 0, 0, 0, // Lane 1: cols 2,3
		4, 36, 5, 37, 0, 0, 0, 0, // Lane 2: cols 4,5
		6, 38, 7, 39, 0, 0, 0, 0, // Lane 3: cols 6,7
	}
	transpose16BitIndices1 = [32]uint16{
		0 + 8, 32 + 8, 1 + 8, 33 + 8, 0, 0, 0, 0, // Lane 0: cols 8,9
		2 + 8, 34 + 8, 3 + 8, 35 + 8, 0, 0, 0, 0, // Lane 1: cols 10,11
		4 + 8, 36 + 8, 5 + 8, 37 + 8, 0, 0, 0, 0, // Lane 2: cols 12,13
		6 + 8, 38 + 8, 7 + 8, 39 + 8, 0, 0, 0, 0, // Lane 3: cols 14,15
	}
	transpose16BitIndices2 = [32]uint16{
		0 + 16, 32 + 16, 1 + 16, 33 + 16, 0, 0, 0, 0, // Lane 0: cols 16,17
		2 + 16, 34 + 16, 3 + 16, 35 + 16, 0, 0, 0, 0, // Lane 1: cols 18,19
		4 + 16, 36 + 16, 5 + 16, 37 + 16, 0, 0, 0, 0, // Lane 2: cols 20,21
		6 + 16, 38 + 16, 7 + 16, 39 + 16, 0, 0, 0, 0, // Lane 3: cols 22,23
	}
	transpose16BitIndices3 = [32]uint16{
		0 + 24, 32 + 24, 1 + 24, 33 + 24, 0, 0, 0, 0, // Lane 0: cols 24,25
		2 + 24, 34 + 24, 3 + 24, 35 + 24, 0, 0, 0, 0, // Lane 1: cols 26,27
		4 + 24, 36 + 24, 5 + 24, 37 + 24, 0, 0, 0, 0, // Lane 2: cols 28,29
		6 + 24, 38 + 24, 7 + 24, 39 + 24, 0, 0, 0, 0, // Lane 3: cols 30,31
	}

	transpose64BitIndices0 = [8]uint64{0, 0, 8, 0, 1, 0, 9, 0}
	transpose64BitIndices1 = [8]uint64{2, 0, 10, 0, 3, 0, 11, 0}
	transpose64BitIndices2 = [8]uint64{4, 0, 12, 0, 5, 0, 13, 0}
	transpose64BitIndices3 = [8]uint64{6, 0, 14, 0, 7, 0, 15, 0}
)

// avx512Transpose4x16x32bits transposes 4 rows of 16x32-bit elements (uint32) into
// 16 rows of 4 32 bit elements, where each 4 rows of 4 elements is output together in one ZMM register.
func avx512Transpose4x16x32bits(v0, v1, v2, v3 archsimd.Uint32x16) (row0to4, row4to8, row8to12, row12to16 archsimd.Uint32x16) {
	{
		t00 := v0.ConcatPermute(v2, archsimd.LoadUint32x16Array(&transpose32BitIndices0))
		t01 := v1.ConcatPermute(v3, archsimd.LoadUint32x16Array(&transpose32BitIndices0))
		row0to4 = t00.InterleaveLoGrouped(t01)
	}
	{
		t10 := v0.ConcatPermute(v2, archsimd.LoadUint32x16Array(&transpose32BitIndices1))
		t11 := v1.ConcatPermute(v3, archsimd.LoadUint32x16Array(&transpose32BitIndices1))
		row4to8 = t10.InterleaveLoGrouped(t11)
	}
	{
		t20 := v0.ConcatPermute(v2, archsimd.LoadUint32x16Array(&transpose32BitIndices2))
		t21 := v1.ConcatPermute(v3, archsimd.LoadUint32x16Array(&transpose32BitIndices2))
		row8to12 = t20.InterleaveLoGrouped(t21)
	}
	{
		t30 := v0.ConcatPermute(v2, archsimd.LoadUint32x16Array(&transpose32BitIndices3))
		t31 := v1.ConcatPermute(v3, archsimd.LoadUint32x16Array(&transpose32BitIndices3))
		row12to16 = t30.InterleaveLoGrouped(t31)
	}
	return
}

// avx512Transpose4x32x16bits transposes 4 rows of 32x16-bit elements (uint16) into
// 32 rows of 4 16-bit elements, where each 8 rows of 4 elements is output together in one ZMM register.
func avx512Transpose4x32x16bits(v0, v1, v2, v3 archsimd.Uint16x32) (row0to8, row8to16, row16to24, row24to32 archsimd.Uint16x32) {
	{
		t00 := v0.ConcatPermute(v2, archsimd.LoadUint16x32Array(&transpose16BitIndices0))
		t01 := v1.ConcatPermute(v3, archsimd.LoadUint16x32Array(&transpose16BitIndices0))
		row0to8 = t00.InterleaveLoGrouped(t01)
	}
	{
		t10 := v0.ConcatPermute(v2, archsimd.LoadUint16x32Array(&transpose16BitIndices1))
		t11 := v1.ConcatPermute(v3, archsimd.LoadUint16x32Array(&transpose16BitIndices1))
		row8to16 = t10.InterleaveLoGrouped(t11)
	}
	{
		t20 := v0.ConcatPermute(v2, archsimd.LoadUint16x32Array(&transpose16BitIndices2))
		t21 := v1.ConcatPermute(v3, archsimd.LoadUint16x32Array(&transpose16BitIndices2))
		row16to24 = t20.InterleaveLoGrouped(t21)
	}
	{
		t30 := v0.ConcatPermute(v2, archsimd.LoadUint16x32Array(&transpose16BitIndices3))
		t31 := v1.ConcatPermute(v3, archsimd.LoadUint16x32Array(&transpose16BitIndices3))
		row24to32 = t30.InterleaveLoGrouped(t31)
	}
	return
}

// avx512Transpose4x8x64bits transposes 4 rows of 8x64-bit elements (uint64) into
// 8 rows of 4 64-bit elements, where each 2 rows of 4 elements is output together in one ZMM register.
func avx512Transpose4x8x64bits(v0, v1, v2, v3 archsimd.Uint64x8) (row0to2, row2to4, row4to6, row6to8 archsimd.Uint64x8) {
	{
		t00 := v0.ConcatPermute(v2, archsimd.LoadUint64x8Array(&transpose64BitIndices0))
		t01 := v1.ConcatPermute(v3, archsimd.LoadUint64x8Array(&transpose64BitIndices0))
		row0to2 = t00.InterleaveLoGrouped(t01)
	}
	{
		t10 := v0.ConcatPermute(v2, archsimd.LoadUint64x8Array(&transpose64BitIndices1))
		t11 := v1.ConcatPermute(v3, archsimd.LoadUint64x8Array(&transpose64BitIndices1))
		row2to4 = t10.InterleaveLoGrouped(t11)
	}
	{
		t20 := v0.ConcatPermute(v2, archsimd.LoadUint64x8Array(&transpose64BitIndices2))
		t21 := v1.ConcatPermute(v3, archsimd.LoadUint64x8Array(&transpose64BitIndices2))
		row4to6 = t20.InterleaveLoGrouped(t21)
	}
	{
		t30 := v0.ConcatPermute(v2, archsimd.LoadUint64x8Array(&transpose64BitIndices3))
		t31 := v1.ConcatPermute(v3, archsimd.LoadUint64x8Array(&transpose64BitIndices3))
		row6to8 = t30.InterleaveLoGrouped(t31)
	}
	return
}

// avx512PackLHSStrided8x16Uint32 packs full strips of height kernelRows (where kernelRows is a multiple of 8)
// of 32-bit elements using 512-bit ZMM 8x16 SIMD register transposes.
func avx512PackLHSStrided8x16Uint32(
	lhsBasePtr, panelBasePtr uintptr,
	lhsRowStart, lhsColStart, lhsCols,
	fullRows, contractingCols, kernelRows int,
) {
	lhsStrideBytes := uintptr(lhsCols) * 4
	lhsColStartBytes := uintptr(lhsColStart) * 4
	panelColStrideBytes := uintptr(kernelRows) * 4
	stripPanelBytes := uintptr(contractingCols) * panelColStrideBytes

	storePair := func(top, bot archsimd.Uint32x8, dst uintptr) {
		top.SetHi(bot.GetLo()).StoreArray((*[8]uint32)(unsafe.Pointer(dst)))
		bot.SetLo(top.GetHi()).StoreArray((*[8]uint32)(unsafe.Pointer(dst + panelColStrideBytes)))
	}

	for stripRowIdx := 0; stripRowIdx < fullRows; stripRowIdx += kernelRows {
		stripPanelBase := panelBasePtr + uintptr(stripRowIdx/kernelRows)*stripPanelBytes
		for subRow := 0; subRow < kernelRows; subRow += 8 {
			r0Ptr := lhsBasePtr + uintptr(lhsRowStart+stripRowIdx+subRow)*lhsStrideBytes + lhsColStartBytes
			r1Ptr := r0Ptr + lhsStrideBytes
			r2Ptr := r1Ptr + lhsStrideBytes
			r3Ptr := r2Ptr + lhsStrideBytes
			r4Ptr := r3Ptr + lhsStrideBytes
			r5Ptr := r4Ptr + lhsStrideBytes
			r6Ptr := r5Ptr + lhsStrideBytes
			r7Ptr := r6Ptr + lhsStrideBytes
			pDst := stripPanelBase + uintptr(subRow)*4

			col := 0
			for ; col+16 <= contractingCols; col += 16 {
				v0 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r0Ptr)))
				v1 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r1Ptr)))
				v2 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r2Ptr)))
				v3 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r3Ptr)))
				top0to4, top4to8, top8to12, top12to16 := avx512Transpose4x16x32bits(v0, v1, v2, v3)

				v4 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r4Ptr)))
				v5 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r5Ptr)))
				v6 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r6Ptr)))
				v7 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(r7Ptr)))
				bot0to4, bot4to8, bot8to12, bot12to16 := avx512Transpose4x16x32bits(v4, v5, v6, v7)

				storePair(top0to4.GetLo(), bot0to4.GetLo(), pDst+0*panelColStrideBytes)
				storePair(top0to4.GetHi(), bot0to4.GetHi(), pDst+2*panelColStrideBytes)
				storePair(top4to8.GetLo(), bot4to8.GetLo(), pDst+4*panelColStrideBytes)
				storePair(top4to8.GetHi(), bot4to8.GetHi(), pDst+6*panelColStrideBytes)
				storePair(top8to12.GetLo(), bot8to12.GetLo(), pDst+8*panelColStrideBytes)
				storePair(top8to12.GetHi(), bot8to12.GetHi(), pDst+10*panelColStrideBytes)
				storePair(top12to16.GetLo(), bot12to16.GetLo(), pDst+12*panelColStrideBytes)
				storePair(top12to16.GetHi(), bot12to16.GetHi(), pDst+14*panelColStrideBytes)

				r0Ptr += 64
				r1Ptr += 64
				r2Ptr += 64
				r3Ptr += 64
				r4Ptr += 64
				r5Ptr += 64
				r6Ptr += 64
				r7Ptr += 64
				pDst += 16 * panelColStrideBytes
			}
			for ; col < contractingCols; col++ {
				*(*[8]uint32)(unsafe.Pointer(pDst)) = [8]uint32{
					*(*uint32)(unsafe.Pointer(r0Ptr)),
					*(*uint32)(unsafe.Pointer(r1Ptr)),
					*(*uint32)(unsafe.Pointer(r2Ptr)),
					*(*uint32)(unsafe.Pointer(r3Ptr)),
					*(*uint32)(unsafe.Pointer(r4Ptr)),
					*(*uint32)(unsafe.Pointer(r5Ptr)),
					*(*uint32)(unsafe.Pointer(r6Ptr)),
					*(*uint32)(unsafe.Pointer(r7Ptr)),
				}
				r0Ptr += 4
				r1Ptr += 4
				r2Ptr += 4
				r3Ptr += 4
				r4Ptr += 4
				r5Ptr += 4
				r6Ptr += 4
				r7Ptr += 4
				pDst += panelColStrideBytes
			}
		}
	}
}

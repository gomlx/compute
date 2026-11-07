// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"simd/archsimd"
	"unsafe"
)

// avx2Transpose4x8x32bits transposes 4 rows of 8x32-bit elements (uint32) into
// 8 rows of 4 32-bit elements, where each 2 rows of 4 elements is output together in one YMM register.
func avx2Transpose4x8x32bits(v0, v1, v2, v3 archsimd.Uint32x8) (row0to2, row2to4, row4to6, row6to8 archsimd.Uint32x8) {
	t0 := v0.InterleaveLoGrouped(v1) // [a0 b0 a1 b1 | a4 b4 a5 b5]
	t1 := v2.InterleaveLoGrouped(v3) // [c0 d0 c1 d1 | c4 d4 c5 d5]
	t2 := v0.InterleaveHiGrouped(v1) // [a2 b2 a3 b3 | a6 b6 a7 b7]
	t3 := v2.InterleaveHiGrouped(v3) // [c2 d2 c3 d3 | c6 d6 c7 d7]

	idxLo := archsimd.LoadUint32x8Array(&[8]uint32{0, 1, 0, 1, 2, 3, 2, 3})
	idxHi := archsimd.LoadUint32x8Array(&[8]uint32{4, 5, 4, 5, 6, 7, 6, 7})
	mask := archsimd.LoadUint32x8Array(&[8]uint32{0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF}).Greater(archsimd.BroadcastUint32x8(0))

	row0to2 = t1.Permute(idxLo).Merge(t0.Permute(idxLo), mask)
	row2to4 = t3.Permute(idxLo).Merge(t2.Permute(idxLo), mask)
	row4to6 = t1.Permute(idxHi).Merge(t0.Permute(idxHi), mask)
	row6to8 = t3.Permute(idxHi).Merge(t2.Permute(idxHi), mask)
	return
}

// avx2Transpose4x16x16bits transposes 4 rows of 16x16-bit elements (uint16) into
// 16 rows of 4 16-bit elements, where each 4 rows of 4 elements is output together in one YMM register.
func avx2Transpose4x16x16bits(v0, v1, v2, v3 archsimd.Uint16x16) (row0to4, row4to8, row8to12, row12to16 archsimd.Uint16x16) {
	t0 := v0.InterleaveLoGrouped(v1) // [a0 b0 a1 b1 a2 b2 a3 b3 | a8 b8 a9 b9 a10 b10 a11 b11]
	t1 := v2.InterleaveLoGrouped(v3) // [c0 d0 c1 d1 c2 d2 c3 d3 | c8 d8 c9 d9 c10 d10 c11 d11]
	t2 := v0.InterleaveHiGrouped(v1) // [a4 b4 a5 b5 a6 b6 a7 b7 | a12 b12 a13 b13 a14 b14 a15 b15]
	t3 := v2.InterleaveHiGrouped(v3) // [c4 d4 c5 d5 c6 d6 c7 d7 | c12 d12 c13 d13 c14 d14 c15 d15]

	t0_32 := t0.AsUint32x8()
	t1_32 := t1.AsUint32x8()
	t2_32 := t2.AsUint32x8()
	t3_32 := t3.AsUint32x8()

	x0 := t0_32.InterleaveLoGrouped(t1_32) // L:[a0 b0 c0 d0 a1 b1 c1 d1], H:[a8 b8 c8 d8 a9 b9 c9 d9]
	x1 := t0_32.InterleaveHiGrouped(t1_32) // L:[a2 b2 c2 d2 a3 b3 c3 d3], H:[a10 b10 c10 d10 a11 b11 c11 d11]
	y0 := t2_32.InterleaveLoGrouped(t3_32) // L:[a4 b4 c4 d4 a5 b5 c5 d5], H:[a12 b12 c12 d12 a13 b13 c13 d13]
	y1 := t2_32.InterleaveHiGrouped(t3_32) // L:[a6 b6 c6 d6 a7 b7 c7 d7], H:[a14 b14 c14 d14 a15 b15 c15 d15]

	idxLo := archsimd.LoadUint32x8Array(&[8]uint32{0, 1, 2, 3, 0, 1, 2, 3})
	idxHi := archsimd.LoadUint32x8Array(&[8]uint32{4, 5, 6, 7, 4, 5, 6, 7})
	maskLane1 := archsimd.LoadUint32x8Array(&[8]uint32{0, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}).Greater(archsimd.BroadcastUint32x8(0))

	row0to4 = x1.Permute(idxLo).Merge(x0.Permute(idxLo), maskLane1).AsUint16x16()
	row8to12 = x1.Permute(idxHi).Merge(x0.Permute(idxHi), maskLane1).AsUint16x16()
	row4to8 = y1.Permute(idxLo).Merge(y0.Permute(idxLo), maskLane1).AsUint16x16()
	row12to16 = y1.Permute(idxHi).Merge(y0.Permute(idxHi), maskLane1).AsUint16x16()
	return
}

// avx2Transpose4x4x64bits transposes 4 rows of 4x64-bit elements (uint64) into
// 4 rows of 4 64-bit elements, where each 1 row of 4 elements is output together in one YMM register.
func avx2Transpose4x4x64bits(v0, v1, v2, v3 archsimd.Uint64x4) (row0, row1, row2, row3 archsimd.Uint64x4) {
	x01_lo := v0.InterleaveLoGrouped(v1).AsUint32x8() // [a0 b0 a2 b2]
	x01_hi := v0.InterleaveHiGrouped(v1).AsUint32x8() // [a1 b1 a3 b3]
	x23_lo := v2.InterleaveLoGrouped(v3).AsUint32x8() // [c0 d0 c2 d2]
	x23_hi := v2.InterleaveHiGrouped(v3).AsUint32x8() // [c1 d1 c3 d3]

	idxLo := archsimd.LoadUint32x8Array(&[8]uint32{0, 1, 2, 3, 0, 1, 2, 3})
	idxHi := archsimd.LoadUint32x8Array(&[8]uint32{4, 5, 6, 7, 4, 5, 6, 7})
	maskLane1 := archsimd.LoadUint32x8Array(&[8]uint32{0, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}).Greater(archsimd.BroadcastUint32x8(0))

	row0 = x23_lo.Permute(idxLo).Merge(x01_lo.Permute(idxLo), maskLane1).AsUint64x4()
	row1 = x23_hi.Permute(idxLo).Merge(x01_hi.Permute(idxLo), maskLane1).AsUint64x4()
	row2 = x23_lo.Permute(idxHi).Merge(x01_lo.Permute(idxHi), maskLane1).AsUint64x4()
	row3 = x23_hi.Permute(idxHi).Merge(x01_hi.Permute(idxHi), maskLane1).AsUint64x4()
	return
}

// avx2PackLHSStrided8x8Uint32 packs full strips of height kernelRows (where kernelRows is a multiple of 8)
// of 32-bit elements using 8x8 SIMD register transposes.
func avx2PackLHSStrided8x8Uint32(
	lhsBasePtr, panelBasePtr uintptr,
	lhsRowStart, lhsColStart, lhsCols,
	fullRows, contractingCols, kernelRows int,
) {
	lhsStrideBytes := uintptr(lhsCols) * 4
	lhsColStartBytes := uintptr(lhsColStart) * 4
	panelColStrideBytes := uintptr(kernelRows) * 4
	stripPanelBytes := uintptr(contractingCols) * panelColStrideBytes

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
			for ; col+8 <= contractingCols; col += 8 {
				v0 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r0Ptr)))
				v1 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r1Ptr)))
				v2 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r2Ptr)))
				v3 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r3Ptr)))
				top01, top23, top45, top67 := avx2Transpose4x8x32bits(v0, v1, v2, v3)

				v4 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r4Ptr)))
				v5 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r5Ptr)))
				v6 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r6Ptr)))
				v7 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(r7Ptr)))
				bot01, bot23, bot45, bot67 := avx2Transpose4x8x32bits(v4, v5, v6, v7)

				top01.SetHi(bot01.GetLo()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst)))
				bot01.SetLo(top01.GetHi()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst + panelColStrideBytes)))
				top23.SetHi(bot23.GetLo()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst + 2*panelColStrideBytes)))
				bot23.SetLo(top23.GetHi()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst + 3*panelColStrideBytes)))
				top45.SetHi(bot45.GetLo()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst + 4*panelColStrideBytes)))
				bot45.SetLo(top45.GetHi()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst + 5*panelColStrideBytes)))
				top67.SetHi(bot67.GetLo()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst + 6*panelColStrideBytes)))
				bot67.SetLo(top67.GetHi()).StoreArray((*[8]uint32)(unsafe.Pointer(pDst + 7*panelColStrideBytes)))

				r0Ptr += 32
				r1Ptr += 32
				r2Ptr += 32
				r3Ptr += 32
				r4Ptr += 32
				r5Ptr += 32
				r6Ptr += 32
				r7Ptr += 32
				pDst += 8 * panelColStrideBytes
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

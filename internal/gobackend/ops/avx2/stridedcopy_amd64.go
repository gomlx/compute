// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"simd/archsimd"
	"unsafe"
)

// DispatchCopyStridedBytesAVX2 copies count blocks of blockBytes bytes from src (stepping by srcStride bytes)
// to dst (stepping by dstStride bytes) using AVX2 (16-byte XMM and 32-byte YMM) vector loads and stores.
// Returns false for blockBytes < 16 or blockBytes > 4096 so the caller falls back to the generic implementation.
func DispatchCopyStridedBytesAVX2(dst, src []byte, count, blockBytes, dstStride, srcStride int) bool {
	if blockBytes < 16 || blockBytes > 4096 {
		return false
	}
	if count <= 0 {
		return true
	}

	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	srcPtr := unsafe.Pointer(unsafe.SliceData(src))

	switch {
	case blockBytes == 16:
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v0.StoreArray((*[16]byte)(dstPtr))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes < 32:
		tailOff := blockBytes - 16
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes == 32:
		for i := 0; ; {
			v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
			v0.StoreArray((*[32]byte)(dstPtr))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes < 64:
		tailOff := blockBytes - 32
		for i := 0; ; {
			v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
			v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[32]byte)(dstPtr))
			v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes == 64:
		for i := 0; ; {
			v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
			v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, 32)))
			v0.StoreArray((*[32]byte)(dstPtr))
			v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, 32)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes <= 96:
		tailOff := blockBytes - 32
		for i := 0; ; {
			v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
			v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, 32)))
			v2 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[32]byte)(dstPtr))
			v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, 32)))
			v2.StoreArray((*[32]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes <= 128:
		tailOff := blockBytes - 32
		for i := 0; ; {
			v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
			v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, 32)))
			v2 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, 64)))
			v3 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[32]byte)(dstPtr))
			v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, 32)))
			v2.StoreArray((*[32]byte)(unsafe.Add(dstPtr, 64)))
			v3.StoreArray((*[32]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	default:
		tailOff := blockBytes - 32
		for i := 0; ; {
			off := 0
			for ; off+128 < blockBytes; off += 128 {
				v0 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off)))
				v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off+32)))
				v2 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off+64)))
				v3 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off+96)))
				v0.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off)))
				v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off+32)))
				v2.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off+64)))
				v3.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off+96)))
			}
			for ; off+32 < blockBytes; off += 32 {
				v0 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off)))
				v0.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off)))
			}
			vTail := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, tailOff)))
			vTail.StoreArray((*[32]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true
	}
}

// DispatchDupStridedBytesAVX2 duplicates each of count blocks of blockBytes bytes from src consecutively into dst.
func DispatchDupStridedBytesAVX2(dst, src []byte, count, blockBytes int) bool {
	if blockBytes < 16 || blockBytes > 4096 {
		return false
	}
	if count <= 0 {
		return true
	}

	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	srcPtr := unsafe.Pointer(unsafe.SliceData(src))
	outStride := 2 * blockBytes

	switch {
	case blockBytes == 16:
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v0.StoreArray((*[16]byte)(dstPtr))
			v0.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 16)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, 32)
			srcPtr = unsafe.Add(srcPtr, 16)
		}
		return true

	case blockBytes < 32:
		tailOff := blockBytes - 16
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, tailOff)))
			dst1 := unsafe.Add(dstPtr, blockBytes)
			v0.StoreArray((*[16]byte)(dst1))
			v1.StoreArray((*[16]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true

	case blockBytes == 32:
		for i := 0; ; {
			v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
			v0.StoreArray((*[32]byte)(dstPtr))
			v0.StoreArray((*[32]byte)(unsafe.Add(dstPtr, 32)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, 64)
			srcPtr = unsafe.Add(srcPtr, 32)
		}
		return true

	case blockBytes < 64:
		tailOff := blockBytes - 32
		for i := 0; ; {
			v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
			v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[32]byte)(dstPtr))
			v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, tailOff)))
			dst1 := unsafe.Add(dstPtr, blockBytes)
			v0.StoreArray((*[32]byte)(dst1))
			v1.StoreArray((*[32]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true

	default:
		tailOff := blockBytes - 32
		for i := 0; ; {
			dst1 := unsafe.Add(dstPtr, blockBytes)
			off := 0
			for ; off+128 < blockBytes; off += 128 {
				v0 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off)))
				v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off+32)))
				v2 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off+64)))
				v3 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off+96)))
				v0.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off)))
				v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off+32)))
				v2.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off+64)))
				v3.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off+96)))
				v0.StoreArray((*[32]byte)(unsafe.Add(dst1, off)))
				v1.StoreArray((*[32]byte)(unsafe.Add(dst1, off+32)))
				v2.StoreArray((*[32]byte)(unsafe.Add(dst1, off+64)))
				v3.StoreArray((*[32]byte)(unsafe.Add(dst1, off+96)))
			}
			for ; off+32 < blockBytes; off += 32 {
				v0 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, off)))
				v0.StoreArray((*[32]byte)(unsafe.Add(dstPtr, off)))
				v0.StoreArray((*[32]byte)(unsafe.Add(dst1, off)))
			}
			vTail := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, tailOff)))
			vTail.StoreArray((*[32]byte)(unsafe.Add(dstPtr, tailOff)))
			vTail.StoreArray((*[32]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true
	}
}

func copyBlockUpTo128AVX2(dstPtr, srcPtr unsafe.Pointer, n int) {
	v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
	vLast := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, n-32)))
	v0.StoreArray((*[32]byte)(dstPtr))
	if n > 64 {
		v1 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, 32)))
		v1.StoreArray((*[32]byte)(unsafe.Add(dstPtr, 32)))
		if n > 96 {
			v2 := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, 64)))
			v2.StoreArray((*[32]byte)(unsafe.Add(dstPtr, 64)))
		}
	}
	vLast.StoreArray((*[32]byte)(unsafe.Add(dstPtr, n-32)))
}

// DispatchConcat2StridedBytesAVX2 concatenates count blocks of size b0 from src0 and b1 from src1 into dst.
func DispatchConcat2StridedBytesAVX2(dst, src0, src1 []byte, count, b0, b1 int) bool {
	if b0 < 32 || b0 > 128 || b1 < 32 || b1 > 128 {
		return false
	}
	if count <= 0 {
		return true
	}
	outStride := b0 + b1
	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	src0Ptr := unsafe.Pointer(unsafe.SliceData(src0))
	src1Ptr := unsafe.Pointer(unsafe.SliceData(src1))
	for i := 0; ; {
		copyBlockUpTo128AVX2(dstPtr, src0Ptr, b0)
		copyBlockUpTo128AVX2(unsafe.Add(dstPtr, b0), src1Ptr, b1)
		if i++; i >= count {
			break
		}
		dstPtr = unsafe.Add(dstPtr, outStride)
		src0Ptr = unsafe.Add(src0Ptr, b0)
		src1Ptr = unsafe.Add(src1Ptr, b1)
	}
	return true
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"simd/archsimd"
	"unsafe"
)

// DispatchCopyStridedBytesNEON copies count blocks of blockBytes bytes from src (stepping by srcStride bytes)
// to dst (stepping by dstStride bytes) using ARM64 NEON 16-byte (128-bit Q-register) vector loads and stores.
// Returns false for blockBytes < 16 or blockBytes > 4096 so the caller falls back to the generic implementation.
func DispatchCopyStridedBytesNEON(dst, src []byte, count, blockBytes, dstStride, srcStride int) bool {
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
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 16)))
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 16)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes < 48:
		tailOff := blockBytes - 16
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 16)))
			v2 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 16)))
			v2.StoreArray((*[16]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes == 48:
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 16)))
			v2 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 32)))
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 16)))
			v2.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 32)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes <= 64:
		tailOff := blockBytes - 16
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 16)))
			v2 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 32)))
			v3 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 16)))
			v2.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 32)))
			v3.StoreArray((*[16]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	default:
		tailOff := blockBytes - 16
		for i := 0; ; {
			off := 0
			for ; off+64 < blockBytes; off += 64 {
				v0 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off)))
				v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off+16)))
				v2 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off+32)))
				v3 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off+48)))
				v0.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off)))
				v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off+16)))
				v2.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off+32)))
				v3.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off+48)))
			}
			for ; off+16 < blockBytes; off += 16 {
				v0 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off)))
				v0.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off)))
			}
			vTail := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, tailOff)))
			vTail.StoreArray((*[16]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true
	}
}

// DispatchDupStridedBytesNEON duplicates each of count blocks of blockBytes bytes from src consecutively into dst.
func DispatchDupStridedBytesNEON(dst, src []byte, count, blockBytes int) bool {
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
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 16)))
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 16)))
			v0.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 32)))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 48)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, 64)
			srcPtr = unsafe.Add(srcPtr, 32)
		}
		return true

	case blockBytes <= 64:
		tailOff := blockBytes - 16
		for i := 0; ; {
			v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
			v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 16)))
			vLast := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, tailOff)))
			dst1 := unsafe.Add(dstPtr, blockBytes)
			v0.StoreArray((*[16]byte)(dstPtr))
			v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 16)))
			v0.StoreArray((*[16]byte)(dst1))
			v1.StoreArray((*[16]byte)(unsafe.Add(dst1, 16)))
			if blockBytes > 48 {
				v2 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, 32)))
				v2.StoreArray((*[16]byte)(unsafe.Add(dstPtr, 32)))
				v2.StoreArray((*[16]byte)(unsafe.Add(dst1, 32)))
			}
			vLast.StoreArray((*[16]byte)(unsafe.Add(dstPtr, tailOff)))
			vLast.StoreArray((*[16]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true

	default:
		tailOff := blockBytes - 16
		for i := 0; ; {
			dst1 := unsafe.Add(dstPtr, blockBytes)
			off := 0
			for ; off+64 < blockBytes; off += 64 {
				v0 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off)))
				v1 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off+16)))
				v2 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off+32)))
				v3 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off+48)))
				v0.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off)))
				v1.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off+16)))
				v2.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off+32)))
				v3.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off+48)))
				v0.StoreArray((*[16]byte)(unsafe.Add(dst1, off)))
				v1.StoreArray((*[16]byte)(unsafe.Add(dst1, off+16)))
				v2.StoreArray((*[16]byte)(unsafe.Add(dst1, off+32)))
				v3.StoreArray((*[16]byte)(unsafe.Add(dst1, off+48)))
			}
			for ; off+16 < blockBytes; off += 16 {
				v0 := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off)))
				v0.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off)))
				v0.StoreArray((*[16]byte)(unsafe.Add(dst1, off)))
			}
			vTail := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, tailOff)))
			vTail.StoreArray((*[16]byte)(unsafe.Add(dstPtr, tailOff)))
			vTail.StoreArray((*[16]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true
	}
}

func copyBlockUpTo128NEON(dstPtr, srcPtr unsafe.Pointer, n int) {
	v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
	vLast := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, n-16)))
	v0.StoreArray((*[16]byte)(dstPtr))
	for off := 16; off+16 < n; off += 16 {
		v := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, off)))
		v.StoreArray((*[16]byte)(unsafe.Add(dstPtr, off)))
	}
	vLast.StoreArray((*[16]byte)(unsafe.Add(dstPtr, n-16)))
}

// DispatchConcat2StridedBytesNEON concatenates count blocks of size b0 from src0 and b1 from src1 into dst.
func DispatchConcat2StridedBytesNEON(dst, src0, src1 []byte, count, b0, b1 int) bool {
	if b0 < 16 || b0 > 128 || b1 < 16 || b1 > 128 {
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
		copyBlockUpTo128NEON(dstPtr, src0Ptr, b0)
		copyBlockUpTo128NEON(unsafe.Add(dstPtr, b0), src1Ptr, b1)
		if i++; i >= count {
			break
		}
		dstPtr = unsafe.Add(dstPtr, outStride)
		src0Ptr = unsafe.Add(src0Ptr, b0)
		src1Ptr = unsafe.Add(src1Ptr, b1)
	}
	return true
}

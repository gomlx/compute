// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"simd/archsimd"
	"unsafe"
)

// DispatchCopyStridedBytesAVX512 copies count blocks of blockBytes bytes from src (stepping by srcStride bytes)
// to dst (stepping by dstStride bytes) using AVX-512 64-byte ZMM vector loads and stores (and 16/32-byte for smaller blocks).
// Returns false for blockBytes < 16 or blockBytes > 4096 so the caller falls back to lower-priority/generic.
func DispatchCopyStridedBytesAVX512(dst, src []byte, count, blockBytes, dstStride, srcStride int) bool {
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
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v0.StoreArray((*[64]byte)(dstPtr))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes <= 128:
		tailOff := blockBytes - 64
		for i := 0; ; {
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[64]byte)(dstPtr))
			v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes <= 192:
		tailOff := blockBytes - 64
		for i := 0; ; {
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 64)))
			v2 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[64]byte)(dstPtr))
			v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 64)))
			v2.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	case blockBytes <= 256:
		tailOff := blockBytes - 64
		for i := 0; ; {
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 64)))
			v2 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 128)))
			v3 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[64]byte)(dstPtr))
			v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 64)))
			v2.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 128)))
			v3.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true

	default:
		tailOff := blockBytes - 64
		for i := 0; ; {
			off := 0
			for ; off+256 < blockBytes; off += 256 {
				v0 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off)))
				v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off+64)))
				v2 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off+128)))
				v3 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off+192)))
				v0.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off)))
				v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off+64)))
				v2.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off+128)))
				v3.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off+192)))
			}
			for ; off+64 < blockBytes; off += 64 {
				v0 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off)))
				v0.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off)))
			}
			vTail := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			vTail.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
		return true
	}
}

// DispatchDupStridedBytesAVX512 duplicates each of count blocks of blockBytes bytes from src consecutively into dst.
func DispatchDupStridedBytesAVX512(dst, src []byte, count, blockBytes int) bool {
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

	case blockBytes == 64:
		for i := 0; ; {
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v0.StoreArray((*[64]byte)(dstPtr))
			v0.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 64)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, 128)
			srcPtr = unsafe.Add(srcPtr, 64)
		}
		return true

	case blockBytes <= 128:
		tailOff := blockBytes - 64
		for i := 0; ; {
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[64]byte)(dstPtr))
			v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			dst1 := unsafe.Add(dstPtr, blockBytes)
			v0.StoreArray((*[64]byte)(dst1))
			v1.StoreArray((*[64]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true

	case blockBytes <= 192:
		tailOff := blockBytes - 64
		for i := 0; ; {
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 64)))
			v2 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[64]byte)(dstPtr))
			v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 64)))
			v2.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			dst1 := unsafe.Add(dstPtr, blockBytes)
			v0.StoreArray((*[64]byte)(dst1))
			v1.StoreArray((*[64]byte)(unsafe.Add(dst1, 64)))
			v2.StoreArray((*[64]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true

	case blockBytes <= 256:
		tailOff := blockBytes - 64
		for i := 0; ; {
			v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
			v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 64)))
			v2 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 128)))
			v3 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			v0.StoreArray((*[64]byte)(dstPtr))
			v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 64)))
			v2.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 128)))
			v3.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			dst1 := unsafe.Add(dstPtr, blockBytes)
			v0.StoreArray((*[64]byte)(dst1))
			v1.StoreArray((*[64]byte)(unsafe.Add(dst1, 64)))
			v2.StoreArray((*[64]byte)(unsafe.Add(dst1, 128)))
			v3.StoreArray((*[64]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true

	default:
		tailOff := blockBytes - 64
		for i := 0; ; {
			dst1 := unsafe.Add(dstPtr, blockBytes)
			off := 0
			for ; off+256 < blockBytes; off += 256 {
				v0 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off)))
				v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off+64)))
				v2 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off+128)))
				v3 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off+192)))
				v0.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off)))
				v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off+64)))
				v2.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off+128)))
				v3.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off+192)))
				v0.StoreArray((*[64]byte)(unsafe.Add(dst1, off)))
				v1.StoreArray((*[64]byte)(unsafe.Add(dst1, off+64)))
				v2.StoreArray((*[64]byte)(unsafe.Add(dst1, off+128)))
				v3.StoreArray((*[64]byte)(unsafe.Add(dst1, off+192)))
			}
			for ; off+64 < blockBytes; off += 64 {
				v0 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, off)))
				v0.StoreArray((*[64]byte)(unsafe.Add(dstPtr, off)))
				v0.StoreArray((*[64]byte)(unsafe.Add(dst1, off)))
			}
			vTail := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, tailOff)))
			vTail.StoreArray((*[64]byte)(unsafe.Add(dstPtr, tailOff)))
			vTail.StoreArray((*[64]byte)(unsafe.Add(dst1, tailOff)))
			if i++; i >= count {
				break
			}
			dstPtr = unsafe.Add(dstPtr, outStride)
			srcPtr = unsafe.Add(srcPtr, blockBytes)
		}
		return true
	}
}

func copyBlockUpTo256AVX512(dstPtr, srcPtr unsafe.Pointer, n int) {
	switch {
	case n < 32:
		v0 := archsimd.LoadUint8x16Array((*[16]byte)(srcPtr))
		vLast := archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(srcPtr, n-16)))
		v0.StoreArray((*[16]byte)(dstPtr))
		vLast.StoreArray((*[16]byte)(unsafe.Add(dstPtr, n-16)))
	case n < 64:
		v0 := archsimd.LoadUint8x32Array((*[32]byte)(srcPtr))
		vLast := archsimd.LoadUint8x32Array((*[32]byte)(unsafe.Add(srcPtr, n-32)))
		v0.StoreArray((*[32]byte)(dstPtr))
		vLast.StoreArray((*[32]byte)(unsafe.Add(dstPtr, n-32)))
	default:
		v0 := archsimd.LoadUint8x64Array((*[64]byte)(srcPtr))
		vLast := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, n-64)))
		v0.StoreArray((*[64]byte)(dstPtr))
		if n > 128 {
			v1 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 64)))
			v1.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 64)))
			if n > 192 {
				v2 := archsimd.LoadUint8x64Array((*[64]byte)(unsafe.Add(srcPtr, 128)))
				v2.StoreArray((*[64]byte)(unsafe.Add(dstPtr, 128)))
			}
		}
		vLast.StoreArray((*[64]byte)(unsafe.Add(dstPtr, n-64)))
	}
}

// DispatchConcat2StridedBytesAVX512 concatenates count blocks of size b0 from src0 and b1 from src1 into dst.
func DispatchConcat2StridedBytesAVX512(dst, src0, src1 []byte, count, b0, b1 int) bool {
	if b0 < 16 || b0 > 256 || b1 < 16 || b1 > 256 {
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
		copyBlockUpTo256AVX512(dstPtr, src0Ptr, b0)
		copyBlockUpTo256AVX512(unsafe.Add(dstPtr, b0), src1Ptr, b1)
		if i++; i >= count {
			break
		}
		dstPtr = unsafe.Add(dstPtr, outStride)
		src0Ptr = unsafe.Add(src0Ptr, b0)
		src1Ptr = unsafe.Add(src1Ptr, b1)
	}
	return true
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
)

const (
	// minParallelCopyBytes is the minimum total byte size to parallelize Slice and Concatenate.
	minParallelCopyBytes = 256 * 1024
	// minParallelCopyChunkBytes is the minimum per-worker byte chunk size (256 KB, fits in L2 cache).
	minParallelCopyChunkBytes = 256 * 1024
)

// copyStridedBytes copies count blocks of blockBytes bytes from src (stepping by srcStride bytes)
// to dst (stepping by dstStride bytes). Works across all tensor DTypes.
func copyStridedBytes(dst, src []byte, count, blockBytes, dstStride, srcStride int) {
	if count <= 0 || blockBytes <= 0 {
		return
	}
	if count == 1 || (dstStride == blockBytes && srcStride == blockBytes) {
		total := count * blockBytes
		copy(dst[:total], src[:total])
		return
	}
	if fn := gobackend.GetCopyStridedBytesArchDispatcher(); fn != nil && fn(dst, src, count, blockBytes, dstStride, srcStride) {
		return
	}
	copyStridedBytesGeneric(dst, src, count, blockBytes, dstStride, srcStride)
}

// dupStridedBytes duplicates each of count blocks of blockBytes bytes from src consecutively
// into dst (i.e. dst[2*i*blockBytes : 2*(i+1)*blockBytes] = src[i*blockBytes : (i+1)*blockBytes] twice).
func dupStridedBytes(dst, src []byte, count, blockBytes int) {
	if count <= 0 || blockBytes <= 0 {
		return
	}
	if fn := gobackend.GetDupStridedBytesArchDispatcher(); fn != nil && fn(dst, src, count, blockBytes) {
		return
	}
	dupStridedBytesGeneric(dst, src, count, blockBytes)
}

// concat2StridedBytes concatenates count blocks of size b0 from src0 and size b1 from src1
// into dst (with output row stride b0 + b1).
func concat2StridedBytes(dst, src0, src1 []byte, count, b0, b1 int) {
	if count <= 0 {
		return
	}
	if b0 == b1 && unsafe.SliceData(src0) == unsafe.SliceData(src1) {
		dupStridedBytes(dst, src0, count, b0)
		return
	}
	if fn := gobackend.GetConcat2StridedBytesArchDispatcher(); fn != nil && fn(dst, src0, src1, count, b0, b1) {
		return
	}
	concat2StridedBytesGeneric(dst, src0, src1, count, b0, b1)
}

func copyStridedBytesGeneric(dst, src []byte, count, blockBytes, dstStride, srcStride int) {
	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	srcPtr := unsafe.Pointer(unsafe.SliceData(src))

	switch {
	case blockBytes == 1:
		for range count {
			*(*byte)(dstPtr) = *(*byte)(srcPtr)
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes == 2:
		for range count {
			*(*uint16)(dstPtr) = *(*uint16)(srcPtr)
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes == 3:
		for range count {
			*(*uint16)(dstPtr) = *(*uint16)(srcPtr)
			*(*byte)(unsafe.Add(dstPtr, 2)) = *(*byte)(unsafe.Add(srcPtr, 2))
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes == 4:
		for range count {
			*(*uint32)(dstPtr) = *(*uint32)(srcPtr)
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes < 8:
		tailOff := blockBytes - 4
		for range count {
			v0 := *(*uint32)(srcPtr)
			v1 := *(*uint32)(unsafe.Add(srcPtr, tailOff))
			*(*uint32)(dstPtr) = v0
			*(*uint32)(unsafe.Add(dstPtr, tailOff)) = v1
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes == 8:
		for range count {
			*(*uint64)(dstPtr) = *(*uint64)(srcPtr)
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes <= 16:
		tailOff := blockBytes - 8
		for range count {
			v0 := *(*uint64)(srcPtr)
			v1 := *(*uint64)(unsafe.Add(srcPtr, tailOff))
			*(*uint64)(dstPtr) = v0
			*(*uint64)(unsafe.Add(dstPtr, tailOff)) = v1
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes <= 32:
		tailOff := blockBytes - 16
		for range count {
			v0 := *(*uint64)(srcPtr)
			v1 := *(*uint64)(unsafe.Add(srcPtr, 8))
			v2 := *(*uint64)(unsafe.Add(srcPtr, tailOff))
			v3 := *(*uint64)(unsafe.Add(srcPtr, tailOff+8))
			*(*uint64)(dstPtr) = v0
			*(*uint64)(unsafe.Add(dstPtr, 8)) = v1
			*(*uint64)(unsafe.Add(dstPtr, tailOff)) = v2
			*(*uint64)(unsafe.Add(dstPtr, tailOff+8)) = v3
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	case blockBytes <= 256:
		tailOff := blockBytes - 8
		for range count {
			off := 0
			for ; off+32 < blockBytes; off += 32 {
				v0 := *(*uint64)(unsafe.Add(srcPtr, off))
				v1 := *(*uint64)(unsafe.Add(srcPtr, off+8))
				v2 := *(*uint64)(unsafe.Add(srcPtr, off+16))
				v3 := *(*uint64)(unsafe.Add(srcPtr, off+24))
				*(*uint64)(unsafe.Add(dstPtr, off)) = v0
				*(*uint64)(unsafe.Add(dstPtr, off+8)) = v1
				*(*uint64)(unsafe.Add(dstPtr, off+16)) = v2
				*(*uint64)(unsafe.Add(dstPtr, off+24)) = v3
			}
			for ; off+8 < blockBytes; off += 8 {
				*(*uint64)(unsafe.Add(dstPtr, off)) = *(*uint64)(unsafe.Add(srcPtr, off))
			}
			*(*uint64)(unsafe.Add(dstPtr, tailOff)) = *(*uint64)(unsafe.Add(srcPtr, tailOff))
			dstPtr = unsafe.Add(dstPtr, dstStride)
			srcPtr = unsafe.Add(srcPtr, srcStride)
		}
	default:
		dstIdx := 0
		srcIdx := 0
		for range count {
			copy(dst[dstIdx:dstIdx+blockBytes], src[srcIdx:srcIdx+blockBytes])
			dstIdx += dstStride
			srcIdx += srcStride
		}
	}
}

func dupStridedBytesGeneric(dst, src []byte, count, blockBytes int) {
	outStride := 2 * blockBytes
	tileRows := max(16, min(256, 16384/max(1, outStride)))
	for r := 0; r < count; r += tileRows {
		tRows := min(tileRows, count-r)
		outOff := r * outStride
		inOff := r * blockBytes
		copyStridedBytes(dst[outOff:], src[inOff:], tRows, blockBytes, outStride, blockBytes)
		copyStridedBytes(dst[outOff+blockBytes:], src[inOff:], tRows, blockBytes, outStride, blockBytes)
	}
}

func concat2StridedBytesGeneric(dst, src0, src1 []byte, count, b0, b1 int) {
	outStride := b0 + b1
	tileRows := max(16, min(256, 16384/max(1, outStride)))
	for r := 0; r < count; r += tileRows {
		tRows := min(tileRows, count-r)
		outOff := r * outStride
		copyStridedBytes(dst[outOff:], src0[r*b0:], tRows, b0, outStride, b0)
		copyStridedBytes(dst[outOff+b0:], src1[r*b1:], tRows, b1, outStride, b1)
	}
}

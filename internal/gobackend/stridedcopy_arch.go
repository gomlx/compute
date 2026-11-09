// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import "slices"

// CopyStridedBytesArchFn copies count blocks of blockBytes bytes from src (stepping by srcStride bytes)
// to dst (stepping by dstStride bytes). Returns true if handled, false to fall back to generic.
type CopyStridedBytesArchFn func(dst, src []byte, count, blockBytes, dstStride, srcStride int) bool

// DupStridedBytesArchFn duplicates each of count blocks of blockBytes bytes from src consecutively
// into dst (i.e. dst[2*i*blockBytes : 2*(i+1)*blockBytes] = src[i*blockBytes : (i+1)*blockBytes] twice).
// Returns true if handled, false to fall back to generic.
type DupStridedBytesArchFn func(dst, src []byte, count, blockBytes int) bool

// Concat2StridedBytesArchFn concatenates count blocks of size b0 from src0 and size b1 from src1
// into dst (with output row stride b0 + b1). Returns true if handled, false to fall back to generic.
type Concat2StridedBytesArchFn func(dst, src0, src1 []byte, count, b0, b1 int) bool

type stridedCopyArchEntry struct {
	priority  RegisterPriority
	copyFn    CopyStridedBytesArchFn
	dupFn     DupStridedBytesArchFn
	concat2Fn Concat2StridedBytesArchFn
}

var (
	stridedCopyArchEntries []stridedCopyArchEntry
	copyStridedArchFn      CopyStridedBytesArchFn
	dupStridedArchFn       DupStridedBytesArchFn
	concat2StridedArchFn   Concat2StridedBytesArchFn
)

// SetStridedCopyArchDispatcher registers architecture-specific strided byte-copy dispatchers with a priority.
// Dispatchers are tried in descending order of priority; if a higher-priority dispatcher returns false,
// the next one is tried before falling back to the generic implementation.
func SetStridedCopyArchDispatcher(
	priority RegisterPriority,
	copyFn CopyStridedBytesArchFn,
	dupFn DupStridedBytesArchFn,
	concat2Fn Concat2StridedBytesArchFn,
) {
	if priority < 0 || (copyFn == nil && dupFn == nil && concat2Fn == nil) {
		return
	}
	entry := stridedCopyArchEntry{
		priority:  priority,
		copyFn:    copyFn,
		dupFn:     dupFn,
		concat2Fn: concat2Fn,
	}
	idx := 0
	for idx < len(stridedCopyArchEntries) && stridedCopyArchEntries[idx].priority > priority {
		idx++
	}
	if idx < len(stridedCopyArchEntries) && stridedCopyArchEntries[idx].priority == priority {
		stridedCopyArchEntries[idx] = entry
	} else {
		stridedCopyArchEntries = slices.Insert(stridedCopyArchEntries, idx, entry)
	}
	rebuildStridedCopyArchFns()
}

func rebuildStridedCopyArchFns() {
	switch len(stridedCopyArchEntries) {
	case 0:
		copyStridedArchFn = nil
		dupStridedArchFn = nil
		concat2StridedArchFn = nil
	case 1:
		e := stridedCopyArchEntries[0]
		copyStridedArchFn = e.copyFn
		dupStridedArchFn = e.dupFn
		concat2StridedArchFn = e.concat2Fn
	default:
		entries := slices.Clone(stridedCopyArchEntries)
		copyStridedArchFn = func(dst, src []byte, count, blockBytes, dstStride, srcStride int) bool {
			for _, e := range entries {
				if e.copyFn != nil && e.copyFn(dst, src, count, blockBytes, dstStride, srcStride) {
					return true
				}
			}
			return false
		}
		dupStridedArchFn = func(dst, src []byte, count, blockBytes int) bool {
			for _, e := range entries {
				if e.dupFn != nil && e.dupFn(dst, src, count, blockBytes) {
					return true
				}
			}
			return false
		}
		concat2StridedArchFn = func(dst, src0, src1 []byte, count, b0, b1 int) bool {
			for _, e := range entries {
				if e.concat2Fn != nil && e.concat2Fn(dst, src0, src1, count, b0, b1) {
					return true
				}
			}
			return false
		}
	}
}

// GetCopyStridedBytesArchDispatcher returns the registered architecture-specific strided byte copy dispatcher, if any.
func GetCopyStridedBytesArchDispatcher() CopyStridedBytesArchFn {
	return copyStridedArchFn
}

// GetDupStridedBytesArchDispatcher returns the registered architecture-specific duplicate strided byte copy dispatcher, if any.
func GetDupStridedBytesArchDispatcher() DupStridedBytesArchFn {
	return dupStridedArchFn
}

// GetConcat2StridedBytesArchDispatcher returns the registered architecture-specific 2-input strided concat dispatcher, if any.
func GetConcat2StridedBytesArchDispatcher() Concat2StridedBytesArchFn {
	return concat2StridedArchFn
}

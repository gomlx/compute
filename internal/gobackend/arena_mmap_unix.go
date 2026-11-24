// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux || darwin || freebsd || openbsd || netbsd

package gobackend

import (
	"syscall"
	"unsafe"
)

const arenaMmapAlign = 2 * 1024 * 1024 // 2 MiB alignment for Linux Transparent Huge Pages.

// mmapConstantArenaChunk allocates an off-heap anonymous virtual memory mapping of at least size
// bytes, rounded up to a 2 MiB boundary and aligned to a 2 MiB starting address (by over-allocating
// and trimming unaligned virtual prefix/suffix pages without touching them).
// On Linux, AdviseHugePages immediately advises MADV_HUGEPAGE on the entire aligned region before
// first touch so the kernel promotes ~100% of the chunk synchronously to 2 MiB huge pages.
func mmapConstantArenaChunk(size int) ([]byte, error) {
	usable := (size + arenaMmapAlign - 1) &^ (arenaMmapAlign - 1)
	if usable <= 0 {
		usable = arenaMmapAlign
	}

	// Over-allocate virtual address space by arenaMmapAlign so we can guarantee a 2 MiB-aligned base.
	// Untouched virtual pages consume zero physical RAM and are immediately unmapped below.
	allocSize := usable + arenaMmapAlign
	raw, err := syscall.Mmap(-1, 0, allocSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		return nil, err
	}

	base := uintptr(unsafe.Pointer(unsafe.SliceData(raw)))
	alignedAddr := (base + arenaMmapAlign - 1) &^ (arenaMmapAlign - 1)
	headPad := int(alignedAddr - base)
	tailStart := headPad + usable

	if headPad > 0 {
		_ = syscall.Munmap(raw[:headPad])
	}
	if tailStart < len(raw) {
		_ = syscall.Munmap(raw[tailStart:])
	}

	aligned := raw[headPad:tailStart:tailStart]
	AdviseHugePages(aligned)
	return aligned, nil
}

// munmapConstantArenaChunk immediately returns an off-heap mmap chunk to the OS.
func munmapConstantArenaChunk(data []byte) {
	if len(data) == 0 {
		return
	}
	_ = syscall.Munmap(data)
}

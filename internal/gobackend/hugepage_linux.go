// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux

package gobackend

import (
	"syscall"
	"unsafe"

	"github.com/gomlx/compute/support/envutil"
)

const (
	// hugePageSize is the standard x86-64 / arm64 Linux Transparent Huge Page size (2 MiB).
	hugePageSize = 2 * 1024 * 1024
)

// UseHugePages controls whether large immutable weight slabs (>= 2 MiB) advise the Linux
// kernel to back their 2 MiB-aligned interior with Transparent Huge Pages via MADV_HUGEPAGE.
// Can be disabled with GOMLX_GO_HUGEPAGES=false.
var UseHugePages = envutil.MustReadBool("GOMLX_GO_HUGEPAGES", true)

// AdviseHugePages hints to the Linux kernel via madvise(MADV_HUGEPAGE) that the 2 MiB-aligned
// interior of data should be backed by 2 MiB Transparent Huge Pages.
//
// Calling this immediately after allocating a fresh large byte slice (>= 2 MiB, e.g. pre-packed
// ConstantData weight slabs) and before writing to it allows Linux (even when THP is set to
// [madvise]) to fault 2 MiB huge pages synchronously on first touch, eliminating L2 dTLB misses
// when streaming multi-gigabyte models during LLM decoding.
//
// Only the full 2 MiB-aligned interior is advised, so zero extra RSS is consumed. Errors are
// silently ignored because MADV_HUGEPAGE is purely an advisory hint with safe 4 KiB fallback.
func AdviseHugePages(data []byte) {
	if !UseHugePages || len(data) < hugePageSize {
		return
	}
	base := uintptr(unsafe.Pointer(unsafe.SliceData(data)))
	aligned := (base + hugePageSize - 1) &^ (hugePageSize - 1)
	offset := int(aligned - base)
	if offset >= len(data) {
		return
	}
	advisedLen := (len(data) - offset) &^ (hugePageSize - 1)
	if advisedLen < hugePageSize {
		return
	}
	_ = syscall.Madvise(data[offset:offset+advisedLen], syscall.MADV_HUGEPAGE)
}

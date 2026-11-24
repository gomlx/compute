// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !linux && !darwin && !freebsd && !openbsd && !netbsd

package gobackend

// mmapConstantArenaChunk falls back to a Go-managed byte slice on platforms without Unix mmap.
func mmapConstantArenaChunk(size int) ([]byte, error) {
	return allocAlignedBytes(size), nil
}

// munmapConstantArenaChunk is a no-op when backed by Go-managed slices.
func munmapConstantArenaChunk(_ []byte) {}

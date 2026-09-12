// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dot

import (
	"sync"
)

// PackedMatrixCache holds pre-packed panel buffers for a constant matrix (LHS or RHS).
// It allows one-time initialization via sync.Once, followed by lock-free concurrent reads.
type PackedMatrixCache struct {
	Once sync.Once

	// Panels holds the pre-packed panels.
	// For RHS: []T or [][]T indexed by (kPanelIdx * numColPanels + colPanelIdx).
	// For LHS: []T or [][]T indexed by (kPanelIdx * numRowPanels + rowPanelIdx).
	// Stored as any to support float32, bfloat16, float16, float64 without code duplication.
	Panels any
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
)

// RunAll runs all generic backend tests on the given backend.
func RunAll(t *testing.T, b compute.Backend) {
	t.Run("BinaryOps", func(t *testing.T) { TestBinaryOps(t, b) })
	t.Run("UnaryOps", func(t *testing.T) { TestUnaryOps(t, b) })
	t.Run("ConvertDType", func(t *testing.T) { TestConvertDType(t, b) })
	t.Run("Bitcast", func(t *testing.T) { TestBitcast(t, b) })
	t.Run("Exec", func(t *testing.T) { TestExec(t, b) })
	t.Run("Functions", func(t *testing.T) { TestFunctions(t, b) })
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/defaultpkgs"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul/neon"
	"github.com/gomlx/compute/support/backendtest"
)

func TestNEON(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	origUseAsm := neon.NEONUseAsm
	origF32 := neon.NEONParamsFloat32
	origF16 := neon.NEONParamsFloat16
	origBF16 := neon.NEONParamsBFloat16
	origF64 := neon.NEONParamsFloat64
	defer func() {
		neon.NEONUseAsm = origUseAsm
		neon.NEONParamsFloat32 = origF32
		neon.NEONParamsFloat16 = origF16
		neon.NEONParamsBFloat16 = origBF16
		neon.NEONParamsFloat64 = origF64
		dot.ResetTestRegistrations()
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = false
	}()
	dot.ResetTestRegistrations()
	neon.RegisterNEONForTests()

	backend := compute.MustNew()
	defer backend.Finalize()

	t.Run("Small", func(t *testing.T) {
		matmul.ForceSmallVariant = true
		matmul.ForceLargeVariant = false
		backendtest.TestDotGeneral(t, backend)
	})

	t.Run("Large/Asm_8x8", func(t *testing.T) {
		neon.NEONUseAsm = true
		neon.NEONParamsFloat32.LHSL1KernelRows, neon.NEONParamsFloat32.RHSL1KernelCols = 8, 8
		neon.NEONParamsFloat16.LHSL1KernelRows, neon.NEONParamsFloat16.RHSL1KernelCols = 8, 8
		neon.NEONParamsBFloat16.LHSL1KernelRows, neon.NEONParamsBFloat16.RHSL1KernelCols = 8, 8
		neon.NEONParamsFloat64.LHSL1KernelRows, neon.NEONParamsFloat64.RHSL1KernelCols = 8, 4
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = true
		backendtest.TestDotGeneral(t, backend)
	})

	t.Run("Large/Asm_4x16", func(t *testing.T) {
		neon.NEONUseAsm = true
		neon.NEONParamsFloat32.LHSL1KernelRows, neon.NEONParamsFloat32.RHSL1KernelCols = 4, 16
		neon.NEONParamsFloat16.LHSL1KernelRows, neon.NEONParamsFloat16.RHSL1KernelCols = 4, 16
		neon.NEONParamsBFloat16.LHSL1KernelRows, neon.NEONParamsBFloat16.RHSL1KernelCols = 4, 16
		neon.NEONParamsFloat64.LHSL1KernelRows, neon.NEONParamsFloat64.RHSL1KernelCols = 4, 8
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = true
		backendtest.TestDotGeneral(t, backend)
	})

	t.Run("Large/GoSIMD_4x16", func(t *testing.T) {
		neon.NEONUseAsm = false
		neon.NEONParamsFloat32.LHSL1KernelRows, neon.NEONParamsFloat32.RHSL1KernelCols = 4, 16
		neon.NEONParamsFloat16.LHSL1KernelRows, neon.NEONParamsFloat16.RHSL1KernelCols = 4, 16
		neon.NEONParamsBFloat16.LHSL1KernelRows, neon.NEONParamsBFloat16.RHSL1KernelCols = 4, 16
		neon.NEONParamsFloat64.LHSL1KernelRows, neon.NEONParamsFloat64.RHSL1KernelCols = 4, 8
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = true
		backendtest.TestDotGeneral(t, backend)
	})

	t.Run("Large/GoSIMD_8x8", func(t *testing.T) {
		neon.NEONUseAsm = false
		neon.NEONParamsFloat32.LHSL1KernelRows, neon.NEONParamsFloat32.RHSL1KernelCols = 8, 8
		neon.NEONParamsFloat16.LHSL1KernelRows, neon.NEONParamsFloat16.RHSL1KernelCols = 8, 8
		neon.NEONParamsBFloat16.LHSL1KernelRows, neon.NEONParamsBFloat16.RHSL1KernelCols = 8, 8
		neon.NEONParamsFloat64.LHSL1KernelRows, neon.NEONParamsFloat64.RHSL1KernelCols = 8, 4
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = true
		backendtest.TestDotGeneral(t, backend)
	})

	if gobackend.IsSMEAllowed {
		t.Run("SME", func(t *testing.T) {
			origSMEF32 := neon.SMEParamsFloat32
			origSMEF16 := neon.SMEParamsFloat16
			origSMEBF16 := neon.SMEParamsBFloat16
			origSMEF64 := neon.SMEParamsFloat64
			defer func() {
				neon.SMEParamsFloat32 = origSMEF32
				neon.SMEParamsFloat16 = origSMEF16
				neon.SMEParamsBFloat16 = origSMEBF16
				neon.SMEParamsFloat64 = origSMEF64
			}()
			dot.ResetTestRegistrations()
			neon.RegisterSMEForTests()
			neon.NEONUseAsm = true

			t.Run("DefaultRouter", func(t *testing.T) {
				matmul.ForceSmallVariant = false
				matmul.ForceLargeVariant = false
				backendtest.TestDotGeneral(t, backend)
			})

			t.Run("Large_32x32", func(t *testing.T) {
				neon.SMEParamsFloat32.LHSL1KernelRows, neon.SMEParamsFloat32.RHSL1KernelCols = 32, 32
				neon.SMEParamsFloat16.LHSL1KernelRows, neon.SMEParamsFloat16.RHSL1KernelCols = 32, 32
				neon.SMEParamsBFloat16.LHSL1KernelRows, neon.SMEParamsBFloat16.RHSL1KernelCols = 32, 32
				neon.SMEParamsFloat64.LHSL1KernelRows, neon.SMEParamsFloat64.RHSL1KernelCols = 16, 16
				matmul.ForceSmallVariant = false
				matmul.ForceLargeVariant = true
				backendtest.TestDotGeneral(t, backend)
			})

			t.Run("Large_16x16", func(t *testing.T) {
				neon.SMEParamsFloat32.LHSL1KernelRows, neon.SMEParamsFloat32.RHSL1KernelCols = 16, 16
				neon.SMEParamsFloat16.LHSL1KernelRows, neon.SMEParamsFloat16.RHSL1KernelCols = 16, 16
				neon.SMEParamsBFloat16.LHSL1KernelRows, neon.SMEParamsBFloat16.RHSL1KernelCols = 16, 16
				neon.SMEParamsFloat64.LHSL1KernelRows, neon.SMEParamsFloat64.RHSL1KernelCols = 8, 8
				matmul.ForceSmallVariant = false
				matmul.ForceLargeVariant = true
				backendtest.TestDotGeneral(t, backend)
			})
		})
	}
}

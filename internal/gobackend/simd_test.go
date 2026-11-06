// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"runtime"
	"slices"
	"testing"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

func TestPriorityConstants(t *testing.T) {
	generic, typed, simd, arch := PriorityGeneric, PriorityTyped, PrioritySIMD, PriorityArch
	neon, avx2, avx512, sve, sme, user := PriorityNEON, PriorityAVX2, PriorityAVX512, PrioritySVE, PrioritySME, PriorityUser
	if !(generic < typed &&
		typed < simd &&
		simd < arch &&
		arch == neon &&
		arch == avx2 &&
		neon < sve &&
		avx2 < avx512 &&
		sve < sme &&
		sme < user) {
		t.Fatalf("unexpected priority ordering: Generic=%d Typed=%d SIMD=%d Arch=%d NEON=%d AVX2=%d AVX512=%d SVE=%d SME=%d User=%d",
			generic, typed, simd, arch, neon, avx2, avx512, sve, sme, user)
	}
}

func TestSIMDDetectionFlags(t *testing.T) {
	t.Logf("GOARCH=%s GOOS=%s: ActiveArchitectures=%v IsAVX2Allowed=%v IsAVX512Allowed=%v IsNEONAllowed=%v IsARM64FP16Allowed=%v IsARM64BF16Allowed=%v IsARM64DotProdAllowed=%v IsARM64I8MMAllowed=%v IsSVEAllowed=%v IsSMEAllowed=%v",
		runtime.GOARCH, runtime.GOOS, ActiveArchitectures,
		IsAVX2Allowed, IsAVX512Allowed,
		IsNEONAllowed, IsARM64FP16Allowed, IsARM64BF16Allowed,
		IsARM64DotProdAllowed, IsARM64I8MMAllowed,
		IsSVEAllowed, IsSMEAllowed)

	expectedOSArch := runtime.GOOS + "/" + runtime.GOARCH
	if len(ActiveArchitectures) == 0 || ActiveArchitectures[0] != expectedOSArch {
		t.Errorf("expected ActiveArchitectures[0] to be %q, got %v", expectedOSArch, ActiveArchitectures)
	}

	switch runtime.GOARCH {
	case "arm64":
		if IsAVX2Allowed || IsAVX512Allowed {
			t.Errorf("expected IsAVX2Allowed and IsAVX512Allowed to be false on arm64")
		}
	case "amd64":
		if IsNEONAllowed || IsARM64FP16Allowed || IsARM64BF16Allowed || IsARM64DotProdAllowed || IsARM64I8MMAllowed || IsSVEAllowed || IsSMEAllowed {
			t.Errorf("expected ARM64 flags to be false on amd64")
		}
	}
}

func TestArchDispatcherChaining(t *testing.T) {
	t.Run("ReduceTrailingSum", func(t *testing.T) {
		saved := slices.Clone(reduceTrailingSumArchEntries)
		defer func() {
			reduceTrailingSumArchEntries = saved
			rebuildReduceTrailingSumArchFn()
		}()
		reduceTrailingSumArchEntries = nil
		rebuildReduceTrailingSumArchFn()

		var callOrder []string
		SetReduceTrailingSumArchDispatcher(PriorityNEON, func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "NEON")
			return dtype == dtypes.Float32
		})
		SetReduceTrailingSumArchDispatcher(PrioritySVE, func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "SVE")
			return B >= 1024
		})

		fn := GetReduceTrailingSumArchDispatcher()
		if fn == nil {
			t.Fatal("expected non-nil dispatcher")
		}

		// Case 1: SVE handles B >= 1024 directly without calling NEON.
		callOrder = nil
		if !fn(nil, nil, 4, 1024, dtypes.Float32) {
			t.Errorf("expected true when SVE handles B=1024")
		}
		if !slices.Equal(callOrder, []string{"SVE"}) {
			t.Errorf("expected callOrder [SVE], got %v", callOrder)
		}

		// Case 2: SVE returns false for B=64, falls back to NEON which returns true for Float32.
		callOrder = nil
		if !fn(nil, nil, 4, 64, dtypes.Float32) {
			t.Errorf("expected true when falling back from SVE to NEON")
		}
		if !slices.Equal(callOrder, []string{"SVE", "NEON"}) {
			t.Errorf("expected callOrder [SVE NEON], got %v", callOrder)
		}

		// Case 3: Both return false for B=64 and Complex64.
		callOrder = nil
		if fn(nil, nil, 4, 64, dtypes.Complex64) {
			t.Errorf("expected false when neither SVE nor NEON handles Complex64")
		}
		if !slices.Equal(callOrder, []string{"SVE", "NEON"}) {
			t.Errorf("expected callOrder [SVE NEON], got %v", callOrder)
		}
	})

	t.Run("ReduceLeadingSum", func(t *testing.T) {
		saved := slices.Clone(reduceLeadingSumArchEntries)
		defer func() {
			reduceLeadingSumArchEntries = saved
			rebuildReduceLeadingSumArchFn()
		}()
		reduceLeadingSumArchEntries = nil
		rebuildReduceLeadingSumArchFn()

		var callOrder []string
		SetReduceLeadingSumArchDispatcher(PriorityNEON, func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "NEON")
			return true
		})
		SetReduceLeadingSumArchDispatcher(PrioritySME, func(operand, output *Buffer, A, B int, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "SME")
			return false
		})

		fn := GetReduceLeadingSumArchDispatcher()
		if !fn(nil, nil, 8, 16, dtypes.Float32) {
			t.Errorf("expected true from chained NEON dispatcher")
		}
		if !slices.Equal(callOrder, []string{"SME", "NEON"}) {
			t.Errorf("expected callOrder [SME NEON], got %v", callOrder)
		}
	})

	t.Run("BinaryTrailing", func(t *testing.T) {
		saved := slices.Clone(binaryTrailingArchEntries)
		defer func() {
			binaryTrailingArchEntries = saved
			rebuildBinaryTrailingArchFn()
		}()
		binaryTrailingArchEntries = nil
		rebuildBinaryTrailingArchFn()

		var callOrder []string
		SetBinaryTrailingArchDispatcher(PriorityNEON, func(op compute.OpType, isLHS bool, lhs, rhs, output *Buffer, A, B int, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "NEON")
			return true
		})
		SetBinaryTrailingArchDispatcher(PrioritySVE, func(op compute.OpType, isLHS bool, lhs, rhs, output *Buffer, A, B int, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "SVE")
			return false
		})

		fn := GetBinaryTrailingArchDispatcher()
		if !fn(compute.OpTypeAdd, false, nil, nil, nil, 4, 16, dtypes.Float32) {
			t.Errorf("expected true from chained NEON binary dispatcher")
		}
		if !slices.Equal(callOrder, []string{"SVE", "NEON"}) {
			t.Errorf("expected callOrder [SVE NEON], got %v", callOrder)
		}
	})

	t.Run("LayerNormTrailing", func(t *testing.T) {
		saved := slices.Clone(layerNormTrailingArchEntries)
		defer func() {
			layerNormTrailingArchEntries = saved
			rebuildLayerNormTrailingArchFn()
		}()
		layerNormTrailingArchEntries = nil
		rebuildLayerNormTrailingArchFn()

		var callOrder []string
		SetLayerNormTrailingArchDispatcher(PriorityNEON, func(inPtr, outPtr, gammaPtr, betaPtr unsafe.Pointer, outerSize, normSize int, epsilon float64, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "NEON")
			return true
		})
		SetLayerNormTrailingArchDispatcher(PrioritySVE, func(inPtr, outPtr, gammaPtr, betaPtr unsafe.Pointer, outerSize, normSize int, epsilon float64, dtype dtypes.DType) bool {
			callOrder = append(callOrder, "SVE")
			return false
		})

		fn := GetLayerNormTrailingArchDispatcher()
		if !fn(nil, nil, nil, nil, 4, 64, 1e-5, dtypes.Float32) {
			t.Errorf("expected true from chained LayerNorm dispatcher")
		}
		if !slices.Equal(callOrder, []string{"SVE", "NEON"}) {
			t.Errorf("expected callOrder [SVE NEON], got %v", callOrder)
		}
	})
}

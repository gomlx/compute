// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package nontransposed

// The AVX512 implementation must have one generate version per dtype pair, since generics are not supported in archsimd.
// (Maybe a later version with go-highway or midway will change that)
//
// For now we only have implemented (InputDType, OutputDType):
//
// - (Float32, Float32)
// - (Bfloat16, Float32)
import (
	"simd/archsimd"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/support/envutil"
)

// Auto-generate alternate specialized versions of AVX512 operations -- for half-precision input data types.
// NOT YET: go:generate go run ../../../cmd/alternates_generator -base=avx512_large.go -tags=bf16

var (
	// AVX512ParamsFloat32 are the parameters to use for Float32, tuned for the 16 registers implementations.
	AVX512ParamsFloat32 = CacheParams{
		LHSL1KernelRows:      4,   // Mr: Uses 4 ZMM registers for accumulation rows, this number must be a multiple of 4
		RHSL1KernelCols:      32,  // Nr: Uses 2 ZMM registers for accumulation cols, each holds 16 values
		PanelContractingSize: 128, // Kc: A strip fits in L1 cache
		LHSPanelCrossSize:    24,  // Mc: Fits in L2 cache (multiple of LHSL1KernelRows), multiple of LHSL1KernelRows, but usually just LHSL1KernelRows.
		RHSPanelCrossSize:    512, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols), multiple of RHSL1KernelRows.
	}

	// AVX512ParamsBFloat16 are the parameters to use for BFloat16, tuned for the 16 registers implementations.
	AVX512ParamsBFloat16 = CacheParams{
		LHSL1KernelRows:      4,   // Mr: Uses 4 ZMM registers for accumulation rows, this number must be a multiple of 4
		RHSL1KernelCols:      32,  // Nr: Uses 2 ZMM registers for accumulation cols, each holds 16 values
		PanelContractingSize: 128, // Kc: A strip fits in L1 cache
		LHSPanelCrossSize:    32,  // Mc: Fits in L2 cache (multiple of LHSL1KernelRows), multiple of LHSL1KernelRows, but usually just LHSL1KernelRows.
		RHSPanelCrossSize:    768, // Nc: Fits in L3 cache (multiple of RHSL1KernelCols), multiple of RHSL1KernelRows.
	}
)

func init() {
	if !envutil.MustReadBool(EnabledEnv, true) {
		return
	}

	allowed := envutil.MustReadBool(envutil.SIMD_AVX512_Env, true)
	if allowed && archsimd.X86.AVX512() {
		registerAVX512(false)
	}
}

func RegisterAVX512ForTests() {
	registerAVX512(true)
}

func registerAVX512(forTests bool) {
	dot.RegisterImplementation("NonTransposed-AVX512", dot.LayoutNonTransposed, dtypes.Float32, dtypes.Float32, avx512LargeFloat32, PriorityAVX512, forTests)
	// dot.RegisterImplementation("NonTransposed-AVX512", dot.LayoutNonTransposed, dtypes.BFloat16, dtypes.BFloat16, avx512LargeBFloat16, PriorityAVX512, forTests)
}

// Package simd implements various SIMD versions of matrix multiplication (used by DotGeneral),
// and despite the name, it includes a backoff no-SIMD implementations.
//
// It includes implementations for "non-transposed" ([N,K]x[K,M] -> [N,M]) and "transposed" ([N,K]x[M,K]->[N,M])
// layouts.
package simd

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/support/envutil"
	"k8s.io/klog/v2"
)

const (
	// EnabledEnv is the environment variable that controls whether the non-transposed
	// implementations are enabled.
	// It's on by default, and can be disabled by setting it to false.
	EnabledEnv = "GOMLX_DOT_SIMD"
)

// Block/packs parameters for current architecture.
type CacheParams struct {
	LHSL1KernelRows int // or Mr: number of lhs kernel rows going to registers.
	RHSL1KernelCols int // or Nr: Register Block Width

	PanelContractingSize int // Kc: LHS cols or RHS rows to fit in L2/L3
	LHSPanelCrossSize    int // Mc: L2 rows
	RHSPanelCrossSize    int // Nc: L3 cols
}

var (
	// Used for tests only.
	ForceSmallVariant = false
	ForceLargeVariant = false
)

const (
	PriorityNoSIMD = gobackend.PriorityTyped
	PriorityAVX2   = gobackend.PriorityArch
	PriorityAVX512 = gobackend.PriorityArch + 1
)

// NumberNonHalf includes the numbers we support in this package, excluding half-precision floats
//
// Notice it doesn't include the complex numbers. They will need to be supported as a separate class.
type NumberNonHalf = dtypes.NumberNotComplex

// Number includes all the numbers we support in this package: integers, floats, and half-precision floats.
// But this doesn't define the half-precision methods.
type Number interface {
	NumberNonHalf | dtypes.NumberHalfPrecision
}

func init() {
	avx512Enabled, err := envutil.ReadBool(envutil.SIMD_AVX512_Env, true)
	if err != nil {
		klog.Fatalf("Invalid value for %q: %+v", envutil.SIMD_AVX512_Env, err)
	}
	_ = avx512Enabled
}

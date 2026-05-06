package nontransposed

import (
	"github.com/gomlx/compute/support/envutil"
	"k8s.io/klog/v2"
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

func init() {
	avx512Enabled, err := envutil.ReadBool(envutil.SIMD_AVX512_Env, true)
	if err != nil {
		klog.Fatalf("Invalid value for %q: %+v", envutil.SIMD_AVX512_Env, err)
	}
	_ = avx512Enabled
}

// package matmul provides the base implementations of matrx multiply for DotGeneral. It includes a no-SIMD and various
// SIMD variations, support for "packing" for large matrices, and support for "non-transposed" ([N,K]x[K,M] -> [N,M])
// and "transposed" ([N,K]x[M,K]->[N,M]) layouts.
package matmul

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/support/envutil"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

const (
	// EnabledEnv is the environment variable that controls whether the default matmul
	// implementations are enabled.
	// It's on by default, and can be disabled by setting it to false.
	EnabledEnv = envutil.GoBackendDotMatmul
)

// Block/packs parameters for current architecture.
type CacheParams struct {
	// LHSL1KernelRows (or Mr), the number of lhs kernel rows going to registers.
	// Set to 2, 4, or multiples of 4.
	LHSL1KernelRows int

	// RHSL1KernelCols (or Nr), the number of rhs kernel columns going to registers.
	// For SIMD it will typically be large enough for one or two SIMD vector loads.
	RHSL1KernelCols int

	// PanelContractingSize (or Kc) is the largest size of the RHS and LHF panels (used when packing)
	// on the contracting dimension.
	// Selected to make the panel fit into L2/L3 caches.
	PanelContractingSize int

	// LHSPanelCrossSize (or Mc) is the "cross" size of the LHS and Output panels (used when packing).
	// Selected to make the panel fit into L2/L3 caches.
	LHSPanelCrossSize int // Mc: L2 rows

	// RHSPanelCrossSize (or Mc) is the "cross" size of the RHS and Output panels (used when packing).
	// Selected to make the panel fit into L2/L3 caches.
	RHSPanelCrossSize int // Nc: L3 cols
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

func init() {
	avx512Enabled, err := envutil.ReadBool(envutil.GoBackendSIMD_AVX512, true)
	if err != nil {
		klog.Fatalf("Invalid value for %q: %+v", envutil.GoBackendSIMD_AVX512, err)
	}
	_ = avx512Enabled
}

// ExecuteWithEpilogue executes matrix multiplication using the registered backend algorithm,
// and applies the epilogue (bias addition and activation) directly to the output while in cache.
func ExecuteWithEpilogue[I, O interface {
	gotype.Numeric | gotype.AnyHalfPrecision
}](
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []I,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []O,
	epilogue Epilogue[O],
) error {
	if backend.NoOps {
		return nil
	}
	inDType := dtypes.FromGenericsType[I]()
	outDType := dtypes.FromGenericsType[O]()
	reg := dot.FindRegisteredImplementation(layout, inDType, outDType)
	if reg == nil {
		return errors.Errorf("no registered matmul implementation for layout=%s, input=%s, output=%s",
			layout, inDType, outDType)
	}

	implFn, ok := reg.ImplFn().(dot.DotGeneralExecFn[I, O])
	if !ok {
		return errors.Errorf("invalid implementation function type for layout=%s, input=%s, output=%s",
			layout, inDType, outDType)
	}
	implFn(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)

	if epilogue.HasWork() {
		ApplyEpilogue(backend, output, batchSize, lhsCrossSize, rhsCrossSize, epilogue)
	}
	return nil
}


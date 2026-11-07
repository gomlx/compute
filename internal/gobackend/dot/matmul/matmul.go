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
	PriorityNEON   = gobackend.PriorityNEON
	PrioritySVE    = gobackend.PrioritySVE
	PrioritySME    = gobackend.PrioritySME
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
	nodeDataOpt ...*dot.NodeData,
) error {
	if backend.NoOps {
		return nil
	}
	var nodeData *dot.NodeData
	if len(nodeDataOpt) > 0 {
		nodeData = nodeDataOpt[0]
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
	implFn(backend, layout, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output, nodeData)

	if epilogue.HasWork() {
		ApplyEpilogue(backend, output, batchSize, lhsCrossSize, rhsCrossSize, epilogue)
	}
	return nil
}

// MultiRHSTarget describes one RHS matrix, output buffer, epilogue, and optional NodeData
// for ExecuteMultiRHSWithEpilogue.
type MultiRHSTarget[I, O interface {
	gotype.Numeric | gotype.AnyHalfPrecision
}] struct {
	RHS          []I
	RHSCrossSize int
	Output       []O
	Epilogue     Epilogue[O]
	NodeData     *dot.NodeData
}

// MultiRHSExecFn is the signature of a multi-RHS matrix multiplication implementation
// that shares packed LHS panels across all targets and applies each target's Epilogue inline.
// It returns false if the caller should fall back to sequential ExecuteWithEpilogue calls.
type MultiRHSExecFn[I, O interface {
	gotype.Numeric | gotype.AnyHalfPrecision
}] func(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs []I,
	batchSize, lhsCrossSize, contractingSize int,
	targets []MultiRHSTarget[I, O],
) bool

type multiRHSKey struct {
	name                    string
	layout                  dot.Layout
	inputDType, outputDType dtypes.DType
}

var registeredMultiRHSImplementations = make(map[multiRHSKey]any)

// RegisterMultiRHSImplementation registers a multi-RHS matmul implementation associated with a dot registration name.
func RegisterMultiRHSImplementation[I, O interface {
	gotype.Numeric | gotype.AnyHalfPrecision
}](
	name string,
	layout dot.Layout,
	inputDType, outputDType dtypes.DType,
	implFn MultiRHSExecFn[I, O],
) {
	key := multiRHSKey{
		name:        name,
		layout:      layout,
		inputDType:  inputDType,
		outputDType: outputDType,
	}
	registeredMultiRHSImplementations[key] = implFn
}

// ExecuteMultiRHSWithEpilogue multiplies the same LHS matrix against multiple RHS targets (e.g., Q, K, V
// in FusedAttentionQKVProjection), packing LHS panels only once per row-chunk in L2 cache and applying
// each target's epilogue inline while the output rows are hot in L1/L2 cache.
func ExecuteMultiRHSWithEpilogue[I, O interface {
	gotype.Numeric | gotype.AnyHalfPrecision
}](
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs []I,
	batchSize, lhsCrossSize, contractingSize int,
	targets []MultiRHSTarget[I, O],
) error {
	if backend.NoOps || len(targets) == 0 {
		return nil
	}
	inDType := dtypes.FromGenericsType[I]()
	outDType := dtypes.FromGenericsType[O]()
	reg := dot.FindRegisteredImplementation(layout, inDType, outDType)
	if reg == nil {
		return errors.Errorf("no registered matmul implementation for layout=%s, input=%s, output=%s",
			layout, inDType, outDType)
	}

	if len(targets) > 1 {
		key := multiRHSKey{
			name:        reg.Name(),
			layout:      layout,
			inputDType:  inDType,
			outputDType: outDType,
		}
		if multiAny, ok := registeredMultiRHSImplementations[key]; ok {
			if multiFn, ok := multiAny.(MultiRHSExecFn[I, O]); ok {
				if multiFn(backend, layout, lhs, batchSize, lhsCrossSize, contractingSize, targets) {
					return nil
				}
			}
		}
	}

	for i := range targets {
		t := &targets[i]
		if err := ExecuteWithEpilogue(
			backend, layout, lhs, t.RHS,
			batchSize, lhsCrossSize, t.RHSCrossSize, contractingSize,
			t.Output, t.Epilogue, t.NodeData,
		); err != nil {
			return err
		}
	}
	return nil
}


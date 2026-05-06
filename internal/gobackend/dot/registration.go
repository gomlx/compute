package dot

import (
	"fmt"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/workerspool"
	"k8s.io/klog/v2"
)

// ImplementationKey reprsents a unique key for a DotGeneral algorithm implementation.
type ImplementationKey struct {
	Layout                  Layout
	InputDType, OutputDType dtypes.DType
}

func (key *ImplementationKey) String() string {
	return fmt.Sprintf("(Layout=%s, InputDType=%s, OutputDType=%s)",
		key.Layout, key.InputDType, key.OutputDType)
}

// DotGeneralExecFn is the signature of a function that implements a specific ImplementationKey
// for DotGeneral.
//
// The lhs and rhs inputs must be shaped according to the layout (batchSize, cross, contracting) and will
// be provided in the specified order.
//
// The output will always be shaped [batchSize, lhsCross, rhsCross], and the required space must have been
// pre-allocated.
type DotGeneralExecFn[I, O dtypes.Supported] func(
	lhs, rhs []I,
	batchSize, lhsCross, lhsContracting, rhsCross, rhsContracting,
	output []O,
	pool *workerspool.Pool)

type ImplementationRegistration struct {
	// implFn holds the DotGeneralExecFn[InputDType, OutputDType] for the specific ImplementationKey.
	name     string
	implFn   any
	priority gobackend.RegisterPriority
}

var (
	registeredAlgorithms = make(map[ImplementationKey]*ImplementationRegistration)
)

// RegisterImplementation registers a DotGeneral algorithm implementation for a specific
// layout and pair of dtypes.
//
// The name is simply a human-readable identifier for the algorithm, used in logging.
//
// This should only be called during initialization.
func RegisterImplementation[I, O dtypes.Supported](
	name string,
	layout Layout, inputDType, outputDType dtypes.DType,
	implFn DotGeneralExecFn[I, O], priority gobackend.RegisterPriority) {
	key := ImplementationKey{Layout: layout, InputDType: inputDType, OutputDType: outputDType}
	current, found := registeredAlgorithms[key]
	if found && priority < current.priority {
		klog.V(1).Infof("DotGeneral algorithm %q for key %s not registered since priority %d is lower than current priority %d",
			name, key, priority, current.priority)
		return
	}
	registeredAlgorithms[key] = &ImplementationRegistration{
		name:     name,
		implFn:   implFn,
		priority: priority,
	}
}

// FindRegisteredImplementation returns the registered implementation for the given
// layout and dtypes, or nil if no implementation is registered.
//
// This can be used during the building of the graph, and the result cached for execution.
func FindRegisteredImplementation(layout Layout, inputDType, outputDType dtypes.DType) *ImplementationRegistration {
	key := ImplementationKey{Layout: layout, InputDType: inputDType, OutputDType: outputDType}
	return registeredAlgorithms[key]
}

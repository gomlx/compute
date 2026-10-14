// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package compute

import (
	"github.com/gomlx/compute/shapes"
)

// Executable is the API for compiled programs ready to execute.
type Executable interface {
	// Finalize immediately frees resources associated to the executable.
	//
	// It is the responsibility of the caller to guarantee that there are no unfinished executions
	// (calls to [Executable.Execute]) before calling Finalize. Calling Finalize while an execution
	// is still running may lead to undefined behavior and potentially crash depending on the backend.
	//
	// Calling Finalize is optional: one can always leave it to the garbage collector (GC), which will
	// properly clean up native resources eventually. Alternatively, callers can rely on higher-level
	// coordinator objects such as graph.Exec or model.Exec to coordinate safe concurrent execution and teardown.
	Finalize()

	// Inputs returns the parameters' names and shapes, in order created by the Builder.Parameter calls.
	Inputs() (names []string, inputShapes []shapes.Shape)

	// Outputs return the computation's output shapes, in the order given to the Builder.Compile call.
	Outputs() (outputShapes []shapes.Shape)

	// Execute the computation.
	// The number and shapes of the inputs must match those of the Executable parameters (returned by [Executable.Inputs]).
	//
	// The inputs marked as donate will become invalid after use.
	// This is useful if the input buffer is no longer needed or if updating a variable
	// so its Buffer space can be reused as an output Buffer.
	//
	// Donated buffers are no longer valid after the call.
	// If donate is nil, it is assumed to be false for all buffers, and no buffer is donated.
	//
	// For portable computations (not compiled with a fixed device assignment), the execution runs on the defaultDevice.
	// For non-portable computations (where the device assignment is fixed), the defaultDevice is ignored.
	//
	// For SPMD distributed computations (see [Builder.DistributedSPMD]), the executable is replicated on each device.
	// There will be multiple inputs per executable parameter, one per device.
	// They are organized as "device-major", that is the input for parameter i on device j is given by inputs[j*numParams + i].
	Execute(inputs []Buffer, donate []bool, defaultDevice DeviceNum) ([]Buffer, error)
}

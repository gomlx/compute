// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"sync"
	"sync/atomic"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// FunctionExecutable contains pre-compiled execution information for any function.
// This is used for both the main function and closures, unifying their execution model.
type FunctionExecutable struct {
	Backend *Backend
	Builder *Builder

	// Function is the source Function this was compiled from.
	Function *Function

	// NumNodesToProcess is the max(outputs.idx)+1.
	// Arrays are sized to this to allow direct idx indexing.
	NumNodesToProcess int

	// Schedule represents the order in which nodes should be executed.
	// This should be optimized to minimize the latency/temporary memory.
	Schedule []int

	// NumUses tracks how many times each node's result is used (indexed by idx).
	NumUses []int

	// Dependents maps each node (by idx) to the list of dependent node idxs.
	Dependents [][]int

	// OutputNodes are the nodes that produce the function's outputs.
	OutputNodes []*Node

	// MaxInputs is the maximum number of inputs any node has.
	MaxInputs int

	// ExecutionBuffersPool allows reuse of execution buffers.
	ExecutionBuffersPool sync.Pool
}

// newFunctionExecutable creates a FunctionExecutable for the given function.
// The function must have Return() called (f.returned == true).
func newFunctionExecutable(f *Function) (*FunctionExecutable, error) {
	if !f.IsReturned {
		return nil, errors.Errorf("function must have Return() called before compilation")
	}

	// Calculate numNodesToProcess from outputs.
	// This has the benefit of immediately discarding nodes with idx > max(outputs.idx),
	// meaning nodes that outputs don't depend on.
	var numNodesToProcess int
	for _, output := range f.Outputs {
		numNodesToProcess = max(numNodesToProcess, output.Index+1)
	}

	fe := &FunctionExecutable{
		Backend:           f.RawBuilder.Backend,
		Builder:           f.RawBuilder,
		Function:          f,
		OutputNodes:       f.Outputs,
		NumNodesToProcess: numNodesToProcess,
		NumUses:           make([]int, numNodesToProcess),
		Dependents:        make([][]int, numNodesToProcess),
	}

	// Find max inputs (the maximum number of inputs any single node has),
	// including for captured inputs, and count uses/dependents
	for nodeIdx := range numNodesToProcess {
		node := f.Nodes[nodeIdx]
		// Total inputs = regular inputs + all captured inputs across closures
		totalCaptured := 0
		for _, closureCaptures := range node.CapturedInputs {
			totalCaptured += len(closureCaptures)
		}
		totalInputs := len(node.Inputs) + totalCaptured
		fe.MaxInputs = max(fe.MaxInputs, totalInputs)
	}

	// Recursively count uses for each node starting from outputs.
	for _, output := range f.Outputs {
		fe.countNodeUsesAndDependents(output)
	}

	// Initialize execution buffers pool
	fe.ExecutionBuffersPool = sync.Pool{
		New: func() any {
			return &funcExecBuffers{
				results:       make([]*Buffer, numNodesToProcess),
				numUsed:       make([]atomic.Int32, numNodesToProcess),
				owned:         make([]bool, numNodesToProcess),
				remainingDeps: make([]atomic.Int32, numNodesToProcess),
				readyQueue:    make(chan int, numNodesToProcess+10),
			}
		},
	}

	fe.InitSchedule()

	if klog.V(1).Enabled() {
		estMem := fe.EstimatedTemporaryMemory()
		var memStr string
		if estMem == int64(shapes.DynamicDim) {
			memStr = "dynamic (unknown at compile time)"
		} else {
			memStr = humanize.Bytes(estMem)
		}
		klog.Infof("* Compiling function %q: estimated max temporary memory: %s\n",
			fe.Function.Name(), memStr)
	}
	return fe, nil
}

// countNodeUsesAndDependents recursively counts how many times a node is used.
// It tracks both regular inputs and captured inputs (for closure-calling ops).
func (fe *FunctionExecutable) countNodeUsesAndDependents(node *Node) {
	nodeIdx := node.Index
	fe.NumUses[nodeIdx]++
	if fe.NumUses[nodeIdx] == 1 {
		// On the first visit, recursively traverse inputs of the node.
		for _, input := range node.Inputs {
			fe.Dependents[input.Index] = append(fe.Dependents[input.Index], nodeIdx)
			fe.countNodeUsesAndDependents(input)
		}
		// Also track captured inputs for closure-calling ops (If, While, Sort, etc.).
		// This ensures captured values are properly tracked in the dependency graph
		// so they can be freed when no longer needed.
		for _, closureCaptures := range node.CapturedInputs {
			for _, capturedInput := range closureCaptures {
				fe.Dependents[capturedInput.Index] = append(fe.Dependents[capturedInput.Index], nodeIdx)
				fe.countNodeUsesAndDependents(capturedInput)
			}
		}
	}
}

// funcExecBuffers holds intermediate results during function execution.
type funcExecBuffers struct {
	// results hold the calculated computations at each step (indexed by idx).
	results []*Buffer

	// numUsed tracks how many times each node has been used already.
	// Uses atomic.Int32 to allow safe concurrent reads in ownership checks.
	numUsed []atomic.Int32

	// owned indicates whether the corresponding buffer is owned by the executor.
	//
	// Notice different rows are written by different threads in parallel mode, but since
	// each row is 1 byte, and Go's thread sanitizer has an 8-bytes resolution, it may trigger
	// false race conditions. See discussion in github.com/gomlx/gomlx/issues/387
	//
	// Future: use a []int (int is 64 bits/8 bytes) instead of bool to avoid the issue?
	owned []bool

	// remainingDeps is the number of remaining dependencies for each node.
	// Uses atomic.Int32 to allow lock-free concurrent decrements in parallel execution.
	remainingDeps []atomic.Int32

	// opsExecutionType can be sequential or parallel.
	opsExecutionType OpsExecutionType

	// Sequential execution-only: reused for each op.
	opInputBuffers []*Buffer
	opInputsOwned  []bool

	// Parallel execution only.
	//
	// readyQueue delivers node indices whose remainingDeps have reached zero
	// to executor goroutines. It is closed by stopExecutionFn to signal
	// termination (either normal completion or error).
	//
	// Invariants:
	//   - Exactly-once dispatch: each node index is sent at most once, because
	//     atomic.Int32.Add(-1) on remainingDeps returns 0 for exactly one
	//     goroutine.
	//   - No send-after-close: nodeCompleted dispatches ready nodes (phase 2)
	//     BEFORE incrementing the completion counter (phase 3). A dispatched
	//     node is not yet completed, so at least 2 completion units are
	//     outstanding (this node's increment + the dispatched node's future
	//     increment), guaranteeing completed < expected and the channel is
	//     still open at dispatch time.
	//   - Capacity: sized to NumNodesToProcess+10, which exceeds the maximum
	//     number of concurrently ready nodes, so sends never block.
	readyQueue chan int
}

// Execute runs the compiled function with the given inputs.
// The inputs must match the function's parameters in count and shape.
// capturedInputs are the values captured from parent scopes (for closures).
// donateCaptures indicates which captured inputs can be donated to the closure.
// If donateCaptures is nil, no captured inputs will be donated.
// spec is the shape specialization for the current execution, or nil for static graphs.
func (fe *FunctionExecutable) Execute(backend *Backend, inputs []*Buffer, donate []bool, capturedInputs []*Buffer,
	donateCaptures []bool, spec *ShapeSpecialization) ([]*Buffer, error) {
	// Use function's parameters (not builder.inputs) for proper function/closure support
	funcParams := fe.Function.Parameters
	if len(inputs) != len(funcParams) {
		return nil, errors.Errorf("function expects %d inputs, got %d",
			len(funcParams), len(inputs))
	}

	// Validate captured inputs count
	if len(capturedInputs) != len(fe.Function.CapturedLocalNodes) {
		return nil, errors.Errorf("function expects %d captured values, got %d",
			len(fe.Function.CapturedLocalNodes), len(capturedInputs))
	}

	// donate defaults to false
	if len(donate) == 0 {
		donate = make([]bool, len(inputs))
	}

	// donateCaptures defaults to false (no donation)
	if len(donateCaptures) == 0 {
		donateCaptures = make([]bool, len(capturedInputs))
	}

	// Get execution buffers from pool and reset
	execBuf := fe.ExecutionBuffersPool.Get().(*funcExecBuffers)
	execBuf.readyQueue = make(chan int, fe.NumNodesToProcess+10)
	for i := range fe.NumNodesToProcess {
		execBuf.numUsed[i].Store(0)
		execBuf.owned[i] = false
		execBuf.results[i] = nil
		execBuf.remainingDeps[i].Store(0)
	}

	// Set up parameters from inputs using idx directly
	for i, inputNode := range funcParams {
		inputIdx := inputNode.Index
		if inputIdx >= fe.NumNodesToProcess || fe.NumUses[inputIdx] == 0 {
			if donate[i] {
				backend.PutBuffer(inputs[i])
			}
			continue
		}
		execBuf.results[inputIdx] = inputs[i]
		execBuf.owned[inputIdx] = donate[i]
	}

	// Set up captured values from parent scope.
	// If donateCaptures[i] is true, the closure takes ownership of the buffer.
	for i, captureNode := range fe.Function.CapturedLocalNodes {
		captureIdx := captureNode.Index
		if captureIdx >= fe.NumNodesToProcess || fe.NumUses[captureIdx] == 0 {
			if donateCaptures[i] {
				backend.PutBuffer(capturedInputs[i])
			}
			continue
		}
		execBuf.results[captureIdx] = capturedInputs[i]
		execBuf.owned[captureIdx] = donateCaptures[i]
	}

	// Decide execution mode
	executionMode := backend.OpsExecutionType
	if executionMode == OpsExecutionDynamic {
		if backend.NumLiveExecutions.Load() <= 1 {
			executionMode = OpsExecutionParallel
		} else {
			executionMode = OpsExecutionSequential
		}
	}
	execBuf.opsExecutionType = executionMode

	// Execute
	var err error
	if executionMode == OpsExecutionSequential {
		err = fe.executeSequentially(backend, execBuf, spec)
	} else {
		err = fe.executeParallel(backend, execBuf, spec)
	}
	if err != nil {
		fe.ExecutionBuffersPool.Put(execBuf)
		return nil, err
	}

	// Collect outputs
	outputs := make([]*Buffer, len(fe.OutputNodes))
	for i, outNode := range fe.OutputNodes {
		outIdx := outNode.Index
		outputs[i] = execBuf.results[outIdx]
		if outputs[i] == nil {
			fe.ExecutionBuffersPool.Put(execBuf)
			return nil, errors.Errorf("output %d not computed", i)
		}
		if !execBuf.owned[outIdx] {
			// Clone the buffer since we don't own it
			outputs[i], err = backend.CloneBuffer(execBuf.results[outIdx])
			if err != nil {
				return nil, err
			}
		} else {
			// We owned it, and now we are giving it to the user.
			// Return a shallow clone so the user can have a fresh handle,
			// while keeping the original handle (if it was a donated input) unchanged.
			//
			// If we return the same pointer, and the user calls Finalize() on it (which clears
			// the memory), then the user would be destroying the output they just received.
			outputs[i] = execBuf.results[outIdx].shallowClone()
		}
		execBuf.results[outIdx] = nil // Set to nil so it's not put back into the pool in the cleanup loop below.
	}

	// Free any remaining owned buffers that weren't outputs
	for idx, buf := range execBuf.results {
		if buf != nil {
			if execBuf.owned[idx] {
				backend.PutBuffer(buf)
			}
			execBuf.results[idx] = nil
		}
	}

	fe.ExecutionBuffersPool.Put(execBuf)
	return outputs, nil
}

// executeSequentially executes nodes one after another in topological order.
func (fe *FunctionExecutable) executeSequentially(backend *Backend, execBuf *funcExecBuffers, spec *ShapeSpecialization) error {
	// Pre-allocate input buffers for reuse
	execBuf.opInputBuffers = make([]*Buffer, fe.MaxInputs)
	execBuf.opInputsOwned = make([]bool, fe.MaxInputs)
	defer func() {
		execBuf.opInputBuffers = nil
		execBuf.opInputsOwned = nil
	}()

	for _, nodeIdx := range fe.Schedule {
		if execBuf.results[nodeIdx] != nil {
			// Already computed (parameter)
			continue
		}
		if fe.NumUses[nodeIdx] == 0 {
			// Not used by any output: this effectively performs a dead code elimination (DCE).
			continue
		}

		node := fe.Function.Nodes[nodeIdx]
		if spec != nil {
			node = spec.resolvedNodes[nodeIdx]
		}
		if err := fe.executeNode(backend, node, execBuf, spec); err != nil {
			return err
		}
	}
	return nil
}

// nodeCompleted records completion of nodeIdx and its sub-outputs (for multi-output ops),
// updates dependencies of dependent nodes, enqueues newly ready nodes, and checks if execution is done.
// If exactly one dependent node becomes ready and localContinuation is not nil and currently -1,
// it sets *localContinuation to that node so the caller executor can immediately continue executing it.
//
// This function is fully lock-free. It operates in three phases:
//  1. Atomic dependency decrements — uses atomic.Int32.Add(-1) on remainingDeps to determine
//     which dependent nodes have all their inputs satisfied (newReady).
//  2. Dispatch ready nodes — sends newly-ready nodes to readyQueue or sets localContinuation.
//     This MUST happen before phase 3, which may close the channel.
//  3. Atomic completion count — increments the completed counter atomically. If the total
//     reaches expected, calls stopExecutionFn to close readyQueue and signal termination.
//
// Memory ordering: the atomic Add on remainingDeps provides happens-before edges that guarantee
// result writes from producer goroutines are visible to consumer goroutines (per Go's memory model).
func (fe *FunctionExecutable) nodeCompleted(execBuf *funcExecBuffers, nodeIdx int,
	completed *atomic.Int32, expected int32, stopExecutionFn func(), localContinuation *int, spec *ShapeSpecialization) {
	node := fe.Function.Nodes[nodeIdx]
	if spec != nil {
		node = spec.resolvedNodes[nodeIdx]
	}

	// Phase 1: Lock-free dependency tracking using atomic decrements.
	// Collect dependent nodes whose remaining dependency count has reached zero.
	var newReady []int
	if node.IsMultiOutputs() {
		// Handle multi-output nodes: update dependents of each output.
		for _, outputNode := range node.MultiOutputsNodes {
			outputIdx := outputNode.Index
			if outputIdx >= fe.NumNodesToProcess || fe.NumUses[outputIdx] == 0 {
				continue
			}
			for _, depIdx := range fe.Dependents[outputIdx] {
				if execBuf.remainingDeps[depIdx].Add(-1) == 0 {
					newReady = append(newReady, depIdx)
				}
			}
		}
	} else {
		// Single output node.
		for _, depIdx := range fe.Dependents[nodeIdx] {
			if execBuf.remainingDeps[depIdx].Add(-1) == 0 {
				newReady = append(newReady, depIdx)
			}
		}
	}

	// Phase 2: Dispatch ready nodes.
	// This must happen BEFORE phase 3 which may close the channel.
	// Safety proof: at dispatch time, at least 2 completion units are outstanding
	// (this node's increment and the dispatched node's future increment), so
	// completed < expected and the channel is guaranteed to still be open.
	for _, depIdx := range newReady {
		if localContinuation != nil && *localContinuation == -1 {
			*localContinuation = depIdx
		} else {
			execBuf.readyQueue <- depIdx
		}
	}

	// Phase 3: Atomic completion tracking.
	// Compute total increment: 1 for the executed node itself, plus 1 for each
	// active multi-output sub-node (which are "completed" by their parent).
	increment := int32(1)
	if node.IsMultiOutputs() {
		for _, outputNode := range node.MultiOutputsNodes {
			if outputNode.Index < fe.NumNodesToProcess && fe.NumUses[outputNode.Index] > 0 {
				increment++
			}
		}
	}
	if completed.Add(increment) >= expected {
		stopExecutionFn()
	}
}

// runExecutor loops executing nodes until the readyQueue is closed or an error occurs.
// When executing a node unlocks another node, it preferentially continues executing it
// locally without routing through the channel.
func (fe *FunctionExecutable) runExecutor(backend *Backend, execBuf *funcExecBuffers, spec *ShapeSpecialization,
	completed *atomic.Int32, expected int32, stopExecutionFn func(), appendErrorFn func(error)) {
	currNode := -1

	for {
		if currNode == -1 {
			var ok bool
			currNode, ok = <-execBuf.readyQueue
			if !ok {
				// readyQueue closed: computation completed or errored out.
				return
			}
		}

		nodeIdx := currNode
		currNode = -1

		node := fe.Function.Nodes[nodeIdx]
		if spec != nil {
			node = spec.resolvedNodes[nodeIdx]
		}

		var execErr error
		if execBuf.results[nodeIdx] == nil && fe.NumUses[nodeIdx] > 0 {
			execErr = fe.executeNode(backend, node, execBuf, spec)
		}

		if execErr != nil {
			appendErrorFn(execErr)
			return
		}

		// Update dependents. If currNode is still -1, nodeCompleted will set it if a dependent is ready.
		fe.nodeCompleted(execBuf, nodeIdx, completed, expected, stopExecutionFn, &currNode, spec)
	}
}

// executeParallel executes nodes concurrently across a fixed pool of persistent worker executors.
// When an executor finishes a node and only one dependent becomes ready, it continues executing it
// directly on the same goroutine (preserving CPU cache locality), only pushing to the ready channel
// when branching causes multiple nodes to become ready simultaneously.
func (fe *FunctionExecutable) executeParallel(backend *Backend, execBuf *funcExecBuffers, spec *ShapeSpecialization) error {
	var (
		collectErrors   []error
		errMu           sync.Mutex
		wg              sync.WaitGroup
		expected        int32
		completed       atomic.Int32
		stopExecutionFn func()
	)

	stopExecutionFn = sync.OnceFunc(func() {
		close(execBuf.readyQueue)
	})

	appendErrorFn := func(err error) {
		errMu.Lock()
		collectErrors = append(collectErrors, err)
		errMu.Unlock()
		stopExecutionFn()
	}

	// Count expected nodes and initialize dependencies.
	// Dependencies include both regular inputs and captured inputs.
	var initialReady []int
	for _, nodeIdx := range fe.Schedule {
		if fe.NumUses[nodeIdx] > 0 {
			expected++
			node := fe.Function.Nodes[nodeIdx]
			totalCaptured := 0
			for _, closureCaptures := range node.CapturedInputs {
				totalCaptured += len(closureCaptures)
			}
			depCount := int32(len(node.Inputs) + totalCaptured)
			execBuf.remainingDeps[nodeIdx].Store(depCount)
			if depCount == 0 {
				initialReady = append(initialReady, nodeIdx)
			}
		}
	}

	if expected == 0 {
		return nil
	}

	// Determine number of executor goroutines to launch:
	// Use backend.NumExecutors, but cap it at maxParallelism and at least 1.
	numExecutors := backend.NumExecutors
	if numExecutors <= 0 {
		numExecutors = DefaultNumExecutors
	}
	if maxP := backend.Workers.MaxParallelism(); maxP > 0 {
		numExecutors = min(numExecutors, maxP)
	}

	// Enqueue initial ready nodes.
	for _, nodeIdx := range initialReady {
		execBuf.readyQueue <- nodeIdx
	}

	// Launch background executors via backend.Workers (numExecutors - 1 workers).
	// The current caller goroutine serves as one of the executors.
	for range numExecutors - 1 {
		wg.Add(1)
		started := backend.Workers.StartIfAvailable(func() {
			defer wg.Done()
			fe.runExecutor(backend, execBuf, spec, &completed, expected, stopExecutionFn, appendErrorFn)
		})
		if !started {
			wg.Done()
			break
		}
	}

	// Run the current caller goroutine as an executor as well.
	fe.runExecutor(backend, execBuf, spec, &completed, expected, stopExecutionFn, appendErrorFn)

	// Wait for all background executors to finish.
	wg.Wait()

	if len(collectErrors) > 0 {
		return collectErrors[0]
	}
	return nil
}

// executeNode executes a single node and stores its result.
func (fe *FunctionExecutable) executeNode(backend *Backend, node *Node, execBuf *funcExecBuffers, spec *ShapeSpecialization) error {
	nodeIdx := node.Index

	// Handle constants specially
	if node.OpType == compute.OpTypeConstant {
		execBuf.owned[nodeIdx] = false
		execBuf.results[nodeIdx] = node.Data.(*Buffer)
		return nil
	}

	// Note: OpTypeParameter and OpTypeCapturedValue nodes have their results
	// set up in Execute() and should never reach executeNode.
	// We don't check for them here for performance (this is the inner execution loop).

	// Prepare inputs
	numInputs := len(node.Inputs)
	var (
		inputBuffers []*Buffer
		inputsOwned  []bool
	)
	if execBuf.opInputBuffers != nil {
		inputBuffers = execBuf.opInputBuffers[:numInputs]
		inputsOwned = execBuf.opInputsOwned[:numInputs]
	} else {
		inputBuffers = make([]*Buffer, numInputs)
		inputsOwned = make([]bool, numInputs)
	}

	// Gather inputs. In parallel mode, we do NOT hold a lock here - the dependency
	// tracking ensures inputs are ready. The lock is only used in cleanup.
	for i, input := range node.Inputs {
		inputIdx := input.Index
		inputBuffers[i] = execBuf.results[inputIdx]
		if inputBuffers[i] == nil {
			return errors.Errorf("input %d for node %s not computed yet", i, node.OpType)
		}
		if !inputBuffers[i].InUse {
			return errors.Errorf("input %d for node %s has been released already!?", i, node.OpType)
		}
		// Only "own" the input if this is the last use of it.
		// The atomic Load is safe for concurrent access - if we miss ownership,
		// the buffer just won't be reused in-place. The important thing
		// is we don't free the buffer until all users have finished (handled in cleanup).
		inputsOwned[i] = execBuf.owned[inputIdx] &&
			fe.NumUses[inputIdx]-int(execBuf.numUsed[inputIdx].Load()) == 1
	}

	// Check for closure executor first (If, While, Sort).
	// Closure executors receive captured inputs separately with explicit ownership tracking.
	closureExecutor := NodeClosureExecutors[node.OpType]
	switch {
	case closureExecutor != nil:
		// Build []ClosureInputs from node.capturedInputs (already grouped per closure).
		closureInputs := make([]ClosureInputs, len(node.CapturedInputs))
		for closureIdx, closureCaptures := range node.CapturedInputs {
			closureInputs[closureIdx] = ClosureInputs{
				Buffers: make([]*Buffer, len(closureCaptures)),
				Owned:   make([]bool, len(closureCaptures)),
			}
			for i, capturedNode := range closureCaptures {
				capturedIdx := capturedNode.Index
				closureInputs[closureIdx].Buffers[i] = execBuf.results[capturedIdx]
				if closureInputs[closureIdx].Buffers[i] == nil {
					return errors.Errorf("captured input %d for closure %d of node %s not computed yet",
						i, closureIdx, node.OpType)
				}
				// Only "own" the captured input if this is the last use of it.
				closureInputs[closureIdx].Owned[i] = execBuf.owned[capturedIdx] &&
					fe.NumUses[capturedIdx]-int(execBuf.numUsed[capturedIdx].Load()) == 1
			}
		}

		outputBuffers, err := closureExecutor(backend, node, inputBuffers, inputsOwned, closureInputs)
		if err != nil {
			return errors.WithMessagef(err, "executing closure op %s", node.OpType)
		}
		if klog.V(1).Enabled() {
			for _, outputBuf := range outputBuffers {
				if err := outputBuf.Check(); err != nil {
					klog.Errorf("executing closure op %s: output buffer %p: %v", node.OpType, outputBuf, err)
				}
			}
		}

		// Check if any captured inputs were consumed (set to nil by the executor).
		// If so, mark execBuf.results as nil to indicate they're no longer available.
		for closureIdx, closureCaptures := range node.CapturedInputs {
			for i, capturedNode := range closureCaptures {
				if closureInputs[closureIdx].Buffers[i] == nil {
					execBuf.results[capturedNode.Index] = nil
				}
			}
		}

		// Write outputs to execBuf (closure ops are always multi-output style), or free those no longer
		// needed.
		for outputIdx, outputBuf := range outputBuffers {
			outputNode := node.MultiOutputsNodes[outputIdx]
			outputNodeIdx := outputNode.Index
			if outputNodeIdx >= fe.NumNodesToProcess || fe.NumUses[outputNodeIdx] == 0 {
				backend.PutBuffer(outputBuf)
				continue
			}
			execBuf.results[outputNodeIdx] = outputBuf
			execBuf.owned[outputNodeIdx] = true
		}

	case node.IsMultiOutputs():
		// Execute the node
		multiExecutor := MultiOutputsNodeExecutors[node.OpType]
		if multiExecutor == nil {
			return errors.Errorf("no multi-output executor for op %s", node.OpType)
		}

		outputBuffers, err := multiExecutor(backend, node, inputBuffers, inputsOwned)
		if err != nil {
			return errors.WithMessagef(err, "executing multi-output %s", node.OpType)
		}
		if klog.V(1).Enabled() {
			for _, outputBuf := range outputBuffers {
				if err := outputBuf.Check(); err != nil {
					klog.Errorf("executing multi-output %s: output buffer %p: %v", node.OpType, outputBuf, err)
				}
			}
		}

		// Write outputs to execBuf (multi-output ops are always multi-output style), or free those no longer
		// needed.
		for outputIdx, outputBuf := range outputBuffers {
			outputNode := node.MultiOutputsNodes[outputIdx]
			outputNodeIdx := outputNode.Index
			if outputNodeIdx >= fe.NumNodesToProcess || fe.NumUses[outputNodeIdx] == 0 {
				// Output of node is not used by any other node, we can immediately release it.
				backend.PutBuffer(outputBuf)
				continue
			}
			execBuf.results[outputNodeIdx] = outputBuf
			execBuf.owned[outputNodeIdx] = true
		}

	default:
		// Single output node:
		executors := nodeExecutors[node.OpType]
		if len(executors) == 0 {
			return errors.Errorf("no executor for op %s", node.OpType)
		}

		var result *Buffer
		var err error

		// Fast path: cached winning executor index (1-based: idx+1).
		cachedIdx := int(node.cachedExecutorIdx.Load())
		if cachedIdx > 0 {
			result, err = executors[cachedIdx-1].executor(backend, node, inputBuffers, inputsOwned)
			if err == ErrFallback {
				node.cachedExecutorIdx.Store(0)
				cachedIdx = 0
			}
		}
		if cachedIdx == 0 {
			for i, entry := range executors {
				result, err = entry.executor(backend, node, inputBuffers, inputsOwned)
				if err == ErrFallback {
					continue
				}
				if err == nil {
					// Cache executor in case of success.
					node.cachedExecutorIdx.Store(int32(i + 1))
				}
				break
			}
		}
		if err != nil {
			return errors.WithMessagef(err, "executing %s", node.OpType)
		}
		if klog.V(1).Enabled() {
			if err := result.Check(); err != nil {
				klog.Errorf("executing single output %s: output buffer %p: %v", node.OpType, result, err)
			}
		}

		// Write result to execBuf (single output node), or free it if not needed.
		execBuf.results[nodeIdx] = result
		execBuf.owned[nodeIdx] = true
	}

	// Update usage counts and free unused buffers.
	// In parallel mode, this is safe without locks:
	//   - numUsed is atomic; only the goroutine whose Add() returns NumUses will free the buffer.
	//   - results[inputIdx] is set to nil only by the last consumer, after all readers have
	//     already copied the pointer into their local inputBuffers.
	for i, input := range node.Inputs {
		inputIdx := input.Index
		if inputBuffers[i] == nil {
			// Input buffer is nil, means it has been consumed by the operation.
			// Mark this input as used and indicate the result is no longer available.
			execBuf.numUsed[inputIdx].Add(1)
			execBuf.results[inputIdx] = nil
			continue
		}
		if !inputBuffers[i].InUse {
			// Sanity check: must be done BEFORE numUsed.Add below. After the Add,
			// another goroutine could become the last consumer and PutBuffer (setting
			// InUse=false), causing a data race on this field.
			return errors.Errorf("input #%d for node %s has been released, but not marked as consumed!?",
				i, node.OpType)
		}
		newCount := execBuf.numUsed[inputIdx].Add(1) // Mark this input as used.
		if int(newCount) == fe.NumUses[inputIdx] && execBuf.owned[inputIdx] {
			// Check if it is reused as one of the outputs -- common for in-place operations, like in exec_binary.go.
			// The contract is that if the input is reused, the operator must set the input buffer to nil in the input slice.
			// If we find the input buffer reused as an output but it is not nil here, it is a bug in the operator implementation.
			if node.IsMultiOutputs() {
				for outIdx, outputNode := range node.MultiOutputsNodes {
					if execBuf.results[outputNode.Index] == inputBuffers[i] {
						return errors.Errorf("op %s (output %d) reused input %d as output but didn't set input to nil in buffer slice", node.OpType, outIdx, i)
					}
				}
			} else if execBuf.results[nodeIdx] == inputBuffers[i] {
				return errors.Errorf("op %s reused input %d as output but didn't set input to nil in buffer slice",
					node.OpType, i)
			}

			// Release the input buffer - all users have finished.
			backend.PutBuffer(inputBuffers[i])
			execBuf.results[inputIdx] = nil
		}
	}
	// Also update usage counts for captured inputs.
	// These are treated as additional inputs for lifetime tracking.
	for _, closureCaptures := range node.CapturedInputs {
		for _, capturedInput := range closureCaptures {
			capturedIdx := capturedInput.Index
			newCount := execBuf.numUsed[capturedIdx].Add(1)
			capturedBuf := execBuf.results[capturedIdx]
			if capturedBuf == nil {
				continue
			}
			if int(newCount) == fe.NumUses[capturedIdx] && execBuf.owned[capturedIdx] {
				// Release the captured buffer - all uses have finished.
				backend.PutBuffer(capturedBuf)
				execBuf.results[capturedIdx] = nil
			}
		}
	}
	if execBuf.opsExecutionType == OpsExecutionSequential {
		// For sequential execution, we store the input buffers and ownership slices
		// to save an allocation for these slices the next time it is executed.
		execBuf.opInputBuffers = inputBuffers
		execBuf.opInputsOwned = inputsOwned
	}
	return nil
}

// EstimatedTemporaryMemory estimates the memory used for the function if executed sequentially.
// Returns int64(shapes.DynamicDim) (-1) if any parameter or node in the function has dynamic dimensions,
// since dynamic models cannot have their memory statically estimated at compile time.
//
// It assumes temporary memory is used for every node output (using Shape.ByteSize()),
// and released everytime one node output is no longer needed (its output has been used
// fe.NumUses[nodeIdx] times). It does not actually execute the nodes, just simulates the
// execution to report back the maximum temporary memory live at any time.
func (fe *FunctionExecutable) EstimatedTemporaryMemory() int64 {
	for _, p := range fe.Function.Parameters {
		if p.Shape.IsDynamic() {
			return int64(shapes.DynamicDim)
		}
	}

	var currentMemory int64
	var maxMemory int64
	numUsed := make([]int, fe.NumNodesToProcess)
	computed := make([]bool, fe.NumNodesToProcess)

	// Pre-mark parameters and captured values as computed.
	for _, node := range fe.Function.Parameters {
		if node.Index < fe.NumNodesToProcess {
			computed[node.Index] = true
		}
	}
	for _, node := range fe.Function.CapturedLocalNodes {
		if node.Index < fe.NumNodesToProcess {
			computed[node.Index] = true
		}
	}

	for _, nodeIdx := range fe.Schedule {
		if computed[nodeIdx] {
			continue
		}
		if fe.NumUses[nodeIdx] == 0 {
			continue
		}

		node := fe.Function.Nodes[nodeIdx]

		if node.OpType == compute.OpTypeParameter ||
			node.OpType == compute.OpTypeConstant ||
			node.OpType == compute.OpTypeCapturedValue {
			computed[nodeIdx] = true
			continue
		}

		// Allocate memory for this node's output(s)
		if node.IsMultiOutputs() {
			for _, outputNode := range node.MultiOutputsNodes {
				if outputNode.Index >= fe.NumNodesToProcess || fe.NumUses[outputNode.Index] == 0 {
					continue
				}
				computed[outputNode.Index] = true
				if outputNode.Shape.IsDynamic() {
					return int64(shapes.DynamicDim)
				}
				currentMemory += outputNode.Shape.ByteSize()
			}
		} else {
			computed[nodeIdx] = true
			if node.Shape.IsDynamic() {
				return int64(shapes.DynamicDim)
			}
			currentMemory += node.Shape.ByteSize()
		}

		if currentMemory > maxMemory {
			maxMemory = currentMemory
		}

		// Simulate the release of inputs
		for _, input := range node.Inputs {
			inputIdx := input.Index
			numUsed[inputIdx]++
			if numUsed[inputIdx] == fe.NumUses[inputIdx] {
				inputNode := fe.Function.Nodes[inputIdx]
				if inputNode.OpType != compute.OpTypeParameter &&
					inputNode.OpType != compute.OpTypeConstant &&
					inputNode.OpType != compute.OpTypeCapturedValue &&
					!inputNode.IsMultiOutputs() {
					currentMemory -= inputNode.Shape.ByteSize()
				}
			}
		}

		// Simulate the release of captured inputs
		for _, closureCaptures := range node.CapturedInputs {
			for _, capturedInput := range closureCaptures {
				capturedIdx := capturedInput.Index
				numUsed[capturedIdx]++
				if numUsed[capturedIdx] == fe.NumUses[capturedIdx] {
					capturedNode := fe.Function.Nodes[capturedIdx]
					if capturedNode.OpType != compute.OpTypeParameter &&
						capturedNode.OpType != compute.OpTypeConstant &&
						capturedNode.OpType != compute.OpTypeCapturedValue &&
						!capturedNode.IsMultiOutputs() {
						currentMemory -= capturedNode.Shape.ByteSize()
					}
				}
			}
		}
	}

	return maxMemory
}

# Parallelism & Scheduler in the Go Backend (`scheduler.Scheduler`)

This document describes the architecture and design plan for parallelism in the Go backend (`internal/gobackend`) and its runtime execution scheduler (`internal/gobackend/scheduler`).

---

## 1. Concepts & Terminology

Parallelism in the Go compute backend is organized around two work units (**Chunks** and **Tasks**) and three hierarchical levels of execution (**Inter-Call**, **Inter-Op**, and **Intra-Op**).

### Work Units: Chunks and Tasks

- **Chunk**: A contiguous piece of computation and memory within an operation (for example, a range of batch items, outer rows in `Softmax` or `LayerNorm`, or a matrix panel/tile in `MatMul`).
  - Chunks are sized and partitioned so that different workers operate on **well-separated memory regions**, preventing false sharing (cache-line bouncing) and maximizing CPU cache locality.
- **Task** (`func()`): A unit of execution submitted to `scheduler.Scheduler` to be run by a worker goroutine.
  - Depending on the operation and available parallelism, an op implementation may map one chunk per task, multiple chunks per task, or execute all chunks inline without creating tasks at all.

### The Three Levels of Parallelism

1. **Inter-Call Parallelism** ( across concurrent `Executable.Execute()` calls ):
   - Occurs when the user's application invokes `Executable.Execute()` concurrently from multiple goroutines (e.g., serving multiple inference requests in parallel or running multi-stream throughput benchmarks).
   - This is the **coarsest and most efficient** level of parallelism: each call operates on completely independent buffers with zero synchronization barriers between calls.
2. **Inter-Op Parallelism** ( across independent graph nodes within a single `Execute()` call ):
   - Occurs when a computation graph branches into multiple independent operations whose inputs are simultaneously ready.
   - Controlled by the backend's execution mode (`ops_sequential` vs. `ops_parallel`).
   - While inter-op parallelism can exploit wide, branching graphs, in most modern ML architectures (such as Transformers / BERT / LLMs) graphs are predominantly sequential chains. Furthermore, running multiple large ops concurrently can increase peak memory usage and cause L2/L3 cache thrashing (which is why runtimes like ONNX Runtime default to `ExecutionModeSequential`).
3. **Intra-Op Parallelism** ( across chunks within a single operation ):
   - Occurs when a single heavy operation (`MatMul`, `Softmax`, `LayerNorm`, `SDPA`, `ConvGeneral`, element-wise ops on large tensors) splits its work into chunks and fans them out across available workers in `scheduler.Scheduler`.
   - This is the primary driver of multi-core speedup for single-stream inference and sequential graphs.

---

## 2. Background & Motivation for Redesign

Historically, `scheduler.Scheduler` (formerly `workerspool.Pool`) and `FunctionExecutable` suffered from four main limitations:

1. **Goroutine Spawn Overhead**:
   - `Scheduler.lockedRunTaskInGoroutine()` spawned a new `go func()` for every task. For fine-grained intra-op chunks executed hundreds of times per forward pass, goroutine creation and scheduling overhead accumulated.
2. **Parked Executor Starvation (Issue / PR #44)**:
   - To avoid spawning goroutines per node in `executeParallel`, `FunctionExecutable` pre-started up to `DefaultNumExecutors - 1` (7) persistent executors from the shared `backend.Scheduler`.
   - On sequential graphs (such as Transformer layers), 6 of those 7 executors sat blocked on `<-readyQueue` while still counting toward `numRunning` in the scheduler. When a large `MatMul` or `Softmax` tried to fan out intra-op tasks via `Saturate` or `WaitToStart`, the scheduler appeared full, starving execution down to a single core.
3. **Manual Tuning Required for Multi-Call Benchmarks**:
   - Previously, running concurrent `Execute()` calls required manually disabling intra-op and inter-op parallelism to avoid over-subscribing CPU cores and paying unnecessary intra-op synchronization costs.
4. **Oversubscribed Default `maxParallelism` (`2 * GOMAXPROCS`)**:
   - Previously, `scheduler.New()` defaulted `maxParallelism` to `2 * runtime.GOMAXPROCS(0)` to compensate for goroutines that blocked waiting on channels or barriers while holding a slot.
   - For pure CPU/SIMD compute workers that do not block, running $2 \times \text{GOMAXPROCS}$ active goroutines oversubscribes the CPU cores, causing Go scheduler preemptions mid-computation, L1/L2 cache eviction, and straggler delays at `WaitGroup` barriers.

---

## 3. Core Design & Invariants

### Default Parallelism (`runtime.GOMAXPROCS(0)`)

- Because workers in the new design never block passively while holding a compute slot, the default `maxParallelism` is set to **`runtime.GOMAXPROCS(0)`** (1:1 with available logical CPU cores) rather than `2 * runtime.GOMAXPROCS(0)`.
- Matching active compute workers 1:1 with `GOMAXPROCS` prevents mid-chunk preemption by the Go scheduler and keeps worker OS threads naturally resident on their CPU cores across consecutive operations, preserving L1/L2 cache warmth.

### Invariant 1: Priority Hierarchy (`Inter-Call` > `Intra-Op` > `Inter-Op`)

When allocating CPU workers (`maxParallelism`, typically `GOMAXPROCS`), `backend.Scheduler` prioritizes work in the following order:

1. **Priority 1 — Inter-Call**: Concurrent calls to `Executable.Execute()` get first claim on parallelism.
   - If $C$ calls are executing concurrently on a scheduler of size $P$, available parallelism is divided among the active calls:
     $$\text{effectiveParallelismPerCall} = \max\left(1, \left\lfloor \frac{P}{C} \right\rfloor\right)$$
   - When $C \ge P$ (e.g., 8 concurrent calls on an 8-core CPU), $\text{effectiveParallelismPerCall} = 1$. Every call automatically executes its graph and ops sequentially on its own goroutine with **zero intra-op or inter-op task creation or synchronization overhead**—no manual configuration required.
   - When $1 < C < P$ (e.g., 2 concurrent calls on 8 cores), each call runs concurrently and shares the remaining cores ($\approx 4$ workers per call) for intra-op parallelism.
   - When $C = 1$ (single-stream inference), all $P$ workers are available for intra-op parallelism.
2. **Priority 2 — Intra-Op**: Within a call, finishing the currently executing operation as quickly as possible takes priority over starting other independent graph nodes.
   - Cooperating on a single op keeps its weights and tiles hot in shared L2/L3 caches and allows input buffers to be freed earlier.
3. **Priority 3 — Inter-Op**: Only when `ops_parallel` is enabled *and* idle workers remain that are not needed by active intra-op tasks will workers pick up ready branch nodes from the inter-op queue.
   - By default, graph execution uses **`OpsExecutionSequential`** (like ONNX Runtime's `ExecutionModeSequential`), avoiding atomic dependency tracking and channel overhead for the common sequential-model case. Users can opt into inter-op parallelism via `ops_parallel` (or a priority option) for heavily branching graphs with small ops.

### Invariant 2: Persistent Worker Pool with Priority Queues

- `scheduler.Scheduler` maintains a fixed pool of persistent worker goroutines created once and kept alive, eliminating per-task `go func()` creation overhead and removing the need for `WorkerIsAsleep` / `WorkerRestarted`.
- Persistent workers pull tasks from prioritized queues:
  - **Intra-Op Queue (`intraOpQueue`)**: Checked first by idle workers (or per configured priority mode).
  - **Inter-Op Queue (`interOpQueue`)**: Checked when no higher-priority intra-op tasks are waiting.

### Invariant 3: Strict Two-Level Task Nesting

- An **Inter-Op Task** (or the main `Execute()` caller goroutine running a node) may either:
  1. Execute its chunks directly inline (when tensor size is small or $\text{effectiveParallelismPerCall} == 1$), or
  2. Spawn **Intra-Op Tasks** required for its completion and wait for them via `Scheduler.WaitIntraOpTasks` / `Scheduler.RunIntraOpTasks`.
- An **Intra-Op Task** is strictly a leaf computation: it executes its assigned chunk(s) and returns immediately. It must never spawn nested tasks or block waiting on other tasks.

### Invariant 4: Deadlock-Free Work-Helping (`WaitIntraOpTasks`)

- When an op spawns intra-op tasks and waits for their completion, **the calling goroutine never blocks passively while unstarted chunks of its op remain**.
- Instead, calling `WaitIntraOpTasks` (or `RunIntraOpTasks`) causes the calling goroutine to immediately execute **its own op's intra-op tasks** at highest priority until no unstarted tasks for that op remain.
- **Guaranteed Forward Progress (Zero Deadlock)**:
  - Once an op starts executing, it always has at least one worker (the caller itself) actively draining its chunks, even if all persistent workers in `Scheduler` are busy with other calls or ops.
  - Only after all of the op's chunks have been started does the caller wait on the op's `sync.WaitGroup` for any helper workers finishing the last in-flight chunk(s). Because intra-op tasks are non-blocking leaf tasks, helper completion is guaranteed in bounded CPU time.

---

## 4. Memory Locality & Chunk Partitioning

To maximize memory bandwidth and cache efficiency when splitting an operation into chunks:

1. **Disjoint Contiguous Memory Regions**:
   - Different workers must write to well-separated, contiguous slices of the output tensor (e.g., partitioning the outer/batch dimension `[0, outerSize)` into contiguous ranges `[start, end)` per worker) rather than interleaved indices.
   - This prevents false sharing (multiple cores writing to the same 64-byte cache line) and allows each core's hardware prefetcher to stream linearly.
2. **Adaptive Granularity**:
   - Ops query `backend.Scheduler.EffectiveParallelism()` (or `AdjustedMaxParallelism()`, accounting for `NumLiveExecutions` / active calls) to determine how many chunks to create.
   - If the tensor is smaller than `minParallelizeChunk` or effective parallelism is 1, the op executes inline without allocating closures or touching the scheduler.
   - When parallelizing across $W$ target workers, work is split into a small multiple of $W$ contiguous chunks (e.g., $W$ or $2W$ chunks, or per-worker contiguous sub-ranges) to avoid creating giant queues of tiny tasks.

---

## 5. On-Demand Inter-Op Execution (`executeParallel`)

When `ops_parallel` is enabled (`FunctionExecutable.executeParallel`):

1. **No Parked Executors**:
   - `executeParallel` does **not** pre-spawn background executors that block on an empty `readyQueue`.
2. **Local Continuation on Linear Chains**:
   - When a node finishes and decrements `remainingDeps` of its consumers, if **one** consumer becomes ready (`remainingDeps == 0`), the current goroutine executes it immediately in its loop (`localContinuation`), avoiding any queue round-trip and preserving L1/L2 cache locality.
3. **On-Demand Branch Tasks**:
   - Only when a node unlocks **more than one** ready consumer simultaneously (a true graph branch) are the extra ready nodes submitted as **inter-op tasks** to `backend.Scheduler`.
   - When an inter-op task finishes a node and unlocks **zero** ready consumers (its branch is waiting for another branch to finish), the task simply **returns**, freeing the worker immediately.

---

## 6. Proposed `scheduler.Scheduler` API

```go
// EnterExecution registers the start of a top-level Executable.Execute() call (inter-call tracking).
// It increments the active call count (and optionally gates entry if limiting concurrent executions),
// returning a function to be deferred when Execute() finishes.
func (s *Scheduler) EnterExecution() (exitExecution func())

// EffectiveParallelism returns the target number of workers an op within the current Execute() call
// should use, dynamically scaled by the number of concurrent active executions:
//   max(1, AdjustedMaxParallelism() / numActiveCalls)
func (s *Scheduler) EffectiveParallelism() int

// RunIntraOpTasks executes the given intra-op tasks for the current operation and waits for all
// of them to finish.
//
// Invariant: The calling goroutine immediately participates in executing these tasks itself,
// while available persistent workers in the scheduler assist concurrently.
func (s *Scheduler) RunIntraOpTasks(tasks []func())

// ParallelizeChunks partitions [0, totalItems) into contiguous, cache-friendly ranges based on
// EffectiveParallelism() and minChunkSize, and executes fn(start, end) across the calling goroutine
// and available intra-op workers. If EffectiveParallelism() == 1 or totalItems < minChunkSize,
// it calls fn(0, totalItems) directly inline with zero allocation.
func (s *Scheduler) ParallelizeChunks(totalItems, minChunkSize int, fn func(start, end int))

// Saturate fans out workerFn across the calling goroutine and up to EffectiveParallelism()-1
// available scheduler workers (used by work-pulling ops like MatMul).
//
// Each workerFn pulls work from a shared queue until no work remains and then returns.
// Once any workerFn returns (indicating the work queue is exhausted), Saturate stops
// spawning/recruiting additional workers and waits for all already-running workerFn
// instances to finish before returning.
func (s *Scheduler) Saturate(workerFn func())

// SaturateUpTo is like Saturate, but caps the total number of participating workers
// (including the calling goroutine) to maxWorkers when maxWorkers > 0.
func (s *Scheduler) SaturateUpTo(maxWorkers int, workerFn func())

// SubmitInterOp enqueues an inter-op task (a ready graph branch when ops_parallel is enabled).
func (s *Scheduler) SubmitInterOp(task func())
```

---

## 7. Implementation Plan

### Step 1: Refactor `scheduler.Scheduler` (`internal/gobackend/scheduler`)
- Change the default `maxParallelism` in `scheduler.New()` from `2 * runtime.GOMAXPROCS(0)` to `runtime.GOMAXPROCS(0)`.
- Replace per-task `go func()` creation with persistent worker goroutines.
- Add `EnterExecution()` / `EffectiveParallelism()` tracking so concurrent `Execute()` calls automatically scale down intra-op fan-out.
- Implement `intraOpQueue` and `interOpQueue` with priority selection (`Intra-Op` > `Inter-Op` by default, configurable if desired).
- Implement `RunIntraOpTasks`, `ParallelizeChunks`, and `Saturate` using the work-helping pattern where the caller executes its own op's tasks before waiting on in-flight helpers.
- Remove `WorkerIsAsleep`, `WorkerRestarted`, and `extraParallelism`.
- Add comprehensive tests in `internal/gobackend/scheduler`:
  - Single-call intra-op saturation across all cores.
  - Multi-call (`Inter-Call`) scaling verifying automatic reduction of intra-op tasks when `numActiveCalls >= maxParallelism`.
  - Priority ordering between `Intra-Op` and `Inter-Op` tasks.
  - Deadlock freedom under nested `SubmitInterOp` + `RunIntraOpTasks` with `maxParallelism = 1, 2, 8`.

### Step 2: Update Graph Execution (`internal/gobackend/executable.go` & `gobackend.go`)
- Use `backend.Scheduler.EnterExecution()` in `FunctionExecutable.Execute()`.
- Default `OpsExecutionType` to `OpsExecutionSequential` (matching ORT's `ExecutionModeSequential`), while keeping `ops_parallel` available for branching graphs.
- Refactor `executeParallel` to use on-demand `SubmitInterOp` tasks on graph branches + `localContinuation` on linear chains, removing the pre-spawned `numExecutors` loop that blocked on `readyQueue`.

### Step 3: Migrate Op Call Sites to Cache-Friendly Chunking
- Update `WaitToStart` loops in `ops/`, `fusedops/`, and `activations/` to use `backend.Scheduler.ParallelizeChunks` (or `RunIntraOpTasks` with contiguous per-worker ranges).
- Update `dot/matmul` to size its workers via `backend.Scheduler.EffectiveParallelism()` and run via the updated `Saturate`.

### Step 4: Verify & Benchmark
- Run compliance tests (`go test ./...`) and benchmarks (`go test -bench=. ./gobackend`).
- Benchmark single-stream latency (e.g., `all-MiniLM-L6-v2` from PR #44) and multi-stream throughput (concurrent `Execute()` calls) without manual flag changes.

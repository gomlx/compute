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
$$
\text{effectiveParallelismPerCall} = \max\left(1, \left\lfloor \frac{P}{C} \right\rfloor\right)
$$
   - When $C \ge P$ (e.g., 8 concurrent calls on an 8-core CPU), $\text{effectiveParallelismPerCall} = 1$. Every call automatically executes its graph and ops sequentially on its own goroutine with **zero intra-op or inter-op task creation or synchronization overhead**—no manual configuration required.
   - When $1 < C < P$ (e.g., 2 concurrent calls on 8 cores), each call runs concurrently and shares the remaining cores ($\approx 4$ workers per call) for intra-op parallelism.
   - When $C = 1$ (single-stream inference), all $P$ workers are available for intra-op parallelism.
2. **Priority 2 — Intra-Op**: Within a call, finishing the currently executing operation as quickly as possible takes priority over starting other independent graph nodes.
   - Cooperating on a single op keeps its weights and tiles hot in shared L2/L3 caches and allows input buffers to be freed earlier.
3. **Priority 3 — Inter-Op**: When `ops_parallel` or `ops_dynamic` (the default when `NumLiveExecutions <= 1`) is active, ready graph branches are submitted to `interOpQueue` only when idle workers remain and no intra-op tasks are waiting (`len(s.intraOpQueue) == 0`).
   - By default, the Go backend uses **`OpsExecutionDynamic`**:
     - When a single execution is running (`NumLiveExecutions <= 1`), it uses `executeParallel` so independent graph branches (e.g., multi-scale attention branches, Q/K/V projections, RoPE sin/cos branches, gated MLP branches) can overlap single-threaded or partially-parallel ops across idle cores.
     - When multiple `Execute()` calls run concurrently (`NumLiveExecutions > 1`), it automatically switches to `executeSequential`, avoiding inter-op task creation and cross-call contention.
   - Crucially, `SubmitInterOp` refuses to launch a new inter-op branch worker while `len(s.intraOpQueue) > 0`. Allowing inter-op branches to launch while an intra-op fan-out is in progress causes multiple heavy ops (such as `MatMul`) to run simultaneously and contend on shared L2/L3 caches, increasing total CPU time.

### Invariant 2: Persistent Worker Pool with Priority Queues

- `scheduler.Scheduler` maintains a fixed pool of persistent worker goroutines created once and kept alive, eliminating per-task `go func()` creation overhead and removing the need for `WorkerIsAsleep` / `WorkerRestarted`.
- Persistent workers pull tasks from prioritized queues:
  - **Intra-Op Queue (`intraOpQueue`)**: Checked first by idle workers (or per configured priority mode).
  - **Inter-Op Queue (`interOpQueue`)**: Checked when no higher-priority intra-op tasks are waiting.

### Invariant 3: Strict Two-Level Task Nesting

- An **Inter-Op Task** (or the main `Execute()` caller goroutine running a node) may either:
  1. Execute its chunks directly inline (when tensor size is small or $\text{effectiveParallelismPerCall} == 1$), or
  2. Spawn **Intra-Op Tasks** required for its completion and wait for them via `Scheduler.RunIntraOpTasks`, `Scheduler.ParallelizeChunks`, or `Scheduler.SaturateUpTo`.
- An **Intra-Op Task** is strictly a leaf computation: it executes its assigned chunk(s) and returns immediately. It must never spawn nested tasks or block waiting on other tasks.

### Invariant 4: Deadlock-Free Work-Helping & Scoped Helper Cancellation

- When an op spawns intra-op tasks and waits for their completion, **the calling goroutine never blocks passively while unstarted chunks of its op remain**.
- Instead, calling `RunIntraOpTasks`, `ParallelizeChunks`, or `SaturateUpTo` causes the calling goroutine to immediately pull and execute **its own op's chunks** from a shared atomic index (`nextIdx.Add(1)`) until no unstarted chunks for that op remain.
- **Scoped Atomic Cancellation (No Cross-Op Queue Draining)**:
  - Once an op's work is exhausted, any helper closures that were enqueued in `s.intraOpQueue` for this op but have not yet been popped by a persistent worker are cancelled in-place via an op-local atomic counter (`unstarted.Swap(0)`). When a persistent worker later dequeues a cancelled helper, it immediately decrements the op's `WaitGroup` and returns without executing `workerFn`.
  - **Why not drain `s.intraOpQueue` in the caller?** When `OpsExecutionDynamic` / `OpsExecutionParallel` is active, two independent graph branches (Branch A and Branch B) may execute parallel ops concurrently. If Branch A's caller drained the shared `s.intraOpQueue` while waiting for its own last helper to finish, it could pop a helper belonging to Branch B—dragging Branch A's thread into Branch B's computation and delaying Branch A's downstream `localContinuation`. Using op-scoped atomic `unstarted` cancellation avoids touching other ops' tasks while still freeing the caller as soon as its own op completes.
- **Guaranteed Forward Progress (Zero Deadlock)**:
  - Once an op starts executing, it always has at least one worker (the caller itself) actively draining its chunks, even if all persistent workers in `Scheduler` are busy with other calls or ops.
  - Only after all of the op's chunks have been started (and any unstarted helpers atomically cancelled) does the caller wait on the op's `sync.WaitGroup` for in-flight helpers to finish their current leaf chunk.

---

## 4. Memory Locality, Chunk Partitioning & Wave Quantization

To maximize memory bandwidth, cache efficiency, and multi-core utilization when splitting an operation into chunks:

1. **Disjoint Contiguous Memory Regions**:
   - Different workers must write to well-separated, contiguous slices of the output tensor (e.g., partitioning the outer/batch dimension `[0, outerSize)` into contiguous ranges `[start, end)` per worker, or iterating output tiles in contiguous output order) rather than interleaved indices.
   - This prevents false sharing (multiple cores writing to the same 64-byte cache line) and allows each core's hardware prefetcher to stream linearly.
2. **Over-Partitioning by $2 \times W$ to Prevent Wave Quantization**:
   - Splitting work into only $1 \times W$ chunks (where $W = \text{EffectiveParallelism()}$) causes severe **wave quantization** and **core starvation**:
     - *Ceiling-Division Starvation*: If `totalItems = 40` on a 32-core CPU and uniform chunk size `ceil(40 / 32) = 2` is used, only `40 / 2 = 20` chunks are created—leaving **12 of 32 cores completely idle**!
     - *Batched Tail-Wave Stalls*: In batched `MatMul` with `batchSize = 20` on 32 cores, targeting $1 \times W$ (`ceil(32 / 20) = 2` splits per batch) creates `40` work items: Wave 1 runs 32 items on 32 cores, and Wave 2 runs the remaining **8 items on 8 cores while 24 cores sit idle**.
     - *Inter-Op Interference*: When `OpsExecutionDynamic` runs a lightweight branch on 1–2 cores concurrently with a parallel op, having only $W$ uniform chunks forces the remaining chunks into a second full-latency wave.
   - Therefore, both `ParallelizeChunks` and `MatMul.AppendWorkItems` target **$2 \times W$ chunks** (`targetChunks := min(2*numWorkers, totalItems / minChunkSize)`), while recruiting at most $W - 1$ helper workers from the pool:
     - Workers dynamically pull from the $2W$ chunks via an atomic index (`nextChunk.Add(1)`), smoothing out uneven chunk execution times and tail waves with zero extra goroutines or scratch buffer allocations.
3. **Proportional Ceiling-First Boundaries (No Round-Up Starvation)**:
   - Instead of computing a single rounded-up `chunkSize` and stepping by `chunkSize`, chunk boundaries for chunk $c \in [0, K)$ are computed proportionally:
$$
\text{start}(c) = \left\lfloor \frac{c \cdot N + K - 1}{K} \right\rfloor, \quad \text{end}(c) = \left\lfloor \frac{(c + 1) \cdot N + K - 1}{K} \right\rfloor
$$
   - Every chunk differs in size by at most 1 item (or 1 register block `mr`/`nr` in `MatMul`), and the exact target number of chunks $K$ is always produced.
4. **MatMul 1D vs. 2D Split Selection (`Choose2DSplit`)**:
   - In `MatMul`, workers allocate per-worker LHS panel caches (`lhsAllocPanels = 64` panels, holding up to 128 rows in L2 cache).
   - `Choose2DSplit` accounts for this per-worker LHS cache capacity (`maxCachedLHSRows`) and prefers 1D row splits (`candN == 1`) whenever `rowChunk >= LHSPanelCrossSize`. Splitting only along $M$ (`candN == 1`) ensures each worker packs its assigned LHS rows once and writes to completely disjoint output rows, avoiding redundant LHS re-packing across column workers.

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

// SubmitInterOp enqueues an inter-op task (a ready graph branch when ops_parallel/ops_dynamic is enabled)
// if an idle worker is available and no intra-op tasks are waiting. Returns false if the caller
// should execute the task later via its local ready queue.
func (s *Scheduler) SubmitInterOp(task func()) bool
```

---

## 7. Implementation Summary

### Step 1: `scheduler.Scheduler` (`internal/gobackend/scheduler`)
- Default `maxParallelism` in `scheduler.New()` is `runtime.GOMAXPROCS(0)`.
- Persistent worker goroutines pull from `intraOpQueue` and `interOpQueue` with priority selection (`Intra-Op` > `Inter-Op` by default).
- `EnterExecution()` / `EffectiveParallelism()` tracking automatically scales down intra-op fan-out when multiple `Execute()` calls run concurrently.
- `RunIntraOpTasks`, `ParallelizeChunks`, and `SaturateUpTo` use work-helping with op-scoped atomic `unstarted` helper cancellation and $2 \times W$ proportional chunking.

### Step 2: Graph Execution (`internal/gobackend/executable.go` & `gobackend.go`)
- Uses `backend.Scheduler.EnterExecution()` in `FunctionExecutable.Execute()`.
- Defaults `OpsExecutionType` to `OpsExecutionDynamic` (`ops_dynamic`): uses on-demand inter-op parallelism (`executeParallel`) when a single execution is active (`NumLiveExecutions <= 1`), and automatically switches to `executeSequential` under concurrent `Execute()` calls (`NumLiveExecutions > 1`).
- `executeParallel` uses on-demand `SubmitInterOp` tasks on graph branches (gated on `len(s.intraOpQueue) == 0`) + `localContinuation` on linear chains, with no parked background executors.

### Step 3: Migrate Op Call Sites to Cache-Friendly Chunking
- Update `WaitToStart` loops in `ops/`, `fusedops/`, and `activations/` to use `backend.Scheduler.ParallelizeChunks` (or `RunIntraOpTasks` with contiguous per-worker ranges).
- Update `dot/matmul` to size its workers via `backend.Scheduler.EffectiveParallelism()` and run via the updated `Saturate`.

### Step 4: Verify & Benchmark
- Run compliance tests (`go test ./...`) and benchmarks (`go test -bench=. ./gobackend`).
- Benchmark single-stream latency (e.g., `all-MiniLM-L6-v2` from PR #44) and multi-stream throughput (concurrent `Execute()` calls) without manual flag changes.

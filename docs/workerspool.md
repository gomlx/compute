# Design Plan Proposal: Two-Level Worker Pool (`workerspool.Pool`)

## Background & Problem Statement

In the Go backend (`internal/gobackend`), parallelism operates at two levels:
1. **Inter-op parallelism** (`FunctionExecutable.executeParallel`): Executing independent nodes in a computation graph concurrently when the graph branches (or when multiple `Execute` calls run concurrently).
2. **Intra-op parallelism** (`MatMul`, `Softmax`, `LayerNorm`, `SDPA`, `ConvGeneral`, etc.): Splitting a single large operation into smaller chunks of work executed across multiple CPU cores.

Historically, `workerspool.Pool` faced two conflicting issues:
- **Goroutine spawn overhead**: Spawning a new goroutine (`go func()`) on every `lockedRunTaskInGoroutine` call adds scheduling and stack-management overhead for small chunks of work.
- **Worker slot starvation (Issue / PR #44)**: To avoid repeatedly spawning graph-level goroutines, `executeParallel` pre-started up to `DefaultNumExecutors - 1` (7) persistent executors from the shared `workerspool.Pool`. On sequential graphs (such as Transformer layers), 6 of those 7 executors blocked on `<-readyQueue` while holding worker pool slots (`numRunning`), starving intra-op operations (`Saturate` and `WaitToStart`) down to a single core.
- **Synchronization vs. Granularity**: When CPU cores are limited, coarse-grained (inter-op) parallelism is generally preferable to fine-grained (intra-op) parallelism because it avoids intra-op barrier synchronization overheads. However, when the graph has no active branching (the common case in deep sequential models), all cores should seamlessly help with intra-op parallelism.

---

## Core Design Invariants

Let a **chunk** be a contiguous unit of work within an operation, and a **`Task`** (`func()`) be a callable unit submitted to `workerspool.Pool`. Ops may map one chunk per task or multiple chunks per task.

### 1. Persistent Worker Goroutines & Two Priority Queues
- `workerspool.Pool` maintains a fixed pool of persistent worker goroutines (sized to `AdjustedMaxParallelism()`, typically `GOMAXPROCS`) that are created once and live for the lifetime of the pool.
- Goroutines are not created or destroyed per task, eliminating `go func()` spawn overhead and the need for `WorkerIsAsleep` / `WorkerRestarted` slot-lending bookkeeping.
- Tasks are organized into **two priority levels**:
  1. **Inter-op queue (`interOpQueue`)**: Higher priority. Holds graph-level node execution tasks (when a computation graph branches into multiple ready nodes).
  2. **Intra-op queue (`intraOpQueue`)**: Lower priority. Holds subtasks spawned by an executing op to parallelize its internal chunks.

### 2. Two-Level Hierarchy (Inter-Op vs. Intra-Op)
- An **inter-op task** executes a graph node (or a chain of single-dependent graph nodes via local continuation). During its execution, an inter-op task can either:
  - Execute its chunks of work directly on the current goroutine, or
  - Create **intra-op tasks** required for its completion and wait for them using `Pool.WaitIntraOpTasks` (or a high-level helper built on it).
- An **intra-op task** is strictly a leaf computation: it executes its assigned chunk(s) and returns. It must never block or wait on other tasks.

### 3. Priority Scheduling Rule
- Idle workers in `workerspool.Pool` **always prioritize `interOpQueue` over `intraOpQueue`**:
  - A pool worker only pulls from `intraOpQueue` when `interOpQueue` is empty.
  - Consequently, if a graph has independent branches ready to run, available cores will first spread out across those coarser inter-op branches.
  - When a graph is sequential (or once parallel branches merge), `interOpQueue` is empty, and all pool workers immediately assist with `intraOpQueue` subtasks.

### 4. Deadlock-Free Work-Helping (`WaitIntraOpTasks`)
- When an **inter-op task** spawns intra-op subtasks and calls `WaitIntraOpTasks` (or `RunIntraOpTasks`) to wait for their completion, **the calling inter-op goroutine does not block passively**.
- Instead, the calling inter-op goroutine immediately processes **its own op's intra-op tasks** to completion (at highest priority for that op).
- **Guaranteed Forward Progress (Zero Deadlock)**:
  - Once an inter-op task starts, it is guaranteed to have at least 1 worker (itself!) actively executing its chunks, even if all other pool workers are busy with other inter-op tasks or `maxParallelism == 1`.
  - Only after all of the op's intra-op tasks have been *started* (and the caller has finished its own share) does the caller wait on the completion barrier (`sync.WaitGroup`) for any helper workers currently finishing the last in-flight chunk(s) of that op. Because intra-op tasks are non-blocking leaf computations, those helpers are guaranteed to finish in bounded CPU time.

---

## Memory Locality & Chunk Partitioning (Avoiding False Sharing)

When splitting an operation's tensor into chunks across workers:
1. **Cache-Line & Page Locality**:
   - Different workers must write to **well-separated memory regions** (contiguous blocks of rows/batches per worker) rather than interleaved elements, preventing cache-line bouncing (false sharing) and maximizing L1/L2 hardware prefetcher efficiency.
2. **Coarse Chunking Based on Available Parallelism**:
   - Ops should inspect `numWorkers := backend.Workers.AdjustedMaxParallelism()` and avoid splitting work into tiny tasks when parallelism is low (avoiding large task queues and synchronization overhead).
   - For example, given `outerSize` rows and `numWorkers` cores:
     - Partition `[0, outerSize)` into $W = \min(\text{numWorkers}, \text{outerSize} / \text{minRowsPerWorker})$ contiguous ranges, or
     - Give each participating worker $w \in [0, W)$ its own contiguous sub-range of chunks so workers pull from disjoint memory regions first.

---

## On-Demand Graph Execution (`executeParallel`)

With the two-level `Pool`, `FunctionExecutable.executeParallel` becomes simpler and no longer pre-allocates idle background executors:

1. **Local Continuation on Linear Chains**:
   - When an inter-op task finishes executing a graph node `nodeIdx` and decrements `remainingDeps` of its outputs:
   - The **first** dependent node that becomes ready (`remainingDeps == 0`) is executed *immediately on the same goroutine* (`localContinuation`), with zero channel or queue round-trip.
2. **On-Demand Inter-Op Tasks on Graph Branches**:
   - Only when a node unlocks **more than one** ready dependent node simultaneously (or when `initialReady` has $>1$ input-free nodes) are the additional ready nodes submitted to `Pool` as **inter-op tasks**.
   - When an inter-op task finishes a node that unlocks **zero** ready dependents (e.g., its branch hasn't merged with another branch yet), that inter-op task simply **returns**, immediately freeing the pool worker to pick up another inter-op task or help with intra-op tasks.

---

## Proposed API Sketch for `workerspool.Pool`

```go
// IntraOpGroup represents a group of intra-op tasks belonging to a single inter-op operation.
type IntraOpGroup struct {
    // Internal queue/slice of tasks or chunk-range callback + sync.WaitGroup.
}

// SubmitInterOp enqueues a coarse-grained inter-op task (e.g., a newly ready graph branch).
func (p *Pool) SubmitInterOp(task func())

// RunIntraOpTasks executes a set of intra-op tasks for the current op and waits for all of them
// to complete.
//
// The calling goroutine immediately executes tasks from this group itself, while idle workers
// in the pool (that have no pending inter-op tasks) help execute the remaining tasks concurrently.
func (p *Pool) RunIntraOpTasks(tasks []func())

// ParallelizeChunks (optional ergonomic helper built on RunIntraOpTasks) partitions [0, totalItems)
// into contiguous, cache-friendly ranges sized for the pool's parallelism and executes fn(start, end)
// across the calling goroutine and available intra-op helpers.
func (p *Pool) ParallelizeChunks(totalItems, minChunkSize int, fn func(start, end int))
```

### How Existing Ops Map to This API

1. **Chunked Range Ops (`Softmax`, `LayerNorm`, `Activations`, `Transpose`, `UnaryOps`, `Where`, `SDPA`)**:
   - Currently: manually compute `numWorkers`, `outerPerChunk`, loop with `wg.Add(1)` + `Workers.WaitToStart(...)`, and call `wg.Wait()`.
   - With new API: call `backend.Workers.ParallelizeChunks(outerSize, minChunkSize, func(start, end int) { ... })` (or `RunIntraOpTasks`). Each worker receives a contiguous `[start, end)` slice of the outer dimension, ensuring disjoint memory writes and zero false sharing.
2. **Work-Queue / `Saturate` Ops (`MatMul`)**:
   - `MatMul` prepares its disjoint panel/batch work items and invokes `Saturate` (implemented via `RunIntraOpTasks`), where the calling goroutine participates alongside up to `AdjustedMaxParallelism() - 1` helper workers from the pool.

---

## Implementation Steps

1. **Step 1: Implement the Two-Level `workerspool.Pool`**
   - Replace per-task `go func()` spawning with persistent worker goroutines initialized to `AdjustedMaxParallelism()`.
   - Implement `interOpQueue` (priority 1) and `intraOpQueue` (priority 2) dispatch in the worker loop.
   - Implement `RunIntraOpTasks` / `WaitIntraOpTasks` (and `Saturate` on top of it) so the calling goroutine actively drains its own intra-op tasks before waiting on in-flight helpers.
   - Remove `WorkerIsAsleep` / `WorkerRestarted` and `extraParallelism`.
   - Add unit tests in `internal/gobackend/workerspool/workerspool_test.go` covering:
     - Single-op saturation (all workers + caller participate).
     - Inter-op priority over intra-op tasks.
     - Nested execution under `maxParallelism = 1`, `2`, `8` with zero deadlocks.
2. **Step 2: Update `FunctionExecutable.executeParallel` (`internal/gobackend/executable.go`)**
   - Remove the pre-spawned `numExecutors - 1` background loop that blocks on `readyQueue`.
   - Execute the primary ready path on the caller goroutine (via `localContinuation`) and submit extra ready nodes (when branching) as inter-op tasks to `backend.Workers`.
3. **Step 3: Migrate Intra-Op Call Sites**
   - Update `WaitToStart` / `Saturate` callers in `ops/`, `fusedops/`, `activations/`, and `dot/matmul/` to use the new intra-op execution methods with contiguous, cache-local chunk ranges.
4. **Step 4: Benchmark & Verify**
   - Run `go test ./...` and `go test -bench=. ./gobackend`.
   - Verify multi-core scaling on sequential transformer workloads (such as the `all-MiniLM-L6-v2` benchmark from PR #44) as well as branching graphs and multi-stream throughput benchmarks.

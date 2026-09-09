# Go Backend

The "go" backend implements a simple, and not very fast, but very portable backend for GoMLX.

The priority is to make something that will work everywhere and is "ergonomic" (doesn't require
installing any associated C/C++/Rust library, or special packages other than Go itself).

A second priority is having a very short dependency list: this aimed at being safe, very low
dependency library.

See `capabilities.go` file to see operations that are implemented.

## To Do's

This can be split into 2 parts: implement missing ops, and optimizations.

### Missing Ops/Functionality

There are still many missing. See file `capabilities.go`. 
But feel free to create issues if there is an Op that you need and would like to see it prioritized.

### Optimizations

The initial implementation was focused on portability and getting it to work.

But there are many relatively "low-hanging fruits" for optimization, a few obvious items:

* Pre-calculate constant sub-expressions.
* Fuse unary ops: it's much faster (for larger data blocks) to loop over the data only once and apply various functions than
  loop over the data many times, each time applying the unary function.
* Fuse binary/unary ops: perform unary functions while traversing the data for binary functions. Again to save
  memory accesses.
* Further in-operation parallelization: only DotGeneral has been parallelized so far: it is usually the one that consumes most of the time.
* Use intrinsics/SIMD on platforms that allow it. It was announced as experimental in Go 1.25.
* ~~Eliminate common sub-expressions.~~

## SIMD Thresholding

While SIMD vectorization dramatically accelerates large tensor computations, it introduces fixed overheads:
- Setting up vector registers, broadcast constants, and partial load/store masks.
- Horizontal reduction across vector lanes (e.g. shuffles, unpack chains, or intermediate stack buffer stores).
- Memory alignment and post-loop scalar tails.

For small reduction axes (particularly in `ReduceTrailing` where the inner axis $B$ is reduced across many rows $A$, or in `ReduceLeading` where $B \le 4$), simple scalar loops in CPU registers outperform SIMD.

### Architecture & Caching Mechanism

1. **Clean Fallback**:
   When a SIMD executor determines that an input axis or total tensor size is below the architecture-specific threshold, it immediately returns `(nil, gobackend.ErrFallback)` **before** allocating any output buffer or consuming inputs.
   The dispatch loop in `executable.go` falls through to the next registered executor (Generic / Scalar).

2. **Node-Level Executor Caching**:
   Each `*Node` contains a thread-safe `cachedExecutorIdx atomic.Int32` (storing the 1-based index `idx + 1` of the winning executor in `nodeExecutors[node.OpType]`).
   - On the first execution, the fallback chain runs to discover the winning executor (SIMD or Scalar).
   - Once an executor succeeds without `ErrFallback`, its 1-based index is stored in `node.cachedExecutorIdx`.
   - Subsequent executions read `nodeExecutors[node.OpType][cachedIdx-1]` directly with zero heap allocations, zero struct wrappers, and **0 ns dispatch overhead**.
   - For **dynamic shapes**, GoMLX uses `ShapeSpecialization` (cached per concrete dimension tuple). Each specialization has its own concrete `resolvedNodes []*Node`, so the optimal executor is cached automatically per specialized shape configuration.

### Finding & Updating Thresholds

The thresholds are discovered empirically using a dedicated benchmark suite that measures the **median** execution duration across iterations using `testutil.DurationSampler` (16K reservoir sampling):

```bash
GOEXPERIMENT=simd go test -v -run TestFindReduceThresholds ./internal/gobackend/ops
```

This prints a Markdown comparison table showing `Scalar Median`, `SIMD Median`, and the `Ratio (SIMD/Scalar)` across data types and dimensions for:
- `ReduceTrailing`: shape $[A, B] \rightarrow [A]$ (reducing inner dimension $B$).
- `ReduceLeading`: shape $[A, B] \rightarrow [B]$ (reducing outer dimension $A$, vectorized across $B$).
- `ReduceAll`: shape $[N] \rightarrow [1]$ (reducing entire tensor).

#### Re-running on AVX-512 Hardware

To calibrate thresholds for AVX-512:
1. Run the benchmark on an AVX-512 machine:
   ```bash
   GOEXPERIMENT=simd go test -v -run TestFindReduceThresholds ./internal/gobackend/ops
   ```
2. Inspect the crossover points (where `Ratio > 1.05` indicates Scalar is faster).
3. Update `avx512ReduceThresholds` in [`compute/internal/gobackend/ops/reduce_thresholds_amd64.go`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/ops/reduce_thresholds_amd64.go).

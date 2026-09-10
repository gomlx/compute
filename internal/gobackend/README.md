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
* In-operation parallelization: `DotGeneral` and `FusedScaledDotProductAttention` are parallelized across worker pools using dynamic lock-free work-stealing.
* Use intrinsics/SIMD on platforms that allow it: AVX2 and AVX-512 SIMD kernels are implemented for `DotGeneral` (GEMM), `FusedScaledDotProductAttention`, `Where`, `FusedLayerNorm`, and activations (`Gelu`).
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

## Assembly Guidelines: Avoiding AVX-SSE Transition Penalties

When writing AMD64 assembly routines for AVX2 or AVX-512, **never mix non-VEX (legacy SSE) instructions with VEX/EVEX instructions**.

### The Problem: AVX $\leftrightarrow$ SSE State Transitions

On x86_64 processors (especially Intel architectures like Skylake through Alder Lake / Raptor Lake), the CPU pipeline manages upper register bits (e.g. bits 128–255 of YMM registers) differently depending on whether VEX-encoded or legacy SSE-encoded instructions are executing.

When a legacy SSE instruction (such as `MOVSS`, `ADDSS`, `DIVSS`, `SQRTSS`, or `CVTSL2SS`) executes while the upper bits of any YMM register are active or not cleanly zeroed:
1. The CPU hardware must save the upper 128 bits of all YMM registers to an internal buffer and transition the register state.
2. Executing a subsequent AVX/VEX instruction requires another transition to restore state.
3. Each transition stalls the execution pipeline for **~70 to 140 CPU cycles**.

In tight loops or row-by-row reductions (such as LayerNorm, RMSNorm, or Softmax), having even 2–4 transitions per row creates a massive **fixed latency penalty**. For example, in a 100-row LayerNorm operation, this introduced an artificial ~18 µs floor regardless of tensor size, completely negating SIMD benefits on small-to-medium batches. Eliminating these transitions yielded a **34× speedup** (from 18.8 µs down to 550 ns for batch size 16) and improved full model training throughput by 20%.

### Rules for Assembly Implementations

1. **Always Use VEX/EVEX Instructions (`V...`)**:
   In any kernel using YMM (`Y0`–`Y15`) or ZMM (`Z0`–`Z31`) registers, **all** scalar floating-point instructions must also use their VEX-prefixed counterparts.

2. **Go Assembly Opcode Names for Scalar VEX Operations**:
   Go's internal assembler has specific naming conventions for 3-operand VEX scalar and conversion instructions:

   | Operation | Legacy SSE (Avoid in AVX code) | VEX Equivalent (Use this) |
   | :--- | :--- | :--- |
   | **Float32 Move** | `MOVSS src, dst` | `VMOVSS src, dst` |
   | **Float64 Move** | `MOVSD src, dst` | `VMOVSD src, dst` |
   | **Float32 Add/Sub/Mul** | `ADDSS / SUBSS / MULSS src, dst` | `VADDSS / VSUBSS / VMULSS src2, src1, dst` |
   | **Float64 Add/Sub/Mul** | `ADDSD / SUBSD / MULSD src, dst` | `VADDSD / VSUBSD / VMULSD src2, src1, dst` |
   | **Float32 Division** | `DIVSS src, dst` | `VDIVSS divisor, dividend, dst` |
   | **Float64 Division** | `DIVSD src, dst` | `VDIVSD divisor, dividend, dst` |
   | **Square Root** | `SQRTSS src, dst` | `VSQRTSS src, src, dst` |
   | **Int32 $\to$ Float32** | `CVTSL2SS reg, xmm` | `VCVTSI2SSL reg, xmm, dst` |
   | **Int64 $\to$ Float64** | `CVTSQ2SD reg, xmm` | `VCVTSI2SDQ reg, xmm, dst` |
   | **Zeroing Registers** | `XORPS xmm, xmm` | `VXORPS ymm, ymm, ymm` |

   > [!NOTE]
   > For `VDIVSS` and `VDIVSD`, the operand order in Go assembly is `VDIVSS divisor, dividend, dst` (e.g. `VDIVSS X0, X1, X1` computes `X1 = X1 / X0`).
   > For `VCVTSI2SSL`, the trailing `L` indicates a 32-bit integer register (`R11` / `R11D`), while `VCVTSI2SDQ` with trailing `Q` indicates a 64-bit integer register.

3. **Always Call `VZEROUPPER` Before Returning (`RET`)**:
   Standard Go compiler code generation uses legacy SSE instructions for scalar float math. If an assembly routine leaves the upper halves of YMM/ZMM registers in a "dirty" state upon returning, the very next Go-compiled SSE instruction will trigger an AVX $\to$ SSE penalty.
   Always emit `VZEROUPPER` immediately before every `RET` instruction in functions using 256-bit or 512-bit registers.

## Fused Operations & Intra-Op Parallelization Architecture

When implementing compute-intensive fused operations (such as `FusedScaledDotProductAttention`), the Go backend employs several core design principles to maximize hardware efficiency on modern multi-core x86_64 processors:

### 1. Dynamic Work-Stealing with Atomic Counters

Rather than using Go channels or statically partitioning work across goroutines:
- **Avoid Go Channels in Inner Loops**: Distributing work units through Go channels introduces channel mutex lock contention, channel buffer overhead, and goroutine parking/unparking latency.
- **Avoid Static Chunking**: Dividing tasks evenly across $W$ workers upfront leads to thread imbalance when individual tasks have variable workloads (e.g. varying sequence lengths due to padding) or when the number of tasks does not divide evenly by worker count.
- **Atomic Work-Stealing**:
  1. Workers are spawned up to $\min(N_{\text{tasks}}, \text{backend.Workers})$.
  2. Workers dynamically steal the next available task index using an atomic counter:
     ```go
     for {
         taskIdx := int(taskCounter.Add(1) - 1)
         if taskIdx >= numTasks {
             break
         }
         // Process taskIdx
     }
     ```
  3. This provides zero-allocation, lock-free dynamic load balancing where faster cores naturally process more tasks.

### 2. Cache-Aware Task Decomposition (KV-Head Grouping in GQA/MQA)

In Grouped Query Attention (GQA) and Multi-Query Attention (MQA), multiple query heads share a single key/value (KV) head ($N_{\text{query}} \ge N_{\text{kv}}$):
- If tasks are decomposed per individual query head, multiple worker goroutines concurrently compete for memory bandwidth fetching identical $K$ and $V$ tensors from L3 cache or DRAM.
- By defining each parallel task as a $(batch, kv\_head)$ chunk, a single worker iterates over all $G = N_{\text{query}} / N_{\text{kv}}$ query heads sequentially.
- This guarantees that the key and value sequences ($S_{\text{kv}} \times D$) are loaded once and remain resident in the core's private L1/L2 cache while computing all $G$ query heads.

### 3. Zero-Transpose Direct Strided Memory Access

Deep learning models frequently switch between layout conventions (e.g. `LayoutBSHD` $[B, S, H, D]$ and `LayoutBHSD` $[B, H, S, D]$):
- Decomposing attention into explicit 4D tensor permutations (`Transpose`) incurs severe memory copying penalties (accounting for ~20% of CPU time and dominating `runtime.memmove`).
- In row-major tensors, the inner head dimension $D$ is contiguous (stride 1) in both layouts.
- By parametrizing inner loops with sequence strides (`qSeqStride`, `kvSeqStride`) and query group strides (`qGroupStride`), compute kernels can read operands and write outputs directly in their native layouts with zero tensor copying.

### 4. In-Cache Softmax & Register-Accumulated Values

- **Avoid Materializing $S \times S$ Matrices in Memory**: Writing intermediate attention logits or probabilities to DRAM creates severe bandwidth bottlenecks.
- **Per-Worker Scratch Buffers**: Each worker allocates a small temporary slice ($S_{\text{kv}}$ floats) reused across tokens.
- **In-Register Fused Math**: Scale factors, causal masks, additive masks, and attention biases are folded directly into vector registers during dot-product accumulation. Softmax exponentiation uses fast vectorized polynomial approximations (`exp512` / `exp256`), and weighted values ($P \cdot V$) accumulate directly into vector registers before writing directly to the final destination buffer.

### 5. Head Dimension Specialization ($D \in \{32, 64, 128\}$)

When the inner loop over dimension $D$ has a dynamic upper bound, the compiler cannot unroll vector loops and must emit loop counter branches. Specializing kernels for standard head dimensions allows completely unrolling into a fixed set of vector registers (e.g. 2 `Float32x16` registers for $D=32$ on AVX-512, 4 for $D=64$, 8 for $D=128$), sustaining near-peak FMA throughput.



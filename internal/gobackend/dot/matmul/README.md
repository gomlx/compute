# Go Backend Matrix Multiplication (`matmul`)

This package implements high-performance CPU matrix multiplication (`DotGeneral`) for the GoMLX `compute` Go backend. It powers tensor contractions, linear layers, and convolutions when running on CPU.

---

## 1. Architectural Overview

Matrix multiplication performance on modern superscalar CPUs is constrained by two bottlenecks:
1. **Memory Hierarchy & Bandwidth**: Keeping the CPU caches (L1, L2, L3) fed without stalling on main memory RAM access.
2. **Instruction Pipeline & Register Utilization**: Keeping the hardware SIMD Fused Multiply-Add (FMA) execution units 100% saturated without register spills or pipeline bubbles.

To address these constraints across varied platforms and problem sizes, `matmul` implements a multi-tier architecture:

```
                      +-----------------------------+
                      |   DotGeneral / Router       |
                      +-----------------------------+
                                     |
              +----------------------+----------------------+
              |                                             |
     [Small Matrix Path]                           [Large Matrix Path]
  (Direct, no packing)                           (BLIS cache blocking)
              |                                             |
   +----------+----------+                       +----------+----------+
   |          |          |                       |          |          |
NoSIMD      AVX2      AVX-512                 NoSIMD      AVX2      AVX-512
                                                            |          |
                                                         Assembly   Assembly
                                                   (Microkernel + Pack)
```

### Supported Data Types & Accumulation Rules
- **Float32**: 32-bit float inputs $\to$ 32-bit float output.
- **Float64**: 64-bit double inputs $\to$ 64-bit double output.
- **Float16 / BFloat16**: 16-bit half-precision inputs $\to$ **Float32 accumulation & output**. This preserves precision during long contracting dot-products.

### Supported Matrix Layouts
- **NonTransposed** ($[M, K] \times [K, N] \to [M, N]$): Standard row-major matrix multiplication.
- **Transposed** ($[M, K] \times [N, K] \to [M, N]$): RHS is stored transposed. In this layout, RHS has the exact same row-major contracting layout as LHS, allowing LHS packing routines to be reused directly.

---

## 2. Small vs. Large Algorithms

The package routes operations based on total arithmetic operations ($\text{FLOPS} = \text{batch} \times M \times N \times K$):

### Small Matrix Algorithm (`*small.go`)
- **When used**: For matrix multiplications below the size threshold (e.g. $M, N, K \le 64$ or small batch sizes).
- **Strategy**: Direct computation on input memory without copying or packing.
- **Rationale**: For small tensors, the memory allocation and copying overhead of panel packing exceeds any cache-locality benefits. Direct vectorized loops or scalar fallbacks minimize latency.

### Large Matrix Algorithm (`*large.go`)
- **When used**: For medium-to-large matrices where arithmetic intensity justifies cache blocking.
- **Strategy**: GotoBLAS / BLIS 5-loop cache-blocking architecture with packed contiguous panels.
- **Cache Hierarchy Blocking**:
  - $M_c \times K_c$ **LHS Panel**: Sized to fit comfortably in **L2 cache** (e.g., $32 \times 192$ for Float32).
  - $K_c \times N_c$ **RHS Panel**: Sized to fit in **L3 cache** (e.g., $192 \times 512$ for Float32).
  - $M_r \times N_r$ **Register Tile**: Kept entirely in CPU vector registers during the innermost microkernel loop ($8 \times 32$ for AVX-512 Float32/Float16/BFloat16, $8 \times 16$ for Float64).

---

## 3. Data Packing & Fast Transposition

Non-unit memory strides in row-major matrices cause CPU cache thrashing and prevent continuous vector loads. Packing reorders sub-matrices into cache-friendly sequential buffers.

### Pack LHS (`unsafePackLHS`)
- Takes $M_c$ rows and $K_c$ contracting columns from the LHS matrix.
- Reorganizes them into panels of $M_r = 8$ rows (or $M_r = 4$ for Go SIMD / AVX2).
- Each 8-row strip is transposed into contiguous 8-element column vectors:
  $$\begin{bmatrix} L_{0,0} & L_{0,1} & \dots \\ L_{1,0} & L_{1,1} & \dots \\ \vdots & \vdots & \ddots \\ L_{7,0} & L_{7,1} & \dots \end{bmatrix} \implies [L_{0,0}, L_{1,0}, \dots, L_{7,0}], [L_{0,1}, L_{1,1}, \dots, L_{7,1}], \dots$$
- **Why this layout?** In the microkernel, each 8-element strip represents the contracting values for the 8 active rows at step $k$. A single sequential read loads all 8 values to be broadcast across the accumulators (`VBROADCASTSS` for F32, `VBROADCASTSD` for F64, or `VCVTPH2PS` / `VPMOVZXWD` for F16/BF16).

### Fast Assembly Transposition vs. Go SIMD
Transposing rows into column vectors is a performance-critical step:
- **Go 1.27.1 SIMD**: Only exposes permutation intrinsics like `archsimd.Permute2x256Float32x16` or `Permute4x64`. Transposing blocks required multiple stages of shuffles and permutations across lanes, creating significant instruction overhead.
- **Handwritten AVX-512 Assembly** (`avx512_pack_amd64_*.s`):
  - **Float32**: Uses direct interleaving unpack instructions (`VUNPCKLPS`, `VUNPCKHPS`) followed by 128-bit lane permutes (`VSHUFF32X4`).
  - **Float64**: Uses `VUNPCKLPD`, `VUNPCKHPD`, and `VSHUFF64X2`.
  - **Float16 / BFloat16**: Uses word-level unpacking (`VPUNPCKLWD`, `VPUNPCKHWD`) followed by `VPERMT2W`.
- **Result**: Packing time dropped by **60% to 75%** (a **2.5× to 4× speedup** in `PackLHS`), reducing packing overhead to only ~5% of the total matrix multiplication time.

### Pack RHS (`packRHS` & `avx512PackRHSFullStripsAsm`)
- Slices $N_c$ columns into blocks of $N_r = 32$ columns (16 for Float64).
- Stores each row of 32 elements sequentially.
- This allows the microkernel to load RHS rows directly into 2 full 512-bit vector registers (`2 × 16 = 32` floats) using unaligned vector loads (`VMOVDQU32`).

---

## 4. AVX-512 GEMM Microkernel Design

The innermost microkernel computes a block of $M_r = 8$ rows $\times N_r = 32$ columns over $K_c$ contracting steps.

```
       RHS Panel (32 columns -> 2 AVX-512 registers: Z16, Z17)
       +-----------------------------------+-----------------------------------+
       |          cols 0..15 (Z16)         |         cols 16..31 (Z17)         |
       +-----------------------------------+-----------------------------------+
LHS    |
Row 0  |        Z0 += Z18 * Z16            |        Z1 += Z18 * Z17
(Z18)  |
Row 1  |        Z2 += Z19 * Z16            |        Z3 += Z19 * Z17
(Z19)  |
Row 2  |        Z4 += Z20 * Z16            |        Z5 += Z20 * Z17
(Z20)  |
Row 3  |        Z6 += Z21 * Z16            |        Z7 += Z21 * Z17
(Z21)  |
Row 4  |        Z8 += Z22 * Z16            |        Z9 += Z22 * Z17
(Z22)  |
Row 5  |        Z10 += Z23 * Z16           |        Z11 += Z23 * Z17
(Z23)  |
Row 6  |        Z12 += Z24 * Z16           |        Z13 += Z24 * Z17
(Z24)  |
Row 7  |        Z14 += Z25 * Z16           |        Z15 += Z25 * Z17
(Z25)  +-----------------------------------+-----------------------------------+
```

### 2-Stage Ping-Pong Pipeline
To fully hide instruction and memory load latency, the microkernel unrolls $K$ by 2 using a 2-stage ping-pong register buffer:
- **Buffer A**: Uses $Z_{16}, Z_{17}$ for RHS and $Z_{18}$–$Z_{25}$ for LHS.
- **Buffer B**: Uses $Z_{26}, Z_{27}$ for RHS and reuses $Z_{18}$–$Z_{25}$ for LHS as each row's FMAs complete.
- As Step A executes FMAs on Buffer A, loads for Step B are interleaved into Buffer B, keeping execution units 100% occupied without stalls.

### Why Assembly Was Essential (Go 1.27.1 SIMD Limitations)

While Go 1.27 introduced experimental SIMD via `simd/archsimd`, achieving peak hardware utilization in GEMM required handwritten assembly:

1. **FMA Instruction Encoding (`VFMADD231PS` vs `VFMADD213PS`)**:
   - In Go's `archsimd`, `a.MulAdd(b, c)` emits `VFMADD213PS a, b, c` which computes $a = a \times b + c$, overwriting operand $a$.
   - For GEMM accumulators, the optimal x86 instruction is `VFMADD231PS src1, src2, dst` which computes $dst = src1 \times src2 + dst$. This keeps accumulators permanently in destination registers and allows operands to be read directly from memory or scratch registers without destructive overwrites.
2. **Register Allocation & 32-Register Pressure**:
   - Modern AVX-512 provides 32 vector registers (`Z0` to `Z31`).
   - The $4 \times 64$ microkernel requires **16 dedicated accumulators** (`Z0` to `Z15`).
   - It also needs 4 broadcast registers for LHS (`Z16` to `Z19`) and 4 vector registers for RHS (`Z20` to `Z23`).
   - Go 1.27.1's register allocator struggles to keep 16 live SIMD variables pinned in hardware registers across unrolled loop iterations, frequently emitting register-to-register moves or spilling to stack memory. See [discussion in github.com/golang/go/issues/78753#issuecomment-5535527697](https://github.com/golang/go/issues/78753#issuecomment-5535527697).
   - Handwritten assembly (`avx512_large_amd64_*.s`) guarantees that `Z0-Z15` never leave the register file throughout the entire contracting loop.
3. **FMA Pipeline Latency Hiding**:
   - Modern x86 cores (e.g. AMD Zen 4/5, Intel Sapphire Rapids) contain two 512-bit FMA execution ports with a **4-cycle pipeline latency**.
   - With 16 independent accumulator registers and loop unrolling in $K$, each accumulator is updated once every 16 instructions. The 4-cycle pipeline latency is completely hidden, sustaining 2 FMAs per cycle (near 100% theoretical peak compute throughput).

---

## 5. Direct Output Accumulation in L2 Cache

### The Problem in Earlier Versions
Earlier implementations wrote microkernel results to an intermediate `packedOutput` buffer. For every contracting panel $K$, a separate function (`avx512ApplyPackedOutput`) read `packedOutput`, read `outputMatrix` from main memory, added them, and wrote back to `outputMatrix`.
- This resulted in repeated read-modify-write cycles across main memory.
- In CPU profiles, `avx512ApplyPackedOutput` consumed **9.54%** of total execution time.

### The In-Place Accumulation Solution
We introduced an `accumulate bool` parameter directly into all microkernels:
- An L2 cache-resident accumulation buffer (`accumBuffer`) is allocated per worker thread.
- **For the first contracting step ($K = 0$)**: The microkernel executes with `accumulate = false`, overwriting the L2 buffer via `VMOVDQU32`.
- **For subsequent contracting steps ($K > 0$)**: The microkernel executes with `accumulate = true`. It loads the existing partial sum from L2 into the accumulators using `VADDPS (DX), Z, Z` before storing back.
- **Final writeback**: Main memory `outputMatrix` is **never touched during the contracting loop**. After all $K$ steps finish, a single sequential copy transfers the finished sum from L2 cache to main memory.

---

## 6. Optimization History & Benchmark Gains

The following benchmarks were recorded on an **AMD Ryzen 9 9950X3D** (16 cores / 32 threads, AVX-512, Ubuntu 26.04, CPU governor: balanced) for large Float32 matrix multiplications (`NoBatch-Large-1`: $1536 \times 1024 \times 1920$):

| Optimization Milestone | Throughput | Latency | Key Changes |
| :--- | :--- | :--- | :--- |
| **Baseline (Pure Go SIMD)** | **1,850 GFlops/s** | ~3.30 ms | Pure Go `archsimd`, 16-var loop, `VFMADD213` |
| **Round 1: Assembly Microkernel** | **2,277 GFlops/s** (+23.1%) | ~2.65 ms | Dedicated `Z0-Z15` accumulators, `VFMADD231PS` |
| **Round 2: Fast Assembly PackLHS** | **2,380 GFlops/s** (+4.5%) | ~2.52 ms | Hardware `VUNPCK` transpositions for F32/F64/F16/BF16 |
| **Round 3: Direct L2 Accumulation** | **2,677 - 2,946 GFlops/s** (+17.6%) | **2.26 ms** | Direct in-place L2 accumulation (`accumulate bool`), bypassing intermediate `packedOutput` |

### Overall Impact (AVX-512)
- **Total Throughput Gain**: **+44.7% to +59.2%** (from 1,850 GFlops/s up to **~2.95 TFlops/s**).
- **Compute Efficiency**: The CPU profile shows compute math (`avx512LargeKernelFloat32Asm`) now accounts for **81.4%** of all CPU cycles, with output writeback reduced to just 4.3% (a single sequential memory write).

### AVX2 Optimization Milestone Results

The following benchmarks were recorded on the **AMD Ryzen 9 9950X3D** (AVX-512 disabled via `GOMLX_GO_SIMD_AVX512=0`, CPU pinned at 3500 MHz, `nice -n -20`):

| Benchmark Case | Matrix Dimensions ($M \times K \times N$) | Baseline (Pure Go SIMD) | Optimized (AVX2 Assembly) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **NoBatch-Large-1** | $1536 \times 1920 \times 1024$ | 704.6 GFlops/s (8.60 ms) | **1,221 GFlops/s** (4.90 ms) | **+73.3%** 🚀 |
| **NoBatch-Large-2** | $1024 \times 1920 \times 1536$ | 720.9 GFlops/s (8.40 ms) | **1,234 GFlops/s** (4.90 ms) | **+71.2%** 🚀 |
| **NoBatch-Large-3** | $2048 \times 2048 \times 2048$ | 766.2 GFlops/s (22.40 ms) | **1,332 GFlops/s** (12.90 ms) | **+73.8%** 🚀 |
| **Batched-Large-1** | $16 \times 1536 \times 1920 \times 1024$ | 843.6 GFlops/s (114.6 ms) | **1,433 GFlops/s** (67.40 ms) | **+69.9%** 🚀 |
| **Batched-Large-2** | $16 \times 1024 \times 1920 \times 1536$ | 822.1 GFlops/s (117.6 ms) | **1,381 GFlops/s** (70.00 ms) | **+68.0%** 🚀 |

---

## 7. Exploration: Alternative Kernel Geometries (4x64 vs 6x48 vs 8x32)

During optimization, we thoroughly evaluated three microkernel geometries on AVX-512 (AMD Zen 5, 32 ZMM registers):
1. **4 rows × 64 cols** ($M_r = 4, N_r = 64$): 16 accumulators ($4 \times 4$ ZMMs), 4 RHS vectors, 4 LHS broadcasts.
2. **6 rows × 48 cols** ($M_r = 6, N_r = 48$): 18 accumulators ($6 \times 3$ ZMMs), 3 RHS vectors, 6 LHS broadcasts.
3. **8 rows × 32 cols** ($M_r = 8, N_r = 32$): 16 accumulators ($8 \times 2$ ZMMs), 2 RHS vectors, 8 LHS broadcasts.

### Single-Core Compute Ceiling
In isolated single-core benchmarks (resident $192 \times 384 \times 192$ panel):
* **4x64**: **351.8 GFlops/s** (99.9% of Zen 5 physical dual-512 FMA pipe capacity).
* **8x32**: **347.8 GFlops/s** (98.8% of Zen 5 physical dual-512 FMA pipe capacity).

All architectures max out the execution units in cache; the real differentiator is memory hierarchy, dimension divisibility, and thread spatial partitioning.

### Full GEMM Benchmark Comparison

| Problem Regime | Matrix Shapes | $4 \times 64$ Baseline | $8 \times 32$ Geometry | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Giant Square** | $2048 \times 2048 \times 2048$ | **3,017 GFlops/s** ($5.7\text{ ms}$) | **2,817 GFlops/s** ($6.1\text{ ms}$) | -6.6% |
| **Large Wide** | $1536 \times 1920 \times 1024$ | **2,695 GFlops/s** ($2.2\text{ ms}$) | **2,423 GFlops/s** ($2.5\text{ ms}$) | -10.0% |
| **Transformer Projections** | $42 \times 48 \times 1536 \times 384$ | **1,610 GFlops/s** ($1.5\text{ ms}$) | **1,929 GFlops/s** ($1.2\text{ ms}$) | **+19.8%** 🚀 |
| **Transformer Projections** | $64 \times 32 \times 1536 \times 384$ | **1,641 GFlops/s** ($1.5\text{ ms}$) | **1,955 GFlops/s** ($1.2\text{ ms}$) | **+19.1%** 🚀 |
| **Transformer Projections** | $85 \times 24 \times 1536 \times 384$ | **1,828 GFlops/s** ($1.3\text{ ms}$) | **1,994 GFlops/s** ($1.2\text{ ms}$) | **+9.1%** 🚀 |
| **Transformer Projections** | $16 \times 128 \times 1536 \times 384$ | **1,806 GFlops/s** ($1.3\text{ ms}$) | **1,956 GFlops/s** ($1.2\text{ ms}$) | **+8.3%** 🚀 |

### Why 8x32 was Adopted as the Standard Architecture
While $4 \times 64$ reaches higher peak throughput on massive square matrices due to lower instruction decoding overhead (8 loads vs 10 loads per 16 FMAs), **$8 \times 32$** is chosen as the standardized geometry:

1. **Massive Wins on Real Transformer Shapes (+8% to +20%)**:
   In modern deep learning (e.g. BAAI embedding, BERT, LLaMA), column projections frequently have $N=384, 512, 768$ and moderate sequence lengths ($M \in [16, 128]$). $N_r = 32$ tiles these shapes with zero remainder padding and enables much finer thread work partitioning across 32+ cores.
2. **Arithmetic Intensity & L1/L2 RHS Bandwidth**:
   In $8 \times 32$, each loaded RHS vector is reused across **8 FMAs** (compared to 4 in $4 \times 64$), cutting RHS memory traffic in half and reducing cache port pressure.
3. **Perfect Cache-Line Alignment**:
   8 rows × 4 bytes = 32 bytes (exactly half of a 64-byte cache line). Two consecutive $K$ steps form a perfectly aligned 64-byte cache line, avoiding the split-cache-line penalties that afflicted $6 \times 48$.
4. **Universal Symmetry Across Data Types**:
   Across all types, the accumulator tile is symmetrically **8 rows × 2 vector registers** (16 ZMM accumulators):
   * Float32: $8 \times 32$
   * Float16 / BFloat16 $\to$ Float32: $8 \times 32$
   * Float64: $8 \times 16$

### Note on 6x48
The $6 \times 48$ geometry delivered +20% to +30% on dimensions that were exact multiples of 48, but suffered severely on powers-of-two ($1024 = 21 \times 48 + 16$) due to fractional remainder tails, and 24-byte LHS strips straddled 64-byte cache lines. $8 \times 32$ captures similar transformer acceleration without any of the alignment or divisibility drawbacks.

---

## 8. Exploration: RHS Packing Optimization (Pre-Packing vs Assembly Unrolling)

During optimization of RHS packing, we investigated two approaches:
1. **Multithreaded Pre-Packing of RHS**: Pre-packing the entire RHS matrix in parallel across all worker threads into a single shared buffer before initiating the GEMM compute phase.
2. **AVX-512 Assembly Microkernel with 4-Row Unrolling**: Accelerating worker-local RHS packing using handwritten AVX-512 assembly.

### 1. Multithreaded Global Pre-Packing (Why It Regressed)
In theory, pre-packing the entire RHS matrix upfront across 32 threads should eliminate redundant packing across workers that share column ranges. However, in benchmarks, this caused a **~15% regression** (dropping throughput from ~2,750 down to 2,285 GFlops/s). Profiling revealed three root causes:

1. **L1/L2 Cache Locality Loss**:
   * In worker-local packing, each worker tiles $N$ into narrow chunks (e.g. 128 cols, $96\text{ KB}$ for $K_c=192$). It packs this $96\text{ KB}$ directly into its private L2 cache ($1\text{ MB}$ per core on Zen 5) immediately before multiplying it by multiple LHS rows. The compute microkernel reads RHS at full L1/L2 bandwidth ($>3\text{ TB/s}$ per core).
   * In global pre-packing, the entire matrix ($7.86\text{ MB}$) is written upfront to memory. By the time GEMM starts, each worker's L1 and L2 caches are completely cold, forcing initial misses to L3/DRAM.
2. **Dual-CCD NUMA / Interconnect Traffic**:
   * On dual-CCD architectures (such as the AMD Ryzen 9 9950X3D with two 8-core CCDs), strips packed by a core on CCD0 must cross the high-latency Infinity Fabric when read by a worker running on CCD1.
3. **Double Synchronization Barrier**:
   * Calling `backend.Workers.Saturate` twice per matrix multiplication (once for pre-packing, once for GEMM) added lock contention and thread synchronization overhead.

### 2. AVX-512 Assembly Microkernel (`avx512PackRHSFullStripsAsm`)
Instead of global pre-packing, we accelerated the worker-local packing path with a dedicated AVX-512 assembly kernel (`avx512_pack_rhs_amd64.s`):
* **Elimination of Compiler Overhead**: Go's pure SIMD loop previously emitted 8 separate `LEAQ` index calculations per row (4 for loads, 4 for stores). The assembly kernel uses direct hardware displacement offsets.
* **4-Row Unrolling with 16 ZMM Registers**:
  * Unrolls 4 consecutive rows ($K$) per iteration: $1024\text{ bytes}$ per iteration for Float32 (256-byte strips), $512\text{ bytes}$ for Float16/BFloat16, and $256\text{ bytes}$ for Float64.
  * Interleaves 16 ZMM loads (`Z0`–`Z15`) and stores, allowing CPU out-of-order execution to saturate memory copy bandwidth at **~70 GB/s** (the physical limit of dual-channel DDR5-6000 memory).
* **Impact**: Keeps RHS hot in each core's private L2 cache while cutting packing latency, lifting Large benchmarks across the board (`NoBatch-Large-2` to **2,733 GFlops/s**, `NoBatch-Large-3` to **2,934 GFlops/s**, and `Batched-Large-1` to **2,802 GFlops/s** under dynamic boost).

---

## 9. Cache Blocking Tuning ($K_c, M_c, N_c$)

To maximize hardware efficiency on multi-core Zen 5 architectures, we performed an empirical grid search over the cache blocking parameters ($K_c$, $M_c$, $N_c$). To eliminate thermal throttling and dynamic frequency scaling noise, the benchmarks were conducted with CPU frequency pinned at 3500 MHz and executed with `nice -n -20`:

### 1. Contracting Dimension Blocking ($K_c$)
The contracting chunk $K_c$ governs how much of the LHS and RHS strips reside simultaneously in the core's private L1 Data cache (48 KB):
* **LHS strip footprint**: $M_r \times K_c \times 4\text{ bytes} = 8 \times K_c \times 4$
* **RHS strip footprint**: $N_r \times K_c \times 4\text{ bytes} = 32 \times K_c \times 4$

| $K_c$ | Total L1 Working Set | NoBatch-Large-1 ($1536 \times 1920 \times 1024$) | NoBatch-Large-2 ($1024 \times 1920 \times 1536$) | NoBatch-Large-3 ($2048 \times 2048 \times 2048$) |
|---|---|---|---|---|
| **128** | 20.0 KB | 1,954 GFlops/s | **2,160 GFlops/s** | **2,392 GFlops/s** |
| **160** | 25.6 KB | 1,981 GFlops/s | 2,194 GFlops/s | 2,324 GFlops/s |
| **192** | 30.7 KB | **2,027 GFlops/s** | 2,115 GFlops/s | 2,355 GFlops/s |
| **256** | 41.0 KB | 1,840 GFlops/s | 1,957 GFlops/s | 2,235 GFlops/s |
| **384** | 61.4 KB (> 48 KB) | 1,915 GFlops/s | 2,059 GFlops/s | 2,328 GFlops/s |

* **Analysis**: $K_c \in [128, 192]$ is optimal. When $K_c \ge 256$, the working set approaches or exceeds the 48 KB capacity of the L1D cache. In a 12-way associative cache, line conflicts and stack variable evictions drop throughput by $\sim 10\%$. $K_c = 192$ evenly divides transformer dimensions ($384, 1536, 1920$), making it the standard choice for Float32.

### 2. LHS Panel Height ($M_c$)
$M_c$ determines how many rows of LHS are packed together into the core's private L2 cache (1 MB):
* **L2 footprint**: $M_c \times K_c \times 4\text{ bytes} = 32 \times 192 \times 4 = 24.6\text{ KB}$.

| $M_c$ | Rows / Strip | NoBatch-Large-1 | NoBatch-Large-2 | NoBatch-Large-3 |
|---|---|---|---|---|
| **16** | 2 | 1,910 GFlops/s | 2,083 GFlops/s | 2,354 GFlops/s |
| **24** | 3 | 1,982 GFlops/s | 2,135 GFlops/s | 2,334 GFlops/s |
| **32** | 4 | 1,842 GFlops/s | 2,005 GFlops/s | **2,369 GFlops/s** |
| **40** | 5 | 1,993 GFlops/s | **2,241 GFlops/s** | 2,345 GFlops/s |
| **48** | 6 | **2,005 GFlops/s** | 2,201 GFlops/s | 2,326 GFlops/s |
| **64** | 8 | 1,902 GFlops/s | 2,037 GFlops/s | 2,343 GFlops/s |

* **Analysis**: $M_c = 32$ to $48$ offers the best tradeoff. Values $\ge 64$ create task granularities that are too coarse for 32 worker threads on matrices where $M \le 1024$, causing thread load imbalance, while $M_c \le 16$ increases packing overhead. $M_c = 32$ is chosen as standard for powers-of-two divisibility.

### 3. RHS Panel Width ($N_c$)
$N_c$ controls the column tile size of RHS kept in the shared L3 cache:

| $N_c$ | 32-Worker Aggregate RHS Footprint | Batched-Large-1 | Batched-Large-2 | NoBatch-Large-3 |
|---|---|---|---|---|
| **256** | 6.25 MB | 2,025 GFlops/s | 2,122 GFlops/s | 2,303 GFlops/s |
| **384** | 9.38 MB | 1,830 GFlops/s | 2,253 GFlops/s | 2,195 GFlops/s |
| **512** | 12.5 MB | **2,365 GFlops/s** | **2,169 GFlops/s** | **2,300 GFlops/s** |
| **768** | 18.75 MB | 2,150 GFlops/s | 2,080 GFlops/s | 2,273 GFlops/s |

* **Analysis**: $N_c = 512$ is **$\sim 16\%$ faster** on large batched matrices (40.9 ms vs 47.7 ms), while keeping the aggregate 32-thread RHS panel footprint well within the 96 MB L3 cache of the AMD 9950X3D.

### Standardized Cache Parameters
| Data Type | $M_r$ (Rows) | $N_r$ (Cols) | $K_c$ (L1 Contracting) | $M_c$ (L2 Rows) | $N_c$ (L3 Cols) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Float32** | 8 | 32 | 192 | 32 | 512 |
| **Float16** | 8 | 32 | 128 | 32 | 768 |
| **BFloat16** | 8 | 32 | 128 | 32 | 768 |
| **Float64** | 8 | 16 | 64 | 16 | 256 |

---

## 10. File Map & Code Generation

Because `matmul` provides high performance across multiple architectures and data types, Go template generation is used to maintain symmetry:

| File | Purpose |
| :--- | :--- |
| `matmul.go` | Cache parameters, priority constants, and feature flags. |
| `avx512_router.go` | Routes between Small and Large AVX-512 kernels. |
| `avx512_large.go` | Base template for AVX-512 large matrix multiplication (Go SIMD + Assembly caller). |
| `avx512_large_amd64.go` | Assembly function forward declarations (`//go:noescape`). |
| `avx512_large_amd64_*.s` | Handwritten AVX-512 GEMM microkernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx512_pack_amd64_*.s` | Handwritten AVX-512 fast LHS transposition and packing kernels. |
| `avx512_pack_rhs_amd64.s` | Handwritten AVX-512 unrolled RHS strip packing kernel. |
| `avx2_router.go` | Routes between Small and Large AVX2 kernels. |
| `avx2_large.go` | Base template for AVX2 large matrix multiplication (Go SIMD + Assembly caller). |
| `avx2_large_amd64.go` | AVX2 assembly function forward declarations (`//go:noescape`). |
| `avx2_large_amd64_*.s` | Handwritten AVX2 GEMM microkernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx2_pack_amd64_*.s` | Handwritten AVX2 fast LHS transposition and packing kernels. |
| `avx2_pack_rhs_amd64.s` | Handwritten AVX2 unrolled RHS strip packing kernel. |
| `avx2_*.go` | AVX2 (256-bit SIMD) router, small kernels, large kernels, and transpositions. |
| `nosimd_*.go` | Architecture-agnostic portable Go fallback with scalar loop blocking. |
| `gen_*` | **Auto-generated files** created by `alternates_generator` for alternative dtypes (`f16`, `bf16`, `f64`). |

### Regenerating Alternates
When modifying any of the base template files (e.g. `avx512_large.go`, `avx2_large.go`, `nosimd_large.go`), regenerate the type-specific files:

```bash
go generate ./internal/gobackend/dot/matmul
```

### Running Tests & Benchmarks

```bash
# Run all matmul internal tests:
go test -v ./internal/gobackend/dot/matmul

# Run backend compliance tests:
go test -v -run TestCompliance ./gobackend

# Run Large DotGeneral benchmarks:
go test -run none -bench BenchmarkCompliance/DotGeneral/Large ./gobackend
```

---

## 11. Future Work: Multi-Geometry Dynamic Microkernel Selection

State-of-the-art inference engines such as Google's **XNNPACK**, **BLIS**, and Intel's **oneDNN** implement families of specialized microkernels rather than a single fixed geometry. Potential future enhancements for `compute/dot/matmul`:

1. **Dynamic Multi-Geometry Microkernel Routing**:
   * Inspect $(M, N, K)$ at dispatch time to choose the optimal microkernel:
     * **4x64**: Selected for massive square or wide matrices where $N \pmod{64} == 0$ and $M \ge 512$ (delivering ~3,000 GFlops/s peak).
     * **8x32**: Selected for transformer projections and moderate sequence lengths ($N \pmod{32} == 0, N \le 512$) (delivering +8% to +20% higher throughput).
     * **6x48**: Specialized for models whose hidden sizes are fixed multiples of 48.
2. **Dedicated GEMV Fast Paths ($M=1$ or $M \le 3$)**:
   * In LLM autoregressive token generation (e.g. sequence length $M=1$), 2D BLAS tiling and LHS packing introduce unnecessary memory copies.
   * A dedicated vector-matrix microkernel ($1 \times 64$ or $1 \times 32$) that directly reads the single activation vector and streams weights without packing can yield substantial speedups for token-by-token generation.
3. **Specialized Edge Remainder Kernels**:
   * For matrix boundaries ($M \pmod{M_r} \ne 0$ or $N \pmod{N_r} \ne 0$), specialized remainder kernels (e.g. $1 \times N_r$, $2 \times N_r$, $3 \times N_r$) avoid zero-padding and wasted FMA cycles on trailing rows.


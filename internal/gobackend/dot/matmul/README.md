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

### Portable Pure-Go Packing (`unsafePackRHS` & `unsafePackLHS`)
For architectures without SIMD support (or when building with pure Go portability):
- **`unsafePackRHS` (Fixed-Array Pointer Loads/Stores)**: For $N_r = 4$, rather than looping over columns individually, each 4-element row chunk is copied in a single operation via pointer cast to a fixed array (`*(*[4]T)(unsafe.Pointer(dstPtr)) = *(*[4]T)(unsafe.Pointer(pSrcRow))`). On modern 64-bit architectures (x86-64, ARM64), the compiler translates this into a single 128-bit load/store instruction pair.
  - **Result**: `PackRHS` latency dropped from **2.30 ms to 371.6 µs** on Float32 (**6.1× speedup**) and from **2.00 ms to 323.6 µs** on BFloat16 (**6.2× speedup**).
- **`unsafePackLHS` (Contiguous Pointers)**: For $M_r = 2$ and $M_r = 4$, the destination pointer advances contiguously through packed memory while unrolling row loads, eliminating index multiplications and bounds checks.
  - **Result**: `PackLHS` latency dropped from **1.86 ms to 869.3 µs** on Float32 (**2.14× speedup**).

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

### Why Assembly Was Essential for Small MatMul As Well

When benchmarking the small matrix multiplication path (where tensors are un-packed and accessed directly in place), we implemented and empirically compared both pure Go `archsimd` and handwritten AVX-512 assembly for the Transposed $4 \times 4$ microkernel:

- **Baseline (1 accumulator)**: 11.34 µs (6.22 GFlops/s)
- **Go SIMD (`archsimd`, 16 accumulators)**: 3.75 µs (18.82 GFlops/s) — 3.0× faster than baseline
- **Handwritten Assembly (`VFMADD231PS`, 16 accumulators)**: **1.11 µs (63.38 GFlops/s)** — **3.4× faster than Go SIMD** and **10.2× faster than baseline**

**Root Cause**:
Disassembly (`go tool objdump`) revealed that `archsimd.Float32x16.MulAdd` emits destructive `VFMADD213PS`. Because the 16 accumulator registers are constantly overwritten, the Go compiler's SSA register allocator fails to keep all 16 accumulators and operands live in hardware registers. It emits dozens of register-to-register moves (`VMOVDQU64`) and spills 512-bit ZMM registers to stack memory (`0x458(SP)`, `0x418(SP)`, `0xd8(SP)`) in every iteration of the contracting loop.
In contrast, handwritten assembly uses non-destructive `VFMADD231PS` to keep `Z0`–`Z15` permanently pinned in the register file with **zero stack spills, zero register moves**, and an 8-instruction vector reduction macro. Consequently, assembly was adopted as the high-performance implementation for small matmul as well.

### Non-Transposed Small MatMul Optimization

In the Non-Transposed layout ($C = A \times B$ where $A$ is $[M, K]$, $B$ is $[K, N]$, $C$ is $[M, N]$), previous versions suffered from two major bottlenecks:
1. **Fallback to Scalar Go**: The router previously checked `if rhsCrossSize > vecWidth`. For narrow matrices with $N \le 16$ (e.g. $N = 1$ and $N = 4$ in `adult-demo`), it dropped completely into scalar pure Go (`noSIMDRouter`), taking over 37 µs for a $[128, 69] \times [69, 4]$ multiplication.
2. **Intermediate Memory RMW in $K$**: For $N > 16$, the loop nested columns inside the $K$ loop, repeatedly reading and writing partial sums to output memory for every single $k$ step.

We resolved this with a three-tiered AVX-512 strategy:
1. **GEMV Direct Routing ($N = 1$)**: When $N = 1$, $B[K, 1]$ in memory is a contiguous vector of length $K$. This is layout-identical to a Transposed matrix $B^T[1, K]$. It is zero-copy routed directly into `avx512SmallFloat32TransposedAsm`.
2. **Stack-Buffered Transposition for Narrow $N$ ($N \in [2, 15]$)**: For $N < 16$, column vectorization cannot fill a 16-wide vector (leaving SIMD lanes idle or falling back to scalar). In contrast, $K$ is typically $\ge 16$. We transpose $B$ into a stack-allocated buffer (e.g. $69 \times 4 = 276$ floats = 1.1 KB in L1 cache, taking ~15 ns) and invoke `avx512SmallFloat32TransposedAsm`, unlocking full 16-wide 512-bit vectorization across $K$.
3. **Column-Vectorized Assembly Microkernels ($N \ge 16$)**: For wide matrices ($N \ge 16$), the loop is inverted so column vectors are processed in the outer loops and $K$ is in the inner loop. Accumulators stay resident in ZMM registers across the entire $K$ loop, writing to output memory once at the end:
   - `avx512SmallNonTransposedTile4x64Float32Asm`: 4 rows $\times$ 64 cols (16 accumulators `Z0`–`Z15`).
   - `avx512SmallNonTransposedTile4x16Float32Asm`: 4 rows $\times$ 16 cols (4 accumulators `Z0`–`Z3`).

#### Non-Transposed Benchmark Results (AMD Ryzen 9 9950X3D):

| Benchmark Case | Baseline (Old Scalar Fallback) | Go SIMD | Handwritten Assembly | Speedup vs. Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **`[128, 69] x [69, 4]`** | 37,227 ns (1.90 GFlops/s) | 3,947 ns (17.90 GFlops/s) | **1,304 ns (54.20 GFlops/s)** | **28.5× faster** 🚀 |
| **`[49, 69] x [69, 4]`** | 12,309 ns (2.20 GFlops/s) | 1,603 ns (16.88 GFlops/s) | **638 ns (42.38 GFlops/s)** | **19.3× faster** 🚀 |
| **`[25, 69] x [69, 4]`** | 4,227 ns (3.26 GFlops/s) | 903 ns (15.29 GFlops/s) | **428 ns (32.21 GFlops/s)** | **9.9× faster** 🚀 |

In end-to-end graph execution (`adult-demo` compliance suite), `non-transposed/[128, 69]x[69, 4]` dropped from **23.5 µs down to 8.7 µs** (a **2.7× end-to-end speedup**, with computation time dropping from 18.3 µs to 3.5 µs).

### AVX2 Small MatMul Optimization (16 YMM Registers)

For processors without AVX-512 (or when `GOMLX_GO_SIMD_AVX512=0`), we adapted the same algorithmic strategies to fit within the 16 256-bit YMM register budget (`Y0`–`Y15`):

1. **Transposed Microkernels (`Tile4x2` and `Tile4x1`)**:
   - `Tile4x2`: 4 LHS rows $\times$ 2 RHS cols = 8 accumulators (`Y0`–`Y7`). 4 LHS vector loads (`Y8`–`Y11`) + 2 RHS vector loads (`Y12`–`Y13`). Total 14 registers with zero spills.
   - `Tile4x1`: 4 LHS rows $\times$ 1 RHS col = 4 accumulators (`Y0`–`Y3`).
   - Fast 6-instruction horizontal vector reduction `REDUCE_Y` (`VEXTRACTF128` $\to$ `VADDPS` $\to$ `VPERMILPS $0xEE` $\to$ `VADDPS` $\to$ `VPERMILPS $0x01` $\to$ `VADDSS`) and `REDUCE_Y_F64` for Float64.
   - Scalar remainder loop unrolls in $K$ directly into scalar registers `X0`–`X7` via `VFMADD231SS` / `VFMADD231SD`.
2. **Non-Transposed Microkernels (`Tile4x16` and `Tile4x8`)**:
   - $N=1$ zero-copy routes directly into Transposed (`Tile4x1`).
   - Narrow $N < 8$ transposes RHS into `stackRhsT` and routes into Transposed (`Tile4x2` / `Tile4x1`).
   - Wide $N \ge 8$ column-vectorized assembly microkernels (`Tile4x16` with 8 accumulators and `Tile4x8` with 4 accumulators).
3. **Multi-DType Support**:
   - Handwritten assembly kernels for `Float32`, `Float64`, `Float16` (via F16C `VCVTPH2PS`), and `BFloat16` (`VPMOVZXWD` + `VPSLLD $16`).

#### AVX2 Small Benchmark Results (AMD Ryzen 9 9950X3D, `GOMLX_GO_SIMD_AVX512=0`):

| Benchmark Case | Baseline (Old Fallback) | Handwritten AVX2 Assembly | Speedup |
| :--- | :--- | :--- | :--- |
| **`NonTransposed/[128, 69] x [69, 4]`** | 37,227 ns (1.90 GFlops/s) | **1,565 ns (45.16 GFlops/s)** | **23.8× faster** 🚀 |
| **`NonTransposed/[49, 69] x [69, 4]`** | 12,309 ns (2.20 GFlops/s) | **917 ns (29.49 GFlops/s)** | **13.4× faster** 🚀 |
| **`NonTransposed/[25, 69] x [69, 4]`** | 4,227 ns (3.26 GFlops/s) | **650 ns (21.23 GFlops/s)** | **6.5× faster** 🚀 |
| **`NonTransposed/[25, 4] x [4, 1]`** | 162 ns (1.23 GFlops/s) | **49.8 ns (4.02 GFlops/s)** | **3.3× faster** 🚀 |
| **`Transposed/[128, 69] x [4, 69]`** | 11,256 ns (6.28 GFlops/s) | **1,418 ns (49.84 GFlops/s)** | **7.9× faster** 🚀 |
| **`Transposed/[25, 69] x [4, 69]`** | 2,197 ns (6.28 GFlops/s) | **502 ns (27.49 GFlops/s)** | **4.4× faster** 🚀 |

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

### Zero-Copy Direct Output Bypass (`canDirectOutput`)
When $K \le K_c$ (single contracting panel) and the sub-panel dimensions align with the kernel tile dimensions ($M_{\text{chunk}} \pmod{M_r} == 0$ and $N_{\text{chunk}} \pmod{N_r} == 0$), intermediate buffer allocation and the final copyback phase can be bypassed completely:
- The microkernel writes its finished results directly into the destination `outputMatrix` using the matrix's native row stride (`rhsCrossSize`).
- This eliminates both the buffer allocation and the final $O(M \times N)$ memory transfer pass, delivering immediate latency improvements especially on smaller matrices and single-panel contractions.

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

### No-SIMD (Portable Pure-Go) Optimization Milestone Results

The No-SIMD implementation provides an architecture-agnostic, pure Go fallback that runs on all platforms without assembly or compiler intrinsics (e.g. ARM64, RISC-V, WebAssembly, or standard x86-64 without SIMD experiments enabled).

The following benchmarks were recorded on the **AMD Ryzen 9 9950X3D** with SIMD completely disabled (`GOEXPERIMENT="" nice -n -20`):

| Component / Benchmark | Baseline (Pure Go) | Optimized (Pure Go) | Improvement |
| :--- | :--- | :--- | :--- |
| **`PackRHS` (Float32, $N_r = 4$)** | 2.30 ms (434 ops/s) | **371.6 µs** (2,691 ops/s) | **6.1× faster** 🚀 |
| **`PackRHS` (BFloat16, $N_r = 4$)** | 2.00 ms (499 ops/s) | **323.6 µs** (3,090 ops/s) | **6.2× faster** 🚀 |
| **`PackLHS` (Float32, $M_r = 2$)** | 1.86 ms (537 ops/s) | **869.3 µs** (1,150 ops/s) | **2.14× faster** 🚀 |
| **`NoBatch-Large-1` ($1536 \times 1920 \times 1024$)** | 101.7 GFlops/s (59.6 ms) | **114.8 GFlops/s** (52.8 ms) | **+13.0%** |
| **`NoBatch-Large-2` ($1024 \times 1920 \times 1536$)** | 98.4 GFlops/s (61.6 ms) | **108.0 GFlops/s** (56.1 ms) | **+9.8%** |
| **`NoBatch-Large-3` ($2048 \times 2048 \times 2048$)** | 99.5 GFlops/s (172.6 ms) | **110.6 GFlops/s** (155.3 ms) | **+11.2%** |

#### No-SIMD Small Matrix Multiplication Optimizations

For small tensor shapes (such as MLP layers and classification heads in `adult-demo`), the No-SIMD kernels translate the key algorithmic insights from the AVX-512 and AVX2 implementations into pure, architecture-agnostic Go:

1. **Zero-Copy GEMV Direct Layout Routing ($N=1$)**:
   - In non-transposed layout with $N=1$, matrix $B[K, 1]$ in memory is a contiguous 1D slice of length $K$, layout-identical to $B^T[1, K]$.
   - Routing $N=1$ directly to the transposed kernel eliminates all strided index arithmetic and transforms column access into a sequential stream.
2. **Dedicated 4-Row GEMV Path**:
   - For $N=1$, accumulates 4 rows of LHS simultaneously (`c0..c3`) using 4 registers, with $K$ unrolled by 2. This cuts RHS memory loads by 75% and achieves **1.8x to 2.2x speedup** on vector multiplications.
3. **Stack-Buffered Transposition for Narrow $N$ ($N \le 16, K \times N \le 2048$)**:
   - In non-transposed small matmul, $B[K, N]$ column access has non-unit stride $N$, causing CPU cache thrashing.
   - Transposing $B[K, N] \to B^T[N, K]$ into a small stack buffer (`[2048]I` = 8 KB for float32, resident in L1 cache) takes negligible time (~15 ns) and converts strided memory access into contiguous streaming reads.
4. **$2 \times 4$ Register-Tiled Transposed Kernel with $K$ Unrolled by 2**:
   - Tiles 2 rows of LHS and 4 columns of RHS, maintaining 8 accumulators (`c00..c13`) in hardware registers without spills on both x86-64 and ARM64.
   - Unrolling $K$ by 2 exposes independent FMAs to the Go compiler, hiding instruction pipeline latency.

##### Small MatMul Performance (`GOEXPERIMENT=""` on AMD 9950X3D):

| Shape ($M \times K \times N$) | Baseline (Pure Go) | Optimized (Pure Go) | Speedup |
| :--- | :--- | :--- | :--- |
| **Transposed $[128, 4] \times [1, 4]$** | 329.0 ns (3.11 GFlops/s) | **180.4 ns** (5.68 GFlops/s) | **1.82× faster** 🚀 |
| **Transposed $[128, 69] \times [4, 69]$** | 11,596 ns (6.09 GFlops/s) | **9,084 ns** (7.78 GFlops/s) | **1.28× faster** 🚀 |
| **Transposed $[25, 4] \times [1, 4]$** | 86.1 ns (2.32 GFlops/s) | **55.9 ns** (3.58 GFlops/s) | **1.54× faster** 🚀 |
| **Transposed $[25, 69] \times [4, 69]$** | 2,324 ns (5.94 GFlops/s) | **1,828 ns** (7.55 GFlops/s) | **1.27× faster** 🚀 |
| **NonTransposed $[128, 4] \times [4, 1]$** | 242.5 ns (4.22 GFlops/s) | **183.4 ns** (5.58 GFlops/s) | **1.32× faster** 🚀 |
| **NonTransposed $[128, 69] \times [69, 4]$** | 16,127 ns (4.38 GFlops/s) | **9,232 ns** (7.65 GFlops/s) | **1.75× faster** 🚀 |
| **NonTransposed $[25, 4] \times [4, 1]$** | 64.9 ns (3.08 GFlops/s) | **55.2 ns** (3.62 GFlops/s) | **1.17× faster** 🚀 |
| **NonTransposed $[25, 69] \times [69, 4]$** | 3,157 ns (4.37 GFlops/s) | **1,955 ns** (7.06 GFlops/s) | **1.61× faster** 🚀 |
| **End-to-End Adult-Demo $[128, 69] \times [69, 4]$** | 22.34 µs (3.16 GFlops/s) | **17.04 µs** (4.15 GFlops/s) | **+31.1% faster** 🚀 |

#### Key Architectural Decisions & Insights for Pure Go:

1. **Register Budget & Zero Stack Spills ($2 \times 4$ vs. $4 \times 4$)**:
   - On x86-64, the Go compiler allocates from 15 general-purpose scalar floating-point / XMM registers (`X0`–`X14`).
   - An experimental $4 \times 4$ microkernel required 16 accumulators + 4 LHS inputs + 4 RHS inputs + temporary calculation registers (>24 FP variables). This forced the Go compiler into heavy register spills (**1,341 `MOVSS ... (SP)` spill instructions**), dropping performance to ~60 GFlops/s.
   - The $2 \times 4$ geometry requires exactly **8 accumulators + 2 LHS + 4 RHS = 14 FP variables $\le 15$**. This fits 100% inside hardware registers with **zero stack spills** on x86-64, while ARM64 (32 vector/FP registers) accommodates it effortlessly.
2. **Constant-Offset 8-Step Unrolling**:
   - Inside the microkernel, the contracting loop is unrolled by 8 iterations using fixed constant offsets (`packedLHS[idxLhs+0..15]`, `packedRHS[idxRhs+0..31]`).
   - The base slice pointers are checked once before the loop (hoisting bounds checks), and indices advance once every 8 steps (`idxLhs += 16`, `idxRhs += 32`). This eliminated 8 pointer arithmetic instructions per iteration.
3. **Decoupled 2D Worker Partitioning (`choose2DSplit`)**:
   - In `choose2DSplit`, `targetRow` was previously bound to `LHSPanelCrossSize` ($M_c$). For No-SIMD, decoupling `targetRow := minRow` allows the scheduler to split work along rows across 32 or 64 worker threads even when $M_c$ is small ($M_c = 2$).
   - This prevents workers from unnecessarily splitting columns of the same row, completely eliminating multi-core cache-line false sharing during output accumulation.

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
| Data Type / Engine | $M_r$ (Rows) | $N_r$ (Cols) | $K_c$ (L1 Contracting) | $M_c$ (L2 Rows) | $N_c$ (L3 Cols) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Float32 (AVX-512)** | 8 | 32 | 192 | 32 | 512 |
| **Float16 (AVX-512)** | 8 | 32 | 128 | 32 | 768 |
| **BFloat16 (AVX-512)** | 8 | 32 | 128 | 32 | 768 |
| **Float64 (AVX-512)** | 8 | 16 | 64 | 16 | 256 |
| **Float32 (AVX2)** | 4 | 16 | 192 | 32 | 512 |
| **Portable Pure-Go (No-SIMD)** | 2 | 4 | 512 | 2 | 512 |

Cache parameters can be tuned or overridden via environment variables:
- AVX-512 / AVX2: `GOMLX_GO_SIMD_KC`, `GOMLX_GO_SIMD_MC`, `GOMLX_GO_SIMD_NC`
- No-SIMD: `GOMLX_GO_NOSIMD_KC`, `GOMLX_GO_NOSIMD_MC`, `GOMLX_GO_NOSIMD_NC`

---

## 10. File Map & Code Generation

Because `matmul` provides high performance across multiple architectures and data types, Go template generation is used to maintain symmetry:

| File | Purpose |
| :--- | :--- |
| `matmul.go` | Cache parameters, priority constants, and feature flags. |
| `avx512_router.go` | Routes between Small and Large AVX-512 kernels. |
| `avx512_small.go` | Dispatcher for AVX-512 small matrix multiplication across layouts. |
| `avx512_small_transposed.go` | Base template for AVX-512 small transposed matmul caller ($4 \times 4$ tiling). |
| `avx512_small_transposed_amd64.s` | Handwritten AVX-512 small transposed $4 \times 4$ assembly kernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx512_small_nontransposed.go` | Base template for AVX-512 small non-transposed matmul caller (GEMV direct, stack transposition, column tiling). |
| `avx512_small_nontransposed_amd64.s` | Handwritten AVX-512 small non-transposed $4 \times 64$ and $4 \times 16$ assembly kernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx512_large.go` | Base template for AVX-512 large matrix multiplication (Go SIMD + Assembly caller). |
| `avx512_large_amd64.go` | Assembly function forward declarations (`//go:noescape`). |
| `avx512_large_amd64_*.s` | Handwritten AVX-512 GEMM microkernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx512_pack_amd64_*.s` | Handwritten AVX-512 fast LHS transposition and packing kernels. |
| `avx512_pack_rhs_amd64.s` | Handwritten AVX-512 unrolled RHS strip packing kernel. |
| `avx2_router.go` | Routes between Small and Large AVX2 kernels. |
| `avx2_small.go` | Dispatcher for AVX2 small matrix multiplication across layouts. |
| `avx2_small_transposed.go` | Base template for AVX2 small transposed matmul caller ($4 \times 2$ and $4 \times 1$ tiling). |
| `avx2_small_transposed_amd64.s` | Handwritten AVX2 small transposed $4 \times 2$ and $4 \times 1$ assembly kernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx2_small_nontransposed.go` | Base template for AVX2 small non-transposed matmul caller (GEMV direct, stack transposition, column tiling). |
| `avx2_small_nontransposed_amd64.s` | Handwritten AVX2 small non-transposed $4 \times 16$ and $4 \times 8$ assembly kernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx2_large.go` | Base template for AVX2 large matrix multiplication (Go SIMD + Assembly caller). |
| `avx2_large_amd64.go` | AVX2 assembly function forward declarations (`//go:noescape`). |
| `avx2_large_amd64_*.s` | Handwritten AVX2 GEMM microkernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx2_pack_amd64_*.s` | Handwritten AVX2 fast LHS transposition and packing kernels. |
| `avx2_pack_rhs_amd64.s` | Handwritten AVX2 unrolled RHS strip packing kernel. |
| `avx2_*.go` | AVX2 (256-bit SIMD) router, small kernels, large kernels, and transpositions. |
| `nosimd.go` | Architecture-agnostic portable Go fallback router and 2D partitioner. |
| `nosimd_large.go` | Base template for No-SIMD large matrix multiplication (zero-copy direct output bypass, 8-step unrolled $2 \times 4$ microkernel). |
| `gen_*` | **Auto-generated files** created by `alternates_generator` for alternative dtypes (`f16`, `bf16`, `f64`, `half`). |

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

# Run All Large DotGeneral benchmarks:
go test -run none -bench BenchmarkCompliance/DotGeneral ./gobackend

# Run Subset Of Large DotGeneral benchmarks:
go test -run none -bench BenchmarkCompliance/DotGeneral/Large ./gobackend

# Disable AVX512, so AVX2 is used instead:
$ GOMLX_GO_SIMD_AVX512=0 go test -run none -bench Compliance/DotGeneral ./gobackend

# Test strictly without SIMD (portable pure Go):
$ GOEXPERIMENT="" nice -n -20 go test -v ./internal/gobackend/dot/matmul -run TestNoSIMD
$ GOEXPERIMENT="" nice -n -20 go test -v -run TestCompliance ./gobackend
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


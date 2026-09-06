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
                                                         Go SIMD    Assembly
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
  - $M_c \times K_c$ **LHS Panel**: Sized to fit comfortably in **L2 cache** (e.g., $32 \times 384$ for Float32).
  - $K_c \times N_c$ **RHS Panel**: Sized to fit in **L3 cache** (e.g., $384 \times 192$ for Float32).
  - $M_r \times N_r$ **Register Tile**: Kept entirely in CPU vector registers during the innermost microkernel loop ($4 \times 64$ for AVX-512 Float32).

---

## 3. Data Packing & Fast Transposition

Non-unit memory strides in row-major matrices cause CPU cache thrashing and prevent continuous vector loads. Packing reorders sub-matrices into cache-friendly sequential buffers.

### Pack LHS (`unsafePackLHS`)
- Takes $M_c$ rows and $K_c$ contracting columns from the LHS matrix.
- Reorganizes them into panels of $M_r = 4$ rows.
- Each 4-row strip is transposed into contiguous 4-element column vectors:
  $$\begin{bmatrix} L_{0,0} & L_{0,1} & L_{0,2} & \dots \\ L_{1,0} & L_{1,1} & L_{1,2} & \dots \\ L_{2,0} & L_{2,1} & L_{2,2} & \dots \\ L_{3,0} & L_{3,1} & L_{3,2} & \dots \end{bmatrix} \implies [L_{0,0}, L_{1,0}, L_{2,0}, L_{3,0}], [L_{0,1}, L_{1,1}, L_{2,1}, L_{3,1}], \dots$$
- **Why this layout?** In the microkernel, each 4-element strip represents the contracting values for the 4 active rows at step $k$. A single sequential read loads all 4 values to be broadcast across the accumulators.

### Fast Assembly Transposition vs. Go SIMD
Transposing 4 rows into 4-element columns is a performance-critical step:
- **Go 1.27.1 SIMD**: Only exposes permutation intrinsics like `archsimd.Permute2x256Float32x16` or `Permute4x64`. Transposing a $4 \times 16$ block required multiple stages of shuffles and permutations across lanes, creating significant instruction overhead.
- **Handwritten AVX-512 Assembly** (`avx512_pack_amd64_*.s`):
  - **Float32**: Uses direct interleaving unpack instructions (`VUNPCKLPS`, `VUNPCKHPS`) followed by 128-bit lane permutes (`VSHUFF32X4`).
  - **Float64**: Uses `VUNPCKLPD`, `VUNPCKHPD`, and `VSHUFF64X2`.
  - **Float16 / BFloat16**: Uses word-level unpacking (`VPUNPCKLWD`, `VPUNPCKHWD`) followed by `VPERMT2W`.
- **Result**: Packing time dropped by **60% to 75%** (a **2.5× to 4× speedup** in `PackLHS`), reducing packing overhead to only ~5% of the total matrix multiplication time.

### Pack RHS (`packRHS`)
- Slices $N_c$ columns into blocks of $N_r = 64$ columns.
- Stores each row of 64 elements sequentially.
- This allows the microkernel to load RHS rows directly into 4 full 512-bit vector registers (`4 × 16 = 64` floats) using unaligned vector loads (`VMOVDQU32`).

---

## 4. AVX-512 GEMM Microkernel Design

The innermost microkernel computes a block of $M_r = 4$ rows $\times N_r = 64$ columns over $K_c$ contracting steps.

```
       RHS Panel (64 columns -> 4 AVX-512 registers: Z20, Z21, Z22, Z23)
       +--------------------+--------------------+--------------------+--------------------+
       |   vec 0 (16 f32)   |   vec 1 (16 f32)   |   vec 2 (16 f32)   |   vec 3 (16 f32)   |
       +--------------------+--------------------+--------------------+--------------------+
LHS    |
Row 0  |  Z0  += Z16 * Z20   |  Z1  += Z16 * Z21   |  Z2  += Z16 * Z22   |  Z3  += Z16 * Z23
(Z16)  |
Row 1  |  Z4  += Z17 * Z20   |  Z5  += Z17 * Z21   |  Z6  += Z17 * Z22   |  Z7  += Z17 * Z23
(Z17)  |
Row 2  |  Z8  += Z18 * Z20   |  Z9  += Z18 * Z21   |  Z10 += Z18 * Z22   |  Z11 += Z18 * Z23
(Z18)  |
Row 3  |  Z12 += Z19 * Z20   |  Z13 += Z19 * Z21   |  Z14 += Z19 * Z22   |  Z15 += Z19 * Z23
(Z19)  +--------------------+--------------------+--------------------+--------------------+
```

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

### Overall Impact
- **Total Throughput Gain**: **+44.7% to +59.2%** (from 1,850 GFlops/s up to **~2.95 TFlops/s**).
- **Compute Efficiency**: The CPU profile shows compute math (`avx512LargeKernelFloat32Asm`) now accounts for **81.4%** of all CPU cycles, with output writeback reduced to just 4.3% (a single sequential memory write).

---

## 7. Exploration: Alternative Kernel Geometries (6x48 vs 4x64)

During optimization, we explored an alternative microkernel geometry for AVX-512 Float32: **6 rows × 48 cols** ($M_r = 6, N_r = 48$) compared to the default **4 rows × 64 cols** ($M_r = 4, N_r = 64$).

### Theoretical Motivation
* **Register Allocation**: AVX-512 has 32 registers (`Z0`–`Z31`).
  * In **4x64**: 16 accumulators ($4 \times 4$), 4 RHS vectors ($4 \times 16$), 4 LHS scalar broadcasts. 24 registers used, 8 spare. Each loaded RHS vector is reused across 4 FMAs. Arithmetic intensity: $\approx 1.88$ Flops/byte.
  * In **6x48**: 18 accumulators ($6 \times 3$), 3 RHS vectors ($3 \times 16$), 6 LHS scalar broadcasts. 27 registers used, 5 spare. Each loaded RHS vector is reused across 6 FMAs (+50% reuse). Arithmetic intensity: $\approx 2.67$ Flops/byte (+42%).
* In microkernel isolation, compute throughput was measured at **~350 GFlops/s** per core, and on whole matrices where dimensions were multiples of 48 (e.g. $N=384, 1536$ in `BAAI-bge-small`), end-to-end performance improved by **+20% to +30%** (jumping from 1,800 to 2,350 GFlops/s).

### Why 4x64 Remains the Default
Despite the higher arithmetic intensity, 6x48 was set aside in favor of 4x64 for universal workloads due to two critical issues:

1. **LHS Cache-Line Straddling (24 bytes vs 16 bytes)**:
   * In **4x64**: Each strip in `PackLHS` is 4 rows × 4 bytes = **16 bytes**. Exactly 4 strips make **64 bytes** (one CPU cache line). Memory writes during packing and broadcast loads during compute are perfectly cache-line aligned; no strip ever crosses a cache line.
   * In **6x48**: Each strip in `PackLHS` is 6 rows × 4 bytes = **24 bytes**. Because 24 does not divide 64, every 3rd strip crosses a 64-byte cache line boundary (e.g., bytes 48–71 span line 0 and line 1). This incurs CPU split-cache-line access penalties and prevents aligned vector packing stores.
2. **Dimension Multiples & Remainder Tails**:
   * Most deep learning models use dimensions that are powers of 2 or multiples of 64 ($N \in \{128, 256, 512, 1024, 2048, 4096\}$).
   * 64 divides all of these cleanly with 0 remainder.
   * 48 leaves fractional remainder tails on common sizes (e.g. $1024 = 21 \times 48 + 16$), creating tiny edge strips, uneven thread work distribution, and severe regressions on batched workloads (e.g. `Batched-Large-1` dropped from 2,770 to 2,007 GFlops/s).
3. **Cross-DType Complexity**:
   * Maintaining 6-row layouts would require dedicated AVX-512 transposition microkernels and packing logic across Float16, BFloat16, and Float64 for ambiguous overall returns.

*Conclusion*: 4x64 remains the standard architecture default. The 6x48 geometry can be revisited in specialized scenarios (e.g., dedicated fused layers with fixed multiples of 48).

---

## 8. File Map & Code Generation

Because `matmul` provides high performance across multiple architectures and data types, Go template generation is used to maintain symmetry:

| File | Purpose |
| :--- | :--- |
| `matmul.go` | Cache parameters, priority constants, and feature flags. |
| `avx512_router.go` | Routes between Small and Large AVX-512 kernels. |
| `avx512_large.go` | Base template for AVX-512 large matrix multiplication (Go SIMD + Assembly caller). |
| `avx512_large_amd64.go` | Assembly function forward declarations (`//go:noescape`). |
| `avx512_large_amd64_*.s` | Handwritten AVX-512 GEMM microkernels (`float32`, `float64`, `float16`, `bfloat16`). |
| `avx512_pack_amd64_*.s` | Handwritten AVX-512 fast transposition and packing kernels. |
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

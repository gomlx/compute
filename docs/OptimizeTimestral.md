# Optimizing Timestral (`28M` & `123M`) on Go Backend (ARM64 NEON/SME & AMD64 AVX2)

This document records the baseline benchmarks, CPU and per-op profiles, architectural analysis, and optimization plan for running [`mistralai/Timestral-28M`](https://huggingface.co/mistralai/Timestral-28M) and [`mistralai/Timestral-123M`](https://huggingface.co/mistralai/Timestral-123M) on the pure-Go backend (`internal/gobackend`) on both Apple Silicon (ARM64 NEON and SME, Sections 1–8) and x86-64 (Intel Core i9-12900K AVX2, Section 9).

It follows the evolution of our ARM64 NEON/SME architecture documented in [`internal/gobackend/simdarch/Neon.md`](../internal/gobackend/simdarch/Neon.md), extending our focus from `DotGeneral`/`FusedDense` to **non-MatMul operations** (`FusedScaledDotProductAttention`, `FusedActivation`/`SwiGLU`, `ReduceSum`, `Mul`/`Add` broadcasting, `Slice`/`Concatenate`, and scheduler/cache-locality interactions).

---

## 1. Benchmark Setup & Hardware Topology

### 1.1 Benchmark Commands

```bash
# Timestral-28M (d_model=384, n_heads=6, head_dim=64, d_ff=1024, 3 stages x 4 blocks)
GOEXPERIMENT=simd go run github.com/gomlx/go-huggingface/examples/timestral/ \
  -backend="go" -model mistralai/Timestral-28M -obs gomlx-simd -mmap=false -runs=10

# Timestral-123M (d_model=640, n_heads=10, head_dim=64, d_ff=1712, 5 stages x 4 blocks)
GOEXPERIMENT=simd go run github.com/gomlx/go-huggingface/examples/timestral/ \
  -backend="go" -model mistralai/Timestral-123M -obs gomlx-simd -mmap=false -runs=10
```

### 1.2 Workload Dimensions

Both models run zero-shot probabilistic time-series forecasting with:
- **Batch size (`B`)**: `2` variates (`[2, 12096]` input context $\to$ `[2, 864, 21]` output quantiles)
- **Encoder tokens (`T_enc`)**: `378` patches (`12096 / 32`)
- **Decoder tokens (`T_dec`)**: `27` horizon patches (`864 / 32`)
- **Total sequence length (`T_total`)**: `405` (`378 + 27`)
- **Architecture**: `N_stages` stages (`3` in `28M`, `5` in `123M`), each containing:
  - `4` Encoder blocks (`T_enc = 378`, causal + group boolean mask `[2, 378, 378]`)
  - `4` Decoder blocks (`T_dec = 27` attending to `T_total = 405`, forecast boolean mask `[2, 27, 405]`)
  - `4` Shared Feed-Forward (`SwiGLU`) blocks applied to concatenated `[2, 405, D]`
  - `1` Variate Attention block applied to transposed `[405, 2, D]` (`[405, 2, 2]` boolean mask) + `1` Variate Feed-Forward (`SwiGLU`) block

| Parameter | `Timestral-28M` | `Timestral-123M` |
| :--- | ---: | ---: |
| `d_model` (`D`) | `384` | `640` |
| `n_heads` (`H`) | `6` | `10` |
| `head_dim` (`D_h`) | `64` | `64` |
| `d_ff` (`SwiGLU` inner dim) | `1024` (`w12` output `2048`) | `1712` (`w12` output `3424`) |
| `n_stages` | `3` (`12` enc + `12` dec + `3` var) | `5` (`20` enc + `20` dec + `5` var) |
| Total Compiled Graph Ops | `1,636` | `2,590` |

### 1.3 Host Hardware Topology (12-Core Apple M6)

As detailed in [`internal/gobackend/simdarch/Neon.md` (Stage 10)](../internal/gobackend/simdarch/Neon.md), the 12-core Apple M6 has an **asymmetric 2-cluster L2 and SME topology**:
- **Cluster 1 (Performance Cluster — CPU IDs `6..11`)**:
  - **2 Super cores + 4 Performance cores** sharing a **20 MB L2 cache**.
  - **1 High-Throughput SME Unit** (`SVL = 512-bit`, $16 \times 16$ `float32` `ZA` tiles): **2,262.5 GFlops/s** single-thread throughput — **more FLOPs than all 12 NEON cores combined (~1,200 GFlops/s)**.
- **Cluster 0 (Efficiency Cluster — CPU IDs `0..5`)**:
  - **6 Efficiency cores** sharing a separate **8 MB L2 cache**.
  - **1 Low-Power SME Unit**: **385.6 GFlops/s** (`5.9×` slower than Cluster 1's SME unit).

---

## 2. Initial Baseline Benchmarks (Before Optimization)

### 2.1 End-to-End Baseline Summary

| Model | Configuration | Best Wall Time | CPU Time | Avg Cores | Speedup vs NEON-Only |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **`Timestral-28M`** | `gomlx-simd` (Default: SME + NEON, parallel graph) | **58.0 ms** | 558.8 ms | 9.6 | **1.43×** |
| **`Timestral-28M`** | `gomlx-simd-seq` (`-backend="go:ops_sequential"`) | **65.2 ms** | 548.8 ms | 8.4 | 1.27× |
| **`Timestral-28M`** | `gomlx-neon-only` (`GOMLX_GO_SIMD_SME=0`) | **82.7 ms** | 832.1 ms | 10.1 | 1.00× |
| **`Timestral-123M`** | `gomlx-simd` (Default: SME + NEON, parallel graph) | **179.4 ms** | 1,860.0 ms | 10.4 | **1.64×** |
| **`Timestral-123M`** | `gomlx-simd-seq` (`-backend="go:ops_sequential"`) | **192.6 ms** | 1,850.0 ms | 9.6 | 1.52× |
| **`Timestral-123M`** | `gomlx-neon-only` (`GOMLX_GO_SIMD_SME=0`) | **293.4 ms** | 3,170.0 ms | 10.8 | 1.00× |

### 2.2 Detailed Run-by-Run Baseline (`gomlx-simd`)

#### `mistralai/Timestral-28M` (`load + compile: 204.8ms, live 0.12 GB, max RSS 0.29 GB`)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Live Mem | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gomlx-simd` | `Timestral-28M` | warm-up | 70.1ms | 593.6ms | 8.5 | 1.91e-06 | 0.22 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 1 | 58.7ms | 562.9ms | 9.6 | 1.91e-06 | 0.24 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 2 | 58.4ms | 561.9ms | 9.6 | 1.91e-06 | 0.27 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 3 | 58.0ms | 558.8ms | 9.6 | 1.91e-06 | 0.29 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 4 | 60.4ms | 585.0ms | 9.7 | 1.91e-06 | 0.32 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 5 | 59.5ms | 569.7ms | 9.6 | 1.91e-06 | 0.34 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 6 | 65.7ms | 584.9ms | 8.9 | 1.91e-06 | 0.36 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 7 | 59.3ms | 575.5ms | 9.7 | 1.91e-06 | 0.39 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 8 | 59.8ms | 578.8ms | 9.7 | 1.91e-06 | 0.41 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 9 | 58.6ms | 565.5ms | 9.7 | 1.91e-06 | 0.38 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 10 | 58.9ms | 569.3ms | 9.7 | 1.91e-06 | 0.40 GB | 0.42 GB |
| **`gomlx-simd`** | **`Timestral-28M`** | **`best`** | **`58.0ms`** | **`558.8ms`** | **`9.6`** | **`1.91e-06`** | **`0.40 GB`** | **`0.42 GB`** |

#### `mistralai/Timestral-123M` (`load + compile: 323.8ms, live 0.50 GB, max RSS 1.10 GB`)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Live Mem | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gomlx-simd` | `Timestral-123M` | warm-up | 220.3ms | 1.9s | 8.7 | - | 1.07 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 1 | 182.4ms | 1.9s | 10.3 | - | 1.13 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 2 | 180.0ms | 1.9s | 10.4 | - | 1.19 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 3 | 180.4ms | 1.9s | 10.4 | - | 1.24 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 4 | 180.3ms | 1.9s | 10.4 | - | 1.30 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 5 | 186.9ms | 1.9s | 10.2 | - | 1.29 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 6 | 179.5ms | 1.9s | 10.4 | - | 1.29 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 7 | 180.1ms | 1.9s | 10.4 | - | 1.35 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 8 | 180.1ms | 1.9s | 10.4 | - | 1.35 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 9 | 179.4ms | 1.9s | 10.4 | - | 1.34 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 10 | 180.3ms | 1.9s | 10.4 | - | 1.33 GB | 1.35 GB |
| **`gomlx-simd`** | **`Timestral-123M`** | **`best`** | **`179.4ms`** | **`1.9s`** | **`10.4`** | **-** | **`1.33 GB`** | **`1.35 GB`** |

---

## 3. CPU Profile (`pprof`) & Per-Op Breakdown

### 3.1 CPU Profile (`go tool pprof`) Flat Top Functions

Sampling CPU profiles collected over 30 timed runs of `Timestral-28M` (`16.71s` total CPU samples) and 15 timed runs of `Timestral-123M` (`27.49s` total CPU samples) reveal a striking pattern:

| Function / Symbol | `28M` Flat CPU | `28M` % | `123M` Flat CPU | `123M` % | Category |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `runtime.pthread_cond_wait` | 8.11s | 48.5% | 12.15s | 44.2% | Worker / GC park & wakeup |
| `runtime.usleep` | 1.50s | 9.0% | 3.74s | 13.6% | Spin/sleep in `lock2` contention |
| `runtime.pthread_cond_signal` | 1.00s | 6.0% | 1.37s | 5.0% | Worker wakeup signaling |
| **`neon.sdpaSingleRowNEONGeneral`** | **2.22s** | **13.3%** | **4.02s** | **14.6%** | **SDPA 1-row `archsimd` fallback!** |
| **`neon.sdpaSoftmaxRowNEON`** | **0.98s** | **5.9%** | **1.42s** | **5.2%** | **SDPA row softmax (`archsimd`)** |
| `neon.packedMicroKernelFloat32Asm_4x16` | 0.90s | 5.4% | 2.13s | 7.8% | NEON workers in hybrid `FusedDense` |
| `activations.SwiGLUFloat32SIMD` | 0.14s | 0.8% | 0.24s | 0.9% | Portable 128-bit `archsimd` SwiGLU |
| `neon.smeLargeKernelFloat32Asm_32x32` | 0.10s | 0.6% | 0.13s | 0.5% | SME $32 \times 32$ `FMOPA` kernel |
| `neon.neonPackRHSFloat32Asm` | 0.08s | 0.5% | 0.18s | 0.7% | `FusedDense` RHS packing |
| `ops.BroadcastMulFloat32SIMD` | 0.07s | 0.4% | 0.09s | 0.3% | Elementwise broadcast `Mul` |
| `runtime.memmove` | 0.05s | 0.3% | 0.13s | 0.5% | `Slice` / `Concatenate` copies |

> [!IMPORTANT]
> **Key Observation from `pprof`**:
> Among actual compute functions (excluding thread-parking `pthread_cond_wait` / `usleep`), **`sdpaSingleRowNEONGeneral` + `sdpaSoftmaxRowNEON` consumes ~60%–65% of all active CPU time** (`3.20s` out of `5.2s` active CPU in `28M`; `5.44s` out of `8.6s` active CPU in `123M`), whereas `smeLargeKernelFloat32Asm_32x32` takes only `0.10s`–`0.13s` because SME is **only** used by `FusedDense` and is **completely bypassed** by `FusedScaledDotProductAttention`!

---

### 3.2 Exact Per-`OpType` Breakdown (`-backend="go:ops_sequential"`, Averaged Over 5 Timed Runs)

Using the built-in Go backend op profiler ([`internal/gobackend/profile.go`](../internal/gobackend/profile.go)) via `GOMLX_OP_PROFILE=1` combined with `-backend="go:ops_sequential"` eliminates overlapping graph op execution and measures the exact wall time, process CPU time, and average core utilization of every operation in a single forward pass:

```bash
GOEXPERIMENT=simd GOMLX_OP_PROFILE=5 go run github.com/gomlx/go-huggingface/examples/timestral/ \
  -backend="go:ops_sequential" -model mistralai/Timestral-28M -obs gomlx-simd-seq -mmap=false -runs=5 -verify=false
```

| OpType | `28M` Calls | `28M` Wall/run | `28M` Wall % | `28M` CPU/run | `28M` Cores | `123M` Calls | `123M` Wall/run | `123M` Wall % | `123M` CPU/run | `123M` Cores |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`FusedScaledDotProductAttention`** | 27 | **25.4 ms** | **39.0%** | **269.8 ms** | 10.6 | 45 | **69.8 ms** | **36.1%** | **753.1 ms** | 10.8 |
| **`FusedDense`** | 75 | **21.9 ms** | **33.6%** | **215.5 ms** | 9.8 | 121 | **85.1 ms** | **44.0%** | **923.4 ms** | 10.9 |
| **`Mul`** | 415 | **5.5 ms** | **8.4%** | **16.6 ms** | 3.0 | 529 | **12.2 ms** | **6.3%** | **44.8 ms** | 3.7 |
| **`FusedActivation` (`SwiGLU`)** | 14 | **2.8 ms** | **4.3%** | **26.4 ms** | 9.5 | 22 | **7.1 ms** | **3.7%** | **73.1 ms** | 10.2 |
| **`Slice`** | 218 | **2.5 ms** | **3.8%** | **5.9 ms** | 2.4 | 312 | **6.3 ms** | **3.3%** | **22.8 ms** | 3.6 |
| **`Add`** | 318 | **2.2 ms** | **3.3%** | **8.4 ms** | 3.9 | 370 | **4.7 ms** | **2.4%** | **21.4 ms** | 4.5 |
| **`Concatenate`** | 48 | **0.85 ms** | **1.3%** | **2.0 ms** | 2.3 | 72 | **1.9 ms** | **1.0%** | **5.9 ms** | 3.1 |
| **`Neg`** | 26 | **0.68 ms** | **1.0%** | **1.8 ms** | 2.6 | 42 | **1.5 ms** | **0.8%** | **5.1 ms** | 3.5 |
| **`ReduceSum`** | 84 | **0.50 ms** | **0.8%** | **0.59 ms** | **1.2** | 102 | **1.3 ms** | **0.7%** | **1.3 ms** | **1.0** |
| **`LogicalAnd`** | 3 | **0.38 ms** | **0.6%** | **0.39 ms** | 1.0 | 3 | **0.38 ms** | **0.2%** | **0.39 ms** | 1.0 |
| **`Transpose`** | 7 | **0.16 ms** | **0.2%** | **0.21 ms** | 1.3 | 11 | **0.33 ms** | **0.2%** | **0.68 ms** | 2.1 |
| **`Iota`** | 13 | **0.32 ms** | **0.5%** | **0.32 ms** | 1.0 | 13 | **0.32 ms** | **0.2%** | **0.33 ms** | 1.0 |
| **All Other Ops** (30+ types) | 388 | **2.11 ms** | **3.2%** | **3.30 ms** | ~1.5 | 948 | **2.27 ms** | **1.2%** | **3.50 ms** | ~1.5 |
| **Total (Sum of All Ops)** | **1,636** | **65.3 ms** | **100.0%** | **551.2 ms** | **8.4** | **2,590** | **193.2 ms** | **100.0%** | **1,855.8 ms** | **9.6** |

---

### 3.3 Top Individual Op Signatures by Wall Time

#### `Timestral-28M` (`d_model=384, n_heads=6, head_dim=64, d_ff=1024`)

| Rank | Op Signature | Calls/run | Wall/run | Wall/call | CPU/run | Avg Cores | Role in Timestral |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | `FusedScaledDotProductAttention([2, 378, 6, 64], ..., (Bool)[2, 378, 378])` | 12 | **21.6 ms** | 1.80 ms | 234.3 ms | 10.8 | Encoder Self-Attention |
| 2 | `FusedDense([2, 405, 384], [384, 2048]) -> [2, 405, 2048]` | 12 | **6.9 ms** | 577.9 µs | 74.9 ms | 10.8 | FFN `w12` (Gate+Up) |
| 3 | `FusedDense([2, 405, 384], [384, 384]) -> [2, 405, 384]` | 27 | **4.4 ms** | 163.6 µs | 37.0 ms | 8.4 | Attention `wO` + Gate |
| 4 | `FusedDense([2, 405, 384], [384, 1152]) -> [2, 405, 1152]` | 12 | **4.1 ms** | 345.2 µs | 40.9 ms | 9.9 | Attention `wQKV` |
| 5 | `FusedDense([2, 405, 1024], [1024, 384]) -> [2, 405, 384]` | 12 | **3.8 ms** | 316.3 µs | 39.0 ms | 10.3 | FFN `w3` (Down) |
| 6 | `FusedScaledDotProductAttention([2, 27, 6, 64], [2, 405, 6, 64], ..., (Bool)[2, 27, 405])` | 12 | **3.5 ms** | 291.1 µs | 33.0 ms | 9.4 | Decoder Cross/Self-Attn |
| 7 | `FusedActivation([2, 405, 2048]) -> [2, 405, 1024]` (`SwiGLU`) | 12 | **2.8 ms** | 231.2 µs | 26.4 ms | 9.5 | FFN `SwiGLU` |
| 8 | `Mul([2, 405, 6, 64], [2, 405, 1, 64]) -> [2, 405, 6, 64]` | 48 | **1.8 ms** | 38.0 µs | 6.4 ms | 3.5 | RoPE `x * cos` & `rot * sin` |
| 9 | `Mul([2, 405, 384], [2, 405, 384]) -> [2, 405, 384]` | 40 | **1.6 ms** | 38.8 µs | 5.5 ms | 3.6 | RMSNorm `x * x` + Gating |
| 10 | `Mul([2, 405, 384], [1, 1, 384]) -> [2, 405, 384]` | 28 | **1.1 ms** | 39.2 µs | 3.4 ms | 3.1 | RMSNorm `* scale` |
| 11 | `FusedDense([405, 2, 384], [384, 1152]) -> [405, 2, 1152]` | 3 | **1.0 ms** | 342.0 µs | 10.3 ms | 10.0 | Variate Attn `wQKV` |
| 12 | `Add([2, 405, 384], [2, 405, 384]) -> [2, 405, 384]` | 27 | **1.0 ms** | 37.7 µs | 5.0 ms | 4.9 | Residual `x + sublayer` |
| 13 | `Add([2, 405, 6, 64], [2, 405, 6, 64]) -> [2, 405, 6, 64]` | 24 | **0.92 ms** | 38.2 µs | 3.1 ms | 3.3 | RoPE `x*cos + rot*sin` |
| 14 | `Slice([2, 405, 3, 6, 64]) -> [2, 405, 1, 6, 64]` | 36 | **0.82 ms** | 22.9 µs | 2.4 ms | 3.0 | Split `wQKV` into `Q, K, V` |
| 15 | `Neg([2, 405, 6, 32]) -> [2, 405, 6, 32]` | 24 | **0.67 ms** | 27.8 µs | 1.8 ms | 2.7 | RoPE `-x2` |
| 16 | `Slice([2, 405, 6, 64]) -> [2, 405, 6, 32]` | 48 | **0.66 ms** | 13.7 µs | 1.2 ms | 1.9 | RoPE split `x1, x2` |
| 17 | `Slice([2, 405, 6, 64]) -> [2, 378, 6, 64]` | 36 | **0.63 ms** | 17.6 µs | 1.5 ms | 2.3 | Slice Encoder `Q, K, V` |
| 18 | `Mul([2, 405, 384], [2, 405, 1]) -> [2, 405, 384]` | 28 | **0.60 ms** | 21.6 µs | 0.61 ms | **1.0** | RMSNorm `x * invRMS` |
| 19 | `Concatenate([2, 405, 6, 32], [2, 405, 6, 32]) -> [2, 405, 6, 64]` | 24 | **0.52 ms** | 21.7 µs | 1.4 ms | 2.7 | RoPE `Concat(-x2, x1)` |
| 20 | `ReduceSum([2, 405, 384]) -> [2, 405]` | 28 | **0.47 ms** | 16.8 µs | 0.55 ms | **1.2** | RMSNorm sum of squares |

#### `Timestral-123M` (`d_model=640, n_heads=10, head_dim=64, d_ff=1712`)

| Rank | Op Signature | Calls/run | Wall/run | Wall/call | CPU/run | Avg Cores | Role in Timestral |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | `FusedScaledDotProductAttention([2, 378, 10, 64], ..., (Bool)[2, 378, 378])` | 20 | **59.8 ms** | 3.00 ms | 652.1 ms | 10.9 | Encoder Self-Attention |
| 2 | `FusedDense([2, 405, 640], [640, 3424]) -> [2, 405, 3424]` | 20 | **28.5 ms** | 1.40 ms | 314.2 ms | 11.0 | FFN `w12` (Gate+Up) |
| 3 | `FusedDense([2, 405, 1712], [1712, 640]) -> [2, 405, 640]` | 20 | **15.8 ms** | 787.6 µs | 173.4 ms | 11.0 | FFN `w3` (Down) |
| 4 | `FusedDense([2, 405, 640], [640, 1920]) -> [2, 405, 1920]` | 20 | **15.6 ms** | 780.2 µs | 170.3 ms | 10.9 | Attention `wQKV` |
| 5 | `FusedDense([2, 405, 640], [640, 640]) -> [2, 405, 640]` | 45 | **13.5 ms** | 300.6 µs | 140.4 ms | 10.4 | Attention `wO` + Gate |
| 6 | `FusedScaledDotProductAttention([2, 27, 10, 64], [2, 405, 10, 64], ..., (Bool)[2, 27, 405])` | 20 | **9.1 ms** | 456.8 µs | 93.1 ms | 10.2 | Decoder Cross/Self-Attn |
| 7 | `FusedActivation([2, 405, 3424]) -> [2, 405, 1712]` (`SwiGLU`) | 20 | **7.1 ms** | 355.0 µs | 72.5 ms | 10.2 | FFN `SwiGLU` |
| 8 | `Mul([2, 405, 10, 64], [2, 405, 1, 64]) -> [2, 405, 10, 64]` | 80 | **4.2 ms** | 51.9 µs | 16.5 ms | 4.0 | RoPE `x * cos` & `rot * sin` |
| 9 | `Mul([2, 405, 640], [2, 405, 640]) -> [2, 405, 640]` | 66 | **3.5 ms** | 53.7 µs | 15.1 ms | 4.3 | RMSNorm `x * x` + Gating |
| 10 | `FusedDense([405, 2, 640], [640, 3424]) -> [405, 2, 3424]` | 5 | **3.4 ms** | 685.6 µs | 36.6 ms | 10.7 | Variate FFN `w12` |
| 11 | `FusedDense([405, 2, 640], [640, 1920]) -> [405, 2, 1920]` | 5 | **3.4 ms** | 674.9 µs | 36.2 ms | 10.7 | Variate Attn `wQKV` |
| 12 | `Mul([2, 405, 640], [1, 1, 640]) -> [2, 405, 640]` | 46 | **2.4 ms** | 52.8 µs | 10.2 ms | 4.2 | RMSNorm `* scale` |
| 13 | `Add([2, 405, 640], [2, 405, 640]) -> [2, 405, 640]` | 45 | **2.2 ms** | 49.0 µs | 11.4 ms | 5.2 | Residual `x + sublayer` |
| 14 | `Add([2, 405, 10, 64], [2, 405, 10, 64]) -> [2, 405, 10, 64]` | 40 | **2.1 ms** | 52.2 µs | 8.7 ms | 4.2 | RoPE `x*cos + rot*sin` |
| 15 | `Slice([2, 405, 3, 10, 64]) -> [2, 405, 1, 10, 64]` | 60 | **1.9 ms** | 32.3 µs | 7.3 ms | 3.8 | Split `wQKV` into `Q, K, V` |
| 16 | `Mul([2, 405, 640], [2, 405, 1]) -> [2, 405, 640]` | 46 | **1.7 ms** | 36.6 µs | 1.7 ms | **1.0** | RMSNorm `x * invRMS` |
| 17 | `Slice([2, 405, 10, 64]) -> [2, 405, 10, 32]` | 80 | **1.7 ms** | 21.0 µs | 5.2 ms | 3.1 | RoPE split `x1, x2` |
| 18 | `Slice([2, 405, 10, 64]) -> [2, 378, 10, 64]` | 60 | **1.6 ms** | 26.8 µs | 6.8 ms | 4.2 | Slice Encoder `Q, K, V` |
| 19 | `FusedDense([405, 2, 1712], [1712, 640]) -> [405, 2, 640]` | 5 | **1.6 ms** | 316.2 µs | 16.4 ms | 10.4 | Variate FFN `w3` |
| 20 | `Neg([2, 405, 10, 32]) -> [2, 405, 10, 32]` | 40 | **1.5 ms** | 37.3 µs | 5.1 ms | 3.5 | RoPE `-x2` |
| 21 | `Concatenate([2, 405, 10, 32], [2, 405, 10, 32]) -> [2, 405, 10, 64]` | 40 | **1.3 ms** | 33.4 µs | 4.6 ms | 3.4 | RoPE `Concat(-x2, x1)` |
| 22 | `ReduceSum([2, 405, 640]) -> [2, 405]` | 46 | **1.3 ms** | 27.6 µs | 1.3 ms | **1.0** | RMSNorm sum of squares |

---

## 4. Architectural Analysis: SME vs. NEON for Non-`DotGeneral` Ops

On the 12-core Apple M6, non-`DotGeneral` operations fall into two distinct hardware regimes:

### Regime A: 2D Outer-Product / Matrix-Like Fused Operations (`FusedScaledDotProductAttention`)
- **Hardware Capabilities**:
  - Cluster 1's SME unit executes $32 \times 32$ (`4` `ZA` tiles of $16 \times 16$) `FMOPA` outer products at **2,262.5 GFlops/s** on a single thread.
  - All 12 NEON cores combined achieve ~`900–1,200 GFlops/s` with $4 \times 16$ microkernels (`packedMicroKernelFloat32Asm_4x16`), and drop to **< 150 GFlops/s** when running the 1-row `archsimd` fallback (`sdpaSingleRowNEONGeneral`).
- **Why SDPA Is Currently Slow in Timestral**:
  - Each Encoder SDPA call (`[2, 378, H, 64]`) performs $2 \times H$ attention heads of `[378, 64] x [64, 378] -> [378, 378]` ($Q K^T$) followed by `Softmax([378, 378])` and `[378, 378] x [378, 64] -> [378, 64]` ($\text{Probs} \times V$).
  - For `Timestral-28M` (`H=6`), one Encoder SDPA call is only **$43.9\text{ MFLOPs}$** ($21.9\text{ MFLOPs}$ for the lower-triangular unmasked region). Yet it takes **1.80 ms** per call (`24 GFlops/s` effective throughput across 10.8 cores!) — **`90×` below Cluster 1's SME peak**.
  - Even on a single thread (no multi-core fan-out at all!), Cluster 1's SME unit can execute all $43.9\text{ MFLOPs}$ in **~25–40 µs** of pure `FMOPA` compute, or **~150–250 µs** including packing and row-wise softmax — a **`7×` to `10×` speedup** over the current `1.80 ms` (`28M`) / `3.00 ms` (`123M`).
  - Furthermore, because `batch * numHeads` is `12` (`28M`) or `20` (`123M`), we can either:
    1. Run **single-threaded SME on Cluster 1** (or 2 SME workers across Clusters 1 & 0) with zero fine-grained sub-head splitting, or
    2. Use **Hybrid SME (Cluster 1) + 4-Row NEON Assembly (`sdpaNEON4RowsFloat32`)** across heads, where each head (`[378, 64]`) is a self-contained unit of work (`K` and `V` packed once per `(batch, kvHead)`).

### Regime B: 1D Vector / Streaming Operations (`SwiGLU`, `RMSNorm`, `RoPE`, `Mul`, `Add`, `ReduceSum`)
- **Can SME Accelerate 1D Elementwise / Reduction Ops?**
  - In **Streaming SVE mode (`SMSTART SM`)**, SME provides 32 vector registers `Z0`–`Z31` of width **`SVL = 512 bits` (16 `float32` per register)** — **$4\times$ wider** than standard 128-bit NEON (`V0`–`V31`, 4 `float32` per register).
  - Standard SVE/SVE2 instructions (`FADD`, `FMUL`, `FMLA`, `FMAX`, `FSQRT`, `FRSQRT`, `FADDV` horizontal reduction) are legal in Streaming SVE mode on Apple M4/M5/M6!
  - However, entering/exiting Streaming SVE mode (`SMSTART SM` / `SMSTOP SM`) requires `runtime.LockOSThread()`, `simdarch.SMEAcquireCluster()`, and ~100–150 ns of mode-switch latency, and 1D operations are **L2/DRAM bandwidth-bound** rather than FLOP-bound.
- **When Single-Thread SME Streaming SVE Beats Multi-Core NEON**:
  - Look at the tensor sizes in Timestral:
    - `28M` hidden state `[2, 405, 384]` = **`311,040` `float32` (`1.24 MB`)**.
    - `123M` hidden state `[2, 405, 640]` = **`518,400` `float32` (`2.07 MB`)**.
  - Both fit effortlessly inside **Cluster 1's 20 MB L2 cache** (~`300–400 GB/s` L2 bandwidth).
  - Currently, `dispatchBinarySIMD` (`minParallelElements = 16,384`) fans out every `1.24 MB` / `2.07 MB` `Mul`, `Add`, `Neg`, `Slice`, and `Concatenate` across **all 12 cores** (8–12 goroutine tasks via `workers.Parallelize`).
  - As a result:
    1. **Cross-Cluster L2 Cache Thrashing**: Half the chunks run on Cluster 0 (E-cores, separate 8 MB L2) and half on Cluster 1 (P-cores, 20 MB L2). Every op invalidates cache lines in the other cluster's L2 cache, forcing inter-cluster coherency traffic on the very next op (`400+` times per forward pass!).
    2. **Goroutine Wakeup & Condvar Overhead**: Each `Mul` or `Add` takes **`38–52 µs` wall time** (`3.5` average cores), spending more time in `pthread_cond_wait` / `pthread_cond_signal` / `runtime.lock2` than in actual SIMD loads/stores (`1.24 MB` at `150 GB/s` takes only **`8 µs`**!).

---

## 5. Ranked Optimization Plan

### Priority 1 (Critical — **39.0% of `28M` / 36.1% of `123M` Wall Time**): Unlock SME & 4-Row NEON Assembly in `FusedScaledDotProductAttention` with `booleanMask`

#### Root Causes Identified in [`internal/gobackend/dot/matmul/neon/sdpa_arm64.go`](../internal/gobackend/dot/matmul/neon/sdpa_arm64.go) and [`internal/gobackend/fusedops/sdpa.go`](../internal/gobackend/fusedops/sdpa.go)

1. **`len(booleanMask) == 0` Gate Completely Disables SME and 4-Row NEON Assembly**:
   - In [`sdpa_arm64.go:51`](../internal/gobackend/dot/matmul/neon/sdpa_arm64.go#L51):
     ```go
     canBatchQueries := !causal && len(booleanMask) == 0 && kvLenUnmaskedBase > 0 && NEONUseAsm
     ```
   - Because Timestral passes boolean attention masks (`(Bool)[2, 378, 378]` in Encoder, `(Bool)[2, 27, 405]` in Decoder, `(Bool)[405, 2, 2]` in Variate), `len(booleanMask) > 0` is **always true**.
   - Consequently:
     - `useSME` is **always `false`** (`sdpaSME32RowsFloat32` is never called).
     - `sdpaNEON4RowsFloat32` ($4 \times 4$ / $4 \times 16$ NEON assembly) is **never called**.
     - Because `headDim == 64` (not `32`), it also skips `sdpaDotNEONDim32Single` and falls all the way back to the 1-row portable Go `archsimd` loop `sdpaSingleRowNEONGeneral`!
2. **Decoder Attention (`qLen = 27`) Blocked by `qValidCount >= 32`**:
   - Even if `canBatchQueries` were true, `sdpaSME32RowsFloat32` requires `qValidCount >= 32` (`sdpa_arm64.go:75`).
   - In Timestral Decoder Self/Cross-Attention (`[2, 27, H, 64]` attending to `[2, 405, H, 64]`), `qLen = 27 < 32`.
   - Yet `smeLargeKernelFloat32Asm_32x32` **already accepts arbitrary `lhsActiveRows` (`1..32`) and `rhsActiveCols` (`1..32`)**! Allowing `sdpaSME32RowsFloat32` to handle tail blocks (or `qLen >= 8`) with `activeRows = min(32, qValidCount)` enables the entire 27-row Decoder attention head to execute in **a single $32 \times 32$ SME pass**!
3. **Redundant $K$ and $V$ Packing & Excessive Cluster Lock Churn**:
   - Currently, `sdpaSME32RowsFloat32` packs the entire $K$ (`[kvLen, headDim]`) and $V$ (`[kvLen, headDim]`) matrices **inside the 32-query-row function**, repeating $K$ and $V$ packing **12 times per head** for `qLen = 378` (`378 / 32 = 12`).
   - Furthermore, `sdpaMultiHeadGeneric` (`sdpa.go:975`) splits a single head (`seqLen = 378`) into `qBlockSize = 76` sub-chunks across workers, so each sub-chunk acquires/releases the SME cluster lock and re-packs $K$ and $V$.
   - In addition, `sdpaSME32RowsFloat32` currently uses scalar/NEON `neonPackLHS` and `neonPackRHSNonTransposed` instead of our hardware in-SME $16 \times 16$ `ZA` transpose packer (`smePackLHS32Float32Asm`).

#### Proposed Implementation for Priority 1
1. **Support `booleanMask` and `causal` in `sdpaSME32RowsFloat32` and `sdpaNEON4RowsFloat32`**:
   - Remove `!causal && len(booleanMask) == 0` from `canBatchQueries`.
   - For each block of `blockRows` (`1..32` in SME, `4` in NEON) queries `[qBase .. qBase+blockRows)`:
     - **Active Key Trimming (`activeKVLen`)**: Compute the maximum unmasked key index across the `blockRows` queries in the block. For Encoder causal/group masks (`[2, 378, 378]`), row `q` is lower-triangular (`k <= q`), so `activeKVLen = qBase + blockRows`. This **cuts $Q K^T$ and $\text{Probs} \times V$ FLOPs and $K/V$ packing in half ($0.5 \times 378^2$)**!
     - **Masked Row Softmax**: In `sdpaScaleMaskSoftmaxRowNEON`, accept `boolMaskRow []bool` (and `causalLimit int`). Apply `scale`, set masked positions (`!boolMaskRow[k]` or `k >= causalLimit`) to `-Inf` in the SIMD `rowMax` pass, and zero out `scores[k]` in the `exp` pass (with all-masked row guard).
2. **Support Partial Query Blocks (`activeRows = 1..32`) in `sdpaSME32RowsFloat32`**:
   - Allow `sdpaSME32RowsFloat32` to process both full 32-row blocks and the tail block (`1..31` rows, including Decoder `qLen = 27` and Encoder tail `378 % 32 = 26` rows!) using `smeLargeKernelFloat32Asm_32x32`'s `lhsActiveRows` parameter.
3. **Pack $K$ and $V$ Once Per Head (or KV-Head Group) Using `smePackLHS32Float32Asm`**:
   - Process an entire query head (`qLen = 378` or `qLen = 27`, or GQA group) inside a single `DispatchSDPAARM64` call so the SME cluster lock is acquired **once per head** (not per 76-row sub-chunk) and $K$ / $V$ tiles are packed incrementally at most **once per head** as `activeKVLen` grows!
- **Expected Impact**:
  - `Timestral-28M` SDPA: **`25.4 ms` $\to$ `~3.5–5.0 ms`** (saving **`~20 ms/run`**).
  - `Timestral-123M` SDPA: **`69.8 ms` $\to$ `~10.0–13.0 ms`** (saving **`~57 ms/run`**).

---

### Priority 2 (High — **4.3% of `28M` / 3.7% of `123M` Wall Time**): Handwritten ARM64 NEON / SME-Streaming Assembly for `SwiGLU` (`FusedActivation`)

#### Root Cause in [`internal/gobackend/activations/`](../internal/gobackend/activations)
- Look at [`internal/gobackend/activations/neon/neon_arm64.go`](../internal/gobackend/activations/neon/neon_arm64.go):
  - `Silu`, `GeluExact`, `GeluApprox`, `Tanh`, and `Sigmoid` all have handwritten ARM64 NEON assembly kernels in `neon_arm64.s`.
  - **`SwiGLU` has NO assembly kernel in `activations/neon/`!** It falls back to [`SwiGLUFloat32SIMD`](../internal/gobackend/activations/activations_simd.go#L418) using 128-bit Go `archsimd` (`Float32x4`), which processes only 4 floats per iteration and splits every row in two (`gate = x[:half]`, `up = x[half:]`) with Go loop/bounds overhead.
- In `Timestral-123M`, `22× FusedActivation(SwiGLU, [2, 405, 3424] -> [2, 405, 1712])` takes **`7.1 ms` wall / `73.1 ms` CPU** (`2.8 ms` wall / `26.4 ms` CPU in `28M`).

#### Proposed Implementation for Priority 2
- Add a handwritten ARM64 assembly kernel `swigluFloat32NEONAsm(in, out unsafe.Pointer, outerDim, halfDim int)` in [`internal/gobackend/activations/neon/`](../internal/gobackend/activations/neon) (or an SME Streaming-SVE 512-bit kernel on Cluster 1):
  - Process `16` floats (`4×` `V` registers of `gate` and `4×` `V` registers of `up`, or `1×` 512-bit `Z` register in SME streaming mode) per inner loop iteration with interleaved Cephes polynomial `exp(-gate)` + `FMLA`, operating across all `outerDim = 810` rows inside a single assembly call per worker chunk.
- **Expected Impact**:
  - `Timestral-28M`: **`2.8 ms` $\to$ `~0.8 ms`** (saving **`~2.0 ms/run`**).
  - `Timestral-123M`: **`7.1 ms` $\to$ `~2.0 ms`** (saving **`~5.1 ms/run`**).

---

### Priority 3 (High — **~8.0 ms in `123M` / ~3.8 ms in `28M`**): Fix Single-Threaded `RMSNorm` Sub-Ops (`Mul` Trailing-1 Broadcast & `ReduceSum`) or Fuse `RMSNorm`

#### Root Causes Identified
In [`models/timestral/layers.go:23-45`](../../go-huggingface/models/timestral/layers.go#L23-L45), `RMSNorm` is executed **`28×` per run in `28M`** and **`46×` per run in `123M`** as 5 separate graph operations:
1. `Mul(x, x)` (`[2, 405, D]`): `1.6 ms` (`28M`) / `3.5 ms` (`123M`)
2. `ReduceSum([2, 405, D] -> [2, 405])`: **`0.47 ms` (`28M`) / `1.3 ms` (`123M`) — running on `1.0` core!**
   - Why does `ReduceSum([2, 405, 640])` (`518,400` floats = `2.07 MB`) run on a single core?
   - In [`internal/gobackend/ops/reduce_simd.go:539`](../internal/gobackend/ops/reduce_simd.go#L539), `registerTrailingReduction[float32]` checks `if fn := trailingArchFullFloat32[opType]; fn != nil { fn(operandFlat, outputFlat, outerSize, reduceSize); return }` **before** the parallelization logic! And `DispatchTrailingSumNEON` in [`internal/gobackend/ops/neon/reduce_arm64.go:22`](../internal/gobackend/ops/neon/reduce_arm64.go#L22) runs a single-threaded loop `for o := range outerSize`!
3. `MulScalar` + `AddScalar` + `Rsqrt` (`[2, 405, 1]`): `~0.15 ms`
4. `Mul([2, 405, D], [2, 405, 1])`: **`0.60 ms` (`28M`) / `1.7 ms` (`123M`) — running on `1.0` core!**
   - Why does `Mul([2, 405, 640], [2, 405, 1])` (`518,400` floats) run on a single core (`Avg Cores = 1.0`)?
   - Because shape `[2, 405, 640]` vs `[2, 405, 1]` has `BroadcastOther` (neither `BroadcastSameShape` nor `BroadcastPrefix` nor `BroadcastSuffix`, because the trailing dimension is `1`, making it a **prefix/outer broadcast** `[810, 640]` $\times$ `[810, 1]` where each row of `640` floats is multiplied by a scalar!).
   - In [`internal/gobackend/ops/binary_ops_simd.go`](../internal/gobackend/ops/binary_ops_simd.go), `BroadcastOther` falls through to a **single-threaded** recursive stride walker (`execBinaryRecursiveSIMD`)!
5. `Mul([2, 405, D], [1, 1, D])`: `1.1 ms` (`28M`) / `2.4 ms` (`123M`).

#### Proposed Implementation for Priority 3
1. **Add `BroadcastOuter` (`[Outer, Inner]` op `[Outer, 1]`) Fast Path in `binary_ops_simd.go`**:
   - Detect when `lhs` can be collapsed to `[Outer, Inner]` and `rhs` to `[Outer, 1]` (or vice versa), where each `Inner` contiguous slice is multiplied/added by `rhs[o]` using `archsimd.BroadcastFloat32x4(rhs[o])` (or NEON assembly), and parallelize over `Outer` when `Outer * Inner >= minParallelElements` (or run on Cluster 1)!
2. **Parallelize ` trailingArchFullFloat32` (`DispatchTrailingSumNEON` / `Max` / `Min`) Over `outerSize`**:
   - In `reduce_simd.go`, instead of bypassing the parallelizer when `trailingArchFullFloat32[opType] != nil`, use `trailingArchFullFloat32` as the per-chunk range kernel (`fn(operandFlat[start*reduceSize:end*reduceSize], outputFlat[start:end], end-start, reduceSize)`) when `totalElements >= minReduceTrailingParallelElements`!
3. **(Optional / Additional) Fused RMSNorm (`FusedLayerNorm` with `center=false` or `FusedRMSNorm`)**:
   - A single fused pass over `x[o, 0..D)` computes `sum(x^2)` in registers (`x` stays hot in L1 cache) and immediately writes `x[o, i] * rsqrt(mean(x^2) + eps) * scale[i]`, eliminating **4 full-tensor reads/writes (`8.3 MB` per call $\times 46 = 380\text{ MB}$ per forward pass)** and **184 graph op dispatches**!
- **Expected Impact**:
  - Fixing `BroadcastOuter` + `ReduceSum` alone saves **`~0.8 ms` on `28M`** and **`~2.3 ms` on `123M`**.
  - Fusing `RMSNorm` saves **`~3.0 ms` on `28M`** and **`~7.5 ms` on `123M`**.

---

### Priority 4 (Medium-High — **~10.8 ms in `123M` / ~4.5 ms in `28M`**): Optimize Rotary Position Embeddings (`applyRotary`) & Strided `Slice`/`Concatenate`

#### Root Causes Identified
In [`models/timestral/layers.go:110-125`](../../go-huggingface/models/timestral/layers.go#L110-L125), `applyRotary` is called **`24×` per run in `28M`** and **`40×` per run in `123M`** on `[2, 405, H, 64]`:
```go
x1 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, 32))
x2 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(32, 64))
rotX := Concatenate([]*Node{Neg(x2), x1}, -1)
return Add(Mul(x, cos), Mul(rotX, sin))
```
Each call executes **6 separate graph operations** over `[2, 405, H, 64]`:
- `2× Slice([2, 405, H, 64] -> [2, 405, H, 32])` (`1.7 ms` in `123M`)
- `1× Neg([2, 405, H, 32])` (`1.5 ms` in `123M`)
- `1× Concatenate([2, 405, H, 32], [2, 405, H, 32] -> [2, 405, H, 64])` (`1.3 ms` in `123M`)
- `2× Mul([2, 405, H, 64], [2, 405, 1, 64])` (`4.2 ms` in `123M`) — note that `[2, 405, H, 64]` $\times$ `[2, 405, 1, 64]` is `BroadcastOther` (`[810, H, 64]` $\times$ `[810, 1, 64]`), which also hits the recursive stride walker `execBinaryRecursiveSIMD`!
- `1× Add([2, 405, H, 64], [2, 405, H, 64])` (`2.1 ms` in `123M`)

#### Proposed Implementation for Priority 4
1. **Optimize 3D Broadcast `[Outer, Middle, Inner]` op `[Outer, 1, Inner]` in `binary_ops_simd.go`**:
   - In `Mul([2, 405, H, 64], [2, 405, 1, 64])`, `Outer = 810`, `Middle = H` (`6` or `10`), `Inner = 64`.
   - Instead of recursive per-axis function calls, load `rhs[o, 0..64)` once into SIMD registers (`4× Float32x4` for `Inner == 64`!) and multiply across all `H` heads in `lhs[o, h, 0..64)` in a tight register-cached loop!
2. **Optimize Contiguous Inner-Block `Slice` and `Concatenate`**:
   - `Slice([2, 405, 3, H, 64] -> [2, 405, 1, H, 64])` (`1.9 ms` in `123M`) and `Slice([2, 405, H, 64] -> [2, 405, H, 32])` (`1.7 ms` in `123M`) copy fixed-size contiguous inner blocks (`H*64` or `32` floats = `128–2560 bytes`) for `810` outer iterations. A specialized 2D strided-row copy loop (`outerCount`, `srcStride`, `copyBytes`) avoids recursive N-dimensional index calculation on every `32`-float slice.
- **Expected Impact**:
  - `Timestral-28M`: saves **`~2.0–2.5 ms/run`**.
  - `Timestral-123M`: saves **`~5.0–6.5 ms/run`**.

---

### Priority 5 (Medium — **~3 ms in `28M` / ~6 ms in `123M`**): L2-Cache-Aware Parallelism Granularity on Asymmetric Apple Silicon

#### Root Cause Identified in `pprof` and `GOMLX_OP_PROFILE`
- Look at the elementwise ops in `Timestral-28M` (`311,040` floats = `1.24 MB`) and `Timestral-123M` (`518,400` floats = `2.07 MB`):
  - `minParallelElements = 16,384` (`64 KB`) and `targetChunkElements = 65,536` (`256 KB`) splits a `1.24 MB` tensor into **6–8 tasks** and a `2.07 MB` tensor into **8–12 tasks**.
  - Each task does only **1–2 µs of SIMD work**, while waking up 8–12 worker goroutines across **both Cluster 1 (P-cores, 20 MB L2) and Cluster 0 (E-cores, 8 MB L2)** takes **20–35 µs** of condvar/channel synchronization and pulls cache lines out of Cluster 1's 20 MB L2 into Cluster 0's 8 MB L2!
  - Notice that in `Timestral-28M` and `Timestral-123M`, all elementwise `Mul`, `Add`, `Neg`, `Slice`, and `Concatenate` ops achieve only **`2.4–4.5` average cores** despite spawning 8–12 tasks, and `pprof` shows **`63%` of all CPU samples** in `pthread_cond_wait` + `usleep` + `pthread_cond_signal`!
- Raising `targetChunkElements` (or capping the number of parallel chunks for memory-bound elementwise ops `< 4 MB` to at most `4–6` workers so they stay on Cluster 1's P-cores and inside Cluster 1's 20 MB L2 cache) dramatically reduces condvar wakeup overhead and cross-cluster L2 cache invalidations.

---

## 6. Summary of Optimization Roadmap & Target Performance

| Priority | Optimization Target | Files Affected | `28M` Baseline | `28M` Actual | `123M` Baseline | `123M` Actual | Status |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| **P1** | **`FusedScaledDotProductAttention`**: Enable SME (`4..32` active rows) & 4-Row NEON Asm (`1..4` rows) with `booleanMask` + `causal`, active-key trimming (`sdpaActiveKVLenForBlock`), lazy once-per-head `smePackLHS32Float32Asm` $K$ packing, fused single-`SMSTART` `smeDirectStridedFloat32Asm_32x32` ($\text{Probs} \times V$), and fine-grained SME lock release during NEON Softmax | `internal/gobackend/dot/matmul/neon/sdpa_arm64.go`, `internal/gobackend/dot/matmul/neon/sme_arm64.{go,s}`, `internal/gobackend/fusedops/sdpa.go` | 25.4 ms | **7.5 ms (`3.39×` faster)** | 69.8 ms | **23.1 ms (`3.02×` faster)** | **Done** |
| **P2** | **`FusedActivation` (`SwiGLU`)**: Handwritten ARM64 NEON assembly kernel (`swigluFloat32NEONAsm`, `4×` 128-bit `float32x4` unrolled per iteration with overlapped `FDIV` + `FRECPE`/`FRECPS` fast reciprocal) + `float64`/`bfloat16`/`float16` NEON dispatchers | `internal/gobackend/activations/neon/swiglu.{go,s}` | 2.8 ms (`6.6 ms` seq) | **1.2 ms (`5.5×` faster vs seq)** | 7.1 ms (`16.5 ms` seq) | **3.6 ms (`4.58×` faster vs seq)** | **Done** |
| **P3** | **`RMSNorm` Sub-Ops & Buffer Pooling**: Parallelize `ReduceTrailing` (`dispatchReduceSumSIMD`) and `BroadcastTrailing` (`tryBinaryTrailingArch`) over outer rows `A`, plus LIFO `bufferPool` (`buffers.go`) eliminating per-`P` `sync.Pool` stranding & GC pool wipes | `internal/gobackend/ops/reduce_simd.go`, `internal/gobackend/ops/binaryops_simd.go`, `internal/gobackend/buffers.go` | 2.9 ms (`ReduceSum`+`Mul[..,1]`) | **0.96 ms (`3.0×` faster)** | 11.2 ms (`ReduceSum`+`Mul[..,1]`) | **4.1 ms (`2.73×` faster)** | **Done** |
| **P4** | **RoPE & QKV Slicing**: Fast `[Outer, Middle, Inner]` $\times$ `[Outer, 1, Inner]` broadcast + 2D strided-block `Slice`/`Concatenate` | `internal/gobackend/ops/binaryops_simd.go`, `internal/gobackend/ops/slice.go` | 5.3 ms | ~2.8 ms | 12.7 ms | ~6.5 ms | Planned |
| **P5** | **L2-Cache-Aware Chunking**: Reduce cross-cluster E-core fan-out & condvar churn on `< 2 MB` bandwidth-bound ops | `internal/gobackend/ops/binaryops_simd.go`, `internal/gobackend/ops/unaryops_simd.go` | 5.2 ms | ~3.5 ms | 11.2 ms | ~7.5 ms | Planned |
| — | `FusedDense` + Remaining Ops | — | 22.8 ms | ~22.0 ms | 83.5 ms | ~81.0 ms | — |
| **Total** | **End-to-End Forward Pass (`gomlx-simd`)** | — | **58.0 ms (`558.8 ms` CPU)** | **39.1 ms (`268.0 ms` CPU, `1.48×` faster wall, `2.09×` less CPU)** | **179.4 ms (`1,860 ms` CPU)** | **122.2 ms (`1,100 ms` CPU, `1.47×` faster wall, `1.69×` less CPU)** | **P1–P3 Done** |

---

## 7. Priority 1 Results: SME + 4-Row NEON Assembly for `FusedScaledDotProductAttention`

### 7.1 What Was Implemented

1. **Unlocked `booleanMask` and `causal` in Both SME (`sdpaSMEBlockFloat32`) and 4-Row NEON (`sdpaNEONBlockFloat32`) ([`internal/gobackend/dot/matmul/neon/sdpa_arm64.go`](../internal/gobackend/dot/matmul/neon/sdpa_arm64.go))**:
   - Removed `!causal && len(booleanMask) == 0` from `canBatchQueries`.
   - Added `sdpaActiveKVLenForBlock`, which inspects the query block `[qBase .. qBase+activeRows)` under `causal` and/or `booleanMask` and computes the tightest `blockKVLen` (trimming trailing masked keys). For causal / lower-triangular masks, early query blocks only pack and multiply the active prefix `[0 .. blockKVLen)`, cutting $Q K^T$ and $\text{Probs} \times V$ work in half.
   - Updated `sdpaScaleMaskSoftmaxRowNEON` to support `booleanMask` with 4-byte `*(*uint32)` fast-paths for `0x01010101` (4 unmasked keys: vector `FMLA` / `FMUL`) and `0x00000000` (4 masked keys: store `-Inf`), plus an all-masked row guard (`rowMax == -Inf` $\to$ zero probabilities).
2. **Partial Query Block Support (`activeRows = 4..32` in SME, `1..4` in NEON Assembly)**:
   - Generalized `sdpaSME32RowsFloat32` to `sdpaSMEBlockFloat32` (`activeRows` in `4..32`, `blockKVLen >= 16`) so Timestral Decoder Self/Cross-Attention (`qLen = 27`) and Encoder tail blocks (`378 % 32 = 26` rows) execute directly on SME!
   - Generalized `sdpaNEON4RowsFloat32` to `sdpaNEONBlockFloat32` (`activeRows` in `1..4`) using `neonSmallTransposedTile4x4Float32Asm` + `Tile4x1` for $Q \times K^T$ and `neonSmallNonTransposedTile4x16Float32Asm` + `Tile4x4` for $\text{Probs} \times V$, so even `qLen = 2` Variate Attention (`[405, 2, H, 64]`) runs in handwritten NEON assembly.
3. **Lazy Once-Per-KV-Head Hardware `ZA` $K$ Packing (`smePackLHS32Float32Asm`)**:
   - Replaced per-32-query-row scalar/NEON $K$ packing with lazy incremental `ZA` hardware transpose packing (`smePackLHS32Float32Asm`) tracked by `packedKUpTo` across the entire KV head (`qStart..qEnd`).
   - Replaced multi-strip `smeLargeKernelFloat32Asm_32x32` loops with a single multi-tile assembly call over `(blockKVLen + 31) &^ 31` columns.
4. **Fused Single-`SMSTART` `ZA` Transpose + Direct Strided $V$ Kernel (`smeDirectStridedFloat32Asm_32x32` in [`internal/gobackend/dot/matmul/neon/sme_arm64.s`](../internal/gobackend/dot/matmul/neon/sme_arm64.s))**:
   - Eliminated $V$ packing (`packedV`) entirely!
   - Added `smeDirectStridedFloat32Asm_32x32`, which enters `SMSTART` once, transposes `scoresScratch[activeRows, blockKVLen]` into L1-resident `packedS[blockKVLen, 32]` via `LD1W ZA*H` + `ST1W ZA*V`, and immediately computes $\text{Probs} \times V$ via `FMOPA` reading `V[blockKVLen, headDim]` directly with byte stride `kvSeqStride * 4` (`headDim * 4` for BHSD or `numKVHeads * headDim * 4` for BSHD).
5. **Fine-Grained Per-Phase SME Lock Release & Cluster-1 Preference ([`internal/gobackend/fusedops/sdpa.go`](../internal/gobackend/fusedops/sdpa.go))**:
   - Inside `sdpaSMEBlockFloat32`, the worker holds the cluster SME lock during **Phase 1** ($K/Q$ `ZA` packing + $Q K^T$ `FMOPA`), **releases the SME lock (`simdarch.ReleaseSME`) before Phase 2** (32-row `sdpaScaleMaskSoftmaxRowNEON` on the core's private NEON unit), and **re-acquires the SME lock for Phase 3** (`smeDirectStridedFloat32Asm_32x32`).
   - Why this matters: For a 32-row block (`kvLen = 378, headDim = 64`), Phase 1 + Phase 3 on Cluster 1's 2.26 TFlops/s SME unit take only **`~0.8 µs`**, while Phase 2 (32 rows of NEON `exp` + normalization) takes **`~2.0 µs`** on a single core. Releasing the SME lock during Phase 2 pipelines all 6 Super/Performance cores on Cluster 1 through NEON Softmax in parallel while time-multiplexing Phase 1 and Phase 3 onto Cluster 1's shared SME unit (`42.4 ms` multi-worker pipelined vs `53.9 ms` single-threaded SME on `Timestral-28M`)!
   - In `sdpaMultiHeadGeneric` (`fusedops/sdpa.go`), kept each head unsplit (`qBlockSize = max(seqLen, 1)`) when `headTasks >= numWorkers` (`12` in `28M`, `20` in `123M`), and added Cluster-1 task preference (`preferFastCluster`) so Efficiency-core workers yield remaining heads (`rem <= 8 * fastWorkers`) to active Cluster 1 workers.

### 7.2 End-to-End Benchmark Comparison (Before vs. After Priority 1)

| Model | Configuration | Baseline (`ecd18aa`) | **After Priority 1** | Wall-Time Speedup | CPU-Time Reduction | Max Abs Diff vs PyTorch |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`Timestral-28M`** | `gomlx-simd` (Default) | 58.0 ms (`558.8 ms` CPU) | **41.5–42.4 ms (`286.2–309.2 ms` CPU)** | **`1.37×–1.40×` (`-27%` to `-28.4%`)** | **`-45%` (`-250 ms` CPU)** | `3.4e-05` (`0.0000`) |
| **`Timestral-28M`** | `gomlx-simd-seq` (`ops_sequential`) | 65.2 ms (`548.8 ms` CPU) | **49.5 ms (`320.2 ms` CPU)** | **`1.32×` (`-24.1%`)** | **`-42%` (`-228 ms` CPU)** | `3.4e-05` (`0.0000`) |
| **`Timestral-123M`** | `gomlx-simd` (Default) | 179.4 ms (`1,860 ms` CPU) | **128.4 ms (`1,300 ms` CPU)** | **`1.40×` (`-28.4%`, `-51.0 ms`)** | **`-30%` (`-560 ms` CPU)** | `1.9e-05` (`0.0000`) |

#### Detailed Run-by-Run After Priority 1 (`gomlx-simd`)

##### `mistralai/Timestral-28M`
| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Live Mem | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gomlx-simd` | `Timestral-28M` | warm-up | 53.7ms | 341.0ms | 6.4 | 3.4e-05 | 0.22 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 1 | 44.3ms | 328.9ms | 7.4 | 3.4e-05 | 0.25 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 2 | 46.7ms | 334.7ms | 7.2 | 3.4e-05 | 0.27 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 3 | 43.3ms | 318.1ms | 7.3 | 3.4e-05 | 0.30 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 4 | 43.7ms | 316.9ms | 7.3 | 3.4e-05 | 0.32 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 5 | 43.0ms | 314.3ms | 7.3 | 3.4e-05 | 0.34 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 6 | 42.4ms | 309.2ms | 7.3 | 3.4e-05 | 0.37 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 7 | 42.7ms | 311.5ms | 7.3 | 3.4e-05 | 0.39 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 8 | 45.2ms | 328.1ms | 7.3 | 3.4e-05 | 0.36 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 9 | 43.5ms | 319.8ms | 7.3 | 3.4e-05 | 0.39 GB | 0.42 GB |
| `gomlx-simd` | `Timestral-28M` | run 10 | 45.7ms | 331.4ms | 7.2 | 3.4e-05 | 0.41 GB | 0.42 GB |
| **`gomlx-simd`** | **`Timestral-28M`** | **`best`** | **`42.4ms`** | **`309.2ms`** | **`7.3`** | **`3.4e-05`** | **`0.41 GB`** | **`0.42 GB`** |

##### `mistralai/Timestral-123M`
| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Live Mem | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gomlx-simd` | `Timestral-123M` | warm-up | 164.7ms | 1.4s | 8.2 | 1.9e-05 | 1.07 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 1 | 132.4ms | 1.3s | 9.9 | 1.9e-05 | 1.13 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 2 | 129.0ms | 1.3s | 9.9 | 1.9e-05 | 1.19 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 3 | 128.4ms | 1.3s | 9.9 | 1.9e-05 | 1.25 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 4 | 131.6ms | 1.3s | 10.0 | 1.9e-05 | 1.30 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 5 | 136.6ms | 1.3s | 9.7 | 1.9e-05 | 1.30 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 6 | 130.9ms | 1.3s | 10.0 | 1.9e-05 | 1.30 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 7 | 132.7ms | 1.3s | 10.0 | 1.9e-05 | 1.35 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 8 | 135.4ms | 1.4s | 10.0 | 1.9e-05 | 1.35 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 9 | 130.9ms | 1.3s | 10.0 | 1.9e-05 | 1.34 GB | 1.35 GB |
| `gomlx-simd` | `Timestral-123M` | run 10 | 131.8ms | 1.3s | 10.0 | 1.9e-05 | 1.34 GB | 1.35 GB |
| **`gomlx-simd`** | **`Timestral-123M`** | **`best`** | **`128.4ms`** | **`1.3s`** | **`9.9`** | **`1.9e-05`** | **`1.34 GB`** | **`1.35 GB`** |

### 7.3 Per-Op `FusedScaledDotProductAttention` Speedup Breakdown (`GOMLX_OP_PROFILE`)

| SDPA Op Signature | Calls/run | Baseline Wall/run | **After P1 Wall/run** | Wall Speedup | Baseline CPU/run | **After P1 CPU/run** | CPU Reduction |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`28M` Encoder**: `SDPA([2, 378, 6, 64], ..., (Bool)[2, 378, 378])` | 12 | 21.6 ms | **6.1 ms** | **`3.54×`** | 234.3 ms | **32.2 ms** | **`7.28×`** |
| **`28M` Decoder**: `SDPA([2, 27, 6, 64], [2, 405, 6, 64], ..., (Bool)[2, 27, 405])` | 12 | 3.5 ms | **1.2 ms** | **`2.92×`** | 33.0 ms | **6.2 ms** | **`5.32×`** |
| **`28M` Total `FusedScaledDotProductAttention`** | **27** | **25.4 ms (`39.0%`)** | **7.8 ms (`15.1%`)** | **`3.26×`** | **269.8 ms** | **39.9 ms** | **`6.76×`** |
| **`123M` Encoder**: `SDPA([2, 378, 10, 64], ..., (Bool)[2, 378, 378])` | 20 | 59.8 ms | **15.7 ms** | **`3.81×`** | 652.1 ms | **113.0 ms** | **`5.77×`** |
| **`123M` Decoder**: `SDPA([2, 27, 10, 64], [2, 405, 10, 64], ..., (Bool)[2, 27, 405])` | 20 | 9.1 ms | **5.8 ms** | **`1.57×`** | 93.1 ms | **46.6 ms** | **`2.00×`** |
| **`123M` Total `FusedScaledDotProductAttention`** | **45** | **69.8 ms (`36.1%`)** | **23.1 ms (`17.6%`)** | **`3.02×`** | **753.1 ms** | **168.4 ms** | **`4.47×`** |

---

## 8. Priority 2 & Priority 3 Results: Handwritten ARM64 NEON `SwiGLU`, Parallel `RMSNorm` Sub-Ops, and LIFO Buffer Pooling

### 8.1 What Was Implemented

1. **Priority 2 — Handwritten ARM64 NEON Assembly for `SwiGLU` ([`internal/gobackend/activations/neon/swiglu.s`](../internal/gobackend/activations/neon/swiglu.s), [`swiglu.go`](../internal/gobackend/activations/neon/swiglu.go))**:
   - Implemented `swigluFloat32NEONAsm(in, out unsafe.Pointer, numRows, hiddenDim, vecWidth int)` (`TEXT ·swigluFloat32NEONAsm(SB), NOSPLIT, $0-40`), registered via `activations.RegisterSwiGLU`.
   - Hoists the 13 Cephes polynomial/range-reduction constants (`V19`–`V31`) outside the outer `numRows` loop and processes **16 `float32` elements (`4×` 128-bit NEON vectors) per inner-loop iteration**, issuing early hardware `FDIV` on the first vector pair overlapped with the second vector pair's Estrin polynomial evaluation, followed by concurrent FMA-pipe fast reciprocal (`FRECPE` + `FRECPS` Newton-Raphson refinement) and hardware `FDIV` on the second pair.
   - Added NEON `SwiGLU` dispatchers for `float64`, `bfloat16`, and `float16` (`SwiGLUNEONFloat64`, `SwiGLUNEONBF16`, `SwiGLUNEONF16`), plus unit and microbenchmark coverage in [`internal/gobackend/activations/neon/neon_test.go`](../internal/gobackend/activations/neon/neon_test.go).
   - **Microbenchmark Speedup (`BenchmarkNEONComparison/.../SwiGLU`)**: **`4.29×–4.46×` faster** than `PortableSIMD` (`80.4 ns` vs `358.5 ns` on `Small_512`; `10.3 µs` vs `44.2 µs` on `Large_65536`).

2. **Priority 3 — Parallel `BroadcastTrailing` (`[A, B]` op `[A, 1]`) and `ReduceTrailing` (`[A, B]` $\to$ `[A]`) ([`internal/gobackend/ops/binaryops_simd.go`](../internal/gobackend/ops/binaryops_simd.go), [`internal/gobackend/ops/reduce_simd.go`](../internal/gobackend/ops/reduce_simd.go))**:
   - In `tryBinaryTrailingArch` (`binaryops_simd.go`), parallelized `BroadcastTrailingRHS` and `BroadcastTrailingLHS` across outer rows `bcastCfg.A` via `backend.Scheduler.ParallelizeChunks` when `bcastCfg.A * bcastCfg.B >= minBinarySIMDParallelElements` (`65,536` elements).
   - In `dispatchReduceSumSIMD` (`reduce_simd.go`), parallelized `ReduceTrailing` across outer rows `cfg.A` via `backend.Scheduler.ParallelizeChunks` when `cfg.A * cfg.B >= 65536`.

3. **Backend Buffer Pool Fix (`bufferPool` in [`internal/gobackend/buffers.go`](../internal/gobackend/buffers.go))**:
   - Replaced `sync.Pool` in `Backend.bufferPools` with a backend-owned **lock-free atomic LIFO Treiber stack** (`bufferPool`, indexed in $O(1)$ by `bufferPoolBucketIndex(bucketedSize)` across a fixed `[128]atomic.Pointer[bufferPool]` array, and released on `Backend.Finalize()`).
   - **ABA-free 64-bit tagged atomic head**: Packs a 32-bit monotonic ABA generation counter in the upper 32 bits and a 1-based slot index in the lower 32 bits of `bufferPool.head` (`atomic.Uint64`), eliminating mutex serialization when up to 32 worker goroutines acquire/release scratch buffers simultaneously.
   - **Zero-allocation `GetBuffer` / `PutBuffer`**: Retains the typed `Buffer.Flat` slice header (`flatDType`, `flatLen`) across pool round-trips and avoids variadic `dimensions ...int` heap escape in `dtypes.DType.SizeForDimensions`, achieving **`0 B/op, 0 allocs/op`** (`13.85 ns/op` sequential round-trip).
   - **Why `sync.Pool` hurt multi-worker graph execution**: In Go's `sync.Pool`, each `P` has a `private` slot that **cannot be stolen by other `P`s**. Whenever the graph executor goroutine blocked at a parallel barrier (`wg.Wait()`) and resumed on a different `P`, buffers returned via `PutBuffer` on the previous `P` were stranded in that `P`'s `private` slot, causing up to $12\times$ (or $32\times$ on 32-core `amd64`) duplicate intermediate buffer allocations and catastrophic `300+ MB` reallocation + page-zeroing storms whenever `runtime.GC` wiped `sync.Pool`.
   - With `bufferPool`, any `P` immediately reuses the most recently released buffer (LIFO cache locality), cutting `Timestral-28M` Max RSS from `0.45 GB` to **`0.31 GB`** and accelerating `Timestral-123M` warm-up from `164.7 ms` to **`150.5 ms`** (best run **`120.0 ms`**).

### 8.2 Per-Op Speedups from Priority 2 & Priority 3 (`GOMLX_OP_PROFILE`, Sequential Mode)

| Operation Signature | Model | Calls/run | Before P2/P3 Wall/run | **After P2/P3 Wall/run** | Wall Speedup |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `FusedActivation((Float32)[2, 405, 2048]) -> (Float32)[2, 405, 1024]` (`SwiGLU`) | `28M` | 12 | 6.6 ms | **1.2 ms** | **`5.50×`** |
| `FusedActivation((Float32)[2, 405, 3424]) -> (Float32)[2, 405, 1712]` (`SwiGLU`) | `123M` | 20 | 16.5 ms | **3.6 ms** | **`4.58×`** |
| `ReduceSum((Float32)[2, 405, 384]) -> (Float32)[2, 405]` | `28M` | 28 | 1.5 ms | **0.40 ms** | **`3.75×`** |
| `ReduceSum((Float32)[2, 405, 640]) -> (Float32)[2, 405]` | `123M` | 46 | 4.7 ms | **1.4 ms** | **`3.36×`** |
| `Mul((Float32)[2, 405, 384], (Float32)[2, 405, 1]) -> (Float32)[2, 405, 384]` | `28M` | 28 | 1.4 ms | **0.56 ms** | **`2.50×`** |
| `Mul((Float32)[2, 405, 640], (Float32)[2, 405, 1]) -> (Float32)[2, 405, 640]` | `123M` | 46 | 6.5 ms | **2.7 ms** | **`2.41×`** |

### 8.3 Cumulative End-to-End Benchmark Progression (Baseline $\to$ Priority 1 $\to$ Priority 2 & 3)

| Model | Stage | Best Wall Time | Best CPU Time | Avg Cores | Live Mem | Speedup vs Baseline | Speedup vs NEON-Only |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Timestral-28M`** | NEON-Only (`GOMLX_GO_SIMD_SME=0`) | 82.7 ms | 832.1 ms | 10.1 | 0.40 GB | 0.70× | 1.00× |
| **`Timestral-28M`** | Initial Baseline (`ecd18aa`) | 58.0 ms | 558.8 ms | 9.6 | 0.40 GB | 1.00× | 1.43× |
| **`Timestral-28M`** | After Priority 1 (`ea8ff08`) | 42.4 ms | 309.2 ms | 7.3 | 0.41 GB | 1.37× | 1.95× |
| **`Timestral-28M`** | **After Priority 2 & 3** | **39.1 ms** | **268.0 ms** | **6.9** | **0.26 GB** | **`1.48×` (`-32.6%` wall, `-52.0%` CPU)** | **`2.12×` wall, `3.10×` CPU** |
| **`Timestral-123M`** | NEON-Only (`GOMLX_GO_SIMD_SME=0`) | 293.4 ms | 3,170.0 ms | 10.8 | 1.35 GB | 0.61× | 1.00× |
| **`Timestral-123M`** | Initial Baseline (`ecd18aa`) | 179.4 ms | 1,860.0 ms | 10.4 | 1.35 GB | 1.00× | 1.64× |
| **`Timestral-123M`** | After Priority 1 (`ea8ff08`) | 128.4 ms | 1,300.0 ms | 9.9 | 1.34 GB | 1.40× | 2.29× |
| **`Timestral-123M`** | **After Priority 2 & 3** | **122.2 ms** | **1,100.0 ms** | **9.4** | **0.65 GB** | **`1.47×` (`-31.9%` wall, `-40.9%` CPU)** | **`2.40×` wall, `2.88×` CPU** |

---

## 9. AMD64 AVX2 Optimization on Intel Core i9-12900K (`Timestral-28M`)

Following the ARM64 NEON/SME optimizations on Apple M6, we profiled and optimized the `go` backend on an x86-64 AVX2 workstation (`12th Gen Intel Core i9-12900K`, 24 logical CPUs: 8 Golden Cove P-cores / 16 threads with 1.25 MB L2 per core + 8 Gracemont E-cores / 8 threads with 2 MB L2 per 4-core cluster, 30 MB shared L3, AVX2 + FMA + F16C, no AVX-512).

### 9.1 Baseline Profile & Root Causes Identified on AVX2

Before AVX2-specific optimizations on this branch:
- **`go` backend (`-backend="go"`, 10 runs)**: **134.4 ms best wall** (`1,899.6 ms` CPU, `14.1` cores), vs `xla:cpu` at **73.7 ms best wall** (`1,089.3 ms` CPU).
- **`go:ops_sequential` (`GOMLX_OP_PROFILE=5`)**: **168.4 ms Sum of Ops** (`173.4 ms` best wall, `1,996.4 ms` CPU), dominated by:
  1. **`FusedScaledDotProductAttention` (`95.1 ms` wall / `56.5%` of total runtime, `1,059.1 ms` CPU)**:
     - `sdpaAVX2BlockFloat32` in [`internal/gobackend/fusedops/avx2/sdpa.go`](../internal/gobackend/fusedops/avx2/sdpa.go) processed only **1 query row at a time** (`sdpaDotProductRowAVX2Asm` + `sdpaValueAccumRowAVX2Asm`), reloading every `K` and `V` vector from L2/L3 cache for every single query row without register-blocking across queries, and computed all `kvLen = 378` keys even when the causal mask zeroed out trailing keys (`50%` wasted FLOPs).
     - `sdpaScaleMaskSoftmaxRowAVX2Asm` lacked a boolean mask (`maskStride != 0`) path, forcing a pure-Go scalar loop (`scores[j] *= scale; if !maskRow[j] { scores[j] = -Inf }`) before calling assembly softmax.
     - Work partitioning in [`internal/gobackend/fusedops/sdpa.go`](../internal/gobackend/fusedops/sdpa.go) hardcoded `qBlockSize = 32`, yielding only $\lceil 378 / 32 \rceil = 12$ tasks per head for `12 × 12 = 144` total tasks across 24 logical CPUs, and `12 × 1 = 12` tasks for Decoder SDPA (`seqLen = 27`, leaving half the 24 logical CPUs completely idle).
  2. **`FusedActivation` / `SwiGLU` (`1.9 ms` wall, `26.8 ms` CPU)**:
     - `internal/gobackend/activations/avx2/` had handwritten AVX2 assembly for `GeluExact`, `GeluApprox`, and `Silu`, **but no AVX2 implementation for `SwiGLU`**, falling back to the generic `archsimd` closure path (`swigluSIMDFloat32`).
  3. **`Iota` (`2.2 ms` wall) & 3-Segment Broadcasts (`LogicalAnd` `791.2 µs` wall)**:
     - `execIotaGeneric` in [`internal/gobackend/ops/iota.go`](../internal/gobackend/ops/iota.go) computed every element inside a 3-level nested loop without `repeatsSize == 1` specialization or doubling `copy()` across `batchSize > 1`.
     - 3D broadcasts of the form `[A, 1, C]` op `[1, B, C]` $\to$ `[A, B, C]` (such as `LogicalAnd([378, 1, 378], [1, 378, 378])` in causal group-mask construction) and `[A, B, 1]` op `[A, 1, C]` $\to$ `[A, B, C]` fell through `ClassifyBroadcast` to `BroadcastGeneric`, executing scalar `NewZippedBroadcastIterator` coordinate div/mod indexing for `285,768` elements per call.
  4. **Go 1.27 `simd` Slice-Header Stack Spills on `amd64`, Indirect Unary Closures, and L2 Cache Thrashing (`Mul` `12.6 ms`, `Add` `4.1 ms`, `Neg` `994 µs`, `ReduceSum` `2.3 ms`, `Concatenate` `2.1 ms`)**:
     - Disassembling `simdV2V1MulFloat32@simd256` and `simdAddVVFloat32@simd256` on `amd64` revealed that because `amd64` has only 14 allocatable general-purpose registers, calling `simd.LoadFloat32s(lhs[i+vLen:])` inside a 4× unrolled loop constructs 12 3-word Go slice headers per iteration with bounds checks and Spectre mitigation masks (`NEGQ; SARQ $63; ANDQ`), spilling **24 64-bit values to the stack** and emitting **893 bytes of machine code** per 32-float inner loop!
     - `simdUnaryFloat32` was not inlined by the Go compiler, so `Neg`, `Abs`, `Sqrt`, and `Rsqrt` performed an indirect function call (`CALL DX`) on every 8-float vector.
     - Splitting a `311,040`-float (`1.24 MB`) tensor into 10 chunks of `32,768` floats (`128 KB`) woke 9 OS threads via `futex_wake` (`~25–30 µs` overhead per op), whereas running single-threaded spilled a `2.48–3.73 MB` working set outside a single P-core's `1.25 MB` L2 cache.

### 9.2 Optimizations Implemented (Priorities 1–4)

1. **Priority 1 — 4-Row Register-Blocked AVX2 SDPA Microkernels, Vectorized Boolean Masking, Active Causal KV Pruning & Dynamic Query Block Sizing**:
   - **4-Row $Q K^T$ & $P V$ AVX2 Assembly ([`internal/gobackend/fusedops/avx2/sdpa_amd64.s`](../internal/gobackend/fusedops/avx2/sdpa_amd64.s))**:
     - `sdpaQKT4RowsAVX2Asm`: Computes 4 query rows $\times$ `activeKVLen` keys simultaneously, keeping 4 query rows cached in L1D while loading each key vector once and accumulating into 4 independent YMM accumulators (`Y0`–`Y3`), cutting `K` memory traffic by $4\times$. Includes a dedicated `headDim == 64` unrolled fast path (8 `VMOVUPS` per key row, zero inner loop branch overhead).
     - `sdpaValueAccum4RowsAVX2Asm`: Accumulates $\sum_j P_{i,j} V_j$ for 4 query rows simultaneously. For `headDim == 64`, pins all 4 output rows in **16 YMM registers (`Y0`–`Y7` for rows 0–1, `Y8`–`Y15` for rows 2–3)** across the `activeKVLen` loop, loading each `V` row (`64` floats = 8 YMM vectors) once from L1/L2 cache and updating all 4 query outputs in registers before writing back once at the end.
     - `sdpaQKT1RowAVX2Asm` & `sdpaValueAccum1RowAVX2Asm`: Single-row tail kernels with a dedicated `headDim == 64` 8-YMM in-register accumulator path (`Y0`–`Y7`) and 2-way unrolled `V` accumulation.
   - **Fused Scale + Boolean Mask + Softmax AVX2 Assembly (`sdpaScaleMaskSoftmaxRowAVX2Asm`)**:
     - Vectorizes boolean mask application (`maskStride != 0`) directly inside the max-finding pass using 8-byte `VMOVQ` + `VPMOVSXBD` (sign-extend 8 bytes to 8 `int32` lanes in YMM) + `VPCMPGTD` + `VBLENDVPS` to blend `-Inf` (`0xFF800000`) where `maskByte == 0`, followed by fused 2-pass Estrin polynomial `exp(x - max)` and reciprocal normalization.
   - **Causal Mask Active-KV Pruning & All-True Fast Path ([`internal/gobackend/fusedops/avx2/sdpa.go`](../internal/gobackend/fusedops/avx2/sdpa.go))**:
     - Added `sdpaActiveKVLenForBlock` to scan backwards from `kvLen - 1` across the 4 query rows in a block to find the last unmasked key index, skipping trailing masked keys entirely in both $Q K^T$ and $P V$ (halving FLOPs on causal encoder masks) and bulk-filling skipped tail positions only when absent from the fused softmax loop.
     - Added `isAllTrueBoolMask` using unaligned 64-bit word loads (`0x0101010101010101`) to detect all-true mask blocks in $O(\text{kvLen}/8)$ and bypass per-element mask blending.
   - **Dynamic Query Block Sizing ([`internal/gobackend/fusedops/sdpa.go`](../internal/gobackend/fusedops/sdpa.go))**:
     - Replaced fixed `qBlockSize = 32` with dynamic sizing targeting `4 * numWorkers` tasks, rounded to multiples of 4 rows (`qBlockSize = max(4, ((totalWork / targetTasks) &^ 3))` capped at `32`), so Decoder SDPA (`seqLen = 27`, `numHeads = 6`) uses `qBlockSize = 4` (`84` tasks across 24 workers instead of `12`).

2. **Priority 2 — Handwritten AVX2 Assembly for `SwiGLU` ([`internal/gobackend/activations/avx2/swiglu_amd64.s`](../internal/gobackend/activations/avx2/swiglu_amd64.s), [`swiglu.go`](../internal/gobackend/activations/avx2/swiglu.go))**:
   - Implemented `swigluFloat32AVX2Asm(in, out unsafe.Pointer, numRows, hiddenDim, vecWidth int)` (`TEXT ·swigluFloat32AVX2Asm(SB), NOSPLIT, $0-40`) and registered `SwiGLUFloat32AVX2`, `SwiGLUAVX2Float64`, `SwiGLUAVX2BF16`, and `SwiGLUAVX2F16` in [`internal/gobackend/activations/avx2/avx2.go`](../internal/gobackend/activations/avx2/avx2.go).
   - Processes **16 `float32` elements (`2×` 256-bit YMM vectors) per inner-loop iteration** with zero stack spills:
     - Evaluates range reduction and degree-5 Estrin polynomial for $e^{-x_0}$ and $e^{-x_1}$ interleaved across both 8-float vectors (`Y0`–`Y9`), keeping constants in `Y10`–`Y15` and loading Cephes `C2..C5` directly as memory operands into `VFMADD213PS` / `VFMADD231PS`.
     - Overlaps hardware `VDIVPS` on the first 8-float vector (`x0 / (1 + exp(-x0))`) with fast Newton-Raphson reciprocal (`VRCPPS` + `VFNMADD213PS` + `VFMADD213PS`) on the second 8-float vector, then multiplies by the `up` projection `u0, u1` from `in[row + hiddenDim + col]`.
   - **Microbenchmark Speedup (`BenchmarkAVX2Comparison/Float32/Large_65536/SwiGLU`)**: **`4.51×` faster** than `PortableSIMD` (`7.33 µs` vs `33.09 µs`).

3. **Priority 3 — `Iota` Doubling Copy & 3-Segment Broadcast Fast Paths**:
   - **Vectorized `Iota` with Doubling Copy ([`internal/gobackend/ops/iota.go`](../internal/gobackend/ops/iota.go))**:
     - Specialized `repeatsSize == 1` (`firstBatch[i] = T(i)`) and `repeatsSize > 1` (`slice[j] = val` on pre-sliced `firstBatch[start:end]`) on the first batch slice (`iotaDimSize * repeatsSize`), then replicated across `batchSize > 1` via exponential doubling `copy(flat[filled:filled+toCopy], flat[:toCopy])`.
   - **3-Segment Broadcast Fast Paths ([`internal/gobackend/broadcast.go`](../internal/gobackend/broadcast.go), [`internal/cmd/gobackend_ops_generator/exec_binary.go`](../internal/gobackend/ops/gen_exec_binary.go), [`internal/gobackend/ops/binaryops_simd.go`](../internal/gobackend/ops/binaryops_simd.go))**:
     - Extended `ClassifyBroadcast` with 4 new 3-segment patterns where 2 adjacent dimensions collapse:
       - `BroadcastRowColMatch`: `[A, 1, C]` op `[1, B, C]` $\to$ `[A, B, C]`
       - `BroadcastColRowMatch`: `[1, B, C]` op `[A, 1, C]` $\to$ `[A, B, C]`
       - `BroadcastMatchRowCol`: `[A, B, 1]` op `[A, 1, C]` $\to$ `[A, B, C]`
       - `BroadcastMatchColRow`: `[A, 1, C]` op `[A, B, 1]` $\to$ `[A, B, C]`
     - Generated typed inner-slice loops in `gen_exec_binary.go` and SIMD dispatchers in `binaryops_simd.go`, eliminating `NewZippedBroadcastIterator` overhead completely.

4. **Priority 4 — Zero-Spill AVX2 Binary Mid/Leading/Contiguous Assembly Kernels, Direct Unary SIMD Loops & L2-Cache-Aware Parallelism**:
   - **Zero-Spill AVX2 Binary Mid/Leading/Contiguous Assembly ([`internal/gobackend/ops/avx2/binary_mid_amd64.s`](../internal/gobackend/ops/avx2/binary_mid_amd64.s), [`internal/gobackend/binary_arch_trailing.go`](../internal/gobackend/binary_arch_trailing.go))**:
     - Added `BinaryMidArchFn` (`DispatchBinaryMidAVX2`) handling `[A, B, C]` op `[A, 1, C]` (`BroadcastMidRHS`/`BroadcastMidLHS`), `[1, B, C]` (`BroadcastLeadingRHS`/`BroadcastLeadingLHS`, with `A = 1`), and contiguous `[N]` (`BroadcastNone`, with `A = 1, B = 1`) for `Float32` `Add`, `Sub`, `Mul`, and `Div`.
     - Eliminates Go 1.27 `simd` slice-header stack spills and bounds checks on `amd64`. Includes a dedicated **`C == 64` register-cached fast path** that loads all 64 floats of the broadcast `[a, 0, :64]` slice into **`Y8`–`Y15` once per outer `a`** and streams across all `B` rows in L1/L2 cache (`8 × VFMADD`/`VMULPS`/`VADDPS` per row without reloading the broadcast operand), plus a 32-float (`4×` YMM) unrolled general loop.
     - Extended `tryBinaryTrailingArch` in [`internal/gobackend/ops/binaryops_simd.go`](../internal/gobackend/ops/binaryops_simd.go) to also route `BroadcastScalarRHS` and `BroadcastScalarLHS` (`A = 1, B = totalElements`) through the zero-spill AVX2 trailing assembly kernel.
   - **Direct Inlined Unary Loops ([`internal/gobackend/ops/unaryops_simd.go`](../internal/gobackend/ops/unaryops_simd.go))**:
     - Replaced `simdUnaryFloat32` / `simdUnaryFloat64` closure calls for `Neg`, `Abs`, `Sqrt`, and `Rsqrt` with direct 4× unrolled functions (`simdNegFloat32`, `simdAbsFloat32`, `simdSqrtFloat32`, `simdRsqrtFloat32`, etc.), eliminating the per-vector indirect `CALL DX`.
   - **L2-Cache-Aware Parallel Chunking ([`binaryops_simd.go`](../internal/gobackend/ops/binaryops_simd.go), [`unaryops_simd.go`](../internal/gobackend/ops/unaryops_simd.go), [`stridedcopy.go`](../internal/gobackend/ops/stridedcopy.go), [`reduce_simd.go`](../internal/gobackend/ops/reduce_simd.go))**:
     - Tuned `minBinarySIMDParallelElements = 131072` and `binarySIMDChunkElements = 65536` (`256 KB` per tensor = `512–768 KB` working set per chunk), recruiting 3–5 P-cores for `311,040`-float (`1.24 MB`) tensors so each chunk fits inside a P-core's `1.25 MB` L2 cache while avoiding 10-way `futex_wake` overhead.

### 9.3 Per-Op Profile Comparison on AVX2 (`GOMLX_OP_PROFILE=5`, `-backend="go:ops_sequential"`)

| Operation Type | Calls/run | Baseline Wall/run | **Optimized Wall/run** | Wall Speedup | Baseline CPU/run | **Optimized CPU/run** | CPU Reduction |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`FusedScaledDotProductAttention`** | 27 | 95.1 ms (`56.5%`) | **8.1 ms (`10.9%`)** | **`11.74×`** | 1,059.1 ms | **94.7 ms** | **`11.18×`** |
| **`FusedDense`** | 93 | 40.3 ms (`23.9%`) | **41.8 ms (`56.2%`)** | `0.96×` | 538.9 ms | **552.7 ms** | `0.98×` |
| **`Mul`** | 179 | 12.6 ms (`7.5%`) | **8.2 ms (`11.0%`)** | **`1.54×`** | 183.7 ms | **73.6 ms** | **`2.50×`** |
| **`Add`** | 65 | 4.1 ms (`2.4%`) | **3.0 ms (`4.0%`)** | **`1.37×`** | 56.1 ms | **30.6 ms** | **`1.83×`** |
| **`DotGeneral`** | 27 | 2.6 ms (`1.5%`) | **2.7 ms (`3.6%`)** | `0.96×` | 20.6 ms | **21.6 ms** | `0.95×` |
| **`ReduceSum`** | 28 | 2.3 ms (`1.4%`) | **1.3 ms (`1.8%`)** | **`1.77×`** | 33.9 ms | **13.5 ms** | **`2.51×`** |
| **`Iota`** | 60 | 2.2 ms (`1.3%`) | **0.28 ms (`0.4%`)** | **`7.92×`** | 3.5 ms | **0.56 ms** | **`6.25×`** |
| **`Concatenate`** | 58 | 2.1 ms (`1.3%`) | **1.7 ms (`2.2%`)** | **`1.24×`** | 30.8 ms | **8.8 ms** | **`3.50×`** |
| **`FusedActivation` (`SwiGLU`)** | 15 | 1.9 ms (`1.1%`) | **1.2 ms (`1.6%`)** | **`1.58×`** | 26.8 ms | **10.0 ms** | **`2.68×`** |
| **`Neg`** | 29 | 0.99 ms (`0.6%`) | **0.65 ms (`0.9%`)** | **`1.53×`** | 14.4 ms | **2.2 ms** | **`6.55×`** |
| **`LogicalAnd`** | 15 | 0.79 ms (`0.5%`) | **0.15 ms (`0.2%`)** | **`5.21×`** | 1.3 ms | **0.35 ms** | **`3.71×`** |
| **Total (`Sum of Ops`)** | **1,636** | **168.4 ms (`100%`)** | **74.3 ms (`100%`)** | **`2.27×`** | **1,996.4 ms** | **871.0 ms** | **`2.29×`** |

### 9.4 End-to-End Benchmark Progression on Intel Core i9-12900K (AVX2, `-backend="go"`, 10 Runs)

| Stage | Best Wall Time | Best CPU Time | Avg Cores | Max Abs Diff vs PyTorch | Speedup vs Baseline | Comparison vs `xla:cpu` (`73.7 ms`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `xla:cpu` Reference (`-backend="xla:cpu"`) | 73.7 ms | 1,089.3 ms | 14.8 | 2.1e-05 | 1.82× | `1.00×` |
| **Initial `go` Backend Baseline** | 134.4 ms | 1,899.6 ms | 14.1 | 3.4e-05 | 1.00× | `0.55×` (`1.82×` slower) |
| **After Priority 1 (4-Row AVX2 SDPA + Dynamic `qBlockSize`)** | 71.1 ms | 1,015.4 ms | 14.3 | 3.4e-05 | **`1.89×`** | **`1.04×` faster than XLA** |
| **After Priority 2 (Handwritten AVX2 `SwiGLU` Assembly)** | 70.6 ms | 1,001.2 ms | 14.2 | 3.4e-05 | **`1.90×`** | **`1.04×` faster than XLA** |
| **After Priority 3 (`Iota` Doubling Copy + 3-Segment Broadcasts)** | 69.5 ms | 1,004.8 ms | 14.5 | 3.4e-05 | **`1.93×`** | **`1.06×` faster than XLA** |
| **After Priority 4 (Zero-Spill AVX2 Binary Mid Asm + Direct Unary + L2 Chunking)** | **60.3 ms** | **887.1 ms** | **14.7** | **3.4e-05** | **`2.23×` (`-55.1%` wall, `-53.3%` CPU)** | **`1.22×` faster than XLA (`-18.6%` CPU)** |

#### Detailed Run-by-Run After Priorities 1–4 (`-backend="go"`, Intel Core i9-12900K)

| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Live Mem | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gomlx-simd` | `Timestral-28M` | warm-up | 82.0ms | 1.1s | 13.7 | 3.4e-05 | 0.30 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 1 | 62.6ms | 917.2ms | 14.7 | 3.4e-05 | 0.25 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 2 | 64.9ms | 945.4ms | 14.6 | 3.4e-05 | 0.27 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 3 | 61.9ms | 909.6ms | 14.7 | 3.4e-05 | 0.29 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 4 | 61.5ms | 902.2ms | 14.7 | 3.4e-05 | 0.24 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 5 | 60.5ms | 886.9ms | 14.6 | 3.4e-05 | 0.26 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 6 | 62.4ms | 914.5ms | 14.7 | 3.4e-05 | 0.29 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 7 | 60.3ms | 887.1ms | 14.7 | 3.4e-05 | 0.31 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 8 | 62.9ms | 923.3ms | 14.7 | 3.4e-05 | 0.26 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 9 | 61.5ms | 903.4ms | 14.7 | 3.4e-05 | 0.28 GB | 0.39 GB |
| `gomlx-simd` | `Timestral-28M` | run 10 | 61.2ms | 899.6ms | 14.7 | 3.4e-05 | 0.30 GB | 0.39 GB |
| **`gomlx-simd`** | **`Timestral-28M`** | **`best`** | **`60.3ms`** | **`887.1ms`** | **`14.7`** | **`3.4e-05`** | **`0.30 GB`** | **`0.39 GB`** |

---

## 10. AMD64 AVX-512 Optimization on AMD Ryzen 9 9950X3D (`Timestral-28M` & `Timestral-123M`)

Following the ARM64 NEON/SME and AMD64 AVX2 optimizations, we profiled and optimized the `go` backend on an AMD Zen 5 workstation (`AMD Ryzen 9 9950X3D`, 16 physical cores / 32 logical CPUs across 2 CCDs, 1 MB L2 cache per core, 96 MB + 32 MB L3 cache, native 512-bit AVX-512 data paths with AVX-512F/BW/DQ/VL/BF16/FP16/VNNI).

### 10.1 Baseline Profile & Root Causes Identified on AVX-512

Before the AVX-512 optimizations in this pass, `Timestral-28M` ran in **`40.5 ms` best wall (`44.9–49.3 ms` typical, `454.2 ms` CPU)**, where several operations either fell back to AVX2 (`PriorityAVX2 = 10`) or generic Go SIMD (`PrioritySIMD = 6`), or lacked the multi-row / cache-blocking optimizations added earlier to AVX2:

1. **`FusedScaledDotProductAttention` (`9.8 ms` wall / `129.4 ms` CPU in `28M`)**:
   - `internal/gobackend/fusedops/avx512` only implemented a single-query-row loop (`sdpaSingleHeadFloat32AVX512`) and did not register with `fusedops.RegisterSDPAArchDispatcher`.
   - Whenever `booleanMask` was present (every Encoder and Decoder attention call in `Timestral`) or `headDim == 64` used multi-query batching, `fusedops/avx512` fell back to either the 256-bit `avx2` 4-row kernel or the 1-row `avx512` path that reloaded every $K$ and $V$ vector $4\times$ more often than necessary and computed all masked trailing keys.
2. **Missing AVX-512 `BinaryMidArch` Dispatcher, Unrolled `BinaryTrailing`, and Small-Block `StridedCopy` (`Mul` `5.4 ms`, `Add` `2.6 ms`, `Concatenate` `0.81 ms`, `BroadcastInDim` `0.55 ms`)**:
   - `internal/gobackend/ops/avx512` did not register `SetBinaryMidArchDispatcher`, so all `[A, B, C]` op `[A, 1, C]` (such as RoPE `[810, H, 64] * [810, 1, 64]`), `[1, B, C]`, and contiguous `[N]` `Float32` binary ops fell back to 256-bit AVX2 (`DispatchBinaryMidAVX2`).
   - `binary_trailing_amd64.s` in `ops/avx512` processed only 1 ZMM (16 floats) per loop iteration instead of 4× unrolled ZMMs (64 floats).
   - `stridedcopy_amd64.go` in `ops/avx512` fell back to generic `copy()` when `blockBytes < 64` in `DispatchDupStridedBytesAVX512` or `b0 < 64 || b1 < 64` in `DispatchConcat2StridedBytesAVX512` (whereas `avx2` handled `>= 16` bytes).
3. **`FusedActivation` (`SwiGLU`) Missing in `activations/avx512` (`0.72 ms` wall / `11.1 ms` CPU in `28M`, and `16.9 µs` vs `6.5 µs` on `Large_65536` microbenchmark)**:
   - `internal/gobackend/activations/avx512` had no `SwiGLU` implementation (`swiglu.go` / `swiglu_amd64.s`), causing `FusedActivation(SwiGLU)` to fall back to `PrioritySIMD` (making `AVX512` `2.6×` slower than `AVX2` on `SwiGLU`!).
   - `Sigmoid` and `Tanh` in `activations/avx512` used Go `archsimd` closures instead of handwritten 32-float (2× ZMM) Estrin assembly (`silu_amd64.s`).
   - `AVX512SimpleMetadata` (`MinParallelChunk: 8192`) and `AVX512TranscendentalMetadata` (`MinParallelChunk: 4096`) over-partitioned medium tensors across 32 threads / 2 CCDs.
4. **MatMul `dot/matmul/avx512` L2 Sub-Chunking, `PackLHSKernelRows8` Half/Double Assembly, and `small_transposed` `Tile4x1` Tail**:
   - `internal/gobackend/dot/matmul/avx512/large.go` was missing the L2 cache sub-chunking (`maxMPanelsPerSubChunk`), `sharedRHSPanels` buffer-skip (`needPackedRHS = false`), and cooperative multi-worker `sharedRHSPanels` packing that were added to `matmul/avx2/large.go`.
   - `matmul/avx512` only had `avx512PackLHSKernelRows8Float32Asm`, using scalar Go loops for `Float64`, `Float16`, and `BFloat16` 8-row LHS micro-panel packing.
   - `matmul/avx512/small_transposed.go` fell back to scalar inner loops for remainder rows/cols when `M` or `N` was not a multiple of 4 (missing `avx512SmallTransposedTile4x1*Asm`).
5. **`Where` (`ops/where_amd64.go`) and `LayerNorm` Pass 3 (`fusedops/avx512/layernorm_amd64.s`)**:
   - `ops/where_amd64.go` only implemented 256-bit AVX2 assembly (`where*Float32AVX2Asm`) instead of 512-bit AVX-512 `K`-mask blend (`VPMOVZXBD` + `VPTESTMD` + `VMOVAPS Z, K1, Z`).
   - `fusedops/avx512/layernorm_amd64.s` Pass 3 only unrolled 4× ZMM when both `gamma` and `beta` were non-nil (`Float32`), leaving the `gamma`-only (RMSNorm / LayerNorm without bias) and `Float64` paths at 1 ZMM per iteration.

### 10.2 Optimizations Implemented (Priorities 1–5)

1. **Priority 1 — Multi-Query 4-Row & 1-Row AVX-512 SDPA Assembly with Vectorized `K`-Mask Boolean Masking & Causal Active-KV Pruning ([`internal/gobackend/fusedops/avx512/sdpa_amd64.s`](../internal/gobackend/fusedops/avx512/sdpa_amd64.s), [`sdpa.go`](../internal/gobackend/fusedops/avx512/sdpa.go))**:
   - Registered `DispatchSDPAAVX512` at `PriorityAVX512` via `fusedops.RegisterSDPAArchDispatcher`.
   - Implemented 4-row batched AVX-512 microkernels (`sdpaDot4RowsFloat32AVX512`, `sdpaScaleExpSum4RowsFloat32AVX512`, `sdpaWeightedAccum4RowsFloat32AVX512`, `sdpaFinalize4RowsFloat32AVX512`) and 1-row kernels (`sdpaDot1RowFloat32AVX512`, `sdpaScaleExpSumFloat32AVX512`, `sdpaWeightedAccum1RowFloat32AVX512`):
     - **`sdpaDot4RowsFloat32AVX512`**: Specializes `headDim == 64` (4 ZMM registers per row) by keeping all 4 query rows (`4 × 4 = 16` ZMM registers) in `Z16`–`Z31` across the entire key loop (`0..activeKVLen`), loading each key vector (`4` ZMMs) once and accumulating into 4 independent ZMM accumulators (`Z0`–`Z3`).
     - **`sdpaScaleExpSum4RowsFloat32AVX512`**: Vectorizes boolean mask blending over 16 keys at once using 16-byte `VMOVDQU` + `VPMOVZXBD` + `VPTESTMD` into hardware mask register `K1` and `VMOVAPS Z_negInf, K1, Z_scores`, followed by 512-bit Estrin degree-5 `exp(x - max)` evaluation.
     - **`sdpaWeightedAccum4RowsFloat32AVX512`**: Specializes `headDim == 64` by keeping all 4 output rows (`4 × 4 = 16` ZMM accumulators) pinned in `Z0`–`Z15` across the entire `activeKVLen` loop, loading each $V$ row (`4` ZMMs into `Z16`–`Z19`) once from L1/L2 cache and accumulating into all 4 query outputs via `VFMADD231PS`.
   - **Microbenchmark Speedup (`BenchmarkSDPAAVX512`)**: **`198.5 µs`** vs `294.6 µs` on `BenchmarkSDPAAVX2` (**`1.48×` faster than AVX2** and **`~3.5×` faster** than previous AVX-512).

2. **Priority 2 — Zero-Spill AVX-512 `BinaryMidArch` Assembly (`C == 64` & `C == 128` Register-Cached Fast Paths), 4× Unrolled `BinaryTrailing`, and `StridedCopy` ([`internal/gobackend/ops/avx512/binary_mid_amd64.s`](../internal/gobackend/ops/avx512/binary_mid_amd64.s), [`binary_trailing_amd64.s`](../internal/gobackend/ops/avx512/binary_trailing_amd64.s), [`stridedcopy_amd64.go`](../internal/gobackend/ops/avx512/stridedcopy_amd64.go))**:
   - Implemented `binaryMid{Add,Sub,Mul,Div,RSub,RDiv}Float32AVX512` and registered `DispatchBinaryMidAVX512` at `PriorityAVX512`:
     - **`C == 64` fast path**: Preloads all 64 broadcast floats (`[a, 0, :64]`) into `Z8`–`Z11` once per outer `a` and streams across `B` rows **2 rows at a time** (`128` floats per inner iteration).
     - **`C == 128` fast path**: Leverages AVX-512's 32 ZMM registers to preload all 128 broadcast floats (`[a, 0, :128]`) into `Z8`–`Z15` once per outer `a` and stream across all `B` rows without reloading the broadcast operand.
     - **General `C` path**: 64-float (`4×` ZMM) unrolled inner loop + 16-float ZMM + 8-float YMM + scalar tail.
   - Unrolled `binaryTrailing{Add,Sub,Mul,Div,RSub,RDiv}Float32AVX512` in `binary_trailing_amd64.s` 4× (64 floats per iteration using `Z1`–`Z4`).
   - Extended `DispatchDupStridedBytesAVX512` and `DispatchConcat2StridedBytesAVX512` to handle `16..63` byte blocks via XMM/YMM registers and `64..256` byte blocks via ZMM registers.

3. **Priority 3 — Handwritten AVX-512 `SwiGLU`, `Sigmoid`, and `Tanh` Assembly ([`internal/gobackend/activations/avx512/swiglu_amd64.s`](../internal/gobackend/activations/avx512/swiglu_amd64.s), [`swiglu.go`](../internal/gobackend/activations/avx512/swiglu.go), [`silu_amd64.s`](../internal/gobackend/activations/avx512/silu_amd64.s))**:
   - Implemented `swigluFloat32AVX512Asm` processing **32 `float32` elements (`2×` 512-bit ZMM vectors) per iteration** with all Cephes/Estrin constants hoisted into `Z16`–`Z29` (zero memory-operand reloads inside the loop) and overlapped `VDIVPS` + `VRCP14PS` Newton-Raphson refinement, plus `SwiGLUAVX512Float64`, `SwiGLUAVX512BF16`, and `SwiGLUAVX512F16`.
   - Added handwritten 32-float (2× ZMM) `sigmoidFloat32AVX512Asm` and `tanhFloat32AVX512Asm` in `silu_amd64.s`, and tuned `AVX512SimpleMetadata` (`MinParallelChunk: 65536`) and `AVX512TranscendentalMetadata` (`MinParallelChunk: 32768`).
   - **Microbenchmark Speedup (`Large_65536/SwiGLU`)**: **`3.25 µs`** on `AVX512` vs `16.87 µs` before (**`5.19×` faster**, and **`2.01×` faster** than `AVX2` at `6.55 µs`).

4. **Priority 4 — AVX-512 MatMul L2 Sub-Chunking, Cooperative RHS Packing, `PackLHSKernelRows8` Half/Double Assembly, and `small_transposed` `Tile4x1` Assembly ([`internal/gobackend/dot/matmul/avx512/large.go`](../internal/gobackend/dot/matmul/avx512/large.go), [`pack_amd64_float64.s`](../internal/gobackend/dot/matmul/avx512/pack_amd64_float64.s), [`pack_amd64_float16.s`](../internal/gobackend/dot/matmul/avx512/pack_amd64_float16.s), [`small_transposed_amd64.s`](../internal/gobackend/dot/matmul/avx512/small_transposed_amd64.s))**:
   - Ported L2 `maxMPanelsPerSubChunk` sub-chunking, `sharedRHSPanels` buffer-allocation skipping (`needPackedRHS = false`), and multi-worker cooperative `sharedRHSPanels` packing to `matmul/avx512/large.go` (and regenerated `gen_large_{bf16,f16,f64}.go`).
   - Implemented `avx512PackLHSKernelRows8Float64Asm`, `avx512PackLHSKernelRows8Float16Asm`, and `avx512PackLHSKernelRows8BFloat16Asm` in assembly, plus `avx512SmallTransposedTile4x1{Float32,Float64,Float16,BFloat16}Asm` for remainder tiles in `small_transposed`.

5. **Priority 5 — AVX-512 `Where` Assembly & `LayerNorm` Pass 3 4× ZMM Unrolling ([`internal/gobackend/ops/avx512/where_amd64.s`](../internal/gobackend/ops/avx512/where_amd64.s), [`internal/gobackend/fusedops/avx512/layernorm_amd64.s`](../internal/gobackend/fusedops/avx512/layernorm_amd64.s))**:
   - Added `where{VV,VS,SV,SS}Float32AVX512Asm` using `VPMOVZXBD` + `VPTESTMD` into `K1`/`K2` + 512-bit masked `VMOVAPS` (32 floats per iteration).
   - Unrolled `pass3_gamma_loop64` (`Float32` gamma-only) and `pass3_64_both_loop32` (`Float64` gamma+beta) 4× in `layernorm_amd64.s`.

### 10.3 Per-Op Profile Comparison on AMD Ryzen 9 9950X3D (`GOMLX_OP_PROFILE=5`, `Timestral-28M`)

| Operation Type | Calls/run | AVX-512 Before Wall (`CPU`) | **AVX-512 After Wall (`CPU`)** | Wall Speedup | CPU Reduction |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`FusedScaledDotProductAttention`** | 27 | 9.8 ms (`129.4 ms`) | **4.2 ms (`44.4 ms`)** | **`2.33×` (`2.85×` on Encoder)** | **`2.91×` (`-85.0 ms` CPU)** |
| **`FusedDense`** | 93 | 17.5 ms (`273.3 ms`) | **16.6 ms (`258.5 ms`)** | **`1.05×`** | **`1.06×` (`-14.8 ms` CPU)** |
| **`Mul`** | 179 | 5.4 ms (`29.8 ms`) | **5.1 ms (`28.4 ms`)** | **`1.06×`** | **`1.05×`** |
| **`Add`** | 65 | 2.6 ms (`17.6 ms`) | **2.2 ms (`14.4 ms`)** | **`1.18×`** | **`1.22×`** |
| **`FusedActivation` (`SwiGLU`)** | 15 | 0.72 ms (`11.1 ms`) | **0.41 ms (`2.1 ms`)** | **`1.76×`** | **`5.29×` (`-9.0 ms` CPU)** |
| **`Concatenate`** | 58 | 0.81 ms (`1.5 ms`) | **0.69 ms (`1.3 ms`)** | **`1.17×`** | **`1.15×`** |
| **`BroadcastInDim`** | 3 | 0.55 ms (`3.3 ms`) | **0.35 ms (`1.8 ms`)** | **`1.57×`** | **`1.83×`** |
| **Total (`Sum of Ops`)** | **1,636** | **42.7 ms (`501.6 ms`)** | **34.7 ms (`382.7 ms`)** | **`1.23×`** | **`1.31×` (`-118.9 ms` CPU)** |

### 10.4 End-to-End Benchmark Comparison on AMD Ryzen 9 9950X3D (`Timestral-28M` & `Timestral-123M`, 10 Runs)

| Model | Backend / Configuration | Best Wall Time | Typical Wall Time | Best CPU Time | Avg Cores | Max Abs Diff vs PyTorch | Speedup vs `xla:cpu` | Speedup vs AVX2-Only |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Timestral-28M`** | `xla:cpu` Reference (`-backend="xla:cpu"`) | 51.8 ms | 52.1–58.3 ms | 702.1 ms | 13.5 | 2.1e-05 | `1.00×` | `0.92×` |
| **`Timestral-28M`** | `go` AVX2-Only (`GOMLX_GO_SIMD_AVX512=0`) | 47.8 ms | 48.5–52.7 ms | 712.9 ms | 14.9 | 3.4e-05 | `1.08×` | `1.00×` |
| **`Timestral-28M`** | `go` AVX-512 Before This Pass | 40.5 ms | 44.9–49.3 ms | 454.2 ms | 11.2 | 3.4e-05 | `1.28×` | `1.18×` |
| **`Timestral-28M`** | **`go` AVX-512 After Priorities 1–5** | **35.8 ms** | **39.0–42.7 ms** | **374.5 ms** | **10.5** | **3.4e-05** | **`1.45×` wall (`1.87×` less CPU)** | **`1.34×` wall (`1.90×` less CPU)** |
| **`Timestral-123M`** | `xla:cpu` Reference (`-backend="xla:cpu"`) | 163.9 ms | 164.0–169.6 ms | 2,770.0 ms | 16.9 | 1.9e-05 | `1.00×` | `0.95×` |
| **`Timestral-123M`** | `go` AVX2-Only (`GOMLX_GO_SIMD_AVX512=0`) | 155.6 ms | 156.8–164.7 ms | 3,080.0 ms | 19.8 | 1.9e-05 | `1.05×` | `1.00×` |
| **`Timestral-123M`** | **`go` AVX-512 After Priorities 1–5** | **124.8 ms** | **125.1–128.5 ms** | **2,050.0 ms** | **16.4** | **1.9e-05** | **`1.31×` wall (`1.35×` less CPU)** | **`1.25×` wall (`1.50×` less CPU)** |

#### Detailed Run-by-Run After AVX-512 Optimizations (`-backend="go"`, AMD Ryzen 9 9950X3D)

##### `mistralai/Timestral-28M`
| Obs | Model | Run | Wall | CPU Time | Cores | Live Mem | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| `gomlx-simd` | `Timestral-28M` | warm-up | 52.8ms | 601.5ms | 11.4 | 0.25 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 1 | 35.8ms | 374.5ms | 10.5 | 0.27 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 2 | 41.3ms | 440.1ms | 10.7 | 0.29 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 3 | 42.1ms | 446.6ms | 10.6 | 0.32 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 4 | 40.3ms | 419.6ms | 10.4 | 0.27 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 5 | 40.8ms | 414.4ms | 10.2 | 0.29 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 6 | 39.7ms | 407.7ms | 10.3 | 0.31 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 7 | 42.7ms | 453.3ms | 10.6 | 0.34 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 8 | 41.0ms | 426.2ms | 10.4 | 0.28 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 9 | 39.0ms | 413.2ms | 10.6 | 0.31 GB | 0.36 GB |
| `gomlx-simd` | `Timestral-28M` | run 10 | 39.8ms | 409.5ms | 10.3 | 0.33 GB | 0.36 GB |
| **`gomlx-simd`** | **`Timestral-28M`** | **`best`** | **`35.8ms`** | **`374.5ms`** | **`10.5`** | **`0.33 GB`** | **`0.36 GB`** |

##### `mistralai/Timestral-123M`
| Obs | Model | Run | Wall | CPU Time | Cores | Live Mem | Max RSS |
| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |
| `gomlx-simd` | `Timestral-123M` | warm-up | 169.3ms | 2.9s | 17.0 | 0.65 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 1 | 125.9ms | 2.1s | 16.5 | 0.71 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 2 | 127.0ms | 2.1s | 16.5 | 0.76 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 3 | 124.8ms | 2.1s | 16.4 | 0.59 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 4 | 128.5ms | 2.1s | 16.6 | 0.65 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 5 | 128.3ms | 2.1s | 16.4 | 0.70 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 6 | 126.9ms | 2.1s | 16.6 | 0.76 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 7 | 126.7ms | 2.1s | 16.4 | 0.61 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 8 | 125.1ms | 2.1s | 16.5 | 0.67 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 9 | 126.5ms | 2.1s | 16.5 | 0.72 GB | 0.87 GB |
| `gomlx-simd` | `Timestral-123M` | run 10 | 126.7ms | 2.1s | 16.4 | 0.78 GB | 0.87 GB |
| **`gomlx-simd`** | **`Timestral-123M`** | **`best`** | **`124.8ms`** | **`2.1s`** | **`16.4`** | **`0.78 GB`** | **`0.87 GB`** |




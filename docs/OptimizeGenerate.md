# Optimizing Autoregressive LLM Generation (`Gemma 4 E2B` BF16) in GoMLX (`gobackend`)

## Executive Summary

This document tracks the profile-driven optimization of single-stream (`batch=1`) autoregressive LLM generation in GoMLX's pure-Go CPU backend (`GOMLX_BACKEND=go`), benchmarked on **`google/gemma-4-E2B-it` (`BFloat16`)** over the `databricks/databricks-dolly-15k` dataset against **Ollama / `llama.cpp` CPU (`gemma4:e2b-it-bf16`)**.

Across the optimization milestones so far, **decode throughput improved 5.9×** (`1.84` → `10.85 tok/s` at `par=16` / `10.70 tok/s` default `par=32`), **prefill throughput on long prompts improved ~2.3×** (`287` → `675 tok/s`), and **peak RSS dropped by ~15–20%** (`18.5 GiB` → `16.0 GiB`), while maintaining **9/10 exact character-for-character greedy output matches** against `llama.cpp`:

| Milestone | Example #1 Cold TTFT (`132 tok`) | Example #5 Warm Prefill (`356 tok`) | Decode Throughput (`tok/s`) | Decode `CPUTime/tok` (`avg_cores`) | Peak `MaxRSS` (10 ex.) | Key Change |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **0. Baseline** (`bf9d162` / `b2971de`) | 866 ms *(uncached)* | — | **1.84 tok/s** | 2.8 s (`5.20` cores) | — | Initial Gemma 4 E2B BF16 (`use_double_wide_mlp`); `avx512LargeMulti` single-threaded on `M=1` |
| **1. Parallel `M=1` Decode + `DonateStore`** (`3d7a574` / `7aab5d70`) | 1.9 s *(incl. prepack)* | 1,240 ms (`287 tok/s`) | **9.64–9.77 tok/s** (**5.3×**) | 1.5 s (`14.7` cores) | 18.5 → 15.1 GiB | Fallback from `avx512LargeMulti` when `M=1` so `avx512Large` splits along `N`; `WithDonateStore(true)` avoids per-token weight re-packing & drops 3.4 GiB RSS |
| **2. `OnlyLastTokenLogits` + Native `AVX512_BF16` (`VDPBF16PS`)** (`08b0c62` / `84d082ba` / `c9b0027`) | 1.1 s *(incl. prepack)* | 787 ms (`452 tok/s`) | **10.31–10.48 tok/s** (**5.7×**) | 1.4 s (`14.4` cores) | 14.7 GiB | Skip unused prompt `lm_head` rows (`[1, 356, 1536] → [1, 1, 1536]`); native `VDPBF16PS` K2-interleaved BF16 GEMM kernels |
| **3. Zero-Copy KV-Cache Decode + Pre-Allocated Cache** (`91b702f0` / `d45a21c`) | 1.2 s *(incl. prepack)* | 942 ms (`378 tok/s`) *(regression from 2× larger prefill bool mask)* | **9.93–10.21 tok/s** (10-ex avg) | 1.4 s (`13.8–14.3` cores) | 16.9 GiB *(grew with static prompt graphs)* | Pre-allocate KV cache up to `MaxLength` (no mid-decode `PadTensors` / recompiles); `WithSeqLens` zero-copy decode; `sdpaMultiHeadHalf` converts only `kvLimit` rows; bulk `DynamicUpdateSlice` prefill |
| **4. Step 1 — Prefill Quick Wins** (`5a5e8c85` / `43af1d6`) | 1.1 s *(incl. prepack)* | **564 ms (`631 tok/s`)** (**+67%**) | **10.54 tok/s** (10-ex avg, **5.7×**) | 1.4 s (`14.47` cores) | **16.0 GiB** (`-0.9 GiB`) | Zero-copy `WithSeqLens` + `WithCausalMask(true)` prefill; head-invariant `[B, Sq, 1, Skv]` masks (no bool transpose); `WithDynamicAxes` on `promptExec`; cap local KV cache to `sliding_window=512` |
| **5. Step 2 — Decode Parallelism & Buffer Tightening** *(uncommitted)* | 1.1 s *(incl. prepack)* | **561 ms (`635 tok/s`)** *(par=32)* / 585 ms *(par=16)* | **10.70 tok/s** *(default par=32)* / **10.85 tok/s** *(par=16, **5.9×**)* | 1.6 s (`16.61` cores) / **1.2 s** (`13.56` cores) | 16.9 GiB *(par=32)* / **16.0 GiB** *(par=16)* | Balanced 1D `N`-split (`128`-col panel-aligned chunks on `M<=mr`); `MinCachedLargeMatMulFlopsPerWorker=128K`; single shared pre-packed LHS for `M<=mr`; `accumOutput` `4 MiB → 3–16 KiB/worker` |
| **Reference: Ollama (`llama.cpp` CPU, `gemma4:e2b-it-bf16`)** | 300 ms *(warm)* | 545 ms (`654 tok/s`) | **11.86–12.27 tok/s** | 1.36 s (`15.93` cores) | ~11.4 GiB | 16 physical threads (`n_threads=16`), spin-wait barrier, GEMV `M=1` BF16 dot kernel |

---

## 1. Workload, Hardware & Reproduction

### Hardware & Toolchain
- **CPU**: AMD Ryzen 9 9950X3D (Zen 5, 16 physical cores / 32 threads, 48 KiB L1d/core, 1 MiB L2/core, 96 MiB 3D V-Cache CCD0 + 32 MiB L3 CCD1)
- **SIMD**: `AVX2`, `FMA`, `AVX-512` (`AVX512F`, `AVX512BW`, `AVX512DQ`, `AVX512VL`, `AVX512_VNNI`, `AVX512_BF16`)
- **Memory**: 59 GiB DDR5 (~56–60 GB/s sustained cold-DRAM read bandwidth measured on 805 MB `lm_head`)
- **Go**: `go1.27.3 linux/amd64` with `GOEXPERIMENT=simd`

### Model Profile (`google/gemma-4-E2B-it`, `BFloat16`)
- **Architecture**: 35 layers (`d_model=1536`, `num_heads=8`, `num_kv_heads=1`), hybrid Local (`sliding_window=512`, `head_dim=256`) and Global (`head_dim=512`, proportional partial RoPE `0.25`) attention, KV-shared layers (`layer >= 15`) with double-wide MLP (`ffn_dim=12288` vs `6144` on layers `0..14`), Per-Layer Embeddings (PLE, `256`-dim), tied embeddings (`vocab_size=262144`).
- **Active weights per token**:
  - 35-layer MLP + attention + PLE weights: **`~3.42 GB`** (`BFloat16`)
  - `lm_head` (`[262144, 1536]`): **`~0.805 GB`** (`BFloat16`)
  - **Total read per decoded token**: **`~4.22 GB/token`** (far exceeding L3 cache → strictly DDR5 bandwidth-bound during `M=1` decode).

### Benchmark & Profiling Commands
Run from `/home/janpf/Projects/gomlx/go-huggingface`:

```bash
# Side-by-side 10-example benchmark vs Ollama (BF16 CPU)
GOEXPERIMENT=simd GOMLX_BACKEND=go go run ./examples/dolly/browse \
  -model=google/gemma-4-E2B-it -ex=1 -n=10 -gen_tokens=64 \
  -ollama=gemma4:e2b-it-bf16 -mmap=false

# Single-example focused prefill/decode profiling
GOEXPERIMENT=simd GOMLX_BACKEND=go go run ./examples/dolly/browse \
  -model=google/gemma-4-E2B-it -ex=4 -n=2 -gen_tokens=32 -mmap=false \
  -cpuprof=/tmp/gemma4.prof
```

---

## 2. Completed Optimizations & Gains So Far

### Milestone 0 → 1: Fixing Single-Threaded `M=1` Decode & Weight Donation (`1.84 → 9.64 tok/s`, **5.3×**)
- **Root causes found via CPU profiling (`67.6s` samples over `11.2s`, `avg_cores=5.20`):**
  1. `avx512LargeMultiBFloat16` ([compute/internal/gobackend/dot/matmul/avx512/large.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/dot/matmul/avx512/large.go#L521-L528)) used `matmul.AppendRowWorkItems`, which splits **only along `batch` and `M`**. For `batch=1, M=1` decode, every `FusedDense` with a fused activation (`gate_proj` + GELU across all 35 layers) ran on **1 core** (~7.4s out of 10.3s decode wall time).
     - **Fix (`3d7a574`)**: Return `false` from `avx512LargeMulti*` when `len(workItems) < maxWorkers/2+1` and `totalRHSCrossSize >= 2*params.RHSL1KernelCols`, falling back to `avx512Large*` which splits across `N`.
  2. **61.8% of CPU time** was spent re-packing BF16 weight matrices on every token because `promptExec` and `genExec` did not donate the `model.Store` buffers into `gobackend.ConstantData`.
     - **Fix (`7aab5d70`)**: Added `Generator.WithDonateStore(bool)`. Donated weights are pre-packed once on the first execution and reused across prefill and decode, freeing raw weight buffers (`MaxRSS` dropped `18.5 GiB → 15.1 GiB`). Also fixed a skipped-tensor leak in `LoadStoreFiltered` (`df9d9b5`).

### Milestone 1 → 2: `OnlyLastTokenLogits` & Native `AVX512_BF16` (`VDPBF16PS`) (`prefill 287 → 452 tok/s`, `decode 9.7 → 10.48 tok/s`)
- **Correctness fixes (`968af69a`, `38cbdd6`)**: Implemented proportional partial RoPE (`WithPartialRotaryFactor(0.25)`) with Float32 RoPE angles for half-precision inputs, and removed an erroneous `attnLogitCap=50.0` on Gemma 4 text attention.
- **Skip unused prompt `lm_head` projections (`84d082ba`, `c9b0027`)**: During prefill of a `356`-token prompt, `LogitsFromEmbeddings` was projecting `[1, 356, 1536] × [262144, 1536]^T` (`~287 GFlops` + `~373 MB` logits buffer) and discarding rows `0..354`. Added `CallOptions.OnlyLastTokenLogits`, slicing `[1, 356, 1536] → [1, 1, 1536]` before `lm_head`.
- **Native `AVX512_BF16` (`VDPBF16PS`) kernels (`08b0c62`)**: Added K2-interleaved panel packing and native `VDPBF16PS` `8×32` microkernel (`494 GFlops/s` single-core, `6.87 TFLOP/s` on `2048³` multi-core).
  - **Result**: Example #5 (`356 tok`) warm prefill jumped from **`1,240 ms` (`287 tok/s`) → `787 ms` (`452 tok/s`)**, and decode rose to **`10.31–10.48 tok/s`**.

### Milestone 2 → 3: Zero-Copy KV-Cache Decode (`91b702f0`, `d45a21c`)
- **Changes**:
  - Pre-allocated KV caches up to `MaxLength` (capped at `4096`) to eliminate mid-decode `PadTensors` copies and graph recompilations; fixed an off-by-one bug in `PadTensors(kvCacheTensors, position)` → `position+1` (improving greedy accuracy to **9/10 exact matches** vs. Ollama).
  - Implemented zero-copy `WithSeqLens` in `buildCachedAttention` for `qSeqLen == 1` decode, skipping dense boolean masks.
  - Updated `sdpaMultiHeadHalf` (`d45a21c`) so `localKV` converts only `kvLimit` rows of K/V to Float32 instead of all padded rows.
  - Skipped redundant `TransposeAllDims` when `kvHeads == 1` and added a bulk `DynamicUpdateSlice` fast path in `KVCache.Update` for prefill at `position == 0`.
- **Observed trade-off**: Because pre-allocating the KV cache increased `cacheSeqLen` from `512` to `1024` during prefill, the still-masked prefill path doubled its `[1, 356, 8, cacheSeqLen]` boolean mask size (`1.45 MB → 2.9 MB` × 35 layers), causing a temporary prefill regression (`787 ms → 942 ms` on Example #5) that directly motivated Step 1 below.

---

## 3. Root-Cause Analysis: GoMLX vs. `llama.cpp` (Before Optimization Steps 1–4)

### Why Sentence Encoding (`yzma_embedder`) Beat `llama.cpp` by 1.5–8× While Single-Stream GenAI Trailed
In [`benchmarks/yzma_embedder/README.md`](file:///home/janpf/Projects/gomlx/benchmarks/yzma_embedder/README.md), GoMLX ran with **`-bucket=0 -num_embedders=32 -dynamic=true`**:
1. **Inter-call vs. intra-op parallelism**: 32 independent single-threaded sequences saturated all 32 hardware threads with **zero** intra-op barriers and full `M ≈ 77` GEMMs. Single-stream GenAI (`batch=1`) has only 1 sequence, so all parallelism must come from intra-op splitting.
2. **Dynamic shapes**: `benchmark_embed` compiled **once** via `WithDynamicAxes`; `generate.go`'s `promptExec` previously recompiled a static graph for every new prompt length.
3. **Cache residency vs. cold DDR5 DRAM**: MiniLM (22M) and BGE-base (110M) largely fit in the 96 MiB 3D V-Cache, whereas Gemma 4 E2B BF16 streams `~4.22 GB/token` from cold DRAM on every decode step.

### Why Ollama Used ~20% More Cores (Measured Side-by-Side)

| Workload | Ollama (`llama-server`) | GoMLX (`parallelism=32`) | GoMLX (`parallelism=16, GOMAXPROCS=16`) |
| :--- | ---: | ---: | ---: |
| **Decode** (`64 tok`) | **`15.93` cores** (`12.27 tok/s`, `81.5 ms/tok`) | **`14.2` cores** (`10.0–10.2 tok/s`) | **`9.2` cores** (`10.2–10.4 tok/s`) |
| **Prefill** (`~360 tok`) | **`15.26` cores** (`537 ms`, `672 tok/s`) | **`7.9` cores** (`931 ms`, `382 tok/s`) | **`~7.9` cores** (`932 ms`, `382 tok/s`) |

1. **Cold-DRAM microbenchmark proves the `VDPBF16PS` matmul kernel is already fast enough to beat `llama.cpp`:**

| Gemma 4 Shape (Cold DRAM, `360 MB` streaming) | `parallelism=16` | `parallelism=32` | Notes |
| :--- | ---: | ---: | :--- |
| `Prefill_gate_12288 [356 × 1536 × 12288]` | **3.15 ms** (`4,260 GFlops/s`) | 3.46 ms (`3,885 GFlops/s`) | All 35 prefill layers sum to **`~320 ms`** vs Ollama's `537 ms` |
| `Prefill_down_12288 [356 × 12288 × 1536]` | 3.85 ms (`3,488 GFlops/s`) | 3.47 ms (`3,875 GFlops/s`) | |
| `Decode_gate_12288 [1 × 1536 × 12288]` | **667 µs** (**`56.6 GB/s`**) | 749 µs (`50.4 GB/s`) | 16 workers; exceeds Ollama's `~50 GB/s` effective decode BW |
| `Decode_down_12288 [1 × 12288 × 1536]` | 751 µs (`50.2 GB/s`) | 745 µs (`50.7 GB/s`) | **Starved at 4 workers** by `Choose2DSplit` (`colChunk < 256`) |
| `Decode_lm_head [1 × 1536 × 262144]` (`805 MB`) | **13.5 ms** (**`59.8 GB/s`**) | 13.7 ms (`58.9 GB/s`) | Full DDR5 read bandwidth (`4.22 GB / 57 GB/s ≈ 74 ms = 13.5 tok/s`) |

2. **Prefill bottleneck breakdown (`931 ms` on Example #5, verified via `prefill_new.prof` & `GOMLX_BACKEND=go:no_ops` = `336.8 ms`):**
   - **`~170 ms` single-threaded**: Per-length static graph build & compile in `promptExec` (no `WithDynamicAxes`).
   - **`~165–310 ms` single-threaded**: `WithQueryKeyMatrixMask` broadcast `[1, 356, 1024]` to `[1, 356, 8, 1024]` (`102 MB` across 35 layers), triggering single-threaded generic byte-by-byte `execTransposeGeneric[bool]` in `execFusedScaledDotProductAttention`.
   - **`~320–420 ms` multi-core**: Actual 35-layer BF16 matmuls + `lm_head` + `Float32→BFloat16` conversion (`250 ms` CPU time) + LHS packing (`380 ms` CPU time).

3. **Decode bottleneck breakdown (`98 ms/tok`, verified via `decode_new.prof` & `go:no_ops` = `1.16 ms/tok`):**
   - **Narrow-`N` core starvation**: `Choose2DSplit` (`nosimd.go:239`) rejects `colChunk < 8 * minCol` (`256` cols), capping every `down_proj`, `o_proj`, and `per_layer_projection` (`N=1536`, >33% of all model weights!) at **4 workers**, `q_proj` (`N=2048`) at **8 workers**, and `k/v_proj` at **1–2 workers**.
   - **8× wasted work on `M=1` GEMM**: `avx512LargeKernelBFloat16BF16Asm` accounts for **94.3% of all decode CPU time** (`86.5s / 91.8s`). On `M=1`, its `8×32` kernel executes 16 `VDPBF16PS` + 8 `VPBROADCASTD` per K-pair (7/8 rows are zeros), writes into `packedOutput`/`accumBuffer`, and copies row 0 out via `avx512ApplyPackedOutputFloat32`.
   - **SMT contention & barrier wakeup**: Running 32 SMT threads on 16 Zen 5 physical cores (which share 512-bit FPU execution units and L1/L2) increases `CPUTime/tok` by ~55% (`896 ms` at 16 threads vs `1,400 ms` at 32 threads) and slows cold-DRAM `Decode_gate_12288` (`56.6 GB/s → 50.4 GB/s`).

---

## 4. Prioritized Optimization Roadmap

### Step 1 — Prefill Quick Wins *(High ROI, Targets `~400–550 ms` on 356-tok prompt)*
1. **Zero-copy `WithSeqLens` + `WithCausalMask(true)` for prefill** in [`gomlx/ml/zoo/transformer/transformer.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/zoo/transformer/transformer.go) & [`gomlx/ml/layers/attention/masks.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/layers/attention/masks.go): avoid materializing `[B, Sq, Skv]` masks at `position == 0` whenever `qSeqLen <= cacheSeqLen` (and `qSeqLen <= slidingWindow` for local layers), letting fused SDPA stop each row at `min(qSeqLen, qIdx+1)`.
2. **Keep head-invariant masks at `heads=1`** in [`WithQueryKeyMatrixMask`](file:///home/janpf/Projects/gomlx/gomlx/ml/layers/attention/masks.go): remove `BroadcastToDims` over `numHeads` so explicit masks stay `[B, Sq, 1, Skv]`, eliminating `numHeads×` mask memory and avoiding `execTransposeGeneric[bool]` in `gobackend` SDPA.
3. **Dynamic `seq_len` (`WithDynamicAxes`) on `promptExec`** in [`gomlx/ml/zoo/transformer/generate/generate.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/zoo/transformer/generate/generate.go), with dynamic `seq_len` support in `KVCache.Update` and `OnlyLastTokenLogits`: compile the prefill graph **once** for all prompt lengths `<= minCacheCap`, saving `~120–170 ms` per new prompt length and preventing RSS growth across prompts.

### Step 2 — Decode Parallelism Fix *(Est. +12–18% decode `tok/s` → `~11.5–12.0 tok/s`)*
1. **Relax `Choose2DSplit` `colChunk` floor for small `M` (`M <= 8`)** in [`compute/internal/gobackend/dot/matmul/nosimd.go`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/dot/matmul/nosimd.go): allow `colChunk` down to `1–2 * minCol` (`32–64`) when `M` cannot supply row parallelism, so `down_proj`, `o_proj`, `q_proj`, `k/v_proj`, and `per_layer_*` use all 16 physical cores.
2. **Lower `MinLargeMatMulFlopsPerWorker` when RHS is pre-packed/cached**: cached weights have zero packing overhead to amortize, so smaller projections (`k_proj`, `v_proj`, `per_layer_input_gate`, `per_layer_projection`) can parallelize effectively across more workers.

### Step 3 — Dedicated `M=1` (and `M=1..4`) `VDPBF16PS` GEMV Microkernel *(Est. +10–15% decode `tok/s` → `~12.5–13.5 tok/s`, 4–6× less CPU/watt)*
- Add a 1-row (`1×64` or `1×128`) `VDPBF16PS` microkernel in [`compute/internal/gobackend/dot/matmul/avx512/large_amd64_bfloat16_bf16.s`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/dot/matmul/avx512/large_amd64_bfloat16_bf16.s) reading the **exact same K2-packed RHS panels** (zero change to cached weight layout):
  - 1 `VPBROADCASTD` shared across 2–4 RHS strips (64–128 cols) with `PREFETCHT0` software prefetching.
  - Writes directly to `outputMatrix` (no `packedOutput`, no `accumBuffer`, no `avx512ApplyPackedOutputFloat32`, no `8`-row zero-padded LHS packing).

### Step 4 — Physical-Core Parallelism Tuning & Scheduler Spin-Wait *(Est. +3–8% decode `tok/s`, ~35% less CPU time)*
- On Zen 5 (SMT shares 512-bit AVX-512 FPU and L1/L2), evaluate defaulting or tuning `gobackend` large-matmul parallelism to physical core count (`16` vs `32`) once Step 2 unlocks all 16 cores on narrow-`N` layers.
- Add a brief bounded spin-wait in [`scheduler.go`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/scheduler/scheduler.go) before parking `workerLoop` goroutines to cut barrier wake-up latency across ~350 ops/token.

### Step 5 — Prefill Epilogue & Multi-RHS Fusion *(Est. -55–90 ms on 356-tok prefill)*
1. **Fuse `Float32 → BFloat16` conversion into the matmul output store/epilogue** in [`execFusedDenseHalf`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/dense/dense.go), removing the `tmpF32` DRAM round-trip and trailing `ParallelConvertSlice` barrier (`250 ms` CPU time in prefill).
2. **Multi-RHS fusion for `gate_proj` + `up_proj` (and `q/k/v_proj`)** to pack LHS once (`380 ms` CPU time in prefill) and cut scheduler barriers.

---

## 5. Step 1 Implementation & Measured Results

### Changes Implemented
1. **Head-invariant masks without head broadcast** ([`gomlx/ml/layers/attention/masks.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/layers/attention/masks.go), [`multiheadattention.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/layers/attention/multiheadattention.go)):
   - `WithQueryKeyMatrixMask` now inserts a size-1 heads axis (`[B, Sq, 1, Skv]`) instead of calling `BroadcastToDims` across `numHeads`, saving `numHeads×` mask memory, preserving dynamic sequence dimensions, and skipping the single-threaded `execTransposeGeneric[bool]` in `gobackend` SDPA.
   - Relaxed `WithCausalMask` and `buildCausalAttentionMask` to validate matching batch size instead of requiring `qSeqLen == kSeqLen`, enabling causal masking against pre-allocated KV caches (`kvLen >= qSeqLen`).
2. **Zero-copy prefill via `WithSeqLens` + `WithCausalMask(true)`** ([`gomlx/ml/zoo/transformer/transformer.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/zoo/transformer/transformer.go), [`gomlx/ml/layers/attention/kvcache/kvcache.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/layers/attention/kvcache/kvcache.go), [`go-huggingface/models/transformer/transformer.go`](file:///home/janpf/Projects/gomlx/go-huggingface/models/transformer/transformer.go)):
   - In `buildCachedAttention`, contiguous prefill at constant `position == 0` with `qSeqLen <= cacheSeqLen` (and `qSeqLen <= slidingWindow` for local layers) now uses `WithSeqLens(qSeqLen, qSeqLen)` + `WithCausalMask(true)` — zero dense mask materialized and SDPA stops each row at `min(qSeqLen, qIdx+1)`.
   - Capped `LocalLayer` KV cache `MaxSeqLenPerLayerType` to `m.Config.SlidingWindow` (`512` for Gemma 4 instead of default `1024`), halving local KV cache memory and activating zero-copy prefill on all 35 layers.
   - Made `KVCache.BuildAttentionMask` dynamic-shape safe (`DynamicIota` + `DynamicBroadcastInDim`) and allowed bulk `DynamicUpdateSlice` in `KVCache.Update` for dynamic `seq_len` at `position == 0` regardless of `SinkPositions` (since non-wrapping writes never trigger sinks).
3. **Dynamic `seq_len` (`WithDynamicAxes`) on `promptExec`** ([`gomlx/ml/zoo/transformer/generate/generate.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/zoo/transformer/generate/generate.go), [`transformer.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/zoo/transformer/transformer.go)):
   - Added `promptExecDynamic` configured with `WithDynamicAxes([]string{"", "seq_len"})` when `backend.Capabilities().HasDynamicShapes()` and `promptLen <= minCacheCap` (falling back to static `promptExec` for prompts longer than a local sliding window so ring-buffer wrapping stays exact).
   - Updated `OnlyLastTokenLogits` and `promptExec` last-token extraction to use zero-transpose `DynamicReshape` + `Gather` on dynamic `seq_len`.
   - Added unit test `TestZeroCopyDynamicPrefill` in [`transformer_test.go`](file:///home/janpf/Projects/gomlx/gomlx/ml/zoo/transformer/transformer_test.go).

### 10-Example Benchmark Results (`google/gemma-4-E2B-it` BF16, `-gen_tokens=64`, `-mmap=false`)

Comparing **Before Step 1** vs. **After Step 1** vs. **Ollama (`llama.cpp` CPU, cold-prompt-cache baseline)**:

| Ex # | Prompt Tokens | Before Step 1 TTFT | After Step 1 TTFT | Step 1 Prefill Speedup | Ollama Prefill *(cold cache)* | Before Decode | After Decode | Ollama Decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| #1 *(cold prepack)* | 132 | 1,234 ms | 1,116 ms | +10.6% | 300.1 ms *(warm)* | 9.96 tok/s | 10.67 tok/s | 12.26 tok/s |
| #2 | 19 | 192.3 ms | **170.8 ms** | +12.6% | 180.5 ms | 10.03 tok/s | 10.82 tok/s | 11.98 tok/s |
| #3 | 18 | 200.3 ms | **168.8 ms** | +18.7% | 171.5 ms | 10.06 tok/s | 10.63 tok/s | 11.91 tok/s |
| #4 | 33 | 245.8 ms | **209.1 ms** | +17.6% | 175.9 ms | 10.04 tok/s | 10.69 tok/s | 12.01 tok/s |
| #5 | 356 | 941.8 ms (`378 tok/s`) | **564.3 ms (`631 tok/s`)** | **+66.9%** | 544.5 ms (`654 tok/s`) | 9.87 tok/s | 10.39 tok/s | 12.13 tok/s |
| #6 | 218 | 638.1 ms (`342 tok/s`) | **416.7 ms (`523 tok/s`)** | **+53.1%** | 374.8 ms (`582 tok/s`) | 9.88 tok/s | 10.44 tok/s | 11.75 tok/s |
| #7 | 234 | 677.2 ms (`346 tok/s`) | **435.4 ms (`537 tok/s`)** | **+55.5%** | 392.0 ms (`597 tok/s`) | 9.90 tok/s | 10.45 tok/s | 11.80 tok/s |
| #8 | 21 | 196.9 ms | **165.3 ms** | +19.1% | 192.5 ms | 10.02 tok/s | 10.74 tok/s | 11.86 tok/s |
| #9 | 15 | 152.0 ms | **126.4 ms** | +20.3% | 173.8 ms | 9.97 tok/s | 10.64 tok/s | 11.59 tok/s |
| #10 | 445 | 1,165.2 ms (`382 tok/s`) | **665.9 ms (`668 tok/s`)** | **+75.0%** | 613.5 ms (`725 tok/s`) | 9.68 tok/s | 10.20 tok/s | 11.91 tok/s |
| **Warm Prefill (#2–#10)** | **1,359** | **4,409.6 ms (`308.2 tok/s`)** | **2,922.7 ms (`465.0 tok/s`)** | **+50.9%** | **2,819.0 ms (`482.1 tok/s`)** | — | — | — |
| **10-Ex Decode Avg** | 487 gen | — | — | — | — | **9.93 tok/s** | **10.54 tok/s (+6.1%)** | **11.86–11.92 tok/s** |
| **Peak `MaxRSS`** | — | 16.9 GiB | **16.0 GiB (`-0.9 GiB`)** | — | ~11.4 GiB | — | — | — |

### Key Takeaways from Step 1
- **Warm prefill across Examples #2–#10 jumped `+50.9%`** (`308.2 → 465.0 tok/s`), reaching **`96.5%` of Ollama/`llama.cpp`'s warm prefill throughput (`482.1 tok/s`)** and **beating Ollama on short prompts (#2, #3, #8, #9 — up to 27% faster on #9)**.
- **Long prompts saw the biggest gains:** `+67%` on #5 (`942 → 564 ms`, `631 tok/s`) and **`+75%` on #10** (`1,165 → 666 ms`, `668 tok/s`).
- **Decode also gained `+6.1%`** (`9.93 → 10.54 tok/s`, now `88.4%` of Ollama's `11.92 tok/s`) from removing the `numHeads` broadcast on explicit decode masks and halving local KV cache size (`1024 → 512`), while **`MaxRSS` dropped by `0.9 GiB`** (`16.9 → 16.0 GiB`) and stopped growing linearly with each new prompt length.
- **Next up (Steps 2 & 3):** Relaxing `Choose2DSplit`'s narrow-`N` floor (`colChunk < 256` currently starving `down_proj`/`o_proj` to 4 workers) and adding a dedicated `M=1` `VDPBF16PS` GEMV microkernel are expected to push decode past **`12.5–13.5 tok/s`** (surpassing `llama.cpp`), while Step 5 (epilogue BF16 conversion + Multi-RHS fusion) pushes long prefill past **`800+ tok/s`**.

---

## 6. Step 2 Implementation, Cold-DRAM Findings & Measured Results

### Changes Implemented
1. **Balanced 1D `N`-splitting & relaxed small-`M` floor** ([`compute/internal/gobackend/dot/matmul/nosimd.go`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/dot/matmul/nosimd.go)):
   - Added `MinCachedLargeMatMulFlopsPerWorker = 128 * 1024` (down from `256 * 1024`) when RHS or LHS panels are pre-packed/cached so smaller projections (`q_proj`, `k/v_proj`, `per_layer_*`) parallelize across more workers.
   - Added a direct 1D `N`-split fast path in `AppendWorkItems` when `totalMBlocks == 1` (`M <= mr`, e.g. `M=1` decode), avoiding `2×` oversubscription and power-of-2 rounding (which created uneven waves like `24` items on `16` workers or `48` tiny 1-strip items on `32` workers). Each chunk is kept at least `4` microkernel strips (`128` cols on AVX-512, dividing `512`-col panels evenly) when `totalNBlocks >= 16` (so `N=1536` splits into `12` exact `128`-col chunks and `N=2048` into `16` exact `128`-col chunks).
   - Relaxed `Choose2DSplit`'s `minColChunk` floor when `maxPossibleM < targetWorkers` for small multi-strip `M` (`9 <= M <= 32`).
2. **Eliminated massive per-worker `M=1` buffer overhead & redundant LHS packing** ([`compute/internal/gobackend/dot/matmul/avx512/large.go`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/dot/matmul/avx512/large.go), [`compute/internal/gobackend/dot/matmul/avx2/large.go`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/dot/matmul/avx2/large.go)):
   - **Shared single-pass LHS pre-pack for `M <= mr` (`batch == 1`)**: previously, all `16–32` workers each allocated a `numKPanels × 32 × 256` (`768 KiB` for `K=12288` → **`24.6 MiB` across 32 workers**) `packedLHS` buffer and redundantly packed the exact same 1-row input vector. Now `lhs` is packed **once** before `SaturateUpTo` into a single compact `numKPanels × mr × paddedPanelK` buffer (`24–192 KiB` total) read-only across all workers via `cachedLHSPanels`.
   - **Tightened `accumOutput` & skipped unused `packedOutput`**: fixed a hardcoded `maxAccumPanels = 64` allocation (`64 × 32 × 512 × 4 = 4 MiB` per worker → **`128 MiB` across 32 workers!**) to `min(64, numRowPanels) × accumAllocRows × accumAllocCols` (**`3–16 KiB` per worker** on `M=1`, fitting inside L1d), and skipped allocating `packedOutput` (`64 KiB/worker`) whenever `alwaysUseAccum` holds (`numKPanels > 1`).
   - Added LHS reuse across stolen work items with identical `(batchIdx, rowStart, rowEnd)` and unit tests (`TestChoose2DSplit`, `TestAppendWorkItemsNarrowM`) in [`internal_test.go`](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/dot/matmul/avx512/internal_test.go).

### Cold-DRAM Microbenchmark Gains (`377 MB` streaming, 10 distinct matrices)

| Shape (Cold DRAM) | Before Step 2 (`par=16`) | After Step 2 (`par=16`) | After Step 2 (`par=12` optimal `down`) | After Step 2 (`par=32`) |
| :--- | ---: | ---: | ---: | ---: |
| `Decode_gate_12288 [1 × 1536 × 12288]` | 667 µs (`56.6 GB/s`) | **640 µs (`59.0 GB/s`)** | 671 µs (`56.3 GB/s`) | 707 µs (`53.4 GB/s`) |
| `Decode_down_12288 [1 × 12288 × 1536]` | 751 µs (`50.2 GB/s`) | **714 µs (`52.8 GB/s`)** | **679 µs (`55.6 GB/s`, +10.8%)** | 757 µs (`49.9 GB/s`) |

### 10-Example Benchmark Results (`google/gemma-4-E2B-it` BF16, `-gen_tokens=64`, `-mmap=false`)

| Ex # | Prompt | Step 1 Decode (`par=32`) | Step 2 Decode (`par=32`) | Step 2 Decode (`par=16`) | Ollama Decode | Step 2 Prefill (`par=16`) | Ollama Prefill *(cold)* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| #1 *(cold)* | 132 | 10.67 tok/s | 10.63 tok/s | **10.91 tok/s** | 12.26 tok/s | 1,084 ms *(cold)* | 300.1 ms *(warm)* |
| #2 | 19 | 10.82 tok/s | 10.80 tok/s | **11.00 tok/s** | 11.98 tok/s | **158.5 ms** | 180.5 ms |
| #3 | 18 | 10.63 tok/s | 10.80 tok/s | **10.99 tok/s** | 11.91 tok/s | **158.0 ms** | 171.5 ms |
| #4 | 33 | 10.69 tok/s | 10.83 tok/s | **10.99 tok/s** | 12.01 tok/s | 187.1 ms | 175.9 ms |
| #5 | 356 | 10.39 tok/s | 10.51 tok/s | **10.63 tok/s** | 12.13 tok/s | 585.0 ms (`609 tok/s`) | 544.5 ms (`654 tok/s`) |
| #6 | 218 | 10.44 tok/s | 10.67 tok/s | **10.80 tok/s** | 11.75 tok/s | 440.0 ms (`495 tok/s`) | 374.8 ms (`582 tok/s`) |
| #7 | 234 | 10.45 tok/s | 10.63 tok/s | **10.78 tok/s** | 11.80 tok/s | 458.1 ms (`511 tok/s`) | 392.0 ms (`597 tok/s`) |
| #8 | 21 | 10.74 tok/s | 10.86 tok/s | **11.02 tok/s** | 11.86 tok/s | **161.4 ms** | 192.5 ms |
| #9 | 15 | 10.64 tok/s | 10.89 tok/s | **10.93 tok/s** | 11.59 tok/s | **125.9 ms** | 173.8 ms |
| #10 | 445 | 10.20 tok/s | 10.33 tok/s | **10.50 tok/s** | 11.91 tok/s | 663.4 ms (`671 tok/s`) | 613.5 ms (`725 tok/s`) |
| **10-Ex Decode Avg** | 487 gen | **10.54 tok/s** (`1.4s` CPU) | **10.70 tok/s** (+1.5%, `1.6s` CPU) | **10.85 tok/s** (**+2.9%**, **`1.2s` CPU**) | **11.86–11.92 tok/s** (`1.36s` CPU) | **Warm #2–#10: 2,937 ms (`462.7 tok/s`)** | **2,819 ms (`482.1 tok/s`)** |
| **Peak `MaxRSS`** | — | 16.0 GiB | 16.9 GiB | **16.0 GiB** | ~11.4 GiB | — | — |

### Crucial Diagnostic Discovery (`GOMLX_OP_PROFILE=30` at `par=16,ops_sequential`)
Running `GOMLX_OP_PROFILE=30` on decode revealed **the exact mathematical location of the entire remaining `~18.7 ms/token` gap vs. `llama.cpp`**:

| OpType / Signature | Calls/tok | Wall/tok | Wall/call | CPU/tok | Avg Cores | Effective BW |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FusedDense` (total, all 277 calls) | 277 | **87.5 ms** (`93.7%`) | — | **1,100 ms** | **12.4 / 16** | `48.2 GB/s` |
| ↳ `gate/up_12288 [1×1536×12288]` | 40 | 27.8 ms | 695.9 µs | 374.8 ms | 13.5 | 54.2 GB/s |
| ↳ `down_12288 [1×12288×1536]` | 20 | 14.4 ms | 722.2 µs | 194.7 ms | 13.5 | 52.3 GB/s |
| ↳ `lm_head [1×1536×262144]` | 1 | 14.4 ms | 14.4 ms | 215.5 ms | 14.9 | 55.9 GB/s |
| ↳ `gate/up_6144 [1×1536×6144]` | 30 | 10.8 ms | 361.7 µs | 128.8 ms | 11.9 | 52.2 GB/s |
| ↳ `down_6144 [1×6144×1536]` | 15 | 5.6 ms | 372.1 µs | 68.0 ms | 12.2 | 50.7 GB/s |
| ↳ `o_proj` / `q_proj` local (`2048↔1536`) | 56 | 7.4 ms | 133 µs | 56.8 ms | **7.1–8.1** | 47.3 GB/s |
| ↳ `k/v_proj` + `per_layer_gate` (`1536×256`) | 59 | 1.8 ms | 31 µs | 2.9 ms | **1.6** | 25.4 GB/s |
| ↳ `per_layer_proj` (`256×1536`) | 35 | 0.78 ms | 22 µs | 0.82 ms | **1.1** | 35.4 GB/s |
| `Reshape` (non-owned KV cache squeezes) | 762 | 1.1 ms (`0.74 ms` in 30 KV reshapes) | — | 1.1 ms | 1.0 | — |
| `ArgMinMax([1, 262144] BF16)` | 1 | 0.67 ms | 675 µs | 0.68 ms | 1.0 | — |
| Decomposed RMSNorm (`ConvertDType`/`Mul`/`ReduceSum`/`Div`/`Sqrt`) | ~176 layers | ~2.5 ms | — | ~2.7 ms | 1.1 | — |
| **Total (Sum of Ops)** | — | **93.3 ms** | — | **1,100 ms** | **11.7 / 16** | — |

**Key Takeaways for Next Steps (Steps 3 & 4):**
1. **Scheduler park/unpark barrier latency (`~18.75 ms/token`) is now the #1 wall-clock bottleneck!** All 277 `FusedDense` calls consume `1,100 ms` of CPU time = **`68.75 ms` of ideal 16-core compute** (`+ 5.8 ms` other ops = **`74.5 ms/tok` = `13.4 tok/s`**), yet spend **`87.5 ms`** in wall time because every `SaturateUpTo` call wakes parked goroutines from `<-intraOpQueue` (Linux `futex` sleep/wake), adding **`~65–100 µs` of barrier/wake-up latency per call** (e.g. `q_proj` needs `59 µs` compute but takes `133 µs` wall; `per_layer_proj` finishes in `22 µs` before helper goroutines even wake up!). **A bounded spin-wait barrier in `scheduler.go` (Step 4) directly targets this `15–19 ms/token` gap and should be prioritized alongside Step 3 (`M=1` GEMV).**
2. **`avx512LargeKernelBFloat16BF16Asm` still accounts for `94.6%` of all decode CPU time (`54.6s / 57.7s`)** due to executing 7/8 zero rows on `M=1`. Step 3's dedicated `1×64`/`1×128` `VDPBF16PS` GEMV microkernel will cut decode CPU time by ~3–4× (lowering core/power draw and helping `down_proj` reach `~58+ GB/s`).
3. **Physical-core parallelism (`par=16` vs `par=32`)**: `par=16` already beats `par=32` in **decode (`10.85 vs 10.70 tok/s`)**, **CPU efficiency (`1.2s vs 1.6s`, `-25%` CPU time)**, and **RAM (`16.0 vs 16.9 GiB`, `-0.9 GiB`)** — and matches warm prefill (`2,937 vs 2,949 ms`).
4. **Additional `~3.9 ms/tok` non-matmul wins surfaced by `GOMLX_OP_PROFILE`**: zero-copy `Reshape` views on non-owned KV cache buffers (`0.74 ms`), SIMD/parallel `ArgMinMax` on `BFloat16` (`0.67 ms`), and fusing Gemma 4 RMSNorm (`~2.5 ms`).



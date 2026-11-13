# Optimizing Quantized & GGUF Models in `compute` (`gobackend`)

## Executive Summary

Across four profile-driven iterations on `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (`Q4_K_M`), end-to-end throughput in GoMLX's pure-Go backend (`GOMLX_BACKEND=go`) increased by **87.0×–87.6×** over the scalar baseline—moving from **49.7× slower** than unquantized `Float32` to **1.55× faster** than optimized `Float32` (**1.75× faster** than initial `Float32`) while consuming **41% less CPU time**:

| Stage / Configuration (`-limit 20 -bucket=8192 -num_embedders=1 -warmup -dynamic=true`) | Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | Speedup vs Baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline GGUF (`Q4_K_M`, Scalar)** | 24.1s | 8.30 items/s | **2.4K tok/s** | 579 tok/s | 744.0s | 1.0× |
| Initial Unquantized `Float32` (`all-MiniLM-L6-v2`) | 484.7ms | 413 items/s | 117.9K tok/s | 28.8K tok/s | 9.8s | 49.7× |
| Updated Unquantized `Float32` (with Step 3 `ops` parallelization) | 429.0ms | 466 items/s | 133.2K tok/s | 32.5K tok/s | 9.8s | 56.2× |
| **Step 1 — K4-Interleaved GEMM + SWAR Unpack (AVX2)** | 603.1ms | 332 items/s | 94.8K tok/s | 23.1K tok/s | 12.7s | 40.0× |
| **Step 1 — K4-Interleaved GEMM + SWAR Unpack (AVX-512 VNNI)** | 382.6ms | 523 items/s | 149.4K tok/s | 36.4K tok/s | 6.6s | 63.0× |
| **Step 2 — Strip-Outer Tiling + Asm Quantizers + Const Cache (AVX2)** | 573.4ms | 349 items/s | 99.7K tok/s | 24.3K tok/s | 12.1s | 42.0× |
| **Step 2 — Strip-Outer Tiling + Asm Quantizers + Const Cache (AVX-512 VNNI)** | 376.5ms | 531 items/s | 151.8K tok/s | 37.0K tok/s | 6.4s | 64.0× |
| **Step 3 — Parallel `BroadcastInDim` / `ConvertDType` / `Gather` / `Embedding` (AVX2)** | 503.0ms | 398 items/s | 113.6K tok/s | 27.7K tok/s | 12.2s | 47.9× |
| **Step 3 — Parallel `BroadcastInDim` / `ConvertDType` / `Gather` / `Embedding` (AVX-512 VNNI)** | 301.1ms | 664 items/s | 189.8K tok/s | 46.3K tok/s | 6.2s | 80.0× |
| **Step 4 — Sibling Q/K/V & Gate/Up Activation Sharing (AVX2)** | **485.9ms** | **412 items/s** | **117.6K tok/s** | **28.7K tok/s** | **11.7s** | **49.6×** |
| **Step 4 — Sibling Q/K/V & Gate/Up Activation Sharing (AVX-512 VNNI)** | **277.2ms** | **721 items/s** | **206.2K tok/s** | **50.3K tok/s** | **5.8s** | **87.0× (1.55× > FP32)** |

Sustained 2,000-passage throughput (`-limit 200`, `236.7K` total tokens / `153.5K` non-pad tokens):
- **Updated Unquantized `Float32` (AVX-512)**: `1.75s` wall, `134.9K tokens/s` (`87.5K` non-pad tok/s), `42.9s` CPU.
- **Step 4 GGUF `Q4_K_M` (AVX2, `GOMLX_GO_SIMD_AVX512=0`)**: `2.08s` wall, `113.6K tokens/s` (`73.7K` non-pad tok/s), `50.6s` CPU.
- **Step 4 GGUF `Q4_K_M` (AVX-512 VNNI)**: **`1.18s` (`1,169–1,196ms`)** wall, **`197.9K–202.5K tokens/s`** (`128.3K–131.3K` non-pad tok/s), **`25.7s` CPU** (**1.50× faster** than `Float32` with **40% less CPU**).

---

## 1. Task, Hardware & Reproduction

### Goal
Profile, investigate, and iteratively optimize `FusedQuantizedDense`, `QuantizedEmbeddingLookup`, and quantized model execution in GoMLX's pure-Go backend (`GOMLX_BACKEND=go`), maximizing end-to-end `tokens/s` on GGUF models while keeping the architecture general across `QuantGGML` (all 12 GGML block formats), `QuantLinear` (`Int8`, `Uint8`, `Int4`, `Uint4`, `Int2`, `Uint2`), and `QuantNF4`.

### Hardware & Toolchain
- **CPU**: AMD Ryzen 9 9950X3D (Zen 5, 16 physical cores / 32 threads, 48 KB L1d/core, 1 MB L2/core, 128 MB 3D V-Cache L3)
- **SIMD**: `AVX2`, `FMA`, `F16C`, `AVX-VNNI`, `AVX-512` (`AVX512F`, `AVX512BW`, `AVX512DQ`, `AVX512VL`, `AVX512VBMI`, `AVX512VBMI2`, `AVX512_VNNI`, `AVX512_BF16`, `AVX512_BITALG`)
- **Go**: `go1.27.3 linux/amd64` with `GOEXPERIMENT=simd`

### Benchmark & Profiling Commands
Run from `/home/janpf/Projects/gomlx/go-huggingface`:

```bash
# Primary GGUF Benchmark (second-state/All-MiniLM-L6-v2-Embedding-GGUF, Q4_K_M)
GOEXPERIMENT=simd GOMLX_BACKEND=go go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed/ \
  -bucket=8192 \
  -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
  -limit 20 \
  -num_embedders=1 \
  -warmup \
  -dynamic=true \
  -cpuprof=/tmp/gguf_baseline.prof

# Unquantized Float32 Safetensors Reference (sentence-transformers/all-MiniLM-L6-v2)
GOEXPERIMENT=simd GOMLX_BACKEND=go go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed/ \
  -bucket=8192 \
  -model sentence-transformers/all-MiniLM-L6-v2 \
  -limit 20 \
  -num_embedders=1 \
  -warmup \
  -dynamic=true

# Per-Op Sequential Timing Breakdown
GOMLX_OP_PROFILE=2 GOEXPERIMENT=simd GOMLX_BACKEND="go:ops_sequential" \
  go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed/ \
  -bucket=8192 \
  -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
  -limit 10 \
  -num_embedders=1 \
  -warmup=false \
  -dynamic=true
```

---

## 2. Baseline Benchmark & CPU Profile

### 2.1 End-to-End Throughput (`-limit 20 -bucket=8192 -num_embedders=1 -warmup -dynamic=true`)

The `-limit 20` run embeds 200 MS MARCO passages (`57.2K` padded tokens, `13.9K` non-pad tokens) packed into 7 dynamic batch shapes (`[42, 192]`, `[170, 48]`, `[128, 64]`, `[64, 128]`, `[85, 96]`, `[256, 32]`, `[1024, 8]`, each with $M = \text{batch} \times \text{seq\_len} \approx 8{,}064\text{--}8{,}192$ tokens):

| Metric | Baseline GGUF (`all-MiniLM-L6-v2-Q4_K_M.gguf`) | Unquantized Float32 (`sentence-transformers/all-MiniLM-L6-v2`) | Baseline Gap |
| :--- | ---: | ---: | ---: |
| **Warmup Wall Time** | 24.0s (`8.33 items/s`) | 491.3ms (`407 items/s`) | 48.8× slower |
| **Benchmark Wall Time** | **24.1s** (`8.30 items/s`) | **484.7ms** (`413 items/s`) | **49.7× slower** |
| **Total Tokens/s** | **2.4K tokens/s** | **117.9K tokens/s** | **49.1× slower** |
| **Non-Pad Tokens/s** | **579 tokens/s** | **28.8K tokens/s** | **49.7× slower** |
| **Benchmark CPU Time** | 12m24s (`744.0s`, `30.9` cores) | `9.8s` (`20.2` cores) | **75.9× more CPU** |

### 2.2 Baseline `pprof` Breakdown (`/tmp/gguf_baseline.prof`, `1485.67s` total CPU across warmup + bench)

| Function | Flat CPU | Flat % | Cum CPU | Cum % | Role |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `fusedops.vecDotQ5_0_Q8_0` | 902.27s | 60.73% | 963.95s | 64.88% | Scalar `Q5_0 × Q8_0` block dot (`ff1` + 22 `[384,384]` projections) |
| `fusedops.vecDotQ4_K_Q8_K` | 234.81s | 15.80% | 254.78s | 17.15% | Scalar `Q4_K × Q8_K` super-block dot (`ff2` in 4 layers) |
| `fusedops.vecDotQ6_K_Q8_K` | 212.69s | 14.32% | 215.22s | 14.49% | Scalar `Q6_K × Q8_K` super-block dot (`ff2` in 2 layers) |
| `float16.Float16.Float32` | 63.85s | 4.30% | 63.85s | 4.30% | Software FP16 scale decode (`ggmlFp16LE` inside `vecDot*`) |
| `fusedops.vecDotQ8_0_Q8_0` | 26.82s | 1.81% | 29.76s | 2.00% | Scalar `Q8_0 × Q8_0` block dot (2 `[384,384]` projections) |
| `fusedops.quantizedDenseGGMLBlockInt.func1` | 11.43s | 0.77% | **1475.21s** | **99.30%** | Outer `(m, n)` worker loop in [quantizeddense_ggml.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_ggml.go) |
| `fusedops.getScaleMinK4` | 7.48s | 0.50% | 7.48s | 0.50% | Unpacking 6-bit `Q4_K` scale/min headers |
| `fusedops.ggmlBsum16` | 6.32s | 0.43% | 6.32s | 0.43% | Re-reading `int16` `Q8_K` block sums inside `vecDotQ4_K_Q8_K` |

### 2.3 Baseline Sequential Per-Op Profile (`GOMLX_OP_PROFILE=2`, per `M ≈ 8192` batch)

| OpType / Signature | Calls/Batch | Wall Time/Batch | % Wall | Avg Cores | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **`FusedQuantizedDense` (total)** | **36** | **3,397.0 ms** | **99.43%** | **31.0** | Dominates execution completely |
| ↳ `Q5_0` `[M, 384] × [1536, 264]` (`ff1`, GELU fused) | 6 | 1,166.4 ms | 34.14% | 31.3 | `194.4 ms`/call |
| ↳ `Q5_0` `[M, 384] × [384, 264]` (`Q/K/V/attn_out`) | 22 | 1,073.6 ms | 31.42% | 30.9 | `48.8 ms`/call |
| ↳ `Q4_K` `[M, 1536] × [384, 864]` (`ff2` layers 0,1,2,4) | 4 | 590.0 ms | 17.27% | 30.8 | `147.5 ms`/call |
| ↳ `Q6_K` `[M, 1536] × [384, 1260]` (`ff2` layers 3,5) | 2 | 490.6 ms | 14.36% | 30.9 | `245.3 ms`/call |
| ↳ `Q8_0` `[M, 384] × [384, 408]` (`V` layers 3,5) | 2 | 70.8 ms | 2.07% | 30.8 | `35.4 ms`/call |
| `DynamicBroadcastInDim` | 3 | 6.6 ms | 0.19% | 1.0 | Single-threaded |
| `FusedLayerNorm` | 13 | 5.7 ms | 0.17% | 14.5 | AVX-512 |
| `FusedScaledDotProductAttention` | 6 | 2.8 ms | 0.08% | 21.5 | AVX-512 (`head_dim=32`) |
| `QuantizedEmbeddingLookup` (`Q8_0` `[30522, 408]`) | 1 | 1.5 ms | 0.04% | 1.0 | Single-threaded, scalar `dequantQ8_0Row` |
| `ConvertDType` (`Bool -> Float32`) | 3 | 1.1 ms | 0.03% | 1.0 | Single-threaded |
| All other ops (`ReduceWindow`, `BroadcastInDim`, `Where`, etc.) | 36 | 1.8 ms | 0.05% | — | Minor |

---

## 3. Root-Cause Analysis

`FusedQuantizedDense` in [quantizeddense_ggml.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_ggml.go) and [quantizeddense_linear.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_linear.go) suffers from two compounding bottlenecks:

1. **$O(M)$ Redundant Weight Decoding (GEMV Loop Structure for GEMM Shapes)**:
   - `quantizedDenseGGMLBlockInt`, `quantizedDenseGGMLFloat`, and `execQuantizedDenseLinear*` iterate `for m := range M { for n := range N { dotFn(wRow[n], xRow[m]) } }`.
   - When $M = 8{,}192$, every compressed weight block (`Q5_0` high-bit shifts + nibble masks, `Q4_K` 6-bit scale/min headers, `Q6_K` 2-bit + 4-bit reconstruction, `Int4`/`NF4` nibble extraction, and software `Float16.Float32()` scale conversions) is unpacked **8,192 times per call** instead of **once**.
   - Even activation scales (`dX`) and `Q8_K` `bsums` are re-parsed from raw bytes $N$ times per row (`63.85s` in `Float16.Float32` alone).
2. **100% Scalar Inner Loops (Zero SIMD / VNNI)**:
   - All `vecDot*`, `quantizeRowQ8_*`, `dequant*Row`, and `QuantLinear` / `QuantNF4` loops execute scalar byte-by-byte arithmetic in Go.
   - Meanwhile, Zen 5 / AVX-512 VNNI executes `VPDPBUSD` (`64` `uint8 × int8 -> int32` MACs per instruction, 2 instructions/cycle/core = **128 INT8 MACs/cycle/core**, **4× higher throughput** than FP32 FMA `32` MACs/cycle/core), and AVX2 / AVX-VNNI executes `VPMADDUBSW`+`VPMADDWD` or VEX `VPDPBUSD` (`32` INT8 MACs/instruction).

---

## 4. Planned Optimization Roadmap

### Step 1: Unified K4-Interleaved Tiled Integer GEMM + SIMD Activation Quantizer (Target: 25×–60× Speedup on `FusedQuantizedDense`)

Every block-quantized integer dot product across **all 12 `QuantGGML` formats** (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`) and **`QuantLinear`** (`Int8`, `Uint8`, `Int4`, `Uint4`, `Int2`, `Uint2`, both dynamic Int8 `QuantizedMatmul` and pre-quantized `XQuantization`) reduces to uniform sub-blocks of size $B_{\text{sub}}$ along $K$ ($B_{\text{sub}} = 32$ for `*_0`/`*_1`/`IQ4_NL`/`Q4_K`/`Q5_K`, $B_{\text{sub}} = 16$ for `Q2_K`/`Q3_K`/`Q6_K`, or `kBlock` for `QuantLinear`):
- Shift signed weight quants by a constant $+C$ into unsigned bytes $u_{w}[i] \in [0, 255]$ ($C=8$ for `Q4_0`/`Int4`, $C=16$ for `Q5_0`, $C=128$ for `Q8_0`/`IQ4_NL`/`Q8_K`/`Int8`, $C=4$ for `Q3_K`, $C=32$ for `Q6_K`, $C=2$ for `Int2`, $C=0$ for unsigned formats), with effective float scale $d'_w$ and effective float offset $m'_w$:
$$
\Delta Z[m, n] = \left(d_{x,m,b} \cdot d'_{w,n,b}\right) \sum_{i=0}^{B_{\text{sub}}-1} u_{w,n,b}[i] \, q_{x,m,b}[i] + \left(d_{x,m,b} \sum_{i=0}^{B_{\text{sub}}-1} q_{x,m,b}[i]\right) m'_{w,n,b}
$$
- **K4-Interleaved Weight Panel Layout**:
  - Process `N` in L2-resident chunks (`Nc = 256` or `512` columns, requiring `< 1.5 MB` pooled scratch — zero persistent RAM bloat even for 70B models).
  - Unpack each `Nc` chunk of `W` **once** (in parallel across `N` with SIMD nibble/bit unpackers) into K4-interleaved column panels `[numSubBlocks, B_sub/4, Nr, 4]` `uint8` (`Nr = 32` for AVX-512, `Nr = 16` for AVX2) plus contiguous `[numSubBlocks, Nr]` `float32` `d'_w` and `m'_w` arrays.
- **SIMD Activation Pre-Unpacking / Quantization**:
  - Quantize float32 `x` `[M, K]` directly (in parallel across `M` using `archsimd` `Float32x16` / `Float32x8` max-abs + `RoundToEven` + saturation) into contiguous `q_x` `[M, K]` `int8`, `d_x` `[M, numSubBlocks]` `float32`, and `dx_bsum` `[M, numSubBlocks]` `float32` (or unpack pre-quantized `Q8_0`/`Q8_K`/`QuantLinear` `x` once into the same canonical buffers).
- **Handwritten Assembly Microkernels** in `compute/internal/gobackend/fusedops/avx512/` and `avx2/`:
  - **AVX-512 VNNI (`VPDPBUSD`)** & **AVX-512 BW fallback**: $8 \times 32$ (`Mr = 8, Nr = 32`, 16 `ZMM` float32 accumulators) microkernel broadcasting 4 `int8` activation bytes (`VPBROADCASTD`) and accumulating `64` INT8 MACs per `VPDPBUSD` into `int32` lanes with **zero horizontal reductions**, converting per sub-block via `VCVTDQ2PS` + `VFMADD231PS`, and applying bias + fused activation on the output tile in L1 cache.
  - **AVX2 (`AVXVNNI` `VPDPBUSD` / `VPMADDUBSW` + `VPMADDWD`)**: $4 \times 16$ (`Mr = 4, Nr = 16`, 8 `YMM` float32 accumulators) microkernel over the identical K4-interleaved layout, safely handling full `[0..255]` unsigned ranges when `AVXVNNI` is absent (`VPSIGNB` / `int16` split so `int16` never saturates).

### Step 2: Small-$M$ (`M <= 4`) Vectorized `vecDot`, Zero-Alloc `M == 1` Parallelization & Tiled Mode A / VJP

- **Small-$M$ Regime (`M = 1..4`, LLM token generation)**:
  - When $M$ is very small, unpacking all of `W` into a panel buffer is unnecessary; dispatch to SIMD `vecDot` kernels (`AVX-512` / `AVX2`) operating directly on raw GGML / `QuantLinear` blocks.
  - Replace closure-slice allocation (`make([]func(), 0, numTiles)`) in `quantizedDenseParallel` (`M == 1` path in [quantizeddense.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense.go#L795-L808)) with zero-allocation `Scheduler.ParallelizeChunks`.
- **Mode A (Weight-Only Float `QuantizedMatmul == false`, including `QuantNF4`) & `FusedQuantizedDenseVJP`**:
  - Dequantize `W` in bounded `Nc` row chunks using SIMD row dequantizers (`dequant*Row`) into a pooled `[Nc, K]` `float32` buffer and invoke GoMLX's AVX-512 / AVX2 `matmul` GEMM engine (`3.7 TFLOP/s`), accelerating `QuantNF4`, float-mode `QuantGGML`/`QuantLinear`, and VJP backward passes.

### Step 3: Non-MatMul & Graph-Level Bottlenecks (Profile-Driven)

Once `FusedQuantizedDense` drops from `3,400 ms`/batch into the `15–40 ms`/batch range, previously hidden operations (`~19.5 ms`/batch total in baseline) will represent a large fraction of runtime. Guided by fresh CPU & op profiles, we will target:
1. **`QuantizedEmbeddingLookup`** ([quantizeddense.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense.go#L697-L743)): Parallelize row gather across `numIndices` using `ParallelizeChunks` + SIMD `dequant*Row` (`1.5 ms` single-threaded today).
2. **Single-Threaded Mask/Broadcast Ops**: Inspect and parallelize/vectorize `DynamicBroadcastInDim` (`6.6 ms`/batch at `1.0` core) and `ConvertDType(Bool -> Float32)` (`1.1 ms`/batch at `1.0` core).
3. **Shared Q/K/V Activation Quantization**: In Transformer self-attention, Q, K, and V `FusedQuantizedDense` projections share the exact same `x` input tensor (`[8192, 384]`), quantizing `x` 3× per layer (`18×` across 6 layers). Evaluate caching/sharing quantized `x` across sibling projections if profiling shows activation quantization is significant.

---

## 5. Iterative Optimization Log

### Step 1: Unified K4-Interleaved Tiled Integer GEMM + SIMD Activation Quantizers + SWAR Weight Unpackers

#### Changes Implemented
1. **CPU Feature Detection** ([simd_x86.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/simdarch/simd_x86.go), [simd.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/simd.go)):
   - Added `IsAVX512VNNIAllowed` (`archsimd.X86.AVX512VNNI()`) and `IsAVXVNNIAllowed` (`archsimd.X86.AVXVNNI()`).
2. **Architecture-Neutral Panel Format & Dispatchers** ([quantizeddense_arch.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/quantizeddense_arch.go), [quantizeddense_panel.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_panel.go), [quantizeddense_ggml_unpack.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_ggml_unpack.go)):
   - Defined a shared 16-column K4-interleaved strip format (`[quads][16][4]uint8` + `[16]float32` `d'_w` + `[16]float32` `m'_w` contiguous per sub-block) shared identically by AVX-512 (`1` ZMM = 16 cols) and AVX2 (`2` YMMs = 16 cols).
   - Implemented branchless 64-bit SWAR weight unpackers across **all 12 GGML types** (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`) and **`QuantLinear`** (`Int8`, `Uint8`, `Int4`, `Uint4`, `Int2`, `Uint2`, both `InputOutputs` and `OutputsInput` layouts, including float zero-points), bounded to a `2 MB` pooled scratch cap.
   - Fused post-matmul activations (`GELU`, `ReLU`, `SiLU`, etc.) directly inside worker row blocks while outputs are warm in L1/L2 cache, avoiding a full-matrix `48 MB` second pass on `ff1`.
3. **Handwritten Assembly Microkernels & SIMD Activation Quantizers**:
   - **AVX-512 VNNI** ([avx512/quantizeddense_amd64.s](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/avx512/quantizeddense_amd64.s), [avx512/quantizeddense.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/avx512/quantizeddense.go)): $6 \times 32$ and $1 \times 32$ `VPDPBUSD` assembly microkernels with unrolled `quads=8` (`B_sub=32`) and `quads=4` (`B_sub=16`) fast paths, plus 512-bit `Float32x16` activation quantizers.
   - **AVX2** ([avx2/quantizeddense_amd64.s](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/avx2/quantizeddense_amd64.s), [avx2/quantizeddense.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/avx2/quantizeddense.go)): Pure-VEX $3 \times 16$ and $1 \times 16$ `VPMADDUBSW`+`VPMADDWD` assembly microkernels with overflow-free 7-bit split (`u_a = (u_w>>1)&0x7F`, `u_b = u_w - u_a`) for 8-bit unsigned ranges, plus 256-bit `Float32x8` activation quantizers.

#### Step 1 Benchmark Results (`-limit 20 -bucket=8192 -num_embedders=1 -warmup -dynamic=true`)

| Configuration | Benchmark Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | Speedup vs Baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline GGUF (Scalar) | 24.1s | 8.30 items/s | 2.4K tok/s | 579 tok/s | 744.0s | 1.0× |
| Unquantized Float32 Reference | 484.7ms | 413 items/s | 117.9K tok/s | 28.8K tok/s | 9.8s | 49.7× |
| **Step 1 — AVX2 (`GOMLX_GO_SIMD_AVX512=0`)** | **603.1ms** | **332 items/s** | **94.8K tok/s** | **23.1K tok/s** | **12.7s** | **40.0×** |
| **Step 1 — AVX-512 VNNI** | **382.6ms** | **523 items/s** | **149.4K tok/s** | **36.4K tok/s** | **6.6s** | **63.0× (1.27× faster than FP32!)** |

#### Step 1 Updated CPU Profile (`-limit 200`, `39.30s` sampled CPU) & Per-Op Profile (`GOMLX_OP_PROFILE=2`)

| Hot Function (`pprof -top`) | Flat CPU | Flat % | Cum CPU | Cum % | Next Optimization Insight |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `avx512.quantizedPanelKernel6x32AVX512VNNIAsm` | 17.38s | 44.22% | 17.38s | 44.22% | Inner `s` loop streams `737–884 KB` panel per 6-row block, evicting L1d (`48 KB`) & thrashing shared SMT L2! Flipping to strip-outer (`s` outer, `mb` inner) keeps `15 KB` strip in L1d! |
| `avx512.QuantizeGGMLActivationsAVX512` (+ inline `reduceMax`/`reduceSum`/`FromFloat32`) | 7.60s | 19.34% | 8.38s | 21.32% | `StoreArray(&[4]int32)` causes 12-cycle store-to-load forwarding stalls + software FP16 conversion. Assembly `VCVTPS2PH`/`VCVTPH2PS` + register reductions will cut this ~5–8×! |
| `avx512.layerNormResidualFloat32AVX512` | 3.47s | 8.83% | 3.47s | 8.83% | Already AVX-512 assembly |
| `avx512.sdpa*` (`FusedSDPA`) | 3.84s | 9.77% | 5.11s | 13.00% | Already AVX-512 assembly |
| `activations/avx512.geluApproxFloat32AVX512Asm` | 0.80s | 2.04% | 0.80s | 2.04% | Already fused in-tile |

| OpType (`GOMLX_OP_PROFILE=2`, sequential) | Calls/Batch | Baseline Wall | Step 1 Wall | Wall % | Avg Cores | Bottleneck |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `FusedQuantizedDense` | 36 | 3,397.0 ms | **33.4 ms** | 62.7% | 21.6 | **101.7× faster!** Still has per-call weight unpack barrier + L2 panel thrashing + `QuantizeGGMLActivations` stalls |
| `DynamicBroadcastInDim` | 3 | 6.6 ms | **7.7 ms** | **14.5%** | **1.0** | **Single-threaded scalar `BroadcastIterator`!** |
| `FusedLayerNorm` | 13 | 5.7 ms | 4.6 ms | 8.6% | 19.4 | AVX-512 |
| `FusedScaledDotProductAttention` | 6 | 2.8 ms | 2.8 ms | 5.3% | 15.3 | AVX-512 |
| `QuantizedEmbeddingLookup` | 1 | 1.5 ms | **1.8 ms** | **3.3%** | **1.0** | **Single-threaded scalar `dequantQ8_0Row`!** |
| `ConvertDType` (`Bool -> Float32`) | 3 | 1.1 ms | **1.2 ms** | **2.2%** | **1.0** | **Single-threaded!** |
| `Gather` (`[512, 384]`) | 2 | 0.7 ms | **0.7 ms** | **1.3%** | **1.0** | **Single-threaded!** |
| **Total Sum of Ops** | — | **3,416.5 ms** | **53.3 ms** | **100.0%** | — | **64.1× faster op sum!** |


---

### Step 2: Cache-Optimal Strip-Outer Tiling, Assembly Activation Quantizers & Constant Panel Caching

#### What Changed
1. **L1d/L2 Cache-Optimal Strip-Outer Tiling + Multi-MBlock Assembly Loop + Small-$M$ 2D Scheduling**:
   - Restructured `DispatchQuantizedPanelGEMMAVX512VNNI` (`mcBlocks = 16` / 96 rows) and `DispatchQuantizedPanelGEMMAVX2` (`mcBlocks = 32` / 96 rows) so each worker iterates **strip-group outer, M-block inner**, pinning the active 32-column (`15–61 KB`) or 16-column (`7.5–30.7 KB`) weight strip in L1d/L2 across up to 96 activation rows before applying fused in-tile `ActFn`.
   - Moved the contiguous full M-block loop inside `quantizedPanelKernel6x32AVX512VNNIAsm` and hoisted `VCVTDQ2PS` conversions across scale loads, dropping flat Go dispatcher/ABI/`VZEROUPPER` overhead by **3.4×** (`1.67s` $\to$ `0.49s` in `-limit 200` `pprof`).
   - Added 2D `(stripGroup, mBlock)` parallel scheduling when `numMBlocks < numWorkers` and lowered `quantizedPanelMinM` from `2` to `1` so even single-token `M = 1` decode saturates all CPU cores with `1x32` / `1x16` SIMD/VNNI kernels.
2. **Handwritten Assembly Activation Quantizers (`AVX-512 + F16C` and `AVX2 + F16C`)**:
   - Implemented `quantizeGGMLQ8_0RowsAVX512Asm` and `quantizeGGMLQ8_KRowsAVX512Asm` in `avx512/quantizeddense_amd64.s`, plus pure-VEX `quantizeGGMLQ8_0RowsAVX2Asm` and `quantizeGGMLQ8_KRowsAVX2Asm` in `avx2/quantizeddense_amd64.s`.
   - Replaced stack `StoreArray` reductions and software FP16 rounding with in-register horizontal reductions, hardware `VCVTPS2PH`/`VCVTPH2PS`, and AVX-512 `VPMOVSDB` / AVX2 `VPACKSSDW`+`VPACKSSWB`+`VPERMQ` byte packing, reducing activation quantization CPU from `8.38s` to `6.68s` (now 80%+ DRAM load bound on `xF32`).
3. **Optional Constant & Donated Weight Panel Pre-Packing (`go:quantized_cache`, Default `off`)**:
   - Implemented `gobackend.ConstantCacheNodeData` and `gobackend.RecomputableNodeData` on `nodeFusedQuantizedDense` for constant/donated `QuantGGML` weights, plus `sync.Once` panel caching for constant `QuantLinear` weights + scales/zero-points.
   - Because unpacking sub-8-bit weights into K4-interleaved `uint8` panels + `float32` scales expands storage to `1.25–1.50 bytes/weight` (~2.2×–2.5× larger than 4-bit), **`quantized_cache` defaults to `off`** (streaming `2 MB` pooled L3-resident chunks with `0 B` persistent overhead) and can be enabled without size limit via `GOMLX_BACKEND="go:quantized_cache"`.

#### Benchmark Progression
- **`-limit 20` (`57.2K` total tokens)**:
  - **Step 2 AVX2 (`GOMLX_GO_SIMD_AVX512=0`)**: **`573.4 ms`** wall, **`99.7K tokens/s`** (`24.3K` non-pad tok/s), `12.1s` CPU (**+5.2%** over Step 1 AVX2, **42.0×** over baseline).
  - **Step 2 AVX-512 VNNI**: **`376.5 ms`** wall, **`151.8K tokens/s`** (`37.0K` non-pad tok/s), `6.4s` CPU (`~Cores = 17.1`, **64.0×** over baseline).
- **`-limit 200` (`236.7K` total tokens, `153.5K` non-pad tokens)**:
  - **Step 1 AVX-512 VNNI**: `1,663 ms` wall, `142.3K tokens/s` (`92.3K` non-pad tok/s), `29.7s` CPU.
  - **Step 2 AVX-512 VNNI**: **`1,561 ms`** wall, **`151.8K tokens/s`** (`98.5K` non-pad tok/s), `28.1s` CPU (**+6.7%** throughput, `-1.6s` CPU).

#### Updated Profile Diagnosis (Why `~Cores = 17.1` out of 32)
With `FusedQuantizedDense` weight unpacking gone (`0.00s`) and kernel dispatch overhead down to `0.49s`, **Amdahl's Law on single-threaded non-matmul ops** is now the #1 wall-clock bottleneck (`~11.0 ms` out of `53.6 ms` per batch = **20.5% of total wall time running on 1 core**):
- `DynamicBroadcastInDim` (`3` calls/batch): **`7.3 ms`/batch** (`1.0` core, `BroadcastIterator`).
- `QuantizedEmbeddingLookup` (`1` call/batch, `Q8_0` `[30522, 408]`): **`1.7 ms`/batch** (`1.0` core).
- `ConvertDType` (`Bool -> Float32`, `3` calls/batch): **`1.3 ms`/batch** (`1.0` core).
- `Gather` (`[512, 384]`, `2` calls/batch): **`0.7 ms`/batch** (`1.0` core).
- In addition, inside `FusedQuantizedDense`, `Q_proj`, `K_proj`, and `V_proj` (`18` of `36` calls/batch, mostly `Q5_0`/`Q8_0` on `[batch, seq_len, 384]`) each independently stream and quantize the same `12.5 MB` `LayerNorm` output from DRAM (`6.68s` CPU in `QuantizeGGMLActivationsAVX512`).

---

### Step 3: Coalesced Parallel `BroadcastInDim`, `ConvertDType`, `Gather`, and `QuantizedEmbeddingLookup`

#### Root-Cause & Changes Implemented
1. **Coalesced Multi-Pattern Parallel `BroadcastInDim` & `DynamicBroadcastInDim`** ([broadcastindim.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/ops/broadcastindim.go)):
   - Discovered a bug/dead-code path in `execBroadcastInDimGeneric`: `if operand.RawShape.Size() > 1` unconditionally initialized `BroadcastIterator` and took the slow element-by-element branch, rendering the prefix `copy()` fast path completely unreachable! For Transformer token-type / pooling broadcasts (`[1, 1, 384] -> [B, S, 384]` and `[B, S] -> [B, S, 384]`), `3.1M` elements were iterated scalar-by-scalar on 1 core (`7.3–7.7 ms/batch`).
   - Replaced it with `analyzeBroadcastPlan`, which coalesces contiguous kept/broadcasted axis runs and dispatches scalar fill (`xslices.FillSlice`), identity copy, prefix `[1, K] -> [Outer, K]` (`repeatSliceDoubling` exponential `copy()`), suffix `[Outer, 1] -> [Outer, Inner]`, middle `[Outer, 1, Inner]`, and `[1, Mid, 1]` patterns across `Scheduler.ParallelizeChunks` (`>= 16,384` output elements), keeping general strided `BroadcastIterator` strictly as the fallback.
2. **Parallel Sub-Byte-Safe `ConvertDType` + 4-Wide Run-Length `Bool` Conversion** ([convertdtype.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/ops/convertdtype.go)):
   - Added 64-element-aligned `subSliceBufferElements` (supporting sub-byte `Int4`/`Uint4`/`Int2`/`Uint2` buffers without shared byte races) and parallelized `execConvertDType` via `Scheduler.ParallelizeChunks` (`>= 16,384` elements).
   - Added a 4-wide all-true / all-false fast path in `execConvertDTypeFromBool` (dominant in padded attention masks).
3. **Parallel Contiguous-Slice Batch `Gather`** ([gather.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/ops/gather.go)):
   - Added exact `gatherBatchOutputIsContiguous` layout verification and parallelized `gatherNoIndexVectorDim` / `gatherWithIndexVectorDim` over `numBatches` via `Scheduler.ParallelizeChunks` (`>= 16,384` output bytes, accelerating positional and vocabulary gathers).
4. **Parallel `QuantizedEmbeddingLookup` + Unrolled Row Dequantizers** ([quantizeddense.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense.go), [quantizeddense_ggml.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_ggml.go)):
   - Eliminated the temporary `Int32 -> Int64` index buffer allocation/conversion by reading `Int32` and `Int64` indices directly, parallelized row lookups across `numIndices` via `Scheduler.ParallelizeChunks`, unrolled `dequantQ8_0Row` 4×, and added BCE hints to `dequantQ4_0Row`.

#### Step 3 Benchmark Results
- **`-limit 20` (`57.2K` total tokens)**:
  - **Step 3 AVX2 (`GOMLX_GO_SIMD_AVX512=0`)**: **`503.0 ms`** wall, **`113.6K tokens/s`** (`27.7K` non-pad tok/s), `12.2s` CPU (**+13.9%** over Step 2 AVX2, **47.9×** over baseline).
  - **Step 3 AVX-512 VNNI**: **`301.1 ms`** wall, **`189.8K tokens/s`** (`46.3K` non-pad tok/s), `6.2s` CPU (`~Cores = 20.5`, **+25.0%** over Step 2, **80.0×** over baseline!).
  - *Bonus*: Unquantized `Float32` `sentence-transformers/all-MiniLM-L6-v2` also sped up by **+13.0%** (`484.7 ms` $\to$ `429.0 ms`, `117.9K` $\to$ `133.2K tokens/s`) from the generic `BroadcastInDim` / `ConvertDType` / `Gather` improvements!
- **`-limit 200` (`236.7K` total tokens)**:
  - **Step 3 AVX-512 VNNI**: **`1,289 ms`** wall, **`183.6K tokens/s`** (`119.0K` non-pad tok/s), `28.6s` CPU (**+21.0%** over Step 2).

#### Step 3 Profile Diagnosis (`/tmp/gguf_step3_200.prof`, `38.68s` sampled CPU)
- Single-threaded non-matmul ops dropped by **10.4×** in sequential op wall time (`11.4 ms/batch` $\to$ `1.1 ms/batch` total across `DynamicBroadcastInDim`, `QuantizedEmbeddingLookup`, `ConvertDType`, and `Gather`).
- In `pprof`, `quantizeGGMLQ8_0RowsAVX512Asm` (`3.70s` / `9.57%`) + `quantizeGGMLQ8_KRowsAVX512Asm` (`3.59s` / `9.28%`) totaled **`7.29s` (`18.85%`)**. Crucially, in every Transformer layer, `Q_proj`, `K_proj`, and `V_proj` all quantize the exact same `[8192, 384]` `LayerNorm` output (`12.5 MB`), and in `ops_parallel` mode all three projections run concurrently, launching three simultaneous 32-worker DRAM streams on the same tensor!

---

### Step 4: Compile-Time Sibling Activation Quantization Sharing (`Q/K/V` & SwiGLU `Gate/Up`)

#### Changes Implemented
1. **Compile-Time Sibling Grouping in `FunctionExecutable`** ([executable.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/executable.go)):
   - Added `FusedQuantizedDenseActShareTarget` and `FunctionExecutable.bindSharedQuantizedDenseActivations()` invoked at graph compile time (`newFunctionExecutable`).
   - Automatically groups sibling `OpTypeFusedQuantizedDense` nodes that share the same input tensor `Inputs[0]` (`x`), `QuantizedMatmul == true`, nil `XQuantization`, and identical activation quantization keys (`ggmlActSubBlockSize` `Q8_0` vs `Q8_K` for `QuantGGML`, or `KBlockSize` for `QuantLinear`).
   - Applies generically across all Transformer architectures: self-attention `Q / K / V` projections (3-way sharing) and SwiGLU / GeGLU `Gate / Up` MLP projections (2-way sharing), for both `QuantGGML` and `QuantLinear`!
2. **Thread-Safe Multi-Mode `sharedQuantActGroup`** ([quantizeddense.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense.go), [quantizeddense_panel.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_panel.go), [quantizeddense_ggml.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_ggml.go), [quantizeddense_linear.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_linear.go)):
   - Implemented `sharedQuantActGroup.acquire(xBuf, M, K, quantizeFn)` / `release(entry)` keyed by `*gobackend.Buffer` identity so multiple concurrent `Execute()` calls on the same `FunctionExecutable` (`num_embedders > 1`), `ops_parallel` sibling goroutines, and `ops_sequential` nodes all share safely without interference.
   - Deferred `bufferToFloat32Slice` (`float16`/`bfloat16 -> float32` input widening) inside the leader's `quantizeFn` so non-`float32` sibling projections also convert `x` only once, and automatically releases the pooled quantized activation buffer as soon as the final sibling (`remaining == 0`) completes.
3. **Concurrent Cold-Start `wConstantData` Fix** ([quantizeddense_ggml.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_ggml.go), [quantizeddense_ggml_unpack.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_ggml_unpack.go)):
   - Fixed a cold-start race uncovered by `TestSiblingSharedQuantActGGMLAndLinear` (`8` concurrent goroutines under `-race`) where `GetOrSetCachedForm` nils out donated `oldBuf.Flat` once `cachedForm` is populated: `execQuantizedDenseGGMLPanel` now reads `wBuf.Flat.([]uint8)` strictly inside `GetOrSetCachedForm`'s `computeFn` closure under `wConstantData.mu`.

#### Step 4 Final Benchmark Results

| Configuration (`-limit 20`, `57.2K` tok) | Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | vs Baseline | vs Updated FP32 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline GGUF (`Q4_K_M`, Scalar) | 24.1s | 8.30/s | 2.4K tok/s | 579 tok/s | 744.0s | 1.0× | 0.018× |
| Updated Unquantized `Float32` (AVX-512) | 429.0ms | 466/s | 133.2K tok/s | 32.5K tok/s | 9.8s | 56.2× | 1.00× |
| **Step 4 GGUF — AVX2 (`GOMLX_GO_SIMD_AVX512=0`)** | **485.9ms** | **412/s** | **117.6K tok/s** | **28.7K tok/s** | **11.7s** | **49.6×** | **0.88× (matches initial AVX-512 FP32!)** |
| **Step 4 GGUF — AVX-512 VNNI** | **277.2ms** | **721/s** | **206.2K tok/s** | **50.3K tok/s** | **5.8s** | **87.0×** | **1.55× faster (-41% CPU!)** |

*(Across 3 consecutive `-limit 20` AVX-512 VNNI runs: `275.4ms` / `207.6K tok/s`, `277.2ms` / `206.2K tok/s`, `283.5ms` / `201.6K tok/s`.)*

| Sustained Configuration (`-limit 200`, `236.7K` tok) | Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | Max RSS Mem |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Updated Unquantized `Float32` (`GOMLX_BACKEND=go`, AVX-512) | 1.8s | 1.1K/s | 128.0K–134.9K tok/s | 83.0K–87.5K tok/s | 42.1–42.9s | **494.8 MiB** |
| **Step 4 GGUF — AVX2 (`GOMLX_GO_SIMD_AVX512=0`)** | **2.08s** | **958/s** | **113.6K tok/s** | **73.7K tok/s** | **50.6s** | **~390 MiB** |
| **Step 4 GGUF — AVX-512 VNNI (Default `GOMLX_BACKEND=go`, `quantized_cache` off)** | **1.3s** | **1.5K/s** | **181.7K tok/s** | **117.8K tok/s** | **26.2s** | **390.5 MiB (-104 MiB vs FP32)** |
| **Step 4 GGUF — AVX-512 VNNI (`GOMLX_BACKEND="go:quantized_cache"`, `quantized_cache` on)** | **1.18–1.3s** | **1.5–1.7K/s** | **183.6K–200.2K tok/s** | **119.1K–129.8K tok/s** | **25.7–25.9s** | **401.8 MiB** |

*(On `-limit 20` compiled binary: GGUF default `GOMLX_BACKEND=go` uses **`235.9 MiB`** Max RSS at **`183.7K tok/s`**, `GOMLX_BACKEND="go:quantized_cache"` uses **`245.1 MiB`** at **`195.7K–206.2K tok/s`**, and unquantized `Float32` uses **`348.2 MiB`** at **`128.9K–133.2K tok/s`**.)*

#### Final Per-Op Sequential Breakdown (`GOMLX_OP_PROFILE=2`, per `M ≈ 8192` batch)

| OpType | Calls/Batch | Baseline Wall | Step 1 Wall | Step 4 Wall | Step 4 % | Avg Cores | Total Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`FusedQuantizedDense`** | 36 | 3,397.0 ms | 33.4 ms | **30.7 ms** | 77.6% | 20.0 | **110.7×** |
| `FusedLayerNorm` | 13 | 5.7 ms | 4.6 ms | **4.2 ms** | 10.7% | 15.6 | 1.36× |
| `FusedScaledDotProductAttention` | 6 | 2.8 ms | 2.8 ms | **2.3 ms** | 5.7% | 19.5 | 1.22× |
| `DynamicBroadcastInDim` | 3 | 6.6 ms | 7.7 ms | **0.35 ms** | 0.9% | 9.0 | **22.0× (vs Step 1)** |
| `Gather` | 2 | 0.7 ms | 0.7 ms | **0.27 ms** | 0.7% | 11.7 | **2.6×** |
| `QuantizedEmbeddingLookup` | 1 | 1.5 ms | 1.8 ms | **0.25 ms** | 0.6% | 14.9 | **7.2× (vs Step 1)** |
| `ConvertDType` (`Bool -> Float32`) | 3 | 1.1 ms | 1.2 ms | **0.24 ms** | 0.6% | 14.9 | **5.0× (vs Step 1)** |
| All other ops | 36 | 1.8 ms | 0.9 ms | **1.3 ms** | 3.2% | — | — |
| **Total Sum of Ops** | — | **3,416.5 ms** | **53.3 ms** | **39.6 ms** | **100.0%** | — | **86.3×** |

#### Final CPU Profile (`/tmp/gguf_step4_200.prof`, `34.80s` sampled CPU across warmup + 2,000 passages)

| Function | Flat CPU | Flat % | Cum CPU | Cum % | Status |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `avx512.quantizedPanelKernel6x32AVX512VNNIAsm` | 17.75s | **51.01%** | 17.75s | 51.01% | Handwritten AVX-512 VNNI `VPDPBUSD` assembly |
| `avx512.quantizeGGMLQ8_KRowsAVX512Asm` | 3.60s | **10.34%** | 3.60s | 10.34% | Handwritten AVX-512+F16C assembly (DRAM load bound on `50.3 MB` `[8192, 1536]` `ff1` output) |
| `avx512.layerNormResidualFloat32AVX512` | 2.36s | **6.78%** | 2.36s | 6.78% | Handwritten AVX-512 assembly (down from `3.06s` thanks to reduced Q/K/V DRAM contention!) |
| `avx512.sdpaQKT4RowsAVX512Asm` | 1.96s | **5.63%** | 1.96s | 5.63% | Handwritten AVX-512 assembly |
| `avx512.quantizeGGMLQ8_0RowsAVX512Asm` | 1.67s | **4.80%** | 1.67s | 4.80% | Handwritten AVX-512+F16C assembly (**-55% CPU** vs Step 3's `3.70s` via sibling Q/K/V sharing!) |
| `activations/avx512.geluApproxFloat32AVX512Asm` | 0.93s | **2.67%** | 0.93s | 2.67% | Handwritten AVX-512 assembly (fused in-tile) |
| `avx512.sdpaValueAccum4RowsAVX512Asm` | 0.81s | **2.33%** | 0.81s | 2.33% | Handwritten AVX-512 assembly |
| `avx512.sdpaScaleMaskSoftmaxRowAVX512Asm` | 0.72s | **2.07%** | 0.72s | 2.07% | Handwritten AVX-512 assembly |
| `runtime.encoderune` + `runtime.slicerunetostring` | 1.29s | **3.71%** | 3.03s | 8.71% | Host BERT WordPiece tokenizer (`go-huggingface`) |

---

### Step 5: Weight-Only Float Mode A (`quantized_matmul=false`), `QuantNF4`, and `FusedQuantizedDenseVJP` Dispatching

#### Root-Cause Diagnosis (`-set="quantized_matmul=false"`)
When users run quantized models with `-set="quantized_matmul=false"` (Mode A: exact float activations multiplied by dequantized weights, or `QuantNF4` where integer activation quantization does not apply):
1. **Redundant $O(M)$ Per-Token Weight Dequantization + Scalar Dot Loop**:
   - The baseline `quantizedDenseGGMLFloat` and `execWeightOnly{InputOutputs,OutputsInput}` dequantized every weight row inside the `M` loop (`8,192` times per batch!) and accumulated dot products with a scalar `for k := range K` loop, running at **`1.6K tokens/s`** (`25.3s` for `-limit 2`).
2. **Unpooled Per-Row Scratch Allocation Explosion**:
   - `quantizedDenseParallelTileCount` returned `M` (`16,384`) when `M > 1`, causing `make([][]float32, 16384)` $\times$ `make([]float32, K)` (`~98 MB` per layer = `1.8 GB` of heap garbage per forward pass!) and inflating Max RSS to **`405.1 MiB`**.

#### Changes Implemented ([quantizeddense_float_matmul.go](file:///home/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/quantizeddense_float_matmul.go))
1. **Bounded Pooled Chunk Dequantization + `matmul.ExecuteWithEpilogue` Dispatch**:
   - Added `execQuantizedFloatChunkedMatmul` covering `QuantGGML` (`dot.LayoutTransposed`), `QuantLinear`, and `QuantNF4` across both `DenseLayoutOutputsInput` (`[N, K]` $\to$ `dot.LayoutTransposed`) and `DenseLayoutInputOutputs` (`[K, N]` $\to$ `dot.LayoutNonTransposed`).
   - Caps temporary pooled `float32` weight buffers (`backend.GetBuffer`) at `defaultQuantizedFloatMaxScratchFloats = 1M` floats (`4 MiB`, `32`-column aligned). Standard Transformer layers (`N * K <= 1,048,576`, including `[1536, 384]`) dequantize once in parallel across workers (`< 0.5%` CPU) and execute in a **single zero-copy `matmul.ExecuteWithEpilogue` call** with inline L1/L2-tiled `Bias + Activation` epilogues; multi-billion-parameter matrices (`N * K > 1M`) stream bounded `4 MiB` `(chunkN, mChunk)` tiles with zero heap allocations after warmup.
   - Tightened BCE hints and hoisted sub-block scales in `dequantQ4_1Row`, `dequantQ5_0Row`, `dequantQ5_1Row`, `dequantIQ4NLRow`, `dequantQ4_KRow`, `dequantQ6_KRow`, and `dequantQ8_KRow`, and parallelized `Float16`/`BFloat16 <-> Float32` buffer conversions.
2. **Opt-In `go:quantized_cache` Support for Mode A**:
   - When `Backend.QuantizedCache` is enabled on constant weights (`wNode.IsConstant()`), attaches a `*dot.NodeData` with `RHSConstantData = gobackend.NewAnonymousConstantData()` so `matmul` packs and caches the FP32 GEMM panels on cold start and skips both dequantization and RHS packing (`rhs = nil`) on all subsequent calls (including across dynamic `M` shapes).
3. **`FusedQuantizedDenseVJP` Backward SIMD Matmul (`M >= 2`)**:
   - Routed `execQuantizedDenseVJPGGML` and `execQuantizedDenseVJPLinearOrNF4` (both `InputOutputs` and `OutputsInput` layouts) for `M >= 2` through `execQuantizedVJPOutputsInputMatmul` / `execQuantizedVJPInputOutputsMatmul` so QLoRA / quantized gradient backpropagation also leverages AVX-512 / AVX2 / NEON `matmul`, while retaining zero-skipping single-row (`[K]float32`) streaming for `M == 1`.

#### Step 5 Mode A Benchmark Results (`-set="quantized_matmul=false"`, `-limit 20`, `57.2K` tok)

| Mode A Configuration (`-set="quantized_matmul=false"`) | Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | Max RSS Mem | Speedup vs Mode A Baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mode A Baseline (`Q4_K_M`, Scalar) | ~35.7s | 5.6/s | 1.6K tok/s | 389 tok/s | ~1,100s | 405.1 MiB | 1.0× |
| **Step 5 Mode A — AVX2 (`GOMLX_GO_SIMD_AVX512=0`)** | **700.8ms** | **285/s** | **81.6K tok/s** | **19.9K tok/s** | **16.8s** | **245.7 MiB** | **51.0× (-159 MiB RSS)** |
| **Step 5 Mode A — AVX-512 (`GOMLX_BACKEND=go`, Default)** | **467.0ms** | **428/s** | **122.4K tok/s** | **29.9K tok/s** | **10.2s** | **255.6 MiB** | **76.5× (-150 MiB RSS)** |
| **Step 5 Mode A — AVX-512 (`GOMLX_BACKEND="go:quantized_cache"`)** | **438.5ms** | **456/s** | **130.3K tok/s** | **31.8K tok/s** | **9.8s** | **291.8 MiB** | **81.4× (matches FP32, -56 MiB vs FP32)** |

*(Mode A `-limit 50` `pprof` breakdown: `72.2%` `avx512LargeKernelFloat32Asm`, `6.3%` `avx512PackLHSKernelRows8Float32Asm`, `5.9%` `DispatchSDPAAVX512`, `4.6%` `layerNormResidualFloat32AVX512`, `1.4%` `geluApproxBiasTileFloat32AVX512Asm`; dense weight dequantization `dequantQ4_KRow`/`dequantQ6_KRow` accounts for `< 0.4%` of total CPU!)*

---

## 6. Stopping Rationale & Future Work (ARM64 NEON / SME)

### Why Stop Here
1. **>85.6% Process CPU (>93% of GoMLX Compute) in Handwritten Assembly**:
   - Every major kernel (`quantizedPanelKernel6x32AVX512VNNIAsm`, `quantizeGGMLQ8_KRowsAVX512Asm`, `layerNormResidualFloat32AVX512`, `sdpa*AVX512Asm`, `quantizeGGMLQ8_0RowsAVX512Asm`, `geluApproxFloat32AVX512Asm`) is now zero-spill handwritten AVX-512 / AVX-512 VNNI / AVX2 assembly.
2. **Remaining Costs Are Hardware-Bandwidth Bound or Outside `compute`**:
   - The single largest non-GEMM item (`quantizeGGMLQ8_KRowsAVX512Asm` at `10.34%`) streams the `50.3 MB` `[8192, 1536]` `ff1` intermediate tensor (`6 layers × 50.3 MB = 302 MB` per batch, exceeding even the 128 MB 3D V-Cache) from DDR5 DRAM. Eliminating that DRAM round-trip would require cross-op producer-consumer tiling fusion between `ff1` (`[M, 384] × [384, 1536] + GELU`) and `ff2` (`[M, 1536] × [1536, 384]`), which requires `ff1` to compute all `1,536` columns of an $M_c$ row block before `ff2` can quantize `256`-element `Q8_K` super-blocks—a major graph-compiler + fused-MLP rewrite for ~4–5% marginal gain.
   - Nearly ~5–8% of remaining wall/CPU time is now spent in the upstream WordPiece string tokenizer (`runtime.encoderune`, `slicerunetostring`) rather than tensor execution.

### Plug-In Path for ARM64 NEON (`SDOT`/`UDOT`/`USMMLA`) & Apple SME
The K4-interleaved panel architecture in [quantizeddense_arch.go](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/quantizeddense_arch.go) (`QuantizedPanelStripCols = 16`, `[quads][16][4]uint8` + `[16]float32` `d'_w` + `[16]float32` `m'_w`), SWAR weight unpackers, constant weight panel cache, and sibling activation quantization sharing are **100% architecture-independent**. Section 7 below details the ARM64 NEON (`FEAT_I8MM` `USDOT` / `FEAT_DotProd` `SDOT`) and Apple SME2 (`SUMOPA` + `FMOPA`) implementations built on top of this infrastructure.

---

## 7. ARM64 NEON (`FEAT_I8MM` / `FEAT_DotProd`) & Apple SME2 (`SUMOPA`) Optimization

### 7.1 Target Hardware & Baseline Profile (`darwin/arm64`)

- **Host CPU**: Apple Silicon M-series (`darwin/arm64`, 12 cores across 2 asymmetric L2 cache clusters: `clusterID=1` with 2 Super + 4 Performance cores sharing a 512-bit SME2 coprocessor at ~2.26 TFlops/s, and `clusterID=0` with 6 Efficiency cores sharing a 512-bit SME2 coprocessor at ~0.38 TFlops/s).
- **Supported ARMv9 Extensions**: `FEAT_DotProd` (`SDOT`/`UDOT`), `FEAT_I8MM` (`USDOT`/`USMMLA`), `FEAT_FP16`, `FEAT_BF16`, `FEAT_SME`, `FEAT_SME2` (`SUMOPA`, multi-vector `MOVA`/`SCVTF`).
- **ARM64 Baseline Benchmark** (`-limit 20`, `-bucket=8192`, `second-state/All-MiniLM-L6-v2-Embedding-GGUF`):
  - Before adding `internal/gobackend/fusedops/neon`, `FusedQuantizedDense` fell back to the scalar `NoSIMD` reference path (`execQuantizedDenseGGML` / `ggmlVecDotRow`):
    - **Default GGUF (`quantized_matmul=true`)**: **`21.5s`** wall time (`9.3 items/s`, **`2.7K tokens/s`**, **`649 non-pad tokens/s`**, CPU time `253.0s`, Max RSS `231.7 MiB`).
    - **Mode A (`-set="quantized_matmul=false"`)**: **`735.1ms`** wall time (`272 items/s`, **`77.7K tokens/s`**, **`19.0K non-pad tokens/s`**, CPU time `5.6s`, Max RSS `329.1 MiB`), as Mode A already leveraged the Step 5 bounded pooled chunk dequantizer + `matmul.ExecuteWithEpilogue` (which routes to the ARM64 SME/NEON `Float32` GEMM in `internal/gobackend/dot/matmul/neon`).
    - **Unquantized `Float32` Model**: **`715.3ms`** wall time (`280 items/s`, **`79.9K tokens/s`**, **`19.5K non-pad tokens/s`**, CPU time `5.5s`, Max RSS `390.0 MiB`).

### 7.2 Step 1 (ARM64): Handwritten NEON `USDOT` / `SDOT` Panel Microkernels & `Q8_0` / `Q8_K` Activation Quantizers

Implemented in [quantizeddense.go](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/neon/quantizeddense.go) and [quantizeddense_arm64.s](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/neon/quantizeddense_arm64.s), registered at `gobackend.PriorityNEON`:

1. **Dual-Mode `FEAT_I8MM` (`USDOT`) and `FEAT_DotProd` (`SDOT`) 3×16 & 1×16 Microkernels (`quantizedPanelKernel3x16NEONAsm`, `quantizedPanelKernel1x16NEONAsm`)**:
   - Consumes the exact same 16-column K4-interleaved weight strip (`[quads][16 cols][4 K-bytes]uint8` + `[16]float32` `d'_w` + `[16]float32` `m'_w`) designed in Step 1.
   - **Register Allocation (32 × 128-bit NEON registers `V0..V31`)**:
     - `V0..V11` (12 registers): 3 rows × 16 columns of `float32` accumulators (`3 × 4 × Float32x4`), held in registers across the entire `K` (`numSubBlocks`) loop and written to `Out` once at the end (with fused `Bias` initialization or `Accumulate` read-modify-write in registers).
     - `V12..V15` (4 registers): Per-sub-block `int32` dot-product accumulators for 1 row × 16 columns (`4 × Int32x4`).
     - `V16..V19` (4 registers): Weight `u_w` vectors (`[16 cols][4 K-bytes]uint8 = 64 bytes` per quad) and later per-sub-block `d'_w` scales (`[16]float32`).
     - `V20..V23` (4 registers): Per-sub-block `m'_w` scales (`[16]float32`).
     - `V24..V26` (3 registers): 3 rows of `Qx` (`[4 quads][4 K-bytes]int8 = 16 bytes` per row).
     - `V27..V30` (4 registers): Per-sub-block `int32` accumulators for Row 1 (`V27..V30`), while Row 2 uses `V20..V23` during the quad loop before `m'_w` is loaded.
   - **Inner `numMBlocks` Row Loop**: `quantizedPanelKernel3x16NEONAsm` loops over `numMBlocks` 3-row blocks inside a single assembly call, avoiding Go-to-assembly call overhead per 3-row step, with unrolled `quads == 8` (`subBlockSize = 32`) and `quads == 4` (`subBlockSize = 16`) fast paths.
   - **Hardware `USDOT` vs Sign-Flipped `SDOT`**:
     - When `gobackend.IsARM64I8MMAllowed` is true (Apple M2+, ARMv8.6-A+), sets `UseUSDOT: true` so weights remain unsigned `uint8` (`[0, 255]`) and executes indexed `USDOT Vd.4S, Vn.16B, Vm.4B[lane]` (`0x4F809C00`, signed `int8` activation × unsigned `uint8` weight $\to$ `int32`).
     - On older ARMv8.2-A CPUs with `FEAT_DotProd` only (`IsARM64I8MMAllowed == false`), sets `UseUSDOT: false` so weight unpackers flip `u_w ^ 0x80` and adjust `m'_w += 128 * d'_w`, and the assembly kernel executes indexed `SDOT Vd.4S, Vn.16B, Vm.4B[lane]` (`0x4F80E000`, signed `int8` × signed `int8` $\to$ `int32`).
2. **Handwritten NEON `Q8_0` & `Q8_K` Activation Quantizers (`quantizeGGMLQ8_0RowsNEONAsm`, `quantizeGGMLQ8_KRowsNEONAsm`)**:
   - Uses `FMAXV S16, V16.4S` for 32-element sub-block max-abs, hardware `FCVT H18, S17` + `FCVT S18, H18` for exact IEEE-754 FP16 `Q8_0` scale rounding (zero lookup tables or bit-shifts), `FRINTN V.4S` + `FCVTZS V.4S` for round-nearest-even quantization, `ADDV S, V.4S` for `sumQx`, and `SQXTN V.8H, V.4S` + `SQXTN2 V.16B, V.8H` + `SQXTN V.8B, V.8H` + `SQXTN2 V.16B, V.8H` to pack 16 `int32` lanes into 16 `int8` bytes in 4 instructions.
3. **Concurrent `FRECPE`+`FRECPS` + `FDIV` in `geluApproxFloat32NEONAsm` & Contiguous Tile `ActFn`**:
   - In [neon_arm64.s](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/activations/neon/neon_arm64.s), replaced the back-to-back `FDIV_4S` instructions in the 8-element `geluApproxFloat32NEONAsm` loop with a 2-step Newton-Raphson `FRECPE_4S` + 2× `FRECPS_4S` on `V0` executing concurrently on the vector FMA pipes while `FDIV_4S` executes on `V1` on the divide pipe, cutting `geluApproxFloat32NEONAsm` CPU time in half (`2.43s -> 1.25s` on `-limit 200`).
   - Invokes `cfg.ActFn` once on the contiguous slice `cfg.Out[mStart*N : mLimit*N]` per worker tile rather than once per row.

#### Step 1–2 (ARM64 NEON) Benchmark Results (`-limit 20` and `-limit 200`, Apple M-Series 12-Core)

| Configuration (`-limit 20`, `57.2K` total tokens, `14.0K` non-pad) | Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | Max RSS Mem | Speedup vs Baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Default GGUF (`Q4_K_M`, Scalar) | 21.5s | 9.3/s | 2.7K tok/s | 649 tok/s | 253.0s | 231.7 MiB | 1.0× |
| Mode A (`-set="quantized_matmul=false"`, `Q4_K_M`) | 735.1ms | 272/s | 77.7K tok/s | 19.0K tok/s | 5.6s | 329.1 MiB | 29.2× |
| Unquantized `Float32` Model | 715.3ms | 280/s | 79.9K tok/s | 19.5K tok/s | 5.5s | 390.0 MiB | 30.1× |
| **Step 1–2 ARM64 NEON (`USDOT` 3×16 + `Q8_0`/`Q8_K` + `GELU`)** | **678.6ms** | **295/s** | **84.2K tok/s** | **20.5K tok/s** | **7.6s** | **233.3 MiB** | **31.7× (faster than FP32, -157 MiB RSS)** |

On **`-limit 200`** (`238.5K` total tokens, `154.6K` non-pad tokens):
- **Step 1–2 ARM64 NEON**: **`3.0s`** wall time (`663 items/s`, **`78.7K tokens/s`**, **`51.0K non-pad tokens/s`**, CPU time `33.7s`, Max RSS `365.7 MiB`).

---

### 7.3 Step 3–4 (ARM64 SME2 + NEON): Hybrid SME2 `SUMOPA` + `FMOPA` Panel GEMM & Handwritten NEON `FusedLayerNorm` / `FusedLayerNormResidual`

Implemented in [quantizeddense_sme.go](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/neon/quantizeddense_sme.go), [quantizeddense_sme_arm64.s](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/neon/quantizeddense_sme_arm64.s), [layernorm.go](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/neon/layernorm.go), and [layernorm_arm64.s](file:///Users/janpf/Projects/gomlx/compute/internal/gobackend/fusedops/neon/layernorm_arm64.s):

1. **SME2 `SUMOPA` + `FMOPA` 16×16 Outer-Product Kernel with Zero-Repack Weight Strips (`smeQuantizedPanelRowsAsm`, registered at `gobackend.PrioritySME`)**:
   - **Zero-Repack Compatibility**: Because each 16-column K4-interleaved weight strip stores `u_w` as `[quads][16 cols][4 K-bytes]uint8` (`64 bytes` per quad = one 512-bit SVE `Z` register) followed by `d'_w` (`[16]float32` = `64 bytes`) and `m'_w` (`[16]float32` = `64 bytes`), SME2's 512-bit `SUMOPA ZA0.S, P0/M, P0/M, Z_qx.B, Z_uw.B` (`0xA080A000`, signed `int8` LHS × unsigned `uint8` RHS 4-way outer product) and `FMOPA ZA1.S, P1/M, P1/M, Z_dxSumQx.S, Z_mw.S` (`0x80800000`, FP32 outer product) consume the exact same `WeightPanels` buffer as the 128-bit NEON `USDOT` kernel with **zero weight repacking**!
   - **Single `SMSTART`..`SMSTOP` Session with In-`ZA` Hardware LHS Transposition**:
     - **Phase 0 (Hardware `ZA0.S` 16×16 Matrix Transpose of `Qx` and `DxPairs`)**: Inside `SMSTART`, transposes 16 rows of `Qx` (`[activeRows, totalQuads]uint32` $\to$ `[totalQuads, 16]uint32`) and `DxPairs` (`[activeRows, numSubBlocks*2]float32` $\to$ `[numSubBlocks*2, 16]float32`, placing `dx[0..15]` at `#0, MUL VL` and `(dx*sumQx)[0..15]` at `#1, MUL VL`) in 16×16 blocks via `LD1W {ZA0H.S[W12, 0]}` + `ST1W {ZA0V.S[W12, 0]}` into a 64 KB `sync.Pool` scratch buffer.
     - **Phase 1 (Dual-Tile `SUMOPA ZA0.S` + `FMOPA ZA1.S` + Multi-Vector `MOVA`/`SCVTF`)**:
       - `ZA1.S` accumulates `Bias` (or zero when `Accumulate`) plus $\sum_{sb} (d_{x,m,sb} \cdot \text{sum}_{q_x,m,sb}) \cdot m'_{w,sb,n}$ via one `FMOPA ZA1.S` instruction (256 FP32 FMAs) per sub-block across all sub-blocks without ever extracting `ZA1.S` inside the $K$-loop.
       - For each sub-block, zeroes only `ZA0.S` (`ZERO {ZA0.S}`, `0xC0080011`), executes 8 (`subBlockSize=32`) or 4 (`subBlockSize=16`) `SUMOPA ZA0.S` instructions (**1,024 INT8 MACs per instruction**), extracts 4 rows at a time via SME2 4-vector `MOVA {Z0.S-Z3.S}, ZA0H.S[W12, 0:3]` + 4-vector `SCVTF {Z0.S-Z3.S}, {Z0.S-Z3.S}`, and accumulates $\sum_{sb} \text{float32}(ZA0_{m,n}) \cdot d_{x,m,sb} \cdot d'_{w,sb,n}$ into 16 streaming `Z` registers (`Z12..Z27`, arranged in four 4-vector aligned groups `{Z12..Z15}`, `{Z16..Z19}`, `{Z20..Z23}`, `{Z24..Z27}`).
       - After all sub-blocks complete, extracts `ZA1H.S` (`Bias + sum_{sb} (dx*sumQx)*m'_w`), adds `Z12..Z27`, stores the completed 16×16 tile into `ZA0H.S`, and verifies the `Z31` upper-lane hardware canary (`LASTB W11, P7, Z31.S`) before touching `Out` memory (protecting against macOS `SIGURG` `sigreturn` upper-bit zeroing of streaming `Z` registers).
     - **Phase 2 (Predicated Store from `ZA0.S`)**: Writes `activeRows × activeCols` to `Out` using column predicate `P2` (`WHILELT P2.S, Xzr, X16`) and verifies the `Z31` canary once more.
2. **Heterogeneous SME2 + NEON Co-Execution over Dual-Ended Atomic Row Queue**:
   - Coordinates workers via a 64-bit `bounds atomic.Uint64` (`(headRow << 32) | tailRow`) queue under `Scheduler.AcquireWholeChip()` FIFO gating:
     - **SME Workers (`headRow` $\to$ up)**: Workers that acquire their L2 cluster's SME lock (`simdarch.AcquireSMEWithLimit(cfg.MaxWorkers, 1)`) claim 16/32-row chunks from `headRow` and execute `smeQuantizedPanelRowsAsm` across all column strips. If XNU migrates a worker between clusters (`simdarch.CurrentSMECluster() != smeCluster`), it immediately releases the old cluster lock and re-acquires for its new cluster.
     - **NEON Workers (`tailRow` $\to$ down)**: Workers on cores whose cluster SME unit is busy claim 3/6/12-row chunks from `tailRow`, running `quantizedPanelKernel3x16NEONAsm` on the same shared `WeightPanels` and yielding early (`remRows <= 16 * smeActive`) while spinning until `allClaimed` so Go's scheduler keeps 1 goroutine pinned per physical CPU core.
3. **Handwritten ARM64 NEON `FusedLayerNorm` & `FusedLayerNormResidual` Assembly (`layerNormFloat32NEONAsm`, `layerNormResidualFloat32NEONAsm`)**:
   - Registered `DispatchLayerNormNEON` and `DispatchLayerNormResidualNEON` at `gobackend.PriorityNEON`.
   - Uses 4-way unrolled 16-float (`64-byte`) vector loads/stores (`VLD1.P 64(Rn), [V4.S4, V5.S4, V6.S4, V7.S4]` and `VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(Rn)`), 4 independent accumulator chains (`V0..V3`) reduced via `VFADDP` + `FADDP`, hardware `FSQRT S1, S1`, and fused `in + residual -> out` in Pass 1 of `layerNormResidualFloat32NEONAsm` so Pass 2 and Pass 3 read/write `out` while hot in L1D cache.

---

### 7.4 Final ARM64 (NEON + SME2) Benchmark Results & Progression

#### Table A: Short Run (`-limit 20`, 200 Passages, `57.2K` Total Tokens, `13.9K` Non-Pad Tokens, Apple M-Series 12-Core)

| Configuration (`-bucket=8192 -num_embedders=1 -warmup -dynamic=true`) | Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | Max RSS Mem | Speedup vs Baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline Default GGUF (`Q4_K_M`, Scalar `NoSIMD`)** | **21.5s** | **9.3/s** | **2.7K tok/s** | **649 tok/s** | **253.0s** | **231.7 MiB** | **1.0×** |
| Unquantized `Float32` Model (`all-MiniLM-L6-v2`) | 715.3ms | 280/s | 79.9K tok/s | 19.5K tok/s | 5.5s | 390.0 MiB | 30.1× |
| Mode A (`-set="quantized_matmul=false"`, `GOMLX_BACKEND=go`) | 721.7ms | 277/s | 79.2K tok/s | 19.3K tok/s | 8.1s | 328.6 MiB | 29.8× (-61 MiB vs FP32) |
| Mode A (`-set="quantized_matmul=false"`, `GOMLX_BACKEND="go:quantized_cache"`) | 705.9ms | 283/s | 81.0K tok/s | 19.8K tok/s | 7.9s | 365.3 MiB | 30.5× (-25 MiB vs FP32) |
| Step 1–2 ARM64 NEON (`USDOT` 3×16 + `Q8_0`/`Q8_K` + `GELU`) | 678.6ms | 295/s | 84.2K tok/s | 20.5K tok/s | 7.6s | 233.3 MiB | 31.7× (-157 MiB vs FP32) |
| **Step 3–4 ARM64 Hybrid SME2 (`SUMOPA`+`FMOPA`) + NEON (`USDOT` + `LayerNorm`)** | **635.6ms** | **315/s** | **89.9K tok/s** | **21.9K tok/s** | **7.1s** | **233.7 MiB** | **33.8× (1.13× faster than FP32, -156 MiB RSS)** |

#### Table B: Extended Run (`-limit 200`, 2,000 Passages, `236.7K` Total Tokens, `153.5K` Non-Pad Tokens, Apple M-Series 12-Core)

| Configuration (`-limit 200 -bucket=8192 -num_embedders=1 -warmup -dynamic=true`) | Wall Time | Passages/s | Total Tokens/s | Non-Pad Tokens/s | CPU Time | Max RSS Mem |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mode A (`-set="quantized_matmul=false"`, `GOMLX_BACKEND=go`) | 3.2s | 627/s | 74.4K tok/s | 48.3K tok/s | 35.6s | 451.0 MiB |
| Step 1–2 ARM64 NEON (`USDOT` 3×16 + `Q8_0`/`Q8_K` + `GELU`) | 3.0s | 663/s | 78.7K tok/s | 51.0K tok/s | 33.7s | 365.7 MiB |
| **Step 3–4 ARM64 Hybrid SME2 (`SUMOPA`+`FMOPA`) + NEON (`USDOT` + `LayerNorm`)** | **2.9s** | **699/s** | **82.9K tok/s** | **53.7K tok/s** | **32.3s** | **373.0 MiB** |

#### Table C: Unpadded Single-Sentence Multi-Embedder Mode (`-bucket=0 -num_embedders=12 -warmup -dynamic=true`, Apple M-Series 12-Core)

When `-bucket=0 -num_embedders=12` is used (`batchSize = 1`, `seqLen` dynamic with zero padding tokens, 12 concurrent embedders across 12 cores where `Scheduler.EffectiveParallelism() == 1`):
- In `DispatchQuantizedPanelGEMMSME`, single-worker / small-$M$ invocations (`EffectiveParallelism() <= 1` or `totalWork <= 8M`) bypass `Scheduler.AcquireWholeChip()` and attempt a non-blocking per-cluster SME lock (`simdarch.AcquireSMEWithLimit`). Up to 2 concurrent embedders (one on Cluster 1 SME, one on Cluster 0 SME) execute `smeQuantizedPanelRowsAsm` while the remaining 10 concurrent embedders execute `quantizedPanelKernel3x16NEONAsm` in parallel on their private per-core NEON `USDOT` pipelines (~`11.7` active cores, zero lock contention).
- Eliminating slice bounds checks in `writeStripSubBlock32` / `writeStripSubBlock16` (`*[160]uint32` / `*[96]uint32` with `col &= 15`) further cut per-sentence weight strip unpacking time by >50%.

| Configuration (`-bucket=0 -num_embedders=12 -warmup -dynamic=true`) | Limit (Passages) | Wall Time | Passages/s | Non-Pad Tokens/s | CPU Time | ~Cores | Max RSS Mem |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Default GGUF (`GOMLX_BACKEND=go`)** | `-limit 20` (200) | **198.7ms** | **1,000/s** | **70.2K tok/s** | **2.3s** | **11.4** | **179.5 MiB** |
| **Default GGUF (`GOMLX_BACKEND=go`)** | `-limit 200` (2,000) | **2.1s** | **952/s** | **73.2K tok/s** | **24.6s** | **11.7** | **321.9 MiB** |
| **Default GGUF (`GOMLX_BACKEND="go:quantized_cache"`)** | `-limit 200` (2,000) | **1.8s** | **1,100/s** | **85.8K tok/s** | **21.0s** | **11.7** | **319.6 MiB** |
| Mode A (`-set="quantized_matmul=false"`, `GOMLX_BACKEND=go`) | `-limit 200` (2,000) | 4.9s | 406/s | 31.3K tok/s | 47.6s | 9.7 | 413.1 MiB |
| Mode A (`-set="quantized_matmul=false"`, `GOMLX_BACKEND="go:quantized_cache"`) | `-limit 200` (2,000) | 2.9s | 685/s | 52.7K tok/s | 25.7s | 8.8 | 488.1 MiB |





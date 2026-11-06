# Diffusion / FlowMatching Performance Analysis & Optimization Log

**Date**: 2026-11-05  
**Reference Workload**: `FlowMatching` U-Net training demo (`batch=32`, $64 \times 64 \times 3$ Oxford Flowers 102, 4-level U-Net with Sinusoidal Embedding, Residual Blocks, `swish`/`silu` activations, and spatial `LayerNorm(axes=[1, 2])`)  
**Hardware**: AMD Ryzen 9 9950X3D (16-Core / 32-Thread, AVX-512, dual-channel DDR5)  
**Command** (run from `../gomlx`):
```bash
GOEXPERIMENT=simd GOMLX_BACKEND=go go run ./examples/FlowMatching/demo/ -set="train_steps=10" -eval=false
```

---

## 1. Baseline Profiling Metrics

- **Baseline Median Train Step Duration**: **`923.0 – 925.8 ms`** (`~925,794 µs`)
- **Target (`xla:cpu` on same hardware)**: **`~615 ms`** (~2/3 of the `go` backend duration)
- **CPU Profile (10 training steps after 1 warm-up step = `9.23 s` wall time)**:
  - **Total CPU time**: **`127.74 s`** (`12.77 s` CPU/step $\implies$ ~13.8$\times$ average thread parallelism across 32 hardware threads)
- **Memory Profile (10 training steps)**:
  - **Total Heap Allocations (`alloc_space`)**: **`5,920.80 MB`** (`~592.1 MB/step`, triggering `~6.8 s` of background GC CPU time across 10 steps)
  - `Backend.GetBuffer` (`sync.Pool` misses across 32 `P`s): **`4,712.70 MB`** (79.6%)
  - `fusedDenseVJPGeneric` (unpooled `make([]float32)` for `xT`, `dZ`, `dZT`): **`687.25 MB`** (11.6%)

### Baseline CPU Profile Breakdown (10 Steps = `127.74 s` Total CPU)

| Category | Function / Operation | Cum CPU (s) | % Total CPU | SIMD Status / Bottleneck |
| :--- | :--- | ---: | ---: | :--- |
| **1. Convolutions** | `execConvStandardAVX512` (`ConvAVX512Kernel8x32` `14.61s` + `ExtractPatches2DFloat32` `7.14s`) | 22.32 s | 17.5% | AVX-512 kernel, but `ExtractPatches2D` spends `6.27s` in `runtime.memmove` |
| | `execConvKernelGradAVX512` (`ConvKernelGradAVX512_8x32Float32Asm`) | 20.01 s | 15.7% | AVX-512 kernel, but re-streams 16 MB input & grad tensors from RAM for every $8\times 32$ channel tile |
| **2. Binary Ops** | `execMulSIMD` (`14.70s`) + `execMulNumericGeneric` (`2.47s`) | 17.17 s | 13.4% | **Falls back to scalar `ZipIterator` (`2.47s`)** on 3-segment `[B, H, W, C] * [1, H, W, 1]`; suboptimal `BroadcastMidRHS` |
| | `execAddSIMD` | 11.79 s | 9.2% | Suboptimal `BroadcastMidRHS` dispatch (calls per-row `vv` on `C=32..128` instead of register-pinned `v2v1`) |
| | `execDivSIMD` | 9.96 s | 7.8% | Used in un-fused LayerNorm backward pass (`[B, H, W, C] / [B, 1, 1, C]`) |
| | `execSubSIMD` | 3.53 s | 2.8% | Used in LayerNorm backward pass & residual blocks |
| **3. LayerNorm** | `execFusedLayerNorm` $\to$ `layerNormStridedAxes` | 7.27 s | 5.7% | **100% Scalar Fallback!** `axes=[1, 2]` on `[B, H, W, C]` is rejected by `execFusedLayerNormSIMD` |
| **4. Concat / Slice** | `execConcatenate` (`4.12s`) + `execSliceGeneric` (`3.62s`) | 7.74 s | 6.1% | Spend `7.25s` in `runtime.memmove` (2$\times$ nearest-neighbor upsampling & skip connections) |
| **5. Activations** | `vjpSiluFloat32SIMD` (`3.65s`) + `siluFloat32SIMD` (`2.37s`) | 6.02 s | 4.7% | Forward `activations.Register` does a redundant `1.64s` single-threaded `copy(out, in)` when `x` is kept for VJP |
| **6. Dense VJP** | `execFusedDenseVJP` $\to$ `transpose2D` | 5.13 s | 4.0% | **100% Scalar & Single-Threaded `transpose2D` (`4.36s`)** + unpooled `make([]T)` (`687 MB`) |
| **7. Reductions** | `execReduceSumGeneric` | 4.15 s | 3.3% | **100% Scalar Fallback!** `ReduceMiddle` (`axes=[1, 2]` on `[B, H, W, C]`) is rejected by `canExecuteReduceSIMD` |

---

## 2. Optimization Opportunities (Ordered by Simplicity)

### Opportunity 1: Add `ReduceMiddle` Support to `reduce_simd.go` (`4.15 s` CPU, 3.3%)
- **Status**: Completed
- **Problem**:
  - In the backward pass of spatial LayerNorm (`axes=[1, 2]`), `ReduceSum` reduces `[B, H, W, C]` $\to$ `[B, 1, 1, C]`, which classifies as `ReduceMiddle` (`A = B = 32`, `B_reduced = H*W`, `C = Channels = 32..128`).
  - `canExecuteReduceSIMD` (`internal/gobackend/ops/reduce_simd.go`) only handled `ReduceTrailing`, `ReduceLeading`, and `ReduceAll`, returning `false` for `ReduceMiddle` and falling back to scalar `execReduceSumGeneric`.
- **Solution**:
  - Enabled `ReduceMiddle` in `canExecuteReduceSIMD` and dispatched it across `dispatchReduceSumSIMD`, `dispatchReduceMaxSIMD`, `dispatchReduceMinSIMD`, and `dispatchReduceProductSIMD` using the existing `ReduceLeading` SIMD and AVX-512 assembly kernels (`reduceLeadingSumFloat32AVX512` with an added 2-vector `col32` unrolled path for `C=32, 96`), parallelized across `A`.
- **Result**:
  - `execReduceSumGeneric` completely eliminated from the profile (`4.15 s` $\to$ `0 s`); replaced by `reduceLeadingSumFloat32AVX512` (`3.42 s` CPU across 32 threads, memory-bandwidth bound on 16 MB tensors).

### Opportunity 2: Parallelize & Pool `FusedDenseVJP` (`transpose2D`) (`5.13 s` CPU, 4.0% + `68.7 MB/step` Heap Allocations)
- **Status**: Completed
- **Problem**:
  - In `fusedDenseVJPGeneric` (`internal/gobackend/fusedops/dense/dense_vjp.go`):
    1. `transpose2D(xFlat, m, k, xT)` and `transpose2D(dZ, m, n, dZT)` ran completely **single-threaded** and scalar (`4.36 s` flat CPU time = **`436 ms/step` of single-threaded critical-path wall time**, blocking all other worker threads while 1 core transposed `m = 131,072` rows!).
    2. `dZ = make([]T, m*n)`, `xT := make([]T, k*m)`, and `dZT := make([]T, n*m)` allocated directly on the Go heap via `make([]T)` instead of `backend.GetBuffer`, generating **`687.25 MB`** of garbage across 10 steps and causing heavy GC + `sync.Pool` eviction.
- **Solution**:
  - Pooled all temporary buffers (`dZ`, `xT`, `dZT`, `tmpF32_dX`, `tmpF32_dW`) via `backend.GetBuffer` / `backend.PutBuffer`.
  - Parallelized `transpose2D` across `backend.Scheduler.ParallelizeChunks` using cache-tiled $32 \times 32$ blocks, and optimized `reduceBiasFloat` to accumulate contiguous column slices row-by-row.
- **Result**:
  - **Median train step duration dropped from `923.5 ms` down to `671.7 ms` (`-251.8 ms/step`, a 27.3% wall-clock speedup!)**.
  - Total heap allocations across 10 steps dropped from **`5,920.8 MB` down to `2,880.1 MB` (`-51.4%` heap allocations!)** because eliminating the `687 MB` of unpooled `make([]T)` allocations prevented mid-step GCs from flushing `Backend.bufferPools` (`sync.Pool`).

### Opportunity 3: Avoid Redundant Out-of-Place `copy(out, in)` in `activations.Register` (`1.64 s` CPU in `runtime.memmove`)
- **Status**: Completed
- **Problem**:
  - During training, `x` (input to `FusedActivation`) is retained for `FusedActivationVJP`, so `inputsOwned[0]` is `false` and `out` is a distinct buffer from `in`.
  - AVX-512 and AVX2 forward activation kernels (`SiluAVX512`, `ReluAVX512`, `TanhAVX512`, etc.) were registered as `InPlaceFn` (`func([]T)`) via `activations.Register` (`internal/gobackend/activations/activations.go`), which copied the entire chunk (`copy(out, in)`) before running `fn(out)` — reading/writing `out` twice (`1.64 s` spent in `runtime.memmove`).
- **Solution**:
  - Converted AVX-512 and AVX2 `float32` forward activation kernels (`SiluKernelAVX512`, `ReluKernelAVX512`, `SigmoidKernelAVX512`, `HardSigmoidKernelAVX512`, `LeakyReluKernelAVX512`, `SeluKernelAVX512`, `HardSwishKernelAVX512`, `TanhKernelAVX512`, and their AVX2 counterparts) to `(in, out []float32)` registered via `RegisterKernel`.
- **Result**:
  - Eliminated the `1.64 s` `activations.Register.func1` `runtime.memmove` copy completely; median train step duration in profiling harness is **`671.0 ms`**.

### Opportunity 4: Fix Binary Op `ZipIterator` Fallback (`[A, B, C]` op `[1, B, 1]`) & `BroadcastMidRHS` Dispatch (`~42.4 s` CPU, 33.2%)
- **Status**: Completed
- **Problem**:
  1. **3-Segment Broadcast Fallback (`2.47 s` CPU in `execMulNumericGeneric` / `ZipIterator`)**:
     - Spatial LayerNorm (`axes=[1, 2]`) creates learned `gamma` and `beta` tensors of shape `[1, H, W, 1]`.
     - Multiplying `[B, H, W, C]` by `[1, H, W, 1]` produces 3 segments: `[modeBcastRHS (B), modeMatch (H*W), modeBcastRHS (C)]`.
     - `DetermineBroadcastConfig` (`internal/gobackend/broadcast.go`) only handled `[modeMatch, modeBcast*, modeMatch]` in `case 3:`, so `[modeBcast*, modeMatch, modeBcast*]` fell back to `BroadcastGeneral` and scalar `ZipIterator`!
  2. **`BroadcastMidRHS` Dispatch (`internal/gobackend/ops/binaryops_simd.go`)**:
     - For `[B, H*W, C]` op `[B, 1, C]` (`A=32, B=4096, C=32..128`), `bc = B*C > 4096`, so `dispatchBinarySIMD` skipped `v2v1` (which pins `rhs` in SIMD registers across `B` rows) and instead looped `for ab := abStart; ab < abEnd; ab++` calling `vv` on tiny slices of length `C=32..128` (`131,072` calls per op).
- **Solution**:
  - Supported `[modeBcastRHS, modeMatch, modeBcastRHS]` (`BroadcastOuterInnerRHS`) and `[modeBcastLHS, modeMatch, modeBcastLHS]` (`BroadcastOuterInnerLHS`) in `DetermineBroadcastConfig` and `dispatchBinarySIMD` (broadcasting each scalar `rhs[b]` across contiguous `C`-element slices via `vs`/`sv`).
  - Used `v2v1` / `v1v2` inside the parallel `BroadcastMidRHS` / `BroadcastMidLHS` loops even when `B*C > 4096`, and added register-pinned `B == 4*vLen, 6*vLen, 8*vLen` (`C = 64, 96, 128`) fast paths to `simdV2V1AddFloat32`, `simdV2V1MulFloat32`, `simdV2V1SubFloat32`, `simdV1V2SubFloat32`, `simdV2V1DivFloat32`, and `simdV1V2DivFloat32`.
- **Result**:
  - **Median train step duration dropped from `668.7 ms` (`671.0 ms` in profiler) down to `575.1 – 577.7 ms` (`-93.6 ms/step`, an additional 14.0% wall-clock speedup — now faster than `xla:cpu` at `~615 ms`!)**.
  - `execMulNumericGeneric` (`ZipIterator` fallback) completely eliminated from the profile.

### Opportunity 5: Vectorize Non-Trailing `FusedLayerNorm` (`axes=[1, 2]` on `[B, H, W, C]`) (`7.27 s` CPU, 5.7%)
- **Status**: Completed
- **Problem**:
  - `execFusedLayerNormSIMD` (`internal/gobackend/fusedops/layernorm_simd.go`) checked `isTrailingAxes(xShape.Rank(), data.axes)` and returned `ErrFallback` for `axes=[1, 2]` on `[B, H, W, C]`.
  - It fell back to generic scalar `layerNormStridedAxes` (`internal/gobackend/fusedops/layernorm.go`) with `aDim = B (32)`, `bDim = H*W`, `cDim = C (32..128)`, and `dDim = 1`.
  - Because `dDim == 1`, the inner loop ran scalarly over a 1-element slice `dDim = 1` instead of vectorizing across the contiguous channel dimension `cDim = C`!
- **Solution**:
  - Added `simdLayerNormMiddleAxesFloat32` in `internal/gobackend/fusedops/layernorm_simd.go` for strided/middle-axis LayerNorm when `dDim == 1 && cDim > 1`, accumulating `means[0:C]` and `variances[0:C]` across `b := range bDim` in register-pinned `4*vLen` / `2*vLen` / `vLen` SIMD vector blocks along `cDim`, and parallelized across `aDim`.
- **Result**:
  - Eliminated scalar `layerNormStridedAxes` (`7.51 s` $\to$ `0 s`); replaced by `simdLayerNormMiddleAxesFloat32` (`4.60 s` cum CPU across 32 threads, memory-bandwidth bound). Median train step duration: **`577.2 ms`**.

### Opportunity 6: `ConvGeneral` (`ConvKernelGrad` Cache Tiling & `ExtractPatches2D` Optimization) (`42.33 s` CPU, 33.1%)
- **Status**: Completed
- **Problem**:
  - `execConvKernelGradAVX512` (`20.01 s` CPU) streamed the full `BatchSize * OutH * OutW` (`16–64 MB`) input and output-gradient tensors from RAM for every `(ky, kx, rowPanel=8, colPanel=32)` task, and had a low task count (`36` tasks across 32 threads = `32 + 4`, leaving 28 out of 32 threads idle during the second wave) when `Cin=32, Cout=32`.
  - `ExtractPatches2DFloat32` (`7.14 s` CPU, `6.27 s` in `runtime.memmove`) materialized full `im2col` patches (`9×` the input tensor size = 144 MB per $64 \times 64$ convolution) even for interior 2D pixels where each filter row `ky` is already contiguous in the `[B, InH, InW, InC]` input tensor with constant horizontal pixel stride `StrideW * InC`.
- **Solution**:
  1. **Added `Diffusion-LargeBatch` and `SmallBatch-LargeSpatial` benchmarks** to `support/backendtest/bench_convgeneral.go` covering both `(BatchSize=32, Spatial=64x64..16x16)` and `(BatchSize=1..2, Spatial=256x256..128x128)` regimes.
  2. **Direct Zero-`im2col` 2D Forward & VJP-Input Microkernels (`ConvAVX512Direct2DKernel8x32Float32Asm`, `ConvAVX512Direct2DKernel4x32Float32Asm`, `ConvAVX2Direct2DKernel4x16Float32Asm`)**:
     - For 2D convolutions with `InDilH == 1 && InDilW == 1 && KerDilW == 1` (`outW >= 16, kW*InC >= 16`), interior horizontal pixels `outX in [outXIntStart, outXDirectEnd)` are multiplied directly from `input` across `numKy = kyEnd - kyStart` valid filter rows of width `kWC = kW * InC`, advancing the 8 (or 4) row pointers by `lhsKyJumpBytes = (KerDilH*InW - kW)*InC*4` between filter rows.
     - Only the border/remainder pixels on each `(b, outY)` row (e.g., 4 out of 64 pixels for a $3 \times 3$ padded convolution — a **16× to 64× reduction** in patch extraction) use `ExtractPatches2DFloat32`.
  3. **Unified `(BatchSize, OutH)` Reduction Slicing in `execConvKernelGradAVX512`**:
     - Kept the high-arithmetic-intensity $8 \times 32$ outer-product microkernel (`rowTile = 8`, 8 FMAs per ZMM load) and split the `totalBY = BatchSize * OutH` reduction domain into `numSlices` (`1, 2, 4, or 8`) slices when `baseTasks` is small or unevenly distributed across `maxParallelism` (e.g., `36 * 8 = 288 = 9 * 32` tasks for `32x32` channels, and `144 * 2 = 288 = 9 * 32` tasks for `64x64` channels).
     - Because slicing splits across `BatchSize` when `BatchSize >= numSlices` and across `OutH` within each batch when `BatchSize < numSlices`, it preserves full 1D inner-loop collapsing and accelerates both `(LargeBatch, MediumSpatial)` and `(SmallBatch, LargeSpatial)` shapes.
- **Result**:
  - `ExtractPatches2DFloat32` (`6.37 s` in baseline profile) dropped out of the top 25 CPU consumers (`< 0.67 s`), and `runtime.memmove` dropped from `15.56 s` to `8.70 s`.
  - **Median train step duration dropped from `577.2 ms` down to `563.0 ms` (`564.2 ms` in profiler; `-362.8 ms/step` vs baseline, a 39.2% total reduction — 1.64× faster than baseline and ~52 ms/step faster than `xla:cpu`!)**.

### `BenchmarkCompliance/ConvGeneral` Results (Before vs. After Opportunity 6)

| Benchmark Suite / Case | Baseline Time | Baseline Throughput | Optimized Time | Optimized Throughput | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `MNIST-CNN/Conv1VJPKernel:32x28x28x1_v32x28x28x32` | 78.1 µs | 185.1 GFlops/s | **54.6 µs** | **264.5 GFlops/s** | **+42.9%** |
| `MNIST-CNN/Conv2VJPKernel:32x14x14x32_v32x14x14x64` | 156.5 µs | 1,478 GFlops/s | **143.0 µs** | **1,617 GFlops/s** | **+9.4%** |
| `Diffusion-LargeBatch/Fwd3x3:32x64x64x32_k3x3x32x32` | 1,296 µs | 1,864 GFlops/s | **1,094 µs** | **2,208 GFlops/s** | **+18.5%** |
| `Diffusion-LargeBatch/Fwd3x3:32x32x32x64_k3x3x64x64` | 988.7 µs | 2,443 GFlops/s | **917.8 µs** | **2,632 GFlops/s** | **+7.7%** |
| `Diffusion-LargeBatch/VJPInput3x3:32x64x64x32_k3x3x32x32` | 1,545 µs | 1,564 GFlops/s | **1,199 µs** | **2,015 GFlops/s** | **+28.8%** |
| `Diffusion-LargeBatch/VJPKernel3x3:32x64x64x32_v32x64x64x32` | 1,072 µs | 2,253 GFlops/s | **834.4 µs** | **2,895 GFlops/s** | **+28.5%** |
| `Diffusion-LargeBatch/VJPKernel3x3:32x32x32x64_v32x32x32x64` | 981.2 µs | 2,462 GFlops/s | **985.3 µs** | **2,452 GFlops/s** | ~1.00× |
| `Diffusion-LargeBatch/VJPKernel3x3:32x16x16x96_v32x16x16x96` | 654.6 µs | 2,076 GFlops/s | **598.0 µs** | **2,273 GFlops/s** | **+9.5%** |
| `SmallBatch-LargeSpatial/Fwd3x3:1x256x256x32_k3x3x32x32` | 755.2 µs | 1,600 GFlops/s | **620.5 µs** | **1,947 GFlops/s** | **+21.7%** |
| `SmallBatch-LargeSpatial/Fwd3x3:2x128x128x64_k3x3x64x64` | 1,001 µs | 2,413 GFlops/s | **868.9 µs** | **2,780 GFlops/s** | **+15.2%** |
| `SmallBatch-LargeSpatial/VJPKernel3x3:1x256x256x32_v1x256x256x32` | 536.0 µs | 2,254 GFlops/s | **468.5 µs** | **2,579 GFlops/s** | **+14.4%** |
| `SmallBatch-LargeSpatial/VJPKernel3x3:2x128x128x64_v2x128x128x64` | 999.9 µs | 2,416 GFlops/s | **946.9 µs** | **2,552 GFlops/s** | **+5.6%** |

---

## 3. Optimization Progress Log

| Step | Optimization | Median Train Step Duration (ms) | Total CPU Time / 10 steps (s) | Heap Allocs / 10 steps (MB) | Notes |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **0. Baseline** | Initial profile | **925.8 ms** | 127.74 s | 5,920.8 MB | Baseline before changes |
| **1. `ReduceMiddle` SIMD** | Vectorize `ReduceMiddle` via `ReduceLeading` SIMD / AVX-512 | **923.5 ms** | 125.31 s | 5,920.8 MB | Eliminated `execReduceSumGeneric` (`4.15 s` $\to$ `0 s`) |
| **2. `FusedDenseVJP` Parallel `transpose2D` & Pooling** | Parallelize `transpose2D` + pool `dZ`/`xT`/`dZT` via `GetBuffer` | **671.7 ms** | 128.43 s | 2,880.1 MB | **`-251.8 ms/step` (-27.3%)**, heap allocs cut in half (`-51.4%`) |
| **3. Out-of-place AVX-512/AVX2 Activations (`RegisterKernel`)** | Convert `SiluAVX512`/`ReluAVX512`/etc. to `(in, out)` kernels | **668.7 ms** (`671.0 ms` prof) | 129.40 s | 2,880.1 MB | Eliminated `1.64 s` `activations.Register` `runtime.memmove` |
| **4. Binary Broadcast `BroadcastOuterInner` & `BroadcastMid` `v2v1`** | Add `BroadcastOuterInnerRHS/LHS` + register-pinned `v2v1` for `C=64,96,128` | **577.7 ms** (`575.1 ms` prof) | 134.88 s | 2,880.1 MB | **`-93.6 ms/step` (-14.0%)**; eliminated `ZipIterator` fallback |
| **5. `FusedLayerNorm` Middle-Axes SIMD** | Vectorize `[aDim, bDim, cDim]` LayerNorm across contiguous `cDim` | **577.2 ms** (`578.3 ms` prof) | 136.46 s | 2,880.1 MB | Eliminated scalar `layerNormStridedAxes` (`7.51 s` $\to$ `4.60 s` SIMD) |
| **6. `ConvGeneral` Direct 2D & `VJPKernel` Slicing** | Direct zero-`im2col` 2D AVX-512/AVX2 kernels + `(BatchSize, OutH)` slice parallelization | **563.0 ms** (`564.2 ms` prof) | 133.51 s | 2,842.7 MB | **`-14.2 ms/step`**; `ExtractPatches2DFloat32` eliminated (`6.37 s` $\to$ `<0.67 s`), `ConvGeneral` at `2.0–2.9 TFlops/s` |




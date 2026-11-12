# Reworking `compute.FusedQuantizedDense` & `FusedQuantizedDenseVJP`

This document proposes a unified redesign of `compute.FusedOps.FusedQuantizedDense` (and a companion `FusedQuantizedDenseVJP`) so that GoMLX can support the major quantization paradigms used in modern ML inference and fine-tuning under a single, consistent backend interface aligned with `FusedDense`.

---

## 0. Dimension & Prefix-Axis Convention

Across all quantization schemes below:

- **$K$ (`inFeatures`)**: number of input features (the contracting dimension of the matrix multiplication).
- **$N$ (`outFeatures`)**: number of output features.
- **`batch...` / $M$**: the input `x` has shape `[batch..., K]`, where `batch...` represents zero or more leading dimensions (for example `[batchSize, seqLen, K]` in Transformers, or `[batchSize, K]` in MLPs).
  - Because the dense projection acts independently on each vector along the last axis, all leading dimensions can be merged into a single flattened row count $M = \prod \text{batch}$ (`x2d` of shape `[M, K]`), producing a 2D output `[M, N]` that is reshaped back to `[batch..., N]`.
  - For clarity, intermediate expressions below are written using the flattened 2D shape `[M, K]` $\to$ `[M, N]`.

---

## 1. Survey of Common Quantization Paradigms (Inputs, Shapes, Math & Outputs)

Quantized linear layers fall into two broad categories:
1. **Weight-Only Quantization** (activations `x` are floating-point; only `weight` is quantized, saving memory capacity and memory bandwidth).
2. **Weight + Activation Quantization** (both `x` and `weight` are quantized integers, enabling fast integer dot-product instructions such as ARM `SDOT`/`USDOT`/`SMOPA` and x86 `VNNI`).

---

### 1.1 GGUF / GGML Block Quantization (`llama.cpp` Ecosystem)

Used by GGUF models (`Q4_0`, `Q8_0`, `IQ4_NL`, and K-quants `Q2_K`..`Q6_K`). Scales (and per-block minimums where applicable) are **packed inline** inside fixed-size byte blocks alongside the quantized nibbles/quants, rather than stored in separate tensors.

- **Parameters**:
  - Weight `GGMLType`: e.g. `GGMLQ4_0` ($B_{\text{vals}} = 32$ values per block, $B_{\text{bytes}} = 18$ bytes per block) or `GGMLQ4_K` ($B_{\text{vals}} = 256$, $B_{\text{bytes}} = 144$).
  - Number of weight blocks per row: $G = K / B_{\text{vals}}$, so $\text{bytesPerRow} = G \cdot B_{\text{bytes}}$.
  - Paired activation `GGMLType` *(for integer dot products)*: `GGMLQ8_0` ($B_{\text{vals}} = 32$, $B_{\text{bytes,Q8}} = 34$, paired with `Q4_0`/`Q8_0`) or `GGMLQ8_K` ($B_{\text{vals}} = 256$, $B_{\text{bytes,Q8}} = 292$, paired with K-quants `Q2_K`..`Q6_K`), giving $\text{bytesPerRowQ8} = G \cdot B_{\text{bytes,Q8}}$.
- **Inputs**:
  - `x`:
    - Floating-point mode (`XQuantization == nil`): `(Float32)[M, K]` (after flattening `[batch..., K]`), **or**
    - Pre-quantized GGML mode (`XQuantization = {Scheme: QuantGGML, GGMLType: GGMLQ8_0 | GGMLQ8_K}`): `(Uint8)[M, bytesPerRowQ8]` — useful in `gomlx/core/graph` when a single quantized `x` is shared across separate Q, K, and V (or Gate and Up) projections.
  - `weight`: `(Uint8)[N, bytesPerRow]` — stored in `OutputsInput` order (`N` rows of `bytesPerRow` raw bytes).
  - `bias` *(optional)*: `(Float32)[N]`.

**Calculation & Intermediate Shapes:**

#### Mode A — Floating-Point Dot Product (`XQuantization == nil` and `QuantizedMatmuls == false`)

**Step 1A — Block dequantization** *(logically; fused on-the-fly without materializing `W_dequant` in RAM)*:
Each row $n \in [0, N)$ and block $g \in [0, G)$ of $B_{\text{bytes}}$ bytes is unpacked into $B_{\text{vals}}$ floats (for `Q4_0`: a `Float16` scale $d_{w,n,g}$ followed by 32 4-bit unsigned codes $q_{w,n,g,i} \in [0..15]$ shifted by $-8$):

$$
W_{\text{dequant}}[n, g \cdot B_{\text{vals}} + i] = \text{float32}(d_{w,n,g}) \cdot (q_{w,n,g,i} - 8) \quad \longrightarrow \quad W_{\text{dequant}} : (\text{Float32})[N, K]
$$

**Step 2A — Floating-point matrix multiplication** *(contracting over $K$)*:

$$
Z = x \cdot W_{\text{dequant}}^T \quad \longrightarrow \quad Z : (\text{Float32})[M, N]
$$

#### Mode B — `llama.cpp`-Style Block-Integer Dot Product (`XQuantization != nil` or `QuantizedMatmuls == true`)

Used by `llama.cpp` (`ggml_mul_mat`) to exploit integer SIMD instructions (`SDOT`/`USDOT`/`VNNI`). Note that because **each block $g \in [0, G)$ along $K$ has its own floating-point scale**, integer dot products happen *within each block of $B_{\text{vals}}$ elements* (`Int32`) and are scaled and accumulated across blocks into a **`Float32` output `[M, N]`**:

**Step 1B — Row quantization of `x` to `Q8_0` / `Q8_K`** *(performed internally into a temporary buffer when `XQuantization == nil && QuantizedMatmuls == true`, or passed in directly when `XQuantization != nil`)*:
For each row $m$ and block $g \in [0, G)$ of $B_{\text{vals}}$ elements (shown here for `Q8_0`, $B_{\text{vals}} = 32$):

$$
d_{x,m,g} = \frac{\max_{i} |x[m, 32g + i]|}{127}, \qquad q_{x,m,g,i} = \text{round}\!\left(\frac{x[m, 32g + i]}{d_{x,m,g}}\right) \in [-127, 127] \quad \longrightarrow \quad x_{\text{Q8}} : (\text{Uint8})[M, \text{bytesPerRowQ8}]
$$

**Step 2B — Per-block `Int32` dot product + `Float32` inter-block accumulation**:

$$
Z[m, n] = \sum_{g=0}^{G-1} \big(\text{float32}(d_{x,m,g}) \cdot \text{float32}(d_{w,n,g})\big) \cdot \underbrace{\left(\sum_{i=0}^{31} \text{int32}(q_{x,m,g,i}) \cdot \text{int32}(q_{w,n,g,i} - 8)\right)}_{\text{Int32 SIMD dot product per block}} \quad \longrightarrow \quad Z : (\text{Float32})[M, N]
$$

#### Common Epilogue (Both Modes A and B)

**Step 3 — Bias & Activation**:

$$
Y = \text{activation}(Z + \text{bias}) \quad \longrightarrow \quad Y : (\text{Float32})[M, N]
$$

- **Output**:
  - `y`: `(Float32)[batch..., N]` (or quantized if `OutputQuantization != nil`).

---

### 1.2 QLoRA (`NF4` NormalFloat4 + Low-Rank Adapter)

Used for memory-efficient fine-tuning of frozen 4-bit base models (Dettmers et al., 2023). Quantized 4-bit codes index a fixed 16-entry NormalFloat codebook $\text{NF4}[0..15] \in [-1.0, 1.0]$. Scales are stored per block (optionally "double-quantized" as `Uint8` codes + second-level `Float32` scales, which are dequantized before the matmul).

- **Inputs (Base Quantized Projection)**:
  - `x`: `(Float32)[M, K]` (or `BFloat16`/`Float16`).
  - `weight`: `(Uint4)[K, N]` (unpacked from packed `(Uint8)[K, N/2]` via `Bitcast`, or `[N, K]` if `OutputsInput`).
  - `wScale`: `(Float32)[K, NumBlocks]` when blocked along axis 1 ($N$, with $\text{NumBlocks} = \lceil N / \text{BlockSize} \rceil$), or `(Float32)[NumBlocks, N]` when blocked along axis 0 ($K$, with $\text{NumBlocks} = \lceil K / \text{BlockSize} \rceil$).
  - `bias` *(optional)*: `(Float32)[N]`.
- **Additional QLoRA Adapter Inputs** *(separate graph ops in `peft.NF4`)*:
  - `loraA` *(trainable)*: `(Float32)[K, R]` where rank $R \ll \min(K, N)$ (e.g. $R = 8, 16$).
  - `loraB` *(trainable)*: `(Float32)[R, N]`.
  - `loraScale`: scalar `Float32` ($\alpha / R$).

**Calculation & Intermediate Shapes:**

**Step 1 — Codebook lookup & scale multiplication** *(fused on-the-fly inside the matmul loop; `W_dequant` is **never** materialized in memory)*:
For `BlockAxis = 1` (`wScale` of shape `[K, NumBlocks]`):

$$
W_{\text{dequant}}[k, n] = \text{NF4}\big[\text{weight}[k, n]\big] \cdot \text{wScale}\big[k, \lfloor n / \text{BlockSize} \rfloor\big] \quad \longrightarrow \quad W_{\text{dequant}} : (\text{Float32})[K, N]
$$

**Step 2 — Base projection + Bias + Activation**:

$$
Y_{\text{base}} = \text{activation}(x \cdot W_{\text{dequant}} + \text{bias}) \quad \longrightarrow \quad Y_{\text{base}} : (\text{Float32})[M, N]
$$

**Step 3 — LoRA low-rank branch** *(when `Activation == ActivationNone` on base)*:

$$
H = x \cdot \text{loraA} \quad \longrightarrow \quad H : (\text{Float32})[M, R]
$$

$$
Y = Y_{\text{base}} + \text{loraScale} \cdot (H \cdot \text{loraB}) \quad \longrightarrow \quad Y : (\text{Float32})[M, N]
$$

- **Output**:
  - `y`: `(Float32)[batch..., N]`.

---

### 1.3 Group-Wise Weight-Only Linear Quantization (GPTQ, AWQ, `torchao` Int4/Int8)

Used widely in HuggingFace / PyTorch (`AutoGPTQ`, `AutoAWQ`, `bitsandbytes` Int8, `torchao`). Unlike `peft.NF4` in GoMLX today (which blocks along $N$, `BlockAxis = 1`), **GPTQ and AWQ group weights along the contracting input dimension $K$ (`BlockAxis = 0`)**, typically with `BlockSize` $G \in \{32, 64, 128\}$.

- **Inputs**:
  - `x`: `(Float32)[M, K]` (or `Float16` / `BFloat16`).
  - `weight`: `(Int4 | Uint4 | Int8 | Uint8)[K, N]` (with `DenseLayoutInputOutputs`, or `[N, K]` with `DenseLayoutOutputsInput`).
  - `wScale`: `(Float32)[NumBlocks, N]` where $\text{NumBlocks} = \lceil K / \text{BlockSize} \rceil$ (for per-channel quantization, `BlockSize = K`, so `NumBlocks = 1`, shape `[1, N]` or `[N]`).
  - `wZeroPoint` *(optional; symmetric if `nil`, asymmetric in AWQ/GPTQ)*: `(Int4 | Uint4 | Int8 | Uint8 | Float32)[NumBlocks, N]`.
  - `bias` *(optional)*: `(Float32)[N]`.

**Calculation & Intermediate Shapes:**

**Step 1 — Affine dequantization** *(fused on-the-fly, **standard zero-point subtraction before scaling**)*:
Let $b(k) = \lfloor k / \text{BlockSize} \rfloor$:

$$
W_{\text{dequant}}[k, n] = \big(\text{float32}(\text{weight}[k, n]) - \text{float32}(\text{wZeroPoint}[b(k), n])\big) \cdot \text{wScale}[b(k), n] \quad \longrightarrow \quad W_{\text{dequant}} : (\text{Float32})[K, N]
$$

**Step 2 — Matrix multiplication + Bias + Activation**:

$$
Y = \text{activation}(x \cdot W_{\text{dequant}} + \text{bias}) \quad \longrightarrow \quad Y : (\text{Float32})[M, N]
$$

- **Output**:
  - `y`: `(Float32)[batch..., N]`.

---

### 1.4 ONNX / SmoothQuant Integer Quantization (Quantized `x` $\times$ Quantized `weight`)

Used by ONNX (`DynamicQuantizeLinear` + `MatMulInteger`, or static `QLinearMatMul`) and SmoothQuant. Here **both** `x` and `weight` arrive at the dense operation already quantized to 8-bit integers (`Uint8` or `Int8`).

*(Note: In ONNX dynamic quantization, `DynamicQuantizeLinear(x_float)` precedes the matmul and produces `x_quant`, `xScale`, and `xZeroPoint`. Keeping `DynamicQuantizeLinear` as a separate graph op before `FusedQuantizedDense` allows a single quantized `x_quant` to be shared across Q, K, and V projections.)*

- **Inputs**:
  - `x`: `(Uint8 | Int8)[M, K]` (after flattening `[batch..., K]`).
  - `xScale`: `(Float32)[]` (scalar per-tensor) or `(Float32)[M, 1]` (per-token/per-row).
  - `xZeroPoint` *(optional; `0` if `nil`)*: `(Uint8 | Int8)[]` (scalar) or `(Uint8 | Int8)[M, 1]`.
  - `weight`: `(Int8 | Uint8)[K, N]` (or `[N, K]` for `OutputsInput`).
  - `wScale`: `(Float32)[]` (scalar per-tensor) or `(Float32)[1, N]` / `(Float32)[N]` (per-channel).
  - `wZeroPoint` *(optional; `0` if `nil`)*: `(Int8 | Uint8)[]` (scalar) or `(Int8 | Uint8)[1, N]` / `(Int8 | Uint8)[N]`.
  - `bias` *(optional)*: `(Float32)[N]` (or `(Int32)[N]` in pre-scaled integer pipelines).
  - `yScale`, `yZeroPoint` *(optional, only for `QLinearMatMul` where the output itself is re-quantized)*: scalar or `[1, N]`.

**Calculation & Intermediate Shapes:**

**Step 1 — Exact `Int32` integer matrix multiplication** *(critical for both ONNX numerical parity and SIMD `VNNI`/`SDOT` throughput)*:

$$
A_{\text{int32}}[m, n] = \sum_{k=0}^{K-1} \big(\text{int32}(x[m, k]) - \text{int32}(\text{xZeroPoint}[m])\big) \cdot \big(\text{int32}(\text{weight}[k, n]) - \text{int32}(\text{wZeroPoint}[n])\big) \quad \longrightarrow \quad A_{\text{int32}} : (\text{Int32})[M, N]
$$

**Step 2 & Step 3 — Combined scale dequantization, `Float` vs `Int32` Bias, & Activation**:

$$
S_{\text{combined}}[m, n] = \text{xScale}[m] \cdot \text{wScale}[n] \quad \longrightarrow \quad S_{\text{combined}} : (\text{Float32})[] \text{ or } [1, N] \text{ or } [M, N]
$$

Where `bias` is added and how it is scaled depends on its `DType`:
- **Floating-point `bias`** (`Float32` / `Float16` / `BFloat16`, shape `[N]`, or `nil`): already in real units, so it is added **after** rescaling `A_int32` by `S_combined` and **before** `activation` (and `OutputQuantization`):

$$
Y_{\text{float}}[m, n] = \text{activation}\!\Big(\text{float32}\big(A_{\text{int32}}[m, n]\big) \cdot S_{\text{combined}}[m, n] + \text{float32}\big(\text{bias}[n]\big)\Big)
$$

- **Pre-scaled `Int32` `bias`** (`Int32`, shape `[N]`): used in static integer pipelines (e.g. TFLite / static ONNX `QLinear` layers) where biases are quantized offline with scale $S_{\text{bias}}[n] = \text{xScale} \cdot \text{wScale}[n]$, i.e. $\text{bias}_{\text{int32}}[n] = \text{round}(\text{bias}_{\text{float}}[n] / (\text{xScale} \cdot \text{wScale}[n]))$. It is added **directly to the `Int32` accumulator $A_{\text{int32}}[m, n]$ before** converting to `Float32` and multiplying by $S_{\text{combined}}[m, n]$:

$$
Y_{\text{float}}[m, n] = \text{activation}\!\Big(\text{float32}\big(A_{\text{int32}}[m, n] + \text{bias}[n]\big) \cdot S_{\text{combined}}[m, n]\Big)
$$

  *Constraint*: `Int32` `bias` is only valid when `XQuantization != nil` with `Scheme == QuantLinear` and **neither `XQuantization.Scale` nor `WeightQuantization.Scale` partitions $K$ into multiple blocks** (so each output element $(m, n)$ has a single well-defined combined scale $\text{xScale}[m] \cdot \text{wScale}[n]$). When $K$ is sub-blocked — or when `xScale[m]` is dynamically computed per token at runtime (e.g. `DynamicQuantizeLinear`) — callers should use a floating-point `bias` (`Float32`).

**Step 4 — Optional Output Quantization** *(only if `OutputQuantization != nil`, e.g. `QLinearMatMul`)*:

$$
Y_{\text{quant}} = \text{cast}\!\left(\text{clamp}\!\left(\text{round}\!\left(\frac{Y_{\text{float}}}{\text{yScale}}\right) + \text{float32}(\text{yZeroPoint}),\; q_{\min},\; q_{\max}\right)\right) \quad \longrightarrow \quad Y_{\text{quant}} : (\text{Uint8 | Int8})[M, N]
$$

- **Output**:
  - `y`: `(Float32)[batch..., N]` (default), or `(Uint8 | Int8)[batch..., N]` when `OutputQuantization != nil`.

---

## 2. What `compute.FusedQuantizedDense` Currently Does

### 2.1 Current Signature (`compute/ops_fused.go`)
```go
FusedQuantizedDense(
    x, weights, bias Value,
    weightsQuantization *Quantization,
    activation ActivationType,
) (Value, error)
```
with:
```go
type Quantization struct {
    Scheme    QuantizationScheme // QuantLinear, QuantNF4, QuantGGML
    Scale     Value              // Must be (Float32)[K, NumBlocks]
    ZeroPoint Value              // Optional: (Float32)[K, NumBlocks]
    BlockAxis int                // Only BlockAxis == 1 is supported
    BlockSize int                // Group size along N; NumBlocks = ceil(N / BlockSize)
    GGMLType  GGMLQuantType
}
```

### 2.2 Current Behavior & Limitations
1. **Weight-only (`Float32` `x` only)**:
   - `gobackend` rejects any `x` whose dtype is not `Float32` (`x` cannot be `Float16`, `BFloat16`, `Int8`, `Uint8`, or GGML `Q8_0`/`Q8_K` blocks).
   - There is no `XQuantization`, `OutputQuantization`, or `QuantizedMatmuls` option, so ONNX `MatMulInteger` / `QLinearMatMul` (`Int32` accumulation) and `llama.cpp`-style `Q8_0`/`Q8_K` integer dot products cannot be represented or controlled.
2. **Rigid `Scale` shape (`[K, NumBlocks]` only) and `BlockAxis == 1` only**:
   - Even for per-tensor (`[]`) or per-channel (`[N]`) scales, callers must broadcast `Scale` across $K$ into `[K, NumBlocks]` (`[K, 1]` or `[K, N]`). For per-channel scales `[N]`, `[K, N]` allocates a `Float32` tensor the exact same size as an unquantized `Float32` weight matrix!
   - `BlockAxis == 0` (blocking along $K$ with `Scale` of shape `[NumBlocks, N]`), which is the standard layout of **GPTQ, AWQ, and `torchao`**, is rejected (`only Axis=1 is supported`).
3. **Inconsistent `ZeroPoint` semantics across backends**:
   - `compute/ops_fused.go`, `gobackend`, and `gomlx/ml/nn/quantized_dense.go` treat `ZeroPoint` as a **post-scale `Float32` additive offset**: `float32(w) * scale + zeroPoint`.
   - `compute-onnx/internal/graph/ops_fused.go` treats `ZeroPoint` as a **pre-scale integer zero-point**: `(float32(w) - float32(zeroPoint)) * scale`.
4. **No `WeightLayout` parameter**:
   - `QuantLinear` and `QuantNF4` hardcode `[K, N]` (`DenseLayoutInputOutputs`), whereas `QuantGGML` hardcodes `[N, bytesPerRow]` (`DenseLayoutOutputsInput`), and PyTorch/HuggingFace/Safetensors checkpoints frequently store quantized weights as `[N, K]`.
5. **Signature diverges from `FusedDense`**:
   - `FusedDense` takes `(x, weight, bias Value, options DenseConfig)` (with `ActivationConfig` and `DenseLayout`), whereas `FusedQuantizedDense` takes plural `weights` and bare `weightsQuantization *Quantization` + `activation ActivationType` arguments, and has no VJP counterpart (`FusedQuantizedDenseVJP`).

---

## 3. Proposed Reworked API: `compute.FusedQuantizedDense`

### 3.1 Signature (Aligned with `FusedDense`)

```go
// FusedQuantizedDense performs fused quantized matrix multiplication + optional bias +
// optional activation (+ optional output re-quantization), without materializing the full
// dequantized weight tensor in memory:
//
//   - Weight-only quantization (options.XQuantization == nil):
//     z = x @ dequant(weight, options.WeightQuantization) + bias
//     (transposing dequant(weight) when options.WeightLayout == DenseLayoutOutputsInput; if
//     options.QuantizedMatmuls is true, backends may dynamically quantize x internally to run
//     integer SIMD dot products).
//   - Pre-quantized x + weight integer matmul (options.XQuantization != nil):
//     supports QuantLinear (exact Int32 accumulation of (x - xZP) @ (w - wZP) rescaled by
//     xScale * wScale) and QuantGGML (GGMLQ8_0 / GGMLQ8_K paired per-block Int32 dot products
//     scaled by d_x * d_w into Float32).
//   - Bias conventions (when bias != nil, shape [N]):
//     Floating-point bias (matching x's float DType when XQuantization == nil, or
//     Float32/Float16/BFloat16 when XQuantization != nil) is already in real units and is added
//     in floating-point AFTER dequantization / (xScale * wScale) rescaling and BEFORE
//     options.Activation / options.OutputQuantization.
//     Int32 bias (valid only for QuantLinear XQuantization when neither XQuantization.Scale nor
//     WeightQuantization.Scale is sub-blocked along K) is pre-quantized at scale xScale * wScale[n]
//     (i.e. bias_int32[n] = round(bias_float[n] / (xScale * wScale[n]))) and is added directly to
//     the Int32 accumulator BEFORE converting to Float32 and multiplying by xScale * wScale:
//     z[m, n] = float32(acc_int32[m, n] + bias[n]) * (xScale[m] * wScale[n]).
//   - Optional output quantization (options.OutputQuantization != nil):
//     re-quantizes the post-activation float output to Int8 or Uint8.
//
// Inputs:
//   - x: [batch..., K] floating-point (when XQuantization == nil) or quantized integer/block
//     tensor ([batch..., K] Int8/Uint8 for QuantLinear, [batch..., bytesPerRowX] Uint8 for QuantGGML).
//   - weight: [K, N] (DenseLayoutInputOutputs) or [N, K] (DenseLayoutOutputsInput) with DType
//     Int2/Uint2/Int4/Uint4/Int8/Uint8 for QuantLinear/QuantNF4; or [N, bytesPerRow] Uint8 for
//     QuantGGML (requires DenseLayoutOutputsInput, bytesPerRow = (K / valuesPerBlock) * bytesPerBlock).
//   - bias: optional [N] tensor (nil for no bias); floating-point (added after dequantization/rescaling)
//     or Int32 (added to the Int32 accumulator before xScale * wScale rescaling when XQuantization uses
//     QuantLinear without K sub-blocking), applied before options.Activation and options.OutputQuantization.
//   - options: FusedQuantizedDenseConfig (WeightQuantization must not be nil).
//
// Note: Like FusedDense, ActivationSwiGLU is not supported here because it halves the output dimension;
// emit FusedQuantizedDense with ActivationNone followed by FusedActivation(ActivationSwiGLU).
//
// Migration note: quantization, layout, and activation options are grouped in FusedQuantizedDenseConfig,
// QuantGGML weights require WeightLayout = DenseLayoutOutputsInput, scalar [] and 1D [N] scales no longer
// need [K, NumBlocks] broadcasts, and QuantLinear ZeroPoint uses the standard pre-scale convention
// real = (float(q) - float(zp)) * scale (legacy additive offsets convert via zero_point = -offset / scale).
FusedQuantizedDense(x, weight, bias Value, options FusedQuantizedDenseConfig) (Value, error)
```

### 3.2 `FusedQuantizedDenseConfig` and Reworked `Quantization`

```go
// FusedQuantizedDenseConfig holds configuration and quantization metadata for FusedQuantizedDense.
type FusedQuantizedDenseConfig struct {
	// WeightQuantization describes how the weight tensor is quantized. Required (must not be nil).
	WeightQuantization *Quantization

	// XQuantization describes how the input x is quantized for integer-matmul schemes:
	//   - QuantLinear (Int8/Uint8 x + Scale/ZeroPoint, e.g. ONNX MatMulInteger / QLinearMatMul / SmoothQuant), or
	//   - QuantGGML (GGMLQ8_0 / GGMLQ8_K packed Uint8 x, e.g. shared Q8 activations across Q/K/V projections).
	// Nil (the default) means x is passed in floating-point.
	XQuantization *Quantization

	// OutputQuantization optionally re-quantizes the post-activation output tensor using caller-provided
	// Scale and ZeroPoint (e.g. ONNX QLinearMatMul).
	// Nil (the default) returns a floating-point output matching x's float dtype (or Float32 when x is quantized).
	OutputQuantization *Quantization

	// QuantizedMatmuls is an optional optimization hint used when XQuantization == nil (floating-point x).
	// If true, the backend may dynamically quantize x on-the-fly into a temporary integer/block buffer
	// (e.g. Q8_0/Q8_K for GGML weights, or per-row Int8/Uint8 for QuantLinear weights) to execute integer
	// SIMD dot products (ARM SDOT/USDOT, x86 VNNI), trading minor quantization rounding for throughput.
	// Backends that do not support on-the-fly quantized matmuls silently ignore this flag and use float arithmetic.
	QuantizedMatmuls bool

	// Activation is applied after matmul/rescaling and bias (and before OutputQuantization if set);
	// defaults to ActivationNone.
	Activation ActivationConfig

	// WeightLayout specifies whether weight is [K, N] (DenseLayoutInputOutputs, default)
	// or [N, K] (DenseLayoutOutputsInput; required for QuantGGML where shape is [N, bytesPerRow]).
	WeightLayout DenseLayout
}
```

And refine `Quantization` so it cleanly represents per-tensor, per-channel, and arbitrary-axis block quantization without forcing `[K, N]` scale broadcasts, while fixing the `ZeroPoint` definition:

```go
// Quantization describes how a quantized tensor maps between quantized codes and real values.
//
// For QuantLinear, the dequantization formula is standard zero-point affine quantization:
//
//	real_value = (float(quant_value) - float(zero_point)) * scale
//
// (When ZeroPoint is nil, zero_point = 0: real_value = float(quant_value) * scale.)
type Quantization struct {
	// Scheme: QuantLinear, QuantNF4, or QuantGGML.
	Scheme QuantizationScheme

	// Scale is the multiplicative scale factor (Float32 / Float16 / BFloat16).
	// Accepted shapes for a 2D tensor [D0, D1] (or flattened [M, K] input):
	//   - Scalar [] or [1]: per-tensor scale.
	//   - 1D [D_other]: per-channel scale along the non-contracting axis (e.g. [N] for weights, [M] for x).
	//   - 2D [ceil(D0/BlockSize), D1] when BlockAxis == 0 (e.g. GPTQ/AWQ [K/G, N]), or
	//     2D [D0, ceil(D1/BlockSize)] when BlockAxis == 1 (e.g. [K, N/G] or [M, 1]).
	// Unused (nil) for QuantGGML.
	Scale Value

	// ZeroPoint is the optional integer zero-point subtracted BEFORE multiplying by Scale
	// (only for QuantLinear). Must be broadcast-compatible with Scale's shape.
	// May have integer dtype (Int4/Uint4/Int8/Uint8/Int32) or float dtype.
	// Nil means symmetric quantization (zero_point = 0).
	ZeroPoint Value

	// BlockAxis is the axis of the quantized tensor partitioned into blocks of size BlockSize
	// (0 or 1 for a 2D weight matrix; -1 or K-axis for input x).
	// Ignored when Scale is scalar or 1D per-channel, or when Scheme == QuantGGML.
	BlockAxis int

	// BlockSize is the number of elements along BlockAxis sharing one scale/zero-point.
	// If 0, inferred from ceil(dim(BlockAxis) / Scale.Shape().Dimensions[BlockAxis]).
	BlockSize int

	// GGMLType specifies the concrete GGML block format when Scheme == QuantGGML.
	GGMLType GGMLQuantType
}
```

### 3.3 Graph-Layer Counterpart (`gomlx/core/graph` & `gomlx/ml/nn`)
- Just like `graph.Quantization` today wraps `*Node` instead of `compute.Value` so that GoMLX tracks `Scale` and `ZeroPoint` in `node.inputNodes` (for topological ordering, CSE/DCE, `CloneWithInputs`, and gradient checkpointing), `graph.QuantizedDenseConfig` mirrors `compute.FusedQuantizedDenseConfig` (`WeightQuantization`, `XQuantization`, `OutputQuantization`, `QuantizedMatmuls`, `Activation`, `WeightLayout`) and converts to `compute.FusedQuantizedDenseConfig` in `BackendFusedQuantizedDense`.
- `nn.QuantizedDense(x, weight, bias *Node, cfg QuantizedDenseConfig, optionalActivation ...activation.Type)` provides automatic `InternalFusedOpCaller` fallback to `quantizedDenseDecomposed` (including exact `Int32` `DotGeneral` decomposition when `XQuantization` uses `QuantLinear`, and GGML block dequantization/quantization helpers in `gomlx/ml/ggml` when `XQuantization` uses `QuantGGML`!).
- Provide a graph-level row-quantizer helper (`ggml.Quantize(x, GGMLQ8_0)` / `ggml.Quantize(x, GGMLQ8_K)`) so GGUF Transformer layers can quantize `x` once and pass it with `XQuantization: &Quantization{Scheme: QuantGGML, GGMLType: ...}` into multiple `nn.QuantizedDense` projections (Q, K, V or Gate, Up).

### 3.4 Migration Guide for Existing Callers

1. **Updated Call Signatures & `WeightLayout`**:
   - **`compute`**:
     ```go
     // Before:
     y, err := f.FusedQuantizedDense(x, weights, bias, weightsQuant, compute.ActivationGelu)
     // After:
     y, err := f.FusedQuantizedDense(x, weight, bias, compute.FusedQuantizedDenseConfig{
         WeightQuantization: weightsQuant,
         Activation:         compute.ActivationConfig{Type: compute.ActivationGelu},
         WeightLayout:       compute.DenseLayoutInputOutputs, // or DenseLayoutOutputsInput (required for QuantGGML)
     })
     ```
   - **`gomlx/core/graph` & `gomlx/ml/nn`** (`bias` is now the 3rd argument before `cfg`, matching `nn.Dense(x, weight, bias, ...)`):
     ```go
     // Before:
     y := nn.QuantizedDense(x, weights, quant, bias, activation.TypeGelu)
     // After:
     y := nn.QuantizedDense(x, weight, bias, graph.QuantizedDenseConfig{
         WeightQuantization: quant,
         WeightLayout:       compute.DenseLayoutInputOutputs, // set DenseLayoutOutputsInput for QuantGGML [N, bytesPerRow]
     }, activation.TypeGelu)
     ```

2. **Migrating to the Standardized `(q - zp) * scale` Zero-Point Convention**:
   - **Symmetric quantization (`ZeroPoint == nil`)**: No change (`real = float(q) * scale`).
   - **Checkpoints with standard pre-scale integer/float zero-points (ONNX `DequantizeLinear` / `MatMulInteger` / `MatMulNBits`, PyTorch, TFLite, GPTQ / AWQ `qzeros`)**: Pass the zero-point tensor **directly** as `ZeroPoint`. Callers that previously worked around the old formula by computing `offset = -float(zp) * scale` should delete that workaround and pass `zp` directly.
   - **Legacy callers storing a post-scale float additive offset `offset`** (where previously `real = float(q) * scale + offset`, e.g., min-max quantizers storing `min_val` in `ZeroPoint`):
     Using the identity $\text{float}(q) \cdot \text{scale} + \text{offset} = \big(\text{float}(q) - (-\text{offset} / \text{scale})\big) \cdot \text{scale}$, convert the legacy additive offset into a pre-scale `Float32` zero-point:
     ```go
     // Host / offline conversion:
     zpNew[i] = -oldAdditiveOffset[i] / scale[i]
     // Or in GoMLX graph ops:
     ZeroPoint: graph.Neg(graph.Div(oldOffsetNode, scaleNode))
     ```
     In weight-only float mode (`XQuantization == nil`), floating-point `ZeroPoint` values are subtracted directly in `Float32` without integer rounding, so `zpNew = -offset / scale` is algebraically equivalent even when `-offset / scale` is non-integer. (In integer-matmul mode `XQuantization != nil`, zero-points represent integer grid offsets and float zero-points are rounded to nearest-even `Int32`.)

3. **Dropping `[K, NumBlocks]` Scale Broadcasts**:
   - Scalar `[]` / `[1]` (per-tensor) and 1D `[N]` (per-output-channel) `Scale` and `ZeroPoint` tensors can now be passed **directly** without broadcasting across `K` to `[K, 1]` or `[K, N]`.
   - K-grouped weights (`[K, N]` weights with `[ceil(K/G), N]` scales, as in GPTQ/AWQ) can now be passed directly with `BlockAxis: 0, BlockSize: G`.

---

## 4. Proposed `compute.FusedQuantizedDenseVJP` (for QLoRA / PEFT Fine-Tuning)

### 4.1 Motivation: Why QLoRA Needs `FusedQuantizedDenseVJP`
In QLoRA (`gomlx/ml/layers/peft/qlora.go`), a frozen quantized linear layer is paired with a trainable low-rank adapter:

$$
Y = \underbrace{\text{FusedQuantizedDense}(x, W_{\text{quant}}, \text{bias}, \text{options})}_{Y_{\text{base}} : (\text{Float32})[M, N]} + \frac{\alpha}{R} (x \cdot A) \cdot B
$$

During fine-tuning across a multi-layer network (e.g., 32 Transformer blocks):
- `W_quant` (and its `Scale` / `ZeroPoint`) are discrete and frozen — they **do not** have gradients (`nil`).
- The LoRA matrices $A_\ell, B_\ell$ in layer $\ell$ get their direct gradients from their own $(x_\ell \cdot A_\ell) \cdot B_\ell$ branch.
- **However**, for gradients to reach the LoRA adapters $A_{\ell-1}, B_{\ell-1}$ in **earlier** layers, the adjoint gradient $\text{dOutput} = \partial \mathcal{L} / \partial Y_{\text{base}}$ (`[M, N]`) **must backpropagate through `FusedQuantizedDense` to produce `dx` (`[M, K]`)** (plus `dBias` `[N]` if `bias` is trainable)!
- **Current bottleneck**: Without `FusedQuantizedDenseVJP`, `nn.QuantizedDense` uses `useDecomposedAlternateVJP = true`, which falls back during autodiff to `quantizedDenseDecomposed` — **unpacking and materializing the entire `[K, N]` `Float32` base weight matrix in memory** on the backward pass, defeating QLoRA's memory savings!

### 4.2 Scope of `FusedQuantizedDenseVJP`
- **Supported**: Weight-only quantization (`options.XQuantization == nil` and `options.OutputQuantization == nil`), where `x` is floating-point (`Float32` / `Float16` / `BFloat16`). (If `options.QuantizedMatmuls` is `true`, the backend may likewise quantize `dZ` on-the-fly if supported, or compute `dx` in float.)
- **Not applicable**: Pre-quantized inputs or quantized outputs (`XQuantization != nil` or `OutputQuantization != nil`), which are discrete step functions (returns `ErrNotImplemented`).
- **Activation handling**: Identical to `FusedDenseVJP`!
  - Works directly when `!options.Activation.Type.VJPRequiresInput()` (`ActivationNone`, `ActivationRelu`, `ActivationSigmoid`, `ActivationTanh`, etc.), computing pre-activation gradient `dZ` from `y` and `dOutput`.
  - When `options.Activation.Type.VJPRequiresInput()` is `true` (`ActivationGelu`, `ActivationSilu`, etc.) and `FusedQuantizedDenseVJP` is available, `nn.QuantizedDense` splits the forward call into `FusedQuantizedDense(..., ActivationNone)` followed by `FusedActivation` (just like `nn.Dense` in `gomlx/ml/nn/dense.go` lines 57–64).

### 4.3 Backward Math & Shapes
Given flattened `dOutput`: `(Float32)[M, N]` and forward output `y`: `(Float32)[M, N]`:

**Step 1 — Activation VJP** *(elementwise, fused)*:

$$
\text{dZ} = \text{dOutput} \odot \sigma'(y) \quad \longrightarrow \quad \text{dZ} : (\text{Float32})[M, N]
$$

**Step 2 — Input gradient `dx`** *(dequantizing `weight` on-the-fly without materializing `W_dequant` `[K, N]` in memory)*:

$$
\text{dx} = \text{dZ} \cdot \text{dequant}(\text{weight}, \text{WeightQuantization})^T \quad \longrightarrow \quad \text{dx} : (\text{Float32})[M, K] \;\xrightarrow{\text{reshape}}\; [\text{batch}..., K]
$$

*(Note the nice symmetry for GGUF/`OutputsInput` weights: in the forward pass we contracted `K`, whereas in the backward pass we contract `N`, streaming through the quantized blocks once!)*

**Step 3 — Optional bias gradient `dBias`** *(only when `bias != nil`)*:

$$
\text{dBias}[n] = \sum_{m=0}^{M-1} \text{dZ}[m, n] \quad \longrightarrow \quad \text{dBias} : (\text{Float32})[N]
$$

**Step 4 — Weight & Quantization parameter gradients**:
`nil` (non-differentiable / frozen).

### 4.4 Proposed `compute.FusedOps` Signature

```go
// FusedQuantizedDenseVJP computes the vector-jacobian product (gradients) of FusedQuantizedDense
// with respect to its floating-point inputs x and (optionally) bias, dequantizing weight on-the-fly
// without materializing the full float weight matrix.
//
// Quantized weights and quantization scales/zero-points are discrete/frozen and receive no gradients.
// Only supported for weight-only quantization (options.XQuantization == nil and
// options.OutputQuantization == nil) and activations where options.Activation.Type.VJPRequiresInput()
// is false.
//
// Parameters:
//   - x: original input tensor to FusedQuantizedDense, shape [batch..., K] (may be nil if not needed by backend,
//     since dx = dZ @ dequant(weight)^T and dBias = sum(dZ) do not depend on x when VJPRequiresInput() is false!).
//   - weight: quantized weight tensor passed to FusedQuantizedDense.
//   - bias: bias tensor passed to FusedQuantizedDense (nil if forward had no bias).
//   - y: output of the forward FusedQuantizedDense call, shape [batch..., N] (nil-able when Activation == ActivationNone).
//   - dOutput: incoming adjoint gradient with respect to y, shape [batch..., N].
//   - options: FusedQuantizedDenseConfig passed to the forward FusedQuantizedDense call.
//
// Outputs:
//   - dx: gradient with respect to x, shape [batch..., K].
//   - dBias: gradient with respect to bias, shape [N] (or nil if bias was nil).
FusedQuantizedDenseVJP(
    x, weight, bias, y, dOutput Value,
    options FusedQuantizedDenseConfig,
) (dx, dBias Value, err error)
```

> **Bonus memory observation on `x` and `y`**: Notice that when `Activation == ActivationNone` (or `!VJPRequiresInput()`), neither `dx = dZ @ dequant(W)^T` nor `dBias = sum(dZ)` even needs `x`! And when `Activation == ActivationNone`, `y` isn't needed either (`dZ = dOutput`)! That means during QLoRA fine-tuning of linear projections (`ActivationNone`), `FusedQuantizedDenseVJP` doesn't have to keep `x` or `y` alive from the forward pass at all (only the small `[M, R]` LoRA branch needs `x` for `dA` and `H` for `dB`).

### 4.5 Wiring into GoMLX Reverse-Mode Autodiff (`gomlx/core/graph`)
In `gomlx/core/graph/rev_autodiff.go`, `accumulateInputVJPs` already skips `nil` entries in the `[]*Node` slice returned by a registered `VJP` function (`if vjp == nil { continue }`).
For `node.inputNodes = [x, weight, wScale?, wZeroPoint?, bias?]`, `fusedQuantizedDenseVJP` simply returns:
```go
// Aligns 1-to-1 with node.inputNodes:
//   - dx for x
//   - nil for weight, wScale, wZeroPoint (frozen quantized tensors)
//   - dBias for bias (if bias != nil)
vjps := make([]*Node, len(node.inputNodes))
vjps[0] = ConvertDType(dx, inputs.x.DType())
if inputs.bias != nil {
    vjps[len(vjps)-1] = ConvertDType(dBias, inputs.bias.DType())
}
return vjps
```
When the LoRA branch `+ loraScale * (x @ A) @ B` is present in the graph alongside `FusedQuantizedDense`, GoMLX's reverse-mode autodiff automatically sums `dx` from `FusedQuantizedDenseVJP` and `dx` from the LoRA branch (`rInput.AccumulatedVJP = Add(rInput.AccumulatedVJP, vjp)`), while computing `dA` and `dB` through standard `DotGeneral` VJPs!

---

## 5. Implementation Roadmap

1. **`compute`**:
   - Add `FusedQuantizedDenseConfig` (`WeightQuantization`, `XQuantization`, `OutputQuantization`, `QuantizedMatmuls`, `Activation`, `WeightLayout`), update `Quantization` docstrings/validation (standard `(q - zp) * scale` zero-point semantics, flexible `Scale`/`ZeroPoint` shapes, `BlockAxis` `0` and `1`), update `FusedQuantizedDense` signature, and add `OpTypeFusedQuantizedDenseVJP` + `FusedQuantizedDenseVJP`.
   - Implement `gobackend` executors for:
     - Weight-only `QuantLinear`, `QuantNF4`, `QuantGGML` (both `BlockAxis=0` and `BlockAxis=1`, `InputOutputs` and `OutputsInput`).
     - Integer matmul via `XQuantization != nil` (both `QuantLinear` `Int8`/`Uint8` $\to$ exact `Int32` accumulation and `QuantGGML` `Q8_0`/`Q8_K` per-block `Int32` dot products $\to$ `Float32` accumulation + bias + activation + optional `OutputQuantization`) as well as on-the-fly `QuantizedMatmuls == true` for `Float32` `x`.
     - `FusedQuantizedDenseVJP` (streaming `dZ @ dequant(weight)^T` and `sum(dZ)` without materializing `[K, N]` float weights).
   - Add backend compliance tests in `compute/support/backendtest/` covering all schemes, block axes, layouts, `XQuantization` (`QuantLinear` and `QuantGGML`), `QuantizedMatmuls`, `OutputQuantization`, and VJPs.
2. **`compute-onnx`, `go-xla`, `go-darwinml`**:
   - Update signatures and generators; implement or return `compute.ErrNotImplemented` for unsupported combinations so GoMLX's decomposed fallback handles them seamlessly.
3. **`gomlx`**:
   - Update `BackendFusedQuantizedDense`, add `BackendFusedQuantizedDenseVJP` and register `NodeTypeFusedQuantizedDense` in `VJPRegistration`.
   - Update `nn.QuantizedDense` (and `peft.NF4`) to use `QuantizedDenseConfig`, add `ggml` row-quantize support for pre-quantized `GGMLQ8_0`/`GGMLQ8_K` `x`, and leverage `FusedQuantizedDenseVJP` when supported by the backend instead of materializing `[K, N]` weights via `vjpAlternateOutputs`.
4. **`onnx-gomlx`**:
   - Update `internal/onnxgomlx/fusion/quantized_dense.go` and `quantized_qkv.go` to keep `DynamicQuantizeLinear(x)` in the graph and emit `nn.QuantizedDense` with `XQuantization` (`xScale`, `xZeroPoint`) and `WeightQuantization` (`wScale`, `wZeroPoint`), then re-enable the fusion detectors.

# Optimizing `KnightsAnalytics/all-MiniLM-L6-v2` (ONNX) in `onnx-gomlx` and `compute`

## 1. Task & Reproduction

### Goal
Optimize the execution of the ONNX benchmark model `KnightsAnalytics/all-MiniLM-L6-v2` (converted via `onnx-gomlx` and executed on the Go backend in `compute`) so that it matches or exceeds the performance of the native Hugging Face safetensors equivalent (`sentence-transformers/all-MiniLM-L6-v2` via `go-huggingface`).

### Benchmark Commands
Run from `/home/janpf/Projects/gomlx/go-huggingface`:

```bash
# 1. ONNX Model (KnightsAnalytics/all-MiniLM-L6-v2)
GOEXPERIMENT=simd GOMLX_BACKEND=go go run ./examples/msmarco/benchmark_embed/ \
  -bucket=8192 \
  -model KnightsAnalytics/all-MiniLM-L6-v2 \
  -limit 100 \
  -num_embedders=1 \
  -warmup \
  -dynamic=true \
  -cpuprof=/tmp/minilm_onnx.prof

# 2. Native Hugging Face Safetensors Model (sentence-transformers/all-MiniLM-L6-v2)
GOEXPERIMENT=simd GOMLX_BACKEND=go go run ./examples/msmarco/benchmark_embed/ \
  -bucket=8192 \
  -model sentence-transformers/all-MiniLM-L6-v2 \
  -limit 100 \
  -num_embedders=1 \
  -warmup \
  -dynamic=true \
  -cpuprof=/tmp/minilm_hf.prof
```

To inspect the CPU profiles:
```bash
go tool pprof -top -nodecount=50 /tmp/minilm_onnx.prof
go tool pprof -top -cum -nodecount=50 /tmp/minilm_onnx.prof
```

---

## 2. Baseline Benchmark & CPU Profile Comparison

Measured on AMD64 (AVX-512 enabled, 24 hardware threads, `-bucket=8192 -limit 100 -num_embedders=1 -warmup -dynamic=true`):

| Metric / Hot Function | ONNX (`KnightsAnalytics/all-MiniLM-L6-v2`) | Native HF (`sentence-transformers/all-MiniLM-L6-v2`) | Delta (ONNX vs HF) |
| :--- | ---: | ---: | ---: |
| **Warmup Wall Time** | 1.4s (700 items/s, 51.5K tok/s, 29.6s CPU) | 1.1s (891 items/s, 65.5K tok/s, 25.3s CPU) | +27% slower |
| **Benchmark Wall Time** | **2.4s** (411 items/s, 57.1K tok/s, 53.2s CPU) | **1.9s** (520 items/s, 72.2K tok/s, 42.8s CPU) | **+26% slower** |
| **Total Sampled CPU Time** (warmup + bench) | **82.88s** | **68.06s** | **+14.82s (+21.8%)** |
| `avx512LargeKernelFloat32Asm` (GEMM kernel) | 38.25s (46.15%) | 28.60s (42.02%) | **+9.65s** |
| `sdpaFloat32AVX512Dim32` (`FusedSDPA`) | **0.00s (0%)** | **9.09s (13.36%)** | **-9.09s (Not used in ONNX!)** |
| `transpose4D_0213Block` (`runtime.memmove`) | **8.81s (10.63%)** | **0.00s (0%)** | **+8.81s** |
| `qkvSplitBiasFloat32` (`copyAndAddBiasFloat32AVX512Asm`) | **6.27s (7.57%)** | **0.00s (0%)** | **+6.27s** |
| `avx512PackRHSFullStripsAsm` (NonTransposed RHS pack) | 4.90s (5.91%) | 0.00s (0%) | **+4.90s** |
| `unsafePackLHS` (Transposed RHS weight pack in HF) | 0.00s (0%) | 16.15s (23.73%) | **-16.15s** |
| `simdMulVSFloat32` (unfused attention score scaling) | 2.16s (2.61%) | 0.00s (0%) | **+2.16s** |
| `fusedSoftmaxRowsFloat32` (`expFloat32AVX512`) | 2.15s (2.59%) | 0.00s (0%) | **+2.15s** |
| `simdV2V1AddFloat32` (unfused mask add + dense bias add) | 2.07s (2.50%) | 1.15s (1.69%) | **+0.92s** |
| `FusedGelu` (`erfFloat32AVX512` vs `geluApprox`) | 3.74s (4.51%) | 1.98s (2.91%) | **+1.76s** |
| `layerNormFloat32AVX512` (`FusedLayerNorm`) | 1.38s (1.67%) | 1.24s (1.82%) | +0.14s |

---

## 3. Detailed Findings & Recommended Optimizations

### Opportunity 1 (Critical — ~18.9s CPU in ONNX): `FusedScaledDotProductAttention` is bypassed when `-dynamic=true`

#### Symptom
In `minilm_onnx.prof`, `FusedScaledDotProductAttention` is completely absent (`0.00s`). Instead, all 6 attention layers execute the decomposed fallback path from `attention.Core`:
1. Two batched `DotGeneral` calls per layer (`Q @ K^T` and `Attn @ V`), adding **+9.65s** in `avx512LargeKernelFloat32Asm` and **+4.90s** in `avx512PackRHSFullStripsAsm` (packing dynamic `K` and `V` tensors at runtime).
2. Materializing the full `[batch, 12, seq_len, seq_len]` attention matrix in DRAM across three memory-bound passes:
   - `simdMulVSFloat32` (**2.16s**): scaling `Q @ K^T` by `0.1767767` ($1/\sqrt{32}$).
   - `simdV2V1AddFloat32` (**2.07s**): adding the attention mask.
   - `fusedSoftmaxRowsFloat32` (**2.15s**): computing softmax over the materialized `[batch, 12, seq_len, seq_len]` matrix.

#### Root Cause (Two Interacting Bugs)

##### 1A. `compute` rejects `FusedScaledDotProductAttention` when mask has `mSeqDim == 1` (`[batch, 1, 1, kv_len]`)
In `compute/internal/gobackend/fusedops/sdpa.go`:
- `buildSDPANode` calls `isSDPASeqDimCompatible(seqDim, mSeqDim)` (lines 242–249):
  ```go
  func isSDPASeqDimCompatible(seqDim, mSeqDim int) bool {
      if seqDim == shapes.DynamicDim || mSeqDim == shapes.DynamicDim {
          return seqDim != 1 && mSeqDim != 1 || seqDim == mSeqDim
      }
      return mSeqDim == seqDim
  }
  ```
- When `mask` has shape `[batch, 1, 1, kv_len]` (a key-padding mask broadcast across all query positions, which is standard in BERT/MiniLM encoders), `mSeqDim == 1` while `seqDim == -1` (`shapes.DynamicDim`) or `seqDim > 1`.
- `isSDPASeqDimCompatible` returns `false`, causing `FusedScaledDotProductAttention` to return `compute.ErrNotImplemented`, and `InternalFusedOpCaller` in `attention.Core` silently falls back to the decomposed graph!

**Fix in `compute/internal/gobackend/fusedops/sdpa.go`**:
- Allow `mSeqDim == 1` in `isSDPASeqDimCompatible` (i.e., `mSeqDim == 1 || mSeqDim == seqDim`, or when either is `shapes.DynamicDim`, always compatible at build time and verified in `Recompute` that `mSeqDim == 1 || mSeqDim == seqLen`).
- In `execFusedSDPA` (`sdpa.go`), compute `mSeqStride`:
  ```go
  mSeqStride := kvLen
  if mask != nil && mask.Shape.Dimensions[2] == 1 {
      mSeqStride = 0
  }
  ```
- Replace `mRowBase := mBatchHeadBase + qIdx*kvLen` with `mRowBase := mBatchHeadBase + qIdx*mSeqStride` across the float32/float64/half loops in `sdpa.go` (lines ~424, 523, 571, 650).
- **Nuance**: The AVX-512 and AVX2 assembly kernels (`sdpaFloat32AVX512Dim32`, `sdpaCausalMaskFloat32AVX512`, etc. in `sdpa_simd.go`) operate on a single query `qIdx` at a time and receive `maskSlice := maskFlat[mRowBase : mRowBase+kvLen]`. Therefore, **no assembly changes are required**—passing `mRowBase` with `mSeqStride = 0` automatically reuses the `[kvLen]` mask row for every query position `qIdx` (which is also much more L1-cache friendly than reading an expanded `[seq_len, kv_len]` mask!).
- Also check `sdpa_vjp.go` (if mask is used in VJP) to ensure `mSeqDim == 1` is handled consistently.

##### 1B. `onnx-gomlx` folds `Equal(dynamic_shape, -1)` to `true` during `Expand` conversion
Why did the mask have shape `[batch=?, 1, 1, seq_len=?]` in `-dynamic=true` mode, whereas in static `[16, 128]` mode it expanded to `[16, 1, 128, 128]`?
- In `KnightsAnalytics/all-MiniLM-L6-v2` (`model.onnx`), PyTorch exported `mask.expand(batch, 1, seq_len, seq_len)` as:
  1. `/Shape` on `input_ids` -> `/Gather` (axis 0 = `batch`) and `/Gather_1` (axis 1 = `seq_len`).
  2. `/Concat` -> `/Reshape` producing `target_shape = [batch, 1, seq_len, seq_len]`.
  3. To handle PyTorch's `.expand(-1, 1, -1, -1)` semantics (where `-1` means "keep existing dimension", whereas ONNX `Expand` uses `1` for "keep existing dimension"), the ONNX graph executes:
     - `/Equal(target_shape, [-1, -1, -1, -1])`
     - `/Where(equal_cond, [1, 1, 1, 1], target_shape)`
     - `/Expand(unsqueezed_mask, where_output)`
- When `onnx-gomlx` builds the graph with dynamic input shapes (`(Int64)[batch=?, seq_len=?]`):
  - `shapes.DynamicDim` is `-1`!
  - `convertShape` (`onnx-gomlx/internal/onnxgomlx/ops.go:592`) creates a constant tensor `[-1, -1]` from `operand.Shape().Dimensions`.
  - `materializeConstantExpression` (`onnx-gomlx/internal/onnxgomlx/materialize.go:160`) also evaluates `Shape` to `[-1, -1]`.
  - Then `/Equal` compares `[-1, 1, -1, -1]` against the PyTorch sentinel `[-1, -1, -1, -1]`, which evaluates to `[true, false, true, true]` because `shapes.DynamicDim` collides with `-1`!
  - `/Where` then replaces the `-1`s with `1`, producing `[1, 1, 1, 1]`.
  - Finally, `convertExpand` (`onnx-gomlx/internal/onnxgomlx/ops.go:1615-1650`) sees `targetDims = [1, 1, 1, 1]`, replaces `1` with `operand.Shape().Dim(ii)`, and since `operand` (`unsqueezed_mask`) has shape `[batch=?, 1, 1, seq_len=?]`, axis 2 stays `1` instead of becoming `seq_len=?`!

**Fix in `onnx-gomlx`**:
- In `onnx-gomlx/internal/onnxgomlx/dynamicshape.go` (`evalDynamicDims`) and `ops.go` (`convertExpand`):
  - Add support for `Where` and `Equal` in `evalDynamicDims`: when `evalDynamicDims` traces a shape tensor through `Where(Equal(shapeExpr, -1), 1, shapeExpr)`, a `dynamicDim` representing a runtime dynamic axis is *not* equal to the constant `-1` sentinel (it is a positive runtime dimension), so `Equal(dynamicDim, -1)` is `false` and `Where` selects the `onFalse` branch (`shapeExpr`)!
  - In `convertExpand` (`ops.go:1615`), if `operand.Shape().IsDynamic()` or the shape input traces to dynamic dimensions via `m.evalDynamicDims(node.Inputs[1])`, use `evalDynamicDims` + `graph.DynamicBroadcastInDim` (or `evalDynamicDims` to resolve which axes are dynamic vs `1`) before falling back to `materializeConstantExpression` (which conflates `shapes.DynamicDim` with `-1`).
  - Furthermore, in `materialize.go` (`evalConstant` for `"Shape"`, line 160), if `inShape.IsDynamic()`, `evalConstant` should return an error (cannot materialize a dynamic shape as a static constant tensor) rather than returning `-1` literals that corrupt downstream constant folding (`Equal`, `Where`, `Mul`, etc.)! Check existing dynamic shape tests in `onnx-gomlx` when making this change.

---

### Opportunity 2 (High Impact — 8.81s / 10.63% CPU in ONNX): Absorb `BSHD <-> BHSD` Transposes into `DetectSDPA`

#### Symptom
`minilm_onnx.prof` spends **8.81s (10.63% of total CPU time)** in `transpose4D_0213Block` (`runtime.memmove`), whereas `minilm_hf.prof` spends **0.00s** in transpose.

#### Root Cause
- In `go-huggingface` (`mltransformer`), `attention.MultiHeadAttention` uses `attention.LayoutBSHD` (`[batch, seq_len, num_heads, head_dim]`). Projecting `x` to `[batch, seq_len, 384]` and reshaping to `[batch, seq_len, 12, 32]` is a zero-copy metadata operation (`Reshape` / `DynamicReshape`), and `FusedScaledDotProductAttention` natively supports `LayoutBSHD` (`data.layout == compute.AxesLayoutBSHD` in `compute/internal/gobackend/fusedops/sdpa.go`), striding directly over `BSHD` memory with zero transposes.
- In ONNX (`KnightsAnalytics/all-MiniLM-L6-v2` and standard PyTorch exports):
  - `Q`, `K`, `V` are reshaped to `[batch, seq_len, 12, 32]` (`BSHD`) and then each is transposed via `Transpose(perm=[0, 2, 1, 3])` to `[batch, 12, seq_len, 32]` (`BHSD`).
  - Attention is computed in `BHSD`.
  - The attention output `[batch, 12, seq_len, 32]` is transposed via `Transpose(perm=[0, 2, 1, 3])` back to `[batch, seq_len, 12, 32]` (`BSHD`) before the final `[batch, seq_len, 384]` reshape.
  - Across 6 layers, that is **24 full 4D tensor transposes per forward pass**.
- Currently, `DetectSDPA` (`onnx-gomlx/internal/onnxgomlx/fusion/sdpa.go`) matches starting from `MatMul(Q, K)` and ending at `MatMul(probs, V)`, emitting `attention.Core(G, q, k, v, scale, mask, nil, attention.LayoutBHSD, false, false)` and leaving all 4 `Transpose(perm=[0, 2, 1, 3])` nodes outside the fused group.

#### Fix in `onnx-gomlx/internal/onnxgomlx/fusion/sdpa.go`
In `tryMatchSDPA`:
1. After matching the core SDPA pattern (`qkMatMul` ... `avMatMul`), inspect:
   - Whether `qInputName`, `kInputName`, and `vInputName` are all produced by `Transpose` nodes with `perm == [0, 2, 1, 3]` (note: if `kTransposeNode` was a 2D transpose `[0, 1, 3, 2]` on `BHSD`, also check the `[0, 2, 1, 3]` transpose feeding `kTransposeNode`; or if `K` was transposed directly from `BSHD` to `BHDS` via `perm=[0, 2, 3, 1]`, handle that too!).
     - Specifically in `KnightsAnalytics/all-MiniLM-L6-v2`:
       - `Q`: `Reshape` -> `Transpose(perm=[0, 2, 1, 3])` -> `qkMatMul`
       - `K`: `Reshape` -> `Transpose(perm=[0, 2, 1, 3])` -> `Transpose(perm=[0, 1, 3, 2])` -> `qkMatMul`
       - `V`: `Reshape` -> `Transpose(perm=[0, 2, 1, 3])` -> `avMatMul`
   - Whether `avMatMul.Outputs[0]` has a single consumer in `consumers` which is a `Transpose` node with `perm == [0, 2, 1, 3]`, and none of the intermediate `BHSD` tensors (`q`, `k`, `v`, `avMatMul.Outputs[0]`) are graph outputs or consumed outside the candidate.
2. When all 4 `BSHD <-> BHSD` transposes are present and internal-only:
   - Add the 3 input `Transpose(perm=[0, 2, 1, 3])` nodes and 1 output `Transpose(perm=[0, 2, 1, 3])` node to `internalNodes`.
   - Set the candidate's `ExternalInputs` to use the pre-transpose `BSHD` tensors (`qBSHD`, `kBSHD`, `vBSHD`).
   - Set the candidate's `ExternalOutputs` to `outTransposeNode.Outputs[0]`.
   - Emit `attention.Core(G, qBSHD, kBSHD, vBSHD, p.Scale, mask, nil, attention.LayoutBSHD, false, false)`.
3. **Nuance on Mask Layout in `attention.Core`**:
   - Verify in `gomlx/ml/layers/attention/core.go` and `compute/shapeinference/shapeinference.go` (`FusedScaledDotProductAttention`): even when `layout == attention.LayoutBSHD`, `attention.Core` and `FusedScaledDotProductAttention` expect the rank-4 mask in `[batch, numHeads/1, qSeq/1, kvSeq]` layout (i.e. `BHSD` broadcastable shape). So the mask input does **not** need to be transposed when switching `Q, K, V` from `LayoutBHSD` to `LayoutBSHD`!

---

### Opportunity 3 (High Impact — 6.27s / 7.57% CPU in ONNX): Eliminate Intermediate Buffer & Split-Bias Copy in `FusedAttentionQKVProjection`

#### Symptom
In `minilm_onnx.prof`, `qkvSplitBiasFloat32` (`copyAndAddBiasFloat32AVX512Asm` in `compute/internal/gobackend/fusedops/attentionqkv.go`) takes **6.27s (7.57% of total CPU time)**.

#### Root Cause
Look at `FusedAttentionQKVProjection` in `compute/internal/gobackend/fusedops/attentionqkv.go`:
1. At graph build time (line 96), it creates a `DotGeneral` sub-node `x @ wQKV` (`[batch, seq_len, 384] @ [384, 1152] -> [batch, seq_len, 1152]`) **without bias**.
2. At execution time, `DotGeneral` allocates a `combined` buffer of size `[batch*seq_len, 1152]` (`8192 * 1152 * 4 = 37.7 MB`) and writes the matmul output to DRAM without adding biases.
3. Next, `execFusedAttentionQKVProjection` (line 123) allocates 3 *additional* buffers (`qBuf`, `kBuf`, `vBuf`, each `[batch*seq_len, 384]`, totaling another `37.7 MB`) and calls `qkvSplitBiasFloat32` (line 176), which reads `37.7 MB` from `combined` and writes `37.7 MB` to `qBuf`, `kBuf`, `vBuf` while adding `biasQ`, `biasK`, `biasV`.
4. Across 6 layers and 100 benchmark batches, that extra DRAM round-trip copies ~45 GB of memory and costs **6.27s** of CPU time.
5. By contrast, in `go-huggingface`, `Q`, `K`, `V` are 3 `FusedDense` ops (`compute/internal/gobackend/fusedops/dense/dense.go`), which call `matmul.ExecuteWithEpilogue` directly—adding biases inside the L1/L2 cache-resident GEMM microkernel epilogue and writing straight to `qBuf`, `kBuf`, `vBuf` with zero intermediate buffer and zero post-matmul copy!

#### Fix Options in `compute/internal/gobackend/fusedops/attentionqkv.go`
- **Option A (Direct `matmul.ExecuteWithEpilogue` per projection or strided output in `FusedAttentionQKVProjection`)**:
  - In `FusedAttentionQKVProjection`, instead of creating an unfused `DotGeneral` node that writes a combined `[batch, qDim+2*kvDim]` buffer, execute the matmul writing directly into `qBuf`, `kBuf`, `vBuf` with their respective bias epilogues (`biasQ`, `biasK`, `biasV`).
  - Even better: if `wQKV` has shape `[inFeatures, qDim + 2*kvDim]` (`LayoutNonTransposed`), its packed RHS panels (`RHSL1KernelCols = 32` in AVX-512, `16` in AVX2) naturally partition along the output columns whenever `qDim` and `kvDim` are multiples of `RHSL1KernelCols` (e.g., `384 = 12 * 32`)! Or `wQKV` can be pre-packed into 3 RHS panel sets (`wQ`, `wK`, `wV`), and `execFusedAttentionQKVProjection` can pack LHS (`x`) **once** and run the 3 RHS panel sets writing directly to `qBuf`, `kBuf`, `vBuf` with `Epilogue{Bias: biasQ/K/V}`!
  - That gives the best of both worlds: LHS `x` is packed only **once** (unlike 3 separate `FusedDense` calls which pack `x` 3 times), AND `qBuf`, `kBuf`, `vBuf` are written directly by the GEMM kernel with fused bias epilogues (eliminating the `37.7 MB` `combined` buffer and the entire **6.27s** `qkvSplitBiasFloat32` pass!).

---

### Opportunity 4 (Medium Impact — ~1.5s–2s CPU in ONNX): Fuse `MatMul + Add(bias)` (with `ActivationNone`) into `nn.Dense`

#### Symptom
In the GoMLX graph generated for `KnightsAnalytics/all-MiniLM-L6-v2`, every transformer layer has two linear projections without an activation function:
- `/encoder/layer.N/attention/output/dense` (`MatMul` + `Add`)
- `/encoder/layer.N/output/dense` (`MatMul` + `Add`)

While `/encoder/layer.N/intermediate/dense` (`MatMul` + `Add` + `Gelu`) is fused into `FusedDense` + `FusedActivation`, the 12 non-activation linear layers are emitted as unfused:
```
Node #280 DotGeneral(lhs=[#278], rhs=[#279]) -> [batch=?, seq_len=?, 384]
Node #281 Reshape(x=[#14], dimensions=[1 1 384]) -> [1, 1, 384]
Node #282 Add(lhs=[#281], rhs=[#280]) -> [batch=?, seq_len=?, 384]
```
This requires an extra full-tensor `simdV2V1AddFloat32` pass over DRAM for each of the 12 layers instead of adding the bias in-register during the `FusedDense` GEMM epilogue.

#### Root Cause
In `onnx-gomlx/internal/onnxgomlx/fusion/dense_gelu.go` (lines 72–77):
```go
actType, actOutputs, actInternal, actExternal := matchActivation(m, addNode.Outputs[0], consumed)
if actType == activation.TypeNone {
    return nil
}
```
`tryMatchDenseActivation` explicitly aborts if no activation follows `MatMul + Add(bias)`.

#### Fix in `onnx-gomlx/internal/onnxgomlx/fusion/dense_gelu.go`
- Allow `tryMatchDenseActivation` to match `MatMul + Add(bias)` even when `actType == activation.TypeNone`!
- **Nuances to watch out for**:
  1. **Interaction with `DetectQKV`**: In `onnx-gomlx/internal/onnxgomlx/fusion.go` (lines 47–51), `DetectQKV` runs *before* `detectDenseActivationCandidates` and marks all Q, K, V `MatMul` and `Add` nodes in `consumed`. Because `detectDenseActivationCandidates` skips any node in `consumed`, QKV nodes will not be stolen by `Dense(ActivationNone)` fusion.
  2. **Output wiring when `actType == activation.TypeNone`**: When `actType == activation.TypeNone`, `matchActivation` returns `nil` for `actOutputs`, `actInternal`, `actExternal`. In that case:
     - `externalOutputs` should be `[]string{addOutName}` (`addNode.Outputs[0]`).
     - `internalNodes` should just be `map[string]bool{matmulNode.Name: true, addNode.Name: true}`.
     - No consumer count restriction is needed on `addOutName` (unlike when an activation follows, where `addOutName` must only be consumed by the activation).
  3. **Verify weight rank / bias rank**: Ensure `biasName` is a 1D initializer of length `outFeatures` (`wShape.Dimensions[wShape.Rank()-1]`) and `wShape.Rank() >= 2` so `nn.Dense(x, w, bias, compute.DenseLayoutInputOutputs, activation.TypeNone)` is valid.

---

### Opportunity 5 (Bonus — 16.15s / 23.73% CPU in Native HF / `LayoutTransposed` GEMMs): SIMD Packing for `LayoutTransposed` RHS in `compute`

#### Symptom
While profiling `sentence-transformers/all-MiniLM-L6-v2` (`minilm_hf.prof`), `unsafePackLHS` consumed **16.15s (23.73% of total CPU time!)**—more than half the time of the AVX-512 GEMM kernel itself (`28.60s`)!

#### Root Cause
1. `go-huggingface` loads Linear weights from `safetensors` in PyTorch layout `[out_features, in_features]` and configures `mltransformer` with `WithTransposedWeights(true)` (`compute.DenseLayoutOutputsInput` -> `dot.LayoutTransposed`).
2. Furthermore, because `go-huggingface` passes model weights as graph `Parameter` nodes (via `model.Store` / `context.Context`) rather than `Constant` nodes (`wNode.IsConstant() == false`), `FusedDense` (`compute/internal/gobackend/fusedops/dense/dense.go:153`) does **not** attach `RHSConstantData`, so weights are not pre-packed once across executions.
3. Consequently, inside `avx512LargeFloat32` (`compute/internal/gobackend/dot/matmul/avx512/large.go:404-408`) and `avx2LargeFloat32` (`avx2/large.go:401-405`), every worker thread packs the `LayoutTransposed` RHS panel on the fly by calling:
   ```go
   // For LayoutTransposed, the rhs has the same layout as the lhs, so we use packLHS instead.
   unsafePackLHS(rhsMatrix, packedRHS, rhsPanelColIdx, contractingPanelIdx, contractingSize,
       rhsPanelWidth, contractingPanelWidth, params.RHSL1KernelCols)
   ```
4. For AVX-512 `float32`, `params.RHSL1KernelCols` is **32** (and for AVX2, **16**).
   - While LHS packing (`LHSL1KernelRows = 8` or `4`) has a fast AVX-512 assembly routine (`avx512PackLHSKernelRows8Float32Asm`, which took only `0.90s`), `unsafePackLHS` (`compute/internal/gobackend/dot/matmul/packing.go:197`) only unrolls `kernelRows == 2` and `kernelRows == 4` and falls back to a nested scalar loop for `kernelRows == 32` (and `16`)!
   - Worse still, because `cachedRHSPanels` is empty when `wNode` is a `Parameter`, every worker thread redundantly packs the same RHS panels in scalar Go code!

#### Fix in `compute/internal/gobackend/dot/matmul`
1. **Add SIMD/assembly packing for `LayoutTransposed` RHS (`kernelRows == 32` on AVX-512, `kernelRows == 16` on AVX2)**:
   - In `avx512`: implement `avx512PackRHSTransposed` (or handle `kernelRows == 32` in `avx512PackLHS`) using 16x16 / 8x8 vector transpose blocks (`VUNPCKLPS`/`VUNPCKHPS` + `VSHUFF32X4` + `VPERMPD`), replacing the scalar `unsafePackLHS` call at `avx512/large.go:406` and `621`.
   - In `avx2`: implement `avx2PackRHSTransposed` for `kernelRows == 16` at `avx2/large.go:403` and `618`.
2. **Pre-pack RHS once per `ExecuteWithEpilogue` call when `lhsCrossSize` is large and `cachedRHSPanels` is nil**:
   - In `avx512LargeFloat32` (`avx512/large.go`), when multiple worker threads split along `M` (`rowStart..rowEnd`) or `N`, if `cachedRHSPanels == nil` and `numMBlocks > 1`, multiple threads currently pack the same RHS panel redundantly. Pre-packing RHS panels in parallel before the tile loop (or caching across M-blocks) avoids redundant work across threads.

---

### Note on `Gelu` vs `GeluApprox` (1.76s difference)
- In `KnightsAnalytics/all-MiniLM-L6-v2` (ONNX), the FFN activation is exported as exact GELU (`x * 0.5 * (1 + Erf(x / Sqrt(2)))`), which `onnx-gomlx` fuses into `activation.TypeGelu` (`erfFloat32AVX512`, taking **3.74s** in `minilm_onnx.prof`).
- In `sentence-transformers/all-MiniLM-L6-v2` (`config.json` has `"hidden_act": "gelu"`), `activation.FromName("gelu")` in GoMLX maps `"gelu"` to `TypeGeluApprox` (`geluApproxFloat32AVX512`, taking **1.98s** in `minilm_hf.prof`).
- This accounts for **1.76s** of the CPU time difference and is expected since the ONNX graph explicitly specifies exact `Erf`-based GELU.

---

## 4. Suggested Implementation Order

1. **Step 1 (`compute` + `onnx-gomlx`)**: Fix `FusedScaledDotProductAttention` for `[batch, 1, 1, kv_len]` masks (`mSeqDim == 1`) in `compute/internal/gobackend/fusedops/sdpa.go`, and fix dynamic shape `Expand` / `Shape` constant folding in `onnx-gomlx/internal/onnxgomlx/{ops.go,materialize.go,dynamicshape.go}`.
2. **Step 2 (`onnx-gomlx`)**: Update `DetectSDPA` in `onnx-gomlx/internal/onnxgomlx/fusion/sdpa.go` to absorb the 3 input `[0, 2, 1, 3]` transposes and 1 output `[0, 2, 1, 3]` transpose and emit `attention.LayoutBSHD`.
3. **Step 3 (`onnx-gomlx`)**: Update `tryMatchDenseActivation` in `onnx-gomlx/internal/onnxgomlx/fusion/dense_gelu.go` to fuse `MatMul + Add(bias)` with `activation.TypeNone` into `nn.Dense`.
4. **Step 4 (`compute`)**: Optimize `FusedAttentionQKVProjection` in `compute/internal/gobackend/fusedops/attentionqkv.go` to eliminate the intermediate `combined` buffer and separate `qkvSplitBiasFloat32` pass.
5. **Step 5 (`compute`)**: Add SIMD RHS packing for `LayoutTransposed` (`kernelRows = 32` for AVX-512, `16` for AVX2) in `compute/internal/gobackend/dot/matmul/{avx512,avx2}`.
6. **Step 6**: Re-run the `benchmark_embed` profile on both models to verify wall-clock and CPU time improvements.

---

## 5. Completed Optimizations (Implemented & Committed)

All five original opportunities above plus three follow-up optimizations have been implemented and verified:

1. **Phase 1 (`compute` commit `d726ba0`, `onnx-gomlx` commit `e985880`)**:
   - **Opportunity 1A (`compute`)**: Allowed `mSeqDim == 1` (`[batch, 1, 1, kv_len]` masks) in `FusedScaledDotProductAttention` (`sdpa.go`, `sdpa_simd.go`, `sdpa_neon_arm64.go`, `sdpa_vjp.go`) using `mSeqStride = 0`.
   - **Opportunity 1B (`onnx-gomlx`)**: Fixed dynamic shape constant folding in `materialize.go`, `dynamicshape.go`, and `ops.go` (`convertExpand`) so `shapes.DynamicDim` (`-1`) no longer collides with PyTorch's `-1` sentinel in `Equal`/`Where`/`Expand`.
   - **Opportunity 2 (`onnx-gomlx`)**: Updated `DetectSDPA` (`fusion/sdpa.go`) to absorb the 3 input `[0, 2, 1, 3]` transposes and 1 output `[0, 2, 1, 3]` transpose and emit `attention.LayoutBSHD`.
   - **Opportunity 3 (`compute`)**: Replaced the intermediate `[batch, qDim+2*kvDim]` buffer and post-matmul `qkvSplitBias` pass in `FusedAttentionQKVProjection` (`attentionqkv.go`) with direct `matmul.ExecuteWithEpilogue` writes into `qBuf`, `kBuf`, `vBuf` using sliced RHS weight views (`sliceLastDimBuffer`).
   - **Opportunity 4 (`onnx-gomlx`)**: Updated `tryMatchDenseActivation` (`fusion/dense_gelu.go`) to fuse `MatMul + Add(bias)` with `activation.TypeNone` into `nn.Dense`.
   - **Opportunity 5 (`compute`)**: Added SIMD packing for `LayoutTransposed` RHS (`kernelRows == 32` in AVX-512 via 8×8 transpose tiles, `kernelRows == 16` in AVX2, and `kernelRows == 8` in ARM64 NEON via 4×4 `TRN1`/`TRN2` tiles), plus parallel pre-packing when `numMBlocks > 1` and `cachedRHSPanels == nil`.

2. **Phase 2 (`compute` commit `9c2838e`, `onnx-gomlx` commit `a298ceb`, `go-huggingface` commit `8376bd3`)**:
   - **Pre-packing Frozen / Constant Weights**:
     - Added content-addressed (FNV-1a + sample + pointer) RHS panel caching in `FusedDense` (`dense.go`) and `FusedAttentionQKVProjection` (`attentionqkv.go`) so even non-`Constant` `Parameter` weights (such as `go-huggingface` safetensors weights) are packed only once on the first step and reused on all subsequent steps.
     - Added `WithConstants(true)` in `go-huggingface/models/hf/model.go` so inference weights are emitted directly as graph `Constant`s.
     - Note: `FusedScaledDotProductAttention` takes dynamic activations (`Q, K, V`) rather than static weights and reads `K, V` directly in `BSHD`/`BHSD` layout without packing.
   - **Propagating `QuerySeqLen` & `KVSeqLen` from 2D `attention_mask` in `onnx-gomlx`**:
     - Extended `DetectSDPA` (`onnx-gomlx/internal/onnxgomlx/fusion/sdpa.go`) to trace additive mask patterns `(1 - Cast(attention_mask)) * -1e9` back to the rank-2 `[batch, seq_len]` integer `attention_mask`, compute per-batch token counts via `ReduceSum(Cast(mask, Int32), -1)`, and pass `.WithQuerySeqLen(seqLens).WithKVSeqLen(seqLens)` to `attention.Core`.
     - Eliminates `O(S²)` attention compute on padding tokens in bucketed batches (reducing `sdpaFloat32` CPU time by **~82%** on `bucket=8192`).
   - **ARM64 NEON Assembly for `GeluExact` & `GeluApprox`**:
     - Added `geluExactFloat32NEONAsm` and `geluApproxFloat32NEONAsm` in `compute/internal/gobackend/activations/{gelu_neon_arm64.s,neon_arm64.go}` using Estrin's parallel polynomial tree, overlapped `VDIV` latency, `(x * exp2x) / (exp2x + 1)`, and 20 pinned constants in `V12`–`V31` (**1.29×–1.60× speedup** on Apple M6).

---

## 6. Follow-Up Opportunity for AMD 9950X3D: AVX2 & AVX-512 GELU Optimization (Implemented)

In `compute/internal/gobackend/activations/avx2/` and `compute/internal/gobackend/activations/avx512/`, `GeluAVX2`/`GeluExactAVX2` and `GeluAVX512`/`GeluExactAVX512` have now been replaced with handwritten AMD64 assembly kernels (`gelu_amd64.s`).

### Expected Impact
- **Kernel Microbenchmark**: **~1.5× to 1.8× speedup** on `GeluExact` and `GeluApprox`.
- **End-to-End Model CPU Time**: **~1.0% to 1.5% reduction** (in the baseline AMD 9950X3D profile, `FusedGelu` accounted for **3.74s / 4.51%** of total CPU time for `GeluExact` in ONNX and **1.98s / 2.91%** for `GeluApprox` in HF; after the ~3× end-to-end speedup from Opportunities 1–5, `FusedGelu`'s relative share of the remaining runtime is roughly 8–12%).

### What Was Implemented (`avx512/gelu_amd64.s` & `avx2/gelu_amd64.s`)

1. **Estrin's Parallel Polynomial Tree + Algebraic Simplifications**:
   - Both `exp` (degree-7 Cephes) and `erf` (degree-5 Abramowitz & Stegun 7.1.26) now evaluate polynomials via Estrin's parallel tree (`q67, q45, q23, q01` and `p45, p23` computed independently in parallel) using non-destructive `VFMADD231PS` / `VFNMADD231PS`.
   - `GeluApprox` (`geluApproxFloat32AVX512Asm`, `geluApproxFloat32AVX2Asm`) uses the algebraic identity:
     $$
     0.5 \cdot x \cdot (1 + \tanh(z)) = \frac{x \cdot e^{2z}}{e^{2z} + 1}
     $$
     folding the factor of $2$ into $\tilde{\alpha} = 2\sqrt{2/\pi} \approx 1.5957691216$ and eliminating 3 vector FP operations per vector (`VSUBPS`, `VADDPS`, `VMULPS` by `0.5`).
   - `GeluExact` (`geluExactFloat32AVX512Asm`, `geluExactFloat32AVX2Asm`) issues `VDIVPS` for $t = 1 / (1 + p|x|)$ *before* evaluating the `exp(-x²)` range reduction and Estrin polynomial so the 11–14 cycle `VDIVPS` latency is completely hidden behind the independent `exp` FMA instructions.
   - Range reduction uses the IEEE-754 magic bias (`12582912.0` = $1.5 \times 2^{23}$) + `VPSLLD $23` + integer exponent addition (`VPADDD`), saving float-to-int conversions.

2. **Zero-Spill Register Allocation**:
   - **AVX-512 (`internal/gobackend/activations/avx512/gelu_amd64.s`)**: Pinned all constants in `Z11`–`Z31` (for `GeluExact`) and `Z16`–`Z31` (for `GeluApprox`) once before the loop from `·geluAVX512Consts(SB)` via `VBROADCASTSS`, and unrolled 2× (`32 float32`s = 2 `ZMM` vectors per iteration) using `Z0`–`Z10` with a 1-vector (`16 float32`s) tail check and a stack-buffer tail in Go for `n % 16 != 0`.
   - **AVX2 (`internal/gobackend/activations/avx2/gelu_amd64.s`)**: Pinned `0.5` (`Y14`) and `1.0` (`Y15`) in registers and read polynomial/range-reduction constants directly from a 256-bit aligned `RODATA` table `·geluAVX2Consts(SB)` via memory operands (`VFMADD231PS offset(CX), Y_src, Y_dst`), allowing 2× unrolling (`16 float32`s = 2 `YMM` vectors per iteration) strictly inside `Y0`–`Y15` with zero stack spills.
   - Both kernels emit `VZEROUPPER` before every `RET`.

3. **Measured Benchmark Results (AMD Ryzen 9 9950X3D)**:
   - **GELU Microbenchmarks (`BenchmarkFlavorsComparison`, `compute` commit `c2ad19b`)**:
     | Benchmark | `NoSIMD` | `PortableSIMD` (`simd.Float32s`) | `AVX2_ArchSIMD` (`gelu_amd64.s`) | `AVX512_ArchSIMD` (`gelu_amd64.s`) | Speedup vs Previous `AVX512` |
     | :--- | ---: | ---: | ---: | ---: | ---: |
     | `Small_512/GeluExact` | 3,589 ns | 129.0 ns | 182.2 ns | **92.3 ns** | **~2.4×** |
     | `Small_512/GeluApprox` | 5,796 ns | 162.6 ns | 162.8 ns | **82.0 ns** | **~2.1×** |
     | `Large_65536/GeluExact` | 904.4 µs | 15.93 µs | 23.20 µs | **11.82 µs** | **2.47×** (was 29.2 µs) |
     | `Large_65536/GeluApprox` | 1,043.0 µs | 20.18 µs | 20.79 µs | **10.45 µs** | **2.11×** (was 22.1 µs) |
   - Note: Because `GeluExact` completely overlaps the `VDIVPS` latency with the `exp(-x²)` Estrin polynomial, `GeluExact` (`11.82 µs`) is now within 13% of `GeluApprox` (`10.45 µs`), completely eliminating the original 1.76s `Erf` penalty in the ONNX model.

---

## 7. `-limit 1000` (10,000 Passages) Benchmark & Profile Analysis

Running with `-limit 1000` (10,000 passages, 1.0M total tokens, 773.3K non-padding tokens) on AMD Ryzen 9 9950X3D:

| Metric | ONNX (`KnightsAnalytics/all-MiniLM-L6-v2`) | Native HF (`sentence-transformers/all-MiniLM-L6-v2`) | Notes |
| :--- | ---: | ---: | :--- |
| **Warmup Duration** (11 unique shapes) | 1.4s (28.9s CPU) | 1.3s (28.5s CPU) | Runs only the first batch of each unique `(batch, seq_len)` shape |
| **Benchmark Wall Time (10K passages)** | **11.9s** (**838 items/s**, 84.2K tok/s) | **12.0s** (**833 items/s**, 83.6K tok/s) | Steady-state peaks at **~890 items/s** (~70.7K non-pad tok/s) |
| **Benchmark CPU Time** | **260.7s** (26.1 ms/passage) | **263.4s** (26.4 ms/passage) | ONNX is now slightly **faster** than Native HF |
| **Total Sampled CPU Time** (warmup + bench) | **289.65s** | **291.90s** | |
| 1. `avx512LargeKernelFloat32Asm` (GEMM) | 152.95s (52.81%) | 159.53s (54.65%) | Core matrix multiplications |
| 2. `sdpaFloat32AVX512Dim32` (cum `FusedSDPA`) | **63.06s (21.77%)** | **59.14s (20.26%)** | **#1 Remaining Optimization Target** |
| — `reduceSum16` / `reduceSum8` (in `Q @ K^T`) | 15.76s (5.44%) | 15.20s (5.21%) | Per-key horizontal reduction inside `ki` loop |
| — `softmaxAndValueAccumDim32` (`P @ V` + `Exp`) | 28.20s (9.74%) | 26.79s (9.18%) | Includes **8.46s in scalar `fastmath.Exp32` tail!** |
| 3. `addBiasFloat32AVX512Asm` (`ApplyEpilogue`) | 14.84s (5.12%) | 16.38s (5.61%) | Separate bias pass before activation |
| 4. `avx512PackLHSKernelRows8Float32Asm` | 13.03s (4.50%) | 12.78s (4.38%) | Dynamic LHS activation packing (3× in QKV) |
| 5. `FusedGelu` (`geluExact` vs `geluApprox` Asm) | **11.81s (4.08%)** | **12.73s (4.36%)** | `geluExact` is now faster than `geluApprox` in-model |
| 6. `simdAddVVFloat32@simd512` (Residual `Add`) | 10.24s (3.54%) | 6.95s (2.38%) | Residual connections |
| 7. `layerNormFloat32AVX512` (`FusedLayerNorm`) | 7.95s (2.74%) | 8.74s (2.99%) | LayerNorm |

### Why `-limit 1000` Behaves Differently From `-limit 100`
1. **Warmup vs. Steady-State Ratio**: During `-warmup`, `runBenchmark` only executes the model on the **first occurrence of each unique `(batchSize, seqLen)` shape** (11 unique shapes = 89.8K tokens) and skips all duplicate shapes (`seenShapes.LoadOrStore`). With `-limit 100` (1K passages = ~143K tokens), warmup + end-of-stream partial bucket flushes dominated the run. With `-limit 1000` (10K passages = 1M tokens), steady-state full-bucket execution (`~890 passages/s`, `70.7K non-pad tokens/s`) dominates, and only drops at the very tail (`9.4K -> 10K`) when partially-filled buckets are flushed when `bucketsInputChan` closes.
2. **Exact Non-Pad `KVSeqLen` Exposes the Scalar Tail in `softmaxAndValueAccumDim32`**: Because Phase 2 trims `kvLenUnmasked` to each sentence's exact non-padded token count (`kvLimit`), `kvLenUnmasked` is no longer a power-of-two bucket multiple (`64`, `96`, `128`)—it is an arbitrary integer (`1..128`), so `kvLenUnmasked % 16 != 0` on **15 out of 16 sentences**, hitting the scalar `fastmath.Exp32` tail loop!

---

## 8. Further Ideas for Optimizing the Sentence Embedder Benchmark (`all-MiniLM-L6-v2`)

After Phases 1 & 2 and the AVX-512/AVX2 GELU assembly kernels, the sentence embedder pipeline (`[64, 128]` / `8192` tokens across 6 Transformer layers, `hidden_size = 384`, `num_heads = 12`, `head_dim = 32`, `intermediate_size = 1536`) spends almost all of its remaining CPU time in three areas:
1. `FusedDense` & `FusedAttentionQKVProjection` (`matmul`: `57.6%`)
2. `FusedScaledDotProductAttention` (`sdpaFloat32AVX512Dim32`: `21.8%`)
3. `FusedDense` epilogues (`addBias`: `5.1%`, `Gelu`: `4.1%`) & `FusedLayerNorm` (`2.7%`)

Below are four concrete follow-up optimizations ranked by measured impact in the `-limit 1000` CPU profile:

### Idea A: Optimize `FusedScaledDotProductAttention` for `head_dim = 32` (`63.1s` / `21.8%` of Total CPU) — **High Impact**

In `all-MiniLM-L6-v2`, every attention head has `head_dim = 32` (exactly **2 `ZMM` registers** on AVX-512 or **4 `YMM` registers** on AVX2). Line-by-line profiling of `compute/internal/gobackend/fusedops/avx512/sdpa.go` (`sdpaFloat32AVX512Dim32` and `softmaxAndValueAccumDim32`) reveals five bottlenecks:

0. **Vectorize the `kvLenUnmasked % 16` Tail in `softmaxAndValueAccumDim32` (`8.46s` in scalar `fastmath.Exp32`!)**:
   - In `avx512/sdpa.go:177-186`, after `for ; k+16 <= kvLenUnmasked; k += 16` (which takes only `2.31s` across all 16-element blocks!), the remaining `1..15` elements fall into a scalar loop calling `fastmath.Exp32(s - rowMax)` (**8.46s** of CPU time!).
   - **Fix**: Pad `scoresScratch[kvLenUnmasked : (kvLenUnmasked+15)&^15]` with `-Inf` (or load a padded 16-float vector for the tail and zero lanes `>= rem`), so the tail is evaluated with a single vectorized `exp512` + `vInvSum` multiply instead of up to 15 scalar `fastmath.Exp32` calls per query row.

1. **Per-Key Horizontal Reduction (`reduceSum16`) Inside the Inner `ki` Loop (`avx512/sdpa.go:296-302`)**:
   - Currently, for each key `ki`, `sdpaFloat32AVX512Dim32` loads 1 key (`k0`, `k1`), computes `sumVec := q0.Mul(k0).Add(q1.Mul(k1))`, and immediately calls `reduceSum16(sumVec)` (which executes `VEXTRACTF32X8` + `VADDPS` + `VEXTRACTF128` + `VADDPS` + `VHADDPS` + scalar extract) for **every single key**!
   - **Fix**: Process **4 keys at a time** (`ki, ki+1, ki+2, ki+3`) using 4 accumulator vectors `s0, s1, s2, s3` (`s0 = q0*k0_0 + q1*k0_1`, etc.) and reduce 4 vectors into 4 contiguous floats simultaneously (via `VUNPCKLPS`/`VUNPCKHPS` or pairwise adds), cutting horizontal reduction instructions by **4×** and hiding FMA latency.

2. **Single-Accumulator Latency Bottleneck & `w == 0` Branch in Value Accumulation (`P @ V`, `avx512/sdpa.go:205-221`)**:
   - In `softmaxAndValueAccumDim32`, the `P @ V` loop accumulates into a single pair of registers (`out0`, `out1`):
     ```go
     out0 = v0.MulAdd(wVec, out0)
     out1 = v1.MulAdd(wVec, out1)
     ```
   - Because `VFMADD` has a 4-cycle latency on Zen 4/5 and `out0`/`out1` are updated every iteration, the loop is bottlenecked by the 4-cycle dependency chain on `out0` and `out1`. Worse, it has a branch `if w == 0 { continue }` on every token `ki`.
   - **Fix**: Remove the `if w == 0` branch inside the unmasked loop (`exp(score - max)` is rarely exact bitwise `0.0` for valid unmasked tokens) and unroll `ki` by **4×** using 4 independent accumulator pairs (`out0_0..out0_3`, `out1_0..out1_3`), hiding the 4-cycle FMA latency completely and increasing `P @ V` FMA throughput up to **4×**.

3. **`exp512` Broadcasts 14 Constants & Uses Serial Horner Inside the Softmax Loop (`avx512/sdpa.go:48-92`)**:
   - `exp512(x)` constructs 14 `archsimd.BroadcastFloat32x16(...)` constants locally and uses a 6-step serial Horner chain + `RoundScaled(0)` + `ConvertToInt32()`.
   - **Fix**: Use the same Estrin parallel polynomial tree + magic-number range reduction (`12582912.0` + `VPSLLD $23`) we just wrote for GELU, or write a dedicated assembly kernel `sdpaDim32Float32AVX512Asm` / `sdpaDim32Float32AVX2Asm`.

4. **Single-Pass Online Softmax (FlashAttention-1/2 Register Rescaling)**:
   - Currently, `softmaxAndValueAccumDim32` makes **3 passes** over `scoresScratch[0:kvLen]` (Pass 1: write `scoresScratch` + find `maxScore`; Pass 2: read `scoresScratch`, compute `exp512`, write `scoresScratch` + sum; Pass 3: read `scoresScratch`, multiply by `invSum`, and accumulate `V`).
   - Because `head_dim = 32` fits in just **2 `ZMM` registers** (`out0, out1`), we can keep `out0, out1` in registers during the `ki` loop (or across 16-key blocks): compute 16 `Q @ K^T` scores into one `ZMM` register, update running `max` and `sum`, rescale `out0, out1` by `exp(oldMax - newMax)`, and immediately accumulate the 16 `V` rows into `out0, out1` without touching `scoresScratch` in memory at all!

#### Idea A Implementation & Measured Results (Completed)

Implemented in `compute/internal/gobackend/fusedops/avx512/sdpa_amd64.s`, `compute/internal/gobackend/fusedops/avx2/sdpa_amd64.s`, `avx512/sdpa.go`, `avx2/sdpa.go`, and `fusedops/sdpa.go`:

1. **Handwritten AVX-512 (`sdpaQueriesDim32NoMaskAVX512Asm`, `sdpaSoftmaxAndValueAccumDim32AVX512Asm`) & AVX2 (`sdpaQueriesDim32NoMaskAVX2Asm`, `sdpaSoftmaxAndValueAccumDim32AVX2Asm`) Assembly Kernels**:
   - **Pre-Scaled Query & 4-Key Unrolled `Q @ K^T`**:
     - Pins all 14 constants (`-Inf`, `scale`, `vMinLog = -88.0296936`, `vMaxLog`, `vLog2E`, `vMagic = 12582912.0`, `vLn2Hi`, `vLn2Lo`, `vP7..vP2`, `vOne/vBias127`) in `Z17`–`Z31` once outside the `qIdx` loop.
     - Pre-scales `q0, q1` by `scale` once per query row (`Z0, Z1`).
     - Processes 4 keys per iteration (`Z2, Z4, Z6, Z8`) and reduces all 4 vectors simultaneously into a single 4-lane `XMM` vector `X2 = [s0, s1, s2, s3]` via `VEXTRACTF32X8` + `VEXTRACTF128` + `VUNPCKLPS`/`VUNPCKHPS` + `VSHUFPS`, storing 4 scores with one `VMOVUPS` and updating the running max with `VMAXPS`.
   - **Zero Scalar `fastmath.Exp32` Tail (`-Inf` Tail Padding)**:
     - Pads 16 floats (AVX-512) or 8 floats (AVX2) at `scoresScratch[kvLenUnmasked]` with `-Inf` (`VMOVUPS Z17, (BX)`), then evaluates the vectorized Estrin degree-7 polynomial + magic-number `exp` loop over `ceil(kvLenUnmasked, VL)`. Because `-Inf - rowMax` clamps to `vMinLog = -88.0296936` ($n = -127 \implies (n+127) \ll 23 = 0.0\text{f}$), padded lanes evaluate to exact `0.0f` with **zero scalar `fastmath.Exp32` calls**.
   - **Deferred `invSum` Output Scaling & 8-Accumulator `P @ V`**:
     - Eliminates Pass 2 (`scoresScratch[k] *= invSum`) completely. Issues `VDIVSS` (`1.0 / sum`) before Stage 3 to hide division latency, accumulates unnormalized weights across 4 keys per iteration using 8 independent accumulators (`Z0`–`Z7` in AVX-512, 2 keys across `Y0`–`Y7` in AVX2) with memory-operand `VFMADD231PS`, and multiplies `out0, out1` by `vInvSum` once before storing.
   - **Applied `-Inf` Tail Padding + Deferred `vInvSum` to `Dim64`, `Dim128`, and `General`** in both `avx512/sdpa.go` and `avx2/sdpa.go`.

2. **Measured `-limit 1000` (10,000 Passages) Results Before vs. After Idea A (AMD Ryzen 9 9950X3D)**:

| Metric | ONNX Before Idea A | **ONNX After Idea A** | HF Before Idea A | **HF After Idea A** | Improvement |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Benchmark Wall Time (10K passages)** | 11.9s | **10.8s** | 12.0s | **10.9s** | **-1.1s (~10.2% faster)** |
| **Throughput (items/s)** | 838 items/s | **929 items/s** (peak **986/s**) | 833 items/s | **917 items/s** (peak **975/s**) | **+10.9% throughput** |
| **Token Throughput (total / non-pad)** | 84.2K / 64.9K tok/s | **93.3K / 71.9K tok/s** | 83.6K / 64.4K tok/s | **92.1K / 71.0K tok/s** | **+10.8% tok/s** |
| **Benchmark CPU Time** | 260.7s (4m21s) | **229.0s (3m49s)** | 263.4s (4m23s) | **232.0s (3m52s)** | **-31.7s CPU time (-12.2%)** |
| **Total Sampled CPU Time** (warmup + bench) | 289.65s | **256.83s** | 291.90s | **258.30s** | **-32.8s sampled CPU** |
| **`sdpaFloat32AVX512Dim32` (cum `FusedSDPA`)** | **63.06s (21.77%)** | **26.39s (10.28%)** | **59.14s (20.26%)** | **22.65s (8.77%)** | **2.39× – 2.61× speedup!** |
| — `sdpaQueriesDim32NoMaskAVX512Asm` | N/A | **25.35s (9.87%)** | N/A | **21.69s (8.40%)** | Entire `qIdx` loop in assembly |
| — `fastmath.Exp32` scalar tail | 8.46s | **0.00s (0%)** | 8.12s | **0.00s (0%)** | **100% eliminated** |
| **1. `avx512LargeKernelFloat32Asm` (GEMM)** | 152.95s (52.81%) | **155.34s (60.48%)** | 159.53s (54.65%) | **161.38s (62.48%)** | GEMM is now >60% of total CPU |
| **3. `addBiasFloat32AVX512Asm` (`ApplyEpilogue`)** | 14.84s (5.12%) | **14.57s (5.67%)** | 16.38s (5.61%) | **17.30s (6.70%)** | Next target (Idea C) |
| **4. `avx512PackLHSKernelRows8Float32Asm`** | 13.03s (4.50%) | **12.88s (5.01%)** | 12.78s (4.38%) | **13.88s (5.37%)** | Next target (Idea B) |
| **5. `FusedGelu` (`geluExact` / `geluApprox` Asm)** | 11.81s (4.08%) | **11.85s (4.61%)** | 12.73s (4.36%) | **12.22s (4.73%)** | Next target (Idea C) |
| **6. `simdAddVVFloat32@simd512` (Residual `Add`)** | 10.24s (3.54%) | **10.54s (4.10%)** | 6.95s (2.38%) | **5.67s (2.20%)** | |
| **7. `layerNormFloat32AVX512` (`FusedLayerNorm`)** | 7.95s (2.74%) | **7.57s (2.95%)** | 8.74s (2.99%) | **9.03s (3.50%)** | |

- **`-limit 100` (1,000 Passages) Reference**:
  - **ONNX**: **1.6s** (**645 items/s**, 89.6K tok/s, 31.9s CPU; down from original baseline 2.4s / 411 items/s = **+57% faster**).
  - **HF**: **1.5s** (**650 items/s**, 90.2K tok/s, 31.8s CPU; down from original baseline 1.9s / 520 items/s = **+25% faster**).

---

### Idea B: Pack Input `x` Only Once in `FusedAttentionQKVProjection` (`fusedops/attentionqkv.go`) — **Medium-High Impact**

- In `compute/internal/gobackend/fusedops/attentionqkv.go` (`qkvExecMatmul`, lines 297–317), after Opportunity 3 split the QKV matmul into 3 direct `matmul.ExecuteWithEpilogue` calls (`wQ`, `wK`, `wV`) to avoid allocating the `[batch, 3*384]` intermediate buffer:
  - The static RHS weights (`wQ`, `wK`, `wV`) are pre-packed once and cached via `wConstantData`.
  - However, the dynamic LHS activation `xFlat` (`[8192, 384]` = 12 MB per layer) is **packed 3 separate times** (once inside `ExecuteWithEpilogue` for `Q`, once for `K`, and once for `V`), and `Scheduler` worker tasks are spawned 3 separate times per layer (18 times per forward pass).
- **Fix**:
  - Either support a multi-output / strided column write in `matmul` so a single `Execute` call with packed `xFlat` writes directly to `qBuf`, `kBuf`, `vBuf` (especially trivial when `qDim = 384` and `kvDim = 384` are exact multiples of `RHSL1KernelCols = 32`), **OR**
  - Allow `ExecuteWithEpilogue` to accept/reuse pre-packed LHS panels across the `Q`, `K`, and `V` calls.
  - This eliminates **2/3 of the LHS packing work** across all 6 attention layers.

#### Idea B Implementation & Measured Results (Completed)

Implemented in `compute/internal/gobackend/dot/matmul/` (`matmul.go`, `epilogue.go`, `nosimd.go`, `avx512/large.go`, `avx512/avx512.go`, `avx2/large.go`, `avx2/avx2.go`, plus generated `BFloat16`, `Float16`, and `Float64` files) and `compute/internal/gobackend/fusedops/attentionqkv.go`:

1. **`matmul.ExecuteMultiRHSWithEpilogue` & `matmul.AppendRowWorkItems`**:
   - Added `MultiRHSTarget[I, O]` and `ExecuteMultiRHSWithEpilogue`, which executes multiple matrix multiplications ($Y_t = \text{Epilogue}_t(X \cdot W_t)$) that share the exact same LHS matrix $X$ (`xFlat`).
   - Added `AppendRowWorkItems` in `dot/matmul/nosimd.go` to partition work across batches and LHS rows (`M`) only (`RHSColStart = 0, RHSColEnd = rhsCrossSize`), aligning coarse row chunks to `params.LHSPanelCrossSize` (`Mc = 32`) when $M$ is large so no two workers split an `Mc` panel.
2. **Single L2-Resident LHS Packing + Inline Cache-Resident Epilogue (`avx512LargeMultiFloat32` & `avx2LargeMultiFloat32`)**:
   - Line-by-line profiling of `avx512LargeMatrixSliceFloat32` (`13.16s` total) revealed that:
     - When `numColPanels > 1` (`MLP Intermediate Dense` `[8192, 384] @ [384, 1536]`, 3 column panels of 512), `canCacheAllLHS` was `true` and packing `[8192, 384]` once across all 3 column panels took only **`2.12s`** across all 6 layers.
     - When `numColPanels == 1` (`Q`, `K`, `V`, `Attention Output`, `MLP Output`, where $N = 384 \le 512$), `canCacheAllLHS` was `false`, taking **`11.04s`**—of which `~6.36s` (`3 × 2.12s`) was spent in `FusedAttentionQKVProjection` packing the exact same `[8192, 384]` input 3 separate times!
   - In `avx512LargeMultiFloat32` and `avx2LargeMultiFloat32`, each worker steps through its assigned row range `[LHSRowStart, LHSRowEnd)` in L2-resident sub-chunks (`maxMPanelsPerSubChunk = max(1, min(4, 16/numKPanels))` = `1..4` `Mc` panels, `48–192 KB`):
     1. **Packs `xFlat[subRowStart:subRowEnd]` into `packedLHS` only on the first target (`Q`)** (`lhsPacked = false`), and reuses the L2-resident `packedLHS` for `K` and `V` (`lhsPacked = true`).
     2. **Applies `biasQ`, `biasK`, `biasV` inline via `matmul.ApplyEpilogueRows`** on each target's `[subRowStart, subRowEnd)` output slice immediately after its GEMM sub-chunk completes, while those `32..128` output rows (`48–192 KB`) are still hot in L1/L2 cache—instead of re-reading `3 × 12 MB` from L3/DRAM after a full-matrix barrier.
     3. **Single Parallel Barrier**: Replaces 6 parallel synchronization barriers per attention layer (`3 × SaturateUpTo` + `3 × ParallelizeChunks`) with **1 `SaturateUpTo` barrier**.

3. **Measured `-limit 1000` (10,000 Passages) Results Before vs. After Idea B (AMD Ryzen 9 9950X3D)**:

| Metric | ONNX After Idea A | **ONNX After Idea B** | Change | Notes |
| :--- | ---: | ---: | ---: | :--- |
| **Benchmark Wall Time (10K passages)** | 10.8s | **10.4s** | **-0.4s (-3.7%)** | Down from **11.9s** before Idea A (-12.6% total) |
| **Throughput (items/s)** | 929 items/s | **963 items/s** | **+3.7% (+34 items/s)** | Steady-state sustains **>1,000 passages/s** (80.9K non-pad tok/s)! |
| **Token Throughput (total / non-pad)** | 93.3K / 71.9K tok/s | **96.7K / 74.6K tok/s** | **+3.6% tok/s** | |
| **Benchmark CPU Time** | 229.0s (3m49s) | **225.0s (3m45s)** | **-4.0s CPU time** | `22.6 ms/passage` (down from `26.1 ms/passage`) |
| **`FusedAttentionQKVProjection` LHS Packing** | ~6.36s | **1.88s (`0.74%`)** | **3.38× faster (-4.48s)** | Packed once per row sub-chunk instead of 3× |
| **`FusedAttentionQKVProjection` `ApplyEpilogueRows` (`addBias`)** | ~3.64s | **0.71s (`0.28%`, `0.52s` asm)** | **5.1× faster (-2.93s)** | L1/L2 cache-resident `32..128` rows vs. L3/DRAM re-read |
| **Total `addBiasFloat32AVX512Asm` (all layers)** | 14.57s (5.67%) | **11.50s (4.55%)** | **-3.07s (-21.1%)** | Remaining `10.98s` is in `FusedDense` (`ApplyEpilogue`) |

- **Why `sentence-transformers/all-MiniLM-L6-v2` (Native HF) stayed at `10.9s` (`915 items/s`)**:
  - In `go-huggingface`, the native HuggingFace BERT builder currently emits 3 separate `nn.Dense` layers for `query`, `key`, and `value` rather than calling `attention.MultiHeadAttention` / `FusedAttentionQKVProjection` (whereas `onnx-gomlx` fuses `Q, K, V` into `FusedAttentionQKVProjection`).
  - Moreover, the profile comparison between `avx512LargeMultiFloat32` (`0.71s` for all 18 Q/K/V bias epilogues in ONNX) and `avx512LargeFloat32` (`10.98s` for the 18 `FusedDense` bias epilogues in ONNX, and `16.27s` for all 36 `FusedDense` bias epilogues in HF!) directly reveals a **bonus optimization for `FusedDense` (Idea C)**:
    - In `FusedDense` (`ExecuteWithEpilogue`), `avx512LargeFloat32` still runs the entire GEMM across `SaturateUpTo` and then calls `matmul.ApplyEpilogue` in a **second `ParallelizeChunks` pass** after the barrier (`11.49s` in ONNX, `16.85s` in HF) plus a second `ParallelizeChunks` pass for `Gelu` (`12.84s` in ONNX, `12.36s` in HF), re-reading the entire `[8192, 1536]` (`48 MB`) and `[8192, 384]` (`12 MB`) output matrices from L3/DRAM twice!
    - Moving `ApplyEpilogueRows` inline into `avx512LargeFloat32` (whenever `RHSColStart == 0 && RHSColEnd == rhsCrossSize`, or per `Mc × Nc` tile) and fusing `Bias + Gelu` in a single register pass will eliminate those cold L3/DRAM passes for **both ONNX and Native HF**!



---

### Idea C: Fuse Bias + Activation in a Single Pass in `matmul.ApplyEpilogue` (`dot/matmul/epilogue.go`) — **Medium-High Impact**

- In `compute/internal/gobackend/dot/matmul/epilogue.go` (lines 71–80), when `FusedDense` has both `Bias` and `Activation` (the 6 intermediate MLP layers `[8192, 384] -> [8192, 1536]` with `Bias + GeluExact`/`GeluApprox`):
  1. `ApplyEpilogue` loops row-by-row (`8,192` rows per layer $\times$ 6 layers = `49,152` calls per step).
  2. For each row of 1,536 floats, it first calls `addBias(row, bias)` (which loads `row`, adds `bias`, stores `row` back to cache, and runs `VZEROUPPER`), and then calls `act(row)` (which reloads all 21 `ZMM` constants in `geluExactFloat32AVX512Asm`, loads `row` again, computes GELU, stores `row`, and runs `VZEROUPPER`).
- **Fix**:
  - Add a fused `addBiasAndGeluExactFloat32` / `addBiasAndGeluApproxFloat32` kernel (or pass `bias` pointer + `numRows, rowSize` into the GELU assembly kernel) so:
    1. All 21 `ZMM` constants are loaded **once per tile/chunk** instead of `49,152` times per step.
    2. `x = row[i] + bias[i]` is computed on the fly in registers as `row` is loaded into the GELU pipeline, eliminating the separate `addBias` load/store pass over the 48 MB intermediate activation tensor (`6 × 8192 × 1536 × 4B`).

#### Idea C Implementation & Measured Results (Completed)

Implemented across `compute/internal/gobackend/activations/` (`activations.go`, `avx512/gelu_amd64.s`, `avx512/avx512.go`, `avx2/gelu_amd64.s`, `avx2/avx2.go`), `compute/internal/gobackend/dot/matmul/` (`epilogue.go`, `matmul.go`, `avx512/large.go`, `avx2/large.go`, plus generated `BFloat16`, `Float16`, `Float64` files), `compute/internal/gobackend/fusedops/` (`dense/dense.go`, `activation.go`), and `compute/internal/gobackend/executable.go`:

1. **Fused 2D Tile `Bias + GELU` Assembly Kernels (`BiasTileFn[T]`)**:
   - Added `BiasTileFn[T](out []T, rowStride, numRows, width int, bias []T)`, `RegisterBiasTile[T]`, and `GetBiasTile[T]` in `internal/gobackend/activations/activations.go`.
   - Implemented handwritten AVX-512 and AVX2 assembly kernels in `internal/gobackend/activations/avx512/gelu_amd64.s` and `internal/gobackend/activations/avx2/gelu_amd64.s`:
     - `geluExactBiasTileFloat32AVX512Asm` (`GeluExactBiasTileAVX512`): Loads all 21 `ZMM` constants (`Z11`–`Z31`) **once per 2D tile** (`numRows × width`), then iterates over `numRows` rows (`rowPtr += rowStride * 4`) and `width` columns (`32` floats/iter unrolled $2\times$ in `Z0`..`Z9` + `16`-float and masked tails), adding `bias` directly in registers (`VADDPS 0(SI), Z0, Z0` / `VADDPS 64(SI), Z1, Z1`) before evaluating the rational Padé $\text{erf}(z)$ pipeline. Emits a single `VZEROUPPER` at the end of the entire tile.
     - `geluApproxBiasTileFloat32AVX512Asm` (`GeluApproxBiasTileAVX512`): Loads all 16 `ZMM` constants (`Z16`–`Z31`) once per 2D tile (`64` floats/iter unrolled $4\times$ in `Z0`..`Z15`), adding `bias` in registers during load.
     - `geluExactBiasTileFloat32AVX2Asm` (`GeluExactBiasTileAVX2`) and `geluApproxBiasTileFloat32AVX2Asm` (`GeluApproxBiasTileAVX2`): Corresponding AVX2 2D tile kernels (`16` floats/iter unrolled $2\times$, with overlapped `8`-float tail).
2. **Inline Per-Tile L1/L2 Cache-Resident Epilogue in Large MatMul (`matmul.ApplyEpilogueTile`)**:
   - Added `matmul.ApplyEpilogueTile(outputMatrix []T, rhsCrossSize, rowStart, rowEnd, colStart, colEnd int, epilogue Epilogue[T])` in `internal/gobackend/dot/matmul/epilogue.go`.
   - In `avx512LargeMatrixSlice*` and `avx2LargeMatrixSlice*` (`internal/gobackend/dot/matmul/{avx512,avx2}/large.go`), immediately after the last contracting $K$-panel (`isLastContractingPanel := contractingPanelIdx+contractingPanelWidth >= contractingSize`) writes an `lhsPanelHeight × rhsPanelWidth` (`Mc × Nc`, e.g., `32 × 512` = 64 KB) tile to `outputMatrix`, `matmul.ApplyEpilogueTile` is invoked **while that 64 KB tile is still hot in L1/L2 cache**!
   - Updated `matmul.ExecuteWithEpilogue` (`internal/gobackend/dot/matmul/matmul.go`) so that whenever `epilogue.HasWork()` is true, single-RHS `FusedDense` dispatches through the registered `MultiRHSExecFn` (`avx512LargeMulti*` / `avx2LargeMulti*`), which:
     1. Partitions work across batches and `Mc`-aligned row chunks (`AppendRowWorkItems`), keeping `packedLHS` L2-resident across all `Nc` column panels (so even `N = 384` single-panel projections use small `48–192 KB` L2-resident `packedLHS` buffers instead of allocating `12 MB` per layer!).
     2. Runs `ApplyEpilogueTile` inline per `Mc × Nc` tile with **zero post-GEMM `ParallelizeChunks` barrier** and **zero L3/DRAM re-reads** for both `KnightsAnalytics/all-MiniLM-L6-v2` (ONNX) and `sentence-transformers/all-MiniLM-L6-v2` (Native HF).
3. **Compile-Time Fusion of Single-Consumer `FusedDense(None) + FusedActivation(cfg)` (`fuseSingleConsumerDenseActivations`)**:
   - Both `gomlx/ml/nn/dense.go` (when `VJPRequiresInput()` is true, in case `Gradient` is later called) and `gomlx/ml/zoo/transformer` (`ff = m.dense(...); ff = activation.Apply(...)`) emit `OpTypeFusedDense(ActivationNone)` followed by a separate `OpTypeFusedActivation(cfg)` node in the backend graph.
   - Added `fuseSingleConsumerDenseActivations()` in `internal/gobackend/executable.go` (`newFunctionExecutable`): after `countNodeUsesAndDependents` counts live references reachable from `f.Outputs`, if an `OpTypeFusedDense(ActivationNone)` node has `fe.NumUses == 1` and its sole dependent is `OpTypeFusedActivation(cfg)` (meaning no `FusedActivationVJP` or other node in the compiled function ever reads the pre-activation tensor), `cfg` is absorbed directly into `nodeFusedDense` and `nodeFusedActivation` becomes a zero-cost in-place pass-through (`ActivationNone`).

4. **Measured `-limit 1000` (10,000 Passages) Results Before vs. After Idea C (AMD Ryzen 9 9950X3D)**:

| Metric | ONNX Before Idea C | **ONNX After Idea C** | HF Before Idea C | **HF After Idea C** | Improvement |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Benchmark Wall Time (10K passages)** | 10.4s (11.9s pre-A) | **9.1s** | 10.9s (12.0s pre-A) | **9.4s** | **-1.3s to -1.5s (-12.5% to -13.8%)** |
| **Throughput (items/s)** | 963 items/s | **1.1K items/s** (peak **1.2K/s**) | 915 items/s | **1.1K items/s** (peak **1.1K/s**) | **Both models >1,000 items/s!** |
| **Token Throughput (total / non-pad)** | 96.7K / 74.6K tok/s | **110.5K / 85.2K tok/s** | 92.1K / 71.0K tok/s | **106.4K / 82.0K tok/s** | **+14.3% to +15.5% tok/s** |
| **Benchmark CPU Time** | 225.0s (3m45s) | **200.0s (3m20s)** | 232.0s (3m52s) | **200.0s (3m20s)** | **-25s to -32s CPU time** |
| **Total Epilogue CPU (`addBias` + `Gelu`)** | **24.34s (9.63%)** | **5.06s (2.25%)** | **28.63s (11.14%)** | **4.14s (1.85%)** | **4.81× (ONNX) / 6.92× (HF) faster!** |
| — `addBiasFloat32AVX512Asm` | 11.50s (4.55%) | **<0.9s (<0.4%)** | 16.27s (6.33%) | **<0.9s (<0.4%)** | **>12× – 18× reduction** |
| — `Gelu` / `GeluBiasTile` Asm | 12.84s (5.08%) | **3.93s (1.74%)** | 12.36s (4.81%) | **3.14s (1.40%)** | **3.27× – 3.94× faster (fused w/ bias)** |
| **1. `avx512LargeKernelFloat32Asm` (GEMM)** | 156.92s (62.08%) | **154.16s (68.42%)** | 158.18s (61.50%) | **154.46s (68.85%)** | GEMM microkernel is now **~69%** of total CPU |

- **`-limit 100` (1,000 Passages) Reference**:
  - **ONNX (`KnightsAnalytics/all-MiniLM-L6-v2`)**: **1.3s** (**775 items/s**, **107.6K tok/s**, 27.8s CPU; down from original baseline `2.4s` / `411 items/s` = **+88.6% faster**).
  - **Native HF (`sentence-transformers/all-MiniLM-L6-v2`)**: **1.3s** (**748 items/s**, **103.8K tok/s**, 27.4s CPU; down from original baseline `1.9s` / `520 items/s` = **+43.8% faster**).

---

### Idea D: Fuse Residual `Add` + `FusedLayerNorm` (`fuseSingleConsumerResidualAddIntoLayerNorm`) — **Medium-High Impact (Implemented)**

- **Why Post-Encoder Mean-Pooling was Deprioritized**:
  - After Ideas A, B, and C, profiling `/tmp/cpu_onnx_1000_ideac2.prof` and `/tmp/cpu_hf_1000_ideac2.prof` showed that the entire post-encoder mean-pooling + L2 normalization accounted for only **`0.36s` (`0.16%`)** in ONNX and **`0.95s` (`0.42%`)** in Native HF.
  - Meanwhile, the 12 Transformer residual connections (`simdAddVVFloat32`: **`8.48s` / `3.76%`** in ONNX, **`7.64s` / `3.41%`** in HF) and the 12 immediately following `FusedLayerNorm` ops (`layerNormFloat32AVX512`: **`8.53s` / `3.79%`** in ONNX, **`8.72s` / `3.89%`** in HF) together consumed **`~17.0s` (`7.55%` of total CPU)**!
  - Because `[64, 128, 384]` is `12 MB` (`> L2 cache`), `simdAddVVFloat32` was streaming `2 × 12 MB` in and writing `12 MB` out to L3/DRAM, only for `FusedLayerNorm` to immediately stream that `12 MB` tensor back from L3/DRAM twice (mean/variance pass + normalize/scale pass).

#### Implementation (`compute/internal/gobackend/`)

1. **Internal Residual Input on `FusedLayerNorm` (`fusedops/layernorm.go`, `fusedops/layernorm_simd.go`, `layernorm_arch_trailing.go`)**:
   - Extended `nodeFusedLayerNorm` with `hasResidual bool` and `TryFuseResidualInput() bool` (implementing `gobackend.FusedLayerNormResidualTarget`), without exposing any new op on the public `compute.Backend` API.
   - When `hasResidual` is true, the residual tensor (which has the exact same shape and dtype as `x`) is passed as an extra trailing input (`inputs[len(inputs)-1]`), and `execFusedLayerNorm` / `execFusedLayerNormSIMD` can donate either `inputs[0]` or `inputs[resInputIdx]` as the output buffer in-place when `inputsOwned` is true.
   - Added `LayerNormResidualTrailingArchFn` (`SetLayerNormResidualTrailingArchDispatcher` / `GetLayerNormResidualTrailingArchDispatcher`) in `internal/gobackend/layernorm_arch_trailing.go`, plus an L1-chunked `in + residual` fallback in `layernorm_simd.go` and `layernorm.go` for non-trailing or fallback paths.
2. **Compile-Time Single-Consumer `Add + FusedLayerNorm` Fusion (`executable.go`)**:
   - Added `fuseSingleConsumerResidualAddIntoLayerNorm()` in `newFunctionExecutable`: when `lnNode.Inputs[0]` is a single-consumer `OpTypeAdd` (`fe.NumUses[addIdx] == 1`) whose two inputs `in0, in1` are distinct (`in0 != in1`) and have identical dtype and dimensions (`in0.Shape.EqualDimensions(in1.Shape)`), `lnNode.Inputs` is rewired to `[in0, gamma?, beta?, in1]`, `fe.MaxInputs` is updated, and `countNodeUsesAndDependents` is re-run so the `OpTypeAdd` node gets `fe.NumUses == 0` (dead-code eliminated).
3. **Handwritten AVX-512 & AVX2 Residual LayerNorm Assembly (`fusedops/avx512/layernorm_amd64.s`, `fusedops/avx2/layernorm_amd64.s`)**:
   - Implemented `layerNormResidualFloat32AVX512`, `layerNormResidualFloat64AVX512`, `layerNormResidualFloat32AVX2`, and `layerNormResidualFloat64AVX2` (unrolled $4\times$ per vector width):
     - **Pass 1 (Fused `Add` + Mean)**: Loads `x` (`SI`) and `residual` (`R14`), computes `s = x + residual`, stores `s` directly into `out` (`DI`, which is `1.5 KB` for `normSize = 384` and stays hot in L1 cache, and supports in-place aliasing when `out == x` or `out == residual`), while accumulating `sum(s)` across 4 independent vector accumulators.
     - **Pass 2 (Variance from L1-resident `out`)**: Reads `s` back from L1-resident `out` (`DI`) to compute `sum((s - mean)²)` across 4 independent FMA accumulators (`VFMADD231PS`).
     - **Pass 3 (Normalize + Gamma/Beta in-place in L1-resident `out`)**: Reads `s` from `out` (`DI`), normalizes `(s - mean) * invStd * gamma + beta` (now also unrolled $4\times$ in `pass3_both_loop64`), and writes the final normalized row back to `out` (`DI`).

#### Measured `-limit 1000` (10,000 Passages) Results Before vs. After Idea D (AMD Ryzen 9 9950X3D)

| Metric | ONNX After Idea C | **ONNX After Idea D** | HF After Idea C | **HF After Idea D** | Improvement |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Benchmark Wall Time (10K passages)** | 9.1s (15.4s orig) | **8.8s** | 9.4s (12.0s pre-A) | **9.4s** | **-0.3s wall time (1.75× vs orig)** |
| **Throughput (items/s)** | 1.1K items/s | **1.1K items/s** (steady **1.2K/s**) | 1.1K items/s | **1.1K items/s** | **Sustained 1,200 passages/s (ONNX)** |
| **Token Throughput (total / non-pad)** | 110.5K / 85.2K tok/s | **113.6K / 87.6K tok/s** | 106.4K / 82.0K tok/s | **107.2K / 82.6K tok/s** | **+3.1K tok/s (ONNX)** |
| **Benchmark CPU Time** | 200.0s (3m20s) | **188.0s (3m8s)** | 200.0s (3m20s) | **192.0s (3m12s)** | **-12s (ONNX) / -8s (HF) CPU time!** |
| **CPU Time per Passage** | 20.0 ms/passage | **18.8 ms/passage** | 20.0 ms/passage | **19.3 ms/passage** | **-6.0% CPU/passage (ONNX)** |
| **Residual `Add` + `LayerNorm` CPU** | **17.01s (7.55%)** | **8.65s (4.08%)** | **16.36s (7.30%)** | **10.26s (4.73%)** | **1.97× (ONNX) / 1.60× (HF) faster!** |
| — `simdAddVVFloat32` (Residual `Add`) | 8.48s (3.76%) | **0.00s (0%)** | 7.64s (3.41%) | **0.00s (0%)** | **100% eliminated** |
| — `layerNorm*Float32AVX512` | 8.53s (3.79%) | **8.65s (4.08%)** | 8.72s (3.89%) | **10.26s (4.73%)** | Absorbed full residual `Add` at ~0 extra cost |
| **1. `avx512LargeKernelFloat32Asm` (GEMM)** | 154.16s (68.42%) | **147.34s (69.55%)** | 154.46s (68.85%) | **150.47s (69.35%)** | Pure GEMM microkernel is **~70%** of CPU |

- **`-limit 100` (1,000 Passages) Reference**:
  - **ONNX (`KnightsAnalytics/all-MiniLM-L6-v2`)**: **1.3s** (**767 items/s**, **106.5K tok/s**, **24.6s CPU** / `24.6 ms/passage`; down from `27.8s` CPU after Idea C and original `2.4s` wall / `411 items/s`).
  - **Native HF (`sentence-transformers/all-MiniLM-L6-v2`)**: **1.3s** (**764 items/s**, **106.0K tok/s**, **26.6s CPU** / `26.6 ms/passage`; down from `27.4s` CPU after Idea C and original `1.9s` wall / `520 items/s`).

---

## AVX2-Focused Optimizations (`12th Gen Intel Core i9-12900K`)

After completing the optimizations above on `AMD Ryzen 9 9950X3D` (AVX-512) and `Apple M6` (ARM64 NEON+SME), we profiled and tuned the AVX2-specific kernels on an AVX2-only hybrid P/E-core system (**12th Gen Intel Core i9-12900K**, 16 cores / 24 threads: 8 P-cores with 48 KB L1D / 1.25 MB L2 + 8 E-cores with 32 KB L1D / 2 MB shared L2 per 4-core module). All changes were scoped exclusively to `internal/gobackend/.../avx2` packages so AVX-512 and ARM64 NEON/SME paths remain untouched.

### Summary of AVX2 Optimizations

1. **L1D/L2 Cache-Resident Loop Ordering & 4× Unrolled K-Loop in AVX2 Large GEMM**:
   - `internal/gobackend/dot/matmul/avx2/large_amd64_{float32,bfloat16,float16,float64}.s`:
     - Swapped `loop_rhs` (outer) and `loop_lhs` (inner) across all 4 data types. Because `curPackedLHS` is `Mc × Kc = 32 × 128 × 4B = 16 KB` and one `packedRHS` strip is `Nr × Kc = 16 × 128 × 4B = 8 KB`, their combined `24 KB` footprint stays 100% resident in L1D cache (`32 KB` on E-cores, `48 KB` on P-cores), whereas the previous `loop_lhs` outer / `loop_rhs` inner order streamed the entire `packedRHS` panel on every 4-row strip and overflowed L1D cache.
     - Unrolled the `Float32` `K`-loop by 4 (`k_quad_loop`) with 2-stage ping-pong register preloading (`Y8`–`Y9` and `Y10`–`Y11`) and eliminated integer $\leftrightarrow$ FP domain crossings (`VMOVDQU` $\to$ `VMOVUPS`/`VMOVUPD`).
   - `internal/gobackend/dot/matmul/avx2/avx2.go`:
     - Tuned `AVX2ParamsFloat32.RHSPanelCrossSize` (`Nc`) from `512` to `256` (`128 KB` per `packedRHS` panel), keeping both `packedRHS` and the active `outputMatrix` sub-chunk (`128 KB`) inside per-core L2 cache across contracting panels (`K`) and inline `ApplyEpilogueTile` (`Gelu`) passes.

2. **AVX2 Packing & Multi-Worker Coordination**:
   - `internal/gobackend/dot/matmul/avx2/large.go` (and generated `gen_large_{bf16,f16,f64}.go`):
     - Eliminated redundant per-worker `packedRHS` buffer allocations in `avx2LargeMultiFloat32` when `cachedRHSPanels` or `sharedRHSPanels` covers all targets.
     - Added cooperative parallel packing of `sharedRHSPanels` in `avx2LargeMatrixSliceFloat32` so workers waiting on a panel help pack other unpacked panels instead of spinning.
     - Added `subRowStart..subRowEnd` L2-resident row sub-chunking and contiguous `LHSPanelCrossSize` LHS panel slicing in `avx2LargeFloat32` so worker `packedLHS` panels always fit in cache across column panels.
   - `internal/gobackend/dot/matmul/avx2/pack_amd64_float32.s` & `pack_rhs_amd64.s`:
     - Replaced `VMOVDQU` with `VMOVUPS` to eliminate integer $\leftrightarrow$ FP domain bypass delays during $4 \times 8$ transpose/packing.

3. **Fused SDPA (`D=32`) & LayerNorm AVX2 Assembly**:
   - `internal/gobackend/fusedops/avx2/sdpa_amd64.s`:
     - Interleaved `VMULPS` and `VFMADD231PS` across `Y4`–`Y7` in `stage1_256_loop4` ($Q \cdot K^T$) to hide 4-cycle FMA latency, and unrolled Stage 3 ($P \cdot V$) by 4 keys per iteration (`stage3_256_loop4` and `accum256_stage3_loop4`).
   - `internal/gobackend/fusedops/avx2/layernorm_amd64.s`:
     - Replaced microcoded `VHADDPS` instructions in Pass 1 and Pass 2 horizontal reductions of `layerNormFloat32AVX2` and `layerNormResidualFloat32AVX2` with `VPERMILPS` + `VADDPS`/`VADDSS`, and added 4×-unrolled (32-float) normalization/affine loops (`pass3_both_loop32`, `res_pass3_both_loop32`).

### Benchmark Results (`Intel Core i9-12900K`, AVX2, `parallelism=24`)

#### 1. `Dense` Microbenchmarks (`./gobackend`)

| Benchmark | Baseline (`origin/main`) | Optimized AVX2 | Speedup |
| :--- | ---: | ---: | ---: |
| `Dense/all-MiniLM/416x384x1152/Float32/Fused/None-24` | `612.0 µs/op` | **`525.6 µs/op`** | **-14.1% (+16.4%)** |
| `Dense/all-MiniLM/416x384x1152/Float32/Fused/Gelu-24` | `692.5 µs/op` | **`629.7 µs/op`** | **-9.1% (+10.0%)** |
| `Dense/large/1024x1024x1024/Float32/Fused/None-24` | `2,156.0 µs/op` | **`2,044.3 µs/op`** | **-5.2% (+5.5%)** |
| `Dense/large/1024x1024x1024/Float32/Fused/Gelu-24` | `2,986.9 µs/op` | **`2,852.5 µs/op`** | **-4.5% (+4.7%)** |

#### 2. End-to-End MS Marco Benchmarks (`-bucket=8192 -limit 1000 -num_embedders=1 -warmup -dynamic=true`)

| Model | Metric | Baseline (`origin/main`) | Optimized AVX2 | Improvement |
| :--- | :--- | ---: | ---: | ---: |
| **ONNX (`KnightsAnalytics/all-MiniLM-L6-v2`)** | Total Duration | `20.8s` | **`20.5s`** | **-0.3s** |
| | Throughput (Passages) | `480 items/s` | **`488 items/s`** | **+1.7%** (steady-state up to `523 items/s`) |
| | Throughput (Tokens / Non-Pad) | `48.2K` / `37.1K tok/s` | **`49.0K` / `37.8K tok/s`** | **+1.9%** |
| | Total CPU Time | `6m34s` (`39.5ms/passage`) | **`6m24s` (`38.5ms/passage`)** | **-10s CPU time (-2.5%)** |
| **Native HF (`sentence-transformers/all-MiniLM-L6-v2`)** | Total Duration | `21.5s` | **`21.1s`** | **-0.4s** |
| | Throughput (Passages) | `464 items/s` | **`473 items/s`** | **+1.9%** (steady-state up to `502 items/s`) |
| | Throughput (Tokens / Non-Pad) | `46.6K` / `35.9K tok/s` | **`47.5K` / `36.6K tok/s`** | **+1.9%** |
| | Total CPU Time | `6m38s` (`39.9ms/passage`) | **`6m28s` (`38.9ms/passage`)** | **-10s CPU time (-2.5%)** |



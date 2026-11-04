- 2026-11-04:
  - Added comprehensive ARM64 NEON and Scalable Matrix Extension (`FEAT_SME`) SIMD acceleration across the Go backend (`internal/gobackend/...`, `dtypes/float16`, `dtypes/bfloat16`):
    - **Runtime Feature Detection (`internal/gobackend/simd_arm64*.go`)**: Added ARM64 capability detection and environment controls (`GOMLX_GO_SIMD_NEON`, `GOMLX_GO_SIMD_SVE`, `GOMLX_GO_SIMD_SME`) for `IsNEONAllowed`, `IsARM64FP16Allowed`, `IsARM64BF16Allowed`, `IsARM64DotProdAllowed`, `IsARM64I8MMAllowed`, `IsSVEAllowed`, and `IsSMEAllowed`.
    - **NEON & SME `DotGeneral` / `FusedDense` GEMM (`internal/gobackend/dot/matmul/neon`)**:
      - Registered `"simd:neon"` (`PriorityNEON`) and `"simd:sme"` (`PrioritySME`) with full support for `Float32`, `Float16`, `BFloat16`, and `Float64`.
      - Implemented handwritten Plan9 ARM64 assembly microkernels for large cache-tiled GEMM (`neonLargeKernel*Asm`, $4 \times 16$ / $8 \times 8$ / $8 \times 4$), zero-packing small GEMM (`neonSmallNonTransposed*Asm`, `neonSmallTransposed*Asm`), and vectorized LHS/RHS panel packing (`pack_arm64.s`).
      - Implemented 1-tile ($16 \times 16$) and 4-tile ($32 \times 32$ and $16 \times 64$) SME outer-product (`FMOPA`) microkernels (`sme_arm64.s`) with a two-phase `Z31` hardware canary to guard against `darwin/arm64` `SIGURG` `sigreturn` upper-bit zeroing of streaming `Z` registers.
      - Implemented `CNTVCT_EL0` core-type probing (`smeProbeCoreClockTicks`), global per-cluster SME locks (`globalSMEClusterLocks`), zero-repack strided NEON co-execution (`neonLargeKernelFromSME*`), 3-phase tapered chunking (`AppendWorkItems`), and a lock-free dual-ended work queue (`bounds atomic.Uint64`) so asymmetric CPU clusters (Super, Performance, and Efficiency cores) co-execute SME + NEON without tail stragglers (reaching up to **2.5 TFlops/s** on Apple M6, **20×–22× faster** than `nosimd`).
    - **ARM64 SME + NEON `FusedScaledDotProductAttention` (`internal/gobackend/dot/matmul/neon/sdpa_arm64.go`, `internal/gobackend/fusedops/sdpa_simd.go`)**:
      - Added `DispatchSDPAARM64` (`PriorityArch`) combining a 32-query-row SME $32 \times 32$ outer-product path (`sdpaSME32RowsFloat32`), a 4-query-row zero-packing NEON assembly path (`sdpaNEON4RowsFloat32`), and 2-row/1-row NEON `archsimd` paths, plus a portable `DispatchSDPASIMD` (`PrioritySIMD`) fallback—reducing `SDPA` CPU time on `all-MiniLM-L6-v2` by **16.8×**.
    - **NEON Reductions, Binary Trailing Broadcasts, Epilogues, `Where`, and Half-Precision Conversions**:
      - Added ARM64 NEON assembly and `archsimd` kernels in `internal/gobackend/ops/neon` (`reduce_arm64.s`, `binary_trailing_arm64.go`), `ops/where_arm64.go`, `ops/convertdtype_arm64.go`, `dot/matmul/epilogue_arm64.go`, `dtypes/float16/neon_arm64.s` (`FCVTL`/`FCVTN`), and `dtypes/bfloat16/neon_arm64.s` (`SHLL`/`USHR`).
    - **Scheduler & Transformer Op Optimizations**:
      - Upgraded `Scheduler.ParallelizeChunks` (`internal/gobackend/scheduler/scheduler.go`) to Guided Self-Scheduling (GSS) for asymmetric CPU topologies.
      - Optimized `GeluExactFloat32SIMD` (`activations/simd_full_base.go`), multi-row `FusedSoftmax` (`fusedops/softmax_simd.go`), 4-way unrolled `FusedLayerNorm` (`fusedops/layernorm_simd.go`), and 4D attention head transpositions (`ops/transpose.go`).

- 2026-11-02:
  - Optimized `FusedScaledDotProductAttention` (`internal/gobackend/fusedops/sdpa.go`, `avx2/sdpa.go`, `avx512/sdpa.go`):
    - Added 2D multi-core work partitioning across `(batchSize * numKVHeads, qBlocks)` so models with small `(batch * numKVHeads)` saturate all CPU cores across query sequence blocks.
    - Reduced per-worker `scores` scratch allocation to a single `kvLen` row (`L1`-resident), inverted the value accumulation loop (`for kvIdx { for d ... }`) for contiguous cache-line access and zero-weight skipping, unrolled the scalar Q·K dot product with 4 independent accumulators, and tracked `activeKVLen` for boolean masks.
    - Used `fastmath.Exp32` in the generic `float32` SDPA softmax and fixed `exp512` in `avx512/sdpa.go` (removed the `+0.5` bias before `RoundScaled(0)` so masked `-Inf` positions evaluate to exact `0` and trigger the zero-weight fast skip).
  - Parallelized remaining Go backend unary ops (`Logistic`, `Exp`, `Expm1`, `Log`, `Log1p`, `Rsqrt`, `Sqrt`, `Sin`, `Cos`, `Tanh`, `Erf`, `Sign`, `Abs`, `LogicalNot`, `Ceil`, `Floor`, `Round`, `IsFinite`) in `internal/gobackend/ops/unaryops.go` using `ParallelizeChunks`.
  - Accelerated scalar (`nosimd`) `Logistic`, `Sigmoid`, `Silu`, `Selu`, and `SwiGLU` kernels in `internal/gobackend/ops/unaryops.go` and `internal/gobackend/activations/nosimd.go` using `fastmath.Exp32`.
  - Added safe stack-staged `simdmath.Load*Part` helpers (`support/simdmath/loadpart.go`) and replaced `simd.Load*Part` across Go backend SIMD kernels to work around https://github.com/golang/go/issues/81692 (`simd/archsimd.Load*Part` on `amd64` emitting an unmasked vector load that can `SIGSEGV` when a slice ends near an unmapped page boundary).

- 2026-10-31:
  - Added `CumMax` backend operator to `StandardOps` (with `OpTypeCumMax`, `shapeinference.CumMax`, Go backend implementation in `internal/gobackend/ops/cummax.go`, and compliance tests in `support/backendtest`), reusing `CumSumOptions` (`Exclusive` and `Reverse`).
  - Replaced `workerspool` with a hierarchical 3-level parallel scheduler (`internal/gobackend/scheduler`, documented in `docs/GoBackendScheduler.md`) featuring inter-call, intra-op, and inter-op priority queues, persistent workers, deadlock-free work-helping, dynamic graph execution (`OpsExecutionDynamic`), and $2 \times W$ proportional chunking (#44, #45, #47).
  - Instrumented and optimized Go backend operations for cache-friendly intra-op parallelization (`ParallelizeChunks` / `SaturateUpTo`), including `DotGeneral` (`matmul` 2D split & LHS panel caching), `Slice`, `Transpose`, binary/comparison ops (with 3-part `BroadcastMidRHS`/`BroadcastMidLHS`), unary ops, `Where`, `Concatenate`, `Reduce` (`ReduceSum`/`ReduceMax`/`ReduceMin`), `ReduceWindow`, `ConvGeneral`, `Gather`, `QuantizedDense`, `FusedLayerNorm`, `FusedSoftmax`, `FusedDense` epilogues, and activations (with adaptive `activations.KernelMetadata`) (#45, #46, #47).

- 2026-10-14:
  - Documented `compute.Executable.Finalize()` lifecycle contract: callers must ensure no executions (`Execute()`) are in-flight before calling `Finalize()` to avoid undefined behavior/crashes, or alternatively leave cleanup to the GC or coordinate via higher-level objects (`graph.Exec` / `model.Exec`).

- 2026-09-12:
  - Enabled constant weight panel caching in `DotGeneral` and `FusedDense` for the Go backend:
    - Advertised `PreferConstantsForVariables: true` in `gobackend` capabilities.
    - Optimized large constant deduplication and equality checking in `buffers.go` using fast byte-slice comparison (`bytes.Equal`) and typed slice equality instead of reflection.
    - Added memoized constant subgraph detection (`node.IsConstant()` and `builder.IsConstantNode()`) to identify deterministic constant subgraphs.
    - Added `PackedLHSCache` and `PackedRHSCache` panel caches (`*dot.PackedMatrixCache` with `sync.Once`) to `DotGeneral` and `FusedDense` `NodeData`, preserved across dynamic shape specializations.
    - Added one-time pre-packing and zero-copy panel reuse in AVX-512 and AVX2 GEMM routines (`avx512LargeFloat32`, `avx2LargeFloat32`), supporting both static and batched matrices.
  - Added SIMD vectorization for `FusedAttentionQKVProjection` epilogue (`CopyAndAddBiasFloat32` and `CopyAndAddBiasFloat64`) in `internal/gobackend/dot/matmul` with AVX2 and AVX-512 assembly implementations (`copyAndAddBiasFloat32AVX2Asm`, `copyAndAddBiasFloat32AVX512Asm`).
  - Added dynamic shape support in `FusedAttentionQKVProjection` in `gobackend` using `shapes.MakeDynamic` preserving axis names.
  - Added SIMD-vectorized executor (`PrioritySIMD`) for `OpTypeFusedSoftmax` in `internal/gobackend/fusedops` targeting contiguous trailing axes, utilizing vector max reduction, `simdmath.ExpFloat32`, and vector normalization.
  - Added backend compliance tests for dynamic `FusedAttentionQKVProjection` and remainder-tail `FusedSoftmax` in `support/backendtest`.

- 2026-09-10:
  - Fixed and streamlined `gobackend.DotGeneral` for both static and dynamic shapes:
    - Removed `MergeAxes` calls in `reshapeToSupportedLayout` and `TransposeToLayout`. Contiguous row-major axis groups naturally match GEMM strides, eliminating unnecessary reshape overhead and cleanly supporting multi-batch or multi-contracting dynamic dimensions.
    - Updated `LayoutForDotGeneral` to accept multiple leading batch axes and sequential trailing contracting axes without merging.
    - Simplified `DotGeneral` to directly construct the node with the desired `outputShape`, eliminating redundant intermediate 3D normalized shapes and subsequent reshapes.
    - Documented flat row-major layout invariants across `LayoutForDotGeneral`, `TransposeToLayout`, `reshapeToSupportedLayout`, `DotGeneral`, and `execDotGeneral`.
  - Added unit test `TestDotGeneralDynamic` in `internal/gobackend/dot/dot_test.go` verifying multi-batch dynamic attention score dot products.

- 2026-09-09:
  - Added adaptive SIMD vs. non-SIMD thresholding for `Reduce` operations (`ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProduct`) in the Go backend. When the reduced axis or tensor size is below the architecture-specific threshold, SIMD executors decline with `ErrFallback` to allow the faster scalar implementation to execute.
  - Implemented thread-safe node-level executor caching (`cachedExecutorIdx` on `*Node`), eliminating dispatch loop and fallback check overhead for subsequent runs across both static graphs and dynamic shape specializations.
  - Added `testutil.DurationSampler` supporting reservoir sampling (default 16K samples) to measure median and percentile durations with constant space.
  - Added parameterized benchmark suite `TestFindReduceThresholds` in `internal/gobackend/ops/reduce_bench_test.go` to empirically measure crossover thresholds for AVX2, AVX-512, and ARM64/default architectures.

- 2026-08-28:
  - Fixed `gobackend.DataEqual` to check `reflect.Type.Comparable()` before equality check to support deduplication of nodes with non-comparable structs (containing slices).
  - Fixed `gobackend.DotGeneral` de-normalization reshape to delegate to `DynamicReshape` when handling dynamic matrices/batches.

- 2026-08-27:
  - Added `DynamicIota` and `DynamicPad` backend operations to `DynamicOps` interface (with `OpTypeDynamicIota` and `OpTypeDynamicPad`), implemented in the Go backend (`gobackend`) and ONNX backend (`compute-onnx`), along with `shapeinference` and backend compliance tests.
  - Added `DynamicBroadcastInDim` backend operation to `DynamicOps` interface (with `OpTypeDynamicBroadcastInDim`), implemented in the Go backend (`gobackend`) and ONNX backend (`compute-onnx`), along with `shapeinference` and backend compliance tests.
  - Added `CumSum` backend operator to `StandardOps` supporting `Exclusive` and `Reverse` options (with Go backend implementation).

- 2026-08-02:
  - Fix dynamic output shape materialization in `gobackend` for `binaryOps`, `Concatenate`, and `Reduce` operations so `Buffer.RawShape` is always materialized/concrete.
  - Implement symbolic dynamic axis naming for `Concatenate` (`=term1+term2`) with support for parsing/resolving composite symbolic names in `shapes.Resolve`.
  - Add backend compliance tests for `binaryOps`, `Concatenate`, and `Reduce` with dynamic shapes under `support/backendtest`.

- 2026-07-28:
  - Updated FusedDense to take a `DenseConfig` options parameter, which now includes layout information of the weights.
  - Renamed `AxesLayout` -> `AttentionAxesLayout` (since it's for attention only).

# Initial release

- Moved GoMLX's `backends/simplego` to `gobackend`.
- Removed all dependencies to `stretchr/testify` and `gomlx`, to trim as much as possible external dependencies.
- Moved `gobackend` generic tests to `support/backendtest`, so they can be used by other backends.
- Package `gobackend`:
  - Fixed definition of `Bitcast` when casting to a larger target dtype: the rank is shrinked by 1.
- Package `support`:
  - The following packages were moved from `github.com/gomlx/gomlx/pkg/support/...` to `support/...`: `xslices`, `xsync`, `sets` and `humanize`.

- Package `shapes`: added initial support for dynamic shapes (see `./docs/DynamicShapes.md` for overall idea):
  - Add `Shape.Resolve(AxisBindings) (Shape, error)` method.
  - Add `Shape.IsDynamic()` method.
  - Add `DynamicDim` type.

- New ops:
  - `SchedulingBarrier` and `OptimizationBarrier`, both implemented in the Go backend.
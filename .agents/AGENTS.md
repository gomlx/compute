ROLE: You are a senior software developer, expert in Go and in machine learning.

# Compute Backends API: github.com/gomlx/compute

Package `compute` provides a modular API for defining and executing
multidimensional computation graphs with pluggable backends.

It defines `shapes` (tensor shapes) and `dtypes` (data types) and the top-level
`compute` package defines a `Backend` API (a series of interfaces), that can be
used to define a computation graph, JIT-compile it, transfer buffers (raw
values) to/from the backend, and execute compiled computations.

It powers [GoMLX](https://github.com/gomlx/gomlx), the machine learning
framework for Go, but can be used directly also. With the caveat that the
`compute.Backend` doesn't aim to be ergonomic, but instead "correct" and
"minimal". For a more convenient API for complex computation, and
auto-differentiation, use GoMLX instead.

## File Structure

- `github.com/gomlx/compute`, the root directory: defines the `Backend` and
  related APIs (interfaces).
- `gobackend`: the "go" backend, purely written in Go: so very portable, nothing
  needs installing, but slower. The default backend. This package is just a
  front to the implementation in `./internal/gobackend` and its subdirectories.
  It also serves to link all the sub-packages that need to be included and it
  includes the "TestCompliance" suite of tests (implemented in
  `support/backendtest`).
- `dtypes`: define the supported data types. Lots of utilities to convert dtypes
  to Go types and vice-versa.
  - `dtypes/float16` and `dtypes/bfloat16`: minimal implementations for
    half-precision types in Go.
- `shapes`: define shape for "tensors", also known as multi-dimensional arrays.
- `shapeinferece`: helper that specifies the output of operations given the
  shapes of inputs.
- `notimplemented`: a trivial backend "implementation", that always returns a
  "not implemented" error. A "base class" that can be used by any backend
  implementation.
- `distributed`: types used for distributed execution, modeled after XLA
  "Shardy". Somewhat experimental for now.
- `internal/cmd`: mostly "generators" used to automatically generate code for
  different packages. Referred in `//go:generate ...` in the various packages.
- `internal/fastmath`: float32 math approximations for go-backend kernels
  where small error is acceptable (e.g. softmax, activations).
- `internal/gobackend`: the implementation of the "go" backend. It consists of
  buffer and execution logic, and "registration" of ops during build and
  execution. The implementation of each op is (or is being moved to) in its
  sub-packages `ops`, `dot`, `conv` and `fusedops` (in works).
- `support`: generic support libraries.
  - `support/testutil`: test utilities that can be used by any `compute.Backend`
    implementation to test. 
  - `support/backendtest`: Backend compliance tests and standard benchmarks, that can be run
    against any backend. Call `RunAll(t *testing.T, b compute.Backend)` from your backend tests,
    and `RunAllBenchmarks(b *testing.B, backend compute.Backend)` from your backend benchmarks.
    Tests/benchmarks check backend capabilities and gracefully skip features returning `ErrNotImplemented`.

## Coding Style In GoMLX projects, including this one.

### Minimal External Dependencies (No Testify)

- The `compute` repository is a foundational, low-level package and must have minimal external dependencies.
- **Do NOT add external dependencies**, especially for tests.
- **In particular, do NOT use or import `github.com/stretchr/testify`** (`assert`, `require`, etc.) anywhere in this repository.
- Use standard Go testing primitives (`t.Fatalf`, `t.Errorf`, `math.Abs`, `cmp.Diff`, etc.) or the locally defined `support/testutil` package (which provides `IsEqual`, `IsInDelta`, `IsInRelativeDelta`, etc.).

### Auto-generated code

Files that start with `gen_` are auto-generated and don't include a copyright line
directly -- the copyright line is in their generators.
Many are created with generators included under `internal/cmd/...`, and the generated file 
includes a comment stating which tool was used to generate them.

### Error Handling

All errors should include a stack-trace, using the `github.com/pkg/errors` package.
Whenever printing an error, use `"%+v"` format so the full stack is printed.

### Shapes

- The Shape object should be immutable semantic after creation: function that need to mutate shapes
  should clone first (see Shape.Clone), mutate, and return the updated (and henceforward immutable) shape.
- It's ok to simply copy shapes (shallow copy) if they are not meant to be mutated.
- Shape can have named axes (see `shapes.MakeDynamic`).
- Shape can be dynamic -- dynamic axes are represented by the sentinel value `shapes.DynamicDim` (-1).
  If they are dynamic, they must be named (see `shapes.MakeDynamic`), and the dynamic axes must have a name != "".
  Non-dynamic axes can also be named, but it's not required (the shape.AxisNames can be nil, or their name == "").

### Modern Go Style

- Use the new `for range` format where applicable.
- Use generics where possible.
- Use `slices` and `maps` package for slice operations.
- Look also into `support/xslices` package for more slice and map helper methods.
- Look into `support/xsync` package for more syncronization helpers.
- Look into `support/sets` package for a generic `Set[T]` structure.
- Use iterators (package `iter`) where it makes sense.
- Use the `for range` construct for loops over slices, maps, etc.
- Use `any` instead of `interface{}`.
- Organize tests in hierarchies using `t.Run()` to group related tests.

### Compliance Tests & Benchmarks

- For backend tests and benchmarks that could be used for any backend, write them in `support/backendtest` so other
  backends can benefit.
- We DONT depend on testify or other test libraries: we are trying to minimize external dependencies.
- Use the locally defined `support/testutil` for test utilities for equality (or InDelta or InRelativeDelta comparisons of buffers, etc.).
- **Standard Benchmarks**:
  - `support/backendtest` also provides a standard benchmark suite (call `backendtest.RunAllBenchmarks(b *testing.B, backend compute.Backend)` in `support/backendtest/benchmarks.go`).
  - Includes benchmarks for standard operations (`BenchmarkDotGeneral`, `BenchmarkDense`, `BenchmarkQuantizedDense`, `BenchmarkSoftmax`, `BenchmarkGelu`, `BenchmarkLayerNorm`).
  - **How to run**: `support/backendtest` is backend-agnostic and does not instantiate a backend directly. Run benchmarks via a concrete backend package (e.g. `gobackend`), for example:
    ```bash
    # Run all benchmarks on the Go backend:
    go test -bench=. -benchmem ./gobackend

    # Run only DotGeneral benchmarks without running unit tests:
    go test -run none -bench BenchmarkGoBackend/DotGeneral ./gobackend

    # Run a specific sub-benchmark model (e.g. the "Large" matrix multiplications):
    go test -run none -bench BenchmarkGoBackend/DotGeneral/Large ./gobackend
    ```
  - **Graceful Skips**: If an op, layout, or data type combination returns `compute.ErrNotImplemented`, compliance benchmarks skip cleanly via `b.Skipf(...)`.
  - **Warm-up & Timer**: Benchmarks use Go's `for b.Loop()`. Warm-up iterations (3 runs) are executed *before* `for b.Loop()`, which allows `b.Loop()` to cleanly reset the benchmark timer on its first call and enables compiler loop-variable keep-alive optimizations.
  - **Reported Metrics**: Benchmarks report standard `ns/op` as well as custom metrics via `b.ReportMetric`:
    - `<duration>/op`: execution time per iteration formatted dynamically (e.g. `µs/op`, `ms/op`, `s/op`) via `humanize.Duration` across all benchmarked ops.
    - `GFlops/s`: throughput for `DotGeneral` operations (calculated as $2 \times \text{outputSize} \times \prod \text{contractingDims}$).

### Follow Existing Patterns

Before writing new code, read neighboring files in the same package to understand the established
patterns (buffer management, dtype dispatch, parallelization, etc.). Reuse existing infrastructure
rather than writing ad-hoc implementations. When in doubt, match the style and approach of the
closest existing operation.

### Copyright Notes

Normal code files are prefixed with the following copyright line:

```
// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
```

Auto-generated files don't need a copyright, but should include a comment with the tool use to generate them.

## How to use SIMD in Go with archsimd and simd

Go 1.26 introduced experimental SIMD support, and Go 1.27 updated it with the `simd` (high-level, architecture-independent) and `simd/archsimd` (low-level, direct hardware registers) packages.

### Enabling SIMD

To use Go SIMD, you must:
1. Use a compatible Go version (1.26+ or 1.27+).
2. Set the environment variable `GOEXPERIMENT=simd` during build and test.
3. Use the `//go:build goexperiment.simd` build tag in your files (along with architecture tags like `amd64` when using `archsimd`).

### Common Vector Types in `simd/archsimd`

Vectors are named by their element type and the number of elements. They usually come in three widths:

| Width | Type Examples | Hardware Target (x86) |
| :--- | :--- | :--- |
| **128-bit** | `Float32x4`, `Int32x4`, `Uint16x8`, `Int8x16` | SSE |
| **256-bit** | `Float32x8`, `Int32x8`, `Uint16x16`, `Int8x32` | AVX, AVX2 |
| **512-bit** | `Float32x16`, `Int32x16`, `Uint16x32`, `Int8x64` | AVX-512 |

Other types include `Float64x2/x4/x8`, `Uint64x2/x4/x8`, and corresponding `Mask` types (e.g., `Mask32x16`).

### Go 1.27 Load/Store API Conventions

In Go 1.27 `simd/archsimd`:
- **Slices**: `archsimd.Load<Type>(slice []T)` and `vec.Store(slice []T)` load and store directly to/from slices.
- **Fixed Arrays / Pointers**: `archsimd.Load<Type>Array(ptr *[N]T)` and `vec.StoreArray(ptr *[N]T)` load and store to/from fixed-size array pointers. Use this for raw pointer operations and microkernels (e.g. GEMM).
- **Pairwise operations**: Pairwise reductions use `ConcatAddPairs`, `ConcatSubPairs`, and `ConcatAddPairsGrouped` (renamed from `AddPairs`/`SubPairs`).

### Basic Usage Example

```go
//go:build amd64 && goexperiment.simd

package mypackage

import "simd/archsimd"

// AddSlicesUsingSlices uses slice-based Load/Store.
func AddSlicesUsingSlices(a, b, res []float32) {
    for i := 0; i < len(a); i += 16 {
        va := archsimd.LoadFloat32x16(a[i : i+16])
        vb := archsimd.LoadFloat32x16(b[i : i+16])
        vres := va.Add(vb)
        vres.Store(res[i : i+16])
    }
}

// AddSlicesUsingArrays uses array-pointer-based Load/Store (zero slice header overhead).
func AddSlicesUsingArrays(a, b, res []float32) {
    for i := 0; i < len(a); i += 16 {
        va := archsimd.LoadFloat32x16Array((*[16]float32)(&a[i]))
        vb := archsimd.LoadFloat32x16Array((*[16]float32)(&b[i]))
        vres := va.Add(vb)
        vres.StoreArray((*[16]float32)(&res[i]))
    }
}
```

### Backend Operations (actually part of the `compute.Function` interface)

The `compute` backend defines operations across four interfaces in `./ops*.go`:

#### 1. Standard Operations (`StandardOps` in [ops.go](./ops.go))

| Category | Operations |
| :--- | :--- |
| **Unary Math & Element-wise** | `Abs`, `Ceil`, `Clz`, `Cos`, `Erf`, `Exp`, `Expm1`, `Floor`, `Log`, `Log1p`, `Logistic`, `Neg`, `Round`, `Rsqrt`, `Sign`, `Sin`, `Sqrt`, `Tanh` |
| **Binary Arithmetic** | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Rem`, `Atan2`, `Min`, `Max`, `Clamp` |
| **Comparison & Predicates** | `Equal`, `NotEqual`, `GreaterThan`, `GreaterOrEqual`, `LessThan`, `LessOrEqual`, `EqualTotalOrder`, `NotEqualTotalOrder`, `GreaterThanTotalOrder`, `GreaterOrEqualTotalOrder`, `LessThanTotalOrder`, `LessOrEqualTotalOrder`, `IsFinite`, `IsNaN` |
| **Logical & Bitwise** | `LogicalAnd`, `LogicalOr`, `LogicalXor`, `LogicalNot`, `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`, `BitCount`, `ShiftLeft`, `ShiftRightArithmetic`, `ShiftRightLogical` |
| **Complex Numbers** | `Complex`, `Real`, `Imag`, `Conj` |
| **Shape & Tensor Manipulation** | `Bitcast`, `BroadcastInDim`, `Concatenate`, `ConvertDType`, `DynamicShape`, `DynamicSlice`, `DynamicUpdateSlice`, `Identity`, `Iota`, `Pad`, `Reshape`, `Reverse`, `Slice`, `Transpose`, `Where` |
| **Reductions & Windowing** | `ArgMinMax`, `CumSum`, `ReduceBitwiseAnd`, `ReduceBitwiseOr`, `ReduceBitwiseXor`, `ReduceLogicalAnd`, `ReduceLogicalOr`, `ReduceLogicalXor`, `ReduceMax`, `ReduceMin`, `ReduceProduct`, `ReduceSum`, `ReduceWindow` |
| **Linear Algebra & Convolutions** | `DotGeneral`, `ConvGeneral` |
| **Gather / Scatter** | `Gather`, `ScatterMax`, `ScatterMin`, `ScatterSum`, `SelectAndScatterMax`, `SelectAndScatterMin` |
| **Neural Network Normalization** | `BatchNormForInference`, `BatchNormForTraining`, `BatchNormGradient` |
| **Spectral / Signal** | `FFT` |
| **Random & Barriers** | `RNGBitGenerator`, `OptimizationBarrier`, `SchedulingBarrier` |

#### 2. Dynamic Operations (`DynamicOps` in [ops_dynamic.go](./ops_dynamic.go))

| Category | Operations |
| :--- | :--- |
| **Dynamic Shapes** | `DynamicBroadcastInDim`, `DynamicDimensionSize`, `DynamicIota`, `DynamicPad`, `DynamicReshape` |

#### 3. Fused Operations (`FusedOps` in [ops_fused.go](./ops_fused.go))

| Category | Operations |
| :--- | :--- |
| **Activations & Normalization** | `FusedSoftmax`, `FusedGelu`, `FusedLayerNorm` |
| **Dense & Projections** | `FusedDense`, `FusedAttentionQKVProjection` |
| **Attention** | `FusedScaledDotProductAttention`, `FusedScaledDotProductAttentionVJP` |
| **Quantized Operations** | `QuantizedEmbeddingLookup`, `FusedQuantizedDense` |

#### 4. Collective Operations (`CollectiveOps` in [ops_collective.go](./ops_collective.go))

| Category | Operations |
| :--- | :--- |
| **Distributed / Cross-Device** | `AllReduce` |

### Working with Masks and Bit Manipulation

Masks are returned by comparison operations (e.g., `v.Equal(zero)` returns a `Mask`).
- **Merging**: `res = trueVal.Merge(falseVal, mask)` returns `trueVal` where `mask` is true, and `falseVal` otherwise.
- **Bitmask Vector**: To get a vector where all bits are set based on a mask, use `mask.ToInt32x16().AsUint32x16()`. This is useful for manual bitwise manipulation when `Merge` behavior is complex.

### Architecture Specifics

While `archsimd` is cross-architecture, some operations may only be available on certain platforms or require specific CPU features (like AVX-512). Always check for support using `archsimd.X86.AVX512()` or similar checks in `init()`.

### Gating Tests by Runtime Support (AVX2 / AVX-512)

SIMD and assembly implementations for specific architectures (such as AVX2 or AVX-512) are compiled and linked into test binaries whenever the OS/architecture matches (e.g. `amd64`), but the binary may run on a host machine that does not support those instructions:
- **Always gate tests with runtime checks**: Every test targeting architecture-specific SIMD or assembly (e.g., in `avx2/` and `avx512/` subpackages) **must** verify that the instructions are supported and allowed on the current host, skipping gracefully if not:
  ```go
  // In AVX2 tests:
  if !gobackend.IsAVX2Allowed {
      t.Skip("AVX2 is not supported or allowed on this host")
  }

  // In AVX-512 tests:
  if !gobackend.IsAVX512Allowed {
      t.Skip("AVX-512 is not supported or allowed on this host")
  }
  ```
- **Use `gobackend.IsAVX2Allowed` / `gobackend.IsAVX512Allowed`**: In `internal/gobackend/...`, prefer these helpers over raw `archsimd.X86.AVX2()` / `archsimd.X86.AVX512()`, as they check both hardware CPUID support and environment variables (`GOMLX_GO_SIMD_AVX2`, `GOMLX_GO_SIMD_AVX512`). Outside `internal/gobackend` (e.g. in `dtypes/float16`), check `archsimd.X86.AVX2()` / `archsimd.X86.AVX512()`.
- **Never invoke instructions directly in tests without gating**: Without this check, executing unsupported instructions triggers a `SIGILL` (illegal instruction) crash on machines that lack those CPU extensions (e.g., running AVX-512 code on a CPU without AVX-512, or in virtualized/containerized environments).

## Guidelines for Writing SIMD Assembly (AMD64 / AVX2 & AVX-512)

Handwritten AMD64 assembly is used in performance-critical paths (e.g., GEMM microkernels, packing/transposition, reductions, fused LayerNorm/Softmax) where Go's experimental SIMD (`archsimd`) or compiler code generation introduces register spills, destructive FMA overwrites, or instruction overheads.

When implementing or modifying assembly routines in `./internal/gobackend/...`, follow these strict guidelines.

### 1. Directory and File Organization

Assembly implementations are organized in subdirectories per SIMD architecture:
- Separate subdirectories: `avx2/` and `avx512/` under the relevant package (e.g. `internal/gobackend/ops/{avx2,avx512}`, `internal/gobackend/dot/matmul/{avx2,avx512}`, `internal/gobackend/fusedops/{avx2,avx512}`).
- **Go Declaration (`*_amd64.go`)**: Forward-declare assembly functions with `//go:noescape`, e.g.:
  ```go
  //go:noescape
  func layerNormFloat32AVX2(in, out, gamma, beta unsafe.Pointer, outerSize, normSize int, epsilon float32)
  ```
- **Assembly Implementation (`*_amd64.s` or `*_amd64_<type>.s`)**:
  - Build tag: `//go:build amd64 && goexperiment.simd` (or `//go:build amd64`).
  - Include `#include "textflag.h"`.
  - Function header: `TEXT ·funcName(SB), NOSPLIT, $0-<frameSize>` (e.g. `$0-52`).
  - Registers follow Go planar assembly names: `X0`–`X15` (128-bit), `Y0`–`Y15` (256-bit AVX2), `Z0`–`Z31` (512-bit AVX-512), and mask registers `K0`–`K7`.

### 2. Avoiding AVX $\leftrightarrow$ SSE State Transitions (Intel CPU Penalty)

> [!CAUTION]
> **Never mix legacy non-VEX (SSE) instructions with VEX/EVEX instructions!**

#### The Problem
On Intel CPUs (Skylake through Alder Lake / Raptor Lake), executing a legacy SSE instruction while the upper bits of any YMM/ZMM register are active (dirty) triggers hardware state saving and stalls the execution pipeline for **~70 to 140 CPU cycles per transition**.
- AMD Zen processors do not incur this penalty, so transitions can easily go unnoticed during development on AMD hardware.
- In row-by-row reductions or tight loops (such as LayerNorm, RMSNorm, or Softmax), just 2–4 transitions per row creates a massive **fixed latency penalty** (e.g., an artificial ~18 µs floor in a 100-row LayerNorm).
- Eliminating these transitions yielded a **34× speedup** on small/medium batches (from 18.8 µs down to 550 ns for batch size 16) and improved model training throughput by 20%.

#### Rule A: Use VEX/EVEX Opcode for ALL Floating-Point Instructions
In any routine using YMM (`Y0`–`Y15`) or ZMM (`Z0`–`Z31`) registers, **all instructions**—including scalar math, loads, stores, conversions, and scalar remainder tails—must use their VEX-prefixed (`V...`) counterparts:

| Operation | Legacy SSE (Forbidden in AVX code) | VEX Equivalent (Required) | Notes |
| :--- | :--- | :--- | :--- |
| **Float32 Move** | `MOVSS src, dst` | `VMOVSS src, dst` | Scalar load/store/copy |
| **Float64 Move** | `MOVSD src, dst` | `VMOVSD src, dst` | Scalar load/store/copy |
| **Float32 Add/Sub/Mul** | `ADDSS / SUBSS / MULSS src, dst` | `VADDSS / VSUBSS / VMULSS src2, src1, dst` | 3-operand form |
| **Float64 Add/Sub/Mul** | `ADDSD / SUBSD / MULSD src, dst` | `VADDSD / VSUBSD / VMULSD src2, src1, dst` | 3-operand form |
| **Float32 Division** | `DIVSS src, dst` | `VDIVSS divisor, dividend, dst` | Operand order: computes `dst = dividend / divisor` |
| **Float64 Division** | `DIVSD src, dst` | `VDIVSD divisor, dividend, dst` | Operand order: computes `dst = dividend / divisor` |
| **Square Root** | `SQRTSS / SQRTSD src, dst` | `VSQRTSS / VSQRTSD src, src, dst` | Scalar square root |
| **Int32 $\to$ Float32** | `CVTSL2SS reg, xmm` | `VCVTSI2SSL reg, xmm, dst` | Trailing `L` = 32-bit int register (`R11`) |
| **Int64 $\to$ Float64** | `CVTSQ2SD reg, xmm` | `VCVTSI2SDQ reg, xmm, dst` | Trailing `Q` = 64-bit int register (`R11`) |
| **Zeroing Registers** | `XORPS / XORPD xmm, xmm` | `VXORPS / VXORPD ymm, ymm, ymm` | Clears all 256/512 bits cleanly |

#### Rule B: Always Emit `VZEROUPPER` Before Every `RET`
Standard Go compiler code generation uses legacy SSE instructions for scalar float math. If an assembly routine returns to Go code leaving the upper halves of YMM/ZMM registers dirty, the very next Go-compiled float instruction triggers an AVX $\to$ SSE penalty.
- **Always** emit `VZEROUPPER` immediately before every `RET` instruction in functions using 256-bit or 512-bit registers.

```assembly
done:
	VZEROUPPER
	RET
```

### 3. Register Budgets and Zero-Spill Tiling

- **AVX2 Budget (16 YMM registers: `Y0`–`Y15`)**:
  - Size register tiles so that accumulators and working vectors strictly fit within 16 registers.
  - Example: Small transposed GEMM `Tile4x2` uses 8 accumulators (`Y0`–`Y7`), 4 LHS loads (`Y8`–`Y11`), and 2 RHS loads (`Y12`–`Y13`) = 14 registers. Zero stack spills.
- **AVX-512 Budget (32 ZMM registers: `Z0`–`Z31` + 8 Mask registers: `K0`–`K7`)**:
  - Allows 16 dedicated accumulators (`Z0`–`Z15`) in an $8 \times 32$ tile (2 RHS vectors, 8 LHS broadcasts).
  - Use 2-stage ping-pong register buffers (interleaving Buffer A and Buffer B loads) to hide memory latency.
- **Non-Destructive Accumulation (`VFMADD231PS` vs `VFMADD213PS`)**:
  - In Go's `archsimd`, `MulAdd` emits `VFMADD213PS` which overwrites operand $a$, causing register pressure and spills.
  - In assembly, use non-destructive `VFMADD231PS src1, src2, dst` ($dst = src1 \times src2 + dst$). This keeps accumulators permanently pinned in hardware registers across loop iterations.
- **Hiding FMA Pipeline Latency**:
  - Modern x86 execution cores have a 4-cycle FMA pipeline latency.
  - Using 16 independent accumulator registers ensures each accumulator is addressed once every 16 instructions, completely hiding latency and sustaining near 100% compute port utilization.

### 4. Fast Transposition & Data Packing

- Avoid multi-step shuffle sequences in pure Go. Use direct hardware interleaving unpack instructions:
  - **Float32**: `VUNPCKLPS`, `VUNPCKHPS` followed by 128-bit lane permutes (`VSHUFF32X4` in AVX-512 or `VEXTRACTF128` in AVX2).
  - **Float64**: `VUNPCKLPD`, `VUNPCKHPD`, and `VSHUFF64X2`.
  - **Float16 / BFloat16**: Word unpacks (`VPUNPCKLWD`, `VPUNPCKHWD`) and permutes (`VPERMT2W` / `F16C`).
- For RHS panel copies, unroll by 4 rows with 16 vector registers (`Z0`–`Z15`) to saturate physical memory bandwidth (~70 GB/s on DDR5).
- Strive for 64-byte cache-line alignment to avoid split-cache-line penalties.

### 5. Horizontal Reductions

When reducing vector lanes to a scalar (e.g. in row sums, LayerNorm mean/variance):
- **AVX2 8-lane reduction**:
  ```assembly
  VEXTRACTF128 $1, Y0, X1  // extract upper 128 bits
  VADDPS X1, X0, X0        // 4 floats
  VPERMILPS $0xEE, X0, X1
  VADDPS X1, X0, X0        // 2 floats
  VPERMILPS $0x01, X0, X1
  VADDSS X1, X0, X0        // 1 scalar float in X0
  ```
- **AVX-512 16-lane reduction**:
  ```assembly
  VEXTRACTF32X8 $1, Z0, Y1 // extract upper 256 bits
  VADDPS Y1, Y0, Y0        // 8 floats in Y0
  // followed by AVX2 horizontal reduction steps above
  ```

### 6. SIMD Thresholding and Clean Fallback

SIMD operations carry fixed overheads (register setup, horizontal reductions, mask handling, scalar tails). For small dimensions, scalar loops in CPU registers are often faster:
- **Clean Fallback**: Return `(nil, gobackend.ErrFallback)` before allocating output buffers or consuming memory if dimensions are below architecture thresholds.
- The executor dispatch loop catches `ErrFallback` and falls through to the scalar executor.
- Winning executors are cached per node in `cachedExecutorIdx` for **0 ns dispatch overhead** on subsequent iterations.

### 7. Gating Assembly Tests by Runtime Support

Handwritten assembly functions (in `*_amd64.s`) bypass Go compiler checks and will immediately trigger a `SIGILL` crash if invoked on a CPU lacking the required instruction set:
- Always gate unit tests in `avx2/` and `avx512/` subpackages by calling `if !gobackend.IsAVX2Allowed { t.Skip(...) }` or `if !gobackend.IsAVX512Allowed { t.Skip(...) }` at the top of each test function.
- Never write tests that execute raw assembly kernels without this gate.


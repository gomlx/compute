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
  sub-packages `ops`, `activations`, `dot`, `conv`, `fusedops`, and `scheduler`
  (see `docs/GoBackendScheduler.md`).
- `support`: generic support libraries (including `CPUTime()` for process CPU time measurement, FLOPs calculation, bucketing, etc.).
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
  - Includes benchmarks for standard operations (`BenchmarkDotGeneral`, `BenchmarkConvGeneral`, `BenchmarkDense`, `BenchmarkQuantizedDense`, `BenchmarkSoftmax`, `BenchmarkGelu`, `BenchmarkLayerNorm`).
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

### Keep AI Agent Files Up-to-date

For every new feature, refactoring or API change, update `.agents/AGENTS.md` and the files 
under `.agents/skills/golang-gomlx/` accordingly.

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
| **Reductions & Windowing** | `ArgMinMax`, `CumMax`, `CumSum`, `ReduceBitwiseAnd`, `ReduceBitwiseOr`, `ReduceBitwiseXor`, `ReduceLogicalAnd`, `ReduceLogicalOr`, `ReduceLogicalXor`, `ReduceMax`, `ReduceMin`, `ReduceProduct`, `ReduceSum`, `ReduceWindow` |
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
| **Activations & Normalization** | `FusedSoftmax`, `FusedActivation`, `FusedActivationVJP`, `FusedLayerNorm`, `FusedLayerNormVJP` |
| **Dense & Projections** | `FusedDense`, `FusedDenseVJP`, `FusedAttentionQKVProjection` |
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

### Gating Tests by Runtime Support (AVX2 / AVX-512 / NEON / SVE / SME)

SIMD and assembly implementations for specific architectures (such as AVX2, AVX-512, or ARM64 NEON/SVE/SME) are compiled and linked into test binaries whenever the OS/architecture matches (e.g. `amd64` or `arm64`), but the binary may run on a host machine that does not support those instructions or where the user disabled them via environment variables:
- **Always gate tests with runtime checks**: Every test targeting architecture-specific SIMD or assembly (e.g., in `avx2/`, `avx512/`, `neon/`, `sve/`, or `sme/` subpackages) **must** verify that the instructions are supported and allowed on the current host, skipping gracefully if not:
  ```go
  // In AVX2 tests:
  if !gobackend.IsAVX2Allowed {
      t.Skip("AVX2 is not supported or allowed on this host")
  }

  // In AVX-512 tests:
  if !gobackend.IsAVX512Allowed {
      t.Skip("AVX-512 is not supported or allowed on this host")
  }

  // In ARM64 NEON tests:
  if !gobackend.IsNEONAllowed {
      t.Skip("ARM64 NEON is not supported or allowed on this host")
  }
  ```
- **Use `gobackend.Is*Allowed` helpers**: In `internal/gobackend/...`, prefer `gobackend.IsAVX2Allowed`, `gobackend.IsAVX512Allowed`, `gobackend.IsNEONAllowed`, `gobackend.IsARM64FP16Allowed`, `gobackend.IsARM64BF16Allowed`, `gobackend.IsARM64DotProdAllowed`, `gobackend.IsARM64I8MMAllowed`, `gobackend.IsSVEAllowed`, and `gobackend.IsSMEAllowed`, as they check both hardware support and environment variables (`GOMLX_GO_SIMD_AVX2`, `GOMLX_GO_SIMD_AVX512`, `GOMLX_GO_SIMD_NEON`, `GOMLX_GO_SIMD_SVE`, `GOMLX_GO_SIMD_SME`).
- **Never invoke instructions directly in tests without gating**: Without this check, executing unsupported instructions triggers a `SIGILL` (illegal instruction) crash on machines that lack those CPU extensions.

## Guidelines for Writing SIMD Assembly (AMD64 / AVX2 & AVX-512)

Handwritten AMD64 assembly is used in performance-critical paths (e.g., GEMM microkernels, packing/transposition, reductions, fused LayerNorm/Softmax) where Go's experimental SIMD (`archsimd`) or compiler code generation introduces register spills, destructive FMA overwrites, or instruction overheads.

When implementing or modifying assembly routines in `./internal/gobackend/...`, follow these strict guidelines.

### 1. Directory and File Organization

All code specific to a SIMD architecture—including Go wrappers, assembly routines, and unit/benchmark tests (`*_test.go`)—must be placed in the corresponding architecture subdirectory, never in the parent generic package directory:
- **Separate subdirectories**: `avx2/`, `avx512/`, `neon/` (and `sve/`, `sme/` where applicable) under the relevant package (e.g. `internal/gobackend/ops/{avx2,avx512,neon}`, `internal/gobackend/activations/{avx2,avx512,neon}`, `internal/gobackend/dot/matmul/{avx2,avx512,neon}`, `internal/gobackend/fusedops/{avx2,avx512}`).
- **Registration & Default Imports**: Subpackages register their implementations in `init()` (using `gobackend.RegisterPriority`) and are blank-imported in `internal/gobackend/defaultpkgs/defaultpkgs_<arch>_simd.go`. Include an untagged `doc.go` in each subdirectory so the package remains valid across all build configurations.
- **Tests in Architecture Subdirectories**:
  - Place architecture-specific unit tests and microbenchmarks inside the subdirectory (e.g. `avx2/avx2_test.go`, `avx512/avx512_test.go`, `neon/neon_test.go`).
  - Rather than reimplementing mathematical formulas in architecture tests, import the parent package and compare the SIMD kernel outputs directly against the baseline `NoSIMD` reference implementation (generic operation semantics belong in `support/backendtest`).
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
  - Example: 1D/2D `ConvGeneral` (`internal/gobackend/ops/avx2/conv_amd64.s`) uses a $4 \times 16$ tile (`ConvAVX2Kernel4x16Float32Asm`, `ConvKernelGradAVX2_4x16Float32Asm`) with 8 accumulators (`Y0`–`Y7`), 4 2-stage ping-pong RHS registers (`Y8`–`Y11`), and 1 broadcast scratch register (`Y12`), reading directly from 4 row pointers via `VBROADCASTSS` with zero LHS transposition overhead. For small `InChannels` (e.g. `Cin == 1` in VJP wrt kernel), `ConvKernelGradAVX2_1x16Float32Asm` unrolls $4\times$ across 4 independent accumulator pairs (`Y0`–`Y7`) to hide 4-cycle FMA latency.
- **AVX-512 Budget (32 ZMM registers: `Z0`–`Z31` + 8 Mask registers: `K0`–`K7`)**:
  - Allows 16 dedicated accumulators (`Z0`–`Z15`) in an $8 \times 32$ tile (2 RHS vectors, 8 LHS broadcasts), as used in GEMM and 1D/2D `ConvGeneral` (`ConvAVX512Kernel8x32Float32Asm`, `ConvKernelGradAVX512_8x32Float32Asm` in `internal/gobackend/ops/avx512/conv_amd64.s`), plus $4 \times 32$ and $1 \times 32$ multi-bank unrolled variants (`ConvAVX512Kernel4x32Float32Asm`, `ConvKernelGradAVX512_4x32Float32Asm`, `ConvKernelGradAVX512_1x32Float32Asm`).
  - Use 2-stage ping-pong register buffers (interleaving Buffer A and Buffer B loads) to hide memory latency.
- **Non-Destructive Accumulation (`VFMADD231PS` vs `VFMADD213PS`)**:
  - In Go's `archsimd`, `MulAdd` emits `VFMADD213PS` which overwrites operand $a$, causing register pressure and spills.
  - In assembly, use non-destructive `VFMADD231PS src1, src2, dst` ($dst = src1 \times src2 + dst$). This keeps accumulators permanently pinned in hardware registers across loop iterations.
- **Hiding FMA Pipeline Latency**:
  - Modern x86 execution cores have a 4-cycle FMA pipeline latency.
  - Using 16 independent accumulator registers ensures each accumulator is addressed once every 16 instructions, completely hiding latency and sustaining near 100% compute port utilization.

### 4. Fast Transposition, Data Packing & L1D/L2 Cache Residency

- Avoid multi-step shuffle sequences in pure Go. Use direct hardware interleaving unpack instructions:
  - **Float32**: `VUNPCKLPS`, `VUNPCKHPS` followed by 128-bit lane permutes (`VSHUFF32X4` in AVX-512 or `VEXTRACTF128` / `VPERM2F128` in AVX2).
  - **Float64**: `VUNPCKLPD`, `VUNPCKHPD`, and `VSHUFF64X2` / `VPERM2F128`.
  - **Float16 / BFloat16**: Word unpacks (`VPUNPCKLWD`, `VPUNPCKHWD`) and permutes (`VPERMT2W` / `F16C`).
- **Avoid Integer $\leftrightarrow$ FP SIMD Domain Crossings (`VMOVUPS`/`VMOVUPD` vs `VMOVDQU`)**:
  - On Intel CPUs, loading or storing floating-point vectors with integer-domain `VMOVDQU` before/after FP-domain `VFMADD231PS`, `VADDPS`, `VUNPCKLPS`, or `VPERM2F128` incurs a 1–2 cycle bypass delay between the integer and floating-point execution domains. Use `VMOVUPS` (`Float32`) and `VMOVUPD` (`Float64`) consistently in FP microkernels and packing routines.
- **L1D/L2 Cache-Resident Loop Ordering in AVX2 Large GEMM (`internal/gobackend/dot/matmul/avx2/large_amd64_*.s`)**:
  - In `avx2LargeKernel*Asm`, iterate `loop_rhs` (columns `0..Nc` in steps of `Nr = 16`) as the **outer** loop and `loop_lhs` (rows `0..Mc` in steps of `Mr = 4`) as the **inner** loop.
  - Because `packedLHS` is `Mc × Kc = 32 × 128 × 4B = 16 KB` and one `packedRHS` strip is `Nr × Kc = 16 × 128 × 4B = 8 KB`, their combined `24 KB` footprint fits 100% inside a 32 KB L1D cache (including Intel Alder Lake / Raptor Lake E-cores), whereas putting `loop_rhs` inside `loop_lhs` streams the entire `128–256 KB` `packedRHS` panel on every 4-row strip and overflows L1D cache.
  - Sizing `AVX2ParamsFloat32.RHSPanelCrossSize` (`Nc`) to `256` (`128 KB` per `packedRHS` panel) keeps both `packedRHS` and the active `outputMatrix` sub-chunk (`128 KB`) inside per-core L2 caches across contracting panels (`K`) and inline `ApplyEpilogueTile` activations.
- For RHS panel copies, unroll by 4 rows with vector registers to saturate physical memory bandwidth (~70 GB/s on DDR5).
- Strive for 64-byte cache-line alignment to avoid split-cache-line penalties.

### 5. Horizontal Reductions

When reducing vector lanes to a scalar (e.g. in row sums, LayerNorm mean/variance):
- **Avoid `VHADDPS` / `VHADDPD`**: Horizontal add instructions are microcoded on both Intel and AMD CPUs (3 uops, 5–7 cycles latency, port contention). Always use `VPERMILPS` / `VSHUFPS` + `VADDPS` / `VADDSS` instead.
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

Handwritten assembly functions (in `*_amd64.s` or `*_arm64.s`) bypass Go compiler checks and will immediately trigger a `SIGILL` crash if invoked on a CPU lacking the required instruction set:
- Always gate unit tests in `avx2/`, `avx512/`, `neon/`, `sve/`, and `sme/` subpackages by calling `if !gobackend.IsAVX2Allowed { t.Skip(...) }`, `if !gobackend.IsAVX512Allowed { t.Skip(...) }`, `if !gobackend.IsNEONAllowed { t.Skip(...) }`, etc., at the top of each test function.
- Never write tests that execute raw assembly kernels without this gate.

## ARM64 SIMD Support (NEON, SVE/SVE2 & SME)

ARM64 SIMD acceleration follows the same multi-tier architecture as AMD64:
1. **Portable `simd` / cross-arch `simd/archsimd` tier**: Shared 128-bit SIMD loops in `internal/gobackend/ops/*_simd.go` (`//go:build (amd64 || arm64) && goexperiment.simd`), `internal/fastmath`, and `internal/gobackend/activations` compile directly to ARM64 NEON via Go 1.27's `simd/archsimd` 128-bit types (`Float32x4`, `Float64x2`, `Int32x4`, `Uint32x4`, `Int16x8`, `Uint16x8`, `Int8x16`, `Uint8x16`, `Int64x2`, `Uint64x2`).
2. **Architecture-specific packages (`neon/`, and optionally `sve/` / `sme/`)**: Dedicated subdirectories under `ops/`, `fusedops/`, and `dot/matmul/` that register at explicit priority levels and fall back cleanly (`gobackend.ErrFallback` or `false` on sub-dispatcher hooks) when dimensions are below thresholds or a layout/dtype is unhandled.

### 1. Registration Priority Ladder & Sub-Dispatcher Chaining

Operations and sub-dispatchers (`SetBinaryTrailingArchDispatcher`, `SetReduceTrailingSumArchDispatcher`, `SetReduceLeadingSumArchDispatcher`, `SetLayerNormTrailingArchDispatcher`, `SetSDPAArchDispatcher`) use the following priority ladder in `internal/gobackend/exec.go`:
- `PriorityGeneric = 0` (scalar fallback)
- `PriorityTyped = 1` (typed Go loops)
- `PrioritySIMD = 6` (portable Go `simd` / `archsimd` implementation)
- `PriorityArch = 10` (`PriorityNEON = PriorityArch`, `PriorityAVX2 = PriorityArch`)
- `PriorityAVX512 = PriorityArch + 1`, `PrioritySVE = PriorityArch + 1`
- `PrioritySME = PriorityArch + 2`
- `PriorityUser = 100`

Sub-dispatchers maintain a priority-sorted slice of registered hooks: if a higher-priority hook (e.g. `PrioritySME` or `PrioritySVE`) returns `false` for a given tensor shape or dtype, the dispatcher automatically chains to the next lower-priority hook (e.g. `PriorityNEON`) before falling back to generic SIMD or scalar code.

### 2. Runtime CPU Feature Detection & Environment Variables

In `internal/gobackend/simd_arm64.go` (and OS-specific helpers `simd_arm64_darwin.go` via `syscall.SysctlUint32` and `simd_arm64_linux.go` via `/proc/self/auxv` `AT_HWCAP`/`AT_HWCAP2`), the backend exposes:
- `IsNEONAllowed`: Baseline ARM64 AdvSIMD (gated by `GOMLX_GO_SIMD` and `GOMLX_GO_SIMD_NEON`).
- `IsARM64FP16Allowed`: `FEAT_FP16` half-precision vector arithmetic & conversions (`FCVTL` / `FCVTN`).
- `IsARM64BF16Allowed`: `FEAT_BF16` (`BFDOT`, `BFMMLA`, `BFCVTN`).
- `IsARM64DotProdAllowed`: `FEAT_DotProd` (`SDOT`, `UDOT` for Int8/Uint8 dot products).
- `IsARM64I8MMAllowed`: `FEAT_I8MM` (`SMMLA`, `UMMLA`, `USMMLA` 8-bit matrix multiply-accumulate).
- `IsSVEAllowed`: `FEAT_SVE` / `FEAT_SVE2` (gated by `GOMLX_GO_SIMD_SVE`, for AWS Graviton3/4, Grace, Neoverse).
- `IsSMEAllowed`: `FEAT_SME` / `FEAT_SME2` (gated by `GOMLX_GO_SIMD_SME`, for Apple M4+).
- `ActiveArchitectures`: Global `[]string` populated with `<GOOS>/<GOARCH>`, plus `"SIMD"` and any active specialized SIMD architectures (`"AVX2"`, `"AVX512"`, `"NEON"`, `"SVE"`, `"SME"`), rendered in `Backend.Description()`.

### 3. Phased Implementation Roadmap

- **Detection & Priority Infrastructure**: CPU feature detection (`IsNEONAllowed`, `IsARM64FP16Allowed`, `IsARM64BF16Allowed`, `IsARM64DotProdAllowed`, `IsARM64I8MMAllowed`, `IsSVEAllowed`, `IsSMEAllowed`), environment variables (`GOMLX_GO_SIMD_NEON`, `GOMLX_GO_SIMD_SVE`, `GOMLX_GO_SIMD_SME`), priority constants, and chained sub-dispatchers.
- **Phase 1 — Half-Precision Conversions & Reduction Thresholds**:
  - `dtypes/float16/neon.go` (`Float16x8` load/store/conversions to pair of `archsimd.Float32x4` and `ToFloat32SIMD`/`FromFloat32SIMD` slice helpers).
  - `dtypes/bfloat16/neon.go` (`BFloat16x8` load/store/conversions to pair of `archsimd.Float32x4` using `ExtendLo4ToUint32` / `HiToLo` and IEEE-754 round-to-nearest-even narrowing).
  - `internal/gobackend/ops/reduce_thresholds_arm64.go` (ARM64 NEON empirically tuned SIMD reduction thresholds).
- **Phase 4 — Standard Operations (`ops/neon`, `where_arm64`, `epilogue_arm64`)**:
  - `internal/gobackend/ops/neon`: NEON trailing-axis sum (`DispatchTrailingSumNEON`), leading-axis sum (`DispatchLeadingSumNEON`), and trailing-broadcast binary ops (`DispatchBinaryTrailingNEON`) across `Float32`, `Float64`, `Float16`, `BFloat16`, and integer types.
  - `internal/gobackend/ops/neon/conv_arm64.go` and `internal/gobackend/ops/neon/conv_arm64.s`: 1D and 2D `ChannelsLast` `ConvGeneral` acceleration for both forward/input-VJP (`ConvNEONKernel4x16Float32Asm`, `ConvNEONDirect2DKernel4x16Float32Asm` zero-`im2col` direct 2D kernel, `ConvSMEKernel32x32Float32Asm`, `ConvSMEKernel16x16Float32Asm`, `ConvPackLHS8StridedFloat32Asm` direct-from-input SME LHS packing, and `SMEAcquireClusterWithLimit(2, 1)` dual-worker fast-cluster pipelining) and zero-packing outer-product kernel-VJP (`ConvKernelGradNEON4x16Float32Asm`, `ConvKernelGradNEON1x16Float32Asm`, `ConvKernelGradSME32x32Float32Asm` with 2D `(Cin x Cout)` panel tiling and unified `(BatchSize, OutH)` reduction slicing), plus $1\times 1$ unit-stride zero-copy GEMM fast path.
  - `internal/gobackend/ops/where_arm64.go`: Vectorized `whereFloat32Arch` (`VV`, `VS`, `SV`, `SS` modes).
  - `internal/gobackend/dot/matmul/epilogue_arm64.go`: Vectorized `addBiasFloat32Arch` and `copyAndAddBiasFloat32Arch`.
- **Phase 5 — ARM64 Assembly Guidelines**: Documented below.
- **Future Phases (Fused Ops & `DotGeneral` / `matmul`)**:
  - `internal/gobackend/fusedops/neon`: Fused LayerNorm, Softmax, GeLU, SDPA (`D=32, 64, 128`), and QuantizedDense (`FEAT_DotProd` `SDOT`/`UDOT`).
  - `internal/gobackend/dot/matmul/neon` (and optional `sve`/`sme`): Dedicated register-tiled GEMM microkernels (e.g. $8 \times 12$ `Float32x4` tile utilizing 24 of the 32 NEON vector registers `V0`–`V31`), benchmarked across small and large matrix regimes.

## Guidelines for Writing SIMD Assembly (ARM64 / NEON, SVE & SME)

When implementing or modifying ARM64 assembly routines in `./internal/gobackend/...` or `./dtypes/...`, follow these guidelines:

### 1. Directory and File Organization
- **Separate subdirectories**: Place all ARM64-specific SIMD code, assembly files, and unit/benchmark tests (`*_test.go`) in `neon/` (and `sve/`, `sme/` where applicable) under the relevant package (e.g. `internal/gobackend/ops/neon`, `internal/gobackend/activations/neon`, `internal/gobackend/fusedops/neon`, `internal/gobackend/dot/matmul/neon`), and blank-import the package in `internal/gobackend/defaultpkgs/defaultpkgs_arm64_simd.go`.
- **Go Declaration (`*_arm64.go`)**: Forward-declare assembly functions with `//go:noescape`.
- **Assembly Implementation (`*_arm64.s`)**:
  - Build tag: `//go:build arm64 && goexperiment.simd` (or `//go:build arm64`).
  - Include `#include "textflag.h"`.
  - Function header: `TEXT ·funcName(SB), NOSPLIT, $0-<frameSize>`.

### 2. ARM64 Register File & Tiling Budget (`V0`–`V31`)
- **32 128-bit Vector Registers (`V0`–`V31`)**:
  - Unlike AVX2 (which has only 16 YMM registers), ARM64 NEON always provides **32 128-bit registers** (`V0`–`V31`), matching AVX-512's register count!
  - In `Float32` GEMM microkernels, an **$8 \times 12$ tile** (8 rows $\times$ 3 vectors of 4 floats) uses **24 accumulator registers (`V0`–`V23`)**, **3 RHS vector registers (`V24`–`V26`)**, and **2–4 LHS registers (`V28`–`V31`)** = 29–31 registers with **zero stack spills**.
- **General-Purpose Registers (`R0`–`R27`, `R29`, `R30`)**:
  - `R0`–`R17` are caller-saved scratch registers.
  - **Never clobber `R18`** (platform reserved register on macOS/iOS; modifying `R18` causes immediate crashes on Darwin) or **`R28`** (Go's current goroutine pointer `g`).

### 3. Non-Destructive FMA and Lane-Indexed Broadcasts
- **Non-Destructive `VFMLA`**:
  - ARM64 `VFMLA Vsrc1.S4, Vsrc2.S4, Vdst.S4` accumulates `Vdst += Vsrc1 * Vsrc2` without overwriting `Vsrc1` or `Vsrc2`.
- **Element-Indexed FMA (Zero Broadcast Instruction Overhead)**:
  - NEON supports multiplying a vector directly by a scalar lane of another vector register (`FMLA Vd.4S, Vn.4S, Vm.S[lane]`).
  - Instead of executing separate broadcast/splat (`VDUP`) instructions for each LHS element, load 4 contiguous LHS elements into `V28.S4` once and issue 4 indexed FMAs against lanes `0, 1, 2, 3`, eliminating 75% of LHS load/broadcast instructions.

### 4. No AVX $\leftrightarrow$ SSE Transition Penalty on NEON (and `SMSTOP` + Signal Canary for SME)
- **NEON (`V0`–`V31`)**: Scalar floating-point (`F0`–`F31` / `S0`–`S31` / `D0`–`D31`) and 128-bit NEON vector instructions share the exact same architectural register file without state transition penalties. No `VZEROUPPER` equivalent is required when returning from NEON routines.
- **SME (`SMSTART` / `SMSTOP` and `ZA` Multi-Tile Latency Hiding)**:
  - If an SME routine enters Streaming SVE / ZA-array mode via `SMSTART`, it **must** execute `SMSTOP` before `RET` (as required by the platform ABI and Go runtime).
  - On 512-bit SVL implementations (e.g. Apple M4/M5/M6), the 4 KB `ZA` matrix array is partitioned into four $16 \times 16$ `.S` tiles (`ZA0.S`–`ZA3.S`). Because `ZA` tile accumulation has a ~4-cycle latency, single-tile $16 \times 16$ kernels stall on back-to-back `FMOPA ZA0.S` instructions; use 4-tile $32 \times 32$ ($2 \times 2$ tiles `ZA0..3.S`) or $16 \times 64$ ($1 \times 4$ tiles `ZA0..3.S`) kernels for full speed (>2.1 TFlops/s per core), stepping down to $16 \times 16$ only when $M < 32$ or $N < 32$.
  - **`darwin/arm64` Async Signal (`SIGURG`) Safety in Streaming Mode**: On macOS (`darwin/arm64`), the POSIX signal context (`_STRUCT_ARM_NEON_STATE64`) only saves/restores the lower 128 bits (`Q0`–`Q31`) of `Z0`–`Z31`. When Go delivers an async `SIGURG` signal during `SMSTART`, `ZA` is preserved, but `sigreturn` zeroes bits `[511:128]` (lanes `4..15`) of all `Z` registers. Always protect SME streaming loops with a `Z31` upper-lane canary (`MOV Z31.S, #1` before the loop, `LASTB W11, P7, Z31.S` + `CBZ R11, retry` before writing `ZA` to memory; see `internal/gobackend/dot/matmul/neon/sme_arm64.s`).
  - **Asymmetric Cluster-Shared SME Units & Global Hybrid SME+NEON Co-Execution**: On Apple M-series SoCs (e.g. M6 with 2 Super, 4 Performance, and 6 Efficiency cores), each of the 3 CPU clusters shares a single physical SME unit (~2.0 TFlops/s on Super/Performance, ~0.9 TFlops/s on Efficiency), whereas 128-bit NEON pipelines are private per core. In multi-worker SME operations (`internal/gobackend/dot/matmul/neon/sme_arm64.go`), pin workers to OS threads (`runtime.LockOSThread()`), probe the thread's cluster once at the start of the op via `CNTVCT_EL0` (`smeProbeCoreClockTicks` with 1 retry and adjacent fast-cluster thermal fallback), and acquire a global per-cluster SME lock (`globalSMEClusterLocks`); workers whose cluster SME unit is already occupied execute strided per-core NEON microkernels (`neonLargeKernelFromSME*`) directly on the shared SME-packed panels.

### 5. Horizontal Reductions on NEON
- **Integer Horizontal Sum (`VADDV`)**:
  - Single-instruction across-vector reduction for `Int8x16`, `Uint8x16`, `Int16x8`, `Uint16x8`, `Int32x4`, `Uint32x4` (exposed in Go 1.27 `archsimd` as `.ReduceSum()`).
- **Floating-Point Pairwise Reduction (`VFADDP`)**:
  - For IEEE-754 compliant ordered reduction or fast tree reduction of `Float32x4`, use `FADDP` (`ConcatAddPairs` in `archsimd` or `FADDP V0.S4, V0.S4, V0.S4` followed by `FADDP` to scalar).
  - Note: When reducing multiple rows simultaneously (e.g. 4 rows in `V0`–`V3`), interleave pairwise reductions across rows (`VFADDP V0.S4, V1.S4, V0.S4`; `VFADDP V2.S4, V3.S4, V2.S4`; `VFADDP V0.S4, V2.S4, V0.S4`) to produce 4 row sums in a single `Float32x4` vector and store all 4 sums with one `VST1`!
- **Horizontal Max/Min (`FMAXNMV` / `FMINNMV` / `FMAXV` / `FMINV`)**:
  - Exposed in `archsimd` as `.ReduceMax()` and `.ReduceMin()`.

### 6. Half-Precision (`Float16` / `BFloat16`) & Int8 Acceleration
- **`Float16` (`FEAT_FP16`)**:
  - `FCVTL` / `FCVTL2` (4 or 8 `Float16` $\to$ `Float32x4`) and `FCVTN` / `FCVTN2` (`Float32x4` $\to$ `Float16`) convert 8 half-precision floats in 2 instructions (~2 cycles total).
- **`BFloat16`**:
  - Widening `BFloat16x8` $\to$ two `Float32x4` vectors uses `UXTL` / `UXTL2` + left shift by 16 (`SHL $16`), or `SSHLL` / `USHLL` (exposed in `archsimd` via `ExtendLo4ToUint32()` and `HiToLo().ExtendLo4ToUint32()`).
- **Encoding Newer ARMv8.2+ and Element-Indexed Instructions in Plan9 Assembly**:
  - Where Go's Plan9 assembler does not yet provide mnemonics for element-indexed `FMLA` or newer ARMv8.2+/v8.6+ instructions (`FCVTL`, `FCVTL2`, `SHLL`, `SHLL2`, `SDOT`, `UDOT`, `BFDOT`, `SMMLA`), encode them using `WORD $0x...` macros with clear comments showing the exact ARM architectural mnemonic and register operands:
    ```c
    // FMLA Vd.4S, Vn.4S, Vm.S[lane] (sz=10 at bits [23:22]):
    #define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
        WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))
    // FMLA Vd.2D, Vn.2D, Vm.D[lane] (sz=11 at bits [23:22]):
    #define FMLA_2D_LANE(Vd, Vn, Vm, lane) \
        WORD $(0x4FC01000 | (((lane)&1)<<11) | ((Vm)<<16) | ((Vn)<<5) | (Vd))
    // FCVTL Vd.4S, Vn.4H and FCVTL2 Vd.4S, Vn.8H:
    #define FCVTL_4S_4H(Vd, Vn)  WORD $(0x0E217800 | ((Vn)<<5) | (Vd))
    #define FCVTL2_4S_8H(Vd, Vn) WORD $(0x4E217800 | ((Vn)<<5) | (Vd))
    // SHLL Vd.4S, Vn.4H, #16 and SHLL2 Vd.4S, Vn.8H, #16 (BFloat16 -> Float32 widening):
    #define SHLL_4S_4H_16(Vd, Vn)  WORD $(0x2E613800 | ((Vn)<<5) | (Vd))
    #define SHLL2_4S_8H_16(Vd, Vn) WORD $(0x6E613800 | ((Vn)<<5) | (Vd))
    ```

## Guidelines for Fused Kernels and Intra-Op Parallelization

Lessons learned from optimizing high-performance fused operations (such as `FusedScaledDotProductAttention`, `FusedLayerNorm`, and `FusedDense`):

### 1. Zero-Transpose Strided Memory Access
Complex multi-dimensional operations frequently encounter multiple tensor layout conventions across frameworks (e.g. `[B, S, H, D]` vs `[B, H, S, D]` in attention):
- **The Pitfall**: Decomposing operations or permuting tensors using explicit transposition (`Transpose`) introduces massive memory copying overhead. In transformer benchmarks, 4D transpositions accounted for **19.8% of total CPU time** and dominated `runtime.memmove` (26.7%).
- **The Principle**: In row-major tensors, the inner head dimension $D$ remains contiguous (unit stride) regardless of whether heads $H$ or sequence $S$ comes first.
- **The Implementation**:
  - Parametrize the inner kernel with sequence strides (`qSeqStride`, `kvSeqStride`) and head/group strides (`qGroupStride`).
  - Read operands and write outputs directly using strided indexing over the native slices without creating intermediate tensors.
  - Completely eliminates intermediate transposition passes and tensor allocations, cutting memory movement by >70%.

### 2. Cache-Aware Task Decomposition (KV-Head Grouping)
In multi-head attention with Grouped Query Attention (GQA) or Multi-Query Attention (MQA), $N_{query}$ query heads share $N_{kv}$ key/value heads ($N_{query} \ge N_{kv}$):
- **Avoid Independent Query Head Parallelization**: If tasks are split per query head, multiple worker threads concurrently compete for memory bandwidth fetching identical $K$ and $V$ tensors from L3 cache or DRAM.
- **Group by KV-Head**: Partition the workload into $N_{tasks} = Batch \times N_{kv}$ chunks.
- Each worker processes an entire $(batch, kv\_head)$ unit, iterating through all $G = N_{query} / N_{kv}$ query heads sequentially. This guarantees that the key and value sequences ($S_{kv} \times D$) are loaded once and stay resident in the core's private L1/L2 cache across all $G$ query heads.

### 3. Intra-Op Parallelization with `backend.Scheduler`
- **Scheduler Overview** (`internal/gobackend/scheduler`, documented in [`docs/GoBackendScheduler.md`](../docs/GoBackendScheduler.md)):
  - `backend.Scheduler` manages a persistent pool of worker goroutines (sized to `runtime.GOMAXPROCS(0)` by default, configurable via `parallelism=N`) with a strict priority hierarchy: **Inter-Call > Intra-Op > Inter-Op**.
  - **Automatic Inter-Call Scaling**: `FunctionExecutable.Execute` registers active executions via `Scheduler.EnterExecution()` / `LeaveExecution()`. `Scheduler.EffectiveParallelism()` automatically divides available parallelism across concurrent `Execute()` calls ($\max(1, \lfloor P / C \rfloor)$). When $C \ge P$, `EffectiveParallelism() == 1`, and ops run inline on the caller goroutine with zero synchronization or scheduling overhead.
  - **Deadlock-Free Work-Helping**: When an op fans out intra-op tasks via `ParallelizeChunks` or `RunIntraOpTasks`, the calling goroutine actively drains and executes its own op's tasks before waiting on the `sync.WaitGroup`, guaranteeing forward progress without deadlocks even when nested inside `ops_parallel` inter-op tasks.
- **Contiguous Chunk Partitioning (`ParallelizeChunks` — Guided Self-Scheduling)**:
  - Prefer `backend.Scheduler.ParallelizeChunks(totalItems, minChunkSize, func(start, end int) { ... })` to partition outer dimensions (`[0, totalItems)`) into contiguous ranges `[start, end)`.
  - `ParallelizeChunks` uses **Guided Self-Scheduling (GSS)** via an atomic compare-and-swap on `nextIdx`: each worker dynamically claims $\max(\text{minChunkSize}, \lceil \text{remaining} / (2 \times \text{numWorkers}) \rceil)$ items. This hands out large contiguous chunks at the start (amortizing scheduling overhead and preserving cache locality) and progressively smaller chunks at the tail (preventing slower Efficiency cores on asymmetric ARM64/x86 CPUs from becoming stragglers).
  - If `totalItems <= minChunkSize` or `!backend.Scheduler.IsEnabled()` (including when `EffectiveParallelism() <= 1`), `ParallelizeChunks` executes `fn(0, totalItems)` directly inline with zero allocations.
- **Kernel Metadata for Adaptive Chunking**:
  - Different kernel implementations (e.g., scalar `nosimd` vs. `simd` vs. `avx2`/`avx512`, or simple ReLU vs. transcendental GeLU/Erf) have vastly different per-element costs.
  - Rather than hardcoding chunk sizes in the executor or inferring SIMD support from registration priority, kernels register associated metadata (e.g., `activations.KernelMetadata{MinParallelChunk: ...}`) so each kernel specifies its own minimum parallel chunk size (and package-level metadata structs can be tuned per platform at runtime).
- **Asymmetric Multi-Worker Loops (`SaturateUpTo`, 3-Phase Tapered Chunking, & Dual-Ended Queue)**:
  - When workers require per-worker state (e.g., per-worker scratch buffers or panel buffers in `MatMul`), use `backend.Scheduler.SaturateUpTo(numWorkers, fn)`.
  - For 3D batched / 2D tiled GEMM (`matmul.AppendWorkItems`), work items are ordered from coarse to fine using **3-Phase Tapered Chunking** (Head multi-batch chunks $\to$ Middle 1-batch chunks $\to$ Tail `< 1 batch` 2D sub-tiles along `M` and `N`).
  - On asymmetric architectures with both fast SME coprocessors and slower per-core NEON units (`internal/gobackend/dot/matmul/neon/large.go`), workers coordinate via a lock-free 64-bit `bounds atomic.Uint64` (`(head << 32) | tail`) dual-ended queue: SME workers pop coarse chunks from `head` while NEON workers pop fine sub-batch tiles from `tail` (and yield the final tail items when `rem <= 3*smeActive`, or `rem <= 6*smeActive` on slow E-cores).

### 4. In-Cache Reductions & Register Accumulation
- **Never Materialize Intermediate Full Matrices**: Materializing $S_q \times S_{kv}$ attention logit matrices in DRAM creates severe memory-bandwidth bottlenecks for sequence lengths $S \ge 512$.
- **Per-Worker Scratch Buffers**: Allocate a small scratch slice ($S_{kv} + 16$ floats) per worker (or reuse a buffer across rows).
- **Fused Math in Vector Registers**:
  - Pre-scale the query vectors ($Q_q \leftarrow Q_q \times \text{scale}$) once outside the inner key loop ($K_k$), or fold scale factors, additive masks, and attention bias into vector registers during the dot product ($Q_q \cdot K_k$).
  - **4-Key Unrolled Dot Product & 4-Way Horizontal Reduction**: Instead of reducing each key's dot product to a scalar individually (`reduceSum16` / `reduceSum8`), process 4 keys per iteration and reduce 4 vectors simultaneously into a single 4-lane `XMM` vector (`[s0, s1, s2, s3]`) via `VUNPCKLPS`/`VUNPCKHPS` + `VSHUFPS`, storing 4 scores with one `VMOVUPS` and updating a 4-lane running max with `VMAXPS`.
  - **Zero Scalar `exp` Tail via `-Inf` Tail Padding**: When `kvLenUnmasked` is not a multiple of the SIMD width (e.g. dynamic non-padded sentence lengths `kvLimit` or causal masks), pad the remaining `1..15` (AVX-512) or `1..7` (AVX2) elements of `scoresScratch` at `scoresScratch[kvLenUnmasked]` with `-Inf`, then evaluate the vectorized Estrin degree-7 polynomial + magic-number `exp512` / `exp256` over `ceil(kvLenUnmasked, VL)`. Because `-Inf - rowMax` clamps to `minLog = -88.0296936` ($n = -127 \implies (n+127) \ll 23 = 0.0\text{f}$), padded lanes evaluate to exact `0.0f` with zero scalar `fastmath.Exp32` calls.
  - **Deferred `invSum` Output Scaling & Multi-Accumulator `P @ V`**: Skip the second memory pass over `scoresScratch` (`scoresScratch[k] *= invSum`); instead, issue `1.0 / sum` (`VDIVSS`) before Stage 3 to hide division latency, accumulate unnormalized weights $P \cdot V$ across 4–8 independent vector accumulators (`Z0`–`Z7` or `Y0`–`Y7`) using non-destructive `VFMADD231PS` to hide 4-cycle FMA pipeline latency, and multiply the final output accumulators by `invSum` once before storing.

### 5. Head Dimension Specialization ($D \in \{32, 64, 128\}$)
- When the inner reduction loop over dimension $D$ has a dynamic or variable upper bound, compilers and SIMD abstractions cannot unroll completely and must maintain loop counters and boundary checks.
- Generating specialized fast paths for standard power-of-two head dimensions ($D=32, 64, 128$):
  - On AVX-512 (`Float32x16`): $D=32$ unrolls into exactly 2 vector registers (implemented in handwritten assembly `sdpaQueriesDim32NoMaskAVX512Asm` and `sdpaSoftmaxAndValueAccumDim32AVX512Asm` in `internal/gobackend/fusedops/avx512/sdpa_amd64.s`), $D=64$ into 4 vectors, $D=128$ into 8 vectors.
  - On AVX2 (`Float32x8`): $D=32$ unrolls into 4 vector registers (implemented in handwritten assembly `sdpaQueriesDim32NoMaskAVX2Asm` and `sdpaSoftmaxAndValueAccumDim32AVX2Asm` in `internal/gobackend/fusedops/avx2/sdpa_amd64.s`), $D=64$ into 8 vectors, $D=128$ into 16 vectors.
  - Completely unrolling the inner dimension eliminates loop overhead and allows optimal compiler/assembler instruction scheduling across FMA execution ports.

### 6. Multi-RHS Single-LHS-Pack Execution & Inline Cache-Resident Epilogues (`ExecuteMultiRHSWithEpilogue`)
- In fused multi-projection operations such as `FusedAttentionQKVProjection` (where the same input `xFlat` is multiplied by `wQ`, `wK`, and `wV`), invoking `matmul.ExecuteWithEpilogue` three separate times causes three major inefficiencies:
  1. **Redundant LHS Packing**: When `qDim` or `kvDim` fits in a single column panel (`Nc = 512`), `numColPanels == 1` triggers `!canCacheAllLHS`, causing `xFlat` to be transposed/packed into panel layout three separate times (once each for `Q`, `K`, and `V`).
  2. **Cold L3/DRAM Epilogue Reads**: Running `ApplyEpilogue` after the full-matrix GEMM barrier forces the bias/activation pass to re-read the entire `Q`, `K`, and `V` output tensors from L3 cache or DRAM instead of modifying each worker's freshly written `Mc` rows while hot in L1/L2 cache.
  3. **Synchronization Overhead**: Three separate `ExecuteWithEpilogue` calls incur 6 parallel synchronization barriers per attention layer (`3 × SaturateUpTo` + `3 × ParallelizeChunks`).
- **The Solution (`matmul.ExecuteMultiRHSWithEpilogue` & `matmul.AppendRowWorkItems`)**:
  - Partition work across batch and LHS rows (`M`) only (`AppendRowWorkItems`), aligning coarse row chunks to `LHSPanelCrossSize` (`Mc = 32`).
  - Within a single `SaturateUpTo` dispatch (`avx512LargeMultiFloat32` / `avx2LargeMultiFloat32` and generated half/double variants), each worker steps through its assigned row range `[LHSRowStart, LHSRowEnd)` in L2-resident sub-chunks (`1..4` `Mc` panels, 48–192 KB), packs `xFlat[subRowStart:subRowEnd]` into `packedLHS` **only once** on the first target (`Q`), reuses the L2-resident `packedLHS` across `K` and `V` (`lhsAlreadyPacked = true`), and immediately applies each target's bias/activation epilogue inline via `matmul.ApplyEpilogueRows` while the output rows are still hot in L1/L2 cache.



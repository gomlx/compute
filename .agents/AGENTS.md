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
- `internal/gobackend`: the implementation of the "go" backend. It consists of
  buffer and execution logic, and "registration" of ops during build and
  execution. The implementation of each op is (or is being moved to) in its
  sub-packages `ops`, `dot`, `conv` and `fusedops` (in works).
- `support`: generic support libraries.
  - `support/testutil`: test utilities that can be used by any `compute.Backend`
    implementation to test. 
  - `support/backendtest`: Backend compliance tests, that can be run against
    any backend. Simply call `RunAll(t *testing.T, b compute.Backend)` from your
    backend tests. These tests always check the capabilities of the backend, and
    tests not implemented by the backend are skipped.

## Coding Style In GoMLX projects, including this one.

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

- Use generics where possible.
- Use `slices` and `maps` package for slice operations.
- Look also into `support/xslices` package for more slice and map helper methods.
- Look into `support/xsync` package for more syncronization helpers.
- Look into `support/sets` package for a generic `Set[T]` structure.
- Use iterators (package `iter`) where it makes sense.
- Use the `for range` construct for loops over slices, maps, etc.
- Use `any` instead of `interface{}`.
- Organize tests in hierarchies using `t.Run()` to group related tests.

### Tests

- For backend tests that could be used for any backend, write them in `support/backendtest` so other
  backends can benefit.
- We DONT depend on testify or other test libraries: we are trying to minimize external dependencies.
- Use the locally defined `support/testutil` for test utilities for equality (or InDelta or InRelativeDelta comparisons of buffers, etc.).

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

## How to use SIMD in Go with archsimd

Go 1.25 introduced experimental SIMD support through the `simd/archsimd` package. This allows for writing architecture-independent SIMD code while still leveraging specialized hardware instructions (like AVX-512 on x86).

### Enabling archsimd

To use archsimd, you must:
1. Use a compatible Go version (1.25+).
2. Set the environment variable `GOEXPERIMENT=simd` during build and test.
3. Use the `//go:build goexperiment.simd` build tag in your files.

### Common Vector Types

Vectors are named by their element type and the number of elements. They usually come in three widths:

| Width | Type Examples | Hardware Target (x86) |
| :--- | :--- | :--- |
| **128-bit** | `Float32x4`, `Int32x4`, `Uint16x8`, `Int8x16` | SSE |
| **256-bit** | `Float32x8`, `Int32x8`, `Uint16x16`, `Int8x32` | AVX, AVX2 |
| **512-bit** | `Float32x16`, `Int32x16`, `Uint16x32`, `Int8x64` | AVX-512 |

Other types include `Float64x2/x4/x8`, `Uint64x2/x4/x8`, and corresponding `Mask` types (e.g., `Mask32x16`).

### Basic Usage Example

```go
//go:build amd64 && goexperiment.simd

package mypackage

import "simd/archsimd"

func AddSlices(a, b, res []float32) {
    for i := 0; i < len(a); i += 16 {
        va := archsimd.LoadFloat32x16((*[16]float32)(&a[i]))
        vb := archsimd.LoadFloat32x16((*[16]float32)(&b[i]))
        vres := va.Add(vb)
        vres.Store((*[16]float32)(&res[i]))
    }
}
```

### Summary of Found Operations

| Category | Operations |
| :--- | :--- |
| **Arithmetic** | `Add`, `Sub`, `Mul`, `Div`, `Sqrt`, `Abs`, `Min`, `Max` |
| **Logical** | `And`, `Or`, `Xor`, `AndNot` |
| **Shift** | `ShiftAllLeft(bits uint64)`, `ShiftAllRight`, `ShiftLeft(v Vector)` (variable per-element) |
| **Conversion** | `ExtendToUint32`, `AsUint32x16` (bit-cast), `ConvertToFloat32` (value cast) |
| **Comparison** | `Equal`, `NotEqual`, `Greater`, `Less`, `GreaterEqual`, `IsZero` |
| **Special** | `LeadingZeros`, `BroadcastUint32x16(val)`, `GetHi()`, `GetLo()` |
| **Masking** | `Merge(other, mask)`, `Masked(mask)`, `mask.ToInt32x16()` |

### Working with Masks and Bit Manipulation

Masks are returned by comparison operations (e.g., `v.Equal(zero)` returns a `Mask`).
- **Merging**: `res = trueVal.Merge(falseVal, mask)` returns `trueVal` where `mask` is true, and `falseVal` otherwise.
- **Bitmask Vector**: To get a vector where all bits are set based on a mask, use `mask.ToInt32x16().AsUint32x16()`. This is useful for manual bitwise manipulation when `Merge` behavior is complex.

### Architecture Specifics

While `archsimd` is cross-architecture, some operations may only be available on certain platforms or require specific CPU features (like AVX-512). Always check for support using `archsimd.X86.AVX512()` or similar checks in `init()`.
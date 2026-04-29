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
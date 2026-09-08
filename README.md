[![Documentation](https://img.shields.io/badge/docs-gomlx.github.io-blue.svg)](https://gomlx.github.io/)
[![Sponsor GoMLX](https://img.shields.io/badge/Sponsor-GoMLX-white?logo=github&style=flat-square)](https://github.com/gomlx/gomlx/blob/main/README.md#-support-the-project)
<br/>
[![Linux/amd64 Tests](https://github.com/gomlx/compute/actions/workflows/linux_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/compute/actions/workflows/linux_amd64_tests.yaml)
[![Linux/arm64 Tests](https://github.com/gomlx/compute/actions/workflows/linux_arm64_tests.yaml/badge.svg)](https://github.com/gomlx/compute/actions/workflows/linux_arm64_tests.yaml)
[![Darwin/arm64 Tests](https://github.com/gomlx/compute/actions/workflows/darwin_tests.yaml/badge.svg)](https://github.com/gomlx/compute/actions/workflows/darwin_tests.yaml)
[![Windows/amd64 Tests](https://github.com/gomlx/compute/actions/workflows/windows_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/compute/actions/workflows/windows_amd64_tests.yaml)


# Compute Backend APIs

Package `compute` provides a modular API for defining and executing multidimensional computation graphs with pluggable backends.

It defines `shapes` (tensor shapes) and `dtypes` (data types) and the top-level `compute` package defines a `Backend` API (a series of interfaces), that
can be used to define a computation graph, JIT-compile it, transfer buffers (raw values) to/from the backend, and execute compiled computations.

It powers [GoMLX](https://github.com/gomlx/gomlx), the machine learning framework for Go, but can be used directly also. With the caveat that `compute.Backend` doesn't aim to be ergonomic, but instead "correct" and "minimal" (including minimal dependencies). 
For a more convenient API for complex computation and auto-differentiation, use GoMLX instead.

The default `compute.New()` function creates a backend object using any registered implementation, or inspects the `GOMLX_BACKEND` environment variable to select a specific backend and optional configuration. For example, to use the Go backend, set `GOMLX_BACKEND=go` (or `GOMLX_BACKEND="go:parallelism=4"` to configure it). If `GOMLX_BACKEND` is not set, `compute.New()` falls back to `compute.DefaultConfig` or the first registered backend.

## Available Backend Implementations

### The Native **"go"** Backend

This repository includes the package `gobackend` that implements the `compute.Backend` API using pure Go. 
It is very portable but relatively slow (when compared with well-supported backends like XLA and ONNX).

It has support for SIMD on AVX2 and AVX-512 (*amd64*) when built with `GOEXPERIMENT=simd`, and there are 
plenty of "low-hanging fruit" to improve performance for anyone interested in contributing.

#### Backend Configuration Options (`GOMLX_BACKEND=go:...`)

When instantiating the Go backend (via `compute.New()` or `GOMLX_BACKEND`), options can be provided as a comma-separated list of `<option>` or `<option>=<value>`:

- `parallelism=<N>`: sets the maximum worker pool concurrency (number of worker goroutines) used to parallelize operations. Defaults to the number of available CPU cores.
- `ops_sequential`: forces graph operations to be executed sequentially.
- `ops_parallel`: forces independent graph operations to be executed in parallel where possible. (By default, ops execute in parallel when only one computation is running, and sequentially otherwise).
- `dependency_order`: schedules graph nodes based on dependency order.
- `creation_order`: schedules graph nodes based on creation order.

Example:
```bash
export GOMLX_BACKEND="go:parallelism=8,ops_parallel"
```

See the [Environment Variables](#environment-variables) section below for fine-grained control over SIMD features and optimizations.

### Other Backends

The `compute.Backend` API is currently implemented by:

- Package
  [`github.com/gomlx/go-xla/compute/xla`](https://github.com/gomlx/go-xla/tree/main/compute/xla): an
  [XLA (PJRT)](https://openxla.org/) based implementation, the same engine used by JAX
  and TensorFlow. It uses CGO (it's a C++ library), but it supports GPUs and
  TPUs, as well as a fast CPU, proper JIT compilation. Limited to static shapes
  though. It includes an optional auto-installer.
- Package [`github.com/gomlx/compute-onnx`](https://github.com/gomlx/compute-onnx): an
  ONNX Runtime (ORT) based implementation. It converts a `compute.Backend` computation graph into an ONNX proto and executes it
  using ORT. It includes an optional auto-installer. Offers broader support for hardware, dynamic shape support on
  some platforms, and accelerated WebAssembly (using `"onnx:webgpu"`) with some caveats.
- The project [go-darwinml](https://github.com/gomlx/go-darwinml/):
  **experimental** support for Apple's CoreML, with acceleration for GPU (Metal)
  and CPU (arm64). It is currently broken and looking for collaborators -- or donations to acquire hardware
  to support it.

## Using the `compute.Backend` interface

To use a backend, import the desired backend package (which registers itself) and call `compute.New()`:

```go
package main

import (
	"log"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend" // Registers the "go" backend
)

func main() {
	// Creates backend: uses $GOMLX_BACKEND if set, or the default registered backend.
	backend, err := compute.New()
	if err != nil {
		log.Fatalf("Failed to initialize backend: %+v", err)
	}
	defer backend.Finalize()

	// Use backend to build and execute computation graphs...
	_ = backend
}
```

## Roadmap

**Short term:**

- [x] Add basic dynamic shapes support. (experimental, see `./docs/DynamicShapes.md`)
- [x] Add initial SIMD implementation for the Go backend (AVX512 and AVX2, using plain `simd/archsimd`).

**Future:**

We are exploring support (Backend implementations) for:

* Integrate more SIMD using go-highway for the Go backend.
* [LiteRT](https://github.com/google-ai-edge/litert): Add LiteRT-based Backend implementation. Broad support for
  edge hardware (as well as WebAssembly), and supposedly very efficient.
* [llama.cpp](https://github.com/ggml-org/llama.cpp): using [github.com/hybridgroup/yzma](https://github.com/hybridgroup/yzma), a "pure-go" binding.

## Implementing your own backend

It's conceptually simple:

- Inherit from `notimplemented`, and return empty capabilities.
- Implement the transferring of buffers to/from your backend.
- Implement the operations that you need.
- Make sure you pass the "compliance" tests in `support/backendtest`, by calling
  the function `backendtest.RunAll(t *testing.T, b compute.Backend)`, or running
  the individual tests. Example:

```go
func TestCompliance(t *testing.T) {
	backendtest.RunAll(t, myBackend)
}
```

- In addition to tests, `support/backendtest` also provides standard compliance benchmarks (by calling `backendtest.RunAllBenchmarks(b *testing.B, backend compute.Backend)`). This includes benchmarks across various sizes and configurations of `DotGeneral` (matrix multiplication), as well as standard neural network operations (`Dense`, `QuantizedDense`, `Softmax`, `LayerNorm`, `Gelu`). Example:

```go
func BenchmarkCompliance(b *testing.B) {
	backendtest.RunAllBenchmarks(b, myBackend)
}
```

Consider using GoMLX tests against your Backend to test that they are working --
just set the environment variable `GOMLX_BACKEND` to your new backend, and you
can run arbitrary tests. Also, once you have enough ops implemented, you can use
some of the example models to benchmark your backend against some of the others.

## Environment Variables

- `GOMLX_BACKEND`: defines the backend engine to use (if using `compute.New()`). The value is formatted as "<backend_name>[:<backend_config>]",
  with the config part being optional. Examples:
  - `GOMLX_BACKEND=go`: Use the "Go backend", the pure Go implementation that is very portable but slow.
  - `GOMLX_BACKEND="go:parallelism=4"`: Use the Go backend configured with at most 4 worker goroutines.
  - `GOMLX_BACKEND="xla:cpu"`: Use XLA (the faster backend, only runs on Linux now) for CPU.
  - `GOMLX_BACKEND="xla:cuda"`: Use XLA for Nvidia CUDA.
  - `GOMLX_BACKEND="xla:/path/to/my/pjrt_plugin.so"`: Use XLA with an arbitrary PJRT. PJRT is a plugin system for XLA to support different hardware.
    One can install PJRTs built for NVIDIA GPUs (there is an installation script for that), there is also one for ROCm (not tested by the author),
    for TPU (Google Cloud) and reports of PJRTs being built for even newer accelerators (e.g.: [TensTorrent XLA](https://github.com/tenstorrent/tt-xla)).
- For the native Go backend:
  - `GOMLX_GO_SIMD_AVX512`: set to `0` or `false` to disable AVX512-specific SIMD implementations in the native Go backend (e.g. optimized matmul and activation kernels). Note: this only controls AVX512-specific code; generic portable SIMD operations will still use hardware vector features if available. The default is enabled if AVX512 is present.
  - `GOMLX_GO_SIMD_AVX2`: set to `0` or `false` to disable AVX2-specific SIMD implementations in the native Go backend (e.g. optimized matmul and activation kernels). Note: this only controls AVX2-specific code; generic portable SIMD operations will still use hardware vector features if available. The default is enabled if AVX2 is present.
  - `GOMLX_GO_AVX512_ASM`: set to `0` or `false` to disable the assembly microkernel for Float32 and use the Go SIMD kernel instead.
  - `GOMLX_GO_AVX512_KC`, `GOMLX_GO_AVX512_MC`, `GOMLX_GO_AVX512_NC`: cache blocking tuning parameters for AVX512 matrix multiplication.
  - `GOMLX_GO_DOT_MATMUL`: set to `0` or `false` to disable the default matrix multiplication implementation in the native Go backend. The default is enabled.
  - `GOMLX_GO_FUSION`: if set to `0`, `false` to disable fused operations in the native Go backend. The default is enabled.
- For the [XLA backend](https://github.com/gomlx/go-xla/tree/main/compute/xla)
  - `PJRT_PLUGIN_LIBRARY_PATH`: the underlying XLA backend uses this variable as an extra directory to search for plugin locations.
    It searches for the systems library paths (`$LD_LIBRARY_PATH`, `/etc/ld.so.conf`), the default `/usr/local/lib/gomlx/pjrt` and `$PJRT_PLUGIN_LIBRARY_PATH` if set.
  - `GOMLX_NO_AUTO_INSTALL`: if set to `1`, GoMLX will not automatically install PJRTs when running on a system without them.
  - `XLA_FLAGS`: optional controls for XLA backend. It should be set to a semicolon (";") separated list of options. If you set to `--help` 
    the backend will print out some help for all options. There is also a description on the page [XLA Flags Guidance](https://openxla.org/xla/flags_guidance).

## WebAssembly (WASM) Support

The "go" backend is well-supported in WebAssembly (WASM) environments.

It passes all compliance tests:

```bash
GOOS=js GOARCH=wasm go test -exec wasmbrowsertest ./... -count=1 -v
```

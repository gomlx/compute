// Package gobackend implements a native Go [compute.Backend]: 100% portable (including WASM)
// with optional SIMD/assembly acceleration (AVX2, AVX-512, ARM64 NEON/SME), fused operations,
// and constant weight caching.
//
// Configuration options can be passed via GOMLX_BACKEND="go:<option1>,<option2>=<value>" or
// compute.NewWithConfig("go:..."). Supported options:
//   - parallelism=<N>: maximum intra-op worker goroutines (default: runtime.GOMAXPROCS(0)).
//   - num_executors=<N>: maximum concurrent node executors in parallel mode (default: 8).
//   - ops_sequential / ops_parallel: force sequential or parallel inter-op execution (default: dynamic).
//   - dependency_order / creation_order: topological scheduling strategy (default: dependency_order).
//   - quantized_cache=<always|balanced|never>: constant weight/scale/zero-point/bias caching policy
//     for FusedQuantizedDense (default: balanced; aliases: true/1 or bare quantized_cache for always;
//     false/none/0 for never). See docs/GoBackendMemoryUsage.md for details.
//   - constant_arena[=<true|false>]: allocate cached/pre-packed constant buffers in 2 MiB-aligned
//     off-heap mmap chunks advised with MADV_HUGEPAGE on Linux, reference-counted per Builder/Executable
//     so chunks are unmapped back to the OS when all referencing executables are finalized (default: true;
//     can also be configured via GOMLX_GO_CONSTANT_ARENA).
//   - no_ops: skip kernel math for scheduling/framework overhead benchmarking.
//
// Passing an unknown option such as GOMLX_BACKEND="go:help" returns an error listing all valid
// options.
//
// This package is a thin wrapper around internal/gobackend.
package gobackend

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"

	// Registers all the ops.
	_ "github.com/gomlx/compute/internal/gobackend/defaultpkgs"
)

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = "go"

// GetBackend returns a singleton Go backend, created with the default configuration.
// The backend is only created at the first call of the function.
//
// The singleton is never destroyed.
var GetBackend = sync.OnceValue(func() compute.Backend {
	backend, err := New("")
	if err != nil {
		panic(err)
	}
	return backend
})

// New constructs a new Go Backend using the comma-separated configuration options in config.
// See package documentation or https://gomlx.github.io/docs/backends/#2-the-go-backend-go
// for supported options.
func New(config string) (compute.Backend, error) {
	return gobackend.New(config)
}

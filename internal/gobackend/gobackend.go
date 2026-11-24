// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package gobackend implements a [compute.Backend] using pure Go (plus optional SIMD/assembly).
//
// Its main goal is portability (including WASM), while still delivering high CPU throughput
// via SIMD/assembly microkernels, fused operations, and constant weight panel caching.
//
// It also serves as a reference implementation for the [compute.Backend] interface.
//
// Supported configuration options (via GOMLX_BACKEND="go:<opt1>,<opt2>=<val>"):
//   - parallelism=<N>: maximum intra-op scheduler worker goroutines (default: GOMAXPROCS).
//   - num_executors=<N>: maximum concurrent node executors in parallel graph mode (default: 8).
//   - ops_sequential / ops_parallel: force sequential or parallel inter-op execution (default: dynamic).
//   - dependency_order / creation_order: topological scheduling strategy (default: dependency_order).
//   - quantized_cache=<always|balanced|never>: constant weight caching policy for FusedQuantizedDense
//     (aliases: true/1 for always; false/none/0 for never; bare "quantized_cache" selects always;
//     default: balanced). See compute/docs/GoBackendMemoryUsage.md for details.
//   - constant_arena[=<true|false>]: allocate transformed/pre-packed constant buffers into off-heap
//     2 MiB-aligned mmap + MADV_HUGEPAGE arena chunks (reference-counted per builder/executable and
//     unmapped when all referencing builders/executables or the backend are finalized). Can also be
//     configured via GOMLX_GO_CONSTANT_ARENA (default: true).
//   - no_ops: skip kernel math (for scheduling/framework benchmarking).
package gobackend

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend/scheduler"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/support/envutil"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/compute/support/xsync"
	"github.com/pkg/errors"
)

// Registers the various generics function instances.
//go:generate go run ../../internal/cmd/gobackend_dtypemap

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = "go"

// Registers New() as the default constructor for "go" backend.
func init() {
	compute.Register(BackendName, New)
}

// KnownOptionsSetters registers handlers for backend configuration options.
// These allow any sub-package to register option setters.
//
// The key is the name of the option: it can't take any values yet.
var KnownOptionsSetters = make(map[string]func(b *Backend, key string) error)

// GetBackend returns a singleton backend for Go backend, created with the default configuration.
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

// NewBackend returns a "go" backend (*compute/internal/gobackend.Backend) for test.
// It returns an error if the backend is not a "go" backend.
func NewBackend() (*Backend, error) {
	backendRaw, err := compute.New()
	if err != nil {
		return nil, err
	}
	be, ok := backendRaw.(*Backend)
	if !ok {
		return nil, errors.Errorf("backend configured is not a Go backend: GOMLX_BACKEND=%q", os.Getenv(compute.ConfigEnvVar))
	}
	return be, nil
}

// New constructs a new Go backend using the comma-separated configuration options in config.
// See https://gomlx.github.io/docs/backends/#2-the-go-backend-go for documentation on supported options.
func New(config string) (compute.Backend, error) {
	b := newDefaultBackend()
	parts := strings.SplitSeq(config, ",")
	for part := range parts {
		key := part
		var value string
		if before, after, ok := strings.Cut(part, "="); ok {
			key, value = before, after
		}
		switch key {
		case "parallelism":
			vInt, err := strconv.Atoi(value)
			if err != nil {
				return nil, errors.Wrapf(err,
					"invalid value for %q in Go backend config: needs an int, got %q", key, value)
			}
			b.Scheduler.SetMaxParallelism(vInt)
			fmt.Printf("Go backend: parallelism set to %d\n", vInt)
		case "ops_sequential":
			// This will force the ops to be executed sequentially.
			b.OpsExecutionType = OpsExecutionSequential
		case "ops_parallel":
			// This will force the ops to be executed in parallel where possible.
			// By default, ops are executed dynamically (parallel when only one execution is live, sequential otherwise).
			b.OpsExecutionType = OpsExecutionParallel
		case "dependency_order":
			b.SchedulingStrategy = DependencyOrderSchedule
		case "creation_order":
			b.SchedulingStrategy = CreationOrderSchedule
		case "no_ops":
			b.NoOps = true
		case "num_executors":
			vInt, err := strconv.Atoi(value)
			if err != nil {
				return nil, errors.Wrapf(err,
					"invalid value for %q in Go backend config: needs an int, got %q", key, value)
			}
			b.NumExecutors = vInt
		case "quantized_cache":
			policy, err := parseQuantizedCachePolicy(value)
			if err != nil {
				return nil, err
			}
			b.QuantizedCache = policy
		case "constant_arena":
			enabled, err := parseBoolOption(key, value, true)
			if err != nil {
				return nil, err
			}
			b.UseConstantArena = enabled
		case "":
			// No-op, just skip.
		default:
			setter, ok := KnownOptionsSetters[key]
			if !ok {
				extraOptions := ""
				if len(KnownOptionsSetters) > 0 {
					extraOptions = ", " + strings.Join(xslices.SortedKeys(KnownOptionsSetters), ", ")
				}
				return nil, errors.Errorf("unknown configuration option %q for the Go backend -- valid configuration options are: "+
					"parallelism=#workers, num_executors=#executors, ops_sequential, ops_parallel, dependency_order, creation_order, no_ops, quantized_cache=<always|balanced|never>, constant_arena[=<true|false>]%s; "+
					"see documentation in https://gomlx.github.io/docs/backends/#2-the-go-backend-go",
					key, extraOptions)
			}
			err := setter(b, key)
			if err != nil {
				return nil, err
			}
		}
	}
	return b, nil
}

// QuantizedCachePolicy controls whether and when constant quantized weights (and their
// supporting constant scales, zero-points, and biases) in FusedQuantizedDense are pre-unpacked
// or converted into cached representations across executions.
type QuantizedCachePolicy int

const (
	// QuantizedCacheBalanced (the default) decides per constant tensor: it caches a constant
	// weight or supporting tensor when its cached size is at most DefaultQuantizedCacheMaxWeightBytes
	// (4 MiB) AND its net extra memory (cachedBytes - releasedRawBytes) is at most
	// DefaultQuantizedCacheMaxExtraRatio (2.0x) of the raw tensor byte size.
	QuantizedCacheBalanced QuantizedCachePolicy = iota

	// QuantizedCacheAlways caches every eligible constant quantized weight (both Mode B K4 panels
	// and Mode A float32 GEMM panels) and supporting constant tensor regardless of size or
	// expansion ratio.
	QuantizedCacheAlways

	// QuantizedCacheNever disables quantized constant caching completely; all quantized weights
	// and supporting tensors are unpacked or converted on the fly in bounded pooled chunks.
	QuantizedCacheNever
)

const (
	// DefaultQuantizedCacheMaxWeightBytes caps the cached byte size of any single tensor cached
	// automatically under QuantizedCacheBalanced (4 MiB). This covers small/medium L3-resident
	// models (e.g. MiniLM, BERT/BGE-base) while avoiding multi-GB expansion and DRAM-bandwidth
	// regressions on large LLMs (1B-70B).
	DefaultQuantizedCacheMaxWeightBytes int64 = 4 * 1024 * 1024

	// DefaultQuantizedCacheMaxExtraRatio caps the net extra memory ratio (extraBytes / rawBytes)
	// under QuantizedCacheBalanced (2.0x). Donated 4-bit to 8-bit Mode B panels sit at 0.10x-1.50x
	// extra (cached), whereas 2-3 bit formats (2.49x-4.0x) and Mode A float32 panels (2.76x-11.2x)
	// stay uncached by default.
	DefaultQuantizedCacheMaxExtraRatio = 2.0
)

// String returns the canonical configuration name of the QuantizedCachePolicy.
func (p QuantizedCachePolicy) String() string {
	switch p {
	case QuantizedCacheBalanced:
		return "balanced"
	case QuantizedCacheAlways:
		return "always"
	case QuantizedCacheNever:
		return "never"
	default:
		return fmt.Sprintf("QuantizedCachePolicy(%d)", int(p))
	}
}

func parseQuantizedCachePolicy(value string) (QuantizedCachePolicy, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "always", "true", "1":
		return QuantizedCacheAlways, nil
	case "balanced":
		return QuantizedCacheBalanced, nil
	case "never", "none", "false", "0":
		return QuantizedCacheNever, nil
	default:
		return QuantizedCacheBalanced, errors.Errorf(
			"invalid value for \"quantized_cache\" in Go backend config: expected always|true|1, balanced, or never|none|false|0, got %q",
			value)
	}
}

func parseBoolOption(key, value string, defaultIfEmpty bool) (bool, error) {
	v := strings.ToLower(strings.TrimSpace(value))
	if v == "" {
		return defaultIfEmpty, nil
	}
	if envutil.TrueValues.Has(v) {
		return true, nil
	}
	if envutil.FalseValues.Has(v) {
		return false, nil
	}
	return false, errors.Errorf("invalid boolean value %q for %q in Go backend config", value, key)
}

// DefaultNumExecutors is the default number of concurrent worker executors
// used in executeParallel to execute ready graph nodes.
const DefaultNumExecutors = 8

// DefaultConstantArenaGCThresholdBytes is the default byte threshold (64 MiB) of released donated
// Go-heap constant buffers after which the backend triggers GC + OS memory scavenging
// (debug.FreeOSMemory) to bound peak MaxRSS while weights are being transferred into the off-heap
// constant_arena.
const DefaultConstantArenaGCThresholdBytes int64 = 64 * 1024 * 1024

func newDefaultBackend() *Backend {
	b := &Backend{
		NumExecutors:                  DefaultNumExecutors,
		OpsExecutionType:              OpsExecutionDynamic,
		QuantizedCache:                QuantizedCacheBalanced,
		UseConstantArena:              envutil.MustReadBool(envutil.GoBackendConstantArena, true),
		ConstantArenaGCThresholdBytes: DefaultConstantArenaGCThresholdBytes,
		constantArena:                 newConstantArena(),
	}
	b.Scheduler = scheduler.New()
	return b
}

// Backend implements the compute.Backend interface.
type Backend struct {
	// bufferPools holds lock-free Treiber stack pools of reusable buffers indexed by
	// bufferPoolBucketIndex(bucketedSize).
	bufferPools [numBufferPoolBuckets]atomic.Pointer[bufferPool]
	Scheduler   *scheduler.Scheduler

	NumLiveExecutions atomic.Int32

	// OpsExecutionType defines how to execute the ops of a computation.
	OpsExecutionType OpsExecutionType

	// lifecycle synchronizes in-flight operations against Backend.Finalize.
	lifecycle xsync.Lifecycle

	// SchedulingStrategy
	SchedulingStrategy ScheduleStrategy

	// NoOps disables the actual calculation in operations when true.
	NoOps bool

	// NumExecutors is the number of concurrent executor goroutines to launch in executeParallel.
	NumExecutors int

	// QuantizedCache controls pre-unpacking/conversion caching of constant quantized weights
	// and supporting tensors in FusedQuantizedDense (configured via
	// "go:quantized_cache=<always|balanced|never>"). Defaults to QuantizedCacheBalanced.
	QuantizedCache QuantizedCachePolicy

	// UseConstantArena controls whether transformed constant caches (e.g. pre-packed GEMM panels)
	// are sub-allocated from off-heap 2 MiB-aligned mmap + MADV_HUGEPAGE arena chunks instead of
	// individual Go heap allocations. Configured via "go:constant_arena" or GOMLX_GO_CONSTANT_ARENA
	// (default: true).
	UseConstantArena bool

	// ConstantArenaGCThresholdBytes controls how many bytes of donated Go-heap constant buffers
	// may be released while transferring weights into constantArena before triggering GC + OS
	// page reclamation (debug.FreeOSMemory) to cap peak MaxRSS. 0 disables forced reclamation.
	// Defaults to DefaultConstantArenaGCThresholdBytes (64 MiB).
	ConstantArenaGCThresholdBytes int64

	// constantArena manages off-heap mmap chunks when UseConstantArena is enabled.
	constantArena *constantArena
}

// Compile-time check that the gobackend.Backend implements compute.Backend.
var _ compute.Backend = &Backend{}

// Name returns the short name of the backend. E.g.: "xla" for the Xla/PJRT plugin.
func (b *Backend) Name() string {
	return "Go Backend"
}

// String implement compute.Backend.
func (b *Backend) String() string { return BackendName }

// Description is a longer description of the Backend that can be used to pretty-print.
func (b *Backend) Description() string {
	extra := ""
	if b.QuantizedCache != QuantizedCacheBalanced {
		extra = fmt.Sprintf(", quantized_cache=%s", b.QuantizedCache)
	}
	return fmt.Sprintf("Go Portable Compute Backend (%s, parallelism=%d%s)",
		strings.Join(ActiveArchitectures, ", "), b.Scheduler.MaxParallelism(), extra)
}

// NumDevices return the number of devices available for this Backend.
func (b *Backend) NumDevices() int {
	return 1
}

// DeviceDescription returns a description of the device with the given deviceNum.
func (b *Backend) DeviceDescription(_ compute.DeviceNum) string {
	return "device#0"
}

// Capabilities method returns information about what is supported by this backend.
func (b *Backend) Capabilities() compute.Capabilities {
	return Capabilities
}

// Builder creates a new builder used to construct a named computation.
func (b *Backend) Builder(name string) compute.Builder {
	builder := &Builder{
		Builder: notimplemented.Builder{
			ErrFn: notImplementedError,
		},
		Backend: b,
		name:    name,
	}

	// Create the main function
	builder.MainFn = &Function{
		Function: notimplemented.Function{
			ErrFn: notImplementedError,
		},
		RawBuilder: builder,
		name:       "main",
		nodeDedup:  make(map[NodeDedupKey][]*Node),
	}
	builder.Functions = append(builder.Functions, builder.MainFn)
	// Set the "not implemented" custom message:
	return builder
}

func notImplementedError(opType compute.OpType) error {
	return errors.Wrapf(notimplemented.NotImplementedError, "sorry, op %q not implemented in the \"go\" backend yet "+
		"-- reach out to github.com/gomlx/gomlx and open an issue if you need this op, this helps us prioritize the work",
		opType)
}

// Finalize releases all the associated resources immediately, and makes the backend invalid.
func (b *Backend) Finalize() {
	if b == nil {
		return
	}
	b.lifecycle.Finalize(func() {
		for i := range b.bufferPools {
			b.bufferPools[i].Store(nil)
		}
		if b.constantArena != nil {
			b.constantArena.Finalize()
		}
	})
}

// IsFinalized returns true if the backend has been finalized.
func (b *Backend) IsFinalized() bool {
	if b == nil {
		return true
	}
	return b.lifecycle.IsFinalized()
}

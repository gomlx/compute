# Portable & Extensible Random Number Generation (`RNGBitGenerator`) Design

This document specifies the redesign of `compute.StandardOps.RNGBitGenerator` and its higher-level counterparts in `gomlx` (`core/graph` and `ml/model`) to make **Random Number Generator (RNG)** / **Pseudo-Random Number Generator (PRNG)** support portable, extensible, and optionally bit-reproducible across all GoMLX backends.

---

## 1. Terminology

- **RNG (Random Number Generator)**: General term for a system or operation that produces random values.
- **PRNG (Pseudo-Random Number Generator)**: A deterministic mathematical algorithm that generates a sequence of numbers whose statistical properties approximate true randomness. Given the same initial **state** (derived from a **seed**), a PRNG always produces the exact same sequence.
- **Counter-Based PRNG**: A parallel-friendly family of PRNGs (introduced by Salmon et al., SC 2011, e.g. **Philox** and **ThreeFry**, as well as counter-mode **SplitMix64** and **ChaCha8**) where random output is computed as a pure bijection $f_{\text{key}}(\text{counter} + i)$ for element/block index $i \in [0, N)$. Because output $i$ does not depend on output $i-1$, counter-based PRNGs vectorize trivially on CPUs (SIMD), GPUs, and TPUs, and advance their state in $O(1)$ by adding $N$ to the counter.
- **State-Transition (Sequential) PRNG**: A classic PRNG (such as **PCG** — Permuted Congruential Generator) where each step mutates internal state $s_{t+1} = g(s_t)$ sequentially. Fast for scalar CPU loops, but not natively parallelizable across GPU/SIMD lanes for a single stream.
- **CSPRNG (Cryptographically Secure Pseudo-Random Number Generator)**: An OS or cipher-backed generator (such as Go's `crypto/rand` or counter-mode `ChaCha8`) whose output is computationally indistinguishable from true random bits even to an adversary.

---

## 2. Audit of Current Backend Implementations & Pain Points

Previously, `compute` hardcoded `compute.RNGStateShape = shapes.Make(dtypes.Uint64, 3)` and assumed every backend implemented Philox (`RNGBitGenerator(state Value, shape shapes.Shape)`). In practice, backends diverged significantly:

| Backend | Current Implementation | Native Capability & Friction Points |
| :--- | :--- | :--- |
| **`xla`** (`go-xla`) | Hardcodes `stablehlotypes.RNGPhilox` with `[3]uint64`. | StableHLO's `rng_bit_generator` natively supports `DEFAULT` (`[2]uint64`), `THREE_FRY` (`[2]uint64` — **rejects** `[3]uint64`), and `PHILOX` (`[2]uint64` or `[3]uint64`). Also, XLA's GSPMD partitioner recognizes `stablehlo.rng_bit_generator` to produce distinct per-shard streams when `#rngState` is replicated. |
| **`go`** (`compute/gobackend`) | Comment claims Philox, but runs sequential `math/rand/v2.NewPCG(state[0], state[1])`. | Ignores `state[2]`, runs sequentially byte-by-byte, and re-seeds `NewPCG` from marshaled bytes instead of `UnmarshalBinary`. Counter-based `Philox` / `ThreeFry` in Go can be parallelized across workers and SIMD-accelerated. |
| **`onnx`** (`compute-onnx`) | Synthesizes counter-based **SplitMix64** from elementary ONNX ops (`Iota`, `BitwiseXor`, `ShiftRightLogical`, `Mul`) using `state[0..1]`. | Standard ONNX `RandomUniform`/`RandomNormal` ops are impure (no state tensor, float-only). In-graph PRNG decomposition is ideal, **but** ONNX Runtime WebGPU does not support `Uint64` GPU buffers — only 32-bit `Uint32`/`Int32`. |
| **`mlx`** (`go-darwinml/compute/mlx`) | Views `[3]uint64` as `[6]uint32`, XOR-folds to `[2]uint32`, calls `mlx::core::random::split` + `bits`, and adds `1` to state. | MLX's native PRNG is key-splitting **ThreeFry-2x32** using a `[2]uint32` key. Key-splitting differs from XLA's counter-advancing `THREE_FRY`. |
| **`mps`** (`go-darwinml/compute/mps`) | Calls **stateless** float `randomTensorWithShape:`, scales by $2^{32}$, casts to `Uint32`, and dummy-adds `1` to `state`. | **Ignores `state` completely** (seeds have no effect on MPS today) and loses 8 bits of mantissa entropy per `Uint32` (only 24 significand bits in `Float32`). MPS can instead fuse a counter-based PRNG directly from integer MPSGraph ops. |

### Unspecified Output `DType` Semantics
`compute/shapeinference` previously had no validation for `RNGBitGenerator`. When passed a float output shape (`Float32`):
- `gobackend` wrote raw random bytes into float memory (producing `NaN`/`Inf`),
- `compute-onnx` cast `Uint64 -> Float32` (producing values in $[0, 2^{64})$),
- `mps` and `mlx` special-cased floats to return $[0, 1)$.

Restricting `RNGBitGenerator` output `shape.DType` strictly to **unsigned integer bit types (`Uint8`, `Uint16`, `Uint32`, `Uint64`)** eliminates this ambiguity across all backends.

---

## 3. Core Design Decisions

### 3.1 Single 1D Integer `Value` State (XLA `Uint64` Convention for Explicit Algorithms)
Every practical PRNG packs its key and counter words into **a single 1D integer tensor**.
- Keeping `state` as a single `compute.Value` preserves ergonomic `(newState, values)` signatures, `While`/`If` loop state carrying, checkpoints, and `model.Store` variable management.
- **Explicit algorithms follow XLA/StableHLO's `Uint64` state convention** (e.g., `[3]uint64` for `RNGPhilox`, `[2]uint64` for `RNGThreeFry`, `RNGSplitMix64`, and `RNGPCG`, `[5]uint64` for `RNGChaCha8`). Backends whose internal arithmetic uses `uint32` words reinterpret the state via a zero-cost bitcast.
- Backends on 32-bit-only targets (e.g., ORT WebGPU where `Uint64` GPU buffers are unavailable) can advertise a 32-bit `DefaultRNGStateShape` (e.g., `[4]uint32`) for `RNGDefault`.

### 3.2 Canonical State Shape Per Algorithm (`RNGAlgorithm.StateShape`) vs. `Capabilities`
- **Explicit algorithms (`algo != RNGDefault`) have a single, canonical, backend-independent state shape** returned by `algo.StateShape()`. Two backends implementing `RNGPhilox` **always** use `[3]uint64` so states, checkpoints, and `model.Store` variables are directly interchangeable.
- **`Capabilities` never overrides state shapes for explicit algorithms.** `Capabilities.RNGAlgorithms` is a plain `map[RNGAlgorithm]bool` (matching `Operations` and `DTypes`). State-shape metadata on `Capabilities` exists solely for **`RNGDefault`**:
  1. Most backends' default RNG is one of the standard algorithms (e.g., `xla` and `go` set `Capabilities.DefaultRNGAlgorithm = RNGPhilox`), so `RNGDefault`'s state shape is simply `DefaultRNGAlgorithm.StateShape()`.
  2. If a backend uses a **backend-private native PRNG** (e.g., `mlx`'s `[2]uint32` key-splitting generator, or ORT WebGPU's 32-bit default), it leaves `DefaultRNGAlgorithm = RNGDefault` and sets `Capabilities.DefaultRNGStateShape` (e.g., `[2]uint32`).

### 3.3 Bit-Exact Golden Reference Vectors in `support/backendtest`
- Every backend claiming native support for an explicit algorithm (`Capabilities.RNGAlgorithms[algo] == true` for `algo != RNGDefault`) **must produce bit-for-bit identical `newState` and `values` outputs** for static shapes, verified against golden reference vectors in `compute/support/backendtest`.
- For `RNGPhilox` and `RNGThreeFry`, the canonical word ordering, counter increment, and bit packing match OpenXLA's `RngBitGeneratorExpander` (`Philox4x32-10` and `ThreeFry2x32-20`, verified directly against PJRT CPU).
- For `RNGDefault`, `backendtest` verifies statistical uniformity, determinism (same state $\to$ same output), and state advancement (`newState != state`, subsequent call produces distinct values).

### 3.4 Per-Algorithm `RNGState` Variables in `model.Store`
Instead of a single global `/#rngState` variable in `gomlx/ml/model.Store`, `Store` maintains **one RNG state variable per resolved algorithm**:
- Leaf variable names keep the `#` prefix (`/#rngState_<key>`, e.g., `/#rngState_philox`, `/#rngState_three_fry`) so code filtering internal `Store` variables via `strings.HasPrefix(v.Name(), "#")` continues to work unchanged.
- When `RNGDefault` is active on `Store`, `Store` checks `backend.Capabilities().ResolveRNGAlgorithm(RNGDefault)`:
  - If it resolves to a named algorithm (e.g., both `xla` and `go` resolve to `RNGPhilox`), **both backends share `/#rngState_philox`** and continue the exact same RNG sequence.
  - If it remains `RNGDefault` (a backend-private native PRNG, such as `mlx`), `Store` uses `/#rngState_default_<backend.Name()>` (e.g., `/#rngState_default_mlx`).
- **Result**: Using the same `model.Store` simultaneously or alternately across two backends with *different* RNG algorithms never overwrites state with incompatible shapes/bits, while backends sharing the *same* algorithm seamlessly share state and follow the same sequence.

### 3.5 Pure-Graph Fallback in `gomlx/core/graph`
If a user explicitly requests a counter-based portable algorithm (`RNGPhilox`, `RNGThreeFry`, `RNGSplitMix64`, `RNGChaCha8`) on a backend that returns `compute.ErrNotImplemented`, `gomlx/core/graph` automatically decomposes that algorithm into elementary integer/bitwise graph ops matching the golden reference specification. (Note: sequential `RNGPCG` is CPU-oriented and is not decomposed into parallel graph ops.)

### 3.6 Converting Random Integer Bits to Uniform Floats in $[0, 1)$
- **Backends do NOT convert random bits to floats**: `RNGBitGenerator` strictly outputs unsigned integer bit tensors (`Uint8`, `Uint16`, `Uint32`, `Uint64`). Keeping float conversion out of `RNGBitGenerator` avoids cross-backend discrepancies (`NaN` vs. $[0, 2^{64})$ vs. $[0, 1)$).
- **Graph helper `compute.BitsToUniformFloat` for `compute.Backend` users**:
  Direct users of `compute.Backend` / `compute.Function` (such as `support/backendtest`, benchmarks, or custom graph builders outside `gomlx`) should not have to re-derive IEEE-754 mantissa shifts per float dtype. Package `compute` provides a helper built purely on standard `Function` operators (`Constant`, `ShiftRightLogical`, `ConvertDType`, `Mul`):
  ```go
  func BitsToUniformFloat(fn Function, randomBits Value, floatDType dtypes.DType) (Value, error)
  ```
  Instead of converting `Uint64 -> Float64` directly and multiplying by $2^{-64}$ (which rounds $u \ge 2^{64} - 1024$ up to $1.0$, requiring clamping), `BitsToUniformFloat` right-shifts to keep the exact IEEE-754 significand width $p$ before converting and scaling by $2^{-p}$ (matching Go's `math/rand/v2.Float64()` / `Float32()`):
  - `Float64` ($p = 53$, requires $\ge 64$-bit input `Uint64`): `(bits >> 11) * 0x1p-53`
  - `Float32` ($p = 24$, requires $\ge 32$-bit input `Uint32` or `Uint64`): `(bits >> (W - 24)) * 0x1p-24`
  - `Float16` ($p = 11$, requires $\ge 16$-bit input `Uint16`/`Uint32`/`Uint64`): `(bits >> (W - 11)) * 0x1p-11`
  - `BFloat16` ($p = 8$, requires $\ge 16$-bit input `Uint16`/`Uint32`/`Uint64`): `(bits >> (W - 8)) * 0x1p-8`

  Every integer in $[0, 2^p - 1]$ converts **without rounding**, producing all $2^p$ equally-spaced dyadic rationals in $[0, 1)$ with exact uniform probability $2^{-p}$ and maximum value $1 - 2^{-p} < 1.0$. `gomlx/core/graph.RandomUniform` uses the exact same 3-op sequence.
- **Why no `FusedBitsToFloat` op in `compute.FusedOps`**:
  1. In neural network training (Dropout, weight initialization, Diffusion/FlowMatching noise), RNG + float conversion is $O(N)$ elementwise work, typically $< 1\%$ of step time compared to $O(N \cdot D)$ `DotGeneral` / `ConvGeneral` / `SDPA`.
  2. Graph-compiling backends (`xla`, `mps`, `onnx`) already fuse `RNGBitGenerator -> ShiftRightLogical -> ConvertDType -> Mul` into a single kernel automatically.
  3. On `gobackend`, a standalone `FusedBitsToFloat(bits, floatDType)` op would still materialize the intermediate `Uint32`/`Uint64` `bits` buffer in memory. If CPU profiling ever warrants it, `gobackend` can instead add an **internal compile-time pattern fusion** in `internal/gobackend/executable.go` (analogous to `fuseSingleConsumerDenseActivations`) matching `RNGBitGenerator -> ShiftRightLogical -> ConvertDType -> Mul` to write uniform floats directly in a single pass with zero `compute` API changes.

---

## 4. Proposed `github.com/gomlx/compute` API

### 4.1 `RNGAlgorithm` (`compute/rng.go`)

```go
// RNGAlgorithm specifies the Pseudo-Random Number Generator (PRNG) algorithm
// used by StandardOps.RNGBitGenerator.
//
// An RNG (Random Number Generator) in compute graphs is implemented as a pure,
// deterministic PRNG (Pseudo-Random Number Generator): given an explicit input
// state tensor, it produces both an updated state tensor and a tensor of
// uniformly distributed random bits.
type RNGAlgorithm int

//go:generate go tool enumer -type=RNGAlgorithm -trimprefix=RNG -output=gen_rngalgorithm_enumer.go -transform=snake rng.go

const (
	// RNGDefault (the zero value) instructs the backend to use its preferred native
	// Random Number Generator (RNG) algorithm.
	//
	// Guarantees:
	//   - Fast and native to the target backend.
	//   - Deterministic across runs on the same backend given the same initial state and shapes.
	//   - Not guaranteed to produce identical bitstreams across different backends (unless two
	//     backends resolve RNGDefault to the same Capabilities.DefaultRNGAlgorithm).
	//
	// State shape: queried via backend.Capabilities().RNGStateShape(RNGDefault).
	// Native backend support: all backends supporting OpTypeRNGBitGenerator ("go", "xla", "onnx", "mlx", "mps").
	RNGDefault RNGAlgorithm = iota

	// RNGPhilox specifies the counter-based Philox-4x32-10 Pseudo-Random Number Generator
	// (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011,
	// https://dl.acm.org/doi/10.1145/2063384.2063405), matching OpenXLA's PHILOX specification.
	//
	// Uses 10 rounds of 32-bit multiplication (mulhi/mullo) and XOR mixing over a 64-bit key
	// and 128-bit counter. Parallelizable across CPU workers/SIMD, GPUs, and TPUs.
	//
	// Canonical state shape: [3]uint64.
	// Native backend support: "xla", "go" (plus automatic graph decomposition fallback in GoMLX).
	RNGPhilox

	// RNGThreeFry specifies the counter-based ThreeFry-2x32-20 Pseudo-Random Number Generator
	// (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011,
	// https://dl.acm.org/doi/10.1145/2063384.2063405), matching OpenXLA's THREE_FRY specification.
	//
	// Uses 20 rounds of pure 32-bit additions, bitwise rotations, and XORs (requiring no
	// integer multiplications), making it fast and portable across CPUs, GPUs, and TPUs.
	//
	// Canonical state shape: [2]uint64.
	// Native backend support: "xla", "go", "onnx", "mps" (plus automatic graph decomposition fallback in GoMLX).
	RNGThreeFry

	// RNGSplitMix64 specifies a counter-based 64-bit SplitMix64 Pseudo-Random Number Generator
	// (Steele, Lea, and Flood, OOPSLA 2014).
	//
	// Uses two 64-bit shift-XOR-multiply mixing steps per output word, making it very compact
	// and fast when cryptographic strength is not required.
	//
	// Canonical state shape: [2]uint64 ([key, counter]).
	// Native backend support: "go", "onnx" (plus automatic graph decomposition fallback in GoMLX
	// on backends supporting Uint64).
	RNGSplitMix64

	// RNGPCG specifies the 128-bit PCG-DXSM (Permuted Congruential Generator — Double Xorshift
	// Multiply) sequential Pseudo-Random Number Generator matching Go's standard library
	// math/rand/v2.PCG.
	//
	// Note: PCG is a sequential state-transition generator rather than a counter-based PRNG,
	// so it is intended for CPU execution and is not decomposed into parallel GPU graph ops.
	//
	// Canonical state shape: [2]uint64 ([hi, lo]).
	// Native backend support: "go".
	RNGPCG

	// RNGChaCha8 specifies the counter-mode 8-round ChaCha8 stream cipher
	// Cryptographically Secure Pseudo-Random Number Generator (CSPRNG) (Bernstein, 2008).
	//
	// Generates 64-byte (16 x uint32) blocks independently from a 256-bit key and 64-bit block
	// counter, providing cryptographic-grade randomness while remaining fully parallelizable
	// across CPU threads/SIMD and GPUs.
	//
	// Canonical state shape: [5]uint64 ([key0, key1, key2, key3, blockCounter]).
	// Native backend support: "go" (plus automatic graph decomposition fallback in GoMLX).
	RNGChaCha8
)

// StateShape returns the canonical state tensor shape for an explicit RNGAlgorithm:
//   - RNGPhilox:     [3]uint64
//   - RNGThreeFry:   [2]uint64
//   - RNGSplitMix64: [2]uint64
//   - RNGPCG:        [2]uint64
//   - RNGChaCha8:    [5]uint64
//   - RNGDefault:    shapes.Invalid() (backend-dependent; use Capabilities.RNGStateShape)
func (a RNGAlgorithm) StateShape() shapes.Shape

// StateFromSeed deterministically initializes a flat []uint64 slice and canonical state Shape
// for an explicit RNGAlgorithm from a 64-bit seed (expanding key words via SplitMix64 and
// setting counter words to 0; for RNGPCG, seeding hi/lo identically to rand.NewPCG).
// Returns an error if called on RNGDefault; use Capabilities.RNGStateFromSeed instead.
func (a RNGAlgorithm) StateFromSeed(seed int64) (flat any, shape shapes.Shape, err error)

// BitsToUniformFloat builds standard compute.Function operations (Constant, ShiftRightLogical,
// ConvertDType, Mul) to convert uniformly distributed unsigned integer bits (Uint16, Uint32, or Uint64)
// into uniformly distributed floating-point values in [0.0, 1.0) of floatDType
// (BFloat16, Float16, Float32, or Float64).
func BitsToUniformFloat(fn Function, randomBits Value, floatDType dtypes.DType) (Value, error)
```

*(Compatibility note: `compute.RNGStateShape` is marked `Deprecated: use RNGPhilox.StateShape() or backend.Capabilities().RNGStateShape(...)` during transition.)*

### 4.2 `Capabilities` Additions (`compute/capabilities.go`)

```go
type Capabilities struct {
	...

	// RNGAlgorithms lists which Random Number Generator (RNG) algorithms are natively
	// supported by this backend (paralleling Operations and DTypes).
	// Backends supporting OpTypeRNGBitGenerator must at least set RNGAlgorithms[RNGDefault] = true.
	RNGAlgorithms map[RNGAlgorithm]bool

	// DefaultRNGAlgorithm indicates which concrete RNGAlgorithm this backend uses when
	// RNGDefault is requested (for example, RNGPhilox).
	//
	// If the backend uses a backend-private native PRNG that does not correspond to a
	// portable RNGAlgorithm constant, DefaultRNGAlgorithm remains RNGDefault and
	// DefaultRNGStateShape must specify its state tensor shape.
	DefaultRNGAlgorithm RNGAlgorithm

	// DefaultRNGStateShape specifies the state tensor shape for RNGDefault ONLY when
	// DefaultRNGAlgorithm == RNGDefault (a backend-private native PRNG).
	// When DefaultRNGAlgorithm != RNGDefault, DefaultRNGAlgorithm.StateShape() is used.
	DefaultRNGStateShape shapes.Shape
}

// SupportsRNG reports whether the backend natively implements the given RNGAlgorithm.
func (c Capabilities) SupportsRNG(algo RNGAlgorithm) bool

// ResolveRNGAlgorithm resolves RNGDefault to c.DefaultRNGAlgorithm (which may still be
// RNGDefault if the backend uses a backend-private native PRNG), or returns algo unchanged.
func (c Capabilities) ResolveRNGAlgorithm(algo RNGAlgorithm) RNGAlgorithm

// RNGStateShape returns the expected state tensor Shape for algo on this backend,
// or an error wrapping ErrNotImplemented if algo is not natively supported.
func (c Capabilities) RNGStateShape(algo RNGAlgorithm) (shapes.Shape, error)

// RNGStateFromSeed creates an initial host slice and state Shape from seed for algo on this backend.
func (c Capabilities) RNGStateFromSeed(algo RNGAlgorithm, seed int64) (flat any, shape shapes.Shape, err error)
```

*(Note: `Capabilities.Clone()` also deep-copies `RNGAlgorithms`.)*

### 4.3 `StandardOps.RNGBitGenerator` & `shapeinference.RNGBitGenerator`

In `compute/ops.go`:
```go
// RNGBitGenerator generates a tensor of the given shape filled with uniform random bits
// using the specified Random Number Generator (RNG) algorithm, returning the updated
// state tensor and the generated random bits.
//
//   - state: 1D state tensor matching backend.Capabilities().RNGStateShape(algorithm)
//     (which equals algorithm.StateShape() for all explicit algorithms).
//   - shape: target output shape. Its DType must be an unsigned integer type
//     (Uint8, Uint16, Uint32, or Uint64). Dimensions may be dynamic on backends
//     that support dynamic shapes.
//   - algorithm: PRNG algorithm to execute (RNGDefault for the backend's fastest native
//     generator, or an explicit algorithm such as RNGPhilox or RNGThreeFry for reproducibility).
//     If unsupported by the backend, returns an error wrapping ErrNotImplemented.
RNGBitGenerator(state Value, shape shapes.Shape, algorithm RNGAlgorithm) (newState Value, values Value, err error)
```

In `compute/shapeinference/shapeinference.go`:
```go
// RNGBitGenerator validates the input state shape and target output shape
// (requiring targetShape.DType to be Uint8, Uint16, Uint32, or Uint64), returning
// (newStateShape, valuesShape, err).
func RNGBitGenerator(
	expectedStateShape, stateShape, targetShape shapes.Shape,
) (newStateShape, valuesShape shapes.Shape, err error)
```

---

## 5. Proposed `github.com/gomlx/gomlx` API

### 5.1 `core/graph`: Algorithm-Tagged `rngState` Nodes, Float Conversion & Graph Fallback

1. **Aliases & State Constructors**:
   - Re-export `type RNGAlgorithm = compute.RNGAlgorithm` and all `RNG*` constants (`RNGDefault`, `RNGPhilox`, `RNGThreeFry`, `RNGSplitMix64`, `RNGPCG`, `RNGChaCha8`) in `core/graph`.
   - Graph-bound constructors (accepting optional `algo ...RNGAlgorithm`, defaulting to `g.DefaultRNGAlgorithm()`, which defaults to `RNGDefault`):
     - `RNGStateForGraph(g *Graph, algo ...RNGAlgorithm) *Node`
     - `RNGStateFromSeedForGraph(g *Graph, seed int64, algo ...RNGAlgorithm) *Node`
   - Tensor constructors:
     - `RNGStateForBackend(backend compute.Backend, algo ...RNGAlgorithm) (*tensors.Tensor, error)`
     - `RNGStateFromSeedForBackend(backend compute.Backend, seed int64, algo ...RNGAlgorithm) (*tensors.Tensor, error)`
     - `RNGState(algo ...RNGAlgorithm) (*tensors.Tensor, error)` and `RNGStateFromSeed(seed int64, algo ...RNGAlgorithm) (*tensors.Tensor, error)` (for explicit algorithms; defaults to `RNGPhilox` when called without a backend, since `RNGDefault` requires a backend to resolve its shape).

2. **Automatic Algorithm Propagation on `*Node` (Zero Call-Site Churn)**:
   - `*Node` carries an internal `rngAlgorithm RNGAlgorithm` field (plus `hasRNGAlgorithm bool`), set by `RNGState*ForGraph` (and `Store` variable parameter binding) and propagated onto `newRNGState` by `RNGBitGenerator` / `RandomUniform` / `RandomNormal` / `RandomIntN` / `RNGStateSplit`.
   - Existing calls (`RandomUniform(rngState, shape)`, `RandomNormal(rngState, shape)`, `RandomIntN(rngState, N, shape)`, `RNGStateSplit(rngState)`) keep their **exact current signatures** (with optional `algo ...RNGAlgorithm` override when `rngState` enters via an untagged `Exec` input parameter).

3. **Uniform Float Conversion in `RandomUniform`**:
   - `RandomUniform` uses the exact same right-shift + `ConvertDType` + `MulScalar(2^-p)` sequence as `compute.BitsToUniformFloat` (plus `StopGradient`), dropping the previous `Abs` and `MinScalar(..., Nextafter(1, 0))` workarounds.

4. **Decomposed Pure-Graph Fallback**:
   - `RNGBitGenerator(rngState *Node, shape shapes.Shape, algo ...RNGAlgorithm) (newState, values *Node)` attempts `backendRNGBitGenerator(rngState, shape, resolvedAlgo)` inside `exceptions.TryCatch`.
   - If it catches `compute.IsNotImplemented(err)` for a counter-based explicit algorithm (`RNGPhilox`, `RNGThreeFry`, `RNGSplitMix64`, `RNGChaCha8`), it falls back to a decomposed GoMLX graph implementation matching the golden reference specification.

### 5.2 `ml/model`: Per-Algorithm `RNGState` Variables in `Store`

1. **Hyperparameter & Configuration**:
   - `ParamRNGAlgorithm = "rng_algorithm"` (accepts `compute.RNGAlgorithm` or string name such as `"default"`, `"philox"`, `"three_fry"`, `"split_mix_64"`, `"pcg"`, `"cha_cha_8"` via `enumer`).
   - Convenience methods `store.WithRNGAlgorithm(algo compute.RNGAlgorithm) *Store` / `store.SetRNGAlgorithm(algo compute.RNGAlgorithm)`.

2. **Per-Algorithm Variable Resolution**:
   - Constants: `RNGStateVariablePrefix = "#rngState_"` (replacing single `#rngState`).
   - Given `g *graph.Graph` (`b := g.Backend()`):
     - Read configured `algo` from `Store` (default `compute.RNGDefault`).
     - Resolve via `resolved := b.Capabilities().ResolveRNGAlgorithm(algo)`.
     - Variable name at root scope `"/"`:
       - If `resolved != compute.RNGDefault`: `fmt.Sprintf("#rngState_%s", resolved)` (e.g., `/#rngState_philox`, `/#rngState_three_fry`), shape `resolved.StateShape()`.
       - If `resolved == compute.RNGDefault`: `fmt.Sprintf("#rngState_default_%s", b.Name())` (e.g., `/#rngState_default_mlx`), shape `b.Capabilities().DefaultRNGStateShape`.

3. **Lazy Seeding & Reset Behavior**:
   - `Store` holds `rngSeed int64` and `hasRNGSeed bool`.
   - `store.SetRNGStateFromSeed(seed int64)` records `rngSeed = seed, hasRNGSeed = true` and deletes/resets any already-materialized `#rngState_*` variables in `Store` so every algorithm re-initializes cleanly from `seed` on next use.
   - `store.ResetRNGState()` checks `ParamInitialSeed`; if unset, draws a 64-bit seed from OS CSPRNG (`crypto/rand`), records it as `rngSeed`, and clears any existing `#rngState_*` variables.
   - When `Store.Random*(g, ...)` accesses `/#rngState_<key>` for the first time, it materializes the variable tensor from `rngSeed` (initializing `rngSeed` via `ResetRNGState()` first if `!hasRNGSeed`).

---

## 6. Per-Backend Implementation Matrix

| Backend | `DefaultRNGAlgorithm` | Native `RNGAlgorithms` | Implementation Notes |
| :--- | :--- | :--- | :--- |
| **`go`** (`compute/gobackend`) | `RNGPhilox` | `RNGDefault`, `RNGPhilox`, `RNGThreeFry`, `RNGSplitMix64`, `RNGPCG`, `RNGChaCha8` | Implement parallelizable Philox-4x32-10, ThreeFry-2x32-20, SplitMix64, ChaCha8, and fixed `UnmarshalBinary`-equivalent PCG-DXSM; restrict output dtypes to `Uint8..Uint64`. |
| **`xla`** (`go-xla`) | `RNGPhilox` | `RNGDefault`, `RNGPhilox`, `RNGThreeFry` | Forward `algorithm` to `stablehlo.RNGBitGenerator` (`RNGDefault` resolves to `RNGPhilox`). `SplitMix64`/`ChaCha8` return `ErrNotImplemented` (handled by GoMLX graph fallback). |
| **`onnx`** (`compute-onnx`) | `RNGSplitMix64` (CPU/CUDA) / `RNGDefault` (`[4]uint32` ThreeFry on WebGPU) | `RNGDefault`, `RNGSplitMix64`, `RNGThreeFry` | Keep fast `SplitMix64` on 64-bit EPs and add 32-bit `ThreeFry-2x32-20` so WebGPU works without `Uint64` GPU buffers. |
| **`mlx`** (`go-darwinml/compute/mlx`) | `RNGDefault` (`DefaultRNGStateShape: [2]uint32`) | `RNGDefault` | Uses MLX's native `[2]uint32` `mlx::core::random::split` + `bits` directly with zero `[3]uint64` XOR-folding overhead; explicit counter-based algorithms run via GoMLX graph fallback. |
| **`mps`** (`go-darwinml/compute/mps`) | `RNGThreeFry` | `RNGDefault`, `RNGThreeFry` | Implement stateful counter-based `ThreeFry-2x32-20` in `Uint32` MPSGraph ops (fuses into a single Metal kernel), fixing the current stateless/24-bit-float bug on MPS. |

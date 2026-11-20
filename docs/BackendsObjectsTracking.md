# Backend Object Lifecycle Tracking & Safe Teardown

## 1. Motivation

Previously, calling `compute.Backend.Finalize()` put all derived objects (`compute.Buffer`, `compute.Builder`, `compute.Executable`) into an undefined state:

- **Concurrent execution races:** Calling `Backend.Finalize()` while another goroutine was inside `Executable.Execute()`, `Builder.Compile()`, or `Buffer.ToFlatData()` could tear down the underlying native context (e.g., `PJRT_Client_Destroy`, Metal `bridge.Context.Destroy`) mid-call and segfault.
- **Post-finalize undefined behavior:** Using a stale `Buffer`, `Builder`, or `Executable` after `Backend.Finalize()` had inconsistent behavior across backends—ranging from silent misuse to panics or C/C++ use-after-free crashes.
- **Native destruction order bugs (notably XLA/PJRT):** OpenXLA's PJRT C API expects `PJRT_Buffer` and `PJRT_LoadedExecutable` handles to be destroyed *before* `PJRT_Client_Destroy`. Previously in `go-xla`, `loadedExecutableC.Destroy()` did not check client validity at all, and `bufferWrapper.Destroy()` checked `client.IsValid()` without synchronization.
- **Delayed GPU/host memory reclamation:** In `compute-onnx`, `Backend.Finalize()` only set `isFinalized = true`, leaving compiled ONNX Runtime sessions (`Executable`s holding potentially gigabytes of GPU weights) alive until Go's garbage collector eventually finalized them.

## 2. Updated `compute` Lifecycle Contract

The `compute` package defines a sentinel error and helper:

- `compute.ErrBackendFinalized` / `compute.IsBackendFinalized(err)`

And tightens the lifecycle contract across `Backend`, `Builder`, `Executable`, and `Buffer`:

1. **`Backend.Finalize()` is goroutine-safe and idempotent.** It waits for all in-flight operations (`Execute`, `Compile`, buffer creation/transfers) to complete before releasing backend resources and any tracked child resources.
2. **No crashes after `Backend.Finalize()`.** Any subsequent operation on the `Backend` or on its derived `Builder`, `Executable`, and `Buffer` objects must return an error wrapping `compute.ErrBackendFinalized` (except read-only metadata like `Name()`/`String()`/`IsFinalized()`, and `Finalize()`).
   - **Shared buffer caveat (`Buffer.Data()`):** As documented on `compute.Buffer.Data`, Go cannot revoke a raw slice that was *already* returned to a caller before finalization. Callers must still not access previously obtained `Data()` slices after the buffer or backend is finalized (though backends avoid eagerly unmapping host-owned shared backing memory when possible).
3. **`Finalize()` on derived objects is idempotent and safe after `Backend.Finalize()`.** Calling `Buffer.Finalize()` or `Executable.Finalize()` multiple times, or after `Backend.Finalize()`, is a safe no-op returning `nil` (for `Buffer.Finalize()`).

---

## 3. Design & Trade-Offs

A natural first instinct is: *"have every `Backend` keep a `weak.Pointer` registry to every `Buffer`, `Builder`, and `Executable`, and finalize them all in `Backend.Finalize()`."*

However, separating **synchronization / safe errors** from **eager native teardown**, and distinguishing **hot-path Buffers** from **cold-path Executables and Builders**, leads to a much faster and simpler design.

### 3.1 Why NOT `weak.Pointer` Back-Links on Every `Buffer`?

Buffers are allocated and destroyed on the **hot path**—often tens of thousands to millions of times per second across concurrent goroutines (e.g., training steps, dynamic graphs, or multi-request serving). Keeping a `weak.Pointer` back-link for every buffer incurs three significant costs in Go:

1. **Runtime weak-handle overhead:** `weak.Make(buf)` allocates/attaches a runtime `specialWeakHandle` on the object's heap span, acquiring the Go runtime's span lock.
2. **No automatic registry eviction:** When a `Buffer` is garbage-collected without an explicit `Finalize()`, its `weak.Pointer` silently turns into `nil`, **leaving a dead entry in the backend's map/slice** unless you either run periodic sweeps or also register a `runtime.AddCleanup` callback on every buffer just to delete its map entry.
3. **Mutex contention on the backend:** Inserting and removing every buffer on a backend-level map introduces lock contention across goroutines.

For pure-Go backends like `gobackend`—where pooled buffers are acquired/released in ~20–50 ns with zero finalizers/cleanups—adding `weak.Pointer` + `runtime.AddCleanup` + map locking to every buffer would cause a severe hot-path regression for **zero** memory-safety benefit (pure-Go slices cannot crash when GC'd after `Backend.Finalize()`).

### 3.2 Shared Primitives in `compute/support/xsync`

To avoid duplicating concurrency boilerplate across backends, `compute/support/xsync` provides two small, reusable building blocks:

#### 1. `xsync.Lifecycle` (Synchronization & Post-Finalize Errors)
`xsync.Lifecycle` wraps a `sync.RWMutex` and an `atomic.Bool`:

- `IsFinalized() bool`: Lock-free atomic read.
- `Enter() bool`: Fast-checks `finalized`, acquires `mu.RLock()`, re-checks `finalized`, and returns `true` if the operation may proceed (otherwise releases `RUnlock()` and returns `false`).
- `Leave()`: Releases `mu.RUnlock()`.
- `Finalize(teardown func())`: Acquires exclusive `mu.Lock()` (blocking until every in-flight `Enter()` calls `Leave()`), marks `finalized = true`, and executes `teardown()` exactly once while holding the exclusive lock. Concurrent calls to `Finalize` wait until `teardown` finishes and return cleanly.

Because every `Buffer`, `Builder`, and `Executable` already holds a pointer to its parent `Backend`, `xsync.Lifecycle` gives **every** object safe concurrent teardown and clean `ErrBackendFinalized` errors with ~5–15 ns uncontended overhead and **zero** per-buffer allocations or registries.

> **Critical non-reentrancy rule:** Go's `sync.RWMutex` is **not reentrant**. If goroutine A holds `RLock()` (via `Enter()`), goroutine B calls `Finalize()` (blocking on `Lock()`), and goroutine A calls `Enter()` again before calling `Leave()`, Go's writer-priority `RWMutex` will **deadlock**.
>
> **Rule:** Acquire `lifecycle.Enter()` **only at public API entry points**. Internal helpers invoked while `Enter()` is already held (such as creating output buffers or destroying donated input buffers inside `Execute()`) must call internal `*Locked` variants that do not re-enter `lifecycle.Enter()`.

#### 2. `xsync.WeakTracker[T]` (Cold-Path Tracking for Executables & Builders)
For low-frequency objects (`Executable`s and native `Builder`s), `xsync.WeakTracker[T]` stores `weak.Pointer[T]` in a mutex-guarded map keyed by a monotonic `uint64` ID:

- `Track(obj *T) WeakHandle[T]` inserts `weak.Make(obj)` and attaches a `runtime.AddCleanup` capturing only the tracker and the `uint64` ID (never `obj`), so GC'd objects automatically remove their own entries without leaking memory.
- `handle.Untrack()` stops the cleanup and deletes the entry on explicit `Finalize()`.
- `Drain() []*T` marks the tracker closed, snapshots all still-live `*T` pointers, clears the map, and **releases the tracker mutex before returning**, so `Backend.Finalize()` can safely finalize each drained object without risking lock-order inversions or deadlocks.

### 3.3 Sharded Intrusive Wrapper List for Strict Native Buffers (`go-xla/pjrt`)

For backends whose native C API strictly requires buffers to be destroyed *before* the parent client is destroyed (specifically `go-xla/pjrt`), we can get deterministic pre-client teardown **without `weak.Pointer` and without any extra heap allocations**:

- `pjrt.Buffer` already separates the user-facing Go handle (`*pjrt.Buffer`) from the inner C wrapper struct (`*bufferWrapper`) used by `runtime.AddCleanup`.
- Because `bufferWrapper` has **no pointer back to `*pjrt.Buffer`**, `pjrt.Client` can maintain a **sharded intrusive doubly-linked list** (e.g., 32 shards, each with a `sync.Mutex` and head pointer) of `*bufferWrapper` directly via `prev, next *bufferWrapper` fields embedded in `bufferWrapper`.
- Holding `*bufferWrapper` in `pjrt.Client` does **not** keep `*pjrt.Buffer` alive! When `*pjrt.Buffer` becomes unreachable, its existing `runtime.AddCleanup` fires, unlinks `bufferWrapper` in $O(1)$, and calls `PJRT_Buffer_Destroy`.
- On `Client.Destroy()` (`Backend.Finalize()`), while holding `lifecycle.Finalize`'s exclusive lock:
  1. All tracked `*loadedExecutableC` wrappers (including their inner `*executableWrapper`) are drained and destroyed.
  2. All shards of `*bufferWrapper` are drained and their C `PJRT_Buffer` handles are destroyed **before** `PJRT_Client_Destroy`.
  3. For shared CPU buffers (`sharedRawStorage != nil`), `AlignedFree(sharedRawStorage)` is deferred to `Buffer.Destroy()` / GC cleanup rather than running inside `Client.Destroy()`, since the host allocation is independent of `PJRT_Client` and deferring avoids unmapping memory under a still-live `Buffer.Data()` slice.

Compared to `weak.Pointer` + a map:
- **0** `weak.Pointer` / runtime special allocations
- **0** extra heap allocations
- **0** map hashing, resizing, or tombstone leaks
- Single sharded mutex (~15 ns), negligible next to CGo `PJRT_Buffer` calls (~1–10 µs).

---

## 4. Per-Backend Strategy Summary

| Backend | Synchronization | Executables & Builders | Buffers |
| :--- | :--- | :--- | :--- |
| **`go`** (`compute`) | `xsync.Lifecycle` on `Backend` + `Executable` | `Lifecycle` check; `WeakTracker` not needed for pure-Go GC memory, though `Executable.Finalize()` uses its own `Lifecycle` to wait for in-flight runs | **Not tracked** (pure Go slices; check `Backend.IsFinalized()` / `Lifecycle`) |
| **`xla`** (`go-xla`) | `xsync.Lifecycle` on `pjrt.Client` (and `xla.Backend`) | Tracked on `pjrt.Client` via detached `loadedExecutableC` (destroys both `PJRT_Executable` and `PJRT_LoadedExecutable` before `PJRT_Client_Destroy`) | **Tracked via 32-shard intrusive `*bufferWrapper` list on `pjrt.Client`** (destroyed before `PJRT_Client_Destroy`; shared host memory freed on `Buffer` GC/Finalize) |
| **`onnx`** (`compute-onnx`) | `xsync.Lifecycle` on `Backend` | **`xsync.WeakTracker`** for native & web `Executable`s so `Backend.Finalize()` immediately destroys all ORT sessions & GPU weights | **Not tracked** (ORT `OrtValue` is owned independently of `Backend`; buffer ops check `Lifecycle` / `IsFinalized()`) |
| **`mps`** (`go-darwinml`) | `xsync.Lifecycle` on `Backend` | **`xsync.WeakTracker`** for `Executable`, `ExecutableWithCF`, and `Builder` (which owns a Metal `bridge.Context`), plus existing ping-pong pool cleanup | **Not tracked** (`MTLBuffer` is ARC-refcounted independently of `bridge.Context`; buffer ops check `Lifecycle` / `IsFinalized()`) |
| **`mlx`** (`go-darwinml`) | `xsync.Lifecycle` on `Backend` | **`xsync.WeakTracker`** for `Executable` and `Builder` (which owns an `mlxbridge.Tape`) | **Not tracked** (`mlx::core::array` is independently C++ refcounted; buffer ops check `Lifecycle` / `IsFinalized()`) |

---

## 5. Compliance Tests

`compute/support/backendtest` provides `TestFinalize(t *testing.T, newBackend func() (compute.Backend, error))` (in `compute/support/backendtest/finalize.go`), which every backend test suite invokes with a fresh backend factory to verify:

1. **Idempotent `Backend.Finalize()`** and `IsFinalized()` transitions.
2. **Post-finalize error handling** (no crashes/panics, returns `compute.IsBackendFinalized(err)` or clean error):
   - `Backend.BufferFromFlatData`, `Backend.NewSharedBuffer` (if supported)
   - `Builder.Compile` (for a builder created before `Backend.Finalize()`)
   - `Executable.Execute` (for an executable compiled before `Backend.Finalize()`)
   - `Buffer.ToFlatData`, `Buffer.Data` (if supported)
3. **Safe post-finalize cleanup**: `Buffer.Finalize()` returns `nil` and `Executable.Finalize()` is a safe no-op after `Backend.Finalize()`.
4. **Concurrent execution vs. `Backend.Finalize()`**: Multiple goroutines continuously executing compiled computations and creating/reading buffers while another goroutine calls `Backend.Finalize()`, verifying zero panics, zero data races (under `-race`), and that every post-finalize call either succeeds cleanly (started before `Finalize`) or returns a clean error (started after `Finalize`).

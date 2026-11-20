// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xsync

import (
	"runtime"
	"sync"
	"sync/atomic"
	"weak"
)

// Lifecycle coordinates concurrent operations against an idempotent Finalize teardown.
//
// In-flight operations call Enter/Leave (acquiring a shared RWMutex read lock), while
// Finalize acquires the exclusive write lock, waiting for all in-flight operations to
// finish before running the teardown callback exactly once.
//
// Zero value is ready to use. Must not be copied after first use.
//
// IMPORTANT: Go's sync.RWMutex is not reentrant. Never call Enter() again on the same
// goroutine before calling Leave(), or a concurrent Finalize() waiting on Lock() will deadlock.
type Lifecycle struct {
	mu        sync.RWMutex
	finalized atomic.Bool
}

// IsFinalized reports whether Finalize has started or completed.
// It is lock-free and safe to call concurrently from any goroutine.
func (l *Lifecycle) IsFinalized() bool {
	if l == nil {
		return true
	}
	return l.finalized.Load()
}

// Enter acquires a shared read lock for an in-flight operation.
//
// If Finalize has already been called, Enter returns false without holding any lock.
// If Enter returns true, the caller MUST call Leave exactly once when the operation finishes.
//
// Do not call Enter reentrantly from a goroutine that already holds Enter.
func (l *Lifecycle) Enter() bool {
	if l == nil || l.finalized.Load() {
		return false
	}
	l.mu.RLock()
	if l.finalized.Load() {
		l.mu.RUnlock()
		return false
	}
	return true
}

// Leave releases the shared read lock acquired by a successful Enter.
func (l *Lifecycle) Leave() {
	l.mu.RUnlock()
}

// Finalize waits for all in-flight Enter/Leave pairs to complete, marks the lifecycle as
// finalized, and invokes teardown exactly once while holding the exclusive write lock.
//
// Concurrent callers to Finalize block until teardown completes and then return without
// invoking teardown again.
func (l *Lifecycle) Finalize(teardown func()) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.finalized.Swap(true) {
		return
	}
	if teardown != nil {
		teardown()
	}
}

// WeakTracker tracks live objects of type *T without preventing them from being garbage-collected.
//
// Intended for low-frequency objects such as compiled Executables and Builders (NOT hot-path Buffers).
// Tracked objects are automatically removed from the tracker if garbage-collected before Untrack is called.
//
// Zero value is ready to use. Must not be copied after first use.
type WeakTracker[T any] struct {
	mu      sync.Mutex
	nextID  uint64
	entries map[uint64]weak.Pointer[T]
	closed  bool
}

// WeakHandle represents a tracked registration in a WeakTracker.
// Calling Untrack on a nil handle or multiple times is a safe no-op.
type WeakHandle[T any] struct {
	tracker atomic.Pointer[WeakTracker[T]]
	id      uint64
	cleanup runtime.Cleanup
}

// Track registers obj in the tracker and returns a handle that should be passed to Untrack
// when obj is explicitly finalized.
//
// If obj is nil or the tracker has already been drained, Track returns nil.
func (t *WeakTracker[T]) Track(obj *T) *WeakHandle[T] {
	if t == nil || obj == nil {
		return nil
	}
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil
	}
	if t.entries == nil {
		t.entries = make(map[uint64]weak.Pointer[T])
	}
	t.nextID++
	id := t.nextID
	t.entries[id] = weak.Make(obj)
	t.mu.Unlock()

	h := &WeakHandle[T]{id: id}
	h.tracker.Store(t)
	h.cleanup = runtime.AddCleanup(obj, func(entryID uint64) {
		t.remove(entryID)
	}, id)
	return h
}

func (t *WeakTracker[T]) remove(id uint64) {
	t.mu.Lock()
	delete(t.entries, id)
	t.mu.Unlock()
}

// Untrack removes the object from its WeakTracker and cancels its GC cleanup.
// Safe to call concurrently, multiple times, or on a nil handle.
func (h *WeakHandle[T]) Untrack() {
	if h == nil {
		return
	}
	t := h.tracker.Swap(nil)
	if t == nil {
		return
	}
	h.cleanup.Stop()
	t.remove(h.id)
}

// Drain marks the tracker closed, clears all entries, and returns strong pointers to all
// currently reachable tracked objects (omitting any already collected by GC).
//
// Drain releases the tracker's mutex before returning so callers can safely finalize each
// returned object (which may call Untrack) without risking deadlocks.
func (t *WeakTracker[T]) Drain() []*T {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	t.closed = true
	if len(t.entries) == 0 {
		t.entries = nil
		t.mu.Unlock()
		return nil
	}
	live := make([]*T, 0, len(t.entries))
	for _, wp := range t.entries {
		if obj := wp.Value(); obj != nil {
			live = append(live, obj)
		}
	}
	t.entries = nil
	t.mu.Unlock()
	return live
}

// Len returns the current number of entries in the tracker (including any not yet pruned by GC cleanup).
func (t *WeakTracker[T]) Len() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	n := len(t.entries)
	t.mu.Unlock()
	return n
}

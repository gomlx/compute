// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xsync

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// trackedItem is >= 16 bytes so Go's tiny allocator will never co-locate separate
// instances in the same 16-byte heap block (which would delay runtime.AddCleanup).
type trackedItem struct {
	id  int
	pad [4]int
}

//go:noinline
func trackEphemeral(tracker *WeakTracker[trackedItem]) {
	ephemeral := &trackedItem{id: 3}
	_ = tracker.Track(ephemeral)
}

func TestLifecycle(t *testing.T) {
	t.Run("BasicAndIdempotent", func(t *testing.T) {
		var lc Lifecycle
		if lc.IsFinalized() {
			t.Fatalf("new Lifecycle should not be finalized")
		}
		if !lc.Enter() {
			t.Fatalf("Enter should succeed before Finalize")
		}
		lc.Leave()

		var teardownCalls atomic.Int32
		lc.Finalize(func() {
			teardownCalls.Add(1)
		})
		if !lc.IsFinalized() {
			t.Fatalf("Lifecycle should be finalized after Finalize")
		}
		if lc.Enter() {
			lc.Leave()
			t.Fatalf("Enter should return false after Finalize")
		}
		// Second Finalize should be a no-op.
		lc.Finalize(func() {
			teardownCalls.Add(1)
		})
		if got := teardownCalls.Load(); got != 1 {
			t.Fatalf("expected teardown to run once, got %d", got)
		}
	})

	t.Run("WaitsForInFlight", func(t *testing.T) {
		var lc Lifecycle
		entered := make(chan struct{})
		release := make(chan struct{})
		finalizeStarting := make(chan struct{})
		teardownStarted := make(chan struct{})
		teardownDone := make(chan struct{})

		if !lc.Enter() {
			t.Fatalf("Enter failed")
		}
		go func() {
			close(entered)
			<-release
			lc.Leave()
		}()
		<-entered

		go func() {
			close(finalizeStarting)
			lc.Finalize(func() {
				close(teardownStarted)
			})
			close(teardownDone)
		}()
		<-finalizeStarting
		// Note: testing/synctest cannot be used here because locking a sync.RWMutex
		// is not a durably blocking operation (synctest.Wait would block forever).
		for range 100 {
			runtime.Gosched()
		}

		select {
		case <-teardownStarted:
			t.Fatalf("teardown ran while in-flight operation still held Enter()")
		default:
		}

		close(release)
		<-teardownDone
	})

	t.Run("ConcurrentStress", func(t *testing.T) {
		var lc Lifecycle
		var inFlight atomic.Int32
		var teardownCalled atomic.Bool
		var wg sync.WaitGroup
		var startedWG sync.WaitGroup

		const workers = 16
		startedWG.Add(workers)
		for range workers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				signaled := false
				defer func() {
					if !signaled {
						startedWG.Done()
					}
				}()
				for range 2000 {
					if !lc.Enter() {
						return
					}
					if teardownCalled.Load() {
						t.Errorf("Enter succeeded after teardown started")
					}
					inFlight.Add(1)
					if !signaled {
						signaled = true
						startedWG.Done()
					}
					runtime.Gosched()
					inFlight.Add(-1)
					lc.Leave()
				}
			}()
		}

		startedWG.Wait()
		lc.Finalize(func() {
			if n := inFlight.Load(); n != 0 {
				t.Errorf("teardown ran with %d in-flight operations", n)
			}
			teardownCalled.Store(true)
		})
		wg.Wait()
		if !teardownCalled.Load() {
			t.Fatalf("teardown was never called")
		}
	})

	t.Run("WeakTracker", func(t *testing.T) {
		var tracker WeakTracker[trackedItem]

		a := &trackedItem{id: 1}
		b := &trackedItem{id: 2}
		ha := tracker.Track(a)
		hb := tracker.Track(b)

		// Track an ephemeral object in a non-inlined frame and let GC collect it.
		// Note: testing/synctest cannot be used here because runtime.AddCleanup callbacks
		// run outside of any synctest bubble.
		trackEphemeral(&tracker)
		for range 3 {
			runtime.GC()
		}
		if got := tracker.Len(); got != 2 {
			t.Fatalf("expected GC cleanup to prune ephemeral entry leaving 2, got %d", got)
		}

		// Untrack a explicitly (idempotent).
		ha.Untrack()
		ha.Untrack()
		if got := tracker.Len(); got != 1 {
			t.Fatalf("expected 1 entry after Untrack(a), got %d", got)
		}

		// Drain should return only b.
		live := tracker.Drain()
		if len(live) != 1 || live[0] != b {
			t.Fatalf("expected Drain to return [b], got %v", live)
		}
		// Untrack after Drain must be a safe no-op.
		hb.Untrack()

		// Tracking after Drain returns nil.
		c := &trackedItem{id: 4}
		if hc := tracker.Track(c); hc != nil {
			t.Fatalf("expected Track after Drain to return nil, got %v", hc)
		}

		runtime.KeepAlive(a)
		runtime.KeepAlive(b)
	})
}

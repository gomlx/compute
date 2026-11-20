// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xsync

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLifecycle_BasicAndIdempotent(t *testing.T) {
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
}

func TestLifecycle_WaitsForInFlight(t *testing.T) {
	var lc Lifecycle
	entered := make(chan struct{})
	release := make(chan struct{})
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
		lc.Finalize(func() {
			close(teardownStarted)
		})
		close(teardownDone)
	}()

	select {
	case <-teardownStarted:
		t.Fatalf("teardown ran while in-flight operation still held Enter()")
	case <-time.After(20 * time.Millisecond):
	}

	close(release)
	select {
	case <-teardownDone:
	case <-time.After(2 * time.Second):
		t.Fatalf("Finalize did not complete after in-flight operation left")
	}
}

func TestLifecycle_ConcurrentStress(t *testing.T) {
	var lc Lifecycle
	var inFlight atomic.Int32
	var teardownCalled atomic.Bool
	var wg sync.WaitGroup

	const workers = 16
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 2000 {
				if !lc.Enter() {
					return
				}
				if teardownCalled.Load() {
					t.Errorf("Enter succeeded after teardown started")
				}
				inFlight.Add(1)
				runtime.Gosched()
				inFlight.Add(-1)
				lc.Leave()
			}
		}()
	}

	time.Sleep(2 * time.Millisecond)
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
}

func TestWeakTracker_TrackUntrackDrainAndGC(t *testing.T) {
	type item struct {
		id int
	}
	var tracker WeakTracker[item]

	a := &item{id: 1}
	b := &item{id: 2}
	ha := tracker.Track(a)
	hb := tracker.Track(b)

	// Track an ephemeral object and let GC collect it.
	func() {
		ephemeral := &item{id: 3}
		_ = tracker.Track(ephemeral)
	}()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		if tracker.Len() == 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
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
	c := &item{id: 4}
	if hc := tracker.Track(c); hc != nil {
		t.Fatalf("expected Track after Drain to return nil, got %v", hc)
	}

	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
}

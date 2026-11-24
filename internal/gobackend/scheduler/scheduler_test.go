// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package scheduler

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/compute/support/xsync"
)

func TestScheduler_Saturate(t *testing.T) {
	// Test saturation where all workers (caller + helpers) participate.
	sched := New()
	wantTasks := min(5, runtime.GOMAXPROCS(0))
	sched.SetMaxParallelism(wantTasks)

	var count atomic.Int32
	doneNewTasks := xsync.NewLatch()
	doneTest := xsync.NewLatch()

	go func() {
		sched.Saturate(func() {
			got := count.Add(1)
			runtime.Gosched()
			if int(got) == wantTasks {
				doneNewTasks.Trigger()
				return
			}
			doneNewTasks.Wait()
		})
		doneTest.Trigger()
	}()

	select {
	case <-doneTest.WaitChan():
		// Success
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("Timeout before all tasks were executed (count=%d, want=%d).", count.Load(), wantTasks)
	}
	if int(count.Load()) != wantTasks {
		t.Fatalf("Expected %d tasks, got %d", wantTasks, count.Load())
	}

	// Test No Parallelism
	schedNoPar := New()
	schedNoPar.SetMaxParallelism(0)
	count.Store(0)
	schedNoPar.Saturate(func() { count.Add(1) })
	if ok, diff := testutil.IsEqual(int32(1), count.Load()); !ok {
		t.Errorf("Scheduler with parallelism 0 result mismatch:\n%s", diff)
	}

	// Test Unlimited
	schedUnlimited := New()
	schedUnlimited.SetMaxParallelism(-1)
	wantUnlimited := runtime.GOMAXPROCS(0)
	var started atomic.Int32
	doneUnlimited := xsync.NewLatch()
	doneTestUnlimited := xsync.NewLatch()

	go func() {
		schedUnlimited.Saturate(func() {
			got := started.Add(1)
			runtime.Gosched()
			if int(got) == wantUnlimited {
				doneUnlimited.Trigger()
				return
			}
			doneUnlimited.Wait()
		})
		doneTestUnlimited.Trigger()
	}()

	select {
	case <-doneTestUnlimited.WaitChan():
		// Success
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Timeout before all unlimited tasks were executed.")
	}

	if runtime.GOMAXPROCS(0) > 1 && started.Load() <= 1 {
		t.Errorf("Expected more than 1 started task for unlimited parallelism, got %d", started.Load())
	}
	if int(started.Load()) != wantUnlimited {
		t.Errorf("Expected started %d to match wantUnlimited %d", started.Load(), wantUnlimited)
	}
}

func TestScheduler_ParallelizeChunks(t *testing.T) {
	sched := New()
	sched.SetMaxParallelism(4)

	const total = 1000
	out := make([]int, total)
	sched.ParallelizeChunks(total, 64, func(start, end int) {
		for i := start; i < end; i++ {
			out[i] = i * 2
		}
	})

	for i := range total {
		if out[i] != i*2 {
			t.Fatalf("out[%d] = %d, want %d", i, out[i], i*2)
		}
	}
}

func TestScheduler_InterCallScaling(t *testing.T) {
	sched := New()
	targetP := min(4, runtime.GOMAXPROCS(0))
	sched.SetMaxParallelism(targetP)

	if got := sched.EffectiveParallelism(); got != targetP {
		t.Fatalf("EffectiveParallelism() = %d, want %d", got, targetP)
	}

	// Simulate targetP concurrent Execute() calls.
	var exits []func()
	for range targetP {
		exits = append(exits, sched.EnterExecution())
	}
	if got := sched.EffectiveParallelism(); got != 1 {
		t.Fatalf("With %d active calls, EffectiveParallelism() = %d, want 1", targetP, got)
	}

	for _, exit := range exits {
		exit()
	}
	if got := sched.EffectiveParallelism(); got != targetP {
		t.Fatalf("After exiting calls, EffectiveParallelism() = %d, want %d", got, targetP)
	}
}

func TestScheduler_PhysicalCores(t *testing.T) {
	phys := PhysicalCores()
	if phys < 1 {
		t.Fatalf("PhysicalCores() = %d, want >= 1", phys)
	}
	def := DefaultPhysicalParallelism()
	if def < 1 || def > runtime.GOMAXPROCS(0) || def > phys {
		t.Fatalf("DefaultPhysicalParallelism() = %d out of bounds (phys=%d, GOMAXPROCS=%d)",
			def, phys, runtime.GOMAXPROCS(0))
	}
	t.Logf("PhysicalCores=%d, GOMAXPROCS=%d, DefaultPhysicalParallelism=%d",
		phys, runtime.GOMAXPROCS(0), def)
}

func TestScheduler_ExecutionThrottling(t *testing.T) {
	sched := New()
	limit := min(4, runtime.GOMAXPROCS(0))
	sched.SetMaxParallelism(limit)

	const totalRequests = 24
	var active atomic.Int32
	var maxActive atomic.Int32
	var completed atomic.Int32
	var wg sync.WaitGroup

	for range totalRequests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			exit := sched.EnterExecution()
			defer exit()
			cur := active.Add(1)
			for {
				prev := maxActive.Load()
				if cur <= prev || maxActive.CompareAndSwap(prev, cur) {
					break
				}
			}
			time.Sleep(2 * time.Millisecond)
			active.Add(-1)
			completed.Add(1)
		}()
	}
	wg.Wait()

	if got := completed.Load(); got != totalRequests {
		t.Fatalf("completed = %d, want %d", got, totalRequests)
	}
	if got := int(maxActive.Load()); got > limit {
		t.Fatalf("max concurrent active executions = %d, exceeded limit %d", got, limit)
	}
	if limit > 1 && int(maxActive.Load()) < 2 {
		t.Fatalf("expected concurrent executions up to limit=%d, got maxActive=%d", limit, maxActive.Load())
	}
}

func TestScheduler_NestedInterOpAndIntraOp_NoDeadlock(t *testing.T) {
	for _, maxP := range []int{1, 2, 4, 8} {
		sched := New()
		sched.SetMaxParallelism(maxP)

		const numInterOpTasks = 16
		const numIntraOpChunks = 32
		var totalCompleted atomic.Int32
		var wg sync.WaitGroup

		for range numInterOpTasks {
			wg.Add(1)
			interTask := func() {
				defer wg.Done()
				tasks := make([]func(), numIntraOpChunks)
				for c := range numIntraOpChunks {
					tasks[c] = func() {
						totalCompleted.Add(1)
					}
				}
				sched.RunIntraOpTasks(tasks)
			}
			if !sched.SubmitInterOp(interTask) {
				interTask()
			}
		}

		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatalf("Deadlock detected with maxParallelism=%d", maxP)
		}

		want := int32(numInterOpTasks * numIntraOpChunks)
		if got := totalCompleted.Load(); got != want {
			t.Fatalf("maxParallelism=%d: totalCompleted = %d, want %d", maxP, got, want)
		}
	}
}

func TestScheduler_WholeChipGate(t *testing.T) {
	sched := New()
	targetP := min(4, runtime.GOMAXPROCS(0))
	if targetP < 2 {
		t.Skip("requires GOMAXPROCS >= 2")
	}
	sched.SetMaxParallelism(targetP)

	// Simulate targetP concurrent active Execute() calls so EffectiveParallelism() == 1,
	// while UnscaledParallelism() remains targetP.
	numCalls := targetP
	exits := make([]func(), numCalls)
	for i := range numCalls {
		exits[i] = sched.EnterExecution()
	}
	defer func() {
		for _, exit := range exits {
			exit()
		}
	}()

	if got := sched.EffectiveParallelism(); got != 1 {
		t.Fatalf("EffectiveParallelism() = %d, want 1", got)
	}
	if got := sched.UnscaledParallelism(); got != targetP {
		t.Fatalf("UnscaledParallelism() = %d, want %d", got, targetP)
	}

	var activeHolders atomic.Int32
	var maxHolders atomic.Int32
	var wg sync.WaitGroup
	for range numCalls {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 5 {
				acquired := sched.AcquireWholeChip()
				if !acquired {
					t.Errorf("expected AcquireWholeChip() == true")
				}
				cur := activeHolders.Add(1)
				for {
					prev := maxHolders.Load()
					if cur <= prev || maxHolders.CompareAndSwap(prev, cur) {
						break
					}
				}
				if !sched.IsWholeChipHeld() {
					t.Errorf("expected IsWholeChipHeld() == true while holding gate")
				}
				var workersRan atomic.Int32
				sched.SaturateUpTo(targetP, func() {
					workersRan.Add(1)
				})
				if workersRan.Load() < 1 {
					t.Errorf("expected SaturateUpTo to run at least 1 worker, got %d", workersRan.Load())
				}
				activeHolders.Add(-1)
				sched.ReleaseWholeChip(acquired)
			}
		}()
	}
	wg.Wait()

	if got := maxHolders.Load(); got != 1 {
		t.Fatalf("max concurrent whole-chip holders = %d, want 1", got)
	}
	if sched.IsWholeChipHeld() {
		t.Fatalf("expected IsWholeChipHeld() == false after all callers released gate")
	}
}

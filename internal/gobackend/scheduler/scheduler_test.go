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
	sched.SetMaxParallelism(4)

	if got := sched.EffectiveParallelism(); got != min(4, runtime.GOMAXPROCS(0)) {
		t.Fatalf("EffectiveParallelism() = %d, want %d", got, min(4, runtime.GOMAXPROCS(0)))
	}

	// Simulate 4 concurrent Execute() calls.
	var exits []func()
	for range 4 {
		exits = append(exits, sched.EnterExecution())
	}
	if got := sched.EffectiveParallelism(); got != 1 {
		t.Fatalf("With 4 active calls, EffectiveParallelism() = %d, want 1", got)
	}

	for _, exit := range exits {
		exit()
	}
	if got := sched.EffectiveParallelism(); got != min(4, runtime.GOMAXPROCS(0)) {
		t.Fatalf("After exiting calls, EffectiveParallelism() = %d, want %d", got, min(4, runtime.GOMAXPROCS(0)))
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

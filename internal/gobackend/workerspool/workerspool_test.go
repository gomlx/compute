package workerspool

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/compute/support/xsync"
)

func TestPool_Saturate(t *testing.T) {
	// Test saturation.
	pool := New()
	wantTasks := 5
	pool.SetMaxParallelism(wantTasks)

	var count atomic.Int32
	doneNewTasks := xsync.NewLatch()
	doneTest := xsync.NewLatch()

	go func() {
		pool.Saturate(func() {
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
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout before all tasks were executed.")
	}
	if int(count.Load()) != wantTasks {
		t.Fatalf("Expected %d tasks, got %d", wantTasks, count.Load())
	}

	// Test No Parallelism
	pool.SetMaxParallelism(0)
	count.Store(0)
	pool.Saturate(func() { count.Add(1) })
	if ok, diff := testutil.IsEqual(int32(1), count.Load()); !ok {
		t.Errorf("Pool with parallelism 0 result mismatch:\n%s", diff)
	}

	// Test Unlimited
	pool.SetMaxParallelism(-1)
	wantUnlimited := runtime.GOMAXPROCS(0)
	var started atomic.Int32
	doneUnlimited := xsync.NewLatch()
	doneTestUnlimited := xsync.NewLatch()

	go func() {
		pool.Saturate(func() {
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
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout before all unlimited tasks were executed.")
	}

	if runtime.GOMAXPROCS(0) > 1 && started.Load() <= 1 {
		t.Errorf("Expected more than 1 started task for unlimited parallelism, got %d", started.Load())
	}
	if int(started.Load()) != wantUnlimited {
		t.Errorf("Expected started %d to match wantUnlimited %d", started.Load(), wantUnlimited)
	}
}

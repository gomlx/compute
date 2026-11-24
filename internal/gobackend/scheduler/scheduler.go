// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package scheduler implements the runtime execution scheduler for the Go compute backend.
//
// It manages a persistent pool of worker goroutines and schedules tasks across three
// hierarchical levels of parallelism:
//  1. Inter-Call: across concurrent calls to Executable.Execute() (highest priority).
//  2. Intra-Op: across contiguous memory chunks within a single operation.
//  3. Inter-Op: across independent graph branches within a single Execute() call (lowest priority).
//
// When an operation spawns intra-op subtasks (via RunIntraOpTasks, ParallelizeChunks, or Saturate),
// the calling goroutine immediately participates in executing its own op's subtasks (work-helping)
// before waiting on the completion barrier. This guarantees forward progress and prevents deadlocks
// even under nested inter-op and intra-op parallelism.
//
// See design details in https://github.com/gomlx/compute/blob/main/docs/GoBackendScheduler.md
package scheduler

import (
	"runtime"
	"sync"
	"sync/atomic"
)

const defaultQueueSize = 1024

var cachedPhysicalCores = sync.OnceValue(detectPhysicalCores)

// PhysicalCores returns the detected number of online physical CPU cores on this machine
// (excluding SMT/Hyper-Threading sibling threads on Linux and macOS; falls back to runtime.NumCPU()
// on other platforms).
func PhysicalCores() int {
	return max(cachedPhysicalCores(), 1)
}

// DefaultPhysicalParallelism returns min(PhysicalCores(), runtime.GOMAXPROCS(0)),
// respecting container/cgroup GOMAXPROCS limits while avoiding SMT oversubscription.
func DefaultPhysicalParallelism() int {
	procs := max(runtime.GOMAXPROCS(0), 1)
	return min(PhysicalCores(), procs)
}

// Scheduler manages persistent worker goroutines and dispatches intra-op and inter-op tasks.
type Scheduler struct {
	// maxParallelism is the configured target limit of parallel workers:
	//   > 0: explicit limit (also throttles concurrent EnterExecution calls to maxInterCallParallelism)
	//   == 0: parallelism disabled
	//   < 0: unlimited (uses runtime.GOMAXPROCS(0))
	maxParallelism atomic.Int32

	// maxIntraCallParallelism optionally caps the maximum number of workers used inside a single
	// Execute() call (intra-op + inter-op) without reducing the inter-call concurrency limit in
	// EnterExecution(). Values <= 0 disable the separate cap (defaulting to maxParallelism).
	maxIntraCallParallelism atomic.Int32

	// gomaxprocs caches runtime.GOMAXPROCS(0) to avoid acquiring runtime.sched.lock
	// on every UnscaledParallelism() / EffectiveParallelism() call.
	gomaxprocs atomic.Int32

	// numActiveCalls tracks the number of concurrent top-level Executable.Execute() calls
	// for inter-call parallelism scaling.
	numActiveCalls atomic.Int32

	// execGateMu / execGateCond / activeExecs throttle concurrent top-level Executable.Execute()
	// calls to UnscaledParallelism() when maxParallelism > 0, preventing oversubscription when
	// callers spawn more concurrent requests than available physical cores.
	execGateMu   sync.Mutex
	execGateCond *sync.Cond
	activeExecs  int

	// maxInterOpWorkers tracks the maximum number of persistent background workers that
	// may concurrently execute inter-op tasks, reserving at least 1 persistent worker
	// (when numWorkers > 1) exclusively for intra-op tasks.
	maxInterOpWorkers int32

	// activeInterOpWorkers tracks the number of persistent workers currently executing
	// an inter-op task submitted via SubmitInterOp.
	activeInterOpWorkers atomic.Int32

	// wholeChipHeld tracks whether a large operation currently holds the whole-chip gate.
	wholeChipHeld atomic.Int32

	initOnce      sync.Once
	intraOpQueue  chan func()
	interOpQueue  chan func()
	wholeChipGate chan struct{}
}

// New returns a new Scheduler with the default parallelism set to runtime.GOMAXPROCS(0).
func New() *Scheduler {
	procs := max(runtime.GOMAXPROCS(0), 1)
	s := &Scheduler{}
	s.maxParallelism.Store(int32(procs))
	s.gomaxprocs.Store(int32(procs))
	s.execGateCond = sync.NewCond(&s.execGateMu)
	return s
}

func (s *Scheduler) cachedGOMAXPROCS() int {
	if procs := int(s.gomaxprocs.Load()); procs > 0 {
		return procs
	}
	procs := max(runtime.GOMAXPROCS(0), 1)
	s.gomaxprocs.Store(int32(procs))
	return procs
}

// ensureWorkersStarted lazily starts the persistent worker goroutines on first parallel use.
func (s *Scheduler) ensureWorkersStarted() {
	s.initOnce.Do(func() {
		s.intraOpQueue = make(chan func(), defaultQueueSize)
		s.interOpQueue = make(chan func(), defaultQueueSize)
		s.wholeChipGate = make(chan struct{}, 1)
		numWorkers := max(s.cachedGOMAXPROCS(), int(s.maxParallelism.Load()), 1)
		if numWorkers > 1 {
			s.maxInterOpWorkers = int32(numWorkers - 1)
		} else {
			s.maxInterOpWorkers = 1
		}
		for range numWorkers {
			go s.workerLoop()
		}
		for range s.maxInterOpWorkers {
			go s.interOpWorkerLoop()
		}
	})
}

// workerLoop is the main loop of each persistent intra-op worker goroutine.
// Using a single channel receive on intraOpQueue enables Go's direct channel handoff fast-path
// and avoids runtime.selectgo / runtime.sellock contention on the critical intra-op path.
func (s *Scheduler) workerLoop() {
	for task := range s.intraOpQueue {
		task()
	}
}

// interOpWorkerLoop executes coarse-grained inter-op branch tasks on a separate pool so
// parked inter-op branch executors (e.g., waiting on AcquireWholeChip or child barriers)
// never deplete the intra-op worker pool.
func (s *Scheduler) interOpWorkerLoop() {
	for task := range s.interOpQueue {
		task()
	}
}

// IsEnabled returns whether parallelism is enabled (maxParallelism != 0).
func (s *Scheduler) IsEnabled() bool {
	return s.maxParallelism.Load() != 0
}

// IsUnlimited returns whether parallelism is unlimited (maxParallelism < 0).
func (s *Scheduler) IsUnlimited() bool {
	return s.maxParallelism.Load() < 0
}

// MaxParallelism returns the configured target for parallelism.
// If set to 0, parallelism is disabled.
// If set to -1, parallelism is unlimited.
func (s *Scheduler) MaxParallelism() int {
	return int(s.maxParallelism.Load())
}

// SetMaxParallelism sets the target maxParallelism.
//
// This should typically be configured before workers start executing; calling it dynamically
// safely wakes any executions parked in EnterExecution.
func (s *Scheduler) SetMaxParallelism(maxParallelism int) {
	s.execGateMu.Lock()
	s.maxParallelism.Store(int32(maxParallelism))
	s.gomaxprocs.Store(int32(max(runtime.GOMAXPROCS(0), 1)))
	if s.execGateCond != nil {
		s.execGateCond.Broadcast()
	}
	s.execGateMu.Unlock()
}

// MaxIntraCallParallelism returns the configured per-call (intra-op + inter-op) worker cap.
// A value <= 0 means no separate cap (same limit as MaxParallelism).
func (s *Scheduler) MaxIntraCallParallelism() int {
	return int(s.maxIntraCallParallelism.Load())
}

// SetMaxIntraCallParallelism sets an upper bound on the number of workers used within any single
// Execute() call (intra-op + inter-op) without reducing inter-call concurrency in EnterExecution().
// Pass 0 (the default) to use the same limit as MaxParallelism.
func (s *Scheduler) SetMaxIntraCallParallelism(maxIntraCall int) {
	if maxIntraCall < 0 {
		maxIntraCall = 0
	}
	s.maxIntraCallParallelism.Store(int32(maxIntraCall))
}

// maxInterCallParallelism returns the total concurrency budget across all concurrent Execute() calls
// (not capped by maxIntraCallParallelism).
func (s *Scheduler) maxInterCallParallelism() int {
	maxP := int(s.maxParallelism.Load())
	if maxP == 0 {
		return 1
	}
	procs := s.cachedGOMAXPROCS()
	if maxP < 0 {
		return procs
	}
	return min(max(maxP, 1), procs)
}

// EnterExecution registers the start of a top-level Executable.Execute() call (inter-call tracking).
// When maxParallelism > 0, it also throttles concurrent active executions to maxInterCallParallelism()
// so callers launching more concurrent requests than available cores do not oversubscribe the CPU.
// Note that maxIntraCallParallelism does NOT reduce this inter-call concurrency limit.
// It returns a function that must be deferred to unregister the call when Execute() finishes.
func (s *Scheduler) EnterExecution() func() {
	if s.maxParallelism.Load() > 0 {
		s.execGateMu.Lock()
		if s.execGateCond == nil {
			s.execGateCond = sync.NewCond(&s.execGateMu)
		}
		for s.maxParallelism.Load() > 0 && s.activeExecs >= s.maxInterCallParallelism() {
			s.execGateCond.Wait()
		}
		s.activeExecs++
		s.execGateMu.Unlock()
		s.numActiveCalls.Add(1)
		return func() {
			s.numActiveCalls.Add(-1)
			s.execGateMu.Lock()
			s.activeExecs--
			s.execGateCond.Signal()
			s.execGateMu.Unlock()
		}
	}
	s.numActiveCalls.Add(1)
	return func() {
		s.numActiveCalls.Add(-1)
	}
}

// EffectiveParallelism returns the target number of workers an operation within the current
// Execute() call should use, dynamically scaled by the number of concurrent active executions
// and capped by MaxIntraCallParallelism (when > 0):
//
//	min(maxIntraCallParallelism, max(1, maxInterCallParallelism / numActiveCalls))
func (s *Scheduler) EffectiveParallelism() int {
	total := s.maxInterCallParallelism()
	activeCalls := max(int(s.numActiveCalls.Load()), 1)
	eff := max(1, total/activeCalls)
	if intraCap := int(s.maxIntraCallParallelism.Load()); intraCap > 0 {
		eff = min(eff, intraCap)
	}
	return eff
}

// UnscaledParallelism returns the full target number of workers for a single Execute() call
// without dividing by numActiveCalls (still respecting MaxIntraCallParallelism when > 0).
// This is used by large operations that acquire the fair FIFO whole-chip gate via
// AcquireWholeChip() so each large op utilizes its allowed workers sequentially rather than
// thrashing shared L2/L3 caches concurrently.
func (s *Scheduler) UnscaledParallelism() int {
	total := s.maxInterCallParallelism()
	if intraCap := int(s.maxIntraCallParallelism.Load()); intraCap > 0 {
		return min(total, intraCap)
	}
	return total
}

// AcquireWholeChip acquires the fair FIFO whole-chip gate for a large operation so that
// concurrent large ops (across inter-op branches or concurrent Execute() calls) queue in
// strict FIFO order without starving or thrashing shared L2/L3 caches and SME units.
// The caller MUST call ReleaseWholeChip(acquired) when the operation finishes.
func (s *Scheduler) AcquireWholeChip() bool {
	if !s.IsEnabled() || s.UnscaledParallelism() <= 1 {
		return false
	}
	s.ensureWorkersStarted()
	s.wholeChipGate <- struct{}{}
	s.wholeChipHeld.Add(1)
	return true
}

// ReleaseWholeChip releases the whole-chip gate previously acquired via AcquireWholeChip.
func (s *Scheduler) ReleaseWholeChip(acquired bool) {
	if !acquired {
		return
	}
	s.wholeChipHeld.Add(-1)
	<-s.wholeChipGate
}

// IsWholeChipHeld reports whether a large operation currently holds the whole-chip gate.
func (s *Scheduler) IsWholeChipHeld() bool {
	return s.wholeChipHeld.Load() > 0
}

// IsBusy reports whether multiple top-level Execute() calls are active concurrently
// or a large operation holds the whole-chip gate.
func (s *Scheduler) IsBusy() bool {
	return s.numActiveCalls.Load() > 1 || s.wholeChipHeld.Load() > 0
}

// AdjustedMaxParallelism returns the effective parallelism for the current execution.
// It is an alias for EffectiveParallelism() kept for convenience across op implementations.
func (s *Scheduler) AdjustedMaxParallelism() int {
	return s.EffectiveParallelism()
}

// RunIntraOpTasks executes the given intra-op tasks for the current operation and waits for all
// of them to finish.
//
// Invariant (Deadlock-Free Work-Helping): The calling goroutine immediately participates in
// executing tasks from this slice itself at highest priority, while available persistent workers
// in the scheduler assist concurrently.
func (s *Scheduler) RunIntraOpTasks(tasks []func()) {
	n := len(tasks)
	if n == 0 {
		return
	}
	if n == 1 || !s.IsEnabled() {
		for _, task := range tasks {
			task()
		}
		return
	}

	eff := s.EffectiveParallelism()
	if s.IsUnlimited() {
		eff = s.UnscaledParallelism()
	}
	if eff <= 1 {
		for _, task := range tasks {
			task()
		}
		return
	}

	s.ensureWorkersStarted()

	var nextIdx atomic.Int32
	var unstarted atomic.Int32
	var wg sync.WaitGroup

	// Number of helper workers to recruit (the caller is 1 worker itself).
	numHelpers := min(n-1, eff-1)
	unstarted.Store(int32(numHelpers))
	wg.Add(numHelpers)

	decrementUnstarted := func() bool {
		for {
			u := unstarted.Load()
			if u <= 0 {
				return false
			}
			if unstarted.CompareAndSwap(u, u-1) {
				return true
			}
		}
	}
	cancelUnstarted := func() {
		for {
			u := unstarted.Load()
			if u <= 0 {
				return
			}
			if unstarted.CompareAndSwap(u, 0) {
				wg.Add(-int(u))
				return
			}
		}
	}

	helperFn := func() {
		if !decrementUnstarted() {
			return
		}
		defer wg.Done()
		for {
			idx := int(nextIdx.Add(1) - 1)
			if idx >= n {
				return
			}
			tasks[idx]()
		}
	}
	for range numHelpers {
		select {
		case s.intraOpQueue <- helperFn:
		default:
			// Queue full: skip recruiting this helper; the caller and already-queued helpers
			// will drain all tasks via nextIdx.
			if decrementUnstarted() {
				wg.Done()
			}
		}
	}

	// Work-helping: the calling goroutine executes its own op's tasks until none remain.
	for {
		idx := int(nextIdx.Add(1) - 1)
		if idx >= n {
			break
		}
		tasks[idx]()
	}

	cancelUnstarted()
	wg.Wait()
}

// ParallelizeChunks partitions [0, totalItems) into contiguous, cache-friendly ranges based on
// EffectiveParallelism() and minChunkSize, and executes fn(start, end) across the calling goroutine
// and available intra-op workers.
//
// Each worker processes contiguous ranges of items to prevent false sharing and preserve L1/L2
// cache locality. If EffectiveParallelism() <= 1 or totalItems <= minChunkSize, it calls
// fn(0, totalItems) directly inline with zero allocations.
func (s *Scheduler) ParallelizeChunks(totalItems, minChunkSize int, fn func(start, end int)) {
	if totalItems <= 0 {
		return
	}
	minChunkSize = max(minChunkSize, 1)
	if !s.IsEnabled() || totalItems <= minChunkSize {
		fn(0, totalItems)
		return
	}

	numWorkers := s.EffectiveParallelism()
	if numWorkers <= 1 {
		fn(0, totalItems)
		return
	}
	if s.numActiveCalls.Load() > 1 {
		minChunkSize *= 4
	}

	maxChunksBySize := (totalItems + minChunkSize - 1) / minChunkSize
	if maxChunksBySize <= 1 {
		fn(0, totalItems)
		return
	}

	s.ensureWorkersStarted()

	var nextIdx atomic.Int64
	var unstarted atomic.Int32
	var wg sync.WaitGroup

	numHelpers := min(maxChunksBySize-1, numWorkers-1)
	unstarted.Store(int32(numHelpers))
	wg.Add(numHelpers)

	decrementUnstarted := func() bool {
		for {
			u := unstarted.Load()
			if u <= 0 {
				return false
			}
			if unstarted.CompareAndSwap(u, u-1) {
				return true
			}
		}
	}
	cancelUnstarted := func() {
		for {
			u := unstarted.Load()
			if u <= 0 {
				return
			}
			if unstarted.CompareAndSwap(u, 0) {
				wg.Add(-int(u))
				return
			}
		}
	}

	// Guided self-scheduling (tapered chunks): early chunks are ~remaining/(2*numWorkers)
	// for cache locality and low atomic traffic, tapering down to minChunkSize at the tail
	// so slower cores (e.g. E-cores on hybrid x86/ARM64 CPUs) never hold a large tail chunk.
	divisor := 2 * numWorkers
	claimChunk := func() (int, int, bool) {
		for {
			cur := int(nextIdx.Load())
			rem := totalItems - cur
			if rem <= 0 {
				return 0, 0, false
			}
			chunk := max(minChunkSize, (rem+divisor-1)/divisor)
			if chunk > rem {
				chunk = rem
			}
			if nextIdx.CompareAndSwap(int64(cur), int64(cur+chunk)) {
				return cur, cur + chunk, true
			}
		}
	}

	helperFn := func() {
		if !decrementUnstarted() {
			return
		}
		defer wg.Done()
		for {
			start, end, ok := claimChunk()
			if !ok {
				return
			}
			fn(start, end)
		}
	}
	for range numHelpers {
		select {
		case s.intraOpQueue <- helperFn:
		default:
			if decrementUnstarted() {
				wg.Done()
			}
		}
	}

	// Work-helping: caller drains chunks alongside helpers.
	for {
		start, end, ok := claimChunk()
		if !ok {
			break
		}
		fn(start, end)
	}

	cancelUnstarted()
	wg.Wait()
}

// Saturate fans out workerFn across the calling goroutine and up to EffectiveParallelism()-1
// available scheduler workers (used by work-pulling ops like MatMul).
//
// Each workerFn pulls work from a shared queue until no work remains and then returns.
func (s *Scheduler) Saturate(workerFn func()) {
	s.SaturateUpTo(0, workerFn)
}

// SaturateUpTo is like Saturate, but caps the total number of participating workers
// (including the calling goroutine) to maxWorkers when maxWorkers > 0.
func (s *Scheduler) SaturateUpTo(maxWorkers int, workerFn func()) {
	if !s.IsEnabled() {
		workerFn()
		return
	}

	limit := s.EffectiveParallelism()
	if s.IsUnlimited() || s.wholeChipHeld.Load() > 0 {
		limit = s.UnscaledParallelism()
	}
	if maxWorkers > 0 && maxWorkers < limit {
		limit = maxWorkers
	}
	if limit <= 1 {
		workerFn()
		return
	}

	s.ensureWorkersStarted()

	var wg sync.WaitGroup
	var unstarted atomic.Int32

	numHelpers := limit - 1
	unstarted.Store(int32(numHelpers))
	wg.Add(numHelpers)

	decrementUnstarted := func() bool {
		for {
			u := unstarted.Load()
			if u <= 0 {
				return false
			}
			if unstarted.CompareAndSwap(u, u-1) {
				return true
			}
		}
	}
	cancelUnstarted := func() {
		for {
			u := unstarted.Load()
			if u <= 0 {
				return
			}
			if unstarted.CompareAndSwap(u, 0) {
				wg.Add(-int(u))
				return
			}
		}
	}

	helperFn := func() {
		if !decrementUnstarted() {
			return
		}
		defer wg.Done()
		workerFn()
	}
	for range numHelpers {
		select {
		case s.intraOpQueue <- helperFn:
		default:
			if decrementUnstarted() {
				wg.Done()
			}
		}
	}

	// The calling goroutine participates directly as one of the workers.
	workerFn()

	cancelUnstarted()
	wg.Wait()
}

// SubmitInterOp enqueues a coarse-grained inter-op task (such as a ready branch in executeParallel)
// to be executed by an available persistent worker when no higher-priority intra-op tasks are waiting.
// It returns true if an inter-op worker slot was claimed and the task was enqueued, or false if
// parallelism is disabled, the inter-op worker budget is full, intra-op tasks are already waiting,
// or the queue is full (so the caller can execute the branch locally or leave it in readyQueue).
func (s *Scheduler) SubmitInterOp(task func()) bool {
	if !s.IsEnabled() || s.EffectiveParallelism() <= 1 {
		return false
	}
	s.ensureWorkersStarted()
	if len(s.intraOpQueue) > 0 {
		return false
	}
	maxInterOp := min(s.maxInterOpWorkers, int32(max(s.EffectiveParallelism()-1, 1)))
	for {
		curr := s.activeInterOpWorkers.Load()
		if curr >= maxInterOp {
			return false
		}
		if s.activeInterOpWorkers.CompareAndSwap(curr, curr+1) {
			break
		}
	}
	wrapped := func() {
		defer s.activeInterOpWorkers.Add(-1)
		task()
	}
	select {
	case s.interOpQueue <- wrapped:
		return true
	default:
		s.activeInterOpWorkers.Add(-1)
		return false
	}
}

// WaitToStart runs the task using an available worker from the intra-op queue, or inline if
// parallelism is disabled or the queue is full.
//
// Prefer ParallelizeChunks or RunIntraOpTasks for new intra-op code so the calling goroutine
// participates in work-helping automatically.
func (s *Scheduler) WaitToStart(task func()) {
	if !s.IsEnabled() || s.EffectiveParallelism() <= 1 {
		task()
		return
	}
	s.ensureWorkersStarted()
	select {
	case s.intraOpQueue <- task:
	default:
		// If the queue is full, run inline on the caller goroutine (work-helping).
		task()
	}
}

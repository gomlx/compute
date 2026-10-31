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

// Scheduler manages persistent worker goroutines and dispatches intra-op and inter-op tasks.
type Scheduler struct {
	// maxParallelism is the configured target limit of parallel workers:
	//   > 0: explicit limit
	//   == 0: parallelism disabled
	//   < 0: unlimited (uses runtime.GOMAXPROCS(0))
	maxParallelism int

	// numActiveCalls tracks the number of concurrent top-level Executable.Execute() calls
	// for inter-call parallelism scaling.
	numActiveCalls atomic.Int32

	// maxInterOpWorkers tracks the maximum number of persistent background workers that
	// may concurrently execute inter-op tasks, reserving at least 1 persistent worker
	// (when numWorkers > 1) exclusively for intra-op tasks.
	maxInterOpWorkers int32

	// activeInterOpWorkers tracks the number of persistent workers currently executing
	// an inter-op task submitted via SubmitInterOp.
	activeInterOpWorkers atomic.Int32

	initOnce     sync.Once
	intraOpQueue chan func()
}

// New returns a new Scheduler with the default parallelism set to runtime.GOMAXPROCS(0).
func New() *Scheduler {
	return &Scheduler{
		maxParallelism: runtime.GOMAXPROCS(0),
	}
}

// ensureWorkersStarted lazily starts the persistent worker goroutines on first parallel use.
func (s *Scheduler) ensureWorkersStarted() {
	s.initOnce.Do(func() {
		s.intraOpQueue = make(chan func(), defaultQueueSize)
		// Start GOMAXPROCS persistent background workers.
		// If SetMaxParallelism was called with a larger positive value before first use,
		// start enough workers to cover it.
		numWorkers := max(runtime.GOMAXPROCS(0), s.maxParallelism, 1)
		if numWorkers > 1 {
			s.maxInterOpWorkers = int32(numWorkers - 1)
		} else {
			s.maxInterOpWorkers = 1
		}
		for range numWorkers {
			go s.workerLoop()
		}
	})
}

// workerLoop is the main loop of each persistent worker goroutine.
// Using a single channel receive on intraOpQueue enables Go's direct channel handoff fast-path
// and avoids runtime.selectgo / runtime.sellock contention on the critical intra-op path.
func (s *Scheduler) workerLoop() {
	for task := range s.intraOpQueue {
		task()
	}
}

// IsEnabled returns whether parallelism is enabled (maxParallelism != 0).
func (s *Scheduler) IsEnabled() bool {
	return s.maxParallelism != 0
}

// IsUnlimited returns whether parallelism is unlimited (maxParallelism < 0).
func (s *Scheduler) IsUnlimited() bool {
	return s.maxParallelism < 0
}

// MaxParallelism returns the configured target for parallelism.
// If set to 0, parallelism is disabled.
// If set to -1, parallelism is unlimited.
func (s *Scheduler) MaxParallelism() int {
	return s.maxParallelism
}

// SetMaxParallelism sets the target maxParallelism.
//
// This should be configured before workers start executing.
func (s *Scheduler) SetMaxParallelism(maxParallelism int) {
	s.maxParallelism = maxParallelism
}

// EnterExecution registers the start of a top-level Executable.Execute() call (inter-call tracking).
// It returns a function that must be deferred to unregister the call when Execute() finishes.
func (s *Scheduler) EnterExecution() func() {
	s.numActiveCalls.Add(1)
	return func() {
		s.numActiveCalls.Add(-1)
	}
}

// EffectiveParallelism returns the target number of workers an operation within the current
// Execute() call should use, dynamically scaled by the number of concurrent active executions:
//
//	max(1, totalParallelism / numActiveCalls)
func (s *Scheduler) EffectiveParallelism() int {
	if s.maxParallelism == 0 {
		return 1
	}
	var total int
	if s.maxParallelism < 0 {
		total = runtime.GOMAXPROCS(0)
	} else {
		total = min(max(s.maxParallelism, 1), runtime.GOMAXPROCS(0))
	}
	activeCalls := max(int(s.numActiveCalls.Load()), 1)
	return max(1, total/activeCalls)
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
		eff = runtime.GOMAXPROCS(0)
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

	maxChunksBySize := (totalItems + minChunkSize - 1) / minChunkSize
	numChunks := min(maxChunksBySize, 2*numWorkers)
	if numChunks <= 1 {
		fn(0, totalItems)
		return
	}

	s.ensureWorkersStarted()

	var nextChunk atomic.Int32
	var unstarted atomic.Int32
	var wg sync.WaitGroup

	numHelpers := min(numChunks-1, numWorkers-1)
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
			c := int(nextChunk.Add(1) - 1)
			if c >= numChunks {
				return
			}
			start := (c*totalItems + numChunks - 1) / numChunks
			end := ((c+1)*totalItems + numChunks - 1) / numChunks
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
		c := int(nextChunk.Add(1) - 1)
		if c >= numChunks {
			break
		}
		start := (c*totalItems + numChunks - 1) / numChunks
		end := ((c+1)*totalItems + numChunks - 1) / numChunks
		fn(start, end)
	}

	cancelUnstarted()
	wg.Wait()
}

// Saturate fans out workerFn across the calling goroutine and up to EffectiveParallelism()-1
// available scheduler workers (used by work-pulling ops like MatMul).
//
// Each workerFn pulls work from a shared queue until no work remains and then returns.
// Once any workerFn returns (indicating the work queue is exhausted), Saturate stops
// recruiting additional workers and waits for all already-running workerFn instances
// to finish before returning.
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
	if s.IsUnlimited() {
		limit = runtime.GOMAXPROCS(0)
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
	var doneFanningOut atomic.Bool

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
		if doneFanningOut.Load() || !decrementUnstarted() {
			return
		}
		defer wg.Done()
		if !doneFanningOut.Load() {
			workerFn()
			doneFanningOut.Store(true)
			cancelUnstarted()
		}
	}
	for range numHelpers {
		if doneFanningOut.Load() {
			if decrementUnstarted() {
				wg.Done()
			}
			continue
		}
		select {
		case s.intraOpQueue <- helperFn:
		default:
			if decrementUnstarted() {
				wg.Done()
			}
		}
	}

	// The calling goroutine participates directly as one of the workers.
	if !doneFanningOut.Load() {
		workerFn()
		doneFanningOut.Store(true)
	}

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
	case s.intraOpQueue <- wrapped:
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

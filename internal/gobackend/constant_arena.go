// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"runtime/debug"
	"slices"
	"sync"
	"sync/atomic"

	"k8s.io/klog/v2"
)

const (
	// defaultConstantArenaChunkSize is the default size (128 MiB) of each off-heap mmap chunk
	// in ConstantArena. Using large 2 MiB-aligned chunks minimizes VMA count and ensures even
	// small (< 2 MiB) weight slabs share 2 MiB Transparent Huge Pages in execution order.
	defaultConstantArenaChunkSize = 128 * 1024 * 1024

	// dedicatedConstantArenaThreshold is the allocation size (>= 32 MiB) above which an allocation
	// that does not fit in the current active chunk receives its own exact 2 MiB-rounded chunk,
	// avoiding leaving large unused tail holes in the active 128 MiB chunk (e.g., 768 MiB lm_head).
	dedicatedConstantArenaThreshold = 32 * 1024 * 1024

	// constantArenaAllocAlign aligns every sub-allocation within an arena chunk to 64 bytes
	// (cache-line / AVX-512 vector width).
	constantArenaAllocAlign = 64
)

// constantArenaChunk represents one off-heap 2 MiB-aligned mmap slab holding one or more
// transformed constant buffers (e.g., pre-packed GEMM panels).
//
// Lifetime is reference-counted across all live Builders/Executables that reference any
// ConstantData stored in this chunk. When the last Builder/Executable referencing this chunk
// is finalized, all ConstantData entries pointing into it are detached and the chunk is
// immediately returned to the OS via munmap.
type constantArenaChunk struct {
	raw       []byte
	offset    int
	builders  map[*Builder]struct{}
	constants []*ConstantData
	draining  bool
	unmapped  bool
}

// constantArena manages off-heap, MADV_HUGEPAGE-backed mmap chunks for cached constant
// representations on a Backend (enabled by default; configure via GOMLX_BACKEND="go:constant_arena=<true|false>"
// or GOMLX_GO_CONSTANT_ARENA).
//
// Lock ordering invariant (strictly enforced to prevent deadlocks):
//   ConstantData.mu -> constantArena.mu
// constantArena.mu must NEVER be held while acquiring any ConstantData.mu.
type constantArena struct {
	mu          sync.Mutex
	chunks      []*constantArenaChunk
	activeChunk *constantArenaChunk

	// releasedDonatedBytes tracks cumulative donated Go-heap constant bytes released after being
	// transformed into cachedForm (typically into this off-heap arena). Once it reaches
	// Backend.ConstantArenaGCThresholdBytes, reclaimReleasedDonatedBytes runs debug.FreeOSMemory()
	// and resets the counter so Go returns freed heap pages to the OS while transferring weights.
	releasedDonatedBytes atomic.Int64
	gcMu                 sync.Mutex

	// pendingPostConcurrentReclaim and pendingConcurrentExits handle a subtle concurrent cold-start
	// race in reclamation effectiveness (not safety):
	//
	// Suppose Call 1 (on the same or another Executable sharing the same donated ConstantData)
	// finishes pre-packing the last large weight (e.g., a 768 MiB lm_head), atomically swaps
	// rawBuffer to nil, resets releasedDonatedBytes to 0, and runs debug.FreeOSMemory(). If
	// concurrent Call 2 had already executed OpTypeConstant just before the swap, Call 2 still
	// holds a live pointer to that old rawBuffer inside its own execBuf.results until Call 2
	// finishes its node. Go's tracing GC safely keeps oldBuf alive during Call 1's sweep, so
	// oldBuf is not reclaimed until Call 2 finishes — at which point releasedDonatedBytes is
	// already 0 and no further constants will be transformed.
	//
	// To ensure such straggling donated buffers are still promptly returned to the OS without
	// waiting for an organic GC cycle, whenever reclaimReleasedDonatedBytes runs with more than
	// one live execution (liveExecutions > 1), it sets pendingPostConcurrentReclaim = true and
	// seeds pendingConcurrentExits = liveExecutions. Once either all live executions drain
	// (remainingLive == 0) or that many executions complete (covering continuously busy servers),
	// onExecutionFinished triggers one trailing debug.FreeOSMemory() and clears the flag.
	// Steady-state warm runs only perform a single false atomic load on pendingPostConcurrentReclaim.
	pendingPostConcurrentReclaim atomic.Bool
	pendingConcurrentExits       atomic.Int32
}

func newConstantArena() *constantArena {
	return &constantArena{}
}

// recordReleasedDonatedBytes accumulates bytes of a donated Go-heap raw constant buffer that has
// been detached from ConstantData.rawBuffer after caching its transformed representation.
func (a *constantArena) recordReleasedDonatedBytes(n int64) {
	if a == nil || n <= 0 {
		return
	}
	a.releasedDonatedBytes.Add(n)
}

// reclaimReleasedDonatedBytes runs GC + synchronous OS page scavenging (debug.FreeOSMemory) when
// at least threshold bytes of donated raw constant buffers have been released, then resets the counter.
// If liveExecutions > 1, it also arms a single trailing reclaim via onExecutionFinished so any
// donated raw buffers still pinned in concurrent executions' execBuf.results during this sweep
// are reclaimed once those concurrent executions finish.
func (a *constantArena) reclaimReleasedDonatedBytes(threshold int64, liveExecutions int32) {
	if a == nil || threshold <= 0 || a.releasedDonatedBytes.Load() < threshold {
		return
	}
	a.gcMu.Lock()
	defer a.gcMu.Unlock()
	if a.releasedDonatedBytes.Load() < threshold {
		return
	}
	a.releasedDonatedBytes.Store(0)
	if liveExecutions > 1 {
		a.pendingConcurrentExits.Store(liveExecutions)
		a.pendingPostConcurrentReclaim.Store(true)
	}
	debug.FreeOSMemory()
}

// onExecutionFinished is called after a top-level Executable.Execute() call has cleared its
// execution buffers and decremented NumLiveExecutions to remainingLive. If a donated-buffer
// reclaim previously ran while multiple executions were concurrently active, this triggers one
// trailing debug.FreeOSMemory() once the concurrent batch has drained (or remainingLive == 0).
func (a *constantArena) onExecutionFinished(remainingLive int32) {
	if a == nil || !a.pendingPostConcurrentReclaim.Load() {
		return
	}
	if remainingLive > 0 && a.pendingConcurrentExits.Add(-1) > 0 {
		return
	}
	a.gcMu.Lock()
	defer a.gcMu.Unlock()
	if !a.pendingPostConcurrentReclaim.Swap(false) {
		return
	}
	a.pendingConcurrentExits.Store(0)
	a.releasedDonatedBytes.Store(0)
	debug.FreeOSMemory()
}

// ConstantArenaStats returns the number of currently mapped off-heap constant arena chunks,
// total mapped bytes, and total sub-allocated bytes.
func (b *Backend) ConstantArenaStats() (chunks int, mappedBytes, usedBytes int64) {
	if b == nil || b.constantArena == nil {
		return 0, 0, 0
	}
	arena := b.constantArena
	arena.mu.Lock()
	defer arena.mu.Unlock()
	for _, chunk := range arena.chunks {
		if chunk.unmapped {
			continue
		}
		chunks++
		mappedBytes += int64(len(chunk.raw))
		usedBytes += int64(chunk.offset)
	}
	return chunks, mappedBytes, usedBytes
}

// AllocConstantBytes allocates byteSize bytes (64-byte aligned) for a long-lived cached constant
// representation belonging to constData.
//
// Caller MUST hold constData.mu (which is guaranteed when called from inside GetOrSetCachedForm's
// computeFn) when constData != nil.
//
// When Backend.UseConstantArena is enabled and constData != nil (with at least one tracked live
// Builder), bytes are sub-allocated from a 2 MiB-aligned off-heap mmap + MADV_HUGEPAGE chunk
// tracked by constData and all live Builders/Executables using it. Otherwise, it falls back to a
// Go-heap aligned allocation with AdviseHugePages.
func (b *Backend) AllocConstantBytes(constData *ConstantData, byteSize int) []byte {
	if byteSize <= 0 {
		return nil
	}
	if b == nil || !b.UseConstantArena || b.constantArena == nil || constData == nil || len(constData.liveBuilders) == 0 {
		buf := allocAlignedBytes(byteSize)
		AdviseHugePages(buf)
		return buf
	}

	alignedSize := (byteSize + constantArenaAllocAlign - 1) &^ (constantArenaAllocAlign - 1)
	arena := b.constantArena

	arena.mu.Lock()
	var chunk *constantArenaChunk
	if arena.activeChunk != nil && !arena.activeChunk.draining && !arena.activeChunk.unmapped &&
		len(arena.activeChunk.raw)-arena.activeChunk.offset >= alignedSize {
		chunk = arena.activeChunk
	} else {
		var chunkSize int
		isDedicated := alignedSize >= dedicatedConstantArenaThreshold
		if isDedicated {
			chunkSize = alignedSize
		} else {
			chunkSize = defaultConstantArenaChunkSize
		}
		raw, err := mmapConstantArenaChunk(chunkSize)
		if err != nil {
			arena.mu.Unlock()
			klog.Warningf("go backend: constant_arena mmap(%d bytes) failed (%v), falling back to Go heap", chunkSize, err)
			buf := allocAlignedBytes(byteSize)
			AdviseHugePages(buf)
			return buf
		}
		chunk = &constantArenaChunk{
			raw:      raw,
			builders: make(map[*Builder]struct{}, max(1, len(constData.liveBuilders))),
		}
		arena.chunks = append(arena.chunks, chunk)
		if !isDedicated {
			arena.activeChunk = chunk
		}
	}

	start := chunk.offset
	chunk.offset += alignedSize
	out := chunk.raw[start : start+byteSize : start+byteSize]

	if !slices.Contains(chunk.constants, constData) {
		chunk.constants = append(chunk.constants, constData)
	}
	for builder := range constData.liveBuilders {
		if _, exists := chunk.builders[builder]; !exists {
			chunk.builders[builder] = struct{}{}
			builder.arenaChunks = append(builder.arenaChunks, chunk)
		}
	}
	arena.mu.Unlock()

	// constData.mu is held by caller (GetOrSetCachedForm).
	if !slices.Contains(constData.arenaChunks, chunk) {
		constData.arenaChunks = append(constData.arenaChunks, chunk)
	}
	return out
}

// registerConstantUser records that builder b references constData and retains any
// off-heap constantArenaChunks already backing constData.cachedForm.
func (b *Builder) registerConstantUser(constData *ConstantData) {
	if b == nil || constData == nil || b.Backend == nil || b.Backend.constantArena == nil {
		return
	}
	if slices.Contains(b.usedConstants, constData) {
		return
	}
	b.usedConstants = append(b.usedConstants, constData)

	constData.mu.Lock()
	if constData.liveBuilders == nil {
		constData.liveBuilders = make(map[*Builder]struct{}, 2)
	}
	constData.liveBuilders[b] = struct{}{}
	chunks := constData.arenaChunks
	if len(chunks) > 0 {
		arena := b.Backend.constantArena
		arena.mu.Lock()
		for _, chunk := range chunks {
			if chunk.unmapped {
				continue
			}
			chunk.draining = false
			if _, exists := chunk.builders[b]; !exists {
				chunk.builders[b] = struct{}{}
				b.arenaChunks = append(b.arenaChunks, chunk)
			}
		}
		arena.mu.Unlock()
	}
	constData.mu.Unlock()
}

// releaseConstantArenaRefs drops builder b's references to all ConstantData and arena chunks.
// Any arena chunk whose builder reference count reaches zero is immediately detached from its
// ConstantData entries and unmapped back to the OS.
func (b *Builder) releaseConstantArenaRefs() {
	if b == nil || b.Backend == nil || b.Backend.constantArena == nil {
		return
	}
	consts := b.usedConstants
	b.usedConstants = nil
	for _, constData := range consts {
		constData.mu.Lock()
		delete(constData.liveBuilders, b)
		constData.mu.Unlock()
	}

	arena := b.Backend.constantArena
	type drainCandidate struct {
		chunk  *constantArenaChunk
		consts []*ConstantData
	}
	var candidates []drainCandidate

	arena.mu.Lock()
	chunks := b.arenaChunks
	b.arenaChunks = nil
	for _, chunk := range chunks {
		if chunk.unmapped {
			continue
		}
		delete(chunk.builders, b)
		if len(chunk.builders) == 0 && !chunk.draining {
			chunk.draining = true
			if arena.activeChunk == chunk {
				arena.activeChunk = nil
			}
			candidates = append(candidates, drainCandidate{
				chunk:  chunk,
				consts: slices.Clone(chunk.constants),
			})
		}
	}
	arena.mu.Unlock()

	for _, cand := range candidates {
		chunk := cand.chunk
		revived := false
		for _, constData := range cand.consts {
			constData.mu.Lock()
			arena.mu.Lock()
			if !chunk.draining || chunk.unmapped {
				arena.mu.Unlock()
				constData.mu.Unlock()
				revived = true
				break
			}
			if len(constData.liveBuilders) > 0 {
				// Another builder concurrently adopted constData before we detached chunk: revive chunk.
				chunk.draining = false
				for liveBuilder := range constData.liveBuilders {
					if _, exists := chunk.builders[liveBuilder]; !exists {
						chunk.builders[liveBuilder] = struct{}{}
						liveBuilder.arenaChunks = append(liveBuilder.arenaChunks, chunk)
					}
				}
				arena.mu.Unlock()
				constData.mu.Unlock()
				revived = true
				break
			}
			// No live builders reference constData and chunk is draining: clear cachedForm so no
			// dangling pointer into chunk.raw can ever survive past munmap.
			constData.cachedForm.Store(nil)
			constData.arenaChunks = slices.DeleteFunc(constData.arenaChunks, func(entryChunk *constantArenaChunk) bool {
				return entryChunk == chunk
			})
			chunk.constants = slices.DeleteFunc(chunk.constants, func(entryConstData *ConstantData) bool {
				return entryConstData == constData
			})
			if constData.Buffer() == nil {
				constData.arenaReleased = true
			}
			arena.mu.Unlock()
			constData.mu.Unlock()
		}
		if revived {
			continue
		}

		arena.mu.Lock()
		if chunk.draining && !chunk.unmapped && len(chunk.builders) == 0 {
			chunk.unmapped = true
			raw := chunk.raw
			chunk.raw = nil
			chunk.constants = nil
			arena.chunks = slices.DeleteFunc(arena.chunks, func(entryChunk *constantArenaChunk) bool {
				return entryChunk == chunk
			})
			arena.mu.Unlock()
			munmapConstantArenaChunk(raw)
		} else {
			arena.mu.Unlock()
		}
	}
}

// Finalize unmaps all remaining chunks owned by this arena (called from Backend.Finalize after
// all in-flight executions have completed).
func (a *constantArena) Finalize() {
	if a == nil {
		return
	}
	a.mu.Lock()
	chunks := a.chunks
	a.chunks = nil
	a.activeChunk = nil
	var toUnmap [][]byte
	for _, chunk := range chunks {
		if chunk.unmapped {
			continue
		}
		chunk.unmapped = true
		chunk.draining = true
		if len(chunk.raw) > 0 {
			toUnmap = append(toUnmap, chunk.raw)
			chunk.raw = nil
		}
		chunk.builders = nil
		chunk.constants = nil
	}
	a.mu.Unlock()

	for _, raw := range toUnmap {
		munmapConstantArenaChunk(raw)
	}
}

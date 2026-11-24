// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/envutil"
)

func newTestArenaBackend(t *testing.T, config string) *Backend {
	t.Helper()
	b, err := New(config)
	if err != nil {
		t.Fatalf("New(%q): %+v", config, err)
	}
	return b.(*Backend)
}

func TestConstantArenaConfig(t *testing.T) {
	t.Run("default_on", func(t *testing.T) {
		t.Setenv(envutil.GoBackendConstantArena, "")
		b := newTestArenaBackend(t, "")
		defer b.Finalize()
		if !b.UseConstantArena {
			t.Fatalf("expected UseConstantArena=true by default")
		}
		if b.ConstantArenaGCThresholdBytes != DefaultConstantArenaGCThresholdBytes {
			t.Fatalf("expected default ConstantArenaGCThresholdBytes=%d, got %d",
				DefaultConstantArenaGCThresholdBytes, b.ConstantArenaGCThresholdBytes)
		}
		chunks, mapped, used := b.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected empty arena stats, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}
	})

	t.Run("config_flag_enable_disable", func(t *testing.T) {
		t.Setenv(envutil.GoBackendConstantArena, "false")
		b := newTestArenaBackend(t, "constant_arena")
		defer b.Finalize()
		if !b.UseConstantArena {
			t.Fatalf("expected UseConstantArena=true for config 'constant_arena'")
		}
	})

	t.Run("config_explicit_true_false", func(t *testing.T) {
		t.Setenv(envutil.GoBackendConstantArena, "true")
		bOff := newTestArenaBackend(t, "constant_arena=false")
		defer bOff.Finalize()
		if bOff.UseConstantArena {
			t.Fatalf("expected explicit constant_arena=false to override env var")
		}

		t.Setenv(envutil.GoBackendConstantArena, "false")
		bOn := newTestArenaBackend(t, "constant_arena=true")
		defer bOn.Finalize()
		if !bOn.UseConstantArena {
			t.Fatalf("expected explicit constant_arena=true to override env var")
		}
	})

	t.Run("env_var_disable", func(t *testing.T) {
		t.Setenv(envutil.GoBackendConstantArena, "0")
		b := newTestArenaBackend(t, "")
		defer b.Finalize()
		if b.UseConstantArena {
			t.Fatalf("expected UseConstantArena=false from %s=0", envutil.GoBackendConstantArena)
		}
	})

	t.Run("invalid_config_value", func(t *testing.T) {
		if _, err := New("constant_arena=maybe"); err == nil {
			t.Fatalf("expected error for invalid constant_arena value")
		}
	})

	t.Run("intra_call_parallelism_config", func(t *testing.T) {
		bDefault := newTestArenaBackend(t, "")
		defer bDefault.Finalize()
		if got := bDefault.Scheduler.MaxIntraCallParallelism(); got != 0 {
			t.Fatalf("expected default MaxIntraCallParallelism=0, got %d", got)
		}

		bCap := newTestArenaBackend(t, "parallelism=8,intra_call_parallelism=2")
		defer bCap.Finalize()
		if got := bCap.Scheduler.MaxParallelism(); got != 8 {
			t.Fatalf("expected MaxParallelism=8, got %d", got)
		}
		if got := bCap.Scheduler.MaxIntraCallParallelism(); got != 2 {
			t.Fatalf("expected MaxIntraCallParallelism=2, got %d", got)
		}
		if got := bCap.Scheduler.EffectiveParallelism(); got != 2 {
			t.Fatalf("expected EffectiveParallelism=2, got %d", got)
		}

		if _, err := New("intra_call_parallelism=-1"); err == nil {
			t.Fatalf("expected error for negative intra_call_parallelism")
		}
		if _, err := New("intra_call_parallelism=abc"); err == nil {
			t.Fatalf("expected error for non-integer intra_call_parallelism")
		}
	})
}

func TestConstantArenaAllocationAndRefcounting(t *testing.T) {
	b := newTestArenaBackend(t, "constant_arena")
	defer b.Finalize()

	makeTrackedConst := func(n int) *ConstantData {
		constData, err := NewConstantData(b, shapes.Make(dtypes.Float32, n), make([]float32, n))
		if err != nil {
			t.Fatalf("NewConstantData: %+v", err)
		}
		return constData
	}

	t.Run("suballoc_alignment_and_release", func(t *testing.T) {
		constData1 := makeTrackedConst(64)
		constData2 := makeTrackedConst(64)
		b1 := b.Builder("b1").(*Builder)
		b2 := b.Builder("b2").(*Builder)

		b1.registerConstantUser(constData1)
		b2.registerConstantUser(constData1)
		b2.registerConstantUser(constData2)

		var p1, p2 []byte
		_ = GetOrSetCachedForm(constData1, func() []byte {
			p1 = b.AllocConstantBytes(constData1, 100)
			return p1
		})
		_ = GetOrSetCachedForm(constData2, func() []byte {
			p2 = b.AllocConstantBytes(constData2, 200)
			return p2
		})

		if len(p1) != 100 || len(p2) != 200 {
			t.Fatalf("unexpected slice lengths: len(p1)=%d len(p2)=%d", len(p1), len(p2))
		}
		addr1 := uintptr(unsafe.Pointer(unsafe.SliceData(p1)))
		addr2 := uintptr(unsafe.Pointer(unsafe.SliceData(p2)))
		if addr1%constantArenaAllocAlign != 0 {
			t.Fatalf("p1 (0x%x) is not %d-byte aligned", addr1, constantArenaAllocAlign)
		}
		if addr2%constantArenaAllocAlign != 0 {
			t.Fatalf("p2 (0x%x) is not %d-byte aligned", addr2, constantArenaAllocAlign)
		}
		if (runtime.GOOS == "linux" || runtime.GOOS == "darwin") && addr1%(2*1024*1024) != 0 {
			t.Fatalf("first allocation in a new chunk (0x%x) is not 2 MiB aligned", addr1)
		}

		chunks, mapped, used := b.ConstantArenaStats()
		if chunks != 1 || mapped != int64(defaultConstantArenaChunkSize) || used != int64(128+256) {
			t.Fatalf("unexpected arena stats: chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}

		// Finalizing b1 must NOT unmap the chunk because b2 still references constData1/constData2.
		b1.Finalize()
		chunks, _, _ = b.ConstantArenaStats()
		if chunks != 1 {
			t.Fatalf("expected chunk to stay mapped while b2 is live, got chunks=%d", chunks)
		}
		if constData1.CachedForm() == nil {
			t.Fatalf("expected constData1.CachedForm() to remain set while b2 is live")
		}

		// Finalizing b2 drops refcount to 0 -> chunk must be unmapped and cachedForms cleared.
		b2.Finalize()
		chunks, mapped, used = b.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected all chunks unmapped after b2.Finalize(), got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}
		if constData1.CachedForm() != nil || constData2.CachedForm() != nil {
			t.Fatalf("expected cachedForms cleared after chunk unmap")
		}
	})

	t.Run("dedicated_large_chunk", func(t *testing.T) {
		constDataLarge := makeTrackedConst(16)
		bLarge := b.Builder("large").(*Builder)
		bLarge.registerConstantUser(constDataLarge)

		const largeSize = dedicatedConstantArenaThreshold + 1234
		var p []byte
		_ = GetOrSetCachedForm(constDataLarge, func() []byte {
			p = b.AllocConstantBytes(constDataLarge, largeSize)
			return p
		})
		if len(p) != largeSize {
			t.Fatalf("expected len(p)=%d, got %d", largeSize, len(p))
		}
		chunks, mapped, _ := b.ConstantArenaStats()
		if chunks != 1 {
			t.Fatalf("expected 1 dedicated chunk, got %d", chunks)
		}
		expectedMapped := int64((largeSize + 2*1024*1024 - 1) &^ (2*1024*1024 - 1))
		if (runtime.GOOS == "linux" || runtime.GOOS == "darwin") && mapped != expectedMapped {
			t.Fatalf("expected mapped=%d, got %d", expectedMapped, mapped)
		}

		bLarge.Finalize()
		chunks, mapped, used := b.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected 0 chunks after large builder finalize, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}
	})

	t.Run("anonymous_constant_falls_back_to_heap", func(t *testing.T) {
		constDataAnon := NewAnonymousConstantData()
		var p []byte
		_ = GetOrSetCachedForm(constDataAnon, func() []byte {
			p = b.AllocConstantBytes(constDataAnon, 4096)
			return p
		})
		if len(p) != 4096 {
			t.Fatalf("expected len(p)=4096, got %d", len(p))
		}
		chunks, _, _ := b.ConstantArenaStats()
		if chunks != 0 {
			t.Fatalf("untracked anonymous constant must not enter refcounted off-heap arena, got chunks=%d", chunks)
		}
	})

	t.Run("backend_finalize_unmaps_remaining_chunks", func(t *testing.T) {
		bTmp := newTestArenaBackend(t, "constant_arena")
		constData, err := NewConstantData(bTmp, shapes.Make(dtypes.Float32, 32), make([]float32, 32))
		if err != nil {
			t.Fatalf("NewConstantData: %+v", err)
		}
		bldr := bTmp.Builder("leaked").(*Builder)
		bldr.registerConstantUser(constData)
		_ = GetOrSetCachedForm(constData, func() []byte {
			return bTmp.AllocConstantBytes(constData, 4096)
		})
		chunks, _, _ := bTmp.ConstantArenaStats()
		if chunks != 1 {
			t.Fatalf("expected 1 chunk before Backend.Finalize, got %d", chunks)
		}

		bTmp.Finalize()
		chunks, mapped, used := bTmp.ConstantArenaStats()
		if chunks != 0 || mapped != 0 || used != 0 {
			t.Fatalf("expected 0 chunks after Backend.Finalize, got chunks=%d mapped=%d used=%d", chunks, mapped, used)
		}

		// Calling Builder.Finalize after Backend.Finalize must be a safe no-op.
		bldr.Finalize()
	})
}

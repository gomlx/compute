// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend_test

import (
	"runtime"
	"sync"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestBuffers_Bytes(t *testing.T) {
	buf, err := backend.(*gobackend.Backend).GetBuffer(shapes.Make(dtypes.Int32, 3))
	if err != nil {
		t.Fatalf("Failed to get buffer: %+v", err)
	}
	buf.Zeros()
	if len(buf.Flat.([]int32)) != 3 {
		t.Fatalf("Expected length 3, got %d", len(buf.Flat.([]int32)))
	}
	flatBytes, err := buf.MutableBytes()
	if err != nil {
		t.Fatalf("Failed to get mutable bytes: %+v", err)
	}
	if len(flatBytes) != 3*int(dtypes.Int32.Size()) {
		t.Fatalf("Expected length %d, got %d", 3*int(dtypes.Int32.Size()), len(flatBytes))
	}
	flatBytes[0] = 1
	flatBytes[4] = 7
	flatBytes[8] = 3
	if ok, diff := testutil.IsEqual([]int32{1, 7, 3}, buf.Flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
	runtime.KeepAlive(buf)
}

func TestBuffers_Fill(t *testing.T) {
	buf, err := backend.(*gobackend.Backend).GetBuffer(shapes.Make(dtypes.Int32, 3))
	if err != nil {
		t.Fatalf("Failed to get buffer: %+v", err)
	}
	if len(buf.Flat.([]int32)) != 3 {
		t.Fatalf("Expected length 3, got %d", len(buf.Flat.([]int32)))
	}
	if err := buf.Fill(int32(3)); err != nil {
		t.Fatalf("Failed to fill buffer: %+v", err)
	}
	if ok, diff := testutil.IsEqual([]int32{3, 3, 3}, buf.Flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}

	buf.Zeros()
	if ok, diff := testutil.IsEqual([]int32{0, 0, 0}, buf.Flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
}

func TestBufferPool_LIFOAndConcurrent(t *testing.T) {
	be, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	shape := shapes.Make(dtypes.Float32, 64)
	b1, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer b1 failed: %+v", err)
	}
	b2, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer b2 failed: %+v", err)
	}
	if b1 == b2 {
		t.Fatalf("Expected distinct buffers when both are in use")
	}

	// Return b1 then b2; LIFO stack must return b2 first, then b1.
	be.PutBuffer(b1)
	be.PutBuffer(b2)

	got1, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer got1 failed: %+v", err)
	}
	if got1 != b2 {
		t.Fatalf("Expected LIFO order: got %p, want b2 (%p)", got1, b2)
	}
	got2, err := be.GetBuffer(shape)
	if err != nil {
		t.Fatalf("GetBuffer got2 failed: %+v", err)
	}
	if got2 != b1 {
		t.Fatalf("Expected LIFO order: got %p, want b1 (%p)", got2, b1)
	}
	be.PutBuffer(got1)
	be.PutBuffer(got2)

	// Concurrent stress test across 32 goroutines hammering the same bucket size.
	const numWorkers = 32
	const iters = 2000
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for w := range numWorkers {
		go func(workerID int) {
			defer wg.Done()
			val := float32(workerID + 1)
			for range iters {
				buf1, err := be.GetBuffer(shape)
				if err != nil {
					t.Errorf("GetBuffer failed: %+v", err)
					return
				}
				flat1 := buf1.Flat.([]float32)
				flat1[0] = val
				flat1[len(flat1)-1] = val

				buf2, err := be.GetBuffer(shape)
				if err != nil {
					t.Errorf("GetBuffer 2 failed: %+v", err)
					return
				}
				flat2 := buf2.Flat.([]float32)
				flat2[0] = -val

				if flat1[0] != val || flat1[len(flat1)-1] != val {
					t.Errorf("Concurrent buffer corruption on worker %d: got (%v, %v), want %v",
						workerID, flat1[0], flat1[len(flat1)-1], val)
					return
				}
				be.PutBuffer(buf2)
				be.PutBuffer(buf1)
			}
		}(w)
	}
	wg.Wait()
}

func BenchmarkBufferPoolSequential(b *testing.B) {
	be, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	shape := shapes.Make(dtypes.Float32, 1024)
	buf, err := be.GetBuffer(shape)
	if err != nil {
		b.Fatalf("GetBuffer failed: %+v", err)
	}
	be.PutBuffer(buf)

	for b.Loop() {
		buf, err := be.GetBuffer(shape)
		if err != nil {
			b.Fatalf("GetBuffer failed: %+v", err)
		}
		be.PutBuffer(buf)
	}
}

func BenchmarkBufferPoolParallel(b *testing.B) {
	be, err := gobackend.NewBackend()
	if err != nil {
		b.Fatalf("NewBackend failed: %+v", err)
	}
	defer be.Finalize()

	shape := shapes.Make(dtypes.Float32, 1024)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf, err := be.GetBuffer(shape)
			if err != nil {
				b.Fatalf("GetBuffer failed: %+v", err)
			}
			be.PutBuffer(buf)
		}
	})
}

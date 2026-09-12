// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil_test

import (
	"testing"
	"time"

	"github.com/gomlx/compute/support/testutil"
)

func TestDurationSampler(t *testing.T) {
	s := testutil.NewDurationSampler(100)

	if s.Count() != 0 || s.ReservoirCount() != 0 {
		t.Fatalf("expected empty sampler, got Count=%d, ReservoirCount=%d", s.Count(), s.ReservoirCount())
	}
	if s.Median() != 0 {
		t.Fatalf("expected 0 median for empty sampler, got %v", s.Median())
	}

	// Insert 100 samples from 1ms to 100ms
	for i := 1; i <= 100; i++ {
		s.Sample(time.Duration(i) * time.Millisecond)
	}

	if s.Count() != 100 || s.ReservoirCount() != 100 {
		t.Fatalf("expected 100 samples, got Count=%d, ReservoirCount=%d", s.Count(), s.ReservoirCount())
	}

	med := s.Median()
	// Median of 1..100 is around 50ms
	if med < 49*time.Millisecond || med > 51*time.Millisecond {
		t.Errorf("expected median ~50ms, got %v", med)
	}

	if s.Min() != 1*time.Millisecond {
		t.Errorf("expected min 1ms, got %v", s.Min())
	}
	if s.Max() != 100*time.Millisecond {
		t.Errorf("expected max 100ms, got %v", s.Max())
	}

	// Insert 1000 more samples beyond capacity (reservoir sampling)
	for i := 101; i <= 1100; i++ {
		s.Sample(time.Duration(i) * time.Millisecond)
	}

	if s.Count() != 1100 {
		t.Errorf("expected count 1100, got %d", s.Count())
	}
	if s.ReservoirCount() != 100 {
		t.Errorf("expected reservoir count 100, got %d", s.ReservoirCount())
	}

	// Median of uniformly distributed samples in 1..1100 should be roughly in the middle (550ms +/- 100ms)
	med = s.Median()
	if med < 350*time.Millisecond || med > 750*time.Millisecond {
		t.Errorf("expected median ~550ms, got %v", med)
	}

	// Reset
	s.Reset()
	if s.Count() != 0 || s.ReservoirCount() != 0 {
		t.Errorf("expected empty sampler after Reset")
	}
}

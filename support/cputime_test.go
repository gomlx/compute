// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package support

import (
	"testing"
	"time"
)

func TestCPUTime(t *testing.T) {
	start := CPUTime()
	if start <= 0 {
		t.Fatalf("CPUTime() returned non-positive duration: %v", start)
	}

	// Burn a small amount of CPU time so the counter advances.
	deadline := time.Now().Add(10 * time.Millisecond)
	acc := 0
	for time.Now().Before(deadline) {
		for i := range 1000 {
			acc += i
		}
	}
	_ = acc

	end := CPUTime()
	if end <= start {
		t.Errorf("Expected CPUTime() to advance after busy work: start=%v, end=%v", start, end)
	}
	t.Logf("CPUTime elapsed over ~10ms busy loop: %v (%d ns)", end-start, (end - start).Nanoseconds())
}

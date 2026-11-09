// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package support

import (
	"testing"
	"time"
)

func TestCPUTime(t *testing.T) {
	start := CPUTime()
	if start < 0 {
		t.Fatalf("CPUTime() returned negative duration: %v", start)
	}

	// Burn a small amount of CPU time until the counter advances (up to 100ms to account
	// for coarse 15.6ms timer ticks in Windows GetProcessTimes).
	deadline := time.Now().Add(100 * time.Millisecond)
	acc := 0
	end := start
	for end <= start && time.Now().Before(deadline) {
		stepDeadline := time.Now().Add(5 * time.Millisecond)
		for time.Now().Before(stepDeadline) {
			for i := range 1000 {
				acc += i
			}
		}
		end = CPUTime()
	}
	_ = acc

	if end <= start {
		t.Errorf("Expected CPUTime() to advance after busy work: start=%v, end=%v", start, end)
	}
	t.Logf("CPUTime elapsed: %v (%d ns)", end-start, (end - start).Nanoseconds())
}

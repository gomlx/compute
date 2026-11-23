// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend_test

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
)

func TestAdviseHugePages(t *testing.T) {
	// Small and empty slices should be safe no-ops.
	gobackend.AdviseHugePages(nil)
	gobackend.AdviseHugePages(make([]byte, 1024))
	gobackend.AdviseHugePages(make([]byte, 2*1024*1024-1))

	// Large slice (>= 2 MiB) should succeed without panicking and remain writable.
	buf := make([]byte, 6*1024*1024)
	gobackend.AdviseHugePages(buf)
	for i := 0; i < len(buf); i += 4096 {
		buf[i] = byte(i >> 12)
	}
	if buf[4096] != 1 {
		t.Fatalf("unexpected byte value after AdviseHugePages: %d", buf[4096])
	}

	if runtime.GOOS == "linux" && gobackend.UseHugePages {
		if raw, err := os.ReadFile("/proc/self/smaps_rollup"); err == nil {
			for _, line := range strings.Split(string(raw), "\n") {
				if strings.HasPrefix(line, "AnonHugePages:") {
					t.Logf("Linux THP status after AdviseHugePages(6 MiB): %s", strings.TrimSpace(line))
				}
			}
		}
	}
}

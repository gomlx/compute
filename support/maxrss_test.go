// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package support

import (
	"runtime"
	"testing"
)

func TestMaxRSS(t *testing.T) {
	rss := MaxRSS()
	switch runtime.GOOS {
	case "linux", "darwin", "freebsd", "openbsd", "netbsd", "dragonfly", "windows":
		if rss <= 0 {
			t.Fatalf("MaxRSS() returned non-positive value on %s: %d", runtime.GOOS, rss)
		}
	default:
		if rss < 0 {
			t.Fatalf("MaxRSS() returned negative value: %d", rss)
		}
	}
	t.Logf("MaxRSS: %d bytes (%.2f MiB)", rss, float64(rss)/(1024*1024))
}

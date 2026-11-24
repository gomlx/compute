// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux

package scheduler

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

type physicalCoreKey struct {
	pkg  string
	die  string
	core string
}

// detectPhysicalCores counts unique online physical CPU cores on Linux by inspecting
// /sys/devices/system/cpu/cpu*/topology/{physical_package_id,die_id,core_id}.
// Falls back to runtime.NumCPU() if sysfs topology information is unavailable.
func detectPhysicalCores() int {
	entries, err := os.ReadDir("/sys/devices/system/cpu")
	if err != nil {
		return max(runtime.NumCPU(), 1)
	}
	seen := make(map[physicalCoreKey]struct{})
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, "cpu") || len(name) <= 3 {
			continue
		}
		allDigits := true
		for i := 3; i < len(name); i++ {
			if name[i] < '0' || name[i] > '9' {
				allDigits = false
				break
			}
		}
		if !allDigits {
			continue
		}
		cpuDir := filepath.Join("/sys/devices/system/cpu", name)
		if onlineBytes, err := os.ReadFile(filepath.Join(cpuDir, "online")); err == nil {
			if strings.TrimSpace(string(onlineBytes)) == "0" {
				continue
			}
		}
		coreBytes, err := os.ReadFile(filepath.Join(cpuDir, "topology", "core_id"))
		if err != nil {
			continue
		}
		pkgID := "0"
		if pkgBytes, err := os.ReadFile(filepath.Join(cpuDir, "topology", "physical_package_id")); err == nil {
			pkgID = strings.TrimSpace(string(pkgBytes))
		}
		dieID := "0"
		if dieBytes, err := os.ReadFile(filepath.Join(cpuDir, "topology", "die_id")); err == nil {
			dieID = strings.TrimSpace(string(dieBytes))
		}
		seen[physicalCoreKey{
			pkg:  pkgID,
			die:  dieID,
			core: strings.TrimSpace(string(coreBytes)),
		}] = struct{}{}
	}
	if len(seen) == 0 {
		return max(runtime.NumCPU(), 1)
	}
	return len(seen)
}

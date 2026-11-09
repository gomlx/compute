// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !darwin || !arm64 || !goexperiment.simd

package simdarch

// ReadTPIDR_EL0 returns 0 on non-darwin/arm64 platforms.
func ReadTPIDR_EL0() uint64 {
	return 0
}

// CurrentSMECluster returns (0, 0, false) on non-darwin/arm64 platforms.
func CurrentSMECluster() (clusterID int, cpuID int, isEfficiencyCore bool) {
	return 0, 0, false
}

// AcquireSME is a no-op stub on non-darwin/arm64 platforms.
func AcquireSME() (clusterID int, useSME bool, isEfficiencyCore bool) {
	return -1, IsSMEAllowed, false
}

// AcquireSMEWithLimit is a no-op stub on non-darwin/arm64 platforms.
func AcquireSMEWithLimit(_, _ int32) (clusterID int, useSME bool, isEfficiencyCore bool) {
	return -1, IsSMEAllowed, false
}

// ReleaseSME is a no-op stub on non-darwin/arm64 platforms.
func ReleaseSME(_ int) {}

// ActiveSMEWorkers returns 0 on non-darwin/arm64 platforms.
func ActiveSMEWorkers() int32 {
	return 0
}

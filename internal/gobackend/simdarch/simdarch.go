// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simdarch

import (
	"runtime"

	"github.com/gomlx/compute/support/envutil"
)

// MaxSMEUnits is the maximum number of L2-attached SME coprocessor units supported.
const MaxSMEUnits = 16

var (
	// ActiveArchitectures lists the active OS/architecture and enabled SIMD architectures.
	ActiveArchitectures = []string{runtime.GOOS + "/" + runtime.GOARCH}

	// AVX2UseAsm enables handwritten Plan9 AMD64 assembly kernels for AVX2.
	// Set GOMLX_GO_AVX2_ASM=false to disable and use pure-Go archsimd kernels.
	AVX2UseAsm = envutil.MustReadBool(envutil.GoBackendAVX2_ASM, true)

	// AVX512UseAsm enables handwritten Plan9 AMD64 assembly kernels for AVX-512.
	// Set GOMLX_GO_AVX512_ASM=false to disable and use pure-Go archsimd kernels.
	AVX512UseAsm = envutil.MustReadBool(envutil.GoBackendAVX512_ASM, true)

	// NEONUseAsm enables handwritten Plan9 ARM64 assembly kernels for NEON and SME.
	// Set GOMLX_GO_NEON_ASM=false to disable and use pure-Go archsimd kernels.
	NEONUseAsm = envutil.MustReadBool(envutil.GoBackendNEON_ASM, true)

	// SMEUseHybridNEON enables global per-L2-cluster SME locking with per-core NEON co-execution
	// on workers whose cluster SME unit is already occupied.
	SMEUseHybridNEON = true

	// SMEWorkersPerCluster is the default maximum number of concurrent SME workers per L2 cluster.
	SMEWorkersPerCluster int32 = 1

	// NumSMEUnits is the total number of physical L2-attached SME coprocessor units detected on the host.
	NumSMEUnits int

	// NumHighPerfSMEUnits is the number of high-performance (Super / Performance) L2-attached SME units.
	NumHighPerfSMEUnits int

	// NumEfficiencySMEUnits is the number of efficiency (E-core) L2-attached SME units.
	NumEfficiencySMEUnits int
)

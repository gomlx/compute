// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package activations

import (
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
)

const PriorityNEON = gobackend.PriorityArch

// NEONTranscendentalMetadata holds metadata for ARM64 NEON assembly activation kernels.
var NEONTranscendentalMetadata = KernelMetadata{MinParallelChunk: 8192}

func init() {
	if !gobackend.IsNEONAllowed {
		return
	}
	RegisterKernel[float32]("neon:gelu", compute.ActivationGelu, GeluExactFloat32NEON, PriorityNEON, &NEONTranscendentalMetadata)
	RegisterKernel[float32]("neon:geluapprox", compute.ActivationGeluApproximate, GeluApproxFloat32NEON, PriorityNEON, &NEONTranscendentalMetadata)
}

//go:noescape
func geluExactFloat32NEONAsm(in, out unsafe.Pointer, n int)

//go:noescape
func geluApproxFloat32NEONAsm(in, out unsafe.Pointer, n int)

// GeluExactFloat32NEON computes exact GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) using ARM64 NEON assembly.
func GeluExactFloat32NEON(in, out []float32) {
	n := len(in)
	if n == 0 {
		return
	}
	_ = out[n-1]
	vecN := n &^ 3
	if vecN > 0 {
		geluExactFloat32NEONAsm(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), vecN)
	}
	if vecN < n {
		GeluExactFloat32SIMD(in[vecN:], out[vecN:])
	}
}

// GeluApproxFloat32NEON computes approximate GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// using ARM64 NEON assembly.
func GeluApproxFloat32NEON(in, out []float32) {
	n := len(in)
	if n == 0 {
		return
	}
	_ = out[n-1]
	vecN := n &^ 3
	if vecN > 0 {
		geluApproxFloat32NEONAsm(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), vecN)
	}
	if vecN < n {
		GeluApproxFloat32SIMD(in[vecN:], out[vecN:])
	}
}

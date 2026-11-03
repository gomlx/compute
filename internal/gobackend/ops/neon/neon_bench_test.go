// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon_test

import (
	"testing"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops/neon"
)

func BenchmarkNEONOps(b *testing.B) {
	if !gobackend.IsNEONAllowed {
		b.Skip("ARM64 NEON not allowed or not supported on this host")
	}

	const A, B = 128, 1024

	b.Run("TrailingSum_Float32_128x1024", func(b *testing.B) {
		in := make([]float32, A*B)
		out := make([]float32, A)
		b.Run("Scalar", func(b *testing.B) {
			for b.Loop() {
				for a := range A {
					var s float32
					row := in[a*B : (a+1)*B]
					for _, v := range row {
						s += v
					}
					out[a] = s
				}
			}
		})
		b.Run("NEON", func(b *testing.B) {
			inPtr := unsafe.Pointer(&in[0])
			outPtr := unsafe.Pointer(&out[0])
			for b.Loop() {
				neon.DispatchTrailingSumNEON(inPtr, outPtr, A, B, dtypes.Float32)
			}
		})
	})

	b.Run("TrailingSum_Float16_128x1024", func(b *testing.B) {
		in := make([]float16.Float16, A*B)
		out := make([]float16.Float16, A)
		b.Run("Scalar", func(b *testing.B) {
			for b.Loop() {
				for a := range A {
					var s float32
					row := in[a*B : (a+1)*B]
					for _, v := range row {
						s += v.Float32()
					}
					out[a] = float16.FromFloat32(s)
				}
			}
		})
		b.Run("NEON", func(b *testing.B) {
			inPtr := unsafe.Pointer(&in[0])
			outPtr := unsafe.Pointer(&out[0])
			for b.Loop() {
				neon.DispatchTrailingSumNEON(inPtr, outPtr, A, B, dtypes.Float16)
			}
		})
	})

	b.Run("TrailingSum_BFloat16_128x1024", func(b *testing.B) {
		in := make([]bfloat16.BFloat16, A*B)
		out := make([]bfloat16.BFloat16, A)
		b.Run("Scalar", func(b *testing.B) {
			for b.Loop() {
				for a := range A {
					var s float32
					row := in[a*B : (a+1)*B]
					for _, v := range row {
						s += v.Float32()
					}
					out[a] = bfloat16.FromFloat32(s)
				}
			}
		})
		b.Run("NEON", func(b *testing.B) {
			inPtr := unsafe.Pointer(&in[0])
			outPtr := unsafe.Pointer(&out[0])
			for b.Loop() {
				neon.DispatchTrailingSumNEON(inPtr, outPtr, A, B, dtypes.BFloat16)
			}
		})
	})

	b.Run("LeadingSum_Float32_128x1024", func(b *testing.B) {
		in := make([]float32, A*B)
		out := make([]float32, B)
		b.Run("Scalar", func(b *testing.B) {
			for b.Loop() {
				clear(out)
				for a := range A {
					row := in[a*B : (a+1)*B]
					for col := range B {
						out[col] += row[col]
					}
				}
			}
		})
		b.Run("NEON", func(b *testing.B) {
			inPtr := unsafe.Pointer(&in[0])
			outPtr := unsafe.Pointer(&out[0])
			for b.Loop() {
				neon.DispatchLeadingSumNEON(inPtr, outPtr, A, B, dtypes.Float32)
			}
		})
	})

	b.Run("LeadingSum_Float16_128x1024", func(b *testing.B) {
		in := make([]float16.Float16, A*B)
		out := make([]float16.Float16, B)
		b.Run("Scalar", func(b *testing.B) {
			for b.Loop() {
				for col := range B {
					var s float32
					for a := range A {
						s += in[a*B+col].Float32()
					}
					out[col] = float16.FromFloat32(s)
				}
			}
		})
		b.Run("NEON", func(b *testing.B) {
			inPtr := unsafe.Pointer(&in[0])
			outPtr := unsafe.Pointer(&out[0])
			for b.Loop() {
				neon.DispatchLeadingSumNEON(inPtr, outPtr, A, B, dtypes.Float16)
			}
		})
	})

	b.Run("BinaryTrailingAdd_Float32_128x1024", func(b *testing.B) {
		lhs := make([]float32, A*B)
		rhs := make([]float32, A)
		out := make([]float32, A*B)
		b.Run("Scalar", func(b *testing.B) {
			for b.Loop() {
				for a := range A {
					s := rhs[a]
					off := a * B
					for col := range B {
						out[off+col] = lhs[off+col] + s
					}
				}
			}
		})
		b.Run("NEON", func(b *testing.B) {
			lhsPtr := unsafe.Pointer(&lhs[0])
			rhsPtr := unsafe.Pointer(&rhs[0])
			outPtr := unsafe.Pointer(&out[0])
			for b.Loop() {
				neon.DispatchBinaryTrailingNEON(compute.OpTypeAdd, false, lhsPtr, rhsPtr, outPtr, A, B, dtypes.Float32)
			}
		})
	})
}

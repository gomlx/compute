// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package activations_test

import (
	"math/rand/v2"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/activations/avx2"
	"github.com/gomlx/compute/internal/gobackend/activations/avx512"
)

func BenchmarkFlavorsComparison(b *testing.B) {
	sizes := []struct {
		name string
		n    int
	}{
		{"Small_512", 512},
		{"Large_65536", 65536},
	}

	for _, sz := range sizes {
		b.Run(sz.name, func(b *testing.B) {
			inF32 := make([]float32, sz.n)
			outF32 := make([]float32, sz.n)
			for i := range inF32 {
				inF32[i] = rand.Float32()*4 - 2
			}

			// 1. RELU
			b.Run("Relu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.ReluNoSIMD(inF32, outF32)
				}
			})
			b.Run("Relu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.ReluFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("Relu/AVX2_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx2.ReluAVX2(outF32)
				}
			})
			b.Run("Relu/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.ReluAVX512(outF32)
				}
			})

			// 2. SILU
			b.Run("Silu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SiluNoSIMD(inF32, outF32)
				}
			})
			b.Run("Silu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SiluFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("Silu/AVX2_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx2.SiluAVX2(outF32)
				}
			})
			b.Run("Silu/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.SiluAVX512(outF32)
				}
			})

			// 3. TANH
			b.Run("Tanh/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.TanhNoSIMD(inF32, outF32)
				}
			})
			b.Run("Tanh/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.TanhFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("Tanh/AVX2_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx2.TanhAVX2(outF32)
				}
			})
			b.Run("Tanh/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.TanhAVX512(outF32)
				}
			})

			// 4. GeluApproximate
			b.Run("GeluApprox/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.GeluApproxNoSIMD(inF32, outF32)
				}
			})
			b.Run("GeluApprox/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.GeluApproxFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("GeluApprox/AVX2_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx2.GeluAVX2(outF32)
				}
			})
			b.Run("GeluApprox/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.GeluAVX512(outF32)
				}
			})

			// 5. Sigmoid
			b.Run("Sigmoid/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SigmoidNoSIMD(inF32, outF32)
				}
			})
			b.Run("Sigmoid/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SigmoidFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("Sigmoid/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.SigmoidAVX512(outF32)
				}
			})

			// 6. HardSigmoid
			b.Run("HardSigmoid/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.HardSigmoidNoSIMD(inF32, outF32)
				}
			})
			b.Run("HardSigmoid/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.HardSigmoidFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("HardSigmoid/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.HardSigmoidAVX512(outF32)
				}
			})

			// 7. HardSwish
			b.Run("HardSwish/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.HardSwishNoSIMD(inF32, outF32)
				}
			})
			b.Run("HardSwish/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.HardSwishFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("HardSwish/AVX2_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx2.HardSwishAVX2(outF32)
				}
			})
			b.Run("HardSwish/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.HardSwishAVX512(outF32)
				}
			})

			// 8. LeakyRelu
			b.Run("LeakyRelu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.LeakyReluNoSIMD(inF32, outF32)
				}
			})
			b.Run("LeakyRelu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.LeakyReluFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("LeakyRelu/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.LeakyReluAVX512(outF32)
				}
			})

			// 9. Selu
			b.Run("Selu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SeluNoSIMD(inF32, outF32)
				}
			})
			b.Run("Selu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SeluFloat32SIMD(inF32, outF32)
				}
			})
			b.Run("Selu/AVX512_ArchSIMD", func(b *testing.B) {
				for b.Loop() {
					copy(outF32, inF32)
					avx512.SeluAVX512(outF32)
				}
			})

			// 10. BFloat16 & Float16
			inBF16 := make([]bfloat16.BFloat16, sz.n)
			outBF16 := make([]bfloat16.BFloat16, sz.n)
			for i := range inBF16 {
				inBF16[i] = bfloat16.FromFloat32(inF32[i])
			}
			b.Run("BFloat16_Silu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SiluBF16NoSIMD(inBF16, outBF16)
				}
			})
			bf16Simd := activations.MakeBF16KernelFromF32(activations.SiluFloat32SIMD)
			b.Run("BFloat16_Silu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					bf16Simd(inBF16, outBF16)
				}
			})

			inF16 := make([]float16.Float16, sz.n)
			outF16 := make([]float16.Float16, sz.n)
			for i := range inF16 {
				inF16[i] = float16.FromFloat32(inF32[i])
			}
			b.Run("Float16_Silu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SiluF16NoSIMD(inF16, outF16)
				}
			})
			f16Simd := activations.MakeF16KernelFromF32(activations.SiluFloat32SIMD)
			b.Run("Float16_Silu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					f16Simd(inF16, outF16)
				}
			})

			// 11. Float64
			inF64 := make([]float64, sz.n)
			outF64 := make([]float64, sz.n)
			for i := range inF64 {
				inF64[i] = float64(inF32[i])
			}
			b.Run("Float64_Relu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.ReluNoSIMD(inF64, outF64)
				}
			})
			b.Run("Float64_Relu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.ReluFloat64SIMD(inF64, outF64)
				}
			})
			b.Run("Float64_Silu/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SiluNoSIMD(inF64, outF64)
				}
			})
			b.Run("Float64_Silu/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SiluFloat64SIMD(inF64, outF64)
				}
			})
			b.Run("Float64_Sigmoid/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SigmoidNoSIMD(inF64, outF64)
				}
			})
			b.Run("Float64_Sigmoid/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.SigmoidFloat64SIMD(inF64, outF64)
				}
			})
			b.Run("Float64_Tanh/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.TanhNoSIMD(inF64, outF64)
				}
			})
			b.Run("Float64_Tanh/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.TanhFloat64SIMD(inF64, outF64)
				}
			})
			b.Run("Float64_GeluApprox/NoSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.GeluApproxNoSIMD(inF64, outF64)
				}
			})
			b.Run("Float64_GeluApprox/PortableSIMD", func(b *testing.B) {
				for b.Loop() {
					activations.GeluApproxFloat64SIMD(inF64, outF64)
				}
			})
		})
	}
}

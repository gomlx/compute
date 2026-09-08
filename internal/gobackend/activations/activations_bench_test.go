// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package activations_test

import (
	"cmp"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/support/testutil"
)

const (
	PriorityNoSIMD       = 0
	PriorityPortableSIMD = 10
	PriorityAVX2         = 20
	PriorityAVX512       = 30
)

type flavorEntry[T any] struct {
	name     string
	priority int
	fn       func(in, out []T)
}

type flavorRegistry[T any] struct {
	ops     []string
	flavors map[string][]flavorEntry[T]
}

func newFlavorRegistry[T any]() *flavorRegistry[T] {
	return &flavorRegistry[T]{
		flavors: make(map[string][]flavorEntry[T]),
	}
}

func (r *flavorRegistry[T]) Register(op, flavor string, priority int, fn func(in, out []T)) {
	if _, exists := r.flavors[op]; !exists {
		r.ops = append(r.ops, op)
	}
	r.flavors[op] = append(r.flavors[op], flavorEntry[T]{name: flavor, priority: priority, fn: fn})
	slices.SortFunc(r.flavors[op], func(a, b flavorEntry[T]) int {
		return cmp.Compare(a.priority, b.priority)
	})
}

func (r *flavorRegistry[T]) RegisterInPlace(op, flavor string, priority int, fn func(data []T)) {
	r.Register(op, flavor, priority, func(in, out []T) {
		copy(out, in)
		fn(out)
	})
}

var (
	f32Registry  = newFlavorRegistry[float32]()
	bf16Registry = newFlavorRegistry[bfloat16.BFloat16]()
	f16Registry  = newFlavorRegistry[float16.Float16]()
	f64Registry  = newFlavorRegistry[float64]()

	baselineOnce sync.Once
)

func registerBaseline() {
	baselineOnce.Do(func() {
		// Float32
		f32Registry.Register("Relu", "NoSIMD", PriorityNoSIMD, activations.ReluNoSIMD[float32])
		f32Registry.Register("Relu", "PortableSIMD", PriorityPortableSIMD, activations.ReluFloat32SIMD)

		f32Registry.Register("Silu", "NoSIMD", PriorityNoSIMD, activations.SiluNoSIMD[float32])
		f32Registry.Register("Silu", "PortableSIMD", PriorityPortableSIMD, activations.SiluFloat32SIMD)

		f32Registry.Register("Tanh", "NoSIMD", PriorityNoSIMD, activations.TanhNoSIMD[float32])
		f32Registry.Register("Tanh", "PortableSIMD", PriorityPortableSIMD, activations.TanhFloat32SIMD)

		f32Registry.Register("GeluApprox", "NoSIMD", PriorityNoSIMD, activations.GeluApproxNoSIMD[float32])
		f32Registry.Register("GeluApprox", "PortableSIMD", PriorityPortableSIMD, activations.GeluApproxFloat32SIMD)

		f32Registry.Register("Sigmoid", "NoSIMD", PriorityNoSIMD, activations.SigmoidNoSIMD[float32])
		f32Registry.Register("Sigmoid", "PortableSIMD", PriorityPortableSIMD, activations.SigmoidFloat32SIMD)

		f32Registry.Register("HardSigmoid", "NoSIMD", PriorityNoSIMD, activations.HardSigmoidNoSIMD[float32])
		f32Registry.Register("HardSigmoid", "PortableSIMD", PriorityPortableSIMD, activations.HardSigmoidFloat32SIMD)

		f32Registry.Register("HardSwish", "NoSIMD", PriorityNoSIMD, activations.HardSwishNoSIMD[float32])
		f32Registry.Register("HardSwish", "PortableSIMD", PriorityPortableSIMD, activations.HardSwishFloat32SIMD)

		f32Registry.Register("LeakyRelu", "NoSIMD", PriorityNoSIMD, activations.LeakyReluNoSIMD[float32])
		f32Registry.Register("LeakyRelu", "PortableSIMD", PriorityPortableSIMD, activations.LeakyReluFloat32SIMD)

		f32Registry.Register("Selu", "NoSIMD", PriorityNoSIMD, activations.SeluNoSIMD[float32])
		f32Registry.Register("Selu", "PortableSIMD", PriorityPortableSIMD, activations.SeluFloat32SIMD)

		// BFloat16
		bf16Registry.Register("BFloat16_Relu", "NoSIMD", PriorityNoSIMD, activations.ReluBF16NoSIMD)
		bf16Registry.Register("BFloat16_Relu", "PortableSIMD", PriorityPortableSIMD, activations.ReluBFloat16SIMD)
		bf16Registry.Register("BFloat16_Silu", "NoSIMD", PriorityNoSIMD, activations.SiluBF16NoSIMD)
		bf16Registry.Register("BFloat16_Silu", "PortableSIMD", PriorityPortableSIMD, activations.SiluBFloat16SIMD)

		// Float16
		f16Registry.Register("Float16_Relu", "NoSIMD", PriorityNoSIMD, activations.ReluF16NoSIMD)
		f16Registry.Register("Float16_Relu", "PortableSIMD", PriorityPortableSIMD, activations.ReluFloat16SIMD)
		f16Registry.Register("Float16_Silu", "NoSIMD", PriorityNoSIMD, activations.SiluF16NoSIMD)
		f16Registry.Register("Float16_Silu", "PortableSIMD", PriorityPortableSIMD, activations.SiluFloat16SIMD)

		// Float64
		f64Registry.Register("Float64_Relu", "NoSIMD", PriorityNoSIMD, activations.ReluNoSIMD[float64])
		f64Registry.Register("Float64_Relu", "PortableSIMD", PriorityPortableSIMD, activations.ReluFloat64SIMD)
		f64Registry.Register("Float64_Silu", "NoSIMD", PriorityNoSIMD, activations.SiluNoSIMD[float64])
		f64Registry.Register("Float64_Silu", "PortableSIMD", PriorityPortableSIMD, activations.SiluFloat64SIMD)
		f64Registry.Register("Float64_Sigmoid", "NoSIMD", PriorityNoSIMD, activations.SigmoidNoSIMD[float64])
		f64Registry.Register("Float64_Sigmoid", "PortableSIMD", PriorityPortableSIMD, activations.SigmoidFloat64SIMD)
		f64Registry.Register("Float64_Tanh", "NoSIMD", PriorityNoSIMD, activations.TanhNoSIMD[float64])
		f64Registry.Register("Float64_Tanh", "PortableSIMD", PriorityPortableSIMD, activations.TanhFloat64SIMD)
		f64Registry.Register("Float64_GeluApprox", "NoSIMD", PriorityNoSIMD, activations.GeluApproxNoSIMD[float64])
		f64Registry.Register("Float64_GeluApprox", "PortableSIMD", PriorityPortableSIMD, activations.GeluApproxFloat64SIMD)
	})
}

func init() {
	registerBaseline()
}

func runRegistryBenchmarks[T any](b *testing.B, r *flavorRegistry[T], in, out []T) {
	for _, op := range r.ops {
		for _, flavor := range r.flavors[op] {
			b.Run(op+"/"+flavor.name, func(b *testing.B) {
				for b.Loop() {
					flavor.fn(in, out)
				}
			})
		}
	}
}

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
			runRegistryBenchmarks(b, f32Registry, inF32, outF32)

			inBF16 := make([]bfloat16.BFloat16, sz.n)
			outBF16 := make([]bfloat16.BFloat16, sz.n)
			for i := range inBF16 {
				inBF16[i] = bfloat16.FromFloat32(inF32[i])
			}
			runRegistryBenchmarks(b, bf16Registry, inBF16, outBF16)

			inF16 := make([]float16.Float16, sz.n)
			outF16 := make([]float16.Float16, sz.n)
			for i := range inF16 {
				inF16[i] = float16.FromFloat32(inF32[i])
			}
			runRegistryBenchmarks(b, f16Registry, inF16, outF16)

			inF64 := make([]float64, sz.n)
			outF64 := make([]float64, sz.n)
			for i := range inF64 {
				inF64[i] = float64(inF32[i])
			}
			runRegistryBenchmarks(b, f64Registry, inF64, outF64)
		})
	}
}

func TestAllAvailableSIMDs(t *testing.T) {
	sizes := []int{1, 7, 8, 15, 16, 31, 32, 65, 512}
	for _, sz := range sizes {
		inF32 := make([]float32, sz)
		for i := range inF32 {
			inF32[i] = float32(i-sz/2) * 0.1
		}

		for _, op := range f32Registry.ops {
			flavors := f32Registry.flavors[op]
			if len(flavors) <= 1 {
				continue
			}
			baseline := make([]float32, sz)
			flavors[0].fn(inF32, baseline)

			for _, flavor := range flavors[1:] {
				t.Run(op+"/"+flavor.name, func(t *testing.T) {
					got := make([]float32, sz)
					flavor.fn(inF32, got)
					for i := range got {
						if ok, diff := testutil.IsInRelativeDelta(baseline[i], got[i], 1e-3); !ok {
							if math.Abs(float64(baseline[i]-got[i])) > 1e-4 {
								t.Fatalf("[%d] got %f, want %f, diff=%s", i, got[i], baseline[i], diff)
							}
						}
					}
				})
			}
		}
	}
}

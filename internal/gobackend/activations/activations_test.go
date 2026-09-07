// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package activations_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend/activations"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx512"
	"github.com/gomlx/compute/support/testutil"
)

func TestActivationsFloat32(t *testing.T) {
	sizes := []int{1, 7, 8, 15, 16, 31, 32, 65, 1000, 8192}

	for _, sz := range sizes {
		inputs := make([]float32, sz)
		for i := range inputs {
			inputs[i] = float32(i-sz/2) * 0.1
		}

		t.Run("Relu", func(t *testing.T) {
			data := append([]float32(nil), inputs...)
			fn := activations.Get[float32](compute.ActivationRelu)
			if fn == nil {
				t.Fatal("Relu implementation not found")
			}
			fn(data)
			for i, x := range inputs {
				want := max(x, 0)
				if data[i] != want {
					t.Fatalf("Relu[%d] got %f, want %f", i, data[i], want)
				}
			}
		})

		t.Run("HardSwish", func(t *testing.T) {
			data := append([]float32(nil), inputs...)
			fn := activations.Get[float32](compute.ActivationHardSwish)
			if fn == nil {
				t.Fatal("HardSwish implementation not found")
			}
			fn(data)
			for i, x := range inputs {
				shapeX := min(max(x*(1.0/6.0)+0.5, 0), 1)
				want := x * shapeX
				if ok, diff := testutil.IsInDelta(want, data[i], 1e-5); !ok {
					t.Fatalf("HardSwish[%d] mismatch: %s", i, diff)
				}
			}
		})

		t.Run("Silu", func(t *testing.T) {
			data := append([]float32(nil), inputs...)
			fn := activations.Get[float32](compute.ActivationSilu)
			if fn == nil {
				t.Fatal("Silu implementation not found")
			}
			fn(data)
			for i, x := range inputs {
				want := float32(float64(x) / (1.0 + math.Exp(float64(-x))))
				// Fast exp allows relative error < 1e-4
				if ok, diff := testutil.IsInRelativeDelta(want, data[i], 1e-4); !ok {
					if math.Abs(float64(want-data[i])) > 1e-5 {
						t.Fatalf("Silu[%d] (x=%f) mismatch: %s", i, x, diff)
					}
				}
			}
		})

		t.Run("Tanh", func(t *testing.T) {
			data := append([]float32(nil), inputs...)
			fn := activations.Get[float32](compute.ActivationTanh)
			if fn == nil {
				t.Fatal("Tanh implementation not found")
			}
			fn(data)
			for i, x := range inputs {
				want := float32(math.Tanh(float64(x)))
				if ok, diff := testutil.IsInDelta(want, data[i], 1e-4); !ok {
					t.Fatalf("Tanh[%d] (x=%f) mismatch: %s", i, x, diff)
				}
			}
		})

		t.Run("Gelu", func(t *testing.T) {
			data := append([]float32(nil), inputs...)
			fn := activations.Get[float32](compute.ActivationGelu)
			if fn == nil {
				t.Fatal("Gelu implementation not found")
			}
			fn(data)
			sqrt2ByPi := float64(math.Sqrt(2.0 / math.Pi))
			for i, x := range inputs {
				xf := float64(x)
				inner := sqrt2ByPi * (xf + 0.044715*xf*xf*xf)
				want := float32(xf * 0.5 * (1.0 + math.Tanh(inner)))
				if ok, diff := testutil.IsInDelta(want, data[i], 1e-4); !ok {
					t.Fatalf("Gelu[%d] (x=%f) mismatch: %s", i, x, diff)
				}
			}
		})
	}
}

func TestActivationsBFloat16(t *testing.T) {
	data := []bfloat16.BFloat16{
		bfloat16.FromFloat32(-2.0),
		bfloat16.FromFloat32(-0.5),
		bfloat16.FromFloat32(0.0),
		bfloat16.FromFloat32(0.5),
		bfloat16.FromFloat32(2.0),
	}
	reluFn := activations.Get[bfloat16.BFloat16](compute.ActivationRelu)
	if reluFn == nil {
		t.Fatal("Relu BF16 implementation not found")
	}
	res := append([]bfloat16.BFloat16(nil), data...)
	reluFn(res)

	if res[0].Float32() != 0 || res[1].Float32() != 0 || res[2].Float32() != 0 {
		t.Errorf("Relu BF16 negative values should be zero")
	}
	if res[3].Float32() != 0.5 || res[4].Float32() != 2.0 {
		t.Errorf("Relu BF16 positive values unchanged")
	}
}

func TestActivationsFloat16(t *testing.T) {
	data := []float16.Float16{
		float16.FromFloat32(-2.0),
		float16.FromFloat32(-0.5),
		float16.FromFloat32(0.0),
		float16.FromFloat32(0.5),
		float16.FromFloat32(2.0),
	}
	reluFn := activations.Get[float16.Float16](compute.ActivationRelu)
	if reluFn == nil {
		t.Fatal("Relu F16 implementation not found")
	}
	res := append([]float16.Float16(nil), data...)
	reluFn(res)

	if res[0].Float32() != 0 || res[1].Float32() != 0 || res[2].Float32() != 0 {
		t.Errorf("Relu F16 negative values should be zero")
	}
	if res[3].Float32() != 0.5 || res[4].Float32() != 2.0 {
		t.Errorf("Relu F16 positive values unchanged")
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
)

func TestAVX2LayerNormCorrectness(t *testing.T) {
	rand.Seed(42)

	// Test a variety of hidden sizes, including odd sizes and sizes around vector boundaries
	testSizes := []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 768, 1024}
	outerSize := 5
	epsilon := 1e-5

	t.Run("Float32", func(t *testing.T) {
		for _, normSize := range testSizes {
			in := make([]float32, outerSize*normSize)
			out := make([]float32, outerSize*normSize)
			ref := make([]float32, outerSize*normSize)
			gamma := make([]float32, normSize)
			beta := make([]float32, normSize)

			for i := range in {
				in[i] = rand.Float32()*10.0 - 5.0
			}
			for i := range gamma {
				gamma[i] = rand.Float32() + 0.5
				beta[i] = rand.Float32()*2.0 - 1.0
			}

			for _, withGamma := range []bool{true, false} {
				for _, withBeta := range []bool{true, false} {
					var gPtr, bPtr unsafe.Pointer
					var gSlice, bSlice []float32
					if withGamma {
						gPtr = unsafe.Pointer(&gamma[0])
						gSlice = gamma
					}
					if withBeta {
						bPtr = unsafe.Pointer(&beta[0])
						bSlice = beta
					}

					// Reference calculation
					normSizeF := float32(normSize)
					for outer := 0; outer < outerSize; outer++ {
						base := outer * normSize
						var sum float32
						for i := 0; i < normSize; i++ {
							sum += in[base+i]
						}
						mean := sum / normSizeF

						var varSum float32
						for i := 0; i < normSize; i++ {
							d := in[base+i] - mean
							varSum += d * d
						}
						variance := varSum / normSizeF
						invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))

						for i := 0; i < normSize; i++ {
							val := (in[base+i] - mean) * invStd
							if gSlice != nil {
								val *= gSlice[i]
							}
							if bSlice != nil {
								val += bSlice[i]
							}
							ref[base+i] = val
						}
					}

					ok := DispatchLayerNormAVX2(
						unsafe.Pointer(&in[0]),
						unsafe.Pointer(&out[0]),
						gPtr, bPtr,
						outerSize, normSize, epsilon,
						dtypes.Float32,
					)
					if !ok {
						t.Fatalf("DispatchLayerNormAVX2 failed for B=%d", normSize)
					}

					for i := range out {
						diff := math.Abs(float64(out[i] - ref[i]))
						if diff > 1e-4 {
							t.Fatalf("B=%d withGamma=%v withBeta=%v mismatch at idx %d: got %f, want %f, diff %f",
								normSize, withGamma, withBeta, i, out[i], ref[i], diff)
						}
					}
				}
			}
		}
	})

	t.Run("Float64", func(t *testing.T) {
		for _, normSize := range testSizes {
			in := make([]float64, outerSize*normSize)
			out := make([]float64, outerSize*normSize)
			ref := make([]float64, outerSize*normSize)
			gamma := make([]float64, normSize)
			beta := make([]float64, normSize)

			for i := range in {
				in[i] = rand.Float64()*10.0 - 5.0
			}
			for i := range gamma {
				gamma[i] = rand.Float64() + 0.5
				beta[i] = rand.Float64()*2.0 - 1.0
			}

			for _, withGamma := range []bool{true, false} {
				for _, withBeta := range []bool{true, false} {
					var gPtr, bPtr unsafe.Pointer
					var gSlice, bSlice []float64
					if withGamma {
						gPtr = unsafe.Pointer(&gamma[0])
						gSlice = gamma
					}
					if withBeta {
						bPtr = unsafe.Pointer(&beta[0])
						bSlice = beta
					}

					// Reference calculation
					normSizeF := float64(normSize)
					for outer := 0; outer < outerSize; outer++ {
						base := outer * normSize
						var sum float64
						for i := 0; i < normSize; i++ {
							sum += in[base+i]
						}
						mean := sum / normSizeF

						var varSum float64
						for i := 0; i < normSize; i++ {
							d := in[base+i] - mean
							varSum += d * d
						}
						variance := varSum / normSizeF
						invStd := 1.0 / math.Sqrt(variance+epsilon)

						for i := 0; i < normSize; i++ {
							val := (in[base+i] - mean) * invStd
							if gSlice != nil {
								val *= gSlice[i]
							}
							if bSlice != nil {
								val += bSlice[i]
							}
							ref[base+i] = val
						}
					}

					ok := DispatchLayerNormAVX2(
						unsafe.Pointer(&in[0]),
						unsafe.Pointer(&out[0]),
						gPtr, bPtr,
						outerSize, normSize, epsilon,
						dtypes.Float64,
					)
					if !ok {
						t.Fatalf("DispatchLayerNormAVX2 failed for B=%d", normSize)
					}

					for i := range out {
						diff := math.Abs(out[i] - ref[i])
						if diff > 1e-5 {
							t.Fatalf("Float64 B=%d withGamma=%v withBeta=%v mismatch at idx %d: got %f, want %f, diff %f",
								normSize, withGamma, withBeta, i, out[i], ref[i], diff)
						}
					}
				}
			}
		}
	})
}

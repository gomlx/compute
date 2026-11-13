// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
)

func TestNEONLayerNormCorrectness(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}
	rng := rand.New(rand.NewSource(42))

	testSizes := []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 384, 768}
	outerSize := 5
	epsilon := 1e-5

	for _, normSize := range testSizes {
		in := make([]float32, outerSize*normSize)
		res := make([]float32, outerSize*normSize)
		out := make([]float32, outerSize*normSize)
		ref := make([]float32, outerSize*normSize)
		gamma := make([]float32, normSize)
		beta := make([]float32, normSize)

		for i := range in {
			in[i] = rng.Float32()*10.0 - 5.0
			res[i] = rng.Float32()*6.0 - 3.0
		}
		for i := range gamma {
			gamma[i] = rng.Float32() + 0.5
			beta[i] = rng.Float32()*2.0 - 1.0
		}

		for _, withResidual := range []bool{false, true} {
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

					normSizeF := float32(normSize)
					for outer := range outerSize {
						base := outer * normSize
						var sum float32
						for i := range normSize {
							v := in[base+i]
							if withResidual {
								v += res[base+i]
							}
							sum += v
						}
						mean := sum / normSizeF

						var varSum float32
						for i := range normSize {
							v := in[base+i]
							if withResidual {
								v += res[base+i]
							}
							d := v - mean
							varSum += d * d
						}
						variance := varSum / normSizeF
						invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))

						for i := range normSize {
							v := in[base+i]
							if withResidual {
								v += res[base+i]
							}
							val := (v - mean) * invStd
							if gSlice != nil {
								val *= gSlice[i]
							}
							if bSlice != nil {
								val += bSlice[i]
							}
							ref[base+i] = val
						}
					}

					var ok bool
					if withResidual {
						ok = DispatchLayerNormResidualNEON(
							unsafe.Pointer(&in[0]),
							unsafe.Pointer(&res[0]),
							unsafe.Pointer(&out[0]),
							gPtr, bPtr,
							outerSize, normSize, epsilon,
							dtypes.Float32,
						)
					} else {
						ok = DispatchLayerNormNEON(
							unsafe.Pointer(&in[0]),
							unsafe.Pointer(&out[0]),
							gPtr, bPtr,
							outerSize, normSize, epsilon,
							dtypes.Float32,
						)
					}
					if !ok {
						t.Fatalf("DispatchLayerNormNEON failed for normSize=%d", normSize)
					}

					for i := range out {
						diff := math.Abs(float64(out[i] - ref[i]))
						if diff > 1e-4 {
							t.Fatalf("normSize=%d res=%v gamma=%v beta=%v mismatch at idx %d: got %f, want %f, diff %f",
								normSize, withResidual, withGamma, withBeta, i, out[i], ref[i], diff)
						}
					}
				}
			}
		}
	}
}

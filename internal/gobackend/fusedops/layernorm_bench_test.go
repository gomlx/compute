// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package fusedops

import (
	"flag"
	"fmt"
	"math/rand"
	"testing"
	"time"

	_ "github.com/gomlx/compute/internal/gobackend/fusedops/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/fusedops/avx512"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
)

var (
	repeatLayerNormBench = flag.Int("repeat_layernorm_bench", 0, "If > 0, runs comparative benchmark table between generic and SIMD LayerNorm")
	layerNormSIMDFlag    = flag.Bool("layernorm_simd", true, "Whether standard benchmarks run with SIMD enabled")
)

func BenchmarkLayerNormTrailing(b *testing.B) {
	be, err := NewBackend()
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	defer be.Finalize()

	outerSize := 100
	for _, normSize := range []int{16, 64, 128, 256, 768, 1024, 4096} {
		for _, withAffine := range []bool{true, false} {
			affineName := "Affine"
			if !withAffine {
				affineName = "NoAffine"
			}
			b.Run(fmt.Sprintf("Float32_%s_B%d", affineName, normSize), func(b *testing.B) {
				inShape := shapes.Make(dtypes.Float32, outerSize, normSize)
				inBuf, _ := be.GetBuffer(inShape)
				outBuf, _ := be.GetBuffer(inShape)
				var gammaBuf, betaBuf *gobackend.Buffer
				if withAffine {
					paramShape := shapes.Make(dtypes.Float32, normSize)
					gammaBuf, _ = be.GetBuffer(paramShape)
					betaBuf, _ = be.GetBuffer(paramShape)
				}
				epsilon := 1e-5

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if *layerNormSIMDFlag {
						simdLayerNormTrailingAxesFloat32(
							inBuf.Flat.([]float32),
							outBuf.Flat.([]float32),
							flatOrNil[float32](gammaBuf),
							flatOrNil[float32](betaBuf),
							normSize,
							epsilon,
						)
					} else {
						layerNormTrailingAxes[float32](
							inBuf.Flat.([]float32),
							outBuf.Flat.([]float32),
							flatOrNil[float32](gammaBuf),
							flatOrNil[float32](betaBuf),
							normSize,
							epsilon,
						)
					}
				}
			})
		}
	}
}

func flatOrNil[T any](b *gobackend.Buffer) []T {
	if b == nil {
		return nil
	}
	return b.Flat.([]T)
}

func TestLayerNormBenchmark(t *testing.T) {
	if *repeatLayerNormBench <= 0 {
		t.Skip("skipping layernorm comparison benchmark; specify -repeat_layernorm_bench=N to run")
	}

	be, err := NewBackend()
	if err != nil {
		t.Fatalf("failed to create backend: %+v", err)
	}
	defer be.Finalize()

	outerSize := 100
	normSizes := []int{16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096}
	epsilon := 1e-5

	fmt.Println("\n### LayerNorm Trailing Benchmark Comparison (Float32)")
	fmt.Println("| HiddenDim (B) | Affine | Generic Latency | SIMD Latency | Speedup (SIMD / Generic) | Winner |")
	fmt.Println("|---|---|---|---|---|---|")

	for _, normSize := range normSizes {
		for _, withAffine := range []bool{true, false} {
			inShape := shapes.Make(dtypes.Float32, outerSize, normSize)
			inBuf, _ := be.GetBuffer(inShape)
			outBuf, _ := be.GetBuffer(inShape)

			inData := inBuf.Flat.([]float32)
			for i := range inData {
				inData[i] = rand.Float32()
			}

			var gammaBuf, betaBuf *gobackend.Buffer
			if withAffine {
				paramShape := shapes.Make(dtypes.Float32, normSize)
				gammaBuf, _ = be.GetBuffer(paramShape)
				betaBuf, _ = be.GetBuffer(paramShape)
				for i := range normSize {
					gammaBuf.Flat.([]float32)[i] = 1.0
					betaBuf.Flat.([]float32)[i] = 0.5
				}
			}

			iters := 200
			if normSize >= 1024 {
				iters = 50
			}

			// Warmup
			for i := 0; i < 5; i++ {
				layerNormTrailingAxes[float32](inData, outBuf.Flat.([]float32), flatOrNil[float32](gammaBuf), flatOrNil[float32](betaBuf), normSize, epsilon)
				simdLayerNormTrailingAxesFloat32(inData, outBuf.Flat.([]float32), flatOrNil[float32](gammaBuf), flatOrNil[float32](betaBuf), normSize, epsilon)
			}

			// Generic
			t0 := time.Now()
			for i := 0; i < iters; i++ {
				layerNormTrailingAxes[float32](inData, outBuf.Flat.([]float32), flatOrNil[float32](gammaBuf), flatOrNil[float32](betaBuf), normSize, epsilon)
			}
			genericDur := time.Since(t0) / time.Duration(iters)

			// SIMD
			t1 := time.Now()
			for i := 0; i < iters; i++ {
				simdLayerNormTrailingAxesFloat32(inData, outBuf.Flat.([]float32), flatOrNil[float32](gammaBuf), flatOrNil[float32](betaBuf), normSize, epsilon)
			}
			simdDur := time.Since(t1) / time.Duration(iters)

			ratio := float64(simdDur) / float64(genericDur)
			winner := "TIE"
			if ratio < 0.90 {
				winner = "SIMD"
			} else if ratio > 1.10 {
				winner = "GENERIC"
			}

			affineStr := "Yes"
			if !withAffine {
				affineStr = "No"
			}
			fmt.Printf("| %13d | %6s | %15s | %12s | %24.2f | %-6s |\n",
				normSize, affineStr, genericDur, simdDur, ratio, winner)
		}
	}
}

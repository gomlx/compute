// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package ops

import (
	"fmt"
	"testing"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/pkg/errors"
)

func runGenericReduce(opType compute.OpType, operand, output *gobackend.Buffer, cfg ReduceConfig, dtype dtypes.DType) error {
	it := &ReduceOutputIterator{Config: cfg}
	var reduceFn GenericReduceFn
	switch opType {
	case compute.OpTypeReduceSum:
		fnAny, err := reduceSumDTypeMap.Get(dtype)
		if err != nil {
			return err
		}
		reduceFn = fnAny.(GenericReduceFn)
	case compute.OpTypeReduceMax:
		fnAny, err := reduceMaxDTypeMap.Get(dtype)
		if err != nil {
			return err
		}
		reduceFn = fnAny.(GenericReduceFn)
	case compute.OpTypeReduceMin:
		fnAny, err := reduceMinDTypeMap.Get(dtype)
		if err != nil {
			return err
		}
		reduceFn = fnAny.(GenericReduceFn)
	case compute.OpTypeReduceProduct:
		fnAny, err := reduceProductDTypeMap.Get(dtype)
		if err != nil {
			return err
		}
		reduceFn = fnAny.(GenericReduceFn)
	default:
		return errors.Errorf("unsupported op: %s", opType)
	}
	return reduceFn(operand, output, it, dtype)
}

func runSIMDReduce(opType compute.OpType, operand, output *gobackend.Buffer, cfg ReduceConfig, dtype dtypes.DType) error {
	switch opType {
	case compute.OpTypeReduceSum:
		return dispatchReduceSumSIMD(operand, output, cfg, dtype)
	case compute.OpTypeReduceMax:
		return dispatchReduceMaxSIMD(operand, output, cfg, dtype)
	case compute.OpTypeReduceMin:
		return dispatchReduceMinSIMD(operand, output, cfg, dtype)
	case compute.OpTypeReduceProduct:
		return dispatchReduceProductSIMD(operand, output, cfg, dtype)
	default:
		return errors.Errorf("unsupported op: %s", opType)
	}
}

func measureMedianDuration(fn func() error, minDuration time.Duration, minIterations int) (time.Duration, error) {
	// 1. Warmup
	for range 200 {
		if err := fn(); err != nil {
			return 0, err
		}
	}

	// 2. Measure single batch to calibrate batch size to ~5 microseconds
	start := time.Now()
	for range 10 {
		if err := fn(); err != nil {
			return 0, err
		}
	}
	elapsed10 := time.Since(start)
	perOp := elapsed10 / 10
	batchSize := 1
	if perOp > 0 {
		batchSize = max(1, int((5*time.Microsecond)/perOp))
	}

	sampler := testutil.NewDurationSampler(16 * 1024)
	startTime := time.Now()
	iterations := 0

	for time.Since(startTime) < minDuration || iterations < minIterations {
		t0 := time.Now()
		for range batchSize {
			if err := fn(); err != nil {
				return 0, err
			}
		}
		tBatch := time.Since(t0)
		sampler.Sample(tBatch / time.Duration(batchSize))
		iterations += batchSize
	}

	return sampler.Median(), nil
}

// TestFindReduceThresholds benchmarks Reduce operations across dimensions, comparing
// Generic (scalar) vs SIMD medians using testutil.DurationSampler.
func TestFindReduceThresholds(t *testing.T) {
	backendGeneric, err := gobackend.New("")
	if err != nil {
		t.Fatalf("failed to create backend: %+v", err)
	}
	be := backendGeneric.(*gobackend.Backend)
	defer be.Finalize()

	testDTypes := []dtypes.DType{
		dtypes.Float32,
		dtypes.Float64,
		dtypes.Int32,
		dtypes.Float16,
		dtypes.BFloat16,
	}

	fmt.Println("\n==========================================================================================")
	fmt.Println("                       REDUCE BENCHMARK: SCALAR vs SIMD (MEDIANS)                         ")
	fmt.Println("==========================================================================================")

	// -------------------------------------------------------------------------
	// 1. REDUCE TRAILING: shape [A, B] -> [A] (inner axis B reduced)
	// -------------------------------------------------------------------------
	fmt.Println("\n### 1. ReduceTrailing: shape [A, B] -> [A] (reducing inner dimension B)")
	fmt.Println("| DType | A | B | Scalar Median | SIMD Median | Ratio (SIMD/Scalar) | Faster |")
	fmt.Println("|:---|---:|---:|---:|---:|---:|:---|")

	for _, dt := range testDTypes {
		bValues := []int{1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256}
		a := 100

		for _, b := range bValues {
			inShape := shapes.Make(dt, a, b)
			outShape := shapes.Make(dt, a)
			inBuf, err := be.GetBuffer(inShape)
			if err != nil {
				t.Fatalf("GetBuffer inShape failed: %+v", err)
			}
			outBuf, err := be.GetBuffer(outShape)
			if err != nil {
				t.Fatalf("GetBuffer outShape failed: %+v", err)
			}

			cfg := ReduceConfig{
				Pattern: ReduceTrailing,
				A:       a,
				B:       b,
				Axes:    []int{1},
			}

			scalarMed, err := measureMedianDuration(func() error {
				return runGenericReduce(compute.OpTypeReduceSum, inBuf, outBuf, cfg, dt)
			}, 10*time.Millisecond, 500)
			if err != nil {
				t.Fatalf("measure scalar failed: %+v", err)
			}

			simdMed, err := measureMedianDuration(func() error {
				return runSIMDReduce(compute.OpTypeReduceSum, inBuf, outBuf, cfg, dt)
			}, 10*time.Millisecond, 500)
			if err != nil {
				t.Fatalf("measure simd failed: %+v", err)
			}

			ratio := float64(simdMed) / float64(scalarMed)
			faster := "SIMD"
			if ratio > 1.05 {
				faster = "**SCALAR**"
			} else if ratio >= 0.95 {
				faster = "TIE"
			}

			fmt.Printf("| %-7s | %3d | %3d | %9s | %9s | %18.2f | %-10s |\n",
				dt, a, b, scalarMed, simdMed, ratio, faster)

			be.PutBuffer(inBuf)
			be.PutBuffer(outBuf)
		}
	}

	// -------------------------------------------------------------------------
	// 2. REDUCE ALL: shape [N] -> [1]
	// -------------------------------------------------------------------------
	fmt.Println("\n### 2. ReduceAll: shape [N] -> [1] (reducing entire tensor)")
	fmt.Println("| DType | N | Scalar Median | SIMD Median | Ratio (SIMD/Scalar) | Faster |")
	fmt.Println("|:---|---:|---:|---:|---:|:---|")

	for _, dt := range testDTypes {
		nValues := []int{2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 1024}
		for _, n := range nValues {
			inShape := shapes.Make(dt, n)
			outShape := shapes.Make(dt)
			inBuf, err := be.GetBuffer(inShape)
			if err != nil {
				t.Fatalf("GetBuffer inShape failed: %+v", err)
			}
			outBuf, err := be.GetBuffer(outShape)
			if err != nil {
				t.Fatalf("GetBuffer outShape failed: %+v", err)
			}

			cfg := ReduceConfig{
				Pattern: ReduceAll,
				A:       n,
				B:       1,
				Axes:    []int{0},
			}

			scalarMed, err := measureMedianDuration(func() error {
				return runGenericReduce(compute.OpTypeReduceSum, inBuf, outBuf, cfg, dt)
			}, 10*time.Millisecond, 500)
			if err != nil {
				t.Fatalf("measure scalar failed: %+v", err)
			}

			simdMed, err := measureMedianDuration(func() error {
				return runSIMDReduce(compute.OpTypeReduceSum, inBuf, outBuf, cfg, dt)
			}, 10*time.Millisecond, 500)
			if err != nil {
				t.Fatalf("measure simd failed: %+v", err)
			}

			ratio := float64(simdMed) / float64(scalarMed)
			faster := "SIMD"
			if ratio > 1.05 {
				faster = "**SCALAR**"
			} else if ratio >= 0.95 {
				faster = "TIE"
			}

			fmt.Printf("| %-7s | %4d | %9s | %9s | %18.2f | %-10s |\n",
				dt, n, scalarMed, simdMed, ratio, faster)

			be.PutBuffer(inBuf)
			be.PutBuffer(outBuf)
		}
	}

	// -------------------------------------------------------------------------
	// 3. REDUCE LEADING: shape [A, B] -> [B] (outer axis A reduced)
	// -------------------------------------------------------------------------
	fmt.Println("\n### 3. ReduceLeading: shape [A, B] -> [B] (reducing outer dimension A)")
	fmt.Println("| DType | A | B | Scalar Median | SIMD Median | Ratio (SIMD/Scalar) | Faster |")
	fmt.Println("|:---|---:|---:|---:|---:|---:|:---|")

	for _, dt := range testDTypes {
		aValues := []int{2, 4, 8, 16, 64}
		bValues := []int{2, 4, 8, 16, 32, 64, 256}

		for _, a := range aValues {
			for _, b := range bValues {
				inShape := shapes.Make(dt, a, b)
				outShape := shapes.Make(dt, b)
				inBuf, err := be.GetBuffer(inShape)
				if err != nil {
					t.Fatalf("GetBuffer inShape failed: %+v", err)
				}
				outBuf, err := be.GetBuffer(outShape)
				if err != nil {
					t.Fatalf("GetBuffer outShape failed: %+v", err)
				}

				cfg := ReduceConfig{
					Pattern: ReduceLeading,
					A:       a,
					B:       b,
					Axes:    []int{0},
				}

				scalarMed, err := measureMedianDuration(func() error {
					return runGenericReduce(compute.OpTypeReduceSum, inBuf, outBuf, cfg, dt)
				}, 10*time.Millisecond, 500)
				if err != nil {
					t.Fatalf("measure scalar failed: %+v", err)
				}

				simdMed, err := measureMedianDuration(func() error {
					return runSIMDReduce(compute.OpTypeReduceSum, inBuf, outBuf, cfg, dt)
				}, 10*time.Millisecond, 500)
				if err != nil {
					t.Fatalf("measure simd failed: %+v", err)
				}

				ratio := float64(simdMed) / float64(scalarMed)
				faster := "SIMD"
				if ratio > 1.05 {
					faster = "**SCALAR**"
				} else if ratio >= 0.95 {
					faster = "TIE"
				}

				fmt.Printf("| %-7s | %3d | %3d | %9s | %9s | %18.2f | %-10s |\n",
					dt, a, b, scalarMed, simdMed, ratio, faster)

				be.PutBuffer(inBuf)
				be.PutBuffer(outBuf)
			}
		}
	}
}

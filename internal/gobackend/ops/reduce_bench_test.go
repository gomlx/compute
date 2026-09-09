// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package ops

import (
	"flag"
	"fmt"
	"testing"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx512"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/pkg/errors"
)

var flagRepeatThresholdTest = flag.Int("repeat_threshold_test", 0, "Number of times to repeat threshold benchmark, taking the minimum value")

type benchResult struct {
	scalarMin time.Duration
	simdMin   time.Duration
}

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

func formatThreshold(val int) string {
	switch val {
	case ThresholdNone:
		return "ThresholdNone"
	case ThresholdAlwaysFallBack:
		return "ThresholdAlwaysFallBack"
	default:
		return fmt.Sprintf("%d", val)
	}
}

// TestFindReduceThresholds benchmarks Reduce operations across dimensions, comparing
// Generic (scalar) vs SIMD medians using testutil.DurationSampler.
func TestFindReduceThresholds(t *testing.T) {
	if *flagRepeatThresholdTest < 1 {
		t.Skip("Run with -repeat_threshold_test=<n> to benchmark, repeating <n> times, with n > 1")
	}
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
		dtypes.Uint32,
		dtypes.Int64,
		dtypes.Uint64,
		dtypes.Int16,
		dtypes.Uint16,
		dtypes.Int8,
		dtypes.Uint8,
		dtypes.Float16,
		dtypes.BFloat16,
	}

	fmt.Println("\n==========================================================================================")
	fmt.Println("                       REDUCE BENCHMARK: SCALAR vs SIMD (MEDIANS)                         ")
	fmt.Println("==========================================================================================")

	repeats := max(1, *flagRepeatThresholdTest)

	// -------------------------------------------------------------------------
	// 1. REDUCE TRAILING: shape [A, B] -> [A] (inner axis B reduced)
	// -------------------------------------------------------------------------
	type trailingKey struct {
		dt dtypes.DType
		b  int
	}
	minTrailing := make(map[trailingKey]benchResult)
	bValuesTrailing := []int{2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024}
	aTrailing := 100

	for range repeats {
		for _, dt := range testDTypes {
			for _, b := range bValuesTrailing {
				inShape := shapes.Make(dt, aTrailing, b)
				outShape := shapes.Make(dt, aTrailing)
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
					A:       aTrailing,
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

				key := trailingKey{dt, b}
				res, ok := minTrailing[key]
				if !ok || scalarMed < res.scalarMin {
					res.scalarMin = scalarMed
				}
				if !ok || simdMed < res.simdMin {
					res.simdMin = simdMed
				}
				minTrailing[key] = res

				be.PutBuffer(inBuf)
				be.PutBuffer(outBuf)
			}
		}
	}

	fmt.Println("\n### 1. ReduceTrailing: shape [A, B] -> [A] (reducing inner dimension B)")
	fmt.Println("| DType | A | B | Scalar Median | SIMD Median | Ratio (SIMD/Scalar) | Faster |")
	fmt.Println("|:---|---:|---:|---:|---:|---:|:---|")

	trailingThresholds := make(map[dtypes.DType]int)
	for _, dt := range testDTypes {
		lastScalarB := -1
		for _, b := range bValuesTrailing {
			res := minTrailing[trailingKey{dt, b}]
			ratio := float64(res.simdMin) / float64(res.scalarMin)
			faster := "SIMD"
			if ratio > 1.15 {
				faster = "**SCALAR**"
				lastScalarB = b
			} else if ratio >= 0.85 {
				faster = "TIE"
			}

			fmt.Printf("| %-7s | %3d | %4d | %9s | %9s | %18.2f | %-10s |\n",
				dt, aTrailing, b, res.scalarMin, res.simdMin, ratio, faster)
		}
		if lastScalarB == -1 {
			trailingThresholds[dt] = ThresholdNone
		} else if lastScalarB == bValuesTrailing[len(bValuesTrailing)-1] {
			trailingThresholds[dt] = ThresholdAlwaysFallBack
		} else {
			trailingThresholds[dt] = lastScalarB
		}
	}

	// -------------------------------------------------------------------------
	// 2. REDUCE ALL: shape [N] -> [1]
	// -------------------------------------------------------------------------
	type allKey struct {
		dt dtypes.DType
		n  int
	}
	minAll := make(map[allKey]benchResult)
	nValuesAll := []int{2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024}

	for range repeats {
		for _, dt := range testDTypes {
			for _, n := range nValuesAll {
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

				key := allKey{dt, n}
				res, ok := minAll[key]
				if !ok || scalarMed < res.scalarMin {
					res.scalarMin = scalarMed
				}
				if !ok || simdMed < res.simdMin {
					res.simdMin = simdMed
				}
				minAll[key] = res

				be.PutBuffer(inBuf)
				be.PutBuffer(outBuf)
			}
		}
	}

	fmt.Println("\n### 2. ReduceAll: shape [N] -> [1] (reducing entire tensor)")
	fmt.Println("| DType | N | Scalar Median | SIMD Median | Ratio (SIMD/Scalar) | Faster |")
	fmt.Println("|:---|---:|---:|---:|---:|:---|")

	allThresholds := make(map[dtypes.DType]int)
	for _, dt := range testDTypes {
		lastScalarN := -1
		for _, n := range nValuesAll {
			res := minAll[allKey{dt, n}]
			ratio := float64(res.simdMin) / float64(res.scalarMin)
			faster := "SIMD"
			if ratio > 1.15 {
				faster = "**SCALAR**"
				lastScalarN = n
			} else if ratio >= 0.85 {
				faster = "TIE"
			}

			fmt.Printf("| %-7s | %4d | %9s | %9s | %18.2f | %-10s |\n",
				dt, n, res.scalarMin, res.simdMin, ratio, faster)
		}
		if lastScalarN == -1 {
			allThresholds[dt] = ThresholdNone
		} else if lastScalarN == nValuesAll[len(nValuesAll)-1] {
			allThresholds[dt] = ThresholdAlwaysFallBack
		} else {
			allThresholds[dt] = lastScalarN
		}
	}

	// -------------------------------------------------------------------------
	// 3. REDUCE LEADING: shape [A, B] -> [B] (outer axis A reduced)
	// -------------------------------------------------------------------------
	type leadingKey struct {
		dt   dtypes.DType
		a, b int
	}
	minLeading := make(map[leadingKey]benchResult)
	aValuesLeading := []int{2, 4, 8, 16, 64, 256, 1024}
	bValuesLeading := []int{2, 4, 8, 16, 32, 64, 256, 512, 1024}

	for range repeats {
		for _, dt := range testDTypes {
			for _, a := range aValuesLeading {
				for _, b := range bValuesLeading {
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

					key := leadingKey{dt, a, b}
					res, ok := minLeading[key]
					if !ok || scalarMed < res.scalarMin {
						res.scalarMin = scalarMed
					}
					if !ok || simdMed < res.simdMin {
						res.simdMin = simdMed
					}
					minLeading[key] = res

					be.PutBuffer(inBuf)
					be.PutBuffer(outBuf)
				}
			}
		}
	}

	fmt.Println("\n### 3. ReduceLeading: shape [A, B] -> [B] (reducing outer dimension A)")
	fmt.Println("| DType | A | B | Scalar Median | SIMD Median | Ratio (SIMD/Scalar) | Faster |")
	fmt.Println("|:---|---:|---:|---:|---:|---:|:---|")

	leadingThresholds := make(map[dtypes.DType]int)
	for _, dt := range testDTypes {
		lastScalarB := -1
		for _, b := range bValuesLeading {
			scalarWon := false
			for _, a := range aValuesLeading {
				res := minLeading[leadingKey{dt, a, b}]
				ratio := float64(res.simdMin) / float64(res.scalarMin)
				faster := "SIMD"
				if ratio > 1.15 {
					faster = "**SCALAR**"
					scalarWon = true
				} else if ratio >= 0.85 {
					faster = "TIE"
				}

				fmt.Printf("| %-7s | %4d | %4d | %9s | %9s | %18.2f | %-10s |\n",
					dt, a, b, res.scalarMin, res.simdMin, ratio, faster)
			}
			if scalarWon {
				lastScalarB = b
			}
		}
		if lastScalarB == -1 {
			leadingThresholds[dt] = ThresholdNone
		} else if lastScalarB == bValuesLeading[len(bValuesLeading)-1] {
			leadingThresholds[dt] = ThresholdAlwaysFallBack
		} else {
			leadingThresholds[dt] = lastScalarB
		}
	}

	// -------------------------------------------------------------------------
	// 4. GENERATED CONFIG STRUCT
	// -------------------------------------------------------------------------
	fmt.Println("\n// ==========================================================================================")
	fmt.Println("//                           RECOMMENDED reduceThresholdsConfig                               ")
	fmt.Println("// ==========================================================================================")
	fmt.Println("var recommendedReduceThresholds = reduceThresholdsConfig{")
	fmt.Println("\tTrailingMinB: map[dtypes.DType]int{")
	for _, dt := range testDTypes {
		keyStr := fmt.Sprintf("dtypes.%s:", dt)
		fmt.Printf("\t\t%-17s %s,\n", keyStr, formatThreshold(trailingThresholds[dt]))
	}
	fmt.Println("\t},")
	fmt.Println("\tLeadingMinB: map[dtypes.DType]int{")
	for _, dt := range testDTypes {
		keyStr := fmt.Sprintf("dtypes.%s:", dt)
		fmt.Printf("\t\t%-17s %s,\n", keyStr, formatThreshold(leadingThresholds[dt]))
	}
	fmt.Println("\t},")
	fmt.Println("\tAllMinN: map[dtypes.DType]int{")
	for _, dt := range testDTypes {
		keyStr := fmt.Sprintf("dtypes.%s:", dt)
		fmt.Printf("\t\t%-17s %s,\n", keyStr, formatThreshold(allThresholds[dt]))
	}
	fmt.Println("\t},")
	fmt.Println("}")
}

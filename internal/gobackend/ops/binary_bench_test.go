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
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx512"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

var (
	flagRepeatBinaryBench = flag.Int("repeat_binary_bench", 0, "Number of times to repeat binary benchmark, taking the minimum value and printing comparison table")
	flagBinarySIMD        = flag.Bool("binary_simd", true, "Enable SIMD for testing.B benchmarks (set to false to benchmark scalar/generic)")
)

func runGenericBinary(op compute.OpType, lhs, rhs, output *gobackend.Buffer, bcastCfg gobackend.BroadcastConfig) error {
	dt := lhs.RawShape.DType
	switch op {
	case compute.OpTypeAdd:
		switch dt {
		case dtypes.Float32:
			execAddNumericGeneric(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float64:
			execAddNumericGeneric(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int32:
			execAddNumericGeneric(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint32:
			execAddNumericGeneric(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int64:
			execAddNumericGeneric(lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint64:
			execAddNumericGeneric(lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int16:
			execAddNumericGeneric(lhs.Flat.([]int16), rhs.Flat.([]int16), output.Flat.([]int16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint16:
			execAddNumericGeneric(lhs.Flat.([]uint16), rhs.Flat.([]uint16), output.Flat.([]uint16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int8:
			execAddNumericGeneric(lhs.Flat.([]int8), rhs.Flat.([]int8), output.Flat.([]int8), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint8:
			execAddNumericGeneric(lhs.Flat.([]uint8), rhs.Flat.([]uint8), output.Flat.([]uint8), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.BFloat16:
			execAddNumericBFloat16(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float16:
			execAddNumericFloat16(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		default:
			return errors.Errorf("unsupported generic add dtype: %s", dt)
		}
	case compute.OpTypeSub:
		switch dt {
		case dtypes.Float32:
			execSubNumericGeneric(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float64:
			execSubNumericGeneric(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int32:
			execSubNumericGeneric(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint32:
			execSubNumericGeneric(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int64:
			execSubNumericGeneric(lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint64:
			execSubNumericGeneric(lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int16:
			execSubNumericGeneric(lhs.Flat.([]int16), rhs.Flat.([]int16), output.Flat.([]int16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint16:
			execSubNumericGeneric(lhs.Flat.([]uint16), rhs.Flat.([]uint16), output.Flat.([]uint16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int8:
			execSubNumericGeneric(lhs.Flat.([]int8), rhs.Flat.([]int8), output.Flat.([]int8), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint8:
			execSubNumericGeneric(lhs.Flat.([]uint8), rhs.Flat.([]uint8), output.Flat.([]uint8), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.BFloat16:
			execSubNumericBFloat16(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float16:
			execSubNumericFloat16(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		default:
			return errors.Errorf("unsupported generic sub dtype: %s", dt)
		}
	case compute.OpTypeMul:
		switch dt {
		case dtypes.Float32:
			execMulNumericGeneric(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float64:
			execMulNumericGeneric(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int32:
			execMulNumericGeneric(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint32:
			execMulNumericGeneric(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int64:
			execMulNumericGeneric(lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint64:
			execMulNumericGeneric(lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int16:
			execMulNumericGeneric(lhs.Flat.([]int16), rhs.Flat.([]int16), output.Flat.([]int16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint16:
			execMulNumericGeneric(lhs.Flat.([]uint16), rhs.Flat.([]uint16), output.Flat.([]uint16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Int8:
			execMulNumericGeneric(lhs.Flat.([]int8), rhs.Flat.([]int8), output.Flat.([]int8), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Uint8:
			execMulNumericGeneric(lhs.Flat.([]uint8), rhs.Flat.([]uint8), output.Flat.([]uint8), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.BFloat16:
			execMulNumericBFloat16(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float16:
			execMulNumericFloat16(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		default:
			return errors.Errorf("unsupported generic mul dtype: %s", dt)
		}
	case compute.OpTypeDiv:
		switch dt {
		case dtypes.Float32:
			execDivNumericGeneric(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float64:
			execDivNumericGeneric(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.BFloat16:
			execDivNumericBFloat16(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		case dtypes.Float16:
			execDivNumericFloat16(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), lhs.RawShape, rhs.RawShape, output.RawShape, bcastCfg)
		default:
			return errors.Errorf("unsupported generic div dtype: %s", dt)
		}
	default:
		return errors.Errorf("unsupported binary op: %s", op)
	}
	return nil
}

func runSIMDBinary(op compute.OpType, lhs, rhs, output *gobackend.Buffer, bcastCfg gobackend.BroadcastConfig) error {
	if tryBinaryTrailingArch(op, lhs, rhs, output, bcastCfg) {
		return nil
	}
	dt := lhs.RawShape.DType
	switch op {
	case compute.OpTypeAdd:
		switch dt {
		case dtypes.Float32:
			return dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
				simdAddVVFloat32, simdAddVSFloat32, func(c float32, r []float32, out []float32) { simdAddVSFloat32(r, c, out) },
				simdV2V1AddFloat32, simdV1V2AddFloat32)
		case dtypes.Float64:
			return dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
				simdAddVVFloat64, simdAddVSFloat64, func(c float64, r []float64, out []float64) { simdAddVSFloat64(r, c, out) },
				nil, nil)
		case dtypes.Int32:
			return dispatchBinarySIMD(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
				simdAddVVInt32, simdAddVSInt32, func(c int32, r []int32, out []int32) { simdAddVSInt32(r, c, out) },
				nil, nil)
		case dtypes.Uint32:
			return dispatchBinarySIMD(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
				simdAddVVUint32, simdAddVSUint32, func(c uint32, r []uint32, out []uint32) { simdAddVSUint32(r, c, out) },
				nil, nil)
		case dtypes.Int64:
			return dispatchBinarySIMD(lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), bcastCfg,
				simdAddVVInt64, simdAddVSInt64, func(c int64, r []int64, out []int64) { simdAddVSInt64(r, c, out) },
				nil, nil)
		case dtypes.Uint64:
			return dispatchBinarySIMD(lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), bcastCfg,
				simdAddVVUint64, simdAddVSUint64, func(c uint64, r []uint64, out []uint64) { simdAddVSUint64(r, c, out) },
				nil, nil)
		case dtypes.BFloat16:
			return dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
				simdAddVVBFloat16, simdAddVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) {
					simdAddVSBFloat16(r, c, out)
				}, nil, nil)
		case dtypes.Float16:
			return dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
				simdAddVVFloat16, simdAddVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdAddVSFloat16(r, c, out) },
				nil, nil)
		default:
			return errors.Errorf("unsupported SIMD add dtype: %s", dt)
		}
	case compute.OpTypeSub:
		switch dt {
		case dtypes.Float32:
			return dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
				simdSubVVFloat32, simdSubVSFloat32, simdSubSVFloat32,
				nil, nil)
		case dtypes.Float64:
			return dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
				simdSubVVFloat64, simdSubVSFloat64, simdSubSVFloat64,
				nil, nil)
		case dtypes.Int32:
			return dispatchBinarySIMD(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
				simdSubVVInt32, simdSubVSInt32, simdSubSVInt32,
				nil, nil)
		case dtypes.Uint32:
			return dispatchBinarySIMD(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
				simdSubVVUint32, simdSubVSUint32, simdSubSVUint32,
				nil, nil)
		case dtypes.Int64:
			return dispatchBinarySIMD(lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), bcastCfg,
				simdSubVVInt64, simdSubVSInt64, simdSubSVInt64,
				nil, nil)
		case dtypes.Uint64:
			return dispatchBinarySIMD(lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), bcastCfg,
				simdSubVVUint64, simdSubVSUint64, simdSubSVUint64,
				nil, nil)
		case dtypes.BFloat16:
			return dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
				simdSubVVBFloat16, simdSubVSBFloat16, simdSubSVBFloat16,
				nil, nil)
		case dtypes.Float16:
			return dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
				simdSubVVFloat16, simdSubVSFloat16, simdSubSVFloat16,
				nil, nil)
		default:
			return errors.Errorf("unsupported SIMD sub dtype: %s", dt)
		}
	case compute.OpTypeMul:
		switch dt {
		case dtypes.Float32:
			return dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
				simdMulVVFloat32, simdMulVSFloat32, func(c float32, r []float32, out []float32) { simdMulVSFloat32(r, c, out) },
				simdV2V1MulFloat32, simdV1V2MulFloat32)
		case dtypes.Float64:
			return dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
				simdMulVVFloat64, simdMulVSFloat64, func(c float64, r []float64, out []float64) { simdMulVSFloat64(r, c, out) },
				nil, nil)
		case dtypes.Int32:
			return dispatchBinarySIMD(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
				simdMulVVInt32, simdMulVSInt32, func(c int32, r []int32, out []int32) { simdMulVSInt32(r, c, out) },
				nil, nil)
		case dtypes.Uint32:
			return dispatchBinarySIMD(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
				simdMulVVUint32, simdMulVSUint32, func(c uint32, r []uint32, out []uint32) { simdMulVSUint32(r, c, out) },
				nil, nil)
		case dtypes.BFloat16:
			return dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
				simdMulVVBFloat16, simdMulVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) {
					simdMulVSBFloat16(r, c, out)
				}, nil, nil)
		case dtypes.Float16:
			return dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
				simdMulVVFloat16, simdMulVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdMulVSFloat16(r, c, out) },
				nil, nil)
		default:
			return errors.Errorf("unsupported SIMD mul dtype: %s", dt)
		}
	case compute.OpTypeDiv:
		switch dt {
		case dtypes.Float32:
			return dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
				simdDivVVFloat32, simdDivVSFloat32, simdDivSVFloat32,
				nil, nil)
		case dtypes.Float64:
			return dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
				simdDivVVFloat64, simdDivVSFloat64, simdDivSVFloat64,
				nil, nil)
		case dtypes.BFloat16:
			return dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
				simdDivVVBFloat16, simdDivVSBFloat16, simdDivSVBFloat16,
				nil, nil)
		case dtypes.Float16:
			return dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
				simdDivVVFloat16, simdDivVSFloat16, simdDivSVFloat16,
				nil, nil)
		default:
			return errors.Errorf("unsupported SIMD div dtype: %s", dt)
		}
	default:
		return errors.Errorf("unsupported binary op: %s", op)
	}
}

// TestBinaryBroadcastBenchmark runs a comparative benchmark between Generic (Scalar) and SIMD
// for BroadcastTrailingRHS and BroadcastTrailingLHS.
func TestBinaryBroadcastBenchmark(t *testing.T) {
	if *flagRepeatBinaryBench < 1 {
		t.Skip("Run with -repeat_binary_bench=<n> to benchmark binary broadcast trailing (n >= 1)")
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
		dtypes.Int64,
	}

	testOps := []compute.OpType{
		compute.OpTypeAdd,
		compute.OpTypeSub,
		compute.OpTypeMul,
		compute.OpTypeDiv,
	}

	bValues := []int{2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024}
	aConst := 100
	repeats := max(1, *flagRepeatBinaryBench)

	for _, pattern := range []gobackend.BroadcastPattern{gobackend.BroadcastTrailingRHS, gobackend.BroadcastTrailingLHS} {
		patternName := "BroadcastTrailingRHS ([A, B] op [A, 1] -> [A, B])"
		if pattern == gobackend.BroadcastTrailingLHS {
			patternName = "BroadcastTrailingLHS ([A, 1] op [A, B] -> [A, B])"
		}

		fmt.Println("\n==========================================================================================")
		fmt.Printf("           BINARY BENCHMARK: %s (MEDIANS)           \n", patternName)
		fmt.Println("==========================================================================================")
		fmt.Println("| Op | DType | A | B | Scalar Median | SIMD Median | Ratio (SIMD/Scalar) | Faster |")
		fmt.Println("|:---|:---|---:|---:|---:|---:|---:|:---|")

		for _, op := range testOps {
			for _, dt := range testDTypes {
				if (op == compute.OpTypeDiv || op == compute.OpTypeMul) && (dt == dtypes.Int32 || dt == dtypes.Int64) {
					if op == compute.OpTypeMul && dt == dtypes.Int32 {
						// Int32 Mul is supported
					} else {
						continue
					}
				}

				for _, b := range bValues {
					var lhsShape, rhsShape shapes.Shape
					if pattern == gobackend.BroadcastTrailingRHS {
						lhsShape = shapes.Make(dt, aConst, b)
						rhsShape = shapes.Make(dt, aConst, 1)
					} else {
						lhsShape = shapes.Make(dt, aConst, 1)
						rhsShape = shapes.Make(dt, aConst, b)
					}
					outShape := shapes.Make(dt, aConst, b)

					lhsBuf, err := be.GetBuffer(lhsShape)
					if err != nil {
						t.Fatalf("GetBuffer lhs failed: %+v", err)
					}
					rhsBuf, err := be.GetBuffer(rhsShape)
					if err != nil {
						t.Fatalf("GetBuffer rhs failed: %+v", err)
					}
					outBuf, err := be.GetBuffer(outShape)
					if err != nil {
						t.Fatalf("GetBuffer out failed: %+v", err)
					}

					// Ensure divisor is non-zero
					if op == compute.OpTypeDiv {
						if dt == dtypes.Float32 {
							if pattern == gobackend.BroadcastTrailingRHS {
								f := rhsBuf.Flat.([]float32)
								for i := range f {
									f[i] = 2.0
								}
							} else {
								f := rhsBuf.Flat.([]float32)
								for i := range f {
									f[i] = 2.0
								}
							}
						} else if dt == dtypes.Float64 {
							if pattern == gobackend.BroadcastTrailingRHS {
								f := rhsBuf.Flat.([]float64)
								for i := range f {
									f[i] = 2.0
								}
							} else {
								f := rhsBuf.Flat.([]float64)
								for i := range f {
									f[i] = 2.0
								}
							}
						}
					}

					cfg := gobackend.BroadcastConfig{
						Pattern: pattern,
						A:       aConst,
						B:       b,
					}

					var scalarMin, simdMin time.Duration
					for r := 0; r < repeats; r++ {
						sMed, err := measureMedianDuration(func() error {
							return runGenericBinary(op, lhsBuf, rhsBuf, outBuf, cfg)
						}, 10*time.Millisecond, 500)
						if err != nil {
							t.Fatalf("scalar measure failed: %+v", err)
						}

						simdMed, err := measureMedianDuration(func() error {
							return runSIMDBinary(op, lhsBuf, rhsBuf, outBuf, cfg)
						}, 10*time.Millisecond, 500)
						if err != nil {
							t.Fatalf("simd measure failed: %+v", err)
						}

						if r == 0 || sMed < scalarMin {
							scalarMin = sMed
						}
						if r == 0 || simdMed < simdMin {
							simdMin = simdMed
						}
					}

					be.PutBuffer(lhsBuf)
					be.PutBuffer(rhsBuf)
					be.PutBuffer(outBuf)

					ratio := float64(simdMin) / float64(scalarMin)
					faster := "SIMD"
					if ratio > 1.15 {
						faster = "**SCALAR**"
					} else if ratio >= 0.85 {
						faster = "TIE"
					}

					fmt.Printf("| %-3s | %-7s | %3d | %4d | %9s | %9s | %18.2f | %-10s |\n",
						op, dt, aConst, b, scalarMin, simdMin, ratio, faster)
				}
			}
		}
	}
}

func benchmarkBroadcastTrailing(b *testing.B, pattern gobackend.BroadcastPattern, op compute.OpType, dt dtypes.DType, A, B int) {
	b.Helper()
	backendGeneric, err := gobackend.New("")
	if err != nil {
		b.Fatalf("failed to create backend: %+v", err)
	}
	be := backendGeneric.(*gobackend.Backend)
	defer be.Finalize()

	var lhsShape, rhsShape shapes.Shape
	if pattern == gobackend.BroadcastTrailingRHS {
		lhsShape = shapes.Make(dt, A, B)
		rhsShape = shapes.Make(dt, A, 1)
	} else {
		lhsShape = shapes.Make(dt, A, 1)
		rhsShape = shapes.Make(dt, A, B)
	}
	outShape := shapes.Make(dt, A, B)

	lhsBuf, _ := be.GetBuffer(lhsShape)
	rhsBuf, _ := be.GetBuffer(rhsShape)
	outBuf, _ := be.GetBuffer(outShape)
	defer be.PutBuffer(lhsBuf)
	defer be.PutBuffer(rhsBuf)
	defer be.PutBuffer(outBuf)

	cfg := gobackend.BroadcastConfig{
		Pattern: pattern,
		A:       A,
		B:       B,
	}

	useSIMD := *flagBinarySIMD

	b.ResetTimer()
	if useSIMD {
		for i := 0; i < b.N; i++ {
			_ = runSIMDBinary(op, lhsBuf, rhsBuf, outBuf, cfg)
		}
	} else {
		for i := 0; i < b.N; i++ {
			_ = runGenericBinary(op, lhsBuf, rhsBuf, outBuf, cfg)
		}
	}
}

func BenchmarkBroadcastTrailingRHS(b *testing.B) {
	for _, bDim := range []int{4, 16, 64, 256, 1024} {
		b.Run(fmt.Sprintf("Add_Float32_B%d", bDim), func(b *testing.B) {
			benchmarkBroadcastTrailing(b, gobackend.BroadcastTrailingRHS, compute.OpTypeAdd, dtypes.Float32, 100, bDim)
		})
	}
}

func BenchmarkBroadcastTrailingLHS(b *testing.B) {
	for _, bDim := range []int{4, 16, 64, 256, 1024} {
		b.Run(fmt.Sprintf("Sub_Float32_B%d", bDim), func(b *testing.B) {
			benchmarkBroadcastTrailing(b, gobackend.BroadcastTrailingLHS, compute.OpTypeSub, dtypes.Float32, 100, bDim)
		})
	}
}

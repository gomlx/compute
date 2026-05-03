// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dot_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/defaultpkgs"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/highway"
	"github.com/gomlx/compute/internal/gobackend/dot/packgemm"
	"github.com/gomlx/compute/internal/must"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support"
	"github.com/gomlx/compute/support/testutil"
	"k8s.io/klog/v2"
)

var backend compute.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", compute.List())
	// Perform your setup logic here
	if os.Getenv(compute.ConfigEnvVar) == "" {
		must.M(os.Setenv(compute.ConfigEnvVar, "go"))
	} else {
		fmt.Printf("\t$%s=%q\n", compute.ConfigEnvVar, os.Getenv(compute.ConfigEnvVar))
	}
	backend = compute.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

func TestDotGeneral_LargeShapesAndCopy(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	// Test #1: batch axes are out-of-order.
	{
		dtype := dtypes.Float64
		sourceShape := shapes.Make(dtype, 2, 1, 3)
		contractingAxes := []int{1}
		batchAxes := []int{2, 0}
		batchSize, crossSize, contractingSize, crossDims := support.DotGeneralFindSizes(sourceShape, contractingAxes, batchAxes)
		if batchSize != 6 {
			t.Fatalf("Expected batchSize 6, got %d", batchSize)
		}
		if crossSize != 1 {
			t.Fatalf("Expected crossSize 1, got %d", crossSize)
		}
		if contractingSize != 1 {
			t.Fatalf("Expected contractingSize 1, got %d", contractingSize)
		}
		if len(crossDims) != 0 {
			t.Fatalf("Expected crossDims length 0, got %d", len(crossDims))
		}

		// Create the source buffer.
		sourceAny, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		source := sourceAny.(*gobackend.Buffer)
		sourceFlat := sourceFlatAny.([]float64)
		for i := range sourceFlat {
			sourceFlat[i] = float64(i + 1)
		}

		// Create a block shape.
		blockLog2Dim := 1 // block dim is 2^1 = 2.
		blockDim := 1 << blockLog2Dim
		be := backend.(*gobackend.Backend)
		outShape := dot.CreateBlockedShape(dtype, batchSize, crossSize, contractingSize, blockLog2Dim)
		// outShape = [6 1 1 2 2]
		if ok, diff := testutil.IsEqual(
			[]int{
				batchSize,
				(crossSize + blockDim - 1) / blockDim,
				(contractingSize + blockDim - 1) / blockDim,
				blockDim,
				blockDim,
			},
			outShape.Dimensions,
		); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		outBlocks, err := be.GetBufferForShape(outShape)
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		outBlocks.Zeros()
		tmpAny, tmpErr := dot.FlatToBlockDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		copyFlatToBlock := tmpAny.(func(
			source, blkOutput *gobackend.Buffer, contractingAxes, batchAxes []int,
			batchSize, crossSize, contractingSize, blkLog2Dim int))
		copyFlatToBlock(
			source,
			outBlocks,
			contractingAxes,
			batchAxes,
			batchSize,
			crossSize,
			contractingSize,
			blockLog2Dim,
		)

		outFlat := outBlocks.Flat.([]float64)
		// Notice the reversal (transposition) of the batch axes:
		want := []float64{
			1, 0, 0, 0,
			4, 0, 0, 0,

			2, 0, 0, 0,
			5, 0, 0, 0,

			3, 0, 0, 0,
			6, 0, 0, 0,
		}
		if ok, diff := testutil.IsEqual(want, outFlat); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	}
}

func TestDotGeneral_SmallNormalize(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	// Test #1: batch axes are out-of-order.
	{
		dtype := dtypes.Float64
		sourceShape := shapes.Make(dtype, 2, 1, 3)
		contractingAxes := []int{1}
		batchAxes := []int{2, 0}
		batchSize, crossSize, contractingSize, crossDims := support.DotGeneralFindSizes(sourceShape, contractingAxes, batchAxes)
		if batchSize != 6 {
			t.Fatalf("Expected batchSize 6, got %d", batchSize)
		}
		if crossSize != 1 {
			t.Fatalf("Expected crossSize 1, got %d", crossSize)
		}
		if contractingSize != 1 {
			t.Fatalf("Expected contractingSize 1, got %d", contractingSize)
		}
		if len(crossDims) != 0 {
			t.Fatalf("Expected crossDims length 0, got %d", len(crossDims))
		}

		// Create the source buffer.
		sourceIf, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		source := sourceIf.(*gobackend.Buffer)
		sourceFlat := sourceFlatAny.([]float64)
		for i := range sourceFlat {
			sourceFlat[i] = float64(i + 1)
		}
		tmpAny, tmpErr := dot.NormalizeShapeDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		normalizeFn := tmpAny.(func(backend *gobackend.Backend, source *gobackend.Buffer, info *dot.NormalizationInfo, batchSize, crossSize, contractingSize int) *gobackend.Buffer)
		info := dot.NormalizePrepare(source.RawShape, contractingAxes, batchAxes)
		output := normalizeFn(
			backend.(*gobackend.Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		if output == nil {
			t.Fatalf("Expected non-nil value")
		}
		if err := output.RawShape.Check(dtype, batchSize, crossSize, contractingSize); err != nil {
			t.Fatalf("Check failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual([]float64{1, 4, 2, 5, 3, 6}, output.Flat.([]float64)); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	}
}

func TestDotGeneral_ForcePaths(t *testing.T) {
	goBackend, ok := backend.(*gobackend.Backend)
	if !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	// Reset dotGeneralForceExecutionPath at exit to default (auto-select).
	defer func() {
		goBackend.DotGeneralForceExecutionPath = int(dot.AutoSelectPath)
	}()

	lhs := [][][]float32{{{1, 2, 3}}, {{4, 5, 6}}}
	rhs := [][][]float32{{{1, 1}, {1, 1}, {1, 1}}}
	want := [][]float32{{1, 4}, {2, 5}, {3, 6}}

	for _, execPath := range []dot.ExecutionPath{dot.NormalizedPath, dot.BlockedPath, dot.SmallMatMulPath, dot.PackgemmPath, dot.HighwayPath, dot.CheckPath} {
		if execPath == dot.PackgemmPath && (!goBackend.EnablePackgemm || !packgemm.HasDTypeSupport(dtypes.Float32, dtypes.Float32)) {
			continue
		}
		if execPath == dot.HighwayPath && (!goBackend.EnableHighway || !highway.HasDTypeSupport(dtypes.Float32, dtypes.Float32)) {
			continue
		}

		goBackend.DotGeneralForceExecutionPath = int(execPath)
		t.Run(execPath.String(), func(t *testing.T) {
			y1, err := testutil.Exec1(backend, []any{lhs, rhs}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.DotGeneral(params[0], []int{1}, []int{2, 0}, params[1], []int{0}, []int{1, 2}, compute.DotGeneralConfig{})
			})
			if err != nil {
				t.Fatalf("testutil.Exec1 failed: %v", err)
			}
			if ok, diff := testutil.IsEqual(want, y1); !ok {
				t.Fatalf("Unexpected result for path %s (-want +got):\n%s", execPath, diff)
			}
		})
	}
}

func TestIsMatMulOrder(t *testing.T) {
	testCases := []struct {
		name               string
		lhsShape           shapes.Shape
		rhsShape           shapes.Shape
		lhsContractingAxes []int
		rhsContractingAxes []int
		lhsBatchAxes       []int
		rhsBatchAxes       []int
		want               bool
	}{
		// Standard 2D matrix multiplication: [M, K] x [K, N]
		{"2D_matmul_standard", shapes.Make(dtypes.Float32, 3, 4), shapes.Make(dtypes.Float32, 4, 5), []int{1}, []int{0}, []int{}, []int{}, true},
		// Transposed LHS: [K, M] x [K, N] - not matmul order
		{"2D_transposed_lhs", shapes.Make(dtypes.Float32, 4, 3), shapes.Make(dtypes.Float32, 4, 5), []int{0}, []int{0}, []int{}, []int{}, false},
		// Transposed RHS: [M, K] x [N, K] - not matmul order
		{"2D_transposed_rhs", shapes.Make(dtypes.Float32, 3, 4), shapes.Make(dtypes.Float32, 5, 4), []int{1}, []int{1}, []int{}, []int{}, false},
		// Matrix x Vector: [M, K] x [K]
		{"matrix_vector", shapes.Make(dtypes.Float32, 3, 4), shapes.Make(dtypes.Float32, 4), []int{1}, []int{0}, []int{}, []int{}, true},
		// Batched matrix multiplication: [B, M, K] x [B, K, N]
		{"batched_matmul", shapes.Make(dtypes.Float32, 2, 3, 4), shapes.Make(dtypes.Float32, 2, 4, 5), []int{2}, []int{1}, []int{0}, []int{0}, true},
		// Multiple contracting axes - not supported by SmallMatMul
		{"multiple_contracting", shapes.Make(dtypes.Float32, 2, 3, 4), shapes.Make(dtypes.Float32, 3, 4, 5), []int{1, 2}, []int{0, 1}, []int{}, []int{}, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := dot.IsMatMulOrder(tc.lhsShape, tc.lhsContractingAxes, tc.lhsBatchAxes,
				tc.rhsShape, tc.rhsContractingAxes, tc.rhsBatchAxes)
			if ok, diff := testutil.IsEqual(tc.want, got); !ok {
				t.Errorf("Unexpected result (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDgUseSmallMatMul(t *testing.T) {
	t.Run("ThresholdBoundaries", func(t *testing.T) {
		testCases := []struct {
			name            string
			batchSize       int
			lhsCrossSize    int
			rhsCrossSize    int
			contractingSize int
			want            bool
		}{
			{"contractingSize_at_threshold", 1, 10, 10, 128, true},
			{"contractingSize_over_threshold", 1, 10, 10, 129, false},
			{"batchSize_at_threshold", 64, 10, 10, 32, true},
			{"batchSize_over_threshold", 65, 10, 10, 32, false},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				lhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.lhsCrossSize, tc.contractingSize)
				rhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.contractingSize, tc.rhsCrossSize)

				params := &dot.NodeData{
					LHSContractingAxes: []int{2},
					LHSBatchAxes:       []int{0},
					RHSContractingAxes: []int{1},
					RHSBatchAxes:       []int{0},
					BatchSize:          tc.batchSize,
					LHSCrossSize:       tc.lhsCrossSize,
					RHSCrossSize:       tc.rhsCrossSize,
					ContractingSize:    tc.contractingSize,
				}

				got := dot.UseSmallMatMul(dtypes.Float32, lhsShape, rhsShape, params)
				if got != tc.want {
					t.Errorf("dgCanUseSmallMatMul with batch=%d, M=%d, N=%d, K=%d: got %v, want %v",
						tc.batchSize, tc.lhsCrossSize, tc.rhsCrossSize, tc.contractingSize, got, tc.want)
				}
			})
		}
	})
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend/highway"
	"github.com/gomlx/compute/internal/gobackend/packgemm"
	"github.com/gomlx/compute/internal/testutil"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

func TestDotGeneral_LargeShapesAndCopy(t *testing.T) {
	if _, ok := backend.(*Backend); !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
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
		source := sourceAny.(*Buffer)
		sourceFlat := sourceFlatAny.([]float64)
		for i := range sourceFlat {
			sourceFlat[i] = float64(i + 1)
		}

		// Create a block shape.
		blockLog2Dim := 1 // block dim is 2^1 = 2.
		blockDim := 1 << blockLog2Dim
		be := backend.(*Backend)
		outShape := dgCreateBlockedShape(dtype, batchSize, crossSize, contractingSize, blockLog2Dim)
		// outShape = [6 1 1 2 2]
		fmt.Printf("\toutShape=%s, size=%d\n", outShape, outShape.Size())
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
		outBlocks, err := be.getBuffer(dtype, outShape.Size())
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		outBlocks.shape = outShape
		outBlocks.Zeros()
		tmpAny, tmpErr := dotGeneralFlatToBlockDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		copyFlatToBlock := tmpAny.(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
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

		outFlat := outBlocks.flat.([]float64)
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

	{ // Test #2
		dtype := dtypes.Float32
		sourceShape := shapes.Make(dtype, 2, 3, 4, 5)
		contractingAxes := []int{1, 2}
		batchAxes := []int{0}
		batchSize, crossSize, contractingSize, crossDims := support.DotGeneralFindSizes(sourceShape, contractingAxes, batchAxes)
		if ok, diff := testutil.IsEqual(2, batchSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual(5, crossSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual(12, contractingSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual([]int{5}, crossDims); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}

		// Create the source buffer.
		sourceAny, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		source := sourceAny.(*Buffer)
		sourceFlat := sourceFlatAny.([]float32)
		for i := range sourceFlat {
			sourceFlat[i] = float32(i + 1)
		}

		// Create a block shape.
		blockLog2Dim := 2 // block dim is 2^2 = 4.
		blockDim := 1 << blockLog2Dim
		be := backend.(*Backend)
		outShape := dgCreateBlockedShape(dtype, batchSize, crossSize, contractingSize, blockLog2Dim)
		// outShape = [2 2 3 4 4]
		fmt.Printf("\toutShape=%s, size=%d\n", outShape, outShape.Size())
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
		outBlocks, err := be.getBuffer(dtype, outShape.Size())
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		outBlocks.shape = outShape
		outBlocks.Zeros()
		tmpAny, tmpErr := dotGeneralFlatToBlockDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		copyFlatToBlock := tmpAny.(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
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

		outFlat := outBlocks.flat.([]float32)
		want := []float32{
			1, 6, 11, 16, // Row 0 of block 0: sourceIdx are {0, 0, [0-3], 0}
			2, 7, 12, 17, // Row 1 of block 0: sourceIdx are {0, 0, [0-3], 1}
			3, 8, 13, 18, 4, 9, 14, 19, // Rows 2 and 3 of block 0

			// Block 1: sourceIdx are {0, 1, [0-3], [0-3]}
			21, 26, 31, 36, 22, 27, 32, 37, 23, 28, 33, 38, 24, 29, 34, 39,

			// Block 2: sourceIdx are {0, 2, [0-3], [0-3]}
			41, 46, 51, 56, 42, 47, 52, 57, 43, 48, 53, 58, 44, 49, 54, 59,

			// Block 4: sourceIdx for row 0 are {0, 0, [0-3], 4}, and the rest is padding.
			5, 10, 15, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

			// ...
			25, 30, 35, 40, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 50, 55, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 66, 71, 76, 62, 67, 72, 77, 63, 68, 73, 78, 64, 69, 74, 79, 81,
			86, 91, 96, 82, 87, 92, 97, 83, 88, 93, 98, 84, 89, 94, 99, 101, 106, 111, 116, 102, 107, 112, 117, 103, 108, 113, 118, 104, 109, 114, 119, 65, 70, 75, 80,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 90, 95, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 110, 115, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		}
		if ok, diff := testutil.IsEqual(want, outFlat); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	}
}

func TestDotGeneral_SmallNormalize(t *testing.T) {
	if _, ok := backend.(*Backend); !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
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
		source := sourceIf.(*Buffer)
		sourceFlat := sourceFlatAny.([]float64)
		for i := range sourceFlat {
			sourceFlat[i] = float64(i + 1)
		}
		tmpAny, tmpErr := dotGeneralNormalizeShapeDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		normalizeFn := tmpAny.(func(backend *Backend, source *Buffer, info *dgNormalizationInfo, batchSize, crossSize, contractingSize int) *Buffer)
		info := dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output := normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		if output == nil {
			t.Fatalf("Expected non-nil value")
		}
		if err := output.shape.Check(dtype, batchSize, crossSize, contractingSize); err != nil {
			t.Fatalf("Check failed: %+v", err)
		}
		if ok, diff := testutil.IsEqual([]float64{1, 4, 2, 5, 3, 6}, output.flat.([]float64)); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	}

	{ // Test #2: cross/contracting axes are inverted.
		dtype := dtypes.Float32
		sourceShape := shapes.Make(dtype, 2, 3, 4, 5)
		contractingAxes := []int{1, 2}
		batchAxes := []int{0}
		batchSize, crossSize, contractingSize, crossDims := support.DotGeneralFindSizes(sourceShape, contractingAxes, batchAxes)
		if ok, diff := testutil.IsEqual(2, batchSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual(5, crossSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual(12, contractingSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual([]int{5}, crossDims); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}

		// Create the source buffer.
		sourceIf, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		source := sourceIf.(*Buffer)
		sourceFlat := sourceFlatAny.([]float32)
		for i := range sourceFlat {
			sourceFlat[i] = float32(i + 1)
		}
		tmpAny, tmpErr := dotGeneralNormalizeShapeDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		normalizeFn := tmpAny.(func(backend *Backend, source *Buffer, info *dgNormalizationInfo, batchSize, crossSize, contractingSize int) *Buffer)
		info := dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output := normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		if output == nil {
			t.Fatalf("Expected non-nil value")
		}
		if err := output.shape.Check(dtype, batchSize, crossSize, contractingSize); err != nil {
			t.Fatalf("Check failed: %+v", err)
		}

		want := []float32{
			// Batch example 1:
			1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56,
			2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57,
			3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58,
			4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59,
			5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,

			// Batch example 2:
			61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116,
			62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117,
			63, 68, 73, 78, 83, 88, 93, 98, 103, 108, 113, 118,
			64, 69, 74, 79, 84, 89, 94, 99, 104, 109, 114, 119,
			65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
		}
		if ok, diff := testutil.IsEqual(want, output.flat.([]float32)); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
	}

	{ // Test #3: order preserved. There should be no transposition, and the output should be nil.
		dtype := dtypes.Float64
		sourceShape := shapes.Make(dtype, 2, 3, 4, 5)
		contractingAxes := []int{2, 3}
		batchAxes := []int{0}
		batchSize, crossSize, contractingSize, _ := support.DotGeneralFindSizes(sourceShape, contractingAxes, batchAxes)
		if ok, diff := testutil.IsEqual(2, batchSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual(3, crossSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		if ok, diff := testutil.IsEqual(20, contractingSize); !ok {
			t.Fatalf("Unexpected result (-want +got):\n%s", diff)
		}
		sourceIf, _, err := backend.NewSharedBuffer(0, sourceShape)
		if err != nil {
			t.Fatalf("Failed: %+v", err)
		}
		source := sourceIf.(*Buffer)
		tmpAny, tmpErr := dotGeneralNormalizeShapeDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		normalizeFn := tmpAny.(func(backend *Backend, source *Buffer, info *dgNormalizationInfo, batchSize, crossSize, contractingSize int) *Buffer)
		info := dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output := normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		if output != nil {
			t.Fatalf("Expected nil value, got %+v", output)
		}

		// If we invert the contracting axes, we need the transposition, and normalizeFn must handle it.
		contractingAxes = []int{3, 2}
		info = dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output = normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		if output == nil {
			t.Fatalf("Expected non-nil value")
		}
		if err := output.shape.Check(dtype, batchSize, crossSize, contractingSize); err != nil {
			t.Fatalf("Check failed: %+v", err)
		}
	}
}

func TestDotGeneral_Shape(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	builder := backend.Builder("DotGeneral Test").(*Builder)
	mainFn := builder.Main().(*Function)
	lhs, err := mainFn.Parameter("lhs", S(F32, 2, 3, 4, 5), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}
	rhs, err := mainFn.Parameter("rhs", S(F32, 5, 1, 2, 3), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}
	gotOp, err := mainFn.DotGeneral(
		lhs, []int{1}, []int{3, 0},
		rhs, []int{3}, []int{0, 2},
		compute.DotGeneralConfig{},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}
	got := gotOp.(*Node)
	// Batch dims: 5 , 2
	// Contracting dims: 3
	// Cross dims: 4 (lhs) and 1 (rhs)
	fmt.Printf("\tdotgeneral.shape=%s\n", got.shape)
	if err := got.shape.Check(F32, 5, 2, 4, 1); err != nil {
		t.Errorf("Unexpected error: %+v", err)
	}
}

func TestDotGeneral_Exec(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		fmt.Printf("Skipping %s, it is meant only for the Go backend, instead backend is ", backend.Name())
		t.SkipNow()
		return
	}

	// Reset dotGeneralForceExecutionPath at exit to default (auto-select).
	defer func() {
		goBackend.dotGeneralForceExecutionPath = autoSelectPath
	}()

	for _, execPath := range []dotGeneralExecutionPath{normalizedPath, blockedPath, smallMatMulPath, packgemmPath, highwayPath, checkPath} {
		if execPath == packgemmPath && (!goBackend.enablePackgemm || !packgemm.HasDTypeSupport(dtypes.Float32, dtypes.Float32)) {
			continue
		}
		if execPath == highwayPath && (!goBackend.enableHighway || !highway.HasDTypeSupport(dtypes.Float32, dtypes.Float32)) {
			continue
		}

		// Force a specific execution path: so we exercise the corresponding algorithm irrespective of the actual size:
		// it may not be efficient for the size, but it should be correct in all sizes.
		goBackend.dotGeneralForceExecutionPath = execPath
		t.Run(execPath.String(), func(t *testing.T) {
			t.Run("Float32", func(t *testing.T) {
				// Larger example, with multiple axes.
				y0, _ := testutil.Exec1(backend, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
					// We construct the input constants directly inside the compute.Function since we don't have
					// nested slice generators handy.
					lhs, _ := f.Constant(xslices.Iota(float32(1), 2*3*1*5), 2, 3, 1, 5)
					rhs, _ := f.Constant(xslices.Iota(float32(1), 5*3*2*4), 5, 3, 2, 4)
					return f.DotGeneral(lhs, []int{1}, []int{3, 0}, rhs, []int{1}, []int{0, 2}, compute.DotGeneralConfig{})
				})
				want := [][][][]float32{
					{
						{{242, 260, 278, 296}},
						{{899, 962, 1025, 1088}},
					}, {
						{{773, 794, 815, 836}},
						{{2522, 2588, 2654, 2720}},
					}, {
						{{1448, 1472, 1496, 1520}},
						{{4289, 4358, 4427, 4496}},
					}, {
						{{2267, 2294, 2321, 2348}},
						{{6200, 6272, 6344, 6416}},
					}, {
						{{3230, 3260, 3290, 3320}},
						{{8255, 8330, 8405, 8480}},
					}}
				if ok, diff := testutil.IsEqual(want, y0); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}
			})

			// Axis transposition example:
			t.Run("AxisTransposition", func(t *testing.T) {
				lhs := [][][]float32{{{1, 2, 3}}, {{4, 5, 6}}}
				rhs := [][][]float32{{{1, 1}, {1, 1}, {1, 1}}}
				y1, err := testutil.Exec1(backend, []any{lhs, rhs}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.DotGeneral(params[0], []int{1}, []int{2, 0}, params[1], []int{0}, []int{1, 2}, compute.DotGeneralConfig{})
				})
				if err != nil {
					t.Fatalf("testutil.Exec1 failed: %v", err)
				}
				want1 := [][]float32{{1, 4}, {2, 5}, {3, 6}}
				if ok, diff := testutil.IsEqual(want1, y1); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}
			})

			// A very large example: expected value computed using XLA.
			t.Run("VeryLarge", func(t *testing.T) {
				y3, err := testutil.Exec1(backend, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) {
					lhsShape := shapes.Make(dtypes.Float64, 16, 13, 384)
					lhsFlat := make([]float64, lhsShape.Size())
					for i := range lhsFlat {
						lhsFlat[i] = (float64(i) + 1.0) * 1e-5
					}
					lhs, _ := f.Constant(lhsFlat, 16, 13, 384)
					rhsShape := shapes.Make(dtypes.Float64, 384, 1536)
					rhsFlat := make([]float64, rhsShape.Size())
					for i := range rhsFlat {
						rhsFlat[i] = 1.0
					}
					rhs, _ := f.Constant(rhsFlat, 384, 1536)
					out, _ := f.DotGeneral(
						lhs, []int{2}, nil,
						rhs, []int{0}, nil, compute.DotGeneralConfig{})
					return f.Slice(out, []int{0, 0, 0}, []int{1, 1, 1}, []int{1, 1, 1})
				})
				if err != nil {
					t.Fatalf("testutil.Exec1 failed: %v", err)
				}
				if ok, diff := testutil.IsInDelta(y3.([][][]float64)[0][0][0], 0.7392, 1e-4); !ok {
					t.Fatalf("Result not within delta 1e-4:\n%s", diff)
				}
			})

			// BFloat16 examples.
			t.Run("BFloat16-with-f32-acc", func(t *testing.T) {
				// The defautl accumulator dtype for half-precision (BFloat16 and Float16) is Float32.
				bf16 := bfloat16.FromFloat32
				y2, err := testutil.Exec1(backend, []any{
					[][]bfloat16.BFloat16{{bf16(1), bf16(2), bf16(3)}},
					[][]bfloat16.BFloat16{{bf16(10)}, {bf16(11)}, {bf16(12)}},
				}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{AccumulatorDType: dtypes.Float32})
				})
				if err != nil {
					t.Fatalf("%s failed with an error: %+v", t.Name(), err)
				}
				if ok, diff := testutil.IsEqual(float32(10+22+36), y2.([][]bfloat16.BFloat16)[0][0].Float32()); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}
			})
			t.Run("BFloat16-no-acc-dtype", func(t *testing.T) {
				bf16 := bfloat16.FromFloat32
				y2, err := testutil.Exec1(backend, []any{
					[][]bfloat16.BFloat16{{bf16(1), bf16(2), bf16(3)}},
					[][]bfloat16.BFloat16{{bf16(10)}, {bf16(11)}, {bf16(12)}},
				}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{AccumulatorDType: dtypes.BFloat16})
				})
				if err != nil {
					t.Fatalf("%s failed with an error: %+v", t.Name(), err)
				}
				if ok, diff := testutil.IsEqual(float32(10+22+36), y2.([][]bfloat16.BFloat16)[0][0].Float32()); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}
			})

			// Float16 example.
			t.Run("Float16", func(t *testing.T) {
				f16 := float16.FromFloat32
				y2, err := testutil.Exec1(backend, []any{
					[][]float16.Float16{{f16(1), f16(2), f16(3)}},
					[][]float16.Float16{{f16(10)}, {f16(11)}, {f16(12)}},
				}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{})
				})
				if err != nil {
					t.Fatalf("testutil.Exec1 failed: %v", err)
				}
				if ok, diff := testutil.IsEqual(float32(10+22+36), y2.([][]float16.Float16)[0][0].Float32()); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}
			})

			// Do not run the larger tests if running -test.short: they will break Github
			// tests:
			if testing.Short() {
				fmt.Printf("\tSkipping larger tests for %s in -short mode\n", execPath)
				return
			}

			// From DotGeneral parameters taken from LLM models that not working during development:
			t.Run("LLM_1-parallel-requests", func(t *testing.T) {
				lhs, err := tensors.Load("dotgeneral_test_lhs.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				rhs, err := tensors.Load("dotgeneral_test_rhs.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				want, err := tensors.Load("dotgeneral_test_out.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}

				builder := backend.Builder("LLM_1-parallel-requests")
				mainFn := builder.Main()
				p0, _ := mainFn.Parameter("lhs", lhs.Shape(), nil)
				p1, _ := mainFn.Parameter("rhs", rhs.Shape(), nil)
				out, _ := mainFn.DotGeneral(p0, []int{2}, []int{0}, p1, []int{2}, []int{0}, compute.DotGeneralConfig{})
				mainFn.Return([]compute.Value{out}, nil)
				exec, err := builder.Compile()
				if err != nil {
					t.Fatalf("Compile failed: %v", err)
				}

				bufLhs, _ := testutil.ToBuffer(backend, lhs.Value())
				bufRhs, _ := testutil.ToBuffer(backend, rhs.Value())
				bufGot, err := exec.Execute([]compute.Buffer{bufLhs, bufRhs}, nil, 0)
				if err != nil {
					t.Fatalf("unexpected error: %+v", err)
				}
				gotRaw, _ := testutil.FromBuffer(backend, bufGot[0])

				if ok, diff := testutil.IsInDelta(want.Value(), gotRaw, 1e-4); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}

				// Run 8 workers in parallel to see if concurrency is a problem:
				const numConcurrent = 16
				errChan := make(chan error, numConcurrent)
				for runnerIdx := range numConcurrent {
					go func(_ int) {
						var err error
						defer func() {
							errChan <- err
						}()
						const numRepeats = 1000
						for range numRepeats {
							bufGotC, errC := exec.Execute([]compute.Buffer{bufLhs, bufRhs}, nil, 0)
							if errC != nil {
								err = errC
								return
							}
							gotRawC, _ := testutil.FromBuffer(backend, bufGotC[0])
							if ok, _ := testutil.IsInDelta(want.Value(), gotRawC, 1e-3); !ok {
								err = errors.Errorf("got result diff from want")
								return
							}
						}
					}(runnerIdx)
				}
				var firstError error
				for range numConcurrent {
					err := <-errChan
					if err != nil {
						if firstError == nil {
							firstError = err
						} else {
							klog.Errorf("Error while running in parallel: %v", err)
						}
					}
				}
				if firstError != nil {
					t.Fatalf("Error while running in parallel: %+v", firstError)
				}
			})

			t.Run("LLM_2", func(t *testing.T) {
				lhs, err := tensors.Load("dotgeneral_test_lhs_2.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				rhs, err := tensors.Load("dotgeneral_test_rhs_2.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				want, err := tensors.Load("dotgeneral_test_out_2.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				fmt.Printf("\tlhs=%s, rhs=%s\n", lhs.Shape(), rhs.Shape())
				gotRaw, err := testutil.Exec1(backend, []any{lhs.Value(), rhs.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.DotGeneral(params[0], []int{2}, []int{0}, params[1], []int{2}, []int{0}, compute.DotGeneralConfig{})
				})
				if err != nil {
					t.Fatalf("testutil.Exec1 failed: %v", err)
				}
				fmt.Printf("\twant=%s\n", want.Shape())
				if ok, diff := testutil.IsInDelta(want.Value(), gotRaw, 1e-3); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}
			})

			t.Run("LLM_2_bfloat16", func(t *testing.T) {
				lhs, err := tensors.Load("dotgeneral_test_lhs_2.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				rhs, err := tensors.Load("dotgeneral_test_rhs_2.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				want, err := tensors.Load("dotgeneral_test_out_2.bin")
				if err != nil {
					t.Fatalf("Failed: %+v", err)
				}
				fmt.Printf("\tlhs=%s, rhs=%s\n", lhs.Shape(), rhs.Shape())
				gotRaw, err := testutil.Exec1(backend, []any{lhs.Value(), rhs.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					l, _ := f.ConvertDType(params[0], dtypes.BFloat16)
					r, _ := f.ConvertDType(params[1], dtypes.BFloat16)
					output, _ := f.DotGeneral(l, []int{2}, []int{0}, r, []int{2}, []int{0}, compute.DotGeneralConfig{})
					return f.ConvertDType(output, dtypes.F32)
				})
				if err != nil {
					t.Fatalf("testutil.Exec1 failed: %v", err)
				}
				fmt.Printf("\t- want=%s\n", want.Shape())
				// Much larger delta, since BFloat16 loses precision.
				if ok, diff := testutil.IsInDelta(want.Value(), gotRaw, 1e-1); !ok {
					t.Fatalf("Unexpected result (-want +got):\n%s", diff)
				}
			})
		})
	}
}

func TestDotGeneral_ConfigDTypes(t *testing.T) {
	// Setup simplego backend
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	// Define common input shapes and values
	lhsData := float16.FromFloat32s(1, 2, 3, 4, 5, 6)
	rhsData := float16.FromFloat32s(7, 8, 9, 10, 11, 12)
	lhsTensor := tensors.FromFlatDataAndDimensions(lhsData, 2, 3)
	rhsTensor := tensors.FromFlatDataAndDimensions(rhsData, 3, 2)

	t.Run("AccumulatorDType", func(t *testing.T) {
		// Compile and execute
		result, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			// Define config with AccumulatorDType = Float32
			return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{AccumulatorDType: dtypes.Float32})
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}

		// Expected value is calculated with Float32 precision
		want := float16.FromFloat32s(1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12)
		gotFlat := testutil.FlattenSlice(result)
		if ok, diff := testutil.IsInDelta(want, gotFlat, 1e-2); !ok {
			t.Fatalf("Result not within delta 1e-2:\n%s", diff)
		}
	})

	t.Run("OutputDType", func(t *testing.T) {
		// Compile and execute
		result, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			// Define config with OutputDType = Float32
			return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{OutputDType: dtypes.Float32})
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}

		// Recompute expected values using Float16 for intermediate results (default behavior without AccumulatorDType)
		// and then convert to Float32 at the end.
		want := []float32{1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12, 4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12}
		gotFlat := testutil.FlattenSlice(result)
		if ok, diff := testutil.IsInDelta(want, gotFlat, 1e-2); !ok {
			t.Fatalf("Result not within delta 1e-2: (-want +got):\n%s", diff)
		}
	})

	t.Run("AccumulatorAndOutputDType", func(t *testing.T) {
		// Compile and execute
		result, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			// Define config with AccumulatorDType = Float32 and OutputDType = BFloat16
			return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{AccumulatorDType: dtypes.Float32, OutputDType: dtypes.BFloat16})
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		// Expected value from Float32 computation, then converted to BFloat16
		want := bfloat16.FromFloat32s(1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12)
		gotFlat := testutil.FlattenSlice(result)
		if ok, diff := testutil.IsInDelta(want, gotFlat, 1e-2); !ok {
			t.Fatalf("Result not within delta 1e-2:\n%s", diff)
		}
	})
}

func TestDotGeneral_Dot(t *testing.T) {
	y0, err := testutil.Exec1(backend, []any{[]float32{1, 2, 3}, []float32{10, 11, 12}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.DotGeneral(params[0], []int{0}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{})
	})
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if ok, diff := testutil.IsEqual(float32(1*10+2*11+3*12), y0); !ok {
		t.Errorf("Unexpected result (-want +got):\n%s", diff)
	}

	y1, err := testutil.Exec1(backend, []any{[][]float32{{1, 2, 3}, {2, 4, 6}}, []float32{10, 11, 12}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{})
	})
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if ok, diff := testutil.IsEqual([]float32{1*10 + 2*11 + 3*12, 2*10 + 4*11 + 6*12}, y1); !ok {
		t.Errorf("Unexpected result (-want +got):\n%s", diff)
	}

	y2, err := testutil.Exec1(backend, []any{[][]float32{{1, 2, 3}, {2, 4, 6}}, [][]float32{{10}, {11}, {12}}}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{})
	})
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if ok, diff := testutil.IsEqual([][]float32{{1*10 + 2*11 + 3*12}, {2*10 + 4*11 + 6*12}}, y2); !ok {
		t.Errorf("Unexpected result (-want +got):\n%s", diff)
	}
}

// TestBlockForDotGeneral_Deduplication tests that the same weight matrix
// is only blocked once when used in multiple DotGeneral operations.
func TestBlockForDotGeneral_Deduplication(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	builder := goBackend.Builder("TestDeduplication").(*Builder)
	mainFn := builder.Main().(*Function)

	// Create a parameter node (simulating weights)
	K, N := 128, 256
	weightsShape := shapes.Make(dtypes.Float32, K, N) // [K, N]
	weights, err := mainFn.Parameter("weights", weightsShape, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}
	weightsNode := weights.(*Node)

	// Get blocked input twice - should return the same node due to deduplication
	// Using blockForDotGeneral with explicit parameters for a 2D weight matrix
	blocked1 := mainFn.blockForDotGeneral(weightsNode, []int{0}, []int{}, 1, N, K)
	blocked2 := mainFn.blockForDotGeneral(weightsNode, []int{0}, []int{}, 1, N, K)

	// Should be the exact same node (pointer equality)
	if blocked1 != blocked2 {
		t.Fatalf("Deduplication should return the same blocked node: %s", "Deduplication should return the same blocked node")
	}

	// Verify the blocked shape is correct
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtypes.Float32]
	expectedCrossBlocks := (N + blockDim - 1) / blockDim
	expectedContractBlocks := (K + blockDim - 1) / blockDim
	if ok, diff := testutil.IsEqual([]int{1, expectedCrossBlocks, expectedContractBlocks, blockDim, blockDim},
		blocked1.shape.Dimensions); !ok {
		t.Fatalf("Unexpected blocked shape dimensions:\n%s", diff)
	}

	builder.Finalize()
}

// TestBlockForDotGeneral_Execution tests that the BlockForDotGeneral operation
// correctly converts a flat tensor to blocked format.
func TestBlockForDotGeneral_Execution(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	// Use a small block size for testing
	// Create a simple 2D tensor [4, 4] with known values
	K, N := 4, 4
	dtype := dtypes.Float32

	// Create source buffer
	sourceShape := shapes.Make(dtype, K, N)
	sourceAny, sourceFlatAny, err := goBackend.NewSharedBuffer(0, sourceShape)
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}
	source := sourceAny.(*Buffer)
	sourceFlat := sourceFlatAny.([]float32)

	// Fill with sequential values: 1, 2, 3, ..., 16
	for i := range sourceFlat {
		sourceFlat[i] = float32(i + 1)
	}

	// Create block data (simulating what blockRHSForDotGeneral would create)
	blockLog2Dim := 2 // Block dim = 4
	blockDim := 1 << blockLog2Dim
	blockedShape := dgCreateBlockedShape(dtype, 1, N, K, blockLog2Dim)

	data := &blockForDotGeneralData{
		blockLog2Dim:    blockLog2Dim,
		blockedShape:    blockedShape,
		batchSize:       1,
		crossSize:       N,
		contractingSize: K,
		contractingAxes: []int{0},
		batchAxes:       []int{},
	}

	// Create a mock node
	node := &Node{
		shape: blockedShape,
		data:  data,
	}

	// Execute the blocking operation
	output, err := execBlockForDotGeneral(goBackend, node, []*Buffer{source}, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}

	// Verify output shape
	if ok, diff := testutil.IsEqual(blockedShape, output.shape); !ok {
		t.Errorf("Unexpected result (-want +got):\n%s", diff)
	}

	// Verify output has correct size
	expectedSize := 1 * 1 * 1 * blockDim * blockDim // [1, 1, 1, 4, 4]
	if len(output.flat.([]float32)) != expectedSize {
		t.Fatalf("Unexpected output size: got %d, wanted %d", len(output.flat.([]float32)), expectedSize)
	}

	// The blocked output should preserve all the values (just reorganized)
	outputFlat := output.flat.([]float32)
	inputSum := float32(0)
	for _, v := range sourceFlat {
		inputSum += v
	}
	outputSum := float32(0)
	for _, v := range outputFlat {
		outputSum += v
	}
	if inputSum != outputSum {
		t.Errorf("Sum of values should be preserved after blocking: want %f, got %f", inputSum, outputSum)
	}
}

// TestDotGeneral_PreBlockedCorrectness tests that DotGeneral with pre-blocked
// weights produces the same results as without pre-blocking.
func TestDotGeneral_PreBlockedCorrectness(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	// Test with matrices large enough to trigger pre-blocking
	// but small enough to run quickly
	M, K, N := 32, 128, 64

	// Create input tensors
	lhsData := make([]float32, M*K)
	rhsData := make([]float32, K*N)
	for i := range lhsData {
		lhsData[i] = float32(i%100) * 0.01
	}
	for i := range rhsData {
		rhsData[i] = float32(i%100) * 0.01
	}

	lhs := tensors.FromFlatDataAndDimensions(lhsData, M, K)
	rhs := tensors.FromFlatDataAndDimensions(rhsData, K, N)

	// First, compute with normalized path (no pre-blocking)
	goBackend.dotGeneralForceExecutionPath = normalizedPath
	wantRaw, err := testutil.Exec1(goBackend, []any{lhs.Value(), rhs.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{})
	})
	if err != nil {
		t.Fatalf("testutil.Exec1 failed: %v", err)
	}

	// Now compute with blocked path (which may use pre-blocking for constant RHS)
	goBackend.dotGeneralForceExecutionPath = blockedPath
	gotRaw, err := testutil.Exec1(goBackend, []any{lhs.Value(), rhs.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.DotGeneral(params[0], []int{1}, nil, params[1], []int{0}, nil, compute.DotGeneralConfig{})
	})
	if err != nil {
		t.Fatalf("testutil.Exec1 failed: %v", err)
	}

	// Reset to default (auto-select)
	goBackend.dotGeneralForceExecutionPath = autoSelectPath

	// Compare results
	wantFlat := testutil.FlattenSlice(wantRaw).([]float32)
	gotFlat := testutil.FlattenSlice(gotRaw).([]float32)
	if ok, diff := testutil.IsInDelta(wantFlat, gotFlat, 1e-4); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
}

// TestBlockForDotGeneralData_Equal tests the Equal method for deduplication.
func TestBlockForDotGeneralData_Equal(t *testing.T) {
	base := &blockForDotGeneralData{
		blockLog2Dim:    5,
		blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
		batchSize:       1,
		crossSize:       128,
		contractingSize: 128,
		contractingAxes: []int{0},
		batchAxes:       []int{},
	}

	tests := []struct {
		name  string
		other *blockForDotGeneralData
		want  bool
	}{
		{
			name: "Identical",
			other: &blockForDotGeneralData{
				blockLog2Dim:    5,
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{0},
				batchAxes:       []int{},
			},
			want: true,
		},
		{
			name: "DifferentBlockLog2Dim",
			other: &blockForDotGeneralData{
				blockLog2Dim:    4, // Different
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{0},
				batchAxes:       []int{},
			},
			want: false,
		},
		{
			name: "DifferentContractingAxes",
			other: &blockForDotGeneralData{
				blockLog2Dim:    5,
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{1}, // Different
				batchAxes:       []int{},
			},
			want: false,
		},
		{
			name: "DifferentBatchAxes",
			other: &blockForDotGeneralData{
				blockLog2Dim:    5,
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{0},
				batchAxes:       []int{0}, // Different
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := base.EqualNodeData(tt.other)
			if ok, diff := testutil.IsEqual(tt.want, got); !ok {
				t.Errorf("Unexpected result (-want +got):\n%s", diff)
			}
		})
	}
}

// TestIsMatMulOrder tests the isMatMulOrder function for various axis configurations.
func TestIsMatMulOrder(t *testing.T) {
	if _, ok := backend.(*Backend); !ok {
		t.Skip("Test requires SimpleGo backend")
	}

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
			got := isMatMulOrder(tc.lhsShape, tc.lhsContractingAxes, tc.lhsBatchAxes,
				tc.rhsShape, tc.rhsContractingAxes, tc.rhsBatchAxes)
			if ok, diff := testutil.IsEqual(tc.want, got); !ok {
				t.Errorf("Unexpected result (-want +got):\n%s", diff)
			}
		})
	}
}

// TestDgUseSmallMatMul tests the build-time SmallMatMul path selection.
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
			// At contracting threshold (128)
			{"contractingSize_at_threshold", 1, 10, 10, 128, true},
			// Over contracting threshold
			{"contractingSize_over_threshold", 1, 10, 10, 129, false},
			// Batch size at threshold (64)
			{"batchSize_at_threshold", 64, 10, 10, 32, true},
			// Batch size over threshold
			{"batchSize_over_threshold", 65, 10, 10, 32, false},
			// M=1 special case - uses higher thresholds for K and N
			{"M_equals_1_moderate_K", 1, 1, 256, 512, true},
			// M=1 with K at M1 threshold (1024) should be accepted
			{"M_equals_1_K_at_M1_threshold", 1, 1, 256, 1024, true},
			// M=1 with K over M1 threshold should be rejected
			{"M_equals_1_K_over_M1_threshold", 1, 1, 256, 1025, false},
			// M=1 with very large K should be rejected
			{"M_equals_1_very_large_K", 1, 1, 256, 2000, false},
			// M=1 with large N should still work (within M1 threshold of 4096)
			{"M_equals_1_large_N", 1, 1, 1000, 256, true},
			// M=1 with very large N should be rejected (over M1 threshold of 4096)
			{"M_equals_1_very_large_N", 1, 1, 5000, 256, false},
			// M=1 with N exactly at M1 threshold (4096) should be accepted
			{"M_equals_1_N_at_M1_threshold", 1, 1, 4096, 256, true},
			// M=1 with N just over M1 threshold should be rejected
			{"M_equals_1_N_over_M1_threshold", 1, 1, 4097, 256, false},
			// M=1 with large batch should be rejected
			{"M_equals_1_large_batch", 100, 1, 256, 512, false},
			// N (rhsCrossSize) at threshold (256)
			{"rhsCrossSize_at_threshold", 1, 10, smallMatMulMaxRhsCrossSize, 64, true},
			// N over threshold
			{"rhsCrossSize_over_threshold", 1, 10, smallMatMulMaxRhsCrossSize + 1, 64, false},
			// Combined thresholds: both K and N at their limits
			{"K_and_N_both_at_threshold", 1, 10, smallMatMulMaxRhsCrossSize, 128, true},
			// Combined thresholds: K at limit, N over
			{"K_at_threshold_N_over", 1, 10, 257, 128, false},
			// Combined thresholds: K over, N at limit
			{"K_over_N_at_threshold", 1, 10, 256, 129, false},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				lhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.lhsCrossSize, tc.contractingSize)
				rhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.contractingSize, tc.rhsCrossSize)

				params := &dotGeneralNodeData{
					lhsContractingAxes: []int{2},
					lhsBatchAxes:       []int{0},
					rhsContractingAxes: []int{1},
					rhsBatchAxes:       []int{0},
					batchSize:          tc.batchSize,
					lhsCrossSize:       tc.lhsCrossSize,
					rhsCrossSize:       tc.rhsCrossSize,
					contractingSize:    tc.contractingSize,
				}

				got := dgUseSmallMatMul(dtypes.Float32, lhsShape, rhsShape, params)
				if got != tc.want {
					t.Errorf("dgCanUseSmallMatMul with batch=%d, M=%d, N=%d, K=%d: got %v, want %v",
						tc.batchSize, tc.lhsCrossSize, tc.rhsCrossSize, tc.contractingSize, got, tc.want)
				}
			})
		}
	})

	t.Run("DTypeSupport", func(t *testing.T) {
		params := &dotGeneralNodeData{
			lhsContractingAxes: []int{1},
			lhsBatchAxes:       []int{},
			rhsContractingAxes: []int{0},
			rhsBatchAxes:       []int{},
			batchSize:          1,
			lhsCrossSize:       4,
			rhsCrossSize:       6,
			contractingSize:    8,
		}

		// All numeric dtypes should be accepted by SmallMatMul
		supportedDTypes := []dtypes.DType{
			dtypes.Float32,
			dtypes.Float64,
			dtypes.BFloat16,
			dtypes.Float16,
			dtypes.Int8,
			dtypes.Int16,
			dtypes.Int32,
			dtypes.Int64,
			dtypes.Uint8,
			dtypes.Uint16,
			dtypes.Uint32,
			dtypes.Uint64,
		}
		for _, dtype := range supportedDTypes {
			lhs := shapes.Make(dtype, 4, 8)
			rhs := shapes.Make(dtype, 8, 6)
			if !dgUseSmallMatMul(dtype, lhs, rhs, params) {
				t.Errorf("Expected smallMatMul for dtype=%s, lhs=%s, rhs=%s, params=%+v",
					dtype, lhs.Shape(), rhs.Shape(), params)
			}
		}

		// Non-numeric dtypes should be rejected
		unsupportedDTypes := []dtypes.DType{
			dtypes.Bool,
			dtypes.Complex64,
			dtypes.Complex128,
		}
		for _, dtype := range unsupportedDTypes {
			lhs := shapes.Make(dtype, 4, 8)
			rhs := shapes.Make(dtype, 8, 6)
			if dgUseSmallMatMul(dtype, lhs, rhs, params) {
				t.Errorf("Expected not to use SmallMatMul for dtype=%s, lhs=%s, rhs=%s, params=%+v",
					dtype, lhs.Shape(), rhs.Shape(), params)
			}
		}
	})

	t.Run("NonMatMulOrderRejected", func(t *testing.T) {
		// Test with non-standard axis order (not [M,K]×[K,N])
		lhsShape := shapes.Make(dtypes.Float32, 8, 4) // [K, M] instead of [M, K]
		rhsShape := shapes.Make(dtypes.Float32, 8, 6) // [K, N]

		params := &dotGeneralNodeData{
			lhsContractingAxes: []int{0}, // K is first, not last
			lhsBatchAxes:       []int{},
			rhsContractingAxes: []int{0},
			rhsBatchAxes:       []int{},
			batchSize:          1,
			lhsCrossSize:       4,
			rhsCrossSize:       6,
			contractingSize:    8,
		}

		if dgUseSmallMatMul(dtypes.Float32, lhsShape, rhsShape, params) {
			t.Errorf("Expected not to use SmallMatMul for dtype=%s, lhs=%s, rhs=%s, params=%+v",
				dtypes.Float32, lhsShape, rhsShape, params)
		}
	})
}

// TestSmallMatMulCorrectness verifies that SmallMatMul produces correct results.
func TestSmallMatMulCorrectness(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	originalForce := goBackend.dotGeneralForceExecutionPath
	defer func() {
		goBackend.dotGeneralForceExecutionPath = originalForce
	}()

	testCases := []struct {
		name     string
		lhsDims  []int
		rhsDims  []int
		lhsContr []int
		lhsBatch []int
		rhsContr []int
		rhsBatch []int
	}{
		{"2D_matmul", []int{4, 8}, []int{8, 6}, []int{1}, []int{}, []int{0}, []int{}},
		{"matrix_vector", []int{4, 8}, []int{8}, []int{1}, []int{}, []int{0}, []int{}},
		{"M_equals_1", []int{1, 64}, []int{64, 32}, []int{1}, []int{}, []int{0}, []int{}},
		{"batched", []int{2, 4, 8}, []int{2, 8, 6}, []int{2}, []int{0}, []int{1}, []int{0}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test data
			lhsSize := 1
			for _, d := range tc.lhsDims {
				lhsSize *= d
			}
			rhsSize := 1
			for _, d := range tc.rhsDims {
				rhsSize *= d
			}

			lhsData := make([]float32, lhsSize)
			for i := range lhsData {
				lhsData[i] = float32(i+1) * 0.01
			}
			rhsData := make([]float32, rhsSize)
			for i := range rhsData {
				rhsData[i] = float32(i+1) * 0.01
			}

			lhsTensor := tensors.FromFlatDataAndDimensions(lhsData, tc.lhsDims...)
			rhsTensor := tensors.FromFlatDataAndDimensions(rhsData, tc.rhsDims...)

			// Compute with auto-select (may use SmallMatMul)
			goBackend.dotGeneralForceExecutionPath = autoSelectPath
			resultAutoRaw, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.DotGeneral(params[0], tc.lhsContr, tc.lhsBatch, params[1], tc.rhsContr, tc.rhsBatch, compute.DotGeneralConfig{})
			})
			if err != nil {
				t.Fatalf("testutil.Exec1 failed: %v", err)
			}

			// Compute with forced checkPath (uses normalized path, not SmallMatMul)
			goBackend.dotGeneralForceExecutionPath = checkPath
			resultNormalizedRaw, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.DotGeneral(params[0], tc.lhsContr, tc.lhsBatch, params[1], tc.rhsContr, tc.rhsBatch, compute.DotGeneralConfig{})
			})
			if err != nil {
				t.Fatalf("testutil.Exec1 failed: %v", err)
			}

			// Compare results
			autoFlat := testutil.FlattenSlice(resultAutoRaw).([]float32)
			normFlat := testutil.FlattenSlice(resultNormalizedRaw).([]float32)
			if ok, diff := testutil.IsInDelta(normFlat, autoFlat, 1e-3); !ok {
				t.Errorf("Results not within 1e-3 tolerance, -want +got:\n%s", diff)
			}
		})
	}

	// Test BFloat16 and Float16 SmallMatMul correctness
	t.Run("BFloat16", func(t *testing.T) {
		// Simple 4x8 × 8x6 matrix multiplication with BFloat16
		bf16 := bfloat16.FromFloat32
		lhsData := make([]bfloat16.BFloat16, 4*8)
		for i := range lhsData {
			lhsData[i] = bf16(float32(i+1) * 0.1)
		}
		rhsData := make([]bfloat16.BFloat16, 8*6)
		for i := range rhsData {
			rhsData[i] = bf16(float32(i+1) * 0.1)
		}
		lhsTensor := tensors.FromFlatDataAndDimensions(lhsData, 4, 8)
		rhsTensor := tensors.FromFlatDataAndDimensions(rhsData, 8, 6)

		// Force SmallMatMul path
		goBackend.dotGeneralForceExecutionPath = smallMatMulPath
		resultSmallMatMulRaw, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}

		// Use normalized path as reference
		goBackend.dotGeneralForceExecutionPath = normalizedPath
		resultNormalizedRaw, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}

		// BFloat16 has limited precision, allow 1% relative error
		smallMatMulData := testutil.FlattenSlice(resultSmallMatMulRaw).([]bfloat16.BFloat16)
		normalizedData := testutil.FlattenSlice(resultNormalizedRaw).([]bfloat16.BFloat16)
		for i := range smallMatMulData {
			if ok, diff := testutil.IsInDelta(smallMatMulData[i].Float32(), normalizedData[i].Float32(), 0.01); !ok {
				t.Fatalf("Mismatch at index %d: %s", i, diff)
			}
		}
	})

	t.Run("Float16", func(t *testing.T) {
		// Simple 4x8 × 8x6 matrix multiplication with Float16
		f16 := float16.FromFloat32
		lhsData := make([]float16.Float16, 4*8)
		for i := range lhsData {
			lhsData[i] = f16(float32(i+1) * 0.1)
		}
		rhsData := make([]float16.Float16, 8*6)
		for i := range rhsData {
			rhsData[i] = f16(float32(i+1) * 0.1)
		}
		lhsTensor := tensors.FromFlatDataAndDimensions(lhsData, 4, 8)
		rhsTensor := tensors.FromFlatDataAndDimensions(rhsData, 8, 6)

		// Force SmallMatMul path
		goBackend.dotGeneralForceExecutionPath = smallMatMulPath
		resultSmallMatMulRaw, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}

		// Use normalized path as reference
		goBackend.dotGeneralForceExecutionPath = normalizedPath
		resultNormalizedRaw, err := testutil.Exec1(goBackend, []any{lhsTensor.Value(), rhsTensor.Value()}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.DotGeneral(params[0], []int{1}, []int{}, params[1], []int{0}, []int{}, compute.DotGeneralConfig{})
		})
		if err != nil {
			t.Fatalf("testutil.Exec1 failed: %v", err)
		}

		// Float16 has better precision than BFloat16, allow 0.1% relative error
		smallMatMulData := testutil.FlattenSlice(resultSmallMatMulRaw).([]float16.Float16)
		normalizedData := testutil.FlattenSlice(resultNormalizedRaw).([]float16.Float16)
		for i := range smallMatMulData {
			if ok, diff := testutil.IsInDelta(smallMatMulData[i].Float32(), normalizedData[i].Float32(), 0.001); !ok {
				t.Fatalf("Mismatch at index %d: %s", i, diff)
			}
		}
	})
}

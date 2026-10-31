// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestSliceFastPathsAndParallel(t *testing.T) {
	// 1. Contiguous outer slice (numOuter == 0, >= 64K elements)
	t.Run("ContiguousBlock_Parallel", func(t *testing.T) {
		inDims := []int{4, 128, 256} // 131,072 elements
		starts := []int{1, 0, 0}
		limits := []int{3, 128, 256} // 65,536 elements
		strides := []int{1, 1, 1}
		verifySlice(t, inDims, starts, limits, strides)
	})

	// 2. Merged 1D outer loop of contiguous inner blocks (numOuter == 1, >= 64K elements)
	t.Run("SingleOuterLoop_Parallel", func(t *testing.T) {
		inDims := []int{2, 64, 8, 128}
		starts := []int{0, 0, 0, 32}
		limits := []int{2, 64, 8, 96} // [2, 64, 8, 64] = 65,536 elements
		strides := []int{1, 1, 1, 1}
		verifySlice(t, inDims, starts, limits, strides)
	})

	// 3. Multi-dimensional outer loop of contiguous inner blocks (numOuter >= 2, >= 64K elements)
	t.Run("MultiOuterLoop_Parallel", func(t *testing.T) {
		inDims := []int{3, 40, 20, 64}
		starts := []int{0, 4, 2, 16}
		limits := []int{3, 36, 18, 64} // [3, 32, 16, 48] = 73,728 elements
		strides := []int{1, 1, 1, 1}
		verifySlice(t, inDims, starts, limits, strides)
	})

	// 4. Strided slice on inner and outer axes (blockSize == 1, >= 64K elements)
	t.Run("StridedInnerAndOuter_Parallel", func(t *testing.T) {
		inDims := []int{2, 130, 1024}
		starts := []int{0, 1, 0}
		limits := []int{2, 129, 1024}
		strides := []int{1, 2, 2} // [2, 64, 512] = 65,536 elements
		verifySlice(t, inDims, starts, limits, strides)
	})
}

func verifySlice(t *testing.T, inDims, starts, limits, strides []int) {
	t.Helper()
	inShape := shapes.Make(dtypes.Int32, inDims...)
	inData := make([]int32, inShape.Size())
	for i := range inData {
		inData[i] = int32(i + 1)
	}

	builder := backend.Builder("test_slice")
	mainFn := builder.Main()
	xParam, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatal(err)
	}
	sliced, err := mainFn.Slice(xParam, starts, limits, strides)
	if err != nil {
		t.Fatal(err)
	}
	if err := mainFn.Return([]compute.Value{sliced}, nil); err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}

	inBuf := makeBuffer(t, inShape, inData)
	outBufs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	outShape, err := outBufs[0].Shape()
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, outShape.Size())
	if err := outBufs[0].ToFlatData(got); err != nil {
		t.Fatal(err)
	}

	// Verify against reference coordinate calculation.
	inStrides := inShape.Strides()
	outDims := outShape.Dimensions
	rank := len(inDims)
	coords := make([]int, rank)
	for outIdx, gotVal := range got {
		inIdx := 0
		for a := range rank {
			inIdx += (starts[a] + coords[a]*strides[a]) * inStrides[a]
		}
		wantVal := inData[inIdx]
		if gotVal != wantVal {
			t.Fatalf("mismatch at flat output index %d (coords=%v): got %d, want %d", outIdx, coords, gotVal, wantVal)
		}
		for a := rank - 1; a >= 0; a-- {
			coords[a]++
			if coords[a] < outDims[a] {
				break
			}
			coords[a] = 0
		}
	}
}

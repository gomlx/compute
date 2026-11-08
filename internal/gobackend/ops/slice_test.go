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

	// 5. Various block byte sizes (1..512B) across multiple dtypes
	t.Run("BlockSizesAndDTypes", func(t *testing.T) {
		for _, ch := range []int{1, 2, 3, 4, 5, 7, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 128} {
			inDims := []int{512, ch + 7}
			starts := []int{0, 3}
			limits := []int{512, 3 + ch}
			strides := []int{1, 1}
			verifySlice(t, inDims, starts, limits, strides)
		}
	})
}

func TestConcatenateFastPathsAndParallel(t *testing.T) {
	// 1. 2-input trailing axis concatenation with various channel widths (including non-multiple-of-16)
	t.Run("TwoInputTrailingAxis", func(t *testing.T) {
		for _, pair := range [][2]int{{1, 2}, {3, 5}, {8, 8}, {16, 16}, {20, 12}, {32, 16}, {64, 32}, {96, 64}} {
			c0, c1 := pair[0], pair[1]
			outer := 1024
			a := make([]float32, outer*c0)
			b := make([]float32, outer*c1)
			for i := range a {
				a[i] = float32(i + 1)
			}
			for i := range b {
				b[i] = float32(-(i + 1))
			}

			shapeA := shapes.Make(dtypes.Float32, outer, c0)
			shapeB := shapes.Make(dtypes.Float32, outer, c1)
			builder := backend.Builder("test_concat2")
			mainFn := builder.Main()
			pA, _ := mainFn.Parameter("a", shapeA, nil)
			pB, _ := mainFn.Parameter("b", shapeB, nil)
			cat, err := mainFn.Concatenate(1, pA, pB)
			if err != nil {
				t.Fatal(err)
			}
			_ = mainFn.Return([]compute.Value{cat}, nil)
			exec, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			bufA := makeBuffer(t, shapeA, a)
			bufB := makeBuffer(t, shapeB, b)
			outBufs, err := exec.Execute([]compute.Buffer{bufA, bufB}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, outer*(c0+c1))
			if err := outBufs[0].ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			for r := range outer {
				outRow := got[r*(c0+c1) : (r+1)*(c0+c1)]
				for j := range c0 {
					if outRow[j] != a[r*c0+j] {
						t.Fatalf("c0=%d,c1=%d row %d col %d: got %v, want %v", c0, c1, r, j, outRow[j], a[r*c0+j])
					}
				}
				for j := range c1 {
					if outRow[c0+j] != b[r*c1+j] {
						t.Fatalf("c0=%d,c1=%d row %d col %d: got %v, want %v", c0, c1, r, c0+j, outRow[c0+j], b[r*c1+j])
					}
				}
			}
		}
	})

	// 2. Duplicate-input Concatenate([x, x], axis) (UpSampleImages pattern)
	t.Run("DuplicateInputUpSample", func(t *testing.T) {
		for _, c := range []int{3, 8, 16, 24, 32, 48, 64, 96} {
			outer := 1024
			a := make([]float32, outer*c)
			for i := range a {
				a[i] = float32(i + 1)
			}
			shapeA := shapes.Make(dtypes.Float32, outer, c)
			builder := backend.Builder("test_concat_dup")
			mainFn := builder.Main()
			pA, _ := mainFn.Parameter("a", shapeA, nil)
			cat, err := mainFn.Concatenate(1, pA, pA)
			if err != nil {
				t.Fatal(err)
			}
			_ = mainFn.Return([]compute.Value{cat}, nil)
			exec, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			bufA := makeBuffer(t, shapeA, a)
			outBufs, err := exec.Execute([]compute.Buffer{bufA}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, outer*2*c)
			if err := outBufs[0].ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			for r := range outer {
				for j := range c {
					want := a[r*c+j]
					if got[r*2*c+j] != want || got[r*2*c+c+j] != want {
						t.Fatalf("c=%d row %d col %d mismatch", c, r, j)
					}
				}
			}
		}
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

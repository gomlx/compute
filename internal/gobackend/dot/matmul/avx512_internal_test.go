// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package matmul

import (
	"fmt"
	"simd/archsimd"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/support/humanize"
)

func TestAVX512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX2 is not supported on this architecture")
	}

	t.Run("Pack", func(t *testing.T) {
		t.Run("Float32", func(t *testing.T) {
			runPackLHSTests(t, avx512PackLHSKernelRows4[float32], 4)
			runPackRHSTests(t, avx512PackRHSNonTransposed[float32], 32)
			runApplyPackedOutputTests(t, avx512ApplyPackedOutputFloat32)
		})
		t.Run("BFloat16", func(t *testing.T) {
			runPackLHSTestsHalfPrecision(t, avx512PackLHSKernelRows4[bfloat16.BFloat16], 4)
			runPackRHSTestsHalfPrecision(t, avx512PackRHSNonTransposed[bfloat16.BFloat16], 32)
		})
		t.Run("Float16", func(t *testing.T) {
			runPackLHSTestsHalfPrecision(t, avx512PackLHSKernelRows4[float16.Float16], 4)
			runPackRHSTestsHalfPrecision(t, avx512PackRHSNonTransposed[float16.Float16], 32)
		})
		t.Run("Float64", func(t *testing.T) {
			runPackLHSTests(t, avx512PackLHSKernelRows4[float64], 4)
			runPackRHSTests(t, avx512PackRHSNonTransposed[float64], 16)
			runApplyPackedOutputTests(t, avx512ApplyPackedOutputFloat64)
		})
	})

	t.Run("Float16AsmDirect", func(t *testing.T) {
		// contractingLen = 2, lhsActiveRows = 2, rhsActiveCols = 2
		// LHS has 4 rows x 2 cols, packed in strips of 4 rows:
		// for col 0: row0, row1, row2, row3
		// for col 1: row0, row1, row2, row3
		lhs := make([]float16.Float16, 4*2)
		// col 0:
		lhs[0] = float16.FromFloat32(1) // row 0
		lhs[1] = float16.FromFloat32(3) // row 1
		lhs[2] = float16.FromFloat32(0) // row 2
		lhs[3] = float16.FromFloat32(0) // row 3
		// col 1:
		lhs[4] = float16.FromFloat32(2) // row 0
		lhs[5] = float16.FromFloat32(4) // row 1
		lhs[6] = float16.FromFloat32(0) // row 2
		lhs[7] = float16.FromFloat32(0) // row 3

		// RHS has 2 rows x 64 cols:
		rhs := make([]float16.Float16, 2*64)
		// row 0: col 0 = 10, col 1 = 11
		rhs[0] = float16.FromFloat32(10)
		rhs[1] = float16.FromFloat32(11)
		// row 1: col 0 = 12, col 1 = 13
		rhs[64] = float16.FromFloat32(12)
		rhs[65] = float16.FromFloat32(13)

		out := make([]float32, 4*64)
		avx512LargeKernelFloat16Asm(lhs, rhs, out, 4, 64, 2, 2, 2, false)
		if out[0] != 34 || out[1] != 37 || out[64] != 78 || out[65] != 85 {
			t.Fatalf("Float16AsmDirect: unexpected output: row0=[%v, %v], row1=[%v, %v]", out[0], out[1], out[64], out[65])
		}

		// Test accumulate = true:
		avx512LargeKernelFloat16Asm(lhs, rhs, out, 4, 64, 2, 2, 2, true)
		if out[0] != 68 || out[1] != 74 || out[64] != 156 || out[65] != 170 {
			t.Fatalf("Float16AsmDirect accumulate: unexpected output: row0=[%v, %v], row1=[%v, %v]", out[0], out[1], out[64], out[65])
		}
	})

	t.Run("Transpose/4x8x64bits", func(t *testing.T) {
		var input [4 * 8]uint64
		for i := range input {
			input[i] = uint64(i)
		}

		v0 := archsimd.LoadUint64x8Array((*[8]uint64)(unsafe.Pointer(&input[0*8])))
		v1 := archsimd.LoadUint64x8Array((*[8]uint64)(unsafe.Pointer(&input[1*8])))
		v2 := archsimd.LoadUint64x8Array((*[8]uint64)(unsafe.Pointer(&input[2*8])))
		v3 := archsimd.LoadUint64x8Array((*[8]uint64)(unsafe.Pointer(&input[3*8])))

		q0, q1, q2, q3 := avx512Transpose4x8x64bits(v0, v1, v2, v3)

		var output [4 * 8]uint64
		q0.StoreArray((*[8]uint64)(unsafe.Pointer(&output[0*8])))
		q1.StoreArray((*[8]uint64)(unsafe.Pointer(&output[1*8])))
		q2.StoreArray((*[8]uint64)(unsafe.Pointer(&output[2*8])))
		q3.StoreArray((*[8]uint64)(unsafe.Pointer(&output[3*8])))

		for c := range 8 { // logical column
			for r := range 4 { // logical row
				expected := uint64(r*8 + c)
				got := output[c*4+r]
				if got != expected {
					t.Errorf("At output col %d, row %d: got %d, expected %d", c, r, got, expected)
				}
			}
		}
	})

	t.Run("Transpose/4x16x32bits", func(t *testing.T) {
		var input [4 * 16]uint32
		for i := range input {
			input[i] = uint32(i)
		}

		v0 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(&input[0*16])))
		v1 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(&input[1*16])))
		v2 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(&input[2*16])))
		v3 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(&input[3*16])))

		q0, q1, q2, q3 := avx512Transpose4x16x32bits(v0, v1, v2, v3)

		// fmt.Printf("\nv0: [%s]\n", transposeIndicesFor4x16x32bits(v0))
		// fmt.Printf("v1: [%s]\n", transposeIndicesFor4x16x32bits(v1))
		// fmt.Printf("v0.InterleaveLoGrouped(v1)= [%s]\n\n",
		// 	transposeIndicesFor4x16x32bits(v0.InterleaveLoGrouped(v1)))

		fmt.Printf("q0: [%s]\n", transposeIndicesFor4x16x32bits(q0))
		fmt.Printf("q1: [%s]\n", transposeIndicesFor4x16x32bits(q1))
		fmt.Printf("q2: [%s]\n", transposeIndicesFor4x16x32bits(q2))
		fmt.Printf("q3: [%s]\n", transposeIndicesFor4x16x32bits(q3))

		var output [4 * 16]uint32
		q0.StoreArray((*[16]uint32)(unsafe.Pointer(&output[0*16])))
		q1.StoreArray((*[16]uint32)(unsafe.Pointer(&output[1*16])))
		q2.StoreArray((*[16]uint32)(unsafe.Pointer(&output[2*16])))
		q3.StoreArray((*[16]uint32)(unsafe.Pointer(&output[3*16])))

		for c := range 16 { // logical column
			for r := range 4 { // logical row
				expected := uint32(r*16 + c)
				got := output[c*4+r]
				if got != expected {
					t.Errorf("At output col %d, row %d: got %d, expected %d", c, r, got, expected)
				}
			}
		}
	})

	t.Run("Transpose/4x32x16bits", func(t *testing.T) {
		var input [4 * 32]uint16
		for i := range input {
			input[i] = uint16(i)
		}

		v0 := archsimd.LoadUint16x32Array((*[32]uint16)(unsafe.Pointer(&input[0*32])))
		v1 := archsimd.LoadUint16x32Array((*[32]uint16)(unsafe.Pointer(&input[1*32])))
		v2 := archsimd.LoadUint16x32Array((*[32]uint16)(unsafe.Pointer(&input[2*32])))
		v3 := archsimd.LoadUint16x32Array((*[32]uint16)(unsafe.Pointer(&input[3*32])))

		q0, q1, q2, q3 := avx512Transpose4x32x16bits(v0, v1, v2, v3)

		// fmt.Printf("\nv0: [%s]\n", transposeIndicesFor4x32x16bits(v0))
		// fmt.Printf("v1: [%s]\n", transposeIndicesFor4x32x16bits(v1))
		// fmt.Printf("v0.InterleaveLoGrouped(v1)= [%s]\n\n",
		// 	transposeIndicesFor4x32x16bits(v0.InterleaveLoGrouped(v1)))

		fmt.Printf("q0: [%s]\n", transposeIndicesFor4x32x16bits(q0))
		fmt.Printf("q1: [%s]\n", transposeIndicesFor4x32x16bits(q1))
		fmt.Printf("q2: [%s]\n", transposeIndicesFor4x32x16bits(q2))
		fmt.Printf("q3: [%s]\n", transposeIndicesFor4x32x16bits(q3))

		var output [4 * 32]uint16
		q0.StoreArray((*[32]uint16)(unsafe.Pointer(&output[0*32])))
		q1.StoreArray((*[32]uint16)(unsafe.Pointer(&output[1*32])))
		q2.StoreArray((*[32]uint16)(unsafe.Pointer(&output[2*32])))
		q3.StoreArray((*[32]uint16)(unsafe.Pointer(&output[3*32])))

		for c := range 32 { // logical column
			for r := range 4 { // logical row
				expected := uint16(r*32 + c)
				got := output[c*4+r]
				if got != expected {
					t.Errorf("At output col %d, row %d: got %d, expected %d", c, r, got, expected)
				}
			}
		}
	})

	t.Run("PackLHS/Float32Asm", func(t *testing.T) {
		for _, rows := range []int{1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 24, 32} {
			for _, cols := range []int{16, 32, 48, 64, 128, 192, 256, 17, 35} {
				lhsCols := cols + 8
				lhs := make([]float32, (rows+4)*lhsCols)
				for i := range lhs {
					lhs[i] = float32(i + 1)
				}
				numStrips := (rows + 3) / 4
				expectedPanel := make([]float32, numStrips*4*cols)
				gotPanel := make([]float32, numStrips*4*cols)

				// Compare AVX512UseAsm = false vs AVX512UseAsm = true
				AVX512UseAsm = false
				avx512PackLHSKernelRows4(lhs, expectedPanel, 2, 3, lhsCols, rows, cols, 4)
				AVX512UseAsm = true
				avx512PackLHSKernelRows4(lhs, gotPanel, 2, 3, lhsCols, rows, cols, 4)

				for i := range expectedPanel {
					if expectedPanel[i] != gotPanel[i] {
						t.Fatalf("Mismatch at rows=%d, cols=%d, idx=%d: expected %v, got %v", rows, cols, i, expectedPanel[i], gotPanel[i])
					}
				}
			}
		}
	})

	t.Run("PackLHS/Float64Asm", func(t *testing.T) {
		for _, rows := range []int{1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 24, 32} {
			for _, cols := range []int{8, 16, 24, 32, 64, 128, 192, 7, 13, 35} {
				lhsCols := cols + 8
				lhs := make([]float64, (rows+4)*lhsCols)
				for i := range lhs {
					lhs[i] = float64(i + 1)
				}
				numStrips := (rows + 3) / 4
				expectedPanel := make([]float64, numStrips*4*cols)
				gotPanel := make([]float64, numStrips*4*cols)

				AVX512UseAsm = false
				avx512PackLHSKernelRows4(lhs, expectedPanel, 2, 3, lhsCols, rows, cols, 4)
				AVX512UseAsm = true
				avx512PackLHSKernelRows4(lhs, gotPanel, 2, 3, lhsCols, rows, cols, 4)

				for i := range expectedPanel {
					if expectedPanel[i] != gotPanel[i] {
						t.Fatalf("Float64 Mismatch at rows=%d, cols=%d, idx=%d: expected %v, got %v", rows, cols, i, expectedPanel[i], gotPanel[i])
					}
				}
			}
		}
	})

	t.Run("PackLHS/Float16Asm", func(t *testing.T) {
		for _, rows := range []int{1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 24, 32} {
			for _, cols := range []int{16, 32, 48, 64, 128, 192, 256, 17, 33, 49} {
				lhsCols := cols + 8
				lhs := make([]float16.Float16, (rows+4)*lhsCols)
				for i := range lhs {
					lhs[i] = float16.FromFloat32(float32(i + 1))
				}
				numStrips := (rows + 3) / 4
				expectedPanel := make([]float16.Float16, numStrips*4*cols)
				gotPanel := make([]float16.Float16, numStrips*4*cols)

				AVX512UseAsm = false
				avx512PackLHSKernelRows4(lhs, expectedPanel, 2, 3, lhsCols, rows, cols, 4)
				AVX512UseAsm = true
				avx512PackLHSKernelRows4(lhs, gotPanel, 2, 3, lhsCols, rows, cols, 4)

				for i := range expectedPanel {
					if expectedPanel[i] != gotPanel[i] {
						t.Fatalf("Float16 Mismatch at rows=%d, cols=%d, idx=%d: expected %v, got %v", rows, cols, i, expectedPanel[i], gotPanel[i])
					}
				}
			}
		}
	})

	t.Run("PackLHS/BFloat16Asm", func(t *testing.T) {
		for _, rows := range []int{1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 24, 32} {
			for _, cols := range []int{16, 32, 48, 64, 128, 192, 256, 17, 33, 49} {
				lhsCols := cols + 8
				lhs := make([]bfloat16.BFloat16, (rows+4)*lhsCols)
				for i := range lhs {
					lhs[i] = bfloat16.FromFloat32(float32(i + 1))
				}
				numStrips := (rows + 3) / 4
				expectedPanel := make([]bfloat16.BFloat16, numStrips*4*cols)
				gotPanel := make([]bfloat16.BFloat16, numStrips*4*cols)

				AVX512UseAsm = false
				avx512PackLHSKernelRows4(lhs, expectedPanel, 2, 3, lhsCols, rows, cols, 4)
				AVX512UseAsm = true
				avx512PackLHSKernelRows4(lhs, gotPanel, 2, 3, lhsCols, rows, cols, 4)

				for i := range expectedPanel {
					if expectedPanel[i] != gotPanel[i] {
						t.Fatalf("BFloat16 Mismatch at rows=%d, cols=%d, idx=%d: expected %v, got %v", rows, cols, i, expectedPanel[i], gotPanel[i])
					}
				}
			}
		}
	})
}

func transposeIndicesFor4x16x32bits(vec archsimd.Uint32x16) string {
	var sb strings.Builder
	var values [16]uint32
	vec.StoreArray(&values)
	for i, val := range values {
		if i > 0 {
			sb.WriteString(", ")
		}
		vecNum := val / 16
		vecIdx := val % 16
		sb.WriteString(fmt.Sprintf("v_{%d,%d}", vecNum, vecIdx))
	}
	return sb.String()
}

func transposeIndicesFor4x32x16bits(vec archsimd.Uint16x32) string {
	var sb strings.Builder
	var values [32]uint16
	vec.StoreArray(&values)
	for i, val := range values {
		if i > 0 {
			sb.WriteString(", ")
		}
		vecNum := val / 32
		vecIdx := val % 32
		sb.WriteString(fmt.Sprintf("v_{%d,%d}", vecNum, vecIdx))
	}
	return sb.String()
}

func BenchmarkAVX512(b *testing.B) {
	sizes := []struct {
		name                 string
		totalRows, totalCols int
		panelRows, panelCols int
	}{
		{"Large-1_1536x1920", 1536, 1920, 32, 192},
		{"Large-2_1024x1920", 1024, 1920, 32, 192},
		{"Large-3_2048x2048", 2048, 2048, 32, 192},
	}

	for _, s := range sizes {
		b.Run(s.name+"/Float32/GoSIMD", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = false
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[float32](b, "float32", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})
		b.Run(s.name+"/Float32/Asm", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = true
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[float32](b, "float32", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})

		b.Run(s.name+"/Float64/GoSIMD", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = false
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[float64](b, "float64", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})
		b.Run(s.name+"/Float64/Asm", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = true
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[float64](b, "float64", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})

		b.Run(s.name+"/Float16/GoSIMD", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = false
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[float16.Float16](b, "float16", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})
		b.Run(s.name+"/Float16/Asm", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = true
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[float16.Float16](b, "float16", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})

		b.Run(s.name+"/BFloat16/GoSIMD", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = false
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[bfloat16.BFloat16](b, "bfloat16", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})
		b.Run(s.name+"/BFloat16/Asm", func(b *testing.B) {
			orig := AVX512UseAsm
			AVX512UseAsm = true
			defer func() { AVX512UseAsm = orig }()
			runBenchmarkPackLHS[bfloat16.BFloat16](b, "bfloat16", avx512PackLHSKernelRows4, s.totalRows, s.totalCols, s.panelRows, s.panelCols, 4)
		})
	}

	rhsSizes := []struct {
		name                 string
		contractingRows, rhsCols int
		panelContracting, panelCols int
		kernelCols           int
	}{
		{"Large-1_1920x1024", 1920, 1024, 192, 384, 64},
		{"Large-2_1920x1536", 1920, 1536, 192, 384, 64},
		{"Large-3_2048x2048", 2048, 2048, 192, 384, 64},
	}
	for _, s := range rhsSizes {
		b.Run("PackRHS/"+s.name+"/Float32", func(b *testing.B) {
			runBenchmarkPackRHS(b, "float32", avx512PackRHSNonTransposed[float32], s.contractingRows, s.rhsCols, s.panelContracting, s.panelCols, s.kernelCols)
		})
	}
}

func runBenchmarkPackRHS(b *testing.B, name string, packFn PackRHSFn[float32], contractingRows, rhsCols, panelContracting, panelCols, kernelCols int) {
	src := make([]float32, contractingRows*rhsCols)
	for i := range src {
		src[i] = float32(i)
	}
	numStrips := (panelCols + kernelCols - 1) / kernelCols
	dstSize := numStrips * panelContracting * kernelCols
	dst := make([]float32, dstSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for colStart := 0; colStart < rhsCols; colStart += panelCols {
			copyCols := min(panelCols, rhsCols-colStart)
			for rowStart := 0; rowStart < contractingRows; rowStart += panelContracting {
				copyRows := min(panelContracting, contractingRows-rowStart)
				packFn(src, dst, rowStart, colStart, rhsCols, copyRows, copyCols, kernelCols)
			}
		}
	}
	elapsed := b.Elapsed()
	if elapsed > 0 && b.N > 0 {
		durationPerOp := time.Duration(float64(elapsed) / float64(b.N))
		durStr := humanize.Duration(durationPerOp)
		b.ReportMetric(durationPerOp.Seconds()*1e6, "µs/op")
		_ = durStr
	}
}
func TestChoose2DSplit(t *testing.T) {
	params := AVX512ParamsFloat32
	for _, tc := range []struct {
		m, n, workers int
	}{
		{1536, 1024, 64},
		{1536, 1024, 32},
		{1536, 1536, 64},
		{2048, 2048, 64},
	} {
		numM, numN := choose2DSplit(tc.m, tc.n, tc.workers, &params)
		t.Logf("M=%d, N=%d, workers=%d -> numM=%d, numN=%d (colChunk=%d, rowChunk=%d)",
			tc.m, tc.n, tc.workers, numM, numN, (tc.n+numN-1)/numN, (tc.m+numM-1)/numM)
	}
}

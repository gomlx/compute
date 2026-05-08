// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package nontransposed

import (
	"fmt"
	"simd/archsimd"
	"strings"
	"testing"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/support/backendtest"
)

func TestAVX512(t *testing.T) {
	// Force AVX512 variant only for NonTransposed.
	defer func() {
		dot.ResetTestRegistrations()
	}()
	dot.ResetTestRegistrations()
	RegisterAVX512ForTests()

	backend := compute.MustNew()
	defer backend.Finalize()
	backendtest.TestDotGeneral(t, backend)

	t.Run("Pack", func(t *testing.T) {
		runPackLHSTests(t, avx512PackLHSKernelRows4, 4)
		runPackRHSTests(t, avx512PackRHSNonTransposed[float32], 32)
		runApplyPackedOutputTests(t, avx512ApplyPackedOutputFloat32)
	})

	t.Run("Transpose/4x16x32bits", func(t *testing.T) {
		var input [4 * 16]uint32
		for i := range input {
			input[i] = uint32(i)
		}

		v0 := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Pointer(&input[0*16])))
		v1 := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Pointer(&input[1*16])))
		v2 := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Pointer(&input[2*16])))
		v3 := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Pointer(&input[3*16])))

		fmt.Printf("v0: [%s]\n", transposeIndices(v0))
		fmt.Printf("v1: [%s]\n", transposeIndices(v1))
		fmt.Printf("v2: [%s]\n", transposeIndices(v2))
		fmt.Printf("v3: [%s]\n", transposeIndices(v3))

		q0, q1, q2, q3 := avx512Transpose4x16x32bits(v0, v1, v2, v3)

		fmt.Printf("q0: [%s]\n", transposeIndices(q0))
		fmt.Printf("q1: [%s]\n", transposeIndices(q1))
		fmt.Printf("q2: [%s]\n", transposeIndices(q2))
		fmt.Printf("q3: [%s]\n", transposeIndices(q3))

		var output [4 * 16]uint32
		q0.Store((*[16]uint32)(unsafe.Pointer(&output[0*16])))
		q1.Store((*[16]uint32)(unsafe.Pointer(&output[1*16])))
		q2.Store((*[16]uint32)(unsafe.Pointer(&output[2*16])))
		q3.Store((*[16]uint32)(unsafe.Pointer(&output[3*16])))

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
}

func transposeIndices(vec archsimd.Uint32x16) string {
	var sb strings.Builder
	var values [16]uint32
	vec.Store(&values)
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

func BenchmarkAVX512(b *testing.B) {
	const totalRows, totalCols = 1536, 1920
	const panelRows, panelCols = 24, 128
	runBenchmarkPackLHS[float32](b, "PackLHS/kernelRows=4/float32", avx512PackLHSKernelRows4, totalRows, totalCols, panelRows, panelCols, 4)
	runBenchmarkPackLHS[bfloat16.BFloat16](b, "PackLHS/kernelRows=4/bfloat16", avx512PackLHSKernelRows4, totalRows, totalCols, panelRows, panelCols, 4)
}

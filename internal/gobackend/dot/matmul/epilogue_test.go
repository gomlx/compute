// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"testing"
)

func TestAddBiasFloat32(t *testing.T) {
	for _, size := range []int{0, 1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 1536} {
		row := make([]float32, size)
		bias := make([]float32, size)
		expected := make([]float32, size)
		for i := 0; i < size; i++ {
			row[i] = float32(i) * 1.5
			bias[i] = float32(i) * 2.5
			expected[i] = row[i] + bias[i]
		}
		addBias(row, bias)
		for i := 0; i < size; i++ {
			if row[i] != expected[i] {
				t.Fatalf("size=%d, idx=%d: got %f, want %f", size, i, row[i], expected[i])
			}
		}
	}
}

func BenchmarkAddBiasFloat32(b *testing.B) {
	rowSize := 1536
	row := make([]float32, rowSize)
	bias := make([]float32, rowSize)
	for i := range bias {
		bias[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addBias(row, bias)
	}
}



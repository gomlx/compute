package ops_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestTransposeIterator(t *testing.T) {
	operand := shapes.Make(dtypes.Int32, 2, 3, 4)
	permutations := []int{2, 0, 1}
	it := ops.NewTransposeIterator(operand, permutations)
	transposedFlatIndices := make([]int, 0, operand.Size())
	for range operand.Size() {
		transposedFlatIndices = append(transposedFlatIndices, it.Next())
	}
	// fmt.Printf("\ttransposedFlatIndices=%#v\n", transposedFlatIndices)
	want := []int{
		// Operand axis 2 (the first being iterated) becomes output axis 0, in row-major order,
		// this is the largest one, with strides of 6:
		0, 6, 12, 18,
		1, 7, 13, 19,
		2, 8, 14, 20,

		3, 9, 15, 21,
		4, 10, 16, 22,
		5, 11, 17, 23}
	if ok, diff := testutil.IsEqual(want, transposedFlatIndices); !ok {
		t.Fatalf("transposeIterator mismatch:\n%s", diff)
	}
}

func TestTransposeFastPaths(t *testing.T) {
	testCases := []struct {
		name         string
		shape        shapes.Shape
		permutations []int
	}{
		{
			name:         "4D_0213_attention",
			shape:        shapes.Make(dtypes.Float32, 1, 64, 4, 32),
			permutations: []int{0, 2, 1, 3},
		},
		{
			name:         "4D_0213_batched",
			shape:        shapes.Make(dtypes.Float32, 2, 8, 4, 16),
			permutations: []int{0, 2, 1, 3},
		},
		{
			name:         "3D_102",
			shape:        shapes.Make(dtypes.Float32, 16, 8, 32),
			permutations: []int{1, 0, 2},
		},
		{
			name:         "2D_matrix_transpose",
			shape:        shapes.Make(dtypes.Float32, 64, 128),
			permutations: []int{1, 0},
		},
		{
			name:         "trailing_preserved_5D",
			shape:        shapes.Make(dtypes.Float32, 2, 3, 4, 5, 8),
			permutations: []int{0, 2, 1, 3, 4},
		},
		{
			name:         "identity_4D",
			shape:        shapes.Make(dtypes.Float32, 2, 3, 4, 5),
			permutations: []int{0, 1, 2, 3},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			size := tc.shape.Size()
			srcData := make([]float32, size)
			for i := range srcData {
				srcData[i] = float32(i + 1)
			}
			srcBuf := makeBuffer(t, tc.shape, srcData)

			outShape, err := shapeinference.Transpose(tc.shape, tc.permutations)
			if err != nil {
				t.Fatalf("Transpose shape failed: %+v", err)
			}
			dstBuf, err := backend.GetBuffer(outShape)
			if err != nil {
				t.Fatalf("GetBuffer failed: %+v", err)
			}

			// Run ExecuteTranspose (fast path)
			ops.ExecuteTranspose(backend, srcBuf, dstBuf, tc.permutations)

			// Compute expected via TransposeIterator
			expected := make([]float32, size)
			it := ops.NewTransposeIterator(tc.shape, tc.permutations)
			for _, val := range srcData {
				expected[it.Next()] = val
			}

			result := dstBuf.Flat.([]float32)
			for i := range result {
				if result[i] != expected[i] {
					t.Fatalf("%s at idx %d: got %f, want %f", tc.name, i, result[i], expected[i])
				}
			}
		})
	}
}

func BenchmarkTranspose4D_Attention_Iterator(b *testing.B) {
	// Shape: [1, 32768, 12, 64] -> [1, 12, 32768, 64]
	// Permutation: [0, 2, 1, 3]
	operandShape := shapes.Make(dtypes.Float32, 1, 32768, 12, 64)
	permutations := []int{0, 2, 1, 3}
	outShape := shapes.Make(dtypes.Float32, 1, 12, 32768, 64)

	operandFlat := make([]float32, operandShape.Size())
	outputFlat := make([]float32, outShape.Size())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		it := ops.NewTransposeIterator(operandShape, permutations)
		for _, value := range operandFlat {
			outputFlat[it.Next()] = value
		}
	}
}

func BenchmarkTranspose4D_Attention_FastPath(b *testing.B) {
	operandShape := shapes.Make(dtypes.Float32, 1, 32768, 12, 64)
	permutations := []int{0, 2, 1, 3}
	outShape := shapes.Make(dtypes.Float32, 1, 12, 32768, 64)

	operandFlat := make([]float32, operandShape.Size())
	srcBuf := makeBuffer(b, operandShape, operandFlat)
	dstBuf, err := backend.GetBuffer(outShape)
	if err != nil {
		b.Fatalf("GetBuffer failed: %+v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ops.ExecuteTranspose(backend, srcBuf, dstBuf, permutations)
	}
}



// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/support/testutil"
)

func testQuantizedGather(t *testing.T, b compute.Backend) {
	t.Run("QuantizedGather", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeQuantizedGather)

		t.Run("GGML", func(t *testing.T) {
			for _, wType := range allComplianceGGMLTypes {
				t.Run(wType.String(), func(t *testing.T) {
					const vocabSize = 4
					tableBytes := make([][]uint8, vocabSize)
					tableDeq := make([][]float32, vocabSize)
					for v := range vocabSize {
						tableBytes[v], tableDeq[v] = makeTestGGMLWeightRow(wType, 1, v+1)
					}
					indices := [][][]int32{
						{{2}, {0}, {3}},
						{{1}, {2}, {0}},
					}
					want := [][][]float32{
						{tableDeq[2], tableDeq[0], tableDeq[3]},
						{tableDeq[1], tableDeq[2], tableDeq[0]},
					}

					got, err := testutil.Exec1(b, []any{tableBytes, indices},
						func(f compute.Function, p []compute.Value) (compute.Value, error) {
							return f.QuantizedGather(p[0], p[1], &compute.Quantization{
								Scheme:   compute.QuantGGML,
								GGMLType: wType,
							})
						})
					if err != nil && compute.IsNotImplemented(err) {
						t.Skipf("Skipping QuantizedGather(%s): %v", wType, err)
					}
					if err != nil {
						t.Fatalf("QuantizedGather(%s) failed: %+v", wType, err)
					}
					if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
						t.Errorf("QuantizedGather(%s) mismatch:\n%s", wType, diff)
					}
				})
			}
		})

		const vocabSize, K, blockSize = 3, 8, 4
		const numBlocks = K / blockSize
		indices := [][][]int32{{{2}, {0}}, {{1}, {2}}}

		t.Run("NilQuant_SubByte", func(t *testing.T) {
			testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)

			t.Run("Int2", func(t *testing.T) {
				testutil.SkipIfMissingDType(t, b, dtypes.Int2)
				// Signed 2-bit values [-2, -1, 0, 1] encoded in 2's complement (2=0b10 -> -2, 3=0b11 -> -1).
				raw := [][]uint8{
					{0, 1, 2, 3, 3, 2, 1, 0},
					{1, 2, 0, 3, 2, 1, 3, 0},
					{3, 3, 1, 0, 0, 1, 2, 2},
				}
				toSigned := func(v uint8) int8 {
					if v >= 2 {
						return int8(v) - 4
					}
					return int8(v)
				}
				want := make([][][]int8, 2)
				for i := range 2 {
					want[i] = make([][]int8, 2)
					for j := range 2 {
						r := indices[i][j][0]
						want[i][j] = make([]int8, K)
						for c := range K {
							want[i][j][c] = toSigned(raw[r][c])
						}
					}
				}
				got, err := testutil.Exec1(b, []any{pack2BitMatrix(raw), indices},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						x, err := f.Bitcast(p[0], dtypes.Int2)
						if err != nil {
							return nil, err
						}
						x, err = f.Reshape(x, vocabSize, K)
						if err != nil {
							return nil, err
						}
						return f.QuantizedGather(x, p[1], nil)
					})
				if err != nil {
					t.Fatalf("QuantizedGather(Int2, nil) failed: %+v", err)
				}
				if ok, diff := testutil.IsEqual(want, got); !ok {
					t.Errorf("QuantizedGather(Int2, nil) mismatch:\n%s", diff)
				}
			})

			t.Run("Uint4", func(t *testing.T) {
				testutil.SkipIfMissingDType(t, b, dtypes.Uint4)
				vals := [][]uint8{
					{0, 5, 10, 15, 1, 6, 11, 14},
					{2, 7, 12, 13, 3, 8, 9, 4},
					{15, 14, 13, 12, 11, 10, 9, 8},
				}
				want := [][][]uint8{
					{vals[2], vals[0]},
					{vals[1], vals[2]},
				}
				got, err := testutil.Exec1(b, []any{pack4BitMatrix(vals), indices},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						x, err := f.Bitcast(p[0], dtypes.Uint4)
						if err != nil {
							return nil, err
						}
						x, err = f.Reshape(x, vocabSize, K)
						if err != nil {
							return nil, err
						}
						return f.QuantizedGather(x, p[1], nil)
					})
				if err != nil {
					t.Fatalf("QuantizedGather(Uint4, nil) failed: %+v", err)
				}
				if ok, diff := testutil.IsEqual(want, got); !ok {
					t.Errorf("QuantizedGather(Uint4, nil) mismatch:\n%s", diff)
				}
			})
		})

		t.Run("Uint4_BlockLinear_Uint4ZP", func(t *testing.T) {
			testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
			testutil.SkipIfMissingDType(t, b, dtypes.Uint4)
			vals := [][]uint8{
				{2, 3, 4, 5, 6, 7, 8, 9},
				{3, 4, 5, 6, 7, 8, 9, 10},
				{1, 2, 3, 4, 5, 6, 7, 8},
			}
			zpVals := [][]uint8{
				{1, 2},
				{3, 4},
				{0, 1},
			}
			scales := [][]float32{
				{0.5, 2.0},
				{1.5, 0.25},
				{1.0, 0.5},
			}
			packedW := pack4BitMatrix(vals)
			packedZP := pack4BitMatrix(zpVals)

			want := make([][][]float32, 2)
			for i := range 2 {
				want[i] = make([][]float32, 2)
				for j := range 2 {
					r := indices[i][j][0]
					want[i][j] = make([]float32, K)
					for c := range K {
						blk := c / blockSize
						want[i][j][c] = (float32(vals[r][c]) - float32(zpVals[r][blk])) * scales[r][blk]
					}
				}
			}

			got, err := testutil.Exec1(b, []any{packedW, indices, scales, packedZP},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					w, err := f.Bitcast(p[0], dtypes.Uint4)
					if err != nil {
						return nil, err
					}
					w, err = f.Reshape(w, vocabSize, K)
					if err != nil {
						return nil, err
					}
					zp, err := f.Bitcast(p[3], dtypes.Uint4)
					if err != nil {
						return nil, err
					}
					zp, err = f.Reshape(zp, vocabSize, numBlocks)
					if err != nil {
						return nil, err
					}
					return f.QuantizedGather(w, p[1], &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[2],
						ZeroPoint: zp,
						BlockAxis: 1,
						BlockSize: blockSize,
					})
				})
			if err != nil {
				t.Fatalf("QuantizedGather Uint4 failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
				t.Errorf("QuantizedGather Uint4 mismatch:\n%s", diff)
			}
		})

		t.Run("Uint2_Symmetric_Float16Scale", func(t *testing.T) {
			testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
			testutil.SkipIfMissingDType(t, b, dtypes.Uint2)
			testutil.SkipIfMissingDType(t, b, dtypes.Float16)
			vals := [][]uint8{
				{0, 1, 2, 3, 3, 2, 1, 0},
				{1, 1, 2, 2, 0, 3, 1, 2},
				{3, 0, 1, 2, 2, 1, 0, 3},
			}
			packedW := pack2BitMatrix(vals)
			scales := []float16.Float16{
				float16.FromFloat32(0.5),
				float16.FromFloat32(1.5),
				float16.FromFloat32(2.0),
			}
			want := make([][][]float16.Float16, 2)
			for i := range 2 {
				want[i] = make([][]float16.Float16, 2)
				for j := range 2 {
					r := indices[i][j][0]
					want[i][j] = make([]float16.Float16, K)
					sc := scales[r].Float32()
					for c := range K {
						want[i][j][c] = float16.FromFloat32(float32(vals[r][c]) * sc)
					}
				}
			}

			got, err := testutil.Exec1(b, []any{packedW, indices, scales},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					w, err := f.Bitcast(p[0], dtypes.Uint2)
					if err != nil {
						return nil, err
					}
					w, err = f.Reshape(w, vocabSize, K)
					if err != nil {
						return nil, err
					}
					return f.QuantizedGather(w, p[1], &compute.Quantization{
						Scheme: compute.QuantLinear,
						Scale:  p[2],
					})
				})
			if err != nil {
				t.Fatalf("QuantizedGather Uint2 Float16 failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-3); !ok {
				t.Errorf("QuantizedGather Uint2 Float16 mismatch:\n%s", diff)
			}
		})

		t.Run("Int8_NonDivisibleBlockSize", func(t *testing.T) {
			const kNonDiv, blk = 6, 4 // 2 blocks: [0..3], [4..5]
			w := makeInt8Matrix(vocabSize, kNonDiv)
			scales := [][]float32{{0.5, 2.0}, {1.0, 0.25}, {1.5, 0.5}}
			zp := [][]int8{{1, -1}, {0, 2}, {-2, 1}}
			want := make([][][]float32, 2)
			for i := range 2 {
				want[i] = make([][]float32, 2)
				for j := range 2 {
					r := indices[i][j][0]
					want[i][j] = make([]float32, kNonDiv)
					for c := range kNonDiv {
						bIdx := c / blk
						want[i][j][c] = (float32(w[r][c]) - float32(zp[r][bIdx])) * scales[r][bIdx]
					}
				}
			}

			got, err := testutil.Exec1(b, []any{w, indices, scales, zp},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.QuantizedGather(p[0], p[1], &compute.Quantization{
						Scheme:    compute.QuantLinear,
						Scale:     p[2],
						ZeroPoint: p[3],
						BlockAxis: 1,
						BlockSize: blk,
					})
				})
			if err != nil {
				t.Fatalf("QuantizedGather Int8 failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
				t.Errorf("QuantizedGather Int8 mismatch:\n%s", diff)
			}
		})

		t.Run("NF4_Uint4", func(t *testing.T) {
			testutil.SkipIfMissing(t, b, compute.OpTypeBitcast)
			testutil.SkipIfMissingDType(t, b, dtypes.Uint4)
			vals := [][]uint8{
				{0, 7, 15, 3, 8, 12, 1, 5},
				{2, 4, 6, 8, 10, 12, 14, 0},
				{15, 14, 13, 12, 11, 10, 9, 8},
			}
			packedW := pack4BitMatrix(vals)
			scales := [][]float32{{0.5, 1.0}, {2.0, 0.25}, {1.5, 0.5}}
			want := make([][][]float32, 2)
			for i := range 2 {
				want[i] = make([][]float32, 2)
				for j := range 2 {
					r := indices[i][j][0]
					want[i][j] = make([]float32, K)
					for c := range K {
						want[i][j][c] = compute.NF4LookupTable[vals[r][c]] * scales[r][c/blockSize]
					}
				}
			}

			got, err := testutil.Exec1(b, []any{packedW, indices, scales},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					w, err := f.Bitcast(p[0], dtypes.Uint4)
					if err != nil {
						return nil, err
					}
					w, err = f.Reshape(w, vocabSize, K)
					if err != nil {
						return nil, err
					}
					return f.QuantizedGather(w, p[1], &compute.Quantization{
						Scheme:    compute.QuantNF4,
						Scale:     p[2],
						BlockAxis: 1,
						BlockSize: blockSize,
					})
				})
			if err != nil {
				t.Fatalf("QuantizedGather NF4 failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
				t.Errorf("QuantizedGather NF4 mismatch:\n%s", diff)
			}
		})

		t.Run("PackedUint8_QuantNone_Int2", func(t *testing.T) {
			raw := [][]uint8{
				{0, 1, 2, 3, 3, 2, 1, 0},
				{1, 2, 0, 3, 2, 1, 3, 0},
				{3, 3, 1, 0, 0, 1, 2, 2},
			}
			want := make([][][]int8, 2)
			for i := range 2 {
				want[i] = make([][]int8, 2)
				for j := range 2 {
					r := indices[i][j][0]
					want[i][j] = make([]int8, K)
					for c := range K {
						v := raw[r][c]
						if v >= 2 {
							want[i][j][c] = int8(v) - 4
						} else {
							want[i][j][c] = int8(v)
						}
					}
				}
			}
			got, err := testutil.Exec1(b, []any{pack2BitMatrix(raw), indices},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.QuantizedGather(p[0], p[1], &compute.Quantization{
						Scheme:      compute.QuantNone,
						PackedDType: dtypes.Int2,
					})
				})
			if err != nil {
				t.Fatalf("QuantizedGather PackedUint8 QuantNone Int2 failed: %+v", err)
			}
			if ok, diff := testutil.IsEqual(want, got); !ok {
				t.Errorf("QuantizedGather PackedUint8 QuantNone Int2 mismatch:\n%s", diff)
			}
		})

		t.Run("PackedUint8_QuantLinear_Uint4_Uint8ZP", func(t *testing.T) {
			vals := [][]uint8{
				{2, 3, 4, 5, 6, 7, 8, 9},
				{3, 4, 5, 6, 7, 8, 9, 10},
				{1, 2, 3, 4, 5, 6, 7, 8},
			}
			zpVals := [][]uint8{
				{1, 2},
				{3, 4},
				{0, 5},
			}
			scales := [][]float32{{0.5, 2.0}, {1.5, 0.25}, {1.0, 0.5}}
			want := make([][][]float32, 2)
			for i := range 2 {
				want[i] = make([][]float32, 2)
				for j := range 2 {
					r := indices[i][j][0]
					want[i][j] = make([]float32, K)
					for c := range K {
						bIdx := c / blockSize
						want[i][j][c] = (float32(vals[r][c]) - float32(zpVals[r][bIdx])) * scales[r][bIdx]
					}
				}
			}
			got, err := testutil.Exec1(b, []any{pack4BitMatrix(vals), indices, scales, zpVals},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.QuantizedGather(p[0], p[1], &compute.Quantization{
						Scheme:      compute.QuantLinear,
						Scale:       p[2],
						ZeroPoint:   p[3],
						BlockAxis:   1,
						BlockSize:   blockSize,
						PackedDType: dtypes.Uint4,
					})
				})
			if err != nil {
				t.Fatalf("QuantizedGather PackedUint8 QuantLinear Uint4 failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
				t.Errorf("QuantizedGather PackedUint8 QuantLinear Uint4 mismatch:\n%s", diff)
			}
		})
	})
}

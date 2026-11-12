// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/support/testutil"
)

var allComplianceGGMLTypes = []compute.GGMLQuantType{
	compute.GGMLQ4_0,
	compute.GGMLQ4_1,
	compute.GGMLQ5_0,
	compute.GGMLQ5_1,
	compute.GGMLQ8_0,
	compute.GGMLIQ4NL,
	compute.GGMLQ2_K,
	compute.GGMLQ3_K,
	compute.GGMLQ4_K,
	compute.GGMLQ5_K,
	compute.GGMLQ6_K,
	compute.GGMLQ8_K,
}

func testQDGGMLAllModesAndFormats(t *testing.T, b compute.Backend) {
	// 1. Exhaustive sweep of all 9 GGML formats across Mode A (Float x), Mode B1 (Pre-quantized Q8_0/Q8_K x),
	//    and Mode B2 (QuantizedMatmul=true dynamic Q8 quantization).
	for _, wType := range allComplianceGGMLTypes {
		t.Run(wType.String(), func(t *testing.T) {
			vpb := wType.ValuesPerBlock()
			const numBlocks = 2
			const M, N = 3, 5
			K := numBlocks * vpb

			xType := compute.GGMLQ8_0
			if vpb == 256 {
				xType = compute.GGMLQ8_K
			}

			xF32 := make([][]float32, M)
			xQ8 := make([][]uint8, M)
			xQ8Deq := make([][]float32, M)
			for m := range M {
				xF32[m] = make([]float32, K)
				if m != 2 { // Row 2 stays all zeros to exercise maxAbs == 0 block handling!
					for k := range K {
						xF32[m][k] = float32(((m*17+k*7)%23)-11) * 0.12
					}
				}
				if xType == compute.GGMLQ8_0 {
					xQ8[m], xQ8Deq[m] = makeTestQ8_0Row(xF32[m])
				} else {
					xQ8[m], xQ8Deq[m] = makeTestQ8_KRow(xF32[m])
				}
			}

			wBytes := make([][]uint8, N)
			wDeq := make([][]float32, N)
			for n := range N {
				wBytes[n], wDeq[n] = makeTestGGMLWeightRow(wType, numBlocks, n+1)
			}
			bias := makeFloat32Vec(N, 0.1)

			wantFloat := make([][]float32, M)
			wantQ8 := make([][]float32, M)
			for m := range M {
				wantFloat[m] = make([]float32, N)
				wantQ8[m] = make([]float32, N)
				for n := range N {
					sf := bias[n]
					sq := bias[n]
					for k := range K {
						sf += xF32[m][k] * wDeq[n][k]
						sq += xQ8Deq[m][k] * wDeq[n][k]
					}
					wantFloat[m][n] = sf
					wantQ8[m][n] = sq
				}
			}

			t.Run("FloatX", func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{xF32, wBytes, bias},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						return f.FusedQuantizedDense(p[0], p[1], p[2], compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: wType},
							WeightLayout:       compute.DenseLayoutOutputsInput,
						})
					})
				skipIfQuantizedDenseNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("GGML(%s) FloatX failed: %+v", wType, err)
				}
				if ok, diff := testutil.IsInDelta(wantFloat, got, 1e-3); !ok {
					t.Errorf("GGML(%s) FloatX mismatch:\n%s", wType, diff)
				}
			})

			t.Run("PreQuantized_"+xType.String(), func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{xQ8, wBytes, bias},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						return f.FusedQuantizedDense(p[0], p[1], p[2], compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: wType},
							XQuantization:      &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: xType},
							WeightLayout:       compute.DenseLayoutOutputsInput,
						})
					})
				skipIfQuantizedDenseNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("GGML(%s x %s) failed: %+v", wType, xType, err)
				}
				if ok, diff := testutil.IsInDelta(wantQ8, got, 1e-2); !ok {
					t.Errorf("GGML(%s x %s) mismatch:\n%s", wType, xType, diff)
				}
			})

			t.Run("QuantizedMatmul", func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{xF32, wBytes, bias},
					func(f compute.Function, p []compute.Value) (compute.Value, error) {
						return f.FusedQuantizedDense(p[0], p[1], p[2], compute.FusedQuantizedDenseConfig{
							WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: wType},
							QuantizedMatmul:    true,
							WeightLayout:       compute.DenseLayoutOutputsInput,
						})
					})
				skipIfQuantizedDenseNotImplemented(t, b, err)
				if err != nil {
					t.Fatalf("GGML(%s) QuantizedMatmul failed: %+v", wType, err)
				}
				// Backends may either quantize dynamically to Q8 (matching wantQ8 within 1e-2)
				// or ignore the QuantizedMatmul hint and compute in float (matching wantFloat within 1e-3).
				okQ8, diffQ8 := testutil.IsInDelta(wantQ8, got, 1e-2)
				okF32, _ := testutil.IsInDelta(wantFloat, got, 1e-3)
				if !okQ8 && !okF32 {
					t.Errorf("GGML(%s) QuantizedMatmul matched neither Q8 nor Float reference:\n%s", wType, diffQ8)
				}
			})
		})
	}

	// 2. GGML 3D pre-quantized Q8_0 activations + ActivationRelu + OutputQuantization (Uint8), and Half-Precision Float16/BFloat16 x!
	t.Run("3D_PreQuantizedQ8_WithReluAndOutputQuant", func(t *testing.T) {
		const B, S, K, N = 2, 2, 64, 4
		x3DQ8 := make([][][]uint8, B)
		x3DDeq := make([][][]float32, B)
		for bi := range B {
			x3DQ8[bi] = make([][]uint8, S)
			x3DDeq[bi] = make([][]float32, S)
			for si := range S {
				src := makeFloat32Vec(K, float32(bi*S+si+1)*0.15)
				x3DQ8[bi][si], x3DDeq[bi][si] = makeTestQ8_0Row(src)
			}
		}
		wBytes := make([][]uint8, N)
		wDeq := make([][]float32, N)
		for n := range N {
			wBytes[n], wDeq[n] = makeTestGGMLWeightRow(compute.GGMLQ4_0, K/32, n+3)
		}
		yScale := float32(0.1)
		yZP := uint8(128)

		want := make([][][]uint8, B)
		for bi := range B {
			want[bi] = make([][]uint8, S)
			for si := range S {
				want[bi][si] = make([]uint8, N)
				for n := range N {
					var sum float32
					for k := range K {
						sum += x3DDeq[bi][si][k] * wDeq[n][k]
					}
					if sum < 0 {
						sum = 0
					}
					q := float32(math.RoundToEven(float64(sum/yScale))) + float32(yZP)
					q = max(0, min(255, q))
					want[bi][si][n] = uint8(int32(math.RoundToEven(float64(q))))
				}
			}
		}

		got, err := testutil.Exec1(b, []any{x3DQ8, wBytes, yScale, yZP},
			func(f compute.Function, p []compute.Value) (compute.Value, error) {
				return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
					WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ4_0},
					XQuantization:      &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0},
					OutputQuantization: &compute.Quantization{Scheme: compute.QuantLinear, Scale: p[2], ZeroPoint: p[3]},
					Activation:         compute.ActivationConfig{Type: compute.ActivationRelu},
					WeightLayout:       compute.DenseLayoutOutputsInput,
				})
			})
		skipIfQuantizedDenseNotImplemented(t, b, err)
		if err != nil {
			t.Fatalf("GGML 3D PreQuantized + OutputQuant failed: %+v", err)
		}
		// Allow +/- 1 LSB for minor float rounding near half-way quantization boundaries.
		if ok, diff := testutil.IsInDelta(want, got, 1); !ok {
			t.Errorf("GGML 3D PreQuantized + OutputQuant mismatch:\n%s", diff)
		}
	})

	t.Run("HalfPrecision_Float16_And_BFloat16_X", func(t *testing.T) {
		const M, K, N = 2, 32, 4
		wBytes := make([][]uint8, N)
		wDeq := make([][]float32, N)
		for n := range N {
			wBytes[n], wDeq[n] = makeTestGGMLWeightRow(compute.GGMLQ8_0, 1, n+1)
		}

		t.Run("Float16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.Float16)
			xF16 := make([][]float16.Float16, M)
			want := make([][]float16.Float16, M)
			for m := range M {
				xF16[m] = make([]float16.Float16, K)
				want[m] = make([]float16.Float16, N)
				for k := range K {
					xF16[m][k] = float16.FromFloat32(float32((m*5+k)%9-4) * 0.25)
				}
				for n := range N {
					var sum float32
					for k := range K {
						sum += xF16[m][k].Float32() * wDeq[n][k]
					}
					want[m][n] = float16.FromFloat32(sum)
				}
			}
			got, err := testutil.Exec1(b, []any{xF16, wBytes},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0},
						WeightLayout:       compute.DenseLayoutOutputsInput,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("GGML Float16 failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 1e-2); !ok {
				t.Errorf("GGML Float16 mismatch:\n%s", diff)
			}
		})

		t.Run("BFloat16", func(t *testing.T) {
			testutil.SkipIfMissingDType(t, b, dtypes.BFloat16)
			xBF16 := make([][]bfloat16.BFloat16, M)
			want := make([][]bfloat16.BFloat16, M)
			for m := range M {
				xBF16[m] = make([]bfloat16.BFloat16, K)
				want[m] = make([]bfloat16.BFloat16, N)
				for k := range K {
					xBF16[m][k] = bfloat16.FromFloat32(float32((m*5+k)%9-4) * 0.25)
				}
				for n := range N {
					var sum float32
					for k := range K {
						sum += xBF16[m][k].Float32() * wDeq[n][k]
					}
					want[m][n] = bfloat16.FromFloat32(sum)
				}
			}
			got, err := testutil.Exec1(b, []any{xBF16, wBytes},
				func(f compute.Function, p []compute.Value) (compute.Value, error) {
					return f.FusedQuantizedDense(p[0], p[1], nil, compute.FusedQuantizedDenseConfig{
						WeightQuantization: &compute.Quantization{Scheme: compute.QuantGGML, GGMLType: compute.GGMLQ8_0},
						WeightLayout:       compute.DenseLayoutOutputsInput,
					})
				})
			skipIfQuantizedDenseNotImplemented(t, b, err)
			if err != nil {
				t.Fatalf("GGML BFloat16 failed: %+v", err)
			}
			if ok, diff := testutil.IsInDelta(want, got, 2e-2); !ok {
				t.Errorf("GGML BFloat16 mismatch:\n%s", diff)
			}
		})
	})
}

func testQuantizedEmbeddingLookup(t *testing.T, b compute.Backend) {
	t.Run("QuantizedEmbeddingLookup", func(t *testing.T) {
		testutil.SkipIfMissing(t, b, compute.OpTypeQuantizedEmbeddingLookup)

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
						return f.QuantizedEmbeddingLookup(p[0], p[1], &compute.Quantization{
							Scheme:   compute.QuantGGML,
							GGMLType: wType,
						})
					})
				if err != nil && compute.IsNotImplemented(err) {
					t.Skipf("Skipping QuantizedEmbeddingLookup(%s): %v", wType, err)
				}
				if err != nil {
					t.Fatalf("QuantizedEmbeddingLookup(%s) failed: %+v", wType, err)
				}
				if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
					t.Errorf("QuantizedEmbeddingLookup(%s) mismatch:\n%s", wType, diff)
				}
			})
		}
	})
}

// --- Self-Contained Reference GGML Block Builders & Dequantizers ---

func makeTestQ8_0Row(src []float32) ([]uint8, []float32) {
	nblocks := len(src) / 32
	out := make([]uint8, nblocks*34)
	deq := make([]float32, len(src))
	for blk := range nblocks {
		span := src[blk*32 : (blk+1)*32]
		var maxAbs float32
		for _, v := range span {
			av := float32(math.Abs(float64(v)))
			if av > maxAbs {
				maxAbs = av
			}
		}
		if maxAbs == 0 {
			continue
		}
		dF16 := float16.FromFloat32(maxAbs / 127.0)
		binary.LittleEndian.PutUint16(out[blk*34:blk*34+2], dF16.Bits())
		d := dF16.Float32()
		for i, v := range span {
			var q int8
			if d > 0 {
				qi := int32(math.RoundToEven(float64(v / d)))
				qi = max(-127, min(127, qi))
				q = int8(qi)
			}
			out[blk*34+2+i] = uint8(q)
			deq[blk*32+i] = d * float32(q)
		}
	}
	return out, deq
}

func makeTestQ8_KRow(src []float32) ([]uint8, []float32) {
	nblocks := len(src) / 256
	out := make([]uint8, nblocks*292)
	deq := make([]float32, len(src))
	for blk := range nblocks {
		span := src[blk*256 : (blk+1)*256]
		var maxAbs float32
		for _, v := range span {
			av := float32(math.Abs(float64(v)))
			if av > maxAbs {
				maxAbs = av
			}
		}
		if maxAbs == 0 {
			continue
		}
		d := maxAbs / 127.0
		binary.LittleEndian.PutUint32(out[blk*292:blk*292+4], math.Float32bits(d))
		for g := range 16 {
			var bsum int16
			for l := range 16 {
				idx := g*16 + l
				qi := int32(math.RoundToEven(float64(span[idx] / d)))
				qi = max(-127, min(127, qi))
				q := int8(qi)
				out[blk*292+4+idx] = uint8(q)
				deq[blk*256+idx] = d * float32(q)
				bsum += int16(q)
			}
			binary.LittleEndian.PutUint16(out[blk*292+260+2*g:blk*292+262+2*g], uint16(bsum))
		}
	}
	return out, deq
}

// makeTestGGMLWeightRow builds deterministic valid GGML block bytes and their exact reference float32 dequantization
// for any of the 9 supported GGMLQuantType formats (including full 6-bit K-quant scales/mins and high-bit masks).
func makeTestGGMLWeightRow(wType compute.GGMLQuantType, numBlocks, seed int) ([]uint8, []float32) {
	bpb := wType.BytesPerBlock()
	vpb := wType.ValuesPerBlock()
	raw := make([]uint8, numBlocks*bpb)
	deq := make([]float32, numBlocks*vpb)

	for blk := range numBlocks {
		bSeed := seed*17 + blk*31 + 1
		bRaw := raw[blk*bpb : (blk+1)*bpb]
		bDeq := deq[blk*vpb : (blk+1)*vpb]

		switch wType {
		case compute.GGMLQ4_0:
			dF16 := float16.FromFloat32(0.08 + float32((bSeed%5))*0.02)
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			d := dF16.Float32()
			for i := range 16 {
				lo := uint8((bSeed + i*3) & 0x0F)
				hi := uint8((bSeed*3 + i*5 + 7) & 0x0F)
				bRaw[2+i] = lo | (hi << 4)
				bDeq[i] = d * float32(int(lo)-8)
				bDeq[16+i] = d * float32(int(hi)-8)
			}

		case compute.GGMLQ4_1:
			dF16 := float16.FromFloat32(0.06 + float32((bSeed%5))*0.02)
			mF16 := float16.FromFloat32(-0.35 + float32((bSeed%7))*0.05)
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			binary.LittleEndian.PutUint16(bRaw[2:4], mF16.Bits())
			d := dF16.Float32()
			m := mF16.Float32()
			for i := range 16 {
				lo := uint8((bSeed + i*3) & 0x0F)
				hi := uint8((bSeed*3 + i*5 + 7) & 0x0F)
				bRaw[4+i] = lo | (hi << 4)
				bDeq[i] = d*float32(lo) + m
				bDeq[16+i] = d*float32(hi) + m
			}

		case compute.GGMLQ5_0:
			dF16 := float16.FromFloat32(0.05 + float32((bSeed%5))*0.015)
			qh := uint32(bSeed*0x9E3779B1) ^ 0xA5A55A5A
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			binary.LittleEndian.PutUint32(bRaw[2:6], qh)
			d := dF16.Float32()
			for i := range 16 {
				lo := uint8((bSeed + i*3) & 0x0F)
				hi := uint8((bSeed*3 + i*5 + 7) & 0x0F)
				bRaw[6+i] = lo | (hi << 4)
				xh0 := ((qh >> uint(i)) << 4) & 0x10
				xh1 := (qh >> uint(i+12)) & 0x10
				bDeq[i] = d * float32(int32(uint32(lo)|xh0)-16)
				bDeq[16+i] = d * float32(int32(uint32(hi)|xh1)-16)
			}

		case compute.GGMLQ5_1:
			dF16 := float16.FromFloat32(0.04 + float32((bSeed%5))*0.01)
			mF16 := float16.FromFloat32(-0.5 + float32((bSeed%6))*0.08)
			qh := uint32(bSeed*0x85EBCA6B) ^ 0x5A5AA5A5
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			binary.LittleEndian.PutUint16(bRaw[2:4], mF16.Bits())
			binary.LittleEndian.PutUint32(bRaw[4:8], qh)
			d := dF16.Float32()
			m := mF16.Float32()
			for i := range 16 {
				lo := uint8((bSeed + i*3) & 0x0F)
				hi := uint8((bSeed*3 + i*5 + 7) & 0x0F)
				bRaw[8+i] = lo | (hi << 4)
				xh0 := ((qh >> uint(i)) << 4) & 0x10
				xh1 := (qh >> uint(i+12)) & 0x10
				bDeq[i] = d*float32(uint32(lo)|xh0) + m
				bDeq[16+i] = d*float32(uint32(hi)|xh1) + m
			}

		case compute.GGMLQ8_0:
			dF16 := float16.FromFloat32(0.03 + float32((bSeed%4))*0.01)
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			d := dF16.Float32()
			for i := range 32 {
				q := int8(((bSeed*5 + i*11) % 61) - 30)
				bRaw[2+i] = uint8(q)
				bDeq[i] = d * float32(q)
			}

		case compute.GGMLIQ4NL:
			dF16 := float16.FromFloat32(0.005 + float32((bSeed%4))*0.002)
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			d := dF16.Float32()
			for i := range 16 {
				lo := uint8((bSeed + i*5) & 0x0F)
				hi := uint8((bSeed*7 + i*3 + 2) & 0x0F)
				bRaw[2+i] = lo | (hi << 4)
				bDeq[i] = d * compute.IQ4NLLookupTable[lo]
				bDeq[16+i] = d * compute.IQ4NLLookupTable[hi]
			}

		case compute.GGMLQ2_K:
			// 84 bytes: scales[16], qs[64], fp16 d, fp16 dmin.
			dF16 := float16.FromFloat32(0.06)
			dminF16 := float16.FromFloat32(0.02)
			binary.LittleEndian.PutUint16(bRaw[80:82], dF16.Bits())
			binary.LittleEndian.PutUint16(bRaw[82:84], dminF16.Bits())
			d := dF16.Float32()
			dmin := dminF16.Float32()
			for j := range 16 {
				sc := uint8(((bSeed + j*3) % 15) + 1)
				mn := uint8((bSeed*2 + j*5) % 12)
				bRaw[j] = sc | (mn << 4)
			}
			for i := range 64 {
				bRaw[16+i] = uint8((bSeed*13 + i*29) & 0xFF)
			}
			idx, is := 0, 0
			for n := 0; n < 256; n += 128 {
				shift := uint(0)
				for range 4 {
					sc := bRaw[is]
					is++
					dl := d * float32(sc&0x0F)
					ml := dmin * float32(sc>>4)
					for l := range 16 {
						bDeq[idx] = dl*float32((bRaw[16+n/4+l]>>shift)&3) - ml
						idx++
					}
					sc = bRaw[is]
					is++
					dl = d * float32(sc&0x0F)
					ml = dmin * float32(sc>>4)
					for l := range 16 {
						bDeq[idx] = dl*float32((bRaw[16+n/4+16+l]>>shift)&3) - ml
						idx++
					}
					shift += 2
				}
			}

		case compute.GGMLQ3_K:
			// 110 bytes: hmask[32], qs[64], scales[12] (16 6-bit scales), fp16 d.
			for i := range 32 {
				bRaw[i] = uint8((bSeed*19 + i*37) & 0xFF)
			}
			for i := range 64 {
				bRaw[32+i] = uint8((bSeed*23 + i*41) & 0xFF)
			}
			var sc6 [16]uint8
			for j := range 16 {
				sc6[j] = uint8((bSeed*3 + j*7) & 0x3F) // Full 6-bit [0..63]
			}
			packTestQ3KScales(sc6, bRaw[96:108])
			dF16 := float16.FromFloat32(0.025)
			binary.LittleEndian.PutUint16(bRaw[108:110], dF16.Bits())
			dAll := dF16.Float32()

			idx, is, qOff := 0, 0, 0
			mBit := uint8(1)
			for range 2 {
				shift := uint(0)
				for range 4 {
					dl := dAll * float32(int32(sc6[is])-32)
					is++
					for l := range 16 {
						q := int8((bRaw[32+qOff+l] >> shift) & 3)
						if bRaw[l]&mBit == 0 {
							q -= 4
						}
						bDeq[idx] = dl * float32(q)
						idx++
					}
					dl = dAll * float32(int32(sc6[is])-32)
					is++
					for l := range 16 {
						q := int8((bRaw[32+qOff+16+l] >> shift) & 3)
						if bRaw[16+l]&mBit == 0 {
							q -= 4
						}
						bDeq[idx] = dl * float32(q)
						idx++
					}
					shift += 2
					mBit <<= 1
				}
				qOff += 32
			}

		case compute.GGMLQ4_K:
			// 144 bytes: fp16 d, fp16 dmin, scales[12] (full 6-bit [0..63] scales & mins!), qs[128].
			dF16 := float16.FromFloat32(0.015)
			dminF16 := float16.FromFloat32(0.008)
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			binary.LittleEndian.PutUint16(bRaw[2:4], dminF16.Bits())
			d := dF16.Float32()
			dmin := dminF16.Float32()

			var scs, mins [8]uint8
			for j := range 8 {
				// Include values >= 16 (up to 63) so the upper 2 bits of 6-bit K4 scale packing are exercised!
				scs[j] = uint8(((bSeed*5 + j*9) % 50) + 1)
				mins[j] = uint8((bSeed*3 + j*11) % 45)
			}
			packTestK4ScalesMins(scs, mins, bRaw[4:16])
			for i := range 128 {
				bRaw[16+i] = uint8((bSeed*11 + i*17) & 0xFF)
			}
			idx := 0
			for chunk := range 4 {
				is := chunk * 2
				d1, m1 := d*float32(scs[is]), dmin*float32(mins[is])
				d2, m2 := d*float32(scs[is+1]), dmin*float32(mins[is+1])
				qOff := chunk * 32
				for l := range 32 {
					bDeq[idx] = d1*float32(bRaw[16+qOff+l]&0x0F) - m1
					idx++
				}
				for l := range 32 {
					bDeq[idx] = d2*float32(bRaw[16+qOff+l]>>4) - m2
					idx++
				}
			}

		case compute.GGMLQ5_K:
			// 176 bytes: fp16 d, fp16 dmin, scales[12], qh[32], qs[128].
			dF16 := float16.FromFloat32(0.01)
			dminF16 := float16.FromFloat32(0.006)
			binary.LittleEndian.PutUint16(bRaw[0:2], dF16.Bits())
			binary.LittleEndian.PutUint16(bRaw[2:4], dminF16.Bits())
			d := dF16.Float32()
			dmin := dminF16.Float32()

			var scs, mins [8]uint8
			for j := range 8 {
				scs[j] = uint8(((bSeed*7 + j*11) % 52) + 1)
				mins[j] = uint8((bSeed*5 + j*9) % 40)
			}
			packTestK4ScalesMins(scs, mins, bRaw[4:16])
			for i := range 32 {
				bRaw[16+i] = uint8((bSeed*29 + i*43) & 0xFF)
			}
			for i := range 128 {
				bRaw[48+i] = uint8((bSeed*13 + i*19) & 0xFF)
			}
			idx, qlOff := 0, 0
			u1, u2 := uint8(1), uint8(2)
			for chunk := range 4 {
				is := chunk * 2
				d1, m1 := d*float32(scs[is]), dmin*float32(mins[is])
				d2, m2 := d*float32(scs[is+1]), dmin*float32(mins[is+1])
				for l := range 32 {
					hb := uint8(0)
					if bRaw[16+l]&u1 != 0 {
						hb = 16
					}
					bDeq[idx] = d1*float32((bRaw[48+qlOff+l]&0x0F)+hb) - m1
					idx++
				}
				for l := range 32 {
					hb := uint8(0)
					if bRaw[16+l]&u2 != 0 {
						hb = 16
					}
					bDeq[idx] = d2*float32((bRaw[48+qlOff+l]>>4)+hb) - m2
					idx++
				}
				qlOff += 32
				u1 <<= 2
				u2 <<= 2
			}

		case compute.GGMLQ6_K:
			// 210 bytes: ql[128], qh[64], int8 scales[16], fp16 d.
			for i := range 128 {
				bRaw[i] = uint8((bSeed*17 + i*23) & 0xFF)
			}
			for i := range 64 {
				bRaw[128+i] = uint8((bSeed*31 + i*37) & 0xFF)
			}
			for j := range 16 {
				sc := int8(((bSeed*3 + j*5) % 21) - 10)
				if sc == 0 {
					sc = 3
				}
				bRaw[192+j] = uint8(sc)
			}
			dF16 := float16.FromFloat32(0.015)
			binary.LittleEndian.PutUint16(bRaw[208:210], dF16.Bits())
			d := dF16.Float32()

			idx, qlOff, qhOff, scOff := 0, 0, 0, 0
			for range 2 {
				for l := range 32 {
					is := l / 16
					q1 := int8((bRaw[qlOff+l]&0x0F)|((bRaw[128+qhOff+l]&3)<<4)) - 32
					q2 := int8((bRaw[qlOff+l+32]&0x0F)|(((bRaw[128+qhOff+l]>>2)&3)<<4)) - 32
					q3 := int8((bRaw[qlOff+l]>>4)|(((bRaw[128+qhOff+l]>>4)&3)<<4)) - 32
					q4 := int8((bRaw[qlOff+l+32]>>4)|(((bRaw[128+qhOff+l]>>6)&3)<<4)) - 32
					bDeq[idx+l] = d * float32(int8(bRaw[192+scOff+is])) * float32(q1)
					bDeq[idx+l+32] = d * float32(int8(bRaw[192+scOff+is+2])) * float32(q2)
					bDeq[idx+l+64] = d * float32(int8(bRaw[192+scOff+is+4])) * float32(q3)
					bDeq[idx+l+96] = d * float32(int8(bRaw[192+scOff+is+6])) * float32(q4)
				}
				idx += 128
				qlOff += 64
				qhOff += 32
				scOff += 8
			}

		case compute.GGMLQ8_K:
			var src [256]float32
			for i := range 256 {
				src[i] = float32(((bSeed*7+i*11)%29)-14) * 0.08
			}
			blkRaw, blkDeq := makeTestQ8_KRow(src[:])
			copy(bRaw, blkRaw)
			copy(bDeq, blkDeq)
		}
	}
	return raw, deq
}

// packTestK4ScalesMins packs 8 6-bit scales and 8 6-bit mins into 12 GGML K4 scale bytes.
func packTestK4ScalesMins(scs, mins [8]uint8, dst []uint8) {
	for j := range 4 {
		dst[j] = (scs[j] & 63) | ((scs[4+j] >> 4) << 6)
		dst[4+j] = (mins[j] & 63) | ((mins[4+j] >> 4) << 6)
		dst[8+j] = (scs[4+j] & 0x0F) | ((mins[4+j] & 0x0F) << 4)
	}
}

// packTestQ3KScales packs 16 6-bit unsigned values [0..63] into 12 GGML Q3_K scale bytes.
func packTestQ3KScales(sc [16]uint8, dst []uint8) {
	for i := range 4 {
		dst[i] = (sc[i] & 0x0F) | ((sc[8+i] & 0x0F) << 4)
		dst[4+i] = (sc[4+i] & 0x0F) | ((sc[12+i] & 0x0F) << 4)
		dst[8+i] = ((sc[i] >> 4) & 3) |
			(((sc[4+i] >> 4) & 3) << 2) |
			(((sc[8+i] >> 4) & 3) << 4) |
			(((sc[12+i] >> 4) & 3) << 6)
	}
}

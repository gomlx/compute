// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
)

func TestDispatchSDPAAVX512_Correctness(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this machine")
	}

	for _, headDim := range []int{16, 32, 64, 128} {
		for _, causal := range []bool{false, true} {
			seqLen := 2
			kvLen := 2
			groupSize := 1

			q := make([]float32, seqLen*headDim)
			k := make([]float32, kvLen*headDim)
			v := make([]float32, kvLen*headDim)
			output := make([]float32, seqLen*headDim)
			scratch := make([]float32, seqLen*kvLen)

			for i := range q {
				q[i] = 1.0
			}
			for i := range k {
				k[i] = 1.0
			}
			// v[0] = 10, v[1] = 20
			for d := range headDim {
				v[0*headDim+d] = 10.0
				v[1*headDim+d] = 20.0
			}

			ok := DispatchSDPAAVX512(
				q, k, v, output,
				0, 0, headDim, headDim, headDim*seqLen,
				nil, nil, 0,
				nil, 0,
				scratch,
				groupSize, 0, seqLen, kvLen, headDim,
				1.0, causal,
				seqLen, kvLen,
			)
			if !ok {
				t.Fatalf("DispatchSDPAAVX512 returned false for headDim=%d", headDim)
			}

			wantQ0 := float32(15.0)
			if causal {
				wantQ0 = 10.0
			}
			wantQ1 := float32(15.0)

			for d := range headDim {
				got0 := output[0*headDim+d]
				if math.Abs(float64(got0-wantQ0)) > 1e-4 {
					t.Errorf("headDim=%d causal=%v: query 0 dim %d mismatch: got %f, want %f", headDim, causal, d, got0, wantQ0)
					break
				}
				got1 := output[1*headDim+d]
				if math.Abs(float64(got1-wantQ1)) > 1e-4 {
					t.Errorf("headDim=%d causal=%v: query 1 dim %d mismatch: got %f, want %f", headDim, causal, d, got1, wantQ1)
					break
				}
			}
		}
	}
}

func TestDispatchSDPAAVX512_VariousKVLengthsAndMasks(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this machine")
	}

	for _, headDim := range []int{24, 32, 64, 128} {
		for _, kvLen := range []int{1, 3, 4, 5, 15, 16, 19, 31, 32, 53} {
			for _, causal := range []bool{false, true} {
				for _, withAddMask := range []bool{false, true} {
					seqLen := 7
					qLimit := 5
					kvLimit := max(1, kvLen-2)
					if kvLen <= 3 {
						kvLimit = kvLen
					}
					numHeads := 3 // test BSHD-style non-unit stride (qSeqStride = numHeads*headDim)
					qSeqStride := numHeads * headDim
					kvSeqStride := numHeads * headDim
					scale := float32(1.0 / math.Sqrt(float64(headDim)))

					q := make([]float32, seqLen*qSeqStride)
					k := make([]float32, kvLen*kvSeqStride)
					v := make([]float32, kvLen*kvSeqStride)
					for i := range q {
						q[i] = float32((i%17)-8) * 0.125
					}
					for i := range k {
						k[i] = float32((i%13)-6) * 0.15
					}
					for i := range v {
						v[i] = float32((i%19)-9) * 0.2
					}

					var addMask []float32
					if withAddMask {
						addMask = make([]float32, kvLen)
						for ki := range kvLen {
							if ki%5 == 4 {
								addMask[ki] = float32(math.Inf(-1))
							} else {
								addMask[ki] = float32(ki%3) * -0.1
							}
						}
					}

					got := make([]float32, seqLen*qSeqStride)
					for i := range got {
						got[i] = -999.0
					}
					scratch := make([]float32, kvLen+16)

					qOff := headDim // test head index 1 offset
					kvOff := headDim
					DispatchSDPAAVX512(
						q, k, v, got,
						qOff, kvOff, qSeqStride, kvSeqStride, headDim,
						addMask, nil, 0,
						nil, 0,
						scratch,
						1, 0, seqLen, kvLen, headDim,
						scale, causal,
						qLimit, kvLimit,
					)

					// Reference computation in float64
					for qIdx := range seqLen {
						outBase := qOff + qIdx*qSeqStride
						if qIdx >= qLimit {
							for d := range headDim {
								if got[outBase+d] != 0 {
									t.Fatalf("headDim=%d kvLen=%d qIdx=%d d=%d: expected 0 beyond qLimit, got %f", headDim, kvLen, qIdx, d, got[outBase+d])
								}
							}
							continue
						}
						kvUnmasked := min(kvLen, kvLimit)
						if causal {
							kvUnmasked = min(kvUnmasked, qIdx+1)
						}
						scores := make([]float64, kvUnmasked)
						rowMax := math.Inf(-1)
						qBase := qOff + qIdx*qSeqStride
						for ki := range kvUnmasked {
							kBase := kvOff + ki*kvSeqStride
							var dot float64
							for d := range headDim {
								dot += float64(q[qBase+d]) * float64(k[kBase+d])
							}
							s := dot * float64(scale)
							if withAddMask {
								s += float64(addMask[ki])
							}
							scores[ki] = s
							if s > rowMax {
								rowMax = s
							}
						}
						if math.IsInf(rowMax, -1) {
							for d := range headDim {
								if got[outBase+d] != 0 {
									t.Fatalf("expected 0 when rowMax=-Inf, got %f", got[outBase+d])
								}
							}
							continue
						}
						var sum float64
						for ki := range kvUnmasked {
							if math.IsInf(scores[ki], -1) {
								scores[ki] = 0
							} else {
								scores[ki] = math.Exp(scores[ki] - rowMax)
								sum += scores[ki]
							}
						}
						for d := range headDim {
							var acc float64
							for ki := range kvUnmasked {
								acc += (scores[ki] / sum) * float64(v[kvOff+ki*kvSeqStride+d])
							}
							if math.Abs(float64(got[outBase+d])-acc) > 1e-3 {
								t.Fatalf("headDim=%d kvLen=%d causal=%v mask=%v qIdx=%d d=%d: got %f, want %f",
									headDim, kvLen, causal, withAddMask, qIdx, d, got[outBase+d], acc)
							}
						}
					}
				}
			}
		}
	}
}


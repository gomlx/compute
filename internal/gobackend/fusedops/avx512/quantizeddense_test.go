// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx512

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/fusedops"
)

func TestQuantizedPanelGEMMAVX512VNNI_SignednessAndShapes(t *testing.T) {
	if !gobackend.IsAVX512VNNIAllowed {
		t.Skip("AVX-512 VNNI is not supported or allowed on this machine")
	}

	// 1. Explicit signedness check: uw = 200 (unsigned > 127) * qx = -5 (signed < 0) = -1000 per element.
	{
		const M, N, numSubBlocks, quads = 7, 35, 2, 4 // 16 elements per sub-block, non-multiples of 6 and 32
		kPadded := numSubBlocks * quads * 4
		qx := make([]int8, M*kPadded)
		for i := range qx {
			qx[i] = -5
		}
		dxPairs := make([]float32, M*numSubBlocks*2)
		for m := range M {
			for b := range numSubBlocks {
				dx := float32(0.5) * float32(m+1)
				dxPairs[(m*numSubBlocks+b)*2] = dx
				dxPairs[(m*numSubBlocks+b)*2+1] = dx * float32(-5*16)
			}
		}

		numStrips := (N + 15) / 16
		subBlkBytes := gobackend.QuantizedPanelStripSubBlockBytes(quads)
		stripBytes := numSubBlocks * subBlkBytes
		panel := make([]uint8, numStrips*stripBytes)
		for s := range numStrips {
			validC := min(16, N-s*16)
			strip := panel[s*stripBytes : (s+1)*stripBytes]
			for b := range numSubBlocks {
				sb := strip[b*subBlkBytes : (b+1)*subBlkBytes]
				for q := range quads {
					for c := range validC {
						off := q*64 + 4*c
						sb[off+0] = 200
						sb[off+1] = 200
						sb[off+2] = 200
						sb[off+3] = 200
					}
				}
				for c := range validC {
					dw := float32(0.25) * float32(c+1)
					mw := float32(-1.5)
					binary.LittleEndian.PutUint32(sb[quads*64+4*c:], math.Float32bits(dw))
					binary.LittleEndian.PutUint32(sb[quads*64+64+4*c:], math.Float32bits(mw))
				}
			}
		}

		bias := make([]float32, N)
		for i := range bias {
			bias[i] = float32(i) * 0.125
		}

		got := make([]float32, M*N)
		want := make([]float32, M*N)
		cfgGot := gobackend.QuantizedPanelGEMMConfig{
			Qx: qx, DxPairs: dxPairs, Panel: panel, Bias: bias, Out: got,
			M: M, N: N, NStart: 0, ChunkCols: N, KPadded: kPadded,
			NumSubBlocks: numSubBlocks, QuadsPerSubBlk: quads, WideUnsigned: true,
		}
		cfgWant := cfgGot
		cfgWant.Out = want

		DispatchQuantizedPanelGEMMAVX512VNNI(&cfgGot)
		gobackend.ExecQuantizedPanelGEMMScalar(&cfgWant)

		for i := range want {
			diff := math.Abs(float64(got[i] - want[i]))
			if diff > 1e-4 {
				t.Fatalf("mismatch at idx %d (m=%d, n=%d): got %v, want %v", i, i/N, i%N, got[i], want[i])
			}
		}
	}

	// 2. Sweep quadsPerSubBlk in {4, 8, 5} (unrolled 4, unrolled 8, generic loop) and tail shapes.
	for _, quads := range []int{4, 8, 5} {
		for _, M := range []int{1, 5, 6, 13} {
			for _, N := range []int{1, 15, 16, 17, 32, 49} {
				const numSubBlocks = 3
				kPadded := numSubBlocks * quads * 4
				qx := make([]int8, M*kPadded)
				for i := range qx {
					qx[i] = int8((i*7)%251 - 125)
				}
				dxPairs := make([]float32, M*numSubBlocks*2)
				for i := range M * numSubBlocks {
					dx := float32((i%7)+1) * 0.03
					var sumQ int32
					m := i / numSubBlocks
					b := i % numSubBlocks
					for k := range quads * 4 {
						sumQ += int32(qx[m*kPadded+b*quads*4+k])
					}
					dxPairs[2*i] = dx
					dxPairs[2*i+1] = dx * float32(sumQ)
				}

				numStrips := (N + 15) / 16
				subBlkBytes := gobackend.QuantizedPanelStripSubBlockBytes(quads)
				stripBytes := numSubBlocks * subBlkBytes
				panel := make([]uint8, numStrips*stripBytes)
				for s := range numStrips {
					validC := min(16, N-s*16)
					strip := panel[s*stripBytes : (s+1)*stripBytes]
					for b := range numSubBlocks {
						sb := strip[b*subBlkBytes : (b+1)*subBlkBytes]
						for q := range quads {
							for c := range validC {
								off := q*64 + 4*c
								sb[off+0] = uint8((s*31 + b*17 + q*13 + c*5) & 0xFF)
								sb[off+1] = uint8((s*29 + b*19 + q*11 + c*7) & 0xFF)
								sb[off+2] = uint8((s*23 + b*13 + q*7 + c*11) & 0xFF)
								sb[off+3] = uint8((s*19 + b*11 + q*5 + c*13) & 0xFF)
							}
						}
						for c := range validC {
							dw := float32((c%5)+1) * 0.02
							mw := float32((c%3)-1) * 0.1
							binary.LittleEndian.PutUint32(sb[quads*64+4*c:], math.Float32bits(dw))
							binary.LittleEndian.PutUint32(sb[quads*64+64+4*c:], math.Float32bits(mw))
						}
					}
				}

				got := make([]float32, M*N)
				want := make([]float32, M*N)
				cfgGot := gobackend.QuantizedPanelGEMMConfig{
					Qx: qx, DxPairs: dxPairs, Panel: panel, Out: got,
					M: M, N: N, NStart: 0, ChunkCols: N, KPadded: kPadded,
					NumSubBlocks: numSubBlocks, QuadsPerSubBlk: quads, WideUnsigned: true,
				}
				cfgWant := cfgGot
				cfgWant.Out = want

				DispatchQuantizedPanelGEMMAVX512VNNI(&cfgGot)
				gobackend.ExecQuantizedPanelGEMMScalar(&cfgWant)

				for i := range want {
					if math.Abs(float64(got[i]-want[i])) > 1e-3 {
						t.Fatalf("quads=%d M=%d N=%d idx=%d: got %v, want %v", quads, M, N, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestQuantizeGGMLActivationsAVX512_MatchesScalar(t *testing.T) {
	if !gobackend.IsAVX512Allowed {
		t.Skip("AVX-512 is not supported or allowed on this machine")
	}
	for _, tc := range []struct {
		K, subBlk int
		q8K       bool
	}{
		{384, 32, false},
		{512, 32, true},
		{512, 16, true},
	} {
		const M = 5
		x := make([]float32, M*tc.K)
		for i := range x {
			if i >= tc.K && i < 2*tc.K {
				x[i] = 0 // test all-zero row handling
			} else {
				x[i] = float32(math.Sin(float64(i)*0.17)) * 3.5
			}
		}
		numSub := tc.K / tc.subBlk
		qxGot := make([]int8, M*tc.K)
		dxGot := make([]float32, M*numSub*2)
		qxWant := make([]int8, M*tc.K)
		dxWant := make([]float32, M*numSub*2)

		QuantizeGGMLActivationsAVX512(x, qxGot, dxGot, 0, M, tc.K, tc.subBlk, tc.q8K)
		fusedops.QuantizeGGMLPanelActivationsScalar(x, qxWant, dxWant, 0, M, tc.K, tc.subBlk, tc.q8K)

		for i := range qxWant {
			if math.Abs(float64(qxGot[i]-qxWant[i])) > 1 {
				t.Fatalf("K=%d sub=%d q8K=%v qx[%d]: got %d, want %d", tc.K, tc.subBlk, tc.q8K, i, qxGot[i], qxWant[i])
			}
		}
		for i := range dxWant {
			if math.Abs(float64(dxGot[i]-dxWant[i])) > 1e-3 {
				t.Fatalf("K=%d sub=%d q8K=%v dx[%d]: got %v, want %v", tc.K, tc.subBlk, tc.q8K, i, dxGot[i], dxWant[i])
			}
		}
	}
}

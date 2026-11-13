// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/fusedops"
)

func TestQuantizedPanelGEMMAVX2_NarrowAndWideSaturation(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this machine")
	}

	// 1. Test extreme 8-bit saturation corner case: uw = 255 * qx = -128 across all 4 bytes!
	// Naive VPMADDUBSW would saturate 255 * -128 * 2 = -65280 to -32768; our wide split must be exact!
	{
		const M, N, numSubBlocks, quads = 5, 19, 2, 4
		kPadded := numSubBlocks * quads * 4
		qx := make([]int8, M*kPadded)
		for i := range qx {
			qx[i] = -128
		}
		dxPairs := make([]float32, M*numSubBlocks*2)
		for i := range M * numSubBlocks {
			dxPairs[2*i] = 0.01
			dxPairs[2*i+1] = 0.01 * float32(-128*16)
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
						sb[off+0] = 255
						sb[off+1] = 255
						sb[off+2] = 255
						sb[off+3] = 255
					}
				}
				for c := range validC {
					dw := float32(0.05) * float32(c+1)
					mw := float32(-6.4)
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

		DispatchQuantizedPanelGEMMAVX2(&cfgGot)
		gobackend.ExecQuantizedPanelGEMMScalar(&cfgWant)

		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-3 {
				t.Fatalf("saturation test mismatch at %d: got %v, want %v", i, got[i], want[i])
			}
		}
	}

	// 2. Test both narrow (uw <= 63) and wide (uw <= 255) across odd M and N shapes.
	for _, wide := range []bool{false, true} {
		for _, M := range []int{1, 2, 3, 7} {
			for _, N := range []int{1, 15, 16, 17, 33} {
				const numSubBlocks, quads = 3, 8
				kPadded := numSubBlocks * quads * 4
				qx := make([]int8, M*kPadded)
				for i := range qx {
					qx[i] = int8((i*11)%255 - 127)
				}
				dxPairs := make([]float32, M*numSubBlocks*2)
				for i := range M * numSubBlocks {
					dx := float32((i%5)+1) * 0.02
					m := i / numSubBlocks
					b := i % numSubBlocks
					var sumQ int32
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
				maxMask := uint8(0x3F)
				if wide {
					maxMask = 0xFF
				}
				for s := range numStrips {
					validC := min(16, N-s*16)
					strip := panel[s*stripBytes : (s+1)*stripBytes]
					for b := range numSubBlocks {
						sb := strip[b*subBlkBytes : (b+1)*subBlkBytes]
						for q := range quads {
							for c := range validC {
								off := q*64 + 4*c
								sb[off+0] = uint8(s*31+b*17+q*13+c*5) & maxMask
								sb[off+1] = uint8(s*29+b*19+q*11+c*7) & maxMask
								sb[off+2] = uint8(s*23+b*13+q*7+c*11) & maxMask
								sb[off+3] = uint8(s*19+b*11+q*5+c*13) & maxMask
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

				bias := make([]float32, N)
				for i := range bias {
					bias[i] = float32(i) * 0.05
				}

				got := make([]float32, M*N)
				want := make([]float32, M*N)
				cfgGot := gobackend.QuantizedPanelGEMMConfig{
					Qx: qx, DxPairs: dxPairs, Panel: panel, Bias: bias, Out: got,
					M: M, N: N, NStart: 0, ChunkCols: N, KPadded: kPadded,
					NumSubBlocks: numSubBlocks, QuadsPerSubBlk: quads, WideUnsigned: wide,
				}
				cfgWant := cfgGot
				cfgWant.Out = want

				DispatchQuantizedPanelGEMMAVX2(&cfgGot)
				gobackend.ExecQuantizedPanelGEMMScalar(&cfgWant)

				for i := range want {
					if math.Abs(float64(got[i]-want[i])) > 1e-3 {
						t.Fatalf("wide=%v M=%d N=%d idx=%d: got %v, want %v", wide, M, N, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestQuantizeGGMLActivationsAVX2_MatchesScalar(t *testing.T) {
	if !gobackend.IsAVX2Allowed {
		t.Skip("AVX2 is not supported or allowed on this machine")
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
			x[i] = float32(math.Cos(float64(i)*0.13)) * 2.7
		}
		numSub := tc.K / tc.subBlk
		qxGot := make([]int8, M*tc.K)
		dxGot := make([]float32, M*numSub*2)
		qxWant := make([]int8, M*tc.K)
		dxWant := make([]float32, M*numSub*2)

		QuantizeGGMLActivationsAVX2(x, qxGot, dxGot, 0, M, tc.K, tc.subBlk, tc.q8K)
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

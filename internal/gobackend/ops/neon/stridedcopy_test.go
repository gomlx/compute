// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon_test

import (
	"bytes"
	"testing"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops/neon"
)

func TestStridedCopyNEON(t *testing.T) {
	if !gobackend.IsNEONAllowed {
		t.Skip("ARM64 NEON is not supported or allowed on this host")
	}

	blockSizes := []int{16, 17, 23, 31, 32, 33, 47, 48, 49, 63, 64, 65, 96, 127, 128, 192, 255, 256, 384, 512, 1024}
	count := 37
	for _, bSize := range blockSizes {
		srcStride := bSize + 13
		dstStride := bSize + 7
		src := make([]byte, count*srcStride)
		for i := range src {
			src[i] = byte((i*31 + 17) & 0xFF)
		}
		dst := make([]byte, count*dstStride)
		want := make([]byte, count*dstStride)
		for r := range count {
			copy(want[r*dstStride:r*dstStride+bSize], src[r*srcStride:r*srcStride+bSize])
		}
		if !neon.DispatchCopyStridedBytesNEON(dst, src, count, bSize, dstStride, srcStride) {
			t.Fatalf("DispatchCopyStridedBytesNEON returned false for bSize=%d", bSize)
		}
		if !bytes.Equal(dst, want) {
			t.Fatalf("DispatchCopyStridedBytesNEON mismatch for bSize=%d", bSize)
		}

		// Test DupStridedBytesNEON
		srcContig := make([]byte, count*bSize)
		for i := range srcContig {
			srcContig[i] = byte((i*73 + 5) & 0xFF)
		}
		dstDup := make([]byte, count*2*bSize)
		wantDup := make([]byte, count*2*bSize)
		for r := range count {
			copy(wantDup[2*r*bSize:(2*r+1)*bSize], srcContig[r*bSize:(r+1)*bSize])
			copy(wantDup[(2*r+1)*bSize:(2*r+2)*bSize], srcContig[r*bSize:(r+1)*bSize])
		}
		if !neon.DispatchDupStridedBytesNEON(dstDup, srcContig, count, bSize) {
			t.Fatalf("DispatchDupStridedBytesNEON returned false for bSize=%d", bSize)
		}
		if !bytes.Equal(dstDup, wantDup) {
			t.Fatalf("DispatchDupStridedBytesNEON mismatch for bSize=%d", bSize)
		}
	}

	concatSizes := []int{16, 19, 32, 48, 63, 64, 96, 128}
	for _, b0 := range concatSizes {
		for _, b1 := range concatSizes {
			src0 := make([]byte, count*b0)
			src1 := make([]byte, count*b1)
			for i := range src0 {
				src0[i] = byte((i*19 + 3) & 0xFF)
			}
			for i := range src1 {
				src1[i] = byte((i*41 + 9) & 0xFF)
			}
			dst := make([]byte, count*(b0+b1))
			want := make([]byte, count*(b0+b1))
			for r := range count {
				outOff := r * (b0 + b1)
				copy(want[outOff:outOff+b0], src0[r*b0:(r+1)*b0])
				copy(want[outOff+b0:outOff+b0+b1], src1[r*b1:(r+1)*b1])
			}
			if !neon.DispatchConcat2StridedBytesNEON(dst, src0, src1, count, b0, b1) {
				t.Fatalf("DispatchConcat2StridedBytesNEON returned false for b0=%d, b1=%d", b0, b1)
			}
			if !bytes.Equal(dst, want) {
				t.Fatalf("DispatchConcat2StridedBytesNEON mismatch for b0=%d, b1=%d", b0, b1)
			}
		}
	}
}

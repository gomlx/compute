// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simdarch

import (
	"syscall"
	"testing"
)

// TestSMEVectorLength: the SME kernels are written for a streaming vector length (SVL) of
// 512 bits, so SME is on only at that length. Linux starts a process at the largest SVL of
// at most 256 bits that the CPU has, or else at the smallest.
//
// Without SME hardware, run it under QEMU user mode with each length:
//
//	GOARCH=arm64 GOEXPERIMENT=simd go test -c -o simdarch.test ./internal/gobackend/simdarch
//	QEMU_CPU=max ./simdarch.test                                   # SVL 256 bits
//	QEMU_CPU=max,sme-default-vector-length=64 ./simdarch.test      # SVL 512 bits
//	QEMU_CPU=max,sme-default-vector-length=128 ./simdarch.test     # SVL 1024 bits
func TestSMEVectorLength(t *testing.T) {
	const prSMEGetVL, prSMEVLLenMask = 64, 0xffff
	r, _, errno := syscall.RawSyscall(syscall.SYS_PRCTL, prSMEGetVL, 0, 0)
	svlBytes := 0
	if errno == 0 {
		svlBytes = int(r & prSMEVLLenMask)
	}
	t.Logf("hasSME=%v, streaming vector length %d bits", arm64CPU.hasSME, 8*svlBytes)
	if arm64CPU.hasSME && svlBytes != 64 {
		t.Fatalf("SME is on at a streaming vector length of %d bits; the SME kernels need 512", 8*svlBytes)
	}
}

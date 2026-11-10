// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux && arm64 && goexperiment.simd

package simdarch

import (
	"encoding/binary"
	"os"
)

const (
	auxvAT_NULL   = 0
	auxvAT_HWCAP  = 16
	auxvAT_HWCAP2 = 26

	// Linux arm64 HWCAP bits (from arch/arm64/include/uapi/asm/hwcap.h).
	hwcapASIMD   = 1 << 1
	hwcapFPHP    = 1 << 9
	hwcapASIMDHP = 1 << 10
	hwcapASIMDDP = 1 << 20
	hwcapSVE     = 1 << 22

	// Linux arm64 HWCAP2 bits.
	hwcap2I8MM = 1 << 13
	hwcap2BF16 = 1 << 14
	hwcap2SME  = 1 << 23
)

func detectARM64Features() arm64Features {
	features := arm64Features{
		hasASIMD: true, // AdvSIMD is mandatory on ARMv8-A.
	}
	buf, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return features
	}
	var hwcap, hwcap2 uint64
	for len(buf) >= 16 {
		tag := binary.LittleEndian.Uint64(buf[0:8])
		val := binary.LittleEndian.Uint64(buf[8:16])
		buf = buf[16:]
		if tag == auxvAT_NULL {
			break
		}
		switch tag {
		case auxvAT_HWCAP:
			hwcap = val
		case auxvAT_HWCAP2:
			hwcap2 = val
		}
	}
	if hwcap != 0 {
		features.hasASIMD = (hwcap & hwcapASIMD) != 0
		features.hasFP16 = (hwcap&hwcapFPHP) != 0 && (hwcap&hwcapASIMDHP) != 0
		features.hasDotProd = (hwcap & hwcapASIMDDP) != 0
		features.hasSVE = (hwcap & hwcapSVE) != 0
	}
	if hwcap2 != 0 {
		features.hasI8MM = (hwcap2 & hwcap2I8MM) != 0
		features.hasBF16 = (hwcap2 & hwcap2BF16) != 0
		features.hasSME = (hwcap2 & hwcap2SME) != 0
	}
	return features
}

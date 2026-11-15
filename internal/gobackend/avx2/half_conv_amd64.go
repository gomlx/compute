// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX2Allowed {
		gobackend.SetHalfSliceConverters(gobackend.PriorityArch, gobackend.HalfSliceConverters{
			Float16ToFloat32:  avx2Float16ToFloat32Slice,
			BFloat16ToFloat32: avx2BFloat16ToFloat32Slice,
			Float32ToFloat16:  avx2Float32ToFloat16Slice,
			Float32ToBFloat16: avx2Float32ToBFloat16Slice,
		})
	}
}

//go:noescape
func float16ToFloat32AVX2Asm(dst, src unsafe.Pointer, n int)

//go:noescape
func float32ToFloat16AVX2Asm(dst, src unsafe.Pointer, n int)

func avx2Float16ToFloat32Slice(dst []float32, src []float16.Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	float16ToFloat32AVX2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), n)
}

func avx2Float32ToFloat16Slice(dst []float16.Float16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	float32ToFloat16AVX2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), n)
}

func avx2BFloat16ToFloat32Slice(dst []float32, src []bfloat16.BFloat16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	srcU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&src[0])), n)
	i := 0
	for ; i+32 <= n; i += 32 {
		v0 := bfloat16.BFloat16x16(archsimd.LoadUint16x16(srcU16[i:]))
		v1 := bfloat16.BFloat16x16(archsimd.LoadUint16x16(srcU16[i+16:]))
		lo0, hi0 := v0.ToFloat32()
		lo1, hi1 := v1.ToFloat32()
		lo0.Store(dst[i:])
		hi0.Store(dst[i+8:])
		lo1.Store(dst[i+16:])
		hi1.Store(dst[i+24:])
	}
	if i+16 <= n {
		v := bfloat16.BFloat16x16(archsimd.LoadUint16x16(srcU16[i:]))
		lo, hi := v.ToFloat32()
		lo.Store(dst[i:])
		hi.Store(dst[i+8:])
		i += 16
	}
	for ; i < n; i++ {
		dst[i] = src[i].Float32()
	}
}

func avx2Float32ToBFloat16Slice(dst []bfloat16.BFloat16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	dstU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&dst[0])), n)
	c7FFF := archsimd.BroadcastUint32x8(0x7FFF)
	c1 := archsimd.BroadcastUint32x8(1)
	i := 0
	for ; i+16 <= n; i += 16 {
		bits0 := archsimd.LoadFloat32x8(src[i:]).ToBits()
		bits1 := archsimd.LoadFloat32x8(src[i+8:]).ToBits()
		bias0 := c7FFF.Add(bits0.ShiftAllRight(16).And(c1))
		bias1 := c7FFF.Add(bits1.ShiftAllRight(16).And(c1))
		packed0 := bits0.Add(bias0).ShiftAllRight(16).TruncToUint16()
		packed1 := bits1.Add(bias1).ShiftAllRight(16).TruncToUint16()
		packed0.Store(dstU16[i:])
		packed1.Store(dstU16[i+8:])
	}
	if i+8 <= n {
		bits := archsimd.LoadFloat32x8(src[i:]).ToBits()
		bias := c7FFF.Add(bits.ShiftAllRight(16).And(c1))
		packed := bits.Add(bias).ShiftAllRight(16).TruncToUint16()
		packed.Store(dstU16[i:])
		i += 8
	}
	for ; i < n; i++ {
		dst[i].SetFloat32(src[i])
	}
}

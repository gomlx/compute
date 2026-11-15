// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd && go1.27

package avx512

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

func init() {
	if gobackend.IsAVX512Allowed {
		gobackend.SetHalfSliceConverters(gobackend.PriorityArch+1, gobackend.HalfSliceConverters{
			Float16ToFloat32:  avx512Float16ToFloat32Slice,
			BFloat16ToFloat32: avx512BFloat16ToFloat32Slice,
			Float32ToFloat16:  avx512Float32ToFloat16Slice,
			Float32ToBFloat16: avx512Float32ToBFloat16Slice,
		})
	}
}

//go:noescape
func float16ToFloat32AVX512Asm(dst, src unsafe.Pointer, n int)

//go:noescape
func float32ToFloat16AVX512Asm(dst, src unsafe.Pointer, n int)

func avx512Float16ToFloat32Slice(dst []float32, src []float16.Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	float16ToFloat32AVX512Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), n)
}

func avx512Float32ToFloat16Slice(dst []float16.Float16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	float32ToFloat16AVX512Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), n)
}

func avx512BFloat16ToFloat32Slice(dst []float32, src []bfloat16.BFloat16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	srcU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&src[0])), n)
	i := 0
	for ; i+64 <= n; i += 64 {
		v0 := bfloat16.BFloat16x32(archsimd.LoadUint16x32(srcU16[i:]))
		v1 := bfloat16.BFloat16x32(archsimd.LoadUint16x32(srcU16[i+32:]))
		lo0, hi0 := v0.ToFloat32()
		lo1, hi1 := v1.ToFloat32()
		lo0.Store(dst[i:])
		hi0.Store(dst[i+16:])
		lo1.Store(dst[i+32:])
		hi1.Store(dst[i+48:])
	}
	if i+32 <= n {
		v := bfloat16.BFloat16x32(archsimd.LoadUint16x32(srcU16[i:]))
		lo, hi := v.ToFloat32()
		lo.Store(dst[i:])
		hi.Store(dst[i+16:])
		i += 32
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

func avx512Float32ToBFloat16Slice(dst []bfloat16.BFloat16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	dstU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&dst[0])), n)
	c7FFF := archsimd.BroadcastUint32x16(0x7FFF)
	c1 := archsimd.BroadcastUint32x16(1)
	i := 0
	for ; i+32 <= n; i += 32 {
		bits0 := archsimd.LoadFloat32x16(src[i:]).ToBits()
		bits1 := archsimd.LoadFloat32x16(src[i+16:]).ToBits()
		bias0 := c7FFF.Add(bits0.ShiftAllRight(16).And(c1))
		bias1 := c7FFF.Add(bits1.ShiftAllRight(16).And(c1))
		packed0 := bits0.Add(bias0).ShiftAllRight(16).TruncToUint16()
		packed1 := bits1.Add(bias1).ShiftAllRight(16).TruncToUint16()
		packed0.Store(dstU16[i:])
		packed1.Store(dstU16[i+16:])
	}
	if i+16 <= n {
		bits := archsimd.LoadFloat32x16(src[i:]).ToBits()
		bias := c7FFF.Add(bits.ShiftAllRight(16).And(c1))
		packed := bits.Add(bias).ShiftAllRight(16).TruncToUint16()
		packed.Store(dstU16[i:])
		i += 16
	}
	if i+8 <= n {
		bits := archsimd.LoadFloat32x8(src[i:]).ToBits()
		bias := archsimd.BroadcastUint32x8(0x7FFF).Add(bits.ShiftAllRight(16).And(archsimd.BroadcastUint32x8(1)))
		packed := bits.Add(bias).ShiftAllRight(16).TruncToUint16()
		packed.Store(dstU16[i:])
		i += 8
	}
	for ; i < n; i++ {
		dst[i].SetFloat32(src[i])
	}
}

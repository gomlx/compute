// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package avx2

import (
	"math"
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

//go:noescape
func swigluFloat32AVX2Asm(in, out unsafe.Pointer, numRows, hiddenDim, vecWidth int)

// SwiGLUFloat32AVX2 computes SwiGLU(in) for float32 using handwritten AVX2 assembly:
//
//	out[m*hiddenDim + j] = SiLU(in[m*2*hiddenDim + j]) * in[m*2*hiddenDim + hiddenDim + j]
func SwiGLUFloat32AVX2(in, out []float32, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	_ = in[numRows*2*hiddenDim-1]
	_ = out[numRows*hiddenDim-1]
	vecWidth := hiddenDim &^ 7
	if vecWidth > 0 {
		swigluFloat32AVX2Asm(
			unsafe.Pointer(&in[0]),
			unsafe.Pointer(&out[0]),
			numRows, hiddenDim, vecWidth,
		)
	}
	if vecWidth < hiddenDim {
		rem := hiddenDim - vecWidth
		var bufIn [16]float32
		var bufOut [8]float32
		for m := range numRows {
			inRowOff := m * 2 * hiddenDim
			outRowOff := m * hiddenDim
			copy(bufIn[:rem], in[inRowOff+vecWidth:inRowOff+hiddenDim])
			copy(bufIn[8:8+rem], in[inRowOff+hiddenDim+vecWidth:inRowOff+2*hiddenDim])
			swigluFloat32AVX2Asm(
				unsafe.Pointer(&bufIn[0]),
				unsafe.Pointer(&bufOut[0]),
				1, 8, 8,
			)
			copy(out[outRowOff+vecWidth:outRowOff+hiddenDim], bufOut[:rem])
		}
	}
}

// SwiGLUAVX2Float64 computes SwiGLU(in) for float64 using AVX2 (archsimd.Float64x4).
func SwiGLUAVX2Float64(in, out []float64, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	for m := range numRows {
		inOff := m * 2 * hiddenDim
		valOff := inOff + hiddenDim
		outOff := m * hiddenDim
		j := 0
		for ; j+8 <= hiddenDim; j += 8 {
			g0 := archsimd.LoadFloat64x4(in[inOff+j : inOff+j+4])
			g1 := archsimd.LoadFloat64x4(in[inOff+j+4 : inOff+j+8])
			v0 := archsimd.LoadFloat64x4(in[valOff+j : valOff+j+4])
			v1 := archsimd.LoadFloat64x4(in[valOff+j+4 : valOff+j+8])
			res0 := g0.Mul(v0).Div(expF64One.Add(exp256Float64(g0.Neg())))
			res1 := g1.Mul(v1).Div(expF64One.Add(exp256Float64(g1.Neg())))
			res0.Store(out[outOff+j : outOff+j+4])
			res1.Store(out[outOff+j+4 : outOff+j+8])
		}
		for ; j+4 <= hiddenDim; j += 4 {
			g0 := archsimd.LoadFloat64x4(in[inOff+j : inOff+j+4])
			v0 := archsimd.LoadFloat64x4(in[valOff+j : valOff+j+4])
			res0 := g0.Mul(v0).Div(expF64One.Add(exp256Float64(g0.Neg())))
			res0.Store(out[outOff+j : outOff+j+4])
		}
		for ; j < hiddenDim; j++ {
			g := in[inOff+j]
			v := in[valOff+j]
			out[outOff+j] = (g * v) / (1.0 + math.Exp(-g))
		}
	}
}

const swigluHalfChunk = 256

// SwiGLUAVX2BF16 computes SwiGLU(in) for BFloat16 using AVX2 float32 SwiGLU assembly.
func SwiGLUAVX2BF16(in, out []bfloat16.BFloat16, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	var bufIn [2 * swigluHalfChunk]float32
	var bufOut [swigluHalfChunk]float32
	for m := range numRows {
		inOff := m * 2 * hiddenDim
		valOff := inOff + hiddenDim
		outOff := m * hiddenDim
		for j := 0; j < hiddenDim; j += swigluHalfChunk {
			c := min(swigluHalfChunk, hiddenDim-j)
			for k := range c {
				bufIn[k] = in[inOff+j+k].Float32()
				bufIn[c+k] = in[valOff+j+k].Float32()
			}
			SwiGLUFloat32AVX2(bufIn[:2*c], bufOut[:c], 1, c)
			for k := range c {
				out[outOff+j+k] = bfloat16.FromFloat32(bufOut[k])
			}
		}
	}
}

// SwiGLUAVX2F16 computes SwiGLU(in) for Float16 using AVX2 float32 SwiGLU assembly.
func SwiGLUAVX2F16(in, out []float16.Float16, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	var bufIn [2 * swigluHalfChunk]float32
	var bufOut [swigluHalfChunk]float32
	for m := range numRows {
		inOff := m * 2 * hiddenDim
		valOff := inOff + hiddenDim
		outOff := m * hiddenDim
		for j := 0; j < hiddenDim; j += swigluHalfChunk {
			c := min(swigluHalfChunk, hiddenDim-j)
			for k := range c {
				bufIn[k] = in[inOff+j+k].Float32()
				bufIn[c+k] = in[valOff+j+k].Float32()
			}
			SwiGLUFloat32AVX2(bufIn[:2*c], bufOut[:c], 1, c)
			for k := range c {
				out[outOff+j+k] = float16.FromFloat32(bufOut[k])
			}
		}
	}
}

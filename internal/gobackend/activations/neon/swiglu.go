// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"math"
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
)

//go:noescape
func swigluFloat32NEONAsm(in, out unsafe.Pointer, numRows, hiddenDim, vecWidth int)

// SwiGLUFloat32NEON computes SwiGLU(in) for float32 using ARM64 NEON assembly:
//
//	out[m*hiddenDim + j] = SiLU(in[m*2*hiddenDim + j]) * in[m*2*hiddenDim + hiddenDim + j]
func SwiGLUFloat32NEON(in, out []float32, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	_ = in[numRows*2*hiddenDim-1]
	_ = out[numRows*hiddenDim-1]
	vecWidth := hiddenDim &^ 3
	if vecWidth > 0 {
		swigluFloat32NEONAsm(
			unsafe.Pointer(&in[0]),
			unsafe.Pointer(&out[0]),
			numRows, hiddenDim, vecWidth,
		)
	}
	if vecWidth < hiddenDim {
		rem := hiddenDim - vecWidth
		var bufIn [8]float32
		var bufOut [4]float32
		for m := range numRows {
			inRowOff := m * 2 * hiddenDim
			outRowOff := m * hiddenDim
			copy(bufIn[:rem], in[inRowOff+vecWidth:inRowOff+hiddenDim])
			copy(bufIn[4:4+rem], in[inRowOff+hiddenDim+vecWidth:inRowOff+2*hiddenDim])
			swigluFloat32NEONAsm(
				unsafe.Pointer(&bufIn[0]),
				unsafe.Pointer(&bufOut[0]),
				1, 4, 4,
			)
			copy(out[outRowOff+vecWidth:outRowOff+hiddenDim], bufOut[:rem])
		}
	}
}

// SwiGLUNEONFloat64 computes SwiGLU(in) for float64 using ARM64 NEON.
func SwiGLUNEONFloat64(in, out []float64, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	for m := range numRows {
		inOff := m * 2 * hiddenDim
		valOff := inOff + hiddenDim
		outOff := m * hiddenDim
		j := 0
		for ; j+4 <= hiddenDim; j += 4 {
			g0 := archsimd.LoadFloat64x2(in[inOff+j : inOff+j+2])
			g1 := archsimd.LoadFloat64x2(in[inOff+j+2 : inOff+j+4])
			v0 := archsimd.LoadFloat64x2(in[valOff+j : valOff+j+2])
			v1 := archsimd.LoadFloat64x2(in[valOff+j+2 : valOff+j+4])
			res0 := g0.Mul(v0).Div(expF64One.Add(exp128Float64(g0.Neg())))
			res1 := g1.Mul(v1).Div(expF64One.Add(exp128Float64(g1.Neg())))
			res0.Store(out[outOff+j : outOff+j+2])
			res1.Store(out[outOff+j+2 : outOff+j+4])
		}
		for ; j+2 <= hiddenDim; j += 2 {
			g0 := archsimd.LoadFloat64x2(in[inOff+j : inOff+j+2])
			v0 := archsimd.LoadFloat64x2(in[valOff+j : valOff+j+2])
			res0 := g0.Mul(v0).Div(expF64One.Add(exp128Float64(g0.Neg())))
			res0.Store(out[outOff+j : outOff+j+2])
		}
		if j < hiddenDim {
			g := in[inOff+j]
			v := in[valOff+j]
			out[outOff+j] = (g * v) / (1.0 + math.Exp(-g))
		}
	}
}

// SwiGLUNEONBF16 computes SwiGLU(in) for BFloat16 using ARM64 NEON conversion and float32 SwiGLU assembly.
func SwiGLUNEONBF16(in, out []bfloat16.BFloat16, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	var bufIn [2 * halfChunk]float32
	var bufOut [halfChunk]float32
	for m := range numRows {
		inOff := m * 2 * hiddenDim
		valOff := inOff + hiddenDim
		outOff := m * hiddenDim
		for j := 0; j < hiddenDim; j += halfChunk {
			c := min(halfChunk, hiddenDim-j)
			bfloat16.ToFloat32SliceNEON(bufIn[:c], in[inOff+j:inOff+j+c])
			bfloat16.ToFloat32SliceNEON(bufIn[c:2*c], in[valOff+j:valOff+j+c])
			SwiGLUFloat32NEON(bufIn[:2*c], bufOut[:c], 1, c)
			bfloat16.FromFloat32SliceNEON(out[outOff+j:outOff+j+c], bufOut[:c])
		}
	}
}

// SwiGLUNEONF16 computes SwiGLU(in) for Float16 using ARM64 NEON conversion and float32 SwiGLU assembly.
func SwiGLUNEONF16(in, out []float16.Float16, numRows, hiddenDim int) {
	if numRows <= 0 || hiddenDim <= 0 || len(in) == 0 || len(out) == 0 {
		return
	}
	var bufIn [2 * halfChunk]float32
	var bufOut [halfChunk]float32
	for m := range numRows {
		inOff := m * 2 * hiddenDim
		valOff := inOff + hiddenDim
		outOff := m * hiddenDim
		for j := 0; j < hiddenDim; j += halfChunk {
			c := min(halfChunk, hiddenDim-j)
			float16.ToFloat32SliceNEON(bufIn[:c], in[inOff+j:inOff+j+c])
			float16.ToFloat32SliceNEON(bufIn[c:2*c], in[valOff+j:valOff+j+c])
			SwiGLUFloat32NEON(bufIn[:2*c], bufOut[:c], 1, c)
			float16.FromFloat32SliceNEON(out[outOff+j:outOff+j+c], bufOut[:c])
		}
	}
}

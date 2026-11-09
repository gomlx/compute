// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package neon

import (
	"math"
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
)

const PriorityNEON = gobackend.PriorityArch

var (
	// NEONSimpleMetadata holds metadata for lightweight element-wise ARM64 NEON kernels.
	NEONSimpleMetadata = activations.KernelMetadata{MinParallelChunk: 32768}

	// NEONTranscendentalMetadata holds metadata for ARM64 NEON assembly activation kernels.
	NEONTranscendentalMetadata = activations.KernelMetadata{MinParallelChunk: 8192}
)

var (
	// Float32 constants for Tanh
	expMaxLog   archsimd.Float32x4
	expMinLog   archsimd.Float32x4
	expLog2E    archsimd.Float32x4
	expHalf     archsimd.Float32x4
	expLn2Hi    archsimd.Float32x4
	expLn2Lo    archsimd.Float32x4
	expP7       archsimd.Float32x4
	expP6       archsimd.Float32x4
	expP5       archsimd.Float32x4
	expP4       archsimd.Float32x4
	expP3       archsimd.Float32x4
	expP2       archsimd.Float32x4
	expOne      archsimd.Float32x4
	exp127      archsimd.Uint32x4
	tanhNine    archsimd.Float32x4
	tanhNegNine archsimd.Float32x4
	tanhTwo     archsimd.Float32x4
	tanhOne     archsimd.Float32x4

	// Float64 constants
	expF64MaxLog    archsimd.Float64x2
	expF64MinLog    archsimd.Float64x2
	expF64Log2E     archsimd.Float64x2
	expF64Ln2Hi     archsimd.Float64x2
	expF64Ln2Lo     archsimd.Float64x2
	expF64One       archsimd.Float64x2
	expF641023      archsimd.Uint64x2
	expF64Magic     archsimd.Float64x2
	expF64MagicBits archsimd.Uint64x2

	expF64C11 archsimd.Float64x2
	expF64C10 archsimd.Float64x2
	expF64C9  archsimd.Float64x2
	expF64C8  archsimd.Float64x2
	expF64C7  archsimd.Float64x2
	expF64C6  archsimd.Float64x2
	expF64C5  archsimd.Float64x2
	expF64C4  archsimd.Float64x2
	expF64C3  archsimd.Float64x2
	expF64C2  archsimd.Float64x2

	erfF64P        archsimd.Float64x2
	erfF64A1       archsimd.Float64x2
	erfF64A2       archsimd.Float64x2
	erfF64A3       archsimd.Float64x2
	erfF64A4       archsimd.Float64x2
	erfF64A5       archsimd.Float64x2
	erfF64One      archsimd.Float64x2
	erfF64Half     archsimd.Float64x2
	erfF64Rsqrt2   archsimd.Float64x2
	erfF64SignMask archsimd.Uint64x2

	geluF64Half      archsimd.Float64x2
	geluF64One       archsimd.Float64x2
	geluF64Sqrt2ByPi archsimd.Float64x2
	geluF64C         archsimd.Float64x2
	tanhF6419        archsimd.Float64x2
	tanhF64Neg19     archsimd.Float64x2
	tanhF64Two       archsimd.Float64x2
	tanhF64One       archsimd.Float64x2
)

func init() {
	if !gobackend.IsNEONAllowed {
		return
	}
	initNEONConstants()
	registerNEON()
}

func initNEONConstants() {
	expMaxLog = archsimd.BroadcastFloat32x4(88.02969187150841)
	expMinLog = archsimd.BroadcastFloat32x4(-88.02969187150841)
	expLog2E = archsimd.BroadcastFloat32x4(1.44269504088896341)
	expHalf = archsimd.BroadcastFloat32x4(0.5)
	expLn2Hi = archsimd.BroadcastFloat32x4(0.693359375)
	expLn2Lo = archsimd.BroadcastFloat32x4(-2.12194440e-4)
	expP7 = archsimd.BroadcastFloat32x4(1.9875691500e-4)
	expP6 = archsimd.BroadcastFloat32x4(1.3981999507e-3)
	expP5 = archsimd.BroadcastFloat32x4(8.3334519073e-3)
	expP4 = archsimd.BroadcastFloat32x4(4.1665795894e-2)
	expP3 = archsimd.BroadcastFloat32x4(1.6666665459e-1)
	expP2 = archsimd.BroadcastFloat32x4(5.0000001201e-1)
	expOne = archsimd.BroadcastFloat32x4(1.0)
	exp127 = archsimd.BroadcastUint32x4(127)

	tanhNine = archsimd.BroadcastFloat32x4(9.0)
	tanhNegNine = archsimd.BroadcastFloat32x4(-9.0)
	tanhTwo = archsimd.BroadcastFloat32x4(2.0)
	tanhOne = archsimd.BroadcastFloat32x4(1.0)

	// Float64
	expF64MaxLog = archsimd.BroadcastFloat64x2(709.7827128933840)
	expF64MinLog = archsimd.BroadcastFloat64x2(-708.3964185322641)
	expF64Log2E = archsimd.BroadcastFloat64x2(1.442695040888963407359924681001892137)
	expF64Ln2Hi = archsimd.BroadcastFloat64x2(0.693147180559945309417232121458176568)
	expF64Ln2Lo = archsimd.BroadcastFloat64x2(2.31904681384629955841777123493867e-17)
	expF64One = archsimd.BroadcastFloat64x2(1.0)
	expF641023 = archsimd.BroadcastUint64x2(1023)
	expF64Magic = archsimd.BroadcastFloat64x2(6755399441055744.0)
	expF64MagicBits = archsimd.BroadcastUint64x2(0x4338000000000000)

	expF64C11 = archsimd.BroadcastFloat64x2(2.505210838544172e-8)
	expF64C10 = archsimd.BroadcastFloat64x2(2.755731922398589e-7)
	expF64C9 = archsimd.BroadcastFloat64x2(2.755731922398589e-6)
	expF64C8 = archsimd.BroadcastFloat64x2(2.480158730158730e-5)
	expF64C7 = archsimd.BroadcastFloat64x2(1.984126984126984e-4)
	expF64C6 = archsimd.BroadcastFloat64x2(1.388888888888889e-3)
	expF64C5 = archsimd.BroadcastFloat64x2(8.333333333333333e-3)
	expF64C4 = archsimd.BroadcastFloat64x2(4.166666666666667e-2)
	expF64C3 = archsimd.BroadcastFloat64x2(1.666666666666667e-1)
	expF64C2 = archsimd.BroadcastFloat64x2(5.000000000000000e-1)

	erfF64P = archsimd.BroadcastFloat64x2(0.3275911)
	erfF64A1 = archsimd.BroadcastFloat64x2(0.254829592)
	erfF64A2 = archsimd.BroadcastFloat64x2(-0.284496736)
	erfF64A3 = archsimd.BroadcastFloat64x2(1.421413741)
	erfF64A4 = archsimd.BroadcastFloat64x2(-1.453152027)
	erfF64A5 = archsimd.BroadcastFloat64x2(1.061405429)
	erfF64One = archsimd.BroadcastFloat64x2(1.0)
	erfF64Half = archsimd.BroadcastFloat64x2(0.5)
	erfF64Rsqrt2 = archsimd.BroadcastFloat64x2(1.0 / math.Sqrt2)
	erfF64SignMask = archsimd.BroadcastUint64x2(0x8000000000000000)

	geluF64Half = archsimd.BroadcastFloat64x2(0.5)
	geluF64One = archsimd.BroadcastFloat64x2(1.0)
	geluF64Sqrt2ByPi = archsimd.BroadcastFloat64x2(math.Sqrt(2.0 / math.Pi))
	geluF64C = archsimd.BroadcastFloat64x2(0.044715)

	tanhF6419 = archsimd.BroadcastFloat64x2(19.0)
	tanhF64Neg19 = archsimd.BroadcastFloat64x2(-19.0)
	tanhF64Two = archsimd.BroadcastFloat64x2(2.0)
	tanhF64One = archsimd.BroadcastFloat64x2(1.0)
}

func registerNEON() {
	// Float32
	activations.RegisterKernel[float32]("neon:relu", compute.ActivationRelu, ReluFloat32NEON, PriorityNEON, &NEONSimpleMetadata)
	activations.RegisterKernel[float32]("neon:hardswish", compute.ActivationHardSwish, HardSwishFloat32NEON, PriorityNEON, &NEONSimpleMetadata)
	activations.RegisterKernel[float32]("neon:silu", compute.ActivationSilu, SiluFloat32NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[float32]("neon:gelu", compute.ActivationGelu, GeluExactFloat32NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[float32]("neon:geluapprox", compute.ActivationGeluApproximate, GeluApproxFloat32NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterBiasTile[float32]("neon:silu", compute.ActivationSilu, SiluBiasTileNEON, PriorityNEON)
	activations.RegisterBiasTile[float32]("neon:gelu", compute.ActivationGelu, GeluExactBiasTileNEON, PriorityNEON)
	activations.RegisterBiasTile[float32]("neon:geluapprox", compute.ActivationGeluApproximate, GeluApproxBiasTileNEON, PriorityNEON)
	activations.RegisterKernel[float32]("neon:tanh", compute.ActivationTanh, TanhFloat32NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterVJPKernel[float32]("neon:vjp_silu", compute.ActivationSilu, SiluVJPFloat32NEON, PriorityNEON, &NEONTranscendentalMetadata)

	// Float64
	activations.RegisterKernel[float64]("neon:silu", compute.ActivationSilu, SiluNEONFloat64, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[float64]("neon:gelu", compute.ActivationGelu, GeluExactNEONFloat64, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[float64]("neon:geluapprox", compute.ActivationGeluApproximate, GeluApproxNEONFloat64, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterVJPKernel[float64]("neon:vjp_silu", compute.ActivationSilu, SiluVJPNEONFloat64, PriorityNEON, &NEONTranscendentalMetadata)

	// BFloat16
	activations.Register[bfloat16.BFloat16]("neon:relu", compute.ActivationRelu, reluBF16NEON, PriorityNEON, &NEONSimpleMetadata)
	activations.Register[bfloat16.BFloat16]("neon:hardswish", compute.ActivationHardSwish, hardSwishBF16NEON, PriorityNEON, &NEONSimpleMetadata)
	activations.Register[bfloat16.BFloat16]("neon:silu", compute.ActivationSilu, siluBF16NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[bfloat16.BFloat16]("neon:silu", compute.ActivationSilu, SiluNEONBF16, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[bfloat16.BFloat16]("neon:gelu", compute.ActivationGelu, GeluExactNEONBF16, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[bfloat16.BFloat16]("neon:geluapprox", compute.ActivationGeluApproximate, GeluApproxNEONBF16, PriorityNEON, &NEONTranscendentalMetadata)
	activations.Register[bfloat16.BFloat16]("neon:tanh", compute.ActivationTanh, tanhBF16NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterVJPKernel[bfloat16.BFloat16]("neon:vjp_silu", compute.ActivationSilu, SiluVJPNEONBF16, PriorityNEON, &NEONTranscendentalMetadata)

	// Float16
	activations.Register[float16.Float16]("neon:relu", compute.ActivationRelu, reluF16NEON, PriorityNEON, &NEONSimpleMetadata)
	activations.Register[float16.Float16]("neon:hardswish", compute.ActivationHardSwish, hardSwishF16NEON, PriorityNEON, &NEONSimpleMetadata)
	activations.Register[float16.Float16]("neon:silu", compute.ActivationSilu, siluF16NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[float16.Float16]("neon:silu", compute.ActivationSilu, SiluNEONF16, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[float16.Float16]("neon:gelu", compute.ActivationGelu, GeluExactNEONF16, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterKernel[float16.Float16]("neon:geluapprox", compute.ActivationGeluApproximate, GeluApproxNEONF16, PriorityNEON, &NEONTranscendentalMetadata)
	activations.Register[float16.Float16]("neon:tanh", compute.ActivationTanh, tanhF16NEON, PriorityNEON, &NEONTranscendentalMetadata)
	activations.RegisterVJPKernel[float16.Float16]("neon:vjp_silu", compute.ActivationSilu, SiluVJPNEONF16, PriorityNEON, &NEONTranscendentalMetadata)
}

//go:noescape
func geluExactFloat32NEONAsm(in, out unsafe.Pointer, n int)

//go:noescape
func geluApproxFloat32NEONAsm(in, out unsafe.Pointer, n int)

//go:noescape
func geluExactBiasTileFloat32NEONAsm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)

//go:noescape
func geluApproxBiasTileFloat32NEONAsm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)

//go:noescape
func siluFloat32NEONAsm(in, out unsafe.Pointer, n int)

//go:noescape
func siluBiasTileFloat32NEONAsm(out, bias unsafe.Pointer, rowStrideBytes, numRows, width int)

//go:noescape
func siluVJPFloat32NEONAsm(x, dOut, dx unsafe.Pointer, n int)

// ReluFloat32NEON computes ReLU(x) = max(x, 0) using ARM64 NEON.
func ReluFloat32NEON(in, out []float32) {
	vZero := archsimd.BroadcastFloat32x4(0)
	i := 0
	for ; i+8 <= len(in); i += 8 {
		v0 := archsimd.LoadFloat32x4(in[i : i+4])
		v1 := archsimd.LoadFloat32x4(in[i+4 : i+8])
		v0.Max(vZero).Store(out[i : i+4])
		v1.Max(vZero).Store(out[i+4 : i+8])
	}
	for ; i+4 <= len(in); i += 4 {
		v := archsimd.LoadFloat32x4(in[i : i+4])
		v.Max(vZero).Store(out[i : i+4])
	}
	for ; i < len(in); i++ {
		if in[i] < 0 {
			out[i] = 0
		} else {
			out[i] = in[i]
		}
	}
}

// HardSwishFloat32NEON computes HardSwish(x) = x * clamp(x/6 + 0.5, 0, 1) using ARM64 NEON.
func HardSwishFloat32NEON(in, out []float32) {
	vZero := archsimd.BroadcastFloat32x4(0)
	vOne := archsimd.BroadcastFloat32x4(1)
	vOneSixth := archsimd.BroadcastFloat32x4(1.0 / 6.0)
	vHalf := archsimd.BroadcastFloat32x4(0.5)

	i := 0
	for ; i+4 <= len(in); i += 4 {
		v := archsimd.LoadFloat32x4(in[i : i+4])
		scaled := v.MulAdd(vOneSixth, vHalf)
		clamped := scaled.Max(vZero).Min(vOne)
		v.Mul(clamped).Store(out[i : i+4])
	}
	for ; i < len(in); i++ {
		x := in[i]
		shapeX := min(max(x*(1.0/6.0)+0.5, 0), 1)
		out[i] = x * shapeX
	}
}

// SiluFloat32NEON computes SiLU(x) = x / (1 + exp(-x)) using ARM64 NEON assembly.
func SiluFloat32NEON(in, out []float32) {
	n := len(in)
	if n == 0 {
		return
	}
	_ = out[n-1]
	vecN := n &^ 3
	if vecN > 0 {
		siluFloat32NEONAsm(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), vecN)
	}
	if vecN < n {
		rem := n - vecN
		var bufIn, bufOut [4]float32
		copy(bufIn[:rem], in[vecN:])
		siluFloat32NEONAsm(unsafe.Pointer(&bufIn[0]), unsafe.Pointer(&bufOut[0]), 4)
		copy(out[vecN:], bufOut[:rem])
	}
}

// SiluBiasTileNEON computes fused Bias + SiLU on a 2D tile using ARM64 NEON assembly.
func SiluBiasTileNEON(out []float32, rowStride, numRows, width int, bias []float32) {
	if numRows <= 0 || width <= 0 || len(out) == 0 {
		return
	}
	_ = bias[width-1]
	_ = out[(numRows-1)*rowStride+width-1]
	vecWidth := width &^ 3
	if vecWidth > 0 {
		siluBiasTileFloat32NEONAsm(
			unsafe.Pointer(&out[0]),
			unsafe.Pointer(&bias[0]),
			rowStride*4, numRows, vecWidth,
		)
	}
	if vecWidth < width {
		rem := width - vecWidth
		var bufIn, bufBias [4]float32
		copy(bufBias[:rem], bias[vecWidth:width])
		for r := range numRows {
			rowOff := r*rowStride + vecWidth
			copy(bufIn[:rem], out[rowOff:rowOff+rem])
			siluBiasTileFloat32NEONAsm(
				unsafe.Pointer(&bufIn[0]),
				unsafe.Pointer(&bufBias[0]),
				16, 1, 4,
			)
			copy(out[rowOff:rowOff+rem], bufIn[:rem])
		}
	}
}

// SiluVJPFloat32NEON computes the backward pass (VJP) of SiLU for float32 using ARM64 NEON assembly.
func SiluVJPFloat32NEON(y, x, dOutput, dx []float32) {
	_ = y
	n := len(dOutput)
	if n == 0 {
		return
	}
	_ = x[n-1]
	_ = dx[n-1]
	vecN := n &^ 3
	if vecN > 0 {
		siluVJPFloat32NEONAsm(
			unsafe.Pointer(&x[0]),
			unsafe.Pointer(&dOutput[0]),
			unsafe.Pointer(&dx[0]),
			vecN,
		)
	}
	if vecN < n {
		rem := n - vecN
		var bufX, bufDOut, bufDx [4]float32
		copy(bufX[:rem], x[vecN:])
		copy(bufDOut[:rem], dOutput[vecN:])
		siluVJPFloat32NEONAsm(
			unsafe.Pointer(&bufX[0]),
			unsafe.Pointer(&bufDOut[0]),
			unsafe.Pointer(&bufDx[0]),
			4,
		)
		copy(dx[vecN:], bufDx[:rem])
	}
}

// GeluExactFloat32NEON computes exact GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) using ARM64 NEON assembly.
func GeluExactFloat32NEON(in, out []float32) {
	n := len(in)
	if n == 0 {
		return
	}
	_ = out[n-1]
	vecN := n &^ 3
	if vecN > 0 {
		geluExactFloat32NEONAsm(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), vecN)
	}
	if vecN < n {
		rem := n - vecN
		var bufIn, bufOut [4]float32
		copy(bufIn[:rem], in[vecN:])
		geluExactFloat32NEONAsm(unsafe.Pointer(&bufIn[0]), unsafe.Pointer(&bufOut[0]), 4)
		copy(out[vecN:], bufOut[:rem])
	}
}

// GeluApproxFloat32NEON computes approximate GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// using ARM64 NEON assembly.
func GeluApproxFloat32NEON(in, out []float32) {
	n := len(in)
	if n == 0 {
		return
	}
	_ = out[n-1]
	vecN := n &^ 3
	if vecN > 0 {
		geluApproxFloat32NEONAsm(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), vecN)
	}
	if vecN < n {
		rem := n - vecN
		var bufIn, bufOut [4]float32
		copy(bufIn[:rem], in[vecN:])
		geluApproxFloat32NEONAsm(unsafe.Pointer(&bufIn[0]), unsafe.Pointer(&bufOut[0]), 4)
		copy(out[vecN:], bufOut[:rem])
	}
}

// GeluExactBiasTileNEON computes fused Bias + Exact GELU on a 2D tile using ARM64 NEON assembly.
func GeluExactBiasTileNEON(out []float32, rowStride, numRows, width int, bias []float32) {
	if numRows <= 0 || width <= 0 || len(out) == 0 {
		return
	}
	_ = bias[width-1]
	_ = out[(numRows-1)*rowStride+width-1]
	vecWidth := width &^ 3
	if vecWidth > 0 {
		geluExactBiasTileFloat32NEONAsm(
			unsafe.Pointer(&out[0]),
			unsafe.Pointer(&bias[0]),
			rowStride*4, numRows, vecWidth,
		)
	}
	if vecWidth < width {
		rem := width - vecWidth
		var bufIn, bufBias [4]float32
		copy(bufBias[:rem], bias[vecWidth:width])
		for r := range numRows {
			rowOff := r*rowStride + vecWidth
			copy(bufIn[:rem], out[rowOff:rowOff+rem])
			geluExactBiasTileFloat32NEONAsm(
				unsafe.Pointer(&bufIn[0]),
				unsafe.Pointer(&bufBias[0]),
				16, 1, 4,
			)
			copy(out[rowOff:rowOff+rem], bufIn[:rem])
		}
	}
}

// GeluApproxBiasTileNEON computes fused Bias + Approximate GELU on a 2D tile using ARM64 NEON assembly.
func GeluApproxBiasTileNEON(out []float32, rowStride, numRows, width int, bias []float32) {
	if numRows <= 0 || width <= 0 || len(out) == 0 {
		return
	}
	_ = bias[width-1]
	_ = out[(numRows-1)*rowStride+width-1]
	vecWidth := width &^ 3
	if vecWidth > 0 {
		geluApproxBiasTileFloat32NEONAsm(
			unsafe.Pointer(&out[0]),
			unsafe.Pointer(&bias[0]),
			rowStride*4, numRows, vecWidth,
		)
	}
	if vecWidth < width {
		rem := width - vecWidth
		var bufIn, bufBias [4]float32
		copy(bufBias[:rem], bias[vecWidth:width])
		for r := range numRows {
			rowOff := r*rowStride + vecWidth
			copy(bufIn[:rem], out[rowOff:rowOff+rem])
			geluApproxBiasTileFloat32NEONAsm(
				unsafe.Pointer(&bufIn[0]),
				unsafe.Pointer(&bufBias[0]),
				16, 1, 4,
			)
			copy(out[rowOff:rowOff+rem], bufIn[:rem])
		}
	}
}

func exp128(x archsimd.Float32x4) archsimd.Float32x4 {
	xClamped := x.Max(expMinLog).Min(expMaxLog)
	z := xClamped.MulAdd(expLog2E, expHalf).Floor()
	g := xClamped.Sub(z.Mul(expLn2Hi)).Sub(z.Mul(expLn2Lo))

	n := z.ConvertToInt32().ToBits().Add(exp127).ShiftAllLeft(23).BitsToFloat32()

	poly := expP7.MulAdd(g, expP6)
	poly = poly.MulAdd(g, expP5)
	poly = poly.MulAdd(g, expP4)
	poly = poly.MulAdd(g, expP3)
	poly = poly.MulAdd(g, expP2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(expOne)

	return n.Mul(poly)
}

func tanh128(x archsimd.Float32x4) archsimd.Float32x4 {
	xClamped := x.Max(tanhNegNine).Min(tanhNine)
	twoX := xClamped.Mul(tanhTwo)
	exp2x := exp128(twoX)
	num := exp2x.Sub(tanhOne)
	den := exp2x.Add(tanhOne)
	return num.Div(den)
}

// TanhFloat32NEON computes tanh(x) for float32 using ARM64 NEON.
func TanhFloat32NEON(in, out []float32) {
	i := 0
	for ; i+4 <= len(in); i += 4 {
		v := archsimd.LoadFloat32x4(in[i : i+4])
		tanh128(v).Store(out[i : i+4])
	}
	for ; i < len(in); i++ {
		out[i] = float32(math.Tanh(float64(in[i])))
	}
}

// -----------------------------------------------------------------------------
// Half-Precision (BFloat16 & Float16) Operations (ARM64 NEON)
// -----------------------------------------------------------------------------

const halfChunk = 64

func reluBF16NEON(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		chunk := data[i:min(i+halfChunk, len(data))]
		bfloat16.ToFloat32SliceNEON(buf[:len(chunk)], chunk)
		ReluFloat32NEON(buf[:len(chunk)], buf[:len(chunk)])
		bfloat16.FromFloat32SliceNEON(chunk, buf[:len(chunk)])
	}
}

func hardSwishBF16NEON(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		chunk := data[i:min(i+halfChunk, len(data))]
		bfloat16.ToFloat32SliceNEON(buf[:len(chunk)], chunk)
		HardSwishFloat32NEON(buf[:len(chunk)], buf[:len(chunk)])
		bfloat16.FromFloat32SliceNEON(chunk, buf[:len(chunk)])
	}
}

func siluBF16NEON(data []bfloat16.BFloat16) {
	SiluNEONBF16(data, data)
}

// SiluNEONBF16 computes SiLU for BFloat16 using ARM64 NEON assembly conversion and float32 SiLU assembly.
func SiluNEONBF16(in, out []bfloat16.BFloat16) {
	var bufIn, bufOut [halfChunk]float32
	for i := 0; i < len(in); i += halfChunk {
		end := min(i+halfChunk, len(in))
		c := end - i
		bfloat16.ToFloat32SliceNEON(bufIn[:c], in[i:end])
		SiluFloat32NEON(bufIn[:c], bufOut[:c])
		bfloat16.FromFloat32SliceNEON(out[i:end], bufOut[:c])
	}
}

// SiluVJPNEONBF16 computes the backward pass (VJP) of SiLU for BFloat16 using ARM64 NEON assembly.
func SiluVJPNEONBF16(y, x, dOutput, dx []bfloat16.BFloat16) {
	_ = y
	var bufX, bufDOut, bufDx [halfChunk]float32
	for i := 0; i < len(dOutput); i += halfChunk {
		end := min(i+halfChunk, len(dOutput))
		c := end - i
		bfloat16.ToFloat32SliceNEON(bufX[:c], x[i:end])
		bfloat16.ToFloat32SliceNEON(bufDOut[:c], dOutput[i:end])
		SiluVJPFloat32NEON(nil, bufX[:c], bufDOut[:c], bufDx[:c])
		bfloat16.FromFloat32SliceNEON(dx[i:end], bufDx[:c])
	}
}

// GeluExactNEONBF16 computes exact GELU for BFloat16 using ARM64 NEON assembly.
func GeluExactNEONBF16(in, out []bfloat16.BFloat16) {
	var bufIn, bufOut [halfChunk]float32
	for i := 0; i < len(in); i += halfChunk {
		end := min(i+halfChunk, len(in))
		c := end - i
		bfloat16.ToFloat32SliceNEON(bufIn[:c], in[i:end])
		GeluExactFloat32NEON(bufIn[:c], bufOut[:c])
		bfloat16.FromFloat32SliceNEON(out[i:end], bufOut[:c])
	}
}

// GeluApproxNEONBF16 computes approximate GELU for BFloat16 using ARM64 NEON assembly.
func GeluApproxNEONBF16(in, out []bfloat16.BFloat16) {
	var bufIn, bufOut [halfChunk]float32
	for i := 0; i < len(in); i += halfChunk {
		end := min(i+halfChunk, len(in))
		c := end - i
		bfloat16.ToFloat32SliceNEON(bufIn[:c], in[i:end])
		GeluApproxFloat32NEON(bufIn[:c], bufOut[:c])
		bfloat16.FromFloat32SliceNEON(out[i:end], bufOut[:c])
	}
}

func tanhBF16NEON(data []bfloat16.BFloat16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		chunk := data[i:min(i+halfChunk, len(data))]
		bfloat16.ToFloat32SliceNEON(buf[:len(chunk)], chunk)
		TanhFloat32NEON(buf[:len(chunk)], buf[:len(chunk)])
		bfloat16.FromFloat32SliceNEON(chunk, buf[:len(chunk)])
	}
}

func reluF16NEON(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		chunk := data[i:min(i+halfChunk, len(data))]
		float16.ToFloat32SliceNEON(buf[:len(chunk)], chunk)
		ReluFloat32NEON(buf[:len(chunk)], buf[:len(chunk)])
		float16.FromFloat32SliceNEON(chunk, buf[:len(chunk)])
	}
}

func hardSwishF16NEON(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		chunk := data[i:min(i+halfChunk, len(data))]
		float16.ToFloat32SliceNEON(buf[:len(chunk)], chunk)
		HardSwishFloat32NEON(buf[:len(chunk)], buf[:len(chunk)])
		float16.FromFloat32SliceNEON(chunk, buf[:len(chunk)])
	}
}

func siluF16NEON(data []float16.Float16) {
	SiluNEONF16(data, data)
}

// SiluNEONF16 computes SiLU for Float16 using ARM64 NEON FCVTL/FCVTN and float32 SiLU assembly.
func SiluNEONF16(in, out []float16.Float16) {
	var bufIn, bufOut [halfChunk]float32
	for i := 0; i < len(in); i += halfChunk {
		end := min(i+halfChunk, len(in))
		c := end - i
		float16.ToFloat32SliceNEON(bufIn[:c], in[i:end])
		SiluFloat32NEON(bufIn[:c], bufOut[:c])
		float16.FromFloat32SliceNEON(out[i:end], bufOut[:c])
	}
}

// SiluVJPNEONF16 computes the backward pass (VJP) of SiLU for Float16 using ARM64 NEON assembly.
func SiluVJPNEONF16(y, x, dOutput, dx []float16.Float16) {
	_ = y
	var bufX, bufDOut, bufDx [halfChunk]float32
	for i := 0; i < len(dOutput); i += halfChunk {
		end := min(i+halfChunk, len(dOutput))
		c := end - i
		float16.ToFloat32SliceNEON(bufX[:c], x[i:end])
		float16.ToFloat32SliceNEON(bufDOut[:c], dOutput[i:end])
		SiluVJPFloat32NEON(nil, bufX[:c], bufDOut[:c], bufDx[:c])
		float16.FromFloat32SliceNEON(dx[i:end], bufDx[:c])
	}
}

// GeluExactNEONF16 computes exact GELU for Float16 using ARM64 NEON assembly.
func GeluExactNEONF16(in, out []float16.Float16) {
	var bufIn, bufOut [halfChunk]float32
	for i := 0; i < len(in); i += halfChunk {
		end := min(i+halfChunk, len(in))
		c := end - i
		float16.ToFloat32SliceNEON(bufIn[:c], in[i:end])
		GeluExactFloat32NEON(bufIn[:c], bufOut[:c])
		float16.FromFloat32SliceNEON(out[i:end], bufOut[:c])
	}
}

// GeluApproxNEONF16 computes approximate GELU for Float16 using ARM64 NEON assembly.
func GeluApproxNEONF16(in, out []float16.Float16) {
	var bufIn, bufOut [halfChunk]float32
	for i := 0; i < len(in); i += halfChunk {
		end := min(i+halfChunk, len(in))
		c := end - i
		float16.ToFloat32SliceNEON(bufIn[:c], in[i:end])
		GeluApproxFloat32NEON(bufIn[:c], bufOut[:c])
		float16.FromFloat32SliceNEON(out[i:end], bufOut[:c])
	}
}

func tanhF16NEON(data []float16.Float16) {
	var buf [halfChunk]float32
	for i := 0; i < len(data); i += halfChunk {
		chunk := data[i:min(i+halfChunk, len(data))]
		float16.ToFloat32SliceNEON(buf[:len(chunk)], chunk)
		TanhFloat32NEON(buf[:len(chunk)], buf[:len(chunk)])
		float16.FromFloat32SliceNEON(chunk, buf[:len(chunk)])
	}
}

// -----------------------------------------------------------------------------
// Float64 Vector Math & Activations (ARM64 NEON)
// -----------------------------------------------------------------------------

func exp128Float64(x archsimd.Float64x2) archsimd.Float64x2 {
	xClamped := x.Max(expF64MinLog).Min(expF64MaxLog)
	z := xClamped.Mul(expF64Log2E)

	// Round to nearest integer using IEEE-754 mantissa alignment.
	zRound := z.Add(expF64Magic)
	nFloat := zRound.Sub(expF64Magic)

	// Integer n in bits via two's complement subtraction.
	nInt := zRound.ToBits().Sub(expF64MagicBits)
	scale := nInt.Add(expF641023).ShiftAllLeft(52).BitsToFloat64()

	// Reduced argument g = x - n*ln(2).
	g := xClamped.Sub(nFloat.Mul(expF64Ln2Hi)).Sub(nFloat.Mul(expF64Ln2Lo))

	poly := expF64C11.MulAdd(g, expF64C10)
	poly = poly.MulAdd(g, expF64C9)
	poly = poly.MulAdd(g, expF64C8)
	poly = poly.MulAdd(g, expF64C7)
	poly = poly.MulAdd(g, expF64C6)
	poly = poly.MulAdd(g, expF64C5)
	poly = poly.MulAdd(g, expF64C4)
	poly = poly.MulAdd(g, expF64C3)
	poly = poly.MulAdd(g, expF64C2)
	poly = poly.Mul(g).Mul(g).Add(g).Add(expF64One)

	return scale.Mul(poly)
}

// SiluNEONFloat64 computes SiLU for float64 using ARM64 NEON.
func SiluNEONFloat64(in, out []float64) {
	i := 0
	for ; i+4 <= len(in); i += 4 {
		x0 := archsimd.LoadFloat64x2(in[i : i+2])
		x1 := archsimd.LoadFloat64x2(in[i+2 : i+4])
		expNeg0 := exp128Float64(x0.Neg())
		expNeg1 := exp128Float64(x1.Neg())
		res0 := x0.Div(expF64One.Add(expNeg0))
		res1 := x1.Div(expF64One.Add(expNeg1))
		res0.Store(out[i : i+2])
		res1.Store(out[i+2 : i+4])
	}
	for ; i+2 <= len(in); i += 2 {
		x := archsimd.LoadFloat64x2(in[i : i+2])
		res := x.Div(expF64One.Add(exp128Float64(x.Neg())))
		res.Store(out[i : i+2])
	}
	if i < len(in) {
		x := in[i]
		out[i] = x / (1.0 + math.Exp(-x))
	}
}

// SiluVJPNEONFloat64 computes the backward pass (VJP) of SiLU for float64 using ARM64 NEON.
func SiluVJPNEONFloat64(y, x, dOutput, dx []float64) {
	_ = y
	n := len(dOutput)
	i := 0
	for ; i+4 <= n; i += 4 {
		x0 := archsimd.LoadFloat64x2(x[i : i+2])
		x1 := archsimd.LoadFloat64x2(x[i+2 : i+4])
		dOut0 := archsimd.LoadFloat64x2(dOutput[i : i+2])
		dOut1 := archsimd.LoadFloat64x2(dOutput[i+2 : i+4])
		s0 := expF64One.Div(expF64One.Add(exp128Float64(x0.Neg())))
		s1 := expF64One.Div(expF64One.Add(exp128Float64(x1.Neg())))
		term0 := expF64One.Add(x0).Sub(x0.Mul(s0))
		term1 := expF64One.Add(x1).Sub(x1.Mul(s1))
		dOut0.Mul(s0).Mul(term0).Store(dx[i : i+2])
		dOut1.Mul(s1).Mul(term1).Store(dx[i+2 : i+4])
	}
	for ; i+2 <= n; i += 2 {
		x0 := archsimd.LoadFloat64x2(x[i : i+2])
		dOut0 := archsimd.LoadFloat64x2(dOutput[i : i+2])
		s0 := expF64One.Div(expF64One.Add(exp128Float64(x0.Neg())))
		term0 := expF64One.Add(x0).Sub(x0.Mul(s0))
		dOut0.Mul(s0).Mul(term0).Store(dx[i : i+2])
	}
	if i < n {
		xi := x[i]
		s := 1.0 / (1.0 + math.Exp(-xi))
		dx[i] = dOutput[i] * s * ((1.0 + xi) - xi*s)
	}
}

func tanh128Float64(x archsimd.Float64x2) archsimd.Float64x2 {
	xClamped := x.Max(tanhF64Neg19).Min(tanhF6419)
	twoX := xClamped.Mul(tanhF64Two)
	exp2x := exp128Float64(twoX)
	num := exp2x.Sub(tanhF64One)
	den := exp2x.Add(tanhF64One)
	return num.Div(den)
}

func erf128Float64(x archsimd.Float64x2) archsimd.Float64x2 {
	absX := x.Abs()
	t := erfF64One.Div(erfF64One.Add(erfF64P.Mul(absX)))

	poly := erfF64A5.MulAdd(t, erfF64A4)
	poly = poly.MulAdd(t, erfF64A3)
	poly = poly.MulAdd(t, erfF64A2)
	poly = poly.MulAdd(t, erfF64A1)
	poly = poly.Mul(t)

	expNegX2 := exp128Float64(absX.Mul(absX).Neg())
	res := erfF64One.Sub(poly.Mul(expNegX2))

	return res.ToBits().Xor(x.ToBits().And(erfF64SignMask)).BitsToFloat64()
}

// GeluExactNEONFloat64 computes exact GELU for float64 using ARM64 NEON.
func GeluExactNEONFloat64(in, out []float64) {
	i := 0
	for ; i+2 <= len(in); i += 2 {
		x := archsimd.LoadFloat64x2(in[i : i+2])
		scaledX := x.Mul(erfF64Rsqrt2)
		erfVal := erf128Float64(scaledX)
		res := erfF64Half.Mul(x).Mul(erfF64One.Add(erfVal))
		res.Store(out[i : i+2])
	}
	if i < len(in) {
		var bufIn, bufOut [2]float64
		bufIn[0] = in[i]
		x := archsimd.LoadFloat64x2(bufIn[:])
		scaledX := x.Mul(erfF64Rsqrt2)
		erfVal := erf128Float64(scaledX)
		res := erfF64Half.Mul(x).Mul(erfF64One.Add(erfVal))
		res.Store(bufOut[:])
		out[i] = bufOut[0]
	}
}

// GeluApproxNEONFloat64 computes approximate GELU for float64 using ARM64 NEON.
func GeluApproxNEONFloat64(in, out []float64) {
	i := 0
	for ; i+4 <= len(in); i += 4 {
		x0 := archsimd.LoadFloat64x2(in[i : i+2])
		x1 := archsimd.LoadFloat64x2(in[i+2 : i+4])
		x3_0 := x0.Mul(x0).Mul(x0)
		x3_1 := x1.Mul(x1).Mul(x1)
		inner0 := geluF64Sqrt2ByPi.Mul(x3_0.MulAdd(geluF64C, x0))
		inner1 := geluF64Sqrt2ByPi.Mul(x3_1.MulAdd(geluF64C, x1))
		t0 := tanh128Float64(inner0)
		t1 := tanh128Float64(inner1)
		res0 := geluF64Half.Mul(x0).Mul(geluF64One.Add(t0))
		res1 := geluF64Half.Mul(x1).Mul(geluF64One.Add(t1))
		res0.Store(out[i : i+2])
		res1.Store(out[i+2 : i+4])
	}
	for ; i+2 <= len(in); i += 2 {
		x := archsimd.LoadFloat64x2(in[i : i+2])
		x3 := x.Mul(x).Mul(x)
		inner := geluF64Sqrt2ByPi.Mul(x3.MulAdd(geluF64C, x))
		t := tanh128Float64(inner)
		res := geluF64Half.Mul(x).Mul(geluF64One.Add(t))
		res.Store(out[i : i+2])
	}
	if i < len(in) {
		var bufIn, bufOut [2]float64
		bufIn[0] = in[i]
		x := archsimd.LoadFloat64x2(bufIn[:])
		x3 := x.Mul(x).Mul(x)
		inner := geluF64Sqrt2ByPi.Mul(x3.MulAdd(geluF64C, x))
		t := tanh128Float64(inner)
		res := geluF64Half.Mul(x).Mul(geluF64One.Add(t))
		res.Store(bufOut[:])
		out[i] = bufOut[0]
	}
}

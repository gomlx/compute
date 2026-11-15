// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd && go1.27

package neon

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
)

//go:noescape
func binaryMidAddFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryMidSubRHSFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryMidSubLHSFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryMidMulFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryMidDivRHSFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryMidDivLHSFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryMidMaxFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryMidMinFloat16NEONAsm(vec, bcast, out unsafe.Pointer, aDim, bDim, cDim int)

//go:noescape
func binaryTrailingAddFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

//go:noescape
func binaryTrailingSubRHSFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

//go:noescape
func binaryTrailingSubLHSFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

//go:noescape
func binaryTrailingMulFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

//go:noescape
func binaryTrailingDivRHSFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

//go:noescape
func binaryTrailingDivLHSFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

//go:noescape
func binaryTrailingMaxFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

//go:noescape
func binaryTrailingMinFloat16NEONAsm(vec, scalar, out unsafe.Pointer, aDim, bDim int)

// DispatchBinaryMidNEON dispatches binary mid/leading/none broadcast operations to ARM64 NEON kernels.
func DispatchBinaryMidNEON(op compute.OpType, isLHS bool, lhs, rhs, out unsafe.Pointer, A, B, C int, dtype dtypes.DType) bool {
	if A <= 0 || B <= 0 || C <= 0 {
		return true
	}
	vec := lhs
	bcast := rhs
	if isLHS {
		vec = rhs
		bcast = lhs
	}

	switch dtype {
	case dtypes.Float16:
		switch op {
		case compute.OpTypeAdd:
			binaryMidAddFloat16NEONAsm(vec, bcast, out, A, B, C)
			return true
		case compute.OpTypeSub:
			if !isLHS {
				binaryMidSubRHSFloat16NEONAsm(vec, bcast, out, A, B, C)
			} else {
				binaryMidSubLHSFloat16NEONAsm(vec, bcast, out, A, B, C)
			}
			return true
		case compute.OpTypeMul:
			binaryMidMulFloat16NEONAsm(vec, bcast, out, A, B, C)
			return true
		case compute.OpTypeDiv:
			if !isLHS {
				binaryMidDivRHSFloat16NEONAsm(vec, bcast, out, A, B, C)
			} else {
				binaryMidDivLHSFloat16NEONAsm(vec, bcast, out, A, B, C)
			}
			return true
		case compute.OpTypeMax:
			binaryMidMaxFloat16NEONAsm(vec, bcast, out, A, B, C)
			return true
		case compute.OpTypeMin:
			binaryMidMinFloat16NEONAsm(vec, bcast, out, A, B, C)
			return true
		}

	case dtypes.BFloat16:
		vecSlice := unsafe.Slice((*bfloat16.BFloat16)(vec), A*B*C)
		bcastSlice := unsafe.Slice((*bfloat16.BFloat16)(bcast), A*C)
		outSlice := unsafe.Slice((*bfloat16.BFloat16)(out), A*B*C)
		switch op {
		case compute.OpTypeAdd:
			binaryMidAddBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		case compute.OpTypeSub:
			if !isLHS {
				binaryMidSubRHSBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			} else {
				binaryMidSubLHSBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			}
			return true
		case compute.OpTypeMul:
			binaryMidMulBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		case compute.OpTypeDiv:
			if !isLHS {
				binaryMidDivRHSBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			} else {
				binaryMidDivLHSBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			}
			return true
		case compute.OpTypeMax:
			binaryMidMaxBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		case compute.OpTypeMin:
			binaryMidMinBFloat16NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		}

	case dtypes.Float32:
		vecSlice := unsafe.Slice((*float32)(vec), A*B*C)
		bcastSlice := unsafe.Slice((*float32)(bcast), A*C)
		outSlice := unsafe.Slice((*float32)(out), A*B*C)
		switch op {
		case compute.OpTypeAdd:
			binaryMidAddFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		case compute.OpTypeSub:
			if !isLHS {
				binaryMidSubRHSFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			} else {
				binaryMidSubLHSFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			}
			return true
		case compute.OpTypeMul:
			binaryMidMulFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		case compute.OpTypeDiv:
			if !isLHS {
				binaryMidDivRHSFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			} else {
				binaryMidDivLHSFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			}
			return true
		case compute.OpTypeMax:
			binaryMidMaxFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		case compute.OpTypeMin:
			binaryMidMinFloat32NEON(vecSlice, bcastSlice, outSlice, A, B, C)
			return true
		}
	}
	return false
}

// ----------------------------------------------------------------------------
// Float32 Binary Mid Kernels
// ----------------------------------------------------------------------------

func binaryMidAddFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))).Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))).Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))).Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = vec[vOff+c] + bcast[bOff+c]
			}
		}
	}
}

func binaryMidSubRHSFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = vec[vOff+c] - bcast[bOff+c]
			}
		}
	}
}

func binaryMidSubLHSFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))).Sub(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bcast[bOff+c] - vec[vOff+c]
			}
		}
	}
}

func binaryMidMulFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = vec[vOff+c] * bcast[bOff+c]
			}
		}
	}
}

func binaryMidDivRHSFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = vec[vOff+c] / bcast[bOff+c]
			}
		}
	}
}

func binaryMidDivLHSFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))).Div(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bcast[bOff+c] / vec[vOff+c]
			}
		}
	}
}

func binaryMidMaxFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Max(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))).Max(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))).Max(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))).Max(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Max(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = max(vec[vOff+c], bcast[bOff+c])
			}
		}
	}
}

func binaryMidMinFloat32NEON(vec, bcast, out []float32, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Min(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+4]))).Min(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+4]))))
				v2 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+8]))).Min(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+8]))))
				v3 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c+12]))).Min(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c+12]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
				v1.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+4])))
				v2.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+8])))
				v3.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c+12])))
			}
			for ; c+4 <= C; c += 4 {
				v0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&vec[vOff+c]))).Min(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&bcast[bOff+c]))))
				v0.StoreArray((*[4]float32)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = min(vec[vOff+c], bcast[bOff+c])
			}
		}
	}
}

// ----------------------------------------------------------------------------
// BFloat16 Binary Mid Kernels
// ----------------------------------------------------------------------------

func binaryMidAddBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+16 <= C; c += 16 {
				v0Lo, v0Hi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				b0Lo, b0Hi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				v1Lo, v1Hi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c+8]))).ToFloat32()
				b1Lo, b1Hi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c+8]))).ToFloat32()
				bfloat16.FromFloat32x4(v0Lo.Add(b0Lo), v0Hi.Add(b0Hi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
				bfloat16.FromFloat32x4(v1Lo.Add(b1Lo), v1Hi.Add(b1Hi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c+8])))
			}
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(vLo.Add(bLo), vHi.Add(bHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(vec[vOff+c].Float32() + bcast[bOff+c].Float32())
			}
		}
	}
}

func binaryMidSubRHSBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(vLo.Sub(bLo), vHi.Sub(bHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(vec[vOff+c].Float32() - bcast[bOff+c].Float32())
			}
		}
	}
}

func binaryMidSubLHSBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(bLo.Sub(vLo), bHi.Sub(vHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(bcast[bOff+c].Float32() - vec[vOff+c].Float32())
			}
		}
	}
}

func binaryMidMulBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(vLo.Mul(bLo), vHi.Mul(bHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(vec[vOff+c].Float32() * bcast[bOff+c].Float32())
			}
		}
	}
}

func binaryMidDivRHSBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(vLo.Div(bLo), vHi.Div(bHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(vec[vOff+c].Float32() / bcast[bOff+c].Float32())
			}
		}
	}
}

func binaryMidDivLHSBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(bLo.Div(vLo), bHi.Div(vHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(bcast[bOff+c].Float32() / vec[vOff+c].Float32())
			}
		}
	}
}

func binaryMidMaxBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(vLo.Max(bLo), vHi.Max(bHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(max(vec[vOff+c].Float32(), bcast[bOff+c].Float32()))
			}
		}
	}
}

func binaryMidMinBFloat16NEON(vec, bcast, out []bfloat16.BFloat16, A, B, C int) {
	for a := range A {
		bOff := a * C
		for b := range B {
			vOff := (a*B + b) * C
			c := 0
			for ; c+8 <= C; c += 8 {
				vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[vOff+c]))).ToFloat32()
				bLo, bHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&bcast[bOff+c]))).ToFloat32()
				bfloat16.FromFloat32x4(vLo.Min(bLo), vHi.Min(bHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[vOff+c])))
			}
			for ; c < C; c++ {
				out[vOff+c] = bfloat16.FromFloat32(min(vec[vOff+c].Float32(), bcast[bOff+c].Float32()))
			}
		}
	}
}

// ----------------------------------------------------------------------------
// BFloat16 Binary Trailing Kernels
// ----------------------------------------------------------------------------

func binaryTrailingAddBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(vLo.Add(sv), vHi.Add(sv)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(vec[off+b].Float32() + s)
		}
	}
}

func binaryTrailingSubRHSBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(vLo.Sub(sv), vHi.Sub(sv)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(vec[off+b].Float32() - s)
		}
	}
}

func binaryTrailingSubLHSBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(sv.Sub(vLo), sv.Sub(vHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(s - vec[off+b].Float32())
		}
	}
}

func binaryTrailingMulBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(vLo.Mul(sv), vHi.Mul(sv)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(vec[off+b].Float32() * s)
		}
	}
}

func binaryTrailingDivRHSBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(vLo.Div(sv), vHi.Div(sv)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(vec[off+b].Float32() / s)
		}
	}
}

func binaryTrailingDivLHSBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(sv.Div(vLo), sv.Div(vHi)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(s / vec[off+b].Float32())
		}
	}
}

func binaryTrailingMaxBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(vLo.Max(sv), vHi.Max(sv)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(max(vec[off+b].Float32(), s))
		}
	}
}

func binaryTrailingMinBFloat16NEON(vec, scalar, out []bfloat16.BFloat16, A, B int) {
	for a := range A {
		s := scalar[a].Float32()
		sv := archsimd.BroadcastFloat32x4(s)
		off := a * B
		b := 0
		for ; b+8 <= B; b += 8 {
			vLo, vHi := bfloat16.LoadBFloat16x8((*[8]bfloat16.BFloat16)(unsafe.Pointer(&vec[off+b]))).ToFloat32()
			bfloat16.FromFloat32x4(vLo.Min(sv), vHi.Min(sv)).Store((*[8]bfloat16.BFloat16)(unsafe.Pointer(&out[off+b])))
		}
		for ; b < B; b++ {
			out[off+b] = bfloat16.FromFloat32(min(vec[off+b].Float32(), s))
		}
	}
}

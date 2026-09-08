// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package ops

import (
	"simd"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
)

const PrioritySIMD = gobackend.PriorityTyped + 5

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeAdd, PrioritySIMD, execAddSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeSub, PrioritySIMD, execSubSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeMul, PrioritySIMD, execMulSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeDiv, PrioritySIMD, execDivSIMD)
}

// -----------------------------------------------------------------------------
// Core SIMD primitives: VectorVector, VectorScalar, ScalarVector
// -----------------------------------------------------------------------------

// Float32
func simdAddVVFloat32(lhs, rhs, out []float32) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Add(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vR, _ := simd.LoadFloat32sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSFloat32(lhs []float32, c float32, out []float32) {
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVFloat32(lhs, rhs, out []float32) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Sub(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vR, _ := simd.LoadFloat32sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSFloat32(lhs []float32, c float32, out []float32) {
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVFloat32(c float32, rhs []float32, out []float32) {
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Sub(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadFloat32sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVFloat32(lhs, rhs, out []float32) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Mul(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vR, _ := simd.LoadFloat32sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSFloat32(lhs []float32, c float32, out []float32) {
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

func simdDivVVFloat32(lhs, rhs, out []float32) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Div(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vR, _ := simd.LoadFloat32sPart(rhs[i:])
		vL.Div(vR).StorePart(out[i:])
	}
}

func simdDivVSFloat32(lhs []float32, c float32, out []float32) {
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat32s(lhs[i:]).Div(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat32sPart(lhs[i:])
		vL.Div(vC).StorePart(out[i:])
	}
}

func simdDivSVFloat32(c float32, rhs []float32, out []float32) {
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Div(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadFloat32sPart(rhs[i:])
		vC.Div(vR).StorePart(out[i:])
	}
}

// Float64
func simdAddVVFloat64(lhs, rhs, out []float64) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Add(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vR, _ := simd.LoadFloat64sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSFloat64(lhs []float64, c float64, out []float64) {
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVFloat64(lhs, rhs, out []float64) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Sub(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vR, _ := simd.LoadFloat64sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSFloat64(lhs []float64, c float64, out []float64) {
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVFloat64(c float64, rhs []float64, out []float64) {
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Sub(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadFloat64sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVFloat64(lhs, rhs, out []float64) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Mul(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vR, _ := simd.LoadFloat64sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSFloat64(lhs []float64, c float64, out []float64) {
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

func simdDivVVFloat64(lhs, rhs, out []float64) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Div(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vR, _ := simd.LoadFloat64sPart(rhs[i:])
		vL.Div(vR).StorePart(out[i:])
	}
}

func simdDivVSFloat64(lhs []float64, c float64, out []float64) {
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadFloat64s(lhs[i:]).Div(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadFloat64sPart(lhs[i:])
		vL.Div(vC).StorePart(out[i:])
	}
}

func simdDivSVFloat64(c float64, rhs []float64, out []float64) {
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Div(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadFloat64sPart(rhs[i:])
		vC.Div(vR).StorePart(out[i:])
	}
}

// Int32
func simdAddVVInt32(lhs, rhs, out []int32) {
	vLen := simd.BroadcastInt32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(lhs[i:]).Add(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt32sPart(lhs[i:])
		vR, _ := simd.LoadInt32sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSInt32(lhs []int32, c int32, out []int32) {
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt32sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVInt32(lhs, rhs, out []int32) {
	vLen := simd.BroadcastInt32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(lhs[i:]).Sub(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt32sPart(lhs[i:])
		vR, _ := simd.LoadInt32sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSInt32(lhs []int32, c int32, out []int32) {
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt32sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVInt32(c int32, rhs []int32, out []int32) {
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Sub(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadInt32sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVInt32(lhs, rhs, out []int32) {
	vLen := simd.BroadcastInt32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(lhs[i:]).Mul(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt32sPart(lhs[i:])
		vR, _ := simd.LoadInt32sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSInt32(lhs []int32, c int32, out []int32) {
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt32sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

// Uint32
func simdAddVVUint32(lhs, rhs, out []uint32) {
	vLen := simd.BroadcastUint32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint32s(lhs[i:]).Add(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint32sPart(lhs[i:])
		vR, _ := simd.LoadUint32sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSUint32(lhs []uint32, c uint32, out []uint32) {
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint32s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint32sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVUint32(lhs, rhs, out []uint32) {
	vLen := simd.BroadcastUint32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint32s(lhs[i:]).Sub(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint32sPart(lhs[i:])
		vR, _ := simd.LoadUint32sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSUint32(lhs []uint32, c uint32, out []uint32) {
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint32s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint32sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVUint32(c uint32, rhs []uint32, out []uint32) {
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Sub(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadUint32sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVUint32(lhs, rhs, out []uint32) {
	vLen := simd.BroadcastUint32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint32s(lhs[i:]).Mul(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint32sPart(lhs[i:])
		vR, _ := simd.LoadUint32sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSUint32(lhs []uint32, c uint32, out []uint32) {
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint32s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint32sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

// Int64
func simdAddVVInt64(lhs, rhs, out []int64) {
	vLen := simd.BroadcastInt64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt64s(lhs[i:]).Add(simd.LoadInt64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt64sPart(lhs[i:])
		vR, _ := simd.LoadInt64sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSInt64(lhs []int64, c int64, out []int64) {
	vC := simd.BroadcastInt64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt64s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt64sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVInt64(lhs, rhs, out []int64) {
	vLen := simd.BroadcastInt64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt64s(lhs[i:]).Sub(simd.LoadInt64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt64sPart(lhs[i:])
		vR, _ := simd.LoadInt64sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSInt64(lhs []int64, c int64, out []int64) {
	vC := simd.BroadcastInt64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt64s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadInt64sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVInt64(c int64, rhs []int64, out []int64) {
	vC := simd.BroadcastInt64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Sub(simd.LoadInt64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadInt64sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

// Uint64
func simdAddVVUint64(lhs, rhs, out []uint64) {
	vLen := simd.BroadcastUint64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint64s(lhs[i:]).Add(simd.LoadUint64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint64sPart(lhs[i:])
		vR, _ := simd.LoadUint64sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSUint64(lhs []uint64, c uint64, out []uint64) {
	vC := simd.BroadcastUint64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint64s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint64sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVUint64(lhs, rhs, out []uint64) {
	vLen := simd.BroadcastUint64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint64s(lhs[i:]).Sub(simd.LoadUint64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint64sPart(lhs[i:])
		vR, _ := simd.LoadUint64sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSUint64(lhs []uint64, c uint64, out []uint64) {
	vC := simd.BroadcastUint64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadUint64s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < len(out) {
		vL, _ := simd.LoadUint64sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVUint64(c uint64, rhs []uint64, out []uint64) {
	vC := simd.BroadcastUint64s(c)
	vLen := vC.Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		vC.Sub(simd.LoadUint64s(rhs[i:])).Store(out[i:])
	}
	if i < len(out) {
		vR, _ := simd.LoadUint64sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

// BFloat16
func simdAddVVBFloat16(lhs, rhs, out []bfloat16.BFloat16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Add(eR), oL.Add(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Add(eR), oL.Add(oR)), out[i:])
	}
}

func simdAddVSBFloat16(lhs []bfloat16.BFloat16, c bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Add(vC), oL.Add(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Add(vC), oL.Add(vC)), out[i:])
	}
}

func simdSubVVBFloat16(lhs, rhs, out []bfloat16.BFloat16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Sub(eR), oL.Sub(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Sub(eR), oL.Sub(oR)), out[i:])
	}
}

func simdSubVSBFloat16(lhs []bfloat16.BFloat16, c bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Sub(vC), oL.Sub(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Sub(vC), oL.Sub(vC)), out[i:])
	}
}

func simdSubSVBFloat16(c bfloat16.BFloat16, rhs []bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(vC.Sub(eR), vC.Sub(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(vC.Sub(eR), vC.Sub(oR)), out[i:])
	}
}

func simdMulVVBFloat16(lhs, rhs, out []bfloat16.BFloat16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Mul(eR), oL.Mul(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Mul(eR), oL.Mul(oR)), out[i:])
	}
}

func simdMulVSBFloat16(lhs []bfloat16.BFloat16, c bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Mul(vC), oL.Mul(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Mul(vC), oL.Mul(vC)), out[i:])
	}
}

func simdDivVVBFloat16(lhs, rhs, out []bfloat16.BFloat16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Div(eR), oL.Div(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Div(eR), oL.Div(oR)), out[i:])
	}
}

func simdDivVSBFloat16(lhs []bfloat16.BFloat16, c bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Div(vC), oL.Div(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Div(vC), oL.Div(vC)), out[i:])
	}
}

func simdDivSVBFloat16(c bfloat16.BFloat16, rhs []bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(vC.Div(eR), vC.Div(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(vC.Div(eR), vC.Div(oR)), out[i:])
	}
}

// Float16
func simdAddVVFloat16(lhs, rhs, out []float16.Float16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Add(eR), oL.Add(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Add(eR), oL.Add(oR)), out[i:])
	}
}

func simdAddVSFloat16(lhs []float16.Float16, c float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Add(vC), oL.Add(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Add(vC), oL.Add(vC)), out[i:])
	}
}

func simdSubVVFloat16(lhs, rhs, out []float16.Float16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Sub(eR), oL.Sub(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Sub(eR), oL.Sub(oR)), out[i:])
	}
}

func simdSubVSFloat16(lhs []float16.Float16, c float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Sub(vC), oL.Sub(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Sub(vC), oL.Sub(vC)), out[i:])
	}
}

func simdSubSVFloat16(c float16.Float16, rhs []float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(vC.Sub(eR), vC.Sub(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(vC.Sub(eR), vC.Sub(oR)), out[i:])
	}
}

func simdMulVVFloat16(lhs, rhs, out []float16.Float16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Mul(eR), oL.Mul(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Mul(eR), oL.Mul(oR)), out[i:])
	}
}

func simdMulVSFloat16(lhs []float16.Float16, c float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Mul(vC), oL.Mul(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Mul(vC), oL.Mul(vC)), out[i:])
	}
}

func simdDivVVFloat16(lhs, rhs, out []float16.Float16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Div(eR), oL.Div(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Div(eR), oL.Div(oR)), out[i:])
	}
}

func simdDivVSFloat16(lhs []float16.Float16, c float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Div(vC), oL.Div(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Div(vC), oL.Div(vC)), out[i:])
	}
}

func simdDivSVFloat16(c float16.Float16, rhs []float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(vC.Div(eR), vC.Div(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(vC.Div(eR), vC.Div(oR)), out[i:])
	}
}

// -----------------------------------------------------------------------------
// Pattern Dispatch Helper
// -----------------------------------------------------------------------------

func dispatchBinarySIMD[T any](
	lhs, rhs, output []T,
	bcastCfg gobackend.BroadcastConfig,
	vv func(lhs, rhs, out []T),
	vs func(lhs []T, c T, out []T),
	sv func(c T, rhs []T, out []T),
) {
	switch {
	case len(rhs) == 1:
		vs(lhs, rhs[0], output)
	case len(lhs) == 1:
		sv(lhs[0], rhs, output)
	case bcastCfg.Pattern == gobackend.BroadcastNone:
		vv(lhs, rhs, output)
	case bcastCfg.Pattern == gobackend.BroadcastLeadingRHS:
		A, B := bcastCfg.A, bcastCfg.B
		for a := range A {
			offset := a * B
			vv(lhs[offset:offset+B], rhs[:B], output[offset:offset+B])
		}
	case bcastCfg.Pattern == gobackend.BroadcastLeadingLHS:
		A, B := bcastCfg.A, bcastCfg.B
		for a := range A {
			offset := a * B
			vv(lhs[:B], rhs[offset:offset+B], output[offset:offset+B])
		}
	case bcastCfg.Pattern == gobackend.BroadcastTrailingRHS:
		A, B := bcastCfg.A, bcastCfg.B
		for a := range A {
			offset := a * B
			vs(lhs[offset:offset+B], rhs[a], output[offset:offset+B])
		}
	case bcastCfg.Pattern == gobackend.BroadcastTrailingLHS:
		A, B := bcastCfg.A, bcastCfg.B
		for a := range A {
			offset := a * B
			sv(lhs[a], rhs[offset:offset+B], output[offset:offset+B])
		}
	case bcastCfg.Pattern == gobackend.BroadcastRowCol:
		A, B := bcastCfg.A, bcastCfg.B
		for a := range A {
			offset := a * B
			sv(lhs[a], rhs[:B], output[offset:offset+B])
		}
	case bcastCfg.Pattern == gobackend.BroadcastColRow:
		A, B := bcastCfg.A, bcastCfg.B
		for a := range A {
			offset := a * B
			vs(lhs[:B], rhs[a], output[offset:offset+B])
		}
	default:
		// BroadcastGeneral: fallback to general iterator.
		zipIter := bcastCfg.ZipIter
		if zipIter == nil {
			panic("missing zipIter for BroadcastGeneral")
		}
		for idx := range zipIter.IterFlatIndices() {
			var singleOut [1]T
			var singleL [1]T
			var singleR [1]T
			singleL[0] = lhs[idx.LHSFlatIdx]
			singleR[0] = rhs[idx.RHSFlatIdx]
			vv(singleL[:], singleR[:], singleOut[:])
			output[idx.TgtFlatIdx] = singleOut[0]
		}
	}
}

// -----------------------------------------------------------------------------
// Node Executors
// -----------------------------------------------------------------------------

func execAddSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
	}
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)

	switch lhs.RawShape.DType {
	case dtypes.Float32:
		dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdAddVVFloat32, simdAddVSFloat32, func(c float32, r []float32, out []float32) { simdAddVSFloat32(r, c, out) })
	case dtypes.Float64:
		dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdAddVVFloat64, simdAddVSFloat64, func(c float64, r []float64, out []float64) { simdAddVSFloat64(r, c, out) })
	case dtypes.BFloat16:
		dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdAddVVBFloat16, simdAddVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) { simdAddVSBFloat16(r, c, out) })
	case dtypes.Float16:
		dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdAddVVFloat16, simdAddVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdAddVSFloat16(r, c, out) })
	case dtypes.Int32:
		dispatchBinarySIMD(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdAddVVInt32, simdAddVSInt32, func(c int32, r []int32, out []int32) { simdAddVSInt32(r, c, out) })
	case dtypes.Uint32:
		dispatchBinarySIMD(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdAddVVUint32, simdAddVSUint32, func(c uint32, r []uint32, out []uint32) { simdAddVSUint32(r, c, out) })
	case dtypes.Int64:
		dispatchBinarySIMD(lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), bcastCfg,
			simdAddVVInt64, simdAddVSInt64, func(c int64, r []int64, out []int64) { simdAddVSInt64(r, c, out) })
	case dtypes.Uint64:
		dispatchBinarySIMD(lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), bcastCfg,
			simdAddVVUint64, simdAddVSUint64, func(c uint64, r []uint64, out []uint64) { simdAddVSUint64(r, c, out) })
	default:
		return execAdd(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execSubSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	lhs, rhs, output, _, _ := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)

	switch lhs.RawShape.DType {
	case dtypes.Float32:
		dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdSubVVFloat32, simdSubVSFloat32, simdSubSVFloat32)
	case dtypes.Float64:
		dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdSubVVFloat64, simdSubVSFloat64, simdSubSVFloat64)
	case dtypes.BFloat16:
		dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdSubVVBFloat16, simdSubVSBFloat16, simdSubSVBFloat16)
	case dtypes.Float16:
		dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdSubVVFloat16, simdSubVSFloat16, simdSubSVFloat16)
	case dtypes.Int32:
		dispatchBinarySIMD(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdSubVVInt32, simdSubVSInt32, simdSubSVInt32)
	case dtypes.Uint32:
		dispatchBinarySIMD(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdSubVVUint32, simdSubVSUint32, simdSubSVUint32)
	case dtypes.Int64:
		dispatchBinarySIMD(lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), bcastCfg,
			simdSubVVInt64, simdSubVSInt64, simdSubSVInt64)
	case dtypes.Uint64:
		dispatchBinarySIMD(lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), bcastCfg,
			simdSubVVUint64, simdSubVSUint64, simdSubSVUint64)
	default:
		return execSub(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execMulSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
	}
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)

	switch lhs.RawShape.DType {
	case dtypes.Float32:
		dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdMulVVFloat32, simdMulVSFloat32, func(c float32, r []float32, out []float32) { simdMulVSFloat32(r, c, out) })
	case dtypes.Float64:
		dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdMulVVFloat64, simdMulVSFloat64, func(c float64, r []float64, out []float64) { simdMulVSFloat64(r, c, out) })
	case dtypes.BFloat16:
		dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdMulVVBFloat16, simdMulVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) { simdMulVSBFloat16(r, c, out) })
	case dtypes.Float16:
		dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdMulVVFloat16, simdMulVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdMulVSFloat16(r, c, out) })
	case dtypes.Int32:
		dispatchBinarySIMD(lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdMulVVInt32, simdMulVSInt32, func(c int32, r []int32, out []int32) { simdMulVSInt32(r, c, out) })
	case dtypes.Uint32:
		dispatchBinarySIMD(lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdMulVVUint32, simdMulVSUint32, func(c uint32, r []uint32, out []uint32) { simdMulVSUint32(r, c, out) })
	default:
		return execMul(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execDivSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	lhs, rhs, output, _, _ := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)

	switch lhs.RawShape.DType {
	case dtypes.Float32:
		dispatchBinarySIMD(lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdDivVVFloat32, simdDivVSFloat32, simdDivSVFloat32)
	case dtypes.Float64:
		dispatchBinarySIMD(lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdDivVVFloat64, simdDivVSFloat64, simdDivSVFloat64)
	case dtypes.BFloat16:
		dispatchBinarySIMD(lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdDivVVBFloat16, simdDivVSBFloat16, simdDivSVBFloat16)
	case dtypes.Float16:
		dispatchBinarySIMD(lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdDivVVFloat16, simdDivVSFloat16, simdDivSVFloat16)
	default:
		return execDiv(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package ops

import (
	"simd"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/support/simdmath"
)

const PrioritySIMD = gobackend.PriorityTyped + 5

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeAdd, PrioritySIMD, execAddSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeSub, PrioritySIMD, execSubSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeMul, PrioritySIMD, execMulSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeDiv, PrioritySIMD, execDivSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeMax, PrioritySIMD, execMaxSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeMin, PrioritySIMD, execMinSIMD)
}

// -----------------------------------------------------------------------------
// Core SIMD primitives: VectorVector, VectorScalar, ScalarVector
// -----------------------------------------------------------------------------

// Float32
func simdAddVVFloat32(lhs, rhs, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Add(simd.LoadFloat32s(rhs[i:]))
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Add(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Add(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Add(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Add(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSFloat32(lhs []float32, c float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Add(vC)
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Add(vC)
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Add(vC)
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Add(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVFloat32(lhs, rhs, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Sub(simd.LoadFloat32s(rhs[i:]))
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Sub(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Sub(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Sub(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Sub(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSFloat32(lhs []float32, c float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Sub(vC)
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Sub(vC)
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Sub(vC)
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Sub(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVFloat32(c float32, rhs []float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Sub(simd.LoadFloat32s(rhs[i:]))
		v1 := vC.Sub(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := vC.Sub(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := vC.Sub(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Sub(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVFloat32(lhs, rhs, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Mul(simd.LoadFloat32s(rhs[i:]))
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Mul(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Mul(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Mul(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Mul(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSFloat32(lhs []float32, c float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Mul(vC)
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Mul(vC)
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Mul(vC)
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Mul(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

func simdDivVVFloat32(lhs, rhs, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Div(simd.LoadFloat32s(rhs[i:]))
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Div(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Div(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Div(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Div(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vL.Div(vR).StorePart(out[i:])
	}
}

func simdDivVSFloat32(lhs []float32, c float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Div(vC)
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Div(vC)
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Div(vC)
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Div(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Div(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vL.Div(vC).StorePart(out[i:])
	}
}

func simdDivSVFloat32(c float32, rhs []float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Div(simd.LoadFloat32s(rhs[i:]))
		v1 := vC.Div(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := vC.Div(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := vC.Div(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Div(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vC.Div(vR).StorePart(out[i:])
	}
}

// Float64
func simdAddVVFloat64(lhs, rhs, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Add(simd.LoadFloat64s(rhs[i:]))
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Add(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Add(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Add(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Add(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSFloat64(lhs []float64, c float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Add(vC)
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Add(vC)
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Add(vC)
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Add(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVFloat64(lhs, rhs, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Sub(simd.LoadFloat64s(rhs[i:]))
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Sub(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Sub(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Sub(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Sub(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSFloat64(lhs []float64, c float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Sub(vC)
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Sub(vC)
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Sub(vC)
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Sub(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVFloat64(c float64, rhs []float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Sub(simd.LoadFloat64s(rhs[i:]))
		v1 := vC.Sub(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := vC.Sub(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := vC.Sub(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Sub(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVFloat64(lhs, rhs, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Mul(simd.LoadFloat64s(rhs[i:]))
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Mul(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Mul(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Mul(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Mul(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSFloat64(lhs []float64, c float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Mul(vC)
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Mul(vC)
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Mul(vC)
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Mul(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

func simdDivVVFloat64(lhs, rhs, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Div(simd.LoadFloat64s(rhs[i:]))
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Div(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Div(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Div(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Div(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vL.Div(vR).StorePart(out[i:])
	}
}

func simdDivVSFloat64(lhs []float64, c float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Div(vC)
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Div(vC)
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Div(vC)
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Div(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Div(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vL.Div(vC).StorePart(out[i:])
	}
}

func simdDivSVFloat64(c float64, rhs []float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Div(simd.LoadFloat64s(rhs[i:]))
		v1 := vC.Div(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := vC.Div(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := vC.Div(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Div(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vC.Div(vR).StorePart(out[i:])
	}
}

// Int32
func simdAddVVInt32(lhs, rhs, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastInt32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Add(simd.LoadInt32s(rhs[i:]))
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Add(simd.LoadInt32s(rhs[i+vLen:]))
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Add(simd.LoadInt32s(rhs[i+vLen2:]))
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Add(simd.LoadInt32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Add(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vR, _ := simdmath.LoadInt32sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSInt32(lhs []int32, c int32, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Add(vC)
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Add(vC)
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Add(vC)
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Add(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVInt32(lhs, rhs, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastInt32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Sub(simd.LoadInt32s(rhs[i:]))
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Sub(simd.LoadInt32s(rhs[i+vLen:]))
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Sub(simd.LoadInt32s(rhs[i+vLen2:]))
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Sub(simd.LoadInt32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Sub(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vR, _ := simdmath.LoadInt32sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSInt32(lhs []int32, c int32, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Sub(vC)
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Sub(vC)
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Sub(vC)
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Sub(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVInt32(c int32, rhs []int32, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Sub(simd.LoadInt32s(rhs[i:]))
		v1 := vC.Sub(simd.LoadInt32s(rhs[i+vLen:]))
		v2 := vC.Sub(simd.LoadInt32s(rhs[i+vLen2:]))
		v3 := vC.Sub(simd.LoadInt32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Sub(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadInt32sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVInt32(lhs, rhs, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastInt32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Mul(simd.LoadInt32s(rhs[i:]))
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Mul(simd.LoadInt32s(rhs[i+vLen:]))
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Mul(simd.LoadInt32s(rhs[i+vLen2:]))
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Mul(simd.LoadInt32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Mul(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vR, _ := simdmath.LoadInt32sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSInt32(lhs []int32, c int32, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Mul(vC)
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Mul(vC)
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Mul(vC)
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Mul(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

// Uint32
func simdAddVVUint32(lhs, rhs, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastUint32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Add(simd.LoadUint32s(rhs[i:]))
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Add(simd.LoadUint32s(rhs[i+vLen:]))
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Add(simd.LoadUint32s(rhs[i+vLen2:]))
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Add(simd.LoadUint32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Add(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vR, _ := simdmath.LoadUint32sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSUint32(lhs []uint32, c uint32, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Add(vC)
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Add(vC)
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Add(vC)
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Add(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVUint32(lhs, rhs, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastUint32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Sub(simd.LoadUint32s(rhs[i:]))
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Sub(simd.LoadUint32s(rhs[i+vLen:]))
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Sub(simd.LoadUint32s(rhs[i+vLen2:]))
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Sub(simd.LoadUint32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Sub(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vR, _ := simdmath.LoadUint32sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSUint32(lhs []uint32, c uint32, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Sub(vC)
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Sub(vC)
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Sub(vC)
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Sub(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVUint32(c uint32, rhs []uint32, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Sub(simd.LoadUint32s(rhs[i:]))
		v1 := vC.Sub(simd.LoadUint32s(rhs[i+vLen:]))
		v2 := vC.Sub(simd.LoadUint32s(rhs[i+vLen2:]))
		v3 := vC.Sub(simd.LoadUint32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Sub(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadUint32sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

func simdMulVVUint32(lhs, rhs, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastUint32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Mul(simd.LoadUint32s(rhs[i:]))
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Mul(simd.LoadUint32s(rhs[i+vLen:]))
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Mul(simd.LoadUint32s(rhs[i+vLen2:]))
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Mul(simd.LoadUint32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Mul(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vR, _ := simdmath.LoadUint32sPart(rhs[i:])
		vL.Mul(vR).StorePart(out[i:])
	}
}

func simdMulVSUint32(lhs []uint32, c uint32, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Mul(vC)
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Mul(vC)
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Mul(vC)
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Mul(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Mul(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vL.Mul(vC).StorePart(out[i:])
	}
}

// Int64
func simdAddVVInt64(lhs, rhs, out []int64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastInt64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt64s(lhs[i:]).Add(simd.LoadInt64s(rhs[i:]))
		v1 := simd.LoadInt64s(lhs[i+vLen:]).Add(simd.LoadInt64s(rhs[i+vLen:]))
		v2 := simd.LoadInt64s(lhs[i+vLen2:]).Add(simd.LoadInt64s(rhs[i+vLen2:]))
		v3 := simd.LoadInt64s(lhs[i+vLen3:]).Add(simd.LoadInt64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt64s(lhs[i:]).Add(simd.LoadInt64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt64sPart(lhs[i:])
		vR, _ := simdmath.LoadInt64sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSInt64(lhs []int64, c int64, out []int64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastInt64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt64s(lhs[i:]).Add(vC)
		v1 := simd.LoadInt64s(lhs[i+vLen:]).Add(vC)
		v2 := simd.LoadInt64s(lhs[i+vLen2:]).Add(vC)
		v3 := simd.LoadInt64s(lhs[i+vLen3:]).Add(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt64s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt64sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVInt64(lhs, rhs, out []int64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastInt64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt64s(lhs[i:]).Sub(simd.LoadInt64s(rhs[i:]))
		v1 := simd.LoadInt64s(lhs[i+vLen:]).Sub(simd.LoadInt64s(rhs[i+vLen:]))
		v2 := simd.LoadInt64s(lhs[i+vLen2:]).Sub(simd.LoadInt64s(rhs[i+vLen2:]))
		v3 := simd.LoadInt64s(lhs[i+vLen3:]).Sub(simd.LoadInt64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt64s(lhs[i:]).Sub(simd.LoadInt64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt64sPart(lhs[i:])
		vR, _ := simdmath.LoadInt64sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSInt64(lhs []int64, c int64, out []int64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastInt64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt64s(lhs[i:]).Sub(vC)
		v1 := simd.LoadInt64s(lhs[i+vLen:]).Sub(vC)
		v2 := simd.LoadInt64s(lhs[i+vLen2:]).Sub(vC)
		v3 := simd.LoadInt64s(lhs[i+vLen3:]).Sub(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt64s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt64sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVInt64(c int64, rhs []int64, out []int64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastInt64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Sub(simd.LoadInt64s(rhs[i:]))
		v1 := vC.Sub(simd.LoadInt64s(rhs[i+vLen:]))
		v2 := vC.Sub(simd.LoadInt64s(rhs[i+vLen2:]))
		v3 := vC.Sub(simd.LoadInt64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Sub(simd.LoadInt64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadInt64sPart(rhs[i:])
		vC.Sub(vR).StorePart(out[i:])
	}
}

// Uint64
func simdAddVVUint64(lhs, rhs, out []uint64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastUint64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint64s(lhs[i:]).Add(simd.LoadUint64s(rhs[i:]))
		v1 := simd.LoadUint64s(lhs[i+vLen:]).Add(simd.LoadUint64s(rhs[i+vLen:]))
		v2 := simd.LoadUint64s(lhs[i+vLen2:]).Add(simd.LoadUint64s(rhs[i+vLen2:]))
		v3 := simd.LoadUint64s(lhs[i+vLen3:]).Add(simd.LoadUint64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint64s(lhs[i:]).Add(simd.LoadUint64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint64sPart(lhs[i:])
		vR, _ := simdmath.LoadUint64sPart(rhs[i:])
		vL.Add(vR).StorePart(out[i:])
	}
}

func simdAddVSUint64(lhs []uint64, c uint64, out []uint64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastUint64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint64s(lhs[i:]).Add(vC)
		v1 := simd.LoadUint64s(lhs[i+vLen:]).Add(vC)
		v2 := simd.LoadUint64s(lhs[i+vLen2:]).Add(vC)
		v3 := simd.LoadUint64s(lhs[i+vLen3:]).Add(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint64s(lhs[i:]).Add(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint64sPart(lhs[i:])
		vL.Add(vC).StorePart(out[i:])
	}
}

func simdSubVVUint64(lhs, rhs, out []uint64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastUint64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint64s(lhs[i:]).Sub(simd.LoadUint64s(rhs[i:]))
		v1 := simd.LoadUint64s(lhs[i+vLen:]).Sub(simd.LoadUint64s(rhs[i+vLen:]))
		v2 := simd.LoadUint64s(lhs[i+vLen2:]).Sub(simd.LoadUint64s(rhs[i+vLen2:]))
		v3 := simd.LoadUint64s(lhs[i+vLen3:]).Sub(simd.LoadUint64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint64s(lhs[i:]).Sub(simd.LoadUint64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint64sPart(lhs[i:])
		vR, _ := simdmath.LoadUint64sPart(rhs[i:])
		vL.Sub(vR).StorePart(out[i:])
	}
}

func simdSubVSUint64(lhs []uint64, c uint64, out []uint64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastUint64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint64s(lhs[i:]).Sub(vC)
		v1 := simd.LoadUint64s(lhs[i+vLen:]).Sub(vC)
		v2 := simd.LoadUint64s(lhs[i+vLen2:]).Sub(vC)
		v3 := simd.LoadUint64s(lhs[i+vLen3:]).Sub(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint64s(lhs[i:]).Sub(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint64sPart(lhs[i:])
		vL.Sub(vC).StorePart(out[i:])
	}
}

func simdSubSVUint64(c uint64, rhs []uint64, out []uint64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = rhs[n-1]
	vC := simd.BroadcastUint64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := vC.Sub(simd.LoadUint64s(rhs[i:]))
		v1 := vC.Sub(simd.LoadUint64s(rhs[i+vLen:]))
		v2 := vC.Sub(simd.LoadUint64s(rhs[i+vLen2:]))
		v3 := vC.Sub(simd.LoadUint64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		vC.Sub(simd.LoadUint64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vR, _ := simdmath.LoadUint64sPart(rhs[i:])
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
// Pattern Dispatch Helper and Broadcast Fast-Paths
// -----------------------------------------------------------------------------

func canExecuteBinarySIMD(node *gobackend.Node, lhs, rhs *gobackend.Buffer, supportedDTypes []dtypes.DType) bool {
	dtype := lhs.RawShape.DType
	if !slices.Contains(supportedDTypes, dtype) {
		return false
	}
	if lhs.RawShape.Size() == 1 || rhs.RawShape.Size() == 1 {
		return true
	}
	cfg := GetBroadcastConfig(node, lhs, rhs, nil)
	return cfg.Pattern != gobackend.BroadcastGeneral
}

func simdV2V1AddFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vR := simd.LoadFloat32s(rhs)
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Add(vR).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Add(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Add(vR1).Store(out[off+vLen:])
		}
		return
	}
	if B == 4*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Add(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Add(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Add(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Add(vR3).Store(out[off+3*vLen:])
		}
		return
	}
	if B == 6*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Add(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Add(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Add(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Add(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Add(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Add(vR5).Store(out[off+5*vLen:])
		}
		return
	}
	if B == 8*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		vR6 := simd.LoadFloat32s(rhs[6*vLen:])
		vR7 := simd.LoadFloat32s(rhs[7*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Add(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Add(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Add(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Add(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Add(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Add(vR5).Store(out[off+5*vLen:])
			simd.LoadFloat32s(lhs[off+6*vLen:]).Add(vR6).Store(out[off+6*vLen:])
			simd.LoadFloat32s(lhs[off+7*vLen:]).Add(vR7).Store(out[off+7*vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		lRow := lhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			simd.LoadFloat32s(lRow[i:]).Add(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
			simd.LoadFloat32s(lRow[i+vLen:]).Add(simd.LoadFloat32s(rhs[i+vLen:])).Store(oRow[i+vLen:])
			simd.LoadFloat32s(lRow[i+2*vLen:]).Add(simd.LoadFloat32s(rhs[i+2*vLen:])).Store(oRow[i+2*vLen:])
			simd.LoadFloat32s(lRow[i+3*vLen:]).Add(simd.LoadFloat32s(rhs[i+3*vLen:])).Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lRow[i:]).Add(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lRow[i:])
			vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
			vL.Add(vR).StorePart(oRow[i:])
		}
	}
}

func simdV1V2AddFloat32(lhs, rhs, out []float32, A, B int) {
	simdV2V1AddFloat32(rhs, lhs, out, A, B)
}

func simdV2V1MulFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vR := simd.LoadFloat32s(rhs)
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Mul(vR).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Mul(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Mul(vR1).Store(out[off+vLen:])
		}
		return
	}
	if B == 4*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Mul(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Mul(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Mul(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Mul(vR3).Store(out[off+3*vLen:])
		}
		return
	}
	if B == 6*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Mul(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Mul(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Mul(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Mul(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Mul(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Mul(vR5).Store(out[off+5*vLen:])
		}
		return
	}
	if B == 8*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		vR6 := simd.LoadFloat32s(rhs[6*vLen:])
		vR7 := simd.LoadFloat32s(rhs[7*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Mul(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Mul(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Mul(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Mul(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Mul(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Mul(vR5).Store(out[off+5*vLen:])
			simd.LoadFloat32s(lhs[off+6*vLen:]).Mul(vR6).Store(out[off+6*vLen:])
			simd.LoadFloat32s(lhs[off+7*vLen:]).Mul(vR7).Store(out[off+7*vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		lRow := lhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			simd.LoadFloat32s(lRow[i:]).Mul(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
			simd.LoadFloat32s(lRow[i+vLen:]).Mul(simd.LoadFloat32s(rhs[i+vLen:])).Store(oRow[i+vLen:])
			simd.LoadFloat32s(lRow[i+2*vLen:]).Mul(simd.LoadFloat32s(rhs[i+2*vLen:])).Store(oRow[i+2*vLen:])
			simd.LoadFloat32s(lRow[i+3*vLen:]).Mul(simd.LoadFloat32s(rhs[i+3*vLen:])).Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lRow[i:]).Mul(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lRow[i:])
			vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
			vL.Mul(vR).StorePart(oRow[i:])
		}
	}
}

func simdV1V2MulFloat32(lhs, rhs, out []float32, A, B int) {
	simdV2V1MulFloat32(rhs, lhs, out, A, B)
}

func simdV2V1SubFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vR := simd.LoadFloat32s(rhs)
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Sub(vR).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Sub(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Sub(vR1).Store(out[off+vLen:])
		}
		return
	}
	if B == 4*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Sub(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Sub(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Sub(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Sub(vR3).Store(out[off+3*vLen:])
		}
		return
	}
	if B == 6*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Sub(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Sub(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Sub(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Sub(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Sub(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Sub(vR5).Store(out[off+5*vLen:])
		}
		return
	}
	if B == 8*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		vR6 := simd.LoadFloat32s(rhs[6*vLen:])
		vR7 := simd.LoadFloat32s(rhs[7*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Sub(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Sub(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Sub(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Sub(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Sub(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Sub(vR5).Store(out[off+5*vLen:])
			simd.LoadFloat32s(lhs[off+6*vLen:]).Sub(vR6).Store(out[off+6*vLen:])
			simd.LoadFloat32s(lhs[off+7*vLen:]).Sub(vR7).Store(out[off+7*vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		lRow := lhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			simd.LoadFloat32s(lRow[i:]).Sub(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
			simd.LoadFloat32s(lRow[i+vLen:]).Sub(simd.LoadFloat32s(rhs[i+vLen:])).Store(oRow[i+vLen:])
			simd.LoadFloat32s(lRow[i+2*vLen:]).Sub(simd.LoadFloat32s(rhs[i+2*vLen:])).Store(oRow[i+2*vLen:])
			simd.LoadFloat32s(lRow[i+3*vLen:]).Sub(simd.LoadFloat32s(rhs[i+3*vLen:])).Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lRow[i:]).Sub(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lRow[i:])
			vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
			vL.Sub(vR).StorePart(oRow[i:])
		}
	}
}

func simdV1V2SubFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vL := simd.LoadFloat32s(lhs)
		for a := range A {
			off := a * B
			vL.Sub(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		for a := range A {
			off := a * B
			vL0.Sub(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Sub(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
		}
		return
	}
	if B == 4*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		vL2 := simd.LoadFloat32s(lhs[2*vLen:])
		vL3 := simd.LoadFloat32s(lhs[3*vLen:])
		for a := range A {
			off := a * B
			vL0.Sub(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Sub(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
			vL2.Sub(simd.LoadFloat32s(rhs[off+2*vLen:])).Store(out[off+2*vLen:])
			vL3.Sub(simd.LoadFloat32s(rhs[off+3*vLen:])).Store(out[off+3*vLen:])
		}
		return
	}
	if B == 6*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		vL2 := simd.LoadFloat32s(lhs[2*vLen:])
		vL3 := simd.LoadFloat32s(lhs[3*vLen:])
		vL4 := simd.LoadFloat32s(lhs[4*vLen:])
		vL5 := simd.LoadFloat32s(lhs[5*vLen:])
		for a := range A {
			off := a * B
			vL0.Sub(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Sub(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
			vL2.Sub(simd.LoadFloat32s(rhs[off+2*vLen:])).Store(out[off+2*vLen:])
			vL3.Sub(simd.LoadFloat32s(rhs[off+3*vLen:])).Store(out[off+3*vLen:])
			vL4.Sub(simd.LoadFloat32s(rhs[off+4*vLen:])).Store(out[off+4*vLen:])
			vL5.Sub(simd.LoadFloat32s(rhs[off+5*vLen:])).Store(out[off+5*vLen:])
		}
		return
	}
	if B == 8*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		vL2 := simd.LoadFloat32s(lhs[2*vLen:])
		vL3 := simd.LoadFloat32s(lhs[3*vLen:])
		vL4 := simd.LoadFloat32s(lhs[4*vLen:])
		vL5 := simd.LoadFloat32s(lhs[5*vLen:])
		vL6 := simd.LoadFloat32s(lhs[6*vLen:])
		vL7 := simd.LoadFloat32s(lhs[7*vLen:])
		for a := range A {
			off := a * B
			vL0.Sub(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Sub(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
			vL2.Sub(simd.LoadFloat32s(rhs[off+2*vLen:])).Store(out[off+2*vLen:])
			vL3.Sub(simd.LoadFloat32s(rhs[off+3*vLen:])).Store(out[off+3*vLen:])
			vL4.Sub(simd.LoadFloat32s(rhs[off+4*vLen:])).Store(out[off+4*vLen:])
			vL5.Sub(simd.LoadFloat32s(rhs[off+5*vLen:])).Store(out[off+5*vLen:])
			vL6.Sub(simd.LoadFloat32s(rhs[off+6*vLen:])).Store(out[off+6*vLen:])
			vL7.Sub(simd.LoadFloat32s(rhs[off+7*vLen:])).Store(out[off+7*vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		rRow := rhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			simd.LoadFloat32s(lhs[i:]).Sub(simd.LoadFloat32s(rRow[i:])).Store(oRow[i:])
			simd.LoadFloat32s(lhs[i+vLen:]).Sub(simd.LoadFloat32s(rRow[i+vLen:])).Store(oRow[i+vLen:])
			simd.LoadFloat32s(lhs[i+2*vLen:]).Sub(simd.LoadFloat32s(rRow[i+2*vLen:])).Store(oRow[i+2*vLen:])
			simd.LoadFloat32s(lhs[i+3*vLen:]).Sub(simd.LoadFloat32s(rRow[i+3*vLen:])).Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lhs[i:]).Sub(simd.LoadFloat32s(rRow[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
			vR, _ := simdmath.LoadFloat32sPart(rRow[i:])
			vL.Sub(vR).StorePart(oRow[i:])
		}
	}
}

func simdV2V1DivFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vR := simd.LoadFloat32s(rhs)
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Div(vR).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Div(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Div(vR1).Store(out[off+vLen:])
		}
		return
	}
	if B == 4*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Div(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Div(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Div(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Div(vR3).Store(out[off+3*vLen:])
		}
		return
	}
	if B == 6*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Div(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Div(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Div(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Div(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Div(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Div(vR5).Store(out[off+5*vLen:])
		}
		return
	}
	if B == 8*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		vR2 := simd.LoadFloat32s(rhs[2*vLen:])
		vR3 := simd.LoadFloat32s(rhs[3*vLen:])
		vR4 := simd.LoadFloat32s(rhs[4*vLen:])
		vR5 := simd.LoadFloat32s(rhs[5*vLen:])
		vR6 := simd.LoadFloat32s(rhs[6*vLen:])
		vR7 := simd.LoadFloat32s(rhs[7*vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Div(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Div(vR1).Store(out[off+vLen:])
			simd.LoadFloat32s(lhs[off+2*vLen:]).Div(vR2).Store(out[off+2*vLen:])
			simd.LoadFloat32s(lhs[off+3*vLen:]).Div(vR3).Store(out[off+3*vLen:])
			simd.LoadFloat32s(lhs[off+4*vLen:]).Div(vR4).Store(out[off+4*vLen:])
			simd.LoadFloat32s(lhs[off+5*vLen:]).Div(vR5).Store(out[off+5*vLen:])
			simd.LoadFloat32s(lhs[off+6*vLen:]).Div(vR6).Store(out[off+6*vLen:])
			simd.LoadFloat32s(lhs[off+7*vLen:]).Div(vR7).Store(out[off+7*vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		lRow := lhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			v0 := simd.LoadFloat32s(lRow[i:]).Div(simd.LoadFloat32s(rhs[i:]))
			v1 := simd.LoadFloat32s(lRow[i+vLen:]).Div(simd.LoadFloat32s(rhs[i+vLen:]))
			v2 := simd.LoadFloat32s(lRow[i+2*vLen:]).Div(simd.LoadFloat32s(rhs[i+2*vLen:]))
			v3 := simd.LoadFloat32s(lRow[i+3*vLen:]).Div(simd.LoadFloat32s(rhs[i+3*vLen:]))
			v0.Store(oRow[i:])
			v1.Store(oRow[i+vLen:])
			v2.Store(oRow[i+2*vLen:])
			v3.Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lRow[i:]).Div(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lRow[i:])
			vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
			vL.Div(vR).StorePart(oRow[i:])
		}
	}
}

func simdV1V2DivFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vL := simd.LoadFloat32s(lhs)
		for a := range A {
			off := a * B
			vL.Div(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		for a := range A {
			off := a * B
			vL0.Div(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Div(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
		}
		return
	}
	if B == 4*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		vL2 := simd.LoadFloat32s(lhs[2*vLen:])
		vL3 := simd.LoadFloat32s(lhs[3*vLen:])
		for a := range A {
			off := a * B
			vL0.Div(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Div(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
			vL2.Div(simd.LoadFloat32s(rhs[off+2*vLen:])).Store(out[off+2*vLen:])
			vL3.Div(simd.LoadFloat32s(rhs[off+3*vLen:])).Store(out[off+3*vLen:])
		}
		return
	}
	if B == 6*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		vL2 := simd.LoadFloat32s(lhs[2*vLen:])
		vL3 := simd.LoadFloat32s(lhs[3*vLen:])
		vL4 := simd.LoadFloat32s(lhs[4*vLen:])
		vL5 := simd.LoadFloat32s(lhs[5*vLen:])
		for a := range A {
			off := a * B
			vL0.Div(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Div(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
			vL2.Div(simd.LoadFloat32s(rhs[off+2*vLen:])).Store(out[off+2*vLen:])
			vL3.Div(simd.LoadFloat32s(rhs[off+3*vLen:])).Store(out[off+3*vLen:])
			vL4.Div(simd.LoadFloat32s(rhs[off+4*vLen:])).Store(out[off+4*vLen:])
			vL5.Div(simd.LoadFloat32s(rhs[off+5*vLen:])).Store(out[off+5*vLen:])
		}
		return
	}
	if B == 8*vLen {
		vL0 := simd.LoadFloat32s(lhs)
		vL1 := simd.LoadFloat32s(lhs[vLen:])
		vL2 := simd.LoadFloat32s(lhs[2*vLen:])
		vL3 := simd.LoadFloat32s(lhs[3*vLen:])
		vL4 := simd.LoadFloat32s(lhs[4*vLen:])
		vL5 := simd.LoadFloat32s(lhs[5*vLen:])
		vL6 := simd.LoadFloat32s(lhs[6*vLen:])
		vL7 := simd.LoadFloat32s(lhs[7*vLen:])
		for a := range A {
			off := a * B
			vL0.Div(simd.LoadFloat32s(rhs[off:])).Store(out[off:])
			vL1.Div(simd.LoadFloat32s(rhs[off+vLen:])).Store(out[off+vLen:])
			vL2.Div(simd.LoadFloat32s(rhs[off+2*vLen:])).Store(out[off+2*vLen:])
			vL3.Div(simd.LoadFloat32s(rhs[off+3*vLen:])).Store(out[off+3*vLen:])
			vL4.Div(simd.LoadFloat32s(rhs[off+4*vLen:])).Store(out[off+4*vLen:])
			vL5.Div(simd.LoadFloat32s(rhs[off+5*vLen:])).Store(out[off+5*vLen:])
			vL6.Div(simd.LoadFloat32s(rhs[off+6*vLen:])).Store(out[off+6*vLen:])
			vL7.Div(simd.LoadFloat32s(rhs[off+7*vLen:])).Store(out[off+7*vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		rRow := rhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			v0 := simd.LoadFloat32s(lhs[i:]).Div(simd.LoadFloat32s(rRow[i:]))
			v1 := simd.LoadFloat32s(lhs[i+vLen:]).Div(simd.LoadFloat32s(rRow[i+vLen:]))
			v2 := simd.LoadFloat32s(lhs[i+2*vLen:]).Div(simd.LoadFloat32s(rRow[i+2*vLen:]))
			v3 := simd.LoadFloat32s(lhs[i+3*vLen:]).Div(simd.LoadFloat32s(rRow[i+3*vLen:]))
			v0.Store(oRow[i:])
			v1.Store(oRow[i+vLen:])
			v2.Store(oRow[i+2*vLen:])
			v3.Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lhs[i:]).Div(simd.LoadFloat32s(rRow[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
			vR, _ := simdmath.LoadFloat32sPart(rRow[i:])
			vL.Div(vR).StorePart(oRow[i:])
		}
	}
}

func simdV2V1MaxFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vR := simd.LoadFloat32s(rhs)
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Max(vR).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Max(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Max(vR1).Store(out[off+vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		lRow := lhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			v0 := simd.LoadFloat32s(lRow[i:]).Max(simd.LoadFloat32s(rhs[i:]))
			v1 := simd.LoadFloat32s(lRow[i+vLen:]).Max(simd.LoadFloat32s(rhs[i+vLen:]))
			v2 := simd.LoadFloat32s(lRow[i+2*vLen:]).Max(simd.LoadFloat32s(rhs[i+2*vLen:]))
			v3 := simd.LoadFloat32s(lRow[i+3*vLen:]).Max(simd.LoadFloat32s(rhs[i+3*vLen:]))
			v0.Store(oRow[i:])
			v1.Store(oRow[i+vLen:])
			v2.Store(oRow[i+2*vLen:])
			v3.Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lRow[i:]).Max(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lRow[i:])
			vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
			vL.Max(vR).StorePart(oRow[i:])
		}
	}
}

func simdV1V2MaxFloat32(lhs, rhs, out []float32, A, B int) {
	simdV2V1MaxFloat32(rhs, lhs, out, A, B)
}

func simdV2V1MinFloat32(lhs, rhs, out []float32, A, B int) {
	vLen := simd.BroadcastFloat32s(0).Len()
	if B == vLen {
		vR := simd.LoadFloat32s(rhs)
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Min(vR).Store(out[off:])
		}
		return
	}
	if B == 2*vLen {
		vR0 := simd.LoadFloat32s(rhs)
		vR1 := simd.LoadFloat32s(rhs[vLen:])
		for a := range A {
			off := a * B
			simd.LoadFloat32s(lhs[off:]).Min(vR0).Store(out[off:])
			simd.LoadFloat32s(lhs[off+vLen:]).Min(vR1).Store(out[off+vLen:])
		}
		return
	}
	for a := range A {
		off := a * B
		lRow := lhs[off : off+B]
		oRow := out[off : off+B]
		i := 0
		for ; i+4*vLen <= B; i += 4 * vLen {
			v0 := simd.LoadFloat32s(lRow[i:]).Min(simd.LoadFloat32s(rhs[i:]))
			v1 := simd.LoadFloat32s(lRow[i+vLen:]).Min(simd.LoadFloat32s(rhs[i+vLen:]))
			v2 := simd.LoadFloat32s(lRow[i+2*vLen:]).Min(simd.LoadFloat32s(rhs[i+2*vLen:]))
			v3 := simd.LoadFloat32s(lRow[i+3*vLen:]).Min(simd.LoadFloat32s(rhs[i+3*vLen:]))
			v0.Store(oRow[i:])
			v1.Store(oRow[i+vLen:])
			v2.Store(oRow[i+2*vLen:])
			v3.Store(oRow[i+3*vLen:])
		}
		for ; i+vLen <= B; i += vLen {
			simd.LoadFloat32s(lRow[i:]).Min(simd.LoadFloat32s(rhs[i:])).Store(oRow[i:])
		}
		if i < B {
			vL, _ := simdmath.LoadFloat32sPart(lRow[i:])
			vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
			vL.Min(vR).StorePart(oRow[i:])
		}
	}
}

func simdV1V2MinFloat32(lhs, rhs, out []float32, A, B int) {
	simdV2V1MinFloat32(rhs, lhs, out, A, B)
}

// -----------------------------------------------------------------------------
// Max and Min Primitives
// -----------------------------------------------------------------------------

// Float32
func simdMaxVVFloat32(lhs, rhs, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Max(simd.LoadFloat32s(rhs[i:]))
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Max(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Max(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Max(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Max(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vL.Max(vR).StorePart(out[i:])
	}
}

func simdMaxVSFloat32(lhs []float32, c float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Max(vC)
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Max(vC)
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Max(vC)
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Max(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Max(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vL.Max(vC).StorePart(out[i:])
	}
}

func simdMinVVFloat32(lhs, rhs, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Min(simd.LoadFloat32s(rhs[i:]))
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Min(simd.LoadFloat32s(rhs[i+vLen:]))
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Min(simd.LoadFloat32s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Min(simd.LoadFloat32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Min(simd.LoadFloat32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat32sPart(rhs[i:])
		vL.Min(vR).StorePart(out[i:])
	}
}

func simdMinVSFloat32(lhs []float32, c float32, out []float32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat32s(lhs[i:]).Min(vC)
		v1 := simd.LoadFloat32s(lhs[i+vLen:]).Min(vC)
		v2 := simd.LoadFloat32s(lhs[i+vLen2:]).Min(vC)
		v3 := simd.LoadFloat32s(lhs[i+vLen3:]).Min(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat32s(lhs[i:]).Min(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat32sPart(lhs[i:])
		vL.Min(vC).StorePart(out[i:])
	}
}

// Float64
func simdMaxVVFloat64(lhs, rhs, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Max(simd.LoadFloat64s(rhs[i:]))
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Max(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Max(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Max(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Max(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vL.Max(vR).StorePart(out[i:])
	}
}

func simdMaxVSFloat64(lhs []float64, c float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Max(vC)
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Max(vC)
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Max(vC)
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Max(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Max(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vL.Max(vC).StorePart(out[i:])
	}
}

func simdMinVVFloat64(lhs, rhs, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastFloat64s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Min(simd.LoadFloat64s(rhs[i:]))
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Min(simd.LoadFloat64s(rhs[i+vLen:]))
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Min(simd.LoadFloat64s(rhs[i+vLen2:]))
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Min(simd.LoadFloat64s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Min(simd.LoadFloat64s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vR, _ := simdmath.LoadFloat64sPart(rhs[i:])
		vL.Min(vR).StorePart(out[i:])
	}
}

func simdMinVSFloat64(lhs []float64, c float64, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastFloat64s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadFloat64s(lhs[i:]).Min(vC)
		v1 := simd.LoadFloat64s(lhs[i+vLen:]).Min(vC)
		v2 := simd.LoadFloat64s(lhs[i+vLen2:]).Min(vC)
		v3 := simd.LoadFloat64s(lhs[i+vLen3:]).Min(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadFloat64s(lhs[i:]).Min(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadFloat64sPart(lhs[i:])
		vL.Min(vC).StorePart(out[i:])
	}
}

// Int32
func simdMaxVVInt32(lhs, rhs, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastInt32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Max(simd.LoadInt32s(rhs[i:]))
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Max(simd.LoadInt32s(rhs[i+vLen:]))
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Max(simd.LoadInt32s(rhs[i+vLen2:]))
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Max(simd.LoadInt32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Max(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vR, _ := simdmath.LoadInt32sPart(rhs[i:])
		vL.Max(vR).StorePart(out[i:])
	}
}

func simdMaxVSInt32(lhs []int32, c int32, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Max(vC)
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Max(vC)
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Max(vC)
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Max(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Max(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vL.Max(vC).StorePart(out[i:])
	}
}

func simdMinVVInt32(lhs, rhs, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastInt32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Min(simd.LoadInt32s(rhs[i:]))
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Min(simd.LoadInt32s(rhs[i+vLen:]))
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Min(simd.LoadInt32s(rhs[i+vLen2:]))
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Min(simd.LoadInt32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Min(simd.LoadInt32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vR, _ := simdmath.LoadInt32sPart(rhs[i:])
		vL.Min(vR).StorePart(out[i:])
	}
}

func simdMinVSInt32(lhs []int32, c int32, out []int32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastInt32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadInt32s(lhs[i:]).Min(vC)
		v1 := simd.LoadInt32s(lhs[i+vLen:]).Min(vC)
		v2 := simd.LoadInt32s(lhs[i+vLen2:]).Min(vC)
		v3 := simd.LoadInt32s(lhs[i+vLen3:]).Min(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadInt32s(lhs[i:]).Min(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadInt32sPart(lhs[i:])
		vL.Min(vC).StorePart(out[i:])
	}
}

// Uint32
func simdMaxVVUint32(lhs, rhs, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastUint32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Max(simd.LoadUint32s(rhs[i:]))
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Max(simd.LoadUint32s(rhs[i+vLen:]))
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Max(simd.LoadUint32s(rhs[i+vLen2:]))
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Max(simd.LoadUint32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Max(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vR, _ := simdmath.LoadUint32sPart(rhs[i:])
		vL.Max(vR).StorePart(out[i:])
	}
}

func simdMaxVSUint32(lhs []uint32, c uint32, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Max(vC)
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Max(vC)
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Max(vC)
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Max(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Max(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vL.Max(vC).StorePart(out[i:])
	}
}

func simdMinVVUint32(lhs, rhs, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	_ = rhs[n-1]
	vLen := simd.BroadcastUint32s(0).Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Min(simd.LoadUint32s(rhs[i:]))
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Min(simd.LoadUint32s(rhs[i+vLen:]))
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Min(simd.LoadUint32s(rhs[i+vLen2:]))
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Min(simd.LoadUint32s(rhs[i+vLen3:]))
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Min(simd.LoadUint32s(rhs[i:])).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vR, _ := simdmath.LoadUint32sPart(rhs[i:])
		vL.Min(vR).StorePart(out[i:])
	}
}

func simdMinVSUint32(lhs []uint32, c uint32, out []uint32) {
	n := len(out)
	if n == 0 {
		return
	}
	_ = lhs[n-1]
	vC := simd.BroadcastUint32s(c)
	vLen := vC.Len()
	vLen2 := 2 * vLen
	vLen3 := 3 * vLen
	vLen4 := 4 * vLen
	i := 0
	for ; i+vLen4 <= n; i += vLen4 {
		v0 := simd.LoadUint32s(lhs[i:]).Min(vC)
		v1 := simd.LoadUint32s(lhs[i+vLen:]).Min(vC)
		v2 := simd.LoadUint32s(lhs[i+vLen2:]).Min(vC)
		v3 := simd.LoadUint32s(lhs[i+vLen3:]).Min(vC)
		v0.Store(out[i:])
		v1.Store(out[i+vLen:])
		v2.Store(out[i+vLen2:])
		v3.Store(out[i+vLen3:])
	}
	for ; i+vLen <= n; i += vLen {
		simd.LoadUint32s(lhs[i:]).Min(vC).Store(out[i:])
	}
	if i < n {
		vL, _ := simdmath.LoadUint32sPart(lhs[i:])
		vL.Min(vC).StorePart(out[i:])
	}
}

// BFloat16
func simdMaxVVBFloat16(lhs, rhs, out []bfloat16.BFloat16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Max(eR), oL.Max(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Max(eR), oL.Max(oR)), out[i:])
	}
}

func simdMaxVSBFloat16(lhs []bfloat16.BFloat16, c bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Max(vC), oL.Max(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Max(vC), oL.Max(vC)), out[i:])
	}
}

func simdMinVVBFloat16(lhs, rhs, out []bfloat16.BFloat16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		eR, oR := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(rhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Min(eR), oL.Min(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		vR, _ := bfloat16.LoadBFloat16sPart(rhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		eR, oR := bfloat16.ToFloat32SIMD(vR)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Min(eR), oL.Min(oR)), out[i:])
	}
}

func simdMinVSBFloat16(lhs []bfloat16.BFloat16, c bfloat16.BFloat16, out []bfloat16.BFloat16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(lhs[i : i+vLen]))
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(eL.Min(vC), oL.Min(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := bfloat16.LoadBFloat16sPart(lhs[i:])
		eL, oL := bfloat16.ToFloat32SIMD(vL)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(eL.Min(vC), oL.Min(vC)), out[i:])
	}
}

// Float16
func simdMaxVVFloat16(lhs, rhs, out []float16.Float16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Max(eR), oL.Max(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Max(eR), oL.Max(oR)), out[i:])
	}
}

func simdMaxVSFloat16(lhs []float16.Float16, c float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Max(vC), oL.Max(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Max(vC), oL.Max(vC)), out[i:])
	}
}

func simdMinVVFloat16(lhs, rhs, out []float16.Float16) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		eR, oR := float16.ToFloat32SIMD(float16.LoadFloat16s(rhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Min(eR), oL.Min(oR)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		vR, _ := float16.LoadFloat16sPart(rhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		eR, oR := float16.ToFloat32SIMD(vR)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Min(eR), oL.Min(oR)), out[i:])
	}
}

func simdMinVSFloat16(lhs []float16.Float16, c float16.Float16, out []float16.Float16) {
	vC := simd.BroadcastFloat32s(c.Float32())
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		eL, oL := float16.ToFloat32SIMD(float16.LoadFloat16s(lhs[i : i+vLen]))
		float16.StoreFloat16s(float16.FromFloat32SIMD(eL.Min(vC), oL.Min(vC)), out[i:i+vLen])
	}
	if i < len(out) {
		vL, _ := float16.LoadFloat16sPart(lhs[i:])
		eL, oL := float16.ToFloat32SIMD(vL)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(eL.Min(vC), oL.Min(vC)), out[i:])
	}
}

const (
	minBinarySIMDParallelElements = 65536
	binarySIMDChunkElements       = 32768
)

func dispatchBinarySIMD[T any](
	backend *gobackend.Backend,
	lhs, rhs, output []T,
	bcastCfg gobackend.BroadcastConfig,
	vv func(lhs, rhs, out []T),
	vs func(lhs []T, c T, out []T),
	sv func(c T, rhs []T, out []T),
	v2v1 func(lhs, rhs, out []T, A, B int),
	v1v2 func(lhs, rhs, out []T, A, B int),
) error {
	totalElements := len(output)
	canParallelize := totalElements >= minBinarySIMDParallelElements &&
		backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled()

	switch {
	case len(rhs) == 1:
		c := rhs[0]
		if canParallelize {
			backend.Scheduler.ParallelizeChunks(totalElements, binarySIMDChunkElements, func(start, end int) {
				vs(lhs[start:end], c, output[start:end])
			})
		} else {
			vs(lhs, c, output)
		}
	case len(lhs) == 1:
		c := lhs[0]
		if canParallelize {
			backend.Scheduler.ParallelizeChunks(totalElements, binarySIMDChunkElements, func(start, end int) {
				sv(c, rhs[start:end], output[start:end])
			})
		} else {
			sv(c, rhs, output)
		}
	case bcastCfg.Pattern == gobackend.BroadcastNone:
		if canParallelize {
			backend.Scheduler.ParallelizeChunks(totalElements, binarySIMDChunkElements, func(start, end int) {
				vv(lhs[start:end], rhs[start:end], output[start:end])
			})
		} else {
			vv(lhs, rhs, output)
		}
	case bcastCfg.Pattern == gobackend.BroadcastLeadingRHS:
		A, B := bcastCfg.A, bcastCfg.B
		if canParallelize && A > 1 {
			minA := max(1, binarySIMDChunkElements/max(1, B))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				off := aStart * B
				count := aEnd - aStart
				if v2v1 != nil {
					v2v1(lhs[off:off+count*B], rhs[:B], output[off:off+count*B], count, B)
				} else {
					for a := aStart; a < aEnd; a++ {
						o := a * B
						vv(lhs[o:o+B], rhs[:B], output[o:o+B])
					}
				}
			})
		} else if v2v1 != nil {
			v2v1(lhs, rhs, output, A, B)
		} else {
			for a := range A {
				offset := a * B
				vv(lhs[offset:offset+B], rhs[:B], output[offset:offset+B])
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastLeadingLHS:
		A, B := bcastCfg.A, bcastCfg.B
		if canParallelize && A > 1 {
			minA := max(1, binarySIMDChunkElements/max(1, B))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				off := aStart * B
				count := aEnd - aStart
				if v1v2 != nil {
					v1v2(lhs[:B], rhs[off:off+count*B], output[off:off+count*B], count, B)
				} else {
					for a := aStart; a < aEnd; a++ {
						o := a * B
						vv(lhs[:B], rhs[o:o+B], output[o:o+B])
					}
				}
			})
		} else if v1v2 != nil {
			v1v2(lhs, rhs, output, A, B)
		} else {
			for a := range A {
				offset := a * B
				vv(lhs[:B], rhs[offset:offset+B], output[offset:offset+B])
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastTrailingRHS:
		A, B := bcastCfg.A, bcastCfg.B
		if canParallelize && A > 1 {
			minA := max(1, binarySIMDChunkElements/max(1, B))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					offset := a * B
					vs(lhs[offset:offset+B], rhs[a], output[offset:offset+B])
				}
			})
		} else {
			for a := range A {
				offset := a * B
				vs(lhs[offset:offset+B], rhs[a], output[offset:offset+B])
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastTrailingLHS:
		A, B := bcastCfg.A, bcastCfg.B
		if canParallelize && A > 1 {
			minA := max(1, binarySIMDChunkElements/max(1, B))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					offset := a * B
					sv(lhs[a], rhs[offset:offset+B], output[offset:offset+B])
				}
			})
		} else {
			for a := range A {
				offset := a * B
				sv(lhs[a], rhs[offset:offset+B], output[offset:offset+B])
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastRowCol:
		A, B := bcastCfg.A, bcastCfg.B
		if canParallelize && A > 1 {
			minA := max(1, binarySIMDChunkElements/max(1, B))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					offset := a * B
					sv(lhs[a], rhs[:B], output[offset:offset+B])
				}
			})
		} else {
			for a := range A {
				offset := a * B
				sv(lhs[a], rhs[:B], output[offset:offset+B])
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastColRow:
		A, B := bcastCfg.A, bcastCfg.B
		if canParallelize && A > 1 {
			minA := max(1, binarySIMDChunkElements/max(1, B))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					offset := a * B
					vs(lhs[:B], rhs[a], output[offset:offset+B])
				}
			})
		} else {
			for a := range A {
				offset := a * B
				vs(lhs[:B], rhs[a], output[offset:offset+B])
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastMidRHS:
		A, B, C := bcastCfg.A, bcastCfg.B, bcastCfg.C
		bc := B * C
		if canParallelize && A >= 8 {
			minA := max(1, binarySIMDChunkElements/max(1, bc))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					lOff := a * bc
					rOff := a * C
					if v2v1 != nil {
						v2v1(lhs[lOff:lOff+bc], rhs[rOff:rOff+C], output[lOff:lOff+bc], B, C)
					} else {
						for b := range B {
							o := lOff + b*C
							vv(lhs[o:o+C], rhs[rOff:rOff+C], output[o:o+C])
						}
					}
				}
			})
		} else if canParallelize && A*B > 1 {
			abTotal := A * B
			minAB := max(1, binarySIMDChunkElements/max(1, C))
			backend.Scheduler.ParallelizeChunks(abTotal, minAB, func(abStart, abEnd int) {
				if v2v1 != nil {
					for ab := abStart; ab < abEnd; {
						a := ab / B
						aEnd := min(abEnd, (a+1)*B)
						bCount := aEnd - ab
						lOff := ab * C
						rOff := a * C
						v2v1(lhs[lOff:lOff+bCount*C], rhs[rOff:rOff+C], output[lOff:lOff+bCount*C], bCount, C)
						ab = aEnd
					}
				} else {
					for ab := abStart; ab < abEnd; ab++ {
						a := ab / B
						lOff := ab * C
						rOff := a * C
						vv(lhs[lOff:lOff+C], rhs[rOff:rOff+C], output[lOff:lOff+C])
					}
				}
			})
		} else {
			for a := range A {
				lOff := a * bc
				rOff := a * C
				if v2v1 != nil {
					v2v1(lhs[lOff:lOff+bc], rhs[rOff:rOff+C], output[lOff:lOff+bc], B, C)
				} else {
					for b := range B {
						o := lOff + b*C
						vv(lhs[o:o+C], rhs[rOff:rOff+C], output[o:o+C])
					}
				}
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastMidLHS:
		A, B, C := bcastCfg.A, bcastCfg.B, bcastCfg.C
		bc := B * C
		if canParallelize && A >= 8 {
			minA := max(1, binarySIMDChunkElements/max(1, bc))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					lOff := a * C
					rOff := a * bc
					if v1v2 != nil {
						v1v2(lhs[lOff:lOff+C], rhs[rOff:rOff+bc], output[rOff:rOff+bc], B, C)
					} else {
						for b := range B {
							o := rOff + b*C
							vv(lhs[lOff:lOff+C], rhs[o:o+C], output[o:o+C])
						}
					}
				}
			})
		} else if canParallelize && A*B > 1 {
			abTotal := A * B
			minAB := max(1, binarySIMDChunkElements/max(1, C))
			backend.Scheduler.ParallelizeChunks(abTotal, minAB, func(abStart, abEnd int) {
				if v1v2 != nil {
					for ab := abStart; ab < abEnd; {
						a := ab / B
						aEnd := min(abEnd, (a+1)*B)
						bCount := aEnd - ab
						lOff := a * C
						rOff := ab * C
						v1v2(lhs[lOff:lOff+C], rhs[rOff:rOff+bCount*C], output[rOff:rOff+bCount*C], bCount, C)
						ab = aEnd
					}
				} else {
					for ab := abStart; ab < abEnd; ab++ {
						a := ab / B
						lOff := a * C
						rOff := ab * C
						vv(lhs[lOff:lOff+C], rhs[rOff:rOff+C], output[rOff:rOff+C])
					}
				}
			})
		} else {
			for a := range A {
				lOff := a * C
				rOff := a * bc
				if v1v2 != nil {
					v1v2(lhs[lOff:lOff+C], rhs[rOff:rOff+bc], output[rOff:rOff+bc], B, C)
				} else {
					for b := range B {
						o := rOff + b*C
						vv(lhs[lOff:lOff+C], rhs[o:o+C], output[o:o+C])
					}
				}
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastOuterInnerRHS:
		A, B, C := bcastCfg.A, bcastCfg.B, bcastCfg.C
		bc := B * C
		if canParallelize && A >= 8 {
			minA := max(1, binarySIMDChunkElements/max(1, bc))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					aOff := a * bc
					for b := range B {
						off := aOff + b*C
						vs(lhs[off:off+C], rhs[b], output[off:off+C])
					}
				}
			})
		} else if canParallelize && A*B > 1 {
			abTotal := A * B
			minAB := max(1, binarySIMDChunkElements/max(1, C))
			backend.Scheduler.ParallelizeChunks(abTotal, minAB, func(abStart, abEnd int) {
				for ab := abStart; ab < abEnd; ab++ {
					b := ab % B
					off := ab * C
					vs(lhs[off:off+C], rhs[b], output[off:off+C])
				}
			})
		} else {
			for a := range A {
				aOff := a * bc
				for b := range B {
					off := aOff + b*C
					vs(lhs[off:off+C], rhs[b], output[off:off+C])
				}
			}
		}
	case bcastCfg.Pattern == gobackend.BroadcastOuterInnerLHS:
		A, B, C := bcastCfg.A, bcastCfg.B, bcastCfg.C
		bc := B * C
		if canParallelize && A >= 8 {
			minA := max(1, binarySIMDChunkElements/max(1, bc))
			backend.Scheduler.ParallelizeChunks(A, minA, func(aStart, aEnd int) {
				for a := aStart; a < aEnd; a++ {
					aOff := a * bc
					for b := range B {
						off := aOff + b*C
						sv(lhs[b], rhs[off:off+C], output[off:off+C])
					}
				}
			})
		} else if canParallelize && A*B > 1 {
			abTotal := A * B
			minAB := max(1, binarySIMDChunkElements/max(1, C))
			backend.Scheduler.ParallelizeChunks(abTotal, minAB, func(abStart, abEnd int) {
				for ab := abStart; ab < abEnd; ab++ {
					b := ab % B
					off := ab * C
					sv(lhs[b], rhs[off:off+C], output[off:off+C])
				}
			})
		} else {
			for a := range A {
				aOff := a * bc
				for b := range B {
					off := aOff + b*C
					sv(lhs[b], rhs[off:off+C], output[off:off+C])
				}
			}
		}
	default:
		return gobackend.ErrFallback
	}
	return nil
}

// -----------------------------------------------------------------------------
// Node Executors
// -----------------------------------------------------------------------------

var supportedAddSubDTypes = []dtypes.DType{
	dtypes.Float32, dtypes.Float64, dtypes.BFloat16, dtypes.Float16, dtypes.Int32, dtypes.Uint32, dtypes.Int64, dtypes.Uint64,
}

var supportedMulMaxMinDTypes = []dtypes.DType{
	dtypes.Float32, dtypes.Float64, dtypes.BFloat16, dtypes.Float16, dtypes.Int32, dtypes.Uint32,
}

var supportedDivDTypes = []dtypes.DType{
	dtypes.Float32, dtypes.Float64, dtypes.BFloat16, dtypes.Float16,
}

func tryBinaryTrailingArch(backend *gobackend.Backend, op compute.OpType, lhs, rhs, output *gobackend.Buffer, bcastCfg gobackend.BroadcastConfig) bool {
	if bcastCfg.Pattern != gobackend.BroadcastTrailingRHS && bcastCfg.Pattern != gobackend.BroadcastTrailingLHS {
		return false
	}
	archFn := gobackend.GetBinaryTrailingArchDispatcher()
	if archFn == nil {
		return false
	}
	isLHS := bcastCfg.Pattern == gobackend.BroadcastTrailingLHS
	dtype := lhs.RawShape.DType
	elemBytes := dtype.Size()
	lhsRowBytes := bcastCfg.B * elemBytes
	rhsRowBytes := elemBytes
	if isLHS {
		lhsRowBytes = elemBytes
		rhsRowBytes = bcastCfg.B * elemBytes
	}
	outRowBytes := bcastCfg.B * elemBytes
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() &&
		bcastCfg.A*bcastCfg.B >= minBinarySIMDParallelElements && bcastCfg.A > 1 &&
		elemBytes > 0 &&
		len(lhs.RawBytes) >= bcastCfg.A*lhsRowBytes &&
		len(rhs.RawBytes) >= bcastCfg.A*rhsRowBytes &&
		len(output.RawBytes) >= bcastCfg.A*outRowBytes {
		minA := max(1, binarySIMDChunkElements/max(1, bcastCfg.B))
		firstEnd := min(bcastCfg.A, minA)
		var firstLhs, firstRhs, firstOut gobackend.Buffer
		firstLhs.Flat = lhs.Flat
		firstRhs.Flat = rhs.Flat
		firstOut.Flat = output.Flat
		firstLhs.RawBytes = lhs.RawBytes[:firstEnd*lhsRowBytes]
		firstRhs.RawBytes = rhs.RawBytes[:firstEnd*rhsRowBytes]
		firstOut.RawBytes = output.RawBytes[:firstEnd*outRowBytes]
		if !archFn(op, isLHS, &firstLhs, &firstRhs, &firstOut, firstEnd, bcastCfg.B, dtype) {
			return false
		}
		if firstEnd < bcastCfg.A {
			backend.Scheduler.ParallelizeChunks(bcastCfg.A-firstEnd, minA, func(start, end int) {
				aStart := firstEnd + start
				aEnd := firstEnd + end
				var localLhs, localRhs, localOut gobackend.Buffer
				localLhs.Flat = lhs.Flat
				localRhs.Flat = rhs.Flat
				localOut.Flat = output.Flat
				localLhs.RawBytes = lhs.RawBytes[aStart*lhsRowBytes : aEnd*lhsRowBytes]
				localRhs.RawBytes = rhs.RawBytes[aStart*rhsRowBytes : aEnd*rhsRowBytes]
				localOut.RawBytes = output.RawBytes[aStart*outRowBytes : aEnd*outRowBytes]
				archFn(op, isLHS, &localLhs, &localRhs, &localOut, aEnd-aStart, bcastCfg.B, dtype)
			})
		}
		return true
	}
	return archFn(op, isLHS, lhs, rhs, output, bcastCfg.A, bcastCfg.B, dtype)
}

func execAddSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	if !node.IsExecutorCached() && !canExecuteBinarySIMD(node, inputs[0], inputs[1], supportedAddSubDTypes) {
		return nil, gobackend.ErrFallback
	}
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
	}
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)
	if tryBinaryTrailingArch(backend, compute.OpTypeAdd, lhs, rhs, output, bcastCfg) {
		return output, nil
	}

	var err error
	switch lhs.RawShape.DType {
	case dtypes.Float32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdAddVVFloat32, simdAddVSFloat32, func(c float32, r []float32, out []float32) { simdAddVSFloat32(r, c, out) },
			simdV2V1AddFloat32, simdV1V2AddFloat32)
	case dtypes.Float64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdAddVVFloat64, simdAddVSFloat64, func(c float64, r []float64, out []float64) { simdAddVSFloat64(r, c, out) },
			nil, nil)
	case dtypes.BFloat16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdAddVVBFloat16, simdAddVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) {
				simdAddVSBFloat16(r, c, out)
			},
			nil, nil)
	case dtypes.Float16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdAddVVFloat16, simdAddVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdAddVSFloat16(r, c, out) },
			nil, nil)
	case dtypes.Int32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdAddVVInt32, simdAddVSInt32, func(c int32, r []int32, out []int32) { simdAddVSInt32(r, c, out) },
			nil, nil)
	case dtypes.Uint32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdAddVVUint32, simdAddVSUint32, func(c uint32, r []uint32, out []uint32) { simdAddVSUint32(r, c, out) },
			nil, nil)
	case dtypes.Int64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), bcastCfg,
			simdAddVVInt64, simdAddVSInt64, func(c int64, r []int64, out []int64) { simdAddVSInt64(r, c, out) },
			nil, nil)
	case dtypes.Uint64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), bcastCfg,
			simdAddVVUint64, simdAddVSUint64, func(c uint64, r []uint64, out []uint64) { simdAddVSUint64(r, c, out) },
			nil, nil)
	default:
		return nil, gobackend.ErrFallback
	}
	return output, err
}

func execSubSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	if !node.IsExecutorCached() && !canExecuteBinarySIMD(node, inputs[0], inputs[1], supportedAddSubDTypes) {
		return nil, gobackend.ErrFallback
	}
	lhs, rhs, output, _, _ := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)
	if tryBinaryTrailingArch(backend, compute.OpTypeSub, lhs, rhs, output, bcastCfg) {
		return output, nil
	}

	var err error
	switch lhs.RawShape.DType {
	case dtypes.Float32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdSubVVFloat32, simdSubVSFloat32, simdSubSVFloat32,
			simdV2V1SubFloat32, simdV1V2SubFloat32)
	case dtypes.Float64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdSubVVFloat64, simdSubVSFloat64, simdSubSVFloat64,
			nil, nil)
	case dtypes.BFloat16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdSubVVBFloat16, simdSubVSBFloat16, simdSubSVBFloat16,
			nil, nil)
	case dtypes.Float16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdSubVVFloat16, simdSubVSFloat16, simdSubSVFloat16,
			nil, nil)
	case dtypes.Int32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdSubVVInt32, simdSubVSInt32, simdSubSVInt32,
			nil, nil)
	case dtypes.Uint32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdSubVVUint32, simdSubVSUint32, simdSubSVUint32,
			nil, nil)
	case dtypes.Int64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]int64), rhs.Flat.([]int64), output.Flat.([]int64), bcastCfg,
			simdSubVVInt64, simdSubVSInt64, simdSubSVInt64,
			nil, nil)
	case dtypes.Uint64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]uint64), rhs.Flat.([]uint64), output.Flat.([]uint64), bcastCfg,
			simdSubVVUint64, simdSubVSUint64, simdSubSVUint64,
			nil, nil)
	default:
		return nil, gobackend.ErrFallback
	}
	return output, err
}

func execMulSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	if !node.IsExecutorCached() && !canExecuteBinarySIMD(node, inputs[0], inputs[1], supportedMulMaxMinDTypes) {
		return nil, gobackend.ErrFallback
	}
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
	}
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)
	if tryBinaryTrailingArch(backend, compute.OpTypeMul, lhs, rhs, output, bcastCfg) {
		return output, nil
	}

	var err error
	switch lhs.RawShape.DType {
	case dtypes.Float32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdMulVVFloat32, simdMulVSFloat32, func(c float32, r []float32, out []float32) { simdMulVSFloat32(r, c, out) },
			simdV2V1MulFloat32, simdV1V2MulFloat32)
	case dtypes.Float64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdMulVVFloat64, simdMulVSFloat64, func(c float64, r []float64, out []float64) { simdMulVSFloat64(r, c, out) },
			nil, nil)
	case dtypes.BFloat16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdMulVVBFloat16, simdMulVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) {
				simdMulVSBFloat16(r, c, out)
			},
			nil, nil)
	case dtypes.Float16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdMulVVFloat16, simdMulVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdMulVSFloat16(r, c, out) },
			nil, nil)
	case dtypes.Int32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdMulVVInt32, simdMulVSInt32, func(c int32, r []int32, out []int32) { simdMulVSInt32(r, c, out) },
			nil, nil)
	case dtypes.Uint32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdMulVVUint32, simdMulVSUint32, func(c uint32, r []uint32, out []uint32) { simdMulVSUint32(r, c, out) },
			nil, nil)
	default:
		return nil, gobackend.ErrFallback
	}
	return output, err
}

func execDivSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	if !node.IsExecutorCached() && !canExecuteBinarySIMD(node, inputs[0], inputs[1], supportedDivDTypes) {
		return nil, gobackend.ErrFallback
	}
	lhs, rhs, output, _, _ := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)
	if tryBinaryTrailingArch(backend, compute.OpTypeDiv, lhs, rhs, output, bcastCfg) {
		return output, nil
	}

	var err error
	switch lhs.RawShape.DType {
	case dtypes.Float32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdDivVVFloat32, simdDivVSFloat32, simdDivSVFloat32,
			simdV2V1DivFloat32, simdV1V2DivFloat32)
	case dtypes.Float64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdDivVVFloat64, simdDivVSFloat64, simdDivSVFloat64,
			nil, nil)
	case dtypes.BFloat16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdDivVVBFloat16, simdDivVSBFloat16, simdDivSVBFloat16,
			nil, nil)
	case dtypes.Float16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdDivVVFloat16, simdDivVSFloat16, simdDivSVFloat16,
			nil, nil)
	default:
		return nil, gobackend.ErrFallback
	}
	return output, err
}

func execMaxSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	if !node.IsExecutorCached() && !canExecuteBinarySIMD(node, inputs[0], inputs[1], supportedMulMaxMinDTypes) {
		return nil, gobackend.ErrFallback
	}
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
	}
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)
	if tryBinaryTrailingArch(backend, compute.OpTypeMax, lhs, rhs, output, bcastCfg) {
		return output, nil
	}

	var err error
	switch lhs.RawShape.DType {
	case dtypes.Float32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdMaxVVFloat32, simdMaxVSFloat32, func(c float32, r []float32, out []float32) { simdMaxVSFloat32(r, c, out) },
			simdV2V1MaxFloat32, simdV1V2MaxFloat32)
	case dtypes.Float64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdMaxVVFloat64, simdMaxVSFloat64, func(c float64, r []float64, out []float64) { simdMaxVSFloat64(r, c, out) },
			nil, nil)
	case dtypes.BFloat16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdMaxVVBFloat16, simdMaxVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) {
				simdMaxVSBFloat16(r, c, out)
			},
			nil, nil)
	case dtypes.Float16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdMaxVVFloat16, simdMaxVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdMaxVSFloat16(r, c, out) },
			nil, nil)
	case dtypes.Int32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdMaxVVInt32, simdMaxVSInt32, func(c int32, r []int32, out []int32) { simdMaxVSInt32(r, c, out) },
			nil, nil)
	case dtypes.Uint32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdMaxVVUint32, simdMaxVSUint32, func(c uint32, r []uint32, out []uint32) { simdMaxVSUint32(r, c, out) },
			nil, nil)
	default:
		return nil, gobackend.ErrFallback
	}
	return output, err
}

func execMinSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	if !node.IsExecutorCached() && !canExecuteBinarySIMD(node, inputs[0], inputs[1], supportedMulMaxMinDTypes) {
		return nil, gobackend.ErrFallback
	}
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutputForExecution(backend, node, inputs, inputsOwned, node.Shape)
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
	}
	if backend.NoOps {
		return output, nil
	}
	bcastCfg := GetBroadcastConfig(node, lhs, rhs, output)
	if tryBinaryTrailingArch(backend, compute.OpTypeMin, lhs, rhs, output, bcastCfg) {
		return output, nil
	}

	var err error
	switch lhs.RawShape.DType {
	case dtypes.Float32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float32), rhs.Flat.([]float32), output.Flat.([]float32), bcastCfg,
			simdMinVVFloat32, simdMinVSFloat32, func(c float32, r []float32, out []float32) { simdMinVSFloat32(r, c, out) },
			simdV2V1MinFloat32, simdV1V2MinFloat32)
	case dtypes.Float64:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float64), rhs.Flat.([]float64), output.Flat.([]float64), bcastCfg,
			simdMinVVFloat64, simdMinVSFloat64, func(c float64, r []float64, out []float64) { simdMinVSFloat64(r, c, out) },
			nil, nil)
	case dtypes.BFloat16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]bfloat16.BFloat16), rhs.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), bcastCfg,
			simdMinVVBFloat16, simdMinVSBFloat16, func(c bfloat16.BFloat16, r []bfloat16.BFloat16, out []bfloat16.BFloat16) {
				simdMinVSBFloat16(r, c, out)
			},
			nil, nil)
	case dtypes.Float16:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]float16.Float16), rhs.Flat.([]float16.Float16), output.Flat.([]float16.Float16), bcastCfg,
			simdMinVVFloat16, simdMinVSFloat16, func(c float16.Float16, r []float16.Float16, out []float16.Float16) { simdMinVSFloat16(r, c, out) },
			nil, nil)
	case dtypes.Int32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]int32), rhs.Flat.([]int32), output.Flat.([]int32), bcastCfg,
			simdMinVVInt32, simdMinVSInt32, func(c int32, r []int32, out []int32) { simdMinVSInt32(r, c, out) },
			nil, nil)
	case dtypes.Uint32:
		err = dispatchBinarySIMD(backend, lhs.Flat.([]uint32), rhs.Flat.([]uint32), output.Flat.([]uint32), bcastCfg,
			simdMinVVUint32, simdMinVSUint32, func(c uint32, r []uint32, out []uint32) { simdMinVSUint32(r, c, out) },
			nil, nil)
	default:
		return nil, gobackend.ErrFallback
	}
	return output, err
}

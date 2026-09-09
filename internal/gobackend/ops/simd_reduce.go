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
	"github.com/gomlx/compute/support/sets"
)

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeReduceSum, PrioritySIMD, execReduceSumSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeReduceMax, PrioritySIMD, execReduceMaxSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeReduceMin, PrioritySIMD, execReduceMinSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeReduceProduct, PrioritySIMD, execReduceProductSIMD)
}

var supportedReduceDTypes = []dtypes.DType{
	dtypes.Float32, dtypes.Float64,
	dtypes.Float16, dtypes.BFloat16,
	dtypes.Int32, dtypes.Uint32,
	dtypes.Int8, dtypes.Uint8,
	dtypes.Int16, dtypes.Uint16,
}

func canExecuteReduceSIMD(cfg ReduceConfig, dtype dtypes.DType, supported []dtypes.DType) bool {
	if !slices.Contains(supported, dtype) {
		return false
	}
	switch cfg.Pattern {
	case ReduceTrailing:
		minB := reduceThresholds.TrailingMinB[dtype]
		if minB == ThresholdAlwaysFallBack || (minB > 0 && cfg.B < minB) {
			return false
		}
		return true
	case ReduceLeading:
		minB := reduceThresholds.LeadingMinB[dtype]
		if minB == ThresholdAlwaysFallBack || (minB > 0 && cfg.B < minB) {
			return false
		}
		return true
	case ReduceAll:
		minN := reduceThresholds.AllMinN[dtype]
		if minN == ThresholdAlwaysFallBack || (minN > 0 && (cfg.A*cfg.B) < minN) {
			return false
		}
		return true
	default:
		return false
	}
}

func prepareReduceBuffers(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer) (*gobackend.Buffer, *gobackend.Buffer, ReduceConfig, error) {
	operand := inputs[0]
	var reduceAxes []int
	var cfg ReduceConfig
	if node != nil && node.Data != nil {
		switch d := node.Data.(type) {
		case *ReduceConfig:
			cfg = *d
			reduceAxes = cfg.Axes
		case ReduceConfig:
			cfg = d
			reduceAxes = cfg.Axes
		case []int:
			reduceAxes = d
			cfg = DetermineReduceConfig(operand.RawShape, reduceAxes)
		}
	} else {
		cfg = DetermineReduceConfig(operand.RawShape, reduceAxes)
		reduceAxes = cfg.Axes
	}
	if len(reduceAxes) == 0 && operand.RawShape.Rank() > 0 {
		for axis := range operand.RawShape.Rank() {
			reduceAxes = append(reduceAxes, axis)
		}
		cfg = DetermineReduceConfig(operand.RawShape, reduceAxes)
	}
	outputShape := node.Shape
	if outputShape.IsDynamic() {
		concreteShape := outputShape.Clone()
		axesSet := sets.MakeWith(reduceAxes...)
		outIdx := 0
		for axis, dim := range operand.RawShape.Dimensions {
			if !axesSet.Has(axis) {
				concreteShape.Dimensions[outIdx] = dim
				outIdx++
			}
		}
		outputShape = concreteShape
		cfg = DetermineReduceConfig(operand.RawShape, reduceAxes)
	}
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, nil, cfg, err
	}
	return operand, output, cfg, nil
}

func simdReduceLeadingSumFloat32(in, out []float32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Add(simd.LoadFloat32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat32s(out[:B])
		vAcc1 := simd.LoadFloat32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Add(simd.LoadFloat32s(in[off:]))
			vAcc1 = vAcc1.Add(simd.LoadFloat32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat32s(out[i:])
			vIn := simd.LoadFloat32s(in[off+i:])
			vOut.Add(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat32sPart(out[i:B])
			vIn, _ := simd.LoadFloat32sPart(in[off+i : off+B])
			vOut.Add(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingSumFloat32(in []float32, out []float32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float32(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res += v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res += v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Add(simd.LoadFloat32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res += v
		}
		for ; i < B; i++ {
			v := row[i]
			res += v
		}
		out[a] = res
	}
}

func simdReduceAllSumFloat32(in []float32, out []float32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res += v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	vAcc0 := simd.LoadFloat32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat32s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Add(simd.LoadFloat32s(in[i:]))
			vAcc1 = vAcc1.Add(simd.LoadFloat32s(in[i+vLen:]))
			vAcc2 = vAcc2.Add(simd.LoadFloat32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Add(simd.LoadFloat32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Add(vAcc1).Add(vAcc2.Add(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Add(simd.LoadFloat32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res += v
	}
	for ; i < n; i++ {
		v := in[i]
		res += v
	}
	out[0] = res
}

func simdReduceLeadingMaxFloat32(in, out []float32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Max(simd.LoadFloat32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat32s(out[:B])
		vAcc1 := simd.LoadFloat32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Max(simd.LoadFloat32s(in[off:]))
			vAcc1 = vAcc1.Max(simd.LoadFloat32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat32s(out[i:])
			vIn := simd.LoadFloat32s(in[off+i:])
			vOut.Max(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat32sPart(out[i:B])
			vIn, _ := simd.LoadFloat32sPart(in[off+i : off+B])
			vOut.Max(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMaxFloat32(in []float32, out []float32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float32(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Max(simd.LoadFloat32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = max(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = max(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMaxFloat32(in []float32, out []float32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(float32)
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = max(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	vAcc0 := simd.LoadFloat32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat32s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Max(simd.LoadFloat32s(in[i:]))
			vAcc1 = vAcc1.Max(simd.LoadFloat32s(in[i+vLen:]))
			vAcc2 = vAcc2.Max(simd.LoadFloat32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Max(simd.LoadFloat32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Max(vAcc1).Max(vAcc2.Max(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Max(simd.LoadFloat32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = max(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = max(res, v)
	}
	out[0] = res
}

func simdReduceLeadingMinFloat32(in, out []float32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Min(simd.LoadFloat32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat32s(out[:B])
		vAcc1 := simd.LoadFloat32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Min(simd.LoadFloat32s(in[off:]))
			vAcc1 = vAcc1.Min(simd.LoadFloat32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat32s(out[i:])
			vIn := simd.LoadFloat32s(in[off+i:])
			vOut.Min(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat32sPart(out[i:B])
			vIn, _ := simd.LoadFloat32sPart(in[off+i : off+B])
			vOut.Min(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMinFloat32(in []float32, out []float32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float32(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Min(simd.LoadFloat32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = min(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = min(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMinFloat32(in []float32, out []float32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(float32)
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = min(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	vAcc0 := simd.LoadFloat32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat32s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Min(simd.LoadFloat32s(in[i:]))
			vAcc1 = vAcc1.Min(simd.LoadFloat32s(in[i+vLen:]))
			vAcc2 = vAcc2.Min(simd.LoadFloat32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Min(simd.LoadFloat32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Min(vAcc1).Min(vAcc2.Min(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Min(simd.LoadFloat32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = min(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = min(res, v)
	}
	out[0] = res
}

func simdReduceLeadingProductFloat32(in, out []float32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Mul(simd.LoadFloat32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat32s(out[:B])
		vAcc1 := simd.LoadFloat32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Mul(simd.LoadFloat32s(in[off:]))
			vAcc1 = vAcc1.Mul(simd.LoadFloat32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat32s(out[i:])
			vIn := simd.LoadFloat32s(in[off+i:])
			vOut.Mul(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat32sPart(out[i:B])
			vIn, _ := simd.LoadFloat32sPart(in[off+i : off+B])
			vOut.Mul(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingProductFloat32(in []float32, out []float32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float32(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res *= v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res *= v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Mul(simd.LoadFloat32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res *= v
		}
		for ; i < B; i++ {
			v := row[i]
			res *= v
		}
		out[a] = res
	}
}

func simdReduceAllProductFloat32(in []float32, out []float32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	vDummy := simd.BroadcastFloat32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res *= v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float32
	var tmp []float32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float32, vLen)
	}
	vAcc0 := simd.LoadFloat32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat32s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Mul(simd.LoadFloat32s(in[i:]))
			vAcc1 = vAcc1.Mul(simd.LoadFloat32s(in[i+vLen:]))
			vAcc2 = vAcc2.Mul(simd.LoadFloat32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Mul(simd.LoadFloat32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Mul(vAcc1).Mul(vAcc2.Mul(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Mul(simd.LoadFloat32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res *= v
	}
	for ; i < n; i++ {
		v := in[i]
		res *= v
	}
	out[0] = res
}

func simdReduceLeadingSumFloat64(in, out []float64, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat64s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Add(simd.LoadFloat64s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat64s(out[:B])
		vAcc1 := simd.LoadFloat64s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Add(simd.LoadFloat64s(in[off:]))
			vAcc1 = vAcc1.Add(simd.LoadFloat64s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat64s(out[i:])
			vIn := simd.LoadFloat64s(in[off+i:])
			vOut.Add(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat64sPart(out[i:B])
			vIn, _ := simd.LoadFloat64sPart(in[off+i : off+B])
			vOut.Add(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingSumFloat64(in []float64, out []float64, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float64(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res += v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat64sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res += v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat64s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Add(simd.LoadFloat64s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res += v
		}
		for ; i < B; i++ {
			v := row[i]
			res += v
		}
		out[a] = res
	}
}

func simdReduceAllSumFloat64(in []float64, out []float64, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res += v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	vAcc0 := simd.LoadFloat64s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat64s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat64s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat64s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Add(simd.LoadFloat64s(in[i:]))
			vAcc1 = vAcc1.Add(simd.LoadFloat64s(in[i+vLen:]))
			vAcc2 = vAcc2.Add(simd.LoadFloat64s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Add(simd.LoadFloat64s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Add(vAcc1).Add(vAcc2.Add(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Add(simd.LoadFloat64s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res += v
	}
	for ; i < n; i++ {
		v := in[i]
		res += v
	}
	out[0] = res
}

func simdReduceLeadingMaxFloat64(in, out []float64, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat64s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Max(simd.LoadFloat64s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat64s(out[:B])
		vAcc1 := simd.LoadFloat64s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Max(simd.LoadFloat64s(in[off:]))
			vAcc1 = vAcc1.Max(simd.LoadFloat64s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat64s(out[i:])
			vIn := simd.LoadFloat64s(in[off+i:])
			vOut.Max(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat64sPart(out[i:B])
			vIn, _ := simd.LoadFloat64sPart(in[off+i : off+B])
			vOut.Max(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMaxFloat64(in []float64, out []float64, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float64(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat64sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat64s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Max(simd.LoadFloat64s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = max(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = max(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMaxFloat64(in []float64, out []float64, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(float64)
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = max(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	vAcc0 := simd.LoadFloat64s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat64s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat64s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat64s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Max(simd.LoadFloat64s(in[i:]))
			vAcc1 = vAcc1.Max(simd.LoadFloat64s(in[i+vLen:]))
			vAcc2 = vAcc2.Max(simd.LoadFloat64s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Max(simd.LoadFloat64s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Max(vAcc1).Max(vAcc2.Max(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Max(simd.LoadFloat64s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = max(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = max(res, v)
	}
	out[0] = res
}

func simdReduceLeadingMinFloat64(in, out []float64, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat64s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Min(simd.LoadFloat64s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat64s(out[:B])
		vAcc1 := simd.LoadFloat64s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Min(simd.LoadFloat64s(in[off:]))
			vAcc1 = vAcc1.Min(simd.LoadFloat64s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat64s(out[i:])
			vIn := simd.LoadFloat64s(in[off+i:])
			vOut.Min(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat64sPart(out[i:B])
			vIn, _ := simd.LoadFloat64sPart(in[off+i : off+B])
			vOut.Min(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMinFloat64(in []float64, out []float64, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float64(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat64sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat64s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Min(simd.LoadFloat64s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = min(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = min(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMinFloat64(in []float64, out []float64, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(float64)
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = min(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	vAcc0 := simd.LoadFloat64s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat64s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat64s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat64s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Min(simd.LoadFloat64s(in[i:]))
			vAcc1 = vAcc1.Min(simd.LoadFloat64s(in[i+vLen:]))
			vAcc2 = vAcc2.Min(simd.LoadFloat64s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Min(simd.LoadFloat64s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Min(vAcc1).Min(vAcc2.Min(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Min(simd.LoadFloat64s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = min(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = min(res, v)
	}
	out[0] = res
}

func simdReduceLeadingProductFloat64(in, out []float64, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadFloat64s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Mul(simd.LoadFloat64s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadFloat64s(out[:B])
		vAcc1 := simd.LoadFloat64s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Mul(simd.LoadFloat64s(in[off:]))
			vAcc1 = vAcc1.Mul(simd.LoadFloat64s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadFloat64s(out[i:])
			vIn := simd.LoadFloat64s(in[off+i:])
			vOut.Mul(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadFloat64sPart(out[i:B])
			vIn, _ := simd.LoadFloat64sPart(in[off+i : off+B])
			vOut.Mul(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingProductFloat64(in []float64, out []float64, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float64(0)
		}
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res *= v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadFloat64sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res *= v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadFloat64s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Mul(simd.LoadFloat64s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res *= v
		}
		for ; i < B; i++ {
			v := row[i]
			res *= v
		}
		out[a] = res
	}
}

func simdReduceAllProductFloat64(in []float64, out []float64, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	vDummy := simd.BroadcastFloat64s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res *= v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]float64
	var tmp []float64
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]float64, vLen)
	}
	vAcc0 := simd.LoadFloat64s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadFloat64s(in[i : i+vLen])
		vAcc2 := simd.LoadFloat64s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadFloat64s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Mul(simd.LoadFloat64s(in[i:]))
			vAcc1 = vAcc1.Mul(simd.LoadFloat64s(in[i+vLen:]))
			vAcc2 = vAcc2.Mul(simd.LoadFloat64s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Mul(simd.LoadFloat64s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Mul(vAcc1).Mul(vAcc2.Mul(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Mul(simd.LoadFloat64s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res *= v
	}
	for ; i < n; i++ {
		v := in[i]
		res *= v
	}
	out[0] = res
}

func simdReduceLeadingSumInt32(in, out []int32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadInt32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Add(simd.LoadInt32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadInt32s(out[:B])
		vAcc1 := simd.LoadInt32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Add(simd.LoadInt32s(in[off:]))
			vAcc1 = vAcc1.Add(simd.LoadInt32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadInt32s(out[i:])
			vIn := simd.LoadInt32s(in[off+i:])
			vOut.Add(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadInt32sPart(out[i:B])
			vIn, _ := simd.LoadInt32sPart(in[off+i : off+B])
			vOut.Add(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingSumInt32(in []int32, out []int32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int32(0)
		}
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res += v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadInt32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res += v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadInt32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Add(simd.LoadInt32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res += v
		}
		for ; i < B; i++ {
			v := row[i]
			res += v
		}
		out[a] = res
	}
}

func simdReduceAllSumInt32(in []int32, out []int32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res += v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	vAcc0 := simd.LoadInt32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadInt32s(in[i : i+vLen])
		vAcc2 := simd.LoadInt32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadInt32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Add(simd.LoadInt32s(in[i:]))
			vAcc1 = vAcc1.Add(simd.LoadInt32s(in[i+vLen:]))
			vAcc2 = vAcc2.Add(simd.LoadInt32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Add(simd.LoadInt32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Add(vAcc1).Add(vAcc2.Add(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Add(simd.LoadInt32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res += v
	}
	for ; i < n; i++ {
		v := in[i]
		res += v
	}
	out[0] = res
}

func simdReduceLeadingMaxInt32(in, out []int32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadInt32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Max(simd.LoadInt32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadInt32s(out[:B])
		vAcc1 := simd.LoadInt32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Max(simd.LoadInt32s(in[off:]))
			vAcc1 = vAcc1.Max(simd.LoadInt32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadInt32s(out[i:])
			vIn := simd.LoadInt32s(in[off+i:])
			vOut.Max(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadInt32sPart(out[i:B])
			vIn, _ := simd.LoadInt32sPart(in[off+i : off+B])
			vOut.Max(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMaxInt32(in []int32, out []int32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int32(0)
		}
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadInt32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadInt32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Max(simd.LoadInt32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = max(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = max(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMaxInt32(in []int32, out []int32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(int32)
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = max(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	vAcc0 := simd.LoadInt32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadInt32s(in[i : i+vLen])
		vAcc2 := simd.LoadInt32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadInt32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Max(simd.LoadInt32s(in[i:]))
			vAcc1 = vAcc1.Max(simd.LoadInt32s(in[i+vLen:]))
			vAcc2 = vAcc2.Max(simd.LoadInt32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Max(simd.LoadInt32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Max(vAcc1).Max(vAcc2.Max(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Max(simd.LoadInt32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = max(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = max(res, v)
	}
	out[0] = res
}

func simdReduceLeadingMinInt32(in, out []int32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadInt32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Min(simd.LoadInt32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadInt32s(out[:B])
		vAcc1 := simd.LoadInt32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Min(simd.LoadInt32s(in[off:]))
			vAcc1 = vAcc1.Min(simd.LoadInt32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadInt32s(out[i:])
			vIn := simd.LoadInt32s(in[off+i:])
			vOut.Min(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadInt32sPart(out[i:B])
			vIn, _ := simd.LoadInt32sPart(in[off+i : off+B])
			vOut.Min(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMinInt32(in []int32, out []int32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int32(0)
		}
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadInt32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadInt32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Min(simd.LoadInt32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = min(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = min(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMinInt32(in []int32, out []int32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(int32)
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = min(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	vAcc0 := simd.LoadInt32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadInt32s(in[i : i+vLen])
		vAcc2 := simd.LoadInt32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadInt32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Min(simd.LoadInt32s(in[i:]))
			vAcc1 = vAcc1.Min(simd.LoadInt32s(in[i+vLen:]))
			vAcc2 = vAcc2.Min(simd.LoadInt32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Min(simd.LoadInt32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Min(vAcc1).Min(vAcc2.Min(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Min(simd.LoadInt32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = min(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = min(res, v)
	}
	out[0] = res
}

func simdReduceLeadingProductInt32(in, out []int32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadInt32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Mul(simd.LoadInt32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadInt32s(out[:B])
		vAcc1 := simd.LoadInt32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Mul(simd.LoadInt32s(in[off:]))
			vAcc1 = vAcc1.Mul(simd.LoadInt32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadInt32s(out[i:])
			vIn := simd.LoadInt32s(in[off+i:])
			vOut.Mul(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadInt32sPart(out[i:B])
			vIn, _ := simd.LoadInt32sPart(in[off+i : off+B])
			vOut.Mul(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingProductInt32(in []int32, out []int32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int32(0)
		}
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res *= v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadInt32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res *= v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadInt32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Mul(simd.LoadInt32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res *= v
		}
		for ; i < B; i++ {
			v := row[i]
			res *= v
		}
		out[a] = res
	}
}

func simdReduceAllProductInt32(in []int32, out []int32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	vDummy := simd.BroadcastInt32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res *= v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]int32
	var tmp []int32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]int32, vLen)
	}
	vAcc0 := simd.LoadInt32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadInt32s(in[i : i+vLen])
		vAcc2 := simd.LoadInt32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadInt32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Mul(simd.LoadInt32s(in[i:]))
			vAcc1 = vAcc1.Mul(simd.LoadInt32s(in[i+vLen:]))
			vAcc2 = vAcc2.Mul(simd.LoadInt32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Mul(simd.LoadInt32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Mul(vAcc1).Mul(vAcc2.Mul(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Mul(simd.LoadInt32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res *= v
	}
	for ; i < n; i++ {
		v := in[i]
		res *= v
	}
	out[0] = res
}

func simdReduceLeadingSumUint32(in, out []uint32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadUint32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Add(simd.LoadUint32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadUint32s(out[:B])
		vAcc1 := simd.LoadUint32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Add(simd.LoadUint32s(in[off:]))
			vAcc1 = vAcc1.Add(simd.LoadUint32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadUint32s(out[i:])
			vIn := simd.LoadUint32s(in[off+i:])
			vOut.Add(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadUint32sPart(out[i:B])
			vIn, _ := simd.LoadUint32sPart(in[off+i : off+B])
			vOut.Add(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingSumUint32(in []uint32, out []uint32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint32(0)
		}
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res += v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadUint32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res += v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadUint32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Add(simd.LoadUint32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res += v
		}
		for ; i < B; i++ {
			v := row[i]
			res += v
		}
		out[a] = res
	}
}

func simdReduceAllSumUint32(in []uint32, out []uint32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res += v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	vAcc0 := simd.LoadUint32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadUint32s(in[i : i+vLen])
		vAcc2 := simd.LoadUint32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadUint32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Add(simd.LoadUint32s(in[i:]))
			vAcc1 = vAcc1.Add(simd.LoadUint32s(in[i+vLen:]))
			vAcc2 = vAcc2.Add(simd.LoadUint32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Add(simd.LoadUint32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Add(vAcc1).Add(vAcc2.Add(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Add(simd.LoadUint32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res += v
	}
	for ; i < n; i++ {
		v := in[i]
		res += v
	}
	out[0] = res
}

func simdReduceLeadingMaxUint32(in, out []uint32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadUint32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Max(simd.LoadUint32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadUint32s(out[:B])
		vAcc1 := simd.LoadUint32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Max(simd.LoadUint32s(in[off:]))
			vAcc1 = vAcc1.Max(simd.LoadUint32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadUint32s(out[i:])
			vIn := simd.LoadUint32s(in[off+i:])
			vOut.Max(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadUint32sPart(out[i:B])
			vIn, _ := simd.LoadUint32sPart(in[off+i : off+B])
			vOut.Max(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMaxUint32(in []uint32, out []uint32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint32(0)
		}
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadUint32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = max(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadUint32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Max(simd.LoadUint32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = max(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = max(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMaxUint32(in []uint32, out []uint32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(uint32)
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = max(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	vAcc0 := simd.LoadUint32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadUint32s(in[i : i+vLen])
		vAcc2 := simd.LoadUint32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadUint32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Max(simd.LoadUint32s(in[i:]))
			vAcc1 = vAcc1.Max(simd.LoadUint32s(in[i+vLen:]))
			vAcc2 = vAcc2.Max(simd.LoadUint32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Max(simd.LoadUint32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Max(vAcc1).Max(vAcc2.Max(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Max(simd.LoadUint32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = max(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = max(res, v)
	}
	out[0] = res
}

func simdReduceLeadingMinUint32(in, out []uint32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadUint32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Min(simd.LoadUint32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadUint32s(out[:B])
		vAcc1 := simd.LoadUint32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Min(simd.LoadUint32s(in[off:]))
			vAcc1 = vAcc1.Min(simd.LoadUint32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadUint32s(out[i:])
			vIn := simd.LoadUint32s(in[off+i:])
			vOut.Min(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadUint32sPart(out[i:B])
			vIn, _ := simd.LoadUint32sPart(in[off+i : off+B])
			vOut.Min(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingMinUint32(in []uint32, out []uint32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint32(0)
		}
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadUint32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res = min(res, v)
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadUint32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Min(simd.LoadUint32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res = min(res, v)
		}
		for ; i < B; i++ {
			v := row[i]
			res = min(res, v)
		}
		out[a] = res
	}
}

func simdReduceAllMinUint32(in []uint32, out []uint32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(uint32)
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res = min(res, v)
		}
		out[0] = res
		return
	}
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	vAcc0 := simd.LoadUint32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadUint32s(in[i : i+vLen])
		vAcc2 := simd.LoadUint32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadUint32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Min(simd.LoadUint32s(in[i:]))
			vAcc1 = vAcc1.Min(simd.LoadUint32s(in[i+vLen:]))
			vAcc2 = vAcc2.Min(simd.LoadUint32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Min(simd.LoadUint32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Min(vAcc1).Min(vAcc2.Min(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Min(simd.LoadUint32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res = min(res, v)
	}
	for ; i < n; i++ {
		v := in[i]
		res = min(res, v)
	}
	out[0] = res
}

func simdReduceLeadingProductUint32(in, out []uint32, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	copy(out[:B], in[:B])
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if B == vLen {
		vAcc := simd.LoadUint32s(out[:B])
		for a := 1; a < A; a++ {
			vAcc = vAcc.Mul(simd.LoadUint32s(in[a*B:]))
		}
		vAcc.Store(out[:B])
		return
	}
	if B == 2*vLen {
		vAcc0 := simd.LoadUint32s(out[:B])
		vAcc1 := simd.LoadUint32s(out[vLen:B])
		for a := 1; a < A; a++ {
			off := a * B
			vAcc0 = vAcc0.Mul(simd.LoadUint32s(in[off:]))
			vAcc1 = vAcc1.Mul(simd.LoadUint32s(in[off+vLen:]))
		}
		vAcc0.Store(out[:B])
		vAcc1.Store(out[vLen:B])
		return
	}
	for a := 1; a < A; a++ {
		off := a * B
		i := 0
		for ; i+vLen <= B; i += vLen {
			vOut := simd.LoadUint32s(out[i:])
			vIn := simd.LoadUint32s(in[off+i:])
			vOut.Mul(vIn).Store(out[i:])
		}
		if i < B {
			vOut, _ := simd.LoadUint32sPart(out[i:B])
			vIn, _ := simd.LoadUint32sPart(in[off+i : off+B])
			vOut.Mul(vIn).StorePart(out[i:B])
		}
	}
}

func simdReduceTrailingProductUint32(in []uint32, out []uint32, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint32(0)
		}
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		if B <= 4 {
			res := row[0]
			for _, v := range row[1:] {
				res *= v
			}
			out[a] = res
			continue
		}
		if B < vLen {
			v, _ := simd.LoadUint32sPart(row)
			v.Store(tmp)
			res := tmp[0]
			for _, v := range tmp[1:B] {
				res *= v
			}
			out[a] = res
			continue
		}
		vAcc := simd.LoadUint32s(row[:vLen])
		i := vLen
		for ; i+vLen <= B; i += vLen {
			vAcc = vAcc.Mul(simd.LoadUint32s(row[i:]))
		}
		vAcc.Store(tmp)
		res := tmp[0]
		for _, v := range tmp[1:vLen] {
			res *= v
		}
		for ; i < B; i++ {
			v := row[i]
			res *= v
		}
		out[a] = res
	}
}

func simdReduceAllProductUint32(in []uint32, out []uint32, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	vDummy := simd.BroadcastUint32s(0)
	vLen := vDummy.Len()
	if n <= 4 || n < vLen {
		res := in[0]
		for _, v := range in[1:] {
			res *= v
		}
		out[0] = res
		return
	}
	var tmpBuf [64]uint32
	var tmp []uint32
	if vLen <= len(tmpBuf) {
		tmp = tmpBuf[:vLen]
	} else {
		tmp = make([]uint32, vLen)
	}
	vAcc0 := simd.LoadUint32s(in[:vLen])
	i := vLen
	if i+3*vLen <= n {
		vAcc1 := simd.LoadUint32s(in[i : i+vLen])
		vAcc2 := simd.LoadUint32s(in[i+vLen : i+2*vLen])
		vAcc3 := simd.LoadUint32s(in[i+2*vLen : i+3*vLen])
		i += 3 * vLen
		for ; i+4*vLen <= n; i += 4 * vLen {
			vAcc0 = vAcc0.Mul(simd.LoadUint32s(in[i:]))
			vAcc1 = vAcc1.Mul(simd.LoadUint32s(in[i+vLen:]))
			vAcc2 = vAcc2.Mul(simd.LoadUint32s(in[i+2*vLen:]))
			vAcc3 = vAcc3.Mul(simd.LoadUint32s(in[i+3*vLen:]))
		}
		vAcc0 = vAcc0.Mul(vAcc1).Mul(vAcc2.Mul(vAcc3))
	}
	for ; i+vLen <= n; i += vLen {
		vAcc0 = vAcc0.Mul(simd.LoadUint32s(in[i:]))
	}
	vAcc0.Store(tmp)
	res := tmp[0]
	for _, v := range tmp[1:vLen] {
		res *= v
	}
	for ; i < n; i++ {
		v := in[i]
		res *= v
	}
	out[0] = res
}

func simdReduceLeadingSumFloat16(in, out []float16.Float16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] += v
		}
	}
	for b := range B {
		out[b] = float16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingSumFloat16(in []float16.Float16, out []float16.Float16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res += vf
		}
		out[a] = float16.FromFloat32(res)
	}
}

func simdReduceAllSumFloat16(in []float16.Float16, out []float16.Float16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = float16.FromFloat32(0)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res += vf
	}
	out[0] = float16.FromFloat32(res)
}

func simdReduceLeadingMaxFloat16(in, out []float16.Float16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] = max(acc[b], v)
		}
	}
	for b := range B {
		out[b] = float16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingMaxFloat16(in []float16.Float16, out []float16.Float16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res = max(res, vf)
		}
		out[a] = float16.FromFloat32(res)
	}
}

func simdReduceAllMaxFloat16(in []float16.Float16, out []float16.Float16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(float16.Float16)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res = max(res, vf)
	}
	out[0] = float16.FromFloat32(res)
}

func simdReduceLeadingMinFloat16(in, out []float16.Float16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] = min(acc[b], v)
		}
	}
	for b := range B {
		out[b] = float16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingMinFloat16(in []float16.Float16, out []float16.Float16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res = min(res, vf)
		}
		out[a] = float16.FromFloat32(res)
	}
}

func simdReduceAllMinFloat16(in []float16.Float16, out []float16.Float16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(float16.Float16)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res = min(res, vf)
	}
	out[0] = float16.FromFloat32(res)
}

func simdReduceLeadingProductFloat16(in, out []float16.Float16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] *= v
		}
	}
	for b := range B {
		out[b] = float16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingProductFloat16(in []float16.Float16, out []float16.Float16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = float16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res *= vf
		}
		out[a] = float16.FromFloat32(res)
	}
}

func simdReduceAllProductFloat16(in []float16.Float16, out []float16.Float16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = float16.FromFloat32(1)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res *= vf
	}
	out[0] = float16.FromFloat32(res)
}

func simdReduceLeadingSumBFloat16(in, out []bfloat16.BFloat16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] += v
		}
	}
	for b := range B {
		out[b] = bfloat16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingSumBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = bfloat16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res += vf
		}
		out[a] = bfloat16.FromFloat32(res)
	}
}

func simdReduceAllSumBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = bfloat16.FromFloat32(0)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res += vf
	}
	out[0] = bfloat16.FromFloat32(res)
}

func simdReduceLeadingMaxBFloat16(in, out []bfloat16.BFloat16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] = max(acc[b], v)
		}
	}
	for b := range B {
		out[b] = bfloat16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingMaxBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = bfloat16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res = max(res, vf)
		}
		out[a] = bfloat16.FromFloat32(res)
	}
}

func simdReduceAllMaxBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(bfloat16.BFloat16)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res = max(res, vf)
	}
	out[0] = bfloat16.FromFloat32(res)
}

func simdReduceLeadingMinBFloat16(in, out []bfloat16.BFloat16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] = min(acc[b], v)
		}
	}
	for b := range B {
		out[b] = bfloat16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingMinBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = bfloat16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res = min(res, vf)
		}
		out[a] = bfloat16.FromFloat32(res)
	}
}

func simdReduceAllMinBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(bfloat16.BFloat16)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res = min(res, vf)
	}
	out[0] = bfloat16.FromFloat32(res)
}

func simdReduceLeadingProductBFloat16(in, out []bfloat16.BFloat16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]float32, B)
	for b := range B {
		acc[b] = in[b].Float32()
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := row[b].Float32()
			acc[b] *= v
		}
	}
	for b := range B {
		out[b] = bfloat16.FromFloat32(acc[b])
	}
}

func simdReduceTrailingProductBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = bfloat16.FromFloat32(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := row[0].Float32()
		for _, v := range row[1:] {
			vf := v.Float32()
			res *= vf
		}
		out[a] = bfloat16.FromFloat32(res)
	}
}

func simdReduceAllProductBFloat16(in []bfloat16.BFloat16, out []bfloat16.BFloat16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = bfloat16.FromFloat32(1)
		return
	}
	res := in[0].Float32()
	for _, v := range in[1:] {
		vf := v.Float32()
		res *= vf
	}
	out[0] = bfloat16.FromFloat32(res)
}

func simdReduceLeadingSumInt8(in, out []int8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] += v
		}
	}
	for b := range B {
		out[b] = int8(acc[b])
	}
}

func simdReduceTrailingSumInt8(in []int8, out []int8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res += vi
		}
		out[a] = int8(res)
	}
}

func simdReduceAllSumInt8(in []int8, out []int8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res += vi
	}
	out[0] = int8(res)
}

func simdReduceLeadingMaxInt8(in, out []int8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = max(acc[b], v)
		}
	}
	for b := range B {
		out[b] = int8(acc[b])
	}
}

func simdReduceTrailingMaxInt8(in []int8, out []int8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = max(res, vi)
		}
		out[a] = int8(res)
	}
}

func simdReduceAllMaxInt8(in []int8, out []int8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(int8)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = max(res, vi)
	}
	out[0] = int8(res)
}

func simdReduceLeadingMinInt8(in, out []int8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = min(acc[b], v)
		}
	}
	for b := range B {
		out[b] = int8(acc[b])
	}
}

func simdReduceTrailingMinInt8(in []int8, out []int8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = min(res, vi)
		}
		out[a] = int8(res)
	}
}

func simdReduceAllMinInt8(in []int8, out []int8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(int8)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = min(res, vi)
	}
	out[0] = int8(res)
}

func simdReduceLeadingProductInt8(in, out []int8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] *= v
		}
	}
	for b := range B {
		out[b] = int8(acc[b])
	}
}

func simdReduceTrailingProductInt8(in []int8, out []int8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res *= vi
		}
		out[a] = int8(res)
	}
}

func simdReduceAllProductInt8(in []int8, out []int8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res *= vi
	}
	out[0] = int8(res)
}

func simdReduceLeadingSumUint8(in, out []uint8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] += v
		}
	}
	for b := range B {
		out[b] = uint8(acc[b])
	}
}

func simdReduceTrailingSumUint8(in []uint8, out []uint8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res += vi
		}
		out[a] = uint8(res)
	}
}

func simdReduceAllSumUint8(in []uint8, out []uint8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res += vi
	}
	out[0] = uint8(res)
}

func simdReduceLeadingMaxUint8(in, out []uint8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = max(acc[b], v)
		}
	}
	for b := range B {
		out[b] = uint8(acc[b])
	}
}

func simdReduceTrailingMaxUint8(in []uint8, out []uint8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = max(res, vi)
		}
		out[a] = uint8(res)
	}
}

func simdReduceAllMaxUint8(in []uint8, out []uint8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(uint8)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = max(res, vi)
	}
	out[0] = uint8(res)
}

func simdReduceLeadingMinUint8(in, out []uint8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = min(acc[b], v)
		}
	}
	for b := range B {
		out[b] = uint8(acc[b])
	}
}

func simdReduceTrailingMinUint8(in []uint8, out []uint8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = min(res, vi)
		}
		out[a] = uint8(res)
	}
}

func simdReduceAllMinUint8(in []uint8, out []uint8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(uint8)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = min(res, vi)
	}
	out[0] = uint8(res)
}

func simdReduceLeadingProductUint8(in, out []uint8, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] *= v
		}
	}
	for b := range B {
		out[b] = uint8(acc[b])
	}
}

func simdReduceTrailingProductUint8(in []uint8, out []uint8, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint8(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res *= vi
		}
		out[a] = uint8(res)
	}
}

func simdReduceAllProductUint8(in []uint8, out []uint8, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res *= vi
	}
	out[0] = uint8(res)
}

func simdReduceLeadingSumInt16(in, out []int16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] += v
		}
	}
	for b := range B {
		out[b] = int16(acc[b])
	}
}

func simdReduceTrailingSumInt16(in []int16, out []int16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res += vi
		}
		out[a] = int16(res)
	}
}

func simdReduceAllSumInt16(in []int16, out []int16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res += vi
	}
	out[0] = int16(res)
}

func simdReduceLeadingMaxInt16(in, out []int16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = max(acc[b], v)
		}
	}
	for b := range B {
		out[b] = int16(acc[b])
	}
}

func simdReduceTrailingMaxInt16(in []int16, out []int16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = max(res, vi)
		}
		out[a] = int16(res)
	}
}

func simdReduceAllMaxInt16(in []int16, out []int16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(int16)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = max(res, vi)
	}
	out[0] = int16(res)
}

func simdReduceLeadingMinInt16(in, out []int16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = min(acc[b], v)
		}
	}
	for b := range B {
		out[b] = int16(acc[b])
	}
}

func simdReduceTrailingMinInt16(in []int16, out []int16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = min(res, vi)
		}
		out[a] = int16(res)
	}
}

func simdReduceAllMinInt16(in []int16, out []int16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(int16)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = min(res, vi)
	}
	out[0] = int16(res)
}

func simdReduceLeadingProductInt16(in, out []int16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] *= v
		}
	}
	for b := range B {
		out[b] = int16(acc[b])
	}
}

func simdReduceTrailingProductInt16(in []int16, out []int16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = int16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res *= vi
		}
		out[a] = int16(res)
	}
}

func simdReduceAllProductInt16(in []int16, out []int16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res *= vi
	}
	out[0] = int16(res)
}

func simdReduceLeadingSumUint16(in, out []uint16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] += v
		}
	}
	for b := range B {
		out[b] = uint16(acc[b])
	}
}

func simdReduceTrailingSumUint16(in []uint16, out []uint16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res += vi
		}
		out[a] = uint16(res)
	}
}

func simdReduceAllSumUint16(in []uint16, out []uint16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 0
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res += vi
	}
	out[0] = uint16(res)
}

func simdReduceLeadingMaxUint16(in, out []uint16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = max(acc[b], v)
		}
	}
	for b := range B {
		out[b] = uint16(acc[b])
	}
}

func simdReduceTrailingMaxUint16(in []uint16, out []uint16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = max(res, vi)
		}
		out[a] = uint16(res)
	}
}

func simdReduceAllMaxUint16(in []uint16, out []uint16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.LowestValue().(uint16)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = max(res, vi)
	}
	out[0] = uint16(res)
}

func simdReduceLeadingMinUint16(in, out []uint16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] = min(acc[b], v)
		}
	}
	for b := range B {
		out[b] = uint16(acc[b])
	}
}

func simdReduceTrailingMinUint16(in []uint16, out []uint16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res = min(res, vi)
		}
		out[a] = uint16(res)
	}
}

func simdReduceAllMinUint16(in []uint16, out []uint16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = dtype.HighestValue().(uint16)
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res = min(res, vi)
	}
	out[0] = uint16(res)
}

func simdReduceLeadingProductUint16(in, out []uint16, A, B int) {
	if A == 0 || B == 0 {
		return
	}
	acc := make([]int32, B)
	for b := range B {
		acc[b] = int32(in[b])
	}
	for a := 1; a < A; a++ {
		off := a * B
		row := in[off : off+B]
		for b := range B {
			v := int32(row[b])
			acc[b] *= v
		}
	}
	for b := range B {
		out[b] = uint16(acc[b])
	}
}

func simdReduceTrailingProductUint16(in []uint16, out []uint16, A, B int, dtype dtypes.DType) {
	if B == 0 {
		for a := range A {
			out[a] = uint16(0)
		}
		return
	}
	for a := range A {
		row := in[a*B : (a+1)*B]
		res := int32(row[0])
		for _, v := range row[1:] {
			vi := int32(v)
			res *= vi
		}
		out[a] = uint16(res)
	}
}

func simdReduceAllProductUint16(in []uint16, out []uint16, dtype dtypes.DType) {
	n := len(in)
	if n == 0 {
		out[0] = 1
		return
	}
	res := int32(in[0])
	for _, v := range in[1:] {
		vi := int32(v)
		res *= vi
	}
	out[0] = uint16(res)
}

func dispatchReduceSumSIMD(operand, output *gobackend.Buffer, cfg ReduceConfig, dtype dtypes.DType) error {
	switch dtype {
	case dtypes.Float32:
		in := operand.Flat.([]float32)
		out := output.Flat.([]float32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumFloat32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumFloat32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumFloat32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float64:
		in := operand.Flat.([]float64)
		out := output.Flat.([]float64)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumFloat64(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumFloat64(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumFloat64(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int32:
		in := operand.Flat.([]int32)
		out := output.Flat.([]int32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumInt32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumInt32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumInt32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint32:
		in := operand.Flat.([]uint32)
		out := output.Flat.([]uint32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumUint32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumUint32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumUint32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float16:
		in := operand.Flat.([]float16.Float16)
		out := output.Flat.([]float16.Float16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.BFloat16:
		in := operand.Flat.([]bfloat16.BFloat16)
		out := output.Flat.([]bfloat16.BFloat16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumBFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumBFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumBFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int8:
		in := operand.Flat.([]int8)
		out := output.Flat.([]int8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumInt8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumInt8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumInt8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint8:
		in := operand.Flat.([]uint8)
		out := output.Flat.([]uint8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumUint8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumUint8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumUint8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int16:
		in := operand.Flat.([]int16)
		out := output.Flat.([]int16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumInt16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumInt16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumInt16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint16:
		in := operand.Flat.([]uint16)
		out := output.Flat.([]uint16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllSumUint16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingSumUint16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingSumUint16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	default:
		return gobackend.ErrFallback
	}
	return nil
}

func execReduceSumSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	dtype := inputs[0].RawShape.DType
	var cfg ReduceConfig
	if node != nil && node.Data != nil {
		if c, ok := node.Data.(*ReduceConfig); ok {
			cfg = *c
		} else if c, ok := node.Data.(ReduceConfig); ok {
			cfg = c
		}
	}
	if cfg.Pattern == 0 && cfg.A == 0 && cfg.B == 0 && len(cfg.Axes) == 0 {
		var reduceAxes []int
		if node != nil && node.Data != nil {
			if a, ok := node.Data.([]int); ok {
				reduceAxes = a
			}
		}
		cfg = DetermineReduceConfig(inputs[0].RawShape, reduceAxes)
	}
	if !canExecuteReduceSIMD(cfg, dtype, supportedReduceDTypes) {
		return nil, gobackend.ErrFallback
	}
	operand, output, cfg, err := prepareReduceBuffers(backend, node, inputs)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	err = dispatchReduceSumSIMD(operand, output, cfg, dtype)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func dispatchReduceMaxSIMD(operand, output *gobackend.Buffer, cfg ReduceConfig, dtype dtypes.DType) error {
	switch dtype {
	case dtypes.Float32:
		in := operand.Flat.([]float32)
		out := output.Flat.([]float32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxFloat32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxFloat32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxFloat32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float64:
		in := operand.Flat.([]float64)
		out := output.Flat.([]float64)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxFloat64(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxFloat64(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxFloat64(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int32:
		in := operand.Flat.([]int32)
		out := output.Flat.([]int32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxInt32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxInt32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxInt32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint32:
		in := operand.Flat.([]uint32)
		out := output.Flat.([]uint32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxUint32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxUint32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxUint32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float16:
		in := operand.Flat.([]float16.Float16)
		out := output.Flat.([]float16.Float16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.BFloat16:
		in := operand.Flat.([]bfloat16.BFloat16)
		out := output.Flat.([]bfloat16.BFloat16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxBFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxBFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxBFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int8:
		in := operand.Flat.([]int8)
		out := output.Flat.([]int8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxInt8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxInt8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxInt8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint8:
		in := operand.Flat.([]uint8)
		out := output.Flat.([]uint8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxUint8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxUint8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxUint8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int16:
		in := operand.Flat.([]int16)
		out := output.Flat.([]int16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxInt16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxInt16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxInt16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint16:
		in := operand.Flat.([]uint16)
		out := output.Flat.([]uint16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMaxUint16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMaxUint16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMaxUint16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	default:
		return gobackend.ErrFallback
	}
	return nil
}

func execReduceMaxSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	dtype := inputs[0].RawShape.DType
	var cfg ReduceConfig
	if node != nil && node.Data != nil {
		if c, ok := node.Data.(*ReduceConfig); ok {
			cfg = *c
		} else if c, ok := node.Data.(ReduceConfig); ok {
			cfg = c
		}
	}
	if cfg.Pattern == 0 && cfg.A == 0 && cfg.B == 0 && len(cfg.Axes) == 0 {
		var reduceAxes []int
		if node != nil && node.Data != nil {
			if a, ok := node.Data.([]int); ok {
				reduceAxes = a
			}
		}
		cfg = DetermineReduceConfig(inputs[0].RawShape, reduceAxes)
	}
	if !canExecuteReduceSIMD(cfg, dtype, supportedReduceDTypes) {
		return nil, gobackend.ErrFallback
	}
	operand, output, cfg, err := prepareReduceBuffers(backend, node, inputs)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	err = dispatchReduceMaxSIMD(operand, output, cfg, dtype)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func dispatchReduceMinSIMD(operand, output *gobackend.Buffer, cfg ReduceConfig, dtype dtypes.DType) error {
	switch dtype {
	case dtypes.Float32:
		in := operand.Flat.([]float32)
		out := output.Flat.([]float32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinFloat32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinFloat32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinFloat32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float64:
		in := operand.Flat.([]float64)
		out := output.Flat.([]float64)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinFloat64(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinFloat64(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinFloat64(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int32:
		in := operand.Flat.([]int32)
		out := output.Flat.([]int32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinInt32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinInt32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinInt32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint32:
		in := operand.Flat.([]uint32)
		out := output.Flat.([]uint32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinUint32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinUint32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinUint32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float16:
		in := operand.Flat.([]float16.Float16)
		out := output.Flat.([]float16.Float16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.BFloat16:
		in := operand.Flat.([]bfloat16.BFloat16)
		out := output.Flat.([]bfloat16.BFloat16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinBFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinBFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinBFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int8:
		in := operand.Flat.([]int8)
		out := output.Flat.([]int8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinInt8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinInt8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinInt8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint8:
		in := operand.Flat.([]uint8)
		out := output.Flat.([]uint8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinUint8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinUint8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinUint8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int16:
		in := operand.Flat.([]int16)
		out := output.Flat.([]int16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinInt16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinInt16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinInt16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint16:
		in := operand.Flat.([]uint16)
		out := output.Flat.([]uint16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllMinUint16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingMinUint16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingMinUint16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	default:
		return gobackend.ErrFallback
	}
	return nil
}

func execReduceMinSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	dtype := inputs[0].RawShape.DType
	var cfg ReduceConfig
	if node != nil && node.Data != nil {
		if c, ok := node.Data.(*ReduceConfig); ok {
			cfg = *c
		} else if c, ok := node.Data.(ReduceConfig); ok {
			cfg = c
		}
	}
	if cfg.Pattern == 0 && cfg.A == 0 && cfg.B == 0 && len(cfg.Axes) == 0 {
		var reduceAxes []int
		if node != nil && node.Data != nil {
			if a, ok := node.Data.([]int); ok {
				reduceAxes = a
			}
		}
		cfg = DetermineReduceConfig(inputs[0].RawShape, reduceAxes)
	}
	if !canExecuteReduceSIMD(cfg, dtype, supportedReduceDTypes) {
		return nil, gobackend.ErrFallback
	}
	operand, output, cfg, err := prepareReduceBuffers(backend, node, inputs)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	err = dispatchReduceMinSIMD(operand, output, cfg, dtype)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func dispatchReduceProductSIMD(operand, output *gobackend.Buffer, cfg ReduceConfig, dtype dtypes.DType) error {
	switch dtype {
	case dtypes.Float32:
		in := operand.Flat.([]float32)
		out := output.Flat.([]float32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductFloat32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductFloat32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductFloat32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float64:
		in := operand.Flat.([]float64)
		out := output.Flat.([]float64)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductFloat64(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductFloat64(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductFloat64(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int32:
		in := operand.Flat.([]int32)
		out := output.Flat.([]int32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductInt32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductInt32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductInt32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint32:
		in := operand.Flat.([]uint32)
		out := output.Flat.([]uint32)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductUint32(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductUint32(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductUint32(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Float16:
		in := operand.Flat.([]float16.Float16)
		out := output.Flat.([]float16.Float16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.BFloat16:
		in := operand.Flat.([]bfloat16.BFloat16)
		out := output.Flat.([]bfloat16.BFloat16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductBFloat16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductBFloat16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductBFloat16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int8:
		in := operand.Flat.([]int8)
		out := output.Flat.([]int8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductInt8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductInt8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductInt8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint8:
		in := operand.Flat.([]uint8)
		out := output.Flat.([]uint8)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductUint8(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductUint8(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductUint8(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Int16:
		in := operand.Flat.([]int16)
		out := output.Flat.([]int16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductInt16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductInt16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductInt16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	case dtypes.Uint16:
		in := operand.Flat.([]uint16)
		out := output.Flat.([]uint16)
		switch cfg.Pattern {
		case ReduceAll:
			simdReduceAllProductUint16(in, out, dtype)
		case ReduceLeading:
			simdReduceLeadingProductUint16(in, out, cfg.A, cfg.B)
		case ReduceTrailing:
			simdReduceTrailingProductUint16(in, out, cfg.A, cfg.B, dtype)
		default:
			return gobackend.ErrFallback
		}
	default:
		return gobackend.ErrFallback
	}
	return nil
}

func execReduceProductSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	dtype := inputs[0].RawShape.DType
	var cfg ReduceConfig
	if node != nil && node.Data != nil {
		if c, ok := node.Data.(*ReduceConfig); ok {
			cfg = *c
		} else if c, ok := node.Data.(ReduceConfig); ok {
			cfg = c
		}
	}
	if cfg.Pattern == 0 && cfg.A == 0 && cfg.B == 0 && len(cfg.Axes) == 0 {
		var reduceAxes []int
		if node != nil && node.Data != nil {
			if a, ok := node.Data.([]int); ok {
				reduceAxes = a
			}
		}
		cfg = DetermineReduceConfig(inputs[0].RawShape, reduceAxes)
	}
	if !canExecuteReduceSIMD(cfg, dtype, supportedReduceDTypes) {
		return nil, gobackend.ErrFallback
	}
	operand, output, cfg, err := prepareReduceBuffers(backend, node, inputs)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	err = dispatchReduceProductSIMD(operand, output, cfg, dtype)
	if err != nil {
		return nil, err
	}
	return output, nil
}

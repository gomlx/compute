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
	"github.com/gomlx/compute/support/simdmath"
)

func init() {
	gobackend.SetNodeExecutor(compute.OpTypeExp, PrioritySIMD, execExpSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeSqrt, PrioritySIMD, execSqrtSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeRsqrt, PrioritySIMD, execRsqrtSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeLogistic, PrioritySIMD, execLogisticSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeTanh, PrioritySIMD, execTanhSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeErf, PrioritySIMD, execErfSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeAbs, PrioritySIMD, execAbsSIMD)
	gobackend.SetNodeExecutor(compute.OpTypeNeg, PrioritySIMD, execNegSIMD)
}

// -----------------------------------------------------------------------------
// Float32 Unary Kernels
// -----------------------------------------------------------------------------

func simdUnaryFloat32(in, out []float32, fn func(simd.Float32s) simd.Float32s) {
	vLen := simd.BroadcastFloat32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		v := simd.LoadFloat32s(in[i:])
		fn(v).Store(out[i:])
	}
	if i < len(out) {
		v, _ := simd.LoadFloat32sPart(in[i:])
		fn(v).StorePart(out[i:])
	}
}

// -----------------------------------------------------------------------------
// Float64 Unary Kernels
// -----------------------------------------------------------------------------

func simdUnaryFloat64(in, out []float64, fn func(simd.Float64s) simd.Float64s) {
	vLen := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		v := simd.LoadFloat64s(in[i:])
		fn(v).Store(out[i:])
	}
	if i < len(out) {
		v, _ := simd.LoadFloat64sPart(in[i:])
		fn(v).StorePart(out[i:])
	}
}

// -----------------------------------------------------------------------------
// Half Precision (BFloat16, Float16) Unary Kernels
// -----------------------------------------------------------------------------

func simdUnaryBFloat16(in, out []bfloat16.BFloat16, fn func(simd.Float32s) simd.Float32s) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		even, odd := bfloat16.ToFloat32SIMD(bfloat16.LoadBFloat16s(in[i : i+vLen]))
		even = fn(even)
		odd = fn(odd)
		bfloat16.StoreBFloat16s(bfloat16.FromFloat32SIMD(even, odd), out[i:i+vLen])
	}
	if i < len(out) {
		v, _ := bfloat16.LoadBFloat16sPart(in[i:])
		even, odd := bfloat16.ToFloat32SIMD(v)
		even = fn(even)
		odd = fn(odd)
		bfloat16.StoreBFloat16sPart(bfloat16.FromFloat32SIMD(even, odd), out[i:])
	}
}

func simdUnaryFloat16(in, out []float16.Float16, fn func(simd.Float32s) simd.Float32s) {
	vLen := simd.BroadcastUint16s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		even, odd := float16.ToFloat32SIMD(float16.LoadFloat16s(in[i : i+vLen]))
		even = fn(even)
		odd = fn(odd)
		float16.StoreFloat16s(float16.FromFloat32SIMD(even, odd), out[i:i+vLen])
	}
	if i < len(out) {
		v, _ := float16.LoadFloat16sPart(in[i:])
		even, odd := float16.ToFloat32SIMD(v)
		even = fn(even)
		odd = fn(odd)
		float16.StoreFloat16sPart(float16.FromFloat32SIMD(even, odd), out[i:])
	}
}

// -----------------------------------------------------------------------------
// Integer Unary Kernels
// -----------------------------------------------------------------------------

func simdAbsInt32(in, out []int32) {
	vLen := simd.BroadcastInt32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(in[i:]).Abs().Store(out[i:])
	}
	if i < len(out) {
		v, _ := simd.LoadInt32sPart(in[i:])
		v.Abs().StorePart(out[i:])
	}
}

func simdNegInt32(in, out []int32) {
	vLen := simd.BroadcastInt32s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt32s(in[i:]).Neg().Store(out[i:])
	}
	if i < len(out) {
		v, _ := simd.LoadInt32sPart(in[i:])
		v.Neg().StorePart(out[i:])
	}
}

func simdNegInt64(in, out []int64) {
	vLen := simd.BroadcastInt64s(0).Len()
	i := 0
	for ; i+vLen <= len(out); i += vLen {
		simd.LoadInt64s(in[i:]).Neg().Store(out[i:])
	}
	if i < len(out) {
		v, _ := simd.LoadInt64sPart(in[i:])
		v.Neg().StorePart(out[i:])
	}
}

// -----------------------------------------------------------------------------
// Unary Node Executors
// -----------------------------------------------------------------------------

func execExpSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), simdmath.ExpFloat32)
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), simdmath.ExpFloat64)
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), simdmath.ExpFloat32)
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), simdmath.ExpFloat32)
	default:
		return execExp(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execSqrtSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), simdmath.SqrtFloat32)
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), simdmath.SqrtFloat64)
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), simdmath.SqrtFloat32)
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), simdmath.SqrtFloat32)
	default:
		return execSqrt(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execRsqrtSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), simdmath.RsqrtFloat32)
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), simdmath.RsqrtFloat64)
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), simdmath.RsqrtFloat32)
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), simdmath.RsqrtFloat32)
	default:
		return execRsqrt(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execLogisticSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), simdmath.SigmoidFloat32)
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), simdmath.SigmoidFloat64)
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), simdmath.SigmoidFloat32)
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), simdmath.SigmoidFloat32)
	default:
		return execLogistic(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execTanhSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), simdmath.TanhFloat32)
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), simdmath.TanhFloat64)
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), simdmath.TanhFloat32)
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), simdmath.TanhFloat32)
	default:
		return execTanh(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execErfSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), simdmath.ErfFloat32)
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), simdmath.ErfFloat64)
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), simdmath.ErfFloat32)
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), simdmath.ErfFloat32)
	default:
		return execErf(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execAbsSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), func(v simd.Float32s) simd.Float32s { return v.Abs() })
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), func(v simd.Float64s) simd.Float64s { return v.Abs() })
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), func(v simd.Float32s) simd.Float32s { return v.Abs() })
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), func(v simd.Float32s) simd.Float32s { return v.Abs() })
	case dtypes.Int32:
		simdAbsInt32(input.Flat.([]int32), output.Flat.([]int32))
	default:
		return execAbs(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

func execNegSIMD(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	input, output, err := unaryOperandAndOutput(backend, inputs, inputsOwned)
	if err != nil {
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}
	switch input.RawShape.DType {
	case dtypes.Float32:
		simdUnaryFloat32(input.Flat.([]float32), output.Flat.([]float32), func(v simd.Float32s) simd.Float32s { return v.Neg() })
	case dtypes.Float64:
		simdUnaryFloat64(input.Flat.([]float64), output.Flat.([]float64), func(v simd.Float64s) simd.Float64s { return v.Neg() })
	case dtypes.BFloat16:
		simdUnaryBFloat16(input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), func(v simd.Float32s) simd.Float32s { return v.Neg() })
	case dtypes.Float16:
		simdUnaryFloat16(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), func(v simd.Float32s) simd.Float32s { return v.Neg() })
	case dtypes.Int32:
		simdNegInt32(input.Flat.([]int32), output.Flat.([]int32))
	case dtypes.Int64:
		simdNegInt64(input.Flat.([]int64), output.Flat.([]int64))
	default:
		return execNeg(backend, node, inputs, inputsOwned)
	}
	return output, nil
}

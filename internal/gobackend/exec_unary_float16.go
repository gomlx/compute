// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

// Float16 unary operations support.
// These wrap the generic unary executors to handle Float16 dtype.

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
)

// Float16 unary operation helpers

func execNegF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(-input.Float32())
	}
}

func execAbsF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		f := input.Float32()
		if f < 0 {
			outputs[ii] = float16.FromFloat32(-f)
		} else {
			outputs[ii] = input
		}
	}
}

func execSignF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		f := input.Float32()
		switch {
		case f < 0:
			outputs[ii] = float16.FromFloat32(-1.0)
		case f > 0:
			outputs[ii] = float16.FromFloat32(1.0)
		default:
			outputs[ii] = float16.FromFloat32(0.0)
		}
	}
}

func execExpF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Exp(float64(input.Float32()))))
	}
}

func execExpm1F16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Expm1(float64(input.Float32()))))
	}
}

func execLogF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Log(float64(input.Float32()))))
	}
}

func execLog1pF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Log1p(float64(input.Float32()))))
	}
}

func execCeilF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Ceil(float64(input.Float32()))))
	}
}

func execFloorF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Floor(float64(input.Float32()))))
	}
}

func execRoundF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Round(float64(input.Float32()))))
	}
}

func execRsqrtF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(1.0 / math.Sqrt(float64(input.Float32()))))
	}
}

func execCosF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Cos(float64(input.Float32()))))
	}
}

func execSinF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Sin(float64(input.Float32()))))
	}
}

func execTanhF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Tanh(float64(input.Float32()))))
	}
}

func execLogisticF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		input64 := float64(input.Float32())
		var output64 float64
		if input64 >= 0 {
			output64 = 1.0 / (1.0 + math.Exp(-input64))
		} else {
			e_x := math.Exp(input64)
			output64 = e_x / (1.0 + e_x)
		}
		outputs[ii] = float16.FromFloat32(float32(output64))
	}
}

func execIsFiniteF16(inputs []float16.Float16, outputs []bool) {
	for ii, input := range inputs {
		f := input.Float32()
		outputs[ii] = !math.IsInf(float64(f), 0) && !math.IsNaN(float64(f))
	}
}

func execErfF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.FromFloat32(float32(math.Erf(float64(input.Float32()))))
	}
}

func makeFloat16UnaryWrapper(
	origExec func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error),
	opFn func(inputs, outputs []float16.Float16),
) func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error) {
	return func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
		if inputs[0].RawShape.DType != dtypes.Float16 {
			return origExec(backend, node, inputs, inputsOwned)
		}
		input, output, _ := unaryOperandAndOutput(backend, inputs, inputsOwned)
		opFn(input.Flat.([]float16.Float16), output.Flat.([]float16.Float16))
		return output, nil
	}
}

func init() {
	// Register Float16 unary wrappers with priorityTyped.
	// These wrap the generic executors (from exec_unary.go) to handle Float16 dtype.
	SetNodeExecutor(compute.OpTypeNeg, PriorityTyped, makeFloat16UnaryWrapper(execNeg, execNegF16))
	SetNodeExecutor(compute.OpTypeAbs, PriorityTyped, makeFloat16UnaryWrapper(execAbs, execAbsF16))
	SetNodeExecutor(compute.OpTypeSign, PriorityTyped, makeFloat16UnaryWrapper(execSign, execSignF16))
	SetNodeExecutor(compute.OpTypeExp, PriorityTyped, makeFloat16UnaryWrapper(execExp, execExpF16))
	SetNodeExecutor(compute.OpTypeExpm1, PriorityTyped, makeFloat16UnaryWrapper(execExpm1, execExpm1F16))
	SetNodeExecutor(compute.OpTypeLog, PriorityTyped, makeFloat16UnaryWrapper(execLog, execLogF16))
	SetNodeExecutor(compute.OpTypeLog1p, PriorityTyped, makeFloat16UnaryWrapper(execLog1p, execLog1pF16))
	SetNodeExecutor(compute.OpTypeCeil, PriorityTyped, makeFloat16UnaryWrapper(execCeil, execCeilF16))
	SetNodeExecutor(compute.OpTypeFloor, PriorityTyped, makeFloat16UnaryWrapper(execFloor, execFloorF16))
	SetNodeExecutor(compute.OpTypeRound, PriorityTyped, makeFloat16UnaryWrapper(execRound, execRoundF16))
	SetNodeExecutor(compute.OpTypeRsqrt, PriorityTyped, makeFloat16UnaryWrapper(execRsqrt, execRsqrtF16))
	SetNodeExecutor(compute.OpTypeCos, PriorityTyped, makeFloat16UnaryWrapper(execCos, execCosF16))
	SetNodeExecutor(compute.OpTypeSin, PriorityTyped, makeFloat16UnaryWrapper(execSin, execSinF16))
	SetNodeExecutor(compute.OpTypeTanh, PriorityTyped, makeFloat16UnaryWrapper(execTanh, execTanhF16))
	SetNodeExecutor(compute.OpTypeLogistic, PriorityTyped, makeFloat16UnaryWrapper(execLogistic, execLogisticF16))
	SetNodeExecutor(compute.OpTypeErf, PriorityTyped, makeFloat16UnaryWrapper(execErf, execErfF16))

	// IsFinite is special - returns bool
	SetNodeExecutor(compute.OpTypeIsFinite, PriorityTyped, func(backend *Backend, node *Node, inputs []*Buffer,
		inputsOwned []bool) (*Buffer, error) {
		if inputs[0].RawShape.DType != dtypes.Float16 {
			return execIsFinite(backend, node, inputs, inputsOwned)
		}
		input := inputs[0]
		output, err := backend.GetBuffer(dtypes.Bool, input.RawShape.Size())
		if err != nil {
			return nil, err
		}
		output.RawShape = node.Shape
		execIsFiniteF16(input.Flat.([]float16.Float16), output.Flat.([]bool))
		return output, nil
	})
}

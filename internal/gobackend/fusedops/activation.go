// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedActivation.Register(FusedActivation, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedActivation, gobackend.PriorityTyped, execFusedActivation)

	gobackend.RegisterFusedActivationVJP.Register(FusedActivationVJP, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedActivationVJP, gobackend.PriorityTyped, execFusedActivationVJP)
}

type nodeFusedActivation struct {
	cfg compute.ActivationConfig
}

func (d *nodeFusedActivation) EqualNodeData(other gobackend.NodeDataComparable) bool {
	return d.cfg == other.(*nodeFusedActivation).cfg
}

// FusedActivation applies the configured activation function.
func FusedActivation(f *gobackend.Function, x compute.Value, cfg compute.ActivationConfig) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("FusedActivation", x)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	outShape, err := shapeinference.FusedActivation(xNode.Shape, cfg)
	if err != nil {
		return nil, err
	}

	data := &nodeFusedActivation{cfg: cfg}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedActivation, outShape, []*gobackend.Node{xNode}, data)
	return node, nil
}

func execFusedActivation(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedActivation)
	input := inputs[0]
	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}

	act := data.cfg.Type
	if act == compute.ActivationSwiGLU {
		rank := input.RawShape.Rank()
		lastDim := input.RawShape.Dimensions[rank-1]
		hiddenDim := lastDim / 2
		totalIn := input.RawShape.Size()
		numRows := totalIn / lastDim

		switch input.RawShape.DType {
		case dtypes.Float32:
			activations.ExecuteSwiGLU(backend, input.Flat.([]float32), output.Flat.([]float32), numRows, hiddenDim)
		case dtypes.Float64:
			activations.ExecuteSwiGLU(backend, input.Flat.([]float64), output.Flat.([]float64), numRows, hiddenDim)
		case dtypes.BFloat16:
			activations.ExecuteSwiGLU(backend, input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), numRows, hiddenDim)
		case dtypes.Float16:
			activations.ExecuteSwiGLU(backend, input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), numRows, hiddenDim)
		default:
			return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedActivation(SwiGLU): dtype %s", input.RawShape.DType)
		}
		return output, nil
	}

	switch input.RawShape.DType {
	case dtypes.Float32:
		activations.Execute(backend, act, input.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		activations.Execute(backend, act, input.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		activations.Execute(backend, act, input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	case dtypes.Float16:
		activations.Execute(backend, act, input.Flat.([]float16.Float16), output.Flat.([]float16.Float16))
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedActivation: dtype %s", input.RawShape.DType)
	}
	return output, nil
}

type nodeFusedActivationVJP struct {
	cfg  compute.ActivationConfig
	hasY bool
	hasX bool
}

func (d *nodeFusedActivationVJP) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedActivationVJP)
	return d.cfg == o.cfg && d.hasY == o.hasY && d.hasX == o.hasX
}

// FusedActivationVJP computes the vector-jacobian product of the configured activation.
func FusedActivationVJP(f *gobackend.Function, y, x, dOutput compute.Value, cfg compute.ActivationConfig) (compute.Value, error) {
	if dOutput == nil {
		return nil, errors.Errorf("FusedActivationVJP: dOutput cannot be nil")
	}

	var inputNodes []*gobackend.Node
	var yShape, xShape shapes.Shape
	if y != nil {
		nodes, err := f.VerifyAndCastValues("FusedActivationVJP", y)
		if err != nil {
			return nil, err
		}
		inputNodes = append(inputNodes, nodes[0])
		yShape = nodes[0].Shape
	}
	if x != nil {
		nodes, err := f.VerifyAndCastValues("FusedActivationVJP", x)
		if err != nil {
			return nil, err
		}
		inputNodes = append(inputNodes, nodes[0])
		xShape = nodes[0].Shape
	}
	dOutNodes, err := f.VerifyAndCastValues("FusedActivationVJP", dOutput)
	if err != nil {
		return nil, err
	}
	inputNodes = append(inputNodes, dOutNodes[0])
	dOutputShape := dOutNodes[0].Shape

	outShape, err := shapeinference.FusedActivationVJP(yShape, xShape, dOutputShape, cfg)
	if err != nil {
		return nil, err
	}

	data := &nodeFusedActivationVJP{
		cfg:  cfg,
		hasY: y != nil,
		hasX: x != nil,
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedActivationVJP, outShape, inputNodes, data)
	return node, nil
}

func execFusedActivationVJP(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedActivationVJP)
	idx := 0
	var yBuf, xBuf *gobackend.Buffer
	if data.hasY {
		yBuf = inputs[idx]
		idx++
	}
	if data.hasX {
		xBuf = inputs[idx]
		idx++
	}
	dOutBuf := inputs[idx]

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}

	act := data.cfg.Type
	if act == compute.ActivationSwiGLU {
		rank := xBuf.RawShape.Rank()
		lastDim := xBuf.RawShape.Dimensions[rank-1]
		hiddenDim := lastDim / 2
		totalIn := xBuf.RawShape.Size()
		numRows := totalIn / lastDim

		switch xBuf.RawShape.DType {
		case dtypes.Float32:
			activations.ExecuteSwiGLUVJP(backend, xBuf.Flat.([]float32), dOutBuf.Flat.([]float32), output.Flat.([]float32), numRows, hiddenDim)
		case dtypes.Float64:
			activations.ExecuteSwiGLUVJP(backend, xBuf.Flat.([]float64), dOutBuf.Flat.([]float64), output.Flat.([]float64), numRows, hiddenDim)
		case dtypes.BFloat16:
			activations.ExecuteSwiGLUVJP(backend, xBuf.Flat.([]bfloat16.BFloat16), dOutBuf.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), numRows, hiddenDim)
		case dtypes.Float16:
			activations.ExecuteSwiGLUVJP(backend, xBuf.Flat.([]float16.Float16), dOutBuf.Flat.([]float16.Float16), output.Flat.([]float16.Float16), numRows, hiddenDim)
		default:
			return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedActivationVJP(SwiGLU): dtype %s", xBuf.RawShape.DType)
		}
		return output, nil
	}

	switch dOutBuf.RawShape.DType {
	case dtypes.Float32:
		var yFlat, xFlat []float32
		if yBuf != nil {
			yFlat = yBuf.Flat.([]float32)
		}
		if xBuf != nil {
			xFlat = xBuf.Flat.([]float32)
		}
		activations.ExecuteVJP(backend, act, yFlat, xFlat, dOutBuf.Flat.([]float32), output.Flat.([]float32))
	case dtypes.Float64:
		var yFlat, xFlat []float64
		if yBuf != nil {
			yFlat = yBuf.Flat.([]float64)
		}
		if xBuf != nil {
			xFlat = xBuf.Flat.([]float64)
		}
		activations.ExecuteVJP(backend, act, yFlat, xFlat, dOutBuf.Flat.([]float64), output.Flat.([]float64))
	case dtypes.BFloat16:
		var yFlat, xFlat []bfloat16.BFloat16
		if yBuf != nil {
			yFlat = yBuf.Flat.([]bfloat16.BFloat16)
		}
		if xBuf != nil {
			xFlat = xBuf.Flat.([]bfloat16.BFloat16)
		}
		activations.ExecuteVJP(backend, act, yFlat, xFlat, dOutBuf.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16))
	case dtypes.Float16:
		var yFlat, xFlat []float16.Float16
		if yBuf != nil {
			yFlat = yBuf.Flat.([]float16.Float16)
		}
		if xBuf != nil {
			xFlat = xBuf.Flat.([]float16.Float16)
		}
		activations.ExecuteVJP(backend, act, yFlat, xFlat, dOutBuf.Flat.([]float16.Float16), output.Flat.([]float16.Float16))
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedActivationVJP: dtype %s", dOutBuf.RawShape.DType)
	}
	return output, nil
}

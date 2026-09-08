// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dense

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedDense.Register(FusedDense, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedDense, gobackend.PriorityTyped, execFusedDense)
}

type nodeFusedDense struct {
	options         compute.DenseConfig
	layout          dot.Layout
	batchSize       int
	lhsCrossSize    int
	rhsCrossSize    int
	contractingSize int
}

func (d *nodeFusedDense) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedDense)
	return d.options == o.options &&
		d.layout == o.layout &&
		d.batchSize == o.batchSize &&
		d.lhsCrossSize == o.lhsCrossSize &&
		d.rhsCrossSize == o.rhsCrossSize &&
		d.contractingSize == o.contractingSize
}

// Recompute implements gobackend.RecomputableNodeData for nodeFusedDense.
func (d *nodeFusedDense) Recompute(backend *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
	resolvedX := resolvedNodes[originalNode.Inputs[0].Index].Shape
	inFeatures := resolvedX.Dimensions[resolvedX.Rank()-1]
	newData := &nodeFusedDense{
		options:         d.options,
		layout:          d.layout,
		batchSize:       d.batchSize,
		lhsCrossSize:    resolvedX.Size() / inFeatures,
		rhsCrossSize:    d.rhsCrossSize,
		contractingSize: d.contractingSize,
	}
	return newData, nil
}

// FusedDense performs fused matrix multiplication + optional bias + optional activation:
//
//	y = activation(x @ W + bias)   (for DenseLayoutInputOutputs)
//	y = activation(x @ W^T + bias) (for DenseLayoutOutputsInput)
//
// It directly delegates to the highly optimized matmul engine with an epilogue hook,
// avoiding intermediate buffer allocations and performing fused cache-resident epilogues.
func FusedDense(f *gobackend.Function, x, weight, bias compute.Value, options compute.DenseConfig) (compute.Value, error) {
	values := []compute.Value{x, weight}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.VerifyAndCastValues("FusedDense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	var biasShape shapes.Shape
	if len(inputs) > 2 {
		biasShape = inputs[2].Shape
	}
	outShape, err := shapeinference.FusedDense(xNode.Shape, wNode.Shape, biasShape, options)
	if err != nil {
		return nil, err
	}

	if wNode.Shape.IsDynamic() || biasShape.IsDynamic() {
		return nil, compute.ErrNotImplemented
	}
	if xNode.Shape.Dimensions[xNode.Shape.Rank()-1] == shapes.DynamicDim {
		return nil, errors.Errorf("FusedDense: x's last dimension (in_features) cannot be dynamic")
	}

	inFeatures := xNode.Shape.Dimensions[xNode.Shape.Rank()-1]
	var layout dot.Layout
	switch options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		layout = dot.LayoutNonTransposed
	case compute.DenseLayoutOutputsInput:
		layout = dot.LayoutTransposed
	default:
		return nil, errors.Errorf("FusedDense: unknown WeightLayout %v", options.WeightLayout)
	}

	lhsCrossSize := 0
	if !xNode.Shape.IsDynamic() {
		lhsCrossSize = xNode.Shape.Size() / inFeatures
	}
	rhsCrossSize := wNode.Shape.Size() / inFeatures
	contractingSize := inFeatures

	data := &nodeFusedDense{
		options:         options,
		layout:          layout,
		batchSize:       1,
		lhsCrossSize:    lhsCrossSize,
		rhsCrossSize:    rhsCrossSize,
		contractingSize: contractingSize,
	}

	node, _ := f.GetOrCreateNode(compute.OpTypeFusedDense, outShape, inputs, data)
	return node, nil
}

// execFusedDense executes y = activation(x @ W + bias).
func execFusedDense(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	x := inputs[0]
	weight := inputs[1]
	var bias *gobackend.Buffer
	if len(inputs) > 2 {
		bias = inputs[2]
	}

	data := node.Data.(*nodeFusedDense)

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}

	if backend.NoOps {
		return output, nil
	}

	switch output.RawShape.DType {
	case dtypes.Float32:
		xFlat := x.Flat.([]float32)
		wFlat := weight.Flat.([]float32)
		outFlat := output.Flat.([]float32)

		var biasFlat []float32
		if bias != nil {
			biasFlat = bias.Flat.([]float32)
		}

		var actFn activations.InPlaceFn[float32]
		if data.options.Activation.Type != compute.ActivationNone {
			actFn = activations.Get[float32](data.options.Activation.Type)
			if actFn == nil {
				return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: activation %s not implemented for %s", data.options.Activation.Type, output.RawShape.DType)
			}
		}
		epilogue := matmul.Epilogue[float32]{
			Bias:       biasFlat,
			Activation: actFn,
		}

		err = matmul.ExecuteWithEpilogue(
			backend, data.layout, xFlat, wFlat,
			data.batchSize, data.lhsCrossSize, data.rhsCrossSize, data.contractingSize,
			outFlat, epilogue,
		)
		if err != nil {
			return nil, err
		}

	case dtypes.Float64:
		xFlat := x.Flat.([]float64)
		wFlat := weight.Flat.([]float64)
		outFlat := output.Flat.([]float64)

		var biasFlat []float64
		if bias != nil {
			biasFlat = bias.Flat.([]float64)
		}

		var actFn activations.InPlaceFn[float64]
		if data.options.Activation.Type != compute.ActivationNone {
			actFn = activations.Get[float64](data.options.Activation.Type)
			if actFn == nil {
				return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: activation %s not implemented for %s", data.options.Activation.Type, output.RawShape.DType)
			}
		}
		epilogue := matmul.Epilogue[float64]{
			Bias:       biasFlat,
			Activation: actFn,
		}

		err = matmul.ExecuteWithEpilogue(
			backend, data.layout, xFlat, wFlat,
			data.batchSize, data.lhsCrossSize, data.rhsCrossSize, data.contractingSize,
			outFlat, epilogue,
		)
		if err != nil {
			return nil, err
		}

	case dtypes.BFloat16:
		xFlat := x.Flat.([]bfloat16.BFloat16)
		wFlat := weight.Flat.([]bfloat16.BFloat16)
		outBF16 := output.Flat.([]bfloat16.BFloat16)

		// Matmul for BFloat16 accumulates into Float32.
		tmpF32 := make([]float32, output.RawShape.Size())

		var biasF32 []float32
		if bias != nil {
			if biasBF16, ok := bias.Flat.([]bfloat16.BFloat16); ok {
				biasF32 = make([]float32, len(biasBF16))
				for i, v := range biasBF16 {
					biasF32[i] = v.Float32()
				}
			} else if bF32, ok := bias.Flat.([]float32); ok {
				biasF32 = bF32
			}
		}

		var actFn activations.InPlaceFn[float32]
		if data.options.Activation.Type != compute.ActivationNone {
			actFn = activations.Get[float32](data.options.Activation.Type)
			if actFn == nil {
				return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: activation %s not implemented for %s", data.options.Activation.Type, output.RawShape.DType)
			}
		}
		epilogue := matmul.Epilogue[float32]{
			Bias:       biasF32,
			Activation: actFn,
		}

		err = matmul.ExecuteWithEpilogue(
			backend, data.layout, xFlat, wFlat,
			data.batchSize, data.lhsCrossSize, data.rhsCrossSize, data.contractingSize,
			tmpF32, epilogue,
		)
		if err != nil {
			return nil, err
		}

		for i, v := range tmpF32 {
			outBF16[i] = bfloat16.FromFloat32(v)
		}

	case dtypes.Float16:
		xFlat := x.Flat.([]float16.Float16)
		wFlat := weight.Flat.([]float16.Float16)
		outF16 := output.Flat.([]float16.Float16)

		tmpF32 := make([]float32, output.RawShape.Size())

		var biasF32 []float32
		if bias != nil {
			if biasF16Slice, ok := bias.Flat.([]float16.Float16); ok {
				biasF32 = make([]float32, len(biasF16Slice))
				for i, v := range biasF16Slice {
					biasF32[i] = v.Float32()
				}
			} else if bF32, ok := bias.Flat.([]float32); ok {
				biasF32 = bF32
			}
		}

		var actFn activations.InPlaceFn[float32]
		if data.options.Activation.Type != compute.ActivationNone {
			actFn = activations.Get[float32](data.options.Activation.Type)
			if actFn == nil {
				return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: activation %s not implemented for %s", data.options.Activation.Type, output.RawShape.DType)
			}
		}
		epilogue := matmul.Epilogue[float32]{
			Bias:       biasF32,
			Activation: actFn,
		}

		err = matmul.ExecuteWithEpilogue(
			backend, data.layout, xFlat, wFlat,
			data.batchSize, data.lhsCrossSize, data.rhsCrossSize, data.contractingSize,
			tmpF32, epilogue,
		)
		if err != nil {
			return nil, err
		}

		for i, v := range tmpF32 {
			outF16[i] = float16.FromFloat32(v)
		}

	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: dtype %s", output.RawShape.DType)
	}

	return output, nil
}

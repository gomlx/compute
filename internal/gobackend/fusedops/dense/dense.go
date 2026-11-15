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
	dotNodeData     *dot.NodeData
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

// Compile-time check that *nodeFusedDense implements gobackend.ConstantCacheNodeData and gobackend.FusedDenseActivationTarget.
var (
	_ gobackend.ConstantCacheNodeData      = (*nodeFusedDense)(nil)
	_ gobackend.FusedDenseActivationTarget = (*nodeFusedDense)(nil)
)

func (d *nodeFusedDense) TryFuseActivation(cfg compute.ActivationConfig) bool {
	if d == nil || d.options.Activation.Type != compute.ActivationNone {
		return false
	}
	if cfg.Type == compute.ActivationNone || cfg.Type == compute.ActivationSwiGLU {
		return false
	}
	d.options.Activation = cfg
	return true
}

func (d *nodeFusedDense) CacheSignature(inputIdx int) string {
	if d == nil || (inputIdx != 0 && inputIdx != 1) {
		return ""
	}
	return d.dotNodeData.CacheSignature(inputIdx)
}

func (d *nodeFusedDense) BindConstantData(inputIdx int, cd *gobackend.ConstantData) {
	if d == nil || (inputIdx != 0 && inputIdx != 1) {
		return
	}
	d.dotNodeData.BindConstantData(inputIdx, cd)
}

func (d *nodeFusedDense) IsInputCached(inputIdx int) bool {
	if d == nil || (inputIdx != 0 && inputIdx != 1) {
		return false
	}
	return d.dotNodeData.IsInputCached(inputIdx)
}

// Recompute implements gobackend.RecomputableNodeData for nodeFusedDense.
func (d *nodeFusedDense) Recompute(backend *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
	resolvedX := resolvedNodes[originalNode.Inputs[0].Index].Shape
	resolvedW := resolvedNodes[originalNode.Inputs[1].Index].Shape
	var biasShape shapes.Shape
	if len(originalNode.Inputs) > 2 {
		biasShape = resolvedNodes[originalNode.Inputs[2].Index].Shape
	}
	outputShape, err := shapeinference.FusedDense(resolvedX, resolvedW, biasShape, d.options)
	if err != nil {
		return nil, err
	}
	resolvedNodes[originalNode.Index].Shape = outputShape

	inFeatures := resolvedX.Dimensions[resolvedX.Rank()-1]
	newData := &nodeFusedDense{
		options:         d.options,
		layout:          d.layout,
		batchSize:       d.batchSize,
		lhsCrossSize:    resolvedX.Size() / inFeatures,
		rhsCrossSize:    d.rhsCrossSize,
		contractingSize: d.contractingSize,
		dotNodeData:     d.dotNodeData,
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

	dotOutDType := outShape.DType
	if wNode.Shape.DType.IsHalfPrecision() {
		dotOutDType = dtypes.Float32
	}
	dotNodeData := &dot.NodeData{
		InputDType:      wNode.Shape.DType,
		OutputDType:     dotOutDType,
		Layout:          layout,
		BatchSize:       1,
		LHSCrossSize:    lhsCrossSize,
		RHSCrossSize:    rhsCrossSize,
		ContractingSize: contractingSize,
	}
	if xNode.IsConstant() {
		dotNodeData.LHSConstantData = gobackend.NewAnonymousConstantData()
	}
	if wNode.IsConstant() {
		dotNodeData.RHSConstantData = gobackend.NewAnonymousConstantData()
	}

	data := &nodeFusedDense{
		options:         options,
		layout:          layout,
		batchSize:       1,
		lhsCrossSize:    lhsCrossSize,
		rhsCrossSize:    rhsCrossSize,
		contractingSize: contractingSize,
		dotNodeData:     dotNodeData,
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

	outputShape := node.Shape
	if outputShape.IsDynamic() {
		xShape := node.Inputs[0].Shape
		if x != nil {
			xShape = x.RawShape
		}
		wShape := node.Inputs[1].Shape
		if weight != nil {
			wShape = weight.RawShape
		}
		var biasShape shapes.Shape
		if len(node.Inputs) > 2 {
			biasShape = node.Inputs[2].Shape
			if bias != nil {
				biasShape = bias.RawShape
			}
		}
		var err error
		outputShape, err = shapeinference.FusedDense(xShape, wShape, biasShape, data.options)
		if err != nil {
			return nil, err
		}
	}
	output, err := backend.GetBuffer(outputShape)
	if err != nil {
		return nil, err
	}

	if backend.NoOps || output.RawShape.Size() == 0 {
		return output, nil
	}

	switch output.RawShape.DType {
	case dtypes.Float32:
		var xFlat, wFlat []float32
		if x != nil && x.Flat != nil {
			xFlat = x.Flat.([]float32)
		}
		if weight != nil && weight.Flat != nil {
			wFlat = weight.Flat.([]float32)
		}
		outFlat := output.Flat.([]float32)

		var biasFlat []float32
		if bias != nil {
			biasFlat = bias.Flat.([]float32)
		}

		var actFn activations.InPlaceFn[float32]
		var biasTileFn activations.BiasTileFn[float32]
		var actMinChunk int
		if data.options.Activation.Type != compute.ActivationNone {
			var actMeta *activations.KernelMetadata
			actFn, actMeta = activations.GetWithMetadata[float32](data.options.Activation.Type)
			if actFn == nil {
				return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: activation %s not implemented for %s", data.options.Activation.Type, output.RawShape.DType)
			}
			actMinChunk = actMeta.EffectiveMinParallelChunk()
			biasTileFn = activations.GetBiasTile[float32](data.options.Activation.Type)
		}
		epilogue := matmul.Epilogue[float32]{
			Bias:                       biasFlat,
			Activation:                 actFn,
			BiasTileActivation:         biasTileFn,
			ActivationMinParallelChunk: actMinChunk,
		}

		err = matmul.ExecuteWithEpilogue(
			backend, data.layout, xFlat, wFlat,
			data.batchSize, data.lhsCrossSize, data.rhsCrossSize, data.contractingSize,
			outFlat, epilogue,
			data.dotNodeData,
		)
		if err != nil {
			return nil, err
		}

	case dtypes.Float64:
		var xFlat, wFlat []float64
		if x != nil && x.Flat != nil {
			xFlat = x.Flat.([]float64)
		}
		if weight != nil && weight.Flat != nil {
			wFlat = weight.Flat.([]float64)
		}
		outFlat := output.Flat.([]float64)

		var biasFlat []float64
		if bias != nil {
			biasFlat = bias.Flat.([]float64)
		}

		var actFn activations.InPlaceFn[float64]
		var biasTileFn activations.BiasTileFn[float64]
		var actMinChunk int
		if data.options.Activation.Type != compute.ActivationNone {
			var actMeta *activations.KernelMetadata
			actFn, actMeta = activations.GetWithMetadata[float64](data.options.Activation.Type)
			if actFn == nil {
				return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: activation %s not implemented for %s", data.options.Activation.Type, output.RawShape.DType)
			}
			actMinChunk = actMeta.EffectiveMinParallelChunk()
			biasTileFn = activations.GetBiasTile[float64](data.options.Activation.Type)
		}
		epilogue := matmul.Epilogue[float64]{
			Bias:                       biasFlat,
			Activation:                 actFn,
			BiasTileActivation:         biasTileFn,
			ActivationMinParallelChunk: actMinChunk,
		}

		err = matmul.ExecuteWithEpilogue(
			backend, data.layout, xFlat, wFlat,
			data.batchSize, data.lhsCrossSize, data.rhsCrossSize, data.contractingSize,
			outFlat, epilogue,
			data.dotNodeData,
		)
		if err != nil {
			return nil, err
		}

	case dtypes.BFloat16:
		conv := gobackend.ActiveHalfSliceConverters()
		if err := execFusedDenseHalf[bfloat16.BFloat16](backend, data, x, weight, bias, output, conv.BFloat16ToFloat32, conv.Float32ToBFloat16); err != nil {
			return nil, err
		}

	case dtypes.Float16:
		conv := gobackend.ActiveHalfSliceConverters()
		if err := execFusedDenseHalf[float16.Float16](backend, data, x, weight, bias, output, conv.Float16ToFloat32, conv.Float32ToFloat16); err != nil {
			return nil, err
		}

	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedDense: dtype %s", output.RawShape.DType)
	}

	return output, nil
}

// execFusedDenseHalf executes FusedDense for half-precision inputs (Float16 or BFloat16):
// matmul accumulates into a pooled Float32 buffer with optional bias/activation epilogue
// before converting back in parallel via SIMD converters.
func execFusedDenseHalf[T float16.Float16 | bfloat16.BFloat16](
	backend *gobackend.Backend,
	data *nodeFusedDense,
	x, weight, bias, output *gobackend.Buffer,
	toF32 func([]float32, []T),
	fromF32 func([]T, []float32),
) error {
	var xFlat, wFlat []T
	if x != nil && x.Flat != nil {
		xFlat = x.Flat.([]T)
	}
	if weight != nil && weight.Flat != nil {
		wFlat = weight.Flat.([]T)
	}
	outHalf := output.Flat.([]T)

	tmpF32Shape := shapes.Make(dtypes.Float32, output.RawShape.Dimensions...)
	tmpF32Buf, err := backend.GetBuffer(tmpF32Shape)
	if err != nil {
		return err
	}
	defer backend.PutBuffer(tmpF32Buf)
	tmpF32 := tmpF32Buf.Flat.([]float32)

	var biasF32 []float32
	if bias != nil {
		if biasHalf, ok := bias.Flat.([]T); ok && len(biasHalf) > 0 {
			var stackBias [2048]float32
			if len(biasHalf) <= len(stackBias) {
				biasF32 = stackBias[:len(biasHalf)]
			} else {
				biasF32Buf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, len(biasHalf)))
				if err != nil {
					return err
				}
				defer backend.PutBuffer(biasF32Buf)
				biasF32 = biasF32Buf.Flat.([]float32)
			}
			toF32(biasF32, biasHalf)
		} else if bF32, ok := bias.Flat.([]float32); ok {
			biasF32 = bF32
		}
	}

	var actFn activations.InPlaceFn[float32]
	var biasTileFn activations.BiasTileFn[float32]
	var actMinChunk int
	if data.options.Activation.Type != compute.ActivationNone {
		var actMeta *activations.KernelMetadata
		actFn, actMeta = activations.GetWithMetadata[float32](data.options.Activation.Type)
		if actFn == nil {
			return errors.Wrapf(compute.ErrNotImplemented, "FusedDense: activation %s not implemented for %s", data.options.Activation.Type, output.RawShape.DType)
		}
		actMinChunk = actMeta.EffectiveMinParallelChunk()
		biasTileFn = activations.GetBiasTile[float32](data.options.Activation.Type)
	}
	epilogue := matmul.Epilogue[float32]{
		Bias:                       biasF32,
		Activation:                 actFn,
		BiasTileActivation:         biasTileFn,
		ActivationMinParallelChunk: actMinChunk,
	}

	if err := matmul.ExecuteWithEpilogue(
		backend, data.layout, xFlat, wFlat,
		data.batchSize, data.lhsCrossSize, data.rhsCrossSize, data.contractingSize,
		tmpF32, epilogue,
		data.dotNodeData,
	); err != nil {
		return err
	}

	gobackend.ParallelConvertSlice(backend, outHalf, tmpF32, fromF32)
	return nil
}

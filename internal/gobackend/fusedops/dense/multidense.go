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
	gobackend.BuildFusedMultiDenseDataFn = buildFusedMultiDenseData
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeFusedDense] = execFusedMultiDense
}

type multiDenseSibling struct {
	options         compute.DenseConfig
	rhsCrossSize    int
	weightInputSlot int // index in leaderNode.Inputs for weight matrix W_j
	biasInputSlot   int // index in leaderNode.Inputs for optional bias_j, or -1
	dotNodeData     *dot.NodeData
}

// nodeFusedMultiDense stores metadata for a compile-time fused group of sibling
// FusedDense projections sharing the same activation input x (node.Inputs[0]).
type nodeFusedMultiDense struct {
	layout          dot.Layout
	inputDType      dtypes.DType
	batchSize       int
	lhsCrossSize    int
	contractingSize int
	siblings        []multiDenseSibling
	slotToSibling   map[int]int // weightInputSlot -> sibling index
}

// Compile-time checks that *nodeFusedMultiDense implements caching & dynamic specialization hooks.
var (
	_ gobackend.ConstantCacheNodeData = (*nodeFusedMultiDense)(nil)
	_ gobackend.RecomputableNodeData  = (*nodeFusedMultiDense)(nil)
)

func buildFusedMultiDenseData(siblings []*gobackend.Node, weightInputSlots, biasInputSlots []int) any {
	if len(siblings) < 2 || len(weightInputSlots) != len(siblings) || len(biasInputSlots) != len(siblings) {
		return nil
	}
	first, ok := siblings[0].Data.(*nodeFusedDense)
	if !ok || first == nil || first.dotNodeData == nil {
		return nil
	}
	md := &nodeFusedMultiDense{
		layout:          first.layout,
		inputDType:      first.dotNodeData.InputDType,
		batchSize:       first.batchSize,
		lhsCrossSize:    first.lhsCrossSize,
		contractingSize: first.contractingSize,
		siblings:        make([]multiDenseSibling, len(siblings)),
		slotToSibling:   make(map[int]int, len(siblings)),
	}
	for i, sibNode := range siblings {
		sibData, ok := sibNode.Data.(*nodeFusedDense)
		if !ok || sibData == nil || sibData.dotNodeData == nil {
			return nil
		}
		md.siblings[i] = multiDenseSibling{
			options:         sibData.options,
			rhsCrossSize:    sibData.rhsCrossSize,
			weightInputSlot: weightInputSlots[i],
			biasInputSlot:   biasInputSlots[i],
			dotNodeData:     sibData.dotNodeData,
		}
		md.slotToSibling[weightInputSlots[i]] = i
	}
	return md
}

func (d *nodeFusedMultiDense) CacheSignature(inputIdx int) string {
	if d == nil {
		return ""
	}
	sibIdx, ok := d.slotToSibling[inputIdx]
	if !ok {
		return ""
	}
	return d.siblings[sibIdx].dotNodeData.CacheSignature(1)
}

func (d *nodeFusedMultiDense) BindConstantData(inputIdx int, cd *gobackend.ConstantData) {
	if d == nil {
		return
	}
	if sibIdx, ok := d.slotToSibling[inputIdx]; ok {
		d.siblings[sibIdx].dotNodeData.BindConstantData(1, cd)
	}
}

func (d *nodeFusedMultiDense) IsInputCached(inputIdx int) bool {
	if d == nil {
		return false
	}
	if sibIdx, ok := d.slotToSibling[inputIdx]; ok {
		return d.siblings[sibIdx].dotNodeData.IsInputCached(1)
	}
	return false
}

// Recompute implements gobackend.RecomputableNodeData for dynamic shapes (e.g., variable seq_len prefill).
func (d *nodeFusedMultiDense) Recompute(
	_ *gobackend.Backend,
	resolvedNodes []*gobackend.Node,
	originalNode *gobackend.Node,
) (any, error) {
	resolvedX := resolvedNodes[originalNode.Inputs[0].Index].Shape
	inFeatures := resolvedX.Dimensions[resolvedX.Rank()-1]
	if len(originalNode.MultiOutputsShapes) == len(d.siblings) {
		for i, sib := range d.siblings {
			resolvedW := resolvedNodes[originalNode.Inputs[sib.weightInputSlot].Index].Shape
			var biasShape shapes.Shape
			if sib.biasInputSlot >= 0 {
				biasShape = resolvedNodes[originalNode.Inputs[sib.biasInputSlot].Index].Shape
			}
			outShape, err := shapeinference.FusedDense(resolvedX, resolvedW, biasShape, sib.options)
			if err != nil {
				return nil, err
			}
			originalNode.MultiOutputsShapes[i] = outShape
			if i < len(originalNode.MultiOutputsNodes) && originalNode.MultiOutputsNodes[i] != nil {
				sibNode := originalNode.MultiOutputsNodes[i]
				if sibNode.Index < len(resolvedNodes) && resolvedNodes[sibNode.Index] != nil {
					resolvedNodes[sibNode.Index].Shape = outShape
				}
			}
		}
	}
	return &nodeFusedMultiDense{
		layout:          d.layout,
		inputDType:      d.inputDType,
		batchSize:       d.batchSize,
		lhsCrossSize:    resolvedX.Size() / inFeatures,
		contractingSize: d.contractingSize,
		siblings:        d.siblings,
		slotToSibling:   d.slotToSibling,
	}, nil
}

// execFusedMultiDense executes a compile-time fused group of sibling FusedDense
// projections sharing x = inputs[0] via matmul.ExecuteMultiRHSWithEpilogue.
func execFusedMultiDense(
	backend *gobackend.Backend,
	node *gobackend.Node,
	inputs []*gobackend.Buffer,
	_ []bool,
) ([]*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedMultiDense)
	x := inputs[0]
	numSibs := len(data.siblings)
	outputs := make([]*gobackend.Buffer, numSibs)

	releaseOutputs := func(count int) {
		for i := range count {
			if outputs[i] != nil {
				backend.PutBuffer(outputs[i])
			}
		}
	}

	allEmpty := true
	for i, sib := range data.siblings {
		outShape := node.MultiOutputsShapes[i]
		if outShape.IsDynamic() {
			xShape := node.Inputs[0].Shape
			if x != nil {
				xShape = x.RawShape
			}
			wNode := node.Inputs[sib.weightInputSlot]
			wShape := wNode.Shape
			if wBuf := inputs[sib.weightInputSlot]; wBuf != nil {
				wShape = wBuf.RawShape
			}
			var biasShape shapes.Shape
			if sib.biasInputSlot >= 0 {
				biasShape = node.Inputs[sib.biasInputSlot].Shape
				if bBuf := inputs[sib.biasInputSlot]; bBuf != nil {
					biasShape = bBuf.RawShape
				}
			}
			var err error
			outShape, err = shapeinference.FusedDense(xShape, wShape, biasShape, sib.options)
			if err != nil {
				releaseOutputs(i)
				return nil, err
			}
		}
		buf, err := backend.GetBuffer(outShape)
		if err != nil {
			releaseOutputs(i)
			return nil, err
		}
		outputs[i] = buf
		if buf.RawShape.Size() > 0 {
			allEmpty = false
		}
	}

	if backend.NoOps || allEmpty {
		return outputs, nil
	}

	var err error
	switch data.inputDType {
	case dtypes.Float32:
		err = execFusedMultiDenseFloat[float32](backend, data, inputs, outputs)
	case dtypes.Float64:
		err = execFusedMultiDenseFloat[float64](backend, data, inputs, outputs)
	case dtypes.BFloat16:
		conv := gobackend.ActiveHalfSliceConverters()
		err = execFusedMultiDenseHalf[bfloat16.BFloat16](
			backend, data, inputs, outputs, conv.BFloat16ToFloat32, conv.Float32ToBFloat16)
	case dtypes.Float16:
		conv := gobackend.ActiveHalfSliceConverters()
		err = execFusedMultiDenseHalf[float16.Float16](
			backend, data, inputs, outputs, conv.Float16ToFloat32, conv.Float32ToFloat16)
	default:
		err = errors.Wrapf(compute.ErrNotImplemented, "FusedMultiDense: dtype %s", data.inputDType)
	}
	if err != nil {
		releaseOutputs(numSibs)
		return nil, err
	}
	return outputs, nil
}

// multiDensePrefillWeightLimitBytes caps combined RHS weight size for fused multi-target
// prefill (M > 1). Above this threshold (e.g. gate+up 36–72 MiB), sweeping all sibling weight
// matrices simultaneously per row-chunk evicts weights from L3/SLC across chunks, whereas
// executing siblings one-by-one keeps each weight matrix L3-resident and reuses a single tmpF32
// buffer. M=1 decode never reuses weights across rows and always executes all siblings together.
const multiDensePrefillWeightLimitBytes = 16 * 1024 * 1024

func shouldExecuteMultiRHSTogether(data *nodeFusedMultiDense) bool {
	if data.lhsCrossSize <= 1 {
		return true
	}
	totalCols := 0
	for _, sib := range data.siblings {
		totalCols += sib.rhsCrossSize
	}
	weightBytes := int64(data.contractingSize) * int64(totalCols) * int64(data.inputDType.Size())
	return weightBytes <= multiDensePrefillWeightLimitBytes
}

func execFusedMultiDenseFloat[T float32 | float64](
	backend *gobackend.Backend,
	data *nodeFusedMultiDense,
	inputs []*gobackend.Buffer,
	outputs []*gobackend.Buffer,
) error {
	var xFlat []T
	if x := inputs[0]; x != nil && x.Flat != nil {
		xFlat = x.Flat.([]T)
	}
	numSibs := len(data.siblings)
	var targetsBuf [8]matmul.MultiRHSTarget[T, T]
	var targets []matmul.MultiRHSTarget[T, T]
	if numSibs <= len(targetsBuf) {
		targets = targetsBuf[:numSibs]
	} else {
		targets = make([]matmul.MultiRHSTarget[T, T], numSibs)
	}

	for i, sib := range data.siblings {
		var wFlat, biasFlat []T
		if w := inputs[sib.weightInputSlot]; w != nil && w.Flat != nil {
			wFlat = w.Flat.([]T)
		}
		if sib.biasInputSlot >= 0 {
			if b := inputs[sib.biasInputSlot]; b != nil && b.Flat != nil {
				biasFlat = b.Flat.([]T)
			}
		}
		var actFn activations.InPlaceFn[T]
		var biasTileFn activations.BiasTileFn[T]
		var actMinChunk int
		if sib.options.Activation.Type != compute.ActivationNone {
			var actMeta *activations.KernelMetadata
			actFn, actMeta = activations.GetWithMetadata[T](sib.options.Activation.Type)
			if actFn == nil {
				return errors.Wrapf(compute.ErrNotImplemented,
					"FusedMultiDense: activation %s not implemented for %s",
					sib.options.Activation.Type, data.inputDType)
			}
			actMinChunk = actMeta.EffectiveMinParallelChunk()
			biasTileFn = activations.GetBiasTile[T](sib.options.Activation.Type)
		}
		targets[i] = matmul.MultiRHSTarget[T, T]{
			RHS:          wFlat,
			RHSCrossSize: sib.rhsCrossSize,
			Output:       outputs[i].Flat.([]T),
			Epilogue: matmul.Epilogue[T]{
				Bias:                       biasFlat,
				Activation:                 actFn,
				BiasTileActivation:         biasTileFn,
				ActivationMinParallelChunk: actMinChunk,
			},
			NodeData: sib.dotNodeData,
		}
	}

	if !shouldExecuteMultiRHSTogether(data) {
		for i := range targets {
			t := &targets[i]
			if err := matmul.ExecuteWithEpilogue[T, T](
				backend, data.layout, xFlat, t.RHS,
				data.batchSize, data.lhsCrossSize, t.RHSCrossSize, data.contractingSize,
				t.Output, t.Epilogue, t.NodeData,
			); err != nil {
				return err
			}
		}
		return nil
	}

	return matmul.ExecuteMultiRHSWithEpilogue[T, T](
		backend, data.layout, xFlat,
		data.batchSize, data.lhsCrossSize, data.contractingSize,
		targets,
	)
}

func execFusedMultiDenseHalf[T float16.Float16 | bfloat16.BFloat16](
	backend *gobackend.Backend,
	data *nodeFusedMultiDense,
	inputs []*gobackend.Buffer,
	outputs []*gobackend.Buffer,
	toF32 func([]float32, []T),
	fromF32 func([]T, []float32),
) error {
	var xFlat []T
	if x := inputs[0]; x != nil && x.Flat != nil {
		xFlat = x.Flat.([]T)
	}
	numSibs := len(data.siblings)
	fuseHalfStore := data.batchSize == 1

	// For large-weight prefill (M > 1), execute each sibling sequentially so only one weight
	// matrix and one tmpF32 scratch buffer occupy L3/SLC at a time, while still using inline
	// TilePostStore (Float32 -> Half) conversion per tile.
	if !shouldExecuteMultiRHSTogether(data) {
		for i, sib := range data.siblings {
			var wFlat []T
			if w := inputs[sib.weightInputSlot]; w != nil && w.Flat != nil {
				wFlat = w.Flat.([]T)
			}
			tmpF32Shape := shapes.Make(dtypes.Float32, outputs[i].RawShape.Dimensions...)
			tmpBuf, err := backend.GetBuffer(tmpF32Shape)
			if err != nil {
				return err
			}
			tmpF32 := tmpBuf.Flat.([]float32)

			var biasF32 []float32
			var biasBuf *gobackend.Buffer
			if sib.biasInputSlot >= 0 {
				if bias := inputs[sib.biasInputSlot]; bias != nil {
					if biasHalf, ok := bias.Flat.([]T); ok && len(biasHalf) > 0 {
						biasBuf, err = backend.GetBuffer(shapes.Make(dtypes.Float32, len(biasHalf)))
						if err != nil {
							backend.PutBuffer(tmpBuf)
							return err
						}
						biasF32 = biasBuf.Flat.([]float32)
						toF32(biasF32, biasHalf)
					} else if bF32, ok := bias.Flat.([]float32); ok {
						biasF32 = bF32
					}
				}
			}

			var actFn activations.InPlaceFn[float32]
			var biasTileFn activations.BiasTileFn[float32]
			var actMinChunk int
			if sib.options.Activation.Type != compute.ActivationNone {
				var actMeta *activations.KernelMetadata
				actFn, actMeta = activations.GetWithMetadata[float32](sib.options.Activation.Type)
				if actFn == nil {
					backend.PutBuffer(tmpBuf)
					if biasBuf != nil {
						backend.PutBuffer(biasBuf)
					}
					return errors.Wrapf(compute.ErrNotImplemented,
						"FusedMultiDense: activation %s not implemented for %s",
						sib.options.Activation.Type, data.inputDType)
				}
				actMinChunk = actMeta.EffectiveMinParallelChunk()
				biasTileFn = activations.GetBiasTile[float32](sib.options.Activation.Type)
			}

			epilogue := matmul.Epilogue[float32]{
				Bias:                       biasF32,
				Activation:                 actFn,
				BiasTileActivation:         biasTileFn,
				ActivationMinParallelChunk: actMinChunk,
			}
			outHalf := outputs[i].Flat.([]T)
			if fuseHalfStore {
				epilogue.TilePostStore = makeHalfTilePostStore(outHalf, fromF32)
			}

			err = matmul.ExecuteWithEpilogue[T, float32](
				backend, data.layout, xFlat, wFlat,
				data.batchSize, data.lhsCrossSize, sib.rhsCrossSize, data.contractingSize,
				tmpF32, epilogue, sib.dotNodeData,
			)
			if err == nil && !fuseHalfStore {
				gobackend.ParallelConvertSlice(backend, outHalf, tmpF32, fromF32)
			}
			backend.PutBuffer(tmpBuf)
			if biasBuf != nil {
				backend.PutBuffer(biasBuf)
			}
			if err != nil {
				return err
			}
		}
		return nil
	}

	var targetsBuf [8]matmul.MultiRHSTarget[T, float32]
	var targets []matmul.MultiRHSTarget[T, float32]
	if numSibs <= len(targetsBuf) {
		targets = targetsBuf[:numSibs]
	} else {
		targets = make([]matmul.MultiRHSTarget[T, float32], numSibs)
	}

	var tmpF32BufsBuf [8]*gobackend.Buffer
	var tmpF32Bufs []*gobackend.Buffer
	if numSibs <= len(tmpF32BufsBuf) {
		tmpF32Bufs = tmpF32BufsBuf[:numSibs]
	} else {
		tmpF32Bufs = make([]*gobackend.Buffer, numSibs)
	}
	var biasF32Bufs []*gobackend.Buffer
	defer func() {
		for _, b := range tmpF32Bufs {
			if b != nil {
				backend.PutBuffer(b)
			}
		}
		for _, b := range biasF32Bufs {
			if b != nil {
				backend.PutBuffer(b)
			}
		}
	}()

	for i, sib := range data.siblings {
		var wFlat []T
		if w := inputs[sib.weightInputSlot]; w != nil && w.Flat != nil {
			wFlat = w.Flat.([]T)
		}
		tmpF32Shape := shapes.Make(dtypes.Float32, outputs[i].RawShape.Dimensions...)
		tmpBuf, err := backend.GetBuffer(tmpF32Shape)
		if err != nil {
			return err
		}
		tmpF32Bufs[i] = tmpBuf
		tmpF32 := tmpBuf.Flat.([]float32)

		var biasF32 []float32
		if sib.biasInputSlot >= 0 {
			if bias := inputs[sib.biasInputSlot]; bias != nil {
				if biasHalf, ok := bias.Flat.([]T); ok && len(biasHalf) > 0 {
					biasBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, len(biasHalf)))
					if err != nil {
						return err
					}
					biasF32Bufs = append(biasF32Bufs, biasBuf)
					biasF32 = biasBuf.Flat.([]float32)
					toF32(biasF32, biasHalf)
				} else if bF32, ok := bias.Flat.([]float32); ok {
					biasF32 = bF32
				}
			}
		}

		var actFn activations.InPlaceFn[float32]
		var biasTileFn activations.BiasTileFn[float32]
		var actMinChunk int
		if sib.options.Activation.Type != compute.ActivationNone {
			var actMeta *activations.KernelMetadata
			actFn, actMeta = activations.GetWithMetadata[float32](sib.options.Activation.Type)
			if actFn == nil {
				return errors.Wrapf(compute.ErrNotImplemented,
					"FusedMultiDense: activation %s not implemented for %s",
					sib.options.Activation.Type, data.inputDType)
			}
			actMinChunk = actMeta.EffectiveMinParallelChunk()
			biasTileFn = activations.GetBiasTile[float32](sib.options.Activation.Type)
		}

		epilogue := matmul.Epilogue[float32]{
			Bias:                       biasF32,
			Activation:                 actFn,
			BiasTileActivation:         biasTileFn,
			ActivationMinParallelChunk: actMinChunk,
		}
		if fuseHalfStore {
			outHalf := outputs[i].Flat.([]T)
			epilogue.TilePostStore = makeHalfTilePostStore(outHalf, fromF32)
		}

		targets[i] = matmul.MultiRHSTarget[T, float32]{
			RHS:          wFlat,
			RHSCrossSize: sib.rhsCrossSize,
			Output:       tmpF32,
			Epilogue:     epilogue,
			NodeData:     sib.dotNodeData,
		}
	}

	if err := matmul.ExecuteMultiRHSWithEpilogue[T, float32](
		backend, data.layout, xFlat,
		data.batchSize, data.lhsCrossSize, data.contractingSize,
		targets,
	); err != nil {
		return err
	}

	if !fuseHalfStore {
		for i := range numSibs {
			outHalf := outputs[i].Flat.([]T)
			tmpF32 := tmpF32Bufs[i].Flat.([]float32)
			gobackend.ParallelConvertSlice(backend, outHalf, tmpF32, fromF32)
		}
	}
	return nil
}

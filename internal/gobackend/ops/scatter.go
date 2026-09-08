package ops

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/sets"
	"github.com/pkg/errors"
)

// ScatterMax implements the compute.Builder interface.

func init() {
	gobackend.RegisterScatterSum.Register(ScatterSum, gobackend.PriorityGeneric)
	gobackend.RegisterScatterMax.Register(ScatterMax, gobackend.PriorityGeneric)
	gobackend.RegisterScatterMin.Register(ScatterMin, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeScatterMax, gobackend.PriorityGeneric, execScatter)
	gobackend.SetNodeExecutor(compute.OpTypeScatterMin, gobackend.PriorityGeneric, execScatter)
	gobackend.SetNodeExecutor(compute.OpTypeScatterSum, gobackend.PriorityGeneric, execScatter)
}

func ScatterSum(f *gobackend.Function,
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return scatterImpls(
		f,
		compute.OpTypeScatterSum,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

func ScatterMax(f *gobackend.Function,
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return scatterImpls(
		f,
		compute.OpTypeScatterMax,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

func ScatterMin(f *gobackend.Function,
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (compute.Value, error) {
	return scatterImpls(
		f,
		compute.OpTypeScatterMin,
		operandOp,
		scatterIndicesOp,
		updatesOp,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
		indicesAreSorted,
		uniqueIndices,
	)
}

func scatterImpls(
	f *gobackend.Function,
	scatterOpType compute.OpType,
	operandOp, scatterIndicesOp, updatesOp compute.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (
	compute.Value, error) {
	inputs, err := f.VerifyAndCastValues(scatterOpType.String(), operandOp, scatterIndicesOp, updatesOp)
	if err != nil {
		return nil, err
	}
	operand, indices, updates := inputs[0], inputs[1], inputs[2]
	// Check that parameters are valid.
	outputShape, err := shapeinference.Scatter(
		operand.Shape,
		indices.Shape,
		updates.Shape,
		indexVectorAxis,
		updateWindowAxes,
		insertedWindowAxes,
		scatterAxesToOperandAxes,
	)
	if err != nil {
		return nil, err
	}

	// The output shape of the scatter is the operand shape.
	data := &scatterNode{
		updateWindowAxes:         updateWindowAxes,
		insertedWindowAxes:       insertedWindowAxes,
		scatterAxesToOperandAxes: scatterAxesToOperandAxes,
		indexVectorAxis:          indexVectorAxis,
		indicesAreSorted:         indicesAreSorted,
		uniqueIndices:            uniqueIndices,
	}
	node, _ := f.GetOrCreateNode(scatterOpType, outputShape, []*gobackend.Node{operand, indices, updates}, data)
	return node, nil
}

// scatterNode is attached to the gobackend.Node.data field for ScatterMax, ScatterMin, ScatterSum.
type scatterNode struct {
	indexVectorAxis                                                int
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int
	indicesAreSorted, uniqueIndices                                bool
}

// EqualNodeData implements nodeDataComparable for scatterNode.
func (s *scatterNode) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*scatterNode)
	if s.indexVectorAxis != o.indexVectorAxis ||
		s.indicesAreSorted != o.indicesAreSorted ||
		s.uniqueIndices != o.uniqueIndices {
		return false
	}
	return slices.Equal(s.updateWindowAxes, o.updateWindowAxes) &&
		slices.Equal(s.insertedWindowAxes, o.insertedWindowAxes) &&
		slices.Equal(s.scatterAxesToOperandAxes, o.scatterAxesToOperandAxes)
}

// execScatter implements the Scatter operation (Max, Min, Sum variants).
func execScatter(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	operand, indices, updates := inputs[0], inputs[1], inputs[2]
	scatterParams, ok := node.Data.(*scatterNode)
	if !ok {
		return nil, errors.Errorf("internal error: node.data for Scatter op is not *scatterData, but %T", node.Data)
	}

	// Output starts as a copy of the operand.
	// We might be able to reuse the operand buffer if it's owned.
	var output *gobackend.Buffer
	var err error
	if inputsOwned[0] {
		output = operand
		inputs[0] = nil // Mark operand as consumed.
	} else {
		output, err = backend.CloneBuffer(operand) // Creates a new buffer with copied data.
		if err != nil {
			return nil, err
		}
	}
	output.RawShape = node.Shape // Output shape is the same as operand shape.
	if backend.NoOps {
		return output, nil
	}

	// Dispatch to a type-specific scatter loop based on the operation type.
	dtype := output.RawShape.DType
	type scatterFnT = func(opType compute.OpType, output, indices, updates *gobackend.Buffer, scatterParams *scatterNode) error
	tmpAny, tmpErr := scatterDTypeMap.Get(dtype)
	if tmpErr != nil {
		panic(tmpErr)
	}
	scatterFn := tmpAny.(scatterFnT)
	err = scatterFn(node.OpType, output, indices, updates, scatterParams)
	if err != nil {
		return nil, err
	}
	return output, nil
}

//gobackend:dtypemap execScatterGeneric ints,uints,floats,half
var scatterDTypeMap = gobackend.NewDTypeMap("ScatterMax")

// execScatterGeneric assumes the operand is already copied to the output.
func execScatterGeneric[T gobackend.SupportedTypesConstraints](opType compute.OpType, output, indices, updates *gobackend.Buffer,
	scatterParams *scatterNode) error {
	// Get combineFn for operand's dtype.
	dtype := output.RawShape.DType
	type combineFnT = func(a, b T) T
	var combineFn combineFnT
	switch opType { //nolint:exhaustive
	case compute.OpTypeScatterMax:
		tmpAny, tmpErr := combineMaxDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		combineFn = tmpAny.(combineFnT) //nolint:errcheck
	case compute.OpTypeScatterMin:
		tmpAny, tmpErr := combineMinDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		combineFn = tmpAny.(combineFnT) //nolint:errcheck
	case compute.OpTypeScatterSum:
		tmpAny, tmpErr := combineSumDTypeMap.Get(dtype)
		if tmpErr != nil {
			panic(tmpErr)
		}
		combineFn = tmpAny.(combineFnT) //nolint:errcheck
	default:
		return errors.Errorf("unsupported scatter op type %q", opType)
	}
	_ = combineFn

	outputShape := output.RawShape
	outputFlat := output.Flat.([]T) //nolint:errcheck  // it will panic
	indicesFlat := indices.Flat
	updatesShape := updates.RawShape
	updatesFlat := updates.Flat.([]T) //nolint:errcheck  // it will panic

	// Initialize gather of the scatter indices.
	indicesShape := indices.RawShape
	tmpAny, tmpErr := dereferenceIntsDTypeMap.Get(indicesShape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	deferenceIndicesFn := tmpAny.(func(flat any, in, out []int))
	_, _ = indicesFlat, deferenceIndicesFn
	indicesIt := newSubIndicesIterator(indices.RawShape, scatterParams.indexVectorAxis)
	indexVectorStride := 1
	indexVectorSize := 1
	if scatterParams.indexVectorAxis != indicesShape.Rank() {
		indexVectorSize = indices.RawShape.Dimensions[scatterParams.indexVectorAxis]
		indexVectorStride = indicesIt.PerAxisStride[scatterParams.indexVectorAxis]
	}
	indirectScatterIndices := make([]int, indexVectorSize)
	elemIndices := make([]int, indexVectorSize)
	// fmt.Printf("\tindexVectorSize=%d, indexVectorStride=%d\n", numBatchAxes, indexVectorStride)

	// Initialize iterator over the updates:
	updatesIt := newSubIndicesIterator(updatesShape, scatterParams.updateWindowAxes...)
	numBatchAxes := indicesShape.Rank() - 1
	if scatterParams.indexVectorAxis == indicesShape.Rank() {
		numBatchAxes++
	}
	// Precompute mapping from updates window axes to operand/output axes.
	insertedWindowAxesSet := sets.MakeWith(scatterParams.insertedWindowAxes...)
	operandUpdatedWindowAxes := make([]int, 0, outputShape.Rank()-len(scatterParams.insertedWindowAxes))
	for axis := range outputShape.Rank() {
		if !insertedWindowAxesSet.Has(axis) {
			operandUpdatedWindowAxes = append(operandUpdatedWindowAxes, axis)
		}
	}
	numWinAxes := len(scatterParams.updateWindowAxes)
	winDims := make([]int, numWinAxes)
	updWinStrides := make([]int, numWinAxes)
	outWinStrides := make([]int, numWinAxes)
	outputStrides := outputShape.Strides()
	updatesStrides := updatesShape.Strides()
	for i, uAxis := range scatterParams.updateWindowAxes {
		oAxis := operandUpdatedWindowAxes[i]
		winDims[i] = updatesShape.Dimensions[uAxis]
		updWinStrides[i] = updatesStrides[uAxis]
		outWinStrides[i] = outputStrides[oAxis]
	}
	winCoords := make([]int, numWinAxes)

	type combineSliceFnT = func(out, upd []T)
	var combineSliceFn combineSliceFnT
	switch opType { //nolint:exhaustive
	case compute.OpTypeScatterMax:
		if tmpAny, tmpErr := combineSliceMaxDTypeMap.Get(dtype); tmpErr == nil {
			combineSliceFn = tmpAny.(combineSliceFnT)
		}
	case compute.OpTypeScatterMin:
		if tmpAny, tmpErr := combineSliceMinDTypeMap.Get(dtype); tmpErr == nil {
			combineSliceFn = tmpAny.(combineSliceFnT)
		}
	case compute.OpTypeScatterSum:
		if tmpAny, tmpErr := combineSliceSumDTypeMap.Get(dtype); tmpErr == nil {
			combineSliceFn = tmpAny.(combineSliceFnT)
		}
	}

	// Determine largest contiguous chunk of elements along trailing window axes.
	chunkElements := 1
	var leadingWinAxes []int
	for axis := numWinAxes - 1; axis >= 0; axis-- {
		if winDims[axis] == 1 {
			continue
		}
		if updWinStrides[axis] == chunkElements && outWinStrides[axis] == chunkElements {
			chunkElements *= winDims[axis]
		} else {
			for a := 0; a <= axis; a++ {
				if winDims[a] > 1 {
					leadingWinAxes = append(leadingWinAxes, a)
				}
			}
			break
		}
	}
	numWinElements := 1
	for _, dim := range winDims {
		numWinElements *= dim
	}
	numChunks := numWinElements / chunkElements

	// Outer-loop: range over the pointed indices
	for {
		// Find scatter indices -> where the values are going to be combined in the output:
		flatIndirectIndex := indicesIt.FlatIdx
		for ii := range indexVectorSize {
			indirectScatterIndices[ii] = flatIndirectIndex
			flatIndirectIndex += indexVectorStride
		}
		deferenceIndicesFn(indicesFlat, indirectScatterIndices, elemIndices)

		// Base offset in output:
		outputBaseIdx := 0
		for scatterAxis, idx := range elemIndices {
			outputAxis := scatterParams.scatterAxesToOperandAxes[scatterAxis]
			outputBaseIdx += idx * outputStrides[outputAxis]
		}

		// Inner-loop: combine slice (window) of update into output.
		if numWinAxes == 0 {
			outputFlat[outputBaseIdx] = combineFn(outputFlat[outputBaseIdx], updatesFlat[updatesIt.FlatIdx])
		} else if combineSliceFn != nil && chunkElements > 1 {
			if numChunks == 1 {
				combineSliceFn(
					outputFlat[outputBaseIdx:outputBaseIdx+chunkElements],
					updatesFlat[updatesIt.FlatIdx:updatesIt.FlatIdx+chunkElements],
				)
			} else {
				for i := range winCoords {
					winCoords[i] = 0
				}
				outOffset := 0
				updOffset := 0
				for range numChunks {
					combineSliceFn(
						outputFlat[outputBaseIdx+outOffset:outputBaseIdx+outOffset+chunkElements],
						updatesFlat[updatesIt.FlatIdx+updOffset:updatesIt.FlatIdx+updOffset+chunkElements],
					)
					for i := len(leadingWinAxes) - 1; i >= 0; i-- {
						axis := leadingWinAxes[i]
						winCoords[axis]++
						outOffset += outWinStrides[axis]
						updOffset += updWinStrides[axis]
						if winCoords[axis] < winDims[axis] {
							break
						}
						winCoords[axis] = 0
						outOffset -= outWinStrides[axis] * winDims[axis]
						updOffset -= updWinStrides[axis] * winDims[axis]
					}
				}
			}
		} else {
			for i := range winCoords {
				winCoords[i] = 0
			}
			outOffset := 0
			updOffset := 0
			for {
				outputIdx := outputBaseIdx + outOffset
				updatesIdx := updatesIt.FlatIdx + updOffset
				outputFlat[outputIdx] = combineFn(outputFlat[outputIdx], updatesFlat[updatesIdx])

				// Step window coordinates:
				axis := numWinAxes - 1
				for ; axis >= 0; axis-- {
					winCoords[axis]++
					outOffset += outWinStrides[axis]
					updOffset += updWinStrides[axis]
					if winCoords[axis] < winDims[axis] {
						break
					}
					winCoords[axis] = 0
					outOffset -= outWinStrides[axis] * winDims[axis]
					updOffset -= updWinStrides[axis] * winDims[axis]
				}
				if axis < 0 {
					break
				}
			}
		}

		// Next in indices:
		if !indicesIt.Increment() {
			break
		}
		updatesIt.Increment()
	}
	return nil
}

type subIndicesIterator struct {
	// FlatIdx is the current flat index to the shape.
	FlatIdx int

	// PerAxisIdx is the current indices in the shape.
	PerAxisIdx []int

	PerAxisSize   []int
	PerAxisStride []int
}

func newSubIndicesIterator(shape shapes.Shape, skipAxes ...int) *subIndicesIterator {
	rank := shape.Rank()
	it := &subIndicesIterator{
		PerAxisIdx:  make([]int, rank),
		PerAxisSize: slices.Clone(shape.Dimensions),
	}
	it.PerAxisStride = shape.Strides()
	for _, axis := range skipAxes {
		if axis < rank {
			// Set size for axis we don't want to iterate over to 1.
			it.PerAxisSize[axis] = 1
		}
	}
	return it
}

// Increment indices. It returns true if the new index is still valid, or false if it reached the end.
func (it *subIndicesIterator) Increment() bool {
	if it.FlatIdx < 0 {
		return false
	}
	rank := len(it.PerAxisSize)
	for axis := rank - 1; axis >= 0; axis-- {
		if it.PerAxisSize[axis] == 1 {
			continue
		}
		it.PerAxisIdx[axis]++
		it.FlatIdx += it.PerAxisStride[axis]
		if it.PerAxisIdx[axis] < it.PerAxisSize[axis] {
			return true
		}

		// We are going to move to the next axis.
		if axis == 0 {
			break
		}
		it.PerAxisIdx[axis] = 0
		it.FlatIdx -= it.PerAxisStride[axis] * it.PerAxisSize[axis] // Rewind FlatIdx to start of the current axis.
	}

	// Reached end.
	it.FlatIdx = -1
	return false
}

//gobackend:dtypemap dereferenceIntsGeneric ints,uints
var dereferenceIntsDTypeMap = gobackend.NewDTypeMap("Scatter Indices")

func dereferenceIntsGeneric[T gobackend.PODIntegerConstraints](flatAny any, indicesIn, indicesOut []int) {
	flat := flatAny.([]T)
	for ii, index := range indicesIn {
		indicesOut[ii] = int(flat[index])
	}
}

var (
	//gobackend:dtypemap combineForScatterMaxGeneric ints,uints,floats
	combineMaxDTypeMap = gobackend.NewDTypeMap("Max(a, b) for ScatterMax")
	//gobackend:dtypemap combineForScatterMinGeneric ints,uints,floats
	combineMinDTypeMap = gobackend.NewDTypeMap("Min(a, b) for ScatterMin")
	//gobackend:dtypemap combineForScatterSumGeneric ints,uints,floats
	combineSumDTypeMap = gobackend.NewDTypeMap("Sum(a, b) for ScatterSum")

	// combineSliceMaxDTypeMap maps DType to func(out, upd []T).
	combineSliceMaxDTypeMap = gobackend.NewDTypeMap("Slice Max(out, upd) for ScatterMax")
	// combineSliceMinDTypeMap maps DType to func(out, upd []T).
	combineSliceMinDTypeMap = gobackend.NewDTypeMap("Slice Min(out, upd) for ScatterMin")
	// combineSliceSumDTypeMap maps DType to func(out, upd []T).
	combineSliceSumDTypeMap = gobackend.NewDTypeMap("Slice Sum(out, upd) for ScatterSum")
)

func init() {
	combineMaxDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, combineForScatterMaxBFloat16)
	combineMinDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, combineForScatterMinBFloat16)
	combineSumDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, combineForScatterSumBFloat16)

	combineMaxDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, combineForScatterMaxFloat16)
	combineMinDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, combineForScatterMinFloat16)
	combineSumDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, combineForScatterSumFloat16)

	combineSliceMaxDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityGeneric, combineSliceForScatterMaxBFloat16)
	combineSliceMinDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityGeneric, combineSliceForScatterMinBFloat16)
	combineSliceSumDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityGeneric, combineSliceForScatterSumBFloat16)

	combineSliceMaxDTypeMap.Register(dtypes.Float16, gobackend.PriorityGeneric, combineSliceForScatterMaxFloat16)
	combineSliceMinDTypeMap.Register(dtypes.Float16, gobackend.PriorityGeneric, combineSliceForScatterMinFloat16)
	combineSliceSumDTypeMap.Register(dtypes.Float16, gobackend.PriorityGeneric, combineSliceForScatterSumFloat16)
}

func combineForScatterMaxGeneric[T gobackend.PODNumericConstraints](a, b T) T {
	return max(a, b)
}

func combineForScatterMaxBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(max(a.Float32(), b.Float32()))
}

func combineForScatterMaxFloat16(a, b float16.Float16) float16.Float16 {
	return float16.FromFloat32(max(a.Float32(), b.Float32()))
}

func combineForScatterMinGeneric[T gobackend.PODNumericConstraints](a, b T) T {
	return min(a, b)
}

func combineForScatterMinBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(min(a.Float32(), b.Float32()))
}

func combineForScatterMinFloat16(a, b float16.Float16) float16.Float16 {
	return float16.FromFloat32(min(a.Float32(), b.Float32()))
}

func combineForScatterSumGeneric[T gobackend.PODNumericConstraints](a, b T) T {
	return a + b
}

func combineForScatterSumBFloat16(a, b bfloat16.BFloat16) bfloat16.BFloat16 {
	return bfloat16.FromFloat32(a.Float32() + b.Float32())
}

func combineForScatterSumFloat16(a, b float16.Float16) float16.Float16 {
	return float16.FromFloat32(a.Float32() + b.Float32())
}

func combineSliceForScatterMaxGeneric[T gobackend.PODNumericConstraints](out, upd []T) {
	for i, v := range upd {
		out[i] = max(out[i], v)
	}
}

func combineSliceForScatterMaxBFloat16(out, upd []bfloat16.BFloat16) {
	for i, v := range upd {
		out[i] = bfloat16.FromFloat32(max(out[i].Float32(), v.Float32()))
	}
}

func combineSliceForScatterMaxFloat16(out, upd []float16.Float16) {
	for i, v := range upd {
		out[i] = float16.FromFloat32(max(out[i].Float32(), v.Float32()))
	}
}

func combineSliceForScatterMinGeneric[T gobackend.PODNumericConstraints](out, upd []T) {
	for i, v := range upd {
		out[i] = min(out[i], v)
	}
}

func combineSliceForScatterMinBFloat16(out, upd []bfloat16.BFloat16) {
	for i, v := range upd {
		out[i] = bfloat16.FromFloat32(min(out[i].Float32(), v.Float32()))
	}
}

func combineSliceForScatterMinFloat16(out, upd []float16.Float16) {
	for i, v := range upd {
		out[i] = float16.FromFloat32(min(out[i].Float32(), v.Float32()))
	}
}

func combineSliceForScatterSumGeneric[T gobackend.PODNumericConstraints](out, upd []T) {
	for i, v := range upd {
		out[i] += v
	}
}

func combineSliceForScatterSumBFloat16(out, upd []bfloat16.BFloat16) {
	for i, v := range upd {
		out[i] = bfloat16.FromFloat32(out[i].Float32() + v.Float32())
	}
}

func combineSliceForScatterSumFloat16(out, upd []float16.Float16) {
	for i, v := range upd {
		out[i] = float16.FromFloat32(out[i].Float32() + v.Float32())
	}
}

func registerGenericCombineSlice[T gobackend.PODNumericConstraints](dtype dtypes.DType) {
	combineSliceMaxDTypeMap.Register(dtype, gobackend.PriorityGeneric, combineSliceForScatterMaxGeneric[T])
	combineSliceMinDTypeMap.Register(dtype, gobackend.PriorityGeneric, combineSliceForScatterMinGeneric[T])
	combineSliceSumDTypeMap.Register(dtype, gobackend.PriorityGeneric, combineSliceForScatterSumGeneric[T])
}

func init() {
	registerGenericCombineSlice[float32](dtypes.Float32)
	registerGenericCombineSlice[float64](dtypes.Float64)
	registerGenericCombineSlice[int8](dtypes.Int8)
	registerGenericCombineSlice[int16](dtypes.Int16)
	registerGenericCombineSlice[int32](dtypes.Int32)
	registerGenericCombineSlice[int64](dtypes.Int64)
	registerGenericCombineSlice[uint8](dtypes.Uint8)
	registerGenericCombineSlice[uint16](dtypes.Uint16)
	registerGenericCombineSlice[uint32](dtypes.Uint32)
	registerGenericCombineSlice[uint64](dtypes.Uint64)
}

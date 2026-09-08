// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

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

func init() {
	gobackend.RegisterReduceMax.Register(ReduceMax, gobackend.PriorityGeneric)
	gobackend.RegisterReduceMin.Register(ReduceMin, gobackend.PriorityGeneric)
	gobackend.RegisterReduceSum.Register(ReduceSum, gobackend.PriorityGeneric)
	gobackend.RegisterReduceProduct.Register(ReduceProduct, gobackend.PriorityGeneric)
	gobackend.RegisterReduceBitwiseAnd.Register(ReduceBitwiseAnd, gobackend.PriorityGeneric)
	gobackend.RegisterReduceBitwiseOr.Register(ReduceBitwiseOr, gobackend.PriorityGeneric)
	gobackend.RegisterReduceBitwiseXor.Register(ReduceBitwiseXor, gobackend.PriorityGeneric)
	gobackend.RegisterReduceLogicalAnd.Register(ReduceLogicalAnd, gobackend.PriorityGeneric)
	gobackend.RegisterReduceLogicalOr.Register(ReduceLogicalOr, gobackend.PriorityGeneric)
	gobackend.RegisterReduceLogicalXor.Register(ReduceLogicalXor, gobackend.PriorityGeneric)

	gobackend.SetNodeExecutor(compute.OpTypeReduceMax, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceMin, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceSum, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceProduct, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceBitwiseAnd, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceBitwiseOr, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceBitwiseXor, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceLogicalAnd, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceLogicalOr, gobackend.PriorityGeneric, execReduce)
	gobackend.SetNodeExecutor(compute.OpTypeReduceLogicalXor, gobackend.PriorityGeneric, execReduce)
}

// ReduceMax reduces the operand by taking the maximum value along the given axes.
func ReduceMax(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceMax, operand, axis...)
}

// ReduceMin reduces the operand by taking the minimum value along the given axes.
func ReduceMin(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceMin, operand, axis...)
}

// ReduceSum reduces the operand by taking the sum of values along the given axes.
func ReduceSum(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceSum, operand, axis...)
}

// ReduceProduct reduces the operand by taking the product of values along the given axes.
func ReduceProduct(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceProduct, operand, axis...)
}

// ReduceBitwiseAnd reduces the operand by taking the bitwise AND of values along the given axes.
func ReduceBitwiseAnd(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceBitwiseAnd, operand, axis...)
}

// ReduceBitwiseOr reduces the operand by taking the bitwise OR of values along the given axes.
func ReduceBitwiseOr(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceBitwiseOr, operand, axis...)
}

// ReduceBitwiseXor reduces the operand by taking the bitwise XOR of values along the given axes.
func ReduceBitwiseXor(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceBitwiseXor, operand, axis...)
}

// ReduceLogicalAnd reduces the operand by taking the logical AND of values along the given axes.
func ReduceLogicalAnd(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceLogicalAnd, operand, axis...)
}

// ReduceLogicalOr reduces the operand by taking the logical OR of values along the given axes.
func ReduceLogicalOr(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceLogicalOr, operand, axis...)
}

// ReduceLogicalXor reduces the operand by taking the logical XOR of values along the given axes.
func ReduceLogicalXor(f *gobackend.Function, operand compute.Value, axis ...int) (compute.Value, error) {
	return reduceImpls(f, compute.OpTypeReduceLogicalXor, operand, axis...)
}

// ReducePattern classifies the geometric shape of the reduction for fast-path execution.
type ReducePattern int

const (
	// ReduceAll indicates all axes of the tensor are being reduced, yielding a scalar or size-1 output.
	ReduceAll ReducePattern = iota

	// ReduceTrailing indicates a contiguous suffix of axes is reduced: operand [A, B] -> output [A].
	ReduceTrailing

	// ReduceLeading indicates a contiguous prefix of axes is reduced: operand [A, B] -> output [B].
	ReduceLeading

	// ReduceMiddle indicates a contiguous middle segment of axes is reduced: operand [A, B, C] -> output [A, C].
	ReduceMiddle

	// ReduceGeneral indicates arbitrary non-contiguous axes requiring index calculation.
	ReduceGeneral
)

// ReduceConfig contains pre-computed metadata for reduction operations.
type ReduceConfig struct {
	Axes    []int
	Pattern ReducePattern
	A, B, C int
}

var (
	_ gobackend.NodeDataComparable   = (*ReduceConfig)(nil)
	_ gobackend.RecomputableNodeData = (*ReduceConfig)(nil)
)

// EqualNodeData implements gobackend.NodeDataComparable.
func (c *ReduceConfig) EqualNodeData(other gobackend.NodeDataComparable) bool {
	if c == nil && other == nil {
		return true
	}
	if c == nil || other == nil {
		return false
	}
	otherC, ok := other.(*ReduceConfig)
	if !ok {
		return false
	}
	return c.Pattern == otherC.Pattern &&
		c.A == otherC.A &&
		c.B == otherC.B &&
		c.C == otherC.C &&
		slices.Equal(c.Axes, otherC.Axes)
}

// Recompute implements gobackend.RecomputableNodeData.
func (c *ReduceConfig) Recompute(backend *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
	operand := resolvedNodes[originalNode.Inputs[0].Index]
	cfg := DetermineReduceConfig(operand.Shape, c.Axes)
	return &cfg, nil
}

// DetermineReduceConfig classifies the reduction pattern for a given operand shape and reduce axes.
func DetermineReduceConfig(shape shapes.Shape, axes []int) ReduceConfig {
	rank := shape.Rank()
	if rank == 0 {
		return ReduceConfig{Pattern: ReduceAll}
	}

	// Normalize and sort axes.
	sortedAxes := slices.Clone(axes)
	if len(sortedAxes) == 0 {
		sortedAxes = make([]int, rank)
		for i := range rank {
			sortedAxes[i] = i
		}
	}
	slices.Sort(sortedAxes)
	sortedAxes = slices.Compact(sortedAxes)

	cfg := ReduceConfig{
		Axes:    sortedAxes,
		Pattern: ReduceGeneral,
	}

	if shape.IsDynamic() {
		return cfg
	}

	if len(sortedAxes) == rank {
		cfg.Pattern = ReduceAll
		return cfg
	}

	// Check if sortedAxes form a contiguous range [k1, k2).
	isContiguous := true
	for i := 1; i < len(sortedAxes); i++ {
		if sortedAxes[i] != sortedAxes[i-1]+1 {
			isContiguous = false
			break
		}
	}

	if !isContiguous {
		return cfg
	}

	k1 := sortedAxes[0]
	k2 := sortedAxes[len(sortedAxes)-1] + 1

	if k1 == 0 && k2 == rank {
		cfg.Pattern = ReduceAll
		return cfg
	}

	if k1 == 0 && k2 < rank {
		// ReduceLeading: prefix [0, k2) is reduced, suffix [k2, rank) is output.
		a := 1
		for axis := 0; axis < k2; axis++ {
			a *= shape.Dimensions[axis]
		}
		b := 1
		for axis := k2; axis < rank; axis++ {
			b *= shape.Dimensions[axis]
		}
		cfg.Pattern = ReduceLeading
		cfg.A = a
		cfg.B = b
		return cfg
	}

	if k1 > 0 && k2 == rank {
		// ReduceTrailing: prefix [0, k1) is output, suffix [k1, rank) is reduced.
		a := 1
		for axis := 0; axis < k1; axis++ {
			a *= shape.Dimensions[axis]
		}
		b := 1
		for axis := k1; axis < rank; axis++ {
			b *= shape.Dimensions[axis]
		}
		cfg.Pattern = ReduceTrailing
		cfg.A = a
		cfg.B = b
		return cfg
	}

	if k1 > 0 && k2 < rank {
		// ReduceMiddle: prefix [0, k1) is output, middle [k1, k2) is reduced, suffix [k2, rank) is output.
		a := 1
		for axis := 0; axis < k1; axis++ {
			a *= shape.Dimensions[axis]
		}
		b := 1
		for axis := k1; axis < k2; axis++ {
			b *= shape.Dimensions[axis]
		}
		c := 1
		for axis := k2; axis < rank; axis++ {
			c *= shape.Dimensions[axis]
		}
		cfg.Pattern = ReduceMiddle
		cfg.A = a
		cfg.B = b
		cfg.C = c
		return cfg
	}

	return cfg
}

func reduceImpls(f *gobackend.Function, reduceOpType compute.OpType, operandValue compute.Value, axes ...int) (*gobackend.Node, error) {
	inputs, err := f.VerifyAndCastValues(reduceOpType.String(), operandValue)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]
	if len(axes) == 0 {
		// Default if no axes are given, is to reduce all axes.
		for axis := range operand.Shape.Rank() {
			axes = append(axes, axis)
		}
	}
	outputShape, err := shapeinference.Reduce(operand.Shape, axes)
	if err != nil {
		return nil, err
	}
	outputShape.DType = operand.Shape.DType

	var data any
	if !operand.Shape.IsDynamic() && !outputShape.IsDynamic() {
		cfg := DetermineReduceConfig(operand.Shape, axes)
		data = &cfg
	} else {
		cfg := ReduceConfig{Axes: slices.Clone(axes), Pattern: ReduceGeneral}
		data = &cfg
	}
	node, _ := f.GetOrCreateNode(reduceOpType, outputShape, inputs, data)
	return node, nil
}

func execReduce(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
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
		// Reduce all axes.
		for axis := range operand.RawShape.Rank() {
			reduceAxes = append(reduceAxes, axis)
		}
		cfg = DetermineReduceConfig(operand.RawShape, reduceAxes)
	}
	if len(reduceAxes) == 0 {
		// Identity.
		output, err := backend.GetBuffer(operand.RawShape)
		if err != nil {
			return nil, err
		}
		gobackend.CopyFlat(output.Flat, operand.Flat)
		return output, nil
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
		return nil, err
	}
	if backend.NoOps {
		return output, nil
	}

	var it *ReduceOutputIterator
	if (node.OpType == compute.OpTypeReduceSum || node.OpType == compute.OpTypeReduceMax || node.OpType == compute.OpTypeReduceMin) && cfg.Pattern != ReduceGeneral {
		it = &ReduceOutputIterator{Config: cfg}
	} else {
		it = NewReduceOutputIterator(operand.RawShape.Dimensions, reduceAxes)
		it.Config = cfg
	}
	dtype := output.RawShape.DType

	var reduceFn GenericReduceFn
	switch node.OpType { //nolint:exhaustive
	case compute.OpTypeReduceMax:
		tmpAny, tmpErr := reduceMaxDTypeMap.Get(dtype)
		if tmpErr != nil {
			return nil, tmpErr
		}
		reduceFn = tmpAny.(GenericReduceFn)
	case compute.OpTypeReduceMin:
		tmpAny, tmpErr := reduceMinDTypeMap.Get(dtype)
		if tmpErr != nil {
			return nil, tmpErr
		}
		reduceFn = tmpAny.(GenericReduceFn)
	case compute.OpTypeReduceSum:
		tmpAny, tmpErr := reduceSumDTypeMap.Get(dtype)
		if tmpErr != nil {
			return nil, tmpErr
		}
		reduceFn = tmpAny.(GenericReduceFn)
	case compute.OpTypeReduceProduct:
		tmpAny, tmpErr := reduceProductDTypeMap.Get(dtype)
		if tmpErr != nil {
			return nil, tmpErr
		}
		reduceFn = tmpAny.(GenericReduceFn)
	case compute.OpTypeReduceBitwiseAnd:
		tmpAny, tmpErr := reduceBitwiseAndDTypeMap.Get(dtype)
		if tmpErr != nil {
			return nil, tmpErr
		}
		reduceFn = tmpAny.(GenericReduceFn)
	case compute.OpTypeReduceBitwiseOr:
		tmpAny, tmpErr := reduceBitwiseOrDTypeMap.Get(dtype)
		if tmpErr != nil {
			return nil, tmpErr
		}
		reduceFn = tmpAny.(GenericReduceFn)
	case compute.OpTypeReduceBitwiseXor:
		tmpAny, tmpErr := reduceBitwiseXorDTypeMap.Get(dtype)
		if tmpErr != nil {
			return nil, tmpErr
		}
		reduceFn = tmpAny.(GenericReduceFn)
	case compute.OpTypeReduceLogicalAnd:
		reduceFn = execReduceLogicalAnd
	case compute.OpTypeReduceLogicalOr:
		reduceFn = execReduceLogicalOr
	case compute.OpTypeReduceLogicalXor:
		reduceFn = execReduceLogicalXor
	default:
		return nil, errors.Errorf("unsupported reduce op %s", node.OpType)
	}
	err = reduceFn(operand, output, it, dtype)
	if err != nil {
		return nil, err
	}
	return output, nil
}

// ReduceOutputIterator is a dynamic iterator that yields output flat indices
// for the corresponding flat index on the input operand, assuming the operand flat index is moving
// incrementally.
type ReduceOutputIterator struct {
	Config ReduceConfig

	flatIdx int // On the output tensor.

	perAxisIdx    []int // On the operand tensor.
	dimensions    []int // Of the operand tensor.
	perAxisStride []int // It is set to 0 for the axes being reduced.
}

// NewReduceOutputIterator creates a dynamic iterator that yields output flat indices
// for the corresponding flat index on the input operand, assuming the operand flat index is moving
// incrementally.
func NewReduceOutputIterator(dimensions []int, reduceAxes []int) *ReduceOutputIterator {
	inputRank := len(dimensions)
	it := &ReduceOutputIterator{
		perAxisIdx: make([]int, inputRank),
		dimensions: slices.Clone(dimensions),
	}
	it.perAxisStride = slices.Clone(dimensions)
	stride := 1
	for _, reduceAxis := range reduceAxes {
		it.perAxisStride[reduceAxis] = 0
	}
	for axis := inputRank - 1; axis >= 0; axis-- {
		if it.perAxisStride[axis] == 0 {
			// Skip the reducing axes and leave stride as 0.
			continue
		}

		// Accumulate (product) axes that are not reduced on the stride.
		newStride := stride * it.perAxisStride[axis]
		it.perAxisStride[axis] = stride
		stride = newStride
	}
	return it
}

// Next returns the next flat index in the output.
func (it *ReduceOutputIterator) Next() int {
	returnIdx := it.flatIdx
	// Move pointer.
	for axis := len(it.perAxisIdx) - 1; axis >= 0; axis-- {
		it.perAxisIdx[axis]++
		it.flatIdx += it.perAxisStride[axis]
		if it.perAxisIdx[axis] < it.dimensions[axis] {
			break
		}

		// Return to the start of the current axis and move to the next axis.
		it.perAxisIdx[axis] = 0
		it.flatIdx -= it.perAxisStride[axis] * it.dimensions[axis]
	}
	return returnIdx
}

var (
	//gobackend:dtypemap execReduceMaxGeneric ints,uints,floats
	reduceMaxDTypeMap = gobackend.NewDTypeMap("ReduceMax")

	//gobackend:dtypemap execReduceMinGeneric ints,uints,floats
	reduceMinDTypeMap = gobackend.NewDTypeMap("ReduceMin")

	//gobackend:dtypemap execReduceSumGeneric ints,uints,floats
	reduceSumDTypeMap = gobackend.NewDTypeMap("ReduceSum")

	//gobackend:dtypemap execReduceProductGeneric ints,uints,floats
	reduceProductDTypeMap = gobackend.NewDTypeMap("ReduceProduct")

	//gobackend:dtypemap execReduceBitwiseAndGeneric ints,uints
	reduceBitwiseAndDTypeMap = gobackend.NewDTypeMap("ReduceBitwiseAnd")

	//gobackend:dtypemap execReduceBitwiseOrGeneric ints,uints
	reduceBitwiseOrDTypeMap = gobackend.NewDTypeMap("ReduceBitwiseOr")

	//gobackend:dtypemap execReduceBitwiseXorGeneric ints,uints
	reduceBitwiseXorDTypeMap = gobackend.NewDTypeMap("ReduceBitwiseXor")
)

func init() {
	// DTypeMap registrations for Half-precision floating point types:

	// ReduceMax
	reduceMaxDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, execReduceMaxBFloat16)
	reduceMaxDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, execReduceMaxFloat16)

	// ReduceMin
	reduceMinDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, execReduceMinBFloat16)
	reduceMinDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, execReduceMinFloat16)

	// ReduceSum
	reduceSumDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, execReduceSumBFloat16)
	reduceSumDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, execReduceSumFloat16)

	// ReduceProduct
	reduceProductDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityTyped, execReduceProductBFloat16)
	reduceProductDTypeMap.Register(dtypes.Float16, gobackend.PriorityTyped, execReduceProductFloat16)
}

// GenericReduceFn is the type for the reduction functions.
type GenericReduceFn = func(operand, output *gobackend.Buffer, it *ReduceOutputIterator, dtype dtypes.DType) error

func execReduceMaxGeneric[T gobackend.PODNumericConstraints](operand, output *gobackend.Buffer, it *ReduceOutputIterator, dtype dtypes.DType) error {
	outputFlat := output.Flat.([]T)
	operandFlat := operand.Flat.([]T)

	switch it.Config.Pattern {
	case ReduceAll:
		if len(operandFlat) == 0 {
			outputFlat[0] = dtype.LowestValue().(T)
			return nil
		}
		m := operandFlat[0]
		for _, v := range operandFlat[1:] {
			m = max(m, v)
		}
		outputFlat[0] = m
		return nil

	case ReduceTrailing:
		A, B := it.Config.A, it.Config.B
		if B == 0 {
			lowest := dtype.LowestValue().(T)
			for a := range A {
				outputFlat[a] = lowest
			}
			return nil
		}
		for a := range A {
			offset := a * B
			row := operandFlat[offset : offset+B]
			m := row[0]
			for _, v := range row[1:] {
				m = max(m, v)
			}
			outputFlat[a] = m
		}
		return nil

	case ReduceLeading:
		A, B := it.Config.A, it.Config.B
		if A == 0 || B == 0 {
			lowest := dtype.LowestValue().(T)
			for b := range B {
				outputFlat[b] = lowest
			}
			return nil
		}
		copy(outputFlat, operandFlat[:B])
		for a := 1; a < A; a++ {
			offset := a * B
			sliceA := operandFlat[offset : offset+B]
			for b, v := range sliceA {
				outputFlat[b] = max(outputFlat[b], v)
			}
		}
		return nil

	case ReduceMiddle:
		A, B, C := it.Config.A, it.Config.B, it.Config.C
		if A == 0 || B == 0 || C == 0 {
			lowest := dtype.LowestValue().(T)
			for i := range outputFlat {
				outputFlat[i] = lowest
			}
			return nil
		}
		for a := range A {
			outOffset := a * C
			outSlice := outputFlat[outOffset : outOffset+C]
			inOffset0 := a * B * C
			copy(outSlice, operandFlat[inOffset0 : inOffset0+C])
			for b := 1; b < B; b++ {
				inOffset := inOffset0 + b*C
				inSlice := operandFlat[inOffset : inOffset+C]
				for c, v := range inSlice {
					outSlice[c] = max(outSlice[c], v)
				}
			}
		}
		return nil

	default:
		initialValue := dtype.LowestValue().(T)
		for outputIdx := range outputFlat {
			outputFlat[outputIdx] = initialValue
		}
		for _, value := range operandFlat {
			outputIdx := it.Next()
			outputFlat[outputIdx] = max(outputFlat[outputIdx], value)
		}
		return nil
	}
}

func execReduceMaxBFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, dtype dtypes.DType) error {
	initialValue := dtype.LowestValue().(bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = bfloat16.FromFloat32(max(a, b))
	}
	return nil
}

func execReduceMaxFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, dtype dtypes.DType) error {
	initialValue := dtype.LowestValue().(float16.Float16)
	outputFlat := output.Flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}
	operandFlat := operand.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.FromFloat32(max(a, b))
	}
	return nil
}

func execReduceMinGeneric[T gobackend.PODNumericConstraints](operand, output *gobackend.Buffer, it *ReduceOutputIterator, dtype dtypes.DType) error {
	outputFlat := output.Flat.([]T)
	operandFlat := operand.Flat.([]T)

	switch it.Config.Pattern {
	case ReduceAll:
		if len(operandFlat) == 0 {
			outputFlat[0] = dtype.HighestValue().(T)
			return nil
		}
		m := operandFlat[0]
		for _, v := range operandFlat[1:] {
			m = min(m, v)
		}
		outputFlat[0] = m
		return nil

	case ReduceTrailing:
		A, B := it.Config.A, it.Config.B
		if B == 0 {
			highest := dtype.HighestValue().(T)
			for a := range A {
				outputFlat[a] = highest
			}
			return nil
		}
		for a := range A {
			offset := a * B
			row := operandFlat[offset : offset+B]
			m := row[0]
			for _, v := range row[1:] {
				m = min(m, v)
			}
			outputFlat[a] = m
		}
		return nil

	case ReduceLeading:
		A, B := it.Config.A, it.Config.B
		if A == 0 || B == 0 {
			highest := dtype.HighestValue().(T)
			for b := range B {
				outputFlat[b] = highest
			}
			return nil
		}
		copy(outputFlat, operandFlat[:B])
		for a := 1; a < A; a++ {
			offset := a * B
			sliceA := operandFlat[offset : offset+B]
			for b, v := range sliceA {
				outputFlat[b] = min(outputFlat[b], v)
			}
		}
		return nil

	case ReduceMiddle:
		A, B, C := it.Config.A, it.Config.B, it.Config.C
		if A == 0 || B == 0 || C == 0 {
			highest := dtype.HighestValue().(T)
			for i := range outputFlat {
				outputFlat[i] = highest
			}
			return nil
		}
		for a := range A {
			outOffset := a * C
			outSlice := outputFlat[outOffset : outOffset+C]
			inOffset0 := a * B * C
			copy(outSlice, operandFlat[inOffset0 : inOffset0+C])
			for b := 1; b < B; b++ {
				inOffset := inOffset0 + b*C
				inSlice := operandFlat[inOffset : inOffset+C]
				for c, v := range inSlice {
					outSlice[c] = min(outSlice[c], v)
				}
			}
		}
		return nil

	default:
		initialValue := dtype.HighestValue().(T)
		for outputIdx := range outputFlat {
			outputFlat[outputIdx] = initialValue
		}
		for _, value := range operandFlat {
			outputIdx := it.Next()
			outputFlat[outputIdx] = min(outputFlat[outputIdx], value)
		}
		return nil
	}
}

func execReduceMinBFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, dtype dtypes.DType) error {
	initialValue := dtype.HighestValue().(bfloat16.BFloat16)
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = bfloat16.FromFloat32(min(a, b))
	}
	return nil
}

func execReduceMinFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, dtype dtypes.DType) error {
	initialValue := dtype.HighestValue().(float16.Float16)
	outputFlat := output.Flat.([]float16.Float16)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}
	operandFlat := operand.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		a, b := outputFlat[outputIdx].Float32(), value.Float32()
		outputFlat[outputIdx] = float16.FromFloat32(min(a, b))
	}
	return nil
}

func execReduceSumGeneric[T gobackend.PODNumericConstraints](operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]T)
	operandFlat := operand.Flat.([]T)

	switch it.Config.Pattern {
	case ReduceAll:
		var sum T
		for _, v := range operandFlat {
			sum += v
		}
		outputFlat[0] = sum
		return nil

	case ReduceTrailing:
		A, B := it.Config.A, it.Config.B
		if B == 0 {
			clear(outputFlat)
			return nil
		}
		for a := range A {
			offset := a * B
			row := operandFlat[offset : offset+B]
			var sum T
			for _, v := range row {
				sum += v
			}
			outputFlat[a] = sum
		}
		return nil

	case ReduceLeading:
		A, B := it.Config.A, it.Config.B
		if A == 0 || B == 0 {
			clear(outputFlat)
			return nil
		}
		copy(outputFlat, operandFlat[:B])
		for a := 1; a < A; a++ {
			offset := a * B
			sliceA := operandFlat[offset : offset+B]
			for b, v := range sliceA {
				outputFlat[b] += v
			}
		}
		return nil

	case ReduceMiddle:
		A, B, C := it.Config.A, it.Config.B, it.Config.C
		if A == 0 || B == 0 || C == 0 {
			clear(outputFlat)
			return nil
		}
		for a := range A {
			outOffset := a * C
			outSlice := outputFlat[outOffset : outOffset+C]
			inOffset0 := a * B * C
			copy(outSlice, operandFlat[inOffset0 : inOffset0+C])
			for b := 1; b < B; b++ {
				inOffset := inOffset0 + b*C
				inSlice := operandFlat[inOffset : inOffset+C]
				for c, v := range inSlice {
					outSlice[c] += v
				}
			}
		}
		return nil

	default:
		clear(outputFlat)
		for _, value := range operandFlat {
			outputIdx := it.Next()
			outputFlat[outputIdx] += value
		}
		return nil
	}
}

func execReduceSumBFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	operandFlat := operand.Flat.([]bfloat16.BFloat16)

	switch it.Config.Pattern {
	case ReduceAll:
		var sum float32
		for _, v := range operandFlat {
			sum += v.Float32()
		}
		outputFlat[0] = bfloat16.FromFloat32(sum)
		return nil

	case ReduceTrailing:
		A, B := it.Config.A, it.Config.B
		if B == 0 {
			clear(outputFlat)
			return nil
		}
		for a := range A {
			offset := a * B
			row := operandFlat[offset : offset+B]
			var sum float32
			for _, v := range row {
				sum += v.Float32()
			}
			outputFlat[a] = bfloat16.FromFloat32(sum)
		}
		return nil

	case ReduceLeading:
		A, B := it.Config.A, it.Config.B
		if A == 0 || B == 0 {
			clear(outputFlat)
			return nil
		}
		backend := output.RawBackend
		tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, B))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tempBuf)
		accumulator := tempBuf.Flat.([]float32)
		for b := range B {
			accumulator[b] = operandFlat[b].Float32()
		}
		for a := 1; a < A; a++ {
			offset := a * B
			sliceA := operandFlat[offset : offset+B]
			for b, v := range sliceA {
				accumulator[b] += v.Float32()
			}
		}
		for b, val := range accumulator {
			outputFlat[b] = bfloat16.FromFloat32(val)
		}
		return nil

	case ReduceMiddle:
		A, B, C := it.Config.A, it.Config.B, it.Config.C
		if A == 0 || B == 0 || C == 0 {
			clear(outputFlat)
			return nil
		}
		backend := output.RawBackend
		tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, C))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tempBuf)
		accumulator := tempBuf.Flat.([]float32)
		for a := range A {
			outOffset := a * C
			inOffset0 := a * B * C
			for c := range C {
				accumulator[c] = operandFlat[inOffset0+c].Float32()
			}
			for b := 1; b < B; b++ {
				inOffset := inOffset0 + b*C
				sliceB := operandFlat[inOffset : inOffset+C]
				for c, v := range sliceB {
					accumulator[c] += v.Float32()
				}
			}
			for c, val := range accumulator {
				outputFlat[outOffset+c] = bfloat16.FromFloat32(val)
			}
		}
		return nil

	default:
		backend := output.RawBackend
		tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, len(outputFlat)))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tempBuf)
		accumulator := tempBuf.Flat.([]float32)
		clear(accumulator)
		for _, value := range operandFlat {
			outputIdx := it.Next()
			accumulator[outputIdx] += value.Float32()
		}
		for i, val := range accumulator {
			outputFlat[i] = bfloat16.FromFloat32(val)
		}
		return nil
	}
}

func execReduceSumFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]float16.Float16)
	operandFlat := operand.Flat.([]float16.Float16)

	switch it.Config.Pattern {
	case ReduceAll:
		var sum float32
		for _, v := range operandFlat {
			sum += v.Float32()
		}
		outputFlat[0] = float16.FromFloat32(sum)
		return nil

	case ReduceTrailing:
		A, B := it.Config.A, it.Config.B
		if B == 0 {
			clear(outputFlat)
			return nil
		}
		for a := range A {
			offset := a * B
			row := operandFlat[offset : offset+B]
			var sum float32
			for _, v := range row {
				sum += v.Float32()
			}
			outputFlat[a] = float16.FromFloat32(sum)
		}
		return nil

	case ReduceLeading:
		A, B := it.Config.A, it.Config.B
		if A == 0 || B == 0 {
			clear(outputFlat)
			return nil
		}
		backend := output.RawBackend
		tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, B))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tempBuf)
		accumulator := tempBuf.Flat.([]float32)
		for b := range B {
			accumulator[b] = operandFlat[b].Float32()
		}
		for a := 1; a < A; a++ {
			offset := a * B
			sliceA := operandFlat[offset : offset+B]
			for b, v := range sliceA {
				accumulator[b] += v.Float32()
			}
		}
		for b, val := range accumulator {
			outputFlat[b] = float16.FromFloat32(val)
		}
		return nil

	case ReduceMiddle:
		A, B, C := it.Config.A, it.Config.B, it.Config.C
		if A == 0 || B == 0 || C == 0 {
			clear(outputFlat)
			return nil
		}
		backend := output.RawBackend
		tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, C))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tempBuf)
		accumulator := tempBuf.Flat.([]float32)
		for a := range A {
			outOffset := a * C
			inOffset0 := a * B * C
			for c := range C {
				accumulator[c] = operandFlat[inOffset0+c].Float32()
			}
			for b := 1; b < B; b++ {
				inOffset := inOffset0 + b*C
				sliceB := operandFlat[inOffset : inOffset+C]
				for c, v := range sliceB {
					accumulator[c] += v.Float32()
				}
			}
			for c, val := range accumulator {
				outputFlat[outOffset+c] = float16.FromFloat32(val)
			}
		}
		return nil

	default:
		backend := output.RawBackend
		tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, len(outputFlat)))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tempBuf)
		accumulator := tempBuf.Flat.([]float32)
		clear(accumulator)
		for _, value := range operandFlat {
			outputIdx := it.Next()
			accumulator[outputIdx] += value.Float32()
		}
		for i, val := range accumulator {
			outputFlat[i] = float16.FromFloat32(val)
		}
		return nil
	}
}

func execReduceProductGeneric[T gobackend.PODNumericConstraints](operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = T(1)
	}
	operandFlat := operand.Flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		outputFlat[outputIdx] *= value
	}
	return nil
}

func execReduceProductBFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	backend := output.RawBackend
	tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, len(outputFlat)))
	if err != nil {
		return err
	}
	defer backend.PutBuffer(tempBuf)
	accumulator := tempBuf.Flat.([]float32)
	for i := range accumulator {
		accumulator[i] = 1.0
	}
	operandFlat := operand.Flat.([]bfloat16.BFloat16)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		accumulator[outputIdx] *= value.Float32()
	}
	for i, val := range accumulator {
		outputFlat[i] = bfloat16.FromFloat32(val)
	}
	return nil
}

func execReduceProductFloat16(operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]float16.Float16)
	backend := output.RawBackend
	tempBuf, err := backend.GetBuffer(shapes.Make(dtypes.Float32, len(outputFlat)))
	if err != nil {
		return err
	}
	defer backend.PutBuffer(tempBuf)
	accumulator := tempBuf.Flat.([]float32)
	for i := range accumulator {
		accumulator[i] = 1.0
	}
	operandFlat := operand.Flat.([]float16.Float16)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		accumulator[outputIdx] *= value.Float32()
	}
	for i, val := range accumulator {
		outputFlat[i] = float16.FromFloat32(val)
	}
	return nil
}

func execReduceBitwiseAndGeneric[T gobackend.PODIntegerConstraints](operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]T)
	initialValue := ^T(0)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = initialValue
	}
	operandFlat := operand.Flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		outputFlat[outputIdx] &= value
	}
	return nil
}

func execReduceBitwiseOrGeneric[T gobackend.PODIntegerConstraints](operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = T(0)
	}
	operandFlat := operand.Flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		outputFlat[outputIdx] |= value
	}
	return nil
}

func execReduceBitwiseXorGeneric[T gobackend.PODIntegerConstraints](operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]T)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = T(0)
	}
	operandFlat := operand.Flat.([]T)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		outputFlat[outputIdx] ^= value
	}
	return nil
}

func execReduceLogicalAnd(operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]bool)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = true
	}
	operandFlat := operand.Flat.([]bool)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		outputFlat[outputIdx] = outputFlat[outputIdx] && value
	}
	return nil
}

func execReduceLogicalOr(operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]bool)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = false
	}
	operandFlat := operand.Flat.([]bool)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		outputFlat[outputIdx] = outputFlat[outputIdx] || value
	}
	return nil
}

func execReduceLogicalXor(operand, output *gobackend.Buffer, it *ReduceOutputIterator, _ dtypes.DType) error {
	outputFlat := output.Flat.([]bool)
	for outputIdx := range outputFlat {
		outputFlat[outputIdx] = false
	}
	operandFlat := operand.Flat.([]bool)
	for _, value := range operandFlat {
		outputIdx := it.Next()
		outputFlat[outputIdx] = outputFlat[outputIdx] != value
	}
	return nil
}

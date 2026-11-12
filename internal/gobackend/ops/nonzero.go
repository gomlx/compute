// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterNonZero.Register(NonZero, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeNonZero, gobackend.PriorityGeneric, execNonZero)
}

func NonZero(f *gobackend.Function, operand compute.Value, cfg compute.NonZeroConfig) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("NonZero", operand)
	if err != nil {
		return nil, err
	}

	knownAxisNames := f.KnownDynamicAxisNames()
	if cfg.CountAxisName == "" {
		for {
			cfg.CountAxisName = f.RawBuilder.Backend.UniqueAxisName("#nonzeros")
			if !knownAxisNames.Has(cfg.CountAxisName) {
				break
			}
		}
	} else if knownAxisNames.Has(cfg.CountAxisName) {
		return nil, errors.Errorf("NonZero: count axis name %q is already in use", cfg.CountAxisName)
	}
	knownAxisNames.Insert(cfg.CountAxisName)

	x := inputs[0]
	shape, err := shapeinference.NonZero(x.Shape, cfg)
	if err != nil {
		return nil, errors.Wrap(err, "NonZero shape inference")
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeNonZero, shape, inputs, cfg)
	return node, nil
}

func execNonZero(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	x := inputs[0]
	var positions []int
	switch values := x.Flat.(type) {
	case []float32:
		positions = nonZeroPositions(values)
	case []float64:
		positions = nonZeroPositions(values)
	case []float16.Float16:
		positions = make([]int, 0)
		for i, value := range values {
			if value.Float32() != 0 {
				positions = append(positions, i)
			}
		}
	case []bfloat16.BFloat16:
		positions = make([]int, 0)
		for i, value := range values {
			if value.Float32() != 0 {
				positions = append(positions, i)
			}
		}
	case []int8:
		positions = nonZeroPositions(values)
	case []int16:
		positions = nonZeroPositions(values)
	case []int32:
		positions = nonZeroPositions(values)
	case []int64:
		positions = nonZeroPositions(values)
	case []uint8:
		positions = nonZeroPositions(values)
	case []uint16:
		positions = nonZeroPositions(values)
	case []uint32:
		positions = nonZeroPositions(values)
	case []uint64:
		positions = nonZeroPositions(values)
	case []bool:
		positions = make([]int, 0)
		for i, value := range values {
			if value {
				positions = append(positions, i)
			}
		}
	case []complex64:
		positions = nonZeroPositions(values)
	case []complex128:
		positions = nonZeroPositions(values)
	default:
		return nil, errors.Errorf("NonZero does not support buffer dtype %s", x.RawShape.DType)
	}
	cfg := node.Data.(compute.NonZeroConfig)
	rank, count := x.RawShape.Rank(), len(positions)
	dimensions := []int{count, rank}
	if cfg.Transpose {
		dimensions = []int{rank, count}
	}
	out, err := backend.GetBuffer(shapes.Make(dtypes.Int64, dimensions...))
	if err != nil {
		return nil, err
	}
	data := out.Flat.([]int64)
	for column, position := range positions {
		for axis := rank - 1; axis >= 0; axis-- {
			dim := x.RawShape.Dimensions[axis]
			coordinate := int64(position % dim)
			if cfg.Transpose {
				data[axis*count+column] = coordinate
			} else {
				data[column*rank+axis] = coordinate
			}
			position /= dim
		}
	}
	return out, nil
}

func nonZeroPositions[T comparable](values []T) []int {
	var positions []int
	var zero T
	for i, value := range values {
		if value != zero {
			positions = append(positions, i)
		}
	}
	return positions
}

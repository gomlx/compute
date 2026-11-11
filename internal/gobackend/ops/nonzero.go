// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"reflect"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterNonZero.Register(NonZero, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeNonZero, gobackend.PriorityGeneric, execNonZero)
}

func NonZero(f *gobackend.Function, operand compute.Value, countAxisName string) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("NonZero", operand)
	if err != nil {
		return nil, err
	}
	if countAxisName == "" {
		return nil, errors.New("NonZero: countAxisName must be nonempty")
	}
	x := inputs[0]
	shape := shapes.MakeDynamic(dtypes.Int64, []int{x.Shape.Rank(), shapes.DynamicDim}, []string{"", countAxisName})
	node, _ := f.GetOrCreateNode(compute.OpTypeNonZero, shape, inputs, countAxisName)
	return node, nil
}

func execNonZero(backend *gobackend.Backend, _ *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	x := inputs[0]
	flat := reflect.ValueOf(x.Flat)
	isNonzero := func(i int) bool { return !flat.Index(i).IsZero() }
	switch values := x.Flat.(type) {
	case []float16.Float16:
		isNonzero = func(i int) bool { return values[i].Float32() != 0 }
	case []bfloat16.BFloat16:
		isNonzero = func(i int) bool { return values[i].Float32() != 0 }
	}
	var positions []int
	for i := range flat.Len() {
		if isNonzero(i) {
			positions = append(positions, i)
		}
	}
	rank, count := x.RawShape.Rank(), len(positions)
	out, err := backend.GetBuffer(shapes.Make(dtypes.Int64, rank, count))
	if err != nil {
		return nil, err
	}
	data := out.Flat.([]int64)
	for column, position := range positions {
		for axis := rank - 1; axis >= 0; axis-- {
			dim := x.RawShape.Dimensions[axis]
			data[axis*count+column] = int64(position % dim)
			position /= dim
		}
	}
	return out, nil
}

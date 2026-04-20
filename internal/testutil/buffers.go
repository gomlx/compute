// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil

import (
	"reflect"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

// ToFlatAndShape converts a nested slice of any into a flat slice and a shape that can be fed into testBackend (or a BufferFromFlatData).
func ToFlatAndShape(v any) (any, shapes.Shape) {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Slice {
		dtype := dtypes.FromGoType(rv.Type())
		slice := reflect.MakeSlice(reflect.SliceOf(rv.Type()), 1, 1)
		slice.Index(0).Set(rv)
		return slice.Interface(), shapes.Make(dtype)
	}

	var dims []int
	curr := rv
	for curr.Kind() == reflect.Slice {
		dims = append(dims, curr.Len())
		if curr.Len() > 0 {
			curr = curr.Index(0)
		} else {
			break
		}
	}
	dtype := dtypes.FromGoType(curr.Type())

	flat := reflect.MakeSlice(reflect.SliceOf(curr.Type()), 0, 0)
	var flatten func(reflect.Value)
	flatten = func(val reflect.Value) {
		if val.Kind() == reflect.Slice {
			for i := 0; i < val.Len(); i++ {
				flatten(val.Index(i))
			}
		} else {
			flat = reflect.Append(flat, val)
		}
	}
	flatten(rv)
	return flat.Interface(), shapes.Make(dtype, dims...)
}

// FlattenSlice is a helper to convert a nested slice of any into a flat slice.
func FlattenSlice(v any) any {
	flat, _ := ToFlatAndShape(v)
	return flat
}

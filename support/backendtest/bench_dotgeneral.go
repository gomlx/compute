// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func BenchmarkDotGeneral(b *testing.B, backend compute.Backend) {
	type dotGeneralCase struct {
		name                                         string
		lhsShape, rhsShape                           shapes.Shape
		lhsContract, lhsBatch, rhsContract, rhsBatch []int
	}
	benchCases := []dotGeneralCase{
		{
			name:        "32x12x13x13_x_32x12x13x32",
			lhsShape:    shapes.Make(dtypes.Float32, 32, 12, 13, 13),
			rhsShape:    shapes.Make(dtypes.Float32, 32, 12, 13, 32),
			lhsContract: []int{3},
			lhsBatch:    []int{0, 1},
			rhsContract: []int{2},
			rhsBatch:    []int{0, 1},
		},
		{
			name:        "32x12x13x32_x_32x12x32x13",
			lhsShape:    shapes.Make(dtypes.Float32, 32, 12, 13, 32),
			rhsShape:    shapes.Make(dtypes.Float32, 32, 12, 32, 13),
			lhsContract: []int{3},
			lhsBatch:    []int{0, 1},
			rhsContract: []int{2},
			rhsBatch:    []int{0, 1},
		},
		{
			name:        "32x13x1536_x_1536x384",
			lhsShape:    shapes.Make(dtypes.Float32, 32, 13, 1536),
			rhsShape:    shapes.Make(dtypes.Float32, 1536, 384),
			lhsContract: []int{2},
			rhsContract: []int{0},
		},
		{
			name:        "32x13x384_x_384x1152",
			lhsShape:    shapes.Make(dtypes.Float32, 32, 13, 384),
			rhsShape:    shapes.Make(dtypes.Float32, 384, 1152),
			lhsContract: []int{2},
			rhsContract: []int{0},
		},
		{
			name:        "32x13x384_x_384x1536",
			lhsShape:    shapes.Make(dtypes.Float32, 32, 13, 384),
			rhsShape:    shapes.Make(dtypes.Float32, 384, 1536),
			lhsContract: []int{2},
			rhsContract: []int{0},
		},
		{
			name:        "32x13x384_x_384x384",
			lhsShape:    shapes.Make(dtypes.Float32, 32, 13, 384),
			rhsShape:    shapes.Make(dtypes.Float32, 384, 384),
			lhsContract: []int{2},
			rhsContract: []int{0},
		},
		{
			name:        "416x384_x_384x1152",
			lhsShape:    shapes.Make(dtypes.Float32, 416, 384),
			rhsShape:    shapes.Make(dtypes.Float32, 384, 1152),
			lhsContract: []int{1},
			rhsContract: []int{0},
		},
	}

	for _, tc := range benchCases {
		lhsData := randomFloat32(tc.lhsShape.Size())
		rhsData := randomFloat32(tc.rhsShape.Size())

		be, err := newBenchExec(backend, []shapes.Shape{tc.lhsShape, tc.rhsShape}, []any{lhsData, rhsData},
			func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.DotGeneral(params[0], tc.lhsContract, tc.lhsBatch, params[1], tc.rhsContract, tc.rhsBatch, compute.DotGeneralConfig{})
			})
		if err != nil {
			b.Fatalf("Failed to create benchmark %s: %+v", tc.name, err)
		}

		b.Run(tc.name, func(b *testing.B) {
			be.run(b)
		})
	}
}

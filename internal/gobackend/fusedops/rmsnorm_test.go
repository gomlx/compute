// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	_ "github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
)

func buildDecomposedRMSNorm(
	t *testing.T,
	f compute.Function,
	x compute.Value,
	dtype dtypes.DType,
	normSize int,
	epsilon float64,
	scale []bfloat16.BFloat16,
	scaleOffset float64,
	dynamic bool,
) compute.Value {
	t.Helper()
	var err error
	xNorm := x
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		xNorm, err = f.ConvertDType(x, dtypes.Float32)
		if err != nil {
			t.Fatal(err)
		}
	}
	x2, err := f.Mul(xNorm, xNorm)
	if err != nil {
		t.Fatal(err)
	}
	xNormShape := xNorm.(*gobackend.Node).Shape
	rank := xNormShape.Rank()
	sum, err := f.ReduceSum(x2, rank-1)
	if err != nil {
		t.Fatal(err)
	}
	var mean compute.Value
	if dynamic {
		var denom compute.Value
		if normSize == 1536 {
			// Match gomlx's ReduceMean on dynamic tensors whose trailing axis is static:
			// ConvertDType(Constant(int64(normSize)), Float32).
			cN, _ := f.Constant([]int64{int64(normSize)})
			denom, err = f.ConvertDType(cN, dtypes.Float32)
		} else {
			dimSz, dErr := f.DynamicDimensionSize(x2, rank-1)
			if dErr != nil {
				t.Fatal(dErr)
			}
			denom, err = f.ConvertDType(dimSz, dtypes.Float32)
		}
		if err != nil {
			t.Fatal(err)
		}
		mean, err = f.Div(sum, denom)
		if err != nil {
			t.Fatal(err)
		}
		specs := make([]compute.DynamicDimensionSpec, rank)
		for ax := range rank - 1 {
			if xNormShape.Dimensions[ax] == shapes.DynamicDim {
				ds, _ := f.DynamicDimensionSize(xNorm, ax)
				specs[ax] = compute.DynamicDimensionSpec{Name: xNormShape.AxisNames[ax], Value: ds}
			} else {
				specs[ax] = compute.DynamicDimensionSpec{Static: xNormShape.Dimensions[ax]}
			}
		}
		specs[rank-1] = compute.DynamicDimensionSpec{Static: 1}
		mean, err = f.DynamicReshape(mean, specs...)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		invN, _ := f.Constant([]float32{float32(1.0 / float64(normSize))})
		mean, err = f.Mul(sum, invN)
		if err != nil {
			t.Fatal(err)
		}
		dims := append(append([]int(nil), sum.(*gobackend.Node).Shape.Dimensions...), 1)
		mean, err = f.Reshape(mean, dims...)
		if err != nil {
			t.Fatal(err)
		}
	}
	if epsilon != 0 {
		epsNode, _ := f.Constant([]float32{float32(epsilon)})
		mean, err = f.Add(mean, epsNode)
		if err != nil {
			t.Fatal(err)
		}
	}
	rms, err := f.Sqrt(mean)
	if err != nil {
		t.Fatal(err)
	}
	normed, err := f.Div(xNorm, rms)
	if err != nil {
		t.Fatal(err)
	}
	res := normed
	if scale != nil {
		sNode, err := f.Constant(scale, normSize)
		if err != nil {
			t.Fatal(err)
		}
		sF32, err := f.ConvertDType(sNode, dtypes.Float32)
		if err != nil {
			t.Fatal(err)
		}
		if scaleOffset != 0 {
			offNode, _ := f.Constant([]float32{float32(scaleOffset)})
			sF32, err = f.Add(sF32, offNode)
			if err != nil {
				t.Fatal(err)
			}
		}
		reshapeDims := make([]int, rank)
		for i := range rank - 1 {
			reshapeDims[i] = 1
		}
		reshapeDims[rank-1] = normSize
		sShaped, err := f.Reshape(sF32, reshapeDims...)
		if err != nil {
			t.Fatal(err)
		}
		res, err = f.Mul(res, sShaped)
		if err != nil {
			t.Fatal(err)
		}
	}
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		res, err = f.ConvertDType(res, dtype)
		if err != nil {
			t.Fatal(err)
		}
	}
	return res
}

func TestRMSNormFusion(t *testing.T) {
	b, err := gobackend.NewBackend()
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	defer b.Finalize()

	for _, tc := range []struct {
		name        string
		outer       int
		normSize    int
		useScale    bool
		scaleOffset float64
		dynamic     bool
	}{
		{"Static_BF16_Scale_1536", 1, 1536, true, 0.0, false},
		{"Static_BF16_ScaleOffset1_1536", 1, 1536, true, 1.0, false},
		{"Static_BF16_NoScale_256", 8, 256, false, 0.0, false},
		{"Dynamic_BF16_Prefill_356x1536", 356, 1536, true, 0.0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const eps = 1e-6
			var scale []bfloat16.BFloat16
			if tc.useScale {
				scale = make([]bfloat16.BFloat16, tc.normSize)
				for i := range scale {
					scale[i] = bfloat16.FromFloat32(float32((i%13)-6) * 0.125)
				}
			}

			builder := b.Builder("rmsnorm_" + tc.name)
			f := builder.Main()
			var inShape shapes.Shape
			if tc.dynamic {
				inShape = shapes.MakeDynamic(dtypes.BFloat16, []int{1, shapes.DynamicDim, tc.normSize}, []string{"", "seq_len", ""})
			} else {
				inShape = shapes.Make(dtypes.BFloat16, 1, tc.outer, tc.normSize)
			}
			p, err := f.Parameter("x", inShape, nil)
			if err != nil {
				t.Fatal(err)
			}
			out := buildDecomposedRMSNorm(t, f, p, dtypes.BFloat16, tc.normSize, eps, scale, tc.scaleOffset, tc.dynamic)
			if err := f.Return([]compute.Value{out}, nil); err != nil {
				t.Fatal(err)
			}
			exec, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer exec.Finalize()

			// Verify that the RMSNormFusion pass actually fused the output node into OpTypeFusedLayerNorm.
			outNode := out.(*gobackend.Node)
			if outNode.OpType != compute.OpTypeFusedLayerNorm {
				t.Fatalf("expected RMSNormFusion to rewrite output node to OpTypeFusedLayerNorm, got %s", outNode.OpType)
			}

			inData := make([]bfloat16.BFloat16, tc.outer*tc.normSize)
			for i := range inData {
				inData[i] = bfloat16.FromFloat32(float32(((i*7)%31)-15) * 0.25)
			}
			concreteShape := shapes.Make(dtypes.BFloat16, 1, tc.outer, tc.normSize)
			inBuf, err := b.BufferFromFlatData(0, inData, concreteShape)
			if err != nil {
				t.Fatal(err)
			}
			defer inBuf.Finalize()

			for run := range 3 {
				outs, err := exec.Execute([]compute.Buffer{inBuf}, []bool{false}, 0)
				if err != nil {
					t.Fatal(err)
				}
				got := make([]bfloat16.BFloat16, len(inData))
				if err := outs[0].ToFlatData(got); err != nil {
					t.Fatal(err)
				}
				outs[0].Finalize()

				// Compare against exact Float32 reference.
				for r := range tc.outer {
					off := r * tc.normSize
					var sumSq float64
					for j := range tc.normSize {
						v := float64(inData[off+j].Float32())
						sumSq += v * v
					}
					invRMS := 1.0 / math.Sqrt(sumSq/float64(tc.normSize)+eps)
					for j := range tc.normSize {
						want := float64(inData[off+j].Float32()) * invRMS
						if scale != nil {
							want *= float64(scale[j].Float32()) + tc.scaleOffset
						}
						wantBF := bfloat16.FromFloat32(float32(want)).Float32()
						gotVal := got[off+j].Float32()
						tol := math.Max(1e-2, 2e-2*math.Abs(float64(wantBF)))
						if math.Abs(float64(gotVal-wantBF)) > tol {
							t.Fatalf("run=%d row=%d col=%d: got %v, want %v", run, r, j, gotVal, wantBF)
						}
					}
				}
			}
		})
	}
}

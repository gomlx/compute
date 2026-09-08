// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build goexperiment.simd

package ops_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func runUnaryTestGeneric[T any](t *testing.T, opName string,
	opFn func(f *gobackend.Function, operand compute.Value) (compute.Value, error),
	shape shapes.Shape, inData []T, expected []T) {
	runUnaryTestApprox(t, opName, opFn, shape, inData, expected, 0)
}

func runUnaryTestApprox[T any](t *testing.T, opName string,
	opFn func(f *gobackend.Function, operand compute.Value) (compute.Value, error),
	shape shapes.Shape, inData []T, expected []T, delta float64) {
	t.Helper()
	builder := backend.Builder(opName).(*gobackend.Builder)
	main := builder.Main().(*gobackend.Function)

	inNode, err := main.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Failed creating parameter: %+v", err)
	}

	outNode, err := opFn(main, inNode)
	if err != nil {
		t.Fatalf("Failed adding op %s: %+v", opName, err)
	}

	err = main.Return([]compute.Value{outNode}, nil)
	if err != nil {
		t.Fatalf("Return failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %+v", err)
	}

	inBuf, err := backend.BufferFromFlatData(0, inData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}

	result := outputs[0].(*gobackend.Buffer).Flat.([]T)
	var ok bool
	var diff string
	if delta > 0 {
		ok, diff = testutil.IsInDelta(expected, result, delta)
	} else {
		ok, diff = testutil.IsEqual(expected, result)
	}
	if !ok {
		t.Errorf("Mismatch in %s:\n%s", opName, diff)
	}
}

func runBinaryTestGeneric[T any](t *testing.T, opName string,
	opFn func(f *gobackend.Function, lhs, rhs compute.Value) (compute.Value, error),
	lhsShape shapes.Shape, lhsData []T,
	rhsShape shapes.Shape, rhsData []T,
	expected []T) {
	t.Helper()
	builder := backend.Builder(opName).(*gobackend.Builder)
	main := builder.Main().(*gobackend.Function)

	lhsNode, err := main.Parameter("lhs", lhsShape, nil)
	if err != nil {
		t.Fatalf("Failed creating lhs parameter: %+v", err)
	}
	rhsNode, err := main.Parameter("rhs", rhsShape, nil)
	if err != nil {
		t.Fatalf("Failed creating rhs parameter: %+v", err)
	}

	outNode, err := opFn(main, lhsNode, rhsNode)
	if err != nil {
		t.Fatalf("Failed adding op %s: %+v", opName, err)
	}

	err = main.Return([]compute.Value{outNode}, nil)
	if err != nil {
		t.Fatalf("Return failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile failed: %+v", err)
	}

	lhsBuf, err := backend.BufferFromFlatData(0, lhsData, lhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData lhs failed: %+v", err)
	}
	rhsBuf, err := backend.BufferFromFlatData(0, rhsData, rhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData rhs failed: %+v", err)
	}

	outputs, err := exec.Execute([]compute.Buffer{lhsBuf, rhsBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute failed: %+v", err)
	}

	result := outputs[0].(*gobackend.Buffer).Flat.([]T)
	if ok, diff := testutil.IsEqual(expected, result); !ok {
		t.Errorf("Mismatch in %s:\n%s", opName, diff)
	}
}

func TestSIMDBinaryOps(t *testing.T) {
	// Test partial vector sizes (e.g. 19 elements, where SIMD vector len is 16 or 8).
	n := 19
	s19 := shapes.Make(dtypes.Float32, n)
	sScalar := shapes.Make(dtypes.Float32, 1)

	lhsF32 := make([]float32, n)
	rhsF32 := make([]float32, n)
	for i := range n {
		lhsF32[i] = float32(i*2 + 10)
		rhsF32[i] = float32(i + 1)
	}

	t.Run("Add_Float32", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = lhsF32[i] + rhsF32[i]
		}
		runBinaryTestGeneric(t, "Add_Float32", ops.Add, s19, lhsF32, s19, rhsF32, expected)
	})

	t.Run("Add_Float32_ScalarRHS", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = lhsF32[i] + 5.0
		}
		runBinaryTestGeneric(t, "Add_Float32_ScalarRHS", ops.Add, s19, lhsF32, sScalar, []float32{5.0}, expected)
	})

	t.Run("Sub_Float32", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = lhsF32[i] - rhsF32[i]
		}
		runBinaryTestGeneric(t, "Sub_Float32", ops.Sub, s19, lhsF32, s19, rhsF32, expected)
	})

	t.Run("Sub_Float32_ScalarLHS", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = 100.0 - rhsF32[i]
		}
		runBinaryTestGeneric(t, "Sub_Float32_ScalarLHS", ops.Sub, sScalar, []float32{100.0}, s19, rhsF32, expected)
	})

	t.Run("Mul_Float32", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = lhsF32[i] * rhsF32[i]
		}
		runBinaryTestGeneric(t, "Mul_Float32", ops.Mul, s19, lhsF32, s19, rhsF32, expected)
	})

	t.Run("Div_Float32", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = lhsF32[i] / rhsF32[i]
		}
		runBinaryTestGeneric(t, "Div_Float32", ops.Div, s19, lhsF32, s19, rhsF32, expected)
	})

	// Int32 Binary
	s19I32 := shapes.Make(dtypes.Int32, n)
	lhsI32 := make([]int32, n)
	rhsI32 := make([]int32, n)
	for i := range n {
		lhsI32[i] = int32(i*3 + 10)
		rhsI32[i] = int32(i + 2)
	}

	t.Run("Add_Int32", func(t *testing.T) {
		expected := make([]int32, n)
		for i := range n {
			expected[i] = lhsI32[i] + rhsI32[i]
		}
		runBinaryTestGeneric(t, "Add_Int32", ops.Add, s19I32, lhsI32, s19I32, rhsI32, expected)
	})

	t.Run("Mul_Int32", func(t *testing.T) {
		expected := make([]int32, n)
		for i := range n {
			expected[i] = lhsI32[i] * rhsI32[i]
		}
		runBinaryTestGeneric(t, "Mul_Int32", ops.Mul, s19I32, lhsI32, s19I32, rhsI32, expected)
	})

	// BFloat16 Binary
	s19BF := shapes.Make(dtypes.BFloat16, n)
	lhsBF := make([]bfloat16.BFloat16, n)
	rhsBF := make([]bfloat16.BFloat16, n)
	expectedBF := make([]bfloat16.BFloat16, n)
	for i := range n {
		lhsBF[i] = bfloat16.FromFloat32(float32(i*2 + 5))
		rhsBF[i] = bfloat16.FromFloat32(float32(i + 1))
		expectedBF[i] = bfloat16.FromFloat32(lhsBF[i].Float32() + rhsBF[i].Float32())
	}
	t.Run("Add_BFloat16", func(t *testing.T) {
		runBinaryTestGeneric(t, "Add_BFloat16", ops.Add, s19BF, lhsBF, s19BF, rhsBF, expectedBF)
	})

	// Float16 Binary
	s19F16 := shapes.Make(dtypes.Float16, n)
	lhsF16 := make([]float16.Float16, n)
	rhsF16 := make([]float16.Float16, n)
	expectedF16 := make([]float16.Float16, n)
	for i := range n {
		lhsF16[i] = float16.FromFloat32(float32(i*2 + 5))
		rhsF16[i] = float16.FromFloat32(float32(i + 1))
		expectedF16[i] = float16.FromFloat32(lhsF16[i].Float32() * rhsF16[i].Float32())
	}
	t.Run("Mul_Float16", func(t *testing.T) {
		runBinaryTestGeneric(t, "Mul_Float16", ops.Mul, s19F16, lhsF16, s19F16, rhsF16, expectedF16)
	})
}

func TestSIMDUnaryOps(t *testing.T) {
	n := 19
	s19 := shapes.Make(dtypes.Float32, n)
	inF32 := make([]float32, n)
	for i := range n {
		inF32[i] = float32(i)*0.2 - 1.8
	}

	t.Run("Abs_Float32", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = float32(math.Abs(float64(inF32[i])))
		}
		runUnaryTestGeneric(t, "Abs_Float32", ops.Abs, s19, inF32, expected)
	})

	t.Run("Neg_Float32", func(t *testing.T) {
		expected := make([]float32, n)
		for i := range n {
			expected[i] = -inF32[i]
		}
		runUnaryTestGeneric(t, "Neg_Float32", ops.Neg, s19, inF32, expected)
	})

	t.Run("Sqrt_Float32", func(t *testing.T) {
		posIn := make([]float32, n)
		expected := make([]float32, n)
		for i := range n {
			posIn[i] = float32(i + 1)
			expected[i] = float32(math.Sqrt(float64(posIn[i])))
		}
		runUnaryTestGeneric(t, "Sqrt_Float32", ops.Sqrt, s19, posIn, expected)
	})

	t.Run("Exp_Float32", func(t *testing.T) {
		in := make([]float32, n)
		expected := make([]float32, n)
		for i := range n {
			in[i] = float32(i)*0.1 - 1.0
			expected[i] = float32(math.Exp(float64(in[i])))
		}
		runUnaryTestApprox(t, "Exp_Float32", ops.Exp, s19, in, expected, 1e-4)
	})

	t.Run("Logistic_Float32", func(t *testing.T) {
		in := make([]float32, n)
		expected := make([]float32, n)
		for i := range n {
			in[i] = float32(i)*0.2 - 2.0
			expected[i] = float32(1.0 / (1.0 + math.Exp(-float64(in[i]))))
		}
		runUnaryTestApprox(t, "Logistic_Float32", ops.Logistic, s19, in, expected, 1e-4)
	})

	t.Run("Tanh_Float32", func(t *testing.T) {
		in := make([]float32, n)
		expected := make([]float32, n)
		for i := range n {
			in[i] = float32(i)*0.3 - 2.5
			expected[i] = float32(math.Tanh(float64(in[i])))
		}
		runUnaryTestApprox(t, "Tanh_Float32", ops.Tanh, s19, in, expected, 1e-4)
	})

	t.Run("Erf_Float32", func(t *testing.T) {
		in := make([]float32, n)
		expected := make([]float32, n)
		for i := range n {
			in[i] = float32(i)*0.2 - 2.0
			expected[i] = float32(math.Erf(float64(in[i])))
		}
		runUnaryTestApprox(t, "Erf_Float32", ops.Erf, s19, in, expected, 1e-4)
	})

	// Int32 Unary
	s19I32 := shapes.Make(dtypes.Int32, n)
	inI32 := make([]int32, n)
	for i := range n {
		inI32[i] = int32(i - 10)
	}

	t.Run("Abs_Int32", func(t *testing.T) {
		expected := make([]int32, n)
		for i := range n {
			val := inI32[i]
			if val < 0 {
				val = -val
			}
			expected[i] = val
		}
		runUnaryTestGeneric(t, "Abs_Int32", ops.Abs, s19I32, inI32, expected)
	})

	t.Run("Neg_Int32", func(t *testing.T) {
		expected := make([]int32, n)
		for i := range n {
			expected[i] = -inI32[i]
		}
		runUnaryTestGeneric(t, "Neg_Int32", ops.Neg, s19I32, inI32, expected)
	})
}

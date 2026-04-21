// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func testExecUnaryOp(t *testing.T, buildFn func(f compute.Function, param compute.Value) (compute.Value, error), input any) *Buffer {
	inputFlat, inputShape := testutil.ToFlatAndShape(input)
	return testBackendMultiInput(t, []shapes.Shape{inputShape}, []any{inputFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return buildFn(f, params[0])
	})
}

func testExecUnaryOpFails(t *testing.T, buildFn func(f compute.Function, param compute.Value) (compute.Value, error), input any) {
	inputFlat, inputShape := testutil.ToFlatAndShape(input)
	_, _, err := buildGraph([]shapes.Shape{inputShape}, []any{inputFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return buildFn(f, params[0])
	})
	if err == nil {
		t.Errorf("Expected error but got nil")
	}
}

func TestExecUnary(t *testing.T) {
	t.Run("Neg", func(t *testing.T) {
		buildFnNeg := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Neg(param) }
		y0 := testExecUnaryOp(t, buildFnNeg, float32(7))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(float32(-7)), y0.flat); !ok {
			t.Errorf("Neg float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnNeg, []int32{-1, 2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{1, -2}), y1.flat); !ok {
			t.Errorf("Neg int32 mismatch:\n%s", diff)
		}
		testExecUnaryOpFails(t, buildFnNeg, []uint32{1, 2, 3})
	})
	t.Run("Abs", func(t *testing.T) {
		buildFnAbs := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Abs(param) }
		y0 := testExecUnaryOp(t, buildFnAbs, float32(-7))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(float32(7)), y0.flat); !ok {
			t.Errorf("Abs float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnAbs, []int32{-1, 2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{1, 2}), y1.flat); !ok {
			t.Errorf("Abs int32 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnAbs, []uint32{1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint32{1, 2, 3}), y2.flat); !ok {
			t.Errorf("Abs uint32 mismatch:\n%s", diff)
		}
	})
	t.Run("Sign", func(t *testing.T) {
		buildFnSign := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Sign(param) }
		y0 := testExecUnaryOp(t, buildFnSign, float32(-7))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(float32(-1)), y0.flat); !ok {
			t.Errorf("Sign float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnSign, []int32{-1, 0, 2})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{-1, 0, 1}), y1.flat); !ok {
			t.Errorf("Sign int32 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnSign, []uint32{1, 0, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint32{1, 0, 1}), y2.flat); !ok {
			t.Errorf("Sign uint32 mismatch:\n%s", diff)
		}
	})
	t.Run("LogicalNot", func(t *testing.T) {
		buildFnLogicalNot := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.LogicalNot(param) }
		y0 := testExecUnaryOp(t, buildFnLogicalNot, true)
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(false), y0.flat); !ok {
			t.Errorf("LogicalNot bool mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnLogicalNot, []bool{true, false, true})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]bool{false, true, false}), y1.flat); !ok {
			t.Errorf("LogicalNot bool slice mismatch:\n%s", diff)
		}
	})
	t.Run("BitwiseNot", func(t *testing.T) {
		buildFnBitwiseNot := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.BitwiseNot(param) }
		y0 := testExecUnaryOp(t, buildFnBitwiseNot, int32(7))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(int32(-8)), y0.flat); !ok {
			t.Errorf("BitwiseNot int32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnBitwiseNot, []int32{-1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{0, -3, -4}), y1.flat); !ok {
			t.Errorf("BitwiseNot int32 slice mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnBitwiseNot, []uint32{1, 2, 3})
		want := []uint32{^uint32(1), ^uint32(2), ^uint32(3)}
		if ok, diff := testutil.IsEqual(want, y2.flat); !ok {
			t.Errorf("BitwiseNot uint32 slice mismatch:\n%s", diff)
		}
	})
	t.Run("BitCount", func(t *testing.T) {
		buildFnBitCount := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.BitCount(param) }
		y0 := testExecUnaryOp(t, buildFnBitCount, int8(7))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(int8(3)), y0.flat); !ok {
			t.Errorf("BitCount int8 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnBitCount, []int8{-1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int8{8, 1, 2}), y1.flat); !ok {
			t.Errorf("BitCount int8 slice mismatch:\n%s", diff)
		}

		y2 := testExecUnaryOp(t, buildFnBitCount, uint16(15))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(uint16(4)), y2.flat); !ok {
			t.Errorf("BitCount uint16 mismatch:\n%s", diff)
		}
		y3 := testExecUnaryOp(t, buildFnBitCount, []uint16{1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint16{1, 1, 2}), y3.flat); !ok {
			t.Errorf("BitCount uint16 slice mismatch:\n%s", diff)
		}

		y4 := testExecUnaryOp(t, buildFnBitCount, int32(31))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(int32(5)), y4.flat); !ok {
			t.Errorf("BitCount int32 mismatch:\n%s", diff)
		}
		y5 := testExecUnaryOp(t, buildFnBitCount, []int32{-1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{32, 1, 2}), y5.flat); !ok {
			t.Errorf("BitCount int32 slice mismatch:\n%s", diff)
		}

		y6 := testExecUnaryOp(t, buildFnBitCount, uint64(63))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(uint64(6)), y6.flat); !ok {
			t.Errorf("BitCount uint64 mismatch:\n%s", diff)
		}
		y7 := testExecUnaryOp(t, buildFnBitCount, []uint64{1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint64{1, 1, 2}), y7.flat); !ok {
			t.Errorf("BitCount uint64 slice mismatch:\n%s", diff)
		}
	})
	t.Run("Clz", func(t *testing.T) {
		buildFnClz := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Clz(param) }
		y0 := testExecUnaryOp(t, buildFnClz, int8(7))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(int8(5)), y0.flat); !ok {
			t.Errorf("Clz int8 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnClz, []int8{1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int8{7, 6, 6}), y1.flat); !ok {
			t.Errorf("Clz int8 slice mismatch:\n%s", diff)
		}

		y2 := testExecUnaryOp(t, buildFnClz, uint16(15))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(uint16(12)), y2.flat); !ok {
			t.Errorf("Clz uint16 mismatch:\n%s", diff)
		}
		y3 := testExecUnaryOp(t, buildFnClz, []uint16{1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint16{15, 14, 14}), y3.flat); !ok {
			t.Errorf("Clz uint16 slice mismatch:\n%s", diff)
		}

		y4 := testExecUnaryOp(t, buildFnClz, int32(31))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(int32(27)), y4.flat); !ok {
			t.Errorf("Clz int32 mismatch:\n%s", diff)
		}
		y5 := testExecUnaryOp(t, buildFnClz, []int32{1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]int32{31, 30, 30}), y5.flat); !ok {
			t.Errorf("Clz int32 slice mismatch:\n%s", diff)
		}

		y6 := testExecUnaryOp(t, buildFnClz, uint64(63))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(uint64(58)), y6.flat); !ok {
			t.Errorf("Clz uint64 mismatch:\n%s", diff)
		}
		y7 := testExecUnaryOp(t, buildFnClz, []uint64{1, 2, 3})
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice([]uint64{63, 62, 62}), y7.flat); !ok {
			t.Errorf("Clz uint64 slice mismatch:\n%s", diff)
		}
	})
	t.Run("Exp", func(t *testing.T) {
		buildFnExp := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Exp(param) }
		y0 := testExecUnaryOp(t, buildFnExp, float32(1.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(2.718281828459045)), y0.flat, 1e-6); !ok {
			t.Errorf("Exp float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnExp, float64(1.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(2.718281828459045), y1.flat, 1e-15); !ok {
			t.Errorf("Exp float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnExp, bfloat16.FromFloat32(1.0))
		want := bfloat16.FromFloat32(float32(math.E)).Float32()
		if ok, diff := testutil.IsInDelta(want, y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Exp bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Expm1", func(t *testing.T) {
		buildFnExpm1 := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Expm1(param) }
		y0 := testExecUnaryOp(t, buildFnExpm1, float32(1.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(1.71828)), y0.flat, 1e-4); !ok {
			t.Errorf("Expm1 float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnExpm1, float64(1.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(1.71828), y1.flat, 1e-4); !ok {
			t.Errorf("Expm1 float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnExpm1, bfloat16.FromFloat32(1.0))
		want := bfloat16.FromFloat32(float32(math.E - 1.0)).Float32()
		if ok, diff := testutil.IsInDelta(want, y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Expm1 bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Log", func(t *testing.T) {
		buildFnLog := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Log(param) }
		y0 := testExecUnaryOp(t, buildFnLog, float32(2.718281828459045))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(1.0)), y0.flat, 1e-6); !ok {
			t.Errorf("Log float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnLog, float64(2.718281828459045))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(1.0), y1.flat, 1e-15); !ok {
			t.Errorf("Log float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnLog, bfloat16.FromFloat32(2.718281828459045))
		if ok, diff := testutil.IsInDelta(float32(1.0), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Log bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Log1p", func(t *testing.T) {
		buildFnLog1p := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Log1p(param) }
		y0 := testExecUnaryOp(t, buildFnLog1p, float32(1.718281828459045))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(1.0)), y0.flat, 1e-6); !ok {
			t.Errorf("Log1p float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnLog1p, float64(1.718281828459045))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(1.0), y1.flat, 1e-15); !ok {
			t.Errorf("Log1p float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnLog1p, bfloat16.FromFloat32(1.718281828459045))
		if ok, diff := testutil.IsInDelta(float32(1.0), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Log1p bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Ceil", func(t *testing.T) {
		buildFnCeil := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Ceil(param) }
		y0 := testExecUnaryOp(t, buildFnCeil, float32(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(float32(2.0)), y0.flat); !ok {
			t.Errorf("Ceil float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnCeil, float64(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(2.0), y1.flat); !ok {
			t.Errorf("Ceil float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnCeil, bfloat16.FromFloat32(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(2.0)), y2.flat); !ok {
			t.Errorf("Ceil bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Floor", func(t *testing.T) {
		buildFnFloor := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Floor(param) }
		y0 := testExecUnaryOp(t, buildFnFloor, float32(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(float32(1.0)), y0.flat); !ok {
			t.Errorf("Floor float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnFloor, float64(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(1.0), y1.flat); !ok {
			t.Errorf("Floor float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnFloor, bfloat16.FromFloat32(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(1.0)), y2.flat); !ok {
			t.Errorf("Floor bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Round", func(t *testing.T) {
		buildFnRound := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Round(param) }
		y0 := testExecUnaryOp(t, buildFnRound, float32(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(float32(2.0)), y0.flat); !ok {
			t.Errorf("Round float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnRound, float64(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(2.0), y1.flat); !ok {
			t.Errorf("Round float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnRound, bfloat16.FromFloat32(1.6))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(bfloat16.FromFloat32(2.0)), y2.flat); !ok {
			t.Errorf("Round bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Rsqrt", func(t *testing.T) {
		buildFnRsqrt := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Rsqrt(param) }
		y0 := testExecUnaryOp(t, buildFnRsqrt, float32(4.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(0.5)), y0.flat, 1e-6); !ok {
			t.Errorf("Rsqrt float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnRsqrt, float64(4.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(0.5), y1.flat, 1e-15); !ok {
			t.Errorf("Rsqrt float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnRsqrt, bfloat16.FromFloat32(4.0))
		if ok, diff := testutil.IsInDelta(float32(0.5), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Rsqrt bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Sqrt", func(t *testing.T) {
		buildFnSqrt := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Sqrt(param) }
		y0 := testExecUnaryOp(t, buildFnSqrt, float32(4.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(2.0)), y0.flat, 1e-6); !ok {
			t.Errorf("Sqrt float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnSqrt, float64(4.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(2.0), y1.flat, 1e-15); !ok {
			t.Errorf("Sqrt float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnSqrt, bfloat16.FromFloat32(4.0))
		if ok, diff := testutil.IsInDelta(float32(2.0), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Sqrt bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Cos", func(t *testing.T) {
		buildFnCos := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Cos(param) }
		y0 := testExecUnaryOp(t, buildFnCos, float32(0.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(1.0)), y0.flat, 1e-6); !ok {
			t.Errorf("Cos float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnCos, float64(0.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(1.0), y1.flat, 1e-15); !ok {
			t.Errorf("Cos float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnCos, bfloat16.FromFloat32(0.0))
		if ok, diff := testutil.IsInDelta(float32(1.0), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Cos bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Sin", func(t *testing.T) {
		buildFnSin := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Sin(param) }
		y0 := testExecUnaryOp(t, buildFnSin, float32(math.Pi/2))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(1.0)), y0.flat, 1e-6); !ok {
			t.Errorf("Sin float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnSin, float64(math.Pi/2))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(1.0), y1.flat, 1e-15); !ok {
			t.Errorf("Sin float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnSin, bfloat16.FromFloat32(float32(math.Pi/2)))
		if ok, diff := testutil.IsInDelta(float32(1.0), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Sin bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Tanh", func(t *testing.T) {
		buildFnTanh := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Tanh(param) }
		y0 := testExecUnaryOp(t, buildFnTanh, float32(0.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(0.0)), y0.flat, 1e-6); !ok {
			t.Errorf("Tanh float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnTanh, float64(0.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(0.0), y1.flat, 1e-15); !ok {
			t.Errorf("Tanh float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnTanh, bfloat16.FromFloat32(0.0))
		if ok, diff := testutil.IsInDelta(float32(0.0), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Tanh bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("Logistic", func(t *testing.T) {
		buildFnLogistic := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Logistic(param) }
		y0 := testExecUnaryOp(t, buildFnLogistic, float32(0.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(0.5)), y0.flat, 1e-6); !ok {
			t.Errorf("Logistic float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnLogistic, float64(2.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(0.8808), y1.flat, 1e-4); !ok {
			t.Errorf("Logistic float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnLogistic, bfloat16.FromFloat32(-2.0))
		if ok, diff := testutil.IsInDelta(float32(0.1192), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Logistic bfloat16 mismatch:\n%s", diff)
		}
	})
	t.Run("IsFinite", func(t *testing.T) {
		buildFnIsFinite := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.IsFinite(param) }

		// Test float32
		y0 := testExecUnaryOp(t, buildFnIsFinite, float32(1.0))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y0.flat); !ok {
			t.Errorf("IsFinite float32 (finite) mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnIsFinite, float32(math.Inf(1)))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(false), y1.flat); !ok {
			t.Errorf("IsFinite float32 (inf) mismatch:\n%s", diff)
		}

		// Test float64
		y2 := testExecUnaryOp(t, buildFnIsFinite, float64(1.0))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y2.flat); !ok {
			t.Errorf("IsFinite float64 (finite) mismatch:\n%s", diff)
		}
		y3 := testExecUnaryOp(t, buildFnIsFinite, math.Inf(-1))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(false), y3.flat); !ok {
			t.Errorf("IsFinite float64 (inf) mismatch:\n%s", diff)
		}

		// Test bfloat16
		y4 := testExecUnaryOp(t, buildFnIsFinite, bfloat16.FromFloat32(float32(math.NaN())))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(false), y4.flat); !ok {
			t.Errorf("IsFinite bfloat16 (nan) mismatch:\n%s", diff)
		}
		y5 := testExecUnaryOp(t, buildFnIsFinite, bfloat16.FromFloat32(1.0))
		if ok, diff := testutil.IsEqual(testutil.FlattenSlice(true), y5.flat); !ok {
			t.Errorf("IsFinite bfloat16 (finite) mismatch:\n%s", diff)
		}
	})
	t.Run("Erf", func(t *testing.T) {
		buildFnErf := func(f compute.Function, param compute.Value) (compute.Value, error) { return f.Erf(param) }
		y0 := testExecUnaryOp(t, buildFnErf, float32(1.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(float32(0.8427)), y0.flat, 1e-4); !ok {
			t.Errorf("Erf float32 mismatch:\n%s", diff)
		}
		y1 := testExecUnaryOp(t, buildFnErf, float64(1.0))
		if ok, diff := testutil.IsInDelta(testutil.FlattenSlice(0.8427), y1.flat, 1e-4); !ok {
			t.Errorf("Erf float64 mismatch:\n%s", diff)
		}
		y2 := testExecUnaryOp(t, buildFnErf, bfloat16.FromFloat32(1.0))
		if ok, diff := testutil.IsInDelta(float32(0.8427), y2.flat.([]bfloat16.BFloat16)[0].Float32(), 1e-2); !ok {
			t.Errorf("Erf bfloat16 mismatch:\n%s", diff)
		}
	})
}
func TestBackendIsSimpleGo(t *testing.T) {
	if panicked, _ := testutil.Try(func() { _ = backend.(*Backend) }); panicked {
		t.Errorf("Expected no panic when casting backend to *Backend")
	}
}

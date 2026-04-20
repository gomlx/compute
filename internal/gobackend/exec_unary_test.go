// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/internal/testutil"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

func TestBackendIsSimpleGo(t *testing.T) {
	if panicked, _ := testutil.Try(func() { _ = backend.(*Backend) }); panicked {
		t.Errorf("Expected no panic when casting backend to *Backend")
	}
}

func TestExecUnary_Neg(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Neg)
	y0 := exec.MustExec(float32(7))[0]
	if ok, diff := testutil.IsEqual(float32(-7), y0.Value()); !ok {
		t.Errorf("Neg float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec([]int32{-1, 2})[0]
	if ok, diff := testutil.IsEqual([]int32{1, -2}, y1.Value()); !ok {
		t.Errorf("Neg int32 mismatch:\n%s", diff)
	}
	if panicked, _ := testutil.Try(func() { _ = exec.MustExec([]uint32{1, 2, 3}) }); !panicked {
		t.Errorf("Expected panic when Neg on uint32")
	}
}

func TestExecUnary_Abs(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Abs)
	y0 := exec.MustExec(float32(-7))[0]
	if ok, diff := testutil.IsEqual(float32(7), y0.Value()); !ok {
		t.Errorf("Abs float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec([]int32{-1, 2})[0]
	if ok, diff := testutil.IsEqual([]int32{1, 2}, y1.Value()); !ok {
		t.Errorf("Abs int32 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec([]uint32{1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]uint32{1, 2, 3}, y2.Value()); !ok {
		t.Errorf("Abs uint32 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Sign(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Sign)
	y0 := exec.MustExec(float32(-7))[0]
	if ok, diff := testutil.IsEqual(float32(-1), y0.Value()); !ok {
		t.Errorf("Sign float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec([]int32{-1, 0, 2})[0]
	if ok, diff := testutil.IsEqual([]int32{-1, 0, 1}, y1.Value()); !ok {
		t.Errorf("Sign int32 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec([]uint32{1, 0, 3})[0]
	if ok, diff := testutil.IsEqual([]uint32{1, 0, 1}, y2.Value()); !ok {
		t.Errorf("Sign uint32 mismatch:\n%s", diff)
	}
}

func TestExecUnary_LogicalNot(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.LogicalNot)
	y0 := exec.MustExec(true)[0]
	if ok, diff := testutil.IsEqual(false, y0.Value()); !ok {
		t.Errorf("LogicalNot bool mismatch:\n%s", diff)
	}
	y1 := exec.MustExec([]bool{true, false, true})[0]
	if ok, diff := testutil.IsEqual([]bool{false, true, false}, y1.Value()); !ok {
		t.Errorf("LogicalNot bool slice mismatch:\n%s", diff)
	}
}

func TestExecUnary_BitwiseNot(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.BitwiseNot)
	y0 := exec.MustExec(int32(7))[0]
	if ok, diff := testutil.IsEqual(int32(-8), y0.Value()); !ok {
		t.Errorf("BitwiseNot int32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec([]int32{-1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]int32{0, -3, -4}, y1.Value()); !ok {
		t.Errorf("BitwiseNot int32 slice mismatch:\n%s", diff)
	}
	y2 := exec.MustExec([]uint32{1, 2, 3})[0]
	want := []uint32{^uint32(1), ^uint32(2), ^uint32(3)}
	if ok, diff := testutil.IsEqual(want, y2.Value()); !ok {
		t.Errorf("BitwiseNot uint32 slice mismatch:\n%s", diff)
	}
}

func TestExecUnary_BitCount(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.BitCount)
	y0 := exec.MustExec(int8(7))[0]
	if ok, diff := testutil.IsEqual(int8(3), y0.Value()); !ok {
		t.Errorf("BitCount int8 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec([]int8{-1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]int8{8, 1, 2}, y1.Value()); !ok {
		t.Errorf("BitCount int8 slice mismatch:\n%s", diff)
	}

	y2 := exec.MustExec(uint16(15))[0]
	if ok, diff := testutil.IsEqual(uint16(4), y2.Value()); !ok {
		t.Errorf("BitCount uint16 mismatch:\n%s", diff)
	}
	y3 := exec.MustExec([]uint16{1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]uint16{1, 1, 2}, y3.Value()); !ok {
		t.Errorf("BitCount uint16 slice mismatch:\n%s", diff)
	}

	y4 := exec.MustExec(int32(31))[0]
	if ok, diff := testutil.IsEqual(int32(5), y4.Value()); !ok {
		t.Errorf("BitCount int32 mismatch:\n%s", diff)
	}
	y5 := exec.MustExec([]int32{-1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]int32{32, 1, 2}, y5.Value()); !ok {
		t.Errorf("BitCount int32 slice mismatch:\n%s", diff)
	}

	y6 := exec.MustExec(uint64(63))[0]
	if ok, diff := testutil.IsEqual(uint64(6), y6.Value()); !ok {
		t.Errorf("BitCount uint64 mismatch:\n%s", diff)
	}
	y7 := exec.MustExec([]uint64{1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]uint64{1, 1, 2}, y7.Value()); !ok {
		t.Errorf("BitCount uint64 slice mismatch:\n%s", diff)
	}
}

func TestExecUnary_Clz(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Clz)
	y0 := exec.MustExec(int8(7))[0]
	if ok, diff := testutil.IsEqual(int8(5), y0.Value()); !ok {
		t.Errorf("Clz int8 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec([]int8{1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]int8{7, 6, 6}, y1.Value()); !ok {
		t.Errorf("Clz int8 slice mismatch:\n%s", diff)
	}

	y2 := exec.MustExec(uint16(15))[0]
	if ok, diff := testutil.IsEqual(uint16(12), y2.Value()); !ok {
		t.Errorf("Clz uint16 mismatch:\n%s", diff)
	}
	y3 := exec.MustExec([]uint16{1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]uint16{15, 14, 14}, y3.Value()); !ok {
		t.Errorf("Clz uint16 slice mismatch:\n%s", diff)
	}

	y4 := exec.MustExec(int32(31))[0]
	if ok, diff := testutil.IsEqual(int32(27), y4.Value()); !ok {
		t.Errorf("Clz int32 mismatch:\n%s", diff)
	}
	y5 := exec.MustExec([]int32{1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]int32{31, 30, 30}, y5.Value()); !ok {
		t.Errorf("Clz int32 slice mismatch:\n%s", diff)
	}

	y6 := exec.MustExec(uint64(63))[0]
	if ok, diff := testutil.IsEqual(uint64(58), y6.Value()); !ok {
		t.Errorf("Clz uint64 mismatch:\n%s", diff)
	}
	y7 := exec.MustExec([]uint64{1, 2, 3})[0]
	if ok, diff := testutil.IsEqual([]uint64{63, 62, 62}, y7.Value()); !ok {
		t.Errorf("Clz uint64 slice mismatch:\n%s", diff)
	}
}

func TestExecUnary_Exp(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Exp)
	y0 := exec.MustExec(float32(1.0))[0]
	if ok, diff := testutil.IsInDelta(float32(2.718281828459045), y0.Value(), 1e-6); !ok {
		t.Errorf("Exp float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(1.0))[0]
	if ok, diff := testutil.IsInDelta(2.718281828459045, y1.Value(), 1e-15); !ok {
		t.Errorf("Exp float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(1.0))[0]
	want := bfloat16.FromFloat32(float32(math.E)).Float32()
	if ok, diff := testutil.IsInDelta(want, y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Exp bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Expm1(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Expm1)
	y0 := exec.MustExec(float32(1.0))[0]
	if ok, diff := testutil.IsInDelta(float32(1.71828), y0.Value(), 1e-4); !ok {
		t.Errorf("Expm1 float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(1.0))[0]
	if ok, diff := testutil.IsInDelta(1.71828, y1.Value(), 1e-4); !ok {
		t.Errorf("Expm1 float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(1.0))[0]
	want := bfloat16.FromFloat32(float32(math.E - 1.0)).Float32()
	if ok, diff := testutil.IsInDelta(want, y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Expm1 bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Log(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Log)
	y0 := exec.MustExec(float32(2.718281828459045))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y0.Value(), 1e-6); !ok {
		t.Errorf("Log float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(2.718281828459045))[0]
	if ok, diff := testutil.IsInDelta(1.0, y1.Value(), 1e-15); !ok {
		t.Errorf("Log float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(2.718281828459045))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Log bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Log1p(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Log1p)
	y0 := exec.MustExec(float32(1.718281828459045))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y0.Value(), 1e-6); !ok {
		t.Errorf("Log1p float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(1.718281828459045))[0]
	if ok, diff := testutil.IsInDelta(1.0, y1.Value(), 1e-15); !ok {
		t.Errorf("Log1p float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(1.718281828459045))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Log1p bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Ceil(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Ceil)
	y0 := exec.MustExec(float32(1.6))[0]
	if ok, diff := testutil.IsEqual(float32(2.0), y0.Value()); !ok {
		t.Errorf("Ceil float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(1.6))[0]
	if ok, diff := testutil.IsEqual(2.0, y1.Value()); !ok {
		t.Errorf("Ceil float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(1.6))[0]
	if ok, diff := testutil.IsEqual(bfloat16.FromFloat32(2.0), y2.Value()); !ok {
		t.Errorf("Ceil bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Floor(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Floor)
	y0 := exec.MustExec(float32(1.6))[0]
	if ok, diff := testutil.IsEqual(float32(1.0), y0.Value()); !ok {
		t.Errorf("Floor float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(1.6))[0]
	if ok, diff := testutil.IsEqual(1.0, y1.Value()); !ok {
		t.Errorf("Floor float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(1.6))[0]
	if ok, diff := testutil.IsEqual(bfloat16.FromFloat32(1.0), y2.Value()); !ok {
		t.Errorf("Floor bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Round(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Round)
	y0 := exec.MustExec(float32(1.6))[0]
	if ok, diff := testutil.IsEqual(float32(2.0), y0.Value()); !ok {
		t.Errorf("Round float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(1.6))[0]
	if ok, diff := testutil.IsEqual(2.0, y1.Value()); !ok {
		t.Errorf("Round float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(1.6))[0]
	if ok, diff := testutil.IsEqual(bfloat16.FromFloat32(2.0), y2.Value()); !ok {
		t.Errorf("Round bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Rsqrt(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Rsqrt)
	y0 := exec.MustExec(float32(4.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.5), y0.Value(), 1e-6); !ok {
		t.Errorf("Rsqrt float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(4.0))[0]
	if ok, diff := testutil.IsInDelta(0.5, y1.Value(), 1e-15); !ok {
		t.Errorf("Rsqrt float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(4.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.5), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Rsqrt bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Sqrt(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Sqrt)
	y0 := exec.MustExec(float32(4.0))[0]
	if ok, diff := testutil.IsInDelta(float32(2.0), y0.Value(), 1e-6); !ok {
		t.Errorf("Sqrt float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(4.0))[0]
	if ok, diff := testutil.IsInDelta(2.0, y1.Value(), 1e-15); !ok {
		t.Errorf("Sqrt float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(4.0))[0]
	if ok, diff := testutil.IsInDelta(float32(2.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Sqrt bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Cos(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Cos)
	y0 := exec.MustExec(float32(0.0))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y0.Value(), 1e-6); !ok {
		t.Errorf("Cos float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(0.0))[0]
	if ok, diff := testutil.IsInDelta(1.0, y1.Value(), 1e-15); !ok {
		t.Errorf("Cos float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(0.0))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Cos bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Sin(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Sin)
	y0 := exec.MustExec(float32(math.Pi / 2))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y0.Value(), 1e-6); !ok {
		t.Errorf("Sin float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(math.Pi / 2))[0]
	if ok, diff := testutil.IsInDelta(1.0, y1.Value(), 1e-15); !ok {
		t.Errorf("Sin float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(float32(math.Pi / 2)))[0]
	if ok, diff := testutil.IsInDelta(float32(1.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Sin bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Tanh(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Tanh)
	y0 := exec.MustExec(float32(0.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.0), y0.Value(), 1e-6); !ok {
		t.Errorf("Tanh float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(0.0))[0]
	if ok, diff := testutil.IsInDelta(0.0, y1.Value(), 1e-15); !ok {
		t.Errorf("Tanh float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(0.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.0), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Tanh bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_Logistic(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Logistic)
	y0 := exec.MustExec(float32(0.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.5), y0.Value(), 1e-6); !ok {
		t.Errorf("Logistic float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(2.0))[0]
	if ok, diff := testutil.IsInDelta(0.8808, y1.Value(), 1e-4); !ok {
		t.Errorf("Logistic float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(-2.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.1192), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Logistic bfloat16 mismatch:\n%s", diff)
	}
}

func TestExecUnary_IsFinite(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.IsFinite)

	// Test float32
	y0 := exec.MustExec(float32(1.0))[0]
	if ok, diff := testutil.IsEqual(true, y0.Value()); !ok {
		t.Errorf("IsFinite float32 (finite) mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float32(math.Inf(1)))[0]
	if ok, diff := testutil.IsEqual(false, y1.Value()); !ok {
		t.Errorf("IsFinite float32 (inf) mismatch:\n%s", diff)
	}

	// Test float64
	y2 := exec.MustExec(float64(1.0))[0]
	if ok, diff := testutil.IsEqual(true, y2.Value()); !ok {
		t.Errorf("IsFinite float64 (finite) mismatch:\n%s", diff)
	}
	y3 := exec.MustExec(math.Inf(-1))[0]
	if ok, diff := testutil.IsEqual(false, y3.Value()); !ok {
		t.Errorf("IsFinite float64 (inf) mismatch:\n%s", diff)
	}

	// Test bfloat16
	y4 := exec.MustExec(bfloat16.FromFloat32(float32(math.NaN())))[0]
	if ok, diff := testutil.IsEqual(false, y4.Value()); !ok {
		t.Errorf("IsFinite bfloat16 (nan) mismatch:\n%s", diff)
	}
	y5 := exec.MustExec(bfloat16.FromFloat32(1.0))[0]
	if ok, diff := testutil.IsEqual(true, y5.Value()); !ok {
		t.Errorf("IsFinite bfloat16 (finite) mismatch:\n%s", diff)
	}
}

func TestExecUnary_Erf(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Erf)
	y0 := exec.MustExec(float32(1.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.8427), y0.Value(), 1e-4); !ok {
		t.Errorf("Erf float32 mismatch:\n%s", diff)
	}
	y1 := exec.MustExec(float64(1.0))[0]
	if ok, diff := testutil.IsInDelta(0.8427, y1.Value(), 1e-4); !ok {
		t.Errorf("Erf float64 mismatch:\n%s", diff)
	}
	y2 := exec.MustExec(bfloat16.FromFloat32(1.0))[0]
	if ok, diff := testutil.IsInDelta(float32(0.8427), y2.Value().(bfloat16.BFloat16).Float32(), 1e-2); !ok {
		t.Errorf("Erf bfloat16 mismatch:\n%s", diff)
	}
}

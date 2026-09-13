// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dot_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/defaultpkgs"
	"github.com/gomlx/compute/internal/must"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"k8s.io/klog/v2"
)

var backend compute.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", compute.List())
	// Perform your setup logic here
	if os.Getenv(compute.ConfigEnvVar) == "" {
		must.M(os.Setenv(compute.ConfigEnvVar, "go"))
	} else {
		fmt.Printf("\t$%s=%q\n", compute.ConfigEnvVar, os.Getenv(compute.ConfigEnvVar))
	}
	backend = compute.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

func TestDotGeneral(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not the Go backend")
	}

	lhs := [][][]float32{{{1, 2, 3}}, {{4, 5, 6}}}
	rhs := [][][]float32{{{1, 1}, {1, 1}, {1, 1}}, {{1, 1}, {1, 1}, {1, 1}}}
	want := [][][]float32{{{6, 6}}, {{15, 15}}}

	y1, err := testutil.Exec1(backend, []any{lhs, rhs}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.DotGeneral(params[0], []int{2}, []int{0}, params[1], []int{1}, []int{0}, compute.DotGeneralConfig{})
	})
	if err != nil {
		t.Fatalf("testutil.Exec1 failed: %v", err)
	}
	if ok, diff := testutil.IsEqual(want, y1); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
}

func TestDotGeneralDynamic(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not the Go backend")
	}

	// Dynamic test simulating attention score Einsum: [batch, heads, seq, dim] x [batch, heads, seq, dim] -> [batch, heads, seq, seq]
	// Batch axes: [0, 1] (batch is dynamic, heads is static 2).
	// Contracting axis: [3] (dim is 4).
	// Cross axis: [2] (seq is dynamic).
	builder := backend.Builder("TestDotGeneralDynamic")
	mainFn := builder.Main()
	sLHS := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 2, shapes.DynamicDim, 4}, []string{"batch", "", "seq", ""})
	sRHS := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 2, shapes.DynamicDim, 4}, []string{"batch", "", "seq", ""})
	lhsParam, err := mainFn.Parameter("lhs", sLHS, nil)
	if err != nil {
		t.Fatal(err)
	}
	rhsParam, err := mainFn.Parameter("rhs", sRHS, nil)
	if err != nil {
		t.Fatal(err)
	}
	dotNode, err := mainFn.DotGeneral(lhsParam, []int{3}, []int{0, 1}, rhsParam, []int{3}, []int{0, 1}, compute.DotGeneralConfig{})
	if err != nil {
		t.Fatal(err)
	}
	err = mainFn.Return([]compute.Value{dotNode}, nil)
	if err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}

	// Concrete shapes: batch=3, heads=2, seq=5, dim=4
	concreteShape := shapes.Make(dtypes.Float32, 3, 2, 5, 4)
	lhsBuf, _ := backend.BufferFromFlatData(0, make([]float32, concreteShape.Size()), concreteShape)
	rhsBuf, _ := backend.BufferFromFlatData(0, make([]float32, concreteShape.Size()), concreteShape)
	outputs, err := exec.Execute([]compute.Buffer{lhsBuf, rhsBuf}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	outShape, err := outputs[0].Shape()
	if err != nil {
		t.Fatal(err)
	}
	if err := outShape.Check(dtypes.Float32, 3, 2, 5, 5); err != nil {
		t.Fatalf("unexpected output shape: %v", err)
	}
}

func TestDotGeneralConstantCaching(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not the Go backend")
	}

	// Test caching of packed constant RHS across dynamic batch executions.
	builder := backend.Builder("TestDotGeneralConstantCaching")
	mainFn := builder.Main()
	sLHS := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 64}, []string{"batch", ""})
	lhsParam, err := mainFn.Parameter("lhs", sLHS, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Constant RHS: [64, 128]
	rhsData := make([]float32, 64*128)
	for i := range rhsData {
		rhsData[i] = float32(i % 7)
	}
	rhsConst, err := mainFn.Constant(rhsData, 64, 128)
	if err != nil {
		t.Fatal(err)
	}
	dotNode, err := mainFn.DotGeneral(lhsParam, []int{1}, nil, rhsConst, []int{0}, nil, compute.DotGeneralConfig{})
	if err != nil {
		t.Fatal(err)
	}
	err = mainFn.Return([]compute.Value{dotNode}, nil)
	if err != nil {
		t.Fatal(err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}

	// Run with batch = 16
	concreteShape1 := shapes.Make(dtypes.Float32, 16, 64)
	lhsData1 := make([]float32, 16*64)
	for i := range lhsData1 {
		lhsData1[i] = 1.0
	}
	lhsBuf1, err := backend.BufferFromFlatData(0, lhsData1, concreteShape1)
	if err != nil {
		t.Fatal(err)
	}
	outputs1, err := exec.Execute([]compute.Buffer{lhsBuf1}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	out1Shape, _ := outputs1[0].Shape()
	if err := out1Shape.Check(dtypes.Float32, 16, 128); err != nil {
		t.Fatalf("unexpected output shape: %v", err)
	}

	// Run with batch = 32 (different dynamic shape specialization)
	concreteShape2 := shapes.Make(dtypes.Float32, 32, 64)
	lhsData2 := make([]float32, 32*64)
	for i := range lhsData2 {
		lhsData2[i] = 1.0
	}
	lhsBuf2, err := backend.BufferFromFlatData(0, lhsData2, concreteShape2)
	if err != nil {
		t.Fatal(err)
	}
	outputs2, err := exec.Execute([]compute.Buffer{lhsBuf2}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	out2Shape, _ := outputs2[0].Shape()
	if err := out2Shape.Check(dtypes.Float32, 32, 128); err != nil {
		t.Fatalf("unexpected output shape: %v", err)
	}

	// Verify values for the first 16 rows match between run 1 and run 2
	out1Data := make([]float32, 16*128)
	if err := outputs1[0].ToFlatData(out1Data); err != nil {
		t.Fatal(err)
	}
	out2Data := make([]float32, 32*128)
	if err := outputs2[0].ToFlatData(out2Data); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16*128; i++ {
		if out1Data[i] != out2Data[i] {
			t.Fatalf("output values mismatch at index %d: %v != %v", i, out1Data[i], out2Data[i])
		}
	}
}

func TestDotGeneralConstantCachingLargeParallel(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not the Go backend")
	}

	const (
		M = 1024 // LHS rows (spans across multiple LHS panels)
		K = 768  // Contracting dim (spans across multiple K panels)
		N = 1024 // RHS cols (spans across multiple RHS panels)
	)

	t.Run("ConstantRHS", func(t *testing.T) {
		builder := backend.Builder("TestDotGeneralConstantCachingLargeParallel_RHS")
		mainFn := builder.Main()

		sLHS := shapes.Make(dtypes.Float32, M, K)
		lhsParam, err := mainFn.Parameter("lhs", sLHS, nil)
		if err != nil {
			t.Fatal(err)
		}

		rhsData := make([]float32, K*N)
		for k := range K {
			val := float32(k % 13)
			for n := range N {
				rhsData[k*N+n] = val
			}
		}
		rhsConst, err := mainFn.Constant(rhsData, K, N)
		if err != nil {
			t.Fatal(err)
		}

		dotNode, err := mainFn.DotGeneral(lhsParam, []int{1}, nil, rhsConst, []int{0}, nil, compute.DotGeneralConfig{})
		if err != nil {
			t.Fatal(err)
		}
		err = mainFn.Return([]compute.Value{dotNode}, nil)
		if err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}

		// Expected dot result for a row of all 1s: sum_{k=0..K-1} (k % 13)
		var expectedSum float32
		for k := range K {
			expectedSum += float32(k % 13)
		}

		// Run 1: LHS all 1s
		lhsData1 := make([]float32, M*K)
		for i := range lhsData1 {
			lhsData1[i] = 1.0
		}
		lhsBuf1, err := backend.BufferFromFlatData(0, lhsData1, sLHS)
		if err != nil {
			t.Fatal(err)
		}
		out1, err := exec.Execute([]compute.Buffer{lhsBuf1}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		out1Data := make([]float32, M*N)
		if err := out1[0].ToFlatData(out1Data); err != nil {
			t.Fatal(err)
		}
		for i, v := range out1Data {
			if v != expectedSum {
				t.Fatalf("Run 1 mismatch at %d: got %v, want %v", i, v, expectedSum)
			}
		}

		// Run 2: LHS all 2s (caching should reuse packed RHS)
		lhsData2 := make([]float32, M*K)
		for i := range lhsData2 {
			lhsData2[i] = 2.0
		}
		lhsBuf2, err := backend.BufferFromFlatData(0, lhsData2, sLHS)
		if err != nil {
			t.Fatal(err)
		}
		out2, err := exec.Execute([]compute.Buffer{lhsBuf2}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		out2Data := make([]float32, M*N)
		if err := out2[0].ToFlatData(out2Data); err != nil {
			t.Fatal(err)
		}
		expectedSum2 := expectedSum * 2
		for i, v := range out2Data {
			if v != expectedSum2 {
				t.Fatalf("Run 2 mismatch at %d: got %v, want %v", i, v, expectedSum2)
			}
		}
	})

	t.Run("ConstantLHS", func(t *testing.T) {
		builder := backend.Builder("TestDotGeneralConstantCachingLargeParallel_LHS")
		mainFn := builder.Main()

		sRHS := shapes.Make(dtypes.Float32, K, N)
		rhsParam, err := mainFn.Parameter("rhs", sRHS, nil)
		if err != nil {
			t.Fatal(err)
		}

		lhsData := make([]float32, M*K)
		for m := range M {
			for k := range K {
				lhsData[m*K+k] = float32(k % 13)
			}
		}
		lhsConst, err := mainFn.Constant(lhsData, M, K)
		if err != nil {
			t.Fatal(err)
		}

		dotNode, err := mainFn.DotGeneral(lhsConst, []int{1}, nil, rhsParam, []int{0}, nil, compute.DotGeneralConfig{})
		if err != nil {
			t.Fatal(err)
		}
		err = mainFn.Return([]compute.Value{dotNode}, nil)
		if err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}

		var expectedSum float32
		for k := range K {
			expectedSum += float32(k % 13)
		}

		// Run 1: RHS all 1s
		rhsData1 := make([]float32, K*N)
		for i := range rhsData1 {
			rhsData1[i] = 1.0
		}
		rhsBuf1, err := backend.BufferFromFlatData(0, rhsData1, sRHS)
		if err != nil {
			t.Fatal(err)
		}
		out1, err := exec.Execute([]compute.Buffer{rhsBuf1}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		out1Data := make([]float32, M*N)
		if err := out1[0].ToFlatData(out1Data); err != nil {
			t.Fatal(err)
		}
		for i, v := range out1Data {
			if v != expectedSum {
				t.Fatalf("Run 1 mismatch at %d: got %v, want %v", i, v, expectedSum)
			}
		}

		// Run 2: RHS all 2s (caching should reuse packed LHS)
		rhsData2 := make([]float32, K*N)
		for i := range rhsData2 {
			rhsData2[i] = 2.0
		}
		rhsBuf2, err := backend.BufferFromFlatData(0, rhsData2, sRHS)
		if err != nil {
			t.Fatal(err)
		}
		out2, err := exec.Execute([]compute.Buffer{rhsBuf2}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		out2Data := make([]float32, M*N)
		if err := out2[0].ToFlatData(out2Data); err != nil {
			t.Fatal(err)
		}
		expectedSum2 := expectedSum * 2
		for i, v := range out2Data {
			if v != expectedSum2 {
				t.Fatalf("Run 2 mismatch at %d: got %v, want %v", i, v, expectedSum2)
			}
		}
	})
}

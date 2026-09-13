package matmul_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/backendtest"
	"github.com/gomlx/compute/support/humanize"
)

func TestNoSIMD(t *testing.T) {
	defer func() {
		dot.ResetTestRegistrations()
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = false
	}()
	dot.ResetTestRegistrations()
	matmul.RegisterNoSIMDForTests()

	t.Run("Small", func(t *testing.T) {
		matmul.ForceSmallVariant = true
		matmul.ForceLargeVariant = false
		backendtest.TestDotGeneral(t, backend)
	})

	t.Run("Large", func(t *testing.T) {
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = true
		backendtest.TestDotGeneral(t, backend)
	})
}

func TestNoSIMDConstantCaching(t *testing.T) {
	defer func() {
		dot.ResetTestRegistrations()
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = false
	}()
	dot.ResetTestRegistrations()
	matmul.RegisterNoSIMDForTests()
	matmul.ForceLargeVariant = true

	const (
		M = 64
		K = 768
		N = 1024
	)

	t.Run("ConstantRHS", func(t *testing.T) {
		builder := backend.Builder("TestNoSIMDConstantCaching_RHS")
		mainFn := builder.Main()

		sLHS := shapes.Make(dtypes.Float32, M, K)
		lhsParam, err := mainFn.Parameter("lhs", sLHS, nil)
		if err != nil {
			t.Fatal(err)
		}

		rhsData := make([]float32, K*N)
		for k := range K {
			val := float32(k % 7)
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

		var expectedSum float32
		for k := range K {
			expectedSum += float32(k % 7)
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
		builder := backend.Builder("TestNoSIMDConstantCaching_LHS")
		mainFn := builder.Main()

		sRHS := shapes.Make(dtypes.Float32, K, N)
		rhsParam, err := mainFn.Parameter("rhs", sRHS, nil)
		if err != nil {
			t.Fatal(err)
		}

		lhsData := make([]float32, M*K)
		for m := range M {
			for k := range K {
				lhsData[m*K+k] = float32(k % 7)
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
			expectedSum += float32(k % 7)
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

func BenchmarkSmallNoSIMD(b *testing.B) {
	defer func() {
		dot.ResetTestRegistrations()
		matmul.ForceSmallVariant = false
		matmul.ForceLargeVariant = false
	}()
	dot.ResetTestRegistrations()
	matmul.RegisterNoSIMDForTests()
	matmul.ForceSmallVariant = true
	matmul.ForceLargeVariant = false

	cases := []struct {
		name    string
		layout  dot.Layout
		M, K, N int
	}{
		{"NonTransposed/[128,4]x[4,1]", dot.LayoutNonTransposed, 128, 4, 1},
		{"NonTransposed/[128,69]x[69,4]", dot.LayoutNonTransposed, 128, 69, 4},
		{"NonTransposed/[25,4]x[4,1]", dot.LayoutNonTransposed, 25, 4, 1},
		{"NonTransposed/[25,69]x[69,4]", dot.LayoutNonTransposed, 25, 69, 4},
		{"NonTransposed/[49,4]x[4,1]", dot.LayoutNonTransposed, 49, 4, 1},
		{"NonTransposed/[49,69]x[69,4]", dot.LayoutNonTransposed, 49, 69, 4},

		{"Transposed/[128,4]x[1,4]", dot.LayoutTransposed, 128, 4, 1},
		{"Transposed/[128,69]x[4,69]", dot.LayoutTransposed, 128, 69, 4},
		{"Transposed/[25,4]x[1,4]", dot.LayoutTransposed, 25, 4, 1},
		{"Transposed/[25,69]x[4,69]", dot.LayoutTransposed, 25, 69, 4},
		{"Transposed/[49,4]x[1,4]", dot.LayoutTransposed, 49, 4, 1},
		{"Transposed/[49,69]x[4,69]", dot.LayoutTransposed, 49, 69, 4},
	}

	for _, tc := range cases {
		M, K, N := tc.M, tc.K, tc.N
		flops := float64(2 * M * N * K)
		lhs := make([]float32, M*K)
		var rhs []float32
		if tc.layout == dot.LayoutNonTransposed {
			rhs = make([]float32, K*N)
		} else {
			rhs = make([]float32, N*K)
		}
		out := make([]float32, M*N)

		b.Run(tc.name, func(b *testing.B) {
			bBackend := backend.(*gobackend.Backend)
			b.ResetTimer()
			for b.Loop() {
				matmul.TestNoSIMDRouterFloat32(bBackend, tc.layout, lhs, rhs, 1, M, N, K, out)
			}
			elapsed := b.Elapsed()
			if elapsed > 0 && b.N > 0 {
				gflops := (flops * float64(b.N) / elapsed.Seconds()) / 1e9
				b.ReportMetric(gflops, "GFlops/s")
				b.ReportMetric(humanize.DurationPerOp(elapsed, b.N))
			}
		})
	}
}

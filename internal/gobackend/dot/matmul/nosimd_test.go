package matmul_test

import (
	"testing"
	"time"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/support/backendtest"
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
				durationPerOp := time.Duration(float64(elapsed) / float64(b.N))
				b.ReportMetric(gflops, "GFlops/s")
				b.ReportMetric(durationPerOp.Seconds()*1e6, "µs/op")
			}
		})
	}
}

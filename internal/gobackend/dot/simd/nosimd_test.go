package simd_test

import (
	"testing"

	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/simd"
	"github.com/gomlx/compute/support/backendtest"
)

func TestNoSIMD(t *testing.T) {
	defer func() {
		dot.ResetTestRegistrations()
		simd.ForceSmallVariant = false
		simd.ForceLargeVariant = false
	}()
	dot.ResetTestRegistrations()
	simd.RegisterNoSIMDForTests()

	t.Run("Small", func(t *testing.T) {
		simd.ForceSmallVariant = true
		simd.ForceLargeVariant = false
		backendtest.TestDotGeneral(t, backend)
	})

	t.Run("Large", func(t *testing.T) {
		simd.ForceSmallVariant = false
		simd.ForceLargeVariant = true
		backendtest.TestDotGeneral(t, backend)
	})
}

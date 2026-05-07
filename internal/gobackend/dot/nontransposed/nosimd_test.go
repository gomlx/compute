package nontransposed_test

import (
	"testing"

	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/nontransposed"
	"github.com/gomlx/compute/support/backendtest"
)

func TestNoSIMD(t *testing.T) {
	defer func() {
		dot.ResetTestRegistrations()
		nontransposed.ForceSmallVariant = false
		nontransposed.ForceLargeVariant = false
	}()
	dot.ResetTestRegistrations()
	nontransposed.RegisterNoSIMDForTests()

	t.Run("Small", func(t *testing.T) {
		nontransposed.ForceSmallVariant = true
		nontransposed.ForceLargeVariant = false
		backendtest.TestDotGeneral(t, backend)
	})

	t.Run("Large", func(t *testing.T) {
		nontransposed.ForceSmallVariant = false
		nontransposed.ForceLargeVariant = true
		backendtest.TestDotGeneral(t, backend)
	})
}

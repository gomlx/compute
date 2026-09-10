package fusedops

import (
	"os"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
)

// NewBackend returns a "go" backend (*compute/internal/gobackend.Backend) for test.
// It returns an error if the backend is not a "go" backend.
func NewBackend() (*gobackend.Backend, error) {
	backendRaw, err := compute.New()
	if err != nil {
		return nil, err
	}
	be, ok := backendRaw.(*gobackend.Backend)
	if !ok {
		return nil, errors.Errorf("backend configured is not a Go backend: GOMLX_BACKEND=%q", os.Getenv("GOMLX_BACKEND"))
	}
	return be, nil
}

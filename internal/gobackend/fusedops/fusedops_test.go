package fusedops

import (
	"github.com/gomlx/compute/internal/gobackend"
)

// NewBackend returns a "go" backend (*compute/internal/gobackend.Backend) for test.
// It returns an error if the backend is not a "go" backend.
func NewBackend() (*gobackend.Backend, error) {
	return gobackend.NewBackend()
}

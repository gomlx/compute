//go:build amd64 && goexperiment.simd

package defaultpkgs

// Imports only for arch AMD64 SIMD

import (
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx512"
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul/avx512"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx512"
)

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package defaultpkgs imports all the sub-packages that implement the gobackend.
//
// It's just a way to simplify the import and initialization of the gobackend.
package defaultpkgs

import (
	// Operations implementations:
	_ "github.com/gomlx/compute/internal/gobackend/dot"
	_ "github.com/gomlx/compute/internal/gobackend/fusedops"
	_ "github.com/gomlx/compute/internal/gobackend/fusedops/dense"
	_ "github.com/gomlx/compute/internal/gobackend/ops"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/ops/avx512"

	// Activations implementations:
	_ "github.com/gomlx/compute/internal/gobackend/activations"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/activations/avx512"

	// Optimization passes:
	_ "github.com/gomlx/compute/internal/gobackend/passes"

	// DotGeneral implementations:
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul"
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/dot/matmul/avx512"
)

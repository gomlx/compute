// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package fusedops

// This file simply import the AVX2 and AVX512 implementations (protected by a build tag)

import (
	_ "github.com/gomlx/compute/internal/gobackend/fusedops/avx2"
	_ "github.com/gomlx/compute/internal/gobackend/fusedops/avx512"
)

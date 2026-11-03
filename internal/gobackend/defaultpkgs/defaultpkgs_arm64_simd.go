// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package defaultpkgs

// Imports only for arch ARM64 SIMD

import (
	_ "github.com/gomlx/compute/internal/gobackend/ops/neon"
)

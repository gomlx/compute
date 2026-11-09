// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

package dense

import "github.com/gomlx/compute/internal/gobackend/ops/neon"

func init() {
	tallSkinnyAtBFloat32Hook = neon.ExecTallSkinnyAtBFloat32ARM64
	reduceBiasFloat32Hook = neon.ReduceBiasFloat32ARM64
}

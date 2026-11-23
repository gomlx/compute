// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !linux

package gobackend

// UseHugePages is a no-op toggle on non-Linux platforms where MADV_HUGEPAGE is unavailable.
var UseHugePages = false

// AdviseHugePages is a no-op on non-Linux platforms.
func AdviseHugePages(_ []byte) {}

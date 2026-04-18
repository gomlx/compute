// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil

// Must panics if there is an error.
func Must(err error) {
	if err != nil {
		panic(err)
	}
}

// Must1 panics if there is an error and returns the value.
func Must1[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil

import (
	"fmt"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
)

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

// messageFromArgs converts a variadic message and arguments slice to a
// formatted string. This allows the user to write optional messages in tests
// in the same style as t.Errorf.
func printMsgAndArgs(msgAndArgs ...any) string {
	if len(msgAndArgs) == 0 {
		return ""
	}
	format, ok := msgAndArgs[0].(string)
	if !ok {
		return fmt.Sprintf("[[INVALID MSG AND ARGS FORMAT:%v]]", msgAndArgs)
	}
	return fmt.Sprintf(format, msgAndArgs[1:]...)
}

func withinDeltaBase[T ~float32 | ~float64](a, b T, delta float64) bool {
	return math.Abs(float64(a-b)) < delta
}

func checkWithinDelta(a, b any, delta float64) (bool, string) {
	opts := []cmp.Option{
		cmp.Comparer(withinDeltaBase[float32]),
		cmp.Comparer(withinDeltaBase[float64]),
	}
	if cmp.Equal(a, b, opts...) {
		return true, ""
	}
	return false, cmp.Diff(a, b, opts...)
}

// WithinDelta reports a test error if a and b are not equal within a given absolute delta.
func WithinDelta(t *testing.T, a, b any, delta float64, msgAndArgs ...any) {
	t.Helper()
	if isEqual, diff := checkWithinDelta(a, b, delta); !isEqual {
		t.Errorf("%sdelta %g mismatch:\n%s", printMsgAndArgs(msgAndArgs...), delta, diff)
	}
}

// MustWithinDelta reports a fatal test error if a and b are not equal within a given absolute delta.
func MustWithinDelta(t *testing.T, a, b any, delta float64, msgAndArgs ...any) {
	t.Helper()
	if isEqual, diff := checkWithinDelta(a, b, delta); !isEqual {
		t.Fatalf("%sdelta %g mismatch:\n%s", printMsgAndArgs(msgAndArgs...), delta, diff)
	}
}

func withinRelativeDeltaBase[T ~float32 | ~float64](a, b T, relDelta float64) bool {
	delta := math.Abs(float64(a) - float64(b))
	mean := math.Abs(float64(b)) + math.Abs(float64(b))
	return delta/mean < relDelta
}

func checkWithinRelativeDelta(a, b any, relDelta float64) (bool, string) {
	opts := []cmp.Option{
		cmp.Comparer(withinRelativeDeltaBase[float32]),
		cmp.Comparer(withinRelativeDeltaBase[float64]),
	}
	if cmp.Equal(a, b, opts...) {
		return true, ""
	}
	return false, cmp.Diff(a, b, opts...)
}

// WithinRelativeDelta reports a test error if a and b are not equal within a given relative delta.
func WithinRelativeDelta(t *testing.T, a, b any, relDelta float64, msgAndArgs ...any) {
	t.Helper()
	if isEqual, diff := checkWithinRelativeDelta(a, b, relDelta); !isEqual {
		t.Errorf("%srelative delta %g mismatch:\n%s", printMsgAndArgs(msgAndArgs...), relDelta, diff)
	}
}

// MustWithinRelativeDelta reports a fatal test error if a and b are not equal within a given relative delta.
func MustWithinRelativeDelta(t *testing.T, a, b any, relDelta float64, msgAndArgs ...any) {
	t.Helper()
	if isEqual, diff := checkWithinRelativeDelta(a, b, relDelta); !isEqual {
		t.Fatalf("%srelative delta %g mismatch:\n%s", printMsgAndArgs(msgAndArgs...), relDelta, diff)
	}
}

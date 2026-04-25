// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestExecBinary_broadcastIterator(t *testing.T) {
	S := func(dims ...int) shapes.Shape {
		return shapes.Make(dtypes.Float32, dims...)
	}

	// Simple [2, 3] shape broadcast simultaneously by 2 different tensors.
	targetShape := S(2, 3)
	bi1 := newBroadcastIterator(S(2, 1), targetShape)
	bi2 := newBroadcastIterator(S(1, 3), targetShape)
	indices1 := make([]int, 0, targetShape.Size())
	indices2 := make([]int, 0, targetShape.Size())
	for range targetShape.Size() {
		indices1 = append(indices1, bi1.Next())
		indices2 = append(indices2, bi2.Next())
	}
	fmt.Printf("\tindices1=%v\n\tindices2=%v\n", indices1, indices2)
	if ok, diff := testutil.IsEqual([]int{0, 0, 0, 1, 1, 1}, indices1); !ok {
		t.Errorf("indices1 mismatch (-want +got):\n%s", diff)
	}
	if ok, diff := testutil.IsEqual([]int{0, 1, 2, 0, 1, 2}, indices2); !ok {
		t.Errorf("indices2 mismatch (-want +got):\n%s", diff)
	}

	// Alternating broadcast axes.
	targetShape = S(3, 2, 4, 2)
	b3 := newBroadcastIterator(S(3, 1, 4, 1), targetShape)
	indices3 := make([]int, 0, targetShape.Size())
	for range targetShape.Size() {
		indices3 = append(indices3, b3.Next())
	}
	fmt.Printf("\tindices3=%v\n", indices3)
	want3 := []int{
		0, 0, 1, 1, 2, 2, 3, 3,
		0, 0, 1, 1, 2, 2, 3, 3,
		4, 4, 5, 5, 6, 6, 7, 7,
		4, 4, 5, 5, 6, 6, 7, 7,
		8, 8, 9, 9, 10, 10, 11, 11,
		8, 8, 9, 9, 10, 10, 11, 11,
	}
	if ok, diff := testutil.IsEqual(want3, indices3); !ok {
		t.Errorf("indices3 mismatch (-want +got):\n%s", diff)
	}
}

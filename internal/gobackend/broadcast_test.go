// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestBroadcastIterator(t *testing.T) {
	S := func(dims ...int) shapes.Shape {
		return shapes.Make(dtypes.Float32, dims...)
	}

	t.Run("[2,1] -> [2, 3]", func(t *testing.T) {
		targetShape := S(2, 3)
		bi := NewBroadcastIterator(S(2, 1), targetShape)
		srcIndices := make([]int, 0, targetShape.Size())
		for srcIdx := range bi.IterFlatIndices() {
			srcIndices = append(srcIndices, srcIdx)
		}
		if ok, diff := testutil.IsEqual([]int{0, 0, 0, 1, 1, 1}, srcIndices); !ok {
			t.Errorf("srcIndices mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("[1,3] -> [2, 3]", func(t *testing.T) {
		targetShape := S(2, 3)
		bi := NewBroadcastIterator(S(1, 3), targetShape)
		srcIndices := make([]int, 0, targetShape.Size())
		for srcIdx := range bi.IterFlatIndices() {
			srcIndices = append(srcIndices, srcIdx)
		}
		if ok, diff := testutil.IsEqual([]int{0, 1, 2, 0, 1, 2}, srcIndices); !ok {
			t.Errorf("srcIndices mismatch (-want +got):\n%s", diff)
		}
	})

	// Alternating broadcast axes.
	t.Run("[3,1,4,1] -> [3,2,4,2]", func(t *testing.T) {
		targetShape := S(3, 2, 4, 2)
		bi := NewBroadcastIterator(S(3, 1, 4, 1), targetShape)
		srcIndices := make([]int, 0, targetShape.Size())
		for srcIdx := range bi.IterFlatIndices() {
			srcIndices = append(srcIndices, srcIdx)
		}
		want := []int{
			0, 0, 1, 1, 2, 2, 3, 3,
			0, 0, 1, 1, 2, 2, 3, 3,
			4, 4, 5, 5, 6, 6, 7, 7,
			4, 4, 5, 5, 6, 6, 7, 7,
			8, 8, 9, 9, 10, 10, 11, 11,
			8, 8, 9, 9, 10, 10, 11, 11}
		if ok, diff := testutil.IsEqual(want, srcIndices); !ok {
			fmt.Printf("\t- want: %#v\n", want)
			fmt.Printf("\t- got:  %#v\n", srcIndices)
			t.Errorf("srcIndices mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestDetermineBroadcastConfig(t *testing.T) {
	S := func(dims ...int) shapes.Shape {
		return shapes.Make(dtypes.Float32, dims...)
	}

	testCases := []struct {
		name string
		lhs  shapes.Shape
		rhs  shapes.Shape
		tgt  shapes.Shape
		want BroadcastConfig
	}{
		{
			name: "MidRHS_4D_attention_mask",
			lhs:  S(2, 10, 378, 378),
			rhs:  S(2, 1, 378, 378),
			tgt:  S(2, 10, 378, 378),
			want: BroadcastConfig{Pattern: BroadcastMidRHS, A: 2, B: 10, C: 378 * 378},
		},
		{
			name: "MidRHS_4D_rope",
			lhs:  S(2, 405, 10, 64),
			rhs:  S(2, 405, 1, 64),
			tgt:  S(2, 405, 10, 64),
			want: BroadcastConfig{Pattern: BroadcastMidRHS, A: 810, B: 10, C: 64},
		},
		{
			name: "MidLHS_4D",
			lhs:  S(2, 1, 10, 64),
			rhs:  S(2, 4, 10, 64),
			tgt:  S(2, 4, 10, 64),
			want: BroadcastConfig{Pattern: BroadcastMidLHS, A: 2, B: 4, C: 640},
		},
		{
			name: "LeadingRHS_with_leading_1s_on_both",
			lhs:  S(1, 64, 1, 32),
			rhs:  S(1, 1, 1, 32),
			tgt:  S(1, 64, 1, 32),
			want: BroadcastConfig{Pattern: BroadcastLeadingRHS, A: 64, B: 32},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := DetermineBroadcastConfig(tc.lhs, tc.rhs, tc.tgt)
			if !got.EqualNodeData(&tc.want) {
				t.Fatalf("DetermineBroadcastConfig(%s, %s, %s) = %+v, want %+v", tc.lhs, tc.rhs, tc.tgt, got, tc.want)
			}
		})
	}
}

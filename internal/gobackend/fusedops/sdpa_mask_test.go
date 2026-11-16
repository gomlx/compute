package fusedops

import (
	"math"
	"testing"
)

func TestOnlyNegativeInfinityMaskCanTrim(t *testing.T) {
	cases := []struct {
		mask []float32
		want int
	}{
		{[]float32{0, -10000}, 2}, {[]float32{0, -1e20}, 2},
		{[]float32{0, float32(math.Inf(-1))}, 1},
		{[]float32{-10000, -10000}, 2}, {[]float32{0, 0}, 2},
	}
	for _, tc := range cases {
		_, n := sdpaTrimAdditiveKeyMaskFloat(tc.mask, 2)
		if n != tc.want {
			t.Errorf("mask %v: limit=%d want=%d", tc.mask, n, tc.want)
		}
	}
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil

import (
	"math/rand/v2"
	"slices"
	"time"
)

// DefaultReservoirSize is the default capacity for the duration sampler (16K samples).
const DefaultReservoirSize = 16 * 1024

// DurationSampler collects time.Duration observations using Algorithm R (Reservoir Sampling)
// with a fixed capacity to compute percentiles (like median) in constant space and O(1) sample addition.
type DurationSampler struct {
	samples []time.Duration
	count   int
}

// NewDurationSampler creates a sampler with the given maximum reservoir capacity.
// If capacity <= 0, DefaultReservoirSize (16384) is used.
func NewDurationSampler(capacity int) *DurationSampler {
	if capacity <= 0 {
		capacity = DefaultReservoirSize
	}
	return &DurationSampler{
		samples: make([]time.Duration, 0, capacity),
	}
}

// Sample adds a duration observation.
// The first capacity samples are stored sequentially. Subsequent samples are retained
// with probability (capacity / count), replacing a randomly chosen sample in the reservoir.
func (s *DurationSampler) Sample(d time.Duration) {
	s.count++
	capSize := cap(s.samples)
	if len(s.samples) < capSize {
		s.samples = append(s.samples, d)
		return
	}
	// Reservoir sampling: keep sample with probability capSize / count.
	// rand.IntN(s.count) returns in [0, s.count). If j < capSize, replace samples[j].
	j := rand.IntN(s.count)
	if j < capSize {
		s.samples[j] = d
	}
}

// Reset clears the samples and count, retaining the allocated reservoir buffer.
func (s *DurationSampler) Reset() {
	s.samples = s.samples[:0]
	s.count = 0
}

// Count returns the total number of samples observed.
func (s *DurationSampler) Count() int {
	return s.count
}

// ReservoirCount returns the number of samples currently retained in the reservoir.
func (s *DurationSampler) ReservoirCount() int {
	return len(s.samples)
}

// Median returns the median duration of the collected samples.
// If no samples were recorded, it returns 0.
func (s *DurationSampler) Median() time.Duration {
	return s.Percentile(0.5)
}

// Percentile returns the p-th percentile duration (0.0 <= p <= 1.0).
// If no samples were recorded, it returns 0.
func (s *DurationSampler) Percentile(p float64) time.Duration {
	if len(s.samples) == 0 {
		return 0
	}
	sorted := slices.Clone(s.samples)
	slices.Sort(sorted)
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1.0 {
		return sorted[len(sorted)-1]
	}
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}

// Min returns the minimum duration recorded, or 0 if empty.
func (s *DurationSampler) Min() time.Duration {
	return s.Percentile(0.0)
}

// Max returns the maximum duration recorded, or 0 if empty.
func (s *DurationSampler) Max() time.Duration {
	return s.Percentile(1.0)
}

// Mean returns the mean duration of the retained samples, or 0 if empty.
func (s *DurationSampler) Mean() time.Duration {
	if len(s.samples) == 0 {
		return 0
	}
	var sum time.Duration
	for _, d := range s.samples {
		sum += d
	}
	return sum / time.Duration(len(s.samples))
}

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backendtest

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

type rngGoldenCase struct {
	name      string
	algorithm compute.RNGAlgorithm
	initState []uint64
	outShape  shapes.Shape
	wantState []uint64
	wantVals  any // nil when outShape.Size() == 0
}

var rngGoldenCases = []rngGoldenCase{
	// Philox-4x32-10 (bit-exact match with OpenXLA PJRT CPU)
	{
		name:      "Philox/Uint8[6]",
		algorithm: compute.RNGPhilox,
		initState: []uint64{0x0123456789abcdef, 0, 0},
		outShape:  shapes.Make(dtypes.Uint8, 6),
		wantState: []uint64{0x0123456789abcdef, 2, 0},
		wantVals:  []uint8{0x2e, 0x4b, 0x20, 0xf9, 0x66, 0x85},
	},
	{
		name:      "Philox/Uint16[6]",
		algorithm: compute.RNGPhilox,
		initState: []uint64{0x0123456789abcdef, 0, 0},
		outShape:  shapes.Make(dtypes.Uint16, 6),
		wantState: []uint64{0x0123456789abcdef, 2, 0},
		wantVals:  []uint16{0x222e, 0xb04b, 0xa020, 0xfff9, 0x1466, 0x0d85},
	},
	{
		name:      "Philox/Uint32[6]",
		algorithm: compute.RNGPhilox,
		initState: []uint64{0x0123456789abcdef, 0, 0},
		outShape:  shapes.Make(dtypes.Uint32, 6),
		wantState: []uint64{0x0123456789abcdef, 2, 0},
		wantVals:  []uint32{0xb850222e, 0xc58cb04b, 0x14a7a020, 0x7a84fff9, 0xadca1466, 0x523e0d85},
	},
	{
		name:      "Philox/Uint32[3]_Offset1",
		algorithm: compute.RNGPhilox,
		initState: []uint64{0x0123456789abcdef, 1, 0},
		outShape:  shapes.Make(dtypes.Uint32, 3),
		wantState: []uint64{0x0123456789abcdef, 2, 0},
		wantVals:  []uint32{0xadca1466, 0x523e0d85, 0x65401425},
	},
	{
		name:      "Philox/Uint32[4]_CounterCarry",
		algorithm: compute.RNGPhilox,
		initState: []uint64{0x0123456789abcdef, math.MaxUint64, 5},
		outShape:  shapes.Make(dtypes.Uint32, 4),
		wantState: []uint64{0x0123456789abcdef, 0, 6},
		wantVals:  []uint32{0xf02791c6, 0xea2026ee, 0x4fe73934, 0xc174f57d},
	},
	{
		name:      "Philox/Uint64[3]_Offset1",
		algorithm: compute.RNGPhilox,
		initState: []uint64{0x0123456789abcdef, 1, 0},
		outShape:  shapes.Make(dtypes.Uint64, 3),
		wantState: []uint64{0x0123456789abcdef, 3, 0},
		wantVals:  []uint64{0x523e0d85adca1466, 0xb299da3f65401425, 0x823ccb40f7ce299f},
	},
	{
		name:      "Philox/Uint64[4]_CounterCarry",
		algorithm: compute.RNGPhilox,
		initState: []uint64{0x0123456789abcdef, math.MaxUint64, 5},
		outShape:  shapes.Make(dtypes.Uint64, 4),
		wantState: []uint64{0x0123456789abcdef, 1, 6},
		wantVals:  []uint64{0xea2026eef02791c6, 0xc174f57d4fe73934, 0x6ae9fc6315706cf1, 0xde25adea7705da3a},
	},
	{
		name:      "Philox/Scalar",
		algorithm: compute.RNGPhilox,
		initState: []uint64{42, 0, 0},
		outShape:  shapes.Make(dtypes.Uint32),
		wantState: []uint64{42, 1, 0},
		wantVals:  uint32(0x9ceaf053),
	},
	{
		name:      "Philox/Empty",
		algorithm: compute.RNGPhilox,
		initState: []uint64{42, 0, 0},
		outShape:  shapes.Make(dtypes.Uint32, 0),
		wantState: []uint64{42, 0, 0},
		wantVals:  []uint32{},
	},

	// ThreeFry-2x32-20 (bit-exact match with OpenXLA PJRT CPU)
	{
		name:      "ThreeFry/Uint8[6]",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{0x0123456789abcdef, 0},
		outShape:  shapes.Make(dtypes.Uint8, 6),
		wantState: []uint64{0x0123456789abcdef, 3},
		wantVals:  []uint8{0x89, 0xf4, 0x07, 0x13, 0xf1, 0xe4},
	},
	{
		name:      "ThreeFry/Uint16[6]",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{0x0123456789abcdef, 0},
		outShape:  shapes.Make(dtypes.Uint16, 6),
		wantState: []uint64{0x0123456789abcdef, 3},
		wantVals:  []uint16{0xd889, 0xc4f4, 0x2907, 0x8613, 0xe5f1, 0x7fe4},
	},
	{
		name:      "ThreeFry/Uint32[6]",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{0x0123456789abcdef, 0},
		outShape:  shapes.Make(dtypes.Uint32, 6),
		wantState: []uint64{0x0123456789abcdef, 3},
		wantVals:  []uint32{0xaaa4d889, 0x6b35c4f4, 0x75a52907, 0xd3dd8613, 0x35d0e5f1, 0x8ed17fe4},
	},
	{
		name:      "ThreeFry/Uint32[3]_Offset1",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{0x0123456789abcdef, 1},
		outShape:  shapes.Make(dtypes.Uint32, 3),
		wantState: []uint64{0x0123456789abcdef, 3},
		wantVals:  []uint32{0x75a52907, 0xd3dd8613, 0x35d0e5f1},
	},
	{
		name:      "ThreeFry/Uint32[4]_CounterWrap",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{0x0123456789abcdef, math.MaxUint64},
		outShape:  shapes.Make(dtypes.Uint32, 4),
		wantState: []uint64{0x0123456789abcdef, 1},
		wantVals:  []uint32{0x0b3d7494, 0xacf8c4dc, 0xaaa4d889, 0x6b35c4f4},
	},
	{
		name:      "ThreeFry/Uint64[3]_Offset1",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{0x0123456789abcdef, 1},
		outShape:  shapes.Make(dtypes.Uint64, 3),
		wantState: []uint64{0x0123456789abcdef, 4},
		wantVals:  []uint64{0xd3dd861375a52907, 0x8ed17fe435d0e5f1, 0x0284dbe37c2f19d8},
	},
	{
		name:      "ThreeFry/Uint64[4]_CounterWrap",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{0x0123456789abcdef, math.MaxUint64},
		outShape:  shapes.Make(dtypes.Uint64, 4),
		wantState: []uint64{0x0123456789abcdef, 3},
		wantVals:  []uint64{0xacf8c4dc0b3d7494, 0x6b35c4f4aaa4d889, 0xd3dd861375a52907, 0x8ed17fe435d0e5f1},
	},
	{
		name:      "ThreeFry/Scalar",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{42, 0},
		outShape:  shapes.Make(dtypes.Uint32),
		wantState: []uint64{42, 1},
		wantVals:  uint32(0xde79f4b9),
	},
	{
		name:      "ThreeFry/Empty",
		algorithm: compute.RNGThreeFry,
		initState: []uint64{42, 0},
		outShape:  shapes.Make(dtypes.Uint32, 0),
		wantState: []uint64{42, 0},
		wantVals:  []uint32{},
	},

	// SplitMix64 (canonical [key, counter])
	{
		name:      "SplitMix64/Uint64[3]",
		algorithm: compute.RNGSplitMix64,
		initState: []uint64{0, 0},
		outShape:  shapes.Make(dtypes.Uint64, 3),
		wantState: []uint64{0, 3},
		wantVals:  []uint64{0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x06c45d188009454f},
	},
	{
		name:      "SplitMix64/Uint32[3]",
		algorithm: compute.RNGSplitMix64,
		initState: []uint64{0, 0},
		outShape:  shapes.Make(dtypes.Uint32, 3),
		wantState: []uint64{0, 3},
		wantVals:  []uint32{0x7b1dcdaf, 0xa1b965f4, 0x8009454f},
	},

	// PCG-DXSM (math/rand/v2.PCG [hi, lo])
	{
		name:      "PCG/Uint64[3]",
		algorithm: compute.RNGPCG,
		initState: []uint64{42, 99},
		outShape:  shapes.Make(dtypes.Uint64, 3),
		wantState: []uint64{0x89feb1fdcdf75779, 0x44e5880734f17668},
		wantVals:  []uint64{0x64917d8b9f3cadc9, 0x3c4160f9017e6e28, 0xf200df35d64b70a1},
	},

	// ChaCha8 (RFC 8439 8-round counter mode [key0, key1, key2, key3, blockCounter])
	{
		name:      "ChaCha8/Uint32[4]",
		algorithm: compute.RNGChaCha8,
		initState: []uint64{0, 0, 0, 0, 0},
		outShape:  shapes.Make(dtypes.Uint32, 4),
		wantState: []uint64{0, 0, 0, 0, 1},
		wantVals:  []uint32{0x2fef003e, 0xd6405f89, 0xe8b85b7f, 0xa1a5091f},
	},
	{
		name:      "ChaCha8/Uint64[2]",
		algorithm: compute.RNGChaCha8,
		initState: []uint64{0, 0, 0, 0, 0},
		outShape:  shapes.Make(dtypes.Uint64, 2),
		wantState: []uint64{0, 0, 0, 0, 1},
		wantVals:  []uint64{0xd6405f892fef003e, 0xa1a5091fe8b85b7f},
	},
}

func runRNGBitGeneratorTests(t *testing.T, b compute.Backend) {
	testutil.SkipIfMissing(t, b, compute.OpTypeRNGBitGenerator)
	caps := b.Capabilities()

	// 1. Default algorithm tests: statistical uniformity, determinism, state advancement, BitsToUniformFloat.
	t.Run("Default", func(t *testing.T) {
		stateShape, err := caps.RNGStateShape(compute.RNGDefault)
		if err != nil {
			t.Fatalf("Capabilities.RNGStateShape(RNGDefault) failed: %v", err)
		}
		testutil.SkipIfMissingDType(t, b, stateShape.DType)
		initFlat, _, err := caps.RNGStateFromSeed(compute.RNGDefault, 42)
		if err != nil {
			t.Fatalf("Capabilities.RNGStateFromSeed(RNGDefault, 42) failed: %v", err)
		}

		const numSamples = 100000
		const numBins = 10
		const tolerance = 0.1

		for _, dt := range []dtypes.DType{dtypes.Uint32, dtypes.Uint64} {
			if !caps.DTypes[dt] {
				continue
			}
			t.Run(dt.String(), func(t *testing.T) {
				shape := shapes.Make(dt, numSamples)
				newState1, vals1, err := testutil.Exec2(b, []any{initFlat}, func(f compute.Function, params []compute.Value) (compute.Value, compute.Value, error) {
					return f.RNGBitGenerator(params[0], shape, compute.RNGDefault)
				})
				if err != nil {
					t.Fatalf("RNGBitGenerator(RNGDefault, %s) failed: %v", shape, err)
				}

				// Determinism: repeating with the same initial state must yield identical state & values.
				newState2, vals2, err := testutil.Exec2(b, []any{initFlat}, func(f compute.Function, params []compute.Value) (compute.Value, compute.Value, error) {
					return f.RNGBitGenerator(params[0], shape, compute.RNGDefault)
				})
				if err != nil {
					t.Fatalf("second RNGBitGenerator failed: %v", err)
				}
				if !reflect.DeepEqual(newState1, newState2) {
					t.Errorf("RNGDefault not deterministic in newState")
				}
				if !reflect.DeepEqual(vals1, vals2) {
					t.Errorf("RNGDefault not deterministic in values")
				}

				// State advancement: feeding newState1 must produce different values.
				if reflect.DeepEqual(initFlat, newState1) {
					t.Errorf("RNGDefault did not advance state: %v", newState1)
				}
				_, vals3, err := testutil.Exec2(b, []any{newState1}, func(f compute.Function, params []compute.Value) (compute.Value, compute.Value, error) {
					return f.RNGBitGenerator(params[0], shape, compute.RNGDefault)
				})
				if err != nil {
					t.Fatalf("chained RNGBitGenerator failed: %v", err)
				}
				if reflect.DeepEqual(vals1, vals3) {
					t.Errorf("chained RNGBitGenerator produced identical values after state update")
				}

				// Uniformity histogram.
				values := make([]float64, numSamples)
				switch dt {
				case dtypes.Uint32:
					scale := math.Ldexp(1.0, -32)
					for i, v := range vals1.([]uint32) {
						values[i] = float64(v) * scale
					}
				case dtypes.Uint64:
					scale := math.Ldexp(1.0, -53)
					for i, v := range vals1.([]uint64) {
						values[i] = float64(v>>11) * scale
					}
				}
				checkUniformHistogram(t, values, numBins, tolerance)
			})
		}

		// BitsToUniformFloat end-to-end test via RNGBitGenerator(RNGDefault).
		if caps.DTypes[dtypes.Uint32] && caps.DTypes[dtypes.Float32] {
			t.Run("BitsToUniformFloat_Float32", func(t *testing.T) {
				outAny, err := testutil.Exec1(b, []any{initFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					_, bitsVal, err := f.RNGBitGenerator(params[0], shapes.Make(dtypes.Uint32, numSamples), compute.RNGDefault)
					if err != nil {
						return nil, err
					}
					return compute.BitsToUniformFloat(f, bitsVal, dtypes.Float32)
				})
				if err != nil {
					t.Fatalf("BitsToUniformFloat(Float32) failed: %v", err)
				}
				got := outAny.([]float32)
				values := make([]float64, len(got))
				for i, v := range got {
					if math.IsNaN(float64(v)) || v < 0.0 || v >= 1.0 {
						t.Fatalf("BitsToUniformFloat(Float32)[%d] = %g out of [0, 1)", i, v)
					}
					values[i] = float64(v)
				}
				checkUniformHistogram(t, values, numBins, tolerance)
			})
		}

		// Invalid output DType must be rejected.
		if caps.DTypes[dtypes.Float32] {
			t.Run("InvalidOutputDType", func(t *testing.T) {
				_, err := testutil.Exec1(b, []any{initFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					_, vals, err := f.RNGBitGenerator(params[0], shapes.Make(dtypes.Float32, 4), compute.RNGDefault)
					return vals, err
				})
				if err == nil {
					t.Fatalf("expected error for Float32 output shape in RNGBitGenerator, got nil")
				}
			})
		}
	})

	// 2. Explicit algorithms: verify ErrNotImplemented when unsupported, and bit-exact golden vectors when supported.
	for _, algo := range compute.RNGAlgorithmValues() {
		if algo == compute.RNGDefault {
			continue
		}
		t.Run(fmt.Sprintf("Algorithm/%s", algo), func(t *testing.T) {
			if !caps.SupportsRNG(algo) {
				shape := algo.StateShape()
				_, err := testutil.Exec1(b, []any{make([]uint64, shape.Size())}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					_, vals, err := f.RNGBitGenerator(params[0], shapes.Make(dtypes.Uint32, 4), algo)
					return vals, err
				})
				if err == nil {
					t.Fatalf("expected ErrNotImplemented for unsupported algorithm %s, got nil", algo)
				}
				if !compute.IsNotImplemented(err) {
					t.Fatalf("expected ErrNotImplemented for unsupported algorithm %s, got: %v", algo, err)
				}
				t.Skipf("Backend %q does not natively support RNG algorithm %s", b.Name(), algo)
				return
			}

			// Statistical sanity test with StateFromSeed on supported explicit algorithm.
			initFlat, _, err := algo.StateFromSeed(12345)
			if err != nil {
				t.Fatalf("%s.StateFromSeed failed: %v", algo, err)
			}
			const numSamples = 50000
			_, valsAny, err := testutil.Exec2(b, []any{initFlat}, func(f compute.Function, params []compute.Value) (compute.Value, compute.Value, error) {
				return f.RNGBitGenerator(params[0], shapes.Make(dtypes.Uint32, numSamples), algo)
			})
			if err != nil {
				t.Fatalf("RNGBitGenerator(%s, Uint32[%d]) failed: %v", algo, numSamples, err)
			}
			valsF64 := make([]float64, numSamples)
			scale := math.Ldexp(1.0, -32)
			for i, v := range valsAny.([]uint32) {
				valsF64[i] = float64(v) * scale
			}
			checkUniformHistogram(t, valsF64, 10, 0.1)

			// Golden bit-exact checks for this algorithm.
			for _, tc := range rngGoldenCases {
				if tc.algorithm != algo {
					continue
				}
				if !caps.DTypes[tc.outShape.DType] {
					continue
				}
				t.Run(tc.name, func(t *testing.T) {
					gotState, gotVals, err := testutil.Exec2(b, []any{tc.initState}, func(f compute.Function, params []compute.Value) (compute.Value, compute.Value, error) {
						return f.RNGBitGenerator(params[0], tc.outShape, tc.algorithm)
					})
					if err != nil {
						t.Fatalf("RNGBitGenerator(%s) failed: %v", tc.name, err)
					}
					if !reflect.DeepEqual(gotState, tc.wantState) {
						t.Errorf("newState mismatch: got %#v, want %#v", gotState, tc.wantState)
					}
					if tc.outShape.Size() > 0 && !reflect.DeepEqual(gotVals, tc.wantVals) {
						t.Errorf("values mismatch:\n\tgot:  %#v\n\twant: %#v", gotVals, tc.wantVals)
					}
				})
			}
		})
	}
}

func checkUniformHistogram(t *testing.T, values []float64, numBins int, tolerance float64) {
	t.Helper()
	hist := make([]int, numBins)
	for _, v := range values {
		bin := int(v * float64(numBins))
		if bin >= numBins {
			bin = numBins - 1
		}
		if bin < 0 {
			bin = 0
		}
		hist[bin]++
	}
	expectedPerBin := len(values) / numBins
	maxDeviation := float64(expectedPerBin) * tolerance
	for bin, count := range hist {
		deviation := math.Abs(float64(count) - float64(expectedPerBin))
		if deviation > maxDeviation {
			t.Errorf("Bin %d count %d deviates too much from expected %d (deviation: %.2f > %.2f)",
				bin, count, expectedPerBin, deviation, maxDeviation)
		}
	}
}

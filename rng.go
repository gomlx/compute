// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package compute

import (
	"encoding/binary"
	"math"
	"math/rand/v2"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// RNGAlgorithm specifies the Pseudo-Random Number Generator (PRNG) algorithm
// used by [StandardOps.RNGBitGenerator].
//
// An RNG (Random Number Generator) in compute graphs is implemented as a pure,
// deterministic PRNG (Pseudo-Random Number Generator): given an explicit input
// state tensor, it produces both an updated state tensor and a tensor of
// uniformly distributed random bits.
type RNGAlgorithm int

//go:generate go tool enumer -type=RNGAlgorithm -trimprefix=RNG -output=gen_rngalgorithm_enumer.go -transform=snake rng.go

const (
	// RNGDefault (the zero value) instructs the backend to use its preferred native
	// Random Number Generator (RNG) algorithm.
	//
	// Guarantees:
	//   - Fast and native to the target backend.
	//   - Deterministic across runs on the same backend given the same initial state and shapes.
	//   - Not guaranteed to produce identical bitstreams across different backends (unless two
	//     backends resolve RNGDefault to the same Capabilities.DefaultRNGAlgorithm).
	//
	// State shape: queried via backend.Capabilities().RNGStateShape(RNGDefault).
	// Native backend support: all backends supporting OpTypeRNGBitGenerator ("go", "xla", "onnx", "mlx", "mps").
	RNGDefault RNGAlgorithm = iota

	// RNGPhilox specifies the counter-based Philox-4x32-10 Pseudo-Random Number Generator
	// (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011,
	// https://dl.acm.org/doi/10.1145/2063384.2063405), matching OpenXLA's PHILOX specification.
	//
	// Uses 10 rounds of 32-bit multiplication (mulhi/mullo) and XOR mixing over a 64-bit key
	// and 128-bit counter. Parallelizable across CPU workers/SIMD, GPUs, and TPUs.
	//
	// Canonical state shape: [3]uint64 ([key, counter_lo, counter_hi]).
	// Native backend support: "xla", "go" (plus automatic graph decomposition fallback in GoMLX).
	RNGPhilox

	// RNGThreeFry specifies the counter-based ThreeFry-2x32-20 Pseudo-Random Number Generator
	// (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011,
	// https://dl.acm.org/doi/10.1145/2063384.2063405), matching OpenXLA's THREE_FRY specification.
	//
	// Uses 20 rounds of pure 32-bit additions, bitwise rotations, and XORs (requiring no
	// integer multiplications), making it fast and portable across CPUs, GPUs, and TPUs.
	//
	// Canonical state shape: [2]uint64 ([key, counter]).
	// Native backend support: "xla", "go", "onnx", "mps" (plus automatic graph decomposition fallback in GoMLX).
	RNGThreeFry

	// RNGSplitMix64 specifies a counter-based 64-bit SplitMix64 Pseudo-Random Number Generator
	// (Steele, Lea, and Flood, OOPSLA 2014).
	//
	// Uses two 64-bit shift-XOR-multiply mixing steps per output word, making it very compact
	// and fast when cryptographic strength is not required.
	//
	// Canonical state shape: [2]uint64 ([key, counter]).
	// Native backend support: "go", "onnx" (plus automatic graph decomposition fallback in GoMLX
	// on backends supporting Uint64).
	RNGSplitMix64

	// RNGPCG specifies the 128-bit PCG-DXSM (Permuted Congruential Generator — Double Xorshift
	// Multiply) sequential Pseudo-Random Number Generator matching Go's standard library
	// math/rand/v2.PCG.
	//
	// Note: PCG is a sequential state-transition generator rather than a counter-based PRNG,
	// so it is intended for CPU execution and is not decomposed into parallel GPU graph ops.
	//
	// Canonical state shape: [2]uint64 ([hi, lo]).
	// Native backend support: "go".
	RNGPCG

	// RNGChaCha8 specifies the counter-mode 8-round ChaCha8 stream cipher
	// Cryptographically Secure Pseudo-Random Number Generator (CSPRNG) (Bernstein, 2008).
	//
	// Generates 64-byte (16 x uint32) blocks independently from a 256-bit key and 64-bit block
	// counter, providing cryptographic-grade randomness while remaining fully parallelizable
	// across CPU threads/SIMD and GPUs.
	//
	// Canonical state shape: [5]uint64 ([key0, key1, key2, key3, blockCounter]).
	// Native backend support: "go" (plus automatic graph decomposition fallback in GoMLX).
	RNGChaCha8
)

// StateShape returns the canonical state tensor shape for an explicit RNGAlgorithm:
//   - [RNGPhilox]:     [3]uint64
//   - [RNGThreeFry]:   [2]uint64
//   - [RNGSplitMix64]: [2]uint64
//   - [RNGPCG]:        [2]uint64
//   - [RNGChaCha8]:    [5]uint64
//   - [RNGDefault]:    [shapes.Invalid] (backend-dependent; use [Capabilities.RNGStateShape])
func (a RNGAlgorithm) StateShape() shapes.Shape {
	switch a {
	case RNGPhilox:
		return shapes.Make(dtypes.Uint64, 3)
	case RNGThreeFry, RNGSplitMix64, RNGPCG:
		return shapes.Make(dtypes.Uint64, 2)
	case RNGChaCha8:
		return shapes.Make(dtypes.Uint64, 5)
	default:
		return shapes.Invalid()
	}
}

// IsCounterBased reports whether a is a parallel counter-based PRNG algorithm whose
// output blocks depend only on (key, counter+blockIndex).
func (a RNGAlgorithm) IsCounterBased() bool {
	switch a {
	case RNGPhilox, RNGThreeFry, RNGSplitMix64, RNGChaCha8:
		return true
	default:
		return false
	}
}

// splitMix64Step advances a 64-bit SplitMix64 state and returns the mixed output word.
func splitMix64Step(state *uint64) uint64 {
	*state += 0x9e3779b97f4a7c15
	z := *state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// StateFromSeed deterministically initializes a flat []uint64 slice and canonical state Shape
// for an explicit RNGAlgorithm from a 64-bit seed (expanding key words via SplitMix64 and
// setting counter words to 0; for [RNGPCG], seeding hi/lo identically to [rand.NewPCG]).
//
// Returns an error if called on [RNGDefault]; use [Capabilities.RNGStateFromSeed] instead.
func (a RNGAlgorithm) StateFromSeed(seed int64) (flat any, shape shapes.Shape, err error) {
	if a == RNGDefault {
		return nil, shapes.Invalid(), errors.New("RNGAlgorithm.StateFromSeed: RNGDefault state shape is backend-dependent; use backend.Capabilities().RNGStateFromSeed(RNGDefault, seed)")
	}
	shape = a.StateShape()
	if !shape.Ok() {
		return nil, shapes.Invalid(), errors.Errorf("RNGAlgorithm.StateFromSeed: unsupported algorithm %v", a)
	}
	s := uint64(seed)
	switch a {
	case RNGPhilox:
		return []uint64{splitMix64Step(&s), 0, 0}, shape, nil
	case RNGThreeFry, RNGSplitMix64:
		return []uint64{splitMix64Step(&s), 0}, shape, nil
	case RNGPCG:
		s1 := splitMix64Step(&s)
		s2 := splitMix64Step(&s)
		rng := rand.NewPCG(s1, s2)
		encoded, err := rng.MarshalBinary()
		if err != nil {
			return nil, shapes.Invalid(), errors.Wrap(err, "RNGPCG.StateFromSeed: MarshalBinary failed")
		}
		if len(encoded) != 20 || string(encoded[:4]) != "pcg:" {
			return nil, shapes.Invalid(), errors.Errorf("RNGPCG.StateFromSeed: unexpected PCG binary encoding length %d", len(encoded))
		}
		hi := binary.BigEndian.Uint64(encoded[4:12])
		lo := binary.BigEndian.Uint64(encoded[12:20])
		return []uint64{hi, lo}, shape, nil
	case RNGChaCha8:
		return []uint64{
			splitMix64Step(&s),
			splitMix64Step(&s),
			splitMix64Step(&s),
			splitMix64Step(&s),
			0,
		}, shape, nil
	default:
		return nil, shapes.Invalid(), errors.Errorf("RNGAlgorithm.StateFromSeed: unsupported algorithm %v", a)
	}
}

// BitsToUniformFloat builds standard [Function] operations (Constant, ShiftRightLogical,
// ConvertDType, Mul) to convert uniformly distributed unsigned integer bits (Uint8, Uint16,
// Uint32, or Uint64) into uniformly distributed floating-point values in the half-open interval
// [0.0, 1.0) of floatDType ([dtypes.BFloat16], [dtypes.Float16], [dtypes.Float32], or [dtypes.Float64]).
//
// It preserves exact IEEE-754 significand precision p by right-shifting (bits >> (W - p)) before
// converting to floatDType and scaling by 2^-p, guaranteeing every output is strictly < 1.0 without
// rounding artifacts or clamping.
//
// Required minimum bit width W of randomBits:
//   - [dtypes.Float64] (p=53): Uint64
//   - [dtypes.Float32] (p=24): Uint32 or Uint64
//   - [dtypes.Float16] (p=11): Uint16, Uint32, or Uint64
//   - [dtypes.BFloat16] (p=8): Uint8, Uint16, Uint32, or Uint64
func BitsToUniformFloat(fn Function, randomBits Value, floatDType dtypes.DType) (Value, error) {
	if fn == nil || randomBits == nil {
		return nil, errors.New("BitsToUniformFloat: fn and randomBits must not be nil")
	}
	bitsShape, err := fn.Shape(randomBits)
	if err != nil {
		return nil, errors.Wrap(err, "BitsToUniformFloat: getting randomBits shape")
	}
	if !bitsShape.DType.IsUnsigned() {
		return nil, errors.Errorf("BitsToUniformFloat: randomBits must have an unsigned integer dtype (Uint8/Uint16/Uint32/Uint64), got %s", bitsShape.DType)
	}
	bitsWidth := bitsShape.DType.Bits()
	if bitsWidth < 8 {
		return nil, errors.Errorf("BitsToUniformFloat: unsupported sub-byte randomBits dtype %s", bitsShape.DType)
	}

	var sigBits int
	var scaleConst any
	switch floatDType {
	case dtypes.Float64:
		sigBits = 53
		scaleConst = []float64{math.Ldexp(1.0, -53)}
	case dtypes.Float32:
		sigBits = 24
		scaleConst = []float32{float32(math.Ldexp(1.0, -24))}
	case dtypes.Float16:
		sigBits = 11
		scaleConst = []float16.Float16{float16.FromFloat32(float32(math.Ldexp(1.0, -11)))}
	case dtypes.BFloat16:
		sigBits = 8
		scaleConst = []bfloat16.BFloat16{bfloat16.FromFloat32(float32(math.Ldexp(1.0, -8)))}
	default:
		return nil, errors.Errorf("BitsToUniformFloat: floatDType must be BFloat16, Float16, Float32, or Float64, got %s", floatDType)
	}

	if bitsWidth < sigBits {
		return nil, errors.Errorf("BitsToUniformFloat: randomBits dtype %s (%d bits) has fewer bits than %s significand (%d bits)",
			bitsShape.DType, bitsWidth, floatDType, sigBits)
	}

	shifted := randomBits
	if shift := bitsWidth - sigBits; shift > 0 {
		var shiftFlat any
		switch bitsShape.DType {
		case dtypes.Uint64:
			shiftFlat = []uint64{uint64(shift)}
		case dtypes.Uint32:
			shiftFlat = []uint32{uint32(shift)}
		case dtypes.Uint16:
			shiftFlat = []uint16{uint16(shift)}
		case dtypes.Uint8:
			shiftFlat = []uint8{uint8(shift)}
		default:
			return nil, errors.Errorf("BitsToUniformFloat: unsupported unsigned dtype %s", bitsShape.DType)
		}
		shiftVal, err := fn.Constant(shiftFlat)
		if err != nil {
			return nil, errors.Wrap(err, "BitsToUniformFloat: creating shift constant")
		}
		shifted, err = fn.ShiftRightLogical(randomBits, shiftVal)
		if err != nil {
			return nil, errors.Wrap(err, "BitsToUniformFloat: ShiftRightLogical")
		}
	}

	asFloat, err := fn.ConvertDType(shifted, floatDType)
	if err != nil {
		return nil, errors.Wrap(err, "BitsToUniformFloat: ConvertDType")
	}
	scaleVal, err := fn.Constant(scaleConst)
	if err != nil {
		return nil, errors.Wrap(err, "BitsToUniformFloat: creating scale constant")
	}
	res, err := fn.Mul(asFloat, scaleVal)
	if err != nil {
		return nil, errors.Wrap(err, "BitsToUniformFloat: Mul")
	}
	return res, nil
}

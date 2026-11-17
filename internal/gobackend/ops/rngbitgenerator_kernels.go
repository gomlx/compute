// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"math/bits"
	"math/rand/v2"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func runRNGKernel(
	backend *gobackend.Backend,
	algorithm compute.RNGAlgorithm,
	targetShape shapes.Shape,
	inState, outState []uint64,
	outBuf *gobackend.Buffer,
) error {
	switch algorithm {
	case compute.RNGPhilox:
		return execRNGPhilox(backend, targetShape, inState, outState, outBuf)
	case compute.RNGThreeFry:
		return execRNGThreeFry(backend, targetShape, inState, outState, outBuf)
	case compute.RNGSplitMix64:
		return execRNGSplitMix64(backend, targetShape, inState, outState, outBuf)
	case compute.RNGPCG:
		return execRNGPCG(targetShape, inState, outState, outBuf)
	case compute.RNGChaCha8:
		return execRNGChaCha8(backend, targetShape, inState, outState, outBuf)
	default:
		return errors.Wrapf(compute.ErrNotImplemented, "execRNGBitGenerator: unsupported RNG algorithm %s", algorithm)
	}
}

func parallelizeRNGBlocks(backend *gobackend.Backend, numBlocks, minBlocksPerChunk int, fn func(startBlock, endBlock int)) {
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && numBlocks > minBlocksPerChunk {
		backend.Scheduler.ParallelizeChunks(numBlocks, minBlocksPerChunk, fn)
		return
	}
	fn(0, numBlocks)
}

// write32BitWordBlock writes up to blockWords generated uint32 words from block b into outBuf,
// starting at element offset b * elemsPerBlock.
func write32BitWordBlock(targetShape shapes.Shape, outBuf *gobackend.Buffer, b, blockWords int, words []uint32) {
	numElems := targetShape.Size()
	switch targetShape.DType {
	case dtypes.Uint64:
		out := outBuf.Flat.([]uint64)
		elemsPerBlock := blockWords / 2
		start := b * elemsPerBlock
		limit := min(elemsPerBlock, numElems-start)
		for j := range limit {
			out[start+j] = uint64(words[2*j]) | (uint64(words[2*j+1]) << 32)
		}
	case dtypes.Uint32:
		out := outBuf.Flat.([]uint32)
		start := b * blockWords
		limit := min(blockWords, numElems-start)
		copy(out[start:start+limit], words[:limit])
	case dtypes.Uint16:
		out := outBuf.Flat.([]uint16)
		start := b * blockWords
		limit := min(blockWords, numElems-start)
		for j := range limit {
			out[start+j] = uint16(words[j])
		}
	case dtypes.Uint8:
		out := outBuf.Flat.([]uint8)
		start := b * blockWords
		limit := min(blockWords, numElems-start)
		for j := range limit {
			out[start+j] = uint8(words[j])
		}
	}
}

// addUint128 adds delta to a little-endian 128-bit counter (lo, hi).
func addUint128(lo, hi, delta uint64) (newLo, newHi uint64) {
	var carry uint64
	newLo, carry = bits.Add64(lo, delta, 0)
	newHi = hi + carry
	return newLo, newHi
}

func execRNGPhilox(backend *gobackend.Backend, targetShape shapes.Shape, inState, outState []uint64, outBuf *gobackend.Buffer) error {
	const (
		w0 uint32 = 0x9E3779B9
		w1 uint32 = 0xBB67AE85
		m0 uint64 = 0xD2511F53
		m1 uint64 = 0xCD9E8D57
	)
	numElems := targetShape.Size()
	wordsPerElem := 1
	if targetShape.DType == dtypes.Uint64 {
		wordsPerElem = 2
	}
	numWords := numElems * wordsPerElem
	numBlocks := (numWords + 3) / 4

	key0 := uint32(inState[0])
	key1 := uint32(inState[0] >> 32)
	baseLo := inState[1]
	baseHi := inState[2]

	parallelizeRNGBlocks(backend, numBlocks, 256, func(startBlock, endBlock int) {
		var words [4]uint32
		for b := startBlock; b < endBlock; b++ {
			ctrLo, ctrHi := addUint128(baseLo, baseHi, uint64(b))
			c0 := uint32(ctrLo)
			c1 := uint32(ctrLo >> 32)
			c2 := uint32(ctrHi)
			c3 := uint32(ctrHi >> 32)
			k0, k1 := key0, key1
			for range 10 {
				p0 := m0 * uint64(c0)
				p1 := m1 * uint64(c2)
				lo0, hi0 := uint32(p0), uint32(p0>>32)
				lo1, hi1 := uint32(p1), uint32(p1>>32)
				c0 = hi1 ^ c1 ^ k0
				c1 = lo1
				c2 = hi0 ^ c3 ^ k1
				c3 = lo0
				k0 += w0
				k1 += w1
			}
			words[0], words[1], words[2], words[3] = c0, c1, c2, c3
			write32BitWordBlock(targetShape, outBuf, b, 4, words[:])
		}
	})

	outState[0] = inState[0]
	outState[1], outState[2] = addUint128(baseLo, baseHi, uint64(numBlocks))
	return nil
}

func execRNGThreeFry(backend *gobackend.Backend, targetShape shapes.Shape, inState, outState []uint64, outBuf *gobackend.Buffer) error {
	rotations := [8]int{13, 15, 26, 6, 17, 29, 16, 24}

	numElems := targetShape.Size()
	wordsPerElem := 1
	if targetShape.DType == dtypes.Uint64 {
		wordsPerElem = 2
	}
	numWords := numElems * wordsPerElem
	numBlocks := (numWords + 1) / 2

	key0 := uint32(inState[0])
	key1 := uint32(inState[0] >> 32)
	key2 := key0 ^ key1 ^ 0x1BD11BDA
	ks := [3]uint32{key0, key1, key2}
	baseCtr := inState[1]

	parallelizeRNGBlocks(backend, numBlocks, 256, func(startBlock, endBlock int) {
		var words [2]uint32
		for b := startBlock; b < endBlock; b++ {
			ctr := baseCtr + uint64(b)
			x0 := uint32(ctr) + ks[0]
			x1 := uint32(ctr>>32) + ks[1]
			for r := range 20 {
				x0 += x1
				x1 = bits.RotateLeft32(x1, rotations[r%8])
				x1 ^= x0
				if (r+1)%4 == 0 {
					inj := (r + 1) / 4
					x0 += ks[inj%3]
					x1 += ks[(inj+1)%3] + uint32(inj)
				}
			}
			words[0], words[1] = x0, x1
			write32BitWordBlock(targetShape, outBuf, b, 2, words[:])
		}
	})

	outState[0] = inState[0]
	outState[1] = baseCtr + uint64(numBlocks)
	return nil
}

func execRNGSplitMix64(backend *gobackend.Backend, targetShape shapes.Shape, inState, outState []uint64, outBuf *gobackend.Buffer) error {
	const (
		gamma uint64 = 0x9e3779b97f4a7c15
		c1    uint64 = 0xbf58476d1ce4e5b9
		c2    uint64 = 0x94d049bb133111eb
	)
	numElems := targetShape.Size()
	key := inState[0]
	baseCtr := inState[1]

	parallelizeRNGBlocks(backend, numElems, 1024, func(startElem, endElem int) {
		switch targetShape.DType {
		case dtypes.Uint64:
			out := outBuf.Flat.([]uint64)
			for i := startElem; i < endElem; i++ {
				z := key + (baseCtr+uint64(i)+1)*gamma
				z = (z ^ (z >> 30)) * c1
				z = (z ^ (z >> 27)) * c2
				out[i] = z ^ (z >> 31)
			}
		case dtypes.Uint32:
			out := outBuf.Flat.([]uint32)
			for i := startElem; i < endElem; i++ {
				z := key + (baseCtr+uint64(i)+1)*gamma
				z = (z ^ (z >> 30)) * c1
				z = (z ^ (z >> 27)) * c2
				out[i] = uint32(z ^ (z >> 31))
			}
		case dtypes.Uint16:
			out := outBuf.Flat.([]uint16)
			for i := startElem; i < endElem; i++ {
				z := key + (baseCtr+uint64(i)+1)*gamma
				z = (z ^ (z >> 30)) * c1
				z = (z ^ (z >> 27)) * c2
				out[i] = uint16(z ^ (z >> 31))
			}
		case dtypes.Uint8:
			out := outBuf.Flat.([]uint8)
			for i := startElem; i < endElem; i++ {
				z := key + (baseCtr+uint64(i)+1)*gamma
				z = (z ^ (z >> 30)) * c1
				z = (z ^ (z >> 27)) * c2
				out[i] = uint8(z ^ (z >> 31))
			}
		}
	})

	outState[0] = key
	outState[1] = baseCtr + uint64(numElems)
	return nil
}

func execRNGPCG(targetShape shapes.Shape, inState, outState []uint64, outBuf *gobackend.Buffer) error {
	var buf [20]byte
	copy(buf[:4], "pcg:")
	binary.BigEndian.PutUint64(buf[4:12], inState[0])
	binary.BigEndian.PutUint64(buf[12:20], inState[1])
	var rng rand.PCG
	if err := rng.UnmarshalBinary(buf[:]); err != nil {
		return errors.Wrap(err, "execRNGPCG: UnmarshalBinary failed")
	}

	numElems := targetShape.Size()
	switch targetShape.DType {
	case dtypes.Uint64:
		out := outBuf.Flat.([]uint64)
		for i := range numElems {
			out[i] = rng.Uint64()
		}
	case dtypes.Uint32:
		out := outBuf.Flat.([]uint32)
		for i := range numElems {
			out[i] = uint32(rng.Uint64())
		}
	case dtypes.Uint16:
		out := outBuf.Flat.([]uint16)
		for i := range numElems {
			out[i] = uint16(rng.Uint64())
		}
	case dtypes.Uint8:
		out := outBuf.Flat.([]uint8)
		for i := range numElems {
			out[i] = uint8(rng.Uint64())
		}
	}

	encoded, err := rng.MarshalBinary()
	if err != nil {
		return errors.Wrap(err, "execRNGPCG: MarshalBinary failed")
	}
	if len(encoded) != 20 || string(encoded[:4]) != "pcg:" {
		return errors.Errorf("execRNGPCG: unexpected PCG binary encoding length %d", len(encoded))
	}
	outState[0] = binary.BigEndian.Uint64(encoded[4:12])
	outState[1] = binary.BigEndian.Uint64(encoded[12:20])
	return nil
}

func chachaQuarterRound(s *[16]uint32, a, b, c, d int) {
	s[a] += s[b]
	s[d] = bits.RotateLeft32(s[d]^s[a], 16)
	s[c] += s[d]
	s[b] = bits.RotateLeft32(s[b]^s[c], 12)
	s[a] += s[b]
	s[d] = bits.RotateLeft32(s[d]^s[a], 8)
	s[c] += s[d]
	s[b] = bits.RotateLeft32(s[b]^s[c], 7)
}

func execRNGChaCha8(backend *gobackend.Backend, targetShape shapes.Shape, inState, outState []uint64, outBuf *gobackend.Buffer) error {
	numElems := targetShape.Size()
	wordsPerElem := 1
	if targetShape.DType == dtypes.Uint64 {
		wordsPerElem = 2
	}
	numWords := numElems * wordsPerElem
	numBlocks := (numWords + 15) / 16

	var baseState [16]uint32
	baseState[0] = 0x61707865
	baseState[1] = 0x3320646e
	baseState[2] = 0x79622d32
	baseState[3] = 0x6b206574
	for i := range 4 {
		baseState[4+2*i] = uint32(inState[i])
		baseState[4+2*i+1] = uint32(inState[i] >> 32)
	}
	baseCtr := inState[4]

	parallelizeRNGBlocks(backend, numBlocks, 64, func(startBlock, endBlock int) {
		var initial, work [16]uint32
		initial = baseState
		for b := startBlock; b < endBlock; b++ {
			ctr := baseCtr + uint64(b)
			initial[12] = uint32(ctr)
			initial[13] = uint32(ctr >> 32)
			work = initial
			for range 4 {
				// Column rounds
				chachaQuarterRound(&work, 0, 4, 8, 12)
				chachaQuarterRound(&work, 1, 5, 9, 13)
				chachaQuarterRound(&work, 2, 6, 10, 14)
				chachaQuarterRound(&work, 3, 7, 11, 15)
				// Diagonal rounds
				chachaQuarterRound(&work, 0, 5, 10, 15)
				chachaQuarterRound(&work, 1, 6, 11, 12)
				chachaQuarterRound(&work, 2, 7, 8, 13)
				chachaQuarterRound(&work, 3, 4, 9, 14)
			}
			for i := range 16 {
				work[i] += initial[i]
			}
			write32BitWordBlock(targetShape, outBuf, b, 16, work[:])
		}
	})

	copy(outState[:4], inState[:4])
	outState[4] = baseCtr + uint64(numBlocks)
	return nil
}

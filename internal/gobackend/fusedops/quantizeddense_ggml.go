// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"encoding/binary"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

var iq4nlIntTable = [16]int8{
	-127, -104, -83, -65, -49, -35, -22, -10,
	1, 13, 25, 38, 53, 69, 89, 113,
}

func execQuantizedDenseGGMLCore(
	backend *gobackend.Backend,
	data *nodeFusedQuantizedDense,
	in unpackedQDInputs,
	M, K, N int,
	outF32 []float32,
) error {
	weights := in.weight.Flat.([]uint8)
	bytesPerRowW := in.weight.RawShape.Dimensions[1]
	wType := data.weightGGMLType

	var biasF32 []float32
	if in.bias != nil {
		var releaseBias func()
		var err error
		biasF32, releaseBias, err = bufferToFloat32Slice(backend, in.bias)
		if err != nil {
			return err
		}
		defer releaseBias()
	}

	// Mode B1: Pre-quantized GGML activations (Q8_0 or Q8_K).
	if data.hasXQuant {
		xBytes := in.x.Flat.([]uint8)
		bytesPerRowX := in.x.RawShape.Dimensions[in.x.RawShape.Rank()-1]
		return quantizedDenseGGMLBlockInt(backend, xBytes, bytesPerRowX, weights, bytesPerRowW, biasF32, outF32, M, K, N, wType)
	}

	// Float x input.
	xF32, releaseX, err := bufferToFloat32Slice(backend, in.x)
	if err != nil {
		return err
	}
	defer releaseX()

	// Mode B2: On-the-fly dynamic Q8_0 / Q8_K quantization when QuantizedMatmuls == true.
	if data.quantizedMatmuls {
		var xType compute.GGMLQuantType
		if wType.ValuesPerBlock() == 32 {
			xType = compute.GGMLQ8_0
		} else {
			xType = compute.GGMLQ8_K
		}
		bytesPerRowX := (K / xType.ValuesPerBlock()) * xType.BytesPerBlock()
		tmpXBuf, err := backend.GetBuffer(shapes.Make(dtypes.Uint8, M, bytesPerRowX))
		if err != nil {
			return err
		}
		defer backend.PutBuffer(tmpXBuf)
		xBytes := tmpXBuf.Flat.([]uint8)
		quantizeGGMLRows(backend, xF32, xBytes, M, K, bytesPerRowX, xType)
		return quantizedDenseGGMLBlockInt(backend, xBytes, bytesPerRowX, weights, bytesPerRowW, biasF32, outF32, M, K, N, wType)
	}

	// Mode A: Weight-only float dequant + dot product.
	return quantizedDenseGGMLFloat(backend, xF32, weights, biasF32, outF32, M, K, N, bytesPerRowW, wType)
}

func quantizedDenseGGMLFloat(
	backend *gobackend.Backend,
	x []float32,
	weights []uint8,
	bias, out []float32,
	M, K, N, bytesPerRow int,
	ggmlType compute.GGMLQuantType,
) error {
	dequantFn, err := ggmlDequantFunc(ggmlType)
	if err != nil {
		return err
	}

	numWorkers := quantizedDenseParallelTileCount(backend, M, K, N)
	scratchBufs := make([][]float32, numWorkers)
	for i := range scratchBufs {
		scratchBufs[i] = make([]float32, K)
	}

	quantizedDenseParallel(backend, M, K, N, func(workerIdx, m, nStart, nEnd int) {
		dequantRow := scratchBufs[workerIdx]
		xRow := x[m*K : (m+1)*K]
		for n := nStart; n < nEnd; n++ {
			rowData := weights[n*bytesPerRow : (n+1)*bytesPerRow]
			dequantFn(rowData, dequantRow)
			var dot float32
			for k := range K {
				dot += xRow[k] * dequantRow[k]
			}
			if bias != nil {
				dot += bias[n]
			}
			out[m*N+n] = dot
		}
	})
	return nil
}

func quantizedDenseGGMLBlockInt(
	backend *gobackend.Backend,
	xBytes []uint8,
	bytesPerRowX int,
	weights []uint8,
	bytesPerRowW int,
	bias, out []float32,
	M, K, N int,
	wType compute.GGMLQuantType,
) error {
	dotFn, err := ggmlVecDotFunc(wType)
	if err != nil {
		return err
	}

	quantizedDenseParallel(backend, M, K, N, func(_, m, nStart, nEnd int) {
		xRow := xBytes[m*bytesPerRowX : (m+1)*bytesPerRowX]
		for n := nStart; n < nEnd; n++ {
			wRow := weights[n*bytesPerRowW : (n+1)*bytesPerRowW]
			dot := dotFn(wRow, xRow)
			if bias != nil {
				dot += bias[n]
			}
			out[m*N+n] = dot
		}
	})
	return nil
}

func quantizeGGMLRows(backend *gobackend.Backend, x []float32, out []uint8, M, K, bytesPerRow int, qType compute.GGMLQuantType) {
	rowFn := quantizeRowQ8_0
	if qType == compute.GGMLQ8_K {
		rowFn = quantizeRowQ8_K
	}
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && M*K >= quantizedDenseMinParallelizeChunk && M > 1 {
		backend.Scheduler.ParallelizeChunks(M, 1, func(mStart, mEnd int) {
			for m := mStart; m < mEnd; m++ {
				rowFn(x[m*K:(m+1)*K], out[m*bytesPerRow:(m+1)*bytesPerRow])
			}
		})
		return
	}
	for m := range M {
		rowFn(x[m*K:(m+1)*K], out[m*bytesPerRow:(m+1)*bytesPerRow])
	}
}

// quantizeRowQ8_0 quantizes a float32 slice (multiple of 32) into GGML Q8_0 blocks (34 bytes per 32 values).
func quantizeRowQ8_0(src []float32, dst []uint8) {
	const qk = 32
	const blockBytes = 34
	nblocks := len(src) / qk
	for b := range nblocks {
		xBlock := src[b*qk : (b+1)*qk]
		outBlock := dst[b*blockBytes : (b+1)*blockBytes]
		var maxAbs float32
		for _, v := range xBlock {
			av := float32(math.Abs(float64(v)))
			if av > maxAbs {
				maxAbs = av
			}
		}
		if maxAbs == 0 {
			clear(outBlock)
			continue
		}
		dF16 := float16.FromFloat32(maxAbs / 127.0)
		dBits := dF16.Bits()
		outBlock[0] = uint8(dBits)
		outBlock[1] = uint8(dBits >> 8)
		dActual := dF16.Float32()
		if dActual == 0 {
			clear(outBlock[2:])
			continue
		}
		invD := 1.0 / dActual
		qs := outBlock[2:]
		for i, v := range xBlock {
			q := int32(math.RoundToEven(float64(v * invD)))
			q = max(-127, min(127, q))
			qs[i] = uint8(int8(q))
		}
	}
}

// quantizeRowQ8_K quantizes a float32 slice (multiple of 256) into GGML Q8_K blocks (292 bytes per 256 values).
// Block layout: float32 d (4) + int8 qs[256] (256) + int16 bsums[16] (32).
func quantizeRowQ8_K(src []float32, dst []uint8) {
	const qk = 256
	const blockBytes = 292
	nblocks := len(src) / qk
	for b := range nblocks {
		xBlock := src[b*qk : (b+1)*qk]
		outBlock := dst[b*blockBytes : (b+1)*blockBytes]
		var maxAbs float32
		for _, v := range xBlock {
			av := float32(math.Abs(float64(v)))
			if av > maxAbs {
				maxAbs = av
			}
		}
		if maxAbs == 0 {
			clear(outBlock)
			continue
		}
		d := maxAbs / 127.0
		invD := 127.0 / maxAbs
		binary.LittleEndian.PutUint32(outBlock[0:4], math.Float32bits(d))
		qs := outBlock[4:260]
		bsums := outBlock[260:292]
		for g := range 16 {
			var bsum int16
			base := g * 16
			for l := range 16 {
				q := int32(math.RoundToEven(float64(xBlock[base+l] * invD)))
				q = max(-127, min(127, q))
				qi := int8(q)
				qs[base+l] = uint8(qi)
				bsum += int16(qi)
			}
			binary.LittleEndian.PutUint16(bsums[2*g:2*g+2], uint16(bsum))
		}
	}
}

// ggmlDequantFunc returns the float32 row dequantizer for any supported GGMLQuantType.
func ggmlDequantFunc(ggmlType compute.GGMLQuantType) (func(data []uint8, output []float32), error) {
	switch ggmlType {
	case compute.GGMLQ4_0:
		return dequantQ4_0Row, nil
	case compute.GGMLQ4_1:
		return dequantQ4_1Row, nil
	case compute.GGMLQ5_0:
		return dequantQ5_0Row, nil
	case compute.GGMLQ5_1:
		return dequantQ5_1Row, nil
	case compute.GGMLQ8_0:
		return dequantQ8_0Row, nil
	case compute.GGMLIQ4NL:
		return dequantIQ4NLRow, nil
	case compute.GGMLQ2_K:
		return dequantQ2_KRow, nil
	case compute.GGMLQ3_K:
		return dequantQ3_KRow, nil
	case compute.GGMLQ4_K:
		return dequantQ4_KRow, nil
	case compute.GGMLQ5_K:
		return dequantQ5_KRow, nil
	case compute.GGMLQ6_K:
		return dequantQ6_KRow, nil
	case compute.GGMLQ8_K:
		return dequantQ8_KRow, nil
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "GGML type %s not supported in fused path", ggmlType)
	}
}

// ggmlVecDotFunc returns the paired block-integer dot-product kernel (against Q8_0 or Q8_K).
func ggmlVecDotFunc(wType compute.GGMLQuantType) (func(wRow, xRow []uint8) float32, error) {
	switch wType {
	case compute.GGMLQ4_0:
		return vecDotQ4_0_Q8_0, nil
	case compute.GGMLQ4_1:
		return vecDotQ4_1_Q8_0, nil
	case compute.GGMLQ5_0:
		return vecDotQ5_0_Q8_0, nil
	case compute.GGMLQ5_1:
		return vecDotQ5_1_Q8_0, nil
	case compute.GGMLQ8_0:
		return vecDotQ8_0_Q8_0, nil
	case compute.GGMLIQ4NL:
		return vecDotIQ4NL_Q8_0, nil
	case compute.GGMLQ2_K:
		return vecDotQ2_K_Q8_K, nil
	case compute.GGMLQ3_K:
		return vecDotQ3_K_Q8_K, nil
	case compute.GGMLQ4_K:
		return vecDotQ4_K_Q8_K, nil
	case compute.GGMLQ5_K:
		return vecDotQ5_K_Q8_K, nil
	case compute.GGMLQ6_K:
		return vecDotQ6_K_Q8_K, nil
	case compute.GGMLQ8_K:
		return vecDotQ8_K_Q8_K, nil
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "GGML vecDot for type %s not supported", wType)
	}
}

func ggmlFp16LE(lo, hi uint8) float32 {
	return float16.FromBits(uint16(lo) | uint16(hi)<<8).Float32()
}

func ggmlFp32LE(b []uint8) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

func ggmlBsum16(bsums []uint8, idx int) int32 {
	return int32(int16(binary.LittleEndian.Uint16(bsums[2*idx : 2*idx+2])))
}

// --- Row Dequantizers ---

func dequantQ8_0Row(data []uint8, output []float32) {
	const blockSize = 34
	const qk = 32
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(blockData[0], blockData[1])
		qs := blockData[2:]
		outOff := b * qk
		for i := range qk {
			output[outOff+i] = d * float32(int8(qs[i]))
		}
	}
}

func dequantQ4_0Row(data []uint8, output []float32) {
	const blockSize = 18
	const qk = 32
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(blockData[0], blockData[1])
		qs := blockData[2:]
		outOff := b * qk
		for i := range 16 {
			lo := int(qs[i] & 0x0F)
			output[outOff+i] = d * float32(lo-8)
			hi := int((qs[i] >> 4) & 0x0F)
			output[outOff+16+i] = d * float32(hi-8)
		}
	}
}

func dequantQ4_1Row(data []uint8, output []float32) {
	const blockSize = 20
	const qk = 32
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(blockData[0], blockData[1])
		m := ggmlFp16LE(blockData[2], blockData[3])
		qs := blockData[4:]
		outOff := b * qk
		for i := range 16 {
			lo := qs[i] & 0x0F
			output[outOff+i] = d*float32(lo) + m
			hi := (qs[i] >> 4) & 0x0F
			output[outOff+16+i] = d*float32(hi) + m
		}
	}
}

func dequantQ5_0Row(data []uint8, output []float32) {
	const blockSize = 22
	const qk = 32
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(blockData[0], blockData[1])
		qh := binary.LittleEndian.Uint32(blockData[2:6])
		qs := blockData[6:]
		outOff := b * qk
		for i := range 16 {
			xh0 := ((qh >> uint(i)) << 4) & 0x10
			xh1 := (qh >> uint(i+12)) & 0x10
			q0 := int32(uint32(qs[i]&0x0F)|xh0) - 16
			q1 := int32(uint32(qs[i]>>4)|xh1) - 16
			output[outOff+i] = d * float32(q0)
			output[outOff+16+i] = d * float32(q1)
		}
	}
}

func dequantQ5_1Row(data []uint8, output []float32) {
	const blockSize = 24
	const qk = 32
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(blockData[0], blockData[1])
		m := ggmlFp16LE(blockData[2], blockData[3])
		qh := binary.LittleEndian.Uint32(blockData[4:8])
		qs := blockData[8:]
		outOff := b * qk
		for i := range 16 {
			xh0 := ((qh >> uint(i)) << 4) & 0x10
			xh1 := (qh >> uint(i+12)) & 0x10
			q0 := uint32(qs[i]&0x0F) | xh0
			q1 := uint32(qs[i]>>4) | xh1
			output[outOff+i] = d*float32(q0) + m
			output[outOff+16+i] = d*float32(q1) + m
		}
	}
}

func dequantIQ4NLRow(data []uint8, output []float32) {
	const blockSize = 18
	const qk = 32
	lut := &compute.IQ4NLLookupTable
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(blockData[0], blockData[1])
		qs := blockData[2:]
		outOff := b * qk
		for i := range 16 {
			lo := qs[i] & 0x0F
			output[outOff+i] = d * lut[lo]
			hi := (qs[i] >> 4) & 0x0F
			output[outOff+16+i] = d * lut[hi]
		}
	}
}

func dequantQ2_KRow(data []uint8, output []float32) {
	const blockSize = 84
	const qk = 256
	nblocks := len(data) / blockSize
	for b := range nblocks {
		src := data[b*blockSize : (b+1)*blockSize]
		scales := src[0:16]
		qs := src[16:80]
		d := ggmlFp16LE(src[80], src[81])
		dmin := ggmlFp16LE(src[82], src[83])

		outOff := b * qk
		idx := 0
		is := 0
		for n := 0; n < 256; n += 128 {
			shift := uint(0)
			for range 4 {
				sc := scales[is]
				is++
				dl := d * float32(sc&0xF)
				ml := dmin * float32(sc>>4)
				for l := range 16 {
					output[outOff+idx] = dl*float32((qs[n/4+l]>>shift)&3) - ml
					idx++
				}

				sc = scales[is]
				is++
				dl = d * float32(sc&0xF)
				ml = dmin * float32(sc>>4)
				for l := range 16 {
					output[outOff+idx] = dl*float32((qs[n/4+16+l]>>shift)&3) - ml
					idx++
				}
				shift += 2
			}
		}
	}
}

func unpackQ3KScales(scaleBytes []uint8) [16]int8 {
	var aux [4]uint32
	aux[0] = binary.LittleEndian.Uint32(scaleBytes[0:4])
	aux[1] = binary.LittleEndian.Uint32(scaleBytes[4:8])
	aux[2] = binary.LittleEndian.Uint32(scaleBytes[8:12])

	const kmask1 = uint32(0x03030303)
	const kmask2 = uint32(0x0f0f0f0f)

	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

	var scales [16]int8
	for i := range 4 {
		scales[i*4+0] = int8(aux[i])
		scales[i*4+1] = int8(aux[i] >> 8)
		scales[i*4+2] = int8(aux[i] >> 16)
		scales[i*4+3] = int8(aux[i] >> 24)
	}
	return scales
}

func dequantQ3_KRow(data []uint8, output []float32) {
	const blockSize = 110
	const qk = 256
	nblocks := len(data) / blockSize
	for b := range nblocks {
		src := data[b*blockSize : (b+1)*blockSize]
		hmask := src[0:32]
		qs := src[32:96]
		scales := unpackQ3KScales(src[96:108])
		dAll := ggmlFp16LE(src[108], src[109])

		outOff := b * qk
		idx := 0
		is := 0
		mBit := uint8(1)
		qOff := 0
		for range 2 {
			shift := uint(0)
			for range 4 {
				dl := dAll * float32(int32(scales[is])-32)
				is++
				for l := range 16 {
					q := int8((qs[qOff+l] >> shift) & 3)
					if hmask[l]&mBit == 0 {
						q -= 4
					}
					output[outOff+idx] = dl * float32(q)
					idx++
				}

				dl = dAll * float32(int32(scales[is])-32)
				is++
				for l := range 16 {
					q := int8((qs[qOff+16+l] >> shift) & 3)
					if hmask[16+l]&mBit == 0 {
						q -= 4
					}
					output[outOff+idx] = dl * float32(q)
					idx++
				}
				shift += 2
				mBit <<= 1
			}
			qOff += 32
		}
	}
}

func getScaleMinK4(j int, scales []byte) (sc, m uint8) {
	if j < 4 {
		sc = scales[j] & 63
		m = scales[j+4] & 63
	} else {
		sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
		m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	}
	return
}

func dequantQ4_KRow(data []uint8, output []float32) {
	const blockSize = 144
	const qk = 256
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(blockData[0], blockData[1])
		dmin := ggmlFp16LE(blockData[2], blockData[3])
		scales := blockData[4:16]
		qs := blockData[16:]
		outOff := b * qk
		idx := 0
		for chunk := range 4 {
			is := chunk * 2
			sc1, m1 := getScaleMinK4(is, scales)
			d1 := d * float32(sc1)
			min1 := dmin * float32(m1)

			sc2, m2 := getScaleMinK4(is+1, scales)
			d2 := d * float32(sc2)
			min2 := dmin * float32(m2)

			qOff := chunk * 32
			for l := range 32 {
				output[outOff+idx] = d1*float32(qs[qOff+l]&0xF) - min1
				idx++
			}
			for l := range 32 {
				output[outOff+idx] = d2*float32(qs[qOff+l]>>4) - min2
				idx++
			}
		}
	}
}

func dequantQ5_KRow(data []uint8, output []float32) {
	const blockSize = 176
	const qk = 256
	nblocks := len(data) / blockSize
	for b := range nblocks {
		src := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp16LE(src[0], src[1])
		dmin := ggmlFp16LE(src[2], src[3])
		scales := src[4:16]
		qh := src[16:48]
		qs := src[48:]

		outOff := b * qk
		idx := 0
		is := 0
		u1, u2 := uint8(1), uint8(2)
		qlOff := 0
		for range 4 {
			sc1, m1 := getScaleMinK4(is, scales)
			d1 := d * float32(sc1)
			min1 := dmin * float32(m1)

			sc2, m2 := getScaleMinK4(is+1, scales)
			d2 := d * float32(sc2)
			min2 := dmin * float32(m2)

			for l := range 32 {
				hbit := uint8(0)
				if qh[l]&u1 != 0 {
					hbit = 16
				}
				output[outOff+idx] = d1*float32((qs[qlOff+l]&0xF)+hbit) - min1
				idx++
			}
			for l := range 32 {
				hbit := uint8(0)
				if qh[l]&u2 != 0 {
					hbit = 16
				}
				output[outOff+idx] = d2*float32((qs[qlOff+l]>>4)+hbit) - min2
				idx++
			}
			qlOff += 32
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

func dequantQ6_KRow(data []uint8, output []float32) {
	const blockSize = 210
	const qk = 256
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		ql := blockData[0:128]
		qh := blockData[128:192]
		sc := blockData[192:208]
		d := ggmlFp16LE(blockData[208], blockData[209])
		outOff := b * qk
		idx := 0
		qlOff := 0
		qhOff := 0
		scOff := 0
		for range 2 {
			for l := range 32 {
				is := l / 16
				q1 := int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]&3)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0xF)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[qlOff+l]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
				output[outOff+idx+l] = d * float32(int8(sc[scOff+is])) * float32(q1)
				output[outOff+idx+l+32] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
				output[outOff+idx+l+64] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
				output[outOff+idx+l+96] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
			}
			idx += 128
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	}
}

func dequantQ8_KRow(data []uint8, output []float32) {
	const blockSize = 292
	const qk = 256
	nblocks := len(data) / blockSize
	for b := range nblocks {
		blockData := data[b*blockSize : (b+1)*blockSize]
		d := ggmlFp32LE(blockData[0:4])
		qs := blockData[4:260]
		outOff := b * qk
		for i := range qk {
			output[outOff+i] = d * float32(int8(qs[i]))
		}
	}
}

// --- Paired Block-Integer Dot Products (Mode B) ---

func vecDotQ8_0_Q8_0(wRow, xRow []uint8) float32 {
	const wBytes = 34
	const xBytes = 34
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		dX := ggmlFp16LE(xb[0], xb[1])
		qw := wb[2:34]
		qx := xb[2:34]
		var sumi int32
		for i := range 32 {
			sumi += int32(int8(qw[i])) * int32(int8(qx[i]))
		}
		acc += (dW * dX) * float32(sumi)
	}
	return acc
}

func vecDotQ4_0_Q8_0(wRow, xRow []uint8) float32 {
	const wBytes = 18
	const xBytes = 34
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		dX := ggmlFp16LE(xb[0], xb[1])
		qw := wb[2:18]
		qx := xb[2:34]
		var sumi int32
		for i := range 16 {
			lo := int32(qw[i]&0x0F) - 8
			hi := int32(qw[i]>>4) - 8
			sumi += lo*int32(int8(qx[i])) + hi*int32(int8(qx[16+i]))
		}
		acc += (dW * dX) * float32(sumi)
	}
	return acc
}

func vecDotQ4_1_Q8_0(wRow, xRow []uint8) float32 {
	const wBytes = 20
	const xBytes = 34
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		mW := ggmlFp16LE(wb[2], wb[3])
		dX := ggmlFp16LE(xb[0], xb[1])
		qw := wb[4:20]
		qx := xb[2:34]
		var sumi, sumx int32
		for i := range 16 {
			x0 := int32(int8(qx[i]))
			x1 := int32(int8(qx[16+i]))
			lo := int32(qw[i] & 0x0F)
			hi := int32(qw[i] >> 4)
			sumi += lo*x0 + hi*x1
			sumx += x0 + x1
		}
		acc += (dW*dX)*float32(sumi) + (mW*dX)*float32(sumx)
	}
	return acc
}

func vecDotQ5_0_Q8_0(wRow, xRow []uint8) float32 {
	const wBytes = 22
	const xBytes = 34
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		qh := binary.LittleEndian.Uint32(wb[2:6])
		dX := ggmlFp16LE(xb[0], xb[1])
		qw := wb[6:22]
		qx := xb[2:34]
		var sumi int32
		for i := range 16 {
			xh0 := ((qh >> uint(i)) << 4) & 0x10
			xh1 := (qh >> uint(i+12)) & 0x10
			q0 := int32(uint32(qw[i]&0x0F)|xh0) - 16
			q1 := int32(uint32(qw[i]>>4)|xh1) - 16
			sumi += q0*int32(int8(qx[i])) + q1*int32(int8(qx[16+i]))
		}
		acc += (dW * dX) * float32(sumi)
	}
	return acc
}

func vecDotQ5_1_Q8_0(wRow, xRow []uint8) float32 {
	const wBytes = 24
	const xBytes = 34
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		mW := ggmlFp16LE(wb[2], wb[3])
		qh := binary.LittleEndian.Uint32(wb[4:8])
		dX := ggmlFp16LE(xb[0], xb[1])
		qw := wb[8:24]
		qx := xb[2:34]
		var sumi, sumx int32
		for i := range 16 {
			x0 := int32(int8(qx[i]))
			x1 := int32(int8(qx[16+i]))
			xh0 := ((qh >> uint(i)) << 4) & 0x10
			xh1 := (qh >> uint(i+12)) & 0x10
			q0 := int32(uint32(qw[i]&0x0F) | xh0)
			q1 := int32(uint32(qw[i]>>4) | xh1)
			sumi += q0*x0 + q1*x1
			sumx += x0 + x1
		}
		acc += (dW*dX)*float32(sumi) + (mW*dX)*float32(sumx)
	}
	return acc
}

func vecDotIQ4NL_Q8_0(wRow, xRow []uint8) float32 {
	const wBytes = 18
	const xBytes = 34
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		dX := ggmlFp16LE(xb[0], xb[1])
		qw := wb[2:18]
		qx := xb[2:34]
		var sumi int32
		for i := range 16 {
			lo := int32(iq4nlIntTable[qw[i]&0x0F])
			hi := int32(iq4nlIntTable[qw[i]>>4])
			sumi += lo*int32(int8(qx[i])) + hi*int32(int8(qx[16+i]))
		}
		acc += (dW * dX) * float32(sumi)
	}
	return acc
}

func vecDotQ2_K_Q8_K(wRow, xRow []uint8) float32 {
	const wBytes = 84
	const xBytes = 292
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		scales := wb[0:16]
		qs := wb[16:80]
		dW := ggmlFp16LE(wb[80], wb[81])
		dminW := ggmlFp16LE(wb[82], wb[83])
		dX := ggmlFp32LE(xb[0:4])
		qx := xb[4:260]
		bsums := xb[260:292]

		var sumi, sumMin int32
		idx := 0
		is := 0
		for n := 0; n < 256; n += 128 {
			shift := uint(0)
			for range 4 {
				sc := scales[is]
				var dot1 int32
				for l := range 16 {
					dot1 += int32((qs[n/4+l]>>shift)&3) * int32(int8(qx[idx+l]))
				}
				sumi += int32(sc&0xF) * dot1
				sumMin += int32(sc>>4) * ggmlBsum16(bsums, is)
				is++
				idx += 16

				sc = scales[is]
				var dot2 int32
				for l := range 16 {
					dot2 += int32((qs[n/4+16+l]>>shift)&3) * int32(int8(qx[idx+l]))
				}
				sumi += int32(sc&0xF) * dot2
				sumMin += int32(sc>>4) * ggmlBsum16(bsums, is)
				is++
				idx += 16

				shift += 2
			}
		}
		acc += (dW*dX)*float32(sumi) - (dminW*dX)*float32(sumMin)
	}
	return acc
}

func vecDotQ3_K_Q8_K(wRow, xRow []uint8) float32 {
	const wBytes = 110
	const xBytes = 292
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		hmask := wb[0:32]
		qs := wb[32:96]
		scales := unpackQ3KScales(wb[96:108])
		dAll := ggmlFp16LE(wb[108], wb[109])
		dX := ggmlFp32LE(xb[0:4])
		qx := xb[4:260]

		var sumi int32
		idx := 0
		is := 0
		mBit := uint8(1)
		qOff := 0
		for range 2 {
			shift := uint(0)
			for range 4 {
				sc1 := int32(scales[is]) - 32
				is++
				var dot1 int32
				for l := range 16 {
					q := int32((qs[qOff+l] >> shift) & 3)
					if hmask[l]&mBit == 0 {
						q -= 4
					}
					dot1 += q * int32(int8(qx[idx+l]))
				}
				sumi += sc1 * dot1
				idx += 16

				sc2 := int32(scales[is]) - 32
				is++
				var dot2 int32
				for l := range 16 {
					q := int32((qs[qOff+16+l] >> shift) & 3)
					if hmask[16+l]&mBit == 0 {
						q -= 4
					}
					dot2 += q * int32(int8(qx[idx+l]))
				}
				sumi += sc2 * dot2
				idx += 16

				shift += 2
				mBit <<= 1
			}
			qOff += 32
		}
		acc += (dAll * dX) * float32(sumi)
	}
	return acc
}

func vecDotQ4_K_Q8_K(wRow, xRow []uint8) float32 {
	const wBytes = 144
	const xBytes = 292
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		dminW := ggmlFp16LE(wb[2], wb[3])
		scales := wb[4:16]
		qs := wb[16:144]
		dX := ggmlFp32LE(xb[0:4])
		qx := xb[4:260]
		bsums := xb[260:292]

		var sumi, sumMin int32
		for chunk := range 4 {
			is := chunk * 2
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			qOff := chunk * 32
			xOff := chunk * 64
			var dot1, dot2 int32
			for l := range 32 {
				dot1 += int32(qs[qOff+l]&0xF) * int32(int8(qx[xOff+l]))
				dot2 += int32(qs[qOff+l]>>4) * int32(int8(qx[xOff+32+l]))
			}
			sumi += int32(sc1)*dot1 + int32(sc2)*dot2
			bs1 := ggmlBsum16(bsums, 2*is) + ggmlBsum16(bsums, 2*is+1)
			bs2 := ggmlBsum16(bsums, 2*(is+1)) + ggmlBsum16(bsums, 2*(is+1)+1)
			sumMin += int32(m1)*bs1 + int32(m2)*bs2
		}
		acc += (dW*dX)*float32(sumi) - (dminW*dX)*float32(sumMin)
	}
	return acc
}

func vecDotQ5_K_Q8_K(wRow, xRow []uint8) float32 {
	const wBytes = 176
	const xBytes = 292
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp16LE(wb[0], wb[1])
		dminW := ggmlFp16LE(wb[2], wb[3])
		scales := wb[4:16]
		qh := wb[16:48]
		qs := wb[48:176]
		dX := ggmlFp32LE(xb[0:4])
		qx := xb[4:260]
		bsums := xb[260:292]

		var sumi, sumMin int32
		u1, u2 := uint8(1), uint8(2)
		for chunk := range 4 {
			is := chunk * 2
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			qlOff := chunk * 32
			xOff := chunk * 64
			var dot1, dot2 int32
			for l := range 32 {
				hb1 := int32(0)
				if qh[l]&u1 != 0 {
					hb1 = 16
				}
				hb2 := int32(0)
				if qh[l]&u2 != 0 {
					hb2 = 16
				}
				dot1 += (int32(qs[qlOff+l]&0xF) + hb1) * int32(int8(qx[xOff+l]))
				dot2 += (int32(qs[qlOff+l]>>4) + hb2) * int32(int8(qx[xOff+32+l]))
			}
			sumi += int32(sc1)*dot1 + int32(sc2)*dot2
			bs1 := ggmlBsum16(bsums, 2*is) + ggmlBsum16(bsums, 2*is+1)
			bs2 := ggmlBsum16(bsums, 2*(is+1)) + ggmlBsum16(bsums, 2*(is+1)+1)
			sumMin += int32(m1)*bs1 + int32(m2)*bs2
			u1 <<= 2
			u2 <<= 2
		}
		acc += (dW*dX)*float32(sumi) - (dminW*dX)*float32(sumMin)
	}
	return acc
}

func vecDotQ6_K_Q8_K(wRow, xRow []uint8) float32 {
	const wBytes = 210
	const xBytes = 292
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		ql := wb[0:128]
		qh := wb[128:192]
		sc := wb[192:208]
		dW := ggmlFp16LE(wb[208], wb[209])
		dX := ggmlFp32LE(xb[0:4])
		qx := xb[4:260]

		var sumi int32
		idx := 0
		qlOff := 0
		qhOff := 0
		scOff := 0
		for range 2 {
			for is := range 2 {
				lBase := is * 16
				var dot1, dot2, dot3, dot4 int32
				for l := lBase; l < lBase+16; l++ {
					q1 := int32(int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]&3)<<4)) - 32)
					q2 := int32(int8((ql[qlOff+l+32]&0xF)|(((qh[qhOff+l]>>2)&3)<<4)) - 32)
					q3 := int32(int8((ql[qlOff+l]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32)
					q4 := int32(int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32)
					dot1 += q1 * int32(int8(qx[idx+l]))
					dot2 += q2 * int32(int8(qx[idx+l+32]))
					dot3 += q3 * int32(int8(qx[idx+l+64]))
					dot4 += q4 * int32(int8(qx[idx+l+96]))
				}
				sumi += int32(int8(sc[scOff+is]))*dot1 +
					int32(int8(sc[scOff+is+2]))*dot2 +
					int32(int8(sc[scOff+is+4]))*dot3 +
					int32(int8(sc[scOff+is+6]))*dot4
			}
			idx += 128
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
		acc += (dW * dX) * float32(sumi)
	}
	return acc
}

func vecDotQ8_K_Q8_K(wRow, xRow []uint8) float32 {
	const wBytes = 292
	const xBytes = 292
	nblocks := len(wRow) / wBytes
	var acc float32
	for b := range nblocks {
		wb := wRow[b*wBytes : (b+1)*wBytes]
		xb := xRow[b*xBytes : (b+1)*xBytes]
		dW := ggmlFp32LE(wb[0:4])
		dX := ggmlFp32LE(xb[0:4])
		qw := wb[4:260]
		qx := xb[4:260]
		var sumi int32
		for i := range 256 {
			sumi += int32(int8(qw[i])) * int32(int8(qx[i]))
		}
		acc += (dW * dX) * float32(sumi)
	}
	return acc
}

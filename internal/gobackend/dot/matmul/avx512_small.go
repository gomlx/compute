// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64 && goexperiment.simd

package matmul

import (
	"simd/archsimd"
	"unsafe"

	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
	//alt:bf16 "github.com/gomlx/compute/dtypes/bfloat16"
)

// avx512SmallFloat32Parallel implements a parallelized version of the AVX512 small matrix
// multiplication.
func avx512SmallFloat32Parallel( //alt:f32
	//alt:bf16 func avx512SmallBFloat16Parallel(
	backend *gobackend.Backend,
	layout dot.Layout,
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32) {

	// Split work by batch items.
	// For small matrices, we only parallelize along the batch dimension if it's large enough.
	maxWorkers := 1
	if backend.Workers != nil {
		maxWorkers = backend.Workers.AdjustedMaxParallelism()
	}

	if maxWorkers <= 1 || batchSize <= 1 {
		if layout == dot.LayoutNonTransposed {
			avx512SmallFloat32( //alt:f32
				//alt:bf16 avx512SmallBFloat16(
				lhs, rhs, 0, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		} else {
			avx512SmallFloat32Transposed( //alt:f32
				//alt:bf16 avx512SmallBFloat16Transposed(
				lhs, rhs, 0, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, output)
		}
		return
	}

	type chunkData struct {
		batchIdx, batchCount int
	}
	numChunks := min(batchSize, maxWorkers*2)
	batchPerChunk := (batchSize + numChunks - 1) / numChunks
	work := make(chan chunkData, numChunks)
	for batchIdx := 0; batchIdx < batchSize; batchIdx += batchPerChunk {
		count := min(batchPerChunk, batchSize-batchIdx)
		work <- chunkData{batchIdx, count}
	}
	close(work)

	backend.Workers.Saturate(func() {
		for chunk := range work {
			if layout == dot.LayoutNonTransposed {
				avx512SmallFloat32( //alt:f32
					//alt:bf16 avx512SmallBFloat16(
					lhs, rhs, chunk.batchIdx, chunk.batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
			} else {
				avx512SmallFloat32Transposed( //alt:f32
					//alt:bf16 avx512SmallBFloat16Transposed(
					lhs, rhs, chunk.batchIdx, chunk.batchCount, lhsCrossSize, rhsCrossSize, contractingSize, output)
			}
		}
	})
}

// avx512SmallFloat32 implements an AVX512 matrix multiplication for small inputs,
// without data packing. It vectorizes the contracting dimension.
// Layout is dot.LayoutNonTransposed.
func avx512SmallFloat32( //alt:f32
	//alt:bf16 func avx512SmallBFloat16(
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32) {

	if batchCount == 0 || lhsCrossSize == 0 || rhsCrossSize == 0 || contractingSize == 0 {
		return
	}

	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize

	var iZero float32 //alt:f32
	//alt:bf16 var iZero bfloat16.BFloat16
	iSize := unsafe.Sizeof(iZero)
	var oZero float32
	oSize := unsafe.Sizeof(oZero)

	lhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(lhs)))
	rhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(rhs)))
	outputPtr := uintptr(unsafe.Pointer(unsafe.SliceData(output)))

	lhsByteStride := uintptr(lhsStride) * iSize
	rhsByteStride := uintptr(rhsStride) * iSize
	outputByteStride := uintptr(outputStride) * oSize

	lhsBase := lhsPtr + uintptr(batchStart)*lhsByteStride
	rhsBase := rhsPtr + uintptr(batchStart)*rhsByteStride
	outputBase := outputPtr + uintptr(batchStart)*outputByteStride

	for range batchCount {
		for row := 0; row < lhsCrossSize; row++ {
			for col := 0; col < rhsCrossSize; col++ {
				lRowBase := lhsBase + uintptr(row*contractingSize)*iSize
				rColBase := rhsBase + uintptr(col)*iSize
				rRowStride := uintptr(rhsCrossSize) * iSize

				// Accumulate dot product using AVX512
				acc := archsimd.BroadcastFloat32x16(0.0)
				k := 0
				// Vector width: 16 for float32, 32 for bfloat16
				const vecWidth = 16 //alt:f32
				//alt:bf16 const vecWidth = 32

				// Unroll by 4 registers on the contracting size
				for ; k+4*vecWidth <= contractingSize; k += 4 * vecWidth {
					// Load LHS (contiguous)
					l0 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k)*iSize))) //alt:f32
					//alt:bf16 l0_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k)*iSize)))
					//alt:bf16 l0a, l0b := l0_bf16.ToFloat32()

					// Load RHS (strided for NonTransposed)
					var r0 archsimd.Float32x16 //alt:f32
					//alt:bf16 var r0a, r0b archsimd.Float32x16
					{
						var tmp [vecWidth]float32 //alt:f32
						//alt:bf16 var tmp [vecWidth]bfloat16.BFloat16
						for i := range vecWidth {
							tmp[i] = *(*float32)(unsafe.Pointer(rColBase + uintptr(k+i)*rRowStride)) //alt:f32
							//alt:bf16 tmp[i] = *(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+i)*rRowStride))
						}
						r0 = archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(&tmp[0]))) //alt:f32
						//alt:bf16 r0_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(&tmp[0])))
						//alt:bf16 r0a, r0b = r0_bf16.ToFloat32()
					}
					acc = l0.MulAdd(r0, acc) //alt:f32
					//alt:bf16 acc = l0a.MulAdd(r0a, acc)
					//alt:bf16 acc = l0b.MulAdd(r0b, acc)

					l1 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k+vecWidth)*iSize))) //alt:f32
					//alt:bf16 l1_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k+vecWidth)*iSize)))
					//alt:bf16 l1a, l1b := l1_bf16.ToFloat32()
					var r1 archsimd.Float32x16 //alt:f32
					//alt:bf16 var r1a, r1b archsimd.Float32x16
					{
						var tmp [vecWidth]float32 //alt:f32
						//alt:bf16 var tmp [vecWidth]bfloat16.BFloat16
						for i := range vecWidth {
							tmp[i] = *(*float32)(unsafe.Pointer(rColBase + uintptr(k+vecWidth+i)*rRowStride)) //alt:f32
							//alt:bf16 tmp[i] = *(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+vecWidth+i)*rRowStride))
						}
						r1 = archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(&tmp[0]))) //alt:f32
						//alt:bf16 r1_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(&tmp[0])))
						//alt:bf16 r1a, r1b = r1_bf16.ToFloat32()
					}
					acc = l1.MulAdd(r1, acc) //alt:f32
					//alt:bf16 acc = l1a.MulAdd(r1a, acc)
					//alt:bf16 acc = l1b.MulAdd(r1b, acc)

					l2 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k+2*vecWidth)*iSize))) //alt:f32
					//alt:bf16 l2_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k+2*vecWidth)*iSize)))
					//alt:bf16 l2a, l2b := l2_bf16.ToFloat32()
					var r2 archsimd.Float32x16 //alt:f32
					//alt:bf16 var r2a, r2b archsimd.Float32x16
					{
						var tmp [vecWidth]float32 //alt:f32
						//alt:bf16 var tmp [vecWidth]bfloat16.BFloat16
						for i := range vecWidth {
							tmp[i] = *(*float32)(unsafe.Pointer(rColBase + uintptr(k+2*vecWidth+i)*rRowStride)) //alt:f32
							//alt:bf16 tmp[i] = *(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+2*vecWidth+i)*rRowStride))
						}
						r2 = archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(&tmp[0]))) //alt:f32
						//alt:bf16 r2_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(&tmp[0])))
						//alt:bf16 r2a, r2b = r2_bf16.ToFloat32()
					}
					acc = l2.MulAdd(r2, acc) //alt:f32
					//alt:bf16 acc = l2a.MulAdd(r2a, acc)
					//alt:bf16 acc = l2b.MulAdd(r2b, acc)

					l3 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k+3*vecWidth)*iSize))) //alt:f32
					//alt:bf16 l3_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k+3*vecWidth)*iSize)))
					//alt:bf16 l3a, l3b := l3_bf16.ToFloat32()
					var r3 archsimd.Float32x16 //alt:f32
					//alt:bf16 var r3a, r3b archsimd.Float32x16
					{
						var tmp [vecWidth]float32 //alt:f32
						//alt:bf16 var tmp [vecWidth]bfloat16.BFloat16
						for i := range vecWidth {
							tmp[i] = *(*float32)(unsafe.Pointer(rColBase + uintptr(k+3*vecWidth+i)*rRowStride)) //alt:f32
							//alt:bf16 tmp[i] = *(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+3*vecWidth+i)*rRowStride))
						}
						r3 = archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(&tmp[0]))) //alt:f32
						//alt:bf16 r3_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(&tmp[0])))
						//alt:bf16 r3a, r3b = r3_bf16.ToFloat32()
					}
					acc = l3.MulAdd(r3, acc) //alt:f32
					//alt:bf16 acc = l3a.MulAdd(r3a, acc)
					//alt:bf16 acc = l3b.MulAdd(r3b, acc)
				}

				// Remaining full vectors
				for ; k+vecWidth <= contractingSize; k += vecWidth {
					lVal := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k)*iSize))) //alt:f32
					//alt:bf16 l_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k)*iSize)))
					//alt:bf16 la, lb := l_bf16.ToFloat32()
					var rVal archsimd.Float32x16 //alt:f32
					//alt:bf16 var ra, rb archsimd.Float32x16
					{
						var tmp [vecWidth]float32 //alt:f32
						//alt:bf16 var tmp [vecWidth]bfloat16.BFloat16
						for i := range vecWidth {
							tmp[i] = *(*float32)(unsafe.Pointer(rColBase + uintptr(k+i)*rRowStride)) //alt:f32
							//alt:bf16 tmp[i] = *(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+i)*rRowStride))
						}
						rVal = archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(&tmp[0]))) //alt:f32
						//alt:bf16 r_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(&tmp[0])))
						//alt:bf16 ra, rb = r_bf16.ToFloat32()
					}
					acc = lVal.MulAdd(rVal, acc) //alt:f32
					//alt:bf16 acc = la.MulAdd(ra, acc)
					//alt:bf16 acc = lb.MulAdd(rb, acc)
				}

				// Horizontal reduction
				res := avx512ReduceSumFloat32x16(acc)
				// Scalar tail
				for ; k < contractingSize; k++ {
					lVal := *(*float32)(unsafe.Pointer(lRowBase + uintptr(k)*iSize)) //alt:f32
					//alt:bf16 lVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k)*iSize))).Float32()
					rVal := *(*float32)(unsafe.Pointer(rColBase + uintptr(k)*rRowStride)) //alt:f32
					//alt:bf16 rVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k)*rRowStride))).Float32()
					res += lVal * rVal
				}

				// Write result
				outputIdx := outputBase + uintptr(row*rhsCrossSize+col)*oSize
				*(*float32)(unsafe.Pointer(outputIdx)) = res
			}
		}

		lhsBase += lhsByteStride
		rhsBase += rhsByteStride
		outputBase += outputByteStride
	}
}

// avx512SmallFloat32Transposed implements an AVX512 matrix multiplication for small inputs,
// without data packing. It vectorizes the contracting dimension.
// Layout is dot.LayoutTransposed.
func avx512SmallFloat32Transposed( //alt:f32
	//alt:bf16 func avx512SmallBFloat16Transposed(
	lhs, rhs []float32, //alt:f32
	//alt:bf16 lhs, rhs []bfloat16.BFloat16,
	batchStart, batchCount, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []float32) {

	if batchCount == 0 || lhsCrossSize == 0 || rhsCrossSize == 0 || contractingSize == 0 {
		return
	}

	lhsStride := lhsCrossSize * contractingSize
	rhsStride := contractingSize * rhsCrossSize
	outputStride := lhsCrossSize * rhsCrossSize

	var iZero float32 //alt:f32
	//alt:bf16 var iZero bfloat16.BFloat16
	iSize := unsafe.Sizeof(iZero)
	var oZero float32
	oSize := unsafe.Sizeof(oZero)

	lhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(lhs)))
	rhsPtr := uintptr(unsafe.Pointer(unsafe.SliceData(rhs)))
	outputPtr := uintptr(unsafe.Pointer(unsafe.SliceData(output)))

	lhsByteStride := uintptr(lhsStride) * iSize
	rhsByteStride := uintptr(rhsStride) * iSize
	outputByteStride := uintptr(outputStride) * oSize

	lhsBase := lhsPtr + uintptr(batchStart)*lhsByteStride
	rhsBase := rhsPtr + uintptr(batchStart)*rhsByteStride
	outputBase := outputPtr + uintptr(batchStart)*outputByteStride

	for range batchCount {
		for row := 0; row < lhsCrossSize; row++ {
			for col := 0; col < rhsCrossSize; col++ {
				lRowBase := lhsBase + uintptr(row*contractingSize)*iSize
				rColBase := rhsBase + uintptr(col*contractingSize)*iSize

				// Accumulate dot product using AVX512
				acc := archsimd.BroadcastFloat32x16(0.0)
				k := 0
				// Vector width: 16 for float32, 32 for bfloat16
				const vecWidth = 16 //alt:f32
				//alt:bf16 const vecWidth = 32

				// Unroll by 4 registers on the contracting size
				for ; k+4*vecWidth <= contractingSize; k += 4 * vecWidth {
					// Load LHS (contiguous)
					l0 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k)*iSize))) //alt:f32
					//alt:bf16 l0_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k)*iSize)))
					//alt:bf16 l0a, l0b := l0_bf16.ToFloat32()

					// Load RHS (contiguous for Transposed)
					r0 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(rColBase + uintptr(k)*iSize))) //alt:f32
					//alt:bf16 r0_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k)*iSize)))
					//alt:bf16 r0a, r0b := r0_bf16.ToFloat32()

					acc = l0.MulAdd(r0, acc) //alt:f32
					//alt:bf16 acc = l0a.MulAdd(r0a, acc)
					//alt:bf16 acc = l0b.MulAdd(r0b, acc)

					l1 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k+vecWidth)*iSize))) //alt:f32
					//alt:bf16 l1_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k+vecWidth)*iSize)))
					//alt:bf16 l1a, l1b := l1_bf16.ToFloat32()
					r1 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(rColBase + uintptr(k+vecWidth)*iSize))) //alt:f32
					//alt:bf16 r1_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+vecWidth)*iSize)))
					//alt:bf16 r1a, r1b := r1_bf16.ToFloat32()
					acc = l1.MulAdd(r1, acc) //alt:f32
					//alt:bf16 acc = l1a.MulAdd(r1a, acc)
					//alt:bf16 acc = l1b.MulAdd(r1b, acc)

					l2 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k+2*vecWidth)*iSize))) //alt:f32
					//alt:bf16 l2_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k+2*vecWidth)*iSize)))
					//alt:bf16 l2a, l2b := l2_bf16.ToFloat32()
					r2 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(rColBase + uintptr(k+2*vecWidth)*iSize))) //alt:f32
					//alt:bf16 r2_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+2*vecWidth)*iSize)))
					//alt:bf16 r2a, r2b := r2_bf16.ToFloat32()
					acc = l2.MulAdd(r2, acc) //alt:f32
					//alt:bf16 acc = l2a.MulAdd(r2a, acc)
					//alt:bf16 acc = l2b.MulAdd(r2b, acc)

					l3 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k+3*vecWidth)*iSize))) //alt:f32
					//alt:bf16 l3_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k+3*vecWidth)*iSize)))
					//alt:bf16 l3a, l3b := l3_bf16.ToFloat32()
					r3 := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(rColBase + uintptr(k+3*vecWidth)*iSize))) //alt:f32
					//alt:bf16 r3_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k+3*vecWidth)*iSize)))
					//alt:bf16 r3a, r3b := r3_bf16.ToFloat32()
					acc = l3.MulAdd(r3, acc) //alt:f32
					//alt:bf16 acc = l3a.MulAdd(r3a, acc)
					//alt:bf16 acc = l3b.MulAdd(r3b, acc)
				}

				// Remaining full vectors
				for ; k+vecWidth <= contractingSize; k += vecWidth {
					lVal := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(lRowBase + uintptr(k)*iSize))) //alt:f32
					//alt:bf16 l_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k)*iSize)))
					//alt:bf16 la, lb := l_bf16.ToFloat32()
					rVal := archsimd.LoadFloat32x16((*[16]float32)(unsafe.Pointer(rColBase + uintptr(k)*iSize))) //alt:f32
					//alt:bf16 r_bf16 := bfloat16.LoadBFloat16x32((*[32]bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k)*iSize)))
					//alt:bf16 ra, rb := r_bf16.ToFloat32()
					acc = lVal.MulAdd(rVal, acc) //alt:f32
					//alt:bf16 acc = la.MulAdd(ra, acc)
					//alt:bf16 acc = lb.MulAdd(rb, acc)
				}

				// Horizontal reduction
				res := avx512ReduceSumFloat32x16(acc)
				// Scalar tail
				for ; k < contractingSize; k++ {
					lVal := *(*float32)(unsafe.Pointer(lRowBase + uintptr(k)*iSize)) //alt:f32
					//alt:bf16 lVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(lRowBase + uintptr(k)*iSize))).Float32()
					rVal := *(*float32)(unsafe.Pointer(rColBase + uintptr(k)*iSize)) //alt:f32
					//alt:bf16 rVal := (*(*bfloat16.BFloat16)(unsafe.Pointer(rColBase + uintptr(k)*iSize))).Float32()
					res += lVal * rVal
				}

				// Write result
				outputIdx := outputBase + uintptr(row*rhsCrossSize+col)*oSize
				*(*float32)(unsafe.Pointer(outputIdx)) = res
			}
		}

		lhsBase += lhsByteStride
		rhsBase += rhsByteStride
		outputBase += outputByteStride
	}
}

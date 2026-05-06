package nontransposed

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend/workerspool"
	"k8s.io/klog/v2"
)

// noSIMDRouter implements a non-SIMD matrix multiplication for the non-transposed layout.
//
// It is used when no SIMD-optimized implementation is available, or for non-supported
// dtype configuration.
func noSIMDRouter[I, O dtypes.NumberNotComplex]( //alt:generic
	//alt:half func noSIMDHalfPrecisionRouter[I dtypes.HalfPrecision[I], O dtypes.NumberNotComplex](
	lhs, rhs []I,
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int,
	output []O,
	pool *workerspool.Pool) {

	// 1. Resolve Strides

	// 2. Check if small matrix multiplication kernel can be used.
	flopsPerMatrix := lhsCrossSize * rhsCrossSize * contractingSize
	flops := batchSize * flopsPerMatrix
	_ = flops
	// if !ForceLargeVariant && (ForceSmallVariant || flops < noSIMDSmallMatMulSizeThreshold) {
	klog.V(1).Infof("Using small variant for NonTransposed non-SIMD dot-product kernel")
	matricesPerWorker := (noSIMDMinMatMulFlopsPerWorker + (flopsPerMatrix - 1)) / flopsPerMatrix
	matricesPerWorker = min(matricesPerWorker, batchSize)
	if matricesPerWorker < 2 {
		// The overhead of distributing the processing is not worth it:
		smallNoSIMDGeneric( //alt:generic
			//alt:half smallNoSIMDHalfPrecision(
			lhs, rhs,
			0, batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			output)
	} else {
		smallNoSIMDGenericParallel( //alt:generic
			//alt:half smallNoSIMDHalfPrecisionParallel(
			lhs, rhs,
			batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
			output, pool, matricesPerWorker)
	}

	// klog.V(1).Infof("Using large variant for NonTransposed non-SIMD dot-product kernel")
	// return basicSymmetricGenericLargeGEMMParallel(
	// 	alpha, beta,
	// 	lhsFlat, rhsFlat, outputFlat,
	// 	batchSize, lhsCrossSize, rhsCrossSize, contractingSize,
	// 	lhsBatchStride, rhsBatchStride, outputBatchStride,
	// 	bufAllocFn, bufReleaseFn,
	// 	pool)
}

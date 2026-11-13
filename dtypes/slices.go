package dtypes

import (
	"slices"
	"strconv"
	"unsafe"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/pkg/errors"
)

// UnsafeByteSliceFromAny casts a slice of any of the supported Go types (feed as type any) to a slice of bytes.
func UnsafeByteSliceFromAny(flatAny any) []byte {
	if flatAny == nil {
		// Zero-sized shapes use a nil flat.
		return nil
	}
	switch flat := flatAny.(type) {
	case []float64:
		return UnsafeByteSlice(flat)
	case []float32:
		return UnsafeByteSlice(flat)
	case []float16.Float16:
		return UnsafeByteSlice(flat)
	case []bfloat16.BFloat16:
		return UnsafeByteSlice(flat)
	case []int:
		return UnsafeByteSlice(flat)
	case []int64:
		return UnsafeByteSlice(flat)
	case []int32:
		return UnsafeByteSlice(flat)
	case []int16:
		return UnsafeByteSlice(flat)
	case []int8:
		return UnsafeByteSlice(flat)
	case []uint64:
		return UnsafeByteSlice(flat)
	case []uint32:
		return UnsafeByteSlice(flat)
	case []uint16:
		return UnsafeByteSlice(flat)
	case []uint8:
		return UnsafeByteSlice(flat)
	case []bool:
		return UnsafeByteSlice(flat)
	case []complex64:
		return UnsafeByteSlice(flat)
	case []complex128:
		return UnsafeByteSlice(flat)
	default:
		panicf("unsupported dtype for UnsafeByteSliceFromAny: %T", flat)
	}
	panic(nil)
}

// UnsafeByteSlice casts a slice of any of the supported Go types to a slice of bytes.
func UnsafeByteSlice[E gotype.Supported](flat []E) []byte {
	if len(flat) == 0 {
		return nil
	}
	var e E
	elementSize := int(unsafe.Sizeof(e))
	return unsafe.Slice((*byte)(unsafe.Pointer(&flat[0])), len(flat)*elementSize)
}

// UnsafeAnySliceFromBytes casts a pointer to a buffer of bytes to a slice of the given dtype and length
// pointing to the same data.
//
// For sub-byte types (Uint1, Uint2, Uint4, Int1, Int2, Int4) it returns a slice of uint8 of
// the length adjusted to hold that many elements (packed), length is always
// given in number of elements (bits, crumbs, nibbles in this case).
//
// Unsafe: bytesPtr must have enough data to hold the []dtype of the given length.
func UnsafeAnySliceFromBytes(bytesPtr unsafe.Pointer, dtype DType, length int) any {
	switch dtype {
	case Float64:
		return UnsafeSliceFromBytes[float64](bytesPtr, length)
	case Float32:
		return UnsafeSliceFromBytes[float32](bytesPtr, length)
	case Float16:
		return UnsafeSliceFromBytes[float16.Float16](bytesPtr, length)
	case BFloat16:
		return UnsafeSliceFromBytes[bfloat16.BFloat16](bytesPtr, length)
	case Int64:
		return UnsafeSliceFromBytes[int64](bytesPtr, length)
	case Int32:
		return UnsafeSliceFromBytes[int32](bytesPtr, length)
	case Int16:
		return UnsafeSliceFromBytes[int16](bytesPtr, length)
	case Int8:
		return UnsafeSliceFromBytes[int8](bytesPtr, length)
	case Uint64:
		return UnsafeSliceFromBytes[uint64](bytesPtr, length)
	case Uint32:
		return UnsafeSliceFromBytes[uint32](bytesPtr, length)
	case Uint16:
		return UnsafeSliceFromBytes[uint16](bytesPtr, length)
	case Uint8:
		return UnsafeSliceFromBytes[uint8](bytesPtr, length)
	case Bool:
		return UnsafeSliceFromBytes[bool](bytesPtr, length)
	case Complex64:
		return UnsafeSliceFromBytes[complex64](bytesPtr, length)
	case Complex128:
		return UnsafeSliceFromBytes[complex128](bytesPtr, length)
	case Int1, Uint1, Int2, Uint2, Int4, Uint4:
		// Sub-byte packed types are stored as uint8.
		return UnsafeSliceFromBytes[uint8](bytesPtr, dtype.SizeForDimensions(length))
	default:
		panicf("unsupported dtype for UnsafeByteSliceFromAny: %s", dtype)
	}
	panic(nil)
}

// UnsafeSliceFromBytes casts a pointer to a buffer of bytes to a slice of the given E type and length
// pointing to the same data.
//
// If length is zero or bytesPtr is nil, it returns nil.
//
// Unsafe: bytesPtr must have enough data to hold the []E of the given length.
func UnsafeSliceFromBytes[E gotype.Supported](bytesPtr unsafe.Pointer, length int) []E {
	if bytesPtr == nil || length == 0 {
		return nil
	}
	return unsafe.Slice((*E)(bytesPtr), length)
}

// MakeAnySlice creates a slice of the given dtype and length, casted to any.
//
// For sub-byte types (Uint1, Uint2, Uint4, Int1, Int2, Int4) it returns a slice of uint8 of
// with an adjusted length -- that is, enough bytes to hold those many bits/crumbs/nibbles.
func MakeAnySlice(dtype DType, length int) any {
	switch dtype {
	case Float64:
		return make([]float64, length)
	case Float32:
		return make([]float32, length)
	case Float16:
		return make([]float16.Float16, length)
	case BFloat16:
		return make([]bfloat16.BFloat16, length)
	case Int64:
		return make([]int64, length)
	case Int32:
		return make([]int32, length)
	case Int16:
		return make([]int16, length)
	case Int8:
		return make([]int8, length)
	case Uint64:
		return make([]uint64, length)
	case Uint32:
		return make([]uint32, length)
	case Uint16:
		return make([]uint16, length)
	case Uint8:
		return make([]uint8, length)
	case Bool:
		return make([]bool, length)
	case Complex64:
		return make([]complex64, length)
	case Complex128:
		return make([]complex128, length)
	case Int1, Uint1, Int2, Uint2, Int4, Uint4:
		// Sub-byte packed types are stored as uint8.
		// This allocates enough bytes to hold those many bits/crumbs/nibbles.
		return make([]uint8, dtype.SizeForDimensions(length))
	default:
		panicf("unsupported dtype for MakeAnySlice: %s", dtype)
	}
	panic(nil)
}

// CopyAnySlice copies the contents of src to dst, both should be slices of the same DType.
//
// Unsafe: dst and src must be slices of the same dtype.
func CopyAnySlice(dst, src any) {
	switch dst := dst.(type) {
	case []float64:
		copy(dst, src.([]float64))
	case []float32:
		copy(dst, src.([]float32))
	case []float16.Float16:
		copy(dst, src.([]float16.Float16))
	case []bfloat16.BFloat16:
		copy(dst, src.([]bfloat16.BFloat16))
	case []int64:
		copy(dst, src.([]int64))
	case []int32:
		copy(dst, src.([]int32))
	case []int16:
		copy(dst, src.([]int16))
	case []int8:
		copy(dst, src.([]int8))
	case []uint64:
		copy(dst, src.([]uint64))
	case []uint32:
		copy(dst, src.([]uint32))
	case []uint16:
		copy(dst, src.([]uint16))
	case []uint8:
		copy(dst, src.([]uint8))
	case []bool:
		copy(dst, src.([]bool))
	case []complex64:
		copy(dst, src.([]complex64))
	case []complex128:
		copy(dst, src.([]complex128))
	default:
		panicf("unsupported dtype for CopyAnySlices: %T", dst)
	}
}

// CloneAnySlice returns a shallow copy of src, which must be nil or a slice of a supported Go type.
// It panics if src is not nil and not a slice of a supported DType.
func CloneAnySlice(src any) any {
	if src == nil {
		return nil
	}
	switch s := src.(type) {
	case []float64:
		return slices.Clone(s)
	case []float32:
		return slices.Clone(s)
	case []float16.Float16:
		return slices.Clone(s)
	case []bfloat16.BFloat16:
		return slices.Clone(s)
	case []int:
		return slices.Clone(s)
	case []int64:
		return slices.Clone(s)
	case []int32:
		return slices.Clone(s)
	case []int16:
		return slices.Clone(s)
	case []int8:
		return slices.Clone(s)
	case []uint64:
		return slices.Clone(s)
	case []uint32:
		return slices.Clone(s)
	case []uint16:
		return slices.Clone(s)
	case []uint8:
		return slices.Clone(s)
	case []bool:
		return slices.Clone(s)
	case []complex64:
		return slices.Clone(s)
	case []complex128:
		return slices.Clone(s)
	default:
		panicf("unsupported dtype for CloneAnySlice: %T", src)
	}
	panic(nil)
}

// FromAnySlice returns the DType and length of a flat slice of any supported Go type without reflection.
// It returns an error if flatAny is not a slice of a supported DType.
func FromAnySlice(flatAny any) (dtype DType, length int, err error) {
	switch flat := flatAny.(type) {
	case []float64:
		return Float64, len(flat), nil
	case []float32:
		return Float32, len(flat), nil
	case []float16.Float16:
		return Float16, len(flat), nil
	case []bfloat16.BFloat16:
		return BFloat16, len(flat), nil
	case []int:
		if strconv.IntSize == 32 {
			return Int32, len(flat), nil
		}
		return Int64, len(flat), nil
	case []int64:
		return Int64, len(flat), nil
	case []int32:
		return Int32, len(flat), nil
	case []int16:
		return Int16, len(flat), nil
	case []int8:
		return Int8, len(flat), nil
	case []uint64:
		return Uint64, len(flat), nil
	case []uint32:
		return Uint32, len(flat), nil
	case []uint16:
		return Uint16, len(flat), nil
	case []uint8:
		return Uint8, len(flat), nil
	case []bool:
		return Bool, len(flat), nil
	case []complex64:
		return Complex64, len(flat), nil
	case []complex128:
		return Complex128, len(flat), nil
	default:
		return InvalidDType, 0, errors.Errorf("expected a slice of a supported DType, got %T", flatAny)
	}
}

// CheckAnySlice validates that flatAny is a slice matching expectedDType and expectedLen,
// and returns its underlying byte slice via UnsafeByteSliceFromAny.
func CheckAnySlice(flatAny any, expectedDType DType, expectedLen int) ([]byte, error) {
	dt, gotLen, err := FromAnySlice(flatAny)
	if err != nil {
		return nil, err
	}
	if expectedDType.IsPacked() {
		if dt != Uint8 {
			return nil, errors.Errorf("packed dtype %s expects []uint8 storage slice, got %s (%T)", expectedDType, dt, flatAny)
		}
		wantBytes := expectedDType.SizeForDimensions(expectedLen)
		if gotLen != wantBytes {
			return nil, errors.Errorf("packed dtype %s slice byte length %d does not match expected %d (for %d elements)",
				expectedDType, gotLen, wantBytes, expectedLen)
		}
		return UnsafeByteSliceFromAny(flatAny), nil
	}
	if dt != expectedDType {
		return nil, errors.Errorf("slice dtype %s (%T) does not match expected dtype %s", dt, flatAny, expectedDType)
	}
	if gotLen != expectedLen {
		return nil, errors.Errorf("slice length %d does not match expected length %d for dtype %s", gotLen, expectedLen, expectedDType)
	}
	return UnsafeByteSliceFromAny(flatAny), nil
}

// SafeCopyAnySlice copies src into dst after verifying that both are slices of the same supported DType
// and have the same length.
func SafeCopyAnySlice(dst, src any) error {
	dstDType, dstLen, err := FromAnySlice(dst)
	if err != nil {
		return errors.Wrap(err, "SafeCopyAnySlice(dst)")
	}
	srcDType, srcLen, err := FromAnySlice(src)
	if err != nil {
		return errors.Wrap(err, "SafeCopyAnySlice(src)")
	}
	if dstDType != srcDType {
		return errors.Errorf("SafeCopyAnySlice: type mismatch dst=%s (%T) vs src=%s (%T)", dstDType, dst, srcDType, src)
	}
	if dstLen != srcLen {
		return errors.Errorf("SafeCopyAnySlice: length mismatch dst=%d vs src=%d", dstLen, srcLen)
	}
	CopyAnySlice(dst, src)
	return nil
}

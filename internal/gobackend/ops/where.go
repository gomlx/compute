package ops

import (
	"slices"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
)

func init() {
	gobackend.RegisterWhere.Register(Where, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeWhere, gobackend.PriorityGeneric, execWhere)
}

// Where implements the compute.Builder interface.
func Where(f *gobackend.Function, conditionOp, onTrueOp, onFalseOp compute.Value) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("Where", conditionOp, onTrueOp, onFalseOp)
	if err != nil {
		return nil, err
	}
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]
	outputShape, err := shapeinference.Where(condition.Shape, onTrue.Shape, onFalse.Shape)
	if err != nil {
		return nil, err
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeWhere, outputShape, []*gobackend.Node{condition, onTrue, onFalse}, nil)
	return node, nil
}

func execWhere(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]

	// Figure out what the outputBuffer is going to be.
	outputShape := node.Shape

	var output *gobackend.Buffer
	var err error
	switch {
	case onTrue.RawShape.Equal(outputShape) && inputsOwned[1]:
		output = onTrue
		inputs[1] = nil
	case onFalse.RawShape.Equal(outputShape) && inputsOwned[2]:
		output = onFalse
		inputs[2] = nil
	default:
		output, err = backend.GetBuffer(outputShape)
		if err != nil {
			return nil, err
		}
	}
	if backend.NoOps {
		return output, nil
	}

	if dispatchWhereParallel(backend, condition, onTrue, onFalse, output) {
		return output, nil
	}

	tmpAny, tmpErr := whereDTypeMap.Get(outputShape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	fn := tmpAny.(func(conditionBuf, onTrueBuf, onFalseBuf, outputBuf *gobackend.Buffer))
	fn(condition, onTrue, onFalse, output)
	return output, nil
}

func dispatchWhereParallel(backend *gobackend.Backend, conditionBuf, onTrueBuf, onFalseBuf, outputBuf *gobackend.Buffer) bool {
	if conditionBuf.RawShape.IsScalar() {
		return false
	}
	outputShape := outputBuf.RawShape
	n := outputShape.Size()
	if backend == nil || backend.Workers == nil || !backend.Workers.IsEnabled() || n < 8192 {
		return false
	}
	if !conditionBuf.RawShape.Equal(outputShape) {
		return false
	}
	if !onTrueBuf.RawShape.Equal(outputShape) && !onTrueBuf.RawShape.IsScalar() {
		return false
	}
	if !onFalseBuf.RawShape.Equal(outputShape) && !onFalseBuf.RawShape.IsScalar() {
		return false
	}
	cond := conditionBuf.Flat.([]bool)
	switch outputBuf.RawShape.DType {
	case dtypes.Float32:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]float32))
	case dtypes.Float64:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]float64))
	case dtypes.BFloat16:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]bfloat16.BFloat16))
	case dtypes.Float16:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]float16.Float16))
	case dtypes.Int32:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]int32))
	case dtypes.Int64:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]int64))
	case dtypes.Int16:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]int16))
	case dtypes.Int8:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]int8))
	case dtypes.Uint32:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]uint32))
	case dtypes.Uint64:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]uint64))
	case dtypes.Uint16:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]uint16))
	case dtypes.Uint8:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]uint8))
	case dtypes.Bool:
		parallelWhere(backend, cond, onTrueBuf, onFalseBuf, outputBuf.Flat.([]bool))
	default:
		return false
	}
	return true
}

func parallelWhere[T any](backend *gobackend.Backend, cond []bool, onTrueBuf, onFalseBuf *gobackend.Buffer, out []T) {
	n := len(cond)
	onTrueIsScalar := onTrueBuf.RawShape.IsScalar()
	onFalseIsScalar := onFalseBuf.RawShape.IsScalar()
	onTrueFlat := onTrueBuf.Flat.([]T)
	onFalseFlat := onFalseBuf.Flat.([]T)
	var onTrueScalar, onFalseScalar T
	if onTrueIsScalar {
		onTrueScalar = onTrueFlat[0]
	}
	if onFalseIsScalar {
		onFalseScalar = onFalseFlat[0]
	}

	numWorkers := backend.Workers.AdjustedMaxParallelism()
	targetChunks := min(n, max(1, numWorkers*2))
	chunkSize := max(16384, (n+targetChunks-1)/targetChunks)
	var wg sync.WaitGroup

	_, isFloat32 := any(onTrueScalar).(float32)

	for start := 0; start < n; start += chunkSize {
		end := min(start+chunkSize, n)
		wg.Add(1)
		backend.Workers.WaitToStart(func() {
			if isFloat32 {
				var onTrueChunk, onFalseChunk []float32
				condChunk := cond[start:end]
				outChunk := any(out[start:end]).([]float32)
				if onTrueIsScalar {
					onTrueChunk = any(onTrueFlat[:1]).([]float32)
				} else {
					onTrueChunk = any(onTrueFlat[start:end]).([]float32)
				}
				if onFalseIsScalar {
					onFalseChunk = any(onFalseFlat[:1]).([]float32)
				} else {
					onFalseChunk = any(onFalseFlat[start:end]).([]float32)
				}
				if whereFloat32Arch(condChunk, onTrueChunk, onFalseChunk, outChunk, onTrueIsScalar, onFalseIsScalar) {
					wg.Done()
					return
				}
			}

			switch {
			case !onTrueIsScalar && onFalseIsScalar:
				for i := start; i < end; i++ {
					if cond[i] {
						out[i] = onTrueFlat[i]
					} else {
						out[i] = onFalseScalar
					}
				}
			case onTrueIsScalar && !onFalseIsScalar:
				for i := start; i < end; i++ {
					if cond[i] {
						out[i] = onTrueScalar
					} else {
						out[i] = onFalseFlat[i]
					}
				}
			case !onTrueIsScalar && !onFalseIsScalar:
				for i := start; i < end; i++ {
					if cond[i] {
						out[i] = onTrueFlat[i]
					} else {
						out[i] = onFalseFlat[i]
					}
				}
			default:
				for i := start; i < end; i++ {
					if cond[i] {
						out[i] = onTrueScalar
					} else {
						out[i] = onFalseScalar
					}
				}
			}
			wg.Done()
		})
	}
	wg.Wait()
}

// ternaryZipIterator iterates over the flat indices of three broadcast operands and the target buffer.
type ternaryZipIterator struct {
	tgtSize          int
	tgtDims          []int
	condStrides      []int
	trueStrides      []int
	falseStrides     []int
	condIsBroadcast  []bool
	trueIsBroadcast  []bool
	falseIsBroadcast []bool
	condIsScalar     bool
	trueIsScalar     bool
	falseIsScalar    bool
}

func newTernaryZipIterator(condShape, trueShape, falseShape, tgtShape shapes.Shape) *ternaryZipIterator {
	rank := tgtShape.Rank()
	zi := &ternaryZipIterator{
		tgtSize:          tgtShape.Size(),
		tgtDims:          slices.Clone(tgtShape.Dimensions),
		condIsScalar:     condShape.IsScalar(),
		trueIsScalar:     trueShape.IsScalar(),
		falseIsScalar:    falseShape.IsScalar(),
		condIsBroadcast:  make([]bool, rank),
		trueIsBroadcast:  make([]bool, rank),
		falseIsBroadcast: make([]bool, rank),
	}
	if !zi.condIsScalar {
		zi.condStrides = condShape.Strides()
		for axis := range rank {
			zi.condIsBroadcast[axis] = condShape.Dimensions[axis] != tgtShape.Dimensions[axis]
		}
	}
	if !zi.trueIsScalar {
		zi.trueStrides = trueShape.Strides()
		for axis := range rank {
			zi.trueIsBroadcast[axis] = trueShape.Dimensions[axis] != tgtShape.Dimensions[axis]
		}
	}
	if !zi.falseIsScalar {
		zi.falseStrides = falseShape.Strides()
		for axis := range rank {
			zi.falseIsBroadcast[axis] = falseShape.Dimensions[axis] != tgtShape.Dimensions[axis]
		}
	}
	return zi
}

//gobackend:dtypemap execWhereGeneric ints,uints,floats,half,bool
var whereDTypeMap = gobackend.NewDTypeMap("Where")

func execWhereGeneric[T gobackend.SupportedTypesConstraints](conditionBuf, onTrueBuf, onFalseBuf, outputBuf *gobackend.Buffer) {
	if conditionBuf.RawShape.IsScalar() {
		// Case 1: condition is a scalar, either we take onTrue or onFalse as a whole (with potential broadcast).
		if conditionBuf.Flat.([]bool)[0] {
			execWhereSetOutputWithValue[T](outputBuf, onTrueBuf)
		} else {
			execWhereSetOutputWithValue[T](outputBuf, onFalseBuf)
		}
		return
	}

	condShape := conditionBuf.RawShape
	trueShape := onTrueBuf.RawShape
	falseShape := onFalseBuf.RawShape
	tgtShape := outputBuf.RawShape

	// Fast path: condition matches target shape and value operands match or are scalar.
	if condShape.Equal(tgtShape) &&
		(trueShape.Equal(tgtShape) || trueShape.IsScalar()) &&
		(falseShape.Equal(tgtShape) || falseShape.IsScalar()) {
		condFlat := conditionBuf.Flat.([]bool)
		trueFlat := onTrueBuf.Flat.([]T)
		falseFlat := onFalseBuf.Flat.([]T)
		outputFlat := outputBuf.Flat.([]T)
		trueIsScalar := trueShape.IsScalar()
		falseIsScalar := falseShape.IsScalar()

		if _, ok := any(trueFlat[0]).(float32); ok {
			var trueChunk, falseChunk []float32
			if trueIsScalar {
				trueChunk = any(trueFlat[:1]).([]float32)
			} else {
				trueChunk = any(trueFlat).([]float32)
			}
			if falseIsScalar {
				falseChunk = any(falseFlat[:1]).([]float32)
			} else {
				falseChunk = any(falseFlat).([]float32)
			}
			if whereFloat32Arch(condFlat, trueChunk, falseChunk, any(outputFlat).([]float32), trueIsScalar, falseIsScalar) {
				return
			}
		}

		var tVal, fVal T
		if trueIsScalar {
			tVal = trueFlat[0]
		}
		if falseIsScalar {
			fVal = falseFlat[0]
		}
		for i, c := range condFlat {
			if c {
				if trueIsScalar {
					outputFlat[i] = tVal
				} else {
					outputFlat[i] = trueFlat[i]
				}
			} else {
				if falseIsScalar {
					outputFlat[i] = fVal
				} else {
					outputFlat[i] = falseFlat[i]
				}
			}
		}
		return
	}

	// General multi-dimensional broadcasting:
	zi := newTernaryZipIterator(condShape, trueShape, falseShape, tgtShape)
	condFlat := conditionBuf.Flat.([]bool)
	trueFlat := onTrueBuf.Flat.([]T)
	falseFlat := onFalseBuf.Flat.([]T)
	outFlat := outputBuf.Flat.([]T)

	rank := len(zi.tgtDims)
	trailingLen := 1
	splitAxis := -1
	for axis := rank - 1; axis >= 0; axis-- {
		condBroadcast := !zi.condIsScalar && zi.condIsBroadcast[axis]
		trueBroadcast := !zi.trueIsScalar && zi.trueIsBroadcast[axis]
		falseBroadcast := !zi.falseIsScalar && zi.falseIsBroadcast[axis]
		if condBroadcast || trueBroadcast || falseBroadcast {
			splitAxis = axis
			break
		}
		trailingLen *= zi.tgtDims[axis]
	}

	trueIsScalar := zi.trueIsScalar
	falseIsScalar := zi.falseIsScalar
	_, isFloat32 := any(trueFlat[0]).(float32)

	perAxesIdx := make([]int, rank)
	for dstIdx := 0; dstIdx < zi.tgtSize; dstIdx += trailingLen {
		condOffset, trueOffset, falseOffset := 0, 0, 0
		if !zi.condIsScalar {
			for a := 0; a <= splitAxis; a++ {
				idx := perAxesIdx[a]
				if zi.condIsBroadcast[a] {
					idx = 0
				}
				condOffset += idx * zi.condStrides[a]
			}
		}
		if !trueIsScalar {
			for a := 0; a <= splitAxis; a++ {
				idx := perAxesIdx[a]
				if zi.trueIsBroadcast[a] {
					idx = 0
				}
				trueOffset += idx * zi.trueStrides[a]
			}
		}
		if !falseIsScalar {
			for a := 0; a <= splitAxis; a++ {
				idx := perAxesIdx[a]
				if zi.falseIsBroadcast[a] {
					idx = 0
				}
				falseOffset += idx * zi.falseStrides[a]
			}
		}

		handled := false
		if isFloat32 {
			var tChunk, fChunk []float32
			if trueIsScalar {
				tChunk = any(trueFlat[:1]).([]float32)
			} else {
				tChunk = any(trueFlat[trueOffset : trueOffset+trailingLen]).([]float32)
			}
			if falseIsScalar {
				fChunk = any(falseFlat[:1]).([]float32)
			} else {
				fChunk = any(falseFlat[falseOffset : falseOffset+trailingLen]).([]float32)
			}
			cChunk := condFlat[condOffset : condOffset+trailingLen]
			oChunk := any(outFlat[dstIdx : dstIdx+trailingLen]).([]float32)
			handled = whereFloat32Arch(cChunk, tChunk, fChunk, oChunk, trueIsScalar, falseIsScalar)
		}

		if !handled {
			for j := 0; j < trailingLen; j++ {
				var tVal, fVal T
				if trueIsScalar {
					tVal = trueFlat[0]
				} else {
					tVal = trueFlat[trueOffset+j]
				}
				if falseIsScalar {
					fVal = falseFlat[0]
				} else {
					fVal = falseFlat[falseOffset+j]
				}
				if condFlat[condOffset+j] {
					outFlat[dstIdx+j] = tVal
				} else {
					outFlat[dstIdx+j] = fVal
				}
			}
		}

		if splitAxis >= 0 {
			for axis := splitAxis; axis >= 0; axis-- {
				perAxesIdx[axis]++
				if perAxesIdx[axis] < zi.tgtDims[axis] {
					break
				}
				perAxesIdx[axis] = 0
			}
		}
	}
}

func execWhereSetOutputWithValue[T gobackend.SupportedTypesConstraints](outputBuf, valueBuf *gobackend.Buffer) {
	if valueBuf == outputBuf {
		// The output is reusing the value buffer, nothing to do.
		return
	}
	outputSlice := outputBuf.Flat.([]T)
	valSlice := valueBuf.Flat.([]T)
	if valueBuf.RawShape.Equal(outputBuf.RawShape) {
		// Copy over values.
		copy(outputSlice, valSlice)
		return
	}
	if valueBuf.RawShape.IsScalar() {
		c := valSlice[0]
		for outputIdx := range outputSlice {
			outputSlice[outputIdx] = c
		}
		return
	}
	// General broadcast from valueBuf to outputBuf.
	zi := gobackend.NewZippedBroadcastIterator(valueBuf.RawShape, valueBuf.RawShape, outputBuf.RawShape)
	for idxs := range zi.IterFlatIndices() {
		outputSlice[idxs.TgtFlatIdx] = valSlice[idxs.LHSFlatIdx]
	}
}

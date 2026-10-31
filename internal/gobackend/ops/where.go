package ops

import (
	"slices"

	"github.com/gomlx/compute"
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

	tmpAny, tmpErr := whereDTypeMap.Get(outputShape.DType)
	if tmpErr != nil {
		panic(tmpErr)
	}
	fn := tmpAny.(func(backend *gobackend.Backend, conditionBuf, onTrueBuf, onFalseBuf, outputBuf *gobackend.Buffer))
	fn(backend, condition, onTrue, onFalse, output)
	return output, nil
}

// where2DConfig describes a Where operation whose axes collapse into at most 2 broadcast segments
// [outerSize, innerSize].
type where2DConfig struct {
	numSegs         int // 1 (1D flat) or 2 (2D [outerSize, innerSize])
	outerSize       int
	innerSize       int
	condBcastOuter  bool
	trueBcastOuter  bool
	falseBcastOuter bool
	condBcastInner  bool
	trueBcastInner  bool
	falseBcastInner bool
}

func determineWhere2DConfig(condShape, trueShape, falseShape, tgtShape shapes.Shape) (where2DConfig, bool) {
	n := tgtShape.Size()
	if n == 0 {
		return where2DConfig{numSegs: 1, outerSize: 1, innerSize: 0}, true
	}
	var segDims [2]int
	var segMasks [2]int
	numSegs := 0

	condScalar := condShape.IsScalar()
	trueScalar := trueShape.IsScalar()
	falseScalar := falseShape.IsScalar()

	for axis, d := range tgtShape.Dimensions {
		if d <= 1 {
			continue
		}
		mask := 0
		if condScalar || condShape.Dimensions[axis] == 1 {
			mask |= 4
		}
		if trueScalar || trueShape.Dimensions[axis] == 1 {
			mask |= 2
		}
		if falseScalar || falseShape.Dimensions[axis] == 1 {
			mask |= 1
		}
		if numSegs > 0 && segMasks[numSegs-1] == mask {
			segDims[numSegs-1] *= d
		} else if numSegs < 2 {
			segDims[numSegs] = d
			segMasks[numSegs] = mask
			numSegs++
		} else {
			return where2DConfig{}, false
		}
	}

	if numSegs == 0 {
		return where2DConfig{
			numSegs:         1,
			outerSize:       1,
			innerSize:       1,
			condBcastInner:  false,
			trueBcastInner:  true,
			falseBcastInner: true,
		}, true
	}
	if numSegs == 1 {
		m := segMasks[0]
		return where2DConfig{
			numSegs:         1,
			outerSize:       1,
			innerSize:       segDims[0],
			condBcastInner:  (m & 4) != 0,
			trueBcastInner:  (m & 2) != 0,
			falseBcastInner: (m & 1) != 0,
		}, true
	}
	m0, m1 := segMasks[0], segMasks[1]
	return where2DConfig{
		numSegs:         2,
		outerSize:       segDims[0],
		innerSize:       segDims[1],
		condBcastOuter:  (m0 & 4) != 0,
		trueBcastOuter:  (m0 & 2) != 0,
		falseBcastOuter: (m0 & 1) != 0,
		condBcastInner:  (m1 & 4) != 0,
		trueBcastInner:  (m1 & 2) != 0,
		falseBcastInner: (m1 & 1) != 0,
	}, true
}

func whereRow[T any](
	cond []bool,
	trueSlice, falseSlice, out []T,
	trueIsScalar, falseIsScalar bool,
	outAliasesTrue, outAliasesFalse bool,
	isFloat32 bool,
) {
	if isFloat32 {
		if whereFloat32Arch(
			cond,
			any(trueSlice).([]float32),
			any(falseSlice).([]float32),
			any(out).([]float32),
			trueIsScalar,
			falseIsScalar,
		) {
			return
		}
	}

	switch {
	case outAliasesFalse && trueIsScalar:
		tVal := trueSlice[0]
		for i, c := range cond {
			if c {
				out[i] = tVal
			}
		}
	case outAliasesFalse && !trueIsScalar:
		for i, c := range cond {
			if c {
				out[i] = trueSlice[i]
			}
		}
	case outAliasesTrue && falseIsScalar:
		fVal := falseSlice[0]
		for i, c := range cond {
			if !c {
				out[i] = fVal
			}
		}
	case outAliasesTrue && !falseIsScalar:
		for i, c := range cond {
			if !c {
				out[i] = falseSlice[i]
			}
		}
	case !trueIsScalar && falseIsScalar:
		fVal := falseSlice[0]
		for i, c := range cond {
			if c {
				out[i] = trueSlice[i]
			} else {
				out[i] = fVal
			}
		}
	case trueIsScalar && !falseIsScalar:
		tVal := trueSlice[0]
		for i, c := range cond {
			if c {
				out[i] = tVal
			} else {
				out[i] = falseSlice[i]
			}
		}
	case !trueIsScalar && !falseIsScalar:
		for i, c := range cond {
			if c {
				out[i] = trueSlice[i]
			} else {
				out[i] = falseSlice[i]
			}
		}
	default:
		tVal := trueSlice[0]
		fVal := falseSlice[0]
		for i, c := range cond {
			if c {
				out[i] = tVal
			} else {
				out[i] = fVal
			}
		}
	}
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

func execWhereGeneric[T gobackend.SupportedTypesConstraints](backend *gobackend.Backend, conditionBuf, onTrueBuf, onFalseBuf, outputBuf *gobackend.Buffer) {
	if conditionBuf.RawShape.Size() == 1 {
		// Case 1: condition has a single boolean element, either we take onTrue or onFalse as a whole (with potential broadcast).
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

	condFlat := conditionBuf.Flat.([]bool)
	trueFlat := onTrueBuf.Flat.([]T)
	falseFlat := onFalseBuf.Flat.([]T)
	outFlat := outputBuf.Flat.([]T)
	if len(outFlat) == 0 {
		return
	}
	_, isFloat32 := any(trueFlat[0]).(float32)
	outAliasesTrue := outputBuf == onTrueBuf
	outAliasesFalse := outputBuf == onFalseBuf

	// Fast path: 1D flat or 2D-collapsible broadcast [outerSize, innerSize].
	if cfg, ok := determineWhere2DConfig(condShape, trueShape, falseShape, tgtShape); ok {
		canParallel := backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && len(outFlat) >= 8192
		if cfg.numSegs == 1 {
			n := cfg.innerSize
			runChunk := func(start, end int) {
				var tChunk, fChunk []T
				if cfg.trueBcastInner {
					tChunk = trueFlat[:1]
				} else {
					tChunk = trueFlat[start:end]
				}
				if cfg.falseBcastInner {
					fChunk = falseFlat[:1]
				} else {
					fChunk = falseFlat[start:end]
				}
				whereRow(condFlat[start:end], tChunk, fChunk, outFlat[start:end],
					cfg.trueBcastInner, cfg.falseBcastInner, outAliasesTrue, outAliasesFalse, isFloat32)
			}
			if canParallel {
				backend.Scheduler.ParallelizeChunks(n, 16384, runChunk)
			} else {
				runChunk(0, n)
			}
			return
		}

		// 2-segment [outerSize, innerSize] broadcast.
		outerSize, innerSize := cfg.outerSize, cfg.innerSize
		if canParallel && outerSize < 4 && innerSize >= 16384 && !cfg.condBcastInner {
			// Few large rows: parallelize within each row.
			for r := range outerSize {
				condR0, trueR0, falseR0 := r, r, r
				if cfg.condBcastOuter {
					condR0 = 0
				}
				if cfg.trueBcastOuter {
					trueR0 = 0
				}
				if cfg.falseBcastOuter {
					falseR0 = 0
				}
				condBase := condR0 * innerSize
				outBase := r * innerSize
				backend.Scheduler.ParallelizeChunks(innerSize, 16384, func(start, end int) {
					var tChunk, fChunk []T
					if cfg.trueBcastInner {
						tChunk = trueFlat[trueR0 : trueR0+1]
					} else {
						tChunk = trueFlat[trueR0*innerSize+start : trueR0*innerSize+end]
					}
					if cfg.falseBcastInner {
						fChunk = falseFlat[falseR0 : falseR0+1]
					} else {
						fChunk = falseFlat[falseR0*innerSize+start : falseR0*innerSize+end]
					}
					whereRow(condFlat[condBase+start:condBase+end], tChunk, fChunk, outFlat[outBase+start:outBase+end],
						cfg.trueBcastInner, cfg.falseBcastInner, outAliasesTrue, outAliasesFalse, isFloat32)
				})
			}
			return
		}

		runRows := func(startRow, endRow int) {
			for r := startRow; r < endRow; r++ {
				condR0, trueR0, falseR0 := r, r, r
				if cfg.condBcastOuter {
					condR0 = 0
				}
				if cfg.trueBcastOuter {
					trueR0 = 0
				}
				if cfg.falseBcastOuter {
					falseR0 = 0
				}
				outRow := outFlat[r*innerSize : (r+1)*innerSize]
				if cfg.condBcastInner {
					if condFlat[condR0] {
						if !outAliasesTrue {
							if cfg.trueBcastInner {
								tVal := trueFlat[trueR0]
								for j := range outRow {
									outRow[j] = tVal
								}
							} else {
								copy(outRow, trueFlat[trueR0*innerSize:(trueR0+1)*innerSize])
							}
						}
					} else {
						if !outAliasesFalse {
							if cfg.falseBcastInner {
								fVal := falseFlat[falseR0]
								for j := range outRow {
									outRow[j] = fVal
								}
							} else {
								copy(outRow, falseFlat[falseR0*innerSize:(falseR0+1)*innerSize])
							}
						}
					}
					continue
				}

				var tRow, fRow []T
				if cfg.trueBcastInner {
					tRow = trueFlat[trueR0 : trueR0+1]
				} else {
					tRow = trueFlat[trueR0*innerSize : (trueR0+1)*innerSize]
				}
				if cfg.falseBcastInner {
					fRow = falseFlat[falseR0 : falseR0+1]
				} else {
					fRow = falseFlat[falseR0*innerSize : (falseR0+1)*innerSize]
				}
				whereRow(condFlat[condR0*innerSize:(condR0+1)*innerSize], tRow, fRow, outRow,
					cfg.trueBcastInner, cfg.falseBcastInner, outAliasesTrue, outAliasesFalse, isFloat32)
			}
		}
		if canParallel && outerSize > 1 {
			rowsPerChunk := max(1, 16384/innerSize)
			backend.Scheduler.ParallelizeChunks(outerSize, rowsPerChunk, runRows)
		} else {
			runRows(0, outerSize)
		}
		return
	}

	// General multi-dimensional broadcasting (>2 segments):
	zi := newTernaryZipIterator(condShape, trueShape, falseShape, tgtShape)

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

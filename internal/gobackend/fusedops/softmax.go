package fusedops

import (
	"math"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/fastmath"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedSoftmax.Register(FusedSoftmax, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedSoftmax, gobackend.PriorityTyped, execFusedSoftmax)
}

type nodeFusedSoftmax struct {
	axis int
}

func (d *nodeFusedSoftmax) EqualNodeData(other gobackend.NodeDataComparable) bool {
	return d.axis == other.(*nodeFusedSoftmax).axis
}

// FusedSoftmax computes softmax along the specified axis.
// The axis must be non-negative (the caller normalizes negative indices).
func FusedSoftmax(f *gobackend.Function, x compute.Value, axis int) (compute.Value, error) {
	inputs, err := f.VerifyAndCastValues("FusedSoftmax", x)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	rank := xNode.Shape.Rank()
	if axis < 0 || axis >= rank {
		return nil, errors.Errorf("FusedSoftmax: axis %d out of range for rank %d", axis, rank)
	}

	data := &nodeFusedSoftmax{axis: axis}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedSoftmax, xNode.Shape.Clone(), []*gobackend.Node{xNode}, data)
	return node, nil
}

// FusedSoftmax =====================================================================================================

// execFusedSoftmax implements optimized softmax with better cache locality.
// Three passes over the axis: find max, compute exp(x-max) and sum, then normalize.
// Float32 uses fastmath.Exp32; float64 uses math.Exp.
func execFusedSoftmax(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedSoftmax)
	axis := data.axis
	input := inputs[0]
	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}

	switch input.RawShape.DType {
	case dtypes.Float32:
		fusedSoftmax(backend, input.Flat.([]float32), output.Flat.([]float32), axis, node.Shape, fastmath.Exp32)
	case dtypes.Float64:
		fusedSoftmax(backend, input.Flat.([]float64), output.Flat.([]float64), axis, node.Shape, math.Exp)
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedSoftmax: dtype %s", input.RawShape.DType)
	}
	return output, nil
}

// fusedSoftmaxComputeAxisStrides returns the outer size, axis size, and inner size for iterating
// over an axis of the given shape. This decomposition allows softmax (and similar
// axis-based ops) to operate on any axis.
func fusedSoftmaxComputeAxisStrides(shape shapes.Shape, axis int) (outerSize, axisSize, innerSize int) {
	dims := shape.Dimensions
	outerSize = 1
	for i := range axis {
		outerSize *= dims[i]
	}
	axisSize = dims[axis]
	innerSize = 1
	for i := axis + 1; i < len(dims); i++ {
		innerSize *= dims[i]
	}
	return
}

func fusedSoftmax[T float32 | float64](backend *gobackend.Backend, input, output []T, axis int, shape shapes.Shape, expFn func(T) T) {
	outerSize, axisSize, innerSize := fusedSoftmaxComputeAxisStrides(shape, axis)
	totalElements := outerSize * axisSize * innerSize

	processOuter := func(start, end int) {
		for outer := start; outer < end; outer++ {
			for inner := range innerSize {
				baseIdx := outer*axisSize*innerSize + inner

				// Pass 1: Find max.
				maxVal := T(math.Inf(-1))
				for i := range axisSize {
					idx := baseIdx + i*innerSize
					if input[idx] > maxVal {
						maxVal = input[idx]
					}
				}

				// Pass 2: Exp and sum.
				var sum T
				for i := range axisSize {
					idx := baseIdx + i*innerSize
					output[idx] = expFn(input[idx] - maxVal)
					sum += output[idx]
				}

				// Pass 3: Normalize.
				invSum := 1.0 / sum
				for i := range axisSize {
					idx := baseIdx + i*innerSize
					output[idx] *= invSum
				}
			}
		}
	}

	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && outerSize > 1 && totalElements > 16384 {
		numWorkers := backend.Workers.AdjustedMaxParallelism()
		targetChunks := min(outerSize, max(1, numWorkers*2))
		outerPerChunk := max(1, (outerSize+targetChunks-1)/targetChunks)
		var wg sync.WaitGroup
		for start := 0; start < outerSize; start += outerPerChunk {
			end := min(start+outerPerChunk, outerSize)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				processOuter(start, end)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		processOuter(0, outerSize)
	}
}

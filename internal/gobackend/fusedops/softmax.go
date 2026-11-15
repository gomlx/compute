package fusedops

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
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
func execFusedSoftmax(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedSoftmax)
	axis := data.axis
	input := inputs[0]
	var output *gobackend.Buffer
	var err error
	if len(inputsOwned) > 0 && inputsOwned[0] && input.RawShape.Equal(node.Shape) {
		output = input
		inputs[0] = nil
	} else {
		output, err = backend.GetBuffer(node.Shape)
		if err != nil {
			return nil, err
		}
	}
	if backend.NoOps {
		return output, nil
	}

	switch input.RawShape.DType {
	case dtypes.Float32:
		fusedSoftmax(backend, input.Flat.([]float32), output.Flat.([]float32), axis, node.Shape, fastmath.Exp32)
	case dtypes.Float64:
		fusedSoftmax(backend, input.Flat.([]float64), output.Flat.([]float64), axis, node.Shape, math.Exp)
	case dtypes.Float16:
		fusedSoftmaxHalf[float16.Float16, *float16.Float16](backend, input.Flat.([]float16.Float16), output.Flat.([]float16.Float16), axis, node.Shape)
	case dtypes.BFloat16:
		fusedSoftmaxHalf[bfloat16.BFloat16, *bfloat16.BFloat16](backend, input.Flat.([]bfloat16.BFloat16), output.Flat.([]bfloat16.BFloat16), axis, node.Shape)
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedSoftmax: dtype %s", input.RawShape.DType)
	}
	return output, nil
}

// fusedSoftmaxHalf executes softmax for half-precision inputs/outputs using Float32
// for internal max-subtraction, exp, sum, and normalization.
// Reading the whole row into a per-worker Float32 buffer makes in-place (input == output) execution safe.
func fusedSoftmaxHalf[T gotype.HalfPrecision[T], P gotype.HalfPrecisionPtr[T]](
	backend *gobackend.Backend, input, output []T, axis int, shape shapes.Shape,
) {
	outerSize, axisSize, innerSize := fusedSoftmaxComputeAxisStrides(shape, axis)
	totalElements := outerSize * axisSize * innerSize
	if outerSize == 0 || axisSize == 0 || innerSize == 0 {
		return
	}

	processOuter := func(start, end int) {
		rowBuf := make([]float32, axisSize)
		for outer := start; outer < end; outer++ {
			for inner := range innerSize {
				baseIdx := outer*axisSize*innerSize + inner

				// Pass 1: Convert row to Float32 and find max.
				maxVal := float32(math.Inf(-1))
				for i := range axisSize {
					v := input[baseIdx+i*innerSize].Float32()
					rowBuf[i] = v
					if v > maxVal {
						maxVal = v
					}
				}

				// Pass 2: Exp and sum in Float32.
				var sum float32
				for i := range axisSize {
					e := fastmath.Exp32(rowBuf[i] - maxVal)
					rowBuf[i] = e
					sum += e
				}

				// Pass 3: Normalize in Float32 and round back to half-precision.
				invSum := float32(1.0) / sum
				for i := range axisSize {
					P(&output[baseIdx+i*innerSize]).SetFloat32(rowBuf[i] * invSum)
				}
			}
		}
	}

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && outerSize > 1 && totalElements > 16384 {
		minOuter := max(1, 16384/max(1, axisSize*innerSize))
		backend.Scheduler.ParallelizeChunks(outerSize, minOuter, processOuter)
	} else {
		processOuter(0, outerSize)
	}
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

	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && outerSize > 1 && totalElements > 16384 {
		minOuter := max(1, 16384/max(1, axisSize*innerSize))
		backend.Scheduler.ParallelizeChunks(outerSize, minOuter, processOuter)
	} else {
		processOuter(0, outerSize)
	}
}

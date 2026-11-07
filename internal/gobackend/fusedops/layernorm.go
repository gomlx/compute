package fusedops

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// FusedLayerNorm applies layer normalization.

func init() {
	gobackend.RegisterFusedLayerNorm.Register(FusedLayerNorm, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedLayerNorm, gobackend.PriorityTyped, execFusedLayerNorm)
}

// nodeFusedLayerNorm is node data for FusedLayerNorm.
type nodeFusedLayerNorm struct {
	axes        []int
	epsilon     float64
	hasResidual bool
}

var _ gobackend.FusedLayerNormResidualTarget = (*nodeFusedLayerNorm)(nil)

func (d *nodeFusedLayerNorm) TryFuseResidualInput() bool {
	if d == nil || d.hasResidual {
		return false
	}
	d.hasResidual = true
	return true
}

func (d *nodeFusedLayerNorm) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedLayerNorm)
	if d.epsilon != o.epsilon || d.hasResidual != o.hasResidual || len(d.axes) != len(o.axes) {
		return false
	}
	for i, a := range d.axes {
		if a != o.axes[i] {
			return false
		}
	}
	return true
}

// FusedLayerNorm applies layer normalization.
func FusedLayerNorm(f *gobackend.Function, x compute.Value, axes []int, epsilon float64, gamma, beta compute.Value) (compute.Value, error) {
	values := []compute.Value{x}
	if gamma != nil {
		values = append(values, gamma)
	}
	if beta != nil {
		values = append(values, beta)
	}
	inputs, err := f.VerifyAndCastValues("FusedLayerNorm", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	// Normalize negative axes.
	rank := xNode.Shape.Rank()
	normalizedAxes := make([]int, len(axes))
	for i, a := range axes {
		if a < 0 {
			a += rank
		}
		if a < 0 || a >= rank {
			return nil, errors.Errorf("FusedLayerNorm: axis %d out of range for rank %d", axes[i], rank)
		}
		normalizedAxes[i] = a
	}

	data := &nodeFusedLayerNorm{axes: normalizedAxes, epsilon: epsilon}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedLayerNorm, xNode.Shape.Clone(), inputs, data)
	return node, nil
}

// execFusedLayerNorm implements layer normalization.
// For each sample: y = (x - mean) / sqrt(var + epsilon) * gamma + beta
func execFusedLayerNorm(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedLayerNorm)
	input := inputs[0]

	paramInputs := inputs
	var residual *gobackend.Buffer
	resInputIdx := -1
	if data.hasResidual && len(inputs) > 1 {
		resInputIdx = len(inputs) - 1
		residual = inputs[resInputIdx]
		paramInputs = inputs[:resInputIdx]
	}

	var output *gobackend.Buffer
	var err error
	if len(inputsOwned) > 0 && inputsOwned[0] && input.RawShape.Equal(node.Shape) {
		output = input
		inputs[0] = nil
	} else if resInputIdx >= 0 && len(inputsOwned) > resInputIdx && inputsOwned[resInputIdx] && residual.RawShape.Equal(node.Shape) {
		output = residual
		inputs[resInputIdx] = nil
	} else {
		output, err = backend.GetBuffer(node.Shape)
		if err != nil {
			return nil, err
		}
	}
	if backend.NoOps {
		return output, nil
	}

	// Determine gamma/beta. paramInputs[0]=x, paramInputs[1]=gamma (optional), paramInputs[2]=beta (optional).
	var gamma, beta *gobackend.Buffer
	if len(paramInputs) > 1 {
		gamma = paramInputs[1]
	}
	if len(paramInputs) > 2 {
		beta = paramInputs[2]
	}

	switch input.RawShape.DType {
	case dtypes.Float32:
		layerNorm[float32](backend, input, residual, output, gamma, beta, data.axes, data.epsilon)
	case dtypes.Float64:
		layerNorm[float64](backend, input, residual, output, gamma, beta, data.axes, data.epsilon)
	default:
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedLayerNorm: dtype %s", input.RawShape.DType)
	}
	return output, nil
}

const minLayerNormParallelElements = 4096

// layerNorm dispatches to the trailing-axes fast path or the general case.
func layerNorm[T float32 | float64](backend *gobackend.Backend, input, residual, output, gamma, beta *gobackend.Buffer, axes []int, epsilon float64) {
	inData := input.Flat.([]T)
	outData := output.Flat.([]T)
	var resData []T
	if residual != nil {
		resData = residual.Flat.([]T)
	}
	dims := input.RawShape.Dimensions
	rank := len(dims)

	normSize := 1
	for _, a := range axes {
		normSize *= dims[a]
	}

	// Check for trailing axes fast path.
	isTrailingAxes := true
	for i, a := range axes {
		if a != rank-len(axes)+i {
			isTrailingAxes = false
			break
		}
	}

	var gammaData, betaData []T
	if gamma != nil {
		gammaData = gamma.Flat.([]T)
	}
	if beta != nil {
		betaData = beta.Flat.([]T)
	}

	if isTrailingAxes {
		outerSize := len(inData) / normSize
		totalElements := len(inData)
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && outerSize > 1 && totalElements > minLayerNormParallelElements {
			minOuter := max(1, minLayerNormParallelElements/max(1, normSize))
			backend.Scheduler.ParallelizeChunks(outerSize, minOuter, func(rStart, rEnd int) {
				rowOffset := rStart * normSize
				endOffset := rEnd * normSize
				srcSlice := inData[rowOffset:endOffset]
				outSlice := outData[rowOffset:endOffset]
				if resData != nil {
					resSlice := resData[rowOffset:endOffset]
					for i := range outSlice {
						outSlice[i] = srcSlice[i] + resSlice[i]
					}
					srcSlice = outSlice
				}
				layerNormTrailingAxes(srcSlice, outSlice, gammaData, betaData, normSize, epsilon)
			})
		} else {
			srcSlice := inData
			if resData != nil {
				for i := range outData {
					outData[i] = inData[i] + resData[i]
				}
				srcSlice = outData
			}
			layerNormTrailingAxes(srcSlice, outData, gammaData, betaData, normSize, epsilon)
		}
	} else if aDim, bDim, cDim, dDim, ok := classifyStridedLayerNormAxes(dims, axes); ok {
		totalElements := len(inData)
		sliceSize := bDim * cDim * dDim
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && aDim > 1 && totalElements > minLayerNormParallelElements {
			minA := max(1, minLayerNormParallelElements/max(1, sliceSize))
			backend.Scheduler.ParallelizeChunks(aDim, minA, func(aStart, aEnd int) {
				inSlice := inData[aStart*sliceSize : aEnd*sliceSize]
				outSlice := outData[aStart*sliceSize : aEnd*sliceSize]
				if resData != nil {
					resSlice := resData[aStart*sliceSize : aEnd*sliceSize]
					for i := range outSlice {
						outSlice[i] = inSlice[i] + resSlice[i]
					}
					inSlice = outSlice
				}
				layerNormStridedAxes(inSlice, outSlice, gammaData, betaData, aEnd-aStart, bDim, cDim, dDim, epsilon)
			})
		} else {
			srcSlice := inData
			if resData != nil {
				for i := range outData {
					outData[i] = inData[i] + resData[i]
				}
				srcSlice = outData
			}
			layerNormStridedAxes(srcSlice, outData, gammaData, betaData, aDim, bDim, cDim, dDim, epsilon)
		}
	} else {
		srcSlice := inData
		if resData != nil {
			for i := range outData {
				outData[i] = inData[i] + resData[i]
			}
			srcSlice = outData
		}
		layerNormArbitraryAxes(srcSlice, outData, gammaData, betaData, dims, axes, normSize, epsilon)
	}
}

// classifyStridedLayerNormAxes checks if dims and axes can be collapsed into [A, B, C, D]
// where A and C are kept dimensions and B and D are normalized dimensions (normSize = B * D).
// This covers [K, R, K] (D=1), [R, K] (A=1, D=1), [K, R, K, R] (e.g. 5D GroupNorm [B, H, W, G, C/G] on [1, 2, 4]),
// and [R, K, R] (A=1).
func classifyStridedLayerNormAxes(dims, axes []int) (aDim, bDim, cDim, dDim int, ok bool) {
	rank := len(dims)
	if rank == 0 || len(axes) == 0 {
		return 0, 0, 0, 0, false
	}
	// Verify axes are in ascending order so gamma/beta indexing matches lexicographical order.
	for i := 1; i < len(axes); i++ {
		if axes[i] <= axes[i-1] {
			return 0, 0, 0, 0, false
		}
	}
	isNorm := make([]bool, rank)
	for _, a := range axes {
		isNorm[a] = true
	}

	aDim, bDim, cDim, dDim = 1, 1, 1, 1
	phase := 0 // 0: kept(A), 1: norm(B), 2: kept(C), 3: norm(D)
	for axis, dim := range dims {
		norm := isNorm[axis]
		switch phase {
		case 0:
			if !norm {
				aDim *= dim
			} else {
				phase = 1
				bDim *= dim
			}
		case 1:
			if norm {
				bDim *= dim
			} else {
				phase = 2
				cDim *= dim
			}
		case 2:
			if !norm {
				cDim *= dim
			} else {
				phase = 3
				dDim *= dim
			}
		case 3:
			if norm {
				dDim *= dim
			} else {
				return 0, 0, 0, 0, false
			}
		}
	}
	if phase < 2 {
		return 0, 0, 0, 0, false
	}
	return aDim, bDim, cDim, dDim, true
}

// layerNormStridedAxes executes LayerNorm on a collapsed [A, B, C, D] tensor where
// B and D are the normalized axes (normSize = B * D) and A, C are kept axes.
func layerNormStridedAxes[T float32 | float64](
	inData, outData, gammaData, betaData []T,
	aDim, bDim, cDim, dDim int,
	epsilon float64,
) {
	normSize := bDim * dDim
	if normSize == 0 || aDim == 0 || cDim == 0 {
		return
	}
	normSizeF := T(normSize)
	strideBCD := bDim * cDim * dDim
	strideCD := cDim * dDim

	var stackMeans [256]T
	var stackVars [256]T
	var means, vars []T
	if cDim <= len(stackMeans) {
		means = stackMeans[:cDim]
		vars = stackVars[:cDim]
	} else {
		means = make([]T, cDim)
		vars = make([]T, cDim)
	}

	for a := range aDim {
		baseA := a * strideBCD
		clear(means)
		clear(vars)

		// Pass 1: accumulate sums per c across B * D.
		for b := range bDim {
			baseB := baseA + b*strideCD
			for c := range cDim {
				row := inData[baseB+c*dDim : baseB+(c+1)*dDim]
				var s T
				for _, v := range row {
					s += v
				}
				means[c] += s
			}
		}
		for c := range cDim {
			means[c] /= normSizeF
		}

		// Pass 2: accumulate variance sums per c across B * D.
		for b := range bDim {
			baseB := baseA + b*strideCD
			for c := range cDim {
				mean := means[c]
				row := inData[baseB+c*dDim : baseB+(c+1)*dDim]
				var vSum T
				for _, v := range row {
					diff := v - mean
					vSum += diff * diff
				}
				vars[c] += vSum
			}
		}
		for c := range cDim {
			vars[c] = T(1.0 / math.Sqrt(float64(vars[c]/normSizeF)+epsilon))
		}

		// Pass 3: normalize and apply optional gamma/beta.
		switch {
		case gammaData != nil && betaData != nil:
			for b := range bDim {
				baseB := baseA + b*strideCD
				gbBase := b * dDim
				gRow := gammaData[gbBase : gbBase+dDim]
				bRow := betaData[gbBase : gbBase+dDim]
				for c := range cDim {
					mean := means[c]
					invStd := vars[c]
					inRow := inData[baseB+c*dDim : baseB+(c+1)*dDim]
					outRow := outData[baseB+c*dDim : baseB+(c+1)*dDim]
					for d, v := range inRow {
						outRow[d] = (v-mean)*invStd*gRow[d] + bRow[d]
					}
				}
			}
		case gammaData != nil:
			for b := range bDim {
				baseB := baseA + b*strideCD
				gbBase := b * dDim
				gRow := gammaData[gbBase : gbBase+dDim]
				for c := range cDim {
					mean := means[c]
					invStd := vars[c]
					inRow := inData[baseB+c*dDim : baseB+(c+1)*dDim]
					outRow := outData[baseB+c*dDim : baseB+(c+1)*dDim]
					for d, v := range inRow {
						outRow[d] = (v - mean) * invStd * gRow[d]
					}
				}
			}
		case betaData != nil:
			for b := range bDim {
				baseB := baseA + b*strideCD
				gbBase := b * dDim
				bRow := betaData[gbBase : gbBase+dDim]
				for c := range cDim {
					mean := means[c]
					invStd := vars[c]
					inRow := inData[baseB+c*dDim : baseB+(c+1)*dDim]
					outRow := outData[baseB+c*dDim : baseB+(c+1)*dDim]
					for d, v := range inRow {
						outRow[d] = (v-mean)*invStd + bRow[d]
					}
				}
			}
		default:
			for b := range bDim {
				baseB := baseA + b*strideCD
				for c := range cDim {
					mean := means[c]
					invStd := vars[c]
					inRow := inData[baseB+c*dDim : baseB+(c+1)*dDim]
					outRow := outData[baseB+c*dDim : baseB+(c+1)*dDim]
					for d, v := range inRow {
						outRow[d] = (v - mean) * invStd
					}
				}
			}
		}
	}
}

// layerNormTrailingAxes handles the common case where normalization axes are the last N axes.
// Each contiguous block of normSize elements is one normalization group.
func layerNormTrailingAxes[T float32 | float64](
	inData, outData, gammaData, betaData []T,
	normSize int,
	epsilon float64,
) {
	normSizeF := T(normSize)
	outerSize := len(inData) / normSize

	for outer := range outerSize {
		base := outer * normSize
		inRow := inData[base : base+normSize]
		outRow := outData[base : base+normSize]

		// Compute mean.
		var sum T
		for _, v := range inRow {
			sum += v
		}
		mean := sum / normSizeF

		// Compute variance.
		var varSum T
		for _, v := range inRow {
			diff := v - mean
			varSum += diff * diff
		}
		variance := varSum / normSizeF
		invStd := T(1.0 / math.Sqrt(float64(variance)+epsilon))

		// Normalize and apply scale/offset.
		switch {
		case gammaData != nil && betaData != nil:
			_ = gammaData[normSize-1]
			_ = betaData[normSize-1]
			for i, v := range inRow {
				outRow[i] = (v-mean)*invStd*gammaData[i] + betaData[i]
			}
		case gammaData != nil:
			_ = gammaData[normSize-1]
			for i, v := range inRow {
				outRow[i] = (v - mean) * invStd * gammaData[i]
			}
		case betaData != nil:
			_ = betaData[normSize-1]
			for i, v := range inRow {
				outRow[i] = (v-mean)*invStd + betaData[i]
			}
		default:
			for i, v := range inRow {
				outRow[i] = (v - mean) * invStd
			}
		}
	}
}

// layerNormArbitraryAxes handles normalization over arbitrary (non-trailing) axes
// using Shape.IterOnAxes for index iteration.
func layerNormArbitraryAxes[T float32 | float64](
	inData, outData, gammaData, betaData []T,
	dims, axes []int,
	normSize int,
	epsilon float64,
) {
	normSizeF := T(normSize)
	rank := len(dims)

	// Build set of norm axes for fast lookup.
	isNormAxis := make([]bool, rank)
	for _, a := range axes {
		isNormAxis[a] = true
	}

	// Build outer axes (those not in normalization set).
	outerAxes := make([]int, 0, rank-len(axes))
	for i := range rank {
		if !isNormAxis[i] {
			outerAxes = append(outerAxes, i)
		}
	}

	// Create shape for iteration. DType is irrelevant for IterOnAxes.
	shape := shapes.Make(dtypes.Float32, dims...)
	strides := shape.Strides()
	outerIndices := make([]int, rank)
	normIndices := make([]int, rank)

	for outerFlatIdx := range shape.IterOnAxes(outerAxes, strides, outerIndices) {
		// Compute mean over norm axes.
		var sum T
		copy(normIndices, outerIndices)
		for flatIdx := range shape.IterOnAxes(axes, strides, normIndices) {
			sum += inData[flatIdx]
		}
		mean := sum / normSizeF

		// Compute variance.
		var varSum T
		copy(normIndices, outerIndices)
		for flatIdx := range shape.IterOnAxes(axes, strides, normIndices) {
			diff := inData[flatIdx] - mean
			varSum += diff * diff
		}
		variance := varSum / normSizeF
		invStd := T(1.0 / math.Sqrt(float64(variance)+epsilon))

		// Normalize and apply scale/offset.
		normFlatIdx := 0
		copy(normIndices, outerIndices)
		for flatIdx := range shape.IterOnAxes(axes, strides, normIndices) {
			normalized := (inData[flatIdx] - mean) * invStd
			if gammaData != nil {
				normalized *= gammaData[normFlatIdx]
			}
			if betaData != nil {
				normalized += betaData[normFlatIdx]
			}
			outData[flatIdx] = normalized
			normFlatIdx++
		}
		_ = outerFlatIdx
	}
}

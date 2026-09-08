// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dense

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedDenseVJP.Register(FusedDenseVJP, gobackend.PriorityGeneric)
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeFusedDenseVJP] = execFusedDenseVJP
}

type nodeFusedDenseVJP struct {
	options         compute.DenseConfig
	layout          dot.Layout
	batchSize       int
	lhsCrossSize    int
	rhsCrossSize    int
	contractingSize int
	hasBias         bool
}

func (d *nodeFusedDenseVJP) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedDenseVJP)
	return d.options == o.options &&
		d.layout == o.layout &&
		d.batchSize == o.batchSize &&
		d.lhsCrossSize == o.lhsCrossSize &&
		d.rhsCrossSize == o.rhsCrossSize &&
		d.contractingSize == o.contractingSize &&
		d.hasBias == o.hasBias
}

// Recompute implements gobackend.RecomputableNodeData for nodeFusedDenseVJP.
func (d *nodeFusedDenseVJP) Recompute(backend *gobackend.Backend, resolvedNodes []*gobackend.Node, originalNode *gobackend.Node) (any, error) {
	resolvedX := resolvedNodes[originalNode.Inputs[0].Index].Shape
	inFeatures := resolvedX.Dimensions[resolvedX.Rank()-1]
	newData := &nodeFusedDenseVJP{
		options:         d.options,
		layout:          d.layout,
		batchSize:       d.batchSize,
		lhsCrossSize:    resolvedX.Size() / inFeatures,
		rhsCrossSize:    d.rhsCrossSize,
		contractingSize: d.contractingSize,
		hasBias:         d.hasBias,
	}
	return newData, nil
}

// FusedDenseVJP computes the Vector-Jacobian Product (gradients) of FusedDense with respect to
// inputs x, weight, and bias.
func FusedDenseVJP(f *gobackend.Function, x, weight, bias, y, dOutput compute.Value, options compute.DenseConfig) (dx, dWeight, dBias compute.Value, err error) {
	values := []compute.Value{x, weight}
	if bias != nil {
		values = append(values, bias)
	}
	values = append(values, y, dOutput)
	inputs, err := f.VerifyAndCastValues("FusedDenseVJP", values...)
	if err != nil {
		return nil, nil, nil, err
	}

	xNode := inputs[0]
	wNode := inputs[1]
	var biasShape shapes.Shape
	if bias != nil {
		biasShape = inputs[2].Shape
	}
	yNode := inputs[len(inputs)-2]
	dOutNode := inputs[len(inputs)-1]

	dxShape, dwShape, dbShape, err := shapeinference.FusedDenseVJP(xNode.Shape, wNode.Shape, biasShape, yNode.Shape, dOutNode.Shape, options)
	if err != nil {
		return nil, nil, nil, err
	}

	if wNode.Shape.IsDynamic() || biasShape.IsDynamic() {
		return nil, nil, nil, compute.ErrNotImplemented
	}
	if xNode.Shape.Dimensions[xNode.Shape.Rank()-1] == shapes.DynamicDim {
		return nil, nil, nil, errors.Errorf("FusedDenseVJP: x's last dimension (in_features) cannot be dynamic")
	}

	inFeatures := xNode.Shape.Dimensions[xNode.Shape.Rank()-1]
	var layout dot.Layout
	switch options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		layout = dot.LayoutNonTransposed
	case compute.DenseLayoutOutputsInput:
		layout = dot.LayoutTransposed
	default:
		return nil, nil, nil, errors.Errorf("FusedDenseVJP: unknown WeightLayout %v", options.WeightLayout)
	}

	lhsCrossSize := 0
	if !xNode.Shape.IsDynamic() {
		lhsCrossSize = xNode.Shape.Size() / inFeatures
	}
	rhsCrossSize := wNode.Shape.Size() / inFeatures
	contractingSize := inFeatures

	data := &nodeFusedDenseVJP{
		options:         options,
		layout:          layout,
		batchSize:       1,
		lhsCrossSize:    lhsCrossSize,
		rhsCrossSize:    rhsCrossSize,
		contractingSize: contractingSize,
		hasBias:         bias != nil,
	}

	var outputShapes []shapes.Shape
	if bias != nil {
		outputShapes = []shapes.Shape{dxShape, dwShape, dbShape}
	} else {
		outputShapes = []shapes.Shape{dxShape, dwShape}
	}

	node := f.NewMultiOutputsNode(compute.OpTypeFusedDenseVJP, outputShapes, inputs...)
	node.Data = data

	dx = node.MultiOutputsNodes[0]
	dWeight = node.MultiOutputsNodes[1]
	if bias != nil {
		dBias = node.MultiOutputsNodes[2]
	}
	return dx, dWeight, dBias, nil
}

// execFusedDenseVJP executes the vector-jacobian product computation.
func execFusedDenseVJP(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) ([]*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedDenseVJP)
	x := inputs[0]
	weight := inputs[1]
	var y, dOutput *gobackend.Buffer
	if data.hasBias {
		y = inputs[3]
		dOutput = inputs[4]
	} else {
		y = inputs[2]
		dOutput = inputs[3]
	}

	dxShape := node.MultiOutputsShapes[0]
	dwShape := node.MultiOutputsShapes[1]

	dxBuf, err := backend.GetBuffer(dxShape)
	if err != nil {
		return nil, err
	}
	dwBuf, err := backend.GetBuffer(dwShape)
	if err != nil {
		backend.PutBuffer(dxBuf)
		return nil, err
	}
	var dbBuf *gobackend.Buffer
	if data.hasBias {
		dbShape := node.MultiOutputsShapes[2]
		dbBuf, err = backend.GetBuffer(dbShape)
		if err != nil {
			backend.PutBuffer(dxBuf)
			backend.PutBuffer(dwBuf)
			return nil, err
		}
	}

	if backend.NoOps {
		if data.hasBias {
			return []*gobackend.Buffer{dxBuf, dwBuf, dbBuf}, nil
		}
		return []*gobackend.Buffer{dxBuf, dwBuf}, nil
	}

	switch dOutput.RawShape.DType {
	case dtypes.Float32:
		err = execFusedDenseVJPGeneric[float32](backend, data, x, weight, y, dOutput, dxBuf, dwBuf, dbBuf)
	case dtypes.Float64:
		err = execFusedDenseVJPGeneric[float64](backend, data, x, weight, y, dOutput, dxBuf, dwBuf, dbBuf)
	case dtypes.BFloat16:
		err = execFusedDenseVJPBF16(backend, data, x, weight, y, dOutput, dxBuf, dwBuf, dbBuf)
	case dtypes.Float16:
		err = execFusedDenseVJPF16(backend, data, x, weight, y, dOutput, dxBuf, dwBuf, dbBuf)
	default:
		err = errors.Wrapf(compute.ErrNotImplemented, "FusedDenseVJP: dtype %s", dOutput.RawShape.DType)
	}

	if err != nil {
		backend.PutBuffer(dxBuf)
		backend.PutBuffer(dwBuf)
		if dbBuf != nil {
			backend.PutBuffer(dbBuf)
		}
		return nil, err
	}

	if data.hasBias {
		return []*gobackend.Buffer{dxBuf, dwBuf, dbBuf}, nil
	}
	return []*gobackend.Buffer{dxBuf, dwBuf}, nil
}

// transpose2D transposes a matrix of shape [rows, cols] to [cols, rows] using cache-friendly blocking.
func transpose2D[T any](src []T, rows, cols int, dst []T) {
	const blockSize = 32
	for r0 := 0; r0 < rows; r0 += blockSize {
		rMax := min(r0+blockSize, rows)
		for c0 := 0; c0 < cols; c0 += blockSize {
			cMax := min(c0+blockSize, cols)
			for r := r0; r < rMax; r++ {
				rOffset := r * cols
				for c := c0; c < cMax; c++ {
					dst[c*rows+r] = src[rOffset+c]
				}
			}
		}
	}
}

// reduceBiasFloat reduces dZ of shape [m, n] along the m dimension into dBias of shape [n].
func reduceBiasFloat[T float32 | float64](backend *gobackend.Backend, dZ []T, m, n int, dBias []T) {
	if m == 1 {
		copy(dBias, dZ)
		return
	}

	const minParallelChunk = 4096
	totalWork := m * n
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && totalWork > minParallelChunk && n > 1 {
		colsPerChunk := max(minParallelChunk/m, 1)
		var wg sync.WaitGroup
		for c0 := 0; c0 < n; c0 += colsPerChunk {
			cEnd := min(c0+colsPerChunk, n)
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				for c := c0; c < cEnd; c++ {
					var sum T
					for r := range m {
						sum += dZ[r*n+c]
					}
					dBias[c] = sum
				}
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		for c := range n {
			var sum T
			for r := range m {
				sum += dZ[r*n+c]
			}
			dBias[c] = sum
		}
	}
}

func execFusedDenseVJPGeneric[T float32 | float64](
	backend *gobackend.Backend,
	data *nodeFusedDenseVJP,
	x, weight, y, dOutput, dxBuf, dwBuf, dbBuf *gobackend.Buffer,
) error {
	m := data.lhsCrossSize
	n := data.rhsCrossSize
	k := data.contractingSize

	xFlat := x.Flat.([]T)
	wFlat := weight.Flat.([]T)
	yFlat := y.Flat.([]T)
	dOutFlat := dOutput.Flat.([]T)
	dxFlat := dxBuf.Flat.([]T)
	dwFlat := dwBuf.Flat.([]T)

	// Step 1: Pre-activation gradient dZ of shape [M, N].
	var dZ []T
	if data.options.Activation.Type == compute.ActivationNone {
		dZ = dOutFlat
	} else {
		dZ = make([]T, m*n)
		if err := activations.ExecuteVJPFromOutput[T](backend, data.options.Activation.Type, yFlat, dOutFlat, dZ); err != nil {
			return err
		}
	}

	// Step 2: Compute dX of shape [M, K].
	// DenseLayoutInputOutputs: dX = dZ @ W^T  (LHS: [M, N], RHS: [K, N] -> LayoutTransposed)
	// DenseLayoutOutputsInput: dX = dZ @ W    (LHS: [M, N], RHS: [N, K] -> LayoutNonTransposed)
	var dXLayout dot.Layout
	switch data.options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		dXLayout = dot.LayoutTransposed
	case compute.DenseLayoutOutputsInput:
		dXLayout = dot.LayoutNonTransposed
	}

	err := matmul.ExecuteWithEpilogue(
		backend, dXLayout, dZ, wFlat,
		1, m, k, n,
		dxFlat, matmul.Epilogue[T]{},
	)
	if err != nil {
		return errors.WithMessage(err, "FusedDenseVJP dX matmul failed")
	}

	// Step 3: Compute dWeight.
	// DenseLayoutInputOutputs: dW = x^T @ dZ  ([K, M] x [M, N] -> [K, N])
	// DenseLayoutOutputsInput: dW = dZ^T @ x  ([N, M] x [M, K] -> [N, K])
	switch data.options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		xT := make([]T, k*m)
		transpose2D(xFlat, m, k, xT)
		err = matmul.ExecuteWithEpilogue(
			backend, dot.LayoutNonTransposed, xT, dZ,
			1, k, n, m,
			dwFlat, matmul.Epilogue[T]{},
		)
		if err != nil {
			return errors.WithMessage(err, "FusedDenseVJP dWeight matmul failed")
		}

	case compute.DenseLayoutOutputsInput:
		dZT := make([]T, n*m)
		transpose2D(dZ, m, n, dZT)
		err = matmul.ExecuteWithEpilogue(
			backend, dot.LayoutNonTransposed, dZT, xFlat,
			1, n, k, m,
			dwFlat, matmul.Epilogue[T]{},
		)
		if err != nil {
			return errors.WithMessage(err, "FusedDenseVJP dWeight matmul failed")
		}
	}

	// Step 4: Compute dBias (if requested).
	if data.hasBias && dbBuf != nil {
		dbFlat := dbBuf.Flat.([]T)
		reduceBiasFloat[T](backend, dZ, m, n, dbFlat)
	}

	return nil
}

func execFusedDenseVJPBF16(
	backend *gobackend.Backend,
	data *nodeFusedDenseVJP,
	x, weight, y, dOutput, dxBuf, dwBuf, dbBuf *gobackend.Buffer,
) error {
	m := data.lhsCrossSize
	n := data.rhsCrossSize
	k := data.contractingSize

	xBF16 := x.Flat.([]bfloat16.BFloat16)
	wBF16 := weight.Flat.([]bfloat16.BFloat16)
	yBF16 := y.Flat.([]bfloat16.BFloat16)
	dOutBF16 := dOutput.Flat.([]bfloat16.BFloat16)
	dxBF16 := dxBuf.Flat.([]bfloat16.BFloat16)
	dwBF16 := dwBuf.Flat.([]bfloat16.BFloat16)

	// Step 1: Pre-activation gradient dZ of shape [M, N].
	var dZ []bfloat16.BFloat16
	if data.options.Activation.Type == compute.ActivationNone {
		dZ = dOutBF16
	} else {
		dZ = make([]bfloat16.BFloat16, m*n)
		if err := activations.ExecuteVJPFromOutput[bfloat16.BFloat16](backend, data.options.Activation.Type, yBF16, dOutBF16, dZ); err != nil {
			return err
		}
	}

	// Step 2: Compute dX of shape [M, K].
	var dXLayout dot.Layout
	switch data.options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		dXLayout = dot.LayoutTransposed
	case compute.DenseLayoutOutputsInput:
		dXLayout = dot.LayoutNonTransposed
	}

	tmpF32_dX := make([]float32, m*k)
	err := matmul.ExecuteWithEpilogue(
		backend, dXLayout, dZ, wBF16,
		1, m, k, n,
		tmpF32_dX, matmul.Epilogue[float32]{},
	)
	if err != nil {
		return errors.WithMessage(err, "FusedDenseVJP dX matmul failed")
	}
	for i, v := range tmpF32_dX {
		dxBF16[i] = bfloat16.FromFloat32(v)
	}

	// Step 3: Compute dWeight.
	switch data.options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		xT := make([]bfloat16.BFloat16, k*m)
		transpose2D(xBF16, m, k, xT)
		tmpF32_dW := make([]float32, k*n)
		err = matmul.ExecuteWithEpilogue(
			backend, dot.LayoutNonTransposed, xT, dZ,
			1, k, n, m,
			tmpF32_dW, matmul.Epilogue[float32]{},
		)
		if err != nil {
			return errors.WithMessage(err, "FusedDenseVJP dWeight matmul failed")
		}
		for i, v := range tmpF32_dW {
			dwBF16[i] = bfloat16.FromFloat32(v)
		}

	case compute.DenseLayoutOutputsInput:
		dZT := make([]bfloat16.BFloat16, n*m)
		transpose2D(dZ, m, n, dZT)
		tmpF32_dW := make([]float32, n*k)
		err = matmul.ExecuteWithEpilogue(
			backend, dot.LayoutNonTransposed, dZT, xBF16,
			1, n, k, m,
			tmpF32_dW, matmul.Epilogue[float32]{},
		)
		if err != nil {
			return errors.WithMessage(err, "FusedDenseVJP dWeight matmul failed")
		}
		for i, v := range tmpF32_dW {
			dwBF16[i] = bfloat16.FromFloat32(v)
		}
	}

	// Step 4: Compute dBias.
	if data.hasBias && dbBuf != nil {
		dbBF16 := dbBuf.Flat.([]bfloat16.BFloat16)
		for c := range n {
			var sum float32
			for r := range m {
				sum += dZ[r*n+c].Float32()
			}
			dbBF16[c] = bfloat16.FromFloat32(sum)
		}
	}

	return nil
}

func execFusedDenseVJPF16(
	backend *gobackend.Backend,
	data *nodeFusedDenseVJP,
	x, weight, y, dOutput, dxBuf, dwBuf, dbBuf *gobackend.Buffer,
) error {
	m := data.lhsCrossSize
	n := data.rhsCrossSize
	k := data.contractingSize

	xF16 := x.Flat.([]float16.Float16)
	wF16 := weight.Flat.([]float16.Float16)
	yF16 := y.Flat.([]float16.Float16)
	dOutF16 := dOutput.Flat.([]float16.Float16)
	dxF16 := dxBuf.Flat.([]float16.Float16)
	dwF16 := dwBuf.Flat.([]float16.Float16)

	// Step 1: Pre-activation gradient dZ of shape [M, N].
	var dZ []float16.Float16
	if data.options.Activation.Type == compute.ActivationNone {
		dZ = dOutF16
	} else {
		dZ = make([]float16.Float16, m*n)
		if err := activations.ExecuteVJPFromOutput[float16.Float16](backend, data.options.Activation.Type, yF16, dOutF16, dZ); err != nil {
			return err
		}
	}

	// Step 2: Compute dX of shape [M, K].
	var dXLayout dot.Layout
	switch data.options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		dXLayout = dot.LayoutTransposed
	case compute.DenseLayoutOutputsInput:
		dXLayout = dot.LayoutNonTransposed
	}

	tmpF32_dX := make([]float32, m*k)
	err := matmul.ExecuteWithEpilogue(
		backend, dXLayout, dZ, wF16,
		1, m, k, n,
		tmpF32_dX, matmul.Epilogue[float32]{},
	)
	if err != nil {
		return errors.WithMessage(err, "FusedDenseVJP dX matmul failed")
	}
	for i, v := range tmpF32_dX {
		dxF16[i] = float16.FromFloat32(v)
	}

	// Step 3: Compute dWeight.
	switch data.options.WeightLayout {
	case compute.DenseLayoutInputOutputs:
		xT := make([]float16.Float16, k*m)
		transpose2D(xF16, m, k, xT)
		tmpF32_dW := make([]float32, k*n)
		err = matmul.ExecuteWithEpilogue(
			backend, dot.LayoutNonTransposed, xT, dZ,
			1, k, n, m,
			tmpF32_dW, matmul.Epilogue[float32]{},
		)
		if err != nil {
			return errors.WithMessage(err, "FusedDenseVJP dWeight matmul failed")
		}
		for i, v := range tmpF32_dW {
			dwF16[i] = float16.FromFloat32(v)
		}

	case compute.DenseLayoutOutputsInput:
		dZT := make([]float16.Float16, n*m)
		transpose2D(dZ, m, n, dZT)
		tmpF32_dW := make([]float32, n*k)
		err = matmul.ExecuteWithEpilogue(
			backend, dot.LayoutNonTransposed, dZT, xF16,
			1, n, k, m,
			tmpF32_dW, matmul.Epilogue[float32]{},
		)
		if err != nil {
			return errors.WithMessage(err, "FusedDenseVJP dWeight matmul failed")
		}
		for i, v := range tmpF32_dW {
			dwF16[i] = float16.FromFloat32(v)
		}
	}

	// Step 4: Compute dBias.
	if data.hasBias && dbBuf != nil {
		dbF16 := dbBuf.Flat.([]float16.Float16)
		for c := range n {
			var sum float32
			for r := range m {
				sum += dZ[r*n+c].Float32()
			}
			dbF16[c] = float16.FromFloat32(sum)
		}
	}

	return nil
}

package fusedops

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot/matmul"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedAttentionQKVProjection.Register(FusedAttentionQKVProjection, gobackend.PriorityGeneric)
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeFusedAttentionQKVProjection] = execFusedAttentionQKVProjection
}

// nodeFusedAttentionQKVProjection stores parameters for the fused QKV projection.
// It does not implement gobackend.NodeDataComparable because multi-output nodes are not
// de-duplicated (see newMultiOutputsNode).
type nodeFusedAttentionQKVProjection struct {
	qDim     int
	kvDim    int
	hasBiasQ bool
	hasBiasK bool
	hasBiasV bool
}

// EqualNodeData compares two node data structures for equality.
// The other value is guaranteed to be of type *nodeFusedAttentionQKVProjection.
func (d *nodeFusedAttentionQKVProjection) EqualNodeData(other gobackend.NodeDataComparable) bool {
	return *d == *(other.(*nodeFusedAttentionQKVProjection))
}

// FusedAttentionQKVProjection performs fused Query-Key-Value projection.
//
// The matmul (x @ wQKV) is delegated to DotGeneral, which selects the optimal
// execution path by a combination of build and runtime. The fused
// executor then splits the result into Q/K/V and adds biases.
func FusedAttentionQKVProjection(f *gobackend.Function, x, wQKV, biasQ, biasK, biasV compute.Value, queryDim, keyValueDim int) (queryOut, keyOut, valueOut compute.Value, err error) {
	values := []compute.Value{x, wQKV}
	if biasQ != nil {
		values = append(values, biasQ)
	}
	if biasK != nil {
		values = append(values, biasK)
	}
	if biasV != nil {
		values = append(values, biasV)
	}
	inputs, err := f.VerifyAndCastValues("AttentionQKVProjection", values...)
	if err != nil {
		return nil, nil, nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	if xNode.Shape.Rank() < 1 {
		return nil, nil, nil, errors.Errorf("AttentionQKVProjection: x must have rank >= 1, got %d", xNode.Shape.Rank())
	}

	batchDims := xNode.Shape.Dimensions[:xNode.Shape.Rank()-1]
	qDims := make([]int, len(batchDims)+1)
	copy(qDims, batchDims)
	qDims[len(batchDims)] = queryDim
	kvDims := make([]int, len(batchDims)+1)
	copy(kvDims, batchDims)
	kvDims[len(batchDims)] = keyValueDim

	var qShape, kShape, vShape shapes.Shape
	if xNode.Shape.IsDynamic() || len(xNode.Shape.AxisNames) > 0 {
		var qAxes, kvAxes []string
		if len(xNode.Shape.AxisNames) > 0 {
			batchAxes := xNode.Shape.AxisNames[:xNode.Shape.Rank()-1]
			qAxes = append(slices.Clone(batchAxes), "")
			kvAxes = append(slices.Clone(batchAxes), "")
		} else {
			qAxes = make([]string, len(qDims))
			kvAxes = make([]string, len(kvDims))
		}
		qShape = shapes.MakeDynamic(xNode.Shape.DType, qDims, qAxes)
		kShape = shapes.MakeDynamic(xNode.Shape.DType, kvDims, kvAxes)
		vShape = shapes.MakeDynamic(xNode.Shape.DType, kvDims, kvAxes)
	} else {
		qShape = shapes.Make(xNode.Shape.DType, qDims...)
		kShape = shapes.Make(xNode.Shape.DType, kvDims...)
		vShape = shapes.Make(xNode.Shape.DType, kvDims...)
	}

	// Build DotGeneral sub-node for the matmul: x @ wQKV.
	// This delegates to the optimized matmul infrastructure.
	dotResult, dotErr := f.DotGeneral(xNode, []int{xNode.Shape.Rank() - 1}, nil, wNode, []int{0}, nil, compute.DotGeneralConfig{})
	if dotErr != nil {
		return nil, nil, nil, errors.WithMessagef(dotErr, "FusedAttentionQKVProjection: DotGeneral")
	}
	dotNode := dotResult.(*gobackend.Node)

	// FusedAttentionQKVProjection inputs: [dotResult, biasQ?, biasK?, biasV?].
	// The matmul is already computed by the DotGeneral sub-node (inputs[0]).
	// Bias nodes are at inputs[2:] in the same order they were appended.
	fusedInputs := append([]*gobackend.Node{dotNode}, inputs[2:]...)

	data := &nodeFusedAttentionQKVProjection{qDim: queryDim, kvDim: keyValueDim, hasBiasQ: biasQ != nil, hasBiasK: biasK != nil, hasBiasV: biasV != nil}
	node := f.NewMultiOutputsNode(compute.OpTypeFusedAttentionQKVProjection, []shapes.Shape{qShape, kShape, vShape}, fusedInputs...)
	node.Data = data
	queryOut = node.MultiOutputsNodes[0]
	keyOut = node.MultiOutputsNodes[1]
	valueOut = node.MultiOutputsNodes[2]
	return
}

// execFusedAttentionQKVProjection implements fused QKV projection.
// inputs[0]: pre-computed DotGeneral result [batch, qDim+2*kvDim]
// inputs[1..]: biasQ, biasK, biasV (optional, determined by node data flags)
// outputs: q [batch, qDim], k [batch, kvDim], v [batch, kvDim]
//
// The matmul (x @ wQKV) is already computed by the DotGeneral sub-node.
// This executor just splits the combined result into Q/K/V and adds biases.
func execFusedAttentionQKVProjection(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) ([]*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedAttentionQKVProjection)
	combined := inputs[0] // DotGeneral result: [batch, qDim+2*kvDim]

	// Determine bias buffers using flags from node data, not positional indexing.
	var biasQ, biasK, biasV *gobackend.Buffer
	biasIdx := 1
	if data.hasBiasQ {
		biasQ = inputs[biasIdx]
		biasIdx++
	}
	if data.hasBiasK {
		biasK = inputs[biasIdx]
		biasIdx++
	}
	if data.hasBiasV {
		biasV = inputs[biasIdx]
	}

	qShape := node.MultiOutputsShapes[0]
	kShape := node.MultiOutputsShapes[1]
	vShape := node.MultiOutputsShapes[2]
	qBuf, err := backend.GetBuffer(qShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", qShape)
	}
	kBuf, err := backend.GetBuffer(kShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", kShape)
	}
	vBuf, err := backend.GetBuffer(vShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", vShape)
	}
	qDim := data.qDim
	kvDim := data.kvDim

	switch combined.RawShape.DType {
	case dtypes.Float32:
		qkvSplitBiasFloat32(combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	case dtypes.Float64:
		qkvSplitBiasFloat64(combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	case dtypes.Float16:
		qkvSplitBiasHalf[float16.Float16](combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	case dtypes.BFloat16:
		qkvSplitBiasHalf[bfloat16.BFloat16](combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	default:
		return nil, errors.Errorf("FusedAttentionQKVProjection: unsupported dtype %s", combined.RawShape.DType)
	}

	return []*gobackend.Buffer{qBuf, kBuf, vBuf}, nil
}

func qkvSplitBiasFloat32(combined, biasQBuf, biasKBuf, biasVBuf, qBuf, kBuf, vBuf *gobackend.Buffer, qDim, kvDim int) {
	src := combined.Flat.([]float32)
	q := qBuf.Flat.([]float32)
	k := kBuf.Flat.([]float32)
	v := vBuf.Flat.([]float32)
	var biasQ, biasK, biasV []float32
	if biasQBuf != nil {
		biasQ = biasQBuf.Flat.([]float32)
	}
	if biasKBuf != nil {
		biasK = biasKBuf.Flat.([]float32)
	}
	if biasVBuf != nil {
		biasV = biasVBuf.Flat.([]float32)
	}

	totalOut := qDim + 2*kvDim
	batchSize := len(src) / totalOut
	for batchIdx := range batchSize {
		srcBase := batchIdx * totalOut
		qBase := batchIdx * qDim
		kBase := batchIdx * kvDim
		vBase := batchIdx * kvDim

		// Q projection
		projectSliceFloat32(q[qBase:qBase+qDim], src[srcBase:srcBase+qDim], biasQ)
		// K projection
		projectSliceFloat32(k[kBase:kBase+kvDim], src[srcBase+qDim:srcBase+qDim+kvDim], biasK)
		// V projection
		projectSliceFloat32(v[vBase:vBase+kvDim], src[srcBase+qDim+kvDim:srcBase+totalOut], biasV)
	}
}

func projectSliceFloat32(dst, src, bias []float32) {
	matmul.CopyAndAddBiasFloat32(dst, src, bias)
}

func qkvSplitBiasFloat64(combined, biasQBuf, biasKBuf, biasVBuf, qBuf, kBuf, vBuf *gobackend.Buffer, qDim, kvDim int) {
	src := combined.Flat.([]float64)
	q := qBuf.Flat.([]float64)
	k := kBuf.Flat.([]float64)
	v := vBuf.Flat.([]float64)
	var biasQ, biasK, biasV []float64
	if biasQBuf != nil {
		biasQ = biasQBuf.Flat.([]float64)
	}
	if biasKBuf != nil {
		biasK = biasKBuf.Flat.([]float64)
	}
	if biasVBuf != nil {
		biasV = biasVBuf.Flat.([]float64)
	}

	totalOut := qDim + 2*kvDim
	batchSize := len(src) / totalOut
	for batchIdx := range batchSize {
		srcBase := batchIdx * totalOut
		qBase := batchIdx * qDim
		kBase := batchIdx * kvDim
		vBase := batchIdx * kvDim

		// Q projection
		projectSliceFloat64(q[qBase:qBase+qDim], src[srcBase:srcBase+qDim], biasQ)
		// K projection
		projectSliceFloat64(k[kBase:kBase+kvDim], src[srcBase+qDim:srcBase+qDim+kvDim], biasK)
		// V projection
		projectSliceFloat64(v[vBase:vBase+kvDim], src[srcBase+qDim+kvDim:srcBase+totalOut], biasV)
	}
}

func projectSliceFloat64(dst, src, bias []float64) {
	matmul.CopyAndAddBiasFloat64(dst, src, bias)
}

func qkvSplitBiasHalf[T gotype.HalfPrecision[T], P gotype.HalfPrecisionPtr[T]](combined, biasQBuf, biasKBuf, biasVBuf, qBuf, kBuf, vBuf *gobackend.Buffer, qDim, kvDim int) {
	src := combined.Flat.([]T)
	q := qBuf.Flat.([]T)
	k := kBuf.Flat.([]T)
	v := vBuf.Flat.([]T)
	var biasQ, biasK, biasV []T
	if biasQBuf != nil {
		biasQ = biasQBuf.Flat.([]T)
	}
	if biasKBuf != nil {
		biasK = biasKBuf.Flat.([]T)
	}
	if biasVBuf != nil {
		biasV = biasVBuf.Flat.([]T)
	}

	projectSliceHalf := func(dst, src, bias []T) {
		if bias == nil {
			copy(dst, src)
			return
		}
		for i := range dst {
			P(&dst[i]).SetFloat32(src[i].Float32() + bias[i].Float32())
		}
	}

	totalOut := qDim + 2*kvDim
	batchSize := len(src) / totalOut
	for batchIdx := range batchSize {
		srcBase := batchIdx * totalOut
		qBase := batchIdx * qDim
		kBase := batchIdx * kvDim
		vBase := batchIdx * kvDim

		projectSliceHalf(q[qBase:qBase+qDim], src[srcBase:srcBase+qDim], biasQ)
		projectSliceHalf(k[kBase:kBase+kvDim], src[srcBase+qDim:srcBase+qDim+kvDim], biasK)
		projectSliceHalf(v[vBase:vBase+kvDim], src[srcBase+qDim+kvDim:srcBase+totalOut], biasV)
	}
}

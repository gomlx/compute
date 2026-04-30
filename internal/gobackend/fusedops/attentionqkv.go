package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

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

// FusedAttentionQKVProjection performs fused Query-Key-Value projection.
//
// The matmul (x @ wQKV) is delegated to DotGeneral, which selects the optimal
// execution path (blocked, packgemm, highway, etc.) at build time. The fused
// executor then splits the result into Q/K/V and adds biases.

func init() {
	gobackend.RegisterFusedAttentionQKVProjection.Register(FusedAttentionQKVProjection, gobackend.PriorityGeneric)
}

func FusedAttentionQKVProjection(f *gobackend.Function, x, wQKV, biasQ, biasK, biasV *gobackend.Node, queryDim, keyValueDim int) (queryOut, keyOut, valueOut *gobackend.Node, err error) {
	values := []*gobackend.Node{x, wQKV}
	if biasQ != nil {
		values = append(values, biasQ)
	}
	if biasK != nil {
		values = append(values, biasK)
	}
	if biasV != nil {
		values = append(values, biasV)
	}
	inputs, err := f.verifyAndCastValues("AttentionQKVProjection", values...)
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

	qShape := shapes.Make(xNode.Shape.DType, qDims...)
	kShape := shapes.Make(xNode.Shape.DType, kvDims...)
	vShape := shapes.Make(xNode.Shape.DType, kvDims...)

	// Build DotGeneral sub-node for the matmul: x @ wQKV.
	// This delegates to the optimized matmul infrastructure (blocked, packgemm, highway, etc.).
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
	node := f.newMultiOutputsNode(compute.OpTypeFusedAttentionQKVProjection, []shapes.Shape{qShape, kShape, vShape}, fusedInputs...)
	node.Data = data
	queryOut = node.MultiOutputsNodes[0]
	keyOut = node.MultiOutputsNodes[1]
	valueOut = node.MultiOutputsNodes[2]
	return
}

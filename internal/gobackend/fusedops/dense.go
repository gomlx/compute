package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

type nodeFusedDense struct {
	activation compute.ActivationType
}

func (d *nodeFusedDense) EqualNodeData(other gobackend.NodeDataComparable) bool {
	return d.activation == other.(*nodeFusedDense).activation
}

// FusedDense performs fused matmul + optional bias + optional activation:
//
//	y = activation(x @ W + bias)
//
// The matmul is delegated to DotGeneral (which selects the optimal execution
// path at build time). FusedDense then adds bias and applies activation on top
// of the DotGeneral result.

func init() {
	gobackend.RegisterFusedDense.Register(FusedDense, gobackend.PriorityGeneric)
}

func FusedDense(f *gobackend.Function, x, weight, bias *gobackend.Node, activation compute.ActivationType) (*gobackend.Node, error) {
	values := []*gobackend.Node{x, weight}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.verifyAndCastValues("FusedDense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	if xNode.Shape.Rank() < 1 || wNode.Shape.Rank() < 2 {
		return nil, errors.Errorf("FusedDense: x must have rank >= 1 (got %d), weight must have rank >= 2 (got %d)",
			xNode.Shape.Rank(), wNode.Shape.Rank())
	}
	inFeatures := xNode.Shape.Dimensions[xNode.Shape.Rank()-1]
	if inFeatures != wNode.Shape.Dimensions[0] {
		return nil, errors.Errorf("FusedDense: x's last dim (%d) must match weight's first dim (%d)",
			inFeatures, wNode.Shape.Dimensions[0])
	}

	outDims := make([]int, xNode.Shape.Rank()-1+wNode.Shape.Rank()-1)
	copy(outDims, xNode.Shape.Dimensions[:xNode.Shape.Rank()-1])
	copy(outDims[xNode.Shape.Rank()-1:], wNode.Shape.Dimensions[1:])
	outShape := shapes.Make(xNode.Shape.DType, outDims...)

	// Build DotGeneral sub-node for the matmul: contract x's last axis with weight's first.
	dotResult, err := f.DotGeneral(xNode, []int{xNode.Shape.Rank() - 1}, nil, wNode, []int{0}, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, errors.WithMessagef(err, "FusedDense: DotGeneral")
	}
	dotNode := dotResult.(*gobackend.Node)

	// FusedDense inputs: [dotResult, x, weight, bias?].
	// The matmul is already computed by the DotGeneral sub-node (inputs[0]).
	// x and weight are included so that SIMD-accelerated executors (highway) can
	// redo the fused matmul+bias+activation from scratch.
	fusedInputs := []*gobackend.Node{dotNode, xNode, wNode}
	if len(inputs) > 2 {
		fusedInputs = append(fusedInputs, inputs[2])
	}

	data := &nodeFusedDense{activation: activation}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedDense, outShape, fusedInputs, data)
	return node, nil
}

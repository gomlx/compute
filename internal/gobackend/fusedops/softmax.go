package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
)

type nodeFusedSoftmax struct {
	axis int
}

func (d *nodeFusedSoftmax) EqualNodeData(other gobackend.NodeDataComparable) bool {
	return d.axis == other.(*nodeFusedSoftmax).axis
}

// FusedSoftmax computes softmax along the specified axis.
// The axis must be non-negative (the caller normalizes negative indices).

func init() {
	gobackend.RegisterFusedSoftmax.Register(FusedSoftmax, gobackend.PriorityGeneric)
}

func FusedSoftmax(f *gobackend.Function, x *gobackend.Node, axis int) (*gobackend.Node, error) {
	inputs, err := f.verifyAndCastValues("FusedSoftmax", x)
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

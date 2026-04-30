package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
)

// FusedLayerNorm applies layer normalization.

func init() {
	gobackend.RegisterFusedLayerNorm.Register(FusedLayerNorm, gobackend.PriorityGeneric)
}

type nodeFusedLayerNorm struct {
	axes    []int
	epsilon float64
}

func (d *nodeFusedLayerNorm) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedLayerNorm)
	if d.epsilon != o.epsilon || len(d.axes) != len(o.axes) {
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

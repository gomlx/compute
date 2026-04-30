package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
)

type nodeFusedGelu struct {
	exact bool
}

func (d *nodeFusedGelu) EqualNodeData(other gobackend.NodeDataComparable) bool {
	return d.exact == other.(*nodeFusedGelu).exact
}

// FusedGelu computes Gaussian Error Linear Unit activation.
// If exact is true, uses the exact GELU (erf); otherwise uses the tanh approximation.

func init() {
	gobackend.RegisterFusedGelu.Register(FusedGelu, gobackend.PriorityGeneric)
}

func FusedGelu(f *gobackend.Function, x *gobackend.Node, exact bool) (*gobackend.Node, error) {
	inputs, err := f.verifyAndCastValues("FusedGelu", x)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]

	data := &nodeFusedGelu{exact: exact}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedGelu, xNode.Shape.Clone(), []*gobackend.Node{xNode}, data)
	return node, nil
}

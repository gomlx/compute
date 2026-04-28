package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterIota.Register(Iota, gobackend.PriorityGeneric)
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func Iota(f *gobackend.Function, shape shapes.Shape, iotaAxis int) (*gobackend.Node, error) {
	if shape.Rank() == 0 {
		return nil, errors.Errorf("Iota: shape must have at least one dimension")
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis (%d) must be in the range [0,%d)", iotaAxis, shape.Rank()-1)
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeIota, shape, nil, iotaAxis)
	return node, nil
}

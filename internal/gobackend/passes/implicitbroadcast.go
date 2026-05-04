package passes

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapeinference"
)

func init() {
	gobackend.RegisterTransformation(100, &ImplicitBroadcastFusion{})
}

// ImplicitBroadcastFusion removes unnecessary broadcast operations that are followed by binary operations:
// the implicit broadcast is performed as part of the binary operation implementation anyway, so they are not needed.
//
// This can save both temporary buffers and time (in memory-bandwidth) while creating them.
type ImplicitBroadcastFusion struct{}

// Name implements compute.GraphTransformation.
func (p *ImplicitBroadcastFusion) Name() string {
	return "ImplicitBroadcastFusion"
}

// Apply implements compute.GraphTransformation.
//
// The pass iterates over all functions and their nodes and checks if the node is a BroadcastInDim operation whose
// output is used by a binary operation. If so, it uses the input of the broadcast operation directly as input to the
// binary operation.
//
// It doesn't remove the broadcast operation, but it will be eliminated if nobody else is using it at a later step of
// the compilation (dead-code elimination).
//
// Binary operations for which implicit broadcasting applies are defined in
// shapeinference.StandardBinaryOperations and shapeinference.ComparisonOperations.
func (p *ImplicitBroadcastFusion) Apply(b *gobackend.Builder) error {
	for _, f := range b.Functions {
		for _, node := range f.Nodes {
			if !shapeinference.StandardBinaryOperations.Has(node.OpType) &&
				!shapeinference.ComparisonOperations.Has(node.OpType) {
				continue
			}

			// Binary operations for which implicit broadcasting applies.
			for i, input := range node.Inputs {
				if input.OpType != compute.OpTypeBroadcastInDim {
					continue
				}

				// Only fuse if rank doesn't change: implicit broadcasting only works if ranks are equal.
				if input.Inputs[0].Shape.Rank() == node.Shape.Rank() {
					// Fuse: use the input of the broadcast directly.
					node.Inputs[i] = input.Inputs[0]
				}
			}
		}
	}
	return nil
}

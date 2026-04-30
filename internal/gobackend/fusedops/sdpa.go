package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
)

type nodeFusedScaledDotProductAttention struct {
	numHeads   int
	numKVHeads int
	axesLayout compute.AxesLayout
	scale      float64
	causal     bool
	options    *compute.ScaledDotProductAttentionConfig
}

func (d *nodeFusedScaledDotProductAttention) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedScaledDotProductAttention)
	return d.numHeads == o.numHeads && d.numKVHeads == o.numKVHeads &&
		d.axesLayout == o.axesLayout && d.scale == o.scale && d.causal == o.causal &&
		d.equalOptions(o)
}

func (d *nodeFusedScaledDotProductAttention) equalOptions(o *nodeFusedScaledDotProductAttention) bool {
	if d.options == nil && o.options == nil {
		return true
	}
	if d.options == nil || o.options == nil {
		return false
	}
	return *d.options == *o.options
}

// FusedScaledDotProductAttention computes multi-head scaled dot-product attention.
// Both AxesLayoutBHSD and AxesLayoutBSHD are supported; the executor transposes
// BSHD inputs to BHSD internally.

func init() {
	gobackend.RegisterFusedScaledDotProductAttention.Register(FusedScaledDotProductAttention, gobackend.PriorityGeneric)
}

func FusedScaledDotProductAttention(f *gobackend.Function, query, key, value, mask *gobackend.Node, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) (*gobackend.Node, error) {
	return f.buildSDPANode(compute.OpTypeFusedScaledDotProductAttention, "FusedScaledDotProductAttention",
		query, key, value, mask, numHeads, numKVHeads, axesLayout, scale, causal, options)
}

// buildSDPANode builds the SDPA computation node.

func init() {
	gobackend.RegisterbuildSDPANode.Register(buildSDPANode, gobackend.PriorityGeneric)
}

func buildSDPANode(f *gobackend.Function, opType compute.OpType, opName string,
	query, key, value, mask *gobackend.Node,
	numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig,
) (*gobackend.Node, error) {
	values := []*gobackend.Node{query, key, value}
	if mask != nil {
		values = append(values, mask)
	}
	inputs, err := f.verifyAndCastValues(opName, values...)
	if err != nil {
		return nil, err
	}
	qNode := inputs[0]

	if qNode.Shape.Rank() != 4 {
		return nil, errors.Errorf("%s: query must have rank 4, got %d", opName, qNode.Shape.Rank())
	}
	if numHeads <= 0 || numKVHeads <= 0 || numHeads%numKVHeads != 0 {
		return nil, errors.Errorf("%s: numHeads (%d) must be positive and divisible by numKVHeads (%d)", opName, numHeads, numKVHeads)
	}

	data := &nodeFusedScaledDotProductAttention{numHeads: numHeads, numKVHeads: numKVHeads, axesLayout: axesLayout, scale: scale, causal: causal, options: options}
	node, _ := f.GetOrCreateNode(opType, qNode.Shape.Clone(), inputs, data)
	return node, nil
}

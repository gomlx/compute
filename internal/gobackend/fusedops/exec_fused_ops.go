// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
)

func init() {
	// Quantized fused executors registered in exec_fused_quantized.go init().
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeFusedAttentionQKVProjection] = execFusedAttentionQKVProjection
}

// FusedAttentionQKVProjection ===================================================================================

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
	qBuf, err := backend.GetBufferForShape(qShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", qShape)
	}
	kBuf, err := backend.GetBufferForShape(kShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", kShape)
	}
	vBuf, err := backend.GetBufferForShape(vShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", vShape)
	}
	qDim := data.qDim
	kvDim := data.kvDim

	switch combined.RawShape.DType {
	case dtypes.Float32:
		qkvSplitBiasGeneric[float32](combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	case dtypes.Float64:
		qkvSplitBiasGeneric[float64](combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	default:
		return nil, errors.Errorf("FusedAttentionQKVProjection: unsupported dtype %s", combined.RawShape.DType)
	}

	return []*gobackend.Buffer{qBuf, kBuf, vBuf}, nil
}

// qkvSplitBiasGeneric splits the pre-computed matmul result [batch, totalOut] into
// Q [batch, qDim], K [batch, kvDim], V [batch, kvDim] and adds optional biases.
func qkvSplitBiasGeneric[T float32 | float64](combined, biasQBuf, biasKBuf, biasVBuf, qBuf, kBuf, vBuf *gobackend.Buffer, qDim, kvDim int) {
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

	totalOut := qDim + 2*kvDim
	batchSize := len(src) / totalOut
	for batchIdx := range batchSize {
		srcBase := batchIdx * totalOut
		qBase := batchIdx * qDim
		kBase := batchIdx * kvDim
		vBase := batchIdx * kvDim

		// Copy Q columns and add bias.
		copy(q[qBase:qBase+qDim], src[srcBase:srcBase+qDim])
		if biasQ != nil {
			for o := range qDim {
				q[qBase+o] += biasQ[o]
			}
		}

		// Copy K columns and add bias.
		copy(k[kBase:kBase+kvDim], src[srcBase+qDim:srcBase+qDim+kvDim])
		if biasK != nil {
			for o := range kvDim {
				k[kBase+o] += biasK[o]
			}
		}

		// Copy V columns and add bias.
		copy(v[vBase:vBase+kvDim], src[srcBase+qDim+kvDim:srcBase+totalOut])
		if biasV != nil {
			for o := range kvDim {
				v[vBase+o] += biasV[o]
			}
		}
	}
}

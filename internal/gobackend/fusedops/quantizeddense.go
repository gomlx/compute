package fusedops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedQuantizedDense.Register(FusedQuantizedDense, gobackend.PriorityGeneric)
}

// nodeFusedQuantizedDense stores parameters for the quantized dense op.
//
// For the QuantGGML scheme, weights are stored in native GGML block format
// (see exec_fused_quantized_ggml.go for format details). The weight tensor is
// [N, bytesPerRow] Uint8, where N is the output-features dimension and
// bytesPerRow = (K / valuesPerBlock) * bytesPerBlock. K (input-features) is
// derived from bytesPerRow at build time via deriveGGMLK.
type nodeFusedQuantizedDense struct {
	scheme       compute.QuantizationScheme
	blockAxis    int // Always 1 (output-features axis); validated in builder. Stored for EqualNodeData.
	blockSize    int
	activation   compute.ActivationType
	hasZeroPoint bool
	hasBias      bool
	// ggmlType specifies the concrete GGML block format (Q4_0, Q8_0, etc.).
	// See exec_fused_quantized_ggml.go for block layouts and references.
	ggmlType compute.GGMLQuantType // Only used when scheme == QuantGGML.
	ggmlN    int                   // Output features (rows in GGML layout). Only for QuantGGML.
	ggmlK    int                   // Input features (logical columns). Only for QuantGGML.
}

func (d *nodeFusedQuantizedDense) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedQuantizedDense)
	return d.scheme == o.scheme && d.blockAxis == o.blockAxis &&
		d.blockSize == o.blockSize && d.activation == o.activation &&
		d.hasZeroPoint == o.hasZeroPoint && d.hasBias == o.hasBias &&
		d.ggmlType == o.ggmlType && d.ggmlN == o.ggmlN && d.ggmlK == o.ggmlK
}

// FusedQuantizedDense performs fused dequantization + matmul + optional bias + optional activation.
//
// Unlike FusedDense, this does not create a DotGeneral sub-node — the quantized matmul
// is fundamentally different (mixed-dtype with per-group scales). The inputs to the
// executor are [x, weights, scales, zeroPoints?, bias?] directly.
//
// Weights should have their dtype set to reflect the actual storage type (e.g. Int4, Int8).
// For sub-byte types, the caller should Bitcast packed byte data to the correct dtype.
func FusedQuantizedDense(
	f *gobackend.Function,
	x, weights, bias compute.Value,
	weightsQuantization *compute.Quantization,
	activation compute.ActivationType) (compute.Value, error) {

	scheme := weightsQuantization.Scheme

	// GGML weights have scales embedded in their native block format.
	// The weight layout is [N, bytesPerRow] Uint8 instead of [K, N].
	if scheme == compute.QuantGGML {
		return FusedQuantizedDenseGGML(f, x, weights, bias, weightsQuantization, activation)
	}

	scales := weightsQuantization.Scale
	zeroPoints := weightsQuantization.ZeroPoint
	blockAxis := weightsQuantization.BlockAxis
	blockSize := weightsQuantization.BlockSize

	values := []compute.Value{x, weights, scales}
	if zeroPoints != nil {
		values = append(values, zeroPoints)
	}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.VerifyAndCastValues("FusedQuantizedDense", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	weightsNode := inputs[1]
	scalesNode := inputs[2]

	// Validate x dtype: only float32 is supported.
	if xNode.Shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("FusedQuantizedDense: x must be float32, got %s", xNode.Shape.DType)
	}

	// Validate x shape: [batch..., K]
	if xNode.Shape.Rank() < 1 {
		return nil, errors.Errorf("FusedQuantizedDense: x must have rank >= 1, got %d", xNode.Shape.Rank())
	}
	K := xNode.Shape.Dimensions[xNode.Shape.Rank()-1]

	// Derive N from weights shape. The weights dtype reflects the storage type.
	if weightsNode.Shape.Rank() != 2 || weightsNode.Shape.Dimensions[0] != K {
		return nil, errors.Errorf("FusedQuantizedDense: weights must be [%d, N], got %v", K, weightsNode.Shape.Dimensions)
	}
	N := weightsNode.Shape.Dimensions[1]

	// Validate scales shape: [K, numBlocks]
	numBlocks := (N + blockSize - 1) / blockSize
	if scalesNode.Shape.Rank() != 2 || scalesNode.Shape.Dimensions[0] != K || scalesNode.Shape.Dimensions[1] != numBlocks {
		return nil, errors.Errorf("FusedQuantizedDense: scales must be [%d, %d], got %v",
			K, numBlocks, scalesNode.Shape.Dimensions)
	}

	// Output shape: [batch..., N]
	outDims := make([]int, xNode.Shape.Rank())
	copy(outDims, xNode.Shape.Dimensions[:xNode.Shape.Rank()-1])
	outDims[xNode.Shape.Rank()-1] = N
	outShape := shapes.Make(xNode.Shape.DType, outDims...)

	// Only blockAxis=1 (output-features axis) is currently supported.
	if blockAxis != 1 {
		return nil, errors.Errorf("FusedQuantizedDense: only Axis=1 is supported, got %d", blockAxis)
	}

	// NF4 quantization uses a fixed lookup table and does not support zero points.
	if scheme == compute.QuantNF4 && zeroPoints != nil {
		return nil, errors.Errorf("FusedQuantizedDense: ZeroPoint must be nil for NF4 quantization scheme")
	}

	data := &nodeFusedQuantizedDense{
		scheme:       scheme,
		blockAxis:    blockAxis,
		blockSize:    blockSize,
		activation:   activation,
		hasZeroPoint: zeroPoints != nil,
		hasBias:      bias != nil,
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedQuantizedDense, outShape, inputs, data)
	return node, nil
}

// validateGGMLTypeSupported checks that the given GGML type has a fused executor implementation.
func validateGGMLTypeSupported(opName string, ggmlType compute.GGMLQuantType) error {
	switch ggmlType {
	case compute.GGMLQ4_0, compute.GGMLQ8_0, compute.GGMLIQ4NL, compute.GGMLQ4_K, compute.GGMLQ6_K:
		return nil
	default:
		return errors.Wrapf(compute.ErrNotImplemented, "%s: GGML type %s not supported in fused path", opName, ggmlType)
	}
}

// deriveGGMLK computes the logical input-features dimension K from bytesPerRow
// and the GGML block format. GGML weights are stored as [N, bytesPerRow] Uint8,
// where each row consists of consecutive quantization blocks. Each block packs
// valuesPerBlock logical float32 values into bytesPerBlock bytes. K is therefore:
//
//	K = (bytesPerRow / bytesPerBlock) * valuesPerBlock
//
// This function validates that bytesPerRow is an exact multiple of bytesPerBlock.
//
// See exec_fused_quantized_ggml.go for per-type block layouts.
// Ref: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
func deriveGGMLK(opName string, bytesPerRow int, ggmlType compute.GGMLQuantType) (int, error) {
	vpb := ggmlType.ValuesPerBlock()
	bpb := ggmlType.BytesPerBlock()
	if vpb == 0 || bpb == 0 {
		return 0, errors.Errorf("%s: unsupported GGML type %s", opName, ggmlType)
	}
	if bytesPerRow%bpb != 0 {
		return 0, errors.Errorf("%s: bytesPerRow %d not divisible by bytesPerBlock %d for %s",
			opName, bytesPerRow, bpb, ggmlType)
	}
	return (bytesPerRow / bpb) * vpb, nil
}

// FusedQuantizedDenseGGML handles the GGML path for FusedQuantizedDense.
// GGML weights are [N, bytesPerRow] Uint8 with native block layout.
// Scales and zero points are embedded in the blocks.
func FusedQuantizedDenseGGML(f *gobackend.Function, x, weights, bias compute.Value,
	wq *compute.Quantization,
	activation compute.ActivationType) (compute.Value, error) {

	values := []compute.Value{x, weights}
	if bias != nil {
		values = append(values, bias)
	}
	inputs, err := f.VerifyAndCastValues("FusedQuantizedDense(GGML)", values...)
	if err != nil {
		return nil, err
	}
	xNode := inputs[0]
	wNode := inputs[1]

	// Validate x dtype: only float32 is supported.
	if xNode.Shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): x must be float32, got %s", xNode.Shape.DType)
	}
	if xNode.Shape.Rank() < 1 {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): x must have rank >= 1, got %d", xNode.Shape.Rank())
	}

	// GGML weights: [N, bytesPerRow] Uint8.
	if wNode.Shape.Rank() != 2 || wNode.Shape.DType != dtypes.Uint8 {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): weights must be [N, bytesPerRow] Uint8, got %v %s",
			wNode.Shape.Dimensions, wNode.Shape.DType)
	}
	N := wNode.Shape.Dimensions[0]
	bytesPerRow := wNode.Shape.Dimensions[1]

	ggmlType := wq.GGMLType
	if err := validateGGMLTypeSupported("FusedQuantizedDense(GGML)", ggmlType); err != nil {
		return nil, err
	}
	K, err := deriveGGMLK("FusedQuantizedDense(GGML)", bytesPerRow, ggmlType)
	if err != nil {
		return nil, err
	}

	// Validate that x's last dimension matches K.
	xK := xNode.Shape.Dimensions[xNode.Shape.Rank()-1]
	if xK != K {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): x's last dim (%d) must match K=%d derived from weights", xK, K)
	}

	// Zero points are not supported for GGML (embedded in blocks).
	if wq.ZeroPoint != nil {
		return nil, errors.Errorf("FusedQuantizedDense(GGML): ZeroPoint must be nil for GGML scheme")
	}

	// Output shape: [batch..., N]
	outDims := make([]int, xNode.Shape.Rank())
	copy(outDims, xNode.Shape.Dimensions[:xNode.Shape.Rank()-1])
	outDims[xNode.Shape.Rank()-1] = N
	outShape := shapes.Make(xNode.Shape.DType, outDims...)

	data := &nodeFusedQuantizedDense{
		scheme:     compute.QuantGGML,
		activation: activation,
		hasBias:    bias != nil,
		ggmlType:   ggmlType,
		ggmlN:      N,
		ggmlK:      K,
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeFusedQuantizedDense, outShape, inputs, data)
	return node, nil
}

// nodeQuantizedEmbeddingLookup stores parameters for the quantized embedding lookup op.
type nodeQuantizedEmbeddingLookup struct {
	ggmlType compute.GGMLQuantType
	ggmlK    int // Logical embedding dimension (valuesPerBlock * numBlocks).
}

func (d *nodeQuantizedEmbeddingLookup) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeQuantizedEmbeddingLookup)
	return d.ggmlType == o.ggmlType && d.ggmlK == o.ggmlK
}

// QuantizedEmbeddingLookup performs a quantized embedding lookup (row gather)
// with on-the-fly dequantization.
// data: [vocabSize, bytesPerRow] Uint8 with native GGML block layout.
// indices: integer tensor with last dim = 1 (same as Gather convention).
// Output: [batch..., K] Float32 where K is derived from the block format.

func init() {
	gobackend.RegisterQuantizedEmbeddingLookup.Register(QuantizedEmbeddingLookup, gobackend.PriorityGeneric)
}

func QuantizedEmbeddingLookup(f *gobackend.Function, data, indices *gobackend.Node,
	wq *compute.Quantization) (*gobackend.Node, error) {

	if wq.Scheme != compute.QuantGGML {
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"QuantizedEmbeddingLookup: only QuantGGML scheme is supported, got %s -- "+
				"please create a feature request if you need support for a different quantization scheme", wq.Scheme)
	}

	inputs, err := f.verifyAndCastValues("QuantizedEmbeddingLookup", data, indices)
	if err != nil {
		return nil, err
	}
	dNode := inputs[0]
	iNode := inputs[1]

	// Validate data: [vocabSize, bytesPerRow] Uint8.
	if dNode.Shape.Rank() != 2 || dNode.Shape.DType != dtypes.Uint8 {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: data must be [vocabSize, bytesPerRow] Uint8, got %v %s",
			dNode.Shape.Dimensions, dNode.Shape.DType)
	}
	bytesPerRow := dNode.Shape.Dimensions[1]

	// Validate indices: must be integer, last dim = 1.
	if !iNode.Shape.DType.IsInt() {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: indices must be integer, got %s", iNode.Shape.DType)
	}
	if iNode.Shape.Rank() < 1 || iNode.Shape.Dimensions[iNode.Shape.Rank()-1] != 1 {
		return nil, errors.Errorf("QuantizedEmbeddingLookup: indices last dim must be 1, got shape %v", iNode.Shape.Dimensions)
	}

	ggmlType := wq.GGMLType
	if err := validateGGMLTypeSupported("QuantizedEmbeddingLookup", ggmlType); err != nil {
		return nil, err
	}

	// Derive K from bytesPerRow.
	K, err := deriveGGMLK("QuantizedEmbeddingLookup", bytesPerRow, ggmlType)
	if err != nil {
		return nil, err
	}

	// Output shape: [batch..., K] Float32.
	// indices shape is [batch..., 1]. Output replaces the last dim with K.
	outDims := make([]int, iNode.Shape.Rank())
	copy(outDims, iNode.Shape.Dimensions)
	outDims[len(outDims)-1] = K
	outShape := shapes.Make(dtypes.Float32, outDims...)

	nodeData := &nodeQuantizedEmbeddingLookup{
		ggmlType: ggmlType,
		ggmlK:    K,
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeQuantizedEmbeddingLookup, outShape, inputs, nodeData)
	return node, nil
}

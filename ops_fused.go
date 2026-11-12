// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package compute

import "github.com/gomlx/compute/dtypes"

// ActivationType specifies the activation function for fused operations.
type ActivationType int

//go:generate go tool enumer -type ActivationType -trimprefix=Activation -output=gen_activationtype_enumer.go ops_fused.go

const (
	// ActivationNone specifies the identity function: y = x.
	ActivationNone ActivationType = iota

	// ActivationRelu is the Rectified Linear Unit activation:
	//
	//	y = max(x, 0)
	//
	// Introduced in "Rectified Linear Units Improve Restricted Boltzmann Machines"
	// (Vinod Nair, Geoffrey E. Hinton, ICML 2010), https://icml.cc/Conferences/2010/papers/432.pdf.
	ActivationRelu

	// ActivationSigmoid is the standard logistic sigmoid activation:
	//
	//	y = 1 / (1 + exp(-x))
	ActivationSigmoid

	// ActivationHardSigmoid is a piecewise linear approximation of the sigmoid function:
	//
	//	y = max(0, min(1, 0.2*x + 0.5))
	//
	// It is computationally cheaper on hardware without fast transcendental functions.
	ActivationHardSigmoid

	// ActivationLeakyRelu allows a small gradient when the unit is not active (x < 0):
	//
	//	y = x if x >= 0; 0.3*x if x < 0
	//
	// Here the negative slope alpha is fixed at 0.3.
	//
	// Introduced in "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
	// (Andrew L. Maas, Awni Y. Hannun, Andrew Y. Ng, ICML 2013).
	ActivationLeakyRelu

	// ActivationSelu stands for Scaled Exponential Linear Unit:
	//
	//	y = scale * x if x > 0; scale * alpha * (exp(x) - 1) if x <= 0
	//
	// with fixed constants alpha ≈ 1.67326324 and scale ≈ 1.05070098.
	//
	// Self-normalizing neural networks induce self-normalizing properties such as
	// variance stabilization.
	//
	// Introduced in "Self-Normalizing Neural Networks"
	// (Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter, NeurIPS 2017),
	// https://arxiv.org/abs/1706.02515.
	ActivationSelu

	// ActivationSilu (also known as Swish) is the Sigmoid-Weighted Linear Unit:
	//
	//	y = x * sigmoid(x)
	//
	// Introduced in "Gaussian Error Linear Units (GELUs)"
	// (Dan Hendrycks, Kevin Gimpel, 2016), https://arxiv.org/abs/1606.08415 and
	// "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
	// Reinforcement Learning" (Stefan Elfwing, Eiji Uchibe, Kenji Doya, 2017),
	// https://arxiv.org/abs/1702.03118, and independently discovered (and called Swish)
	// in "Searching for Activation Functions"
	// (Prajit Ramachandran, Barret Zoph, Quoc V. Le, 2017), https://arxiv.org/abs/1710.05941.
	ActivationSilu

	// ActivationHardSwish is a piecewise linear approximation of Swish:
	//
	//	y = x * ReLU6(x + 3) / 6 = x * max(0, min(1, x/6 + 0.5))
	//
	// Introduced in "Searching for MobileNetV3"
	// (Andrew Howard et al., ICCV 2019), https://arxiv.org/abs/1905.02244.
	ActivationHardSwish

	// ActivationTanh is the hyperbolic tangent activation:
	//
	//	y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
	ActivationTanh

	// ActivationGelu computes the Gaussian Error Linear Unit activation using the exact erf formulation:
	//
	//	y = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
	//
	// Introduced in "Gaussian Error Linear Units (GELUs)"
	// (Dan Hendrycks, Kevin Gimpel, 2016), https://arxiv.org/abs/1606.08415.
	ActivationGelu

	// ActivationGeluApproximate computes the GELU activation using the tanh approximation:
	//
	//	y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
	//
	// Introduced in "Gaussian Error Linear Units (GELUs)"
	// (Dan Hendrycks, Kevin Gimpel, 2016), https://arxiv.org/abs/1606.08415.
	ActivationGeluApproximate

	// ActivationSwiGLU applies the Swish-Gated Linear Unit activation to a pre-projected
	// tensor of shape [..., 2*hiddenDim], splitting the last dimension in half into gate
	// and value components:
	//
	//	SwiGLU(x) = Swish(x_gate) * x_value
	//
	// Output has shape [..., hiddenDim].
	//
	// Introduced in "GLU Variants Improve Transformer"
	// (Noam Shazeer, 2020), https://arxiv.org/abs/2002.05202.
	ActivationSwiGLU
)

// ActivationSwish is an alias for ActivationSilu.
const ActivationSwish = ActivationSilu

// VJPRequiresInput indicates whether the VJP of the activation requires the forward input `x`
// when the forward output `y` is already available.
//
// This indicates whether FusedDense+FusedDenseVJP can fuse the activation into a single
// kernel during training:
//   - If false: the VJP can be calculated solely from `y` and `dOutput` (i.e. `x` does not need
//     to be preserved during the forward pass). In this case, MatMul + Bias + Activation can be
//     fully fused during training.
//   - If true: the VJP requires the pre-activation input `x` (output of MatMul+Bias). In this
//     case, the activation cannot be fully fused with the dense layer during training, and
//     MatMul+Bias should be fused while leaving Activation as a separate node.
//
// Derivations when y is given:
//   - ActivationNone: dy/dx = 1 (no inputs needed).
//   - ActivationRelu: dy/dx = 1 if y > 0 else 0 (since y > 0 <=> x > 0).
//   - ActivationSigmoid: dy/dx = y * (1 - y).
//   - ActivationHardSigmoid: dy/dx = 0.2 if 0 < y < 1 else 0.
//   - ActivationLeakyRelu: dy/dx = 1 if y >= 0 else 0.3 (since alpha=0.3 > 0 preserves sign).
//   - ActivationSelu: dy/dx = scale if y > 0 else y + scale*alpha.
//   - ActivationTanh: dy/dx = 1 - y^2.
//   - ActivationSilu, ActivationHardSwish, ActivationGelu, ActivationGeluApproximate, ActivationSwiGLU:
//     require x for the derivative computation.
func (a ActivationType) VJPRequiresInput() bool {
	switch a {
	case ActivationNone,
		ActivationRelu,
		ActivationSigmoid,
		ActivationHardSigmoid,
		ActivationLeakyRelu,
		ActivationSelu,
		ActivationTanh:
		return false
	default:
		return true
	}
}

// ActivationConfig holds the configuration for activation functions.
// The Type is obligatory, other fields are optional and specific for activations.
type ActivationConfig struct {
	// Type of activation, see ActivationType.
	Type ActivationType
}

// AttentionAxesLayout specifies the ordering of axes in 4D attention tensors.
type AttentionAxesLayout int

const (
	// AttentionAxesLayoutBHSD is the [batch, heads, seq, dim] layout used by PyTorch's F.scaled_dot_product_attention,
	// ONNX, and most inference runtimes.
	AttentionAxesLayoutBHSD AttentionAxesLayout = iota

	// AttentionAxesLayoutBSHD is the [batch, seq, heads, dim] layout used internally by MultiHeadAttention
	// (after Dense projections which produce [batch, seq, heads, dim]).
	AttentionAxesLayoutBSHD
)

// String returns the name of the layout.
func (l AttentionAxesLayout) String() string {
	switch l {
	case AttentionAxesLayoutBHSD:
		return "BHSD"
	case AttentionAxesLayoutBSHD:
		return "BSHD"
	default:
		return "unknown"
	}
}

// SeqAxis returns the axis index for the sequence dimension.
func (l AttentionAxesLayout) SeqAxis() int {
	switch l {
	case AttentionAxesLayoutBSHD:
		return 1
	default: // AttentionAxesLayoutBHSD
		return 2
	}
}

// HeadsAxis returns the axis index for the heads dimension.
func (l AttentionAxesLayout) HeadsAxis() int {
	switch l {
	case AttentionAxesLayoutBSHD:
		return 2
	default: // AttentionAxesLayoutBHSD
		return 1
	}
}

// DenseLayout specifies the layout (axis ordering) of the weight matrix in Dense operations.
type DenseLayout int

const (
	// DenseLayoutInputOutputs specifies that weights have shape [in_features, out_features...] (0, default).
	DenseLayoutInputOutputs DenseLayout = iota
	// DenseLayoutOutputsInput specifies that weights have shape [out_features..., in_features].
	DenseLayoutOutputsInput
)

// String returns the name of the dense weight layout.
func (l DenseLayout) String() string {
	switch l {
	case DenseLayoutInputOutputs:
		return "InputOutputs"
	case DenseLayoutOutputsInput:
		return "OutputsInput"
	default:
		return "unknown"
	}
}

// DenseConfig holds configuration parameters for FusedDense operations.
type DenseConfig struct {
	// Activation is applied after the matmul+bias; set to ActivationNone for no activation.
	Activation ActivationConfig
	// WeightLayout specifies the layout of the weight matrix (DenseLayoutInputOutputs or DenseLayoutOutputsInput).
	WeightLayout DenseLayout
}

// QuantizationScheme specifies how quantized integer values map to floating-point values.
type QuantizationScheme int

const (
	// QuantLinear is standard affine quantization: real_value = (float(quant_value) - float(zero_point)) * scale.
	// Used with Int2/Uint2/Int4/Uint4/Int8/Uint8 weights or activations (symmetric when ZeroPoint == nil).
	QuantLinear QuantizationScheme = iota

	// QuantNF4 is 4-bit NormalFloat from QLoRA: nibble indices are looked up in a fixed
	// 16-entry table, then multiplied by scale.
	QuantNF4

	// QuantGGML indicates that the values are stored in native GGML block format
	// (e.g. Q4_0, Q8_0, K-quants, Q8_K). The scales and zero points are embedded in the
	// blocks themselves, so Scale/ZeroPoint/BlockAxis/BlockSize in Quantization are
	// unused; the GGMLType field specifies the concrete block format.
	// Weights must be [N, bytesPerRow] Uint8 with native GGML block layout;
	// GGML activations must be [batch..., bytesPerRow] Uint8 (GGMLQ8_0 or GGMLQ8_K).
	QuantGGML
)

// String returns the name of the quantization scheme.
func (q QuantizationScheme) String() string {
	switch q {
	case QuantLinear:
		return "Linear"
	case QuantNF4:
		return "NF4"
	case QuantGGML:
		return "GGML"
	default:
		return "unknown"
	}
}

// NF4LookupTable contains the 16 fixed QLoRA NormalFloat4 dequantization values.
// Used by both the fused executor and the decomposed graph-level fallback.
var NF4LookupTable = [16]float32{
	-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
	-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0,
	0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
	0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
}

// IQ4NLLookupTable contains the 16 fixed IQ4_NL non-linear dequantization values.
// These map 4-bit nibble indices to pre-normalization integer values (not final floats).
// Final dequantized value = per-block scale * IQ4NLLookupTable[nibble].
// Values from llama.cpp's kvalues_iq4nl.
var IQ4NLLookupTable = [16]float32{
	-127, -104, -83, -65, -49, -35, -22, -10,
	1, 13, 25, 38, 53, 69, 89, 113,
}

// GGMLQuantType identifies the specific GGML block quantization format.
// Enum values are aligned with go-highway's gguf.QuantType for future integration.
type GGMLQuantType int

const (
	GGMLQ4_0  GGMLQuantType = iota // 18 bytes/block, 32 values
	GGMLQ8_0                       // 34 bytes/block, 32 values
	GGMLIQ4NL                      // 18 bytes/block, 32 values (non-linear lookup)
	GGMLQ2_K                       // 84 bytes/block, 256 values
	GGMLQ3_K                       // 110 bytes/block, 256 values
	GGMLQ4_K                       // 144 bytes/block, 256 values
	GGMLQ5_K                       // 176 bytes/block, 256 values
	GGMLQ6_K                       // 210 bytes/block, 256 values
	GGMLQ8_K                       // 292 bytes/block, 256 values (used for paired K-quant activations)
	GGMLQ4_1                       // 20 bytes/block, 32 values (fp16 d + fp16 m + 16B nibbles)
	GGMLQ5_0                       // 22 bytes/block, 32 values (fp16 d + 4B high bits + 16B nibbles)
	GGMLQ5_1                       // 24 bytes/block, 32 values (fp16 d + fp16 m + 4B high bits + 16B nibbles)
)

// ValuesPerBlock returns the number of float32 values represented by one block.
func (t GGMLQuantType) ValuesPerBlock() int {
	switch t {
	case GGMLQ4_0, GGMLQ4_1, GGMLQ5_0, GGMLQ5_1, GGMLQ8_0, GGMLIQ4NL:
		return 32
	case GGMLQ2_K, GGMLQ3_K, GGMLQ4_K, GGMLQ5_K, GGMLQ6_K, GGMLQ8_K:
		return 256
	default:
		return 0
	}
}

// BytesPerBlock returns the byte size of one quantized block.
func (t GGMLQuantType) BytesPerBlock() int {
	switch t {
	case GGMLQ4_0, GGMLIQ4NL:
		return 18
	case GGMLQ4_1:
		return 20
	case GGMLQ5_0:
		return 22
	case GGMLQ5_1:
		return 24
	case GGMLQ8_0:
		return 34
	case GGMLQ2_K:
		return 84
	case GGMLQ3_K:
		return 110
	case GGMLQ4_K:
		return 144
	case GGMLQ5_K:
		return 176
	case GGMLQ6_K:
		return 210
	case GGMLQ8_K:
		return 292
	default:
		return 0
	}
}

// String returns the name of the GGML quantization type.
func (t GGMLQuantType) String() string {
	switch t {
	case GGMLQ4_0:
		return "Q4_0"
	case GGMLQ4_1:
		return "Q4_1"
	case GGMLQ5_0:
		return "Q5_0"
	case GGMLQ5_1:
		return "Q5_1"
	case GGMLQ8_0:
		return "Q8_0"
	case GGMLIQ4NL:
		return "IQ4_NL"
	case GGMLQ2_K:
		return "Q2_K"
	case GGMLQ3_K:
		return "Q3_K"
	case GGMLQ4_K:
		return "Q4_K"
	case GGMLQ5_K:
		return "Q5_K"
	case GGMLQ6_K:
		return "Q6_K"
	case GGMLQ8_K:
		return "Q8_K"
	default:
		return "unknown"
	}
}

// Quantization describes how a quantized tensor maps to floating-point values
// (or how a floating-point output is re-quantized).
//
// For QuantLinear, the dequantization formula is the standard integer zero-point
// convention shared by ONNX QuantizeLinear/DequantizeLinear/QLinearMatMul/MatMulNBits,
// TFLite, PyTorch, and CoreML:
//
//	real_value = (float(quant_value) - float(zero_point)) * scale
//
// When ZeroPoint is nil, zero_point = 0 (symmetric quantization: real_value = float(quant_value) * scale).
//
// Migration note for existing callers:
//   - Callers with standard pre-scale integer/float zero-points (e.g. ONNX, PyTorch, TFLite, GPTQ/AWQ qzeros)
//     can pass them directly as ZeroPoint without negating or pre-multiplying by Scale.
//   - Callers migrating from the legacy post-scale additive offset convention
//     (real_value = float(quant_value) * scale + offset) should convert the offset via
//     zero_point = -offset / scale (in weight-only float mode XQuantization == nil, float ZeroPoint
//     values are subtracted directly in Float32 without integer rounding, preserving exact equivalence).
//   - Scalar [] / [1] and 1D [N] Scale/ZeroPoint tensors can now be passed directly without
//     broadcasting across K to [K, NumBlocks].
type Quantization struct {
	// Scheme selects QuantLinear, QuantNF4, or QuantGGML.
	Scheme QuantizationScheme

	// Scale is the multiplicative scale factor (Float32, Float16, or BFloat16).
	// Supported shapes for a 2D weight tensor:
	//   - Scalar [] or [1]: per-tensor scale.
	//   - 1D [N]: per-output-channel scale (broadcast along K).
	//   - 2D [ceil(D0/BlockSize), D1] when BlockAxis == 0, or [D0, ceil(D1/BlockSize)] when BlockAxis == 1.
	//     Note: [1, N] on [K, N] weights is naturally BlockAxis=0, BlockSize=K (or BlockSize=0 meaning full axis).
	// Supported shapes for activations / outputs ([batch..., features]):
	//   - Scalar [] or [1] (per-tensor), [features] (per-channel), or broadcast-compatible per-token/per-block shapes.
	// Unused (must be nil) for QuantGGML.
	Scale Value

	// ZeroPoint is the pre-scale offset subtracted from quantized values (only for QuantLinear).
	// May have integer dtype matching the quantized tensor (e.g. Int8, Uint8, Int4, Uint4) or float dtype
	// matching Scale. Its shape must match Scale (or be broadcast-compatible with Scale).
	// Unused (must be nil) for QuantNF4 and QuantGGML.
	ZeroPoint Value

	// BlockAxis is the axis of the quantized tensor partitioned into blocks of BlockSize elements:
	//   - For 2D weights [K, N]: BlockAxis=0 blocks along K (standard GPTQ/AWQ/ONNX MatMulNBits [K/G, N]);
	//     BlockAxis=1 blocks along N ([K, N/G]).
	//   - For 2D weights [N, K]: BlockAxis=1 blocks along K ([N, K/G]); BlockAxis=0 blocks along N ([N/G, K]).
	// Ignored when Scale is scalar or 1D, or when Scheme == QuantGGML.
	BlockAxis int

	// BlockSize is the number of elements along BlockAxis sharing one scale/zero-point.
	// When BlockSize == 0 and Scale is 2D, BlockSize is inferred as ceil(dim(BlockAxis) / scale.Dim(BlockAxis))
	// when divisible.
	// Unused for QuantGGML.
	BlockSize int

	// GGMLType specifies the concrete GGML block format (Q4_0, Q8_0, Q4_K, Q8_K, etc.).
	// Only used when Scheme == QuantGGML.
	GGMLType GGMLQuantType

	// QuantizedDType optionally specifies the target quantized integer dtype (Int8 or Uint8)
	// when Quantization is used as OutputQuantization. If InvalidDType, the dtype is inferred
	// from ZeroPoint (if ZeroPoint is Int8 or Uint8), or defaults to Int8.
	QuantizedDType dtypes.DType
}

// FusedQuantizedDenseConfig configures FusedQuantizedDense and FusedQuantizedDenseVJP.
//
// Like ScaledDotProductAttentionConfig, fields fall into two categories:
//   - Correctness-affecting fields (WeightQuantization, XQuantization, OutputQuantization,
//     Activation, WeightLayout): backends that cannot honor the requested combination MUST
//     return errors.Wrapf(compute.ErrNotImplemented, ...) so callers fall back cleanly.
//   - Pure optimization hints (QuantizedMatmul): backends that do not support dynamic
//     integer matmul when XQuantization == nil may ignore the hint and compute in float.
type FusedQuantizedDenseConfig struct {
	// WeightQuantization describes how weight is quantized. Required (must not be nil).
	WeightQuantization *Quantization

	// XQuantization optionally describes a pre-quantized input activation x.
	//   - nil (default): x is floating-point (Float32, Float16, BFloat16).
	//   - QuantLinear: x is Int8 or Uint8 [batch..., K]; matmul accumulates in Int32
	//     (adding optional Int32 bias before rescaling, or floating-point bias after rescaling
	//     by xScale * wScale into Float32) before Activation and OutputQuantization.
	//   - QuantGGML: x is Uint8 [batch..., bytesPerRowX] with GGMLType == GGMLQ8_0 (paired with
	//     32-element weight blocks Q4_0/Q8_0/IQ4NL) or GGMLQ8_K (paired with 256-element K-quant
	//     weight blocks Q2_K..Q6_K).
	XQuantization *Quantization

	// OutputQuantization optionally specifies re-quantization of the post-activation output
	// (e.g. for ONNX QLinearMatMul where output is Int8 or Uint8 given yScale and yZeroPoint).
	// When nil (default), the output is floating-point (matching x's float dtype, or Float32
	// when x is pre-quantized).
	OutputQuantization *Quantization

	// QuantizedMatmul is an optimization hint used only when XQuantization == nil:
	// if true, backends with fast integer dot-product kernels may dynamically quantize
	// floating-point x on-the-fly (e.g. to Q8_0/Q8_K blocks for QuantGGML weights, or
	// per-row symmetric Int8 for QuantLinear weights) and accumulate in Int32.
	// Backends that do not support on-the-fly dynamic quantization ignore this hint.
	QuantizedMatmul bool

	// Activation is applied after matmul/rescaling and bias (and before OutputQuantization if set).
	// Defaults to ActivationNone (zero value).
	Activation ActivationConfig

	// WeightLayout specifies the logical axis order of weight:
	//   - DenseLayoutInputOutputs (0, default for Linear/NF4): weight is [K, N].
	//   - DenseLayoutOutputsInput (1, required for QuantGGML, also supported for Linear/NF4):
	//     weight is [N, K] (or [N, bytesPerRow] for QuantGGML).
	WeightLayout DenseLayout
}

// ScaledDotProductAttentionConfig holds optional optimization hints and fused-attention
// parameters for FusedScaledDotProductAttention.
// A nil *ScaledDotProductAttentionConfig means "use defaults" (all optimizations disabled).
// Correctness-affecting fields (Bias, QuerySeqLen, KeyValueSeqLen, dropout, etc.): backends
// that cannot honor them MUST return ErrNotImplemented so the caller falls back to the
// decomposed path.
// Pure optimization hints (e.g. QuantizedMatmuls): backends that do not support them may
// silently ignore them and compute a numerically equivalent result in float arithmetic.
// nil/zero means "unused".
type ScaledDotProductAttentionConfig struct {
	// QuantizedMatmuls: if true, the backend may use dynamic per-head symmetric
	// affine quantization (scale-only, no zero point) to convert float32 Q/K/V slices
	// to uint8 for the Q@K^T and attn@V matmul stages. Accumulation is done in int32,
	// then dequantized back to float32. Softmax and masking remain in float32.
	// This matches ONNX DynamicQuantizeLinear semantics and trades some numerical
	// precision for throughput on hardware with fast integer dot-product instructions
	// (e.g. ARM SDOT/UDOT, x86 VNNI). Backends that do not support quantized matmuls
	// ignore this flag and use float arithmetic.
	QuantizedMatmuls bool

	// QuerySeqLen, KeyValueSeqLen are optional per-batch actual sequence lengths
	// (int32 tensors, shape [B]). When set, the backend masks by sequence length
	// (padding mask) instead of a materialized [S,Skv] mask. Combined with causal=true
	// this is a padding-causal mask. nil = unused.
	QuerySeqLen, KeyValueSeqLen Value

	// Bias is an optional additive attention-score bias broadcast to [B,H,S,Skv]
	// (ALiBi / relative-position). NOT the Q/K/V projection bias. nil = unused.
	Bias Value

	// Scale is the scaling factor applied to query @ key^T.
	// If 0 (the default), it means 1/sqrt(headDim).
	Scale float64

	// Causal specifies whether to apply causal (lower-triangular) masking.
	Causal bool

	// Mask is an optional attention mask of shape [seqLen, kvLen] (or broadcastable).
	// Boolean mask: true = attend, false = ignore.
	// Float/additive mask: added to scores before softmax.
	Mask Value
}

// FusedOps defines optional fused operations. Backends may implement these for
// better performance; the graph layer falls back to decomposed operations when
// unavailable.
//
// Like with standard ops, if the backend doesn't implement the fused op, return
// ErrNotImplemented (wrapped with a stack).
type FusedOps interface {
	// FusedActivation applies the configured activation function.
	// cfg.Type is obligatory, other fields are optional and specific for activations.
	FusedActivation(x Value, cfg ActivationConfig) (Value, error)

	// FusedActivationVJP computes the vector-jacobian product of the configured activation:
	// dx = dOutput * f'(x) (or dOutput * f'(y)).
	//
	// Parameters:
	//   - y: output of the activation, calculated presumably by FusedActivation.
	//   - x: input to the activation. If cfg.Type.VJPRequiresInput() is false, x can be nil if y is provided.
	//   - dOutput: the incoming adjoint gradient (the "V" in "VJP"), with the same shape as y.
	//   - cfg: the activation configuration.
	FusedActivationVJP(y, x, dOutput Value, cfg ActivationConfig) (Value, error)

	// FusedSoftmax computes softmax along the specified axis.
	//
	// Note: unlike the generic softmax in GoMLX's graph package, the fused
	// softmax only accepts one axis. The axis must be non-negative (the caller
	// normalizes negative indices before calling).
	FusedSoftmax(x Value, axis int) (Value, error)

	// FusedLayerNorm applies layer normalization over specified axes.
	// gamma and beta can be nil if no learned scale/offset.
	// epsilon: numerical stability constant (typically 1e-5).
	FusedLayerNorm(x Value, axes []int, epsilon float64, gamma, beta Value) (Value, error)

	// FusedLayerNormVJP computes the vector-jacobian product (gradients) of FusedLayerNorm
	// with respect to its inputs (x, gamma, and beta).
	//
	// Parameters:
	//   - x: original input tensor to FusedLayerNorm.
	//   - dOutput: incoming adjoint gradient with respect to the output of FusedLayerNorm (same shape as x).
	//   - axes: normalization axes passed to the forward FusedLayerNorm call.
	//   - epsilon: numerical stability constant passed to the forward FusedLayerNorm call.
	//   - gamma: scale tensor passed to FusedLayerNorm (can be nil if FusedLayerNorm was called without gamma).
	//   - beta: offset tensor passed to FusedLayerNorm (can be nil if FusedLayerNorm was called without beta).
	//
	// Outputs:
	//   - dx: gradient with respect to x (same shape as x).
	//   - dGamma: gradient with respect to gamma (same shape as gamma, or nil if gamma was nil).
	//   - dBeta: gradient with respect to beta (same shape as beta, or nil if beta was nil).
	FusedLayerNormVJP(x, dOutput Value, axes []int, epsilon float64, gamma, beta Value) (dx, dGamma, dBeta Value, err error)

	// FusedDense performs fused matmul + optional bias + optional activation.
	//
	// It does y = activation(x @ W + bias) for DenseLayoutInputOutputs, or
	// y = activation(x @ W^T + bias) for DenseLayoutOutputsInput.
	// Where @ is a standard matmul.
	//
	// - x: [batch..., in_features]
	// - weight: [in_features, out_features...] (if WeightLayout is DenseLayoutInputOutputs)
	//   or [out_features..., in_features] (if WeightLayout is DenseLayoutOutputsInput)
	// - bias: [out_features...] (nil-able).
	// - options: DenseConfig options (activation and weight layout).
	//
	// Note: ActivationSwiGLU is explicitly not supported for FusedDense because it halves the
	// last dimension (output shape changes). This would require allocating temporary memory
	// and complicate the matmul implementation, eliminating the cache-locality gains of epilogue
	// fusion. Instead, use FusedDense with no activation, followed by FusedActivation with
	// ActivationSwiGLU as a separate operation.
	FusedDense(x, weight, bias Value, options DenseConfig) (Value, error)

	// FusedDenseVJP computes the vector-jacobian product (gradients) of FusedDense with respect to
	// its inputs (x, weight, and bias).
	//
	// Parameters:
	//   - x: original input tensor to FusedDense, of shape [batch..., in_features].
	//   - weight: weight tensor to FusedDense.
	//   - bias: bias tensor to FusedDense (can be nil if FusedDense was called without bias).
	//   - y: output of the forward FusedDense call.
	//   - dOutput: incoming adjoint gradient with respect to y (same shape as y).
	//   - options: configuration passed to the forward FusedDense call.
	//
	// Outputs:
	//   - dx: gradient with respect to x (same shape as x).
	//   - dWeight: gradient with respect to weight (same shape as weight).
	//   - dBias: gradient with respect to bias (same shape as bias, or nil if bias was nil).
	//
	// Note: FusedDenseVJP only works for activations that return false to
	// ActivationType.VJPRequiresInput() (such as ActivationNone, ActivationRelu, ActivationSigmoid,
	// ActivationHardSigmoid, ActivationLeakyRelu, ActivationSelu, and ActivationTanh).
	// If the activation requires the input (where VJPRequiresInput() returns true, such as
	// ActivationSilu, ActivationHardSwish, ActivationGelu, and ActivationGeluApproximate), it is
	// recommended to use FusedDense without an activation (ActivationNone), and then use
	// FusedActivation separately so that the pre-activation intermediate can be preserved for
	// FusedActivationVJP.
	FusedDenseVJP(x, weight, bias, y, dOutput Value, options DenseConfig) (dx, dWeight, dBias Value, err error)

	// FusedScaledDotProductAttention computes multi-head scaled dot-product attention.
	//
	// output = softmax(query @ key^T * scale + mask) @ value, computed per-head with GQA support.
	//
	// Conventions:
	// - "numHeads" refers to the number of heads for the query, and "numKVHeads" for the key and value.
	// - "seqLen" refers to the sequence length of the query, and "kvLen" is the sequence length for the key and value.
	//
	// Inputs:
	//   - query, key, value: 4D tensors whose axis ordering is determined by axesLayout.
	//     For AttentionAxesLayoutBHSD: query [batch, numHeads, seqLen, headDim],
	//                         key/value [batch, numKVHeads, kvLen, headDim].
	//     For AttentionAxesLayoutBSHD: query [batch, seqLen, numHeads, headDim],
	//                         key/value [batch, kvLen, numKVHeads, headDim].
	//   - axesLayout: determines the axis ordering of query/key/value tensors
	//   - options: optional parameters (nil uses defaults). See ScaledDotProductAttentionConfig.
	//
	// Outputs:
	//   - output: same shape as query.
	//   - statesForVJP: optional (may be nil/empty). When the backend supports a fused backward
	//     (FusedScaledDotProductAttentionVJP), it returns whatever backward-pass state that
	//     backend's fused VJP needs.
	//     The number, semantics and shapes of these values are backend-specific; the VJP is only
	//     ever called with the exact slice this call returned.
	//     Backends without a fused backward return nil/empty here and ErrNotImplemented from the
	//     VJP, so the caller differentiates the decomposed attention instead.
	FusedScaledDotProductAttention(
		query, key, value Value,
		axesLayout AttentionAxesLayout,
		options *ScaledDotProductAttentionConfig) (output Value, statesForVJP []Value, err error)

	// FusedScaledDotProductAttentionVJP computes the gradients (dQuery, dKey, dValue) of
	// FusedScaledDotProductAttention given the forward output, the statesForVJP it returned, and
	// the adjoint output gradient dOutput. The query/key/value and options parameters are
	// the same values passed to the forward call.
	//
	// Backends that return non-nil/non-empty statesForVJP from the forward and implement a fused
	// backward (e.g. the cuDNN flash backward) implement this. Others return ErrNotImplemented so
	// the caller falls back to differentiating the decomposed attention.
	FusedScaledDotProductAttentionVJP(
		query, key, value Value,
		axesLayout AttentionAxesLayout,
		options *ScaledDotProductAttentionConfig,
		output Value, statesForVJP []Value, dOutput Value) (dQuery, dKey, dValue Value, err error)

	// QuantizedEmbeddingLookup performs a quantized embedding lookup (row gather)
	// with on-the-fly dequantization.
	//
	// This is the quantized analogue of Gather for embedding lookups, inspired by
	// llama.cpp's ggml_get_rows. For now it is only implemented for the GGML
	// quantization scheme, but could be extended for others if/when needed.
	//
	// Inputs:
	//   - data: [vocabSize, bytesPerRow] Uint8 with native GGML block layout.
	//   - indices: integer tensor with last dimension = 1 (same as Gather convention).
	//     For embeddings: [batch, seqLen, 1].
	//   - dataQuantization: describes how to dequantize the data rows. Must not be nil.
	//     Only QuantGGML scheme is currently supported.
	//
	// Output: float32 tensor with shape [batch..., K] where K = (bytesPerRow / bytesPerBlock) * valuesPerBlock.
	//   For embeddings with indices [batch, seqLen, 1]: output is [batch, seqLen, K].
	QuantizedEmbeddingLookup(data, indices Value,
		dataQuantization *Quantization) (Value, error)

	// FusedQuantizedDense performs fused quantized matrix multiplication + optional bias +
	// optional activation (+ optional output re-quantization), without materializing the full
	// dequantized weight tensor in memory:
	//
	//   - Weight-only quantization (options.XQuantization == nil):
	//     z = x @ dequant(weight, options.WeightQuantization) + bias
	//     (transposing dequant(weight) when options.WeightLayout == DenseLayoutOutputsInput; if
	//     options.QuantizedMatmul is true, backends may dynamically quantize x internally to run
	//     integer SIMD dot products).
	//   - Pre-quantized x + weight integer matmul (options.XQuantization != nil):
	//     supports QuantLinear (exact Int32 accumulation of (x - xZP) @ (w - wZP) rescaled by
	//     xScale * wScale) and QuantGGML (GGMLQ8_0 / GGMLQ8_K paired per-block Int32 dot products
	//     scaled by d_x * d_w into Float32).
	//   - Bias conventions (when bias != nil, shape [N]):
	//     Floating-point bias (matching x's float DType when XQuantization == nil, or
	//     Float32/Float16/BFloat16 when XQuantization != nil) is already in real units and is added
	//     in floating-point AFTER dequantization / (xScale * wScale) rescaling and BEFORE
	//     options.Activation / options.OutputQuantization.
	//     Int32 bias (valid only for QuantLinear XQuantization when neither XQuantization.Scale nor
	//     WeightQuantization.Scale is sub-blocked along K) is pre-quantized at scale xScale * wScale[n]
	//     (i.e. bias_int32[n] = round(bias_float[n] / (xScale * wScale[n]))) and is added directly to
	//     the Int32 accumulator BEFORE converting to Float32 and multiplying by xScale * wScale:
	//     z[m, n] = float32(acc_int32[m, n] + bias[n]) * (xScale[m] * wScale[n]).
	//   - Optional output quantization (options.OutputQuantization != nil):
	//     re-quantizes the post-activation float output to Int8 or Uint8.
	//
	// Inputs:
	//   - x: [batch..., K] floating-point (when XQuantization == nil) or quantized integer/block
	//     tensor ([batch..., K] Int8/Uint8 for QuantLinear, [batch..., bytesPerRowX] Uint8 for QuantGGML).
	//   - weight: [K, N] (DenseLayoutInputOutputs) or [N, K] (DenseLayoutOutputsInput) with DType
	//     Int2/Uint2/Int4/Uint4/Int8/Uint8 for QuantLinear/QuantNF4; or [N, bytesPerRow] Uint8 for
	//     QuantGGML (requires DenseLayoutOutputsInput, bytesPerRow = (K / valuesPerBlock) * bytesPerBlock).
	//   - bias: optional [N] tensor (nil for no bias); floating-point (added after dequantization/rescaling)
	//     or Int32 (added to the Int32 accumulator before xScale * wScale rescaling when XQuantization uses
	//     QuantLinear without K sub-blocking), applied before options.Activation and options.OutputQuantization.
	//   - options: FusedQuantizedDenseConfig (WeightQuantization must not be nil).
	//
	// Note: Like FusedDense, ActivationSwiGLU is not supported here because it halves the output dimension;
	// emit FusedQuantizedDense with ActivationNone followed by FusedActivation(ActivationSwiGLU).
	//
	// Migration note: quantization, layout, and activation options are grouped in FusedQuantizedDenseConfig,
	// QuantGGML weights require WeightLayout = DenseLayoutOutputsInput, scalar [] and 1D [N] scales no longer
	// need [K, NumBlocks] broadcasts, and QuantLinear ZeroPoint uses the standard pre-scale convention
	// real = (float(q) - float(zp)) * scale (legacy additive offsets convert via zero_point = -offset / scale).
	FusedQuantizedDense(x, weight, bias Value, options FusedQuantizedDenseConfig) (Value, error)

	// FusedQuantizedDenseVJP computes the vector-Jacobian product (backward pass) of FusedQuantizedDense
	// with respect to floating-point x and optional floating-point bias, keeping quantized weight frozen
	// (essential for QLoRA and quantized fine-tuning where gradients flow through frozen base weights into
	// trainable adapters without materializing [K, N] float weights in memory).
	//
	// Let z = x @ dequant(weight) + bias and y = activation(z). Given upstream dOutput = dL/dy:
	//   dZ    = dOutput * activation'(z)   (derived from y when Activation != ActivationNone)
	//   dx    = dZ @ dequant(weight)^T     (respecting options.WeightLayout)
	//   dBias = sum_{batch...}(dZ)         (nil if bias == nil)
	//
	// Memory-saving nil inputs:
	//   - x may be nil if !options.Activation.VJPRequiresInput() (since quantized weights are frozen,
	//     dWeight is not computed, so x does not need to be saved across the forward-backward boundary!).
	//   - y may be nil when options.Activation.Type == ActivationNone.
	//
	// Only supported when options.XQuantization == nil and options.OutputQuantization == nil.
	FusedQuantizedDenseVJP(x, weight, bias, y, dOutput Value, options FusedQuantizedDenseConfig) (dx, dBias Value, err error)

	// FusedAttentionQKVProjection performs fused Query-Key-Value projection: a single large matmul
	// merged with a scatter into separate query (Q), key (K), value (V) outputs with optional
	// per-projection bias.
	//
	// The caller is expected to flatten any leading dimensions (e.g. batch and sequence) into a
	// single "batch" axis before calling, and reshape the outputs afterwards. For example, with
	// BSHD layout the caller reshapes [batch, seqLen, inFeatures] → [batch*seqLen, inFeatures],
	// calls this method, then reshapes each output back to [batch, seqLen, ...].
	//
	// Inputs:
	//   - x: [batch, inFeatures] (batch may include a merged sequence dimension)
	//   - wQKV: [inFeatures, queryDim+2*keyValueDim] (Q/K/V weights concatenated along last axis)
	//   - biasQ: [queryDim] (optional, nil for no bias)
	//   - biasK: [keyValueDim] (optional, nil for no bias)
	//   - biasV: [keyValueDim] (optional, nil for no bias)
	//
	// Parameters:
	//   - queryDim: output dimension for query projection
	//   - keyValueDim: output dimension for key and value projections
	//
	// Outputs: query [batch, queryDim], key [batch, keyValueDim], value [batch, keyValueDim]
	FusedAttentionQKVProjection(
		x, wQKV, biasQ, biasK, biasV Value,
		queryDim, keyValueDim int) (
		query, key, value Value, err error)
}

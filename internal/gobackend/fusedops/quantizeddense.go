// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusedops

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/activations"
	"github.com/gomlx/compute/internal/gobackend/dot"
	"github.com/gomlx/compute/internal/gobackend/ops"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterFusedQuantizedDense.Register(FusedQuantizedDense, gobackend.PriorityGeneric)
	gobackend.RegisterFusedQuantizedDenseVJP.Register(FusedQuantizedDenseVJP, gobackend.PriorityGeneric)

	gobackend.SetNodeExecutor(compute.OpTypeFusedQuantizedDense, gobackend.PriorityTyped, execFusedQuantizedDense)
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeFusedQuantizedDenseVJP] = execFusedQuantizedDenseVJP
}

type weightCacheKind uint8

const (
	weightCacheNone weightCacheKind = iota
	weightCacheGGMLPanel
	weightCacheLinearPanel
	weightCacheFloatMatmul
)

type supportCacheKind uint8

const (
	supportCacheNone supportCacheKind = iota
	supportCacheF32
	supportCacheI32
)

// bakedConstantSentinel is stored in ConstantData.cachedForm for supporting constants
// (wScale, wZeroPoint) whose values have been baked into the associated weight's cached
// panel so their donated raw buffers can be released cleanly.
type bakedConstantSentinel struct {
	baked bool
}

func markConstantBaked(cd *gobackend.ConstantData) {
	if cd == nil {
		return
	}
	_ = gobackend.GetOrSetCachedForm(cd, func() bakedConstantSentinel {
		return bakedConstantSentinel{baked: true}
	})
}

func supportCacheSig(k supportCacheKind) string {
	switch k {
	case supportCacheF32:
		return "QDF32Slice"
	case supportCacheI32:
		return "QDI32Slice"
	default:
		return ""
	}
}

func shouldCacheByPolicy(policy gobackend.QuantizedCachePolicy, rawBytes, cachedBytes int64, willFreeRaw bool) bool {
	switch policy {
	case gobackend.QuantizedCacheNever:
		return false
	case gobackend.QuantizedCacheAlways:
		return cachedBytes > 0
	case gobackend.QuantizedCacheBalanced:
		if cachedBytes <= 0 || cachedBytes > gobackend.DefaultQuantizedCacheMaxWeightBytes || rawBytes <= 0 {
			return false
		}
		var extraBytes int64
		if willFreeRaw {
			extraBytes = max(int64(0), cachedBytes-rawBytes)
		} else {
			extraBytes = cachedBytes
		}
		return float64(extraBytes) <= gobackend.DefaultQuantizedCacheMaxExtraRatio*float64(rawBytes)
	default:
		return false
	}
}

// nodeFusedQuantizedDense stores compile-time validated metadata for FusedQuantizedDense.
//
// Node inputs order:
//
//	[x, weight, wScale?, wZeroPoint?, xScale?, xZeroPoint?, outScale?, outZeroPoint?, bias?]
type nodeFusedQuantizedDense struct {
	weightScheme     compute.QuantizationScheme
	weightGGMLType   compute.GGMLQuantType
	weightDType      dtypes.DType
	wZPDType         dtypes.DType
	weightLayout     compute.DenseLayout
	hasXQuant        bool
	xScheme          compute.QuantizationScheme
	xGGMLType        compute.GGMLQuantType
	hasOutputQuant   bool
	outputQuantDType dtypes.DType
	quantizedMatmul  bool
	activation       compute.ActivationConfig

	hasWScale   bool
	hasWZP      bool
	hasXScale   bool
	hasXZP      bool
	hasOutScale bool
	hasOutZP    bool
	hasBias     bool

	// Input slot indices in node.Inputs (-1 if absent).
	wScaleIdx   int
	wZPIdx      int
	xScaleIdx   int
	xZPIdx      int
	outScaleIdx int
	outZPIdx    int
	biasIdx     int

	info shapeinference.QuantizedDenseInfo

	// Compile-time constant caching metadata & policy.
	cachePolicy       gobackend.QuantizedCachePolicy
	weightCacheKind   weightCacheKind
	weightCacheSig    string
	weightRawBytes    int64
	weightCachedBytes int64
	floatLayout       dot.Layout

	wScaleCacheKind   supportCacheKind
	wScaleRawBytes    int64
	wScaleCachedBytes int64
	wZPCacheKind      supportCacheKind
	wZPRawBytes       int64
	wZPCachedBytes    int64
	xScaleCacheKind   supportCacheKind
	xScaleRawBytes    int64
	xScaleCachedBytes int64
	xZPCacheKind      supportCacheKind
	xZPRawBytes       int64
	xZPCachedBytes    int64
	outScaleCacheKind supportCacheKind
	outScaleRawBytes  int64
	outScaleCachedB   int64
	outZPCacheKind    supportCacheKind
	outZPRawBytes     int64
	outZPCachedBytes  int64
	biasCacheKind     supportCacheKind
	biasRawBytes      int64
	biasCachedBytes   int64

	// Bound shared ConstantData handles populated by setupConstantSubgraphCaching.
	wConstantData        *gobackend.ConstantData
	floatDotNodeData     *dot.NodeData
	wScaleConstantData   *gobackend.ConstantData
	wScaleBaked          bool
	wZPConstantData      *gobackend.ConstantData
	wZPBaked             bool
	xScaleConstantData   *gobackend.ConstantData
	xZPConstantData      *gobackend.ConstantData
	outScaleConstantData *gobackend.ConstantData
	outZPConstantData    *gobackend.ConstantData
	biasConstantData     *gobackend.ConstantData

	actShareGroup *sharedQuantActGroup
}

var (
	_ gobackend.FusedDenseActivationTarget        = (*nodeFusedQuantizedDense)(nil)
	_ gobackend.FusedQuantizedDenseActShareTarget = (*nodeFusedQuantizedDense)(nil)
	_ gobackend.ConstantCacheNodeData             = (*nodeFusedQuantizedDense)(nil)
	_ gobackend.ConditionalConstantCacheNodeData  = (*nodeFusedQuantizedDense)(nil)
	_ gobackend.RecomputableNodeData              = (*nodeFusedQuantizedDense)(nil)
)

func (d *nodeFusedQuantizedDense) QuantActShareKey() (string, bool) {
	if d == nil || d.hasXQuant || !d.quantizedMatmul || d.info.K <= 0 || d.info.N <= 0 {
		return "", false
	}
	switch d.info.FloatComputeDType {
	case dtypes.Float32, dtypes.Float16, dtypes.BFloat16:
	default:
		return "", false
	}
	switch d.weightScheme {
	case compute.QuantGGML:
		subBlockSize, _, ok := ggmlPanelSpec(d.weightGGMLType)
		if !ok || d.info.K%subBlockSize != 0 {
			return "", false
		}
		q8K := d.weightGGMLType.ValuesPerBlock() == 256
		return fmt.Sprintf("GGML:K=%d,sub=%d,q8k=%t", d.info.K, subBlockSize, q8K), true
	case compute.QuantLinear:
		kBlock, ok := linearPanelKBlock(d)
		if !ok {
			return "", false
		}
		return fmt.Sprintf("Linear:K=%d,kBlock=%d", d.info.K, kBlock), true
	default:
		return "", false
	}
}

func (d *nodeFusedQuantizedDense) BindSharedQuantActSiblings(siblings []gobackend.FusedQuantizedDenseActShareTarget) {
	if len(siblings) < 2 {
		return
	}
	group := &sharedQuantActGroup{numConsumers: len(siblings)}
	for _, s := range siblings {
		if nd, ok := s.(*nodeFusedQuantizedDense); ok {
			nd.actShareGroup = group
		}
	}
}

func (d *nodeFusedQuantizedDense) weightIsCached() bool {
	if d == nil {
		return false
	}
	return d.wConstantData != nil || (d.floatDotNodeData != nil && d.floatDotNodeData.RHSConstantData != nil)
}

func (d *nodeFusedQuantizedDense) weightCachedFormReady() bool {
	if d == nil {
		return false
	}
	if d.wConstantData != nil {
		return d.wConstantData.CachedForm() != nil
	}
	if d.floatDotNodeData != nil && d.floatDotNodeData.RHSConstantData != nil {
		return d.floatDotNodeData.RHSConstantData.CachedForm() != nil
	}
	return false
}

func (d *nodeFusedQuantizedDense) markBakedWeightSupportConstants() {
	if d == nil {
		return
	}
	if d.wScaleBaked && d.wScaleConstantData != nil {
		markConstantBaked(d.wScaleConstantData)
	}
	if d.wZPBaked && d.wZPConstantData != nil {
		markConstantBaked(d.wZPConstantData)
	}
}

func (d *nodeFusedQuantizedDense) ConstantCachePriority(inputIdx int) int {
	if d == nil {
		return 0
	}
	if d.weightCacheKind != weightCacheNone && (inputIdx == d.wScaleIdx || inputIdx == d.wZPIdx) {
		return 1
	}
	return 0
}

func (d *nodeFusedQuantizedDense) CacheSignature(inputIdx int) string {
	if d == nil || d.cachePolicy == gobackend.QuantizedCacheNever {
		return ""
	}
	switch inputIdx {
	case 1:
		if d.weightCacheKind == weightCacheNone {
			return ""
		}
		return d.weightCacheSig
	case d.wScaleIdx:
		if d.wScaleIdx < 0 {
			return ""
		}
		if d.weightIsCached() {
			return fmt.Sprintf("QDBaked(%s,slot=wScale)", d.weightCacheSig)
		}
		return supportCacheSig(d.wScaleCacheKind)
	case d.wZPIdx:
		if d.wZPIdx < 0 {
			return ""
		}
		if d.weightIsCached() {
			return fmt.Sprintf("QDBaked(%s,slot=wZP)", d.weightCacheSig)
		}
		return supportCacheSig(d.wZPCacheKind)
	case d.xScaleIdx:
		if d.xScaleIdx < 0 {
			return ""
		}
		return supportCacheSig(d.xScaleCacheKind)
	case d.xZPIdx:
		if d.xZPIdx < 0 {
			return ""
		}
		return supportCacheSig(d.xZPCacheKind)
	case d.outScaleIdx:
		if d.outScaleIdx < 0 {
			return ""
		}
		return supportCacheSig(d.outScaleCacheKind)
	case d.outZPIdx:
		if d.outZPIdx < 0 {
			return ""
		}
		return supportCacheSig(d.outZPCacheKind)
	case d.biasIdx:
		if d.biasIdx < 0 {
			return ""
		}
		return supportCacheSig(d.biasCacheKind)
	default:
		return ""
	}
}

func (d *nodeFusedQuantizedDense) ShouldCacheInput(inputIdx int, _ *gobackend.ConstantData, willFreeRaw bool) bool {
	if d == nil || d.cachePolicy == gobackend.QuantizedCacheNever {
		return false
	}
	switch inputIdx {
	case 1:
		if d.weightCacheKind == weightCacheNone {
			return false
		}
		rawBytes := d.weightRawBytes + d.wScaleRawBytes + d.wZPRawBytes
		return shouldCacheByPolicy(d.cachePolicy, rawBytes, d.weightCachedBytes, willFreeRaw)
	case d.wScaleIdx:
		if d.wScaleIdx < 0 {
			return false
		}
		if d.weightIsCached() {
			// Already baked into the cached weight panel: 0 extra bytes and releases raw scale if donated.
			return true
		}
		return d.wScaleCacheKind != supportCacheNone &&
			shouldCacheByPolicy(d.cachePolicy, d.wScaleRawBytes, d.wScaleCachedBytes, willFreeRaw)
	case d.wZPIdx:
		if d.wZPIdx < 0 {
			return false
		}
		if d.weightIsCached() {
			// Already baked into the cached weight panel: 0 extra bytes and releases raw zero-point if donated.
			return true
		}
		return d.wZPCacheKind != supportCacheNone &&
			shouldCacheByPolicy(d.cachePolicy, d.wZPRawBytes, d.wZPCachedBytes, willFreeRaw)
	case d.xScaleIdx:
		return d.xScaleIdx >= 0 && d.xScaleCacheKind != supportCacheNone &&
			shouldCacheByPolicy(d.cachePolicy, d.xScaleRawBytes, d.xScaleCachedBytes, willFreeRaw)
	case d.xZPIdx:
		return d.xZPIdx >= 0 && d.xZPCacheKind != supportCacheNone &&
			shouldCacheByPolicy(d.cachePolicy, d.xZPRawBytes, d.xZPCachedBytes, willFreeRaw)
	case d.outScaleIdx:
		return d.outScaleIdx >= 0 && d.outScaleCacheKind != supportCacheNone &&
			shouldCacheByPolicy(d.cachePolicy, d.outScaleRawBytes, d.outScaleCachedB, willFreeRaw)
	case d.outZPIdx:
		return d.outZPIdx >= 0 && d.outZPCacheKind != supportCacheNone &&
			shouldCacheByPolicy(d.cachePolicy, d.outZPRawBytes, d.outZPCachedBytes, willFreeRaw)
	case d.biasIdx:
		return d.biasIdx >= 0 && d.biasCacheKind != supportCacheNone &&
			shouldCacheByPolicy(d.cachePolicy, d.biasRawBytes, d.biasCachedBytes, willFreeRaw)
	default:
		return false
	}
}

func (d *nodeFusedQuantizedDense) BindConstantData(inputIdx int, cd *gobackend.ConstantData) {
	if d == nil || cd == nil {
		return
	}
	switch inputIdx {
	case 1:
		switch d.weightCacheKind {
		case weightCacheGGMLPanel, weightCacheLinearPanel:
			d.wConstantData = cd
		case weightCacheFloatMatmul:
			lhsCrossSize := 0
			if !d.info.OutputShape.IsDynamic() && d.info.N > 0 {
				lhsCrossSize = d.info.OutputShape.Size() / d.info.N
			}
			d.floatDotNodeData = &dot.NodeData{
				InputDType:      dtypes.Float32,
				OutputDType:     dtypes.Float32,
				Layout:          d.floatLayout,
				BatchSize:       1,
				LHSCrossSize:    lhsCrossSize,
				RHSCrossSize:    d.info.N,
				ContractingSize: d.info.K,
				RHSConstantData: cd,
			}
		}
	case d.wScaleIdx:
		if d.wScaleIdx >= 0 {
			d.wScaleConstantData = cd
			d.wScaleBaked = d.weightIsCached()
		}
	case d.wZPIdx:
		if d.wZPIdx >= 0 {
			d.wZPConstantData = cd
			d.wZPBaked = d.weightIsCached()
		}
	case d.xScaleIdx:
		if d.xScaleIdx >= 0 {
			d.xScaleConstantData = cd
		}
	case d.xZPIdx:
		if d.xZPIdx >= 0 {
			d.xZPConstantData = cd
		}
	case d.outScaleIdx:
		if d.outScaleIdx >= 0 {
			d.outScaleConstantData = cd
		}
	case d.outZPIdx:
		if d.outZPIdx >= 0 {
			d.outZPConstantData = cd
		}
	case d.biasIdx:
		if d.biasIdx >= 0 {
			d.biasConstantData = cd
		}
	}
}

func (d *nodeFusedQuantizedDense) IsInputCached(inputIdx int) bool {
	if d == nil {
		return false
	}
	switch inputIdx {
	case 1:
		return d.weightCachedFormReady()
	case d.wScaleIdx:
		if d.wScaleIdx < 0 {
			return false
		}
		if d.wScaleBaked {
			return d.weightCachedFormReady()
		}
		return d.wScaleConstantData.CachedForm() != nil
	case d.wZPIdx:
		if d.wZPIdx < 0 {
			return false
		}
		if d.wZPBaked {
			return d.weightCachedFormReady()
		}
		return d.wZPConstantData.CachedForm() != nil
	case d.xScaleIdx:
		return d.xScaleIdx >= 0 && d.xScaleConstantData.CachedForm() != nil
	case d.xZPIdx:
		return d.xZPIdx >= 0 && d.xZPConstantData.CachedForm() != nil
	case d.outScaleIdx:
		return d.outScaleIdx >= 0 && d.outScaleConstantData.CachedForm() != nil
	case d.outZPIdx:
		return d.outZPIdx >= 0 && d.outZPConstantData.CachedForm() != nil
	case d.biasIdx:
		return d.biasIdx >= 0 && d.biasConstantData.CachedForm() != nil
	default:
		return false
	}
}

func (d *nodeFusedQuantizedDense) Recompute(_ *gobackend.Backend, _ []*gobackend.Node, _ *gobackend.Node) (any, error) {
	return d, nil
}

func (d *nodeFusedQuantizedDense) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeFusedQuantizedDense)
	return d.weightScheme == o.weightScheme &&
		d.weightGGMLType == o.weightGGMLType &&
		d.weightLayout == o.weightLayout &&
		d.hasXQuant == o.hasXQuant &&
		d.xScheme == o.xScheme &&
		d.xGGMLType == o.xGGMLType &&
		d.hasOutputQuant == o.hasOutputQuant &&
		d.outputQuantDType == o.outputQuantDType &&
		d.quantizedMatmul == o.quantizedMatmul &&
		d.activation == o.activation &&
		d.hasWScale == o.hasWScale &&
		d.hasWZP == o.hasWZP &&
		d.hasXScale == o.hasXScale &&
		d.hasXZP == o.hasXZP &&
		d.hasOutScale == o.hasOutScale &&
		d.hasOutZP == o.hasOutZP &&
		d.hasBias == o.hasBias &&
		d.info.K == o.info.K &&
		d.info.N == o.info.N &&
		d.info.FloatComputeDType == o.info.FloatComputeDType &&
		d.info.WeightScaleLayout == o.info.WeightScaleLayout &&
		d.info.WeightZeroPointLayout == o.info.WeightZeroPointLayout &&
		d.info.XScaleLayout == o.info.XScaleLayout &&
		d.info.XZeroPointLayout == o.info.XZeroPointLayout &&
		d.info.OutputScaleLayout == o.info.OutputScaleLayout &&
		d.info.OutputZeroPointLayout == o.info.OutputZeroPointLayout &&
		d.cachePolicy == o.cachePolicy &&
		d.weightCacheKind == o.weightCacheKind &&
		d.weightCacheSig == o.weightCacheSig
}

// TryFuseActivation implements gobackend.FusedDenseActivationTarget so single-consumer
// FusedQuantizedDense(ActivationNone) -> FusedActivation(cfg) pairs fuse automatically.
func (d *nodeFusedQuantizedDense) TryFuseActivation(cfg compute.ActivationConfig) bool {
	if d == nil || d.hasOutputQuant || d.activation.Type != compute.ActivationNone {
		return false
	}
	if cfg.Type == compute.ActivationNone || cfg.Type == compute.ActivationSwiGLU {
		return false
	}
	d.activation = cfg
	return true
}

// quantizationToShapes converts a *compute.Quantization into a *shapeinference.QuantizationShapes,
// verifying and casting any non-nil Scale / ZeroPoint compute.Value nodes along the way.
func extractQuantizationShapes(
	f *gobackend.Function,
	opName string,
	q *compute.Quantization,
) (qs *shapeinference.QuantizationShapes, scaleNode, zpNode *gobackend.Node, err error) {
	if q == nil {
		return nil, nil, nil, nil
	}
	qs = &shapeinference.QuantizationShapes{
		Scheme:         q.Scheme,
		BlockAxis:      q.BlockAxis,
		BlockSize:      q.BlockSize,
		GGMLType:       q.GGMLType,
		QuantizedDType: q.QuantizedDType,
		PackedDType:    q.PackedDType,
	}
	if q.Scale != nil {
		nodes, err := f.VerifyAndCastValues(opName, q.Scale)
		if err != nil {
			return nil, nil, nil, err
		}
		scaleNode = nodes[0]
		qs.Scale = scaleNode.Shape
	}
	if q.ZeroPoint != nil {
		nodes, err := f.VerifyAndCastValues(opName, q.ZeroPoint)
		if err != nil {
			return nil, nil, nil, err
		}
		zpNode = nodes[0]
		qs.ZeroPoint = zpNode.Shape
	}
	return qs, scaleNode, zpNode, nil
}

// FusedQuantizedDense builds an OpTypeFusedQuantizedDense node in the Go backend graph.
func FusedQuantizedDense(
	f *gobackend.Function,
	x, weight, bias compute.Value,
	options compute.FusedQuantizedDenseConfig,
) (compute.Value, error) {
	if options.WeightQuantization == nil {
		return nil, errors.Errorf("FusedQuantizedDense: options.WeightQuantization must not be nil")
	}

	baseInputs, err := f.VerifyAndCastValues("FusedQuantizedDense", x, weight)
	if err != nil {
		return nil, err
	}
	xNode, wNode := baseInputs[0], baseInputs[1]

	wqs, wScaleNode, wZPNode, err := extractQuantizationShapes(f, "FusedQuantizedDense", options.WeightQuantization)
	if err != nil {
		return nil, err
	}
	xqs, xScaleNode, xZPNode, err := extractQuantizationShapes(f, "FusedQuantizedDense", options.XQuantization)
	if err != nil {
		return nil, err
	}
	oqs, oScaleNode, oZPNode, err := extractQuantizationShapes(f, "FusedQuantizedDense", options.OutputQuantization)
	if err != nil {
		return nil, err
	}

	var biasNode *gobackend.Node
	var biasShape shapes.Shape
	if bias != nil {
		bNodes, err := f.VerifyAndCastValues("FusedQuantizedDense", bias)
		if err != nil {
			return nil, err
		}
		biasNode = bNodes[0]
		biasShape = biasNode.Shape
	}

	cfgShapes := shapeinference.QuantizedDenseConfigShapes{
		WeightQuantization: wqs,
		XQuantization:      xqs,
		OutputQuantization: oqs,
		QuantizedMatmul:    options.QuantizedMatmul,
		Activation:         options.Activation,
		WeightLayout:       options.WeightLayout,
	}
	info, err := shapeinference.FusedQuantizedDenseWithInfo(xNode.Shape, wNode.Shape, biasShape, cfgShapes)
	if err != nil {
		return nil, err
	}

	// Extra GGML implementation check so unsupported enum values fail at build time.
	if wqs.Scheme == compute.QuantGGML {
		if _, err := ggmlDequantFunc(wqs.GGMLType); err != nil {
			return nil, err
		}
	}

	inputs := make([]*gobackend.Node, 0, 9)
	inputs = append(inputs, xNode, wNode)
	wScaleIdx, wZPIdx, xScaleIdx, xZPIdx, outScaleIdx, outZPIdx, biasIdx := -1, -1, -1, -1, -1, -1, -1
	if wScaleNode != nil {
		wScaleIdx = len(inputs)
		inputs = append(inputs, wScaleNode)
	}
	if wZPNode != nil {
		wZPIdx = len(inputs)
		inputs = append(inputs, wZPNode)
	}
	if xScaleNode != nil {
		xScaleIdx = len(inputs)
		inputs = append(inputs, xScaleNode)
	}
	if xZPNode != nil {
		xZPIdx = len(inputs)
		inputs = append(inputs, xZPNode)
	}
	if oScaleNode != nil {
		outScaleIdx = len(inputs)
		inputs = append(inputs, oScaleNode)
	}
	if oZPNode != nil {
		outZPIdx = len(inputs)
		inputs = append(inputs, oZPNode)
	}
	if biasNode != nil {
		biasIdx = len(inputs)
		inputs = append(inputs, biasNode)
	}

	cachePolicy := gobackend.QuantizedCacheBalanced
	if f.RawBuilder != nil && f.RawBuilder.Backend != nil {
		cachePolicy = f.RawBuilder.Backend.QuantizedCache
	}

	data := &nodeFusedQuantizedDense{
		weightScheme:     wqs.Scheme,
		weightGGMLType:   wqs.GGMLType,
		weightDType:      wNode.Shape.DType,
		weightLayout:     options.WeightLayout,
		hasXQuant:        xqs != nil,
		hasOutputQuant:   oqs != nil,
		outputQuantDType: info.OutputShape.DType,
		quantizedMatmul:  options.QuantizedMatmul,
		activation:       options.Activation,
		hasWScale:        wScaleNode != nil,
		hasWZP:           wZPNode != nil,
		hasXScale:        xScaleNode != nil,
		hasXZP:           xZPNode != nil,
		hasOutScale:      oScaleNode != nil,
		hasOutZP:         oZPNode != nil,
		hasBias:          biasNode != nil,
		wScaleIdx:        wScaleIdx,
		wZPIdx:           wZPIdx,
		xScaleIdx:        xScaleIdx,
		xZPIdx:           xZPIdx,
		outScaleIdx:      outScaleIdx,
		outZPIdx:         outZPIdx,
		biasIdx:          biasIdx,
		info:             info,
		cachePolicy:      cachePolicy,
	}
	if xqs != nil {
		data.xScheme = xqs.Scheme
		data.xGGMLType = xqs.GGMLType
	}
	if wZPNode != nil {
		data.wZPDType = wZPNode.Shape.DType
	}
	configureQuantizedDenseCaching(data, wNode, wScaleNode, wZPNode, xScaleNode, xZPNode, oScaleNode, oZPNode, biasNode)

	node, _ := f.GetOrCreateNode(compute.OpTypeFusedQuantizedDense, info.OutputShape, inputs, data)
	return node, nil
}

func configureSupportFloat32Cache(node *gobackend.Node) (supportCacheKind, int64, int64) {
	if node == nil || !node.IsConstant() || node.Shape.IsDynamic() || node.Shape.Size() == 0 {
		return supportCacheNone, 0, 0
	}
	dt := node.Shape.DType
	if dt == dtypes.Float16 || dt == dtypes.BFloat16 {
		return supportCacheF32, node.Shape.ByteSize(), int64(node.Shape.Size()) * 4
	}
	return supportCacheNone, 0, 0
}

func configureSupportZeroPointCache(node *gobackend.Node, wantInt32 bool) (supportCacheKind, int64, int64) {
	if node == nil || !node.IsConstant() || node.Shape.IsDynamic() || node.Shape.Size() == 0 {
		return supportCacheNone, 0, 0
	}
	dt := node.Shape.DType
	if wantInt32 {
		if dt == dtypes.Int32 {
			return supportCacheNone, 0, 0
		}
		return supportCacheI32, node.Shape.ByteSize(), int64(node.Shape.Size()) * 4
	}
	if dt == dtypes.Float32 {
		return supportCacheNone, 0, 0
	}
	return supportCacheF32, node.Shape.ByteSize(), int64(node.Shape.Size()) * 4
}

func configureQuantizedDenseCaching(
	data *nodeFusedQuantizedDense,
	wNode, wScaleNode, wZPNode, xScaleNode, xZPNode, oScaleNode, oZPNode, biasNode *gobackend.Node,
) {
	if data.cachePolicy == gobackend.QuantizedCacheNever {
		return
	}

	// Supporting constant tensors (scales, zero-points, bias) that require dtype conversion.
	data.wScaleCacheKind, data.wScaleRawBytes, data.wScaleCachedBytes = configureSupportFloat32Cache(wScaleNode)
	if wScaleNode != nil && !wScaleNode.Shape.IsDynamic() {
		data.wScaleRawBytes = wScaleNode.Shape.ByteSize()
	}
	wZPWantI32 := data.weightScheme == compute.QuantLinear &&
		(data.hasXQuant || (data.quantizedMatmul && !canCacheQuantLinearPanel(data) && !data.wZPDType.IsFloat()))
	data.wZPCacheKind, data.wZPRawBytes, data.wZPCachedBytes = configureSupportZeroPointCache(wZPNode, wZPWantI32)
	if wZPNode != nil && !wZPNode.Shape.IsDynamic() {
		data.wZPRawBytes = wZPNode.Shape.ByteSize()
	}
	data.xScaleCacheKind, data.xScaleRawBytes, data.xScaleCachedBytes = configureSupportFloat32Cache(xScaleNode)
	data.xZPCacheKind, data.xZPRawBytes, data.xZPCachedBytes = configureSupportZeroPointCache(xZPNode, true)
	data.outScaleCacheKind, data.outScaleRawBytes, data.outScaleCachedB = configureSupportFloat32Cache(oScaleNode)
	data.outZPCacheKind, data.outZPRawBytes, data.outZPCachedBytes = configureSupportZeroPointCache(oZPNode, false)
	if biasNode != nil && !(data.weightScheme == compute.QuantLinear && data.hasXQuant && biasNode.Shape.DType == dtypes.Int32) {
		data.biasCacheKind, data.biasRawBytes, data.biasCachedBytes = configureSupportFloat32Cache(biasNode)
	}

	// Primary weight caching candidate.
	if wNode == nil || !wNode.IsConstant() || wNode.Shape.IsDynamic() || data.info.K <= 0 || data.info.N <= 0 {
		return
	}
	data.weightRawBytes = wNode.Shape.ByteSize()

	// Helper to build pointer-stable identity signatures for constant wScale / wZP.
	scaleAndZPSigs := func() (string, string, bool) {
		var sSig, zpSig string
		if wScaleNode != nil {
			var ok bool
			sSig, ok = gobackend.ConstantIdentitySignature(wScaleNode)
			if !ok {
				return "", "", false
			}
		}
		if wZPNode != nil {
			var ok bool
			zpSig, ok = gobackend.ConstantIdentitySignature(wZPNode)
			if !ok {
				return "", "", false
			}
		}
		return sSig, zpSig, true
	}

	if data.weightScheme == compute.QuantGGML && (data.quantizedMatmul || data.hasXQuant) &&
		canCacheGGMLPanel(data.weightGGMLType, data.info.K, data.info.N) {
		subBlockSize, _, _ := ggmlPanelSpec(data.weightGGMLType)
		numSubBlocks := data.info.K / subBlockSize
		subBlkBytes := QuantizedPanelStripSubBlockBytes(subBlockSize / 4)
		totalStrips := (data.info.N + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
		data.weightCacheKind = weightCacheGGMLPanel
		data.weightCachedBytes = int64(totalStrips) * int64(numSubBlocks) * int64(subBlkBytes)
		data.weightCacheSig = fmt.Sprintf("QDGGMLPanel(type=%s,K=%d,N=%d)", data.weightGGMLType, data.info.K, data.info.N)
		return
	}

	if kBlock, ok := linearPanelKBlock(data); ok {
		sSig, zpSig, constOK := scaleAndZPSigs()
		if constOK {
			numSubBlocks := (data.info.K + kBlock - 1) / kBlock
			bPad := (kBlock + 3) &^ 3
			subBlkBytes := QuantizedPanelStripSubBlockBytes(bPad / 4)
			totalStrips := (data.info.N + QuantizedPanelStripCols - 1) / QuantizedPanelStripCols
			data.weightCacheKind = weightCacheLinearPanel
			data.weightCachedBytes = int64(totalStrips) * int64(numSubBlocks) * int64(subBlkBytes)
			data.weightCacheSig = fmt.Sprintf(
				"QDLinearPanel(dt=%s,layout=%v,K=%d,N=%d,kBlock=%d,sL=%+v,zpL=%+v,s=%s,zp=%s)",
				wNode.Shape.DType, data.weightLayout, data.info.K, data.info.N, kBlock,
				data.info.WeightScaleLayout, data.info.WeightZeroPointLayout, sSig, zpSig,
			)
		}
		return
	}

	// Mode A: weight-only float32 matmul (QuantGGML, QuantLinear, or QuantNF4).
	if !data.quantizedMatmul && !data.hasXQuant {
		sSig, zpSig, constOK := scaleAndZPSigs()
		if !constOK {
			return
		}
		layout := dot.LayoutTransposed
		if data.weightLayout == compute.DenseLayoutInputOutputs && data.weightScheme != compute.QuantGGML {
			layout = dot.LayoutNonTransposed
		}
		data.weightCacheKind = weightCacheFloatMatmul
		data.floatLayout = layout
		data.weightCachedBytes = int64(data.info.N) * int64(data.info.K) * 4
		if data.weightScheme == compute.QuantGGML {
			data.weightCacheSig = fmt.Sprintf("QDGGMLFloatMatmul(type=%s,layout=%v,K=%d,N=%d)",
				data.weightGGMLType, layout, data.info.K, data.info.N)
		} else {
			data.weightCacheSig = fmt.Sprintf(
				"QDLinearFloatMatmul(scheme=%v,dt=%s,layout=%v,K=%d,N=%d,sL=%+v,zpL=%+v,s=%s,zp=%s)",
				data.weightScheme, wNode.Shape.DType, layout, data.info.K, data.info.N,
				data.info.WeightScaleLayout, data.info.WeightZeroPointLayout, sSig, zpSig,
			)
		}
	}
}

type unpackedQDInputs struct {
	x        *gobackend.Buffer
	weight   *gobackend.Buffer
	wScale   *gobackend.Buffer
	wZP      *gobackend.Buffer
	xScale   *gobackend.Buffer
	xZP      *gobackend.Buffer
	outScale *gobackend.Buffer
	outZP    *gobackend.Buffer
	bias     *gobackend.Buffer
}

func unpackQDInputs(data *nodeFusedQuantizedDense, inputs []*gobackend.Buffer) unpackedQDInputs {
	var res unpackedQDInputs
	res.x = inputs[0]
	res.weight = inputs[1]
	idx := 2
	if data.hasWScale {
		res.wScale = inputs[idx]
		idx++
	}
	if data.hasWZP {
		res.wZP = inputs[idx]
		idx++
	}
	if data.hasXScale {
		res.xScale = inputs[idx]
		idx++
	}
	if data.hasXZP {
		res.xZP = inputs[idx]
		idx++
	}
	if data.hasOutScale {
		res.outScale = inputs[idx]
		idx++
	}
	if data.hasOutZP {
		res.outZP = inputs[idx]
		idx++
	}
	if data.hasBias {
		res.bias = inputs[idx]
	}
	return res
}

// execFusedQuantizedDense dispatches FusedQuantizedDense across weight-only, pre-quantized,
// and on-the-fly quantized matmul kernels, followed by activation and optional output quantization.
func execFusedQuantizedDense(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, _ []bool) (*gobackend.Buffer, error) {
	data := node.Data.(*nodeFusedQuantizedDense)
	in := unpackQDInputs(data, inputs)

	output, err := backend.GetBuffer(node.Shape)
	if err != nil {
		return nil, err
	}
	if backend.NoOps || node.Shape.Size() == 0 {
		return output, nil
	}

	N := data.info.N
	K := data.info.K
	M := node.Shape.Size() / N

	// Allocate or alias Float32 accumulation buffer [M, N].
	var outF32 []float32
	var tmpOutBuf *gobackend.Buffer
	if !data.hasOutputQuant && output.RawShape.DType == dtypes.Float32 {
		outF32 = output.Flat.([]float32)
	} else {
		tmpOutBuf, err = backend.GetBuffer(shapes.Make(dtypes.Float32, M, N))
		if err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		defer backend.PutBuffer(tmpOutBuf)
		outF32 = tmpOutBuf.Flat.([]float32)
	}

	var inTileAct activations.KernelFn[float32]
	if data.activation.Type != compute.ActivationNone {
		inTileAct, _ = activations.GetKernelWithMetadata[float32](data.activation.Type)
	}
	var appliedAct bool

	switch data.weightScheme {
	case compute.QuantGGML:
		appliedAct, err = execQuantizedDenseGGMLCore(backend, data, in, M, K, N, outF32, inTileAct)
	case compute.QuantLinear, compute.QuantNF4:
		appliedAct, err = execQuantizedDenseLinearOrNF4Core(backend, data, in, M, K, N, outF32, inTileAct)
	default:
		err = errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDense: unsupported scheme %s", data.weightScheme)
	}
	if err != nil {
		backend.PutBuffer(output)
		return nil, err
	}

	// Apply post-matmul+bias activation in Float32 if not already fused in-tile.
	if !appliedAct {
		applyActivationConfigFloat32(backend, data.activation, outF32)
	}

	// Optional OutputQuantization (e.g. ONNX QLinearMatMul).
	if data.hasOutputQuant {
		if err := quantizeOutputFloat32(backend, outF32, M, N, in.outScale, data.outScaleConstantData, in.outZP, data.outZPConstantData, data.info.OutputScaleLayout, data.info.OutputZeroPointLayout, output); err != nil {
			backend.PutBuffer(output)
			return nil, err
		}
		return output, nil
	}

	// Convert Float32 -> Float16 / BFloat16 if needed.
	switch output.RawShape.DType {
	case dtypes.Float32:
		// Already written directly into output.Flat.
	case dtypes.Float16:
		convertFloat32SliceToFloat16(backend, outF32, output.Flat.([]float16.Float16))
	case dtypes.BFloat16:
		convertFloat32SliceToBFloat16(backend, outF32, output.Flat.([]bfloat16.BFloat16))
	default:
		backend.PutBuffer(output)
		return nil, errors.Wrapf(compute.ErrNotImplemented, "FusedQuantizedDense: unsupported output DType %s", output.RawShape.DType)
	}
	return output, nil
}

func convertFloat32SliceToFloat16(backend *gobackend.Backend, src []float32, dst []float16.Float16) {
	n := len(src)
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && n >= 32768 {
		backend.Scheduler.ParallelizeChunks(n, 16384, func(start, end int) {
			s := src[start:end]
			d := dst[start:end]
			for i, v := range s {
				d[i] = float16.FromFloat32(v)
			}
		})
		return
	}
	for i, v := range src {
		dst[i] = float16.FromFloat32(v)
	}
}

func convertFloat32SliceToBFloat16(backend *gobackend.Backend, src []float32, dst []bfloat16.BFloat16) {
	n := len(src)
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && n >= 32768 {
		backend.Scheduler.ParallelizeChunks(n, 16384, func(start, end int) {
			s := src[start:end]
			d := dst[start:end]
			for i, v := range s {
				d[i] = bfloat16.FromFloat32(v)
			}
		})
		return
	}
	for i, v := range src {
		dst[i] = bfloat16.FromFloat32(v)
	}
}

func applyActivationConfigFloat32(backend *gobackend.Backend, cfg compute.ActivationConfig, data []float32) {
	if cfg.Type == compute.ActivationNone || len(data) == 0 {
		return
	}
	activations.Apply(backend, cfg.Type, data)
}

func applyActivationVJPConfigFloat32(backend *gobackend.Backend, cfg compute.ActivationConfig, y, dOut, dz []float32) error {
	if len(dOut) == 0 {
		return nil
	}
	return activations.ExecuteVJPFromOutput(backend, cfg.Type, y, dOut, dz)
}

func quantizeOutputFloat32(
	backend *gobackend.Backend,
	yF32 []float32,
	M, N int,
	scaleBuf *gobackend.Buffer,
	scaleCD *gobackend.ConstantData,
	zpBuf *gobackend.Buffer,
	zpCD *gobackend.ConstantData,
	scaleLayout, zpLayout shapeinference.ActivationParamLayout,
	outBuf *gobackend.Buffer,
) error {
	scales, releaseScale, err := resolveCachedFloat32Slice(backend, scaleBuf, scaleCD)
	if err != nil {
		return err
	}
	defer releaseScale()

	var zps []float32
	if zpBuf != nil || zpCD != nil {
		var releaseZP func()
		zps, releaseZP, err = resolveCachedZeroPointFloat32Slice(backend, zpBuf, zpCD)
		if err != nil {
			return err
		}
		defer releaseZP()
	}

	switch outBuf.RawShape.DType {
	case dtypes.Int8:
		dst := outBuf.Flat.([]int8)
		for m := range M {
			rowOff := m * N
			for n := range N {
				sc := scales[scaleLayout.Index(m, n)]
				var zp float32
				if zps != nil {
					zp = zps[zpLayout.Index(m, n)]
				}
				q := float32(math.RoundToEven(float64(yF32[rowOff+n]/sc))) + zp
				q = max(float32(-128), min(float32(127), q))
				dst[rowOff+n] = int8(int32(math.RoundToEven(float64(q))))
			}
		}
	case dtypes.Uint8:
		dst := outBuf.Flat.([]uint8)
		for m := range M {
			rowOff := m * N
			for n := range N {
				sc := scales[scaleLayout.Index(m, n)]
				var zp float32
				if zps != nil {
					zp = zps[zpLayout.Index(m, n)]
				}
				q := float32(math.RoundToEven(float64(yF32[rowOff+n]/sc))) + zp
				q = max(float32(0), min(float32(255), q))
				dst[rowOff+n] = uint8(int32(math.RoundToEven(float64(q))))
			}
		}
	default:
		return errors.Errorf("FusedQuantizedDense: unsupported OutputQuantization DType %s", outBuf.RawShape.DType)
	}
	return nil
}

// resolveCachedFloat32Slice returns a []float32 view from cd.CachedForm() when bound and
// cached, populating cd with an exact-sized non-pooled converted []float32 slice on cold start,
// or falling back to bufferToFloat32Slice when cd is nil.
func resolveCachedFloat32Slice(backend *gobackend.Backend, buf *gobackend.Buffer, cd *gobackend.ConstantData) ([]float32, func(), error) {
	if cd == nil {
		return bufferToFloat32Slice(backend, buf)
	}
	if cached := cd.CachedForm(); cached != nil {
		return cached.([]float32), func() {}, nil
	}
	var convErr error
	res := gobackend.GetOrSetCachedForm(cd, func() []float32 {
		s, release, err := bufferToFloat32Slice(backend, buf)
		if err != nil {
			convErr = err
			return nil
		}
		defer release()
		cp := make([]float32, len(s))
		copy(cp, s)
		return cp
	})
	if convErr != nil {
		return nil, nil, convErr
	}
	return res, func() {}, nil
}

// resolveCachedZeroPointFloat32Slice returns a cached []float32 zero-point slice via cd when bound,
// or falls back to bufferToZeroPointFloat32Slice when cd is nil.
func resolveCachedZeroPointFloat32Slice(backend *gobackend.Backend, buf *gobackend.Buffer, cd *gobackend.ConstantData) ([]float32, func(), error) {
	if cd == nil {
		return bufferToZeroPointFloat32Slice(backend, buf)
	}
	if cached := cd.CachedForm(); cached != nil {
		return cached.([]float32), func() {}, nil
	}
	var convErr error
	res := gobackend.GetOrSetCachedForm(cd, func() []float32 {
		s, release, err := bufferToZeroPointFloat32Slice(backend, buf)
		if err != nil {
			convErr = err
			return nil
		}
		defer release()
		cp := make([]float32, len(s))
		copy(cp, s)
		return cp
	})
	if convErr != nil {
		return nil, nil, convErr
	}
	return res, func() {}, nil
}

// resolveCachedZeroPointInt32Slice returns a cached []int32 zero-point slice via cd when bound,
// or falls back to bufferToZeroPointInt32Slice when cd is nil.
func resolveCachedZeroPointInt32Slice(backend *gobackend.Backend, buf *gobackend.Buffer, cd *gobackend.ConstantData) ([]int32, func(), error) {
	if cd == nil {
		return bufferToZeroPointInt32Slice(backend, buf)
	}
	if cached := cd.CachedForm(); cached != nil {
		return cached.([]int32), func() {}, nil
	}
	var convErr error
	res := gobackend.GetOrSetCachedForm(cd, func() []int32 {
		s, release, err := bufferToZeroPointInt32Slice(backend, buf)
		if err != nil {
			convErr = err
			return nil
		}
		defer release()
		cp := make([]int32, len(s))
		copy(cp, s)
		return cp
	})
	if convErr != nil {
		return nil, nil, convErr
	}
	return res, func() {}, nil
}

// bufferToFloat32Slice returns a []float32 view (zero-copy when already Float32) or a pooled converted buffer.
func bufferToFloat32Slice(backend *gobackend.Backend, buf *gobackend.Buffer) ([]float32, func(), error) {
	if buf == nil {
		return nil, func() {}, nil
	}
	switch buf.RawShape.DType {
	case dtypes.Float32:
		return buf.Flat.([]float32), func() {}, nil
	case dtypes.Float16:
		n := buf.RawShape.Size()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, n))
		if err != nil {
			return nil, nil, err
		}
		src := buf.Flat.([]float16.Float16)[:n]
		dst := tmp.Flat.([]float32)[:n]
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && n >= 32768 {
			backend.Scheduler.ParallelizeChunks(n, 16384, func(start, end int) {
				s := src[start:end]
				d := dst[start:end]
				for i, v := range s {
					d[i] = v.Float32()
				}
			})
		} else {
			for i, v := range src {
				dst[i] = v.Float32()
			}
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	case dtypes.BFloat16:
		n := buf.RawShape.Size()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, n))
		if err != nil {
			return nil, nil, err
		}
		src := buf.Flat.([]bfloat16.BFloat16)[:n]
		dst := tmp.Flat.([]float32)[:n]
		if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && n >= 32768 {
			backend.Scheduler.ParallelizeChunks(n, 16384, func(start, end int) {
				s := src[start:end]
				d := dst[start:end]
				for i, v := range s {
					d[i] = v.Float32()
				}
			})
		} else {
			for i, v := range src {
				dst[i] = v.Float32()
			}
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	default:
		return nil, nil, errors.Errorf("expected float buffer (Float32/Float16/BFloat16), got %s", buf.RawShape.DType)
	}
}

// isSubByteQuantDType reports whether dt is a packed sub-byte integer type (Int2/Uint2/Int4/Uint4).
func isSubByteQuantDType(dt dtypes.DType) bool {
	switch dt {
	case dtypes.Int2, dtypes.Uint2, dtypes.Int4, dtypes.Uint4:
		return true
	default:
		return false
	}
}

// bufferToZeroPointFloat32Slice converts an integer or float ZeroPoint buffer into a []float32 slice.
func bufferToZeroPointFloat32Slice(backend *gobackend.Backend, buf *gobackend.Buffer) ([]float32, func(), error) {
	if buf == nil {
		return nil, func() {}, nil
	}
	if buf.RawShape.DType == dtypes.Float32 {
		return buf.Flat.([]float32), func() {}, nil
	}
	if buf.RawShape.DType == dtypes.Float16 || buf.RawShape.DType == dtypes.BFloat16 {
		return bufferToFloat32Slice(backend, buf)
	}
	if isSubByteQuantDType(buf.RawShape.DType) {
		n := buf.RawShape.Size()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, n))
		if err != nil {
			return nil, nil, err
		}
		tmpI8 := make([]int8, n)
		unpackSubByteSpanInt8(buf.RawShape.DType, buf.Flat.([]byte), 0, n, tmpI8)
		dst := tmp.Flat.([]float32)
		for i, v := range tmpI8 {
			dst[i] = float32(v)
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	}

	tmp, err := backend.GetBuffer(shapes.Make(dtypes.Float32, buf.RawShape.Dimensions...))
	if err != nil {
		return nil, nil, err
	}
	convertFnAny, err := ops.ConvertDTypePairMap.Get(buf.RawShape.DType, dtypes.Float32)
	if err != nil {
		backend.PutBuffer(tmp)
		return nil, nil, err
	}
	convertFnAny.(ops.ConvertFnType)(buf, tmp)
	return tmp.Flat.([]float32), func() { backend.PutBuffer(tmp) }, nil
}

// bufferToZeroPointInt32Slice converts a ZeroPoint buffer into a []int32 slice for integer matmul kernels.
func bufferToZeroPointInt32Slice(backend *gobackend.Backend, buf *gobackend.Buffer) ([]int32, func(), error) {
	if buf == nil {
		return nil, func() {}, nil
	}
	if buf.RawShape.DType == dtypes.Int32 {
		return buf.Flat.([]int32), func() {}, nil
	}
	if buf.RawShape.DType.IsFloat() {
		f32, releaseF32, err := bufferToFloat32Slice(backend, buf)
		if err != nil {
			return nil, nil, err
		}
		defer releaseF32()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Int32, len(f32)))
		if err != nil {
			return nil, nil, err
		}
		dst := tmp.Flat.([]int32)
		for i, v := range f32 {
			dst[i] = int32(math.RoundToEven(float64(v)))
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	}
	if isSubByteQuantDType(buf.RawShape.DType) {
		n := buf.RawShape.Size()
		tmp, err := backend.GetBuffer(shapes.Make(dtypes.Int32, n))
		if err != nil {
			return nil, nil, err
		}
		tmpI8 := make([]int8, n)
		unpackSubByteSpanInt8(buf.RawShape.DType, buf.Flat.([]byte), 0, n, tmpI8)
		dst := tmp.Flat.([]int32)
		for i, v := range tmpI8 {
			dst[i] = int32(v)
		}
		return dst, func() { backend.PutBuffer(tmp) }, nil
	}

	tmp, err := backend.GetBuffer(shapes.Make(dtypes.Int32, buf.RawShape.Dimensions...))
	if err != nil {
		return nil, nil, err
	}
	convertFnAny, err := ops.ConvertDTypePairMap.Get(buf.RawShape.DType, dtypes.Int32)
	if err != nil {
		backend.PutBuffer(tmp)
		return nil, nil, err
	}
	convertFnAny.(ops.ConvertFnType)(buf, tmp)
	return tmp.Flat.([]int32), func() { backend.PutBuffer(tmp) }, nil
}

// unpackSubByteSpanInt8 unpacks `count` sub-byte elements starting at logical element index `startElem`
// from `packed` into `dst` (as signed int8 for Int4/Int2, or 0..15 / 0..3 int8 for Uint4/Uint2, which all fit in int8!).
func unpackSubByteSpanInt8(dt dtypes.DType, packed []byte, startElem, count int, dst []int8) {
	switch dt {
	case dtypes.Int4:
		if startElem&1 == 0 && count&1 == 0 {
			byteOff := startElem >> 1
			half := count >> 1
			src := packed[byteOff : byteOff+half]
			for i, b := range src {
				lo := int8(b & 0x0F)
				if lo >= 8 {
					lo -= 16
				}
				hi := int8(b) >> 4
				dst[2*i] = lo
				dst[2*i+1] = hi
			}
			return
		}
		for i := range count {
			idx := startElem + i
			nib := int8((packed[idx>>1] >> uint((idx&1)*4)) & 0x0F)
			if nib >= 8 {
				nib -= 16
			}
			dst[i] = nib
		}

	case dtypes.Uint4:
		if startElem&1 == 0 && count&1 == 0 {
			byteOff := startElem >> 1
			half := count >> 1
			src := packed[byteOff : byteOff+half]
			for i, b := range src {
				dst[2*i] = int8(b & 0x0F)
				dst[2*i+1] = int8(b >> 4)
			}
			return
		}
		for i := range count {
			idx := startElem + i
			dst[i] = int8((packed[idx>>1] >> uint((idx&1)*4)) & 0x0F)
		}

	case dtypes.Int2:
		for i := range count {
			idx := startElem + i
			v := int8((packed[idx>>2] >> uint((idx&3)*2)) & 0x03)
			if v >= 2 {
				v -= 4
			}
			dst[i] = v
		}

	case dtypes.Uint2:
		for i := range count {
			idx := startElem + i
			dst[i] = int8((packed[idx>>2] >> uint((idx&3)*2)) & 0x03)
		}
	}
}

func convertIndicesToInt64(backend *gobackend.Backend, indicesBuf *gobackend.Buffer) (*gobackend.Buffer, bool, error) {
	if indicesBuf.RawShape.DType == dtypes.Int64 {
		return indicesBuf, false, nil
	}
	outBuf, err := backend.GetBuffer(shapes.Make(dtypes.Int64, indicesBuf.RawShape.Dimensions...))
	if err != nil {
		return nil, false, err
	}
	convertFnAny, err := ops.ConvertDTypePairMap.Get(indicesBuf.RawShape.DType, dtypes.Int64)
	if err != nil {
		backend.PutBuffer(outBuf)
		return nil, false, err
	}
	convertFnAny.(ops.ConvertFnType)(indicesBuf, outBuf)
	return outBuf, true, nil
}

const quantizedDenseMinParallelizeChunk = 4096

func quantizedDenseParallelTileCount(backend *gobackend.Backend, M, K, N int) int {
	totalWork := M * K * N
	if backend == nil || !backend.Scheduler.IsEnabled() || totalWork <= quantizedDenseMinParallelizeChunk {
		return max(M, 1)
	}
	if M > 1 {
		return M
	}
	tileSize := max(quantizedDenseMinParallelizeChunk/max(K, 1), 1)
	return (N + tileSize - 1) / tileSize
}

func quantizedDenseParallel(backend *gobackend.Backend, M, K, N int, rowFn func(workerIdx, m, nStart, nEnd int)) {
	if M == 0 || N == 0 {
		return
	}
	totalWork := M * K * N
	if backend == nil || !backend.Scheduler.IsEnabled() || totalWork <= quantizedDenseMinParallelizeChunk {
		for m := range M {
			rowFn(m, m, 0, N)
		}
		return
	}

	if M > 1 {
		backend.Scheduler.ParallelizeChunks(M, 1, func(mStart, mEnd int) {
			for m := mStart; m < mEnd; m++ {
				rowFn(m, m, 0, N)
			}
		})
	} else {
		tileSize := max(quantizedDenseMinParallelizeChunk/max(K, 1), 1)
		numTiles := (N + tileSize - 1) / tileSize
		tasks := make([]func(), 0, numTiles)
		workerIdx := 0
		for nStart := 0; nStart < N; nStart += tileSize {
			nEnd := min(nStart+tileSize, N)
			idx := workerIdx
			tasks = append(tasks, func() {
				rowFn(idx, 0, nStart, nEnd)
			})
			workerIdx++
		}
		backend.Scheduler.RunIntraOpTasks(tasks)
	}
}

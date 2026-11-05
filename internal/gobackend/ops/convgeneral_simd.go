// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
)

// Conv2DKind classifies a 1D or 2D convolution into a specialized SIMD execution family.
type Conv2DKind int

const (
	// Conv2DUnsupported indicates the convolution axes/grouping cannot use the 1D/2D ChannelsLast SIMD fast paths.
	Conv2DUnsupported Conv2DKind = iota

	// Conv2DStandard is a 1D or 2D ChannelsLast spatial convolution:
	//   input:  [B, H, W, Cin] (or [B, W, Cin] normalized to H=1)
	//   kernel: [kH, kW, Cin, Cout] (IsKernelTransposed=false, forward)
	//        or [kH, kW, Cout, Cin] (IsKernelTransposed=true, VJP wrt input)
	//   output: [B, OutH, OutW, Cout]
	Conv2DStandard

	// Conv2DKernelGrad is a 1D or 2D ChannelsLast VJP wrt kernel convolution:
	//   input (x):  [B, H, W, Cin]  (InputBatch=rank-1 (Cin), InputSpatial={1..}, InputChannels=0 (B))
	//   kernel (v): [B, OutH, OutW, Cout] (KernelSpatial={1..}, KernelInputChannels=0 (B), KernelOutputChannels=rank-1 (Cout))
	//   output:     [kH, kW, Cin, Cout] (OutputSpatial={0..}, OutputBatch=rank-2 (Cin), OutputChannels=rank-1 (Cout))
	Conv2DKernelGrad
)

// Conv2DParams holds normalized 2D parameters for 1D and 2D ChannelsLast convolutions.
// For 1D convolutions, the height dimension is normalized to 1 (InH=1, OutH=1, KernelH=1, StrideH=1, PadTop=0, InDilH=1, KerDilH=1).
type Conv2DParams struct {
	Kind Conv2DKind

	BatchSize   int
	InH, InW    int
	InChannels  int
	OutH, OutW  int
	OutChannels int
	KernelH     int
	KernelW     int

	StrideH, StrideW int
	PadTop, PadLeft  int
	InDilH, InDilW   int
	KerDilH, KerDilW int

	// IsKernelTransposed is true for VJP wrt input where kernel shape is [kH, kW, Cout, Cin]
	// (KernelInputChannels=rank-1, KernelOutputChannels=rank-2).
	IsKernelTransposed bool

	// Is1x1Unit is true for Conv2DStandard when KernelH==1, KernelW==1, StrideH==1, StrideW==1,
	// PadTop==0, PadLeft==0, InDilH==1, InDilW==1, OutH==InH, OutW==InW.
	Is1x1Unit bool
}

// GetConv2DParams returns the precomputed Conv2DParams for a ConvGeneral node.
func GetConv2DParams(node *gobackend.Node) Conv2DParams {
	if c, ok := node.Data.(*convNode); ok {
		return c.conv2D
	}
	return Conv2DParams{Kind: Conv2DUnsupported}
}

func computeConv2DParams(
	inputShape, kernelShape, outputShape shapes.Shape,
	axes compute.ConvolveAxesConfig,
	strides []int,
	paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int,
) Conv2DParams {
	if channelGroupCount != 1 || batchGroupCount != 1 {
		return Conv2DParams{Kind: Conv2DUnsupported}
	}
	rank := inputShape.Rank()
	if (rank != 3 && rank != 4) || kernelShape.Rank() != rank || outputShape.Rank() != rank {
		return Conv2DParams{Kind: Conv2DUnsupported}
	}
	spatialRank := rank - 2
	if len(axes.InputSpatial) != spatialRank || len(axes.KernelSpatial) != spatialRank || len(axes.OutputSpatial) != spatialRank {
		return Conv2DParams{Kind: Conv2DUnsupported}
	}

	sH, sW := 1, 1
	padTop, padLeft := 0, 0
	inDilH, inDilW := 1, 1
	kerDilH, kerDilW := 1, 1

	if spatialRank == 1 {
		if len(strides) >= 1 && strides[0] > 0 {
			sW = strides[0]
		}
		if len(paddings) >= 1 {
			padLeft = paddings[0][0]
		}
		if len(inputDilations) >= 1 && inputDilations[0] > 0 {
			inDilW = inputDilations[0]
		}
		if len(kernelDilations) >= 1 && kernelDilations[0] > 0 {
			kerDilW = kernelDilations[0]
		}
	} else {
		if len(strides) >= 2 {
			if strides[0] > 0 {
				sH = strides[0]
			}
			if strides[1] > 0 {
				sW = strides[1]
			}
		}
		if len(paddings) >= 2 {
			padTop = paddings[0][0]
			padLeft = paddings[1][0]
		}
		if len(inputDilations) >= 2 {
			if inputDilations[0] > 0 {
				inDilH = inputDilations[0]
			}
			if inputDilations[1] > 0 {
				inDilW = inputDilations[1]
			}
		}
		if len(kernelDilations) >= 2 {
			if kernelDilations[0] > 0 {
				kerDilH = kernelDilations[0]
			}
			if kernelDilations[1] > 0 {
				kerDilW = kernelDilations[1]
			}
		}
	}

	// Check Family 1: Conv2DStandard (ChannelsLast Forward and VJP wrt Input)
	isStandardSpatial := false
	if spatialRank == 1 {
		isStandardSpatial = axes.InputSpatial[0] == 1 && axes.OutputSpatial[0] == 1 && axes.KernelSpatial[0] == 0
	} else {
		isStandardSpatial = axes.InputSpatial[0] == 1 && axes.InputSpatial[1] == 2 &&
			axes.OutputSpatial[0] == 1 && axes.OutputSpatial[1] == 2 &&
			axes.KernelSpatial[0] == 0 && axes.KernelSpatial[1] == 1
	}
	if isStandardSpatial &&
		axes.InputBatch == 0 && axes.InputChannels == rank-1 &&
		axes.OutputBatch == 0 && axes.OutputChannels == rank-1 {
		isForward := axes.KernelInputChannels == rank-2 && axes.KernelOutputChannels == rank-1
		isTransposed := axes.KernelInputChannels == rank-1 && axes.KernelOutputChannels == rank-2
		if isForward || isTransposed {
			var inH, inW, outH, outW, kH, kW int
			if spatialRank == 1 {
				inH, inW = 1, inputShape.Dimensions[1]
				outH, outW = 1, outputShape.Dimensions[1]
				kH, kW = 1, kernelShape.Dimensions[0]
			} else {
				inH, inW = inputShape.Dimensions[1], inputShape.Dimensions[2]
				outH, outW = outputShape.Dimensions[1], outputShape.Dimensions[2]
				kH, kW = kernelShape.Dimensions[0], kernelShape.Dimensions[1]
			}
			is1x1Unit := kH == 1 && kW == 1 &&
				sH == 1 && sW == 1 &&
				padTop == 0 && padLeft == 0 &&
				inDilH == 1 && inDilW == 1 &&
				outH == inH && outW == inW
			return Conv2DParams{
				Kind:               Conv2DStandard,
				BatchSize:          inputShape.Dimensions[0],
				InH:                inH,
				InW:                inW,
				InChannels:         inputShape.Dimensions[rank-1],
				OutH:               outH,
				OutW:               outW,
				OutChannels:        outputShape.Dimensions[rank-1],
				KernelH:            kH,
				KernelW:            kW,
				StrideH:            sH,
				StrideW:            sW,
				PadTop:             padTop,
				PadLeft:            padLeft,
				InDilH:             inDilH,
				InDilW:             inDilW,
				KerDilH:            kerDilH,
				KerDilW:            kerDilW,
				IsKernelTransposed: isTransposed,
				Is1x1Unit:          is1x1Unit,
			}
		}
	}

	// Check Family 2: Conv2DKernelGrad (ChannelsLast VJP wrt Kernel)
	isKernelGradSpatial := false
	if spatialRank == 1 {
		isKernelGradSpatial = axes.InputSpatial[0] == 1 && axes.KernelSpatial[0] == 1 && axes.OutputSpatial[0] == 0
	} else {
		isKernelGradSpatial = axes.InputSpatial[0] == 1 && axes.InputSpatial[1] == 2 &&
			axes.KernelSpatial[0] == 1 && axes.KernelSpatial[1] == 2 &&
			axes.OutputSpatial[0] == 0 && axes.OutputSpatial[1] == 1
	}
	if isKernelGradSpatial &&
		inDilH == 1 && inDilW == 1 &&
		axes.InputBatch == rank-1 && axes.InputChannels == 0 &&
		axes.KernelInputChannels == 0 && axes.KernelOutputChannels == rank-1 &&
		axes.OutputBatch == rank-2 && axes.OutputChannels == rank-1 {
		var inH, inW, outH, outW, kH, kW int
		if spatialRank == 1 {
			inH, inW = 1, inputShape.Dimensions[1]
			outH, outW = 1, kernelShape.Dimensions[1]
			kH, kW = 1, outputShape.Dimensions[0]
		} else {
			inH, inW = inputShape.Dimensions[1], inputShape.Dimensions[2]
			outH, outW = kernelShape.Dimensions[1], kernelShape.Dimensions[2]
			kH, kW = outputShape.Dimensions[0], outputShape.Dimensions[1]
		}
		return Conv2DParams{
			Kind:        Conv2DKernelGrad,
			BatchSize:   inputShape.Dimensions[0],
			InH:         inH,
			InW:         inW,
			InChannels:  inputShape.Dimensions[rank-1],
			OutH:        outH,
			OutW:        outW,
			OutChannels: kernelShape.Dimensions[rank-1],
			KernelH:     kH,
			KernelW:     kW,
			StrideH:     sH,
			StrideW:     sW,
			PadTop:      padTop,
			PadLeft:     padLeft,
			InDilH:      inDilH,
			InDilW:      inDilW,
			KerDilH:     kerDilH,
			KerDilW:     kerDilW,
		}
	}

	return Conv2DParams{Kind: Conv2DUnsupported}
}

// KernelGradValidRange computes the valid output spatial range [outYStart, outYEnd) and [outXStart, outXEnd)
// and the corresponding starting input coordinates (inYStart, inXStart) for a given filter position (ky, kx)
// in Conv2DKernelGrad (when InDilH == 1 && InDilW == 1).
func (p *Conv2DParams) KernelGradValidRange(ky, kx int) (outYStart, outYEnd, inYStart, outXStart, outXEnd, inXStart int) {
	// inY = ky*StrideH + outY*KerDilH - PadTop
	minNeedY := p.PadTop - ky*p.StrideH
	if minNeedY > 0 {
		outYStart = (minNeedY + p.KerDilH - 1) / p.KerDilH
	}
	maxNeedY := p.InH + p.PadTop - ky*p.StrideH
	if maxNeedY <= 0 {
		outYEnd = 0
	} else {
		outYEnd = min(p.OutH, (maxNeedY+p.KerDilH-1)/p.KerDilH)
	}
	inYStart = ky*p.StrideH + outYStart*p.KerDilH - p.PadTop

	// inX = kx*StrideW + outX*KerDilW - PadLeft
	minNeedX := p.PadLeft - kx*p.StrideW
	if minNeedX > 0 {
		outXStart = (minNeedX + p.KerDilW - 1) / p.KerDilW
	}
	maxNeedX := p.InW + p.PadLeft - kx*p.StrideW
	if maxNeedX <= 0 {
		outXEnd = 0
	} else {
		outXEnd = min(p.OutW, (maxNeedX+p.KerDilW-1)/p.KerDilW)
	}
	inXStart = kx*p.StrideW + outXStart*p.KerDilW - p.PadLeft
	return
}

// ExtractPatches2DFloat32 extracts `mCount` output pixel patches starting at flat pixel index `mStart`
// into `patchBuf` of shape `[mCount, K]` where `K = p.KernelH * p.KernelW * p.InChannels`.
func ExtractPatches2DFloat32(p *Conv2DParams, input, patchBuf []float32, mStart, mCount int) {
	inH := p.InH
	inW := p.InW
	inC := p.InChannels
	outH := p.OutH
	outW := p.OutW
	kH := p.KernelH
	kW := p.KernelW
	sH := p.StrideH
	sW := p.StrideW
	padTop := p.PadTop
	padLeft := p.PadLeft
	kerDilH := p.KerDilH
	kerDilW := p.KerDilW
	inDilH := p.InDilH
	inDilW := p.InDilW

	kWC := kW * inC
	K := kH * kWC
	inRowStride := inW * inC
	inBatchStride := inH * inRowStride

	outHW := outH * outW
	b := mStart / outHW
	rem := mStart % outHW
	outY := rem / outW
	outX := rem % outW

	if inDilH == 1 && inDilW == 1 && kerDilW == 1 {
		for m := range mCount {
			dstRow := patchBuf[m*K : (m+1)*K]
			inBatch := input[b*inBatchStride : (b+1)*inBatchStride]
			inX0 := outX*sW - padLeft

			if inX0 >= 0 && inX0+kW <= inW {
				// Fast interior pixel: each ky row is a single contiguous slice of kW * inC floats.
				xOff := inX0 * inC
				for ky := range kH {
					dstSlice := dstRow[ky*kWC : (ky+1)*kWC]
					inY := outY*sH + ky*kerDilH - padTop
					if uint(inY) < uint(inH) {
						srcOff := inY*inRowStride + xOff
						copy(dstSlice, inBatch[srcOff:srcOff+kWC])
					} else {
						clear(dstSlice)
					}
				}
			} else {
				// Border pixel: valid kx in [kxStart, kxEnd) are still contiguous when kerDilW == 1.
				clear(dstRow)
				kxStart := max(0, -inX0)
				kxEnd := min(kW, inW-inX0)
				if kxStart < kxEnd {
					spanFloats := (kxEnd - kxStart) * inC
					dstXOff := kxStart * inC
					srcXOff := (inX0 + kxStart) * inC
					for ky := range kH {
						inY := outY*sH + ky*kerDilH - padTop
						if uint(inY) < uint(inH) {
							dstOff := ky*kWC + dstXOff
							srcOff := inY*inRowStride + srcXOff
							copy(dstRow[dstOff:dstOff+spanFloats], inBatch[srcOff:srcOff+spanFloats])
						}
					}
				}
			}

			outX++
			if outX == outW {
				outX = 0
				outY++
				if outY == outH {
					outY = 0
					b++
				}
			}
		}
		return
	}

	// General dilated path (inDilH > 1 || inDilW > 1 || kerDilW > 1).
	dilInH := (inH-1)*inDilH + 1
	dilInW := (inW-1)*inDilW + 1
	for m := range mCount {
		dstRow := patchBuf[m*K : (m+1)*K]
		clear(dstRow)
		inBatch := input[b*inBatchStride : (b+1)*inBatchStride]

		for ky := range kH {
			dilY := outY*sH + ky*kerDilH - padTop
			if uint(dilY) >= uint(dilInH) {
				continue
			}
			if inDilH > 1 {
				if dilY%inDilH != 0 {
					continue
				}
				dilY /= inDilH
			}
			inYBase := dilY * inRowStride
			dstYBase := ky * kWC

			for kx := range kW {
				dilX := outX*sW + kx*kerDilW - padLeft
				if uint(dilX) >= uint(dilInW) {
					continue
				}
				if inDilW > 1 {
					if dilX%inDilW != 0 {
						continue
					}
					dilX /= inDilW
				}
				srcOff := inYBase + dilX*inC
				dstOff := dstYBase + kx*inC
				copy(dstRow[dstOff:dstOff+inC], inBatch[srcOff:srcOff+inC])
			}
		}

		outX++
		if outX == outW {
			outX = 0
			outY++
			if outY == outH {
				outY = 0
				b++
			}
		}
	}
}

// PackConvKernelFloat32 packs the convolution kernel into panels of width `panelCols`:
// shape `[numPanels, K, panelCols]` where `numPanels = (OutChannels + panelCols - 1) / panelCols`
// and `K = KernelH * KernelW * InChannels`.
// Handles both standard layout `[kH, kW, InChannels, OutChannels]` and transposed channel layout
// `[kH, kW, OutChannels, InChannels]` (used in VJP wrt Input).
func PackConvKernelFloat32(p *Conv2DParams, kernel, packedRHS []float32, panelCols int) {
	inC := p.InChannels
	outC := p.OutChannels
	spatialSize := p.KernelH * p.KernelW
	K := spatialSize * inC
	numPanels := (outC + panelCols - 1) / panelCols
	panelSize := K * panelCols

	if !p.IsKernelTransposed {
		// kernel is [K, outC]
		for panelIdx := range numPanels {
			colStart := panelIdx * panelCols
			validCols := min(panelCols, outC-colStart)
			dstPanel := packedRHS[panelIdx*panelSize : (panelIdx+1)*panelSize]
			if validCols == panelCols {
				for k := range K {
					srcOff := k*outC + colStart
					dstOff := k * panelCols
					copy(dstPanel[dstOff:dstOff+panelCols], kernel[srcOff:srcOff+panelCols])
				}
			} else {
				clear(dstPanel)
				for k := range K {
					srcOff := k*outC + colStart
					dstOff := k * panelCols
					copy(dstPanel[dstOff:dstOff+validCols], kernel[srcOff:srcOff+validCols])
				}
			}
		}
		return
	}

	// kernel is [spatialSize, outC, inC] (transposed channel axes for VJP wrt Input)
	for panelIdx := range numPanels {
		colStart := panelIdx * panelCols
		validCols := min(panelCols, outC-colStart)
		dstPanel := packedRHS[panelIdx*panelSize : (panelIdx+1)*panelSize]
		if validCols < panelCols {
			clear(dstPanel)
		}
		for s := range spatialSize {
			srcSpatial := kernel[s*outC*inC : (s+1)*outC*inC]
			dstSpatial := dstPanel[s*inC*panelCols : (s+1)*inC*panelCols]
			for cOut := range validCols {
				srcRow := srcSpatial[(colStart+cOut)*inC : (colStart+cOut+1)*inC]
				for cIn := range inC {
					dstSpatial[cIn*panelCols+cOut] = srcRow[cIn]
				}
			}
		}
	}
}

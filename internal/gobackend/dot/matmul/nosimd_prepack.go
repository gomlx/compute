// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package matmul

import (
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/gobackend/dot"
)

// noSIMDPrepackRHS pre-packs the RHS matrix into a single flat buffer and returns panel slices.
func noSIMDPrepackRHS[I gotype.ScalarNotComplex](
	backend *gobackend.Backend,
	layout dot.Layout,
	rhs []I,
	batchSize, rhsCrossSize, contractingSize int,
	params CacheParams,
) (*gobackend.Buffer, [][]I) {
	numKPanels := (contractingSize + params.PanelContractingSize - 1) / params.PanelContractingSize
	numColPanels := (rhsCrossSize + params.RHSPanelCrossSize - 1) / params.RHSPanelCrossSize
	rhsPanelsPerBatch := numKPanels * numColPanels
	rhsBatchStride := rhsCrossSize * contractingSize

	totalElements := 0
	for colPanelIdx := range numColPanels {
		rhsPanelColIdx := colPanelIdx * params.RHSPanelCrossSize
		rhsPanelWidth := min(params.RHSPanelCrossSize, rhsCrossSize-rhsPanelColIdx)
		numStrips := (rhsPanelWidth + params.RHSL1KernelCols - 1) / params.RHSL1KernelCols
		for kPanelIdx := range numKPanels {
			contractingPanelIdx := kPanelIdx * params.PanelContractingSize
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			totalElements += contractingPanelWidth * numStrips * params.RHSL1KernelCols
		}
	}
	totalElements *= batchSize

	ref, flatBuf, ok := GetBuffer[I](backend, totalElements)
	if !ok {
		return nil, nil
	}

	panels := make([][]I, batchSize*rhsPanelsPerBatch)

	offset := 0
	rhsFlatIdx := 0
	for b := range batchSize {
		batchRHS := rhs[rhsFlatIdx : rhsFlatIdx+rhsBatchStride]
		for colPanelIdx := range numColPanels {
			rhsPanelColIdx := colPanelIdx * params.RHSPanelCrossSize
			rhsPanelWidth := min(params.RHSPanelCrossSize, rhsCrossSize-rhsPanelColIdx)
			numStrips := (rhsPanelWidth + params.RHSL1KernelCols - 1) / params.RHSL1KernelCols
			for kPanelIdx := range numKPanels {
				contractingPanelIdx := kPanelIdx * params.PanelContractingSize
				contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
				panelLen := contractingPanelWidth * numStrips * params.RHSL1KernelCols
				panelBuf := flatBuf[offset : offset+panelLen]
				offset += panelLen

				if layout == dot.LayoutNonTransposed {
					unsafePackRHS(batchRHS, panelBuf, contractingPanelIdx, rhsPanelColIdx, rhsCrossSize, contractingPanelWidth, rhsPanelWidth, params.RHSL1KernelCols)
				} else {
					unsafePackLHS(batchRHS, panelBuf, rhsPanelColIdx, contractingPanelIdx, contractingSize, rhsPanelWidth, contractingPanelWidth, params.RHSL1KernelCols)
				}
				panels[b*rhsPanelsPerBatch+kPanelIdx*numColPanels+colPanelIdx] = panelBuf
			}
		}
		rhsFlatIdx += rhsBatchStride
	}
	return ref, panels
}

// noSIMDPrepackLHS pre-packs the LHS matrix into a single flat buffer and returns panel slices.
func noSIMDPrepackLHS[I gotype.ScalarNotComplex](
	backend *gobackend.Backend,
	lhs []I,
	batchSize, lhsCrossSize, contractingSize int,
	params CacheParams,
) (*gobackend.Buffer, [][]I) {
	numKPanels := (contractingSize + params.PanelContractingSize - 1) / params.PanelContractingSize
	numRowPanels := (lhsCrossSize + params.LHSPanelCrossSize - 1) / params.LHSPanelCrossSize
	lhsPanelsPerBatch := numKPanels * numRowPanels
	lhsBatchStride := lhsCrossSize * contractingSize

	totalElements := 0
	for rowPanelIdx := range numRowPanels {
		lhsPanelRowIdx := rowPanelIdx * params.LHSPanelCrossSize
		lhsPanelHeight := min(params.LHSPanelCrossSize, lhsCrossSize-lhsPanelRowIdx)
		numStrips := (lhsPanelHeight + params.LHSL1KernelRows - 1) / params.LHSL1KernelRows
		for kPanelIdx := range numKPanels {
			contractingPanelIdx := kPanelIdx * params.PanelContractingSize
			contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
			totalElements += contractingPanelWidth * numStrips * params.LHSL1KernelRows
		}
	}
	totalElements *= batchSize

	ref, flatBuf, ok := GetBuffer[I](backend, totalElements)
	if !ok {
		return nil, nil
	}

	panels := make([][]I, batchSize*lhsPanelsPerBatch)

	offset := 0
	lhsFlatIdx := 0
	for b := range batchSize {
		batchLHS := lhs[lhsFlatIdx : lhsFlatIdx+lhsBatchStride]
		for rowPanelIdx := range numRowPanels {
			lhsPanelRowIdx := rowPanelIdx * params.LHSPanelCrossSize
			lhsPanelHeight := min(params.LHSPanelCrossSize, lhsCrossSize-lhsPanelRowIdx)
			numStrips := (lhsPanelHeight + params.LHSL1KernelRows - 1) / params.LHSL1KernelRows
			for kPanelIdx := range numKPanels {
				contractingPanelIdx := kPanelIdx * params.PanelContractingSize
				contractingPanelWidth := min(params.PanelContractingSize, contractingSize-contractingPanelIdx)
				panelLen := contractingPanelWidth * numStrips * params.LHSL1KernelRows
				panelBuf := flatBuf[offset : offset+panelLen]
				offset += panelLen

				unsafePackLHS(batchLHS, panelBuf, lhsPanelRowIdx, contractingPanelIdx, contractingSize, lhsPanelHeight, contractingPanelWidth, params.LHSL1KernelRows)
				panels[b*lhsPanelsPerBatch+kPanelIdx*numRowPanels+rowPanelIdx] = panelBuf
			}
		}
		lhsFlatIdx += lhsBatchStride
	}
	return ref, panels
}

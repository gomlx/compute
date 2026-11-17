// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapeinference

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestQuantizedGather(t *testing.T) {
	t.Run("NilQuant_SubByte", func(t *testing.T) {
		indices := S(dtypes.Int32, 2, 5, 1)
		for _, tc := range []struct {
			in, want dtypes.DType
		}{
			{dtypes.Int2, dtypes.Int8},
			{dtypes.Int4, dtypes.Int8},
			{dtypes.Int8, dtypes.Int8},
			{dtypes.Uint2, dtypes.Uint8},
			{dtypes.Uint4, dtypes.Uint8},
			{dtypes.Uint8, dtypes.Uint8},
		} {
			out, err := QuantizedGather(S(tc.in, 100, 16), indices, nil)
			if err != nil {
				t.Fatalf("QuantizedGather(%s, nil) unexpected error: %+v", tc.in, err)
			}
			if !out.Equal(S(tc.want, 2, 5, 16)) {
				t.Fatalf("QuantizedGather(%s, nil): expected %s[2, 5, 16], got %s", tc.in, tc.want, out)
			}
		}

		if _, err := QuantizedGather(S(F32, 10, 4), indices, nil); err == nil {
			t.Fatalf("expected error for Float32 with xQuantization=nil")
		}
	})

	t.Run("QuantNone_And_PackedDType", func(t *testing.T) {
		indices := S(dtypes.Int32, 2, 5, 1)

		// Native Int4 with explicit QuantNone.
		out, err := QuantizedGather(S(dtypes.Int4, 100, 16), indices, &QuantizationShapes{
			Scheme: compute.QuantNone,
		})
		if err != nil || !out.Equal(S(dtypes.Int8, 2, 5, 16)) {
			t.Fatalf("QuantNone native Int4: got %s, err=%+v", out, err)
		}

		// Packed Uint8 [100, 4] with PackedDType=Int2 -> [2, 5, 16] Int8.
		out, err = QuantizedGather(S(dtypes.Uint8, 100, 4), indices, &QuantizationShapes{
			Scheme:      compute.QuantNone,
			PackedDType: dtypes.Int2,
		})
		if err != nil || !out.Equal(S(dtypes.Int8, 2, 5, 16)) {
			t.Fatalf("QuantNone PackedDType=Int2: got %s, err=%+v", out, err)
		}

		// Packed Uint8 [100, 24] with PackedDType=Uint4 + QuantLinear block scale.
		scale := S(dtypes.Float16, 100, 2) // logical K=48, BlockSize=32 -> 2 blocks
		out, err = QuantizedGather(S(dtypes.Uint8, 100, 24), indices, &QuantizationShapes{
			Scheme:      compute.QuantLinear,
			Scale:       scale,
			BlockAxis:   1,
			BlockSize:   32,
			PackedDType: dtypes.Uint4,
		})
		if err != nil || !out.Equal(S(dtypes.Float16, 2, 5, 48)) {
			t.Fatalf("QuantLinear PackedDType=Uint4: got %s, err=%+v", out, err)
		}

		// Validation errors.
		if _, err := QuantizedGather(S(dtypes.Int8, 10, 4), indices, &QuantizationShapes{
			Scheme:      compute.QuantNone,
			PackedDType: dtypes.Int4,
		}); err == nil {
			t.Fatalf("expected error when PackedDType is used on non-Uint8 x")
		}
		if _, err := QuantizedGather(S(dtypes.Uint8, 10, 4), indices, &QuantizationShapes{
			Scheme:      compute.QuantNone,
			PackedDType: dtypes.Int8,
		}); err == nil {
			t.Fatalf("expected error when PackedDType is non-packed (Int8)")
		}
		if _, err := QuantizedGather(S(dtypes.Uint8, 10, 4), indices, &QuantizationShapes{
			Scheme: compute.QuantNone,
			Scale:  S(F32, 10),
		}); err == nil {
			t.Fatalf("expected error when QuantNone sets Scale")
		}
		if _, err := QuantizedGather(S(dtypes.Uint8, 10, 36), indices, &QuantizationShapes{
			Scheme:      compute.QuantGGML,
			GGMLType:    compute.GGMLQ4_0,
			PackedDType: dtypes.Int4,
		}); err == nil {
			t.Fatalf("expected error when PackedDType is used with QuantGGML")
		}
	})

	t.Run("GGML", func(t *testing.T) {
		x := S(dtypes.Uint8, 1000, 36) // Q4_0: K=64
		indices := S(dtypes.Int32, 2, 5, 1)
		out, err := QuantizedGather(x, indices, &QuantizationShapes{
			Scheme:   compute.QuantGGML,
			GGMLType: compute.GGMLQ4_0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !out.Equal(S(F32, 2, 5, 64)) {
			t.Fatalf("expected F32[2, 5, 64], got %s", out)
		}
	})

	t.Run("Linear_BlockQuantized_DynamicIndices", func(t *testing.T) {
		x := S(dtypes.Uint4, 100, 48)
		indices := shapes.MakeDynamic(dtypes.Int64, []int{shapes.DynamicDim, 3, 1}, []string{"batch", "", ""})
		scale := S(dtypes.Float16, 100, 2) // BlockSize=32, non-divisible K=48 -> 2 blocks
		zp := S(dtypes.Uint4, 100, 2)
		info, err := QuantizedGatherWithInfo(x, indices, &QuantizationShapes{
			Scheme:    compute.QuantLinear,
			Scale:     scale,
			ZeroPoint: zp,
			BlockAxis: 1,
			BlockSize: 32,
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		want := shapes.MakeDynamic(dtypes.Float16, []int{shapes.DynamicDim, 3, 48}, []string{"batch", "", ""})
		if !info.OutputShape.Equal(want) {
			t.Fatalf("expected %s, got %s", want, info.OutputShape)
		}
		if info.K != 48 || info.ScaleLayout.ColBlockSize != 32 || info.ScaleLayout.NumColBlocks != 2 {
			t.Fatalf("unexpected info: %+v", info)
		}
	})

	t.Run("Linear_Uint2_Symmetric", func(t *testing.T) {
		x := S(dtypes.Uint2, 50, 16)
		indices := S(dtypes.Int32, 4, 1)
		scale := S(dtypes.BFloat16, 50)
		out, err := QuantizedGather(x, indices, &QuantizationShapes{
			Scheme: compute.QuantLinear,
			Scale:  scale,
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !out.Equal(S(dtypes.BFloat16, 4, 16)) {
			t.Fatalf("expected BF16[4, 16], got %s", out)
		}
	})

	t.Run("NF4", func(t *testing.T) {
		x := S(dtypes.Uint4, 64, 32)
		indices := S(dtypes.Int32, 2, 1)
		scale := S(dtypes.Float32, 64, 1)
		out, err := QuantizedGather(x, indices, &QuantizationShapes{
			Scheme:    compute.QuantNF4,
			Scale:     scale,
			BlockAxis: 1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if !out.Equal(S(F32, 2, 32)) {
			t.Fatalf("expected F32[2, 32], got %s", out)
		}
	})
}

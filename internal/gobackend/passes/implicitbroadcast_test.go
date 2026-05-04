package passes_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/ops"
	_ "github.com/gomlx/compute/internal/gobackend/passes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestImplicitBroadcastFusion(t *testing.T) {
	backend := compute.MustNew()
	defer backend.Finalize()

	t.Run("MatchRankFusion", func(t *testing.T) {
		builder := backend.Builder("MatchRankFusion").(*gobackend.Builder)
		main := builder.Main().(*gobackend.Function)

		// Input: [2, 3], Shape: [2, 3]
		x, _ := main.Parameter("x", shapes.Make(dtypes.Float32, 2, 1), nil)
		// Broadcast to [2, 3]
		xBroadcast, _ := main.BroadcastInDim(x, shapes.Make(dtypes.Float32, 2, 3), []int{0, 1})
		// Another input [2, 3]
		y, _ := main.Parameter("y", shapes.Make(dtypes.Float32, 2, 3), nil)

		// Binary op: Add [2, 3] + [2, 3]
		z, _ := main.Add(xBroadcast, y)
		_ = main.Return([]compute.Value{z}, nil)

		// Verify xBroadcast is used by z before optimization (Wait, optimization happens during Compile)
		zNode := z.(*gobackend.Node)
		if zNode.Inputs[0].OpType != compute.OpTypeBroadcastInDim {
			t.Fatalf("Expected first input of Add to be BroadcastInDim before optimization, got %s", zNode.Inputs[0].OpType)
		}

		// Compile triggers Optimization.
		_, err := builder.Compile()
		if err != nil {
			t.Fatalf("Failed to compile: %+v", err)
		}

		// Verify that z's input was fused to x.
		if zNode.Inputs[0].OpType == compute.OpTypeBroadcastInDim {
			t.Errorf("BroadcastInDim should have been fused out")
		}
		if zNode.Inputs[0] != x.(*gobackend.Node) {
			t.Errorf("Expected first input of Add to be x, got %s", zNode.Inputs[0].OpType)
		}

		// Execute to ensure correctness
		exec, _ := builder.Compile() // Re-compile because we want to test execution
		xData := []float32{1, 2}
		yData := []float32{10, 20, 30, 40, 50, 60}
		xBuf, _ := backend.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, 2, 1))
		yBuf, _ := backend.BufferFromFlatData(0, yData, shapes.Make(dtypes.Float32, 2, 3))
		outputs, err := exec.Execute([]compute.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		result := outputs[0].(*gobackend.Buffer).Flat.([]float32)
		expected := []float32{11, 21, 31, 42, 52, 62} // 1 broadcasted to [10, 20, 30], 2 broadcasted to [40, 50, 60]
		if ok, diff := testutil.IsEqual(expected, result); !ok {
			t.Errorf("Incorrect result:\n%s", diff)
		}
	})

	t.Run("DifferentRankNoFusion", func(t *testing.T) {
		builder := backend.Builder("DifferentRankNoFusion").(*gobackend.Builder)
		main := builder.Main().(*gobackend.Function)

		// Input: [2], Shape: [2]
		x, _ := main.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
		// Broadcast to [2, 3]
		xBroadcast, _ := main.BroadcastInDim(x, shapes.Make(dtypes.Float32, 2, 3), []int{0})
		// Another input [2, 3]
		y, _ := main.Parameter("y", shapes.Make(dtypes.Float32, 2, 3), nil)

		// Binary op: Add [2, 3] + [2, 3]
		z, _ := main.Add(xBroadcast, y)
		_ = main.Return([]compute.Value{z}, nil)

		zNode := z.(*gobackend.Node)
		_, err := builder.Compile()
		if err != nil {
			t.Fatalf("Failed to compile: %+v", err)
		}

		// Verify that z's input was NOT fused because ranks differ (1 vs 2).
		if zNode.Inputs[0].OpType != compute.OpTypeBroadcastInDim {
			t.Errorf("BroadcastInDim should NOT have been fused out because ranks differ")
		}

		// Execute to ensure correctness
		exec, _ := builder.Compile()
		xData := []float32{1, 2}
		yData := []float32{10, 20, 30, 40, 50, 60}
		xBuf, _ := backend.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, 2))
		yBuf, _ := backend.BufferFromFlatData(0, yData, shapes.Make(dtypes.Float32, 2, 3))
		outputs, err := exec.Execute([]compute.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}
		result := outputs[0].(*gobackend.Buffer).Flat.([]float32)
		expected := []float32{11, 21, 31, 42, 52, 62}
		if ok, diff := testutil.IsEqual(expected, result); !ok {
			t.Errorf("Incorrect result:\n%s", diff)
		}
	})
}

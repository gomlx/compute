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

	t.Run("BothInputsBroadcast", func(t *testing.T) {
		builder := backend.Builder("BothInputsBroadcast").(*gobackend.Builder)
		main := builder.Main().(*gobackend.Function)

		// lhs: [1, 1, 1], Broadcasted to [2, 1, 5]
		x, _ := main.Parameter("x", shapes.Make(dtypes.Float32, 1, 1, 1), nil)
		xBroadcast, _ := main.BroadcastInDim(x, shapes.Make(dtypes.Float32, 2, 1, 5), []int{0, 1, 2})

		// rhs: [2, 1, 1], Broadcasted to [2, 3, 1]
		y, _ := main.Parameter("y", shapes.Make(dtypes.Float32, 2, 1, 1), nil)
		yBroadcast, _ := main.BroadcastInDim(y, shapes.Make(dtypes.Float32, 2, 3, 1), []int{0, 1, 2})

		// Binary op: Add [2, 1, 5] + [2, 3, 1] -> Implicit broadcast to [2, 3, 5]
		z, _ := main.Add(xBroadcast, yBroadcast)

		// Add another node to check that the graph is properly re-sorted after optimization.
		zSqrt, err := main.Sqrt(z)
		if err != nil {
			t.Fatalf("Failed to create Sqrt node: %+v", err)
		}

		_ = main.Return([]compute.Value{zSqrt}, nil)

		zNode := z.(*gobackend.Node)
		// Before compilation, inputs are BroadcastInDim.
		if zNode.Inputs[0].OpType != compute.OpTypeBroadcastInDim || zNode.Inputs[1].OpType != compute.OpTypeBroadcastInDim {
			t.Fatalf("Expected both inputs of Add to be BroadcastInDim before optimization")
		}

		// Compile triggers Optimization.
		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Failed to compile: %+v", err)
		}

		// Verify that z's input was fused to x and y.
		if zNode.Inputs[0].OpType == compute.OpTypeBroadcastInDim || zNode.Inputs[1].OpType == compute.OpTypeBroadcastInDim {
			t.Errorf("BroadcastInDim should have been fused out")
		}
		if zNode.Inputs[0] != x.(*gobackend.Node) || zNode.Inputs[1] != y.(*gobackend.Node) {
			t.Errorf("Expected inputs of Add to be x and y")
		}

		// Execute to ensure correctness
		xData := []float32{2}
		yData := []float32{10, 20}
		xBuf, _ := backend.BufferFromFlatData(0, xData, shapes.Make(dtypes.Float32, 1, 1, 1))
		yBuf, _ := backend.BufferFromFlatData(0, yData, shapes.Make(dtypes.Float32, 2, 1, 1))
		outputs, err := exec.Execute([]compute.Buffer{xBuf, yBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute failed: %+v", err)
		}

		resultBuf := outputs[0].(*gobackend.Buffer)
		shape, _ := resultBuf.Shape()
		if !shape.Equal(shapes.Make(dtypes.Float32, 2, 3, 5)) {
			t.Errorf("Incorrect result shape: %s", shape)
		}

		result := resultBuf.Flat.([]float32)
		expected := make([]float32, 30)
		for i := 0; i < 15; i++ {
			expected[i] = 12
		}
		for i := 15; i < 30; i++ {
			expected[i] = 22
		}

		if ok, diff := testutil.IsEqual(expected, result); !ok {
			t.Errorf("Incorrect result:\n%s", diff)
		}
	})
}

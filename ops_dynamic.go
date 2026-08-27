package compute

// DynamicDimensionSpec specifies a target dimension for DynamicReshape.
// It can be one of three:
//   - Static: known at graph building time, e.g.: `DynamicDimensionSpec{Static: 16}`.
//   - Dynamic: named dynamic dimension given with a graph.Value: e.g.: `DynmicDimensionSpec{Name: "batch", Value: batchSize}`.
//   - Inferred: named dynamic dimension given with a inferred dimension: e.g.: `DynmicDimensionSpec{Name: "seq_len"}`.
type DynamicDimensionSpec struct {
	// Static dimension size (>= 0). It is ignored if a Name is set.
	Static int

	// Name of a dynamic axis dimension or for an inferred dimension (at most one inferred
	// dimension). Empty string for static dimensions.
	Name string

	// Scalar integer value for runtime dimension size (nil if static or auto-inferred).
	Value Value
}

// DynamicOps defines the operations that expect or operate on dynamic shapes.
type DynamicOps interface {
	// DynamicDimensionSize returns the dimension of the given axis of the operand as a dynamic scalar value.
	// This is only supported by backends that support dynamic shapes (see Capabilities.DynamicAxes).
	DynamicDimensionSize(operand Value, axis int) (Value, error)

	// DynamicReshape reshapes x to target dimensions specified by dimensions.
	//
	// Each dimension can be:
	// - Static;
	// - Dynamic: a Name and (dynamic) Value are provided.
	// - Auto-inferred: only a Name is provided, at most one axis can be auto-inferred.
	//
	// Usually, this operation is only supported if the backend supports dynamic axes (Capabilities.DynamicAxes).
	DynamicReshape(operand Value, dimensions ...DynamicDimensionSpec) (Value, error)
}

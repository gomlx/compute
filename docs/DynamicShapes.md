# Dynamic Shapes

Dynamic shapes support for the `compute.Backend` for now entails allowing a computation graph to be defined
with input parameters having symbolic axes: these axes have unspecified dimensions but can be named (e.g.: "batchSize").

These names are validated (if operands have named axes, their names must match). 

Compilation leave the concrete shapes unresolved: symbolic shapes are only made concrete during executiong when the input shape is known. This avoids recompilation when only batch size (or other marked axes) change between calls.

## What is not included

Data dynamism, when the shapes depend on the content of the data (e.g.: it won't support a `Unique(x)` operation,
whose results depends on how many unique values there are in `x`).

## Phases

- **Phase 0**: Named dynamic axes foundation — `DynamicDim` sentinel, axis names on shapes, `Shape.Resolve()` / `Shape.HasDynamicDims()` / `AxisBindings`
- **Phase 1**: Two-level JIT shape specialization — `ShapeSpecialization` with shallow node copies and resolved shapes, cached per binding key via `sync.Map`
- **Phase 2**: Power-of-2 buffer pool bucketing — prevents excessive pool fragmentation when buffer sizes vary across specializations
- **Phase 3**: Dynamic shapes for DotGeneral, ConvGeneral, Gather, Broadcast, Reshape, ReduceMean, and Concatenate — graph builders accept dynamic inputs and specialization recomputes shape-dependent `node.data`

## Key design decisions

- **Per-specialization `node.data` override**: Shallow copies in `createSpecialization` already create new `*Node` structs. For DotGeneral/ConvGeneral/Gather, new data structs are created with values recomputed from resolved concrete shapes. Executors read `node.data` unchanged — they get the specialization-specific copy.
- **No executor changes**: All `exec*.go` files are untouched. Executors work on resolved nodes with concrete shapes and data.
- **Axis name propagation**: Every `DynamicDim` in output shapes carries an axis name so `Shape.Resolve()` works at specialization time. `propagateDynamicAxisNames` in `function_dedup.go` ensures this for ops that don't explicitly set names.
- **`DynamicAxes` capability flag**: Backends declare support; `graph.Exec.WithDynamicAxes()` checks the flag and returns a clear error if unsupported.

# Dynamic Shapes

Dynamic shapes support in `compute.Backend` allows computation graphs to be built and compiled once when one or more tensor dimensions are unknown at compile time (`shapes.DynamicDim` = `-1`) and resolved to concrete non-negative sizes (`>= 0`) at execution time.

## Axis Naming Rules

Every dynamic axis (`Dimensions[i] == shapes.DynamicDim`) **must** have a non-empty axis name (`AxisNames[i] != ""`), used to track symbolic relationships and resolve concrete shapes at runtime:

1. **User-Named Dynamic Axes (e.g. `"batch"`, `"seq"`):**
   - Strictly validated at graph-build time. Pairing `"batch"` with `"seq"` on axes that must match is an immediate build-time error.
2. **`#`-Prefixed Generated Axes (`shapes.GeneratedAxisPrefix = "#"`, `shapes.UniqueAxisName(prefix)`):**
   - Used for non-user-named dynamic dimensions and data-dependent outputs (such as `#nonzeros_<n>` from `NonZero`, `#conv_<n>` from strided `ConvGeneral`, `#pad_<n>` from `DynamicPad`, or `#concat_<n>` from `Concatenate`).
   - A generated axis matches itself statically (`#a_1 == #a_1`, preserving self-identity in expressions like `x * (x + 1)`), while two *distinct* generated axes (or a generated axis paired with a user-named axis) unify permissively at graph-build time via `shapes.UnifyAxisName` and **must be verified to have equal concrete dimensions at runtime**.
3. **Symbolic Sum Expressions (`shapes.SymbolicAxisPrefix = "="`, e.g. `"=4+batch"` or `"=a+b"`):**
   - Produced by operations like `Concatenate` along a dynamic axis and evaluated against `shapes.AxisBindings` during shape resolution.

---

## Broadcasting Rules with Dynamic and Zero-Sized Shapes

### 1. Implicit Broadcasting (`BinaryOp`, `ComparisonOp`, `Where`)

Implicit broadcasting in `compute.Backend` (modeled after XLA/StableHLO) is strictly a **compile-time** property:

- **Static unit dimension (`1`) or scalar:** Implicitly broadcasts to the other operand's dimension $N$ — whether $N$ is static (`> 0`), **`0`** (repeating a slice of length `1` zero times yields an empty axis of size `0`), or dynamic (`shapes.DynamicDim`). Conversely, a non-`1` static dimension (including `0` or `>= 2`) cannot broadcast to a different dimension.
- **Dynamic vs. Dynamic:** Two compatible dynamic dimensions pair element-wise without broadcasting. Backends enforce at runtime that their concrete sizes are equal — a dynamic axis whose concrete runtime size happens to be `1` does **not** implicitly broadcast to another dynamic axis of size `N != 1`.
- **Dynamic vs. Non-`1` Static Dimension (Build-Time Error):** Pairing an unknown/dynamic dimension with a non-`1` static dimension (for example, `Add(["batch", 12], ["batch", "#my_features"])`) is rejected at graph-build time so shape bugs and missing `ExpandAxes` calls are not hidden until runtime:
  - To **assert** that `"#my_features"` equals `12` at runtime, explicitly reshape the operand with `DynamicReshape` to `["batch", 12]`.
  - To **broadcast** `"#my_features"` at runtime when its concrete size may be either `1` or `12` (as in ONNX / PyTorch multidirectional broadcasting), use `DynamicBroadcastInDim`.

### 2. Explicit Dynamic Broadcasting (`DynamicBroadcastInDim`)

Unlike `BinaryOp`, `DynamicBroadcastInDim` explicitly broadcasts an operand to target `DynamicDimensionSpec`s:

- **Static `1` $\to$ Static or Dynamic:** Broadcasts `1` to any target size `>= 0` (including `0`).
- **Dynamic $\to$ Static or Dynamic:** Allowed when axis names are compatible (and, if the target dynamic axis name differs from the operand's, the target spec provides a scalar `Value` or references a known dynamic axis). At runtime, the concrete operand dimension must either equal `1` (broadcasting `1 -> outDim`, including `1 -> 0`) or equal `outDim`.
- **Static `!= 1` (`0` or `>= 2`) $\to$ Dynamic (Static Narrowing):** Allowed at build time. Because a non-`1` static axis cannot broadcast to any other size, `inDim` is the **only** valid concrete runtime value; `shapeinference.DynamicBroadcastInDim` narrows the inferred output dimension to static `inDim` (preserving static shape information for downstream ops), while backends still validate at runtime that the dynamic target spec evaluates to `inDim`.

---

## Zero-Sized Shapes and Dead-Code Elimination (`gobackend`)

A shape with at least one static `0` dimension (`Shape.IsZeroSize() == true`) contains zero elements regardless of other axes:

1. **Fully Static Zero-Size Outputs (`!shape.IsDynamic() && shape.IsZeroSize()`, e.g. `Float32[0, 4]`):**
   - During `FunctionExecutable` compilation (`foldZeroSizeNodes` in `internal/gobackend/zero_size.go`), ops with no pending runtime validation are folded in-place into leaf `OpTypeConstant` nodes (`Inputs = nil`). Any upstream subgraph whose only consumer was the zero-sized op has `NumUses == 0` and is completely dead-code-eliminated.
2. **Dynamic Zero-Size Outputs (`shape.IsDynamic() && shape.IsZeroSize()`, e.g. `Float32["batch", 0]`):**
   - Even though zero data elements exist, the concrete runtime value of `"batch"` must still propagate to the output shape.
   - `foldZeroSizeNodes` converts safe ops into no-op shape-dependency nodes (`Node.IsZeroSizeShapeOnly = true`) and rewires their `Inputs` directly to the earliest defining node(s) in `Function.dynamicAxisSources` (such as the input `Parameter`, including across `"=..."` symbolic sum expressions). All intermediate compute nodes between the parameter and the zero-sized op are dead-code-eliminated, and execution simply acquires a pooled 0-byte `*Buffer` header (`0 B/op, 0 allocs/op`) carrying the resolved concrete shape.
3. **Runtime-Zero Dynamic Shapes (e.g. `Add([1, 4], ["batch", 4])` when `batch == 0` at runtime):**
   - Handled via two-stage shape specialization (`specializeFunction` for parameter-bound `shapes.AxisBindings`, and `specializeNode` for data-dependent / intermediate axes) followed by immediate `Size() == 0` no-op returns in op executors.

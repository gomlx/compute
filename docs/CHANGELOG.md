# Initial release

- Moved GoMLX's `backends/simplego` to `gobackend`.
- Removed all dependencies to `stretchr/testify` and `gomlx`, to trim as much as possible external dependencies.
- Moved `gobackend` generic tests to `support/backendtest`, so they can be used by other backends.
- Package `gobackend`:
  - Fixed definition of `Bitcast` when casting to a larger target dtype: the rank is shrinked by 1.
- Package `support`:
  - The following packages were moved from `github.com/gomlx/gomlx/pkg/support/...` to `support/...`: `xslices`, `xsync`, `sets` and `humanize`.

- Package `shapes`: added initial support for dynamic shapes (see `./docs/DynamicShapes.md` for overall idea):
  - Add `Shape.Resolve(AxisBindings) (Shape, error)` method.
  - Add `Shape.IsDynamic()` method.
  - Add `DynamicDim` type.

- New ops:
  - `SchedulingBarrier` and `OptimizationBarrier`, both implemented in the Go backend.
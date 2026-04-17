# Compute

Package `compute` provides a modular API for defining and executing multidimensional computation graphs with pluggable backends.

It defines `shapes` (tensor shapes) and `dtypes` (data types) and the top-level `compute` package defines a `Backend` API (a series of interfaces), that
can be used to define a computation graph, JIT-compile it, transfor buffers (raw values) to/from the backend, and execute compiled computations.

It powers [GoMLX](https://github.com/gomlx/gomlx), the machine learning framework for Go, but can be used directly also. With the caveat that the `compute.Backend` doesn't aim to be ergonomic, but instead "correct" and "minimal". For a more convenient API for complex computation, and auto-differentiation, use GoMLX instead.

There are currently 3 backend implementations:

- Package `native`: provides a native Go implementation, hence very portable (including it runs in WASM). It covers 80% of the API (some ops are still missing). EXPERIMENTAL: we are working on SIMD versions, for architectures that support it (with runtime dispatching for instance on _amd64_, so one binary will take advantage of AVX2 / AVX512, whichever is available), it offers huge gains.
- Package `xla`: provides an [XLA (PJRT)](https://openxla.org/) based implementation, the same used by Jax and TensorFlow. It uses CGO (it's a C++ library), but it supports GPUs and TPUs, as well as a fast CPU, proper JIT compilation. Limited to static shapes though.
- The project [go-darwinml](https://github.com/gomlx/go-darwinml/) is an **experimental** support to Apple's CoreML, with accelerate for GPU (Metal) and CPU (arm64).

Other future backends: ONNX backend, [llama.cpp](https://github.com/ggml-org/llama.cpp) (using [github.com/hybridgroup/yzma](https://github.com/hybridgroup/yzma) a "pure-go" binding to it), [WebNN](https://learn.microsoft.com/en-us/windows/ai/directml/webnn-overview) or WebGL.


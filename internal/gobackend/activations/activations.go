// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package activations

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// InPlaceFn applies an activation function in-place to a contiguous slice of elements.
type InPlaceFn[T any] func(data []T)

// KernelFn applies an activation function from in to out (in and out can be the same slice).
type KernelFn[T any] func(in, out []T)

// SwiGLUFn applies SwiGLU activation: out[m*H + j] = silu(gate[m*2H + j]) * val[m*2H + H + j].
type SwiGLUFn[T any] func(in, out []T, numRows, hiddenDim int)

// VJPKernelFn applies the VJP of an activation: dx = dOutput * f'(y or x).
// Note: y and/or x may be empty/nil depending on whether the activation requires x.
type VJPKernelFn[T any] func(y, x, dOutput, dx []T)

// SwiGLUVJPFn applies SwiGLU VJP:
// x: input of shape [numRows, 2*hiddenDim]
// dOutput: adjoint gradient of shape [numRows, hiddenDim]
// dx: resulting gradient of shape [numRows, 2*hiddenDim]
type SwiGLUVJPFn[T any] func(x, dOutput, dx []T, numRows, hiddenDim int)

// DefaultMinParallelChunk is the default minimum number of elements per parallel worker chunk.
const DefaultMinParallelChunk = 32768

// KernelMetadata holds properties and platform-tunable parameters for a registered activation kernel.
// Because registrations store a *KernelMetadata pointer, package-level metadata structs can be
// initialized or adjusted at runtime for the target platform without re-registering kernels.
type KernelMetadata struct {
	// MinParallelChunk is the minimum number of elements per parallel worker chunk.
	// If <= 0, DefaultMinParallelChunk is used.
	MinParallelChunk int
}

// EffectiveMinParallelChunk returns MinParallelChunk if positive, or DefaultMinParallelChunk.
func (m *KernelMetadata) EffectiveMinParallelChunk() int {
	if m != nil && m.MinParallelChunk > 0 {
		return m.MinParallelChunk
	}
	return DefaultMinParallelChunk
}

// DefaultKernelMetadata is the default metadata used when nil is passed to Register*.
var DefaultKernelMetadata = KernelMetadata{
	MinParallelChunk: DefaultMinParallelChunk,
}

type registrationKey struct {
	act   compute.ActivationType
	dtype dtypes.DType
}

type registeredImpl struct {
	name     string
	inPlace  any
	kernel   any
	priority gobackend.RegisterPriority
	meta     *KernelMetadata
}

type registeredVJPImpl struct {
	name     string
	kernel   any
	priority gobackend.RegisterPriority
	meta     *KernelMetadata
}

type registeredSwiGLUImpl struct {
	name     string
	kernel   any
	priority gobackend.RegisterPriority
	meta     *KernelMetadata
}

var (
	registryMu        sync.RWMutex
	registry          = make(map[registrationKey]*registeredImpl)
	swigluRegistry    = make(map[dtypes.DType]*registeredSwiGLUImpl)
	vjpRegistry       = make(map[registrationKey]*registeredVJPImpl)
	swigluVJPRegistry = make(map[dtypes.DType]*registeredSwiGLUImpl)
)

// Register registers an in-place activation function for a specific activation type and data type.
// If an implementation is already registered with a higher priority, this call is ignored.
func Register[T gotype.Supported](name string, act compute.ActivationType, fn InPlaceFn[T], priority gobackend.RegisterPriority, meta *KernelMetadata) {
	kernel := func(in, out []T) {
		if len(in) > 0 && &in[0] != &out[0] {
			copy(out, in)
		}
		fn(out)
	}
	registerInternal[T](name, act, fn, kernel, priority, meta)
}

// RegisterKernel registers a kernel function (supporting both in-place and out-of-place).
func RegisterKernel[T gotype.Supported](name string, act compute.ActivationType, fn KernelFn[T], priority gobackend.RegisterPriority, meta *KernelMetadata) {
	inPlace := func(data []T) {
		fn(data, data)
	}
	registerInternal[T](name, act, inPlace, fn, priority, meta)
}

func registerInternal[T gotype.Supported](name string, act compute.ActivationType, inPlace InPlaceFn[T], kernel KernelFn[T], priority gobackend.RegisterPriority, meta *KernelMetadata) {
	if meta == nil {
		meta = &DefaultKernelMetadata
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.Lock()
	defer registryMu.Unlock()

	current, exists := registry[key]
	if exists && priority < current.priority {
		klog.V(2).Infof("Activation %s/%s (%q) ignored: priority %d < existing %d",
			act, dtype, name, priority, current.priority)
		return
	}
	registry[key] = &registeredImpl{
		name:     name,
		inPlace:  inPlace,
		kernel:   kernel,
		priority: priority,
		meta:     meta,
	}
	klog.V(2).Infof("Registered activation %s/%s: %q (priority %d)", act, dtype, name, priority)
}

// RegisterSwiGLU registers a SwiGLU kernel for a specific data type.
func RegisterSwiGLU[T gotype.Supported](name string, fn SwiGLUFn[T], priority gobackend.RegisterPriority, meta *KernelMetadata) {
	if meta == nil {
		meta = &DefaultKernelMetadata
	}
	dtype := dtypes.FromGenericsType[T]()
	registryMu.Lock()
	defer registryMu.Unlock()

	current, exists := swigluRegistry[dtype]
	if exists && priority < current.priority {
		return
	}
	swigluRegistry[dtype] = &registeredSwiGLUImpl{
		name:     name,
		kernel:   fn,
		priority: priority,
		meta:     meta,
	}
	klog.V(2).Infof("Registered SwiGLU/%s: %q (priority %d)", dtype, name, priority)
}

// Get returns the registered in-place activation function for the given activation type and data type T.
// Returns nil if activation is ActivationNone or if no implementation is registered.
func Get[T gotype.Supported](act compute.ActivationType) InPlaceFn[T] {
	fn, _ := GetWithMetadata[T](act)
	return fn
}

// GetWithMetadata returns the registered in-place activation function and its metadata.
// Returns (nil, nil) if activation is ActivationNone or if no implementation is registered.
func GetWithMetadata[T gotype.Supported](act compute.ActivationType) (InPlaceFn[T], *KernelMetadata) {
	if act == compute.ActivationNone {
		return nil, nil
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.RLock()
	impl, exists := registry[key]
	registryMu.RUnlock()

	if !exists || impl == nil {
		return nil, nil
	}
	return impl.inPlace.(InPlaceFn[T]), impl.meta
}

// GetKernel returns the registered kernel function for the given activation type and data type T.
func GetKernel[T gotype.Supported](act compute.ActivationType) KernelFn[T] {
	fn, _ := GetKernelWithMetadata[T](act)
	return fn
}

// GetKernelWithMetadata returns the registered kernel function and its metadata.
func GetKernelWithMetadata[T gotype.Supported](act compute.ActivationType) (KernelFn[T], *KernelMetadata) {
	if act == compute.ActivationNone {
		return nil, nil
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.RLock()
	impl, exists := registry[key]
	registryMu.RUnlock()

	if !exists || impl == nil {
		return nil, nil
	}
	return impl.kernel.(KernelFn[T]), impl.meta
}

// GetSwiGLU returns the registered SwiGLU function for data type T.
func GetSwiGLU[T gotype.Supported]() SwiGLUFn[T] {
	fn, _ := GetSwiGLUWithMetadata[T]()
	return fn
}

// GetSwiGLUWithMetadata returns the registered SwiGLU function and its metadata for data type T.
func GetSwiGLUWithMetadata[T gotype.Supported]() (SwiGLUFn[T], *KernelMetadata) {
	dtype := dtypes.FromGenericsType[T]()
	registryMu.RLock()
	impl, exists := swigluRegistry[dtype]
	registryMu.RUnlock()
	if !exists || impl == nil {
		return nil, nil
	}
	return impl.kernel.(SwiGLUFn[T]), impl.meta
}

// Apply applies the given activation function in-place across the slice.
// If the slice is large and workers are available, work is parallelized in chunks.
func Apply[T gotype.Supported](backend *gobackend.Backend, act compute.ActivationType, data []T) {
	Execute[T](backend, act, data, data)
}

// Execute applies the given activation function from in to out (in and out can be the same slice).
func Execute[T gotype.Supported](backend *gobackend.Backend, act compute.ActivationType, in, out []T) {
	if len(in) == 0 {
		return
	}
	if act == compute.ActivationNone {
		if &in[0] != &out[0] {
			copy(out, in)
		}
		return
	}
	fn, meta := GetKernelWithMetadata[T](act)
	if fn == nil {
		klog.Warningf("No registered activation implementation for act=%s dtype=%s",
			act, dtypes.FromGenericsType[T]())
		return
	}

	n := len(in)
	minChunk := meta.EffectiveMinParallelChunk()
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && n > minChunk {
		backend.Scheduler.ParallelizeChunks(n, minChunk, func(start, end int) {
			fn(in[start:end], out[start:end])
		})
	} else {
		fn(in, out)
	}
}

// ExecuteSwiGLU applies SwiGLU activation across numRows rows of width 2*hiddenDim to out of width hiddenDim.
func ExecuteSwiGLU[T gotype.Supported](backend *gobackend.Backend, in, out []T, numRows, hiddenDim int) {
	fn, meta := GetSwiGLUWithMetadata[T]()
	if fn == nil {
		klog.Warningf("No registered SwiGLU implementation for dtype=%s", dtypes.FromGenericsType[T]())
		return
	}

	totalWork := numRows * hiddenDim
	minChunk := meta.EffectiveMinParallelChunk()
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && totalWork > minChunk && numRows > 1 {
		minRows := max(minChunk/hiddenDim, 1)
		backend.Scheduler.ParallelizeChunks(numRows, minRows, func(rStart, rEnd int) {
			chunkRows := rEnd - rStart
			inOffset := rStart * 2 * hiddenDim
			outOffset := rStart * hiddenDim
			fn(in[inOffset:inOffset+chunkRows*2*hiddenDim], out[outOffset:outOffset+chunkRows*hiddenDim], chunkRows, hiddenDim)
		})
	} else {
		fn(in, out, numRows, hiddenDim)
	}
}

// RegisterVJPKernel registers a VJP kernel function for a specific activation type and data type.
func RegisterVJPKernel[T gotype.Supported](name string, act compute.ActivationType, fn VJPKernelFn[T], priority gobackend.RegisterPriority, meta *KernelMetadata) {
	if meta == nil {
		meta = &DefaultKernelMetadata
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.Lock()
	defer registryMu.Unlock()

	current, exists := vjpRegistry[key]
	if exists && priority < current.priority {
		return
	}
	vjpRegistry[key] = &registeredVJPImpl{
		name:     name,
		kernel:   fn,
		priority: priority,
		meta:     meta,
	}
	klog.V(2).Infof("Registered activation VJP %s/%s: %q (priority %d)", act, dtype, name, priority)
}

// RegisterSwiGLUVJP registers a SwiGLU VJP kernel for a specific data type.
func RegisterSwiGLUVJP[T gotype.Supported](name string, fn SwiGLUVJPFn[T], priority gobackend.RegisterPriority, meta *KernelMetadata) {
	if meta == nil {
		meta = &DefaultKernelMetadata
	}
	dtype := dtypes.FromGenericsType[T]()
	registryMu.Lock()
	defer registryMu.Unlock()

	current, exists := swigluVJPRegistry[dtype]
	if exists && priority < current.priority {
		return
	}
	swigluVJPRegistry[dtype] = &registeredSwiGLUImpl{
		name:     name,
		kernel:   fn,
		priority: priority,
		meta:     meta,
	}
	klog.V(2).Infof("Registered SwiGLU VJP/%s: %q (priority %d)", dtype, name, priority)
}

// GetVJPKernel returns the registered VJP kernel function for the given activation type and data type T.
func GetVJPKernel[T gotype.Supported](act compute.ActivationType) VJPKernelFn[T] {
	fn, _ := GetVJPKernelWithMetadata[T](act)
	return fn
}

// GetVJPKernelWithMetadata returns the registered VJP kernel function and its metadata.
func GetVJPKernelWithMetadata[T gotype.Supported](act compute.ActivationType) (VJPKernelFn[T], *KernelMetadata) {
	if act == compute.ActivationNone {
		return func(y, x, dOutput, dx []T) {
			copy(dx, dOutput)
		}, &DefaultKernelMetadata
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.RLock()
	impl, exists := vjpRegistry[key]
	registryMu.RUnlock()

	if !exists || impl == nil {
		return nil, nil
	}
	return impl.kernel.(VJPKernelFn[T]), impl.meta
}

// GetSwiGLUVJP returns the registered SwiGLU VJP function for data type T.
func GetSwiGLUVJP[T gotype.Supported]() SwiGLUVJPFn[T] {
	fn, _ := GetSwiGLUVJPWithMetadata[T]()
	return fn
}

// GetSwiGLUVJPWithMetadata returns the registered SwiGLU VJP function and its metadata for data type T.
func GetSwiGLUVJPWithMetadata[T gotype.Supported]() (SwiGLUVJPFn[T], *KernelMetadata) {
	dtype := dtypes.FromGenericsType[T]()
	registryMu.RLock()
	impl, exists := swigluVJPRegistry[dtype]
	registryMu.RUnlock()
	if !exists || impl == nil {
		return nil, nil
	}
	return impl.kernel.(SwiGLUVJPFn[T]), impl.meta
}

// ExecuteVJP applies the VJP of the given activation function:
//
//	dx = dOutput * f'(y or x)
//
// Note: For activations where act.VJPRequiresInput() == false (e.g. Relu, Sigmoid, HardSigmoid,
// LeakyRelu, Selu, Tanh, None), the parameter x is optional (can be nil or empty), and the calculation
// will use only y.
func ExecuteVJP[T gotype.Supported](backend *gobackend.Backend, act compute.ActivationType, y, x, dOutput, dx []T) {
	if len(dOutput) == 0 {
		return
	}
	if act == compute.ActivationNone {
		if len(dx) > 0 && len(dOutput) > 0 && &dOutput[0] != &dx[0] {
			copy(dx, dOutput)
		}
		return
	}
	fn, meta := GetVJPKernelWithMetadata[T](act)
	if fn == nil {
		klog.Warningf("No registered activation VJP implementation for act=%s dtype=%s",
			act, dtypes.FromGenericsType[T]())
		return
	}

	n := len(dOutput)
	minChunk := meta.EffectiveMinParallelChunk()
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && n > minChunk {
		backend.Scheduler.ParallelizeChunks(n, minChunk, func(start, end int) {
			var yChunk, xChunk []T
			if len(y) > 0 {
				yChunk = y[start:end]
			}
			if len(x) > 0 {
				xChunk = x[start:end]
			}
			fn(yChunk, xChunk, dOutput[start:end], dx[start:end])
		})
	} else {
		fn(y, x, dOutput, dx)
	}
}

// ExecuteVJPFromOutput applies the VJP of an activation that only depends on the forward output y:
//
//	dz = dOutput * f'(y)
//
// It requires that act.VJPRequiresInput() == false.
// dz and dOutput can be the same slice (in-place).
func ExecuteVJPFromOutput[T gotype.Supported](backend *gobackend.Backend, act compute.ActivationType, y, dOutput, dz []T) error {
	if act.VJPRequiresInput() {
		return errors.Errorf("ExecuteVJPFromOutput: activation %s requires input x (VJPRequiresInput() == true)", act)
	}
	if len(dOutput) == 0 {
		return nil
	}
	if act == compute.ActivationNone {
		if len(dz) > 0 && len(dOutput) > 0 && &dOutput[0] != &dz[0] {
			copy(dz, dOutput)
		}
		return nil
	}
	ExecuteVJP[T](backend, act, y, nil, dOutput, dz)
	return nil
}

// ExecuteSwiGLUVJP applies SwiGLU VJP across numRows rows.
func ExecuteSwiGLUVJP[T gotype.Supported](backend *gobackend.Backend, x, dOutput, dx []T, numRows, hiddenDim int) {
	fn, meta := GetSwiGLUVJPWithMetadata[T]()
	if fn == nil {
		klog.Warningf("No registered SwiGLU VJP implementation for dtype=%s", dtypes.FromGenericsType[T]())
		return
	}

	totalWork := numRows * hiddenDim
	minChunk := meta.EffectiveMinParallelChunk()
	if backend != nil && backend.Scheduler != nil && backend.Scheduler.IsEnabled() && totalWork > minChunk && numRows > 1 {
		minRows := max(minChunk/hiddenDim, 1)
		backend.Scheduler.ParallelizeChunks(numRows, minRows, func(rStart, rEnd int) {
			chunkRows := rEnd - rStart
			xOffset := rStart * 2 * hiddenDim
			dOutOffset := rStart * hiddenDim
			xChunk := x[xOffset : xOffset+chunkRows*2*hiddenDim]
			dOutChunk := dOutput[dOutOffset : dOutOffset+chunkRows*hiddenDim]
			dxChunk := dx[xOffset : xOffset+chunkRows*2*hiddenDim]
			fn(xChunk, dOutChunk, dxChunk, chunkRows, hiddenDim)
		})
	} else {
		fn(x, dOutput, dx, numRows, hiddenDim)
	}
}

// MakeBF16VJPKernelFromF32 wraps a float32 VJP kernel into a BFloat16 VJP kernel using chunked conversion.
func MakeBF16VJPKernelFromF32(f32Kernel VJPKernelFn[float32]) VJPKernelFn[bfloat16.BFloat16] {
	return func(y, x, dOutput, dx []bfloat16.BFloat16) {
		const halfChunk = 512
		var bufY, bufX, bufDOut, bufDx [halfChunk]float32
		n := len(dOutput)
		for i := 0; i < n; i += halfChunk {
			end := min(i+halfChunk, n)
			chunkLen := end - i
			var slY, slX []float32
			if len(y) > 0 {
				yChunk := y[i:end]
				for j, v := range yChunk {
					bufY[j] = v.Float32()
				}
				slY = bufY[:chunkLen]
			}
			if len(x) > 0 {
				xChunk := x[i:end]
				for j, v := range xChunk {
					bufX[j] = v.Float32()
				}
				slX = bufX[:chunkLen]
			}
			dOutChunk := dOutput[i:end]
			for j, v := range dOutChunk {
				bufDOut[j] = v.Float32()
			}
			slDOut := bufDOut[:chunkLen]
			slDx := bufDx[:chunkLen]
			f32Kernel(slY, slX, slDOut, slDx)
			dxChunk := dx[i:end]
			for j := range dxChunk {
				dxChunk[j] = bfloat16.FromFloat32(slDx[j])
			}
		}
	}
}

// MakeF16VJPKernelFromF32 wraps a float32 VJP kernel into a Float16 VJP kernel using chunked conversion.
func MakeF16VJPKernelFromF32(f32Kernel VJPKernelFn[float32]) VJPKernelFn[float16.Float16] {
	return func(y, x, dOutput, dx []float16.Float16) {
		const halfChunk = 512
		var bufY, bufX, bufDOut, bufDx [halfChunk]float32
		n := len(dOutput)
		for i := 0; i < n; i += halfChunk {
			end := min(i+halfChunk, n)
			chunkLen := end - i
			var slY, slX []float32
			if len(y) > 0 {
				yChunk := y[i:end]
				for j, v := range yChunk {
					bufY[j] = v.Float32()
				}
				slY = bufY[:chunkLen]
			}
			if len(x) > 0 {
				xChunk := x[i:end]
				for j, v := range xChunk {
					bufX[j] = v.Float32()
				}
				slX = bufX[:chunkLen]
			}
			dOutChunk := dOutput[i:end]
			for j, v := range dOutChunk {
				bufDOut[j] = v.Float32()
			}
			slDOut := bufDOut[:chunkLen]
			slDx := bufDx[:chunkLen]
			f32Kernel(slY, slX, slDOut, slDx)
			dxChunk := dx[i:end]
			for j := range dxChunk {
				dxChunk[j] = float16.FromFloat32(slDx[j])
			}
		}
	}
}

// MakeBF16SwiGLUVJPFromF32 wraps a float32 SwiGLU VJP kernel for BFloat16.
func MakeBF16SwiGLUVJPFromF32(f32SwiGLUVJP SwiGLUVJPFn[float32]) SwiGLUVJPFn[bfloat16.BFloat16] {
	return func(x, dOutput, dx []bfloat16.BFloat16, numRows, hiddenDim int) {
		xBuf := make([]float32, len(x))
		for i, v := range x {
			xBuf[i] = v.Float32()
		}
		dOutBuf := make([]float32, len(dOutput))
		for i, v := range dOutput {
			dOutBuf[i] = v.Float32()
		}
		dxBuf := make([]float32, len(dx))
		f32SwiGLUVJP(xBuf, dOutBuf, dxBuf, numRows, hiddenDim)
		for i := range dx {
			dx[i] = bfloat16.FromFloat32(dxBuf[i])
		}
	}
}

// MakeF16SwiGLUVJPFromF32 wraps a float32 SwiGLU VJP kernel for Float16.
func MakeF16SwiGLUVJPFromF32(f32SwiGLUVJP SwiGLUVJPFn[float32]) SwiGLUVJPFn[float16.Float16] {
	return func(x, dOutput, dx []float16.Float16, numRows, hiddenDim int) {
		xBuf := make([]float32, len(x))
		for i, v := range x {
			xBuf[i] = v.Float32()
		}
		dOutBuf := make([]float32, len(dOutput))
		for i, v := range dOutput {
			dOutBuf[i] = v.Float32()
		}
		dxBuf := make([]float32, len(dx))
		f32SwiGLUVJP(xBuf, dOutBuf, dxBuf, numRows, hiddenDim)
		for i := range dx {
			dx[i] = float16.FromFloat32(dxBuf[i])
		}
	}
}

// Ensure standard dtypes are referenced to satisfy compiler.
var (
	_ = float16.FromFloat32
	_ = bfloat16.FromFloat32
)

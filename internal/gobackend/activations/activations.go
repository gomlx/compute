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

type registrationKey struct {
	act   compute.ActivationType
	dtype dtypes.DType
}

type registeredImpl struct {
	name     string
	inPlace  any
	kernel   any
	priority gobackend.RegisterPriority
}

type registeredVJPImpl struct {
	name     string
	kernel   any
	priority gobackend.RegisterPriority
}

var (
	registryMu          sync.RWMutex
	registry            = make(map[registrationKey]*registeredImpl)
	swigluRegistry      = make(map[dtypes.DType]any)
	swigluPriorities    = make(map[dtypes.DType]gobackend.RegisterPriority)
	vjpRegistry         = make(map[registrationKey]*registeredVJPImpl)
	swigluVJPRegistry   = make(map[dtypes.DType]any)
	swigluVJPPriorities = make(map[dtypes.DType]gobackend.RegisterPriority)
)

// Register registers an in-place activation function for a specific activation type and data type.
// If an implementation is already registered with a higher priority, this call is ignored.
func Register[T gotype.Supported](name string, act compute.ActivationType, fn InPlaceFn[T], priority gobackend.RegisterPriority) {
	kernel := func(in, out []T) {
		if len(in) > 0 && &in[0] != &out[0] {
			copy(out, in)
		}
		fn(out)
	}
	registerInternal[T](name, act, fn, kernel, priority)
}

// RegisterKernel registers a kernel function (supporting both in-place and out-of-place).
func RegisterKernel[T gotype.Supported](name string, act compute.ActivationType, fn KernelFn[T], priority gobackend.RegisterPriority) {
	inPlace := func(data []T) {
		fn(data, data)
	}
	registerInternal[T](name, act, inPlace, fn, priority)
}

func registerInternal[T gotype.Supported](name string, act compute.ActivationType, inPlace InPlaceFn[T], kernel KernelFn[T], priority gobackend.RegisterPriority) {
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
	}
	klog.V(2).Infof("Registered activation %s/%s: %q (priority %d)", act, dtype, name, priority)
}

// RegisterSwiGLU registers a SwiGLU kernel for a specific data type.
func RegisterSwiGLU[T gotype.Supported](name string, fn SwiGLUFn[T], priority gobackend.RegisterPriority) {
	dtype := dtypes.FromGenericsType[T]()
	registryMu.Lock()
	defer registryMu.Unlock()

	currPriority, exists := swigluPriorities[dtype]
	if exists && priority < currPriority {
		return
	}
	swigluRegistry[dtype] = fn
	swigluPriorities[dtype] = priority
	klog.V(2).Infof("Registered SwiGLU/%s: %q (priority %d)", dtype, name, priority)
}

// Get returns the registered in-place activation function for the given activation type and data type T.
// Returns nil if activation is ActivationNone or if no implementation is registered.
func Get[T gotype.Supported](act compute.ActivationType) InPlaceFn[T] {
	if act == compute.ActivationNone {
		return nil
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.RLock()
	impl, exists := registry[key]
	registryMu.RUnlock()

	if !exists || impl == nil {
		return nil
	}
	return impl.inPlace.(InPlaceFn[T])
}

// GetKernel returns the registered kernel function for the given activation type and data type T.
func GetKernel[T gotype.Supported](act compute.ActivationType) KernelFn[T] {
	if act == compute.ActivationNone {
		return nil
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.RLock()
	impl, exists := registry[key]
	registryMu.RUnlock()

	if !exists || impl == nil {
		return nil
	}
	return impl.kernel.(KernelFn[T])
}

// GetSwiGLU returns the registered SwiGLU function for data type T.
func GetSwiGLU[T gotype.Supported]() SwiGLUFn[T] {
	dtype := dtypes.FromGenericsType[T]()
	registryMu.RLock()
	fn, exists := swigluRegistry[dtype]
	registryMu.RUnlock()
	if !exists || fn == nil {
		return nil
	}
	return fn.(SwiGLUFn[T])
}

const minParallelizeChunk = 4096

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
	fn := GetKernel[T](act)
	if fn == nil {
		klog.Warningf("No registered activation implementation for act=%s dtype=%s",
			act, dtypes.FromGenericsType[T]())
		return
	}

	n := len(in)
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && n > minParallelizeChunk {
		var wg sync.WaitGroup
		for i := 0; i < n; i += minParallelizeChunk {
			end := min(i+minParallelizeChunk, n)
			inChunk := in[i:end]
			outChunk := out[i:end]
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				fn(inChunk, outChunk)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		fn(in, out)
	}
}

// ExecuteSwiGLU applies SwiGLU activation across numRows rows of width 2*hiddenDim to out of width hiddenDim.
func ExecuteSwiGLU[T gotype.Supported](backend *gobackend.Backend, in, out []T, numRows, hiddenDim int) {
	fn := GetSwiGLU[T]()
	if fn == nil {
		klog.Warningf("No registered SwiGLU implementation for dtype=%s", dtypes.FromGenericsType[T]())
		return
	}

	totalWork := numRows * hiddenDim
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && totalWork > minParallelizeChunk && numRows > 1 {
		rowsPerChunk := max(minParallelizeChunk/hiddenDim, 1)
		var wg sync.WaitGroup
		for r := 0; r < numRows; r += rowsPerChunk {
			rEnd := min(r+rowsPerChunk, numRows)
			chunkRows := rEnd - r
			inOffset := r * 2 * hiddenDim
			outOffset := r * hiddenDim
			inChunk := in[inOffset : inOffset+chunkRows*2*hiddenDim]
			outChunk := out[outOffset : outOffset+chunkRows*hiddenDim]
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				fn(inChunk, outChunk, chunkRows, hiddenDim)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		fn(in, out, numRows, hiddenDim)
	}
}

// RegisterVJPKernel registers a VJP kernel function for a specific activation type and data type.
func RegisterVJPKernel[T gotype.Supported](name string, act compute.ActivationType, fn VJPKernelFn[T], priority gobackend.RegisterPriority) {
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
	}
	klog.V(2).Infof("Registered activation VJP %s/%s: %q (priority %d)", act, dtype, name, priority)
}

// RegisterSwiGLUVJP registers a SwiGLU VJP kernel for a specific data type.
func RegisterSwiGLUVJP[T gotype.Supported](name string, fn SwiGLUVJPFn[T], priority gobackend.RegisterPriority) {
	dtype := dtypes.FromGenericsType[T]()
	registryMu.Lock()
	defer registryMu.Unlock()

	currPriority, exists := swigluVJPPriorities[dtype]
	if exists && priority < currPriority {
		return
	}
	swigluVJPRegistry[dtype] = fn
	swigluVJPPriorities[dtype] = priority
	klog.V(2).Infof("Registered SwiGLU VJP/%s: %q (priority %d)", dtype, name, priority)
}

// GetVJPKernel returns the registered VJP kernel function for the given activation type and data type T.
func GetVJPKernel[T gotype.Supported](act compute.ActivationType) VJPKernelFn[T] {
	if act == compute.ActivationNone {
		return func(y, x, dOutput, dx []T) {
			copy(dx, dOutput)
		}
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.RLock()
	impl, exists := vjpRegistry[key]
	registryMu.RUnlock()

	if !exists || impl == nil {
		return nil
	}
	return impl.kernel.(VJPKernelFn[T])
}

// GetSwiGLUVJP returns the registered SwiGLU VJP function for data type T.
func GetSwiGLUVJP[T gotype.Supported]() SwiGLUVJPFn[T] {
	dtype := dtypes.FromGenericsType[T]()
	registryMu.RLock()
	fn, exists := swigluVJPRegistry[dtype]
	registryMu.RUnlock()
	if !exists || fn == nil {
		return nil
	}
	return fn.(SwiGLUVJPFn[T])
}

// ExecuteVJP applies the VJP of the given activation function.
// y and/or x may be empty/nil depending on whether the activation VJP requires x.
func ExecuteVJP[T gotype.Supported](backend *gobackend.Backend, act compute.ActivationType, y, x, dOutput, dx []T) {
	if len(dOutput) == 0 {
		return
	}
	if act == compute.ActivationNone {
		copy(dx, dOutput)
		return
	}
	fn := GetVJPKernel[T](act)
	if fn == nil {
		klog.Warningf("No registered activation VJP implementation for act=%s dtype=%s",
			act, dtypes.FromGenericsType[T]())
		return
	}

	n := len(dOutput)
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && n > minParallelizeChunk {
		var wg sync.WaitGroup
		for i := 0; i < n; i += minParallelizeChunk {
			end := min(i+minParallelizeChunk, n)
			var yChunk, xChunk []T
			if len(y) > 0 {
				yChunk = y[i:end]
			}
			if len(x) > 0 {
				xChunk = x[i:end]
			}
			dOutChunk := dOutput[i:end]
			dxChunk := dx[i:end]
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				fn(yChunk, xChunk, dOutChunk, dxChunk)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		fn(y, x, dOutput, dx)
	}
}

// ExecuteSwiGLUVJP applies SwiGLU VJP across numRows rows.
func ExecuteSwiGLUVJP[T gotype.Supported](backend *gobackend.Backend, x, dOutput, dx []T, numRows, hiddenDim int) {
	fn := GetSwiGLUVJP[T]()
	if fn == nil {
		klog.Warningf("No registered SwiGLU VJP implementation for dtype=%s", dtypes.FromGenericsType[T]())
		return
	}

	totalWork := numRows * hiddenDim
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && totalWork > minParallelizeChunk && numRows > 1 {
		rowsPerChunk := max(minParallelizeChunk/hiddenDim, 1)
		var wg sync.WaitGroup
		for r := 0; r < numRows; r += rowsPerChunk {
			rEnd := min(r+rowsPerChunk, numRows)
			chunkRows := rEnd - r
			xOffset := r * 2 * hiddenDim
			dOutOffset := r * hiddenDim
			xChunk := x[xOffset : xOffset+chunkRows*2*hiddenDim]
			dOutChunk := dOutput[dOutOffset : dOutOffset+chunkRows*hiddenDim]
			dxChunk := dx[xOffset : xOffset+chunkRows*2*hiddenDim]
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				fn(xChunk, dOutChunk, dxChunk, chunkRows, hiddenDim)
				wg.Done()
			})
		}
		wg.Wait()
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

